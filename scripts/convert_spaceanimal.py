#!/usr/bin/env python
"""Convert the SpaceAnimal COCO keypoint exports to tailcycle-dataset format.

The source annotations are sampled COCO frames while ``data/<clip>`` contains the complete
image sequence.  Each source clip becomes one annotated 2-D session; unannotated context frames
remain in the group and are represented by absent label rows.  Pixels are symlinked individually
under the canonical ``%06d.jpg`` names required by the tailcycle format.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/spaceanimal')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
           'tailcycle-datasets/spaceanimal')

DATASETS = {
    'spaceanimal-celegans': {
        'source': 'C. elegans', 'prefix': 'worm', 'animal_key': 'worm_id',
    },
    'spaceanimal-fly': {
        'source': 'Drosophila', 'prefix': 'drosophila', 'animal_key': 'fly_id',
    },
    'spaceanimal-zebrafish': {
        'source': 'Zebrafish', 'prefix': 'zebrafish', 'animal_key': 'fish_id',
    },
}


def _natural_key(path: Path) -> tuple[int, str]:
    """Sort source image names by their trailing frame number."""
    m = re.search(r'_(\d+)$', path.stem)
    return (int(m.group(1)), path.name) if m else (10**12, path.name)


def _skeleton(category: dict, names: list[str]) -> list[tuple[str, str]]:
    """Build a connected, anatomical skeleton for each SpaceAnimal keypoint axis.

    The fly export omits all six leg attachments and connects the eyes to ``mouth``. Anchor
    its limbs at ``back`` instead; other species use the graph declared in their COCO category.
    """
    if 'leftleg1_1' in names:
        edges = [('mouth', 'head'), ('head', 'back'), ('back', 'tail'),
                 ('head', 'lefteye'), ('head', 'righteye'),
                 ('back', 'leftwing'), ('back', 'rightwing')]
        for leg in range(1, 4):
            for side in ('left', 'right'):
                edges.extend([
                    ('back', f'{side}leg{leg}_1'),
                    (f'{side}leg{leg}_1', f'{side}leg{leg}_2'),
                    (f'{side}leg{leg}_2', f'{side}leg{leg}_3'),
                ])
        return edges

    edges = category.get('skeleton', [])
    if not edges:
        return []
    base = 1 if min(min(edge) for edge in edges) >= 1 else 0
    return [(names[a - base], names[b - base]) for a, b in edges]


def _flip_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Declare bilateral pairs from SpaceAnimal's explicit left/right keypoint names."""
    pairs = []
    for name in names:
        if name.startswith('left'):
            other = 'right' + name[len('left'):]
        elif name.startswith('Left'):
            other = 'Right' + name[len('Left'):]
        else:
            continue
        if other in names and (other, name) not in pairs:
            pairs.append((name, other))
    return pairs


def _load_annotations(cfg: dict, split: str) -> tuple[dict, dict, list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    """Return image metadata, grouped annotations, keypoint names, and skeleton."""
    path = SRC / cfg['source'] / 'annotations' / f"{cfg['prefix']}_{split}.json"
    doc = json.loads(path.read_text())
    categories = doc.get('categories', [])
    if len(categories) != 1:
        raise RuntimeError(f'{path}: expected exactly one category, got {len(categories)}')
    category = categories[0]
    names = list(category['keypoints'])
    if not names or len(set(names)) != len(names):
        raise RuntimeError(f'{path}: invalid keypoint names')

    images = {int(row['id']): row for row in doc['images']}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for ann in doc['annotations']:
        image = images.get(int(ann['image_id']))
        if image is None:
            raise RuntimeError(f'{path}: annotation references unknown image {ann["image_id"]}')
        group = Path(image['file_name']).parent.name
        by_group[group].append(ann)
    return images, by_group, names, _skeleton(category, names), _flip_pairs(names)


def _nominal_rig(size: tuple[int, int]) -> fmt.Rig:
    """Build the required one-camera nominal calibration for this 2-D source."""
    from aniposelib.cameras import CameraGroup

    camera = fmt.nominal_camera('cam0', size)
    return fmt.Rig(CameraGroup([camera]), offset={'cam0': (0.0, 0.0)},
                   moving={'cam0': False}, calibrated={'cam0': False})


def _frame_files(data_dir: Path) -> list[Path]:
    """All source frames in temporal order, rejecting non-JPEG or duplicate indices."""
    files = sorted((p for p in data_dir.iterdir() if p.suffix.lower() == '.jpg'),
                   key=_natural_key)
    if not files:
        raise RuntimeError(f'{data_dir}: no .jpg frames')
    nums = [_natural_key(p)[0] for p in files]
    if len(set(nums)) != len(nums):
        raise RuntimeError(f'{data_dir}: duplicate frame numbers')
    return files


def _image_size(path: Path) -> tuple[int, int]:
    """Read an image's pixel width and height."""
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _write_clip(out: Path, source_root: Path, split: str, group_name: str,
                images: dict[int, dict], annotations: list[dict], names: list[str],
                skeleton: list[tuple[str, str]], flip_pairs: list[tuple[str, str]],
                animal_key: str, ann_path: Path) -> None:
    """Write one source clip as one tailcycle session.

    COCO visibility 1 is a positioned but not-visible assessment and maps to ``missing``;
    visibility 2 retains its coordinates as ``visible``. Source clips have fixed resolution, so
    only one JPEG is inspected while every annotated frame's JSON dimensions are checked.
    Pixels are linked by absolute path to preserve JPEG bytes without per-frame realpath lookups.
    """
    data_dir = source_root / 'data' / group_name
    files = _frame_files(data_dir)
    frame_index = {p.name: i for i, p in enumerate(files)}
    ann_by_frame: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        image = images[int(ann['image_id'])]
        name = Path(image['file_name']).name
        if name not in frame_index:
            raise RuntimeError(f'{ann_path}: {name} is missing from {data_dir}')
        ann_by_frame[frame_index[name]].append(ann)

    animal_values = sorted({int(ann[animal_key]) for ann in annotations})
    if not animal_values:
        raise RuntimeError(f'{ann_path}: clip {group_name} has no annotations')
    animal_ids = [f'a{value:02d}' for value in animal_values]
    animal_index = {value: i for i, value in enumerate(animal_values)}
    T, K = len(files), len(names)
    labels = fmt.empty_labels(len(animal_ids), T, K, 1, mode3d=False,
                              animal_ids=animal_ids)
    labels.boxes = np.full((len(animal_ids), T, 1, 4), np.nan, np.float32)
    labels.instance = np.full((len(animal_ids), T, 1), fmt.INST_NONE, np.int8)

    for frame, anns in ann_by_frame.items():
        for ann in anns:
            animal = int(ann[animal_key])
            a = animal_index[animal]
            if labels.instance[a, frame, 0] != fmt.INST_NONE:
                raise RuntimeError(f'{ann_path}: duplicate {animal_key}={animal} at frame {frame}')
            x0, y0, width, height = (float(v) for v in ann['bbox'])
            labels.boxes[a, frame, 0] = (x0, y0, x0 + width, y0 + height)
            labels.instance[a, frame, 0] = fmt.INST_LABELED
            keypoints = ann['keypoints']
            if len(keypoints) != 3 * K:
                raise RuntimeError(f'{ann_path}: annotation {ann["id"]} has wrong keypoint length')
            for k in range(K):
                x, y, visibility = keypoints[3 * k:3 * k + 3]
                if int(visibility) == 2:
                    labels.points2d[a, frame, k, 0] = (float(x), float(y))
                    labels.vis2d[a, frame, k, 0] = fmt.VISIBLE
                elif int(visibility) == 1:
                    labels.vis2d[a, frame, k, 0] = fmt.MISSING
                elif int(visibility) != 0:
                    raise RuntimeError(f'{ann_path}: unsupported COCO visibility {visibility!r}')

    size = _image_size(files[0])
    metadata_sizes = {(int(row['width']), int(row['height'])) for row in images.values()
                      if Path(row['file_name']).parent.name == group_name}
    if metadata_sizes and metadata_sizes != {size}:
        raise RuntimeError(f'{data_dir}: JSON/image dimensions disagree: {metadata_sizes} vs {size}')

    dst = out / split / group_name
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f'{dst}: already exists; use --clean to replace the output root')
    pixel_dir = dst / 'groups' / 'clip' / 'cam0'
    pixel_dir.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(files):
        fmt.link(pixel_dir / f'{i:06d}.jpg', source)

    group = fmt.Group('clip', T, fps=float('nan'), source_video=str(data_dir),
                      source_frame_start=0, source_frame_step=1,
                      notes='COCO annotations are sparse sampled frames; all source frames retained')
    rig = _nominal_rig(size)
    fmt.write_session(
        dst, mode='2d', units='px', label_source='annotated', names=names, rig=rig,
        groups={'clip': group}, labels={'clip': labels}, skeleton=skeleton,
        flip_pairs=flip_pairs,
        provenance={
            'source': str(source_root), 'annotation_file': str(ann_path),
            'annotator': '', 'annotator_tool': 'SpaceAnimal COCO keypoint annotations',
            'converter': 'scripts/convert_spaceanimal.py',
        })
    print(f'{split}/{group_name}: {T} frames, {len(annotations)} annotations, '
          f'{len(animal_ids)} animals, {size[0]}x{size[1]}')


def convert(name: str, clean: bool = False, workers: int = 32) -> None:
    """Convert one named SpaceAnimal dataset, including train and val, in parallel.

    Clips are independent and the hot path is filesystem metadata and symlink creation, so
    threads overlap I/O without copying large image arrays into worker processes.
    """
    cfg = DATASETS[name]
    source_root = SRC / cfg['source']
    out = OUT / name
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ('train', 'val'):
        ann_path = source_root / 'annotations' / f"{cfg['prefix']}_{split}.json"
        images, by_group, names, skeleton, flip_pairs = _load_annotations(cfg, split)
        for group_name in sorted(by_group):
            jobs.append((split, group_name, images, by_group[group_name], names, skeleton,
                         flip_pairs, ann_path))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_write_clip, out, source_root, split, group_name, images, anns,
                               names, skeleton, flip_pairs, cfg['animal_key'], ann_path)
                   for split, group_name, images, anns, names, skeleton, flip_pairs, ann_path in jobs]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    """Parse CLI options and convert selected SpaceAnimal datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), action='append',
                        help='dataset to convert; repeatable (default: all)')
    parser.add_argument('--clean', action='store_true', help='remove selected output roots first')
    parser.add_argument('--workers', type=int, default=min(32, (os.cpu_count() or 1) * 2),
                        help='parallel clip writers (default: 2x CPUs, capped at 32)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    for name in args.dataset or sorted(DATASETS):
        convert(name, clean=args.clean, workers=args.workers)


if __name__ == '__main__':
    main()
