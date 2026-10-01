#!/usr/bin/env python
"""Convert CSDLLab RP-5.7K LabelMe stills into the tailcycle-dataset format.

    pixi run python scripts/convert_rp5k.py

The split is a reproducible, global random sample of 100 images for val; all remaining images
are train. Pixel files are symlinked from the downloaded source root.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/rat-rp-5.7k')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
           'tailcycle-datasets/rat-rp-5.7k')
SUBSETS = ('concret', 'grass', 'ground', 'kegong_pegion', 'kegong_uav', 'kegong_webcam')
CAMERA = 'cam0'
NAMES = [
    'right_hindlimb', 'left_hindlimb', 'right_forelimb', 'left_forelimb',
    'tail_root', 'head', 'neck', 'spine_midpoint', 'tail_midpoint', 'tail_endpoint',
]
LABEL_TO_NAME = {str(i): NAMES[i - 1] for i in range(1, len(NAMES) + 1)}
FLIP_PAIRS = [('right_hindlimb', 'left_hindlimb'),
              ('right_forelimb', 'left_forelimb')]
LABEL_RE = re.compile(r'^(10|[1-9])(?:_\d+)?$')


def read_record(subset: str, annotation: Path, image_dir: Path) -> dict:
    """Parse one LabelMe still and check its 10 named points and one animal box.

    If `imagePath` does not match the archive filename, the JSON stem is used; absent point
    shapes remain unassessed and are reported on the output group.
    """
    with annotation.open(encoding='utf-8') as f:
        doc = json.load(f)
    image_name = Path(doc.get('imagePath', '')).name
    image = image_dir / image_name
    if not image.is_file():
        image = image_dir / f'{annotation.stem}.jpg'
    if not image.is_file() or image.stem != annotation.stem:
        raise ValueError(f'{annotation}: imagePath {image_name!r} and JSON stem do not '
                         'resolve to a matching source image')

    points = {}
    boxes = []
    for shape in doc.get('shapes', []):
        kind = shape.get('shape_type')
        if kind == 'point':
            label = str(shape.get('label', ''))
            match = LABEL_RE.fullmatch(label)
            if match is None:
                raise ValueError(f'{annotation}: unknown point label {label!r}')
            name = LABEL_TO_NAME[match.group(1)]
            xy = shape.get('points', [])
            if len(xy) != 1 or len(xy[0]) != 2:
                raise ValueError(f'{annotation}: malformed point {label!r}')
            if name in points:
                raise ValueError(f'{annotation}: duplicate point {label!r}')
            points[name] = tuple(float(v) for v in xy[0])
        elif kind == 'rectangle':
            xy = shape.get('points', [])
            if len(xy) != 2 or any(len(v) != 2 for v in xy):
                raise ValueError(f'{annotation}: malformed rectangle')
            (ax, ay), (bx, by) = xy
            boxes.append((min(float(ax), float(bx)), min(float(ay), float(by)),
                          max(float(ax), float(bx)), max(float(ay), float(by))))
        else:
            raise ValueError(f'{annotation}: unexpected shape type {kind!r}')
    if len(boxes) > 1:
        raise ValueError(f'{annotation}: expected at most one animal box, got {len(boxes)}')
    box = boxes[0] if boxes else None
    if box is not None:
        x0, y0, x1, y1 = box
        if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
            raise ValueError(f'{annotation}: invalid/non-empty box {box}')
    with Image.open(image) as im:
        size = tuple(int(v) for v in im.size)
    for name, xy in points.items():
        if not np.isfinite(xy).all():
            raise ValueError(f'{annotation}: non-finite {name} point {xy}')
    return {
        'subset': subset,
        'stem': annotation.stem,
        'image': image.resolve(),
        'source_name': image_name,
        'size': size,
        'points': points,
        'missing_points': sorted(set(NAMES) - set(points)),
        'box': box,
    }


def collect_records(source: Path) -> list[dict]:
    """Read the six source subsets in stable order; refuse missing or mismatched inputs."""
    records = []
    for subset in SUBSETS:
        ann_dir, image_dir = source / 'annotations' / subset, source / 'imgs' / subset
        if not ann_dir.is_dir() or not image_dir.is_dir():
            raise SystemExit(f'missing RP-5.7K subset: {ann_dir} or {image_dir}')
        annotations = sorted(ann_dir.glob('*.json'))
        images = sorted(image_dir.glob('*.jpg'))
        if len(annotations) != len(images):
            raise SystemExit(f'{subset}: {len(annotations)} annotations but {len(images)} JPGs')
        subset_records = [read_record(subset, path, image_dir) for path in annotations]
        sizes = {r['size'] for r in subset_records}
        if len(sizes) != 1:
            raise SystemExit(f'{subset}: source images do not share one image size: '
                             f'{sorted(sizes)}')
        records.extend(subset_records)
        print(f'{subset}: {len(annotations)} paired annotation/image files')
    if len({(r['subset'], r['stem']) for r in records}) != len(records):
        raise SystemExit('duplicate source image stems within a subset')
    return records


def main() -> None:
    """Convert the source dataset and validate the generated root.

    The stable input order and seed define the frame-level split. Each still has one annotated
    rat; the only boxless source still uses its positioned-keypoint extent as a `labeled` box.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, default=SRC)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--val-frames', type=int, default=100)
    parser.add_argument('--seed', type=int, default=5700)
    args = parser.parse_args()

    if args.val_frames < 1:
        raise SystemExit('--val-frames must be positive')
    if args.out.exists():
        raise SystemExit(f'{args.out} already exists; move/remove it before converting')
    records = collect_records(args.src)
    if args.val_frames >= len(records):
        raise SystemExit(f'need fewer validation frames than the {len(records)} source images')

    val_indices = set(random.Random(args.seed).sample(range(len(records)), args.val_frames))
    for i, record in enumerate(records):
        record['split'] = 'val' if i in val_indices else 'train'

    from aniposelib.cameras import CameraGroup

    partitions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        partitions[(record['split'], record['subset'])].append(record)

    args.out.mkdir(parents=True)
    manifest = args.out / 'split_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=('split', 'subset', 'group_id', 'image'))
        writer.writeheader()
        for record in records:
            writer.writerow({'split': record['split'], 'subset': record['subset'],
                             'group_id': f"{record['subset']}_{record['stem']}",
                             'image': f"{record['subset']}/{record['source_name']}"})

    total = defaultdict(int)
    for split in ('train', 'val'):
        for subset in SUBSETS:
            part = partitions[(split, subset)]
            if not part:
                continue
            session_id = f'{subset}__{split}'
            session_path = args.out / split / session_id
            camera = fmt.nominal_camera(CAMERA, part[0]['size'])
            rig = fmt.Rig(CameraGroup([camera]), offset={CAMERA: (0.0, 0.0)},
                          moving={CAMERA: False}, calibrated={CAMERA: False})
            groups, labels = {}, {}
            for record in part:
                gid = f"{subset}_{record['stem']}"
                note = f"source_image={subset}/{record['source_name']}"
                if record['missing_points']:
                    note += "; point_shape_absent_unassessed=" + ','.join(record['missing_points'])
                box = record['box']
                if box is None:
                    pts = np.asarray(list(record['points'].values()), np.float64)
                    w, h = part[0]['size']
                    box = (max(0.0, pts[:, 0].min()), max(0.0, pts[:, 1].min()),
                           min(float(w), pts[:, 0].max()), min(float(h), pts[:, 1].max()))
                    if not (box[2] > box[0] and box[3] > box[1]):
                        raise SystemExit(f'{gid}: boxless still has an empty keypoint extent')
                    note += '; box_from_keypoint_extent'
                groups[gid] = fmt.Group(group_id=gid, n_frames=1, notes=note)
                lab = fmt.empty_labels(1, 1, len(NAMES), 1, mode3d=False,
                                       animal_ids=['rat_0'])
                for k, name in enumerate(NAMES):
                    if name not in record['points']:
                        continue
                    lab.points2d[0, 0, k, 0] = record['points'][name]
                    lab.vis2d[0, 0, k, 0] = fmt.VISIBLE
                lab.boxes = np.full((1, 1, 1, 4), np.nan, np.float32)
                lab.boxes[0, 0, 0] = box
                lab.instance = np.full((1, 1, 1), fmt.INST_LABELED, np.int8)
                labels[gid] = lab

                pixel_dir = session_path / 'groups' / gid / CAMERA
                pixel_dir.mkdir(parents=True, exist_ok=True)
                fmt.link(pixel_dir / '000000.jpg', record['image'])

            fmt.write_session(
                session_path, mode='2d', units='px', label_source='annotated', names=NAMES,
                rig=rig, groups=groups, labels=labels, flip_pairs=FLIP_PAIRS,
                provenance={
                    'source': 'CSDLLab/RP-5.7K (Google Drive release)',
                    'annotator_tool': 'LabelMe 5.1.1',
                    'converter': 'scripts/convert_rp5k.py',
                    'source_subset': subset,
                    'split': split,
                    'split_method': f'global random frame split; seed={args.seed}; '
                                    f'validation_frames={args.val_frames}',
                    'visibility_note': 'keypoints encoded as visible per dataset owner direction; '
                                       'absent point shapes are unassessed and remain no-row holes',
                    'instance_note': 'one labeled rat per still; the one boxless partial pose uses '
                                     'its positioned-keypoint extent',
                })
            total[split] += len(part)
            print(f'{split}/{session_id}: {len(part)} still groups')

    print(f"\nSplit totals: train={total['train']}, val={total['val']}, "
          f"all={len(records)} (seed={args.seed})")
    print(f'Split manifest: {manifest}')
    errors = fmt.validate_dataset(fmt.load_dataset(args.out), check_images=True)
    for error in errors:
        print(('WARN ' if 'WARNING' in error else 'FAIL ') + error)
    failures = [e for e in errors if 'WARNING' not in e]
    print(f'Validation: {len(failures)} errors, {len(errors) - len(failures)} warnings')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
