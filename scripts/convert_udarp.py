#!/usr/bin/env python
"""Convert UDARP-9.4K still-image annotations into the tailcycle-dataset format.

The converter creates one one-frame group per source image, samples exactly 100 images
uniformly from the combined Lawn + Platform image pool for val, and places all remaining
images in train. The split is deterministic for a given seed.

    pixi run python scripts/convert_udarp.py --validate
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/rat-udarp-9.4k/UDARP-9.4K')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
           'tailcycle-datasets/rat-udarp-9.4k')
SEED = 9400
VAL_FRAMES = 100
CAMERA = 'cam0'

COLLECTIONS = {
    'lawn': {
        'directory': 'Lawn scenarios',
        'csv': 'Lawn_scenarios.csv',
        'names': ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head', 'neck', 'spine',
                  'tail_middle', 'tail_end'],
    },
    'platform': {
        'directory': 'Platform scenarios',
        'csv': 'Platform_scenarios.csv',
        'names': ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head'],
    },
}
FLIP_PAIRS = [['rRP', 'lRP'], ['rFP', 'lFP']]


def read_records(src: Path) -> list[dict]:
    """Read all annotation CSV rows, validate their stills, and return a sorted image pool."""
    records = []
    for collection, spec in COLLECTIONS.items():
        csv_path = src / spec['directory'] / spec['csv']
        if not csv_path.is_file():
            raise SystemExit(f'missing annotation table: {csv_path}')
        with csv_path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            expected = {'imageName', 'Rat_num', 'BBox', *spec['names']}
            if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
                raise SystemExit(f'{csv_path}: expected columns {sorted(expected)}; '
                                 f'got {reader.fieldnames}')
            for row_number, row in enumerate(reader, start=2):
                rel = (Path(spec['directory']) / row['imageName'].lstrip('/')).as_posix()
                image = src / rel
                if not image.is_file():
                    raise SystemExit(f'{csv_path}:{row_number}: image does not exist: {image}')
                with Image.open(image) as im:
                    width, height = im.size
                points = {}
                invalid_points = []
                zero_points = []
                for name in spec['names']:
                    value = (row.get(name) or '').strip()
                    parts = value.split('_')
                    if len(parts) != 3:
                        raise SystemExit(f'{csv_path}:{row_number}: malformed {name}={value!r}')
                    try:
                        x, y = float(parts[0]), float(parts[1])
                    except ValueError as exc:
                        raise SystemExit(f'{csv_path}:{row_number}: bad coordinates '
                                         f'{name}={value!r}') from exc
                    if not np.isfinite([x, y]).all():
                        raise SystemExit(f'{csv_path}:{row_number}: non-finite coordinates '
                                         f'{name}={value!r}')
                    if x == 0 and y == 0:
                        # The source's unlabelled-point sentinel; it lies outside the BBox.
                        zero_points.append(name)
                        continue
                    if not (0 <= x < width and 0 <= y < height):
                        invalid_points.append((name, value))
                        continue
                    points[name] = (x, y)
                if not points:
                    raise SystemExit(f'{csv_path}:{row_number}: no positioned keypoints')
                box, box_note = instance_box(row['BBox'], points, width, height,
                                             f'{csv_path}:{row_number}')
                records.append({
                    'collection': collection,
                    'relative_image': rel,
                    'image': image,
                    'width': width,
                    'height': height,
                    'animal_id': f"rat{row['Rat_num']}",
                    'points': points,
                    'invalid_points': invalid_points,
                    'zero_points': zero_points,
                    'box': box,
                    'box_note': box_note,
                })
    records.sort(key=lambda r: r['relative_image'])
    paths = [r['relative_image'] for r in records]
    if len(paths) != len(set(paths)):
        raise SystemExit('annotation CSVs contain duplicate source images')
    return records


def instance_box(value: str, points: dict, width: int, height: int,
                 where: str) -> tuple[tuple[float, float, float, float], str]:
    """Return the image-clipped union of the source BBox and positioned keypoints.

    Source corners are not consistently ordered, so they are sorted per axis. A degenerate or
    too-small source box is expanded by the keypoint extent; the note records any such expansion.
    """
    parts = value.strip().split('_')
    if len(parts) != 4:
        raise SystemExit(f'{where}: malformed BBox={value!r}')
    try:
        a, b, c, d = map(float, parts)
    except ValueError as exc:
        raise SystemExit(f'{where}: bad BBox={value!r}') from exc
    source = (min(a, c), min(b, d), max(a, c), max(b, d))
    pts = np.asarray(list(points.values()), np.float64)
    box = (min(source[0], pts[:, 0].min()), min(source[1], pts[:, 1].min()),
           max(source[2], pts[:, 0].max()), max(source[3], pts[:, 1].max()))
    box = (max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]),
           min(float(height), box[3]))
    if not (box[2] > box[0] and box[3] > box[1]):
        raise SystemExit(f'{where}: empty instance box from BBox={value!r}')
    note = ''
    if any(abs(x - y) > 3.0 for x, y in zip(box, source)):
        note = f'bbox_expanded_to_keypoints=source:{value}'
    return tuple(float(v) for v in box), note


def write_pixels(session_dir: Path, group_id: str, source: Path) -> None:
    """Link JPEG pixels, or losslessly convert BMP pixels to PNG in the group frame directory."""
    camera_dir = session_dir / 'groups' / group_id / CAMERA
    camera_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix in {'.jpg', '.jpeg', '.png'}:
        target = camera_dir / f'000000{suffix}'
        fmt.link(target, source.resolve())
    elif suffix == '.bmp':
        target = camera_dir / '000000.png'
        with Image.open(source) as image:
            image.save(target, format='PNG', optimize=True)
    else:
        raise SystemExit(f'unsupported source image extension: {source}')


def labels_for(record: dict, names: list[str]) -> fmt.Labels:
    """Build one rat, one still, one camera worth of positioned 2D labels.

    Provided points are visible per owner instruction. Each still has one annotated rat, so its
    corner-sorted, keypoint-covering source box is a `labeled` instance.
    """
    lab = fmt.empty_labels(1, 1, len(names), 1, mode3d=False,
                           animal_ids=[record['animal_id']])
    for index, name in enumerate(names):
        if name not in record['points']:
            continue
        lab.points2d[0, 0, index, 0] = record['points'][name]
        lab.vis2d[0, 0, index, 0] = fmt.VISIBLE
    lab.boxes = np.asarray(record['box'], np.float32).reshape(1, 1, 1, 4)
    lab.instance = np.full((1, 1, 1), fmt.INST_LABELED, np.int8)
    return lab


def convert(src: Path, out: Path, *, val_frames: int = VAL_FRAMES,
            seed: int = SEED, clean: bool = False, validate: bool = False) -> None:
    """Write the deterministic train/val conversion and optionally run the format validator."""
    if out.exists():
        if not clean:
            raise SystemExit(f'{out} already exists; use --clean to replace it')
        shutil.rmtree(out)
    records = read_records(src)
    if val_frames < 1 or val_frames >= len(records):
        raise SystemExit(f'val frame count must be in [1, {len(records) - 1}]')
    val_indices = set(random.Random(seed).sample(range(len(records)), val_frames))

    buckets: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for index, record in enumerate(records):
        split = 'val' if index in val_indices else 'train'
        key = (split, record['collection'], record['width'], record['height'])
        buckets[key].append((index, record))

    date_created = date.today().isoformat()
    for (split, collection, width, height), members in sorted(buckets.items()):
        names = COLLECTIONS[collection]['names']
        session_id = f'udarp_{collection}_{width}x{height}'
        session_dir = out / split / session_id
        from aniposelib.cameras import CameraGroup
        camera = fmt.nominal_camera(CAMERA, (width, height))
        rig = fmt.Rig(cgroup=CameraGroup([camera]), offset={CAMERA: (0.0, 0.0)},
                      moving={CAMERA: False}, calibrated={CAMERA: False})
        groups, labels = {}, {}
        for index, record in members:
            group_id = f'g{index:05d}'
            note = f"source_image={record['relative_image']}"
            if record['invalid_points']:
                invalid = ','.join(f'{name}={value}' for name, value in record['invalid_points'])
                note += f'; excluded_out_of_bounds={invalid}'
            if record['zero_points']:
                note += '; excluded_zero_sentinel=' + ','.join(record['zero_points'])
            if record['box_note']:
                note += f"; {record['box_note']}"
            groups[group_id] = fmt.Group(
                group_id, 1, source_frame_start=0, source_frame_step=1, notes=note)
            labels[group_id] = labels_for(record, names)
            write_pixels(session_dir, group_id, record['image'])
        fmt.write_session(
            session_dir, mode='2d', units='px', label_source='annotated', names=names,
            rig=rig, groups=groups, labels=labels, flip_pairs=FLIP_PAIRS,
            provenance={
                'source': 'UDARP-9.4K dataset release (CSDLLab)',
                'source_csv': COLLECTIONS[collection]['csv'],
                'converter': 'scripts/convert_udarp.py',
                'created': date_created,
                'split_method': 'uniform random sample of distinct images across Lawn and Platform',
                'split_seed': int(seed),
                'validation_frames_total': int(val_frames),
                'source_image_count_total': int(len(records)),
                'keypoint_status_note': 'all source keypoint coordinates are stored as visible per dataset '
                                        'owner instruction; source cells use x_y_0; out-of-bounds '
                                        'coordinates and 0_0 sentinels are omitted and named in '
                                        'their group notes',
                'instance_note': 'one labeled rat per still; source BBox corners are sorted per '
                                 'axis, unioned with positioned keypoints, and clipped to the '
                                 'image; expansions are named in group notes',
            })
        print(f'{split}/{session_id}: {len(groups)} image group(s), '
              f'{sum(len(r[1]["points"]) for r in members)} positioned keypoints')

    train_count = sum(len(rows) for key, rows in buckets.items() if key[0] == 'train')
    val_count = sum(len(rows) for key, rows in buckets.items() if key[0] == 'val')
    if (train_count, val_count) != (len(records) - val_frames, val_frames):
        raise RuntimeError(f'split count mismatch: train={train_count}, val={val_count}')
    invalid_count = sum(len(r['invalid_points']) for r in records)
    zero_count = sum(len(r['zero_points']) for r in records)
    expanded = sum(bool(r['box_note']) for r in records)
    print(f'converted {len(records)} images: {train_count} train, {val_count} val '
          f'(seed={seed}); excluded {invalid_count} out-of-bounds and {zero_count} 0_0 '
          f'keypoint(s); expanded {expanded} source box(es); notes in groups.pq; output {out}')

    if validate:
        print('\n-- validation')
        errors = fmt.validate_dataset(fmt.load_dataset(out), check_images=True)
        for error in errors:
            print(('WARN ' if 'WARNING' in error else 'FAIL ') + error)
        hard = [error for error in errors if 'WARNING' not in error]
        print(f'{len(hard)} error(s), {len(errors) - len(hard)} warning(s)')
        if hard:
            raise SystemExit(1)


def main() -> None:
    """Parse conversion options and write the deterministic UDARP split."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, default=SRC)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--val-frames', type=int, default=VAL_FRAMES)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--clean', action='store_true', help='remove the output root first')
    parser.add_argument('--validate', action='store_true', help='validate output and pixels')
    args = parser.parse_args()
    convert(args.src, args.out, val_frames=args.val_frames, seed=args.seed,
            clean=args.clean, validate=args.validate)


if __name__ == '__main__':
    main()
