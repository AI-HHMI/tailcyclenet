#!/usr/bin/env python
"""Convert the Facemap Lightning Pose export into tailcycle 2D sessions.

    pixi run python scripts/convert_facemap.py --validate

The source provides 2,400 InD labeled stills and 100 OOD labeled stills, but no InD videos.
Each still is therefore represented as a one-frame group with the original source frame number;
source videos/context are not fabricated. One hundred InD stills are randomly held out for val,
while the original OOD set remains test.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/facemap')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/facemap')
VAL_COUNT = 100
SPLIT_SEED = 42
ANIMAL_ID = 'mouse0'
LEFT = ('eye_back', 'eye_bottom', 'eye_front', 'eye_top', 'nose_bottom',
        'whisker_c1', 'whisker_c2', 'whisker_d1')
MIDDLE = ('lowerlip', 'mouth', 'nose_top', 'nosebridge', 'nose_tip')
RIGHT = ('paw', 'nose_r')


def keypoint_mapping(source_names: list[str]) -> tuple[list[str], dict[str, str], list[list[str]]]:
    """Expand one-sided source labels to bilateral names and declare flip pairs."""
    expected = set(LEFT) | set(MIDDLE) | set(RIGHT)
    if set(source_names) != expected:
        raise RuntimeError(f'unexpected Facemap keypoints: {source_names}')
    names = ([f'l_{name}' for name in LEFT + RIGHT] + list(MIDDLE)
             + [f'r_{name}' for name in LEFT + RIGHT])
    source_to_label = {name: f'l_{name}' for name in LEFT}
    source_to_label.update({name: name for name in MIDDLE})
    source_to_label.update({name: f'r_{name}' for name in RIGHT})
    pairs = [[f'l_{name}', f'r_{name}'] for name in LEFT + RIGHT]
    return names, source_to_label, pairs


def read_labels(path: Path, src: Path) -> tuple[list[str], list[dict], list[list[str]]]:
    """Read a three-header-row DLC CSV into keyed sparse point rows.

    Blank x,y pairs mean no recorded determination and are omitted from the labels.
    """
    with path.open(newline='') as stream:
        rows = csv.reader(stream)
        scorer = next(rows)
        bodyparts = next(rows)
        coords = next(rows)
        if not (len(scorer) == len(bodyparts) == len(coords)):
            raise RuntimeError(f'{path}: inconsistent CSV header widths')
        names = []
        for i in range(1, len(bodyparts), 2):
            if coords[i:i + 2] != ['x', 'y'] or bodyparts[i] != bodyparts[i + 1]:
                raise RuntimeError(f'{path}: expected x,y pairs for each bodypart')
            names.append(bodyparts[i])
        if not names or len(set(names)) != len(names):
            raise RuntimeError(f'{path}: empty or duplicate bodypart names')
        output_names, source_to_label, flip_pairs = keypoint_mapping(names)

        result = []
        seen_paths = set()
        for line_number, row in enumerate(rows, start=4):
            if len(row) != len(bodyparts):
                raise RuntimeError(f'{path}:{line_number}: unexpected CSV row width')
            rel = Path(row[0])
            if rel.is_absolute() or '..' in rel.parts or not (src / rel).is_file():
                raise RuntimeError(f'{path}:{line_number}: missing/unsafe image path {rel}')
            if rel in seen_paths:
                raise RuntimeError(f'{path}:{line_number}: duplicate image path {rel}')
            seen_paths.add(rel)
            image = src / rel
            with Image.open(image) as im:
                width, height = im.size
            points = {}
            for j, name in enumerate(names):
                x_text, y_text = row[1 + 2 * j:3 + 2 * j]
                if bool(x_text) != bool(y_text):
                    raise RuntimeError(f'{path}:{line_number}: partial coordinate for {name}')
                if not x_text:
                    continue
                x, y = float(x_text), float(y_text)
                if not np.isfinite((x, y)).all():
                    raise RuntimeError(f'{path}:{line_number}: non-finite coordinate for {name}')
                if x < 0 or y < 0 or x >= width or y >= height:
                    raise RuntimeError(f'{path}:{line_number}: {name} point outside {image}')
                points[source_to_label[name]] = (x, y)
            session_id = rel.parent.name
            camera_ids = re.findall(r'cam_?(\d+)', session_id)
            if len(camera_ids) != 1:
                raise RuntimeError(f'{path}:{line_number}: cannot uniquely identify camera '
                                   f'from {session_id}')
            camera = f'cam{camera_ids[0]}'
            try:
                source_frame = int(image.stem.removeprefix('img'))
            except ValueError as exc:
                raise RuntimeError(f'{path}:{line_number}: expected img<frame>.png') from exc
            result.append({'rel': rel, 'image': image, 'session_id': session_id,
                           'camera': camera, 'source_frame': source_frame,
                           'width': width, 'height': height, 'points': points})
    return output_names, result, flip_pairs


def write_split(out: Path, split: str, grouped: dict[str, list[dict]], names: list[str],
                flip_pairs: list[list[str]], source_csv: Path, split_note: str) -> int:
    """Write one session per source recording/camera, with a group per labeled still."""
    n_frames = 0
    for session_id, rows in sorted(grouped.items()):
        cam = rows[0]['camera']
        size = (rows[0]['width'], rows[0]['height'])
        if any((row['camera'], row['width'], row['height']) != (cam, *size) for row in rows):
            raise RuntimeError(f'{session_id}: camera or image size changes within session')
        dst = out / split / session_id
        camera = fmt.nominal_camera(cam, size)
        from aniposelib.cameras import CameraGroup
        rig = fmt.Rig(cgroup=CameraGroup([camera]), offset={cam: (0.0, 0.0)},
                      moving={cam: False}, calibrated={cam: False})
        groups = {}
        labels = {}
        for row in sorted(rows, key=lambda item: item['source_frame']):
            frame = row['source_frame']
            gid = f'frame_{frame:06d}'
            if gid in groups:
                raise RuntimeError(f'{session_id}: duplicate source frame {frame}')
            groups[gid] = fmt.Group(
                gid, 1, source_video='', source_frame_start=frame,
                notes='one isolated labeled still; source provides no temporal context')
            lab = fmt.empty_labels(1, 1, len(names), 1, mode3d=False,
                                   animal_ids=[ANIMAL_ID])
            for k, name in enumerate(names):
                if name in row['points']:
                    lab.points2d[0, 0, k, 0] = row['points'][name]
                    lab.vis2d[0, 0, k, 0] = fmt.VISIBLE
            labels[gid] = lab

            pixel_dir = dst / 'groups' / gid / cam
            pixel_dir.mkdir(parents=True, exist_ok=True)
            fmt.link(pixel_dir / '000000.png', row['image'].resolve())

        fmt.write_session(
            dst, mode='2d', units='px', label_source='annotated', names=names,
            rig=rig, groups=groups, labels=labels, skeleton=[], flip_pairs=flip_pairs,
            provenance={
                'source': f'Facemap dataset ({source_csv.name})',
                'annotator': '',
                'annotator_tool': 'DeepLabCut-format CSV',
                'created': '2026-09-30',
                'converter': 'scripts/convert_facemap.py',
                'source_dataset': 'https://doi.org/10.25378/janelia.23712957',
                'source_split': 'InD' if split in {'train', 'val'} else 'OOD',
                'split_note': split_note,
                'coordinate_note': 'Only populated x,y pairs were written as visible; blank cells '
                                   'were omitted as unassessed, not labeled missing.',
                'context_note': 'The source has isolated 400x400 stills and no InD videos; '
                                'each labeled still is a legal n_frames=1 group.',
            },
        )
        n_frames += len(rows)
    return n_frames


def convert(src: Path, out: Path, clean: bool = False, validate: bool = False) -> None:
    """Convert InD/OOD label CSVs to train/val/test without copying source images."""
    if out.exists():
        if not clean:
            raise RuntimeError(f'{out} exists; pass --clean to replace it')
        shutil.rmtree(out)
    names, ind, flip_pairs = read_labels(src / 'CollectedData.csv', src)
    test_names, ood, test_flip_pairs = read_labels(src / 'CollectedData_test.csv', src)
    if names != test_names or flip_pairs != test_flip_pairs:
        raise RuntimeError('InD and OOD keypoint axes differ')
    if len(ind) != 2400 or len(ood) != 100:
        raise RuntimeError(f'expected 2,400 InD and 100 OOD rows, got {len(ind)} and {len(ood)}')

    rng = random.Random(SPLIT_SEED)
    val_rows = rng.sample(ind, VAL_COUNT)
    val_paths = {row['rel'] for row in val_rows}
    train_rows = [row for row in ind if row['rel'] not in val_paths]

    def by_session(rows: list[dict]) -> dict[str, list[dict]]:
        """Group stills by source recording and camera for session writing."""
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row['session_id']].append(row)
        return grouped

    split_note = (f'Random frame-level InD validation holdout: {VAL_COUNT} of {len(ind)} '
                  f'labeled stills, Python random seed {SPLIT_SEED}; train has the remainder. '
                  'Source recordings may occur in both train and val.')
    counts = {
        'train': write_split(out, 'train', by_session(train_rows), names, flip_pairs,
                             src / 'CollectedData.csv', split_note),
        'val': write_split(out, 'val', by_session(val_rows), names, flip_pairs,
                           src / 'CollectedData.csv', split_note),
        'test': write_split(out, 'test', by_session(ood), names, flip_pairs,
                            src / 'CollectedData_test.csv',
                            'Original source OOD split preserved as test.'),
    }
    print(f'Wrote {counts} labeled stills ({len(names)} keypoints, '
          f'{len(flip_pairs)} flip pairs) under {out}')
    print('Images are symlinked to the source; no context frames were synthesized.')

    if validate:
        dataset = fmt.load_dataset(out)
        issues = fmt.validate_dataset(dataset, check_images=True)
        if issues:
            print('\n'.join(issues))
        errors = [issue for issue in issues if 'WARNING]' not in issue]
        if errors:
            raise SystemExit(f'validation reported {len(errors)} error(s)')
        print(f'Validation passed for all sessions and pixel links '
              f'({len(issues)} expected warning(s)).')


def main() -> None:
    """Parse CLI options and convert the Facemap source into the requested output root."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default=SRC)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    convert(args.src, args.out, clean=args.clean, validate=args.validate)


if __name__ == '__main__':
    main()
