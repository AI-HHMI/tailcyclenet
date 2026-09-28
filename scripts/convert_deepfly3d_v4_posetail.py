#!/usr/bin/env python3
"""Convert the tailcycle ``deepfly3d-v4`` root to Posetail's legacy training layout.

Example::

    pixi run python scripts/convert_deepfly3d_v4_posetail.py \
        --src /groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/deepfly3d-v4 \
        --out /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v6/deepfly3d-v4

Each tailcycle session becomes one Posetail trial under its source fly-family session:
``<split>/<condition__date__FlyN>/<source-session>/``. The split assignments from
``split_manifest.json`` are preserved. Images are symlinked, never copied.

The source 3D labels are already quality-masked and expressed in millimetres. Only positioned
``points3d.pq`` rows are copied to ``pose``; non-positioned points remain NaN. Per-camera
``vis`` is 1 for ``visible``, 0 for ``missing``, and NaN where no visibility determination was
made. Posetail currently converts NaN visibility entries to visible when loading, so the 3D
coordinate mask remains the authoritative quality mask. Camera geometry is copied from the
source calibration, converting Rodrigues rotations to 4x4 world-to-camera matrices.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

DEFAULT_SOURCE = Path(
    '/groups/karashchuk/karashchuklab/animal-datasets-processed/'
    'tailcycle-datasets/deepfly3d-v4'
)
DEFAULT_OUTPUT = Path(
    '/groups/karashchuk/karashchuklab/animal-datasets-processed/'
    'posetail-finetuning-v6/deepfly3d-v4'
)
SPLITS = ('train', 'val', 'test')
_FAMILY = re.compile(r'^(\d{6}).*?(Fly\d+)_')
_POSITIONED_3D = {'visible', 'projected'}
_VISIBILITY_STATES = {'visible', 'missing', 'projected', 'unlabeled'}


def _read_toml(path: Path) -> dict[str, Any]:
    """Read one UTF-8 TOML document."""
    with path.open('rb') as f:
        return tomllib.load(f)


def _family_id(condition: str, archive: str) -> str:
    """Recover the family key used by the source split manifest."""
    match = _FAMILY.match(archive)
    if match is None:
        raise ValueError(f'{archive!r}: cannot recover acquisition date and FlyN')
    return f'{condition}__{match.group(1)}__{match.group(2)}'


def _camera_blocks(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    """Return calibrated camera blocks in source camera-index order."""
    blocks = [(key, value) for key, value in calibration.items()
              if re.fullmatch(r'cam_\d+', key)]
    blocks.sort(key=lambda pair: int(pair[0].split('_', 1)[1]))
    if not blocks:
        raise ValueError('calibration.toml contains no [cam_N] camera blocks')
    cameras = [block['name'] for _, block in blocks]
    if len(cameras) != len(set(cameras)):
        raise ValueError(f'calibration.toml has duplicate camera names: {cameras!r}')
    return [block for _, block in blocks]


def _dense_arrays(session_dir: Path, group_id: str, names: list[str],
                  cameras: list[str], n_frames: int) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Scatter one tailcycle session's sparse Parquet tables into Posetail arrays."""
    points = pq.read_table(session_dir / 'points3d.pq').to_pydict()
    keypoints = pq.read_table(session_dir / 'keypoints.pq').to_pydict()
    animal_ids = sorted(set(points['animal_id']) | set(keypoints['animal_id']))
    if not animal_ids:
        raise ValueError(f'{session_dir}: no animal ids in points3d.pq or keypoints.pq')

    index = {name: i for i, name in enumerate(names)}
    animal_index = {animal_id: i for i, animal_id in enumerate(animal_ids)}
    camera_index = {camera: i for i, camera in enumerate(cameras)}
    if len(index) != len(names):
        raise ValueError(f'{session_dir}: duplicate keypoint names in session.toml')

    pose = np.full((len(animal_ids), n_frames, len(names), 3), np.nan, dtype=np.float32)
    seen_3d: set[tuple[int, int, int]] = set()
    n_positioned = 0
    for row in zip(points['group_id'], points['frame'], points['animal_id'],
                   points['bodypart'], points['status'], points['x'], points['y'], points['z']):
        gid, frame, animal, bodypart, status, x, y, z = row
        if gid != group_id:
            raise ValueError(f'{session_dir}: unexpected group id {gid!r}')
        if status not in _POSITIONED_3D:
            if status not in {'missing', 'unlabeled'}:
                raise ValueError(f'{session_dir}: unknown points3d status {status!r}')
            continue
        if bodypart not in index:
            raise ValueError(f'{session_dir}: points3d bodypart {bodypart!r} absent from names')
        if not 0 <= frame < n_frames:
            raise ValueError(f'{session_dir}: points3d frame {frame} outside [0,{n_frames})')
        if None in (x, y, z):
            raise ValueError(f'{session_dir}: positioned 3D row has null coordinates')
        xyz = np.asarray((x, y, z), dtype=np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError(f'{session_dir}: positioned 3D row has non-finite coordinates')
        key = (animal_index[animal], frame, index[bodypart])
        if key in seen_3d:
            raise ValueError(f'{session_dir}: duplicate positioned 3D key {key!r}')
        seen_3d.add(key)
        pose[key[0], key[1], key[2]] = xyz
        n_positioned += 1


    vis = np.full((len(animal_ids), n_frames, len(names), len(cameras)),
                  np.nan, dtype=np.float32)
    seen_vis: set[tuple[int, int, int, int]] = set()
    for row in zip(keypoints['group_id'], keypoints['frame'], keypoints['animal_id'],
                   keypoints['camera'], keypoints['bodypart'], keypoints['status']):
        gid, frame, animal, camera, bodypart, status = row
        if gid != group_id:
            raise ValueError(f'{session_dir}: unexpected group id {gid!r}')
        if status not in _VISIBILITY_STATES:
            raise ValueError(f'{session_dir}: unknown keypoints status {status!r}')
        if not 0 <= frame < n_frames:
            raise ValueError(f'{session_dir}: keypoints frame {frame} outside [0,{n_frames})')
        if camera not in camera_index:
            raise ValueError(f'{session_dir}: keypoints camera {camera!r} absent from calibration')
        if bodypart not in index:
            raise ValueError(f'{session_dir}: keypoints bodypart {bodypart!r} absent from names')
        key = (animal_index[animal], frame, index[bodypart], camera_index[camera])
        if key in seen_vis:
            raise ValueError(f'{session_dir}: duplicate keypoints key {key!r}')
        seen_vis.add(key)
        if status == 'visible':
            vis[key] = 1.0
        elif status == 'missing':
            vis[key] = 0.0


    return pose, vis, animal_ids, n_positioned


def _posetail_metadata(calibration: dict[str, Any], camera_blocks: list[dict[str, Any]],
                       n_frames: int, units: str, calibration_status: str,
                       fps: float | None) -> dict[str, Any]:
    """Translate tailcycle's camera calibration into Posetail's metadata.yaml schema."""
    metadata: dict[str, Any] = {
        'intrinsic_matrices': {},
        'extrinsic_matrices': {},
        'distortion_matrices': {},
        'camera_heights': {},
        'camera_widths': {},
        'num_cameras': len(camera_blocks),
        'num_frames': n_frames,
        'units': units,
        'calibration_status': calibration_status,
    }
    if fps is not None:
        metadata['fps'] = float(fps)

    for block in camera_blocks:
        name = block['name']
        rotation = np.asarray(block['rotation'], dtype=np.float64)
        translation = np.asarray(block['translation'], dtype=np.float64)
        matrix = np.asarray(block['matrix'], dtype=np.float64)
        distortions = np.asarray(block['distortions'], dtype=np.float64).reshape(-1)
        size = block['size']
        if rotation.shape != (3,) or translation.shape != (3,):
            raise ValueError(f'{name}: expected 3-vector Rodrigues rotation and translation')
        if matrix.shape != (3, 3) or len(size) != 2 or len(distortions) != 5:
            raise ValueError(f'{name}: malformed intrinsics, size, or distortion coefficients')
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation_matrix
        world_to_camera[:3, 3] = translation
        metadata['intrinsic_matrices'][name] = matrix.tolist()
        metadata['extrinsic_matrices'][name] = world_to_camera.tolist()

        metadata['distortion_matrices'][name] = [distortions.tolist()]
        metadata['camera_widths'][name] = int(size[0])
        metadata['camera_heights'][name] = int(size[1])

    offsets = {
        block['name']: [float(value) for value in block.get('offset', (0.0, 0.0))[:2]]
        for block in camera_blocks
    }
    if any(value != 0.0 for offset in offsets.values() for value in offset):
        metadata['offset_dict'] = offsets
    return metadata


def _convert_trial(source_session: Path, target_trial: Path, *, split: str,
                   family: str, archive: str, verbose: bool = False) -> dict[str, Any]:
    """Write one source session as one Posetail training trial."""
    session = _read_toml(source_session / 'session.toml')
    calibration = _read_toml(source_session / 'calibration.toml')
    groups = pq.read_table(source_session / 'groups.pq').to_pylist()
    if len(groups) != 1:
        raise ValueError(f'{source_session}: expected one group, found {len(groups)}')
    group = groups[0]
    group_id, n_frames = group['group_id'], int(group['n_frames'])
    names = list(session['names'])
    camera_blocks = _camera_blocks(calibration)
    cameras = sorted(block['name'] for block in camera_blocks)
    source_group = source_session / 'groups' / group_id
    source_camera_dirs = {p.name: p for p in source_group.iterdir() if p.is_dir()}
    if set(source_camera_dirs) != set(cameras):
        raise ValueError(f'{source_session}: pixel cameras {sorted(source_camera_dirs)} do not '
                         f'match calibration cameras {cameras}')
    for camera, directory in source_camera_dirs.items():
        n_images = sum(1 for path in directory.iterdir()
                       if path.suffix.lower() in {'.jpg', '.png'})
        if n_images != n_frames:
            raise ValueError(f'{source_session}: {camera} has {n_images} image frames, '
                             f'groups.pq declares {n_frames}')

    pose, vis, animal_ids, n_positioned = _dense_arrays(
        source_session, group_id, names, cameras, n_frames)
    target_trial.mkdir(parents=True)
    np.savez_compressed(
        target_trial / 'pose3d.npz',
        pose=pose,
        vis=vis,
        keypoints=np.asarray(names),
        ids=np.asarray(animal_ids),
    )

    image_root = target_trial / 'img'
    image_root.mkdir()
    for camera in cameras:
        link = image_root / camera
        link.symlink_to(source_camera_dirs[camera].resolve(), target_is_directory=True)

    provenance = session.get('provenance', {})
    metadata = _posetail_metadata(
        calibration,
        camera_blocks,
        n_frames,
        session.get('units', 'mm'),
        str(provenance.get('status', 'unknown')),
        group.get('fps'),
    )
    (target_trial / 'metadata.yaml').write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding='utf-8')

    if verbose:
        print(f'  {split}/{family}/{archive}: T={n_frames}, K={len(names)}, '
              f'C={len(cameras)}, positioned={n_positioned:,}')
    return {
        'split': split,
        'family': family,
        'trial': archive,
        'source_session': source_session.name,
        'frames': n_frames,
        'cameras': cameras,
        'keypoints': len(names),
        'animal_ids': animal_ids,
        'positioned_3d_rows': n_positioned,
        'pose3d_npz_bytes': (target_trial / 'pose3d.npz').stat().st_size,
        'fps': group.get('fps'),
        'calibration_status': provenance.get('status', 'unknown'),
    }


def convert(source: Path, output: Path, *, force: bool, verbose: bool) -> None:
    """Convert all source sessions into a staged Posetail root, then publish it."""
    source, output = source.resolve(), output.absolute()
    if not source.is_dir():
        raise SystemExit(f'missing source dataset root: {source}')
    split_manifest_path = source / 'split_manifest.json'
    if not split_manifest_path.is_file():
        raise SystemExit(f'missing source split manifest: {split_manifest_path}')
    split_assignment = json.loads(split_manifest_path.read_text(encoding='utf-8'))['assignment']

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f'.{output.name}.tmp')
    if output.exists() and not force:
        raise SystemExit(f'output already exists (use --force to replace): {output}')
    if staging.exists():
        shutil.rmtree(staging)
    for split in SPLITS:
        (staging / split).mkdir(parents=True)

    records: list[dict[str, Any]] = []
    try:
        for split in SPLITS:
            split_dir = source / split
            if not split_dir.is_dir():
                continue
            for source_session in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                session = _read_toml(source_session / 'session.toml')
                provenance = session.get('provenance', {})
                condition = provenance.get('condition')
                archive = provenance.get('source_archive')
                if not condition or not archive:
                    raise ValueError(f'{source_session}: session provenance lacks condition/archive')
                family = _family_id(str(condition), str(archive))
                if split_assignment.get(family) != split:
                    raise ValueError(f'{source_session}: family {family!r} is not assigned to '
                                     f'{split!r} in split_manifest.json')
                target_trial = staging / split / family / str(archive)
                record = _convert_trial(source_session, target_trial, split=split,
                                        family=family, archive=str(archive), verbose=verbose)
                records.append(record)
                if len(records) % 25 == 0:
                    print(f'converted {len(records)} source sessions')

        manifest = {
            'dataset': 'deepfly3d-v4',
            'format': 'Posetail legacy training tree: pose3d.npz + metadata.yaml + img/<camera>',
            'source_root': str(source),
            'created': date.today().isoformat(),
            'labels': 'tracked',
            'split_assignment_source': 'split_manifest.json (preserved)',
            'family_policy': 'condition/date/FlyN groups source trials within each split',
            'units': 'mm',
            'calibration_status': 'provisional_unvalidated (copied from source provenance)',
            'pose_policy': 'positioned points3d rows copied; all non-positioned coordinates remain NaN',
            'visibility_policy': 'visible=1; missing=0; projected/unlabeled/unassessed=NaN',
            'pixel_policy': 'camera image directories symlinked to source session; no image copy',
            'camera_policy': 'tailcycle intrinsics and world-to-camera extrinsics converted to Posetail YAML',
            'fps_policy': 'preserved from groups.pq when declared; otherwise omitted',
            'split_counts': {split: sum(r['split'] == split for r in records)
                             for split in SPLITS},
            'sessions': records,
        }
        (staging / 'conversion_manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')

        if output.exists():
            backup = output.with_name(f'{output.name}.backup-{time.time_ns()}')
            os.replace(output, backup)
            print(f'previous output moved to {backup}')
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f'converted {len(records)} trials to {output}')
    print('split counts: ' + ', '.join(f'{s}={manifest["split_counts"][s]}' for s in SPLITS))


def main() -> None:
    """Parse conversion options and run the dataset conversion."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, default=DEFAULT_SOURCE,
                        help='tailcycle deepfly3d-v4 dataset root')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT,
                        help='Posetail output root (must not exist unless --force is set)')
    parser.add_argument('--force', action='store_true',
                        help='replace an existing output (retained as a timestamped backup)')
    parser.add_argument('--verbose', action='store_true', help='print per-trial conversion details')
    args = parser.parse_args()
    convert(args.src, args.out, force=args.force, verbose=args.verbose)


if __name__ == '__main__':
    main()
