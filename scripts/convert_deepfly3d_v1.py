#!/usr/bin/env python3
"""Convert the Ramdya DeepFly3D outputs into a conservative tailcycle 2-D root.

The bundled DeepFly3D calibration has not been rig-validated, so the default and only production
mode here is one single-camera ``mode = 2d`` session per source camera.  A future 3-D converter must
first establish a validated calibration and units; this script intentionally refuses to invent them.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from tailcyclenet import format as fmt

SOURCE_CONDITIONS = (
    'aDN-GAL4_Control',
    'MDN-GAL4_Control',
    'aDN-GAL4_UAS-CsChrimson',
    'MDN-GAL4_UAS-CsChrimson',
)
N_CAMERAS = 7
N_FRAMES = 900
IMAGE_SIZE = (960, 480)  # width, height
CAMERA_ORDER = (6, 5, 4, 3, 2, 1, 0)

# DeepFly3D's fixed 38-joint axis.  The source skeleton_fly.py is the authority for these
# positions; names are deliberately explicit so a consumer cannot silently sort the axis.
NAMES = [
    'r_front_body_coxa', 'r_front_coxa_femur', 'r_front_femur_tibia',
    'r_front_tibia_tarsus', 'r_front_tarsus_tip',
    'r_middle_body_coxa', 'r_middle_coxa_femur', 'r_middle_femur_tibia',
    'r_middle_tibia_tarsus', 'r_middle_tarsus_tip',
    'r_hind_body_coxa', 'r_hind_coxa_femur', 'r_hind_femur_tibia',
    'r_hind_tibia_tarsus', 'r_hind_tarsus_tip',
    'r_antenna', 'r_stripe_0', 'r_stripe_1', 'r_stripe_2',
    'l_front_body_coxa', 'l_front_coxa_femur', 'l_front_femur_tibia',
    'l_front_tibia_tarsus', 'l_front_tarsus_tip',
    'l_middle_body_coxa', 'l_middle_coxa_femur', 'l_middle_femur_tibia',
    'l_middle_tibia_tarsus', 'l_middle_tarsus_tip',
    'l_hind_body_coxa', 'l_hind_coxa_femur', 'l_hind_femur_tibia',
    'l_hind_tibia_tarsus', 'l_hind_tarsus_tip',
    'l_antenna', 'l_stripe_0', 'l_stripe_1', 'l_stripe_2',
]
_SOURCE_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8), (8, 9),
    (10, 11), (11, 12), (12, 13), (13, 14),
    (16, 17), (17, 18),
    (19, 20), (20, 21), (21, 22), (22, 23),
    (24, 25), (25, 26), (26, 27), (27, 28),
    (29, 30), (30, 31), (31, 32), (32, 33),
    (35, 36), (36, 37),
)
SKELETON = [[NAMES[a], NAMES[b]] for a, b in _SOURCE_BONES] + [[NAMES[15], NAMES[34]]]
FLIP_PAIRS = [[NAMES[i], NAMES[i + 19]] for i in range(19)]


@dataclass(frozen=True)
class Record:
    condition: str
    archive: str
    images: Path
    result_dir: Path
    result: Path | None
    family: str
    reason: str | None = None


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def family_id(condition: str, archive: str) -> str:
    m = re.match(r'^(\d+)_.*?_Fly(\d+)_', archive)
    if not m:
        raise ValueError(f'cannot derive fly family from {archive!r}')
    return f'{condition}__{m.group(1)}__Fly{m.group(2)}'


def discover(source: Path, outputs: Path) -> list[Record]:
    records: list[Record] = []
    for condition in SOURCE_CONDITIONS:
        extracted = source / condition / 'extracted'
        result_root = outputs / condition
        if not extracted.is_dir():
            raise SystemExit(f'missing extracted directory: {extracted}')
        for d in sorted(extracted.glob('*_behData_images')):
            if not d.is_dir():
                continue
            archive = d.name
            images = d / 'images'
            result_dir = result_root / archive
            results = sorted(result_dir.glob('df3d_result_*.pkl')) if result_dir.is_dir() else []
            reason = None
            result = None
            if not images.is_dir():
                reason = 'missing_images'
            elif len(results) != 1:
                reason = 'missing_result' if not results else 'ambiguous_result'
            else:
                result = results[0]
            records.append(Record(condition, archive, images, result_dir, result,
                                  family_id(condition, archive), reason))
    return records


def camera_files(images: Path, camera: int) -> list[Path]:
    files = sorted(images.glob(f'camera_{camera}_img_*.jpg'))
    expected = [images / f'camera_{camera}_img_{i:06d}.jpg' for i in range(N_FRAMES)]
    if files != expected:
        raise ValueError(f'{images}: camera {camera} is not exactly 000000..000899')
    return files


def inspect_record(record: Record) -> dict:
    out = {
        'condition': record.condition,
        'archive': record.archive,
        'family': record.family,
        'images': str(record.images),
        'result': str(record.result) if record.result else None,
        'reason': record.reason,
        'camera_counts': [],
        'result_sha256': None,
        'result_shape': None,
        'camera_ordering': None,
    }
    if record.reason:
        return out
    try:
        counts = [len(camera_files(record.images, c)) for c in range(N_CAMERAS)]
        out['camera_counts'] = counts
        with record.result.open('rb') as f:
            prediction = pickle.load(f)
        points = np.asarray(prediction['points2d'])
        out['result_shape'] = list(points.shape)
        ordering = np.asarray(prediction.get('camera_ordering', []), dtype=int).tolist()
        out['camera_ordering'] = ordering
        if points.ndim != 4 or points.shape[0] != N_CAMERAS or points.shape[2:] != (38, 2):
            raise ValueError(f'points2d has unexpected shape {points.shape}')
        if points.shape[1] < N_FRAMES:
            raise ValueError(f'points2d has only {points.shape[1]} frames')
        if ordering != list(CAMERA_ORDER):
            raise ValueError(f'camera_ordering {ordering} != {list(CAMERA_ORDER)}')
        out['result_sha256'] = sha256(record.result)
    except (KeyError, OSError, ValueError, pickle.PickleError) as exc:
        out['reason'] = 'invalid_result:' + str(exc)
    return out


def split_map(records: list[Record]) -> dict[str, str]:
    families = sorted({r.family for r in records})
    result = {}
    for i, family in enumerate(families):
        result[family] = ('train', 'val', 'test')[2 if i % 10 == 9 else 1 if i % 10 == 8 else 0]
    return result


def nominal_rig():
    from aniposelib.cameras import CameraGroup
    cam = fmt.nominal_camera('cam0', IMAGE_SIZE)
    return fmt.Rig(cgroup=CameraGroup([cam]), offset={'cam0': (0.0, 0.0)},
                   moving={'cam0': False}, calibrated={'cam0': False})


def load_points(result: Path) -> np.ndarray:
    with result.open('rb') as f:
        prediction = pickle.load(f)
    points = np.asarray(prediction['points2d'], dtype=np.float32)
    if points.shape != (N_CAMERAS, N_FRAMES + 1, 38, 2):
        if points.ndim != 4 or points.shape[0] != N_CAMERAS or points.shape[1] < N_FRAMES \
                or points.shape[2:] != (38, 2):
            raise ValueError(f'{result}: unexpected points2d shape {points.shape}')
    ordering = np.asarray(prediction.get('camera_ordering', []), dtype=int).tolist()
    if ordering != list(CAMERA_ORDER):
        raise ValueError(f'{result}: camera_ordering {ordering} != {list(CAMERA_ORDER)}')
    return points[:, :N_FRAMES]


def link_pixels(view_dir: Path, source_files: list[Path]) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(source_files):
        dst = view_dir / f'{i:06d}.jpg'
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve() != source.resolve():
                raise RuntimeError(f'pixel link collision: {dst}')
            continue
        dst.symlink_to(source)


def convert_record(record: Record, split: str, root: Path, *, force: bool) -> dict:
    if record.reason:
        return {'condition': record.condition, 'archive': record.archive, 'split': split,
                'status': 'quarantined', 'reason': record.reason}
    # This function is called once for each physical camera so that mode=2d has exactly one
    # declared camera. The caller appends the camera number to the session id.
    raise AssertionError('convert_record requires a camera; call convert_camera')


def convert_camera(record: Record, camera: int, split: str, root: Path, *, force: bool) -> dict:
    points = load_points(record.result)
    session_id = f'{record.condition}__{record.archive}__cam{camera}'
    session_dir = root / split / session_id
    if session_dir.exists():
        if not force:
            raise SystemExit(f'output exists (use --force to replace): {session_dir}')
        shutil.rmtree(session_dir)
    temp = root / split / f'.{session_id}.tmp'
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    # DeepFly3D stores normalized (row, column), while tailcycle stores image (x, y) pixels.
    raw = points[camera]
    valid = np.isfinite(raw).all(axis=-1) & ~((raw == 0).all(axis=-1))
    xy = np.stack((raw[..., 1] * IMAGE_SIZE[0], raw[..., 0] * IMAGE_SIZE[1]), axis=-1)
    dense_xy = np.full((1, N_FRAMES, 38, 1, 2), np.nan, np.float32)
    dense_vis = np.full((1, N_FRAMES, 38, 1), fmt.UNLABELED, np.int8)
    dense_xy[0, :, :, 0] = xy
    dense_vis[0, :, :, 0][valid] = fmt.PROJECTED

    group = fmt.Group(
        group_id='g000', n_frames=N_FRAMES, source_video=f'{record.condition}/{record.archive}',
        source_frame_start=0, source_frame_step=1,
        notes='DeepFly3D tracked 2D fallback; no visibility assessment; 901th output frame trimmed',
    )
    labels = fmt.Labels(
        animal_ids=['fly0'], points3d=None, vis3d=None, points2d=dense_xy,
        vis2d=dense_vis, boxes=None, instance=None, regions=np.zeros((0, 6), np.float64),
    )
    provenance = {
        'source': 'DeepFly3D v1 Ramdya Dataverse v2 tracked output',
        'annotator': '',
        'annotator_tool': 'DeepFly3D commit 03125320e7e81bfd47e40baf5f7fe765406a06fa',
        'created': date.today().isoformat(),
        'source_condition': record.condition,
        'source_archive': record.archive,
        'source_camera': str(camera),
        'source_result': str(record.result),
        'calibration_policy': '2d_fallback_3d_calibration_unvalidated',
        'source_camera_ordering': '6,5,4,3,2,1,0',
        'frame_policy': 'first_900_source_frames_from_901_output_frames',
    }
    fmt.write_session(temp, mode='2d', units='px', label_source='tracked', names=NAMES,
                      rig=nominal_rig(), groups={'g000': group}, labels={'g000': labels},
                      skeleton=SKELETON, flip_pairs=FLIP_PAIRS, provenance=provenance)

    source_files = camera_files(record.images, camera)
    # Keep the image bytes in the source archive. The session contains one camera directory link;
    # the shared target-side view directory contains only numbered symlinks.
    view_dir = root / '.pixel_views' / f'{record.condition}__{record.archive}__cam{camera}'
    link_pixels(view_dir, source_files)
    pixel_link = temp / 'groups' / 'g000' / 'cam0'
    pixel_link.parent.mkdir(parents=True, exist_ok=True)
    pixel_link.symlink_to(view_dir)

    loaded = fmt.Session.load(temp)
    errors = fmt.validate_session(loaded, check_images=True)
    if errors:
        shutil.rmtree(temp)
        raise RuntimeError('\n'.join(errors[:20]))
    temp.rename(session_dir)
    return {
        'condition': record.condition, 'archive': record.archive, 'camera': camera,
        'split': split, 'session': str(session_dir.relative_to(root)), 'status': 'converted',
        'result': str(record.result), 'result_sha256': sha256(record.result),
        'source_images': str(record.images), 'source_frame_count': N_FRAMES,
        'positioned_2d_rows': int(valid.sum()),
    }


def inventory_command(args) -> int:
    records = discover(Path(args.source), Path(args.outputs))
    report = [inspect_record(r) for r in records]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({'expected_sessions': len(records), 'records': report},
                                            indent=2, sort_keys=True) + '\n')
    bad = [r for r in report if r['reason']]
    print(f'inventory: {len(records)} sessions; {len(bad)} with issues; report={args.report}')
    return 1 if bad and args.strict else 0


def convert_condition(payload) -> list[dict]:
    """Convert one condition in a worker; source clips and target sessions are disjoint."""
    records, splits, root, force = payload
    inventory = []
    for record in records:
        split = splits[record.family]
        if record.reason:
            inventory.append({'condition': record.condition, 'archive': record.archive,
                              'split': split, 'status': 'quarantined', 'reason': record.reason})
            continue
        for camera in range(N_CAMERAS):
            inventory.append(convert_camera(record, camera, split, root, force=force))
        print(f'converted {record.condition}/{record.archive}', file=sys.stderr)
    return inventory


def convert_command(args) -> int:
    if args.mode != '2d-fallback':
        raise SystemExit('3-D conversion is refused: calibration and units are not rig-validated; '
                         'use --mode 2d-fallback')
    source, outputs, root = Path(args.source), Path(args.outputs), Path(args.out)
    records = discover(source, outputs)
    if root.exists() and any(root.iterdir()) and not args.force:
        raise SystemExit(f'output root is non-empty (use --force to replace): {root}')
    if root.exists() and args.force:
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for split in fmt.SPLITS:
        (root / split).mkdir()
    splits = split_map(records)
    (root / 'split_manifest.json').write_text(json.dumps({
        'policy': 'family grouped; condition + recording date + FlyN stays in one split',
        'assignment': splits,
    }, indent=2, sort_keys=True) + '\n')

    by_condition = {
        condition: [r for r in records if r.condition == condition]
        for condition in SOURCE_CONDITIONS
    }
    payloads = [(by_condition[condition], splits, root, args.force)
                for condition in SOURCE_CONDITIONS]
    inventory = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for result in pool.map(convert_condition, payloads):
            inventory.extend(result)

    manifest = {
        'dataset': 'deepfly3d-v1',
        'format': 'tailcycle-dataset; docs/annotation_format.md',
        'mode': '2d-fallback',
        'units': 'px',
        'labels': 'tracked',
        'source_root': str(source),
        'outputs_root': str(outputs),
        'created': date.today().isoformat(),
        'keypoint_names_sha256': hashlib.sha256('\n'.join(NAMES).encode()).hexdigest(),
        'skeleton': SKELETON,
        'flip_pairs': FLIP_PAIRS,
        'camera_ordering_required': list(CAMERA_ORDER),
        'session_policy': 'one target mode=2d session per source clip and physical camera',
        'calibration_policy': '3-D rejected pending rig validation; nominal one-camera calibration',
        'expected_source_sessions': len(records),
        'converted_sessions': sum(x.get('status') == 'converted' for x in inventory),
        'quarantined_source_sessions': sum(x.get('status') == 'quarantined' for x in inventory),
        'target_sessions': sum(x.get('status') == 'converted' for x in inventory),
        'records': inventory,
    }
    (root / 'conversion_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'converted {manifest["converted_sessions"]} camera sessions from '
          f'{manifest["expected_source_sessions"]} source clips; '
          f'quarantined {manifest["quarantined_source_sessions"]} clips')
    return 0


def main() -> int:
    """Dispatch the inventory or conversion subcommand."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--source', required=True)
    inv.add_argument('--outputs', required=True)
    inv.add_argument('--report', required=True)
    inv.add_argument('--strict', action='store_true')
    inv.set_defaults(func=inventory_command)
    conv = sub.add_parser('convert')
    conv.add_argument('--source', required=True)
    conv.add_argument('--outputs', required=True)
    conv.add_argument('--out', required=True)
    conv.add_argument('--mode', choices=('2d-fallback', '3d'), default='2d-fallback')
    conv.add_argument('--force', action='store_true')
    conv.add_argument('--workers', type=int, default=4,
                      help='parallel condition workers (default: 4)')
    conv.set_defaults(func=convert_command)
    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
