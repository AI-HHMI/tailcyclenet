#!/usr/bin/env python3
"""Convert DeepFly3D results into a provisional calibrated tailcycle 3-D root.

This is deliberately labelled provisional: the DeepFly3D calibration has no metric-unit
metadata and its reprojection/rig audit is not a scientific calibration acceptance test.  The
mechanical conversion preserves the source calibration and records the audit in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from tailcyclenet import format as fmt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_deepfly3d_v1 import (
    CAMERA_ORDER, IMAGE_SIZE, N_CAMERAS, N_FRAMES, NAMES, SKELETON, FLIP_PAIRS,
    SOURCE_CONDITIONS, Record, camera_files, discover, sha256, split_map,
)

UNITS = 'deepfly3d_calibrated'
V1_ROOT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
               'tailcycle-datasets/deepfly3d-v1')


def load_prediction(record: Record) -> dict:
    if record.result is None or record.reason:
        raise ValueError(f'{record.condition}/{record.archive}: no usable result')
    with record.result.open('rb') as f:
        p = pickle.load(f)
    p2 = np.asarray(p['points2d'], dtype=np.float32)
    p3 = np.asarray(p['points3d_wo_procrustes'], dtype=np.float32)
    if p2.shape != (N_CAMERAS, N_FRAMES + 1, 38, 2):
        raise ValueError(f'{record.result}: points2d shape {p2.shape}, expected '
                         f'{(N_CAMERAS, N_FRAMES + 1, 38, 2)}')
    if p3.shape != (N_FRAMES + 1, 38, 3):
        raise ValueError(f'{record.result}: points3d_wo_procrustes shape {p3.shape}, expected '
                         f'{(N_FRAMES + 1, 38, 3)}')
    if np.asarray(p.get('camera_ordering', []), dtype=int).tolist() != list(CAMERA_ORDER):
        raise ValueError(f'{record.result}: unexpected camera_ordering')
    return p


def calibration_rig(pred: dict, where: str) -> fmt.Rig:
    """Translate DeepFly3D's physical camera dictionaries without inverting them.

    DeepFly3D uses X_camera = R @ X_world + tvec.  aniposelib's serialized calibration uses a
    Rodrigues rotation vector and the same additive translation convention for this rig.
    Result dictionary key c is the physical camera_c; camera_ordering is not applied again.
    """
    blocks = {}
    for c in range(N_CAMERAS):
        src = pred[c]
        rvec, _ = cv2.Rodrigues(np.asarray(src['R'], dtype=np.float64))
        blocks[f'cam_{c}'] = {
            'name': f'cam{c}',
            'size': list(IMAGE_SIZE),
            'matrix': np.asarray(src['intr'], dtype=np.float64).tolist(),
            'distortions': np.asarray(src['distort'], dtype=np.float64).ravel().tolist(),
            'rotation': rvec.ravel().tolist(),
            'translation': np.asarray(src['tvec'], dtype=np.float64).ravel().tolist(),
            'offset': [0.0, 0.0],
            'moving': False,
            'fisheye': False,
        }
    blocks['metadata'] = {
        'source': 'DeepFly3D result calibration dictionaries',
        'quality': 'provisional; unvalidated rough calibration',
        'units': UNITS,
    }
    return fmt.rig_from_doc(blocks, where)


def calibration_fingerprint(pred: dict) -> str:
    payload = []
    for c in range(N_CAMERAS):
        payload.append({k: np.asarray(pred[c][k]).tolist() for k in ('R', 'tvec', 'intr', 'distort')})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_prediction(pred: dict) -> dict:
    """Record mechanical projection/depth diagnostics; do not silently gate the requested root."""
    p2 = np.asarray(pred['points2d'], dtype=np.float64)[:, :N_FRAMES]
    p3 = np.asarray(pred['points3d_wo_procrustes'], dtype=np.float64)[:N_FRAMES]
    valid2 = np.isfinite(p2).all(-1) & ~((p2 == 0).all(-1))
    finite3 = np.isfinite(p3).all(-1)
    support = valid2.sum(axis=0)
    per_camera = []
    all_errors = []
    all_positive = []
    for c in range(N_CAMERAS):
        src = pred[c]
        rvec, _ = cv2.Rodrigues(np.asarray(src['R'], dtype=np.float64))
        projected, _ = cv2.projectPoints(
            p3.reshape(-1, 3), rvec, np.asarray(src['tvec'], dtype=np.float64),
            np.asarray(src['intr'], dtype=np.float64), np.asarray(src['distort'], dtype=np.float64),
        )
        projected = projected.reshape(N_FRAMES, 38, 2)
        observed = np.stack((p2[c, ..., 1] * IMAGE_SIZE[0], p2[c, ..., 0] * IMAGE_SIZE[1]), -1)
        mask = valid2[c] & finite3
        errors = np.linalg.norm(projected - observed, axis=-1)[mask]
        cam_xyz = (np.asarray(src['R'], dtype=np.float64) @ p3.reshape(-1, 3).T
                   + np.asarray(src['tvec'], dtype=np.float64)[:, None])
        depth = cam_xyz[2].reshape(N_FRAMES, 38)
        positive = depth[mask] > 0
        all_errors.extend(errors.tolist())
        all_positive.extend(positive.tolist())
        per_camera.append({
            'camera': c,
            'n_2d': int(mask.sum()),
            'median_px': float(np.median(errors)) if len(errors) else None,
            'p95_px': float(np.percentile(errors, 95)) if len(errors) else None,
            'max_px': float(np.max(errors)) if len(errors) else None,
            'positive_depth_fraction': float(np.mean(positive)) if len(positive) else None,
        })
    return {
        'min_2d_support': int(support.min()),
        'max_2d_support': int(support.max()),
        'finite_3d_rows': int(finite3.sum()),
        'points3d_rows': int((finite3 & (support >= 2)).sum()),
        'overall_median_reprojection_px': float(np.median(all_errors)) if all_errors else None,
        'overall_p95_reprojection_px': float(np.percentile(all_errors, 95)) if all_errors else None,
        'overall_max_reprojection_px': float(np.max(all_errors)) if all_errors else None,
        'overall_positive_depth_fraction': float(np.mean(all_positive)) if all_positive else None,
        'per_camera': per_camera,
    }


def pose_correction_file(record: Record) -> Path | None:
    files = sorted(record.result_dir.glob('pose_corr_*.pkl'))
    if len(files) > 1:
        raise ValueError(f'{record.result_dir}: multiple pose correction files')
    return files[0] if files else None


def v1_pixel_dir(record: Record, split: str, camera: int) -> Path:
    """Reuse the already-validated v1 symlink farm; its numbered links resolve to source JPEGs."""
    p = V1_ROOT / split / f'{record.condition}__{record.archive}__cam{camera}' \
        / 'groups' / 'g000' / 'cam0'
    if not p.is_dir() or not (p / '000000.jpg').exists():
        raise FileNotFoundError(f'missing v1 pixel farm for {record.condition}/{record.archive}/cam{camera}: {p}')
    return p


def convert_record(record: Record, split: str, root: Path, *, force: bool) -> dict:
    if record.reason:
        return {'condition': record.condition, 'archive': record.archive, 'split': split,
                'status': 'quarantined', 'reason': record.reason}
    pred = load_prediction(record)
    p2 = np.asarray(pred['points2d'], dtype=np.float32)[:, :N_FRAMES]
    p3 = np.asarray(pred['points3d_wo_procrustes'], dtype=np.float32)[:N_FRAMES]
    valid2 = np.isfinite(p2).all(-1) & ~((p2 == 0).all(-1))
    support = valid2.sum(axis=0)
    finite3 = np.isfinite(p3).all(-1)
    valid3 = finite3 & (support >= 2)
    if not valid3.all():
        raise ValueError(f'{record.result}: 3-D rows lack finite coordinates or two-view support')

    session_id = f'{record.condition}__{record.archive}'
    session_dir = root / split / session_id
    if session_dir.exists():
        if not force:
            raise SystemExit(f'output exists (use --force to replace): {session_dir}')
        shutil.rmtree(session_dir)
    temp = root / split / f'.{session_id}.tmp'
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    xy = np.stack((p2[..., 1] * IMAGE_SIZE[0], p2[..., 0] * IMAGE_SIZE[1]), axis=-1)
    dense_xy = np.transpose(xy, (1, 2, 0, 3))[None]
    dense_vis2d = np.full((1, N_FRAMES, 38, N_CAMERAS), fmt.UNLABELED, np.int8)
    dense_vis2d[0][np.transpose(valid2, (1, 2, 0))] = fmt.PROJECTED
    dense_xyz = p3[None].copy()
    dense_vis3d = np.full((1, N_FRAMES, 38), fmt.UNLABELED, np.int8)
    dense_vis3d[0][valid3] = fmt.VISIBLE

    audit = audit_prediction(pred)
    corr = pose_correction_file(record)
    group = fmt.Group(
        group_id='g000', n_frames=N_FRAMES, source_video=f'{record.condition}/{record.archive}',
        source_frame_start=0, source_frame_step=1,
        notes='DeepFly3D tracked provisional 3D; calibration and metric scale are unvalidated; '
              'frame 900 trimmed from 901-result output',
    )
    labels = fmt.Labels(
        animal_ids=['fly0'], points3d=dense_xyz, vis3d=dense_vis3d,
        points2d=dense_xy, vis2d=dense_vis2d, boxes=None, instance=None,
        regions=np.zeros((0, 6), np.float64),
    )
    provenance = {
        'source': 'DeepFly3D v1 Ramdya Dataverse v2 tracked output',
        'annotator': '',
        'annotator_tool': 'DeepFly3D commit 03125320e7e81bfd47e40baf5f7fe765406a06fa',
        'created': date.today().isoformat(),
        'source_condition': record.condition,
        'source_archive': record.archive,
        'source_result': str(record.result),
        'source_result_sha256': sha256(record.result),
        'source_pose_correction_sha256': sha256(corr) if corr else '',
        'source_weight': 'sh8_front_j8.tar (sh8_deepfly.tar)',
        'runner_camera_order': '6,5,4,3,2,1,0',
        'camera_dictionary_policy': 'result key c maps to physical camera c; no second reversal',
        'calibration_policy': 'provisional source calibration; unvalidated rough rig and scale',
        'calibration_fingerprint': calibration_fingerprint(pred),
        'point3d_source': 'points3d_wo_procrustes',
        'point3d_status_policy': 'visible iff finite and at least two finite nonzero 2D views',
        'point2d_status_policy': 'projected iff finite and nonzero; zero is structural no-row sentinel',
        'frame_policy': 'first_900_source_frames_from_901_output_frames',
        'camera3_policy': 'retain calibration/pixels; source points2d all zero, emit no 2D rows',
        'audit_status': 'provisional_unvalidated',
        'audit_min_2d_support': int(audit['min_2d_support']),
        'audit_median_reprojection_px': audit['overall_median_reprojection_px'],
        'audit_p95_reprojection_px': audit['overall_p95_reprojection_px'],
        'audit_positive_depth_fraction': audit['overall_positive_depth_fraction'],
    }
    rig = calibration_rig(pred, str(record.result))
    fmt.write_session(temp, mode='3d', units=UNITS, label_source='tracked', names=NAMES,
                      rig=rig, groups={'g000': group}, labels={'g000': labels},
                      skeleton=SKELETON, flip_pairs=FLIP_PAIRS, provenance=provenance)

    for camera in range(N_CAMERAS):
        source_dir = v1_pixel_dir(record, split, camera)
        pixel_link = temp / 'groups' / 'g000' / f'cam{camera}'
        pixel_link.parent.mkdir(parents=True, exist_ok=True)
        pixel_link.symlink_to(source_dir)

    loaded = fmt.Session.load(temp)
    errors = fmt.validate_session(loaded, check_images=True)
    if errors:
        shutil.rmtree(temp)
        raise RuntimeError('\n'.join(errors[:20]))
    temp.rename(session_dir)
    return {
        'condition': record.condition, 'archive': record.archive, 'split': split,
        'session': str(session_dir.relative_to(root)), 'status': 'converted',
        'source_result': str(record.result), 'source_result_sha256': sha256(record.result),
        'source_pose_correction_sha256': sha256(corr) if corr else None,
        'calibration_fingerprint': calibration_fingerprint(pred),
        'point3d_rows': int(valid3.sum()),
        'keypoint_rows': int(valid2.sum()),
        'audit': audit,
    }


def worker(payload) -> list[dict]:
    records, splits, root, force = payload
    out = []
    for record in records:
        out.append(convert_record(record, splits[record.family], root, force=force))
        print(f'converted {record.condition}/{record.archive}', file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--outputs', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    records = discover(args.source, args.outputs)
    root = args.out
    if root.exists() and any(root.iterdir()) and not args.force:
        raise SystemExit(f'output root is non-empty (use --force to replace): {root}')
    if root.exists() and args.force:
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for split in fmt.SPLITS:
        (root / split).mkdir()

    splits = split_map(records)
    (root / 'split_manifest.json').write_text(json.dumps({
        'policy': 'family grouped; condition + acquisition date + FlyN stays in one split',
        'assignment': splits,
    }, indent=2, sort_keys=True) + '\n')
    groups = {condition: [r for r in records if r.condition == condition]
              for condition in SOURCE_CONDITIONS}
    payloads = [(groups[c], splits, root, args.force) for c in SOURCE_CONDITIONS]
    inventory = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(worker, payloads):
            inventory.extend(result)

    root_manifest = {
        'dataset': 'deepfly3d-v2',
        'format': 'tailcycle-dataset; docs/annotation_format.md',
        'mode': '3d',
        'units': UNITS,
        'labels': 'tracked',
        'status': 'provisional_unvalidated',
        'source_root': str(args.source),
        'outputs_root': str(args.outputs),
        'created': date.today().isoformat(),
        'keypoint_names_sha256': hashlib.sha256('\n'.join(NAMES).encode()).hexdigest(),
        'camera_ordering': list(CAMERA_ORDER),
        'camera_policy': 'seven physical cameras; result key c maps to camera c',
        'calibration_policy': 'preserve per-session source R/tvec/intr/distort; do not call metric',
        'point3d_source': 'points3d_wo_procrustes, first 900 frames only',
        'camera3_policy': 'calibration and pixels retained; no keypoint rows because source packer emits all-zero points2d[3]',
        'pixel_policy': 'session camera directories symlink to v1 validated source-image farms',
        'expected_source_sessions': len(records),
        'converted_sessions': sum(x.get('status') == 'converted' for x in inventory),
        'quarantined_source_sessions': sum(x.get('status') == 'quarantined' for x in inventory),
        'split_counts': {s: sum(x.get('status') == 'converted' and x.get('split') == s for x in inventory)
                         for s in fmt.SPLITS},
        'records': inventory,
    }
    (root / 'conversion_manifest.json').write_text(json.dumps(root_manifest, indent=2, sort_keys=True) + '\n')
    print(f'converted {root_manifest["converted_sessions"]} calibrated 3-D sessions; '
          f'quarantined {root_manifest["quarantined_source_sessions"]}')
    print('WARNING: this is a provisional unvalidated calibration/scale product; '
          'see conversion_manifest.json audit metrics.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
