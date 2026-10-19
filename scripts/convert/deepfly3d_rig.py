"""The bundle-adjusted camera rig of a native ``deeperfly`` result, and its reprojection audit.

Used by ``deepfly3d_v3.py``; imported as ``from deepfly3d_rig import ...``.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from tailcyclenet import format as fmt

from deepfly3d_h5 import N_CAMERAS, N_KEYPOINTS, NativeResult, valid_2d


def _camera_matrix(intr: np.ndarray) -> np.ndarray:
    """Expand compact intrinsics into an OpenCV matrix."""
    fx, fy, cx, cy = [float(v) for v in intr]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def audit(result: NativeResult, *, max_reprojection_px: float | None = None) -> dict:
    """Reproject native 3D and collect support, residual and depth diagnostics."""
    p2, p3 = result.points2d, result.points3d
    observed = valid_2d(result)
    support = observed.sum(axis=0)
    finite3 = np.isfinite(p3).all(axis=-1)
    if not finite3.any():
        raise ValueError('triangulation/points3d has no finite rows')
    per_camera = []
    errors_all: list[float] = []
    depths_all: list[float] = []
    for c in range(N_CAMERAS):
        xyz = p3.reshape(-1, 3)
        projected, _ = cv2.projectPoints(xyz, result.rvecs[c], result.tvecs[c],
                                         _camera_matrix(result.intrinsics[c]), result.distortions[c])
        projected = projected.reshape(result.n_frames, N_KEYPOINTS, 2)
        rmat, _ = cv2.Rodrigues(result.rvecs[c])
        depth = (rmat @ xyz.T + result.tvecs[c, :, None])[2].reshape(result.n_frames, N_KEYPOINTS)
        mask = observed[c] & finite3
        errors = np.linalg.norm(projected - p2[c], axis=-1)[mask]
        depths = depth[mask]
        if np.isfinite(errors).any():
            errors_all.extend(errors[np.isfinite(errors)].tolist())
        if np.isfinite(depths).any():
            depths_all.extend(depths[np.isfinite(depths)].tolist())
        per_camera.append({
            'camera': result.camera_names[c],
            'source_camera_index': c,
            'n_2d': int(observed[c].sum()),
            'n_reprojected': int(mask.sum()),
            'median_px': float(np.median(errors)) if errors.size else None,
            'p95_px': float(np.percentile(errors, 95)) if errors.size else None,
            'max_px': float(np.max(errors)) if errors.size else None,
            'positive_depth_fraction': float(np.mean(depths > 0)) if depths.size else None,
        })
    reproj = np.asarray(errors_all, dtype=float)
    depths = np.asarray(depths_all, dtype=float)
    out = {
        'min_2d_support': int(support.min()),
        'max_2d_support': int(support.max()),
        'support_ge_2_rows': int((support >= 2).sum()),
        'finite_3d_rows': int(finite3.sum()),
        'point3d_rows_after_support': int((finite3 & (support >= 2)).sum()),
        'overall_median_reprojection_px': float(np.median(reproj)) if reproj.size else None,
        'overall_p95_reprojection_px': float(np.percentile(reproj, 95)) if reproj.size else None,
        'overall_max_reprojection_px': float(np.max(reproj)) if reproj.size else None,
        'overall_positive_depth_fraction': float(np.mean(depths > 0)) if depths.size else None,
        'per_camera': per_camera,
    }
    if max_reprojection_px is not None and reproj.size and float(np.max(reproj)) > max_reprojection_px:
        raise ValueError(f'max reprojection {float(np.max(reproj)):.3f}px exceeds '
                         f'--max-reprojection-px {max_reprojection_px:g}')
    if int(out['point3d_rows_after_support']) == 0:
        raise ValueError('no finite 3D point has two finite 2D views')
    return out


def calibration_fingerprint(result: NativeResult, scale: float) -> str:
    """Hash the scaled calibration and camera order for provenance/idempotence audits."""
    payload = {
        'names': result.camera_names,
        'intrinsics': result.intrinsics.tolist(),
        'distortions': [d.tolist() for d in result.distortions],
        'rvecs': result.rvecs.tolist(),
        'scaled_tvecs': (result.tvecs * scale).tolist(),
        'sizes': [list(s) for s in result.image_sizes],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def calibration_rig(result: NativeResult, scale: float) -> fmt.Rig:
    """Build a tailcycle rig from native BA geometry, scaling translations to millimetres."""
    blocks: dict[str, dict] = {}
    for c, name in enumerate(result.camera_names):
        blocks[f'cam_{c}'] = {
            'name': name,
            'size': list(result.image_sizes[c]),
            'matrix': _camera_matrix(result.intrinsics[c]).tolist(),
            'distortions': result.distortions[c].tolist(),
            'rotation': result.rvecs[c].tolist(),
            'translation': (result.tvecs[c] * scale).tolist(),
            'offset': [0.0, 0.0],
            'moving': False,
            'fisheye': False,
        }
    blocks['metadata'] = {
        'source': 'native deeperfly bundle_adjustment/cameras',
        'units': 'mm',
        'status': 'provisional_unvalidated',
        'scale_policy': '0.456 mm / mean lengths of front coxa segments 0-1 and 19-20',
    }
    return fmt.rig_from_doc(blocks, str(result))


def skeleton_for(result: NativeResult) -> tuple[list[list[str]], list[list[str]]]:
    """Translate native index bones and bilateral pairs to name pairs."""
    skeleton = [[result.names[a], result.names[b]] for a, b in result.bones]
    by_name = {n: i for i, n in enumerate(result.names)}
    pairs = []
    for i in range(19):
        left, right = result.names[i], result.names[i + 19]
        if by_name.get(left) != i or by_name.get(right) != i + 19:
            raise ValueError('native fly38 bilateral axis is not in expected left/right order')
        pairs.append([left, right])
    return skeleton, pairs
