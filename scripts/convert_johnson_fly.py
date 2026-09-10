#!/usr/bin/env python
"""Convert a Johnson Lab Fly50 JARVIS release to tailcycle-dataset format.

Fly50_V7 uses telecentric/orthographic 3x4 projection matrices.  This converter
uses the closed-form telecentric-to-telephoto-pinhole construction documented in
``posetail-preprocessing/preprocess.md`` and preserves the JARVIS train/val
frameset membership.  The source's projectionMatrix already contains the
JARVIS ``scale: 10`` multiplier conversion; it is therefore used as-is.

The source has no native 3D table.  3D points are triangulated from the 2D
annotations with the converted pinhole rig, while the original 2D annotations
remain untouched and are marked ``projected`` because Fly50 records placement,
not per-camera visibility judgments.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/johnson/johnsonlab/flypose/releases/Fly50_V7')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
           'tailcycle-datasets/johnson-fly-v7')
SPLITS = ('train', 'val')
EPS_PIXEL = 0.5
MAD_K = 10.0



# source reading


def parse_projection(path: Path) -> np.ndarray:
    """Read FlyPose's OpenCV ``projectionMatrix`` without cv2."""
    text = path.read_text()
    m = re.search(r'projectionMatrix:.*?rows:\s*(\d+).*?cols:\s*(\d+)'
                  r'.*?data:\s*\[(.*?)\]', text, re.S)
    if m is None:
        raise RuntimeError(f'{path}: no projectionMatrix')
    rows, cols = int(m.group(1)), int(m.group(2))
    values = np.fromstring(m.group(3), sep=',', dtype=np.float64)
    if (rows, cols) != (3, 4) or values.size != 12 or not np.isfinite(values).all():
        raise RuntimeError(f'{path}: expected a finite 3x4 projectionMatrix, got '
                           f'{rows}x{cols} with {values.size} values')
    P = values.reshape(3, 4)
    if not np.allclose(P[2], [0, 0, 0, 1], atol=1e-9):
        raise RuntimeError(f'{path}: projectionMatrix is not telecentric: {P[2]!r}')
    return P


def read_split(src: Path, split: str) -> dict:
    """Read one JARVIS split and index images, annotations and framesets."""
    with open(src / 'annotations' / f'instances_{split}.json') as f:
        data = json.load(f)
    images = {int(im['id']): im for im in data['images']}
    annotations = {}
    for ann in data['annotations']:
        iid = int(ann['image_id'])
        if iid in annotations:
            raise RuntimeError(f'{split}: image {iid} has more than one annotation')
        annotations[iid] = ann

    framesets: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    for key, fs in data['framesets'].items():
        session, frame_text = key.rsplit('/Frame_', 1)
        frame = int(frame_text)
        views = {}
        for iid in fs['frames']:
            if int(iid) not in images:
                raise RuntimeError(f'{split}/{key}: unknown image id {iid}')
            parts = images[int(iid)]['file_name'].split('/')
            if len(parts) != 3 or parts[0] != session:
                raise RuntimeError(f'{split}/{key}: bad image path {images[int(iid)]["file_name"]!r}')
            camera = parts[1]
            if camera in views:
                raise RuntimeError(f'{split}/{key}: duplicate camera {camera!r}')
            views[camera] = int(iid)
        framesets[session][frame] = views
    data['_images'] = images
    data['_annotations'] = annotations
    data['_framesets'] = framesets
    return data


def source_cameras(data: dict, session: str) -> list[str]:
    """Return the release's calibration key order for one session."""
    return list(data['calibrations'][session])


def group_runs(frames: list[int], max_gap: int) -> list[list[int]]:
    """Partition into runs with one exact source-frame stride.

    A single ``source_frame_step`` cannot honestly describe a run whose source
    gaps change, so a changed stride starts a new group.  Large gaps also start
    a group.  Singleton groups are retained: dropping or duplicating a source
    frameset would invent data.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    step: int | None = None
    for frame in frames:
        if not current:
            current = [frame]
            continue
        diff = frame - current[-1]
        if diff <= 0 or diff > max_gap or (step is not None and diff != step):
            groups.append(current)
            current = [frame]
            step = None
        else:
            current.append(frame)
            step = diff
    if current:
        groups.append(current)
    return groups


def run_step(run: list[int]) -> int:
    """Return the exact stride, or 1 for a singleton."""
    if len(run) < 2:
        return 1
    diffs = np.diff(run)
    if not np.all(diffs == diffs[0]):
        raise RuntimeError(f'non-constant group stride: {run[:4]}...')
    return int(diffs[0])


# telecentric -> pinhole calibration


def affine_support(data_by_split: dict[str, dict], session: str, cameras: list[str],
                   projections: dict[str, np.ndarray], K: int) -> np.ndarray:
    """Triangulate representative 3D support points directly from affine DLTs."""
    support = []
    for data in data_by_split.values():
        annotations = data['_annotations']
        for frame_views in data['_framesets'].get(session, {}).values():
            for k in range(K):
                rows, values = [], []
                for camera in cameras:
                    ann = annotations.get(frame_views.get(camera))
                    if ann is None:
                        continue
                    kp = np.asarray(ann['keypoints'], dtype=np.float64).reshape(K, 3)
                    x, y, visibility = kp[k]
                    if visibility <= 0 or not np.isfinite([x, y]).all():
                        continue
                    P = projections[camera]
                    rows.extend((P[0, :3], P[1, :3]))
                    values.extend((x - P[0, 3], y - P[1, 3]))
                if len(rows) >= 6:
                    xyz = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)[0]
                    if np.isfinite(xyz).all():
                        support.append(xyz)
    X = np.asarray(support, dtype=np.float64).reshape(-1, 3)
    if not len(X):
        raise RuntimeError(f'{session}: no 3D support points can be affine-triangulated')
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad < 1e-12, 1.0, mad)
    keep = ((X >= med - MAD_K * mad) & (X <= med + MAD_K * mad)).all(axis=1)
    dropped = int((~keep).sum())
    X = X[keep]
    print(f'   {session}: calibration support {len(X)} points ({dropped} MAD outliers dropped), '
          f'range={X.min(axis=0).round(3).tolist()}..{X.max(axis=0).round(3).tolist()}')
    return X


def orient_direction(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Orient the telecentric viewing direction using the support data."""
    direction = np.cross(A[0], A[1])
    direction /= np.linalg.norm(direction)
    if np.median(X @ direction) < 0:
        direction = -direction
    return direction


def build_pinhole_rig(src: Path, data: dict, session: str, cameras: list[str],
                      sizes: dict[str, tuple[int, int]], support: np.ndarray) -> fmt.Rig:
    """Build the reference telephoto-pinhole approximation for one session."""
    from aniposelib.cameras import CameraGroup
    import cv2

    raw = []
    for camera in cameras:
        P = parse_projection(src / data['calibrations'][session][camera])
        A = P[:2, :3]
        raw.append({'name': camera, 'P': P, 'A': A, 't_proj': P[:2, 3]})
    for cam in raw:
        cam['pd'] = orient_direction(cam['A'], support)

    d_pinhole, d_positive, extents = [], [], []
    for cam in raw:
        A, pd = cam['A'], cam['pd']
        A_pinv = A.T @ np.linalg.inv(A @ A.T)
        origins = (A_pinv @ (A @ support.T)).T
        depth = support @ pd
        sigma = np.linalg.svd(A, compute_uv=False)[0]
        d_pinhole.append((2.0 / EPS_PIXEL) * sigma *
                         float(np.max(np.linalg.norm(origins, axis=1) * np.abs(depth))))
        d_positive.append(float(-depth.min()))
        extents.append(float(depth.max() - depth.min()))
    D = max(max(d_pinhole), max(d_positive) + 0.1 * max(extents))
    if not np.isfinite(D) or D <= 0:
        raise RuntimeError(f'{session}: invalid pinhole distance D={D}')

    camera_dicts = []
    worst_reference = 0.0
    branches = []
    worst_actual = 0.0
    for cam in raw:
        A, pd = cam['A'], cam['pd']
        uy = A[1] / np.linalg.norm(A[1])
        ux_proper = np.cross(uy, pd)
        if float(A[0] @ ux_proper) >= 0:
            uz, ux, center, branch = pd, ux_proper, -D * pd, '+pd'
        else:
            uz, ux, center, branch = -pd, -ux_proper, D * pd, '-pd'
        R = np.stack([ux, uy, uz])
        t = -R @ center
        fx = D * float(A[0] @ ux)
        fy = D * float(np.linalg.norm(A[1]))
        skew = D * float(A[0] @ uy)
        cx, cy = cam['t_proj']
        K = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        Xh = np.concatenate([support, np.ones((len(support), 1))], axis=1)
        pcam = (Xh @ np.block([[R, t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).T)[:, :3]
        pp = (pcam @ K.T)[:, :2] / pcam[:, 2, None]
        affine = support @ A.T + cam['t_proj']
        reference_error = np.linalg.norm(pp - affine, axis=1)
        worst_reference = max(worst_reference, float(reference_error.max()))

        if np.linalg.norm(R @ R.T - np.eye(3)) >= 1e-5 or np.linalg.det(R) <= 0.999:
            raise RuntimeError(f'{session}/{cam["name"]}: pinhole rotation is not proper SE(3)')
        if fx <= 0 or fy <= 0 or np.min(pcam[:, 2]) <= 0:
            raise RuntimeError(f'{session}/{cam["name"]}: invalid pinhole depth or focal length')
        if reference_error.max() >= EPS_PIXEL:
            raise RuntimeError(f'{session}/{cam["name"]}: reference approximation exceeds '
                               f'{EPS_PIXEL}px ({reference_error.max():.4f}px)')

        camera_dicts.append({
            'name': cam['name'], 'size': list(sizes[cam['name']]),
            'matrix': K.tolist(), 'distortions': [0.0] * 5,
            'rotation': cv2.Rodrigues(R)[0].ravel().tolist(),
            'translation': t.tolist(),
        })
        branches.append(branch)

    rig = fmt.Rig(
        CameraGroup.from_dicts(camera_dicts),
        offset={camera: (0.0, 0.0) for camera in cameras},
        moving={camera: False for camera in cameras},
        calibrated={camera: True for camera in cameras},
    )
    # This is the actual production projection path, unlike aniposelib's
    # Camera.project(), which drops K[0,1].  It retains the full skew.
    import torch
    from posetail.posetail.cube import project_points_torch
    projected = project_points_torch(rig.posetail(), torch.as_tensor(support, dtype=torch.float64))
    projected = projected.detach().cpu().numpy()
    worst_actual = 0.0
    for ci, cam in enumerate(raw):
        affine = support @ cam['A'].T + cam['t_proj']
        worst_actual = max(worst_actual, float(np.linalg.norm(projected[ci] - affine, axis=1).max()))
    if worst_reference >= EPS_PIXEL or worst_actual >= EPS_PIXEL:
        raise RuntimeError(f'{session}: telecentric->pinhole projection exceeds '
                           f'{EPS_PIXEL}px (reference={worst_reference:.4f}, '
                           f'consumer={worst_actual:.4f})')
    print(f'   {session}: telecentric->pinhole D={D:.6g}, reference max={worst_reference:.4f}px, '
          f'consumer max={worst_actual:.4f}px, branches={branches.count("-pd")} -pd')
    return rig


# labels


def _triangulate_skew_aware(rig: fmt.Rig, p2d: np.ndarray) -> np.ndarray:
    """Triangulate zero-distortion pixels with full-K normalization and posetail's DLT."""
    import torch
    from posetail.posetail.cube import triangulate_simple_batch

    C, N, _ = p2d.shape
    normalized = np.zeros_like(p2d, dtype=np.float64)
    valid = np.isfinite(p2d).all(axis=-1)
    extrinsics = []
    for ci, cam in enumerate(rig.cameras):
        distortion = cam.dist.detach().cpu().numpy() if torch.is_tensor(cam.dist) else cam.dist
        if np.any(np.asarray(distortion, dtype=np.float64) != 0.0):
            raise ValueError(f'{cam.get_name()}: skew-aware DLT requires zero distortion')
        K = cam.matrix.detach().cpu().numpy().astype(np.float64)
        ext = cam.get_extrinsics_mat().detach().cpu().numpy().astype(np.float64)
        extrinsics.append(ext)
        sensor = p2d[ci] + np.asarray(rig.offset[cam.get_name()], dtype=np.float64)
        homogeneous = np.concatenate([sensor, np.ones((N, 1))], axis=1)
        ray = homogeneous @ np.linalg.inv(K).T
        normalized[ci] = ray[:, :2] / ray[:, 2, None]
    normalized[~valid] = 0.0
    weights = torch.as_tensor(valid.astype(np.float64))
    result = triangulate_simple_batch(
        torch.as_tensor(normalized, dtype=torch.float64),
        torch.as_tensor(np.stack(extrinsics), dtype=torch.float64), weights)
    p3 = result.detach().cpu().numpy().astype(np.float64)
    p3[valid.sum(axis=0) < 2] = np.nan
    p3[~np.isfinite(p3).all(axis=1)] = np.nan
    return p3


def _reprojection_error(rig: fmt.Rig, p3d: np.ndarray, p2d: np.ndarray) -> np.ndarray:
    """Pixel residual through posetail's full-matrix projection path."""
    import torch
    from posetail.posetail.cube import project_points_torch

    finite = np.isfinite(p3d).all(axis=1)
    projected = np.full_like(p2d, np.nan, dtype=np.float64)
    if finite.any():
        pred = project_points_torch(rig.posetail(), torch.as_tensor(p3d[finite], dtype=torch.float64))
        projected[:, finite] = pred.detach().cpu().numpy()
    return projected - p2d


def triangulate_robust(rig: fmt.Rig, p2d: np.ndarray, reject_px: float,
                       iters: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Skew-aware DLT with gross-view rejection, preserving source 2D labels."""
    p2 = p2d.copy()
    p3 = _triangulate_skew_aware(rig, p2)
    rejected = np.zeros(p2.shape[:2], dtype=bool)
    for _ in range(iters):
        error = np.linalg.norm(_reprojection_error(rig, p3, p2), axis=-1)
        finite_error = np.isfinite(error)
        med = np.full(error.shape[1], np.inf, dtype=np.float64)
        for k in range(error.shape[1]):
            if finite_error[:, k].any():
                med[k] = np.median(error[finite_error[:, k], k])
        bad = finite_error & (error > np.maximum(reject_px, 5.0 * med))
        if not bad.any():
            break
        rejected |= bad
        p2[bad] = np.nan
        p3 = _triangulate_skew_aware(rig, p2)
    return p3, rejected


def build_labels(data: dict, run: list[int], views: dict[int, dict[str, int]], rig: fmt.Rig,
                 names: list[str], reject_px: float) -> tuple[fmt.Labels, int]:
    """Build 2D, boxes, instances and derived 3D for one group."""
    K, C, T = len(names), len(rig), len(run)
    labels = fmt.empty_labels(1, T, K, C, mode3d=True, animal_ids=['a00'])
    labels.boxes = np.full((1, T, C, 4), np.nan, np.float32)
    labels.instance = np.full((1, T, C), fmt.INST_NONE, np.int8)
    for t, frame in enumerate(run):
        for ci, camera in enumerate(rig.names):
            iid = views[frame].get(camera)
            ann = data['_annotations'].get(iid)
            if ann is None:
                continue
            kp = np.asarray(ann['keypoints'], dtype=np.float64).reshape(K, 3)
            placed = (kp[:, 2] > 0) & np.isfinite(kp[:, :2]).all(axis=1)
            labels.vis2d[0, t, placed, ci] = fmt.PROJECTED
            labels.points2d[0, t, placed, ci] = kp[placed, :2].astype(np.float32)
            x, y, w, h = (float(v) for v in ann['bbox'])
            labels.boxes[0, t, ci] = (x, y, x + w, y + h)
            labels.instance[0, t, ci] = fmt.INST_LABELED

    p2d = np.moveaxis(labels.points2d[0], 2, 0).reshape(C, T * K, 2).astype(np.float64)
    p3d, rejected = triangulate_robust(rig, p2d, reject_px)
    p3d = p3d.reshape(T, K, 3)
    good = np.isfinite(p3d).all(axis=-1)
    labels.vis3d[0][good] = fmt.VISIBLE
    labels.points3d[0][good] = p3d[good].astype(np.float32)
    # Rejection is only for the derived 3D fit.  The source 2D placements are
    # human annotations and must remain intact in keypoints.pq/instances.pq.
    return labels, int(rejected.sum())


# conversion


def convert(src: Path, out: Path, max_gap: int, reject_px: float,
            only: list[str] | None, max_groups: int | None, dry_run: bool) -> None:
    """Convert all Fly50 train/val sessions."""
    by_split = {split: read_split(src, split) for split in SPLITS}
    # Parse every calibration before creating any destination files.  This makes
    # malformed integer/decimal/scientific YAML payloads fail preflight rather
    # than leaving a half-written root.
    calibration_paths = set()
    for data in by_split.values():
        for session, cameras in data['calibrations'].items():
            for camera, relpath in cameras.items():
                path = src / relpath
                parse_projection(path)
                calibration_paths.add(path)
    print(f'   calibration preflight: {len(calibration_paths)} files')
    train = by_split['train']
    names = list(train['keypoint_names'])
    if any(list(data['keypoint_names']) != names for data in by_split.values()):
        raise RuntimeError('train and val keypoint_names/order disagree')
    skeleton = [[edge['keypointA'], edge['keypointB']] for edge in train['skeleton']]
    name_set = set(names)
    flip_pairs = []
    for left in names:
        if left.endswith('L'):
            right = left[:-1] + 'R'
            if right in name_set:
                flip_pairs.append([left, right])
        elif left.startswith(('WingL_', 'T1L_', 'T2L_', 'T3L_')):
            right = left.replace('L_', 'R_', 1)
            if right in name_set:
                flip_pairs.append([left, right])

    sessions = sorted(set().union(*(data['_framesets'] for data in by_split.values())))
    for session in sessions:
        if only and session not in only:
            continue
        if session not in train['calibrations']:
            raise RuntimeError(f'{session}: missing train calibration')
        cameras = source_cameras(train, session)
        projections = {camera: parse_projection(src / train['calibrations'][session][camera])
                       for camera in cameras}
        data0 = next(data for data in by_split.values() if session in data['_framesets'])
        frames0 = next(iter(data0['_framesets'][session].values()))
        sizes = {camera: (int(data0['_images'][frames0[camera]]['width']),
                          int(data0['_images'][frames0[camera]]['height']))
                 for camera in cameras}
        for data in by_split.values():
            if session not in data['_framesets']:
                continue
            for frame, views in data['_framesets'][session].items():
                if set(views) != set(cameras):
                    raise RuntimeError(f'{session}/{frame}: camera set differs from calibration')
        support = affine_support(by_split, session, cameras, projections, len(names))
        rig = build_pinhole_rig(src, train, session, cameras, sizes, support)

        for split, data in by_split.items():
            if session not in data['_framesets']:
                continue
            framesets = data['_framesets'][session]
            clips = group_runs(sorted(framesets), max_gap)
            if max_groups is not None:
                clips = clips[:max_groups]
            groups, labels = {}, {}
            rejected_total = 0
            for run in clips:
                gid = f'{run[0]:06d}'
                groups[gid] = fmt.Group(
                    gid, len(run), fps=float('nan'),
                    source_video=str(src / split / session),
                    source_frame_start=run[0], source_frame_step=run_step(run),
                    notes='telecentric source; exact source-frame stride',
                )
                labels[gid], n_rejected = build_labels(
                    data, run, framesets, rig, names, reject_px)
                rejected_total += n_rejected
                if not dry_run:
                    for camera in cameras:
                        cdir = out / split / session / 'groups' / gid / camera
                        cdir.mkdir(parents=True, exist_ok=True)
                        for t, frame in enumerate(run):
                            image = data['_images'][framesets[frame][camera]]['file_name']
                            fmt.link(cdir / f'{t:06d}.jpg', (src / split / image).resolve())
            print(f'   {split}/{session}: {len(groups)} group(s), '
                  f'{sum(g.n_frames for g in groups.values())} frames, {len(cameras)} cams, '
                  f'{rejected_total} derived outlier observations rejected')
            if dry_run:
                continue
            fmt.write_session(
                out / split / session, mode='3d', units='mm', label_source='annotated',
                names=names, rig=rig, groups=groups, labels=labels,
                skeleton=skeleton, flip_pairs=flip_pairs,
                assoc_res_max_px=30.0,
                provenance={
                    'source': f'Fly50_V7/{split}/{session}',
                    'annotator': '', 'annotator_tool': 'FlyPose JARVIS release',
                    'points3d_source': 'telecentric DLT-derived pinhole triangulation; '
                                       'not native 3D',
                    'animal_id_source': 'single tethered fly per frameset',
                    'calibration_source': 'FlyPose projectionMatrix, telecentric-to-pinhole '
                                          'construction from posetail-preprocessing/preprocess.md',
                })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--max-gap', type=int, default=8)
    ap.add_argument('--sessions', nargs='+', default=None)
    ap.add_argument('--max-groups', type=int, default=None)
    ap.add_argument('--reject-px', type=float, default=20.0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()
    if args.clean and args.out.exists() and not args.dry_run:
        shutil.rmtree(args.out)
    convert(args.src, args.out, args.max_gap, args.reject_px, args.sessions,
            args.max_groups, args.dry_run)
    if args.dry_run:
        return
    ds = fmt.load_dataset(args.out)
    errors = fmt.validate_dataset(ds, check_images=True)
    for error in errors:
        print(('WARN ' if 'WARNING' in error else 'FAIL ') + error)
    hard = [error for error in errors if 'WARNING' not in error]
    print(f'validate: {len(hard)} error(s), {len(errors) - len(hard)} warning(s)')
    sys.exit(1 if hard else 0)


if __name__ == '__main__':
    main()
