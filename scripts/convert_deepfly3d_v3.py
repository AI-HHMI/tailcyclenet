#!/usr/bin/env python3
"""Convert native ``deeperfly`` HDF5 results into tailcycle 3-D sessions.

The native result is kept as the authority for both the reconstructed points and the
bundle-adjusted camera rig.  Native coordinates have no metric scale, so each trial is
scaled using the mean length of the two front coxa segments (lf 0--1 and rf 19--20),
whose reference length is 0.456 mm.  This is a mechanical conversion, not a calibration
validation or an accuracy claim; the audit is recorded in the manifests and provenance.

Typical use::

    pixi run python scripts/convert_deepfly3d_v3.py inventory \\
        --source /path/ramdya-fly --outputs /path/deeperfly-results \\
        --report deepfly3d-v3-inventory.json
    pixi run python scripts/convert_deepfly3d_v3.py convert \\
        --source /path/ramdya-fly --outputs /path/deeperfly-results --out deepfly3d-v3

``--outputs`` may be a result root containing one ``results.h5`` below each trial, or
one HDF5 file when converting a single trial.  Source images are never copied.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tailcyclenet import format as fmt

# Import the family convention from v1.  The v1 module is deliberately used only for the
# condition names and split rule; v3 has its own HDF5 and pixel path handling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_deepfly3d_v1 import SOURCE_CONDITIONS, family_id, split_map  # noqa: E402

N_CAMERAS = 7
N_KEYPOINTS = 38
REFERENCE_COXA_MM = 0.456
FRONT_COXA_PAIRS = ((0, 1), (19, 20))
POLICY_VERSION = 'deepfly3d-v3-native-h5-scale-front-coxa-v1'
CAMERA_ORDERING = ('rh', 'rm', 'rf', 'f', 'lf', 'lm', 'lh')
# This is the native deeperfly fly38 order (the HDF5 skeleton is checked against it when
# present).  Keeping this fallback makes a small, older HDF5 fixture readable while never
# inferring a coordinate axis from alphabetic sorting.
NATIVE_NAMES = [
    'lf_thorax_coxa', 'lf_coxa_trochanter', 'lf_femur_tibia',
    'lf_tibia_tarsus', 'lf_claw',
    'lm_thorax_coxa', 'lm_coxa_trochanter', 'lm_femur_tibia',
    'lm_tibia_tarsus', 'lm_claw',
    'lh_thorax_coxa', 'lh_coxa_trochanter', 'lh_femur_tibia',
    'lh_tibia_tarsus', 'lh_claw',
    'l_antenna', 'l_abdomen0', 'l_abdomen1', 'l_abdomen2',
    'rf_thorax_coxa', 'rf_coxa_trochanter', 'rf_femur_tibia',
    'rf_tibia_tarsus', 'rf_claw',
    'rm_thorax_coxa', 'rm_coxa_trochanter', 'rm_femur_tibia',
    'rm_tibia_tarsus', 'rm_claw',
    'rh_thorax_coxa', 'rh_coxa_trochanter', 'rh_femur_tibia',
    'rh_tibia_tarsus', 'rh_claw',
    'r_antenna', 'r_abdomen0', 'r_abdomen1', 'r_abdomen2',
]


@dataclass(frozen=True)
class Record:
    """One extracted source trial and its native result."""

    condition: str
    archive: str
    images: Path
    result: Path | None
    family: str
    reason: str | None = None

    @property
    def session_id(self) -> str:
        """Collision-proof target session id."""
        return f'{self.condition}__{self.archive}'


@dataclass
class NativeResult:
    """Materialized arrays and calibration from one native HDF5 result."""

    points2d: np.ndarray  # (C,T,K,2), stored-image x,y pixels
    confidence: np.ndarray  # (C,T,K)
    points3d: np.ndarray  # (T,K,3), native units
    camera_names: list[str]
    intrinsics: np.ndarray  # (C,4), fx,fy,cx,cy (or supplied 3x3 converted)
    distortions: list[np.ndarray]
    rvecs: np.ndarray  # (C,3), world -> camera Rodrigues
    tvecs: np.ndarray  # (C,3), world -> camera native units
    names: list[str]
    bones: list[tuple[int, int]]
    image_sizes: list[tuple[int, int]]
    source_camera_names: list[str]
    points2d_source: str = 'pose2d/points'

    @property
    def n_frames(self) -> int:
        """Number of frames represented by the native view-leading arrays."""
        return int(self.points2d.shape[1])


# ----------------------------- small, deterministic helpers -----------------------------


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Hash one file without loading it all into memory."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _scalar(value: Any) -> Any:
    """Convert an HDF5 scalar/bytes value to a JSON-friendly Python value."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.generic):
        return _scalar(value.item())
    return value


def _strings(value: np.ndarray | list[Any]) -> list[str]:
    """Decode a one-dimensional HDF5 string dataset."""
    return [str(_scalar(v)) for v in np.asarray(value).reshape(-1)]


def _read(f: h5py.File, name: str, *, required: bool = True):
    """Read one HDF5 dataset, with a useful schema error."""
    if name not in f:
        if required:
            raise ValueError(f'missing HDF5 dataset {name!r}')
        return None
    return f[name][()]


def _archive_for(record: Record) -> Path | None:
    """Find the source ZIP alongside an extracted trial, if one exists."""
    # Extraction uses the ZIP stem as the directory name.  Do not require it: a user may
    # intentionally retain only extracted images.
    root = record.images.parent.parent
    candidates = [root / f'{record.archive}.zip', root.parent / f'{record.archive}.zip']
    return next((p for p in candidates if p.is_file()), None)


def _candidate_result_paths(outputs: Path) -> list[Path]:
    """List native result files below a result root, in deterministic order."""
    if outputs.is_file():
        return [outputs] if outputs.name == 'results.h5' else []
    if not outputs.is_dir():
        return []
    return sorted(p for p in outputs.rglob('results.h5') if p.is_file())


def _result_matches(result: Path, record: Record) -> bool:
    """Match an HDF5 result to a source trial by path, then native footage metadata."""
    names = {record.archive, record.archive.removesuffix('_behData_images'),
             record.archive.removesuffix('_images')}
    parts = set(result.parts)
    if result.parent.name in names and record.condition in parts:
        return True
    if result.parent.name in names:
        return True
    # Native deeperfly stores the absolute image paths as pose2d/footage metadata.  Reading one
    # small attribute is safer than guessing when output names were changed by a batch runner.
    try:
        with h5py.File(result, 'r') as f:
            raw = f.get('pose2d')
            raw = raw.attrs.get('footage', '') if raw is not None else ''
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            return str(record.images) in str(raw) or record.archive in str(raw)
    except (OSError, KeyError, ValueError, TypeError, EOFError):
        return False


def _find_result(outputs: Path, record: Record, paths: list[Path]) -> tuple[Path | None, str | None]:
    """Resolve exactly one native result for a trial without quadratic HDF5 metadata reads."""
    if outputs.is_file() and len(paths) == 1:
        return paths[0], None
    names = {record.archive, record.archive.removesuffix('_behData_images'),
             record.archive.removesuffix('_images')}
    # Production workers preserve the source session as the results directory name.  Resolve
    # that cheap path key first; the old metadata fallback is only needed for renamed outputs.
    matched = [p for p in paths if p.parent.name in names and record.condition in p.parts]
    if not matched:
        matched = [p for p in paths if p.parent.name in names]
    if not matched:
        matched = [p for p in paths if _result_matches(p, record)]
    if not matched:
        return None, 'missing_result'
    if len(matched) != 1:
        return None, 'ambiguous_result:' + ','.join(str(p) for p in matched[:3])
    return matched[0], None


def camera_files(images: Path, camera: int, n_frames: int | None = None) -> list[Path]:
    """Return source camera files and enforce the native contiguous naming convention."""
    files = sorted(images.glob(f'camera_{camera}_img_*.jpg'))
    if not files:
        files = sorted(images.glob(f'camera_{camera}_img_*.png'))
    if n_frames is None:
        n_frames = len(files)
    if len(files) < n_frames:
        raise ValueError(f'{images}: camera {camera} has {len(files)} frames, '
                         f'needs at least {n_frames}')
    expected = [images / f'camera_{camera}_img_{i:06d}{files[0].suffix}'
                for i in range(n_frames)] if files else []
    # Native H5 pilots may contain a prefix of a longer extracted trial.  The result frame
    # count defines the converted interval; require that prefix to be contiguous and return
    # only it.  Full production results still exercise the exact 000000..000899 path.
    if files[:n_frames] != expected:
        raise ValueError(f'{images}: camera {camera} is not contiguous 000000..{n_frames - 1:06d}')
    return files[:n_frames]


def discover(source: Path, outputs: Path) -> list[Record]:
    """Discover extracted source trials and associate native ``results.h5`` files."""
    source, outputs = Path(source), Path(outputs)
    if not source.is_dir():
        raise SystemExit(f'missing source directory: {source}')
    # Keep v1's four-condition order where present, while allowing a fixture with only a subset.
    known_conditions = [c for c in SOURCE_CONDITIONS if (source / c / 'extracted').is_dir()]
    extra_conditions = sorted(p.name for p in source.iterdir()
                              if p.is_dir() and (p / 'extracted').is_dir()
                              and p.name not in SOURCE_CONDITIONS)
    conditions = known_conditions + extra_conditions
    if not conditions:
        raise SystemExit(f'{source}: no condition/extracted directories')
    candidates: list[tuple[str, str, Path, str]] = []
    for condition in conditions:
        extracted = source / condition / 'extracted'
        for d in sorted(extracted.iterdir()):
            if d.is_dir() and d.name.endswith('_behData_images'):
                try:
                    fam = family_id(condition, d.name)
                    reason = None
                except ValueError as exc:
                    # Keep malformed archives visible in inventory and assign a deterministic
                    # quarantine-only family so split_map can still build a complete manifest.
                    fam = f'{condition}__invalid__{d.name}'
                    reason = 'invalid_family_id:' + str(exc)
                candidates.append((condition, d.name, d / 'images', fam, reason))
    paths = _candidate_result_paths(outputs)
    records: list[Record] = []
    # A direct HDF5 is unambiguous only for a one-trial source.  Assigning it to every
    # extracted trial would silently duplicate one experiment (and was an easy CLI footgun).
    direct = outputs.is_file()
    for condition, archive, images, fam, reason in candidates:
        if reason is None and not images.is_dir():
            reason = 'missing_images'
        result = None
        if reason is None:
            if direct:
                if len(candidates) != 1:
                    reason = 'direct_result_requires_one_source_trial'
                else:
                    result = paths[0] if paths else None
                    reason = None if result else 'missing_result'
            else:
                result, reason = _find_result(
                    outputs, Record(condition, archive, images, None, fam), paths)
        records.append(Record(condition, archive, images, result, fam, reason))
    return records


def _shape_error(name: str, value: Any, expected: str) -> ValueError:
    """Construct a concise shape error for inventory and CLI output."""
    return ValueError(f'{name} has shape {np.asarray(value).shape}, expected {expected}')


def _camera_names(f: h5py.File, group: str) -> list[str]:
    """Read native camera names and normalize the two known HDF5 spellings."""
    names = _read(f, f'{group}/names')
    out = _strings(names)
    if len(out) != N_CAMERAS or len(set(out)) != N_CAMERAS:
        raise ValueError(f'{group}/names must contain seven unique camera names, got {out!r}')
    if out != list(CAMERA_ORDERING):
        raise ValueError(f'{group}/names {out!r} != native camera order {list(CAMERA_ORDERING)!r}')
    return out


def _intrinsics(raw: np.ndarray, n: int, where: str) -> np.ndarray:
    """Normalize native [fx,fy,cx,cy] or 3x3 intrinsics to compact rows."""
    a = np.asarray(raw, dtype=np.float64)
    if a.shape == (n, 4):
        out = a
    elif a.shape == (n, 3, 3):
        if not np.isfinite(a).all():
            raise ValueError(f'{where}/intrs contains non-finite values')
        out = np.stack((a[:, 0, 0], a[:, 1, 1], a[:, 0, 2], a[:, 1, 2]), axis=1)
    else:
        raise _shape_error(f'{where}/intrs', a, f'({n},4) or ({n},3,3)')
    if not np.isfinite(out).all() or (out[:, :2] <= 0).any():
        raise ValueError(f'{where}/intrs has invalid focal lengths or non-finite values')
    return out


def _distortions(raw: np.ndarray, n: int, where: str) -> list[np.ndarray]:
    """Normalize native per-camera distortion rows."""
    a = np.asarray(raw, dtype=np.float64)
    if a.ndim == 1 and n == 1:
        a = a[None]
    if a.ndim != 2 or a.shape[0] != n:
        raise _shape_error(f'{where}/dists', a, f'({n},D)')
    return [np.asarray(row, dtype=np.float64).ravel() for row in a]


def load_result(path: Path, *, source_images: Path | None = None) -> NativeResult:
    """Load and schema-check one native deeperfly HDF5 file.

    Native ``pose2d/points`` are already stored-image pixels in ``(x,y)`` order.  The
    compact camera rows are ``(fx, fy, cx, cy)`` and the native BA vectors are Rodrigues
    world-to-camera vectors plus additive translations.
    """
    path = Path(path)
    try:
        with h5py.File(path, 'r') as f:
            # Native PoseResult chooses the most-derived per-view coordinates.  The
            # triangulation stage may have filtered a detector outlier, while confidence
            # remains the pose2d detector's raw peak for that point.
            p2_name = next((name for name in ('triangulation/points',
                                               'pictorial_structures/points',
                                               'pose2d/points') if name in f), None)
            if p2_name is None:
                raise ValueError('missing HDF5 per-view points (triangulation, pictorial_structures, or pose2d)')
            p2 = np.asarray(_read(f, p2_name), dtype=np.float64)
            conf = np.asarray(_read(f, 'pose2d/conf'), dtype=np.float64)
            p3 = np.asarray(_read(f, 'triangulation/points3d'), dtype=np.float64)
            if p2.ndim != 4 or p2.shape[0] != N_CAMERAS or p2.shape[2:] != (N_KEYPOINTS, 2):
                raise _shape_error('pose2d/points', p2,
                                   f'({N_CAMERAS},T,{N_KEYPOINTS},2)')
            t = p2.shape[1]
            if t < 1:
                raise ValueError('pose2d/points has no frames')
            if conf.shape == (N_CAMERAS, t, N_KEYPOINTS, 1):
                conf = conf[..., 0]
            if conf.shape != (N_CAMERAS, t, N_KEYPOINTS):
                raise _shape_error('pose2d/conf', conf,
                                   f'({N_CAMERAS},{t},{N_KEYPOINTS})')
            if p3.shape != (t, N_KEYPOINTS, 3):
                raise _shape_error('triangulation/points3d', p3,
                                   f'({t},{N_KEYPOINTS},3)')

            ba = 'bundle_adjustment/cameras'
            cams = _camera_names(f, ba)
            # Pose arrays are view-leading too.  If the native file carries the pose camera
            # table, require its order to agree with BA rather than silently pairing points
            # from one view with calibration from another.
            if 'pose2d/cameras' in f and _camera_names(f, 'pose2d/cameras') != cams:
                raise ValueError('pose2d/cameras order differs from bundle_adjustment/cameras')
            intr = _intrinsics(_read(f, f'{ba}/intrs'), N_CAMERAS, ba)
            dist = _distortions(_read(f, f'{ba}/dists'), N_CAMERAS, ba)
            rv = np.asarray(_read(f, f'{ba}/rvecs'), dtype=np.float64)
            tv = np.asarray(_read(f, f'{ba}/tvecs'), dtype=np.float64)
            if rv.shape == (N_CAMERAS, 3, 3):
                rv = np.stack([cv2.Rodrigues(r)[0].ravel() for r in rv])
            if rv.shape != (N_CAMERAS, 3):
                raise _shape_error(f'{ba}/rvecs', rv, f'({N_CAMERAS},3)')
            if tv.shape != (N_CAMERAS, 3):
                raise _shape_error(f'{ba}/tvecs', tv, f'({N_CAMERAS},3)')
            if not np.isfinite(rv).all() or not np.isfinite(tv).all():
                raise ValueError(f'{ba}: rvecs/tvecs contain non-finite values')

            names_raw = _read(f, 'skeleton/point_names', required=False)
            names = _strings(names_raw) if names_raw is not None else list(NATIVE_NAMES)
            if len(names) != N_KEYPOINTS or len(set(names)) != N_KEYPOINTS:
                raise ValueError(f'skeleton/point_names must contain 38 unique names, got {names!r}')
            if names != NATIVE_NAMES:
                raise ValueError('skeleton/point_names does not match the native fly38 axis')
            bones_raw = _read(f, 'skeleton/bones', required=False)
            bones: list[tuple[int, int]] = []
            if bones_raw is not None:
                bones_arr = np.asarray(bones_raw)
                if bones_arr.ndim != 2 or bones_arr.shape[1] != 2:
                    raise _shape_error('skeleton/bones', bones_arr, '(B,2)')
                for a, b in bones_arr:
                    a, b = int(a), int(b)
                    if not 0 <= a < N_KEYPOINTS or not 0 <= b < N_KEYPOINTS:
                        raise ValueError(f'skeleton/bones contains index {(a,b)} outside fly38')
                    bones.append((a, b))
            if not bones:
                # The native fly38 skeleton has these chains; do not silently produce a
                # misleading empty skeleton when a minimal result omits the optional group.
                bones = [(i, i + 1) for start in (0, 5, 10, 19, 24, 29)
                          for i in range(start, start + 4)] + [(16, 17), (17, 18), (35, 36), (36, 37)]
            if (15, 34) not in bones and (34, 15) not in bones:
                bones.append((15, 34))

            sizes: list[tuple[int, int]] = []
            attrs = f.get('pose2d')
            raw_sizes = attrs.attrs.get('image_sizes', '') if attrs is not None else ''
            if isinstance(raw_sizes, bytes):
                raw_sizes = raw_sizes.decode('utf-8', errors='replace')
            if raw_sizes:
                try:
                    doc = json.loads(raw_sizes)
                    # Native metadata is {view: [height,width]}.
                    sizes = [(int(doc[name][1]), int(doc[name][0])) for name in cams]
                except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
                    raise ValueError('pose2d image_sizes metadata is malformed') from exc
                if any(w <= 0 or h <= 0 for w, h in sizes):
                    raise ValueError('pose2d image_sizes metadata has a non-positive size')
            if not sizes and source_images is not None:
                from PIL import Image
                for ci in range(N_CAMERAS):
                    files = camera_files(source_images, ci, t)
                    with Image.open(files[0]) as im:
                        sizes.append(tuple(int(v) for v in im.size))
            if not sizes:
                # Native deeperfly's extracted Ramdya images are 960x480.  This fallback is
                # used only for small HDF5 fixtures and is visible in provenance.
                sizes = [(960, 480)] * N_CAMERAS
            if len(sizes) != N_CAMERAS:
                raise ValueError('could not determine seven image sizes')
            source_camera_names = list(cams)
            return NativeResult(p2, conf, p3, cams, intr, dist, rv, tv, names, bones,
                                sizes, source_camera_names, p2_name)
    except OSError as exc:
        raise ValueError(f'{path}: cannot read HDF5 ({exc})') from exc


def valid_2d(result: NativeResult) -> np.ndarray:
    """Return C,T,K mask for finite native points.

    Native confidence is metadata, not a visibility threshold: a finite point is a
    prediction even when its confidence is NaN or outside [0, 1].
    """
    return np.isfinite(result.points2d).all(axis=-1)


def scale_from_front_coxa(points3d: np.ndarray) -> tuple[float, dict[str, float]]:
    """Compute the per-trial metric scale from only the two front coxa segments."""
    p = np.asarray(points3d, dtype=np.float64)
    values = []
    means = {}
    for a, b in FRONT_COXA_PAIRS:
        d = np.linalg.norm(p[:, a] - p[:, b], axis=-1)
        d = d[np.isfinite(d)]
        if not d.size:
            raise ValueError(f'front coxa segment {(a,b)} has no finite lengths')
        means[f'{a}-{b}'] = float(np.mean(d))
        values.extend(d.tolist())
    native_mean = float(np.mean(values))
    if not np.isfinite(native_mean) or native_mean <= 0:
        raise ValueError(f'invalid front coxa mean length {native_mean!r}')
    scale = REFERENCE_COXA_MM / native_mean
    means['combined'] = native_mean
    means['scale_to_mm'] = float(scale)
    return float(scale), means


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


def _source_files_for(result: NativeResult, images: Path) -> list[list[Path]]:
    """Resolve source sequences and verify representative dimensions against native metadata.

    The extraction/worker checks already establish the complete contiguous 900-frame layout.
    Opening and decoding all 1.24 million JPEGs again on the shared filesystem made inventory
    needlessly take hours, so inspect the first and last frame of each camera here; the generic
    session validator repeats the first-frame calibration-size check after writing.
    """
    from PIL import Image
    out = []
    for c in range(N_CAMERAS):
        files = camera_files(images, c, result.n_frames)
        check_frames = sorted({0, len(files) - 1})
        for frame in check_frames:
            path = files[frame]
            try:
                with Image.open(path) as im:
                    got = tuple(int(v) for v in im.size)
                    im.verify()
            except (OSError, ValueError, SyntaxError) as exc:
                raise ValueError(f'camera {c} frame {frame} is not a readable image: {path}') from exc
            if got != result.image_sizes[c]:
                raise ValueError(f'camera {c} frame {frame} source image is {got}, '
                                 f'HDF5 metadata is {result.image_sizes[c]}')
        out.append(files)
    return out


def link_pixels(view_dir: Path, source_files: list[Path]) -> None:
    """Create a deterministic numbered symlink farm without copying image bytes.

    New farms are empty by construction, so avoid an existence/stat/resolve round trip for every
    frame on the shared filesystem.  The slower collision-checking path remains for idempotent
    reruns or manually resumed sessions.
    """
    fresh = not view_dir.exists()
    view_dir.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(source_files):
        dst = view_dir / f'{i:06d}{source.suffix.lower()}'
        if fresh:
            os.symlink(os.fspath(source), os.fspath(dst))
            continue
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve() != source.resolve():
                raise RuntimeError(f'pixel link collision: {dst}')
            continue
        dst.symlink_to(source)


def _append_scores(path: Path, scores: np.ndarray) -> None:
    """Add native detector confidence to keypoints.pq (the generic Labels API has no score)."""
    table = pq.read_table(path)
    vals = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(table) != len(vals):
        raise RuntimeError(f'{path}: score rows {len(vals)} != keypoint rows {len(table)}')
    table = table.append_column('score', pa.array(vals, type=pa.float32()))
    tmp = path.with_name('.' + path.name + '.score.tmp')
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, path)


def _fsync_tree(path: Path) -> None:
    """Best-effort fsync for a completed session directory before its rename."""
    for name in ('session.toml', 'calibration.toml', 'groups.pq', 'keypoints.pq',
                 'points3d.pq', 'regions.pq'):
        p = path / name
        if p.exists():
            with p.open('rb') as f:
                os.fsync(f.fileno())
    try:
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


def git_commit() -> str:
    """Return the converter checkout commit, or an explicit unknown marker."""
    try:
        return subprocess.run(('git', 'rev-parse', 'HEAD'), check=True, capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def _source_hashes(record: Record) -> dict[str, Any]:
    """Stable input identity used by idempotency checks without rereading 137 GiB of ZIPs."""
    archive = _archive_for(record)
    names = (sorted(p.name for p in record.images.iterdir() if p.name.startswith('camera_'))
             if record.images.is_dir() else [])
    return {
        'result_sha256': sha256(record.result) if record.result else None,
        'source_archive': str(archive) if archive else None,
        'source_archive_size': int(archive.stat().st_size) if archive else None,
        'source_images': str(record.images),
        'source_image_count': len(names),
        'source_image_names_sha256': hashlib.sha256('\\n'.join(names).encode()).hexdigest(),
    }


def convert_record(record: Record, split: str, root: Path, *, force: bool = False,
                   max_reprojection_px: float | None = None) -> dict:
    """Convert one source trial atomically and validate its resulting session."""
    if record.reason:
        return {'condition': record.condition, 'archive': record.archive, 'split': split,
                'status': 'quarantined', 'reason': record.reason}
    assert record.result is not None
    result = load_result(record.result, source_images=record.images)
    source_files = _source_files_for(result, record.images)
    observed = valid_2d(result)
    support = observed.sum(axis=0)
    scale, scale_meta = scale_from_front_coxa(result.points3d)
    audit_meta = audit(result, max_reprojection_px=max_reprojection_px)
    session_id = record.session_id
    session_dir = root / split / session_id
    if session_dir.exists():
        if not force:
            raise SystemExit(f'output exists (use --force to replace): {session_dir}')
        shutil.move(str(session_dir), str(root / f'.quarantine-{session_id}-{time.time_ns()}'))
    temp = root / split / f'.{session_id}.tmp'
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    farms: list[Path] = []
    try:
        names = result.names
        skeleton, flip_pairs = skeleton_for(result)
        # Native arrays are C,T,K; tailcycle dense arrays are S,T,K,C.
        points2d = np.transpose(result.points2d, (1, 2, 0, 3))[None].astype(np.float32)
        # The native pathway plan is the only per-view visibility information: an output
        # coordinate means the point was predicted in that view; NaN means the pathway did
        # not predict that anatomical point.  Preserve both outcomes as explicit statuses.
        vis2d = np.full((1, result.n_frames, N_KEYPOINTS, N_CAMERAS), fmt.MISSING, np.int8)
        vis2d[0][np.transpose(observed, (1, 2, 0))] = fmt.VISIBLE
        points3d = (result.points3d * scale)[None].astype(np.float32)
        vis3d = np.full((1, result.n_frames, N_KEYPOINTS), fmt.UNLABELED, np.int8)
        valid3 = np.isfinite(result.points3d).all(axis=-1) & (support >= 2)
        vis3d[0][valid3] = fmt.PROJECTED
        # The arrays for failed 3D points stay NaN; write_session emits no row for them.
        points3d[0][~np.isfinite(result.points3d).all(axis=-1)] = np.nan
    
        group = fmt.Group(
            group_id='g000', n_frames=result.n_frames, source_video=f'{record.condition}/{record.archive}',
            source_frame_start=0, source_frame_step=1,
            notes='native deeperfly tracked 3D; scale and calibration are provisional_unvalidated',
        )
        labels = fmt.Labels(
            animal_ids=['fly0'], points3d=points3d, vis3d=vis3d,
            points2d=points2d, vis2d=vis2d, boxes=None, instance=None,
            regions=np.zeros((0, 6), np.float64),
        )
        provenance = {
            'source': 'native deeperfly results.h5',
            'annotator': '',
            'annotator_tool': 'deeperfly native HDF5 result',
        'converter_git_commit': git_commit(),
            'created': date.today().isoformat(),
            'status': 'provisional_unvalidated',
            'condition': record.condition,
            'source_archive': record.archive,
            'source_images': str(record.images),
            'source_result': str(record.result),
            'source_result_sha256': sha256(record.result),
            'source_archive_size': (_archive_for(record).stat().st_size
                                     if _archive_for(record) else None),
            'native_camera_order': ','.join(result.camera_names),
        'calibration_fingerprint': calibration_fingerprint(result, scale),
            'source_camera_policy': 'HDF5 camera index c maps to source camera_c image sequence',
            'point3d_source': 'triangulation/points3d',
            'point2d_source': result.points2d_source + ' (stored x,y pixels)',
            'confidence_source': 'pose2d/conf; copied to keypoints.pq score without thresholding',
            'point2d_status_policy': 'visible iff selected native per-view point is finite; missing iff NaN; confidence never thresholded',
            'point3d_status_policy': 'projected iff finite and at least two finite 2D predictions; no independent 3D visibility assessment',
            'scale_policy': '0.456 mm divided by mean native lengths of front coxa segments 0-1 and 19-20',
            'scale_reference_mm': REFERENCE_COXA_MM,
            'scale_native_front_coxa_0_1': scale_meta['0-1'],
            'scale_native_front_coxa_19_20': scale_meta['19-20'],
            'scale_native_mean': scale_meta['combined'],
            'scale_to_mm': scale,
            'audit_status': 'provisional_unvalidated',
            'audit_min_2d_support': audit_meta['min_2d_support'],
            'audit_overall_median_reprojection_px': audit_meta['overall_median_reprojection_px'],
            'audit_overall_p95_reprojection_px': audit_meta['overall_p95_reprojection_px'],
            'audit_overall_max_reprojection_px': audit_meta['overall_max_reprojection_px'],
            'audit_positive_depth_fraction': audit_meta['overall_positive_depth_fraction'],
            'frame_policy': 'all native results.h5 frames; no implicit trim',
        }
        rig = calibration_rig(result, scale)
        fmt.write_session(temp, mode='3d', units='mm', label_source='tracked', names=names,
                          rig=rig, groups={'g000': group}, labels={'g000': labels},
                          skeleton=skeleton, flip_pairs=flip_pairs, provenance=provenance)
        # Score ordering is exactly the dense status order: frame, bodypart, camera.
        _append_scores(temp / 'keypoints.pq', result.confidence.transpose(1, 2, 0).reshape(-1))
    
        for c, cname in enumerate(result.camera_names):
            farm = root / '.pixel_views' / f'{session_id}__{cname}'
            farms.append(farm)
            link_pixels(farm, source_files[c])
            target = temp / 'groups' / 'g000' / cname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(farm.resolve())
        _fsync_tree(temp)
        loaded = fmt.Session.load(temp)
        errors = fmt.validate_session(loaded, check_images=True)
        if errors:
            raise RuntimeError('\n'.join(errors[:20]))
        temp.rename(session_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        for farm in farms:
            # A failed validation must not leave a misleading successful pixel farm behind.
            shutil.rmtree(farm, ignore_errors=True)
        raise
    return {
        'condition': record.condition, 'archive': record.archive, 'split': split,
        'session': str(session_dir.relative_to(root)), 'status': 'converted',
        'source_result': str(record.result), 'source_result_sha256': sha256(record.result),
        'source_archive_size': (_archive_for(record).stat().st_size
                                 if _archive_for(record) else None),
        'source_camera_names': result.camera_names,
        'source_frame_counts': [len(files) for files in source_files],
        'pixel_view_dirs': {name: str(root / '.pixel_views' / f'{session_id}__{name}')
                            for name in result.camera_names},
        'calibration_fingerprint': calibration_fingerprint(result, scale),
        'n_frames': result.n_frames,
        'scale_reference_mm': REFERENCE_COXA_MM,
        'scale_native_front_coxa_0_1': scale_meta['0-1'],
        'scale_native_front_coxa_19_20': scale_meta['19-20'],
        'scale_native_mean': scale_meta['combined'],
        'scale_to_mm': scale,
        'keypoint_rows': int(observed.sum()),
        'point3d_rows': int(valid3.sum()),
        'audit': audit_meta,
    }


def inspect_record(record: Record, *, max_reprojection_px: float | None = None) -> dict:
    """Inventory one record without writing a session."""
    out: dict[str, Any] = {
        'condition': record.condition, 'archive': record.archive,
        'family': record.family, 'images': str(record.images),
        'result': str(record.result) if record.result else None,
        'status': 'quarantined' if record.reason else 'ready',
        'reason': record.reason,
    }
    if record.reason:
        return out
    try:
        assert record.result is not None
        result = load_result(record.result, source_images=record.images)
        files = _source_files_for(result, record.images)
        scale, scale_meta = scale_from_front_coxa(result.points3d)
        a = audit(result, max_reprojection_px=max_reprojection_px)
        out.update({
            'status': 'ready', 'result_sha256': sha256(record.result),
            'n_frames': result.n_frames, 'camera_names': result.camera_names,
            'image_sizes': result.image_sizes, 'source_frame_counts': [len(x) for x in files],
            'scale': scale_meta, 'audit': a,
        })
    except (AssertionError, OSError, ValueError, RuntimeError, KeyError, TypeError,
            EOFError, OverflowError) as exc:
        out['status'] = 'quarantined'
        out['reason'] = 'invalid_result:' + str(exc)
    return out


# ---------------------------------- locking / manifests ----------------------------------

@contextlib.contextmanager
def root_lock(root: Path):
    """Serialize conversions targeting one root."""
    import fcntl
    lock = root.parent / f'.{root.name}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict) -> None:
    """Write one JSON manifest and atomically replace its previous version."""
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def orphan_results(records: list[Record], outputs: Path) -> list[str]:
    """Return result files not associated with an extracted trial."""
    known = {str(r.result.resolve()) for r in records if r.result is not None}
    return [str(p) for p in _candidate_result_paths(Path(outputs)) if str(p.resolve()) not in known]


def _policy_signature(records: list[Record], splits: dict[str, str], extras: list[str] = (),
                      max_reprojection_px: float | None = None) -> dict:
    """Stable conversion inputs used to make reruns no-ops or explicit refusals."""
    return {
        'policy_version': POLICY_VERSION,
        'units': 'mm',
        'scale_reference_mm': REFERENCE_COXA_MM,
        'scale_front_coxa_pairs': [list(p) for p in FRONT_COXA_PAIRS],
        'axis': NATIVE_NAMES,
        'splits': splits,
        'records': {f'{r.condition}/{r.archive}': _source_hashes(r) for r in records},
        'orphan_results': list(extras),
        'max_reprojection_px': max_reprojection_px,
    }


def _same_policy(root: Path, policy: dict) -> bool:
    """Return whether a previous root is identical and its sessions still validate."""
    p = root / 'conversion_manifest.json'
    if not p.exists():
        return False
    try:
        old = json.loads(p.read_text())
        if old.get('input_signature') != policy:
            return False
        for entry in old.get('records', []):
            if entry.get('status') != 'converted':
                continue
            rel = entry.get('session')
            if not rel:
                return False
            session = root / rel
            if not (session / 'session.toml').exists():
                return False
            loaded = fmt.Session.load(session)
            if fmt.validate_session(loaded, check_images=False):
                return False
        return True
    except Exception:
        # A damaged prior session is not an idempotent success; the caller will refuse it
        # unless --force explicitly quarantines the root.
        return False


def inventory_command(args) -> int:
    """Run discovery and schema/audit checks, writing a JSON inventory."""
    records = discover(Path(args.source), Path(args.outputs))
    report = [inspect_record(r, max_reprojection_px=args.max_reprojection_px) for r in records]
    extras = orphan_results(records, Path(args.outputs))
    payload = {
        'dataset': 'deepfly3d-v3', 'converter_policy': POLICY_VERSION,
        'source_root': str(args.source), 'outputs_root': str(args.outputs),
        'expected_source_sessions': len(records), 'orphan_results': extras, 'records': report,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, payload)
    bad = [r for r in report if r['status'] != 'ready']
    bad.extend({'status': 'quarantined'} for _ in extras)
    print(f'inventory: {len(records)} trials; {len(bad)} with issues; report={report_path}')
    return 1 if bad and args.strict else 0


def convert_command(args) -> int:
    """Convert all ready trials, quarantining malformed trials in the manifest."""
    source, outputs, root = Path(args.source), Path(args.outputs), Path(args.out)
    records = discover(source, outputs)
    splits = split_map(records)
    extras = orphan_results(records, outputs)
    policy = _policy_signature(records, splits, extras, args.max_reprojection_px)
    root.parent.mkdir(parents=True, exist_ok=True)
    with root_lock(root):
        if root.exists():
            if _same_policy(root, policy):
                print(f'conversion is already complete and identical: {root}')
                return 0
            if any(root.iterdir()) and not args.force:
                raise SystemExit(f'output root is non-empty or differs from previous conversion '
                                 f'(use --force to quarantine it): {root}')
            if args.force and any(root.iterdir()):
                old = root.with_name(f'{root.name}.quarantine-{time.time_ns()}')
                shutil.move(str(root), str(old))
        root.mkdir(parents=True, exist_ok=True)
        for split in fmt.SPLITS:
            split_dir = root / split
            split_dir.mkdir(exist_ok=True)
            for stale in split_dir.glob('.*.tmp'):
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
        for stale in root.glob('.*.tmp'):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        (root / '.pixel_views').mkdir(exist_ok=True)
        _atomic_json(root / 'split_manifest.json', {
            'dataset': 'deepfly3d-v3',
            'policy': 'family grouped; condition + acquisition date + FlyN stays in one split',
            'source_family_split': 'convert_deepfly3d_v1.split_map (v2 family policy)',
            'assignment': splits,
        })
        inventory = []
        for record in records:
            split = splits[record.family]
            try:
                inventory.append(convert_record(record, split, root, force=False,
                                                 max_reprojection_px=args.max_reprojection_px))
            except (AssertionError, OSError, ValueError, RuntimeError, KeyError, TypeError,
                    EOFError, OverflowError) as exc:
                # A malformed trial is quarantined; a session that was partially written is
                # removed by convert_record before raising.  Other trials remain convertible.
                inventory.append({'condition': record.condition, 'archive': record.archive,
                                  'split': split, 'status': 'quarantined',
                                  'reason': 'conversion_failed:' + str(exc),
                                  'source_result': str(record.result) if record.result else None})
        manifest = {
            'dataset': 'deepfly3d-v3', 'format': 'tailcycle-dataset; docs/annotation_format.md',
            'mode': '3d', 'units': 'mm', 'labels': 'tracked',
            'status': 'provisional_unvalidated', 'converter_policy': POLICY_VERSION,
            'source_root': str(source), 'outputs_root': str(outputs),
            'created': date.today().isoformat(), 'converter_git_commit': git_commit(),
            'keypoint_names': NATIVE_NAMES,
            'keypoint_names_sha256': hashlib.sha256('\n'.join(NATIVE_NAMES).encode()).hexdigest(),
            'camera_order': list(CAMERA_ORDERING),
            'camera_policy': 'native HDF5 camera order; index c maps to source camera_c',
            'calibration_policy': 'native bundle_adjustment camera geometry; translations scaled to mm',
            'scale_policy': '0.456 mm / mean native front coxa lengths (0-1, 19-20)',
            'scale_reference_mm': REFERENCE_COXA_MM,
            'pixel_policy': 'numbered per-camera symlink farms target original extracted images',
            'keypoint_status_policy': 'finite selected 2D prediction=visible; NaN=missing (native pathway plan)',
            'point3d_status_policy': 'finite triangulated point with >=2 views=projected; no independent visibility',
            'confidence_policy': 'raw pose2d/conf retained as non-spec score; never used as a threshold',
            'split_counts': {s: sum(x.get('status') == 'converted' and x.get('split') == s
                                    for x in inventory) for s in fmt.SPLITS},
            'expected_source_sessions': len(records),
            'converted_sessions': sum(x.get('status') == 'converted' for x in inventory),
            'quarantined_source_sessions': sum(x.get('status') == 'quarantined' for x in inventory),
            'orphan_results': extras, 'input_signature': policy, 'records': inventory,
        }
        _atomic_json(root / 'conversion_manifest.json', manifest)
    print(f'converted {manifest["converted_sessions"]} native 3-D sessions; '
          f'quarantined {manifest["quarantined_source_sessions"]}')
    print('WARNING: units are scaled provisional mm; calibration remains provisional_unvalidated',
          file=sys.stderr)
    return 0


def main() -> int:
    """Dispatch inventory or conversion."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory', help='discover and audit native HDF5 results')
    inv.add_argument('--source', required=True, type=Path)
    inv.add_argument('--outputs', required=True, type=Path)
    inv.add_argument('--report', required=True, type=Path)
    inv.add_argument('--max-reprojection-px', type=float, default=None,
                     help='optional hard audit gate; default records residuals without rejecting')
    inv.add_argument('--strict', action='store_true')
    inv.set_defaults(func=inventory_command)
    conv = sub.add_parser('convert', help='write tailcycle mode=3d sessions')
    conv.add_argument('--source', required=True, type=Path)
    conv.add_argument('--outputs', required=True, type=Path)
    conv.add_argument('--out', required=True, type=Path)
    conv.add_argument('--force', action='store_true',
                      help='quarantine an existing root before replacing it')
    conv.add_argument('--workers', type=int, default=1,
                      help='reserved for compatibility; conversion is serialized per root')
    conv.add_argument('--max-reprojection-px', type=float, default=None,
                      help='optional hard audit gate; default records residuals without rejecting')
    conv.set_defaults(func=convert_command)
    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
