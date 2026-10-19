"""Reading a native ``deeperfly`` HDF5 result (points, cameras, scores) for ``deepfly3d_v3.py``.

Also the source-image naming rule (``camera_files``) and the metric scale from the front coxa
segments. Imported by its siblings as ``from deepfly3d_h5 import ...``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np




N_CAMERAS = 7


N_KEYPOINTS = 38


REFERENCE_COXA_MM = 0.456


FRONT_COXA_PAIRS = ((0, 1), (19, 20))


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


@dataclass
class NativeResult:
    """Materialized arrays and calibration from one native HDF5 result."""

    points2d: np.ndarray
    confidence: np.ndarray
    points3d: np.ndarray
    camera_names: list[str]
    intrinsics: np.ndarray
    distortions: list[np.ndarray]
    rvecs: np.ndarray
    tvecs: np.ndarray
    names: list[str]
    bones: list[tuple[int, int]]
    image_sizes: list[tuple[int, int]]
    source_camera_names: list[str]
    points2d_source: str = 'pose2d/points'

    @property
    def n_frames(self) -> int:
        """Number of frames represented by the native view-leading arrays."""
        return int(self.points2d.shape[1])


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
    if files[:n_frames] != expected:
        raise ValueError(f'{images}: camera {camera} is not contiguous 000000..{n_frames - 1:06d}')
    return files[:n_frames]


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
