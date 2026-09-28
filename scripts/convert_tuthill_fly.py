#!/usr/bin/env python
"""Convert the Tuthill fly DLC export to tailcycle-dataset format.

The export contains ordinary single-camera DLC folders and six-camera annotation sets.  A
six-camera set is detected by a common folder prefix after removing the final ``[-_]A`` ...
``[-_]F`` suffix and by the presence of ``anipose_metadata.csv`` in every member.  Metadata rows
are checked for synchronized source frame numbers before being combined into one 3D session.

The checked-in Tuthill calibration files are used as the starting calibration (VGGT is optional
and is not installed in the project environment).  For synchronized sets, aniposelib's PyTorch
bundle adjustment refines extrinsics using the labelled points, then observations with a final
reprojection residual above ``--reject-px`` are removed from both the 2D and derived 3D layers.
Pixels are symlinked, never copied.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/tuthill-fly/labeled-data')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
           'tailcycle-datasets/tuthill-fly-annotated')
SARAH_CAL_ROOT = Path('scratch/tuthill_sarah_calibrations')
CAMERAS = tuple('ABCDEF')
NAMES = ('L1A', 'L1B', 'L1C', 'L1D', 'L1E', 'L2A', 'L2B', 'L2C', 'L2D', 'L2E',
         'L3A', 'L3B', 'L3C', 'L3D', 'L3E', 'R1A', 'R1B', 'R1C', 'R1D', 'R1E',
         'R2A', 'R2B', 'R2C', 'R2D', 'R2E', 'R3A', 'R3B', 'R3C', 'R3D', 'R3E')
_SCHEME = (('L1A', 'L1B', 'L1C', 'L1D', 'L1E'),
           ('L2A', 'L2B', 'L2C', 'L2D', 'L2E'),
           ('L3A', 'L3B', 'L3C', 'L3D', 'L3E'),
           ('R1A', 'R1B', 'R1C', 'R1D', 'R1E'),
           ('R2A', 'R2B', 'R2C', 'R2D', 'R2E'),
           ('R3A', 'R3B', 'R3C', 'R3D', 'R3E'))
SKELETON = tuple((chain[i], chain[i + 1]) for chain in _SCHEME for i in range(len(chain) - 1))
FLIP_PAIRS = tuple((f'L{leg}{segment}', f'R{leg}{segment}')
                   for leg in range(1, 4) for segment in 'ABCDE')
_SUFFIX = re.compile(r'[-_]([A-F])$')


def _session_id(name: str) -> str:
    """Make a stable tailcycle session id from a source folder prefix."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name.rstrip('-_'))


def read_dlc(path: Path) -> tuple[list[str], list[dict]]:
    """Read a DLC ``CollectedData`` CSV into image names and Kx2 float arrays."""
    with path.open(newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 4 or rows[1][0] != 'bodyparts' or rows[2][0] != 'coords':
        raise RuntimeError(f'{path}: not a three-header-row DLC CSV')
    names = tuple(rows[1][1::2])
    if names != NAMES:
        raise RuntimeError(f'{path}: unexpected keypoint axis ({len(names)} names)')
    out = []
    for row in rows[3:]:
        if not row or not row[0]:
            continue
        vals = row[1:1 + 2 * len(names)]
        if len(vals) != 2 * len(names):
            raise RuntimeError(f'{path}: row for {row[0]!r} has {len(vals)} coordinate fields')
        xy = np.full((len(names), 2), np.nan, np.float64)
        for i, value in enumerate(vals):
            if value != '':
                try:
                    xy.flat[i] = float(value)
                except ValueError as e:
                    raise RuntimeError(f'{path}: non-numeric coordinate {value!r}') from e
        out.append({'image': Path(row[0].replace('\\', '/')).name, 'xy': xy})
    return list(names), out


def image_for(folder: Path, name: str) -> Path:
    """Resolve a CSV/metadata image basename without trusting its stale absolute prefix."""
    path = folder / name
    if path.exists():
        return path
    matches = sorted(p for p in folder.iterdir() if p.is_file() and p.name == name)
    if len(matches) != 1:
        raise RuntimeError(f'{folder}: cannot resolve image {name!r}')
    return matches[0]


def config_name(prefix: str) -> str:
    """Map a source recording prefix to its archived calibration/config filename.

    Offball exports reference a missing workstation calibration; use the matching pre-Sarah
    8.13.19 rig rather than 7.24.20, whose residuals are about 50 pixels.
    """
    if prefix.startswith('2019-09-04'):
        return '5.22.19'
    if prefix.startswith('evyn6_2020-04-15'):
        return '5.22.19'
    if prefix.startswith('evyn6_2020-05-07'):
        return '8.13.19'
    if prefix.startswith('offball_flies_'):
        return '8.13.19'
    if prefix.startswith('sarah6_2020-07-29'):
        return '7.24.20'
    m = re.search(r'sarah6_(\d+\.\d+\.\d+)', prefix)
    if m:
        return m.group(1)
    raise RuntimeError(f'{prefix}: no archived calibration mapping')


def calibrated_rig(src: Path, prefix: str, sizes: dict[str, tuple[int, int]],
                   sarah_cal_root: Path | None = None) -> tuple[fmt.Rig, str]:
    """Load an exact Sarah date calibration and attach its crop offsets/sizes.

    Sarah sessions are eligible for 3D only when the rclone-populated cache contains both
    ``<date>/metadata/config.toml`` and ``<date>/calibration.toml``. No archived fallback is
    allowed; a missing pair is handled by the caller as 2D. Some older Anipose ROI exports use
    inclusive width/height endpoints, so decoded image dimensions are authoritative.
    """
    stem = config_name(prefix)
    if prefix.startswith('sarah6_'):
        if sarah_cal_root is None:
            raise FileNotFoundError('Sarah calibration cache was not supplied')
        config_path = sarah_cal_root / stem / 'metadata' / 'config.toml'
        cal_path = sarah_cal_root / stem / 'metadata' / 'calibration.toml'
        if not cal_path.is_file():
            cal_path = sarah_cal_root / stem / 'calibration.toml'
        if not cal_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f'{stem}: require {config_path} and metadata/calibration.toml')
    else:
        cal_path = src / 'calibrations' / f'{stem}.toml'
        config_path = src / 'configs' / f'{stem}.toml'
    rig = fmt.load_calibration(cal_path)
    with config_path.open('rb') as f:
        cfg = tomllib.load(f)
    cameras = cfg.get('cameras', {})
    for cam in rig.names:
        values = cameras.get(cam, {}).get('offset')
        if values is None or len(values) != 4:
            raise RuntimeError(f'{stem}: missing four-value crop offset for camera {cam}')
        off = tuple(float(v) for v in values[:2])
        cfg_size = tuple(int(v) for v in values[2:])
        rig.offset[cam] = off
        rig.by_name(cam).set_size(sizes.get(cam, cfg_size))
        rig.moving[cam] = False
    return rig, stem


def _np(value) -> np.ndarray:
    """Detach a PyTorch tensor returned by the pytorch aniposelib branch."""
    return np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value, dtype=np.float64)


def reprojection(rig: fmt.Rig, p2_sensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate and return (3D, per-camera pixel residual norm)."""
    p3 = _np(rig.cgroup.triangulate(p2_sensor, progress=False))
    err = np.linalg.norm(_np(rig.cgroup.reprojection_error(p3, p2_sensor)), axis=-1)
    return p3, err


def refine_rig(rig: fmt.Rig, p2_sensor: np.ndarray, reject_px: float) -> tuple[fmt.Rig, np.ndarray, np.ndarray, dict]:
    """Reject gross observations, bundle-adjust extrinsics, and apply the final gate.

    Keep the adjustment only when its median residual is not plainly worse than the archived
    calibration. The camera group is mutated by bundle adjustment, so callers must reload it if
    that guard rejects the result.
    """
    work = p2_sensor.copy()
    initial_p3, initial_err = reprojection(rig, work)
    for _ in range(2):
        _, err = reprojection(rig, work)
        work[err > reject_px] = np.nan
    filtered = np.isfinite(work).all(axis=(0, 2))
    ba_rig = rig
    ba_message = 'not enough complete points'
    if int(filtered.sum()) >= 8:
        try:
            rig.cgroup.bundle_adjust(work[:, filtered], loss='huber', only_extrinsics=True,
                                     max_nfev=200, verbose=False)
            ba_message = 'ok'
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            ba_message = f'{type(e).__name__}: {e}'
    final_p3, final_err = reprojection(ba_rig, work)
    finite_initial = initial_err[np.isfinite(initial_err)]
    finite_final = final_err[np.isfinite(final_err)]
    if (not len(finite_final) or (len(finite_initial) and
                                  np.nanmedian(finite_final) > max(1.0, 2 * np.nanmedian(finite_initial)))):
        ba_rig = rig
        ba_message = 'rejected: bundle adjustment worsened residuals'
    bad = np.isfinite(final_err) & (final_err > reject_px)
    work[bad] = np.nan
    final_p3, final_err = reprojection(ba_rig, work)
    stats = {
        'initial_p50': float(np.nanmedian(initial_err)) if np.isfinite(initial_err).any() else float('nan'),
        'final_p50': float(np.nanmedian(final_err)) if np.isfinite(final_err).any() else float('nan'),
        'removed': int(bad.sum()), 'ba': ba_message, 'complete_points': int(filtered.sum()),
    }
    return ba_rig, work, final_p3, stats


def make_multiview(src: Path, out: Path, prefix: str, folders: list[Path], reject_px: float,
                   clean: bool, sarah_cal_root: Path | None = None) -> dict:
    """Convert one synchronized six-camera prefix into one 3D session.

    Metadata can include unlabeled images; retain only images present in every camera's DLC CSV.
    Preserve source DLC coordinates as 2D labels: sensor-offset coordinates are used only for
    triangulation and gating, and filtered observations are omitted rather than replaced by
    reprojections.
    """
    folders = sorted(folders, key=lambda p: p.name[-1])
    if tuple(p.name[-1] for p in folders) != CAMERAS:
        raise RuntimeError(f'{prefix}: expected cameras A-F, got {[p.name for p in folders]}')
    csv_rows = [read_dlc(p / 'CollectedData_TuthillLab.csv')[1] for p in folders]
    csv_maps = [{r['image']: r for r in rows} for rows in csv_rows]
    with (folders[0] / 'anipose_metadata.csv').open(newline='') as f:
        metadata_all = list(csv.DictReader(f))
    common_images = set.intersection(*(set(m) for m in csv_maps))
    metadata = [r for r in metadata_all if Path(r['img'].replace('\\', '/')).name in common_images]
    if not metadata:
        raise RuntimeError(f'{prefix}: no metadata rows correspond to labeled images')
    csv_data = [[m[Path(r['img'].replace('\\', '/')).name] for r in metadata] for m in csv_maps]
    frame_lists = []
    for p in folders:
        with (p / 'anipose_metadata.csv').open(newline='') as f:
            rows = {Path(r['img'].replace('\\', '/')).name: r for r in csv.DictReader(f)}
        frame_lists.append([int(rows[Path(m['img'].replace('\\', '/')).name]['framenum'])
                            for m in metadata])
    if any(frames != frame_lists[0] for frames in frame_lists[1:]):
        raise RuntimeError(f'{prefix}: metadata does not contain synchronized frame numbers')

    sizes = {}
    for cam, folder, rows in zip(CAMERAS, folders, csv_data):
        import cv2
        im = cv2.imread(str(image_for(folder, rows[0]['image'])), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f'{folder}: cannot decode {rows[0]["image"]}')
        sizes[cam] = (int(im.shape[1]), int(im.shape[0]))
    rig, cal_name = calibrated_rig(src, prefix, sizes, sarah_cal_root)
    raw = np.stack([np.stack([r['xy'] for r in rows]).reshape(-1, 2) for rows in csv_data])
    offsets = np.asarray([rig.offset[c] for c in CAMERAS])[:, None, :]
    rig, work, p3, stats = refine_rig(rig, raw + offsets, reject_px)

    dst = out / 'train' / _session_id(prefix)
    if clean and dst.exists():
        shutil.rmtree(dst)
    groups, labels = {}, {}
    K = len(NAMES)
    for i, meta in enumerate(metadata):
        gid = f'{i:06d}'
        lab = fmt.empty_labels(1, 1, K, len(CAMERAS), mode3d=True, animal_ids=['a00'])
        raw_points = raw[:, i * K:(i + 1) * K]
        source_missing = ~np.isfinite(raw_points).all(-1)
        removed = np.isfinite(raw_points).all(-1) & ~np.isfinite(work[:, i * K:(i + 1) * K]).all(-1)
        points = raw_points.copy()
        points[removed] = np.nan
        finite = np.isfinite(points).all(-1)
        lab.points2d[0, 0, :, :, :] = np.moveaxis(points, 0, 1).astype(np.float32)
        lab.vis2d[0, 0][source_missing.T] = fmt.MISSING
        lab.vis2d[0, 0][removed.T] = fmt.UNLABELED
        lab.vis2d[0, 0][finite.T] = fmt.VISIBLE
        p = p3[i * K:(i + 1) * K]
        good3 = np.isfinite(p).all(-1) & (np.isfinite(points).all(-1).sum(0) >= 2)
        lab.points3d[0, 0] = p.astype(np.float32)
        lab.vis3d[0, 0] = fmt.MISSING
        lab.vis3d[0, 0, good3] = fmt.VISIBLE
        groups[gid] = fmt.Group(gid, 1, fps=float('nan'), source_video=meta['video'],
                                source_frame_start=int(meta['framenum']), source_frame_step=1,
                                notes=f'synchronized annotation sample; calibration={cal_name}')
        labels[gid] = lab
        for ci, folder in enumerate(folders):
            cdir = dst / 'groups' / gid / CAMERAS[ci]
            cdir.mkdir(parents=True, exist_ok=True)
            fmt.link(cdir / '000000.png', image_for(folder, csv_data[ci][i]['image']).resolve())
    if not dst.exists():
        dst.mkdir(parents=True)
    fmt.write_session(
        dst, mode='3d', units='mm', label_source='annotated', names=list(NAMES), rig=rig,
        groups=groups, labels=labels, skeleton=SKELETON, flip_pairs=FLIP_PAIRS,
        provenance={'source': f'tuthill-fly/labeled-data/{prefix}', 'annotator': '',
                    'annotator_tool': 'DeepLabCut/anipose', 'converter': 'scripts/convert_tuthill_fly.py',
                    'calibration_init': f'rclone date calibration {cal_name}; VGGT unavailable',
                    'calibration_refinement': 'aniposelib pytorch bundle_adjust (extrinsics only)',
                    'reprojection_gate_px': str(reject_px), 'points3d_source': 'triangulated from 2D labels'})
    print(f'3d {prefix}: {len(groups)} groups, {stats}')
    return {'mode': '3d', 'groups': len(groups), **stats}


def nominal_rig(size: tuple[int, int]) -> fmt.Rig:
    """Build the required one-camera nominal calibration for an uncalibrated 2D folder."""
    from aniposelib.cameras import CameraGroup
    cam = fmt.nominal_camera('A', size, None)
    return fmt.Rig(CameraGroup([cam]), offset={'A': (0.0, 0.0)},
                   moving={'A': False}, calibrated={'A': False})


def make_single(src: Path, out: Path, folder: Path, clean: bool) -> dict:
    """Convert one ordinary DLC folder to a sparse 2D session."""
    _, rows = read_dlc(folder / 'CollectedData_TuthillLab.csv')
    if not rows:
        raise RuntimeError(f'{folder}: no labelled images')
    import cv2
    im = cv2.imread(str(image_for(folder, rows[0]['image'])), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f'{folder}: cannot decode {rows[0]["image"]}')
    size = (int(im.shape[1]), int(im.shape[0]))
    rig = nominal_rig(size)
    dst = out / 'train' / _session_id(folder.name)
    if clean and dst.exists():
        shutil.rmtree(dst)
    groups, labels = {}, {}
    for i, row in enumerate(rows):
        gid = f'{i:06d}'
        lab = fmt.empty_labels(1, 1, len(NAMES), 1, mode3d=False, animal_ids=['a00'])
        finite = np.isfinite(row['xy']).all(-1)
        lab.points2d[0, 0, :, 0] = row['xy'].astype(np.float32)
        lab.vis2d[0, 0, finite, 0] = fmt.VISIBLE
        lab.vis2d[0, 0, ~finite, 0] = fmt.MISSING
        groups[gid] = fmt.Group(gid, 1, fps=float('nan'), source_video=str(folder),
                                source_frame_start=i, source_frame_step=1)
        labels[gid] = lab
        cdir = dst / 'groups' / gid / 'A'
        cdir.mkdir(parents=True, exist_ok=True)
        fmt.link(cdir / '000000.png', image_for(folder, row['image']).resolve())
    fmt.write_session(
        dst, mode='2d', units='px', label_source='annotated', names=list(NAMES), rig=rig,
        groups=groups, labels=labels, skeleton=SKELETON, flip_pairs=FLIP_PAIRS,
        provenance={'source': f'tuthill-fly/labeled-data/{folder.name}', 'annotator': '',
                    'annotator_tool': 'DeepLabCut', 'converter': 'scripts/convert_tuthill_fly.py'})
    print(f'2d {folder.name}: {len(groups)} groups')
    return {'mode': '2d', 'groups': len(groups)}


def convert(src: Path, out: Path, reject_px: float, clean: bool,
            sarah_cal_root: Path | None = None) -> None:
    """Convert all folders containing the Tuthill DLC CSV.

    Only Sarah6 sets can become 3D, and only with an exact rclone-sourced calibration pair.
    Every other folder is exported as an independent 2D session.
    """
    folders = sorted(p.parent for p in src.glob('*/CollectedData_TuthillLab.csv'))
    by_prefix: dict[str, list[Path]] = {}
    singles = []
    for folder in folders:
        if (folder / 'anipose_metadata.csv').exists():
            prefix = _SUFFIX.sub('', folder.name).rstrip('-_')
            by_prefix.setdefault(prefix, []).append(folder)
        else:
            singles.append(folder)
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for prefix, members in sorted(by_prefix.items()):
        if len(members) != 6:
            raise RuntimeError(f'{prefix}: metadata folders are not a complete six-camera set')
        if prefix.startswith('sarah6_'):
            try:
                stats.append(make_multiview(src, out, prefix, members, reject_px, clean,
                                            sarah_cal_root))
                continue
            except FileNotFoundError as e:
                print(f'2d {prefix}: {e}; no exact rclone calibration pair')
        for folder in members:
            stats.append(make_single(src, out, folder, clean))
    for folder in singles:
        stats.append(make_single(src, out, folder, clean))
    print(f'converted {len(stats)} sessions ({sum(s["mode"] == "3d" for s in stats)} 3D, '
          f'{sum(s["mode"] == "2d" for s in stats)} 2D)')


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--reject-px', type=float, default=10.0)
    ap.add_argument('--sarah-cal-root', type=Path, default=SARAH_CAL_ROOT,
                    help='rclone cache with <date>/metadata/config.toml and calibration.toml')
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    convert(args.src, args.out, args.reject_px, args.clean, args.sarah_cal_root)


if __name__ == '__main__':
    main()
