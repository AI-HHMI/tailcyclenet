#!/usr/bin/env python
"""Convert Rodent3D (Patel et al., IJCV 2023; BU Betke lab) into the tailcycle-dataset format.

    pixi run python scripts/convert_rodent3d.py --validate

Source: the public Dropbox share linked from https://www.cs.bu.edu/faculty/betke/Rodent3D/,
extracted under `--src`. Its `Rodent3D_V3/<recording>/split_<a>-<b>/` folders hold six
synchronised 60 fps videos (thermal Green/Orange/Red at 1024x1024, RGB-D Blue/Pink/Yellow at
848x480), per-RGB-D-camera depth `.hdf5` (not converted), and -- for 29 of the 40 splits -- an
`OptiPose3D_<a>-<b>.csv` of 3D keypoints. There are NO per-camera 2D labels in the release (the
yaml names DeepLabCut CSVs that were never uploaded), so every session is `points3d.pq` only,
`labels = "tracked"`: the 3D is OptiPose's output, not a human annotation.

Calibration. Each recording's `*_sample.yaml` carries EasyWand DLT coefficients per camera plus an
`OptiPose` block (rotation R, translation T, computed_scale cs) relating the DLT world to the CSV
frame. The toolkit's own reprojection (bu-cvkit 0.0.3.1, `DLTDeconstruction`) is

    X_dlt = ((X_csv - T*cs) / cs * [1, 1, -1]) @ inv(R);   (u, v) = DLT(L, X_dlt)

That map is affine, so it is folded into each camera's 3x4 DLT matrix, which is then RQ-decomposed
into an aniposelib pinhole K[R|t] with the CSV frame (mm) as world. DLT skew (|s| <= 0.02) is the
only thing dropped; `check_projection` asserts the aniposelib projection reproduces the DLT on the
session's own points. The camera whose name matches the video is the DLT to use -- the yaml's
`annotation.<cam>.view` field pairs cameras differently and is NOT the calibration.

Units: the CSV frame is OptiPose's scaled frame. Rat anatomy is consistent across both
calibration days (ear-to-ear ~29, snout-headBase ~53), so it is declared "mm".

Frames: CSV row i is video frame i (verified against background-subtracted rat centroids, best
offset 0). Videos carry 1-2 trailing frames beyond the CSV; `n_frames` is the CSV length, and a
camera whose video decodes fewer frames truncates the group (reported).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/rodent-3d/Rodent3D/Rodent3D_V3')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/rodent-3d')

# One session per recording. Two calibration days, each with one recording in train and one held
# out, so held-out numbers never rely on a calibration training never saw.
SPLIT = {'2022_05_10': 'train', '2022_05_11_2': 'train',
         '2022_05_10_1': 'val', '2022_05_11_1': 'test'}
CAMERAS = ('Blue', 'Green', 'Orange', 'Pink', 'Red', 'Yellow')
FLIP_PAIRS = [['leftEar', 'rightEar']]
FPS = 60.0
_SPLIT_DIR = re.compile(r'^split_(\d+)-(\d+)$')
_POINT = re.compile(r'[-+0-9.eE]+|nan', re.IGNORECASE)


# calibration

def world_projection(meta: dict, cam: str) -> np.ndarray:
    """The 3x4 matrix taking homogeneous CSV-frame points to this camera's pixels.

    Inputs: meta -- the recording's parsed *_sample.yaml; cam -- camera (= DLT view) name.
    Outputs: P (3, 4) float64, the DLT with bu-cvkit's CSV->DLT-world map folded in.

    bu-cvkit's row form X_dlt = ((X - T cs)/cs * D) @ inv(R) is the column form X_dlt = A X + b.
    """
    o = meta['OptiPose']
    R = np.asarray(o['rotation_matrix'], float)
    T = np.asarray(o['translation_matrix'], float)
    cs = float(o['computed_scale'])
    M = np.linalg.inv(R).T @ np.diag([1.0, 1.0, -1.0])
    A, b = M / cs, -(M @ T)
    L = np.asarray(meta['views'][cam]['dlt_coefficients'], float).reshape(3, 4)
    return L @ np.block([[A, b[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])


def decompose(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RQ-decompose a finite camera P ~ K [R | t].

    Outputs: K (3, 3) upper-triangular, positive diagonal, K[2,2] = 1 (skew kept here);
             R (3, 3) proper rotation; t (3,).
    """
    from scipy.linalg import rq

    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    K, R = rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    lam = K[2, 2]
    K = K / lam
    t = np.linalg.solve(K, P[:, 3]) / lam
    assert np.linalg.det(R) > 0
    return K, R, t


def dlt_project(meta: dict, cam: str, X: np.ndarray) -> np.ndarray:
    """bu-cvkit's DLTDeconstruction, verbatim, on CSV-frame points (N, 3) -> (N, 2) px."""
    o = meta['OptiPose']
    R = np.asarray(o['rotation_matrix'], float)
    T = np.asarray(o['translation_matrix'], float)
    cs = float(o['computed_scale'])
    Xd = ((X - T * cs) / cs * np.array([1.0, 1.0, -1.0])) @ np.linalg.inv(R)
    L = np.asarray(meta['views'][cam]['dlt_coefficients'], float)
    X1 = np.c_[Xd, np.ones(len(Xd))]
    return np.c_[X1 @ L[0:4] / (X1 @ L[8:12]), X1 @ L[4:8] / (X1 @ L[8:12])]


def build_rig(meta: dict, sizes: dict[str, tuple[int, int]]) -> tuple[fmt.Rig, float]:
    """aniposelib Rig for one recording, and the largest dropped skew.

    Inputs: meta -- parsed yaml; sizes -- {cam: (w, h)} of the videos on disk.
    """
    import cv2
    from aniposelib.cameras import Camera, CameraGroup

    cams, max_skew = [], 0.0
    for cam in CAMERAS:
        K, R, t = decompose(world_projection(meta, cam))
        max_skew = max(max_skew, abs(K[0, 1]))
        K = K.copy()
        K[0, 1] = 0.0
        cams.append(Camera(matrix=K, dist=np.zeros(5), rvec=cv2.Rodrigues(R)[0].ravel(),
                           tvec=t, name=cam, size=sizes[cam]))
    rig = fmt.Rig(CameraGroup(cams), offset={c: (0.0, 0.0) for c in CAMERAS},
                  moving={c: False for c in CAMERAS}, calibrated={c: True for c in CAMERAS})
    return rig, max_skew


def check_projection(rig: fmt.Rig, meta: dict, X: np.ndarray, tol_px: float = 0.05) -> float:
    """Max |aniposelib - bu-cvkit DLT| over X on every camera; raises above `tol_px`."""
    X = X[np.isfinite(X).all(1)]
    proj = rig.cgroup.project(X)
    proj = proj.detach().cpu().numpy() if hasattr(proj, 'detach') else np.asarray(proj)
    worst = 0.0
    for i, cam in enumerate(rig.names):
        worst = max(worst, float(np.abs(proj[i] - dlt_project(meta, cam, X)).max()))
    if worst > tol_px:
        raise RuntimeError(f'aniposelib projection disagrees with the DLT by {worst:.3f} px')
    return worst


# source reading

def read_optipose_csv(path: Path, names: list[str], start: int) -> np.ndarray:
    """OptiPose3D CSV (';'-separated, one '[x, y, z]' cell per keypoint) -> (T, K, 3) float32.

    The header names the columns; they are matched to `names` by name, never by position. The
    index column is the ABSOLUTE source frame (`start` + row) in every shipped CSV; asserted.
    """
    lines = path.read_text().splitlines()
    header = lines[0].split(';')
    cols = [header.index(n) for n in names]
    out = np.full((len(lines) - 1, len(names), 3), np.nan, np.float32)
    for i, line in enumerate(lines[1:]):
        cells = line.split(';')
        if int(cells[0]) != start + i:
            raise RuntimeError(f'{path}: row {i} is labelled frame {cells[0]}')
        for k, c in enumerate(cols):
            v = [float(x) for x in _POINT.findall(cells[c])]
            if len(v) >= 3:
                out[i, k] = v[:3]
    return out


def video_info(path: Path) -> tuple[tuple[int, int], int]:
    """((w, h), frames that actually decode) for one video, via the repo's PyAV reader path."""
    import av

    n = 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        wh = (s.codec_context.width, s.codec_context.height)
        for _ in c.decode(s):
            n += 1
    return wh, n


# conversion

def convert_recording(rec: Path, out_root: Path, dry_run: bool) -> None:
    """One recording -> one session under out_root/<split>/<recording>."""
    from concurrent.futures import ThreadPoolExecutor

    meta = yaml.safe_load(next(rec.glob('*_sample.yaml')).read_text())
    names = list(meta['body_parts'])
    split = SPLIT[rec.name]
    dst = out_root / split / rec.name

    splits = []
    for d in rec.iterdir():
        m = _SPLIT_DIR.match(d.name)
        if d.is_dir() and m:
            splits.append((int(m.group(1)), int(m.group(2)), d))
    splits.sort()

    groups, labels, sources, sizes, skipped = {}, {}, {}, {}, []
    all_pts = []
    for a, b, d in splits:
        csvs = list(d.glob('OptiPose3D_*.csv'))
        if not csvs:
            skipped.append(d.name)
            continue
        vids = {}
        for cam in CAMERAS:
            v = [p for p in d.glob('*.mp4') if p.stem.split('_')[-2] == cam]
            if len(v) != 1:
                raise RuntimeError(f'{d}: expected one {cam} video, found {len(v)}')
            vids[cam] = v[0].resolve()
        with ThreadPoolExecutor(len(CAMERAS)) as ex:
            info = dict(zip(CAMERAS, ex.map(video_info, [vids[c] for c in CAMERAS])))
        for cam, (wh, _) in info.items():
            if sizes.setdefault(cam, wh) != wh:
                raise RuntimeError(f'{d}: {cam} is {wh}, earlier splits were {sizes[cam]}')

        pts = read_optipose_csv(csvs[0], names, a)
        n_csv = len(pts)
        if n_csv != b - a:
            print(f'   ! {rec.name}/{d.name}: CSV has {n_csv} rows for a {b - a}-frame split')
        T = min([n_csv] + [n for _, n in info.values()])
        short = {c: n for c, (_, n) in info.items() if n < n_csv}
        if short:
            print(f'   ! {rec.name}/{d.name}: video(s) decode fewer frames than the CSV '
                  f'({n_csv}): {short}; group truncated to {T}')
        pts = pts[:T]
        all_pts.append(pts.reshape(-1, 3))

        gid = f'{a:05d}-{b:05d}'
        lab = fmt.empty_labels(1, T, len(names), len(CAMERAS), mode3d=True, animal_ids=['rat'])
        finite = np.isfinite(pts).all(-1)
        lab.vis3d[0][finite] = fmt.VISIBLE
        lab.points3d[0][finite] = pts[finite]
        lab.points2d, lab.vis2d = None, None
        groups[gid] = fmt.Group(gid, T, fps=FPS, source_video=str(d.relative_to(rec.parent.parent)),
                                source_frame_start=a, source_frame_step=1,
                                notes=f'decoded frames per camera: '
                                      f'{ {c: n for c, (_, n) in info.items()} }')
        labels[gid] = lab
        sources[gid] = vids

    rig, skew = build_rig(meta, sizes)
    err = check_projection(rig, meta, np.concatenate(all_pts)[::7])
    print(f'   {split}/{rec.name}: {len(groups)} group(s), '
          f'{sum(g.n_frames for g in groups.values())} frames; dropped {len(skipped)} split(s) '
          f'with no OptiPose3D CSV; |skew| <= {skew:.3f}, aniposelib vs DLT <= {err:.4f} px')
    if skipped:
        print(f'     no labels: {", ".join(skipped)}')
    if dry_run:
        return

    fmt.write_session(
        dst, mode='3d', units='mm', label_source='tracked', names=names, rig=rig,
        groups=groups, labels=labels, skeleton=[list(p) for p in meta['skeleton']],
        flip_pairs=FLIP_PAIRS,
        provenance={
            'source': f'Rodent3D (https://www.cs.bu.edu/faculty/betke/Rodent3D/) '
                      f'Rodent3D_V3/{rec.name}',
            'annotator': '',
            'annotator_tool': 'OptiPose 3D output (OptiPose3D_*.csv); '
                              'converted by scripts/convert_rodent3d.py',
            'calibration': 'EasyWand DLT + OptiPose R/T/computed_scale (bu-cvkit '
                           'DLTDeconstruction) folded and RQ-decomposed to pinhole; '
                           f'max |skew| dropped {skew:.3f}, max reprojection change {err:.4f} px',
            'created': str(np.datetime64('today')),
        })
    for gid, vids in sources.items():
        gdir = dst / 'groups' / gid
        gdir.mkdir(parents=True, exist_ok=True)
        for cam, src in vids.items():
            fmt.link(gdir / f'{cam}.mp4', src)


def main() -> None:
    """Convert every Rodent3D_V3 recording; optionally validate. Exit 1 on validation errors."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--dry-run', action='store_true', help='report, write nothing')
    ap.add_argument('--validate', action='store_true', help='validate after writing')
    ap.add_argument('--clean', action='store_true', help='remove --out first')
    args = ap.parse_args()

    if args.clean and args.out.exists() and not args.dry_run:
        shutil.rmtree(args.out)
    for rec in sorted(p for p in args.src.iterdir() if p.is_dir()):
        if rec.name not in SPLIT:
            print(f'   ! {rec.name}: not in SPLIT, skipped')
            continue
        convert_recording(rec, args.out, args.dry_run)

    if args.validate and not args.dry_run:
        errs = fmt.validate_dataset(fmt.load_dataset(args.out))
        hard = [e for e in errs if 'WARNING' not in e]
        for e in errs:
            print(('  WARN ' if 'WARNING' in e else '  FAIL ') + e)
        print(f'rodent-3d: {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')
        sys.exit(1 if hard else 0)


if __name__ == '__main__':
    main()
