#!/usr/bin/env python
"""Convert `johnson-mouse/merge` into the tailcycle-dataset format.

    pixi run python scripts/convert_johnson.py --dry-run
    pixi run python scripts/convert_johnson.py --clean

A 16-camera, single-mouse, 24-keypoint COCO-like JSON with per-trial OpenCV calibration and a
ready-made train/val partition. One session per (split, trial); the same 21 trial names appear in
both splits, which trips `[rule 14 WARNING]` -- these really are the same recordings, and the
split is per-frameset, so the warning is honest rather than something to rename around.

Three source facts decide the shape of this script:

- **`cv2.FileStorage` silently returns garbage** for 13 of the 21 calibration trials. The newer
  yamls write bare integers (`0` not `0.`) under `dt: d`, which OpenCV misparses into ~2**32 and
  -5.9e18. `parse_calib` is a regex reader instead. The convention -- verified against all four
  transpose combinations on all 21 trials -- is `matrix = intrinsicMatrix.T`,
  `rvec = Rodrigues(R.T)`, `tvec = T`, and ALL FIVE distortion coefficients: truncating to k1,k2
  (as posetail-preprocessing's `camera_from_path` does) costs 7.7 px on 2025_10_20.

- **There is no 3D in the source**, but a `mode="3d"` session is unusable without points3d.pq
  (`dataset.py` and `detector/data.py` both index `lab.points3d` unconditionally) and `mode="2d"`
  demands exactly one camera. So the 3D layer here is DERIVED -- triangulated from the 16-view 2D,
  which reprojects at ~0.6 px. `provenance.points3d_source` says so, and `--check` measures it.
  The fit rejects outliers (`triangulate_robust`) because plain least squares has no breakdown
  point and one bad view is enough to make all sixteen look bad.

- **1,871 ghost jpgs** sit on disk unreferenced by the current annotations. Everything is driven
  from the JSON `images` list; a filesystem walk would silently enrol unlabelled frames.

Annotated frames come in strided runs (modal source step 1, 4 or 6 depending on trial), so groups
are runs cut wherever the source frame gap exceeds `--max-gap`. That buys real temporal context
for 63% of frames instead of 3,614 single-frame groups.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/johnson-mouse/merge')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/'
           'johnson-mouse-annotated')
SPLITS = ('train', 'val')

_MAT = re.compile(r'^(\w+):\s*!!opencv-matrix\s*\n\s*rows:\s*(\d+)\s*\n\s*cols:\s*(\d+)'
                  r'\s*\n\s*dt:\s*\w+\s*\n\s*data:\s*\[(.*?)\]', re.S | re.M)
_NUM = re.compile(r'-?[\d.]+(?:e[-+]?\d+)?')


# calibration

def parse_calib(path: Path) -> dict[str, np.ndarray]:
    """Read an OpenCV FileStorage yaml WITHOUT cv2.

    `cv2.FileStorage` returns ~2**32 and -5.9e18 for every trial from 2026_04_20 onward, because
    those files write bare integers under `dt: d`. Nothing about the failure is loud.
    """
    txt = path.read_text()
    out: dict[str, np.ndarray] = {}
    for m in _MAT.finditer(txt):
        vals = [float(v) for v in _NUM.findall(m.group(4))]
        out[m.group(1)] = np.array(vals, dtype=np.float64).reshape(int(m.group(2)),
                                                                   int(m.group(3)))
    for key in ('intrinsicMatrix', 'distortionCoefficients', 'R', 'T'):
        if key not in out:
            raise RuntimeError(f'{path}: no {key}')
    return out


def build_rig(src: Path, calib_paths: dict[str, str], sizes: dict[str, tuple[int, int]]) -> fmt.Rig:
    """One aniposelib camera per view. MATLAB exports column-major, hence the transposes."""
    import cv2
    from aniposelib.cameras import Camera, CameraGroup

    cams, offset, moving, calibrated = [], {}, {}, {}
    for name in sorted(calib_paths):
        c = parse_calib(src / calib_paths[name])
        rvec = cv2.Rodrigues(c['R'].T)[0].ravel()
        cam = Camera(matrix=c['intrinsicMatrix'].T,
                     dist=c['distortionCoefficients'].ravel()[:5],
                     rvec=rvec, tvec=c['T'].ravel(), name=name)
        # The yaml's own image_width/height is 0 for five trials; the JSON always knows.
        cam.set_size(sizes[name])
        cams.append(cam)
        offset[name] = (0.0, 0.0)               # full sensor, no crop
        moving[name] = False
        calibrated[name] = True
    return fmt.Rig(CameraGroup(cams), offset=offset, moving=moving, calibrated=calibrated)


# source reading

def read_split(src: Path, split: str) -> dict:
    """The annotation JSON, indexed. `images[i].id == i` holds but is not relied on."""
    with open(src / 'annotations' / f'instances_{split}.json') as f:
        d = json.load(f)

    imgs = {im['id']: im for im in d['images']}
    anns: dict[int, dict] = {}
    for a in d['annotations']:
        if a['image_id'] in anns:
            raise RuntimeError(f'{split}: image {a["image_id"]} has >1 annotation')
        anns[a['image_id']] = a

    # trial -> {source frame index: {camera: image_id}}
    framesets: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    for key, fs in d['framesets'].items():
        trial, frame = key.rsplit('/Frame_', 1)
        views = {}
        for iid in fs['frames']:
            parts = imgs[iid]['file_name'].split('/')
            if parts[0] != trial:
                raise RuntimeError(f'{split}: frameset {key} holds image from {parts[0]}')
            views[parts[1]] = iid
        framesets[trial][int(frame)] = views

    d['_imgs'], d['_anns'], d['_framesets'] = imgs, anns, framesets
    return d


def runs(frames: list[int], max_gap: int) -> list[list[int]]:
    """Cut a sorted frame list into contiguous-enough clips."""
    out: list[list[int]] = [[frames[0]]]
    for a, b in zip(frames, frames[1:]):
        if b - a > max_gap:
            out.append([])
        out[-1].append(b)
    return out


def modal_step(frames: list[int]) -> int:
    if len(frames) < 2:
        return 1
    return int(Counter(np.diff(frames).tolist()).most_common(1)[0][0])


# labels

def triangulate_robust(cgroup, p2d: np.ndarray, reject_px: float,
                       iters: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """DLT, then drop grossly disagreeing observations and refit. Returns (p3d, rejected mask).

    Plain least squares has no breakdown point: ONE view off by an image height puts every one of
    the other 15 reprojections at 130-400 px, which reads as "all views disagree" when in fact 15
    of them agree to 0.6 px. That is exactly what 2025_02_12/Cam2006054 does on 10 framesets.

    Rejection is per (point, camera), not per camera, so the offending view still contributes its
    other keypoints. The threshold is `max(reject_px, 5x the point's median residual)`: relative,
    because on the first pass a dragged fit makes every residual large, and absolute, so a clean
    fit at 0.6 px never starts trimming its own tail.
    """
    p2 = p2d.copy()
    p3 = _np(cgroup.triangulate(p2, progress=False))
    rejected = np.zeros(p2.shape[:2], bool)                                      # (C, N)
    for _ in range(iters):
        e = np.linalg.norm(_np(cgroup.reprojection_error(p3, p2)), axis=-1)      # (C, N)
        with np.errstate(invalid='ignore'):
            med = np.nanmedian(np.where(np.isfinite(e), e, np.nan), axis=0)      # (N,)
        bad = e > np.maximum(reject_px, 5.0 * med)
        if not bad.any():
            break
        rejected |= bad
        p2[bad] = np.nan
        p3 = _np(cgroup.triangulate(p2, progress=False))
    return p3, rejected


def build_labels(d: dict, run: list[int], views: dict[int, dict[str, int]], rig: fmt.Rig,
                 K: int, reject_px: float) -> tuple[fmt.Labels, int]:
    """Dense arrays for one group. 2D + boxes from the JSON, 3D triangulated from the 2D.

    An image with no annotation leaves every cell UNLABELED (no row) rather than `present`: the
    source records no determination about those views, and inventing one would be an annotation
    nobody made.

    Outlier 2D is rejected from the TRIANGULATION only. It is still exported verbatim to
    keypoints.pq: it is the annotators' work, and the derived layer is the one that gets to be
    opinionated about it.
    """
    cams = rig.names
    C, T = len(cams), len(run)
    lab = fmt.empty_labels(1, T, K, C, mode3d=True, animal_ids=['a00'])
    # boxes and instance must be allocated together -- write_session indexes boxes wherever
    # instance says a row exists.
    lab.boxes = np.full((1, T, C, 4), np.nan, np.float32)
    lab.instance = np.full((1, T, C), fmt.INST_NONE, np.int8)

    for t, frame in enumerate(run):
        for ci, cam in enumerate(cams):
            iid = views[frame].get(cam)
            if iid is None:
                continue
            a = d['_anns'].get(iid)
            if a is None:                      # image exists, nobody labelled it
                continue
            kp = np.asarray(a['keypoints'], dtype=np.float64).reshape(K, 3)
            seen = kp[:, 2] > 0
            # PROJECTED, not VISIBLE. The source flags 1,235,334 points "visible" against 18 "not
            # visible" across 16 views of a mouse on a dome -- the far-side limbs cannot all have
            # been seen, so these are inferred positions and the visibility flag asserts nothing.
            # The 18 zero-flag points carry (0,0), the source's null marker, so they get no row at
            # all: `missing` would claim someone looked and judged them occluded (spec 13).
            lab.vis2d[0, t, seen, ci] = fmt.PROJECTED
            lab.points2d[0, t, seen, ci] = kp[seen, :2].astype(np.float32)

            x, y, w, h = (float(v) for v in a['bbox'])
            lab.boxes[0, t, ci] = (x, y, x + w, y + h)
            lab.instance[0, t, ci] = fmt.INST_LABELED

    # 2D -> 3D. aniposelib's triangulate is NaN-safe and returns NaN below 2 views.
    p2d = np.moveaxis(lab.points2d[0], 2, 0).reshape(C, T * K, 2).astype(np.float64)
    p3d, rejected = triangulate_robust(rig.cgroup, p2d, reject_px)
    p3d = p3d.reshape(T, K, 3)
    ok = np.isfinite(p3d).all(-1)
    lab.vis3d[0][ok] = fmt.VISIBLE
    lab.points3d[0][ok] = p3d[ok].astype(np.float32)

    # A rejected observation is demonstrably in the wrong place, so it leaves keypoints.pq too --
    # as NO ROW, not `missing`: `missing` would claim someone looked and judged it occluded.
    bad = rejected.reshape(C, T, K).transpose(1, 2, 0)                    # (T,K,C)
    lab.vis2d[0][bad] = fmt.UNLABELED
    lab.points2d[0][bad] = np.nan
    # A view that lost every observation also loses its box: the source box is the keypoint hull,
    # so it carries the same displacement. Dropping it keeps rule 11 (a `labeled` instance must
    # still have keypoint rows) as well.
    dead = (lab.vis2d[0] == fmt.UNLABELED).all(1) & (lab.instance[0] != fmt.INST_NONE)
    lab.instance[0][dead] = fmt.INST_NONE
    lab.boxes[0][dead] = np.nan
    return lab, int(rejected.sum())


def _np(x) -> np.ndarray:
    """aniposelib is on the pytorch branch: its outputs carry requires_grad."""
    return np.asarray(x.detach().cpu() if hasattr(x, 'detach') else x, dtype=np.float64)


# conversion

def convert(src: Path, out: Path, max_gap: int, only: list[str] | None, max_groups: int | None,
            dry_run: bool, reject_px: float) -> None:
    for split in SPLITS:
        d = read_split(src, split)
        names = list(d['keypoint_names'])
        K = len(names)
        skeleton = [[j['keypointA'], j['keypointB']] for j in d['skeleton']]
        flip_pairs = [[n, n[:-1] + 'R'] for n in names
                      if n.endswith('L') and n[:-1] + 'R' in names]

        for trial in sorted(d['_framesets']):
            if only and trial not in only:
                continue
            fsets = d['_framesets'][trial]
            frames = sorted(fsets)

            cams = sorted(d['calibrations'][trial])
            for f in frames:
                if sorted(fsets[f]) != cams:
                    raise RuntimeError(f'{split}/{trial}/{f}: cameras {sorted(fsets[f])} '
                                       f'disagree with calibration {cams}')
            sizes = {c: (d['_imgs'][fsets[frames[0]][c]]['width'],
                         d['_imgs'][fsets[frames[0]][c]]['height']) for c in cams}
            rig = build_rig(src, d['calibrations'][trial], sizes)

            clips = runs(frames, max_gap)
            if max_groups:
                clips = clips[:max_groups]

            dst = out / split / trial
            groups, labels, gated = {}, {}, 0
            for run in clips:
                gid = f'{run[0]:06d}'
                groups[gid] = fmt.Group(gid, len(run), fps=float('nan'),
                                        source_video=str(src / split / trial),
                                        source_frame_start=run[0],
                                        source_frame_step=modal_step(run))
                labels[gid], n = build_labels(d, run, fsets, rig, K, reject_px)
                gated += n
                if dry_run:
                    continue
                for cam in rig.names:
                    cdir = dst / 'groups' / gid / cam
                    cdir.mkdir(parents=True, exist_ok=True)
                    for t, frame in enumerate(run):
                        s = (src / split / d['_imgs'][fsets[frame][cam]]['file_name']).resolve()
                        fmt.link(cdir / f'{t:06d}.jpg', s)

            n_lab = sum(int((lab.vis2d != fmt.UNLABELED).any(2).sum()) for lab in labels.values())
            note = (f', {gated} outlier 2D observation(s) rejected from the triangulation'
                    if gated else '')
            print(f'   {split}/{trial}: {len(groups)} group(s), '
                  f'{sum(g.n_frames for g in groups.values())} frames, {len(rig)} cams, '
                  f'{n_lab} labelled views{note}')
            if dry_run:
                continue
            fmt.write_session(
                dst, mode='3d', units='mm', label_source='annotated', names=names, rig=rig,
                groups=groups, labels=labels,
                skeleton=skeleton, flip_pairs=flip_pairs,
                provenance={
                    'source': f'johnson-mouse/merge/{split}/{trial}',
                    'annotator': '', 'annotator_tool': 'scripts/convert_johnson.py',
                    'points3d_source': 'DLT triangulation of the 16-view 2D labels (derived, '
                                       'not annotated)',
                    'animal_id_source': 'single animal per session',
                })


# the calibration check

def check_reprojection(out: Path, limit: float) -> int:
    """Reproject the written 3D onto the written 2D, through the written calibration.

    Everything is re-read from disk on purpose: a mangled calibration.toml round-trip and a bad
    calibration both surface here, and 1684 px is what the wrong transpose convention looks like.
    """
    ds = fmt.load_dataset(out)
    print(f'\n== reprojection ({out.name})')
    print(f'{"split/session":<34}{"n":>9}{"p50":>8}{"p95":>8}{"p99":>8}{"no 3D":>8}')
    bad = 0
    for split, sessions in ds.sessions.items():
        for sess in sessions:
            errs, few = [], 0
            for gid in sess.groups:
                lab = sess.labels(gid)
                T, K, C = lab.points2d.shape[1], lab.points2d.shape[2], lab.points2d.shape[3]
                p3d = lab.points3d[0].reshape(T * K, 3).astype(np.float64)
                p2d = np.moveaxis(lab.points2d[0], 2, 0).reshape(C, T * K, 2).astype(np.float64)
                # labelled in at least one view but carrying no 3D: seen by <2 cameras, or
                # dropped by the triangulation gate in build_labels
                nviews = np.isfinite(p2d).all(-1).sum(0)
                few += int(((nviews >= 1) & ~np.isfinite(p3d).all(-1)).sum())
                e = np.linalg.norm(_np(sess.rig.cgroup.reprojection_error(p3d, p2d)), axis=-1)
                errs.append(e[np.isfinite(e)])
            e = np.concatenate(errs) if errs else np.zeros(0)
            p50, p95, p99 = (np.percentile(e, [50, 95, 99]) if len(e) else (np.nan,) * 3)
            flag = ' FAIL' if p50 > limit else ''
            bad += p50 > limit
            print(f'{split + "/" + sess.session_id:<34}{len(e):>9}{p50:>8.2f}{p95:>8.2f}'
                  f'{p99:>8.2f}{few:>8}{flag}')
    return int(bad)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--max-gap', type=int, default=8,
                    help='source-frame gap above which a group is cut (default 8)')
    ap.add_argument('--trials', nargs='+', default=None, help='convert only these trials')
    ap.add_argument('--max-groups', type=int, default=None, help='cap groups per session')
    ap.add_argument('--dry-run', action='store_true', help='report counts, write nothing')
    ap.add_argument('--no-check', action='store_true', help='skip the reprojection check')
    ap.add_argument('--no-image-check', action='store_true',
                    help='skip opening one image per group during validation')
    ap.add_argument('--max-reproj-px', type=float, default=2.0,
                    help='fail a session whose median reprojection exceeds this (default 2.0)')
    ap.add_argument('--reject-px', type=float, default=20.0,
                    help='reject a 2D observation from the triangulation above this residual '
                         '(default 20)')
    ap.add_argument('--clean', action='store_true', help='remove the output dir first')
    args = ap.parse_args()

    if args.clean and args.out.exists() and not args.dry_run:
        shutil.rmtree(args.out)
    convert(args.src, args.out, args.max_gap, args.trials, args.max_groups, args.dry_run,
            args.reject_px)
    if args.dry_run:
        return

    errs = fmt.validate_dataset(fmt.load_dataset(args.out),
                                check_images=not args.no_image_check)
    hard = [e for e in errs if 'WARNING' not in e]
    for e in errs:
        print(('  WARN ' if 'WARNING' in e else '  FAIL ') + e)
    print(f'validate: {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')

    bad = 0 if args.no_check else check_reprojection(args.out, args.max_reproj_px)
    sys.exit(1 if hard or bad else 0)


if __name__ == '__main__':
    main()
