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
  demands exactly one camera. So the 3D layer here is DERIVED -- DLT-triangulated from the 16-view
  2D, which reprojects at ~0.6 px. `provenance.points3d_source` says so, and `--check` measures it.

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


# ----------------------------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------
# source reading
# ----------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------------------------

def build_labels(d: dict, run: list[int], views: dict[int, dict[str, int]], rig: fmt.Rig,
                 K: int, max_reproj: float) -> tuple[fmt.Labels, int]:
    """Dense arrays for one group. 2D + boxes from the JSON, 3D triangulated from the 2D.

    An image with no annotation leaves every cell UNLABELED (no row) rather than `present`: the
    source records no determination about those views, and inventing one would be an annotation
    nobody made.

    The 3D layer is derived, so it carries a quality gate the 2D does not: a point whose median
    reprojection exceeds `max_reproj` gets no 3D row. 240 points -- 10 consecutive framesets at
    the head of 2025_02_12, where all 16 views disagree by 200+ px -- fail it, against a p99 of
    1.3 px everywhere else. The 2D behind them is still exported verbatim; it is the source's
    annotation and not ours to censor.
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
            # flag 0 is a real (if near-extinct: 18 rows in train) assessed-but-absent mark
            lab.vis2d[0, t, ~seen, ci] = fmt.MISSING
            lab.vis2d[0, t, seen, ci] = fmt.VISIBLE
            lab.points2d[0, t, seen, ci] = kp[seen, :2].astype(np.float32)

            x, y, w, h = (float(v) for v in a['bbox'])
            lab.boxes[0, t, ci] = (x, y, x + w, y + h)
            lab.instance[0, t, ci] = fmt.INST_LABELED

    # 2D -> 3D. aniposelib's triangulate is NaN-safe and returns NaN below 2 views.
    p2d = np.moveaxis(lab.points2d[0], 2, 0).reshape(C, T * K, 2).astype(np.float64)
    p3d = _np(rig.cgroup.triangulate(p2d, progress=False))
    e = np.linalg.norm(_np(rig.cgroup.reprojection_error(p3d, p2d)), axis=-1)
    with np.errstate(invalid='ignore'):
        med = np.nanmedian(np.where(np.isfinite(e), e, np.nan), axis=0)
    p3d = p3d.reshape(T, K, 3)
    finite = np.isfinite(p3d).all(-1)
    ok = finite & (med.reshape(T, K) <= max_reproj)
    lab.vis3d[0][ok] = fmt.VISIBLE
    lab.points3d[0][ok] = p3d[ok].astype(np.float32)
    return lab, int(finite.sum() - ok.sum())


def _np(x) -> np.ndarray:
    """aniposelib is on the pytorch branch: its outputs carry requires_grad."""
    return np.asarray(x.detach().cpu() if hasattr(x, 'detach') else x, dtype=np.float64)


def link(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


# ----------------------------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------------------------

def convert(src: Path, out: Path, max_gap: int, only: list[str] | None, max_groups: int | None,
            dry_run: bool, max_reproj: float) -> None:
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
                labels[gid], n = build_labels(d, run, fsets, rig, K, max_reproj)
                gated += n
                if dry_run:
                    continue
                for cam in rig.names:
                    cdir = dst / 'groups' / gid / cam
                    cdir.mkdir(parents=True, exist_ok=True)
                    for t, frame in enumerate(run):
                        s = (src / split / d['_imgs'][fsets[frame][cam]]['file_name']).resolve()
                        link(cdir / f'{t:06d}.jpg', s)

            n_lab = sum(int((lab.vis2d != fmt.UNLABELED).any(2).sum()) for lab in labels.values())
            note = f', {gated} point(s) failed the {max_reproj:g}px 3D gate' if gated else ''
            print(f'   {split}/{trial}: {len(groups)} group(s), '
                  f'{sum(g.n_frames for g in groups.values())} frames, {len(rig)} cams, '
                  f'{n_lab} labelled views{note}')
            if dry_run:
                continue
            fmt.write_session(
                dst, mode='3d', units='mm', names=names, rig=rig, groups=groups, labels=labels,
                skeleton=skeleton, flip_pairs=flip_pairs,
                provenance={
                    'source': f'johnson-mouse/merge/{split}/{trial}',
                    'annotator': '', 'annotator_tool': 'scripts/convert_johnson.py',
                    'points3d_source': 'DLT triangulation of the 16-view 2D labels (derived, '
                                       'not annotated)',
                    'animal_id_source': 'single animal per session',
                })


# ----------------------------------------------------------------------------------------------
# the calibration check
# ----------------------------------------------------------------------------------------------

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
    ap.add_argument('--gate-reproj-px', type=float, default=10.0,
                    help='drop the 3D row for a point reprojecting worse than this (default 10)')
    ap.add_argument('--clean', action='store_true', help='remove the output dir first')
    args = ap.parse_args()

    if args.clean and args.out.exists() and not args.dry_run:
        shutil.rmtree(args.out)
    convert(args.src, args.out, args.max_gap, args.trials, args.max_groups, args.dry_run,
            args.gate_reproj_px)
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
