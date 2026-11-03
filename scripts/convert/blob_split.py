#!/usr/bin/env python
"""Split roots (train / val / test by WHOLE VIDEO) with re-segmented blob boxes, for the idtracker.ai and UDMT benchmarks.

The eval-only roots of `idtrackerai.py` / `udmt.py` (every video under `test/`) are untouched; this writes `<name>-split`,
whose sessions carry the SAME `centroid` rows (those converters' `video_labels` / `session_labels`) plus `instances.pq`:
- idtracker.ai: every frame is re-segmented with the session's own idtracker.ai v5.2.5 parameters (session.json
  `intensity_ths`, `area_ths`, `resolution_reduction`, `roi_list`, `use_bkg` = median of `number_of_frames_for_background`
  evenly spaced frames; `segmentation.py::process_frame`, incl. 5.2.5's "(0, v) with a background means (v, 255)" rule).
  Each validated centroid (`validated.npy`) takes the blob containing it, else the nearest blob centroid within 0.5 BL.
  A blob owned by exactly one animal, with its centroid within 0.25 BL of that animal's, area <= 2x and longer side
  >= 0.5x the video's median individual blob, gives its bounding box (`blob`). Every other animal-frame (merged /
  crossing blob, no blob) gets a SQUARE of the video's median individual-blob side on the validated centroid (inside
  crossings that centroid is idtracker's interpolation: a placement, not a measurement).
- A `name@start-end` entry is source frames [start, end) of that video: labels are re-indexed to the clip and the clip is
  written losslessly (H.264 qp 0, same yuv420p planes; `trim_video` asserts decode equality) as the group's own pixels, so
  one video can feed two splits (udmt-celegans: train / val are two time ranges of celegans6, test the other video).
- UDMT: only the annotated frames (1 / 150-600) have identities, so only they get boxes. Background = median of 50 evenly
  spaced frames; mask (darker / brighter / |diff| closed 5x5, all opened 3x3) and level are tuned per video on its own
  annotated frames (`udmt_boxes`; a sole blob counts only at 0.5-2x the annotators' box side). Matching and the square
  are as above, with the ownership test 0.5 x the median blob side (the clicked point is not the blob centroid). TRAIN /
  VAL checkpoints are carried +-6 frames along each clean individual's own blob (`propagate`; tracker rows, not clicks).
A camera-frame where every animal has a position and box is `labeled`; any other is `present`. Boxes are clipped to the
image. Pixels are LINKED: idtracker to the eval root's H.264 (`<root>/test/<video>/groups/g0/cam0.mp4`, must exist),
UDMT to the raw mp4. The split and box statistics go to the root's README.md (+ box_stats.json). No unit tests
(converter rule); check with --validate, scripts/dataset_tools/check_status.py and a render.

Usage:
  pixi run python scripts/convert/blob_split.py --dataset idtrackerai-mice --validate
  pixi run python scripts/convert/blob_split.py --dataset udmt-multianimal --only mice7 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from tailcyclenet import format as fmt
from _common import validate_root
import idtrackerai
import udmt

OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
SPLITS = {
    'idtrackerai-mice': {'train': ['mice_2_1', 'mice_2_2', 'mice_4_1'], 'val': ['mice_2_3'],
                         'test': ['mice_2_4', 'mice_4_2']},
    'idtrackerai-zebrafish': {'train': ['zebrafish_2', 'zebrafish_5', 'zebrafish_7', 'zebrafish_10_1', 'zebrafish_10_4'],
                              'val': ['zebrafish_10_2'], 'test': ['zebrafish_8', 'zebrafish_20', 'zebrafish_10_3']},
    'udmt-multianimal': {'train': ['mice3', 'mice5_94hz', 'whitemice5', 'rat1mice2', 'flies12', 'flies17', 'flies18_illum'],
                         'val': ['mice7', 'flies23'], 'test': ['mice5', 'mice10', 'flies18']},
    # two videos: test = the whole unseen celegans7; celegans6 is cut in time (`name@start-end`, half-open source frames)
    # into train (annotations 1-79) and val (80-98), each clip re-encoded losslessly as its own group-local video.
    'udmt-celegans': {'train': ['celegans6@0-11850'], 'val': ['celegans6@11850-14569'], 'test': ['celegans7']},
}
SPECIES = {'idtrackerai-mice': 'mice', 'idtrackerai-zebrafish': 'zebrafish'}
MAX_AREA_X = 2.0
MIN_SIDE_X = 0.5
NEAR_BL = 0.5
IDT_OWN_BL = 0.25
UDMT_OWN_SIDE = 0.5
UDMT_LEVELS = (8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120)
SIZE_BAND = (0.5, 2.0)
UDMT_EXTEND = 6
UDMT_STEP_SIDE = 0.25


def parse_entry(entry: str) -> tuple[str, tuple[int, int] | None]:
    """`name` -> (name, None); `name@start-end` -> (name, (start, end)) source frames, half-open."""
    if '@' not in entry:
        return entry, None
    name, rng = entry.split('@')
    a, b = (int(x) for x in rng.split('-'))
    return name, (a, b)


def session_name(entry: str) -> str:
    """The session directory of a SPLITS entry (`celegans6@0-11850` -> `celegans6_000000-011850`)."""
    name, rng = parse_entry(entry)
    return name if rng is None else f'{name}_{rng[0]:06d}-{rng[1]:06d}'


def trim_video(src: Path, dst: Path, a: int, b: int) -> None:
    """Write source frames [a, b) of `src` to `dst` losslessly (libx264 qp 0, the decoded yuv420p planes re-encoded).

    Asserts the frame count and that every 500th written frame decodes to the same planes as the source."""
    import av
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = {}
    with av.open(str(src)) as ci, av.open(str(dst), 'w') as co:
        si = ci.streams.video[0]
        so = co.add_stream('libx264', rate=si.average_rate)
        so.width, so.height, so.pix_fmt = si.codec_context.width, si.codec_context.height, 'yuv420p'
        so.options = {'qp': '0', 'preset': 'medium'}
        for i, f in enumerate(ci.decode(si)):
            if i >= b:
                break
            if i < a:
                continue
            if f.format.name != 'yuv420p':
                raise RuntimeError(f'{src}: pixel format {f.format.name}, expected yuv420p')
            if (i - a) % 500 == 0:
                kept[i - a] = f.to_ndarray()
            f.pts, f.time_base = i - a, 1 / si.average_rate
            for pk in so.encode(f):
                co.mux(pk)
        for pk in so.encode():
            co.mux(pk)
    with av.open(str(dst)) as c:
        n = 0
        for j, f in enumerate(c.decode(video=0)):
            if j in kept and not np.array_equal(f.to_ndarray(), kept[j]):
                raise RuntimeError(f'{dst}: frame {j} differs from source frame {a + j}')
            n += 1
    if n != b - a:
        raise RuntimeError(f'{dst}: {n} frames written, expected {b - a}')


def clip_labels(v: dict, a: int, b: int, video: Path) -> dict:
    """`udmt.session_labels` output restricted to source frames [a, b), re-indexed to the clip at `video`."""
    lab = v['lab']
    for k in ('points2d', 'vis2d', 'boxes', 'instance'):
        x = getattr(lab, k)
        if x is not None:
            setattr(lab, k, x[:, a:b].copy())
    f = v['frames']
    g = v['group']
    g.n_frames, g.source_frame_start = b - a, a
    g.notes += f'; source frames [{a}, {b}) of {g.source_video}, re-encoded lossless (H.264 qp 0)'
    return dict(v, lab=lab, video=video, frames=f[(f >= a) & (f < b)] - a)


def gray_frames(path: Path, keep=None):
    """Yield (index, uint8 gray frame) in decode order, as OpenCV's BGR -> GRAY would produce it (idtracker reads with
    OpenCV). `keep`: an optional sorted index array -- only those frames are converted, decoding stops after the last."""
    import av
    last = None if keep is None else int(keep[-1])
    want = None if keep is None else set(int(k) for k in keep)
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = 'AUTO'
        for i, f in enumerate(c.decode(st)):
            if want is None or i in want:
                yield i, cv2.cvtColor(f.to_ndarray(format='bgr24'), cv2.COLOR_BGR2GRAY)
            if last is not None and i >= last:
                return


def roi_mask(sj: dict) -> np.ndarray | None:
    """idtracker.ai's `build_ROI_mask_from_list` (5.2.5): polygons / ellipses at the reduced resolution, or None."""
    rl, rr = sj.get('roi_list'), float(sj['resolution_reduction'])
    if rl is None:
        return None
    m = np.zeros((int(sj['original_height'] * rr + 0.5), int(sj['original_width'] * rr + 0.5)), np.uint8)
    for line in ([rl] if isinstance(rl, str) else rl):
        data = json.loads(line[10:].replace("'", '"'))
        if line[2:9] == 'Polygon':
            v = np.asarray(data)
        elif line[2:9] == 'Ellipse':
            v = np.asarray(cv2.ellipse2Poly(data['center'], data['axes'], data['angle'], 0, 360, 2))
        else:
            raise TypeError(line)
        cv2.fillPoly(m, ((v * rr + 0.5).astype(np.int32),), color=255 if line[0] == '+' else 0)
    return m


def blobs(seg: np.ndarray, amin: float, amax: float):
    """External contours of a binary mask with amin <= contourArea <= amax (idtracker's filter).

    Returns (label image int32 with blob i filled as i+1, centroids (B, 2), boxes (B, 4) x0 y0 x1 y1, areas (B,)),
    all in the mask's pixels."""
    cs = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    lab = np.zeros(seg.shape, np.int32)
    cen, box, area = [], [], []
    for cn in cs:
        a = cv2.contourArea(cn)
        if not (amin <= a <= amax) or a <= 0:
            continue
        m = cv2.moments(cn)
        x, y, w, h = cv2.boundingRect(cn)
        cv2.drawContours(lab, [cn], -1, len(cen) + 1, thickness=-1)
        cen.append((m['m10'] / m['m00'], m['m01'] / m['m00']))
        box.append((x, y, x + w, y + h))
        area.append(a)
    return lab, np.asarray(cen, np.float64).reshape(-1, 2), np.asarray(box, np.float64).reshape(-1, 4), np.asarray(area)


def match(points: np.ndarray, lab: np.ndarray, cen: np.ndarray, near: float):
    """Each animal's blob: the one containing its point, else the nearest centroid within `near` (mask px); -1 if none.

    Returns (owner (N,), number of animals sharing the owner blob (N,), point -> owner-centroid distance (N,))."""
    N = len(points)
    own = np.full(N, -1)
    H, W = lab.shape
    for a, p in enumerate(points):
        if not np.isfinite(p).all():
            continue
        x, y = int(round(p[0])), int(round(p[1]))
        j = lab[y, x] - 1 if (0 <= x < W and 0 <= y < H) else -1
        if j < 0 and len(cen):
            d = np.linalg.norm(cen - p, axis=1)
            j = int(np.argmin(d)) if d.min() <= near else -1
        own[a] = j
    count = np.array([(own == j).sum() if j >= 0 else 0 for j in own])
    dist = np.array([np.linalg.norm(cen[j] - points[a]) if j >= 0 else np.inf for a, j in enumerate(own)])
    return own, count, dist


def finalize(pos, box, area, count, dist, own_tol, W, H):
    """Per animal-frame boxes from the matched blobs: blob box for a clean individual, else the median-side square.

    Inputs: pos (F, N, 2) full-res centroids; box (F, N, 4) owner blob boxes (full-res, NaN if none); area (F, N) owner
    blob areas; count / dist (F, N) from `match` (dist in full-res px); own_tol (full-res px). Returns (boxes (F, N, 4),
    kind (F, N) 0 = no position, 1 = blob, 2 = square, stats dict)."""
    single = (count == 1) & (dist <= own_tol)
    med_area = float(np.median(area[single])) if single.any() else np.inf
    allside = np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1])
    indiv = single & (area <= MAX_AREA_X * med_area)
    side = float(np.median(allside[indiv])) if indiv.any() else float('nan')
    indiv &= allside >= MIN_SIDE_X * side
    out = np.full(box.shape, np.nan, np.float32)
    out[indiv] = box[indiv]
    fin = np.isfinite(pos).all(-1)
    sq = fin & ~indiv
    c = pos[sq]
    out[sq] = np.concatenate([c - side / 2, c + side / 2], -1)
    out[..., [0, 2]] = np.clip(out[..., [0, 2]], 0, W)
    out[..., [1, 3]] = np.clip(out[..., [1, 3]], 0, H)
    kind = np.where(indiv & fin, 1, np.where(sq, 2, 0))
    return out, kind, {'median_blob_area_px': med_area, 'square_side_px': side,
                       'blob_frac': float((kind == 1).sum() / max(fin.sum(), 1)),
                       'square_frac': float((kind == 2).sum() / max(fin.sum(), 1))}


def idtracker_boxes(v: dict):
    """Re-segment every frame of an idtracker.ai video with its own session parameters and box each animal."""
    sj, traj, bl = v['session'], v['traj'], v['body_length']
    rr = float(sj['resolution_reduction'])
    T, N, _ = traj.shape
    W, H = int(sj['original_width']), int(sj['original_height'])
    ith, (amin, amax) = sj['intensity_ths'], sj['area_ths']
    roi = roi_mask(sj)
    bkg = None
    if sj['use_bkg']:
        idx = np.linspace(0, T - 1, int(sj['number_of_frames_for_background']), dtype=int)
        stack = np.stack([f for _, f in gray_frames(v['avi'], idx)])
        stat = {'median': np.median, 'mean': np.mean, 'max': np.max, 'min': np.min}[sj['background_subtraction_stat']]
        bkg = stat(stack, axis=0).astype(np.uint8)
        if rr != 1:
            bkg = cv2.resize(bkg, None, fx=rr, fy=rr, interpolation=cv2.INTER_AREA)
        if ith[0] == 0:
            ith = (ith[1], 255)
    box = np.full((T, N, 4), np.nan)
    area = np.full((T, N), np.nan)
    count = np.zeros((T, N), int)
    dist = np.full((T, N), np.inf)
    n_blobs = np.zeros(T, int)
    for t, fr in gray_frames(v['avi']):
        if rr != 1:
            fr = cv2.resize(fr, None, fx=rr, fy=rr, interpolation=cv2.INTER_AREA)
        seg = cv2.inRange(fr, *ith) if bkg is None else (cv2.absdiff(bkg, fr) > ith[0]).astype(np.uint8)
        if roi is not None:
            cv2.bitwise_and(seg, roi, seg)
        lab, cen, bx, ar = blobs(seg, amin, amax)
        n_blobs[t] = len(cen)
        own, cnt, dd = match(traj[t] * rr, lab, cen, NEAR_BL * bl * rr)
        ok = own >= 0
        box[t, ok] = bx[own[ok]] / rr
        area[t, ok] = ar[own[ok]]
        count[t], dist[t] = cnt, dd / rr
    if t != T - 1:
        raise RuntimeError(f'{v["avi"]}: decoded {t + 1} frames, expected {T}')
    out, kind, st = finalize(traj, box, area, count, dist, IDT_OWN_BL * bl, W, H)
    st['frames_with_N_blobs'] = float((n_blobs == N).mean())
    st['own_dist_median_px'] = float(np.median(dist[count == 1])) if (count == 1).any() else float('nan')
    return out, kind, st


def udmt_boxes(v: dict, extend: int = 0):
    """Background-subtract + threshold a UDMT video's annotated frames (level tuned per video) and box each animal.

    The score of a (mask, level) is the share of annotated animals that solely own an ANIMAL-SIZED blob (a dark spot on a
    white mouse is a sole owner too, and is not a box). Of the best mask, the LOWEST level within 0.01 of the best score
    is kept: a higher level erodes the blob to its core.

    `extend` > 0 also returns `ext` {frame: (N, 2)} -- each clean individual's checkpoint carried up to `extend` frames
    each way by `propagate`; NaN where it stopped. Otherwise `ext` is None."""
    lab_ = v['lab']
    frames = v['frames']
    T = v['group'].n_frames
    pts = lab_.points2d[:, frames, 0, 0, :].transpose(1, 0, 2).astype(np.float64)
    gt_side = float(np.nanmedian(lab_.boxes[:, frames, 0, 2] - lab_.boxes[:, frames, 0, 0]))
    near = (frames[:, None] + np.arange(-extend, extend + 1)[None]).ravel()
    idx = np.unique(np.concatenate([np.linspace(0, T - 1, 50, dtype=int), near[(near >= 0) & (near < T)]]))
    got = dict(gray_frames(v['video'], idx))
    bkg = np.median(np.stack([got[i] for i in np.linspace(0, T - 1, 50, dtype=int)]), 0).astype(np.int16)
    imgs = [got[int(f)].astype(np.int16) for f in frames]
    H, W = imgs[0].shape
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)

    def mask(im, mode, level):
        """Binary foreground of one frame for a mask mode and level."""
        if mode == 'absdiff+close':
            seg = cv2.morphologyEx((np.abs(im - bkg) > level).astype(np.uint8), cv2.MORPH_OPEN, k3)
            return cv2.morphologyEx(seg, cv2.MORPH_CLOSE, k5)
        sign = -1 if mode == 'darker' else 1
        return cv2.morphologyEx(((sign * (im - bkg)) > level).astype(np.uint8), cv2.MORPH_OPEN, k3)

    def run(mode, level):
        """Match every annotated frame's animals to the blobs of one (mode, level): (box, area, count, dist)."""
        box = np.full(pts.shape[:2] + (4,), np.nan)
        area = np.full(pts.shape[:2], np.nan)
        count = np.zeros(pts.shape[:2], int)
        dist = np.full(pts.shape[:2], np.inf)
        for i, im in enumerate(imgs):
            lab, cen, bx, ar = blobs(mask(im, mode, level), 15, np.inf)
            own, cnt, dd = match(pts[i], lab, cen, NEAR_BL * gt_side)
            ok = own >= 0
            box[i, ok], area[i, ok] = bx[own[ok]], ar[own[ok]]
            count[i], dist[i] = cnt, dd
        return box, area, count, dist

    tried = []
    for mode in ('darker', 'brighter', 'absdiff+close'):
        for level in UDMT_LEVELS:
            r = run(mode, level)
            side = np.maximum(r[0][..., 2] - r[0][..., 0], r[0][..., 3] - r[0][..., 1])
            score = float(((r[2] == 1) & (side >= SIZE_BAND[0] * gt_side) & (side <= SIZE_BAND[1] * gt_side)).mean())
            tried.append((score, mode, level, r))
    top = max(t[0] for t in tried)
    best_mode = max(tried, key=lambda t: t[0])[1]
    score, mode, level, (box, area, count, dist) = min(
        (t for t in tried if t[1] == best_mode and t[0] >= top - 0.01), key=lambda t: t[2])
    single = count == 1
    sides = np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1])[single]
    own_tol = UDMT_OWN_SIDE * float(np.median(sides)) if sides.size else 0.0
    out, kind, st = finalize(pts, box, area, count, dist, own_tol, W, H)
    st.update({'mask': mode, 'level': level, 'sized_sole_owner_frac': score,
               'gt_box_px': gt_side})
    ext = None
    if extend > 0:
        seg = {f: blobs(mask(im.astype(np.int16), mode, level), 15, np.inf) for f, im in got.items()}
        ext = propagate(seg, frames, pts, kind, extend, UDMT_STEP_SIDE * st['square_side_px'], st['median_blob_area_px'], T)
        n = sum(int(np.isfinite(p).all(-1).sum()) for p in ext.values())
        st.update({'extend': extend, 'propagated_rows': n,
                   'propagated_per_checkpoint_animal': n / max(int((kind == 1).sum()), 1)})
    return out, kind, st, ext


def propagate(seg: dict, frames, pts, kind, extend: int, step_tol: float, med_area: float, T: int) -> dict:
    """Carry each clean individual's annotated point up to `extend` frames each way along its own blob.

    At a checkpoint an animal with a `blob` box (kind 1) keeps offset = clicked point - blob centroid. Each step takes the
    ONE blob whose centroid is within `step_tol` (0.25 median blob side) of the previous centroid, with area within 0.5-2x the video's median
    individual-blob area and claimed by no other animal at that frame; the label is that centroid + the offset. Zero or
    several candidates, a shared blob or a size jump stops that animal in that direction (a merge ends the carry).
    Inputs: seg {frame: `blobs` output}; frames (F,) checkpoints; pts (F, N, 2); kind (F, N). Returns {frame: (N, 2)}."""
    out = {}
    for i, f in enumerate(frames):
        lab0, cen0, _, _ = seg[int(f)]
        if not len(cen0):
            continue
        own, _, _ = match(pts[i], lab0, cen0, step_tol)
        for d in (-1, 1):
            prev = {a: cen0[own[a]] for a in range(len(own)) if kind[i, a] == 1 and own[a] >= 0}
            off = {a: pts[i, a] - prev[a] for a in prev}
            for k in range(1, extend + 1):
                t = int(f) + d * k
                if t < 0 or t >= T or not prev:
                    break
                _, cen, _, ar = seg[t]
                pick = {}
                for a, p in prev.items():
                    c = np.flatnonzero((np.linalg.norm(cen - p, axis=1) <= step_tol)
                                       & (ar >= 0.5 * med_area) & (ar <= MAX_AREA_X * med_area)) if len(cen) else []
                    if len(c) == 1:
                        pick[a] = int(c[0])
                taken = [j for j in pick.values()]
                prev = {a: cen[j] for a, j in pick.items() if taken.count(j) == 1}
                row = out.setdefault(t, np.full(pts.shape[1:], np.nan))
                for a, c in prev.items():
                    row[a] = c + off[a]
    return out


def build(dataset: str, split: str, name: str, out: Path, dry: bool) -> dict:
    """Convert one video into `<out>/<split>/<name>/` with boxes; returns its box statistics."""
    if dataset in SPECIES:
        raw, eval_root, _ = idtrackerai.SPECIES[SPECIES[dataset]]
        v = idtrackerai.video_labels(raw, name)
        boxes, kind, st = idtracker_boxes(v)
        pixels = eval_root / 'test' / name / 'groups' / 'g0' / 'cam0.mp4'
        if not pixels.exists():
            raise FileNotFoundError(f'{pixels}: run idtrackerai.py --species {SPECIES[dataset]} first')
        frames = np.arange(v['group'].n_frames)
    else:
        name, rng = parse_entry(name)
        v = udmt.session_labels(name)
        if rng is not None:
            clip = out / split / session_name(f'{name}@{rng[0]}-{rng[1]}') / 'groups' / 'g0' / 'cam0.mp4'
            if dry:
                clip = Path(tempfile.mkdtemp()) / 'cam0.mp4'
            trim_video(v['video'], clip, *rng)
            v = clip_labels(v, *rng, clip)
            name = session_name(f'{name}@{rng[0]}-{rng[1]}')
        boxes, kind, st, ext = udmt_boxes(v, UDMT_EXTEND if split in ('train', 'val') else 0)
        pixels, frames = v['video'], v['frames']
        for t, row in (ext or {}).items():
            ok = np.isfinite(row).all(-1) & (v['lab'].vis2d[:, t, 0, 0] == fmt.UNLABELED)
            v['lab'].points2d[ok, t, 0, 0] = row[ok]
            v['lab'].vis2d[ok, t, 0, 0] = fmt.PROJECTED
    lab = v['lab']
    N = len(lab.animal_ids)
    lab.boxes = np.full((N, v['group'].n_frames, 1, 4), np.nan, np.float32)
    lab.instance = np.full((N, v['group'].n_frames, 1), fmt.INST_NONE, np.int8)
    lab.boxes[:, frames, 0] = boxes.transpose(1, 0, 2)
    complete = (kind > 0).all(1)
    lab.instance[:, frames, 0] = np.where(complete, fmt.INST_LABELED, fmt.INST_PRESENT)[None, :]
    st.update({'video': name, 'split': split, 'frames_boxed': int(len(frames)),
               'frames_incomplete': int((~complete).sum()), 'n_animals': N})
    print(f'{dataset}/{split}/{name}: ' + ' '.join(f'{k}={x:.4g}' if isinstance(x, float) else f'{k}={x}'
                                                    for k, x in st.items()), flush=True)
    if dry:
        return st
    v['group'].notes += (f'; boxes=blob-resegmented (blob {st["blob_frac"]:.3f} / square {st["square_frac"]:.3f}, '
                         f'square side {st["square_side_px"]:.1f}px)')
    prov = dict(v['provenance'], converter=v['provenance']['converter'] + ' + scripts/convert/blob_split.py')
    sess = out / split / name
    fmt.write_session(sess, mode='2d', units='px', label_source='annotated', names=['centroid'], rig=v['rig'],
                      groups={'g0': v['group']}, labels={'g0': lab}, flip_pairs=[], provenance=prov)
    (sess / 'groups' / 'g0').mkdir(parents=True, exist_ok=True)
    if pixels != sess / 'groups' / 'g0' / 'cam0.mp4':
        fmt.link(sess / 'groups' / 'g0' / 'cam0.mp4', pixels)
    return st


def write_readme(dataset: str, out: Path, stats: list[dict]) -> None:
    """The split, the box rule and the per-video box statistics, as the root's README.md."""
    rows = '\n'.join(f'| {s["split"]} | {s["video"]} | {s["n_animals"]} | {s["frames_boxed"]} | {s["blob_frac"]:.3f} | '
                     f'{s["square_frac"]:.3f} | {s["square_side_px"]:.1f} | {s["frames_incomplete"]} |' for s in stats)
    sp = SPLITS[dataset]
    clipped = any('@' in n for names in sp.values() for n in names)
    text = (f'# {dataset}-split\n\nThe `{dataset}` benchmark split by WHOLE VIDEO (every test video unseen in '
            f'training' + ('; an `@start-end` entry is a half-open source-frame range of one video, so train and val may '
                           'be two time ranges of the same recording' if clipped else '; val videos unseen too') + '), with detector boxes in `instances.pq`. Written by `scripts/convert/blob_split.py` (see its '
            f'docstring for the box rule); keypoint rows are identical to the eval-only root `{dataset}` (one '
            f'`centroid` keypoint).\n\n## Split\n\n' + '\n'.join(f'- **{k}**: {", ".join(sp[k])}' for k in sp)
            + '\n\n## Boxes (per video)\n\n`blob` = the re-segmented blob box of a clean individual; `square` = a '
            'fixed square (the video\'s median individual-blob side) on the identity centroid (merged / crossing '
            'blob, or no blob).\n\n| split | video | N | frames boxed | blob | square | square side px | '
            'incomplete frames |\n|---|---|---|---|---|---|---|---|\n' + rows + '\n'
            + ('' if dataset in SPECIES else
               f'\n## Train / val label extension\n\nUDMT annotates 1 frame in 150-600, and one keypoint at one frame cannot '
               f'train (a window needs >= 2 finite coordinates). Each train / val checkpoint is therefore carried '
               f'+-{UDMT_EXTEND} frames along the clean individual\'s own blob (`propagate`): blob centroid + the '
               f'checkpoint\'s clicked-point offset, stopping at any merge or ambiguity. Those rows are tracker-derived, '
               f'not human clicks. TEST sessions keep the human checkpoints only.\n'))
    (out / 'README.md').write_text(text)
    (out / 'box_stats.json').write_text(json.dumps(stats, indent=1))


def _one(args):
    """`build` on one argument tuple (a picklable ProcessPoolExecutor target)."""
    return build(*args)


def main() -> int:
    """Parse the CLI, convert the requested videos in parallel, write the README, optionally validate."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=sorted(SPLITS), required=True)
    ap.add_argument('--only', nargs='*', help='subset of video names')
    ap.add_argument('--out', type=Path, default=None, help='default: tailcycle-datasets/<dataset>-split')
    ap.add_argument('--clean', action='store_true', help='remove the output first')
    ap.add_argument('--dry-run', action='store_true', help='report, write nothing')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--no-image-check', action='store_true')
    a = ap.parse_args()
    out = a.out or OUT / f'{a.dataset}-split'
    if a.clean and out.exists() and not a.dry_run:
        shutil.rmtree(out)
    todo = [(a.dataset, sp, n, out, a.dry_run) for sp, names in SPLITS[a.dataset].items() for n in names
            if not a.only or n in a.only or parse_entry(n)[0] in a.only]
    with ProcessPoolExecutor(max(1, min(a.jobs, len(todo)))) as ex:
        stats = list(ex.map(_one, todo))
    if not a.dry_run and not a.only:
        write_readme(a.dataset, out, stats)
    if a.validate and not a.dry_run:
        return 1 if validate_root(out, check_images=not a.no_image_check) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
