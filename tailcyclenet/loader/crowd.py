"""The all-animals loader extras of the v7 2D MoE (E2's training data) and the crowded-window
sampler. Pure functions of a realised item; `PoseDataset` calls them AFTER its own draws, so a run
without a set decoder (and `hard_frac = 0`) consumes exactly the stock RNG stream.

`set_targets` -> one dict per item (collated to `batch.set_targets`), every animal in the crop in
the item's crop-pixel frame, own animal first, the rest by distance from the crop centre:
  all (A,T,K,2)        coords (NaN where unlabelled / outside the crop)
  assessed (A,T)       the row has a label somewhere in the source frame at t
  ok (A,)              a row exists
  all_assessed (T,)    every animal in the window was assessed at t (and none beyond A)
  anchors (A,T,4)      the row's TIGHT GT instance box (instances.pq; else its keypoint extent +5%),
                       +-10% per edge per frame; non-target anchors dropped w.p. 0.15 (free queries
                       must pick up animals the detector missed); NaN outside the crop
  prior (A,K,2)        a position prior per anchored row at its best-labelled frame (whole-item
                       dropout 0.5, 5 px whole-body offset, 2.5 px per-keypoint noise), NaN = none
  prior_t (A,)         the frame that prior describes
(B1p's recipe, report 76 crowd-detr; fixed, not configured.)

The row count is the model's M (`PoseDataset.set_rows`): 8 for a 2D root's MoE, 4 (`set_decoder.M_ONLY`)
for a set-only run (PBM4) or a 3D root's MoE. A 3D item's targets are `set_targets_3d`: every animal's WORLD points in the item's
gauge-rotated frame (`View.all_world`), and PER-CAMERA anchors (A,T,C,4) = the projection extent
of the row's keypoints, each side padded by `[data].crop_pad_frac` x the longer side, squared (the
crop rule's box without the floor; `square_pad`). A 2D item (true 2D, or `prob_2d_only`) takes
the 2D targets at the same row count.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from ..data.camera import apply_affine, mask_outside

# A_MAX / N_ANCHORED are set_decoder.M / set_decoder.NA; HARD_BASE is the natural crowded share
# the rejection rate is computed against
A_MAX = 8
N_ANCHORED = 4
ANCHOR_JITTER = 0.1
ANCHOR_DROP = 0.15
PRIOR_DROP = 0.5
PRIOR_NOISE_PX = 2.5
PRIOR_OFFSET_PX = 5.0
HARD_BASE = 0.34
HARD_TRIES = 12


def _box_to_crop(b, view):
    """A source-px xyxy box -> the axis-aligned extent of its corners in the view's crop px."""
    c = np.array([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]], np.float64)
    rot = view.rotation[0]
    if rot is not None:
        Mr = np.asarray(rot[0], np.float64)
        c = c @ Mr[:, :2].T + Mr[:, 2]
    box0 = np.asarray(view.boxes[0], np.float64)[:2]
    sc = np.asarray(view.scale[0], np.float64).reshape(-1)
    q = (c - box0) * sc
    return np.r_[q.min(0), q.max(0)]


def _jit(b, rng, j):
    """Each edge moved by U(-j, j) x the box side (no draw without rng or a finite box)."""
    if rng is None or j <= 0 or not np.isfinite(b).all():
        return b
    w, h = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
    return b + np.array([rng.uniform(-j, j) * w, rng.uniform(-j, j) * h,
                         rng.uniform(-j, j) * w, rng.uniform(-j, j) * h])


def set_targets(ds, sel, view, rng, train):
    """E2's per-item targets (module docstring): 2D, or (a set-only run) `set_targets_3d`."""
    rows = getattr(ds, 'set_rows', 0) or A_MAX
    if view.r == 3:
        return set_targets_3d(ds, sel, view, rng, train, rows)
    return _set_targets_2d(ds, sel, view, rng, train, rows)


def _set_targets_2d(ds, sel, view, rng, train, A_MAX):
    """The 2D targets at `A_MAX` rows."""
    lab, frames = sel.lab, sel.frames
    T, K = len(frames), sel.n_keypoints
    allc = torch.full((A_MAX, T, K, 2), float('nan'))
    ass = torch.zeros((A_MAX, T), dtype=torch.bool)
    ok = torch.zeros(A_MAX, dtype=torch.bool)
    abox = torch.full((A_MAX, T, 4), float('nan'))
    box, scale, cam = view.boxes[0], view.scale[0], view.cgroup[0]
    W, H = float(cam['size'][0]), float(cam['size'][1])
    perm = view.kpt_perm
    flipped = perm is not None and not torch.equal(perm, torch.arange(len(perm)))
    rows = []
    for r in range(lab.n_animals):
        raw = ds._hybrid_2d_coords(lab, r, frames, sel.cgroup[0], sel.cam_ix[0])
        if raw is None:
            continue
        a_src = torch.isfinite(raw).all(-1).any(-1)
        raw = apply_affine(raw, view.rotation[0])
        if flipped:
            raw = raw[:, perm]
        cfull = (raw - torch.as_tensor(box[:2]).to(raw.dtype)) * torch.as_tensor(scale, dtype=raw.dtype)
        c = mask_outside(cfull, cam['size'])
        n_in = torch.isfinite(c).all(-1).sum(-1)
        d = float('inf')
        if (n_in >= 1).any():
            cen = torch.nanmean(torch.where(torch.isfinite(c), c, torch.nan).reshape(-1, 2), 0)
            d = float((cen - torch.tensor([W / 2, H / 2])).norm())
        if r == sel.animal:
            d = -1.0
        ab = np.full((T, 4), np.nan)
        for t, f in enumerate(frames):
            b = None
            if lab.boxes is not None:
                bb = np.asarray(lab.boxes[r, int(f), sel.cam_ix[0]], np.float64)
                if np.isfinite(bb).all():
                    b = _box_to_crop(bb, view)
            if b is None:
                p = cfull[t].numpy()
                fm = np.isfinite(p).all(-1)
                if fm.sum() >= 2:
                    q = p[fm]
                    lo, hi = q.min(0), q.max(0)
                    pad = 0.05 * (hi - lo)
                    b = np.r_[lo - pad, hi + pad]
            if b is not None:
                ab[t] = b
        rows.append((d, r, c, a_src, ab))
    rows.sort(key=lambda z: z[0])
    extra_in = torch.zeros(T, dtype=torch.bool)
    jr = rng if train else None
    for i, (d, r, c, a_src, ab) in enumerate(rows):
        if i >= A_MAX:
            extra_in |= torch.isfinite(c).all(-1).any(-1)
            continue
        allc[i] = c
        ass[i] = a_src
        ok[i] = True
        inside = (np.isfinite(ab).all(-1) & (ab[:, 2] > 0) & (ab[:, 0] < W)
                  & (ab[:, 3] > 0) & (ab[:, 1] < H))
        if not inside.any():
            continue
        a1 = np.stack([_jit(b, jr, ANCHOR_JITTER) for b in ab])
        if train and i > 0 and rng.random() < ANCHOR_DROP:
            a1[:] = np.nan
        abox[i] = torch.from_numpy(a1).float()
    all_ass = torch.ones(T, dtype=torch.bool)
    for d, r, c, a_src, ab in rows:
        all_ass &= a_src
    all_ass &= ~extra_in
    kpri = torch.full((A_MAX, K, 2), float('nan'))
    kpt = torch.zeros(A_MAX, dtype=torch.long)
    drop_all = train and rng.random() < PRIOR_DROP
    for i in range(N_ANCHORED):
        if drop_all or not torch.isfinite(abox[i]).all(-1).any():
            continue
        cnt = torch.isfinite(allc[i]).all(-1).sum(-1)
        if int(cnt.max()) < 2:
            continue
        t0 = int(torch.argmax(cnt))
        p = allc[i, t0].clone()
        if train:
            p = (p + torch.from_numpy(rng.normal(0, PRIOR_OFFSET_PX / np.sqrt(2), 2)).float()
                 + torch.from_numpy(rng.normal(0, PRIOR_NOISE_PX, (K, 2))).float())
        kpri[i] = p
        kpt[i] = t0
    return {'all': allc, 'assessed': ass, 'ok': ok, 'all_assessed': all_ass, 'anchors': abox,
            'prior': kpri, 'prior_t': kpt}


def square_pad(b, pad):
    """xyxy -> the square about its centre whose sides are the longer side + 2 x pad x that side."""
    m = max(b[2] - b[0], b[3] - b[1])
    cx, cy, hs = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, m / 2 + pad * m
    return np.array([cx - hs, cy - hs, cx + hs, cy + hs])


def set_targets_3d(ds, sel, view, rng, train, A_MAX):
    """A 3D item's set-only targets (module docstring): `all` (A,T,K,3) world, `anchors`
    (A,T,C,4) crop px per camera, `prior` (A,K,3) world (noise scaled by the row's extent / 100,
    ~ world units per crop px). An animal counts only where >= 2 of its keypoints project inside
    some camera's crop."""
    from tracktail.model.cube import project_points_torch
    lab, frames = sel.lab, sel.frames
    T, K = len(frames), sel.n_keypoints
    cg = view.cgroup
    C = len(cg)
    allr = view.all_world
    if allr is None:
        allr = torch.as_tensor(lab.points3d[:, frames], dtype=torch.float32)
    pad = float(ds.cfg.crop_pad_frac)
    allc = torch.full((A_MAX, T, K, 3), float('nan'))
    ass = torch.zeros((A_MAX, T), dtype=torch.bool)
    ok = torch.zeros(A_MAX, dtype=torch.bool)
    abox = torch.full((A_MAX, T, C, 4), float('nan'))
    rows = []
    for r in range(min(lab.n_animals, allr.shape[0])):
        c = allr[r]
        if not torch.isfinite(c).any():
            continue
        p = project_points_torch(cg, c)
        inside = torch.zeros(p.shape[:3], dtype=torch.bool)
        for ci, cam in enumerate(cg):
            W, H = float(cam['size'][0]), float(cam['size'][1])
            inside[ci] = (torch.isfinite(p[ci]).all(-1) & (p[ci, ..., 0] >= 0) & (p[ci, ..., 0] < W)
                          & (p[ci, ..., 1] >= 0) & (p[ci, ..., 1] < H))
        if not bool((inside.sum(-1).amax(0) >= 2).any()):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d = (-1.0 if r == sel.animal else
                 float((torch.nanmean(c.reshape(-1, 3), 0)
                        - torch.nanmean(view.coords.reshape(-1, 3), 0)).norm()))
        ab = np.full((T, C, 4), np.nan)
        for ci in range(C):
            for t in range(T):
                q = p[ci, t][torch.isfinite(p[ci, t]).all(-1)].numpy()
                if len(q) >= 2:
                    ab[t, ci] = square_pad(np.r_[q.min(0), q.max(0)], pad)
        rows.append((d, r, c, torch.isfinite(c).all(-1).any(-1), ab))
    rows.sort(key=lambda z: z[0])
    jr = rng if train else None
    for i, (d, r, c, a_src, ab) in enumerate(rows[:A_MAX]):
        allc[i] = c
        ass[i] = a_src
        ok[i] = True
        a1 = np.stack([[_jit(b, jr, ANCHOR_JITTER) for b in abt] for abt in ab])
        if train and i > 0 and rng.random() < ANCHOR_DROP:
            a1[:] = np.nan
        abox[i] = torch.from_numpy(a1).float()
    all_ass = torch.ones(T, dtype=torch.bool)
    for d, r, c, a_src, ab in rows:
        all_ass &= a_src
    if len(rows) > A_MAX:
        all_ass[:] = False
    kpri = torch.full((A_MAX, K, 3), float('nan'))
    kpt = torch.zeros(A_MAX, dtype=torch.long)
    drop_all = train and rng.random() < PRIOR_DROP
    for i in range(min(N_ANCHORED, A_MAX)):
        if drop_all or not torch.isfinite(abox[i]).all(-1).any():
            continue
        cnt = torch.isfinite(allc[i]).all(-1).sum(-1)
        if int(cnt.max()) < 2:
            continue
        t0 = int(torch.argmax(cnt))
        pp = allc[i, t0].clone()
        if train:
            fin = torch.isfinite(pp).all(-1)
            ext = float((pp[fin].amax(0) - pp[fin].amin(0)).max()) if fin.sum() >= 2 else 0.0
            sc = ext / 100.0
            pp = (pp + torch.from_numpy(rng.normal(0, PRIOR_OFFSET_PX * sc / np.sqrt(3), 3)).float()
                  + torch.from_numpy(rng.normal(0, PRIOR_NOISE_PX * sc, (K, 3))).float())
        kpri[i] = pp
        kpt[i] = t0
    return {'all': allc, 'assessed': ass, 'ok': ok, 'all_assessed': all_ass, 'anchors': abox,
            'prior': kpri, 'prior_t': kpt}


def is_crowded(lab, a, frames):
    """Is another animal within 0.8 x this animal's extent (centroid distance) in this window?"""
    if lab.n_animals < 2:
        return False
    if lab.points2d is not None and np.isfinite(lab.points2d[a][frames]).any():
        P = lab.points2d[:, :, :, 0][:, frames]
    else:
        P = lab.points3d[:, frames]
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cen = np.nanmean(P, 2)
        ext = np.nanmax(np.nanmax(P[a], 1) - np.nanmin(P[a], 1), -1)
    for o in range(lab.n_animals):
        if o != a and np.any(np.linalg.norm(cen[o] - cen[a], axis=-1) < 0.8 * ext):
            return True
    return False


def keep_probability(hard_frac):
    """P(keep an uncrowded draw) that makes crowded windows `hard_frac` of the stream."""
    return min(1.0, HARD_BASE / max(1e-6, 1 - HARD_BASE) * (1 - hard_frac) / hard_frac)


def nearest_neighbour(lab, a, frames):
    """The animal nearest `a` (mean centroid distance over the window), or None."""
    if lab.n_animals < 2:
        return None
    use3 = lab.points3d is not None and np.isfinite(lab.points3d[a][frames]).any()
    P = lab.points3d if use3 else lab.points2d[:, :, :, 0]
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cen = np.nanmean(P[:, frames], 2)
        d = [np.nanmean(np.linalg.norm(cen[o] - cen[a], axis=-1)) if o != a else np.inf
             for o in range(lab.n_animals)]
    d = np.nan_to_num(np.asarray(d, float), nan=np.inf, posinf=np.inf)
    nb = int(np.argmin(d))
    return nb if np.isfinite(d[nb]) else None
