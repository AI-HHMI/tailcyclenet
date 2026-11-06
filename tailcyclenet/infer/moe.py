"""The v7 2D MoE at deploy: which expert writes each (slot, frame), and E2's inputs.

Routing (report 76 crowd-moe, lever B): a slot ENTERS E2 at a frame where its detector box is in
contact with another slot's -- IoU > 0.3 or centre distance < 0.75 x the two boxes' mean longer
side (the frozen contact rule) -- and EXITS only once it is clear of every other box at the looser
1.25x thresholds (IoU < 0.24 and centre distance > 0.9375 x mean side) AND has dwelt >= 4 frames in
E2. A frame with no own box holds its state; the state crosses windows and blocks (per group).
`route = 'e1'` / `'e2'` force one expert (diagnostics, and the per-root seam a 3D expert choice will
use). A one-keypoint axis always routes to E1 (owner, 2026-11-06: E2 collapses to the pair midpoint
at K = 1).

E2's anchors are the detector's TIGHT boxes of this slot and of the 3 nearest slots whose boxes
touch the crop (crop px, NaN outside it); the anchored queries' position priors are this slot's carry
(query 0, from the loop) and the other slots' latest WRITTEN rows (the routed rows the loop also
carries). Everything is label-free.
"""
from __future__ import annotations

import numpy as np
import torch

NEAR_IOU = 0.3
NEAR_CENTRE = 0.75
EXIT_SCALE = 1.25
DWELL = 4
# HISTORY: frames of routing state / written rows kept behind the current window
HISTORY = 64
ROUTES = ('contact', 'e1', 'e2')
_K1_LOGGED = [False]


def _iou(b, o):
    """IoU of two xyxy boxes."""
    ix = max(0, min(b[2], o[2]) - max(b[0], o[0]))
    iy = max(0, min(b[3], o[3]) - max(b[1], o[1]))
    u = (b[2] - b[0]) * (b[3] - b[1]) + (o[2] - o[0]) * (o[3] - o[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def in_contact(b, others, iou_t=NEAR_IOU, centre=NEAR_CENTRE):
    """The contact rule for one box against the other slots' boxes (source px) -> bool, or None
    when this slot has no box at this frame."""
    if not np.isfinite(b).all():
        return None
    side = max(b[2] - b[0], b[3] - b[1])
    c = (b[:2] + b[2:]) / 2
    for o in others:
        if not np.isfinite(o).all():
            continue
        so = max(o[2] - o[0], o[3] - o[1])
        if _iou(b, o) > iou_t or np.linalg.norm(c - (o[:2] + o[2:]) / 2) < centre * (side + so) / 2:
            return True
    return False


def hysteresis(history, frames, bb, a):
    """The routing mask for slot `a` over one window, updating `history` ({frame: (in E2, entry
    frame)}). bb (S,T,4) every slot's boxes at the window's frames. Idempotent per window: it
    reads only frames before the window and rewrites the window's own."""
    S = bb.shape[0]
    prev = [f for f in history if f < int(frames[0])]
    st, ent = history[max(prev)] if prev else (False, -1)
    m = np.zeros(len(frames), bool)
    for t, f in enumerate(frames):
        f = int(f)
        others = [bb[s, t] for s in range(S) if s != a]
        if not st:
            if in_contact(bb[a, t], others):
                st, ent = True, f
        elif (in_contact(bb[a, t], others, NEAR_IOU / EXIT_SCALE, EXIT_SCALE * NEAR_CENTRE) is False
              and f - ent >= DWELL):
            st = False
        m[t] = st
        history[f] = (st, ent)
    for f in [f for f in history if f < int(frames[0]) - HISTORY]:
        del history[f]
    return m


class GroupRouter:
    """One group's MoE deploy state: per-slot routing history and written rows."""

    def __init__(self, route, K):
        """`route` one of ROUTES; a K == 1 axis is forced to 'e1' (logged once per process)."""
        assert route in ROUTES, route
        if K == 1 and route != 'e1':
            if not _K1_LOGGED[0]:
                print(f'moe: K == 1 keypoint axis -> every window routes to E1 (requested '
                      f'{route!r}; owner rule 2026-11-06)', flush=True)
                _K1_LOGGED[0] = True
            route = 'e1'
        self.route = route
        self.history = {}
        self.rows = {}
        self.stats = {'windows': 0, 'frames': 0, 'frames_e2': 0}

    def inputs(self, a, frames, win_boxes, boxes, scales, cgroup, dev):
        """`set_inputs` for slot `a`'s window, or None when the window is E1's alone (the plain
        forward). win_boxes (S,T,C,4) the detector boxes at the window's frames (source px)."""
        T = len(frames)
        self.stats['windows'] += 1
        self.stats['frames'] += T
        if self.route == 'e1':
            return None
        bb = np.asarray(win_boxes[:, :, 0], np.float64)
        mask = (np.ones(T, bool) if self.route == 'e2'
                else hysteresis(self.history.setdefault(a, {}), frames, bb, a))
        if not mask.any():
            return None
        self.stats['frames_e2'] += int(mask.sum())
        origin = np.asarray(boxes[0][:2], np.float32)
        sc = np.asarray(scales[0], np.float32).reshape(-1)
        sc = np.concatenate([sc, sc])[:2] if sc.size == 1 else sc[:2]
        un = np.asarray(win_boxes[:, :, 0], np.float32)
        cb = np.concatenate([(un[..., :2] - origin) * sc, (un[..., 2:] - origin) * sc], -1)
        Wc, Hc = (float(v) for v in cgroup[0]['size'][:2])
        fin = np.isfinite(cb).all(-1)
        ins = fin & (cb[..., 2] > 0) & (cb[..., 0] < Wc) & (cb[..., 3] > 0) & (cb[..., 1] < Hc)
        cen = np.where(fin[..., None], (cb[..., :2] + cb[..., 2:]) / 2, np.nan)
        with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            ca = np.nanmean(cen[a], 0)
            dist = [np.nanmean(np.linalg.norm(cen[s] - ca, axis=-1)) for s in range(cb.shape[0])]
        from ..pose.set_decoder import NA
        order = [a] + [int(s) for s in np.argsort(np.nan_to_num(dist, nan=1e9))
                       if s != a and ins[s].any()]
        anchors = np.full((1, NA, T, 1, 4), np.nan, np.float32)
        for j, s in enumerate(order[:NA]):
            anchors[0, j, :, 0] = np.where(ins[s][..., None], cb[s], np.nan)
        out = {'route': mask, 'anchors': torch.from_numpy(anchors).to(dev),
               'slot_boxes': cb, 'slot': int(a)}
        K = next((v.shape[0] for st in self.rows.values() for v in st.values()), 0)
        if K:
            pr = np.full((1, NA, K, 2), np.nan, np.float32)
            pt = np.zeros((1, NA, K), np.int64)
            src = np.asarray(frames)
            for j, s in enumerate(order[:NA]):
                if j == 0:
                    continue
                st = self.rows.get(s, {})
                fs = [int(f) for f in src if int(f) in st]
                if fs:
                    f = fs[-1]
                    pr[0, j] = (st[f] - origin) * sc
                    pt[0, j] = int(np.flatnonzero(src == f)[0])
            out['anchor_prior'] = torch.from_numpy(pr).to(dev)
            out['anchor_prompt_t'] = torch.from_numpy(pt).to(dev)
        return out

    def record(self, a, frames, p_src):
        """Slot `a`'s written (routed) rows for this window, source px (T,K,2)."""
        st = self.rows.setdefault(a, {})
        for t, f in enumerate(np.asarray(frames)[:len(p_src)]):
            if np.isfinite(p_src[t]).any():
                st[int(f)] = np.asarray(p_src[t], np.float32)
        for f in [f for f in st if f < int(frames[0]) - HISTORY]:
            del st[f]
