"""The DATA side of the box prompt: compute a per-frame animal box for the loader.

The box is the target animal's TIGHT crop-frame box -- a NON-position channel: its own extent
(the stored `instances.pq` box where the run's `box_source` is `instances` and one exists, else the
keypoint extent), padded by `BOX_PROMPT_PAD` px and floored at `BOX_PROMPT_MIN_DIM` PER AXIS, never
squared (`tight_prompt_box`, owner decision 2026-11-06). Deployment builds it from the detector's
tight box with the same function (`infer.prior.deploy_box_prompt`), so train and deploy agree on
one box. Runs trained before that recorded no `[data].box_prompt_geometry` and saw the SQUARE
crop-rule box of the keypoint extent; they deploy on the tight box anyway (zero-shot tight on a
square-trained model was inert, report 76 nonsq-crop). The encoder side lives in
`query_encoder.py`.
"""
from __future__ import annotations

import torch

from tracktail.model.cube import project_points_torch

BOX_PROMPT_MIN_DIM = 32
BOX_PROMPT_PAD = 10
BOX_PROMPT_FRAMES = ('all', 'first')
# What `[data].box_prompt_geometry` may say. Only 'tight' trains; an ABSENT key on a run folder
# means 'square' (every run before 2026-11-06), read by `trained_geometry`.
BOX_PROMPT_GEOMETRIES = ('tight',)


def trained_geometry(config: dict) -> str:
    """The prompt geometry a run TRAINED on: its `[data].box_prompt_geometry`, absent = 'square'."""
    return str((config or {}).get('data', {}).get('box_prompt_geometry', 'square'))


def tight_prompt_box(pts: torch.Tensor, size, min_dim: int = BOX_PROMPT_MIN_DIM,
                     pad: float = BOX_PROMPT_PAD) -> torch.Tensor | None:
    """THE prompt rule: (N,2) points (any finite subset; box corners or keypoints) -> float32
    [x1, y1, x2, y2], or None when nothing is finite.

    Per axis: the extent padded by `pad` on both sides, grown about its centre to `min_dim` when
    shorter, then clamped to [0, size]. No squaring, no int rounding.
    """
    p = pts.reshape(-1, 2).to(torch.float32)
    p = p[torch.isfinite(p).all(-1)]
    if not p.shape[0]:
        return None
    lo, hi = p.min(0).values - float(pad), p.max(0).values + float(pad)
    c, half = (lo + hi) / 2, torch.clamp((hi - lo) / 2, min=float(min_dim) / 2)
    lim = torch.as_tensor(size, dtype=torch.float32).reshape(-1)[:2]
    lo = torch.minimum(torch.clamp(c - half, min=0.0), lim)
    hi = torch.minimum(torch.clamp(c + half, min=0.0), lim)
    return torch.cat([lo, hi])


def compute_box_prompt(coords: torch.Tensor, cgroup: list[dict], mode: str,
                       min_dim: int = BOX_PROMPT_MIN_DIM, pad: int = BOX_PROMPT_PAD,
                       box_pts=None) -> torch.Tensor:
    """(T, C, 4) xyxy crop-pixel box around the target, one per (frame, camera); NaN where
    nothing was finite. `coords` is (T,K,2) crop pixels or (T,K,3) world mm through this window's
    crop chain. `box_pts` -- optional per-camera (C, T, 4, 2) crop-pixel corners of the stored
    instance box; a (camera, frame) with a finite one uses it instead of the keypoints.
    """
    is_2d = mode == '2d'
    if is_2d:
        assert len(cgroup) == 1
        p2d = coords[None]
    else:
        p2d = project_points_torch(cgroup, coords)
    C, T = p2d.shape[0], p2d.shape[1]
    out = torch.full((T, C, 4), float('nan'), dtype=torch.float32)
    for c in range(C):
        size = cgroup[c]['size']
        bp = None if box_pts is None else box_pts[c]
        for t in range(T):
            src = p2d[c, t]
            if bp is not None and torch.isfinite(bp[t]).all():
                src = bp[t]
            box = tight_prompt_box(src, size, min_dim, pad)
            if box is not None:
                out[t, c] = box
    return out


def apply_frames_mode(box_prompt: torch.Tensor, mode: str) -> torch.Tensor:
    """'all' (per-frame, unchanged) or 'first' (every frame gets the starting frame's box)."""
    if mode == 'all':
        return box_prompt
    if mode == 'first':
        return box_prompt[:1].expand_as(box_prompt).clone()
    raise ValueError(f'box_prompt_frames must be one of {BOX_PROMPT_FRAMES}, got {mode!r}')


def apply_jitter(box_prompt: torch.Tensor, rng, shift_frac: float, scale_frac: float
                 ) -> torch.Tensor:
    """Exposure-bias jitter: the DEPLOYED box comes from a detector, not from labels. One draw
    for the whole window; NaN rows stay NaN."""
    if shift_frac <= 0 and scale_frac <= 0:
        return box_prompt
    s = 1.0 + float(rng.uniform(-scale_frac, scale_frac))
    dx = float(rng.uniform(-shift_frac, shift_frac))
    dy = float(rng.uniform(-shift_frac, shift_frac))
    ok = torch.isfinite(box_prompt).all(-1)
    x0, y0, x1, y1 = box_prompt.unbind(-1)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2 + w * dx, (y0 + y1) / 2 + h * dy
    nw, nh = w * s / 2, h * s / 2
    jittered = torch.stack([cx - nw, cy - nh, cx + nw, cy + nh], -1)
    return torch.where(ok[..., None], jittered, box_prompt)
