"""The DATA side of the box prompt (report 27): compute a per-frame animal box for the loader.

The box a box-prompt model consumes is the target animal's extent in the crop frame, per frame
per camera -- a NON-position channel (it carries no per-keypoint position, only which box the
animal occupies). `compute_box_prompt` derives it POST HOC from what the loader already produced
(`coords`, `cgroup`), reusing THE crop rule, so there is no second copy of the rotate/crop/resize
chain. The ENCODER side (how the box is consumed) lives in `query_encoder.py`
(`BoxFilmEncoder` / `BoxTermEncoder`).

The box floor is deliberately SMALLER than the crop rule's own `min_crop_dim`: this is an
occupancy DESCRIPTION, not a crop target, so flooring it to the model's minimum crop size would
make a small distant animal look as big as a close one.
"""
from __future__ import annotations

import torch

from posetail.posetail.cube import project_points_torch

from . import crop as cropmod

BOX_PROMPT_MIN_DIM = 32
BOX_PROMPT_PAD = 10
BOX_PROMPT_FRAMES = ('all', 'first')


def compute_box_prompt(coords: torch.Tensor, cgroup: list[dict], mode: str,
                       min_dim: int = BOX_PROMPT_MIN_DIM, pad: int = BOX_PROMPT_PAD
                       ) -> torch.Tensor:
    """(T, C, 4) xyxy crop-pixel box around the target, one per (frame, camera). A NaN row where
    nothing was finite that frame in that camera -- the encoder substitutes its no-box token there.

    `coords`: (T,K,2) crop pixels (2D) or (T,K,3) world mm (3D, already through this window's
    crop/resize chain). `cgroup`: this window's own cropped+resized cameras.
    """
    is_2d = mode == '2d'
    if is_2d:
        assert len(cgroup) == 1
        p2d = coords[None]                               # (1,T,K,2)
    else:
        p2d = project_points_torch(cgroup, coords)        # (C,T,K,2)
    C, T = p2d.shape[0], p2d.shape[1]
    out = torch.full((T, C, 4), float('nan'), dtype=torch.float32)
    for c in range(C):
        size = cgroup[c]['size']
        for t in range(T):
            box = cropmod.crop_box_for_points(p2d[c, t], size, min_dim, pad)
            if box is not None:
                out[t, c] = box.to(torch.float32)
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
    """Exposure-bias jitter: the DEPLOYED box comes from a detector, not from labels. ONE draw for
    the whole window (matching `crop.jitter_box`'s per-item contract). NaN rows stay NaN."""
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
