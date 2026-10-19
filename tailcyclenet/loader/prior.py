"""The query prior and its training corruptions, in DRAW ORDER (the order is RNG-load-bearing;
`tests/loader/test_select_realise.py` guards it).

`kpt_prior` is the pose at `prompt_t`, the first labelled frame per keypoint. Corruptions, in
order: `prompt_dropout` (PER ITEM, not per keypoint, because scene geometry derives from the whole
prior set), stale priors, `prompt_swap_animal`, `prompt_swap_kpt_pairs`, the whole-body
`prompt_offset_px`, and the i.i.d. `prompt_noise_px`. Pixel sigmas are scaled to world units in 3D
by `get_camera_scale`. See dev/plans/prompt_prior_corruptions.md.
"""
from __future__ import annotations

import numpy as np
import torch

from tracktail.model.cube import get_camera_scale

from ..data.camera import prior_out_of_bounds


def corrupt_prior(cfg, train, coords, cgroup, neighbour_full, K, R, rng, pose_only_active):
    """The (possibly corrupted) query prior for one item. Returns `(kpt_prior, prompt_t)`.

    Inputs: cfg -- the `LoaderConfig`; train -- whether corruptions apply; coords -- (T,K,R)
            targets; cgroup -- the realised cameras; neighbour_full -- one neighbour animal's
            (T,K,R) pose or None; K/R -- keypoint count and target dimension; rng -- the item's
            generator; pose_only_active -- skip dropout/offset/noise (`pose_only_prob`).
    Outputs: kpt_prior (K,R) with NaN where no prior; prompt_t (K,) int32.
    Side effects: draws from `rng` in the fixed order above.
    """
    finite = torch.isfinite(coords).all(-1)
    prompt_t = torch.where(finite.any(0), finite.float().argmax(0), torch.zeros(K).long())
    prompt_t = prompt_t.to(torch.int32)
    kpt_prior = coords[prompt_t, torch.arange(K)].clone()
    kpt_prior[~finite.any(0)] = float('nan')
    dropped_fully = False
    if train and not pose_only_active and rng.random() < cfg.prompt_dropout:
        kpt_prior[:] = float('nan')
        dropped_fully = True
    px = 1.0
    if R == 3 and (cfg.prompt_noise_px > 0 or cfg.prompt_offset_px > 0) \
            and bool(torch.isfinite(kpt_prior).any()):
        pts = kpt_prior[torch.isfinite(kpt_prior).all(-1)][None]
        px = float(torch.nanmedian(get_camera_scale(cgroup, pts)))
        if not np.isfinite(px):
            px = 1.0
    if train and cfg.prompt_stale_frames > 0 \
            and bool(torch.isfinite(kpt_prior).any()):
        d = int(rng.integers(0, int(cfg.prompt_stale_frames) + 1))
        if d:
            t_alt = torch.clamp(prompt_t.long() + d, max=coords.shape[0] - 1)
            alt = coords[t_alt, torch.arange(K)]
            swap = torch.isfinite(alt).all(-1) & torch.isfinite(kpt_prior).all(-1)
            kpt_prior = torch.where(swap[:, None], alt, kpt_prior)
    if neighbour_full is not None:
        mode_str = '2d' if R == 2 else '3d'
        neighbour_prior = neighbour_full[prompt_t, torch.arange(K)].clone()
        neighbour_prior[prior_out_of_bounds(neighbour_prior, mode_str, cgroup)] = float('nan')
        jump = (torch.as_tensor(rng.random(K)) < cfg.prompt_swap_animal) \
            & torch.isfinite(neighbour_prior).all(-1)
        if not dropped_fully:
            kpt_prior = torch.where(jump[:, None], neighbour_prior, kpt_prior)
    if train and cfg.prompt_swap_kpt_pairs > 0:
        finite_idx = torch.isfinite(kpt_prior).all(-1).nonzero(as_tuple=True)[0]
        m = len(finite_idx)
        if m >= 2:
            sel = torch.as_tensor(rng.random(m)) < cfg.prompt_swap_kpt_pairs
            if bool(sel.any()):
                local = torch.arange(m)[sel]
                offset = torch.from_numpy(rng.integers(1, m, size=int(sel.sum())))
                src_idx = finite_idx[(local + offset) % m]
                original = kpt_prior
                kpt_prior = kpt_prior.clone()
                kpt_prior[finite_idx[sel]] = original[src_idx]
    if train and not pose_only_active and cfg.prompt_offset_px > 0 \
            and bool(torch.isfinite(kpt_prior).any()):
        kpt_prior += torch.as_tensor(
            rng.normal(0.0, float(cfg.prompt_offset_px) * px, (1, R)),
            dtype=kpt_prior.dtype)
    if train and not pose_only_active and cfg.prompt_noise_px > 0 \
            and bool(torch.isfinite(kpt_prior).any()):
        kpt_prior += torch.as_tensor(
            rng.normal(0.0, float(cfg.prompt_noise_px) * px, kpt_prior.shape),
            dtype=kpt_prior.dtype)
    return kpt_prior, prompt_t
