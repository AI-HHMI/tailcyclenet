"""`--oracle-corrupt`: the oracle (`--anchor labels`) prior, broken on purpose to measure the echo coefficient.

A diagnostic, never on the deploy path.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.camera import prior_out_of_bounds
from .crops import prior_to_model_frame


ORACLE_CORRUPTIONS = ('off', 'stale', 'other', 'near', 'swap')


def _nearest_eligible_row(src, a, t0, boxes, scales, mode, cgroup):
    """The nearest other label row whose pose at this frame would survive `prior_out_of_bounds` --
    the same eligibility test `dataset.py`'s `prompt_swap_animal` applies. Returns a row index, or
    None if no other row qualifies.

    "Nearest" is measured in the model's own frame over keypoints both rows have finite and in
    bounds -- the frame the model will see the corrupted prior in.
    """
    target = prior_to_model_frame(torch.as_tensor(src[a][t0], dtype=torch.float32),
                                   mode, boxes, scales)
    best_row, best_dist = None, None
    for row in range(len(src)):
        if row == a:
            continue
        cand = prior_to_model_frame(torch.as_tensor(src[row][t0], dtype=torch.float32),
                                     mode, boxes, scales)
        oob = prior_out_of_bounds(cand, mode, cgroup)
        if int((~oob).sum()) < 2:
            continue
        both = torch.isfinite(target).all(-1) & torch.isfinite(cand).all(-1) & ~oob
        if int(both.sum()) < 1:
            continue
        d = float(torch.linalg.norm((cand - target)[both], dim=-1).mean())
        if best_dist is None or d < best_dist:
            best_row, best_dist = row, d
    return best_row


def _swap_kpt_pairs(p, n_pairs, seed):
    """`swap:<n>` -- n transpositions among `p`'s own finite keypoints: the direct inference probe
    for `dataset.py`'s `prompt_swap_kpt_pairs`.
    """
    finite = torch.isfinite(p).all(-1).nonzero(as_tuple=True)[0]
    if len(finite) < 2:
        return p
    rng = np.random.default_rng(seed)
    idx = finite[torch.from_numpy(rng.permutation(len(finite)))]
    n = min(n_pairs, len(idx) // 2)
    p = p.clone()
    for i in range(n):
        j, k = int(idx[2 * i]), int(idx[2 * i + 1])
        p[j], p[k] = p[k].clone(), p[j].clone()
    return p


def corrupt_prior(cfg, src, a, n_lab, frames, boxes, mode, cgroup, scales=None):
    """The oracle prior, optionally broken on purpose. Returns (pose in the SOURCE frame, qt).

    Measures alpha = d(output)/d(prior), the echo coefficient, which decides whether the prompt
    needs retraining. None of the corruptions is in training except through `dataset.py`'s
    `prompt_swap_kpt_pairs` / `prompt_swap_animal`, which `swap` and `near` probe directly:

    - `off:<x>`   a whole-body offset of x crop widths (the shape of a lag)
    - `stale:<n>` the pose from n frames earlier
    - `other`     the neighbouring animal's pose (`a + 1 % n_lab`)
    - `near`      the nearest eligible animal's pose instead of the fixed `a + 1` row
    - `swap:<n>`  n transpositions of this row's own keypoints

    Magnitudes are in CROP WIDTHS: sessions in one root mix pixels and millimetres. The direction
    is drawn from a generator seeded on (row, window), so two arms over one clip corrupt
    identically and the comparison is matched.

    `off` displaces by ONE crop width converted into whatever units the prior lives in:
    `get_camera_scale` is offset-invariant (a Jacobian), and 0.3.5 collapses a per-frame offset
    inside `get_camera_scale` itself; the width is the camera's own (`cgroup`'s camera is the
    reduced pass-1 one under `--refine-px`), not `cfg.image_size`.
    """
    kind, _, amt = (cfg.oracle_corrupt or '').partition(':')
    row, t0 = a, int(frames[0])
    if kind == 'other' and n_lab > 1:
        row = (a + 1) % n_lab
    elif kind == 'near' and n_lab > 1:
        nr = _nearest_eligible_row(src, a, t0, boxes, scales, mode, cgroup)
        if nr is not None:
            row = nr
    elif kind == 'stale':
        t0 = max(0, t0 - int(amt))
    p = torch.as_tensor(src[row][t0], dtype=torch.float32)
    if kind == 'swap':
        p = _swap_kpt_pairs(p, max(1, int(amt) if amt else 1), [a, t0, 0x5A7])
    if kind != 'off':
        return p, int(t0) - int(frames[0]) if kind == 'stale' else 0
    if mode == '2d':
        b = np.asarray(boxes[0], np.float64)
        width = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
    else:
        from tracktail.model.cube import get_camera_scale
        fin = p[torch.isfinite(p).all(-1)]
        if not len(fin):
            return p, 0
        scale = torch.nanmedian(get_camera_scale(cgroup, fin[None]))
        if not torch.isfinite(scale):
            return p, 0
        width = float(scale) * int(cgroup[0]['size'].max())
    rng = np.random.default_rng([a, int(frames[0])])
    v = rng.normal(size=p.shape[-1])
    v = v / max(float(np.linalg.norm(v)), 1e-9)
    return p + torch.as_tensor(float(amt) * width * v, dtype=p.dtype), 0
