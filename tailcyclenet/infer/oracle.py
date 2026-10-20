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


BOX_CORRUPTIONS = ('drop', 'shift', 'shrink', 'swap')


def parse_box_corrupt(spec):
    """`--box-corrupt SPEC` -> a tuple of (kind, p, amount); raises ValueError naming the fault.

    SPEC is comma-joined `drop:<p>`, `shift:<p>:<sides>`, `shrink:<p>:<f>`, `swap:<p>`; p is the
    per-(animal, window, camera) probability, applied in the order written.
    """
    out = []
    for part in str(spec).split(','):
        bits = part.strip().split(':')
        kind = bits[0]
        if kind not in BOX_CORRUPTIONS:
            raise ValueError(f'{part!r}: kind must be one of {BOX_CORRUPTIONS}')
        want = 3 if kind in ('shift', 'shrink') else 2
        if len(bits) != want:
            raise ValueError(f'{part!r}: {kind} takes {want - 1} number(s), e.g. '
                             + {'drop': 'drop:0.25', 'swap': 'swap:0.25',
                                'shift': 'shift:0.25:0.5', 'shrink': 'shrink:0.25:0.6'}[kind])
        p = float(bits[1])
        amt = float(bits[2]) if want == 3 else None
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'{part!r}: probability {p} is not in [0, 1]')
        if kind == 'shrink' and not 0.0 < amt < 1.0:
            raise ValueError(f'{part!r}: the shrink factor must be in (0, 1)')
        if kind == 'shift' and amt < 0:
            raise ValueError(f'{part!r}: the shift is a distance in box sides, >= 0')
        out.append((kind, p, amt))
    return tuple(out)


def _union(v):
    """The union xyxy of a (T,4) box stack over its finite rows, or None."""
    v = v[np.isfinite(v).all(-1)]
    if not len(v):
        return None
    return np.array([v[:, 0].min(), v[:, 1].min(), v[:, 2].max(), v[:, 3].max()], np.float64)


def corrupt_boxes(bb, specs, gid, start, seed=0):
    """`--box-corrupt`: one window's detector boxes, broken on purpose. A diagnostic, never a
    deployment arm.

    Inputs: bb -- (S,T,C,4) the window's per-frame boxes (not modified); specs --
        `parse_box_corrupt`'s tuple; gid / start -- the group and the window's first SOURCE
        frame, which with (animal, camera) seed the draw.
    Outputs: a corrupted (S,T,C,4) float32 copy.

    One draw per (animal, window, camera) and spec, from a generator seeded on those indices, so
    the result is deterministic given the flag and independent of prefetch order or block
    partition. Every spec draws its numbers whether or not it fires, so adding a spec never
    moves another's stream. `shift` moves every frame's box by `sides` x the window union's mean
    side in one random direction; `shrink` scales each frame's box about its own centre; `swap`
    takes the nearest OTHER row's boxes in that camera (union-centre distance, uncorrupted
    boxes), and is a no-op where no other row has one.
    """
    import zlib

    bb = np.asarray(bb, np.float32)
    out = bb.copy()
    S, _, C, _ = bb.shape
    g = zlib.crc32(str(gid).encode())
    for a in range(S):
        for c in range(C):
            rng = np.random.default_rng([int(seed), g, int(start), a, c])
            for kind, p, amt in specs:
                u, ang = rng.random(), rng.random() * 2 * np.pi
                if u >= p:
                    continue
                v = out[a, :, c]
                un = _union(v)
                if un is None:
                    continue
                if kind == 'drop':
                    out[a, :, c] = np.nan
                elif kind == 'shift':
                    side = 0.5 * ((un[2] - un[0]) + (un[3] - un[1]))
                    dx, dy = amt * side * np.cos(ang), amt * side * np.sin(ang)
                    out[a, :, c] = v + np.array([dx, dy, dx, dy], np.float32)
                elif kind == 'shrink':
                    cx, cy = (v[:, 0] + v[:, 2]) / 2, (v[:, 1] + v[:, 3]) / 2
                    hw, hh = (v[:, 2] - v[:, 0]) * amt / 2, (v[:, 3] - v[:, 1]) * amt / 2
                    out[a, :, c] = np.stack([cx - hw, cy - hh, cx + hw, cy + hh], -1)
                else:
                    ca = np.array([(un[0] + un[2]) / 2, (un[1] + un[3]) / 2])
                    best, bd = None, None
                    for b in range(S):
                        ub = _union(bb[b, :, c]) if b != a else None
                        if ub is None:
                            continue
                        d = float(np.hypot((ub[0] + ub[2]) / 2 - ca[0], (ub[1] + ub[3]) / 2 - ca[1]))
                        if bd is None or d < bd:
                            best, bd = b, d
                    if best is not None:
                        out[a, :, c] = bb[best, :, c]
    return out
