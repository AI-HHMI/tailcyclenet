"""Synthetic track corruption and triplet assembly for THIS repo's loader.

     good   = clean track,      view A             label +1, every point
     bad    = corrupt(good),    view A (SAME)      label -1, every point
     anchor = good or bad 50/50, view B (indep.)   label = its source's

`good` and `bad` SHARE pixels and camera geometry, so the only difference between them is
coordinates -- which is what makes the good-vs-bad gap a track-quality signal rather than an
appearance signal. The anchor is the invariance target.

WHAT TRANSFERS FROM THE REFERENCE (`posetail/datasets/scorer_corruption.py`) AND WHAT DOES NOT.
`GENERATORS`, `PointCorruptor`, `apply_drop_mask` and `build_corruptors` are imported verbatim.
`compute_drop_mask` is FORKED (see `_compute_drop_mask`). `make_triplet` and every
rotation/crop/resize helper are rewritten: the reference builds them against a loader that returns
FULL, UN-AUGMENTED frames and owns the whole geometry chain itself, whereas this repo's
`PoseDataset` returns already-cropped, already-resized, already-augmented uint8 with the rotation
baked into the decode. Nothing about that ordering is recoverable by an adapter shim.

The seam (`PoseDataset._select` / `_realise`) is what makes the rewrite possible: ONE selection,
TWO independent realisations. Calling `__getitem__` twice would select a DIFFERENT sample -- `_pick`
re-draws the pool entry and `_frames` re-draws the anchor and stride.
"""
import math

import numpy as np
import torch
from einops import rearrange

from posetail.datasets.scorer_corruption import (GENERATORS, PointCorruptor, apply_drop_mask,
                                                 build_corruptors)
from posetail.posetail.cube import (get_camera_scale, is_point_visible,
                                     project_points_torch)

TRIPLET_KEYS = ('good', 'bad', 'anchor')

# The corruption types that can move a SINGLE observed slot, which is all a sparse window can be
# trained on. `gradual_drift` and `sinusoid` ramp from or through zero, so at one observed frame
# they may corrupt nothing at all -- and a corruption that changes nothing would teach the scorer
# to call a corrupt sample clean, which is worse than excluding the window. `const_offset` is
# FORCED for sparse keypoints (see `build_corruptors_for`); `frame_noise` rides alongside it.
SPARSE_TYPES = ('const_offset', 'frame_noise')

# Full-trajectory generators are optionally restricted to a contiguous interval.  ``frame_noise``
# already has this behaviour in posetail and is intentionally not included here.  Keeping the
# names in one place also makes the per-type mask's last axis stable (it is the order of
# ``GENERATORS``, not the reduced sparse menu).
SEGMENT_TYPES = ('const_offset', 'gradual_drift', 'sinusoid')
SPARSE_DRAW_RETRIES = 16


def _unique_source_frames(frames):
    """Return source-frame values and an inverse gather in first-occurrence order.

    ``np.unique`` sorts values, whereas a window's temporal order is part of the corruption
    distribution.  The loader normally supplies an increasing lattice, but preserving first
    occurrence makes this helper correct for a caller that supplies a wrapped or custom window as
    well.  The inverse maps each local slot to one unique source-frame slot.

    Inputs: ``frames`` -- a one-dimensional sequence of integer source frame ids.
    Outputs: ``(unique, inverse)`` as int64 numpy arrays, both length-compatible with ``frames``.
    Side effects: none.
    """
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    arr = np.asarray(frames, dtype=np.int64).reshape(-1)
    unique = []
    inverse = np.empty(arr.size, dtype=np.int64)
    lookup = {}
    for i, value in enumerate(arr.tolist()):
        slot = lookup.get(value)
        if slot is None:
            slot = len(unique)
            lookup[value] = slot
            unique.append(value)
        inverse[i] = slot
    return np.asarray(unique, dtype=np.int64), inverse


def source_frame_weights(frames, shape=None, eligible=None, device=None,
                         dtype=torch.float32):
    """Inverse-multiplicity weights for local slots of a source-frame window.

    A clamped edge window can contain the same source frame many times.  With these weights the
    eligible local occurrences for each ``(batch, source frame, keypoint)`` sum to one, so an edge
    frame cannot dominate a framewise loss merely because it was padding-gathered.  ``eligible``
    is useful after view/drop filtering; ineligible occurrences receive zero and the remaining
    occurrences are renormalised for that source frame/keypoint.

    Inputs:
      frames -- ``[T]`` source frame ids (numpy, list or tensor).
      shape -- optional ``(B,T,K)`` output shape.  If omitted and ``eligible`` is omitted, returns
               ``[T]``; if ``eligible`` is supplied its shape determines the output.
      eligible -- optional bool ``[B,T,K]``.  Defaults to every local slot.
      device / dtype -- output placement and floating dtype.
    Outputs: a tensor of shape ``[T]`` or ``[B,T,K]``.
    Side effects: none.
    """
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    arr = np.asarray(frames, dtype=np.int64).reshape(-1)
    if eligible is not None:
        if eligible.ndim != 3 or eligible.shape[1] != len(arr):
            raise ValueError(
                f'eligible must be [B,T,K] with T={len(arr)}, got {tuple(eligible.shape)}')
        b, t, k = eligible.shape
        dev = eligible.device if device is None else device
        out = torch.zeros((b, t, k), dtype=dtype, device=dev)
        ok = eligible.to(device=dev, dtype=torch.bool)
        for value in dict.fromkeys(arr.tolist()):
            ts = np.flatnonzero(arr == value).tolist()
            slots = ok[:, ts, :]
            denom = slots.sum(dim=1, keepdim=True).to(dtype)
            val = torch.where(denom > 0, 1.0 / denom, torch.zeros_like(denom))
            out[:, ts, :] = slots.to(dtype) * val
        return out

    if shape is None:
        out = torch.zeros(len(arr), dtype=dtype, device=device)
        for value in dict.fromkeys(arr.tolist()):
            ts = np.flatnonzero(arr == value)
            out[torch.as_tensor(ts, device=out.device)] = 1.0 / len(ts)
        return out

    if len(shape) != 3 or int(shape[1]) != len(arr):
        raise ValueError(f'shape must be (B,T,K) with T={len(arr)}, got {shape}')
    b, t, k = (int(x) for x in shape)
    return source_frame_weights(arr, eligible=torch.ones((b, t, k), dtype=torch.bool,
                                                          device=device), device=device,
                                dtype=dtype)


def _segment_range(cfg, n_unique):
    """Parse the inclusive segment length range and full-window share."""
    raw = cfg.get('segment_len_frames', cfg.get('segment_length_frames', (1, n_unique)))
    if isinstance(raw, (int, float)):
        lo = hi = int(raw)
    else:
        vals = list(raw)
        if len(vals) != 2:
            raise ValueError('segment_len_frames must be an inclusive [lo, hi] pair')
        lo, hi = (int(vals[0]), int(vals[1]))
    if lo < 1 or hi < lo:
        raise ValueError(f'segment_len_frames must satisfy 1 <= lo <= hi, got {(lo, hi)}')
    lo = min(lo, max(n_unique, 1))
    hi = min(hi, max(n_unique, 1))
    full = cfg.get('segment_full_window_share',
                   cfg.get('full_window_share', cfg.get('segment_full_window_prob', 0.0)))
    full = float(full)
    if not 0.0 <= full <= 1.0:
        raise ValueError(f'segment full-window share must be in [0,1], got {full}')
    return lo, hi, full


def _segment_mask(P, n_unique, device, cfg, names):
    """Draw per-point/type contiguous segment masks and serialisable metadata.

    A segment gate is a *partial-subsequence probability*: when a full-trajectory corruption type
    fires, ``segment_prob`` says how often its otherwise full-window contribution is restricted to
    a random interval.  ``segment_full_window_share`` is an explicit escape hatch for the identity
    swap/failure case.  Segment starts and lengths are sampled over unique source-frame positions,
    never over clamp-duplicated local slots.

    Outputs are ``(mask, metadata)``.  ``mask`` is ``[P,T_unique,len(GENERATORS)]`` and metadata is
    a list of ``P`` dictionaries, each with one list of segments per generator name.  A type not in
    ``names`` has an empty list and an all-ones mask.
    """
    p = float(cfg.get('segment_prob', 0.0))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f'segment_prob must be in [0,1], got {p}')
    count_cfg = cfg.get('n_segments', cfg.get('segment_count', 1))
    if isinstance(count_cfg, (list, tuple)):
        if len(count_cfg) != 2:
            raise ValueError('segment_count must be an integer or inclusive [lo, hi] pair')
        count_lo, count_hi = int(count_cfg[0]), int(count_cfg[1])
    else:
        count_lo = count_hi = int(count_cfg)
    if count_lo < 1 or count_hi < count_lo:
        raise ValueError(f'segment_count must satisfy 1 <= lo <= hi, got {(count_lo, count_hi)}')
    lo, hi, full_share = _segment_range(cfg, n_unique)
    all_masks = torch.ones((P, n_unique, len(GENERATORS)), dtype=torch.bool, device=device)
    names_set = set(names)
    by_name = {name: i for i, name in enumerate(GENERATORS)}
    requested = cfg.get('segment_types', SEGMENT_TYPES)
    if isinstance(requested, str):
        requested = (requested,)
    requested = tuple(requested)
    unknown = set(requested) - set(SEGMENT_TYPES)
    if unknown:
        raise ValueError(f'segment_types cannot be gated: {sorted(unknown)}')
    metadata = [{name: [] for name in GENERATORS} for _ in range(P)]
    for name in requested:
        if name not in names_set:
            continue
        type_ix = by_name[name]
        gated = torch.rand(P, device=device) < p
        for pi in range(P):
            if not bool(gated[pi]):
                metadata[pi][name] = [{'start': 0, 'length': int(n_unique),
                                       'full_window': True}]
                continue
            count = int(torch.randint(count_lo, count_hi + 1, (), device=device))
            mask = torch.zeros(n_unique, dtype=torch.bool, device=device)
            pieces = []
            for _ in range(count):
                is_full = bool(torch.rand((), device=device) < full_share)
                length = n_unique if is_full else int(torch.randint(lo, hi + 1, (),
                                                                     device=device))
                length = min(max(length, 1), n_unique)
                start = 0 if length == n_unique else int(torch.randint(
                    0, n_unique - length + 1, (), device=device))
                mask[start:start + length] = True
                pieces.append({'start': start, 'length': length, 'full_window': length == n_unique,
                               'source_start': None, 'source_frames': None})
            all_masks[pi, :, type_ix] = mask
            metadata[pi][name] = pieces
    return all_masks, metadata


def tagged_draw_segmented(corruptor, P, T, D, device, cfg):
    """`tagged_draw` with optional full-trajectory segment gates.

    This deliberately lives beside (rather than inside) :func:`tagged_draw`: the latter is a
    byte-for-byte reproduction of posetail's draw and is used by the legacy sequence regression.
    """
    names = corruptor.names
    slot = {n: i for i, n in enumerate(GENERATORS)}
    fired = torch.zeros(P, len(GENERATORS), dtype=torch.bool, device=device)
    contributions = {}
    applied = torch.zeros(P, dtype=torch.bool, device=device)
    total = torch.zeros(P, T, D, device=device)
    segment_masks, metadata = _segment_mask(P, T, device, cfg, names)
    for name in names:
        shift = GENERATORS[name](P, T, D, device, corruptor.mags[name])
        contributions[name] = shift
        mask = torch.rand(P, device=device) < corruptor.probs[name]
        type_ix = slot[name]
        shift = shift * segment_masks[:, :, type_ix, None].float()
        total = total + shift * mask[:, None, None].float()
        applied = applied | mask
        fired[:, type_ix] |= mask

    need = ~applied
    if need.any():
        pick = torch.randint(0, len(names), (P,), device=device)
        for ci, name in enumerate(names):
            m = need & (pick == ci)
            if m.any():
                type_ix = slot[name]
                shift = contributions[name] * segment_masks[:, :, type_ix, None].float()
                total = total + shift * m[:, None, None].float()
                fired[:, type_ix] |= m
    return total, fired, segment_masks, metadata


def tagged_draw(corruptor, P, T, D, device):
    """A `PointCorruptor` draw that ALSO reports which types fired for each point.

    Gate B needs `triplet_acc` broken out per corruption type: an aggregate that hides
    "sinusoid 0.99 / const_offset 0.51" is not a measurement, and a type at chance means that
    corruption is invisible and its magnitude is wrong. The library's `PointCorruptor.__call__`
    returns only the summed shift, so this reproduces its algorithm exactly and adds tags.

    Kept in step with the library by `tests/test_scorer_triplet.py`, which asserts this and
    `PointCorruptor` produce the SAME tensor from the same RNG state -- without that, the tags
    would describe a different draw from the one the model was trained on.

    Inputs: corruptor -- a `PointCorruptor`; P, T, D -- points, frames, coordinate dim;
            device -- where to build the draw.
    Outputs: (shift [P,T,D] in unit magnitude space, fired [P, len(GENERATORS)] bool).
    Side effects: draws from the ambient torch RNG.
    """
    names = corruptor.names
    slot = {n: i for i, n in enumerate(GENERATORS)}
    fired = torch.zeros(P, len(GENERATORS), dtype=torch.bool, device=device)
    contributions = {}
    applied = torch.zeros(P, dtype=torch.bool, device=device)
    total = torch.zeros(P, T, D, device=device)
    for name in names:
        shift = GENERATORS[name](P, T, D, device, corruptor.mags[name])
        contributions[name] = shift
        mask = torch.rand(P, device=device) < corruptor.probs[name]
        total = total + shift * mask[:, None, None].float()
        applied = applied | mask
        fired[:, slot[name]] |= mask

    need = ~applied
    if need.any():
        pick = torch.randint(0, len(names), (P,), device=device)
        for ci, name in enumerate(names):
            m = need & (pick == ci)
            if m.any():
                total = total + contributions[name] * m[:, None, None].float()
                fired[:, slot[name]] |= m
    return total, fired


def build_corruptors_for(cfg):
    """(dense, sparse) `PointCorruptor` pairs for 3D and 2D.

    The dense pair is the reference's full menu -- all four types, each gated at its configured
    probability. The sparse pair is what a keypoint with too few distinct observed frames gets, and
    it FORCES `const_offset` (probability 1.0) rather than gating it at the configured value.

    Forcing it is section 3.8b's own wording -- "`const_offset` always qualifies" for a sparse
    keypoint -- and it is the difference between a sparse window that trains and one that does not.
    `frame_noise` moves a CONTIGUOUS window, and a sparse keypoint observed in only the last few
    frames of a window is missed by that window often; measured on 3dpop, a sparse keypoint was
    left uncorrupted by ~83% of draws with `const_offset` merely gated at 0.5. `const_offset` adds
    one offset to EVERY frame, so forcing it makes a sparse keypoint's corruption deterministic.

    Inputs: cfg -- the `[scorer.corruption]` block.
    Outputs: (dense_3d, dense_2d, sparse_3d, sparse_2d).
    Side effects: none, or raises AssertionError when the forced magnitude is zero.
    """
    dense_3d, dense_2d = build_corruptors(cfg)
    probs = {n: 0.0 for n in GENERATORS}
    probs['const_offset'] = 1.0
    for n in SPARSE_TYPES:
        if n != 'const_offset':
            probs[n] = cfg.get(f'{n}_prob', 0.0)
    mag_3d = dict(cfg.get('mag_3d', {}))
    mag_2d = dict(cfg.get('mag_2d', {}))
    if not (mag_3d.get('const_offset', 0.0) > 0 and mag_2d.get('const_offset', 0.0) > 0):
        raise ValueError(
            'the sparse corruption menu forces const_offset, so its magnitude must be nonzero in '
            f'BOTH mag_3d ({mag_3d.get("const_offset")}) and mag_2d '
            f'({mag_2d.get("const_offset")}); otherwise a sparse keypoint is corrupted by nothing '
            'and the window would train the scorer to call a corrupt sample clean.')
    sparse_3d = PointCorruptor(probs, mag_3d)
    sparse_2d = PointCorruptor(probs, mag_2d)
    return dense_3d, dense_2d, sparse_3d, sparse_2d


def observed_frame_counts(coords, frames):
    """Per keypoint, how many DISTINCT SOURCE FRAMES carry a finite coordinate.

    `_frames` clamp-pads at group edges, so a T=12 window over a 4-frame group repeats frames 0..3
    three times. Counting finite SLOTS would then accept a track that is really four observations
    shown three times each, which is not the same evidence and must not pass `min_valid_frames`.

    Inputs: coords -- [b,t,k,R]; frames -- [t] source frame indices for this window.
    Outputs: [b,k] long tensor of distinct observed frame counts.
    Side effects: none.
    """
    valid = torch.isfinite(coords).all(-1)
    b, t, k = valid.shape
    frames = [int(f) for f in np.asarray(frames).reshape(-1)]
    assert len(frames) == t, f'{len(frames)} frame indices for a T={t} window'
    counts = torch.zeros((b, k), dtype=torch.long, device=coords.device)
    for value in sorted(set(frames)):
        ts = [i for i, f in enumerate(frames) if f == value]
        counts = counts + valid[:, ts, :].any(dim=1).long()
    return counts


def _compute_drop_mask(coords, cfg, counts):
    """Boolean [b,t,k] mask of coord slots to NaN out, SHARED by good and bad.

    FORKED from the reference, and only in what it counts. The reference's floor is the number of
    finite SLOTS (`valid.sum(dim=1)`); this one is `counts`, the distinct observed SOURCE frames
    from `observed_frame_counts`, so clamp-padded repeats cannot satisfy it. Everything else -- the
    50/50 contiguous-window vs per-frame-bernoulli choice, the `point_drop_prob` gate, the
    restriction to currently-valid frames, the cumulative cap -- is the reference's.

    A missing slot is orthogonal to track quality, which is the whole point of the shared mask and
    the `missing_point` token: if good and bad dropped DIFFERENT slots, the gap would partly be a
    missingness signal.

    Inputs: coords -- [b,t,k,R]; cfg -- the `[scorer.corruption]` block; counts -- [b,k] distinct
            observed frame counts (the per-point floor).
    Outputs: [b,t,k] bool, or None when `point_drop_prob` is 0.
    Side effects: draws from the ambient torch RNG.
    """
    p = cfg.get('point_drop_prob', 0.0)
    if p <= 0.0:
        return None
    max_frac = cfg.get('point_drop_max_frac', 0.4)
    rate = cfg.get('point_drop_bernoulli_rate', 0.2)
    b, t, k, _ = coords.shape
    device = coords.device

    valid = torch.isfinite(coords).all(dim=-1)
    max_drop = (counts - int(cfg.get('min_valid_frames', 1))).clamp(min=0)

    L_max = max(1, int(math.ceil(max_frac * t)))
    length = torch.randint(1, L_max + 1, (b, k), device=device)
    start = (torch.rand(b, k, device=device) * (t - length + 1).clamp(min=1).float()).long()
    ar = torch.arange(t, device=device)[None, :, None]
    window = (ar >= start[:, None, :]) & (ar < (start + length)[:, None, :])

    bern = torch.rand(b, t, k, device=device) < rate

    use_window = (torch.rand(b, k, device=device) < 0.5)[:, None, :]
    gate = (torch.rand(b, k, device=device) < p)[:, None, :]
    cand = torch.where(use_window, window, bern) & gate & valid

    order = cand.cumsum(dim=1)
    return cand & (order <= max_drop[:, None, :])


def view_affine_2d(view):
    """The source-pixels -> this-view-pixels affine, as a 3x3 float64 matrix.

    A 2D view's coordinates ARE pixels of its own final crop, so every step of the chain moves
    them. The chain is rotate -> crop -> resize, and the composed matrix is
    `S(scale) @ T(-box[:2]) @ R(rot)`. It is built from the pieces the realisation ALREADY
    returned rather than re-derived from the camera dicts: a re-derivation is how two frames come
    to disagree about where the animal is.

    Inputs: view -- a `View` from a single-camera 2D realisation.
    Outputs: a (3,3) float64 tensor.
    Side effects: none.
    """
    assert len(view.cgroup) == 1, 'a 2D view is single-camera'
    scale = torch.as_tensor(view.scale[0], dtype=torch.float64)
    if scale.ndim == 0:
        scale = scale.repeat(2)
    box = view.boxes[0]
    x1, y1 = float(box[0]), float(box[1])

    H = torch.eye(3, dtype=torch.float64)
    rot = view.rotation[0]
    if rot is not None:
        M = np.asarray(rot[0], dtype=np.float64)
        R = torch.eye(3, dtype=torch.float64)
        R[:2, :2] = torch.as_tensor(M[:, :2])
        R[:2, 2] = torch.as_tensor(M[:, 2])
        H = R @ H
    T = torch.eye(3, dtype=torch.float64)
    T[0, 2] = -x1
    T[1, 2] = -y1
    H = T @ H
    S = torch.eye(3, dtype=torch.float64)
    S[0, 0], S[1, 1] = scale[0], scale[1]
    return S @ H


def transfer_points_2d(pts, H_from, H_to):
    """Move (...,2) pixel points from one 2D view's frame into another's.

    `x_to = H_to @ inv(H_from) @ x_from` -- undo the source view back to source pixels, then apply
    the target view's chain. Both matrices are source->view, so the composition is exact rather
    than approximate. NaN slots stay NaN, which is what keeps a dropped point dropped.

    Inputs: pts -- (...,2) tensor; H_from / H_to -- (3,3) source->view matrices.
    Outputs: (...,2) tensor in the target view's frame.
    Side effects: none.
    """
    M = (H_to @ torch.linalg.inv(H_from)).to(pts.dtype)
    flat = pts.reshape(-1, 2)
    ones = torch.ones((flat.shape[0], 1), dtype=flat.dtype, device=flat.device)
    out = torch.cat([flat, ones], dim=-1) @ M.t()
    return out[:, :2].reshape(pts.shape)


def _visible_points_mask(coords, cgroup, mode, cam_thresh):
    """Per-point mask: visible in >= `cam_thresh` cameras in at least one frame.

    Mirrors the reference's, and exists because an in-plane rotation can swing an off-centre track
    fully off-frame, where the crop (clamped to the image) cannot recover it. NaN counts as
    not-visible, so a point survives while it is seen in any un-dropped frame.

    Inputs: coords -- [t,k,R] for ONE member; cgroup -- its cameras; mode -- '2d' or '3d';
            cam_thresh -- how many cameras must see it.
    Outputs: [k] bool.
    Side effects: none.
    """
    t, k, _ = coords.shape
    if mode == '3d':
        flat = rearrange(coords, 't k r -> (t k) r')
        vis = torch.stack([is_point_visible(cam, flat) for cam in cgroup])
        vis = rearrange(vis, 'cams (t k) -> t k cams', t=t, k=k)
        n_cams_vis = vis.sum(dim=-1)
    else:
        size = cgroup[0]['size']
        c = torch.nan_to_num(coords, nan=-1e9)
        w, h = float(size[0]), float(size[1])
        inside = ((c[..., 0] >= 0) & (c[..., 0] < w) & (c[..., 1] >= 0) & (c[..., 1] < h))
        n_cams_vis = inside.to(torch.int64)
    return (n_cams_vis >= cam_thresh).any(dim=0)



def _frame_visibility(coords, camera_group, mode, cam_thresh=1):
    """Return per-frame visibility and the per-camera visibility matrix.

    ``coords`` is ``[B,T,K,R]`` in the final view coordinate system.  The first result is true
    only when the coordinate is finite and visible in at least ``cam_thresh`` cameras.  Keeping
    the camera matrix as a second result is important for 3-D displacement: a ray-parallel shift
    must be measured only in cameras where the clean point is actually visible.
    """
    if coords.ndim != 4:
        raise ValueError(f'coords must be [B,T,K,R], got {tuple(coords.shape)}')
    finite = torch.isfinite(coords).all(-1)
    if mode == '2d':
        if len(camera_group) != 1:
            raise ValueError('2D scorer triplets require exactly one camera')
        w, h = (float(x) for x in camera_group[0]['size'][:2])
        inside = (finite & (coords[..., 0] >= 0) & (coords[..., 0] < w)
                  & (coords[..., 1] >= 0) & (coords[..., 1] < h))
        return inside, inside.unsqueeze(-1)
    if mode != '3d':
        raise ValueError(f'unknown scorer triplet mode {mode!r}')
    b, t, k, _ = coords.shape
    flat = coords.reshape(-1, coords.shape[-1])
    cams = []
    for cam in camera_group:
        vis = is_point_visible(cam, flat).reshape(b, t, k)
        cams.append(vis & finite)
    if not cams:
        per_cam = torch.zeros((*finite.shape, 0), dtype=torch.bool, device=coords.device)
    else:
        per_cam = torch.stack(cams, dim=-1)
    return per_cam.sum(-1) >= int(cam_thresh), per_cam


def _projected_displacement(good, bad, camera_group, good_per_camera=None):
    """Image-plane displacement of ``good`` to ``bad`` in final-crop pixels.

    The scalar is the maximum over cameras on which the clean point is visible.  The companion
    mean is useful for a census but is not used as the active gate.  No visible camera yields NaN;
    callers keep that slot out of all masks rather than silently treating missing geometry as a
    zero displacement.
    """
    if good.shape != bad.shape or good.ndim != 4:
        raise ValueError('good and bad must have identical [B,T,K,R] shapes')
    if good_per_camera is None:
        _, good_per_camera = _frame_visibility(good, camera_group, '3d')
    gp = project_points_torch(camera_group, good)
    bp = project_points_torch(camera_group, bad)
    finite = torch.isfinite(gp).all(-1) & torch.isfinite(bp).all(-1)
    d = torch.linalg.vector_norm(bp - gp, dim=-1)
    use = good_per_camera.permute(3, 0, 1, 2) & finite
    neg_inf = torch.full_like(d, -torch.inf)
    max_d = torch.where(use, d, neg_inf).amax(dim=0)
    sum_d = torch.where(use, d, torch.zeros_like(d)).sum(dim=0)
    n = use.sum(dim=0)
    max_d = torch.where(n > 0, max_d, torch.full_like(max_d, float('nan')))
    mean_d = torch.where(n > 0, sum_d / n.clamp_min(1), torch.full_like(sum_d, float('nan')))
    return max_d, mean_d, gp, bp, good_per_camera


def _distance_2d(a, b):
    """Pixel distance for two final-view 2-D coordinate tensors, preserving NaN slots."""
    finite = torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)
    d = torch.linalg.vector_norm(b - a, dim=-1)
    return torch.where(finite, d, torch.full_like(d, float('nan')))


def _reference_distances(good, bad, reference, mode, camera_group, good_per_camera,
                         good_proj=None, bad_proj=None):
    """Return ``(good-reference, bad-reference)`` pixel distances.

    ``reference`` is expected in view A's final coordinate system.  For 3-D all distances use the
    same cameras where ``good`` is visible, matching the primary displacement gate.  A reference
    with no finite/evaluable coordinate yields NaN and therefore cannot certify a negative.
    """
    if reference is None:
        return None, None
    if reference.ndim == good.ndim - 1:
        reference = reference.unsqueeze(0)
    if tuple(reference.shape) != tuple(good.shape):
        raise ValueError(
            f'reference must match good shape {tuple(good.shape)}, got {tuple(reference.shape)}')
    if mode == '2d':
        return _distance_2d(good, reference), _distance_2d(bad, reference)
    rp = project_points_torch(camera_group, reference)
    if good_proj is None:
        good_proj = project_points_torch(camera_group, good)
    if bad_proj is None:
        bad_proj = project_points_torch(camera_group, bad)
    finite_ref = torch.isfinite(rp).all(-1)
    gfinite = torch.isfinite(good_proj).all(-1)
    bfinite = torch.isfinite(bad_proj).all(-1)
    good_use = good_per_camera.permute(3, 0, 1, 2)
    use_g = good_use & finite_ref & gfinite
    use_b = good_use & finite_ref & bfinite
    gd = torch.linalg.vector_norm(good_proj - rp, dim=-1)
    bd = torch.linalg.vector_norm(bad_proj - rp, dim=-1)
    ng, nb = use_g.sum(0), use_b.sum(0)
    g = torch.where(use_g, gd, torch.zeros_like(gd)).sum(0) / ng.clamp_min(1)
    b = torch.where(use_b, bd, torch.zeros_like(bd)).sum(0) / nb.clamp_min(1)
    g = torch.where(ng > 0, g, torch.full_like(g, float('nan')))
    b = torch.where(nb > 0, b, torch.full_like(b, float('nan')))
    return g, b


def displacement_masks(good, bad, anchor, camera_group, mode, cfg=None, *,
                       anchor_camera_group=None, moved_mask=None, reference=None,
                       cam_thresh=1):
    """Build framewise displacement, visibility, reference and supervision masks.

    This is the data-side contract consumed by :class:`FrameTripletScorerLoss`.  All returned
    masks have ``[B,T,K]`` shape (except the per-camera diagnostic tensors), and all thresholds are
    in final-crop pixels.  In 3-D the displacement is the maximum selected-camera reprojection
    distance over cameras where the clean point is visible; in 2-D it is the direct pixel norm.

    ``reference`` is optional independent/reference coordinates in view A.  With
    ``reference_gate='independent_far'`` it is required.  ``source_far`` deliberately uses the
    stored clean point as a proxy and is reported in ``reference_source``; it is not a claim that
    the stored track is ground truth.  ``reference_gate='none'`` leaves the base far state alone.

    Outputs include ``base_far_mask`` (before reference gating), ``reference_far_mask``,
    ``reference_rejected_mask``, ``far_mask``, ``near_mask``, ``ambiguous_mask``, ``active_mask``,
    ``observed_mask``, ``anchor_observed_mask``, ``in_view_mask``, ``moved_mask``,
    ``displacement_px`` and ``displacement_mean_px``.  The result also carries reference distance
    diagnostics and the gate/source names as scalar Python metadata.
    """
    cfg = {} if cfg is None else dict(cfg)
    if good.ndim != 4 or bad.shape != good.shape or anchor.ndim != good.ndim:
        raise ValueError('good, bad and anchor must be [B,T,K,R] tensors of matching shape')
    if anchor.shape != good.shape:
        raise ValueError(
            f'anchor shape {tuple(anchor.shape)} must match good {tuple(good.shape)}')
    anchor_camera_group = camera_group if anchor_camera_group is None else anchor_camera_group
    mode = str(mode)
    obs = torch.isfinite(good).all(-1) & torch.isfinite(bad).all(-1)
    anchor_obs = torch.isfinite(anchor).all(-1)
    good_view, good_per_cam = _frame_visibility(good, camera_group, mode, cam_thresh)
    bad_view, _ = _frame_visibility(bad, camera_group, mode, cam_thresh)
    anchor_view, _ = _frame_visibility(anchor, anchor_camera_group, mode, cam_thresh)
    in_view = good_view & bad_view & anchor_view

    good_proj = bad_proj = None
    if mode == '2d':
        displacement = _distance_2d(good, bad)
        displacement_mean = displacement.clone()
    elif mode == '3d':
        (displacement, displacement_mean, good_proj, bad_proj, good_per_cam) = (
            _projected_displacement(good, bad, camera_group, good_per_cam))
    else:
        raise ValueError(f'unknown scorer triplet mode {mode!r}')

    finite_d = torch.isfinite(displacement)
    if moved_mask is None:
        moved = finite_d & (displacement > 0)
    else:
        moved = moved_mask.to(device=good.device, dtype=torch.bool) & finite_d
    min_px = float(cfg.get('min_corrupt_px', 0.0))
    max_px = float(cfg.get('max_clean_px', 0.0))
    if (min_px < 0 or max_px < 0 or (min_px > 0 and min_px <= max_px)
            or (min_px == 0 and max_px > 0)):
        raise ValueError(
            f'distance thresholds must satisfy 0 <= max_clean_px < min_corrupt_px when a '
            f'far threshold is configured, or both zero when using moved-mask fallback; '
            f'got min={min_px}, max={max_px}')
    base_far = finite_d & moved & (displacement >= min_px if min_px > 0 else moved)
    near = finite_d & (displacement <= max_px)
    ambiguous = finite_d & ~base_far & ~near

    gate_default = 'source_far'
    reference_gate = str(cfg.get('reference_gate', gate_default))
    reference_source = 'none'
    good_ref = bad_ref = None
    if reference_gate == 'none':
        reference_far = torch.ones_like(base_far)
    elif reference_gate == 'source_far':
        reference_source = 'stored_good_proxy'
        good_ref = torch.zeros_like(displacement)
        bad_ref = displacement.clone()
        reference_far = finite_d & moved & (bad_ref >= min_px if min_px > 0 else moved)
    elif reference_gate == 'independent_far':
        if reference is None:
            raise ValueError(
                'reference_gate="independent_far" requires independent reference coordinates')
        reference_source = 'independent'
        good_ref, bad_ref = _reference_distances(
            good, bad, reference, mode, camera_group, good_per_cam, good_proj, bad_proj)
        margin = float(cfg.get('reference_margin_px', 0.0))
        if margin < 0:
            raise ValueError(f'reference_margin_px must be non-negative, got {margin}')
        reference_far = (torch.isfinite(good_ref) & torch.isfinite(bad_ref)
                         & (bad_ref >= min_px if min_px > 0 else (bad_ref > 0))
                         & (bad_ref >= good_ref + margin))
    else:
        raise ValueError(
            f'unknown reference_gate {reference_gate!r}; expected none, source_far or '
            'independent_far')

    reference_rejected = base_far & ~reference_far
    far = base_far & reference_far
    active = far & obs & in_view & anchor_obs
    out = {
        'observed_mask': obs,
        'anchor_observed_mask': anchor_obs,
        'in_view_mask': in_view,
        'moved_mask': moved if moved_mask is None else moved_mask.to(torch.bool),
        'displacement_px': displacement,
        'displacement_mean_px': displacement_mean,
        'base_far_mask': base_far,
        'reference_far_mask': reference_far,
        'reference_rejected_mask': reference_rejected,
        'far_mask': far,
        'near_mask': near,
        'ambiguous_mask': ambiguous,
        'active_mask': active,
        'reference_gate': reference_gate,
        'reference_source': reference_source,
        'good_reference_distance_px': good_ref,
        'bad_reference_distance_px': bad_ref,
    }
    return out


def _attach_segment_frames(metadata, unique_frames):
    """Fill segment metadata's source-frame fields in place and return it."""
    for point in metadata:
        for pieces in point.values():
            for piece in pieces:
                st = int(piece['start'])
                length = int(piece['length'])
                piece['source_start'] = int(unique_frames[st])
                piece['source_frames'] = [int(x) for x in unique_frames[st:st + length]]
    return metadata


def _metadata_for_k(metadata, b, k):
    """Reshape flattened ``[B*K]`` segment metadata to ``[B][K]``."""
    return [[metadata[bi * k + ki] for ki in range(k)] for bi in range(b)]


def _make_anchor(source, anchor_label, mode, view_a, view_b):
    """Carry a source coordinate tensor from view A into the independent anchor view."""
    if mode == '3d':
        return source
    return transfer_points_2d(source[0], view_affine_2d(view_a),
                              view_affine_2d(view_b))[None]


def make_triplet(dataset, sel, rng, cfg, corruptors, cam_thresh=1, reference=None):
    """Build a (good, bad, anchor) triplet and its framewise supervision metadata.

    The historical geometry is intentionally unchanged: one clean view A is shared by ``good``
    and ``bad``, while ``anchor`` is an independently realised view B.  New metadata is computed
    after the shared drop mask and final crop.  ``frames`` is carried explicitly, and corruption
    draws use the unique source-frame sequence before being gathered back to local slots.

    ``reference`` is optional independent coordinates in the *source* coordinate system.  It is
    transformed into view A for 2-D and used directly for 3-D.  Normal tracked-root operation has
    no independent reference; selecting ``reference_gate='source_far'`` therefore records the
    stored clean track as a proxy rather than silently implying ground truth.

    Outputs retain the old keys (including ``fired`` [B,K,G], good/bad/anchor tuples and counts)
    and add aligned ``[B,T,K]`` masks: ``observed_mask``, ``anchor_observed_mask``,
    ``in_view_mask``, ``moved_mask``, ``displacement_px``, ``reference_far_mask``, ``far_mask``,
    ``near_mask``, ``ambiguous_mask``, ``active_mask`` and ``source_frame_weight``.  The per-type
    ``corruption_type_mask`` is ``[B,T,K,G]`` and means type-fired *and* final-far.  ``segments``
    and ``unique_frames`` are retained as Python metadata for provenance/QC.

    The function returns ``None`` for ordinary view/keypoint failures, or for a distance-gated draw
    with no evaluable active row.  Sparse keypoints redraw their corruption up to the existing
    retry cap when the distance gate leaves them with no active far slot.
    """
    dense_3d, dense_2d, sparse_3d, sparse_2d = corruptors
    view_a = dataset._realise(sel, rng, world_gauge=False)
    if view_a is None:
        return None
    view_b = dataset._realise(sel, rng, world_gauge=False)
    if view_b is None:
        return None

    mode = '2d' if view_a.r == 2 else '3d'
    coords = view_a.coords[None]
    if coords.shape[0] != 1:
        raise ValueError('the scorer builds one window per triplet (batch_size 1)')
    frames_np, inverse = _unique_source_frames(sel.frames)
    counts = observed_frame_counts(coords, sel.frames)

    alive = counts[0] > 0
    if int(alive.sum()) < 2:
        return None
    coords = coords[:, :, alive]
    counts = counts[:, alive]
    K = coords.shape[2]

    cube_scale_b = None
    if mode == '3d':
        scale = get_camera_scale(view_a.cgroup, rearrange(coords, 'b t k r -> b (t k) r'))
        cube_scale_b = torch.median(scale, dim=0).values

    dense = dense_3d if mode == '3d' else dense_2d
    sparse = sparse_3d if mode == '3d' else sparse_2d
    is_dense = (counts >= int(cfg.get('min_valid_frames', 1)))[0]

    shift_dense, fired_dense, seg_dense, meta_dense = _draw_shift(
        dense, coords, frames=sel.frames, cfg=cfg, return_metadata=True)
    shift_sparse, fired_sparse, seg_sparse, meta_sparse = _draw_sparse_moved(
        sparse, coords, is_dense, frames=sel.frames, cfg=cfg, return_metadata=True)
    pick = is_dense.view(1, 1, K, 1)
    pick_type = is_dense.view(1, K, 1)
    shift = torch.where(pick, shift_dense, shift_sparse)
    fired = torch.where(pick_type, fired_dense, fired_sparse)
    seg_gate = torch.where(pick_type[:, None, :, :], seg_dense, seg_sparse)
    metadata_flat = [meta_dense[i] if bool(is_dense[i]) else meta_sparse[i] for i in range(K)]
    if mode == '3d':
        shift = shift * cube_scale_b[:, None, None, None]

    drop_mask = _compute_drop_mask(coords, cfg, counts)
    good = apply_drop_mask(coords, drop_mask)

    anchor_label = 1.0 if float(rng.random()) < 0.5 else -1.0
    reference_a = None
    if reference is not None:
        reference_a = torch.as_tensor(reference, dtype=coords.dtype, device=coords.device)
        if reference_a.ndim == 3:
            reference_a = reference_a[None]
        if mode == '2d':
            identity = torch.eye(3, dtype=torch.float64, device=reference_a.device)
            reference_a = transfer_points_2d(reference_a[0], identity,
                                             view_affine_2d(view_a))[None]
        if reference_a.ndim != 4 or reference_a.shape[:2] != coords.shape[:2]:
            raise ValueError(
                f'reference must align with source [T,K,R], got {tuple(reference_a.shape)}')
        reference_a = reference_a[:, :, alive]

    frame_mode = str(cfg.get('output_granularity', 'sequence')) == 'frame'
    sequence_far_gate = bool(cfg.get('sequence_far_gate', False))
    mask_data = None
    keep = None
    for sparse_attempt in range(SPARSE_DRAW_RETRIES + 1):
        bad = apply_drop_mask(coords + shift, drop_mask)
        source = good if anchor_label > 0 else bad
        anchor = _make_anchor(source, anchor_label, mode, view_a, view_b)
        mask_data = displacement_masks(
            good, bad, anchor, view_a.cgroup, mode, cfg,
            anchor_camera_group=view_b.cgroup,
            moved_mask=shift.abs().sum(-1) > 0,
            reference=reference_a,
            cam_thresh=cam_thresh)
        keep = (_visible_points_mask(good[0], view_a.cgroup, mode, cam_thresh)
                & _visible_points_mask(bad[0], view_a.cgroup, mode, cam_thresh)
                & _visible_points_mask(anchor[0], view_b.cgroup, mode, cam_thresh))
        sparse_active = mask_data['active_mask'][0].any(dim=0) & (~is_dense)
        if (not frame_mode and not sequence_far_gate) or bool(
                sparse_active.all() if bool((~is_dense).any()) else True):
            break
        if sparse_attempt >= SPARSE_DRAW_RETRIES:
            return None
        shift_sparse, fired_sparse, seg_sparse, meta_sparse = _draw_sparse_moved(
            sparse, coords, is_dense, frames=sel.frames, cfg=cfg, return_metadata=True)
        shift = torch.where(pick, shift_dense, shift_sparse)
        fired = torch.where(pick_type, fired_dense, fired_sparse)
        seg_gate = torch.where(pick_type[:, None, :, :], seg_dense, seg_sparse)
        metadata_flat = [meta_dense[i] if bool(is_dense[i]) else meta_sparse[i]
                         for i in range(K)]
        if mode == '3d':
            shift = shift * cube_scale_b[:, None, None, None]

    n_keep = int(keep.sum())
    if n_keep < 2:
        return None

    active_before_keep = mask_data['active_mask']
    if (frame_mode or sequence_far_gate) and not bool(active_before_keep.any()):
        return None

    gv = [v[None] for v in view_a.views]
    av = [v[None] for v in view_b.views]
    masks = {key: value[:, :, keep] if torch.is_tensor(value) and value.ndim >= 3 else value
             for key, value in mask_data.items()}
    if (frame_mode or sequence_far_gate) and not bool(masks['active_mask'].any()):
        return None
    fired_keep = fired[:, keep, :]
    type_mask = fired_keep[:, None, :, :].expand(-1, coords.shape[1], -1, -1)
    type_mask = type_mask & masks['far_mask'][..., None]
    eligible = masks['observed_mask'] & masks['in_view_mask'] & masks['anchor_observed_mask']
    weights = source_frame_weights(sel.frames, eligible=eligible, device=coords.device)
    metadata_flat = _attach_segment_frames(metadata_flat, frames_np)
    seg_local = seg_gate[:, :, keep, :]
    kpt_ids = dataset._kpt_ids[sel.sess.path][alive][keep][None]
    out = {
        'good': (gv, good[:, :, keep], view_a.cgroup),
        'bad': (gv, bad[:, :, keep], view_a.cgroup),
        'anchor': (av, anchor[:, :, keep], view_b.cgroup),
        'kpt_ids': kpt_ids,
        'anchor_label': anchor_label,
        'mode': mode,
        'source': sel.sess.label_source,
        'sequence_far_gate': sequence_far_gate,
        'reuse_scene_for_anchor': False,
        'occlusion': None,
        'counts': counts[:, keep],
        'fired': fired_keep,
        'frames': torch.as_tensor(sel.frames, dtype=torch.long, device=coords.device),
        'unique_frames': torch.as_tensor(frames_np, dtype=torch.long, device=coords.device),
        'frame_inverse': torch.as_tensor(inverse, dtype=torch.long, device=coords.device),
        'n_unique_frames': int(len(frames_np)),
        'n_duplicate_slots': int(len(sel.frames) - len(frames_np)),
        'segment_mask': seg_local,
        'segments': [[metadata_flat[i] for i in range(K) if bool(keep[i])]],
        'corruption_type_mask': type_mask,
        'fired_frame': fired_keep[:, None, :, :].expand(-1, coords.shape[1], -1, -1),
        'source_frame_weight': weights,
        'max_clean_px': float(cfg.get('max_clean_px', 0.0)),
        'n_dense': int(is_dense[keep].sum()),
        'n_sparse': int((~is_dense[keep]).sum()),
    }
    for key, value in masks.items():
        if key in ('reference_gate', 'reference_source'):
            out[key] = value
        elif key not in ('good_reference_distance_px', 'bad_reference_distance_px'):
            out[key] = value
        elif value is not None:
            out[key] = value
    out['shift'] = shift[:, :, keep]
    return out


def _draw_shift(corruptor, coords, frames=None, cfg=None, return_metadata=False):
    """Draw corruption on unique source frames, then gather it to local slots.

    With an ordinary non-duplicated window and no segment gate this calls ``tagged_draw`` with the
    same ``(P,T,D)`` and therefore preserves the historical RNG realization exactly.  A clamped
    window first draws ``T_unique`` slots and gathers by inverse source-frame index, so repeated
    local copies cannot receive contradictory noise.  ``return_metadata`` adds the segment mask and
    per-point interval records used by :func:`make_triplet`.
    """
    b, t, k, dim = coords.shape
    if frames is None:
        frames = np.arange(t, dtype=np.int64)
    cfg = {} if cfg is None else cfg
    gated_mode = (str(cfg.get('output_granularity', 'sequence')) == 'frame'
                  or bool(cfg.get('sequence_far_gate', False)))
    if gated_mode:
        unique, inverse = _unique_source_frames(frames)
    else:
        unique = np.asarray(frames, dtype=np.int64)
        inverse = np.arange(len(unique), dtype=np.int64)
    use_segments = gated_mode and float(cfg.get('segment_prob', 0.0)) > 0.0
    if use_segments:
        shifts_u, fired, seg_u, metadata = tagged_draw_segmented(
            corruptor, b * k, len(unique), dim, coords.device, cfg)
    else:
        shifts_u, fired = tagged_draw(corruptor, b * k, len(unique), dim, coords.device)
        seg_u = torch.ones((b * k, len(unique), len(GENERATORS)), dtype=torch.bool,
                           device=coords.device)
        metadata = [
            {name: ([{'start': 0, 'length': int(len(unique)), 'full_window': True,
                      'source_start': None, 'source_frames': None}]
                     if name in corruptor.names else []) for name in GENERATORS}
            for _ in range(b * k)]
    gather = torch.as_tensor(inverse, dtype=torch.long, device=coords.device)
    shifts = shifts_u.index_select(1, gather)
    seg = seg_u.index_select(1, gather)
    shift = shifts.reshape(b, k, t, dim).permute(0, 2, 1, 3).contiguous()
    fired = fired.reshape(b, k, len(GENERATORS))
    seg = seg.reshape(b, k, t, len(GENERATORS)).permute(0, 2, 1, 3).contiguous()
    if return_metadata:
        return shift, fired, seg, metadata
    return shift, fired


def _sparse_stuck(shift, coords, is_dense):
    """[b,k] bool: a SPARSE keypoint whose drawn corruption moves no observed slot.

    `const_offset` moves every frame once it fires, and `frame_noise` moves a contiguous window --
    which need not cover the ONE frame a sparse keypoint is observed in. A sparse keypoint whose
    `const_offset` did not fire and whose `frame_noise` window missed is corrupted by nothing, and
    training on it would teach the scorer that a corrupt sample is clean.

    Inputs: shift -- [b,t,k,R] the drawn (unit-space) shift; coords -- [b,t,k,R];
            is_dense -- [b,k] bool.
    Outputs: [b,k] bool.
    Side effects: none.
    """
    valid = torch.isfinite(coords).all(-1)
    moved = (shift.abs().sum(-1).masked_fill(~valid, 0.0) > 0).any(dim=1)
    return (~is_dense) & (~moved)


def _draw_sparse_moved(sparse_corruptor, coords, is_dense, frames=None, cfg=None,
                        return_metadata=False):
    """Draw sparse corruption, redrawing until every sparse point moves an observed slot.

    The optional distance-gated retry is handled by ``make_triplet`` after drop/view masks are
    available.  This helper retains the old moved-slot invariant and its two-return-value API by
    default; callers requesting metadata receive ``(shift, fired, segment_mask, metadata)``.
    """
    def draw():
        """Draw one sparse candidate on the configured source-frame lattice."""
        return _draw_shift(sparse_corruptor, coords, frames=frames, cfg=cfg,
                           return_metadata=True)
    shift, fired, seg, metadata = draw()
    for _ in range(SPARSE_DRAW_RETRIES):
        stuck = _sparse_stuck(shift, coords, is_dense)
        if not bool(stuck.any()):
            return ((shift, fired, seg, metadata) if return_metadata else (shift, fired))
        shift, fired, seg, metadata = draw()
    stuck = _sparse_stuck(shift, coords, is_dense)
    raise ValueError(
        f'{int(stuck.sum())} sparse keypoint(s) still moved no observed slot after '
        f'{SPARSE_DRAW_RETRIES} redraws of a menu restricted to types that cannot cancel. That '
        'is not a sampling accident, so the observed-frame count or the magnitudes are wrong.')


def triplet_collate(batch):
    """Collate for the scorer: one full batch-1 triplet dict per step.

    The scorer runs `batch_size = 1` because each camera's rotated crop has its own variable size,
    so there is nothing to stack along a batch axis.

    Inputs: batch -- a list holding exactly one triplet dict (or None).
    Outputs: that dict, or None.
    Side effects: none.
    """
    assert len(batch) == 1, 'the scorer runs batch_size=1 (variable per-camera rotated sizes)'
    return batch[0]


def seed_worker(worker_id):
    """DataLoader `worker_init_fn`: decorrelate numpy's RNG across workers.

    torch and Python's `random` are auto-seeded per worker, but numpy is NOT -- and
    `PosetailDataset.rotate_camera_group` draws the 3D world-gauge angle from global numpy.

    Inputs: worker_id -- the DataLoader's worker index.
    Outputs: none.
    Side effects: reseeds numpy's global RNG in this worker process.
    """
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)
