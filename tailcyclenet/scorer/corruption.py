"""Synthetic track corruption: which corruptions a window can take, and drawing them in order.

`GENERATORS`, `PointCorruptor` and `build_corruptors` are tracktail's, imported verbatim;
`compute_drop_mask` is FORKED from the reference's `compute_drop_mask`. The draw order here is an
RNG invariant: `tagged_draw` is held in lockstep with `PointCorruptor` by a test.
"""
import math

import numpy as np
import torch

from tracktail.datasets.scorer_corruption import GENERATORS, PointCorruptor, build_corruptors


# The corruption types that can move a SINGLE observed slot, which is all a sparse window can be
# trained on. `gradual_drift` and `sinusoid` ramp from or through zero, so at one observed frame
# they may corrupt nothing at all -- and a corruption that changes nothing would teach the scorer
# to call a corrupt sample clean, which is worse than excluding the window. `const_offset` is
# FORCED for sparse keypoints (see `build_corruptors_for`); `frame_noise` rides alongside it.
SPARSE_TYPES = ('const_offset', 'frame_noise')

# Full-trajectory generators are optionally restricted to a contiguous interval.  ``frame_noise``
# already has this behaviour in tracktail and is intentionally not included here.  Keeping the
# names in one place also makes the per-type mask's last axis stable (it is the order of
# ``GENERATORS``, not the reduced sparse menu).
SEGMENT_TYPES = ('const_offset', 'gradual_drift', 'sinusoid')
SPARSE_DRAW_RETRIES = 16


def unique_source_frames(frames):
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
    raw = cfg.get('segment_len_frames', (1, n_unique))
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
    full = float(cfg.get('full_window_share', 0.0))
    if not 0.0 <= full <= 1.0:
        raise ValueError(f'segment full-window share must be in [0,1], got {full}')
    return lo, hi, full


def _segment_mask(P, n_unique, device, cfg, names):
    """Draw per-point/type contiguous segment masks and serialisable metadata.

    A segment gate is a *partial-subsequence probability*: when a full-trajectory corruption type
    fires, ``segment_prob`` says how often its otherwise full-window contribution is restricted to
    a random interval.  ``full_window_share`` is an explicit escape hatch for the identity
    swap/failure case.  Segment starts and lengths are sampled over unique source-frame positions,
    never over clamp-duplicated local slots.

    Outputs are ``(mask, metadata)``.  ``mask`` is ``[P,T_unique,len(GENERATORS)]`` and metadata is
    a list of ``P`` dictionaries, each with one list of segments per generator name.  A type not in
    ``names`` has an empty list and an all-ones mask.
    """
    p = float(cfg.get('segment_prob', 0.0))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f'segment_prob must be in [0,1], got {p}')
    count_cfg = cfg.get('segment_count', 1)
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


def tagged_draw(corruptor, P, T, D, device, segment_masks=None):
    """A `PointCorruptor` draw that ALSO reports which types fired for each point.

    Gate B needs `triplet_acc` broken out per corruption type: an aggregate that hides
    "sinusoid 0.99 / const_offset 0.51" is not a measurement, and a type at chance means that
    corruption is invisible and its magnitude is wrong. The library's `PointCorruptor.__call__`
    returns only the summed shift, so this reproduces its algorithm exactly and adds tags.

    Kept in step with the library by `tests/scorer/test_triplet.py`, which asserts this and
    `PointCorruptor` produce the SAME tensor from the same RNG state -- without that, the tags
    would describe a different draw from the one the model was trained on.

    `segment_masks` ([P,T,len(GENERATORS)] bool, from `_segment_mask`, drawn by the caller BEFORE
    this call so the RNG order is unchanged) restricts each type's contribution to its segments;
    None is the library's full-trajectory draw, byte for byte.

    Inputs: corruptor -- a `PointCorruptor`; P, T, D -- points, frames, coordinate dim;
            device -- where to build the draw; segment_masks -- optional per-type segment gates.
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
        if segment_masks is not None:
            shift = shift * segment_masks[:, :, slot[name], None].float()
        total = total + shift * mask[:, None, None].float()
        applied = applied | mask
        fired[:, slot[name]] |= mask

    need = ~applied
    if need.any():
        pick = torch.randint(0, len(names), (P,), device=device)
        for ci, name in enumerate(names):
            m = need & (pick == ci)
            if m.any():
                shift = contributions[name]
                if segment_masks is not None:
                    shift = shift * segment_masks[:, :, slot[name], None].float()
                total = total + shift * m[:, None, None].float()
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


def compute_drop_mask(coords, cfg, counts):
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


def draw_shift(corruptor, coords, frames=None, cfg=None, return_metadata=False):
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
    gated_mode = str(cfg.get('output_granularity', 'sequence')) == 'frame'
    if gated_mode:
        unique, inverse = unique_source_frames(frames)
    else:
        unique = np.asarray(frames, dtype=np.int64)
        inverse = np.arange(len(unique), dtype=np.int64)
    use_segments = gated_mode and float(cfg.get('segment_prob', 0.0)) > 0.0
    if use_segments:
        seg_u, metadata = _segment_mask(b * k, len(unique), coords.device, cfg, corruptor.names)
        shifts_u, fired = tagged_draw(corruptor, b * k, len(unique), dim, coords.device, seg_u)
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


def draw_sparse_moved(sparse_corruptor, coords, is_dense, frames=None, cfg=None,
                        return_metadata=False, allow_stuck=False):
    """Draw sparse corruption, redrawing until every sparse point moves an observed slot.

    The optional distance-gated retry is handled by ``make_triplet`` after drop/view masks are
    available.  This helper retains the old moved-slot invariant and its two-return-value API by
    default; callers requesting metadata receive ``(shift, fired, segment_mask, metadata)``.

    Only the STUCK keypoints are redrawn; a keypoint that already moved an observed slot keeps
    its draw. Every point is drawn independently, so this per-keypoint rejection accepts the same
    distribution as redrawing all keypoints until all succeed at once -- but it does not collapse
    when many keypoints are sparse (a joint acceptance that is the product of per-keypoint ones).
    ``allow_stuck`` returns the last draw instead of raising when some keypoint is still stuck;
    the gated ``make_triplet`` path then drops that keypoint from the window. The scorer builds
    batch size 1, so the stuck mask reshapes to ``[b, k]`` with ``b == 1``.
    """
    def draw():
        """Draw one sparse candidate on the configured source-frame lattice."""
        return draw_shift(sparse_corruptor, coords, frames=frames, cfg=cfg,
                           return_metadata=True)
    shift, fired, seg, metadata = draw()
    k = shift.shape[2]
    for _ in range(SPARSE_DRAW_RETRIES):
        stuck = _sparse_stuck(shift, coords, is_dense)
        if not bool(stuck.any()):
            return ((shift, fired, seg, metadata) if return_metadata else (shift, fired))
        new_shift, new_fired, new_seg, new_metadata = draw()
        redo = stuck.reshape(-1, k)
        shift = torch.where(redo[:, None, :, None], new_shift, shift)
        fired = torch.where(redo[:, :, None], new_fired, fired)
        seg = torch.where(redo[:, None, :, None], new_seg, seg)
        flat = redo.reshape(-1)
        metadata = [new_metadata[i] if bool(flat[i]) else metadata[i]
                    for i in range(len(metadata))]
    stuck = _sparse_stuck(shift, coords, is_dense)
    if not bool(stuck.any()) or allow_stuck:
        return ((shift, fired, seg, metadata) if return_metadata else (shift, fired))
    raise ValueError(
        f'{int(stuck.sum())} sparse keypoint(s) still moved no observed slot after '
        f'{SPARSE_DRAW_RETRIES} redraws of a menu restricted to types that cannot cancel. That '
        'is not a sampling accident, so the observed-frame count or the magnitudes are wrong.')
