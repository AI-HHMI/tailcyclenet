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
from posetail.posetail.cube import get_camera_scale, is_point_visible

TRIPLET_KEYS = ('good', 'bad', 'anchor')

# The corruption types that can move a SINGLE observed slot, which is all a sparse window can be
# trained on. `gradual_drift` and `sinusoid` ramp from or through zero, so at one observed frame
# they may corrupt nothing at all -- and a corruption that changes nothing would teach the scorer
# to call a corrupt sample clean, which is worse than excluding the window. `const_offset` is
# FORCED for sparse keypoints (see `build_corruptors_for`); `frame_noise` rides alongside it.
SPARSE_TYPES = ('const_offset', 'frame_noise')


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
    scale = float(view.scale[0])
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
    S[0, 0] = S[1, 1] = scale
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


def make_triplet(dataset, sel, rng, cfg, corruptors, cam_thresh=1):
    """Build a (good, bad, anchor) triplet from ONE `Selection` and TWO realisations.

    The selection is realised TWICE with independent streams: `view_a` gives the pixels and
    cameras that good and bad SHARE, `view_b` gives the anchor's own. The anchor's COORDINATES are
    the chosen source's, carried into view B -- for 3D that is the identical world track (the
    image-plane rotation is camera-only, and the world-gauge rotation is off, so one tensor serves
    all three members); for 2D it is `transfer_points_2d`, because there the coordinates ARE the
    crop's pixels.

    The crop for every member comes from the CLEAN, PRE-DROP track, fixed BEFORE any corruption is
    drawn. This one is a real leak if violated: a crop that followed each candidate's own
    coordinates would give the bad sample different pixels from the good one, and the pair would
    stop being pixel-matched.

    Views come back with the batch axis added, because `_realise` returns what the pose collate
    stacks per camera and the scorer bypasses that collate.

    Inputs: dataset -- the `PoseDataset` owning `sel`; sel -- a `Selection`; rng -- the triplet's
            stream (the two views derive their own); cfg -- the `[scorer.corruption]` block;
            corruptors -- `build_corruptors_for`'s quadruple; cam_thresh -- visibility threshold.
    Outputs: the triplet dict, or None when a view fails to build or fewer than 2 points survive.
    Side effects: decodes video frames twice; draws from `rng` and the ambient torch/imgaug RNGs.
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

    counts = observed_frame_counts(coords, sel.frames)

    # Section 3.9: a keypoint with NO observed frame is DROPPED here, before anything is drawn.
    # It has by construction no slot any corruption could move, so leaving it in would (a) train
    # the scorer on a point whose only content is the missing token, and (b) make the sparse
    # "did the corruption move an observed slot" check unsatisfiable for it.
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
    shift_dense, fired_dense = _draw_shift(dense, coords)
    shift_sparse, fired_sparse = _draw_sparse_moved(sparse, coords, is_dense)
    pick = is_dense.view(1, 1, K, 1)
    pick_type = is_dense.view(1, K, 1)
    shift = torch.where(pick, shift_dense, shift_sparse)
    fired = torch.where(pick_type, fired_dense, fired_sparse)
    if mode == '3d':
        shift = shift * cube_scale_b[:, None, None, None]

    drop_mask = _compute_drop_mask(coords, cfg, counts)
    good = apply_drop_mask(coords, drop_mask)
    bad = apply_drop_mask(coords + shift, drop_mask)

    anchor_label = 1.0 if float(rng.random()) < 0.5 else -1.0
    source = good if anchor_label > 0 else bad
    if mode == '3d':
        anchor = source
    else:
        anchor = transfer_points_2d(source[0], view_affine_2d(view_a),
                                    view_affine_2d(view_b))[None]

    keep = (_visible_points_mask(good[0], view_a.cgroup, mode, cam_thresh)
            & _visible_points_mask(bad[0], view_a.cgroup, mode, cam_thresh)
            & _visible_points_mask(anchor[0], view_b.cgroup, mode, cam_thresh))
    n_keep = int(keep.sum())
    if n_keep < 2:
        return None

    gv = [v[None] for v in view_a.views]
    av = [v[None] for v in view_b.views]
    return {
        'good': (gv, good[:, :, keep], view_a.cgroup),
        'bad': (gv, bad[:, :, keep], view_a.cgroup),
        'anchor': (av, anchor[:, :, keep], view_b.cgroup),
        'kpt_ids': dataset._kpt_ids[sel.sess.path][alive][keep][None],
        'anchor_label': anchor_label,
        'mode': mode,
        'reuse_scene_for_anchor': False,
        'occlusion': None,
        'counts': counts[:, keep],
        'fired': fired[:, keep, :],
        'n_dense': int(is_dense[keep].sum()),
        'n_sparse': int((~is_dense[keep]).sum()),
    }


def _draw_shift(corruptor, coords):
    """One corruption draw for every (batch, keypoint) point, in UNIT magnitude space.

    Inputs: corruptor -- a `PointCorruptor`; coords -- [b,t,k,R].
    Outputs: ([b,t,k,R] shift, [b,k,n_types] fired); 3D shifts are in cube-scale units (the
        caller scales them).
    Side effects: draws from the ambient torch RNG.
    """
    b, t, k, dim = coords.shape
    shifts, fired = tagged_draw(corruptor, b * k, t, dim, coords.device)
    return (shifts.reshape(b, k, t, dim).permute(0, 2, 1, 3),
            fired.reshape(b, k, len(GENERATORS)))


SPARSE_DRAW_RETRIES = 16


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


def _draw_sparse_moved(sparse_corruptor, coords, is_dense):
    """Draw the sparse corruption, RESAMPLING until every sparse keypoint is actually moved.

    Section 3.8b invariant 1: the sampled corruption must change an observed slot, so a draw that
    misses one is redrawn rather than accepted. The whole draw is redrawn rather than patched,
    because patching would bias the mix toward the type used to patch it.

    Inputs: sparse_corruptor -- the reduced-menu `PointCorruptor`; coords -- [b,t,k,R];
            is_dense -- [b,k] bool.
    Outputs: (shift [b,t,k,R], fired [b,k,n_types]).
    Side effects: draws from the ambient torch RNG.
    """
    shift, fired = _draw_shift(sparse_corruptor, coords)
    for _ in range(SPARSE_DRAW_RETRIES):
        stuck = _sparse_stuck(shift, coords, is_dense)
        if not bool(stuck.any()):
            return shift, fired
        shift, fired = _draw_shift(sparse_corruptor, coords)
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
