"""Synthetic track corruption and triplet assembly for THIS repo's loader.

     good   = clean track,      view A             label +1, every point
     bad    = corrupt(good),    view A (SAME)      label -1, every point
     anchor = good or bad 50/50, view B (indep.)   label = its source's

`good` and `bad` SHARE pixels and camera geometry, so the only difference between them is
coordinates -- which is what makes the good-vs-bad gap a track-quality signal rather than an
appearance signal. The anchor is the invariance target.

WHAT TRANSFERS FROM THE REFERENCE (`tracktail/datasets/scorer_corruption.py`) AND WHAT DOES NOT.
`GENERATORS`, `PointCorruptor`, `apply_drop_mask` and `build_corruptors` are imported verbatim.
`compute_drop_mask` is FORKED (see `corruption.compute_drop_mask`). `make_triplet` and every
rotation/crop/resize helper are rewritten: the reference builds them against a loader that returns
FULL, UN-AUGMENTED frames and owns the whole geometry chain itself, whereas this repo's
`PoseDataset` returns already-cropped, already-resized, already-augmented uint8 with the rotation
baked into the decode. Nothing about that ordering is recoverable by an adapter shim.

The seam (`PoseDataset.select` / `realise`) is what makes the rewrite possible: ONE selection,
TWO independent realisations. Calling `__getitem__` twice would select a DIFFERENT sample -- `_pick`
re-draws the pool entry and `_frames` re-draws the anchor and stride.
"""
import numpy as np
import torch
from einops import rearrange

from tracktail.datasets.scorer_corruption import apply_drop_mask
from tracktail.model.cube import get_camera_scale

from .corruption import (SPARSE_DRAW_RETRIES, compute_drop_mask, draw_shift, draw_sparse_moved,
                         observed_frame_counts, source_frame_weights, unique_source_frames)
from .masks import displacement_masks, visible_points_mask
from .views import make_anchor


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


def make_triplet(dataset, sel, rng, cfg, corruptors, cam_thresh=1):
    """Build a (good, bad, anchor) triplet and its framewise supervision metadata.

    The historical geometry is intentionally unchanged: one clean view A is shared by ``good``
    and ``bad``, while ``anchor`` is an independently realised view B.  New metadata is computed
    after the shared drop mask and final crop.  ``frames`` is carried explicitly, and corruption
    draws use the unique source-frame sequence before being gathered back to local slots.

    A tracked root has no independent reference; ``reference_gate='source_far'`` therefore records
    the stored clean track as a proxy rather than silently implying ground truth.

    Outputs retain the old keys (including ``fired`` [B,K,G], good/bad/anchor tuples and counts)
    and add aligned ``[B,T,K]`` masks: ``observed_mask``, ``anchor_observed_mask``,
    ``in_view_mask``, ``moved_mask``, ``displacement_px``, ``reference_far_mask``, ``far_mask``,
    ``near_mask``, ``ambiguous_mask``, ``active_mask`` and ``source_frame_weight``.  The per-type
    ``corruption_type_mask`` is ``[B,T,K,G]`` and means type-fired *and* final-far.  ``segments``
    and ``unique_frames`` are retained as Python metadata for provenance/QC.

    The function returns ``None`` for ordinary view/keypoint failures, or for a distance-gated draw
    with no evaluable active row.  Sparse keypoints redraw their corruption up to the existing
    retry cap when the distance gate leaves them with no active far slot. Pin-memory workers
    require the returned `fired_frame` frame axis to be materialized, not an expanded view.

    Frame mode redraws only unresolved sparse keypoints, then
    drop any still unresolved after the cap via ``keep`` instead of rejecting the whole window
    (which would discard most mocap windows with marker dropouts); legacy sequence mode raises.
    """
    dense_3d, dense_2d, sparse_3d, sparse_2d = corruptors
    view_a = dataset.realise(sel, rng, world_gauge=False)
    if view_a is None:
        return None
    view_b = dataset.realise(sel, rng, world_gauge=False)
    if view_b is None:
        return None

    mode = '2d' if view_a.r == 2 else '3d'
    coords = view_a.coords[None]
    if coords.shape[0] != 1:
        raise ValueError('the scorer builds one window per triplet (batch_size 1)')
    frames_np, inverse = unique_source_frames(sel.frames)
    counts = observed_frame_counts(coords, sel.frames)

    alive = counts[0] > 0
    if int(alive.sum()) < 2:
        return None
    alive_slots = torch.nonzero(alive, as_tuple=True)[0]
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

    shift_dense, fired_dense, seg_dense, meta_dense = draw_shift(
        dense, coords, frames=sel.frames, cfg=cfg, return_metadata=True)
    frame_mode = str(cfg.get('output_granularity', 'sequence')) == 'frame'
    shift_sparse, fired_sparse, seg_sparse, meta_sparse = draw_sparse_moved(
        sparse, coords, is_dense, frames=sel.frames, cfg=cfg, return_metadata=True,
        allow_stuck=frame_mode)
    pick = is_dense.view(1, 1, K, 1)
    pick_type = is_dense.view(1, K, 1)
    shift = torch.where(pick, shift_dense, shift_sparse)
    fired = torch.where(pick_type, fired_dense, fired_sparse)
    seg_gate = torch.where(pick_type[:, None, :, :], seg_dense, seg_sparse)
    metadata_flat = [meta_dense[i] if bool(is_dense[i]) else meta_sparse[i] for i in range(K)]
    if mode == '3d':
        shift = shift * cube_scale_b[:, None, None, None]

    drop_mask = compute_drop_mask(coords, cfg, counts)
    good = apply_drop_mask(coords, drop_mask)

    anchor_label = 1.0 if float(rng.random()) < 0.5 else -1.0
    mask_data = None
    keep = None
    unresolved_sparse = torch.zeros_like(is_dense)
    for sparse_attempt in range(SPARSE_DRAW_RETRIES + 1):
        bad = apply_drop_mask(coords + shift, drop_mask)
        source = good if anchor_label > 0 else bad
        anchor = make_anchor(
            source, anchor_label, mode, view_a, view_b,
            source_slots=alive_slots if mode == '2d' else None)
        mask_data = displacement_masks(
            good, bad, anchor, view_a.cgroup, mode, cfg,
            anchor_camera_group=view_b.cgroup,
            moved_mask=shift.abs().sum(-1) > 0,
            cam_thresh=cam_thresh)
        keep = (visible_points_mask(good[0], view_a.cgroup, mode, cam_thresh)
                & visible_points_mask(bad[0], view_a.cgroup, mode, cam_thresh)
                & visible_points_mask(anchor[0], view_b.cgroup, mode, cam_thresh))
        sparse_active = mask_data['active_mask'][0].any(dim=0) & (~is_dense)
        redo = (~is_dense) & (~sparse_active)
        if not frame_mode or not bool(redo.any()):
            break
        if sparse_attempt >= SPARSE_DRAW_RETRIES:
            unresolved_sparse = redo
            break
        shift_sparse, fired_sparse, seg_sparse, meta_sparse = draw_sparse_moved(
            sparse, coords, is_dense, frames=sel.frames, cfg=cfg, return_metadata=True,
            allow_stuck=True)
        if mode == '3d':
            shift_sparse = shift_sparse * cube_scale_b[:, None, None, None]
        shift = torch.where(redo.view(1, 1, K, 1), shift_sparse, shift)
        fired = torch.where(redo.view(1, K, 1), fired_sparse, fired)
        seg_gate = torch.where(redo.view(1, 1, K, 1), seg_sparse, seg_gate)
        metadata_flat = [meta_sparse[i] if bool(redo[i]) else metadata_flat[i]
                         for i in range(K)]
    if bool(unresolved_sparse.any()):
        keep = keep & (~unresolved_sparse)

    n_keep = int(keep.sum())
    if n_keep < 2:
        return None

    active_before_keep = mask_data['active_mask']
    if frame_mode and not bool(active_before_keep.any()):
        return None

    gv = [v[None] for v in view_a.views]
    av = [v[None] for v in view_b.views]
    masks = {key: value[:, :, keep] if torch.is_tensor(value) and value.ndim >= 3 else value
             for key, value in mask_data.items()}
    if frame_mode and not bool(masks['active_mask'].any()):
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
        'fired_frame': fired_keep[:, None, :, :].expand(-1, coords.shape[1], -1, -1).contiguous(),
        'source_frame_weight': weights,
        'max_clean_px': float(cfg.get('max_clean_px', 0.0)),
        'n_dense': int(is_dense[keep].sum()),
        'n_sparse': int((~is_dense[keep]).sum()),
        'n_sparse_dropped': int(unresolved_sparse.sum()),
    }
    for key, value in masks.items():
        if value is not None or key not in ('good_reference_distance_px',
                                            'bad_reference_distance_px'):
            out[key] = value
    out['shift'] = shift[:, :, keep]
    return out


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
    `TracktailDataset.rotate_camera_group` draws the 3D world-gauge angle from global numpy.

    Inputs: worker_id -- the DataLoader's worker index.
    Outputs: none.
    Side effects: reseeds numpy's global RNG in this worker process.
    """
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)
