"""Scoring a box predictor. Shared by `scripts/eval_detector.py` and the training loop.

Here rather than in either script because the two must not disagree: the number training selects
a checkpoint on and the number a run is reported with have to be the same number.

Four choices, each of which the old in-loop `evaluate` got the generous way round:

- **Greedy one-to-one matching.** `box_iou(gt, pred).max(1)` is not a matching -- in a huddle one
  well-placed box lands within IoU 0.5 of several animals and is counted for every one of them.
- **`top_k` is the session's animal count**, what deployment supplies, never this frame's true
  count. The latter is an oracle.
- **Mean IoU over EVERY labelled box**, zero where unmatched. Mean-over-matched rewards a
  detector for declining the hard animals.
- **False positives are counted.** Recall alone cannot see them.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..metrics import mota
from .assign import box_iou, decode
from .data import ChunkShuffle, box_collate


def greedy_match(gt, pred):
    """Greedy one-to-one IoU matching. -> (iou per gt (G,), n_unmatched_pred).

    Highest IoU first, each prediction spent once.
    """
    ious = np.zeros(gt.shape[0])
    if pred.shape[0] == 0:
        return ious, 0
    m = box_iou(gt, pred).numpy()
    order = np.stack(np.unravel_index(np.argsort(-m, axis=None), m.shape), 1)
    used_g, used_p = set(), set()
    for g, p in order:
        if m[g, p] <= 0:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(int(g))
        used_p.add(int(p))
        ious[g] = m[g, p]
    return ious, pred.shape[0] - len(used_p)


def _corners(boxes):
    """(N,4) xyxy -> (N,2,2): a box as its two diagonal corners.

    `metrics.mota` matches point sets, so a box enters as a two-"keypoint" instance and the match
    distance is the mean corner displacement -- translation and size, rather than IoU.
    """
    return np.asarray(boxes, float).reshape(-1, 2, 2)


def box_mota(store):
    """Box-only MOTA over one camera view's sampled VIEWS.

    `idsw` here is NOT a tracking number: frames are subsampled per group, so consecutive rows
    are not consecutive in time. It is comparable between arms and nothing more.

    `store` maps a per-VIEW key to `(pred (P,4), gt (S,4), ig (S,) | None, ig_boxes (S,4) | None)`,
    everything already in the same input pixels. The key is the loader's ITEM INDEX and not the
    frame number, because under tiling one frame yields SEVERAL views and keying by frame silently
    kept only the last tile of each. The ignore rows arrive pre-transformed from
    `BoxDataset.ignore_for` for the same reason -- their transform is per item.
    """
    keys = sorted(store)
    T = len(keys)
    P = max(1, max(v[0].shape[0] for v in store.values()))
    S = max(1, max(v[1].shape[0] for v in store.values()))
    pred = np.full((P, T, 2, 2), np.nan)
    true = np.full((S, T, 2, 2), np.nan)
    have_ig = any(v[2] is not None for v in store.values())
    have_igb = any(v[3] is not None for v in store.values())
    ig = np.zeros((S, T), bool) if have_ig else None
    ig_boxes = np.full((S, T, 4), np.nan) if have_igb else None
    for t, k in enumerate(keys):
        p, g, gi, gb = store[k]
        pred[:p.shape[0], t] = _corners(p)
        true[:g.shape[0], t] = _corners(g)
        if ig is not None and gi is not None:
            ig[:min(S, len(gi)), t] = gi[:S]
        if ig_boxes is not None and gb is not None:
            ig_boxes[:min(S, len(gb)), t] = gb[:S]

    with np.errstate(all='ignore'):
        diag = np.nanmedian(np.linalg.norm(true[:, :, 1] - true[:, :, 0], axis=-1))
    radius = float(diag) * 0.5 if np.isfinite(diag) else np.inf
    return mota(pred, true, radius, ignore=ig, ignore_boxes=ig_boxes)


def _summarise(s):
    """Reduce one group's raw score dict to its reported metrics (per-GT rates).

    Inputs: s -- per-group dict built by `score_dataset`.
    Outputs: dict with n_gt and per-GT rates (r50, r75, iou, fp, mota, fp_dup, fp_none, miss).

    Notes.

    `fp` is `greedy_match`'s IoU-based count -- ONE scalar that already conflates a real
    duplicate detection with a real over-detection on empty ground. `box_mota`'s point-distance
    pass (already run, for `mota`) decomposes the SAME decoded boxes into THREE separate numbers
    instead of one: `fp_dup` (near an already-claimed GT -- arbitration/NMS removes it), `fp_none`
    (on no animal -- a detection problem) and `misses` (a real animal with no box claiming it at
    all -- the OTHER direction, and easy to lose sight of while reading fp). They want DIFFERENT
    fixes and none of the three should be read off the other two.

    NONE OF THIS ESCAPES THE BUDGET CAP: `box_mota` scores whatever `decode` already returned at
    `top_k = n_want` (the ground-truth animal count by default, same as `fp`), so raising
    `--max-animals` moves `fp`/`fp_dup`/`fp_none`/`miss` together, not `fp_dup`/`fp_none` alone.
    The decomposition is what changes here, not the budget.
    """
    n = max(s['n_gt'], 1)
    m = s.get('mota', [])
    gt = sum(r['gt'] for r in m)
    return {'n_gt': s['n_gt'], 'r50': s['hit50'] / n, 'r75': s['hit75'] / n,
            'iou': s['iou'] / n, 'fp': s['fp'] / n,
            'mota': (sum(r['mota'] * r['gt'] for r in m if np.isfinite(r['mota'])) / gt
                     if gt else float('nan')),
            'fp_ignored': sum(r['fp_ignored'] for r in m),
            'fp_dup': (sum(r['fp_dup'] for r in m) / gt) if gt else float('nan'),
            'fp_none': (sum(r['fp_none'] for r in m) / gt) if gt else float('nan'),
            'miss': (sum(r['misses'] for r in m) / gt) if gt else float('nan')}


@torch.no_grad()
def score_dataset(model, ds, device, batch_size=16, batches=40, seed=0, score_thresh=0.05,
                  num_workers=4, max_animals=None, out_scores=None, iou_thresh=0.5,
                  center_dist_thresh=0.5):
    """{group_key: metrics} for `model` over a sample of `ds`. Leaves the model in eval mode.

    Inputs:
        model, ds, device -- the detector, dataset, and compute device.
        batch_size, batches, num_workers -- loader shape; `seed` drives the `ChunkShuffle`
            sample (never read off the front of an unshuffled loader -- the index is built
            session by session, so the first 800 views ARE the first session).
        score_thresh, iou_thresh, center_dist_thresh -- the `decode` thresholds the sample is
            scored at.
        max_animals -- per-group detection cap (default: the group's labelled animal count).
        out_scores -- when a list is given, collects every decoded objectness this pass saw:
            the distribution `--det-score` cuts, recorded at training time because it is a
            property of the CHECKPOINT (an old hard-1.0-target checkpoint saturates near 1.0;
            the current `iou_aware_obj` recipe does not -- why the shipped default moved to
            0.05).
    Outputs:
        {group_key: {n_gt, r50, r75, iou, fp, mota, fp_ignored, fp_dup, fp_none, miss}}.
    Side effects:
        Sets the model to eval and `ds.augment` to False, restoring both at the end. Scoring
        is deliberately unaugmented: `ignore_for` takes no warp, so augmented scoring would
        compare warped predictions against unwarped `instances.pq` boxes.
    Notes:
        Per-(group, camera) state is keyed by the ITEM index, not the frame `f` (tiling makes
        one frame several views). The head output is INDEXED, not unpacked -- fixed-arity
        unpacking crashed eval when the head grew an extra return.
    """
    was_training = model.training
    model.eval()
    order = list(iter(ChunkShuffle(len(ds), chunk=ds.chunk, seed=seed)))[:batches * batch_size]

    aug_was = ds.augment
    ds.augment = False
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=order, num_workers=num_workers,
        collate_fn=box_collate)

    per_group = defaultdict(lambda: {'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0.0, 'fp': 0})
    tracks = defaultdict(dict)
    sessions, n_want = {}, {}
    for bi, batch in enumerate(loader):
        x, gt = batch[0], batch[1]
        _out = model(x.to(device))
        obj, pred_boxes = _out[0], _out[1]
        for j in range(x.shape[0]):
            item = order[bi * batch_size + j]
            sess, gid, f, ci = ds.index[item]
            key = f'{sess.session_id}/{gid}'
            if key not in n_want:
                sessions[key] = sess
                n_want[key] = max_animals or max(1, len(sess.labels(gid).animal_ids))
            p, sc_j = decode(obj[j], pred_boxes[j], top_k=n_want[key], score_thresh=score_thresh,
                             iou_thresh=iou_thresh, center_dist_thresh=center_dist_thresh)
            if out_scores is not None and sc_j.numel():
                out_scores.append(sc_j.detach().cpu().numpy())
            g_all = gt[j]
            g = g_all[torch.isfinite(g_all).all(-1)]
            p = p.cpu()
            tracks[(key, ci)][item] = (p.numpy(), g_all.numpy(), *ds.ignore_for(item))
            if not g.numel():
                per_group[key]['fp'] += p.shape[0]
                continue
            ious, n_fp = greedy_match(g, p)
            s = per_group[key]
            s['n_gt'] += len(ious)
            s['hit50'] += int((ious >= 0.5).sum())
            s['hit75'] += int((ious >= 0.75).sum())
            s['iou'] += float(ious.sum())
            s['fp'] += n_fp

    for (key, ci), store in tracks.items():
        per_group[key].setdefault('mota', []).append(box_mota(store))
    ds.augment = aug_was
    if was_training:
        model.train()
    return {g: _summarise(s) for g, s in per_group.items()}


def overall(rows):
    """Weight groups by their labelled-box count. The per-group table is the unweighted view.

    Notes.

    `fp_ignored` is a raw count, like every other place it is reported -- see `_summarise`.
    `fp_dup`/`fp_none`/`miss` are already per-GT rates (see `_summarise`), so they are weighted
    by `n_gt` the same way `mota` is, and the same way a group with none of `box_mota`'s inputs
    (no `mota` rows at all) drops out rather than contaminating the mean with a NaN.
    """
    n = sum(r['n_gt'] for r in rows.values()) or 1
    out = {k: sum(r[k] * r['n_gt'] for r in rows.values()) / n
           for k in ('r50', 'r75', 'iou', 'fp')}
    m = [r for r in rows.values() if np.isfinite(r['mota'])]
    out['mota'] = (sum(r['mota'] * r['n_gt'] for r in m) / sum(r['n_gt'] for r in m)
                   if m else float('nan'))
    out['n_gt'] = sum(r['n_gt'] for r in rows.values())
    out['fp_ignored'] = sum(r.get('fp_ignored', 0) for r in rows.values())
    for k in ('fp_dup', 'fp_none', 'miss'):
        mk = [r for r in rows.values() if np.isfinite(r.get(k, float('nan')))]
        out[k] = (sum(r[k] * r['n_gt'] for r in mk) / sum(r['n_gt'] for r in mk)
                  if mk else float('nan'))
    return out


def _labelled_frames(sess, gid):
    """(F,) int frame indices with any labelled keypoint in this group, or empty."""
    from ..format import UNLABELED
    lab = sess.labels(gid)
    vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
    if vis is None:
        return np.zeros(0, int)
    v = vis.reshape(vis.shape[0], vis.shape[1], -1)
    return np.flatnonzero((v != UNLABELED).any((0, 2)))


def _gt_crop_sides(sess, gid, min_crop_dim, max_frames=0, cap=200):
    """Crop-rule box side (SOURCE px) for every labelled (animal, frame[, camera]), a POPULATION
    -- not paired to any detector row. `deployment_score` compares this against the union-box
    side distribution the box path actually produces, unpaired; pairing a detector row to a GT
    identity needs a matcher (MOTA's), and this is meant to run before spending a GPU-hour on
    either half of the box path, not to replace `box_mota`.

    2D: `points2d[:, f, :, ci]` directly. 3D: `points3d[:, f]` projected through frame `f`'s own
    camera group, the same branch `BoxDataset._points_2d` takes (mirrored here rather than
    imported, because building a whole `BoxDataset` -- augmentation RNG, tiling, the multi-root
    refusal -- to read one projection is the wrong tool for a label-only pass).

    `cap` bounds the frames sampled per group (evenly spaced), so a 20,000-frame clip costs the
    same as a 500-frame one -- this is a population read, not a census.
    """
    from ..crop import crop_box_for_points
    from posetail.posetail.cube import project_points_torch
    frames = _labelled_frames(sess, gid)
    if max_frames:
        frames = frames[frames < max_frames]
    if frames.size > cap:
        frames = frames[np.linspace(0, frames.size - 1, cap).astype(int)]
    lab = sess.labels(gid)
    sides = []
    for f in frames.tolist():
        if sess.mode == '3d':
            cams = sess.cgroup(gid, int(f))
            pts3d = torch.as_tensor(lab.points3d[:, f], dtype=torch.float32)
            for ci, cam in enumerate(cams):
                p2d = project_points_torch([cam], pts3d)[0]
                w, h = sess.rig.size(sess.cam_names[ci])
                for s in range(p2d.shape[0]):
                    box = crop_box_for_points(p2d[s], torch.tensor([w, h]), min_crop_dim)
                    if box is not None:
                        sides.append(float(0.5 * ((box[2] - box[0]) + (box[3] - box[1]))))
        else:
            for ci, cam_name in enumerate(sess.cam_names):
                w, h = sess.rig.size(cam_name)
                p2d = torch.as_tensor(lab.points2d[:, f, :, ci], dtype=torch.float32)
                for s in range(p2d.shape[0]):
                    box = crop_box_for_points(p2d[s], torch.tensor([w, h]), min_crop_dim)
                    if box is not None:
                        sides.append(float(0.5 * ((box[2] - box[0]) + (box[3] - box[1]))))
    return np.array(sides)


def deployment_score(model, sess, gid, input_wh, device='cpu', top_k=24, max_animals=None,
                     det_score=0.01, track=True, link=False, min_views=2, max_move=1.0,
                     min_crop_dim=64, reduce=False, tile_scale=None, max_frames=0,
                     n_frames=24, overlap=4, min_box_frames=1, batch=16, iou_thresh=0.5,
                     center_dist_thresh=0.5):
    """Deployment-shaped detector quality over ONE WHOLE CLIP, no frame sampling.

    `score_dataset` answers "what fraction of SAMPLED frames does the detector recall a box on",
    which does not predict the box path's real cost: current-generation checkpoints read val r@.5
    0.91-0.93 and IoU 0.79 while costing 43% of pose coverage and +25 px MPJPE. This runs the SAME
    functions deployment does -- `detect_raw` then `associate_group`, and `infer._window_starts`
    for the window rule `run_group` uses -- over a whole test group, and reports what a pose
    window actually gets:

        det_fill       fraction of (frame, camera) with >=1 box surviving `det_score` --
                       the DETECTION-ONLY ceiling, independent of row identity.
        slot_fill      fraction of (row, frame, camera) with a finite box AFTER association --
                       identity-bearing only under `track=True` (3D multiview) or `link=True`
                       (2D). Comparing it against `det_fill` separates "the detector missed the
                       animal" from "association dropped a detection it had".
        window_miss    fraction of (row, window) with fewer than `min_box_frames` finite boxes
                       anywhere in the window -- the EXACT quantity `infer.run_group` marks
                       `no box`, using its own `_window_starts` rule.
        union_side_px  quantiles of each window's per-camera UNION box side (SOURCE px), the
                       crop-inflation number `--refine` exists to survive.
        gt_side_px     quantiles of the crop-RULE box side over this group's own labels, as an
                       UNPAIRED population comparison against `union_side_px`.

    Returns a dict of floats/arrays; `n_gt` and `n_windows` say how much each rested on.
    """
    from . import detect_raw
    from . import associate_group as _associate_group
    from ..infer import _window_starts

    n_animals = max_animals or max(1, len(sess.labels(gid).animal_ids))
    raw = detect_raw(model, input_wh, sess, gid, top_k=max(top_k, n_animals), device=device,
                     batch=batch, score_thresh=det_score, reduce=reduce, max_frames=max_frames,
                     tile_scale=tile_scale, iou_thresh=iou_thresh,
                     center_dist_thresh=center_dist_thresh)
    r_box, r_sc, r_kp = raw
    D, T, C = r_sc.shape
    det_fill = float(np.isfinite(r_sc[0]).mean()) if D else 0.0

    out, sc, kp = _associate_group(raw, sess, gid, n_animals, link=link, min_views=min_views,
                                   track=track, max_move=max_move)
    S, T, C = sc.shape
    slot_fill = float(np.isfinite(sc).mean())

    starts = _window_starts(T, n_frames, overlap)
    miss, union_sides = [], []
    for a in range(S):
        for st in starts:
            frames = np.arange(st, min(st + n_frames, T))
            bb = out[a][frames]
            n_ok = int(np.isfinite(bb).all(-1).sum())
            miss.append(n_ok < min_box_frames)
            if n_ok:
                for ci in range(C):
                    v = bb[:, ci][np.isfinite(bb[:, ci]).all(-1)]
                    if len(v):
                        x0, y0 = v[:, 0].min(), v[:, 1].min()
                        x1, y1 = v[:, 2].max(), v[:, 3].max()
                        union_sides.append(0.5 * ((x1 - x0) + (y1 - y0)))

    gt_sides = _gt_crop_sides(sess, gid, min_crop_dim, max_frames=max_frames)
    q = (0.5, 0.9, 0.99)

    def _quant(a):
        """{p: quantile} at q=(0.5, 0.9, 0.99); NaN-filled when `a` is empty."""
        a = np.asarray(a, float)
        return {p: float(np.quantile(a, p)) for p in q} if a.size else {p: float('nan') for p in q}

    return {'det_fill': det_fill, 'slot_fill': slot_fill,
           'window_miss': float(np.mean(miss)) if miss else float('nan'),
           'n_windows': len(miss), 'n_gt': int(gt_sides.size),
           'union_side_px': _quant(union_sides), 'gt_side_px': _quant(gt_sides)}
