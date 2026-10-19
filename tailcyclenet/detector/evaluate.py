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

from ..evaluation.metrics import mota
from .decode import decode
from .boxes import NMS_CENTER_DIST_SELECT, box_iou
from .sampling import ChunkShuffle, box_collate


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
                  center_dist_thresh=NMS_CENTER_DIST_SELECT, subset_indices=None,
                  root_qualified_keys=False):
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
        one frame several views).
    """
    was_training = model.training
    model.eval()
    order = list(iter(ChunkShuffle(len(ds), chunk=ds.chunk, seed=seed)))
    if subset_indices is not None:
        allowed = set(int(i) for i in subset_indices)
        order = [i for i in order if i in allowed][:batches * batch_size]
    else:
        order = order[:batches * batch_size]

    aug_was = ds.augment
    ds.augment = False
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=order, num_workers=num_workers,
        collate_fn=box_collate)

    per_group = defaultdict(lambda: {'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0.0, 'fp': 0})
    tracks = defaultdict(dict)
    sessions, n_want = {}, {}
    for bi, batch in enumerate(loader):
        x, gt = batch['x'], batch['boxes']
        obj, pred_boxes = model(x.to(device))
        for j in range(x.shape[0]):
            item = order[bi * batch_size + j]
            sess, gid, f, ci = ds.index[item]
            key = f'{sess.session_id}/{gid}'
            if root_qualified_keys:
                root_id = ds.root_ids[item]
                key = f'{ds.root_names[root_id]}/{key}'
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


@torch.no_grad()
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


