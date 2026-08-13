"""Scoring a box predictor. Shared by `scripts/eval_detector.py` and the training loop.

Here rather than in either script because the two must not disagree: the number training selects
a checkpoint on and the number a run is reported with have to be the same number.

Four choices, each of which the old in-loop `evaluate` got the generous way round:

- **Greedy one-to-one matching.** `box_iou(gt, pred).max(1)` is not a matching -- in a huddle one
  well-placed box lands within IoU 0.5 of several animals and is counted for every one of them.
- **`top_k` is the session's animal count**, what `scripts/infer.py:132` supplies at deployment,
  never this frame's true count. The latter is an oracle.
- **Mean IoU over EVERY labelled box**, zero where unmatched. Mean-over-matched rewards a
  detector for declining the hard animals -- eval rule 6 in box form.
- **False positives are counted.** Recall alone cannot see them.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..format import INST_PRESENT
from ..metrics import mota
from .assign import box_iou, decode
from .data import ChunkShuffle, box_collate, letterbox_transform


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


def box_mota(sess, gid, ci, store, input_wh):
    """Box-only MOTA over this group's sampled frames in one camera view.

    `idsw` here is NOT a tracking number: frames are subsampled per group, so consecutive rows
    are not consecutive in time. It is comparable between arms and nothing more.
    """
    lab = sess.labels(gid)
    frames = sorted(store)
    T = len(frames)
    P = max(1, max(v[0].shape[0] for v in store.values()))
    S = max(1, max(v[1].shape[0] for v in store.values()))
    pred = np.full((P, T, 2, 2), np.nan)
    true = np.full((S, T, 2, 2), np.nan)
    for t, f in enumerate(frames):
        p, g = store[f]
        pred[:p.shape[0], t] = _corners(p)
        true[:g.shape[0], t] = _corners(g)

    # Present-but-unannotated animals: rat-city ships 26,021 of these since `43ff495`, and
    # counting them as false positives is measuring the annotator, not the detector.
    ig = ig_boxes = None
    if lab.instance is not None:
        scale, pad = letterbox_transform(sess.rig.size(sess.cam_names[ci]), input_wh)
        n = min(S, lab.instance.shape[0])
        ig = np.zeros((S, T), bool)
        ig[:n] = lab.instance[:n][:, frames, ci] == INST_PRESENT
        if lab.boxes is not None:
            ig_boxes = np.full((S, T, 4), np.nan)
            b = lab.boxes[:n][:, frames, ci].astype(float)
            b[..., 0::2] = b[..., 0::2] * scale + pad[0]
            b[..., 1::2] = b[..., 1::2] * scale + pad[1]
            ig_boxes[:n] = b

    with np.errstate(all='ignore'):
        diag = np.nanmedian(np.linalg.norm(true[:, :, 1] - true[:, :, 0], axis=-1))
    radius = float(diag) * 0.5 if np.isfinite(diag) else np.inf
    return mota(pred, true, radius, ignore=ig, ignore_boxes=ig_boxes)


def _summarise(s):
    n = max(s['n_gt'], 1)
    m = s.get('mota', [])
    gt = sum(r['gt'] for r in m)
    return {'n_gt': s['n_gt'], 'r50': s['hit50'] / n, 'r75': s['hit75'] / n,
            'iou': s['iou'] / n, 'fp': s['fp'] / n,
            'mota': (sum(r['mota'] * r['gt'] for r in m if np.isfinite(r['mota'])) / gt
                     if gt else float('nan')),
            'fp_ignored': sum(r['fp_ignored'] for r in m)}


@torch.no_grad()
def score_dataset(model, ds, device, batch_size=16, batches=40, seed=0, score_thresh=0.05,
                  num_workers=4, max_animals=None):
    """{group_key: metrics} for `model` over a sample of `ds`. Leaves the model in eval mode.

    Sampled with `ChunkShuffle` rather than read off the front of an unshuffled loader: the index
    is built session by session, so the first 800 views ARE the first session.
    """
    was_training = model.training
    model.eval()
    order = list(iter(ChunkShuffle(len(ds), chunk=ds.chunk, seed=seed)))[:batches * batch_size]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=order, num_workers=num_workers,
        collate_fn=box_collate)

    per_group = defaultdict(lambda: {'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0.0, 'fp': 0})
    tracks = defaultdict(dict)                  # (key, ci) -> frame -> (pred (P,4), gt (S,4))
    sessions, n_want = {}, {}
    # `batch[2:]` is the keypoint target when the loader is emitting one. Scoring here is
    # box-only by design -- `n_keypoints` must not change what r@.5 means -- so it is dropped
    # rather than unpacked, and a keypoint-trained detector stays comparable to every box-only
    # number in reports 10-13.
    for bi, batch in enumerate(loader):
        x, gt = batch[0], batch[1]
        obj, pred_boxes, _ = model(x.to(device))
        for j in range(x.shape[0]):
            sess, gid, f, ci = ds.index[order[bi * batch_size + j]]
            key = f'{sess.session_id}/{gid}'
            if key not in n_want:
                sessions[key] = sess
                n_want[key] = max_animals or max(1, len(sess.labels(gid).animal_ids))
            p, _ = decode(obj[j], pred_boxes[j], top_k=n_want[key], score_thresh=score_thresh)
            g_all = gt[j]
            g = g_all[torch.isfinite(g_all).all(-1)]
            p = p.cpu()
            tracks[(key, ci)][f] = (p.numpy(), g_all.numpy())
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
        per_group[key].setdefault('mota', []).append(
            box_mota(sessions[key], key.split('/', 1)[1], ci, store, ds.input_wh))
    if was_training:
        model.train()
    return {g: _summarise(s) for g, s in per_group.items()}


def overall(rows):
    """Weight groups by their labelled-box count. The per-group table is the unweighted view."""
    n = sum(r['n_gt'] for r in rows.values()) or 1
    out = {k: sum(r[k] * r['n_gt'] for r in rows.values()) / n
           for k in ('r50', 'r75', 'iou', 'fp')}
    m = [r for r in rows.values() if np.isfinite(r['mota'])]
    out['mota'] = (sum(r['mota'] * r['n_gt'] for r in m) / sum(r['n_gt'] for r in m)
                   if m else float('nan'))
    out['n_gt'] = sum(r['n_gt'] for r in rows.values())
    return out
