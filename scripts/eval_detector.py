#!/usr/bin/env python
"""Score a detector run against the crop rule's boxes. Offline, one dataset, one split.

    pixi run python scripts/eval_detector.py --run runs/det-calms21 --data <root> --split test

`train_detector.evaluate` is a training-progress readout and is generous four ways, all of which
inflate: it matches with `box_iou(gt, pred).max(1)` so ONE well-placed box can satisfy every
animal in a huddle; it hands `decode` the true per-frame animal count, which nothing supplies at
deployment; it reports recall alone, so false positives are structurally invisible; and it reads
50 batches off an unshuffled loader, i.e. the first 800 views of the first session. This is the
number to select on.

What it prints, and why each column is there:

- `r@.5` / `r@.75` -- recall under GREEDY ONE-TO-ONE matching. Two thresholds because calms21
  saturates at 0.5 (0.967 at 0.66M params after 6k iters); a flat 0.5 column there is not
  evidence of no effect.
- `IoU` -- mean over EVERY labelled box, zero for an unmatched one. Mean-over-matched would
  reward a detector that declines the hard animals, which is eval rule 6 in box form.
- `fp` -- unmatched predictions per labelled box, the term recall cannot see.
- `MOTA` -- box-only, from the same greedy pairing, with `instances.pq` PRESENT rows as ignore
  regions. **`idsw` here is not a tracking number**: frames are subsampled per group, so
  "consecutive" rows are not consecutive in time. It is comparable between arms, nothing more.

`--boxes` must match what the arm was TRAINED on. Scoring an `instances`-trained detector against
keypoint boxes measures the crop source and calls it accuracy (eval rule 2).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import BOX_SOURCES
from tailcyclenet.dataset import worker_init
from tailcyclenet.detector import (BoxDataset, ChunkShuffle, box_collate, box_iou, decode,
                                   letterbox_transform, load_detector)
from tailcyclenet.format import INST_PRESENT
from tailcyclenet.metrics import mota, paired_bootstrap


def greedy_match(gt, pred):
    """Greedy one-to-one IoU matching. -> (iou per gt (G,), n_unmatched_pred).

    Highest IoU first, each prediction spent once. `box_iou(gt, pred).max(1)` is not one-to-one:
    in a huddle a single box lands within IoU 0.5 of several animals and is counted for all of
    them.
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


def as_corner_points(boxes):
    """(N,4) xyxy -> (N,2,2), a box as its two diagonal corners.

    `metrics.mota` matches point sets, so a box enters as a two-"keypoint" instance and the match
    distance is the mean corner displacement -- a translation-and-size metric rather than IoU.
    """
    return np.asarray(boxes, float).reshape(-1, 2, 2)


@torch.no_grad()
def score(run, data, split, boxes_source, device, batch_size=16, batches=40, seed=0,
          score_thresh=0.05, frames_per_group=40, num_workers=4, max_animals=None):
    model, wh, _ = load_detector(run, device=device)
    ds = BoxDataset(data, split, input_wh=wh, box_source=boxes_source,
                    max_frames_per_group=frames_per_group)
    order = list(iter(ChunkShuffle(len(ds), seed=seed)))[:batches * batch_size]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, sampler=order, num_workers=num_workers,
        collate_fn=box_collate, worker_init_fn=worker_init)

    per_group = defaultdict(lambda: {'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0.0, 'fp': 0})
    tracks = defaultdict(dict)                  # (key, ci) -> frame -> (pred (P,4), gt (S,4))
    sessions, n_want = {}, {}
    for bi, (x, gt) in enumerate(loader):
        obj, pred_boxes = model(x.to(device))
        for j in range(x.shape[0]):
            sess, gid, f, ci = ds.index[order[bi * batch_size + j]]
            key = f'{sess.session_id}/{gid}'
            if key not in n_want:
                # What deployment supplies: the session's animal count, NOT this frame's true
                # count (`train_detector.evaluate` passes `g.shape[0]`, an oracle).
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
        gid = key.split('/', 1)[1]
        per_group[key].setdefault('mota', []).append(
            box_mota(sessions[key], gid, ci, store, wh))
    return {g: summarise(s) for g, s in per_group.items()}


def box_mota(sess, gid, ci, store, input_wh):
    """Box-only MOTA over this group's sampled frames in one camera view."""
    lab = sess.labels(gid)
    frames = sorted(store)
    T = len(frames)
    P = max(1, max(v[0].shape[0] for v in store.values()))
    S = max(1, max(v[1].shape[0] for v in store.values()))
    pred = np.full((P, T, 2, 2), np.nan)
    true = np.full((S, T, 2, 2), np.nan)
    for t, f in enumerate(frames):
        p, g = store[f]
        pred[:p.shape[0], t] = as_corner_points(p)
        true[:g.shape[0], t] = as_corner_points(g)

    ig = ig_boxes = None
    if lab.instance is not None:
        scale, pad = letterbox_transform(sess.rig.size(sess.cam_names[ci]), input_wh)
        ig = np.zeros((S, T), bool)
        n = min(S, lab.instance.shape[0])
        ig[:n] = (lab.instance[:n][:, frames, ci] == INST_PRESENT)
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


def summarise(s):
    n = max(s['n_gt'], 1)
    m = s.get('mota', [])
    gt = sum(r['gt'] for r in m)
    return {'n_gt': s['n_gt'], 'r50': s['hit50'] / n, 'r75': s['hit75'] / n,
            'iou': s['iou'] / n, 'fp': s['fp'] / n,
            'mota': (sum(r['mota'] * r['gt'] for r in m if np.isfinite(r['mota'])) / gt
                     if gt else float('nan')),
            'fp_ignored': sum(r['fp_ignored'] for r in m)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path, help='detector run folder or .pth')
    ap.add_argument('--data', required=True, type=Path, help='ONE dataset root')
    ap.add_argument('--split', default='test',
                    help="3dpop's val split is ONE session -- score it on test. Pass `train` for "
                         'the train/val gap, which is what selects between augmentation and '
                         'resolution.')
    ap.add_argument('--boxes', default='keypoints', choices=BOX_SOURCES,
                    help='MUST match what the run was trained on')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--batches', type=int, default=40)
    ap.add_argument('--frames-per-group', type=int, default=40)
    ap.add_argument('--max-animals', type=int, default=None,
                    help='top_k for decode; default is the session\'s own animal count, which is '
                         'what scripts/infer.py supplies')
    ap.add_argument('--score-thresh', type=float, default=0.05)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    rows = score(args.run, args.data, args.split, args.boxes, device,
                 batch_size=args.batch_size, batches=args.batches, seed=args.seed,
                 score_thresh=args.score_thresh, frames_per_group=args.frames_per_group,
                 num_workers=args.num_workers, max_animals=args.max_animals)

    print(f'{args.run}  {args.data.name}/{args.split}  boxes={args.boxes}\n')
    print(f'{"group":40s} {"n_gt":>6s} {"r@.5":>7s} {"r@.75":>7s} {"IoU":>7s} {"fp":>7s} '
          f'{"MOTA":>7s} {"fp_ig":>6s}')
    for g, r in sorted(rows.items()):
        print(f'{g[:40]:40s} {r["n_gt"]:6d} {r["r50"]:7.3f} {r["r75"]:7.3f} {r["iou"]:7.3f} '
              f'{r["fp"]:7.3f} {r["mota"]:7.3f} {r["fp_ignored"]:6d}')

    n_gt = sum(r['n_gt'] for r in rows.values())
    print(f'\n{len(rows)} group(s), {n_gt} labelled boxes')
    for name in ('r50', 'r75', 'iou', 'fp', 'mota'):
        b = paired_bootstrap([r[name] for r in rows.values()], seed=args.seed)
        ci = ('DEGENERATE (one group -- no interval exists)' if b['n'] < 2
              else f'[{b["lo"]:.3f}, {b["hi"]:.3f}] 95% over {b["n"]} groups')
        print(f'{name:>5s} {b["mean"]:7.3f}  {ci}')


if __name__ == '__main__':
    main()
