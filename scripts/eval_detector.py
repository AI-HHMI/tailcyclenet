#!/usr/bin/env python
"""Score a detector run against the crop rule's boxes. Offline, one dataset, one split.

    pixi run python scripts/eval_detector.py --run runs/det-calms21 --data <root> --split test

The metric itself lives in `tailcyclenet.detector.evaluate`, shared with the training loop so the
number a checkpoint is selected on and the number it is reported with cannot diverge. What the
columns mean, and why each is there:

- `r@.5` / `r@.75` -- recall under GREEDY ONE-TO-ONE matching. Two thresholds because calms21
  saturates at 0.5 (0.967 at 0.66M params after 6k iters); a flat 0.5 column there is not
  evidence of no effect.
- `IoU` -- mean over EVERY labelled box, zero for an unmatched one.
- `fp` -- unmatched predictions per labelled box, the term recall cannot see.
- `MOTA` -- box-only, with `instances.pq` PRESENT rows as ignore regions. Its `idsw` component is
  not a tracking number; see `evaluate.box_mota`.

Two things to get right when using it:

- `--boxes` must match what the arm was TRAINED on. Scoring an `instances`-trained detector
  against keypoint boxes measures the crop source and calls it accuracy (eval rule 2).
- 3dpop's val split is ONE session, so score it on `test`. `--split train` gives the train/val
  gap, which is what decides between augmentation and resolution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import BOX_SOURCES
from tailcyclenet.detector import BoxDataset, load_detector
from tailcyclenet.detector.evaluate import score_dataset
from tailcyclenet.metrics import paired_bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path, help='detector run folder or .pth')
    ap.add_argument('--data', required=True, type=Path, help='ONE dataset root')
    ap.add_argument('--split', default='test')
    ap.add_argument('--boxes', default='keypoints', choices=BOX_SOURCES,
                    help='MUST match what the run was trained on')
    ap.add_argument('--min-crop-dim', type=int, default=None,
                    help='default: the checkpoint\'s own, which is the pose model\'s')
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
    model, wh, _, mcd = load_detector(args.run, device=device)
    ds = BoxDataset(args.data, args.split, input_wh=wh, box_source=args.boxes,
                    min_crop_dim=args.min_crop_dim or mcd,
                    max_frames_per_group=args.frames_per_group)
    rows = score_dataset(model, ds, device, batch_size=args.batch_size, batches=args.batches,
                         seed=args.seed, score_thresh=args.score_thresh,
                         num_workers=args.num_workers, max_animals=args.max_animals)

    print(f'{args.run}  {args.data.name}/{args.split}  {wh[0]}x{wh[1]}  boxes={args.boxes}  '
          f'min_crop_dim={ds.min_crop_dim}\n')
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
