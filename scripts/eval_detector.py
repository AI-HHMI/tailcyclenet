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


def _tiled(run, tile_scale):
    """Refuse a tiled checkpoint rather than score it at the wrong scale.

    Refusing rather than supporting: this script does ONE whole-frame forward per item, which is
    also what deployment does, so the honest fix is to letterbox the frame at `tile_scale` -- and
    that is `detect_group`'s job, per camera, because a root can ship two frame sizes
    (rat-city-annotated is 4696x2048 beside 4500x2050). Scoring a tiled arm goes through
    `scripts/infer.py`; this raises so that the choice is visible instead of silent.
    """
    if tile_scale:
        raise SystemExit(
            f'{run}: trained on tiles (tile_scale={tile_scale}), so its input_wh is a TILE size '
            'and letterboxing whole frames into it would score the weights at the wrong scale. '
            'Score it through scripts/infer.py, which derives the input size per camera.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path, help='detector run folder or .pth')
    ap.add_argument('--compare', type=Path, default=None,
                    help='a second run, scored on the SAME groups, reported as `--run` minus this '
                         'one under a PAIRED bootstrap. Unpaired intervals on the same groups '
                         'overstate the uncertainty enough to hide a real effect (eval rule 3): '
                         'augmentation on 3dpop reads +0.005 r@.5 [-0.000, +0.010] paired, and '
                         'two overlapping [0.93, 0.98] intervals unpaired.')
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
    ap.add_argument('--score-thresh', type=float, default=0.05,
                    help='0.05, where DEPLOYMENT runs at 0.99 (`scripts/infer.py --det-score`). '
                         'Deliberately not the same number: this scores the detector as trained, and '
                         'every figure in dev/reports/10 is at 0.05. Pass 0.99 to see the boxes the '
                         'pose model is actually served.')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model, wh, _, mcd, red, trained_on, tile_scale, _objq = load_detector(args.run, device=device)
    # A TILED CHECKPOINT'S `input_wh` IS ITS TILE SIZE, NOT ITS DEPLOYMENT INPUT SIZE (gotcha 12's
    # shape). `tile_scale` was unpacked here and never used, so `BoxDataset(input_wh=wh)`
    # letterboxed the WHOLE FRAME into one tile -- the 1/scale shift `tiled_input_wh` and
    # `detect_group` exist to prevent, which `scripts/infer.py` guards and this script did not.
    # A tiled arm scored here reads near-zero recall for a reason that is nothing to do with its
    # weights, and that is precisely the arm this script exists to compare.
    _tiled(args.run, tile_scale)
    if trained_on != args.boxes:
        print(f'WARNING: {args.run} was trained on {trained_on!r} boxes and is being scored '
              f'against {args.boxes!r} ones. That measures the crop source, not accuracy '
              '(eval rule 2).')
    ds = BoxDataset(args.data, args.split, input_wh=wh, box_source=args.boxes,
                    min_crop_dim=args.min_crop_dim or mcd, reduce=red,
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

    if args.compare:
        m2, wh2, _, mcd2, red2, trained_on2, tile2, _ = load_detector(args.compare,
                                                          device=device)
        _tiled(args.compare, tile2)
        if trained_on2 != trained_on:
            print(f'note: {args.run} was trained on {trained_on!r} boxes and {args.compare} on '
                  f'{trained_on2!r}. The paired delta below moves TWO keys (eval rule 4).')
        if wh2 != wh:
            # Pairable, and worth saying why: a letterbox is a uniform scale plus a translation
            # applied to the prediction AND the ground truth alike, and IoU is invariant under
            # that. So recall, IoU and fp/box compare directly across input sizes, and box-MOTA
            # does too -- its match radius is derived from the median box diagonal in whichever
            # space it is measuring. What does NOT carry across is anything in absolute pixels,
            # and this scorer reports none.
            print(f'note: {args.compare} runs at {wh2[0]}x{wh2[1]} and --run at {wh[0]}x{wh[1]}. '
                  'Each is scored in its own letterbox; IoU is scale-invariant, so the columns '
                  'below are comparable.')
        ds2 = BoxDataset(args.data, args.split, input_wh=wh2, box_source=args.boxes,
                         min_crop_dim=args.min_crop_dim or mcd2, reduce=red2,
                         max_frames_per_group=args.frames_per_group)
        other = score_dataset(m2, ds2, device, batch_size=args.batch_size, batches=args.batches,
                              seed=args.seed, score_thresh=args.score_thresh,
                              num_workers=args.num_workers, max_animals=args.max_animals)
        keys = sorted(set(rows) & set(other))
        print(f'\nPAIRED: {args.run} minus {args.compare}, over {len(keys)} shared group(s)')
        for name in ('r50', 'r75', 'iou', 'fp', 'mota'):
            d = paired_bootstrap([rows[k][name] for k in keys],
                                 [other[k][name] for k in keys], seed=args.seed)
            if d['n'] < 2:
                print(f'{name:>5s} {d["mean"]:+7.4f}  DEGENERATE (one group)')
                continue
            # A sign flip inside the interval means the arms are not distinguished on this
            # column. Saying so beats leaving a reader to compare two overlapping intervals.
            star = '' if d['lo'] <= 0 <= d['hi'] else '  *'
            print(f'{name:>5s} {d["mean"]:+7.4f}  [{d["lo"]:+.4f}, {d["hi"]:+.4f}]{star}')


if __name__ == '__main__':
    main()
