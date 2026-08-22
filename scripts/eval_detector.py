#!/usr/bin/env python
"""Score a detector run against the crop rule's boxes. Offline, one dataset, one split.

    pixi run python scripts/eval_detector.py --run runs/det-calms21 --data <root> --split test

The metric lives in `tailcyclenet.detector.evaluate`, shared with the training loop. `r@.5`/
`r@.75` are greedy one-to-one recall, `IoU` a mean over every labelled box, `fp` unmatched
predictions per labelled box, `MOTA` box-only with PRESENT rows ignored. `--boxes` must match
what the arm was trained on; score 3dpop on `test` (its val is one session).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import BOX_SOURCES
from tailcyclenet.detector import BoxDataset, TEMPORAL_INPUT_BY_CHANNELS, load_detector
from tailcyclenet.detector.evaluate import deployment_score, score_dataset
from tailcyclenet.format import load_datasets
from tailcyclenet.metrics import paired_bootstrap


def _tiled(run, tile_scale):
    """Refuse a tiled checkpoint rather than score it at the wrong scale.

    This script does ONE whole-frame forward per item, so a tiled arm's `input_wh` is a tile size
    and letterboxing whole frames into it would score the weights at the wrong scale -- score
    tiled arms through `scripts/infer.py`, which derives the input per camera.
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
                         'one under a PAIRED bootstrap.')
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
                    help='0.05, where deployment runs at 0.99. This scores the detector as '
                         'trained; pass 0.99 to see the boxes the pose model is actually served.')
    ap.add_argument('--nms-iou', type=float, default=0.5,
                    help="decode's box-NMS IoU threshold; 0.5 was hardcoded and unreachable "
                         'before detector_v2 plan A1. Sweep upward (RTMDet 0.65, DLC/SLEAP '
                         'instance-level 0.8), not around 0.5.')
    ap.add_argument('--nms-center-dist', type=float, default=None,
                    help='centre-distance NMS threshold in units of box side (scale-free); a '
                         'candidate is also dropped if its centre sits within this many box '
                         'sides of an already-kept box, regardless of IoU (detector_v2 plan A5). '
                         'None (default) is off.')
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--deploy', action='store_true',
                    help='switch to the DEPLOYMENT-SHAPED score: '
                         'det_fill/slot_fill/window_miss/union_side/gt_side over WHOLE test '
                         'groups via detect_raw+associate_group, not the per-view sampled recall. '
                         'Ignores --compare/--batches/--frames-per-group.')
    ap.add_argument('--track', dest='deploy_track', action='store_true', default=True,
                    help='deploy mode only: CrossViewTracker for C>1 (the default, matches '
                         'scripts/infer.py)')
    ap.add_argument('--no-track', dest='deploy_track', action='store_false')
    ap.add_argument('--link-boxes', action='store_true',
                    help='deploy mode only: link_rows for C==1 (2D single-camera identity)')
    ap.add_argument('--n-frames', type=int, default=24, help='deploy mode only: window size')
    ap.add_argument('--overlap', type=int, default=4, help='deploy mode only: window overlap')
    ap.add_argument('--min-box-frames', type=int, default=1,
                    help='deploy mode only: matches infer.InferConfig.min_box_frames')
    ap.add_argument('--top-k', type=int, default=24, help='deploy mode only: detection budget')
    ap.add_argument('--det-max-frames', type=int, default=0,
                    help='deploy mode only: 0 = the whole group; matches infer.py --max-frames')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.deploy:
        return main_deploy(args, device)
    model, wh, _, mcd, red, trained_on, tile_scale, _objq = load_detector(args.run, device=device)
    # A tiled checkpoint's `input_wh` is its TILE size, not its deployment input size. `tile_scale`
    # was unpacked here and never used, so `BoxDataset(input_wh=wh)` letterboxed the WHOLE frame
    # into one tile -- the 1/scale shift `tiled_input_wh`/`detect_group` exist to prevent.
    _tiled(args.run, tile_scale)
    if trained_on != args.boxes:
        print(f'WARNING: {args.run} was trained on {trained_on!r} boxes and is being scored '
              f'against {args.boxes!r} ones. That measures the crop source, not accuracy.')
    # A temporal-input checkpoint's `BoxDataset` must supply the same stacked-frame shape it was
    # trained on; `model.in_channels` (part of the weights) is the source of truth, not a CLI flag.
    ds = BoxDataset(args.data, args.split, input_wh=wh, box_source=args.boxes,
                    min_crop_dim=args.min_crop_dim or mcd, reduce=red,
                    max_frames_per_group=args.frames_per_group,
                    temporal_input=TEMPORAL_INPUT_BY_CHANNELS[model.in_channels])
    rows = score_dataset(model, ds, device, batch_size=args.batch_size, batches=args.batches,
                         seed=args.seed, score_thresh=args.score_thresh,
                         num_workers=args.num_workers, max_animals=args.max_animals,
                         iou_thresh=args.nms_iou, center_dist_thresh=args.nms_center_dist)

    print(f'{args.run}  {args.data.name}/{args.split}  {wh[0]}x{wh[1]}  boxes={args.boxes}  '
          f'min_crop_dim={ds.min_crop_dim}  max_animals={args.max_animals or "(GT count)"}\n')
    # `fp` is `greedy_match`'s count at `top_k = max_animals or GT count` -- BUDGET-CAPPED, and on
    # a single-view root it is close to `1 - r@.5` restated, not an independent quantity. `fp_dup`/
    # `fp_none` come from `box_mota`'s own uncapped pass and are what CLAUDE.md's standing rule
    # means by "want opposite fixes" -- never read `fp` alone as an over-detection number.
    print(f'{"group":40s} {"n_gt":>6s} {"r@.5":>7s} {"r@.75":>7s} {"IoU":>7s} {"fp":>7s} '
          f'{"MOTA":>7s} {"fp_ig":>6s} {"fp_dup":>7s} {"fp_none":>8s} {"miss":>7s}')
    for g, r in sorted(rows.items()):
        print(f'{g[:40]:40s} {r["n_gt"]:6d} {r["r50"]:7.3f} {r["r75"]:7.3f} {r["iou"]:7.3f} '
              f'{r["fp"]:7.3f} {r["mota"]:7.3f} {r["fp_ignored"]:6d} {r["fp_dup"]:7.3f} '
              f'{r["fp_none"]:8.3f} {r["miss"]:7.3f}')

    n_gt = sum(r['n_gt'] for r in rows.values())
    print(f'\n{len(rows)} group(s), {n_gt} labelled boxes')
    for name in ('r50', 'r75', 'iou', 'fp', 'mota', 'fp_dup', 'fp_none', 'miss'):
        b = paired_bootstrap([r[name] for r in rows.values()], seed=args.seed)
        ci = ('DEGENERATE (one group -- no interval exists)' if b['n'] < 2
              else f'[{b["lo"]:.3f}, {b["hi"]:.3f}] 95% over {b["n"]} groups')
        print(f'{name:>7s} {b["mean"]:7.3f}  {ci}')
    fp_ig = sum(r['fp_ignored'] for r in rows.values())
    print(f'fp_ignored (raw count, quote beside MOTA on any 3D root -- CLAUDE.md): {fp_ig}')

    if args.compare:
        m2, wh2, _, mcd2, red2, trained_on2, tile2, _ = load_detector(args.compare,
                                                          device=device)
        _tiled(args.compare, tile2)
        if trained_on2 != trained_on:
            print(f'note: {args.run} was trained on {trained_on!r} boxes and {args.compare} on '
                  f'{trained_on2!r}. The paired delta below moves TWO keys.')
        if wh2 != wh:
            # Pairable, and worth saying why: a letterbox is a uniform scale plus a translation
            # applied to the prediction AND the ground truth alike, and IoU is invariant under
            # that, so recall, IoU and fp/box compare directly across input sizes (box-MOTA too
            # -- its radius derives from the median box diagonal in whichever space it measures).
            print(f'note: {args.compare} runs at {wh2[0]}x{wh2[1]} and --run at {wh[0]}x{wh[1]}. '
                  'Each is scored in its own letterbox; IoU is scale-invariant, so the columns '
                  'below are comparable.')
        ds2 = BoxDataset(args.data, args.split, input_wh=wh2, box_source=args.boxes,
                         min_crop_dim=args.min_crop_dim or mcd2, reduce=red2,
                         max_frames_per_group=args.frames_per_group,
                         temporal_input=TEMPORAL_INPUT_BY_CHANNELS[m2.in_channels])
        other = score_dataset(m2, ds2, device, batch_size=args.batch_size, batches=args.batches,
                              seed=args.seed, score_thresh=args.score_thresh,
                              num_workers=args.num_workers, max_animals=args.max_animals,
                              iou_thresh=args.nms_iou, center_dist_thresh=args.nms_center_dist)
        keys = sorted(set(rows) & set(other))
        print(f'\nPAIRED: {args.run} minus {args.compare}, over {len(keys)} shared group(s)')
        for name in ('r50', 'r75', 'iou', 'fp', 'mota', 'fp_dup', 'fp_none', 'miss'):
            d = paired_bootstrap([rows[k][name] for k in keys],
                                 [other[k][name] for k in keys], seed=args.seed)
            if d['n'] < 2:
                print(f'{name:>7s} {d["mean"]:+7.4f}  DEGENERATE (one group)')
                continue
            # A sign flip inside the interval means the arms are not distinguished on this
            # column; say so rather than leaving two overlapping intervals to compare.
            star = '' if d['lo'] <= 0 <= d['hi'] else '  *'
            print(f'{name:>7s} {d["mean"]:+7.4f}  [{d["lo"]:+.4f}, {d["hi"]:+.4f}]{star}')
        fp_ig1 = sum(rows[k]['fp_ignored'] for k in keys)
        fp_ig2 = sum(other[k]['fp_ignored'] for k in keys)
        print(f'fp_ignored: {args.run}={fp_ig1}  {args.compare}={fp_ig2}  '
              '(raw counts, not paired-bootstrapped)')


def main_deploy(args, device):
    model, wh, _, mcd, red, trained_on, tile_scale, _objq = load_detector(args.run, device=device)
    _tiled(args.run, tile_scale)
    ds = load_datasets(args.data)[0]
    sessions = ds.sessions.get(args.split, [])
    if not sessions:
        raise SystemExit(f'{args.data}: no {args.split!r} split')

    print(f'{args.run}  {args.data.name}/{args.split}  {wh[0]}x{wh[1]}  boxes={trained_on}  '
          f'track={args.deploy_track} link={args.link_boxes}  det_score={args.score_thresh}\n')
    print(f'{"group":40s} {"T":>6s} {"det_fill":>9s} {"slot_fill":>10s} {"win_miss":>9s} '
          f'{"union_p50":>10s} {"union_p90":>10s} {"gt_p50":>8s}')
    rows = []
    for sess in sessions:
        for gid, group in sess.groups.items():
            r = deployment_score(model, sess, gid, input_wh=wh, device=device,
                                 top_k=args.top_k, max_animals=args.max_animals,
                                 det_score=args.score_thresh, track=args.deploy_track,
                                 link=args.link_boxes, min_crop_dim=args.min_crop_dim or mcd,
                                 reduce=red, tile_scale=tile_scale,
                                 max_frames=args.det_max_frames, n_frames=args.n_frames,
                                 overlap=args.overlap, min_box_frames=args.min_box_frames,
                                 iou_thresh=args.nms_iou, center_dist_thresh=args.nms_center_dist)
            rows.append(r)
            # The frames `deployment_score` actually scored, not the group's raw length: a
            # `--det-max-frames` prefix would otherwise print as full-length.
            t_scored = min(group.n_frames, args.det_max_frames) if args.det_max_frames \
                else group.n_frames
            print(f'{f"{sess.session_id}/{gid}"[:40]:40s} {t_scored:6d} '
                  f'{r["det_fill"]:9.4f} {r["slot_fill"]:10.4f} {r["window_miss"]:9.4f} '
                  f'{r["union_side_px"][0.5]:10.1f} {r["union_side_px"][0.9]:10.1f} '
                  f'{r["gt_side_px"][0.5]:8.1f}')

    print(f'\n{len(rows)} group(s)')
    for name in ('det_fill', 'slot_fill', 'window_miss'):
        b = paired_bootstrap([r[name] for r in rows], seed=args.seed)
        ci = ('DEGENERATE (one group -- no interval exists)' if b['n'] < 2
              else f'[{b["lo"]:.3f}, {b["hi"]:.3f}] 95% over {b["n"]} groups')
        print(f'{name:>12s} {b["mean"]:7.4f}  {ci}')
    # union/gt side quantiles are pooled per group (each group already a quantile of many
    # windows/points), so a mean-of-medians is reported rather than bootstrapped a second time.
    for k in (0.5, 0.9, 0.99):
        us = [r['union_side_px'][k] for r in rows if r['union_side_px'][k] == r['union_side_px'][k]]
        gs = [r['gt_side_px'][k] for r in rows if r['gt_side_px'][k] == r['gt_side_px'][k]]
        print(f'  p{int(k*100):>2d}  union_side mean-of-groups {np.mean(us) if us else float("nan"):7.1f} px'
              f'   gt_side mean-of-groups {np.mean(gs) if gs else float("nan"):7.1f} px')


if __name__ == '__main__':
    main()
