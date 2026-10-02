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
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import BOX_SOURCES
from tailcyclenet.detector import BoxDataset, load_detector
from tailcyclenet.detector.evaluate import deployment_score, score_dataset, score_fixed_views
from tailcyclenet.format import load_datasets
from tailcyclenet.metrics import paired_bootstrap


class FixedSplitDataset:
    """Index a set of BoxDatasets as one deterministic evaluation dataset."""

    def __init__(self, datasets):
        """Combine split-specific datasets while preserving their item order."""
        self.parts = datasets
        self.offsets = np.cumsum([0] + [len(ds) for ds in datasets]).tolist()
        self.index = [item for ds in datasets for item in ds.index]
        self.root_ids = [root_id for ds in datasets for root_id in ds.root_ids]
        self.root_names = list(dict.fromkeys(name for ds in datasets for name in ds.root_names))
        self.datasets = list(dict((id(root), root) for ds in datasets for root in ds.datasets).values())
        self.chunk = 1
        self.min_crop_dim = datasets[0].min_crop_dim
        self._augment = False

    @property
    def augment(self):
        """Return the shared augmentation setting."""
        return self._augment

    @augment.setter
    def augment(self, value):
        """Set augmentation on every split dataset."""
        self._augment = value
        for ds in self.parts:
            ds.augment = value

    @property
    def tt_transform(self):
        """Return the shared test-time image transform."""
        return self.parts[0].tt_transform

    @tt_transform.setter
    def tt_transform(self, value):
        """Set the test-time transform on every split dataset."""
        for ds in self.parts:
            ds.tt_transform = value

    def _locate(self, index):
        """Map a combined item index to its child dataset and local index."""
        part = max(i for i, start in enumerate(self.offsets[:-1]) if start <= index)
        return self.parts[part], index - self.offsets[part]

    def __len__(self):
        """Return the total number of views."""
        return len(self.index)

    def __getitem__(self, index):
        """Load one view from the matching split dataset."""
        ds, local = self._locate(index)
        return ds[local]

    def ignore_for(self, index):
        """Return transformed ignore annotations for one combined item."""
        ds, local = self._locate(index)
        return ds.ignore_for(local)

    def boxes_for(self, index):
        """Return target boxes for one combined item."""
        ds, local = self._locate(index)
        return ds.boxes_for(local)


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
    """Score a detector run against crop-rule boxes; exit via SystemExit on bad config.

    Inputs: argv (via argparse): --run, --compare, --data, --split, --boxes,
            --min-crop-dim, --batch-size, --batches, --frames-per-group,
            --max-animals, --score-thresh, --nms-iou, --nms-center-dist,
            --num-workers, --seed, --device, --deploy, --track/--no-track,
            --link-boxes, --n-frames, --overlap, --min-box-frames, --top-k,
            --det-max-frames.
    Side effects: prints the per-group table and bootstrapped aggregates.

    A temporal-input checkpoint's BoxDataset must supply the same stacked-frame shape it was
    trained on; `model.in_channels` (part of the weights) is the source of truth, not a CLI
    flag. `fp` is `greedy_match`'s count at `top_k = max_animals or GT count` -- BUDGET-CAPPED,
    and on a single-view root it is close to `1 - r@.5` restated, not an independent quantity;
    `fp_dup`/`fp_none` come from `box_mota`'s own uncapped pass. Two runs at different
    `input_wh` are pairable: a letterbox is a uniform scale plus a translation applied to the
    prediction and the ground truth alike, and IoU is invariant under that. A sign flip inside
    a paired interval means the arms are not distinguished on that column.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path, help='detector run folder or .pth')
    ap.add_argument('--checkpoint', default='latest', help='checkpoint file within run folder')
    ap.add_argument('--evalset', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--tt-gain', type=float, nargs='+', default=[1.0])
    ap.add_argument('--tt-gamma', type=float, nargs='+', default=[1.0])
    ap.add_argument('--det-input-wh', type=int, nargs=2)
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
    ap.add_argument('--score-thresh', type=float, nargs='+', default=[0.05],
                    help='0.05: score_dataset\'s own long-standing "as-trained" convention '
                         '(unrelated to deployment tuning -- this is what the model candidate-'
                         'decoded, not what a deployment run would keep). ALSO used as --det-score '
                         'in --deploy mode, where it means something different: infer.py\'s own '
                         '--det-score default is now 0.01 (CHANGED from 0.5 then 0.05, dev/reports/'
                         '44_detector_recommended_defaults.md) -- pass --score-thresh 0.01 '
                         'explicitly under --deploy to match it; this flag\'s own default stays '
                         '0.05 for its primary, non-deploy purpose.')
    ap.add_argument('--nms-iou', type=float, default=0.5,
                    help="decode's box-NMS IoU threshold; 0.5 was hardcoded and unreachable "
                         'before detector_v2 plan A1. Sweep upward (RTMDet 0.65, DLC/SLEAP '
                         'instance-level 0.8), not around 0.5.')
    ap.add_argument('--nms-center-dist', type=float, default=0.3,
                    help='centre-distance NMS threshold in units of box side (scale-free); a '
                         'candidate is also dropped if its centre sits within this many box '
                         'sides of an already-kept box, regardless of IoU. 0.3 is the deployment '
                         'default from paired CALMS/rat-city controls; pass 0.5 for the historical '
                         'A5 operating point or a negative value to disable.')
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
    model, wh, _, mcd, red, trained_on, tile_scale, _objq = load_detector(args.run, device=device, checkpoint=args.checkpoint, input_wh=args.det_input_wh)
    if args.det_input_wh:
        wh = tuple(args.det_input_wh)
    score_thresh = args.score_thresh[0]
    _tiled(args.run, tile_scale)
    if trained_on != args.boxes:
        print(f'WARNING: {args.run} was trained on {trained_on!r} boxes and is being scored '
              f'against {args.boxes!r} ones. That measures the crop source, not accuracy.')
    from tailcyclenet.detector.data import tt_photometric_transform
    eval_spec = json.loads(args.evalset.read_text()) if args.evalset else None
    split_names = sorted({item.get('split', args.split) for item in eval_spec['frames']}) if eval_spec else args.split.split(',')
    parts = [BoxDataset(args.data, split, input_wh=wh, box_source=args.boxes,
                        min_crop_dim=args.min_crop_dim or mcd, reduce=red,
                        max_frames_per_group=args.frames_per_group,
                        box_target=getattr(model, 'box_target', 'crop'),
                        antialias=getattr(model, 'antialias', False)) for split in split_names]
    if eval_spec:
        for split in split_names:
            part = parts[split_names.index(split)]
            present = {(sess.session_id, gid, frame, ci) for sess, gid, frame, ci in part.index}
            for entry in eval_spec['frames']:
                if entry.get('split', args.split) != split:
                    continue
                sess = next((sess for root in part.datasets for sess in root.all_sessions()
                             if sess.session_id == entry['session']), None)
                if sess is None:
                    raise SystemExit(f"{entry['session']}: session not found in split {split}")
                for ci in range(len(sess.cam_names)):
                    key = (sess.session_id, entry['group'], int(entry['frame']), ci)
                    if key not in present:
                        part.index.append((sess, entry['group'], int(entry['frame']), ci))
                        part.origins.append(None)
                        part.root_ids.append(0)
                        present.add(key)
    ds = parts[0] if len(parts) == 1 else FixedSplitDataset(parts)
    index_splits = [split for split, part in zip(split_names, parts) for _ in part.index]
    fixed = None
    if args.evalset:
        spec = eval_spec
        fixed = set()
        for item in spec['frames']:
            for i, (sess, gid, frame, ci) in enumerate(ds.index):
                if (sess.session_id == item['session'] and gid == item['group']
                        and frame == int(item['frame'])
                        and index_splits[i] == item.get('split', split_names[0])):
                    fixed.add(i)
        if not fixed:
            raise SystemExit(f'{args.evalset}: no matching views in {args.data}/{args.split}')
    # A fixed eval set is scored in the PARENT by `score_fixed_views` (num_workers=0); a later
    # forked loader worker then inherits a decoder mid-state and deadlocks on video roots
    # (CLAUDE.md gotcha 10 -- qdmouse's mp4 guard hung for hours at 0% CPU). The set is at most a
    # few hundred views, so load it in-process throughout.
    workers = 0 if fixed is not None else args.num_workers
    result_blocks = []
    for gain in args.tt_gain:
        for gamma in args.tt_gamma:
            ds.tt_transform = lambda img, g=gain, ga=gamma: tt_photometric_transform(img, g, ga)
            blocks = []
            for threshold in args.score_thresh:
                scores_for_threshold = []
                rows = score_dataset(model, ds, device, batch_size=args.batch_size,
                                     batches=max(args.batches, (len(fixed) + args.batch_size - 1)//args.batch_size) if fixed else args.batches,
                                     seed=args.seed, score_thresh=threshold,
                                     num_workers=workers, max_animals=args.max_animals,
                                     iou_thresh=args.nms_iou, center_dist_thresh=args.nms_center_dist,
                                     subset_indices=fixed, out_scores=scores_for_threshold)
                from tailcyclenet.detector.evaluate import overall
                summary = overall(rows)
                values = np.concatenate(scores_for_threshold) if scores_for_threshold else np.zeros(0)
                summary['objectness_quantiles'] = ({f'q{int(q * 100):02d}': float(np.quantile(values, q))
                                                    for q in (0.01, 0.1, 0.5, 0.9)}
                                                   if values.size else {})
                view_rows = (score_fixed_views(model, ds, fixed, device, score_thresh=threshold,
                                               batch_size=args.batch_size, max_animals=args.max_animals,
                                               iou_thresh=args.nms_iou,
                                               center_dist_thresh=args.nms_center_dist)
                             if fixed is not None else [])
                blocks.append({'score_thresh': threshold, 'aggregate': summary,
                               'per_view': view_rows})
            result_blocks.append({'tt_gain': gain, 'tt_gamma': gamma,
                                  'input_wh': list(wh), 'thresholds': blocks})
    ds.tt_transform = None
    rows = score_dataset(model, ds, device, batch_size=args.batch_size, batches=args.batches,
                         seed=args.seed, score_thresh=score_thresh,
                         num_workers=workers, max_animals=args.max_animals,
                         iou_thresh=args.nms_iou, center_dist_thresh=args.nms_center_dist,
                         subset_indices=fixed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {'evalset': str(args.evalset) if args.evalset else None,
                   'blocks': result_blocks}
        args.out.write_text(json.dumps(payload, indent=2, default=float) + '\n')

    print(f'{args.run}  {args.data.name}/{args.split}  {wh[0]}x{wh[1]}  boxes={args.boxes}  '
          f'min_crop_dim={ds.min_crop_dim}  max_animals={args.max_animals or "(GT count)"}\n')
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
            print(f'note: {args.compare} runs at {wh2[0]}x{wh2[1]} and --run at {wh[0]}x{wh[1]}. '
                  'Each is scored in its own letterbox; IoU is scale-invariant, so the columns '
                  'below are comparable.')
        if getattr(m2, 'box_target', 'crop') != getattr(model, 'box_target', 'crop'):
            print(f'note: {args.run} regresses {getattr(model, "box_target", "crop")!r} boxes and '
                  f'{args.compare} {getattr(m2, "box_target", "crop")!r} ones; each is scored '
                  'against its own target, so IoU/r@.75 compare different boxes. Use --deploy '
                  '(crop-rule boxes for both) for a like-for-like comparison.')
        ds2 = BoxDataset(args.data, args.split, input_wh=wh2, box_source=args.boxes,
                         min_crop_dim=args.min_crop_dim or mcd2, reduce=red2,
                         max_frames_per_group=args.frames_per_group,
                         box_target=getattr(m2, 'box_target', 'crop'),
                         antialias=getattr(m2, 'antialias', False))
        other = score_dataset(m2, ds2, device, batch_size=args.batch_size, batches=args.batches,
                              seed=args.seed, score_thresh=args.score_thresh[0],
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
            star = '' if d['lo'] <= 0 <= d['hi'] else '  *'
            print(f'{name:>7s} {d["mean"]:+7.4f}  [{d["lo"]:+.4f}, {d["hi"]:+.4f}]{star}')
        fp_ig1 = sum(rows[k]['fp_ignored'] for k in keys)
        fp_ig2 = sum(other[k]['fp_ignored'] for k in keys)
        print(f'fp_ignored: {args.run}={fp_ig1}  {args.compare}={fp_ig2}  '
              '(raw counts, not paired-bootstrapped)')


def main_deploy(args, device):
    """The deployment-shaped score: whole groups through detect_raw+associate_group.

    Inputs: args -- the parsed CLI args (deploy mode's subset).
            device -- the torch device.
    Side effects: prints per-group det_fill/slot_fill/window_miss and side-quantile rows.

    DEPLOYMENT's detect_raw path handles tile_scale itself: it derives a whole-frame input size
    per camera while preserving the animal's trained input-pixel scale; `_tiled` belongs only to
    score_dataset's one-whole-frame loader. `t_scored` is the frame count actually scored, so a
    `--det-max-frames` prefix does not print as full-length. union/gt side quantiles are pooled
    per group (each group is already a quantile of many windows/points), so a mean-of-medians is
    reported rather than bootstrapped a second time.
    """
    model, wh, _, mcd, red, trained_on, tile_scale, _objq = load_detector(args.run, device=device, checkpoint=args.checkpoint, input_wh=args.det_input_wh)
    if args.det_input_wh:
        wh = tuple(args.det_input_wh)
    score_thresh = args.score_thresh[0]
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
                                 det_score=score_thresh, track=args.deploy_track,
                                 link=args.link_boxes, min_crop_dim=args.min_crop_dim or mcd,
                                 reduce=red, tile_scale=tile_scale,
                                 max_frames=args.det_max_frames, n_frames=args.n_frames,
                                 overlap=args.overlap, min_box_frames=args.min_box_frames,
                                 iou_thresh=args.nms_iou, center_dist_thresh=args.nms_center_dist)
            rows.append(r)
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
    for k in (0.5, 0.9, 0.99):
        us = [r['union_side_px'][k] for r in rows if r['union_side_px'][k] == r['union_side_px'][k]]
        gs = [r['gt_side_px'][k] for r in rows if r['gt_side_px'][k] == r['gt_side_px'][k]]
        print(f'  p{int(k*100):>2d}  union_side mean-of-groups {np.mean(us) if us else float("nan"):7.1f} px'
              f'   gt_side mean-of-groups {np.mean(gs) if gs else float("nan"):7.1f} px')


if __name__ == '__main__':
    main()
