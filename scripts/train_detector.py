#!/usr/bin/env python
"""Train the box predictor. ONE DETECTOR PER DATASET.

    pixi run python scripts/train_detector.py --data <dataset root> --out runs/det-rat-city

The input size is dataset-specific and it matters more than it looks. rat-city's frames are
4696x2048 (2.29:1); letterboxed into a square 416 that scales by min(416/2048, 416/4696) = 0.089,
so the frame becomes 416x181 in a 416x416 canvas -- 56% black padding -- and the median rat
arrives at 15.8 x 12.5 px. YOLOX pools at strides 8/16/32, so that animal is ~2 x 1.6 cells at
the finest level and absent from the other two: two thirds of the FPN cannot represent it.
Measured against branson-fly (same detector, same 416, but a square 1024x1024 frame) the median
fly arrives at 26.5 x 28.1 px and reaches AP50 0.985 where rat-city sits near 0.50.

So `--input-wh` defaults to an aspect-matched size rather than a square, and training the
detector across datasets is not offered: one letterbox cannot serve both.

`--boxes instances` regresses the dataset's own `instances.pq` extent instead of the keypoint
extent. rat-city wants it: its converter dropped noisy points, so 26k train instances carry no
finite keypoint at all and would otherwise be trained as "no animal here".

Every checkpoint is written as its own `detector_it<n>.pth` WITH its scores inside, plus a
`metrics.json` of the whole history, and both splits are scored each time. A single rolling
`detector.pth` carrying no score cannot be selected on -- johnson peaked at val recall 0.871 and
shipped 0.706 -- and the TRAIN score is what says whether a dataset's problem is generalisation
(a train/val gap) or capacity to fit at all (no gap, low absolute recall). Scoring is
`detector.evaluate`, the same code `scripts/eval_detector.py` reports with.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import BOX_SOURCES
from tailcyclenet.dataset import worker_init
from tailcyclenet.format import load_datasets
from tailcyclenet.detector import (BoxDataset, ChunkShuffle, YOLOXNano, box_collate,
                                  detector_loss, split_batch, tiled_input_wh)
from tailcyclenet.detector.evaluate import overall, score_dataset


def param_groups(model, no_decay_norm):
    """What AdamW is handed. `model.parameters()` unless `--no-decay-norm`, and then two groups.

    RANK IS THE TEST, not the parameter's name. Every conv weight in YOLOX-Nano is rank 4 and
    every GroupNorm affine and every head bias is rank 1, so `p.ndim <= 1` separates them without
    matching on strings that a refactor would silently break. Returned as a plain iterator when the
    flag is off so the optimizer sees exactly what it always saw.
    """
    if not no_decay_norm:
        return model.parameters()
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    plain = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return [{'params': decay}, {'params': plain, 'weight_decay': 0.0}]


def default_input_wh(dataset, target_px=416 * 416):
    """An input size matched to the frame's aspect ratio, at roughly a square-416 pixel budget."""
    sess = next(iter(next(iter(dataset.sessions.values()))))
    w, h = sess.rig.size(sess.cam_names[0])
    ar = w / h
    ow = int(round((target_px * ar) ** 0.5 / 32) * 32)
    oh = int(round((target_px / ar) ** 0.5 / 32) * 32)
    return max(ow, 64), max(oh, 64)


def input_wh_for(path, dataset, box_source, min_box_px=32, max_px=4 * 416 * 416):
    """Aspect-matched, then RAISED until the median animal is `min_box_px` across.

    A pixel budget is a property of the frame; what the detector can represent is a property of
    the ANIMAL. YOLOX pools at strides 8/16/32, and an object spans at least one cell of stride s
    only when its side is at least s -- so `min_box_px = 32`, the coarsest stride, is the size at
    which the typical animal exists at all three FPN levels instead of only the finest. Below it,
    two thirds of the pyramid is being trained on something it cannot see.

    Measured at the plain 416^2 budget, median animal side in detector pixels (p10 in brackets):

        calms21    108  [87]      already 3.4x the coarsest stride
        rat-city    32  [26]      exactly at it
        3dpop       23  [17]      HALF a cell at stride 32 for the smaller decile

    This is a FLOOR, never a ceiling: calms21's rule-implied size is 0.30x its budget, and
    shrinking a dataset that already works to make its animals merely adequate would be a strange
    thing to do with the saving. `max_px` caps the other end, and the cap is reported rather than
    applied quietly -- a dataset that hits it is one where the animals stay unrepresentable and
    that is a fact about the run.
    """
    base = default_input_wh(dataset)
    if min_box_px <= 0:
        return base
    ds = BoxDataset(path, 'train', input_wh=base, box_source=box_source, max_frames_per_group=4)
    ix = np.random.default_rng(0).choice(len(ds), min(300, len(ds)), replace=False)
    sides = torch.cat([(b[:, 2:] - b[:, :2]).flatten()
                       for b in (ds.boxes_for(int(i)) for i in ix)])
    sides = sides[torch.isfinite(sides)]
    if not sides.numel():
        return base
    med, p10 = float(sides.median()), float(sides.quantile(0.1))
    print(f'median animal {med:.1f} px (p10 {p10:.1f}) at {base[0]}x{base[1]}', end='')
    if med >= min_box_px:
        print(f' -- already >= {min_box_px} (stride 32), keeping it')
        return base
    s = min_box_px / med
    if base[0] * base[1] * s * s > max_px:
        s = (max_px / (base[0] * base[1])) ** 0.5
        print(f' -- want x{min_box_px / med:.2f}, CAPPED at x{s:.2f} by max_px={max_px}; the '
              f'median animal lands at {med * s:.1f} px, still under {min_box_px}', end='')
    ow = max(64, int(round(base[0] * s / 32) * 32))
    oh = max(64, int(round(base[1] * s / 32) * 32))
    print(f' -> {ow}x{oh}')
    return ow, oh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, type=Path, help='ONE dataset root')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--input-wh', type=int, nargs=2, default=None,
                    help='overrides --min-box-px entirely')
    ap.add_argument('--min-box-px', type=int, default=32,
                    help='raise the input size until the MEDIAN animal is this many detector '
                         'pixels across. 32 is YOLOX\'s coarsest stride, i.e. the size at which '
                         'the typical animal spans at least one cell at every FPN level. A floor '
                         'only -- it never shrinks a dataset whose animals are already large. '
                         '0 disables it and restores the plain 416^2 aspect-matched budget. '
                         'MEASURED on 3dpop test, paired over 16 groups against the 23 px '
                         'baseline: 32 -> +0.049 r@.75 at 1.9x the pixels, 64 -> +0.102 at 7.7x. '
                         'The curve keeps rising past 32; 32 is where it is free enough to be a '
                         'default, not where it stops paying.')
    ap.add_argument('--max-input-px', type=int, default=4 * 416 * 416,
                    help='ceiling on --min-box-px. It BINDS: 3dpop needs 4.3x the 416^2 budget '
                         'for a 48 px median and 7.7x for 64, so the cap is the difference '
                         'between a rule and a blank cheque. Reported when it applies.')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup-frac', type=float, default=0.0,
                    help='fraction of --iters spent on a LinearLR warmup from 1%% of --lr before '
                         'the cosine takes over. 0 (the default) is the shipped schedule, byte-'
                         'identical. WHY IT IS AN ARM: we run AdamW at 1e-3 from RANDOM INIT with '
                         'no warmup, where DeepLabCut, SLEAP and APT are all at 1e-4. APT '
                         'documents the failure by name -- "RTMDet needs its LinearLR warmup, '
                         'since AdamW at the nominal lr of 4e-3 can diverge on a freshly '
                         'initialized head" -- and caps its own warmup at dl_steps // 10. Random '
                         'init is precisely the case warmup protects, and the one pathology on '
                         'record here (whole-frame dense recall peaking at 4-8k of 20k and falling '
                         'monotonically) is an early-training shape. 0.05 is APT\'s ratio.')
    ap.add_argument('--no-decay-norm', action='store_true',
                    help='exclude GroupNorm affines and biases (every parameter of rank <= 1) from '
                         'weight decay. Off by default, which is byte-identical. AdamW currently '
                         'decays them along with the conv weights, and with GroupNorm that is more '
                         'than cosmetic: decaying a norm\'s scale toward zero fights the '
                         'normalisation it exists to provide. None of the three reference systems '
                         'decays norm parameters either.')
    ap.add_argument('--ema-decay', type=float, default=0.0, metavar='D',
                    help='keep an exponential moving average of the weights at decay D and score '
                         'and checkpoint it ALONGSIDE the raw weights. 0 (the default) is off and '
                         'byte-identical -- the averaged model is never built, so the raw path is '
                         'untouched arithmetic. 0.9998 is YOLOX\'s value. EMA is YOLOX\'s signature '
                         'and we inherited everything but it; it is also the standard treatment '
                         'for the one pathology on record here, whole-frame dense recall peaking '
                         'at 4-8k of 20k and falling monotonically, which best-on-val selection '
                         'currently works around rather than fixes. THE POINT OF SCORING BOTH IN '
                         'ONE RUN is that the arms are then PAIRED BY CONSTRUCTION -- same data '
                         'order, same augmentation draws, same seed -- at no extra training cost, '
                         'where two separate runs would differ by seed noise as well as by EMA.')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--frames-per-group', type=int, default=40)
    ap.add_argument('--min-crop-dim', type=int, default=64,
                    help='MUST equal the pose run\'s [data].min_crop_dim -- it is the same crop '
                         'rule. Stored in the checkpoint, and scripts/infer.py refuses a mismatch.')
    ap.add_argument('--boxes', default='instances', choices=BOX_SOURCES,
                    help='what the regression target bounds. DEFAULTS TO `instances`, matching '
                         '`dataset.LoaderConfig.box_source` so the detector reproduces the crop '
                         'the pose model is trained on -- the two must agree or the detector '
                         'serves a different crop rule, silently. Inert where a root ships no '
                         'instances.pq (3dpop, allen, branson-fly: falls back per view); an exact '
                         'no-op on rat-city-annotated; worth 3.7% of animal-frames on '
                         'rat-city-tracked, whose keypoint box is ~2.4 of 4 points. It DOES '
                         'retarget calms21 (MARS) and johnson-mouse (COCO), whose boxes agree with '
                         'the crop rule on 0.000 of instances -- pass `keypoints` there to '
                         'reproduce reports 10-13. See dataset.LoaderConfig.box_source.')
    ap.add_argument('--val-frames-per-group', type=int, default=8,
                    help='A DATASET WITH ONE GROUP GETS ONE GROUP\'S WORTH OF VAL. rat-city and '
                         'branson-fly each hold a single val group, so the default 8 makes the '
                         'recall readout 8 images; raise it to the group\'s labelled length.')
    ap.add_argument('--augment', action='store_true',
                    help='random similarity + brightness on the TRAIN split. Helps where a '
                         'dataset fits its training data and lags on val; on one that fits '
                         'neither it is the wrong lever. Read the train/val gap first.')
    ap.add_argument('--reduce', action='store_true',
                    help='decode JPEGs at 1/N via libjpeg where the frame is far above the '
                         'letterbox target. A KEY, not a loader detail: it changes which source '
                         'pixels reach the model. rat-city 37.0 -> 21.8 ms/item, and it replaces '
                         'a 7.3x INTER_LINEAR downscale -- which samples 2x2 of every 7x7 block '
                         '-- with a proper box filter. Stored in the checkpoint; inference '
                         'reads it back. No effect on a video root.')
    ap.add_argument('--rotate-deg', type=float, default=0.0,
                    help='in-plane rotation drawn from [-deg, +deg], on top of --augment. 0 is '
                         'off and off is byte-identical (the draw is skipped, not zeroed). 180 is '
                         'a full circle. IT IS THE REPLACEMENT FOR THE FLIP --keypoints COSTS, and '
                         'strictly easier: a mirror permutes left/right names and needs a '
                         'flip_pairs map, a rotation permutes nothing. FREE WHEN TILING, because '
                         'the warp turns about the tile centre and a tile interior to the frame '
                         'pulls real neighbouring pixels in at every angle. Not free on whole '
                         'frames: on a 2.29:1 frame the mean real-pixel fraction is 0.92 at 15, '
                         '0.79 at 45 and 0.644 (min 0.437) at BOTH 90 and 180 -- the whole cost '
                         'is paid by the first 90 degrees, so stopping short of a circle saves '
                         'nothing. Stored in the checkpoint.')
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--eval-batches', type=int, default=25,
                    help='batches per split at each checkpoint. TRAIN is scored too: the '
                         'train/val gap is what says whether a dataset needs augmentation '
                         '(a gap) or resolution (no gap, low absolute recall).')
    ap.add_argument('--keypoints', action='store_true',
                    help='train a bottom-up KEYPOINT BRANCH beside the box head. Off by default '
                         'and off means NOT CONSTRUCTED: with this absent the model, the loader '
                         'and the loss are byte-identical to every recorded detector, so no '
                         'existing recipe needs a new flag to reproduce. K is derived from the '
                         "dataset's own registry, never configured. Turning it on also disables "
                         'the horizontal flip -- see `random_affine`, whose no-`flip_pairs` '
                         'justification holds only while the target is a box.')
    ap.add_argument('--no-hflip', action='store_true',
                    help='drop the horizontal flip from the augmentation. `--keypoints` already '
                         'does this implicitly, so this exists for the box-only CONTROL arm that '
                         'has to match it -- otherwise the control differs in two levers.')
    ap.add_argument('--tile-wh', type=int, nargs=2, default=None,
                    help='train on TILES of this size in INPUT pixels instead of whole frames. '
                         'Off by default and off means whole frames, byte-identical to every '
                         'recorded detector. This is the model\'s input size; the tile\'s SOURCE '
                         'extent is --tile-wh / --tile-scale. Inference is unchanged -- one '
                         'whole-frame forward at --tile-scale, derived per camera. Its ONE '
                         'justification is --use-regions (report 16 §5.3b refuted the other).')
    ap.add_argument('--tile-scale', type=float, default=1.0,
                    help='source -> input scale for --tile-wh. 1.0 = native. DO NOT ALSO RESIZE '
                         'THE TILE: the invariant is the animal\'s size in INPUT pixels, and '
                         'tiling-then-downscaling is the number-one reported failure of this '
                         'pattern. MEASURED on rat-city-annotated: this is what sets the mask\'s '
                         'positive rate (5.2% at 1.0, 10.2% at 0.7, 17.5% at 0.5, 48% at 0.25) '
                         'because CENTER_RADIUS is 2.5 CELLS, so the certified region has to span '
                         'many cells, not much area.')
    ap.add_argument('--tile-bg-per-frame', type=int, default=1,
                    help='background tiles per frame, centres inside the certified area')
    ap.add_argument('--use-regions', action='store_true',
                    help='mask the objectness loss to the area regions.pq certifies as completely '
                         'labelled. ORTHOGONAL to --tile-wh so an arm can move one lever. MEASURED '
                         'DEAD on whole frames: 69%% of supervised anchors are positive against '
                         '0.68%% unmasked and 17%% of frames have no certified negative at all. '
                         'Use it WITH tiles. A session with no regions.pq claims exhaustive '
                         'labelling and is unmasked.')
    ap.add_argument('--kpt-weight', type=float, default=1.0)
    ap.add_argument('--kpt-score-weight', type=float, default=1.0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    # Just the camera size, so just the discovery -- building a BoxDataset here scattered every
    # session's parquet into dense arrays to read two integers.
    roots = load_datasets(args.data)
    if len(roots) != 1:
        raise SystemExit(f'{args.data}: the detector is trained per dataset; found {len(roots)}')
    probe_sess = roots[0].all_sessions()[0]
    wh = (tuple(args.input_wh) if args.input_wh
          else input_wh_for(args.data, roots[0], args.boxes, args.min_box_px,
                            args.max_input_px))
    print(f'input {wh[0]}x{wh[1]}  (frame {probe_sess.rig.size(probe_sess.cam_names[0])})')

    tiling = dict(tile_wh=args.tile_wh, tile_scale=args.tile_scale,
                  tile_bg_per_frame=args.tile_bg_per_frame, use_regions=args.use_regions)
    train = BoxDataset(args.data, 'train', input_wh=wh, box_source=args.boxes,
                       min_crop_dim=args.min_crop_dim, augment=args.augment, reduce=args.reduce,
                       max_frames_per_group=args.frames_per_group, keypoints=args.keypoints,
                       hflip=0.0 if args.no_hflip else None, rotate_deg=args.rotate_deg, **tiling)
    # THE CHECKPOINT'S `input_wh` MUST BE THE SIZE THE MODEL SAW. When tiling, `BoxDataset`
    # resolves it to the tile, so read it back from there rather than from `input_wh_for` -- which
    # returned the whole-frame letterbox size and would have recorded a size the weights never saw.
    wh = train.input_wh
    if args.tile_wh:
        ext = train._tile_extent()
        print(f'tiling: {args.tile_wh[0]}x{args.tile_wh[1]} input px at scale '
              f'{args.tile_scale:g} = {ext[0]:.0f}x{ext[1]:.0f} SOURCE px, '
              f'{args.tile_bg_per_frame} background tile(s)/frame')
        print(f'  DEPLOYMENT INPUT is the whole frame at this scale, NOT the tile size: '
              f'{tiled_input_wh(probe_sess.rig.size(probe_sess.cam_names[0]), args.tile_scale)}')
    print(f'train: {len(train)} views')
    # DERIVED from the registry, never configured -- the same rule `n_keypoints` follows on the
    # pose side. A configured K that disagreed with the data would mis-index every target.
    n_kpts = len(roots[0].names) if args.keypoints else 0
    if args.keypoints:
        print(f'keypoint branch: {n_kpts} keypoints, hflip disabled')
        # A MASKED KEYPOINT GETS ZERO GRADIENT, so a rarely-labelled one is never trained -- and
        # it still emits a number at inference, off the conv bias. That is the accepted cost of
        # "predict all K always", but it must be VISIBLE: a hollow output should be a line in this
        # log rather than a mystery at eval time. Sampled, not exhaustive; the point is the shape.
        cen = np.zeros(n_kpts)
        step = max(1, len(train) // 200)
        seen = 0
        for j in range(0, len(train), step):
            _, k = train.boxes_for(j, None, with_keypoints=True)
            cen += np.isfinite(k[..., :2].numpy()).all(-1).sum(0)
            seen += k.shape[0]
        frac = cen / max(seen, 1)
        names = roots[0].names
        thin = [f'{names[i]} {frac[i]:.2f}' for i in np.argsort(frac)[:5]]
        print(f'  labelled fraction per keypoint over {seen} sampled instances: '
              f'min {frac.min():.3f}  median {np.median(frac):.3f}  max {frac.max():.3f}')
        print(f'  thinnest: {", ".join(thin)}', flush=True)
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, sampler=ChunkShuffle(len(train), chunk=train.chunk),
        num_workers=args.num_workers,
        collate_fn=box_collate, drop_last=True, persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init)
    val = None
    try:
        val = BoxDataset(args.data, 'val', input_wh=wh, box_source=args.boxes,
                         min_crop_dim=args.min_crop_dim, reduce=args.reduce,
                         max_frames_per_group=args.val_frames_per_group,
                         keypoints=args.keypoints, **tiling)
        print(f'val:   {len(val)} views')
    except ValueError as e:
        print(f'val:   none ({e})')

    model = YOLOXNano(n_keypoints=n_kpts).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'YOLOX-Nano: {n / 1e6:.2f}M params')
    opt = torch.optim.AdamW(param_groups(model, args.no_decay_norm), lr=args.lr, weight_decay=5e-4)
    # THE SCHEDULE IS BUILT IN ONE BRANCH OR THE OTHER, NEVER BOTH. Constructing a
    # CosineAnnealingLR sets the optimizer's lr as a side effect at __init__, so building a spare
    # one to hand to SequentialLR and then discarding it would leave the warmup starting from a
    # perturbed lr -- and `--warmup-frac 0` would stop being the shipped schedule.
    if args.warmup_frac > 0:
        w = max(1, int(round(args.iters * args.warmup_frac)))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=w),
                  torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.iters - w))],
            milestones=[w])
        print(f'schedule: LinearLR 0.01 -> 1.0 over {w} iters, then cosine over {args.iters - w}')
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    # THE AVERAGED MODEL IS NOT BUILT AT ALL WHEN OFF. `AveragedModel` deep-copies the module, so
    # constructing it unconditionally would allocate a second network and change nothing else --
    # harmless, but it would also make "off" a different code path from the one every recorded
    # detector number came from, which is the thing every flag in this file is careful not to do.
    ema = None
    if args.ema_decay:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
        print(f'EMA: decay {args.ema_decay}, scored and checkpointed beside the raw weights')

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    # A SECOND best-tracker, because the two arms peak at different iterations -- that is most of
    # what EMA is for, and sharing one `best_score` would let whichever arm happened to be ahead
    # decide which checkpoint the OTHER arm ships.
    best_ema = -float('inf')
    # `detector.pth` IS THE BEST CHECKPOINT, NOT THE LAST ONE. It used to be the last: the file was
    # rewritten at every evaluation and `best` was computed after the loop and only PRINTED, so a
    # run measured its own peak and then threw it away. That is not a tie-break -- on
    # rat-city-annotated recall PEAKS AT 4-8k AND FALLS MONOTONICALLY to 20k (whole-frame dense
    # r@.5: tile 0.387 -> 0.278, tilemask 0.350 -> 0.288, tilemask-rot 0.407 -> 0.372), because a
    # root whose labelled frame names 2 of ~10 rats spends 20,000 iterations teaching the objectness
    # head that most rats are background. Every `detector_it*.pth` is still written, so a run that
    # wants the last one still has it.
    best_score = -float('inf')
    it, t0, running = 0, time.time(), []
    model.train()
    while it < args.iters:
        for batch in loader:
            if it >= args.iters:
                break
            # BY RANK, not by tuple length: with --keypoints off and --use-regions on, the
            # third element is regions, and reading it as `gt_kpts` would train the keypoint
            # branch against rectangles.
            x, gt, gt_kpts, gt_regions = split_batch(batch)
            x, gt = x.to(device), gt.to(device)
            gt_kpts = None if gt_kpts is None else gt_kpts.to(device)
            gt_regions = None if gt_regions is None else gt_regions.to(device)
            obj, boxes, kpt = model(x)
            anchors = model.anchor_points(x.shape[-2], x.shape[-1], device)
            loss, parts = detector_loss(obj, boxes, anchors, gt, kpts=kpt, gt_kpts=gt_kpts,
                                        kpt_weight=args.kpt_weight,
                                        kpt_score_weight=args.kpt_score_weight,
                                        regions=gt_regions)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            # AFTER `opt.step()`, so the average is over the weights the optimiser actually
            # produced. Before it, every update would average in the PREVIOUS step twice.
            if ema is not None:
                ema.update_parameters(model)
            sched.step()
            running.append(float(loss.detach()))
            it += 1
            if it % 50 == 0:
                kp = (f'  kpt {parts["kpt"]:6.3f}  kscore {parts["kpt_score"]:5.3f}'
                      if 'kpt' in parts else '')
                # The certified FRACTION, printed because masking shrinks the objectness sum
                # without shrinking its divisor -- it silently reweights obj against box_weight,
                # and that has to be a number in the log rather than an inference from a curve.
                kp += f'  cert {parts["certified"]:5.3f}' if 'certified' in parts else ''
                print(f'{it:7d}/{args.iters}  loss {np.mean(running):7.4f}  '
                      f'obj {parts["obj"]:6.3f}  box {parts["box"]:6.3f}{kp}  '
                      f'pos {parts["n_pos"]:4d}  {(time.time() - t0) / 50:5.3f}s/it', flush=True)
                running, t0 = [], time.time()
            if it % args.eval_every == 0 or it == args.iters:
                # Both splits, EVERY checkpoint, and the score stored beside the weights. A
                # rolling `detector.pth` with no score cannot be selected on: johnson peaked at
                # val recall 0.871 and shipped 0.706, branson peaked 0.885 and shipped 0.833.
                scores = {}
                for name, ds in (('train', train), ('val', val)):
                    if ds is None:
                        continue
                    scores[name] = overall(score_dataset(
                        model, ds, device, batch_size=args.batch_size,
                        batches=args.eval_batches, num_workers=2))
                # `n_keypoints` rides in the checkpoint beside `input_wh` for the same reason:
                # it is part of the WEIGHTS, not a runtime choice, and absent reads as 0 -- which
                # is a fact about the file ("no keypoint weights here"), not an assertion about
                # how weights nobody recorded were trained. That distinction is what gotcha 12
                # cost, one level down.
                ckpt = {'iteration': it, 'model_state': model.state_dict(), 'input_wh': wh,
                        'n_keypoints': n_kpts, 'norm': 'gn',
                        # `input_wh` above is the TILE size when tiling, which is NOT the
                        # deployment input size -- `load_detector` raises if this is missing so
                        # nobody can run a tiled detector at its tile size on a whole frame.
                        'tile_wh': args.tile_wh, 'tile_scale': args.tile_scale,
                        'use_regions': args.use_regions,
                        'dataset': train.ds.name, 'box_source': args.boxes,
                        'min_crop_dim': args.min_crop_dim, 'augment': args.augment,
                        'reduce': args.reduce, 'rotate_deg': args.rotate_deg,
                        # Recorded for the same reason `rotate_deg` is: neither changes a tensor's
                        # shape, so a checkpoint trained under one schedule and compared against
                        # another is indistinguishable from the file alone. Gotcha 12's shape.
                        'warmup_frac': args.warmup_frac, 'no_decay_norm': args.no_decay_norm,
                        'ema_decay': args.ema_decay, 'ema': False,
                        'eval': scores}
                torch.save(ckpt, args.out / f'detector_it{it:06d}.pth')
                # Selected on `val` where there is one, `train` otherwise -- the same key the
                # end-of-run `best` line reports, so the printed winner and the shipped file are
                # now the same checkpoint instead of two different ones.
                sel = scores.get('val', scores.get('train', {})).get('r50', -float('inf'))
                if sel >= best_score:
                    best_score = sel
                    torch.save(ckpt, args.out / 'detector.pth')
                # THE EMA ARM, scored through the IDENTICAL path on the IDENTICAL windows. The
                # averaged weights live inside `ema.module`, so `score_dataset` never learns there
                # is an EMA at all and the two arms cannot diverge through their evaluator.
                escores = {}
                if ema is not None:
                    for name, ds in (('train', train), ('val', val)):
                        if ds is None:
                            continue
                        escores[name] = overall(score_dataset(
                            ema.module, ds, device, batch_size=args.batch_size,
                            batches=args.eval_batches, num_workers=2))
                    eck = dict(ckpt, model_state=ema.module.state_dict(), eval=escores, ema=True)
                    torch.save(eck, args.out / f'detector_ema_it{it:06d}.pth')
                    esel = escores.get('val', escores.get('train', {})).get('r50', -float('inf'))
                    if esel >= best_ema:
                        best_ema = esel
                        torch.save(eck, args.out / 'detector_ema.pth')
                    for name, s in escores.items():
                        print(f'   ema {name:3s} r@.5 {s["r50"]:.4f}  r@.75 {s["r75"]:.4f}  '
                              f'IoU {s["iou"]:.4f}  fp {s["fp"]:.3f}  MOTA {s["mota"]:.3f}',
                              flush=True)
                history.append({'iteration': it,
                                **{f'{k}_{m}': v[m] for k, v in scores.items()
                                   for m in ('r50', 'r75', 'iou', 'fp', 'mota')},
                                **{f'ema_{k}_{m}': v[m] for k, v in escores.items()
                                   for m in ('r50', 'r75', 'iou', 'fp', 'mota')}})
                (args.out / 'metrics.json').write_text(json.dumps(history, indent=1))
                for name, s in scores.items():
                    print(f'   {name:5s} r@.5 {s["r50"]:.4f}  r@.75 {s["r75"]:.4f}  '
                          f'IoU {s["iou"]:.4f}  fp {s["fp"]:.3f}  MOTA {s["mota"]:.3f}',
                          flush=True)
                t0 = time.time()               # evaluation is not part of the s/it readout
    best = max(history, key=lambda h: h.get('val_r50', h['train_r50'])) if history else None
    print(f'done: {it} iterations -> {args.out}')
    if best:
        print(f'best: it {best["iteration"]} (this is what detector.pth holds)  ' +
              '  '.join(f'{k} {v:.4f}' for k, v in best.items() if k != 'iteration'))


if __name__ == '__main__':
    main()
