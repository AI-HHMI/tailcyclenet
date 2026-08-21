#!/usr/bin/env python
"""Train the box predictor. ONE DETECTOR PER DATASET.

    pixi run python scripts/train_detector.py --config configs/detector.toml

The recipe lives in the CONFIG, not on the command line -- the same split the pose side uses
(`scripts/train.py` + `configs/base.toml`). `configs/detector.toml` ships every current default
with its reasoning attached; a user overlay may `extends` it one level deep and change only what
differs (see `checkpoints.load_config`). Per-root recipes from dev/reports/32 section 2.2 are
expressed as key overrides in your config. Only three CLI knobs remain: `--out`, `--iters` and
`--device` override `[training]` for a smoke test or a one-off. An unknown key in any block
RAISES, so a typo cannot silently train at defaults.

The run folder records the effective `config.toml` and `provenance.toml` (commit + dirty flag)
before training starts, so a detector run carries the same reproducibility record a pose run
does -- gotcha 12's shape. The checkpoint itself is byte-compatible with every detector on
record: `load_detector`, `scripts/eval_detector.py` and `scripts/infer.py` read the same fields
they always did.

The input size is dataset-specific and it matters more than it looks. rat-city's frames are
4696x2048 (2.29:1); letterboxed into a square 416 that scales by min(416/2048, 416/4696) = 0.089,
so the frame becomes 416x181 in a 416x416 canvas -- 56% black padding -- and the median rat
arrives at 15.8 x 12.5 px. YOLOX pools at strides 8/16/32, so that animal is ~2 x 1.6 cells at
the finest level and absent from the other two: two thirds of the FPN cannot represent it.
Measured against branson-fly (same detector, same 416, but a square 1024x1024 frame) the median
fly arrives at 26.5 x 28.1 px and reaches AP50 0.985 where rat-city sits near 0.50.

So `[data].input_wh` defaults to an aspect-matched size rather than a square, and training the
detector across datasets is not offered: one letterbox cannot serve both. `[data].min_box_px`
raises that size until the median animal is representable at every FPN level.

`[data].boxes = "instances"` regresses the dataset's own `instances.pq` extent instead of the
keypoint extent. rat-city wants it: its converter dropped noisy points, so 26k train instances
carry no finite keypoint at all and would otherwise be trained as "no animal here". Set it to
`"keypoints"` on calms21 / johnson-mouse / 3dpop-comparability runs -- see the config.

`[model].yolox` is a MODEL-CAPACITY switch, DEFAULT `tiny` per user instruction
(dev/reports/30 section 5.3 found this default is NOT evidence-backed -- `tiny`'s apparent
downstream lead over `trimmed` on rat-city-combined did not survive its own seed-replicate
check). `trimmed` is the repo's original bespoke ~0.66M-param net, byte-identical to every
detector on record before dev/reports/28; set `yolox = "trimmed"` to restore it. The named tiers
build the canonical YOLOX backbone at Megvii's own (depth_mul, width_mul, depthwise) -- see
`tailcyclenet.detector.yolox.YOLOX_TIERS` -- to test whether the detector is capacity-limited
rather than resolution-limited (dev/reports/16 §5.3b reached "capacity-limited" for the keypoint
branch by elimination, never by scaling the model). It is recorded in the checkpoint and
`load_detector` reconstructs the matching architecture; absent means `trimmed` (every checkpoint
written before this flag existed).

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

import toml

from tailcyclenet.checkpoints import provenance
from tailcyclenet.dataset import worker_init
from tailcyclenet.detector import (BoxDataset, ChunkShuffle, TEMPORAL_INPUT_CHANNELS, YOLOXNano,
                                   box_collate, detector_loss, split_batch, tiled_input_wh)
from tailcyclenet.detector.config import load_detector_config
from tailcyclenet.detector.evaluate import overall, score_dataset
from tailcyclenet.detector.pretrained import load_coco_backbone, load_pretrained_backbone
from tailcyclenet.format import load_datasets


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


def _record_run(run: Path, config: dict) -> None:
    """Write the run folder's reproducibility record: the effective config + provenance.

    `config` is the dict `load_detector_config` returned. `None` values (an absent `input_wh` /
    `tile_wh`) are dropped rather than serialised, so the recorded file round-trips through
    `load_detector_config` to the same recipe. Checkpoint contents are untouched -- this is the
    folder's record, not the weights'.
    """
    import copy

    run.mkdir(parents=True, exist_ok=True)
    record = copy.deepcopy(config)
    for block in record.values():
        if isinstance(block, dict):
            for k in [k for k, v in block.items() if v is None]:
                del block[k]
    (run / 'config.toml').write_text(toml.dumps(record))
    prov = provenance()
    if prov:
        (run / 'provenance.toml').write_text(toml.dumps(prov))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path,
                    help='the detector training config (see configs/detector.toml)')
    ap.add_argument('--out', type=Path, default=None, help='override [training].out')
    ap.add_argument('--iters', type=int, default=None, help='override [training].iters')
    ap.add_argument('--device', default=None, help='override [training].device')
    args = ap.parse_args()

    config = load_detector_config(args.config, out=args.out, iters=args.iters,
                                  device=args.device)
    data_cfg, model_cfg, train_cfg = config['data'], config['model'], config['training']

    torch.manual_seed(train_cfg['seed'])
    np.random.seed(train_cfg['seed'])
    device = train_cfg['device'] if torch.cuda.is_available() else 'cpu'
    run = Path(train_cfg['out'])
    _record_run(run, config)

    # Just the camera size, so just the discovery -- building a BoxDataset here scattered every
    # session's parquet into dense arrays to read two integers.
    roots = load_datasets(data_cfg['path'])
    if len(roots) != 1:
        raise SystemExit(f'{data_cfg["path"]}: the detector is trained per dataset; '
                         f'found {len(roots)}')
    probe_sess = roots[0].all_sessions()[0]
    wh = (tuple(data_cfg['input_wh']) if data_cfg['input_wh']
          else input_wh_for(data_cfg['path'], roots[0], data_cfg['boxes'],
                            data_cfg['min_box_px'], data_cfg['max_input_px']))
    print(f'input {wh[0]}x{wh[1]}  (frame {probe_sess.rig.size(probe_sess.cam_names[0])})')

    tiling = dict(tile_wh=data_cfg['tile_wh'], tile_scale=data_cfg['tile_scale'],
                  tile_bg_per_frame=data_cfg['tile_bg_per_frame'],
                  use_regions=data_cfg['use_regions'],
                  ignore_present=data_cfg['ignore_present'])
    train = BoxDataset(data_cfg['path'], 'train', input_wh=wh,
                       box_source=data_cfg['boxes'], min_crop_dim=data_cfg['min_crop_dim'],
                       augment=data_cfg['augment'], reduce=data_cfg['reduce'],
                       max_frames_per_group=data_cfg['frames_per_group'],
                       keypoints=data_cfg['keypoints'],
                       hflip=0.0 if not data_cfg['hflip'] else None,
                       rotate_deg=data_cfg['rotate_deg'], strong=data_cfg['augment_strong'],
                       temporal_input=data_cfg['temporal_input'],
                       seed=train_cfg['seed'], **tiling)
    # THE CHECKPOINT'S `input_wh` MUST BE THE SIZE THE MODEL SAW. When tiling, `BoxDataset`
    # resolves it to the tile, so read it back from there rather than from `input_wh_for` -- which
    # returned the whole-frame letterbox size and would have recorded a size the weights never saw.
    wh = train.input_wh
    if data_cfg['tile_wh']:
        ext = train._tile_extent()
        print(f'tiling: {data_cfg["tile_wh"][0]}x{data_cfg["tile_wh"][1]} input px at scale '
              f'{data_cfg["tile_scale"]:g} = {ext[0]:.0f}x{ext[1]:.0f} SOURCE px, '
              f'{data_cfg["tile_bg_per_frame"]} background tile(s)/frame')
        print(f'  DEPLOYMENT INPUT is the whole frame at this scale, NOT the tile size: '
              f'{tiled_input_wh(probe_sess.rig.size(probe_sess.cam_names[0]), data_cfg["tile_scale"])}')
    print(f'train: {len(train)} views')
    # DERIVED from the registry, never configured -- the same rule `n_keypoints` follows on the
    # pose side. A configured K that disagreed with the data would mis-index every target.
    n_kpts = len(roots[0].names) if data_cfg['keypoints'] else 0
    if data_cfg['keypoints']:
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
        train, batch_size=train_cfg['batch_size'],
        sampler=ChunkShuffle(len(train), chunk=train.chunk, seed=train_cfg['seed']),
        num_workers=train_cfg['num_workers'],
        collate_fn=box_collate, drop_last=True,
        persistent_workers=train_cfg['num_workers'] > 0,
        worker_init_fn=worker_init)
    # NO `val/` IS THE ONLY THING SWALLOWED HERE, and it is tested for rather than caught:
    # `BoxDataset.__init__` also raises ValueError for real config errors (a non-positive
    # `tile_scale`, the multi-root refusal, `strong` beside `use_regions`), and those are meant to
    # FAIL AT CONSTRUCTION rather than degrade to a printed note on the val loader.
    val = None
    root = Path(data_cfg['path'])
    if not ((root / 'val').is_dir() or any(
            (c / 'val').is_dir() for c in root.iterdir() if c.is_dir())):
        print(f'val:   none (no val/ split under {root})')
    else:
        val = BoxDataset(data_cfg['path'], 'val', input_wh=wh,
                         box_source=data_cfg['boxes'], min_crop_dim=data_cfg['min_crop_dim'],
                         reduce=data_cfg['reduce'],
                         max_frames_per_group=data_cfg['val_frames_per_group'],
                         keypoints=data_cfg['keypoints'], seed=train_cfg['seed'],
                         temporal_input=data_cfg['temporal_input'], **tiling)
        print(f'val:   {len(val)} views')

    # T4.2 (dev/plans/detector_accuracy.md): the stem's input width is DERIVED from
    # `[data].temporal_input`, never independently configured -- `TEMPORAL_INPUT_CHANNELS` is the
    # one place that map lives, so the loader (which supplies the channels) and the model (which
    # consumes them) cannot disagree.
    in_channels = TEMPORAL_INPUT_CHANNELS[data_cfg['temporal_input']]
    model = YOLOXNano(n_keypoints=n_kpts, version=model_cfg['yolox'],
                      bottleneck_expansion=model_cfg['bottleneck_expansion'],
                      p2=model_cfg['p2'], in_channels=in_channels).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'YOLOX [{model_cfg["yolox"]}]: {n / 1e6:.2f}M params'
          f'  (bottleneck_expansion={model_cfg["bottleneck_expansion"]:g}, '
          f'temporal_input={data_cfg["temporal_input"]!r}, in_channels={in_channels})')
    if model_cfg['pretrained'] == 'coco':
        n_loaded, n_total = load_coco_backbone(model, model_cfg['yolox'])
        print(f'  loaded COCO backbone: {n_loaded}/{n_total} conv tensors '
              f'(dev/plans/detector_accuracy.md T4.1)', flush=True)
    elif model_cfg['pretrained']:
        # T4.1b: an in-domain backbone from scripts/pretrain_detector_backbone.py. No scale/
        # channel correction (see load_pretrained_backbone's own docstring) -- this repo's own
        # pretraining already speaks this repo's own [0,1] RGB convention.
        n_loaded = load_pretrained_backbone(model, model_cfg['pretrained'])
        print(f'  loaded in-domain backbone: {n_loaded} conv tensors from '
              f'{model_cfg["pretrained"]} (dev/plans/detector_accuracy.md T4.1b)', flush=True)

    # DIFFERENTIAL LR (T4.1/T4.1b): ANY pretrained backbone (COCO or in-domain) at
    # BACKBONE_LR_SCALE x lr, the fresh neck/head at lr -- mirroring the pose side's own
    # discipline for a similarly asymmetric unfreeze (`video_encoder_requires_grad`), NOT its
    # staged-unfreeze machinery, which exists to solve schedule-free's `1/k` averaging problem and
    # this optimiser is plain AdamW throughout. For COCO, conv weights trained under BatchNorm
    # statistics landing in a fresh GroupNorm net is the reason for the lower backbone LR (see
    # `pretrained.py`'s own docstring); for an in-domain backbone there is no BN/GN mismatch, but
    # the general asymmetry -- a fresh head paired with an already-informative backbone -- still
    # applies, so the same scale is used rather than inventing an unmeasured second number.
    # A `pretrained=''` run builds ONE param group, unconditionally -- so this is byte-identical
    # to every optimizer on record whenever pretraining is not asked for.
    BACKBONE_LR_SCALE = 0.1
    if model_cfg['pretrained']:
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
        other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        opt = torch.optim.AdamW(
            [{'params': backbone_params, 'lr': train_cfg['lr'] * BACKBONE_LR_SCALE},
             {'params': other_params, 'lr': train_cfg['lr']}],
            weight_decay=5e-4)
        print(f'  differential LR: backbone {train_cfg["lr"] * BACKBONE_LR_SCALE:g}  '
              f'neck/head {train_cfg["lr"]:g}', flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg['iters'])

    history = []
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
    while it < train_cfg['iters']:
        for batch in loader:
            if it >= train_cfg['iters']:
                break
            # BY RANK, not by tuple length: with --keypoints off and --use-regions on, the
            # third element is regions, and reading it as `gt_kpts` would train the keypoint
            # branch against rectangles. `split_batch`'s 4th slot is `regions` OR `ignore_present`
            # boxes -- `BoxDataset.__init__` refuses to build both at once, so this dataset's OWN
            # two flags (never both true) say which one it is; `split_batch` cannot, since a
            # rank-2 (M,4) tensor looks the same either way.
            x, gt, gt_kpts, gt_tail = split_batch(batch)
            x, gt = x.to(device), gt.to(device)
            gt_kpts = None if gt_kpts is None else gt_kpts.to(device)
            gt_tail = None if gt_tail is None else gt_tail.to(device)
            gt_regions = gt_tail if data_cfg['use_regions'] else None
            gt_ignore = gt_tail if data_cfg['ignore_present'] else None
            out = model(x)
            obj, boxes, kpt = out[0], out[1], out[2]
            anchors = model.anchor_points(x.shape[-2], x.shape[-1], device)
            loss, parts = detector_loss(obj, boxes, anchors, gt, kpts=kpt, gt_kpts=gt_kpts,
                                        kpt_weight=train_cfg['kpt_weight'],
                                        kpt_score_weight=train_cfg['kpt_score_weight'],
                                        regions=gt_regions, ignore=gt_ignore,
                                        iou_aware=train_cfg['iou_aware_obj'],
                                        iou_aware_warmup=train_cfg['iou_aware_warmup'], it=it,
                                        max_pos_per_gt=train_cfg['max_pos_per_gt'] or None,
                                        box_weight=train_cfg['box_weight'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
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
                kp += f'  ign {parts["ignored"]:5.3f}' if 'ignored' in parts else ''
                kp += f'  id {parts["ident"]:6.3f}' if 'ident' in parts else ''
                kp += f'  iouT {parts["iou_target"]:5.3f}' if 'iou_target' in parts else ''
                print(f'{it:7d}/{train_cfg["iters"]}  loss {np.mean(running):7.4f}  '
                      f'obj {parts["obj"]:6.3f}  box {parts["box"]:6.3f}{kp}  '
                      f'pos {parts["n_pos"]:4d}  {(time.time() - t0) / 50:5.3f}s/it', flush=True)
                running, t0 = [], time.time()
            if it % train_cfg['eval_every'] == 0 or it == train_cfg['iters']:
                # Both splits, EVERY checkpoint, and the score stored beside the weights. A
                # rolling `detector.pth` with no score cannot be selected on: johnson peaked at
                # val recall 0.871 and shipped 0.706, branson peaked 0.885 and shipped 0.833.
                scores, obj_scores = {}, []
                for name, ds in (('train', train), ('val', val)):
                    if ds is None:
                        continue
                    # `obj_scores` is collected from the LAST split scored (val where there is
                    # one), which is the distribution deployment meets.
                    obj_scores.clear()
                    scores[name] = overall(score_dataset(
                        model, ds, device, batch_size=train_cfg['batch_size'],
                        batches=train_cfg['eval_batches'], num_workers=2,
                        out_scores=obj_scores))
                # THE OBJECTNESS DISTRIBUTION, RECORDED BECAUSE `--det-score` CANNOT BE A CONSTANT.
                # 0.99 was chosen against detectors whose objectness is saturated (98.5% of
                # rat-city's boxes at exactly 1.0) and is wrong for the tiled/masked generation,
                # which reads q01 0.45-0.84 and loses two thirds of its detections to it --
                # coverage 0.703 against 0.986 at 0.50 (dev/reports/21 0b). Saturation is a
                # property of the RECIPE, not the dataset, so the only durable fix is to record
                # what this checkpoint actually produces and let `infer.py` warn.
                obj_q = {}
                if obj_scores:
                    a = np.concatenate(obj_scores)
                    obj_q = {f'q{int(q * 100):02d}': float(np.quantile(a, q))
                             for q in (0.01, 0.10, 0.50, 0.90)}
                    print(f'   objectness q01 {obj_q["q01"]:.4f}  q10 {obj_q["q10"]:.4f}  '
                          f'q50 {obj_q["q50"]:.4f}  q90 {obj_q["q90"]:.4f}'
                          + ('   <-- NOT saturated: --det-score 0.99 would drop most of these'
                             if obj_q['q50'] < 0.99 else ''), flush=True)
                # `n_keypoints` rides in the checkpoint beside `input_wh` for the same reason:
                # it is part of the WEIGHTS, not a runtime choice, and absent reads as 0 -- which
                # is a fact about the file ("no keypoint weights here"), not an assertion about
                # how weights nobody recorded were trained. That distinction is what gotcha 12
                # cost, one level down.
                # `yolox_version` is the fifth field of this shape (gotcha 12): the architecture
                # is part of the WEIGHTS, and absent reads as 'trimmed' -- a fact about every
                # checkpoint written before this switch existed, not a guess.
                ckpt = {'iteration': it, 'model_state': model.state_dict(), 'input_wh': wh,
                        'n_keypoints': n_kpts, 'norm': 'gn', 'yolox_version': model_cfg['yolox'],
                        # T4.1 (dev/plans/detector_accuracy.md), part of the WEIGHTS like
                        # `yolox_version` beside it -- absent means 0.5 / '', a fact about every
                        # checkpoint written before this key existed, not a guess.
                        'bottleneck_expansion': model_cfg['bottleneck_expansion'],
                        'pretrained': model_cfg['pretrained'],
                        # T4.3 (dev/plans/detector_accuracy.md): same gotcha-12 shape -- absent
                        # means False, a fact about every checkpoint written before the P2 level
                        # existed, not a guess.
                        'p2': model_cfg['p2'],
                        # T4.2 (dev/plans/detector_accuracy.md): same gotcha-12 shape. `in_channels`
                        # is what `load_detector` needs to rebuild the right stem (absent means 3,
                        # every checkpoint written before this key existed); `temporal_input` rides
                        # beside it purely for provenance/debugging -- nothing reconstructs the
                        # model from it directly.
                        'in_channels': in_channels,
                        'temporal_input': data_cfg['temporal_input'],
                        'seed': train_cfg['seed'],
                        # `input_wh` above is the TILE size when tiling, which is NOT the
                        # deployment input size -- `load_detector` raises if this is missing so
                        # nobody can run a tiled detector at its tile size on a whole frame.
                        'tile_wh': data_cfg['tile_wh'], 'tile_scale': data_cfg['tile_scale'],
                        'use_regions': data_cfg['use_regions'],
                        'ignore_present': data_cfg['ignore_present'],
                        'dataset': train.ds.name, 'box_source': data_cfg['boxes'],
                        'min_crop_dim': data_cfg['min_crop_dim'],
                        'augment': data_cfg['augment'],
                        'augment_strong': data_cfg['augment_strong'],
                        'reduce': data_cfg['reduce'], 'rotate_deg': data_cfg['rotate_deg'],
                        'obj_quantiles': obj_q,
                        'eval': scores}
                torch.save(ckpt, run / f'detector_it{it:06d}.pth')
                # Selected on `val` where there is one, `train` otherwise -- the same key the
                # end-of-run `best` line reports, so the printed winner and the shipped file are
                # now the same checkpoint instead of two different ones.
                sel = scores.get('val', scores.get('train', {})).get('r50', -float('inf'))
                if sel >= best_score:
                    best_score = sel
                    torch.save(ckpt, run / 'detector.pth')
                history.append({'iteration': it,
                                **{f'{k}_{m}': v[m] for k, v in scores.items()
                                   for m in ('r50', 'r75', 'iou', 'fp', 'mota')}})
                (run / 'metrics.json').write_text(json.dumps(history, indent=1))
                for name, s in scores.items():
                    print(f'   {name:5s} r@.5 {s["r50"]:.4f}  r@.75 {s["r75"]:.4f}  '
                          f'IoU {s["iou"]:.4f}  fp {s["fp"]:.3f}  MOTA {s["mota"]:.3f}',
                          flush=True)
                t0 = time.time()               # evaluation is not part of the s/it readout
    best = max(history, key=lambda h: h.get('val_r50', h['train_r50'])) if history else None
    print(f'done: {it} iterations -> {run}')
    if best:
        print(f'best: it {best["iteration"]} (this is what detector.pth holds)  ' +
              '  '.join(f'{k} {v:.4f}' for k, v in best.items() if k != 'iteration'))


if __name__ == '__main__':
    main()
