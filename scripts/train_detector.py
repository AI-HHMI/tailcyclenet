#!/usr/bin/env python
"""Train the box predictor, one detector per dataset.

    pixi run python scripts/train_detector.py --config configs/detector.toml

The recipe lives in the config file; the run folder records the effective config + provenance
before training starts, and the checkpoint carries its own scores.
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
from tailcyclenet.detector import (BoxDataset, ChunkShuffle, CohortSampler,
                                   TEMPORAL_INPUT_CHANNELS, YOLOXNano, box_collate,
                                   detector_loss, split_batch, tiled_input_wh)
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
    """Aspect-matched, then raised until the median animal is `min_box_px` across.

    At stride 32 (the coarsest FPN level) an animal smaller than 32 px exists at only the
    finest level. A floor, never a ceiling; `max_px` caps the other end and the cap is
    reported, not applied quietly.
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


# Any pretrained backbone gets this x lr; see main()'s own docstring note.
BACKBONE_LR_SCALE = 0.1


def build_detector_optimizer(model, train_cfg, model_cfg):
    """Build the detector's optimizer + optional LR scheduler.

    Returns (optimizer, scheduler_or_None). Under 'adamw' the pair is byte-identical to the
    pre-existing AdamW + CosineAnnealingLR path. Under 'muon' the scheduler is None
    (schedule-free replaces it).

    Muon routing: 2D .weight -> Muon (wrapped in ScheduleFree), everything else -> AdamW-SF.
    `torch.optim.Muon` ONLY ACCEPTS 2D TENSORS and raises on 4D conv weights. For a pure CNN
    (tiny/nano/s/cspnext), every conv weight is 4D, so Muon gets NOTHING and the optimizer is
    just AdamW-ScheduleFree (no cosine schedule). For a ViT backbone, ~96% of params are 2D and
    Muon routes them all -- confirmed by `dev/scratch/prototype_muon_detector.py`. The routing
    rule is the same as the pose model's in tailcyclenet/optim.py: 2D .weight tensors that are
    NOT nn.Embedding -> Muon; everything else (4D conv, 1D bias, GroupNorm/LayerNorm affine) ->
    AdamW-SF. The detector has no nn.Embedding layers. Decay is applied by the
    ScheduleFreeWrapper, so the Muon group is decay-free.
    """
    kind = train_cfg['optimizer']
    lr = train_cfg['lr']
    wd = train_cfg['weight_decay']

    def _split_decay(params):
        """Param groups split by weight decay (dim > 1 decays, dim <= 1 does not)."""
        params = list(params)
        if not train_cfg['no_decay_norm_bias']:
            return [{'params': params, 'weight_decay': wd}]
        decay = [p for p in params if p.dim() > 1]
        no_decay = [p for p in params if p.dim() <= 1]
        out = []
        if decay:
            out.append({'params': decay, 'weight_decay': wd})
        if no_decay:
            out.append({'params': no_decay, 'weight_decay': 0.0})
        return out

    if kind == 'adamw':
        groups = []
        trainable = [p for p in model.parameters() if p.requires_grad]
        if model_cfg['pretrained']:
            backbone_ids = {id(p) for p in model.backbone.parameters()}
            backbone_params = [p for p in trainable if id(p) in backbone_ids]
            other_params = [p for p in trainable if id(p) not in backbone_ids]
            for g in _split_decay(backbone_params):
                groups.append({**g, 'lr': lr * BACKBONE_LR_SCALE})
            for g in _split_decay(other_params):
                groups.append({**g, 'lr': lr})
        else:
            for g in _split_decay(trainable):
                groups.append({**g, 'lr': lr})
        opt = torch.optim.AdamW(groups, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg['iters'])
        return opt, sched

    from torch.optim import Muon as TorchMuon
    from schedulefree import AdamWScheduleFree, ScheduleFreeWrapper

    backbone_ids = ({id(p) for p in model.backbone.parameters()}
                    if model_cfg['pretrained'] else set())
    muon_groups, adamw_groups = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = id(p) in backbone_ids
        base_lr = lr * (BACKBONE_LR_SCALE if is_backbone and model_cfg['pretrained'] else 1.0)

        if p.ndim == 2 and name.endswith('.weight'):
            muon_groups.append({
                'params': [p],
                'lr': base_lr * train_cfg['muon_lr_scale'],
                'weight_decay': 0.0,
            })
        else:
            adamw_groups.append({
                'params': [p],
                'lr': base_lr,
                'weight_decay': (wd if p.dim() > 1 or not train_cfg['no_decay_norm_bias']
                                 else 0.0),
            })

    betas = (train_cfg['beta1'], train_cfg['beta2'])
    opt_adam = AdamWScheduleFree(
        [g for g in adamw_groups if g['params']], lr=lr, weight_decay=wd,
        warmup_steps=train_cfg['warmup_steps'], betas=betas)

    if muon_groups:
        base_muon = TorchMuon(
            [g for g in muon_groups if g['params']], lr=lr, weight_decay=0.0,
            momentum=train_cfg['muon_momentum'], adjust_lr_fn='match_rms_adamw')
        opt_muon = ScheduleFreeWrapper(base_muon, momentum=0.9, weight_decay_at_y=wd)
        from posetail.posetail.muon import DualOptimizer
        opt = DualOptimizer(opt_muon, opt_adam)
    else:
        opt = opt_adam

    n_muon = sum(p.numel() for g in muon_groups for p in g['params'])
    n_adamw = sum(p.numel() for g in adamw_groups for p in g['params'])
    print(f'optimizer: muon | {n_muon/1e6:.2f}M Muon-routed (2D), '
          f'{n_adamw/1e6:.2f}M AdamW-SF (4D conv + bias + norm)')
    return opt, None


def _record_run(run: Path, config: dict) -> None:
    """Write the run folder's reproducibility record: the effective config + provenance.

    `None` values are dropped so the recorded file round-trips to the same recipe.
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
    """Train the box predictor; exit via SystemExit on bad config.

    Inputs: argv (via argparse): --config, --out, --iters, --device.
    Side effects: writes checkpoints + metrics.json + config/provenance under
                  the run folder; prints progress and eval lines.

    Notes:
    - D2 is a HALF-RESOLUTION stage: `det_scale` rescales whatever `wh` the
      branches resolved to, rounded to a multiple of 32 and floored at 64 (1.0
      is byte-identical to every checkpoint on record); A2.5 needs input
      divisible by LCM(14, 32) = 224. A tiled checkpoint's `input_wh` is its
      TILE size, read back from `BoxDataset`.
    - `n_keypoints` is derived from the registry, never configured; a masked
      keypoint still emits a number (conv bias), so the labelled fraction is
      logged (sampled, not exhaustive).
    - THE COHORT MIX IS A CONFIGURED NUMBER OR IT IS AN ACCIDENT: `annot_frac`
      names the annotated:tracked ratio (without it, rat-city-combined hands
      95.7% of train views to 37 annotated sessions); None keeps `ChunkShuffle`
      byte-identical to every detector on record. `alpha` reweights within
      `cohort_w` by group size; W1.1/A6/A6c are separate draw shares.
    - No `val/` is the only thing swallowed; config errors fail at construction.
      A2.6 freezes the ViT backbone BEFORE the optimizer is built; a pretrained
      backbone gets BACKBONE_LR_SCALE x lr (COCO conv weights land in a fresh
      GroupNorm net).
    - `detector.pth` is the BEST checkpoint, not the last one (recall peaks
      early on partially-labelled roots); every `detector_it*.pth` is still
      written. Evaluation scores both splits, stores the score beside the
      weights, and selects on `val` where there is one (`train` otherwise);
      the loader does NOT trust `detector_last.pth`.
    """
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

    roots = load_datasets(data_cfg['path'])
    if len(roots) != 1:
        raise SystemExit(f'{data_cfg["path"]}: the detector is trained per dataset; '
                         f'found {len(roots)}')
    probe_sess = roots[0].all_sessions()[0]
    wh = (tuple(data_cfg['input_wh']) if data_cfg['input_wh']
          else input_wh_for(data_cfg['path'], roots[0], data_cfg['boxes'],
                            data_cfg['min_box_px'], data_cfg['max_input_px']))
    if data_cfg['det_scale'] != 1.0:
        _wh0 = wh
        wh = tuple(max(64, int(round(v * data_cfg['det_scale'] / 32)) * 32) for v in wh)
        print(f'det_scale={data_cfg["det_scale"]:g}: input {_wh0[0]}x{_wh0[1]} -> {wh[0]}x{wh[1]}')
    if model_cfg['yolox'].startswith('vit_'):
        _wh0 = wh
        wh = tuple(max(64, int(-(-v // 224)) * 224) for v in wh)
        if wh != _wh0:
            print(f'ViT input rounded to a multiple of 224: {_wh0[0]}x{_wh0[1]} -> '
                  f'{wh[0]}x{wh[1]}')
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
                       scale_jitter=data_cfg['scale_jitter'],
                       aug_switch_off_iter=data_cfg['aug_switch_off_iter'],
                       negative_frac=data_cfg['negative_frac'],
                       negative_crop_frac=data_cfg['negative_crop_frac'],
                       augment_copypaste=data_cfg['augment_copypaste'],
                       copypaste_max=data_cfg['copypaste_max'],
                       hard_event_manifest=data_cfg['hard_event_manifest'],
                       hard_event_frac=data_cfg['hard_event_frac'],
                       seed=train_cfg['seed'], **tiling)
    wh = train.input_wh
    if data_cfg['tile_wh']:
        ext = train._tile_extent()
        print(f'tiling: {data_cfg["tile_wh"][0]}x{data_cfg["tile_wh"][1]} input px at scale '
              f'{data_cfg["tile_scale"]:g} = {ext[0]:.0f}x{ext[1]:.0f} SOURCE px, '
              f'{data_cfg["tile_bg_per_frame"]} background tile(s)/frame')
        print(f'  DEPLOYMENT INPUT is the whole frame at this scale, NOT the tile size: '
              f'{tiled_input_wh(probe_sess.rig.size(probe_sess.cam_names[0]), data_cfg["tile_scale"])}')
    print(f'train: {len(train)} views')
    n_kpts = len(roots[0].names) if data_cfg['keypoints'] else 0
    if data_cfg['keypoints']:
        print(f'keypoint branch: {n_kpts} keypoints, hflip disabled')
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
    cohort_w = train.cohort_weights(data_cfg['annot_frac'])
    alpha_w = train.alpha_weights(data_cfg['alpha'])
    hard_w = train.hard_event_weights(data_cfg['hard_event_frac'])
    if hard_w is not None:
        print(f'hard_event_frac={data_cfg["hard_event_frac"]:g}: '
              f'{int(train.hard_event_mask.sum())} hard-event views / {len(train)} total')
    neg_w = (train.negative_weights(data_cfg['negative_frac'], source='absent')
             if data_cfg['negative_frac'] is not None else None)
    neg_crop_w = (train.negative_weights(data_cfg['negative_crop_frac'], source='crop')
                 if data_cfg['negative_crop_frac'] is not None else None)
    weights = [w for w in (cohort_w, alpha_w, hard_w, neg_w, neg_crop_w)
               if w is not None]
    combined_w = None
    for w in weights:
        combined_w = w if combined_w is None else combined_w * w
    if combined_w is None:
        sampler = ChunkShuffle(len(train), chunk=train.chunk, seed=train_cfg['seed'])
        if data_cfg['annot_frac'] is not None:
            print(f'annot_frac={data_cfg["annot_frac"]:g} is INERT here: '
                  f'{train.ds.name} train holds one cohort '
                  f'({", ".join(train.cohort_mix())}) -- keeping ChunkShuffle')
    else:
        sampler = CohortSampler(combined_w, seed=train_cfg['seed'])
        if cohort_w is not None:
            was = train.cohort_mix()
            now = train.cohort_mix(combined_w)
            print(f'annot_frac={data_cfg["annot_frac"]:g}: cohort mix '
                  + '  '.join(f'{k} {was[k]:.3f}->{now[k]:.3f}' for k in sorted(now)))
        if data_cfg['alpha'] is not None:
            print(f'alpha={data_cfg["alpha"]:g}: group-size draw exponent applied '
                  f'({"composed with annot_frac" if cohort_w is not None else "alone"})')
        if data_cfg['negative_frac'] is not None:
            n_neg = int((train.is_negative & (train.negative_source == 'absent')).sum())
            print(f'negative_frac={data_cfg["negative_frac"]:g}: {n_neg} verified-empty '
                  f'(frame, camera) view(s) in the index, drawn at that share')
        if data_cfg['negative_crop_frac'] is not None:
            n_neg_crop = int((train.is_negative & (train.negative_source == 'crop')).sum())
            print(f'negative_crop_frac={data_cfg["negative_crop_frac"]:g}: {n_neg_crop} '
                  f'crop-level negative view(s) in the index, drawn at that share')
    loader = torch.utils.data.DataLoader(
        train, batch_size=train_cfg['batch_size'],
        sampler=sampler,
        num_workers=train_cfg['num_workers'],
        collate_fn=box_collate, drop_last=True,
        persistent_workers=train_cfg['num_workers'] > 0,
        worker_init_fn=worker_init)
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

    in_channels = TEMPORAL_INPUT_CHANNELS[data_cfg['temporal_input']]
    model = YOLOXNano(n_keypoints=n_kpts, version=model_cfg['yolox'],
                      bottleneck_expansion=model_cfg['bottleneck_expansion'],
                      p2=model_cfg['p2'], in_channels=in_channels,
                      head_depthwise=train_cfg['head_depthwise'],
                      pretrained=model_cfg['pretrained'],
                      shared_head=train_cfg['shared_head'],
                      fpn_upsample=train_cfg['fpn_upsample'],
                      p2_bottomup=train_cfg['p2_bottomup']).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'YOLOX [{model_cfg["yolox"]}]: {n / 1e6:.2f}M params'
          f'  (bottleneck_expansion={model_cfg["bottleneck_expansion"]:g}, '
          f'temporal_input={data_cfg["temporal_input"]!r}, in_channels={in_channels})')
    if model_cfg['pretrained'] == 'coco':
        n_loaded, n_total = load_coco_backbone(model, model_cfg['yolox'])
        print(f'  loaded COCO backbone: {n_loaded}/{n_total} conv tensors', flush=True)
    elif model_cfg['pretrained'] == 'dinov2':
        print('  loaded DINOv2 hub weights into the ViT backbone', flush=True)
    elif model_cfg['pretrained']:
        n_loaded = load_pretrained_backbone(model, model_cfg['pretrained'])
        print(f'  loaded in-domain backbone: {n_loaded} conv tensors from '
              f'{model_cfg["pretrained"]}', flush=True)

    if train_cfg['freeze_backbone'] and hasattr(model.backbone, 'freeze_backbone'):
        model.backbone.freeze_backbone()
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f'freeze_backbone: {n_frozen} frozen, {n_trainable} trainable')

    opt, sched = build_detector_optimizer(model, train_cfg, model_cfg)
    if model_cfg['pretrained'] and train_cfg['optimizer'] == 'adamw':
        print(f'  differential LR: backbone {train_cfg["lr"] * BACKBONE_LR_SCALE:g}  '
              f'neck/head {train_cfg["lr"]:g}', flush=True)
    if train_cfg['no_decay_norm_bias'] and train_cfg['optimizer'] == 'adamw':
        n_no_decay = sum(p.numel() for p in model.parameters()
                         if p.requires_grad and p.dim() <= 1)
        print(f'  no_decay_norm_bias: {n_no_decay} params (dim<=1) excluded from weight decay',
              flush=True)

    history = []
    best_score = -float('inf')
    it, t0, running = 0, time.time(), []
    model.train()
    if hasattr(opt, 'train'):
        opt.train()
    while it < train_cfg['iters']:
        for batch in loader:
            if it >= train_cfg['iters']:
                break
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
                                        box_weight=train_cfg['box_weight'],
                                        assignment=train_cfg['assignment'],
                                        box_loss_fn=train_cfg['box_loss'],
                                        focal_obj=train_cfg['focal_obj'],
                                        focal_gamma=train_cfg['focal_gamma'],
                                        tal_topk=train_cfg['tal_topk'],
                                        tal_alpha=train_cfg['tal_alpha'],
                                        tal_beta=train_cfg['tal_beta'],
                                        tal_soft_prior=train_cfg['tal_soft_prior'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg['optimizer'] == 'muon':
                adamw_groups = (opt.opt_adam.param_groups if hasattr(opt, 'opt_adam')
                                else opt.param_groups)
                adamw_ps = [p for g in adamw_groups for p in g['params'] if p.grad is not None]
                if adamw_ps:
                    torch.nn.utils.clip_grad_norm_(adamw_ps, 10.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            if sched is not None:
                sched.step()
            running.append(float(loss.detach()))
            it += 1
            train.set_iter(it)
            if it % 50 == 0:
                kp = (f'  kpt {parts["kpt"]:6.3f}  kscore {parts["kpt_score"]:5.3f}'
                      if 'kpt' in parts else '')
                kp += f'  cert {parts["certified"]:5.3f}' if 'certified' in parts else ''
                kp += f'  ign {parts["ignored"]:5.3f}' if 'ignored' in parts else ''
                kp += f'  id {parts["ident"]:6.3f}' if 'ident' in parts else ''
                kp += f'  iouT {parts["iou_target"]:5.3f}' if 'iou_target' in parts else ''
                print(f'{it:7d}/{train_cfg["iters"]}  loss {np.mean(running):7.4f}  '
                      f'obj {parts["obj"]:6.3f}  box {parts["box"]:6.3f}{kp}  '
                      f'pos {parts["n_pos"]:4d}  {(time.time() - t0) / 50:5.3f}s/it', flush=True)
                running, t0 = [], time.time()
            if it % train_cfg['eval_every'] == 0 or it == train_cfg['iters']:
                if hasattr(opt, 'eval'):
                    opt.eval()
                scores, obj_scores = {}, []
                for name, ds in (('train', train), ('val', val)):
                    if ds is None:
                        continue
                    obj_scores.clear()
                    scores[name] = overall(score_dataset(
                        model, ds, device, batch_size=train_cfg['batch_size'],
                        batches=train_cfg['eval_batches'], num_workers=2,
                        out_scores=obj_scores, iou_thresh=train_cfg['nms_iou_thresh'],
                        center_dist_thresh=train_cfg['nms_center_dist_thresh']))
                if hasattr(opt, 'train'):
                    opt.train()
                obj_q = {}
                if obj_scores:
                    a = np.concatenate(obj_scores)
                    obj_q = {f'q{int(q * 100):02d}': float(np.quantile(a, q))
                             for q in (0.01, 0.10, 0.50, 0.90)}
                    print(f'   objectness q01 {obj_q["q01"]:.4f}  q10 {obj_q["q10"]:.4f}  '
                          f'q50 {obj_q["q50"]:.4f}  q90 {obj_q["q90"]:.4f}'
                          + ('   <-- NOT saturated: --det-score 0.99 would drop most of these'
                             if obj_q['q50'] < 0.99 else ''), flush=True)
                ckpt = {'iteration': it, 'model_state': model.state_dict(), 'config': config,
                        'input_wh': wh, 'n_keypoints': n_kpts, 'norm': 'gn',
                        'yolox_version': model_cfg['yolox'],
                        'optimizer_kind': train_cfg['optimizer'],
                        'bottleneck_expansion': model_cfg['bottleneck_expansion'],
                        'pretrained': model_cfg['pretrained'],
                        'head_depthwise': train_cfg['head_depthwise'],
                        'assignment': train_cfg['assignment'],
                        'box_loss': train_cfg['box_loss'],
                        'focal_obj': train_cfg['focal_obj'],
                        'focal_gamma': train_cfg['focal_gamma'],
                        'tal_topk': train_cfg['tal_topk'],
                        'tal_alpha': train_cfg['tal_alpha'],
                        'tal_beta': train_cfg['tal_beta'],
                        'tal_soft_prior': train_cfg['tal_soft_prior'],
                        'shared_head': train_cfg['shared_head'],
                        'fpn_upsample': train_cfg['fpn_upsample'],
                        'p2_bottomup': train_cfg['p2_bottomup'],
                        'p2': model_cfg['p2'],
                        'in_channels': in_channels,
                        'temporal_input': data_cfg['temporal_input'],
                        'seed': train_cfg['seed'],
                        'tile_wh': data_cfg['tile_wh'], 'tile_scale': data_cfg['tile_scale'],
                        'use_regions': data_cfg['use_regions'],
                        'ignore_present': data_cfg['ignore_present'],
                        'dataset': train.ds.name, 'box_source': data_cfg['boxes'],
                        'annot_frac': data_cfg['annot_frac'],
                        'weight_decay': train_cfg['weight_decay'],
                        'min_crop_dim': data_cfg['min_crop_dim'],
                        'augment': data_cfg['augment'],
                        'augment_strong': data_cfg['augment_strong'],
                        'reduce': data_cfg['reduce'], 'rotate_deg': data_cfg['rotate_deg'],
                        'obj_quantiles': obj_q,
                        'eval': scores}
                torch.save(ckpt, run / f'detector_it{it:06d}.pth')
                torch.save(ckpt, run / 'detector_last.pth')
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
                t0 = time.time()
    best = max(history, key=lambda h: h.get('val_r50', h['train_r50'])) if history else None
    print(f'done: {it} iterations -> {run}')
    if best:
        print(f'best: it {best["iteration"]} (this is what detector.pth holds)  ' +
              '  '.join(f'{k} {v:.4f}' for k, v in best.items() if k != 'iteration'))


if __name__ == '__main__':
    main()
