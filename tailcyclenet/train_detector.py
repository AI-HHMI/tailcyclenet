"""Train one box predictor on one dataset root or a folder of roots.

Body of `tailcyclenet train-detector` (`tailcyclenet/__main__.py`) and of
`scripts/train_detector.py` (`pixi run python scripts/train_detector.py --config
configs/detector.toml`) -- both call this module's `main()` unchanged.

The recipe lives in the config file; the run folder records the effective config + provenance
before training starts, and the checkpoint carries its own scores.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import toml

from tailcyclenet.checkpoints import _DETECTOR_CONFIG, provenance
from tailcyclenet.dataset import worker_init
from tailcyclenet.detector import (BoxDataset, CohortSampler, YOLOXNano,
                                   box_collate, detector_loss, split_batch, tiled_input_wh)
from tailcyclenet.detector.config import load_detector_config
from tailcyclenet.detector.evaluate import overall, score_dataset
from tailcyclenet.detector.pretrained import load_coco_backbone
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


def input_wh_for_roots(roots, box_source, boxes_by_dataset=None, min_box_px=32,
                       max_px=4 * 416 * 416):
    """One deterministic aspect/scale compromise for multiple detector roots."""
    ratios = []
    for ds in roots:
        values = [w / h for sess in ds.sessions.get('train', []) for cam in sess.cam_names
                  for w, h in [sess.rig.size(cam)]]
        if values:
            ratios.append(float(np.median(values)))
    if not ratios:
        raise ValueError('multi-root input sizing found no training cameras')
    ar = float(np.median(ratios))
    base = (max(64, int(round((416 * 416 * ar) ** 0.5 / 32) * 32)),
            max(64, int(round((416 * 416 / ar) ** 0.5 / 32) * 32)))
    if min_box_px <= 0:
        return base

    by = boxes_by_dataset or {}
    stats = {}
    for ds in roots:
        root_boxes = {ds.name: by[ds.name]} if ds.name in by else {}
        sampled = BoxDataset(ds.root, 'train', input_wh=base, box_source=box_source,
                             boxes_by_dataset=root_boxes, max_frames_per_group=4)
        ix = np.random.default_rng(0).choice(len(sampled), min(300, len(sampled)), replace=False)
        sides = torch.cat([(b[:, 2:] - b[:, :2]).flatten()
                           for b in (sampled.boxes_for(int(i)) for i in ix)])
        sides = sides[torch.isfinite(sides)]
        if sides.numel():
            stats[ds.name] = (float(sides.median()), float(sides.quantile(0.1)))
    if not stats:
        return base
    required = max([1.0, *(min_box_px / med for med, _ in stats.values() if med > 0)])
    cap = (max_px / (base[0] * base[1])) ** 0.5
    scale = min(required, cap)
    capped = required > cap
    for name, (med, p10) in stats.items():
        print(f'{name}: median box {med:.1f}px (p10 {p10:.1f}) at {base[0]}x{base[1]}; '
              f'common x{scale:.2f} -> median {med * scale:.1f}px'
              + (' (cap below min_box_px)' if capped and med * scale < min_box_px else ''))
    if required > 1:
        print(f'multi-root input sizing: common aspect {ar:.3f}, scale x{scale:.2f}'
              + (f' capped by max_input_px={max_px}' if capped else ''))
    return (max(64, int(round(base[0] * scale / 32) * 32)),
            max(64, int(round(base[1] * scale / 32) * 32)))


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

    if kind == 'adamw':
        groups = []
        trainable = [p for p in model.parameters() if p.requires_grad]
        if model_cfg['pretrained']:
            backbone_ids = {id(p) for p in model.backbone.parameters()}
            backbone_params = [p for p in trainable if id(p) in backbone_ids]
            other_params = [p for p in trainable if id(p) not in backbone_ids]
            groups.append({'params': backbone_params, 'lr': lr * BACKBONE_LR_SCALE,
                           'weight_decay': wd})
            groups.append({'params': other_params, 'lr': lr, 'weight_decay': wd})
        else:
            groups.append({'params': trainable, 'lr': lr, 'weight_decay': wd})
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
                'weight_decay': wd,
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


def main(argv: list[str] | None = None):
    """Train the box predictor; exit via SystemExit on bad config.

    Inputs: argv (via argparse): --config, --out, --iters, --device.
    Side effects: writes checkpoints + metrics.json + config/provenance under
                  the run folder; prints progress and eval lines.

    Notes:
    - A tiled checkpoint's `input_wh` is its TILE size, read back from `BoxDataset`.
    - `n_keypoints` is derived from the registry, never configured; a masked
      keypoint still emits a number (conv bias), so the labelled fraction is
      logged (sampled, not exhaustive).
    - THE TRAIN LOADER IS UNCAPPED AND WEIGHTED, NEVER SUBSAMPLED.
      `[data].frames_per_group` is DELETED: every labelled frame is indexed and
      `BoxDataset.default_train_weights` draws view-uniformly WITHIN a cohort, so
      a 57,594-frame tracked group neither freezes to 40 images nor swamps the
      split. `CohortSampler` therefore ALWAYS runs, `ChunkShuffle` is no longer a
      train fallback, `annot_frac` overrides the otherwise-natural cohort share,
      and `alpha`/W1.1/A6/A6c compose on top. The realised mix is PRINTED.
    - No `val/` is the only thing swallowed; config errors fail at construction.
      A COCO-pretrained backbone gets BACKBONE_LR_SCALE x lr (COCO convs, fresh
      GroupNorm net).
    - `detector.pth` is the BEST checkpoint, not the last one (recall peaks
      early on partially-labelled roots); every `detector_it*.pth` is still
      written. Evaluation scores both splits, stores the score beside the
      weights, and selects on `val` where there is one (`train` otherwise);
      the loader does NOT trust `detector_last.pth`.
    """
    ap = argparse.ArgumentParser(prog='tailcyclenet train-detector', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=None,
                    help='defaults to the packaged configs/detector.toml (the shipped recipe, '
                         'unmodified) when omitted -- pass one that overlays it for anything else')
    ap.add_argument('--out', type=Path, default=None, help='override [training].out')
    ap.add_argument('--iters', type=int, default=None, help='override [training].iters')
    ap.add_argument('--device', default=None, help='override [training].device')
    ap.add_argument('--weights-dir', type=Path, default=None,
                    help='cache directory for the COCO-pretrained YOLOX backbone ([model].'
                         'pretrained = "coco"); defaults to $TAILCYCLENET_CACHE_DIR/weights or '
                         '~/.cache/tailcyclenet/weights, auto-fetched from Megvii\'s tagged '
                         '0.1.1rc0 GitHub release if not already cached there. Pass this to use a '
                         'pre-staged directory instead (e.g. on a host with no internet access).')
    args = ap.parse_args(argv)

    config = load_detector_config(args.config or _DETECTOR_CONFIG, out=args.out,
                                  iters=args.iters, device=args.device)
    data_cfg, model_cfg, train_cfg = config['data'], config['model'], config['training']

    torch.manual_seed(train_cfg['seed'])
    np.random.seed(train_cfg['seed'])
    device = train_cfg['device'] if torch.cuda.is_available() else 'cpu'
    run = Path(train_cfg['out'])
    _record_run(run, config)

    all_roots = load_datasets(data_cfg['path'])
    roots = load_datasets(data_cfg['path'], split='train')
    multi = len(roots) > 1
    if multi and data_cfg['keypoints']:
        raise SystemExit(f'{data_cfg["path"]}: multi-root detector training is box-only; '
                         '[data].keypoints = true is explicitly refused when more than one '
                         'training root is given.')
    root_names = [ds.name for ds in roots]
    unknown_box_roots = set(data_cfg['boxes_by_dataset']) - {ds.name for ds in all_roots}
    if unknown_box_roots:
        raise SystemExit(f'[data].boxes_by_dataset names unknown dataset roots '
                         f'{sorted(unknown_box_roots)}.')
    probe_sess = roots[0].all_sessions()[0]
    if data_cfg['input_wh']:
        wh = tuple(data_cfg['input_wh'])
    elif multi:
        wh = input_wh_for_roots(roots, data_cfg['boxes'], data_cfg['boxes_by_dataset'],
                                data_cfg['min_box_px'], data_cfg['max_input_px'])
    else:
        wh = input_wh_for(data_cfg['path'], roots[0], data_cfg['boxes'],
                          data_cfg['min_box_px'], data_cfg['max_input_px'])
    print(f'input {wh[0]}x{wh[1]}  training roots={root_names}')

    tiling = dict(tile_wh=data_cfg['tile_wh'], tile_scale=data_cfg['tile_scale'],
                  tile_bg_per_frame=data_cfg['tile_bg_per_frame'])
    train = BoxDataset(data_cfg['path'], 'train', input_wh=wh,
                       box_source=data_cfg['boxes'], boxes_by_dataset=data_cfg['boxes_by_dataset'],
                       min_crop_dim=data_cfg['min_crop_dim'], augment=data_cfg['augment'],
                       reduce=data_cfg['reduce'], max_frames_per_group=0,
                       keypoints=data_cfg['keypoints'],
                       hflip=0.0 if not data_cfg['hflip'] else None,
                       rotate_deg=data_cfg['rotate_deg'], strong=data_cfg['augment_strong'],
                       seed=train_cfg['seed'], **tiling)
    wh = train.input_wh
    if data_cfg['tile_wh']:
        ext = train._tile_extent()
        print(f'tiling: {data_cfg["tile_wh"][0]}x{data_cfg["tile_wh"][1]} input px at scale '
              f'{data_cfg["tile_scale"]:g} = {ext[0]:.0f}x{ext[1]:.0f} SOURCE px, '
              f'{data_cfg["tile_bg_per_frame"]} background tile(s)/frame')
        print(f'  DEPLOYMENT INPUT is the whole frame at this scale, NOT the tile size: '
              f'{tiled_input_wh(probe_sess.rig.size(probe_sess.cam_names[0]), data_cfg["tile_scale"])}')
    print(f'train: {len(train)} views across {len(train.root_names)} root(s)')
    effective_sources = {name: data_cfg['boxes_by_dataset'].get(name, data_cfg['boxes'])
                         for name in train.root_names}
    print(f'box sources={effective_sources}; box-only instance fallback: '
          f'{len(train.box_only_fallback_sessions)} session(s)')
    n_kpts = len(all_roots[0].names) if data_cfg['keypoints'] else 0
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
        names = all_roots[0].names
        thin = [f'{names[i]} {frac[i]:.2f}' for i in np.argsort(frac)[:5]]
        print(f'  labelled fraction per keypoint over {seen} sampled instances: '
              f'min {frac.min():.3f}  median {np.median(frac):.3f}  max {frac.max():.3f}')
        print(f'  thinnest: {", ".join(thin)}', flush=True)
    combined_w = train.train_weights(data_cfg['annot_frac'], data_cfg['alpha'],
                                     data_cfg['balance_datasets'])
    sampler = CohortSampler(combined_w, seed=train_cfg['seed'])
    was = train.cohort_mix()
    now = train.cohort_mix(combined_w)
    root_mix = train.root_mix(combined_w)
    af = ('natural' if data_cfg['annot_frac'] is None
          else f'annot_frac={data_cfg["annot_frac"]:g}')
    print(f'sampling: balance_datasets={data_cfg["balance_datasets"]}; view-uniform within '
          f'cohort, cohort share {af}; roots ' + '  '.join(
              f'{k} {v:.3f}' for k, v in root_mix.items()))
    if multi:
        print('  cohort mix by root: ' + '  '.join(
            f'{name} ' + ', '.join(f'{cohort}={share:.3f}' for cohort, share in cohorts.items())
            for name, cohorts in train.root_cohort_mix(combined_w).items()))
    print('  cohort mix (frame-uniform -> realised) ' + '  '.join(
        f'{k} {was[k]:.3f}->{now[k]:.3f}' for k in sorted(now)))
    if data_cfg['annot_frac'] is not None and len(now) < 2:
        held = ', '.join(train.root_names)
        print(f'  annot_frac is INERT here: {held} train holds one cohort')
    if data_cfg['alpha'] is not None:
        print(f'alpha={data_cfg["alpha"]:g}: group-size draw exponent applied on top of the '
              'view/cohort base (its 0/1 landmarks shift by one -- see alpha_weights)')
    loader = torch.utils.data.DataLoader(
        train, batch_size=train_cfg['batch_size'], sampler=sampler,
        num_workers=train_cfg['num_workers'], collate_fn=box_collate, drop_last=True,
        persistent_workers=train_cfg['num_workers'] > 0, worker_init_fn=worker_init)
    val_roots = [ds for ds in all_roots if ds.name in root_names and (ds.root / 'val').is_dir()]
    val = None
    if not val_roots:
        print(f'val:   none (no val/ split under {data_cfg["path"]}); selection falls back to train')
    else:
        val = BoxDataset(data_cfg['path'], 'val', input_wh=wh,
                         box_source=data_cfg['boxes'], boxes_by_dataset=data_cfg['boxes_by_dataset'],
                         min_crop_dim=data_cfg['min_crop_dim'], reduce=data_cfg['reduce'],
                         max_frames_per_group=data_cfg['val_frames_per_group'],
                         keypoints=data_cfg['keypoints'], seed=train_cfg['seed'],
                         include_roots=root_names, **tiling)
        missing_val = sorted(set(root_names) - set(val.root_names))
        print(f'val:   {len(val)} views across {val.root_names}; '
              f'no val split: {missing_val or "none"}')
    eval_roots = len(train.root_names) + (0 if val is None else len(val.root_names))
    print(f'eval work/pass: up to {train_cfg["eval_batches"]} batches x {eval_roots} '
          f'root/split evaluations')

    model = YOLOXNano(n_keypoints=n_kpts,
                      version=model_cfg['yolox'],
                      bottleneck_expansion=model_cfg['bottleneck_expansion'],
                      p2=model_cfg['p2'],
                      pretrained=model_cfg['pretrained'],
                      shared_head=train_cfg['shared_head'],
                      fpn_upsample=train_cfg['fpn_upsample']).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'YOLOX [{model_cfg["yolox"]}]: {n / 1e6:.2f}M params'
          f'  (bottleneck_expansion={model_cfg["bottleneck_expansion"]:g})')
    if model_cfg['pretrained'] == 'coco':
        n_loaded, n_total = load_coco_backbone(model, model_cfg['yolox'], weights_dir=args.weights_dir)
        print(f'  loaded COCO backbone: {n_loaded}/{n_total} conv tensors', flush=True)

    opt, sched = build_detector_optimizer(model, train_cfg, model_cfg)
    if model_cfg['pretrained'] and train_cfg['optimizer'] == 'adamw':
        print(f'  differential LR: backbone {train_cfg["lr"] * BACKBONE_LR_SCALE:g}  '
              f'neck/head {train_cfg["lr"]:g}', flush=True)

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
            x, gt, gt_kpts, gt_negative = split_batch(batch)
            x, gt = x.to(device), gt.to(device)
            gt_kpts = None if gt_kpts is None else gt_kpts.to(device)
            gt_negative = None if gt_negative is None else gt_negative.to(device)
            if gt_negative is not None and bool(gt_negative.all()):
                gt_negative = None
            out = model(x)
            obj, boxes, kpt = out[0], out[1], out[2]
            anchors = model.anchor_points(x.shape[-2], x.shape[-1], device)
            loss, parts = detector_loss(obj, boxes, anchors, gt, kpts=kpt, gt_kpts=gt_kpts,
                                        kpt_weight=train_cfg['kpt_weight'],
                                        kpt_score_weight=train_cfg['kpt_score_weight'],
                                        negative_supervision=gt_negative,
                                        iou_aware=train_cfg['iou_aware_obj'],
                                        iou_aware_warmup=train_cfg['iou_aware_warmup'], it=it,
                                        max_pos_per_gt=train_cfg['max_pos_per_gt'] or None,
                                        box_weight=train_cfg['box_weight'],
                                        assignment=train_cfg['assignment'],
                                        box_loss_fn=train_cfg['box_loss'],
                                        tal_topk=train_cfg['tal_topk'],
                                        tal_alpha=train_cfg['tal_alpha'],
                                        tal_beta=train_cfg['tal_beta'])
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

            if it % 50 == 0:
                kp = (f'  kpt {parts["kpt"]:6.3f}  kscore {parts["kpt_score"]:5.3f}'
                      if 'kpt' in parts else '')
                kp += (f'  negf {parts["negative_frames"]:5.3f}'
                       if 'negative_frames' in parts else '')
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
                selection_kind = ('macro' if data_cfg['balance_datasets'] else 'pooled')
                for split_name, ds in (('train', train), ('val', val)):
                    if ds is None:
                        continue
                    split_obj_scores, root_rows, root_scores = [], {}, {}
                    for root_id, root_name in enumerate(ds.root_names):
                        subset = (np.flatnonzero(np.asarray(ds.root_ids) == root_id)
                                  if multi else None)
                        root_obj_scores = []
                        rows = score_dataset(
                            model, ds, device, batch_size=train_cfg['batch_size'],
                            batches=train_cfg['eval_batches'], num_workers=2,
                            out_scores=root_obj_scores, iou_thresh=train_cfg['nms_iou_thresh'],
                            center_dist_thresh=train_cfg['nms_center_dist_thresh'],
                            subset_indices=subset, root_qualified_keys=multi)
                        root_rows[root_name] = rows
                        root_scores[root_name] = overall(rows)
                        split_obj_scores.extend(root_obj_scores)
                    obj_scores = split_obj_scores
                    if not multi:
                        scores[split_name] = root_scores[ds.root_names[0]]
                        continue
                    pooled_rows = {key: row for rows in root_rows.values()
                                   for key, row in rows.items()}
                    pooled = overall(pooled_rows)
                    macro = {'n_gt': sum(v['n_gt'] for v in root_scores.values()),
                             'fp_ignored': sum(v['fp_ignored'] for v in root_scores.values())}
                    for metric in ('r50', 'r75', 'iou', 'fp', 'mota', 'fp_dup', 'fp_none', 'miss'):
                        vals = [v[metric] for v in root_scores.values()]
                        macro[metric] = float(np.mean(vals)) if vals else float('nan')
                    scores[f'{split_name}_macro'] = macro
                    scores[f'{split_name}_pooled'] = pooled
                    for root_name, value in root_scores.items():
                        scores[f'{split_name}_root_{root_name}'] = value
                    scores[split_name] = macro if selection_kind == 'macro' else pooled
                if hasattr(opt, 'train'):
                    opt.train()
                obj_q = {}
                if obj_scores:
                    a = np.concatenate(obj_scores)
                    obj_q = {f'q{int(q * 100):02d}': float(np.quantile(a, q))
                             for q in (0.01, 0.10, 0.50, 0.90)}
                    print(f'   objectness q01 {obj_q["q01"]:.4f}  q10 {obj_q["q10"]:.4f}  '
                          f'q50 {obj_q["q50"]:.4f}  q90 {obj_q["q90"]:.4f}', flush=True)
                select_split = 'val' if val is not None else 'train'
                selection_metric = (f'{select_split}_{selection_kind}_r50' if multi
                                    else f'{select_split}_r50')
                selected_score = scores[select_split]['r50']
                ckpt = {'iteration': it, 'model_state': model.state_dict(), 'config': config,
                        'input_wh': wh, 'n_keypoints': n_kpts,
                        'norm': 'gn',
                        'yolox_version': model_cfg['yolox'],
                        'optimizer_kind': train_cfg['optimizer'],
                        'bottleneck_expansion': model_cfg['bottleneck_expansion'],
                        'pretrained': model_cfg['pretrained'],
                        'assignment': train_cfg['assignment'],
                        'box_loss': train_cfg['box_loss'],
                        'tal_topk': train_cfg['tal_topk'],
                        'tal_alpha': train_cfg['tal_alpha'],
                        'tal_beta': train_cfg['tal_beta'],
                        'shared_head': train_cfg['shared_head'],
                        'fpn_upsample': train_cfg['fpn_upsample'],
                        'p2': model_cfg['p2'],
                        'seed': train_cfg['seed'],
                        'tile_wh': data_cfg['tile_wh'], 'tile_scale': data_cfg['tile_scale'],
                        'dataset': train.root_names[0] if len(train.root_names) == 1 else '',
                        'datasets': list(train.root_names), 'box_source': data_cfg['boxes'],
                        'box_sources': effective_sources,
                        'balance_datasets': data_cfg['balance_datasets'],
                        'root_mix': root_mix, 'selection_metric': selection_metric,
                        'selection_score': selected_score,
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
                sel = selected_score
                if sel >= best_score:
                    best_score = sel
                    torch.save(ckpt, run / 'detector.pth')
                history.append({'iteration': it, 'selection_metric': selection_metric,
                                'selection_score': selected_score,
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
              '  '.join(f'{k} {v:.4f}' for k, v in best.items()
                        if k not in ('iteration', 'selection_metric')
                        and isinstance(v, (int, float))))


if __name__ == '__main__':
    main()
