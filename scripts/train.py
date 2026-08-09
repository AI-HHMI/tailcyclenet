#!/usr/bin/env python
"""Finetune a posetail tracker into a pose estimator.

    pixi run python scripts/train.py --config configs/w9.toml
    pixi run python scripts/train.py --config configs/w9.toml --iters 200   # smoke test

The config is the source of truth for everything that defines a run; the CLI carries only
overrides and one-offs. The run folder is the output, and it holds the config, the keypoint
registry and the checkpoints -- so eval and inference take only `--run <folder>`.
"""
from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posetail.posetail.losses import TotalLoss

from tailcyclenet.checkpoints import (check_image_size, resolve_checkpoint, save_checkpoint,
                                      save_run_meta, warm_start)
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.format import Registry
from tailcyclenet.model import build_model


def build_optimizer(model, fresh: set[str], cfg: dict):
    """Three parameter groups.

    Freshly initialised params -- the keypoint identity table, the no-query tokens, the rebuilt
    fusion gate -- start from noise and need a much higher rate than weights that arrive from
    ~1M pretraining steps. The video encoder, when unfrozen, needs a much lower one.
    """
    lr = float(cfg['learning_rate'])
    kpt_lr = float(cfg.get('kpt_lr', lr))
    enc_scale = float(cfg.get('encoder_lr_scale', 1.0))
    groups = {'fresh': [], 'encoder': [], 'rest': []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in fresh or name.startswith('query_encoder.kpt_'):
            groups['fresh'].append(p)
        elif name.startswith('scene_encoder.encoder.'):
            groups['encoder'].append(p)
        else:
            groups['rest'].append(p)

    from schedulefree import AdamWScheduleFree
    param_groups = [
        {'params': groups['fresh'], 'lr': kpt_lr},
        {'params': groups['encoder'], 'lr': lr * enc_scale},
        {'params': groups['rest'], 'lr': lr},
    ]
    opt = AdamWScheduleFree(
        [g for g in param_groups if g['params']], lr=lr,
        weight_decay=float(cfg.get('weight_decay', 0.0)),
        warmup_steps=int(cfg.get('warmup_steps', 0)),
        betas=(float(cfg.get('beta1', 0.9)), float(cfg.get('beta2', 0.95))))
    print(f'optimizer: {len(groups["fresh"])} fresh @ {kpt_lr:g}, '
          f'{len(groups["encoder"])} encoder @ {lr * enc_scale:g}, '
          f'{len(groups["rest"])} rest @ {lr:g}')
    return opt


def to_device(batch, device):
    views = [v.to(device, non_blocking=True) for v in batch.views]
    cgroup = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
              for c in batch.cgroup]
    return views, cgroup


def run_batch(model, loss_fn, batch, device):
    views, cgroup = to_device(batch, device)
    mode = batch.sample_info['mode']
    out = model(views, batch.kpt_ids.to(device), cgroup, mode=mode,
                kpt_prior=batch.kpt_prior.to(device), prompt_time=batch.prompt_t.to(device))
    loss = loss_fn(
        model, out, coords_true=batch.coords.to(device),
        vis_true=None if batch.vis is None else batch.vis.to(device),
        vis_true_cams=None if batch.vis_2d is None else batch.vis_2d.to(device),
        cgroup=cgroup, p2d=None if batch.p2d is None else batch.p2d.to(device), device=device)
    return loss, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--iters', type=int, default=None, help='override n_iterations')
    ap.add_argument('--out', type=Path, default=None, help='override the run folder')
    ap.add_argument('--data', type=Path, default=None, help='override [data].path')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--no-warm-start', action='store_true')
    args = ap.parse_args()

    with open(args.config, 'rb') as f:
        config = tomllib.load(f)
    check_image_size(config)
    data_cfg, train_cfg = config['data'], config['training']
    if args.data:
        data_cfg['path'] = str(args.data)
    if args.iters:
        train_cfg['n_iterations'] = args.iters
    run = Path(args.out or train_cfg['out'])
    n_iter = int(train_cfg['n_iterations'])
    device = args.device if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(int(train_cfg.get('seed', 0)))
    np.random.seed(int(train_cfg.get('seed', 0)))

    # -- data ------------------------------------------------------------------------------
    lc = LoaderConfig(**{k: v for k, v in data_cfg.items()
                         if k in LoaderConfig.__dataclass_fields__})
    train_ds = PoseDataset(data_cfg['path'], 'train', lc)
    registry = train_ds.registry
    print(f'train: {len(train_ds)} windows across {len(train_ds.datasets)} dataset(s), '
          f'{registry.n_keypoints} keypoints')
    val_ds = None
    try:
        val_ds = PoseDataset(data_cfg['path'], 'val', lc, registry=registry)
        print(f'val:   {len(val_ds)} windows')
    except (ValueError, KeyError) as e:
        print(f'val:   none ({e})')

    nw = args.num_workers if args.num_workers is not None else int(data_cfg.get('num_workers', 8))
    # batch_size is structurally 1: posetail's collate keeps only item 0's camera group, and the
    # model takes one camera group per batch. This is also why there is no DDP.
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=nw, collate_fn=pose_collate,
        persistent_workers=nw > 0, pin_memory=True, drop_last=True)

    # -- model -----------------------------------------------------------------------------
    model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    fresh: set[str] = set()
    if not args.no_warm_start and train_cfg.get('checkpoint_path'):
        fresh = warm_start(model, resolve_checkpoint(Path(train_cfg['checkpoint_path'])))
    if train_cfg.get('freeze_encoder', True):
        n = 0
        for name, p in model.named_parameters():
            if name.startswith('scene_encoder.encoder.'):
                p.requires_grad_(False)
                n += 1
        print(f'encoder frozen: {n} tensors')
    model = model.to(device)

    opt = build_optimizer(model, fresh, config['training']['optimizer'])
    loss_fn = TotalLoss(**config['training']['losses'])
    save_run_meta(run, config, registry)
    print(f'run folder: {run.resolve()}')

    # -- loop ------------------------------------------------------------------------------
    max_grad = float(train_cfg.get('max_grad_norm', 0)) or None
    print_freq = int(train_cfg.get('print_freq', 20))
    ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
    model.train()
    opt.train()
    it, skipped, t0, running = 0, 0, time.time(), []
    while it < n_iter:
        for batch in loader:
            if it >= n_iter:
                break
            loss, _ = run_batch(model, loss_fn, batch, device)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad or 1e9)
            if not torch.isfinite(gn):
                # A non-finite gradient is counted, not hidden: a run that silently skips half
                # its steps looks like a run that trained.
                skipped += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            opt.step()
            running.append(float(loss))
            it += 1

            if it % print_freq == 0:
                dt = (time.time() - t0) / print_freq
                print(f'{it:7d}/{n_iter}  loss {np.mean(running):8.4f}  '
                      f'{dt:5.2f}s/it  skipped {skipped}  '
                      f'[{batch.sample_info["dataset"]}/{batch.sample_info["mode"]}'
                      f'{"/1cam" if batch.sample_info["single_view"] else ""}]', flush=True)
                running, t0 = [], time.time()
            if it % ckpt_freq == 0 or it == n_iter:
                p = save_checkpoint(run, it, model, opt, config)
                print(f'saved {p}')
    print(f'done: {it} iterations, {skipped} skipped')


if __name__ == '__main__':
    main()
