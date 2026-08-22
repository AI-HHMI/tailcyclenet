#!/usr/bin/env python
"""Finetune a posetail tracker into a pose estimator.

    pixi run python scripts/train.py --config configs/3d.toml [--devices N]
"""
# E402: the tailcyclenet imports below must follow this file's sys.path.insert.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# file_system sharing avoids the default's fd-per-tensor lifetime; harmless, keep it.
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import distributed as dist_utils
from tailcyclenet.checkpoints import (check_image_size, load_config, prior_provenance,
                                      resolve_checkpoint, save_checkpoint, save_run_meta,
                                      warm_start)
from tailcyclenet.dataset import (LoaderConfig, PoseDataset, StepSampler, pose_collate,
                                  worker_init)
from tailcyclenet.format import Registry
from tailcyclenet.losses import PoseLoss
from tailcyclenet.model import build_model
from tailcyclenet.unfreeze import (apply_norms_extension, apply_staged_unfreeze,
                                   replay_staged_unfreeze, trainable_encoder_params)


def build_optimizer(model, fresh: set[str], cfg: dict):
    """The run's optimizer, selected by `[training.optimizer].optimizer`; an absent key is "muon".

    `"schedulefree"` is the AdamW-SF recipe a resume needs. Fresh params (identity table,
    no-query tokens, fusion gate) get `kpt_lr`; a trainable encoder gets a lower one.
    """
    kind = str(cfg.get('optimizer', 'muon'))
    if kind == 'muon':
        from tailcyclenet.optim import build_muon
        return build_muon(model, fresh, cfg)
    if kind != 'schedulefree':
        raise SystemExit(f'[training.optimizer].optimizer = {kind!r} is not one of '
                         '"muon" | "schedulefree".')
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


def init_wandb(config: dict, run: Path, disabled: bool = False):
    """wandb from the existing `[wandb]` block, or None. The run folder stays `--out`."""
    cfg = config.get('wandb')
    if disabled or not cfg:
        return None
    try:
        import wandb
    except ImportError:
        print('wandb: not installed, skipping')
        return None
    Path(cfg.get('path', '.')).mkdir(parents=True, exist_ok=True)
    # Timestamp + run folder name: the folder alone is not unique, and resumes reuse it.
    name = f'{time.strftime("%Y%m%d_%H%M%S")}-{run.name}'
    wandb.init(project=cfg.get('project_name', 'tailcyclenet'), dir=cfg.get('path'),
               mode=cfg.get('mode', 'offline'), name=name, config=config)
    print(f'wandb: {cfg.get("mode", "offline")} | {wandb.run.name}')
    return wandb


def log(wb, values: dict, step: int) -> None:
    if wb is not None:
        wb.log(values, step=step)


def _gpu_peak_gb(device, reset: bool = False) -> float:
    """Peak allocator bytes since the last reset, in GB (0.0 on CPU); max, not current."""
    # `torch.device` since the loop takes `fabric.device`, `str` from the tests and older callers.
    if str(device).split(':')[0] != 'cuda' or not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return peak


def _brief(m: dict) -> str:
    keys = ('mpjpe', 'mte', 'delta_x_avg', 'survival_rate', 'n_nonfinite')
    return ' '.join(f'{k}={m[k]:.4g}' for k in keys if k in m)


def base_registry(run: Path, checkpoint_path):
    """An existing keypoint registry to append to, or None: run folder first, then the warm-start
    checkpoint's folder. Renumbering would point embedding rows at different body parts."""
    candidates = [run / 'keypoint_registry.toml']
    if checkpoint_path:
        candidates.append(Path(checkpoint_path).parent / 'keypoint_registry.toml')
    for p in candidates:
        if p.exists():
            print(f'keypoint registry: appending to {p}')
            return Registry.load(p)
    return None


def timed(loader, acc):
    """Yield from `loader`, accumulating seconds blocked on `next()` into `acc[0]` -- the loader
    starvation canary. Wrapping the iterator stays correct across `continue` paths."""
    it = iter(loader)
    while True:
        t = time.time()
        try:
            batch = next(it)
        except StopIteration:
            return
        acc[0] += time.time() - t
        yield batch


def to_device(batch, device):
    views = [v.to(device, non_blocking=True) for v in batch.views]
    cgroup = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
              for c in batch.cgroup]
    return views, cgroup


def _tune_smoothness(loss_fn, T, stride=1):
    """Clamp each smoothness loss's order below the window length and undo the stride, in place.

    `torch.diff` is an undivided difference, so on a stride-s lattice the k-th difference grows
    like s^k; dividing restores per-frame units. The hinge itself is scale-invariant, so striding
    genuinely loosens the term -- a real change of meaning, not a bug.
    """
    for name in ('smoothness_loss_3d', 'smoothness_loss_2d'):
        sl = getattr(loss_fn, name, None)      # a stub loss in the tests has neither
        if sl is None:
            continue
        # Latch the configured values on first use: re-derived per batch, so idempotent.
        if not hasattr(sl, '_configured_order'):
            sl._configured_order, sl._configured_weight = sl.order, sl.weight
        sl.order = min(sl._configured_order, max(1, T - 1))
        sl.weight = sl._configured_weight / float(stride) ** sl.order


def run_batch(model, loss_fn, batch, device, raw=None):
    """One forward + loss; a non-finite sub-loss returns NaN, never raises.

    `model` is the DDP-wrapped forward handle; `raw` the unwrapped module the loss reads.
    posetail raises on a poisoned sub-loss; this converts it to the NaN the loop's skip path
    already handles. Watch `skipped`: a rising fraction means the run is dead, not intermittent.
    """
    views, cgroup = to_device(batch, device)
    mode = batch.sample_info['mode']
    _tune_smoothness(loss_fn, int(views[0].shape[1]), batch.sample_info.get('stride', 1))
    box = getattr(batch, 'box_prompt', None)
    out = model(views, batch.kpt_ids.to(device), cgroup, mode=mode,
                kpt_prior=batch.kpt_prior.to(device), prompt_time=batch.prompt_t.to(device),
                box_prompt=None if box is None else box.to(device))
    coords_true = batch.coords.to(device)
    # `batch.vis_2d` is `vis_2d_true` (PoseLoss's own 2D term) in 2D and `vis_true_cams`
    # (posetail's per-camera term) in 3D -- they must never mix.
    vis_true = vis_true_cams = vis_2d_true = None
    if mode == '2d':
        vis_2d_true = None if batch.vis_2d is None else batch.vis_2d.to(device)
    else:
        vis_true = None if batch.vis is None else batch.vis.to(device)
        vis_true_cams = None if batch.vis_2d is None else batch.vis_2d.to(device)
    try:
        loss = loss_fn(
            model if raw is None else raw, out, coords_true=coords_true,
            vis_true=vis_true, vis_true_cams=vis_true_cams, vis_2d_true=vis_2d_true,
            cgroup=cgroup, p2d=None if batch.p2d is None else batch.p2d.to(device), device=device)
    except ValueError as e:
        if 'non-finite' not in str(e):
            raise
        print(f'  non-finite loss -> skipping step: {e}', flush=True)
        loss = torch.tensor(float('nan'), device=device)
    return loss, out


@torch.no_grad()
def evaluate(model, batches, optimizer, device, fabric=None):
    """Val metrics on a FIXED set of windows, both regimes from one pass -> (prior_free, self).

    Batches are this rank's shard; the per-window dicts are gathered before reducing. `model` is
    the unwrapped module (no DDP hooks in a no-grad forward). Prior-free is the honest number
    (GT-derived priors are gated off); the self-prompted pass re-queries at the model's own
    frame-0 prediction and is the deployable regime. `optimizer.eval()` swaps in the averaged
    iterate; the `finally` puts `y` back.
    """
    from posetail.posetail.eval_metrics import get_eval_metrics, get_vis_true

    from tailcyclenet.infer import self_prompt
    from tailcyclenet.model import share_scene

    model.eval()
    if hasattr(optimizer, 'eval'):
        optimizer.eval()
    free, prompted = [], []
    try:
        for batch in batches:
            views, cgroup = to_device(batch, device)
            kpt_ids, coords = batch.kpt_ids.to(device), batch.coords.to(device)
            mode = batch.sample_info['mode']
            vis_true = (batch.vis.to(device) if batch.vis is not None else get_vis_true(coords))

            def score(out):
                m = get_eval_metrics(vis_pred=out['vis_pred'], vis_true=vis_true,
                                     coords_pred=out['coords_pred'], coords_true=coords, prefix='')
                return {k: float(v) for k, v in m.items() if np.ndim(v) == 0}

            with share_scene(model):
                bx = getattr(batch, 'box_prompt', None)
                bx = None if bx is None else bx.to(device)
                out = model(views, kpt_ids, cgroup, mode=mode, kpt_prior=None, prompt_time=None,
                            box_prompt=bx)
                free.append(score(out))
                prompted.append(score(self_prompt(model, views, kpt_ids, cgroup, mode, out,
                                                  box_prompt=bx)))
    finally:
        model.train()
        if hasattr(optimizer, 'train'):
            optimizer.train()
    return (_reduce(dist_utils.gather_metrics(fabric, free)),
            _reduce(dist_utils.gather_metrics(fabric, prompted)))


def _reduce(mets):
    """nanmean over windows, with the non-finite count alongside."""
    if not mets:
        return {}
    skipped = sum(not all(np.isfinite(v) for v in m.values()) for m in mets)
    out = {k: float(np.nanmean([m[k] for m in mets])) for k in mets[0]}
    out['n_batches'] = len(mets)
    out['n_nonfinite'] = skipped
    return out


def launch(args):
    """The Fabric, launched. Everything after this call runs once PER RANK.

    `--devices 1` is strategy `auto` (no wrapper); above it, `ddp_find_unused_parameters_true`
    because 2D and 3D items genuinely drive different parameters. Fabric re-executes this script
    per rank, so everything above `launch()` runs on every rank -- keep it video-free (forked
    workers deadlock on an open container).
    """
    from lightning.fabric import Fabric

    devices = int(args.devices)
    if args.precision == '16-mixed':
        raise SystemExit(
            '--precision 16-mixed needs a GradScaler applied to the optimizer, and this loop steps\n'
            'its own optimizer rather than handing it to Fabric (PoseDualOptimizer is not an\n'
            'Optimizer subclass, and at 32-true/bf16-mixed a Fabric optimizer wrapper adds\n'
            'nothing). Use --precision 32-true, or bf16-mixed.')
    if devices != 1 and args.device != 'cuda:0':
        raise SystemExit(
            f'--device {args.device} names ONE gpu and --devices {args.devices} asks for several. '
            'At --devices != 1 the launcher places every rank, so --device would be silently '
            'ignored -- pick one (CUDA_VISIBLE_DEVICES selects WHICH gpus).')
    strategy = args.strategy or ('auto' if devices == 1 else 'ddp_find_unused_parameters_true')
    # `--device cpu` still means CPU, even when CUDA is available.
    if not torch.cuda.is_available() or (devices == 1 and str(args.device).startswith('cpu')):
        accelerator, dev_arg = 'cpu', 1
    elif devices == 1:
        # `--device cuda:2` still selects the gpu, as it always did.
        accelerator = 'gpu'
        dev_arg = [int(str(args.device).rsplit(':', 1)[-1])] if ':' in str(args.device) else 1
    else:
        accelerator, dev_arg = 'gpu', devices
    fabric = Fabric(accelerator=accelerator, devices=dev_arg, strategy=strategy,
                    precision=args.precision)
    fabric.launch()
    if not fabric.is_global_zero:
        # Tag, don't silence: the setup phase's prints are the evidence when one rank disagrees.
        sys.stdout = dist_utils.RankPrefix(sys.stdout, fabric.global_rank)
    # Reader caches/workers are per process, so the RAM ceiling divides by the world size; set
    # before any dataset exists (the derivation must not probe video).
    os.environ['TAILCYCLENET_LOCAL_WORLD_SIZE'] = str(fabric.world_size)
    if fabric.world_size > 1:
        fabric.print(f'distributed: {fabric.world_size} ranks, strategy {strategy!r}, '
                     f'precision {args.precision}. n_iterations/val_freq/checkpoint_freq/'
                     f'print_freq and the unfreeze iteration are TOTALS across ranks; the '
                     f'learning rate is scaled by sqrt({fabric.world_size}) = '
                     f'{fabric.world_size ** 0.5:.3f}')
    if args.precision == 'bf16-mixed':
        fabric.print('--precision bf16-mixed is UNMEASURED on this repo\'s roots: every number in '
                     'every number on record was measured at 32-true.')
    return fabric


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--iters', type=int, default=None, help='override n_iterations')
    ap.add_argument('--out', type=Path, default=None, help='override the run folder')
    ap.add_argument('--data', type=Path, default=None, help='override [data].path')
    ap.add_argument('--device', default='cuda:0',
                    help='which gpu, at --devices 1. Refused with --devices != 1, where placement '
                         'is the launcher\'s job')
    ap.add_argument('--devices', default=1,
                    help='gpus to train on: a count, or -1 for every visible one. Above 1, one '
                         'rank per gpu and DDP averages their gradients, so the effective batch '
                         'is the world size (default: 1)')
    ap.add_argument('--strategy', default=None,
                    help='lightning strategy. Default `auto` at one device -- no wrapper at all, '
                         'so a 1-gpu run is unchanged -- and `ddp_find_unused_parameters_true` '
                         'above it')
    ap.add_argument('--precision', default='32-true',
                    help='32-true (default) or bf16-mixed. 16-mixed is refused by name')
    ap.add_argument('--num-workers', type=int, default=None,
                    help='loader workers PER RANK; the total is this times the world size')
    ap.add_argument('--no-warm-start', action='store_true')
    ap.add_argument('--no-resume', action='store_true',
                    help='start from iteration 0 even if the run folder holds a checkpoint_last')
    ap.add_argument('--no-wandb', action='store_true', help='ignore the [wandb] config block')
    ap.add_argument('--no-checkpoints', action='store_true',
                    help='write no checkpoints at all, including the final one. For probes '
                         '(memory, throughput) whose weights are worthless and whose files are '
                         '~5.6 GB each -- checkpoint_freq alone cannot express this, because the '
                         'last iteration always writes.')
    args = ap.parse_args()
    fabric = launch(args)
    world, is0 = fabric.world_size, fabric.is_global_zero

    config = load_config(args.config)
    check_image_size(config)
    data_cfg, train_cfg = config['data'], config['training']
    if args.data:
        data_cfg['path'] = str(args.data)
    if args.iters:
        train_cfg['n_iterations'] = args.iters
    run = Path(args.out or train_cfg['out'])
    n_iter = int(train_cfg['n_iterations'])
    device = fabric.device
    # AN UNKNOWN [training] KEY IS A TYPO, NOT A COMMENT. `val_frequency` for `val_freq` yields
    # val_freq = 0: no validation, no `checkpoint_best.pth`, and a run that prints fine for 60,000
    # iterations. The sub-blocks are guarded on their own (`[training.optimizer]` by name below,
    # `[training.losses]` by `TotalLoss(**losses)`), so they are allowed through here as blocks.
    known_training = {'n_iterations', 'seed', 'checkpoint_path', 'max_grad_norm', 'checkpoint_freq',
                      'val_freq', 'val_batches', 'print_freq', 'out', 'optimizer', 'losses'}
    unknown_training = set(train_cfg) - known_training - {'freeze_encoder'}
    if unknown_training:
        raise SystemExit(
            f'[training]: unknown key(s) {sorted(unknown_training)}. Nothing reads them, so this '
            f'run would train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(known_training)}')

    torch.manual_seed(int(train_cfg.get('seed', 23)))
    np.random.seed(int(train_cfg.get('seed', 23)))

    # -- data ------------------------------------------------------------------------------
    # The filter is needed: `path` and `num_workers` live in this block and are not
    # LoaderConfig fields.
    known = set(LoaderConfig.__dataclass_fields__) | {'path', 'num_workers'}
    unknown = set(data_cfg) - known
    if unknown:
        raise SystemExit(
            f'[data]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(known)}')
    lc = LoaderConfig(**{k: v for k, v in data_cfg.items()
                         if k in LoaderConfig.__dataclass_fields__})
    # The loader emits a box only for a box model; the mode comes from [model].box_prompt.
    lc = replace(lc, box_prompt=config['model'].get('box_prompt', 'none'))
    # Hoisted: `warm_start` needs the same registry, to prove the checkpoint's keypoint table
    # is a prefix of this run's.
    base_reg = base_registry(run, train_cfg.get('checkpoint_path'))
    train_ds = PoseDataset(data_cfg['path'], 'train', lc, registry_base=base_reg)
    registry = train_ds.registry
    # Every rank must build the same registry: an independent filesystem listing that ordered
    # differently would point embedding rows at different body parts.
    dist_utils.check_registry(fabric, registry.names)
    print(f'train: {len(train_ds)} windows across {len(train_ds.datasets)} dataset(s), '
          f'{registry.n_keypoints} keypoints')
    # The sampling mix, not the window count: an index entry is one (session, group, animal)
    # whatever the group's length.
    print('train: mix ' + '  '.join(f'{k}={v:.1%}' for k, v in train_ds.mix().items()))
    # No `val/` is the only thing swallowed here, tested for rather than caught -- config errors
    # in `PoseDataset.__init__` are meant to fail loud. Val gets its own (smaller) camera count.
    val_ds = None
    # `[data].path` is a dataset root OR a folder of them, so both shapes are checked.
    root = Path(data_cfg['path'])
    has_val = (root / 'val').is_dir() or any(
        (c / 'val').is_dir() for c in root.iterdir() if c.is_dir())
    if not has_val:
        print(f'val:   none (no val/ split under {root})')
    else:
        val_lc = replace(lc, cams_to_sample=lc.val_cams_to_sample)
        val_ds = PoseDataset(data_cfg['path'], 'val', val_lc, registry=registry)
        print(f'val:   {len(val_ds)} windows, {lc.val_cams_to_sample} camera(s) each')

    nw = args.num_workers if args.num_workers is not None else int(data_cfg.get('num_workers', 8))
    # `num_workers` is per rank; the host is shared, so `world * nw` is the real process count.
    if world * nw > (os.cpu_count() or 1):
        fabric.print(f'WARNING: {world} rank(s) x {nw} loader worker(s) = {world * nw} decoding '
                     f'processes on {os.cpu_count()} cores. Lower [data].num_workers or '
                     f'--num-workers; the loader is this repo\'s documented bottleneck and '
                     f'oversubscribing it is not free.')
    # One epoch for the whole run: sampling is with replacement, so iterator resets are pure
    # overhead. `batch_size` is structurally 1 per rank (the collate keeps item 0's camera
    # group); `spawn` because `persistent_workers` forks after `model.to(device)`, and a forked
    # worker inherits a live CUDA context. One generator per rank -- not a DistributedSampler,
    # since an index entry is a poor sampling weight -- and `StepSampler`'s ordinal keeps every
    # rank's step-k item the same cost.
    gen = None
    if world > 1:
        gen = torch.Generator().manual_seed(int(train_cfg.get('seed', 23)) + fabric.global_rank)
    local_iters = dist_utils.per_rank(n_iter, world)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, num_workers=nw, collate_fn=pose_collate,
        sampler=StepSampler(len(train_ds), num_samples=local_iters, generator=gen),
        generator=gen,
        persistent_workers=nw > 0, pin_memory=True, drop_last=True,
        prefetch_factor=4 if nw > 0 else None, worker_init_fn=worker_init,
        multiprocessing_context='spawn' if nw > 0 else None)

    # The same windows every time (val items are seeded by index), spread across the index --
    # the index is ordered dataset -> session -> group, so a prefix is a prefix of the session
    # list and would under-represent later sessions.
    val_freq = int(train_cfg.get('val_freq', 0))
    val_batches, idxs = [], []
    if val_ds is not None and val_freq:
        n_val = min(int(train_cfg.get('val_batches', 20)), len(val_ds))
        # A list, not an ndarray: it is used as a truth value below.
        idxs = [int(i) for i in np.unique(np.linspace(0, len(val_ds) - 1, n_val).round().astype(int))]
        # Decoded in a one-worker child, never in the parent: forking the train workers while the
        # parent holds an open video container deadlocks them in a futex, forever. One worker, no
        # shuffle, in `idxs` order; the shard `idxs[rank::world]` is gathered back in `evaluate`.
        mine = [int(i) for i in idxs[fabric.global_rank::world]]
        val_batches = list(torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_ds, mine), batch_size=1,
            num_workers=1, collate_fn=pose_collate, worker_init_fn=worker_init))
        seen = Counter(val_ds.index[int(i)].session.session_id for i in idxs)
        fabric.print(f'val:   {len(idxs)} fixed window(s) every {val_freq} iterations, over '
                     f'{len(seen)}/{len({it.session.session_id for it in val_ds.index})} '
                     f'session(s)' + (f', {len(mine)} on this rank' if world > 1 else ''))
        for sid, n in sorted(seen.items()):
            fabric.print(f'         {n:3d}  {sid}')

    # -- model -----------------------------------------------------------------------------
    model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    fresh: set[str] = set()
    if not args.no_warm_start and train_cfg.get('checkpoint_path'):
        fresh = warm_start(model, resolve_checkpoint(Path(train_cfg['checkpoint_path'])),
                           base_names=base_reg.names if base_reg else None)
    # The encoder's gradient state is `[model].video_encoder_requires_grad` and nothing else;
    # `[training].freeze_encoder` was removed (it silently overrode the model key).
    if 'freeze_encoder' in train_cfg:
        raise SystemExit(
            '[training].freeze_encoder was removed: it re-froze the encoder AFTER build_model, so '
            'it silently overrode [model].video_encoder_requires_grad and would defeat the staged '
            'unfreeze. Set [model].video_encoder_requires_grad instead -- false (never train the '
            'encoder), true (from step 0), or an ITERATION to unfreeze at -- with '
            '[model].video_encoder_finetune_last_n_layers for how many trailing blocks.')
    enc_grad = config['model'].get('video_encoder_requires_grad', False)
    n_last = config['model'].get('video_encoder_finetune_last_n_layers')
    n_frozen = sum(1 for n, p in model.named_parameters()
                   if n.startswith('scene_encoder.encoder.') and not p.requires_grad)
    if isinstance(enc_grad, bool):
        # `true` unfreezes in the constructor, so the norms extension must be applied here, before
        # `build_optimizer`, to match the int path's parameter set.
        ext = apply_norms_extension(model)
        n_frozen = sum(1 for n, p in model.named_parameters()
                       if n.startswith('scene_encoder.encoder.') and not p.requires_grad)
        if not enc_grad:
            print(f'encoder: frozen for the whole run ({n_frozen} frozen tensors)')
        else:
            print(f'encoder: trainable from step 0, last '
                  f'{n_last if n_last is not None else "all"} block(s)'
                  f'{f" + norms {ext}" if ext else ""} ({n_frozen} frozen tensors)')
    else:
        print(f'encoder: frozen now, last {n_last if n_last is not None else "all"} block(s) '
              f'unfreeze at iteration {int(enc_grad)} ({n_frozen} frozen tensors)')
    model = model.to(device)

    opt_cfg = config['training']['optimizer']
    from tailcyclenet.optim import KNOWN_OPTIMIZER_KEYS
    opt_unknown = set(opt_cfg) - KNOWN_OPTIMIZER_KEYS
    if opt_unknown:
        raise SystemExit(
            f'[training.optimizer]: unknown key(s) {sorted(opt_unknown)}. Nothing reads them, so '
            f'this run would train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(KNOWN_OPTIMIZER_KEYS)}')
    # Read the checkpoint before building the optimizer: `start_it` decides whether the staged
    # unfreeze has fired, which changes the param groups the state is keyed to by position.
    start_it, ck, resumed = 0, None, run / 'checkpoints' / 'checkpoint_last.pth'
    if resumed.exists() and not args.no_resume:
        ck = torch.load(resumed, map_location=device, weights_only=False)
        start_it = int(ck['iteration'])
    # The scaled copy reaches every consumer: `optim.group_lr` re-reads this dict at the staged
    # unfreeze. `config` itself keeps the configured rate, or every resume would re-scale it.
    opt_cfg_scaled = dist_utils.scale_optimizer_cfg(opt_cfg, world)
    if world > 1:
        fabric.print(f'lr: scaled by sqrt({world}) -> learning_rate '
                     f'{opt_cfg_scaled["learning_rate"]:g}'
                     + (f', kpt_lr {opt_cfg_scaled["kpt_lr"]:g}' if 'kpt_lr' in opt_cfg_scaled
                        else '')
                     + ' (the multipliers -- encoder_lr_scale, muon_lr_scale -- are unchanged)')
    opt = build_optimizer(model, fresh, opt_cfg_scaled)
    # The model and loss must agree on `per_camera_cube_scale`; derive it from [model] rather
    # than duplicating the gauge.
    losses = dict(config['training']['losses'])
    losses.setdefault('per_camera_cube_scale',
                      bool(config['model'].get('per_camera_cube_scale', False)))
    # PoseLoss adds only the 2D visibility term; an unset weight (0.0) is bit-identical to it.
    loss_fn = PoseLoss(**losses)

    # Resume: re-running into the same --out is the documented restart path, and it must not
    # destroy what the previous run earned (it would overwrite `checkpoint_last.pth` and reset
    # `saved_mpjpe = inf`). Restore `model_state` (the raw iterate), not `model_state_eval`;
    # the sampler position is deliberately not restored -- sampling is with replacement, so it
    # carries no contract.
    if ck is not None:
        model.load_state_dict(ck['model_state'])
        # Replay the staged unfreeze before loading state: the optimizer was built in the frozen
        # layout, and `load_state_dict` matches groups by position.
        info = replay_staged_unfreeze(model, opt, opt_cfg_scaled, start_it, fresh=fresh)
        if info:
            print(f'resume: replayed the encoder unfreeze (blocks {info["blocks"]}, '
                  f'norms {info["norms"]}, {info["n_params"]:,} params)')
        # Refuse, by name, an optimizer state written by a different optimizer than this config
        # builds -- otherwise a resume under the wrong optimizer dies as a bare KeyError.
        from tailcyclenet.optim import refuse_mismatched_optimizer_state
        refuse_mismatched_optimizer_state(
            opt, ck['optimizer_state'], resumed,
            resolved=str(config['training']['optimizer'].get('optimizer', 'muon')),
            explicit='optimizer' in config['training']['optimizer'])
        opt.load_state_dict(ck['optimizer_state'])
        print(f'resuming {resumed} at iteration {start_it} of {n_iter} '
              '(--no-resume to start over, which OVERWRITES both checkpoints)')
    elif resumed.exists():
        print(f'--no-resume: {resumed} and any checkpoint_best.pth beside it WILL be overwritten')

    # Wrap after the resume replay: DDP registers parameters at wrap time and never re-checks, so
    # an already-unfrozen encoder must be wrapped or it is never all-reduced. `raw` stays the
    # handle for everything that is not the forward.
    raw = model
    wrap = world > 1 or args.precision != '32-true'
    model = fabric.setup_module(raw) if wrap else raw

    # Record the resolved `box_source`: its default moved, and a run folder must say what it
    # trained as.
    config.setdefault('data', {})['box_source'] = train_ds.cfg.box_source
    # Record the resolved optimizer too: an absent key means "muon", and the folder must say so.
    opt_cfg['optimizer'] = str(opt_cfg.get('optimizer', 'muon'))
    # Record the world size in `provenance.toml` (not `[training]`, which would reject it on
    # resume): the world size moves batch and lr together, so a 4-gpu and a 1-gpu number are not
    # one lever apart.
    prior = prior_provenance(run)
    if prior.get('world_size') and int(prior['world_size']) != world:
        fabric.print(
            f'WARNING: this run folder was written by a {prior["world_size"]}-rank run and this '
            f'is a {world}-rank one. The learning rate (sqrt of the world size) and the '
            f'iteration-to-sample mapping both change, so the continued run is not the run it '
            f'continues. Allowed on purpose -- a preempted job often comes back on fewer gpus -- '
            f'but say so when reporting it.')
    if is0:
        save_run_meta(run, config, registry, extra={
            'world_size': world, 'devices': str(args.devices), 'precision': args.precision,
            'lr_effective': float(opt_cfg_scaled['learning_rate']),
            'kpt_lr_effective': float(opt_cfg_scaled.get('kpt_lr',
                                                         opt_cfg_scaled['learning_rate']))})
    fabric.barrier()
    fabric.print(f'run folder: {run.resolve()}')
    wb = init_wandb(config, run, disabled=args.no_wandb or not is0)
    log_path = run / 'log.jsonl'                  # the numbers survive without wandb

    def record(rec):
        # Rank 0 only: N ranks appending the same reduced metrics would corrupt the history.
        if not is0:
            return
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    # The mix is not recoverable from `config.toml` (its shares are realised conditionals), so
    # it goes in the log.
    record({'iter': 0, 'mix': train_ds.mix(), 'n_windows': len(train_ds),
            'n_keypoints': registry.n_keypoints})

    # -- loop ------------------------------------------------------------------------------
    max_grad = float(train_cfg.get('max_grad_norm', 0)) or None
    # Clip only the AdamW-routed params on a Muon run: Muon orthogonalises its own gradients, so
    # their raw norm is not a step size, and the 2D matrices carry most of the global norm -- a
    # shared clip would throttle the fresh heads. Schedule-free is every param, as before.
    def _clip_targets():
        """Recomputed after an unfreeze: a stale list leaves new tensors unclipped/unchecked."""
        trainable = [p for p in raw.parameters() if p.requires_grad]
        return getattr(opt, 'adamw_params', trainable), getattr(opt, 'muon_params', [])

    clip_params, unclipped_params = _clip_targets()
    # The Muon half is unclipped but still checked for finiteness; empty for schedule-free.
    print_freq = int(train_cfg.get('print_freq', 20))
    ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
    # Every frequency is a total across ranks, so each becomes a local step count; the global
    # `it = step * world` keeps schedules aligned (a rank-dependent gate would desynchronise
    # the collectives).
    local_val_freq = dist_utils.per_rank(val_freq, world) if val_freq else 0
    local_ckpt_freq = dist_utils.per_rank(ckpt_freq, world)
    local_print_freq = dist_utils.per_rank(print_freq, world)
    # `best` is written at checkpoint boundaries, so a val must land on one; warn if not.
    if idxs and local_val_freq and local_ckpt_freq % local_val_freq:
        fabric.print(f'WARNING: checkpoint_freq {ckpt_freq} is not a multiple of val_freq '
                     f'{val_freq} (as {local_ckpt_freq} and {local_val_freq} local steps on '
                     f'{world} rank(s)), so no evaluation lands on a checkpoint boundary and '
                     f'checkpoint_best.pth will never be written. Pick freqs that divide.')
    model.train()
    opt.train()
    # `step` is this rank's local count; `it = step * world` is global. On resume the local
    # position is `ceil_div(start_it, world)` -- a fresh run legitimately starts at step 0.
    step = dist_utils.ceil_div(start_it, world)
    it, skipped, t0, running, clipped = step * world, 0, time.time(), [], []
    # `best_mpjpe` is the best val at any step; `saved_mpjpe` is the metric of the file on disk
    # as checkpoint_best.pth (only its write costs a checkpoint). `latest` is the most recent val.
    best_mpjpe, best_iter, saved_mpjpe = float('inf'), start_it, float('inf')
    # On resume `saved_mpjpe` describes a file that still exists; read it back from the log or a
    # good `checkpoint_best.pth` gets replaced by the first val of the resumed run.
    if start_it and log_path.exists():
        prev = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
        marks = [r for r in prev if 'saved_mpjpe' in r]
        if marks:
            saved_mpjpe = float(marks[-1]['saved_mpjpe'])
            best_mpjpe, best_iter = saved_mpjpe, int(marks[-1]['iter'])
            print(f'resuming: checkpoint_best.pth holds mpjpe {saved_mpjpe:.4g} '
                  f'from iteration {best_iter}; it is only replaced by something better')
        else:
            print('resuming: no saved_mpjpe in log.jsonl, so checkpoint_best.pth is unattributed '
                  'and the first val of this run will replace it')
    # The workers spawn on the loader's first `next()`, after this process freed its scratch --
    # hand them the smallest possible parent to inherit.
    ctypes.CDLL('libc.so.6').malloc_trim(0)

    latest = (float('inf'), -1)
    waited, evalled, ckpted = [0.0], [0.0], [0.0]
    while step < local_iters:
        for batch in timed(loader, waited):
            if step >= local_iters:
                break
            # The staged unfreeze: upstream owns the gate; this adds the newly-trainable tensors
            # to the optimizer (a frozen param is in no param group).
            unfroze = apply_staged_unfreeze(raw, opt, opt_cfg_scaled, it, fresh=fresh)
            if unfroze:
                clip_params, unclipped_params = _clip_targets()
                fabric.print(f'  UNFROZE the video encoder at iteration {it}: blocks '
                             f'{unfroze["blocks"]} of {unfroze["n_blocks"]}, norms '
                             f'{unfroze["norms"]}, {unfroze["n_tensors"]} tensors / '
                             f'{unfroze["n_params"]:,} params (+{unfroze["muon_groups"]} muon, '
                             f'+{unfroze["adamw_groups"]} adamw group(s))', flush=True)
                log(wb, {'train/encoder_unfrozen_at': it}, it)
                record({'iter': it, 'unfroze_encoder': unfroze})
                if wrap:
                    # Re-wrap or the encoder is never all-reduced: DDP registered its parameters
                    # at wrap time and never re-checks, so each rank would train its own encoder.
                    # Drop the old wrapper first (its Reducer's destructor removes the hooks);
                    # every rank reaches this on the same global `it`.
                    model = None
                    gc.collect()
                    model = fabric.setup_module(raw)
                    fabric.print(f'  re-wrapped DDP at iteration {it} over '
                                 f'{sum(p.requires_grad for p in raw.parameters())} trainable '
                                 f'tensor(s) -- the encoder is in the reducer from here',
                                 flush=True)
                    record({'iter': it, 'ddp_rewrapped': True})
            loss, _ = run_batch(model, loss_fn, batch, device, raw=raw)
            # A skipped step is a collective decision: one rank skipping its backward leaves the
            # others waiting in the gradient all-reduce, so the flag is reduced first.
            if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(loss))):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                it = step * world
                continue
            opt.zero_grad(set_to_none=True)
            fabric.backward(loss)
            gn = torch.nn.utils.clip_grad_norm_(clip_params, max_grad or 1e9)
            # `gn` is the AdamW half only; the Muon half is checked separately (it is stepped
            # unclipped).
            mgrads = [p.grad for p in unclipped_params if p.grad is not None]
            mgn = torch.nn.utils.get_total_norm(mgrads) if mgrads else gn.new_zeros(())
            # Collective, like the loss check: a step skipped on one rank only is a hang.
            if not dist_utils.all_ranks_finite(
                    fabric, bool(torch.isfinite(gn) and torch.isfinite(mgn))):
                # Counted, not hidden: a run silently skipping half its steps looks trained.
                skipped += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                it = step * world
                continue
            opt.step()
            # The batch mean, not rank 0's own draw: one item per rank is a one-sample statistic.
            running.append(dist_utils.all_ranks_mean(fabric, float(loss.detach())))
            clipped.append(float(max_grad is not None and float(gn) > max_grad))
            step += 1
            it = step * world

            if step % local_print_freq == 0:
                wall = time.time() - t0
                # `s/it` is the training step, with eval and checkpointing taken out.
                elapsed = max(wall - evalled[0] - ckpted[0], 1e-9)
                # Per local step (i.e. per `world` samples) -- the wall-clock quantity, comparable
                # to a 1-gpu run's s/it.
                dt = elapsed / local_print_freq
                # Fraction of the step the GPU spent blocked on pixels -- the loader starvation
                # signal.
                wait_frac = waited[0] / elapsed
                eval_frac = evalled[0] / wall if wall > 0 else 0.0
                # Read here, reset in the log call below, so both show the same peak.
                peak_gb = _gpu_peak_gb(device)
                # `wait`/`eval`/`peak` and the sampled cell are rank 0's.
                fabric.print(f'{it:7d}/{n_iter}  loss {np.mean(running):8.4f}  '
                             f'{dt:5.2f}s/it  wait {wait_frac:4.0%}  eval {eval_frac:4.0%}  '
                             f'peak {peak_gb:5.1f}G  skipped {skipped * world}  '
                             f'[{batch.sample_info["dataset"]}/{batch.sample_info["mode"]}'
                             f'{"/1cam" if batch.sample_info["single_view"] else ""}]', flush=True)
                # `clipped` is the running fraction of steps hitting max_grad_norm;
                # `grad_norm` is the clipped AdamW half, `grad_norm_muon` the unclipped Muon half
                # (a raw pre-orthogonalisation norm, not a step size). 0 for a schedulefree run.
                log(wb, {'train/loss': float(np.mean(running)), 'train/grad_norm': float(gn),
                         'train/grad_norm_muon': float(mgn),
                         # Plain iteration, not wandb step: resumes carry different step offsets.
                         'train/iteration': it,
                         'train/encoder_trainable_params': trainable_encoder_params(raw),
                         # Peak bytes since the last print, reset each window -- whether a recipe
                         # fits is a property of the peak.
                         'train/gpu_peak_gb': _gpu_peak_gb(device, reset=True),
                         'train/clipped': float(np.mean(clipped[-200:])) if clipped else 0.0,
                         'train/sec_per_it': dt, 'train/loader_wait_frac': wait_frac,
                         'train/eval_frac': eval_frac,
                         'train/ckpt_frac': ckpted[0] / wall if wall > 0 else 0.0,
                         'train/skipped_frac': skipped / max(step, 1),
                         'train/world_size': world}, it)
                running, t0 = [], time.time()
                waited[0] = evalled[0] = ckpted[0] = 0.0

            # `idxs`, not `val_batches`: an empty shard must still enter the collective gather.
            if idxs and local_val_freq and step % local_val_freq == 0:
                t_val = time.time()
                m, ms = evaluate(raw, val_batches, opt, device, fabric=fabric)
                evalled[0] += time.time() - t_val
                fabric.print(f'  EVAL  ({time.time() - t_val:.0f}s)  prior-free {_brief(m)}\n'
                             f'        self-prompted {_brief(ms)}', flush=True)
                log(wb, {f'val/{k}': v for k, v in m.items()}, it)
                log(wb, {f'val_self/{k}': v for k, v in ms.items()}, it)
                # Tracked on every val (just a number); the checkpoint is written only at
                # `checkpoint_freq` boundaries.
                if np.isfinite(m.get('mpjpe', np.nan)):
                    latest = (m['mpjpe'], it)
                    if m['mpjpe'] < best_mpjpe:
                        best_mpjpe, best_iter = m['mpjpe'], it
                # Plateaus here break late; the span is a number instead of an eyeball.
                log(wb, {'val/no_new_best_span': it - best_iter}, it)
                if wb is not None:
                    wb.run.summary['best_mpjpe'] = best_mpjpe
                    wb.run.summary['best_iter'] = best_iter
                record({'iter': it, 'val': m, 'val_self': ms,
                        'best_mpjpe': best_mpjpe, 'best_iter': best_iter})

            if step % local_ckpt_freq == 0 or step == local_iters:
                t_ck = time.time()
                # The ranks must still hold one model; this guard runs even under
                # `--no-checkpoints`, and is collective so every rank raises together.
                drift, p = dist_utils.check_ranks_agree(fabric, raw), None
                if not args.no_checkpoints:
                    # `write=is0`, called on every rank: the schedule-free eval/train toggle is
                    # not float-exact, so only rank 0 may skip it.
                    p = save_checkpoint(run, it, raw, opt, config, write=is0)
                    # `best` is decided only at checkpoint boundaries -- the only iterations whose
                    # weights exist -- against the metric of the file on disk, using this
                    # iteration's val. Reduced metrics make the condition identical on every rank.
                    if latest[1] == it and latest[0] < saved_mpjpe:
                        saved_mpjpe = latest[0]
                        save_checkpoint(run, it, raw, opt, config, name='best', write=is0)
                        fabric.print(f'  new best: mpjpe {saved_mpjpe:.4g} -> '
                                     f'checkpoint_best.pth')
                        # The `best` file's metric, in the log: the file does not record it, and a
                        # resume needs it.
                        record({'iter': it, 'saved_mpjpe': saved_mpjpe})
                # The barrier keeps `ckpt_frac` on the same wall time and stops rank 0's write
                # from being raced.
                fabric.barrier()
                ckpted[0] += time.time() - t_ck
                if p is not None or world > 1:
                    fabric.print(f'saved {p} ({time.time() - t_ck:.0f}s'
                                 + (f', rank drift {drift:.1e}' if world > 1 else '') + ')')
    fabric.print(f'done: {it} iterations, {skipped * world} skipped')
    record({'iter': it, 'done': True, 'skipped': skipped * world, 'world_size': world,
            'best_mpjpe': best_mpjpe, 'best_iter': best_iter})
    if wb is not None:
        wb.finish()


if __name__ == '__main__':
    main()
