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
from tailcyclenet.checkpoints import (check_image_size, is_hf_repo_id, load_config,
                                      prior_provenance, resolve_checkpoint, resolve_hf_checkpoint,
                                      save_checkpoint, save_run_meta, warm_start)
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
    """wandb from the existing `[wandb]` block, or None. The run folder stays `--out`.

    The run name carries a timestamp because the folder alone is not unique across resumes.
    """
    cfg = config.get('wandb')
    if disabled or not cfg:
        return None
    try:
        import wandb
    except ImportError:
        print('wandb: not installed, skipping')
        return None
    Path(cfg.get('path', '.')).mkdir(parents=True, exist_ok=True)
    name = f'{time.strftime("%Y%m%d_%H%M%S")}-{run.name}'
    wandb.init(project=cfg.get('project_name', 'tailcyclenet'), dir=cfg.get('path'),
               mode=cfg.get('mode', 'offline'), name=name, config=config)
    print(f'wandb: {cfg.get("mode", "offline")} | {wandb.run.name}')
    return wandb


def log(wb, values: dict, step: int) -> None:
    """Log `values` at `step` if wandb is active."""
    if wb is not None:
        wb.log(values, step=step)


def _gpu_peak_gb(device, reset: bool = False) -> float:
    """Peak allocator bytes since the last reset, in GB (0.0 on CPU); max, not current.

    `device` may be a `torch.device` (the loop passes `fabric.device`) or a `str` (tests and
    older callers), so it is compared as a string.
    """
    if str(device).split(':')[0] != 'cuda' or not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return peak


def _brief(m: dict) -> str:
    """The interesting metric keys of a val dict, space-joined for a one-line print."""
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
    """Move a batch's views and cgroup onto `device`, non-blocking where safe."""
    views = [v.to(device, non_blocking=True) for v in batch.views]
    cgroup = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
              for c in batch.cgroup]
    return views, cgroup


def _tune_smoothness(loss_fn, T, stride=1):
    """Clamp each smoothness loss's order below the window length and undo the stride, in place.

    `torch.diff` is an undivided difference, so on a stride-s lattice the k-th difference grows
    like s^k; dividing restores per-frame units. The hinge itself is scale-invariant, so striding
    genuinely loosens the term -- a real change of meaning, not a bug. A stub loss in the tests
    has neither loss, so both are fetched with `getattr`. The configured order/weight are latched
    on first use (the losses are re-derived per batch, making the call idempotent).
    """
    for name in ('smoothness_loss_3d', 'smoothness_loss_2d'):
        sl = getattr(loss_fn, name, None)
        if sl is None:
            continue
        if not hasattr(sl, '_configured_order'):
            sl._configured_order, sl._configured_weight = sl.order, sl.weight
        sl.order = min(sl._configured_order, max(1, T - 1))
        sl.weight = sl._configured_weight / float(stride) ** sl.order


def run_batch(model, loss_fn, batch, device, raw=None):
    """One forward + loss; a non-finite sub-loss returns NaN, never raises.

    `model` is the DDP-wrapped forward handle; `raw` the unwrapped module the loss reads.
    posetail raises on a poisoned sub-loss; this converts it to the NaN the loop's skip path
    already handles. Watch `skipped`: a rising fraction means the run is dead, not intermittent.
    `batch.vis_2d` is `vis_2d_true` (PoseLoss's own 2D term) in 2D and `vis_true_cams`
    (posetail's per-camera term) in 3D -- the two must never mix.
    """
    views, cgroup = to_device(batch, device)
    mode = batch.sample_info['mode']
    _tune_smoothness(loss_fn, int(views[0].shape[1]), batch.sample_info.get('stride', 1))
    box = getattr(batch, 'box_prompt', None)
    out = model(views, batch.kpt_ids.to(device), cgroup, mode=mode,
                kpt_prior=batch.kpt_prior.to(device), prompt_time=batch.prompt_t.to(device),
                box_prompt=None if box is None else box.to(device))
    coords_true = batch.coords.to(device)
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
                """The scalar eval metrics of one forward, as a plain dict."""
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
    workers deadlock on an open container). CPU is still CPU even when CUDA is available, and
    `--device cuda:2` still selects that gpu at one device. Non-zero ranks TAG their stdout
    rather than silencing it: the setup phase's prints are the evidence when one rank disagrees.
    The RAM ceiling divides by the world size (reader caches and workers are per process) and is
    set before any dataset exists, so its derivation never probes video.
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
    if not torch.cuda.is_available() or (devices == 1 and str(args.device).startswith('cpu')):
        accelerator, dev_arg = 'cpu', 1
    elif devices == 1:
        accelerator = 'gpu'
        dev_arg = [int(str(args.device).rsplit(':', 1)[-1])] if ':' in str(args.device) else 1
    else:
        accelerator, dev_arg = 'gpu', devices
    fabric = Fabric(accelerator=accelerator, devices=dev_arg, strategy=strategy,
                    precision=args.precision)
    fabric.launch()
    if not fabric.is_global_zero:
        sys.stdout = dist_utils.RankPrefix(sys.stdout, fabric.global_rank)
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
    """Finetune a posetail tracker into a pose estimator.

    Inputs: argv (via argparse): --config, --iters, --out, --data, --device,
            --devices, --strategy, --precision, --num-workers, --no-warm-start,
            --no-resume, --no-wandb, --no-checkpoints.
    Side effects: trains; writes checkpoints + log.jsonl under the run folder;
                  launches Fabric (re-executes per rank).

    Notes:
    - Unknown `[training]`/`[data]` keys are refused (a typo would train at the
      defaults and report as the arm it is not); a missing `val/` split is the
      only error swallowed.
    - Every rank builds the same registry; `warm_start` reuses it to prove the
      checkpoint's keypoint table is a prefix of this run's.
    - One epoch for the whole run (sampling is with replacement); `num_workers` and
      the RAM ceiling are per rank. Val windows are decoded in a one-worker child --
      forking the train workers while the parent holds an open video container deadlocks.
    - The checkpoint is read before the optimizer is built (`start_it` decides
      whether the staged unfreeze has fired); resume restores `model_state`, not
      `model_state_eval`.
    - DDP registers parameters at wrap time, so the module is re-wrapped after
      the resume replay and after an unfreeze.
    - The resolved `box_source`, optimizer kind and world size are recorded in
      `provenance.toml`; metrics always go to log.jsonl.
    - Every frequency is a total across ranks; `step` is this rank's local count,
      `it = step * world` global; a skipped step is a collective decision.
      `grad_norm` is the clipped AdamW half only; `saved_mpjpe` is the metric of
      `checkpoint_best.pth` on disk.
    """
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
    known_training = {'n_iterations', 'seed', 'checkpoint_path', 'checkpoint_revision',
                      'max_grad_norm', 'checkpoint_freq', 'val_freq', 'val_batches', 'print_freq',
                      'out', 'optimizer', 'losses'}
    unknown_training = set(train_cfg) - known_training - {'freeze_encoder'}
    if unknown_training:
        raise SystemExit(
            f'[training]: unknown key(s) {sorted(unknown_training)}. Nothing reads them, so this '
            f'run would train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(known_training)}')

    torch.manual_seed(int(train_cfg.get('seed', 23)))
    np.random.seed(int(train_cfg.get('seed', 23)))

    known = set(LoaderConfig.__dataclass_fields__) | {'path', 'num_workers'}
    unknown = set(data_cfg) - known
    if unknown:
        raise SystemExit(
            f'[data]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(known)}')
    lc = LoaderConfig(**{k: v for k, v in data_cfg.items()
                         if k in LoaderConfig.__dataclass_fields__})
    lc = replace(lc, box_prompt=config['model'].get('box_prompt', 'none'))
    base_reg = base_registry(run, train_cfg.get('checkpoint_path'))
    train_ds = PoseDataset(data_cfg['path'], 'train', lc, registry_base=base_reg)
    registry = train_ds.registry
    dist_utils.check_registry(fabric, registry.names)
    print(f'train: {len(train_ds)} windows across {len(train_ds.datasets)} dataset(s), '
          f'{registry.n_keypoints} keypoints')
    print('train: mix ' + '  '.join(f'{k}={v:.1%}' for k, v in train_ds.mix().items()))
    val_ds = None
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
    if world * nw > (os.cpu_count() or 1):
        fabric.print(f'WARNING: {world} rank(s) x {nw} loader worker(s) = {world * nw} decoding '
                     f'processes on {os.cpu_count()} cores. Lower [data].num_workers or '
                     f'--num-workers; the loader is this repo\'s documented bottleneck and '
                     f'oversubscribing it is not free.')
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

    val_freq = int(train_cfg.get('val_freq', 0))
    val_batches, idxs = [], []
    if val_ds is not None and val_freq:
        n_val = min(int(train_cfg.get('val_batches', 20)), len(val_ds))
        idxs = [int(i) for i in np.unique(np.linspace(0, len(val_ds) - 1, n_val).round().astype(int))]
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

    model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    fresh: set[str] = set()
    if not args.no_warm_start and train_cfg.get('checkpoint_path'):
        checkpoint_ref = train_cfg['checkpoint_path']
        if is_hf_repo_id(checkpoint_ref):
            checkpoint = resolve_hf_checkpoint(checkpoint_ref,
                                               revision=train_cfg.get('checkpoint_revision'))
        else:
            checkpoint = resolve_checkpoint(Path(checkpoint_ref))
        fresh = warm_start(model, checkpoint, base_names=base_reg.names if base_reg else None)
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
    start_it, ck, resumed = 0, None, run / 'checkpoints' / 'checkpoint_last.pth'
    if resumed.exists() and not args.no_resume:
        ck = torch.load(resumed, map_location=device, weights_only=False)
        start_it = int(ck['iteration'])
    opt_cfg_scaled = dist_utils.scale_optimizer_cfg(opt_cfg, world)
    if world > 1:
        fabric.print(f'lr: scaled by sqrt({world}) -> learning_rate '
                     f'{opt_cfg_scaled["learning_rate"]:g}'
                     + (f', kpt_lr {opt_cfg_scaled["kpt_lr"]:g}' if 'kpt_lr' in opt_cfg_scaled
                        else '')
                     + ' (the multipliers -- encoder_lr_scale, muon_lr_scale -- are unchanged)')
    opt = build_optimizer(model, fresh, opt_cfg_scaled)
    losses = dict(config['training']['losses'])
    losses.setdefault('per_camera_cube_scale',
                      bool(config['model'].get('per_camera_cube_scale', False)))
    loss_fn = PoseLoss(**losses)

    if ck is not None:
        model.load_state_dict(ck['model_state'])
        info = replay_staged_unfreeze(model, opt, opt_cfg_scaled, start_it, fresh=fresh)
        if info:
            print(f'resume: replayed the encoder unfreeze (blocks {info["blocks"]}, '
                  f'norms {info["norms"]}, {info["n_params"]:,} params)')
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

    raw = model
    wrap = world > 1 or args.precision != '32-true'
    model = fabric.setup_module(raw) if wrap else raw

    config.setdefault('data', {})['box_source'] = train_ds.cfg.box_source
    opt_cfg['optimizer'] = str(opt_cfg.get('optimizer', 'muon'))
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
    log_path = run / 'log.jsonl'

    def record(rec):
        """Append one JSON line to log.jsonl -- rank 0 only, reduced metrics.

        Rank 0 only: N ranks appending the same reduced metrics would corrupt the history.
        """
        if not is0:
            return
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    record({'iter': 0, 'mix': train_ds.mix(), 'n_windows': len(train_ds),
            'n_keypoints': registry.n_keypoints})

    max_grad = float(train_cfg.get('max_grad_norm', 0)) or None

    def _clip_targets():
        """Recomputed after an unfreeze: a stale list leaves new tensors unclipped/unchecked."""
        trainable = [p for p in raw.parameters() if p.requires_grad]
        return getattr(opt, 'adamw_params', trainable), getattr(opt, 'muon_params', [])

    clip_params, unclipped_params = _clip_targets()
    print_freq = int(train_cfg.get('print_freq', 20))
    ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
    local_val_freq = dist_utils.per_rank(val_freq, world) if val_freq else 0
    local_ckpt_freq = dist_utils.per_rank(ckpt_freq, world)
    local_print_freq = dist_utils.per_rank(print_freq, world)
    if idxs and local_val_freq and local_ckpt_freq % local_val_freq:
        fabric.print(f'WARNING: checkpoint_freq {ckpt_freq} is not a multiple of val_freq '
                     f'{val_freq} (as {local_ckpt_freq} and {local_val_freq} local steps on '
                     f'{world} rank(s)), so no evaluation lands on a checkpoint boundary and '
                     f'checkpoint_best.pth will never be written. Pick freqs that divide.')
    model.train()
    opt.train()
    step = dist_utils.ceil_div(start_it, world)
    it, skipped, t0, running, clipped = step * world, 0, time.time(), [], []
    best_mpjpe, best_iter, saved_mpjpe = float('inf'), start_it, float('inf')
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
    ctypes.CDLL('libc.so.6').malloc_trim(0)

    latest = (float('inf'), -1)
    waited, evalled, ckpted = [0.0], [0.0], [0.0]
    while step < local_iters:
        for batch in timed(loader, waited):
            if step >= local_iters:
                break
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
                    model = None
                    gc.collect()
                    model = fabric.setup_module(raw)
                    fabric.print(f'  re-wrapped DDP at iteration {it} over '
                                 f'{sum(p.requires_grad for p in raw.parameters())} trainable '
                                 f'tensor(s) -- the encoder is in the reducer from here',
                                 flush=True)
                    record({'iter': it, 'ddp_rewrapped': True})
            loss, _ = run_batch(model, loss_fn, batch, device, raw=raw)
            if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(loss))):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                it = step * world
                continue
            opt.zero_grad(set_to_none=True)
            fabric.backward(loss)
            gn = torch.nn.utils.clip_grad_norm_(clip_params, max_grad or 1e9)
            mgrads = [p.grad for p in unclipped_params if p.grad is not None]
            mgn = torch.nn.utils.get_total_norm(mgrads) if mgrads else gn.new_zeros(())
            if not dist_utils.all_ranks_finite(
                    fabric, bool(torch.isfinite(gn) and torch.isfinite(mgn))):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                it = step * world
                continue
            opt.step()
            running.append(dist_utils.all_ranks_mean(fabric, float(loss.detach())))
            clipped.append(float(max_grad is not None and float(gn) > max_grad))
            step += 1
            it = step * world

            if step % local_print_freq == 0:
                wall = time.time() - t0
                elapsed = max(wall - evalled[0] - ckpted[0], 1e-9)
                dt = elapsed / local_print_freq
                wait_frac = waited[0] / elapsed
                eval_frac = evalled[0] / wall if wall > 0 else 0.0
                peak_gb = _gpu_peak_gb(device)
                fabric.print(f'{it:7d}/{n_iter}  loss {np.mean(running):8.4f}  '
                             f'{dt:5.2f}s/it  wait {wait_frac:4.0%}  eval {eval_frac:4.0%}  '
                             f'peak {peak_gb:5.1f}G  skipped {skipped * world}  '
                             f'[{batch.sample_info["dataset"]}/{batch.sample_info["mode"]}'
                             f'{"/1cam" if batch.sample_info["single_view"] else ""}]', flush=True)
                log(wb, {'train/loss': float(np.mean(running)), 'train/grad_norm': float(gn),
                         'train/grad_norm_muon': float(mgn),
                         'train/iteration': it,
                         'train/encoder_trainable_params': trainable_encoder_params(raw),
                         'train/gpu_peak_gb': _gpu_peak_gb(device, reset=True),
                         'train/clipped': float(np.mean(clipped[-200:])) if clipped else 0.0,
                         'train/sec_per_it': dt, 'train/loader_wait_frac': wait_frac,
                         'train/eval_frac': eval_frac,
                         'train/ckpt_frac': ckpted[0] / wall if wall > 0 else 0.0,
                         'train/skipped_frac': skipped / max(step, 1),
                         'train/world_size': world}, it)
                running, t0 = [], time.time()
                waited[0] = evalled[0] = ckpted[0] = 0.0

            if idxs and local_val_freq and step % local_val_freq == 0:
                t_val = time.time()
                m, ms = evaluate(raw, val_batches, opt, device, fabric=fabric)
                evalled[0] += time.time() - t_val
                fabric.print(f'  EVAL  ({time.time() - t_val:.0f}s)  prior-free {_brief(m)}\n'
                             f'        self-prompted {_brief(ms)}', flush=True)
                log(wb, {f'val/{k}': v for k, v in m.items()}, it)
                log(wb, {f'val_self/{k}': v for k, v in ms.items()}, it)
                if np.isfinite(m.get('mpjpe', np.nan)):
                    latest = (m['mpjpe'], it)
                    if m['mpjpe'] < best_mpjpe:
                        best_mpjpe, best_iter = m['mpjpe'], it
                log(wb, {'val/no_new_best_span': it - best_iter}, it)
                if wb is not None:
                    wb.run.summary['best_mpjpe'] = best_mpjpe
                    wb.run.summary['best_iter'] = best_iter
                record({'iter': it, 'val': m, 'val_self': ms,
                        'best_mpjpe': best_mpjpe, 'best_iter': best_iter})

            if step % local_ckpt_freq == 0 or step == local_iters:
                t_ck = time.time()
                drift, p = dist_utils.check_ranks_agree(fabric, raw), None
                if not args.no_checkpoints:
                    p = save_checkpoint(run, it, raw, opt, config, write=is0)
                    if latest[1] == it and latest[0] < saved_mpjpe:
                        saved_mpjpe = latest[0]
                        save_checkpoint(run, it, raw, opt, config, name='best', write=is0)
                        fabric.print(f'  new best: mpjpe {saved_mpjpe:.4g} -> '
                                     f'checkpoint_best.pth')
                        record({'iter': it, 'saved_mpjpe': saved_mpjpe})
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
