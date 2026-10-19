"""Finetune a tracktail tracker into a pose estimator.

Body of `tailcyclenet train` (`tailcyclenet/__main__.py`) and of `scripts/train.py`
(`pixi run python scripts/train.py --config configs/base.toml [--devices N]`) -- both call this
module's `main()` unchanged.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tailcyclenet import distributed as dist_utils
from tailcyclenet.checkpoints import save_checkpoint, save_run_meta
from tailcyclenet.config import BASE_CONFIG, check_image_size, load_config
from tailcyclenet.dataset import (PoseDataset, SpawnHandoff, StepSampler, loader_config_from,
                                  pose_collate, worker_init)
from tailcyclenet.pose.losses import PoseLoss
from tailcyclenet.pose.model import build_model
from tailcyclenet.optim import KNOWN_OPTIMIZER_KEYS
from tailcyclenet.runtime import gpu_peak_gb, init_wandb, launch, log, timed
from tailcyclenet.start import build_start, plan_start
from tailcyclenet.unfreeze import apply_staged_unfreeze, trainable_encoder_params

# file_system sharing avoids the default's fd-per-tensor lifetime; harmless, keep it.
torch.multiprocessing.set_sharing_strategy('file_system')


def _brief(m: dict) -> str:
    """The interesting metric keys of a val dict, space-joined for a one-line print."""
    keys = ('mpjpe', 'mte', 'delta_x_avg', 'survival_rate', 'n_nonfinite')
    return ' '.join(f'{k}={m[k]:.4g}' for k in keys if k in m)


KNOWN_TRAINING_KEYS = frozenset({
    'n_iterations', 'seed', 'checkpoint_path', 'checkpoint_revision', 'max_grad_norm',
    'checkpoint_freq', 'val_freq', 'val_batches', 'print_freq', 'train_metric_freq', 'out',
    'optimizer', 'losses'})


def check_training_keys(train_cfg: dict) -> None:
    """Refuse an unknown `[training]` key (a typo would train at the defaults and report as the
    arm it is not), and the deleted `freeze_encoder` by name."""
    if 'freeze_encoder' in train_cfg:
        raise SystemExit(
            '[training].freeze_encoder was removed: it re-froze the encoder AFTER build_model, so '
            'it silently overrode [model].video_encoder_requires_grad and would defeat the staged '
            'unfreeze. Set [model].video_encoder_requires_grad instead -- false (never train the '
            'encoder), true (from step 0), or an ITERATION to unfreeze at -- with '
            '[model].video_encoder_finetune_last_n_layers for how many trailing blocks.')
    unknown = set(train_cfg) - KNOWN_TRAINING_KEYS
    if unknown:
        raise SystemExit(
            f'[training]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(KNOWN_TRAINING_KEYS)}')


def has_val_split(root: Path) -> bool:
    """Whether a dataset root (or any root in a folder of roots) has a `val/` split. Only a
    MISSING split skips validation; an error building a present one propagates."""
    root = Path(root)
    return (root / 'val').is_dir() or any(
        (c / 'val').is_dir() for c in root.iterdir() if c.is_dir())


def sampler_generator(seed: int, rank: int, world: int, first_step: int):
    """The train sampler's RNG. A fresh run keeps the stream it always had (the global RNG at one
    rank, `seed + rank` above); a resume at `first_step > 0` draws a NEW stream keyed on where it
    resumes, instead of replaying the indices the run already saw."""
    if first_step:
        return torch.Generator().manual_seed(
            int(seed) * 1_000_003 + int(rank) * 7_919 + int(first_step))
    if world > 1:
        return torch.Generator().manual_seed(int(seed) + int(rank))
    return None


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


_SMOOTHNESS_KEYS = ('smoothness_3d_loss', 'smoothness_2d_loss')
# Scalar outputs shared with validation; delta_x_1..16 are per-batch arrays and are omitted by
# the same scalar filter. The visibility-based 2D values are diagnostics alongside the optional
# 2D visibility-loss head.
_TRAIN_METRIC_KEYS = (
    'mte', 'delta_x_avg', 'occlusion_acc', 'avg_jaccard', 'survival_rate', 'mpjpe',
    'jaccard_1', 'jaccard_2', 'jaccard_4', 'jaccard_8', 'jaccard_16',
)
_TRAIN_METRIC_MODES = ('2d', '3d')


@torch.no_grad()
def train_metrics(out, batch):
    """Return finite pose metrics for one accepted training forward.

    These are the same metrics as validation, but the training forward uses the batch's
    ground-truth prompt.  They are therefore an easier, prompt-conditioned diagnostic and
    must not be compared directly with ``val/`` or ``val_self/``.  Metrics are kept separate
    by mode because 2D values are pixels while 3D values are world units.
    """
    if 'vis_pred' not in out or 'coords_pred' not in out:
        return {}
    from tracktail.model.eval_metrics import get_eval_metrics, get_vis_true

    coords = batch.coords
    vis_true = batch.vis if batch.vis is not None else get_vis_true(coords)
    metrics = get_eval_metrics(vis_pred=out['vis_pred'], vis_true=vis_true,
                               coords_pred=out['coords_pred'], coords_true=coords, prefix='')
    return {k: float(v) for k, v in metrics.items()
            if k in _TRAIN_METRIC_KEYS and np.ndim(v) == 0 and np.isfinite(v)}


def _clear_loss_history(loss_fn):
    """Clear recorded loss samples in place, preserving ``defaultdict`` lazy keys."""
    for samples in loss_fn.loss_history.values():
        samples.clear()


def _drain_smoothness(loss_fn):
    """Drain one batch's smoothness samples as ``name -> (finite sum, finite count)``.

    The training loop calls this after every batch.  It can therefore discard a skipped batch
    before adding anything to the print interval, while the in-place clear prevents history growth
    without replacing ``TotalLoss.loss_history`` with a plain dict.
    """
    stats = {}
    for name in _SMOOTHNESS_KEYS:
        samples = loss_fn.loss_history.get(name, ())
        finite = [float(value) for value in samples if np.isfinite(value)]
        if finite:
            stats[name] = (float(np.sum(finite)), len(finite))
    _clear_loss_history(loss_fn)
    return stats


def _interval_means(stats, keys, prefix, fabric=None):
    """Reduce finite sum/count pairs and return count-weighted interval means.

    The reduction is one fixed-size collective so every rank participates even when its local
    interval has no samples.  A metric is omitted only when its global count is zero; genuine
    zero-valued terms are retained.
    """
    keys = tuple(keys)
    if fabric is not None and fabric.world_size > 1:
        local = []
        for key in keys:
            total, count = stats.get(key, (0.0, 0))
            local.extend((total, float(count)))
        reduced = fabric.all_reduce(
            torch.tensor(local, dtype=torch.float64, device=fabric.device), reduce_op='sum')
        stats = {key: (float(reduced[2 * i]), int(round(float(reduced[2 * i + 1]))))
                 for i, key in enumerate(keys)}
    values = {}
    for key in keys:
        total, count = stats.get(key, (0.0, 0))
        if count:
            values[f'{prefix}{key}'] = total / count
    return values


def _smoothness_log_values(stats, fabric=None):
    """Reduce smoothness samples and return interval means for the W&B payload."""
    return _interval_means(stats, _SMOOTHNESS_KEYS, 'train/', fabric)


def _metric_log_values(stats, fabric=None):
    """Return mode-separated training metric interval means for W&B."""
    flat = {f'{mode}/{key}': pair for mode in _TRAIN_METRIC_MODES
            for key, pair in stats.get(mode, {}).items()}
    return _interval_means(flat,
                           tuple(f'{mode}/{key}' for mode in _TRAIN_METRIC_MODES
                                 for key in _TRAIN_METRIC_KEYS),
                           'train/', fabric)


def run_batch(model, loss_fn, batch, device, raw=None):
    """One forward + loss; a non-finite sub-loss returns NaN, never raises.

    `model` is the DDP-wrapped forward handle; `raw` the unwrapped module the loss reads.
    tracktail raises on a poisoned sub-loss; this converts it to the NaN the loop's skip path
    already handles. Watch `skipped`: a rising fraction means the run is dead, not intermittent.
    `batch.vis_2d` is `vis_2d_true` (PoseLoss's own 2D term) in 2D and `vis_true_cams`
    (tracktail's per-camera term) in 3D -- the two must never mix.
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
    from tracktail.model.eval_metrics import get_eval_metrics, get_vis_true

    from tailcyclenet.pose.prompt import self_prompt
    from tailcyclenet.pose.model import share_scene

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


def main(argv: list[str] | None = None):
    """Finetune a tracktail tracker into a pose estimator.

    Side effects: trains; writes checkpoints + log.jsonl under the run folder; launches Fabric
                  (re-executes per rank).

    Notes:
    - Unknown `[training]`/`[data]` keys are refused (a typo would train at the
      defaults and report as the arm it is not); a missing `val/` split is the
      only error swallowed.
    - Every rank builds the same registry, appended to the start checkpoint's own, so
      `warm_start` can prove the checkpoint's keypoint table is a prefix of this run's.
    - One epoch for the whole run (sampling is with replacement); `num_workers` and the RAM
      ceiling are per rank. Val windows decode in a one-worker child, not the parent. A resumed
      run's sampler continues at its step (`StepSampler(start=)`) on a new index stream.
    - How the run starts -- fresh, warm start, or resume -- is `start.plan_start`'s decision,
      made before anything is built; `start.build_start` carries it out (resume restores
      `model_state`, not `model_state_eval`, and replays the staged unfreeze).
    - DDP registers parameters at wrap time, so the module is wrapped after the resume replay
      and re-wrapped after an unfreeze.
    - The resolved `box_source`, optimizer kind and world size are recorded in
      `provenance.toml`; metrics always go to log.jsonl.
    - `train/{2d,3d}/*` uses the batch's GT prompt (`kpt_prior`/`prompt_t`); `val/*` is prompt-free, so do not plot them on one axis.
    - Every frequency is a total across ranks; `step` is this rank's local count,
      `it = step * world` global; a skipped step is a collective decision.
      `grad_norm` is the clipped AdamW half only; `saved_mpjpe` is the metric of
      `checkpoint_best.pth` on disk.
    """
    ap = argparse.ArgumentParser(prog='tailcyclenet train', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, default=None,
                    help='defaults to the packaged configs/base.toml (the shipped recipe, '
                         'unmodified) when omitted -- pass one that overlays it for anything else')
    ap.add_argument('--iters', type=int, default=None, help='override n_iterations')
    ap.add_argument('--out', type=Path, default=None, help='override the run folder')
    ap.add_argument('--data', type=Path, default=None, help='override [data].path')
    ap.add_argument('--device', default='cuda:0',
                    help='which gpu, at --devices 1. Refused with --devices != 1, where placement '
                         'is the launcher\'s job')
    ap.add_argument('--devices', default=-1,
                    help='gpus to train on: a count, or -1 for every visible one (the default). '
                         'Above 1, one rank per gpu and DDP averages their gradients, so the '
                         'effective batch is the world size')
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
    args = ap.parse_args(argv)
    fabric = launch(args)
    world, is0 = fabric.world_size, fabric.is_global_zero

    config = load_config(args.config or BASE_CONFIG)
    check_image_size(config)
    data_cfg, train_cfg = config['data'], config['training']
    if args.data:
        data_cfg['path'] = str(args.data)
    if args.iters:
        train_cfg['n_iterations'] = args.iters
    run = Path(args.out or train_cfg['out'])
    n_iter = int(train_cfg['n_iterations'])
    device = fabric.device
    check_training_keys(train_cfg)
    opt_cfg = config['training']['optimizer']
    opt_unknown = set(opt_cfg) - KNOWN_OPTIMIZER_KEYS
    if opt_unknown:
        raise SystemExit(
            f'[training.optimizer]: unknown key(s) {sorted(opt_unknown)}. Nothing reads them, so '
            f'this run would train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(KNOWN_OPTIMIZER_KEYS)}')

    torch.manual_seed(int(train_cfg.get('seed', 23)))
    np.random.seed(int(train_cfg.get('seed', 23)))

    lc = loader_config_from(data_cfg)
    lc = replace(lc, box_prompt=config['model'].get('box_prompt', 'none'))
    plan = plan_start(run, train_cfg.get('checkpoint_path'),
                      revision=train_cfg.get('checkpoint_revision'), no_resume=args.no_resume,
                      no_warm_start=args.no_warm_start,
                      opt_kind=str(opt_cfg.get('optimizer', 'muon')), world=world)
    if plan.registry is not None:
        print(f'keypoint registry: appending to the {plan.mode} checkpoint\'s ({plan.path})')
    train_ds = PoseDataset(data_cfg['path'], 'train', lc, registry_base=plan.registry)
    registry = train_ds.registry
    dist_utils.check_registry(fabric, registry.names)
    print(f'train: {len(train_ds)} windows across {len(train_ds.datasets)} dataset(s), '
          f'{registry.n_keypoints} keypoints')
    print('train: mix ' + '  '.join(f'{k}={v:.1%}' for k, v in train_ds.mix().items()))
    val_ds = None
    root = Path(data_cfg['path'])
    if not has_val_split(root):
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

    opt_cfg_scaled = dist_utils.scale_optimizer_cfg(opt_cfg, world)
    if world > 1:
        fabric.print(f'lr: scaled by sqrt({world}) -> learning_rate '
                     f'{opt_cfg_scaled["learning_rate"]:g}'
                     + (f', kpt_lr {opt_cfg_scaled["kpt_lr"]:g}' if 'kpt_lr' in opt_cfg_scaled
                        else '')
                     + ' (the multipliers -- encoder_lr_scale, muon_lr_scale -- are unchanged)')

    def make_model(pretrained):
        """This run's pose model; the backbone download is skipped when a checkpoint overwrites it."""
        cfg = config['model'] if pretrained else {**config['model'],
                                                  'video_encoder_pretrained': False}
        return build_model(cfg, n_keypoints=registry.n_keypoints)

    started = build_start(plan, make_model, opt_cfg_scaled, config['model'], device)
    model, opt, fresh, plan = started.model, started.optimizer, started.fresh, started.plan
    start_it = plan.start_it
    if not plan.resumes and (run / 'checkpoints' / 'checkpoint_last.pth').exists():
        print(f'--no-resume: {run / "checkpoints"} checkpoint_last.pth and any checkpoint_best.pth '
              'beside it WILL be overwritten')
    losses = dict(config['training']['losses'])
    losses.setdefault('per_camera_cube_scale',
                      bool(config['model'].get('per_camera_cube_scale', False)))
    loss_fn = PoseLoss(**losses)

    local_iters = dist_utils.per_rank(n_iter, world)
    first_step = dist_utils.ceil_div(start_it, world)
    gen = sampler_generator(int(train_cfg.get('seed', 23)), fabric.global_rank, world, first_step)
    loader = torch.utils.data.DataLoader(
        SpawnHandoff(train_ds), batch_size=1, num_workers=nw, collate_fn=pose_collate,
        sampler=StepSampler(len(train_ds), num_samples=max(local_iters - first_step, 0),
                            generator=gen, start=first_step),
        generator=gen,
        persistent_workers=nw > 0, pin_memory=True, drop_last=True,
        prefetch_factor=4 if nw > 0 else None, worker_init_fn=worker_init,
        multiprocessing_context='spawn' if nw > 0 else None)

    raw = model
    wrap = world > 1 or args.precision != '32-true'
    model = fabric.setup_module(raw) if wrap else raw

    config.setdefault('data', {})['box_source'] = train_ds.cfg.box_source
    opt_cfg['optimizer'] = str(opt_cfg.get('optimizer', 'muon'))
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
    smoothness_running = {name: [0.0, 0] for name in _SMOOTHNESS_KEYS}
    metrics_running = {
        mode: {name: [0.0, 0] for name in _TRAIN_METRIC_KEYS}
        for mode in _TRAIN_METRIC_MODES}
    metric_freq = int(train_cfg.get('train_metric_freq', 1))
    local_metric_freq = dist_utils.per_rank(metric_freq, world) if metric_freq else 0
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
            loss, out = run_batch(model, loss_fn, batch, device, raw=raw)
            if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(loss))):
                _drain_smoothness(loss_fn)
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
                _drain_smoothness(loss_fn)
                skipped += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                it = step * world
                continue
            opt.step()
            running.append(dist_utils.all_ranks_mean(fabric, float(loss.detach())))
            if local_metric_freq and step % local_metric_freq == 0:
                bucket = metrics_running.get(batch.sample_info['mode'])
                if bucket is not None:
                    for name, value in train_metrics(out, batch).items():
                        bucket[name][0] += value
                        bucket[name][1] += 1
            for name, (total, count) in _drain_smoothness(loss_fn).items():
                smoothness_running[name][0] += total
                smoothness_running[name][1] += count
            clipped.append(float(max_grad is not None and float(gn) > max_grad))
            step += 1
            it = step * world

            if step % local_print_freq == 0:
                wall = time.time() - t0
                elapsed = max(wall - evalled[0] - ckpted[0], 1e-9)
                dt = elapsed / local_print_freq
                wait_frac = waited[0] / elapsed
                eval_frac = evalled[0] / wall if wall > 0 else 0.0
                peak_gb = gpu_peak_gb(device)
                fabric.print(f'{it:7d}/{n_iter}  loss {np.mean(running):8.4f}  '
                             f'{dt:5.2f}s/it  wait {wait_frac:4.0%}  eval {eval_frac:4.0%}  '
                             f'peak {peak_gb:5.1f}G  skipped {skipped * world}  '
                             f'[{batch.sample_info["dataset"]}/{batch.sample_info["mode"]}'
                             f'{"/1cam" if batch.sample_info["single_view"] else ""}]', flush=True)
                smoothness = _smoothness_log_values(smoothness_running, fabric)
                train_metric_values = _metric_log_values(metrics_running, fabric)
                log(wb, {'train/loss': float(np.mean(running)), 'train/grad_norm': float(gn),
                         'train/grad_norm_muon': float(mgn),
                         'train/iteration': it,
                         'train/encoder_trainable_params': trainable_encoder_params(raw),
                         'train/gpu_peak_gb': gpu_peak_gb(device, reset=True),
                         'train/clipped': float(np.mean(clipped[-200:])) if clipped else 0.0,
                         'train/sec_per_it': dt, 'train/loader_wait_frac': wait_frac,
                         'train/eval_frac': eval_frac,
                         'train/ckpt_frac': ckpted[0] / wall if wall > 0 else 0.0,
                         'train/skipped_frac': skipped / max(step, 1),
                         'train/world_size': world, **smoothness, **train_metric_values}, it)
                record({'iter': it, 'train': train_metric_values})
                smoothness_running = {name: [0.0, 0] for name in _SMOOTHNESS_KEYS}
                metrics_running = {
                    mode: {name: [0.0, 0] for name in _TRAIN_METRIC_KEYS}
                    for mode in _TRAIN_METRIC_MODES}
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
                    p = save_checkpoint(run, it, raw, opt, config, write=is0, registry=registry,
                                        fresh_names=fresh, world_size=world)
                    if latest[1] == it and latest[0] < saved_mpjpe:
                        saved_mpjpe = latest[0]
                        save_checkpoint(run, it, raw, opt, config, name='best', write=is0,
                                        registry=registry, fresh_names=fresh, world_size=world)
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
