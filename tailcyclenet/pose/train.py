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

from .. import distributed as dist_utils
from ..checkpoints import save_checkpoint, save_run_meta
from ..config import BASE_CONFIG, check_image_size, load_config
from ..loader.config import loader_config_from
from ..loader.dataset import PoseDataset
from ..loader.sampling import SpawnHandoff, StepSampler, pose_collate, worker_init
from ..runtime import gpu_peak_gb, init_wandb, launch, log, timed
from .evaluate import (SMOOTHNESS_KEYS, TRAIN_METRIC_KEYS, TRAIN_METRIC_MODES, brief,
                       drain_smoothness, evaluate, metric_log_values, smoothness_log_values,
                       to_device, train_metrics)
from .losses import PoseLoss
from .model import build_model
from .optim import KNOWN_OPTIMIZER_KEYS
from .start import build_start, plan_start
from .unfreeze import apply_staged_unfreeze, trainable_encoder_params

# file_system sharing avoids the default's fd-per-tensor lifetime; harmless, keep it.
torch.multiprocessing.set_sharing_strategy('file_system')


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


def build_parser() -> argparse.ArgumentParser:
    """The `tailcyclenet train` command line."""
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
    ap.add_argument('--no-tf32', action='store_true',
                    help='disable TF32 matmuls/convs (on by default)')
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
    return ap


def load_train_config(args) -> tuple[dict, Path]:
    """The layered config with the CLI overrides applied and every unknown key refused -> (config, run)."""
    config = load_config(args.config or BASE_CONFIG)
    check_image_size(config)
    data_cfg, train_cfg = config['data'], config['training']
    if args.data:
        data_cfg['path'] = str(args.data)
    if args.iters:
        train_cfg['n_iterations'] = args.iters
    run = Path(args.out or train_cfg['out'])
    check_training_keys(train_cfg)
    opt_cfg = config['training']['optimizer']
    opt_unknown = set(opt_cfg) - KNOWN_OPTIMIZER_KEYS
    if opt_unknown:
        raise SystemExit(
            f'[training.optimizer]: unknown key(s) {sorted(opt_unknown)}. Nothing reads them, so '
            f'this run would train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(KNOWN_OPTIMIZER_KEYS)}')
    return config, run


def build_datasets(data_cfg: dict, lc, plan):
    """(train_ds, val_ds or None): a resume appends to the start checkpoint's registry; a warm
    start builds the registry from the data and `warm_start` maps the identity rows by name."""
    base = plan.registry if plan.resumes else None
    if base is not None:
        print(f'keypoint registry: appending to the {plan.mode} checkpoint\'s ({plan.path})')
    elif plan.registry is not None:
        print(f'keypoint registry: built from the data; identity rows copied by name from '
              f'{plan.path}')
    train_ds = PoseDataset(data_cfg['path'], 'train', lc, registry_base=base)
    registry = train_ds.registry
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
    return train_ds, val_ds


def fixed_val_batches(val_ds, train_cfg: dict, fabric) -> tuple[list, list[int]]:
    """This rank's shard of the fixed val windows, decoded in a one-worker child -> (batches, idxs).
    A window that fails to build is skipped (named), never replaced by another."""
    world = fabric.world_size
    val_freq = int(train_cfg.get('val_freq', 0))
    val_batches, idxs = [], []
    if val_ds is not None and val_freq:
        n_val = min(int(train_cfg.get('val_batches', 20)), len(val_ds))
        idxs = [int(i) for i in np.unique(np.linspace(0, len(val_ds) - 1, n_val).round().astype(int))]
        mine = [int(i) for i in idxs[fabric.global_rank::world]]
        val_batches = list(torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_ds, mine), batch_size=1,
            num_workers=1, collate_fn=pose_collate, worker_init_fn=worker_init))
        failed = [i for i, b in zip(mine, val_batches) if b is None]
        if failed:
            fabric.print(f'val: {len(failed)} window(s) failed to build and are SKIPPED: '
                         f'{failed[:10]}')
        val_batches = [b for b in val_batches if b is not None]
        seen = Counter(val_ds.index[int(i)].session.session_id for i in idxs)
        fabric.print(f'val:   {len(idxs)} fixed window(s) every {val_freq} iterations, over '
                     f'{len(seen)}/{len({it.session.session_id for it in val_ds.index})} '
                     f'session(s)' + (f', {len(mine)} on this rank' if world > 1 else ''))
        for sid, n in sorted(seen.items()):
            fabric.print(f'         {n:3d}  {sid}')
    return val_batches, idxs


def train_loader(train_ds, nw: int, local_iters: int, first_step: int, gen):
    """The one-epoch, with-replacement train loader; spawned workers, continuing at `first_step`."""
    return torch.utils.data.DataLoader(
        SpawnHandoff(train_ds), batch_size=1, num_workers=nw, collate_fn=pose_collate,
        sampler=StepSampler(len(train_ds), num_samples=max(local_iters - first_step, 0),
                            generator=gen, start=first_step),
        generator=gen,
        persistent_workers=nw > 0, pin_memory=True, drop_last=True,
        prefetch_factor=4 if nw > 0 else None, worker_init_fn=worker_init,
        multiprocessing_context='spawn' if nw > 0 else None)


def resumed_best(start_it: int, log_path: Path) -> tuple[float, int, float]:
    """(best_mpjpe, best_iter, saved_mpjpe) carried over from a resumed run's log.jsonl."""
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
    return best_mpjpe, best_iter, saved_mpjpe


def _fresh_metric_buckets():
    """Empty (smoothness, per-mode train metric) sum/count accumulators for one print interval."""
    return ({name: [0.0, 0] for name in SMOOTHNESS_KEYS},
            {mode: {name: [0.0, 0] for name in TRAIN_METRIC_KEYS} for mode in TRAIN_METRIC_MODES})


class StepLoop:
    """The step loop's state for one run; `run_loop()` trains until this rank's last local step.

    `step` is this rank's local count and `it = step * world` the global one. A skipped step is a
    collective decision and skips the print/val/checkpoint checks too.
    """

    def __init__(self, *, fabric, args, config, run, registry, raw, model, opt, opt_cfg_scaled,
                 fresh, loss_fn, loader, val_batches, idxs, wb, train_ds, start_it, wrap):
        """Bind everything the loop reads; the frequencies become per-rank local step counts."""
        train_cfg = config['training']
        world = fabric.world_size
        self.fabric, self.args, self.config, self.run, self.registry = fabric, args, config, run, registry
        self.raw, self.model, self.opt, self.opt_cfg_scaled, self.fresh = raw, model, opt, opt_cfg_scaled, fresh
        self.loss_fn, self.loader, self.val_batches, self.idxs, self.wb = loss_fn, loader, val_batches, idxs, wb
        self.train_ds, self.wrap, self.world, self.is0 = train_ds, wrap, world, fabric.is_global_zero
        self.device = fabric.device
        self.n_iter = int(train_cfg['n_iterations'])
        self.local_iters = dist_utils.per_rank(self.n_iter, world)
        self.log_path = run / 'log.jsonl'
        self.max_grad = float(train_cfg.get('max_grad_norm', 0)) or None
        val_freq = int(train_cfg.get('val_freq', 0))
        print_freq = int(train_cfg.get('print_freq', 20))
        ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
        self.local_val_freq = dist_utils.per_rank(val_freq, world) if val_freq else 0
        self.local_ckpt_freq = dist_utils.per_rank(ckpt_freq, world)
        self.local_print_freq = dist_utils.per_rank(print_freq, world)
        if idxs and self.local_val_freq and self.local_ckpt_freq % self.local_val_freq:
            fabric.print(f'WARNING: checkpoint_freq {ckpt_freq} is not a multiple of val_freq '
                         f'{val_freq} (as {self.local_ckpt_freq} and {self.local_val_freq} local '
                         f'steps on {world} rank(s)), so no evaluation lands on a checkpoint '
                         f'boundary and checkpoint_best.pth will never be written. Pick freqs '
                         f'that divide.')
        self.clip_params, self.unclipped_params = self.clip_targets()
        metric_freq = int(train_cfg.get('train_metric_freq', 1))
        self.local_metric_freq = dist_utils.per_rank(metric_freq, world) if metric_freq else 0
        self.step = dist_utils.ceil_div(start_it, world)
        self.it, self.skipped, self.t0, self.running, self.clipped = self.step * world, 0, time.time(), [], []
        self.smoothness_running, self.metrics_running = _fresh_metric_buckets()
        self.best_mpjpe, self.best_iter, self.saved_mpjpe = resumed_best(start_it, self.log_path)
        self.latest = (float('inf'), -1)
        self.waited, self.evalled, self.ckpted = [0.0], [0.0], [0.0]

    def record(self, rec):
        """Append one JSON line to log.jsonl -- rank 0 only, reduced metrics.

        Rank 0 only: N ranks appending the same reduced metrics would corrupt the history.
        """
        if not self.is0:
            return
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def clip_targets(self):
        """Recomputed after an unfreeze: a stale list leaves new tensors unclipped/unchecked."""
        trainable = [p for p in self.raw.parameters() if p.requires_grad]
        return getattr(self.opt, 'adamw_params', trainable), getattr(self.opt, 'muon_params', [])

    def run_loop(self):
        """Train until this rank's last local step, then log the run's summary line."""
        fabric, world = self.fabric, self.world
        self.model.train()
        self.opt.train()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        while self.step < self.local_iters:
            for batch in timed(self.loader, self.waited):
                if self.step >= self.local_iters:
                    break
                self.maybe_unfreeze()
                stepped = self.train_step(batch)
                if stepped is None:
                    continue
                gn, mgn = stepped
                if self.step % self.local_print_freq == 0:
                    self.print_interval(batch, gn, mgn)
                if self.idxs and self.local_val_freq and self.step % self.local_val_freq == 0:
                    self.validate()
                if self.step % self.local_ckpt_freq == 0 or self.step == self.local_iters:
                    self.checkpoint()
        fabric.print(f'done: {self.it} iterations, {self.skipped * world} skipped')
        self.record({'iter': self.it, 'done': True, 'skipped': self.skipped * world,
                     'world_size': world, 'best_mpjpe': self.best_mpjpe,
                     'best_iter': self.best_iter})
        if self.wb is not None:
            self.wb.finish()

    def maybe_unfreeze(self):
        """Fire the staged encoder unfreeze at its iteration; re-wrap DDP over the new params."""
        fabric, it, raw = self.fabric, self.it, self.raw
        unfroze = apply_staged_unfreeze(raw, self.opt, self.opt_cfg_scaled, it, fresh=self.fresh)
        if not unfroze:
            return
        self.clip_params, self.unclipped_params = self.clip_targets()
        fabric.print(f'  UNFROZE the video encoder at iteration {it}: blocks '
                     f'{unfroze["blocks"]} of {unfroze["n_blocks"]}, norms '
                     f'{unfroze["norms"]}, {unfroze["n_tensors"]} tensors / '
                     f'{unfroze["n_params"]:,} params (+{unfroze["muon_groups"]} muon, '
                     f'+{unfroze["adamw_groups"]} adamw group(s))', flush=True)
        log(self.wb, {'train/encoder_unfrozen_at': it}, it)
        self.record({'iter': it, 'unfroze_encoder': unfroze})
        if self.wrap:
            self.model = None
            gc.collect()
            self.model = fabric.setup_module(raw)
            fabric.print(f'  re-wrapped DDP at iteration {it} over '
                         f'{sum(p.requires_grad for p in raw.parameters())} trainable '
                         f'tensor(s) -- the encoder is in the reducer from here',
                         flush=True)
            self.record({'iter': it, 'ddp_rewrapped': True})

    def _skip(self):
        """Discard this step on every rank: drop its smoothness samples and its gradients."""
        drain_smoothness(self.loss_fn)
        self.skipped += 1
        self.opt.zero_grad(set_to_none=True)
        self.step += 1
        self.it = self.step * self.world

    def train_step(self, batch):
        """One optimizer step -> (grad_norm, grad_norm_muon), or None when it was skipped."""
        fabric, opt = self.fabric, self.opt
        loss, out = run_batch(self.model, self.loss_fn, batch, self.device, raw=self.raw)
        if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(loss))):
            self._skip()
            return None
        opt.zero_grad(set_to_none=True)
        fabric.backward(loss)
        gn = torch.nn.utils.clip_grad_norm_(self.clip_params, self.max_grad or 1e9)
        mgrads = [p.grad for p in self.unclipped_params if p.grad is not None]
        mgn = torch.nn.utils.get_total_norm(mgrads) if mgrads else gn.new_zeros(())
        if not dist_utils.all_ranks_finite(
                fabric, bool(torch.isfinite(gn) and torch.isfinite(mgn))):
            self._skip()
            return None
        opt.step()
        self.running.append(dist_utils.all_ranks_mean(fabric, float(loss.detach())))
        if self.local_metric_freq and self.step % self.local_metric_freq == 0:
            bucket = self.metrics_running.get(batch.sample_info['mode'])
            if bucket is not None:
                for name, value in train_metrics(out, batch).items():
                    bucket[name][0] += value
                    bucket[name][1] += 1
        for name, (total, count) in drain_smoothness(self.loss_fn).items():
            self.smoothness_running[name][0] += total
            self.smoothness_running[name][1] += count
        self.clipped.append(float(self.max_grad is not None and float(gn) > self.max_grad))
        self.step += 1
        self.it = self.step * self.world
        return gn, mgn

    def print_interval(self, batch, gn, mgn):
        """The interval line, its W&B payload and log.jsonl record; then reset the interval."""
        fabric, it, world, device = self.fabric, self.it, self.world, self.device
        wall = time.time() - self.t0
        elapsed = max(wall - self.evalled[0] - self.ckpted[0], 1e-9)
        dt = elapsed / self.local_print_freq
        wait_frac = self.waited[0] / elapsed
        eval_frac = self.evalled[0] / wall if wall > 0 else 0.0
        peak_gb = gpu_peak_gb(device)
        fabric.print(f'{it:7d}/{self.n_iter}  loss {np.mean(self.running):8.4f}  '
                     f'{dt:5.2f}s/it  wait {wait_frac:4.0%}  eval {eval_frac:4.0%}  '
                     f'peak {peak_gb:5.1f}G  skipped {self.skipped * world}  '
                     f'[{batch.sample_info["dataset"]}/{batch.sample_info["mode"]}'
                     f'{"/1cam" if batch.sample_info["single_view"] else ""}]', flush=True)
        smoothness = smoothness_log_values(self.smoothness_running, fabric)
        train_metric_values = metric_log_values(self.metrics_running, fabric)
        log(self.wb, {'train/loss': float(np.mean(self.running)), 'train/grad_norm': float(gn),
                      'train/grad_norm_muon': float(mgn),
                      'train/iteration': it,
                      'train/encoder_trainable_params': trainable_encoder_params(self.raw),
                      'train/gpu_peak_gb': gpu_peak_gb(device, reset=True),
                      'train/clipped': (float(np.mean(self.clipped[-200:])) if self.clipped
                                        else 0.0),
                      'train/sec_per_it': dt, 'train/loader_wait_frac': wait_frac,
                      'train/eval_frac': eval_frac,
                      'train/ckpt_frac': self.ckpted[0] / wall if wall > 0 else 0.0,
                      'train/skipped_frac': self.skipped / max(self.step, 1),
                      'train/world_size': world, **smoothness, **train_metric_values}, it)
        self.record({'iter': it, 'train': train_metric_values})
        self.smoothness_running, self.metrics_running = _fresh_metric_buckets()
        self.running, self.t0 = [], time.time()
        self.waited[0] = self.evalled[0] = self.ckpted[0] = 0.0

    def validate(self):
        """Both val regimes on the fixed windows; track the best prior-free mpjpe."""
        fabric, it, wb = self.fabric, self.it, self.wb
        t_val = time.time()
        m, ms = evaluate(self.raw, self.val_batches, self.opt, self.device, fabric=fabric)
        self.evalled[0] += time.time() - t_val
        fabric.print(f'  EVAL  ({time.time() - t_val:.0f}s)  prior-free {brief(m)}\n'
                     f'        self-prompted {brief(ms)}', flush=True)
        log(wb, {f'val/{k}': v for k, v in m.items()}, it)
        log(wb, {f'val_self/{k}': v for k, v in ms.items()}, it)
        if np.isfinite(m.get('mpjpe', np.nan)):
            self.latest = (m['mpjpe'], it)
            if m['mpjpe'] < self.best_mpjpe:
                self.best_mpjpe, self.best_iter = m['mpjpe'], it
        log(wb, {'val/no_new_best_span': it - self.best_iter}, it)
        if wb is not None:
            wb.run.summary['best_mpjpe'] = self.best_mpjpe
            wb.run.summary['best_iter'] = self.best_iter
        self.record({'iter': it, 'val': m, 'val_self': ms,
                     'best_mpjpe': self.best_mpjpe, 'best_iter': self.best_iter})

    def checkpoint(self):
        """Check the ranks agree, then write `last` (and `best` when this val beat the saved one)."""
        fabric, it, world, raw, opt, run = self.fabric, self.it, self.world, self.raw, self.opt, self.run
        t_ck = time.time()
        drift, p = dist_utils.check_ranks_agree(fabric, raw), None
        if not self.args.no_checkpoints:
            p = save_checkpoint(run, it, raw, opt, self.config, write=self.is0,
                                registry=self.registry, fresh_names=self.fresh, world_size=world)
            if self.latest[1] == it and self.latest[0] < self.saved_mpjpe:
                self.saved_mpjpe = self.latest[0]
                save_checkpoint(run, it, raw, opt, self.config, name='best', write=self.is0,
                                registry=self.registry, fresh_names=self.fresh, world_size=world)
                fabric.print(f'  new best: mpjpe {self.saved_mpjpe:.4g} -> '
                             f'checkpoint_best.pth')
                self.record({'iter': it, 'saved_mpjpe': self.saved_mpjpe})
        fabric.barrier()
        self.ckpted[0] += time.time() - t_ck
        if p is not None or world > 1:
            fabric.print(f'saved {p} ({time.time() - t_ck:.0f}s'
                         + (f', rank drift {drift:.1e}' if world > 1 else '') + ')')


def main(argv: list[str] | None = None):
    """Finetune a tracktail tracker into a pose estimator.

    Side effects: trains; writes checkpoints + log.jsonl under the run folder; launches Fabric
                  (re-executes per rank).

    Notes:
    - Unknown `[training]`/`[data]` keys are refused (a typo would train at the
      defaults and report as the arm it is not); a missing `val/` split is the
      only error swallowed.
    - Every rank builds the same registry: appended to the checkpoint's own on a resume, built
      from the data on a warm start (identity rows then copied by name).
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
    args = build_parser().parse_args(argv)
    fabric = launch(args)
    world, is0 = fabric.world_size, fabric.is_global_zero

    config, run = load_train_config(args)
    data_cfg, train_cfg = config['data'], config['training']
    opt_cfg = train_cfg['optimizer']
    n_iter = int(train_cfg['n_iterations'])
    device = fabric.device
    torch.manual_seed(int(train_cfg.get('seed', 23)))
    np.random.seed(int(train_cfg.get('seed', 23)))

    lc = loader_config_from(data_cfg)
    lc = replace(lc, box_prompt=config['model'].get('box_prompt', 'none'))
    plan = plan_start(run, train_cfg.get('checkpoint_path'),
                      revision=train_cfg.get('checkpoint_revision'), no_resume=args.no_resume,
                      no_warm_start=args.no_warm_start,
                      opt_kind=str(opt_cfg.get('optimizer', 'muon')), world=world)
    train_ds, val_ds = build_datasets(data_cfg, lc, plan)
    registry = train_ds.registry
    dist_utils.check_registry(fabric, registry.names)

    nw = args.num_workers if args.num_workers is not None else int(data_cfg.get('num_workers', 8))
    if world * nw > (os.cpu_count() or 1):
        fabric.print(f'WARNING: {world} rank(s) x {nw} loader worker(s) = {world * nw} decoding '
                     f'processes on {os.cpu_count()} cores. Lower [data].num_workers or '
                     f'--num-workers; the loader is this repo\'s documented bottleneck and '
                     f'oversubscribing it is not free.')
    val_batches, idxs = fixed_val_batches(val_ds, train_cfg, fabric)

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

    started = build_start(plan, make_model, opt_cfg_scaled, config['model'], device,
                          names=registry.names)
    raw, opt, fresh, plan = started.model, started.optimizer, started.fresh, started.plan
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
    loader = train_loader(train_ds, nw, local_iters, first_step, gen)

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

    loop = StepLoop(fabric=fabric, args=args, config=config, run=run, registry=registry, raw=raw,
                    model=model, opt=opt, opt_cfg_scaled=opt_cfg_scaled, fresh=fresh,
                    loss_fn=loss_fn, loader=loader, val_batches=val_batches, idxs=idxs, wb=wb,
                    train_ds=train_ds, start_it=start_it, wrap=wrap)
    loop.record({'iter': 0, 'mix': train_ds.mix(), 'n_windows': len(train_ds),
                 'n_keypoints': registry.n_keypoints})
    loop.run_loop()


if __name__ == '__main__':
    main()
