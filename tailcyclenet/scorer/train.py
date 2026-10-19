"""Train the track-quality scorer.

Cloned from `train.py`'s shape but NOT from its loop: the objective is a triplet ranking loss, the
best checkpoint is selected on held-out synthetic triplet accuracy rather than on MPJPE, and the
staged encoder unfreeze, the pose losses and the pose eval all have no meaning here. What IS
reused, by import, is the parts that must not diverge: `config.load_config` (so the scorer
family layers over `configs/scorer.toml`), `pose.checkpoint.warm_start` (the pose checkpoint),
`scorer.checkpoint.save_scorer_checkpoint` / `save_run_meta` with `kind='scorer'`, and
`optim.build_optimizer` (so Muon routing and the fresh-parameter rule are one implementation).

    pixi run python scripts/train_scorer.py --config configs/scorer.toml --data <root> \
        --out <results>/scorers/scorer-3dpop --checkpoint <pose run folder>
"""
from __future__ import annotations

import argparse
import time
import tomllib
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checkpoints import full_training_state, prior_provenance, save_run_meta
from .checkpoint import save_scorer_checkpoint
from .contract import (require_scorer_contract, require_scorer_granularity,
                       scorer_checkpoint_granularity, scorer_contract_mismatches)
from .losses import build_scorer_loss
from ..config import SCORER_CONFIG, load_config
from ..dataset import SCORER_PLUMBING, LoaderConfig, PoseDataset, StepSampler, loader_config_from
from ..format import Registry
from ..memory import peak_gb as _cpu_peak_gb
from ..pose.checkpoint import warm_start
from ..pose.optim import build_optimizer, check_resume
from ..runtime import gpu_peak_gb, init_wandb, launch, log, timed
from .. import distributed as dist_utils
from .config import (corruption_config, resolve_selection_metric, scorer_kwargs,
                     validate_scorer_config)
from .dataset import ScorerDataset, scorer_collate, triplet_to_device
from .evaluate import evaluate, gather_eval, scorer_loss
from .model import build_scorer
from .qc import refuse_deleted_data_keys
from .triplet import seed_worker


def warm_start_names(base_reg) -> tuple[str, ...] | None:
    """The registry `warm_start` must be given: the SOURCE run's, never this run's grown one.

    `warm_start` copies a checkpoint's identity table row-for-row only when the name list it is
    handed has exactly the table's length; that check is what keeps a row on its own keypoint. The
    grown registry (source names plus this dataset's appended ones) is longer than the table, so
    passing it refuses the copy and reinitialises EVERY row -- including the source rows that
    should have been preserved. Same call as the pose trainer's.

    Inputs: base_reg -- the registry read from the source run folder, or None when it has none.
    Outputs: its names, or None (then no copy is attempted, which is the correct refusal).
    Side effects: none.
    """
    return tuple(base_reg.names) if base_reg is not None else None


def loader_config(data_cfg: dict, model_cfg: dict) -> LoaderConfig:
    """`[data]` -> a `LoaderConfig`, refusing unknown keys and forcing the model's box setting.

    An unknown key would otherwise be ignored and the run would report as an arm it is not -- the
    same refusal `train.py` makes, for the same reason. `box_prompt` is taken from `[model]`
    because the loader and the model must agree on it; the scorer's config sets it to `none` so no
    box is ever computed.

    Inputs: data_cfg -- the `[data]` block; model_cfg -- the `[model]` block.
    Outputs: a `LoaderConfig`.
    Side effects: none, or raises SystemExit naming the unknown keys.
    """
    data_cfg = dict(data_cfg)
    refuse_deleted_data_keys(data_cfg)
    lc = loader_config_from(data_cfg, extra=SCORER_PLUMBING)
    return replace(lc, box_prompt=model_cfg.get('box_prompt', 'none'))


def build_datasets(config: dict, registry_base: Registry | None, *, rank: int = 0,
                   world_size: int = 1):
    """(train, val) `ScorerDataset`s, with val None when the root has no val split.

    Inputs: config -- the merged run config; registry_base -- a registry to append to, or None.
    Outputs: (train_dataset, val_dataset_or_None, registry).
    Side effects: reads the dataset root; prints the window counts and the source mix.

    Validation samples use `val_cams_to_sample`, a fixed camera count, rather than the training
    camera-count draw, so held-out windows have comparable difficulty.
    """
    data_cfg = config['data']
    loader_data = {**data_cfg, 'two_d_sampling': config['scorer'].get(
        'two_d_sampling', 'true-2d-hybrid-v1')}
    lc = loader_config(loader_data, config['model'])
    corr = corruption_config(config)
    train_base = PoseDataset(data_cfg['path'], 'train', lc, registry_base=registry_base,
                             rank=rank, world_size=world_size)
    registry = train_base.registry
    train_ds = ScorerDataset(train_base, corr)
    print(f'train: {len(train_ds)} windows, {registry.n_keypoints} keypoints')
    print('train: mix ' + '  '.join(f'{k}={v:.1%}' for k, v in train_base.mix().items()))

    root = Path(data_cfg['path'])
    has_val = (root / 'val').is_dir() or any(
        (c / 'val').is_dir() for c in root.iterdir() if c.is_dir())
    if not has_val:
        return train_ds, None, registry
    val_lc = replace(lc, cams_to_sample=lc.val_cams_to_sample)
    val_base = PoseDataset(data_cfg['path'], 'val', val_lc, registry=registry,
                           rank=rank, world_size=world_size)
    return train_ds, ScorerDataset(val_base, corr), registry


def scorer_worker_init(worker_id: int, *, cv_threads: int = 2):
    """Initialise a scorer worker without disabling useful OpenCV parallelism.

    NumPy's scorer RNG retains the existing worker-specific seed.  Torch's intra-op pool is
    limited to one thread because each worker already has its own decode/transform work.  OpenCV
    remains parallel at two threads by default (rather than being forced to one); callers can set
    ``worker_cv_threads=0`` to leave OpenCV's own default untouched or choose another small value.
    """
    seed_worker(worker_id)
    torch.set_num_threads(1)
    cv_threads = int(cv_threads)
    if cv_threads < 0:
        raise ValueError(f'worker_cv_threads must be >= 0, got {cv_threads}')
    if cv_threads:
        import cv2
        cv2.setNumThreads(cv_threads)


def _loaders(train_ds, val_ds, config: dict, seed: int, *, world: int = 1,
             rank: int = 0, num_samples: int | None = None,
             val_indices: list[int] | None = None):
    """Build rank-local loaders; one triplet (batch=1) is the per-rank batch.

    Distributed training uses independent rank-seeded replacement streams, giving the world-size
    batch without replaying a shuffle permutation. Validation receives a deterministic strided
    shard; callers gather its sufficient metrics before selecting a checkpoint. Train and
    validation worker counts are separate because validation is small but should remain asynchronous.
    """
    data_cfg = config['data']
    nw = int(data_cfg.get('num_workers', 2))
    val_nw = int(data_cfg.get('val_num_workers', 1))
    prefetch = int(data_cfg.get('prefetch_factor', 1))
    cv_threads = int(data_cfg.get('worker_cv_threads', 2))
    if nw < 0 or val_nw < 0:
        raise ValueError(f'worker counts must be >= 0, got train={nw}, val={val_nw}')
    if prefetch < 1:
        raise ValueError(f'prefetch_factor must be >= 1, got {prefetch}')
    if cv_threads < 0:
        raise ValueError(f'worker_cv_threads must be >= 0, got {cv_threads}')
    worker_init = partial(scorer_worker_init, cv_threads=cv_threads)
    kwargs = dict(batch_size=1, collate_fn=scorer_collate, num_workers=nw,
                  prefetch_factor=prefetch if nw else None, persistent_workers=bool(nw),
                  pin_memory=(world == 1), worker_init_fn=worker_init)
    if world > 1:
        gen = torch.Generator().manual_seed(int(seed) + int(rank))
        kwargs['sampler'] = StepSampler(len(train_ds), int(num_samples), generator=gen)
    else:
        kwargs['shuffle'] = True
    train_loader = DataLoader(train_ds, **kwargs)
    if val_ds is None:
        return train_loader, None
    if val_indices is not None:
        val_ds = torch.utils.data.Subset(val_ds, val_indices)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=scorer_collate,
                            num_workers=val_nw, prefetch_factor=prefetch if val_nw else None,
                            persistent_workers=bool(val_nw), pin_memory=(world == 1),
                            worker_init_fn=worker_init)
    return train_loader, val_loader


def scorer_checkpoint_mode_mismatch(requested: str, checkpoint: dict) -> bool:
    """Whether a scorer checkpoint must take the weights-only warm-start branch."""
    return scorer_checkpoint_granularity(checkpoint) != str(requested).lower()


def require_progress(stepped_this_epoch: int, stepped_this_run: int, step: int,
                     local_target: int) -> None:
    """Refuse an epoch that took no optimizer step, or a run that reached its budget without one.

    A rejected triplet or a non-finite loss/gradient still advances `step`, so the step counter
    alone cannot tell an all-rejecting recipe from a training one. An epoch cut short by the
    budget is excused unless the whole run never stepped.
    """
    if stepped_this_epoch or (step >= local_target and stepped_this_run):
        return
    raise RuntimeError(
        'scorer DataLoader produced no successful triplets in an epoch; all samples were '
        'rejected. Check corruption magnitudes, distance thresholds, segment settings, '
        'visibility, and the training split instead of retrying forever.')


def _scorer_config(config_path, out: Path, fresh: bool, data_path, num_workers) -> tuple[dict, str]:
    """The layered scorer config with the CLI overrides applied -> (config, output_granularity).

    A run folder whose config predates `output_granularity` is a legacy sequence-mode run.
    """
    config = load_config(config_path, base=SCORER_CONFIG)
    prior_config = out / 'config.toml'
    if prior_config.exists() and not fresh:
        with prior_config.open('rb') as f:
            prior = tomllib.load(f)
        if 'output_granularity' not in prior.get('scorer', {}):
            config['scorer'] = {**config.get('scorer', {}), 'output_granularity': 'sequence'}
            print(f'{prior_config}: absent scorer output_granularity -> legacy sequence mode')
    output_granularity = validate_scorer_config(config)
    config['scorer'] = {**config.get('scorer', {}),
                        'output_granularity': output_granularity}
    if data_path:
        config['data'] = {**config['data'], 'path': data_path}
    if num_workers is not None:
        config['data'] = {**config['data'], 'num_workers': int(num_workers)}
    return config, output_granularity


def _start_reference(out: Path, checkpoint, train_cfg: dict, fresh: bool):
    """(checkpoint path or None, the registry to append to or None): own folder first."""
    resumed = out / 'checkpoints' / 'checkpoint_last.pth'
    if resumed.exists() and not fresh:
        ckpt_path = str(resumed)
        registry_ref = out
        print(f'resuming this run folder: {resumed}')
    else:
        ckpt_path = checkpoint or train_cfg.get('checkpoint_path') or None
        registry_ref = None
    base_reg = None
    ref = Path(ckpt_path) if ckpt_path else registry_ref
    if ref is not None:
        ref = ref.parent.parent if ref.is_file() else ref
        if (ref / 'keypoint_registry.toml').exists():
            base_reg = Registry.load(ref / 'keypoint_registry.toml')
            print(f'keypoint registry: appending to {ref / "keypoint_registry.toml"}')
    return ckpt_path, base_reg


def _probe_checkpoint(ckpt_path, config: dict, output_granularity: str) -> tuple[bool, bool]:
    """(contract mismatch, mode mismatch) of a scorer start checkpoint; either forces a warm start."""
    checkpoint_contract_mismatch = False
    checkpoint_mode_mismatch = False
    if ckpt_path:
        probe = Path(ckpt_path)
        if probe.is_dir():
            from ..checkpoints import resolve_checkpoint
            probe = resolve_checkpoint(probe / 'checkpoints')
        if probe.exists():
            checkpoint_probe = torch.load(probe, map_location='cpu', weights_only=False)
            if checkpoint_probe.get('kind', 'pose') == 'scorer':
                checkpoint_mode = scorer_checkpoint_granularity(checkpoint_probe)
                checkpoint_mode_mismatch = scorer_checkpoint_mode_mismatch(output_granularity, checkpoint_probe)
                if checkpoint_mode_mismatch:
                    if full_training_state(checkpoint_probe):
                        print(f'{probe}: scorer mode {checkpoint_mode!r} differs from requested '
                              f'{output_granularity!r}; refusing full-state resume, using weights-only '
                              'warm start')
                    else:
                        print(f'{probe}: legacy scorer mode {checkpoint_mode!r} differs from '
                              f'{output_granularity!r}; allowing weights-only warm start')
                else:
                    try:
                        require_scorer_contract(config, checkpoint_probe, where=str(probe))
                    except ValueError as e:
                        checkpoint_contract_mismatch = True
                        print(f'{probe}: {e}; using warm-start weights')
    return checkpoint_contract_mismatch, checkpoint_mode_mismatch


def _started_loaders(train_ds, val_ds, config: dict, seed: int, n_target: int, fabric, world: int,
                     is0: bool):
    """(train, val) loaders, with their workers started before the model allocates CUDA memory."""
    val_batches = int(config['training'].get('val_batches', 20))
    val_indices = None
    if val_ds is not None:
        n_val = min(dist_utils.per_rank(val_batches, world), len(val_ds))
        val_indices = ([int(i) for i in np.unique(
            np.linspace(0, len(val_ds) - 1, n_val).round().astype(int))]
                       if n_val else [])
    train_loader, val_loader = _loaders(
        train_ds, val_ds, config, seed, world=world, rank=(fabric.global_rank if fabric else 0),
        num_samples=dist_utils.ceil_div(n_target, world), val_indices=val_indices)
    if train_loader.num_workers or (val_loader is not None and val_loader.num_workers):
        iter(train_loader)
        if val_loader is not None and val_loader.num_workers:
            iter(val_loader)
        if is0:
            print(f'loader workers: train={train_loader.num_workers} '
                  f'val={val_loader.num_workers if val_loader is not None else 0} '
                  f'prefetch={config["data"].get("prefetch_factor", 1)} '
                  '(started before model CUDA allocation)')
    return train_loader, val_loader


def _resume_candidate(model, ckpt_path, output_granularity: str, base_reg,
                      contract_mismatch: bool, mode_mismatch: bool):
    """(resume_state, resume_file, fresh names): a matching full scorer state, else a warm start."""
    resume_state = None
    resume_file = None
    fresh: set[str] = set()
    if ckpt_path:
        ckpt_file = Path(ckpt_path)
        if ckpt_file.is_dir():
            from ..checkpoints import resolve_checkpoint
            ckpt_file = resolve_checkpoint(ckpt_file / 'checkpoints')
        loaded = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        if (loaded.get('kind', 'pose') == 'scorer' and full_training_state(loaded)
                and not contract_mismatch and not mode_mismatch):
            require_scorer_granularity(output_granularity,
                                       scorer_checkpoint_granularity(loaded),
                                       where=str(ckpt_file))
            resume_state = loaded
            resume_file = ckpt_file
            print(f'resume candidate: {ckpt_file} at iteration {loaded.get("iteration", "?")} '
                  f'(output_granularity={output_granularity})')
        else:
            if (loaded.get('kind', 'pose') == 'scorer'
                    and not mode_mismatch):
                require_scorer_granularity(output_granularity,
                                           scorer_checkpoint_granularity(loaded),
                                           where=str(ckpt_file))
            fresh = warm_start(model, ckpt_file, base_names=warm_start_names(base_reg))
    fresh = set(fresh) | {n for n, _ in model.named_parameters()
                          if n.startswith(('attn_pool.', 'score_', 'missing_point',
                                           'precision_head', 'frame_loss.'))}
    return resume_state, resume_file, fresh


def _optimizer_and_resume(model, fresh: set, opt_cfg: dict, resume_state, resume_file, base_reg):
    """(optimizer, resume_state): a resume that does not load falls back to a clean warm start."""
    optimizer = build_optimizer(model, fresh, opt_cfg)
    if resume_state is not None:
        try:
            model.load_state_dict(resume_state['model_state'], strict=True)
            verdict = check_resume(optimizer, resume_state['optimizer_state'])
            if not verdict:
                raise ValueError(verdict.reason)
            optimizer.load_state_dict(resume_state['optimizer_state'])
            print(f'resuming {resume_file} at iteration {resume_state.get("iteration", "?")}')
        except (KeyError, RuntimeError, ValueError, SystemExit) as e:
            print(f'resume: {resume_file} is not reusable ({e}); warm-starting at iteration 0')
            fresh.update(warm_start(model, resume_file, base_names=warm_start_names(base_reg)))
            optimizer = build_optimizer(model, fresh, opt_cfg)
            resume_state = None
    return optimizer, resume_state


def _prior_best(out: Path, config: dict, selection_metric: str, resuming: bool, fabric, is0: bool):
    """(best value, best iteration, locked) from a resumed run's `checkpoint_best.pth`.

    A best written under a different contract is discarded; one with no comparable value is left
    untouched and locks the best slot for this run.
    """
    best_acc = float('-inf')
    best_iter = -1
    best_locked = False
    prior_best = out / 'checkpoints' / 'checkpoint_best.pth'
    if resuming and prior_best.exists():
        prior_best_payload = torch.load(prior_best, map_location='cpu', weights_only=False)
        prior_contract_ok = not scorer_contract_mismatches(config, prior_best_payload)
        if (prior_contract_ok
                and prior_best_payload.get('scorer_selection_metric') == selection_metric
                and np.isfinite(prior_best_payload.get('scorer_selection_value', np.nan))):
            best_acc = float(prior_best_payload['scorer_selection_value'])
            best_iter = int(prior_best_payload.get('iteration', -1))
            print(f'prior best: val/{selection_metric} {best_acc:.4f} at iteration {best_iter}')
        elif not prior_contract_ok:
            if is0:
                prior_best.unlink(missing_ok=True)
            if fabric is not None:
                fabric.barrier()
            if is0:
                print(f'prior best {prior_best} uses a different scorer contract; discarding it')
        else:
            best_locked = True
            print(f'prior best {prior_best} has no comparable {selection_metric} value; '
                  'leaving it untouched')
    return best_acc, best_iter, best_locked


class ScorerLoop:
    """The scorer's step loop for one run; `run_loop()` trains to this rank's local target.

    A synchronized DDP step advances the headline iteration by `world`; `step` counts completed
    local steps (a rejected or non-finite triplet still advances it).
    """

    def __init__(self, *, fabric, world, is0, device, config, out, registry, train_loader,
                 val_loader, raw_model, wrap, optimizer, loss_fn, val_loss_fn,
                 output_granularity, wb, start_iter, n_target, best):
        """Bind the run; the frequencies become per-rank local step counts."""
        train_cfg = config['training']
        self.fabric, self.world, self.is0, self.device = fabric, world, is0, device
        self.config, self.out, self.registry, self.wb = config, out, registry, wb
        self.train_loader, self.val_loader = train_loader, val_loader
        self.raw_model, self.wrap, self.optimizer = raw_model, wrap, optimizer
        self.model = fabric.setup_module(raw_model) if wrap else raw_model
        self.loss_fn, self.val_loss_fn = loss_fn, val_loss_fn
        self.output_granularity = output_granularity
        val_freq = int(train_cfg.get('val_freq', 200))
        ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
        self.print_freq = int(train_cfg.get('print_freq', 20))
        self.max_norm = float(train_cfg.get('max_grad_norm', 10.0))
        self.step = dist_utils.ceil_div(start_iter, world)
        self.skipped = 0
        self.stepped_this_run = 0
        self.local_val_freq = dist_utils.per_rank(val_freq, world) if val_freq else 0
        self.local_ckpt_freq = dist_utils.per_rank(ckpt_freq, world)
        self.local_print_freq = dist_utils.per_rank(self.print_freq, world)
        self.local_target = dist_utils.ceil_div(n_target, world)
        self.waited, self.evalled, self.ckpted = [0.0], [0.0], [0.0]
        self.window: dict[str, list] = {}
        self.selection_metric = resolve_selection_metric(config, output_granularity)
        self.best_acc, self.best_iter, self.best_locked = best

    def selected_value(self, values: dict) -> float | None:
        """Read the explicitly configured validation metric, never infer it from the loss."""
        candidates = [f'val/{self.selection_metric}', self.selection_metric]
        if self.output_granularity == 'frame' and self.selection_metric == 'triplet_acc':
            candidates.insert(0, 'val/active_triplet_acc')
        for key in candidates:
            value = values.get(key)
            if value is not None and np.isfinite(value):
                return float(value)
        return None

    def run_loop(self) -> None:
        """Step until the local target; every pass over the loader must step at least once."""
        self.model.train()
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
        self.t0 = time.time()
        while self.step < self.local_target:
            stepped_this_epoch = 0
            for trip in timed(self.train_loader, self.waited):
                if self.step >= self.local_target:
                    break
                grad_norm = self.train_step(trip)
                if grad_norm is None:
                    continue
                stepped_this_epoch += 1
                self.after_step(grad_norm)
            require_progress(stepped_this_epoch, self.stepped_this_run, self.step,
                             self.local_target)

    def train_step(self, trip):
        """One optimizer step -> its grad norm, or None when every rank skipped it."""
        fabric, optimizer = self.fabric, self.optimizer
        accepted = dist_utils.all_ranks_finite(fabric, trip is not None)
        if not accepted:
            self.skipped += 1
            self.step += 1
            return None
        trip = triplet_to_device(trip, self.device)
        if self.wrap:
            scores, precision, labels = self.model(trip)
        else:
            scores, precision, labels = self.model.score_triplet(trip)
        total = scorer_loss(self.loss_fn, scores, precision, labels, trip,
                            self.output_granularity)
        if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(total))):
            optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            self.step += 1
            return None
        optimizer.zero_grad(set_to_none=True)
        if fabric is not None:
            fabric.backward(total)
        else:
            total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.max_norm)
        if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(grad_norm))):
            optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            self.step += 1
            return None
        optimizer.step()
        self.step += 1
        self.stepped_this_run += 1
        return grad_norm

    def after_step(self, grad_norm) -> None:
        """Window the train metrics, then validate, report and checkpoint on their cadences."""
        fabric, step = self.fabric, self.step
        iteration = step * self.world
        hist = self.loss_fn.collapse_history(prefix='')
        self.loss_fn.reset_history()
        for k, v in hist.items():
            self.window.setdefault(k, []).append(v)
        values = {f'train/{k}': float(np.mean(vs)) for k, vs in self.window.items()}
        if fabric is not None:
            values = {k: dist_utils.all_ranks_mean(fabric, v) for k, v in values.items()}
        if step % self.local_print_freq == 0:
            self.window.clear()
        if self.val_loader is not None and self.local_val_freq and step % self.local_val_freq == 0:
            self.validate(values, iteration)
        if step % self.local_print_freq == 0:
            self.report(values, iteration, grad_norm)
        elif self.wb is not None and self.is0:
            log(self.wb, values, iteration)
        if step % self.local_ckpt_freq == 0 or step == self.local_target:
            started = time.time()
            if fabric is not None:
                dist_utils.check_ranks_agree(fabric, self.raw_model)
            save_scorer_checkpoint(self.out, iteration, self.raw_model, self.optimizer,
                                   self.config, write=self.is0, registry=self.registry)
            if fabric is not None:
                fabric.barrier()
            self.ckpted[0] += time.time() - started

    def validate(self, values: dict, iteration: int) -> None:
        """The val pass into `values`; a better selected metric writes `checkpoint_best.pth`."""
        fabric, optimizer = self.fabric, self.optimizer
        started = time.time()
        val_values = evaluate(self.raw_model, self.val_loader, self.val_loss_fn, self.device,
                              len(self.val_loader), optimizer, self.output_granularity)
        val_values = gather_eval(fabric, val_values)
        values.update(val_values)
        self.evalled[0] += time.time() - started
        val_hist = self.val_loss_fn.collapse_history(prefix='val/')
        values.update(gather_eval(fabric, val_hist))
        self.val_loss_fn.reset_history()
        if hasattr(optimizer, 'train'):
            optimizer.train()
        acc = self.selected_value(values)
        if not self.best_locked and acc is not None and acc > self.best_acc:
            self.best_acc, self.best_iter = float(acc), iteration
            started = time.time()
            save_scorer_checkpoint(self.out, iteration, self.raw_model, optimizer, self.config,
                                   name='best', write=self.is0, registry=self.registry,
                                   selection_metric=self.selection_metric,
                                   selection_value=self.best_acc)
            if fabric is not None:
                fabric.barrier()
            self.ckpted[0] += time.time() - started
        if self.is0:
            print(f'[{iteration}] ' + '  '.join(
                f'{k}={v:.4g}' for k, v in values.items() if k.startswith('val/')))
            if acc is not None:
                print(f'[{iteration}] best val/{self.selection_metric} {self.best_acc:.4f} at '
                      f'iteration {self.best_iter}')

    def report(self, values: dict, iteration: int, grad_norm) -> None:
        """The print-interval line and W&B payload; then reset the interval clocks."""
        step = self.step
        wall = time.time() - self.t0
        elapsed = max(wall - self.evalled[0] - self.ckpted[0], 1e-9)
        report_steps = max(1, min(self.local_print_freq, step))
        dt = elapsed / report_steps
        wait_frac = self.waited[0] / elapsed
        eval_frac = self.evalled[0] / wall if wall > 0 else 0.0
        values.update({
            'train/iteration': iteration,
            'train/grad_norm': float(grad_norm),
            'train/gpu_peak_gb': gpu_peak_gb(self.device, reset=True),
            'train/cpu_peak_gb': _cpu_peak_gb(),
            'train/sec_per_it': dt,
            'train/loader_wait_frac': wait_frac,
            'train/eval_frac': eval_frac,
            'train/ckpt_frac': self.ckpted[0] / wall if wall > 0 else 0.0,
            'train/skipped_frac': self.skipped / max(step, 1),
            'train/world_size': self.world,
        })
        train_acc = values.get('train/active_triplet_acc',
                               values.get('train/triplet_acc', float('nan')))
        train_gap = values.get('train/active_score_gap',
                               values.get('train/score_gap', float('nan')))
        if self.is0:
            print(f'[{iteration}] last {self.print_freq} steps: '
                  f'loss={values.get("train/scorer_loss", float("nan")):.4g} '
                  f'acc={train_acc:.3f} gap={train_gap:.4g} '
                  f'({dt:.2f}s/it wait {wait_frac:.0%} eval {eval_frac:.0%})')
            if self.wb is not None:
                log(self.wb, values, iteration)
        self.t0 = time.time()
        self.waited[0] = self.evalled[0] = self.ckpted[0] = 0.0


def run(config_path, data_path, out: Path, checkpoint: str | None, device,
        max_iterations: int | None, no_wandb: bool, fresh: bool = False, fabric=None,
        num_workers: int | None = None, devices_arg: int | None = None) -> None:
    """Train one scorer run.

    The per-step training metrics are averaged over the last `print_freq` steps before printing,
    because ONE step is one window: with K keypoints that is K triplets (7 on calms21), so a
    single step's `triplet_acc` is quantised in units of 1/K and swings over the full range while
    the model learns. The val pass is the number to read; this is the trace. A synchronized DDP
    step advances the headline iteration by `world`; `step` counts completed local optimizer steps.

    Inputs: config_path -- a config layering over `configs/scorer.toml`; data_path -- the dataset
            root (overrides `[data].path`); out -- the run folder; checkpoint -- a pose run folder
            or checkpoint to warm-start from, or a scorer run folder's checkpoint to RESUME from;
            device -- torch device; max_iterations -- override `[training].n_iterations`;
            no_wandb -- skip wandb; fresh -- ignore this run folder's own checkpoint and start
            from the warm start, which is what `--fresh` is for.
    A warm start receives the source run's registry names, not this run's grown registry, so its
    identity rows remain keyed to the same keypoints.
    `[training].n_iterations` (and `--iterations`) is a TOTAL, not an increment: resuming a
    60000-iteration run whose checkpoint sits at 40000 runs 20000 more and stops at 60000. Without
    that, a re-submitted job would silently train past its own budget.

    Outputs: none.
    Side effects: writes the run folder, checkpoints and wandb logs.
    """
    world = int(fabric.world_size) if fabric is not None else 1
    is0 = bool(fabric.is_global_zero) if fabric is not None else True
    device = fabric.device if fabric is not None else device
    config, output_granularity = _scorer_config(config_path, out, fresh, data_path, num_workers)
    train_cfg = config['training']
    seed = int(train_cfg.get('seed', 23))
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path, base_reg = _start_reference(out, checkpoint, train_cfg, fresh)
    contract_mismatch, mode_mismatch = _probe_checkpoint(ckpt_path, config, output_granularity)

    train_ds, val_ds, registry = build_datasets(
        config, base_reg, rank=(fabric.global_rank if fabric is not None else 0),
        world_size=world)
    if fabric is not None:
        dist_utils.check_registry(fabric, registry.names)
    n_target = int(max_iterations or train_cfg.get('n_iterations', 10000))
    train_loader, val_loader = _started_loaders(train_ds, val_ds, config, seed, n_target,
                                                fabric, world, is0)

    model = build_scorer({**config['model'], 'video_encoder_pretrained': False},
                         registry.n_keypoints, **scorer_kwargs(config)).to(device)
    loss_fn = build_scorer_loss(config, output_granularity)
    if output_granularity == 'frame':
        model.add_module('frame_loss', loss_fn.to(device))
    resume_state, resume_file, fresh_names = _resume_candidate(
        model, ckpt_path, output_granularity, base_reg, contract_mismatch, mode_mismatch)

    opt_cfg = dist_utils.scale_optimizer_cfg(config['training']['optimizer'], world)
    if world > 1 and is0:
        print(f'lr: scaled by sqrt({world}) -> learning_rate {opt_cfg["learning_rate"]:g}'
              + (f', kpt_lr {opt_cfg["kpt_lr"]:g}' if 'kpt_lr' in opt_cfg else ''))
    optimizer, resume_state = _optimizer_and_resume(model, fresh_names, opt_cfg, resume_state,
                                                    resume_file, base_reg)

    val_loss_fn = build_scorer_loss(config, output_granularity).to(device)
    if output_granularity == 'frame':
        val_loss_fn.pointwise_log_scale = model.frame_loss.pointwise_log_scale

    prior_world = prior_provenance(out).get('world_size')
    if prior_world and int(prior_world) != world and fabric is not None:
        fabric.print(f'WARNING: run folder was written by {prior_world} rank(s), this run has '
                     f'{world}; iteration/sample mapping and sqrt(world) rates change.')
    if is0:
        save_run_meta(out, config, registry, kind='scorer', extra={
            'world_size': world, 'devices': str(devices_arg if devices_arg is not None else world),
            'lr_effective': float(opt_cfg['learning_rate']),
            'kpt_lr_effective': float(opt_cfg.get('kpt_lr', opt_cfg['learning_rate']))})
    if fabric is not None:
        fabric.barrier()
    wb = None if no_wandb or not is0 else init_wandb(config, out)
    start_iter = int(resume_state.get('iteration', 0)) if resume_state else 0
    n_iter = max(0, n_target - start_iter)
    resuming = (out / 'checkpoints' / 'checkpoint_last.pth').exists() and not fresh
    loop = ScorerLoop(
        fabric=fabric, world=world, is0=is0, device=device, config=config, out=out,
        registry=registry, train_loader=train_loader, val_loader=val_loader, raw_model=model,
        wrap=fabric is not None and world > 1, optimizer=optimizer, loss_fn=loss_fn,
        val_loss_fn=val_loss_fn, output_granularity=output_granularity, wb=wb,
        start_iter=start_iter, n_target=n_target,
        best=_prior_best(out, config, resolve_selection_metric(config, output_granularity),
                         resuming, fabric, is0))
    loop.run_loop()
    if wb is not None:
        wb.finish()
    if is0:
        if loop.best_iter >= 0:
            print(f'best: val/{loop.selection_metric} {loop.best_acc:.4f} at iteration '
                  f'{loop.best_iter} (checkpoints/checkpoint_best.pth; '
                  f'output_granularity={output_granularity})')
        print(f'done: {n_iter} iterations into {out}')


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Inputs: argv -- argument list, or None for `sys.argv`.
    Outputs: a process exit code.
    Side effects: trains and writes the run folder.
    """
    parser = argparse.ArgumentParser(prog='tailcyclenet train-scorer',
                                     description='Train the track-quality scorer.')
    parser.add_argument('--config', type=Path, default=None,
                        help='defaults to the packaged configs/scorer.toml (the shipped recipe, '
                             'unmodified) when omitted -- pass one that overlays it for anything '
                             'else')
    parser.add_argument('--data', default=None, help='dataset root, overriding [data].path')
    parser.add_argument('--out', required=True, help='the run folder, e.g. scorers/scorer-3dpop')
    parser.add_argument('--checkpoint', default=None,
                        help='a pose run folder/checkpoint to warm-start from, or a scorer '
                             'checkpoint to resume from')
    parser.add_argument('--device', default='cuda:0',
                        help='GPU for --devices 1, or cpu for the bounded CPU smoke')
    parser.add_argument('--devices', type=int, default=1,
                        help='number of devices/ranks; each rank contributes one triplet')
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--precision', default='32-true')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='loader workers per rank')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--fresh', action='store_true',
                        help="start from iteration 0 even if this run folder holds a "
                             "checkpoint_last")
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = SCORER_CONFIG
    fabric = launch(args)
    run(args.config, args.data, Path(args.out), args.checkpoint, args.device,
        args.iterations, args.no_wandb, args.fresh, fabric=fabric,
        num_workers=args.num_workers, devices_arg=args.devices)
    return 0
