"""Train the track-quality scorer.

Cloned from `train.py`'s shape but NOT from its loop: the objective is a triplet ranking loss, the
best checkpoint is selected on held-out synthetic triplet accuracy rather than on MPJPE, and the
staged encoder unfreeze, the pose losses and the pose eval all have no meaning here. What IS
reused, by import, is the parts that must not diverge: `checkpoints.load_config` (so the scorer
family layers over `configs/scorer.toml`), `checkpoints.warm_start` (the pose checkpoint),
`checkpoints.save_checkpoint` / `save_run_meta` with `kind='scorer'`, and `train.build_optimizer`
(so Muon routing and the fresh-parameter rule are one implementation).

    pixi run python scripts/train_scorer.py --config configs/scorer-3dpop.toml --data <root> \
        --out <results>/scorers/scorer-3dpop --checkpoint <pose run folder>
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from posetail.datasets.scorer_corruption import GENERATORS
from posetail.posetail.losses_scorer import TripletScorerLoss

from ..checkpoints import (_SCORER_CONFIG, full_training_state, load_config, save_checkpoint,
                           save_run_meta, warm_start)
from ..dataset import LoaderConfig, PoseDataset
from ..format import Registry
from ..train import build_optimizer, init_wandb, log
from .dataset import ScorerDataset, scorer_collate, triplet_to_device
from .model import build_scorer
from .triplet import seed_worker

SCORER_HEAD_KEYS = ('pool_num_heads', 'score_hidden', 'use_precision')


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
    known = set(LoaderConfig.__dataclass_fields__) | {'path', 'num_workers'}
    unknown = set(data_cfg) - known
    if unknown:
        raise SystemExit(
            f'[data]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at the defaults and report as the arm it is not. Known keys: {sorted(known)}')
    lc = LoaderConfig(**{k: v for k, v in data_cfg.items()
                         if k in LoaderConfig.__dataclass_fields__})
    return replace(lc, box_prompt=model_cfg.get('box_prompt', 'none'))


def scorer_kwargs(config: dict) -> dict:
    """The scorer's own head settings out of `[scorer]`.

    Inputs: config -- the merged run config.
    Outputs: a dict with exactly `SCORER_HEAD_KEYS`.
    Side effects: none.
    """
    cfg = dict(config.get('scorer', {}))
    cfg.pop('corruption', None)
    return {k: cfg[k] for k in SCORER_HEAD_KEYS if k in cfg}


def corruption_config(config: dict) -> dict:
    """The `[scorer.corruption]` block, with `min_valid_frames` folded in from `[scorer]`.

    `min_valid_frames` lives on `[scorer]` and is consumed by the corruption path, so it is copied
    down rather than looked up twice.

    Inputs: config -- the merged run config.
    Outputs: the corruption dict.
    Side effects: none.
    """
    cfg = dict(config['scorer'].get('corruption', {}))
    cfg['min_valid_frames'] = int(config['scorer'].get('min_valid_frames', 6))
    return cfg


def build_datasets(config: dict, registry_base: Registry | None):
    """(train, val) `ScorerDataset`s, with val None when the root has no val split.

    Inputs: config -- the merged run config; registry_base -- a registry to append to, or None.
    Outputs: (train_dataset, val_dataset_or_None, registry).
    Side effects: reads the dataset root; prints the window counts and the source mix.
    """
    data_cfg = config['data']
    lc = loader_config(data_cfg, config['model'])
    corr = corruption_config(config)
    train_base = PoseDataset(data_cfg['path'], 'train', lc, registry_base=registry_base)
    registry = train_base.registry
    train_ds = ScorerDataset(train_base, corr)
    print(f'train: {len(train_ds)} windows, {registry.n_keypoints} keypoints')
    print('train: mix ' + '  '.join(f'{k}={v:.1%}' for k, v in train_base.mix().items()))

    root = Path(data_cfg['path'])
    has_val = (root / 'val').is_dir() or any(
        (c / 'val').is_dir() for c in root.iterdir() if c.is_dir())
    if not has_val:
        return train_ds, None, registry
    val_base = PoseDataset(data_cfg['path'], 'val', lc, registry=registry)
    return train_ds, ScorerDataset(val_base, corr), registry


def _loaders(train_ds, val_ds, config: dict, seed: int):
    """DataLoaders for train and (optionally) val.

    `batch_size` is structurally 1: each camera's rotated crop has its own size, so there is no
    batch axis to stack along. Train shuffles with replacement-free sampling and entropy-seeded
    workers; val is a fixed enumeration so the held-out number is the same set every time.

    Inputs: train_ds / val_ds -- `ScorerDataset`s; config -- the merged run config; seed -- the run
            seed.
    Outputs: (train_loader, val_loader_or_None).
    Side effects: forks worker processes.
    """
    nw = int(config['data'].get('num_workers', 8))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=scorer_collate,
                              num_workers=nw, prefetch_factor=2 if nw else None,
                              persistent_workers=bool(nw), pin_memory=True,
                              worker_init_fn=seed_worker)
    if val_ds is None:
        return train_loader, None
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=scorer_collate,
                            num_workers=nw, prefetch_factor=2 if nw else None,
                            persistent_workers=bool(nw), pin_memory=True,
                            worker_init_fn=seed_worker)
    return train_loader, val_loader


def _per_type_accuracy(scores: torch.Tensor, fired: torch.Tensor) -> dict:
    """`P(triplet_acc)` broken out per corruption type.

    An aggregate hides the failure this breakdown exists to catch: one corruption type at chance
    means that type is invisible and its magnitude is wrong, even when the headline accuracy looks
    healthy. Columns with no fired point are omitted rather than reported as 0.

    Inputs: scores -- [K,3] (good, bad, anchor); fired -- [K, len(GENERATORS)] bool.
    Outputs: {'val/acc_<type>': float} for every type that fired at least once.
    Side effects: none.
    """
    correct = (scores[:, 0] > scores[:, 1])
    out = {}
    for i, name in enumerate(GENERATORS):
        mask = fired[:, i]
        if bool(mask.any()):
            out[f'val/acc_{name}'] = float(correct[mask].float().mean())
    return out


def evaluate(model, loader, loss_fn, device, max_batches: int) -> dict:
    """The held-out synthetic pass: loss history plus per-type triplet accuracy.

    Inputs: model -- the scorer; loader -- the val loader; loss_fn -- a `TripletScorerLoss`
            (its history is collapsed and reset by the caller); device -- where to run;
            max_batches -- how many triplets to score.
    Outputs: {'val/<metric>': float} including the per-type accuracies.
    Side effects: switches the model to eval and back to train.
    """
    model.eval()
    per_type: dict[str, list] = {}
    seen = 0
    with torch.no_grad():
        for trip in loader:
            if trip is None:
                continue
            trip = triplet_to_device(trip, device)
            scores, precision, labels = model.score_triplet(trip)
            loss_fn(scores, precision, labels)
            for k, v in _per_type_accuracy(scores, trip['fired'][0]).items():
                per_type.setdefault(k, []).append(v)
            seen += 1
            if seen >= max_batches:
                break
    model.train()
    out = {k: float(np.mean(v)) for k, v in per_type.items()}
    out['val/n_scored'] = float(seen)
    return out


def run(config_path, data_path, out: Path, checkpoint: str | None, device,
        max_iterations: int | None, no_wandb: bool) -> None:
    """Train one scorer run.

    The per-step training metrics are averaged over the last `print_freq` steps before printing,
    because ONE step is one window: with K keypoints that is K triplets (7 on calms21), so a
    single step's `triplet_acc` is quantised in units of 1/K and swings over the full range while
    the model learns. The val pass is the number to read; this is the trace.

    Inputs: config_path -- a config layering over `configs/scorer.toml`; data_path -- the dataset
            root (overrides `[data].path`); out -- the run folder; checkpoint -- a pose run folder
            or checkpoint to warm-start from, or a scorer run folder's checkpoint to RESUME from;
            device -- torch device; max_iterations -- override `[training].n_iterations`;
            no_wandb -- skip wandb.
    Outputs: none.
    Side effects: writes the run folder, checkpoints and wandb logs.
    """
    config = load_config(config_path, base=_SCORER_CONFIG)
    if data_path:
        config['data'] = {**config['data'], 'path': data_path}
    train_cfg = config['training']
    seed = int(train_cfg.get('seed', 23))
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = checkpoint or train_cfg.get('checkpoint_path') or None
    base_reg = None
    if ckpt_path:
        ref = Path(ckpt_path)
        ref = ref.parent.parent if ref.is_file() else ref
        if (ref / 'keypoint_registry.toml').exists():
            base_reg = Registry.load(ref / 'keypoint_registry.toml')
            print(f'keypoint registry: appending to {ref / "keypoint_registry.toml"}')

    train_ds, val_ds, registry = build_datasets(config, base_reg)
    model = build_scorer({**config['model'], 'video_encoder_pretrained': False},
                         registry.n_keypoints, **scorer_kwargs(config)).to(device)

    resume_state = None
    fresh: set[str] = set()
    if ckpt_path:
        ckpt_file = Path(ckpt_path)
        if ckpt_file.is_dir():
            from ..checkpoints import resolve_checkpoint
            ckpt_file = resolve_checkpoint(ckpt_file / 'checkpoints')
        loaded = torch.load(ckpt_file, map_location='cpu', weights_only=False)
        if loaded.get('kind', 'pose') == 'scorer' and full_training_state(loaded):
            resume_state = loaded
            print(f'resume: {ckpt_file} at iteration {loaded.get("iteration", "?")}')
        else:
            fresh = warm_start(model, ckpt_file, base_names=tuple(registry.names))
    fresh = set(fresh) | {n for n, _ in model.named_parameters()
                          if n.startswith(('attn_pool.', 'score_', 'missing_point',
                                           'precision_head'))}

    optimizer = build_optimizer(model, fresh, config['training']['optimizer'])
    if resume_state is not None:
        model.load_state_dict(resume_state['model_state'], strict=False)
        try:
            optimizer.load_state_dict(resume_state['optimizer_state'])
        except (KeyError, ValueError) as e:
            print(f'resume: optimizer state not reusable ({e}); continuing without it')

    loss_fn = TripletScorerLoss(margin=float(config['scorer'].get('triplet_margin', 0.25)),
                                precision_reg_weight=float(
                                    config['scorer'].get('precision_reg_weight', 0.01)),
                                score_reg_weight=float(
                                    config['scorer'].get('score_reg_weight', 0.0)))
    val_loss_fn = TripletScorerLoss(margin=float(config['scorer'].get('triplet_margin', 0.25)),
                                    precision_reg_weight=float(
                                        config['scorer'].get('precision_reg_weight', 0.01)),
                                    score_reg_weight=float(
                                        config['scorer'].get('score_reg_weight', 0.0)))

    save_run_meta(out, config, registry, kind='scorer')
    wb = None if no_wandb else init_wandb(config, out)
    train_loader, val_loader = _loaders(train_ds, val_ds, config, seed)

    n_iter = int(max_iterations or train_cfg.get('n_iterations', 10000))
    val_freq = int(train_cfg.get('val_freq', 200))
    val_batches = int(train_cfg.get('val_batches', 20))
    ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
    print_freq = int(train_cfg.get('print_freq', 20))
    max_norm = float(train_cfg.get('max_grad_norm', 10.0))
    start_iter = int(resume_state.get('iteration', 0)) if resume_state else 0

    model.train()
    if hasattr(optimizer, 'train'):
        optimizer.train()
    step = 0
    t0 = time.time()
    window: dict[str, list] = {}
    while step < n_iter:
        for trip in train_loader:
            if step >= n_iter:
                break
            if trip is None:
                continue
            trip = triplet_to_device(trip, device)
            optimizer.zero_grad(set_to_none=True)
            scores, precision, labels = model.score_triplet(trip)
            total = loss_fn(scores, precision, labels)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            iteration = start_iter + step
            hist = loss_fn.collapse_history(prefix='')
            loss_fn.reset_history()
            for k, v in hist.items():
                window.setdefault(k, []).append(v)
            values = {f'train/{k}': float(np.mean(vs)) for k, vs in window.items()}
            if step % print_freq == 0:
                window.clear()

            if val_loader is not None and step % val_freq == 0:
                values.update(evaluate(model, val_loader, val_loss_fn, device, val_batches))
                values.update(val_loss_fn.collapse_history(prefix='val/'))
                val_loss_fn.reset_history()
                if hasattr(optimizer, 'train'):
                    optimizer.train()
                print(f'[{iteration}] ' + '  '.join(
                    f'{k}={v:.4g}' for k, v in values.items() if k.startswith('val/')))

            if wb is not None:
                log(wb, values, iteration)
            if step % print_freq == 0:
                el = time.time() - t0
                print(f'[{iteration}] last {print_freq} steps: '
                      f'loss={values.get("train/scorer_loss", float("nan")):.4g} '
                      f'acc={values.get("train/triplet_acc", float("nan")):.3f} '
                      f'gap={values.get("train/score_gap", float("nan")):.4g} '
                      f'({el:.0f}s)')

            if step % ckpt_freq == 0 or step + 1 == n_iter:
                save_checkpoint(out, iteration, model, optimizer, config, registry=registry,
                                kind='scorer')
            step += 1
    if wb is not None:
        wb.finish()
    print(f'done: {n_iter} iterations into {out}')


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Inputs: argv -- argument list, or None for `sys.argv`.
    Outputs: a process exit code.
    Side effects: trains and writes the run folder.
    """
    parser = argparse.ArgumentParser(description='Train the track-quality scorer.')
    parser.add_argument('--config', required=True, help='a config over configs/scorer.toml')
    parser.add_argument('--data', default=None, help='dataset root, overriding [data].path')
    parser.add_argument('--out', required=True, help='the run folder, e.g. scorers/scorer-3dpop')
    parser.add_argument('--checkpoint', default=None,
                        help='a pose run folder/checkpoint to warm-start from, or a scorer '
                             'checkpoint to resume from')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args(argv)
    run(args.config, args.data, Path(args.out), args.checkpoint, args.device,
        args.iterations, args.no_wandb)
    return 0
