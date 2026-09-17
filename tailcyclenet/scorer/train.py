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
import os
import sys
import time
import tomllib
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from posetail.datasets.scorer_corruption import GENERATORS
from posetail.posetail.losses_scorer import TripletScorerLoss

from ..checkpoints import (SCORER_OUTPUT_GRANULARITIES, _SCORER_CONFIG,
                           full_training_state, load_config, require_scorer_granularity,
                           save_checkpoint, save_run_meta, scorer_checkpoint_granularity,
                           scorer_contract_mismatches, scorer_output_granularity, warm_start,
                           prior_provenance)
from ..dataset import LoaderConfig, PoseDataset, StepSampler
from ..format import Registry
from ..memory import peak_gb as _cpu_peak_gb
from ..train import _gpu_peak_gb, build_optimizer, init_wandb, log
from .. import distributed as dist_utils
from .dataset import ScorerDataset, scorer_collate, triplet_to_device
from .model import build_scorer, flatten_sequence_triplet
from .triplet import SEGMENT_TYPES, seed_worker

SCORER_HEAD_KEYS = ('pool_num_heads', 'score_hidden', 'use_precision', 'output_granularity')

# Settings consumed by the scorer head/loss.  Keeping this list here gives configs the same typo
# refusal as the pose trainer: a misspelled loss arm must not silently run the historical default.
SCORER_CONFIG_KEYS = {
    *SCORER_HEAD_KEYS,
    'triplet_margin', 'precision_reg_weight', 'score_reg_weight', 'min_valid_frames',
    'pointwise_weight', 'pointwise_balance', 'pointwise_label_smoothing',
    'inactive_consistency_weight', 'anchor_consistency_weight', 'selection_metric',
    'loss_schema', 'corruption_mask_semantics', 'source_frame_duplicate_policy',
    'two_d_sampling',
}
FRAME_SELECTION_METRICS = {
    'triplet_acc', 'active_triplet_acc', 'score_gap', 'active_score_gap',
    'pointwise_auroc', 'pointwise_ap', 'localization', 'active_fraction',
}
SEQUENCE_SELECTION_METRICS = {'triplet_acc', 'score_gap'}


def resolve_selection_metric(config: dict, output_granularity: str) -> str:
    """Resolve a validation metric that the selected mode actually emits."""
    configured = str(config.get('scorer', {}).get(
        'selection_metric',
        'active_triplet_acc' if output_granularity == 'frame' else 'triplet_acc')).removeprefix('val/')
    if output_granularity == 'sequence':
        aliases = {'active_triplet_acc': 'triplet_acc', 'active_score_gap': 'score_gap'}
        configured = aliases.get(configured, configured)
        if configured not in SEQUENCE_SELECTION_METRICS:
            raise SystemExit(
                f'[scorer].selection_metric = {configured!r} is not emitted in sequence mode; '
                f'choose one of {sorted(SEQUENCE_SELECTION_METRICS)}')
    return configured


def resolve_output_granularity(config: dict) -> str:
    """Resolve and validate the scorer output mode for a merged/new config.

    New scorer configs inherit an explicit ``frame`` from the family base.  A config read directly
    from an old run folder is handled by :func:`checkpoints.scorer_output_granularity`, where an
    absent key means the legacy sequence path.  This small wrapper is kept in the trainer so unit
    tests and callers do not need to know which compatibility helper owns the default.
    """
    return scorer_output_granularity(config)


def validate_scorer_config(config: dict, *, legacy_run: bool = False) -> str:
    """Validate scorer-only settings and return the resolved output mode.

    The validation is intentionally performed before a dataset/model is built.  Frame-only terms
    in sequence mode are a semantic mismatch, not harmless unused knobs; likewise a sequence
    keypoint gate cannot be mixed into the framewise active mask.  ``legacy_run`` is useful when
    validating a pre-frame run folder whose absent mode has already been resolved to sequence.
    """
    scorer = dict(config.get('scorer', {}))
    mode = scorer_output_granularity(config)
    if legacy_run and 'output_granularity' not in scorer:
        mode = 'sequence'
    unknown = set(scorer) - SCORER_CONFIG_KEYS - {'corruption'}
    if unknown:
        raise SystemExit(
            f'[scorer]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at defaults and report as the arm it is not. Known keys: '
            f'{sorted(SCORER_CONFIG_KEYS | {"corruption"})}')
    if mode not in SCORER_OUTPUT_GRANULARITIES:
        raise SystemExit(f'[scorer].output_granularity must be one of '
                         f'{SCORER_OUTPUT_GRANULARITIES}, got {mode!r}')
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    if 'n_frames' in data_cfg and 'stride_length' in model_cfg:
        if int(data_cfg['n_frames']) != int(model_cfg['stride_length']):
            raise SystemExit(
                '[data].n_frames must equal [model].stride_length for scorer windows, got '
                f'{data_cfg["n_frames"]} and {model_cfg["stride_length"]}')
    corr = dict(scorer.get('corruption', {}))
    if 'output_granularity' in corr and str(corr['output_granularity']).lower() != mode:
        raise SystemExit(
            '[scorer.corruption].output_granularity must match '
            f'[scorer].output_granularity={mode!r}')
    if ('source_frame_duplicate_policy' in corr
            and corr['source_frame_duplicate_policy'] != scorer.get(
                'source_frame_duplicate_policy', 'inverse_multiplicity')):
        raise SystemExit(
            '[scorer.corruption].source_frame_duplicate_policy must match the top-level '
            '[scorer] value; set only one policy')
    pointwise = float(scorer.get('pointwise_weight', 0.0))
    frame_only = ('pointwise_weight', 'pointwise_balance', 'pointwise_label_smoothing',
                  'inactive_consistency_weight', 'anchor_consistency_weight')
    if mode == 'sequence':
        active = {k: scorer.get(k) for k in frame_only
                  if k in scorer and k not in ('pointwise_balance', 'pointwise_label_smoothing')}
        bad = {k: v for k, v in active.items() if float(v or 0.0) != 0.0}
        if bad:
            raise SystemExit(
                f'[scorer]: frame-only setting(s) {bad} require '
                'output_granularity = "frame"; sequence mode keeps the legacy loss.')
    else:
        if bool(corr.get('sequence_far_gate', False)):
            raise SystemExit(
                '[scorer.corruption].sequence_far_gate cannot be enabled in frame mode: '
                'frame mode always gates active rows per (frame, keypoint) slot')
        if pointwise > 0 and float(scorer.get('score_reg_weight', 0.0)) != 0.0:
            raise SystemExit(
                '[scorer]: score_reg_weight must be 0 when pointwise_weight > 0; the signed '
                'pointwise level anchor and score regularizer would fight each other')
    selection = str(scorer.get('selection_metric',
                                 'active_triplet_acc' if mode == 'frame' else 'triplet_acc'))
    if selection.startswith('val/'):
        selection = selection[4:]
    if selection not in FRAME_SELECTION_METRICS:
        raise SystemExit(
            f'[scorer].selection_metric = {selection!r} is unknown. Choose one of '
            f'{sorted(FRAME_SELECTION_METRICS)}')
    known_corr = {
        'const_offset_prob', 'frame_noise_prob', 'gradual_drift_prob', 'sinusoid_prob',
        'point_drop_prob', 'point_drop_max_frac', 'point_drop_bernoulli_rate', 'min_valid_frames',
        'mag_3d', 'mag_2d', 'min_corrupt_px', 'max_clean_px', 'reference_gate',
        'reference_margin_px', 'sequence_far_gate', 'segment_prob', 'segment_count',
        'n_segments', 'segment_len_frames', 'segment_length_frames', 'full_window_share',
        'segment_full_window_share', 'segment_full_window_prob', 'segment_types',
        'no_active_slot_policy',
        'out_of_view_policy', 'output_granularity', 'source_frame_duplicate_policy',
    }
    alias_pairs = (
        ('n_segments', 'segment_count'),
        ('segment_full_window_share', 'full_window_share'),
        ('segment_full_window_share', 'segment_full_window_prob'),
        ('segment_len_frames', 'segment_length_frames'),
    )
    for first, second in alias_pairs:
        if first in corr and second in corr and corr[first] != corr[second]:
            raise SystemExit(
                f'[scorer.corruption]: {first} and {second} disagree; set only one alias')
    unknown_corr = set(corr) - known_corr
    if unknown_corr:
        raise SystemExit(
            f'[scorer.corruption]: unknown key(s) {sorted(unknown_corr)}. Known keys: '
            f'{sorted(known_corr)}')
    segment_prob = float(corr.get('segment_prob', 0.0))
    if not 0.0 <= segment_prob <= 1.0:
        raise SystemExit('[scorer.corruption].segment_prob must be in [0, 1]')
    full_share = float(corr.get('segment_full_window_share',
                                 corr.get('full_window_share',
                                          corr.get('segment_full_window_prob', 0.0))))
    if not 0.0 <= full_share <= 1.0:
        raise SystemExit('[scorer.corruption] segment full-window share must be in [0, 1]')
    count_value = corr.get('n_segments', corr.get('segment_count', 1))
    count_values = list(count_value) if isinstance(count_value, (list, tuple)) else [count_value]
    if len(count_values) not in (1, 2):
        raise SystemExit('[scorer.corruption] segment count must be an integer or [lo, hi]')
    count_values = [int(value) for value in count_values]
    if count_values[0] < 1 or (len(count_values) == 2 and count_values[1] < count_values[0]):
        raise SystemExit('[scorer.corruption] segment count must satisfy 1 <= lo <= hi')
    length_value = corr.get('segment_len_frames', corr.get('segment_length_frames', [1, 1]))
    length_values = list(length_value) if isinstance(length_value, (list, tuple)) else [length_value]
    if len(length_values) not in (1, 2):
        raise SystemExit('[scorer.corruption] segment length must be an integer or [lo, hi]')
    length_values = [int(value) for value in length_values]
    if length_values[0] < 1 or (len(length_values) == 2 and length_values[1] < length_values[0]):
        raise SystemExit('[scorer.corruption] segment length must satisfy 1 <= lo <= hi')
    requested_types = corr.get('segment_types', SEGMENT_TYPES)
    if requested_types is None:
        raise SystemExit('[scorer.corruption].segment_types must be a sequence of names')
    if isinstance(requested_types, str):
        requested_types = (requested_types,)
    unknown_types = set(requested_types) - set(SEGMENT_TYPES)
    if unknown_types:
        raise SystemExit(
            f'[scorer.corruption].segment_types cannot be gated: {sorted(unknown_types)}')
    min_corrupt = float(corr.get('min_corrupt_px', 0.0))
    max_clean = float(corr.get('max_clean_px', 0.0))
    if (min_corrupt < 0 or max_clean < 0
            or (min_corrupt > 0 and min_corrupt <= max_clean)
            or (min_corrupt == 0 and max_clean > 0)):
        raise SystemExit(
            '[scorer.corruption] requires 0 <= max_clean_px < min_corrupt_px when a far '
            f'threshold is configured, or both zero for moved-mask fallback; '
            f'got min={min_corrupt}, max={max_clean}')
    reference_margin = float(corr.get('reference_margin_px', 0.0))
    if reference_margin < 0:
        raise SystemExit('[scorer.corruption].reference_margin_px must be non-negative')
    duplicate_policy = str(scorer.get('source_frame_duplicate_policy', 'inverse_multiplicity'))
    if duplicate_policy != 'inverse_multiplicity':
        raise SystemExit(
            '[scorer].source_frame_duplicate_policy must be "inverse_multiplicity"; no other '
            'frame weighting policy is implemented')
    reference_gate = str(corr.get('reference_gate', 'source_far'))
    out_of_view = str(corr.get('out_of_view_policy', 'exclude'))
    if out_of_view != 'exclude':
        raise SystemExit(
            '[scorer.corruption].out_of_view_policy must be "exclude"; visibility-preserving '
            'resampling is a separate unimplemented arm')
    no_active = str(corr.get('no_active_slot_policy', 'retry'))
    if no_active != 'retry':
        raise SystemExit(
            '[scorer.corruption].no_active_slot_policy must be "retry"; the dataset retry '
            'chain is the only implemented policy')
    if reference_gate not in {'none', 'source_far', 'independent_far'}:
        raise SystemExit(
            f'[scorer.corruption].reference_gate = {reference_gate!r} is not one of '
            '"none" | "source_far" | "independent_far"')
    if reference_gate == 'independent_far':
        raise SystemExit(
            '[scorer.corruption].reference_gate = "independent_far" requires an explicit aligned '
            'reference-coordinate provider; the standard scorer dataset has none. Use '
            '"source_far" or add the provider before enabling this mode.')
    return mode


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
    known = set(LoaderConfig.__dataclass_fields__) | {
        'path', 'num_workers', 'val_num_workers', 'prefetch_factor', 'worker_cv_threads'}
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
    cfg.setdefault('output_granularity', scorer_output_granularity(config))
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
    cfg.setdefault('output_granularity', scorer_output_granularity(config))
    cfg.setdefault('source_frame_duplicate_policy', config['scorer'].get(
        'source_frame_duplicate_policy', 'inverse_multiplicity'))
    return cfg


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
    lc = loader_config(data_cfg, config['model'])
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


def timed(loader, wait: list[float]):
    """Yield loader items while accumulating time blocked in ``next()``."""
    iterator = iter(loader)
    while True:
        started = time.time()
        try:
            item = next(iterator)
        except StopIteration:
            return
        wait[0] += time.time() - started
        yield item


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



def build_scorer_loss(config: dict, output_granularity: str | None = None):
    """Construct the mode-specific loss while keeping the legacy path untouched.

    The installed ``TripletScorerLoss`` deliberately remains the sequence compatibility loss.  The
    framewise implementation is repo-owned in ``scorer.losses``; importing it lazily keeps old
    sequence-only consumers able to import this module with an older posetail install and makes
    the missing implementation fail by name at the first frame run.
    """
    mode = output_granularity or scorer_output_granularity(config)
    cfg = config.get('scorer', {})
    common = dict(margin=float(cfg.get('triplet_margin', 0.25)),
                  precision_reg_weight=float(cfg.get('precision_reg_weight', 0.01)),
                  score_reg_weight=float(cfg.get('score_reg_weight', 0.0)))
    if mode == 'sequence':
        return TripletScorerLoss(**common)
    try:
        from .losses import FrameTripletScorerLoss
    except ImportError as e:
        raise RuntimeError(
            'frame scorer requested but tailcyclenet.scorer.losses.FrameTripletScorerLoss is '
            'unavailable; install the framewise scorer implementation') from e
    optional = {
        'pointwise_weight': float(cfg.get('pointwise_weight', 0.0)),
        'pointwise_balance': bool(cfg.get('pointwise_balance', True)),
        'pointwise_label_smoothing': float(cfg.get('pointwise_label_smoothing', 0.0)),
        'inactive_consistency_weight': float(cfg.get('inactive_consistency_weight', 0.0)),
        'anchor_consistency_weight': float(cfg.get('anchor_consistency_weight', 0.0)),
        'max_clean_px': float(cfg.get('corruption', {}).get('max_clean_px', 0.0)),
    }
    import inspect
    params = inspect.signature(FrameTripletScorerLoss).parameters
    kwargs = {k: v for k, v in {**common, **optional}.items() if k in params}
    return FrameTripletScorerLoss(**kwargs)


def _frame_loss_kwargs(loss_fn, trip: dict) -> dict:
    """Select metadata accepted by the local frame loss from a triplet dictionary.

    ``FrameTripletScorerLoss`` is intentionally a normal PyTorch module rather than a change to
    posetail's installed loss.  Filtering by its signature lets the trainer work with the initial
    mask-only implementation and with later versions that add a diagnostic mask without silently
    dropping the masks both versions understand.
    """
    import inspect
    values = {
        'active_mask': trip.get('active_mask'),
        'observed_mask': trip.get('observed_mask'),
        'in_view_mask': trip.get('in_view_mask'),
        'anchor_observed_mask': trip.get('anchor_observed_mask'),
        'near_mask': trip.get('near_mask'),
        'ambiguous_mask': trip.get('ambiguous_mask'),
        'far_mask': trip.get('far_mask'),
        'fired': trip.get('corruption_type_mask', trip.get('fired_frame')),
        'corruption_type_mask': trip.get('corruption_type_mask', trip.get('fired_frame')),
        'source_frame_weight': trip.get('source_frame_weight'),
        'corrupted_keypoint': trip.get('corrupted_keypoint'),
        'triplet': trip,
        'metadata': trip,
    }
    try:
        params = inspect.signature(loss_fn.forward).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_any = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_any:
        return {k: v for k, v in values.items() if v is not None}
    return {k: v for k, v in values.items() if v is not None and k in params}


def scorer_loss(loss_fn, scores, precision, labels, trip=None, output_granularity='sequence'):
    """Call a scorer loss with the right shape and frame metadata.

    Sequence mode is intentionally a three-positional-argument call, preserving the installed loss
    algebra and all historical tests.  Frame mode keeps ``[B,T,K,3]`` until the local loss applies
    its active/duplicate masks; flattening here would make the frame labels impossible to audit.
    """
    if output_granularity == 'sequence':
        if scores.ndim == 3:
            if trip is not None and bool(trip.get('sequence_far_gate', False)):
                far = trip.get('far_mask')
                if far is None:
                    raise ValueError("sequence_far_gate requires trip['far_mask'] metadata")
                keypoint_mask = far.any(dim=1)
                scores = scores[keypoint_mask]
                precision = precision[keypoint_mask]
                labels = labels[keypoint_mask]
            else:
                scores, precision, labels = flatten_sequence_triplet(scores, precision, labels)
        return loss_fn(scores, precision, labels)
    if trip is None:
        raise ValueError('framewise scorer loss requires the triplet masks')
    return loss_fn(scores, precision, labels, **_frame_loss_kwargs(loss_fn, trip))


def _per_type_accuracy(scores: torch.Tensor, fired: torch.Tensor,
                       active_mask: torch.Tensor | None = None) -> dict:
    """`P(good > bad)` broken out by corruption type at the active granularity.

    Legacy inputs are ``[K,3]``/``[K,G]``.  Framewise inputs are ``[B,T,K,3]`` (or unbatched
    ``[T,K,3]``) and ``[B,T,K,G]``.  No inactive/missing frame is allowed into a type denominator.
    """
    if scores.ndim == 2:
        correct = scores[:, 0] > scores[:, 1]
        type_mask = fired.bool()
        if active_mask is not None:
            type_mask = type_mask & active_mask.reshape(-1, 1).to(type_mask.device)
    else:
        if scores.ndim == 4:
            scores = scores[..., :]
        correct = scores[..., 0] > scores[..., 1]
        type_mask = fired.bool()
        if active_mask is not None:
            am = active_mask.bool()
            if am.ndim == 2:
                am = am[None]
            type_mask = type_mask & am[..., None].to(type_mask.device)
        correct = correct.to(type_mask.device)
    out = {}
    for i, name in enumerate(GENERATORS):
        mask = type_mask[..., i]
        if bool(mask.any()):
            out[f'val/acc_{name}'] = float(correct[mask].float().mean())
    return out


def _frame_pointwise_arrays(scores: torch.Tensor, labels: torch.Tensor,
                             trip: dict, type_index: int | None = None):
    """Return signed labels/scores/weights, optionally restricted to one corruption type."""
    from .losses import signed_pointwise_targets

    shape = scores.shape[:-1]
    device = scores.device
    def mask(name, fill):
        """Read one frame mask and broadcast its default to the score lattice."""
        value = trip.get(name)
        if value is None:
            return torch.full(shape, fill, dtype=torch.bool, device=device)
        return value.to(device=device, dtype=torch.bool)

    observed = mask('observed_mask', True)
    in_view = mask('in_view_mask', True)
    anchor = mask('anchor_observed_mask', True)
    far = mask('far_mask', False)
    near = mask('near_mask', False)
    ambiguous = mask('ambiguous_mask', False)
    rejected = mask('reference_rejected_mask', False)
    good_ref = trip.get('good_reference_distance_px')
    if good_ref is not None:
        good_ref = good_ref.to(device=device)
    targets, target_mask = signed_pointwise_targets(
        labels, observed, in_view, anchor, far, near,
        anchor_label=trip.get('anchor_label'), ambiguous_mask=ambiguous,
        reference_rejected_mask=rejected, reference_gate=trip.get('reference_gate', 'source_far'),
        good_reference_distance_px=good_ref,
        max_clean_px=float(trip.get('max_clean_px', 0.0)))
    if type_index is not None:
        type_mask = trip.get('corruption_type_mask', trip.get('fired_frame'))
        if type_mask is None:
            return (np.empty(0), np.empty(0), np.empty(0))
        type_mask = type_mask.to(device=device, dtype=torch.bool)
        if type_mask.ndim == 3:
            type_mask = type_mask[:, None]
        target_mask &= type_mask[..., type_index, None]
    weight = trip.get('source_frame_weight')
    if weight is None:
        weight = torch.ones(shape, dtype=scores.dtype, device=device)
    else:
        weight = weight.to(device=device, dtype=scores.dtype)
    weights = weight[..., None].expand_as(target_mask)
    return (scores[target_mask].detach().cpu().numpy(),
            targets[target_mask].detach().cpu().numpy(),
            weights[target_mask].detach().cpu().numpy())


def _weighted_binary_metrics(scores, targets, weights):
    """Weighted AUROC/AP for +/-1 synthetic labels, with a finite empty contract."""
    if len(scores) == 0:
        return float('nan'), float('nan')
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(targets) > 0
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores, positive, weights = scores[finite], positive[finite], weights[finite]
    pos = float(weights[positive].sum())
    neg = float(weights[~positive].sum())
    if not pos or not neg:
        return float('nan'), float('nan')
    order = np.argsort(scores, kind='stable')
    s, y, w = scores[order], positive[order], weights[order]
    auc_num = 0.0
    neg_before = 0.0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        pw = float(w[i:j][y[i:j]].sum())
        nw = float(w[i:j][~y[i:j]].sum())
        auc_num += pw * (neg_before + 0.5 * nw)
        neg_before += nw
        i = j
    auc = auc_num / (pos * neg)
    order = np.argsort(-scores, kind='stable')
    y, w = positive[order], weights[order]
    seen, ap_num = 0.0, 0.0
    for i in range(len(y)):
        seen += float(w[i])
        if y[i]:
            ap_num += float(w[i]) * (float((w[:i + 1] * y[:i + 1]).sum()) / seen)
    return float(auc), float(ap_num / pos)


def _frame_localization_counts(scores: torch.Tensor, trip: dict):
    """Count whether each corrupted keypoint's lowest bad frame lies in its far interval."""
    far = trip.get('far_mask')
    if far is None:
        return 0, 0
    far = far.to(device=scores.device, dtype=torch.bool)
    valid = trip.get('observed_mask', torch.ones_like(far)).to(dtype=torch.bool)
    valid &= trip.get('in_view_mask', torch.ones_like(far)).to(dtype=torch.bool)
    valid &= trip.get('anchor_observed_mask', torch.ones_like(far)).to(dtype=torch.bool)
    valid &= ~trip.get('ambiguous_mask', torch.zeros_like(far)).to(dtype=torch.bool)
    valid &= ~trip.get('reference_rejected_mask', torch.zeros_like(far)).to(dtype=torch.bool)
    good = bad = 0
    for bi in range(scores.shape[0]):
        for ki in range(scores.shape[2]):
            candidate = valid[bi, :, ki]
            if bool(far[bi, :, ki].any()) and bool(candidate.any()):
                frame_scores = scores[bi, :, ki, 1].masked_fill(~candidate, torch.inf)
                ix = int(frame_scores.argmin())
                good += int(bool(far[bi, ix, ki]))
                bad += 1
    return good, bad


def _frame_direct_metrics(scores: torch.Tensor, trip: dict) -> dict:
    """Count-based active metrics for one framewise triplet (not a window average)."""
    active = trip.get('active_mask')
    if active is None:
        active = torch.ones(scores.shape[:-1], dtype=torch.bool, device=scores.device)
    active = active.bool()
    if active.ndim == scores.ndim - 1 and active.shape != scores.shape[:-1]:
        active = active.reshape(scores.shape[:-1])
    correct = scores[..., 0] > scores[..., 1]
    gap = scores[..., 0] - scores[..., 1]
    n = int(active.sum())
    out = {
        'val/n_active_rows': float(n),
        'val/active_fraction': float(active.float().mean()),
    }
    if n:
        out['val/active_triplet_acc'] = float(correct[active].float().mean())
        out['val/active_score_gap'] = float(gap[active].mean())
    return out


def _move_triplet_metadata(trip: dict, device) -> dict:
    """Move framewise masks/weights that are not part of the three model members."""
    for key, value in list(trip.items()):
        if torch.is_tensor(value):
            trip[key] = value.to(device)
    return trip


def evaluate(model, loader, loss_fn, device, max_batches: int,
             optimizer=None, output_granularity: str = 'sequence') -> dict:
    """The held-out synthetic pass, with count-based frame metrics in frame mode."""
    model.eval()
    averaged = optimizer is not None and hasattr(optimizer, 'eval')
    per_type: dict[str, list] = {}
    direct: dict[str, list] = {}
    type_totals: dict[str, list[float]] = {}
    pointwise_chunks = []
    pointwise_type_chunks: dict[str, list] = {}
    localization = [0, 0]
    view_counts = {'out_of_view': 0, 'observed': 0, 'reference_rejected': 0,
                   'base_far': 0}
    frame_totals = {'active_weight': 0.0, 'active_correct': 0.0, 'active_gap': 0.0,
                    'valid_weight': 0.0}
    rejection_draws = 0
    seen = 0
    try:
        if averaged:
            optimizer.eval()
        with torch.no_grad():
            for trip in loader:
                if trip is None:
                    continue
                rejection_draws += int(trip.get('rejected_draws', 0))
                trip = triplet_to_device(trip, device)
                trip = _move_triplet_metadata(trip, device)
                scores, precision, labels = model.score_triplet(trip)
                scorer_loss(loss_fn, scores, precision, labels, trip, output_granularity)
                if output_granularity == 'frame':
                    fired = trip.get('corruption_type_mask', trip.get('fired_frame'))
                    active = trip.get('active_mask')
                    if fired is not None and active is not None:
                        active = active.bool()
                        correct = scores[..., 0] > scores[..., 1]
                        gap = scores[..., 0] - scores[..., 1]
                        weight = trip.get('source_frame_weight')
                        if weight is None or tuple(weight.shape) != tuple(active.shape):
                            weight = torch.ones_like(active, dtype=scores.dtype)
                        else:
                            weight = weight.to(dtype=scores.dtype)
                        observed = trip.get('observed_mask', torch.ones_like(active))
                        in_view = trip.get('in_view_mask', torch.ones_like(active))
                        anchor = trip.get('anchor_observed_mask', torch.ones_like(active))
                        ambiguous = trip.get('ambiguous_mask', torch.zeros_like(active))
                        rejected = trip.get('reference_rejected_mask',
                                             torch.zeros_like(active))
                        valid = (observed.bool() & in_view.bool() & anchor.bool()
                                 & ~ambiguous.bool() & ~rejected.bool())
                        frame_totals['active_weight'] += float(weight[active].sum())
                        frame_totals['active_correct'] += float(weight[active & correct].sum())
                        frame_totals['active_gap'] += float((weight * gap)[active].sum())
                        frame_totals['valid_weight'] += float(weight[valid].sum())
                        for i, name in enumerate(GENERATORS):
                            mask = active & fired[..., i].bool()
                            if name not in type_totals:
                                type_totals[name] = [0.0, 0.0]
                            type_totals[name][0] += float(weight[mask].sum())
                            type_totals[name][1] += float((weight[mask]
                                                            * correct[mask].to(weight.dtype)).sum())
                        pointwise_chunks.append(_frame_pointwise_arrays(scores, labels, trip))
                        for gi, name in enumerate(GENERATORS):
                            pointwise_type_chunks.setdefault(name, []).append(
                                _frame_pointwise_arrays(scores, labels, trip, gi))
                        loc_hit, loc_n = _frame_localization_counts(scores, trip)
                        localization[0] += loc_hit
                        localization[1] += loc_n
                        observed = trip.get('observed_mask', torch.ones_like(active)).bool()
                        view_counts['observed'] += int(observed.sum())
                        view_counts['out_of_view'] += int((observed & ~trip.get(
                            'in_view_mask', torch.ones_like(active)).bool()).sum())
                        view_counts['reference_rejected'] += int(trip.get(
                            'reference_rejected_mask', torch.zeros_like(active)).bool().sum())
                        view_counts['base_far'] += int(trip.get(
                            'base_far_mask', trip.get('far_mask', torch.zeros_like(active))).bool().sum())
                else:
                    fired = trip['fired']
                    sequence_active = None
                    if bool(trip.get('sequence_far_gate', False)):
                        far = trip.get('far_mask')
                        if far is None:
                            raise ValueError('sequence_far_gate requires trip[far_mask] metadata')
                        sequence_active = far.any(dim=1)
                    if scores.ndim == 2:
                        fired = fired[0]
                        if sequence_active is not None:
                            sequence_active = sequence_active[0]
                    type_metrics = (_per_type_accuracy(scores, fired) if sequence_active is None
                                    else _per_type_accuracy(scores, fired, sequence_active))
                    for k, v in type_metrics.items():
                        per_type.setdefault(k, []).append(v)
                seen += 1
                if seen >= max_batches:
                    break
    finally:
        try:
            if averaged:
                optimizer.train()
        finally:
            model.train()
    out = {k: float(np.mean(v)) for k, v in per_type.items()}
    if output_granularity == 'frame':
        active_weight = frame_totals['active_weight']
        valid_weight = frame_totals['valid_weight']
        if active_weight:
            out['val/n_active_rows'] = active_weight
            out['val/active_triplet_acc'] = frame_totals['active_correct'] / active_weight
            out['val/active_score_gap'] = frame_totals['active_gap'] / active_weight
        if valid_weight:
            out['val/active_fraction'] = active_weight / valid_weight
        for name, (denom, hits) in type_totals.items():
            if denom:
                out[f'val/acc_{name}'] = hits / denom
        if pointwise_chunks:
            point_scores = np.concatenate([x[0] for x in pointwise_chunks])
            point_targets = np.concatenate([x[1] for x in pointwise_chunks])
            point_weights = np.concatenate([x[2] for x in pointwise_chunks])
            auc, ap = _weighted_binary_metrics(point_scores, point_targets, point_weights)
            out['val/pointwise_auroc'] = auc
            out['val/pointwise_ap'] = ap
        for name, chunks in pointwise_type_chunks.items():
            typed = [chunk for chunk in chunks if len(chunk[0])]
            if not typed:
                continue
            typed_scores = np.concatenate([x[0] for x in typed])
            typed_targets = np.concatenate([x[1] for x in typed])
            typed_weights = np.concatenate([x[2] for x in typed])
            type_auc, type_ap = _weighted_binary_metrics(
                typed_scores, typed_targets, typed_weights)
            out[f'val/pointwise_auroc_{name}'] = type_auc
            out[f'val/pointwise_ap_{name}'] = type_ap
        if localization[1]:
            out['val/localization'] = localization[0] / localization[1]
        if view_counts['observed']:
            out['val/out_of_view_fraction'] = (view_counts['out_of_view']
                                               / view_counts['observed'])
        if view_counts['base_far']:
            out['val/reference_rejected_fraction'] = (
                view_counts['reference_rejected'] / view_counts['base_far'])
    else:
        out.update({k: float(np.mean(v)) for k, v in direct.items()})
    out['val/n_scored'] = float(seen)
    out['val/n_rejected_draws'] = float(rejection_draws)
    out['val/rejection_rate'] = float(rejection_draws / (rejection_draws + seen)
                                      if rejection_draws + seen else 0.0)
    return out


def _gather_eval(fabric, metrics: dict) -> dict:
    """Reduce rank-local validation dictionaries to one deterministic world result."""
    if fabric is None or fabric.world_size <= 1:
        return metrics
    import torch.distributed as dist
    gathered = [None] * fabric.world_size
    dist.all_gather_object(gathered, metrics)
    gathered = [m for m in gathered if isinstance(m, dict)]
    if not gathered:
        return {}
    out = {}
    keys = set().union(*(m for m in gathered))
    for key in keys:
        vals = [float(m[key]) for m in gathered
                if key in m and np.isfinite(m[key])]
        if not vals:
            continue
        if key in {'val/n_scored', 'val/n_rejected_draws', 'val/n_active_rows'}:
            out[key] = float(sum(vals))
            continue
        if key == 'val/rejection_rate':
            n = sum(float(m.get('val/n_scored', 0.0)) for m in gathered)
            r = sum(float(m.get('val/n_rejected_draws', 0.0)) for m in gathered)
            out[key] = r / (r + n) if r + n else 0.0
            continue
        if key == 'val/active_fraction':
            active = sum(float(m.get('val/n_active_rows', 0.0)) for m in gathered)
            valid = sum(float(m.get('val/n_active_rows', 0.0)) /
                        float(m[key]) for m in gathered if m.get(key, 0.0))
            out[key] = active / valid if valid else float('nan')
            continue
        weights = [float(m.get('val/n_active_rows', m.get('val/n_scored', 0.0)))
                   for m in gathered if key in m and np.isfinite(m[key])]
        den = sum(weights)
        out[key] = (sum(v * w for v, w in zip(vals, weights)) / den
                    if den else float(np.mean(vals)))
    return out

def scorer_checkpoint_mode_mismatch(requested: str, checkpoint: dict) -> bool:
    """Whether a scorer checkpoint must take the weights-only warm-start branch."""
    return scorer_checkpoint_granularity(checkpoint) != str(requested).lower()

def run(config_path, data_path, out: Path, checkpoint: str | None, device,
        max_iterations: int | None, no_wandb: bool, fresh: bool = False, fabric=None,
        num_workers: int | None = None, devices_arg: int | None = None) -> None:
    """Train one scorer run.

    The per-step training metrics are averaged over the last `print_freq` steps before printing,
    because ONE step is one window: with K keypoints that is K triplets (7 on calms21), so a
    single step's `triplet_acc` is quantised in units of 1/K and swings over the full range while
    the model learns. The val pass is the number to read; this is the trace.

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
    config = load_config(config_path, base=_SCORER_CONFIG)
    fresh_requested = fresh
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
    train_cfg = config['training']
    seed = int(train_cfg.get('seed', 23))
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    checkpoint_probe = None
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
                        from ..checkpoints import require_scorer_contract
                        require_scorer_contract(config, checkpoint_probe, where=str(probe))
                    except ValueError as e:
                        checkpoint_contract_mismatch = True
                        print(f'{probe}: {e}; using warm-start weights')

    train_ds, val_ds, registry = build_datasets(
        config, base_reg, rank=(fabric.global_rank if fabric is not None else 0),
        world_size=world)
    if fabric is not None:
        dist_utils.check_registry(fabric, registry.names)

    val_batches = int(train_cfg.get('val_batches', 20))
    n_target = int(max_iterations or train_cfg.get('n_iterations', 10000))
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

    model = build_scorer({**config['model'], 'video_encoder_pretrained': False},
                         registry.n_keypoints, **scorer_kwargs(config)).to(device)
    loss_fn = build_scorer_loss(config, output_granularity)
    if output_granularity == 'frame':
        model.add_module('frame_loss', loss_fn.to(device))

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
                and not checkpoint_contract_mismatch and not checkpoint_mode_mismatch):
            require_scorer_granularity(output_granularity,
                                       scorer_checkpoint_granularity(loaded),
                                       where=str(ckpt_file))
            resume_state = loaded
            resume_file = ckpt_file
            print(f'resume candidate: {ckpt_file} at iteration {loaded.get("iteration", "?")} '
                  f'(output_granularity={output_granularity})')
        else:
            if (loaded.get('kind', 'pose') == 'scorer'
                    and not checkpoint_mode_mismatch):
                require_scorer_granularity(output_granularity,
                                           scorer_checkpoint_granularity(loaded),
                                           where=str(ckpt_file))
            fresh = warm_start(model, ckpt_file, base_names=warm_start_names(base_reg))
    fresh = set(fresh) | {n for n, _ in model.named_parameters()
                          if n.startswith(('attn_pool.', 'score_', 'missing_point',
                                           'precision_head', 'pointwise_', 'frame_pool',
                                           'frame_loss.'))}

    opt_cfg = dist_utils.scale_optimizer_cfg(config['training']['optimizer'], world)
    if world > 1 and is0:
        print(f'lr: scaled by sqrt({world}) -> learning_rate {opt_cfg["learning_rate"]:g}'
              + (f', kpt_lr {opt_cfg["kpt_lr"]:g}' if 'kpt_lr' in opt_cfg else ''))
    optimizer = build_optimizer(model, fresh, opt_cfg)
    if resume_state is not None:
        from tailcyclenet.optim import optimizer_layout_matches, state_matches_optimizer_kind
        opt_kind = str(config['training']['optimizer'].get('optimizer', 'muon'))
        try:
            model.load_state_dict(resume_state['model_state'], strict=True)
            if not state_matches_optimizer_kind(resume_state['optimizer_state'], opt_kind):
                raise ValueError(f'optimizer kind does not match config optimizer={opt_kind!r}')
            if not optimizer_layout_matches(optimizer, resume_state['optimizer_state']):
                raise ValueError('optimizer param-group layout does not match the current config')
            optimizer.load_state_dict(resume_state['optimizer_state'])
            print(f'resuming {resume_file} at iteration {resume_state.get("iteration", "?")}')
        except (KeyError, RuntimeError, ValueError, SystemExit) as e:
            print(f'resume: {resume_file} is not reusable ({e}); warm-starting at iteration 0')
            fresh.update(warm_start(model, resume_file, base_names=warm_start_names(base_reg)))
            optimizer = build_optimizer(model, fresh, opt_cfg)
            resume_state = None
            resume_file = None

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
    val_freq = int(train_cfg.get('val_freq', 200))
    ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
    print_freq = int(train_cfg.get('print_freq', 20))
    max_norm = float(train_cfg.get('max_grad_norm', 10.0))
    start_iter = int(resume_state.get('iteration', 0)) if resume_state else 0
    n_iter = max(0, n_target - start_iter)
    raw_model = model
    wrap = fabric is not None and world > 1
    model = fabric.setup_module(raw_model) if wrap else raw_model
    model.train()
    if hasattr(optimizer, 'train'):
        optimizer.train()
    step = dist_utils.ceil_div(start_iter, world)
    skipped = 0
    local_val_freq = dist_utils.per_rank(val_freq, world) if val_freq else 0
    local_ckpt_freq = dist_utils.per_rank(ckpt_freq, world)
    local_print_freq = dist_utils.per_rank(print_freq, world)
    t0 = time.time()
    waited, evalled, ckpted = [0.0], [0.0], [0.0]
    window: dict[str, list] = {}
    selection_metric = resolve_selection_metric(config, output_granularity)
    best_acc = float('-inf')
    best_iter = -1
    best_locked = False
    prior_best = out / 'checkpoints' / 'checkpoint_best.pth'
    if resumed.exists() and not fresh_requested and prior_best.exists():
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

    def selected_value(values: dict) -> float | None:
        """Read the explicitly configured validation metric, never infer it from the loss."""
        candidates = [f'val/{selection_metric}', selection_metric]
        if output_granularity == 'frame' and selection_metric == 'triplet_acc':
            candidates.insert(0, 'val/active_triplet_acc')
        for key in candidates:
            value = values.get(key)
            if value is not None and np.isfinite(value):
                return float(value)
        return None
    local_target = dist_utils.ceil_div(n_target, world)
    while step < local_target:
        epoch_start = step
        for trip in timed(train_loader, waited):
            if step >= local_target:
                break
            accepted = dist_utils.all_ranks_finite(fabric, trip is not None)
            if not accepted:
                skipped += 1
                step += 1
                continue
            trip = triplet_to_device(trip, device)
            trip = _move_triplet_metadata(trip, device)
            if wrap:
                scores, precision, labels = model(trip)
            else:
                scores, precision, labels = model.score_triplet(trip)
            total = scorer_loss(loss_fn, scores, precision, labels, trip,
                                output_granularity)
            if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(total))):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                step += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            if fabric is not None:
                fabric.backward(total)
            else:
                total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm)
            if not dist_utils.all_ranks_finite(fabric, bool(torch.isfinite(grad_norm))):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                step += 1
                continue
            optimizer.step()
            iteration = step * world
            hist = loss_fn.collapse_history(prefix='')
            loss_fn.reset_history()
            for k, v in hist.items():
                window.setdefault(k, []).append(v)
            values = {f'train/{k}': float(np.mean(vs)) for k, vs in window.items()}
            if fabric is not None:
                values = {k: dist_utils.all_ranks_mean(fabric, v) for k, v in values.items()}
            if step % local_print_freq == 0:
                window.clear()

            if val_loader is not None and local_val_freq and step % local_val_freq == 0:
                started = time.time()
                val_values = evaluate(raw_model, val_loader, val_loss_fn, device,
                                      len(val_loader), optimizer, output_granularity)
                val_values = _gather_eval(fabric, val_values)
                values.update(val_values)
                evalled[0] += time.time() - started
                val_hist = val_loss_fn.collapse_history(prefix='val/')
                values.update(_gather_eval(fabric, val_hist))
                val_loss_fn.reset_history()
                if hasattr(optimizer, 'train'):
                    optimizer.train()
                acc = selected_value(values)
                if not best_locked and acc is not None and acc > best_acc:
                    best_acc, best_iter = float(acc), iteration
                    started = time.time()
                    save_checkpoint(out, iteration, raw_model, optimizer, config,
                                    name='best', write=is0, registry=registry, kind='scorer',
                                    scorer_selection_metric=selection_metric,
                                    scorer_selection_value=best_acc)
                    if fabric is not None:
                        fabric.barrier()
                    ckpted[0] += time.time() - started
                if is0:
                    print(f'[{iteration}] ' + '  '.join(
                    f'{k}={v:.4g}' for k, v in values.items() if k.startswith('val/')))
                    if acc is not None:
                        print(f'[{iteration}] best val/{selection_metric} {best_acc:.4f} at '
                              f'iteration {best_iter}')

            if step % local_print_freq == 0:
                wall = time.time() - t0
                elapsed = max(wall - evalled[0] - ckpted[0], 1e-9)
                report_steps = max(1, min(local_print_freq, step + 1))
                dt = elapsed / report_steps
                wait_frac = waited[0] / elapsed
                eval_frac = evalled[0] / wall if wall > 0 else 0.0
                values.update({
                    'train/iteration': iteration,
                    'train/grad_norm': float(grad_norm),
                    'train/gpu_peak_gb': _gpu_peak_gb(device, reset=True),
                    'train/cpu_peak_gb': _cpu_peak_gb(),
                    'train/sec_per_it': dt,
                    'train/loader_wait_frac': wait_frac,
                    'train/eval_frac': eval_frac,
                    'train/ckpt_frac': ckpted[0] / wall if wall > 0 else 0.0,
                    'train/skipped_frac': skipped / max(step + skipped, 1),
                    'train/world_size': world,
                })
                train_acc = values.get('train/active_triplet_acc',
                                        values.get('train/triplet_acc', float('nan')))
                train_gap = values.get('train/active_score_gap',
                                       values.get('train/score_gap', float('nan')))
                if is0:
                    print(f'[{iteration}] last {print_freq} steps: '
                          f'loss={values.get("train/scorer_loss", float("nan")):.4g} '
                          f'acc={train_acc:.3f} gap={train_gap:.4g} '
                          f'({dt:.2f}s/it wait {wait_frac:.0%} eval {eval_frac:.0%})')
                    if wb is not None:
                        log(wb, values, iteration)
                t0 = time.time()
                waited[0] = evalled[0] = ckpted[0] = 0.0
            elif wb is not None and is0:
                log(wb, values, iteration)

            if step % local_ckpt_freq == 0 or step + 1 == local_target:
                started = time.time()
                if fabric is not None:
                    dist_utils.check_ranks_agree(fabric, raw_model)
                save_checkpoint(out, iteration, raw_model, optimizer, config, write=is0,
                                registry=registry, kind='scorer')
                if fabric is not None:
                    fabric.barrier()
                ckpted[0] += time.time() - started
            step += 1
        if step == epoch_start:
            raise RuntimeError(
                'scorer DataLoader produced no successful triplets in an epoch; all samples were '
                'rejected. Check corruption magnitudes, distance thresholds, segment settings, '
                'visibility, and the training split instead of retrying forever.')
    if wb is not None:
        wb.finish()
    if is0:
        if best_iter >= 0:
            print(f'best: val/{selection_metric} {best_acc:.4f} at iteration {best_iter} '
                  f'(checkpoints/checkpoint_best.pth; output_granularity={output_granularity})')
        print(f'done: {n_iter} iterations into {out}')


def launch(args):
    """Launch Fabric once per rank; one triplet remains the per-rank batch."""
    from lightning.fabric import Fabric
    devices = int(args.devices)
    if devices < 1:
        raise SystemExit('--devices must be a positive count (scorer defaults to one)')
    if devices != 1 and str(args.device).startswith('cuda:') and str(args.device) != 'cuda:0':
        raise SystemExit('--device names one GPU and cannot be combined with --devices > 1')
    if args.precision == '16-mixed':
        raise SystemExit('--precision 16-mixed is unsupported by the scorer optimizer; use 32-true')
    cpu = str(args.device).startswith('cpu') or not torch.cuda.is_available()
    accelerator = 'cpu' if cpu else 'gpu'
    dev_arg = 1 if cpu and devices == 1 else (1 if devices == 1 else devices)
    strategy = args.strategy or ('auto' if devices == 1 else 'ddp_find_unused_parameters_true')
    fabric = Fabric(accelerator=accelerator, devices=dev_arg, strategy=strategy,
                    precision=args.precision)
    fabric.launch()
    os.environ['TAILCYCLENET_LOCAL_WORLD_SIZE'] = str(fabric.world_size)
    if not fabric.is_global_zero:
        sys.stdout = dist_utils.RankPrefix(sys.stdout, fabric.global_rank)
    if fabric.world_size > 1:
        fabric.print(f'distributed: {fabric.world_size} ranks, strategy {strategy!r}; '
                     f'n_iterations are totals and absolute learning rates scale by '
                     f'sqrt({fabric.world_size})')
    return fabric

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
    fabric = launch(args)
    run(args.config, args.data, Path(args.out), args.checkpoint, args.device,
        args.iterations, args.no_wandb, args.fresh, fabric=fabric,
        num_workers=args.num_workers, devices_arg=args.devices)
    return 0
