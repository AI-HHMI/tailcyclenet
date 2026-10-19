"""Scorer run folders and checkpoint files: load one, and save one with its contract stamped."""
from __future__ import annotations

import tomllib
from pathlib import Path

import torch

from ..checkpoints import (file_bundle, prior_provenance, print_provenance, require_ckpt_kind,
                           require_kind, resolve_checkpoint, save_checkpoint)
from ..config import check_image_size
from ..format import Registry
from .contract import (require_scorer_contract, require_scorer_granularity,
                       scorer_checkpoint_fields, scorer_checkpoint_granularity,
                       scorer_output_granularity)
from .losses import build_scorer_loss
from .model import build_scorer


def save_scorer_checkpoint(run: Path, iteration: int, model, optimizer, config: dict, *,
                           selection_metric: str | None = None,
                           selection_value: float | None = None, **kwargs):
    """`checkpoints.save_checkpoint` with `kind='scorer'` and the scorer's contract fields."""
    return save_checkpoint(run, iteration, model, optimizer, config, kind='scorer',
                           extra=scorer_checkpoint_fields(config, selection_metric,
                                                          selection_value), **kwargs)


def load_scorer_run(run: Path, checkpoint: str | None = None, device='cpu'):
    """(model, config, registry, checkpoint_path) from a scorer run folder or checkpoint file.

    A FILE is self-contained when it embeds its config and registry (every scorer checkpoint
    `save_checkpoint` writes does); an older one falls back to the run folder it sits in. What
    makes a file safe to load alone is the checkpoint's own `kind`, checked before anything is
    built, plus the output-granularity and semantic-contract checks a run folder gets.

    Resume vs warm start: a checkpoint carrying a full training state is RESUMED by the caller
    (`train_scorer.py`), and a pose checkpoint is a WARM START. Both are readable from here; what
    is refused is a run folder of the wrong family, or a checkpoint whose recorded `kind` is not
    the one the caller asked for.
    """
    run = Path(run)
    if run.is_file():
        if checkpoint is not None:
            raise ValueError(f'{run}: --scorer-checkpoint selects a file inside a run folder, but '
                             '--scorer already names a checkpoint file')
        path = run
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f'{path}: checkpoint must be a dictionary, got {type(ckpt).__name__}')
        require_ckpt_kind(ckpt, 'scorer', path)
        config, registry, prov = file_bundle(path, ckpt, 'scorer')
    else:
        with open(run / 'config.toml', 'rb') as f:
            config = tomllib.load(f)
        require_kind(config, 'scorer', run)
        registry = Registry.load(run / 'keypoint_registry.toml')
        path = resolve_checkpoint(run / 'checkpoints', checkpoint)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        require_ckpt_kind(ckpt, 'scorer', path)
        prov = prior_provenance(run)
    mode = scorer_output_granularity(config)
    config = dict(config)
    config['scorer'] = {**config.get('scorer', {}), 'output_granularity': mode}
    check_image_size(config)
    checkpoint_mode = scorer_checkpoint_granularity(ckpt)
    embedded = ckpt.get('config')
    if isinstance(embedded, dict) and isinstance(embedded.get('scorer'), dict):
        embedded_mode = scorer_output_granularity(embedded)
        require_scorer_granularity(embedded_mode, checkpoint_mode,
                                    where=f'{path} metadata/config')
    require_scorer_granularity(mode, checkpoint_mode, where=str(path))
    require_scorer_contract(config, ckpt, where=str(path))
    scorer_cfg = dict(config.get('scorer', {}))
    scorer_cfg.pop('corruption', None)
    model = build_scorer({**config['model'], 'video_encoder_pretrained': False},
                         registry.n_keypoints, **{k: v for k, v in scorer_cfg.items()
                                                  if k in ('pool_num_heads', 'score_hidden',
                                                           'use_precision', 'output_granularity')})
    if mode == 'frame':
        model.add_module('frame_loss', build_scorer_loss(config, mode))
    state = ckpt.get('model_state_eval') or ckpt['model_state']
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(f'{path}: scorer checkpoint does not exactly match its run model: {e}') from e
    if prov:
        print_provenance(prov, path)
    model.run_provenance = prov
    return model.to(device).eval(), config, registry, path
