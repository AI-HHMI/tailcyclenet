"""Pose checkpoint loading: a run folder or a packaged `.pth` -> model, and the warm start."""
from __future__ import annotations

import tomllib
from pathlib import Path

import torch

from tracktail.model.train_utils import (_convert_cross_attn, _filter_shape_mismatch,
                                           _interp_res_params)

from ..checkpoints import (file_bundle, prior_provenance, print_provenance, report_load,
                           require_ckpt_kind, require_kind, resolve_checkpoint)
from ..config import check_image_size
from ..format import Registry
from .model import build_model


def _load_packaged_pose(path: Path, device='cpu', model_overrides: dict | None = None):
    """Load a self-contained pose checkpoint, whether packaged or copied from training."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f'{path}: checkpoint must be a dictionary, got {type(ckpt).__name__}')
    if ckpt.get('kind', 'pose') != 'pose':
        require_ckpt_kind(ckpt, 'pose', path)
    config, registry, prov = file_bundle(path, ckpt, 'pose')
    print_provenance(prov, path)
    check_image_size(config)
    if model_overrides:
        config['model'] = {**config.get('model', {}), **model_overrides}
        print(f'load_run: [model] OVERRIDDEN {model_overrides} -- this is an assertion about what '
              'the checkpoint was trained with, not something read from it')

    state = ckpt.get('model_state_eval') or ckpt.get('model_state')
    if not isinstance(state, dict):
        raise ValueError(f'{path}: pose checkpoint has no model_state dictionary')
    source = ckpt.get('source_run', '?')
    selected = ckpt.get('source_checkpoint', path.name)
    print(f'packaged pose checkpoint: {selected} from {source} '
          f'(iteration {ckpt.get("iteration", "?")})')
    model = build_model({**config['model'], 'video_encoder_pretrained': False},
                        n_keypoints=registry.n_keypoints)
    missing, unexpected = model.load_state_dict(state, strict=False)
    report_load('load_run', missing, unexpected, [])
    model.run_provenance = prov
    return model.to(device).eval(), config, registry, path


def load_run(run: Path, checkpoint: str | None = None, device='cpu',
             model_overrides: dict | None = None):
    """(model, config, registry, checkpoint_path) from a pose run folder or pose checkpoint."""
    run = Path(run)
    if run.is_file():
        if checkpoint is not None:
            raise ValueError(f'{run}: --checkpoint selects a file inside a run folder, but --run '
                             'already names a checkpoint file')
        return _load_packaged_pose(run, device=device, model_overrides=model_overrides)
    with open(run / 'config.toml', 'rb') as f:
        config = tomllib.load(f)
    require_kind(config, 'pose', run)
    check_image_size(config)
    if model_overrides:
        config['model'] = {**config.get('model', {}), **model_overrides}
        print(f'load_run: [model] OVERRIDDEN {model_overrides} -- this is an assertion about what '
              'the checkpoint was trained with, not something read from it')
    prov = prior_provenance(run)
    if prov:
        print_provenance(prov, run)
    else:
        print(f'{run}: no provenance.toml -- this run predates commit recording, so which '
              'architecture the weights were trained under cannot be read back from the folder')
    registry = Registry.load(run / 'keypoint_registry.toml')
    path = resolve_checkpoint(run / 'checkpoints', checkpoint)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    model = build_model({**config['model'], 'video_encoder_pretrained': False},
                        n_keypoints=registry.n_keypoints)
    state = ckpt.get('model_state_eval')
    if state is None:
        print(f'{path.name}: no model_state_eval; falling back to the raw training weights')
        state = ckpt['model_state']
    missing, unexpected = model.load_state_dict(state, strict=False)
    report_load('load_run', missing, unexpected, [])
    model.run_provenance = prov
    return model.to(device).eval(), config, registry, path


def warm_start(model, checkpoint_path: Path, verbose: bool = True,
               base_names: tuple[str, ...] | None = None) -> set[str]:
    """Load the base tracker into a pose model. Returns the names of the params left fresh.

    Base migrations run first and every `strict=False` drop is named. A GROWN REGISTRY KEEPS ITS
    ROWS: the checkpoint's (n0, d) identity table is copied into the first n0 rows of this
    model's (n, d) one -- refused if `base_names`'s length does not match n0, because a
    mis-applied copy points each row at a different body part. `_interp_res_params` returns
    (dict, BOOL): `interpolated` is the flag, not a key list.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    state = dict(ckpt.get('model_state_eval') or ckpt['model_state'])

    state = _convert_cross_attn(state, model)
    state, interpolated = _interp_res_params(state, model)
    if interpolated and verbose:
        print('warm start: resolution-coupled tensors checked; see any res-interp lines above')

    msd = model.state_dict()
    for k, v in list(state.items()):
        if not k.endswith('kpt_embed.weight') or k not in msd:
            continue
        n0, n = v.shape[0], msd[k].shape[0]
        if not (n0 < n and v.shape[1:] == msd[k].shape[1:]):
            continue
        if base_names is not None and len(base_names) != n0:
            if verbose:
                print(f'warm start: {k} is {n0} rows but the base registry names '
                      f'{len(base_names)}, so it is NOT that registry\'s table -- the rows are '
                      'left fresh rather than copied onto the wrong keypoints')
            continue
        grown = msd[k].clone()
        grown[:n0] = v
        state[k] = grown
        if verbose:
            print(f'warm start: {k} widened {n0} -> {n} rows, {n0} preserved')

    state, dropped = _filter_shape_mismatch(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        report_load('warm start', missing, unexpected, dropped)
    return set(missing) | set(dropped)
