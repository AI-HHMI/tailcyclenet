"""Run folders, warm start, save and load.

A run folder is the unit of reproducibility: it holds the config it was launched with, the
keypoint registry that fixes what every embedding row means, and the checkpoints. Everything
downstream (eval, inference, rendering) takes only `--run <folder>`, so a config/checkpoint
mismatch is impossible by construction rather than by discipline.

Schedule-free training keeps TWO iterates. `model_state` is the raw `y` and is what you resume
from; `model_state_eval` is the averaged `x` and is what you evaluate. Loading the wrong one is
a silent accuracy loss, so both are saved explicitly and `load_run` defaults to the eval weights.
"""
from __future__ import annotations

import time
import tomllib
from pathlib import Path

import torch

from posetail.posetail.train_utils import (_convert_cross_attn, _filter_shape_mismatch,
                                           _interp_res_params)

from .format import Registry
from .model import build_model


def check_image_size(config: dict) -> None:
    """`[model].image_size` and `[data].image_size` must agree. Nothing else notices if they do not.

    The loader resizes every crop so its long side is `[data].image_size`, while the weights bake
    `[model].image_size` into the decode arithmetic -- `PadToSize` (tracker_encoder.py:192),
    `points_pred + image_size // 2` for the absolute 2D bins (:609), and
    `p3d_cams * image_size` for gridresid's metric motion (:697). `PadToSize` only ever pads UP,
    so a smaller data size leaves the crop in the corner of a zero-padded canvas while the
    cameras describe the unpadded one; either way 2D shifts by half the difference and 3D scales
    by their ratio. Both silent.
    """
    model_px = config.get('model', {}).get('image_size')
    data_px = config.get('data', {}).get('image_size')
    if model_px is None or data_px is None or int(model_px) == int(data_px):
        return
    raise ValueError(
        f'[model].image_size = {model_px} but [data].image_size = {data_px}. These must agree: '
        f'the loader resizes crops to {data_px} while the model decodes as if they were '
        f'{model_px}, shifting 2D predictions by {(int(model_px) - int(data_px)) // 2} px and '
        f'scaling the 3D residual by {int(model_px) / int(data_px):g}.')


def resolve_checkpoint(folder: Path, checkpoint: str | None = None, min_age_s: float = 60.0):
    """The newest complete checkpoint in `folder`.

    The age guard is not paranoia: these files are ~5.6 GB, and a half-written one is
    indistinguishable from a complete one to `glob`. Picking one up mid-write fails deep inside
    `torch.load` with an error that names nothing.
    """
    folder = Path(folder)
    if checkpoint:
        p = folder / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    files = sorted(folder.glob('checkpoint_*.pth'))
    if not files:
        raise FileNotFoundError(f'{folder}: no checkpoint_*.pth')
    now = time.time()
    ready = [f for f in files if now - f.stat().st_mtime >= min_age_s]
    if not ready:
        # Everything is young; the newest may still be being written, so take the one before it.
        ready = files[:-1] or files
    return ready[-1]


def save_run_meta(run: Path, config: dict, registry: Registry) -> None:
    import toml
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.toml').write_text(toml.dumps(config))
    registry.save(run / 'keypoint_registry.toml')


def save_checkpoint(run: Path, iteration: int, model, optimizer, config: dict) -> Path:
    """Save both schedule-free iterates.

    `model_state` is the raw training weight (resume from this). `model_state_eval` is the
    averaged weight (evaluate with this) and is captured by flipping the optimizer into eval
    mode, which swaps the model's parameter tensors in place, then flipping back.
    """
    ckpt_dir = run / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    eval_state = None
    if hasattr(optimizer, 'eval') and hasattr(optimizer, 'train'):
        optimizer.eval()
        eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer.train()
    path = ckpt_dir / f'checkpoint_{iteration:08d}.pth'
    torch.save({'iteration': iteration, 'model_state': state,
                'model_state_eval': eval_state,
                'optimizer_state': optimizer.state_dict(), 'model_config': config.get('model')},
               path)
    return path


def load_run(run: Path, checkpoint: str | None = None, device='cpu', eval_weights: bool = True):
    """(model, config, registry, checkpoint_path) from a run folder. Nothing else is needed."""
    run = Path(run)
    with open(run / 'config.toml', 'rb') as f:
        config = tomllib.load(f)
    check_image_size(config)
    registry = Registry.load(run / 'keypoint_registry.toml')
    path = resolve_checkpoint(run / 'checkpoints', checkpoint)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    state = ckpt.get('model_state_eval') if eval_weights else None
    if state is None:
        if eval_weights:
            print(f'{path.name}: no model_state_eval; falling back to the raw training weights')
        state = ckpt['model_state']
    missing, unexpected = model.load_state_dict(state, strict=False)
    _report('load_run', missing, unexpected, [])
    return model.to(device).eval(), config, registry, path


def warm_start(model, checkpoint_path: Path, verbose: bool = True) -> set[str]:
    """Load the base tracker into a pose model. Returns the names of the params left fresh.

    Three things happen that a plain `load_state_dict` would get wrong:

    1. **The fusion gate is inflated, not dropped.** The base gate is an N-term Linear; this
       model has N+1 terms, so the shapes differ and a strict load would fail while a filtered
       load would throw away pretrained fusion behaviour. `inflate_stock_gate` places each of the
       base's terms into the slot the SAME term occupies here, by name, leaving the identity row
       and block zero -- so the model starts at the pretrained fusion behaviour rather than noise.
    2. **The base's migrations run first.** `_convert_cross_attn` handles the old fused
       `nn.MultiheadAttention` layout, `_interp_res_params` interpolates the resolution-coupled
       tensors if `image_size` changed.
    3. **Everything dropped is named.** `strict=False` silently discards; a base checkpoint from
       the abandoned memory branch would quietly lose its `memory_*` subtrees, and a keypoint
       table sized for a different registry would quietly reset. Both are printed.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    state = dict(ckpt.get('model_state_eval') or ckpt['model_state'])

    gate_w, gate_b = 'query_encoder.gate.0.weight', 'query_encoder.gate.0.bias'
    if gate_w in state and state[gate_w].shape != model.query_encoder.gate[0].weight.shape:
        state[gate_w], state[gate_b] = model.query_encoder.inflate_stock_gate(
            state[gate_w], state[gate_b])
        if verbose:
            print(f'warm start: fusion gate inflated '
                  f'{len(model.query_encoder.stock_term_names())} -> '
                  f'{len(model.query_encoder.term_names())} terms, by name')

    state = _convert_cross_attn(state, model)
    state, interpolated = _interp_res_params(state, model)   # returns (dict, changed_keys)
    if interpolated and verbose:
        print(f'warm start: {len(interpolated)} resolution-coupled tensor(s) interpolated')
    state, dropped = _filter_shape_mismatch(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        _report('warm start', missing, unexpected, dropped)
    return set(missing) | set(dropped)


def _report(what, missing, unexpected, dropped):
    """Name everything. A silent drop here is a whole training run spent on the wrong weights."""
    def head(xs, n=8):
        xs = list(xs)
        return ', '.join(xs[:n]) + (f' … (+{len(xs) - n} more)' if len(xs) > n else '')

    if dropped:
        print(f'{what}: {len(dropped)} tensor(s) dropped on a SHAPE MISMATCH: {head(dropped)}')
    if missing:
        print(f'{what}: {len(missing)} param(s) left at fresh init: {head(missing)}')
    if unexpected:
        print(f'{what}: {len(unexpected)} checkpoint key(s) UNUSED: {head(unexpected)}')
    if not (dropped or missing or unexpected):
        print(f'{what}: exact match, nothing fresh and nothing discarded')
