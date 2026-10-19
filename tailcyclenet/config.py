"""Config files: the three family bases and the layering every config goes through.

`configs/` is packaged as `tailcyclenet.configs` and read back via importlib.resources, so the
bases resolve under an editable install and a built wheel alike (never a repo-relative path).
"""
from __future__ import annotations

import tomllib
from importlib.resources import files as _pkg_files
from pathlib import Path

BASE_CONFIG = _pkg_files('tailcyclenet.configs') / 'base.toml'
DETECTOR_CONFIG = _pkg_files('tailcyclenet.configs') / 'detector.toml'
SCORER_CONFIG = _pkg_files('tailcyclenet.configs') / 'scorer.toml'


def deep_merge(base: dict, over: dict) -> dict:
    """Merge `over` into `base`, RECURSING when both sides are dicts.

    An overlay may set one key deep inside a block without restating its siblings; a block can
    therefore never be replaced wholesale, only merged into.
    """
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path, base: Path | None = None) -> dict:
    """A run config layered over its family's base: `BASE_CONFIG` (pose, the default),
    `DETECTOR_CONFIG` or `SCORER_CONFIG`.

    The `extends` key is deleted and RAISES by name: the overlay IS the whole difference.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    if 'extends' in cfg:
        raise SystemExit(
            f'{path.name}: `extends` is deleted -- every config layers over '
            f'`{(base or BASE_CONFIG).name}` automatically, so the key is not needed. '
            'Delete the line and the recipe is unchanged.')
    with open(base or BASE_CONFIG, 'rb') as f:
        base_cfg = tomllib.load(f)
    return deep_merge(base_cfg, cfg)


def check_image_size(config: dict) -> None:
    """`[model].image_size` and `[data].image_size` must agree; nothing else notices if they do
    not. The loader resizes crops to the data value while the weights bake the model value into
    the decode arithmetic -- both silent otherwise. `--refine-px` is the only thing that may walk
    past this, because `pose.model.PoseTrackerEncoder.forward` compensates for it.
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
