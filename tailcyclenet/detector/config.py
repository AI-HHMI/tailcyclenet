"""Detector training config: load, validate, apply CLI overrides.

The detector is trained from a TOML config, the same way the pose side is (`scripts/train.py` +
`checkpoints.load_config`). One shipped recipe lives in `configs/detector.toml`; a user overlay
may `extends` it one level deep. Every key is validated against an explicit allowed set --
an unknown key is a typo, not a comment, and must not silently train at defaults (the same guard
`scripts/train.py` applies to `[data]`).

Blocks:
    [data]      the loader and what the regression target bounds
    [model]     the architecture: `yolox` (capacity tier), plus the T4.1 pretraining prototype
                (`bottleneck_expansion`, `pretrained` -- dev/plans/detector_accuracy.md)
    [training]  schedule, run folder, device

`weight_decay` (5e-4) and the cosine schedule are not configurable: they were never flags.
"""
from __future__ import annotations

from pathlib import Path

from ..crop import BOX_SOURCES
from .yolox import YOLOX_TIERS

# THE ALLOWED KEYS, PER BLOCK. Anything else raises -- see module docstring. These are the
# one-to-one names of the argparse flags `scripts/train_detector.py` used to take (minus
# `--boxes`' dash), so a config value means exactly what the flag meant.
DATA_KEYS = frozenset({
    'path', 'boxes', 'min_crop_dim', 'input_wh', 'min_box_px', 'max_input_px',
    'frames_per_group', 'val_frames_per_group', 'augment', 'augment_strong', 'rotate_deg',
    'reduce', 'keypoints', 'hflip', 'tile_wh', 'tile_scale', 'tile_bg_per_frame',
    'use_regions', 'ignore_present',
})
MODEL_KEYS = frozenset({'yolox', 'bottleneck_expansion', 'pretrained'})
TRAINING_KEYS = frozenset({
    'out', 'iters', 'batch_size', 'lr', 'num_workers', 'seed', 'device', 'eval_every',
    'eval_batches', 'kpt_weight', 'kpt_score_weight', 'iou_aware_obj', 'iou_aware_warmup',
})
BLOCKS = (('data', DATA_KEYS), ('model', MODEL_KEYS), ('training', TRAINING_KEYS))
YOLOX_CHOICES = ('trimmed', *sorted(YOLOX_TIERS))


def _raise_unknown(block: str, cfg: dict, known: frozenset) -> None:
    unknown = set(cfg) - known
    if unknown:
        raise SystemExit(
            f'[{block}]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run '
            f'would train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(known)}')


def _pair(key: str, value):
    """A `[w, h]` list from TOML, or None when empty. Any other shape raises."""
    if value in (None, [], ''):
        return None
    err = f'{key}: expected [width, height], got {value!r}'
    try:
        pair = [int(v) for v in value]
    except (TypeError, ValueError):
        raise SystemExit(err)
    if len(pair) != 2 or min(pair) <= 0:
        raise SystemExit(err)
    return pair


def load_detector_config(path, out=None, iters=None, device=None) -> dict:
    """Load + validate a detector config; return the effective dict (blocks nested).

    `out` / `iters` / `device` override `[training]` -- the only CLI knobs left. The returned
    dict is what the run folder records as `config.toml`, so the record is the effective recipe.
    """
    from tailcyclenet.checkpoints import load_config

    cfg = load_config(Path(path))
    missing = [b for b, _ in BLOCKS if b not in cfg]
    if missing:
        raise SystemExit(f'{path}: missing block(s) {missing}. A detector config needs '
                         f'[data], [model] and [training].')
    for block, known in BLOCKS:
        _raise_unknown(block, cfg[block], known)
    data, model, train = cfg['data'], cfg['model'], cfg['training']

    # THE CLI OVERRIDE MUST LAND BEFORE THE REQUIRED-FIELD CHECK, OR IT CANNOT RESCUE ANYTHING.
    # `configs/detector.toml` ships `out = ""` on the promise that `--out` fills it in -- and
    # every per-root overlay under `configs/detector/` follows that promise, leaving `out` for
    # the CLI. Checking `train.get('out')` before this ran the promise into the requirement: a
    # config with `out = ""` and a caller passing `out=`  still raised "required", because the
    # override at the bottom of this function never got the chance to apply. `path` has no CLI
    # override to rescue it (train_detector.py exposes none), so its check stays where it was.
    if out is not None:
        train['out'] = str(out)
    if iters is not None:
        train['iters'] = int(iters)
    if device is not None:
        train['device'] = str(device)

    # REQUIRED, never defaulted: a config that forgot the dataset would otherwise train on
    # whatever the CWD happened to be, silently.
    if not data.get('path'):
        raise SystemExit('[data].path is required: ONE dataset root (has train/, optionally '
                         'val/ and test/). The detector is trained per dataset.')
    if not train.get('out'):
        raise SystemExit('[training].out is required: the run folder for checkpoints, '
                         'metrics.json and the recorded config, and neither the config nor '
                         '--out supplied one.')

    # CHOICE GUARDS, the same class as the pose side's `build_model` raise: a bad value must
    # fail at load, not train a different recipe than the one the config names.
    if data.get('boxes', 'instances') not in BOX_SOURCES:
        raise SystemExit(f'[data].boxes must be one of {BOX_SOURCES}, got '
                         f'{data["boxes"]!r}.')
    if model.get('yolox', 'tiny') not in YOLOX_CHOICES:
        raise SystemExit(f'[model].yolox must be one of {YOLOX_CHOICES}, got '
                         f'{model["yolox"]!r}.')

    # TOML has no null, so an "absent" pair is an empty list. Normalise to None for the code
    # that has always taken None-or-pair.
    data['input_wh'] = _pair('input_wh', data.get('input_wh'))
    data['tile_wh'] = _pair('tile_wh', data.get('tile_wh'))

    for k in ('iters', 'batch_size', 'num_workers', 'seed', 'eval_every', 'eval_batches',
              'iou_aware_warmup'):
        train[k] = int(train.get(k, {'iters': 20000, 'batch_size': 16, 'num_workers': 8,
                                     'seed': 0, 'eval_every': 2000, 'eval_batches': 25,
                                     'iou_aware_warmup': 2000}[k]))
    # T2.3 (dev/plans/detector_accuracy.md): the BCE objectness target at a positive anchor
    # becomes the detached IoU between its predicted and GT box, instead of a hard 1.0, once past
    # `iou_aware_warmup` iterations. Default OFF -- byte-identical to every checkpoint on record.
    train['iou_aware_obj'] = bool(train.get('iou_aware_obj', False))
    for k in ('min_crop_dim', 'min_box_px', 'max_input_px', 'frames_per_group',
              'val_frames_per_group', 'tile_bg_per_frame'):
        data[k] = int(data.get(k, {'min_crop_dim': 64, 'min_box_px': 32,
                                   'max_input_px': 4 * 416 * 416, 'frames_per_group': 40,
                                   'val_frames_per_group': 8,
                                   'tile_bg_per_frame': 1}[k]))
    for k in ('lr', 'kpt_weight', 'kpt_score_weight'):
        train[k] = float(train.get(k, {'lr': 1e-3, 'kpt_weight': 1.0,
                                       'kpt_score_weight': 1.0}[k]))
    for k, default in (('rotate_deg', 45.0), ('tile_scale', 1.0)):
        data[k] = float(data.get(k, default))
    data['augment'] = bool(data.get('augment', True))
    data['augment_strong'] = bool(data.get('augment_strong', True))
    data['reduce'] = bool(data.get('reduce', False))
    data['keypoints'] = bool(data.get('keypoints', False))
    data['hflip'] = bool(data.get('hflip', True))
    data['use_regions'] = bool(data.get('use_regions', False))
    data['ignore_present'] = bool(data.get('ignore_present', False))
    if data['ignore_present'] and data['use_regions']:
        raise SystemExit('[data].ignore_present and [data].use_regions cannot both be set: both '
                         'are the one opt-in (M,4) tuple slot box_collate/split_batch dispatch '
                         'by rank. See BoxDataset.__init__.')
    data['boxes'] = str(data.get('boxes', 'instances'))
    model['yolox'] = str(model.get('yolox', 'tiny'))

    # T4.1 (dev/plans/detector_accuracy.md), PROTOTYPE. `bottleneck_expansion`: 0.5 (default) is
    # byte-identical to every checkpoint on record; 1.0 is the shape a Megvii COCO backbone
    # actually loads into (see `yolox.Bottleneck`). `pretrained`: '' (default, from scratch, every
    # run on record) or 'coco' (load the tier's COCO backbone -- `detector.pretrained`). An
    # arbitrary path is NOT YET SUPPORTED (that is T4.1b's in-domain-pretraining artefact, which
    # does not exist yet) -- raise by name rather than silently ignoring it.
    model['bottleneck_expansion'] = float(model.get('bottleneck_expansion', 0.5))
    model['pretrained'] = str(model.get('pretrained', ''))
    if model['yolox'] == 'trimmed' and model['bottleneck_expansion'] != 0.5:
        raise SystemExit(
            f"[model].bottleneck_expansion={model['bottleneck_expansion']} was set alongside "
            "yolox='trimmed', but trimmed's backbone does not take it -- it stays at 0.5 "
            "permanently. Use a canonical tier (one of the YOLOX_TIERS names) for a "
            "COCO-compatible backbone.")
    if model['pretrained'] not in ('', 'coco'):
        raise SystemExit(
            f"[model].pretrained={model['pretrained']!r}: only '' (from scratch) and 'coco' are "
            "supported by this prototype. An arbitrary path (T4.1b, in-domain pretraining) is not "
            "wired up yet.")
    if model['pretrained'] == 'coco':
        if model['yolox'] == 'trimmed':
            raise SystemExit("[model].pretrained='coco' requires a canonical yolox tier -- "
                             "'trimmed' has no COCO counterpart to load.")
        if model['bottleneck_expansion'] != 1.0:
            raise SystemExit(
                "[model].pretrained='coco' requires [model].bottleneck_expansion=1.0 -- at 0.5 "
                "every bottleneck conv is half Megvii's width and the load would silently take "
                "only 19 of 35 backbone tensors (dev/plans/detector_accuracy.md T4.1).")
    return cfg
