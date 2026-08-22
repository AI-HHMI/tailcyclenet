"""Detector training config: load, validate, apply CLI overrides.

The detector is trained from a TOML config, the same way the pose side is (`scripts/train.py` +
`checkpoints.load_config`). One shipped recipe lives in `configs/detector.toml`; a user overlay
may `extends` it one level deep. Every key is validated against an explicit allowed set -- an
unknown key is a typo, not a comment, and must not silently train at defaults.

Blocks:
    [data]      the loader and what the regression target bounds
    [model]     the architecture: `yolox` (capacity tier), plus the pretraining prototype keys
                (`bottleneck_expansion`, `pretrained`)
    [training]  schedule, run folder, device

`weight_decay` (5e-4) and the cosine schedule are not configurable: they were never flags.
"""
from __future__ import annotations

from pathlib import Path

from ..crop import BOX_SOURCES
from .data import TEMPORAL_INPUTS
from .yolox import YOLOX_TIERS

# THE ALLOWED KEYS, PER BLOCK. Anything else raises -- see module docstring. These are the
# one-to-one names of the argparse flags `scripts/train_detector.py` used to take (minus
# `--boxes`' dash), so a config value means exactly what the flag meant.
DATA_KEYS = frozenset({    'path', 'boxes', 'min_crop_dim', 'input_wh', 'min_box_px', 'max_input_px',
    'frames_per_group', 'val_frames_per_group', 'annot_frac', 'augment', 'augment_strong',
    'rotate_deg',
    'reduce', 'keypoints', 'hflip', 'tile_wh', 'tile_scale', 'tile_bg_per_frame',
    'use_regions', 'ignore_present', 'temporal_input',
})
MODEL_KEYS = frozenset({'yolox', 'bottleneck_expansion', 'pretrained', 'p2'})
TRAINING_KEYS = frozenset({
    'out', 'iters', 'batch_size', 'lr', 'num_workers', 'seed', 'device', 'eval_every',
    'eval_batches', 'kpt_weight', 'kpt_score_weight', 'iou_aware_obj', 'iou_aware_warmup',
    'max_pos_per_gt', 'box_weight', 'weight_decay',
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
    # `configs/detector.toml` ships `out = ""` on the promise that `--out` fills it in; checking
    # `train.get('out')` before this ran the promise into the requirement. `path` has no CLI
    # override to rescue it, so its check stays where it was.
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
              'iou_aware_warmup', 'max_pos_per_gt'):
        train[k] = int(train.get(k, {'iters': 20000, 'batch_size': 16, 'num_workers': 8,
                                     'seed': 0, 'eval_every': 2000, 'eval_batches': 25,
                                     'iou_aware_warmup': 2000, 'max_pos_per_gt': 0}[k]))
    # T2.3: the BCE objectness target at a positive anchor becomes the detached IoU between its
    # predicted and GT box, instead of a hard 1.0, once past `iou_aware_warmup` iterations.
    # Default OFF -- byte-identical to every checkpoint on record.
    train['iou_aware_obj'] = bool(train.get('iou_aware_obj', False))
    for k in ('min_crop_dim', 'min_box_px', 'max_input_px', 'frames_per_group',
              'val_frames_per_group', 'tile_bg_per_frame'):
        data[k] = int(data.get(k, {'min_crop_dim': 64, 'min_box_px': 32,
                                   'max_input_px': 4 * 416 * 416, 'frames_per_group': 40,
                                   'val_frames_per_group': 8,
                                   'tile_bg_per_frame': 1}[k]))
    # T2.4: `box_weight` was `detector_loss`'s own hardcoded default (5.0), never exposed to a
    # config or CLI flag -- "the two untuned scalars" alongside `lr`. 5.0 here is BYTE-IDENTICAL
    # to every checkpoint on record, since that is also `detector_loss`'s own Python default; this
    # key only matters once a config states something else.
    # AdamW's DECOUPLED decay: p <- p*(1 - lr*wd). 5e-4 was hardcoded and is YOLOX's *SGD* number,
    # where decay is coupled into the gradient; carried into AdamW at lr=1e-3 it is 5e-7 per step,
    # i.e. a total shrink of x0.995 over a 20000-step cosine -- half a percent, on a detector every
    # capacity sweep calls generalisation-limited (report 28). 5e-4 stays the DEFAULT and is
    # byte-identical to every checkpoint on record; this key only matters once a config says else.
    for k in ('lr', 'kpt_weight', 'kpt_score_weight', 'box_weight', 'weight_decay'):
        train[k] = float(train.get(k, {'lr': 1e-3, 'kpt_weight': 1.0, 'kpt_score_weight': 1.0,
                                       'box_weight': 5.0, 'weight_decay': 5e-4}[k]))
    if train['weight_decay'] < 0:
        raise SystemExit(f"[training].weight_decay must be >= 0, got {train['weight_decay']}.")
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
    # T4.2: frame t-1 stacked beside frame t. Default `'none'` is byte-identical to every
    # checkpoint on record. See `TEMPORAL_INPUT_CHANNELS` (`tailcyclenet/detector/data.py`) for
    # what each mode does to the stem's input width.
    data['temporal_input'] = str(data.get('temporal_input', 'none'))
    if data['temporal_input'] not in TEMPORAL_INPUTS:
        raise SystemExit(f"[data].temporal_input must be one of {TEMPORAL_INPUTS}, got "
                         f"{data['temporal_input']!r}.")
    if data['temporal_input'] != 'none' and data['augment_strong']:
        raise SystemExit('[data].temporal_input != "none" is undefined under '
                         '[data].augment_strong (mosaic-lite): the pasted source item needs its '
                         'own t-1 frame too, not built yet. See BoxDataset.__init__.')
    # P(a training draw comes from an `annotated` session), the detector's counterpart to
    # `LoaderConfig.annot_frac`. TOML has no null: absent (or an empty string) means "do not
    # weight", which keeps `ChunkShuffle` and is byte-identical to every checkpoint on record.
    # Also INERT on a single-cohort split -- `BoxDataset.cohort_weights` returns None there, so
    # 3dpop/calms21/branson-fly are unaffected whatever this says.
    af = data.get('annot_frac', None)
    data['annot_frac'] = None if af in (None, '', []) else float(af)
    if data['annot_frac'] is not None and not 0.0 <= data['annot_frac'] <= 1.0:
        raise SystemExit(f"[data].annot_frac must be in [0, 1], got {data['annot_frac']}.")
    data['boxes'] = str(data.get('boxes', 'instances'))
    model['yolox'] = str(model.get('yolox', 'tiny'))

    # T4.1. `bottleneck_expansion`: 0.5 (default) is byte-identical to every checkpoint on record;
    # 1.0 is the shape a Megvii COCO backbone actually loads into (see `yolox.Bottleneck`).
    # `pretrained`: '' (default, from scratch, every run on record), 'coco' (load the tier's COCO
    # backbone -- `detector.load_coco_backbone`), or ANY OTHER non-empty string is a PATH to an
    # in-domain backbone-only checkpoint (`scripts/pretrain_detector_backbone.py` ->
    # `detector.load_pretrained_backbone`). Unlike 'coco', a path has no tier or
    # bottleneck_expansion restriction -- the pretrain script builds whatever architecture its own
    # config names, so the two just need to AGREE, and `load_pretrained_backbone` is what checks
    # that agreement. Required to exist NOW, at config load: a typo'd path should fail before
    # 20000 iterations of training a randomly-initialised "pretrained" backbone, not after.
    model['bottleneck_expansion'] = float(model.get('bottleneck_expansion', 0.5))
    model['pretrained'] = str(model.get('pretrained', ''))
    if model['yolox'] == 'trimmed' and model['bottleneck_expansion'] != 0.5:
        raise SystemExit(
            f"[model].bottleneck_expansion={model['bottleneck_expansion']} was set alongside "
            "yolox='trimmed', but trimmed's backbone does not take it -- it stays at 0.5 "
            "permanently. Use a canonical tier (one of the YOLOX_TIERS names) for a "
            "COCO-compatible backbone.")
    if model['pretrained'] == 'coco':
        if model['yolox'] == 'trimmed':
            raise SystemExit("[model].pretrained='coco' requires a canonical yolox tier -- "
                             "'trimmed' has no COCO counterpart to load.")
        if model['bottleneck_expansion'] != 1.0:
            raise SystemExit(
                "[model].pretrained='coco' requires [model].bottleneck_expansion=1.0 -- at 0.5 "
                "every bottleneck conv is half Megvii's width and the load would silently take "
                "only 19 of 35 backbone tensors.")
    elif model['pretrained'] not in ('', 'coco'):
        if not Path(model['pretrained']).exists():
            raise SystemExit(
                f"[model].pretrained={model['pretrained']!r}: no such file -- expected '' (from "
                "scratch), 'coco', or a path to a T4.1b in-domain backbone checkpoint from "
                "scripts/pretrain_detector_backbone.py.")

    # T4.3: a stride-4 FPN level, on top of EITHER backbone (canonical tier or `trimmed`) --
    # unlike `bottleneck_expansion`, not tier-restricted. Default `false` is byte-identical to
    # every checkpoint on record (`YOLOXNano`'s own default).
    model['p2'] = bool(model.get('p2', False))
    return cfg
