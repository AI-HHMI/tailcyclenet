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

`frames_per_group` is DELETED and now RAISES as an unknown key. The train loader indexes every
labelled frame and weights the draw view-uniformly within a cohort (`BoxDataset
.default_train_weights`), so the per-group frame cap no longer exists to be configured -- it was an
implicit, uncontrolled cohort-mixing knob (`dev/plans/detector_iteration_budget.md` SS3.1b).
`val_frames_per_group` STAYS: val/test keep the deterministic capped enumeration so every existing
val number stays comparable.
"""
from __future__ import annotations

from pathlib import Path

from ..crop import BOX_SOURCES
from .data import TEMPORAL_INPUTS
from .yolox import VIT_BACKBONES, YOLOX_TIERS

# THE ALLOWED KEYS, PER BLOCK. Anything else raises -- see module docstring. These are the
# one-to-one names of the argparse flags `scripts/train_detector.py` used to take (minus
# `--boxes`' dash), so a config value means exactly what the flag meant.
DATA_KEYS = frozenset({    'path', 'boxes', 'min_crop_dim', 'input_wh', 'min_box_px', 'max_input_px',
    'val_frames_per_group', 'annot_frac', 'augment', 'augment_strong',
    'rotate_deg',
    'reduce', 'keypoints', 'hflip', 'tile_wh', 'tile_scale', 'tile_bg_per_frame',
    'use_regions', 'ignore_present', 'temporal_input', 'negative_frac', 'negative_crop_frac',
    'scale_jitter', 'aug_switch_off_iter', 'alpha', 'det_scale',
    'hard_event_manifest', 'hard_event_frac', 'augment_copypaste', 'copypaste_max',
})
MODEL_KEYS = frozenset({'yolox', 'bottleneck_expansion', 'pretrained', 'p2'})
TRAINING_KEYS = frozenset({
    'out', 'iters', 'batch_size', 'lr', 'num_workers', 'seed', 'device', 'eval_every',
    'eval_batches', 'kpt_weight', 'kpt_score_weight', 'iou_aware_obj', 'iou_aware_warmup',
    'max_pos_per_gt', 'box_weight', 'weight_decay', 'no_decay_norm_bias', 'nms_iou_thresh',
    'nms_center_dist_thresh', 'neg_loss_weight', 'assignment', 'box_loss', 'focal_obj',
    'focal_gamma', 'tal_topk', 'tal_alpha', 'tal_beta', 'head_depthwise',
    'optimizer', 'muon_momentum', 'muon_lr_scale', 'warmup_steps', 'beta1', 'beta2',
    'freeze_backbone', 'shared_head', 'fpn_upsample', 'p2_bottomup', 'tal_soft_prior',
})
BLOCKS = (('data', DATA_KEYS), ('model', MODEL_KEYS), ('training', TRAINING_KEYS))
YOLOX_CHOICES = ('trimmed', *sorted(YOLOX_TIERS), *sorted(VIT_BACKBONES), 'hybrid', 'cspnext_s')


def _raise_unknown(block: str, cfg: dict, known: frozenset) -> None:
    """Exit with a listing of `cfg`'s keys that are not in `known`.

    Inputs: block -- TOML block name for the error message; cfg -- the block's dict;
            known -- the allowed key set.
    Side effects: raises SystemExit when any unknown key is present.
    """
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


def _float_pair(key: str, value):
    """A `[lo, hi]` list of FLOATS from TOML, or None when empty. `_pair`'s float sibling --
    `_pair` truncates to int (fine for a pixel size, wrong for `scale_jitter`'s (0.4, 1.0)-shaped
    range).
    """
    if value in (None, [], ''):
        return None
    err = f'{key}: expected [lo, hi], got {value!r}'
    try:
        pair = [float(v) for v in value]
    except (TypeError, ValueError):
        raise SystemExit(err)
    if len(pair) != 2 or pair[0] <= 0 or pair[1] < pair[0]:
        raise SystemExit(err)
    return tuple(pair)


def load_detector_config(path, out=None, iters=None, device=None) -> dict:
    """Load + validate a detector config; return the effective dict (blocks nested).

    Inputs:
        path -- the TOML config file.
        out / iters / device -- CLI overrides for [training], applied BEFORE the
            required-field check (`configs/detector.toml` ships `out = ""` on the promise
            that `--out` fills it in).
    Outputs:
        The effective dict (blocks nested) -- exactly what the run folder records as
        `config.toml`. Required fields are never defaulted (a missing dataset must fail at
        load, not silently train on the CWD); unknown keys and choice violations raise. An
        "absent" TOML pair (empty list -- TOML has no null) is normalised to None.
    Notes:
        Every optional key defaults to OFF, byte-identical to every checkpoint on record, so
        an arm moves one key at a time; shorthand references `dev/plans/detector_v2.md`.
        Highlights: `neg_loss_weight` is a DEAD KEY, REFUSED (the training call never
        receives it); `nms_center_dist_thresh` defaults ON (A5, '' restores the old off);
        `pretrained` accepts 'coco' or a path (required to exist NOW); the recommended
        shipped recipe is TAL + CIoU. The remaining keys (iou_aware_obj, box_weight,
        weight_decay, no_decay_norm_bias, det_scale, temporal_input, annot_frac, alpha,
        negative_frac, negative_crop_frac, scale_jitter, aug_switch_off_iter,
        bottleneck_expansion, p2, optimizer, freeze_backbone, shared_head, fpn_upsample,
        p2_bottomup, tal_soft_prior) each only matter once a config states something else.
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

    if out is not None:
        train['out'] = str(out)
    if iters is not None:
        train['iters'] = int(iters)
    if device is not None:
        train['device'] = str(device)

    if not data.get('path'):
        raise SystemExit('[data].path is required: ONE dataset root (has train/, optionally '
                         'val/ and test/). The detector is trained per dataset.')
    if not train.get('out'):
        raise SystemExit('[training].out is required: the run folder for checkpoints, '
                         'metrics.json and the recorded config, and neither the config nor '
                         '--out supplied one.')

    if data.get('boxes', 'instances') not in BOX_SOURCES:
        raise SystemExit(f'[data].boxes must be one of {BOX_SOURCES}, got '
                         f'{data["boxes"]!r}.')
    if model.get('yolox', 'tiny') not in YOLOX_CHOICES:
        raise SystemExit(f'[model].yolox must be one of {YOLOX_CHOICES}, got '
                         f'{model["yolox"]!r}.')

    data['input_wh'] = _pair('input_wh', data.get('input_wh'))
    data['tile_wh'] = _pair('tile_wh', data.get('tile_wh'))
    data['hard_event_manifest'] = str(data.get('hard_event_manifest', '') or '')
    hef = data.get('hard_event_frac')
    data['hard_event_frac'] = None if hef in (None, '') else float(hef)
    if data['hard_event_frac'] is not None and not 0.0 <= data['hard_event_frac'] <= 1.0:
        raise SystemExit(f"[data].hard_event_frac must be in [0, 1], got {hef}")

    for k in ('iters', 'batch_size', 'num_workers', 'seed', 'eval_every', 'eval_batches',
              'iou_aware_warmup', 'max_pos_per_gt'):
        train[k] = int(train.get(k, {'iters': 20000, 'batch_size': 16, 'num_workers': 8,
                                     'seed': 0, 'eval_every': 2000, 'eval_batches': 25,
                                     'iou_aware_warmup': 2000, 'max_pos_per_gt': 0}[k]))
    train['iou_aware_obj'] = bool(train.get('iou_aware_obj', False))
    for k in ('min_crop_dim', 'min_box_px', 'max_input_px',
              'val_frames_per_group', 'tile_bg_per_frame'):
        data[k] = int(data.get(k, {'min_crop_dim': 64, 'min_box_px': 32,
                                   'max_input_px': 4 * 416 * 416,
                                   'val_frames_per_group': 8,
                                   'tile_bg_per_frame': 1}[k]))
    for k in ('lr', 'kpt_weight', 'kpt_score_weight', 'box_weight', 'weight_decay'):
        train[k] = float(train.get(k, {'lr': 1e-3, 'kpt_weight': 1.0, 'kpt_score_weight': 1.0,
                                       'box_weight': 5.0, 'weight_decay': 5e-4}[k]))
    if train['weight_decay'] < 0:
        raise SystemExit(f"[training].weight_decay must be >= 0, got {train['weight_decay']}.")
    train['no_decay_norm_bias'] = bool(train.get('no_decay_norm_bias', False))
    train['nms_iou_thresh'] = float(train.get('nms_iou_thresh', 0.5))
    ncd = train.get('nms_center_dist_thresh', 0.5)
    train['nms_center_dist_thresh'] = None if ncd in (None, '', []) else float(ncd)
    train['neg_loss_weight'] = float(train.get('neg_loss_weight', 1.0))
    if train['neg_loss_weight'] != 1.0:
        raise SystemExit(
            f"[training].neg_loss_weight={train['neg_loss_weight']:g} was set, but nothing reads "
            'it -- `detector_loss` has no negative-item weighting term, so this would train '
            'byte-identically to the default 1.0 and report as an arm it is not. Wire it into '
            '`detector_loss` before configuring a non-default value, or unset the key.')
    for k, default in (('rotate_deg', 45.0), ('tile_scale', 1.0)):
        data[k] = float(data.get(k, default))
    data['det_scale'] = float(data.get('det_scale', 1.0))
    if data['det_scale'] <= 0:
        raise SystemExit(f"[data].det_scale must be > 0, got {data['det_scale']}.")
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
    data['temporal_input'] = str(data.get('temporal_input', 'none'))
    if data['temporal_input'] not in TEMPORAL_INPUTS:
        raise SystemExit(f"[data].temporal_input must be one of {TEMPORAL_INPUTS}, got "
                         f"{data['temporal_input']!r}.")
    if data['temporal_input'] != 'none' and data['augment_strong']:
        raise SystemExit('[data].temporal_input != "none" is undefined under '
                         '[data].augment_strong (mosaic-lite): the pasted source item needs its '
                         'own t-1 frame too, not built yet. See BoxDataset.__init__.')
    af = data.get('annot_frac', None)
    data['annot_frac'] = None if af in (None, '', []) else float(af)
    if data['annot_frac'] is not None and not 0.0 <= data['annot_frac'] <= 1.0:
        raise SystemExit(f"[data].annot_frac must be in [0, 1], got {data['annot_frac']}.")
    al = data.get('alpha', None)
    data['alpha'] = None if al in (None, '', []) else float(al)
    nf = data.get('negative_frac', None)
    data['negative_frac'] = None if nf in (None, '', []) else float(nf)
    if data['negative_frac'] is not None and not 0.0 <= data['negative_frac'] <= 1.0:
        raise SystemExit(f"[data].negative_frac must be in [0, 1], got {data['negative_frac']}.")
    ncf = data.get('negative_crop_frac', None)
    data['negative_crop_frac'] = None if ncf in (None, '', []) else float(ncf)
    if data['negative_crop_frac'] is not None and not 0.0 <= data['negative_crop_frac'] <= 1.0:
        raise SystemExit(f"[data].negative_crop_frac must be in [0, 1], got "
                         f"{data['negative_crop_frac']}.")
    data['scale_jitter'] = _float_pair('scale_jitter', data.get('scale_jitter'))
    data['aug_switch_off_iter'] = int(data.get('aug_switch_off_iter', 0) or 0)
    data['augment_copypaste'] = bool(data.get('augment_copypaste', False))
    data['copypaste_max'] = int(data.get('copypaste_max', 3))
    if data['copypaste_max'] < 1:
        raise SystemExit(f"[data].copypaste_max must be >= 1, got {data['copypaste_max']}")
    data['boxes'] = str(data.get('boxes', 'instances'))
    model['yolox'] = str(model.get('yolox', 'tiny'))

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
        if model['yolox'] in VIT_BACKBONES or model['yolox'] in ('hybrid', 'cspnext_s'):
            raise SystemExit(f"[model].pretrained='coco' has no counterpart for "
                             f"yolox={model['yolox']!r} -- use 'dinov2'/'dinov3' for a ViT "
                             "backbone or '' (from scratch) for hybrid/cspnext_s.")
        if model['bottleneck_expansion'] != 1.0:
            raise SystemExit(
                "[model].pretrained='coco' requires [model].bottleneck_expansion=1.0 -- at 0.5 "
                "every bottleneck conv is half Megvii's width and the load would silently take "
                "only 19 of 35 backbone tensors.")
    elif model['pretrained'] in ('dinov2', 'dinov3'):
        matching = {v for v, (_, _, key) in VIT_BACKBONES.items() if key == model['pretrained']}
        if model['yolox'] not in matching:
            raise SystemExit(f"[model].pretrained={model['pretrained']!r} requires yolox to be "
                             f"one of {sorted(matching)}, got {model['yolox']!r}.")
    elif model['pretrained'] not in ('', 'coco'):
        if not Path(model['pretrained']).exists():
            raise SystemExit(
                f"[model].pretrained={model['pretrained']!r}: no such file -- expected '' (from "
                "scratch), 'coco', 'dinov2'/'dinov3' (ViT only), or a path to a T4.1b in-domain "
                "backbone checkpoint from scripts/pretrain_detector_backbone.py.")

    model['p2'] = bool(model.get('p2', False))

    train['assignment'] = str(train.get('assignment', 'tal'))
    if train['assignment'] not in ('center', 'tal'):
        raise SystemExit(f"[training].assignment must be 'center' or 'tal', got "
                         f"{train['assignment']!r}")
    train['box_loss'] = str(train.get('box_loss', 'ciou'))
    if train['box_loss'] not in ('giou', 'ciou'):
        raise SystemExit(f"[training].box_loss must be 'giou' or 'ciou', got "
                         f"{train['box_loss']!r}")
    train['focal_obj'] = bool(train.get('focal_obj', False))
    train['focal_gamma'] = float(train.get('focal_gamma', 2.0))
    if train['focal_gamma'] < 0:
        raise SystemExit(f"[training].focal_gamma must be >= 0, got {train['focal_gamma']}")
    train['tal_topk'] = int(train.get('tal_topk', 13))
    train['tal_alpha'] = float(train.get('tal_alpha', 1.0))
    train['tal_beta'] = float(train.get('tal_beta', 6.0))
    if train['tal_topk'] < 1 or train['tal_alpha'] < 0 or train['tal_beta'] < 0:
        raise SystemExit('[training].tal_topk must be >= 1 and tal_alpha/tal_beta must be >= 0')
    train['head_depthwise'] = train.get('head_depthwise', None)
    if train['head_depthwise'] not in (None, True, False):
        raise SystemExit('[training].head_depthwise must be true, false, or absent')

    train['optimizer'] = str(train.get('optimizer', 'adamw'))
    if train['optimizer'] not in ('adamw', 'muon'):
        raise SystemExit(f"[training].optimizer must be 'adamw' or 'muon', got "
                         f"{train['optimizer']!r}")
    train['muon_momentum'] = float(train.get('muon_momentum', 0.95))
    train['muon_lr_scale'] = float(train.get('muon_lr_scale', 1.0))
    train['warmup_steps'] = int(train.get('warmup_steps', 500))
    train['beta1'] = float(train.get('beta1', 0.9))
    train['beta2'] = float(train.get('beta2', 0.95))

    train['freeze_backbone'] = bool(train.get('freeze_backbone', False))

    train['shared_head'] = bool(train.get('shared_head', True))
    train['fpn_upsample'] = str(train.get('fpn_upsample', 'nearest'))
    if train['fpn_upsample'] not in ('nearest', 'bilinear'):
        raise SystemExit(f"[training].fpn_upsample must be 'nearest' or 'bilinear', got "
                         f"{train['fpn_upsample']!r}")
    train['p2_bottomup'] = bool(train.get('p2_bottomup', False))
    train['tal_soft_prior'] = bool(train.get('tal_soft_prior', False))
    return cfg
