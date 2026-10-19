"""Detector training config: load, validate, apply CLI overrides.

The detector is trained from a TOML config, the same way the pose side is (`scripts/train.py` +
`config.load_config`). One shipped recipe lives in `configs/detector.toml`; EVERY detector
config layers over it automatically via `load_config`'s `base` argument.
Absent keys inherit the shipped recipe; `extends` is deleted and raises by name. Keys are
validated against an explicit allowed set so typos cannot silently train at defaults.

Blocks:
    [data]      the loader and what the regression target bounds
    [model]     the architecture: `yolox` (`convnext-t` or `hybrid`), `pretrained`, `p2`
    [training]  schedule, run folder, device; optional `[wandb]` config enables tracking

`frames_per_group` is DELETED and now RAISES as an unknown key. The train loader indexes every
labelled frame and weights the draw view-uniformly within a cohort (`BoxDataset
.default_train_weights`), so the per-group frame cap no longer exists to be configured -- it was an
implicit, uncontrolled cohort-mixing knob (`dev/plans/detector_iteration_budget.md` SS3.1b).
`val_frames_per_group` STAYS: val/test keep the deterministic capped enumeration so every existing
val number stays comparable.

`dev/plans/detector_dead_code_removal.md` deleted a set of measured-refuted or never-completed
keys: `det_scale`, `scale_jitter`, `aug_switch_off_iter`, `ignore_present`, `negative_frac`,
`negative_crop_frac`, `hard_event_manifest`/`hard_event_frac`, `focal_obj`/`focal_gamma`,
`augment_copypaste`/`copypaste_max`, `neg_loss_weight`, `temporal_input`, `p2_bottomup`,
`tal_soft_prior`, `no_decay_norm_bias`, `freeze_backbone`, `head_depthwise`, the ViT backbones
and CSPNeXt. See that plan for the evidence behind each.

`use_regions` is DELETED with `regions.pq` itself: the format no longer has certified areas. A
camera-frame's negatives are governed by its `instances.pq` statuses (`BoxDataset
.negative_supervision_for`), which needs no key.
"""
from __future__ import annotations

from pathlib import Path

from ..data.crop import BOX_SOURCES
from .boxes import NMS_CENTER_DIST
from .yolox import ARCHITECTURES, PRETRAINED_TRUNKS

# THE ALLOWED KEYS, PER BLOCK. Anything else raises -- see module docstring. These are the
# one-to-one names of the argparse flags `scripts/train_detector.py` used to take (minus
# `--boxes`' dash), so a config value means exactly what the flag meant.
DATA_KEYS = frozenset({
    'path', 'boxes', 'min_crop_dim', 'input_wh', 'min_box_px', 'max_input_px',
    'val_frames_per_group', 'annot_frac', 'augment', 'augment_strong', 'rotate_deg',
    'reduce', 'hflip', 'tile_wh', 'tile_scale', 'tile_bg_per_frame',
    'alpha', 'balance_datasets', 'boxes_by_dataset',
    'box_target', 'antialias', 'grayscale_prob', 'vflip',
    'input_norm', 'scale_range', 'background_prob', 'invert_prob',
})
# Measured and deleted (report 73). Named here so an old config fails with the reason, not as a
# generic unknown key -- and never trains silently without the lever it asked for.
DELETED_DATA_KEYS = {
    'exposure_prob': 'null on both transfer pairs (report 73 §5)',
    'background_own': "the root's own estimated backgrounds never beat synthetic canvases "
                      '(report 73 §9); compositing is synthetic-only, `background_prob` alone',
    'background_generic': 'the generic COCO image pack never beat synthetic canvases (report 73 '
                          '§9); compositing is synthetic-only, `background_prob` alone',
    'background_weights': 'compositing is synthetic-only (report 73 §9); `background_prob` alone',
    'keypoints': 'the detector keypoint branch is deleted (dormant: every shipped detector had '
                 'n_keypoints = 0; detector keypoints are a declared non-goal; v0.3.4 is the last release that has it)',
}
DELETED_MODEL_KEYS = {
    'bottleneck_expansion': 'it shaped only the deleted canonical YOLOX tiers (v0.3.4 is the last release that has it)',
}
DELETED_TRAINING_KEYS = {
    'kpt_weight': 'the detector keypoint branch is deleted (v0.3.4 is the last release that has it)',
    'kpt_score_weight': 'the detector keypoint branch is deleted (v0.3.4 is the last release that has it)',
}
DELETED_KEYS = (('data', DELETED_DATA_KEYS), ('model', DELETED_MODEL_KEYS),
                ('training', DELETED_TRAINING_KEYS))
_LEGACY_ARCH = ('a legacy architecture, not shipped; v0.3.4 is the last release that builds and '
                'loads it')
DELETED_YOLOX_NAMES = {name: _LEGACY_ARCH for name in ('trimmed', 'nano', 'tiny', 's', 'm', 'l', 'x')}
DELETED_YOLOX = {
    'dinov2-': 'worse than convnext-t on both transfer pairs and in-domain (report 73 §8)',
    'hybrid-': 'worse than convnext-t on both transfer pairs (report 73 §8)',
    'cspvit-': 'worse than convnext-t on both transfer pairs (report 73 §8)',
}


def deleted_yolox_reason(version: str):
    """Why a deleted `[model].yolox` value was removed, or None for a live one."""
    if str(version) in DELETED_YOLOX_NAMES:
        return DELETED_YOLOX_NAMES[str(version)]
    return next((why for pre, why in DELETED_YOLOX.items() if str(version).startswith(pre)), None)


MODEL_KEYS = frozenset({'yolox', 'pretrained', 'p2'})
TRAINING_KEYS = frozenset({
    'out', 'iters', 'batch_size', 'lr', 'num_workers', 'seed', 'device', 'eval_every',
    'eval_batches', 'iou_aware_obj', 'iou_aware_warmup',
    'max_pos_per_gt', 'box_weight', 'weight_decay', 'nms_iou_thresh',
    'nms_center_dist_thresh', 'assignment', 'box_loss',
    'tal_topk', 'tal_alpha', 'tal_beta',
    'optimizer', 'muon_momentum', 'muon_lr_scale', 'warmup_steps', 'beta1', 'beta2',
    'shared_head', 'fpn_upsample', 'save_every',
})
BLOCKS = (('data', DATA_KEYS), ('model', MODEL_KEYS), ('training', TRAINING_KEYS))
YOLOX_CHOICES = ARCHITECTURES


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


INPUT_MULTIPLE = {'convnext-t': 32}


def input_size_error(version: str, wh):
    """Why `wh` cannot be fed to a `version` backbone, or None when it can.

    ConvNeXt's patchify/downsample convs floor `side / stride`, so a side that is not a multiple
    of the coarsest stride (32) breaks the PAFPN concat in the first forward. Every other
    architecture pads its strided convs and takes any size.
    """
    multiple = INPUT_MULTIPLE.get(str(version))
    if not multiple or wh is None or all(int(v) % multiple == 0 for v in wh):
        return None
    return (f'yolox={version!r} needs both input sides divisible by {multiple}, got '
            f'{list(wh)}; round each side to a multiple of {multiple}.')


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
        An ABSENT key inherits the shipped recipe's value (`configs/detector.toml`, the
        implicit base), so a bare config already runs the recommended arm and an experiment
        moves one key at a time; the code-level fallbacks only apply to keys the shipped file
        does not state. Shorthand references `dev/plans/detector_v2.md`.
        `nms_center_dist_thresh` defaults ON (A5, '' restores the old off); `pretrained`
        accepts '' or 'imagenet' (convnext-t; 'coco' is deleted); ABSENT means the
        architecture's own source (convnext-t -> 'imagenet', every other -> scratch), so an
        overlay that changes `yolox` never inherits a source that cannot load into it, and an
        explicit '' trains from scratch. Deleted keys / `yolox` values (report 73) raise with
        their reason. The recommended shipped recipe is TAL + CIoU.
        The remaining keys (iou_aware_obj, box_weight, weight_decay, annot_frac, alpha,
        p2, optimizer, shared_head, fpn_upsample) each only matter
        once a config states something else. The implicit base is `config.DETECTOR_CONFIG`
        (one source of truth with the pose side's `BASE_CONFIG`, both resolved via
        `importlib.resources` rather than a repo-relative path recomputed here, so a pip install
        works too), not a path this module computes on its own.
    """
    from tailcyclenet.config import DETECTOR_CONFIG, load_config

    cfg = load_config(Path(path), base=DETECTOR_CONFIG)
    missing = [b for b, _ in BLOCKS if b not in cfg]
    if missing:
        raise SystemExit(f'{path}: missing block(s) {missing}. A detector config needs '
                         f'[data], [model] and [training].')
    for block, deleted in DELETED_KEYS:
        gone = sorted(set(cfg[block]) & set(deleted))
        if gone:
            raise SystemExit(f'[{block}]: deleted key(s) ' + '; '.join(
                f'{k!r}: {deleted[k]}' for k in gone) + '. Remove them.')
    why = deleted_yolox_reason(cfg['model'].get('yolox', ''))
    if why:
        raise SystemExit(f"[model].yolox={cfg['model']['yolox']!r} is deleted: {why}. "
                         "Use 'convnext-t' (the default) or 'hybrid'.")
    for block, known in BLOCKS:
        _raise_unknown(block, cfg[block], known)
    data, model, train = cfg['data'], cfg['model'], cfg['training']

    if out is not None:
        train['out'] = str(out)
    if iters is not None:
        train['iters'] = int(iters)
    if device is not None:
        train['device'] = str(device)

    wandb_cfg = cfg.get('wandb')
    if wandb_cfg is not None:
        if not isinstance(wandb_cfg, dict):
            raise SystemExit('[wandb] must be a table.')
        _raise_unknown('wandb', wandb_cfg, frozenset({'project_name', 'path', 'mode'}))
        wandb_cfg.setdefault('project_name', 'tailcyclenet')
        wandb_cfg.setdefault('path', '.')
        wandb_cfg.setdefault('mode', 'offline')
        if not isinstance(wandb_cfg['project_name'], str) or not wandb_cfg['project_name']:
            raise SystemExit('[wandb].project_name must be a non-empty string.')
        if not isinstance(wandb_cfg['path'], str) or not wandb_cfg['path']:
            raise SystemExit('[wandb].path must be a non-empty string.')
        if wandb_cfg['mode'] not in ('online', 'offline'):
            raise SystemExit("[wandb].mode must be 'online' or 'offline'.")

    if not data.get('path'):
        raise SystemExit('[data].path is required: a dataset root (has train/, optionally '
                         'val/ and test/) or a folder whose children are dataset roots.')
    if not train.get('out'):
        raise SystemExit('[training].out is required: the run folder for checkpoints, '
                         'metrics.json and the recorded config, and neither the config nor '
                         '--out supplied one.')

    if data['boxes'] not in BOX_SOURCES:
        raise SystemExit(f'[data].boxes must be one of {BOX_SOURCES}, got '
                         f'{data["boxes"]!r}.')
    if model['yolox'] not in YOLOX_CHOICES:
        raise SystemExit(f'[model].yolox must be one of {YOLOX_CHOICES}, got '
                         f'{model["yolox"]!r}.')

    data['input_wh'] = _pair('input_wh', data.get('input_wh'))
    data['tile_wh'] = _pair('tile_wh', data.get('tile_wh'))

    for k in ('iters', 'batch_size', 'num_workers', 'seed', 'eval_every', 'eval_batches',
              'iou_aware_warmup', 'max_pos_per_gt'):
        train[k] = int(train[k])
    train['save_every'] = int(train.get('save_every', train['eval_every']))
    if train['save_every'] < 1:
        raise SystemExit('[training].save_every must be >= 1.')
    train['iou_aware_obj'] = bool(train['iou_aware_obj'])
    for k in ('min_crop_dim', 'min_box_px', 'max_input_px',
              'val_frames_per_group', 'tile_bg_per_frame'):
        data[k] = int(data[k])
    for k in ('lr', 'box_weight', 'weight_decay'):
        train[k] = float(train[k])
    if train['weight_decay'] < 0:
        raise SystemExit(f"[training].weight_decay must be >= 0, got {train['weight_decay']}.")
    train['nms_iou_thresh'] = float(train.get('nms_iou_thresh', 0.5))
    ncd = train.get('nms_center_dist_thresh', NMS_CENTER_DIST)
    train['nms_center_dist_thresh'] = None if ncd in (None, '', []) else float(ncd)
    for k in ('rotate_deg', 'tile_scale'):
        data[k] = float(data[k])
    data['augment'] = bool(data['augment'])
    data['augment_strong'] = bool(data['augment_strong'])
    data['reduce'] = bool(data['reduce'])
    data['hflip'] = bool(data['hflip'])
    data['vflip'] = bool(data['vflip'])
    data['antialias'] = bool(data['antialias'])
    data['box_target'] = str(data['box_target'])
    if data['box_target'] not in ('crop', 'extent'):
        raise SystemExit(f"[data].box_target must be 'crop' or 'extent', got "
                         f"{data['box_target']!r}.")
    data['grayscale_prob'] = float(data['grayscale_prob'])
    if not 0.0 <= data['grayscale_prob'] <= 1.0:
        raise SystemExit(f"[data].grayscale_prob must be in [0, 1], got {data['grayscale_prob']}.")
    data['input_norm'] = str(data['input_norm'])
    if data['input_norm'] not in ('none', 'equalize', 'percentile'):
        raise SystemExit(f"[data].input_norm must be 'none', 'equalize' or 'percentile', got "
                         f"{data['input_norm']!r}.")
    data['background_prob'] = float(data['background_prob'])
    data['invert_prob'] = float(data['invert_prob'])
    for k in ('background_prob', 'invert_prob'):
        if not 0.0 <= data[k] <= 1.0:
            raise SystemExit(f'[data].{k} must be in [0, 1], got {data[k]}.')
    sr = data['scale_range']
    if (not isinstance(sr, (list, tuple)) or len(sr) != 2
            or not 0 < float(sr[0]) <= float(sr[1])):
        raise SystemExit(f'[data].scale_range must be [lo, hi] with 0 < lo <= hi, got {sr!r}.')
    data['scale_range'] = [float(sr[0]), float(sr[1])]
    af = data['annot_frac']
    data['annot_frac'] = None if af in (None, '', []) else float(af)
    if data['annot_frac'] is not None and not 0.0 <= data['annot_frac'] <= 1.0:
        raise SystemExit(f"[data].annot_frac must be in [0, 1], got {data['annot_frac']}.")
    al = data.get('alpha', None)
    data['alpha'] = None if al in (None, '', []) else float(al)
    data['boxes'] = str(data['boxes'])
    bal = data['balance_datasets']
    if not isinstance(bal, bool):
        raise SystemExit(f'[data].balance_datasets must be true or false, got {bal!r}.')
    data['balance_datasets'] = bal
    by = data['boxes_by_dataset']
    if not isinstance(by, dict):
        raise SystemExit('[data].boxes_by_dataset must be a table of dataset-root name -> box '
                         f'source, got {by!r}.')
    bad = {k: v for k, v in by.items() if v not in BOX_SOURCES}
    if bad:
        raise SystemExit(f'[data].boxes_by_dataset: values must be one of {BOX_SOURCES}, got '
                         f'{bad}.')
    data['boxes_by_dataset'] = {str(k): str(v) for k, v in by.items()}
    model['yolox'] = str(model['yolox'])

    pre = model.get('pretrained')
    model['pretrained'] = PRETRAINED_TRUNKS.get(model['yolox'], '') if pre is None else str(pre)
    if model['pretrained'] == 'coco':
        raise SystemExit("[model].pretrained='coco' is deleted: it loaded Megvii's COCO backbone "
                         "into the canonical YOLOX tiers, which are deleted (v0.3.4 is the last release that has it).")
    if model['pretrained'] not in ('', 'imagenet'):
        raise SystemExit(
            f"[model].pretrained={model['pretrained']!r}: expected '' (from scratch) or "
            "'imagenet' (convnext-t).")
    if model['yolox'] in PRETRAINED_TRUNKS or model['pretrained'] == 'imagenet':
        want = PRETRAINED_TRUNKS.get(model['yolox'])
        if model['pretrained'] not in ('', want):
            raise SystemExit(f"[model].yolox={model['yolox']!r} loads {want!r} weights; "
                             f"pretrained={model['pretrained']!r} does not match it.")

    model['p2'] = bool(model['p2'])
    for key in ('input_wh', 'tile_wh'):
        why = input_size_error(model['yolox'], data[key])
        if why:
            raise SystemExit(f'[data].{key}: {why}')

    train['assignment'] = str(train['assignment'])
    if train['assignment'] not in ('center', 'tal'):
        raise SystemExit(f"[training].assignment must be 'center' or 'tal', got "
                         f"{train['assignment']!r}")
    train['box_loss'] = str(train['box_loss'])
    if train['box_loss'] not in ('giou', 'ciou'):
        raise SystemExit(f"[training].box_loss must be 'giou' or 'ciou', got "
                         f"{train['box_loss']!r}")
    train['tal_topk'] = int(train['tal_topk'])
    train['tal_alpha'] = float(train.get('tal_alpha', 1.0))
    train['tal_beta'] = float(train.get('tal_beta', 6.0))
    if train['tal_topk'] < 1 or train['tal_alpha'] < 0 or train['tal_beta'] < 0:
        raise SystemExit('[training].tal_topk must be >= 1 and tal_alpha/tal_beta must be >= 0')

    train['optimizer'] = str(train['optimizer'])
    if train['optimizer'] not in ('adamw', 'muon'):
        raise SystemExit(f"[training].optimizer must be 'adamw' or 'muon', got "
                         f"{train['optimizer']!r}")
    train['muon_momentum'] = float(train.get('muon_momentum', 0.95))
    train['muon_lr_scale'] = float(train.get('muon_lr_scale', 1.0))
    train['warmup_steps'] = int(train.get('warmup_steps', 500))
    train['beta1'] = float(train.get('beta1', 0.9))
    train['beta2'] = float(train.get('beta2', 0.95))

    train['shared_head'] = bool(train['shared_head'])
    train['fpn_upsample'] = str(train['fpn_upsample'])
    if train['fpn_upsample'] not in ('nearest', 'bilinear'):
        raise SystemExit(f"[training].fpn_upsample must be 'nearest' or 'bilinear', got "
                         f"{train['fpn_upsample']!r}")
    return cfg
