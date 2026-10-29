"""Detector checkpoints: resolve a run folder to a file, and build the `LoadedDetector` it describes."""
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .yolox import YOLOXNano
from ..data.crop import crop_rule


def resolve_detector_checkpoint(path, checkpoint='latest'):
    """Resolve a detector file from a run directory.

    Directory deployment defaults to the highest numbered checkpoint, not historical
    ``detector.pth`` (which is the validation-selected *best* checkpoint). ``last`` and ``best``
    remain explicit selectors; an explicit filename is also an override. ``latest`` only checks
    that the filename's iteration matches the checkpoint's own saved iteration -- it does not
    require the run's configured iteration count or metrics log to claim training completed, so
    a stopped run's highest numbered checkpoint still resolves. Only ``last`` requires
    ``detector_last.pth`` to exist.
    """
    p = Path(path)
    if not p.is_dir():
        return p
    selector = str(checkpoint or 'latest')
    if selector == 'last':
        out = p / 'detector_last.pth'
        if not out.exists():
            raise ValueError(f'{p}: --detector-checkpoint last requested, but detector_last.pth '
                             'is absent')
        return out
    if selector == 'best':
        out = p / 'detector.pth'
        if not out.exists():
            raise ValueError(f'{p}: --detector-checkpoint best requested, but detector.pth is absent')
        return out
    if selector != 'latest':
        out = p / selector
        if not out.exists():
            raise ValueError(f'{p}: explicit detector checkpoint {selector!r} does not exist')
        return out

    files = []
    for candidate in p.glob('detector_it*.pth'):
        match = re.fullmatch(r'detector_it(\d+)\.pth', candidate.name)
        if match:
            files.append((int(match.group(1)), candidate))
    if not files:
        raise ValueError(
            f'{p}: latest detector checkpoint requested, but no detector_it*.pth files exist; '
            'pass --detector-checkpoint best or an explicit filename for a legacy run')
    files.sort(reverse=True)
    for iteration, candidate in files:
        ckpt = torch.load(candidate, map_location='cpu', weights_only=False)
        saved_iteration = ckpt.get('iteration')
        if saved_iteration is not None and int(saved_iteration) == iteration:
            return candidate

    raise ValueError(f'{p}: no numbered detector checkpoint has matching iteration metadata')


@dataclass
class LoadedDetector:
    """A detector checkpoint, built and ready: the model plus the facts it was trained under.

    `input_wh`, `min_crop_dim`, `box_source` and `tile_scale` are part of the weights: the
    letterbox a detector was trained under decides what an animal looks like to it, and a mismatch
    serves boxes from a different crop rule silently. `tile_scale` is None for an untiled run.
    `obj_quantiles` describe the objectness distribution this checkpoint produces (`--det-score`
    is not portable across detector generations). `path` is the resolved checkpoint file.
    """
    model: torch.nn.Module
    input_wh: tuple
    dataset: str
    min_crop_dim: int
    reduce: bool
    # the pad rule its targets were cut with (data/crop.py)
    crop_pad_frac: float
    box_source: str
    tile_scale: float | None
    obj_quantiles: dict
    path: Path


def load_detector(path, device='cpu', checkpoint='latest'):
    """Build the detector a checkpoint (or run directory) describes -> `LoadedDetector`.

    For a run directory, `checkpoint='latest'` selects and verifies the highest complete
    `detector_it*.pth`; `'last'` selects `detector_last.pth`; `'best'` the historical
    `detector.pth`. A size the architecture cannot take is refused here, not in the first forward.
    `yolox_version`, `p2`, `shared_head` and `fpn_upsample` record the architecture the weights
    were shaped for. An absent `yolox_version` is a pre-key `trimmed` detector; it and the
    canonical tiers are deleted architectures (v0.3.4 is the last release that loads them), as is
    a keypoint branch (`n_keypoints > 0`) or a non-RGB stem (`in_channels != 3`).

    Absent `norm` means BatchNorm, which is refused (the model is GroupNorm). `tile_scale` is
    meaningless without `tile_wh`, so it is dropped (untiled runs record 1.0).
    `box_target`/`antialias`/`input_norm` (what the head regresses, how its input was resampled
    and normalised) ride on the model for `detect_raw`; absent = crop-rule boxes, bilinear
    shrinking, no normalisation.
    """
    p = resolve_detector_checkpoint(path, checkpoint=checkpoint)
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f'{p}: checkpoint must be a dictionary, got {type(ckpt).__name__}')
    if ckpt.get('kind', 'detector') != 'detector':
        raise ValueError(f"{p}: this is a {ckpt['kind']!r} checkpoint, not a detector one")
    wh = ckpt.get('input_wh') or ckpt.get('det_input_wh')
    if wh is None:
        raise ValueError(f'{p}: no input_wh in the checkpoint -- a posetail-pose detector keeps '
                         'it in its dataset config; it is not loadable here.')
    from .config import deleted_yolox_reason, input_size_error
    version = str(ckpt.get('yolox_version', 'trimmed'))
    why = deleted_yolox_reason(version)
    if why:
        raise ValueError(f'{p}: yolox={version!r} is a deleted architecture ({why}); retrain '
                         'this detector on the default (convnext-t).')
    if int(ckpt.get('n_keypoints', 0)):
        raise ValueError(f"{p}: trained with a keypoint branch (n_keypoints="
                         f"{ckpt['n_keypoints']}); the detector keypoint branch is deleted "
                         '(v0.3.4 is the last release that loads it).')
    if int(ckpt.get('in_channels', 3)) != 3:
        raise ValueError(f"{p}: in_channels={ckpt['in_channels']}; detection forwards one "
                         '3-channel frame at a time.')
    why = input_size_error(version, wh)
    if why:
        raise ValueError(f'{p}: input_wh {list(wh)}: {why}')
    norm = str(ckpt.get('norm', 'bn'))
    if norm != 'gn':
        raise ValueError(
            f'{p}: trained with {norm} normalisation; the model is GroupNorm now (there are no '
            'running statistics to load into). Retrain this detector -- see '
            '`tailcyclenet/detector/yolox.py:conv_norm_act` for why the switch was made.')
    model = YOLOXNano(version=version, p2=bool(ckpt.get('p2', False)),
                      shared_head=bool(ckpt.get('shared_head', True)),
                      fpn_upsample=str(ckpt.get('fpn_upsample', 'nearest') or 'nearest'))
    model.load_state_dict(ckpt['model_state'])
    trained_datasets = ckpt.get('datasets')
    if not trained_datasets:
        trained_datasets = [ckpt.get('dataset', '')] if ckpt.get('dataset') else []
    model.trained_datasets = [str(name) for name in trained_datasets]
    model.box_sources = {str(k): str(v) for k, v in (ckpt.get('box_sources') or {}).items()}
    model.box_target = str(ckpt.get('box_target', 'crop'))
    model.antialias = bool(ckpt.get('antialias', False))
    model.input_norm = str(ckpt.get('input_norm', 'none'))
    pad_frac, mcd = crop_rule(ckpt)
    model.min_crop_dim, model.crop_pad_frac = mcd, pad_frac
    model.box_source = str(ckpt.get('box_source', 'keypoints'))
    ts = ckpt.get('tile_scale')
    if ckpt.get('tile_wh') is not None and ts is None:
        raise ValueError(
            f'{p}: trained on tiles ({ckpt["tile_wh"]}) but carries no `tile_scale`, so the '
            'deployment input size cannot be derived. `input_wh` here is the TILE size, not the '
            'whole-frame size -- running the frame at it is a scale shift, not a smaller input.')
    return LoadedDetector(
        model=model.to(device).eval(), input_wh=tuple(wh), dataset=str(ckpt.get('dataset', '')),
        min_crop_dim=mcd, reduce=bool(ckpt.get('reduce', False)),
        crop_pad_frac=pad_frac,
        box_source=str(ckpt.get('box_source', 'keypoints')),
        tile_scale=None if ts is None or ckpt.get('tile_wh') is None else float(ts),
        obj_quantiles=dict(ckpt.get('obj_quantiles') or {}), path=Path(p))


def tiled_input_wh(src_wh, tile_scale):
    """Whole-frame input size a tile-trained detector must be deployed at.

    The invariant is the animal's size in INPUT pixels (a convnet is not scale-invariant), so a
    tile-trained detector must see the whole frame at native scale. Rounded to a multiple of 32,
    the coarsest stride, as `detector.train.input_wh_for` rounds.
    """
    return tuple(max(64, int(round(float(v) * float(tile_scale) / 32) * 32)) for v in src_wh)


