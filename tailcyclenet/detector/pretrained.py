"""Load a Megvii COCO YOLOX backbone into this repo's canonical-tier `CSPDarknet`.

**PROTOTYPE, not yet wired into a shipped recipe.** Two facts make this non-trivial, and both
are corrected here, once, in the loader -- never in the data path, so a run that does not ask
for pretraining is byte-unchanged:

- **SCALE.** This repo feeds `[0, 1]` (`x / 255.0` in `detector/data.py`); Megvii's `0.1.1rc0`
  release trains on raw `[0, 255]` with no mean/std. Every conv here has no bias
  (`conv_norm_act`), so it is exactly LINEAR in the input -- `conv(w, 255*x) == conv(255*w, x)` --
  and only the FIRST conv touching raw pixels needs correcting.
- **CHANNEL ORDER.** This repo is RGB; Megvii is BGR. `Focus` makes this a reversal WITHIN each
  group of 3, not across all 12: the stem's 12 input channels are 4 spatial shifts x 3 colour
  channels each.

A third mismatch is structural, not a numeric correction, and is fixed in `yolox.py` rather than
here: `Bottleneck` used to halve its channel count unconditionally where Megvii's own `CSPLayer`
builds its inner `Bottleneck`s at `expansion=1.0`, so 16 of 35 backbone tensors could not load at
all under the shipped shape. `[model].bottleneck_expansion` fixes the SHAPE; this file assumes the
caller already built the model at `1.0` and RAISES if it did not, rather than silently taking a
partial `strict=False` load.

**Scope is the BACKBONE ONLY** (`model.backbone`). The neck does not transfer at any width,
because this repo unifies all three FPN levels to one output width where Megvii's neck has three
per-level widths.
"""
from pathlib import Path

import torch

from .yolox import YOLOX_TIERS

# The 4 Focus groups (top_left, bot_left, top_right, bot_right -- `Focus.forward`'s own
# concatenation order), each 3 channels, BGR in the Megvii checkpoint. Reversing each 3-block is
# its own inverse, so the SAME permutation converts BGR-trained weights to see RGB input:
# if w_new = w_orig[:, perm] and perm reverses each 3-block, then conv(w_new, x_rgb) ==
# conv(w_orig, x_bgr) exactly, because summing over the channel axis does not care which physical
# channel carries which colour as long as weight and input agree.
BGR_TO_RGB_FOCUS_PERM = [i for g in range(4) for i in (g * 3 + 2, g * 3 + 1, g * 3 + 0)]

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / 'scratch' / 'weights'


def _remap_backbone_key(k):
    """Megvii `backbone.backbone.<x>.conv.weight` -> this repo's `<x>.0.weight`, or None.

    Two structural differences and nothing else: the CSPDarknet sits at `backbone.backbone` inside
    Megvii's `YOLOPAFPN`, and Megvii's `BaseConv` is a Module with `.conv`/`.bn` where this repo's
    `conv_norm_act` is an `nn.Sequential`, so the conv is index `.0`. The `.bn.*` half is never
    remapped -- GroupNorm has no BatchNorm counterpart.
    """
    if not k.startswith('backbone.backbone.'):
        return None
    k = k[len('backbone.backbone.'):]
    if k.endswith('.conv.weight'):
        return k[:-len('.conv.weight')] + '.0.weight'
    return None


def load_coco_backbone(model, tier, weights_dir=None):
    """Load Megvii's COCO backbone into `model.backbone` (a `CSPDarknet`), IN PLACE.

    `model` must already be built at `bottleneck_expansion=1.0` (`YOLOXNano(..., version=tier,
    bottleneck_expansion=1.0)`) -- this function only corrects SCALE and CHANNEL ORDER, never
    SHAPE, and raises rather than accepting a partial load if the shape is wrong.

    Returns `(n_loaded, n_total)` backbone conv tensors (GroupNorm affine is never touched -- it
    stays at its fresh `nn.GroupNorm` init, weight 1 / bias 0, which is deliberate: conv weights
    trained under BatchNorm statistics landing in a GroupNorm net is exactly why the backbone
    wants a lower LR than the fresh neck/head, not a reason to fake BN stats that do not exist).
    """
    if tier == 'trimmed' or tier not in YOLOX_TIERS:
        raise ValueError(f"load_coco_backbone: tier must be a canonical YOLOX_TIERS name "
                         f"({sorted(YOLOX_TIERS)}), not {tier!r} -- 'trimmed' has no COCO "
                         "counterpart to load.")
    got = float(getattr(model, 'bottleneck_expansion', 0.5))
    if abs(got - 1.0) > 1e-9:
        raise ValueError(
            f'load_coco_backbone: model.bottleneck_expansion must be 1.0 (canonical) to accept a '
            f'COCO backbone, got {got!r}. At 0.5 every bottleneck conv is half Megvii\'s width and '
            'a strict=False load would silently take only 19 of 35 backbone tensors -- build the '
            "model with bottleneck_expansion=1.0 first.")
    w = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    p = w / f'yolox_{tier}.pth'
    if not p.exists():
        raise FileNotFoundError(
            f'{p}: no cached Megvii COCO checkpoint. Fetch it and cache it at this path (see '
            'scratch/check_coco_transfer.py for the ones already cached), or pass weights_dir=.')
    ck = torch.load(p, map_location='cpu', weights_only=False)
    src = ck['model'] if 'model' in ck else ck
    remapped = {}
    for k, v in src.items():
        rk = _remap_backbone_key(k)
        if rk is not None:
            remapped[rk] = v

    own = model.backbone.state_dict()
    to_load, n_loaded, n_total = {}, 0, 0
    for k, v in own.items():
        # GroupNorm affine -- no BN source.
        if v.dim() != 4:
            continue
        n_total += 1
        if k not in remapped or remapped[k].shape != v.shape:
            continue
        w_src = remapped[k].clone()
        if k == 'stem.conv.0.weight':
            # Megvii [0,255] -> this repo's [0,1] input, then BGR -> RGB per Focus group.
            w_src = w_src * 255.0
            w_src = w_src[:, BGR_TO_RGB_FOCUS_PERM]
        to_load[k] = w_src
        n_loaded += 1

    result = model.backbone.load_state_dict(to_load, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    return n_loaded, n_total


def load_pretrained_backbone(model, path):
    """Load an IN-DOMAIN backbone-only checkpoint (`scripts/pretrain_detector_backbone.py`)
    into `model.backbone`, IN PLACE.

    Unlike `load_coco_backbone`, no scale or channel-order correction: this repo's own pretraining
    loop already decodes through `BoxDataset`, i.e. already `[0, 1]` and RGB, so a backbone trained
    here and one fine-tuned here speak the same convention from the start. That is the whole point
    of the in-domain control -- it isolates "does PRETRAINING help" from "does leaving this repo's
    own domain for COCO's help".

    Architecture must match EXACTLY -- `version` (tier), `bottleneck_expansion` and `in_channels`
    all come from the checkpoint's own recorded facts (absent means the PRE-key-existing default
    for each) and are compared against `model`'s own attributes BEFORE touching
    `load_state_dict`, so a mismatch raises with a clear cause instead of a wall of shape-mismatch
    key names or a `strict=False` partial load. `p2` is NOT checked: it changes the NECK/head,
    never `model.backbone`'s own tensors, so a backbone pretrained at `p2=False` loads unchanged
    into a `p2=True` fine-tune.
    """
    ck = torch.load(Path(path), map_location='cpu', weights_only=False)
    tier = str(ck.get('yolox_version', getattr(model, 'version', '')))
    exp = float(ck.get('bottleneck_expansion', getattr(model, 'bottleneck_expansion', 0.5)))
    in_ch = int(ck.get('in_channels', getattr(model, 'in_channels', 3)))
    if tier != model.version:
        raise ValueError(f'{path}: pretrained at yolox={tier!r}, model built at '
                         f'yolox={model.version!r} -- these must match exactly.')
    if abs(exp - model.bottleneck_expansion) > 1e-9:
        raise ValueError(f'{path}: pretrained at bottleneck_expansion={exp:g}, model built at '
                         f'{model.bottleneck_expansion:g} -- these must match exactly.')
    if in_ch != model.in_channels:
        raise ValueError(f'{path}: pretrained at in_channels={in_ch}, model built at '
                         f'{model.in_channels} -- these must match exactly.')
    backbone_state = ck.get('backbone_state')
    if backbone_state is None:
        raise ValueError(f'{path}: no "backbone_state" key -- not a '
                         'scripts/pretrain_detector_backbone.py checkpoint.')
    own = model.backbone.state_dict()
    if set(own) != set(backbone_state):
        raise ValueError(f'{path}: backbone key set does not match this model\'s backbone -- '
                         'built with a different architecture than the tier/expansion/in_channels '
                         'checks above should have already caught.')
    result = model.backbone.load_state_dict(backbone_state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    return len(backbone_state)
