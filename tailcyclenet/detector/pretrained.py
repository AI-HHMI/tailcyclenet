"""Load a Megvii COCO YOLOX backbone into this repo's canonical-tier `CSPDarknet`.

dev/plans/detector_accuracy.md T4.1 -- **PROTOTYPE, not yet wired into a shipped recipe.** Two
facts make this non-trivial, and both are corrected here, once, in the loader -- never in the data
path, so a run that does not ask for pretraining is byte-unchanged:

- **SCALE.** This repo feeds `[0, 1]` (`x / 255.0` in `detector/data.py`); Megvii's `0.1.1rc0`
  release trains on raw `[0, 255]` with no mean/std. Every conv here has no bias
  (`conv_norm_act`), so it is exactly LINEAR in the input -- `conv(w, 255*x) == conv(255*w, x)` --
  and only the FIRST conv touching raw pixels needs correcting; everything downstream sees an
  already-corrected activation, so the fix is one tensor, not one per layer.
- **CHANNEL ORDER.** This repo is RGB (`dataset.read_frames` does `COLOR_BGR2RGB`; decord gives
  RGB); Megvii is BGR (`cv2.imread`, unswapped). `Focus` makes this a reversal WITHIN each group of
  3, not across all 12: the stem's 12 input channels are 4 spatial shifts (top_left, bot_left,
  top_right, bot_right -- this repo's `Focus.forward` reproduces Megvii's own grouping order
  exactly, checked in `tests/test_detector.py`) x 3 colour channels each.

A third mismatch is structural, not a numeric correction, and is fixed in `yolox.py` rather than
here: `Bottleneck` used to halve its channel count unconditionally (`hidden = cout // 2`) where
Megvii's own `CSPLayer` builds its inner `Bottleneck`s at `expansion=1.0` (full width), so 16 of
35 backbone tensors could not load at all under the shipped shape. `[model].bottleneck_expansion`
(`yolox.py`) fixes the SHAPE; this file assumes the caller already built the model at `1.0` and
RAISES if it did not, rather than silently taking a partial `strict=False` load -- gotcha 12's
shape: a model that took 19 of 35 tensors trains to a healthy-looking curve with half the
pretraining silently absent.

**Scope is the BACKBONE ONLY** (`model.backbone`). The neck does not transfer at any width,
because this repo unifies all three FPN levels to one output width where Megvii's neck has three
per-level widths (see `yolox.py`'s own module docstring) -- widening it would cost params and
transfer nothing.

Weights: `scratch/check_coco_transfer.py` already cached Megvii's own release checkpoints at
`scratch/weights/yolox_{tiny,s}.pth`. `scratch/` is UNTRACKED (CLAUDE.md) -- this is a
research/prototype location, not a shipped weights path; a caller that wants this durable should
pass its own `weights_dir`.
"""
from pathlib import Path

import torch

from .yolox import YOLOX_TIERS

# The 4 Focus groups (top_left, bot_left, top_right, bot_right -- `Focus.forward`'s own
# concatenation order), each 3 channels, BGR in the Megvii checkpoint. Reversing each 3-block is
# its own inverse (a permutation matrix that is its own transpose), so the SAME permutation
# converts BGR-trained weights to see RGB input -- see the module docstring's derivation: if
# w_new = w_orig[:, perm] and perm reverses each 3-block, then conv(w_new, x_rgb) ==
# conv(w_orig, x_bgr) exactly, because summing over the channel axis does not care which physical
# channel carries which colour as long as weight and input agree.
BGR_TO_RGB_FOCUS_PERM = [i for g in range(4) for i in (g * 3 + 2, g * 3 + 1, g * 3 + 0)]

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / 'scratch' / 'weights'


def _remap_backbone_key(k):
    """Megvii `backbone.backbone.<x>.conv.weight` -> this repo's `<x>.0.weight`, or None.

    Two structural differences and nothing else (see `scratch/check_coco_transfer.py`, the dry
    count this supersedes for anything beyond a shape audit): the CSPDarknet sits at
    `backbone.backbone` inside Megvii's `YOLOPAFPN`, and Megvii's `BaseConv` is a Module with
    `.conv`/`.bn` where this repo's `conv_norm_act` is an `nn.Sequential`, so the conv is index
    `.0`. The `.bn.*` half is never remapped -- GroupNorm has no BatchNorm counterpart, and a key
    this function does not return is a key `load_coco_backbone` will never try to load.
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
            "model with bottleneck_expansion=1.0 first (dev/plans/detector_accuracy.md T4.1).")
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
        if v.dim() != 4:                                       # GroupNorm affine -- no BN source
            continue
        n_total += 1
        if k not in remapped or remapped[k].shape != v.shape:
            continue
        w_src = remapped[k].clone()
        if k == 'stem.conv.0.weight':
            w_src = w_src * 255.0             # Megvii [0,255] -> this repo's [0,1] input
            w_src = w_src[:, BGR_TO_RGB_FOCUS_PERM]     # Megvii BGR -> this repo's RGB, per group
        to_load[k] = w_src
        n_loaded += 1

    result = model.backbone.load_state_dict(to_load, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    return n_loaded, n_total
