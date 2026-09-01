"""Load a Megvii COCO YOLOX backbone into this repo's canonical-tier `CSPDarknet`. Two facts make this non-trivial, and both
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
import os
import shutil
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

# NOT scratch/weights (repo-relative -- doesn't exist under a pip install, and site-packages is
# typically unwritable even in a checkout). A per-user cache, overridable for HPC/air-gapped
# nodes via $TAILCYCLENET_CACHE_DIR or `weights_dir=`/`--weights-dir`.
_DEFAULT_CACHE_DIR = Path(os.environ.get('TAILCYCLENET_CACHE_DIR',
                                         Path.home() / '.cache' / 'tailcyclenet')) / 'weights'

# Megvii's own tagged 0.1.1rc0 GitHub release assets -- a versioned URL (unlike aniposelib's
# branch HEAD), named yolox_<tier>.pth for exactly the tier names YOLOX_TIERS already uses.
_COCO_RELEASE = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0'


def resolve_coco_weights(tier: str, cache_dir: Path | None = None) -> Path:
    """A local, cached Megvii COCO YOLOX checkpoint for `tier`, downloading once if absent.

    Checked first: `cache_dir` (or $TAILCYCLENET_CACHE_DIR/weights, or ~/.cache/tailcyclenet/
    weights) already has `yolox_<tier>.pth` -- an admin can pre-populate this once on an
    air-gapped/no-internet compute node and every later run is untouched, byte-identical to the
    explicit `weights_dir=` contract `load_coco_backbone` has always had. Otherwise, download it
    from Megvii's own tagged 0.1.1rc0 GitHub release (the exact release this module's docstring
    already names) into a `.part` file, then atomic-rename -- so a killed job never leaves a
    truncated checkpoint that silently loads. Uses `urlopen(timeout=30)`, not `urlretrieve`
    (which has no timeout and can hang indefinitely on a routable-but-filtered host -- the
    common HPC shape, unlike a DNS failure, which fails fast on its own).
    """
    cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f'yolox_{tier}.pth'
    if p.exists():
        return p
    import urllib.request
    url = f'{_COCO_RELEASE}/yolox_{tier}.pth'
    tmp = p.with_suffix('.part')
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, 'wb') as f:
            shutil.copyfileobj(resp, f)
    except (OSError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        raise FileNotFoundError(
            f'{p}: no cached Megvii COCO checkpoint, and fetching {url} failed ({e}). On a host '
            f'with no internet access, pre-populate {p} yourself (copy it from a host that has '
            f'one, or pass weights_dir= / --weights-dir explicitly).') from e
    tmp.rename(p)
    return p


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

    An explicit `weights_dir` means "only ever read from here" -- byte-identical to the contract
    this function has always had, raising if the checkpoint is not already there. Only the
    default (`weights_dir=None`) path auto-fetches, via `resolve_coco_weights`.

    Returns `(n_loaded, n_total)` backbone conv tensors (GroupNorm affine is never touched -- it
    stays at its fresh `nn.GroupNorm` init, weight 1 / bias 0, which is deliberate: conv weights
    trained under BatchNorm statistics landing in a GroupNorm net is exactly why the backbone
    wants a lower LR than the fresh neck/head, not a reason to fake BN stats that do not exist).

    The stem's first conv (`stem.conv.0.weight`) also gets the two numeric corrections the module
    docstring sets up: Megvii [0,255] input -> this repo's [0,1] input (the loader feeds
    `x / 255.0`; the conv is bias-free so it is exactly linear in the input), then BGR -> RGB per
    Focus group via `BGR_TO_RGB_FOCUS_PERM`.
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
    if weights_dir is not None:
        p = Path(weights_dir) / f'yolox_{tier}.pth'
        if not p.exists():
            raise FileNotFoundError(
                f'{p}: no cached Megvii COCO checkpoint at the given weights_dir. Fetch it and '
                'cache it at this path, or omit weights_dir to auto-fetch it.')
    else:
        p = resolve_coco_weights(tier)
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
        if v.dim() != 4:
            continue
        n_total += 1
        if k not in remapped or remapped[k].shape != v.shape:
            continue
        w_src = remapped[k].clone()
        if k == 'stem.conv.0.weight':
            w_src = w_src * 255.0
            w_src = w_src[:, BGR_TO_RGB_FOCUS_PERM]
        to_load[k] = w_src
        n_loaded += 1

    result = model.backbone.load_state_dict(to_load, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    return n_loaded, n_total



