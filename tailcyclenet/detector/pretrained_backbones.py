"""The ImageNet-initialised detector backbone (`[model].yolox = "convnext-t"`, the shipped default).

It satisfies `yolox.py`'s backbone contract: forward returns `(p2, p3, p4, p5)` (or `(p3, p4,
p5)` without `p2`) at strides 4/8/16/32, `self.out_channels` per level. The neck and head are the
usual PAFPN + Head.

- `ConvNeXtBackbone` -- torchvision ConvNeXt-T, ImageNet-1k. Natively hierarchical (strides
  4/8/16/32, widths 96/192/384/768) and LayerNorm throughout, so pretrained weights land with no
  BatchNorm statistics to lose (the BN->GN problem report 47 named for COCO YOLOX).

Report 73 §8 measured it against COCO yolox tiny/m, DINOv2 ViT-S/B (with the ImageNet input
normalisation an earlier implementation lacked), a COCO-stem hybrid and COCO CSPDarknet + attention:
ConvNeXt-T won both transfer pairs with no in-domain cost; the others were deleted.

Construction builds the ARCHITECTURE only (what `load_detector` needs); `load_weights()` loads the
ImageNet weights (what training does once) from `$TAILCYCLENET_CACHE_DIR/torch/hub/checkpoints`
(default `~/.cache/tailcyclenet/...`), downloading them there on first use.
`pretrained_parameters()` names the pretrained trunk, which alone gets the backbone LR scale.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONVNEXT_T_URL = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'


def hub_dir() -> Path:
    """The torch-hub cache these backbones read (repo code + checkpoints)."""
    root = Path(os.environ.get('TAILCYCLENET_CACHE_DIR', Path.home() / '.cache' / 'tailcyclenet'))
    return root / 'torch' / 'hub'


def _state_dict(url):
    """A cached checkpoint from `url` under `hub_dir()/checkpoints` (downloads if absent).

    A compute node without internet gets an error naming the one file to pre-populate.
    """
    dest = hub_dir() / 'checkpoints' / url.rsplit('/', 1)[-1]
    try:
        return torch.hub.load_state_dict_from_url(url, model_dir=str(dest.parent),
                                                  map_location='cpu', progress=False)
    except (OSError, RuntimeError) as e:
        if dest.exists():
            raise
        raise FileNotFoundError(
            f'{dest}: ImageNet weights not cached, and fetching {url} failed ({e}). On a host '
            f'without internet, copy that file there from one that has it (or set '
            f'TAILCYCLENET_CACHE_DIR to a cache that does).') from e


class _ImageNetNorm(nn.Module):
    """(x - mean) / std for [0, 1] RGB input -- what the ImageNet trunk was trained on."""

    def __init__(self):
        """Register the ImageNet statistics as non-trainable buffers."""
        super().__init__()
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        """Normalise."""
        return (x - self.mean) / self.std


class ConvNeXtBackbone(nn.Module):
    """torchvision ConvNeXt-T trunk; `features[1,3,5,7]` are strides 4/8/16/32."""

    def __init__(self, p2=True, in_channels=3):
        """Build the ConvNeXt-T architecture (no weights)."""
        super().__init__()
        if in_channels != 3:
            raise ValueError('convnext-t is a 3-channel ImageNet trunk')
        from torchvision.models import convnext_tiny
        self.norm = _ImageNetNorm()
        self.features = convnext_tiny(weights=None).features
        self.p2 = bool(p2)
        widths = (96, 192, 384, 768)
        self.out_channels = widths if self.p2 else widths[1:]

    def load_weights(self):
        """Load ImageNet-1k weights into the trunk; returns (n_loaded, n_total)."""
        sd = {k[len('features.'):]: v for k, v in _state_dict(CONVNEXT_T_URL).items()
              if k.startswith('features.')}
        res = self.features.load_state_dict(sd, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        return len(sd), len(self.features.state_dict())

    def pretrained_parameters(self):
        """The ImageNet trunk's parameters."""
        return list(self.features.parameters())

    def forward(self, x):
        """Return the four (or three) stage outputs."""
        x = self.norm(x)
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (1, 3, 5, 7):
                outs.append(x)
        return tuple(outs) if self.p2 else tuple(outs[1:])

