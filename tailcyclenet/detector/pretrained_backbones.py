"""ImageNet / DINOv2-initialised detector backbones (`[model].yolox = "convnext-t" | "dinov2-s" |
"dinov2-b"`).

Both satisfy `yolox.py`'s backbone contract: forward returns `(p2, p3, p4, p5)` (or `(p3, p4,
p5)` without `p2`) at strides 4/8/16/32, `self.out_channels` per level. The neck and head are the
usual PAFPN + Head.

- `ConvNeXtBackbone` -- torchvision ConvNeXt-T, ImageNet-1k. Natively hierarchical (strides
  4/8/16/32, widths 96/192/384/768) and LayerNorm throughout, so pretrained weights land with no
  BatchNorm statistics to lose (the BN->GN problem report 47 named for COCO YOLOX).
- `DinoV2Backbone` -- DINOv2 ViT-S/14 or ViT-B/14 with a Simple Feature Pyramid from four
  intermediate blocks (ViTDet). Restored from `74917c9` (deleted in `bfc2bab` after report 47
  measured it in-domain only); FIXED here: the input is ImageNet-normalised, which the old code
  never did (it fed raw [0, 1] pixels to a network trained on normalised ones).

Construction builds the ARCHITECTURE only (what `load_detector` needs); `load_weights()` loads the
pretrained weights (what training does once). Everything is read from
`$TAILCYCLENET_CACHE_DIR/torch/hub` (default `~/.cache/tailcyclenet/torch/hub`) -- the DINOv2
repo code itself is loaded from there with `source='local'`, so a compute node without internet
works once the cache is populated (`load_weights` downloads into it when it can).
`pretrained_parameters()` names the pretrained trunk, which alone gets the backbone LR scale.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolox import conv_norm_act, norm_groups

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONVNEXT_T_URL = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'
DINOV2_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vit{s}14/dinov2_vit{s}14_pretrain.pth'


def hub_dir() -> Path:
    """The torch-hub cache these backbones read (repo code + checkpoints)."""
    root = Path(os.environ.get('TAILCYCLENET_CACHE_DIR', Path.home() / '.cache' / 'tailcyclenet'))
    return root / 'torch' / 'hub'


def _state_dict(url):
    """A cached checkpoint from `url` under `hub_dir()/checkpoints` (downloads if absent)."""
    return torch.hub.load_state_dict_from_url(url, model_dir=str(hub_dir() / 'checkpoints'),
                                              map_location='cpu', progress=False)


class _ImageNetNorm(nn.Module):
    """(x - mean) / std for [0, 1] RGB input -- what both pretrained trunks were trained on."""

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


class DinoV2Backbone(nn.Module):
    """DINOv2 ViT/14 + Simple Feature Pyramid from blocks 2/5/8/11.

    The ViT emits one stride (14); transposed / strided convs build 4/8/16/32 and a bilinear
    resize lands each level on exactly H//s x W//s, which `PAFPN`/`Head` expect. The input is
    padded to a multiple of 14 with the normalised-space zero (i.e. the ImageNet mean colour).
    """

    LAYER_INDICES = (2, 5, 8, 11)

    def __init__(self, size='s', p2=True, in_channels=3, out_channels=(96, 192, 384, 384)):
        """Build the DINOv2 architecture from the cached hub repo (no weights) and the SFP."""
        super().__init__()
        if in_channels != 3:
            raise ValueError('dinov2 is a 3-channel ImageNet-statistics trunk')
        repo = hub_dir() / 'facebookresearch_dinov2_main'
        if not (repo / 'hubconf.py').exists():
            raise FileNotFoundError(
                f'{repo}: DINOv2 repo code not cached. On a host with internet run once: '
                f'TAILCYCLENET_CACHE_DIR=... python -c "import torch; torch.hub.set_dir('
                f"'{hub_dir()}'); torch.hub.load('facebookresearch/dinov2', "
                f"'dinov2_vit{size}14', trust_repo=True)\"")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='xFormers is not available')
            self.vit = torch.hub.load(str(repo), f'dinov2_vit{size}14', source='local',
                                      pretrained=False)
        self.size = size
        self.norm = _ImageNetNorm()
        self.patch_size = int(self.vit.patch_size)
        C = int(self.vit.embed_dim)
        self.embed_dim = C
        self.p2 = bool(p2)

        def up(cin, cout):
            """A 2x transposed-conv upsample with GroupNorm + SiLU."""
            return [nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2),
                    nn.GroupNorm(norm_groups(cout), cout), nn.SiLU(inplace=True)]
        if self.p2:
            self.adapt_p2 = nn.Sequential(*up(C, out_channels[0]),
                                          *up(out_channels[0], out_channels[0]))
        self.adapt_p3 = nn.Sequential(*up(C, out_channels[1]))
        self.adapt_p4 = conv_norm_act(C, out_channels[2], 1)
        self.adapt_p5 = conv_norm_act(C, out_channels[3], 3, 2)
        self.out_channels = tuple(out_channels) if self.p2 else tuple(out_channels[1:])

    def load_weights(self):
        """Load the DINOv2 pretrain checkpoint into the ViT; returns (n_loaded, n_total)."""
        sd = _state_dict(DINOV2_URL.format(s=self.size))
        res = self.vit.load_state_dict(sd, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        return len(sd), len(self.vit.state_dict())

    def pretrained_parameters(self):
        """The DINOv2 ViT's parameters."""
        return list(self.vit.parameters())

    def forward(self, x):
        """Tap four ViT blocks and adapt each to its stride."""
        B, _, H, W = x.shape
        x = self.norm(x)
        ph = (-H) % self.patch_size
        pw = (-W) % self.patch_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), value=0.0)
        h_tok, w_tok = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        feats = self.vit.get_intermediate_layers(x, n=list(self.LAYER_INDICES))
        maps = [f.reshape(B, h_tok, w_tok, self.embed_dim).permute(0, 3, 1, 2).contiguous()
                for f in feats]
        adapters = ([self.adapt_p2] if self.p2 else []) + [self.adapt_p3, self.adapt_p4,
                                                           self.adapt_p5]
        strides = ((4,) if self.p2 else ()) + (8, 16, 32)
        taps = maps if self.p2 else maps[1:]
        return tuple(F.interpolate(a(m), size=(H // s, W // s), mode='bilinear',
                                   align_corners=False)
                     for a, m, s in zip(adapters, taps, strides))
