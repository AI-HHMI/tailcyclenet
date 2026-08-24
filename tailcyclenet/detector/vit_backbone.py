"""ViT and hybrid backbones for the detector.

Both classes satisfy `yolox.py`'s backbone contract: forward returns a tuple `(p2, p3, p4, p5)`
(when `p2=True`) or `(p3, p4, p5)`, at strides `(4, 8, 16, 32)` or `(8, 16, 32)`, with
`self.out_channels` recording each level's channel count. `PAFPN` and `Head` are unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolox import conv_norm_act, norm_groups


class ViTBackbone(nn.Module):
    """DINOv2/DINOv3 ViT backbone with a Simple Feature Pyramid (SFP) for detection.

    The ViT outputs tokens at ONE native stride (`patch_size`) -- every intermediate layer has
    the SAME spatial resolution, unlike a CNN's hierarchy. The SFP generates all four detection
    strides {4, 8, 16, 32} from four intermediate transformer layers using up/downsampling
    convolutions, each followed by `F.interpolate` to the EXACT spatial size the downstream
    `PAFPN`/`Head` expect (H // stride x W // stride) -- the transposed/strided convs land close
    to but not exactly on that size (`patch_size` is not generally a power of 2).

    `hub_repo`/`model_name` generalise across DINOv2 (`facebookresearch/dinov2`, patch_size=14,
    fully public weights) and DINOv3 (`facebookresearch/dinov3`, patch_size=16, weights GATED
    behind Meta's own license-request form -- `pretrained=True` 403s from
    `dl.fbaipublicfiles.com/dinov3` without an approved request; the architecture itself still
    builds and trains from scratch with `pretrained=False`). Both hub repos expose the identical
    `.patch_size` / `.embed_dim` / `.get_intermediate_layers()` surface this class reads, so one
    implementation serves both.
    """

    # Which intermediate layers to tap. Both DINOv2-S/B and DINOv3-S/B have 12 blocks (indices
    # 0-11); 2, 5, 8, 11 evenly sample the depth -- the ViTDet convention.
    LAYER_INDICES = [2, 5, 8, 11]

    def __init__(self, model_name='dinov2_vits14', p2=True, pretrained=True,
                out_channels=(96, 192, 384, 384), hub_repo='facebookresearch/dinov2'):
        super().__init__()
        self.p2 = bool(p2)
        # NOTE: no `verbose=` kwarg -- DINOv2's entrypoint accepts and swallows it, but DINOv3's
        # forwards **kwargs straight into `DinoVisionTransformer.__init__`, which does not.
        self.vit = torch.hub.load(hub_repo, model_name, pretrained=pretrained, trust_repo=True)
        self.patch_size = self.vit.patch_size    # 14 (DINOv2) or 16 (DINOv3)
        self.embed_dim = self.vit.embed_dim      # 384 for the S tier, 768 for B
        C = self.embed_dim

        if self.p2:
            self.adapt_p2 = nn.Sequential(
                nn.ConvTranspose2d(C, out_channels[0], kernel_size=2, stride=2),
                nn.GroupNorm(norm_groups(out_channels[0]), out_channels[0]),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=2, stride=2),
                nn.GroupNorm(norm_groups(out_channels[0]), out_channels[0]),
                nn.SiLU(inplace=True))
            self.adapt_p3 = nn.Sequential(
                nn.ConvTranspose2d(C, out_channels[1], kernel_size=2, stride=2),
                nn.GroupNorm(norm_groups(out_channels[1]), out_channels[1]),
                nn.SiLU(inplace=True))
            self.adapt_p4 = conv_norm_act(C, out_channels[2], 1)
            self.adapt_p5 = conv_norm_act(C, out_channels[3], 3, 2)
            self.out_channels = tuple(out_channels)
        else:
            # p2's own width (out_channels[0]) is unused here -- p3/p4/p5 take out_channels[1:4]
            # to match `self.out_channels` below.
            self.adapt_p3 = nn.Sequential(
                nn.ConvTranspose2d(C, out_channels[1], kernel_size=2, stride=2),
                nn.GroupNorm(norm_groups(out_channels[1]), out_channels[1]),
                nn.SiLU(inplace=True))
            self.adapt_p4 = conv_norm_act(C, out_channels[2], 1)
            self.adapt_p5 = conv_norm_act(C, out_channels[3], 3, 2)
            self.out_channels = tuple(out_channels[1:])

    def freeze_backbone(self):
        """Freeze the DINOv2 ViT weights. Only the SFP adapters remain trainable."""
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        B, _, H, W = x.shape
        # Pad input to a multiple of patch_size (required by DINOv2's patch embedding).
        pH = (self.patch_size - H % self.patch_size) % self.patch_size
        pW = (self.patch_size - W % self.patch_size) % self.patch_size
        if pH or pW:
            x = F.pad(x, (0, pW, 0, pH), value=0.45)  # ~ImageNet mean grey

        Hp, Wp = x.shape[2], x.shape[3]
        h_tok, w_tok = Hp // self.patch_size, Wp // self.patch_size

        feats = self.vit.get_intermediate_layers(x, n=self.LAYER_INDICES)
        spatial = [f.reshape(B, h_tok, w_tok, self.embed_dim).permute(0, 3, 1, 2).contiguous()
                  for f in feats]

        # Target size at stride S for the UNPADDED input (H, W) is (H // S, W // S) -- the
        # F.interpolate calls crop/resize the adapter output to that exact size.
        if self.p2:
            p2 = F.interpolate(self.adapt_p2(spatial[0]), size=(H // 4, W // 4),
                               mode='bilinear', align_corners=False)
            p3 = F.interpolate(self.adapt_p3(spatial[1]), size=(H // 8, W // 8),
                               mode='bilinear', align_corners=False)
            p4 = F.interpolate(self.adapt_p4(spatial[2]), size=(H // 16, W // 16),
                               mode='bilinear', align_corners=False)
            p5 = F.interpolate(self.adapt_p5(spatial[3]), size=(H // 32, W // 32),
                               mode='bilinear', align_corners=False)
            return p2, p3, p4, p5
        else:
            p3 = F.interpolate(self.adapt_p3(spatial[0]), size=(H // 8, W // 8),
                               mode='bilinear', align_corners=False)
            p4 = F.interpolate(self.adapt_p4(spatial[1]), size=(H // 16, W // 16),
                               mode='bilinear', align_corners=False)
            p5 = F.interpolate(self.adapt_p5(spatial[2]), size=(H // 32, W // 32),
                               mode='bilinear', align_corners=False)
            return p3, p4, p5
