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


class TransformerBlock(nn.Module):
    """Pre-norm transformer block for 2D feature maps.

    Reshapes (B,C,H,W) -> (B,H*W,C) for self-attention, then back. LayerNorm on the token dim,
    `nn.MultiheadAttention` (batch_first=True) -- trained from scratch, no pretrained weights.
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)           # (B, H*W, C)
        t = self.norm1(tokens)
        tokens = tokens + self.attn(t, t, t, need_weights=False)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class HybridBackbone(nn.Module):
    """CNN stem (strides 2/4/8) + transformer blocks (strides 16/32).

    Trained from scratch. Gets global attention at the coarse levels where objectness is
    decided, without ViT cost at the fine (stride-4) level.

    At the shipped defaults (`base_channels=64`, `n_transformer_blocks=(4, 2)`) the backbone is
    **44.85M params, not the ~4.5M the plan doc originally estimated** -- the transformer blocks'
    MLP (4x expansion) at 512- and 1024-dim tokens dominate (37.8M of the 44.85M), an order of
    magnitude past a CNN-only estimate. Confirmed by `dev/scratch/prototype_hybrid_backbone.py`;
    kept at these hyperparameters (matching the plan's literal spec) rather than shrunk, since
    Wave 1 measures accuracy/VRAM/wall-clock empirically rather than assuming a param target.
    """

    def __init__(self, base_channels=64, n_transformer_blocks=(4, 2), n_heads=8, mlp_ratio=4.0,
                p2=True, in_channels=3):
        super().__init__()
        c = base_channels  # 64 -> out_channels (128, 256, 512, 1024)
        self.p2 = bool(p2)

        # CNN stages: stride 2 -> 4 -> 8.
        self.stem = nn.Sequential(
            conv_norm_act(in_channels, c, 3, 2),         # /2
            conv_norm_act(c, c, 3, 1))                   # /2 still
        self.stage2 = nn.Sequential(
            conv_norm_act(c, c * 2, 3, 2),               # /4
            conv_norm_act(c * 2, c * 2, 3, 1))
        self.stage3 = nn.Sequential(
            conv_norm_act(c * 2, c * 4, 3, 2),           # /8
            conv_norm_act(c * 4, c * 4, 3, 1))

        # Transformer stages: stride 16 -> 32. Each starts with a stride-2 conv for spatial
        # downsampling, then N transformer blocks.
        self.stage4_down = conv_norm_act(c * 4, c * 8, 3, 2)    # /16
        self.stage4_blocks = nn.Sequential(
            *[TransformerBlock(c * 8, n_heads, mlp_ratio)
              for _ in range(n_transformer_blocks[0])])
        self.stage5_down = conv_norm_act(c * 8, c * 16, 3, 2)   # /32
        self.stage5_blocks = nn.Sequential(
            *[TransformerBlock(c * 16, n_heads, mlp_ratio)
              for _ in range(n_transformer_blocks[1])])

        self.out_channels = ((c * 2, c * 4, c * 8, c * 16) if self.p2
                             else (c * 4, c * 8, c * 16))

    def forward(self, x):
        s2 = self.stage2(self.stem(x))                    # stride 4,  c*2 channels
        s3 = self.stage3(s2)                              # stride 8,  c*4 channels
        s4 = self.stage4_blocks(self.stage4_down(s3))     # stride 16, c*8 channels
        s5 = self.stage5_blocks(self.stage5_down(s4))     # stride 32, c*16 channels
        return (s2, s3, s4, s5) if self.p2 else (s3, s4, s5)
