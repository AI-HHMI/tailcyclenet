"""Hybrid CNN-transformer backbone for the detector.

Satisfies `yolox.py`'s backbone contract: forward returns a tuple `(p2, p3, p4, p5)` (when
`p2=True`) or `(p3, p4, p5)`, at strides `(4, 8, 16, 32)` or `(8, 16, 32)`, with
`self.out_channels` recording each level's channel count. `PAFPN` and `Head` are unchanged.
"""
import torch.nn as nn

from .yolox import conv_norm_act


class TransformerBlock(nn.Module):
    """Pre-norm transformer block for 2D feature maps.

    Reshapes (B,C,H,W) -> (B,H*W,C) for self-attention, then back. LayerNorm on the token dim,
    `nn.MultiheadAttention` (batch_first=True) -- trained from scratch, no pretrained weights.
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, zero_init=False):
        """Build the pre-norm block: layernorm, attention, MLP. `zero_init` zeroes both residual
        branches' output projections, so the block starts as the identity."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, dim))
        if zero_init:
            for lin in (self.attn.out_proj, self.mlp[2]):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def forward(self, x):
        """Tokenised attention path with residual connections."""
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
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
        """Build the hybrid backbone: CNN stages at strides 2/4/8, transformers at 16/32.

        Notes.

        The transformer stages each start with a stride-2 conv for spatial downsampling -- the
        stride-16 stage halves to /16, then the stride-32 stage halves to /32 -- followed by
        `n_transformer_blocks` transformer blocks.
        """
        super().__init__()
        c = base_channels
        self.p2 = bool(p2)

        self.stem = nn.Sequential(
            conv_norm_act(in_channels, c, 3, 2),
            conv_norm_act(c, c, 3, 1))
        self.stage2 = nn.Sequential(
            conv_norm_act(c, c * 2, 3, 2),
            conv_norm_act(c * 2, c * 2, 3, 1))
        self.stage3 = nn.Sequential(
            conv_norm_act(c * 2, c * 4, 3, 2),
            conv_norm_act(c * 4, c * 4, 3, 1))

        self.stage4_down = conv_norm_act(c * 4, c * 8, 3, 2)
        self.stage4_blocks = nn.Sequential(
            *[TransformerBlock(c * 8, n_heads, mlp_ratio)
              for _ in range(n_transformer_blocks[0])])
        self.stage5_down = conv_norm_act(c * 8, c * 16, 3, 2)
        self.stage5_blocks = nn.Sequential(
            *[TransformerBlock(c * 16, n_heads, mlp_ratio)
              for _ in range(n_transformer_blocks[1])])

        self.out_channels = ((c * 2, c * 4, c * 8, c * 16) if self.p2
                             else (c * 4, c * 8, c * 16))

    def forward(self, x):
        """Run the CNN stages then the transformer stages; return 4 or 3 pyramid levels."""
        s2 = self.stage2(self.stem(x))
        s3 = self.stage3(s2)
        s4 = self.stage4_blocks(self.stage4_down(s3))
        s5 = self.stage5_blocks(self.stage5_down(s4))
        return (s2, s3, s4, s5) if self.p2 else (s3, s4, s5)


class HybridCSPBackbone(HybridBackbone):
    """The hybrid with its CNN half swapped for a canonical YOLOX tier's stem/dark2/dark3.

    Strides 2/4/8 are the tier's own `Focus` + CSP stages (module names `stem`/`dark2`/`dark3`,
    exactly `CSPDarknet`'s, so `load_coco_backbone` maps Megvii's COCO weights onto them); the
    stride-16/32 transformer stages are the hybrid's, from scratch. `pretrained_parameters()`
    names the COCO-initialised half so only it gets the backbone LR scale.
    """

    def __init__(self, tier='tiny', n_transformer_blocks=(4, 2), n_heads=8, mlp_ratio=4.0,
                 p2=True, in_channels=3, bottleneck_expansion=1.0, token_dims=(512, 1024)):
        """Build the tier's CNN stages, then the hybrid's two transformer stages."""
        nn.Module.__init__(self)
        from .yolox import YOLOX_TIERS, CSPDarknet
        depth_mul, width_mul, depthwise = YOLOX_TIERS[tier]
        cnn = CSPDarknet(width_mul, depth_mul, depthwise=depthwise,
                         bottleneck_expansion=bottleneck_expansion, p2=True,
                         in_channels=in_channels)
        self.p2 = bool(p2)
        self.stem, self.dark2, self.dark3 = cnn.stem, cnn.dark2, cnn.dark3
        c2, c3 = cnn.out_channels[0], cnn.out_channels[1]
        d4, d5 = token_dims
        self.stage4_down = conv_norm_act(c3, d4, 3, 2)
        self.stage4_blocks = nn.Sequential(
            *[TransformerBlock(d4, n_heads, mlp_ratio) for _ in range(n_transformer_blocks[0])])
        self.stage5_down = conv_norm_act(d4, d5, 3, 2)
        self.stage5_blocks = nn.Sequential(
            *[TransformerBlock(d5, n_heads, mlp_ratio) for _ in range(n_transformer_blocks[1])])
        self.out_channels = (c2, c3, d4, d5) if self.p2 else (c3, d4, d5)

    def pretrained_parameters(self):
        """The COCO-loadable CNN half's parameters."""
        return [p for m in (self.stem, self.dark2, self.dark3) for p in m.parameters()]

    def forward(self, x):
        """CSP stages at strides 2/4/8, transformer stages at 16/32."""
        s2 = self.dark2(self.stem(x))
        s3 = self.dark3(s2)
        s4 = self.stage4_blocks(self.stage4_down(s3))
        s5 = self.stage5_blocks(self.stage5_down(s4))
        return (s2, s3, s4, s5) if self.p2 else (s3, s4, s5)


class CSPTransformerBackbone(nn.Module):
    """A canonical YOLOX tier's WHOLE CSPDarknet (COCO-loadable, stem..dark5 incl. SPP) with the
    hybrid's transformer blocks inserted after dark4 (stride 16) and dark5 (stride 32).

    The blocks are ZERO-INITIALISED residuals, so at step 0 the backbone computes exactly the
    COCO network; attention is learned as a correction on top. Module names `stem`/`dark2..5` are
    `CSPDarknet`'s, so `load_coco_backbone` loads every conv (the blocks have no 4-D tensors).
    """

    def __init__(self, tier='m', n_transformer_blocks=(4, 2), n_heads=8, mlp_ratio=4.0,
                 p2=True, in_channels=3, bottleneck_expansion=1.0):
        """Build the tier's CSPDarknet and the two identity-initialised transformer stacks."""
        super().__init__()
        from .yolox import YOLOX_TIERS, CSPDarknet
        depth_mul, width_mul, depthwise = YOLOX_TIERS[tier]
        cnn = CSPDarknet(width_mul, depth_mul, depthwise=depthwise,
                         bottleneck_expansion=bottleneck_expansion, p2=True,
                         in_channels=in_channels)
        self.stem, self.dark2, self.dark3 = cnn.stem, cnn.dark2, cnn.dark3
        self.dark4, self.dark5 = cnn.dark4, cnn.dark5
        c2, c3, c4, c5 = cnn.out_channels
        self.blocks4 = nn.Sequential(*[TransformerBlock(c4, n_heads, mlp_ratio, zero_init=True)
                                       for _ in range(n_transformer_blocks[0])])
        self.blocks5 = nn.Sequential(*[TransformerBlock(c5, n_heads, mlp_ratio, zero_init=True)
                                       for _ in range(n_transformer_blocks[1])])
        self.p2 = bool(p2)
        self.out_channels = (c2, c3, c4, c5) if self.p2 else (c3, c4, c5)

    def pretrained_parameters(self):
        """The COCO CSPDarknet's parameters (not the transformer blocks)."""
        return [p for m in (self.stem, self.dark2, self.dark3, self.dark4, self.dark5)
                for p in m.parameters()]

    def forward(self, x):
        """CSPDarknet with attention after dark4 and dark5."""
        s2 = self.dark2(self.stem(x))
        s3 = self.dark3(s2)
        s4 = self.blocks4(self.dark4(s3))
        s5 = self.blocks5(self.dark5(s4))
        return (s2, s3, s4, s5) if self.p2 else (s3, s4, s5)
