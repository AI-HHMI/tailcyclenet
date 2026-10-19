"""Compact YOLOX-style box predictor over a `hybrid` or `convnext-t` backbone.

One class, so YOLOX simplifies hard:
  * a backbone (`convnext-t`, ImageNet ConvNeXt-T, shipped; or `hybrid`, a CNN-transformer trained
    from scratch) + PAFPN neck, strides 8/16/32 (plus 4 under `p2`)
  * decoupled anchor-free head with the CLASSIFICATION branch dropped -- a single class
    means objectness alone carries all the information
  * centre-prior or task-aligned assignment (see assign.py)
  * BCE(objectness) + GIoU/CIoU(box)

The bespoke `trimmed` net and the canonical YOLOX tiers (`nano`..`x`, with their COCO loader) are
deleted; v0.3.4 is the last release that builds or loads them (`config.DELETED_YOLOX`).

GroupNorm throughout, not BatchNorm -- see `conv_norm_act`. The detector reproduces the SAME crop
box the pose pipeline uses (`tailcyclenet.crop.crop_box_for_points`): either by regressing it
directly (`box_target = 'crop'`, legacy) or by regressing the tight extent the rule is a function
of and applying the rule after detection (`'extent'`, shipped -- see `detector/data.py`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


CH_PER_GROUP = 8

PRETRAINED_TRUNKS = {'convnext-t': 'imagenet'}
ARCHITECTURES = ('hybrid', *PRETRAINED_TRUNKS)
NECK_WIDTH = 256


def norm_groups(c, per_group=CH_PER_GROUP):
    """A group count that DIVIDES `c`, at about `per_group` channels each (walked down to a divisor)."""
    g = max(1, c // per_group)
    while c % g:
        g -= 1
    return g


def conv_norm_act(cin, cout, k=3, s=1, groups=1):
    """THE normalisation chokepoint: every conv in this net goes through here.

    GroupNorm, not BatchNorm, for two reasons that are one change. It has NO RUNNING STATISTICS,
    so train and inference are the same computation -- which is what makes training on crops and
    inferring on a whole frame safe, where BN's statistics would be collected on animal-rich
    crops and applied to a mostly-empty arena. And it is batch-independent, so an arm that can
    only hold a small batch is merely slower to optimise, not a differently-normalised model.
    """
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
        nn.GroupNorm(norm_groups(cout), cout),
        nn.SiLU(inplace=True))


class Bottleneck(nn.Module):
    """1x1 down-projection to half width, 3x3 conv, optional shortcut."""
    def __init__(self, cin, cout, shortcut=True):
        """Build the bottleneck: 1x1 down-projection, 3x3 conv, optional shortcut."""
        super().__init__()
        hidden = int(cout * 0.5)
        self.conv1 = conv_norm_act(cin, hidden, 1)
        self.conv2 = conv_norm_act(hidden, cout, 3)
        self.add = shortcut and cin == cout

    def forward(self, x):
        """Apply conv1 then conv2, shortcut-added when `self.add`."""
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class CSPLayer(nn.Module):
    """A CSP layer: cross-stage partial split, `n` Bottlenecks on one path, merge conv."""
    def __init__(self, cin, cout, n=1, shortcut=True):
        """Build the layer: two 1x1 splits, `n` Bottlenecks, one merge conv."""
        super().__init__()
        hidden = cout // 2
        self.conv1 = conv_norm_act(cin, hidden, 1)
        self.conv2 = conv_norm_act(cin, hidden, 1)
        self.conv3 = conv_norm_act(2 * hidden, cout, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut) for _ in range(n)])

    def forward(self, x):
        """CSP merge of the bottleneck path and the shortcut path."""
        return self.conv3(torch.cat([self.m(self.conv1(x)), self.conv2(x)], dim=1))


class PAFPN(nn.Module):
    """Top-down + bottom-up feature pyramid. `p2` adds a 4th, finer level (stride 4).

    Without `p2` the `p2`-only modules (`lat2`/`mrg2`/`down2`/`out3`) are not constructed. With
    it, p2 becomes the finest level (top-down only) and p3 gains the bottom-up refinement stage
    (`out3`, mirroring `out4`/`out5`).
    """

    def __init__(self, chans=(96, 192, 192), out=96, p2=False, fpn_upsample='nearest'):
        """Build the PAFPN neck: lateral convs, top-down merges, bottom-up merges.

        `fpn_upsample` is 'nearest' or 'bilinear' (RTMDet's choice; the shipped recipe).
        """
        super().__init__()
        self.p2 = bool(p2)
        self.fpn_upsample = str(fpn_upsample)
        if self.fpn_upsample not in ('nearest', 'bilinear'):
            raise ValueError(f"fpn_upsample must be 'nearest' or 'bilinear', got "
                             f"{fpn_upsample!r}")

        if self.p2:
            c2, c3, c4, c5 = chans
        else:
            c3, c4, c5 = chans
        self.lat5 = conv_norm_act(c5, out, 1)
        self.lat4 = conv_norm_act(c4, out, 1)
        self.lat3 = conv_norm_act(c3, out, 1)
        self.mrg4 = CSPLayer(2 * out, out, 1, shortcut=False)
        self.mrg3 = CSPLayer(2 * out, out, 1, shortcut=False)
        self.down3 = conv_norm_act(out, out, 3, 2)
        self.down4 = conv_norm_act(out, out, 3, 2)
        self.out4 = CSPLayer(2 * out, out, 1, shortcut=False)
        self.out5 = CSPLayer(2 * out, out, 1, shortcut=False)
        if self.p2:
            self.lat2 = conv_norm_act(c2, out, 1)
            self.mrg2 = CSPLayer(2 * out, out, 1, shortcut=False)
            self.down2 = conv_norm_act(out, out, 3, 2)
            self.out3 = CSPLayer(2 * out, out, 1, shortcut=False)

    def _upsample(self, x, size):
        """Upsample `x` to `size` with the configured mode (nearest or bilinear)."""
        if self.fpn_upsample == 'nearest':
            return F.interpolate(x, size=size, mode='nearest')
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, feats):
        """Top-down then bottom-up fusion over the pyramid levels; returns 3 or 4 outputs."""
        if self.p2:
            p2, p3, p4, p5 = feats
        else:
            p3, p4, p5 = feats
        p5 = self.lat5(p5)
        p4 = self.mrg4(torch.cat([self._upsample(p5, p4.shape[-2:]), self.lat4(p4)], 1))
        p3 = self.mrg3(torch.cat([self._upsample(p4, p3.shape[-2:]), self.lat3(p3)], 1))
        if self.p2:
            p2 = self.mrg2(torch.cat([self._upsample(p3, p2.shape[-2:]), self.lat2(p2)], 1))
            n3 = self.out3(torch.cat([self.down2(p2), p3], 1))
            n4 = self.out4(torch.cat([self.down3(n3), p4], 1))
            n5 = self.out5(torch.cat([self.down4(n4), p5], 1))
            return p2, n3, n4, n5
        n4 = self.out4(torch.cat([self.down3(p3), p4], 1))
        n5 = self.out5(torch.cat([self.down4(n4), p5], 1))
        return p3, n4, n5


class Head(nn.Module):
    """Decoupled head, objectness + ltrb. No classification branch (single class).

    DECOUPLED, not shared, and that is a measured choice: the second tower costs ~1.15x the whole
    network against 1.01x for a shared one, which is a rounding error on a real run's wall clock
    (the forward is <1% of it). Compute is not the constraint here; accuracy is.
    """

    def __init__(self, cin=96, n_levels=3, shared_head=True):
        """Build the head: per-level stems, reg (and, unless `shared_head`, obj) towers.

        `shared_head=True` runs obj AND reg through the SAME `reg_convs` tower; `False` builds a
        SEPARATE `obj_convs` tower (YOLOX's own decoupled head, the shipped recipe). The
        objectness bias is initialised to YOLOX's rare-positive prior (-4.595).
        """
        super().__init__()
        self.shared_head = bool(shared_head)
        self.stems = nn.ModuleList([conv_norm_act(cin, cin, 1) for _ in range(n_levels)])
        self.reg_convs = nn.ModuleList(
            [nn.Sequential(conv_norm_act(cin, cin, 3), conv_norm_act(cin, cin, 3))
             for _ in range(n_levels)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(cin, 4, 1) for _ in range(n_levels)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(cin, 1, 1) for _ in range(n_levels)])
        for m in self.obj_pred:
            nn.init.constant_(m.bias, -4.595)
        if not self.shared_head:
            self.obj_convs = nn.ModuleList(
                [nn.Sequential(conv_norm_act(cin, cin, 3), conv_norm_act(cin, cin, 3))
                 for _ in range(n_levels)])

    def forward(self, feats):
        """Per level: stem -> towers -> (obj, reg) predictions; a list of 2-tuples."""
        outs = []
        for i, f in enumerate(feats):
            stem = self.stems[i](f)
            x = self.reg_convs[i](stem)
            obj_x = x if self.shared_head else self.obj_convs[i](stem)
            outs.append((self.obj_pred[i](obj_x), self.reg_pred[i](x)))
        return outs


class YOLOXNano(nn.Module):
    """The box predictor: `version` backbone + PAFPN + decoupled head.

    `version` is one of `ARCHITECTURES`: 'convnext-t' (ImageNet ConvNeXt-T trunk, the shipped
    default; its weights load in training via `backbone.load_weights()`, not here) or 'hybrid'
    (CNN stem + transformer blocks, from scratch). Both feed a 256-wide neck. `p2` adds a stride-4
    FPN level and sets `self.STRIDES = (4, 8, 16, 32)`. `shared_head` and `fpn_upsample` are the
    head/neck switches (`Head`/`PAFPN`).
    """
    STRIDES = (8, 16, 32)

    def __init__(self, version='convnext-t', p2=False, shared_head=True, fpn_upsample='nearest'):
        """Build the box predictor for `version` (see the class docstring for arguments)."""
        super().__init__()
        self.version = str(version)
        self.p2 = bool(p2)
        if self.version in PRETRAINED_TRUNKS:
            from .pretrained_backbones import ConvNeXtBackbone
            self.backbone = ConvNeXtBackbone(p2=self.p2)
        elif self.version == 'hybrid':
            from .vit_backbone import HybridBackbone
            self.backbone = HybridBackbone(p2=self.p2)
        else:
            raise ValueError(f'yolox version {version!r}: must be one of {ARCHITECTURES}')
        self.neck = PAFPN(chans=self.backbone.out_channels, out=NECK_WIDTH, p2=self.p2,
                          fpn_upsample=fpn_upsample)
        self.head = Head(NECK_WIDTH, n_levels=4 if self.p2 else 3, shared_head=shared_head)
        if self.p2:
            self.STRIDES = (4, 8, 16, 32)

    def forward(self, x):
        """x: (B,3,H,W) normalized to [0,1] -> (obj_logits (B,A), boxes (B,A,4)).

        Boxes are xyxy INPUT-image pixels; A is the total anchor count across the levels. Boxes
        decode from ltrb distances: exp keeps them strictly positive (YOLOX's own choice), scaled
        by stride, around each cell centre.
        """
        outs = self.head(self.neck(self.backbone(x)))
        obj_all, box_all = [], []
        for (obj, reg), stride in zip(outs, self.STRIDES):
            B, _, h, w = obj.shape
            gy, gx = torch.meshgrid(torch.arange(h, device=x.device),
                                    torch.arange(w, device=x.device), indexing='ij')
            cx = (gx.reshape(-1) + 0.5) * stride
            cy = (gy.reshape(-1) + 0.5) * stride
            ltrb = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
            ltrb = torch.exp(ltrb.clamp(-6, 6)) * stride
            boxes = torch.stack([cx[None] - ltrb[..., 0], cy[None] - ltrb[..., 1],
                                 cx[None] + ltrb[..., 2], cy[None] + ltrb[..., 3]], dim=-1)
            obj_all.append(obj.reshape(B, -1))
            box_all.append(boxes)
        return torch.cat(obj_all, 1), torch.cat(box_all, 1)

    def anchor_points(self, h, w, device):
        """(A, 3) of (cx, cy, stride) matching forward()'s flattening order."""
        pts = []
        for stride in self.STRIDES:
            fh, fw = (h + stride - 1) // stride, (w + stride - 1) // stride
            gy, gx = torch.meshgrid(torch.arange(fh, device=device),
                                    torch.arange(fw, device=device), indexing='ij')
            cx = (gx.reshape(-1) + 0.5) * stride
            cy = (gy.reshape(-1) + 0.5) * stride
            pts.append(torch.stack([cx, cy, torch.full_like(cx, stride)], -1))
        return torch.cat(pts, 0)
