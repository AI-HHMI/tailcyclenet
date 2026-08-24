"""Compact YOLOX-style box predictor, plus the canonical YOLOX tiers as an opt-in capacity switch.

One class, so YOLOX simplifies hard:
  * CSPDarknet backbone (depthwise-separable, at the default tier) + PAFPN neck, strides 8/16/32
  * decoupled anchor-free head with the CLASSIFICATION branch dropped -- a single class
    means objectness alone carries all the information
  * centre-prior assignment instead of SimOTA (see assign.py)
  * BCE(objectness) + GIoU(box)

**`version='trimmed'` (the default) is the repo's bespoke ~0.66M-param net** -- a 4-effective-width
backbone `(24,48,96,192)` with a plain-conv stem, unchanged from before this switch existed and
BYTE-IDENTICAL to it: every checkpoint trained under the old `YOLOXNano()` still loads.

**`version in {'nano','tiny','s','m','l','x'}` builds the CANONICAL YOLOX backbone** (Focus stem,
5-stage CSPDarknet at `base_channels = 64`) at that tier's official `(depth_mul, width_mul,
depthwise)` -- see `YOLOX_TIERS`. It exists to test whether the detector is CAPACITY-limited.

**NOT byte-identical to Megvii's release**, in two ways, both deliberate: (1) GroupNorm
throughout, not BatchNorm -- see `conv_norm_act`; (2) the neck unifies all three pyramid levels to
ONE output width rather than Megvii's per-level neck width with three separate head stems.

The regression target is the SAME crop box the pose pipeline uses
(`tailcyclenet.crop.crop_box_for_points`), so the detector reproduces the crop the pose model
was trained on rather than some other box, at every version. `tests/test_dataset.py` keeps that
true.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


CH_PER_GROUP = 8

# (depth_mul, width_mul, depthwise), Megvii's own values. Only `nano` is depthwise-separable --
# tiny/s/m/l/x are full-convolution. `base_channels = round8(64 * width_mul)` and
# `base_depth = max(1, round(3 * depth_mul))` are applied in `CSPDarknet` below.
YOLOX_TIERS = {
    'nano': (0.33, 0.25, True),
    'tiny': (0.33, 0.375, False),
    's':    (0.33, 0.50, False),
    'm':    (0.67, 0.75, False),
    'l':    (1.00, 1.00, False),
    'x':    (1.33, 1.25, False),
}

# A2: ViT backbone versions -> (hub repo, hub entrypoint, the [model].pretrained value that loads
# real weights for it). DINOv2's checkpoints are fully public; DINOv3's are GATED behind Meta's
# own license-request form -- `pretrained="dinov3"` still builds the architecture and attempts
# the download, but 403s from `dl.fbaipublicfiles.com/dinov3` without an approved request. Both
# hub repos expose the identical interface `vit_backbone.ViTBackbone` reads.
VIT_BACKBONES = {
    'vit_s14':    ('facebookresearch/dinov2', 'dinov2_vits14', 'dinov2'),
    'vit_b14':    ('facebookresearch/dinov2', 'dinov2_vitb14', 'dinov2'),
    'vit_s16_v3': ('facebookresearch/dinov3', 'dinov3_vits16', 'dinov3'),
    'vit_b16_v3': ('facebookresearch/dinov3', 'dinov3_vitb16', 'dinov3'),
}


def round8(v):
    """Round to the nearest multiple of 8 (floor 8) -- keeps `norm_groups` clean at any width."""
    return max(8, int(round(v / 8)) * 8)


def norm_groups(c, per_group=CH_PER_GROUP):
    """A group count that DIVIDES `c`, at about `per_group` channels each.

    The usual `G = 32` is unusable at `trimmed`'s widths, so the count is derived and walked down
    to a divisor; the walk-down makes this correct for any channel count a canonical tier produces
    too.
    """
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


def dw_conv(cin, cout, k=3, s=1):
    """Depthwise-separable conv (the YOLOX-Nano building block)."""
    return nn.Sequential(conv_norm_act(cin, cin, k, s, groups=cin),
                         conv_norm_act(cin, cout, 1, 1))


def conv3(cin, cout, k=3, s=1, depthwise=True):
    """The one 'kxk conv' building block used everywhere below, switched by `depthwise`.

    Megvii's tiers differ on exactly this: `nano` (and this repo's `trimmed`) use the
    depthwise-separable form; `tiny/s/m/l/x` use a single full convolution. One call site for
    both is what makes `depthwise` a single lever through the backbone, neck and head.
    """
    return dw_conv(cin, cout, k, s) if depthwise else conv_norm_act(cin, cout, k, s)


class Bottleneck(nn.Module):
    """`expansion` is Megvii's own parameter name: `CSPLayer` calls this with `cin == cout ==
    hidden` (its OWN CSP-split width), and this class used to halve that AGAIN unconditionally,
    so every bottleneck conv in the shipped net was HALF Megvii's own width -- 16 of 35 backbone
    tensors could not load from a COCO checkpoint at all. `0.5` reproduces that byte-identically;
    `1.0` is what Megvii's `CSPLayer` actually passes to its inner `Bottleneck`s (full width, no
    second halving).
    """
    def __init__(self, cin, cout, shortcut=True, depthwise=True, expansion=0.5):
        super().__init__()
        hidden = int(cout * expansion)
        self.conv1 = conv_norm_act(cin, hidden, 1)
        self.conv2 = conv3(hidden, cout, 3, depthwise=depthwise)
        self.add = shortcut and cin == cout

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class CSPNeXtBottleneck(nn.Module):
    """5x5 depthwise + SE channel attention, RTMDet's own bottleneck block.

    Same __init__ signature as `Bottleneck` so `CSPNeXtLayer` can substitute it without changing
    any other code. At `bottleneck_expansion=0.5` (this net's own default) the SE block's two 1x1
    convs push CSPNeXt noticeably past a plain-Bottleneck param estimate for the same width/depth
    -- confirmed against `Bottleneck`+`CSPLayer` at the same `width_mul`/`depth_mul` by
    `dev/scratch/prototype_cspnext.py`.
    """
    def __init__(self, cin, cout, shortcut=True, depthwise=True, expansion=0.5):
        super().__init__()
        hidden = int(cout * expansion)
        self.conv1 = conv_norm_act(cin, hidden, 3)                    # 3x3 standard
        # 5x5 depthwise (the CSPNeXt signature: larger kernel than YOLOX's 3x3)
        self.dw = conv_norm_act(hidden, hidden, 5, groups=hidden)     # 5x5 depthwise
        self.pw = conv_norm_act(hidden, cout, 1)                      # 1x1 pointwise
        # Squeeze-Excitation: global avg pool -> FC down -> SiLU -> FC up -> Sigmoid
        sq = max(1, cout // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cout, sq, 1), nn.SiLU(inplace=True),
            nn.Conv2d(sq, cout, 1), nn.Sigmoid())
        self.add = shortcut and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(self.conv1(x)))
        y = y * self.se(y)
        return x + y if self.add else y


class CSPNeXtLayer(nn.Module):
    """`CSPLayer` using `CSPNeXtBottleneck` instead of `Bottleneck`."""
    def __init__(self, cin, cout, n=1, shortcut=True, depthwise=True, bottleneck_expansion=0.5):
        super().__init__()
        hidden = cout // 2
        self.conv1 = conv_norm_act(cin, hidden, 1)
        self.conv2 = conv_norm_act(cin, hidden, 1)
        self.conv3 = conv_norm_act(2 * hidden, cout, 1)
        self.m = nn.Sequential(
            *[CSPNeXtBottleneck(hidden, hidden, shortcut, depthwise=depthwise,
                                expansion=bottleneck_expansion) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat([self.m(self.conv1(x)), self.conv2(x)], dim=1))


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n=1, shortcut=True, depthwise=True, bottleneck_expansion=0.5):
        super().__init__()
        hidden = cout // 2
        self.conv1 = conv_norm_act(cin, hidden, 1)
        self.conv2 = conv_norm_act(cin, hidden, 1)
        self.conv3 = conv_norm_act(2 * hidden, cout, 1)
        self.m = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut, depthwise=depthwise,
                        expansion=bottleneck_expansion) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat([self.m(self.conv1(x)), self.conv2(x)], dim=1))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, sizes=(5, 9, 13)):
        super().__init__()
        hidden = cin // 2
        self.conv1 = conv_norm_act(cin, hidden, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(s, 1, s // 2) for s in sizes])
        self.conv2 = conv_norm_act(hidden * (len(sizes) + 1), cout, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.pools], dim=1))


class CSPDarknetNano(nn.Module):
    """`trimmed`'s backbone. Strides 8/16/32 feature maps. Unchanged from before this switch.

    `p2`: `dark2`'s own output (stride 4) is already computed on every forward -- it was just
    never RETURNED. `False` (default) keeps the 3-tuple `(p3, p4, p5)` return exactly as before;
    `True` returns the 4-tuple `(p2, p3, p4, p5)` and widens `out_channels` to match.

    `in_channels` is the stem's own input width; `3` (default) is byte-identical to every
    checkpoint on record, and a temporal-input caller widens only this one conv -- everything
    downstream of the stem is unaffected.
    """

    def __init__(self, w=(24, 48, 96, 192), p2=False, in_channels=3):
        super().__init__()
        c1, c2, c3, c4 = w
        self.p2 = bool(p2)
        self.stem = conv_norm_act(int(in_channels), c1, 3, 2)  # /2
        self.dark2 = nn.Sequential(conv3(c1, c2, 3, 2), CSPLayer(c2, c2, 1))          # /4
        self.dark3 = nn.Sequential(conv3(c2, c3, 3, 2), CSPLayer(c3, c3, 3))          # /8
        self.dark4 = nn.Sequential(conv3(c3, c4, 3, 2), CSPLayer(c4, c4, 3))          # /16
        self.dark5 = nn.Sequential(conv3(c4, c4, 3, 2), SPPBottleneck(c4, c4),
                                   CSPLayer(c4, c4, 1, shortcut=False))               # /32
        # p3, p4, p5 output widths -- dark4 and dark5 share `c4` in this 4-effective-width net,
        # unlike the canonical 5-stage backbone below where all three differ.
        self.out_channels = (c2, c3, c4, c4) if self.p2 else (c3, c4, c4)

    def forward(self, x):
        p2 = self.dark2(self.stem(x))
        p3 = self.dark3(p2)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return (p2, p3, p4, p5) if self.p2 else (p3, p4, p5)


class Focus(nn.Module):
    """Space-to-depth stride-2 stem, YOLOX's classic first layer.

    Concatenates four 2x2-subsampled copies of the input on the channel axis -- a LOSSLESS
    stride-2 reduction, unlike a strided conv -- then one k x k conv. Only `CSPDarknet` (the
    canonical tiers) uses this; `trimmed`'s stem is a plain strided conv, kept as-is for
    checkpoint back-compat.
    """

    def __init__(self, cin, cout, k=3, depthwise=False):
        super().__init__()
        self.conv = conv3(cin * 4, cout, k, 1, depthwise=depthwise)

    def forward(self, x):
        tl = x[..., ::2, ::2]
        tr = x[..., ::2, 1::2]
        bl = x[..., 1::2, ::2]
        br = x[..., 1::2, 1::2]
        return self.conv(torch.cat([tl, bl, tr, br], dim=1))


class CSPDarknet(nn.Module):
    """The CANONICAL 5-stage YOLOX backbone: Focus stem, `base_channels x {1,2,4,8,16}`.

    Same contract as `CSPDarknetNano` (strides 8/16/32, returns p3/p4/p5), built at Megvii's
    official `(depth_mul, width_mul, depthwise)` for the six named tiers in `YOLOX_TIERS`. See the
    module docstring for the two deliberate deviations. `in_channels` widens the stem: `Focus`
    space-to-depth SPLITS whatever channel count `x` has, so this one argument is the whole of it
    (`3` default is byte-identical to every checkpoint on record).
    """

    def __init__(self, width_mul=1.0, depth_mul=1.0, depthwise=False, bottleneck_expansion=0.5,
                p2=False, in_channels=3):
        super().__init__()
        c = round8(64 * width_mul)
        d = max(1, round(3 * depth_mul))
        be = bottleneck_expansion
        self.p2 = bool(p2)
        self.stem = Focus(int(in_channels), c, 3, depthwise=depthwise)                    # /2
        self.dark2 = nn.Sequential(conv3(c, c * 2, 3, 2, depthwise=depthwise),
                                   CSPLayer(c * 2, c * 2, d, depthwise=depthwise,
                                           bottleneck_expansion=be))                      # /4
        self.dark3 = nn.Sequential(conv3(c * 2, c * 4, 3, 2, depthwise=depthwise),
                                   CSPLayer(c * 4, c * 4, d * 3, depthwise=depthwise,
                                           bottleneck_expansion=be))                      # /8
        self.dark4 = nn.Sequential(conv3(c * 4, c * 8, 3, 2, depthwise=depthwise),
                                   CSPLayer(c * 8, c * 8, d * 3, depthwise=depthwise,
                                           bottleneck_expansion=be))                      # /16
        self.dark5 = nn.Sequential(
            conv3(c * 8, c * 16, 3, 2, depthwise=depthwise),
            SPPBottleneck(c * 16, c * 16),
            CSPLayer(c * 16, c * 16, d, shortcut=False, depthwise=depthwise,
                    bottleneck_expansion=be))                                             # /32
        # p2, p3, p4, p5 -- FOUR distinct widths when `p2` (T4.3); the 3-tuple contract is
        # unchanged at the default, matching `CSPDarknetNano`'s own `p2` switch.
        self.out_channels = (c * 2, c * 4, c * 8, c * 16) if self.p2 else (c * 4, c * 8, c * 16)

    def forward(self, x):
        p2 = self.dark2(self.stem(x))
        p3 = self.dark3(p2)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return (p2, p3, p4, p5) if self.p2 else (p3, p4, p5)


class CSPNeXt(nn.Module):
    """CSPNeXt backbone (RTMDet-style). Same stage structure as `CSPDarknet` but with
    `CSPNeXtBottleneck` (5x5 depthwise + SE) instead of the standard `Bottleneck`.

    Uses the `Focus` stem. Width/depth controlled by multipliers, same as `CSPDarknet`. At
    `width_mul=0.5, depth_mul=0.33` (this net's own `cspnext_s` tier) the SE blocks push this
    past a same-shape `CSPDarknet`'s param count -- confirmed by
    `dev/scratch/prototype_cspnext.py`: 3.65M at `bottleneck_expansion=0.5` (this net's default),
    not the plan doc's original ~2.5M estimate.
    """
    def __init__(self, width_mul=0.5, depth_mul=0.33, bottleneck_expansion=1.0,
                p2=False, in_channels=3):
        super().__init__()
        c = round8(64 * width_mul)       # base_channels, e.g. 32 at width_mul=0.5
        d = max(1, round(3 * depth_mul)) # base_depth, e.g. 1 at depth_mul=0.33
        be = bottleneck_expansion
        self.p2 = bool(p2)
        # Same stem as CSPDarknet canonical tiers
        self.stem = Focus(int(in_channels), c, 3, depthwise=False)            # /2
        self.dark2 = nn.Sequential(
            conv_norm_act(c, c * 2, 3, 2),
            CSPNeXtLayer(c * 2, c * 2, d, bottleneck_expansion=be))           # /4
        self.dark3 = nn.Sequential(
            conv_norm_act(c * 2, c * 4, 3, 2),
            CSPNeXtLayer(c * 4, c * 4, d * 3, bottleneck_expansion=be))       # /8
        self.dark4 = nn.Sequential(
            conv_norm_act(c * 4, c * 8, 3, 2),
            CSPNeXtLayer(c * 8, c * 8, d * 3, bottleneck_expansion=be))       # /16
        self.dark5 = nn.Sequential(
            conv_norm_act(c * 8, c * 16, 3, 2),
            SPPBottleneck(c * 16, c * 16),
            CSPNeXtLayer(c * 16, c * 16, d, shortcut=False,
                        bottleneck_expansion=be))                            # /32
        self.out_channels = ((c * 2, c * 4, c * 8, c * 16) if self.p2
                             else (c * 4, c * 8, c * 16))

    def forward(self, x):
        p2 = self.dark2(self.stem(x))
        p3 = self.dark3(p2)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return (p2, p3, p4, p5) if self.p2 else (p3, p4, p5)


class PAFPN(nn.Module):
    """`p2`: a 4th, finer level (stride 4). `False` (default) is the original 3-level PAFPN,
    byte-identical -- the `p2`-only modules (`lat2`/`mrg2`/`down2`/`out3`) are not even
    CONSTRUCTED, the same "not built and ignored" contract the keypoint branch uses.

    The extra level slots into the EXISTING top-down/bottom-up pattern: p2 becomes the new finest
    level (top-down only, like p3 was before), and p3 -- no longer the finest -- gains the
    bottom-up refinement stage it never needed before (`out3`, mirroring `out4`/`out5`).
    `down3`'s WEIGHTS are unchanged from the 3-level case; only what feeds it changes.
    """

    def __init__(self, chans=(96, 192, 192), out=96, depthwise=True, p2=False):
        super().__init__()
        self.p2 = bool(p2)
        if self.p2:
            c2, c3, c4, c5 = chans
        else:
            c3, c4, c5 = chans
        self.lat5 = conv_norm_act(c5, out, 1)
        self.lat4 = conv_norm_act(c4, out, 1)
        self.lat3 = conv_norm_act(c3, out, 1)
        self.mrg4 = CSPLayer(2 * out, out, 1, shortcut=False, depthwise=depthwise)
        self.mrg3 = CSPLayer(2 * out, out, 1, shortcut=False, depthwise=depthwise)
        self.down3 = conv3(out, out, 3, 2, depthwise=depthwise)
        self.down4 = conv3(out, out, 3, 2, depthwise=depthwise)
        self.out4 = CSPLayer(2 * out, out, 1, shortcut=False, depthwise=depthwise)
        self.out5 = CSPLayer(2 * out, out, 1, shortcut=False, depthwise=depthwise)
        if self.p2:
            self.lat2 = conv_norm_act(c2, out, 1)
            self.mrg2 = CSPLayer(2 * out, out, 1, shortcut=False, depthwise=depthwise)
            self.down2 = conv3(out, out, 3, 2, depthwise=depthwise)
            self.out3 = CSPLayer(2 * out, out, 1, shortcut=False, depthwise=depthwise)

    def forward(self, feats):
        if self.p2:
            p2, p3, p4, p5 = feats
        else:
            p3, p4, p5 = feats
        p5 = self.lat5(p5)
        p4 = self.mrg4(torch.cat([F.interpolate(p5, size=p4.shape[-2:], mode='nearest'),
                                  self.lat4(p4)], 1))
        p3 = self.mrg3(torch.cat([F.interpolate(p4, size=p3.shape[-2:], mode='nearest'),
                                  self.lat3(p3)], 1))
        if self.p2:
            p2 = self.mrg2(torch.cat([F.interpolate(p3, size=p2.shape[-2:], mode='nearest'),
                                      self.lat2(p2)], 1))
            n3 = self.out3(torch.cat([self.down2(p2), p3], 1))
            n4 = self.out4(torch.cat([self.down3(n3), p4], 1))
            n5 = self.out5(torch.cat([self.down4(n4), p5], 1))
            return p2, n3, n4, n5
        n4 = self.out4(torch.cat([self.down3(p3), p4], 1))
        n5 = self.out5(torch.cat([self.down4(n4), p5], 1))
        return p3, n4, n5


class Head(nn.Module):
    """Decoupled head, objectness + ltrb. No classification branch (single class).

    `n_keypoints > 0` adds a KEYPOINT BRANCH: a second tower off the same stem, and a 1x1 emitting
    `(dx, dy, score)` per keypoint. At the default 0 nothing is constructed -- not built and
    ignored -- so an existing checkpoint's `state_dict` is unchanged.

    DECOUPLED, not shared, and that is a measured choice: the second tower costs ~1.15x the whole
    network against 1.01x for a shared one, which is a rounding error on a real run's wall clock
    (the forward is <1% of it). Compute is not the constraint here; accuracy is.

    `depthwise` matches the tier's own choice: `trimmed`/`nano` use depthwise-separable towers,
    `tiny` and up use full convolutions.
    """

    def __init__(self, cin=96, n_levels=3, n_keypoints=0, depthwise=True):
        super().__init__()
        self.n_keypoints = int(n_keypoints)
        # THE IDENTITY BRANCH -- a per-anchor softmax over a CLOSED animal set (what rat-city is,
        # 12 fixed rats) -- was built and deleted; see the module docstring's refutation.
        self.stems = nn.ModuleList([conv_norm_act(cin, cin, 1) for _ in range(n_levels)])
        self.reg_convs = nn.ModuleList(
            [nn.Sequential(conv3(cin, cin, 3, depthwise=depthwise),
                           conv3(cin, cin, 3, depthwise=depthwise)) for _ in range(n_levels)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(cin, 4, 1) for _ in range(n_levels)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(cin, 1, 1) for _ in range(n_levels)])
        for m in self.obj_pred:                      # rare-positive prior, as in YOLOX
            nn.init.constant_(m.bias, -4.595)
        if self.n_keypoints:
            self.kpt_convs = nn.ModuleList(
                [nn.Sequential(conv3(cin, cin, 3, depthwise=depthwise),
                               conv3(cin, cin, 3, depthwise=depthwise))
                 for _ in range(n_levels)])
            self.kpt_pred = nn.ModuleList(
                [nn.Conv2d(cin, 3 * self.n_keypoints, 1) for _ in range(n_levels)])

    def forward(self, feats):
        outs = []
        for i, f in enumerate(feats):
            stem = self.stems[i](f)
            x = self.reg_convs[i](stem)
            kpt = self.kpt_pred[i](self.kpt_convs[i](stem)) if self.n_keypoints else None
            outs.append((self.obj_pred[i](x), self.reg_pred[i](x), kpt))
        return outs


class YOLOXNano(nn.Module):
    """The box predictor. `version='trimmed'` (default) is byte-identical to the pre-switch net.

    `version` in `{'nano','tiny','s','m','l','x'}` instead builds `CSPDarknet` at that tier's
    official `(depth_mul, width_mul, depthwise)` (`YOLOX_TIERS`), with this file's GroupNorm,
    single-class crop-rule head and optional keypoint branch unchanged. `width` only applies to
    `version='trimmed'`; for a canonical tier the neck/head width is DERIVED from `width_mul`, and
    passing a non-default `width` alongside a canonical `version` raises.

    `bottleneck_expansion` applies only to a CANONICAL tier's BACKBONE (`0.5` default is
    byte-identical to every checkpoint on record; `1.0` is the shape a Megvii COCO backbone
    actually loads into). Passing a non-default value alongside `version='trimmed'` raises.

    `p2` adds a stride-4 FPN level, on top of EITHER backbone: `False` (default) is byte-identical
    to every checkpoint on record; `True` widens backbone, neck and head and sets
    `self.STRIDES = (4, 8, 16, 32)` -- `forward`/`anchor_points` already loop over `self.STRIDES`
    generically.

    `in_channels` is the ARCHITECTURE HALF ONLY: `3` (default) is byte-identical to every
    checkpoint on record; a temporal-input caller passes a wider value and the stem absorbs it.
    NOT YET WIRED: `BoxDataset`'s loader still emits 3-channel RGB items and `detect_raw` still
    decodes single frames, so passing `in_channels != 3` today builds a model that trains on
    garbage unless the CALLER also supplies genuinely wider `x`. This constructor argument is
    scaffolding for that follow-on, not a complete feature.
    """
    STRIDES = (8, 16, 32)

    def __init__(self, width=96, n_keypoints=0, version='trimmed', bottleneck_expansion=0.5,
                p2=False, in_channels=3, head_depthwise=None, pretrained=''):
        super().__init__()
        self.n_keypoints = int(n_keypoints)
        self.version = str(version)
        self.bottleneck_expansion = float(bottleneck_expansion)
        self.p2 = bool(p2)
        self.in_channels = int(in_channels)
        self.head_depthwise = head_depthwise
        # A2: which pretrained weights (if any) a ViT backbone loads. '' (default) trains from
        # scratch; 'dinov2' loads the DINOv2 hub checkpoint. Only meaningful for a `vit_*`
        # version -- `config.py` refuses the combination otherwise.
        self.pretrained_source = str(pretrained) if isinstance(pretrained, str) else ''
        if self.version == 'trimmed':
            if self.bottleneck_expansion != 0.5:
                raise ValueError(
                    f"bottleneck_expansion={bottleneck_expansion} was passed alongside "
                    "version='trimmed', but trimmed's CSPDarknetNano does not take it -- it stays "
                    "at 0.5 permanently (it is the repo's own deliberately-narrow net, not a bug "
                    "to fix). Use a canonical tier for a COCO-compatible backbone.")
            self.backbone = CSPDarknetNano(p2=self.p2, in_channels=self.in_channels)
            neck_out, depthwise = width, True
        elif self.version in VIT_BACKBONES:
            # A2: DINOv2/DINOv3 ViT backbone + Simple Feature Pyramid. `in_channels != 3` is
            # refused -- a ViT patch embedding is built for RGB and has no wider-stem path.
            if self.in_channels != 3:
                raise ValueError(f"in_channels={self.in_channels} was passed alongside "
                                 f"version={version!r}, but the ViT backbone's patch embedding "
                                 "only accepts 3-channel RGB.")
            from .vit_backbone import ViTBackbone
            hub_repo, model_name, pretrained_key = VIT_BACKBONES[self.version]
            use_pretrained = (self.pretrained_source == pretrained_key)
            self.backbone = ViTBackbone(model_name, p2=self.p2, pretrained=use_pretrained,
                                        hub_repo=hub_repo)
            neck_out = round8(256 * 0.5)   # 128 -- match the 's' tier's neck width
            depthwise = False               # ViT arms use full-conv neck/head
        elif self.version == 'hybrid':
            # A3: CNN stem (strides 2/4/8) + transformer blocks (strides 16/32), from scratch.
            from .vit_backbone import HybridBackbone
            self.backbone = HybridBackbone(p2=self.p2, in_channels=self.in_channels)
            neck_out = round8(256)    # 256, matching the c*4 = 256 at stride-8
            depthwise = False          # full-conv neck/head
        elif self.version == 'cspnext_s':
            # A4: RTMDet-style backbone (5x5 depthwise + SE bottleneck) on the existing
            # CSPDarknet stage structure.
            self.backbone = CSPNeXt(width_mul=0.5, depth_mul=0.33,
                                    bottleneck_expansion=self.bottleneck_expansion,
                                    p2=self.p2, in_channels=self.in_channels)
            neck_out = round8(256 * 0.5)   # 128
            depthwise = False               # CSPNeXt uses full-conv in its towers
        else:
            if self.version not in YOLOX_TIERS:
                raise ValueError(f"yolox version {version!r}: must be 'trimmed' or one of "
                                 f"{sorted(YOLOX_TIERS)}")
            if width != 96:
                raise ValueError(f"width={width} was passed alongside version={version!r}, but "
                                 "width only applies to version='trimmed' -- a canonical tier "
                                 "derives its neck/head width from width_mul.")
            depth_mul, width_mul, depthwise = YOLOX_TIERS[self.version]
            self.backbone = CSPDarknet(width_mul, depth_mul, depthwise=depthwise,
                                       bottleneck_expansion=self.bottleneck_expansion,
                                       p2=self.p2, in_channels=self.in_channels)
            neck_out = round8(256 * width_mul)
        self.neck = PAFPN(chans=self.backbone.out_channels, out=neck_out, depthwise=depthwise,
                          p2=self.p2)
        n_levels = 4 if self.p2 else 3
        head_dw = depthwise if head_depthwise is None else bool(head_depthwise)
        self.head = Head(neck_out, n_levels=n_levels, n_keypoints=self.n_keypoints,
                         depthwise=head_dw)
        if self.p2:
            self.STRIDES = (4, 8, 16, 32)

    def forward(self, x):
        """x: (B,`self.in_channels`,H,W) normalized to [0,1] -- 3 unless a temporal-input caller
        widened the stem.

        Returns obj_logits (B, A) and boxes (B, A, 4) in xyxy INPUT-image pixels, where A is
        the total number of anchor points across the three levels. With `n_keypoints > 0` it
        returns a third tensor, keypoints (B, A, K, 3) -- (x, y, score_logit), x/y also in input
        pixels -- and `None` otherwise, so existing two-value callers must be updated but their
        behaviour is unchanged.
        """
        outs = self.head(self.neck(self.backbone(x)))
        obj_all, box_all, kpt_all = [], [], []
        for (obj, reg, kpt), stride in zip(outs, self.STRIDES):
            B, _, h, w = obj.shape
            gy, gx = torch.meshgrid(torch.arange(h, device=x.device),
                                    torch.arange(w, device=x.device), indexing='ij')
            # cell centres in input pixels
            cx = (gx.reshape(-1) + 0.5) * stride
            cy = (gy.reshape(-1) + 0.5) * stride
            # anchor-free ltrb distances, positive via exp (YOLOX uses exp on wh)
            ltrb = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
            ltrb = torch.exp(ltrb.clamp(-6, 6)) * stride
            boxes = torch.stack([cx[None] - ltrb[..., 0], cy[None] - ltrb[..., 1],
                                 cx[None] + ltrb[..., 2], cy[None] + ltrb[..., 3]], dim=-1)
            obj_all.append(obj.reshape(B, -1))
            box_all.append(boxes)
            if kpt is not None:
                k = kpt.permute(0, 2, 3, 1).reshape(B, -1, self.n_keypoints, 3)
                # NO `exp` HERE. ltrb distances are positive so the box decode exponentiates;
                # keypoint offsets are SIGNED, and exponentiating them would fold every keypoint
                # to one side of its anchor, silently.
                #
                # BOUNDED AT 1.25 BOX HALF-WIDTHS via tanh, borrowed from RTMO's bin range: a
                # keypoint physically cannot land outside 1.25x its own box, which directly
                # attacks the identity-relevant failure -- a keypoint flying onto the NEIGHBOURING
                # animal. `.detach()` on the box so the keypoint loss cannot perturb the box
                # branch.
                half = torch.stack([(ltrb[..., 0] + ltrb[..., 2]) / 2,
                                    (ltrb[..., 1] + ltrb[..., 3]) / 2], -1).detach()
                ctr = torch.stack([cx, cy], -1)[None]                    # (1,A,2)
                xy = ctr[:, :, None] + 1.25 * half[:, :, None] * torch.tanh(k[..., :2])
                kpt_all.append(torch.cat([xy, k[..., 2:]], -1))
        k = torch.cat(kpt_all, 1) if kpt_all else None
        return torch.cat(obj_all, 1), torch.cat(box_all, 1), k

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
