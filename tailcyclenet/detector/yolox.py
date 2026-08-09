"""Compact YOLOX-style box predictor.

One class, so YOLOX simplifies hard:
  * CSPDarknet-nano backbone (depthwise-separable) + PAFPN neck, strides 8/16/32
  * decoupled anchor-free head with the CLASSIFICATION branch dropped -- a single class
    means objectness alone carries all the information
  * centre-prior assignment instead of SimOTA (see assign.py)
  * BCE(objectness) + GIoU(box)

~1M params. The regression target is the SAME crop box the pose pipeline uses
(`tailcyclenet.crop.crop_box_for_points`), so the detector reproduces the crop the pose model
was trained on rather than some other box. `tests/test_crop.py` is what keeps that true.

Lifted from posetail-pose unchanged: it is a clean YOLOX-Nano with no exploration in it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin, cout, k=3, s=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True))


def dw_conv(cin, cout, k=3, s=1):
    """Depthwise-separable conv (the YOLOX-Nano building block)."""
    return nn.Sequential(conv_bn_act(cin, cin, k, s, groups=cin),
                         conv_bn_act(cin, cout, 1, 1))


class Bottleneck(nn.Module):
    def __init__(self, cin, cout, shortcut=True):
        super().__init__()
        hidden = cout // 2
        self.conv1 = conv_bn_act(cin, hidden, 1)
        self.conv2 = dw_conv(hidden, cout, 3)
        self.add = shortcut and cin == cout

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n=1, shortcut=True):
        super().__init__()
        hidden = cout // 2
        self.conv1 = conv_bn_act(cin, hidden, 1)
        self.conv2 = conv_bn_act(cin, hidden, 1)
        self.conv3 = conv_bn_act(2 * hidden, cout, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat([self.m(self.conv1(x)), self.conv2(x)], dim=1))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, sizes=(5, 9, 13)):
        super().__init__()
        hidden = cin // 2
        self.conv1 = conv_bn_act(cin, hidden, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(s, 1, s // 2) for s in sizes])
        self.conv2 = conv_bn_act(hidden * (len(sizes) + 1), cout, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.pools], dim=1))


class CSPDarknetNano(nn.Module):
    """Strides 8/16/32 feature maps."""

    def __init__(self, w=(24, 48, 96, 192)):
        super().__init__()
        c1, c2, c3, c4 = w
        self.stem = conv_bn_act(3, c1, 3, 2)                 # /2
        self.dark2 = nn.Sequential(dw_conv(c1, c2, 3, 2), CSPLayer(c2, c2, 1))     # /4
        self.dark3 = nn.Sequential(dw_conv(c2, c3, 3, 2), CSPLayer(c3, c3, 3))     # /8
        self.dark4 = nn.Sequential(dw_conv(c3, c4, 3, 2), CSPLayer(c4, c4, 3))     # /16
        self.dark5 = nn.Sequential(dw_conv(c4, c4, 3, 2), SPPBottleneck(c4, c4),
                                   CSPLayer(c4, c4, 1, shortcut=False))            # /32

    def forward(self, x):
        x = self.dark2(self.stem(x))
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5


class PAFPN(nn.Module):
    def __init__(self, chans=(96, 192, 192), out=96):
        super().__init__()
        c3, c4, c5 = chans
        self.lat5 = conv_bn_act(c5, out, 1)
        self.lat4 = conv_bn_act(c4, out, 1)
        self.lat3 = conv_bn_act(c3, out, 1)
        self.mrg4 = CSPLayer(2 * out, out, 1, shortcut=False)
        self.mrg3 = CSPLayer(2 * out, out, 1, shortcut=False)
        self.down3 = dw_conv(out, out, 3, 2)
        self.down4 = dw_conv(out, out, 3, 2)
        self.out4 = CSPLayer(2 * out, out, 1, shortcut=False)
        self.out5 = CSPLayer(2 * out, out, 1, shortcut=False)

    def forward(self, feats):
        p3, p4, p5 = feats
        p5 = self.lat5(p5)
        p4 = self.mrg4(torch.cat([F.interpolate(p5, size=p4.shape[-2:], mode='nearest'),
                                  self.lat4(p4)], 1))
        p3 = self.mrg3(torch.cat([F.interpolate(p4, size=p3.shape[-2:], mode='nearest'),
                                  self.lat3(p3)], 1))
        n4 = self.out4(torch.cat([self.down3(p3), p4], 1))
        n5 = self.out5(torch.cat([self.down4(n4), p5], 1))
        return p3, n4, n5


class Head(nn.Module):
    """Decoupled head, objectness + ltrb. No classification branch (single class)."""

    def __init__(self, cin=96, n_levels=3):
        super().__init__()
        self.stems = nn.ModuleList([conv_bn_act(cin, cin, 1) for _ in range(n_levels)])
        self.reg_convs = nn.ModuleList(
            [nn.Sequential(dw_conv(cin, cin, 3), dw_conv(cin, cin, 3)) for _ in range(n_levels)])
        self.reg_pred = nn.ModuleList([nn.Conv2d(cin, 4, 1) for _ in range(n_levels)])
        self.obj_pred = nn.ModuleList([nn.Conv2d(cin, 1, 1) for _ in range(n_levels)])
        for m in self.obj_pred:                      # rare-positive prior, as in YOLOX
            nn.init.constant_(m.bias, -4.595)

    def forward(self, feats):
        outs = []
        for i, f in enumerate(feats):
            x = self.reg_convs[i](self.stems[i](f))
            outs.append((self.obj_pred[i](x), self.reg_pred[i](x)))
        return outs


class YOLOXNano(nn.Module):
    STRIDES = (8, 16, 32)

    def __init__(self, width=96):
        super().__init__()
        self.backbone = CSPDarknetNano()
        self.neck = PAFPN(out=width)
        self.head = Head(width)

    def forward(self, x):
        """x: (B,3,H,W) normalized to [0,1].

        Returns obj_logits (B, A) and boxes (B, A, 4) in xyxy INPUT-image pixels, where A is
        the total number of anchor points across the three levels.
        """
        outs = self.head(self.neck(self.backbone(x)))
        obj_all, box_all = [], []
        for (obj, reg), stride in zip(outs, self.STRIDES):
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
