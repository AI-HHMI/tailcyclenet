"""CondInst-style instance-mask head on a frozen pose-pipeline detector, deploy path only.

The head reads the frozen YOLOX neck (stride-4 ``p2`` + stride-8 ``n3``), predicts an 8-channel mask
feature map at the head's output stride, and a per-anchor controller of 169 dynamic parameters (three
1x1 convs, CondInst) that turns that map into one soft mask per detection. The identity pipeline uses
the masks only for crop geometry (`identity.crops.mask_crops`): moment centroid, body axis, skew sign.

Checkpoint layout (``maskhead.pth``): ``{'state': state_dict, 'arch': MaskHead kwargs}``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4
# dynamic parameters per instance: conv 10->8, conv 8->8, conv 8->1 (weights + biases)
NPARAM = (10 * 8 + 8) + (8 * 8 + 8) + (8 + 1)


def _cna(cin, cout, k=3):
    """Conv + GroupNorm + SiLU (largest group count <= cout // 8 that divides cout)."""
    g = max(1, cout // 8)
    while cout % g:
        g -= 1
    return nn.Sequential(nn.Conv2d(cin, cout, k, 1, k // 2, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True))


class MaskHead(nn.Module):
    """Mask-feature branch (n3 fused into p2, output stride 4 or 2, 8 ch) + per-anchor controller."""

    def __init__(self, cin=256, mid=128, mch=8, ctrl=256, out_stride=4):
        """`cin` neck width, `mid` branch width, `mch` mask channels; `out_stride` 4 (three convs) or 2
        (two convs at s4, bilinear x2, two convs at s2)."""
        super().__init__()
        self.out_stride = int(out_stride)
        self.l8 = _cna(cin, mid, 1)
        self.l4 = _cna(cin, mid, 1)
        if self.out_stride == 4:
            self.tower = nn.Sequential(_cna(mid, mid), _cna(mid, mid), _cna(mid, mid), nn.Conv2d(mid, mch, 1))
        else:
            self.tower = nn.Sequential(_cna(mid, mid), _cna(mid, mid))
            self.up = nn.Sequential(_cna(mid, 64), _cna(64, 64), nn.Conv2d(64, mch, 1))
        self.ctrl = nn.Sequential(_cna(cin, ctrl), _cna(ctrl, ctrl), nn.Conv2d(ctrl, NPARAM, 3, 1, 1))

    def mask_feats(self, feats):
        """Neck features -> (B, 8, h, w) mask features at `out_stride`."""
        p2, n3 = feats[0], feats[1]
        x = self.l4(p2) + F.interpolate(self.l8(n3), size=p2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.tower(x)
        if self.out_stride == 4:
            return x
        return self.up(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

    def params(self, feats):
        """Neck features -> (B, A, 169) dynamic parameters, anchors in the detector's own order."""
        return torch.cat([self.ctrl(f).flatten(2).transpose(1, 2) for f in feats], 1)

    @staticmethod
    def dynamic(mf, prm, centre, size, stride=STRIDE):
        """mf (8,h,w) one image; prm (N,169); centre (N,2) and size (N,) in input px -> mask logits (N,h,w).

        The two extra input channels are the grid's offset from each anchor centre in units of the
        detection's half long side (floored at 8 px)."""
        C, h, w = mf.shape
        N = prm.shape[0]
        ys = (torch.arange(h, device=mf.device, dtype=mf.dtype) + 0.5) * stride - 0.5
        xs = (torch.arange(w, device=mf.device, dtype=mf.dtype) + 0.5) * stride - 0.5
        s = size.clamp_min(8.0)[:, None]
        rx = ((xs[None, :] - centre[:, 0, None]) / s)[:, None, None, :].expand(N, 1, h, w)
        ry = ((ys[None, :] - centre[:, 1, None]) / s)[:, None, :, None].expand(N, 1, h, w)
        x = torch.cat([mf[None].expand(N, C, h, w), rx, ry], 1).reshape(1, N * 10, h, w)
        sizes = (80, 8, 64, 8, 8, 1)
        w1, b1, w2, b2, w3, b3 = torch.split(prm, sizes, dim=1)
        x = F.relu(F.conv2d(x, w1.reshape(N * 8, 10, 1, 1), b1.reshape(N * 8), groups=N))
        x = F.relu(F.conv2d(x, w2.reshape(N * 8, 8, 1, 1), b2.reshape(N * 8), groups=N))
        x = F.conv2d(x, w3.reshape(N, 8, 1, 1), b3.reshape(N), groups=N)
        return x.reshape(N, h, w)


def load_maskhead(path, device='cpu'):
    """A ``maskhead.pth`` -> MaskHead in eval mode on `device`."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    mh = MaskHead(**ck.get('arch', {}))
    mh.load_state_dict(ck['state'])
    return mh.to(device).eval()


def _det_forward(model, x):
    """Frozen detector forward that also returns the neck features. x (B,3,H,W) in [0,1].

    Returns (feats, objectness logits (B,A), boxes xyxy input px (B,A,4), anchor centres (A,2))."""
    feats = model.neck(model.backbone(x))
    obj_all, box_all, ctr = [], [], []
    for (obj, reg), stride in zip(model.head(feats), model.STRIDES):
        B, _, h, w = obj.shape
        gy, gx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
        cx = (gx.reshape(-1) + 0.5) * stride
        cy = (gy.reshape(-1) + 0.5) * stride
        ltrb = torch.exp(reg.permute(0, 2, 3, 1).reshape(B, -1, 4).clamp(-6, 6)) * stride
        boxes = torch.stack([cx[None] - ltrb[..., 0], cy[None] - ltrb[..., 1], cx[None] + ltrb[..., 2], cy[None] + ltrb[..., 3]], -1)
        obj_all.append(obj.reshape(B, -1))
        box_all.append(boxes)
        ctr.append(torch.stack([cx, cy], -1))
    return feats, torch.cat(obj_all, 1), torch.cat(box_all, 1), torch.cat(ctr, 0)


@torch.no_grad()
def detect_masks(model, mh, imgs, top_k, score_thresh=0.01, device='cuda'):
    """Prepared detector inputs (list of HxWx3 uint8) -> per image (boxes (n,4) input px, scores (n,),
    soft masks (n,h,w) float32 on the head's grid). Selection is the shipped decode (`detector.decode`)."""
    from ..detector.decode import decode
    x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2))).to(device).float() / 255.0
    feats, obj, boxes, ctr = _det_forward(model, x)
    mf = mh.mask_feats(feats)
    prm = mh.params(feats)
    out = []
    for j in range(x.shape[0]):
        b, s, tr = decode(obj[j], boxes[j], top_k=top_k, score_thresh=score_thresh, return_trace=True)
        ix = tr['nms_index'][:top_k]
        if ix.numel():
            size = 0.5 * torch.maximum(boxes[j, ix, 2] - boxes[j, ix, 0], boxes[j, ix, 3] - boxes[j, ix, 1])
            m = MaskHead.dynamic(mf[j], prm[j, ix], ctr[ix], size, stride=mh.out_stride).sigmoid()
            out.append((b.cpu().numpy(), s.cpu().numpy(), m.float().cpu().numpy()))
        else:
            out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0,) + mf.shape[-2:], np.float32)))
    return out
