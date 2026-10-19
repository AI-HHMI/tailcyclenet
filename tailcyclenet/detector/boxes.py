"""Box arithmetic on (N,4) xyxy arrays: centres, sides, and the IoU family. One copy.

`centres`/`sides` take a torch tensor or a numpy array and return the same kind, so the torch
paths (NMS, association, tracking) and the numpy path (`link_rows`) share one formula.
"""
import numpy as np
import torch

# The centre-distance NMS gate, in box sides: ONE value for training's scoring and checkpoint
# selection and for deployment (`infer --det-nms-center-dist`, the eval scripts). Selection used
# 0.5 before the detector-recipe bump that unified it (detector B15).
NMS_CENTER_DIST = 0.3


def _stack(xs, like):
    """Stack `xs` on a new last axis with the library `like` belongs to."""
    return torch.stack(xs, -1) if torch.is_tensor(like) else np.stack(xs, -1)


def centres(boxes):
    """Box centres: (N,4) xyxy -> (N,2) (cx, cy)."""
    return _stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], boxes)


def sides(boxes):
    """Mean side of each box, (N,4) -> (N,): the length every identity distance is measured in."""
    return 0.5 * ((boxes[:, 2] - boxes[:, 0]) + (boxes[:, 3] - boxes[:, 1]))


def box_iou(a, b, eps=1e-7):
    """Pairwise IoU. a: (N,4), b: (M,4) -> (N,M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None] - inter + eps)


def paired_iou(a, b, eps=1e-7):
    """Elementwise IoU between MATCHED rows -- a[i] against b[i], not every pair.

    `box_iou` is the cross-product form (`(N,M)`), which is what NMS and evaluation want.
    Objectness supervision wants the OTHER shape: one IoU per POSITIVE anchor, against the one GT
    box `assign` gave it -- `box_iou(boxes[pos], gt).diagonal()` would compute N^2 pairs to keep
    N of them. Same maths as `box_iou`, no broadcasting.
    """
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    return inter / (area_a + area_b - inter + eps)


def box_center_dist(a, b, eps=1e-7):
    """Pairwise centre-distance ratio: Euclidean centre distance / mean box side. a:(N,4), b:(M,4)
    -> (N,M), scale-free (units of box side, not pixels).

    SLEAP's `min_centroid_distance` / `filters.py`'s own reasoning: IoU and OKS are both
    degenerate for point-like or near-concentric detections -- two boxes of very different size
    but (nearly) the same centre score a low IoU and survive NMS, while a duplicate detection on
    one animal is exactly that shape (report 42 SS3.6's near-concentric `fp_dup` measurement).
    Centre distance in units of the pair's own mean side catches that case directly and needs no
    scale calibration across roots -- the same normalisation `link_rows` already uses for its
    identity gate.
    """
    d = torch.linalg.norm(centres(a)[:, None] - centres(b)[None], dim=-1)
    sa, sb = sides(a), sides(b)
    side = (0.5 * (sa[:, None] + sb[None])).clamp_min(eps)
    return d / side
