"""Assignment, loss and decoding for the box predictor.

Centre-prior assignment rather than SimOTA: an anchor is a positive for a ground-truth box when
its cell centre falls inside that box AND within a fixed radius of the box centre, at any level.
With one class and a handful of instances per frame, SimOTA's dynamic-k machinery buys nothing
and adds a second thing that can be wrong.

The regression target is the crop rule's box, so what the detector learns is "reproduce the crop
the pose model was trained on" -- not "find an animal". Those are different objectives and only
the first one keeps the downstream accuracy.
"""
import torch
import torch.nn.functional as F

CENTER_RADIUS = 2.5      # in cells; YOLOX's own value


def box_iou(a, b, eps=1e-7):
    """Pairwise IoU. a: (N,4), b: (M,4) -> (N,M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None] - inter + eps)


def giou_loss(pred, target, eps=1e-7):
    """1 - GIoU, elementwise over matched pairs. pred/target: (N,4) xyxy."""
    ap = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
    at = (target[:, 2] - target[:, 0]).clamp(0) * (target[:, 3] - target[:, 1]).clamp(0)
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = ap + at - inter + eps
    iou = inter / union
    clt = torch.min(pred[:, :2], target[:, :2])
    crb = torch.max(pred[:, 2:], target[:, 2:])
    cwh = (crb - clt).clamp(min=0)
    carea = cwh[:, 0] * cwh[:, 1] + eps
    return 1.0 - (iou - (carea - union) / carea)


def assign(anchors, gt_boxes):
    """Positive anchors for each ground-truth box.

    Args:
        anchors: (A,3) of (cx, cy, stride)
        gt_boxes: (G,4) xyxy; rows with any non-finite value are skipped (an animal that is
            not croppable in this view -- the loader emits a NaN box rather than dropping the
            frame, so objectness still learns "no animal here").

    Returns (pos_anchor_ix, pos_gt_ix). Empty when there is no finite box.
    """
    keep = torch.isfinite(gt_boxes).all(-1)
    if not keep.any():
        return (torch.zeros(0, dtype=torch.long, device=anchors.device),) * 2
    gt = gt_boxes[keep]
    gt_ix = torch.nonzero(keep, as_tuple=True)[0]

    cx, cy, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    inside = ((cx[:, None] > gt[None, :, 0]) & (cx[:, None] < gt[None, :, 2]) &
              (cy[:, None] > gt[None, :, 1]) & (cy[:, None] < gt[None, :, 3]))
    gcx = (gt[:, 0] + gt[:, 2]) / 2
    gcy = (gt[:, 1] + gt[:, 3]) / 2
    r = CENTER_RADIUS * stride[:, None]
    near = ((cx[:, None] - gcx[None]).abs() < r) & ((cy[:, None] - gcy[None]).abs() < r)
    ok = inside | near
    if not ok.any():
        return (torch.zeros(0, dtype=torch.long, device=anchors.device),) * 2

    # An anchor can only serve one box: give it the one whose centre it is closest to, so two
    # overlapping animals do not both claim it and cancel.
    d = (cx[:, None] - gcx[None]) ** 2 + (cy[:, None] - gcy[None]) ** 2
    d = torch.where(ok, d, torch.full_like(d, float('inf')))
    best = d.argmin(1)
    pos = torch.nonzero(torch.isfinite(d.min(1).values), as_tuple=True)[0]
    return pos, gt_ix[best[pos]]


def detector_loss(obj_logits, boxes, anchors, gt_boxes, box_weight=5.0):
    """BCE(objectness) over every anchor + GIoU over the positives.

    Objectness is the whole classification signal: with one class, "is there an animal here"
    is all there is to say.
    """
    device = obj_logits.device
    B = obj_logits.shape[0]
    target = torch.zeros_like(obj_logits)
    losses_box, n_pos = [], 0
    for b in range(B):
        pos, gix = assign(anchors, gt_boxes[b])
        if pos.numel():
            target[b, pos] = 1.0
            losses_box.append(giou_loss(boxes[b, pos], gt_boxes[b][gix]))
            n_pos += pos.numel()
    obj = F.binary_cross_entropy_with_logits(obj_logits, target, reduction='sum') / max(n_pos, 1)
    box = (torch.cat(losses_box).sum() / max(n_pos, 1) if losses_box
           else torch.zeros((), device=device))
    return obj + box_weight * box, {'obj': float(obj.detach()), 'box': float(box.detach()),
                                    'n_pos': n_pos}


def decode(obj_logits, boxes, top_k=1, score_thresh=0.05, iou_thresh=0.5):
    """Top boxes for one image, NMS'd. Returns (boxes (N,4), scores (N,)).

    `top_k` is the expected animal count, not a hard cap: it is applied AFTER NMS so a frame
    with fewer animals returns fewer boxes rather than padding with duplicates.
    """
    scores = obj_logits.sigmoid()
    keep = scores >= score_thresh
    if not keep.any():
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,))
    order = scores[keep].argsort(descending=True)
    b, s = boxes[keep][order], scores[keep][order]

    kept_b, kept_s = [], []
    while b.numel() and len(kept_b) < top_k:
        kept_b.append(b[:1])
        kept_s.append(s[:1])
        survives = box_iou(b[:1], b)[0] < iou_thresh
        b, s = b[survives], s[survives]
    return torch.cat(kept_b), torch.cat(kept_s)
