"""Anchor assignment for the box predictor: centre prior (`assign`) or task-aligned (`assign_tal`).

Centre-prior rather than SimOTA: an anchor is a positive for a ground-truth box when its cell
centre falls inside that box AND within a fixed radius of the box centre, at any level. With one
class and a handful of instances per frame, SimOTA's dynamic-k machinery buys nothing.
"""
import torch

from .boxes import box_iou


# In cells; YOLOX's own value.
CENTER_RADIUS = 2.5


def assign_tal(anchors, gt_boxes, obj_logits, boxes, topk=13, alpha=1.0, beta=6.0):
    """Task-aligned assignment (YOLOv8/RTMDet style).

    Candidate anchors are inside each finite GT box, then the detached prediction quality
    ``sigmoid(obj)**alpha * IoU**beta`` selects up to ``topk`` anchors per GT.  An anchor that
    appears in several GT top-k sets is assigned to the GT with the greatest alignment score.
    """
    empty = (torch.zeros(0, dtype=torch.long, device=anchors.device),) * 2
    finite = torch.isfinite(gt_boxes).all(-1)
    if not finite.any():
        return empty
    gt = gt_boxes[finite]
    gt_ix = torch.nonzero(finite, as_tuple=True)[0]
    cx, cy = anchors[:, 0], anchors[:, 1]
    candidate = ((cx[:, None] > gt[None, :, 0]) & (cx[:, None] < gt[None, :, 2]) &
                 (cy[:, None] > gt[None, :, 1]) & (cy[:, None] < gt[None, :, 3]))
    if not candidate.any():
        return empty
    with torch.no_grad():
        quality = (obj_logits.sigmoid()[:, None].pow(alpha) *
                   box_iou(boxes, gt).clamp_min(0).pow(beta))
        quality = torch.where(candidate, quality, torch.zeros_like(quality))
        k = min(max(int(topk), 1), anchors.shape[0])
        _, top_idx = quality.topk(k, dim=0)
        pos_mask = torch.zeros_like(candidate)
        pos_mask.scatter_(0, top_idx, candidate.gather(0, top_idx))
        if not pos_mask.any():
            return empty
        multi = pos_mask.sum(1) > 1
        if multi.any():
            best_gt = quality[multi].argmax(1)
            pos_mask[multi] = False
            pos_mask[multi, best_gt] = True
    pos_a, pos_g = torch.nonzero(pos_mask, as_tuple=True)
    return pos_a, gt_ix[pos_g]


def assign(anchors, gt_boxes, max_pos_per_gt=None):
    """Positive anchors for each ground-truth box.

    Inputs:
        anchors -- (A,3) of (cx, cy, stride).
        gt_boxes -- (G,4) xyxy; rows with any non-finite value are skipped (the loader emits a
            NaN box for an animal that is not croppable in this view, so objectness still
            learns "no animal here").
        max_pos_per_gt -- caps each GT's CANDIDATE set to its closest anchors by centre
            distance, before the per-anchor uniqueness resolution; None (default) is uncapped
            and byte-identical to every checkpoint on record. An anchor capped out of one GT's
            top-k may still be closest to a DIFFERENT GT that still wants it.
    Outputs:
        (pos_anchor_ix, pos_gt_ix). Empty when there is no finite box.
    Side effects:
        None.
    Notes:
        Candidacy is AND, not OR: `yolox.py` builds a box as `centre +- exp(ltrb) * stride`, so
        a predicted box ALWAYS contains its own anchor centre -- an anchor outside its assigned
        GT box has a target it cannot reach. An anchor serves only the box whose centre it is
        closest to, so overlapping animals do not both claim it and cancel.
    """
    empty = (torch.zeros(0, dtype=torch.long, device=anchors.device),) * 2
    keep = torch.isfinite(gt_boxes).all(-1)
    if not keep.any():
        return empty
    gt = gt_boxes[keep]
    gt_ix = torch.nonzero(keep, as_tuple=True)[0]

    cx, cy, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    inside = ((cx[:, None] > gt[None, :, 0]) & (cx[:, None] < gt[None, :, 2]) &
              (cy[:, None] > gt[None, :, 1]) & (cy[:, None] < gt[None, :, 3]))
    gcx = (gt[:, 0] + gt[:, 2]) / 2
    gcy = (gt[:, 1] + gt[:, 3]) / 2
    r = CENTER_RADIUS * stride[:, None]
    near = ((cx[:, None] - gcx[None]).abs() < r) & ((cy[:, None] - gcy[None]).abs() < r)
    ok = inside & near
    if not ok.any():
        return empty

    d = (cx[:, None] - gcx[None]) ** 2 + (cy[:, None] - gcy[None]) ** 2
    d = torch.where(ok, d, torch.full_like(d, float('inf')))
    if max_pos_per_gt is not None and max_pos_per_gt < d.shape[0]:
        _, top_idx = torch.topk(d, max_pos_per_gt, dim=0, largest=False)
        capped = torch.zeros_like(ok)
        capped.scatter_(0, top_idx, ok.gather(0, top_idx))
        d = torch.where(capped, d, torch.full_like(d, float('inf')))
    best = d.argmin(1)
    pos = torch.nonzero(torch.isfinite(d.min(1).values), as_tuple=True)[0]
    out = (pos, gt_ix[best[pos]])
    return out
