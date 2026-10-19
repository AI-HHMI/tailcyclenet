"""Box-predictor losses: objectness BCE + GIoU/CIoU over the assigned positives.

The regression target is the crop rule's box (or, under `box_target = 'extent'`, the tight
extent the rule is applied to after detection), so the detector learns "reproduce the crop the
pose model was trained on", not "find an animal"; only the first keeps downstream accuracy.
"""
import torch
import torch.nn.functional as F

from .assign import assign, assign_tal
from .boxes import paired_iou


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


def ciou_loss(pred, target, eps=1e-7):
    """1 - CIoU, elementwise over matched ``xyxy`` box pairs."""
    import math
    ap = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
    at = (target[:, 2] - target[:, 0]).clamp(0) * (target[:, 3] - target[:, 1]).clamp(0)
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, 0] * wh[:, 1]
    union = ap + at - inter + eps
    iou = inter / union
    clt = torch.min(pred[:, :2], target[:, :2])
    crb = torch.max(pred[:, 2:], target[:, 2:])
    cwh = (crb - clt).clamp_min(0)
    c2 = cwh[:, 0].square() + cwh[:, 1].square() + eps
    pcx, pcy = (pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2
    tcx, tcy = (target[:, 0] + target[:, 2]) / 2, (target[:, 1] + target[:, 3]) / 2
    rho2 = (pcx - tcx).square() + (pcy - tcy).square()
    pw = (pred[:, 2] - pred[:, 0]).clamp_min(eps)
    ph = (pred[:, 3] - pred[:, 1]).clamp_min(eps)
    tw = (target[:, 2] - target[:, 0]).clamp_min(eps)
    th = (target[:, 3] - target[:, 1]).clamp_min(eps)
    v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)).square()
    with torch.no_grad():
        a = v / (1 - iou + v + eps)
    return 1.0 - iou + rho2 / c2 + a * v


def detector_loss(obj_logits, boxes, anchors, gt_boxes, box_weight=5.0,
                  negative_supervision=None, iou_aware=False, iou_aware_warmup=2000, it=None,
                  max_pos_per_gt=None, assignment='center', box_loss_fn='giou',
                  tal_topk=13, tal_alpha=1.0, tal_beta=6.0):
    """BCE(objectness) over every anchor + GIoU over the positives.

    Inputs:
        obj_logits (B,A), boxes (B,A,4), anchors (A,3), gt_boxes (B,G,4).
        iou_aware -- at a positive anchor the BCE target becomes the DETACHED predicted/GT IoU
            instead of a hard 1.0, so the score becomes a localisation-quality estimate. False
            (default) is byte-identical to every checkpoint on record.
        iou_aware_warmup / it -- the IoU target stays at hard 1.0 while `it < warmup`
            (chicken-and-egg: with the head's rare-positive bias, predicted IoU starts near 0);
            `it=None` behaves as "past warmup".
        negative_supervision (B,) bool -- whether each image's non-positive anchors are known
            background. False keeps only positive objectness terms; it is reported as
            `parts['negative_frames']`.
        box_weight -- the box term's weight; max_pos_per_gt -- `assign`'s
            cap, passed straight through; assignment / box_loss_fn / tal_* -- assigner and loss
            selection.
    Outputs:
        (total, parts): total = obj + box_weight*box; parts carries obj, box, n_pos and any
        optional diagnostics (negative_frames, iou_target).
    Side effects:
        None.
    Notes:
        The normaliser is deliberately `/ max(n_pos, B)`: masking shrinks the objectness SUM
        without shrinking its divisor, silently reweighting `obj` against `box_weight`, and
        `parts['negative_frames']` reports the shift. The negative weight-forcing keys on
        WHETHER an anchor is positive (`pos_mask`, binary), never on its target VALUE -- under
        `iou_aware` those come apart.
    """
    device = obj_logits.device
    B = obj_logits.shape[0]
    target = torch.zeros_like(obj_logits)
    pos_mask = torch.zeros_like(obj_logits, dtype=torch.bool)
    use_iou_target = iou_aware and (it is None or it >= iou_aware_warmup)
    if negative_supervision is not None:
        negative_supervision = torch.as_tensor(negative_supervision, dtype=torch.bool,
                                               device=device).reshape(-1)
        if negative_supervision.numel() != B:
            raise ValueError(f'negative_supervision has {negative_supervision.numel()} values '
                             f'for batch size {B}')
    weight = None if negative_supervision is None else torch.ones_like(obj_logits)
    losses_box, n_pos = [], 0
    for b in range(B):
        if assignment == 'tal':
            pos, gix = assign_tal(anchors, gt_boxes[b], obj_logits[b].detach(),
                                  boxes[b].detach(), topk=tal_topk, alpha=tal_alpha,
                                  beta=tal_beta)
        else:
            pos, gix = assign(anchors, gt_boxes[b], max_pos_per_gt=max_pos_per_gt)
        if negative_supervision is not None and not bool(negative_supervision[b]):
            weight[b].zero_()
        if pos.numel():
            pos_mask[b, pos] = True
            if use_iou_target:
                with torch.no_grad():
                    target[b, pos] = paired_iou(boxes[b, pos], gt_boxes[b][gix]).clamp(0.0, 1.0)
            else:
                target[b, pos] = 1.0
            loss_fn = ciou_loss if box_loss_fn == 'ciou' else giou_loss
            losses_box.append(loss_fn(boxes[b, pos], gt_boxes[b][gix]))
            n_pos += pos.numel()
    if weight is None:
        obj_all = F.binary_cross_entropy_with_logits(obj_logits, target, reduction='sum')
    else:
        weight = torch.maximum(weight, pos_mask.to(weight.dtype))
        obj_all = (F.binary_cross_entropy_with_logits(obj_logits, target, reduction='none')
                   * weight).sum()
    obj = obj_all / max(n_pos, B)
    box = (torch.cat(losses_box).sum() / max(n_pos, 1) if losses_box
           else torch.zeros((), device=device))
    total = obj + box_weight * box
    parts = {'obj': float(obj.detach()), 'box': float(box.detach()), 'n_pos': n_pos}
    if negative_supervision is not None:
        parts['negative_frames'] = float(negative_supervision.float().mean())
    if iou_aware:
        parts['iou_target'] = float(target[pos_mask].mean()) if bool(pos_mask.any()) else 0.0
    return total, parts


