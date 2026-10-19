"""Assignment, loss and decoding for the box predictor.

Centre-prior assignment rather than SimOTA: an anchor is a positive for a ground-truth box when
its cell centre falls inside that box AND within a fixed radius of the box centre, at any level.
With one class and a handful of instances per frame, SimOTA's dynamic-k machinery buys nothing
and adds a second thing that can be wrong.

The regression target is the crop rule's box (or, under `box_target = 'extent'`, the tight
extent the rule is applied to after detection), so what the detector learns is "reproduce the
crop the pose model was trained on" -- not "find an animal". Those are different objectives and only
the first one keeps the downstream accuracy.
"""
import torch
import torch.nn.functional as F

from .boxes import box_center_dist, box_iou, paired_iou

# In cells; YOLOX's own value.
CENTER_RADIUS = 2.5


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


def decode(obj_logits, boxes, top_k=1, score_thresh=0.05, iou_thresh=0.5,
          center_dist_thresh=0.5, return_trace=False):
    """Top boxes for one image, NMS'd. Returns (boxes (N,4), scores (N,)).

    Inputs:
        obj_logits, boxes -- per-anchor scores and boxes for one image.
        top_k -- the expected animal count, applied AFTER NMS (a frame with fewer animals
            returns fewer boxes, not padded duplicates).
        score_thresh -- minimum objectness to survive (default 0.05).
        iou_thresh -- IoU NMS gate (default 0.5, byte-identical to every checkpoint on
            record); reachable from a config only because `detect_raw`/`score_dataset` thread
            it through.
        center_dist_thresh -- second, independent survival condition in units of box side (see
            `box_center_dist`): suppresses near-concentric duplicates IoU alone lets through.
            Default 0.5 (plan A5, CONFIRMED -- a deliberate break from older checkpoints); pass
            None for the pre-A5 byte-identical behaviour.
        return_trace -- adds an output-neutral diagnostic dict (candidate counts, pre/post-NMS
            boxes and scores); opt-in because it retains tensors.
    Outputs:
        (boxes (N,4), scores (N,)), plus the trace when requested.
    Side effects:
        None.
    Notes:
        The sort is STABLE: these scores are saturated near 1.0, so an unstable tie-break would
        decide which overlapping box survives greedy NMS -- and the row order `associate` and
        `CrossViewTracker` birth into.
    """
    scores = obj_logits.sigmoid()
    keep = scores >= score_thresh
    if not keep.any():
        empty = (boxes.new_zeros((0, 4)), scores.new_zeros((0,)))
        index = torch.zeros(0, dtype=torch.long, device=boxes.device)
        all_order = scores.argsort(descending=True, stable=True)
        trace = {'n_total': int(scores.numel()), 'n_score': 0, 'n_nms': 0, 'n_top_k': 0,
                 'all_boxes': boxes[all_order], 'all_scores': scores[all_order],
                 'all_index': all_order,
                 'score_boxes': boxes.new_zeros((0, 4)), 'score_scores': scores.new_zeros((0,)),
                 'nms_boxes': boxes.new_zeros((0, 4)), 'nms_scores': scores.new_zeros((0,)),
                 'nms_index': index}
        return (*empty, trace) if return_trace else empty
    order = scores[keep].argsort(descending=True, stable=True)
    b, s = boxes[keep][order], scores[keep][order]
    ix = keep.nonzero().flatten()[order]

    score_b, score_s = b, s
    kept_b, kept_s, kept_i = [], [], []
    while b.numel():
        kept_b.append(b[:1])
        kept_s.append(s[:1])
        kept_i.append(ix[:1])
        survives = box_iou(b[:1], b)[0] < iou_thresh
        if center_dist_thresh is not None:
            survives &= box_center_dist(b[:1], b)[0] >= center_dist_thresh
        b, s, ix = b[survives], s[survives], ix[survives]
    nms_b, nms_s, nms_i = torch.cat(kept_b), torch.cat(kept_s), torch.cat(kept_i)
    out = (nms_b[:top_k], nms_s[:top_k])
    if return_trace:
        all_order = scores.argsort(descending=True, stable=True)
        trace = {'n_total': int(scores.numel()), 'n_score': int(keep.sum()),
                 'n_nms': int(nms_b.shape[0]), 'n_top_k': int(out[0].shape[0]),
                 'all_boxes': boxes[all_order], 'all_scores': scores[all_order],
                 'all_index': all_order,
                 'score_boxes': score_b, 'score_scores': score_s,
                 'nms_boxes': nms_b, 'nms_scores': nms_s, 'nms_index': nms_i}
        out = (*out, trace)
    return out
