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
    # AND, not OR. `yolox.py` builds a box as `centre +- exp(ltrb) * stride` and `exp` is strictly
    # positive, so a predicted box ALWAYS contains its own anchor centre: an anchor outside its
    # assigned GT box has a target it cannot reach, while line 96 below simultaneously teaches
    # objectness to fire there. Upstream YOLOX takes the OR as a SimOTA *candidate* set and then
    # prunes it; with no SimOTA the candidate set is the positive set. Measured under `|`: 71% of
    # rat-city's positives sat outside their own box and the model could not fit 64 images it had
    # seen 1200 times (0.364 train recall at 0.66M params, 0.352 at 5.8M -- capacity is not the
    # limit, the labels were). Under `&`, 0.94-0.97.
    ok = inside & near
    if not ok.any():
        return (torch.zeros(0, dtype=torch.long, device=anchors.device),) * 2

    # An anchor can only serve one box: give it the one whose centre it is closest to, so two
    # overlapping animals do not both claim it and cancel.
    d = (cx[:, None] - gcx[None]) ** 2 + (cy[:, None] - gcy[None]) ** 2
    d = torch.where(ok, d, torch.full_like(d, float('inf')))
    best = d.argmin(1)
    pos = torch.nonzero(torch.isfinite(d.min(1).values), as_tuple=True)[0]
    return pos, gt_ix[best[pos]]


def keypoint_loss(pred, target, gt_boxes):
    """L1 over box sides + BCE on the score channel, both masked by the LABEL, not by K.

    `pred` (P,K,3) at the assigned positives, `target` (P,K,3) of (x, y, vis), `gt_boxes` (P,4).

    PREDICT ALL K ALWAYS, SUPERVISE ONLY WHERE THE LABEL IS FINITE. The first half is structural
    -- the head emits 3K channels at every anchor unconditionally -- and this function is the
    second half. Four things here are silent if got wrong:

    1. **Mask, never fill.** `nan_to_num(target)` then L1 would supervise every unlabelled
       keypoint toward (0, 0) -- the top-left corner -- with a healthy-looking loss curve the
       whole time. A non-finite target is the OFF SWITCH for that element, exactly as
       `losses.py`'s `grid_softmax_loss` drops non-finite targets. And do not multiply a filled
       tensor by the mask either: `0 * NaN` is `NaN` and poisons the whole batch's gradient.
    2. **Normalise by the FINITE COUNT, not by K.** Otherwise an animal with 2 of 9 points
       labelled contributes 2/9 the gradient of a fully-labelled one and the term's magnitude
       rides on label density rather than on error. Sparse labels are the norm here: rat-city
       labels 2.02 of 4 points per animal-frame.
    3. **The score channel trains against `status`, not against coordinate-finiteness.** They come
       apart in both directions -- the format permits `x, y` null on a VISIBLE row -- so the two
       masks cannot be the same tensor. `data.py` already NaNs `vis` where the session made no
       assessment, and a NaN there withholds the score loss rather than asserting "not visible".
    4. **L1 over the box side, not OKS.** OKS buys scale-invariance through a per-keypoint sigma
       table we would have to invent per dataset; dividing by the side gets the same invariance
       with no table, and this head does not need the precision OKS exists to protect.
    """
    if pred.numel() == 0:
        z = pred.sum() * 0.0
        return z, z, 0, 0
    side = (0.5 * ((gt_boxes[:, 2] - gt_boxes[:, 0]) + (gt_boxes[:, 3] - gt_boxes[:, 1]))
            ).clamp_min(1.0)[:, None]                                    # (P,1)
    xy_ok = torch.isfinite(target[..., :2]).all(-1)                      # (P,K)
    # SELECT, do not multiply -- see note 1. `target[xy_ok]` is (n_finite, 2) with no NaN in it.
    if xy_ok.any():
        d = (pred[..., :2][xy_ok] - target[..., :2][xy_ok]).abs().sum(-1)
        reg = d / side.expand_as(xy_ok)[xy_ok]
        reg = reg.sum() / xy_ok.sum()                                    # note 2
    else:
        reg = pred.sum() * 0.0
    v_ok = torch.isfinite(target[..., 2])                                # note 3
    if v_ok.any():
        sc = F.binary_cross_entropy_with_logits(
            pred[..., 2][v_ok], target[..., 2][v_ok], reduction='mean')
    else:
        sc = pred.sum() * 0.0
    return reg, sc, int(xy_ok.sum()), int(v_ok.sum())


def detector_loss(obj_logits, boxes, anchors, gt_boxes, box_weight=5.0,
                  kpts=None, gt_kpts=None, kpt_weight=1.0, kpt_score_weight=1.0):
    """BCE(objectness) over every anchor + GIoU over the positives.

    Objectness is the whole classification signal: with one class, "is there an animal here"
    is all there is to say.

    `kpts` / `gt_kpts` add the keypoint branch's terms over the SAME positives the box term uses
    -- the centre-prior assignment already selects them, and reusing it is what keeps the two
    branches trained on the same notion of "this anchor owns this animal". Absent, nothing about
    this function changes.
    """
    device = obj_logits.device
    B = obj_logits.shape[0]
    target = torch.zeros_like(obj_logits)
    losses_box, n_pos = [], 0
    kpt_reg, kpt_sc, n_kpt, n_vis = [], [], 0, 0
    for b in range(B):
        pos, gix = assign(anchors, gt_boxes[b])
        if pos.numel():
            target[b, pos] = 1.0
            losses_box.append(giou_loss(boxes[b, pos], gt_boxes[b][gix]))
            n_pos += pos.numel()
            if kpts is not None and gt_kpts is not None:
                r, s, nk, nv = keypoint_loss(kpts[b, pos], gt_kpts[b][gix], gt_boxes[b][gix])
                kpt_reg.append(r)
                kpt_sc.append(s)
                n_kpt += nk
                n_vis += nv
    # Divide by the image count, never by 1, when a batch has no positive at all: every animal
    # absent from every view is real on a multi-camera dataset, and a `sum` over 16 x 3780 anchors
    # over 1 is a loss of order 600 and one enormous gradient step.
    obj = (F.binary_cross_entropy_with_logits(obj_logits, target, reduction='sum')
           / max(n_pos, B))
    box = (torch.cat(losses_box).sum() / max(n_pos, 1) if losses_box
           else torch.zeros((), device=device))
    total = obj + box_weight * box
    parts = {'obj': float(obj.detach()), 'box': float(box.detach()), 'n_pos': n_pos}
    if kpts is not None and gt_kpts is not None:
        # Mean over the IMAGES that had a positive, matching how `box` is normalised. Both lists
        # are empty when nothing was assigned, and then these are exact zeros with no gradient.
        kr = (torch.stack(kpt_reg).mean() if kpt_reg else torch.zeros((), device=device))
        ks = (torch.stack(kpt_sc).mean() if kpt_sc else torch.zeros((), device=device))
        total = total + kpt_weight * kr + kpt_score_weight * ks
        parts |= {'kpt': float(kr.detach()), 'kpt_score': float(ks.detach()),
                  'n_kpt': n_kpt, 'n_vis': n_vis}
    return total, parts


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
