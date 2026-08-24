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

# In cells; YOLOX's own value.
CENTER_RADIUS = 2.5


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

    `box_iou` above is the cross-product form (`(N,M)`), which is what NMS and evaluation want.
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


def assign_tal(anchors, gt_boxes, obj_logits, boxes, topk=13, alpha=1.0, beta=6.0,
               soft_prior=False):
    """Task-aligned assignment (YOLOv8/RTMDet style).

    Candidate anchors are inside each finite GT box, then the detached prediction quality
    ``sigmoid(obj)**alpha * IoU**beta`` selects up to ``topk`` anchors per GT.  An anchor that
    appears in several GT top-k sets is assigned to the GT with the greatest alignment score.

    G4 (`soft_prior=True`, detector_architecture_sweep plan): the strict `inside` candidacy mask
    is relaxed to `inside | near`, reusing `assign()`'s own `CENTER_RADIUS=2.5`-cell radius --
    an anchor near (but outside) the box can still be a candidate, which helps edge-truncated
    animals where the GT box centre sits near the frame border and few anchors land inside.
    Default `False` is byte-identical to every checkpoint on record.
    """
    empty = (torch.zeros(0, dtype=torch.long, device=anchors.device),) * 2
    finite = torch.isfinite(gt_boxes).all(-1)
    if not finite.any():
        return empty
    gt = gt_boxes[finite]
    gt_ix = torch.nonzero(finite, as_tuple=True)[0]
    cx, cy = anchors[:, 0], anchors[:, 1]
    inside = ((cx[:, None] > gt[None, :, 0]) & (cx[:, None] < gt[None, :, 2]) &
              (cy[:, None] > gt[None, :, 1]) & (cy[:, None] < gt[None, :, 3]))
    if soft_prior:
        stride = anchors[:, 2]
        gcx = (gt[:, 0] + gt[:, 2]) / 2
        gcy = (gt[:, 1] + gt[:, 3]) / 2
        r = CENTER_RADIUS * stride[:, None]
        near = ((cx[:, None] - gcx[None]).abs() < r) & ((cy[:, None] - gcy[None]).abs() < r)
        candidate = inside | near
    else:
        candidate = inside
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


def certified_anchors(anchors, regions, gt_boxes):
    """(A,) bool: which anchors sit in area an annotator certified as completely labelled.

    An anchor is certified when its centre is inside any `regions.pq` rectangle OR inside any
    finite GT box -- the union is APT's own rule, and without it an animal's own anchors would be
    unsupervised wherever the annotator drew no Label Box, which is most frames. Non-finite rows
    in either input are dropped, so `box_collate`'s NaN padding certifies nothing rather than
    certifying the origin.

    OUTSIDE this set the objectness target is UNKNOWN, not negative: on a labelled root a frame
    names a median of 2 rats where the tracker finds 11, so training the other ~9 as background
    is a false negative per rat per frame.
    """
    cx, cy = anchors[:, 0], anchors[:, 1]
    ok = torch.zeros(cx.shape, dtype=torch.bool, device=anchors.device)
    for r in (regions, gt_boxes):
        if r is None or r.numel() == 0:
            continue
        r = r[torch.isfinite(r).all(-1)]
        if r.numel() == 0:
            continue
        ok |= ((cx[:, None] > r[None, :, 0]) & (cx[:, None] < r[None, :, 2]) &
               (cy[:, None] > r[None, :, 1]) & (cy[:, None] < r[None, :, 3])).any(1)
    return ok


def keypoint_loss(pred, target, gt_boxes):
    """L1 over box sides + BCE on the score channel, both masked by the LABEL, not by K.

    `pred` (P,K,3) at the assigned positives, `target` (P,K,3) of (x, y, vis), `gt_boxes` (P,4).

    PREDICT ALL K ALWAYS, SUPERVISE ONLY WHERE THE LABEL IS FINITE. Four things here are silent
    if got wrong:

    1. **Mask, never fill.** `nan_to_num(target)` then L1 would supervise every unlabelled
       keypoint toward (0, 0) with a healthy-looking loss curve. A non-finite target is the OFF
       SWITCH for that element; and do not multiply a filled tensor by the mask either: `0 * NaN`
       is `NaN` and poisons the whole batch's gradient.
    2. **Normalise by the FINITE COUNT, not by K.** Otherwise an animal with 2 of 9 points
       labelled contributes 2/9 the gradient of a fully-labelled one and the term's magnitude
       rides on label density rather than on error.
    3. **The score channel trains against `status`, not against coordinate-finiteness.** They
       come apart in both directions -- the format permits `x, y` null on a VISIBLE row -- so the
       two masks cannot be the same tensor. `data.py` already NaNs `vis` where the session made
       no assessment, and a NaN there withholds the score loss rather than asserting "not
       visible".
    4. **L1 over the box side, not OKS.** OKS buys scale-invariance through a per-keypoint sigma
       table we would have to invent per dataset; dividing by the side gets the same invariance
       with no table.

    Notes.

    The finite coordinates are SELECTED, not multiplied by a mask (see note 1 above):
    `target[xy_ok]` is (n_finite, 2) with no NaN in it, so `0 * NaN` can never poison the
    gradient. The normaliser is the per-animal box side `(P,1)`.
    """
    if pred.numel() == 0:
        z = pred.sum() * 0.0
        return z, z, 0, 0
    side = (0.5 * ((gt_boxes[:, 2] - gt_boxes[:, 0]) + (gt_boxes[:, 3] - gt_boxes[:, 1]))
            ).clamp_min(1.0)[:, None]
    xy_ok = torch.isfinite(target[..., :2]).all(-1)
    if xy_ok.any():
        d = (pred[..., :2][xy_ok] - target[..., :2][xy_ok]).abs().sum(-1)
        reg = d / side.expand_as(xy_ok)[xy_ok]
        reg = reg.sum() / xy_ok.sum()
    else:
        reg = pred.sum() * 0.0
    v_ok = torch.isfinite(target[..., 2])
    if v_ok.any():
        sc = F.binary_cross_entropy_with_logits(
            pred[..., 2][v_ok], target[..., 2][v_ok], reduction='mean')
    else:
        sc = pred.sum() * 0.0
    return reg, sc, int(xy_ok.sum()), int(v_ok.sum())


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


def quality_focal_loss(logits, targets, gamma=2.0):
    """Quality Focal Loss per element for continuous quality targets in ``[0, 1]``."""
    sigmoid = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (targets - sigmoid).abs().pow(gamma) * bce


def detector_loss(obj_logits, boxes, anchors, gt_boxes, box_weight=5.0,
                  kpts=None, gt_kpts=None, kpt_weight=1.0, kpt_score_weight=1.0,
                  regions=None, ignore=None, iou_aware=False, iou_aware_warmup=2000, it=None,
                  max_pos_per_gt=None, assignment='center', box_loss_fn='giou',
                  focal_obj=False, focal_gamma=2.0, tal_topk=13, tal_alpha=1.0,
                  tal_beta=6.0, tal_soft_prior=False):
    """BCE(objectness) over every anchor + GIoU over the positives.

    Inputs:
        obj_logits (B,A), boxes (B,A,4), anchors (A,3), gt_boxes (B,G,4).
        kpts / gt_kpts -- add the keypoint branch's terms over the SAME positives the box term
            uses; absent, nothing about this function changes.
        iou_aware -- at a positive anchor the BCE target becomes the DETACHED predicted/GT IoU
            instead of a hard 1.0, so the score becomes a localisation-quality estimate. False
            (default) is byte-identical to every checkpoint on record.
        iou_aware_warmup / it -- the IoU target stays at hard 1.0 while `it < warmup`
            (chicken-and-egg: with the head's rare-positive bias, predicted IoU starts near 0);
            `it=None` behaves as "past warmup".
        regions (B,M,4) -- restrict the objectness BCE to CERTIFIED area; ignore (B,M,4) --
            the opposite polarity (`instances.pq` PRESENT boxes). Independent; both reported in
            `parts` (`certified`/`ignored`).
        box_weight / kpt_weight / kpt_score_weight -- term weights; max_pos_per_gt -- `assign`'s
            cap, passed straight through; assignment / box_loss_fn / focal_* / tal_* -- assigner
            and loss selection.
    Outputs:
        (total, parts): total = obj + box_weight*box (+ keypoint terms); parts carries obj,
        box, n_pos and any optional diagnostics (certified, ignored, iou_target, kpt, kpt_score).
    Side effects:
        None.
    Notes:
        The normaliser is deliberately `/ max(n_pos, B)`: masking shrinks the objectness SUM
        without shrinking its divisor, silently reweighting `obj` against `box_weight`, and
        `parts['certified']` reports the shift. The certified/ignore weight-forcing keys on
        WHETHER an anchor is positive (`pos_mask`, binary), never on its target VALUE -- under
        `iou_aware` those come apart.
    """
    device = obj_logits.device
    B = obj_logits.shape[0]
    target = torch.zeros_like(obj_logits)
    pos_mask = torch.zeros_like(obj_logits, dtype=torch.bool)
    use_iou_target = iou_aware and (it is None or it >= iou_aware_warmup)
    weight = None if regions is None and ignore is None else torch.ones_like(obj_logits)
    n_cert, n_ignored = 0.0, 0.0
    losses_box, n_pos = [], 0
    kpt_reg, kpt_sc, n_kpt, n_vis = [], 0, 0, 0
    for b in range(B):
        if assignment == 'tal':
            pos, gix = assign_tal(anchors, gt_boxes[b], obj_logits[b].detach(),
                                  boxes[b].detach(), topk=tal_topk, alpha=tal_alpha,
                                  beta=tal_beta, soft_prior=tal_soft_prior)
        else:
            pos, gix = assign(anchors, gt_boxes[b], max_pos_per_gt=max_pos_per_gt)
        if regions is not None:
            cert = certified_anchors(anchors, regions[b], gt_boxes[b])
            weight[b] *= cert.to(weight.dtype)
            n_cert += float(cert.float().mean())
        if ignore is not None:
            ig = certified_anchors(anchors, ignore[b], None)
            weight[b] *= (~ig).to(weight.dtype)
            n_ignored += float(ig.float().mean())
        if pos.numel():
            pos_mask[b, pos] = True



            if focal_obj or use_iou_target:
                with torch.no_grad():
                    target[b, pos] = paired_iou(boxes[b, pos], gt_boxes[b][gix]).clamp(0.0, 1.0)
            else:
                target[b, pos] = 1.0
            loss_fn = ciou_loss if box_loss_fn == 'ciou' else giou_loss
            losses_box.append(loss_fn(boxes[b, pos], gt_boxes[b][gix]))
            n_pos += pos.numel()
            if kpts is not None and gt_kpts is not None:
                r, s, nk, nv = keypoint_loss(kpts[b, pos], gt_kpts[b][gix], gt_boxes[b][gix])
                kpt_reg.append(r)
                kpt_sc.append(s)
                n_kpt += nk
                n_vis += nv
    if focal_obj:
        obj_all = quality_focal_loss(obj_logits, target, gamma=focal_gamma)
        if weight is not None:
            obj_all = obj_all * weight
        obj_all = obj_all.sum()
    elif weight is None:
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
    if regions is not None:
        parts['certified'] = n_cert / max(B, 1)
    if ignore is not None:
        parts['ignored'] = n_ignored / max(B, 1)
    if iou_aware:
        parts['iou_target'] = float(target[pos_mask].mean()) if bool(pos_mask.any()) else 0.0
    if kpts is not None and gt_kpts is not None:
        kr = (torch.stack(kpt_reg).mean() if kpt_reg else torch.zeros((), device=device))
        ks = (torch.stack(kpt_sc).mean() if kpt_sc else torch.zeros((), device=device))
        total = total + kpt_weight * kr + kpt_score_weight * ks
        parts |= {'kpt': float(kr.detach()), 'kpt_score': float(ks.detach()),
                  'n_kpt': n_kpt, 'n_vis': n_vis}
    return total, parts


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
    ca = torch.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], -1)
    cb = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], -1)
    d = torch.linalg.norm(ca[:, None] - cb[None], dim=-1)
    sa = 0.5 * ((a[:, 2] - a[:, 0]) + (a[:, 3] - a[:, 1]))
    sb = 0.5 * ((b[:, 2] - b[:, 0]) + (b[:, 3] - b[:, 1]))
    side = (0.5 * (sa[:, None] + sb[None])).clamp_min(eps)
    return d / side



def decode(obj_logits, boxes, top_k=1, score_thresh=0.05, iou_thresh=0.5,
          center_dist_thresh=0.5, return_index=False, return_trace=False):
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
        return_index -- adds each kept box's ANCHOR index (the only way to pair a surviving
            box with its own per-anchor keypoints).
        return_trace -- adds an output-neutral diagnostic dict (candidate counts, pre/post-NMS
            boxes and scores); opt-in because it retains tensors.
    Outputs:
        (boxes (N,4), scores (N,)), plus the anchor index and/or trace when requested.
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
        out = (*empty, index) if return_index else empty
        return (*out, trace) if return_trace else out
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
    if return_index:
        out = (*out, nms_i[:top_k])
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
