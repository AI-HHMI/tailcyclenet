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


def assign(anchors, gt_boxes, return_band=False):
    """Positive anchors for each ground-truth box.

    Args:
        anchors: (A,3) of (cx, cy, stride)
        gt_boxes: (G,4) xyxy; rows with any non-finite value are skipped (an animal that is
            not croppable in this view -- the loader emits a NaN box rather than dropping the
            frame, so objectness still learns "no animal here").
        return_band: also return an (A,) bool of the anchors that sit INSIDE some GT box but are
            positive for none -- the band `--ignore-band` exists to stop supervising. Off by
            default so the two-value return every caller already unpacks is unchanged.

    Returns (pos_anchor_ix, pos_gt_ix), plus the band when `return_band`. Empty when there is no
    finite box.
    """
    empty = (torch.zeros(0, dtype=torch.long, device=anchors.device),) * 2
    no_band = torch.zeros(anchors.shape[0], dtype=torch.bool, device=anchors.device)
    keep = torch.isfinite(gt_boxes).all(-1)
    if not keep.any():
        return (*empty, no_band) if return_band else empty
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
    # THE IGNORE BAND: inside an animal, positive for none. `detector_loss` starts from a zeros
    # target and sets only the positives, so today every one of these is supervised as "no animal
    # here" WHILE SITTING ON ONE. Screened at the shipped geometry it is 43.5% of every anchor in
    # the image on rat-city's 640x640 tiles at scale 1.0 -- 13.5x the positive count -- and
    # provably 0.00% on branson-fly, where a 30 px fly is smaller than the centre radius at every
    # stride so `inside` is a subset of `near`. That makes branson-fly a free inertness control.
    # APT's assigner has the same three bands and ignores this one deliberately.
    band = inside.any(-1) & ~ok.any(-1) if return_band else no_band
    if not ok.any():
        return (*empty, band) if return_band else empty

    # An anchor can only serve one box: give it the one whose centre it is closest to, so two
    # overlapping animals do not both claim it and cancel.
    d = (cx[:, None] - gcx[None]) ** 2 + (cy[:, None] - gcy[None]) ** 2
    d = torch.where(ok, d, torch.full_like(d, float('inf')))
    best = d.argmin(1)
    pos = torch.nonzero(torch.isfinite(d.min(1).values), as_tuple=True)[0]
    out = (pos, gt_ix[best[pos]])
    return (*out, band) if return_band else out


def certified_anchors(anchors, regions, gt_boxes):
    """(A,) bool: which anchors sit in area an annotator certified as completely labelled.

    An anchor is certified when its centre is inside any `regions.pq` rectangle OR inside any
    finite GT box. The union with the boxes is APT's own rule -- `APT_interface.py:331` concatenates
    `extra_roi` with the per-target loss masks and treats the union as the labelled area -- and
    without it an animal's own anchors would be unsupervised wherever the annotator drew no Label
    Box, which is most frames.

    Non-finite rows in either input are dropped, so `box_collate`'s NaN padding certifies nothing
    rather than certifying the origin.

    OUTSIDE this set the objectness target is UNKNOWN, not negative. That is the whole point: on
    `rat-city-annotated` a labelled frame names a median of 2 rats where the tracker finds 11, so
    training the other ~9 as background is a false negative per rat per frame.
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
                  kpts=None, gt_kpts=None, kpt_weight=1.0, kpt_score_weight=1.0,
                  regions=None, ignore_band=False):
    """BCE(objectness) over every anchor + GIoU over the positives.

    Objectness is the whole classification signal: with one class, "is there an animal here"
    is all there is to say.

    `kpts` / `gt_kpts` add the keypoint branch's terms over the SAME positives the box term uses
    -- the centre-prior assignment already selects them, and reusing it is what keeps the two
    branches trained on the same notion of "this anchor owns this animal". Absent, nothing about
    this function changes.

    `regions` (B,M,4) restricts the objectness BCE to anchors inside the CERTIFIED area (see
    `certified_anchors`). Absent, this function is bit-identical to what every recorded detector
    trained on -- which `tests/test_detector.py` asserts, because that equality is the only thing
    keeping reports 10-15's numbers comparable.

    **THE NORMALISER IS DELIBERATELY UNCHANGED**, `/ max(n_pos, B)`. Masking shrinks the objectness
    SUM without shrinking its divisor, so it silently reweights `obj` against `box_weight` -- by
    ~100x on a full frame (104 certified of 7,056) and ~10x on a good tile. That is a real effect
    and the alternative (normalising by the certified count) would have made the masked and
    unmasked arms differ in two things at once. So the shift is left in and `parts['certified']`
    REPORTS it, because a silent reweighting is exactly what an arm would misattribute to the mask.
    """
    device = obj_logits.device
    B = obj_logits.shape[0]
    target = torch.zeros_like(obj_logits)
    # ONE `weight` TENSOR SERVES BOTH MASKS. `--use-regions` certifies where objectness may be
    # supervised at all; `--ignore-band` withdraws the anchors that sit on an animal without being
    # positive for it. They compose by multiplication and either alone allocates it.
    weight = None if (regions is None and not ignore_band) else torch.ones_like(obj_logits)
    n_band, n_cert = 0, 0.0
    losses_box, n_pos = [], 0
    kpt_reg, kpt_sc, n_kpt, n_vis = [], [], 0, 0
    for b in range(B):
        if ignore_band:
            pos, gix, band = assign(anchors, gt_boxes[b], return_band=True)
        else:
            pos, gix = assign(anchors, gt_boxes[b])
            band = None
        if regions is not None:
            weight[b] = certified_anchors(anchors, regions[b], gt_boxes[b]).to(weight.dtype)
            n_cert += float(weight[b].mean())
        if band is not None:
            # AFTER the certification, so the two masks INTERSECT: an anchor must be both
            # certified and not-in-the-band to be supervised. `n_cert` is accumulated above rather
            # than read off `weight` at the end, or `parts['certified']` would report the PRODUCT
            # of the two masks under an arm running both and understate the certified area.
            weight[b] = weight[b] * (~band).to(weight.dtype)
            n_band += int(band.sum())
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
    if weight is None:
        obj_all = F.binary_cross_entropy_with_logits(obj_logits, target, reduction='sum')
    else:
        # A POSITIVE IS CERTIFIED BY CONSTRUCTION -- `assign` only fires inside a GT box and
        # `certified_anchors` unions those boxes in -- but it is forced here rather than assumed,
        # because a positive dropped from the objectness term would be an animal trained as
        # nothing, and no loss curve would show it.
        weight = torch.maximum(weight, target)
        obj_all = (F.binary_cross_entropy_with_logits(obj_logits, target, reduction='none')
                   * weight).sum()
    obj = obj_all / max(n_pos, B)
    box = (torch.cat(losses_box).sum() / max(n_pos, 1) if losses_box
           else torch.zeros((), device=device))
    total = obj + box_weight * box
    parts = {'obj': float(obj.detach()), 'box': float(box.detach()), 'n_pos': n_pos}
    if regions is not None:
        parts['certified'] = n_cert / max(B, 1)
    if ignore_band:
        # THE BAND FRACTION, printed for the same reason `certified` is: masking shrinks the
        # objectness sum without shrinking its `/ max(n_pos, B)` divisor, so it silently reweights
        # `obj` against `box_weight`. At the screened 43.5% that is a ~2x shift an arm would
        # otherwise misattribute to the band itself.
        parts['ignored'] = n_band / max(obj_logits.numel(), 1)
    if kpts is not None and gt_kpts is not None:
        # Mean over the IMAGES that had a positive, matching how `box` is normalised. Both lists
        # are empty when nothing was assigned, and then these are exact zeros with no gradient.
        kr = (torch.stack(kpt_reg).mean() if kpt_reg else torch.zeros((), device=device))
        ks = (torch.stack(kpt_sc).mean() if kpt_sc else torch.zeros((), device=device))
        total = total + kpt_weight * kr + kpt_score_weight * ks
        parts |= {'kpt': float(kr.detach()), 'kpt_score': float(ks.detach()),
                  'n_kpt': n_kpt, 'n_vis': n_vis}
    return total, parts


def decode(obj_logits, boxes, top_k=1, score_thresh=0.05, iou_thresh=0.5, return_index=False):
    """Top boxes for one image, NMS'd. Returns (boxes (N,4), scores (N,)).

    `top_k` is the expected animal count, not a hard cap: it is applied AFTER NMS so a frame
    with fewer animals returns fewer boxes rather than padding with duplicates.

    `return_index=True` adds the ANCHOR index of each kept box. The keypoint branch emits per
    anchor, so that index is the only way to pair a surviving box with its own keypoints --
    recovering it afterwards by matching box geometry is ambiguous wherever two anchors decode to
    near-identical boxes, which is exactly what NMS is there to collapse.
    """
    scores = obj_logits.sigmoid()
    keep = scores >= score_thresh
    if not keep.any():
        empty = (boxes.new_zeros((0, 4)), scores.new_zeros((0,)))
        return (*empty, torch.zeros(0, dtype=torch.long, device=boxes.device)) \
            if return_index else empty
    # STABLE, because these scores are SATURATED: 98.5% of rat-city's objectness and 99.98% of
    # 3dpop's sit at exactly 1.0, so almost every comparison in this sort is a tie and an unstable
    # tie-break decides which of two overlapping boxes survives greedy NMS -- and in what row
    # order the survivors leave, which is the order `associate` and `CrossViewTracker` birth into.
    # Without this the box set is reproducible only for a fixed torch version and device, which is
    # exactly the property `--det-cache` exists to guarantee across two arms.
    order = scores[keep].argsort(descending=True, stable=True)
    b, s = boxes[keep][order], scores[keep][order]
    ix = keep.nonzero().flatten()[order]

    kept_b, kept_s, kept_i = [], [], []
    while b.numel() and len(kept_b) < top_k:
        kept_b.append(b[:1])
        kept_s.append(s[:1])
        kept_i.append(ix[:1])
        survives = box_iou(b[:1], b)[0] < iou_thresh
        b, s, ix = b[survives], s[survives], ix[survives]
    out = (torch.cat(kept_b), torch.cat(kept_s))
    return (*out, torch.cat(kept_i)) if return_index else out
