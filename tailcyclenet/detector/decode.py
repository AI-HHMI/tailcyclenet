"""Decoding one image's per-anchor outputs into NMS'd boxes."""
import torch

from .boxes import NMS_CENTER_DIST, box_center_dist, box_iou


def decode(obj_logits, boxes, top_k=1, score_thresh=0.05, iou_thresh=0.5,
          center_dist_thresh=NMS_CENTER_DIST, return_trace=False):
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
            Default `NMS_CENTER_DIST` (plan A5; selection and deployment alike); pass
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
