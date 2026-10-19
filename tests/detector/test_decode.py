"""`decode`: score threshold, NMS, top-k and the trace side channel."""
import torch

from tailcyclenet.detector.decode import decode


def test_decode_suppresses_duplicates():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.], [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2, 'the two overlapping boxes must collapse to one'
    assert s[0] > s[1]


def test_decode_trace_reports_score_nms_and_top_k_without_changing_output():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.],
                          [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    plain = decode(logits, boxes, top_k=1, iou_thresh=0.5)
    traced = decode(logits, boxes, top_k=1, iou_thresh=0.5, return_trace=True)
    got = traced[:2]
    trace = traced[2]
    for a, b in zip(plain, got):
        assert torch.equal(a, b), 'diagnostic tracing must not alter deployed boxes/scores'
    assert (trace['n_total'], trace['n_score'], trace['n_nms'], trace['n_top_k']) == (3, 3, 2, 1)
    assert trace['score_boxes'].shape[0] == 3
    assert trace['nms_boxes'].shape[0] == 2


def test_decode_center_dist_is_on_by_default():
    """A5: a near-concentric duplicate (IoU 0.16) survives IoU-only NMS; the default
    centre-distance NMS collapses it, and `center_dist_thresh=None` restores IoU-only."""
    boxes = torch.tensor([[10., 10., 50., 50.], [-20., -20., 80., 80.], [400., 400., 440., 440.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2 and s[0] > s[1]
    assert decode(logits, boxes, top_k=5, iou_thresh=0.5, center_dist_thresh=None)[0].shape[0] == 3
