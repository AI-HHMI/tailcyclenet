"""Detector target assignment (SimOTA/TAL) and the box/objectness losses."""
import pytest
import torch

from tailcyclenet.detector.assign import assign, assign_tal
from tailcyclenet.detector.boxes import paired_iou
from tailcyclenet.detector.loss import ciou_loss, detector_loss, giou_loss
from tailcyclenet.detector.yolox import YOLOXNano


def test_giou_is_zero_for_a_perfect_box():
    b = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    assert float(giou_loss(b, b)) < 1e-5
    assert float(giou_loss(b, b + 200)) > 1.0        # disjoint -> worse than any overlap


def test_tal_finds_edge_candidates_without_center_radius():
    anchors = torch.tensor([[4., 4., 8.], [12., 12., 8.], [20., 20., 8.], [80., 80., 8.]])
    gt = torch.tensor([[0., 0., 100., 100.]])
    obj = torch.zeros(4)
    boxes = torch.tensor([[0., 0., 8., 8.], [4., 4., 16., 16.],
                          [12., 12., 24., 24.], [70., 70., 90., 90.]])
    tal_pos, tal_gt = assign_tal(anchors, gt, obj, boxes, topk=3)
    center_pos, _ = assign(anchors, gt)
    assert center_pos.numel() <= 1
    assert tal_pos.numel() >= 3 and bool((tal_gt == 0).all())


def test_ciou_has_finite_nonzero_gradient_for_disjoint_boxes():
    pred = torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
    target = torch.tensor([[5., 5., 6., 6.]])
    loss = ciou_loss(pred, target).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0


def test_recall_v2_loss_switches_are_live_together():
    torch.manual_seed(11)
    anchors = YOLOXNano(version='hybrid').anchor_points(128, 128, 'cpu')
    obj = torch.randn(1, anchors.shape[0], requires_grad=True)
    boxes = torch.rand(1, anchors.shape[0], 4) * 128
    # Make every decoded box valid for the synthetic assignment quality calculation.
    boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:].abs() + 1.0
    gt = torch.tensor([[[8., 8., 96., 96.]]])
    loss, parts = detector_loss(obj, boxes, anchors, gt, assignment='tal', box_loss_fn='ciou')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(obj.grad).all() and parts['n_pos'] > 0


def test_detector_loss_giou_path_is_byte_identical():
    torch.manual_seed(12)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 64
    gt = torch.tensor([[[10., 10., 40., 40.]]])
    a, ap = detector_loss(obj, boxes, anchors, gt)
    b, bp = detector_loss(obj, boxes, anchors, gt, box_loss_fn='giou')
    assert torch.equal(a, b) and ap == bp


def test_assign_ignores_nan_boxes():
    """An animal absent from a view gets a NaN box, and objectness must learn "nothing here"."""
    m = YOLOXNano(version='hybrid')
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[float('nan')] * 4])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == 0

    gt = torch.tensor([[float('nan')] * 4, [30.0, 30.0, 90.0, 90.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    assert (gix == 1).all(), 'positives must point at the finite box, not the NaN one'


def test_every_positive_anchor_is_inside_its_own_box():
    """A positive whose centre is outside its GT box has an unreachable regression target.
    `inside | near` shipped and left 71% of rat-city's positives unreachable."""
    m = YOLOXNano(version='hybrid')
    anchors = m.anchor_points(256, 256, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 60.0, 70.0],        # small: radius reaches well past it
                       [100.0, 30.0, 240.0, 200.0]])    # large: inside reaches past the radius
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    cx, cy = anchors[pos, 0], anchors[pos, 1]
    box = gt[gix]
    assert ((cx > box[:, 0]) & (cx < box[:, 2]) & (cy > box[:, 1]) & (cy < box[:, 3])).all(), \
        'a positive anchor outside its assigned box cannot reach the target it is given'
    assert len(set(gix.tolist())) == 2, 'both boxes must keep positives'


def test_assign_gives_each_anchor_one_box():
    m = YOLOXNano(version='hybrid')
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [30.0, 30.0, 110.0, 110.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == len(set(pos.tolist())), 'an anchor claimed by two boxes cancels'


def test_max_pos_per_gt_caps_a_large_box_and_leaves_a_small_one_alone():
    """T2.2: None is uncapped and unchanged; K caps a big box's candidacy without evicting a
    small box's own anchors, in `assign` and in the loss's `n_pos`."""
    anchors = YOLOXNano(version='hybrid').anchor_points(256, 256, torch.device('cpu'))
    for a, b in zip(assign(anchors, torch.tensor([[20.0, 20.0, 100.0, 100.0]])),
                    assign(anchors, torch.tensor([[20.0, 20.0, 100.0, 100.0]]),
                           max_pos_per_gt=None)):
        torch.testing.assert_close(a, b)

    whole = torch.tensor([[10.0, 10.0, 246.0, 246.0]])
    assert assign(anchors, whole)[0].numel() > 10
    pos, gix = assign(anchors, whole, max_pos_per_gt=10)
    assert pos.numel() <= 10 and bool((gix == 0).all())

    gt = torch.tensor([[10.0, 10.0, 130.0, 130.0], [200.0, 200.0, 216.0, 216.0]])
    small_u = int((assign(anchors, gt)[1] == 1).sum())
    assert 0 < small_u < 10, 'the small box needs an uncapped count under K'
    _, gix_c = assign(anchors, gt, max_pos_per_gt=10)
    assert int((gix_c == 1).sum()) == small_u and int((gix_c == 0).sum()) <= 10

    torch.manual_seed(8)
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 256
    _, up = detector_loss(obj, boxes, anchors, whole[None])
    _, cp = detector_loss(obj, boxes, anchors, whole[None], max_pos_per_gt=10)
    assert up['n_pos'] > 10 >= cp['n_pos']


def test_loss_is_finite_with_no_animal_anywhere():
    m = YOLOXNano(version='hybrid')
    x = torch.zeros(2, 3, 128, 128)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 128, x.device)
    gt = torch.full((2, 1, 4), float('nan'))
    loss, parts = detector_loss(obj, boxes, anchors, gt)
    assert torch.isfinite(loss) and parts['n_pos'] == 0


def test_detector_loss_drops_negatives_on_incomplete_frames_but_keeps_positives():
    anchors = torch.tensor([[5.0, 5.0, 8.0], [40.0, 40.0, 8.0]])
    obj = torch.zeros(2, 2, requires_grad=True)
    pred_boxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [35.0, 35.0, 45.0, 45.0]]] * 2)
    gt = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]] * 2)
    neg = torch.tensor([True, False])
    loss, parts = detector_loss(obj, pred_boxes, anchors, gt, negative_supervision=neg)
    loss.backward()
    assert parts['negative_frames'] == 0.5
    assert obj.grad[0, 1] != 0, 'a complete frame trains its background negative'
    assert obj.grad[1, 0] != 0, 'an incomplete frame still trains its positive'
    assert obj.grad[1, 1] == 0, 'an incomplete frame trains no background negative'
    same, _ = detector_loss(obj.detach(), pred_boxes, anchors, gt,
                            negative_supervision=torch.tensor([True, True]))
    base, _ = detector_loss(obj.detach(), pred_boxes, anchors, gt)
    assert float(same) == float(base)


def test_detector_loss_iou_aware_holds_hard_target_during_warmup():
    """`it < iou_aware_warmup` keeps the hard 1.0 target: an early near-zero IoU must not teach
    objectness to stay off."""
    torch.manual_seed(3)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5             # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    warm, wp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000, it=0)
    hard, hp = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(warm) == float(hard), 'during warmup the loss must equal the hard-target one'
    assert wp['iou_target'] == pytest.approx(1.0)


def test_detector_loss_iou_aware_switches_to_iou_target_after_warmup():
    """Past warmup (or `it=None`) the positive's target is the DETACHED IoU of its predicted and
    GT box, matching a direct `paired_iou`."""
    torch.manual_seed(4)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5              # bad boxes -> IoU well under 1
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    past, pp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000,
                             it=2000)
    hard, _ = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(past) != float(hard), 'past warmup the loss must differ from the hard-target one'
    pos, gix = assign(anchors, gt[0])
    assert pos.numel()
    want = float(paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1).mean())
    assert pp['iou_target'] == pytest.approx(want, abs=1e-5) and want < 1.0
    none_it, p1 = detector_loss(obj, boxes, anchors, gt, iou_aware=True, it=None)
    torch.testing.assert_close(none_it, past)
    assert p1['iou_target'] == pytest.approx(pp['iou_target'])


def test_iou_aware_weight_forcing_is_keyed_on_pos_mask_not_target_value():
    """The guard forcing a true positive's objectness weight back to 1 under
    `negative_supervision` must key on `pos_mask`, not the target value -- under `iou_aware`
    with bad boxes the target is near 0."""
    torch.manual_seed(6)
    anchors = YOLOXNano(version='hybrid').anchor_points(64, 64, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 5              # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]]])
    no_negatives = torch.tensor([False])                   # supervises NO negatives

    _, sp = detector_loss(obj, boxes, anchors, gt, negative_supervision=no_negatives,
                          iou_aware=True, iou_aware_warmup=0, it=1)
    assert sp['iou_target'] < 0.5, 'the test needs a genuinely low IoU to be a real check'
    # The OBJ term (not the box term, which iou_aware never touches) must be identical: the
    # positive's weight was forced to 1 in both cases, only the BCE TARGET value differs, and a
    # BCE(logit, 1.0) vs BCE(logit, iou) at weight=1 are different numbers -- so this checks obj
    # is computed off the same forced weight, not that the two total losses match.
    pos, gix = assign(anchors, gt[0])
    assert pos.numel()
    logit = obj[0, pos]
    target_hard = torch.ones_like(logit)
    target_soft = paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1)
    import torch.nn.functional as F
    norm = max(pos.numel(), 1)          # `detector_loss`'s own `obj_all / max(n_pos, B)`, B=1 here
    expected_hard_obj = F.binary_cross_entropy_with_logits(
        logit, target_hard, reduction='sum') / norm
    expected_soft_obj = F.binary_cross_entropy_with_logits(
        logit, target_soft, reduction='sum') / norm
    # Re-derived directly rather than relying on `hard`/`soft` above (which also include
    # box_weight * box, identical in both arms and therefore not informative here) -- read `obj`
    # off `parts` instead.
    _, hp = detector_loss(obj, boxes, anchors, gt, negative_supervision=no_negatives, iou_aware=False)
    assert hp['obj'] == pytest.approx(float(expected_hard_obj), rel=1e-4)
    assert sp['obj'] == pytest.approx(float(expected_soft_obj), rel=1e-4)
