"""The detector, and the property that makes it worth having.

The point of this detector is not that it finds animals -- it is that it reproduces THE CROP
RULE'S box. If it learned some other plausible box, every downstream pose number would shift.
"""
import numpy as np
import torch

from tailcyclenet.crop import crop_box_for_points
from tailcyclenet.detector import (BoxDataset, YOLOXNano, assign, box_collate, box_iou, decode,
                                   detector_loss, giou_loss, letterbox, unletterbox_boxes)


def test_forward_shapes_and_anchor_order():
    m = YOLOXNano()
    x = torch.zeros(2, 3, 128, 160)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0], \
        'anchor_points must match forward()s flattening order exactly'
    assert boxes.shape == (2, anchors.shape[0], 4)
    assert (boxes[..., 2] >= boxes[..., 0]).all() and (boxes[..., 3] >= boxes[..., 1]).all()


def test_giou_is_zero_for_a_perfect_box():
    b = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    assert float(giou_loss(b, b)) < 1e-5
    assert float(giou_loss(b, b + 200)) > 1.0        # disjoint -> worse than any overlap


def test_assign_ignores_nan_boxes():
    """An animal absent from a view gets a NaN box, and objectness must learn "nothing here"."""
    m = YOLOXNano()
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[float('nan')] * 4])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == 0

    gt = torch.tensor([[float('nan')] * 4, [30.0, 30.0, 90.0, 90.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    assert (gix == 1).all(), 'positives must point at the finite box, not the NaN one'


def test_assign_gives_each_anchor_one_box():
    m = YOLOXNano()
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [30.0, 30.0, 110.0, 110.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == len(set(pos.tolist())), 'an anchor claimed by two boxes cancels'


def test_loss_is_finite_with_no_animal_anywhere():
    m = YOLOXNano()
    x = torch.zeros(2, 3, 128, 128)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 128, x.device)
    gt = torch.full((2, 1, 4), float('nan'))
    loss, parts = detector_loss(obj, boxes, anchors, gt)
    assert torch.isfinite(loss) and parts['n_pos'] == 0


def test_decode_suppresses_duplicates():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.], [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2, 'the two overlapping boxes must collapse to one'
    assert s[0] > s[1]


def test_letterbox_round_trip():
    img = np.zeros((200, 400, 3), np.uint8)
    out, scale, pad = letterbox(img, (416, 416))
    assert out.shape == (416, 416, 3)
    box = torch.tensor([[10.0, 20.0, 300.0, 150.0]])
    moved = box.clone()
    moved[:, 0::2] = moved[:, 0::2] * scale + pad[0]
    moved[:, 1::2] = moved[:, 1::2] * scale + pad[1]
    torch.testing.assert_close(unletterbox_boxes(moved, scale, pad), box, atol=1e-4, rtol=0)


def test_targets_are_the_crop_rule(tiny_root):
    """The regression target must BE crop_box_for_points, not something similar to it."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2)
    x, boxes = ds[0]
    assert x.shape == (3, 128, 128)
    sess, gid, f, ci = ds.index[0]
    lab = sess.labels(gid)
    cam = sess.rig.posetail()[ci]
    for s in range(boxes.shape[0]):
        want = crop_box_for_points(
            torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32),
            cam['size'], ds.min_crop_dim)
        got = boxes[s]
        if want is None:
            assert torch.isnan(got).all()
            continue
        # the stored box is the crop rule's box, letterboxed; undo that and it must come back
        img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
        _, scale, pad = letterbox(img, ds.input_wh)
        back = unletterbox_boxes(got[None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_collate_pads_uneven_animal_counts():
    a = (torch.zeros(3, 8, 8), torch.zeros(2, 4))
    b = (torch.zeros(3, 8, 8), torch.zeros(5, 4))
    x, boxes = box_collate([a, b])
    assert x.shape == (2, 3, 8, 8) and boxes.shape == (2, 5, 4)
    assert torch.isnan(boxes[0, 2:]).all()
