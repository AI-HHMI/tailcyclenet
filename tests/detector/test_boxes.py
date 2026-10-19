"""The box arithmetic in `detector/boxes.py` is the one copy."""
import numpy as np
import pytest
import torch

from tailcyclenet.detector.boxes import box_iou, paired_iou


def test_box_arithmetic_has_one_copy():
    """`detector.boxes` is the one copy of centre/side/IoU arithmetic: numpy (`link_rows`) and
    torch (association/NMS) agree, and `paired_iou` is `box_iou`'s diagonal."""
    from tailcyclenet.detector.boxes import box_center_dist, centres, sides
    b = np.array([[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 7.0, 9.0]], np.float32)
    np.testing.assert_array_equal(centres(b), centres(torch.from_numpy(b)).numpy())
    np.testing.assert_array_equal(sides(b), sides(torch.from_numpy(b)).numpy())
    np.testing.assert_array_equal(centres(b), [[5.0, 10.0], [6.0, 7.0]])
    np.testing.assert_array_equal(sides(b), [15.0, 3.0])

    a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    c = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [5.0, 5.0, 15.0, 15.0]])
    torch.testing.assert_close(paired_iou(a, c), box_iou(a, c).diagonal())
    assert 0.0 < float(paired_iou(a, c)[1]) < 1.0
    # centre distance is in units of box side, so it is scale-free
    assert box_center_dist(a * 2, c * 2)[1, 1].item() == pytest.approx(
        box_center_dist(a, c)[1, 1].item())
