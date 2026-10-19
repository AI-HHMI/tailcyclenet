"""Letterboxing: reduction, source-unit scale, unletterbox bounds, antialiasing."""
from __future__ import annotations

import numpy as np
import torch

from tailcyclenet.detector.letterbox import letterbox, unletterbox_boxes, warp_image


def test_reduce_factor_never_decodes_below_the_target():
    """A reduced decode must still downscale into the target; under tiling the comparison is
    against the deployed whole-frame size, where tile_scale 1.0 decodes natively."""
    from tailcyclenet.detector.letterbox import reduce_factor

    assert reduce_factor((4696, 2048), (640, 288)) == 4       # rat-city: 1174x512, still above
    assert reduce_factor((4696, 2048), (896, 384)) == 4       # 1174x512 still clears 896x384
    assert reduce_factor((1024, 570), (544, 320)) == 1        # calms21: 1/2 is already below
    assert reduce_factor((640, 480), (640, 480)) == 1
    for src, out in (((4696, 2048), (640, 288)), ((3840, 2160), (544, 320))):
        n = reduce_factor(src, out)
        assert src[0] / n >= out[0] and src[1] / n >= out[1], 'the remaining resize must be a '\
            'downscale, or the letterbox is upsampling a decimated frame'
    size = (4696, 2048)                                   # rat-city, tiled at tile_scale 1.0
    for tile in ((640, 640), (640, 288)):
        assert reduce_factor(size, tile) > 1, f'{tile}: the old comparison must decimate'
    assert reduce_factor(size, size) == 1, 'deployment decodes natively, so training must too'
    assert reduce_factor(size, (size[0] * 0.125, size[1] * 0.125)) > 1


def test_letterbox_scale_is_in_source_units_under_reduction():
    """`unletterbox_boxes` never sees the decode: a reduced decode that changed the returned
    scale would move every box by the reduction factor (4x on rat-city)."""
    full = np.zeros((2048, 4696, 3), np.uint8)
    quarter = np.zeros((512, 1174, 3), np.uint8)
    a, s_a, p_a = letterbox(full, (640, 288))
    b, s_b, p_b = letterbox(quarter, (640, 288), src_wh=(4696, 2048))
    assert a.shape == b.shape
    assert s_a == s_b and p_a == p_b
    box = torch.tensor([[100.0, 50.0, 700.0, 400.0]])
    moved = box.clone()
    moved[:, 0::2] = moved[:, 0::2] * s_b + p_b[0]
    moved[:, 1::2] = moved[:, 1::2] * s_b + p_b[1]
    torch.testing.assert_close(unletterbox_boxes(moved, s_b, p_b), box, atol=1e-3, rtol=0)


def test_unletterbox_clamps_a_runaway_box_into_the_frame():
    """A decoded side can reach ~12,910 px; IoU-only NMS cannot suppress it, so the clamp bounds
    it into the frame and a zero-area box comes back NaN.
    """
    from tailcyclenet.detector.letterbox import unletterbox_boxes

    b = torch.tensor([[-5000.0, -5000.0, 20000.0, 20000.0], [10.0, 10.0, 9.0, 40.0]])
    out = unletterbox_boxes(b, 1.0, (0, 0), src_wh=(640, 480))
    assert out[0].tolist() == [0.0, 0.0, 640.0, 480.0]
    assert torch.isnan(out[1]).all(), 'a box with no positive area is not a detection'
    # Without `src_wh` -- the training path, which has no frame to clamp against -- nothing changes.
    assert unletterbox_boxes(b, 1.0, (0, 0))[0, 2].item() == 20000.0


def _checker(n=256):
    """A 1-px checkerboard: the worst case for a bilinear shrink."""
    yy, xx = np.mgrid[:n, :n]
    img = (((yy + xx) % 2) * 255).astype(np.uint8)
    return np.repeat(img[..., None], 3, -1)


def test_antialias_changes_the_filter_not_the_geometry():
    """Off = `cv2.warpAffine` byte-identical; on = a 1-px checkerboard shrinks to flat grey (no
    aliasing), and a blob's centroid lands where the plain warp puts it; `letterbox` likewise."""
    import cv2

    M = np.array([[0.2, 0.0, 7.3], [0.0, 0.2, 3.1]], np.float32)
    img = _checker()
    plain = warp_image(img, M, (64, 64), antialias=False)
    assert np.array_equal(plain, cv2.warpAffine(img, M, (64, 64), borderValue=(114, 114, 114)))
    inner = (slice(10, 50), slice(10, 50))
    assert warp_image(img, M, (64, 64), antialias=True)[inner].std() < 2.0

    blob = np.full((400, 400, 3), 114, np.uint8)
    cv2.circle(blob, (230, 170), 40, (255, 255, 255), -1)

    def centroid(a):
        """Centre of the blob: weight = departure from the grey that is both border and field."""
        w = np.abs(a[..., 0].astype(np.float64) - 114)
        yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
        return np.array([(xx * w).sum(), (yy * w).sum()]) / w.sum()

    want = np.array([230.0, 170.0]) * 0.2 + [7.3, 3.1]
    np.testing.assert_allclose(centroid(warp_image(blob, M, (96, 96), antialias=True)), want,
                               atol=0.15)
    np.testing.assert_allclose(centroid(warp_image(blob, M, (96, 96), antialias=False)), want,
                               atol=0.15)

    # deployment (`detect_raw` -> `letterbox`): INTER_AREA under antialias, same scale/pad
    img = _checker(400)
    a, sa, pa = letterbox(img, (100, 80), antialias=True)
    b, sb, pb = letterbox(img, (100, 80))
    assert (sa, pa) == (sb, pb)
    want = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
    assert np.array_equal(a[:, pa[0]:pa[0] + 80], want)
    assert a[:, pa[0]:pa[0] + 80].std() < b[:, pb[0]:pb[0] + 80].std()
