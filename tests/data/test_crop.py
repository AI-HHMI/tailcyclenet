"""THE crop rule's helpers: `crop_boxes_from_extents` (int32-exact against the rule) and `inflate_box`."""
from __future__ import annotations

import torch

from tailcyclenet.data.crop import (crop_box_for_points, crop_boxes_from_extents, crop_rule,
                                    MIN_CROP_DIM, PAD_FRAC)


def test_inflate_box_widens_about_its_centre():
    """THE ONE INFLATION RULE, shared with the wide-crop training regime
    (`dataset.LoaderConfig.crop_inflate`) so the model trains and deploys on the same geometry."""
    from tailcyclenet.data.crop import inflate_box as _inflate_box
    box = torch.tensor([100, 100, 140, 160], dtype=torch.int32)     # 40x60, centre (120,130)
    size = torch.tensor([1000, 1000])
    out = _inflate_box(box, size, 1.5)
    assert int((out[0] + out[2]) / 2) == 120 and int((out[1] + out[3]) / 2) == 130
    assert int(out[2] - out[0]) == 60 and int(out[3] - out[1]) == 90       # 1.5x
    # clamps to the image rather than going negative
    clamped = _inflate_box(torch.tensor([0, 0, 40, 40], dtype=torch.int32), size, 4.0)
    assert int(clamped[0]) == 0 and int(clamped[1]) == 0


def test_crop_boxes_from_extents_is_the_crop_rule_int32_exact():
    """Vectorised over extents, it must return `crop_box_for_points` on the extent's corners,
    bit for bit -- including clamping, the floor, the square and the frame cap."""
    g = torch.Generator().manual_seed(0)
    for size in ((4696, 2048), (63, 50)):
        W, H = size
        for mcd, pad, frac in ((64, 20, 0.0), (8, 0, 0.0), (200, 5, 0.0),
                               (32, 0, 0.05), (32, 0, 0.5), (8, 3, 0.05)):
            n = 120
            x0 = torch.rand(n, generator=g) * 1.4 * W - 0.2 * W
            y0 = torch.rand(n, generator=g) * 1.4 * H - 0.2 * H
            w = torch.rand(n, generator=g) ** 3 * W
            h = torch.rand(n, generator=g) ** 3 * H
            boxes = torch.stack([x0, y0, x0 + w, y0 + h], -1)
            boxes[::17] = float('nan')
            got = crop_boxes_from_extents(boxes, size, mcd, pad, frac)
            for i in range(n):
                if not torch.isfinite(boxes[i]).all():
                    assert torch.isnan(got[i]).all()
                    continue
                want = crop_box_for_points(boxes[i].view(2, 2), torch.tensor(size), mcd, pad, frac)
                assert torch.equal(want.float(), got[i]), (size, mcd, pad, frac, boxes[i])


def test_the_rule_pads_by_a_fraction_of_the_longer_side():
    """THE rule pads every side by `crop_pad_frac` x the extent's LONGER side: a 30x20 px animal at
    0.05 gets 1.5 px (33x23, squared to 33, above the 32 floor); a 400x200 one gets 20 px."""
    size = torch.tensor([1000, 1000])
    small = torch.tensor([[100.0, 100.0], [130.0, 120.0]])
    box = crop_box_for_points(small, size, 32, 0.0, 0.05)
    assert int(box[2] - box[0]) == 33 and int(box[3] - box[1]) == 33
    assert int(box[0]) == 98                                            # floor(100 - 1.5)
    big = torch.tensor([[100.0, 100.0], [500.0, 300.0]])
    b = crop_box_for_points(big, size, 32, 0.0, 0.05)
    assert int(b[0]) == 80 and int(b[2]) == 520                         # 0.05 x 400
    # the primitive's defaults pad nothing: a caller outside the rule (box prompt, a stored extent)
    # is never padded by accident
    assert torch.equal(crop_box_for_points(small, size, 32), crop_box_for_points(small, size, 32, 0.0, 0.0))
    assert int(crop_box_for_points(small, size, 8)[0]) == 100


def test_the_rule_helpers_default_to_the_shipped_rule():
    """`crop_to_points_2d` (the rule, not the primitive) defaults to 5% / 32, as do the configs."""
    from tailcyclenet.data.crop import crop_to_points_2d
    from tailcyclenet.loader.config import LoaderConfig
    cam = {'size': torch.tensor([1000, 1000], dtype=torch.int32), 'offset': torch.zeros(2, dtype=torch.int32)}
    pts = torch.tensor([[100.0, 100.0], [500.0, 300.0]])
    _, box, _ = crop_to_points_2d(cam, pts)
    assert box.tolist() == [80, 0, 520, 440]                              # 20 px pad; 440 square, clamped
    assert crop_rule({}) == (PAD_FRAC, MIN_CROP_DIM) == (0.05, 32)
    assert crop_rule({'crop_pad_frac': 0.1, 'min_crop_dim': 64}) == (0.1, 64)
    cfg = LoaderConfig()
    assert (cfg.crop_pad_frac, cfg.min_crop_dim) == (0.05, 32)
