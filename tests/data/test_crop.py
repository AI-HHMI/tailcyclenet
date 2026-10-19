"""THE crop rule's helpers: `crop_boxes_from_extents` (int32-exact against the rule) and `inflate_box`."""
from __future__ import annotations

import torch

from tailcyclenet.data.crop import crop_box_for_points, crop_boxes_from_extents


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
        for mcd, pad in ((64, 20), (8, 0), (200, 5)):
            n = 120
            x0 = torch.rand(n, generator=g) * 1.4 * W - 0.2 * W
            y0 = torch.rand(n, generator=g) * 1.4 * H - 0.2 * H
            w = torch.rand(n, generator=g) ** 3 * W
            h = torch.rand(n, generator=g) ** 3 * H
            boxes = torch.stack([x0, y0, x0 + w, y0 + h], -1)
            boxes[::17] = float('nan')
            got = crop_boxes_from_extents(boxes, size, mcd, pad)
            for i in range(n):
                if not torch.isfinite(boxes[i]).all():
                    assert torch.isnan(got[i]).all()
                    continue
                want = crop_box_for_points(boxes[i].view(2, 2), torch.tensor(size), mcd, pad)
                assert torch.equal(want.float(), got[i]), (size, mcd, pad, boxes[i])
