"""THE crop rule.

The pose model is trained on crops produced by this rule, and the detector is trained to
reproduce *that box*. That is the whole reason a detector crop costs ~0.02 mm rather than
whatever an independently-plausible box rule would cost, so the two must never drift apart.

The arithmetic is lifted verbatim from posetail-pose's verified copy, which was itself checked
line by line against `PosetailDataset.crop_cgroup_to_points` (`posetail_dataset.py:1189-1236`):
same 20 px pad, same clamp to [0, size], same `base = max(min_crop_dim, w, h)`, same per-axis cap
at the image dimension, same centre-and-clamp when an axis is under `base`.
`tests/test_dataset.py` asserts it is int32-EXACT against what the library actually produces. If
that assertion ever fails, every detector number is invalid.

posetail exposes the rule only inline inside `crop_cgroup_to_points`, with no way to reach it for
a single camera -- hence a free function here rather than a call into the library.
"""
import numpy as np
import torch

from posetail.posetail.cube import project_points_torch

# What the crop rule bounds, shared by the pose loader, the detector and inference so the three
# cannot end up naming the same thing differently. See `crop_to_points_2d`'s `crop_pts`.
BOX_SOURCES = ('keypoints', 'instances')


def crop_box_for_points(p2d, size, min_crop_dim=64, pad=20):
    """Crop box around 2D points -> int32 [x1, y1, x2, y2], or None if nothing is finite.

    Args:
        p2d: (..., 2) pixel coordinates. Non-finite entries are dropped, which is the normal
            case -- allen pose is 81% finite and a point behind a camera projects to NaN.
        size: (2,) [W, H] of the image these coordinates live in.
        min_crop_dim: floor on the box side.
        pad: px added around the point extent. 20 is THE rule; `pad=0` exists so a caller
            holding an already-padded extent -- the two corners `instances.pq` stores for
            rat-city -- can re-enter the floor/clamp arithmetic below without padding twice.
            That is exact: min/max over the two corners is min/max over the points they came
            from, so the box is identical to the one the raw points would have produced.

    Returns None when no point is finite. The library's inline version has no such guard and
    raises on an all-NaN camera; the detector depends on getting None so it can emit a NaN box
    and make the objectness target "no animal here" instead of crashing the loader.
    """
    pflat = p2d.reshape(-1, 2)
    pflat = pflat[torch.all(torch.isfinite(pflat), dim=1)]
    if pflat.shape[0] == 0:
        return None

    size = size.to(torch.float32)
    zero = torch.zeros(2)
    low = torch.clamp(torch.min(pflat, dim=0).values - pad, zero, size).to(torch.int32)
    high = torch.clamp(torch.max(pflat, dim=0).values + pad, zero, size).to(torch.int32)

    current_width = high[0] - low[0]
    current_height = high[1] - low[1]

    # Each axis is capped at the image dimension so the crop never exceeds image bounds. Without
    # the cap, a wide bbox (700 px on a 540-tall image) forces min_dim = 700 > size[1] = 540,
    # making torch.clamp(x, 0, size[1] - min_dim) return a negative max and producing a negative
    # cam['offset'] that breaks project_cam.
    base = max(min_crop_dim, int(current_width), int(current_height))
    min_dim_x = min(base, int(size[0]))
    min_dim_y = min(base, int(size[1]))

    if current_width < min_dim_x:
        center_x = (low[0] + high[0]) // 2
        low[0] = torch.clamp(center_x - min_dim_x // 2, 0, int(size[0]) - min_dim_x)
        high[0] = low[0] + min_dim_x

    if current_height < min_dim_y:
        center_y = (low[1] + high[1]) // 2
        low[1] = torch.clamp(center_y - min_dim_y // 2, 0, int(size[1]) - min_dim_y)
        high[1] = low[1] + min_dim_y

    return torch.cat([low, high])


def box_corners(boxes):
    """(..., 4) xyxy -> (..., 4, 2), ALL FOUR corners. numpy in, numpy out; torch in, torch out.

    Two diagonal corners are not enough. An in-plane rotation turns the box into a rotated
    rectangle, and the extent of its two diagonal corners is strictly inside the extent of all
    four, so a two-corner version crops the animal. Shared by the loader (`dataset._crop_pts`,
    which rotates them) and by inference (`infer.run_group`, which unions them over a window), so
    the two cannot end up bounding different things.
    """
    lib = torch if torch.is_tensor(boxes) else np
    b = boxes
    return lib.stack([b[..., [0, 1]], b[..., [2, 1]], b[..., [2, 3]], b[..., [0, 3]]], -2)


def apply_crop(cam, box):
    """A copy of `cam` describing the cropped image: origin moves, size shrinks."""
    x1, y1, x2, y2 = box
    out = dict(cam)
    out['offset'] = cam['offset'] + torch.tensor([x1, y1], dtype=torch.int32)
    out['size'] = torch.tensor([x2 - x1, y2 - y1], dtype=torch.int32)
    return out


def _crop_source(coords, crop_pts):
    """(what to bound, what pad it needs). THE fallback rule for `crop_pts`, in one place.

    `crop_pts` is a stored `instances.pq` extent that is ALREADY padded, so it enters with pad 0.
    An all-NaN one means this view carries no stored box -- rat-city's boxes come from a tracker
    that loses animals, and a multi-root run mixes a root that has the table with roots that do
    not -- so the points are the only source left and the 20 px rule applies to them as usual.
    """
    if crop_pts is not None and torch.isfinite(crop_pts).any():
        return crop_pts, 0
    return coords, 20


def crop_to_points_3d(cgroup, coords, min_crop_dim=64, jitter=None, crop_pts=None):
    """Crop every camera to the projection of `coords`. Returns (cgroup, boxes).

    `crop_pts` is an optional (C, ..., 2) of pixel points to bound INSTEAD of the projection --
    the fallback is decided per camera, so one view with a stored box and one without is fine.
    Only what gets bounded changes; `coords` still drives `apply_crop` and everything downstream.
    """
    p2d = project_points_torch(cgroup, coords)
    out, boxes = [], []
    for cnum, cam in enumerate(cgroup):
        src, pad = _crop_source(p2d[cnum], None if crop_pts is None else crop_pts[cnum])
        box = crop_box_for_points(src, cam['size'], min_crop_dim, pad)
        if box is None:
            return None, None
        if jitter is not None:
            box = jitter(box, cam['size'])
        out.append(apply_crop(cam, box))
        boxes.append(box)
    return out, boxes


def crop_to_points_2d(cam, coords, min_crop_dim=64, jitter=None, crop_pts=None):
    """Single-camera pixel-space crop. Returns (cam, box, coords shifted into the crop)."""
    src, pad = _crop_source(coords, crop_pts)
    box = crop_box_for_points(src, cam['size'], min_crop_dim, pad)
    if box is None:
        return None, None, None
    if jitter is not None:
        box = jitter(box, cam['size'])
    shifted = coords - box[:2].to(coords.dtype)
    return apply_crop(cam, box), box, shifted


def jitter_box(rng, shift_frac, scale_frac):
    """A crop-box jitterer. 0.3/0.3 measurably beat a tight box in posetail-pose.

    It cut the cost of using detector crops instead of GT crops from +0.545 mm to +0.115 mm AND
    improved base accuracy -- a tight box teaches the model that the animal always fills the
    frame, which a detector cannot deliver.
    """
    def apply(box, size):
        box = box.to(torch.float32)
        w, h = box[2] - box[0], box[3] - box[1]
        s = 1.0 + float(rng.uniform(-scale_frac, scale_frac))
        cx = (box[0] + box[2]) / 2 + w * float(rng.uniform(-shift_frac, shift_frac))
        cy = (box[1] + box[3]) / 2 + h * float(rng.uniform(-shift_frac, shift_frac))
        nw, nh = w * s / 2, h * s / 2
        out = torch.tensor([cx - nw, cy - nh, cx + nw, cy + nh])
        out = torch.clamp(out, torch.zeros(4),
                          torch.cat([size, size]).to(torch.float32))
        out = out.to(torch.int32)
        # a degenerate jitter must not produce an empty crop
        if out[2] - out[0] < 8 or out[3] - out[1] < 8:
            return box.to(torch.int32)
        return out
    return apply
