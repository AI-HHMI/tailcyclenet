"""THE crop rule.

The pose model trains on crops produced by this rule and the detector is trained to reproduce
that box, so the two must never drift apart. The arithmetic is int32-exact against posetail
0.3.5's public `crop_box_for_points`; the local copy adds `pad` (0 for an already-padded
stored extent) and returns None on an all-NaN input.
"""
import numpy as np
import torch

from posetail.posetail.cube import project_points_torch

# What the crop rule bounds, shared by the pose loader, the detector and inference.
BOX_SOURCES = ('keypoints', 'instances')


def crop_box_for_points(p2d, size, min_crop_dim=64, pad=20):
    """Crop box around 2D points -> int32 [x1, y1, x2, y2], or None if nothing is finite.

    Non-finite entries are dropped (a point behind a camera projects to NaN). `pad = 20` is THE
    rule; `pad = 0` lets a caller holding an already-padded stored extent re-enter the
    floor/clamp arithmetic without padding twice. None is how the detector says "no animal here".
    Each axis is capped at the image dimension so the crop never exceeds image bounds: without
    the cap, a wide bbox forces min_dim > size and a negative cam['offset'] that breaks
    project_cam.
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

    Two diagonal corners are not enough: under an in-plane rotation their extent is strictly
    inside the four-corner one, so a two-corner version crops the animal.
    """
    lib = torch if torch.is_tensor(boxes) else np
    b = boxes
    return lib.stack([b[..., [0, 1]], b[..., [2, 1]], b[..., [2, 3]], b[..., [0, 3]]], -2)


def inflate_box(box, size, factor):
    """Widen an xyxy crop box about its centre by `factor`, clamped to the image. Static (4,)
    boxes only. Returns int32, the same convention `crop_box_for_points` returns.

    The one inflation rule, shared by wide-crop training and `--crop-inflate` deployment: the
    model must train and deploy on the SAME geometry. `factor = 1.0` is an exact no-op.
    """
    if factor == 1.0:
        return box.to(torch.int32) if torch.is_tensor(box) else torch.as_tensor(box, dtype=torch.int32)
    b = box.to(torch.float32) if torch.is_tensor(box) else torch.as_tensor(box, dtype=torch.float32)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    hw, hh = (b[2] - b[0]) * factor / 2, (b[3] - b[1]) * factor / 2
    W, H = float(size[0]), float(size[1])
    out = torch.tensor([max(0.0, float(cx - hw)), max(0.0, float(cy - hh)),
                        min(W, float(cx + hw)), min(H, float(cy + hh))])
    return out.round().to(torch.int32)


def apply_crop(cam, box):
    """A copy of `cam` describing the cropped image: origin moves, size shrinks."""
    x1, y1, x2, y2 = box
    out = dict(cam)
    out['offset'] = cam['offset'] + torch.tensor([x1, y1], dtype=torch.int32)
    out['size'] = torch.tensor([x2 - x1, y2 - y1], dtype=torch.int32)
    return out


def _crop_source(coords, crop_pts):
    """(what to bound, what pad it needs). THE fallback rule for `crop_pts`, in one place.

    A stored `instances.pq` extent is already padded, so it enters with pad 0; an all-NaN one
    means this view carries no stored box, so the raw points apply the usual 20 px rule.
    """
    if crop_pts is not None and torch.isfinite(crop_pts).any():
        return crop_pts, 0
    return coords, 20


def crop_to_points_3d(cgroup, coords, min_crop_dim=64, jitter=None, crop_pts=None, inflate=1.0):
    """Crop every camera to the projection of `coords`. Returns (cgroup, boxes).

    `crop_pts` bounds a stored box instead of the projection, decided per camera. `inflate`
    widens each box about its centre after jitter; 1.0 is an exact no-op.
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
        if inflate != 1.0:
            box = inflate_box(box, cam['size'], inflate)
        out.append(apply_crop(cam, box))
        boxes.append(box)
    return out, boxes


def crop_to_points_2d(cam, coords, min_crop_dim=64, jitter=None, crop_pts=None, inflate=1.0):
    """Single-camera pixel-space crop. Returns (cam, box, coords shifted into the crop).

    `inflate` widens the box about its centre after jitter; 1.0 is an exact no-op.
    """
    src, pad = _crop_source(coords, crop_pts)
    box = crop_box_for_points(src, cam['size'], min_crop_dim, pad)
    if box is None:
        return None, None, None
    if jitter is not None:
        box = jitter(box, cam['size'])
    if inflate != 1.0:
        box = inflate_box(box, cam['size'], inflate)
    shifted = coords - box[:2].to(coords.dtype)
    return apply_crop(cam, box), box, shifted


def jitter_box(rng, shift_frac, scale_frac):
    """A crop-box jitterer. A loose box teaches the model the animal does not always fill the
    frame, which a detector cannot deliver.
    """
    def apply(box, size):
        """Jitter one box and clamp it to the image.

        Inputs: box -- xyxy crop box.
                size -- image (w, h).
        Outputs: the jittered int32 box, or the input when the result would be degenerate: a
        degenerate jitter must not produce an empty crop.
        """
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
        if out[2] - out[0] < 8 or out[3] - out[1] < 8:
            return box.to(torch.int32)
        return out
    return apply
