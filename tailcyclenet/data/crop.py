"""THE crop rule.

The pose model trains on crops produced by this rule and the detector is trained to reproduce
that box, so the two must never drift apart. The arithmetic is int32-exact against tracktail
0.3.5's public `crop_box_for_points`; the local copy adds `pad` (0 for an already-padded
stored extent) and returns None on an all-NaN input.

THE crop rule pads a keypoint extent by `crop_pad_frac` x its LONGER side on every side (shipped
0.05) and floors the square at `min_crop_dim` (shipped 32) -- owner decision 2026-10-29, replacing the
fixed 20 px pad. `pad` (px, additive) is kept on the primitive for callers outside the rule (the box
prompt's own occupancy box); the rule itself passes pad 0.
"""
import numpy as np
import torch

from tracktail.model.cube import project_points_torch

# What the crop rule bounds, shared by the pose loader, the detector and inference.
BOX_SOURCES = ('keypoints', 'instances')

# THE rule's two parameters (owner decision 2026-10-29). A run's config names them as
# `[data].crop_pad_frac` / `[data].min_crop_dim`; these are the shipped values.
PAD_FRAC = 0.05
MIN_CROP_DIM = 32


def crop_rule(data_cfg):
    """(pad_frac, min_crop_dim) a `[data]` block or checkpoint dict names (absent = the shipped rule)."""
    return (float(data_cfg.get('crop_pad_frac', PAD_FRAC)),
            int(data_cfg.get('min_crop_dim', MIN_CROP_DIM)))


def effective_pad(lo, hi, pad=0.0, pad_frac=0.0):
    """The pad one box gets: `pad` px plus `pad_frac` x the extent's longer side.

    `lo`/`hi` are the extent's min/max corners (torch, (..., 2)). With `pad_frac = 0` this returns
    `pad` itself, untouched, so a fixed-pixel caller's arithmetic is bit-identical.
    """
    if not pad_frac:
        return pad
    return (hi - lo).max(-1).values * float(pad_frac) + float(pad)


def crop_box_for_points(p2d, size, min_crop_dim=MIN_CROP_DIM, pad=0.0, pad_frac=0.0):
    """Crop box around 2D points -> int32 [x1, y1, x2, y2], or None if nothing is finite.

    Non-finite entries are dropped (a point behind a camera projects to NaN). Every side is padded
    by `pad` px + `pad_frac` x the points' longer extent (`effective_pad`); `pad = pad_frac = 0`
    lets a caller holding an already-padded stored extent re-enter the floor/clamp arithmetic
    without padding twice (and is the primitive's default: THE rule's caller passes its own
    `pad_frac`). None is how the detector says "no animal here".
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
    mn, mx = torch.min(pflat, dim=0).values, torch.max(pflat, dim=0).values
    pad = effective_pad(mn, mx, pad, pad_frac)
    low = torch.clamp(mn - pad, zero, size).to(torch.int32)
    high = torch.clamp(mx + pad, zero, size).to(torch.int32)

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


def crop_boxes_from_extents(boxes, size, min_crop_dim=MIN_CROP_DIM, pad=0.0, pad_frac=0.0):
    """(N,4) xyxy EXTENTS -> (N,4) crop-rule boxes, `crop_box_for_points` row by row, vectorised.

    The crop rule only reads the min/max of its points, so an extent's two corners give exactly
    the box its points would. That is what lets the detector regress the TIGHT extent and have
    the crop rule (pad, square, floor -- all in SOURCE pixels, which a letterboxed network cannot
    see) applied afterwards, here. int32-exact against `crop_box_for_points` (a test asserts it);
    a row with any non-finite value comes back NaN, as "no box here" does everywhere else.
    Returns float32 torch.
    """
    b = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    out = torch.full_like(b, float('nan'))
    ok = torch.isfinite(b).all(-1)
    if not ok.any():
        return out
    W, H = (float(v) for v in size)
    Wi, Hi = int(W), int(H)
    e = b[ok]
    mn, mx = torch.minimum(e[:, :2], e[:, 2:]), torch.maximum(e[:, :2], e[:, 2:])
    p = effective_pad(mn, mx, pad, pad_frac)
    p = p[:, None] if torch.is_tensor(p) else p
    lo, hi = mn - p, mx + p
    lim = torch.tensor([W, H])
    low = torch.minimum(torch.clamp(lo, min=0.0), lim).to(torch.int32)
    high = torch.minimum(torch.clamp(hi, min=0.0), lim).to(torch.int32)
    wh = high - low
    base = torch.clamp(wh.max(-1).values, min=int(min_crop_dim))
    for ax, lim_i in ((0, Wi), (1, Hi)):
        md = torch.clamp(base, max=lim_i)
        short = wh[:, ax] < md
        c = torch.div(low[:, ax] + high[:, ax], 2, rounding_mode='floor')
        start = torch.minimum(torch.clamp(c - torch.div(md, 2, rounding_mode='floor'), min=0),
                              lim_i - md)
        low[:, ax] = torch.where(short, start, low[:, ax])
        high[:, ax] = torch.where(short, start + md, high[:, ax])
    out[ok] = torch.cat([low, high], -1).to(torch.float32)
    return out


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


def _crop_source(coords, crop_pts, pad_frac=PAD_FRAC):
    """(what to bound, what pad_frac it needs). THE fallback rule for `crop_pts`, in one place.

    A stored `instances.pq` extent is already padded, so it enters with pad 0; an all-NaN one
    means this view carries no stored box, so the raw points apply the run's pad rule.
    """
    if crop_pts is not None and torch.isfinite(crop_pts).any():
        return crop_pts, 0.0
    return coords, pad_frac


def crop_to_points_3d(cgroup, coords, min_crop_dim=MIN_CROP_DIM, jitter=None, crop_pts=None,
                      inflate=1.0, pad_frac=PAD_FRAC):
    """Crop every camera to the projection of `coords`. Returns (cgroup, boxes).

    `crop_pts` bounds a stored box instead of the projection, decided per camera. `inflate`
    widens each box about its centre after jitter; 1.0 is an exact no-op. `pad_frac` is the
    run's pad rule for point-derived boxes (a stored extent always enters with pad 0).
    """
    p2d = project_points_torch(cgroup, coords)
    out, boxes = [], []
    for cnum, cam in enumerate(cgroup):
        src, f = _crop_source(p2d[cnum], None if crop_pts is None else crop_pts[cnum], pad_frac)
        box = crop_box_for_points(src, cam['size'], min_crop_dim, 0.0, f)
        if box is None:
            return None, None
        if jitter is not None:
            box = jitter(box, cam['size'])
        if inflate != 1.0:
            box = inflate_box(box, cam['size'], inflate)
        out.append(apply_crop(cam, box))
        boxes.append(box)
    return out, boxes


def crop_to_points_2d(cam, coords, min_crop_dim=MIN_CROP_DIM, jitter=None, crop_pts=None,
                      inflate=1.0, pad_frac=PAD_FRAC):
    """Single-camera pixel-space crop. Returns (cam, box, coords shifted into the crop).

    `inflate` widens the box about its centre after jitter; 1.0 is an exact no-op.
    """
    src, f = _crop_source(coords, crop_pts, pad_frac)
    box = crop_box_for_points(src, cam['size'], min_crop_dim, 0.0, f)
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
