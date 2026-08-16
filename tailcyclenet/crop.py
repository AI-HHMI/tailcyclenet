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

    ROTATING THESE FOUR CORNERS AND RE-BOUNDING THEM IS THE RIGHT RULE, AND THAT IS MEASURED
    RATHER THAN ASSUMED -- see `scratch/rat-city/check_rotation.py`. The objection is real: the
    axis-aligned hull of a rotated rectangle is up to sqrt(2) wider on the side, and where a
    stored box is a crop-rule box it is ALREADY padded and ALREADY squared, so re-bounding it
    rotates the pad and the squaring, neither of which is a property of the animal. The
    alternative that follows from that -- hold the side, move the centre, since a SQUARED extent
    is approximately rotation-invariant -- was built and REFUTED on rat-city-annotated: against
    truth (the crop rule on the same animal's rotated keypoints, over 548 instances x 36 angles)
    the corner rule reads |ratio - 1| median 0.084 and holding the side reads 0.111.

    The reason is the premise: 96% of that root's stored boxes are NOT square (aspect long/short
    median 1.737, p90 2.940). They are APT's tight animal boxes, not `crop_box_for_points` output --
    report 14 records that they round-trip exactly from the `.lbl` -- and
    for a tight elongated box the rotated corners track the rotated animal better than any
    rotation-invariant square can. Holding the side is UNBIASED there (median ratio 1.000 against
    the corner rule's 1.084) and more variable, which is the worse trade.

    So the sqrt(2) inflation is only wrong for a box that IS a squared crop extent -- tracked
    rat-city's, not this root's -- and nothing trains on those today. Do not "fix" this without
    re-running that script on the root you mean to fix.
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


def moving_boxes(extents, size, min_crop_dim=64, scale=1.0, shift=(0.0, 0.0)):
    """(T,4) int32 crops of ONE CONSTANT SIDE that FOLLOW the animal, or None.

    THE ONE MOVING-CROP RULE, shared by the training loader (`crop_to_points_2d_moving`) and
    inference (`infer.run_group`) for the same reason `crop_box_for_points` is shared: the pose
    model must meet the same geometry in both places, and two copies of this arithmetic would
    drift the way gotcha 8 describes.

    A window crop is a union over the window's frames, so it is inflated by however far the animal
    walked while the box stood still -- measured on rat-city, union side p50 1.23x and p90 1.92x
    the median per-frame side. These boxes translate instead.

    THE SIDE IS CONSTANT ACROSS THE WINDOW AND ONLY THE ORIGIN MOVES, which is what keeps this
    expressible at all: `apply_crop` folds the crop into the camera as a per-camera `offset` and
    `size`, one value for the whole window, so a per-frame SIZE would be a per-frame intrinsic and
    would have to be taught to `project_points_torch`, the triangulation and both camera
    embeddings. A per-frame ORIGIN is just a translation, and in 2D it is absorbed by shifting the
    coordinates per frame. This is also why the gain is bounded: sizing one box for the window's
    LARGEST frame leaves most of the union's inflation in place (measured 9% off the median side,
    not the ~20% the p50 ratio suggests).

    `extents` is one `[x1,y1,x2,y2]` per frame, or None where that frame had nothing finite. The
    centre is carried through the Nones by interpolation, so a frame the detector missed does not
    snap the crop to the image corner and take the animal out of shot.

    `scale` and `shift` are ONE crop jitter for the whole window -- see `jitter_params`. They must
    be one draw rather than one per frame: a per-frame scale is the per-frame size this rule exists
    to avoid, and a per-frame shift would add a random walk to the crop the labels do not follow.

    THE CENTRE IS WHAT GETS CLAMPED, not the corners. Clamping corners shrinks the box at the
    arena edge, which is a per-frame scale change by the back door.
    """
    T = len(extents)
    c = np.full((T, 2), np.nan, np.float64)
    sides = np.full(T, np.nan, np.float64)
    for t, e in enumerate(extents):
        if e is None:
            continue
        x0, y0, x1, y1 = (float(q) for q in e)
        c[t] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        sides[t] = max(x1 - x0, y1 - y0)
    ok = np.isfinite(c).all(1)
    if not ok.any():
        return None
    W, H = int(size[0]), int(size[1])
    side = max(float(np.nanmax(sides)) * float(scale), float(min_crop_dim))
    side = min(int(np.ceil(side)), W, H)
    t = np.arange(T)
    for i in range(2):
        c[:, i] = np.interp(t, t[ok], c[ok, i])
    cx = np.clip(c[:, 0] + side * float(shift[0]), side / 2.0, W - side / 2.0)
    cy = np.clip(c[:, 1] + side * float(shift[1]), side / 2.0, H - side / 2.0)
    x0 = np.clip(np.round(cx - side / 2.0), 0, W - side).astype(np.int32)
    y0 = np.clip(np.round(cy - side / 2.0), 0, H - side).astype(np.int32)
    return np.stack([x0, y0, x0 + side, y0 + side], 1).astype(np.int32)


def crop_to_points_2d_moving(cam, coords, min_crop_dim=64, jitter=None, crop_pts=None):
    """`crop_to_points_2d`'s moving twin. Returns (cam, boxes (T,4), coords shifted per frame).

    Same crop rule, same `_crop_source` fallback, same floor -- the box translates per frame
    instead of standing still, and the returned `cam` carries the CONSTANT side as its `size` and
    frame 0's origin as its `offset`. The per-frame origin is not in the camera and cannot be: it
    is applied to the coordinates here and to the pixels by `read_frames`.

    2D ONLY. In 3D the points are world-metric and `p2d` comes from `project_points_torch(cgroup,
    coords)`, so a per-frame origin would need a per-frame camera -- the change this rule is
    written to avoid.
    """
    src, pad = _crop_source(coords, crop_pts)
    per = [crop_box_for_points(src[t], cam['size'], min_crop_dim, pad) for t in range(len(src))]
    s, dx, dy = jitter if jitter is not None else (1.0, 0.0, 0.0)
    boxes = moving_boxes(per, cam['size'], min_crop_dim, scale=s, shift=(dx, dy))
    if boxes is None:
        return None, None, None
    origin = torch.as_tensor(boxes[:, :2].astype(np.float32))       # (T,2)
    # `coords` is (T,K,2); the origin broadcasts per frame, which is the whole 2D change.
    shifted = coords - origin[:, None, :].to(coords.dtype)
    return apply_crop(cam, boxes[0]), boxes, shifted


def static_offset(off):
    """A camera offset as a `(2,)`, collapsing the time axis a MOVING CROP puts there.

    Not every consumer of a camera is a per-frame one, and the ones that are not need a single
    origin. Two kinds, and they want this for different reasons:

    - OFFSET-INVARIANT quantities. `get_camera_scale` is a projection Jacobian and a constant
      image-plane translation has zero derivative, so collapsing is EXACT there, not an
      approximation.
    - PER-CAMERA DESCRIPTORS fed to the fusion, i.e. the principal-point term. Under a moving crop
      the principal point genuinely moves per frame; giving that term a time axis is a model change
      with its own consequences and is not what the moving crop is testing, so it takes the mean.

    Exact for a static offset, so nothing without a moving crop is affected.
    """
    return off if off.ndim == 1 else off.to(torch.float32).mean(dim=tuple(range(off.ndim - 1)))


def with_static_offset(cam):
    """`cam` with its offset collapsed by `static_offset`. See there for when that is legitimate."""
    return cam if cam['offset'].ndim == 1 else dict(cam, offset=static_offset(cam['offset']))


def apply_crop_moving(cam, boxes):
    """`apply_crop` for a crop that TRANSLATES: `offset` gains a time axis, `size` does not.

    This is the whole of the camera change a moving crop needs, in 2D and in 3D alike. `apply_crop`
    only ever wrote `offset` and `size`, and `moving_boxes` holds the side constant -- so `mat`,
    `ext` and `dist` are untouched and no per-frame INTRINSIC is involved. A `(T,2)` offset is
    exactly what `tailcyclenet.patches` teaches `project_cam` to right-align, and it is the same
    shape convention the library already uses for a moving camera's `(T,4,4)` `ext`.
    """
    out = dict(cam)
    off = torch.as_tensor(boxes[:, :2].astype(np.int32), dtype=cam['offset'].dtype)
    out['offset'] = cam['offset'][None, :] + off                    # (T,2)
    out['size'] = torch.tensor([int(boxes[0, 2] - boxes[0, 0]), int(boxes[0, 3] - boxes[0, 1])],
                               dtype=torch.int32)
    return out


def crop_to_points_3d_moving(cgroup, coords, min_crop_dim=64, jitter=None, crop_pts=None):
    """`crop_to_points_3d`'s moving twin. Returns (cgroup, boxes) with boxes[c] of shape (T,4).

    Same rule, same `_crop_source` fallback, same per-camera independence -- each camera follows
    the animal in ITS OWN view, which is the point: the union crop is per camera too, so an animal
    that stays put in one view and crosses another inflates only the second. Measured on 3dpop, the
    moving side is p50 0.918 and p10 0.584 of the union side, a LARGER inflation than any 2D root.

    In 3D the coordinates are world-metric and are NOT shifted here -- there is nothing to shift.
    The crop enters the geometry through the camera's `offset` alone, so `project_points_torch`
    lands points in moving-crop pixels by itself. That is why this needs `tailcyclenet.patches`
    and the 2D twin does not: 2D absorbs the origin into the coordinates, 3D cannot.
    """
    p2d = project_points_torch(cgroup, coords)
    out, boxes = [], []
    s, dx, dy = jitter if jitter is not None else (1.0, 0.0, 0.0)
    for cnum, cam in enumerate(cgroup):
        src, pad = _crop_source(p2d[cnum], None if crop_pts is None else crop_pts[cnum])
        per = [crop_box_for_points(src[t], cam['size'], min_crop_dim, pad)
               for t in range(len(src))]
        mb = moving_boxes(per, cam['size'], min_crop_dim, scale=s, shift=(dx, dy))
        if mb is None:
            return None, None
        out.append(apply_crop_moving(cam, mb))
        boxes.append(mb)
    return out, boxes


def jitter_params(rng, shift_frac, scale_frac):
    """The three numbers `jitter_box` draws, for a caller that must apply ONE draw to MANY boxes.

    Same draws in the same order as `jitter_box`, so a moving-crop item consumes the rng
    identically to a static one and the two arms stay comparable draw for draw.
    """
    return (1.0 + float(rng.uniform(-scale_frac, scale_frac)),
            float(rng.uniform(-shift_frac, shift_frac)),
            float(rng.uniform(-shift_frac, shift_frac)))


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
