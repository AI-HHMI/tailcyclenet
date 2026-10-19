"""Letterbox / tile / warp geometry between a source frame and the detector input.

Every detector box comes back to source pixels through `unletterbox_boxes`, the one place it is
bounded to the frame.
"""
from __future__ import annotations

import numpy as np


def reduce_factor(size, out_wh):
    """The largest `cv2.IMREAD_REDUCED_COLOR_N` that still decodes above the letterbox target.

    libjpeg's DCT-domain decimation is a proper box filter AND halves the decode, where
    `cv2.resize`'s default INTER_LINEAR at this downscale is aliasing. N stays a power of two at
    or below 8 and never takes the decode below the letterbox target.
    """
    w, h = float(size[0]), float(size[1])
    n = 1
    while n < 8 and w / (2 * n) >= out_wh[0] and h / (2 * n) >= out_wh[1]:
        n *= 2
    return n


def letterbox(img, out_wh, src_wh=None, antialias=False):
    """Resize preserving aspect ratio, pad with grey. Returns (img, scale, (padx, pady)).

    `src_wh` is the size of the image BEFORE any decode-time reduction. The returned `scale` is
    then dst<-source rather than dst<-decoded, which is what keeps `unletterbox_boxes` and the box
    target correct without either of them knowing a reduction happened.

    `antialias=True` shrinks with `INTER_AREA` (a box filter) instead of `cv2.resize`'s default
    INTER_LINEAR, which at the 3-6x downscales of a 4K rig samples a 2x2 neighbourhood and aliases
    -- a per-rig texture signature a detector can learn. Upscaling stays linear. A checkpoint
    records which one it was trained under (`antialias`, absent = False) and deploys the same.
    """
    import cv2
    H, W = img.shape[:2]
    ow, oh = out_wh
    sw, sh = (W, H) if src_wh is None else (float(src_wh[0]), float(src_wh[1]))
    s = min(ow / sw, oh / sh)
    nw, nh = int(round(sw * s)), int(round(sh * s))
    if antialias and (nw < W or nh < H):
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(img, (nw, nh))
    canvas = np.full((oh, ow, 3), 114, np.uint8)
    px, py = (ow - nw) // 2, (oh - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, s, (px, py)


def letterbox_transform(size, out_wh):
    """The (scale, (padx, pady)) `letterbox` would return for an image of `size` = (w, h).

    Same arithmetic, no pixels: the box target and assignment diagnostics need the transform
    without paying the source decode to learn it.
    """
    w, h = float(size[0]), float(size[1])
    s = min(out_wh[0] / w, out_wh[1] / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    return s, ((out_wh[0] - nw) // 2, (out_wh[1] - nh) // 2)


def warp_image(img, M, out_wh, antialias=False):
    """`cv2.warpAffine(img, M, out_wh)` on the grey border, optionally anti-aliased.

    warpAffine has no area filter: it samples bilinearly however far it shrinks. With `antialias`
    the shrinking part of `M` (its smaller column norm, a similarity's scale) is done first by an
    INTER_AREA resize, and `M` is re-composed with the exact pixel-centre map of that resize, so
    the geometry is unchanged and only the filter differs. A non-shrinking `M` is untouched.
    """
    import cv2
    M = np.asarray(M, np.float64)
    if antialias:
        s = min(np.hypot(M[0, 0], M[1, 0]), np.hypot(M[0, 1], M[1, 1]))
        h, w = img.shape[:2]
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        if s < 1.0 and (nw < w or nh < h):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            fx, fy = w / nw, h / nh
            M = np.vstack([M, [0.0, 0.0, 1.0]]) @ np.array(
                [[fx, 0.0, 0.5 * fx - 0.5], [0.0, fy, 0.5 * fy - 0.5], [0.0, 0.0, 1.0]])
            M = M[:2]
    return cv2.warpAffine(img, M.astype(np.float32), (int(out_wh[0]), int(out_wh[1])),
                          borderValue=(114, 114, 114))


def tile_transform(origin, scale):
    """The `(scale, (padx, pady))` that renders a source-pixel tile at `origin` into the input.

    Deliberately the SAME two-number form `letterbox_transform` returns: a tile at source origin
    `(ox, oy)` rendered at scale `s` is exactly `(s, (-ox*s, -oy*s))`, so every geometry line here
    tiles by substituting this for the letterbox. No padding term: the tile's source extent is
    `input_wh / scale` by construction, so a letterbox would be a no-op.
    """
    s = float(scale)
    return s, (-float(origin[0]) * s, -float(origin[1]) * s)


def unletterbox_boxes(boxes, scale, pad, src_wh=None):
    """Detector-input boxes -> source-image boxes. With `src_wh`, clamped into the frame.

    THE CHOKE POINT EVERY DETECTOR BOX GOES THROUGH, which is why the bound is here and not in the
    four callers: a decoded side can be ~12,910 px, and IoU-only NMS cannot suppress a box that
    big (its IoU with the box it swallows is ~0). Downstream that box becomes a crop, so one such
    frame delivers the whole arena as a thumbnail. A box with no positive area after clamping
    comes back NaN, which every consumer already reads as "no box here". `src_wh` is optional
    only because `BoxDataset`'s training path has no frame to clamp against.
    """
    out = boxes.clone().float()
    out[:, 0::2] = (out[:, 0::2] - pad[0]) / scale
    out[:, 1::2] = (out[:, 1::2] - pad[1]) / scale
    if src_wh is not None:
        w, h = float(src_wh[0]), float(src_wh[1])
        out[:, 0::2] = out[:, 0::2].clamp(0.0, w)
        out[:, 1::2] = out[:, 1::2].clamp(0.0, h)
        dead = (out[:, 2] <= out[:, 0]) | (out[:, 3] <= out[:, 1])
        out[dead] = float('nan')
    return out
