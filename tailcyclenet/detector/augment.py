"""Detector augmentations and input normalisation: geometry, photometric suite, cutout, transfer levers.

- `random_affine` / `photometric` and the strong suite (`color_jitter`, `additive_noise`,
  `salt_pepper`, `motion_blur`, `grayscale`, `cutout_rects`) -- the in-domain recipe; their RNG
  draw order is part of the training stream.
- `normalize_input` -- a DETERMINISTIC per-image map on the model INPUT at training AND deployment
  (`detect_raw`), recorded in the checkpoint like `antialias`. The grey letterbox/warp border
  (exactly 114,114,114) is excluded from the statistics and kept at 114.
- `invert` (photometric negative) and `composite` (the item's animals pasted onto a procedural
  `synthetic_canvas` with decoys) -- the cross-rig levers (report 73).
"""
from __future__ import annotations

import numpy as np
import torch

PAD = 114
INPUT_NORMS = ('none', 'equalize', 'percentile')


def _content_mask(img):
    """True where a pixel is not the exact grey letterbox/warp border."""
    return ~((img[..., 0] == PAD) & (img[..., 1] == PAD) & (img[..., 2] == PAD))


def normalize_input(img, mode):
    """Deterministic per-image intensity normalisation of an RGB uint8 model input."""
    if mode in (None, '', 'none'):
        return img
    import cv2
    keep = _content_mask(img)
    if keep.sum() < 16:
        return img
    if mode == 'percentile':
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[keep]
        lo, hi = np.percentile(g, [0.5, 99.5])
        lut = np.clip((np.arange(256, dtype=np.float32) - lo) * 255.0 / max(hi - lo, 8.0), 0, 255)
        out = cv2.LUT(img, lut.astype(np.uint8))
    elif mode == 'equalize':
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y = ycc[..., 0]
        hist = np.bincount(y[keep].ravel(), minlength=256).astype(np.float64)
        cdf = np.cumsum(hist)
        cdf_min = cdf[np.flatnonzero(hist)[0]]
        lut = np.clip(np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1.0) * 255.0), 0, 255)
        ycc[..., 0] = lut.astype(np.uint8)[y]
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    else:
        raise ValueError(f'input_norm must be one of {INPUT_NORMS}, got {mode!r}')
    out[~keep] = PAD
    return out


def invert(img):
    """Photometric negative of the content, letterbox grey kept at 114.

    Coat colour and lighting flip contrast polarity across rigs (qdmouse: dark mouse, bright
    rim-lit tail, dark floor; allen: black mouse, dark tail, bright floor), so a detector that
    keys on the sign of an edge cannot transfer. A random negative makes the sign uninformative.
    """
    keep = _content_mask(img)
    out = 255 - img
    out[~keep] = PAD
    return out


def synthetic_canvas(wh, rng, blob_side=40.0):
    """A procedural canvas: gradient + band-limited noise + 'dead leaves' + dark blobs.

    The blobs are drawn at the animal's own scale because Wave 1's transfer FPs land on compact
    dark objects; a canvas that has them and no box teaches that dark-and-compact is not enough.
    """
    import cv2
    W, H = int(wh[0]), int(wh[1])
    base = rng.uniform(0, 255, 3)
    tilt = rng.uniform(-120, 120, (2, 3))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = (base[None, None] + tilt[0][None, None] * (xx / W)[..., None]
           + tilt[1][None, None] * (yy / H)[..., None])
    for _ in range(int(rng.integers(5, 40))):
        c = rng.uniform(0, 255, 3)
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        r = int(rng.uniform(4, 0.3 * max(W, H)))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c.tolist(), -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(rng.uniform(4, r))), c.tolist(), -1)
    sigma = rng.uniform(0.5, 6.0)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigma)
    img += (noise / max(noise.std(), 1e-6) * rng.uniform(0, 30))[..., None]
    for _ in range(int(rng.integers(0, 4))):
        side = blob_side * rng.uniform(0.5, 1.5)
        ax = (int(side / 2), int(side / 2 * rng.uniform(0.3, 1.0)))
        cv2.ellipse(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), ax,
                    float(rng.uniform(0, 180)), 0, 360, rng.uniform(0, 60, 3).tolist(), -1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
    return img


def _feather(h, w, frac=0.15):
    """(h, w, 1) alpha: 1 inside, a smooth ramp over `frac` of each side at the border."""
    ry = np.minimum(np.arange(h), np.arange(h)[::-1]) / max(1.0, frac * h)
    rx = np.minimum(np.arange(w), np.arange(w)[::-1]) / max(1.0, frac * w)
    a = np.clip(np.minimum(ry[:, None], rx[None, :]), 0, 1)
    return a[..., None].astype(np.float32)


def _place(canvas, patch, alpha, rng, taken, match_prob=0.7):
    """Paste `patch` at a random spot not overlapping `taken`; return (dx, dy) or None.

    With `match_prob` the patch is gain-matched so its BORDER mean equals the canvas region's
    mean -- otherwise a dark rig's animal carries a dark rectangle onto a bright canvas and the
    rectangle, not the animal, becomes the cue. Decoys go through the same call, so whatever
    edge a paste leaves is shared by animals and non-animals alike.
    """
    H, W = canvas.shape[:2]
    h, w = patch.shape[:2]
    if h > H or w > W:
        return None
    for _ in range(10):
        dx, dy = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        if all(dx + w <= a or a2 <= dx or dy + h <= b or b2 <= dy for a, b, a2, b2 in taken):
            region = canvas[dy:dy + h, dx:dx + w].astype(np.float32)
            if rng.random() < match_prob:
                ring = alpha[..., 0] < 0.999
                pm = float(patch[ring].mean()) if ring.any() else float(patch.mean())
                g = np.clip(float(region.mean()) / max(pm, 4.0), 0.25, 8.0)
                patch = np.clip(patch * g, 0, 255)
            canvas[dy:dy + h, dx:dx + w] = (alpha * patch + (1 - alpha) * region).astype(np.uint8)
            taken.append((dx, dy, dx + w, dy + h))
            return dx, dy
    return None


def composite(img, boxes, rng, wh, empty_prob=0.25, decoys=(0, 2), decoys_ok=True):
    """Paste `img`'s animals (and decoy patches) onto a synthetic canvas. Returns (img, boxes).

    `boxes` (N,4) in input px, NaN rows = no box. Each animal is cut with a quarter-side margin
    and feathered; its box is translated with it. With `empty_prob` no animal is pasted (a pure
    negative canvas). Decoys are same-sized patches from the frame's own area away from every
    box -- only when the frame's negatives are supervised (`decoys_ok`), otherwise a decoy could
    carry an unlabelled animal.
    """
    fin = torch.isfinite(boxes).all(-1)
    sides = (boxes[fin, 2:] - boxes[fin, :2]).max(-1).values if fin.any() else torch.tensor([40.])
    canvas = synthetic_canvas(wh, rng, blob_side=float(sides.median()))
    H, W = canvas.shape[:2]
    taken, out = [], []
    if fin.any() and rng.random() >= empty_prob:
        for b in boxes[fin]:
            m = 0.25 * float(max(b[2] - b[0], b[3] - b[1]))
            x0, y0 = int(max(0, b[0] - m)), int(max(0, b[1] - m))
            x1, y1 = int(min(img.shape[1], b[2] + m)), int(min(img.shape[0], b[3] + m))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            got = _place(canvas, img[y0:y1, x0:x1].astype(np.float32),
                         _feather(y1 - y0, x1 - x0), rng, taken)
            if got is not None:
                dx, dy = got
                out.append(b + torch.tensor([dx - x0, dy - y0, dx - x0, dy - y0], dtype=b.dtype))
    if decoys_ok:
        side = int(max(8, float(sides.median()) * 1.5))
        for _ in range(int(rng.integers(decoys[0], decoys[1] + 1))):
            if side >= img.shape[0] or side >= img.shape[1]:
                break
            for _try in range(10):
                x0 = int(rng.integers(0, img.shape[1] - side))
                y0 = int(rng.integers(0, img.shape[0] - side))
                clear = all(x0 + side <= float(b[0]) or float(b[2]) <= x0 or
                            y0 + side <= float(b[1]) or float(b[3]) <= y0 for b in boxes[fin])
                patch = img[y0:y0 + side, x0:x0 + side]
                if clear and _content_mask(patch).mean() > 0.9:
                    _place(canvas, patch.astype(np.float32), _feather(side, side), rng, taken)
                    break
    new = torch.stack(out) if out else torch.full((0, 4), float('nan'), dtype=boxes.dtype)
    return canvas, new


def photometric(img, rng):
    """The appearance half of `--augment`: one multiplicative gain, exactly as shipped.

    DLC's extended form (additive brightness + per-channel gaussian noise) was built, measured
    and REFUTED here: it costs accuracy in-domain, and neither shipped split tests the
    out-of-domain gain it buys.
    """
    out = img * rng.uniform(0.7, 1.3)
    return np.clip(out, 0, 255).astype(np.uint8)


# THE STRONG SUITE -- `--augment-strong`. Layered AFTER `photometric`, gated on
# `self.strong and self.augment and self.train`, off by default and off means none of these draws
# happen at all. Every op here is APPEARANCE-ONLY or, for cutout, an ERASURE -- neither moves a
# box target. Mirrors tracktail's own `aug_per_camera` / `aug_per_image` / `cutout_rects`.
def color_jitter(img, rng):
    """Gamma contrast, then HSV saturation and hue shifts -- tracktail's `aug_per_camera` triplet
    (DefocusBlur is folded into `motion_blur` below rather than duplicated).
    """
    import cv2
    gamma = rng.uniform(0.6, 1.8)
    lut = np.clip((np.arange(256, dtype=np.float64) / 255.0) ** gamma * 255.0, 0, 255)
    out = cv2.LUT(img, lut.astype(np.uint8))
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.uniform(-50, 30), 0, 255)
    hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-10, 10), 180)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def additive_noise(img, rng, sigma_max=10.2):
    """Per-channel gaussian noise, sigma ~ U(0, sigma_max). tracktail's `AdditiveGaussianNoise`
    (scale=(0, 0.04*255), i.e. sigma up to 10.2). Refuted alone on a detector with no capacity
    lever; re-tested here as part of the whole mix.
    """
    sigma = rng.uniform(0.0, sigma_max)
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def salt_pepper(img, rng, frac=0.004):
    """tracktail's `SaltAndPepper(0.004)`: a fraction of pixels forced to black or white."""
    out = img.copy()
    h, w = img.shape[:2]
    n = int(round(frac * h * w))
    if n <= 0:
        return out
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    salt = rng.random(n) < 0.5
    out[ys[salt], xs[salt]] = 255
    out[ys[~salt], xs[~salt]] = 0
    return out


def motion_blur(img, rng, k=(3, 5)):
    """Horizontal motion kernel, size drawn from `k`. tracktail's `MotionBlur(k=(3,5))`."""
    import cv2
    ksize = int(rng.choice(k))
    kernel = np.zeros((ksize, ksize), np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def grayscale(img):
    """RGB -> luminance replicated to 3 channels: an IR / monochrome rig, which no hue or
    saturation shift reaches. The pose loader draws the same thing (`grayscale_prob`)."""
    import cv2
    return np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)


def cutout_rects(wh, rng, n=(1, 3), frac=0.15):
    """1-3 rects of `frac` x `frac` of `wh` (INPUT pixels), random RGB fill.

    tracktail's `cutout_rects`: an erasure, not a geometric transform, so it never touches a box
    target -- only the pixels under it.
    """
    w, h = int(wh[0]), int(wh[1])
    rw, rh = max(1, int(w * frac)), max(1, int(h * frac))
    out = []
    for _ in range(int(rng.integers(n[0], n[1] + 1))):
        rx = int(rng.integers(0, max(w - rw, 1)))
        ry = int(rng.integers(0, max(h - rh, 1)))
        fill = tuple(int(v) for v in rng.integers(0, 256, 3))
        out.append((rx, ry, rx + rw, ry + rh, fill))
    return out


def apply_cutout(img, rects):
    """Fill each `(x0, y0, x1, y1, fill)` rect. Never touches a box target -- erasure only."""
    out = img.copy()
    for x0, y0, x1, y1, fill in rects:
        out[y0:y1, x0:x1] = fill
    return out


def random_affine(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5,
                  rotate_deg=0.0, centre=None, vflip=0.0):
    """A random similarity about `centre` (default the image centre), source px in and out, 2x3.

    Deliberately a similarity and not YOLOX's shear-and-perspective: the target is
    `crop_box_for_points`, an AXIS-ALIGNED extent, and a shear turns a box into a parallelogram
    whose extent is no longer the crop rule's box for anything.

    `rotate_deg` REPLACES THE FLIP and is strictly easier: a rotation moves every keypoint through
    the same affine and permutes nothing, where a mirror needs a `flip_pairs` map. Off by default
    here; the shipped detector config sets 180 (full circle) for cross-rig generalization, by owner
    decision 2026-10-02, although 180 was refuted IN-DOMAIN on two roots (report 17). A full circle
    costs no more retained area than a quarter one.

    `centre` IS WHY TILING AND ROTATION COMPOSE: the tile is cut AFTER this warp, so about the
    frame centre a scale of 0.8 moves an animal 2,000 px out by 400 -- more than a 640 px tile --
    and a rotation moves it out of the tile entirely. Passing the tile's own centre pins the tile's
    content and makes rotation free there; on whole frames the corners pull real pixels in at
    every angle.

    No `flip_pairs`. The detector emits one box, and a box is the extent of a SET of points, so
    relabelling left to right permutes the set and the extent is unchanged.

    `vflip` (probability of a vertical mirror) is box-only-safe for the same reason. It draws
    from `rng` ONLY when nonzero, so at 0 the draw sequence -- and every augmentation stream on
    record -- is unchanged.
    """
    w, h = float(size[0]), float(size[1])
    s = rng.uniform(*scale)
    sx = -s if rng.random() < hflip else s
    sy = (-s if rng.random() < vflip else s) if vflip else s
    cx, cy = (w / 2, h / 2) if centre is None else (float(centre[0]), float(centre[1]))
    A = np.array([[sx, 0.0], [0.0, sy]], np.float64)
    if rotate_deg:
        a = np.radians(rng.uniform(-rotate_deg, rotate_deg))
        A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ A
    t = (np.array([cx, cy]) - A @ np.array([cx, cy])
         + np.array([rng.uniform(-translate, translate) * w,
                     rng.uniform(-translate, translate) * h]))
    return np.concatenate([A, t[:, None]], 1).astype(np.float32)
