"""Cross-rig detector levers (report 73): input normalisation, photometric negative, compositing.

Each lever is a `[data]` key; at its off value it draws nothing.

- `normalize_input` -- a DETERMINISTIC per-image map applied to the model INPUT (after the
  letterbox/warp) at training AND deployment (`detect_raw`), recorded in the checkpoint like
  `antialias`. `equalize` = histogram equalisation of luma; `percentile` = a 0.5/99.5 stretch.
  The grey letterbox/warp border (exactly 114,114,114) is excluded from the statistics and kept
  at 114, so padding means the same thing on every rig.
- `invert` -- a photometric negative (`invert_prob`).
- `composite` -- copy-paste the item's own (already warped) animals onto a procedural
  `synthetic_canvas` with animal-sized dark distractors (`background_prob`). Decoy patches cut
  from the same frame's animal-free area are pasted too, so a pasted rectangle is not itself the
  cue. Synthetic canvases only: report 73 §9 measured the root's own estimated empty-arena
  backgrounds and a generic COCO image pack and neither beat synthetic, so both were deleted
  (with `scripts/estimate_backgrounds.py` / `build_generic_backgrounds.py`). `exposure` (a wide
  gain/gamma augmentation) was deleted too -- null on both transfer pairs (report 73 §5).
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
