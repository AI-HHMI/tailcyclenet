"""Box training data: labels -> the crop rule's box, letterboxed into the detector's input.

The target is `tailcyclenet.crop.crop_box_for_points` applied to the same points the pose loader
crops on, so the detector reproduces the crop the pose model was trained on. An animal with no
finite point in a view gets a NaN box, not a dropped frame -- objectness still has to learn "no
animal here". `box_source='instances'` takes the extent from `instances.pq` instead, opt-in,
for a dataset whose stored keypoints are too sparse to bound the animal.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.posetail.cube import project_points_torch

from ..crop import BOX_SOURCES, crop_box_for_points
from ..dataset import _apply_affine, read_frames
from ..format import INST_PRESENT, PROJECTED, UNLABELED, VISIBLE, load_datasets

# The stem's `in_channels` for each loader mode. `'none'` (default) is 3 channels, no second
# frame. Kept as ONE dict so the training and eval scripts derive `in_channels` from the same
# rule instead of each hardcoding it and one of them drifting.
TEMPORAL_INPUT_CHANNELS = {'none': 3, 'stack2': 6}
TEMPORAL_INPUTS = tuple(TEMPORAL_INPUT_CHANNELS)
# The inverse: a loaded model only carries `in_channels`, so a caller that needs the matching
# `BoxDataset(temporal_input=...)` to score it reads it off this map.
TEMPORAL_INPUT_BY_CHANNELS = {v: k for k, v in TEMPORAL_INPUT_CHANNELS.items()}


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


def letterbox(img, out_wh, src_wh=None):
    """Resize preserving aspect ratio, pad with grey. Returns (img, scale, (padx, pady)).

    `src_wh` is the size of the image BEFORE any decode-time reduction. The returned `scale` is
    then dst<-source rather than dst<-decoded, which is what keeps `unletterbox_boxes` and the box
    target correct without either of them knowing a reduction happened.
    """
    import cv2
    H, W = img.shape[:2]
    ow, oh = out_wh
    sw, sh = (W, H) if src_wh is None else (float(src_wh[0]), float(src_wh[1]))
    s = min(ow / sw, oh / sh)
    nw, nh = int(round(sw * s)), int(round(sh * s))
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


def unletterbox_keypoints(kpts, scale, pad, src_wh=None):
    """Detector-input keypoints (N,K,3) -> source-image keypoints. Sibling of `unletterbox_boxes`.

    Lives HERE, beside the box version, because it is the one inverse every consumer goes through,
    and a keypoint un-letterboxed by a different rule than its own box is invisible in every
    downstream number. Out-of-frame keypoints go NaN rather than clamping: a box is clamped because
    a partly-visible animal still has a real extent, but a keypoint outside the frame was not
    observed there and clamping it to the border would invent a position.
    """
    out = kpts.clone().float()
    out[..., 0] = (out[..., 0] - pad[0]) / scale
    out[..., 1] = (out[..., 1] - pad[1]) / scale
    if src_wh is not None:
        w, h = float(src_wh[0]), float(src_wh[1])
        bad = ((out[..., 0] < 0) | (out[..., 0] > w) | (out[..., 1] < 0) | (out[..., 1] > h))
        out[..., :2] = torch.where(bad[..., None], torch.nan, out[..., :2])
    return out


def _photometric(img, rng, gain=None):
    """The appearance half of `--augment`: one multiplicative gain, exactly as shipped.

    DLC's extended form (additive brightness + per-channel gaussian noise) was built, measured
    and REFUTED here: it costs accuracy in-domain, and neither shipped split tests the
    out-of-domain gain it buys. `gain=None` draws its own; `gain=<value>` applies a PRE-DRAWN one
    so a stacked (t-1, t) pair sees the same appearance shift, not an independent one.
    """
    g = rng.uniform(0.7, 1.3) if gain is None else gain
    out = img * g
    return np.clip(out, 0, 255).astype(np.uint8)


# THE STRONG SUITE -- `--augment-strong`. Layered AFTER `_photometric`, gated on
# `self.strong and self.augment and self.train`, off by default and off means none of these draws
# happen at all. Every op here is APPEARANCE-ONLY or, for cutout, an ERASURE -- neither moves a
# box target. Mirrors posetail's own `aug_per_camera` / `aug_per_image` / `cutout_rects`.
def _color_jitter(img, rng):
    """Gamma contrast, then HSV saturation and hue shifts -- posetail's `aug_per_camera` triplet
    (DefocusBlur is folded into `_motion_blur` below rather than duplicated).
    """
    import cv2
    gamma = rng.uniform(0.6, 1.8)
    lut = np.clip((np.arange(256, dtype=np.float64) / 255.0) ** gamma * 255.0, 0, 255)
    out = cv2.LUT(img, lut.astype(np.uint8))
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.uniform(-50, 30), 0, 255)
    hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-10, 10), 180)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _additive_noise(img, rng, sigma_max=10.2):
    """Per-channel gaussian noise, sigma ~ U(0, sigma_max). posetail's `AdditiveGaussianNoise`
    (scale=(0, 0.04*255), i.e. sigma up to 10.2). Refuted alone on a detector with no capacity
    lever; re-tested here as part of the whole mix.
    """
    sigma = rng.uniform(0.0, sigma_max)
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _salt_pepper(img, rng, frac=0.004):
    """posetail's `SaltAndPepper(0.004)`: a fraction of pixels forced to black or white."""
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


def _motion_blur(img, rng, k=(3, 5)):
    """Horizontal motion kernel, size drawn from `k`. posetail's `MotionBlur(k=(3,5))`."""
    import cv2
    ksize = int(rng.choice(k))
    kernel = np.zeros((ksize, ksize), np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def _cutout_rects(wh, rng, n=(1, 3), frac=0.15):
    """1-3 rects of `frac` x `frac` of `wh` (INPUT pixels), random RGB fill.

    posetail's `cutout_rects`: an erasure, not a geometric transform, so it never touches a box
    target -- only the pixels under it, and (in `__getitem__`) the keypoints it covers.
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


def _apply_cutout(img, rects):
    """Fill each `(x0, y0, x1, y1, fill)` rect. Never touches a box target -- erasure only."""
    out = img.copy()
    for x0, y0, x1, y1, fill in rects:
        out[y0:y1, x0:x1] = fill
    return out


def _keypoints_in_rects(kpts_xy, rects):
    """(..., 2) input-pixel keypoints, `rects` from `_cutout_rects` -> (...) bool, covered or not.

    Half-open like the rects themselves, so a keypoint exactly on the far edge is not erased.
    """
    if not rects:
        return torch.zeros(kpts_xy.shape[:-1], dtype=torch.bool)
    x, y = kpts_xy[..., 0], kpts_xy[..., 1]
    mask = torch.zeros_like(x, dtype=torch.bool)
    for x0, y0, x1, y1, _ in rects:
        mask = mask | ((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
    return mask


def random_affine(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5,
                  rotate_deg=0.0, centre=None):
    """A random similarity about `centre` (default the image centre), source px in and out, 2x3.

    Deliberately a similarity and not YOLOX's shear-and-perspective: the target is
    `crop_box_for_points`, an AXIS-ALIGNED extent, and a shear turns a box into a parallelogram
    whose extent is no longer the crop rule's box for anything.

    `rotate_deg` REPLACES THE FLIP and is strictly easier: a rotation moves every keypoint through
    the same affine and permutes nothing, where a mirror needs a `flip_pairs` map. Default-off on
    evidence (its 180-degree setting is refuted on two roots), kept because a full circle costs no
    more retained area than a quarter one.

    `centre` IS WHY TILING AND ROTATION COMPOSE: the tile is cut AFTER this warp, so about the
    frame centre a scale of 0.8 moves an animal 2,000 px out by 400 -- more than a 640 px tile --
    and a rotation moves it out of the tile entirely. Passing the tile's own centre pins the tile's
    content and makes rotation free there; on whole frames the corners pull real pixels in at
    every angle.

    No `flip_pairs`. The detector emits one box, and a box is the extent of a SET of points, so
    relabelling left to right permutes the set and the extent is unchanged -- but only while
    keypoints are not a target, which is why `BoxDataset` passes `hflip=0` whenever it emits them.
    """
    w, h = float(size[0]), float(size[1])
    s = rng.uniform(*scale)
    sx = -s if rng.random() < hflip else s
    cx, cy = (w / 2, h / 2) if centre is None else (float(centre[0]), float(centre[1]))
    A = np.array([[sx, 0.0], [0.0, s]], np.float64)
    if rotate_deg:
        a = np.radians(rng.uniform(-rotate_deg, rotate_deg))
        A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ A
    t = (np.array([cx, cy]) - A @ np.array([cx, cy])
         + np.array([rng.uniform(-translate, translate) * w,
                     rng.uniform(-translate, translate) * h]))
    return np.concatenate([A, t[:, None]], 1).astype(np.float32)


def _drop_outside(x, bounds):
    """(...,2) points -> the same shape, non-finite where outside `bounds` = (lo_x,lo_y,hi_x,hi_y).

    A point warped off the region it belongs to is not that point any more: dropping it shrinks a
    box to its visible part, and `crop_box_for_points` returns None when every point of an animal
    is gone. Shared by `boxes_for` and `present_ignore_boxes_for` so there is one copy of this
    rule.
    """
    lo_x, lo_y, hi_x, hi_y = bounds
    out = ((x[..., 0] < lo_x) | (x[..., 0] > hi_x) |
          (x[..., 1] < lo_y) | (x[..., 1] > hi_y))
    return torch.where(out[..., None], torch.nan, x)


def _warp_region(rects, M):
    """(M,4) certified rects through an in-plane similarity, ROUNDING DOWN. Returns (M,4).

    A certified region is a CLAIM -- "everything in here is labelled" -- and a claim must shrink
    under a transform that makes it approximate, never grow: the axis-aligned hull of a rotated
    rectangle claims area the annotator never marked, re-admitting the unlabelled animals
    `regions.pq` exists to exclude. This inscribes instead, the same
    `_rotated_rect_max_inscribed` computation the rotated image canvas uses. Under every warp that
    existed before rotation the inscribed and circumscribed rects coincide, so this is a no-op
    against the old code.
    """
    A = np.asarray(M)[:, :2]
    # The rotation angle of a similarity, reflection-safe: normalise the columns first so a scale
    # or a flip cannot be read as an angle.
    ang = float(np.arctan2(A[1, 0], A[0, 0]))
    sin_a, cos_a = abs(np.sin(ang)), abs(np.cos(ang))
    x0, y0, x1, y1 = rects.unbind(-1)
    c = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2], -1)
    c = _apply_affine(c, (M, None))
    s = float(np.sqrt(abs(np.linalg.det(A))))
    w, h = (x1 - x0) * s, (y1 - y0) * s
    if sin_a > 1e-9:
        # Largest axis-aligned rect inside a w x h rect rotated by `ang`, the exact same branch
        # structure as `_rotated_rect_max_inscribed` -- written on tensors so a whole (M,4) goes
        # through at once.
        long_, short = torch.maximum(w, h), torch.minimum(w, h)
        degen = short <= 2 * sin_a * cos_a * long_
        if abs(sin_a - cos_a) < 1e-10:      # exactly 45 degrees: the normal branch is singular
            degen = torch.ones_like(degen)
        x, wide = 0.5 * short, w >= h
        cw_d = torch.where(wide, x / max(sin_a, 1e-12), x / max(cos_a, 1e-12))
        ch_d = torch.where(wide, x / max(cos_a, 1e-12), x / max(sin_a, 1e-12))
        denom = cos_a * cos_a - sin_a * sin_a
        denom = denom if abs(denom) > 1e-12 else 1e-12
        cw_n = (w * cos_a - h * sin_a) / denom
        ch_n = (h * cos_a - w * sin_a) / denom
        w = torch.clamp(torch.where(degen, cw_d, cw_n), min=0.0)
        h = torch.clamp(torch.where(degen, ch_d, ch_n), min=0.0)
    half = torch.stack([w, h], -1) / 2
    return torch.cat([c - half, c + half], -1)


class BoxDataset(Dataset):
    """One item = one camera view of one frame, with every animal's crop box in it.

    Deliberately per-view and per-frame rather than per-window: a box predictor has no temporal
    model, and giving it 24 near-identical frames as one item would just correlate the batch.
    """

    def __init__(self, path, split: str, input_wh=(416, 416), min_crop_dim=64,
                 max_frames_per_group: int = 40, seed: int = 23, box_source='keypoints',
                 augment=False, reduce=False, keypoints=False, hflip=None, rotate_deg=0.0,
                 tile_wh=None, tile_scale=1.0, tile_bg_per_frame=1, use_regions=False,
                 strong=False, ignore_present=False, temporal_input='none'):
        assert box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {box_source!r}'
        self.box_source = box_source
        # TILING, off by default and off means the whole frame -- with `tile_wh=None` every
        # geometry path takes the letterbox branch and this loader is byte-identical to what every
        # recorded detector was trained on.
        #
        # `tile_wh` is the tile in INPUT pixels, i.e. the model's input size, so it REPLACES
        # `input_wh`. `tile_scale` is the source -> input scale and is the only scale there is:
        # the tile's source extent is `tile_wh / tile_scale`. DO NOT ADD A SECOND RESIZE --
        # tiling and then downscaling the tiles breaks the invariant the scheme rests on, the
        # animal's size in INPUT pixels at train == deployment.
        self.tile_wh = None if tile_wh is None else tuple(int(v) for v in tile_wh)
        self.tile_scale = float(tile_scale)
        self.tile_bg_per_frame = int(tile_bg_per_frame)
        if self.tile_wh is not None and self.tile_scale <= 0:
            raise ValueError(f'tile_scale must be > 0, got {tile_scale}')
        # Mask the objectness loss to the area an annotator certified as completely labelled
        # (`regions.pq`). ORTHOGONAL to tiling and off by default, so an arm can move one lever at
        # a time -- but MEASURED DEAD on full-frame input (a labelled rat-city frame carries ~69%
        # certified positives against 0.68% unmasked), where tiling is what fixes it. The rate is
        # set by how many stride-8 CELLS the certified region spans (`CENTER_RADIUS` is 2.5 cells),
        # so it is a resolution knob and `tile_scale` is what sets it.
        self.use_regions = bool(use_regions)
        # T2.1: mask an `instances.pq` PRESENT box -- an animal in this view that was NOT annotated
        # -- out of the objectness BACKGROUND target, instead of training the detector to call it
        # "no animal here". Off by default, so with it absent this loader is byte-identical to
        # every recorded detector.
        #
        # MUTUALLY EXCLUSIVE with `use_regions`: both are the opt-in 4th tuple element
        # `box_collate`/`split_batch` dispatch by RANK alone (M,4), and that dispatch cannot tell
        # two rank-2 tensors apart. Raised at construction rather than letting one clobber the
        # other.
        self.ignore_present = bool(ignore_present)
        if self.ignore_present and self.use_regions:
            raise ValueError('ignore_present and use_regions cannot both be set: both are the '
                             'one opt-in (M,4) tuple slot box_collate/split_batch dispatch by '
                             'rank, and that dispatch cannot tell two rank-2 tensors apart. See '
                             'BoxDataset.__init__.')
        # Opt-in, default off: with it absent this loader is byte-identical to what every recorded
        # detector was trained on. See `random_affine` for why it also kills the horizontal flip.
        self.keypoints = bool(keypoints)
        # `hflip=None` means "decide from `keypoints`", the safe default. Separable only so a
        # box-only CONTROL arm can match the keypoint arm's augmentation exactly: the keypoint arm
        # loses the flip, so a control that keeps it differs in two levers and measures neither.
        self.hflip = (0.0 if self.keypoints else 0.5) if hflip is None else float(hflip)
        # 0 is off and off is byte-identical: `random_affine` skips the draw.
        self.rotate_deg = float(rotate_deg)
        # The APPEARANCE half of --augment. False is the shipped single gain and is byte-identical;
        # True adds nothing extra by itself -- `--augment-strong` (below) layers the rest of the
        # mix on top. Requested explicitly, not inferred from the split: `self.train` still gates
        # it, so a val or test loader built with `augment=True` blindly stays deterministic.
        self.augment = augment
        # THE STRONG SUITE. Off by default and off means byte-identical to every recorded
        # detector -- every new draw below is gated on `self.strong and self.augment and
        # self.train`. It is a SINGLE recipe (color jitter, additive noise, salt & pepper, motion
        # blur, cutout, mosaic-lite), not five independently-swept levers.
        self.strong = bool(strong)
        if self.strong and self.use_regions:
            # FAIL AT CONSTRUCTION, not on the ~20% of items that happen to draw mosaic-lite: a
            # composite frame has no expressible certified-area mask.
            raise ValueError('--augment-strong (mosaic-lite) is undefined under --use-regions')
        # T4.2: stack frame t-1 beside frame t along the channel axis, so the detector has a signal
        # a single-frame net categorically cannot -- motion is what separates an animal from arena
        # clutter. `'none'` (default) is byte-identical to every checkpoint on record. See
        # `TEMPORAL_INPUT_CHANNELS` for the loader-mode -> channel-count map.
        if temporal_input not in TEMPORAL_INPUTS:
            raise ValueError(f'temporal_input must be one of {TEMPORAL_INPUTS}, got '
                             f'{temporal_input!r}')
        self.temporal_input = str(temporal_input)
        if self.temporal_input != 'none' and self.strong:
            # FAIL AT CONSTRUCTION: mosaic-lite pastes another item's OWN frame-t crop into this
            # item's composite, and doing that under a temporal input needs the source item's t-1
            # frame too, at the same pasted location -- not built yet.
            raise ValueError('temporal_input is undefined under --augment-strong (mosaic-lite): '
                             'the pasted source item needs its own t-1 frame too, not built yet')
        # Off by default, and a KEY rather than a loader detail: it changes which source pixels
        # reach the model, so every arm measured without it stays comparable only against other
        # arms without it. It rides in the checkpoint and `detect_group` reads it back, because a
        # detector fed differently-sampled pixels at deployment is off its own training
        # distribution with nothing in the output to say so.
        self.reduce = reduce
        self.seed = seed
        self.datasets = load_datasets(path)
        if len(self.datasets) != 1:
            raise ValueError(
                f'{path}: the detector is trained per dataset (input size and box statistics are '
                f'dataset-specific); found {len(self.datasets)} dataset roots')
        self.ds = self.datasets[0]
        # The tile IS the input when tiling; `input_wh` is then the tile size and the deployment
        # input size is a DIFFERENT number (`frame_wh * tile_scale`), recorded in the checkpoint.
        self.input_wh = tuple(input_wh) if self.tile_wh is None else self.tile_wh
        self.min_crop_dim = min_crop_dim
        self.train = split == 'train'
        rng = np.random.default_rng(seed)

        # Parallel to `self.index`, one entry per item: the tile's source-pixel origin, or None
        # for the whole frame. A parallel list rather than a fifth tuple element because `index`
        # is unpacked as a 4-tuple in `evaluate.py` too, and widening it would break those
        # silently at the call site rather than loudly here.
        self.origins: list = []
        self.index = []
        for sess in self.ds.sessions.get(split, []):
            sess.preload()
            for gid, group in sess.groups.items():
                lab = sess.labels(gid)
                vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
                if vis is None:
                    continue
                v = vis.reshape(vis.shape[0], vis.shape[1], -1)
                frames = np.flatnonzero((v != UNLABELED).any((0, 2)))
                if frames.size > max_frames_per_group:
                    frames = rng.choice(frames, max_frames_per_group, replace=False)
                for f in sorted(frames):
                    for ci in range(len(sess.rig)):
                        origins = ([None] if self.tile_wh is None
                                   else self._tile_origins(sess, gid, int(f), ci, rng))
                        for o in origins:
                            self.index.append((sess, gid, int(f), ci))
                            self.origins.append(o)
        if not self.index:
            raise ValueError(f'{path}: split {split!r} has no labelled frames')
        # ONE CONTAINER'S WORTH OF INDEX POSITIONS, which is what `ChunkShuffle` needs a block to
        # be. Its old hardcoded 512 spanned 13 videos on a session with 40 positions per group,
        # which is not locality at all -- the reader cache thrashed and OOM-killed the workers.
        n_src = len({(s.session_id, g, c) for s, g, _, c in self.index})   # (group, camera) = file
        self.chunk = max(1, len(self.index) // n_src)

    def __len__(self):
        return len(self.index)

    def cohort_weights(self, annot_frac):
        """Per-entry sampling weights giving `annotated` sessions probability `annot_frac`.

        The detector analogue of `PoseDataset._pool_weights`, and it exists for the same reason:
        WITHOUT it the cohort mix is whatever `frames_per_group` happens to produce. On
        rat-city-combined the cap leaves the one tracked session (57,594 labelled frames,
        truncated to 40) at 4.3% of train views against 37 annotated sessions' 95.7% -- a ratio
        nothing in the config names and nobody chose. `annot_frac` names it.

        Uniform WITHIN a cohort, so a group's weight does not scale with how many frames survived
        the cap; the two cohort totals are then set to `annot_frac` / `1 - annot_frac`. Returns
        None when the question does not arise -- `annot_frac` unset, or the split holds one
        cohort (3dpop, calms21 and branson-fly are entirely `tracked`) -- and None means the
        caller uses `ChunkShuffle`, byte-identical to every detector on record.
        """
        if annot_frac is None:
            return None
        if not 0.0 <= float(annot_frac) <= 1.0:
            raise ValueError(f'annot_frac must be in [0, 1], got {annot_frac}')
        src = np.array([s.label_source for s, _, _, _ in self.index])
        present = [c for c in ('annotated', 'tracked') if (src == c).any()]
        if len(present) < 2:
            return None
        w = np.zeros(len(self.index), dtype=np.float64)
        for cohort in present:
            m = src == cohort
            p = float(annot_frac) if cohort == 'annotated' else 1.0 - float(annot_frac)
            w[m] = p / int(m.sum())
        if w.sum() <= 0:
            raise ValueError('sampling weights are all zero -- check annot_frac')
        return w

    def cohort_mix(self, weights=None):
        """Realised share of train draws per cohort. Reporting only -- the mix is invisible in
        the loss curve, so `train_detector.py` prints it, exactly as `PoseDataset.mix` is.
        """
        src = np.array([s.label_source for s, _, _, _ in self.index])
        w = (np.full(len(self.index), 1.0 / max(len(self.index), 1)) if weights is None
             else np.asarray(weights, dtype=np.float64) / np.sum(weights))
        return {c: float(w[src == c].sum()) for c in sorted(set(src.tolist()))}

    # ------------------------------------------------------------------------------------------
    # tiling
    # ------------------------------------------------------------------------------------------

    def _tile_extent(self):
        """The tile's extent in SOURCE pixels. `input_wh / tile_scale`, and nothing else."""
        return (self.tile_wh[0] / self.tile_scale, self.tile_wh[1] / self.tile_scale)

    def _warp_centre(self, i):
        """What `random_affine` turns and scales about: the TILE's centre, or None for the frame.

        `__getitem__` composes `tile @ warp @ decode`, so the tile is cut after the warp; about
        the frame centre a scale of 0.8 moves an animal 400 px out -- more than a 640 px tile --
        so a tile chosen for a region or an animal would come back holding something else.
        """
        if self.origins[i] is None:
            return None
        ox, oy = self.origins[i]
        tw, th = self._tile_extent()
        return (ox + tw / 2, oy + th / 2)

    def _region_rects(self, sess, gid, f, ci):
        """(M,4) certified rects in SOURCE px for one (frame, camera), or None if no regions.pq.

        `None` and an empty `(0,4)` are DIFFERENT ANSWERS and both are returned faithfully: None is
        "the session carries no regions.pq, so it claims exhaustive labelling and every pixel is
        certified", `(0,4)` is "the file exists and certifies nothing here".
        """
        r = sess.labels(gid).regions
        if r is None:
            return None
        if not len(r):
            return np.zeros((0, 4), np.float64)
        sel = (r[:, 0].astype(int) == f) & (r[:, 1].astype(int) == ci)
        return np.asarray(r[sel][:, 2:], np.float64)

    def _tile_origins(self, sess, gid, f, ci, rng):
        """Source-pixel tile origins for one (frame, camera).

        One tile per certified region, one per animal, plus `tile_bg_per_frame` background tiles.
        Origins are JITTERED by up to a quarter of the tile: a detector trained only on animals at
        dead centre has never seen one near a border, and `assign`'s centre prior would then be
        learned as a centre-of-image prior.

        Where background tiles may be drawn from is exactly the None/empty distinction: `regions
        is None` (exhaustively labelled) allows anywhere in the frame; non-empty restricts to
        inside a certified rect (elsewhere is UNKNOWN, not background); empty `(0,4)` means no
        certified background exists and none is drawn.
        """
        W, H = (float(v) for v in sess.rig.size(sess.cam_names[ci]))
        tw, th = self._tile_extent()
        regions = self._region_rects(sess, gid, f, ci)
        out = []

        def push(cx, cy):
            jx, jy = rng.uniform(-0.25, 0.25, 2) * np.array([tw, th])
            # Clamped so a tile always overlaps the frame, but NOT forced inside it: a tile at the
            # frame edge is a real thing to train on and `warpAffine`'s grey border is what a
            # detector sees there.
            out.append((float(np.clip(cx - tw / 2 + jx, -tw / 4, max(0.0, W - 3 * tw / 4))),
                        float(np.clip(cy - th / 2 + jy, -th / 4, max(0.0, H - 3 * th / 4)))))

        if regions is not None:
            for r in regions.reshape(-1, 4):
                push((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        for c in self._animal_centres(sess, gid, f, ci):
            push(c[0], c[1])
        for _ in range(self.tile_bg_per_frame):
            if regions is None:
                push(rng.uniform(0, W), rng.uniform(0, H))
            elif len(regions):
                r = regions[rng.integers(len(regions))]
                push(rng.uniform(r[0], r[2]), rng.uniform(r[1], r[3]))
        # A frame with no animal, no region and no certified background still yields ONE tile:
        # objectness has to learn "nothing here", and dropping the frame would train a detector
        # that has never seen an empty image.
        return out or [(max(0.0, (W - tw) / 2), max(0.0, (H - th) / 2))]

    def _animal_centres(self, sess, gid, f, ci):
        """(n,2) source-pixel centroids of the animals labelled in this view.

        The centroid, not the crop box: an origin only needs to be approximately on an animal, and
        running the crop rule for every animal of every frame at index-build time would pay for
        precision nothing here uses.
        """
        p2d = self._points_2d(sess, gid, f, ci).numpy()
        out = []
        for s in range(p2d.shape[0]):
            ok = np.isfinite(p2d[s]).all(-1)
            if ok.any():
                out.append(p2d[s][ok].mean(0))
        return out

    def _points_2d(self, sess, gid, f, ci):
        """(S,K,2) source-pixel points for one (frame, camera). 3D sessions project.

        Shared by `boxes_for` and the tile-origin sampler so there is one copy of the
        "2D reads the table, 3D projects" branch.
        """
        lab = sess.labels(gid)
        if sess.mode == '3d':
            # Frame-indexed, not whole-group: axis -3 of `pts` is the ANIMAL, so a moving camera's
            # (T,4,4) extrinsic would project animal `i` through frame `i`'s pose.
            cam = sess.cgroup(gid, f)[ci]
            pts = torch.as_tensor(lab.points3d[:, f], dtype=torch.float32)
            return project_points_torch([cam], pts)[0]
        return torch.as_tensor(lab.points2d[:, f, :, ci], dtype=torch.float32)

    def _transform(self, i, size):
        """The `(scale, (padx, pady))` for item `i`: its tile's, or the whole-frame letterbox.

        THE one place that choice is made; `boxes_for` and `regions_for` both come here, because a
        region transformed by a different rule than its own boxes is invisible in a loss curve.
        """
        if self.origins[i] is None:
            return letterbox_transform(size, self.input_wh)
        return tile_transform(self.origins[i], self.tile_scale)

    def regions_for(self, i, warp=None):
        """Certified rectangles for item `i` in INPUT pixels, `(M,4)`, or None.

        None means the session carries no `regions.pq` and therefore claims to be exhaustively
        labelled -- every anchor is supervised. An empty `(0,4)` means the file exists and this
        view certifies nothing. See `_region_rects`.
        """
        sess, gid, f, ci = self.index[i]
        rects = self._region_rects(sess, gid, f, ci)
        if rects is None:
            return None
        out = torch.as_tensor(rects, dtype=torch.float32).reshape(-1, 4)
        if warp is not None and out.numel():
            out = _warp_region(out, warp)
        scale, pad = self._transform(i, sess.rig.size(sess.cam_names[ci]))
        out[:, 0::2] = out[:, 0::2] * scale + pad[0]
        out[:, 1::2] = out[:, 1::2] * scale + pad[1]
        return out

    def present_ignore_boxes_for(self, i, warp=None):
        """Crop-rule boxes for `instances.pq` PRESENT rows -- an animal in this view that was NOT
        annotated -- in INPUT pixels, `(M,4)`, or None.

        These rows are trained as background by DEFAULT (`boxes_for` never sees them, so
        `detector_loss`'s zeros-target treats every anchor inside one as "no animal here"). This
        is the training-side counterpart of `ignore_for`, which reports the SAME population as a
        scoring exclusion but takes no `warp`; the loss needs the boxes warped exactly as the GT
        boxes beside them are, or a rotated ignore box masks the wrong pixels.

        FOUR corners through the warp, then RE-DERIVED by `crop_box_for_points` -- the same
        `box_source = 'instances'` geometry `boxes_for` uses -- not `_warp_region`'s inscribed
        rectangle: a certified region is a claim that must shrink toward safety, but an ignore box
        exists to keep covering an animal that might not be annotated.

        None when the session ships no `instances.pq` boxes at all; empty `(0,4)` when it does but
        nothing in this view is PRESENT -- the same None/empty distinction `regions_for` keeps.
        """
        sess, gid, f, ci = self.index[i]
        lab = sess.labels(gid)
        if lab.instance is None or lab.boxes is None:
            return None
        present = np.flatnonzero(lab.instance[:, f, ci] == INST_PRESENT)
        if not present.size:
            return torch.zeros((0, 4), dtype=torch.float32)
        cam = sess.cgroup(gid, f)[ci]
        tile_box = None
        if self.origins[i] is not None:
            ox, oy = self.origins[i]
            tw, th = self._tile_extent()
            tile_box = (ox, oy, ox + tw, oy + th)

        out = []
        for s in present.tolist():
            b = torch.as_tensor(lab.boxes[s, f, ci], dtype=torch.float32)
            if not torch.isfinite(b).all():
                continue
            x0, y0, x1, y1 = b
            src = torch.stack([torch.stack([x0, y0]), torch.stack([x1, y0]),
                               torch.stack([x1, y1]), torch.stack([x0, y1])])
            if warp is not None:
                src = _apply_affine(src, (warp, None))
                src = _drop_outside(src, (0.0, 0.0, float(cam['size'][0]), float(cam['size'][1])))
            if tile_box is not None:
                src = _drop_outside(src, tile_box)
            box = crop_box_for_points(src, cam['size'], self.min_crop_dim, pad=0)
            if box is not None:
                out.append(box.float())
        if not out:
            return torch.zeros((0, 4), dtype=torch.float32)
        boxes = torch.stack(out)
        scale, pad = self._transform(i, cam['size'])
        boxes = boxes.clone()
        boxes[:, 0::2] = boxes[:, 0::2] * scale + pad[0]
        boxes[:, 1::2] = boxes[:, 1::2] * scale + pad[1]
        return boxes

    def ignore_for(self, i):
        """`(ig (S,) bool, ig_boxes (S,4))` for item `i`, in INPUT pixels. Or `(None, None)`.

        The `instances.pq` PRESENT rows -- an animal in this view that was not annotated, so a
        prediction on it is neither a true nor a false positive; scoring them as false positives
        would measure the annotator. HERE rather than in `evaluate.py` because it needs item `i`'s
        own transform, which under tiling is the tile's and not the frame letterbox.
        """
        sess, gid, f, ci = self.index[i]
        lab = sess.labels(gid)
        if lab.instance is None:
            return None, None
        ig = lab.instance[:, f, ci] == INST_PRESENT
        boxes = None
        if lab.boxes is not None:
            scale, pad = self._transform(i, sess.rig.size(sess.cam_names[ci]))
            b = np.asarray(lab.boxes[:, f, ci], np.float64).copy()
            b[..., 0::2] = b[..., 0::2] * scale + pad[0]
            b[..., 1::2] = b[..., 1::2] * scale + pad[1]
            boxes = b
        return ig, boxes

    def boxes_for(self, i, warp=None, with_keypoints=False):
        """The letterboxed target boxes for item `i`, without decoding its image.

        `with_keypoints=True` also returns the KEYPOINT target, (S,K,3) of (x, y, vis), in the
        same letterboxed pixels -- free, because the keypoints are the input `crop_box_for_points`
        is already called on, so there is no second data path to disagree about the transform.

        The `vis` channel is the format's `status`, NOT coordinate-finiteness: supervising
        `isfinite(x, y)` teaches "was this annotated", which on a root that writes every point
        VISIBLE is an all-true target. `vis` is NaN where the session made no assessment, so the
        score loss is withheld there rather than asserting "not visible".

        Split out of `__getitem__` so the assignment diagnostic can read what the loss is actually
        assigned over without paying for the pixels.

        `warp` is an augmentation's 2x3 in SOURCE pixels. The geometry moves through it and the
        box is then RE-DERIVED by the crop rule, never scaled: the 20 px pad would scale with the
        image but the `min_crop_dim` floor would not, so a floored box scaled by 0.8 is a box the
        rule can never emit.
        """
        sess, gid, f, ci = self.index[i]
        lab = sess.labels(gid)
        # Frame-indexed, not whole-group: `pts` below is (S,K,3) whose axis -3 is the ANIMAL, so a
        # moving camera's (T,4,4) extrinsic would project animal `i` through frame `i`'s pose.
        cam = sess.cgroup(gid, f)[ci]
        p2d = self._points_2d(sess, gid, f, ci)
        if sess.mode == '3d':
            vis = None if lab.vis3d is None else lab.vis3d[:, f]                 # (S,K)
        else:
            vis = None if lab.vis2d is None else lab.vis2d[:, f, :, ci]

        # A point outside the TILE is not in this image, so it is dropped exactly as an
        # out-of-frame point is: the box shrinks to the visible part, and if every point goes
        # `crop_box_for_points` returns None and the animal is correctly "not here". In SOURCE
        # pixels and applied AFTER the warp, because the tile is defined in post-warp coordinates
        # -- `__getitem__` composes `tile @ warp @ decode`, in that order.
        tile_box = None
        if self.origins[i] is not None:
            ox, oy = self.origins[i]
            tw, th = self._tile_extent()
            tile_box = (ox, oy, ox + tw, oy + th)

        drop_outside = _drop_outside

        # The keypoint target rides the SAME warp and the SAME letterbox as the boxes below, and
        # is derived here so there is exactly one copy of that transform.
        kpts = None
        if with_keypoints:
            k = p2d.clone()
            if warp is not None:
                k = _apply_affine(k, (warp, None))
                k = drop_outside(k, (0.0, 0.0, float(cam['size'][0]), float(cam['size'][1])))
            if tile_box is not None:
                k = drop_outside(k, tile_box)
            # COORDINATES LIVE ON `VISIBLE` *OR* `PROJECTED`, but only VISIBLE is a visibility
            # CLAIM -- `projected` is a position from a source that never recorded occlusion. So a
            # projected point keeps its coordinates and gets NO score target.
            if vis is None:
                v = torch.full(k.shape[:-1], float('nan'))
            else:
                vt = torch.as_tensor(np.asarray(vis))
                v = torch.where(vt == PROJECTED, torch.nan,
                                (vt == VISIBLE).to(torch.float32))
            kpts = torch.cat([k, v[..., None].to(k.dtype)], -1)                 # (S,K,3)

        boxes = []
        for s in range(p2d.shape[0]):
            # A stored box is an ALREADY-PADDED extent, so it re-enters the rule at pad 0 and
            # picks up only the min_crop_dim floor and the clamp. Per animal, not per session:
            # rat-city's boxes come from a tracker that loses animals, and where it did the
            # keypoints are still the best -- and only -- source left.
            src, pad = p2d[s], 20
            if self.box_source == 'instances' and lab.boxes is not None:
                b = torch.as_tensor(lab.boxes[s, f, ci], dtype=torch.float32)
                if torch.isfinite(b).all():
                    # FOUR corners, not two. Under a rotation or a flip the extent of the two
                    # diagonal corners is strictly inside the extent of all four, so a two-corner
                    # warp crops the animal the box exists to enclose.
                    x0, y0, x1, y1 = b
                    src = torch.stack([torch.stack([x0, y0]), torch.stack([x1, y0]),
                                       torch.stack([x1, y1]), torch.stack([x0, y1])])
                    pad = 0
            if warp is not None:
                src = _apply_affine(src, (warp, None))   # shared with the pose loader's rotation
                # A point warped off the frame is not a point. Dropping it shrinks the box to the
                # visible part, which is what a real crop of a half-out animal looks like; drop
                # them all and `crop_box_for_points` returns None, i.e. "no animal here".
                src = drop_outside(src, (0.0, 0.0, float(cam['size'][0]), float(cam['size'][1])))
            if tile_box is not None:
                src = drop_outside(src, tile_box)
            # `min_crop_dim` and the pad stay in SOURCE units and the box is RE-DERIVED here, never
            # scaled -- the pad would scale with the tile but the floor would not, so a floored box
            # scaled by `tile_scale` is a box the rule can never emit.
            box = crop_box_for_points(src, cam['size'], self.min_crop_dim, pad)
            boxes.append(torch.full((4,), float('nan')) if box is None else box.float())
        boxes = torch.stack(boxes)
        scale, pad = self._transform(i, cam['size'])
        boxes[:, 0::2] = boxes[:, 0::2] * scale + pad[0]
        boxes[:, 1::2] = boxes[:, 1::2] * scale + pad[1]
        if kpts is None:
            return boxes
        kpts[..., 0] = kpts[..., 0] * scale + pad[0]
        kpts[..., 1] = kpts[..., 1] * scale + pad[1]
        return boxes, kpts

    def _load_letterbox(self, i, with_keypoints=False):
        """Decode item `i` and letterbox (or tile) it with NO augmentation whatsoever.

        `(img_uint8, boxes, kpts_or_None)`, all in INPUT pixels. Used only by mosaic-lite as a
        source of another frame's pixels and its own crop-rule box -- it must not recurse into
        `__getitem__`'s augmentation path, or a mosaic source could itself be mosaicked.
        """
        import cv2

        sess, gid, f, ci = self.index[i]
        size = tuple(sess.rig.size(sess.cam_names[ci]))
        out_wh = (self.input_wh if self.tile_wh is None
                  else (size[0] * self.tile_scale, size[1] * self.tile_scale))
        r = reduce_factor(size, out_wh) if self.reduce else 1
        img = read_frames(sess.groups[gid], sess.cam_names[ci], [f], reduce=r)[0]
        if img is None:
            raise RuntimeError(f'{gid}/{sess.cam_names[ci]}: frame {f} unreadable')
        dec = (img.shape[1], img.shape[0])
        d = size[0] / dec[0]
        if self.origins[i] is None:
            img, _, _ = letterbox(img, self.input_wh, src_wh=size)
        else:
            scale, pad = self._transform(i, size)
            L = np.array([[scale, 0.0, pad[0]], [0.0, scale, pad[1]], [0.0, 0.0, 1.0]], np.float32)
            D = np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, 1.0]], np.float32)
            M = (L @ D)[:2]
            img = cv2.warpAffine(img, M, self.input_wh, borderValue=(114, 114, 114))
        got = self.boxes_for(i, None, with_keypoints=with_keypoints)
        boxes, kpts = got if with_keypoints else (got, None)
        return img, boxes, kpts

    def _mosaic_paste(self, i, boxes, kpts, img, rng):
        """Copy-paste one WHOLE, untruncated crop-rule box from a DIFFERENT frame into empty
        space of this item's rendered image, at a location fully interior to `input_wh`.

        Deliberately copy-paste and not true YOLOX 4-quadrant mosaic: a quadrant clips whatever
        animal sits on its seam, and a clipped extent is not `crop_box_for_points`'s box for
        anything -- it would train the detector off a target it never emits. A relocated WHOLE box
        has exactly the crop rule's extent, translated. Never clips: a placement that would not
        fit entirely inside `input_wh` is redrawn, up to a few tries, and the item is returned
        unchanged if none fits.
        """
        if self.use_regions:
            raise RuntimeError('mosaic-lite is undefined under --use-regions: a composite frame '
                               'has no expressible certified-area mask')
        if len(self) < 2:
            return boxes, kpts, img
        for _ in range(4):
            j = int(rng.integers(len(self)))
            if j == i:
                continue
            src_img, src_boxes, src_kpts = self._load_letterbox(j, with_keypoints=self.keypoints)
            finite = torch.isfinite(src_boxes).all(-1)
            if not bool(finite.any()):
                continue
            cand = torch.nonzero(finite).flatten().tolist()
            s = int(rng.choice(cand))
            b = src_boxes[s]
            x0, y0, x1, y1 = (int(v) for v in b.round().tolist())
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0 or bw > self.input_wh[0] or bh > self.input_wh[1]:
                continue
            dx = int(rng.integers(0, self.input_wh[0] - bw + 1))
            dy = int(rng.integers(0, self.input_wh[1] - bh + 1))
            img = img.copy()
            img[dy:dy + bh, dx:dx + bw] = src_img[y0:y1, x0:x1]
            offset = torch.tensor([dx - x0, dy - y0, dx - x0, dy - y0], dtype=boxes.dtype)
            boxes = torch.cat([boxes, (b + offset).to(boxes.dtype)[None]], 0)
            if self.keypoints and kpts is not None:
                k = src_kpts[s].clone()
                k[..., 0] += (dx - x0)
                k[..., 1] += (dy - y0)
                # A keypoint of the source instance that falls outside the PASTED box is not
                # part of this composite's crop -- the same rule a point warped off-frame follows.
                outside = ((k[..., 0] < dx) | (k[..., 0] >= dx + bw) |
                          (k[..., 1] < dy) | (k[..., 1] >= dy + bh))
                k[..., :2] = torch.where(outside[..., None], torch.nan, k[..., :2])
                kpts = torch.cat([kpts, k[None]], 0)
            break
        return boxes, kpts, img

    def __getitem__(self, i):
        import cv2

        sess, gid, f, ci = self.index[i]
        size = tuple(sess.rig.size(sess.cam_names[ci]))
        # FRESH ENTROPY PER VISIT ON TRAIN, seeded by index on eval. The old per-item seeding
        # delivered the identical `random_affine` and brightness on every revisit -- one frozen
        # pre-augmented copy of the dataset. `default_rng(None)` draws from OS entropy per call, so
        # there is no shared stream for workers to share, exactly as the pose loader does. Eval
        # still gets NO augmentation at all.
        rng = (np.random.default_rng(None) if self.augment and self.train else None)
        warp = (random_affine(size, rng, hflip=self.hflip, rotate_deg=self.rotate_deg,
                              centre=self._warp_centre(i)) if rng is not None else None)
        got = self.boxes_for(i, warp, with_keypoints=self.keypoints)
        boxes, kpts = got if self.keypoints else (got, None)
        # `regions` and `ignore_boxes` share ONE opt-in (M,4) tuple slot -- `__init__` raises if
        # both flags are set, so at most one of the two is non-None below.
        regions = self.regions_for(i, warp) if self.use_regions else None
        ignore_boxes = self.present_ignore_boxes_for(i, warp) if self.ignore_present else None

        # WHAT THE PIXELS ARE HEADED FOR, WHICH UNDER TILING IS NOT `input_wh`. Tiled,
        # `self.input_wh` IS the tile size, so comparing the whole frame against it gave r = 2-4
        # and the tile became a UPSAMPLE of a decimated frame -- while deployment letterboxes the
        # whole frame at `tiled_input_wh(src, tile_scale)`, where `reduce_factor` returns 1 and the
        # detector sees native pixels. The deployment-equivalent target is the frame at
        # `tile_scale`.
        out_wh = (self.input_wh if self.tile_wh is None
                  else (size[0] * self.tile_scale, size[1] * self.tile_scale))
        r = reduce_factor(size, out_wh) if self.reduce else 1
        img = read_frames(sess.groups[gid], sess.cam_names[ci], [f], reduce=r)[0]
        if img is None:
            raise RuntimeError(f'{gid}/{sess.cam_names[ci]}: frame {f} unreadable')
        dec = (img.shape[1], img.shape[0])
        # A video root IGNORES `reduce`, so the frame comes back full size -- both are legal, and
        # what is not legal is a frame that matches neither: the box transform is derived from the
        # rig's recorded size, so that would letterbox the boxes and the pixels differently.
        want = tuple(-(-size[a] // r) for a in (0, 1))                  # libjpeg rounds up
        assert dec == want or dec == size, \
            f'{gid}/{sess.cam_names[ci]} frame {f}: decoded {dec}, expected {want} at reduce={r} '\
            f'or {size} unreduced'
        d = size[0] / dec[0]                     # decoded pixels -> source pixels, 1.0 for video
        M = None            # non-None only on the warpAffine branch -- reused below for t-1
        gain = None         # the ONE photometric draw for this item -- reused below for t-1
        if warp is None and self.origins[i] is None:
            img, _, _ = letterbox(img, self.input_wh, src_wh=size)
        else:
            # ONE warpAffine for the decode scale, the augmentation AND the letterbox-or-tile.
            # Three would resample the frame three times, and the loader is the expensive half of
            # an iteration. `borderValue` is already what a tile hanging off the frame edge should
            # see.
            scale, pad = self._transform(i, size)
            L = np.array([[scale, 0.0, pad[0]], [0.0, scale, pad[1]], [0.0, 0.0, 1.0]], np.float32)
            D = np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, 1.0]], np.float32)
            W = np.vstack([warp, [0, 0, 1]]) if warp is not None else np.eye(3, dtype=np.float32)
            M = (L @ W @ D)[:2]
            img = cv2.warpAffine(img, M, self.input_wh, borderValue=(114, 114, 114))
            if warp is not None:
                gain = rng.uniform(0.7, 1.3)
                img = _photometric(img, rng, gain=gain)
                # THE STRONG SUITE. Gated on `self.strong` alone here -- `rng is not None` already
                # means `self.augment and self.train`. FIXED ORDER, one independent per-op draw
                # each: color jitter -> additive noise -> salt & pepper -> motion blur.
                # Appearance-only, so no box or keypoint target moves through any of these four.
                if self.strong:
                    if rng.random() < 0.5:
                        img = _color_jitter(img, rng)
                    if rng.random() < 0.3:
                        img = _additive_noise(img, rng)
                    if rng.random() < 0.2:
                        img = _salt_pepper(img, rng)
                    if rng.random() < 0.2:
                        img = _motion_blur(img, rng)
        # T4.2: frame t-1, warped by the EXACT SAME `M` (so both frames of a stacked pair are in
        # register) and given the SAME photometric gain as frame t (so no fake per-frame brightness
        # "motion" enters the stack). Cannot coexist with `self.strong` (raised at construction),
        # so this never has to interact with cutout or mosaic-lite below.
        prev_img = None
        if self.temporal_input != 'none':
            prev_f = max(f - 1, 0)     # clip start: no t-1 exists, so repeat frame t
            prev_raw = read_frames(sess.groups[gid], sess.cam_names[ci], [prev_f], reduce=r)[0]
            if prev_raw is None:
                raise RuntimeError(f'{gid}/{sess.cam_names[ci]}: frame {prev_f} (t-1 for '
                                   'temporal_input) unreadable')
            if M is None:
                prev_img, _, _ = letterbox(prev_raw, self.input_wh, src_wh=size)
            else:
                prev_img = cv2.warpAffine(prev_raw, M, self.input_wh, borderValue=(114, 114, 114))
                if gain is not None:
                    prev_img = _photometric(prev_img, rng, gain=gain)
        if self.strong and rng is not None:
            # CUTOUT: an erasure, so it moves no box -- but a keypoint it covers must lose BOTH
            # its coordinate (-> NaN) and its score target (-> 0), or either loss would still
            # supervise a point the pixels no longer show.
            if rng.random() < 0.5:
                rects = _cutout_rects(self.input_wh, rng)
                img = _apply_cutout(img, rects)
                if kpts is not None:
                    mask = _keypoints_in_rects(kpts[..., :2], rects)
                    kpts[..., 0] = torch.where(mask, torch.nan, kpts[..., 0])
                    kpts[..., 1] = torch.where(mask, torch.nan, kpts[..., 1])
                    kpts[..., 2] = torch.where(mask, torch.zeros_like(kpts[..., 2]), kpts[..., 2])
            # MOSAIC-LITE: copy-paste one whole crop-rule box from another frame into empty space.
            # The only op here that adds a box -- appearance and cutout never do.
            if rng.random() < 0.2:
                boxes, kpts, img = self._mosaic_paste(i, boxes, kpts, img, rng)
        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if prev_img is not None:
            # ORDER: (t-1, t), oldest first -- `TEMPORAL_INPUT_CHANNELS['stack2'] == 6` and
            # `YOLOXNano(in_channels=6)`'s stem takes whatever six channels it is given; any caller
            # supplying frames at inference must match it.
            xp = torch.as_tensor(prev_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            x = torch.cat([xp, x], dim=0)
        # `regions` rides along as its own element rather than being folded into `boxes`: a region
        # is not an animal and must never reach `assign`. Appended only under `use_regions`, so
        # without it the item shape is exactly what every recorded detector trained on.
        #
        # `regions_for` returning None means the session claims exhaustive labelling, and the
        # faithful encoding of that HERE is one rect covering the whole input -- then "certified"
        # is all-True and the masked loss equals the unmasked one, with no special case anywhere
        # downstream.
        out = (x, boxes) if kpts is None else (x, boxes, kpts)
        if self.ignore_present:
            # Unlike `regions`, `None` and empty mean the SAME thing for an ignore box -- "nothing
            # to mask here" either way -- so an empty (0,4) is already correct.
            return (*out, ignore_boxes if ignore_boxes is not None
                   else torch.zeros((0, 4), dtype=torch.float32))
        if not self.use_regions:
            return out
        if regions is None:
            regions = torch.tensor([[0.0, 0.0, float(self.input_wh[0]), float(self.input_wh[1])]])
        return (*out, regions)


class ChunkShuffle(torch.utils.data.Sampler):
    """A shuffle that keeps a worker inside a few videos at a time.

    `BoxDataset.index` is built session-by-session, so a run of contiguous index positions stays
    inside one video, and a plain `shuffle=True` would send every item to a different container
    and thrash `dataset._reader`'s cache (486 ms per batch of 16 against a 16 ms GPU step, on a
    multi-session root). Shuffling BLOCKS instead, and pooling `mix` of them so a batch still
    spans several sessions, costs 40 ms -- and a batch drawn from one video is still a correlated
    gradient step. PASS `chunk=dataset.chunk`; the default 512 was set against a one-group root.

    Image-directory datasets do not need this and are not harmed by it: their frames are separate
    files, so locality buys nothing and costs nothing.
    """

    def __init__(self, n, chunk=512, mix=4, seed=23):
        self.n, self.chunk, self.mix, self.seed = n, chunk, mix, seed
        self.epoch = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        starts = np.arange(0, self.n, self.chunk)
        rng.shuffle(starts)
        for i in range(0, len(starts), self.mix):
            pool = np.concatenate([np.arange(s, min(s + self.chunk, self.n))
                                   for s in starts[i:i + self.mix]])
            rng.shuffle(pool)
            yield from (int(j) for j in pool)


class CohortSampler(torch.utils.data.Sampler):
    """Draw index positions WITH REPLACEMENT at fixed per-cohort probability.

    Built from `BoxDataset.cohort_weights(annot_frac)`, which is None unless the split actually
    holds both an `annotated` and a `tracked` session -- so this sampler is constructed only on
    the roots where the question exists (rat-city-combined, allen-mouse-combined,
    johnson-mouse-combined-aug). On the other three `train_detector.py` keeps `ChunkShuffle` and
    the run is byte-identical to every detector on record.

    REPLACEMENT, and therefore no epoch boundary, for `StepSampler`'s reason on the pose side: a
    fixed cohort share is a property of the STEP, and an epoch that must contain each entry once
    cannot hold one. `__len__` stays at `n` so `iters` keeps meaning what it meant.

    LOCALITY IS DELIBERATELY DROPPED HERE, and only here. `ChunkShuffle` exists to keep a worker
    inside a few video containers; all three roots that reach this sampler read IMAGE
    DIRECTORIES on train (`Group.source` -> 'frames'), where random access is free -- measured on
    rat-city-combined at 30.6 ms/frame random against 30.2 ms/frame sequential. calms21, the one
    train-time video root, is single-cohort and never gets here. If that ever changes, this is
    the line that has to be revisited, not `ChunkShuffle`.
    """

    def __init__(self, weights, num_samples=None, seed=23):
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or not w.size:
            raise ValueError(f'weights must be a non-empty 1-D array, got shape {w.shape}')
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError('weights must be finite, non-negative and not all zero')
        # One cumulative array and one `searchsorted` per draw, as `PoseDataset._pick` does.
        self.cum = np.cumsum(w / w.sum())
        self.num_samples = int(len(w) if num_samples is None else num_samples)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        draws = np.searchsorted(self.cum, rng.random(self.num_samples))
        # `searchsorted` can return `len(cum)` on a draw of exactly 1.0; clamp rather than let a
        # one-in-2^53 sample raise IndexError 14,000 iterations in.
        yield from (int(j) for j in np.minimum(draws, len(self.cum) - 1))


def box_collate(batch):
    """Pad the box axis with NaN so a batch can hold different animal counts.

    NaN, not zero, and that carries all the way to the loss: a padded row is "no animal", which is
    the same signal a real animal with no finite point in this view sends. Keypoints pad the same
    way, so a padded (S,K,3) slice is non-finite in every channel and every mask drops it.

    An item is `(x, boxes[, kpts][, regions])` and the optional tails are told apart BY RANK --
    keypoints are (S,K,3) and regions (M,4). Dispatching on tuple length alone is ambiguous at
    three elements, which is the kind of thing that silently feeds regions to the keypoint loss.
    """
    xs = torch.stack([b[0] for b in batch])
    n = max(b[1].shape[0] for b in batch)
    boxes = torch.full((len(batch), n, 4), float('nan'))
    for i, b in enumerate(batch):
        boxes[i, :b[1].shape[0]] = b[1]
    tails = [t for t in batch[0][2:]]
    out = [xs, boxes]

    if any(t.dim() == 3 for t in tails):
        K = next(t for t in tails if t.dim() == 3).shape[1]
        kpts = torch.full((len(batch), n, K, 3), float('nan'))
        for i, b in enumerate(batch):
            k = next(t for t in b[2:] if t.dim() == 3)
            kpts[i, :k.shape[0]] = k
        out.append(kpts)

    if any(t.dim() == 2 for t in tails):
        # NaN-padded like the boxes, and for the same reason: `certified_anchors` drops a
        # non-finite rect, so a padded row certifies nothing rather than certifying the origin.
        rs = [next(t for t in b[2:] if t.dim() == 2) for b in batch]
        m = max(1, max(r.shape[0] for r in rs))
        regions = torch.full((len(batch), m, 4), float('nan'))
        for i, r in enumerate(rs):
            regions[i, :r.shape[0]] = r
        out.append(regions)
    return tuple(out)


def split_batch(batch):
    """A collated batch -> `(x, boxes, kpts_or_None, regions_or_None)`.

    THE one place the optional tails are told apart, by RANK: keypoints are (B,S,K,3) and regions
    (B,M,4). `len(batch) > 2` cannot do it -- with keypoints off and regions on, a three-element
    batch's third element is regions, and reading it as `gt_kpts` would feed rectangles to the
    keypoint loss and train the branch against them.
    """
    x, boxes = batch[0], batch[1]
    kpts = next((t for t in batch[2:] if t.dim() == 4), None)
    regions = next((t for t in batch[2:] if t.dim() == 3), None)
    return x, boxes, kpts, regions
