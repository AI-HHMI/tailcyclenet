"""Box training data: labels -> the crop rule's box, letterboxed into the detector's input.

The target is `tailcyclenet.crop.crop_box_for_points` applied to the same points the pose loader
crops on. That is the point of the whole detector: it reproduces the crop the pose model was
trained on, so swapping a GT crop for a detector crop costs a fraction of a millimetre instead
of whatever an independently-plausible box rule would cost.

An animal with no finite point in a view gets a **NaN box**, not a dropped frame. Objectness
still has to learn "no animal here" for that view, and silently dropping those frames would
train a detector that has never seen an empty image.

`box_source='instances'` takes the extent from `instances.pq` instead, for a dataset whose stored
keypoints are too sparse to bound the animal: rat-city's converter dropped noisy points, leaving
25,777 train instances with NO finite point at all -- each one a NaN box teaching "no animal here"
where a rat plainly is. It is opt-in rather than "use the table when it exists" because an
`instances.pq` box is not a crop box in general (spec §9 says so outright; johnson-mouse ships COCO
boxes and calms21 ships MARS ones), and defaulting to it would silently retarget those detectors.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.posetail.cube import project_points_torch

from ..crop import BOX_SOURCES, crop_box_for_points
from ..dataset import _apply_affine, read_frames
from ..format import INST_PRESENT, PROJECTED, UNLABELED, VISIBLE, load_datasets


def reduce_factor(size, out_wh):
    """The largest `cv2.IMREAD_REDUCED_COLOR_N` that still decodes above the letterbox target.

    rat-city's frames are 4696x2048 and the detector letterboxes them into 640x288 -- a 7.3x
    downscale, which `cv2.resize`'s default INTER_LINEAR does by sampling 2x2 of every 7x7 block.
    That is not a resize, it is aliasing: an animal 32 px across in the target is being built out
    of a handful of source pixels chosen by position rather than content, and which ones depends
    on the sub-pixel phase.

    libjpeg's DCT-domain decimation is a proper box filter AND halves the decode, so the fix is to
    ask for fewer pixels rather than to throw more away afterwards. `INTER_AREA` would also filter
    correctly but costs 15.4 ms against INTER_LINEAR's 0.26 ms at this scale -- roughly +40% on a
    loader-bound iteration -- where reducing at decode is free twice over.

    N stays a power of two at or below 8 (what libjpeg offers) and never takes the decode below
    the letterbox target, so the remaining resize is still a downscale.
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

    Same arithmetic, no pixels: the box target and the assignment diagnostics need the transform
    without paying rat-city's 39 ms JPEG decode to learn it.
    """
    w, h = float(size[0]), float(size[1])
    s = min(out_wh[0] / w, out_wh[1] / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    return s, ((out_wh[0] - nw) // 2, (out_wh[1] - nh) // 2)


def tile_transform(origin, scale):
    """The `(scale, (padx, pady))` that renders a source-pixel tile at `origin` into the input.

    Deliberately the SAME two-number form `letterbox_transform` returns, and that is the whole
    reason tiling is not a second data path in this file: every geometry line here applies a
    transform as `x * scale + pad`, and a tile whose top-left source pixel is `(ox, oy)`, rendered
    at scale `s`, is exactly `(s, (-ox*s, -oy*s))`. So `boxes_for`, `regions_for` and
    `__getitem__`'s single `warpAffine` all tile by substituting this for the letterbox.

    There is no padding term to compute because the tile's source extent is `input_wh / scale` by
    construction, so its aspect ratio already matches the input and a letterbox would be a no-op.
    """
    s = float(scale)
    return s, (-float(origin[0]) * s, -float(origin[1]) * s)


def unletterbox_boxes(boxes, scale, pad, src_wh=None):
    """Detector-input boxes -> source-image boxes. With `src_wh`, clamped into the frame.

    THE CHOKE POINT EVERY DETECTOR BOX GOES THROUGH, which is why the bound is here and not in the
    four callers. Nothing else bounds one: `yolox.py:167` decodes a side as
    `exp(clamp(-6, 6)) * stride`, i.e. up to ~12,910 px per side at stride 32, and this function
    then divides by a letterbox scale that can be 1/7 -- ~137,000 source pixels. IoU-only NMS
    cannot suppress a box that big either, because its IoU with the real box it swallows is ~0.

    Downstream that box becomes a crop: `run_group` unions the window's boxes and resizes the
    result to 256 px, so ONE such frame delivers the whole arena as a thumbnail. A box with no
    positive area after clamping is not a detection at all and comes back NaN, which every
    consumer already reads as "no box here" -- `associate` skips a non-finite centre and
    `run_group` counts it out of the union.

    `src_wh` is optional only because `BoxDataset`'s training path has no frame to clamp against at
    the point it calls this; every deployment caller passes it.
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

    Lives HERE, beside the box version, for the reason the box version gives for living here: it
    is the one inverse every consumer goes through, and a keypoint un-letterboxed by a different
    rule than its own box is the class of bug that is invisible in every downstream number.

    Out-of-frame keypoints go NaN rather than clamping. A box is clamped because a partly-visible
    animal still has a real extent inside the frame; a keypoint outside the frame was not observed
    there, and clamping it to the border would invent a position on the edge -- which is precisely
    what `dataset.py`'s prior bounds mask exists to refuse.
    """
    out = kpts.clone().float()
    out[..., 0] = (out[..., 0] - pad[0]) / scale
    out[..., 1] = (out[..., 1] - pad[1]) / scale
    if src_wh is not None:
        w, h = float(src_wh[0]), float(src_wh[1])
        bad = ((out[..., 0] < 0) | (out[..., 0] > w) | (out[..., 1] < 0) | (out[..., 1] > h))
        out[..., :2] = torch.where(bad[..., None], torch.nan, out[..., :2])
    return out


def _photometric(img, rng):
    """The appearance half of `--augment`: one multiplicative gain, exactly as shipped.

    DLC's extended form -- additive brightness plus per-channel gaussian noise -- was built,
    measured and REFUTED here: MOTA -0.127 [-0.256, -0.016] and miss +0.064, both significant, on
    rat-city in-domain. DLC buys it for an OUT-of-domain gain (-0.25 mAP on marmosets without it)
    and neither rat-city split tests that, so the honest summary is that it costs accuracy in the
    only regime this repo can measure. Re-add it with a root that has a genuinely different camera
    or enclosure, not before.
    """
    # THE `extended` BRANCH WAS DELETED, NOT DISABLED. The commit that pulled `--augment-
    # photometric` (`3dbb0a1`) removed the `extended` parameter from this function's signature
    # and from `BoxDataset.__init__`, but left an `if extended:` block referencing it -- a
    # `NameError` on every call, live in every checkpoint trained with `--augment` since. Nothing
    # in the test suite calls `BoxDataset.__getitem__` with `augment=True` through to a real pixel
    # decode (the augmentation tests only exercise `boxes_for`, which never reaches this
    # function), so it went undetected until a real training run hit it.
    out = img * rng.uniform(0.7, 1.3)
    return np.clip(out, 0, 255).astype(np.uint8)


def random_affine(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5,
                  rotate_deg=0.0, centre=None):
    """A random similarity about `centre` (default the image centre), source px in and out, 2x3.

    Deliberately a similarity and not YOLOX's shear-and-perspective: the target is
    `crop_box_for_points`, an AXIS-ALIGNED extent, and a shear turns a box into a parallelogram
    whose extent is no longer the crop rule's box for anything.

    `rotate_deg` IS THE REPLACEMENT FOR THE FLIP, and it is strictly easier than the flip was: a
    mirror permutes left/right keypoint names and so needs a `flip_pairs` map, while a rotation
    moves every keypoint through the same affine and permutes nothing. With `--keypoints` on
    everywhere `hflip` is 0 everywhere, which left scale and translate as the whole of the
    geometric augmentation.

    `centre` IS WHY TILING AND ROTATION COMPOSE. `__getitem__` builds `tile @ warp @ decode`, so
    the tile is cut AFTER this warp: about the frame centre, a scale of 0.8 moves an animal 2,000
    px out by 400 -- already more than a 640 px tile -- and a large rotation moves it out of the
    tile entirely, so every animal- and region-chosen tile would come back holding something else.
    Passing the tile's own centre pins the tile's content and makes rotation FREE there, because a
    tile interior to a 4696x2048 frame pulls real neighbouring pixels in at every angle. On whole
    frames it is not free: the mean real-pixel fraction of a 2.29:1 frame is 0.92 at +-15, 0.79 at
    +-45 and 0.644 (min 0.437) at BOTH +-90 and +-180 -- so the whole cost is paid by the first 90
    degrees and nothing is saved by stopping short of a full circle.

    No `flip_pairs`. The detector emits one box, and a box is the extent of a SET of points, so
    relabelling left to right is a permutation of that set and the extent is unchanged.

    **THAT JUSTIFICATION DIES THE MOMENT KEYPOINTS ARE A TARGET**, which is why `BoxDataset`
    passes `hflip=0` whenever it is emitting them. A mirrored frame swaps every left/right pair,
    so without a `flip_pairs` map the head sees `left_wing` labelled at the right wing half the
    time, learns their mean, and collapses both toward the midline -- degrading exactly the
    lateral asymmetry cross-view association depends on. Real `flip_pairs` need a per-dataset name
    mapping and branson's names are placeholders (`kp00..kp20`, `names_provisional = true`), so
    losing the flip is the cheap correct answer rather than inventing one.
    """
    w, h = float(size[0]), float(size[1])
    s = rng.uniform(*scale)
    sx = -s if rng.random() < hflip else s
    cx, cy = (w / 2, h / 2) if centre is None else (float(centre[0]), float(centre[1]))
    A = np.array([[sx, 0.0], [0.0, s]], np.float64)
    # The draw is SKIPPED at rotate_deg 0, not drawn and multiplied by zero: that keeps both the
    # matrix and the rng stream bit-identical to every detector arm recorded before this key.
    #
    # DEFAULT-OFF ON EVIDENCE, not on caution. At +-180 on rat-city it is significantly WORSE on
    # two roots and two label sources, one direction: MPJPE +3.33 px [+0.08, +7.06] on hand labels
    # and +1.28 [+0.63, +1.88] on the tracker clip, coverage -0.009, idsw x1.7 (0.0189 -> 0.0328),
    # err p99 +29.9. The knob is kept because the refutation is of that SETTING on that root, and
    # because a full circle costs no more retained area than a quarter one
    # (`_rotated_rect_max_inscribed` is 90-degree periodic), so a smaller amplitude is not
    # obviously the same trade. dev/reports/21 sections 3a and 3b.
    if rotate_deg:
        a = np.radians(rng.uniform(-rotate_deg, rotate_deg))
        A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ A
    t = (np.array([cx, cy]) - A @ np.array([cx, cy])
         + np.array([rng.uniform(-translate, translate) * w,
                     rng.uniform(-translate, translate) * h]))
    return np.concatenate([A, t[:, None]], 1).astype(np.float32)


def _warp_region(rects, M):
    """(M,4) certified rects through an in-plane similarity, ROUNDING DOWN. Returns (M,4).

    A certified region is a CLAIM -- "everything in here is labelled" -- and a claim must shrink
    under a transform that makes it approximate, never grow. Taking four corners through the warp
    and then their extent grows it: the axis-aligned hull of a rotated rectangle claims area the
    annotator never marked, which re-admits exactly the unlabelled animals `regions.pq` exists to
    exclude (this root labels a median of 2 rats where the tracked one finds 11). So this inscribes
    instead, the same largest-axis-aligned-rect-in-a-rotated-rect computation
    `posetail_dataset._rotated_rect_max_inscribed` uses for the rotated image canvas.

    Under every warp that existed before rotation -- translation, scale, flip -- an axis-aligned
    rect stays axis-aligned, inscribed and circumscribed coincide, and this is a no-op against the
    old code. `test_a_region_is_unchanged_by_a_scale_and_translate_warp` is what holds that.
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
                 max_frames_per_group: int = 40, seed: int = 0, box_source='keypoints',
                 augment=False, reduce=False, keypoints=False, hflip=None, rotate_deg=0.0,
                 tile_wh=None, tile_scale=1.0, tile_bg_per_frame=1, use_regions=False):
        assert box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {box_source!r}'
        self.box_source = box_source
        # TILING, off by default and off means the whole frame -- with `tile_wh=None` every
        # geometry path below takes the letterbox branch and this loader is byte-identical to what
        # every recorded detector was trained on, the same discipline `keypoints` follows.
        #
        # `tile_wh` is the tile in INPUT pixels, i.e. the model's input size, so it REPLACES
        # `input_wh`. `tile_scale` is the source -> input scale and is the only scale there is:
        # the tile's source extent is `tile_wh / tile_scale`. DO NOT ADD A SECOND RESIZE. Tiling
        # and then downscaling the tiles is the number-one reported failure of this pattern (one
        # practitioner reported zero precision at full-size inference from exactly that), because
        # it breaks the invariant the whole scheme rests on -- the animal's size in INPUT pixels
        # must be the same at train and at deployment.
        self.tile_wh = None if tile_wh is None else tuple(int(v) for v in tile_wh)
        self.tile_scale = float(tile_scale)
        self.tile_bg_per_frame = int(tile_bg_per_frame)
        if self.tile_wh is not None and self.tile_scale <= 0:
            raise ValueError(f'tile_scale must be > 0, got {tile_scale}')
        # Mask the objectness loss to the area an annotator certified as completely labelled
        # (`regions.pq`, spec §9b). ORTHOGONAL to tiling and off by default, so an arm can move one
        # lever at a time (eval rule 4) -- but MEASURED DEAD on full-frame input: at 896x384's
        # 7,056 anchors a labelled rat-city frame carries a median of 104 certified anchors of
        # which 48 are positive, a ~69% positive rate against 0.68% unmasked, and 17% of frames
        # have no certified negative at all. Tiling is what fixes that: at 640x640 tiles rendered
        # at scale 1.0 the same measurement reads 5.2% positive with 0% of tiles lacking a
        # certified negative (`scratch/tile_certified_rate.py`). The rate is set by how many
        # stride-8 CELLS the certified region spans, not by its area, because `CENTER_RADIUS` is
        # 2.5 cells -- so it is a resolution knob and `tile_scale` is what sets it.
        self.use_regions = bool(use_regions)
        # Opt-in, default off: with it absent this loader is byte-identical to what every recorded
        # detector was trained on. See `random_affine` for why it also kills the horizontal flip.
        self.keypoints = bool(keypoints)
        # `hflip=None` means "decide from `keypoints`", which is the safe default. It is separable
        # ONLY so a box-only CONTROL arm can match the keypoint arm's augmentation exactly: the
        # keypoint arm necessarily loses the flip, so a control that keeps it differs in two
        # levers and measures neither (eval rule 4).
        self.hflip = (0.0 if self.keypoints else 0.5) if hflip is None else float(hflip)
        # 0 is off and off is byte-identical: `random_affine` skips the draw. See there for
        # why the knob survives a refutation of its 180-degree setting.
        self.rotate_deg = float(rotate_deg)
        # 0 is off and off is byte-identical: `random_affine` skips the draw entirely. See there
        # for why this is free when tiling and not when not, and why a full circle costs no more
        # than a half one.
        # The APPEARANCE half of --augment. False is the shipped single gain and is
        # byte-identical; True adds DLC's additive brightness and gaussian noise.
        # Off by default and requested explicitly, not inferred from the split: it is a key, and
        # an arm that turns it on has to be able to say so. `self.train` still gates it, so a val
        # or test loader built by a script that passes `augment=True` blindly stays deterministic.
        self.augment = augment
        # Off by default, and it is a KEY rather than a loader detail: it changes which source
        # pixels reach the model, so every arm measured without it stays comparable only against
        # other arms without it. It rides in the checkpoint and `detect_group` reads it back,
        # because a detector fed differently-sampled pixels at deployment is off its own training
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
        # input size is a DIFFERENT number (`frame_wh * tile_scale`). See `train_detector.py`, which
        # records `tile_scale` in the checkpoint for exactly that reason.
        self.input_wh = tuple(input_wh) if self.tile_wh is None else self.tile_wh
        self.min_crop_dim = min_crop_dim
        self.train = split == 'train'
        rng = np.random.default_rng(seed)

        # Parallel to `self.index`, one entry per item: the tile's source-pixel origin, or None for
        # the whole frame. A parallel list rather than a fifth tuple element because `index` is
        # unpacked as a 4-tuple in `evaluate.py` and `scripts/diag_assign.py` too, and widening it
        # would break those silently at the call site rather than loudly here.
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
        # be. Its old hardcoded 512 was set against rat-city, where the whole split is one group;
        # on calms21 a session contributes `max_frames_per_group` = 40 positions, so a 512-block
        # spanned 13 videos and a 4-block pool spanned 52. The reader cache then held 8 of 52 and
        # ran at a 16% hit rate -- and at the cache size that thrash used to need, ~1 GB of open
        # decord reader each, it OOM-killed the workers.
        n_src = len({(s.session_id, g, c) for s, g, _, c in self.index})   # (group, camera) = file
        self.chunk = max(1, len(self.index) // n_src)

    def __len__(self):
        return len(self.index)

    # ------------------------------------------------------------------------------------------
    # tiling
    # ------------------------------------------------------------------------------------------

    def _tile_extent(self):
        """The tile's extent in SOURCE pixels. `input_wh / tile_scale`, and nothing else."""
        return (self.tile_wh[0] / self.tile_scale, self.tile_wh[1] / self.tile_scale)

    def _warp_centre(self, i):
        """What `random_affine` turns and scales about: the TILE's centre, or None for the frame.

        `__getitem__` composes `tile @ warp @ decode`, so the tile is cut after the warp. About the
        frame centre a scale of 0.8 moves an animal 2,000 px out by 400 -- more than a 640 px tile
        -- and a rotation moves it out entirely, so a tile chosen for a region or an animal comes
        back holding something else and the tiled arms quietly degrade into background-only tiles.
        The targets were always right (boxes and regions ride the same warp and are then dropped
        outside the tile); what was wrong was which pixels the item contained.
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
        certified", `(0,4)` is "the file exists and certifies nothing here". Collapsing them
        inverts the claim the table exists to make (spec §9b).
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

        The background tiles are the population the mask exists to supply, and where they may be
        drawn from is exactly the None/empty distinction:

        - `regions is None` -- the session claims exhaustive labelling, so anywhere in the frame is
          certified background and the centre is uniform over the frame.
        - `regions` non-empty -- only inside a certified rect. Anywhere else is UNKNOWN, not
          background, and sampling it would reintroduce the unlabelled animals this table exists
          to exclude.
        - `regions` empty `(0,4)` -- nothing here is certified, so there is no certified background
          and none is drawn.
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
        # that has never seen an empty image (the same reason a NaN box is emitted rather than the
        # frame skipped).
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

        THE one place that choice is made. `boxes_for` and `regions_for` both come here, because a
        region transformed by a different rule than its own boxes is invisible in a loss curve --
        the same failure `unletterbox_keypoints` is placed beside `unletterbox_boxes` to avoid.
        """
        if self.origins[i] is None:
            return letterbox_transform(size, self.input_wh)
        return tile_transform(self.origins[i], self.tile_scale)

    def regions_for(self, i, warp=None):
        """Certified rectangles for item `i` in INPUT pixels, `(M,4)`, or None.

        None means the session carries no `regions.pq` and therefore claims to be exhaustively
        labelled -- every anchor is supervised. An empty `(0,4)` means the file exists and this
        view certifies nothing, so nothing is supervised. See `_region_rects`.
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

    def ignore_for(self, i):
        """`(ig (S,) bool, ig_boxes (S,4))` for item `i`, in INPUT pixels. Or `(None, None)`.

        The `instances.pq` PRESENT rows -- an animal that is in this view and was not annotated, so
        a prediction on it is neither a true nor a false positive. rat-city ships 26,021 of them
        and scoring them as false positives measures the annotator.

        HERE rather than in `evaluate.py` because it needs item `i`'s own transform, which under
        tiling is the tile's and not the frame letterbox. `evaluate.py` used to call
        `letterbox_transform` itself, which is right for a whole frame and silently wrong for a
        tile -- and it is the ignore mask, so being wrong makes real animals into false positives.
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
        same letterboxed pixels. It is free: the keypoints are the input `crop_box_for_points`
        is already called on, so there is no second data path and no second chance to disagree
        about the transform -- which is the failure mode here (a keypoint letterboxed by a
        different rule than its own box is invisible in the loss curve).

        The `vis` channel is the format's `status`, NOT coordinate-finiteness. The two come apart
        in both directions and conflating them re-creates a failure this repo has paid for:
        supervising `isfinite(x, y)` teaches "was this annotated", which on calms21 -- whose
        converter writes every point VISIBLE -- is an all-true target, `occlusion_acc` 1.0, and a
        row gate worth -0.037 to -0.123 MOTA. `vis` is NaN where the session made no assessment
        at all, so the score loss is withheld there rather than asserting "not visible".

        Split out of `__getitem__` so `scripts/diag_assign.py` can read what the loss is actually
        assigned over a few hundred views without paying for the pixels -- and so there is one
        copy of the box rule rather than a second one in the diagnostic that can drift from it.

        `warp` is an augmentation's 2x3 in SOURCE pixels. The geometry moves through it and the
        box is then RE-DERIVED by the crop rule, never scaled: the 20 px pad would scale with the
        image but the `min_crop_dim` floor would not, so a floored box scaled by 0.8 is a box the
        rule can never emit and the detector would be trained off its own target (gotcha 8).
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
        # out-of-frame point is: the box shrinks to the visible part, which is what a crop of a
        # half-out animal looks like, and if every point goes `crop_box_for_points` returns None
        # and the animal is correctly "not here". In SOURCE pixels and applied AFTER the warp,
        # because the tile is defined in post-warp coordinates -- `__getitem__` composes
        # `tile @ warp @ decode`, in that order.
        tile_box = None
        if self.origins[i] is not None:
            ox, oy = self.origins[i]
            tw, th = self._tile_extent()
            tile_box = (ox, oy, ox + tw, oy + th)

        def drop_outside(x, bounds):
            lo_x, lo_y, hi_x, hi_y = bounds
            out = ((x[..., 0] < lo_x) | (x[..., 0] > hi_x) |
                   (x[..., 1] < lo_y) | (x[..., 1] > hi_y))
            return torch.where(out[..., None], torch.nan, x)

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
            # COORDINATES LIVE ON `VISIBLE` *OR* `PROJECTED` (fmt.POSITIONED), but only VISIBLE is
            # a visibility CLAIM -- `projected` is a position from a source that never recorded
            # occlusion. So a projected point keeps its coordinates and gets NO score target.
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
                    # warp crops the animal the box exists to enclose. Re-bounding all four under
                    # a ROTATION is also the measured-best rule -- see `crop.box_corners`, which
                    # records the alternative that was built and refuted.
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
            # scaled by `tile_scale` is a box the rule can never emit (gotcha 8).
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

    def __getitem__(self, i):
        import cv2

        sess, gid, f, ci = self.index[i]
        size = tuple(sess.rig.size(sess.cam_names[ci]))
        # FRESH ENTROPY PER VISIT ON TRAIN, seeded by index on eval. `default_rng([self.seed, i])`
        # claimed to be "per item and per epoch" and was only per item: nothing supplies an epoch
        # (`ChunkShuffle.epoch` only permutes the ORDER, and `persistent_workers=True` would keep
        # a forked copy stale anyway). So item `i` got the identical `random_affine` and the
        # identical brightness on every one of its ~300 revisits, i.e. `--augment` and
        # `--rotate-deg` delivered ONE frozen pre-augmented copy of the dataset instead of a fresh
        # draw -- which understates what both levers are worth.
        #
        # `default_rng(None)` draws from OS entropy per call, so there is no shared stream for the
        # workers to share and no dependence on how the DataLoader interleaved them -- the two
        # things the old seeding was defending against. This is what the pose loader already does
        # (`dataset.py:769`), for the same reason. Eval still gets NO augmentation at all.
        rng = (np.random.default_rng(None) if self.augment and self.train else None)
        warp = (random_affine(size, rng, hflip=self.hflip, rotate_deg=self.rotate_deg,
                              centre=self._warp_centre(i)) if rng is not None else None)
        got = self.boxes_for(i, warp, with_keypoints=self.keypoints)
        boxes, kpts = got if self.keypoints else (got, None)
        regions = self.regions_for(i, warp)

        # WHAT THE PIXELS ARE HEADED FOR, WHICH UNDER TILING IS NOT `input_wh`. Tiled,
        # `self.input_wh` IS the tile size, so comparing the whole 4696x2048 frame against it gave
        # r = 2 for a 640x640 tile and r = 4 for 640x288 -- and `M = L @ W @ D` below multiplies
        # the decode scale back up, so the tile became a 2-4x UPSAMPLE of a decimated frame.
        # Deployment does the opposite: `detect_group` letterboxes the whole frame to
        # `tiled_input_wh(src, tile_scale)`, where `reduce_factor` returns 1 and the detector sees
        # native pixels. That is exactly the train/deploy sampling skew `reduce` is stamped into
        # the checkpoint to PREVENT. The deployment-equivalent target is the frame at `tile_scale`.
        out_wh = (self.input_wh if self.tile_wh is None
                  else (size[0] * self.tile_scale, size[1] * self.tile_scale))
        r = reduce_factor(size, out_wh) if self.reduce else 1
        img = read_frames(sess.groups[gid], sess.cam_names[ci], [f], reduce=r)[0]
        if img is None:
            raise RuntimeError(f'{gid}/{sess.cam_names[ci]}: frame {f} unreadable')
        dec = (img.shape[1], img.shape[0])
        # A video root IGNORES `reduce`, so the frame comes back full size -- both are legal, and
        # what is not legal is a frame that matches neither. The box transform is derived from the
        # rig's recorded size, so that would letterbox the boxes and the pixels differently and
        # say nothing about it.
        want = tuple(-(-size[a] // r) for a in (0, 1))                  # libjpeg rounds up
        assert dec == want or dec == size, \
            f'{gid}/{sess.cam_names[ci]} frame {f}: decoded {dec}, expected {want} at reduce={r} '\
            f'or {size} unreduced'
        d = size[0] / dec[0]                     # decoded pixels -> source pixels, 1.0 for video
        if warp is None and self.origins[i] is None:
            img, _, _ = letterbox(img, self.input_wh, src_wh=size)
        else:
            # ONE warpAffine for the decode scale, the augmentation AND the letterbox-or-tile.
            # Three would resample rat-city's frame three times, and the loader is the expensive
            # half of an iteration. `L` is whichever transform `_transform` chose, which is what
            # makes a tile cost no extra resample and no extra code -- and `borderValue` is already
            # what a tile hanging off the frame edge should see.
            scale, pad = self._transform(i, size)
            L = np.array([[scale, 0.0, pad[0]], [0.0, scale, pad[1]], [0.0, 0.0, 1.0]], np.float32)
            D = np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, 1.0]], np.float32)
            W = np.vstack([warp, [0, 0, 1]]) if warp is not None else np.eye(3, dtype=np.float32)
            M = (L @ W @ D)[:2]
            img = cv2.warpAffine(img, M, self.input_wh, borderValue=(114, 114, 114))
            if warp is not None:
                img = _photometric(img, rng)
        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # `regions` rides along as its own element rather than being folded into `boxes`: a region
        # is not an animal and must never reach `assign`. Appended only under `use_regions`, so
        # without it the item shape is exactly what every recorded detector trained on.
        #
        # `regions_for` returning None means the session claims exhaustive labelling, and the
        # faithful encoding of that HERE is one rect covering the whole input -- then "certified"
        # is all-True and the masked loss equals the unmasked one, with no special case anywhere
        # downstream. The None/empty distinction stays intact in `regions_for`, which is where a
        # consumer that needs it can see it.
        out = (x, boxes) if kpts is None else (x, boxes, kpts)
        if not self.use_regions:
            return out
        if regions is None:
            regions = torch.tensor([[0.0, 0.0, float(self.input_wh[0]), float(self.input_wh[1])]])
        return (*out, regions)


class ChunkShuffle(torch.utils.data.Sampler):
    """A shuffle that keeps a worker inside a few videos at a time.

    `BoxDataset.index` is built session-by-session, so a run of contiguous index positions stays
    inside one video. A plain `shuffle=True` therefore sends every item to a different container:
    on calms21's 63 one-mp4 sessions that thrashes `dataset._reader`'s cache and costs 486 ms per
    batch of 16, against a 16 ms GPU step. Shuffling BLOCKS instead, and pooling `mix` of them so
    a batch still spans several sessions, costs 40 ms -- 12x faster, and 7 sessions per batch.
    (That diversity was originally about BatchNorm not normalising one animal's lighting;
    normalisation is GroupNorm now and does not care, but a batch drawn from one video is still a
    correlated gradient step, and the 12x is the reason this class exists either way.)

    PASS `chunk=dataset.chunk`. The default 512 was set against rat-city, whose whole split is one
    group; a dataset with 40 index positions per video gets 13 videos to a block and 52 to a pool,
    which is not locality at all. `BoxDataset` derives the right value from its own index.

    Image-directory datasets do not need this and are not harmed by it: their frames are separate
    files, so locality buys nothing and costs nothing.
    """

    def __init__(self, n, chunk=512, mix=4, seed=0):
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
