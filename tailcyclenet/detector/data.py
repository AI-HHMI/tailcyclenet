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
from ..format import PROJECTED, UNLABELED, VISIBLE, load_datasets


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


def random_affine(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5):
    """A random similarity about the image centre, source pixels in and out, as a 2x3.

    Deliberately a similarity and not YOLOX's shear-and-perspective: the target is
    `crop_box_for_points`, an AXIS-ALIGNED extent, and a shear turns a box into a parallelogram
    whose extent is no longer the crop rule's box for anything.

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
    cx, cy = w / 2, h / 2
    return np.array([[sx, 0.0, cx - sx * cx + rng.uniform(-translate, translate) * w],
                     [0.0, s, cy - s * cy + rng.uniform(-translate, translate) * h]], np.float32)


class BoxDataset(Dataset):
    """One item = one camera view of one frame, with every animal's crop box in it.

    Deliberately per-view and per-frame rather than per-window: a box predictor has no temporal
    model, and giving it 24 near-identical frames as one item would just correlate the batch.
    """

    def __init__(self, path, split: str, input_wh=(416, 416), min_crop_dim=64,
                 max_frames_per_group: int = 40, seed: int = 0, box_source='keypoints',
                 augment=False, reduce=False, keypoints=False, hflip=None):
        assert box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {box_source!r}'
        self.box_source = box_source
        # Opt-in, default off: with it absent this loader is byte-identical to what every recorded
        # detector was trained on. See `random_affine` for why it also kills the horizontal flip.
        self.keypoints = bool(keypoints)
        # `hflip=None` means "decide from `keypoints`", which is the safe default. It is separable
        # ONLY so a box-only CONTROL arm can match the keypoint arm's augmentation exactly: the
        # keypoint arm necessarily loses the flip, so a control that keeps it differs in two
        # levers and measures neither (eval rule 4).
        self.hflip = (0.0 if self.keypoints else 0.5) if hflip is None else float(hflip)
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
        self.input_wh = tuple(input_wh)
        self.min_crop_dim = min_crop_dim
        self.train = split == 'train'
        rng = np.random.default_rng(seed)

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
                        self.index.append((sess, gid, int(f), ci))
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

        if sess.mode == '3d':
            pts = torch.as_tensor(lab.points3d[:, f], dtype=torch.float32)      # (S,K,3)
            p2d = project_points_torch([cam], pts)[0]                            # (S,K,2)
            vis = None if lab.vis3d is None else lab.vis3d[:, f]                 # (S,K)
        else:
            p2d = torch.as_tensor(lab.points2d[:, f, :, ci], dtype=torch.float32)
            vis = None if lab.vis2d is None else lab.vis2d[:, f, :, ci]

        # The keypoint target rides the SAME warp and the SAME letterbox as the boxes below, and
        # is derived here so there is exactly one copy of that transform.
        kpts = None
        if with_keypoints:
            k = p2d.clone()
            if warp is not None:
                k = _apply_affine(k, (warp, None))
                w, h = float(cam['size'][0]), float(cam['size'][1])
                out = (k[..., 0] < 0) | (k[..., 0] > w) | (k[..., 1] < 0) | (k[..., 1] > h)
                k = torch.where(out[..., None], torch.nan, k)
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
                w, h = float(cam['size'][0]), float(cam['size'][1])
                out = (src[..., 0] < 0) | (src[..., 0] > w) | (src[..., 1] < 0) | (src[..., 1] > h)
                src = torch.where(out[..., None], torch.nan, src)
            box = crop_box_for_points(src, cam['size'], self.min_crop_dim, pad)
            boxes.append(torch.full((4,), float('nan')) if box is None else box.float())
        boxes = torch.stack(boxes)
        scale, pad = letterbox_transform(cam['size'], self.input_wh)
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
        # Per item and per epoch, off the item index -- a worker-local RNG would hand every
        # worker the same stream, and a shared one would make the draw depend on how the
        # DataLoader happened to interleave.
        rng = (np.random.default_rng([self.seed, i]) if self.augment and self.train else None)
        warp = random_affine(size, rng, hflip=self.hflip) if rng is not None else None
        got = self.boxes_for(i, warp, with_keypoints=self.keypoints)
        boxes, kpts = got if self.keypoints else (got, None)

        r = reduce_factor(size, self.input_wh) if self.reduce else 1
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
        if warp is None:
            img, _, _ = letterbox(img, self.input_wh, src_wh=size)
        else:
            # ONE warpAffine for the decode scale, the augmentation AND the letterbox. Three would
            # resample rat-city's frame three times, and the loader is the expensive half of an
            # iteration.
            scale, pad = letterbox_transform(size, self.input_wh)
            L = np.array([[scale, 0.0, pad[0]], [0.0, scale, pad[1]], [0.0, 0.0, 1.0]], np.float32)
            D = np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, 1.0]], np.float32)
            M = (L @ np.vstack([warp, [0, 0, 1]]) @ D)[:2]
            img = cv2.warpAffine(img, M, self.input_wh, borderValue=(114, 114, 114))
            img = np.clip(img * rng.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return (x, boxes) if kpts is None else (x, boxes, kpts)


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
    """
    xs = torch.stack([b[0] for b in batch])
    n = max(b[1].shape[0] for b in batch)
    boxes = torch.full((len(batch), n, 4), float('nan'))
    for i, b in enumerate(batch):
        boxes[i, :b[1].shape[0]] = b[1]
    if len(batch[0]) < 3:
        return xs, boxes
    K = batch[0][2].shape[1]
    kpts = torch.full((len(batch), n, K, 3), float('nan'))
    for i, b in enumerate(batch):
        kpts[i, :b[2].shape[0]] = b[2]
    return xs, boxes, kpts
