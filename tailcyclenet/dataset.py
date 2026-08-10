"""The training loader: tailcycle-dataset on disk -> the batch posetail's model consumes.

Three sampling modes, decided per item:

- **3D multiview**   -- a `mode = "3d"` session, `cams_to_sample` cameras, targets in world mm
- **3D single-view** -- a 3D session shown ONE camera, targets still world mm, plus the 2D
  reprojection as `p2d`. Fired with `prob_2d_only`; this is exactly what posetail's own
  `prob_2d_only` does (`posetail_dataset.py:838-846, 971-975`) despite the name -- it subsets
  the cameras and adds `p2d`, and never converts `coords` to pixels. It is what teaches the
  model to recover metric 3D from a single view.
- **2D single-view** -- a `mode = "2d"` session: one camera, targets in crop pixels (R=2).

Mode is a property of the sampled SESSION, not of the run, so one `train/` may hold both, and
both head-bank slots (`mode_idx` 0 and 1) get gradient in one run.

Two rules that are not negotiable:

1. **Keypoints are never filtered.** The library's `filter_keypoints` drops keypoints seen by
   too few views, which shrinks N so array position stops equalling keypoint identity -- and
   nothing in the loss curve shows it. Every item here carries all K of its session's keypoints,
   in registry order.
2. **T >= 2, always.** posetail computes `gT = T // tubelet_size`, which is 0 at T=1 and yields a
   zero-length positional embedding (`encoder_decoder.py:748`). Short groups clamp-pad.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.datasets.posetail_dataset import (custom_collate, rotate_camera_image_plane_3d,
                                                rotate_points_image_plane)
from posetail.posetail.cube import is_point_visible, project_points_torch

from . import crop as cropmod
from .format import MISSING, UNLABELED, VISIBLE, Registry, load_datasets


@dataclass
class LoaderConfig:
    """Everything the loader is allowed to vary. Deliberately short."""
    n_frames: int = 24
    image_size: int = 256              # cameras are resized so max(W,H) == this
    cams_to_sample: int = 0            # 0 -> all cameras of a 3D session
    prob_2d_only: float = 0.25         # rate at which a 3D session is shown a single camera
    balance_datasets: bool = True      # sample datasets uniformly, not proportionally
    aug_prob: float = 0.25             # in-plane rotation, per-camera appearance, cutout
    per_image_aug_prob: float = 0.25   # per-FRAME appearance: motion blur, sensor noise
    grayscale_prob: float = 0.2        # rate at which a train item drops colour entirely
    crop_jitter: float = 0.3           # box centre jitter, fraction of box size
    crop_jitter_scale: float = 0.3     # box scale jitter
    min_crop_dim: int = 64
    prompt_dropout: float = 0.4        # fraction of keypoints whose prior is withheld
    val_stride: int = 0                # 0 -> non-overlapping windows for val/test


# ----------------------------------------------------------------------------------------------
# pixels
# ----------------------------------------------------------------------------------------------

def _crop_affine(src_wh, crop_coords, target_size, rotation):
    """The one dst<-src affine for rotate -> crop -> resize. Returns (M_2x3, (w, h)) or None.

    None means all three are no-ops and the caller should not warp at all (the detector reads
    whole frames).

    Composing them is not a micro-optimisation. Done in sequence, the rotation warps the WHOLE
    frame and the crop then throws >95% of it away: 44 ms per frame on rat-city's 4696x2048
    against 0.2 ms for the composed warp, plus an expanded rotation canvas and a zero-filled crop
    buffer that both disappear here. It also resamples once instead of twice.

    The composition uses the CORNER convention, `x_dst = (x_src - x1) * sx`, not `cv2.resize`'s
    half-pixel one. That is deliberate rather than incidental: `crop.apply_crop` sets
    `cam['offset'] += x1` and `_resize_camera` scales `cam['mat']`, which is exactly this affine on
    continuous coordinates -- so the pixels now agree with the intrinsics that
    `project_points_torch` reprojects through, where the old `resize` was off by half a pixel.
    `test_the_fused_warp_agrees_with_the_camera` is what holds that.

    Out-of-source pixels arrive as BORDER_CONSTANT zeros, which is what the old pad-safe crop
    buffer existed to produce. One behavioural difference, and it is unreachable in practice: a box
    reaching outside the rotated canvas used to be zero-filled there, and now samples the source
    instead (real pixels the inscribed-rect canvas had excluded). `crop_box_for_points` clamps
    every box to the camera's own size, so the loader cannot produce one.
    """
    w, h = src_wh
    M = np.eye(3)
    if rotation is not None:
        M_rot, (w, h) = rotation
        M[:2] = M_rot
    box = (0, 0, w, h) if crop_coords is None else tuple(int(c) for c in crop_coords)
    x1, y1, x2, y2 = box
    tw, th = ((x2 - x1, y2 - y1) if target_size is None
              else (int(target_size[0]), int(target_size[1])))
    if rotation is None and box == (0, 0, w, h) and (tw, th) == (w, h):
        return None
    sx, sy = tw / (x2 - x1), th / (y2 - y1)
    A = np.array([[sx, 0.0, -sx * x1], [0.0, sy, -sy * y1]])
    return (A @ M).astype(np.float32), (tw, th)


def load_image(path, crop_coords=None, target_size=None, rotation=None):
    """One decode and one affine -> (H,W,3) uint8 RGB. None if the file will not decode.

    Replaces the library's `load_image`, which did the rotation, the crop and the resize as three
    separate full-size buffers. BGR->RGB runs on the OUTPUT, which is the small one.
    """
    import cv2

    img = cv2.imread(path)
    if img is None:
        return None
    aff = _crop_affine((img.shape[1], img.shape[0]), crop_coords, target_size, rotation)
    if aff is not None:
        img = cv2.warpAffine(img, aff[0], aff[1], flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@lru_cache(maxsize=8)
def _reader(path: str):
    """One `VideoReader` per file per process. Opening the container and building its frame index
    is not per-window work, but `read_frames` is called once per window per camera -- so a
    windowed pass over 3dpop's test videos paid it hundreds of times. Small cache: the readers
    hold decode buffers, and inference walks one video at a time."""
    from decord import VideoReader

    return VideoReader(path)


def _read_video(path, frames, crop_coords, target_size, rotation):
    """Frames from a video file. Only 3dpop's test split needs this."""
    import cv2

    imgs = _reader(str(path)).get_batch(list(frames)).asnumpy()    # decord hands back RGB
    aff = _crop_affine((imgs.shape[2], imgs.shape[1]), crop_coords, target_size, rotation)
    if aff is None:
        return list(imgs)
    return [cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR) for im in imgs]


def read_frames(group, cam, frames, crop_coords=None, target_size=None, rotation=None,
                pool: ThreadPoolExecutor | None = None):
    """(T,H,W,3) uint8 RGB for one camera, from an image directory or a video."""
    kind, src, ext = group.source(cam)
    if kind == 'video':
        return _read_video(src, frames, crop_coords, target_size, rotation)
    # Names are computed, not listed. Frame files are `%06d.<ext>` contiguous from 000000 by spec
    # (§12, enforced by `validate_session`), and listing the directory to select T of them cost
    # 0.90 s of a 1.06 s rat-city item -- its `cam0` holds 57,594 entries.
    paths = [os.path.join(src, f'{i:06d}{ext}') for i in frames]
    if pool is None:
        return [load_image(p, crop_coords, target_size, rotation) for p in paths]
    return [f.result() for f in
            [pool.submit(load_image, p, crop_coords, target_size, rotation) for p in paths]]


# ----------------------------------------------------------------------------------------------
# appearance augmentation
# ----------------------------------------------------------------------------------------------

def _build_augmenters(cfg):
    """The two appearance pipelines, taken from the reference (`posetail_dataset.py:570-588`).

    The SPLIT is the point, not the list. `per_camera` is sampled once per camera and replayed
    frame by frame, so a camera's colour, gamma and focus hold steady down a clip: appearance is
    an identity cue for a tracker, and re-rolling hue every frame teaches that it is noise.
    `per_image` is resampled per frame, which is what sensor noise and motion blur actually are.

    Cost, measured on 24 crops of 256x256 with every augmenter firing: 0.141 s for `per_camera`
    (`DefocusBlur` is 4.2 ms/frame of it, the most expensive single entry) and 0.052 s for
    `per_image`. Watch `train/loader_wait_frac`; if it climbs, DefocusBlur is the first to drop.
    """
    import imgaug.augmenters as iaa

    p, q = cfg.aug_prob, cfg.per_image_aug_prob
    # DefocusBlur is kept OUT of the sequential because it cannot run on a small crop: the
    # imagecorruptions functions it wraps assert both sides >= 32 px (`imgcorruptlike.py:175`) and
    # raise otherwise, which inside a worker is a dead run rather than a skipped augmentation. A
    # crop is square-ish, so real data clears 32 comfortably -- but "comfortably" is not a
    # guarantee, and `_augment` gates on the actual crop instead of assuming.
    defocus = iaa.Sometimes(p, iaa.imgcorruptlike.DefocusBlur(severity=(1, 1)))
    per_camera = iaa.Sequential([
        iaa.Sometimes(p, iaa.GammaContrast((0.6, 1.8))),
        iaa.Sometimes(p, iaa.AddToSaturation((-50, 30))),
        iaa.Sometimes(p, iaa.AddToHue((-10, 10))),
    ])
    per_image = iaa.Sequential([
        iaa.Sometimes(q, iaa.MotionBlur(k=(3, 5))),
        iaa.Sometimes(q, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
        iaa.Sometimes(q, iaa.Multiply((0.9, 1.1))),
        iaa.Sometimes(q, iaa.SaltAndPepper(0.004)),
    ])
    return defocus, per_camera, per_image


def _cutout_rects(rng, size, p2d, vis_2d, cnum):
    """Random-erasing rectangles for one camera, in crop pixels. Mutates `vis_2d` in place.

    A keypoint underneath a rect is no longer visible, and saying so is the whole point: without
    it the model is asked to report "visible" for a patch that has been painted over, which is the
    one label that is definitely wrong.

    `vis_2d` here is THREE-state -- NaN means "no one assessed this camera". Cutout overwrites
    NaN with 0, and that is right rather than an invention: the pixels are now literally covered,
    so "not visible" became a fact about the image we just produced.
    """
    w, h = int(size[0]), int(size[1])
    rects = []
    for _ in range(int(rng.integers(1, 4))):
        rw, rh = int(w * 0.15), int(h * 0.15)
        rx = int(rng.integers(0, max(w - rw, 1)))
        ry = int(rng.integers(0, max(h - rh, 1)))
        rects.append((rx, ry, rx + rw, ry + rh, rng.integers(0, 256, 3).tolist()))
        if vis_2d is not None:
            pts = p2d[cnum]                                    # (T,K,2), crop pixels
            inside = ((pts[..., 0] >= rx) & (pts[..., 0] <= rx + rw) &
                      (pts[..., 1] >= ry) & (pts[..., 1] <= ry + rh))
            vis_2d[:, :, cnum][inside] = 0
    return rects


# ----------------------------------------------------------------------------------------------
# the dataset
# ----------------------------------------------------------------------------------------------

@dataclass
class _Item:
    """One addressable training unit: an animal in a group, optionally at a fixed start."""
    ds: int
    session: object
    gid: str
    animal: int
    start: int = -1                 # -1 -> pick at random (train)


class PoseDataset(Dataset):
    def __init__(self, path, split: str, cfg: LoaderConfig, registry: Registry | None = None,
                 train: bool | None = None, seed: int = 0,
                 registry_base: Registry | None = None):
        # Gotcha #1, and the clamp-pad in `_frames` does NOT cover it: that pads a short GROUP up
        # to `cfg.n_frames`, which does nothing when `cfg.n_frames` is itself 1. A 1-frame window
        # gives posetail `gT = T // tubelet_size = 0` and a zero-length positional embedding.
        assert cfg.n_frames >= 2, (
            f'n_frames = {cfg.n_frames} is not usable: posetail computes gT = T // tubelet_size '
            '(encoder_decoder.py:748), which is 0 at T=1 and yields a zero-length pos_embed. '
            'Use n_frames >= 2; short groups are clamp-padded up to it.')
        self.cfg = cfg
        self.split = split
        self.train = (split == 'train') if train is None else train
        self.datasets = load_datasets(path)
        # `registry_base` makes the ids APPEND-ONLY against a run that already exists, so the
        # embedding rows behind them survive a warm start. Without it a second run over the same
        # datasets in a different order silently remaps every row of `kpt_embed` -- gotcha #4,
        # and invisible in the loss curve. `Registry.build` raises if an old id would move.
        self.registry = registry or Registry.build(self.datasets, registry_base)
        self.seed = seed
        # Appearance augmentation is train-only, and `None` is also the flag the pixel path reads.
        # Val must stay clean: a metric computed on augmented pixels is not comparable to the last
        # one, and `test_val_windows_are_deterministic` would fail outright.
        self._aug = _build_augmenters(cfg) if self.train and cfg.aug_prob > 0 else None

        # Scatter every session's parquet into dense arrays HERE, in the parent process, and drop
        # the tables. Forked workers then share the arrays copy-on-write instead of each holding
        # its own copy of a 44 MB table (12 workers x rat-city would be half a gigabyte).
        self.index: list[_Item] = []
        self.by_dataset: list[list[int]] = []
        # Ids are per SESSION, not per dataset: a session may declare the root's keypoints in a
        # different order or use only a subset of them, and its dense K axis follows its OWN
        # `names`. Resolved here so a root that cannot be mapped fails at construction rather
        # than in the middle of an epoch.
        self._kpt_ids: dict[Path, torch.Tensor] = {}
        for di, ds in enumerate(self.datasets):
            mine = []
            for sess in ds.sessions.get(split, []):
                sess.preload()
                self._kpt_ids[sess.path] = torch.as_tensor(
                    self.registry.ids_for(ds.name, sess.names), dtype=torch.long)
                for gid, group in sess.groups.items():
                    lab = sess.labels(gid)
                    vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
                    if vis is None:
                        continue
                    for a in range(len(lab.animal_ids)):
                        starts = self._starts(vis, a, group.n_frames)
                        for st in starts:
                            mine.append(len(self.index))
                            self.index.append(_Item(di, sess, gid, a, st))
            self.by_dataset.append(mine)
        if not self.index:
            raise ValueError(f'{path}: split {split!r} yielded no usable windows')

    # -- indexing ------------------------------------------------------------------------
    def _labelled_frames(self, vis, a):
        """Frames where this animal has at least one assessed keypoint."""
        v = vis[a]
        v = v.reshape(v.shape[0], -1) if v.ndim > 2 else v
        return np.flatnonzero((v != UNLABELED).any(-1))

    def _starts(self, vis, a, n_frames):
        """Window starts for one animal.

        Train indexes at animal granularity and picks the start inside `__getitem__`, so a
        57,594-frame rat-city group costs 12 index entries instead of 691,000. Val and test
        enumerate fixed windows so a metric is reproducible.

        THE FIRST FRAME OF A WINDOW NEED NOT BE LABELLED, on either path. The old v4 loader
        admitted a training window only if its first frame had a finite coordinate, so a group
        whose labels sat in the middle yielded zero windows -- and the natural annotation shape,
        a label with context on both sides, was silently unusable. Here the window is placed
        around the label instead of the label being required at the window's edge.

        Windows are also clamped into the group rather than running off the end: a start beyond
        `n_frames - T` would be clamp-padded with duplicates of the last frame while real context
        sat unused earlier in the group.
        """
        labelled = self._labelled_frames(vis, a)
        if labelled.size == 0:
            return []
        if self.train:
            return [-1]
        T = self.cfg.n_frames
        stride = self.cfg.val_stride or T
        lo, hi = int(labelled[0]), int(labelled[-1])
        limit = max(0, n_frames - T)
        # Start half a window before the first label so the label sits inside the window rather
        # than at frame 0 -- frame 0 is the one frame where per-frame anchoring contributes
        # nothing, so putting every label there would measure the wrong thing.
        first = int(np.clip(lo - T // 2, 0, limit))
        return sorted({min(s, limit) for s in range(first, hi + 1, stride)}) or [0]

    def __len__(self):
        return len(self.index)

    def _pick(self, idx, rng):
        if not (self.train and self.cfg.balance_datasets and len(self.datasets) > 1):
            return self.index[idx]
        # Uniform over datasets, then uniform within: without this, branson-fly's 194 groups
        # would outvote allen-mouse's 45 by 4:1 for no reason anyone chose.
        pool = self.by_dataset[rng.integers(len(self.by_dataset))]
        return self.index[pool[rng.integers(len(pool))]]

    # -- item ----------------------------------------------------------------------------
    def __getitem__(self, idx):
        # Entropy-seeded on train so workers do not replay one another's augmentation;
        # index-seeded on val/test so a metric is reproducible.
        rng = np.random.default_rng(None if self.train else (self.seed, idx))
        for _ in range(8):
            out = self._item(idx, rng)
            if out is not None:
                return out
            idx = int(rng.integers(len(self.index)))
        raise RuntimeError(f'{self.split}: 8 consecutive items failed to build')

    def _frames(self, item, lab, group, rng):
        """T frame indices, clamp-padded so a short group still yields T >= 2."""
        T = self.cfg.n_frames
        vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
        labelled = self._labelled_frames(vis, item.animal)
        if labelled.size == 0:
            return None
        if item.start >= 0:
            start = item.start
        else:
            # Anchor on a labelled frame, then place the window around it. The old v4 loader
            # required the window's FIRST frame to be labelled, which silently discarded any
            # group whose labels sat in the middle; here the window moves to the label.
            anchor = int(labelled[rng.integers(labelled.size)])
            lo = max(0, anchor - T + 1)
            hi = min(anchor, max(0, group.n_frames - T))
            start = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        f = np.clip(np.arange(start, start + T), 0, group.n_frames - 1)
        return f

    def _item(self, idx, rng):
        item = self._pick(idx, rng)
        sess, group = item.session, item.session.groups[item.gid]
        lab = sess.labels(item.gid)
        frames = self._frames(item, lab, group, rng)
        if frames is None:
            return None
        a, T = item.animal, len(frames)
        K = sess.n_keypoints

        cgroup = sess.cgroup(item.gid, frames)

        true_2d = sess.mode == '2d'
        single_view = (not true_2d and self.train
                       and self.cfg.prob_2d_only > 0 and rng.random() < self.cfg.prob_2d_only)

        # -- pick cameras ----------------------------------------------------------------
        if true_2d:
            cam_ix = [0]
        elif single_view:
            cam_ix = [int(rng.integers(len(cgroup)))]
        elif self.cfg.cams_to_sample and self.cfg.cams_to_sample < len(cgroup):
            cam_ix = sorted(rng.choice(len(cgroup), self.cfg.cams_to_sample, replace=False))
        else:
            cam_ix = list(range(len(cgroup)))
        cgroup = [cgroup[i] for i in cam_ix]
        cam_names = [sess.cam_names[i] for i in cam_ix]

        # -- targets and visibility ------------------------------------------------------
        if true_2d:
            coords = torch.as_tensor(lab.points2d[a][frames][:, :, 0], dtype=torch.float32)
            vis = vis_2d = None
        else:
            coords = torch.as_tensor(lab.points3d[a][frames], dtype=torch.float32)
            if lab.vis2d is not None:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]            # (T,K,c), three-state
                # PER-CAMERA: three states, passed through as three states. NaN means "not
                # assessed", and posetail >= 0.3.2 masks it out of the visibility BCE so those
                # entries produce no gradient instead of being trained as "not visible". Under
                # 0.3.0 a NaN here silently returned NaN gradients for every parameter while the
                # loss curve looked healthy, and this had to be collapsed to two states.
                vis_2d = torch.as_tensor(np.where(v2 == UNLABELED, np.nan,
                                                  (v2 == VISIBLE).astype(np.float32)))
                # 3D NOISY-OR: bool, and two-state by construction -- the loss inverts it with
                # `~` to build its occluded-point target (`losses.py:440`), which no float can
                # satisfy. That is the right semantics anyway: this layer answers "is the point
                # reconstructible in 3D", and where no camera assessed it there is no 3D label
                # either, so `False` is a fact rather than a guess.
                vis = torch.as_tensor((v2 == VISIBLE).any(-1))
            else:
                # No per-camera assessment (3dpop): let the loss derive both masks
                # geometrically. vis and vis_2d are both-or-neither -- one without the other
                # dies inside einops.
                vis = vis_2d = None

        if torch.isfinite(coords).all(-1).sum() < 2:
            return None

        # -- geometry: rotate, crop, resize ----------------------------------------------
        rotation_info = [None] * len(cgroup)
        if true_2d:
            cam = cgroup[0]
            coords = _mask_outside(coords, cam['size'])
            if self.train and rng.random() < self.cfg.aug_prob:
                cam, coords, rot = rotate_points_image_plane(cam, coords,
                                                             float(rng.uniform(-45, 45)))
                rotation_info = [rot]
            jit = self._jitter(rng)
            cam, box, coords = cropmod.crop_to_points_2d(cam, coords, self.cfg.min_crop_dim, jit)
            if cam is None:
                return None
            cam, scale = _resize_camera(cam, self.cfg.image_size)
            coords = coords * scale
            coords = _mask_outside(coords, cam['size'])
            cgroup, boxes = [cam], [box]
            p2d = p2d_all = coords[None]
            R = 2
        else:
            # 3D path, one camera or several. Single-view differs ONLY in how many cameras are
            # shown -- the targets stay world-metric, which is the whole point.
            if self.train:
                rotated = []
                for cam in cgroup:
                    if rng.random() < self.cfg.aug_prob:
                        cam_r, rot = rotate_camera_image_plane_3d(cam,
                                                                  float(rng.uniform(-45, 45)))
                        rotated.append((cam_r, rot))
                    else:
                        rotated.append((cam, None))
                cgroup = [c for c, _ in rotated]
                rotation_info = [r for _, r in rotated]
                if vis_2d is not None:
                    for cnum, cam in enumerate(cgroup):
                        if rotation_info[cnum] is not None:
                            vis_2d[:, :, cnum][~is_point_visible(cam, coords)] = 0
            jit = self._jitter(rng)
            cgroup, boxes = cropmod.crop_to_points_3d(cgroup, coords, self.cfg.min_crop_dim, jit)
            if cgroup is None:
                return None
            cgroup = [_resize_camera(c, self.cfg.image_size)[0] for c in cgroup]
            if self.train:
                # Random world rotation, applied to points AND cameras together, so the model
                # cannot learn a fixed world gauge. Called unbound because the library's method
                # reads no instance state -- calling it is what keeps this from drifting.
                from posetail.datasets.posetail_dataset import PosetailDataset
                cgroup, coords = PosetailDataset.rotate_camera_group(None, cgroup, coords)
            # `p2d` is the reprojection AFTER crop/resize/rotate, so it lands in crop pixels.
            # Cutout needs it too, in the same frame -- projected once and shared, since a second
            # `project_points_torch` is a float64 reprojection of every point in the window.
            p2d_all = (project_points_torch(cgroup, coords)
                       if single_view or self._aug is not None else None)
            p2d = p2d_all if single_view else None
            R = 3

        # -- pixels ----------------------------------------------------------------------
        # Appearance augmentation runs HERE, on the final crops, not on the source frames: the
        # crops are ~256 px where a source frame can be 4696x2048, so it is the same augmentation
        # for a fraction of the work.
        gray = self._aug is not None and rng.random() < self.cfg.grayscale_prob
        with ThreadPoolExecutor(max_workers=16) as pool:
            views = []
            for cnum, cam_name in enumerate(cam_names):
                imgs = read_frames(group, cam_name, frames, crop_coords=boxes[cnum],
                                   target_size=cgroup[cnum]['size'].tolist(),
                                   rotation=rotation_info[cnum], pool=pool)
                if any(im is None for im in imgs):
                    return None
                if self._aug is not None:
                    imgs = self._augment(imgs, cnum, cgroup[cnum]['size'], p2d_all, vis_2d,
                                         gray, rng)
                # UINT8, not float32/255. The model divides on device (`model.py`), where it is
                # free, and this is 4x fewer bytes to collate, queue and pin -- 33 MB instead of
                # 132 MB for a 7-camera window, so 12 workers x prefetch 2 hold ~0.8 GB of pinned
                # host memory rather than ~3.2 GB.
                views.append(torch.from_numpy(np.asarray(imgs)))

        # -- the query prior -------------------------------------------------------------
        # kpt_prior is the pose at the prompt frame: at training time the GT, at deployment the
        # previous window's own prediction. prompt_t is the first frame each keypoint is
        # labelled at, which is NOT always 0 -- it was frame > 0 on 19.5% of rat-city windows.
        finite = torch.isfinite(coords).all(-1)                    # (T,K)
        prompt_t = torch.where(finite.any(0), finite.float().argmax(0), torch.zeros(K).long())
        prompt_t = prompt_t.to(torch.int32)
        kpt_prior = coords[prompt_t, torch.arange(K)].clone()      # (K,R)
        kpt_prior[~finite.any(0)] = float('nan')
        if self.train and self.cfg.prompt_dropout > 0:
            drop = torch.as_tensor(rng.random(K) < self.cfg.prompt_dropout)
            kpt_prior[drop] = float('nan')

        kpt_ids = self._kpt_ids[sess.path]      # aligned to THIS session's axis, not the root's
        query_times = torch.zeros(K, dtype=torch.int32)
        query_occlusion = torch.full((K, len(cgroup)), -1, dtype=torch.int64)
        row = {'dataset': self.datasets[item.ds].name, 'session': sess.session_id,
               'group': item.gid, 'animal': lab.animal_ids[a], 'mode': '2d' if R == 2 else '3d',
               'single_view': single_view, 'start': int(frames[0]), 'cameras': cam_names}

        return (views, coords, vis, torch.as_tensor(frames), cgroup, row, query_times,
                vis_2d, p2d, query_occlusion, kpt_ids, kpt_prior, prompt_t)

    def _augment(self, imgs, cnum, size, p2d, vis_2d, gray, rng):
        """Appearance augmentation for one camera's T crops. `vis_2d` is mutated by cutout."""
        import cv2

        defocus, per_camera, per_image = self._aug
        # to_deterministic() freezes this camera's sampled parameters so every frame gets the
        # SAME gamma/hue/blur -- see `_build_augmenters`. Only the apply is per-frame.
        cam_det = per_camera.to_deterministic()
        blur = defocus.to_deterministic() if min(imgs[0].shape[:2]) >= 32 else None
        imgs = [per_image(image=cam_det(image=im if blur is None else blur(image=im)))
                for im in imgs]
        if rng.random() < self.cfg.aug_prob:
            for x1, y1, x2, y2, fill in _cutout_rects(rng, size, p2d, vis_2d, cnum):
                for im in imgs:
                    im[y1:y2, x1:x2] = fill
        if gray:
            imgs = [np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)] * 3, -1) for im in imgs]
        return imgs

    def _jitter(self, rng):
        if not self.train or self.cfg.crop_jitter <= 0:
            return None
        return cropmod.jitter_box(rng, self.cfg.crop_jitter, self.cfg.crop_jitter_scale)


def _mask_outside(coords, size):
    """A point outside the image is not a label. Drops it rather than clamping it to the edge."""
    w, h = float(size[0]), float(size[1])
    bad = ((coords[..., 0] < 0) | (coords[..., 0] >= w) |
           (coords[..., 1] < 0) | (coords[..., 1] >= h))
    coords = coords.clone()
    coords[bad] = float('nan')
    return coords


def _resize_camera(cam, target_res):
    """Scale a camera so its long side is `target_res`. Returns (cam, scale)."""
    cam = dict(cam)
    size = cam['size']
    scale = float(target_res) / float(max(size))
    cam['size'] = torch.round(size * scale).to(torch.int32)
    cam['mat'] = cam['mat'] * scale
    cam['mat'][2, 2] = 1
    cam['offset'] = cam['offset'] * scale
    return cam, scale


def worker_init(worker_id):
    """A DataLoader `worker_init_fn`. Two things, both per-worker and both easy to miss.

    1. **Pin cv2's thread pool to one thread.** OpenCV sizes it to the machine -- 128 threads here
       -- and each of `num_workers` processes runs a 16-thread `ThreadPoolExecutor` on top of
       that, so the nesting is pure contention: the frames of a window are already being decoded
       in parallel. Measured 0.210 -> 0.180 s/it at 12 workers, and the gap widens on a smaller
       machine.

    2. **Reseed imgaug.** It keeps its OWN global RNG, which fork copies and nothing else
       reseeds, so every worker's k-th `to_deterministic()` would draw the same gamma, hue and
       blur -- the appearance diversity silently divides by `num_workers`, and the loss curve
       looks identical either way. The seed is drawn from `np.random`, which torch HAS already
       decorrelated per worker (`torch/utils/data/_utils/worker.py:261-265`).

    numpy itself is deliberately NOT reseeded here, for that same reason. The library's
    `make_worker_init_fn` exists to fold in a DDP rank, which torch's seeding ignores; there is
    no DDP here (batch_size is structurally 1), so reusing it would only downgrade torch's seed
    derivation to `base + worker_id`.
    `tests/test_dataset.py::test_workers_do_not_share_a_random_stream` pins both halves.
    """
    import cv2
    cv2.setNumThreads(1)
    try:
        import imgaug
    except ImportError:
        return
    imgaug.random.seed(int(np.random.randint(2 ** 31)))


def pose_collate(batch):
    """posetail's collate for the first ten fields, plus this repo's three.

    `custom_collate` keeps only item 0's `cgroup` and asserts a batch does not mix 2D and 3D --
    which is why batch_size is structurally 1 here and why there is no DDP. The same reason
    covers K: sessions may carry different keypoint subsets, so a batch that mixed two of them
    would fail the `kpt_ids` stack below. Loud, and unreachable at batch_size 1.
    """
    batch = [b for b in batch if b is not None]
    out = custom_collate([b[:10] for b in batch])
    out['kpt_ids'] = torch.stack([b[10] for b in batch])
    out['kpt_prior'] = torch.stack([b[11] for b in batch])
    out['prompt_t'] = torch.stack([b[12] for b in batch])
    return out
