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

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.datasets.posetail_dataset import (custom_collate, load_image,
                                                rotate_camera_image_plane_3d,
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
    aug_prob: float = 0.25             # in-plane rotation + photometric
    crop_jitter: float = 0.3           # box centre jitter, fraction of box size
    crop_jitter_scale: float = 0.3     # box scale jitter
    min_crop_dim: int = 64
    prompt_dropout: float = 0.4        # fraction of keypoints whose prior is withheld
    val_stride: int = 0                # 0 -> non-overlapping windows for val/test


# ----------------------------------------------------------------------------------------------
# pixels
# ----------------------------------------------------------------------------------------------

def _read_video(path, frames, crop_coords, target_size, rotation):
    """Frames from a video file. Only 3dpop's test split needs this."""
    import cv2
    from decord import VideoReader

    vr = VideoReader(str(path))
    imgs = vr.get_batch(list(frames)).asnumpy()
    out = []
    for img in imgs:
        if rotation is not None:
            M, new_size = rotation
            img = cv2.warpAffine(img, M, new_size)
        if crop_coords is not None:
            x1, y1, x2, y2 = (int(c) for c in crop_coords)
            h, w = img.shape[:2]
            buf = np.zeros((y2 - y1, x2 - x1, img.shape[2]), img.dtype)
            sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if sx2 > sx1 and sy2 > sy1:
                buf[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
            img = buf
        if target_size is not None:
            img = cv2.resize(img, tuple(target_size))
        out.append(img)
    return out


def read_frames(group, cam, frames, crop_coords=None, target_size=None, rotation=None,
                pool: ThreadPoolExecutor | None = None):
    """(T,H,W,3) uint8 RGB for one camera, from an image directory or a video."""
    kind, src = group.pixels(cam)
    if kind == 'video':
        return _read_video(src, frames, crop_coords, target_size, rotation)
    names = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1] in ('.png', '.jpg'))
    paths = [os.path.join(src, names[i]) for i in frames]
    if pool is None:
        return [load_image(p, crop_coords, target_size, rotation) for p in paths]
    return [f.result() for f in
            [pool.submit(load_image, p, crop_coords, target_size, rotation) for p in paths]]


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
                 train: bool | None = None, seed: int = 0):
        self.cfg = cfg
        self.split = split
        self.train = (split == 'train') if train is None else train
        self.datasets = load_datasets(path)
        self.registry = registry or Registry.build(self.datasets)
        self.seed = seed

        # Scatter every session's parquet into dense arrays HERE, in the parent process, and drop
        # the tables. Forked workers then share the arrays copy-on-write instead of each holding
        # its own copy of a 44 MB table (12 workers x rat-city would be half a gigabyte).
        self.index: list[_Item] = []
        self.by_dataset: list[list[int]] = []
        for di, ds in enumerate(self.datasets):
            mine = []
            for sess in ds.sessions.get(split, []):
                sess.preload()
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
        augment = self.train and rng.random() < 1.0        # per-camera rolls happen below

        moving_ext = None
        if lab.ext is not None:
            moving_ext = {n: torch.as_tensor(lab.ext[i][frames], dtype=torch.float)
                          for i, n in enumerate(sess.cam_names) if sess.rig.moving[n]}
        cgroup = sess.rig.posetail(moving_ext=moving_ext)

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
            p2d = coords[None]
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
            p2d = project_points_torch(cgroup, coords) if single_view else None
            R = 3

        # -- pixels ----------------------------------------------------------------------
        with ThreadPoolExecutor(max_workers=16) as pool:
            views = []
            for cnum, cam_name in enumerate(cam_names):
                imgs = read_frames(group, cam_name, frames, crop_coords=boxes[cnum],
                                   target_size=cgroup[cnum]['size'].tolist(),
                                   rotation=rotation_info[cnum], pool=pool)
                if any(im is None for im in imgs):
                    return None
                views.append(torch.as_tensor(np.asarray(imgs), dtype=torch.float32) / 255.0)

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

        kpt_ids = torch.as_tensor(self.registry.ids_for(self.datasets[item.ds].name),
                                  dtype=torch.long)
        query_times = torch.zeros(K, dtype=torch.int32)
        query_occlusion = torch.full((K, len(cgroup)), -1, dtype=torch.int64)
        row = {'dataset': self.datasets[item.ds].name, 'session': sess.session_id,
               'group': item.gid, 'animal': lab.animal_ids[a], 'mode': '2d' if R == 2 else '3d',
               'single_view': single_view, 'start': int(frames[0]), 'cameras': cam_names}

        return (views, coords, vis, torch.as_tensor(frames), cgroup, row, query_times,
                vis_2d, p2d, query_occlusion, kpt_ids, kpt_prior, prompt_t)

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


def pose_collate(batch):
    """posetail's collate for the first ten fields, plus this repo's three.

    `custom_collate` keeps only item 0's `cgroup` and asserts a batch does not mix 2D and 3D --
    which is why batch_size is structurally 1 here and why there is no DDP.
    """
    batch = [b for b in batch if b is not None]
    out = custom_collate([b[:10] for b in batch])
    out['kpt_ids'] = torch.stack([b[10] for b in batch])
    out['kpt_prior'] = torch.stack([b[11] for b in batch])
    out['prompt_t'] = torch.stack([b[12] for b in batch])
    return out
