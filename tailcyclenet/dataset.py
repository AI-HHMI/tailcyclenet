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
from posetail.posetail.cube import get_camera_scale, is_point_visible, project_points_torch

from . import crop as cropmod
from .format import MISSING, PROJECTED, UNLABELED, VISIBLE, Registry, load_datasets


@dataclass
class LoaderConfig:
    """Everything the loader is allowed to vary. Deliberately short."""
    n_frames: int = 24
    image_size: int = 256              # cameras are resized so max(W,H) == this
    # An int, or a [low, high] pair drawn per item -- posetail's own `sample_cameras`
    # (`posetail_dataset.py:1258`). 0 means every camera. The pretrained tracker this finetunes
    # from was trained at [1, 8] (`config_encoder_3d_finetuning_h100.toml:22`), so a fixed count
    # above that is out of distribution as well as slow: johnson-mouse's 16-camera sessions ran
    # at 2.9 s/it against branson's 0.25, and s/it tracks camera count almost linearly.
    cams_to_sample: int | list = 0
    val_cams_to_sample: int | list = 5  # the reference's [dataset.val] value
    prob_2d_only: float = 0.25         # rate at which a 3D session is shown a single camera
    balance_datasets: bool = True      # sample datasets uniformly, not proportionally
    aug_prob: float = 0.25             # in-plane rotation, per-camera appearance, cutout
    per_image_aug_prob: float = 0.25   # per-FRAME appearance: motion blur, sensor noise
    grayscale_prob: float = 0.2        # rate at which a train item drops colour entirely
    crop_jitter: float = 0.3           # box centre jitter, fraction of box size
    crop_jitter_scale: float = 0.3     # box scale jitter
    min_crop_dim: int = 64
    prompt_dropout: float = 0.4        # fraction of TRAINING STEPS that run fully query-free
    prompt_noise_px: float = 0.0       # sigma on the prior, in PIXELS (3D scales by cube_scale)
    val_stride: int = 0                # 0 -> non-overlapping windows for val/test
    # Frame stride for a TRAIN window, drawn per item -- posetail's `interval`
    # (`posetail_dataset.py:343-361`). [1] is consecutive frames, i.e. no augmentation. Repeat an
    # entry to weight it: [1, 1, 2, 4] draws stride 1 half the time. Val/test are always 1.
    frame_strides: list = field(default_factory=lambda: [1])
    # Train sampling mix, a TWO-LEVEL draw: source first, then mode within that source. Either
    # level is skipped where a dataset offers no choice -- an all-tracked root ignores
    # `annot_frac`, a single-mode root ignores `mode_3d_frac` -- so one setting serves every root.
    # None leaves that level alone (entries keep their natural share); both None is uniform, which
    # is what every run before this change did. See `_pool_weights` for why this is load-bearing.
    annot_frac: float | None = None    # P(a step comes from an `annotated` session)
    mode_3d_frac: float | None = None  # P(3d | source), i.e. applied WITHIN each source


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
    #
    # DECODE EACH DISTINCT FRAME ONCE. `_frames` clamp-pads a window that runs past the end of its
    # group, and a group shorter than `n_frames` pads entirely -- 251 of johnson-mouse's 624 train
    # windows come from `n_frames = 1` groups, where all 24 indices are frame 0. Without the dedupe
    # that window decodes 24 copies of one image per camera (384 decodes for 16 distinct frames).
    want = [int(i) for i in frames]
    path = lambda i: os.path.join(src, f'{i:06d}{ext}')           # noqa: E731
    if pool is None:
        got = {i: load_image(path(i), crop_coords, target_size, rotation) for i in set(want)}
    else:
        fs = {i: pool.submit(load_image, path(i), crop_coords, target_size, rotation)
              for i in set(want)}
        got = {i: f.result() for i, f in fs.items()}
    if any(v is None for v in got.values()):
        return [got[i] for i in want]          # the caller checks for None and drops the item
    # A repeat gets a COPY, not the same array object. `_augment`'s cutout writes in place, so an
    # aliased list would have every repeat sharing one buffer. That is harmless today (imgaug
    # returns fresh arrays, and painting a constant rect twice is idempotent) but it is a trap not
    # worth leaving: the copy is ~20 us against the 27 ms decode it replaces.
    seen, out = set(), []
    for i in want:
        out.append(got[i].copy() if i in seen else got[i])
        seen.add(i)
    return out


# ----------------------------------------------------------------------------------------------
# appearance augmentation
# ----------------------------------------------------------------------------------------------

def _n_cams(spec, rng):
    """How many cameras this item shows. `spec` is an int, or a [low, high] pair drawn per item.

    The pair form is posetail's (`PosetailDataset.sample_cameras`, `posetail_dataset.py:1258-1266`)
    and it is what the pretrained tracker was finetuned with, at [1, 8]. Returning a number larger
    than the session has is fine -- the caller takes every camera in that case, exactly as the
    reference's `if len(cam_names) > num_cams_to_sample` guard does.
    """
    if isinstance(spec, (list, tuple)):
        lo, hi = int(spec[0]), int(spec[1])
        return int(rng.integers(lo, hi + 1))
    return int(spec)


def _even_span(span, ceiling):
    """Round a labelled span up to an even window length in [2, ceiling].

    EVEN because the scene encoder tokenises in tubelets of 2 (`vjepa2.py:103`,
    `gT = view.shape[1] // tubelet_size`), so an odd T silently drops a frame's worth of tokens.
    FLOOR OF 2 because T = 1 gives `gT = 0` and a zero-length pos_embed -- gotcha 1, the failure
    that cost the `memory` branch.
    """
    hi = max(2, int(ceiling) - int(ceiling) % 2)
    return min(max(2, int(span) + int(span) % 2), hi)


def _build_augmenters(cfg):
    """The two appearance pipelines, taken from the reference (`posetail_dataset.py:570-588`).

    The SPLIT is the point, not the list. `per_camera` is sampled once per camera and replayed
    frame by frame, so a camera's colour, gamma and focus hold steady down a clip: appearance is
    an identity cue for a tracker, and re-rolling hue every frame teaches that it is noise.
    `per_image` is resampled per frame, which is what sensor noise and motion blur actually are.

    Cost was last measured at 0.141 s for `per_camera` and 0.052 s for `per_image` on 24 crops of
    256x256 with every augmenter firing, but that predates dropping `DefocusBlur` -- which was
    4.2 ms/frame, the most expensive single entry -- so `per_camera` is now well under it. Watch
    `train/loader_wait_frac`; re-measure the same way before blaming augmentation for it.
    """
    import imgaug.augmenters as iaa

    p, q = cfg.aug_prob, cfg.per_image_aug_prob
    per_camera = iaa.Sequential([
        iaa.Sometimes(p, iaa.GammaContrast((0.6, 1.8))),
        iaa.Sometimes(p, iaa.AddToSaturation((-50, 30))),
        iaa.Sometimes(p, iaa.AddToHue((-10, 10))),
    ])
    per_image = iaa.Sequential([
        iaa.Sometimes(q, iaa.MotionBlur(k=(3, 5))),
        iaa.Sometimes(q, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
        iaa.Sometimes(q, iaa.Multiply((0.9, 1.1))),
    ])
    return per_camera, per_image


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

        # Sampling pools. A pool is a set of index positions plus an optional cumulative weight
        # array; `_pick` draws a pool, then an entry inside it. Balancing across datasets is the
        # only thing that makes more than one pool, and it is train-only -- val and test address
        # `self.index` directly so a window's identity stays tied to its index.
        multi = self.train and cfg.balance_datasets and len(self.datasets) > 1
        pools = self.by_dataset if multi else [list(range(len(self.index)))]
        self._pools = [(np.asarray(p, dtype=np.int64),
                        self._pool_weights(p) if self.train else None) for p in pools if p]

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

    def _pool_weights(self, pool):
        """Cumulative per-entry weights for one pool, or None to sample it uniformly.

        WHY THIS EXISTS. `_starts` returns one index entry per (session, group, animal) on train,
        whatever the group's length, and the sampler draws entries uniformly -- so an entry IS a
        sampling weight, and it is decoupled from how much data sits behind it. On
        allen-mouse-combined that put 90.4% of steps on 63 per-camera 2D sessions holding 1,023
        labelled frames between them, and 3.9% on the tracked session holding 21,500: a 500-frame
        tracked clip is one group, so it buys exactly one entry, the same price a single
        hand-annotated still pays. `mode='2d'` routes to head bank 0 and fires ~3 of 15 loss terms,
        so the 3D bank that validation reads was getting under a tenth of the gradient. On
        johnson-mouse-combined the tracked source sat at 0.3%.

        THE DRAW IS TWO-LEVEL: source (`annot_frac`), then mode WITHIN that source
        (`mode_3d_frac`), then uniform over the entries in that cell. The levels are independent
        by construction rather than jointly fitted, so they can never be mutually infeasible --
        `mode_3d_frac` is a conditional P(3d | source), not a second marginal.

        A level with nothing to choose between is skipped, which is what lets one setting serve
        every root: an all-tracked dataset ignores `annot_frac`, a single-mode one ignores
        `mode_3d_frac`, and a root that is both -- 3dpop, branson-fly, rat-city -- is untouched.
        A level left at None keeps its cells' natural entry shares, so it is not a silent
        rebalance.

        Returned flattened: the product of the two levels is one cumulative array and one
        `searchsorted`, which is the same distribution as drawing twice for less work.
        """
        cfg = self.cfg
        if cfg.annot_frac is None and cfg.mode_3d_frac is None:
            return None
        for name, v in (('annot_frac', cfg.annot_frac), ('mode_3d_frac', cfg.mode_3d_frac)):
            if v is not None and not 0.0 <= float(v) <= 1.0:
                raise ValueError(f'{name} must be in [0, 1], got {v}')

        cells: dict[tuple[str, str], list[int]] = {}
        for j in pool:
            sess = self.index[j].session
            cells.setdefault((sess.label_source, sess.mode), []).append(j)

        def share(key, present, frac, n):
            """P(key) across `present`: the configured fraction, or the natural entry share."""
            if len(present) == 1:
                return 1.0
            if frac is None:
                return n[key] / sum(n.values())
            return float(frac) if key == present[0] else 1.0 - float(frac)

        srcs = [s for s in ('annotated', 'tracked') if any(k[0] == s for k in cells)]
        n_src = {s: sum(len(v) for k, v in cells.items() if k[0] == s) for s in srcs}
        pos = {j: i for i, j in enumerate(pool)}
        w = np.zeros(len(pool), dtype=np.float64)
        for s in srcs:
            p_src = share(s, srcs, cfg.annot_frac, n_src)
            modes = [m for m in ('3d', '2d') if (s, m) in cells]
            n_mode = {m: len(cells[(s, m)]) for m in modes}
            for m in modes:
                js = cells[(s, m)]
                w[[pos[j] for j in js]] = (p_src * share(m, modes, cfg.mode_3d_frac, n_mode)
                                           / len(js))
        if w.sum() <= 0:
            raise ValueError('sampling weights are all zero -- check annot_frac / mode_3d_frac')
        return np.cumsum(w / w.sum())

    def mix(self):
        """Realised share of train steps per (label_source, mode) cell. Reporting only.

        Printed at startup because the mix is the single easiest thing here to get wrong by
        accident and the hardest to see afterwards: it is invisible in the loss curve, and the
        arithmetic that produces it lives in three places (index construction, dataset balancing,
        the two fractions).
        """
        out: dict[str, float] = {}
        for p, cum in self._pools:
            wt = np.diff(cum, prepend=0.0) if cum is not None else np.full(len(p), 1.0 / len(p))
            for j, x in zip(p, wt / len(self._pools)):
                sess = self.index[j].session
                out[f'{sess.mode}-{sess.label_source}'] = (
                    out.get(f'{sess.mode}-{sess.label_source}', 0.0) + float(x))
        return dict(sorted(out.items()))

    def _pick(self, idx, rng):
        # Val and test address the index directly -- a window's identity is its index, which is
        # what `test_val_windows_are_deterministic` rests on.
        if not self.train:
            return self.index[idx]
        # Uniform over datasets, then weighted within: without the outer level, branson-fly's 194
        # groups would outvote allen-mouse's 45 by 4:1 for no reason anyone chose.
        pool, cum = (self._pools[rng.integers(len(self._pools))] if len(self._pools) > 1
                     else self._pools[0])
        if cum is None:
            # Byte-for-byte the previous behaviour, so an unconfigured run is unchanged: one pool
            # replays `idx` straight from the sampler, many pools draw uniformly inside the pool.
            return self.index[idx if len(self._pools) == 1 else pool[rng.integers(len(pool))]]
        return self.index[int(pool[np.searchsorted(cum, rng.random())])]

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
        """T frame indices, clamp-padded so a short group still yields T >= 2.

        ON TRAIN, T IS DERIVED FROM THE LABELS, and `cfg.n_frames` is only its ceiling. The
        annotated sessions carry ONE labelled frame per 65-frame group, so a fixed T = 24 encodes
        24 frames to supervise 1 -- and with `annot_frac = 0.4` that is 40% of all steps paying 12x
        for nothing. Sizing the window to the labelled span makes those steps T = 2.

        This does NOT mean every encoded frame is supervised: two frames is the floor (gotcha 1 --
        a single-frame window gives `gT = 0` and a zero-length pos_embed), so one label still
        carries an unsupervised partner, and labels that are not adjacent force every frame
        between them to be encoded to span the pair. The claim is only that no frame is encoded
        which is not needed to reach a label.

        ON TRAIN THE FRAMES MAY ALSO BE STRIDED, by `cfg.frame_strides` -- posetail's `interval`
        (`posetail_dataset.py:343-361`), which this loader dropped when it moved to picking the
        start inside `__getitem__`. A stride of s widens the window to s times the wall time for
        the same T, so the model meets motion at more than one time scale. Everything below is the
        derived-T rule re-expressed on a LATTICE of spacing s: a strided window through the anchor
        can only reach labels congruent to it mod s, so the span is measured in lattice steps and
        the start is snapped onto the lattice.

        VAL AND TEST ARE UNTOUCHED -- `_starts` enumerates fixed `cfg.n_frames` windows there at
        stride 1, and a metric whose window geometry moved would not be comparable across
        checkpoints.
        """
        T = self.cfg.n_frames
        vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
        labelled = self._labelled_frames(vis, item.animal)
        if labelled.size == 0:
            return None
        s = 1
        if item.start >= 0:
            start = item.start
        else:
            # A stride is admissible if the group holds the FLOOR window of two frames at it; T
            # is then capped by what the group actually has room for at that stride. Testing the
            # full ceiling instead would reject stride exactly where it is most useful -- an
            # allen group is 65 frames with T = 24, and a 24-wide stride-4 window needs 93.
            fit = [x for x in self.cfg.frame_strides if x <= group.n_frames - 1]
            s = int(fit[rng.integers(len(fit))]) if fit else 1
            # Anchor on a labelled frame, then place the window around it. The old v4 loader
            # required the window's FIRST frame to be labelled, which silently discarded any
            # group whose labels sat in the middle; here the window moves to the label.
            anchor = int(labelled[rng.integers(labelled.size)])
            # Cap T at what the group holds ON THIS ANCHOR'S LATTICE -- the offset eats into the
            # room, and measuring from frame 0 instead lets the last frames clamp onto the end,
            # which reads as a shorter window rather than as the error it is.
            T = min(T, (group.n_frames - 1 - anchor % s) // s + 1)
            # Shrink T to the labelled frames this window could actually reach. `near` is every
            # label some placement of a full-width window over the anchor would cover, so
            # spanning first..last of them is the most T ever has to be. Off-lattice labels are
            # unreachable at this stride, so they are dropped before the span is measured.
            near = labelled[(labelled > anchor - T * s) & (labelled < anchor + T * s)]
            near = near[(near - anchor) % s == 0]
            first, last = int(near[0]), int(near[-1])
            T = _even_span((last - first) // s + 1, T)
            if last - first > (T - 1) * s:
                # The span is wider than the ceiling allows; no placement covers it, so fall back
                # to the anchor and let the draw pick which end of the span it lands on.
                first = last = anchor
            # Bounds that COVER first..last rather than merely containing the anchor -- sizing T
            # to a span and then placing the window off it would pay for frames it never reads.
            span = (T - 1) * s
            lo = max(0, last - span)
            hi = min(first, max(0, group.n_frames - 1 - span))
            lo += (anchor - lo) % s                 # snap up onto the anchor's lattice
            start = int(lo + s * rng.integers(0, (hi - lo) // s + 1)) if hi > lo else lo
        f = np.clip(np.arange(start, start + T * s, s), 0, group.n_frames - 1)
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
        else:
            n = _n_cams(self.cfg.cams_to_sample, rng)
            # sorted(), where the reference leaves the draw unsorted: camera order is arbitrary
            # either way, and a stable one makes a window comparable to itself across runs.
            cam_ix = (sorted(rng.choice(len(cgroup), n, replace=False)) if 0 < n < len(cgroup)
                      else list(range(len(cgroup))))
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
                # `projected` joins UNLABELED on the NaN side: it is a POSITION with no
                # visibility claim (johnson-mouse labels all 24 keypoints in all 16 views,
                # including ones the body hides), so training BCE on it would teach the head
                # "always visible" from 1.4M points that assert nothing.
                vis_2d = torch.as_tensor(np.where(np.isin(v2, (UNLABELED, PROJECTED)), np.nan,
                                                  (v2 == VISIBLE).astype(np.float32)))
                # 3D NOISY-OR: bool, and two-state by construction -- the loss inverts it with
                # `~` to build its occluded-point target (`losses.py:440`), which no float can
                # satisfy. That is the right semantics anyway: this layer answers "is the point
                # reconstructible in 3D", and where no camera assessed it there is no 3D label
                # either, so `False` is a fact rather than a guess.
                vis = torch.as_tensor((v2 == VISIBLE).any(-1))
                if not torch.isfinite(vis_2d).any():
                    # Nothing in this window carries a visibility ASSESSMENT -- every row is
                    # `projected`. There is no noisy-OR to take: all-False would assert that no
                    # point is reconstructible in 3D, which is a stronger lie than all-True.
                    # Drop to the 3dpop path and let the loss derive both masks geometrically.
                    vis = vis_2d = None
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
        # PER ITEM, NOT PER KEYPOINT. The reference draws one coin for the whole window
        # (`posetail_pose/model.py:619-621`, shape (B,1,1) broadcast over K), which is what makes
        # `prompt_dropout` the fraction of TRAINING STEPS that run fully query-free -- so one set
        # of weights serves both the first window of a clip and every window after it. Drawing
        # i.i.d. per keypoint instead put P(a fully unprompted window) at 0.4^47 ~ 1e-19: the
        # query-free forward, which is the path val and `best_mpjpe` score, was never trained.
        # It also left the prompted windows partially dense where `--anchor carry` supplies a
        # 100%-dense prior, so training and deployment disagreed on prompt density too.
        if self.train and rng.random() < self.cfg.prompt_dropout:
            kpt_prior[:] = float('nan')
        # EXPOSURE BIAS. The prior trains on exact GT and deploys as the model's own prediction;
        # an arm trained only on exact priors learns to trust a precision the carried signal does
        # not have. On rat-city, over the same 138 instance-windows, the (0, 0) control had carry
        # HURT by +1.815 mm while the (0.4, 5.7) recipe had it HELP by -0.749. NaN + noise is
        # still NaN, so a withheld prior stays withheld with no mask.
        #
        # SIGMA IS IN PIXELS, AND 3D CONVERTS. One scalar in the SESSION's own units cannot work:
        # `allen-mouse-combined` alone holds 63 px sessions beside 14 mm ones, so a single 1.0
        # meant a 1 px nudge on one and a 1 mm nudge on the other. `cube_scale` is world units per
        # pixel -- the same conversion `WeightedMAELoss` uses to enter the Huber in pixels
        # (`losses.py:1070-1080`) -- so scaling by it makes ONE setting mean the same visual
        # displacement on a 30 px fly and a 57,594-frame rat rig alike. Measured at the prior's own
        # position, through THIS window's cropped and resized cameras, so crop jitter is included.
        if self.train and self.cfg.prompt_noise_px > 0 and bool(torch.isfinite(kpt_prior).any()):
            sigma = float(self.cfg.prompt_noise_px)
            if R == 3:
                pts = kpt_prior[torch.isfinite(kpt_prior).all(-1)][None]     # (1,n,3)
                scale = torch.nanmedian(get_camera_scale(cgroup, pts))
                sigma *= float(scale)
            kpt_prior += torch.as_tensor(
                rng.normal(0.0, sigma, kpt_prior.shape), dtype=kpt_prior.dtype)

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

        per_camera, per_image = self._aug
        # to_deterministic() freezes this camera's sampled parameters so every frame gets the
        # SAME gamma/hue -- see `_build_augmenters`. Only the apply is per-frame.
        cam_det = per_camera.to_deterministic()
        imgs = [per_image(image=cam_det(image=im)) for im in imgs]
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
