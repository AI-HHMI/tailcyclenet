"""Box training data: labels -> a box target, letterboxed into the detector's input.

The target is derived from the same points the pose loader crops on, so the detector reproduces
the crop the pose model was trained on. Under `box_target='crop'` (legacy) it IS
`tailcyclenet.crop.crop_box_for_points`; under `'extent'` (shipped) it is the tight extent and
the crop rule -- whose pad and floor are SOURCE pixels, invisible after a letterbox -- is applied
after detection (`detect_raw`), which yields the identical crop. An animal with no
finite point in a view gets a NaN box, not a dropped frame -- objectness still has to learn "no
animal here". `box_source='instances'` takes the extent from `instances.pq` instead, opt-in,
for a dataset whose stored keypoints are too sparse to bound the animal.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import Dataset

from tracktail.model.cube import project_points_torch

from ..data.crop import BOX_SOURCES, crop_box_for_points
from ..data.camera import apply_affine
from ..data.pixels import read_frames
from ..format import (INST_ABSENT, INST_LABELED, INST_NONE, INST_PRESENT, UNLABELED,
                      load_datasets)
from .augment import (additive_noise, apply_cutout, color_jitter, cutout_rects, grayscale,
                      motion_blur, photometric, random_affine, salt_pepper)
from .letterbox import letterbox, letterbox_transform, reduce_factor, tile_transform, warp_image
from .sampling import IndexWeights


# The smallest TIGHT-extent side (INPUT pixels) a `box_target = 'extent'` target is floored at,
# symmetrically about its centre. A one- or two-point extent is degenerate (zero area), and TAL
# only takes anchors strictly inside a box; 8 px holds 2x2 stride-4 anchors. Resolution-free by
# construction -- unlike `min_crop_dim`, which is SOURCE pixels and is applied after detection.
MIN_EXTENT_PX = 8.0

BOX_TARGETS = ('crop', 'extent')


def _drop_outside(x, bounds):
    """(...,2) points -> the same shape, non-finite where outside `bounds` = (lo_x,lo_y,hi_x,hi_y).

    A point warped off the region it belongs to is not that point any more: dropping it shrinks a
    box to its visible part, and `crop_box_for_points` returns None when every point of an animal
    is gone. `boxes_for` uses this one rule for every box's geometry.
    """
    lo_x, lo_y, hi_x, hi_y = bounds
    out = ((x[..., 0] < lo_x) | (x[..., 0] > hi_x) |
          (x[..., 1] < lo_y) | (x[..., 1] > hi_y))
    return torch.where(out[..., None], torch.nan, x)


def _extent(points, size):
    """(...,2) points -> float (4,) tight xyxy extent of the finite ones, clamped into the frame,
    or None when none is finite. The crop rule's own min/max/clamp with no pad, square or floor.
    """
    pts = points.reshape(-1, 2)
    pts = pts[torch.isfinite(pts).all(-1)]
    if not pts.shape[0]:
        return None
    lim = torch.as_tensor(size, dtype=torch.float32).reshape(2)
    lo = torch.minimum(torch.clamp(pts.min(0).values, min=0.0), lim)
    hi = torch.minimum(torch.clamp(pts.max(0).values, min=0.0), lim)
    return torch.cat([lo, hi])


def _floor_extent(boxes, min_side):
    """Grow each (N,4) box symmetrically to at least `min_side` per axis; NaN rows stay NaN."""
    out = boxes.clone()
    for a in (0, 1):
        grow = torch.clamp(min_side - (out[:, a + 2] - out[:, a]), min=0.0) / 2
        out[:, a] = out[:, a] - grow
        out[:, a + 2] = out[:, a + 2] + grow
    return out


def _overlaps_any(region, boxes) -> bool:
    """Whether the (x0, y0, x1, y1) `region` intersects any row of `boxes` (N, 4) with positive
    area."""
    if not len(boxes):
        return False
    x0, y0, x1, y1 = region
    iw = (torch.minimum(boxes[:, 2], torch.tensor(float(x1)))
          - torch.maximum(boxes[:, 0], torch.tensor(float(x0))))
    ih = (torch.minimum(boxes[:, 3], torch.tensor(float(y1)))
          - torch.maximum(boxes[:, 1], torch.tensor(float(y0))))
    return bool(((iw > 0) & (ih > 0)).any())


class BoxDataset(IndexWeights, Dataset):
    """One item = one camera view of one frame, with every animal's crop box in it.

    Deliberately per-view and per-frame rather than per-window: a box predictor has no temporal
    model, and giving it 24 near-identical frames as one item would just correlate the batch.
    """

    def __init__(self, path, split: str, input_wh=(416, 416), min_crop_dim=64,
                 crop_pad_px=20.0, crop_pad_frac=float('inf'),
                 max_frames_per_group: int = 40, seed: int = 23, box_source='keypoints',
                 augment=False, reduce=False, hflip=0.5, rotate_deg=0.0,
                 tile_wh=None, tile_scale=1.0, tile_bg_per_frame=1, strong=False,
                 boxes_by_dataset=None, include_roots=None, datasets=None,
                 box_target='crop', antialias=False, grayscale_prob=0.0, vflip=0.0,
                 input_norm='none', scale_range=(0.8, 1.25), background_prob=0.0,
                 invert_prob=0.0):
        """Build the per-view/per-frame index of labelled items for one or more dataset roots.

        Every opt-in lever defaults to OFF, so an arm moves one key at a time. Absent-only frames
        give all-negative examples; present-only frames need a positive; no-box frames are omitted.

        Inputs:
            path, split -- dataset root and split directory.
            datasets -- optional pre-loaded all-splits Dataset list (as `load_datasets(path)`),
                so several BoxDatasets share one set of Sessions and their label caches. It
                is narrowed to `split` (same Session objects), so the pickle a loader worker
                receives carries no other split.
            input_wh -- model input size; replaced by the tile size under tiling.
            tile_wh / tile_scale / tile_bg_per_frame -- tiling: the tile is the model's INPUT
                size; `tile_scale` is the source -> input scale and the only scale there is.
            reduce -- a KEY, not a loader detail: changes which source pixels reach the model.
            box_target -- 'crop' (legacy): `crop_box_for_points` itself. 'extent': the tight
                extent, floored at MIN_EXTENT_PX input px; `detect_raw` applies the rule.
            antialias / grayscale_prob / vflip -- area-filtered shrinking and the
                train-augmentation probabilities.
            input_norm / invert_prob / scale_range / background_prob -- `augment` (report 73).
            max_frames_per_group -- per-group cap (0 = uncapped); train passes 0 (the draw is
                weighted instead), val passes `val_frames_per_group`.
        Outputs:
            Builds `self.index`, `self.origins` (tile origin or None, parallel to `index`), and
            `self.chunk` (one (group, camera) file's positions, `ChunkShuffle`'s locality block).
            Views with neither a positive box nor supervised negatives are omitted.
        """

        assert box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {box_source!r}'
        self.box_source = box_source
        self.box_sources = dict(boxes_by_dataset or {})
        bad_sources = {name: value for name, value in self.box_sources.items()
                       if value not in BOX_SOURCES}
        if bad_sources:
            raise ValueError(f'boxes_by_dataset values must be one of {BOX_SOURCES}, '
                             f'got {bad_sources}.')
        self.tile_wh = None if tile_wh is None else tuple(int(v) for v in tile_wh)
        self.tile_scale = float(tile_scale)
        self.tile_bg_per_frame = int(tile_bg_per_frame)
        if self.tile_wh is not None and self.tile_scale <= 0:
            raise ValueError(f'tile_scale must be > 0, got {tile_scale}')
        self.hflip = float(hflip)
        if box_target not in BOX_TARGETS:
            raise ValueError(f'box_target must be one of {BOX_TARGETS}, got {box_target!r}')
        self.box_target = box_target
        self.antialias = bool(antialias)
        from .augment import INPUT_NORMS
        if input_norm not in INPUT_NORMS:
            raise ValueError(f'input_norm must be one of {INPUT_NORMS}, got {input_norm!r}')
        self.input_norm = input_norm
        self.invert_prob = float(invert_prob)
        self.scale_range = tuple(float(v) for v in scale_range)
        self.background_prob = float(background_prob)
        self.grayscale_prob = float(grayscale_prob)
        self.vflip = float(vflip)
        self.rotate_deg = float(rotate_deg)
        self.augment = augment
        self.strong = bool(strong)
        self.reduce = reduce
        self.seed = seed
        self._visits = 0
        all_datasets = load_datasets(path) if datasets is None else datasets
        split_datasets = (load_datasets(path, split=split) if datasets is None else
                          [replace(ds, sessions={split: ds.sessions[split]})
                           for ds in all_datasets if split in ds.sessions])
        if include_roots is None:
            self.datasets = split_datasets
        else:
            include_roots = set(include_roots)
            all_root_names = {ds.name for ds in all_datasets}
            unknown_roots = include_roots - all_root_names
            if unknown_roots:
                raise ValueError(f'include_roots names unknown dataset roots '
                                 f'{sorted(unknown_roots)}.')
            self.datasets = [ds for ds in split_datasets if ds.name in include_roots]
            if not self.datasets:
                raise ValueError(f'{path}: split {split!r} has no included roots with data')
        self.ds = self.datasets[0] if len(self.datasets) == 1 else None
        self.root_names = [ds.name for ds in self.datasets]
        all_root_names = {ds.name for ds in all_datasets}
        unknown = set(self.box_sources) - all_root_names
        if unknown:
            raise ValueError(f'boxes_by_dataset names unknown dataset roots {sorted(unknown)}; '
                             f'available roots are {sorted(all_root_names)}.')
        self.session_dataset = {sess.path: ds.name for ds in self.datasets
                                for sess in ds.all_sessions()}
        prediction_sessions = [s.path for ds in all_datasets for s in ds.all_sessions()
                               if s.prediction_session]
        if prediction_sessions:
            raise ValueError('prediction sessions are not detector-training annotations: '
                             + ', '.join(map(str, prediction_sessions[:5])))
        self.input_wh = tuple(input_wh) if self.tile_wh is None else self.tile_wh
        self.min_crop_dim = min_crop_dim
        self.crop_pad_px, self.crop_pad_frac = float(crop_pad_px), float(crop_pad_frac)
        self.train = split == 'train'
        rng = np.random.default_rng(seed)

        self.origins: list = []
        self.index = []
        self.root_ids: list[int] = []
        self.box_only_fallback_sessions = set()
        for root_id, ds in enumerate(self.datasets):
            before = len(self.index)
            for sess in ds.sessions.get(split, []):
                sess.preload()
                for gid, group in sess.groups.items():
                    lab = sess.labels(gid)
                    vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
                    box_only = lab.points2d is None and lab.points3d is None
                    if (self._box_source_for(sess) == 'keypoints' and box_only
                            and self._uses_instance_boxes(sess, lab)):
                        positive = (lab.instance == INST_LABELED) | (lab.instance == INST_PRESENT)
                        if (positive & np.isfinite(lab.boxes).all(-1)).any():
                            self.box_only_fallback_sessions.add(sess.path)
                    frames = np.zeros(group.n_frames, dtype=bool)
                    if vis is not None and vis.shape[0]:
                        v = vis.reshape(vis.shape[0], vis.shape[1], -1)
                        frames |= (v != UNLABELED).any((0, 2))
                    if lab.instance is not None:
                        frames |= (lab.instance != INST_NONE).any((0, 2))
                    frames = np.flatnonzero(frames)
                    if max_frames_per_group and frames.size > max_frames_per_group:
                        keep = rng.choice(frames, max_frames_per_group, replace=False)
                        frames = np.asarray(sorted(keep.tolist()), dtype=np.int64)
                    for f in sorted(frames):
                        for ci in range(len(sess.rig)):
                            origins = ([None] if self.tile_wh is None
                                       else self._tile_origins(sess, gid, int(f), ci, rng))
                            if not self._has_target(sess, gid, int(f), ci, lab):
                                continue
                            for o in origins:
                                self.index.append((sess, gid, int(f), ci))
                                self.origins.append(o)
                                self.root_ids.append(root_id)
            count = len(self.index) - before
            if count == 0:
                raise ValueError(f'{ds.root}: split {split!r} has no usable supervised views')
        if not self.index:
            raise ValueError(f'{path}: split {split!r} has no labelled frames')
        n_src = len({(str(s.path), g, c) for s, g, _, c in self.index})
        self.chunk = max(1, len(self.index) // n_src)

    def __len__(self):
        """Number of indexed items (one camera view of one frame each)."""
        return len(self.index)

    def _box_source_for(self, sess):
        """Effective box source for a session's owning dataset root."""
        name = self.session_dataset.get(sess.path)
        return self.box_sources.get(name, self.box_source)

    def _uses_instance_boxes(self, sess, lab):
        """Resolve the box source per session, including box-only fallback.

        The configured source remains authoritative where pose labels exist. A session with no
        2D or 3D pose layer cannot produce keypoint boxes, so its `instances.pq` boxes are used
        even when the split-wide setting says `keypoints`.
        """
        has_instances = lab.boxes is not None and lab.instance is not None
        box_only = lab.points2d is None and lab.points3d is None
        return has_instances and (self._box_source_for(sess) == 'instances' or box_only)

    def negative_supervision_for(self, sess, gid, f, ci, lab=None):
        """Whether background anchors are known negatives for this camera-frame.

        Without `instances.pq`, the session is exhaustively labelled (spec §9). With an
        instance table, a `present` row (even with no box) marks the frame incomplete and disables
        all background negatives. Otherwise at least one explicit `labeled` or `absent` row is
        required; an unrecorded frame makes no negative claim. When instance boxes are the target
        source, a `labeled` row without a finite box is an animal with no target, so it disables
        negatives too rather than teaching its pixels to be background.
        """
        lab = sess.labels(gid) if lab is None else lab
        if lab.instance is None:
            return True
        status = lab.instance[:, f, ci]
        if (status == INST_PRESENT).any():
            return False
        if self._uses_instance_boxes(sess, lab):
            boxed = np.isfinite(np.asarray(lab.boxes[:, f, ci])).all(-1)
            if ((status == INST_LABELED) & ~boxed).any():
                return False
        return bool(((status == INST_LABELED) | (status == INST_ABSENT)).any())

    def _has_target(self, sess, gid, f, ci, lab):
        """Whether this (frame, view) yields a positive target or supervised negatives."""
        if self._uses_instance_boxes(sess, lab):
            positive = ((lab.instance[:, f, ci] == INST_LABELED) |
                        (lab.instance[:, f, ci] == INST_PRESENT))
            finite = np.isfinite(np.asarray(lab.boxes[:, f, ci])).all(-1)
            if (positive & finite).any():
                return True
        elif self._positioned_views(sess, gid, lab)[f, ci]:
            return True
        if lab.instance is None or (lab.instance[:, f, ci] == INST_LABELED).any():
            return False
        return self.negative_supervision_for(sess, gid, f, ci, lab)

    # ------------------------------------------------------------------------------------------
    # tiling
    # ------------------------------------------------------------------------------------------

    def _tile_extent(self):
        """The tile's extent in SOURCE pixels. `input_wh / tile_scale`, and nothing else.

        Every caller that needs a tile origin is gated on `self.origins[i] is not None`.
        """
        wh = self.tile_wh if self.tile_wh is not None else self.input_wh
        return (wh[0] / self.tile_scale, wh[1] / self.tile_scale)

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

    def _tile_origins(self, sess, gid, f, ci, rng):
        """Source-pixel tile origins for one (frame, camera).

        One tile per animal, plus `tile_bg_per_frame` background tiles. Origins are JITTERED by up
        to a quarter of the tile: a detector trained only on animals at dead centre has never seen
        one near a border, and `assign`'s centre prior would then be learned as a centre-of-image
        prior.

        Background tiles are drawn only where the camera-frame's background is a known negative
        (`negative_supervision_for`); on an incompletely labelled frame a background tile would
        carry no loss at all.

        A frame with no animal and no background tile still yields ONE tile.
        """
        W, H = (float(v) for v in sess.rig.size(sess.cam_names[ci]))
        tw, th = self._tile_extent()
        out = []

        def push(cx, cy):
            """Append a jittered, edge-clamped tile origin centred on (cx, cy).

            Clamped so a tile always overlaps the frame but NOT forced fully inside: a tile
            hanging off the frame edge is a real thing to train on, and `warpAffine`'s grey
            border is what a detector sees there.
            """
            jx, jy = rng.uniform(-0.25, 0.25, 2) * np.array([tw, th])
            out.append((float(np.clip(cx - tw / 2 + jx, -tw / 4, max(0.0, W - 3 * tw / 4))),
                        float(np.clip(cy - th / 2 + jy, -th / 4, max(0.0, H - 3 * th / 4)))))

        for c in self._animal_centres(sess, gid, f, ci):
            push(c[0], c[1])
        if self.negative_supervision_for(sess, gid, f, ci):
            for _ in range(self.tile_bg_per_frame):
                push(rng.uniform(0, W), rng.uniform(0, H))
        return out or [(max(0.0, (W - tw) / 2), max(0.0, (H - th) / 2))]

    def _animal_centres(self, sess, gid, f, ci):
        """(n,2) source-pixel centroids of the animals labelled in this view.

        The centroid, not the crop box: an origin only needs to be approximately on an animal, and
        running the crop rule for every animal of every frame at index-build time would pay for
        precision nothing here uses.
        """
        lab = sess.labels(gid)
        if (lab.points2d is None and lab.points3d is None
                and lab.instance is not None and lab.boxes is not None):
            boxes = np.asarray(lab.boxes[:, f, ci])
            positive = ((lab.instance[:, f, ci] == INST_LABELED) |
                        (lab.instance[:, f, ci] == INST_PRESENT))
            finite = np.isfinite(boxes).all(-1) & positive
            return [((b[:2] + b[2:]) / 2).tolist() for b in boxes[finite]]
        p2d = self._points_2d(sess, gid, f, ci).numpy()
        out = []
        for s in range(p2d.shape[0]):
            ok = np.isfinite(p2d[s]).all(-1)
            if ok.any():
                out.append(p2d[s][ok].mean(0))
        return out

    def _positioned_views(self, sess, gid, lab):
        """(T, C) bool: whether `_points_2d(sess, gid, f, ci)` has any finite point, for a group.

        The whole-group form of `_has_target`'s per-view check: one projection per camera per
        group. Shapes: points3d (S, T, K, 3) projects to (C, S, T, K, 2); points2d is (S, T, K, C, 2).

        Cached per group ON THE SESSION (pure label geometry), so the input-size probe, the
        train index and val share one set of projections.
        """
        cache = sess.__dict__.setdefault('_detector_positioned_cache', {})
        if gid in cache:
            return cache[gid]
        n_frames, n_cams = sess.groups[gid].n_frames, len(sess.rig)
        if sess.mode == '3d' and lab.points3d is not None:
            cams = (sess.cgroup(gid) if any(sess.rig.moving.values())
                    else [self._camera(sess, gid, 0, ci) for ci in range(n_cams)])
            pts = torch.as_tensor(lab.points3d, dtype=torch.float32)
            proj = project_points_torch(cams, pts)
            mask = torch.isfinite(proj).all(-1).any(-1).any(1).T.numpy()
        elif lab.points2d is not None:
            mask = np.isfinite(np.asarray(lab.points2d)).all(-1).any((0, 2))
        else:
            mask = np.zeros((n_frames, n_cams), dtype=bool)
        cache[gid] = mask
        return mask

    def _camera(self, sess, gid, f, ci):
        """One tracktail camera dict for `(frame, camera)`, read-only.

        A static rig's cameras are frame-invariant, so they are built once per session: rebuilding
        the whole rig per call was 81% of index-construction time on a 3D multi-root detector.
        A moving rig still goes through `Session.cgroup(gid, f)` for its per-frame extrinsics.
        """
        if any(sess.rig.moving.values()):
            return sess.cgroup(gid, f)[ci]
        cache = self.__dict__.setdefault('_static_cameras', {})
        key = str(sess.path)
        if key not in cache:
            cache[key] = sess.rig.tracktail()
        return cache[key][ci]

    def _points_2d(self, sess, gid, f, ci):
        """(S,K,2) source-pixel points for one (frame, camera). 3D sessions project.

        Shared by `boxes_for` and the tile-origin sampler so there is one copy of the
        "2D reads the table, 3D projects" branch. Frame-indexed, not whole-group: axis -3 of
        `pts` is the ANIMAL, so a moving camera's (T,4,4) extrinsic would project animal `i`
        through frame `i`'s pose.
        """
        lab = sess.labels(gid)
        if sess.mode == '3d' and lab.points3d is not None:
            cam = self._camera(sess, gid, f, ci)
            pts = torch.as_tensor(lab.points3d[:, f], dtype=torch.float32)
            return project_points_torch([cam], pts)[0]
        if lab.points2d is not None:
            return torch.as_tensor(lab.points2d[:, f, :, ci], dtype=torch.float32)
        return torch.full((len(lab.animal_ids), sess.n_keypoints, 2), float('nan'))

    def _transform(self, i, size):
        """The `(scale, (padx, pady))` for item `i`: its tile's, or the whole-frame letterbox.

        THE one place that choice is made; `boxes_for` and `ignore_for` both come here, because a
        rectangle transformed by a different rule than its own boxes is invisible in a loss curve.
        """
        if self.origins[i] is None:
            return letterbox_transform(size, self.input_wh)
        return tile_transform(self.origins[i], self.tile_scale)

    def ignore_for(self, i):
        """`(ig (S,) bool, ig_boxes (S,4))` for evaluation, in INPUT pixels.

        The `instances.pq` PRESENT rows are animals in this view that were not annotated, so a
        prediction on them is neither a true nor a false positive; scoring them as false positives
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

    def boxes_for(self, i, warp=None):
        """Return letterboxed (S,4) target boxes without decoding the image.

        The method is separate from `__getitem__` so assignment diagnostics can inspect targets
        without decoding pixels. `warp` is a source-pixel 2x3 augmentation: points move through it
        and boxes are re-derived by `crop_box_for_points`, rather than scaled.

        Points are frame-indexed (axis -3 is the animal), including moving-camera extrinsics.
        Points outside the frame or tile are dropped. An `instances.pq` box is an already-padded
        extent, re-entered with pad 0 and warped through all four corners.

        When `instances.pq` is the box source, finite `labeled` and `present` rows are positives;
        `absent` rows have no positive box. A `present` row also disables background-negative
        supervision for the entire view/frame, even when its box is null. Box-only sessions
        automatically use their instance boxes even when the configured source is `keypoints`.
        """
        sess, gid, f, ci = self.index[i]
        lab = sess.labels(gid)
        cam = self._camera(sess, gid, f, ci)
        p2d = self._points_2d(sess, gid, f, ci)

        tile_box = None
        if self.origins[i] is not None:
            ox, oy = self.origins[i]
            tw, th = self._tile_extent()
            tile_box = (ox, oy, ox + tw, oy + th)

        drop_outside = _drop_outside

        boxes = []
        use_instances = self._uses_instance_boxes(sess, lab)
        for s in range(p2d.shape[0]):
            if use_instances:
                b = torch.as_tensor(lab.boxes[s, f, ci], dtype=torch.float32)
                if (lab.instance[s, f, ci] not in (INST_LABELED, INST_PRESENT)
                        or not torch.isfinite(b).all()):
                    boxes.append(torch.full((4,), float('nan')))
                    continue
                x0, y0, x1, y1 = b
                src = torch.stack([torch.stack([x0, y0]), torch.stack([x1, y0]),
                                   torch.stack([x1, y1]), torch.stack([x0, y1])])
                pad = 0
            else:
                src, pad = p2d[s], self.crop_pad_px
            if warp is not None:
                src = apply_affine(src, (warp, None))
                src = drop_outside(src, (0.0, 0.0, float(cam['size'][0]), float(cam['size'][1])))
            if tile_box is not None:
                src = drop_outside(src, tile_box)
            if self.box_target == 'extent':
                box = _extent(src, cam['size'])
            else:
                box = crop_box_for_points(src, cam['size'], self.min_crop_dim, pad,
                                          self.crop_pad_frac)
            boxes.append(torch.full((4,), float('nan')) if box is None else box.float())
        boxes = torch.stack(boxes)
        scale, pad = self._transform(i, cam['size'])
        boxes[:, 0::2] = boxes[:, 0::2] * scale + pad[0]
        boxes[:, 1::2] = boxes[:, 1::2] * scale + pad[1]
        if self.box_target == 'extent':
            boxes = _floor_extent(boxes, MIN_EXTENT_PX)
        return boxes

    def _load_letterbox(self, i):
        """Decode item `i` and letterbox (or tile) it with NO augmentation whatsoever.

        `(img_uint8, boxes)`, both in INPUT pixels. Used only by mosaic-lite as a
        source of another frame's pixels and its own crop-rule box -- it must not recurse into
        `__getitem__`'s augmentation path, or a mosaic source could itself be mosaicked.
        """
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
            img, _, _ = letterbox(img, self.input_wh, src_wh=size, antialias=self.antialias)
        else:
            scale, pad = self._transform(i, size)
            L = np.array([[scale, 0.0, pad[0]], [0.0, scale, pad[1]], [0.0, 0.0, 1.0]], np.float32)
            D = np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, 1.0]], np.float32)
            M = (L @ D)[:2]
            img = warp_image(img, M, self.input_wh, self.antialias)
        return img, self.boxes_for(i, None)

    def _mosaic_paste(self, i, boxes, img, rng):
        """Copy-paste one WHOLE, untruncated crop-rule box from a DIFFERENT frame into empty
        space of this item's rendered image, at a location fully interior to `input_wh`.

        Not a 4-quadrant mosaic: a quadrant clips the animal on its seam, and a clipped extent is
        not a box the crop rule ever emits. "Empty" is tested: a placement whose pasted region
        intersects any of this item's own boxes is redrawn (it would cover a real animal whose box
        stays a positive target), and the item is returned unchanged if none fits.

        An `extent` target stops at the outermost POINT,
        not the animal's edge, so its paste carries a quarter-side margin of context.
        """
        if len(self) < 2:
            return boxes, img
        for _ in range(4):
            j = int(rng.integers(len(self)))
            if j == i:
                continue
            src_img, src_boxes = self._load_letterbox(j)
            finite = torch.isfinite(src_boxes).all(-1)
            if not bool(finite.any()):
                continue
            cand = torch.nonzero(finite).flatten().tolist()
            s = int(rng.choice(cand))
            b = src_boxes[s]
            region = b
            if self.box_target == 'extent':
                m = 0.25 * float(0.5 * ((b[2] - b[0]) + (b[3] - b[1])))
                region = torch.stack([(b[0] - m).clamp(min=0), (b[1] - m).clamp(min=0),
                                      (b[2] + m).clamp(max=self.input_wh[0]),
                                      (b[3] + m).clamp(max=self.input_wh[1])])
            x0, y0, x1, y1 = (int(v) for v in region.round().tolist())
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0 or bw > self.input_wh[0] or bh > self.input_wh[1]:
                continue
            own = boxes[torch.isfinite(boxes).all(-1)]
            for _ in range(8):
                dx = int(rng.integers(0, self.input_wh[0] - bw + 1))
                dy = int(rng.integers(0, self.input_wh[1] - bh + 1))
                if not _overlaps_any((dx, dy, dx + bw, dy + bh), own):
                    break
            else:
                continue
            img = img.copy()
            img[dy:dy + bh, dx:dx + bw] = src_img[y0:y1, x0:x1]
            offset = torch.tensor([dx - x0, dy - y0, dx - x0, dy - y0], dtype=boxes.dtype)
            boxes = torch.cat([boxes, (b + offset).to(boxes.dtype)[None]], 0)
            break
        return boxes, img

    def _visit_rng(self):
        """The augmentation generator of this process's next train visit (see `__getitem__`)."""
        info = torch.utils.data.get_worker_info()
        self._visits += 1
        return np.random.default_rng([int(self.seed), 0 if info is None else info.id + 1,
                                      self._visits])

    def __getitem__(self, i):
        """Decode item `i`: letterboxed/tiled pixels, boxes, and whether the view's background
        negatives are supervised.

        A train visit draws its augmentation from `(seed, worker, visit)`, so `[training].seed`
        fixes the stream for a given `num_workers` (the DataLoader hands batches to workers in a
        fixed order); evaluation draws nothing. Under tiling the
        input is the frame at `tile_scale`, not the tile size. One warpAffine composes decode
        scale, augmentation and letterbox-or-tile geometry. The strong suite is appearance-only
        except cutout, which withholds targets it covers.
        """
        sess, gid, f, ci = self.index[i]
        size = tuple(sess.rig.size(sess.cam_names[ci]))
        rng = self._visit_rng() if self.augment and self.train else None
        warp = (random_affine(size, rng, scale=self.scale_range, hflip=self.hflip,
                              rotate_deg=self.rotate_deg, centre=self._warp_centre(i),
                              vflip=self.vflip)
                if rng is not None else None)
        boxes = self.boxes_for(i, warp)
        negative_supervision = self.negative_supervision_for(sess, gid, f, ci)

        out_wh = (self.input_wh if self.tile_wh is None
                  else (size[0] * self.tile_scale, size[1] * self.tile_scale))
        r = reduce_factor(size, out_wh) if self.reduce else 1
        img = read_frames(sess.groups[gid], sess.cam_names[ci], [f], reduce=r)[0]
        if img is None:
            raise RuntimeError(f'{gid}/{sess.cam_names[ci]}: frame {f} unreadable')
        dec = (img.shape[1], img.shape[0])
        want = tuple(-(-size[a] // r) for a in (0, 1))
        assert dec == want or dec == size, \
            f'{gid}/{sess.cam_names[ci]} frame {f}: decoded {dec}, expected {want} at reduce={r} '\
            f'or {size} unreduced'
        if warp is None and self.origins[i] is None:
            img, _, _ = letterbox(img, self.input_wh, src_wh=size, antialias=self.antialias)
        else:
            scale, pad = self._transform(i, size)
            d = size[0] / dec[0]
            L = np.array([[scale, 0.0, pad[0]], [0.0, scale, pad[1]], [0.0, 0.0, 1.0]], np.float32)
            D = np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, 1.0]], np.float32)
            W = np.vstack([warp, [0, 0, 1]]) if warp is not None else np.eye(3, dtype=np.float32)
            img = warp_image(img, (L @ W @ D)[:2], self.input_wh, self.antialias)
            if warp is not None:
                img = photometric(img, rng)
                if self.strong:
                    if rng.random() < 0.5:
                        img = color_jitter(img, rng)
                    if rng.random() < 0.3:
                        img = additive_noise(img, rng)
                    if rng.random() < 0.2:
                        img = salt_pepper(img, rng)
                    if rng.random() < 0.2:
                        img = motion_blur(img, rng)
        if self.strong and rng is not None:
            if rng.random() < 0.5:
                rects = cutout_rects(self.input_wh, rng)
                img = apply_cutout(img, rects)
            if rng.random() < 0.2:
                boxes, img = self._mosaic_paste(i, boxes, img, rng)
        if rng is not None and self.invert_prob and rng.random() < self.invert_prob:
            from .augment import invert
            img = invert(img)
        if rng is not None and self.background_prob and rng.random() < self.background_prob:
            from .augment import composite
            img, boxes = composite(img, boxes, rng, self.input_wh,
                                   decoys_ok=bool(negative_supervision))
            negative_supervision = True
        if rng is not None and self.grayscale_prob and rng.random() < self.grayscale_prob:
            img = grayscale(img)
        if self.input_norm != 'none':
            from .augment import normalize_input
            img = normalize_input(img, self.input_norm)
        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return {'x': x, 'boxes': boxes, 'negative_supervision': negative_supervision}


