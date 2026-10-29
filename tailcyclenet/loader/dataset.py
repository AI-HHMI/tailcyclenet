"""The pose training Dataset: tailcycle-dataset on disk -> the item tracktail's model consumes.

Three sampling modes, decided per item: 3D multiview (`cams_to_sample` cameras, world-mm targets),
3D single-view (one camera, world-mm targets, possible when `cams_to_sample` draws one), and 2D
single-view (crop-pixel targets, including the training-only `prob_2d_only` coin). Mode is a
property of the sampled SESSION except for that explicit 2D coin, so one `train/` may hold both
and both head-bank slots get gradient. Two non-negotiable rules: keypoints are never filtered
(the library's `filter_keypoints` shrinks N so array position stops equalling identity), and
T >= 2 always (a T=1 window gives tracktail a zero-length pos_embed).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tracktail.model.cube import is_point_visible, project_points_torch

from ..data import crop as cropmod
from ..data.camera import apply_affine, flip_2d, mask_outside, resize_camera, rotate_2d
from ..data.crop import BOX_SOURCES
from ..data.pixels import decode_errors, read_frames, warn_decode_failure
from ..format import MISSING, PROJECTED, UNLABELED, VISIBLE, Registry, load_datasets
from ..tracktail_compat import rotate_camera_image_plane_3d
from .augment import augment_views, build_augmenters, crop_inflate, n_cams, rotate_camera_group_with_neighbours
from .config import LoaderConfig
from .prior import corrupt_prior
from .sampling import shard_sessions


def _vis2d_target(v2):
    """The visibility TARGET from a raw per-camera `vis2d` status array. STATUS POLICY (CLAUDE.md):
    `projected` is a position with no visibility claim and must never reach a visibility target,
    so `unlabeled`/`projected` both NaN out; only `visible` reads as a positive 1.0.

    Inputs: v2 -- per-camera status codes, any shape.
    Outputs: a float32 tensor of the same shape, NaN where the status makes no visibility claim.
    Side effects: none.
    """
    return torch.as_tensor(np.where(np.isin(v2, (UNLABELED, PROJECTED)), np.nan,
                                    (v2 == VISIBLE).astype(np.float32)))


def _even_span(span, ceiling):
    """Round a labelled span up to an even window length in [2, ceiling].

    EVEN because the scene encoder tokenises in tubelets of 2 (`gT = T // tubelet_size`), so an odd
    T silently drops tokens; FLOOR OF 2 because T = 1 gives `gT = 0` and a zero-length pos_embed.
    """
    hi = max(2, int(ceiling) - int(ceiling) % 2)
    return min(max(2, int(span) + int(span) % 2), hi)


@dataclass
class _Item:
    """One addressable training unit: an animal in a group, optionally at a fixed start."""
    ds: int
    session: object
    gid: str
    animal: int
    # -1 -> pick at random (train)
    start: int = -1


@dataclass
class Selection:
    """Everything about a window that does NOT depend on how it is viewed.

    Exists so one selection can be realised into MORE than one view. A track-quality triplet
    needs the same track under two independent views, and calling `_item` twice would not give
    that: `_pick` re-draws the pool entry and `_frames` re-draws the anchor and stride, so the
    second call is a different sample. Deterministic given `(idx, rng, shape)`.

    `coords` is the CLEAN, pre-transform track (NaN where unlabelled), so a crop computed from
    it is the clean track's crop -- the one thing a corrupted view must NOT re-derive. `cgroup`
    is already subset to `cam_ix` and is pre-transform.
    """
    item: _Item
    sess: object
    group: object
    lab: object
    frames: np.ndarray
    animal: int
    n_keypoints: int
    shape: dict
    true_2d: bool
    single_view: bool
    cam_ix: list
    cam_names: list
    cgroup: list
    crop_pts: object
    coords: torch.Tensor
    vis: object
    vis_2d: object
    attempt_swap_animal: bool
    neighbour_row: object
    inflate: float


@dataclass
class View:
    """One realisation of a `Selection`: pixels, cameras and coords in ONE view's frame.

    `rotation`/`boxes`/`scale` are the raw pieces of the source->view affine, kept because a 2D
    consumer has to move coordinates BETWEEN two realisations of the same selection and the
    composed matrix is the only correct way to do it (re-deriving it from the camera dicts is how
    the two frames come to disagree). `rotation` and `scale` are per camera; 2D has one entry.
    `kpt_perm` maps each view slot to its source/session slot when a semantic flip permutation was
    applied; it is needed by scorer transfer because an affine cannot encode a keypoint-axis map.
    """
    views: list
    coords: torch.Tensor
    vis: object
    vis_2d: object
    cgroup: list
    boxes: object
    cam_names: list
    r: int
    single_view: bool
    p2d: object
    neighbour_full: object
    rotation: list
    scale: list
    kpt_perm: torch.Tensor | None = None


def _session_flip_permutation(sess):
    """Return the session-axis permutation implied by its declared ``flip_pairs``."""
    if not all(isinstance(name, str) for name in sess.names):
        raise ValueError(f'{sess.path}: session names must all be strings for flip_pairs remapping')
    if len(set(sess.names)) != len(sess.names):
        raise ValueError(f'{sess.path}: session names must be unique for flip_pairs remapping')
    slots = {name: i for i, name in enumerate(sess.names)}
    perm = np.arange(len(sess.names), dtype=np.int64)
    paired: set[str] = set()
    for pair in sess.flip_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f'{sess.path}: flip_pairs entries must be [left, right] name pairs, got {pair!r}')
        left, right = pair
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError(
                f'{sess.path}: flip_pairs names must be strings, got {pair!r}')
        if left not in slots or right not in slots:
            raise ValueError(
                f'{sess.path}: flip_pairs names must be in session names, got {pair!r}')
        if left == right:
            raise ValueError(f'{sess.path}: flip_pairs cannot self-pair {left!r}')
        if left in paired or right in paired:
            raise ValueError(
                f'{sess.path}: flip_pairs must be an involution with each keypoint in one pair, '
                f'got {pair!r}')
        paired.update((left, right))
        perm[slots[left]], perm[slots[right]] = slots[right], slots[left]
    return torch.as_tensor(perm, dtype=torch.long), [n for n in sess.names if n not in paired]


class PoseDataset(Dataset):
    """The pose training loader: one item is `shape` -> `select` -> `realise` -> `_targets`.

    `shape(rng)`, `select(idx, rng, shape)` and `realise(sel, rng, world_gauge=)` are the public
    seam the scorer builds on (one selection realised into two views). `shape` draws from the
    rank-shared stream; `select` then `realise` consume the item's own stream in that order, so a
    caller must keep that order to reproduce `__getitem__`.
    """

    def __init__(self, path, split: str, cfg: LoaderConfig, registry: Registry | None = None,
                 train: bool | None = None, seed: int = 23,
                 registry_base: Registry | None = None, rank: int = 0,
                 world_size: int = 1):
        """Build the window index for one split of a dataset (or folder of datasets).

        Scatters requested split tables in the parent process for copy-on-write workers, resolves
        each session's keypoint axis, and refuses tables that cannot supervise the requested mode.

        Inputs: path -- a dataset root or a folder of dataset roots.
                split -- 'train', 'val' or 'test'.
                cfg -- the loader configuration.
                registry -- an existing registry to use instead of building one.
                train -- override the train/val flag (defaults to split == 'train').
                seed -- the RNG seed for reproducible val/test sampling.
                registry_base -- a base registry whose ids must be preserved (warm start).
                rank/world_size -- optional disjoint session shard; only this rank's sessions
                                   are preloaded. Empty validation shards are allowed.
        Side effects: reads every requested split label table; prints the box_source coverage per
        dataset.

        `n_frames >= 2` (T = 1 gives a zero-length pos_embed); `box_source` is checked against
        `BOX_SOURCES`. `registry_base` makes ids APPEND-ONLY so embedding rows survive warm start.
        Keypoint ids are per SESSION (a session may reorder or subset the root's names), so an
        unmappable root, and a mode/table mismatch, fail here rather than mid-epoch. Dataset
        balancing is train-only, so a val/test window's identity stays tied to its index.
        """
        assert cfg.n_frames >= 2, (
            f'n_frames = {cfg.n_frames} is not usable: tracktail computes gT = T // tubelet_size '
            '(encoder_decoder.py:748), which is 0 at T=1 and yields a zero-length pos_embed. '
            'Use n_frames >= 2; short groups are clamp-padded up to it.')
        assert cfg.box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {cfg.box_source!r}'
        self.cfg = cfg
        self.split = split
        self.train = (split == 'train') if train is None else train
        self.datasets = load_datasets(path, split=split)
        prediction_sessions = [s.path for ds in self.datasets for s in ds.all_sessions()
                              if s.prediction_session]
        if prediction_sessions:
            raise ValueError('prediction sessions are not annotation inputs: '
                             + ', '.join(map(str, prediction_sessions[:5])))
        self.rank, self.world_size = int(rank), int(world_size)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f'invalid dataset shard rank={self.rank}, world_size={self.world_size}')
        self.registry = registry or Registry.build(self.datasets, registry_base)
        self.seed = seed
        self._aug = build_augmenters(cfg) if self.train and cfg.aug_prob > 0 else None

        self.index: list[_Item] = []
        self.by_dataset: list[list[int]] = []
        self._kpt_ids: dict[Path, torch.Tensor] = {}
        self._flip_perm: dict[Path, torch.Tensor] = {}
        if self.train and cfg.flip_2d_prob > 0:
            for ds in self.datasets:
                for sess in ds.sessions.get(split, []):
                    eligible = sess.mode == '2d' or (
                        sess.mode == '3d' and cfg.prob_2d_only > 0)
                    if not eligible:
                        continue
                    if not getattr(sess, 'flip_pairs_declared', False):
                        raise ValueError(
                            f'{sess.path}: flip_2d_prob > 0 requires a declared flip_pairs key. '
                            'Add flip_pairs (possibly []), set flip_2d_prob = 0, or for a 3D '
                            'session set prob_2d_only = 0.')
                    perm, unpaired = _session_flip_permutation(sess)
                    self._flip_perm[sess.path] = perm
                    if unpaired and self.rank == 0:
                        print(f'{sess.path}: flip_pairs leaves unpaired keypoints: '
                              + ', '.join(unpaired))
        boxed: list[tuple[str, int, int]] = []
        for di, ds in enumerate(self.datasets):
            mine, n_box, n_sess = [], 0, 0
            for sess in shard_sessions(ds.sessions.get(split, []), self.rank, self.world_size):
                sess.preload()
                n_sess += 1
                self._kpt_ids[sess.path] = torch.as_tensor(
                    self.registry.ids_for(ds.name, sess.names), dtype=torch.long)
                for gid, group in sess.groups.items():
                    lab = sess.labels(gid)
                    if lab.points2d is None and lab.points3d is None:
                        continue
                    need = 'points3d' if sess.mode == '3d' else 'points2d'
                    if getattr(lab, need) is None:
                        raise ValueError(
                            f'{sess.path}: mode is {sess.mode!r} so every window needs '
                            f'{need}, and this session carries none. The format allows it '
                            '(no label table is required) but training on it does not: there is '
                            'nothing to supervise the targets with.')
                    vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
                    if vis is None:
                        continue
                    for a in range(len(lab.animal_ids)):
                        starts = self._starts(vis, a, group.n_frames)
                        for st in starts:
                            mine.append(len(self.index))
                            self.index.append(_Item(di, sess, gid, a, st))
                n_box += (sess.path / 'instances.pq').exists()
            boxed.append((ds.name, n_box, n_sess))
            self.by_dataset.append(mine)
        if not self.index and self.train:
            raise ValueError(f'{path}: split {split!r} yielded no usable windows on rank '
                             f'{self.rank}/{self.world_size}')
        if cfg.box_source == 'instances':
            print(f'{split}: box_source=instances  ' + '  '.join(
                f'{n}/{t} {name}' + ('' if n == t else ' (keypoint fallback)')
                for name, n, t in boxed))
        multi = self.train and cfg.balance_datasets and len(self.datasets) > 1
        pools = self.by_dataset if multi else [list(range(len(self.index)))]
        names = ([ds.name for ds in self.datasets] if multi else [None])
        kept = [(n, p) for n, p in zip(names, pools) if p]
        self._pools = [(np.asarray(p, dtype=np.int64),
                        self._pool_weights(p) if self.train else None) for _, p in kept]
        self._pool_cum = self._dataset_pool_weights([n for n, _ in kept], multi)

    # -- indexing ------------------------------------------------------------------------
    def _labelled_frames(self, vis, a):
        """Frames where this animal has at least one assessed keypoint."""
        v = vis[a]
        v = v.reshape(v.shape[0], -1) if v.ndim > 2 else v
        return np.flatnonzero((v != UNLABELED).any(-1))

    def _starts(self, vis, a, n_frames):
        """Window starts for one animal.

        Train indexes at animal granularity (the start is picked inside `__getitem__`), so a
        57,594-frame group costs 12 index entries instead of 691,000; val/test enumerate fixed
        windows so a metric is reproducible. The FIRST frame of a window need not be labelled --
        the window is placed AROUND the label, and clamped into the group rather than running
        off the end. The first start is half a window before the first label so the label sits
        inside the window rather than at frame 0, where per-frame anchoring contributes nothing.
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
        first = int(np.clip(lo - T // 2, 0, limit))
        return sorted({int(np.clip(s, 0, limit))
                       for s in range(first, hi + 1, stride)}) or [0]

    def __len__(self):
        """Number of index entries (one per session/group/animal window)."""
        return len(self.index)

    def _pool_weights(self, pool):
        """Cumulative per-entry weights for one pool, or None to sample it uniformly.

        WHY: `_starts` returns one entry per (session, group, animal), so an entry IS a sampling
        weight decoupled from how much data sits behind it -- without balancing, a tracked
        session's one long group bought the same weight as a single hand-annotated still. The
        draw is TWO-LEVEL (source, then mode within source), independent by construction; a
        level with nothing to choose between is skipped, and a level left at None keeps its
        natural shares. Flattened into one cumulative array for a single `searchsorted`.
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

    def _dataset_pool_weights(self, names, multi):
        """Cumulative P(pool) from `[data].dataset_weights`, or None for the uniform draw.

        A relative weight per dataset-root NAME (absent = 1.0); renormalised over the pools that
        hold train windows. None (the default) keeps `_pick`'s uniform `rng.integers` draw, so a
        config without the key is byte-identical. A name that matches no root is refused (a typo
        would otherwise train at the defaults and report as the arm it is not).
        """
        dw = self.cfg.dataset_weights
        if not dw or not self.train:
            return None
        if not multi:
            raise ValueError('dataset_weights needs balance_datasets = true and a folder of '
                             'dataset roots on the train split')
        known = {ds.name for ds in self.datasets}
        unknown = sorted(set(dw) - known)
        if unknown:
            raise ValueError(f'dataset_weights: unknown dataset root(s) {unknown}; '
                             f'known: {sorted(known)}')
        w = np.asarray([float(dw.get(n, 1.0)) for n in names], dtype=np.float64)
        if not np.all(np.isfinite(w)) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f'dataset_weights must be finite, >= 0 and not all zero, got {dw}')
        return np.cumsum(w / w.sum())

    def pool_shares(self):
        """P(pool) per train pool, in `_pools` order (uniform unless `dataset_weights`)."""
        n = len(self._pools)
        if self._pool_cum is None:
            return np.full(n, 1.0 / n)
        return np.diff(self._pool_cum, prepend=0.0)

    def mix(self):
        """Source-session share per (label_source, mode) cell before the 2-D path coin.

        Reporting only: printed at startup because the source mix is invisible in the loss curve;
        3-D sessions contribute `prob_2d_only` of their train draws to the runtime 2-D path.
        """
        out: dict[str, float] = {}
        for (p, cum), share in zip(self._pools, self.pool_shares()):
            wt = np.diff(cum, prepend=0.0) if cum is not None else np.full(len(p), 1.0 / len(p))
            for j, x in zip(p, wt * share):
                sess = self.index[j].session
                out[f'{sess.mode}-{sess.label_source}'] = (
                    out.get(f'{sess.mode}-{sess.label_source}', 0.0) + float(x))
        return dict(sorted(out.items()))

    def _pick(self, idx, rng):
        """The `_Item` an index addresses: the index itself on val/test, a pool draw on train.

        Inputs: idx -- the requested index.
                rng -- the item's RNG stream.
        Outputs: the selected `_Item`.

        Val and test address the index directly -- a window's identity is its index. On train
        the draw is uniform over datasets, then weighted within -- otherwise one dataset's many
        groups would outvote another's; with no configured weights it is byte-for-byte the
        previous behaviour.
        """
        if not self.train:
            return self.index[idx]
        if len(self._pools) == 1:
            pool, cum = self._pools[0]
        elif self._pool_cum is None:
            pool, cum = self._pools[rng.integers(len(self._pools))]
        else:
            k = min(int(np.searchsorted(self._pool_cum, rng.random(), side='right')),
                    len(self._pools) - 1)
            pool, cum = self._pools[k]
        if cum is None:
            return self.index[idx if len(self._pools) == 1 else pool[rng.integers(len(pool))]]
        return self.index[int(pool[np.searchsorted(cum, rng.random())])]

    # -- item ----------------------------------------------------------------------------
    def __getitem__(self, idx):
        """One window. `idx` is an index, or `(ordinal, index)` from `StepSampler`.

        The ordinal makes the item's COST the same on every rank: a DDP step costs the SLOWEST
        rank's item, so `shape_rng` keyed on the ordinal draws the same camera count everywhere
        while the window, session and augmentation stay independent -- the marginal distribution
        is untouched, and it applies at every world size. Drawn ONCE, outside the retry loop, or
        ranks would fall out of step. The item RNG is entropy-seeded on train so workers do not
        replay one another's augmentation, and index-seeded on val/test so a metric is
        reproducible. Only train replaces a window that fails to build; a val/test window
        returns None (SKIPPED -- `pose_collate` drops it) rather than a random other window that
        the metric would then count twice.
        """
        ordinal, idx = idx if isinstance(idx, tuple) else (None, idx)
        rng = np.random.default_rng(None if self.train else (self.seed, idx))
        shape_rng = (rng if ordinal is None
                     else np.random.default_rng((self.seed, 0x5AFE, int(ordinal))))
        shape = self.shape(shape_rng)
        if not self.train:
            return self._item(idx, rng, shape)
        for _ in range(8):
            out = self._item(idx, rng, shape)
            if out is not None:
                return out
            idx = int(rng.integers(len(self.index)))
        raise RuntimeError(f'{self.split}: 8 consecutive items failed to build')

    def get_once(self, idx):
        """Build exactly the indexed item once, without the ordinary retry replacement.

        This is for deterministic QC/evaluation callers that must not silently substitute a
        different window when decoding or realisation fails -- what `__getitem__` also does off
        the train split; on train `__getitem__` retries.

        Inputs: idx -- an integer index into ``self.index``.
        Outputs: the indexed item, or ``None`` when that item cannot be built.
        Side effects: decodes and realises one item, just like the first ``__getitem__`` attempt.
        """
        if isinstance(idx, tuple):
            raise TypeError('get_once expects an integer index, not an ordinal/index tuple')
        idx = int(idx)
        rng = np.random.default_rng(None if self.train else (self.seed, idx))
        return self._item(idx, rng, self.shape(rng))

    def shape(self, rng) -> dict:
        """The cost-determining draws, made from a stream every rank shares. See `__getitem__`.

        Exactly the two draws that change how much WORK an item is and nothing else: the camera
        count and the 2-D path coin. The cell, session, start, T and every augmentation stay on
        the item's own stream.
        """
        return {'n_cams': n_cams(self.cfg.cams_to_sample, rng),
                'single_view_draw': float(rng.random())}

    def _frames(self, item, lab, group, rng):
        """T frame indices, clamp-padded so T >= 2.

        ON TRAIN, T IS DERIVED FROM THE LABELS and `cfg.n_frames` is only its ceiling: annotated
        sessions carry ONE labelled frame per 65-frame group, so a fixed T would encode many
        frames to supervise one (the floor of 2 still means one label carries an unsupervised
        partner -- the claim is only that no frame is encoded which is not needed to reach a
        label). Train frames may also be STRIDED by `cfg.frame_strides` on a lattice of spacing
        s. VAL AND TEST ARE UNTOUCHED -- fixed `cfg.n_frames` windows at stride 1, so a metric is
        comparable across checkpoints.

        On train, a stride is admissible if the group holds two frames at it; the window is
        anchored on a labelled frame and placed AROUND it, with T shrunk to the labelled frames
        some placement on that lattice could reach (falling back to the anchor when the span
        exceeds the ceiling).
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
            fit = [x for x in self.cfg.frame_strides if x <= group.n_frames - 1]
            s = int(fit[rng.integers(len(fit))]) if fit else 1
            anchor = int(labelled[rng.integers(labelled.size)])
            T = min(T, (group.n_frames - 1 - anchor % s) // s + 1)
            near = labelled[(labelled > anchor - T * s) & (labelled < anchor + T * s)]
            near = near[(near - anchor) % s == 0]
            first, last = int(near[0]), int(near[-1])
            T = _even_span((last - first) // s + 1, T)
            if last - first > (T - 1) * s:
                first = last = anchor
            span = (T - 1) * s
            lo = max(0, last - span)
            hi = min(first, max(0, group.n_frames - 1 - span))
            lo += (anchor - lo) % s
            start = int(lo + s * rng.integers(0, (hi - lo) // s + 1)) if hi > lo else lo
        f = np.clip(np.arange(start, start + T * s, s), 0, group.n_frames - 1)
        return f

    def _crop_pts(self, lab, a, frames, cam_ix):
        """One animal's stored boxes over a window, as points the crop rule can bound.

        FOUR corners per box, not the two the table stores: an in-plane rotation makes the
        two-corner extent strictly inside the four-corner one. None when the switch is off or the
        session has no `instances.pq`; an all-NaN result is legal and falls back per view.
        """
        if self.cfg.box_source != 'instances' or lab.boxes is None:
            return None
        b = lab.boxes[a][frames][:, cam_ix]
        return torch.as_tensor(cropmod.box_corners(b), dtype=torch.float32)

    def _hybrid_2d_coords(self, lab, a, frames, camera, camera_index):
        """Return stored 2-D labels with projected 3-D labels filling missing slots.

        Inputs: lab -- the session labels; a -- animal index; frames -- source frame ids;
        camera -- one pre-transform camera dictionary; camera_index -- its session camera axis.
        Outputs: `[T,K,2]` source-image pixels. Side effects: none.
        """
        projected = None
        if lab.points3d is not None:
            xyz = torch.as_tensor(lab.points3d[a][frames], dtype=torch.float32)
            projected = project_points_torch([camera], xyz)[0]
            visible = is_point_visible(camera, xyz)
            projected = projected.masked_fill(~visible.unsqueeze(-1), float('nan'))
        if lab.points2d is None:
            return projected
        stored = torch.as_tensor(lab.points2d[a][frames, :, camera_index, :], dtype=torch.float32)
        if projected is None:
            return stored
        finite = torch.isfinite(stored).all(-1, keepdim=True)
        if lab.vis2d is None:
            return torch.where(finite, stored, projected)
        status = torch.as_tensor(lab.vis2d[a][frames, :, camera_index], dtype=torch.int8)
        missing = status.eq(MISSING).unsqueeze(-1)
        fallback = projected.masked_fill(missing, float('nan'))
        coords = torch.where(finite, stored, fallback)
        return coords.masked_fill(missing, float('nan'))

    def _item(self, idx, rng, shape=None):
        """One window, built as `select` -> `realise` -> `_targets`.

        The three-way split is a SEAM, not a tidy-up: a view-independent selection can be
        realised into more than one view (`tailcyclenet/scorer/`), which a fused `_item` cannot
        express. The three parts share ONE `rng` stream and consume it in the same order they
        did when this was a single function, so the pose path is byte-identical -- asserted by
        `tests/loader/test_select_realise.py`, which is the condition the split was approved
        on, not a nicety.
        """
        sel = self.select(idx, rng, shape)
        if sel is None:
            return None
        view = self.realise(sel, rng)
        if view is None:
            return None
        return self._targets(sel, view, rng)

    def select(self, idx, rng, shape=None) -> Selection | None:
        """Draw the view-independent part of a window, or None when the item cannot be built.

        Inputs: idx -- the index to pick.
                rng -- the item's RNG stream.
                shape -- a pre-drawn cost-determining shape dict (see `shape`).
        Outputs: the `Selection`, or None (an index that resolves to no labelled frames, or a
        track with fewer than 2 finite coordinates).

        Consumes the shape/cell draw, the camera-count and single-view draws, the crop-inflate
        draw, the swap-animal neighbour draw and the frame draw -- everything whose value is a
        property of WHICH window this is rather than of how it is shown.

        Visibility: `vis` (3D noisy-OR) is unused at R == 2; `vis_2d` is the per-camera
        THREE-STATE target (NaN = "not assessed", masked out of the BCE; `projected` joins
        UNLABELED), withheld when `sess.has_visibility_assessment` is False or every
        assessed row is projected/unlabeled; both-or-neither.

        `coords` here is the CLEAN track at the stored coordinates -- untransformed, NaN where
        unlabelled. Every view-dependent coordinate lives in the `View`.
        """
        shape = shape or self.shape(rng)
        item = self._pick(idx, rng)
        sess, group = item.session, item.session.groups[item.gid]
        lab = sess.labels(item.gid)
        frames = self._frames(item, lab, group, rng)
        if frames is None:
            return None
        a = item.animal
        K = sess.n_keypoints

        attempt_swap_animal = self.train and self.cfg.prompt_swap_animal > 0 and lab.n_animals >= 2
        neighbour_row = None
        if attempt_swap_animal:
            other_rows = [i for i in range(lab.n_animals) if i != a]
            neighbour_row = other_rows[int(rng.integers(len(other_rows)))]

        cgroup = sess.cgroup(item.gid, frames)
        inflate = crop_inflate(self.cfg, rng, self.train)

        session_2d = sess.mode == '2d'
        if self.cfg.two_d_sampling == 'true-2d-hybrid-v1':
            force_2d = (not session_2d and self.train
                        and self.cfg.prob_2d_only > 0
                        and shape['single_view_draw'] < self.cfg.prob_2d_only)
            true_2d = session_2d or force_2d
            single_view = False
        elif self.cfg.two_d_sampling == 'legacy-3d-single-view':
            true_2d = session_2d
            single_view = (not true_2d and self.train
                           and self.cfg.prob_2d_only > 0
                           and shape['single_view_draw'] < self.cfg.prob_2d_only)
        else:
            raise ValueError(f'unknown two_d_sampling={self.cfg.two_d_sampling!r}')

        if true_2d:
            cam_ix = ([0] if session_2d else
                      [int(rng.integers(len(cgroup)))])
        else:
            n = shape['n_cams']
            cam_ix = (sorted(rng.choice(len(cgroup), n, replace=False)) if 0 < n < len(cgroup)
                      else list(range(len(cgroup))))
        cgroup = [cgroup[i] for i in cam_ix]
        cam_names = [sess.cam_names[i] for i in cam_ix]
        crop_pts = self._crop_pts(lab, a, frames, cam_ix)

        if true_2d:
            coords = self._hybrid_2d_coords(lab, a, frames, cgroup[0], cam_ix[0])
            if coords is None:
                return None
            vis = vis_2d = None
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]
                vis_2d = _vis2d_target(v2)
                if not torch.isfinite(vis_2d).any():
                    vis_2d = None
        else:
            coords = torch.as_tensor(lab.points3d[a][frames], dtype=torch.float32)
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]
                vis_2d = _vis2d_target(v2)
                vis = torch.as_tensor((v2 == VISIBLE).any(-1))
                if not torch.isfinite(vis_2d).any():
                    vis = vis_2d = None
            else:
                vis = vis_2d = None

        if torch.isfinite(coords).all(-1).sum() < 2:
            return None

        return Selection(item=item, sess=sess, group=group, lab=lab, frames=frames, animal=a,
                         n_keypoints=K, shape=shape, true_2d=true_2d, single_view=single_view,
                         cam_ix=cam_ix, cam_names=cam_names, cgroup=cgroup, crop_pts=crop_pts,
                         coords=coords, vis=vis, vis_2d=vis_2d,
                         attempt_swap_animal=attempt_swap_animal, neighbour_row=neighbour_row,
                         inflate=inflate)

    def realise(self, sel, rng, world_gauge=True) -> View | None:
        """Realise ONE view of a `Selection`: rotation, crop, resize, decode, appearance aug.

        Applies per-camera rotation, optional 2D flip, crop jitter and appearance augmentation.
        The flip draw follows the complete 2D rotation block and is made only for positive
        configured probability.
        Returns the `View`, or None when the crop or the decode fails.
        `sel` is NOT mutated, so one selection may be realised repeatedly (the scorer's anchor);
        asserted in `tests/loader/test_select_realise.py`.

        Geometry: camera and crop-inflate draws are one per item (camera draw sorted). Points
        outside the source frame or the FINAL crop are flipped out of `vis_2d`; a rotation that
        loses the animal to the inscribed crop is REVERTED, not retried.

        Pixels: appearance augmentation runs on the final ~256 px crops; views are UINT8 (4x
        fewer bytes to collate/queue/pin; the model divides on device).

        `world_gauge` gates the 3D WORLD-GAUGE rotation (`rotate_camera_group_with_neighbours`),
        not the per-camera image-plane rotation. The scorer sets it False: the gauge rotation moves
        `coords` too, so two realisations would land in different world gauges.

        Inputs: sel -- a `Selection`; rng -- this view's own stream; world_gauge -- see above.
        Outputs: the `View`, or None when the crop fails or a frame will not decode; side effects:
        decodes video frames through the reader cache and draws from `rng`.
        """
        group, lab = sel.group, sel.lab
        frames = sel.frames
        true_2d, single_view = sel.true_2d, sel.single_view
        cam_names, crop_pts, inflate = sel.cam_names, sel.crop_pts, sel.inflate
        attempt_swap_animal, neighbour_row = sel.attempt_swap_animal, sel.neighbour_row
        coords, vis = sel.coords, sel.vis
        vis_2d = None if sel.vis_2d is None else sel.vis_2d.clone()
        cgroup = [dict(c) for c in sel.cgroup]
        scales = [None] * len(cgroup)

        rot_p = (self.cfg.aug_prob if self.cfg.aug_rotation_prob is None
                 else self.cfg.aug_rotation_prob)
        rot_deg = self.cfg.aug_rotation_deg
        rotation_info = [None] * len(cgroup)
        neighbour_full = None
        flipped = False
        kpt_perm = None
        if true_2d:
            cam = cgroup[0]
            kpt_perm = torch.arange(sel.n_keypoints, dtype=torch.long)
            coords = mask_outside(coords, cam['size'])
            if vis_2d is not None:
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            cp = None if crop_pts is None else crop_pts[:, 0]
            if self.train and rng.random() < rot_p:
                cam, coords, rot = rotate_2d(cam, coords,
                                              float(rng.uniform(-rot_deg, rot_deg)))
                rotation_info = [rot]
            if (self.train and self.cfg.flip_2d_prob > 0
                    and rng.random() < self.cfg.flip_2d_prob):
                mode = self.cfg.flip_2d_modes[int(rng.integers(len(self.cfg.flip_2d_modes)))]
                kpt_perm = self._flip_perm[sel.sess.path]
                cam, coords, flip = flip_2d(cam, coords, mode, rotation_info[0], kpt_perm)
                rotation_info = [flip]
                flipped = True
                if vis_2d is not None:
                    vis_2d = vis_2d[:, kpt_perm]
            cp = apply_affine(cp, rotation_info[0])
            jit = self._jitter(rng)
            cam, box, coords = cropmod.crop_to_points_2d(cam, coords, self.cfg.min_crop_dim,
                                                         jit, crop_pts=cp,
                                                         inflate=inflate,
                                                         pad=self.cfg.crop_pad_px,
                                                         pad_frac=self.cfg.crop_pad_frac)
            if cam is None:
                return None
            cam, scale = resize_camera(cam, self.cfg.image_size)
            scales = [scale]
            coords = coords * scale
            coords = mask_outside(coords, cam['size'])
            if vis_2d is not None:
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            if attempt_swap_animal:
                raw = self._hybrid_2d_coords(
                    lab, neighbour_row, frames, sel.cgroup[0], sel.cam_ix[0])
                raw = apply_affine(raw, rotation_info[0])
                if flipped:
                    raw = raw[:, self._flip_perm[sel.sess.path]]
                neighbour_full = (raw - box[:2].to(raw.dtype)) * scale
            cgroup, boxes = [cam], [box]
            p2d = p2d_all = coords[None]
        else:
            if self.train:
                rotated = []
                for cam in cgroup:
                    if rng.random() < rot_p:
                        cam_r, rot = rotate_camera_image_plane_3d(
                            cam, float(rng.uniform(-rot_deg, rot_deg)))
                        if int(is_point_visible(cam_r, coords).sum()) < 2 <= int(
                                is_point_visible(cam, coords).sum()):
                            cam_r, rot = cam, None
                        rotated.append((cam_r, rot))
                    else:
                        rotated.append((cam, None))
                cgroup = [c for c, _ in rotated]
                rotation_info = [r for _, r in rotated]
                if vis_2d is not None:
                    for cnum, cam in enumerate(cgroup):
                        if rotation_info[cnum] is not None:
                            gone = ~is_point_visible(cam, coords) & (vis_2d[:, :, cnum] == 1)
                            vis_2d[:, :, cnum][gone] = 0
            others_raw = None
            if attempt_swap_animal:
                others_raw = torch.as_tensor(lab.points3d[neighbour_row][frames],
                                             dtype=torch.float32)
            if self.train and world_gauge:
                cgroup, coords, others_raw = rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            cp3 = None if crop_pts is None else [
                apply_affine(crop_pts[:, i], rotation_info[i]) for i in range(len(cgroup))]
            jit = self._jitter(rng)
            cgroup, boxes = cropmod.crop_to_points_3d(cgroup, coords, self.cfg.min_crop_dim,
                                                      jit, crop_pts=cp3,
                                                      inflate=inflate,
                                                      pad=self.cfg.crop_pad_px,
                                                      pad_frac=self.cfg.crop_pad_frac)
            if cgroup is None:
                return None
            resized = [resize_camera(c, self.cfg.image_size) for c in cgroup]
            cgroup = [c for c, _ in resized]
            scales = [s for _, s in resized]
            if self.train and world_gauge:
                cgroup, coords, others_raw = rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            if others_raw is not None:
                neighbour_full = others_raw
            p2d_all = (project_points_torch(cgroup, coords)
                       if single_view or self._aug is not None else None)
            p2d = p2d_all if single_view else None

        gray = self._aug is not None and rng.random() < self.cfg.grayscale_prob
        invert = (self._aug is not None and self.cfg.invert_prob > 0
                  and rng.random() < self.cfg.invert_prob)
        use_pool = group.source(cam_names[0])[0] != 'video'
        with (ThreadPoolExecutor(max_workers=16) if use_pool else nullcontext()) as pool:
            views = []
            for cnum, cam_name in enumerate(cam_names):
                try:
                    imgs = read_frames(group, cam_name, frames, crop_coords=boxes[cnum],
                                       target_size=cgroup[cnum]['size'].tolist(),
                                       rotation=rotation_info[cnum], pool=pool)
                except decode_errors() as e:
                    if use_pool:
                        raise
                    warn_decode_failure(group, cam_name, e)
                    return None
                if any(im is None for im in imgs):
                    return None
                if self._aug is not None:
                    imgs = augment_views(self._aug, self.cfg, imgs, cnum, cgroup[cnum]['size'], p2d_all,
                                         vis_2d, gray, rng, invert=invert)
                views.append(torch.from_numpy(np.asarray(imgs)))

        if vis is not None and vis_2d is not None:
            vis = (vis_2d == 1).any(-1)

        return View(views=views, coords=coords, vis=vis, vis_2d=vis_2d, cgroup=cgroup,
                    boxes=boxes, cam_names=cam_names, r=2 if true_2d else 3,
                    single_view=single_view, p2d=p2d,
                    neighbour_full=neighbour_full, rotation=rotation_info, scale=scales,
                    kpt_perm=kpt_perm)

    def _targets(self, sel, view, rng):
        """Turn a realised `View` into the pose training targets and the item tuple.

        The query prior and its corruptions are `prior.corrupt_prior` (draw order there).
        `pose_only_prob` (0 = byte-identical) hoists the box-dropout coin EARLY so a box-dropped
        item can also skip the corruptions, REPLACING not adding to `box_prompt_dropout`'s
        fraction; the box-prompt block reuses it. The stride is read back off `frames` (MEDIAN: a
        group-edge window repeats its last frame). The scorer never calls this.
        """
        item, sess, lab, frames = sel.item, sel.sess, sel.lab, sel.frames
        a, K = sel.animal, sel.n_keypoints
        views, coords, cgroup = view.views, view.coords, view.cgroup
        vis, vis_2d = view.vis, view.vis_2d
        cam_names, single_view = view.cam_names, view.single_view
        p2d = view.p2d
        neighbour_full, R = view.neighbour_full, view.r

        box_dropped_early = None
        pose_only_active = False
        if self.train and self.cfg.pose_only_prob > 0 and self.cfg.box_prompt != 'none' \
                and self.cfg.box_prompt_dropout > 0:
            box_dropped_early = rng.random() < self.cfg.box_prompt_dropout
            if box_dropped_early:
                pose_only_active = rng.random() < self.cfg.pose_only_prob

        kpt_prior, prompt_t = corrupt_prior(self.cfg, self.train, coords, cgroup, neighbour_full,
                                            K, R, rng, pose_only_active)

        kpt_ids = self._kpt_ids[sess.path]
        query_times = torch.zeros(K, dtype=torch.int32)
        query_occlusion = torch.full((K, len(cgroup)), -1, dtype=torch.int64)
        stride = max(1, int(np.median(np.diff(frames)))) if len(frames) > 1 else 1

        if vis is not None and vis_2d is not None:
            vis = (vis_2d == 1).any(-1)

        row = {'dataset': self.datasets[item.ds].name, 'session': sess.session_id,
               'group': item.gid, 'animal': lab.animal_ids[a], 'mode': '2d' if R == 2 else '3d',
               'single_view': single_view, 'start': int(frames[0]), 'cameras': cam_names,
               'stride': stride}

        out = [views, coords, vis, torch.as_tensor(frames), cgroup, row, query_times,
               vis_2d, p2d, query_occlusion, kpt_ids, kpt_prior, prompt_t]
        if self.cfg.box_prompt != 'none':
            from ..pose import box_prompt as bpmod
            box = bpmod.compute_box_prompt(coords, cgroup, '2d' if R == 2 else '3d')
            box = bpmod.apply_frames_mode(box, self.cfg.box_prompt_frames)
            if self.train:
                dropped = (box_dropped_early if box_dropped_early is not None
                          else (self.cfg.box_prompt_dropout > 0
                                and rng.random() < self.cfg.box_prompt_dropout))
                if dropped:
                    box = torch.full_like(box, float('nan'))
                elif self.cfg.box_prompt_jitter > 0 or self.cfg.box_prompt_scale_jitter > 0:
                    box = bpmod.apply_jitter(box, rng, self.cfg.box_prompt_jitter,
                                             self.cfg.box_prompt_scale_jitter)
            out.append(box)
        return tuple(out)

    def _jitter(self, rng):
        """A crop jitter box for this item, or None when jitter is off (val/test)."""
        if not self.train or self.cfg.crop_jitter <= 0:
            return None
        return cropmod.jitter_box(rng, self.cfg.crop_jitter, self.cfg.crop_jitter_scale)
