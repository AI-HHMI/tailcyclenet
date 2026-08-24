"""The training loader: tailcycle-dataset on disk -> the batch posetail's model consumes.

Three sampling modes, decided per item: 3D multiview (`cams_to_sample` cameras, world-mm targets),
3D single-view (one camera, targets still world mm, fired at `prob_2d_only`), and 2D single-view
(crop-pixel targets). Mode is a property of the sampled SESSION, so one `train/` may hold both
and both head-bank slots get gradient. Two non-negotiable rules: keypoints are never filtered
(the library's `filter_keypoints` shrinks N so array position stops equalling identity), and
T >= 2 always (a T=1 window gives posetail a zero-length pos_embed).
"""
from __future__ import annotations

import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
import threading
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.datasets.posetail_dataset import custom_collate, rotate_camera_image_plane_3d
from posetail.posetail.cube import get_camera_scale, is_point_visible, project_points_torch

from . import crop as cropmod
from . import memory as _memory
from .crop import BOX_SOURCES
from .format import PROJECTED, UNLABELED, VISIBLE, Registry, load_datasets


@dataclass
class LoaderConfig:
    """Everything the loader is allowed to vary. Deliberately short."""
    n_frames: int = 24
    # cameras are resized so max(W,H) == this
    image_size: int = 256
    # int, or a [low, high] pair drawn per item (posetail's own `sample_cameras`); 0 means every
    # camera. The pretrained tracker this finetunes from was trained at [1, 8], so a fixed count
    # above that is out of distribution as well as slow.
    cams_to_sample: int | list = 0
    # the reference's [dataset.val] value
    val_cams_to_sample: int | list = 5
    # rate at which a 3D session is shown a single camera
    prob_2d_only: float = 0.25
    # sample datasets uniformly, not proportionally
    balance_datasets: bool = True
    # in-plane rotation, per-camera appearance, cutout
    aug_prob: float = 0.25
    # In-plane rotation magnitude and rate, split out of `aug_prob` because they are what a root
    # sets. 180 is a FULL 360 draw and costs nothing over 45 on a wide frame: the border-free
    # canvas is 90-degree periodic (`_rotated_rect_max_inscribed`), and the wider draw covers
    # more heading.
    aug_rotation_deg: float = 45.0
    # None means "follow aug_prob"; set it to dial rotation without moving appearance jitter.
    aug_rotation_prob: float | None = None
    # per-FRAME appearance: motion blur, sensor noise
    per_image_aug_prob: float = 0.25
    # rate at which a train item drops colour entirely
    grayscale_prob: float = 0.2
    # box centre jitter, fraction of box size
    crop_jitter: float = 0.3
    # box scale jitter
    crop_jitter_scale: float = 0.3
    min_crop_dim: int = 64
    # What the crop rule bounds: `keypoints` (the labels) or `instances` (the `instances.pq`
    # box), for a root whose stored keypoints are too sparse to enclose the animal. INERT where
    # a root ships no table (falls back to keypoints per view).
    box_source: str = 'instances'
    # fraction of TRAINING STEPS that run fully query-free
    prompt_dropout: float = 0.4
    # sigma on the prior, in PIXELS (3D scales by cube_scale)
    prompt_noise_px: float = 0.0
    # The two corruptions that have the SHAPE of a deployment failure, both off by default.
    # sigma of a WHOLE-BODY offset, one vector per item
    prompt_offset_px: float = 0.0
    # max frames the prior may describe away from `prompt_t`
    prompt_stale_frames: int = 0
    # TWO MORE FAILURES WITH THE SHAPE OF A DEPLOYMENT ERROR, both off by default and both aimed
    # at `--anchor carry` specifically: a carried pose that swapped two keypoints, or latched
    # onto the wrong animal. BOTH ARE PER-KEYPOINT PROBABILITIES -- every prompted keypoint
    # independently draws its own Bernoulli, not a single per-item coin.
    #
    # A SELECTED keypoint is REPLACED by another finite keypoint's ORIGINAL position, drawn
    # UNIFORMLY and INDEPENDENTLY per selected point -- not a permutation, so the prior SET is
    # not preserved (the same way `prompt_noise_px`/`prompt_offset_px` already perturb it).
    prompt_swap_kpt_pairs: float = 0.0
    # Bernoulli(`prompt_swap_animal`), independently PER KEYPOINT, against the ONE neighbour
    # animal picked for this item -- so P(a given keypoint's prior becomes that neighbour's own
    # point) is `prompt_swap_animal`, CONDITIONAL on that point itself being usable there
    # (`prior_out_of_bounds`, the same rule `--anchor carry` uses). The CONFIGURED rate is
    # therefore an upper bound on the PRESENTED one. A session with fewer than two animals never
    # reaches for this key at all.
    prompt_swap_animal: float = 0.0
    # 0 -> non-overlapping windows for val/test
    val_stride: int = 0
    # Frame stride for a TRAIN window, drawn per item; repeat an entry to weight it.
    # Val/test are always 1.
    frame_strides: list = field(default_factory=lambda: [1])
    # Train sampling mix, a TWO-LEVEL draw: source first, then mode within that source. Either
    # level is skipped where a dataset offers no choice; None leaves that level alone (entries
    # keep their natural share).
    # P(a step comes from an `annotated` session)
    annot_frac: float | None = None
    # P(3d | source), i.e. applied WITHIN each source
    mode_3d_frac: float | None = None
    # THE BOX PROMPT, the DATA side. 'none' | 'film': when not 'none' the loader emits a
    # per-frame animal box (`box_prompt.compute_box_prompt`) as a non-position channel; 'none'
    # emits nothing, so a plain run is byte-identical. Set by `scripts/train.py` from
    # `[model].box_prompt` so the two cannot disagree.
    box_prompt: str = 'none'
    # 'all' (per frame) | 'first' (the starting frame's box)
    box_prompt_frames: str = 'all'
    # fraction of STEPS the box is withheld (no-box token)
    box_prompt_dropout: float = 0.0
    # exposure bias: the deployed box is a DETECTOR box
    box_prompt_jitter: float = 0.0
    box_prompt_scale_jitter: float = 0.0
    # WIDE-CROP TRAINING: widen the crop-rule box about its centre by this factor BEFORE the
    # coords are shifted into it, so the animal sits off-centre in a wider crop and the box
    # (computed post-hoc from the returned coords) is the only non-centred cue for which animal.
    # 1.0 is INERT and byte-identical to a config without this key.
    #
    # A float, or a `[low, high]` pair drawn per TRAIN item (the same `cams_to_sample` contract);
    # VAL/TEST NEVER DRAW: `_crop_inflate` returns the range's MIDPOINT there, exactly as
    # `crop_jitter` is gated to train-only -- `checkpoint_best` selection has to read the SAME
    # crop geometry every val pass, or the val curve would be noise from the geometry draw.
    crop_inflate: float | list = 1.0


# pixels

def _rotate_2d(cam, coords, angle_deg):
    """`rotate_points_image_plane` WITHOUT its border-free inscribed crop. Same return shape.

    The inscribed crop silently NaN'd the labels of items whose animal sat outside it (the 3D
    path stays on the library helper because it recomputes `vis_2d` after rotating, which 2D
    cannot). The full canvas costs nothing: the follow-on crop only meets the canvas edge for a
    frame-edge animal, where `_crop_affine` already renders BORDER_CONSTANT zeros.

    Only the principal point tracks the canvas expansion; `ext`/`ext_inv`/`center` stay
    untouched -- the 2D coords ARE the target, and moving them through the same affine is
    the whole rule.
    """
    import cv2

    w, h = cam['size'].tolist()
    center_x = float(cam['mat'][0, 2].item()) - float(cam['offset'][0].item())
    center_y = float(cam['mat'][1, 2].item()) - float(cam['offset'][1].item())
    M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    rot = corners @ M[:, :2].T + M[:, 2]
    tx, ty = -rot.min(axis=0)
    M[0, 2] += tx
    M[1, 2] += ty
    cw = int(np.ceil(rot[:, 0].max() - rot[:, 0].min()))
    ch = int(np.ceil(rot[:, 1].max() - rot[:, 1].min()))

    out = dict(cam)
    out['mat'] = cam['mat'].clone()
    out['mat'][0, 2] = cam['mat'][0, 2] + tx
    out['mat'][1, 2] = cam['mat'][1, 2] + ty
    out['offset'] = cam['offset'].clone()
    out['size'] = torch.tensor([cw, ch], dtype=torch.int32, device=cam['size'].device)

    Mt = torch.as_tensor(M, dtype=coords.dtype, device=coords.device)
    return out, coords @ Mt[:, :2].T + Mt[:, 2], (M, (cw, ch))


def _apply_affine(pts, rotation):
    """Move (...,2) pixel points through a rotation's own 2x3, or pass through untouched.

    Shared so a stored box and the labels can never end up in different frames; a list is a
    per-frame rotation over `pts`' leading (time) axis.
    """
    if pts is None or rotation is None:
        return pts
    if isinstance(rotation, list):
        return torch.stack([_apply_affine(pts[t], rotation[min(t, len(rotation) - 1)])
                            for t in range(pts.shape[0])])
    M = torch.as_tensor(rotation[0], dtype=pts.dtype)
    return pts @ M[:, :2].T + M[:, 2]


def prior_out_of_bounds(p, mode, cgroup):
    """Which keypoints of a prior are NOT usable as one. (K,) bool, in the MODEL's own frame.

    A prior outside the crop is not a prior: in 2D that is a point outside the crop rectangle; in
    3D a point no PAIR of cameras can see. ONE copy of the rule, shared by the loader's prior
    corruptions and `infer.py` -- a second copy is how two prompted regimes came to disagree. A
    moving camera answers per frame ((T,K)).
    """
    if mode == '2d':
        w, h = (float(x) for x in cgroup[0]['size'][:2])
        return (p[:, 0] < 0) | (p[:, 0] >= w) | (p[:, 1] < 0) | (p[:, 1] >= h)
    seen = []
    for c in cgroup:
        v = is_point_visible(c, p, margin=2)
        seen.append(v.any(0) if v.ndim > 1 else v)
    return torch.stack(seen).sum(0) < 2


def _rotate_camera_group_with_neighbours(cgroup, coords, others):
    """`PosetailDataset.rotate_camera_group`, optionally carrying a SECOND point set through the
    identical draw. Returns `(cgroup, coords, others)`; `others` is None in, None out.

    With `others` None (the default) this is EXACTLY the library call. With it, `others` is one
    neighbour's (T,K,R) pose and rides the same `rotmat` via a temporary concatenation on the
    keypoint axis, which changes nothing about `coords`' own output. Callers must NOT pass the
    combined tensor to `crop_to_points_3d`: the crop is the TARGET's extent, and a neighbour
    would change the item's geometry.
    """
    from posetail.datasets.posetail_dataset import PosetailDataset

    if others is None:
        cgroup, coords = PosetailDataset.rotate_camera_group(None, cgroup, coords)
        return cgroup, coords, None
    K = coords.shape[1]
    combined = torch.cat([coords, others], dim=1)
    cgroup, combined = PosetailDataset.rotate_camera_group(None, cgroup, combined)
    return cgroup, combined[:, :K], combined[:, K:]


def _crop_affine(src_wh, crop_coords, target_size, rotation):
    """The one dst<-src affine for rotate -> crop -> resize. Returns (M_2x3, (w, h)) or None.

    None means all three are no-ops and the caller should not warp at all. Composition is not a
    micro-optimisation: sequential rotate-then-crop warps the whole frame and resamples twice.
    The CORNER convention (`x_dst = (x_src - x1) * sx`) matches `crop.apply_crop`'s offset and
    `_resize_camera`'s mat scale, so the pixels agree with the intrinsics `project_points_torch`
    reprojects through. Out-of-source pixels arrive as BORDER_CONSTANT zeros.
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


def load_warps(path, specs, target_size=None, reduce=1):
    """ONE decode, N affines -> list of (H,W,3) uint8 RGB. All None if the file will not decode.

    Decode cost is per FILE and warp cost per OUTPUT, so one `load_image` call per output frame
    charges the first at the rate of the second -- invisible until a caller wants several crops of
    the same frame. BGR is kept until after the warp so the colour convert runs on the SMALL
    output. `reduce` in {1,2,4,8} decodes at 1/N via libjpeg DCT decimation, only valid for
    whole-frame callers (asserted).
    """
    import cv2

    if reduce == 1:
        img = cv2.imread(path)
    else:
        assert all(c is None and r is None for c, r in specs), \
            'reduce decodes at 1/N, so source-pixel crop_coords/rotation would be misplaced'
        flag = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
        img = cv2.imread(path, flag)
    if img is None:
        return [None] * len(specs)
    src_wh = (img.shape[1], img.shape[0])
    out = []
    for crop_coords, rotation in specs:
        aff = _crop_affine(src_wh, crop_coords, target_size, rotation)
        warped = (img if aff is None
                  else cv2.warpAffine(img, aff[0], aff[1], flags=cv2.INTER_LINEAR))
        out.append(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return out


# An OPEN reader retains about a gigabyte, so the cache size is a MEMORY BUDGET, not a hint:
# 32 live readers across 8 workers is how the calms21 runs got OOM-killed. Sized per process by
# `_reader_cache_size`; `TAILCYCLENET_READER_CACHE` only overrides it.


# GB of retained memory per OPEN reader per megapixel of frame (PyAV, measured, rounded up).
_READER_GB_PER_MP = 0.06


def reader_cache_ram_gb(n_cams: int, wh, workers: int | None = None,
                        procs: int | None = None) -> float:
    """The `--max-ram` that would hold a reader for EVERY camera of this rig.

    The inverse of `_reader_cache_size`, only possible because the price is LINEAR (a quadratic
    law does not invert into a useful number); undoes `FRACTION_READERS` and `DEFAULT_FRACTION`
    because `--max-ram` names the PROCESS, not the buffers.

    The process floor is on top, not inside: the ceiling that yields `need` GB of buffers is
    `max(need/fraction, need + floor)`. The result is nudged up by 1e-6 because
    `_reader_cache_size` ends in `int()` and the algebraically exact figure truncates one
    reader short of the rig -- which is the whole cliff.
    """
    k = _READER_GB_PER_MP * max(int(wh[0]) * int(wh[1]) / 1e6, 1e-3)
    share = max(workers or 1, 1) * max(int(procs or 1), 1)
    readers_gb = max(int(n_cams), 1) * k * share
    need = readers_gb / _memory.FRACTION_READERS
    floor = _memory.current().floor_gb
    exact = max(need / _memory.DEFAULT_FRACTION, need + floor)
    return exact * (1 + 1e-6)


def _reader_cache_size(n_cams: int, wh, workers: int | None, ram_gb: float | None = None,
                       procs: int | None = None) -> int:
    """How many open readers this process may hold.

    MUST NOT DECODE, STAT OR OPEN ANYTHING: opening a reader in the parent to measure anything is
    what deadlocks the forked workers -- it reads the environment, `/proc`, `/sys` and a camera
    count. Two axes: a single streaming process wants the WHOLE rig (a cycle below the camera
    count misses on every call); a loader worker wants 4 (`ChunkShuffle.mix`). The count is a
    wish, RAM the constraint, per PROCESS while the workers multiply it, and `procs` ranks
    multiply again. The price is LINEAR on PyAV (~0.053 GB/MP/reader), so RAM inverts into a real
    figure.

    The budget is the JOB's, not the machine's -- `SC_PHYS_PAGES` cannot see a cgroup cap, and
    an explicit `--max-ram` must bind here or it means nothing. The per-reader price is rounded
    UP so the estimate errs toward a smaller cache: the safe direction, since an over-estimate
    costs gigabytes and, inside workers, the job.
    """
    env = os.environ.get('TAILCYCLENET_READER_CACHE')
    if env:
        return max(1, int(env))
    if ram_gb is None:
        ram_gb = _memory.current().budget_gb
    if procs is None:
        procs = int(os.environ.get('TAILCYCLENET_LOCAL_WORLD_SIZE', '1') or 1)
    want = 4 if workers else max(int(n_cams), 4)
    k = _READER_GB_PER_MP * max(int(wh[0]) * int(wh[1]) / 1e6, 1e-3)
    share = max(workers or 1, 1) * max(int(procs), 1)
    n = int(max(_memory.FRACTION_READERS * ram_gb, 0.0) / share / k)
    return max(1, min(want, n))


# Built at the FIRST video read, not at import: the size is a function of the rig, and the rig is
# not known until a group is in hand.
#
# NOT AN LRU, AND THAT IS WORTH 2.4x OF WALL CLOCK. The access here is a CYCLE -- every window
# touches all `C` cameras, in the same order, window after window -- and a cycle is the
# pathological case for LRU: with capacity `k < C` the entry evicted is always the one needed
# next, so the hit rate is not `k/C`, it is ZERO. Simulated on the real pattern (16 cameras, 60
# passes): `lru_cache(11)` takes 0 hits in 960 accesses where random eviction takes 44.4%.
#
# Measured end to end on johnson before this was fixed: 16 readers ran the clip in 61 s and ELEVEN
# readers took 149 s, nearly as slow as two (222 s) -- eleven cached readers were buying nothing.
# A cache holding two thirds of a rig should pay a third of the misses, not all of them.
#
# RANDOM EVICTION rather than "pin the first k": pinning would fill with the first group's cameras
# and then never serve the second, and a deployment run walks many groups.
_readers = None
_cache_lock = threading.Lock()


class _ReaderCache:
    """At most `maxsize` open readers, evicting a RANDOM entry on overflow -- see above."""

    def __init__(self, maxsize: int):
        """A cache of at most `maxsize` open readers. Seeded so eviction does not vary run to run."""
        self.maxsize = max(1, int(maxsize))
        self.hits = self.misses = 0
        self._d: dict[str, object] = {}
        self._rng = random.Random(0)

    def __call__(self, path: str):
        """The open reader for `path`, opening it and evicting randomly when full.

        Inputs: path -- the video file path.
        Outputs: the open reader object.
        Side effects: opens a container and may drop an existing reader's last reference --
        dropping the last reference frees the decoder's frame pool, which is why the cache is a
        memory budget and not a handle count.
        """
        got = self._d.get(path)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        if len(self._d) >= self.maxsize:
            del self._d[self._rng.choice(list(self._d))]
        got = self._d[path] = _open_reader(path)
        return got

    def cache_clear(self):
        """Drop every cached reader."""
        self._d.clear()


def _open_reader(path: str):
    """One reader per file per process. Opening the container and building its frame index is not
    per-window work, but `read_frames` is called once per window per camera -- so a windowed pass
    over 3dpop's test videos paid it hundreds of times.

    **THE BACKEND IS `tailcyclenet/video.py`, AND THAT IS A MEMORY FIX.** PyAV keeps decoded
    frames bounded instead of loading whole containers into memory, which is survivable on a
    600-frame clip but matters on a 21 GB recording -- a 16-camera `--videos` run peaked at
    456 GB under `--max-ram 24`. PyAV is bit-identical on every video root here, so nothing
    downstream moves.
    """
    from . import video

    return video.open_reader(path)


def _reader(path: str, group, cam: str):
    """The cached `_open_reader`, sized on first use from the rig this group belongs to.

    `group.session` is not checked: `group.source(cam)` reached us through `Group.dir`, which
    already dereferences `self.session.path`, so a session-less Group cannot get this far.

    FIRST VIDEO READ WINS. A process that later walks a root with a wider rig keeps the size it
    picked, because resizing means dropping live readers.
    # ponytail: first-rig sizing, revisit if one process ever streams two video roots of different
    # widths -- the memory clamp bounds it either way.

    `_cache_lock` covers BOTH the late build and the lookup, because the cameras of one rig are now
    decoded concurrently (`detector.detect_group`): two threads racing the `is None` test would
    build two caches and the first one's readers would leak, and two cache misses on the same
    path would open the same 4K container twice. Opening is once per file and off the hot path, so
    holding the lock across it costs nothing measurable; the DECODE is not under it.

    THE DOCUMENTED CLIFF: the pose loop touches every camera inside one window, so a cache below
    the camera count misses on nearly every call -- measured on PyAV over 12 windows of a
    16-camera 3208x2200 rig (3 replicates): cache 4 runs 7.4 s against cache 16's 1.5 s (5.1x),
    and it is a THRESHOLD rather than a curve (1 -> 8 buys 24%, 8 -> 16 buys 4.2x), because a
    cycle only pays off once the whole cycle fits (the old 2.5x figure came from a different
    decoder). A silently 5x slower run should say so rather than being diagnosed from a
    stopwatch, and the warning says what it would take -- the price is LINEAR, so it inverts
    into an actual `--max-ram` figure -- while not blaming the budget when the env var forced
    the size.
    """
    global _readers
    with _cache_lock:
        if _readers is None:
            info = torch.utils.data.get_worker_info()
            rig = group.session.rig
            size = _reader_cache_size(
                len(rig), rig.size(cam), None if info is None else info.num_workers)
            if info is None and size < len(rig):
                forced = os.environ.get('TAILCYCLENET_READER_CACHE')
                why = (f'TAILCYCLENET_READER_CACHE={forced} forced this; unset it to size from '
                       'the RAM budget instead.' if forced else
                       f'--max-ram {reader_cache_ram_gb(len(rig), rig.size(cam)):.0f} would hold '
                       f'all {len(rig)}. Raise --max-ram / TAILCYCLENET_MAX_RAM_GB, or accept the '
                       'slowdown.')
                warnings.warn(
                    f'video reader cache is {size} for a {len(rig)}-camera rig: every window '
                    f'touches all {len(rig)} cameras, so this misses on nearly every call and '
                    f'reopens up to {len(rig) - size} container(s) per window. {why}',
                    stacklevel=2)
            _readers = _ReaderCache(size)
        return _readers(path)


# ONE LOCK PER CONTAINER AROUND THE DECODE, because `_readers` is a module-level cache of STATEFUL
# readers. `VideoReader.get_batch` seeks, so two threads reading the SAME container interleave their
# seeks and get each other's frames -- or crash inside the decoder. That did not matter while every
# caller was sequential; `scripts/infer.py` renders one group on a background thread while the loop
# predicts the next, and on a video-backed root both touch the same reader.
#
# IT USED TO BE ONE GLOBAL LOCK, WHICH ALSO SERIALISED DIFFERENT FILES -- and different files share
# no state at all, so that was a bound nothing needed. It cost the whole of the multi-camera win:
# 3dpop's four 3840x2160 cameras decode at ~62 ms/frame each on one thread and at ~18 ms/frame-camera
# when the four containers run concurrently (3.5x, measured; PyAV releases the GIL inside
# `get_batch`). A 16-camera rig is the same argument four times over. The per-path dict is guarded by
# `_lock_lock`, which is held only for the dict lookup and never across a decode.
_lock_lock = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def _read_lock_for(path: str) -> threading.Lock:
    """The one lock guarding reads of a given container path."""
    with _lock_lock:
        lk = _path_locks.get(path)
        if lk is None:
            lk = _path_locks[path] = threading.Lock()
        return lk


def _read_video(path, group, cam, frames, crop_coords, target_size, rotation):
    """Frames from a video file. Only 3dpop's test split needs this.

    Each DISTINCT index is decoded once: a clamp-padded window repeats its last frame, and a
    synthetic-motion window is ONE index T times under T different crops -- handing `get_batch`
    the repeats decodes them. A (T,4) `crop_coords` or a list of rotations is a MOVING crop: one
    affine per frame rather than one for the window, with the same count of `warpAffine` calls.
    A repeat gets a COPY, not a shared buffer: `_augment`'s cutout writes in place, and the
    dedupe would otherwise make repeats alias one buffer (same rule as `read_frames`).
    """
    import cv2

    key = str(path)
    want = [int(i) for i in frames]
    uniq = list(dict.fromkeys(want))
    with _read_lock_for(key):
        dec = _reader(key, group, cam).get_batch(uniq)
    at = {i: dec[n] for n, i in enumerate(uniq)}
    src_wh = (dec.shape[2], dec.shape[1])
    cc = None if crop_coords is None else np.asarray(crop_coords)
    per_box = cc is not None and cc.ndim == 2
    per_rot = isinstance(rotation, list)
    out, seen = [], set()
    for t, i in enumerate(want):
        aff = _crop_affine(src_wh,
                           cc[min(t, len(cc) - 1)] if per_box else crop_coords,
                           target_size,
                           rotation[min(t, len(rotation) - 1)] if per_rot else rotation)
        im = at[i]
        if aff is not None:
            out.append(cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR))
        else:
            out.append(im.copy() if i in seen else im)
        seen.add(i)
    return out


def read_frames(group, cam, frames, crop_coords=None, target_size=None, rotation=None,
                pool: ThreadPoolExecutor | None = None, reduce=1):
    """(T,H,W,3) uint8 RGB for one camera, from an image directory or a video.

    `reduce` is honoured for image directories and IGNORED for video: the video decoder has no
    decode-time decimation, so a video root silently returns full-size frames. The caller must
    therefore letterbox with `src_wh` and never assume the returned size.

    Names are computed, not listed (frame files are `%06d.<ext>` contiguous from 000000 by spec,
    §12) -- listing a directory to select T frames cost 0.90 s of a 1.06 s rat-city item. Each
    DISTINCT frame is decoded once: `_frames` clamp-pads a window that runs past its group's end,
    and a group shorter than `n_frames` pads entirely, so the dedupe turns 384 decodes into 16.

    Per-frame crops move the dedupe key: a per-frame box is a function of the POSITION in the
    window, not of the source index, so the key is (index, box, rotation slot) -- with one box
    and one rotation those halves are constant and this is the old behaviour exactly. But the
    dedupe key is not the decode key: a synthetic-motion window is ONE index under T distinct
    crops, so keys group by source index and `load_warps` decodes once per group. A repeat gets a
    COPY, not the same array object: `_augment`'s cutout writes in place, and an aliased list
    would make every repeat share one buffer (the copy is ~20 us against the 27 ms decode).
    """
    kind, src, ext = group.source(cam)
    if kind == 'video':
        return _read_video(src, group, cam, frames, crop_coords, target_size, rotation)
    want = [int(i) for i in frames]
    path = lambda i: os.path.join(src, f'{i:06d}{ext}')           # noqa: E731
    cc = None if crop_coords is None else np.asarray(crop_coords)
    per_box = cc is not None and cc.ndim == 2
    per_rot = isinstance(rotation, list)
    keys = [(w, tuple(int(v) for v in cc[min(t, len(cc) - 1)]) if per_box else None,
             min(t, len(rotation) - 1) if per_rot else None)
            for t, w in enumerate(want)]
    uniq = list(dict.fromkeys(keys))
    by_index = {}
    for k in uniq:
        by_index.setdefault(k[0], []).append(k)
    spec = lambda k: (list(k[1]) if per_box else crop_coords,          # noqa: E731
                      rotation[k[2]] if per_rot else rotation)
    batch = lambda i, ks: load_warps(path(i), [spec(k) for k in ks],   # noqa: E731
                                     target_size, reduce)
    if pool is None:
        res = {i: batch(i, ks) for i, ks in by_index.items()}
    else:
        fs = {i: pool.submit(batch, i, ks) for i, ks in by_index.items()}
        res = {i: f.result() for i, f in fs.items()}
    got = {k: im for i, ks in by_index.items() for k, im in zip(ks, res[i])}
    if any(v is None for v in got.values()):
        return [got[k] for k in keys]
    seen, out = set(), []
    for i in keys:
        out.append(got[i].copy() if i in seen else got[i])
        seen.add(i)
    return out


# appearance augmentation

def _crop_inflate(cfg, rng, train):
    """THIS item's crop-inflate factor. `cfg.crop_inflate` is a float, or a `[low, high]` pair --
    same contract as `_n_cams`/`cams_to_sample`, one continuous draw instead of `_n_cams`'
    integer one.

    TRAIN draws uniformly in `[low, high]` per item, so a wide-crop run shows the model a RANGE of
    context rather than one fixed geometry every step. VAL/TEST NEVER DRAW -- they get the range's
    MIDPOINT, for the same reason `_jitter` is gated to `self.train` alone: `checkpoint_best` is
    selected on the val curve, and a val geometry that moved step to step would put noise from the
    crop draw into a number that is supposed to isolate the weights. A plain scalar has no
    range to be a midpoint of and returns unchanged either way, so a scalar config -- everything
    shipped before this key existed a range -- is untouched.
    """
    spec = cfg.crop_inflate
    if not isinstance(spec, (list, tuple)):
        return float(spec)
    lo, hi = float(spec[0]), float(spec[1])
    return float(rng.uniform(lo, hi)) if train else (lo + hi) / 2.0


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

    EVEN because the scene encoder tokenises in tubelets of 2 (`gT = T // tubelet_size`), so an odd
    T silently drops tokens; FLOOR OF 2 because T = 1 gives `gT = 0` and a zero-length pos_embed.
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
            pts = p2d[cnum]
            inside = ((pts[..., 0] >= rx) & (pts[..., 0] <= rx + rw) &
                      (pts[..., 1] >= ry) & (pts[..., 1] <= ry + rh))
            vis_2d[:, :, cnum][inside] = 0
    return rects


# the dataset

@dataclass
class _Item:
    """One addressable training unit: an animal in a group, optionally at a fixed start."""
    ds: int
    session: object
    gid: str
    animal: int
    # -1 -> pick at random (train)
    start: int = -1


class PoseDataset(Dataset):
    def __init__(self, path, split: str, cfg: LoaderConfig, registry: Registry | None = None,
                 train: bool | None = None, seed: int = 23,
                 registry_base: Registry | None = None):
        """Build the window index for one split of a dataset (or folder of datasets).

        Scatters every session's parquet into dense arrays in the parent process so forked
        workers share them copy-on-write, resolves each session's keypoint axis against the
        registry, and refuses sessions whose label tables cannot supervise the requested mode.

        Inputs: path -- a dataset root or a folder of dataset roots.
                split -- 'train', 'val' or 'test'.
                cfg -- the loader configuration.
                registry -- an existing registry to use instead of building one.
                train -- override the train/val flag (defaults to split == 'train').
                seed -- the RNG seed for reproducible val/test sampling.
                registry_base -- a base registry whose ids must be preserved (warm start).
        Side effects: reads every label table; prints the box_source coverage per dataset.

        `n_frames` must be >= 2 (T = 1 gives posetail `gT = 0` and a zero-length
        pos_embed, which the clamp-pad does NOT cover); `box_source` is asserted against
        `BOX_SOURCES` (a typo would silently mean `keypoints`). `registry_base` makes the
        ids APPEND-ONLY so embedding rows survive warm start; `Registry.build` raises if
        an old id would move. The parquet is scattered HERE, in the parent process, so
        forked workers share the dense arrays copy-on-write. Keypoint ids are per SESSION,
        not per dataset (a session may reorder or subset the root's names), so an
        unmappable root fails at construction rather than mid-epoch; a mode/table mismatch
        is refused here too (`_item` picks its target off `sess.mode` alone, and a 3d
        session carrying only keypoints.pq would crash mid-epoch). With
        `box_source = 'instances'` the roots the switch actually reached are printed (a
        root with no `instances.pq` silently falls back to keypoints). Balancing across
        datasets is train-only, so a window's identity stays tied to its index.
        """
        assert cfg.n_frames >= 2, (
            f'n_frames = {cfg.n_frames} is not usable: posetail computes gT = T // tubelet_size '
            '(encoder_decoder.py:748), which is 0 at T=1 and yields a zero-length pos_embed. '
            'Use n_frames >= 2; short groups are clamp-padded up to it.')
        assert cfg.box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {cfg.box_source!r}'
        self.cfg = cfg
        self.split = split
        self.train = (split == 'train') if train is None else train
        self.datasets = load_datasets(path)
        self.registry = registry or Registry.build(self.datasets, registry_base)
        self.seed = seed
        self._aug = _build_augmenters(cfg) if self.train and cfg.aug_prob > 0 else None

        self.index: list[_Item] = []
        self.by_dataset: list[list[int]] = []
        self._kpt_ids: dict[Path, torch.Tensor] = {}
        boxed: list[tuple[str, int, int]] = []
        for di, ds in enumerate(self.datasets):
            mine, n_box, n_sess = [], 0, 0
            for sess in ds.sessions.get(split, []):
                sess.preload()
                n_sess += 1
                self._kpt_ids[sess.path] = torch.as_tensor(
                    self.registry.ids_for(ds.name, sess.names), dtype=torch.long)
                for gid, group in sess.groups.items():
                    lab = sess.labels(gid)
                    need = 'points3d' if sess.mode == '3d' else 'points2d'
                    if getattr(lab, need) is None:
                        raise ValueError(
                            f'{sess.path}: mode is {sess.mode!r} so every window needs '
                            f'{need}, and this session carries none. The format allows it '
                            '(one label table is enough) but training on it does not: there is '
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
        if not self.index:
            raise ValueError(f'{path}: split {split!r} yielded no usable windows')
        if cfg.box_source == 'instances':
            print(f'{split}: box_source=instances  ' + '  '.join(
                f'{n}/{t} {name}' + ('' if n == t else ' (keypoint fallback)')
                for name, n, t in boxed))
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
        return sorted({min(s, limit) for s in range(first, hi + 1, stride)}) or [0]

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

    def mix(self):
        """Realised share of train steps per (label_source, mode) cell. Reporting only: printed
        at startup because the mix is invisible in the loss curve.
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
        pool, cum = (self._pools[rng.integers(len(self._pools))] if len(self._pools) > 1
                     else self._pools[0])
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
        reproducible.
        """
        ordinal, idx = idx if isinstance(idx, tuple) else (None, idx)
        rng = np.random.default_rng(None if self.train else (self.seed, idx))
        shape_rng = (rng if ordinal is None
                     else np.random.default_rng((self.seed, 0x5AFE, int(ordinal))))
        shape = self._shape(shape_rng)
        for _ in range(8):
            out = self._item(idx, rng, shape)
            if out is not None:
                return out
            idx = int(rng.integers(len(self.index)))
        raise RuntimeError(f'{self.split}: 8 consecutive items failed to build')

    def _shape(self, rng) -> dict:
        """The cost-determining draws, made from a stream every rank shares. See `__getitem__`.

        Exactly the two draws that change how much WORK an item is and nothing else: the camera
        count and the single-view coin. The cell, session, start, T and every augmentation stay
        on the item's own stream.
        """
        return {'n_cams': _n_cams(self.cfg.cams_to_sample, rng),
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

        On train, a stride is admissible if the group holds the FLOOR window of two frames at it;
        T is then capped by what the group has room for on that anchor's lattice. The window is
        anchored on a labelled frame and placed AROUND it (the old loader required the FIRST
        frame to be labelled, silently discarding mid-group labels), and T is shrunk to the
        labelled frames some placement could reach -- off-lattice labels are unreachable at this
        stride. If the span exceeds the ceiling, no placement covers it, so the draw falls back
        to the anchor and lets the placement pick which end of the span it lands on. Bounds COVER
        first..last rather than merely containing the anchor -- sizing T to a span and placing
        the window off it would pay for frames it never reads.
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

    def _item(self, idx, rng, shape=None):
        """Build one window's tensors, or None when the item cannot be built.

        Inputs: idx -- the index to pick.
                rng -- the item's RNG stream.
                shape -- a pre-drawn cost-determining shape dict (see `_shape`).
        Outputs: the 13- or 14-field item tuple `__getitem__` hands to the collate, or None.

        Visibility: `vis` (3D noisy-OR) is unused at R == 2; `vis_2d` is the per-camera
        THREE-STATE target (NaN = "not assessed", masked out of the BCE; `projected` joins
        UNLABELED), withheld when `sess.has_visibility_assessment` is False or every
        assessed row is projected/unlabeled; both-or-neither.

        Geometry: camera and crop-inflate draws are one per item (camera draw sorted).
        Points outside the source frame or the FINAL crop are flipped out of `vis_2d`;
        a rotation that loses the animal to the inscribed crop is REVERTED, not retried.

        Pixels: appearance augmentation runs on the final ~256 px crops; views are
        UINT8 (4x fewer bytes to collate/queue/pin; the model divides on device).

        The query prior: `kpt_prior` is the pose at the prompt frame (GT at training,
        the previous window's own prediction at deployment); `prompt_t` is the first
        labelled frame; `prompt_dropout` is PER ITEM, not per keypoint. Corruptions,
        in order: exposure bias, noise/offset in PIXELS, stale priors,
        `prompt_swap_animal`, `prompt_swap_kpt_pairs` (finite entries independently
        replaced by another keypoint's ORIGINAL position), and a whole-body offset --
        ONE vector per item.

        The stride is read back off `frames` (MEDIAN: a group-edge window repeats its
        last frame); the final 3D noisy-OR is over the FINAL `vis_2d`.
        """
        shape = shape or self._shape(rng)
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
        inflate = _crop_inflate(self.cfg, rng, self.train)

        true_2d = sess.mode == '2d'
        single_view = (not true_2d and self.train
                       and self.cfg.prob_2d_only > 0
                       and shape['single_view_draw'] < self.cfg.prob_2d_only)

        if true_2d:
            cam_ix = [0]
        elif single_view:
            cam_ix = [int(rng.integers(len(cgroup)))]
        else:
            n = shape['n_cams']
            cam_ix = (sorted(rng.choice(len(cgroup), n, replace=False)) if 0 < n < len(cgroup)
                      else list(range(len(cgroup))))
        cgroup = [cgroup[i] for i in cam_ix]
        cam_names = [sess.cam_names[i] for i in cam_ix]
        crop_pts = self._crop_pts(lab, a, frames, cam_ix)

        if true_2d:
            coords = torch.as_tensor(lab.points2d[a][frames][:, :, 0], dtype=torch.float32)
            vis = vis_2d = None
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]
                vis_2d = torch.as_tensor(np.where(np.isin(v2, (UNLABELED, PROJECTED)), np.nan,
                                                  (v2 == VISIBLE).astype(np.float32)))
                if not torch.isfinite(vis_2d).any():
                    vis_2d = None
        else:
            coords = torch.as_tensor(lab.points3d[a][frames], dtype=torch.float32)
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]
                vis_2d = torch.as_tensor(np.where(np.isin(v2, (UNLABELED, PROJECTED)), np.nan,
                                                  (v2 == VISIBLE).astype(np.float32)))
                vis = torch.as_tensor((v2 == VISIBLE).any(-1))
                if not torch.isfinite(vis_2d).any():
                    vis = vis_2d = None
            else:
                vis = vis_2d = None

        if torch.isfinite(coords).all(-1).sum() < 2:
            return None

        rot_p = (self.cfg.aug_prob if self.cfg.aug_rotation_prob is None
                 else self.cfg.aug_rotation_prob)
        rot_deg = self.cfg.aug_rotation_deg
        rotation_info = [None] * len(cgroup)
        neighbour_full = None
        if true_2d:
            cam = cgroup[0]
            coords = _mask_outside(coords, cam['size'])
            if vis_2d is not None:
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            cp = None if crop_pts is None else crop_pts[:, 0]
            if self.train and rng.random() < rot_p:
                cam, coords, rot = _rotate_2d(cam, coords,
                                              float(rng.uniform(-rot_deg, rot_deg)))
                rotation_info = [rot]
                cp = _apply_affine(cp, rot)
            jit = self._jitter(rng)
            cam, box, coords = cropmod.crop_to_points_2d(cam, coords, self.cfg.min_crop_dim,
                                                         jit, crop_pts=cp,
                                                         inflate=inflate)
            if cam is None:
                return None
            cam, scale = _resize_camera(cam, self.cfg.image_size)
            coords = coords * scale
            coords = _mask_outside(coords, cam['size'])
            if vis_2d is not None:
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            if attempt_swap_animal:
                raw = torch.as_tensor(lab.points2d[neighbour_row][frames][..., 0, :],
                                      dtype=torch.float32)
                raw = _apply_affine(raw, rotation_info[0])
                neighbour_full = (raw - box[:2].to(raw.dtype)) * scale
            cgroup, boxes = [cam], [box]
            p2d = p2d_all = coords[None]
            R = 2
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
                            vis_2d[:, :, cnum][~is_point_visible(cam, coords)] = 0
            others_raw = None
            if attempt_swap_animal:
                others_raw = torch.as_tensor(lab.points3d[neighbour_row][frames],
                                             dtype=torch.float32)
            if self.train:
                cgroup, coords, others_raw = _rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            cp3 = None if crop_pts is None else [
                _apply_affine(crop_pts[:, i], rotation_info[i]) for i in range(len(cgroup))]
            jit = self._jitter(rng)
            cgroup, boxes = cropmod.crop_to_points_3d(cgroup, coords, self.cfg.min_crop_dim,
                                                      jit, crop_pts=cp3,
                                                      inflate=inflate)
            if cgroup is None:
                return None
            cgroup = [_resize_camera(c, self.cfg.image_size)[0] for c in cgroup]
            if self.train:
                cgroup, coords, others_raw = _rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            if others_raw is not None:
                neighbour_full = others_raw
            p2d_all = (project_points_torch(cgroup, coords)
                       if single_view or self._aug is not None else None)
            p2d = p2d_all if single_view else None
            R = 3

        gray = self._aug is not None and rng.random() < self.cfg.grayscale_prob
        use_pool = group.source(cam_names[0])[0] != 'video'
        with (ThreadPoolExecutor(max_workers=16) if use_pool else nullcontext()) as pool:
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
                views.append(torch.from_numpy(np.asarray(imgs)))

        finite = torch.isfinite(coords).all(-1)
        prompt_t = torch.where(finite.any(0), finite.float().argmax(0), torch.zeros(K).long())
        prompt_t = prompt_t.to(torch.int32)
        kpt_prior = coords[prompt_t, torch.arange(K)].clone()
        kpt_prior[~finite.any(0)] = float('nan')
        if self.train and rng.random() < self.cfg.prompt_dropout:
            kpt_prior[:] = float('nan')
        px = 1.0
        if R == 3 and (self.cfg.prompt_noise_px > 0 or self.cfg.prompt_offset_px > 0) \
                and bool(torch.isfinite(kpt_prior).any()):
            pts = kpt_prior[torch.isfinite(kpt_prior).all(-1)][None]
            px = float(torch.nanmedian(get_camera_scale(cgroup, pts)))
            if not np.isfinite(px):
                px = 1.0
        if self.train and self.cfg.prompt_stale_frames > 0 \
                and bool(torch.isfinite(kpt_prior).any()):
            d = int(rng.integers(0, int(self.cfg.prompt_stale_frames) + 1))
            if d:
                t_alt = torch.clamp(prompt_t.long() + d, max=coords.shape[0] - 1)
                alt = coords[t_alt, torch.arange(K)]
                swap = torch.isfinite(alt).all(-1) & torch.isfinite(kpt_prior).all(-1)
                kpt_prior = torch.where(swap[:, None], alt, kpt_prior)
        if neighbour_full is not None:
            mode_str = '2d' if R == 2 else '3d'
            neighbour_prior = neighbour_full[prompt_t, torch.arange(K)].clone()
            neighbour_prior[prior_out_of_bounds(neighbour_prior, mode_str, cgroup)] = float('nan')
            jump = (torch.as_tensor(rng.random(K)) < self.cfg.prompt_swap_animal) \
                & torch.isfinite(neighbour_prior).all(-1)
            kpt_prior = torch.where(jump[:, None], neighbour_prior, kpt_prior)
        if self.train and self.cfg.prompt_swap_kpt_pairs > 0:
            finite_idx = torch.isfinite(kpt_prior).all(-1).nonzero(as_tuple=True)[0]
            m = len(finite_idx)
            if m >= 2:
                sel = torch.as_tensor(rng.random(m)) < self.cfg.prompt_swap_kpt_pairs
                if bool(sel.any()):
                    local = torch.arange(m)[sel]
                    offset = torch.from_numpy(rng.integers(1, m, size=int(sel.sum())))
                    src_idx = finite_idx[(local + offset) % m]
                    original = kpt_prior
                    kpt_prior = kpt_prior.clone()
                    kpt_prior[finite_idx[sel]] = original[src_idx]
        if self.train and self.cfg.prompt_offset_px > 0 and bool(torch.isfinite(kpt_prior).any()):
            kpt_prior += torch.as_tensor(
                rng.normal(0.0, float(self.cfg.prompt_offset_px) * px, (1, R)),
                dtype=kpt_prior.dtype)
        if self.train and self.cfg.prompt_noise_px > 0 and bool(torch.isfinite(kpt_prior).any()):
            kpt_prior += torch.as_tensor(
                rng.normal(0.0, float(self.cfg.prompt_noise_px) * px, kpt_prior.shape),
                dtype=kpt_prior.dtype)

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
            from . import box_prompt as bpmod
            box = bpmod.compute_box_prompt(coords, cgroup, '2d' if R == 2 else '3d')
            box = bpmod.apply_frames_mode(box, self.cfg.box_prompt_frames)
            if self.train:
                if self.cfg.box_prompt_dropout > 0 and rng.random() < self.cfg.box_prompt_dropout:
                    box = torch.full_like(box, float('nan'))
                elif self.cfg.box_prompt_jitter > 0 or self.cfg.box_prompt_scale_jitter > 0:
                    box = bpmod.apply_jitter(box, rng, self.cfg.box_prompt_jitter,
                                             self.cfg.box_prompt_scale_jitter)
            out.append(box)
        return tuple(out)

    def _augment(self, imgs, cnum, size, p2d, vis_2d, gray, rng):
        """Appearance augmentation for one camera's T crops. `vis_2d` is mutated by cutout.

        `to_deterministic()` freezes this camera's sampled parameters so every frame gets the
        SAME gamma/hue; only the apply is per-frame.
        """
        import cv2

        per_camera, per_image = self._aug
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
        """A crop jitter box for this item, or None when jitter is off (val/test)."""
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
    """Scale a camera so its long side is `target_res`. Returns (cam, scale).

    A zero side is the WHOLE frame to cv2.warpAffine (a 0 in `dsize` means "use source size"),
    silently returning the full frame while the camera dict claims a sliver -- hence the clamp
    to 1.
    """
    cam = dict(cam)
    size = cam['size']
    scale = float(target_res) / float(max(size))
    cam['size'] = torch.round(size * scale).to(torch.int32).clamp_min(1)
    cam['mat'] = cam['mat'] * scale
    cam['mat'][2, 2] = 1
    cam['offset'] = cam['offset'] * scale
    return cam, scale


class StepSampler(torch.utils.data.Sampler):
    """Draws indices with replacement and yields `(ordinal, index)`.

    The ordinal is the global training step -- the SAME on every rank while the index is not --
    and `PoseDataset.__getitem__` keys its cost-determining draws on it. Replacement, and no
    `DistributedSampler`: an index entry is one (session, group, animal) whatever the group's
    length, so it is a poor sampling weight and `_pick` re-draws inside `__getitem__`;
    independent per-rank streams is what the single-gpu loop already did.
    """

    def __init__(self, n: int, num_samples: int, generator=None, start: int = 0):
        """A sampler yielding `(ordinal, index)` pairs, drawing indices with replacement.

        Inputs: n -- the number of index entries.
                num_samples -- how many steps to yield.
                generator -- torch RNG for the index draws.
                start -- the first ordinal (the global step).
        """
        self.n, self.num_samples, self.generator, self.start = int(n), int(num_samples), generator, int(start)

    def __len__(self):
        """Number of steps this sampler will yield."""
        return self.num_samples

    def __iter__(self):
        """Yield `(start + k, index)` for k over `num_samples` random draws."""
        idx = torch.randint(0, self.n, (self.num_samples,), generator=self.generator)
        for k, i in enumerate(idx.tolist()):
            yield (self.start + k, i)


def worker_init(worker_id):
    """A DataLoader `worker_init_fn`: pin cv2's and torch's thread pools to one thread, reseed
    imgaug (its global RNG survives fork otherwise, so every worker draws the same augmentations),
    and force imgaug's hue LUT cache to build here (a `spawn`ed dataset never re-runs `__init__`).
    numpy is deliberately NOT reseeded -- torch already decorrelates it per worker.
    """
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        import imgaug
        import imgaug.augmenters as iaa
    except ImportError:
        return
    imgaug.random.seed(int(np.random.randint(2 ** 31)))
    iaa.AddToHue()


def pose_collate(batch):
    """posetail's collate for the first ten fields, plus this repo's three.

    `custom_collate` keeps only item 0's `cgroup` and forbids mixing 2D and 3D -- which is why
    batch_size is structurally 1 PER RANK. The same reason covers K: sessions may carry different
    keypoint subsets, so a mixed batch would fail the `kpt_ids` stack. Loud, and unreachable at
    batch_size 1.

    The box prompt field exists only when a box model is training (`_item` appends a 14th field
    then); it is absent for a plain run, so nothing downstream sees it.
    """
    batch = [b for b in batch if b is not None]
    out = custom_collate([b[:10] for b in batch])
    out['kpt_ids'] = torch.stack([b[10] for b in batch])
    out['kpt_prior'] = torch.stack([b[11] for b in batch])
    out['prompt_t'] = torch.stack([b[12] for b in batch])
    if len(batch[0]) > 13:
        out['box_prompt'] = torch.stack([b[13] for b in batch])
    return out
