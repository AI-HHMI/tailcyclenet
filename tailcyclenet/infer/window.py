"""The inference path. One window loop.

Per group: for each window of T frames stepping by T - overlap, get a crop box per camera, read
the pixels, build the prior (none / carry / self / labels), forward, and map the prediction back
into the source coordinate frame. `carry` seeds from the model's own previous prediction and
needs `overlap >= 1`; `self` re-queries within one window and carries nothing; `labels` is a GT
oracle, not a deployment number.
"""
from __future__ import annotations

import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..data import crop as cropmod
from .. import memory
from ..data.camera import resize_camera
from ..pose.prompt import self_prompt
from ..format import Labels, Session
from ..pose.model import share_scene
from .crops import (boxes_from_points, crop_views, fill_box_agreement, overlaps,
                    to_device)
from .prior import build_prior, deploy_box_prompt, give_box_this_window
from .store import FrameStore
from .traces import (FRAME_KEYS, WINDOW_KEYS, capture_model_confidence,
                     capture_overlap_agreement, census_block_boundary, count_prior_oob)
from .window_records import logit_probability, window_record_base

ANCHORS = ('none', 'carry', 'self', 'labels')
CARRY_SOURCES = ('triangulate', 'pred')

# One animal's crop plan for one window: `use` is the camera indices the crop touched, `boxes`
# their (int32) crop boxes, `cgroup` the resized cameras the forward reads, `scales` each
# camera's resize factor, `uncropped` the pre-resize cameras `--refine`'s fallback rebuilds from.
# A plain tuple here would let a future field insertion silently reorder at one of the several
# construction sites without the others noticing; unpacking a WindowPlan is unchanged from
# unpacking a tuple, so every existing positional unpack site is untouched.
WindowPlan = namedtuple('WindowPlan', 'animal use boxes cgroup scales uncropped')

# Why an (animal, window) produced nothing -- separate codes so coverage loss is attributable.
OUTCOMES = ('ok', 'no box', 'no camera', 'no points', 'crop failed', 'decode failed')

# How many cameras `decode_crops` may decode at once. A memory bound, not a core count -- see there.
_CAM_DECODE = 4

# Smallest store worth building: avoids per-block overhead on small-frame roots.
_MIN_STORE_BYTES = 512 << 20


@dataclass
class InferConfig:
    n_frames: int = 24
    overlap: int = 4
    image_size: int = 256
    min_crop_dim: int = 64
    anchor: str = 'carry'
    # 0 -> every animal the box source offers
    max_animals: int = 0
    # 0 -> the whole group; else its first `max_frames` frames
    max_frames: int = 0
    # THE FRAME RANGE, half-open [frame_start, frame_stop). A window-loop lever, not an
    # input-format one. `frame` in the output is always the SOURCE index and rows exist only
    # inside the range (the spec's sparsity rule), so `load_predictions` densifies back to a
    # full-length array that is NaN outside it. One quantity with `max_frames` (`--max-frames N`
    # IS `--start-frame 0 --end-frame N`), resolved in `run_blocks`.
    frame_start: int = 0
    # 0 -> to the end of the group
    frame_stop: int = 0
    # 0 -> decode every keypoint in one pass
    kpt_chunk: int = 0
    # None -> report every row. A float withholds a row whose median `vis_pred` logit across
    # keypoints is below it; not portable across roots, so there is no default.
    vis_thresh: float | None = None
    # Re-crop each window to the first pass's own prediction and predict again; costs one extra
    # forward and decode per animal per window. None -> on in 3D, off in 2D (the two disagree).
    refine: bool | None = None
    # Pass 1's input resolution under `--refine`; None -> `image_size`. Refine's gain is
    # magnification, not coordinate frame, so pass 1 only has to localise. No shipped default.
    refine_px: int | None = None
    # How many finite (frame, camera) boxes a row needs before it gets a window crop; raising it
    # deliberately lowers reported coverage (one box used to fabricate a whole window's).
    min_box_frames: int = 1
    # What `carry` feeds back: 'triangulate' is the anchor-free estimate (breaks the feedback
    # loop); 'pred' is the reported prediction, which under gridresid_offset = "query" is
    # `prior + residual` and integrates its own error. 2D is identical either way.
    carry_source: str = 'triangulate'
    # Deliberately break the oracle prior to measure the echo coefficient; see `corrupt_prior`.
    # Never a deployment arm.
    oracle_corrupt: str | None = None
    device: str = 'cuda:0'
    # From the run's own `[data]`, never a CLI flag: scoring against a crop rule the model never
    # saw would be silent otherwise.
    box_source: str = 'keypoints'
    # Deployment box prompt, per camera, 2D or 3D: 'none' | 'labels' (GT, an oracle) |
    # 'detector'. 'none' + crop_inflate 1.0 is byte-identical to a run without these keys. A
    # camera with no finite point gets a NaN column, which the learned no-box token substitutes.
    box_prompt: str = 'none'
    # Owner-set default (2026-09-05): give the box prompt at each animal's FIRST window only,
    # withholding it once a carry is ACTUALLY USED as this window's prior -- give it while
    # first-detecting, then let carry hold identity. False (the pre-existing behaviour) gives the
    # box prompt every window `box_prompt` is on for; the gate is `not box_prompt_first_only or
    # carried[a] is None or prior is None` (fixed: a stale/out-of-bounds carry that `build_prior`
    # itself declined to use no longer silently withholds the box too -- report 56 Session 9/13's
    # cross-clip-replicated finding), which reduces to the old unconditional check when this is
    # False.
    box_prompt_first_only: bool = False
    # Inflate every crop about its centre (the wide pass-1 regime); 1.0 is the unchanged behaviour.
    crop_inflate: float = 1.0
    # How many windows ahead to decode while the current one forwards. Bit-exact at any value:
    # `_build_plans`/`decode_crops` never read `carried`, which is touched only in window order
    # on the main thread. 0 is the exact old serial path.
    prefetch_windows: int = 1
    # Retain block-local long-form pre-gate records for the optional window sidecar.
    capture_window_predictions: bool = False


def window_starts(span: int, window: int, overlap: int, start: int = 0):
    """Starts of `window`-frame windows covering [start, start + span), stepping by
    `window - overlap`.

    The last window is pulled back to end at the last frame rather than padded, so no frame is
    predicted from duplicated pixels. `start` and the returned starts are SOURCE indices; at the
    default 0 the values are what they always were, so an unset range is byte-identical.
    """
    step = max(1, window - overlap)
    if span <= window:
        return [start]
    starts = list(range(start, start + span - window + 1, step))
    if starts[-1] + window < start + span:
        starts.append(start + span - window)
    return starts


def last_window_start(starts, window: int, t: int) -> int | None:
    """The seam rule: frame `t` belongs to the LAST window containing it -> that window's start.

    `starts` are ascending window starts and `window` the window length. None when no window
    covers `t` (before the first start, past the last window's end, or no windows at all).
    """
    starts = np.asarray(starts)
    pos = int(np.searchsorted(starts, t, side='right'))
    if pos == 0:
        return None
    start = int(starts[pos - 1])
    return start if t < start + window else None


def _plan_blocks(starts, n_frames, T_total, frame_cost, store_bytes):
    """Group the windows into blocks whose frames fit the store at once -> [(w0, w1), ...].

    Greedy and maximal: bigger blocks mean fewer boundaries, each costing one prefetch stall and
    a re-read of the seam frames. Always at least one window -- `run_blocks` refuses before this.
    """
    blocks, w0 = [], 0
    while w0 < len(starts):
        m = 1
        while w0 + m < len(starts):
            span = min(T_total, starts[w0 + m] + n_frames) - starts[w0]
            if span * frame_cost > store_bytes:
                break
            m += 1
        blocks.append((w0, w0 + m))
        w0 += m
    return blocks


def _plan_windows(session_id, gid, cam_sizes, n_frames, overlap, T_total, frame_start,
                 budget=None):
    """Pure prefix of `run_blocks`: window starts, the frame/store budget, and the block
    partition. -> (starts, blocks, cam_decode, pipeline_det). Raises `SystemExit` when one
    window's frames do not fit the budget -- there is no re-decode fallback.

    Inputs: session_id/gid -- named only in the refusal message; cam_sizes -- (w, h) per camera
        actually in use this run; n_frames/overlap -- `cfg.n_frames`/`cfg.overlap`; T_total/
        frame_start -- the source frame range; budget -- a `memory.Budget`, `memory.current()`
        when omitted (the production path; a caller injects one to test this function without
        reading the real machine).
    Side effects: none. Never allocates, decodes, or opens a container -- this only does
    arithmetic on sizes already known from parsed config and the rig.
    """
    starts = window_starts(T_total - frame_start, n_frames, overlap, start=frame_start)
    _frame_bytes = max(int(w) * int(h) for w, h in cam_sizes) * 3
    _frame_cost = len(cam_sizes) * _frame_bytes
    _budget = memory.current() if budget is None else budget
    _one = n_frames * _frame_cost
    _want_store = (float('inf') if _budget.stated else max(2 * _one, _MIN_STORE_BYTES))
    _share = _budget.share(memory.FRACTION_STORE)
    _pipeline_det = _share >= 2 * _one
    _store_bytes = min(_share / (2 if _pipeline_det else 1), _want_store)
    cam_decode = memory.fits(_store_bytes / 2, _frame_bytes * n_frames,
                             want=min(_CAM_DECODE, len(cam_sizes)))
    if _one > _store_bytes:
        raise SystemExit(
            f'{session_id}/{gid}: one window of frames does not fit, and there is no '
            're-decode fallback.\n'
            f'    {n_frames} frames x {len(cam_sizes)} camera(s) x {_frame_bytes / 1e6:.1f} MB '
            f'= {_one / (1 << 30):.2f} GB\n'
            f'    frame store {_store_bytes / (1 << 30):.2f} GB  '
            f'({_budget.budget_gb:.1f} GB budget x {memory.FRACTION_STORE:g}, {_budget.source})\n'
            f'  --max-ram {_one / (memory.FRACTION_STORE * memory.DEFAULT_FRACTION) / (1 << 30):.0f}'
            '   holds one window. The flag is a ceiling on the PROCESS, not an allowance for the '
            'buffers, so the budget is a fraction of it.\n'
            f'  --n-frames N     a shorter window needs proportionally less -- but it is NOT '
            f'output-neutral: this run was trained at {n_frames} and a shorter window is a '
            'different prediction.\n'
            'Refine pass 2 crops from the SAME frames pass 1 did, so a window\'s frames cannot be '
            'dropped and re-read: re-decoding would double the wall clock, and re-cropping from '
            'the stored crop double-resamples and is not output-neutral.')
    blocks = _plan_blocks(starts, n_frames, T_total, _frame_cost, _store_bytes)
    return starts, blocks, cam_decode, _pipeline_det


def merge_blocks(blocks):
    """Every block of `run_blocks` stitched into the one dict `run_group` used to return.

    Frame- and window-indexed columns concatenate (axis 0 for the 1-D `window_start`); everything
    else is a per-group constant from the first block.
    """
    blocks = list(blocks)
    out = dict(blocks[0])
    for k in FRAME_KEYS + WINDOW_KEYS:
        if k not in out:
            continue
        axis = 0 if np.asarray(out[k]).ndim == 1 else 1
        out[k] = np.concatenate([b[k] for b in blocks], axis=axis)
    if 'window_records' in out:
        out['window_records'] = [r for block in blocks for r in block.get('window_records', [])]
    return out


@torch.no_grad()
def run_group(model, session: Session, gid: str, registry, dataset_name: str,
              cfg: InferConfig, box_points=None, boxes_for=None, n_rows=None, *, scorer=None,
              shared_scene_encoder=False) -> dict:
    """`run_blocks` for a whole group, merged. See both for what a block is."""
    return merge_blocks(run_blocks(model, session, gid, registry, dataset_name, cfg,
                                   box_points=box_points, boxes_for=boxes_for, n_rows=n_rows,
                                   scorer=scorer, shared_scene_encoder=shared_scene_encoder))


@torch.no_grad()

def vis_gate(v2, vlogit, thresh):
    """The `--vis-thresh` gate for one (animal, window): which rows are withheld.

    Per camera: a camera's 2D row at a frame is gated when the median over keypoints of its own
    visibility logit (`vis_pred_2d`) is below `thresh`; the frame's 3D row is gated only when EVERY
    camera is (per-camera AUROC 0.975 vs 0.687 for the noisy-OR aggregate, dev/plans/vis_thresh_head.md).
    Without per-camera logits it falls back to the median of the aggregate `vis_pred`.

    Inputs: v2 -- (C, T, K) per-camera logits or None; vlogit -- (T, K) aggregate logits or None.
    Outputs: (drop (T,) bool, cam_drop (C, T) bool or None).
    """
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        if v2 is not None:
            cam_drop = ~(np.nanmedian(np.asarray(v2, dtype=np.float64), axis=-1) >= thresh)
            return cam_drop.all(axis=0), cam_drop
        return ~(np.nanmedian(vlogit, axis=-1) >= thresh), None

def run_blocks(model, session: Session, gid: str, registry, dataset_name: str,
               cfg: InferConfig, box_points=None, boxes_for=None, n_rows=None, stats=None, *,
               scorer=None, shared_scene_encoder=False):
    """Predict every animal in one group, a block of windows at a time. Yields one dict per block.

    Arrays are in the SOURCE coordinate frame. Crops come from exactly one of two sources, not
    comparable: `box_points` (S,T,K,R) points the crop rule follows (the labels themselves are
    the GT-crop upper bound), or `boxes_for(store, lo, hi) -> (boxes, scores)` from a
    detector or detections file (the deployment number); whichever was used is recorded.

    A block owns a bounded frame span, so nothing is proportional to the clip's length. A block's
    frames are exactly `[starts[w0], starts[w1])`, partitioning the clip -- a frame in an overlap
    belongs to the last window containing it, so seam frames are dropped here and block output is
    byte-identical to whole-clip output. `boxes_for` is a callback so the caller detects only the
    frames it needs; `n_rows` is `S` on that path.

    `refine` defaults by dimensionality, resolved and folded into `cfg` here; `T_total` is the
    SOURCE STOP INDEX (a bound, not a count). `ids_for` aligns the per-session keypoint axis to
    the registry by name. Detector rows are not label rows (score- or association-ordered), so
    every row wears an invented id. `carried` holds the per-animal prior for the next window
    (read live out of `out` in `forward()`); `outcome` holds per-(animal, window) diagnostics.

    The pixel budget is a ceiling sized from parsed toml, never by opening a container: a STATED
    budget (--max-ram) is a grant, an inferred one is spare memory and the cap stands; a budget
    too small for ONE window is refused (no re-decode fallback). Block-local state is rebound
    once per block; arrays are indexed `[a, frame - f0]` or `[a, wi - w0]`. The driver decodes
    window wi+1 while wi forwards, bounded by `cfg.prefetch_windows` -- bit-exact, only
    wall-clock overlap is new. The store is evicted by window position, not LRU; `memory.trim()`
    runs at window boundaries (peak checked at block boundaries; warns once, never kills).
    Telemetry goes in `stats`, never the block dict. In the yielded dict, `n_frames` is a COUNT,
    not the stop index, and `refine` is the resolved value.
    """
    assert cfg.anchor in ANCHORS, f'anchor must be one of {ANCHORS}'
    assert cfg.carry_source in CARRY_SOURCES, \
        f'carry_source must be one of {CARRY_SOURCES}, got {cfg.carry_source!r}'
    if cfg.anchor == 'carry' and cfg.overlap < 1:
        raise ValueError("anchor='carry' carries a pose across windows and needs "
                         'overlap >= 1; got 0')

    group = session.groups[gid]
    lab: Labels = session.labels(gid)
    mode = session.mode
    K = session.n_keypoints
    R = 3 if mode == '3d' else 2
    if cfg.refine is None:
        cfg = replace(cfg, refine=(mode == '3d'))
    pass1_res = cfg.refine_px if (cfg.refine and cfg.refine_px) else cfg.image_size
    frame_start = max(0, int(cfg.frame_start))
    T_total = min(group.n_frames, int(cfg.frame_stop or cfg.max_frames or group.n_frames))
    if frame_start >= T_total:
        raise ValueError(
            f'{session.session_id}/{gid}: frame range [{frame_start}, {T_total}) is empty -- the '
            f'group has {group.n_frames} frames. The driver skips such a group by name; reaching '
            'here means the range was set programmatically.')
    kpt_ids = torch.as_tensor(registry.ids_for(dataset_name, session.names),
                              dtype=torch.long)[None]
    assert kpt_ids.shape[1] == K

    src = (box_points if box_points is not None else
           None if boxes_for is not None else
           (lab.points3d if mode == '3d' else lab.points2d[..., 0, :]))
    inst_boxes = (lab.boxes if (boxes_for is None and box_points is None
                                and cfg.box_source == 'instances' and lab.boxes is not None
                                and bool(np.isfinite(lab.boxes).any())) else None)
    n_src = (n_rows if boxes_for is not None else src.shape[0])
    S = n_src if cfg.max_animals == 0 else min(n_src, cfg.max_animals)
    cam_ix = [0] if mode == '2d' else list(range(len(session.rig)))
    n_lab = 0 if src is None else len(src)
    animal_ids = ([f'det{a:02d}' for a in range(S)] if boxes_for is not None else
                  [lab.animal_ids[a] if a < len(lab.animal_ids) else f'det{a:02d}'
                   for a in range(S)])

    carried = [None] * S
    window_records = []
    cam_sizes = [session.rig.size(session.cam_names[ci]) for ci in cam_ix]
    starts, blocks, cam_decode, _pipeline_det = _plan_windows(
        session.session_id, gid, cam_sizes, cfg.n_frames, cfg.overlap, T_total, frame_start)
    store = FrameStore(group, session.cam_names)

    f0, w0 = 0, 0
    pred = conf = box_agree = outcome = crop = crop_refined = box_cams = None
    pred2d = conf2d = None
    boxes_stc = None
    _boxes_from = ('detector' if boxes_for is not None else
                   ('given points' if box_points is not None else
                    ('instances.pq' if inst_boxes is not None else 'labels')))

    def _build_plans(wi, start):
        """Everything the loop does before any pixel touches: pure geometry.

        Returns (frames, window_cams, plans). Writes into `outcome`/`crop` pre-allocated and
        indexed by `wi`; different windows touch disjoint slices, so it may run for window wi+1
        on a background thread. Never reads `carried` -- read only inside `forward()`, on the
        main thread, in window order -- so preparing a future window's pixels changes no pixel
        and no order.

        A too-short window is clamped to 2 frames (T=1 hits tracktail's gT = 0 bug), floored at
        `frame_start`; `window_cams` is one camera group per window carrying per-frame
        extrinsics where a camera moves. Geometry first, pixels second: every animal's crop
        boxes are settled before anything decodes, so each (camera, frame) is read once and
        shared by every animal.

        A row needs `min_box_frames` all-finite boxes (one finite box must not fabricate a whole
        window's crop); with boxes, one box per camera is the UNION over the window's frames
        (only the cameras that saw the row), NOT re-squared through `crop_box_for_points`
        (a near-square union only grows); results are int32 and clamped. Stored (instances.pq)
        boxes ARE squared through the rule (`pad=0`). The camera must describe the box the
        pixels were cut with: `apply_crop` sets `size`, `resize_camera` makes it `scales`,
        and `forward` divides by that -- a mismatch scales every keypoint.

        Wide-crop deployment (`crop_inflate != 1.0`) inflates each crop about its centre before
        the resize -- the regime where the box prompt is load-bearing; 1.0 leaves
        `boxes`/`cgroup` untouched. Pass 1 runs at a reduced resolution under `--refine-px`;
        `uncropped` is only read by the refine fallback. The crop box is recorded BEFORE the
        pixels, so a decode failure still shows what it was reaching for.
        """
        frames = np.arange(start, min(start + cfg.n_frames, T_total))
        if len(frames) < 2:
            frames = np.clip(np.arange(start, start + 2), frame_start, T_total - 1)
        fl = frames - f0
        wl = wi - w0
        window_cams = session.cgroup(gid, frames)
        plans = []
        for a in range(S):
            bb = None
            if boxes_stc is not None:
                bb = boxes_stc[a][fl]
                if int(np.isfinite(bb).all(-1).sum()) < cfg.min_box_frames:
                    continue
            elif inst_boxes is not None and a < len(inst_boxes):
                bb = inst_boxes[a][frames]
                if int(np.isfinite(bb).all(-1).sum()) < cfg.min_box_frames:
                    bb = None
            if bb is not None:
                use, boxes = [], []
                for i, ci in enumerate(cam_ix):
                    v = bb[:, i][np.isfinite(bb[:, i]).all(-1)]
                    if not len(v):
                        continue
                    w, h = (int(x) for x in session.rig.size(session.cam_names[ci]))
                    if inst_boxes is not None:
                        corners = torch.as_tensor(
                            np.concatenate([v[:, :2], v[:, 2:]], 0), dtype=torch.float32)
                        box = cropmod.crop_box_for_points(
                            corners, torch.tensor([w, h]), cfg.min_crop_dim, pad=0)
                        if box is None:
                            continue
                    else:
                        x0 = int(np.clip(np.floor(v[:, 0].min()), 0, w - 1))
                        y0 = int(np.clip(np.floor(v[:, 1].min()), 0, h - 1))
                        x1 = int(np.clip(np.ceil(v[:, 2].max()), x0 + 1, w))
                        y1 = int(np.clip(np.ceil(v[:, 3].max()), y0 + 1, h))
                        box = torch.tensor([x0, y0, x1, y1], dtype=torch.int32)
                    boxes.append(box)
                    use.append(ci)
                if not use:
                    outcome[a, wl] = OUTCOMES.index('no camera')
                    continue
                cgroup = [cropmod.apply_crop(window_cams[ci], b) for ci, b in zip(use, boxes)]
            else:
                pts = torch.as_tensor(src[a][frames], dtype=torch.float32)
                if not torch.isfinite(pts).all(-1).any():
                    outcome[a, wl] = OUTCOMES.index('no points')
                    continue
                use = cam_ix
                cgroup, boxes = boxes_from_points(pts, [window_cams[i] for i in cam_ix],
                                                  cfg.min_crop_dim, mode)
                if cgroup is None:
                    outcome[a, wl] = OUTCOMES.index('crop failed')
                    continue
            if cfg.crop_inflate != 1.0:
                boxes = [cropmod.inflate_box(b, window_cams[ci]['size'], cfg.crop_inflate)
                         for ci, b in zip(use, boxes)]
                cgroup = [cropmod.apply_crop(window_cams[ci], b) for ci, b in zip(use, boxes)]
            scales = []
            uncropped = list(cgroup)
            for i, cam in enumerate(cgroup):
                cgroup[i], s = resize_camera(cam, pass1_res)
                scales.append(s)
            for i, ci in enumerate(use):
                crop[a, wl, ci] = np.asarray(boxes[i], np.float32)
            outcome[a, wl] = OUTCOMES.index('decode failed')
            plans.append(WindowPlan(a, use, boxes, cgroup, scales, uncropped))
        return frames, window_cams, plans

    def decode_crops(frames, plans):
        """`frames` is a parameter so this can run for a future window on a background thread
        while the current window's forward runs on the main thread.

        One decode per (camera, frame) per window, shared by every animal in it. The cameras
        decode concurrently, up to `_CAM_DECODE` of them (the cap is MEMORY, not cores: each
        task holds one camera's whole window of FULL frames). `crops` is written from those
        threads -- keys are distinct per (animal, camera) and a dict store is atomic under the
        GIL. `one` runs in this pool and waits on futures in `pool`, hence a SECOND POOL: a pool
        that waits on itself deadlocks as soon as both are full.
        """
        crops = {}
        cams = sorted({c for _, use, *_ in plans for c in use})

        def one(ci, pool):
            """Decode one camera and crop it for every animal whose plan wants it.

            `store` is keyed by (camera, SOURCE frame) and holds the FULL decoded frame, which
            is what lets one decode serve every consumer: refine wants the same pixels under a
            different crop, and `crop_views` never mutates what it is given. A file that will
            not decode takes out every animal that wanted this camera.
            """
            imgs = store.read(ci, session.cam_names[ci], frames, pool=pool)
            ok = not any(im is None for im in imgs)
            for a, use, boxes, cgroup, *_ in plans:
                if ci in use:
                    i = use.index(ci)
                    bx = boxes[i]
                    crops[a, ci] = (crop_views(imgs, bx, cgroup[i]['size'].tolist())
                                    if ok else None)
            del imgs

        with ThreadPoolExecutor(max_workers=8) as pool:
            if len(cams) < 2:
                one(cams[0], pool) if cams else None
            else:
                with ThreadPoolExecutor(max_workers=min(cam_decode, len(cams))) as cpool:
                    list(cpool.map(lambda ci: one(ci, pool), cams))
        return crops

    def forward(frames, plan, crops, wi, *, score_quality=False, trace=True):
        """One animal, one window -> its prediction in the SOURCE frame, or None.

        `trace` gates the overlap-trace probes (`prior_oob`, `model_conf`); `--refine`'s pass 1
        passes False so each (animal, window) is recorded once, from the pass it reports.

        `frames` is a parameter for the same reason as `decode_crops`. `carried` is read here on
        the main thread only, in window order -- the window-order guarantee lives in the caller.

        The views are uint8; the model divides on device -- the same contract as the training
        loader. One encode serves both passes of `self`: the pixels are identical (a no-op under
        `kpt_chunk`). The deployment box prompt (2D and 3D, per camera) is guarded:
        `box_prompt == 'none'` passes no kwarg, so the stock model sees the identical call; when
        on, a camera with nothing finite gets a NaN column, which the encoder's missing-box
        token substitutes. 'labels' is an ORACLE (the animal's GT keypoints name which animal to
        return); 'detector' is deployable (the detector's own box for this animal slot, per
        camera).

        `q` is what the next window opens on, and it is not always what this window reports:
        under `gridresid_offset = "query"` the reported output is `query + R @ residual`, so
        feeding it back closes a loop with gain. `3d_pred_triangulate` is the anchor-free
        estimate, re-derived from this window's pixels every frame; carrying it leaves the
        reported output untouched and breaks the feedback path only. In 2D, `q` is deliberately
        the same tensor as `p`: at one camera there is nothing to triangulate, so every 2D root
        is bit-identical under either `carry_source`. In 3D single-view no key is written, so
        `carried[a]` is left alone and the staleness bound in `build_prior` retires it. The
        2D->source conversion undoes the resize, then adds the crop origin; the per-camera 2D
        pose takes the same expression per camera, so camera 0's per-camera pose is bit-identical
        to the reported prediction (a free exact check).
        """
        a, use, boxes, cgroup, scales, *_ = plan
        views = [crops[a, ci] for ci in use]
        if any(v is None for v in views):
            return None
        prior, prompt_t = build_prior(cfg, carried[a], src, a, n_lab, frames, boxes,
                                       scales, mode, K, R, cgroup)
        if (trace and stats is not None and stats.get('capture_overlap_agreement')
                and prior is not None):
            count_prior_oob(stats, a, int(frames[0]), prior)
        dev = cfg.device
        chunk = cfg.kpt_chunk or None
        views = [v.to(dev) for v in views]
        cgroup_d = to_device(cgroup, dev)
        scene_encoding = scene_features = views_norm = None
        if scorer is not None:
            input_size = max(int(cam['size'].max()) for cam in cgroup_d)
            views_norm = model.normalize_scene_views(views, input_size=input_size)
            scene_encoding = model.encode_scene_backbone(views_norm)
            scene_features = model.project_scene(scene_encoding)
        mkw = {}
        give_box = give_box_this_window(cfg, carried[a], prior)
        if give_box:
            if cfg.box_prompt == 'labels':
                if src is None or a >= n_lab:
                    box_t = None
                else:
                    box_t = deploy_box_prompt(mode, src, None, frames, a, use, boxes, scales,
                                              cgroup, dev)
            elif cfg.box_prompt == 'detector':
                if boxes_stc is None and box_points is not None:
                    box_t = (deploy_box_prompt(mode, src, None, frames, a, use, boxes, scales,
                                                cgroup, dev) if a < n_lab else None)
                elif boxes_stc is None:
                    raise ValueError('box_prompt = "detector" needs detector boxes '
                                     '(--detector or --boxes); none were supplied.')
                else:
                    box_t = deploy_box_prompt(mode, None, boxes_stc, frames - f0,
                                              a, use, boxes, scales, cgroup, dev)
            if box_t is not None:
                mkw['box_prompt'] = box_t
                box_cams[a, wi - w0] = int(torch.isfinite(box_t).all(-1).any(1)[0].sum())
        scene_ctx = (share_scene(model, scene_features=scene_features)
                     if cfg.anchor == 'self' else nullcontext())
        with scene_ctx:
            out = model(views, kpt_ids.to(dev), cgroup_d, mode=mode,
                        kpt_prior=None if prior is None else prior.to(dev),
                        prompt_time=None if prompt_t is None else prompt_t.to(dev),
                        kpt_chunk=chunk, **({} if scene_features is None else
                                            {'scene_features': scene_features}), **mkw)
            if cfg.anchor == 'self':
                out = self_prompt(model, views, kpt_ids.to(dev), cgroup_d, mode, out,
                                  kpt_chunk=chunk, box_prompt=mkw.get('box_prompt'))
        if trace and stats is not None and stats.get('capture_overlap_agreement'):
            capture_model_confidence(stats, a, frames, out)
        p = out['coords_pred'][0].detach().cpu().numpy()
        q = None
        if mode == '2d':
            p = p / np.asarray(scales[0], dtype=np.float32) + np.asarray(
                boxes[0][:2], np.float32)
            q = p
        elif cfg.carry_source == 'pred':
            q = p
        elif out.get('3d_pred_triangulate') is not None:
            q = out['3d_pred_triangulate'][0].detach().cpu().numpy()
        p2 = v2 = c2 = None
        if '2d_pred' in out:
            p2 = out['2d_pred'][:, 0].detach().cpu().numpy().copy()
            for i in range(len(use)):
                p2[i] = p2[i] / np.asarray(scales[i], dtype=np.float32) + np.asarray(
                    boxes[i][:2], np.float32)
            if out.get('vis_pred_2d') is not None:
                v2 = out['vis_pred_2d'][:, 0].detach().cpu().numpy()
            if out.get('conf_pred_2d') is not None:
                c2 = out['conf_pred_2d'][:, 0].detach().cpu().numpy()
        quality = quality_precision = None
        if scorer is not None and score_quality:
            if shared_scene_encoder:
                score_scene = scorer.project_scene(scene_encoding)
                scores, precision = scorer.score(
                    views_norm, score_scene, out['coords_pred'], cgroup_d, kpt_ids.to(dev),
                    kpt_chunk=chunk)
            else:
                scores, precision = scorer(
                    views, out['coords_pred'], cgroup_d, kpt_ids.to(dev), kpt_chunk=chunk)
            quality = scores[0].detach().cpu().numpy()
            quality_precision = precision[0].detach().cpu().numpy()
        return p, q, out, p2, v2, c2, quality, quality_precision

    def _process_window(wi, frames, window_cams, plans, crops):
        """Forward and write every column for one window, given already-decoded `crops`.

        `crops` is a parameter -- `_prepare` computes it on a background thread, and this must
        consume that result rather than decoding again. Runs on the main thread, in window
        order: `carried` is read and written only here.

        Crop refinement is label-free: the first pass's own prediction re-enters the crop rule
        as if it were the labels. It costs one extra forward and one extra decode per animal per
        window -- the crop moved, so neither pixels nor scene encode are shared. A failed
        refinement keeps the first-pass BOX, but NOT its camera under `--refine-px` (that one is
        at the reduced resolution; `_at_image_size` rebuilds it). A refined box that does not
        overlap the box it came from is not a refinement: the crop rule SQUARES the extent, so a
        wandered pose produces a box somewhere else entirely, and overlap with the first-pass box
        is the weakest test that catches "somewhere else". `crop` keeps the first-pass box: it
        is the record of what the detector offered.

        `j` is the pose the NEXT window opens on, clamped from the front: a group shorter than
        `overlap` gives a window with fewer frames than the step, and a plain negative index
        would run off the start of it. The `vis_thresh` row gate applies to what is REPORTED and
        never to what is carried: `carried` reads `p`, which is untouched by the gate -- a gate
        that blinded the next window's prompt would be measuring the prompt. The gate uses the
        MEDIAN over keypoints, since a mean lets one confident keypoint carry a row the model
        otherwise declined; an all-NaN row is legal (its RuntimeWarning is suppressed) but is not
        a passing row, because `NaN < thresh` is False. It is applied to `p` before recording (a
        gated frame must be left out of the mean, not blanked once and averaged back in); `q` is
        untouched, so the carried prompt is unaffected, and `conf`/`box_agree` follow the same
        rule as `p`. In 2D `p` IS the per-camera pose, so `pred2d`/`conf2d` are masked the same
        way or `points2d.pq` would disagree with what `pred` reports; 3D leaves its per-camera
        overlay untouched, a diagnostic rather than the primary output.
        """
        if cfg.refine:
            def _at_image_size(plan):
                """The same plan with pass 1 rebuilt at full `image_size` (a no-op when it
                already is at full resolution)."""
                a, use, boxes, _, _, uncropped = plan
                if pass1_res == cfg.image_size:
                    return plan
                cg, sc = [], []
                for cam in uncropped:
                    c, s = resize_camera(cam, cfg.image_size)
                    cg.append(c)
                    sc.append(s)
                return WindowPlan(a, use, boxes, cg, sc, uncropped)

            refined = []
            for plan in plans:
                a, use, boxes, cgroup, scales, *_ = plan
                got = forward(frames, plan, crops, wi, trace=False)
                if got is None:
                    refined.append(_at_image_size(plan))
                    continue
                pts = torch.as_tensor(got[0], dtype=torch.float32)
                cg2, b2 = boxes_from_points(pts, [window_cams[i] for i in use],
                                            cfg.min_crop_dim, mode)
                if cg2 is None:
                    refined.append(_at_image_size(plan))
                    continue
                if any(not overlaps(b2[i], boxes[i]) for i in range(len(use))):
                    refined.append(_at_image_size(plan))
                    continue
                uncropped2, sc2 = list(cg2), []
                for i, cam in enumerate(cg2):
                    cg2[i], s = resize_camera(cam, cfg.image_size)
                    sc2.append(s)
                for i, ci in enumerate(use):
                    crop_refined[a, wi - w0, ci] = np.asarray(b2[i], np.float32)
                refined.append(WindowPlan(a, use, b2, cg2, sc2, uncropped2))
            plans = refined
            crops = decode_crops(frames, plans)

        for plan in plans:
            a, use, boxes, cgroup, scales, *_ = plan
            got = forward(frames, plan, crops, wi,
                          score_quality=scorer is not None)
            if got is None:
                continue
            p, q, out, p2, v2, c2, quality_window, precision_window = got
            outcome[a, wi - w0] = OUTCOMES.index('ok')
            if quality_window is not None:
                quality[a, frames - f0] = quality_window
                quality_precision[a, frames - f0] = precision_window
            fill_box_agreement(box_agree, a, frames - f0, use, boxes, p, mode, window_cams)
            if p2 is not None:
                for i, ci in enumerate(use):
                    pred2d[a, frames - f0, ci] = p2[i]
                    if v2 is not None:
                        conf2d[a, frames - f0, ci] = v2[i]
                    if c2 is not None:
                        model_conf2d[a, frames - f0, ci] = c2[i]
            vlogit = None
            tri = out.get('3d_pred_triangulate')
            if tri is not None:
                triangulated[a, frames - f0] = tri[0].detach().cpu().numpy()
            if 'vis_pred' in out:
                v = out['vis_pred'][0].detach().cpu().numpy().reshape(len(frames), K)
                conf[a, frames - f0] = v
                vlogit = v
            drop = np.zeros(len(frames), dtype=bool)
            cam_drop = None
            gate = cfg.vis_thresh is not None and (v2 is not None or vlogit is not None)
            if gate:
                drop, cam_drop = vis_gate(v2, vlogit, cfg.vis_thresh)
            if cfg.capture_window_predictions:
                start_w = int(frames[0])
                stop_w = min(start_w + cfg.n_frames, T_total)
                animal_id = str(animal_ids[a])
                tri_np = (None if tri is None else tri[0].detach().cpu().numpy())
                for ti, frame in enumerate(frames):
                    gated = bool(drop[ti])
                    if mode == '3d':
                        for k in range(K):
                            if not np.isfinite(p[ti, k]).all():
                                continue
                            rec_k = window_record_base(gid, wi, start_w, stop_w, a, animal_id,
                                                        int(frame), str(session.names[k]), 'point3d')
                            logit = float(vlogit[ti, k]) if vlogit is not None else None
                            rec_k.update({'x': float(p[ti, k, 0]),
                                          'y': float(p[ti, k, 1]),
                                          'z': float(p[ti, k, 2]),
                                          'score_logit': logit,
                                          'score': (logit_probability(logit)
                                                    if logit is not None else None),
                                          'status': ('visible' if logit is not None and
                                                     logit > 0 else 'missing'),
                                          'gated': gated})
                            if quality_window is not None and mode == '3d':
                                score_value = float(quality_window[ti, k])
                                precision_value = float(precision_window[ti, k])
                                rec_k['quality_score'] = (score_value if np.isfinite(score_value)
                                                          else None)
                                rec_k['quality_precision'] = (
                                    precision_value if np.isfinite(precision_value) else None)
                            if tri_np is not None and np.isfinite(tri_np[ti, k]).all():
                                rec_k.update({'triangulated_x': float(tri_np[ti, k, 0]),
                                              'triangulated_y': float(tri_np[ti, k, 1]),
                                              'triangulated_z': float(tri_np[ti, k, 2])})
                            window_records.append(rec_k)
                    if p2 is not None:
                        for i, ci in enumerate(use):
                            for k in range(K):
                                if not np.isfinite(p2[i, ti, k]).all():
                                    continue
                                vis_logit = (float(v2[i, ti, k]) if v2 is not None else None)
                                conf_logit = (float(c2[i, ti, k]) if c2 is not None else None)
                                rec2 = window_record_base(
                                    gid, wi, start_w, stop_w, a, animal_id, int(frame),
                                    str(session.names[k]), 'point2d')
                                rec2.update({'camera': str(session.cam_names[ci]),
                                             'x': float(p2[i, ti, k, 0]),
                                             'y': float(p2[i, ti, k, 1]),
                                             'visibility_logit': vis_logit,
                                             'visibility_probability': (logit_probability(vis_logit)
                                                                        if vis_logit is not None else None),
                                             'confidence_logit': conf_logit,
                                             'confidence_probability': (logit_probability(conf_logit)
                                                                        if conf_logit is not None else None),
                                             'gated': bool(cam_drop[i, ti]) if cam_drop is not None
                                             else (gated if mode == '2d' else False)})
                                if quality_window is not None and mode == '2d':
                                    score_value = float(quality_window[ti, k])
                                    precision_value = float(precision_window[ti, k])
                                    rec2['quality_score'] = (score_value
                                                             if np.isfinite(score_value) else None)
                                    rec2['quality_precision'] = (
                                        precision_value if np.isfinite(precision_value) else None)
                                window_records.append(rec2)
                    for ci, cam in enumerate(session.cam_names):
                        box = det_box[a, int(frame) - f0, ci]
                        agree = box_agree[a, int(frame) - f0, ci]
                        if np.isfinite(box).all() or np.isfinite(agree):
                            reci = window_record_base(
                                gid, wi, start_w, stop_w, a, animal_id, int(frame), None,
                                'instance', str(cam))
                            reci.update({'x0': float(box[0]) if np.isfinite(box[0]) else None,
                                         'y0': float(box[1]) if np.isfinite(box[1]) else None,
                                         'x1': float(box[2]) if np.isfinite(box[2]) else None,
                                         'y1': float(box[3]) if np.isfinite(box[3]) else None,
                                         'score': (float(det_score[a, int(frame) - f0, ci])
                                                   if np.isfinite(det_score[a, int(frame) - f0, ci])
                                                   else None),
                                         'box_agree': float(agree) if np.isfinite(agree) else None})
                            window_records.append(reci)
            j = max(0, len(frames) - cfg.overlap) if cfg.overlap else len(frames) - 1
            if gate:
                if quality_window is not None:
                    quality[a, frames[drop] - f0] = np.nan
                    quality_precision[a, frames[drop] - f0] = np.nan
                p = p.copy()
                p[drop] = np.nan
                conf[a, frames[drop] - f0] = np.nan
                box_agree[a, frames[drop] - f0] = np.nan
                if p2 is not None and (cam_drop is not None or mode == '2d'):
                    for i, ci in enumerate(use):
                        off = frames[cam_drop[i] if cam_drop is not None else drop] - f0
                        pred2d[a, off, ci] = np.nan
                        box_agree[a, off, ci] = np.nan
                        if v2 is not None:
                            conf2d[a, off, ci] = np.nan
                        if c2 is not None:
                            model_conf2d[a, off, ci] = np.nan
            if stats is not None and stats.get('capture_overlap_agreement') and cfg.overlap:
                capture_overlap_agreement(stats, a, frames, f0, pred, p)
            pred[a, frames - f0] = p
            if q is not None:
                carried[a] = (torch.as_tensor(q[j]), int(frames[j]))

        if cfg.capture_window_predictions:
            start_w = int(frames[0])
            stop_w = min(start_w + cfg.n_frames, T_total)
            for animal in range(S):
                animal_id = str(animal_ids[animal])
                outcome_name = OUTCOMES[int(outcome[animal, wi - w0])]
                for ci, cam in enumerate(session.cam_names):
                    recw = window_record_base(gid, wi, start_w, stop_w, animal,
                                               animal_id, None, None, 'window', str(cam))
                    recw['outcome'] = outcome_name
                    crop_box = crop[animal, wi - w0, ci]
                    refined_box = (None if crop_refined is None else
                                   crop_refined[animal, wi - w0, ci])
                    for prefix, values in (('crop_', crop_box), ('crop_refined_', refined_box)):
                        for axis, value in zip(('x0', 'y0', 'x1', 'y1'),
                                               (values if values is not None else
                                                [np.nan] * 4)):
                            recw[prefix + axis] = float(value) if np.isfinite(value) else None
                    recw['box_prompt_cams'] = (None if box_cams is None else
                                               int(box_cams[animal, wi - w0]))
                    window_records.append(recw)

    n_ahead = max(0, int(cfg.prefetch_windows))
    _prefetch_pool = ThreadPoolExecutor(max_workers=1) if n_ahead else None

    def _prepare(wi, start):
        """Build the plans and decode the pixels for window `wi` at source frame `start`."""
        frames, window_cams, plans = _build_plans(wi, start)
        crops = decode_crops(frames, plans)
        return frames, window_cams, plans, crops

    _det_pool = (ThreadPoolExecutor(max_workers=1)
                 if (boxes_for is not None and _pipeline_det) else None)

    def _detect(bi):
        """Boxes for block `bi`, or None past the end. Runs on `_det_pool`, in block order."""
        if bi >= len(blocks):
            return None
        a, b = blocks[bi]
        return boxes_for(store, int(starts[a]),
                         int(min(T_total, starts[b - 1] + cfg.n_frames)))

    try:
        _pending_det = _det_pool.submit(_detect, 0) if _det_pool is not None else None
        for bi, (w0, w1) in enumerate(blocks):
            f0 = int(starts[w0])
            f_read = int(min(T_total, starts[w1 - 1] + cfg.n_frames))
            f_own = int(starts[w1]) if w1 < len(starts) else T_total
            n_blk, n_win = f_read - f0, w1 - w0

            boxes_stc = None
            if _det_pool is not None:
                boxes_stc, _scores = _pending_det.result()
                _pending_det = _det_pool.submit(_detect, bi + 1)
            elif boxes_for is not None:
                boxes_stc, _scores = _detect(bi)

            det_box = np.full((S, n_blk, len(session.rig), 4), np.nan, np.float32)
            det_score = np.full((S, n_blk, len(session.rig)), np.nan, np.float32)
            if boxes_stc is not None:
                det_box[...] = boxes_stc
                det_score[...] = _scores

            owner_window = np.full(n_blk, -1, np.int32)
            triangulated = np.full((S, n_blk, K, 3), np.nan, np.float32)
            pred = np.full((S, n_blk, K, R), np.nan, np.float32)
            quality = (np.full((S, n_blk, K), np.nan, np.float32) if scorer is not None else None)
            quality_precision = (np.full((S, n_blk, K), np.nan, np.float32)
                                 if scorer is not None else None)
            conf = np.full((S, n_blk, K), np.nan, np.float32)
            pred2d = np.full((S, n_blk, len(session.rig), K, 2), np.nan, np.float32)
            conf2d = np.full((S, n_blk, len(session.rig), K), np.nan, np.float32)
            model_conf2d = np.full((S, n_blk, len(session.rig), K), np.nan, np.float32)
            box_agree = np.full((S, n_blk, len(session.rig)), np.nan, np.float32)
            outcome = np.full((S, n_win), OUTCOMES.index('no box'), np.int8)
            crop = np.full((S, n_win, len(session.rig), 4), np.nan, np.float32)
            crop_refined = (np.full_like(crop, np.nan) if cfg.refine else None)
            box_cams = (np.full((S, n_win), -1, np.int8) if cfg.box_prompt != 'none' else None)

            window_records = []
            pending = {}
            if _prefetch_pool is not None:
                for j in range(min(n_ahead, n_win - 1)):
                    pending[w0 + j + 1] = _prefetch_pool.submit(
                        _prepare, w0 + j + 1, starts[w0 + j + 1])
            for wi in range(w0, w1):
                if wi in pending:
                    frames, window_cams, plans, crops = pending.pop(wi).result()
                else:
                    frames, window_cams, plans, crops = _prepare(wi, starts[wi])
                nxt = wi + n_ahead
                if _prefetch_pool is not None and nxt < w1 and nxt not in pending:
                    pending[nxt] = _prefetch_pool.submit(_prepare, nxt, starts[nxt])
                _process_window(wi, frames, window_cams, plans, crops)
                owner_window[frames - f0] = wi
                if wi + 1 < w1:
                    store.evict_below(int(starts[max(w0, wi - n_ahead + 1)]))
                memory.trim()
            if stats is not None:
                for oi, name in enumerate(OUTCOMES):
                    key = f'window_{name.replace(" ", "_")}'
                    stats[key] = stats.get(key, 0) + int((outcome == oi).sum())
            store.evict_below(f_own)
            memory.trim()
            if stats is not None:
                stats['decode_s'] = store.decode_s
                stats['decode_hits'], stats['decode_misses'] = store.hits, store.misses
            memory.check_peak(f'the window loop ({session.session_id}/{gid})')
            keep = f_own - f0
            if stats is not None and stats.get('capture_overlap_agreement') and cfg.overlap:
                census_block_boundary(stats, f_read, f_own)
            yield {'pred': pred[:, :keep], 'conf': conf[:, :keep],
                   **({} if quality is None else {'quality': quality[:, :keep],
                                                  'quality_precision': quality_precision[:, :keep]}),
                   'pred2d': pred2d[:, :keep], 'conf2d': conf2d[:, :keep],
                   'model_conf2d': model_conf2d[:, :keep],
                   'box_agree': box_agree[:, :keep],
                   'det_box': det_box[:, :keep], 'det_score': det_score[:, :keep],
                   'owner_window': owner_window[:keep],
                   'triangulated': triangulated[:, :keep],
                   'slot': np.arange(S, dtype=np.int32),
                   'animal_ids': np.asarray(animal_ids, object),
                   'outcome': outcome, 'crop': crop,
                   'window_start': np.asarray(starts[w0:w1]),
                   'window_stop': np.asarray([min(int(starts[j]) + cfg.n_frames, T_total)
                                              for j in range(w0, w1)], dtype=np.int32),
                   'window_ordinal': np.arange(w0, w1, dtype=np.int32),
                   'window_records': window_records,
                   'outcome_names': np.asarray(OUTCOMES, object),
                   'mode': mode, 'group_id': gid, 'session': session.session_id,
                   'dataset': dataset_name, 'anchor': cfg.anchor,
                   'carry_source': cfg.carry_source, 'n_frames': T_total - frame_start,
                   'frame_start': frame_start, 'frame_stop': T_total,
                   'refine': bool(cfg.refine), 'refine_px': cfg.refine_px or 0,
                   **({} if crop_refined is None else {'crop_refined': crop_refined}),
                   **({} if box_cams is None else {'box_prompt_cams': box_cams}),
                   'boxes_from': _boxes_from}
    finally:
        if _det_pool is not None:
            _det_pool.shutdown(wait=True)
        if _prefetch_pool is not None:
            _prefetch_pool.shutdown(wait=True)
        store.clear()


