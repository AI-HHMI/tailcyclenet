"""The inference path. One window loop.

Per group: for each window of T frames stepping by T - overlap, get a crop box per camera, read
the pixels, build the prior (none / carry / self / labels), forward, and map the prediction back
into the source coordinate frame. `carry` seeds from the model's own previous prediction and
needs `overlap >= 1`; `labels` is a GT oracle, not a deployment number.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace

import numpy as np
import torch

from posetail.posetail.cube import project_points_torch

from .. import crop as cropmod
from .. import memory
from ..dataset import _crop_affine, _resize_camera, prior_out_of_bounds
from ..format import Labels, Session
from ..model import share_scene
from .store import FrameStore

ANCHORS = ('none', 'carry', 'self', 'labels')
CARRY_SOURCES = ('triangulate', 'pred')

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
    # Where the window's crop comes from: 'boxes' unions the detector's per-frame boxes;
    # 'keypoints' runs the crop rule on the detector's own keypoints (needs a keypoint-trained
    # detector; ignored without one).
    crop_source: str = 'boxes'
    # How many finite (frame, camera) boxes a row needs before it gets a window crop; raising it
    # deliberately lowers reported coverage (one box used to fabricate a whole window's).
    min_box_frames: int = 1
    # What `carry` feeds back: 'triangulate' is the anchor-free estimate (breaks the feedback
    # loop); 'pred' is the reported prediction, which under gridresid_offset = "query" is
    # `prior + residual` and integrates its own error. 2D is identical either way.
    carry_source: str = 'triangulate'
    # Deliberately break the oracle prior to measure the echo coefficient; see `_corrupt_prior`.
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
    # Inflate every crop about its centre (the wide pass-1 regime); 1.0 is the unchanged behaviour.
    crop_inflate: float = 1.0
    # How many windows ahead to decode while the current one forwards. Bit-exact at any value:
    # `_build_plans`/`decode_crops` never read `carried`, which is touched only in window order
    # on the main thread. 0 is the exact old serial path.
    prefetch_windows: int = 1


def _window_starts(n_frames: int, T: int, overlap: int, start: int = 0):
    """Contiguous windows covering [start, start + n_frames), stepping by T - overlap.

    The last window is pulled back to end at the last frame rather than padded, so no frame is
    predicted from duplicated pixels. `start` and the returned starts are SOURCE indices; at the
    default 0 the values are what they always were, so an unset range is byte-identical.
    """
    step = max(1, T - overlap)
    if n_frames <= T:
        return [start]
    starts = list(range(start, start + n_frames - T + 1, step))
    if starts[-1] + T < start + n_frames:
        starts.append(start + n_frames - T)
    return starts


def boxes_from_points(points, cgroup, min_crop_dim, mode):
    """Crop boxes for one animal in one window, from points -- the crop rule, shared with training.

    3D points are world coordinates and get projected; 2D are already pixels. None when nothing
    is finite. `cgroup` is built by the caller once per window (per-animal builds dropped the
    per-frame extrinsics).
    """
    if mode == '3d':
        cg, boxes = cropmod.crop_to_points_3d(cgroup, points, min_crop_dim)
        return (cg, boxes) if cg is not None else (None, None)
    cam, box, _ = cropmod.crop_to_points_2d(cgroup[0], points, min_crop_dim)
    return ([cam], [box]) if cam is not None else (None, None)


def _deploy_box_prompt(mode, src_pts, boxes_stc, frames, a, use, boxes, scales, cgroup, dev):
    """The box-prompt tensor for one animal's window, per camera, 2D or 3D: (1,T,C,4) in crop
    pixels, C = len(use) -- column order matches cgroup/views/use.

    `src_pts` names which animal: the GT points under 'labels' (an oracle), or None under
    'detector', where boxes_stc[a] supplies per-camera boxes. 3D + labels reuses the exact
    training-time `compute_box_prompt` on the window's cropped cgroup -- identical to what
    `_item` computes at training time. 3D + detector boxes each camera's own box corners in its
    crop frame: a 2D computation per camera, since the detector box is a per-camera detection
    to begin with. 2D is the single-camera case. A camera absent from `use` is absent from the
    output too (never a NaN at the wrong index).
    """
    from .. import box_prompt as bpmod

    if mode == '3d' and src_pts is not None:
        pts = torch.as_tensor(src_pts[a][frames], dtype=torch.float32)
        return bpmod.compute_box_prompt(pts, cgroup, '3d')[None].to(dev)

    if mode == '3d':
        T = len(frames)
        C = len(use)
        out = torch.full((T, C, 4), float('nan'), dtype=torch.float32)
        for i, ci in enumerate(use):
            db = torch.as_tensor(boxes_stc[a][frames][:, ci], dtype=torch.float32)
            if not torch.isfinite(db).any():
                continue
            corners = cropmod.box_corners(db)
            origin = torch.as_tensor(boxes[i][:2], dtype=torch.float32)
            cf = (corners - origin) * float(scales[i])
            size = cgroup[i]['size']
            for t in range(T):
                bx = cropmod.crop_box_for_points(cf[t], size,
                                                 bpmod.BOX_PROMPT_MIN_DIM, bpmod.BOX_PROMPT_PAD)
                if bx is not None:
                    out[t, i] = bx.to(torch.float32)
        return out[None].to(dev)

    source = (torch.as_tensor(src_pts[a][frames], dtype=torch.float32) if src_pts is not None
              else cropmod.box_corners(torch.as_tensor(boxes_stc[a][frames][:, use[0]],
                                                       dtype=torch.float32)))
    origin = torch.as_tensor(boxes[0][:2], dtype=torch.float32)
    cf = (source - origin) * float(scales[0])
    size = cgroup[0]['size']
    T = cf.shape[0]
    out = torch.full((T, 1, 4), float('nan'), dtype=torch.float32)
    for t in range(T):
        bx = cropmod.crop_box_for_points(cf[t], size, bpmod.BOX_PROMPT_MIN_DIM,
                                         bpmod.BOX_PROMPT_PAD)
        if bx is not None:
            out[t, 0] = bx.to(torch.float32)
    return out[None].to(dev)


def _to_device(cgroup, device):
    """Move each camera's tensors to `device`; non-tensors (e.g. `size`) are left as-is."""
    return [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
            for c in cgroup]


def _crop_views(imgs, box, target_size):
    """Crop+resize one camera's already-decoded window -> (1,T,H,W,3) uint8.

    Decode is per (camera, frame) and crop is per (animal, camera, frame), so cropping inside the
    animal loop paid the full-frame decode once per animal. `box` is one [x1,y1,x2,y2] for the
    whole window or a (T,4) of per-frame boxes (kept for the loader's per-frame crops).
    """
    import cv2

    b = np.asarray(box)
    src_wh = (imgs[0].shape[1], imgs[0].shape[0])
    if b.ndim == 1:
        aff = _crop_affine(src_wh, box, target_size, None)
        out = [im if aff is None else cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR)
               for im in imgs]
    else:
        out = []
        for t, im in enumerate(imgs):
            aff = _crop_affine(src_wh, b[min(t, len(b) - 1)], target_size, None)
            out.append(im if aff is None else
                       cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR))
    return torch.from_numpy(np.asarray(out))[None]


def self_prompt(model, views, kpt_ids, cgroup, mode, first, kpt_chunk=None, box_prompt=None):
    """Re-query at the model's own frame-0 prediction -- the label-free prompted regime.

    `first` is a completed prior-free pass; its frame-0 pose becomes the prior for a second pass,
    already in the model's own frame (the pose is `(1,K,R)`). It gets the same bounds mask
    `carry` gets: a keypoint outside its own crop is NaN, which is what the no-query tokens key
    off. The box prompt carries into the second pass unchanged: the window has not moved. Shared
    with the trainer so training and inference report the same number.
    """
    p = first['coords_pred'][0].detach()
    prior = p[0][None].clone()
    prior[0, prior_out_of_bounds(prior[0], mode, cgroup)] = float('nan')
    qt = torch.zeros(prior.shape[:2], dtype=torch.int32, device=prior.device)
    mkw = {} if box_prompt is None else {'box_prompt': box_prompt}
    return model(views, kpt_ids, cgroup, mode=mode, kpt_prior=prior, prompt_time=qt,
                 kpt_chunk=kpt_chunk, **mkw)


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


# Frame/window-indexed columns stitched by `merge_blocks`; anything else is a per-group constant.
_FRAME_KEYS = ('pred', 'conf', 'pred2d', 'conf2d', 'box_agree', 'det_box', 'det_score')
_WINDOW_KEYS = ('outcome', 'crop', 'crop_refined', 'box_prompt_cams', 'window_start')


def merge_blocks(blocks):
    """Every block of `run_blocks` stitched into the one dict `run_group` used to return.

    Frame- and window-indexed columns concatenate (axis 0 for the 1-D `window_start`); everything
    else is a per-group constant from the first block.
    """
    blocks = list(blocks)
    out = dict(blocks[0])
    for k in _FRAME_KEYS + _WINDOW_KEYS:
        if k not in out:
            continue
        axis = 0 if np.asarray(out[k]).ndim == 1 else 1
        out[k] = np.concatenate([b[k] for b in blocks], axis=axis)
    return out


@torch.no_grad()
def run_group(model, session: Session, gid: str, registry, dataset_name: str,
              cfg: InferConfig, box_points=None, boxes_for=None, n_rows=None) -> dict:
    """`run_blocks` for a whole group, merged. See both for what a block is."""
    return merge_blocks(run_blocks(model, session, gid, registry, dataset_name, cfg,
                                   box_points=box_points, boxes_for=boxes_for, n_rows=n_rows))


@torch.no_grad()
def run_blocks(model, session: Session, gid: str, registry, dataset_name: str,
               cfg: InferConfig, box_points=None, boxes_for=None, n_rows=None, stats=None):
    """Predict every animal in one group, a block of windows at a time. Yields one dict per block.

    Arrays are in the SOURCE coordinate frame. Crops come from exactly one of two sources, not
    comparable: `box_points` (S,T,K,R) points the crop rule follows (the labels themselves are
    the GT-crop upper bound), or `boxes_for(store, lo, hi) -> (boxes, scores, kpts)` from a
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
    if cfg.anchor in ('carry', 'self') and cfg.overlap < 1:
        raise ValueError(f'anchor={cfg.anchor!r} carries a pose across windows and needs '
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

    src = box_points if box_points is not None else (
        lab.points3d if mode == '3d' else lab.points2d[..., 0, :])
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
    starts = _window_starts(T_total - frame_start, cfg.n_frames, cfg.overlap, start=frame_start)

    _frame_bytes = max(int(w) * int(h) for w, h in
                       (session.rig.size(session.cam_names[ci]) for ci in cam_ix)) * 3
    _frame_cost = len(cam_ix) * _frame_bytes
    _budget = memory.current()
    _one = cfg.n_frames * _frame_cost
    _want_store = (float('inf') if _budget.stated else max(2 * _one, _MIN_STORE_BYTES))
    _share = _budget.share(memory.FRACTION_STORE)
    _pipeline_det = _share >= 2 * _one
    _store_bytes = min(_share / (2 if _pipeline_det else 1), _want_store)
    cam_decode = memory.fits(_store_bytes / 2, _frame_bytes * cfg.n_frames,
                             want=min(_CAM_DECODE, len(cam_ix)))
    if _one > _store_bytes:
        raise SystemExit(
            f'{session.session_id}/{gid}: one window of frames does not fit, and there is no '
            're-decode fallback.\n'
            f'    {cfg.n_frames} frames x {len(cam_ix)} camera(s) x {_frame_bytes / 1e6:.1f} MB '
            f'= {_one / (1 << 30):.2f} GB\n'
            f'    frame store {_store_bytes / (1 << 30):.2f} GB  '
            f'({_budget.budget_gb:.1f} GB budget x {memory.FRACTION_STORE:g}, {_budget.source})\n'
            f'  --max-ram {_one / (memory.FRACTION_STORE * memory.DEFAULT_FRACTION) / (1 << 30):.0f}'
            '   holds one window. The flag is a ceiling on the PROCESS, not an allowance for the '
            'buffers, so the budget is a fraction of it.\n'
            f'  --n-frames N     a shorter window needs proportionally less -- but it is NOT '
            f'output-neutral: this run was trained at {cfg.n_frames} and a shorter window is a '
            'different prediction.\n'
            'Refine pass 2 crops from the SAME frames pass 1 did, so a window\'s frames cannot be '
            'dropped and re-read: re-decoding would double the wall clock, and re-cropping from '
            'the stored crop double-resamples and is not output-neutral.')
    blocks = _plan_blocks(starts, cfg.n_frames, T_total, _frame_cost, _store_bytes)
    store = FrameStore(group, session.cam_names)

    f0, w0 = 0, 0
    pred = conf = box_agree = outcome = crop = crop_refined = box_cams = None
    pred2d = conf2d = None
    boxes_stc = det_kpts_stc = None
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

        A too-short window is clamped to 2 frames (T=1 hits posetail's gT = 0 bug), floored at
        `frame_start`; `window_cams` is one camera group per window carrying per-frame
        extrinsics where a camera moves. Geometry first, pixels second: every animal's crop
        boxes are settled before anything decodes, so each (camera, frame) is read once and
        shared by every animal.

        A row needs `min_box_frames` all-finite boxes (one finite box must not fabricate a whole
        window's crop); with boxes, one box per camera is the UNION over the window's frames
        (only the cameras that saw the row), NOT re-squared through `crop_box_for_points`
        (a near-square union only grows); results are int32 and clamped. Stored (instances.pq)
        boxes ARE squared through the rule (`pad=0`); under `crop_source = 'keypoints'` the
        detector keypoints are unioned then squared once. The camera must describe the box the
        pixels were cut with: `apply_crop` sets `size`, `_resize_camera` makes it `scales`,
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
                    if cfg.crop_source == 'keypoints' and det_kpts_stc is not None:
                        kk = det_kpts_stc[a, fl, ci][..., :2].reshape(-1, 2)
                        kk = kk[np.isfinite(kk).all(-1)]
                        if len(kk):
                            kb = cropmod.crop_box_for_points(
                                torch.as_tensor(kk, dtype=torch.float32),
                                torch.tensor([w, h], dtype=torch.float32), cfg.min_crop_dim)
                            if kb is not None and _overlaps(kb, box):
                                box = kb.to(torch.int32)
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
                cgroup[i], s = _resize_camera(cam, pass1_res)
                scales.append(s)
            for i, ci in enumerate(use):
                crop[a, wl, ci] = np.asarray(boxes[i], np.float32)
            outcome[a, wl] = OUTCOMES.index('decode failed')
            plans.append((a, use, boxes, cgroup, scales, uncropped))
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
            different crop, and `_crop_views` never mutates what it is given. A file that will
            not decode takes out every animal that wanted this camera.
            """
            imgs = store.read(ci, session.cam_names[ci], frames, pool=pool)
            ok = not any(im is None for im in imgs)
            for a, use, boxes, cgroup, *_ in plans:
                if ci in use:
                    i = use.index(ci)
                    bx = boxes[i]
                    crops[a, ci] = (_crop_views(imgs, bx, cgroup[i]['size'].tolist())
                                    if ok else None)
            del imgs

        with ThreadPoolExecutor(max_workers=8) as pool:
            if len(cams) < 2:
                one(cams[0], pool) if cams else None
            else:
                with ThreadPoolExecutor(max_workers=min(cam_decode, len(cams))) as cpool:
                    list(cpool.map(lambda ci: one(ci, pool), cams))
        return crops

    def forward(frames, plan, crops, wi):
        """One animal, one window -> its prediction in the SOURCE frame, or None.

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
        `carried[a]` is left alone and the staleness bound in `_build_prior` retires it. The
        2D->source conversion undoes the resize, then adds the crop origin; the per-camera 2D
        pose takes the same expression per camera, so camera 0's per-camera pose is bit-identical
        to the reported prediction (a free exact check).
        """
        a, use, boxes, cgroup, scales, *_ = plan
        views = [crops[a, ci] for ci in use]
        if any(v is None for v in views):
            return None
        prior, prompt_t = _build_prior(cfg, carried[a], src, a, n_lab, frames, boxes,
                                       scales, mode, K, R, cgroup)
        dev = cfg.device
        chunk = cfg.kpt_chunk or None
        views = [v.to(dev) for v in views]
        cgroup_d = _to_device(cgroup, dev)
        mkw = {}
        if cfg.box_prompt != 'none':
            if cfg.box_prompt == 'labels':
                if src is None or a >= n_lab:
                    box_t = None
                else:
                    box_t = _deploy_box_prompt(mode, src, None, frames, a, use, boxes, scales,
                                              cgroup, dev)
            elif cfg.box_prompt == 'detector':
                if boxes_stc is None:
                    raise ValueError('box_prompt = "detector" needs detector boxes '
                                     '(--detector or --boxes); none were supplied.')
                box_t = _deploy_box_prompt(mode, None, boxes_stc, frames - f0, a, use, boxes,
                                          scales, cgroup, dev)
            if box_t is not None:
                mkw['box_prompt'] = box_t
                box_cams[a, wi - w0] = int(torch.isfinite(box_t).all(-1).any(1)[0].sum())
        with share_scene(model) if cfg.anchor == 'self' else nullcontext():
            out = model(views, kpt_ids.to(dev), cgroup_d, mode=mode,
                        kpt_prior=None if prior is None else prior.to(dev),
                        prompt_time=None if prompt_t is None else prompt_t.to(dev),
                        kpt_chunk=chunk, **mkw)
            if cfg.anchor == 'self':
                out = self_prompt(model, views, kpt_ids.to(dev), cgroup_d, mode, out,
                                  kpt_chunk=chunk, box_prompt=mkw.get('box_prompt'))
        p = out['coords_pred'][0].detach().cpu().numpy()
        q = None
        if mode == '2d':
            p = p / scales[0] + np.asarray(boxes[0][:2], np.float32)
            q = p
        elif cfg.carry_source == 'pred':
            q = p
        elif out.get('3d_pred_triangulate') is not None:
            q = out['3d_pred_triangulate'][0].detach().cpu().numpy()
        p2 = v2 = None
        if '2d_pred' in out:
            p2 = out['2d_pred'][:, 0].detach().cpu().numpy().copy()
            for i in range(len(use)):
                p2[i] = p2[i] / scales[i] + np.asarray(boxes[i][:2], np.float32)
            if out.get('vis_pred_2d') is not None:
                v2 = out['vis_pred_2d'][:, 0].detach().cpu().numpy()
        return p, q, out, p2, v2

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
        rule as `p`.
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
                    c, s = _resize_camera(cam, cfg.image_size)
                    cg.append(c)
                    sc.append(s)
                return (a, use, boxes, cg, sc, uncropped)

            refined = []
            for plan in plans:
                a, use, boxes, cgroup, scales, *_ = plan
                got = forward(frames, plan, crops, wi)
                if got is None:
                    refined.append(_at_image_size(plan))
                    continue
                pts = torch.as_tensor(got[0], dtype=torch.float32)
                cg2, b2 = boxes_from_points(pts, [window_cams[i] for i in use],
                                            cfg.min_crop_dim, mode)
                if cg2 is None:
                    refined.append(_at_image_size(plan))
                    continue
                if any(not _overlaps(b2[i], boxes[i]) for i in range(len(use))):
                    refined.append(_at_image_size(plan))
                    continue
                uncropped2, sc2 = list(cg2), []
                for i, cam in enumerate(cg2):
                    cg2[i], s = _resize_camera(cam, cfg.image_size)
                    sc2.append(s)
                for i, ci in enumerate(use):
                    crop_refined[a, wi - w0, ci] = np.asarray(b2[i], np.float32)
                refined.append((a, use, b2, cg2, sc2, uncropped2))
            plans = refined
            crops = decode_crops(frames, plans)

        for plan in plans:
            a, use, boxes, cgroup, scales, *_ = plan
            got = forward(frames, plan, crops, wi)
            if got is None:
                continue
            p, q, out, p2, v2 = got
            outcome[a, wi - w0] = OUTCOMES.index('ok')
            _fill_box_agreement(box_agree, a, frames - f0, use, boxes, p, mode, window_cams)
            if p2 is not None:
                for i, ci in enumerate(use):
                    pred2d[a, frames - f0, ci] = p2[i]
                    if v2 is not None:
                        conf2d[a, frames - f0, ci] = v2[i]
            vlogit = None
            if 'vis_pred' in out:
                v = out['vis_pred'][0].detach().cpu().numpy().reshape(len(frames), K)
                conf[a, frames - f0] = v
                vlogit = v
            j = max(0, len(frames) - cfg.overlap) if cfg.overlap else len(frames) - 1
            if cfg.vis_thresh is not None and vlogit is not None:
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore', RuntimeWarning)
                    med = np.nanmedian(vlogit, axis=-1)
                drop = ~(med >= cfg.vis_thresh)
                p = p.copy()
                p[drop] = np.nan
                conf[a, frames[drop] - f0] = np.nan
                box_agree[a, frames[drop] - f0] = np.nan
            pred[a, frames - f0] = p
            if q is not None:
                carried[a] = (torch.as_tensor(q[j]), int(frames[j]),
                              None if vlogit is None else torch.as_tensor(vlogit[j]))

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

            boxes_stc = det_kpts_stc = None
            if _det_pool is not None:
                boxes_stc, _scores, det_kpts_stc = _pending_det.result()
                _pending_det = _det_pool.submit(_detect, bi + 1)
            elif boxes_for is not None:
                boxes_stc, _scores, det_kpts_stc = _detect(bi)

            det_box = np.full((S, n_blk, len(session.rig), 4), np.nan, np.float32)
            det_score = np.full((S, n_blk, len(session.rig)), np.nan, np.float32)
            if boxes_stc is not None:
                det_box[...] = boxes_stc
                det_score[...] = _scores

            pred = np.full((S, n_blk, K, R), np.nan, np.float32)
            conf = np.full((S, n_blk, K), np.nan, np.float32)
            pred2d = np.full((S, n_blk, len(session.rig), K, 2), np.nan, np.float32)
            conf2d = np.full((S, n_blk, len(session.rig), K), np.nan, np.float32)
            box_agree = np.full((S, n_blk, len(session.rig)), np.nan, np.float32)
            outcome = np.full((S, n_win), OUTCOMES.index('no box'), np.int8)
            crop = np.full((S, n_win, len(session.rig), 4), np.nan, np.float32)
            crop_refined = (np.full_like(crop, np.nan) if cfg.refine else None)
            box_cams = (np.full((S, n_win), -1, np.int8) if cfg.box_prompt != 'none' else None)

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
            yield {'pred': pred[:, :keep], 'conf': conf[:, :keep],
                   'pred2d': pred2d[:, :keep], 'conf2d': conf2d[:, :keep],
                   'box_agree': box_agree[:, :keep],
                   'det_box': det_box[:, :keep], 'det_score': det_score[:, :keep],
                   'animal_ids': np.asarray(animal_ids, object),
                   'outcome': outcome, 'crop': crop,
                   'window_start': np.asarray(starts[w0:w1]),
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


def _overlaps(a, b):
    """Do two xyxy boxes share any area? Either being non-finite is not an overlap."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    return bool(min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))


def _fill_box_agreement(box_agree, a, frames, use, boxes, p, mode, window_cams):
    """Distance from the predicted centroid to each crop box's centre, in units of one box side.

    A 3D pose is reprojected through the source camera per frame (a moving rig is handled by
    `project_points_torch`'s own (T,4,4) alignment). The box is the window's union box, i.e. what
    the pixels were cut with. In 2D `p` is already source pixels. A frame with no finite
    keypoint makes `nanmean` warn, which is suppressed.
    """
    for i, ci in enumerate(use):
        b = np.asarray(boxes[i], np.float64)
        side = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
        if not (np.isfinite(b).all() and side > 0):
            continue
        if mode == '2d':
            q = np.asarray(p, np.float64)
        else:
            q = project_points_torch([window_cams[ci]],
                                     torch.as_tensor(p, dtype=torch.float32))[0].numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            c = np.nanmean(q, axis=-2)
        centre = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        box_agree[a, frames, ci] = np.linalg.norm(c - centre, axis=-1) / side


ORACLE_CORRUPTIONS = ('off', 'stale', 'other', 'near', 'swap')


def _prior_to_model_frame(p, mode, boxes, scales):
    """Source pixels/world -> the model's own crop frame (3D passes straight through). Factored
    out so `_corrupt_prior`'s `near` applies the identical conversion to a candidate neighbour.
    """
    if mode == '2d':
        return (p - torch.as_tensor(np.asarray(boxes[0][:2], np.float32))) * scales[0]
    return p


def _nearest_eligible_row(src, a, t0, boxes, scales, mode, cgroup):
    """The nearest other label row whose pose at this frame would survive `prior_out_of_bounds` --
    the same eligibility test `dataset.py`'s `prompt_swap_animal` applies. Returns a row index, or
    None if no other row qualifies.

    "Nearest" is measured in the model's own frame over keypoints both rows have finite and in
    bounds -- the frame the model will see the corrupted prior in.
    """
    target = _prior_to_model_frame(torch.as_tensor(src[a][t0], dtype=torch.float32),
                                   mode, boxes, scales)
    best_row, best_dist = None, None
    for row in range(len(src)):
        if row == a:
            continue
        cand = _prior_to_model_frame(torch.as_tensor(src[row][t0], dtype=torch.float32),
                                     mode, boxes, scales)
        oob = prior_out_of_bounds(cand, mode, cgroup)
        if int((~oob).sum()) < 2:
            continue
        both = torch.isfinite(target).all(-1) & torch.isfinite(cand).all(-1) & ~oob
        if int(both.sum()) < 1:
            continue
        d = float(torch.linalg.norm((cand - target)[both], dim=-1).mean())
        if best_dist is None or d < best_dist:
            best_row, best_dist = row, d
    return best_row


def _swap_kpt_pairs(p, n_pairs, seed):
    """`swap:<n>` -- n transpositions among `p`'s own finite keypoints: the direct inference probe
    for `dataset.py`'s `prompt_swap_kpt_pairs`.
    """
    finite = torch.isfinite(p).all(-1).nonzero(as_tuple=True)[0]
    if len(finite) < 2:
        return p
    rng = np.random.default_rng(seed)
    idx = finite[torch.from_numpy(rng.permutation(len(finite)))]
    n = min(n_pairs, len(idx) // 2)
    p = p.clone()
    for i in range(n):
        j, k = int(idx[2 * i]), int(idx[2 * i + 1])
        p[j], p[k] = p[k].clone(), p[j].clone()
    return p


def _corrupt_prior(cfg, src, a, n_lab, frames, boxes, mode, cgroup, scales=None):
    """The oracle prior, optionally broken on purpose. Returns (pose in the SOURCE frame, qt).

    Measures alpha = d(output)/d(prior), the echo coefficient, which decides whether the prompt
    needs retraining. None of the corruptions is in training except through `dataset.py`'s
    `prompt_swap_kpt_pairs` / `prompt_swap_animal`, which `swap` and `near` probe directly:

    - `off:<x>`   a whole-body offset of x crop widths (the shape of a lag)
    - `stale:<n>` the pose from n frames earlier
    - `other`     the neighbouring animal's pose (`a + 1 % n_lab`)
    - `near`      the nearest eligible animal's pose instead of the fixed `a + 1` row
    - `swap:<n>`  n transpositions of this row's own keypoints

    Magnitudes are in CROP WIDTHS: sessions in one root mix pixels and millimetres. The direction
    is drawn from a generator seeded on (row, window), so two arms over one clip corrupt
    identically and the comparison is matched.

    `off` displaces by ONE crop width converted into whatever units the prior lives in:
    `get_camera_scale` is offset-invariant (a Jacobian), and 0.3.5 collapses a per-frame offset
    inside `get_camera_scale` itself; the width is the camera's own (`cgroup`'s camera is the
    reduced pass-1 one under `--refine-px`), not `cfg.image_size`.
    """
    kind, _, amt = (cfg.oracle_corrupt or '').partition(':')
    row, t0 = a, int(frames[0])
    if kind == 'other' and n_lab > 1:
        row = (a + 1) % n_lab
    elif kind == 'near' and n_lab > 1:
        nr = _nearest_eligible_row(src, a, t0, boxes, scales, mode, cgroup)
        if nr is not None:
            row = nr
    elif kind == 'stale':
        t0 = max(0, t0 - int(amt))
    p = torch.as_tensor(src[row][t0], dtype=torch.float32)
    if kind == 'swap':
        p = _swap_kpt_pairs(p, max(1, int(amt) if amt else 1), [a, t0, 0x5A7])
    if kind != 'off':
        return p, int(t0) - int(frames[0]) if kind == 'stale' else 0
    if mode == '2d':
        b = np.asarray(boxes[0], np.float64)
        width = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
    else:
        from posetail.posetail.cube import get_camera_scale
        fin = p[torch.isfinite(p).all(-1)]
        if not len(fin):
            return p, 0
        scale = torch.nanmedian(get_camera_scale(cgroup, fin[None]))
        if not torch.isfinite(scale):
            return p, 0
        width = float(scale) * int(cgroup[0]['size'].max())
    rng = np.random.default_rng([a, int(frames[0])])
    v = rng.normal(size=p.shape[-1])
    v = v / max(float(np.linalg.norm(v)), 1e-9)
    return p + torch.as_tensor(float(amt) * width * v, dtype=p.dtype), 0


def _build_prior(cfg, carried, src, a, n_lab, frames, boxes, scales, mode, K, R, cgroup):
    """The per-keypoint prior for this window, in the model's coordinate frame.

    The prompt frame is not always 0: `carried[1]` holds the frame the carried pose describes,
    which differs on the last window of a group. And a prior outside the crop is not a prior --
    NaN is the right value, since it is what the no-query tokens key off.

    `labels` is an ORACLE: ground truth as the prior, not a deployment number, and a detector row
    with no label row behind it gets none. `carry` treats a stale prior as no prior: `carried` is
    only written on a window that predicted, so a lost animal hands back a pose from before this
    window; `qt < 0` (not `-qt > overlap`) is the exact test -- consecutive windows give qt == 0
    and a pulled-back last window qt > 0, so a negative qt happens iff a window was skipped.
    """
    if cfg.anchor in ('none', 'self'):
        return None, None
    if cfg.anchor == 'labels':
        if src is None or a >= n_lab:
            return None, None
        p, qt = _corrupt_prior(cfg, src, a, n_lab, frames, boxes, mode, cgroup, scales)
        if p is None:
            return None, None
    else:
        if carried is None:
            return None, None
        p = carried[0].clone().float()
        qt = int(carried[1]) - int(frames[0])
        if qt < 0:
            return None, None
    if p.shape != (K, R):
        return None, None
    p = _prior_to_model_frame(p, mode, boxes, scales)
    p = p.clone()
    p[prior_out_of_bounds(p, mode, cgroup)] = float('nan')
    qt = min(max(qt, 0), len(frames) - 1)
    return p[None], torch.full((1, K), qt, dtype=torch.int32)
