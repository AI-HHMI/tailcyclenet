"""THE inference path. One window loop.

posetail-pose had ten scripts that ran a model and three separate window loops, and all three
got the loop wrong in different ways. There is one here, and everything else -- eval, rendering,
long clips, multi-animal -- is an argument to it.

The loop, per group:

    for each window of T frames, stepping by T - overlap:
        get a crop box per camera        (from labels, from a detections file, or from a detector)
        build the cropped+resized cameras
        read the pixels
        build the prior                  (none / carry / self)
        forward
        map the prediction back into the source coordinate frame

**The prompt regime is the thing to get right.** `none` is query-free. `carry` seeds each window
from the model's own previous prediction -- label-free, and what deployment actually does; it
requires `overlap >= 1` or there is nothing to carry. `self` runs each window twice, seeding the
second pass from the first. `labels` seeds from ground truth and is an ORACLE: it is not a
deployment number and is off by default, because in the project this descends from, ungated
GT-derived priors inflated every anchored number that was ever published.
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

# WHY AN (ANIMAL, WINDOW) PRODUCED NOTHING. Five separate aborts in the loop below wrote the same
# NaN, so a coverage number could not be decomposed at all -- "the detector offered no box", "the
# association matched no camera", "the crop rule refused" and "the file would not decode" are four
# different problems with four different fixes, and they arrived indistinguishable.
OUTCOMES = ('ok', 'no box', 'no camera', 'no points', 'crop failed', 'decode failed')

# How many cameras `decode_crops` may decode at once. A memory bound, not a core count -- see there.
_CAM_DECODE = 4

# The smallest frame store worth building, so a root with small frames does not take two-window
# blocks and pay per-block overhead thousands of times over a long clip. Frames, not windows, is
# the right unit here: what makes a block cheap is bytes.
_MIN_STORE_BYTES = 512 << 20


@dataclass
class InferConfig:
    n_frames: int = 24
    overlap: int = 4
    image_size: int = 256
    min_crop_dim: int = 64
    anchor: str = 'carry'
    max_animals: int = 0          # 0 -> every animal the box source offers
    max_frames: int = 0           # 0 -> the whole group; else its first `max_frames` frames
    kpt_chunk: int = 0            # 0 -> decode every keypoint in one pass
    # None -> report every row predicted. A float withholds an (animal, frame) row whose MEDIAN
    # `vis_pred` logit across keypoints is below it. NOT PORTABLE across roots (logit medians
    # differ by an order of magnitude), so there is no default; score it against a rate-matched
    # random rejection, which is the only honest control.
    vis_thresh: float | None = None
    # Re-crop each window to the FIRST PASS's own prediction and predict again. Label-free, and it
    # costs one extra forward AND one extra decode per animal per window (the crop moves, so no
    # pixels and no scene encode can be shared).
    # None -> DERIVED FROM THE SESSION'S MODE: on in 3D, off in 2D. 3D is a clean win; 2D is a
    # TRADE -- bulk accuracy improves while multi-animal identity gets significantly worse -- so
    # 2D keeps it off and `--refine` turns it on.
    refine: bool | None = None
    # PASS 1'S INPUT RESOLUTION under `--refine`. None -> `image_size`. Refine's gain is
    # MAGNIFICATION, not coordinate frame, so pass 1 only has to LOCALISE. No shipped default: the
    # floor is patch-size- and root-dependent, and 64 is a cliff in both dimensions. Sweep it.
    # `model.PoseTrackerEncoder.forward` is what makes a smaller input correct; see `_input_extent`.
    refine_px: int | None = None
    # WHERE THE WINDOW'S CROP COMES FROM. 'boxes' unions the detector's per-frame boxes.
    # 'keypoints' runs THE CROP RULE on the detector's own keypoints over the window -- see the
    # union comment below. Needs a keypoint-trained detector; ignored without one.
    crop_source: str = 'boxes'
    # How many finite (frame, camera) boxes a row needs before it gets a window crop at all. 1 is
    # what the loop always did, and it is why coverage can be FABRICATED: one box out of T x C
    # positions a crop for every frame and marks them all `ok`. Raising it LOWERS reported
    # coverage, and that is the point.
    min_box_frames: int = 1
    # WHAT `carry` FEEDS BACK.
    #   'triangulate' -- the ANCHOR-FREE estimate (`3d_pred_triangulate`). Breaks the loop.
    #   'pred'        -- the reported prediction, which under gridresid_offset = "query" IS
    #                    `prior + residual` and so integrates its own error window over window.
    # 2D is identical either way: no triangulation at one camera, and `coords_pred` is an absolute
    # pixel decode, so nothing is being fed its own anchor.
    carry_source: str = 'triangulate'
    # DELIBERATELY BREAK THE ORACLE PRIOR, to measure how far the output follows it. `--anchor
    # labels` + this gives the echo coefficient alpha = d(output)/d(prior) without a training run.
    # `off:<x>` | `stale:<n>` | `other`. See `_corrupt_prior`. Never a deployment arm.
    oracle_corrupt: str | None = None
    device: str = 'cuda:0'
    # Read from the RUN's own `[data]`, like `min_crop_dim` -- never from a CLI flag. A model
    # trained on `instances` crops and evaluated on keypoint crops is being scored against a crop
    # rule it never saw, and nothing in the output would say so.
    box_source: str = 'keypoints'
    # DEPLOYMENT BOX PROMPT. Which-animal-occupies-this-box, fed as a non-position channel to a
    # `[model].box_prompt` model. 'none' | 'labels' (GT boxes, ORACLE -- gate off like
    # `--anchor labels`) | 'detector'. GUARDED: 'none' + crop_inflate 1.0 is byte-identical to a
    # run without these keys, which `tests/test_infer.py` pins. LIVE IN 2D AND 3D, PER CAMERA. A
    # camera with no finite point that frame gets a NaN column, which the encoder's learned
    # no-box token substitutes -- the regime `box_prompt_dropout` trains.
    box_prompt: str = 'none'
    # Inflate every crop about its centre -- the WIDE pass-1 regime where the box is load-bearing.
    # 1.0 is today's behaviour exactly.
    crop_inflate: float = 1.0
    # HOW MANY WINDOWS AHEAD TO DECODE while the current window's forward runs on the GPU.
    # BIT-EXACT: `_build_plans`/`decode_crops` depend only on the box source and window geometry,
    # never on `carried` or any model output, so `carried` is still read and written strictly in
    # window order on the main thread inside `_process_window`. 0 is the exact old serial path.
    prefetch_windows: int = 1


def _window_starts(n_frames: int, T: int, overlap: int):
    """Contiguous windows covering [0, n_frames), stepping by T - overlap.

    The last window is pulled back to end exactly at n_frames rather than padded, so no frame is
    predicted from duplicated pixels.
    """
    step = max(1, T - overlap)
    if n_frames <= T:
        return [0]
    starts = list(range(0, n_frames - T + 1, step))
    if starts[-1] + T < n_frames:
        starts.append(n_frames - T)
    return starts


def boxes_from_points(points, cgroup, min_crop_dim, mode):
    """Crop boxes for one animal in one window, from points. THE crop rule, shared with training.

    In 3D the points are world coordinates and get projected; in 2D they are already pixels.
    Returns None when nothing is finite -- the animal is not croppable in this window.

    `cgroup` is built by the caller (once per window, via `Session.cgroup`) rather than here:
    building it per animal both dropped the per-frame extrinsics and rebuilt every camera once
    per animal.
    """
    if mode == '3d':
        cg, boxes = cropmod.crop_to_points_3d(cgroup, points, min_crop_dim)
        return (cg, boxes) if cg is not None else (None, None)
    cam, box, _ = cropmod.crop_to_points_2d(cgroup[0], points, min_crop_dim)
    return ([cam], [box]) if cam is not None else (None, None)


def _deploy_box_prompt(mode, src_pts, boxes_stc, frames, a, use, boxes, scales, cgroup, dev):
    """The box-prompt tensor for one animal's window, PER CAMERA, 2D or 3D. (1,T,C,4) in crop
    pixels, C = len(use) -- column order matches cgroup/views/use.

    `src_pts` names WHICH animal to return: the GT points under box_prompt = 'labels' (an
    ORACLE), or None under 'detector', where boxes_stc[a] supplies per-camera boxes instead.
    `frames`/`a`/`use` index into whichever source is live.

    3D + labels calls box_prompt.compute_box_prompt DIRECTLY on the animal's world points and
    this window's own cropped+resized cgroup -- byte-for-byte the training-time computation
    (dataset._item), not a re-derivation, which is what removes the class of bug gotcha 8 exists
    for. 3D + detector maps each camera's own box into that camera's crop frame and runs THE crop
    rule on it, per camera -- under --track these are one cross-view target's own per-camera
    boxes, so "which animal" is already cross-view consistent and no new association is needed.

    A camera absent from `use` is absent from the output too (never a NaN placeholder at the
    wrong index): _box_features normalises columns against ctx['sizes'], built from
    preprocessed_views in the SAME `use` order, so index i must mean the same camera in both.
    """
    from .. import box_prompt as bpmod

    if mode == '3d' and src_pts is not None:
        # LABELS, 3D: identical to what _item computes at training time -- cgroup here is
        # already this window's cropped+resized camera list.
        pts = torch.as_tensor(src_pts[a][frames], dtype=torch.float32)      # (T,K,3) world
        return bpmod.compute_box_prompt(pts, cgroup, '3d')[None].to(dev)

    if mode == '3d':
        # DETECTOR, 3D: per camera, map that camera's own box corners into ITS crop frame and
        # box them there -- a 2D box-in-one-view computation repeated per camera, not a 3D
        # projection, because the detector box is a per-camera detection to begin with.
        T = len(frames)
        C = len(use)
        out = torch.full((T, C, 4), float('nan'), dtype=torch.float32)
        for i, ci in enumerate(use):
            db = torch.as_tensor(boxes_stc[a][frames][:, ci], dtype=torch.float32)  # (T,4) src px
            if not torch.isfinite(db).any():
                continue
            corners = cropmod.box_corners(db)                                # (T,4,2) src px
            origin = torch.as_tensor(boxes[i][:2], dtype=torch.float32)
            cf = (corners - origin) * float(scales[i])                       # (T,4,2) crop px
            size = cgroup[i]['size']
            for t in range(T):
                bx = cropmod.crop_box_for_points(cf[t], size,
                                                 bpmod.BOX_PROMPT_MIN_DIM, bpmod.BOX_PROMPT_PAD)
                if bx is not None:
                    out[t, i] = bx.to(torch.float32)
        return out[None].to(dev)

    # 2D, single camera -- UNCHANGED result from before this fix.
    source = (torch.as_tensor(src_pts[a][frames], dtype=torch.float32) if src_pts is not None
              else cropmod.box_corners(torch.as_tensor(boxes_stc[a][frames][:, use[0]],
                                                       dtype=torch.float32)))
    origin = torch.as_tensor(boxes[0][:2], dtype=torch.float32)
    cf = (source - origin) * float(scales[0])                    # (T,K,2) crop px
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
    return [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
            for c in cgroup]


def _crop_views(imgs, box, target_size):
    """Crop+resize one camera's ALREADY-DECODED window -> (1,T,H,W,3) uint8.

    The same fused rotate/crop/resize affine `load_image` applies at decode time, applied here
    instead. The decode is per (camera, frame) and the crop is per (animal, camera, frame), so
    doing both inside the animal loop paid the full-frame decode once per animal: rat-city's
    twelve rats over a 24-frame window decoded the same 24 images twelve times. The warp itself
    is ~0.2 ms against ~27 ms for the decode it no longer repeats.

    `box` is either ONE `[x1,y1,x2,y2]` for the whole window, or a (T,4) of per-frame boxes (a
    decode-level capability kept for the loader's per-frame crops; no deployed flag uses it). The
    one-box path is kept as a single affine computed once, so the common case is the same
    arithmetic it always was.
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
    """Re-query at the model's OWN frame-0 prediction. THE label-free prompted regime.

    `first` is a completed prior-free pass. Its frame-0 pose becomes the prior for a second pass,
    which is what a deployed model does on the first window of a clip and what the periodic val
    eval reports alongside the prior-free number. No ground truth is read, so no gate reopens.

    The frame-0 pose is already in the model's own frame, so it needs no conversion -- but it does
    need the same BOUNDS MASK `carry` gets, and it did not have one. A keypoint the first pass put
    outside its own crop, or in 3D somewhere no camera pair can see, was handed back as a confident
    prior; NaN is the right value, because it is what the no-query tokens key off, so the keypoint
    degrades to "I was not told" instead of "I was told a lie".

    Shared with the trainer deliberately: this repo has one window loop and it should have one
    self-prompt, or the number training reports and the number inference produces drift apart.
    """
    p = first['coords_pred'][0].detach()
    prior = p[0][None].clone()                         # (1,K,R), the frame-0 pose
    prior[0, prior_out_of_bounds(prior[0], mode, cgroup)] = float('nan')
    qt = torch.zeros(prior.shape[:2], dtype=torch.int32, device=prior.device)
    # THE BOX PROMPT carries into the second pass unchanged (report 27): the window has not moved,
    # so it describes the same crop. `None` (the default) is a plain model, unaffected.
    mkw = {} if box_prompt is None else {'box_prompt': box_prompt}
    return model(views, kpt_ids, cgroup, mode=mode, kpt_prior=prior, prompt_time=qt,
                 kpt_chunk=kpt_chunk, **mkw)


def _plan_blocks(starts, n_frames, T_total, frame_cost, store_bytes):
    """Group the windows into BLOCKS whose frames fit the store at once. -> [(w0, w1), ...].

    A block is the unit that holds pixels and emits rows, so its size is what keeps peak memory
    off the length of the clip. Greedy and maximal: the bigger the block, the fewer boundaries,
    and a boundary costs one prefetch stall and one re-read of the seam frames.

    At least one window always, which is what the refusal in `run_blocks` guarantees is affordable
    before this is called -- so the caller never has to handle an empty plan.
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


# Which axis each returned column runs along, so blocks can be stitched back into a whole clip.
# Anything not named here is a scalar or a per-group constant and is taken from the first block.
_FRAME_KEYS = ('pred', 'conf', 'pred2d', 'conf2d', 'box_agree')
_WINDOW_KEYS = ('outcome', 'crop', 'crop_refined', 'box_prompt_cams', 'window_start')


def merge_blocks(blocks):
    """Every block of `run_blocks` stitched into the one dict `run_group` used to return.

    Frame- and window-indexed columns concatenate on axis 1 (axis 0 for the 1-D `window_start`);
    everything else is a per-group constant and comes from the first block. This is also the
    ORACLE the block-invariance test compares against -- the same role `prefetch_windows = 0`
    plays for the prefetch test -- so it is not test-only scaffolding.
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
               cfg: InferConfig, box_points=None, boxes_for=None, n_rows=None):
    """Predict every animal in one group, a BLOCK OF WINDOWS AT A TIME. Yields one dict per block.

    Arrays are in the SOURCE coordinate frame. Crops come from exactly one of two sources, and
    they are NOT comparable:

    - `box_points` (S,T,K,R): points the crop rule follows, shaped like the labels. Passing the
      labels themselves is the GT-crop upper bound.
    - `boxes_for(store, lo, hi) -> (boxes, scores, kpts)`: boxes for frames `[lo, hi)`, from a
      detector or a detections file, `(S, hi-lo, C, 4)`. This is the deployment number.

    Whichever was used is recorded in the result so a caller cannot quote one as the other.

    **WHY BLOCKS.** Nothing here may be proportional to the length of the clip: a 200 fps hour is
    720,000 frames and an ordinary recording, and the arrays it used to allocate up front came to
    82 GB (dev/reports/38 §4). A block owns a bounded frame span, allocates only that, and is
    handed to the caller to write out before the next one starts.

    **A BLOCK'S FRAMES ARE EXACTLY `[starts[w0], starts[w1])`, which PARTITIONS the clip.** A frame
    in an overlap belongs to the LAST window containing it (eval rule 11), so the seam frames a
    block's final window also touches belong to the NEXT block and are dropped here -- exactly as
    the whole-clip loop overwrote them. That is what makes block-wise output byte-identical to
    whole-clip output, and it is why `merge_blocks` can simply concatenate.

    `boxes_for` is a callback and not an array because the array is the thing that would not fit:
    it lets the caller detect only the frames this block needs, over the pixels the store already
    holds. `n_rows` is `S` on that path, since the row count has to be known before the first
    block and the boxes do not exist yet.
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
    # `refine` DEFAULTS BY DIMENSIONALITY -- see `InferConfig.refine`. Resolved here, where `mode`
    # is known, and folded back into `cfg` so every consumer downstream (and the recorded output)
    # sees one concrete value instead of a tri-state nobody else should have to interpret.
    if cfg.refine is None:
        cfg = replace(cfg, refine=(mode == '3d'))
    # A reduced pass-1 resolution is only a thing when there IS a second pass. Without `--refine`,
    # pass 1 is the only pass and its output is the answer.
    pass1_res = cfg.refine_px if (cfg.refine and cfg.refine_px) else cfg.image_size
    # A PREFIX of the group, not a sample of it: `carry` needs the frames contiguous, and
    # rat-city's posetail-pose protocol is frames 0-479 of its one test trial.
    T_total = min(group.n_frames, cfg.max_frames or group.n_frames)
    # The registry is per DATASET and the keypoint axis is per SESSION -- and `scripts/infer.py`
    # will happily hand us a bare session directory. `ids_for` aligns to the axis we actually
    # hold, by name, so a session that reorders the root's keypoints or carries a subset of them
    # is predicted correctly instead of being silently relabelled.
    kpt_ids = torch.as_tensor(registry.ids_for(dataset_name, session.names),
                              dtype=torch.long)[None]
    assert kpt_ids.shape[1] == K

    src = box_points if box_points is not None else (
        lab.points3d if mode == '3d' else lab.points2d[..., 0, :])
    # The run's own crop rule, applied to the GT-crop path. `lab.boxes` is ALREADY the (S,T,C,4)
    # `boxes_stc` shape and the union-over-window logic below is already right for a pre-padded
    # box, so this needs no second code path. An explicit detector or `--boxes` npz still wins:
    # those are what the caller asked for, and both are recorded in `boxes_from`.
    #
    # Kept SEPARATE from `boxes_stc` rather than assigned into it, because the two fail
    # differently. A detector that offers no box for an animal has said something; a tracker that
    # lost it has not, and the loader's own answer there is to fall back to the keypoints. Folding
    # this into `boxes_stc` would drop those animals from the window instead -- pure lost
    # coverage, silently, and coverage is a number this repo reports.
    inst_boxes = (lab.boxes if (boxes_for is None and box_points is None
                                and cfg.box_source == 'instances' and lab.boxes is not None
                                and bool(np.isfinite(lab.boxes).any())) else None)
    # `n_rows` ON THE BOX PATH, because the boxes do not exist yet. `S` used to come from
    # `boxes_stc.shape[0]`, and there is no whole-clip box array any more -- the caller knows the
    # row count (it is `--max-animals` or the session's own animal count) and states it.
    n_src = (n_rows if boxes_for is not None else src.shape[0])
    S = n_src if cfg.max_animals == 0 else min(n_src, cfg.max_animals)
    # ONE camera in 2D, exactly as the loader picks it (`dataset.py`, `true_2d -> cam_ix = [0]`).
    # A 2D session may still ship a multi-camera rig -- rat-city's `calibration.toml` describes the
    # arena, not the pose input -- and the library asserts a single view for R == 2. Unbranched,
    # the box path reached that assert and the keypoint path raised IndexError before it, because
    # `boxes_from_points` returns one box while `use` stayed the full rig.
    cam_ix = [0] if mode == '2d' else list(range(len(session.rig)))
    # A DETECTOR ROW IS NOT A LABEL ROW. `S` comes from the box source, which on the deployment
    # path is the detector and can offer more animals than the session ever labelled -- so `src`
    # is indexable only up to its own length. And once boxes come from a detector, row `a` is not
    # label row `a` for ANY `a`: the rows are score-ordered, or association-ordered, and the
    # labels' own ids would be a claim about identity that nothing established. So every row wears
    # an invented id on that path, and `eval.py` Hungarian-matches rather than trusting the index.
    n_lab = 0 if src is None else len(src)
    animal_ids = ([f'det{a:02d}' for a in range(S)] if boxes_for is not None else
                  [lab.animal_ids[a] if a < len(lab.animal_ids) else f'det{a:02d}'
                   for a in range(S)])

    # THE ANCHOR-FREE ESTIMATE IS NOT AN OUTPUT COLUMN, but it is still what `carry` feeds back.
    # `3d_pred_triangulate` is read live out of `out` in `forward()` below and handed to the next
    # window; `--carry-source triangulate` is the shipped 3D default and does not move. What went
    # is the `(S,T,K,3)` array that shadowed it for the whole clip, plus its `tri_degenerate`
    # companion -- two of the arrays that made a long clip unrepresentable, for a diagnostic no
    # protocol reads.
    carried = [None] * S                      # per-animal prior for the next window
    # THE DIAGNOSTICS, per (animal, window): why it produced nothing, and what box it was given.
    # Both are what makes a coverage delta readable -- 08's crop-inflation measurement needed the
    # box, and every one of the five aborts below needed to be distinguishable from the others.
    starts = _window_starts(T_total, cfg.n_frames, cfg.overlap)

    # THE PIXEL BUDGET, AND THE ONE REFUSAL IT CAN RAISE.
    #
    # `cam_decode` bounds how many cameras decode CONCURRENTLY; each task holds one camera's whole
    # window of FULL frames, so unbounded concurrency on johnson's 16-camera 3208x2200 rig is
    # 16 x 12 x 21.2 MB = 4.1 GB of transient host memory for a rig with one animal in it.
    #
    # `store` holds the frames themselves, and unlike the cache it replaces it does NOT degrade.
    # The old one shrank to nothing on a tight budget and re-decoded, which was a silent 3x on
    # wall clock; a block is instead sized so its frames FIT, and a budget too small for even one
    # window is refused with the arithmetic. There is no third option: refine pass 2 crops from the
    # same frames pass 1 did, so dropping them means either decoding twice or re-cropping from a
    # stored crop, and the second is a double resample -- not output-neutral.
    #
    # SIZED FROM PARSED TOML (`rig.size`), never by opening a container (gotcha 10). `max` over the
    # cameras, not camera 0: rat-city-annotated ships 4696x2048 beside 4500x2050, and pricing the
    # rig off one camera under-sizes it.
    _frame_bytes = max(int(w) * int(h) for w, h in
                       (session.rig.size(session.cam_names[ci]) for ci in cam_ix)) * 3
    _frame_cost = len(cam_ix) * _frame_bytes            # one frame INDEX, across the rig
    _budget = memory.current()
    _one = cfg.n_frames * _frame_cost
    # SPEND WHAT THE WORK NEEDS, NOT WHAT THE HOST HAPPENS TO HAVE.
    #
    # A block only has to hold the window being forwarded plus the one being prefetched, and those
    # overlap by `n_frames - step`. Past that, a bigger block buys exactly one thing: fewer block
    # boundaries, each worth one prefetch stall -- the seam frames are NOT re-decoded, since
    # eviction keeps them for the next block. **MEASURED FLAT:** johnson, 120 frames x 16 cameras
    # of 3208x2200, full detector-to-pose path, ran 62.4 s with a store of one window and 61.7 s
    # with the whole clip resident (a 40.3 GB peak against 10.5). Wall clock over 10..288 GB of
    # budget spans 60.9-65.1 s with no trend.
    #
    # So the budget is a CEILING and this is the ask. An unconstrained host would otherwise pull
    # the whole clip into one block simply because it fits, which is how a 120-frame clip came to
    # hold 40 GB of frames to do 10 GB of work.
    #
    # The floor keeps a small-frame root from paying per-block overhead for nothing: a 2D rig with
    # 6 MB frames would otherwise take blocks of two windows and split a 57,000-frame clip into
    # thousands of them.
    _want_store = max(2 * _one, _MIN_STORE_BYTES)
    _store_bytes = min(_budget.share(memory.FRACTION_STORE), _want_store)
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

    # BLOCK-LOCAL STATE, REBOUND ONCE PER BLOCK. The nested functions below close over these names
    # and read them at call time, so rebinding here is what makes them block-scoped without
    # threading a context object through six signatures. `f0` is the block's first frame and `w0`
    # its first window: every array below is indexed `[a, frame - f0]` or `[a, wi - w0]`.
    f0, w0 = 0, 0
    pred = conf = box_agree = outcome = crop = crop_refined = box_cams = None
    pred2d = conf2d = None
    boxes_stc = det_kpts_stc = None
    # Settled once: which of the three crop sources produced these pixels. Recorded on every block
    # so a reader of one block knows as much as a reader of the merged group.
    _boxes_from = ('detector' if boxes_for is not None else
                   ('given points' if box_points is not None else
                    ('instances.pq' if inst_boxes is not None else 'labels')))
    #
    def _build_plans(wi, start):
        """Everything the old inline loop did BEFORE any pixel touches: pure geometry.

        Returns (frames, window_cams, plans). Writes into `outcome`/`crop`, which are
        pre-allocated and indexed by `wi` -- two different windows' calls touch disjoint
        slices, so this may run for window wi+1 on a background thread while window wi's
        forward is still running on the main thread. IT NEVER READS `carried`: pass-1 crop
        boxes depend only on the box source (`boxes_stc`/`inst_boxes`/`src`) and window
        geometry, never on the carried prior or any model output -- `carried` is read only
        inside `forward()`, on the main thread, in window order. So preparing a future
        window's pixels changes no pixel and no order. See the driver below
        (`cfg.prefetch_windows`, dev/reports/31).
        """
        frames = np.arange(start, min(start + cfg.n_frames, T_total))
        if len(frames) < 2:                   # T=1 hits posetail's gT = T // tubelet = 0 bug
            frames = np.clip(np.arange(start, start + 2), 0, T_total - 1)
        fl = frames - f0                      # into this block's arrays; see `run_blocks`
        wl = wi - w0
        # ONE camera group per window, carrying per-frame extrinsics where a camera moves. Built
        # here rather than per animal: the old per-animal build dropped `moving_ext` entirely and
        # cost O(C^2) `format_camera` calls per animal.
        window_cams = session.cgroup(gid, frames)
        # GEOMETRY FIRST, PIXELS SECOND. Every animal's crop boxes are settled before anything is
        # decoded, so the decode loop below can read each (camera, frame) once and hand the same
        # buffer to every animal that wants it.
        plans = []
        for a in range(S):
            bb = None
            if boxes_stc is not None:
                bb = boxes_stc[a][fl]                              # (t, C, 4)
                # NOT `.any()`: one finite box out of T x C used to position a crop for the whole
                # window and mark every frame `ok`. See `cfg.min_box_frames`.
                if int(np.isfinite(bb).all(-1).sum()) < cfg.min_box_frames:
                    continue
            elif inst_boxes is not None and a < len(inst_boxes):
                bb = inst_boxes[a][frames]
                if int(np.isfinite(bb).all(-1).sum()) < cfg.min_box_frames:
                    bb = None                                      # keypoint fallback, per animal
            if bb is not None:
                # Boxes given directly (detector / detections file / instances.pq). One box per
                # camera for this window: the union over the window's frames, so the animal does
                # not walk out of its own crop mid-window.
                # USE THE CAMERAS THAT SAW IT, not all or nothing -- a detector legitimately misses
                # a view, and requiring a box in every camera dropped the whole animal.
                # WHETHER A ONE-CAMERA WINDOW IS A *TRAINED* INPUT IS A PROPERTY OF THE RUN: check
                # its own `[data].prob_2d_only` before reading a single-view arm.
                use, boxes = [], []
                for i, ci in enumerate(cam_ix):
                    v = bb[:, i][np.isfinite(bb[:, i]).all(-1)]
                    if not len(v):
                        continue
                    # THE UNION EXTENT, NOT `crop_box_for_points`, and that is measured rather than
                    # lazy: re-squaring an already near-square union grows the p90 box AREA by 82%
                    # and costs both MPJPE and MOTA. A detector box IS a crop-rule box, so the
                    # union already satisfies the `min_crop_dim` floor.
                    #
                    # THE UNION IS PER CAMERA, over that camera's OWN finite frames, so two
                    # cameras' crops need not be contemporaneous. Left alone on measurement: the
                    # contributing spans overlap over ~92-100% of the window and only ~2% are
                    # disjoint. `box_agree` is what makes it visible where it happens.
                    #
                    # int32 and clamped into the image, exactly like the crop rule's own box: a
                    # float or off-frame box produces a negative cam['offset'] and breaks
                    # project_cam far downstream.
                    w, h = (int(x) for x in session.rig.size(session.cam_names[ci]))
                    if inst_boxes is not None:
                        # STORED BOXES ARE NOT DETECTOR BOXES, AND TRAINING PUTS THEM THROUGH
                        # THE RULE. `instances.pq` boxes are often non-square, and the loader
                        # routes them through `crop_box_for_points(..., pad=0)`, which squares the
                        # extent and applies the floor. Serving a tight unfloored box here would
                        # put the animal at a different scale than any crop it trained on. This is
                        # the SAME arithmetic as training, not a second rule; `pad=0` because the
                        # stored extent is already padded.
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
                    # THE ONE THING THE UNION-OF-BOXES CANNOT DO, and the comment above says so:
                    # "the per-frame extents that would have to be unioned BEFORE squaring are not
                    # recoverable from the boxes". Detector KEYPOINTS are exactly those extents, so
                    # with them the training crop rule can be applied once over the whole window --
                    # union the points, then square once -- instead of squaring per frame and
                    # unioning squares. That is why this is not the 08 §1.3 proposal that measured
                    # +3.06 mm worse: that one re-squared an already-square union.
                    if cfg.crop_source == 'keypoints' and det_kpts_stc is not None:
                        kk = det_kpts_stc[a, fl, ci][..., :2].reshape(-1, 2)
                        kk = kk[np.isfinite(kk).all(-1)]
                        if len(kk):
                            kb = cropmod.crop_box_for_points(
                                torch.as_tensor(kk, dtype=torch.float32),
                                torch.tensor([w, h], dtype=torch.float32), cfg.min_crop_dim)
                            # THE SAME BOUND `--refine` CARRIES, for the same reason: the rule
                            # squares the extent, so a wandering keypoint set lands somewhere else
                            # entirely. Falling back to the union is the conservative direction.
                            if kb is not None and _overlaps(kb, box):
                                box = kb.to(torch.int32)
                    boxes.append(box)
                    use.append(ci)
                if not use:
                    outcome[a, wl] = OUTCOMES.index('no camera')
                    continue
                # THE CAMERA MUST DESCRIBE THE BOX THE PIXELS WERE ACTUALLY CUT WITH. `apply_crop`
                # sets `size`, `_resize_camera` turns it into `scales`, and `forward` divides by
                # that -- a camera that described a different box would scale the decode by
                # union_side/moving_side (p50 1.23 on this root) and land every keypoint short of
                # where it belongs. It cost pck@10 0.841 -> 0.383 at unchanged coverage, which is
                # the signature: the rows are all there and every one of them is in the wrong
                # place.
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
            # WIDE-CROP DEPLOYMENT (report 27): inflate each crop about its centre before the
            # resize, so the target sits off-centre in a wider crop that includes neighbours --
            # the regime where the box prompt is load-bearing. Guarded: crop_inflate 1.0 leaves
            # `boxes`/`cgroup` untouched, so this is a no-op on every existing run. Static boxes
            # only; each box is (4,) per camera here.
            if cfg.crop_inflate != 1.0:
                boxes = [cropmod.inflate_box(b, window_cams[ci]['size'], cfg.crop_inflate)
                         for ci, b in zip(use, boxes)]
                cgroup = [cropmod.apply_crop(window_cams[ci], b) for ci, b in zip(use, boxes)]
            scales = []
            # PASS 1, which under `--refine-px` runs at a reduced resolution. The distinction
            # between the two passes lives HERE, at the site, rather than in `_resize_camera` --
            # that is a `dataset.py` helper shared with the loader and has no business knowing
            # about inference passes.
            uncropped = list(cgroup)      # pre-resize; only read by the refine fallback below
            for i, cam in enumerate(cgroup):
                cgroup[i], s = _resize_camera(cam, pass1_res)
                scales.append(s)
            # The box BEFORE the pixels, so a decode failure still shows what it was reaching for.
            for i, ci in enumerate(use):
                crop[a, wl, ci] = np.asarray(boxes[i], np.float32)
            outcome[a, wl] = OUTCOMES.index('decode failed')
            plans.append((a, use, boxes, cgroup, scales, uncropped))
        return frames, window_cams, plans

    # ONE DECODE PER (CAMERA, FRAME) PER WINDOW, shared by every animal in it.
    #
    # ponytail: peak memory is one camera's window of FULL frames plus every animal's crops --
    # 24 x 21 MB on johnson-mouse's 3208x2200 rig, which has one animal. A wide rig with many
    # animals would want the frame loop chunked; nothing shipped is both.
    # AND THE CAMERAS OVERLAP, up to `_CAM_DECODE` of them. This loop used to be serial across
    # cameras because `dataset._read_video` held ONE global lock, so a second thread would only
    # have queued on it; the lock is now per container, and different cameras are different
    # containers. Video is where it matters -- 3dpop decodes at ~44 ms per frame-camera against
    # a pose forward two orders of magnitude cheaper -- but an image directory gains too, since
    # `read_frames`'s own per-frame pool is per call and one camera at a time could not fill it.
    #
    # THE CAP IS MEMORY, NOT CORES. Each task holds one camera's whole window of FULL frames:
    # 24 x 21 MB on johnson-mouse's 3208x2200 16-camera rig, so unbounded concurrency there is
    # 8 GB of transient host memory for a rig that has one animal in it. 4 is decord's own
    # `ChunkShuffle.mix` number and bounds it at four windows.
    #
    # `crops` is written from those threads: the keys are distinct per (animal, camera) and a
    # dict store is atomic under the GIL, so the result does not depend on the order they land.
    def decode_crops(frames, plans):
        """`frames` is a PARAMETER (was a closure read of the enclosing loop's variable),
        so this can run for a future window on a background thread while the current
        window's forward runs on the main thread. Same body otherwise."""
        crops = {}
        cams = sorted({c for _, use, *_ in plans for c in use})

        def one(ci, pool):
            # SEVEN DECODES PER (FRAME, CAMERA) WAS THE SHIPPED DEFAULT, and this is where six of
            # them were paid. Windows step by `T - overlap`, so at the box recipe's `n_frames = 12`
            # and `--overlap 8` every frame sits in THREE windows; `--refine` (on by default in
            # 3D) then calls this function a SECOND time per window against the re-cropped plan,
            # with the identical `frames`; and `detect_raw` had already decoded the frame once on
            # its own pass over the group. `read_frames` dedupes within one call and nothing
            # remembered anything across calls.
            #
            # `store` is keyed by (camera, SOURCE frame index) and holds the FULL decoded frame,
            # which is what makes it serve every one of them: the refine pass wants the same
            # pixels under a different crop, and `_crop_views` never mutates what it is given
            # (it warps into a new array, or returns the frame and lets `np.asarray` copy).
            imgs = store.read(ci, session.cam_names[ci], frames, pool=pool)
            # A file that will not decode is a property of the file, not of the animal, so
            # it takes out every animal that wanted this camera -- which is what the
            # per-animal decode did too, one animal at a time.
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
                # A SECOND POOL: `one` runs in this one and waits on futures in `pool`, and a
                # pool that waits on itself deadlocks as soon as both are full.
                with ThreadPoolExecutor(max_workers=min(cam_decode, len(cams))) as cpool:
                    list(cpool.map(lambda ci: one(ci, pool), cams))
        return crops

    def forward(frames, plan, crops, wi):
        """One animal, one window -> its prediction in the SOURCE frame, or None.

        `frames` is a parameter for the same reason as `decode_crops`. `carried` is read
        here on the MAIN THREAD ONLY, in window order -- the one place the prior loop's
        state is read, so the window-order guarantee lives entirely in the caller."""
        a, use, boxes, cgroup, scales, *_ = plan
        # uint8; the model divides on device. Same contract as the training loader.
        views = [crops[a, ci] for ci in use]
        if any(v is None for v in views):
            return None                     # already marked 'decode failed' above
        prior, prompt_t = _build_prior(cfg, carried[a], src, a, n_lab, frames, boxes,
                                       scales, mode, K, R, cgroup)
        dev = cfg.device
        chunk = cfg.kpt_chunk or None
        views = [v.to(dev) for v in views]
        cgroup_d = _to_device(cgroup, dev)
        # ONE encode for both passes of `self`: the pixels are identical, and the encode is
        # the bulk of the forward. A no-op under `kpt_chunk` (`model._forward_window`).
        # DEPLOYMENT BOX PROMPT (report 27), 2D AND 3D, PER CAMERA. GUARDED: when
        # `box_prompt == 'none'` NO kwarg is passed, so the stock model (whose forward has no
        # box_prompt parameter) sees the identical call it always did -- byte-identical, pinned
        # by `tests/test_infer.py`. When on, the box names animal `a` per camera (its own source
        # points under 'labels', an ORACLE, or a detector's own per-camera box); the model must be
        # a `[model].box_prompt` (film/term) checkpoint. A camera with nothing finite that frame
        # gets a NaN column, which the encoder's `missing_film`/`missing_box` token substitutes --
        # the regime `box_prompt_dropout` trains, not a silent failure.
        mkw = {}
        if cfg.box_prompt != 'none':
            if cfg.box_prompt == 'labels':
                # ORACLE: the animal's GT keypoints name which animal to return.
                if src is None or a >= n_lab:
                    box_t = None
                else:
                    box_t = _deploy_box_prompt(mode, src, None, frames, a, use, boxes, scales,
                                              cgroup, dev)
            elif cfg.box_prompt == 'detector':
                # DEPLOYABLE: the detector's own box for this animal slot, per camera.
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
        p = out['coords_pred'][0].detach().cpu().numpy()          # (t,K,R)
        # WHAT THE NEXT WINDOW OPENS ON, and it is not always what this window reports.
        #
        # Under `gridresid_offset = "query"` the reported 3D output is
        # `query + R @ residual` with ONE anchor per keypoint for the whole window, so feeding
        # it back as the next window's query closes a loop with GAIN: whatever the prior was
        # wrong by is re-added, and only the residual carries new information. Measured on
        # johnson-mouse, where the run is consistently trained so this is not RC0: a drift of
        # p50 1.3-2.2 mm that changes by only 0.13-0.28 mm/frame (a smoothness ratio of 8-10),
        # profiled as a SAWTOOTH locked to the window boundary, costing 30% of the animal's
        # motion. The model has never seen a wrong prior either -- `dataset.py` offers GT
        # +- i.i.d. 2.5 px or nothing at all -- so nothing trained it to correct one.
        #
        # `3d_pred_triangulate` is the anchor-free estimate: re-derived from THIS window's
        # pixels every frame, supervised on every keypoint
        # (`coords_loss_triangulate_weight` 0.1, `..._reproj_weight` 2.0), and already
        # repaired for a degenerate solve by `model._query_anchored`. Carrying it leaves the
        # reported output untouched -- so published numbers stay comparable -- and breaks the
        # feedback path only.
        q = None
        if mode == '2d':
            # crop pixels -> source pixels: undo the resize, then the crop origin
            p = p / scales[0] + np.asarray(boxes[0][:2], np.float32)
            # THE SAME TENSOR, deliberately: at one camera there is nothing to triangulate and
            # the 2D grid head decodes ABSOLUTE pixel bins, so no anchor is being fed its own
            # output. Every 2D root is therefore bit-identical under either `carry_source`,
            # which is the free invariance check on this change.
            q = p
        elif cfg.carry_source == 'pred':
            q = p
        elif out.get('3d_pred_triangulate') is not None:
            q = out['3d_pred_triangulate'][0].detach().cpu().numpy()
        # else 3D SINGLE-VIEW: `_query_anchored` substitutes `_rays_fallback` there and
        # deliberately does not write the key back, and a conf-weighted ray mean is too weak a
        # position to seed the next window with. `carried[a]` is left alone, so the staleness
        # bound in `_build_prior` retires it -- which is the honest answer, not a silent one.
        #
        # THE PER-CAMERA 2D POSE, WHICH A 3D RUN USED TO THROW AWAY. `2d_pred` is
        # `(cams, b, t, K, 2)` in the INPUT PIXELS of that camera's crop canvas
        # (`tracker_encoder`: `points_pred * px_scale`), so the inverse is this plan's own crop --
        # undo the resize, add the crop origin -- per camera.
        #
        # IT IS THE SAME EXPRESSION `p` TAKES IN 2D ABOVE, and deliberately not the more exact
        # per-axis inverse in `dataset._crop_affine`. In 2D `coords_pred` IS `2d_pred[0]`, so
        # using one expression makes camera 0's per-camera pose bit-identical to the reported
        # prediction -- a free exact check, pinned as a test. The sub-pixel difference between the
        # two comes from `_resize_camera` rounding `cam['size']`; changing it here would move
        # every published 2D number.
        p2 = v2 = None
        if '2d_pred' in out:
            p2 = out['2d_pred'][:, 0].detach().cpu().numpy().copy()   # (C_use, t, K, 2)
            for i in range(len(use)):
                p2[i] = p2[i] / scales[i] + np.asarray(boxes[i][:2], np.float32)
            if out.get('vis_pred_2d') is not None:
                v2 = out['vis_pred_2d'][:, 0].detach().cpu().numpy()  # (C_use, t, K) logits
        return p, q, out, p2, v2

    def _process_window(wi, frames, window_cams, plans, crops):
        """Forward and write every column for one window, given ALREADY-DECODED `crops`.

        `crops` is a parameter, not recomputed here -- this is the whole point of prefetching:
        `_prepare` (below) does `_build_plans` + `decode_crops` on a background thread, and this
        function must consume that result rather than paying for a second, synchronous decode of
        the same window. Runs on the MAIN THREAD, in window order -- `carried` is read and
        written only here, so the prompt sequence (and every published number) is exactly what
        it was before window-level decode-ahead existed.
        """
        if cfg.refine:
            # CROP REFINEMENT, label-free. The first pass's own prediction re-enters the crop rule
            # as if it were the labels, so the second pass sees the box a GT crop would have given
            # -- which is the only arm that beat every detector crop on 3dpop.
            #
            # This is the third answer to what item 17 measured: shortening the window bought MOTA
            # +0.130 purely by shrinking the crop union, and paid pck@10 0.103 -> 0.067 in lost
            # temporal context. Refining shrinks the union at FULL T, so it should buy the first
            # without the second. It costs one extra forward AND one extra decode per animal per
            # window -- the crop moved, so neither the pixels nor `share_scene` can be reused.
            #
            # An animal whose refined crop fails keeps its first-pass BOX rather than being
            # dropped: a bad prediction must not cost coverage a loose box already had.
            #
            # ...but it must NOT keep the first-pass CAMERA under `--refine-px`, which is what
            # re-appending `plan` did. That camera is at `pass1_res`, so the fallback silently ran
            # the SECOND pass at the first pass's reduced resolution -- and it is the fallback, i.e.
            # exactly the animal whose first pass already went wrong. `_at_image_size` rebuilds it.
            def _at_image_size(plan):
                a, use, boxes, _, _, uncropped = plan
                if pass1_res == cfg.image_size:
                    return plan                          # bit-identical: nothing to rebuild
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
                # A REFINED BOX THAT DOES NOT OVERLAP THE BOX IT CAME FROM IS NOT A REFINEMENT.
                # `boxes_from_points` runs the first pass's own prediction through
                # `crop_box_for_points`, which SQUARES the extent -- so a pose that wandered (RC1's
                # drift, or a crop that covered two animals) produces a box somewhere else
                # entirely, at 2x the pose compute, and the giant squares in `rat-city_best.npz`
                # are exactly that. Nothing bounded it. Requiring overlap with the first-pass box
                # is the weakest test that catches "somewhere else", and it costs nothing.
                if any(not _overlaps(b2[i], boxes[i]) for i in range(len(use))):
                    refined.append(_at_image_size(plan))
                    continue
                uncropped2, sc2 = list(cg2), []
                for i, cam in enumerate(cg2):
                    cg2[i], s = _resize_camera(cam, cfg.image_size)   # PASS 2, always full res
                    sc2.append(s)
                # `crop` KEEPS THE FIRST-PASS BOX. Overwriting it lost the only record of what the
                # detector actually offered, which is the box every coverage and crop-inflation
                # number in reports 08 and 11 is computed from.
                for i, ci in enumerate(use):
                    crop_refined[a, wi - w0, ci] = np.asarray(b2[i], np.float32)
                refined.append((a, use, b2, cg2, sc2, uncropped2))
            plans = refined
            crops = decode_crops(frames, plans)

        for plan in plans:
            a, use, boxes, cgroup, scales, *_ = plan
            got = forward(frames, plan, crops, wi)
            if got is None:
                continue                        # already marked 'decode failed' above
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
            # The pose the NEXT window opens on. Clamped from the front: a group shorter than
            # `overlap` gives a window with fewer frames than the step, and a plain negative
            # index runs off the start of it.
            j = max(0, len(frames) - cfg.overlap) if cfg.overlap else len(frames) - 1
            # THE ROW GATE, applied to what is REPORTED and never to what is carried -- `carried`
            # reads `p`, which is untouched below. Two levers in one flag would be one lever too
            # many (eval rule 4), and a gate that also blinded the next window's prompt would be
            # measuring the prompt.
            if cfg.vis_thresh is not None and vlogit is not None:
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore', RuntimeWarning)       # an all-NaN row is legal
                    # The MEDIAN over keypoints: a mean lets one confident keypoint carry a row the
                    # model otherwise declined.
                    med = np.nanmedian(vlogit, axis=-1)
                # AN UNSCORABLE ROW IS NOT A PASSING ROW. `NaN < thresh` is False, so a row whose
                # every keypoint confidence was NaN used to sail through the gate -- the one row the
                # model said least about was the one the gate could not touch.
                drop = ~(med >= cfg.vis_thresh)
                # APPLIED TO `p` BEFORE IT IS RECORDED, not to `pred` after: a
                # gated frame must be left OUT of the mean rather than blanked once and then averaged
                # back in by the next window. `q` is untouched, so the carried prompt is unaffected --
                # two levers in one flag would be one too many (eval rule 4).
                p = p.copy()
                p[drop] = np.nan
                # Under blend the gated frames were already accumulated above, so they are removed
                # from the accumulator rather than from the finished array -- same rule as `p`:
                conf[a, frames[drop] - f0] = np.nan
                box_agree[a, frames[drop] - f0] = np.nan
            pred[a, frames - f0] = p
            if q is not None:
                carried[a] = (torch.as_tensor(q[j]), int(frames[j]),
                              None if vlogit is None else torch.as_tensor(vlogit[j]))

    # THE DRIVER: decode window wi+1 while window wi is forwarded on the GPU, bounded by
    # `cfg.prefetch_windows` (default 1; dev/reports/31). BIT-EXACT: `_build_plans` and
    # `decode_crops` never read or write `carried`, and `_process_window` still runs
    # strictly in window order on the main thread -- so the SEQUENCE of forwards and the
    # SEQUENCE of `carried` updates are byte-identical to the serial loop; only the
    # wall-clock OVERLAP with the next window's decode is new. `tests/test_infer.py` pins
    # this against `prefetch_windows = 0`, which is the exact old code path unchanged --
    # `_prefetch_pool` is never even created there.
    #
    # A prefetched window holds one extra window's worth of `crops` (already cropped to
    # `image_size`, not full source frames -- `decode_crops` never keeps the full decode
    # past its own call), so the memory cost of `prefetch_windows = 1` is one more
    # small-buffer dict, not a second copy of `_CAM_DECODE`'s full-frame budget.
    n_ahead = max(0, int(cfg.prefetch_windows))
    _prefetch_pool = ThreadPoolExecutor(max_workers=1) if n_ahead else None

    def _prepare(wi, start):
        frames, window_cams, plans = _build_plans(wi, start)
        crops = decode_crops(frames, plans)
        return frames, window_cams, plans, crops

    try:
        for w0, w1 in blocks:
            # THE BLOCK'S FRAMES. `f_read` is what its windows TOUCH; `f_own` is what it KEEPS.
            # They differ by the seam: the last window reaches past `starts[w1]`, and those frames
            # belong to the next block because a frame belongs to the last window containing it
            # (eval rule 11). The whole-clip loop overwrote them for the same reason; here they
            # are simply not emitted, which is what makes the two identical.
            f0 = int(starts[w0])
            f_read = int(min(T_total, starts[w1 - 1] + cfg.n_frames))
            f_own = int(starts[w1]) if w1 < len(starts) else T_total
            n_blk, n_win = f_read - f0, w1 - w0

            boxes_stc = det_kpts_stc = None
            if boxes_for is not None:
                boxes_stc, _scores, det_kpts_stc = boxes_for(store, f0, f_read)

            pred = np.full((S, n_blk, K, R), np.nan, np.float32)
            conf = np.full((S, n_blk, K), np.nan, np.float32)
            # THE PER-CAMERA 2D POSE AND ITS OWN VISIBILITY, which a 3D run discarded entirely --
            # only the triangulated world point ever reached the output. `(S,t,C,K,2)` and
            # `(S,t,C,K)`; see `forward` for the crop inverse.
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
                # CLAMPED TO THE BLOCK: a window of the next block would want boxes this block
                # did not fetch. Prefetch only ever reduces wall clock, never changes a pixel, so
                # clamping it costs one stall per boundary and nothing else.
                nxt = wi + n_ahead
                if _prefetch_pool is not None and nxt < w1 and nxt not in pending:
                    pending[nxt] = _prefetch_pool.submit(_prepare, nxt, starts[nxt])
                _process_window(wi, frames, window_cams, plans, crops)
                # EVICT BY WINDOW POSITION, NOT BY AN LRU, because the access pattern is known
                # exactly: `starts` is monotone increasing and window `wi` reads `[start, start+T)`,
                # so a frame below the OLDEST still-live window's start can never be asked for
                # again. Exact rather than heuristic, and it bounds the store at one window plus
                # the prefetch depth regardless of clip length.
                if wi + 1 < w1:
                    store.evict_below(int(starts[max(w0, wi - n_ahead + 1)]))
                # AT THE WINDOW BOUNDARY, where a window's worth of full frames has just been
                # dropped. Without it the freed blocks stay in glibc's arena and RSS ratchets to
                # whatever the host allows -- 123 GB on a run whose live buffers were budgeted at
                # 8 (see `memory.trim`). Microseconds against a window of decode.
                memory.trim()
            # THE SEAM FRAMES STAY, everything before them goes: the next block's first window
            # opens on `f_own` and would otherwise decode them a second time.
            store.evict_below(f_own)
            memory.trim()
            keep = f_own - f0
            yield {'pred': pred[:, :keep], 'conf': conf[:, :keep],
                   'pred2d': pred2d[:, :keep], 'conf2d': conf2d[:, :keep],
                   'box_agree': box_agree[:, :keep],
                   'animal_ids': np.asarray(animal_ids, object),
                   'outcome': outcome, 'crop': crop,
                   'window_start': np.asarray(starts[w0:w1]),
                   'outcome_names': np.asarray(OUTCOMES, object),
                   'mode': mode, 'group_id': gid, 'session': session.session_id,
                   'dataset': dataset_name, 'anchor': cfg.anchor,
                   'carry_source': cfg.carry_source, 'n_frames': T_total,
                   # THE RESOLVED VALUE, not the tri-state the caller passed. `refine` defaults by
                   # dimensionality, so the file has to say which way it went -- otherwise a 3D run
                   # and a `--no-refine` 3D run are indistinguishable in their own provenance.
                   'refine': bool(cfg.refine), 'refine_px': cfg.refine_px or 0,
                   **({} if crop_refined is None else {'crop_refined': crop_refined}),
                   **({} if box_cams is None else {'box_prompt_cams': box_cams}),
                   'boxes_from': _boxes_from}
    finally:
        store.clear()
        if _prefetch_pool is not None:
            _prefetch_pool.shutdown(wait=True)


def _overlaps(a, b):
    """Do two xyxy boxes share any area? Either being non-finite is not an overlap."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    return bool(min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))


def _fill_box_agreement(box_agree, a, frames, use, boxes, p, mode, window_cams):
    """Distance from the predicted centroid to each crop box's centre, in units of one box side.

    A 3D pose is REPROJECTED through the source camera to be compared with a source-pixel box --
    per frame, so a moving rig is handled by `project_points_torch`'s own (T,4,4) alignment rather
    than by picking one extrinsic. The box is the window's union box, which is what the pixels were
    actually cut with, so a value near 0 means the pose sits where the crop said the animal is.
    """
    for i, ci in enumerate(use):
        b = np.asarray(boxes[i], np.float64)
        side = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
        if not (np.isfinite(b).all() and side > 0):
            continue
        if mode == '2d':
            q = np.asarray(p, np.float64)                      # already source pixels
        else:
            q = project_points_torch([window_cams[ci]],
                                     torch.as_tensor(p, dtype=torch.float32))[0].numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)    # a frame with no finite keypoint
            c = np.nanmean(q, axis=-2)                         # (t,2)
        centre = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        box_agree[a, frames, ci] = np.linalg.norm(c - centre, axis=-1) / side


ORACLE_CORRUPTIONS = ('off', 'stale', 'other', 'near', 'swap')


def _prior_to_model_frame(p, mode, boxes, scales):
    """The one conversion `_build_prior` applies before its bounds mask: source pixels/world to
    the model's own crop frame. 3D passes straight through -- world points are never crop-scaled,
    only cameras are. Factored out so `_corrupt_prior`'s `near` can apply the IDENTICAL conversion
    to a candidate neighbour before testing it against `prior_out_of_bounds`, instead of a second
    copy of this one line.
    """
    if mode == '2d':
        return (p - torch.as_tensor(np.asarray(boxes[0][:2], np.float32))) * scales[0]
    return p


def _nearest_eligible_row(src, a, t0, boxes, scales, mode, cgroup):
    """The nearest OTHER label row whose pose at this frame would survive `prior_out_of_bounds` --
    the same eligibility test `dataset.py`'s `prompt_swap_animal` applies (dev/plans/
    prompt_prior_corruptions.md). Returns a row index, or None if no other row qualifies.

    "Nearest" is measured in the MODEL's own frame (after `_prior_to_model_frame`), over whichever
    keypoints both the target and the candidate have finite AND in bounds -- the same frame the
    model itself will see the corrupted prior in, rather than source pixels or world mm, which
    would rank rows by a distance the model never has to close.
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
    """`swap:<n>` -- n transpositions among `p`'s own finite keypoints, seeded like every other
    oracle corruption. The direct inference probe for `dataset.py`'s `prompt_swap_kpt_pairs`: a
    decode that swapped two keypoints, not one that latched onto the wrong animal.
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

    THIS MEASURES alpha = d(output)/d(prior), the echo coefficient, and it is the cheapest thing in
    the whole programme: no training, one afternoon, and it decides whether the prompt needs
    retraining at all. Near 0 means the model corrects a bad prior from the pixels; near 1 means it
    echoes it, and a loop that feeds the model its own output then has gain.

    Five corruptions now, and NONE of them is in training except through `dataset.py`'s own
    `prompt_swap_kpt_pairs` / `prompt_swap_animal` (dev/plans/prompt_prior_corruptions.md), which
    `swap` and `near` are the direct probes for:

    - `off:<x>`   a WHOLE-BODY offset of x crop widths, one direction for the whole pose. This is
                  the shape of a lag: every keypoint wrong the same way, which i.i.d. noise never is.
    - `stale:<n>` the pose from n frames earlier, which is what `carried` degrades into when the box
                  source loses an animal for a few windows.
    - `other`     the NEIGHBOURING animal's pose (`a + 1 % n_lab`), which is what a row swap hands
                  the next window. KEPT AS-IS -- numbers on record used this exact row rule, and
                  redefining it would make them silently incomparable.
    - `near`      the NEAREST ELIGIBLE animal's pose instead of the fixed `a + 1` row -- the row a
                  real identity mix-up is most likely to hand back.
    - `swap:<n>`  n transpositions of THIS row's own keypoints -- a decode that swapped two
                  keypoints rather than one that picked up a different animal.

    The magnitude is in CROP WIDTHS, not pixels or millimetres, for the same reason
    `prompt_noise_px` is in pixels and converts: `allen-mouse-combined` alone holds 63 px sessions
    beside 14 mm ones. In 3D the conversion is `get_camera_scale` -- world units per crop pixel,
    the same one `dataset.py` uses -- times `image_size`, i.e. one crop across.

    The direction is drawn from a generator seeded on (row, window), so two arms over one clip
    corrupt identically and the comparison is matched rather than merely similar.
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
    # ONE crop width, converted into whatever units the prior lives in.
    if mode == '2d':
        b = np.asarray(boxes[0], np.float64)
        width = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
    else:
        from posetail.posetail.cube import get_camera_scale
        fin = p[torch.isfinite(p).all(-1)]
        if not len(fin):
            return p, 0
        # Offset-invariant (a Jacobian) and `fin[None]` carries no time axis; posetail 0.3.5
        # collapses a per-frame offset inside get_camera_scale itself.
        scale = torch.nanmedian(get_camera_scale(cgroup, fin[None]))
        if not torch.isfinite(scale):
            return p, 0
        # THE CAMERA'S OWN WIDTH, not `cfg.image_size`. `scale` is world-per-pixel of the camera
        # in `cgroup`, which under `--refine-px` is the reduced pass-1 camera -- so pairing it with
        # the baked 256 made the injected offset `image_size/refine_px` times too big. More
        # correct today too, for a crop clamped non-square against a frame edge.
        width = float(scale) * int(cgroup[0]['size'].max())
    rng = np.random.default_rng([a, int(frames[0])])
    v = rng.normal(size=p.shape[-1])
    v = v / max(float(np.linalg.norm(v)), 1e-9)
    return p + torch.as_tensor(float(amt) * width * v, dtype=p.dtype), 0


def _build_prior(cfg, carried, src, a, n_lab, frames, boxes, scales, mode, K, R, cgroup):
    """The per-keypoint prior for this window, in the model's coordinate frame.

    Two things the prompt has to get right, both of which were wrong here and both silent:

    THE PROMPT FRAME IS NOT ALWAYS 0. `carried[1]` already holds the frame the carried pose
    describes. Hardcoding 0 is correct for interior windows and wrong on the LAST window of every
    group, which `_window_starts` pulls back to `n_frames - T` so it overlaps its predecessor by
    more than `overlap` -- the carried pose then describes a frame in the middle of the window and
    the model samples its patch from the wrong one.

    A PRIOR OUTSIDE THE CROP IS NOT A PRIOR. A carried keypoint that left the new box was being
    handed in as confident. Masking it was worth MOTA +0.041, miss -0.032 SIG and idsw 24 -> 13 on
    rat-city. NaN is the right value: it is exactly what the no-query tokens key off, so a
    departed keypoint degrades to "I was not told" instead of "I was told a lie".
    """
    if cfg.anchor in ('none', 'self'):
        return None, None
    if cfg.anchor == 'labels':
        # ORACLE. Ground truth as the prior; not a deployment number.
        if src is None or a >= n_lab:             # a detector row with no label row behind it
            return None, None
        p, qt = _corrupt_prior(cfg, src, a, n_lab, frames, boxes, mode, cgroup, scales)
        if p is None:
            return None, None
    else:                                    # 'carry'
        if carried is None:
            return None, None
        p = carried[0].clone().float()
        qt = int(carried[1]) - int(frames[0])
        # A STALE PRIOR IS NOT A PRIOR. `carried` is only written on a window that predicted, so an
        # animal the box source lost for a few windows keeps handing back a pose from before this
        # one -- and `qt` clamps to 0 below, presenting it as this window's first frame. Within the
        # overlap that is the ordinary case (the carried frame IS in the previous window's tail);
        # beyond it the pose describes a frame the model was never shown, and in 3D the bounds mask
        # cannot catch it, because a pose that is stale but still visible to two cameras passes.
        # `qt < 0`, NOT `-qt > overlap`. The invariant is exact: `j = len(frames) - overlap` makes
        # the carried frame the NEXT window's start, so consecutive windows give qt == 0 and the
        # last window (pulled back by `_window_starts`) gives qt > 0. A negative qt therefore
        # happens if and only if a window was SKIPPED, in multiples of `n_frames - overlap`.
        # Against `overlap` that test only fired when `n_frames > 2 * overlap` -- so at the swept
        # `--n-frames 24 --overlap 12` it never fired at all, and a prior from before an animal was
        # lost was presented as this window's first frame. There is no budget to spend: either the
        # carried frame is inside this window or it predates it.
        if qt < 0:
            return None, None
    if p.shape != (K, R):
        return None, None
    p = _prior_to_model_frame(p, mode, boxes, scales)
    p = p.clone()
    p[prior_out_of_bounds(p, mode, cgroup)] = float('nan')
    qt = min(max(qt, 0), len(frames) - 1)
    return p[None], torch.full((1, K), qt, dtype=torch.int32)
