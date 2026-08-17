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
from dataclasses import dataclass

import numpy as np
import torch

from posetail.posetail.cube import is_point_visible, project_points_torch

from . import crop as cropmod
from .dataset import _crop_affine, _resize_camera, read_frames
from .format import VISIBLE, Labels, Session
from .model import share_scene

ANCHORS = ('none', 'carry', 'self', 'labels')
CARRY_SOURCES = ('triangulate', 'pred')

# WHY AN (ANIMAL, WINDOW) PRODUCED NOTHING. Five separate aborts in the loop below wrote the same
# NaN, so a coverage number could not be decomposed at all -- "the detector offered no box", "the
# association matched no camera", "the crop rule refused" and "the file would not decode" are four
# different problems with four different fixes, and they arrived indistinguishable.
OUTCOMES = ('ok', 'no box', 'no camera', 'no points', 'crop failed', 'decode failed')

# How many cameras `decode_crops` may decode at once. A memory bound, not a core count -- see there.
_CAM_DECODE = 4


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
    # `vis_pred` logit across keypoints is below it. Measured against a rate-matched random
    # rejection, which is the only honest control (any rejection improves a mean over matched
    # points): 3dpop MOTA +0.049 [+0.011, +0.110] SIG at 7.3% of rows, where the control reads
    # +0.001 [-0.017, +0.028]; rat-city 0.601 -> 0.628 at 14%, control 0.493. THE THRESHOLD IS NOT
    # PORTABLE -- rat-city's logits have median +2.7 and 3dpop's +15.4 -- so there is no default and
    # a shipped one would have to be a quantile. See dev/reports/11 §3 item 20.
    vis_thresh: float | None = None
    # Re-crop each window to the FIRST PASS's own prediction and predict again. Label-free, and it
    # costs one extra forward AND one extra decode per animal per window (the crop moves, so no
    # pixels and no scene encode can be shared). See `run_group`.
    refine: bool = False
    # PASS 1'S INPUT RESOLUTION under `--refine`. None -> `image_size`, i.e. today's behaviour
    # exactly. Refine's gain is MAGNIFICATION, not coordinate frame (an ablation that re-centred
    # the crop at a fixed side recovered 42% of the mean gain, 3% of the median and 0.4% of
    # pck@10), so pass 1 only has to LOCALISE and does not need full resolution. calms21 2D: 96 px
    # beats full-res refine outright, 6.651 against 6.765, at a third of the overhead. 3dpop 3D:
    # 192 is a NULL against 256 (+0.062 mm paired) at a quarter of the pixels, and 96-128 trade
    # ~1.7 mm for +0.021 coverage and +0.03 MOTA. 64 is the cliff on BOTH. No shipped default --
    # the floor is patch-size- and root-dependent, which is the `--vis-thresh` lesson.
    # `model.PoseTrackerEncoder.forward` is what makes a smaller input correct; see `_input_extent`.
    refine_px: int | None = None
    # WHERE THE WINDOW'S CROP COMES FROM. 'boxes' unions the detector's per-frame boxes, which is
    # what every recorded number uses. 'keypoints' runs THE CROP RULE on the detector's own
    # keypoints over the window -- see the long comment at the union below for why that is the one
    # thing boxes cannot reproduce. Needs a keypoint-trained detector; ignored without one.
    crop_source: str = 'boxes'
    # How many finite (frame, camera) boxes a row needs before it gets a window crop at all. 1 is
    # what the loop always did, and it is the reason coverage can be FABRICATED: `3dpop_nogate.npz`
    # reports 0.000 of (row, frame) missing a pose while its box cache has 2.1-2.2% of (row, frame)
    # with no camera at all -- one box out of T x C positions a crop for all 24 frames and every one
    # of them is marked `ok`. Raising it LOWERS reported coverage, and that is the point.
    min_box_frames: int = 1
    # WHAT `carry` FEEDS BACK. See `run_group`'s carry note.
    #   'triangulate' -- the ANCHOR-FREE estimate (`3d_pred_triangulate`). Breaks the loop.
    #   'pred'        -- the reported prediction, which under gridresid_offset = "query" IS
    #                    `prior + residual` and so integrates its own error window over window.
    # 2D is identical either way: there is no triangulation at one camera and `coords_pred` is an
    # absolute pixel decode, so nothing is being fed its own anchor.
    carry_source: str = 'triangulate'
    # DELIBERATELY BREAK THE ORACLE PRIOR, to measure how far the output follows it. `--anchor
    # labels` + this is the only cheap way to get the echo coefficient alpha = d(output)/d(prior)
    # without a training run, and alpha is what decides whether the prior needs retraining at all.
    # `off:<x>` | `stale:<n>` | `other`. See `_corrupt_prior`. Never a deployment arm.
    oracle_corrupt: str | None = None
    device: str = 'cuda:0'
    # Read from the RUN's own `[data]`, like `min_crop_dim` -- never from a CLI flag. A model
    # trained on `instances` crops and evaluated on keypoint crops is being scored against a crop
    # rule it never saw, and there would be nothing in the output to say so.
    box_source: str = 'keypoints'


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

    `box` is either ONE `[x1,y1,x2,y2]` for the whole window, or a (T,4) of per-frame boxes under
    `--moving-crop`. The one-box path is kept as a single affine computed once, so the flag off is
    the same arithmetic it always was.
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




def prior_out_of_bounds(p, mode, cgroup):
    """Which keypoints of a prior are NOT usable as one. (K,) bool, in the MODEL's own frame.

    A PRIOR OUTSIDE THE CROP IS NOT A PRIOR. In 2D that is a point outside the crop rectangle; in
    3D it is a point no PAIR of cameras can see, since a point one camera sees is not
    reconstructible and so cannot be a position the model should trust. A MOVING camera answers per
    frame ((T,K) rather than (K,)) and the prior is one pose for the whole window, so a camera
    counts if it saw the point at any point during it.

    ONE COPY OF THE RULE, called from both prompted regimes. `carry` had it and `self` did not, so
    the two label-free regimes disagreed about what counts as a prior -- and `self` is the one the
    periodic val eval reports, so training and deployment were being scored under different rules.
    Masking this was worth MOTA +0.041, miss -0.032 SIG and idsw 24 -> 13 on rat-city under `carry`.
    """
    if mode == '2d':
        w, h = (float(x) for x in cgroup[0]['size'][:2])
        return (p[:, 0] < 0) | (p[:, 0] >= w) | (p[:, 1] < 0) | (p[:, 1] >= h)
    seen = []
    for c in cgroup:
        v = is_point_visible(c, p, margin=2)
        seen.append(v.any(0) if v.ndim > 1 else v)
    return torch.stack(seen).sum(0) < 2


def self_prompt(model, views, kpt_ids, cgroup, mode, first, kpt_chunk=None):
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
    return model(views, kpt_ids, cgroup, mode=mode, kpt_prior=prior, prompt_time=qt,
                 kpt_chunk=kpt_chunk)


@torch.no_grad()
def run_group(model, session: Session, gid: str, registry, dataset_name: str,
              cfg: InferConfig, box_points=None, boxes_stc=None, det_kpts_stc=None) -> dict:
    """Predict every animal in one group. Returns arrays in the SOURCE coordinate frame.

    Crops come from exactly one of two sources, and they are NOT comparable:

    - `box_points` (S,T,K,R): points the crop rule follows, shaped like the labels. Passing the
      labels themselves is the GT-crop upper bound.
    - `boxes_stc` (S,T,C,4): boxes given directly, from a detector or a detections file. This is
      the deployment number.

    Whichever was used is recorded in the result so a caller cannot quote one as the other.
    """
    assert cfg.anchor in ANCHORS, f'anchor must be one of {ANCHORS}'
    assert cfg.carry_source in CARRY_SOURCES, \
        f'carry_source must be one of {CARRY_SOURCES}, got {cfg.carry_source!r}'
    if cfg.anchor in ('carry', 'self') and cfg.overlap < 1:
        raise ValueError(f'anchor={cfg.anchor!r} carries a pose across windows and needs '
                         'overlap >= 1; got 0')
    # A reduced pass-1 resolution is only a thing when there IS a second pass. Without `--refine`,
    # pass 1 is the only pass and its output is the answer.
    pass1_res = cfg.refine_px if (cfg.refine and cfg.refine_px) else cfg.image_size

    group = session.groups[gid]
    lab: Labels = session.labels(gid)
    mode = session.mode
    K = session.n_keypoints
    R = 3 if mode == '3d' else 2
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
    inst_boxes = (lab.boxes if (boxes_stc is None and box_points is None
                                and cfg.box_source == 'instances' and lab.boxes is not None
                                and bool(np.isfinite(lab.boxes).any())) else None)
    n_src = boxes_stc.shape[0] if boxes_stc is not None else src.shape[0]
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
    animal_ids = ([f'det{a:02d}' for a in range(S)] if boxes_stc is not None else
                  [lab.animal_ids[a] if a < len(lab.animal_ids) else f'det{a:02d}'
                   for a in range(S)])

    pred = np.full((S, T_total, K, R), np.nan, np.float32)
    conf = np.full((S, T_total, K), np.nan, np.float32)
    # THE ANCHOR-FREE ESTIMATE, kept beside the prediction in 3D. It is what `carry` now feeds back,
    # every fix below rests on it, and nothing had ever scored it against the labels -- so it goes in
    # the npz where `eval.py` can. Independent of `carry_source`, deliberately: it is a property of
    # the forward, not of what the loop chose to do with it.
    pred_tri = np.full((S, T_total, K, R), np.nan, np.float32) if mode == '3d' else None
    tri_bad = np.zeros((S, T_total, K), bool) if mode == '3d' else None
    carried = [None] * S                      # per-animal prior for the next window
    # THE DIAGNOSTICS, per (animal, window): why it produced nothing, and what box it was given.
    # Both are what makes a coverage delta readable -- 08's crop-inflation measurement needed the
    # box, and every one of the five aborts below needed to be distinguishable from the others.
    starts = _window_starts(T_total, cfg.n_frames, cfg.overlap)
    outcome = np.full((S, len(starts)), OUTCOMES.index('no box'), np.int8)
    crop = np.full((S, len(starts), len(session.rig), 4), np.nan, np.float32)
    # DOES THE POSE AGREE WITH THE BOX IT WAS CROPPED FROM? Per (animal, frame, camera), the distance
    # from the predicted centroid to that camera's crop-box centre, in units of one box side.
    #
    # The pipeline holds TWO independent statements about where an animal is -- the box and the pose
    # -- and nothing ever compared them. Every artifact worth fixing shows up here: a box that
    # teleported onto another animal, a prompt loop whose pose lags the box it is being cropped by, a
    # union crop covering the arena. And it is in ANIMAL-SIZE units, so unlike `vis_pred`'s logit
    # (medians +2.7 / +4.0 / +15.4 across three roots, and a sign that flips per dataset) one value
    # means the same thing everywhere.
    box_agree = np.full((S, T_total, len(session.rig)), np.nan, np.float32)
    # (S,T,C,K) pose-against-detector-keypoints, only when a keypoint-trained detector supplied
    # them. Unlike `box_agree` this is per keypoint and structurally UNBOUNDED -- see
    # `_fill_kpt_agreement`.
    kpt_agree = (None if det_kpts_stc is None else
                 np.full((S, T_total, len(session.rig), det_kpts_stc.shape[3]),
                         np.nan, np.float32))
    crop_refined = (np.full_like(crop, np.nan) if cfg.refine else None)

    # THE COMPANION COLUMNS BLEND TOO, and they used not to. `pred` became the nan-aware mean of
    # every window that decoded a frame while `conf`, `box_agree`, `kpt_agree` and `pred_tri`
    # stayed plain assignments -- so they described the LAST window's decode of a frame whose
    # reported pose is an average of several. `--vis-thresh` made it worse: it NaNs `conf` and
    # `box_agree` for frames it dropped in this window while the blend still carries an earlier,
    # ungated window's contribution to `pred`, so a frame could have a finite blended pose and a
    # NaN confidence. `eval.py` reads `box_agree` per group and quotes `--vis-thresh` off `conf`,
    # so the two blend arms were reporting mismatched columns.
    #
    for wi, start in enumerate(starts):
        frames = np.arange(start, min(start + cfg.n_frames, T_total))
        if len(frames) < 2:                   # T=1 hits posetail's gT = T // tubelet = 0 bug
            frames = np.clip(np.arange(start, start + 2), 0, T_total - 1)
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
                bb = boxes_stc[a][frames]                          # (t, C, 4)
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
                # USE THE CAMERAS THAT SAW IT, not all or nothing. A detector legitimately
                # misses a view -- cross-view association leaves an unmatched camera NaN -- and
                # requiring a box in every camera dropped the whole animal for the window even
                # when two views had it, which is pure lost coverage -- and coverage is a number
                # this repo reports.
                #
                # WHETHER A SUBSET IS A *TRAINED* INPUT IS A PROPERTY OF THE RUN. `cams_to_sample`
                # picks camera subsets and `prob_2d_only` trains the one-camera case, and both are
                # per-run: the `3dpop-*` and `rat-city-*` runs set `prob_2d_only = 0.25`, while
                # `configs/w9.toml` ships 0 ("golden spent 0% of its steps on this path"). Under a
                # config like w9's, a one-camera window is an untrained input shape rather than a
                # supported one. Predicting it beats dropping it either way, but do not read a
                # single-view arm without checking the run's own `[data].prob_2d_only`.
                use, boxes = [], []
                for i, ci in enumerate(cam_ix):
                    v = bb[:, i][np.isfinite(bb[:, i]).all(-1)]
                    if not len(v):
                        continue
                    # THE UNION EXTENT, NOT `crop_box_for_points`, and that is measured rather than
                    # lazy. 08 §1.3 asks for the crop rule here, on the grounds that the deployment
                    # path must use the rule the model was trained on (gotcha 8). Running it costs
                    # 3dpop +3.06 mm [+1.86, +4.41] MPJPE and -0.032 MOTA, both SIG over 58 groups,
                    # and rat-city -0.040 MOTA. The reason is in the crop field of those two npz
                    # files: the union of per-frame crop-rule boxes is ALREADY near-square (aspect
                    # median 1.047), and squaring it again grows the p90 box AREA by 82% -- an
                    # elongated union, which is an animal crossing the window, becomes a large
                    # square and the resize to 256 px shrinks the animal inside it.
                    #
                    # The premise is also weaker than it reads: a detector box IS a crop-rule box,
                    # so it already satisfies the `min_crop_dim` floor, and a union of boxes that
                    # each satisfy the floor satisfies it too. And the rule cannot be reproduced
                    # exactly from boxes in any case -- `pad = 0` would fix double-padding, but the
                    # per-frame extents that would have to be unioned BEFORE squaring are not
                    # recoverable from the boxes. See dev/reports/11 §3 item 16.
                    #
                    # THE UNION IS PER CAMERA, over that camera's OWN finite frames, so camera A's
                    # crop can be positioned by frame 0 and camera B's by frame 23 -- the model then
                    # triangulates across crops that are not contemporaneous. That is left alone on
                    # measurement (`scratch/phase4/union_spans.py`, over every 3dpop box cache): of
                    # 480 multi-camera animal-windows, the intersection of the contributing cameras'
                    # spans has median 0.92-1.00 of the window, 12-15% fall below half, and only
                    # 6-12 (1.3-2.5%) are DISJOINT with nothing contemporaneous at all. Both
                    # alternatives -- requiring the spans to overlap, or unioning only over frames
                    # two cameras share -- cost coverage on the other 97.5% to fix that, and a rule
                    # tuned on 9 windows is a rule tuned on noise. `box_agree` is what makes it
                    # visible where it happens: a crop cut from the wrong time is a camera whose box
                    # the reprojected pose does not sit on.
                    #
                    # int32 and clamped into the image, exactly like the crop rule's own box: a
                    # float or off-frame box produces a negative cam['offset'] and breaks
                    # project_cam far downstream.
                    w, h = (int(x) for x in session.rig.size(session.cam_names[ci]))
                    if inst_boxes is not None:
                        # STORED BOXES ARE NOT DETECTOR BOXES, AND TRAINING PUTS THEM THROUGH THE
                        # RULE. Everything above is about a DETECTOR box, which is already a
                        # crop-rule box. `instances.pq` is not: the loader routes it through
                        # `_crop_source` -> `crop_box_for_points(..., pad=0)`, which SQUARES the
                        # extent and applies the `min_crop_dim` floor -- and 96% of rat-city's
                        # stored boxes are non-square (aspect p50 1.737). So an `instances`-trained
                        # run was served a tight, unfloored, non-square crop here and a squared,
                        # floored one in training, which after `_resize_camera` puts the animal at
                        # a different scale on a different-aspect canvas than any crop it ever saw.
                        #
                        # This is the SAME arithmetic as training, not a second rule: training
                        # unions the window's corners per camera and squares ONCE, which is what
                        # `crop_box_for_points` over the union corners does. `pad=0` because the
                        # stored extent is already padded -- see that function's own docstring.
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
                        kk = det_kpts_stc[a, frames, ci][..., :2].reshape(-1, 2)
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
                    outcome[a, wi] = OUTCOMES.index('no camera')
                    continue
                # THE CAMERA MUST DESCRIBE THE BOX THE PIXELS WERE ACTUALLY CUT WITH, and under
                # `--moving-crop` that is the moving box, not the window union. `apply_crop` sets
                # `size`, `_resize_camera` turns it into `scales`, and `forward` divides by that --
                # so leaving the union box here scales the decode by union_side/moving_side (p50
                # 1.23 on this root) and lands every keypoint short of where it belongs. It cost
                # pck@10 0.841 -> 0.383 at unchanged coverage, which is the signature: the rows are
                # all there and every one of them is in the wrong place.
                cgroup = [cropmod.apply_crop(window_cams[ci], b) for ci, b in zip(use, boxes)]
            else:
                pts = torch.as_tensor(src[a][frames], dtype=torch.float32)
                if not torch.isfinite(pts).all(-1).any():
                    outcome[a, wi] = OUTCOMES.index('no points')
                    continue
                use = cam_ix
                cgroup, boxes = boxes_from_points(pts, [window_cams[i] for i in cam_ix],
                                                  cfg.min_crop_dim, mode)
                if cgroup is None:
                    outcome[a, wi] = OUTCOMES.index('crop failed')
                    continue
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
                crop[a, wi, ci] = np.asarray(boxes[i], np.float32)
            outcome[a, wi] = OUTCOMES.index('decode failed')
            plans.append((a, use, boxes, cgroup, scales, uncropped))

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
        def decode_crops(plans):
            crops = {}
            cams = sorted({c for _, use, *_ in plans for c in use})

            def one(ci, pool):
                imgs = read_frames(group, session.cam_names[ci], frames, pool=pool)
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
                    with ThreadPoolExecutor(max_workers=min(_CAM_DECODE, len(cams))) as cpool:
                        list(cpool.map(lambda ci: one(ci, pool), cams))
            return crops

        def forward(plan, crops):
            """One animal, one window -> its prediction in the SOURCE frame, or None."""
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
            with share_scene(model) if cfg.anchor == 'self' else nullcontext():
                out = model(views, kpt_ids.to(dev), cgroup_d, mode=mode,
                            kpt_prior=None if prior is None else prior.to(dev),
                            prompt_time=None if prompt_t is None else prompt_t.to(dev),
                            kpt_chunk=chunk)
                if cfg.anchor == 'self':
                    out = self_prompt(model, views, kpt_ids.to(dev), cgroup_d, mode, out,
                                      kpt_chunk=chunk)
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
            return p, q, out

        crops = decode_crops(plans)

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
                got = forward(plan, crops)
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
                    crop_refined[a, wi, ci] = np.asarray(b2[i], np.float32)
                refined.append((a, use, b2, cg2, sc2, uncropped2))
            plans = refined
            crops = decode_crops(plans)

        for plan in plans:
            a, use, boxes, cgroup, scales, *_ = plan
            got = forward(plan, crops)
            if got is None:
                continue                        # already marked 'decode failed' above
            p, q, out = got
            outcome[a, wi] = OUTCOMES.index('ok')
            if pred_tri is not None and out.get('3d_pred_triangulate') is not None:
                pred_tri[a, frames] = out['3d_pred_triangulate'][0].detach().cpu().numpy()
                if out.get('tri_degenerate') is not None:
                    # A DEGENERATE SOLVE IS REPAIRED FROM THE RAYS, and `carry` now seeds the next
                    # window from this tensor -- so how often that happened has to be visible rather
                    # than absorbed into one silently-substituted array.
                    tri_bad[a, frames] = out['tri_degenerate'][0].detach().cpu().numpy()
            _fill_box_agreement(box_agree, a, frames, use, boxes, p, mode, window_cams)
            if kpt_agree is not None:
                _fill_kpt_agreement(kpt_agree, a, frames, use,
                                    [det_kpts_stc[a, frames, ci] for ci in use],
                                    p, mode, window_cams)
            vlogit = None
            if 'vis_pred' in out:
                v = out['vis_pred'][0].detach().cpu().numpy().reshape(len(frames), K)
                conf[a, frames] = v
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
                conf[a, frames[drop]] = np.nan
                box_agree[a, frames[drop]] = np.nan
            pred[a, frames] = p
            if q is not None:
                carried[a] = (torch.as_tensor(q[j]), int(frames[j]),
                              None if vlogit is None else torch.as_tensor(vlogit[j]))

    out_npz = {'pred': pred, 'conf': conf, 'animal_ids': np.asarray(animal_ids, object),
               'outcome': outcome, 'crop': crop, 'box_agree': box_agree,
               **({} if kpt_agree is None else {'kpt_agree': kpt_agree}),
               'window_start': np.asarray(starts),
               'outcome_names': np.asarray(OUTCOMES, object),
               'mode': mode, 'group_id': gid, 'session': session.session_id,
               'dataset': dataset_name, 'anchor': cfg.anchor,
               'carry_source': cfg.carry_source, 'n_frames': T_total,
               'boxes_from': 'detector' if boxes_stc is not None else
                             ('given points' if box_points is not None else
                              ('instances.pq' if inst_boxes is not None else 'labels'))}
    if pred_tri is not None:
        out_npz['pred_tri'] = pred_tri
        out_npz['tri_degenerate'] = tri_bad
    if crop_refined is not None:
        out_npz['crop_refined'] = crop_refined
    return out_npz


def _overlaps(a, b):
    """Do two xyxy boxes share any area? Either being non-finite is not an overlap."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    return bool(min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))


def _fill_kpt_agreement(kpt_agree, a, frames, use, det_kpts, p, mode, window_cams):
    """PER-KEYPOINT distance from the pose to the DETECTOR's own keypoint, in box sides.

    The third statement about where the animal is. `box_agree` compares the pose against its crop
    BOX, and report 13 withdrew its 2D half because the pose is decoded inside that very box, which
    bounds the statistic at about half a side by construction (every 2D arm reads p99 0.31-0.56).
    **This has no such bound**: the detector regresses in the full frame, independent of the crop,
    so pose and detector can disagree without limit and a large value means something in 2D too.

    Per KEYPOINT rather than per centroid, so it localises WHICH joint the two disagree about --
    and it needs no learned head and no per-dataset threshold, unlike `--vis-thresh`, whose logit
    has no portable value across roots.

    NOT a gate. Recorded as a diagnostic first; any gate built on it needs the rate-matched random
    control that `--vis-thresh` is quoted with, for the same reason.
    """
    for i, ci in enumerate(use):
        dk = det_kpts[i]                                  # (t,K,3) source pixels, or None
        if dk is None:
            continue
        if mode == '2d':
            q = np.asarray(p, np.float64)                 # already source pixels
        else:
            q = project_points_torch([window_cams[ci]],
                                     torch.as_tensor(p, dtype=torch.float32))[0].numpy()
        d = np.asarray(dk, np.float64)
        # The detector's own box side, per frame, from its keypoint extent -- the same quantity
        # every other distance in this file is normalised by.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lo = np.nanmin(d[..., :2], axis=-2)
            hi = np.nanmax(d[..., :2], axis=-2)
        side = 0.5 * ((hi[..., 0] - lo[..., 0]) + (hi[..., 1] - lo[..., 1]))
        side = np.where(np.isfinite(side) & (side > 1e-6), side, np.nan)
        kpt_agree[a, frames, ci] = (np.linalg.norm(q - d[..., :2], axis=-1)
                                    / side[..., None])


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


ORACLE_CORRUPTIONS = ('off', 'stale', 'other')


def _corrupt_prior(cfg, src, a, n_lab, frames, boxes, scales, mode, cgroup):
    """The oracle prior, optionally broken on purpose. Returns (pose in the SOURCE frame, qt).

    THIS MEASURES alpha = d(output)/d(prior), the echo coefficient, and it is the cheapest thing in
    the whole programme: no training, one afternoon, and it decides whether the prompt needs
    retraining at all. Near 0 means the model corrects a bad prior from the pixels; near 1 means it
    echoes it, and a loop that feeds the model its own output then has gain.

    The three corruptions are the three failures deployment actually produces, and NONE of them is
    in training -- `dataset.py` offers GT plus i.i.d. Gaussian jitter, or nothing at all:

    - `off:<x>`   a WHOLE-BODY offset of x crop widths, one direction for the whole pose. This is
                  the shape of a lag: every keypoint wrong the same way, which i.i.d. noise never is.
    - `stale:<n>` the pose from n frames earlier, which is what `carried` degrades into when the box
                  source loses an animal for a few windows.
    - `other`     the NEIGHBOURING animal's pose, which is what a row swap hands the next window.

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
    elif kind == 'stale':
        t0 = max(0, t0 - int(amt))
    p = torch.as_tensor(src[row][t0], dtype=torch.float32)
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
        # Offset-invariant (a Jacobian) and `fin[None]` carries no time axis -- same reason as
        # dataset.py's camera-scale probe, and the same collapse.
        scale = torch.nanmedian(get_camera_scale(
            [cropmod.with_static_offset(c) for c in cgroup], fin[None]))
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
        p, qt = _corrupt_prior(cfg, src, a, n_lab, frames, boxes, scales, mode, cgroup)
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
    if mode == '2d':
        # the carried/labelled pose is in SOURCE pixels; the model works in crop pixels
        #
        p = (p - torch.as_tensor(np.asarray(boxes[0][:2], np.float32))) * scales[0]
    p = p.clone()
    p[prior_out_of_bounds(p, mode, cgroup)] = float('nan')
    qt = min(max(qt, 0), len(frames) - 1)
    return p[None], torch.full((1, K), qt, dtype=torch.int32)
