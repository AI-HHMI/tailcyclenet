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

# WHY AN (ANIMAL, WINDOW) PRODUCED NOTHING. Five separate aborts in the loop below wrote the same
# NaN, so a coverage number could not be decomposed at all -- "the detector offered no box", "the
# association matched no camera", "the crop rule refused" and "the file would not decode" are four
# different problems with four different fixes, and they arrived indistinguishable.
OUTCOMES = ('ok', 'no box', 'no camera', 'no points', 'crop failed', 'decode failed')


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
    # None -> carry every keypoint the bounds mask keeps. A float drops the carried keypoints whose
    # own `vis_pred` LOGIT is below it, per keypoint, before they become a prior. Not a row gate:
    # this decides what the model is told, not what it reports.
    prior_vis_thresh: float | None = None
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
    """
    import cv2

    aff = _crop_affine((imgs[0].shape[1], imgs[0].shape[0]), box, target_size, None)
    out = [im if aff is None else cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR)
           for im in imgs]
    return torch.from_numpy(np.asarray(out))[None]


def self_prompt(model, views, kpt_ids, cgroup, mode, first, kpt_chunk=None):
    """Re-query at the model's OWN frame-0 prediction. THE label-free prompted regime.

    `first` is a completed prior-free pass. Its frame-0 pose becomes the prior for a second pass,
    which is what a deployed model does on the first window of a clip and what the periodic val
    eval reports alongside the prior-free number. No ground truth is read, so no gate reopens.

    Shared with the trainer deliberately: this repo has one window loop and it should have one
    self-prompt, or the number training reports and the number inference produces drift apart.
    """
    p = first['coords_pred'][0].detach()
    prior = p[0][None].clone()                         # (1,K,R), the frame-0 pose
    qt = torch.zeros(prior.shape[:2], dtype=torch.int32, device=prior.device)
    return model(views, kpt_ids, cgroup, mode=mode, kpt_prior=prior, prompt_time=qt,
                 kpt_chunk=kpt_chunk)


@torch.no_grad()
def run_group(model, session: Session, gid: str, registry, dataset_name: str,
              cfg: InferConfig, box_points=None, boxes_stc=None) -> dict:
    """Predict every animal in one group. Returns arrays in the SOURCE coordinate frame.

    Crops come from exactly one of two sources, and they are NOT comparable:

    - `box_points` (S,T,K,R): points the crop rule follows, shaped like the labels. Passing the
      labels themselves is the GT-crop upper bound.
    - `boxes_stc` (S,T,C,4): boxes given directly, from a detector or a detections file. This is
      the deployment number.

    Whichever was used is recorded in the result so a caller cannot quote one as the other.
    """
    assert cfg.anchor in ANCHORS, f'anchor must be one of {ANCHORS}'
    if cfg.anchor in ('carry', 'self') and cfg.overlap < 1:
        raise ValueError(f'anchor={cfg.anchor!r} carries a pose across windows and needs '
                         'overlap >= 1; got 0')

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
                if not np.isfinite(bb).all(-1).any():
                    continue
            elif inst_boxes is not None and a < len(inst_boxes):
                bb = inst_boxes[a][frames]
                if not np.isfinite(bb).all(-1).any():
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
                    # int32 and clamped into the image, exactly like the crop rule's own box: a
                    # float or off-frame box produces a negative cam['offset'] and breaks
                    # project_cam far downstream.
                    w, h = (int(x) for x in session.rig.size(session.cam_names[ci]))
                    x0 = int(np.clip(np.floor(v[:, 0].min()), 0, w - 1))
                    y0 = int(np.clip(np.floor(v[:, 1].min()), 0, h - 1))
                    x1 = int(np.clip(np.ceil(v[:, 2].max()), x0 + 1, w))
                    y1 = int(np.clip(np.ceil(v[:, 3].max()), y0 + 1, h))
                    boxes.append(torch.tensor([x0, y0, x1, y1], dtype=torch.int32))
                    use.append(ci)
                if not use:
                    outcome[a, wi] = OUTCOMES.index('no camera')
                    continue
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
            for i, cam in enumerate(cgroup):
                cgroup[i], s = _resize_camera(cam, cfg.image_size)
                scales.append(s)
            # The box BEFORE the pixels, so a decode failure still shows what it was reaching for.
            for i, ci in enumerate(use):
                crop[a, wi, ci] = np.asarray(boxes[i], np.float32)
            outcome[a, wi] = OUTCOMES.index('decode failed')
            plans.append((a, use, boxes, cgroup, scales))

        # ONE DECODE PER (CAMERA, FRAME) PER WINDOW, shared by every animal in it.
        #
        # ponytail: peak memory is one camera's window of FULL frames plus every animal's crops --
        # 24 x 21 MB on johnson-mouse's 3208x2200 rig, which has one animal. A wide rig with many
        # animals would want the frame loop chunked; nothing shipped is both.
        def decode_crops(plans):
            crops = {}
            with ThreadPoolExecutor(max_workers=8) as pool:
                for ci in sorted({c for _, use, *_ in plans for c in use}):
                    imgs = read_frames(group, session.cam_names[ci], frames, pool=pool)
                    # A file that will not decode is a property of the file, not of the animal, so
                    # it takes out every animal that wanted this camera -- which is what the
                    # per-animal decode did too, one animal at a time.
                    ok = not any(im is None for im in imgs)
                    for a, use, boxes, cgroup, _ in plans:
                        if ci in use:
                            i = use.index(ci)
                            crops[a, ci] = (_crop_views(imgs, boxes[i],
                                                        cgroup[i]['size'].tolist())
                                            if ok else None)
                    del imgs
            return crops

        def forward(plan, crops):
            """One animal, one window -> its prediction in the SOURCE frame, or None."""
            a, use, boxes, cgroup, scales = plan
            # uint8; the model divides on device. Same contract as the training loader.
            views = [crops[a, ci] for ci in use]
            if any(v is None for v in views):
                return None                     # already marked 'decode failed' above
            prior, prompt_t = _build_prior(cfg, carried[a], src[a] if a < n_lab else None,
                                           frames, boxes, scales, mode, K, R, cgroup)
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
            if mode == '2d':
                # crop pixels -> source pixels: undo the resize, then the crop origin
                p = p / scales[0] + np.asarray(boxes[0][:2], np.float32)
            return p, out

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
            # An animal whose refined crop fails keeps its first-pass plan rather than being
            # dropped: a bad prediction must not cost coverage a loose box already had.
            refined = []
            for plan in plans:
                a, use, boxes, cgroup, scales = plan
                got = forward(plan, crops)
                if got is None:
                    refined.append(plan)
                    continue
                pts = torch.as_tensor(got[0], dtype=torch.float32)
                cg2, b2 = boxes_from_points(pts, [window_cams[i] for i in use],
                                            cfg.min_crop_dim, mode)
                if cg2 is None:
                    refined.append(plan)
                    continue
                sc2 = []
                for i, cam in enumerate(cg2):
                    cg2[i], s = _resize_camera(cam, cfg.image_size)
                    sc2.append(s)
                for i, ci in enumerate(use):
                    crop[a, wi, ci] = np.asarray(b2[i], np.float32)
                refined.append((a, use, b2, cg2, sc2))
            plans = refined
            crops = decode_crops(plans)

        for plan in plans:
            a, use, boxes, cgroup, scales = plan
            got = forward(plan, crops)
            if got is None:
                continue                        # already marked 'decode failed' above
            p, out = got
            outcome[a, wi] = OUTCOMES.index('ok')
            pred[a, frames] = p
            _fill_box_agreement(box_agree, a, frames, use, boxes, p, mode, window_cams)
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
                with np.errstate(all='ignore'):
                    # The MEDIAN over keypoints: a mean lets one confident keypoint carry a row the
                    # model otherwise declined.
                    drop = np.nanmedian(vlogit, axis=-1) < cfg.vis_thresh
                pred[a, frames[drop]] = np.nan
                conf[a, frames[drop]] = np.nan
                box_agree[a, frames[drop]] = np.nan
            carried[a] = (torch.as_tensor(p[j]), int(frames[j]),
                          None if vlogit is None else torch.as_tensor(vlogit[j]))

    return {'pred': pred, 'conf': conf, 'animal_ids': np.asarray(animal_ids, object),
            'outcome': outcome, 'crop': crop, 'box_agree': box_agree,
            'window_start': np.asarray(starts),
            'outcome_names': np.asarray(OUTCOMES, object),
            'mode': mode, 'group_id': gid, 'session': session.session_id,
            'dataset': dataset_name, 'anchor': cfg.anchor, 'n_frames': T_total,
            'boxes_from': 'detector' if boxes_stc is not None else
                          ('given points' if box_points is not None else
                           ('instances.pq' if inst_boxes is not None else 'labels'))}


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


def _build_prior(cfg, carried, src_animal, frames, boxes, scales, mode, K, R, cgroup):
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
        if src_animal is None:                   # a detector row with no label row behind it
            return None, None
        p = torch.as_tensor(src_animal[frames[0]], dtype=torch.float32)
        qt = 0
    else:                                    # 'carry'
        if carried is None:
            return None, None
        p = carried[0].clone().float()
        qt = int(carried[1]) - int(frames[0])
        # A KEYPOINT THE MODEL ITSELF DOUBTED IS NOT A PRIOR EITHER. The bounds mask below drops
        # what left the crop; this drops what the previous window reported as not visible, in the
        # same currency the no-query tokens already speak (NaN = "I was not told"). `vis_pred`, not
        # `conf`: `conf_pred_2d` is an unnormalised sigmoid whose scale means nothing here.
        # posetail-pose ships this default-off and byte-identical, and it was refuted for the
        # purpose it was built for -- so off is the default here too.
        vis = carried[2] if len(carried) > 2 else None
        if cfg.prior_vis_thresh is not None and vis is not None and vis.shape == p.shape[:1]:
            p[vis < cfg.prior_vis_thresh] = float('nan')
        # A STALE PRIOR IS NOT A PRIOR. `carried` is only written on a window that predicted, so an
        # animal the box source lost for a few windows keeps handing back a pose from before this
        # one -- and `qt` clamps to 0 below, presenting it as this window's first frame. Within the
        # overlap that is the ordinary case (the carried frame IS in the previous window's tail);
        # beyond it the pose describes a frame the model was never shown, and in 3D the bounds mask
        # cannot catch it, because a pose that is stale but still visible to two cameras passes.
        if -qt > cfg.overlap:
            return None, None
    if p.shape != (K, R):
        return None, None
    if mode == '2d':
        # the carried/labelled pose is in SOURCE pixels; the model works in crop pixels
        p = (p - torch.as_tensor(np.asarray(boxes[0][:2], np.float32))) * scales[0]
        w, h = (float(x) for x in cgroup[0]['size'][:2])
        outside = ((p[:, 0] < 0) | (p[:, 0] >= w) | (p[:, 1] < 0) | (p[:, 1] >= h))
    else:
        # A 3D point no PAIR of cameras can see is not reconstructible, so it cannot be a prior
        # the model should trust -- the 3D analogue of leaving the crop. A MOVING camera answers
        # per frame ((T,K), not (K,)), and the prior is one pose for the whole window, so a
        # camera counts if it sees the point at any point during it.
        seen = []
        for c in cgroup:
            v = is_point_visible(c, p, margin=2)
            seen.append(v.any(0) if v.ndim > 1 else v)
        outside = torch.stack(seen).sum(0) < 2
    p = p.clone()
    p[outside] = float('nan')
    qt = min(max(qt, 0), len(frames) - 1)
    return p[None], torch.full((1, K), qt, dtype=torch.int32)
