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

from dataclasses import dataclass

import numpy as np
import torch

from . import crop as cropmod
from .dataset import _resize_camera, read_frames
from .format import VISIBLE, Labels, Session

ANCHORS = ('none', 'carry', 'self', 'labels')


@dataclass
class InferConfig:
    n_frames: int = 24
    overlap: int = 4
    image_size: int = 256
    min_crop_dim: int = 64
    anchor: str = 'carry'
    max_animals: int = 0          # 0 -> every animal the box source offers
    kpt_chunk: int = 0            # 0 -> decode every keypoint in one pass
    device: str = 'cuda:0'


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
    T_total = group.n_frames
    # The registry is per DATASET and the keypoint axis is per SESSION -- and `scripts/infer.py`
    # will happily hand us a bare session directory. `ids_for` aligns to the axis we actually
    # hold, by name, so a session that reorders the root's keypoints or carries a subset of them
    # is predicted correctly instead of being silently relabelled.
    kpt_ids = torch.as_tensor(registry.ids_for(dataset_name, session.names),
                              dtype=torch.long)[None]
    assert kpt_ids.shape[1] == K

    src = box_points if box_points is not None else (
        lab.points3d if mode == '3d' else lab.points2d[..., 0, :])
    n_src = boxes_stc.shape[0] if boxes_stc is not None else src.shape[0]
    S = n_src if cfg.max_animals == 0 else min(n_src, cfg.max_animals)
    cam_ix = list(range(len(session.rig)))
    # A DETECTOR ROW IS NOT A LABEL ROW. `S` comes from the box source, which on the deployment
    # path is the detector and can offer more animals than the session ever labelled -- so `src`
    # is indexable only up to its own length, and the surplus rows get an invented id.
    n_lab = 0 if src is None else len(src)
    animal_ids = [lab.animal_ids[a] if a < len(lab.animal_ids) else f'det{a:02d}'
                  for a in range(S)]

    pred = np.full((S, T_total, K, R), np.nan, np.float32)
    conf = np.full((S, T_total, K), np.nan, np.float32)
    carried = [None] * S                      # per-animal prior for the next window

    for start in _window_starts(T_total, cfg.n_frames, cfg.overlap):
        frames = np.arange(start, min(start + cfg.n_frames, T_total))
        if len(frames) < 2:                   # T=1 hits posetail's gT = T // tubelet = 0 bug
            frames = np.clip(np.arange(start, start + 2), 0, T_total - 1)
        # ONE camera group per window, carrying per-frame extrinsics where a camera moves. Built
        # here rather than per animal: the old per-animal build dropped `moving_ext` entirely and
        # cost O(C^2) `format_camera` calls per animal.
        window_cams = session.cgroup(gid, frames)
        for a in range(S):
            if boxes_stc is not None:
                # Boxes given directly (detector / detections file). One box per camera for
                # this window: the union over the window's frames, so the animal does not walk
                # out of its own crop mid-window.
                bb = boxes_stc[a][frames]                          # (t, C, 4)
                if not np.isfinite(bb).all(-1).any():
                    continue
                # USE THE CAMERAS THAT SAW IT, not all or nothing. A detector legitimately
                # misses a view -- cross-view association leaves an unmatched camera NaN -- and
                # requiring a box in every camera dropped the whole animal for the window even
                # when two views had it. The model is trained on camera subsets already
                # (`cams_to_sample`, and `prob_2d_only` trains the one-camera case outright), so
                # a subset is a supported input; silently discarding the animal is pure lost
                # coverage, and coverage is a number this repo reports.
                use, boxes = [], []
                for i, ci in enumerate(cam_ix):
                    v = bb[:, i][np.isfinite(bb[:, i]).all(-1)]
                    if not len(v):
                        continue
                    # int32 and clamped into the image, exactly like the crop rule's own box:
                    # a float or off-frame box produces a negative cam['offset'] and breaks
                    # project_cam far downstream.
                    w, h = (int(x) for x in session.rig.size(session.cam_names[ci]))
                    x0 = int(np.clip(np.floor(v[:, 0].min()), 0, w - 1))
                    y0 = int(np.clip(np.floor(v[:, 1].min()), 0, h - 1))
                    x1 = int(np.clip(np.ceil(v[:, 2].max()), x0 + 1, w))
                    y1 = int(np.clip(np.ceil(v[:, 3].max()), y0 + 1, h))
                    boxes.append(torch.tensor([x0, y0, x1, y1], dtype=torch.int32))
                    use.append(ci)
                if not use:
                    continue
                cgroup = [cropmod.apply_crop(window_cams[ci], b) for ci, b in zip(use, boxes)]
            else:
                pts = torch.as_tensor(src[a][frames], dtype=torch.float32)
                if not torch.isfinite(pts).all(-1).any():
                    continue
                use = cam_ix
                cgroup, boxes = boxes_from_points(pts, [window_cams[i] for i in cam_ix],
                                                  cfg.min_crop_dim, mode)
                if cgroup is None:
                    continue
            scales = []
            for i, cam in enumerate(cgroup):
                cgroup[i], s = _resize_camera(cam, cfg.image_size)
                scales.append(s)

            views = []
            for i, ci in enumerate(use):
                imgs = read_frames(group, session.cam_names[ci], frames,
                                   crop_coords=boxes[i],
                                   target_size=cgroup[i]['size'].tolist())
                if any(im is None for im in imgs):
                    break
                # uint8; the model divides on device. Same contract as the training loader.
                views.append(torch.from_numpy(np.asarray(imgs))[None])
            if len(views) != len(use):
                continue

            prior, prompt_t = _build_prior(cfg, carried[a], src[a] if a < n_lab else None,
                                           frames, boxes, scales, mode, K, R)
            dev = cfg.device
            chunk = cfg.kpt_chunk or None
            out = model([v.to(dev) for v in views], kpt_ids.to(dev), _to_device(cgroup, dev),
                        mode=mode,
                        kpt_prior=None if prior is None else prior.to(dev),
                        prompt_time=None if prompt_t is None else prompt_t.to(dev),
                        kpt_chunk=chunk)
            if cfg.anchor == 'self':
                out = self_prompt(model, [v.to(dev) for v in views], kpt_ids.to(dev),
                                  _to_device(cgroup, dev), mode, out, kpt_chunk=chunk)

            p = out['coords_pred'][0].detach().cpu().numpy()          # (t,K,R)
            if mode == '2d':
                # crop pixels -> source pixels: undo the resize, then the crop origin
                p = p / scales[0] + np.asarray(boxes[0][:2], np.float32)
            pred[a, frames] = p
            if 'vis_pred' in out:
                v = out['vis_pred'][0].detach().cpu().numpy()
                conf[a, frames] = v.reshape(len(frames), K)
            # The pose the NEXT window opens on. Clamped from the front: a group shorter than
            # `overlap` gives a window with fewer frames than the step, and a plain negative
            # index runs off the start of it.
            j = max(0, len(frames) - cfg.overlap) if cfg.overlap else len(frames) - 1
            carried[a] = (torch.as_tensor(p[j]), int(frames[j]))

    return {'pred': pred, 'conf': conf, 'animal_ids': np.asarray(animal_ids, object),
            'mode': mode, 'group_id': gid, 'session': session.session_id,
            'dataset': dataset_name, 'anchor': cfg.anchor, 'n_frames': T_total,
            'boxes_from': 'detector' if boxes_stc is not None else
                          ('given points' if box_points is not None else 'labels')}


def _build_prior(cfg, carried, src_animal, frames, boxes, scales, mode, K, R):
    """The per-keypoint prior for this window, in the model's coordinate frame."""
    if cfg.anchor in ('none', 'self'):
        return None, None
    if cfg.anchor == 'labels':
        # ORACLE. Ground truth as the prior; not a deployment number.
        if src_animal is None:                   # a detector row with no label row behind it
            return None, None
        p = torch.as_tensor(src_animal[frames[0]], dtype=torch.float32)
    else:                                    # 'carry'
        if carried is None:
            return None, None
        p = carried[0].clone().float()
    if p.shape != (K, R):
        return None, None
    if mode == '2d':
        # the carried/labelled pose is in SOURCE pixels; the model works in crop pixels
        p = (p - torch.as_tensor(np.asarray(boxes[0][:2], np.float32))) * scales[0]
    return p[None], torch.zeros((1, K), dtype=torch.int32)
