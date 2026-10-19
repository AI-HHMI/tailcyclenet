"""The per-keypoint prior and the box prompt a window is given (`--anchor`, `--box-prompt`)."""
from __future__ import annotations

import torch

from ..data import crop as cropmod
from ..data.camera import prior_out_of_bounds
from .crops import prior_to_model_frame
from .oracle import corrupt_prior


def give_box_this_window(cfg, carried_a, prior):
    """Whether this animal's window should receive a box prompt this call.

    `box_prompt_first_only` withholds the box once a carry EXISTS and IS ACTUALLY USED as this
    window's prior -- `carried_a is None` alone under-withholds: `build_prior` can ALSO decline
    a non-None carry as stale (a skipped window, `qt < 0`) or shape-mismatched, in which case no
    identity signal reached the model at all and the box should compensate, exactly as it would
    at the animal's true first window. `prior is None` is `build_prior`'s own verdict for THIS
    window, so this can never disagree with what the model actually received.
    """
    return (cfg.box_prompt != 'none'
            and (not cfg.box_prompt_first_only or carried_a is None or prior is None))


def deploy_box_prompt(mode, src_pts, boxes_stc, frames, a, use, boxes, scales, cgroup, dev):
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
            cf = (corners - origin) * torch.as_tensor(scales[i], dtype=corners.dtype)
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
    cf = (source - origin) * torch.as_tensor(scales[0], dtype=source.dtype)
    size = cgroup[0]['size']
    T = cf.shape[0]
    out = torch.full((T, 1, 4), float('nan'), dtype=torch.float32)
    for t in range(T):
        bx = cropmod.crop_box_for_points(cf[t], size, bpmod.BOX_PROMPT_MIN_DIM,
                                         bpmod.BOX_PROMPT_PAD)
        if bx is not None:
            out[t, 0] = bx.to(torch.float32)
    return out[None].to(dev)


def build_prior(cfg, carried, src, a, n_lab, frames, boxes, scales, mode, K, R, cgroup):
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
        p, qt = corrupt_prior(cfg, src, a, n_lab, frames, boxes, mode, cgroup, scales)
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
    p = prior_to_model_frame(p, mode, boxes, scales)
    p = p.clone()
    p[prior_out_of_bounds(p, mode, cgroup)] = float('nan')
    qt = min(max(qt, 0), len(frames) - 1)
    return p[None], torch.full((1, K), qt, dtype=torch.int32)
