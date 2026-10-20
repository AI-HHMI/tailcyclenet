"""Crop planning for one window: box selection, per-camera crops, and the source <-> crop-frame maps."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from tracktail.model.cube import project_points_torch

from ..data import crop as cropmod
from ..data.pixels import crop_affine


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


def to_device(cgroup, device):
    """Move each camera's tensors to `device`; non-tensors (e.g. `size`) are left as-is."""
    return [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
            for c in cgroup]


def crop_views(imgs, box, target_size):
    """Crop+resize one camera's already-decoded window -> (1,T,H,W,3) uint8.

    Decode is per (camera, frame) and crop is per (animal, camera, frame), so cropping inside the
    animal loop paid the full-frame decode once per animal. `box` is one [x1,y1,x2,y2] for the
    whole window or a (T,4) of per-frame boxes (kept for the loader's per-frame crops).
    """
    import cv2

    b = np.asarray(box)
    src_wh = (imgs[0].shape[1], imgs[0].shape[0])
    if b.ndim == 1:
        aff = crop_affine(src_wh, box, target_size, None)
        out = [im if aff is None else cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR)
               for im in imgs]
    else:
        out = []
        for t, im in enumerate(imgs):
            aff = crop_affine(src_wh, b[min(t, len(b) - 1)], target_size, None)
            out.append(im if aff is None else
                       cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR))
    return torch.from_numpy(np.asarray(out))[None]


def overlaps(a, b):
    """Do two xyxy boxes share any area? Either being non-finite is not an overlap."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    return bool(min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))


def fill_box_agreement(box_agree, a, frames, use, boxes, p, mode, window_cams):
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


def prior_to_model_frame(p, mode, boxes, scales):
    """Source pixels/world -> the model's own crop frame (3D passes straight through). Factored
    out so `corrupt_prior`'s `near` applies the identical conversion to a candidate neighbour.
    """
    if mode == '2d':
        return (p - torch.as_tensor(np.asarray(boxes[0][:2], np.float32))) * scales[0]
    return p


def refine_every_camera(points, use, boxes, cam_ix, window_cams, min_crop_dim, min_inside=0.5):
    """TR1: pass-2 crops for EVERY calibrated camera from pass 1's 3D pose, tested per camera.

    Inputs: points -- (T,K,3) pass-1 world pose; use / boxes -- pass 1's cameras and crop
        boxes; cam_ix -- every camera index; window_cams -- the window's camera group;
        min_crop_dim -- the crop rule's floor; min_inside -- fraction of the finite projected
        points that must land inside the image for a camera pass 1 did not use.
    Outputs: (use2, boxes2, cgroup2) in `cam_ix` order, cropped but not resized; empty lists
        when no camera survives (the caller keeps pass 1's plan).

    A pass-1 camera is kept only when its refined box overlaps its pass-1 box (the per-camera
    form of the all-or-nothing test, so one wrong-animal camera no longer cancels the others).
    A camera with no pass-1 box is added when the pose reprojects into its frame and the crop
    rule returns a box. Each camera's box is the crop rule on that camera alone, which is what
    `crop_to_points_3d` computes per camera, so a kept camera's box is byte-identical to the
    shipped refine's.
    """
    use2, boxes2, cg2 = [], [], []
    for ci in cam_ix:
        cam = window_cams[ci]
        cg, bx = cropmod.crop_to_points_3d([cam], points, min_crop_dim)
        if cg is None:
            continue
        if ci in use:
            if not overlaps(bx[0], boxes[use.index(ci)]):
                continue
        else:
            uv = project_points_torch([cam], points)[0].reshape(-1, 2)
            uv = uv[torch.isfinite(uv).all(-1)]
            w, h = (float(x) for x in cam['size'])
            inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
            if not len(uv) or float(inside.float().mean()) < min_inside:
                continue
        use2.append(ci)
        boxes2.append(bx[0])
        cg2.append(cg[0])
    return use2, boxes2, cg2
