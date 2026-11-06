"""Crop planning for one window: box selection, per-camera crops, and the source <-> crop-frame maps."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from tracktail.model.cube import project_points_torch

from ..data import crop as cropmod
from ..data.pixels import crop_affine


def boxes_from_points(points, cgroup, min_crop_dim, mode, pad_frac=0.05):
    """Crop boxes for one animal in one window, from points -- the crop rule, shared with training.

    3D points are world coordinates and get projected; 2D are already pixels. None when nothing
    is finite. `cgroup` is built by the caller once per window (per-animal builds dropped the
    per-frame extrinsics).
    """
    if mode == '3d':
        cg, boxes = cropmod.crop_to_points_3d(cgroup, points, min_crop_dim, pad_frac=pad_frac)
        return (cg, boxes) if cg is not None else (None, None)
    cam, box, _ = cropmod.crop_to_points_2d(cgroup[0], points, min_crop_dim, pad_frac=pad_frac)
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


def floor_box_side(box, size, side):
    """Grow an xyxy box about its centre until both sides are >= `side` (each capped at the
    image), shifted back inside the image. int32; a box already at least `side` on both axes (or
    a non-finite `side`) comes back unchanged."""
    b = torch.as_tensor(box).to(torch.int32).clone()
    if not np.isfinite(side):
        return b
    W, H = (int(v) for v in size)
    for ax, lim in ((0, W), (1, H)):
        want = min(int(np.ceil(side)), lim)
        if int(b[ax + 2] - b[ax]) >= want:
            continue
        c = (int(b[ax]) + int(b[ax + 2])) // 2
        lo = min(max(c - want // 2, 0), lim - want)
        b[ax], b[ax + 2] = lo, lo + want
    return b


def floor_refine_box(box2, box1, size, crop_inflate):
    """`--refine` pass 2 must not zoom past the animal's own (de-inflated) pass-1 box.

    Pass 2 crops from pass 1's KEYPOINTS. Where the run's training crop was not keypoint-derived
    (`box_source = "instances"` with a skeleton that does not span the animal -- a 1-keypoint
    centroid root is the extreme: its extent is ~0 and the rule floors at `min_crop_dim`), that
    crop is many times smaller than anything the model trained on and pass 2 returns noise of
    the order of the crop half-width. When the pass-2 box is under half the de-inflated pass-1
    side (the same 2x scale gate as `scale_agrees`), grow it about its centre to that side;
    otherwise return it unchanged (int32). Returns (box, changed). The refine path applies it only
    when the run's `box_source` is not `keypoints` (a keypoint-trained run's pass-2 crop IS its
    training crop)."""
    b2 = torch.as_tensor(box2).to(torch.int32)
    b1 = np.asarray(box1, np.float64)
    side = max(b1[2] - b1[0], b1[3] - b1[1]) / max(float(crop_inflate), 1.0)
    s2 = float(max(b2[2] - b2[0], b2[3] - b2[1]))
    if not np.isfinite(side) or s2 >= 0.5 * side:
        return b2, False
    nb = floor_box_side(b2, size, side)
    return nb, not torch.equal(nb, b2)


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


def window_crop_box(v, size, min_crop_dim, pad_frac=0.0):
    """THE deployment crop: (N,4) finite TIGHT boxes (one slot, one camera, one window) -> the
    int32 crop-rule box of their union, or None when there are none.

    The crop rule runs ONCE, on the union's corners: pad every side by `pad_frac` x the union's
    longer side, square about the centre, floor at `min_crop_dim`, clamp -- exactly what training
    does to the window's keypoints (or, `pad_frac = 0`, to its stored instance boxes). Per-frame
    boxes are never squared first: a union of squares only grows (+3.06 mm when re-squared).
    """
    v = np.asarray(v, np.float64).reshape(-1, 4)
    v = v[np.isfinite(v).all(-1)]
    if not len(v):
        return None
    corners = torch.as_tensor(np.concatenate([v[:, :2], v[:, 2:]], 0), dtype=torch.float32)
    return cropmod.crop_box_for_points(corners, torch.as_tensor(size).to(torch.float32),
                                       min_crop_dim, 0.0, pad_frac)


def union_box(v):
    """The xyxy union of a (T,4) per-frame box stack over its finite rows; NaN when none is."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v).all(-1)]
    if not len(v):
        return np.full(4, np.nan)
    return np.array([v[:, 0].min(), v[:, 1].min(), v[:, 2].max(), v[:, 3].max()])


def adaptive_inflation(box, size, others, f_max, fallback, step=0.05):
    """R3: the pass-1 inflation for one (animal, window, camera) -> a factor.

    The largest factor on a `step` grid in [1.0, f_max] whose `inflate_box` does not share area
    with any of `others` (the OTHER animals' boxes in this camera this window); `fallback` (the
    shipped `crop_inflate`) when no other animal has a box here. Inflation about a fixed centre
    only grows the box, so the first factor that touches a neighbour ends the search; a box that
    already touches one gets 1.0. The discriminator this spends is crop side vs distance to the
    nearest other animal: wide where the animal is alone, tight where it is crowded.
    """
    others = [o for o in others if np.isfinite(np.asarray(o, np.float64)).all()]
    if not others:
        return float(fallback)
    best = 1.0
    for n in range(int(round((float(f_max) - 1.0) / step)) + 1):
        f = min(float(f_max), round(1.0 + n * step, 6))
        b = cropmod.inflate_box(box, size, f)
        if any(overlaps(b, o) for o in others):
            break
        best = f
    return best


def box_disagrees(box, other, sides=1.0):
    """R4: are two xyxy boxes' centres more than `sides` of `box`'s mean side apart?"""
    b, o = np.asarray(box, np.float64), np.asarray(other, np.float64)
    side = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
    d = np.hypot((b[0] + b[2] - o[0] - o[2]) / 2, (b[1] + b[3] - o[1] - o[3]) / 2)
    return bool(d > sides * side)


def carry_duplicate(points, others, cams, min_crop_dim, radius, min_shared=2, pad_frac=0.05):
    """R4 gate: is a carried 3D pose the same animal as another slot's carried pose?

    Inputs: points -- (K,3) this slot's carried pose; others -- the OTHER slots' fresh carried
        poses, each (K,3); cams -- the window's camera group; min_crop_dim -- the crop rule's
        floor; radius -- duplicate radius in crop-box sides (0 turns the gate off).
    Output: True when, in at least `min_shared` cameras where both poses are in frame, every
        gap between the two poses' projected centroids is within `radius` sides of the smaller
        crop box -- the 3D tracker's duplicate rule, in its unit system.

    The carried pose is the slot's own previous answer, so it is evidence about WHERE the slot
    is only while no other slot claims the same animal. A carry that duplicates another slot's
    is a drifted or ghost row (Seq29 clean: 10 of R4's 13 windows, carries 1-11 mm apart, while
    under drop/swap corruption the 5th-percentile gap is 185-430 mm on Seq59/P02 Seq41 and only
    2-9% of acting windows are duplicates); cropping from it overrides the detector box that
    would have pulled the row back.
    """
    if radius <= 0 or not others:
        return False
    pts = torch.as_tensor(points, dtype=torch.float32)

    def per_cam(p):
        """{camera: (projected centroid, crop-box mean side)} for the cameras `p` is in."""
        out = {}
        p = p[torch.isfinite(p).all(-1)]
        if not len(p):
            return out
        cen = p.mean(0).reshape(1, 1, 3)
        for ci, cam in enumerate(cams):
            if not in_frame(p[None], cam):
                continue
            cg, bx = cropmod.crop_to_points_3d([cam], p[None], min_crop_dim, pad_frac=pad_frac)
            if cg is None:
                continue
            b = np.asarray(bx[0], np.float64)
            uv = project_points_torch([cam], cen)[0].reshape(-1, 2).mean(0).numpy()
            out[ci] = (uv, 0.5 * ((b[2] - b[0]) + (b[3] - b[1])))
        return out

    mine = per_cam(pts)
    for o in others:
        theirs = per_cam(torch.as_tensor(o, dtype=torch.float32))
        shared = sorted(set(mine) & set(theirs))
        if len(shared) < min_shared:
            continue
        gaps = [float(np.linalg.norm(mine[c][0] - theirs[c][0]))
                / max(min(mine[c][1], theirs[c][1]), 1e-6) for c in shared]
        if max(gaps) <= radius:
            return True
    return False


def scale_agrees(box, ref_side, max_ratio):
    """R4 gate: is `box`'s mean side within a factor `max_ratio` of `ref_side` (either way)?

    The carried pose's crop must be the size of the animal the detector boxes describe: a carry
    whose crop is more than 2x larger or smaller than the slot's own boxes this window is an
    exploded or collapsed triangulation, not the animal (3dpop test, clean boxes: 65 of 163
    acting camera-windows fall outside [0.5, 2] -- one 2-bird clip, P02 Seq50, carries the
    whole clean regression; under swap:0.25 / drop:0.25 only 2.7% / 1.8% do). `max_ratio` 0
    turns the gate off; a NaN or non-positive reference agrees.
    """
    if max_ratio <= 0 or not np.isfinite(ref_side) or ref_side <= 0:
        return True
    b = np.asarray(box, np.float64)
    r = 0.5 * ((b[2] - b[0]) + (b[3] - b[1])) / ref_side
    return bool(1.0 / max_ratio <= r <= max_ratio)


def in_frame(points, cam, min_inside=0.5):
    """Does a 3D pose reproject into `cam`'s image? At least `min_inside` of its finite projected
    points must land inside; no finite point is not in frame."""
    uv = project_points_torch([cam], points)[0].reshape(-1, 2)
    uv = uv[torch.isfinite(uv).all(-1)]
    w, h = (float(x) for x in cam['size'])
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    return bool(len(uv)) and float(inside.float().mean()) >= min_inside


def camera_consensus(points, p2, v2, use, boxes, window_cams, res_max, vis_min):
    """R1: which pass-1 cameras disagree with the pose the cameras made together.

    Inputs: points -- (T,K,3) pass-1 world pose; p2 -- (C_use,T,K,2) each camera's own 2D head
        in SOURCE pixels; v2 -- (C_use,T,K) per-camera visibility logits or None; use / boxes --
        pass 1's cameras and crop boxes; res_max -- residual threshold in box sides; vis_min --
        visibility-logit threshold.
    Outputs: (res, vis, outlier), each (C_use,). `res` is the median over finite (frame, keypoint)
        of |project(points) - p2| in units of that camera's crop-box side; `vis` the median of
        its logits. A camera is an outlier when res > res_max or vis < vis_min; an unassessable
        statistic (NaN) never makes one, so a camera with nothing finite is left alone.

    A wrong-animal or truncated crop shows up here as a camera whose own 2D head describes a
    different body than the fused pose: the other cameras outvote it in the triangulation, so its
    residual is large, or its 2D head says the animal it was asked for is not there.
    """
    n = len(use)
    res = np.full(n, np.nan)
    vis = np.full(n, np.nan)
    for i, ci in enumerate(use):
        b = np.asarray(boxes[i], np.float64)
        side = 0.5 * ((b[2] - b[0]) + (b[3] - b[1]))
        if p2 is not None and side > 0:
            q = project_points_torch([window_cams[ci]],
                                     torch.as_tensor(points, dtype=torch.float32))[0].numpy()
            d = np.linalg.norm(q - np.asarray(p2[i], np.float64), axis=-1)
            d = d[np.isfinite(d)]
            if len(d):
                res[i] = float(np.median(d)) / side
        if v2 is not None:
            v = np.asarray(v2[i], np.float64)
            v = v[np.isfinite(v)]
            if len(v):
                vis[i] = float(np.median(v))
    with np.errstate(invalid='ignore'):
        outlier = (res > res_max) | (vis < vis_min)
    return res, vis, outlier


def refine_every_camera(points, use, boxes, cam_ix, window_cams, min_crop_dim, min_inside=0.5,
                        outliers=(), pad_frac=0.05):
    """TR1: pass-2 crops for EVERY calibrated camera from pass 1's 3D pose, tested per camera.

    Inputs: points -- (T,K,3) pass-1 world pose; use / boxes -- pass 1's cameras and crop
        boxes; cam_ix -- every camera index; window_cams -- the window's camera group;
        min_crop_dim -- the crop rule's floor; min_inside -- fraction of the finite projected
        points that must land inside the image for a camera pass 1 did not use.
        outliers -- pass-1 cameras `camera_consensus` rejected (R1): their pass-1 box is not
            evidence, so they take the in-frame test a camera without a pass-1 box takes.
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
        cg, bx = cropmod.crop_to_points_3d([cam], points, min_crop_dim, pad_frac=pad_frac)
        if cg is None:
            continue
        if ci in use and ci not in outliers:
            if not overlaps(bx[0], boxes[use.index(ci)]):
                continue
        elif not in_frame(points, cam, min_inside):
            continue
        use2.append(ci)
        boxes2.append(bx[0])
        cg2.append(cg[0])
    return use2, boxes2, cg2
