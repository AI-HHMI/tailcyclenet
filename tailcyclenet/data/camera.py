"""Image-plane camera transforms shared by the loader and inference: resize, 2D rotate/flip (with the
2x3 affine the pixels render through), point-through-affine, and the out-of-image masks.

A camera is tracktail's dict (`mat`, `offset`, `size`, `dist`, `ext`, ...); every transform returns a
new dict and never mutates its input.
"""
from __future__ import annotations

import numpy as np
import torch

from tracktail.model.cube import is_point_visible


def rotate_2d(cam, coords, angle_deg):
    """`rotate_points_image_plane` WITHOUT its border-free inscribed crop. Same return shape.

    The inscribed crop silently NaN'd the labels of items whose animal sat outside it (the 3D
    path stays on the library helper because it recomputes `vis_2d` after rotating, which 2D
    cannot). The full canvas costs nothing: the follow-on crop only meets the canvas edge for a
    frame-edge animal, where `crop_affine` already renders BORDER_CONSTANT zeros.

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


def flip_2d(cam, coords, mode, prior_rotation=None, perm=None):
    """Reflect one 2D camera on its current source canvas and permute semantic keypoints.

    The reflection is composed into the source-to-canvas affine so image pixels, crop geometry,
    camera metadata and labels stay in one coordinate frame. ``W - 1 - x`` and ``H - 1 - y`` are
    pixel-center conventions, matching ``cv2.flip``. The intrinsic matrix is left-multiplied by
    the reflection, so 3D projection remains consistent as well; because the reflection acts after
    distortion, the distortion coefficients remain unchanged. ``perm`` is a session-axis
    involution; when omitted, the helper only performs the geometric transform for callers testing
    the affine.
    """
    if mode not in ('horizontal', 'vertical'):
        raise ValueError(f'2D flip mode must be "horizontal" or "vertical", got {mode!r}')
    W, H = (int(v) for v in cam['size'][:2])
    if mode == 'horizontal':
        F = np.array([[-1.0, 0.0, W - 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    else:
        F = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, H - 1.0]], dtype=np.float64)

    if prior_rotation is None:
        M_prior = np.eye(3, dtype=np.float64)
    else:
        M_prior_2x3 = np.asarray(prior_rotation[0], dtype=np.float64)
        if M_prior_2x3.shape != (2, 3):
            raise ValueError(f'2D rotation must be a 2x3 affine, got {M_prior_2x3.shape}')
        M_prior = np.vstack([M_prior_2x3, [0.0, 0.0, 1.0]])
    F_h = np.vstack([F, [0.0, 0.0, 1.0]])
    composed = (F_h @ M_prior)[:2]

    out = dict(cam)
    out['mat'] = cam['mat'].clone()
    out['offset'] = cam['offset'].clone()
    F_t = torch.as_tensor(F, dtype=cam['mat'].dtype, device=cam['mat'].device)
    out['mat'][:2, :2] = F_t[:, :2] @ cam['mat'][:2, :2]
    pp = cam['mat'][:2, 2] - cam['offset']
    pp_flip = F_t @ torch.cat([pp, torch.ones(1, dtype=pp.dtype, device=pp.device)])
    out['mat'][:2, 2] = pp_flip + cam['offset']
    Ft = torch.as_tensor(F, dtype=coords.dtype, device=coords.device)
    transformed = coords @ Ft[:, :2].T + Ft[:, 2]
    if perm is not None:
        perm = torch.as_tensor(perm, dtype=torch.long, device=coords.device)
        transformed = transformed[:, perm]
    return out, transformed, (composed, (W, H))


def apply_affine(pts, rotation):
    """Move (...,2) pixel points through a rotation/reflection's own 2x3, or pass through untouched.

    Shared so a stored box and the labels can never end up in different frames.
    """
    if pts is None or rotation is None:
        return pts
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


def mask_outside(coords, size):
    """A point outside the image is not a label. Drops it rather than clamping it to the edge."""
    w, h = float(size[0]), float(size[1])
    bad = ((coords[..., 0] < 0) | (coords[..., 0] >= w) |
           (coords[..., 1] < 0) | (coords[..., 1] >= h))
    coords = coords.clone()
    coords[bad] = float('nan')
    return coords


def resize_camera(cam, target_res):
    """Scale a camera so its long side is `target_res`. Returns (cam, (sx, sy)).

    The output dimensions are integer pixels, so the affine actually applied by OpenCV has
    independently rounded x/y scales. Return those exact scales and apply them to the complete
    intrinsic matrix and offset; using the pre-rounding scalar leaves projections a fraction of a
    pixel away from the warped image on nonsquare crops.
    """
    cam = dict(cam)
    size = cam['size']
    nominal = float(target_res) / float(max(size))
    new_size = torch.round(size * nominal).to(torch.int32).clamp_min(1)
    scale = new_size.to(dtype=cam['mat'].dtype) / size.to(dtype=cam['mat'].dtype)
    cam['size'] = new_size
    cam['mat'] = cam['mat'].clone()
    cam['mat'][0] *= scale[0]
    cam['mat'][1] *= scale[1]
    cam['mat'][2, 2] = 1
    cam['offset'] = cam['offset'] * scale
    return cam, scale
