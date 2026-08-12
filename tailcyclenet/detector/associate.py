"""Cross-view association: per-camera boxes -> 3D instances.

A 3D pose needs to know that the box in `cam088` and the box in `cam091` are the SAME animal.
With one animal per view that is free; with ten pigeons and four cameras it is the whole problem.

The method: treat each box centre as a ray, triangulate every cross-camera pair, and keep the
groups whose triangulated point reprojects into every contributing view within
`assoc_res_max_px`. Greedy by residual, each box used once.

This is deliberately simple. It uses only the box CENTRE, so two animals that overlap in one
view and are separated in another will be resolved by the second view but not the first, and a
group of two views is accepted on a residual that two rays can always satisfy exactly -- hence
the `min_views` floor. The alternative (appearance features, or an epipolar-consistency solve
over all cameras at once) is a real piece of work and is not needed until a measurement says
association is the bottleneck.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from posetail.posetail.cube import project_points_torch


def _centres(boxes):
    return torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], -1)


def _triangulate(cgroup, cams, pts):
    """One point from `len(cams)` views. pts: (n_views, 2) in each camera's own pixels."""
    from posetail.posetail.cube import undistort_points

    mats, rays = [], []
    for ci, p in zip(cams, pts):
        cam = cgroup[ci]
        und = undistort_points(cam, p.reshape(1, 2).to(torch.float64))
        rays.append(torch.cat([und[0], und.new_ones(1)]))
        ext = cam['ext'][0] if cam['ext'].ndim == 3 else cam['ext']
        mats.append(ext.to(torch.float64))
    A = []
    for ray, M in zip(rays, mats):
        A.append(ray[0] * M[2] - ray[2] * M[0])
        A.append(ray[1] * M[2] - ray[2] * M[1])
    A = torch.stack(A)
    _, _, vh = torch.linalg.svd(A)
    x = vh[-1]
    return (x[:3] / x[3]).to(torch.float32)


def _residual(cgroup, cams, pts, p3d):
    proj = project_points_torch([cgroup[c] for c in cams], p3d.reshape(1, 1, 3))[:, 0, 0]
    return float(torch.linalg.norm(proj - pts, dim=-1).median())


def associate(cgroup, boxes_per_cam, max_res_px=30.0, min_views=2, max_instances=0):
    """Group boxes across cameras into 3D instances.

    Args:
        cgroup: posetail camera dicts, one per camera, in the SOURCE frame (not cropped)
        boxes_per_cam: list of (N_c, 4) tensors, per camera, in that camera's pixels
        max_res_px: a group whose median reprojection residual exceeds this is rejected

    Returns a list of dicts: {'point': (3,), 'boxes': {cam_ix: box}, 'residual': float}.
    """
    centres = [_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
    n_cams = len(cgroup)

    # Every cross-camera pair, scored by its own reprojection residual.
    cands = []
    for ca, cb in itertools.combinations(range(n_cams), 2):
        for ia in range(centres[ca].shape[0]):
            for ib in range(centres[cb].shape[0]):
                pts = torch.stack([centres[ca][ia], centres[cb][ib]])
                p3d = _triangulate(cgroup, (ca, cb), pts)
                if not torch.isfinite(p3d).all():
                    continue
                cands.append((_residual(cgroup, (ca, cb), pts, p3d), (ca, ia), (cb, ib), p3d))
    cands.sort(key=lambda c: c[0])

    used = set()
    out = []
    for res, a, b, p3d in cands:
        if a in used or b in used or res > max_res_px:
            continue
        members = {a[0]: a[1], b[0]: b[1]}
        # Grow the group with any unused box in another camera that agrees with the point.
        for c in range(n_cams):
            if c in members:
                continue
            best, best_res = None, max_res_px
            for i in range(centres[c].shape[0]):
                if (c, i) in used:
                    continue
                r = _residual(cgroup, (c,), centres[c][i].reshape(1, 2), p3d)
                if r < best_res:
                    best, best_res = i, r
            if best is not None:
                members[c] = best
        if len(members) < min_views:
            continue
        cams = tuple(sorted(members))
        pts = torch.stack([centres[c][members[c]] for c in cams])
        refined = _triangulate(cgroup, cams, pts)
        out.append({'point': refined, 'residual': _residual(cgroup, cams, pts, refined),
                    'boxes': {c: boxes_per_cam[c][members[c]] for c in cams}})
        used.update((c, members[c]) for c in members)
        if max_instances and len(out) >= max_instances:
            break
    return out
