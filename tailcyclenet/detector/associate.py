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

import torch

from posetail.posetail.cube import project_points_torch


def _centres(boxes):
    """Box centres: (N,4) xyxy -> (N,2) (cx, cy)."""
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
    """Median reprojection residual in pixels of `p3d` into the given cameras.

    Inputs: cgroup -- the camera group; cams -- camera indices; pts -- (n_views, 2)
            observed centres; p3d -- (3,) world point.
    Outputs: float, the median of the per-view pixel distances.
    """
    proj = project_points_torch([cgroup[c] for c in cams], p3d.reshape(1, 1, 3))[:, 0, 0]
    return float(torch.linalg.norm(proj - pts, dim=-1).median())


def _support_count(cgroup, cams_excl, centres, used, p3d, max_res_px):
    """How many cameras OTHER than `cams_excl` have an UNUSED detection within `max_res_px` of
    `p3d`'s reprojection. detector_v2 plan E1a/E3.

    `used` is the SAME set `associate`'s own consumption loop checks -- built once, up front, from
    every non-finite centre (see `associate`'s own comment on it) and grown as groups are accepted.
    Filtering by it here means a detection already claimed by an earlier, higher-support group
    cannot corroborate a later, lower-support candidate for the same box twice.

    THE MECHANISM E1a REPLACES: `associate`'s candidate list used to sort by the SEEDING PAIR's
    own 2-view epipolar residual alone -- a number two rays can always satisfy near-exactly, so a
    phantom (animal A's ray in one view x animal B's ray in another) with a low pair residual was
    considered BEFORE a real pair and consumed greedily, turning one false positive into a false
    positive plus a miss plus an identity switch (plan SS1.3/SS2.2).

    WHY A COUNT, NOT A MEAN OR SUM (plan SS3.2.2): fusing corroboration as an average residual (or
    JARVIS's confidence-weighted mean) lets one very-corroborated view mask one or two views that
    do not corroborate at all -- exactly how a 2-camera phantom can look as good as a 3-camera real
    animal under a mean. An INTEGER count of independently-corroborating cameras is not fooled the
    same way: a ghost supported by 2 cameras needs a THIRD ray to also land near it by chance,
    which is a measure-zero coincidence unless cameras are near-collinear, where a real occluded
    animal only needs ITS OWN other views to have detected it at all.

    Only DISTANCE is used here (nearest live detection in that camera to the projected point),
    not yet a refit -- see plan SS2.2 item 3 ('growth still matches against the pair's un-refit
    p3d'), left as a separate, un-landed item.
    """
    count = 0
    for c in range(len(cgroup)):
        if c in cams_excl:
            continue
        idx = [i for i in range(centres[c].shape[0]) if (c, i) not in used]
        if not idx:
            continue
        proj = project_points_torch([cgroup[c]], p3d.reshape(1, 1, 3))[0, 0]
        d = torch.linalg.norm(centres[c][idx] - proj, dim=-1)
        if bool((d < max_res_px).any()):
            count += 1
    return count


def associate(cgroup, boxes_per_cam, max_res_px=30.0, min_views=2, max_instances=0,
             corroborate=True):
    """Group boxes across cameras into 3D instances.

    Inputs:
        cgroup -- posetail camera dicts, one per camera, in the SOURCE frame (not cropped).
        boxes_per_cam -- list of (N_c, 4) tensors, per camera, in that camera's pixels.
        max_res_px -- a group whose median reprojection residual exceeds this is rejected.
        min_views -- the floor on how many cameras an instance may be built from. 2 is the
            algorithm, not a parameter (every group starts from a cross-camera PAIR); `1`
            emits each leftover box as its own single-view instance with no triangulated point.
        max_instances -- cap on returned groups (0 = uncapped).
        corroborate -- plan E1a/E3: rank candidates by all-camera support count first, pair
            residual as tie-break, instead of the old seeding-pair-only epipolar residual
            (which a phantom can always satisfy). True (default); False restores the pre-E1a
            ordering for a same-checkpoint control arm.
    Outputs:
        List of dicts: {'point': (3,), 'boxes': {cam_ix: box}, 'residual': float,
        'members': {cam_ix: det_ix}}. A single-view instance has `point` all-NaN and
        `residual` inf -- nothing to triangulate from one ray.
    Side effects:
        None.
    Notes:
        A NaN box is "no detection here" and is seeded into `used` up front: `_triangulate`
        ends in SVD, which RAISES on non-finite input. The group is accepted on the residual of
        the PAIR that seeded it, then refit over all members; the refit gate is
        `not res_fit <= max_res_px`, so a degenerate NaN refit cannot pass.
    """
    assert min_views in (1, 2), f'min_views is 1 or 2, got {min_views}'
    centres = [_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
    n_cams = len(cgroup)

    used = {(c, i) for c in range(n_cams)
            for i in range(centres[c].shape[0]) if not bool(torch.isfinite(centres[c][i]).all())}

    cands = []
    for ca, cb in itertools.combinations(range(n_cams), 2):
        for ia in range(centres[ca].shape[0]):
            for ib in range(centres[cb].shape[0]):
                if (ca, ia) in used or (cb, ib) in used:
                    continue
                pts = torch.stack([centres[ca][ia], centres[cb][ib]])
                p3d = _triangulate(cgroup, (ca, cb), pts)
                if not torch.isfinite(p3d).all():
                    continue
                res = _residual(cgroup, (ca, cb), pts, p3d)
                support = (_support_count(cgroup, (ca, cb), centres, used, p3d, max_res_px)
                          if corroborate else 0)
                cands.append((support, res, (ca, ia), (cb, ib), p3d))
    cands.sort(key=lambda c: (-c[0], c[1]))
    out = []
    for _support, res, a, b, p3d in cands:
        if a in used or b in used or res > max_res_px:
            continue
        members = {a[0]: a[1], b[0]: b[1]}
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
        res_fit = _residual(cgroup, cams, pts, refined)
        if not res_fit <= max_res_px:
            continue
        out.append({'point': refined, 'residual': res_fit,
                    'boxes': {c: boxes_per_cam[c][members[c]] for c in cams},
                    'members': {c: members[c] for c in cams}})
        used.update((c, members[c]) for c in members)
        if max_instances and len(out) >= max_instances:
            break

    if min_views == 1:
        for c in range(n_cams):
            for i in range(centres[c].shape[0]):
                if max_instances and len(out) >= max_instances:
                    return out
                if (c, i) in used:
                    continue
                out.append({'point': torch.full((3,), float('nan')), 'residual': float('inf'),
                            'boxes': {c: boxes_per_cam[c][i]}, 'members': {c: i}})
                used.add((c, i))
    return out
