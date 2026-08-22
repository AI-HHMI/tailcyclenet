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

        min_views: the floor on how many cameras an instance may be built from. **2 is not a
            parameter, it is the algorithm**: every group below starts from a cross-camera PAIR, so
            `len(members) >= 2` always and the check never fires. `min_views = 1` is therefore a
            different rule, not a looser threshold -- it emits each LEFTOVER box, one no pair
            claimed, as its own single-view instance with no triangulated point.

            **Whether the pose model can use that input is a property of the RUN, not of the
            model.** A single-view 3D window is the `prob_2d_only` path (`dataset.py`), and
            `configs/3d.toml` ships `prob_2d_only = 0`, so under it a one-camera instance is an
            UNTRAINED input shape. Check the run's own `[data].prob_2d_only` before reading
            anything into a `min_views = 1` arm. It cannot be free either: a leftover box is one
            the geometry never corroborated, so it is also where a false positive lives.
        `dup_res_px` WAS HERE and is deleted: under `--track` (the default) it produced a
            byte-identical prediction file because the tracker claims the leftovers this gate
            existed to police.

    Returns a list of dicts: {'point': (3,), 'boxes': {cam_ix: box}, 'residual': float}. A
    single-view instance has `point` all-NaN and `residual` inf: there is nothing to triangulate
    from one ray, and inventing a depth would be a position no camera claimed.
    """
    assert min_views in (1, 2), f'min_views is 1 or 2, got {min_views}'
    centres = [_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
    n_cams = len(cgroup)

    # A NaN BOX IS "NO DETECTION HERE", AND IT USED TO RAISE. `unletterbox_boxes` returns NaN for
    # a box the frame clamp left with no area, and `_triangulate` ends in `torch.linalg.svd`,
    # which RAISES on non-finite input -- so `detect_group` died mid-clip, after hours of decode,
    # on the 57k-frame group. `used` already means "this detection is not available", and both
    # loops below honour it, so seeding it here is the one place that covers every consumer.
    used = {(c, i) for c in range(n_cams)
            for i in range(centres[c].shape[0]) if not bool(torch.isfinite(centres[c][i]).all())}

    # Every cross-camera pair, scored by its own reprojection residual.
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
                cands.append((_residual(cgroup, (ca, cb), pts, p3d), (ca, ia), (cb, ib), p3d))
    cands.sort(key=lambda c: c[0])
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
        # THE REFIT RESIDUAL WAS COMPUTED, STORED, AND NEVER TESTED. The group is accepted on the
        # residual of the PAIR that seeded it -- a two-view reprojection residual, which is only an
        # epipolar statistic and which two rays can always satisfy -- and then grown by cameras
        # each accepted against the PAIR's point. Re-triangulating over all of them can land
        # somewhere none of the views agrees with; testing the number that was already here is the
        # one place the geometry can speak with more than two rays.
        res_fit = _residual(cgroup, cams, pts, refined)
        # NOT `res_fit > max_res_px`: that is False for NaN, so a degenerate refit passed the gate
        # and planted an instance whose `point` is non-finite -- which `CrossViewTracker` then
        # holds forever (a target with a non-finite point is filtered out of `slots`, so it is
        # never matched, never aged and never retired, and its row goes dead for the clip).
        if not res_fit <= max_res_px:
            continue
        out.append({'point': refined, 'residual': res_fit,
                    'boxes': {c: boxes_per_cam[c][members[c]] for c in cams},
                    # WHICH detection in each camera, so a caller holding a per-camera score array
                    # can follow it through the association instead of re-deriving the match.
                    'members': {c: members[c] for c in cams}})
        used.update((c, members[c]) for c in members)
        if max_instances and len(out) >= max_instances:
            break

    if min_views == 1:
        # Whatever no pair claimed, in camera order then score order (the boxes arrive score-ordered
        # from `decode`). Deterministic, so two arms over one clip see the same rows.
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
