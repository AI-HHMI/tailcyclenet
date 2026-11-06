"""Association: per-camera detections -> one row per animal.

`associate` groups one frame's boxes across cameras into 3D instances; `associate_group` runs it
(or `CrossViewTracker`, or 2D `link_rows`) over a whole clip.

A 3D pose needs to know that the box in `cam088` and the box in `cam091` are the SAME animal.
With one animal per view that is free; with ten pigeons and four cameras it is the whole problem.

The method: treat each box centre as a ray, triangulate every cross-camera pair, and keep the
groups whose triangulated point reprojects into every contributing view within
`ASSOC_RES_MAX_BODY` (world distance / body size). Greedy by residual, each box used once.

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

from tracktail.model.cube import project_points_torch

from ..detector.boxes import centres as box_centres

ASSOC_RES_MAX_BODY = 0.15
"""The default cross-view gate in BODY units (owner decision 2026-11-05): every association residual is a WORLD
distance (point-to-ray, or ray-to-ray for a seeding pair) divided by the session's body size. In those units
the true same-animal residuals agree across rigs (pair ray-distance p99 0.13-0.19 body on CMU Panoptic and
3dpop; third-view growth p90 0.10-0.16), while in pixels the same noise spans 19x across rigs and in box sides
6x. 0.15 = the third-view growth p90 on 3dpop + Panoptic, confirmed on 9 single-animal rigs. No pixel gate
constant remains: `associate`'s pixel engine runs only on an explicit `max_res_px`. Report 76, '### detr3d-diag'."""


def _triangulate(cgroup, cams, pts):
    """One point from `len(cams)` views. pts: (n_views, 2) in each camera's own pixels."""
    from tracktail.model.cube import undistort_points

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

    `used` is `associate`'s set as it stands while candidates are RANKED: the non-finite
    centres only. Support is counted once per candidate, before any group is accepted, so a
    count can include a detection that a higher-ranked group later consumes; the consumption
    loop (not this count) is what keeps each detection in one group.

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


def associate(cgroup, boxes_per_cam, max_res_px=None, min_views=2, max_instances=0,
              scores_per_cam=None, score_order=False, fast=True,
              random_drop=None, max_res_body=None, body_size=None):
    """Group boxes across cameras into 3D instances.

    Inputs:
        cgroup -- tracktail camera dicts, one per camera, in the SOURCE frame (not cropped).
        boxes_per_cam -- list of (N_c, 4) tensors, per camera, in that camera's pixels.
        max_res_px -- the PIXEL engine's gate (explicit only; no default): a group whose median
            reprojection residual exceeds it is rejected. Unused under `max_res_body`.
        min_views -- the floor on how many cameras an instance may be built from. 2 is the
            algorithm, not a parameter (every group starts from a cross-camera PAIR); `1`
            emits each leftover box as its own single-view instance with no triangulated point.
        max_instances -- cap on returned groups (0 = uncapped).
        scores_per_cam / score_order -- TR6a: rank by (-support, -pair_score, residual),
            pair_score = the lower box score of the pair; False = (-support, residual).
        fast -- TR7 (`_fast_candidates`): the same groups with less work. Default False.
        random_drop -- (p, rng) or None: DIAGNOSTIC rule-14 control, drops passing groups.
        max_res_body / body_size -- the BODY-UNIT gate (`_associate_body`); None = the pixel gate.
    Candidates rank by all-camera support count first, pair residual as tie-break (plan
    E1a/E3), never by the seeding pair's residual alone (which a phantom can always satisfy).
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
    if score_order and scores_per_cam is None:
        raise ValueError('score_order needs scores_per_cam')
    if max_res_body is not None:
        if body_size is None or not float(body_size) > 0:
            raise ValueError(f'max_res_body needs a positive body_size, got {body_size!r}')
        return _associate_body(cgroup, boxes_per_cam, float(max_res_body) * float(body_size),
                               min_views, max_instances, random_drop, scores_per_cam, score_order)
    if max_res_px is None:
        raise ValueError('associate needs max_res_body + body_size (the body-unit gate) or an explicit '
                         'max_res_px (the pixel engine); there is no default pixel gate')
    centres = [box_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
    n_cams = len(cgroup)

    used = {(c, i) for c in range(n_cams)
            for i in range(centres[c].shape[0]) if not bool(torch.isfinite(centres[c][i]).all())}

    def res(cams, pts, p3d, idx):
        """The gated residual in pixels (`idx`, the members, is unused on this path)."""
        return _residual(cgroup, cams, pts, p3d)

    if fast:
        cands, proj_of, triangulate = _fast_candidates(
            cgroup, centres, used, max_res_px, scores_per_cam, score_order, res)
    else:
        proj_of, triangulate = None, (lambda cams, pts, members: _triangulate(cgroup, cams, pts))
        cands = _candidates(cgroup, centres, used, max_res_px, scores_per_cam, score_order, res)
    if score_order:
        cands.sort(key=lambda c: (-c[0], -c[5], c[1]))
    else:
        cands.sort(key=lambda c: (-c[0], c[1]))
    out = []
    for _support, r, a, b, p3d, _pair in cands:
        if a in used or b in used or r > max_res_px:
            continue
        members = {a[0]: a[1], b[0]: b[1]}
        proj = {} if proj_of is not None else None
        for c in range(n_cams):
            if c in members:
                continue
            best, best_res = None, max_res_px
            for i in range(centres[c].shape[0]):
                if (c, i) in used:
                    continue
                if proj is None:
                    ri = res((c,), centres[c][i].reshape(1, 2), p3d, (i,))
                else:
                    if c not in proj:
                        proj[c] = proj_of(c, p3d)
                    d = torch.linalg.norm(proj[c] - centres[c][i].reshape(1, 2), dim=-1)
                    ri = float(d.median())
                if ri < best_res:
                    best, best_res = i, ri
            if best is not None:
                members[c] = best
        if len(members) < min_views:
            continue
        cams = tuple(sorted(members))
        pts = torch.stack([centres[c][members[c]] for c in cams])
        refined = triangulate(cams, pts, tuple(members[c] for c in cams))
        res_fit = res(cams, pts, refined, tuple(members[c] for c in cams))
        if not res_fit <= max_res_px:
            continue
        if random_drop is not None and random_drop[1].random() < random_drop[0]:
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



def _associate_body(cgroup, boxes_per_cam, gate, min_views=2, max_instances=0, random_drop=None,
                    scores_per_cam=None, score_order=False):
    """`associate` with every residual in WORLD units (the body-unit gate; `gate` = max_res_body x body_size).

    The algorithm is the pixel path's: seed from a cross-camera pair, rank by support then the pair's residual,
    grow one view per other camera, refit over all members, accept if the refit passes. Only the residual
    changes: a seeding pair's is the closest approach of its two back-projected box-centre rays; support,
    growth and refit use the point-to-ray distance of a view's ray to the (pair / refined) triangulated point.
    In those units a box centre that is not the projection of a fixed 3D point (tall bodies, truncation at
    the image border) no longer inflates the third-view test the way a pixel reprojection of a 2-view point
    does (report 76: Panoptic third-view residuals p50 42-53 px, 68-74% over 30 px).
    float64 numpy throughout; `fast` does not apply. TR6a `score_order` ranks by (support, the pair's lower
    box score, residual) exactly as on the pixel path. Same output dicts as `associate`, with
    `residual` the refit's median point-to-ray distance in world units.
    """
    from tracktail.model.cube import undistort_points

    n_cams = len(cgroup)
    mats = [(c['ext'][0] if c['ext'].ndim == 3 else c['ext']).double().numpy() for c in cgroup]
    cen = [box_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
    keys = [(c, i) for c in range(n_cams) for i in range(cen[c].shape[0])
            if bool(torch.isfinite(cen[c][i]).all())]
    out = []
    used = set()
    if keys:
        rn = {k: undistort_points(cgroup[k[0]], cen[k[0]][k[1]].reshape(1, 2).double())[0].numpy()
              for k in keys}
        origin, direction = {}, {}
        for k in keys:
            m = mats[k[0]]
            rot, tr = m[:3, :3], m[:3, 3]
            origin[k] = -rot.T @ tr
            d = rot.T @ np.r_[rn[k], 1.0]
            direction[k] = d / np.linalg.norm(d)

        def tri(ks):
            """DLT over the cached undistorted rays of detections `ks`."""
            rows = []
            for (c, i) in ks:
                r, m = rn[(c, i)], mats[c]
                rows.append(r[0] * m[2] - m[0])
                rows.append(r[1] * m[2] - m[1])
            _, _, vh = np.linalg.svd(np.stack(rows))
            x = vh[-1]
            return x[:3] / x[3]

        def ray_ray(a, b):
            """Closest approach of the two detections' back-projected rays (world units)."""
            n = np.cross(direction[a], direction[b])
            nn = np.linalg.norm(n)
            return float(abs(np.dot(origin[a] - origin[b], n)) / max(nn, 1e-12))

        def point_ray(x, k):
            """Distance of world point `x` from detection `k`'s back-projected ray."""
            v = x - origin[k]
            return float(np.linalg.norm(v - np.dot(v, direction[k]) * direction[k]))

        cands = []
        for a, b in itertools.combinations(keys, 2):
            if a[0] == b[0]:
                continue
            x = tri([a, b])
            if not np.isfinite(x).all():
                continue
            r = ray_ray(a, b)
            if r > gate:
                continue
            support = sum(1 for c in range(n_cams) if c not in (a[0], b[0])
                          and any(point_ray(x, k) <= gate for k in keys if k[0] == c))
            pair = (min(float(scores_per_cam[a[0]][a[1]]), float(scores_per_cam[b[0]][b[1]]))
                    if score_order else 0.0)
            cands.append((support, r, a, b, x, pair))
        if score_order:
            cands.sort(key=lambda z: (-z[0], -z[5], z[1]))
        else:
            cands.sort(key=lambda z: (-z[0], z[1]))
        for _support, _r, a, b, x, _pair in cands:
            if a in used or b in used:
                continue
            members = {a[0]: a, b[0]: b}
            for c in range(n_cams):
                if c in members:
                    continue
                ds = sorted((point_ray(x, k), k) for k in keys if k[0] == c and k not in used)
                if ds and ds[0][0] <= gate:
                    members[c] = ds[0][1]
            if len(members) < min_views:
                continue
            ks = [members[c] for c in sorted(members)]
            xr = tri(ks)
            res_fit = float(np.median([point_ray(xr, k) for k in ks]))
            if not res_fit <= gate:
                continue
            if random_drop is not None and random_drop[1].random() < random_drop[0]:
                continue
            out.append({'point': torch.as_tensor(xr, dtype=torch.float32), 'residual': res_fit,
                        'boxes': {c: boxes_per_cam[c][members[c][1]] for c in sorted(members)},
                        'members': {c: members[c][1] for c in sorted(members)}})
            used.update(ks)
            if max_instances and len(out) >= max_instances:
                break
    if min_views == 1:
        for c in range(n_cams):
            for i in range(cen[c].shape[0]):
                if max_instances and len(out) >= max_instances:
                    return out
                if (c, i) in used or not bool(torch.isfinite(cen[c][i]).all()):
                    continue
                out.append({'point': torch.full((3,), float('nan')), 'residual': float('inf'),
                            'boxes': {c: boxes_per_cam[c][i]}, 'members': {c: i}})
                used.add((c, i))
    return out

def _candidates(cgroup, centres, used, max_res_px, scores_per_cam, score_order, res):
    """`associate`'s reference candidate list: every cross-camera pair of unused detections."""
    n_cams = len(cgroup)
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
                r = res((ca, cb), pts, p3d, (ia, ib))
                support = _support_count(cgroup, (ca, cb), centres, used, p3d, max_res_px)
                pair = (min(float(scores_per_cam[ca][ia]), float(scores_per_cam[cb][ib]))
                        if score_order else 0.0)
                cands.append((support, r, (ca, ia), (cb, ib), p3d, pair))
    return cands


def _fast_candidates(cgroup, centres, used, max_res_px, scores_per_cam, score_order, res):
    """TR7: `_candidates` with the same output, computed with less work.

    Three savings, each exact: (1) a detection's undistorted ray is computed ONCE per frame,
    not once per pair it joins (the same `undistort_points` call on the same input, so the
    same bits); (2) a pair whose own residual already fails the gate is dropped before its
    support is counted -- the acceptance loop skips it unconditionally, so it can neither
    consume a detection nor change the order of the pairs that can; (3) a candidate's
    projection into a camera is computed once and reused by the growth step (`proj_of`).
    Returns (cands, proj_of, triangulate) for `associate`'s acceptance loop.
    """
    from tracktail.model.cube import undistort_points

    n_cams = len(cgroup)
    mats = []
    for cam in cgroup:
        ext = cam['ext'][0] if cam['ext'].ndim == 3 else cam['ext']
        mats.append(ext.to(torch.float64))
    rays = {}
    for c in range(n_cams):
        for i in range(centres[c].shape[0]):
            if (c, i) in used:
                continue
            und = undistort_points(cgroup[c], centres[c][i].reshape(1, 2).to(torch.float64))
            rays[(c, i)] = torch.cat([und[0], und.new_ones(1)])

    def tri(cams, idx):
        """`_triangulate` on cached rays: the same rows, the same SVD."""
        A = []
        for c, i in zip(cams, idx):
            ray, M = rays[(c, i)], mats[c]
            A.append(ray[0] * M[2] - ray[2] * M[0])
            A.append(ray[1] * M[2] - ray[2] * M[1])
        _, _, vh = torch.linalg.svd(torch.stack(A))
        x = vh[-1]
        return (x[:3] / x[3]).to(torch.float32)

    def proj_of(c, p3d):
        """`_residual`'s own projection of `p3d` into camera `c`, computed once per candidate."""
        return project_points_torch([cgroup[c]], p3d.reshape(1, 1, 3))[:, 0, 0]

    cands = []
    for ca, cb in itertools.combinations(range(n_cams), 2):
        for ia in range(centres[ca].shape[0]):
            for ib in range(centres[cb].shape[0]):
                if (ca, ia) in used or (cb, ib) in used:
                    continue
                pts = torch.stack([centres[ca][ia], centres[cb][ib]])
                p3d = tri((ca, cb), (ia, ib))
                if not torch.isfinite(p3d).all():
                    continue
                r = res((ca, cb), pts, p3d, (ia, ib))
                if r > max_res_px:
                    continue
                support = _support_count(cgroup, (ca, cb), centres, used, p3d, max_res_px)
                pair = (min(float(scores_per_cam[ca][ia]), float(scores_per_cam[cb][ib]))
                        if score_order else 0.0)
                cands.append((support, r, (ca, ia), (cb, ib), p3d, pair))
    return cands, proj_of, (lambda cams, pts, members: tri(cams, members))



def provisional_body_size(raw_box, raw_sc, cgroup_at, frames, gate=ASSOC_RES_MAX_BODY):
    """A body size from DETECTIONS alone (no pose, no pixel gate): the bootstrap of the body-unit gate.

    For each listed frame and camera pair, every pair of detections is triangulated (2-view DLT of the box
    centres) and given a box-implied size s = mean over its two views of (box diagonal px x depth / focal);
    a pair is kept when it is MUTUAL nearest in that camera pair under the scale-free ratio
    (ray-ray closest approach) / s and that ratio is <= `gate`. -> (median s over kept pairs, n_kept); NaN when
    nothing passes. Measured vs the GT extent (report 76 detr3d-diag): x0.99 CMU Panoptic, x1.09 / x1.22
    social-dannce, x1.29 / x1.36 3dpop (box diagonals overstate pigeons), x0.94 / x1.09 single-animal
    qdmouse4m / rat7m -- inside the body gate's flat band (+-20-30%), which is all a bootstrap needs.
    `cgroup_at(t)` is the frame's camera group; `raw_box` (D, T, C, 4), `raw_sc` (D, T, C) as `detect_raw`.
    """
    from tracktail.model.cube import undistort_points

    vals = []
    for t in frames:
        cg = cgroup_at(int(t))
        n_cams = len(cg)
        mats = [(c['ext'][0] if c['ext'].ndim == 3 else c['ext']).double().numpy() for c in cg]
        focal = [float((c['mat'][0, 0] + c['mat'][1, 1]) / 2) for c in cg]
        views = {}
        for c in range(n_cams):
            ok = np.isfinite(raw_sc[:, t, c]) & np.isfinite(raw_box[:, t, c]).all(-1)
            b = np.asarray(raw_box[ok, t, c], np.float64)
            if not len(b):
                continue
            cen = (b[:, :2] + b[:, 2:]) / 2
            rn = undistort_points(cg[c], torch.as_tensor(cen, dtype=torch.float64)).numpy()
            rot, tr = mats[c][:3, :3], mats[c][:3, 3]
            d = (rot.T @ np.c_[rn, np.ones(len(rn))].T).T
            views[c] = (rn, -rot.T @ tr, d / np.linalg.norm(d, axis=1, keepdims=True),
                        np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))
        for ca, cb in itertools.combinations(sorted(views), 2):
            ra, oa, da, ga = views[ca]
            rb, ob, db, gb = views[cb]
            ratio = np.full((len(ra), len(rb)), np.inf)
            size = np.full((len(ra), len(rb)), np.nan)
            for i in range(len(ra)):
                for j in range(len(rb)):
                    rows = [ra[i][0] * mats[ca][2] - mats[ca][0], ra[i][1] * mats[ca][2] - mats[ca][1],
                            rb[j][0] * mats[cb][2] - mats[cb][0], rb[j][1] * mats[cb][2] - mats[cb][1]]
                    x = np.linalg.svd(np.stack(rows))[2][-1]
                    x = x[:3] / x[3]
                    za = (mats[ca][:3, :3] @ x + mats[ca][:3, 3])[2]
                    zb = (mats[cb][:3, :3] @ x + mats[cb][:3, 3])[2]
                    if not (za > 0 and zb > 0):
                        continue
                    s = (ga[i] * za / focal[ca] + gb[j] * zb / focal[cb]) / 2
                    n = np.cross(da[i], db[j])
                    rr = abs(np.dot(oa - ob, n)) / max(np.linalg.norm(n), 1e-12)
                    ratio[i, j], size[i, j] = rr / s, s
            for i in range(len(ra)):
                j = int(np.argmin(ratio[i]))
                if np.isfinite(ratio[i, j]) and int(np.argmin(ratio[:, j])) == i and ratio[i, j] <= gate:
                    vals.append(size[i, j])
    return (float(np.median(vals)) if vals else float('nan')), len(vals)


def associate_group(raw, session, gid, max_instances, link=False, min_views=2,
                    track=True, max_move=1.25, max_age=8, stats=None,
                    state=None, duplicate_radius=0.75, duplicate_persist=5, frame_base=0,
                    assoc_score_order=False,
                    track_age_decay=None, track_affinity_2d=0.0, track_camera_claims=False,
                    assoc_fast=True, assoc_random_drop=0.0, assoc_res_max_body=ASSOC_RES_MAX_BODY,
                    body_size=None, link_two_stage_2d=True):
    """The ASSOCIATION half: per-camera detections -> ONE ROW PER ANIMAL. Microseconds per frame.

    Inputs:
        raw -- `detect_raw`'s (boxes, scores[, extents]); session, gid, max_instances (rows S).
        track -- 3D multiview default: `CrossViewTracker`, one cross-view target set carried
            across frames; `False` restores the memoryless per-frame `associate`.
        link -- tracker-less linking: 2D `link_two_stage` (link_two_stage_2d=False: `link_rows`).
        min_views / max_move -- passed through to the tracker / `associate`; max_move defaults
            to the measured 1.25 box-side gate.
        max_age -- the tracker's and `link_rows`' SHARED patience window: frames without
            evidence before a slot or row is retired. The measured default is 8.
        stats -- accumulates association counters when given.
        state -- makes calls equal to one concatenated call; `None` builds fresh state.
        duplicate_radius / duplicate_persist -- the 3D --track backstop's duplicate-retirement
            geometry; inert without a tracker (2D or track=False).
        frame_base -- the SOURCE frame of `raw`'s own frame 0, threaded to a freshly built
            tracker so its `identity_events` are source-absolute even on a ranged run whose
            tracker is first built on an aligned lead-in batch; inert once `state` already
            holds one (the tracker is built exactly once).
        assoc_score_order -- `associate`'s `score_order` (TR6a), with or without a tracker.
        track_age_decay / track_affinity_2d / track_camera_claims -- T2 / TR3 / TR2 (tracker).
        assoc_fast -- `associate`'s `fast` (TR7): identical groups with less work.
        assoc_random_drop -- DIAGNOSTIC rule-14 control: drop passing groups at this rate.
        assoc_res_max_body / body_size -- the body gate; body_size None = `provisional_body_size` from
            these detections (<= 48 frames, once per `state`, `stats['body_provisional']`; none -> no rows).
    Outputs:
        The same pair re-indexed so row `a` is one animal -- across cameras always, and across
        frames wherever a tracker or `link_rows` ran. In 2D the row index is the only identity.
    Notes:
        Cross-view geometry is built once for a static rig, per frame for a moving one.
    """
    import numpy as np
    import torch

    r_box, r_sc = raw[0], raw[1]
    r_ext = raw[2] if len(raw) > 2 else None
    D, T, C = r_sc.shape
    S = max_instances or 1
    out = np.full((S, T, C, 4), np.nan, np.float32)
    sc = np.full((S, T, C), np.nan, np.float32)
    moving = any(session.rig.moving.values())
    cgroup = None if moving else session.cgroup(gid)
    if state is None:
        state = {}
    if assoc_res_max_body is None:
        assoc_res_max_body = ASSOC_RES_MAX_BODY
    if C > 1 and body_size is None:
        body_size = state.get('body_provisional')
        if body_size is None:
            pick = np.unique(np.linspace(0, T - 1, min(T, 48)).round().astype(int)) if T else []
            est, n_pairs = provisional_body_size(
                r_box, r_sc, (lambda t: session.cgroup(gid, frame_base + t)) if moving else (lambda t: cgroup),
                pick)
            if stats is not None:
                stats['body_provisional'] = est
                stats['body_provisional_pairs'] = n_pairs
            if not est == est:
                return out, sc
            body_size = state['body_provisional'] = est
    if 'tracker' not in state:
        tracker = None
        if track and C > 1:
            from .track import CrossViewTracker
            tracker = CrossViewTracker(S,
                                       min_views=min_views,
                                       max_move=max_move, max_age=max_age,
                                       duplicate_radius=duplicate_radius,
                                       duplicate_persist=duplicate_persist,
                                       frame_base=frame_base,
                                       score_order=assoc_score_order,
                                       age_decay=track_age_decay,
                                       affinity_2d=track_affinity_2d,
                                       camera_claims=track_camera_claims,
                                       fast=assoc_fast,
                                       random_drop=assoc_random_drop,
                                       max_res_body=assoc_res_max_body,
                                       body_size=body_size)
        state['tracker'] = tracker
    tracker = state['tracker']

    def _cam(t, c):
        """This frame-camera's decoded detections as torch, plus their raw indices.

        LIVENESS IS THE SCORE, NOT THE BOX: `unletterbox_boxes` returns NaN for a detection clamped
        to no positive area, so filtering on the box would silently drop detections the unsplit
        pass kept, shifting every later index and with it the `claimed` gather.
        """
        ok = np.flatnonzero(np.isfinite(r_sc[:, t, c]))
        return torch.from_numpy(r_box[ok, t, c]), torch.from_numpy(r_sc[ok, t, c]), ok

    import time

    t_start = time.perf_counter()
    raw_offered = 0
    for t in range(T):
        per_cam = [_cam(t, c) for c in range(C)]
        raw_offered += sum(len(p[2]) for p in per_cam)
        if C == 1:
            b, s, ok = per_cam[0]
            n = min(S, len(ok))
            out[:n, t, 0] = b[:n].numpy()
            sc[:n, t, 0] = s[:n].numpy()
            continue
        cams = session.cgroup(gid, frame_base + t) if moving else cgroup
        if tracker is not None:
            out[:, t], sc[:, t], _ = tracker.step(
                cams, [p[0] for p in per_cam], [p[1] for p in per_cam])
            continue
        groups = associate(cams, [p[0] for p in per_cam],
                           max_instances=S,
                           min_views=min_views,
                           scores_per_cam=[p[1] for p in per_cam],
                           score_order=assoc_score_order, fast=assoc_fast,
                           random_drop=(None if not assoc_random_drop else
                                        (assoc_random_drop, state.setdefault(
                                            'drop_rng', np.random.default_rng(0)))),
                           max_res_body=assoc_res_max_body, body_size=body_size)
        if stats is not None:
            stats['association_groups'] = stats.get('association_groups', 0) + len(groups)
        for a, g in enumerate(groups[:S]):
            for c, box in g['boxes'].items():
                d = g['members'][c]
                out[a, t, c] = box.numpy()
                sc[a, t, c] = float(per_cam[c][1][d])
    if stats is not None:
        stats['association_s'] = stats.get('association_s', 0.0) + time.perf_counter() - t_start
    if stats is not None and tracker is not None:
        stats['association_groups'] = tracker.n_groups
        stats['association_camera_claims'] = tracker.n_camera_claims
    pre_link = int(np.isfinite(out).all(-1).sum())
    if link and tracker is None and C == 1 and link_two_stage_2d:
        rows, rsc = link_two_stage(r_box[:, :, 0], r_sc[:, :, 0], S,
                                   extents=None if r_ext is None else r_ext[:, :, 0],
                                   max_move=max_move, max_age=max_age,
                                   state=state.setdefault('link2', {}))
        out, sc = rows[:, :, None].astype(np.float32), rsc[:, :, None].astype(np.float32)
    elif link and tracker is None:
        out, sc = link_rows(out, sc, max_move=max_move, max_age=max_age,
                            state=state.setdefault('link', {}))
    if stats is not None:
        stats['association_raw_offered'] = stats.get('association_raw_offered', 0) + raw_offered
        stats['association_pre_link'] = stats.get('association_pre_link', 0) + pre_link
    if stats is not None:
        stats['association_kept'] = stats.get('association_kept', 0) + \
            int(np.isfinite(out).all(-1).sum())

    return out, sc


LINK_HI_FRAC = 0.57
"""Two-stage 2D linking: a top-S box opens or continues a row on its own only when its score is at least
this fraction of the clip's reference score (the median S-th-ranked score over the first `LINK_CALIB_FRAMES`)."""
LINK_LO_SCORE = 0.05
"""Two-stage 2D linking: the floor for second-stage boxes, which may only EXTEND a live row."""
LINK_TOP = 20
"""Two-stage 2D linking: ranks considered per frame (max(LINK_TOP, S)); `detect_raw` must return at least this many."""
LINK_CALIB_FRAMES = 500
"""Two-stage 2D linking: frames whose S-th-ranked score form the reference (causal: a running median until full)."""


def link_two_stage(boxes, scores, n_rows, extents=None, max_move=1.25, max_age=8, state=None,
                   hi_frac=LINK_HI_FRAC, lo_score=LINK_LO_SCORE, top=LINK_TOP,
                   calib_frames=LINK_CALIB_FRAMES):
    """2D (one camera) score-aware two-stage linking: ranked detections -> `n_rows` rows, one animal each.

    Inputs:
        boxes (D,T,4), scores (D,T) -- `detect_raw`'s ranked detections for ONE camera (crop-rule boxes).
        n_rows -- S, the row count.
        extents (D,T,4) -- the detector head's TIGHT boxes, which the match is computed on (a crop
            box's pad and floor blur centres and sides at contact); None = link on `boxes`.
        max_move / max_age -- `link_rows`' gate (box sides) and patience.
        state -- dict carried across calls so blocks link exactly as one causal pass.
    Outputs:
        (rows (S,T,4), row_scores (S,T)): the chosen detections' CROP boxes and scores; NaN = no box.
    Notes:
        Stage 1 matches live rows (last known tight box) to the top-S boxes whose score is at least
        `hi_frac` x the reference score -- the median S-th-ranked score over the first `calib_frames`
        frames (a running median until then). Stage 2 matches the rows still unmatched to every other
        box of rank < max(top, S) with score >= `lo_score`; such a box only EXTENDS a live row and
        never opens one. Unclaimed stage-1 boxes open empty rows. The cost is `link_rows`' centre
        distance in box sides, gated at `max_move`.
        Why (report 76 relink): low-score top-S boxes are contact duplicates and half-boxes and drive
        ~40% of row swaps on crowded look-alike mice; promoting a rate-matched random set instead is null.
    """
    from scipy.optimize import linear_sum_assignment

    from ..detector.boxes import centres as box_centres
    from ..detector.boxes import sides as box_sides

    S = int(n_rows)
    D, T, _ = boxes.shape
    K = min(D, max(int(top), S))
    ext = boxes if extents is None else extents
    if state is None:
        state = {}
    if 'last' not in state:
        state.update(last=np.full((S, 4), np.nan), age=np.zeros(S, int), calib=[], started=False)
    last, age, calib = state['last'], state['age'], state['calib']
    rows = np.full((S, T, 4), np.nan, np.float32)
    rsc = np.full((S, T), np.nan, np.float32)
    rank = np.arange(K)
    for t in range(T):
        s = scores[:K, t]
        if len(calib) < calib_frames and S - 1 < K and np.isfinite(s[S - 1]):
            calib.append(float(s[S - 1]))
        ref = float(np.median(calib)) if calib else 0.0
        s_hi = max(hi_frac * ref, 0.01)
        e = ext[:K, t]
        ok = np.isfinite(s) & np.isfinite(e).all(-1)
        hi_ix = np.flatnonzero(ok & (s >= s_hi) & (rank < S))
        lo_ix = np.flatnonzero(ok & (s >= lo_score) & ((s < s_hi) | (rank >= S)))
        taken = {}
        if not state['started']:
            if len(hi_ix):
                taken = {r: int(d) for r, d in enumerate(hi_ix[:S])}
                state['started'] = True
        else:
            alive = np.isfinite(last).all(-1)

            def _match(r_ix, d_ix):
                """Hungarian of rows `r_ix` (last tight box) to detections `d_ix`, gated at `max_move`."""
                if not len(r_ix) or not len(d_ix):
                    return {}
                a, b = last[r_ix], e[d_ix]
                d = np.linalg.norm(box_centres(a)[:, None] - box_centres(b)[None], axis=-1)
                side = 0.5 * (box_sides(a)[:, None] + box_sides(b)[None])
                aff = np.clip(1.0 - d / (max_move * np.maximum(side, 1e-6)), 0.0, None)
                rr, cc = linear_sum_assignment(-aff)
                return {int(r_ix[i]): int(d_ix[j]) for i, j in zip(rr, cc) if aff[i, j] > 0}

            r_ix = np.flatnonzero(alive)
            taken.update(_match(r_ix, hi_ix))
            if len(lo_ix):
                taken.update(_match(np.array([r for r in r_ix if r not in taken], int), lo_ix))
            used = set(taken.values())
            free = [int(d) for d in hi_ix if d not in used]
            open_rows = [r for r in range(S) if r not in taken and not alive[r]]
            for r, d in zip(open_rows, free):
                taken[r] = d
        for r, d in taken.items():
            rows[r, t] = boxes[d, t]
            rsc[r, t] = scores[d, t]
            last[r] = e[d]
        seen = np.zeros(S, bool)
        seen[list(taken)] = True
        age[:] = np.where(seen, 0, age + 1)
        last[age > max_age] = np.nan
    return rows, rsc


def link_rows(boxes, scores=None, max_move=1.0, max_age=24, state=None):
    """Reorder instance rows frame by frame so a row follows ONE animal. In place, returns both.

    Inputs:
        boxes (S,T,C,4), scores (S,T,C) -- `scores` rides the same permutation.
        max_move -- the identity gate, in units of the animal's own mean box side (never
            pixels), gated at one side.
        max_age -- frames without evidence before a row's last box is forgotten.
        state -- carries the matcher across a call boundary so a clip processed in blocks links
            exactly as the whole clip does.
    Outputs:
        boxes (and scores when given), reordered so row `a` follows one animal.
    Side effects:
        In place on `boxes` and `scores`. An unmatched row stays EMPTY rather than taking
        an arbitrary leftover (a force-assigned row teleports across the frame and widens the
        crop union); an unclaimed detection may still START a row, but only an empty or expired
        one.
    Notes:
        Matching is against each row's LAST KNOWN box (not frame t-1), so a one-frame
        detector miss does not break the chain. Frame 0 of the clip is the only frame
        nothing is matched against; with carried state a block's frame 0 is permuted
        against the previous block's `last`, exactly as the whole-clip pass does mid-clip.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from ..detector.boxes import centres as box_centres
    from ..detector.boxes import sides as box_sides

    S, T, C, _ = boxes.shape
    if state is None or 'last' not in state:
        last = boxes[:, 0].copy()
        age = np.zeros(S, int)
        t0 = 1
    else:
        last, age, t0 = state['last'], state['age'], 0
    for t in range(t0, T):
        cur = boxes[:, t]
        cost = np.zeros((S, S), np.float32)
        for c in range(C):
            ok_p = np.isfinite(last[:, c]).all(-1)
            ok_c = np.isfinite(cur[:, c]).all(-1)
            if not (ok_p.any() and ok_c.any()):
                continue
            a, b = last[ok_p, c], cur[ok_c, c]
            d = np.linalg.norm(box_centres(a)[:, None] - box_centres(b)[None], axis=-1)
            sa, sb = box_sides(a), box_sides(b)
            side = 0.5 * (sa[:, None] + sb[None])
            gap = np.where(side > 0, d / (max_move * np.maximum(side, 1e-6)), np.inf)
            cost[np.ix_(ok_p, ok_c)] += np.clip(1.0 - gap, 0.0, None)
        rows, cols = linear_sum_assignment(-cost)
        taken = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] > 0}
        claimed = set(taken.values())
        free_dets = [c for c in range(S)
                     if c not in claimed and np.isfinite(cur[c]).all(-1).any()]
        open_rows = [r for r in range(S)
                     if r not in taken and not np.isfinite(last[r]).all(-1).any()]
        for r, c in zip(open_rows, free_dets):
            taken[r] = c
        out = np.full_like(cur, np.nan)
        sc = None if scores is None else np.full_like(scores[:, t], np.nan)
        for r, c in taken.items():
            out[r] = cur[c]
            if sc is not None:
                sc[r] = scores[:, t][c]
        boxes[:, t] = out
        if sc is not None:
            scores[:, t] = sc
        seen = np.isfinite(boxes[:, t]).all(-1)
        last = np.where(seen[..., None], boxes[:, t], last)
        age = np.where(seen.any(-1), 0, age + 1)
        last[age > max_age] = np.nan
    if state is not None:
        state['last'], state['age'] = last, age
    return boxes if scores is None else (boxes, scores)
