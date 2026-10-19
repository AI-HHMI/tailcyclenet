"""Association: per-camera detections -> one row per animal.

`associate` groups one frame's boxes across cameras into 3D instances; `associate_group` runs it
(or `CrossViewTracker`, or 2D `link_rows`) over a whole clip.

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

from tracktail.model.cube import project_points_torch

from ..detector.boxes import centres as box_centres


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


def associate(cgroup, boxes_per_cam, max_res_px=30.0, min_views=2, max_instances=0):
    """Group boxes across cameras into 3D instances.

    Inputs:
        cgroup -- tracktail camera dicts, one per camera, in the SOURCE frame (not cropped).
        boxes_per_cam -- list of (N_c, 4) tensors, per camera, in that camera's pixels.
        max_res_px -- a group whose median reprojection residual exceeds this is rejected.
        min_views -- the floor on how many cameras an instance may be built from. 2 is the
            algorithm, not a parameter (every group starts from a cross-camera PAIR); `1`
            emits each leftover box as its own single-view instance with no triangulated point.
        max_instances -- cap on returned groups (0 = uncapped).
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
    centres = [box_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
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
                support = _support_count(cgroup, (ca, cb), centres, used, p3d, max_res_px)
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


def associate_group(raw, session, gid, max_instances, link=False, min_views=2,
                    track=True, max_move=1.25, max_age=8, stats=None,
                    state=None, duplicate_radius=0.75, duplicate_persist=5, frame_base=0):
    """The ASSOCIATION half: per-camera detections -> ONE ROW PER ANIMAL. Microseconds per frame.

    Inputs:
        raw -- `detect_raw`'s (boxes, scores); session, gid, max_instances (row count S).
        track -- 3D multiview default: `CrossViewTracker`, one cross-view target set carried
            across frames; `False` restores the memoryless per-frame `associate`.
        link -- adds `link_rows` in 2D only (the tracker builds when `track and C > 1`).
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
    Outputs:
        The same pair re-indexed so row `a` is one animal -- across cameras always, and across
        frames wherever a tracker or `link_rows` ran. In 2D the row index is the only identity.
    Notes:
        Cross-view geometry is built once for a static rig, per frame for a moving one.
    """
    import numpy as np
    import torch

    r_box, r_sc = raw
    D, T, C = r_sc.shape
    S = max_instances or 1
    out = np.full((S, T, C, 4), np.nan, np.float32)
    sc = np.full((S, T, C), np.nan, np.float32)
    moving = any(session.rig.moving.values())
    cgroup = None if moving else session.cgroup(gid)
    if state is None:
        state = {}
    if 'tracker' not in state:
        tracker = None
        if track and C > 1:
            from .track import CrossViewTracker
            tracker = CrossViewTracker(S, max_res_px=session.assoc_res_max_px,
                                       min_views=min_views,
                                       max_move=max_move, max_age=max_age,
                                       duplicate_radius=duplicate_radius,
                                       duplicate_persist=duplicate_persist,
                                       frame_base=frame_base)
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
                           max_res_px=session.assoc_res_max_px, max_instances=S,
                           min_views=min_views)
        for a, g in enumerate(groups[:S]):
            for c, box in g['boxes'].items():
                d = g['members'][c]
                out[a, t, c] = box.numpy()
                sc[a, t, c] = float(per_cam[c][1][d])
    pre_link = int(np.isfinite(out).all(-1).sum())
    if link and tracker is None:
        out, sc = link_rows(out, sc, max_move=max_move, max_age=max_age,
                            state=state.setdefault('link', {}))
    if stats is not None:
        stats['association_raw_offered'] = stats.get('association_raw_offered', 0) + raw_offered
        stats['association_pre_link'] = stats.get('association_pre_link', 0) + pre_link
    if stats is not None:
        stats['association_kept'] = stats.get('association_kept', 0) + \
            int(np.isfinite(out).all(-1).sum())

    return out, sc


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
