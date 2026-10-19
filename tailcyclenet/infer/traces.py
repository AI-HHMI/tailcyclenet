"""Output-neutral diagnostics: `--overlap-trace` probes (overlap agreement, model confidence, prior
out-of-bounds, block-boundary census) and the `--det-trace` / `--overlap-trace` writers.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import torch


# Frame/window-indexed columns stitched by `merge_blocks`; anything else is a per-group constant.
FRAME_KEYS = ('pred', 'conf', 'pred2d', 'conf2d', 'model_conf2d', 'box_agree', 'det_box', 'det_score', 'owner_window', 'triangulated', 'quality', 'quality_precision')


WINDOW_KEYS = ('outcome', 'crop', 'crop_refined', 'box_prompt_cams', 'window_start', 'window_stop', 'window_ordinal')


def capture_overlap_agreement(stats, a, frames, f0, pred, p):
    """Compare this window's overlap-region prediction against the PREVIOUS window's, before it
    is overwritten by the seam rule (report 53 follow-up, independent review's overlap-reuse
    idea).  A frame in `frames` may already hold a finite prediction from the window before this
    one -- the exact quantity the seam rule discards -- so this is a free test-retest signal: the
    same physical frame predicted from two different crop framings and partially different
    `carried` context, with zero extra forward passes.  Opt-in
    (`stats['capture_overlap_agreement']`) since the comparison is cheap but the accumulated
    list is not needed on a normal run.

    BOTH VALUES are recorded, not only their distance (plan 7.0): the seam rule always keeps the
    LATER window's estimate, and asking whether that is the better choice (plan 7.1) requires
    scoring each against the labels separately, which a distance cannot support.  `pos` is the
    frame's position WITHIN the overlap region, counted from the later window's own start -- 0 is
    that window's anchor frame, where `carried` was read, so disagreement as a function of `pos`
    reads out prior influence directly on unlabelled footage (plan 7.3, the echo coefficient).

    A census is accumulated beside the rows because several plan-7 ideas can die on population
    size alone, and that is the cheapest possible refutation: `pairs_seen` counts (row, seam)
    pairs actually compared, `pairs_no_prev` those skipped because the earlier window wrote
    nothing finite for that row, and `pairs_no_shared` those where the two estimates share no
    finite keypoint.  The last two are a SELECTION BIAS toward easy rows -- the hook only ever
    sees a seam where the earlier window already succeeded -- so they are counted rather than
    silently dropped.
    """
    census = stats.setdefault('overlap_census', {'pairs_seen': 0, 'pairs_no_prev': 0,
                                                'pairs_no_shared': 0})
    prev = pred[a, frames - f0]
    had_prev = np.isfinite(prev).all(-1).any(-1)
    census['pairs_no_prev'] += int((~had_prev).sum())
    if not had_prev.any():
        return
    shared = np.isfinite(p[had_prev]).all(-1) & np.isfinite(prev[had_prev]).all(-1)
    d = np.linalg.norm(p[had_prev] - prev[had_prev], axis=-1)
    rows = stats.setdefault('overlap_agreement', [])
    where = np.flatnonzero(had_prev)
    for row, t, dist, n_sh in zip(where, frames[had_prev],
                                  np.where(shared, d, np.nan).mean(-1), shared.sum(-1)):
        if not np.isfinite(dist):
            census['pairs_no_shared'] += 1
            continue
        census['pairs_seen'] += 1
        rows.append({'animal': int(a), 'frame': int(t), 'pos': int(row),
                     'dist': float(dist), 'n_shared': int(n_sh),
                     'prev': prev[row].copy(), 'new': p[row].copy()})


def capture_model_confidence(stats, a, frames, out):
    """Record the model's OWN per-camera confidence, which the window loop otherwise discards.

    Plan 5.1 (B2). `window.py` reads `coords_pred`, `2d_pred`, `vis_pred`, `vis_pred_2d` and
    `3d_pred_triangulate` and throws the rest away; `conf_pred_2d` is a per-(camera, frame,
    keypoint) logit the model already computed. It is a different QUANTITY from everything the
    refuted geometric identity levers spent (`--kpt-affinity`, `--kpt-centre`, `--axis-*`,
    `--dup-res-px`), all of which derive from output COORDINATES: this is the model's own
    uncertainty, and it is the one thing on that axis a better model can actually change -- the
    standing objection to spending a weak cue.

    `conf_pred_2d` is `(C, B, T, K)` logits; `v` takes batch 0 to `(C, T, K)` and `per_cam` is its
    per-keypoint median, `(C, T)`. Recorded per (animal, frame) as the mean and min over cameras,
    plus the camera count, so the offline question (is confidence depressed on the
    duplicate-implicated rows during the known episode, relative to the same rows outside it AND
    to non-implicated rows in the same frames?) can be asked without keeping a (C, T, K) tensor
    per window. Opt-in under the same flag as the overlap capture, and read-only.
    """
    c2 = out.get('conf_pred_2d')
    if c2 is None:
        return
    v = c2[:, 0].detach().float().cpu().numpy()
    if v.ndim != 3 or v.shape[1] != len(frames):
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        per_cam = np.nanmedian(v, axis=-1)
        mean_t = np.nanmean(per_cam, axis=0)
        min_t = np.nanmin(per_cam, axis=0)
    rows = stats.setdefault('model_conf', [])
    for i, t in enumerate(frames):
        rows.append((int(a), int(t), float(mean_t[i]), float(min_t[i]), int(v.shape[0])))


def count_prior_oob(stats, a, f0, prior):
    """Count the prior keypoints `prior_out_of_bounds` NaN'd for this (animal, window).

    THE CHEAPER ALTERNATIVE plan 7.2 says must be beaten before its overlap mechanism is built:
    "count how many prior keypoints `prior_out_of_bounds` NaN'd per window -- a free
    identity-discontinuity smoke alarm needing no overlap mechanism at all. If that alone finds
    the known switches, this idea is redundant and should not be built."

    The reasoning is that a carried prior lands outside its own crop exactly when the slot's crop
    and its remembered pose stopped describing the same animal -- which is what an identity
    discontinuity IS, seen from inside the pipeline and needing no second forward pass. Recorded
    under the same opt-in flag as the overlap capture so one probe answers both questions, and
    counted rather than acted on: rule 12 makes this a smoke alarm, never an arm-ranker.
    """
    n_nan = int(torch.isnan(prior[0]).all(-1).sum())
    stats.setdefault('prior_oob', []).append((int(a), int(f0), n_nan,
                                              int(prior.shape[1])))


def census_block_boundary(stats, f_read, f_own):
    """Count the seam a block boundary makes uncomparable, so the census cannot overstate itself.

    `capture_overlap_agreement` is BLIND HERE, and silence would look exactly like clean
    coverage. A block emits only `[f0, f_own)`; the frames in `[f_own, f_read)` were predicted by
    this block's last window and are re-predicted from scratch by the NEXT block's first window,
    so that seam's two estimates never coexist in one array and no comparison is possible at all.
    Plan 7.0 asks for this number by name -- a census that reported only what it managed to see
    would understate how much of the overlap population is structurally invisible.
    """
    c = stats.setdefault('overlap_census', {'pairs_seen': 0, 'pairs_no_prev': 0,
                                            'pairs_no_shared': 0})
    if f_read > f_own:
        c['block_boundary_seams'] = c.get('block_boundary_seams', 0) + 1
        c['block_boundary_frames'] = c.get('block_boundary_frames', 0) + int(f_read - f_own)


def write_overlap_trace(path, per_group):
    """Write the overlap test-retest capture as one npz. Output-neutral, a scratch diagnostic.

    Inputs: path -- destination npz; per_group -- {group key: (rows, census)} as accumulated in
        `stats` by `capture_overlap_agreement`.
    Outputs: None.
    Side effects: writes `path`.

    Flat parallel arrays rather than one object array per row, because the consumer is an offline
    join against labels and `windows.pq` (plan 7.1/7.2/7.5) and a structured npz is what pandas
    and numpy both read without unpickling. `prev`/`new` are the two estimates of the SAME frame
    -- the earlier window's discarded one and the later window's kept one -- stacked as
    `(n, K, R)`; keeping both is the whole point of the capture, since asking which the seam rule
    SHOULD keep needs each scored against labels separately.

    The census travels with the rows, not in a separate file: a coverage figure separated from
    the population it describes is how a selection bias gets quoted as a result.
    """
    import json

    keys, animal, frame, pos, dist, n_shared, prev, new = [], [], [], [], [], [], [], []
    ob_key, ob_animal, ob_frame, ob_nan, ob_k = [], [], [], [], []
    mc_animal, mc_frame, mc_mean, mc_min = [], [], [], []
    for key, (_rows, _census, oob, conf) in sorted(per_group.items()):
        for a, f0, n_nan, n_k in oob:
            ob_key.append(key)
            ob_animal.append(a)
            ob_frame.append(f0)
            ob_nan.append(n_nan)
            ob_k.append(n_k)
        for a, t, mean_c, min_c, _n_cam in conf:
            mc_animal.append(a)
            mc_frame.append(t)
            mc_mean.append(mean_c)
            mc_min.append(min_c)
    for key, (rows, _census, _oob, _conf) in sorted(per_group.items()):
        for r in rows:
            keys.append(key)
            animal.append(r['animal'])
            frame.append(r['frame'])
            pos.append(r['pos'])
            dist.append(r['dist'])
            n_shared.append(r['n_shared'])
            prev.append(r['prev'])
            new.append(r['new'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        group=np.array(keys, dtype=object), animal=np.asarray(animal, np.int32),
        frame=np.asarray(frame, np.int32), pos=np.asarray(pos, np.int32),
        dist=np.asarray(dist, np.float32), n_shared=np.asarray(n_shared, np.int32),
        prev=(np.stack(prev) if prev else np.zeros((0, 0, 0), np.float32)),
        new=(np.stack(new) if new else np.zeros((0, 0, 0), np.float32)),
        oob_group=np.array(ob_key, dtype=object), oob_animal=np.asarray(ob_animal, np.int32),
        oob_frame=np.asarray(ob_frame, np.int32), oob_nan=np.asarray(ob_nan, np.int32),
        oob_k=np.asarray(ob_k, np.int32),
        conf_animal=np.asarray(mc_animal, np.int32), conf_frame=np.asarray(mc_frame, np.int32),
        conf_mean=np.asarray(mc_mean, np.float32), conf_min=np.asarray(mc_min, np.float32),
        census=np.array(json.dumps({k: c for k, (_r, c, _o, _m) in sorted(per_group.items())})))
    total = {k: sum(c.get(k, 0) for _r, c, _o, _m in per_group.values())
             for k in ('pairs_seen', 'pairs_no_prev', 'pairs_no_shared',
                       'block_boundary_seams', 'block_boundary_frames')}
    print(f'wrote overlap trace {path}: {len(dist)} (row, seam) pair(s); census {total}')


def write_det_trace(path, groups):
    """Write the `--det-trace` JSON: one decode trace list per group key, newline-terminated."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(groups, indent=1) + '\n')
