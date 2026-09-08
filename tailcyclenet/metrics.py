"""Metrics, written to be hard to fool.

1. A non-finite prediction is a MISS, never a hit (a nansum/full-denominator mean credits NaN).
2. Error and coverage are reported together -- a mean over matched points flatters decline.
3. The bootstrap is paired: resample the windows once, take the difference within each resample.
"""
from __future__ import annotations

import warnings

import numpy as np

from scipy.optimize import linear_sum_assignment


def _dist(pred, true):
    """Per-point Euclidean distance, NaN where either side is missing."""
    d = np.linalg.norm(pred - true, axis=-1)
    ok = np.isfinite(pred).all(-1) & np.isfinite(true).all(-1)
    return np.where(ok, d, np.nan)


#: Upper quantiles of the matched-distance vector, reported beside `err`. A mean cannot show a
#: tail, and every localisation failure this repo has found was found in a quantile.
ERR_PCTS = (75, 90, 95, 99)


def _err_pcts(d) -> dict:
    """`{'p75': ..., 'p90': ...}` over the finite entries of a distance vector."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not d.size:
        return {f'p{p}': float('nan') for p in ERR_PCTS}
    return {f'p{p}': float(np.percentile(d, p)) for p in ERR_PCTS}


def _frame_stats(frame_means) -> dict:
    """Per-FRAME aggregation of a matched-distance array -> the two numbers a ratio estimator
    needs. `frame_means` is one mean per frame, NaN where the frame matched nothing.

    Inputs: frame_means -- 1-D per-frame mean distances (NaN = frame contributed no matched
            point).
    Outputs: {'frame_err', 'frame_sum', 'frame_n'} -- the group's per-frame mean error, the sum
            of its per-frame means, and the number of frames behind it.

    Two numbers rather than one because the cross-group estimand is a RATIO --
    `sum(frame_sum) / sum(frame_n)` -- so that every labelled frame counts once regardless of how
    long its clip is. A plain mean of per-group `frame_err` values would re-introduce exactly the
    group weighting this is meant to remove (report 57 sections 22/24: on horse10, 696 short
    groups outvote the 19 holding 73.5% of all labelled keypoints).

    Frames that matched nothing are EXCLUDED rather than scored as zero or infinity, which keeps
    `err`'s standing contract -- a mean over matched points, read beside `coverage` (eval rule 6).
    """
    fm = np.asarray(frame_means, float)
    ok = np.isfinite(fm)
    return {'frame_err': float(fm[ok].mean()) if ok.any() else float('nan'),
            'frame_sum': float(fm[ok].sum()), 'frame_n': int(ok.sum())}


def _per_frame_means(d):
    """Mean matched distance per frame. `d` is `(..., T, K)`; the frame axis is -2.

    A frame that matched nothing is an all-NaN slice, which `nanmean` answers with NaN and a
    RuntimeWarning; the NaN is the wanted answer (`_frame_stats` drops it), so the warning is
    suppressed rather than the frame being special-cased.
    """
    d = np.asarray(d, float)
    axes = tuple(i for i in range(d.ndim) if i != d.ndim - 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(d, axis=axes)


def error_and_coverage(pred, true) -> dict:
    """MPJPE over points BOTH sides have, plus the coverage that produced it. `n_true` is the
    denominator that matters -- coverage = n_matched / n_true.

    Also returns `frame_err`/`frame_sum`/`frame_n` (see `_frame_stats`): `err` weights every
    matched POINT equally, so a frame with more visible keypoints counts for more. The per-frame
    figures are what `--agg frame` aggregates.
    """
    d = _dist(np.asarray(pred, float), np.asarray(true, float))
    labelled = np.isfinite(np.asarray(true, float)).all(-1)
    matched = np.isfinite(d)
    n_true = int(labelled.sum())
    return {
        'err': float(np.nanmean(d)) if matched.any() else float('nan'),
        'median': float(np.nanmedian(d)) if matched.any() else float('nan'),
        **_err_pcts(d),
        **_frame_stats(_per_frame_means(d)),
        'n_true': n_true,
        'n_matched': int(matched.sum()),
        'coverage': float(matched.sum() / n_true) if n_true else float('nan'),
    }


def pck(pred, true, thresholds) -> dict:
    """Fraction of LABELLED points predicted within each threshold -- a declined point is a
    failure at every threshold, not an abstention.
    """
    d = _dist(np.asarray(pred, float), np.asarray(true, float))
    n_true = int(np.isfinite(np.asarray(true, float)).all(-1).sum())
    if not n_true:
        return {f'pck@{t:g}': float('nan') for t in thresholds}
    return {f'pck@{t:g}': float(np.nansum(d <= t) / n_true) for t in thresholds}


def paired_bootstrap(per_unit_a, per_unit_b=None, n=10000, seed=0, alpha=0.05, weights=None):
    """Resample UNITS (windows, groups) -- not points -- and report the interval. With
    `per_unit_b`, the difference is taken inside each resample (paired); points within a window
    are correlated, so resampling points would be several times too tight.

    Pairing is complete-case: a unit where either side is non-finite leaves the comparison,
    which flatters the arm that failed more -- the count is returned rather than absorbed.

    `weights` (optional, one per unit) switches the statistic from a plain mean of unit values to
    the RATIO `sum(w*a)/sum(w)`, recomputed inside every resample. That is what `--agg frame`
    needs: the estimand is per-frame but the resampling unit stays the GROUP, because frames
    within a clip are correlated and resampling frames would give an interval several times too
    tight (report 57 section 24). A unit whose weight is non-finite or <= 0 leaves the
    comparison like a non-finite value. With `weights=None` this function is unchanged, down to
    the RNG draw.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(per_unit_a, float)
    keep = np.isfinite(a)
    b = None
    w = None
    if per_unit_b is not None:
        b = np.asarray(per_unit_b, float)
        keep &= np.isfinite(b)
    if weights is not None:
        w = np.asarray(weights, float)
        keep &= np.isfinite(w) & (w > 0)
    if b is not None:
        b = b[keep]
    if w is not None:
        w = w[keep]
    a = a[keep]
    dropped = int((~keep).sum())
    if a.size == 0:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0,
                'n_dropped': dropped}
    idx = rng.integers(0, a.size, size=(n, a.size))
    v = a if b is None else a - b
    if w is None:
        stat = v[idx].mean(1)
        point = float(v.mean())
    else:
        stat = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
        point = float((v * w).sum() / w.sum())
    return {'mean': point, 'lo': float(np.quantile(stat, alpha / 2)),
            'hi': float(np.quantile(stat, 1 - alpha / 2)), 'n': int(a.size),
            'n_dropped': dropped}


def motion_ratio(pred, ref) -> dict:
    """Predicted path length over a reference's, over the steps BOTH sides have. `ref` is the
    labels, or one position per instance-frame (the prediction's centroid then moves); both must
    live in the SAME space. The paired form (`scripts/eval.py --vs`) is what licenses a claim.

    With one reference position per instance-frame, the prediction's CENTROID is what moves: it
    is kept as a length-1 keypoint axis so the time axis stays at -3 for both shapes, and an
    all-NaN instance-frame is legal. The comparison runs over (..., T, K) entries where both
    sides are finite.
    """
    p, r = np.asarray(pred, float), np.asarray(ref, float)
    if r.ndim == p.ndim - 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            p = np.nanmean(p, axis=-2, keepdims=True)
        r = r[..., None, :]
    if p.shape != r.shape:
        raise ValueError(
            f'motion_ratio: pred {p.shape} vs ref {r.shape}. The two must be in the same space -- '
            'a 3D world path divided by a 2D pixel path is a number in no unit. Reproject the '
            'prediction before comparing it with a box centre.')
    ok = np.isfinite(p).all(-1) & np.isfinite(r).all(-1)
    both = ok[..., :-1, :] & ok[..., 1:, :]
    dp = np.linalg.norm(np.diff(p, axis=-3), axis=-1)
    dr = np.linalg.norm(np.diff(r, axis=-3), axis=-1)
    if not both.any():
        return {'ratio': float('nan'), 'pred_path': 0.0, 'ref_path': 0.0, 'n_steps': 0}
    a, b = float(dp[both].sum()), float(dr[both].sum())
    return {'ratio': a / b if b else float('nan'), 'pred_path': a, 'ref_path': b,
            'n_steps': int(both.sum())}


# multi-instance

def match_instances(pred, true, max_dist=np.inf, min_kpts_frac=0.0, cost='mean'):
    """Hungarian match predicted instances to labelled ones, per frame. Returns a list per frame
    of (pred_ix, true_ix, dist). pred/true: (S,T,K,R), NaN where absent; max_dist: a pair
    further apart is not a match. cost: `'mean'` (default) divides by the SHARED count;
    `'penalised'` charges declined labelled keypoints at max_dist, so a sparse row cannot
    out-bid a dense one (needs finite max_dist). min_kpts_frac: fraction of K a pair must share
    to be scored at all -- a FRACTION, not a count, since K ranges 4..47 across roots.

    Per frame the inputs are (Sp,K,R) vs (St,K,R) with (Sp,St,K) pairwise keypoint distances.
    Under `'penalised'` the LABEL's own count is the denominator, so a prediction cannot shrink
    the denominator by declining points -- which is the whole of the 'mean' hazard.

    Admissibility (`c <= max_dist`) is enforced BEFORE the Hungarian solve, not just after: a
    rectangular `linear_sum_assignment` always returns `min(Sp, St)` pairs regardless of cost,
    so leaving an inadmissible-but-cheap edge in the matrix lets the solver's SUM-minimisation
    pick it over a valid, individually-more-expensive edge that was available -- discarding it
    afterward then reports a miss+fp where a real match existed. `big` is a cost above the SUM
    of every admissible cost this frame, the standard bound that makes the solver prefer ANY
    achievable all-admissible matching over one using even a single inadmissible edge, so
    admissible edges are used whenever a feasible matching can use them, on cost second.
    """
    if cost not in ('mean', 'penalised'):
        raise ValueError(f"match_instances: cost must be 'mean' or 'penalised', got {cost!r}")
    penalise = cost == 'penalised' and np.isfinite(max_dist)
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    T, K = true.shape[1], true.shape[2]
    need = max(1, int(np.ceil(min_kpts_frac * K)))
    out = []
    with np.errstate(invalid='ignore'):
        for t in range(T):
            p, q = pred[:, t], true[:, t]
            d = np.linalg.norm(p[:, None] - q[None, :], axis=-1)
            ok = np.isfinite(p).all(-1)[:, None] & np.isfinite(q).all(-1)[None, :]
            n_ok = ok.sum(-1)
            if penalise:
                n_lab = np.broadcast_to(np.isfinite(q).all(-1).sum(-1)[None, :], n_ok.shape)
                num = np.where(ok, d, 0.0).sum(-1) + max_dist * (n_lab - n_ok)
                c = np.where(n_ok >= need, num / np.maximum(n_lab, 1), np.nan)
            else:
                c = np.where(n_ok >= need,
                             np.where(ok, d, 0.0).sum(-1) / np.maximum(n_ok, 1), np.nan)
            admissible = np.isfinite(c) & (c <= max_dist)
            big = float(np.sum(c, where=admissible)) + 1.0 if admissible.any() else 1.0
            ri, ci = linear_sum_assignment(np.where(admissible, c, big))
            out.append([(int(i), int(j), float(c[i, j])) for i, j in zip(ri, ci)
                        if admissible[i, j]])
    return out


def mota(pred, true, max_dist, ignore=None, ignore_boxes=None, min_kpts_frac=0.0,
         cost='mean', last=None, detail=False) -> dict:
    """MOTA and its three components, with an explicit ignore region.

    `detail=True` (default False, byte-identical) adds `idsw_detail` (a list of `{frame,
    gt_row, from_row, to_row}` per switch) and `fp_detail` (a list of `{frame, pred_row, kind}`,
    kind in `('dup', 'none')`, per false positive) -- both walk the SAME per-frame
    correspondence this function already builds, an ADDITIVE diagnostic that can never disagree
    with the `idsw`/`fp_dup`/`fp_none` counts above it.

    MOTA = 1 - (misses + fp + idsw) / labelled instances; report the components, since a split
    is not a method. `ignore` (St,T) marks PRESENT-but-unannotated instances: with
    `ignore_boxes` (St,T,4) an unmatched prediction is excused only inside a box, without them
    presence alone excuses it -- either way the count is `fp_ignored`. The FP term is split into
    `fp_dup` (near an already-claimed GT; arbitration removes it) and `fp_none` (on no animal).

    (St,T) and (Sp,T) record which instances exist at each frame. An instance with no finite
    keypoint has no centroid, and NaN is the answer -- not a warning; `_in_ignore` and the
    duplicate test both check for it explicitly. The duplicate test compares (Sp,T,R) and
    (St,T,R) centroids.

    `last` is the GT-index -> pred-index correspondence dict, in the same in/out shape
    `link_rows`/`CrossViewTracker` already thread across block calls: pass a dict and it is
    MUTATED in place so a caller can carry it into the next call over the CONTINUATION of this
    same sequence (e.g. the next `--chunk` unit of one group) -- a switch straddling that seam is
    then counted once, not lost to each call's own reset. `None` (the default) is byte-identical
    to every published number: a fresh dict is used and discarded, exactly as before this
    parameter existed.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    matches = match_instances(pred, true, max_dist, min_kpts_frac, cost)
    T = true.shape[1]
    true_present = np.isfinite(true).all(-1).any(-1)
    pred_present = np.isfinite(pred).all(-1).any(-1)
    if ignore is not None:
        ignore = np.asarray(ignore, bool)
    if ignore_boxes is not None:
        ignore_boxes = np.asarray(ignore_boxes, float)

    misses = fps = switches = gt = ignored = dups = 0
    switch_detail, fp_detail = [], []
    last = {} if last is None else last
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        centroid = np.nanmean(pred, axis=2)
        true_centroid = np.nanmean(true, axis=2)
    for t in range(T):
        pairs = matches[t]
        matched_true = {j for _, j, _ in pairs}
        matched_pred = {i for i, _, _ in pairs}
        present = np.flatnonzero(true_present[:, t])
        gt += len(present)
        misses += sum(1 for j in present if j not in matched_true)
        rows = np.flatnonzero(ignore[:, t]) if ignore is not None else np.empty(0, int)
        claimed = np.asarray(sorted(matched_true), int)
        for i in np.flatnonzero(pred_present[:, t]):
            if i in matched_pred:
                continue
            if len(rows) and _in_ignore(centroid[i, t], rows, t, ignore_boxes):
                ignored += 1
                continue
            fps += 1
            is_dup = False
            if len(claimed) and np.isfinite(centroid[i, t]).all():
                d = np.linalg.norm(true_centroid[claimed, t] - centroid[i, t], axis=-1)
                is_dup = bool(np.nanmin(d) <= max_dist) if np.isfinite(d).any() else False
            dups += int(is_dup)
            if detail:
                fp_detail.append({'frame': t, 'pred_row': int(i),
                                  'kind': 'dup' if is_dup else 'none'})
        for i, j, _ in pairs:
            if last.get(j) is not None and last[j] != i:
                switches += 1
                if detail:
                    switch_detail.append({'frame': t, 'gt_row': int(j),
                                          'from_row': int(last[j]), 'to_row': int(i)})
            last[j] = i
    out = {'mota': 1.0 - (misses + fps + switches) / gt if gt else float('nan'),
           'misses': misses, 'fp': fps, 'idsw': switches, 'gt': gt,
           'fp_ignored': ignored, 'fp_dup': dups, 'fp_none': fps - dups,
           'miss_rate': misses / gt if gt else float('nan'),
           'fp_rate': fps / gt if gt else float('nan'),
           'fp_dup_rate': dups / gt if gt else float('nan'),
           'fp_none_rate': (fps - dups) / gt if gt else float('nan'),
           'idsw_rate': switches / gt if gt else float('nan')}
    if detail:
        out['idsw_detail'] = switch_detail
        out['fp_detail'] = fp_detail
    return out


def idsw_stability_band(pred, true, max_dist, ignore=None, ignore_boxes=None, min_kpts_frac=0.0,
                        cost='mean', n=32, seed=0) -> dict:
    """`idsw` under N deterministic row-order permutations of `pred`/`true`, reporting a band
    instead of a bare count (identity_review_followthrough plan 3.2). MOTA has had a noise floor
    of +-0.023 since CLAUDE.md's eval rule 8; `idsw` has never had one, and it needs one for a
    structural reason no amount of code fixing can remove: at an EXACT crossing (two predicted
    points equidistant from two ground-truth points) every permutation of instances is an
    equally optimal Hungarian solution, and `linear_sum_assignment` picks one deterministically
    FROM THE MATRIX'S OWN ROW/COLUMN ORDER -- a real, physical crossing that resolves back to the
    correct identities can therefore read as 0 switches or as a full round-trip swap (2 * S
    switches) depending only on which order the instances happened to be stored in, with no
    change to any prediction, label, or gate. This is not a bug `match_instances`/`mota` can fix
    (any frame-independent Hungarian identity metric has the identical tie), so the actionable
    response is to quote `idsw` as this band rather than as an exact integer whenever ties are a
    live possibility (crowded, multi-animal footage).

    Each of the `n` trials independently permutes the S (instance) axis of BOTH `pred` and `true`
    -- `ignore`/`ignore_boxes`, when given, are permuted along `true`'s axis to stay aligned with
    it -- and re-scores with a fresh `mota()` call (no `last` carried between trials: each trial
    is an independent draw of "what if the rows had been stored in this order", not a
    continuation of the previous one). A permutation changes no pairwise COST, only which
    equal-cost assignment the solver returns, so this targets exactly the mechanism above and
    nothing else (it is not a general robustness check).

    Returns `{median, min, max, p5, p95, n, values}` over the `n` draws' `idsw` counts. A band of
    `{min: k, max: k}` (zero width) means no tie in this clip/arm is close enough to be resolved
    by order alone -- report it as zero and move on, per the plan's own acceptance rule.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    Sp, St = pred.shape[0], true.shape[0]
    ig = None if ignore is None else np.asarray(ignore, bool)
    igb = None if ignore_boxes is None else np.asarray(ignore_boxes, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pi, ti = rng.permutation(Sp), rng.permutation(St)
        r = mota(pred[pi], true[ti], max_dist,
                ignore=None if ig is None else ig[ti],
                ignore_boxes=None if igb is None else igb[ti],
                min_kpts_frac=min_kpts_frac, cost=cost)
        vals.append(r['idsw'])
    vals = np.asarray(vals, float)
    return {'median': float(np.median(vals)), 'min': float(vals.min()), 'max': float(vals.max()),
            'p5': float(np.percentile(vals, 5)), 'p95': float(np.percentile(vals, 95)),
            'n': int(n), 'values': [int(v) for v in vals]}



def _in_ignore(centroid, rows, t, ignore_boxes) -> bool:
    """Does this prediction land on a present-but-unannotated animal? No boxes -> presence alone
    excuses it; that is the blanket rule and why `fp_ignored` is reported.
    """
    if ignore_boxes is None:
        return True
    if not np.isfinite(centroid[:2]).all():
        return False
    x, y = centroid[0], centroid[1]
    for j in rows:
        b = ignore_boxes[j, t]
        if np.isfinite(b).all() and b[0] <= x <= b[2] and b[1] <= y <= b[3]:
            return True
    return False


def matched_error(pred, true, max_dist=np.inf, min_kpts_frac=0.0, cost='mean') -> dict:
    """MPJPE over HUNGARIAN-MATCHED instances, for multi-animal predictions. Row index is not
    identity once boxes come from a detector. `unmatched_true` is part of the answer: a method
    that predicts one animal well and ignores nine looks excellent on `err` alone.

    The returned counts are the POINT counts, not just the instance counts -- quote matched
    error beside its coverage.

    `frame_err`/`frame_sum`/`frame_n` are the per-FRAME aggregation (see `_frame_stats`). A frame
    is one unit however many instances it holds, so a frame with two matched animals counts once
    -- the mean is taken over every matched point in that frame, across instances.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    pairs = match_instances(pred, true, max_dist, min_kpts_frac, cost)
    T = true.shape[1]
    dists, n_true_inst, n_matched_inst = [], 0, 0
    frame_means = np.full(T, np.nan)
    for t in range(T):
        present = np.isfinite(true[:, t]).all(-1).any(-1)
        n_true_inst += int(present.sum())
        n_matched_inst += len(pairs[t])
        per_frame = []
        for i, j, _ in pairs[t]:
            dt = _dist(pred[i, t], true[j, t])
            dists.append(dt)
            per_frame.append(dt)
        if per_frame:
            arr = np.concatenate(per_frame)
            if np.isfinite(arr).any():
                frame_means[t] = float(np.nanmean(arr))
    n_true = int(np.isfinite(true).all(-1).sum())
    if not dists:
        return {'err': float('nan'), 'median': float('nan'), **_err_pcts([]),
                **_frame_stats(frame_means), 'coverage': 0.0,
                'n_true': n_true, 'n_matched': 0,
                'n_true_inst': n_true_inst, 'n_matched_inst': 0,
                'unmatched_true': n_true_inst}
    d = np.concatenate(dists)
    return {'err': float(np.nanmean(d)), 'median': float(np.nanmedian(d)), **_err_pcts(d),
            **_frame_stats(frame_means),
            'coverage': float(np.isfinite(d).sum() / max(1, n_true)),
            'n_true': n_true, 'n_matched': int(np.isfinite(d).sum()),
            'n_true_inst': n_true_inst, 'n_matched_inst': n_matched_inst,
            'unmatched_true': n_true_inst - n_matched_inst}
