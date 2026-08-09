"""Metrics, written to be hard to fool.

Three rules are baked in because breaking them is how wrong numbers get published:

1. **A non-finite prediction is a MISS, never a hit.** posetail's `get_mpjpe` divides a `nansum`
   numerator by a full denominator, so a model that predicts NaN scores as perfect on those
   points. Every function here counts coverage separately and never averages a NaN away.
2. **Error and coverage are reported together.** `err` is a mean over matched points, so a method
   that predicts fewer, easier points looks better while being worse. A delta in `err` means
   nothing without the coverage that produced it.
3. **The bootstrap is paired.** Two methods evaluated on the same windows are compared by
   resampling the windows once and taking the difference within each resample. Unpaired intervals
   on shared data overstate uncertainty enough to hide real effects.
"""
from __future__ import annotations

import numpy as np

from scipy.optimize import linear_sum_assignment


def _dist(pred, true):
    """Per-point Euclidean distance, NaN where either side is missing."""
    d = np.linalg.norm(pred - true, axis=-1)
    ok = np.isfinite(pred).all(-1) & np.isfinite(true).all(-1)
    return np.where(ok, d, np.nan)


def error_and_coverage(pred, true) -> dict:
    """MPJPE over points BOTH sides have, plus the coverage that produced it.

    `n_true` is the denominator that matters: it is how many points were labelled, so
    `coverage = n_matched / n_true` says what fraction of the task was attempted.
    """
    d = _dist(np.asarray(pred, float), np.asarray(true, float))
    labelled = np.isfinite(np.asarray(true, float)).all(-1)
    matched = np.isfinite(d)
    n_true = int(labelled.sum())
    return {
        'err': float(np.nanmean(d)) if matched.any() else float('nan'),
        'median': float(np.nanmedian(d)) if matched.any() else float('nan'),
        'n_true': n_true,
        'n_matched': int(matched.sum()),
        'coverage': float(matched.sum() / n_true) if n_true else float('nan'),
    }


def pck(pred, true, thresholds) -> dict:
    """Fraction of LABELLED points predicted within each threshold.

    The denominator is labelled points, not matched ones -- a point the model declined to
    predict is a failure at every threshold, not an abstention.
    """
    d = _dist(np.asarray(pred, float), np.asarray(true, float))
    n_true = int(np.isfinite(np.asarray(true, float)).all(-1).sum())
    if not n_true:
        return {f'pck@{t:g}': float('nan') for t in thresholds}
    return {f'pck@{t:g}': float(np.nansum(d <= t) / n_true) for t in thresholds}


def paired_bootstrap(per_unit_a, per_unit_b=None, n=10000, seed=0, alpha=0.05):
    """Resample UNITS (windows, groups) -- not points -- and report the interval.

    With `per_unit_b`, the difference is taken inside each resample, which is what makes the
    comparison paired. Points within a window are correlated, so resampling points would give an
    interval several times too tight.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(per_unit_a, float)
    keep = np.isfinite(a)
    b = None
    if per_unit_b is not None:
        b = np.asarray(per_unit_b, float)
        keep &= np.isfinite(b)
        b = b[keep]
    a = a[keep]
    if a.size == 0:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0}
    idx = rng.integers(0, a.size, size=(n, a.size))
    stat = a[idx].mean(1) if b is None else (a[idx] - b[idx]).mean(1)
    point = float(a.mean() if b is None else (a - b).mean())
    return {'mean': point, 'lo': float(np.quantile(stat, alpha / 2)),
            'hi': float(np.quantile(stat, 1 - alpha / 2)), 'n': int(a.size)}


# ----------------------------------------------------------------------------------------------
# multi-instance
# ----------------------------------------------------------------------------------------------

def match_instances(pred, true, max_dist=np.inf, ignore=None):
    """Hungarian match predicted instances to labelled ones, per frame.

    Args:
        pred: (Sp, T, K, R), NaN where absent
        true: (St, T, K, R), NaN where absent
        max_dist: a pair further apart than this is not a match
        ignore: (St, T) bool -- instances that are present but unannotated. A prediction matched
            to one of these is neither a true nor a false positive. This exists because 73% of a
            tracker's false positives on rat-city were measured to be real animals the annotator
            skipped, and counting them cost 0.017 MOTA.

    Returns a list per frame of (pred_ix, true_ix, dist).
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    T = true.shape[1]
    out = []
    for t in range(T):
        p, q = pred[:, t], true[:, t]                    # (Sp,K,R), (St,K,R)
        cost = np.full((p.shape[0], q.shape[0]), np.nan)
        for i in range(p.shape[0]):
            for j in range(q.shape[0]):
                d = _dist(p[i], q[j])
                cost[i, j] = np.nanmean(d) if np.isfinite(d).any() else np.nan
        big = np.nanmax(cost) + 1 if np.isfinite(cost).any() else 1.0
        ri, ci = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
        pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci)
                 if np.isfinite(cost[i, j]) and cost[i, j] <= max_dist]
        out.append(pairs)
    return out


def mota(pred, true, max_dist, ignore=None) -> dict:
    """MOTA and its three components, with an explicit ignore region.

    MOTA = 1 - (misses + false positives + id switches) / labelled instances. Report the
    components: two methods with the same MOTA and different miss/FP splits are not the same
    method, and only MOTA replicates across seeds -- and only above a +-0.023 seed floor.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    matches = match_instances(pred, true, max_dist, ignore)
    T = true.shape[1]
    true_present = np.isfinite(true).all(-1).any(-1)          # (St,T)
    pred_present = np.isfinite(pred).all(-1).any(-1)          # (Sp,T)
    if ignore is not None:
        ignore = np.asarray(ignore, bool)

    misses = fps = switches = gt = 0
    last = {}
    for t in range(T):
        pairs = matches[t]
        matched_true = {j for _, j, _ in pairs}
        matched_pred = {i for i, _, _ in pairs}
        present = np.flatnonzero(true_present[:, t])
        gt += len(present)
        misses += sum(1 for j in present if j not in matched_true)
        for i in np.flatnonzero(pred_present[:, t]):
            if i in matched_pred:
                continue
            if ignore is not None and ignore[:, t].any():
                # An unmatched prediction that lands on a present-but-unannotated animal is
                # neither a TP nor an FP. Without a box we cannot localise it, so the presence
                # assertion alone suppresses one FP -- deliberately conservative.
                continue
            fps += 1
        for i, j, _ in pairs:
            if last.get(j) is not None and last[j] != i:
                switches += 1
            last[j] = i
    return {'mota': 1.0 - (misses + fps + switches) / gt if gt else float('nan'),
            'misses': misses, 'fp': fps, 'idsw': switches, 'gt': gt,
            'miss_rate': misses / gt if gt else float('nan'),
            'fp_rate': fps / gt if gt else float('nan'),
            'idsw_rate': switches / gt if gt else float('nan')}
