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


def error_and_coverage(pred, true) -> dict:
    """MPJPE over points BOTH sides have, plus the coverage that produced it. `n_true` is the
    denominator that matters -- coverage = n_matched / n_true.
    """
    d = _dist(np.asarray(pred, float), np.asarray(true, float))
    labelled = np.isfinite(np.asarray(true, float)).all(-1)
    matched = np.isfinite(d)
    n_true = int(labelled.sum())
    return {
        'err': float(np.nanmean(d)) if matched.any() else float('nan'),
        'median': float(np.nanmedian(d)) if matched.any() else float('nan'),
        **_err_pcts(d),
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


def paired_bootstrap(per_unit_a, per_unit_b=None, n=10000, seed=0, alpha=0.05):
    """Resample UNITS (windows, groups) -- not points -- and report the interval. With
    `per_unit_b`, the difference is taken inside each resample (paired); points within a window
    are correlated, so resampling points would be several times too tight.
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
    dropped = int((~keep).sum())
    if a.size == 0:
        return {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'n': 0,
                'n_dropped': dropped}
    idx = rng.integers(0, a.size, size=(n, a.size))
    stat = a[idx].mean(1) if b is None else (a[idx] - b[idx]).mean(1)
    point = float(a.mean() if b is None else (a - b).mean())
    # Pairing is complete-case: a unit where either side is non-finite leaves the comparison,
    # which flatters the arm that failed more -- the count is returned rather than absorbed.
    return {'mean': point, 'lo': float(np.quantile(stat, alpha / 2)),
            'hi': float(np.quantile(stat, 1 - alpha / 2)), 'n': int(a.size),
            'n_dropped': dropped}


def motion_ratio(pred, ref) -> dict:
    """Predicted path length over a reference's, over the steps BOTH sides have. `ref` is the
    labels, or one position per instance-frame (the prediction's centroid then moves); both must
    live in the SAME space. The paired form (`scripts/eval.py --vs`) is what licenses a claim.
    """
    p, r = np.asarray(pred, float), np.asarray(ref, float)
    if r.ndim == p.ndim - 1:
        # One reference position per instance-frame: the prediction's CENTROID is what moves. Kept
        # as a length-1 keypoint axis so the time axis stays at -3 for both shapes.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # an all-NaN instance-frame is legal
            p = np.nanmean(p, axis=-2, keepdims=True)
        r = r[..., None, :]
    if p.shape != r.shape:
        raise ValueError(
            f'motion_ratio: pred {p.shape} vs ref {r.shape}. The two must be in the same space -- '
            'a 3D world path divided by a 2D pixel path is a number in no unit. Reproject the '
            'prediction before comparing it with a box centre.')
    ok = np.isfinite(p).all(-1) & np.isfinite(r).all(-1)          # (..., T, K)
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
            p, q = pred[:, t], true[:, t]                # (Sp,K,R), (St,K,R)
            d = np.linalg.norm(p[:, None] - q[None, :], axis=-1)          # (Sp,St,K)
            ok = np.isfinite(p).all(-1)[:, None] & np.isfinite(q).all(-1)[None, :]
            n_ok = ok.sum(-1)
            if penalise:
                # The LABEL's own count is the denominator, so a prediction cannot shrink the
                # denominator by declining points -- which is the whole of the 'mean' hazard.
                n_lab = np.broadcast_to(np.isfinite(q).all(-1).sum(-1)[None, :], n_ok.shape)
                num = np.where(ok, d, 0.0).sum(-1) + max_dist * (n_lab - n_ok)
                c = np.where(n_ok >= need, num / np.maximum(n_lab, 1), np.nan)
            else:
                c = np.where(n_ok >= need,
                             np.where(ok, d, 0.0).sum(-1) / np.maximum(n_ok, 1), np.nan)
            big = np.nanmax(c) + 1 if np.isfinite(c).any() else 1.0
            ri, ci = linear_sum_assignment(np.where(np.isfinite(c), c, big))
            out.append([(int(i), int(j), float(c[i, j])) for i, j in zip(ri, ci)
                        if np.isfinite(c[i, j]) and c[i, j] <= max_dist])
    return out


def mota(pred, true, max_dist, ignore=None, ignore_boxes=None, min_kpts_frac=0.0,
         cost='mean') -> dict:
    """MOTA and its three components, with an explicit ignore region.

    MOTA = 1 - (misses + fp + idsw) / labelled instances; report the components, since a split
    is not a method. `ignore` (St,T) marks PRESENT-but-unannotated instances: with
    `ignore_boxes` (St,T,4) an unmatched prediction is excused only inside a box, without them
    presence alone excuses it -- either way the count is `fp_ignored`. The FP term is split into
    `fp_dup` (near an already-claimed GT; arbitration removes it) and `fp_none` (on no animal).
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    matches = match_instances(pred, true, max_dist, min_kpts_frac, cost)
    T = true.shape[1]
    true_present = np.isfinite(true).all(-1).any(-1)          # (St,T)
    pred_present = np.isfinite(pred).all(-1).any(-1)          # (Sp,T)
    if ignore is not None:
        ignore = np.asarray(ignore, bool)
    if ignore_boxes is not None:
        ignore_boxes = np.asarray(ignore_boxes, float)

    misses = fps = switches = gt = ignored = dups = 0
    last = {}
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        # An instance with no finite keypoint has no centroid, and NaN is the answer -- not a
        # warning. `_in_ignore` and the duplicate test below both check for it explicitly.
        warnings.simplefilter('ignore', RuntimeWarning)
        centroid = np.nanmean(pred, axis=2)                   # (Sp,T,R)
        true_centroid = np.nanmean(true, axis=2)              # (St,T,R)
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
            if len(claimed) and np.isfinite(centroid[i, t]).all():
                d = np.linalg.norm(true_centroid[claimed, t] - centroid[i, t], axis=-1)
                dups += int(np.nanmin(d) <= max_dist) if np.isfinite(d).any() else 0
        for i, j, _ in pairs:
            if last.get(j) is not None and last[j] != i:
                switches += 1
            last[j] = i
    return {'mota': 1.0 - (misses + fps + switches) / gt if gt else float('nan'),
            'misses': misses, 'fp': fps, 'idsw': switches, 'gt': gt,
            'fp_ignored': ignored, 'fp_dup': dups, 'fp_none': fps - dups,
            'miss_rate': misses / gt if gt else float('nan'),
            'fp_rate': fps / gt if gt else float('nan'),
            'fp_dup_rate': dups / gt if gt else float('nan'),
            'fp_none_rate': (fps - dups) / gt if gt else float('nan'),
            'idsw_rate': switches / gt if gt else float('nan')}


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
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    pairs = match_instances(pred, true, max_dist, min_kpts_frac, cost)
    T = true.shape[1]
    dists, n_true_inst, n_matched_inst = [], 0, 0
    for t in range(T):
        present = np.isfinite(true[:, t]).all(-1).any(-1)
        n_true_inst += int(present.sum())
        n_matched_inst += len(pairs[t])
        for i, j, _ in pairs[t]:
            dists.append(_dist(pred[i, t], true[j, t]))
    # The POINT counts, not just the instance counts -- quote matched error beside its coverage.
    n_true = int(np.isfinite(true).all(-1).sum())
    if not dists:
        return {'err': float('nan'), 'median': float('nan'), **_err_pcts([]), 'coverage': 0.0,
                'n_true': n_true, 'n_matched': 0,
                'n_true_inst': n_true_inst, 'n_matched_inst': 0,
                'unmatched_true': n_true_inst}
    d = np.concatenate(dists)
    return {'err': float(np.nanmean(d)), 'median': float(np.nanmedian(d)), **_err_pcts(d),
            'coverage': float(np.isfinite(d).sum() / max(1, n_true)),
            'n_true': n_true, 'n_matched': int(np.isfinite(d).sum()),
            'n_true_inst': n_true_inst, 'n_matched_inst': n_matched_inst,
            'unmatched_true': n_true_inst - n_matched_inst}
