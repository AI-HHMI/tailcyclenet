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

import warnings

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

def match_instances(pred, true, max_dist=np.inf):
    """Hungarian match predicted instances to labelled ones, per frame.

    Args:
        pred: (Sp, T, K, R), NaN where absent
        true: (St, T, K, R), NaN where absent
        max_dist: a pair further apart than this is not a match

    Returns a list per frame of (pred_ix, true_ix, dist).

    The pairwise cost is vectorised rather than a double Python loop. rat-city is ONE group of
    57,594 frames, so ten animals meant 5.7M `_dist` calls per eval and multi-animal scoring was
    effectively unrunnable.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    T = true.shape[1]
    out = []
    with np.errstate(invalid='ignore'):
        for t in range(T):
            p, q = pred[:, t], true[:, t]                # (Sp,K,R), (St,K,R)
            d = np.linalg.norm(p[:, None] - q[None, :], axis=-1)          # (Sp,St,K)
            ok = np.isfinite(p).all(-1)[:, None] & np.isfinite(q).all(-1)[None, :]
            n_ok = ok.sum(-1)
            cost = np.where(n_ok > 0, np.where(ok, d, 0.0).sum(-1) / np.maximum(n_ok, 1), np.nan)
            big = np.nanmax(cost) + 1 if np.isfinite(cost).any() else 1.0
            ri, ci = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
            out.append([(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci)
                        if np.isfinite(cost[i, j]) and cost[i, j] <= max_dist])
    return out


def mota(pred, true, max_dist, ignore=None, ignore_boxes=None) -> dict:
    """MOTA and its three components, with an explicit ignore region.

    MOTA = 1 - (misses + false positives + id switches) / labelled instances. Report the
    components: two methods with the same MOTA and different miss/FP splits are not the same
    method, and only MOTA replicates across seeds -- and only above a +-0.023 seed floor.

    Args:
        ignore: (St, T) bool -- instances asserted PRESENT but not annotated. 73% of a tracker's
            false positives on rat-city were measured to be real animals the annotator skipped,
            and counting them cost 0.017 MOTA.
        ignore_boxes: (St, T, 4) xyxy for those instances. WITH boxes, an unmatched prediction is
            excused only if its own centroid falls inside one. WITHOUT them there is nothing to
            localise against, so presence alone excuses every unmatched prediction on the frame
            -- which can zero the FP term outright. Either way the count is returned as
            `fp_ignored` rather than folded silently into the score.

    THE FP TERM IS SPLIT, because the two halves want opposite fixes. `fp_dup` is an unmatched
    prediction sitting within `max_dist` of a GT that something else already claimed -- two crops
    on one animal, which arbitration removes. `fp_none` landed on no labelled animal at all, which
    arbitration cannot touch and a detector threshold can. They are one undifferentiated counter in
    MOTA itself and were one here, so "MOTA is FP-limited" said nothing about what to build.
    Duplicates are not otherwise representable: `match_instances` is one-to-one by construction
    (`linear_sum_assignment`), so no second prediction is ever assigned to a claimed GT -- it falls
    out as an ordinary false positive and the proximity has to be tested for separately.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    matches = match_instances(pred, true, max_dist)
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
    """Does this prediction land on a present-but-unannotated animal?

    No boxes -> the frame-wide fallback: presence alone excuses it. That is the blanket rule, and
    it is why `fp_ignored` is reported.
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


def matched_error(pred, true, max_dist=np.inf) -> dict:
    """MPJPE over HUNGARIAN-MATCHED instances, for multi-animal predictions.

    Row index is not identity. When boxes come from a detector, prediction row `a` and label row
    `a` are different animals, and a row-indexed MPJPE then measures nothing -- it reported 385 px
    on a fly dataset whose animals are ~30 px across. Match first, then measure.

    `unmatched_true` is part of the answer, not a footnote: a method that predicts one animal
    well and ignores nine looks excellent on `err` alone.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    pairs = match_instances(pred, true, max_dist)
    T = true.shape[1]
    dists, n_true_inst, n_matched_inst = [], 0, 0
    for t in range(T):
        present = np.isfinite(true[:, t]).all(-1).any(-1)
        n_true_inst += int(present.sum())
        n_matched_inst += len(pairs[t])
        for i, j, _ in pairs[t]:
            dists.append(_dist(pred[i, t], true[j, t]))
    # The POINT counts, not just the instance counts. The caller reports `err` over matched
    # points and must report the coverage that produced it -- quoting a matched error beside a
    # row-indexed coverage describes two different quantities as if they were one.
    n_true = int(np.isfinite(true).all(-1).sum())
    if not dists:
        return {'err': float('nan'), 'median': float('nan'), 'coverage': 0.0,
                'n_true': n_true, 'n_matched': 0,
                'n_true_inst': n_true_inst, 'n_matched_inst': 0,
                'unmatched_true': n_true_inst}
    d = np.concatenate(dists)
    return {'err': float(np.nanmean(d)), 'median': float(np.nanmedian(d)),
            'coverage': float(np.isfinite(d).sum() / max(1, n_true)),
            'n_true': n_true, 'n_matched': int(np.isfinite(d).sum()),
            'n_true_inst': n_true_inst, 'n_matched_inst': n_matched_inst,
            'unmatched_true': n_true_inst - n_matched_inst}
