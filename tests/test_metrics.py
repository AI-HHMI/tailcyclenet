"""The numbers that get published.

Two properties here are the kind that a refactor breaks quietly: the vectorised Hungarian cost
must equal the loop it replaced, and the ignore region must excuse only what it actually covers.
"""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from tailcyclenet.metrics import _dist, match_instances, matched_error, mota


def _naive_match(pred, true, max_dist=np.inf):
    """The double Python loop `match_instances` replaced. Kept as an independent derivation."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = []
    for t in range(true.shape[1]):
        p, q = pred[:, t], true[:, t]
        cost = np.full((p.shape[0], q.shape[0]), np.nan)
        for i in range(p.shape[0]):
            for j in range(q.shape[0]):
                d = _dist(p[i], q[j])
                cost[i, j] = np.nanmean(d) if np.isfinite(d).any() else np.nan
        big = np.nanmax(cost) + 1 if np.isfinite(cost).any() else 1.0
        ri, ci = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
        out.append([(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci)
                    if np.isfinite(cost[i, j]) and cost[i, j] <= max_dist])
    return out


def test_vectorised_matching_equals_the_loop_it_replaced():
    """rat-city is ONE group of 57,594 frames; at ten animals the loop was 5.7M calls per eval."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        Sp, St = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        T, K = int(rng.integers(1, 4)), int(rng.integers(1, 6))
        R = int(rng.choice([2, 3]))
        pred = rng.normal(size=(Sp, T, K, R))
        true = rng.normal(size=(St, T, K, R))
        pred[rng.random(pred.shape[:3]) < 0.3] = np.nan     # absent points, the normal case
        true[rng.random(true.shape[:3]) < 0.3] = np.nan
        md = float(rng.choice([np.inf, 1.0, 2.0]))
        for got, want in zip(match_instances(pred, true, md), _naive_match(pred, true, md)):
            assert [(i, j) for i, j, _ in got] == [(i, j) for i, j, _ in want]
            assert np.allclose([d for *_, d in got], [d for *_, d in want])


def test_ignore_boxes_excuse_only_what_they_cover():
    """An ignore region is a place, not a licence for the whole frame.

    Without boxes there is nothing to localise against, so presence alone excuses every unmatched
    prediction -- which can zero the FP term. That fallback is kept, but it must SAY how many it
    swallowed, or a method with no false positives is indistinguishable from a frame that
    forgave them all.
    """
    true = np.full((1, 1, 2, 2), np.nan)                      # nothing annotated
    pred = np.array([[[[5., 5.], [5., 5.]]],                  # inside the ignore box
                     [[[900., 900.], [900., 900.]]]])         # nowhere near it
    ignore = np.array([[True]])
    boxes = np.array([[[0., 0., 10., 10.]]])

    with_boxes = mota(pred, true, 1.0, ignore=ignore, ignore_boxes=boxes)
    assert (with_boxes['fp'], with_boxes['fp_ignored']) == (1, 1)

    blanket = mota(pred, true, 1.0, ignore=ignore)
    assert (blanket['fp'], blanket['fp_ignored']) == (0, 2), 'the concession must be counted'

    plain = mota(pred, true, 1.0)
    assert (plain['fp'], plain['fp_ignored']) == (2, 0)


def test_matched_error_reports_the_counts_behind_its_coverage():
    """`err` is a mean over MATCHED points, so it needs the matched count, not a rowwise one."""
    true = np.zeros((2, 3, 4, 2))
    pred = np.zeros((2, 3, 4, 2))
    pred[1] = np.nan                                          # one instance never predicted
    m = matched_error(pred, true)
    assert m['n_true'] == int(np.isfinite(true).all(-1).sum())
    assert m['n_matched'] == pytest.approx(m['coverage'] * m['n_true'])
    assert m['unmatched_true'] == m['n_true_inst'] - m['n_matched_inst']
