"""The numbers that get published: matching, MOTA/IDF1, chunked scoring, aggregation."""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from tailcyclenet.evaluation.metrics import (
    ERR_PCTS,
    _dist,
    error_and_coverage,
    idf1,
    idsw_stability_band,
    label_extent,
    match_instances,
    matched_error,
    mota,
    motion_ratio,
    paired_bootstrap,
)


def _naive_match(pred, true, max_dist=np.inf):
    """The double loop `match_instances` replaced; admissibility is applied to the SOLVER
    (`big` exceeds the sum of admissible costs), not filtered after it."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = []
    for t in range(true.shape[1]):
        p, q = pred[:, t], true[:, t]
        cost = np.full((p.shape[0], q.shape[0]), np.nan)
        for i in range(p.shape[0]):
            for j in range(q.shape[0]):
                d = _dist(p[i], q[j])
                cost[i, j] = np.nanmean(d) if np.isfinite(d).any() else np.nan
        admissible = np.isfinite(cost) & (cost <= max_dist)
        big = float(np.sum(cost, where=admissible)) + 1.0 if admissible.any() else 1.0
        ri, ci = linear_sum_assignment(np.where(admissible, cost, big))
        out.append([(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci)
                    if admissible[i, j]])
    return out


def test_vectorised_matching_equals_the_loop_it_replaced():
    """rat-city is ONE group of 57,594 frames; the loop was 5.7M calls per eval."""
    rng = np.random.default_rng(0)
    for _ in range(20):
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


def test_an_inadmissible_edge_cannot_displace_an_available_admissible_match():
    """Gating cost > max_dist only AFTER the Hungarian solve let a cheap inadmissible pairing
    displace the one admissible match (costs [[1, 2], [2, 5]] at 1.5) -- report 53 follow-up."""
    pred = np.array([[[[0.0]]], [[[3.0]]]])
    true = np.array([[[[1.0]]], [[[-2.0]]]])
    d = np.linalg.norm(pred[:, 0, :, 0][:, None] - true[:, 0, :, 0][None, :], axis=-1)
    assert np.allclose(d, [[1.0, 2.0], [2.0, 5.0]])
    got = match_instances(pred, true, max_dist=1.5)
    assert [(i, j) for i, j, _ in got[0]] == [(0, 0)]
    m = mota(pred, true, max_dist=1.5)
    assert (m['misses'], m['fp'], m['idsw']) == (1, 1, 0)


def test_ignore_boxes_excuse_only_what_they_cover():
    """An ignore region is a place: boxes excuse only what they cover, and the count says how many."""
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


def test_surplus_rows_are_fp_split_into_dup_and_none():
    """Surplus rows are false positives (eval once truncated them away); `fp_dup` vs `fp_none` want
    opposite fixes; `matched_error` reports the counts behind its coverage."""
    true = np.zeros((1, 3, 4, 2))
    pred = np.zeros((3, 3, 4, 2))
    pred[1:] = 500.0                                          # two animals that are not there
    r = mota(pred, true, 1.0)
    assert (r['fp'], r['misses'], r['gt']) == (6, 0, 3)
    assert matched_error(pred, true)['unmatched_true'] == 0

    true = np.zeros((1, 1, 2, 2))                             # one animal at the origin
    pred = np.array([[[[0., 0.], [0., 0.]]],                  # matches it
                     [[[1., 1.], [1., 1.]]],                  # a second crop on the SAME animal
                     [[[500., 500.], [500., 500.]]]])         # on nothing at all
    r = mota(pred, true, 10.0)
    assert (r['fp'], r['fp_dup'], r['fp_none']) == (2, 1, 1)
    # The radius is the whole definition of "the same animal": tighten it and the duplicate is a
    # box on nothing instead.
    assert mota(pred, true, 0.5)['fp_dup'] == 0

    true = np.zeros((2, 3, 4, 2))
    pred = np.zeros((2, 3, 4, 2))
    pred[1] = np.nan                                          # one instance never predicted
    m = matched_error(pred, true)
    assert m['n_true'] == int(np.isfinite(true).all(-1).sum())
    assert m['n_matched'] == pytest.approx(m['coverage'] * m['n_true'])
    assert m['unmatched_true'] == m['n_true_inst'] - m['n_matched_inst']


def test_motion_ratio_reads_a_frozen_prediction_and_takes_a_centroid_reference():
    """A locked pose loses the motion every consistency number rewards; a box centre reference
    compares against the prediction centroid."""
    true = np.zeros((1, 5, 2, 3))
    true[0, :, :, 0] = np.arange(5)[:, None]                   # the animal walks along x
    moving = true.copy()
    frozen = np.zeros_like(true)                               # predicts the same pose every frame

    assert motion_ratio(moving, true)['ratio'] == pytest.approx(1.0)
    assert motion_ratio(frozen, true)['ratio'] == 0.0
    # A step with a missing end is SKIPPED, not bridged -- bridging would charge the whole
    # excursion across the gap to one step and read as MORE motion than the continuous arm.
    gappy = moving.copy()
    gappy[0, 2] = np.nan

    pred = np.zeros((1, 3, 4, 2))
    pred[0, :, :, 0] = np.arange(3)[:, None] * 2.0             # centroid moves 2 px per frame
    ref = np.zeros((1, 3, 2))
    ref[0, :, 0] = np.arange(3)                                # the box moves 1 px per frame
    assert motion_ratio(pred, ref)['ratio'] == pytest.approx(2.0)


def test_a_one_keypoint_instance_frame_does_not_collapse_the_match_radius():
    """A one-keypoint instance-frame has extent 0, and radius 0 admits nothing -- MOTA read
    catastrophic from the radius alone. A non-positive extent is no radius at all."""
    # Two animals, two frames, three keypoints -- but only ONE labelled point per instance-frame.
    true = np.full((2, 2, 3, 2), np.nan, np.float32)
    true[0, :, 0] = [10.0, 10.0]
    true[1, :, 0] = [90.0, 90.0]
    extent = label_extent(true)
    assert extent == 0.0, 'the degenerate case must actually be degenerate, or this proves nothing'

    pred = true + 0.5                       # a very good prediction: half a pixel out
    assert matched_error(pred, true, max_dist=extent)['n_matched'] == 0, \
        'radius 0 admits nothing short of an EXACT hit -- that is the bug'
    # What eval.py now does instead: a non-positive extent is no radius at all.
    max_dist = extent if np.isfinite(extent) and extent > 0 else np.inf
    assert matched_error(pred, true, max_dist=max_dist)['n_matched'] == 4


def test_idsw_stability_band_is_wide_at_an_exact_crossing_and_zero_without_one():
    """At an exact tie the Hungarian choice depends on storage order only; a clean sequence has a
    zero-width band."""
    true = np.zeros((2, 3, 1, 2))
    true[0, :, 0] = [0.0, 0.0]
    true[1, :, 0] = [10.0, 10.0]

    tie = true.copy()
    tie[0, 1, 0] = tie[1, 1, 0] = [5.0, 5.0]     # frame 1: an exact tie at the crossing

    band = idsw_stability_band(tie, true, max_dist=100.0, n=64, seed=0)
    assert band['min'] == 0, 'a crossing that resolves back must be readable as zero switches'
    assert band['max'] > band['min'], \
        f'no tie sensitivity found at an exact crossing: {band}'

    clean = idsw_stability_band(true, true, max_dist=100.0, n=64, seed=0)
    assert clean['min'] == clean['max'] == 0, \
        f'an unambiguous sequence must have a zero-width band, got {clean}'


def test_err_percentiles_describe_the_tail_the_mean_hides():
    """p75..p99 come from the matched vector `err` averages; all-NaN is NaN, not 0."""
    true = np.zeros((1, 1000, 1, 2))
    pred = np.zeros((1, 1000, 1, 2))
    pred[0, 980:, 0, 0] = 1000.0               # 2% of frames catastrophic, 98% exact
    m = error_and_coverage(pred, true)
    assert m['p75'] == 0.0 and m['p90'] == 0.0, 'the bulk is exact; only the tail moves'
    assert m['p99'] == 1000.0, 'p99 must land ON the failures, not average them away'
    assert m['err'] == pytest.approx(20.0), 'while the mean reports a 20 px model'

    # And an all-NaN vector is NaN, not a crash and not a zero: no matched point is not "no error".
    empty = error_and_coverage(np.full((1, 2, 1, 2), np.nan), true[:, :2])
    assert all(np.isnan(empty[f'p{p}']) for p in ERR_PCTS)


def test_penalised_cost_charges_declined_keypoints_and_mean_stays_default():
    """'penalised' charges unshared label keypoints max_dist; the default 'mean' is byte-identical
    to the pre-flag behaviour (an arm, not a silent correction)."""
    true = np.zeros((1, 1, 4, 2))
    pred = np.full((2, 1, 4, 2), np.nan)
    pred[0, 0, 0] = [0.0, 0.0]                 # ONE keypoint, perfect
    pred[1, 0] = 0.5                           # all four, 0.5 px off in each axis

    mean = match_instances(pred, true, max_dist=20.0, cost='mean')[0]
    pen = match_instances(pred, true, max_dist=20.0, cost='penalised')[0]
    assert mean[0][0] == 0, 'the one-point row wins on a mean over shared keypoints'
    assert pen[0][0] == 1, 'and must lose once the three it declined are charged'
    # The arithmetic, not just the ranking: (0 + 20*3)/4 = 15 against sqrt(0.5)*4/4.
    assert pen[0][2] == pytest.approx(np.hypot(0.5, 0.5))

    # A COMPLETE PREDICTION IS UNAFFECTED -- which is why this is a no-op on every arm on record.
    dense = np.zeros((2, 1, 4, 2)) + 0.5
    a = match_instances(dense, true, max_dist=20.0, cost='mean')[0]
    b = match_instances(dense, true, max_dist=20.0, cost='penalised')[0]
    assert a == b

    # An infinite radius has no finite charge to levy, so it falls back rather than returning inf.
    assert (match_instances(pred, true, cost='penalised')[0]
            == match_instances(pred, true, cost='mean')[0])
    with pytest.raises(ValueError):
        match_instances(pred, true, cost='oks')

    rng = np.random.default_rng(0)
    true = rng.normal(size=(3, 20, 5, 2)) * 10
    pred = true + rng.normal(size=true.shape)
    pred[np.asarray(rng.random(pred.shape[:3]) < 0.2)[..., None].repeat(2, -1)] = np.nan
    for kw in ({}, {'cost': 'mean'}):
        assert (match_instances(pred, true, max_dist=8.0, **kw)
                == match_instances(pred, true, max_dist=8.0))
    assert (mota(pred, true, 8.0)['mota']
            == mota(pred, true, 8.0, cost='mean')['mota'])
    assert (matched_error(pred, true, max_dist=8.0)['err']
            == matched_error(pred, true, max_dist=8.0, cost='mean')['err'])


def test_mota_detail_is_additive_and_agrees_with_the_counts():
    """`detail=True` changes no existing field; its idsw/fp items match `idsw`/`fp_dup`/`fp_none`."""
    pred = np.array([[[[0.0, 0.0]], [[100.0, 100.0]]],
                     [[[100.0, 100.0]], [[0.0, 0.0]]]])   # (Sp=2, T=2, K=1, R=2)
    true = np.array([[[[0.0, 0.0]], [[0.0, 0.0]]]])       # (St=1, T=2, K=1, R=2)

    plain = mota(pred, true, max_dist=10.0)
    detailed = mota(pred, true, max_dist=10.0, detail=True)

    assert 'idsw_detail' not in plain, 'detail=False (the default) must not add the key'
    assert 'fp_detail' not in plain
    for k in plain:
        assert detailed[k] == plain[k] or (np.isnan(detailed[k]) and np.isnan(plain[k])), \
            f'{k} differs between detail=False and detail=True -- not additive'

    assert plain['idsw'] == 1
    assert detailed['idsw_detail'] == [{'frame': 1, 'gt_row': 0, 'from_row': 0, 'to_row': 1}]

    # St=1 GT row at (0,0); Sp=2 predictions: row 0 matches the GT, row 1 is a near-duplicate
    # (within max_dist of the claimed GT), row 2 is far from everything (a bare false positive).
    pred = np.array([[[[0.0, 0.0]]], [[[5.0, 0.0]]], [[[500.0, 500.0]]]])  # (Sp=3, T=1, K=1, R=2)
    true = np.array([[[[0.0, 0.0]]]])                                     # (St=1, T=1, K=1, R=2)

    d = mota(pred, true, max_dist=10.0, detail=True)
    assert d['fp'] == 2 and d['fp_dup'] == 1 and d['fp_none'] == 1

    by_kind = {'dup': [], 'none': []}
    for item in d['fp_detail']:
        by_kind[item['kind']].append(item['pred_row'])
    assert by_kind['dup'] == [1], 'the near-duplicate row must be classified dup'
    assert by_kind['none'] == [2], 'the bare false positive must be classified none'


def test_frame_aggregation_and_the_unweighted_bootstrap():
    """`--agg frame` equals `--agg group` only for equal-sized groups (report 57 22/24), and
    `weights=None` leaves every published interval unchanged, down to the RNG draw."""
    one_frame = []
    for err in (1.0, 5.0, 9.0):
        true = np.zeros((1, 1, 4, 2))
        pred = np.zeros((1, 1, 4, 2))
        pred[..., 0] = err
        one_frame.append(error_and_coverage(pred, true))
    grp = paired_bootstrap([m['err'] for m in one_frame], n=200, seed=0)
    frm = paired_bootstrap([m['frame_err'] for m in one_frame], n=200, seed=0,
                           weights=[m['frame_n'] for m in one_frame])
    assert grp['mean'] == frm['mean'], 'one frame per group leaves no weighting freedom'

    long_true, long_pred = np.zeros((1, 100, 4, 2)), np.zeros((1, 100, 4, 2))
    long_pred[..., 0] = 10.0
    short_true, short_pred = np.zeros((1, 1, 4, 2)), np.zeros((1, 1, 4, 2))
    short_pred[..., 0] = 0.0
    uneven = [error_and_coverage(long_pred, long_true), error_and_coverage(short_pred, short_true)]
    grp2 = paired_bootstrap([m['err'] for m in uneven], n=200, seed=0)
    frm2 = paired_bootstrap([m['frame_err'] for m in uneven], n=200, seed=0,
                            weights=[m['frame_n'] for m in uneven])
    assert grp2['mean'] == 5.0, 'per-group gives the 1-frame clip the same vote as the 100-frame'
    assert abs(frm2['mean'] - 1000 / 101) < 1e-9, 'per-frame weights by the frames behind it'

    rng = np.random.default_rng(7)
    a, b = rng.normal(10, 2, 30), rng.normal(9, 2, 30)
    assert paired_bootstrap(a, seed=5) == paired_bootstrap(a, seed=5, weights=None)
    assert paired_bootstrap(a, b, seed=5) == paired_bootstrap(a, b, seed=5, weights=None)
    flat = paired_bootstrap(a, seed=5)
    ones = paired_bootstrap(a, seed=5, weights=np.ones(a.size))
    assert np.isclose(flat['mean'], ones['mean']) and np.isclose(flat['lo'], ones['lo'])


def test_idf1_sees_a_parked_swap_counts_fp_misses_and_ignore():
    """A half-clip swap is idsw 2 but IDF1 0.5 under one global map; spare rows are IDFP unless
    inside a present box (as `fp_ignored`); a missed animal is IDFN."""
    T = 10
    true = np.zeros((2, T, 1, 2))
    true[1, :, 0] = [100.0, 100.0]
    swapped = true.copy()
    swapped[:, T // 2:] = true[::-1, T // 2:]
    assert mota(swapped, true, 1.0)['idsw'] == 2
    r = idf1(swapped, true, 1.0)
    assert r['idf1'] == pytest.approx(0.5) and r['idtp'] == T and r['n_pred'] == r['n_true'] == 2 * T
    assert idf1(true, true, 1.0)['idf1'] == 1.0
    assert idf1(true[::-1], true, 1.0)['idf1'] == 1.0, 'row order is not identity: one global map'

    true = np.full((2, 4, 1, 2), np.nan)
    true[0, :, 0] = [0.0, 0.0]
    pred = np.full((2, 4, 1, 2), np.nan)
    pred[0] = true[0]
    pred[1, :, 0] = [50.0, 50.0]
    r = idf1(pred, true, 1.0)
    assert (r['idtp'], r['idfp'], r['idfn']) == (4, 4, 0)
    ignore = np.zeros((2, 4), bool)
    ignore[1] = True
    boxes = np.full((2, 4, 4), np.nan)
    boxes[1] = [40.0, 40.0, 60.0, 60.0]
    assert idf1(pred, true, 1.0, ignore=ignore, ignore_boxes=boxes)['idfp'] == 0
    boxes[1] = [0.0, 0.0, 1.0, 1.0]
    assert idf1(pred, true, 1.0, ignore=ignore, ignore_boxes=boxes)['idfp'] == 4
    half = pred[:1].copy()
    half[:, 2:] = np.nan
    assert idf1(half, true, 1.0)['idfn'] == 2
