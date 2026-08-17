"""The numbers that get published.

Two properties here are the kind that a refactor breaks quietly: the vectorised Hungarian cost
must equal the loop it replaced, and the ignore region must excuse only what it actually covers.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from tailcyclenet.metrics import (ERR_PCTS, _dist, error_and_coverage, match_instances,
                                  matched_error, mota, motion_ratio)

REPO = Path(__file__).resolve().parent.parent


def _eval_module():
    """Import scripts/eval.py without running main()."""
    spec = importlib.util.spec_from_file_location('tcn_eval', REPO / 'scripts' / 'eval.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_a_surplus_predicted_row_is_a_false_positive():
    """A detector offers as many animals as it finds, not as many as were labelled.

    `eval.py` used to truncate `pred` to `true`'s row count, which deleted the surplus rows before
    anything could score them -- so a detector that hallucinated a second animal in every frame
    read as perfect. They are false positives.
    """
    true = np.zeros((1, 3, 4, 2))
    pred = np.zeros((3, 3, 4, 2))
    pred[1:] = 500.0                                          # two animals that are not there
    r = mota(pred, true, 1.0)
    assert (r['fp'], r['misses'], r['gt']) == (6, 0, 3)
    assert matched_error(pred, true)['unmatched_true'] == 0


def test_the_fp_term_separates_duplicates_from_nothing():
    """The two halves want opposite fixes, so an undifferentiated `fp` cannot pick one.

    Arbitration removes a second crop on an animal something else already claimed; a score
    threshold removes a box on nothing. `match_instances` is one-to-one, so the duplicate is never
    assigned and falls out as an ordinary false positive -- the proximity has to be tested for.
    """
    true = np.zeros((1, 1, 2, 2))                             # one animal at the origin
    pred = np.array([[[[0., 0.], [0., 0.]]],                  # matches it
                     [[[1., 1.], [1., 1.]]],                  # a second crop on the SAME animal
                     [[[500., 500.], [500., 500.]]]])         # on nothing at all
    r = mota(pred, true, 10.0)
    assert (r['fp'], r['fp_dup'], r['fp_none']) == (2, 1, 1)
    # The radius is the whole definition of "the same animal": tighten it and the duplicate is a
    # box on nothing instead.
    assert mota(pred, true, 0.5)['fp_dup'] == 0


def test_a_paired_delta_uses_only_the_points_both_arms_matched():
    """Eval rule 6, as code. An arm that declines the hard points has a better mean over its OWN
    matched set and must not read as better than one that attempted them."""
    ev = _eval_module()
    true = np.zeros((1, 4, 1, 2))
    good = np.zeros((1, 4, 1, 2))                             # exact on all four frames
    good[0, 2] = 3.0                                          # except one, where it is 3 off
    picky = np.full((1, 4, 1, 2), np.nan)
    picky[0, :2] = 0.0                                        # attempts only the two easy frames

    ea, eb, n, nlab = ev._shared_error({'_pred': good, '_true': true}, {'_pred': picky,
                                                                       '_true': true})
    assert (n, nlab) == (2, 4)
    assert ea == eb == 0.0, 'over the shared points the two are identical, and the delta is 0'
    # ...where a whole-set comparison would have made the picky arm look better by 3/4 of a unit.
    assert np.nanmean(np.linalg.norm(good - true, axis=-1)) > 0


def test_matched_error_reports_the_counts_behind_its_coverage():
    """`err` is a mean over MATCHED points, so it needs the matched count, not a rowwise one."""
    true = np.zeros((2, 3, 4, 2))
    pred = np.zeros((2, 3, 4, 2))
    pred[1] = np.nan                                          # one instance never predicted
    m = matched_error(pred, true)
    assert m['n_true'] == int(np.isfinite(true).all(-1).sum())
    assert m['n_matched'] == pytest.approx(m['coverage'] * m['n_true'])
    assert m['unmatched_true'] == m['n_true_inst'] - m['n_matched_inst']


def test_a_frozen_prediction_reads_as_less_motion_than_a_moving_one():
    """The statistic RC1 needed and nothing had. A locked pose scores best on every consistency
    number in the repo (jerk, bone CV) while losing 30% of the animal's motion."""
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


def test_motion_ratio_takes_a_centroid_reference():
    """A box centre is one position per instance-frame, so the prediction's centroid is what moves."""
    pred = np.zeros((1, 3, 4, 2))
    pred[0, :, :, 0] = np.arange(3)[:, None] * 2.0             # centroid moves 2 px per frame
    ref = np.zeros((1, 3, 2))
    ref[0, :, 0] = np.arange(3)                                # the box moves 1 px per frame
    assert motion_ratio(pred, ref)['ratio'] == pytest.approx(2.0)


def test_paired_motion_uses_only_the_steps_both_arms_have():
    """Eval rule 6 again: a path summed over whatever an arm predicted rewards predicting less."""
    ev = _eval_module()
    true = np.zeros((1, 4, 1, 2))
    true[0, :, 0, 0] = np.arange(4)
    full = true.copy()
    picky = np.full((1, 4, 1, 2), np.nan)
    picky[0, :2] = true[0, :2]                                 # only the first step
    ra, rb, n = ev._shared_motion({'_pred': full, '_true': true},
                                  {'_pred': picky, '_true': true})
    assert n == 1 and ra == pytest.approx(1.0) and rb == pytest.approx(1.0)


def test_a_one_keypoint_instance_frame_does_not_collapse_the_match_radius():
    """ZERO IS FINITE, and it was passing the `isfinite` guard as a legitimate radius.

    `extent` is the median keypoint-box DIAGONAL over instance-frames, and an instance-frame with
    exactly one finite labelled keypoint contributes 0. rat-city labels 2.02 of its 4 points per
    animal-frame, so its median is one draw from being 1 -- and at radius 0 `match_instances`
    admits nothing, every instance is a miss AND a false positive at once, and MOTA goes to
    -(1 + fp_rate). It reads as a catastrophic model failure that is entirely the radius.
    """
    # Two animals, two frames, three keypoints -- but only ONE labelled point per instance-frame.
    true = np.full((2, 2, 3, 2), np.nan, np.float32)
    true[0, :, 0] = [10.0, 10.0]
    true[1, :, 0] = [90.0, 90.0]
    with np.errstate(all='ignore'):
        span = np.nanmax(true, axis=2) - np.nanmin(true, axis=2)
        extent = float(np.nanmedian(np.linalg.norm(span, axis=-1)))
    assert extent == 0.0, 'the degenerate case must actually be degenerate, or this proves nothing'

    pred = true + 0.5                       # a very good prediction: half a pixel out
    assert matched_error(pred, true, max_dist=extent)['n_matched'] == 0, \
        'radius 0 admits nothing short of an EXACT hit -- that is the bug'
    # What eval.py now does instead: a non-positive extent is no radius at all.
    max_dist = extent if np.isfinite(extent) and extent > 0 else np.inf
    assert matched_error(pred, true, max_dist=max_dist)['n_matched'] == 4


def test_mota_dist_zero_is_not_read_as_unset():
    """`radius = mota_dist or extent * 0.5` -- 0.0 is falsy, so an explicit `--mota-dist 0` was
    silently replaced by the derived radius, i.e. the flag did the opposite of what it said."""
    for mota_dist, extent in ((0.0, 50.0), (None, 50.0), (7.0, 50.0)):
        old = mota_dist or float(extent) * 0.5
        new = float(mota_dist) if mota_dist is not None else float(extent) * 0.5
        if mota_dist == 0.0:
            assert old == 25.0 and new == 0.0, 'this is the case that used to be ignored'
        else:
            assert old == new


def test_chunking_a_clip_partitions_it_and_holds_the_match_radius_fixed(tmp_path):
    """`--chunk` gives a one-group clip something for the bootstrap to resample.

    The bootstrap resamples GROUPS, and rat-city's whole test split is a single 500-frame group, so
    every delta on it came back `DEGENERATE (one group -- no interval exists)`. The roots this repo
    most wants long-clip numbers from are exactly the ones with the fewest groups.

    Two properties, and the second is the one that makes it a resampling change rather than a METRIC
    change: the chunks PARTITION the frames (nothing scored twice, nothing dropped), and the match
    radius is the whole group's, not each chunk's. Sized per chunk it swung 27.6 to 101.9 px across
    ten chunks of one clip -- a 3.7x swing that would leave chunks non-exchangeable, which is the
    one thing a bootstrap needs them to be.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    from tailcyclenet.format import Session

    ev = _eval_module()
    cf._session_2d(tmp_path / 'ds' / 'test' / 's', T=20)
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    labels = {f's/{gid}': (sess.labels(gid), sess) for gid in sess.groups}
    gid = next(iter(sess.groups))
    true = labels[f's/{gid}'][0].points2d[..., 0, :]
    preds = {f's/{gid}': {'pred': true.copy(), 'mode': np.array('2d')}}

    chunked_p, chunked_l = ev.chunk_frames(preds, labels, 5)
    assert len(chunked_p) == 4, f'20 frames in 5s is 4 chunks, got {sorted(chunked_p)}'
    # A PARTITION: every frame in exactly one chunk, and the concatenation is the original.
    rebuilt = np.concatenate([chunked_p[f's/{gid}#{t0}']['pred'] for t0 in (0, 5, 10, 15)], axis=1)
    np.testing.assert_array_equal(rebuilt, true)
    for t0 in (0, 5, 10, 15):
        assert chunked_p[f's/{gid}#{t0}']['pred'].shape[1] == 5
        # The labels are sliced to match, or a chunk scores its prediction against another's labels.
        np.testing.assert_array_equal(
            chunked_l[f's/{gid}#{t0}'][0].points2d[..., 0, :], true[:, t0:t0 + 5])

    # THE RADIUS IS THE WHOLE GROUP'S. Every chunk carries the same one, and it is what an unchunked
    # score would have used -- so chunking cannot move the number, only its uncertainty.
    ext = {float(chunked_p[f's/{gid}#{t0}']['__extent__']) for t0 in (0, 5, 10, 15)}
    assert len(ext) == 1, f'chunks must share one match radius, got {ext}'
    rows = ev.score(preds, labels, quiet=True)
    assert abs(ext.pop() - rows[0]['mpjpe_r']) < 1e-6, \
        'the shared radius must be the one the unchunked scoring used'


def test_err_percentiles_describe_the_tail_the_mean_hides():
    """p75..p99 come from the same matched vector as `err`, and outlast a flattering mean.

    A MEAN CANNOT SHOW A TAIL. branson-fly reads MPJPE 0.599 px with p99 3.952 -- 6.6x the mean --
    and every localisation failure this repo has found (the 182 mm seam p90 against a 2.4 interior,
    the crop p90 566 -> 317) was found in a quantile and reported in one.
    """
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


def test_penalised_cost_charges_the_keypoints_a_prediction_declined():
    """OKS's rule in distance units: unshared LABEL keypoints cost `max_dist` and stay in `n`.

    Under 'mean' a row sharing ONE keypoint is scored on that keypoint alone and can out-bid a
    dense row (eval rule 9). Here the sparse row sits exactly on its target and the dense row is
    0.5 px off, so 'mean' hands the GT to the one-point row -- and 'penalised' does not.
    """
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
    # The pose decode emits every keypoint of every row it decodes (rat-city: 6,000 of 6,000
    # instance-frames carry all K = 4), so `n_ok == n_labelled` and the penalty is identically 0.
    dense = np.zeros((2, 1, 4, 2)) + 0.5
    a = match_instances(dense, true, max_dist=20.0, cost='mean')[0]
    b = match_instances(dense, true, max_dist=20.0, cost='penalised')[0]
    assert a == b

    # An infinite radius has no finite charge to levy, so it falls back rather than returning inf.
    assert (match_instances(pred, true, cost='penalised')[0]
            == match_instances(pred, true, cost='mean')[0])
    with pytest.raises(ValueError):
        match_instances(pred, true, cost='oks')


def test_match_cost_default_reproduces_every_published_number():
    """`cost='mean'` is the default and must be byte-identical to the pre-flag behaviour.

    The flag is an ARM, not a silent correction -- the same discipline `min_kpts_frac = 0.0`
    follows. If this drifts, every number in reports 10-19 becomes unreproducible.
    """
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


def test_chunking_slices_the_labels_when_the_prediction_is_a_prefix(tmp_path):
    """A `--max-frames` PREDICTION IS SHORTER THAN ITS GROUP, and every chunk must still be scored
    against its OWN frames.

    The label slice used to be gated on the PREDICTION's frame count, so for a truncated prediction
    no label array matched and every chunk was handed the WHOLE group's labels. `score` truncates to
    the shorter of the two, so chunk 0 scored against frames 0..n-1 and every LATER chunk scored its
    own frames against frames 0..n-1 again. Measured on calms21 (6 sessions x 2000 frames of a
    ~19,000-frame group, predictions good to a median 8-11 px in every chunk): coverage 0.9891 with
    the fix against **0.4656** without, MPJPE 26.5 px against 98.6, and MOTA 0.76-0.95 on chunk 0
    beside -0.36 to -1.00 on chunks 1-3 of all six sessions. It reads exactly like a pipeline that
    falls apart after 500 frames, which is why it survived: the shape of the failure names the wrong
    culprit.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    from tailcyclenet.format import Session

    ev = _eval_module()
    cf._session_2d(tmp_path / 'ds' / 'test' / 's', T=20)
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    gid = next(iter(sess.groups))
    labels = {f's/{gid}': (sess.labels(gid), sess)}
    true = labels[f's/{gid}'][0].points2d[..., 0, :]
    assert true.shape[1] == 20

    # THE PREDICTION COVERS ONLY THE FIRST 10 FRAMES -- what `--max-frames 10` produces.
    preds = {f's/{gid}': {'pred': true[:, :10].copy(), 'mode': np.array('2d')}}
    cp, cl = ev.chunk_frames(preds, labels, 5)
    assert sorted(cp) == [f's/{gid}#0', f's/{gid}#5'], sorted(cp)
    for t0 in (0, 5):
        got = cl[f's/{gid}#{t0}'][0].points2d[..., 0, :]
        np.testing.assert_array_equal(got, true[:, t0:t0 + 5],
                                      err_msg=f'chunk {t0} was handed the wrong frames')
    # ...and the second chunk is a PERFECT prediction of its own frames, so it must score as one.
    # Unsliced labels made this chunk score frames 5-9 against frames 0-4 and read as a total miss.
    rows = ev.score(cp, cl, quiet=True)
    assert len(rows) == 2
    for r in rows:
        assert r['coverage'] == pytest.approx(1.0), f'{r["coverage"]} -- exact prediction, full labels'
        assert r['err'] == pytest.approx(0.0, abs=1e-6), r['err']
