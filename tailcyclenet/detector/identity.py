"""Coarse, AGGREGATE identity cues from detector keypoints. Pure functions, no state, no model.

Report 15 measured a per-keypoint error of **0.107 box sides** and concluded keypoints were "8x too
coarse for identity". Report 16 §8 retracted the general form of that: 0.107 refutes ONE statistic --
a median-of-norms epipolar residual, where the noise does not average down -- and sits comfortably
inside the bar for every cue that is coarse and aggregate. Orientation ranks contested rows correctly
**98.8%** of the time at that same measured error.

So everything here obeys two rules, and both are the difference between this and the refuted version:

- **AGGREGATE OVER K, never per keypoint.** A cue that reads one keypoint inherits the noise floor;
  a cue that reads all of them averages it down as 1/sqrt(K). PCA over K is the primary axis rather
  than a two-point one for exactly this: the noise budget is ~7 deg against ~11 deg, and on 3dpop the
  named head keypoint `hd_beak` is among the WORST-localised (0.117 box sides) while `bp1`/`bp3` are
  the best (0.068-0.075). Picking the two points whose names sound most like an axis picks two of the
  noisiest.
- **A VETO OVER AN UNCHANGED CENTROID COST, never a per-keypoint term added to one.** Report 15
  item 4's rule. The incumbent affinity stays exactly what it was; these only ever REMOVE an edge
  that was already inside the centre gate. That bounds the damage of a wrong cue to a missed match,
  and it keeps every calibrated threshold (`max_move` in box sides) untouched.

**A MISSING CUE ABSTAINS, IT NEVER VETOES.** Every function here returns NaN where it has too little
to say, and every caller must treat NaN as "no opinion" -- a detector that emitted two valid
keypoints must not be able to reject a match that the centroid accepted.
"""
from __future__ import annotations

import numpy as np

MIN_PTS = 3          # below this, PCA over 2D points is either undefined or exactly the two points


def _valid(kpts):
    """(K,2+) -> (n,2) of the finite rows. Detector keypoints carry (x, y, score); score is ignored
    here on purpose -- thresholding it is a second lever and would be a second calibration."""
    k = np.asarray(kpts, float)
    if k.ndim != 2 or k.shape[0] == 0:
        return np.empty((0, 2))
    xy = k[:, :2]
    return xy[np.isfinite(xy).all(1)]


def iso_null(n):
    """The s2/s1 an ISOTROPIC cloud of `n` points typically shows -- the abstention threshold.

    **A FIXED RATIO CANNOT SERVE TWO ROOTS, AND THIS IS THE ONE PLACE A FIXED CONSTANT WOULD HAVE
    BEEN SILENTLY WRONG.** A round animal does not give s2/s1 = 1; a FINITE sample of an isotropic
    distribution is elongated by sampling noise alone, and how much depends entirely on `n`.
    Measured over 4,000 draws per K (`identity.demo`):

        K            4      5     10     17     20     44
        median    0.415  0.492  0.656  0.741  0.752  0.832

    which is `1 - 1.1/sqrt(n)` to within 0.005 at every K measured. So a threshold of 0.9 -- the
    first thing anyone writes -- rejects nothing at all at K = 4 and only the roundest half at
    K = 44, and the cue would fire confidently on pure noise at both ends.

    The median rather than a tail quantile because ABSTAINING IS THE SAFE ERROR: a veto that
    abstains simply does not fire, while a veto that reads noise as an orientation removes a correct
    edge. At K = 17 this rejects ~0% of a 4:1 animal at the measured keypoint noise (p95 0.412) and
    half of a round one.

    **AT K = 4 THE GUARD BARELY SEPARATES ANYTHING** -- an isotropic cloud reads p50 0.415 against a
    4:1 animal's p95 0.345 -- so on a 4-keypoint root like rat-city the orientation cue is weak by
    construction, not by tuning. Read that as a bound on the 2D half rather than a threshold to
    loosen.
    """
    return 1.0 - 1.1 / np.sqrt(max(int(n), 2))


def body_axis(kpts, iso_max=None):
    """(K,2+) -> (2,) unit vector along the animal's long axis, or NaN if undecidable.

    PCA-1, i.e. the first right-singular vector of the centred points. UNDIRECTED: the sign of a
    principal component is arbitrary, so head-forward and tail-forward come back as the same axis.
    That is the honest output -- fixing the sign needs named keypoints, which `axis_sign` does.

    Returns NaN when fewer than `MIN_PTS` points are valid, and ALSO when the point cloud is round:
    a singular-value ratio above `iso_max` (default `iso_null(n)`, which is K-AWARE -- see there)
    has no long axis, and reporting the numerically-first direction of a circle as an orientation is
    how a noise-only cue gets a confident value.
    """
    p = _valid(kpts)
    if p.shape[0] < MIN_PTS:
        return np.full(2, np.nan)
    p = p - p.mean(0)
    try:
        _, s, vt = np.linalg.svd(p, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full(2, np.nan)
    lim = iso_null(p.shape[0]) if iso_max is None else float(iso_max)
    if s.shape[0] < 2 or s[0] <= 1e-9 or s[1] / s[0] > lim:
        return np.full(2, np.nan)
    return vt[0] / max(np.linalg.norm(vt[0]), 1e-12)


def axis_sign(kpts, head, tail):
    """Resolve `body_axis`'s sign from two NAMED keypoints -> (2,) or NaN.

    Only where the names exist and both points are finite. The named pair fixes the sign; it does
    NOT set the direction, because the two worst-localised keypoints on a root are often exactly the
    extremities somebody would name (3dpop `hd_beak`, 0.117 box sides). Axis from PCA, sign from the
    names -- each from what it is good at.
    """
    v = body_axis(kpts)
    k = np.asarray(kpts, float)
    if not np.isfinite(v).all() or not (0 <= head < k.shape[0] and 0 <= tail < k.shape[0]):
        return v
    d = k[head, :2] - k[tail, :2]
    if not np.isfinite(d).all():
        return v
    return v if float(d @ v) >= 0 else -v


def angle_gap(a, b, directed=False):
    """Angle between two axis vectors, in DEGREES. NaN if either abstains.

    `directed=False` (the default) folds to [0, 90]: `body_axis` has no sign, so an animal and the
    same animal reversed must not read as 180 deg apart. Pass `directed=True` only for vectors that
    went through `axis_sign`, where the flip is real information.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    c = float(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1.0, 1.0))
    ang = np.degrees(np.arccos(c))
    return ang if directed else min(ang, 180.0 - ang)


def body_length(kpts):
    """A scale for the animal, in pixels, robust to one flung keypoint. NaN if undecidable.

    TWICE THE RMS DISTANCE FROM THE CENTROID ALONG THE LONG AXIS, not the max pairwise extent. The
    max is an extremum over K noisy points, so it is biased UPWARD by the noise and the bias grows
    with K -- which is exactly wrong for a cue whose job is to say two detections are different
    sizes. An RMS is a mean and is not.
    """
    p, v = _valid(kpts), body_axis(kpts)
    if p.shape[0] < MIN_PTS or not np.isfinite(v).all():
        return np.nan
    t = (p - p.mean(0)) @ v
    return float(2.0 * np.sqrt(np.mean(t ** 2)))


def size_ratio(a, b):
    """`body_length` ratio, always >= 1 so a caller compares against one bound. NaN if either is."""
    la, lb = body_length(a), body_length(b)
    if not (np.isfinite(la) and np.isfinite(lb)) or min(la, lb) <= 1e-9:
        return np.nan
    return float(max(la, lb) / min(la, lb))


def kpt_in_box_frac(kpts, box):
    """Fraction of an animal's valid keypoints inside a box. NaN if it has none. In [0,1].

    Report 12's R5 term, never built until now. INSIDE/OUTSIDE ONLY -- a binary per keypoint, summed
    over K -- which is why it is the cue with the most headroom: a 0.107 box-side positional error
    moves a point across the boundary only where it was already within 0.107 of it, so the count is
    far more stable than any distance built from the same points.
    """
    p = _valid(kpts)
    b = np.asarray(box, float)
    if p.shape[0] == 0 or not np.isfinite(b).all():
        return np.nan
    inside = ((p[:, 0] >= b[0]) & (p[:, 0] <= b[2])
              & (p[:, 1] >= b[1]) & (p[:, 1] <= b[3]))
    return float(inside.mean())


def centroid(kpts, box=None, min_pts=MIN_PTS):
    """(K,2+) -> (2,) keypoint centroid, falling back to the box centre. Report 16 §9.2 item 4.

    **THE ONLY EDGE-CONSERVING CUE IN THE FAMILY.** Every other function here feeds a VETO, which
    removes a candidate pair; this one moves the affinity's POINT and leaves the pair set untouched.
    Report 19 §3 measured the matcher as starved of candidates and §4 measured the veto family as a
    net loss on exactly that ground, so the distinction is the whole reason this is worth building.

    It can only help by the part of the per-keypoint error that AVERAGES DOWN -- a whole-animal shift
    is common to every keypoint and is exactly what the box centre already carries. Report 19 §7
    measures that split on 3dpop: median per-animal common fraction **0.42**, middle half 0.20-0.75,
    so most of a typical animal's error IS independent and a centroid over K = 17 does average it
    down. (Read the robust split there, not the mean: mis-assigned boxes displace every keypoint
    together, which is perfect common mode, and they dominate a squared mean.)

    FALLS BACK RATHER THAN ABSTAINS, and that is the difference from the vetoes. A veto with no
    opinion must not fire; an affinity point must exist for every detection or the pair vanishes,
    which would make this a veto by the back door. Too few keypoints -> the box centre, unchanged.
    """
    p = _valid(kpts)
    if p.shape[0] < min_pts:
        if box is None:
            return np.full(2, np.nan)
        b = np.asarray(box, float)
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    return p.mean(0)


def epipolar(res, signed=False):
    """Reduce a set of per-keypoint epipolar residuals to ONE number. NaN if there are none.

    `res` is (K,) SIGNED distances from the epipolar line -- signed, because that is the whole
    point. Report 15's birth test took a median of NORMS, which is a median of |noise| and therefore
    a positive number for a correct pairing: it cannot average down, and it was refuted on exactly
    that ground. The SIGNED mean does average down as 1/sqrt(K), because a correct pairing's
    residuals straddle the line while a wrong pairing's sit to one side of it.

    `signed=False` reproduces the refuted statistic, kept so the two can be run as one lever apart
    on the same residuals rather than compared across reports.
    """
    r = np.asarray(res, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    return float(np.mean(r)) if signed else float(np.median(np.abs(r)))


def step_cost(prev_k, cur_k):
    """How surprising is it that `cur_k` is the same animal as `prev_k` one frame later?

    Axis turn in degrees plus the RELATIVE length change scaled to degrees, so the two are summed in
    one unit rather than with a weight nobody calibrated. 90 deg is the scale factor because that is
    the axis measure's full range (it is undirected), so a doubling in length costs the same as a
    right-angle turn. NaN where either cue abstains -- the caller must treat that as "no opinion",
    never as zero.

    This is the SAME definition report 19 §6 measured as a label-free swap proxy, and that
    circularity is the point and the trap: a repair that fires on this must NOT be scored on it.
    Score it on `idsw`/MOTA against labels (§6's `axis-veto 60` has the best median axis turn of any
    arm and 2.6x the baseline's idsw).
    """
    va, vb = body_axis(prev_k), body_axis(cur_k)
    la, lb = body_length(prev_k), body_length(cur_k)
    ang = angle_gap(va, vb)
    if not np.isfinite(ang) or not (np.isfinite(la) and np.isfinite(lb)) or min(la, lb) <= 1e-9:
        return np.nan
    return float(ang + 90.0 * abs(la - lb) / min(la, lb))


def swap_repair(boxes, kpts, scores=None, min_gain=30.0):
    """Report 16 §9.2 item 5: exchange two rows' SUFFIXES where a swap is indicated. In place.

    **THE ONLY PROPOSAL OF THE SIX STILL STANDING** after report 19: items 3 and 6 are dead on
    population size (§9), 1 and 2 are net losses in their veto form (§4), 4 is refuted on both roots
    (§11). This is the one that RE-SEATS rather than rejects -- it conserves every edge, which §4
    identifies as necessary on a matcher that is starved of candidates, and it uses the cue signal §4
    measured as real (both cues beat their rate-matched controls on idsw).

    A swap is a PERSISTENT event, so the repair is persistent: if rows a and b exchanged animals at
    frame t, then a holds b's animal for the rest of the clip, and the fix is to swap `[t:]` of both.
    Swapping a single frame would repair the discontinuity at t and create a second one at t+1.

    `min_gain` is in the units of `step_cost` (degrees), and it exists because two rows are ALWAYS
    exchangeable: with no margin this fires on noise everywhere and reorders the whole clip. 30 deg
    is a starting value, not a calibrated one -- sweep it and report the fire rate, per §4.

    O(T x S^2) over arrays that already exist, no model, no labels.
    """
    b = np.asarray(boxes)
    k = np.asarray(kpts)
    S, T = b.shape[0], b.shape[1]
    swaps = 0
    for t in range(1, T):
        for i in range(S):
            for j in range(i + 1, S):
                # Camera 0 decides. A row is ONE animal across cameras by construction here, so a
                # swap is a property of the row pair rather than of a view, and testing every camera
                # would weight the decision by how many cameras happened to see them.
                ci, cj = step_cost(k[i, t - 1, 0], k[i, t, 0]), step_cost(k[j, t - 1, 0], k[j, t, 0])
                si, sj = step_cost(k[i, t - 1, 0], k[j, t, 0]), step_cost(k[j, t - 1, 0], k[i, t, 0])
                if not np.isfinite([ci, cj, si, sj]).all():
                    continue                      # an abstention is not evidence for a swap
                if (ci + cj) - (si + sj) > min_gain:
                    for arr in (b, k) + ((np.asarray(scores),) if scores is not None else ()):
                        tmp = arr[i, t:].copy()
                        arr[i, t:] = arr[j, t:]
                        arr[j, t:] = tmp
                    swaps += 1
    return swaps


def demo():
    """assert-based, dependency-free:  pixi run python -m tailcyclenet.detector.identity"""
    rng = np.random.default_rng(0)
    # A 10-point animal 100 px long, lying along x, at (500, 300).
    t = np.linspace(-50, 50, 10)
    animal = np.stack([500 + t, 300 + 0 * t], -1)

    v = body_axis(animal)
    assert abs(abs(v[0]) - 1) < 1e-6, f'axis of an x-aligned animal must be x: {v}'
    assert angle_gap(v, [1, 0]) < 1e-6 and angle_gap(v, [-1, 0]) < 1e-6, \
        'the axis is UNDIRECTED: a reversed animal is the same axis'
    assert abs(angle_gap(v, [0, 1]) - 90) < 1e-6

    # Abstention, which is the property that keeps a veto from firing on nothing.
    assert not np.isfinite(body_axis(animal[:2])).any(), 'two points must abstain'
    assert np.isnan(kpt_in_box_frac(np.full((5, 2), np.nan), [0, 0, 10, 10]))

    # THE ISOTROPY GUARD IS K-AWARE, and the null it is set from is measured here rather than
    # assumed. A round animal must abstain at least half the time at every K; a fixed 0.9 -- the
    # obvious constant -- abstains on essentially NOTHING at small K, which is where the cue would
    # have read pure sampling noise as a confident orientation.
    for K in (4, 5, 10, 17, 20, 44):
        blobs = [rng.normal(0, 10, size=(K, 2)) for _ in range(600)]
        got = np.mean([np.isfinite(body_axis(b)).all() for b in blobs])
        assert got < 0.6, f'K={K}: an isotropic cloud got an axis {got:.0%} of the time'
        naive = np.mean([np.isfinite(body_axis(b, iso_max=0.9)).all() for b in blobs])
        assert naive > got, f'K={K}: a fixed 0.9 must be the LOOSER rule this replaced'
    assert np.isnan(angle_gap(body_axis(rng.normal(0, 10, size=(20, 2)) * 0 + 1), v)), \
        'an abstention must propagate as NaN'

    # ...and an elongated animal must still get one, at the measured keypoint noise. This is the
    # other side of the same threshold and the reason it is the null MEDIAN and not a tail.
    for K in (10, 17):
        tt = np.linspace(-50, 50, K)
        el = [np.stack([tt, 0 * tt], -1) + rng.normal(0, 10.7, (K, 2)) for _ in range(600)]
        keep = np.mean([np.isfinite(body_axis(e)).all() for e in el])
        assert keep > 0.95, f'K={K}: a 4:1 animal must keep its axis, got {keep:.0%}'

    # NOISE AVERAGES DOWN, which is the whole claim these cues rest on. At the measured 0.107 box
    # sides -- ~10.7 px on a 100 px animal -- the axis must still be good to a few degrees.
    errs = [angle_gap(body_axis(animal + rng.normal(0, 10.7, animal.shape)), v) for _ in range(400)]
    p95 = float(np.percentile(errs, 95))
    assert p95 < 25.0, f'axis p95 error {p95:.1f} deg at the measured keypoint noise'

    # ...and a TWO-POINT axis is worse on the same points, which is why PCA is the primary form.
    two = [angle_gap((animal + rng.normal(0, 10.7, animal.shape))[[0, -1]][1]
                     - (animal + rng.normal(0, 10.7, animal.shape))[[0, -1]][0], v)
           for _ in range(400)]
    assert float(np.percentile(two, 95)) > p95, 'PCA over K must beat the two-point axis'

    # Length is a scale, and the ratio is orientation-free.
    turned = np.stack([500 + 0 * t, 300 + t], -1)
    assert abs(body_length(animal) - body_length(turned)) < 1e-6
    assert abs(size_ratio(animal, animal) - 1.0) < 1e-9
    assert size_ratio(animal, 500 + (animal - 500) * 0.5) > 1.5, 'half-size must read as a ratio'
    assert abs(angle_gap(body_axis(turned), v) - 90) < 1e-6

    # The centroid FALLS BACK rather than abstaining -- an affinity point must exist for every
    # detection, or moving the point would silently become a veto.
    assert np.allclose(centroid(animal), [500, 300])
    assert np.allclose(centroid(animal[:1], box=[0, 0, 10, 20]), [5, 10]), 'must fall back'
    assert not np.isfinite(centroid(animal[:1])).any(), 'no box and no points -> NaN, not a guess'
    # AND IT AVERAGES INDEPENDENT NOISE DOWN, which is the entire case for item 4: at K points the
    # centroid's error is 1/sqrt(K) of one keypoint's. A common-mode shift is NOT reduced -- that is
    # the null this is measured against in report 19 §7.
    ind = np.mean([np.linalg.norm(centroid(animal + rng.normal(0, 10.7, animal.shape)) - [500, 300])
                   for _ in range(400)])
    assert ind < 10.7 / 2, f'independent noise must average down over K=10, got {ind:.2f} px'
    shifted = [centroid(animal + np.array([8.0, 0.0])) for _ in range(5)]
    assert abs(shifted[0][0] - 508) < 1e-6, 'a common-mode shift must pass straight through'

    # in-box fraction, and the sign convention of the epipolar reduction.
    assert kpt_in_box_frac(animal, [440, 240, 560, 360]) == 1.0
    assert kpt_in_box_frac(animal, [495, 295, 505, 305]) < 0.3
    straddle = np.array([-3.0, 2.0, -1.0, 4.0, -2.0])          # a CORRECT pairing: noise about 0
    one_side = straddle + 8.0                                   # a WRONG one: displaced
    assert abs(epipolar(straddle, signed=True)) < abs(epipolar(one_side, signed=True)), \
        'the signed mean must separate a straddling residual from a displaced one'
    assert epipolar(straddle) > 0, 'the median-of-norms is positive even for a correct pairing'

    # SWAP REPAIR. Two animals at right angles to each other, whose rows swap at frame 10 -- the
    # exact event the repair exists to undo. Build it as ground truth, corrupt it, and check the
    # repair restores the original rather than merely reducing some cost.
    T, K = 20, 8
    u = np.linspace(-40, 40, K)
    true_k = np.zeros((2, T, 1, K, 2))
    for t in range(T):
        true_k[0, t, 0] = np.stack([300 + u, 200 + 0 * u], -1)          # along x
        true_k[1, t, 0] = np.stack([700 + 0 * u, 600 + u], -1)          # along y
    true_b = np.zeros((2, T, 1, 4))
    for s in range(2):
        for t in range(T):
            p = true_k[s, t, 0]
            true_b[s, t, 0] = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
    bad_k, bad_b = true_k.copy(), true_b.copy()
    bad_k[[0, 1], 10:] = bad_k[[1, 0], 10:]
    bad_b[[0, 1], 10:] = bad_b[[1, 0], 10:]
    n = swap_repair(bad_b, bad_k)
    assert n == 1, f'exactly one swap should be found, got {n}'
    assert np.allclose(bad_k, true_k), 'the repair must RESTORE the rows, not just reduce a cost'

    # AND IT MUST NOT FIRE ON CLEAN DATA. A repair that reorders a correct clip is worse than none,
    # and with no margin two rows are always exchangeable -- which is what `min_gain` is for.
    clean_b, clean_k = true_b.copy(), true_k.copy()
    assert swap_repair(clean_b, clean_k) == 0, 'must not fire on a clip with no swap'
    assert np.allclose(clean_k, true_k)
    print('identity: ok')


if __name__ == '__main__':
    demo()


def pose_nms(boxes, kpts, scores=None, thresh=0.8, stats=None):
    """INSTANCE-LEVEL NMS on the pose rows. maDLC's rule, not IoU. Report 20 lead 1.

    Per frame, for every pair of live rows, the overlap is

        min(#kpts of A inside B's box / |A|,  #kpts of B inside A's box / |B|)

    -- `Assembly.intersection_with` in DeepLabCut -- and above `thresh` the LOWER-SCORED row is
    dropped. Modifies `boxes` (and `kpts`) in place; returns the number of rows dropped.

    WHY NOT IoU, WHICH IS THE OBVIOUS CHOICE. The same reason `--link-boxes` abandoned it: two
    touching animals overlap almost equally by IoU, and it is exactly zero under fast motion where
    it cannot rank at all. Replaying calms21 frame 301->302, IoU scored the WRONG mouse at 0.512
    against the right one's 0.233. A keypoint-containment fraction asks a different question -- "is
    this row's animal the same animal as that row's" -- and degrades gracefully under occlusion,
    where a box shrinks and IoU falls off a cliff.

    WHAT IT IS FOR. `fp_dup` is a second prediction on an animal something else already claimed, and
    nothing in this repo addresses duplicates at the INSTANCE level: the detector's own NMS is
    per-box IoU inside `decode`, before any row assignment. On calms21 90% of the detector-minus-GT
    FP rise is `fp_dup`, against ~10% on 3dpop -- the two roots want opposite fixes, so this is
    aimed at the crowded-overlap case and should be a near-no-op on the sparse one.

    ASYMMETRIC BY CONSTRUCTION, and the `min` is what makes it safe: a small animal wholly inside a
    large animal's box scores 1.0 one way and a small fraction the other, so `min` keeps it -- two
    animals, one occluding the other, are not duplicates. Only a genuine double-detection scores
    high BOTH ways.
    """
    # NO dtype CONVERSION. `np.asarray(x, float)` on a float32 array returns a COPY, so every
    # in-place drop below would land on a temporary and the caller would see nothing -- which is
    # exactly what the first version did, and it looked like "the lever fires and changes nothing".
    b, k = boxes, kpts
    S, T = b.shape[0], b.shape[1]
    sc = scores
    dropped, pairs = 0, 0
    if k is None:
        return 0
    for t in range(T):
        live = [i for i in range(S) if np.isfinite(b[i, t, 0]).all()]
        for ii, i in enumerate(live):
            for j in live[ii + 1:]:
                if not np.isfinite(b[i, t, 0]).all() or not np.isfinite(b[j, t, 0]).all():
                    continue
                a = kpt_in_box_frac(k[i, t, 0], b[j, t, 0])
                c = kpt_in_box_frac(k[j, t, 0], b[i, t, 0])
                if not (np.isfinite(a) and np.isfinite(c)):
                    continue
                ov = min(a, c)
                pairs += 1
                if ov <= thresh:
                    continue
                # Lower score loses; with no scores, the higher row index loses (stable).
                si = -np.inf if sc is None else np.nanmax(sc[i, t])
                sj = -np.inf if sc is None else np.nanmax(sc[j, t])
                loser = j if (sj < si or (sj == si and j > i)) else i
                b[loser, t] = np.nan
                k[loser, t] = np.nan
                dropped += 1
    if stats is not None:
        stats['nms_pairs'] = stats.get('nms_pairs', 0) + pairs
        stats['nms_dropped'] = stats.get('nms_dropped', 0) + dropped
    return dropped


def stitch_rows(boxes, kpts=None, scores=None, max_gap=24, max_move=1.0, stats=None):
    """Bridge a row's temporal GAPS to another row's fragment. Report 20 lead 2, APT's rung.

    `link_rows` seats a birth only into a row whose `last` is ENTIRELY non-finite, which takes
    `max_age = 24` frames of absence -- so an animal that vanishes for a few frames and comes back
    is picked up by whatever row is free, and one animal ends up split across two rows with
    complementary gaps. Report 18 §5 measured the other end of this: 34.4% of offered detections
    dropped, **29 percentage points of them INSIDE the gate**, i.e. they lost the Hungarian and had
    nowhere to go. Report 19 §3's S sweep is the same fact again -- coverage and MPJPE improve
    monotonically from S = 12 to S = 36 while MOTA collapses through the FP term alone: the
    information exists and the bookkeeping discards it.

    THE RUNG, NOT THE LADDER. DeepLabCut solves this as a min-cost flow, which is a node-disjoint
    path COVER -- every fragment is forced onto some animal, including junk. That is exactly the
    failure `--birth-age` already produced here (union crop p99 590 -> 3,804 px against a 244 px
    rat), so this takes APT's cheaper form: greedy, gap-bounded, and it only ever MERGES two rows
    whose live frames do not overlap.

    Merge rule, greedy over pairs sorted by gap length:
      * the two rows must not both be live on any frame (disjoint supports);
      * the gap between one's last live frame and the other's first must be <= `max_gap`;
      * the box centres either side of the gap must be within `max_move` mean box sides, the SAME
        gate `link_rows` uses, so a merge cannot do what a single link step would have refused.

    Returns the number of merges. In place.
    """
    b = boxes
    S, T = b.shape[0], b.shape[1]
    live = [np.flatnonzero(np.isfinite(b[i, :, 0]).all(-1)) for i in range(S)]
    merged = 0
    cand = []
    for i in range(S):
        if live[i].size == 0:
            continue
        for j in range(S):
            if i == j or live[j].size == 0:
                continue
            # j must start strictly after i ends, with a bounded gap.
            gap = int(live[j][0]) - int(live[i][-1])
            if gap <= 0 or gap > max_gap:
                continue
            if np.intersect1d(live[i], live[j]).size:
                continue
            cand.append((gap, i, j))
    cand.sort()
    dead = set()
    for gap, i, j in cand:
        if i in dead or j in dead or live[i].size == 0 or live[j].size == 0:
            continue
        ta, tb = int(live[i][-1]), int(live[j][0])
        ba, bb = b[i, ta, 0], b[j, tb, 0]
        if not (np.isfinite(ba).all() and np.isfinite(bb).all()):
            continue
        ca = np.array([(ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2])
        cb = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        sa = 0.5 * ((ba[2] - ba[0]) + (ba[3] - ba[1]))
        sb = 0.5 * ((bb[2] - bb[0]) + (bb[3] - bb[1]))
        side = 0.5 * (sa + sb)
        if side <= 0 or np.linalg.norm(ca - cb) > max_move * side:
            continue
        # Move j's frames into i, then retire j.
        sel = live[j]
        b[i, sel] = b[j, sel]
        b[j, sel] = np.nan
        if kpts is not None:
            kpts[i, sel] = kpts[j, sel]
            kpts[j, sel] = np.nan
        if scores is not None:
            scores[i, sel] = scores[j, sel]
            scores[j, sel] = np.nan
        live[i] = np.union1d(live[i], sel)
        live[j] = np.array([], int)
        dead.add(j)
        merged += 1
    if stats is not None:
        stats['stitch_candidates'] = stats.get('stitch_candidates', 0) + len(cand)
        stats['stitch_merged'] = stats.get('stitch_merged', 0) + merged
    return merged
