"""Instance-level pose NMS from detector keypoints. Pure functions, no state, no model.

WHAT IS LEFT HERE IS THE ONE KEYPOINT IDENTITY LEVER THAT SURVIVED MEASUREMENT, and the history is
the useful part of this file. Report 16 section 9 ranked six coarse aggregate cues; all six were
built, measured on BOTH roots and refuted, in three successive forms:

- as a VETO over an unchanged centroid cost (`--axis-veto`, `--kpt-affinity`): each beats its
  rate-matched random control on `idsw`, and each still LOSES against not vetoing at all (MOTA
  -0.257 and -0.142 on rat-city), because rejection is the wrong currency on a matcher already
  starved of candidates. On 3dpop -- where the cue is 3.4x more surgical AND the matcher is not
  starved, i.e. the friendliest case -- the veto is significantly worse on MPJPE, MOTA, miss and
  `idsw` alike. So the FORM was refuted rather than mis-thresholded.
- as a PERMUTATION (`--swap-repair`, `--kpt-centre`): conserves every edge exactly as designed and
  still makes `idsw` monotonically worse in its margin, because it triggers on a statistic report
  19 section 6 measured as a smoke alarm rather than a metric.
- as a HUNGARIAN COST (`--axis-cost`), the form report 19 section 14 argued for specifically
  because "a wrong cue shifts a ranking rather than deleting a candidate": it buys nothing at any
  weight on either root, and is harmful at K = 4 and at W = 1.0 on K = 17.

The ranking a K = 4 body axis supplies is too noisy to spend in ANY of those three forms. **Do not
re-propose a fourth without first raising the cue's quality.** dev/reports/19 and 21 (6, 6b).

`pose_nms` is different in kind, and that is why it lives: it does not rank a match at all, it
REMOVES a row that duplicates another row's animal, by keypoint containment rather than IoU. On
rat-city it is worth MOTA +0.0223 and beats its rate-matched random control by +0.058, removing
3.7x more duplicates at 17x less coverage cost. It is HARMFUL on calms21, so it is root-conditional
and default-off, and the discriminator is whether `fp_dup` is a live term for it to attack.

**IT IS QUANTISED BY K.** The overlap can only take {0, .25, .5, .75, 1} at K = 4, so the flag has
five settings and `0.6` and `0.7` are byte-identical. Quote it as a COUNT ("3 of 4"), never as a
fraction -- the `--min-match-kpts` trap again.

**A MISSING CUE ABSTAINS.** Every function here returns NaN where it has too little to say, and
every caller must read NaN as "no opinion".
"""
from __future__ import annotations

import numpy as np


def _valid(kpts):
    """(K,2+) -> (n,2) of the finite rows. Detector keypoints carry (x, y, score); score is ignored
    here on purpose -- thresholding it is a second lever and would be a second calibration."""
    k = np.asarray(kpts, float)
    if k.ndim != 2 or k.shape[0] == 0:
        return np.empty((0, 2))
    xy = k[:, :2]
    return xy[np.isfinite(xy).all(1)]


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
