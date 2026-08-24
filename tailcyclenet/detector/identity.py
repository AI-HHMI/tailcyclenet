"""Instance-level pose NMS from detector keypoints. Pure functions, no state, no model.

WHAT IS LEFT HERE IS THE ONE KEYPOINT IDENTITY LEVER THAT SURVIVED MEASUREMENT. Six coarse
aggregate cues were built, measured on BOTH roots and refuted, in three successive forms -- as a
VETO over an unchanged centroid cost, as a PERMUTATION, and as a HUNGARIAN COST -- and the
ranking a K = 4 body axis supplies is too noisy to spend in ANY of those three forms. **Do not
re-propose a fourth without first raising the cue's quality.**

`pose_nms` is different in kind, and that is why it lives: it does not rank a match at all, it
REMOVES a row that duplicates another row's animal, by keypoint containment rather than IoU. It
is root-conditional and default-off: it helps where `fp_dup` is a live term and is harmful on a
root at ceiling.

**IT IS QUANTISED BY K.** The overlap can only take {0, .25, .5, .75, 1} at K = 4, so the flag
has five settings and `0.6` and `0.7` are byte-identical. Quote it as a COUNT ("3 of 4"), never
as a fraction.

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

    INSIDE/OUTSIDE ONLY -- a binary per keypoint, summed over K -- which is why it is the cue with
    the most headroom: a 0.107 box-side positional error moves a point across the boundary only
    where it was already within 0.107 of it, so the count is far more stable than any distance
    built from the same points.
    """
    p = _valid(kpts)
    b = np.asarray(box, float)
    if p.shape[0] == 0 or not np.isfinite(b).all():
        return np.nan
    inside = ((p[:, 0] >= b[0]) & (p[:, 0] <= b[2])
              & (p[:, 1] >= b[1]) & (p[:, 1] <= b[3]))
    return float(inside.mean())


def pose_nms(boxes, kpts, scores=None, thresh=0.8, stats=None):
    """INSTANCE-LEVEL NMS on the pose rows. maDLC's rule, not IoU.

    Inputs:
        boxes (S,T,C,4), kpts (S,T,C,K,3) -- the seated pose rows; kpts=None is a no-op.
        scores -- per-row scores (S,T,C), or None (higher row index loses on ties).
        thresh -- minimum pairwise containment overlap to drop the lower-scored row.
        stats -- optional dict; accumulates `nms_pairs` / `nms_dropped`.
    Outputs:
        Number of rows dropped.
    Side effects:
        Modifies `boxes` and `kpts` in place. No dtype conversion is deliberate:
        `np.asarray(x, float)` on float32 returns a COPY, so a conversion would drop onto a
        temporary and the caller would see nothing.
    Notes:
        Overlap is min(A's kpts inside B's box / |A|, B's kpts inside A's box / |B|) -- a
        keypoint-containment fraction, not IoU: IoU is ~equal for touching animals and exactly
        zero under fast motion. The `min` makes it safe: a small animal wholly inside a large
        one scores high only one way, so occlusion is not a duplicate. It addresses INSTANCE-
        level `fp_dup` (the detector's own NMS is per-box, before any row assignment); aimed at
        crowded-overlap roots, near-no-op on sparse ones. 3D-aware: liveness is "finite in ANY
        camera" and the pair overlap is the mean of the per-camera fractions over co-occurring
        cameras; on C=1 byte-identical to the old camera-0-only computation.
    """
    b, k = boxes, kpts
    S, T, C = b.shape[0], b.shape[1], b.shape[2]
    sc = scores
    dropped, pairs = 0, 0
    if k is None:
        return 0
    for t in range(T):
        live = [i for i in range(S) if np.isfinite(b[i, t]).any()]
        for ii, i in enumerate(live):
            for j in live[ii + 1:]:
                cams = [c for c in range(C)
                       if np.isfinite(b[i, t, c]).all() and np.isfinite(b[j, t, c]).all()]
                if not cams:
                    continue
                a_c = [kpt_in_box_frac(k[i, t, c], b[j, t, c]) for c in cams]
                c_c = [kpt_in_box_frac(k[j, t, c], b[i, t, c]) for c in cams]
                if not (np.isfinite(a_c).any() and np.isfinite(c_c).any()):
                    continue
                a = float(np.nanmean(a_c))
                c = float(np.nanmean(c_c))
                ov = min(a, c)
                pairs += 1
                if ov <= thresh:
                    continue
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
