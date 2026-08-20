#!/usr/bin/env python
"""Score a prediction file against the labels. Offline and model-free.

    pixi run python scripts/eval.py pred.npz --data <dataset> --split test

Nothing here loads a checkpoint, so a metric can be recomputed, decomposed or re-thresholded
without a GPU and without any risk of scoring a different model than the one that predicted.

What it prints, and why each column is there:

- `err` alone is a trap: it is a mean over MATCHED points, so a model that declines the hard
  points looks better. `cov` is the fraction of labelled points it attempted; read them together.
- the interval is a PAIRED bootstrap over groups, not over points. Points inside a group are
  correlated, and resampling them would give an interval several times too tight.
- multi-animal rows add MOTA with its components split out. Two methods with the same MOTA and
  different miss/FP splits are not the same method, and only MOTA replicates across seeds --
  above a +-0.023 seed floor. The FP term splits again into `dup` (a second prediction on an
  animal something else already claimed) and `none` (a prediction on no labelled animal): the
  first is what arbitration removes and the second what a score threshold removes.

`--vs other.npz` reports the difference between two prediction files over the groups both scored
and the points BOTH matched, under a paired bootstrap. That restriction is the point: an arm which
declines the hard points has a better mean over its own matched set, so an unrestricted delta
rewards lost coverage (eval rule 6).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.format import INST_PRESENT, UNLABELED, VISIBLE, sessions_for
from tailcyclenet.infer.predictions import load_predictions
from tailcyclenet.metrics import (ERR_PCTS, error_and_coverage, match_instances, matched_error,
                                  mota, motion_ratio, paired_bootstrap, pck)

_PCT_KEYS = tuple(f'p{p}' for p in ERR_PCTS)


def label_lookup(data: Path, split: str):
    """{session/group: (Labels, Session)} for every group in the split."""
    _, sessions = sessions_for(Path(data), split)
    out = {}
    for sess in sessions:
        sess.preload()
        for gid in sess.groups:
            out[f'{sess.session_id}/{gid}'] = (sess.labels(gid), sess)
    return out


def chunk_frames(preds, labels, n):
    """Split every group into `n`-frame pieces, each its own scoring unit. -> (preds, labels).

    **THE BOOTSTRAP RESAMPLES GROUPS, AND A LONG CLIP IS ONE GROUP.** rat-city's whole test split is
    a single 500-frame group, so every delta measured on it comes back `DEGENERATE (one group -- no
    interval exists)` -- and the roots this repo most wants long-clip numbers from are exactly the
    ones with the fewest groups. The protocol and the statistic disagreed about what a sample is.

    A chunk is the right resampling unit for a clip: consecutive frames are correlated, whole clips
    are too few, and a chunk of a few seconds is long enough to contain the episodic failures that
    matter (report 19 §1: bad windows come in contiguous bursts) and short enough to give a usable n.

    It is NOT free of assumptions and the direction is knowable: chunks from one clip are more alike
    than independent clips would be, so an interval from this is OPTIMISTIC. Quote it as within-clip
    uncertainty, never as though the clip were a sample of clips.
    """
    import dataclasses

    out_p, out_l = {}, {}
    for key, out in preds.items():
        if key not in labels:
            out_p[key] = out
            continue
        lab, sess = labels[key]
        T = int(np.asarray(out['pred']).shape[1])
        # THE MATCH RADIUS MUST COME FROM THE WHOLE GROUP, NOT THE CHUNK. `score` derives it as the
        # median animal extent over whatever it is given, so per-chunk it varies with the sample --
        # measured on this clip at 27.6 to 101.9 px across ten 50-frame chunks, a 3.7x swing. That
        # would make chunking change the METRIC rather than only the resampling unit, and chunks
        # scored under different radii are not exchangeable, which is the one thing a bootstrap
        # needs them to be. Computed once here and carried in.
        full = lab.points3d if str(out['mode']) == '3d' else lab.points2d[..., 0, :]
        with np.errstate(all='ignore'):
            span = np.nanmax(full, axis=2) - np.nanmin(full, axis=2)
            extent = float(np.nanmedian(np.linalg.norm(span, axis=-1)))
        # THE LABELS' TIME AXIS IS NOT THE PREDICTION'S. A prediction may be a PREFIX of its group
        # (`--max-frames`), and the label slice below used to be gated on `v.shape[1] == T` -- the
        # PREDICTION's length -- so for a truncated prediction NO label array matched and every
        # chunk was handed the WHOLE group's labels. `score` then truncates to the shorter, so chunk
        # 0 scored against frames 0..n-1 and every later chunk scored its own frames against frames
        # 0..n-1 AGAIN, i.e. against the wrong ones. Measured on calms21 (6 sessions x 2000 frames,
        # predictions good to a median 8-11 px in EVERY chunk): coverage 0.9903 unchunked against
        # 0.4656 at `--chunk 500`, with MOTA 0.76-0.95 on chunk 0 and -0.36 to -1.00 on chunks 1-3
        # of all six. It reads exactly like a pipeline that falls apart after 500 frames.
        # Frame indices are absolute within the group for both sides, so the fix is to gate on the
        # LABELS' own length.
        lab_T = int(np.asarray(full).shape[1])
        for t0 in range(0, T, n):
            t1 = min(t0 + n, T)
            if t1 - t0 < 2:                       # gotcha 1: T = 1 is not a usable window
                continue
            sub = {}
            for k, v in out.items():
                a = np.asarray(v)
                # AXIS 1 IS TIME, but only for the arrays that are frame-indexed. `outcome` and
                # `crop` are WINDOW-indexed on the same axis, so they are left whole rather than
                # sliced by a frame range that means nothing to them.
                sub[k] = a[:, t0:t1] if (a.ndim >= 2 and a.shape[1] == T) else v
            fields = {f.name: getattr(lab, f.name) for f in dataclasses.fields(lab)}
            for k, v in fields.items():
                if isinstance(v, np.ndarray) and v.ndim >= 2 and k != 'regions':
                    # `ext` is (C,T,4,4) -- time on axis 1 as well, so the same rule covers it.
                    if v.shape[1] == lab_T:
                        fields[k] = v[:, t0:t1]
            if isinstance(fields.get('regions'), np.ndarray) and fields['regions'].size:
                r = fields['regions']
                fields['regions'] = r[(r[:, 0] >= t0) & (r[:, 0] < t1)]
            sub['__extent__'] = np.float64(extent)
            out_p[f'{key}#{t0}'] = sub
            out_l[f'{key}#{t0}'] = (dataclasses.replace(lab, **fields), sess)
    return out_p, out_l


def score(preds, labels, mota_dist=None, quiet=False, min_kpts_frac=0.0, match_cost='mean'):
    """One row per group: the error, the coverage behind it, and MOTA where there are instances.

    Factored out of `main` so `--vs` scores the second file through the IDENTICAL path. A baseline
    computed a different way is not a baseline (eval rule 2), and a paired comparison whose two
    sides derived their matching separately is not paired.
    """
    rows = []
    for key, out in sorted(preds.items()):
        if key not in labels:
            if not quiet:
                print(f'{key}: no labels, skipped')
            continue
        lab, sess = labels[key]
        mode = str(out['mode'])
        true = lab.points3d if mode == '3d' else lab.points2d[..., 0, :]
        pred = out['pred']
        # SURPLUS PREDICTED ROWS ARE NOT DISCARDABLE. A detector offers as many animals as it
        # finds, which on 3dpop's ten pigeons is routinely more or fewer than the label table
        # holds, and truncating `pred` to `true`'s row count silently deleted whichever rows sat
        # past the end -- as coverage, not as false positives. `matched_error`, `match_instances`
        # and `mota` all take Sp != St; only `error_and_coverage` needs equal shapes, and it is the
        # row-indexed number that is meaningless under detector boxes anyway.
        T = min(pred.shape[1], true.shape[1])
        pred, true = pred[:, :T], true[:, :T]
        Sp, St = pred.shape[0], true.shape[0]
        S = max(Sp, St)

        m = error_and_coverage(pred[:min(Sp, St)], true[:min(Sp, St)])
        if S > 1:
            # Row index is not identity once boxes come from a detector. Match, then measure --
            # and take the matched COUNTS too, so the coverage quoted below describes the same
            # points as the error above it.
            if '__extent__' in out:
                extent = float(out['__extent__'])      # the WHOLE group's, carried by --chunk
            else:
                with np.errstate(all='ignore'):
                    span = np.nanmax(true, axis=2) - np.nanmin(true, axis=2)
                    extent = float(np.nanmedian(np.linalg.norm(span, axis=-1)))
            # ZERO IS NOT A RADIUS, AND IT IS FINITE. An instance-frame with exactly ONE finite
            # labelled keypoint has a keypoint-box diagonal of 0, not NaN, so the median can be 0
            # on a sparsely-labelled root -- rat-city labels 2.02 of 4 points per animal-frame,
            # one draw away. `match_instances` then admits nothing, err/median come back NaN,
            # coverage 0 and MOTA goes to -(1 + fp_rate): a catastrophic model failure that is
            # entirely an artefact of the radius.
            max_dist = extent if np.isfinite(extent) and extent > 0 else np.inf
            mm = matched_error(pred, true, max_dist=max_dist,
                               min_kpts_frac=min_kpts_frac, cost=match_cost)
            m.update({k: v for k, v in mm.items()
                      if k in ('err', 'median', 'coverage', 'n_true', 'n_matched')
                      or k in _PCT_KEYS})
            m['unmatched'] = mm.get('unmatched_true', 0)
            # AND PCK GETS THE SAME PAIRING. It reads `_pred` positionally, so on detector boxes
            # it was scoring animal i's prediction against animal j's label under a heading that
            # said HUNGARIAN-MATCHED -- branson-fly read MPJPE 0.705 px beside pck@5 0.0000,
            # which cannot both be true of one set of points.
            #
            # NOT shared with MOTA, which reads the RAW rows below. Hungarian-aligning rows
            # before MOTA would zero `idsw` by construction -- an identity switch is precisely a
            # frame where the best assignment changed, and re-matching every frame independently
            # defines that away.
            # Shaped like TRUE, not like pred: `j` is a label row, and PCK below concatenates this
            # against `_true`. With more predicted rows than labelled ones the pred-shaped version
            # merely carried dead rows; with fewer, `aligned[j]` indexed off the end.
            aligned = np.full_like(true, np.nan)
            for t, frame_pairs in enumerate(match_instances(pred, true, max_dist,
                                                            min_kpts_frac)):
                for i, j, _ in frame_pairs:
                    aligned[j, t] = pred[i, t]
            m['_pred_matched'] = aligned
            m['mpjpe_r'] = max_dist
        m['group'] = key
        m['mode'] = mode
        m['S'] = S
        m['S_pred'], m['S_true'] = Sp, St
        m['_pred'], m['_true'] = pred, true      # already sliced to (S, T); PCK reuses these
        # HOW MUCH THE PREDICTION MOVES, against how much the animal did. Unpaired here and only a
        # screen: a carried prompt low-passes the prediction, which no error, coverage or MOTA column
        # can see, and which every consistency statistic rewards. `--vs` reports the paired form.
        mr = motion_ratio(m.get('_pred_matched', pred), true)
        m['motion_ratio'] = mr['ratio'] if mr['n_steps'] else None
        # WHERE THE POSE LANDED RELATIVE TO THE BOX IT WAS CROPPED FROM, in units of one box side.
        # Written by `run_group` per (animal, window, camera): a pose that does not sit on its own
        # crop is not a prediction of that animal, and it is the same quantity in every unit system.
        if 'box_agree' in out:
            ba = np.asarray(out['box_agree'], float)
            ba = ba[np.isfinite(ba)]
            m['box_agree'] = float(np.median(ba)) if ba.size else None
            m['box_agree_p99'] = float(np.quantile(ba, 0.99)) if ba.size else None
        # `kpt_agree` IS THE 2D HALF OF THE SAME CHECK, and it was written and never read.
        # `box_agree` is structurally bounded in 2D -- the pose is decoded inside its own crop, so
        # its centroid can be at most about half a box side from the centre by construction, and
        # every 2D arm measures p99 0.31-0.56 (branson-fly 0.153). The DETECTOR's keypoints are
        # regressed in the full frame, so `kpt_agree` has no such ceiling and is the diagnostic
        # `box_agree` can only be in 3D. Same units -- one box side -- so one threshold means the
        # same thing on every root, which is exactly what `--vis-thresh`'s logit does not.
        if 'kpt_agree' in out:
            ka = np.asarray(out['kpt_agree'], float)
            ka = ka[np.isfinite(ka)]
            m['kpt_agree'] = float(np.median(ka)) if ka.size else None
            m['kpt_agree_p99'] = float(np.quantile(ka, 0.99)) if ka.size else None
        m.update(_vis_confusion(out, lab, mode, T))
        if S > 1:
            m['mota_r'], m['mota'] = _mota_for(m, lab, mota_dist, min_kpts_frac,
                                               extent_override=out.get('__extent__'),
                                               match_cost=match_cost)
        rows.append(m)
    return rows


def _vis_confusion(out, lab, mode, T):
    """Score the DECISION to omit a keypoint, separately from where the kept ones landed.

    SLEAP reports this as `vis.precision` / `vis.recall` and nothing here did. It matters because
    `vis_pred` is read as a row gate by `--vis-thresh` and its TARGET is corrupted on two roots by
    conversion decisions that are documented and therefore only caught by whoever read the docs:
    calms21's converter writes every point `VISIBLE`, so the head trained against an all-true
    target and reported `occlusion_acc = 1.0`; `rat-city-annotated` writes APT's occluded-but-placed
    points (2,500 slots, 22.7% of its visible rows) as `visible` by decision, leaving 171 real
    negatives in 11,168 rows.

    Printed, both of those become a `base` near 1.000 with `recall` 1.000 and `precision` at the
    base rate -- a number that says "this head has no negatives, do not gate on it" on ANY new root,
    without anyone having to remember. It is a guard and not an accuracy figure: a clean confusion
    matrix is still not a licence to gate, which is what the rate-matched random control is for.
    """
    # `conf` IS the per-keypoint `vis_pred` logit, (S,T,K) -- `run_group` writes it under that name.
    # The label side is the status channel: `vis3d` in 3D, and the FIRST camera's `vis2d` in 2D,
    # which is the same slice `true` was taken from above.
    st = lab.vis3d if mode == '3d' else (None if lab.vis2d is None else lab.vis2d[..., 0])
    if 'conf' not in out or st is None:
        return {}
    vp, st = np.asarray(out['conf'], float), np.asarray(st, int)
    n, k = min(vp.shape[0], st.shape[0]), min(vp.shape[2], st.shape[2])
    vp, st = vp[:n, :T, :k], st[:n, :T, :k]
    if vp.shape != st.shape:
        return {}
    # UNLABELED is not a negative -- it is the absence of an assessment, and counting it as
    # "not visible" would manufacture the very negatives these two roots do not have.
    ok = np.isfinite(vp) & (st != UNLABELED)
    if not ok.any():
        return {}
    yhat, y = vp[ok] > 0.0, st[ok] == VISIBLE
    tp, fp, fn = int((yhat & y).sum()), int((yhat & ~y).sum()), int((~yhat & y).sum())
    return {'vis_precision': tp / (tp + fp) if tp + fp else float('nan'),
            'vis_recall': tp / (tp + fn) if tp + fn else float('nan'),
            'vis_base': float(y.mean()), 'vis_n': int(y.size)}


def _mota_for(m, lab, mota_dist, min_kpts_frac=0.0, extent_override=None, match_cost='mean'):
    """(radius, mota dict) for one group. The ignore region is built here, from the labels.

    `extent_override` is the WHOLE group's animal extent, supplied by `--chunk`. Without it each
    chunk sizes its own radius from its own sample and the radius swings 3.7x across ten chunks of
    one clip -- which makes chunks scored under different radii non-exchangeable, and exchangeable
    is exactly what a bootstrap needs them to be.
    """
    pred, true = m['_pred'], m['_true']
    St, T = true.shape[0], true.shape[1]
    # Match radius scaled to the animal's own size: the DIAGONAL of its keypoint bounding box, not
    # a per-axis span. Taking the per-axis span and median-ing it across axes gave 1.6 px on
    # rat-city -- tighter than the labelling noise, so every instance read as a miss AND a false
    # positive and MOTA went negative.
    if extent_override is not None:
        extent = float(extent_override)
    else:
        with np.errstate(all='ignore'):
            span = np.nanmax(true, axis=2) - np.nanmin(true, axis=2)
            extent = np.nanmedian(np.linalg.norm(span, axis=-1))
    # `is None`, not `or`: `--mota-dist 0` is falsy and was silently read as "unset". And the same
    # degenerate-extent case as above -- a zero radius makes every instance a miss AND a false
    # positive at once, which is the failure this line's own comment describes.
    radius = float(mota_dist) if mota_dist is not None else float(extent) * 0.5
    if not radius > 0:
        radius = np.inf
        print(f'  MOTA: the labelled extent is {extent}, so the match radius is degenerate -- '
              'scoring without one. Too few labelled keypoints per instance-frame to size it.')
    # Present-but-unannotated animals, WITH their boxes where the format carries them: an unmatched
    # prediction is then excused only if it actually lands on one. Without boxes the fallback
    # excuses every unmatched prediction on such a frame, so `fp_ignored` is printed to show how
    # much of the FP term that took.
    ig = ig_boxes = None
    if lab.instance is not None:
        ig = (lab.instance[:St, :T] == INST_PRESENT).any(-1)
        # BOXES ARE PIXELS. `_in_ignore` compares a prediction centroid against them componentwise,
        # and in 3D that centroid is world millimetres in a frame the boxes know nothing about --
        # so every test fails and the whole ignore region silently degrades to the box-free
        # fallback, which excuses MORE. Presence alone, printed as `fp_ignored`, is the honest
        # answer there.
        if lab.boxes is not None and m['mode'] == '2d':
            ig_boxes = lab.boxes[:St, :T, 0]              # xyxy in the first camera
    return radius, mota(pred, true, radius, ignore=ig, ignore_boxes=ig_boxes,
                        min_kpts_frac=min_kpts_frac, cost=match_cost)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('predictions', type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--split', default='test')
    ap.add_argument('--pck', default=None, help='comma-separated thresholds; default per mode')
    ap.add_argument('--mota-dist', type=float, default=None,
                    help='match radius for MOTA; default is half the median animal extent '
                         '(the DIAGONAL of its keypoint box, not a per-axis span)')
    ap.add_argument('--vs', type=Path, default=None,
                    help='a second prediction npz, reported as `predictions` minus this one under '
                         'a PAIRED bootstrap over the groups both scored, on the points BOTH '
                         'matched. Without the shared point set an error delta is confounded by '
                         'coverage (eval rule 6): the arm that declined more points reads better.')
    ap.add_argument('--min-match-kpts', type=float, default=0.0,
                    help='the FRACTION of the keypoint set a predicted instance must share with a '
                         'labelled one before its mean distance may stand for the pair. At 0 (the '
                         'default, and what every published number here used) a row sharing ONE '
                         'keypoint is scored on that keypoint and can hijack a GT row -- which '
                         'inflates coverage and MOTA for whatever made the rows sparse, '
                         '`--vis-thresh` being exactly such a thing. A fraction and not a count: K '
                         'is 4 on rat-city and 47 on allen.')
    ap.add_argument('--chunk', type=int, default=0,
                    help='split each group into N-frame pieces, each its own scoring unit, so the '
                         'bootstrap has something to resample. THE BOOTSTRAP RESAMPLES GROUPS and a '
                         'long clip is ONE group -- rat-city\'s whole test split is a single '
                         '500-frame group, so every delta on it reads DEGENERATE. An interval from '
                         'chunks of one clip is WITHIN-CLIP uncertainty and is optimistic against '
                         'the between-clip kind; say which one a number is (eval rule 1).')
    ap.add_argument('--match-cost', choices=('mean', 'penalised'), default='mean',
                    help="how a candidate pair's per-keypoint distances become one number. "
                         "'mean' (the default, and what every published number here used) divides "
                         'by the SHARED count, so a row sharing ONE keypoint is scored on that '
                         'keypoint and can out-bid a dense row. "penalised" is OKS\'s answer in '
                         'distance units: every keypoint the LABEL has and the prediction declined '
                         'is charged at the match radius and stays in the denominator. It buys '
                         '--min-match-kpts\'s protection without its punitiveness at small K, '
                         'because it degrades continuously instead of rejecting the pair. An arm, '
                         'not a correction -- it moves every number in this repo.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    preds, meta = load_predictions(args.predictions)
    labels = label_lookup(args.data, args.split)
    if args.chunk:
        preds, labels = chunk_frames(preds, labels, args.chunk)
        print(f'--chunk {args.chunk}: {len(preds)} scoring unit(s) -- WITHIN-clip uncertainty, '
              'not between-clip')
    print(f'run={meta["run"]}  anchor={meta["anchor"]}  boxes={meta["boxes"]}'
          + (f'  box_source={meta["box_source"]}' if meta.get('box_source') else ''))
    if meta['anchor'] == 'labels':
        print('*** ORACLE: the model was seeded with ground truth. Not a deployment number. ***')

    rows = score(preds, labels, args.mota_dist, min_kpts_frac=args.min_match_kpts,
                 match_cost=args.match_cost)
    if not rows:
        print('nothing scored')
        return

    multi_any = any(m['S'] > 1 for m in rows)
    if multi_any:
        print('\nmulti-animal: err is over HUNGARIAN-MATCHED instances. Row index is not '
              'identity once boxes come from a detector.')
    print(f'\n{"group":48s} {"S p/t":>7s} {"err":>9s} {"med":>8s} {"cov":>7s} {"unmatched":>10s}')
    for m in rows:
        print(f'{m["group"][:48]:48s} {m["S_pred"]:3d}/{m["S_true"]:<3d} {m["err"]:9.3f} '
              f'{m["median"]:8.3f} {m["coverage"]:7.3f} {m.get("unmatched", 0):10d}')

    # ONE AGGREGATE PER MODE. A split may hold both 2D and 3D sessions, and averaging pixels
    # with millimetres produces a number in no unit at all.
    for mode in ('3d', '2d'):
        block = [m for m in rows if m['mode'] == mode]
        if not block:
            continue
        unit = 'mm' if mode == '3d' else 'px'
        boot = paired_bootstrap([m['err'] for m in block], seed=args.seed)
        n_true = sum(m['n_true'] for m in block)
        n_match = sum(m['n_matched'] for m in block)
        print(f'\n[{mode}] MPJPE {boot["mean"]:.3f} {unit}  '
              f'[{boot["lo"]:.3f}, {boot["hi"]:.3f}] 95% bootstrap over {boot["n"]} group(s)')
        # NO LABELS IN THE SCORED PREFIX IS A DIAGNOSIS, NOT A ZeroDivisionError. `n_true` counts
        # over `true[:, :T]` with `T = min(pred, true)`, so `infer.py --max-frames 24` on an
        # annotated root -- 65-frame groups whose one labelled frame is at index 32 -- gives every
        # group `n_true = 0`, and the script died on this line after printing a table of NaN.
        if not n_true:
            print(f'[{mode}] coverage n/a: no labelled points in the scored frames. The prediction '
                  'is shorter than the labels (--max-frames?) and the labelled frames are past '
                  'its end.')
            continue
        print(f'[{mode}] coverage {n_match / n_true:.4f}  ({n_match} of {n_true} labelled '
              f'points{", matched" if any(m["S"] > 1 for m in block) else ""})')
        mr = [m['motion_ratio'] for m in block if m.get('motion_ratio') is not None]
        if mr:
            print(f'[{mode}] motion_ratio {np.mean(mr):.3f}  (predicted path / label path over the '
                  f'steps both have, {len(mr)} group(s) -- UNPAIRED, a screen not a claim; '
                  '--vs pairs it)')
        # THE TAIL, because the mean above cannot show one. An arm that declines five sixths of the
        # animals has a flattering mean and a p90 that says so (eval rule 6), and it is the one
        # column comparable to the literature at all -- APT reports raw Euclidean percentiles and
        # no OKS, and SLEAP reports p50..p99 beside an mAP whose sigma matches nobody's.
        pcts = [k for k in _PCT_KEYS if any(np.isfinite(m.get(k, np.nan)) for m in block)]
        if pcts:
            cells = '  '.join(f'{k} {np.nanmean([m.get(k, np.nan) for m in block]):.3f}'
                              for k in pcts)
            print(f'[{mode}] err {cells} {unit}  (mean over {len(block)} group(s))')
        for name, label in (('box_agree', 'pose centroid to its own crop box'),
                            # UNBOUNDED IN 2D where `box_agree` is capped at ~half a box side by
                            # construction: the detector's keypoints are regressed in the full
                            # frame, the pose is decoded inside its crop.
                            ('kpt_agree', 'pose to its own detector keypoints')):
            vals = [m[name] for m in block if m.get(name) is not None]
            if vals:
                p99 = [m[f'{name}_p99'] for m in block if m.get(f'{name}_p99') is not None]
                print(f'[{mode}] {name} median {np.mean(vals):.3f} box-side(s), p99 '
                      f'{np.mean(p99):.3f}  ({label})')
        # THE HEAD'S TARGET, NOT ITS OUTPUT. A `base` at 1.000 means the converter wrote every
        # point VISIBLE and there is nothing for `--vis-thresh` to learn -- calms21 exactly, where
        # the gate is worth -0.037 to -0.123 MOTA. Print it wherever `vis_pred` exists so the next
        # root declares itself instead of being documented.
        vb = [m for m in block if m.get('vis_base') is not None]
        if vb:
            def _mean(k):
                v = [m[k] for m in vb if np.isfinite(m[k])]
                return np.mean(v) if v else float('nan')
            print(f'[{mode}] vis precision {_mean("vis_precision"):.4f}  recall '
                  f'{_mean("vis_recall"):.4f}  base {_mean("vis_base"):.4f} '
                  f'({sum(m["vis_n"] for m in vb)} assessed points, logit > 0). A base near 1.000 '
                  'means the target has no negatives -- do not gate on this head.')

        thresholds = ([float(t) for t in args.pck.split(',')] if args.pck
                      else ([2.0, 5.0, 10.0] if unit == 'mm' else [5.0, 10.0, 20.0]))
        # The SAME arrays the table was computed from. Rebuilding them from the npz re-derived
        # the slicing and got it wrong whenever pred and true disagreed on S or T, silently
        # comparing one animal's points against another's.
        allp = np.concatenate([m.get('_pred_matched', m['_pred']).reshape(-1, m['_pred'].shape[-1])
                               for m in block])
        allt = np.concatenate([m['_true'].reshape(-1, m['_true'].shape[-1]) for m in block])
        for k, v in pck(allp, allt, thresholds).items():
            print(f'[{mode}] {k} ({unit})  {v:.4f}')

    multi = [m for m in rows if m['S'] > 1]
    if multi:
        print()
        for m in multi:
            r, unit = m['mota'], 'mm' if m['mode'] == '3d' else 'px'
            # THE FP TERM SPLIT: `dup` landed on an animal something else already claimed, `none`
            # landed on no labelled animal at all. Arbitration can only remove the first kind and a
            # score threshold only the second, so an undifferentiated `fp` said nothing about which
            # to build. BOTH radii too, because they are different quantities and always were:
            # MPJPE matches at the animal's full keypoint-box diagonal and MOTA at half of it, so
            # the two tables can disagree about whether an instance was found at all.
            print(f'{m["group"][:40]:40s} MOTA {r["mota"]:.3f}  miss {r["miss_rate"]:.3f}  '
                  f'fp {r["fp_rate"]:.3f} (dup {r["fp_dup_rate"]:.3f} none '
                  f'{r["fp_none_rate"]:.3f})  idsw {r["idsw_rate"]:.4f}  '
                  f'fp_ignored {r["fp_ignored"]:d}  '
                  f'(mota r={m["mota_r"]:.1f}, mpjpe r={m["mpjpe_r"]:.1f} {unit})')

    if args.vs:
        other, ometa = load_predictions(args.vs)
        # THE SECOND FILE MUST BE CHUNKED THE SAME WAY, or the two sides key on different names and
        # the pairing silently finds nothing in common -- which prints `no shared groups` and is
        # easy to read as "these arms scored different clips". `labels` here is ALREADY the chunked
        # lookup, so this reuses it rather than rebuilding one that could differ.
        if args.chunk:
            other, _ = chunk_frames(other, label_lookup(args.data, args.split), args.chunk)
        print(f'\nPAIRED: {args.predictions} minus {args.vs}')
        print(f'  other: run={ometa["run"]}  anchor={ometa["anchor"]}  boxes={ometa["boxes"]}')
        by_key = {m['group']: m for m in score(other, labels, args.mota_dist, quiet=True,
                                              min_kpts_frac=args.min_match_kpts,
                                              match_cost=args.match_cost)}
        pairs = [(m, by_key[m['group']]) for m in rows if m['group'] in by_key]
        if not pairs:
            print('  no shared groups')
            return
        for mode in ('3d', '2d'):
            block = [(a, b) for a, b in pairs if a['mode'] == mode]
            if not block:
                continue
            unit = 'mm' if mode == '3d' else 'px'
            ea, eb, shared, nlab = [], [], 0, 0
            for a, b in block:
                da, db, n, nl = _shared_error(a, b)
                ea.append(da)
                eb.append(db)
                shared += n
                nlab += nl
            d = paired_bootstrap(ea, eb, seed=args.seed)
            ci = ('DEGENERATE (one group -- no interval exists)' if d['n'] < 2
                  else f'[{d["lo"]:+.4f}, {d["hi"]:+.4f}]'
                       + ('' if d['lo'] <= 0 <= d['hi'] else '  *'))
            # THE DROPPED COUNT, because pairing is complete-case and that flatters the arm that
            # failed more: a group where either side produced nothing leaves the comparison.
            drop = f"  ({d['n_dropped']} group(s) DROPPED, one side non-finite)" if d.get(
                'n_dropped') else ''
            print(f'[{mode}] MPJPE {d["mean"]:+.4f} {unit}  {ci}{drop}')
            # THE SHARED SET IS THE HEADLINE, not a footnote. A delta over points only one arm
            # attempted measures which arm declined more.
            # Same degenerate case as the coverage line above: no labelled point in the scored
            # prefix makes this a ZeroDivisionError rather than a statement about the two arms.
            frac = f'{shared / nlab:.4f}' if nlab else 'n/a, nothing labelled in the scored frames'
            print(f'[{mode}] over {shared} points BOTH matched, of {nlab} labelled '
                  f'({frac}) in {len(block)} group(s)')

            def paired(name, get):
                got = [(get(a), get(b)) for a, b in block]
                got = [(x, y) for x, y in got if x is not None and y is not None]
                if not got:
                    return
                dm = paired_bootstrap([x for x, _ in got], [y for _, y in got], seed=args.seed)
                tail = ('DEGENERATE (one group)' if dm['n'] < 2
                        else f'[{dm["lo"]:+.4f}, {dm["hi"]:+.4f}]'
                             + ('' if dm['lo'] <= 0 <= dm['hi'] else '  *'))
                print(f'[{mode}] {name:>13s} {dm["mean"]:+.4f}  {tail}')

            # COVERAGE IS PAIRED TOO. It is half of every claim in this file -- `err` is a mean over
            # matched points and means nothing without it -- and it was the one column a `--vs` run
            # could not put an interval on.
            paired('coverage', lambda m: m['coverage'])
            # MOTION IS PAIRED THE SAME WAY THE ERROR IS, and it has to be: a path length over an
            # arm's own matched set rewards declining points, exactly like `err`. Both arms are
            # measured against the SAME label path over the steps both attempted, so the delta is
            # attenuation and nothing else.
            mot = [(_shared_motion(a, b)) for a, b in block]
            mot = [(x, y) for x, y, n in mot if n and np.isfinite(x) and np.isfinite(y)]
            if mot:
                dm = paired_bootstrap([x for x, _ in mot], [y for _, y in mot], seed=args.seed)
                tail = ('DEGENERATE (one group)' if dm['n'] < 2
                        else f'[{dm["lo"]:+.4f}, {dm["hi"]:+.4f}]'
                             + ('' if dm['lo'] <= 0 <= dm['hi'] else '  *'))
                print(f'[{mode}] {"motion_ratio":>13s} {dm["mean"]:+.4f}  {tail}')
            paired('box_agree', lambda m: m.get('box_agree'))
            paired('kpt_agree', lambda m: m.get('kpt_agree'))
            for _k in _PCT_KEYS:
                paired(f'err {_k}', (lambda k: lambda m: m.get(k))(_k))
            # And the FP term SPLIT, not just its total: `dup` is what arbitration could remove and
            # `none` is what a detector threshold could, so a paired `fp_rate` alone cannot say
            # which of the two an arm moved.
            for name in ('mota', 'miss_rate', 'fp_rate', 'fp_dup_rate', 'fp_none_rate',
                         'idsw_rate'):
                paired(name, lambda m, k=name: m['mota'][k] if 'mota' in m else None)


def _shared_error(a, b):
    """(err_a, err_b, n_shared, n_labelled) over the points BOTH arms matched.

    Both sides' predictions are already aligned to the LABEL rows (`_pred_matched`), so the two are
    directly comparable point by point. Restricting to the intersection is what makes the delta a
    statement about accuracy rather than about coverage: an arm that predicts fewer, easier points
    has a better mean over its own set and a worse one here.
    """
    pa = a.get('_pred_matched', a['_pred'])
    pb = b.get('_pred_matched', b['_pred'])
    true = a['_true']
    S = min(pa.shape[0], pb.shape[0], true.shape[0])
    T = min(pa.shape[1], pb.shape[1], true.shape[1])
    pa, pb, true = pa[:S, :T], pb[:S, :T], true[:S, :T]
    ok = (np.isfinite(pa).all(-1) & np.isfinite(pb).all(-1) & np.isfinite(true).all(-1))
    n_lab = int(np.isfinite(true).all(-1).sum())
    if not ok.any():
        return float('nan'), float('nan'), 0, n_lab
    da = np.linalg.norm(pa[ok] - true[ok], axis=-1)
    db = np.linalg.norm(pb[ok] - true[ok], axis=-1)
    return float(da.mean()), float(db.mean()), int(ok.sum()), n_lab


def _shared_motion(a, b):
    """(motion_ratio_a, motion_ratio_b, n_steps) over the STEPS both arms and the label have.

    The same restriction `_shared_error` makes, for the same reason: a path length summed over
    whatever an arm happened to predict rewards the arm that predicted less. Both numerators and the
    single shared denominator come from one step mask, so the difference is how much each arm moved
    where they were both looking.
    """
    pa = a.get('_pred_matched', a['_pred'])
    pb = b.get('_pred_matched', b['_pred'])
    true = a['_true']
    S = min(pa.shape[0], pb.shape[0], true.shape[0])
    T = min(pa.shape[1], pb.shape[1], true.shape[1])
    pa, pb, true = pa[:S, :T], pb[:S, :T], true[:S, :T]
    ok = np.isfinite(pa).all(-1) & np.isfinite(pb).all(-1) & np.isfinite(true).all(-1)
    both = ok[:, :-1] & ok[:, 1:]
    if not both.any():
        return float('nan'), float('nan'), 0
    dt = np.linalg.norm(np.diff(true, axis=1), axis=-1)[both].sum()
    if not dt:
        return float('nan'), float('nan'), 0
    da = np.linalg.norm(np.diff(pa, axis=1), axis=-1)[both].sum()
    db = np.linalg.norm(np.diff(pb, axis=1), axis=-1)[both].sum()
    return float(da / dt), float(db / dt), int(both.sum())


if __name__ == '__main__':
    main()
