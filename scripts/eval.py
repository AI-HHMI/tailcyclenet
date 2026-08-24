#!/usr/bin/env python
"""Score a prediction file against the labels. Offline and model-free.

    pixi run python scripts/eval.py pred.npz --data <dataset> --split test

`err` is a mean over MATCHED points -- read it beside `cov`; the interval is a PAIRED bootstrap
over groups; MOTA's FP splits into `dup`/`none`; `--vs` pairs over points BOTH files matched.
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

    The bootstrap resamples groups, and a long clip is ONE group -- so chunks give a long clip a
    usable n. Chunks of one clip are more alike than independent clips, so the interval is
    WITHIN-clip uncertainty and optimistic against the between-clip kind.
    """
    import dataclasses

    out_p, out_l = {}, {}
    for key, out in preds.items():
        if key not in labels:
            out_p[key] = out
            continue
        lab, sess = labels[key]
        T = int(np.asarray(out['pred']).shape[1])
        # The match radius comes from the whole group, not the chunk: chunks scored under
        # different radii are not exchangeable, which a bootstrap needs them to be.
        full = lab.points3d if str(out['mode']) == '3d' else lab.points2d[..., 0, :]
        with np.errstate(all='ignore'):
            span = np.nanmax(full, axis=2) - np.nanmin(full, axis=2)
            extent = float(np.nanmedian(np.linalg.norm(span, axis=-1)))
        # The labels' time axis is not the prediction's: a `--max-frames` prefix must slice labels
        # by the LABELS' own length, or every chunk past the first re-scores frames 0..n-1.
        # Frame indices are absolute within the group for both sides.
        lab_T = int(np.asarray(full).shape[1])
        for t0 in range(0, T, n):
            t1 = min(t0 + n, T)
            # T = 1 is not a usable window.
            if t1 - t0 < 2:
                continue
            sub = {}
            for k, v in out.items():
                a = np.asarray(v)
                # Axis 1 is time only for frame-indexed arrays; `outcome`/`crop` are
                # window-indexed and stay whole.
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
    """One row per group: error, the coverage behind it, MOTA where there are instances.

    Factored out so `--vs` scores the second file through the identical path.
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
        # Surplus predicted rows are not discardable: truncating `pred` to `true`'s row count
        # deleted them as coverage, not as false positives. The matchers all take Sp != St; only
        # `error_and_coverage` needs equal shapes, and row-indexed error is meaningless under
        # detector boxes.
        T = min(pred.shape[1], true.shape[1])
        pred, true = pred[:, :T], true[:, :T]
        Sp, St = pred.shape[0], true.shape[0]
        S = max(Sp, St)

        m = error_and_coverage(pred[:min(Sp, St)], true[:min(Sp, St)])
        if S > 1:
            # Row index is not identity under detector boxes: match, then measure, taking the
            # matched counts so coverage describes the same points as the error.
            if '__extent__' in out:
                # The WHOLE group's, carried by --chunk.
                extent = float(out['__extent__'])
            else:
                with np.errstate(all='ignore'):
                    span = np.nanmax(true, axis=2) - np.nanmin(true, axis=2)
                    extent = float(np.nanmedian(np.linalg.norm(span, axis=-1)))
            # Zero is a finite, valid radius: one finite labelled keypoint gives a 0 diagonal, so
            # a sparse root can read as a catastrophic failure that is an artefact of the radius.
            max_dist = extent if np.isfinite(extent) and extent > 0 else np.inf
            mm = matched_error(pred, true, max_dist=max_dist,
                               min_kpts_frac=min_kpts_frac, cost=match_cost)
            m.update({k: v for k, v in mm.items()
                      if k in ('err', 'median', 'coverage', 'n_true', 'n_matched')
                      or k in _PCT_KEYS})
            m['unmatched'] = mm.get('unmatched_true', 0)
            # PCK gets the same pairing: it reads positionally, so detector boxes need the
            # aligned rows. NOT shared with MOTA, which reads the raw rows -- aligning them
            # first would zero `idsw` by construction. Shaped like true (a label row index), not
            # pred.
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
        # Already sliced to (S, T); PCK reuses these.
        m['_pred'], m['_true'] = pred, true
        # How much the prediction moved vs the animal -- a screen: a carried prompt low-passes the
        # prediction, which no error/coverage/MOTA column can see. `--vs` pairs it.
        mr = motion_ratio(m.get('_pred_matched', pred), true)
        m['motion_ratio'] = mr['ratio'] if mr['n_steps'] else None
        # Where the pose landed relative to its own crop box, in units of one box side: a pose
        # off its crop is not a prediction of that animal.
        if 'box_agree' in out:
            ba = np.asarray(out['box_agree'], float)
            ba = ba[np.isfinite(ba)]
            m['box_agree'] = float(np.median(ba)) if ba.size else None
            m['box_agree_p99'] = float(np.quantile(ba, 0.99)) if ba.size else None
        # `kpt_agree` is the 2D half of the same check: `box_agree` is structurally bounded in 2D
        # (the pose is decoded inside its own crop), while the detector's keypoints are regressed
        # in the full frame, so `kpt_agree` has no ceiling and is the 2D diagnostic.
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

    A `base` near 1.000 means the target has no negatives -- the converter wrote everything
    visible -- so the head cannot be gated on. A guard, not an accuracy figure.
    """
    # `conf` is the per-keypoint `vis_pred` logit; the label side is the status channel (`vis3d`,
    # or the first camera's `vis2d`, the same slice `true` came from).
    st = lab.vis3d if mode == '3d' else (None if lab.vis2d is None else lab.vis2d[..., 0])
    if 'conf' not in out or st is None:
        return {}
    vp, st = np.asarray(out['conf'], float), np.asarray(st, int)
    n, k = min(vp.shape[0], st.shape[0]), min(vp.shape[2], st.shape[2])
    vp, st = vp[:n, :T, :k], st[:n, :T, :k]
    if vp.shape != st.shape:
        return {}
    # UNLABELED is not a negative: counting it as "not visible" manufactures negatives.
    ok = np.isfinite(vp) & (st != UNLABELED)
    if not ok.any():
        return {}
    yhat, y = vp[ok] > 0.0, st[ok] == VISIBLE
    tp, fp, fn = int((yhat & y).sum()), int((yhat & ~y).sum()), int((~yhat & y).sum())
    return {'vis_precision': tp / (tp + fp) if tp + fp else float('nan'),
            'vis_recall': tp / (tp + fn) if tp + fn else float('nan'),
            'vis_base': float(y.mean()), 'vis_n': int(y.size)}


def _mota_for(m, lab, mota_dist, min_kpts_frac=0.0, extent_override=None, match_cost='mean'):
    """(radius, mota dict) for one group; the ignore region is built here, from the labels.

    `extent_override` (from `--chunk`) is the whole group's extent, so every chunk shares a radius.
    """
    pred, true = m['_pred'], m['_true']
    St, T = true.shape[0], true.shape[1]
    # Match radius = the diagonal of the keypoint bounding box, not a per-axis span: a per-axis
    # median ran tighter than the labelling noise and made every instance a miss AND an FP.
    if extent_override is not None:
        extent = float(extent_override)
    else:
        with np.errstate(all='ignore'):
            span = np.nanmax(true, axis=2) - np.nanmin(true, axis=2)
            extent = np.nanmedian(np.linalg.norm(span, axis=-1))
    # `is None`, not `or`: `--mota-dist 0` is falsy but a legitimate value.
    radius = float(mota_dist) if mota_dist is not None else float(extent) * 0.5
    if not radius > 0:
        radius = np.inf
        print(f'  MOTA: the labelled extent is {extent}, so the match radius is degenerate -- '
              'scoring without one. Too few labelled keypoints per instance-frame to size it.')
    # Present-but-unannotated animals: an unmatched prediction is excused only if it lands on one,
    # where the format carries boxes; without boxes the fallback excuses them all (`fp_ignored`).
    ig = ig_boxes = None
    if lab.instance is not None:
        ig = (lab.instance[:St, :T] == INST_PRESENT).any(-1)
        # Boxes are pixels, but in 3D the centroid is world millimetres -- every test fails and
        # the region degrades to the box-free fallback, so presence alone is used (`fp_ignored`).
        if lab.boxes is not None and m['mode'] == '2d':
            # xyxy in the first camera.
            ig_boxes = lab.boxes[:St, :T, 0]
    return radius, mota(pred, true, radius, ignore=ig, ignore_boxes=ig_boxes,
                        min_kpts_frac=min_kpts_frac, cost=match_cost)


def main():
    """Score a prediction file; exit via SystemExit on bad config.

    Inputs: argv (via argparse): predictions, --data, --split, --pck,
            --mota-dist, --vs, --min-match-kpts, --chunk, --match-cost,
            --seed.
    Side effects: prints per-group rows and per-mode aggregates.
    """
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
                    help='a second prediction, reported as `predictions` minus this one under a '
                         'PAIRED bootstrap over the groups both scored, on the points BOTH matched.')
    ap.add_argument('--min-match-kpts', type=float, default=0.0,
                    help='the FRACTION of the keypoint set a predicted instance must share with a '
                         'labelled one before its mean distance may stand for the pair. A fraction '
                         'and not a count (K is 4 to 47); use 0.5 for deltas.')
    ap.add_argument('--chunk', type=int, default=0,
                    help='split each group into N-frame scoring units (the bootstrap resamples '
                         'groups, so a long clip is one group and reads DEGENERATE). Chunks of one '
                         'clip are within-clip uncertainty; say which a number is.')
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

    # One aggregate per mode: a split may hold 2D and 3D sessions, and their units do not mix.
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
        # No labels in the scored prefix is a diagnosis, not a ZeroDivisionError: a short
        # prediction over an annotated root gives every group `n_true = 0`.
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
        # The tail, because the mean cannot show one: an arm that declines hard animals has a
        # flattering mean and a p90 that says so -- and percentiles are the literature's units.
        pcts = [k for k in _PCT_KEYS if any(np.isfinite(m.get(k, np.nan)) for m in block)]
        if pcts:
            cells = '  '.join(f'{k} {np.nanmean([m.get(k, np.nan) for m in block]):.3f}'
                              for k in pcts)
            print(f'[{mode}] err {cells} {unit}  (mean over {len(block)} group(s))')
        for name, label in (('box_agree', 'pose centroid to its own crop box'),
                            # Unbounded in 2D where `box_agree` is capped by construction.
                            ('kpt_agree', 'pose to its own detector keypoints')):
            vals = [m[name] for m in block if m.get(name) is not None]
            if vals:
                p99 = [m[f'{name}_p99'] for m in block if m.get(f'{name}_p99') is not None]
                print(f'[{mode}] {name} median {np.mean(vals):.3f} box-side(s), p99 '
                      f'{np.mean(p99):.3f}  ({label})')
        # The head's target, not its output: a `base` at 1.000 means there is nothing for
        # `--vis-thresh` to learn. Print it wherever `vis_pred` exists.
        vb = [m for m in block if m.get('vis_base') is not None]
        if vb:
            def _mean(k):
                """Finite mean of metric `k` across the groups that have it (NaN when none)."""
                v = [m[k] for m in vb if np.isfinite(m[k])]
                return np.mean(v) if v else float('nan')
            print(f'[{mode}] vis precision {_mean("vis_precision"):.4f}  recall '
                  f'{_mean("vis_recall"):.4f}  base {_mean("vis_base"):.4f} '
                  f'({sum(m["vis_n"] for m in vb)} assessed points, logit > 0). A base near 1.000 '
                  'means the target has no negatives -- do not gate on this head.')

        thresholds = ([float(t) for t in args.pck.split(',')] if args.pck
                      else ([2.0, 5.0, 10.0] if unit == 'mm' else [5.0, 10.0, 20.0]))
        # The same arrays the table was computed from: re-deriving the slicing from the npz got
        # it wrong whenever pred and true disagreed on S or T.
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
            # The FP term split: `dup` landed on an already-claimed animal (arbitration removes
            # it), `none` on no labelled animal (a threshold removes it). Both radii too -- MPJPE
            # matches at the full box diagonal, MOTA at half of it.
            print(f'{m["group"][:40]:40s} MOTA {r["mota"]:.3f}  miss {r["miss_rate"]:.3f}  '
                  f'fp {r["fp_rate"]:.3f} (dup {r["fp_dup_rate"]:.3f} none '
                  f'{r["fp_none_rate"]:.3f})  idsw {r["idsw_rate"]:.4f}  '
                  f'fp_ignored {r["fp_ignored"]:d}  '
                  f'(mota r={m["mota_r"]:.1f}, mpjpe r={m["mpjpe_r"]:.1f} {unit})')

    if args.vs:
        other, ometa = load_predictions(args.vs)
        # The second file must be chunked the same way or the pairing finds nothing in common;
        # `labels` is already the chunked lookup, so reuse it.
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
            # Complete-case pairing flatters the arm that failed more; print the drop count.
            drop = f"  ({d['n_dropped']} group(s) DROPPED, one side non-finite)" if d.get(
                'n_dropped') else ''
            print(f'[{mode}] MPJPE {d["mean"]:+.4f} {unit}  {ci}{drop}')
            # The shared set is the headline: a delta over points only one arm attempted measures
            # which arm declined more. Degenerate when nothing is labelled in the scored frames.
            frac = f'{shared / nlab:.4f}' if nlab else 'n/a, nothing labelled in the scored frames'
            print(f'[{mode}] over {shared} points BOTH matched, of {nlab} labelled '
                  f'({frac}) in {len(block)} group(s)')

            def paired(name, get):
                """Print the paired bootstrap interval of `get(m)` between the two files.

                Inputs: name -- the label for the row.
                        get -- m -> metric value (None when the arm lacks it).
                Side effects: prints one interval line when both arms have the metric.
                """
                got = [(get(a), get(b)) for a, b in block]
                got = [(x, y) for x, y in got if x is not None and y is not None]
                if not got:
                    return
                dm = paired_bootstrap([x for x, _ in got], [y for _, y in got], seed=args.seed)
                tail = ('DEGENERATE (one group)' if dm['n'] < 2
                        else f'[{dm["lo"]:+.4f}, {dm["hi"]:+.4f}]'
                             + ('' if dm['lo'] <= 0 <= dm['hi'] else '  *'))
                print(f'[{mode}] {name:>13s} {dm["mean"]:+.4f}  {tail}')

            # Coverage is paired too: `err` means nothing without it, and it was the one column
            # `--vs` could not put an interval on.
            paired('coverage', lambda m: m['coverage'])
            # Motion is paired like the error: a path length over an arm's own matched set rewards
            # declining points, so both are measured against the same label path.
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
            # The FP split too: a paired `fp_rate` alone cannot say which term an arm moved.
            for name in ('mota', 'miss_rate', 'fp_rate', 'fp_dup_rate', 'fp_none_rate',
                         'idsw_rate'):
                paired(name, lambda m, k=name: m['mota'][k] if 'mota' in m else None)


def _shared_error(a, b):
    """(err_a, err_b, n_shared, n_labelled) over the points BOTH arms matched.

    Restricting to the intersection makes the delta about accuracy, not coverage.
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

    Same restriction as `_shared_error`: a path length over whatever an arm predicted rewards
    predicting less.
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
