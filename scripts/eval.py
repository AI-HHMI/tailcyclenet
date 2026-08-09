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
  above a +-0.023 seed floor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.format import Session, load_dataset
from tailcyclenet.metrics import (error_and_coverage, matched_error, mota,
                                  paired_bootstrap, pck)


def load_predictions(path: Path):
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['__keys__']]
    out = {}
    for key in keys:
        out[key] = {f.split('|', 1)[1]: z[f] for f in z.files
                    if f.startswith(key + '|')}
    meta = {'run': str(z['__run__']), 'anchor': str(z['__anchor__']),
            'boxes': str(z['__boxes__'])}
    return out, meta


def label_lookup(data: Path, split: str):
    """{session/group: (Labels, Session)} for every group in the split."""
    data = Path(data)
    sessions = ([Session.load(data)] if (data / 'session.toml').exists()
                else load_dataset(data).sessions.get(split, []))
    out = {}
    for sess in sessions:
        sess.preload()
        for gid in sess.groups:
            out[f'{sess.session_id}/{gid}'] = (sess.labels(gid), sess)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('predictions', type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--split', default='test')
    ap.add_argument('--pck', default=None, help='comma-separated thresholds; default per mode')
    ap.add_argument('--mota-dist', type=float, default=None,
                    help='match radius for MOTA; default 0.1 x the median animal extent')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    preds, meta = load_predictions(args.predictions)
    labels = label_lookup(args.data, args.split)
    print(f'run={meta["run"]}  anchor={meta["anchor"]}  boxes={meta["boxes"]}')
    if meta['anchor'] == 'labels':
        print('*** ORACLE: the model was seeded with ground truth. Not a deployment number. ***')

    per_group, rows = [], []
    for key, out in sorted(preds.items()):
        if key not in labels:
            print(f'{key}: no labels, skipped')
            continue
        lab, sess = labels[key]
        mode = str(out['mode'])
        true = lab.points3d if mode == '3d' else lab.points2d[..., 0, :]
        pred = out['pred']
        S = min(pred.shape[0], true.shape[0])
        T = min(pred.shape[1], true.shape[1])
        pred, true = pred[:S, :T], true[:S, :T]

        m = error_and_coverage(pred, true)
        if S > 1:
            # Row index is not identity once boxes come from a detector. Match, then measure.
            with np.errstate(all='ignore'):
                span = np.nanmax(true, axis=2) - np.nanmin(true, axis=2)
                extent = float(np.nanmedian(np.linalg.norm(span, axis=-1)))
            mm = matched_error(pred, true, max_dist=extent if np.isfinite(extent) else np.inf)
            m['err_rowwise'] = m['err']
            m.update({k: v for k, v in mm.items() if k in ('err', 'median', 'coverage')})
            m['unmatched'] = mm.get('unmatched_true', 0)
        m['group'] = key
        m['mode'] = mode
        m['S'] = S
        rows.append(m)
        per_group.append(m['err'])

    if not rows:
        print('nothing scored')
        return

    unit = 'mm' if rows[0]['mode'] == '3d' else 'px'
    multi_any = any(m['S'] > 1 for m in rows)
    if multi_any:
        print('\nmulti-animal: err is over HUNGARIAN-MATCHED instances. Row index is not '
              'identity once boxes come from a detector.')
    print(f'\n{"group":48s} {"S":>3s} {"err":>9s} {"med":>8s} {"cov":>7s} {"unmatched":>10s}')
    for m in rows:
        print(f'{m["group"][:48]:48s} {m["S"]:3d} {m["err"]:9.3f} {m["median"]:8.3f} '
              f'{m["coverage"]:7.3f} {m.get("unmatched", 0):10d}')

    boot = paired_bootstrap(per_group, seed=args.seed)
    n_true = sum(m['n_true'] for m in rows)
    n_match = sum(m['n_matched'] for m in rows)
    print(f'\nMPJPE {boot["mean"]:.3f} {unit}  '
          f'[{boot["lo"]:.3f}, {boot["hi"]:.3f}] 95% paired bootstrap over {boot["n"]} groups')
    print(f'coverage {n_match / n_true:.4f}  ({n_match} of {n_true} labelled points)')

    thresholds = ([float(t) for t in args.pck.split(',')] if args.pck
                  else ([2.0, 5.0, 10.0] if unit == 'mm' else [5.0, 10.0, 20.0]))
    allp = np.concatenate([preds[m['group']]['pred'].reshape(-1, preds[m['group']]['pred']
                                                             .shape[-1]) for m in rows])
    allt = np.concatenate([
        (labels[m['group']][0].points3d if m['mode'] == '3d'
         else labels[m['group']][0].points2d[..., 0, :])[:preds[m['group']]['pred'].shape[0],
                                                         :preds[m['group']]['pred'].shape[1]]
        .reshape(-1, preds[m['group']]['pred'].shape[-1]) for m in rows])
    n = min(len(allp), len(allt))
    for k, v in pck(allp[:n], allt[:n], thresholds).items():
        print(f'{k} ({unit})  {v:.4f}')

    multi = [m for m in rows if m['S'] > 1]
    if multi:
        print()
        for m in multi:
            lab, _ = labels[m['group']]
            out = preds[m['group']]
            true = lab.points3d if m['mode'] == '3d' else lab.points2d[..., 0, :]
            S, T = m['S'], min(out['pred'].shape[1], true.shape[1])
            # Match radius scaled to the animal's own size: the DIAGONAL of its keypoint
            # bounding box, not a per-axis span. Taking the per-axis span and median-ing it
            # across axes gave 1.6 px on rat-city -- tighter than the labelling noise, so
            # every instance read as a miss AND a false positive and MOTA went negative.
            with np.errstate(all='ignore'):
                span = np.nanmax(true[:S, :T], axis=2) - np.nanmin(true[:S, :T], axis=2)
                extent = np.nanmedian(np.linalg.norm(span, axis=-1))
            radius = args.mota_dist or float(extent) * 0.5
            r = mota(out['pred'][:S, :T], true[:S, :T], radius,
                     ignore=None if lab.instance is None else
                     (lab.instance[:S, :T] == 1).any(-1))
            print(f'{m["group"][:40]:40s} MOTA {r["mota"]:.3f}  miss {r["miss_rate"]:.3f}  '
                  f'fp {r["fp_rate"]:.3f}  idsw {r["idsw_rate"]:.4f}  (r={radius:.1f}{unit})')


if __name__ == '__main__':
    main()
