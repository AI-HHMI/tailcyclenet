"""Score a handful of windows under several window offsets, fast, in one process.

The full QC pass is the wrong shape for a question about FRAMING. A sweep by
`scripts/score_session.py` pays the model load once per offset (5.6 s, a 3.2 GB checkpoint) and
re-decodes every window, and it scores thousands of windows to answer a question that a hundred
can answer better. This loads the scorer ONCE and walks the requested offsets over the same small
set of windows.

The windows come from a spans CSV -- `session,group,animal,span_start,span_len` -- which is how a
caller aims the probe at a clip's bad stretch. Windows on the dataset's OWN lattice are selected
per offset, so each arm is a genuine re-framing of the same frames rather than a subsetting of one
arm's windows.

The statistic is the rank correlation between the score and the track's distance to the reference,
which is Gate D's measure. **Density of bad windows is the point**: on easy clips the score has
little to rank, so a framing difference is invisible there. Read the delta between offsets, not the
level -- a span chosen for its large disagreement restricts the error range by construction, so the
level is a selected statistic and the comparison is not.

    pixi run python scripts/scorer_offset_probe.py --run <scorers>/scorer-3dpop \\
        --spans-csv spans.csv --pred-root scratch/gated/hybrid-hard --ref-root <3dpop root> \\
        --offsets 0 6 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt  # noqa: E402
from tailcyclenet.checkpoints import (load_config, load_scorer_run, scorer_output_granularity,
                                      _SCORER_CONFIG)  # noqa: E402
from tailcyclenet.scorer.qc import _loader_config, _to_device  # noqa: E402


def _load_spans(path):
    """Read the spans CSV into {(session, group, animal): (lo, hi)}.

    Inputs: path -- a CSV with session, group, animal, span_start and span_len columns.
    Outputs: the dict, both bounds inclusive.
    Side effects: reads a file; exits with a message when a column is missing.
    """
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            lo = int(r['span_start'])
            out[(r['session'], r['group'], str(r['animal']))] = (lo, lo + int(r['span_len']))
    if not out:
        raise SystemExit(f'{path}: no spans')
    return out


def _targets(ds, spans):
    """Dataset indices whose window starts inside a requested span.

    Inputs: ds -- a `PoseDataset` built with `train=False`; spans -- `_load_spans`' dict.
    Outputs: a list of (index, session, group, animal, start).
    Side effects: none.
    """
    out = []
    for i, it in enumerate(ds.index):
        rng = spans.get((it.session.path.name, it.gid, str(it.session.groups[it.gid]
                                                           .labels().animal_ids[it.animal])))
        if rng is not None and rng[0] <= it.start <= rng[1]:
            out.append((i, it.session.path.name, it.gid,
                        str(it.session.groups[it.gid].labels().animal_ids[it.animal]), it.start))
    return out


def _disagreement(pred_root, ref_root, split, session, group, animal_id, T):
    """[n_frames, K] distance from the prediction to the reference for one track.

    Inputs: pred_root / ref_root -- roots holding <split>/<session>; session -- the session name;
            group -- the group id; animal_id -- the animal's ID string; T -- unused, kept for
            symmetry with the caller.
    Outputs: (array of per-frame distance with NaN where either side is missing, n_frames).
    Side effects: reads parquet tables. Callers must cache the result -- it re-reads the session
        tables, which is far more expensive than scoring one window.
    """
    p = fmt.Session.load(Path(pred_root) / split / session)
    r = fmt.Session.load(Path(ref_root) / split / session)
    pl, rl = p.groups[group].labels(), r.groups[group].labels()
    pa = [str(x) for x in pl.animal_ids].index(str(animal_id))
    ra = [str(x) for x in rl.animal_ids].index(str(animal_id))
    n = min(pl.points3d.shape[1], rl.points3d.shape[1], p.groups[group].n_frames)
    pnames = [str(name) for name in p.names]
    rnames = [str(name) for name in r.names]
    d = np.full((n, len(pnames)), np.nan, dtype=np.float64)
    for ki, name in enumerate(pnames):
        if name not in rnames:
            continue
        ri = rnames.index(name)
        d[:, ki] = np.linalg.norm(
            pl.points3d[pa, :n, ki] - rl.points3d[ra, :n, ri], axis=-1)
    return d, n


def _dedupe_frame_rows(rows):
    """Keep the last contextual window for each source-frame/keypoint row."""
    latest = {}
    for row in rows:
        key = (row['session'], row['animal'], row['frame'], row['keypoint'])
        old = latest.get(key)
        if old is None or row['start'] >= old['start']:
            latest[key] = row
    return list(latest.values())


def main(argv=None) -> int:
    """Entry point for the fast window-offset probe.

    Inputs: argv -- argument list, defaulting to sys.argv.
    Outputs: process exit code.
    Side effects: prints the comparison; decodes video for the selected windows only.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a scorer run folder')
    ap.add_argument('--config', required=True, help='the config the run was trained with')
    ap.add_argument('--spans-csv', required=True)
    ap.add_argument('--pred-root', required=True)
    ap.add_argument('--ref-root', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--offsets', type=int, nargs='+', default=[0, 6])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save-rows', default=None,
                    help='write every scored row to this parquet, so the arm delta can be '
                         'bootstrapped over tracks')
    args = ap.parse_args(argv)

    from tailcyclenet.dataset import PoseDataset

    spans = _load_spans(args.spans_csv)
    print(f'{len(spans)} span(s) requested')

    config = load_config(args.config, base=_SCORER_CONFIG)
    T = int(config['data']['n_frames'])
    model, rcfg, registry, ckpt = load_scorer_run(Path(args.run), device=args.device)
    model.eval()
    output_granularity = scorer_output_granularity(rcfg)
    lc = _loader_config(rcfg)
    print(f'checkpoint {ckpt.name}, T={T}, K={registry.n_keypoints}')

    dists = {}
    for (session, group, animal) in spans:
        dists[(session, group, animal)] = _disagreement(
            args.pred_root, args.ref_root, args.split, session, group, animal, T)

    arms = {}
    for off in args.offsets:
        t0 = time.time()
        ds = PoseDataset(args.pred_root, args.split, replace(lc, val_offset=off), train=False)
        tgt = _targets(ds, spans)
        rows = []
        with torch.no_grad():
            for i, session, group, animal, start in tgt:
                item = ds[i]
                if item is None:
                    continue
                views, coords, _v, frames, cgroup, _row, _qt, _v2, _p2d, _occ, kpt_ids, _pr, _pt = \
                    item[:13]
                views, coords, cgroup, kpt_ids = _to_device(
                    views, coords, cgroup, kpt_ids, args.device)
                scores, _prec = model(views, coords, cgroup, kpt_ids[None])
                scores = scores[0].cpu().numpy()
                dist, nf = dists[(session, group, animal)]
                names = list(ds.index[i].session.names)
                if output_granularity == 'frame':
                    source_frames = (frames.detach().cpu().numpy()
                                     if torch.is_tensor(frames) else np.asarray(frames))
                    source_frames = np.asarray(source_frames).reshape(-1)
                    if scores.ndim != 2 or len(source_frames) != scores.shape[0]:
                        raise RuntimeError(
                            f'framewise scorer returned {scores.shape!r} for '
                            f'{len(source_frames)} source frames')
                    for local_t, source_frame in enumerate(source_frames):
                        source_frame = int(source_frame)
                        if not (0 <= source_frame < nf):
                            continue
                        for ki in range(scores.shape[1]):
                            if (ki >= dist.shape[1] or not np.isfinite(scores[local_t, ki])
                                    or not np.isfinite(dist[source_frame, ki])):
                                continue
                            rows.append({'offset': off, 'session': session, 'animal': animal,
                                         'start': start, 'frame': source_frame,
                                         'keypoint': names[ki],
                                         'score': float(scores[local_t, ki]),
                                         'error': float(dist[source_frame, ki])})
                else:
                    if scores.ndim != 1:
                        raise RuntimeError(
                            f'sequence scorer returned {scores.shape!r}; expected [K]')
                    lo, hi = start, min(start + T, nf)
                    seg = dist[lo:hi]
                    if not np.isfinite(seg).any():
                        continue
                    for ki in range(scores.shape[0]):
                        rows.append({'offset': off, 'session': session, 'animal': animal,
                                     'start': start, 'keypoint': names[ki],
                                     'score': float(scores[ki]),
                                     'error': float(np.nanmean(seg))})
        if output_granularity == 'frame':
            rows = _dedupe_frame_rows(rows)
        arms[off] = rows
        print(f'  offset {off}: scored {len(tgt)} windows -> {len(rows)} rows '
              f'in {time.time() - t0:.1f}s')

    print(f'\n{"offset":>7}  {"rows":>6}  {"overall rho":>12}  {"mean err":>9}')
    per = {}
    for off, rows in arms.items():
        if len(rows) < 12:
            print(f'{off:>7}  {len(rows):>6}  too few rows to correlate')
            continue
        d = rows
        rho = stats.spearmanr([r['score'] for r in d], [r['error'] for r in d])[0]
        per[off] = rho
        print(f'{off:>7}  {len(d):>6}  {rho:>+12.4f}  '
              f'{np.mean([r["error"] for r in d]):>9.2f}')
    if args.save_rows:
        row_schema = {
            'offset': pl.Int64, 'session': pl.String, 'animal': pl.String,
            'start': pl.Int64, 'keypoint': pl.String, 'score': pl.Float64,
            'error': pl.Float64,
        }
        if output_granularity == 'frame':
            row_schema['frame'] = pl.Int64
        pl.DataFrame([r for rows in arms.values() for r in rows], schema=row_schema).write_parquet(
            args.save_rows, compression='snappy'
        )
        print(f'wrote {args.save_rows}')

    if len(per) > 1:
        best = min(per.items(), key=lambda kv: kv[1])
        base = per.get(0)
        if base is not None and best[0] != 0:
            print(f'\nbest arm: offset {best[0]} at {best[1]:+.4f}, against offset 0 at '
                  f'{base:+.4f} (delta {best[1] - base:+.4f}).')
        else:
            print(f'\nbest arm: offset {best[0]} at {best[1]:+.4f} -- offset 0 is at least '
                  'as good as every shift tried here.')
        print('Negative rho is the working sign: a higher score means a smaller disagreement.')
    print('\nRead the DELTA between arms, not the level: spans picked for large disagreement '
          'restrict the error range by construction.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
