#!/usr/bin/env python
"""Compare a sweep of scorer window offsets: does scoring off the prediction's lattice help?

A prediction is produced on its own window lattice (stride `T - overlap`). The scorer enumerates
its own windows over the stored track. When the two lattices share a phase, some scored windows
are IDENTICAL to a prediction window, and every prediction seam falls on a scored window's edge --
the position where a within-window statistic is least able to see it. Shifting the scorer's lattice
by an offset removes that coincidence, which is the hypothesis this compares.

`scripts/score_session.py --window-offset N` produces each arm; this reads the resulting
`error_correlation_rows.pq` files and reports the statistic the hypothesis is about -- the rank
correlation between the score and the pose model's real disagreement with the reference. The arms
are scored over the same sessions, so the comparison is of framings, not of data.

    pixi run python scripts/compare_scorer_offsets.py scratch/gated/sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _coincidence(offset, T, pred_stride):
    """Whether a lattice phase ever puts a window exactly on a prediction window.

    Inputs: offset -- the scorer lattice's phase; T -- window length; pred_stride -- the
            prediction's stride.
    Outputs: (always_coincides, fraction_of_windows_that_do) over a 24-window sample.
    Side effects: none.
    """
    hits = sum(1 for k in range(24) if (offset + k * T) % pred_stride == 0)
    return hits == 24, hits / 24.0


def main(argv=None) -> int:
    """Entry point for the offset comparison.

    Inputs: argv -- argument list, defaulting to sys.argv; the one positional argument is a
            directory holding `off<N>/error_correlation_rows.pq`.
    Outputs: process exit code, 1 when nothing was readable.
    Side effects: prints the comparison table.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('sweep', help='directory holding off<N>/ subdirectories')
    ap.add_argument('--T', type=int, default=12, help='scorer window length')
    ap.add_argument('--pred-stride', type=int, default=8,
                    help="the prediction's stride (T minus --overlap)")
    args = ap.parse_args(argv)

    root = Path(args.sweep)
    arms = {}
    for d in sorted(root.glob('off*')):
        f = d / 'error_correlation_rows.pq'
        if f.exists():
            arms[int(d.name[3:])] = pd.read_parquet(f)
    if not arms:
        print(f'no off<N>/error_correlation_rows.pq under {root}')
        return 1

    print(f'{"offset":>7}  {"rows":>6}  {"overall rho":>12}  {"median kpt rho":>15}  '
          f'{"kpts neg":>9}  {"same-lattice?":>14}')
    rows = []
    for off, df in sorted(arms.items()):
        overall = stats.spearmanr(df['score'], df['error'])[0]
        per = [stats.spearmanr(s['score'], s['error'])[0]
               for _k, s in df.groupby('keypoint') if len(s) >= 30]
        med = float(np.median(per)) if per else float('nan')
        neg = sum(1 for r in per if r < 0)
        _, frac = _coincidence(off, args.T, args.pred_stride)
        rows.append((off, len(df), overall, med, neg, len(per), frac))
        print(f'{off:>7}  {len(df):>6}  {overall:>+12.4f}  {med:>+15.4f}  '
              f'{neg:>4}/{len(per):<4}  {frac:>13.0%}')

    best = max(rows, key=lambda r: -r[2])
    base = next((r for r in rows if r[0] == 0), None)
    print()
    if base is not None and best[0] != 0:
        print(f'best offset {best[0]}: overall rho {best[2]:+.4f} against offset 0\'s '
              f'{base[2]:+.4f} ({best[2] - base[2]:+.4f})')
    elif base is not None:
        print(f'offset 0 is the best of {len(rows)} arms; shifting the lattice did not help')
    nulls = [r for r in rows if r[6] == 0.0]
    if nulls:
        print('off-lattice arms (no scored window is ever a prediction window): '
              + ', '.join(f'{r[0]} rho {r[2]:+.4f}' for r in nulls))
    print('\nA NEGATIVE rho is the working sign: a better score means a smaller disagreement.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
