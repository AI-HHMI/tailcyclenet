"""Gate D: does the scorer's score track the pose model's REAL error? (plan section 7.4)

Scores a set of PREDICTION sessions, measures the pose model's per-keypoint disagreement with the
reference coordinates, and reports the Spearman correlation between the two -- overall, and WITHIN
each keypoint.

Within-keypoint because a keypoint with systematically larger error would otherwise carry the
correlation on its own: a scorer that merely learned "the tail tip is always worse" would score a
spurious positive overall.

**On a tracked root this measures disagreement with a REFERENCE TRACKER, not verified ground-truth
error.** 3dpop is `tracked`, so a scorer that correctly flags a genuine reference error is
PENALISED by this statistic. Read every number below as "reference disagreement".

Windows come from the QC table itself, so the error and the score are framed identically. Frames
are assigned to the LAST window containing them (CLAUDE.md eval rule 11) -- a per-window statistic
must use the seam rule's own frame->window assignment.

    pixi run python scripts/scorer_error_correlation.py --scores qc/scores.pq \\
        --pred-root scratch/gated/predroot --ref-root <3dpop root> --config configs/scorer-3dpop.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt  # noqa: E402
from tailcyclenet.checkpoints import load_config, _SCORER_CONFIG  # noqa: E402


def _frame_to_window(starts, n_frames, T):
    """Map every source frame to the LAST window containing it.

    Inputs: starts -- sorted window start frames; n_frames -- the group's length; T -- window
            length.
    Outputs: {frame: start}.
    Side effects: none.
    """
    assign = {}
    for s in sorted(starts):
        for f in range(int(s), min(int(s) + T, n_frames)):
            assign[f] = int(s)
    return assign


def _per_window_error(pred_sess, ref_sess, group_id, animal_idx, starts, T, keypoint):
    """Mean disagreement for one (group, animal, keypoint) over each window's assigned frames.

    Inputs: pred_sess / ref_sess -- loaded Sessions of the same underlying clip; group_id -- which
            group; animal_idx -- which animal; starts -- that animal's window starts; T -- window
            length; keypoint -- the keypoint NAME, resolved per session.
    Outputs: {start: mean distance} for windows with at least one frame observed in both.
    Side effects: reads parquet tables.
    """
    plab = pred_sess.groups[group_id].labels()
    rlab = ref_sess.groups[group_id].labels()
    if plab.points3d is None or rlab.points3d is None:
        return {}
    if animal_idx >= plab.n_animals or animal_idx >= rlab.n_animals:
        return {}
    n_frames = pred_sess.groups[group_id].n_frames
    pk = plab.animal_ids and list(pred_sess.names).index(keypoint)
    rk = list(ref_sess.names).index(keypoint)
    p = plab.points3d[animal_idx, :, pk, :]
    r = rlab.points3d[animal_idx, :, rk, :]
    n = min(p.shape[0], r.shape[0], n_frames)
    d = np.linalg.norm(p[:n] - r[:n], axis=-1)
    assign = _frame_to_window(starts, n, T)
    buckets: dict[int, list] = {}
    for f, s in assign.items():
        if np.isfinite(d[f]):
            buckets.setdefault(s, []).append(float(d[f]))
    return {s: float(np.mean(v)) for s, v in buckets.items() if v}


def main(argv=None) -> int:
    """Entry point for the Gate D correlation report.

    Inputs: argv -- argument list, defaulting to sys.argv.
    Outputs: process exit code.
    Side effects: prints the correlation table; reads prediction and reference parquet tables.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True, help="a QC run's scores.pq")
    ap.add_argument('--pred-root', required=True,
                    help='a root whose <split>/ holds the prediction session dirs')
    ap.add_argument('--ref-root', required=True, help='the source dataset root')
    ap.add_argument('--split', default='test')
    ap.add_argument('--config', required=True, help='the config the scorer was trained with')
    ap.add_argument('--min-windows', type=int, default=8,
                    help='skip a keypoint with fewer than this many scored windows')
    args = ap.parse_args(argv)

    config = load_config(args.config, base=_SCORER_CONFIG)
    T = int(config['data']['n_frames'])
    table = pd.read_parquet(args.scores)
    print(f'{len(table)} scored rows over {table["session"].nunique()} session(s), T={T}')

    rows = []
    for (session, group, animal), sub in table.groupby(['session', 'group', 'animal']):
        pred_dir = Path(args.pred_root) / args.split / session
        ref_dir = Path(args.ref_root) / args.split / session
        if not pred_dir.exists() or not ref_dir.exists():
            print(f'  skip {session}/{group}: '
                  f'{"no prediction" if not pred_dir.exists() else "no reference"}')
            continue
        pred_sess, ref_sess = fmt.Session.load(pred_dir), fmt.Session.load(ref_dir)
        if group not in pred_sess.groups or group not in ref_sess.groups:
            continue
        starts = sorted(sub['start'].unique())
        for kpt, ksub in sub.groupby('keypoint'):
            errs = _per_window_error(pred_sess, ref_sess, group, int(animal), starts, T, kpt)
            for r in ksub.itertuples():
                e = errs.get(int(r.start))
                if e is None:
                    continue
                rows.append({'session': session, 'group': group, 'animal': int(animal),
                             'start': int(r.start), 'keypoint': kpt, 'score': float(r.score),
                             'error': e})

    if not rows:
        print('no (window, keypoint) row had both a score and a reference disagreement')
        return 1
    df = pd.DataFrame(rows)
    print(f'\n{len(df)} (window, keypoint) rows carry both a score and a disagreement')
    print(f'disagreement: median {df["error"].median():.3f} mm, '
          f'p90 {df["error"].quantile(0.9):.3f} mm, max {df["error"].max():.3f} mm')

    rho, p = stats.spearmanr(df['score'], df['error'])
    print(f'\nOVERALL Spearman(score, disagreement) = {rho:+.4f}  (p={p:.3g}, n={len(df)})')
    print('  score is a QUALITY score, so a working scorer gives a NEGATIVE rho here:')
    print('  higher score <-> smaller disagreement.')

    print(f'\nWITHIN KEYPOINT (>= {args.min_windows} scored windows):')
    print(f'  {"keypoint":>16}  {"rho":>8}  {"n":>6}')
    per = []
    for kpt, sub in df.groupby('keypoint'):
        if len(sub) < args.min_windows:
            continue
        r, _pv = stats.spearmanr(sub['score'], sub['error'])
        per.append((kpt, r, len(sub)))
    for kpt, r, n in sorted(per, key=lambda x: x[1]):
        print(f'  {kpt:>16}  {r:>+8.4f}  {n:>6}')
    if per:
        rhos = np.array([x[1] for x in per])
        print(f'\n  {int((rhos < 0).sum())}/{len(per)} keypoints have the working sign '
              f'(negative); median rho {np.median(rhos):+.4f}')

    out = Path(args.scores).parent
    df.to_parquet(out / 'error_correlation_rows.pq')
    print(f'\nwrote {out / "error_correlation_rows.pq"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
