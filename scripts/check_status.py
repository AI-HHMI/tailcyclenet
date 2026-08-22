#!/usr/bin/env python
"""Audit the `status` policy across every tailcycle-dataset root.

    pixi run python scripts/check_status.py [root ...] [--report]

Policy: `missing` claims a real assessment (only on roots in `MISSING_OK`); holes are checked
only on `annotated` sessions against exact `HOLE_EXEMPTIONS` counts; `projected` never mixes with
`visible`. Keyed on the session's declared `labels`, not the folder name. Symlink-farm roots are
skipped by structure, not by name.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tailcyclenet import format as fmt

ROOT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')

# Roots that may carry a `missing` row; every other root's tables must be `missing`-free.
MISSING_OK = {
    # Written from the npz's own per-camera `vis`, only where the 3D point is finite.
    'allen-mouse-tracked': ('keypoints',),
    # points3d `missing` only where 0 views were visible and every camera assessed the point.
    'allen-mouse-annotated': ('keypoints', 'points3d'),
    # APT's Inf sentinel (fully occluded) stays `missing`; its NaN skip stays absent.
    'rat-city-annotated': ('keypoints',),
}

# root -> {table: exact hole count}, summed over every session in the root, checked only for
# `annotated` roots. A moved count is a silent re-conversion, not noise.
HOLE_EXEMPTIONS = {
    # APT's `NaN` (annotator skip) stays absent -- `missing` would claim an assessment nobody
    # made; the `Inf` sentinel is already counted in MISSING_OK.
    'rat-city-annotated': {'keypoints': 220},
    # 1-view and over-gate triangulations are left as no row; the 0-view-all-assessed case is the
    # 178 `missing` rows counted in MISSING_OK, not a hole.
    'allen-mouse-annotated': {'points3d': 211},
    # Outlier 2D is dropped as no row -- a rejected measurement, not an occlusion.
    'johnson-mouse-annotated': {'keypoints': 19},
    # The aug root re-runs the same gate over its own variant set, independently -- 18 vs 19 is
    # not a discrepancy.
    'johnson-mouse-annotated-aug': {'keypoints': 18},
}


def _status_counts(table) -> Counter:
    arr = table.column('status').combine_chunks()
    codes = arr.indices.to_numpy(zero_copy_only=False)
    vocab = arr.dictionary.to_pylist()
    c = Counter()
    for v, n in zip(*np.unique(codes, return_counts=True)):
        c[vocab[int(v)]] += int(n)
    return c


def _holes(table, key_cols: list[str]) -> int:
    """Rows with NO entry inside a (key_cols) group that has >= 1 entry, per bodypart.

    `key_cols` is `(group_id, frame, animal_id[, camera])`; the missing axis is `bodypart`.
    """
    import pandas as pd  # noqa: F401 -- pulled in by pyarrow's to_pandas()

    df = table.select([*key_cols, 'bodypart']).to_pandas()
    n_kpts = df['bodypart'].nunique()
    live_keys = df.groupby(key_cols, observed=True).ngroups   # instances with >= 1 row
    return int(live_keys * n_kpts - len(df))


def _is_symlink_farm(root_dir: Path) -> bool:
    """True if the root itself or every session directory under it is a symlink."""
    if root_dir.is_symlink():
        return True
    for split in ('train', 'val', 'test'):
        d = root_dir / split
        if not d.is_dir():
            continue
        sessions = [p for p in d.iterdir() if p.is_dir() or p.is_symlink()]
        if sessions:
            return all(p.is_symlink() for p in sessions)
    return False


def audit_session(sess: fmt.Session, root_name: str, hole_totals: Counter,
                  report: list[str]) -> list[str]:
    problems = []
    is_annotated = sess.label_source == 'annotated'
    for stem, key_cols in (('keypoints', ['group_id', 'frame', 'animal_id', 'camera']),
                           ('points3d', ['group_id', 'frame', 'animal_id'])):
        table = sess._tables[stem]
        if table is None:
            continue
        counts = _status_counts(table)
        if counts['missing'] and stem not in MISSING_OK.get(root_name, ()):
            problems.append(f'{root_name}/{sess.session_id}: {stem}.pq carries '
                            f'{counts["missing"]} `missing` row(s) but {root_name!r} is not in '
                            f'MISSING_OK for {stem!r}')
        if counts['projected'] and counts['visible']:
            problems.append(f'{root_name}/{sess.session_id}: {stem}.pq mixes `projected` '
                            f'({counts["projected"]}) with `visible` ({counts["visible"]})')

        line = (f'   {sess.session_id:<40} {stem:<10} '
               + ', '.join(f'{k}={v}' for k, v in sorted(counts.items()) if v))
        if is_annotated:
            holes = _holes(table, key_cols)
            hole_totals[(root_name, stem)] += holes
            line += f'  holes={holes}'
        report.append(line)
    return problems


def audit_root(root_dir: Path, report: list[str]) -> list[str]:
    name = root_dir.name
    problems = []
    hole_totals: Counter = Counter()
    report.append(f'== {name}')
    for split in ('train', 'val', 'test'):
        d = root_dir / split
        if not d.is_dir():
            continue
        for sp in sorted(p for p in d.iterdir() if p.is_dir()):
            try:
                sess = fmt.Session.load(sp)
            except fmt.FormatError as e:
                problems.append(f'{name}/{split}/{sp.name}: {e}')
                continue
            if sess.label_source not in ('annotated', 'tracked'):
                problems.append(f'{name}/{split}/{sp.name}: unknown label_source '
                                f'{sess.label_source!r}')
            problems.extend(audit_session(sess, name, hole_totals, report))

    want = HOLE_EXEMPTIONS.get(name, {})
    seen_tables = {stem for (r, stem) in hole_totals if r == name}
    for stem in seen_tables | set(want):
        got, exp = hole_totals.get((name, stem), 0), want.get(stem, 0)
        if got != exp:
            problems.append(f'{name}: {stem}.pq has {got} hole(s) across annotated sessions, '
                            f'HOLE_EXEMPTIONS says {exp} -- a re-conversion silently changed '
                            f'this. Update HOLE_EXEMPTIONS deliberately if the change is wanted.')
    if want:
        report.append('   TOTAL holes (annotated sessions only): '
                      + ', '.join(f'{k}={hole_totals.get((name, k), 0)} (want {v})'
                                  for k, v in sorted(want.items())))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='*', help='root name(s); default is every non-symlink-farm '
                                              'root under --data')
    ap.add_argument('--data', type=Path, default=ROOT)
    ap.add_argument('--report', action='store_true',
                    help='print the per-session table and exit 0 even on a violation')
    args = ap.parse_args()

    if args.roots:
        names = args.roots
    else:
        names = sorted(p.name for p in args.data.iterdir()
                       if p.is_dir() and not _is_symlink_farm(p))

    report: list[str] = []
    problems: list[str] = []
    for name in names:
        root_dir = args.data / name
        if not root_dir.is_dir():
            print(f'{name}: not a directory under {args.data}', file=sys.stderr)
            sys.exit(2)
        problems.extend(audit_root(root_dir, report))

    print('\n'.join(report))
    if problems:
        print(f'\n{len(problems)} problem(s):', file=sys.stderr)
        for p in problems:
            print(f'  ! {p}', file=sys.stderr)
        if not args.report:
            sys.exit(1)
    else:
        print('\nOK: every root matches the status policy.')


if __name__ == '__main__':
    main()
