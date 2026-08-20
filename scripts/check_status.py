#!/usr/bin/env python
"""Audit the `status` policy across every tailcycle-dataset root.

    pixi run python scripts/check_status.py                      # every known root
    pixi run python scripts/check_status.py rat-city-annotated    # one root
    pixi run python scripts/check_status.py --report              # never fail, just print

THE POLICY (dev/plans/status_consistency_and_occlusion.md, decisions A-H):

  1. **`missing` claims a real occlusion ASSESSMENT.** It is legal only on the roots this file
     names in `MISSING_OK`, one line per root with the evidence -- an allow-list, not a rule
     derivable from the tables (a converter has to say what its source recorded).
  2. **A hole -- a keypoint slot with NO ROW inside an (animal, frame[, camera]) that carries at
     least one row -- is checked ONLY on `annotated` sessions, and only against a NAMED, EXACT
     count in `HOLE_EXEMPTIONS`, summed over the whole root.** A `tracked` root's dense per-frame
     tracking routinely leaves SOME keypoints untracked while others in the same frame are --
     rat-city-tracked alone has 713,678 such slots -- and that is normal tracker behaviour, not a
     policy violation; this rule exists for the SMALL, DELIBERATE exceptions on hand-annotated
     data (an annotator skip, a triangulation gate, a quality filter), where an unexplained hole
     is much more likely to be a converter bug. A count that MOVES on an `annotated` root is the
     failure this script exists to catch, not the presence of the exemption itself.
  3. **`projected` never appears alongside `visible` in the same table.** They are different
     claims about the same annotator's judgement of the same point; a table carrying both is a
     converter bug, not a root with two kinds of assessment.
  4. **`missing` never appears on an (animal, frame) that carries NO positioned row anywhere in
     the same table.** `PoseDataset._labelled_frames` (dataset.py) counts a `missing` row as
     labelled, so a `missing`-only frame yields index entries `PoseDataset._item` then rejects
     forever (`isfinite(coords).sum() < 2`) -- invisible in the loss curve, and this is the check
     that catches it before a training job does.
  5. **The policy keys on the session's DECLARED `labels`** (`session.toml`'s `labels`, exposed as
     `Session.label_source`), not on the root's folder name -- calms21 declares `tracked` and
     needed a named exemption (`MISSING_OK` has none for it: decision G kept it `visible`)
     rather than a folder-name special case.

Symlink-farm roots (`rat-city` -> `rat-city-tracked`, every `*-combined*` root) are skipped by
STRUCTURE (their session directories are themselves symlinks), not by name -- auditing them again
would just re-read the same parquet through a different path.
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

# Roots that may carry a `missing` row at all, and why. Every other root's tables must be
# `missing`-free -- see decision A/C/D/E/F/G in dev/plans/status_consistency_and_occlusion.md.
MISSING_OK = {
    # decision A: `convert_v4.py` writes these from the npz's OWN per-camera `vis` array, only
    # at slots whose 3D point is finite -- a real assessment, not a NaN standing in for one.
    'allen-mouse-tracked': ('keypoints',),
    # decision D: `convert_annotated.py::triangulate_group` writes `missing` in points3d.pq only
    # where 0 views were visible and EVERY camera assessed the point -- the honest 3D occlusion.
    # Its keypoints table also carries a per-camera `vis` assessment, same shape as
    # allen-mouse-tracked.
    'allen-mouse-annotated': ('keypoints', 'points3d'),
    # decision C left the 171 real `Inf` (APT's fully-occluded sentinel) as `missing`; the 220
    # `NaN` (APT's own "unlabeled" skip) stay absent -- see HOLE_EXEMPTIONS below.
    'rat-city-annotated': ('keypoints',),
}

# root -> {table: exact hole count}, summed over EVERY session in the root. Checked only for
# `annotated` roots (rule 2). Computed by `scratch/survey_fill.py` at the time these decisions
# were taken; re-run it and update this table deliberately if a re-conversion is expected to
# change one of these counts.
HOLE_EXEMPTIONS = {
    # decision C: APT's own `NaN` (annotator skip, `occ == 0`) -- writing these `missing` would
    # claim an assessment nobody made (spec S13). APT's occlusion sentinel is `Inf`, already
    # written `missing` and counted in MISSING_OK above.
    'rat-city-annotated': {'keypoints': 220},
    # decision D: `triangulate_group` already implements the agreed rule and needed no change --
    # the 1-view case and the >=2-views-over-`--max-reproj-px` case are both left as no row
    # ("unlabeled"); a point with 0 views and every camera assessed is the SEPARATE 178 rows this
    # root writes `missing` (counted in MISSING_OK above), not a hole.
    'allen-mouse-annotated': {'points3d': 211},
    # decision E: convert_johnson.py drops outlier 2D as no row on purpose -- a rejected
    # measurement, not an occlusion.
    'johnson-mouse-annotated': {'keypoints': 19},
    # `convert_johnson_aug.py` re-runs the SAME outlier gate over its own (larger, augmented) set
    # of frame variants, independently of the plain root -- 18, not 19, is not a discrepancy: a
    # frameset drop (`n_fs dropped` in that script's own log) can remove the one variant carrying
    # the plain root's 19th outlier without implying anything about the other 18.
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

    `key_cols` is `(group_id, frame, animal_id[, camera])`; the missing axis is `bodypart`. A
    hole is counted per keypoint, matching `scratch/survey_fill.py`'s definition exactly.
    """
    import pandas as pd  # noqa: F401 -- pulled in by pyarrow's to_pandas()

    df = table.select([*key_cols, 'bodypart']).to_pandas()
    n_kpts = df['bodypart'].nunique()
    live_keys = df.groupby(key_cols, observed=True).ngroups   # instances with >= 1 row
    return int(live_keys * n_kpts - len(df))


def _is_symlink_farm(root_dir: Path) -> bool:
    """True if this root's OWN identity is a symlink -- either the root itself, or every session
    directory under it. `rat-city` is the first shape; every `*-combined*` root is the second."""
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
