#!/usr/bin/env python
"""Add the required `labels` key to every existing session.toml. ONE-OFF -- delete after running.

    pixi run python scripts/backfill_labels_key.py --data <datasets-base> --dry-run
    pixi run python scripts/backfill_labels_key.py --data <datasets-base>
    pixi run python scripts/backfill_labels_key.py --data <datasets-base> --check

`labels` became required in docs/annotation_format.md §4 (decision 6), which means `Session.load`
now raises on every root written before it existed. This backfills the key in place: TOML only, no
parquet rewritten and no symlink touched. That is the whole reason it exists rather than a
converter rerun -- `scratch/sweep/submit.sh`'s preflight was added after a converter running
against a live tree killed two johnson jobs hours in with `FormatError: camera has neither a frame
dir nor a video`.

THE PARTITION IS DECLARED, NOT INFERRED. `ANNOTATED` below is the whole rule; every other root is
tracked. `provenance.annotator_tool` would give the same split today, but it is exactly the
inconsistent metadata that motivated adding `labels` in the first place -- it reads 'anivia' for one
annotated root and 'scripts/convert_johnson.py' for the other -- so keying on it would launder a
guess into a required field. A new root that is neither is a decision someone should have to make.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.format import LABEL_SOURCES, SPLITS, Session

ANNOTATED = {'allen-mouse-annotated', 'johnson-mouse-annotated'}


def session_files(root: Path) -> tuple[list[Path], int]:
    """(every REAL session.toml under one dataset root, count of symlinked sessions skipped).

    Session directories that are symlinks are skipped, which is what makes the `-combined` roots
    inherit the key instead of being written a second time through a different path: they are
    trees of symlinks into the `-annotated` and `-tracked` roots, so writing through them would
    edit the same file twice and, worse, report a file count that double-counts the data.

    The skipped count comes back so the caller can tell a symlink tree (0 real files, many links,
    entirely correct) from a root that has genuinely gone missing (0 of both).
    """
    files, linked = [], 0
    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            continue
        for sess in sorted(d.iterdir()):
            if sess.is_symlink():
                linked += 1
                continue
            if sess.is_dir() and (sess / 'session.toml').exists():
                files.append(sess / 'session.toml')
    return files, linked


def insert_key(text: str, value: str) -> str:
    """Insert `labels = "<value>"` immediately after the `units` line.

    Textual rather than a tomllib -> toml.dumps round-trip: the round-trip reformats the whole
    file, so a diff would show every session.toml as fully rewritten and hide the one line that
    actually changed. `units` is required by §4, so the anchor always exists; if it does not, this
    file is not what it claims to be and stopping is correct.
    """
    m = re.search(r'^units\s*=.*$', text, re.MULTILINE)
    if not m:
        raise ValueError('no `units` line to anchor to')
    return f'{text[:m.end()]}\nlabels = "{value}"{text[m.end():]}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, type=Path, help='folder holding the dataset roots')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check', action='store_true',
                    help='no writes; load every session through Session.load and report')
    args = ap.parse_args()

    roots = sorted(p for p in args.data.iterdir() if p.is_dir())
    if not roots:
        print(f'FATAL: {args.data} holds no dataset roots', file=sys.stderr)
        return 1
    # Typo guard. A misspelled entry here would silently mark a hand-annotated root as tracked,
    # which is invisible afterwards -- the key would be present and valid, just wrong.
    missing = ANNOTATED - {p.name for p in roots}
    if missing:
        print(f'FATAL: ANNOTATED names no root under {args.data}: {sorted(missing)}',
              file=sys.stderr)
        return 1

    total, changed, empty = 0, 0, []
    for root in roots:
        want = 'annotated' if root.name in ANNOTATED else 'tracked'
        files, linked = session_files(root)
        n_ok, n_write, n_wrong = 0, 0, 0
        for cfg in files:
            with open(cfg, 'rb') as f:
                have = tomllib.load(f).get('labels')
            if have in LABEL_SOURCES:
                n_ok += 1
                if have != want:
                    n_wrong += 1
                    print(f'  WRONG {cfg}: has {have!r}, this root should be {want!r}')
                continue
            if have is not None:
                print(f'  WRONG {cfg}: has {have!r}, not in {LABEL_SOURCES}')
                n_wrong += 1
            if not (args.dry_run or args.check):
                cfg.write_text(insert_key(cfg.read_text(), want))
            n_write += 1
        total += len(files)
        changed += n_write
        # Zero real files is correct for a symlink tree -- that IS how the `-combined` roots
        # inherit the key -- and a failure for anything else: a renamed or moved root would
        # otherwise look like a clean no-op run.
        if files:
            note = ''
        elif linked:
            note = f'  (symlink tree, {linked} sessions inherit)'
        else:
            note = '  <-- NO SESSIONS'
            empty.append(root.name)
        print(f'  {root.name:26} {want:10} files={len(files):4d} '
              f'already={n_ok:4d} {"would-write" if args.dry_run or args.check else "wrote"}='
              f'{n_write:4d} wrong={n_wrong}{note}')

    print(f'{total} session.toml, {changed} '
          f'{"to write" if args.dry_run or args.check else "written"}')
    if empty:
        print(f'FATAL: roots with neither real nor symlinked sessions: {empty}', file=sys.stderr)
        return 1

    if args.check:
        bad = 0
        for root in roots:
            for cfg in session_files(root)[0]:
                try:
                    sess = Session.load(cfg.parent)
                except Exception as e:                      # noqa: BLE001 -- report, do not stop
                    print(f'  FAIL {cfg.parent}: {e}')
                    bad += 1
                    continue
                want = 'annotated' if root.name in ANNOTATED else 'tracked'
                if sess.label_source != want:
                    print(f'  FAIL {cfg.parent}: label_source {sess.label_source!r} != {want!r}')
                    bad += 1
        print(f'check: {"OK" if not bad else f"{bad} FAILURE(S)"}')
        return 1 if bad else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
