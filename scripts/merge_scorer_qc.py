#!/usr/bin/env python
"""Merge disjoint scorer-QC outputs into one split-level result."""
import argparse
import re
import sys
from pathlib import Path

import polars as pl
import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.scorer.qc import _canonical_frame_table, rank


def _load_provenance(part: Path) -> dict:
    path = part / 'provenance.toml'
    if not path.exists():
        raise SystemExit(f'{part}: missing provenance.toml')
    return toml.load(path)


def _check_common(provenance: list[dict]) -> None:
    keys = ('scorer_run', 'source_root', 'split', 'session', 'checkpoint_file',
            'checkpoint_iteration', 'output_granularity', 'val_stride', 'window_offset')
    first = provenance[0]
    for index, current in enumerate(provenance[1:], 1):
        for key in keys:
            if current.get(key) != first.get(key):
                raise SystemExit(
                    f'part {index} disagrees on provenance key {key!r}: '
                    f'{current.get(key)!r} != {first.get(key)!r}')


def _part_range(part: Path) -> tuple[int, int]:
    match = re.fullmatch(r'part-(\d+)-(\d+)', part.name)
    if match is None:
        raise SystemExit(f'{part}: expected directory name part-START-STOP')
    start, stop = (int(value) for value in match.groups())
    if stop < start:
        raise SystemExit(f'{part}: invalid range [{start}, {stop})')
    return start, stop


def _check_ranges(parts: list[Path], expected_total: int) -> list[tuple[int, int]]:
    ranges = [_part_range(part) for part in parts]
    if len(set(ranges)) != len(ranges):
        raise SystemExit('duplicate train part ranges')
    ordered = sorted(ranges)
    cursor = 0
    for start, stop in ordered:
        if start != cursor:
            raise SystemExit(f'train part ranges have a gap or overlap at {cursor}: {ordered}')
        cursor = stop
    if expected_total is not None and cursor != expected_total:
        raise SystemExit(f'train parts end at {cursor}, expected {expected_total}')
    return ordered


def _merge_coverage(parts: list[Path], out: Path, expected_total: int,
                    expected_index_start: int) -> None:
    paths = [part / 'coverage.csv' for part in parts]
    present = [path.exists() for path in paths]
    if any(present) and not all(present):
        raise SystemExit('coverage.csv must be present for every part or none')
    if not all(present):
        return
    coverage_schema = {
        'index': pl.Int64, 'session': pl.String, 'group': pl.String,
        'animal': pl.String, 'start': pl.Int64, 'status': pl.String, 'reason': pl.String,
    }
    tables = [pl.read_csv(path, schema_overrides=coverage_schema) for path in paths]
    coverage = pl.concat(tables).sort('index')
    if coverage.height and coverage.get_column('index').n_unique() != coverage.height:
        raise SystemExit('coverage contains duplicate dataset indices across parts')
    if expected_total is not None:
        expected = set(range(expected_index_start, expected_index_start + expected_total))
        actual = set(coverage.get_column('index').to_list())
        if actual != expected:
            raise SystemExit(
                f'coverage indices are not a complete partition: missing={len(expected - actual)}, '
                f'extra={len(actual - expected)}')
    coverage.write_csv(out / 'coverage.csv')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', required=True, nargs='+', type=Path,
                        help='part output directories to merge')
    parser.add_argument('--out', required=True, type=Path,
                        help='final split output directory')
    parser.add_argument('--expected-total', type=int, required=True,
                        help='expected number of global window indices')
    parser.add_argument('--expected-index-start', type=int, default=0,
                        help='first global window index expected in coverage.csv')
    args = parser.parse_args(argv)
    parts = args.part
    if not parts:
        parser.error('at least one --part is required')
    if len(set(parts)) != len(parts):
        raise SystemExit('the same part directory was supplied more than once')
    ranges = _check_ranges(parts, args.expected_total)
    part_by_range = {_part_range(part): part for part in parts}
    parts = [part_by_range[rng] for rng in ranges]
    provenance = [_load_provenance(part) for part in parts]
    _check_common(provenance)
    for part, current, (start, stop) in zip(parts, provenance, ranges):
        if (current.get('window_start_index'), current.get('window_stop_index')) != (start, stop):
            raise SystemExit(f'{part}: provenance window range disagrees with directory name')
    frame_mode = provenance[0].get('output_granularity') == 'frame'
    filename = 'window_scores.pq' if frame_mode else 'scores.pq'
    paths = [part / filename for part in parts]
    if not all(path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise SystemExit(f'missing part output(s): {missing}')
    tables = [pl.read_parquet(path) for path in paths]
    if any(table.schema != tables[0].schema for table in tables[1:]):
        raise SystemExit('part parquet schemas differ')
    raw = pl.concat(tables)
    canonical = _canonical_frame_table(raw) if frame_mode else raw
    args.out.mkdir(parents=True, exist_ok=True)
    if frame_mode:
        raw.write_parquet(args.out / 'window_scores.pq', compression='snappy')
    canonical.write_parquet(args.out / 'scores.pq', compression='snappy')
    (args.out / 'report.txt').write_text(rank(canonical) + '\n')
    _merge_coverage(parts, args.out, args.expected_total, args.expected_index_start)

    merged = dict(provenance[0])
    merged.pop('window_start_index', None)
    merged.pop('window_stop_index', None)
    merged.update({
        'merged_parts': [str(part) for part in parts],
        'part_ranges': [[start, stop] for start, stop in ranges],
        'n_parts': len(parts),
        'n_rows': int(canonical.height),
    })
    if frame_mode:
        merged.update({
            'n_raw_rows': int(raw.height),
            'frame_reducer': 'last_window_last_occurrence',
            'score_table': 'scores.pq',
            'raw_score_table': 'window_scores.pq',
        })
    (args.out / 'provenance.toml').write_text(toml.dumps(merged))
    print(f'wrote merged {args.out}/scores.pq from {len(parts)} parts')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
