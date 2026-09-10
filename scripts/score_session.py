#!/usr/bin/env python
"""CLI shim for `tailcyclenet.scorer.qc`."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.scorer.qc import rank, score_root, write_outputs


def main(argv=None):
    """Score a tracked root and write a worst-first QC report.

    Inputs: argv -- argument list, or None for `sys.argv`.
    Outputs: a process exit code.
    Side effects: writes `scores.pq`, `report.txt` and `provenance.toml` under `--out`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='a scorer run folder')
    parser.add_argument('--data', required=True, help='the tracked root to score')
    parser.add_argument('--split', default='test', help='which split of --data to score')
    parser.add_argument('--out', required=True, help='where the QC artefacts go')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--window-offset', type=int, default=None,
                        help='shift the window lattice by N frames, so a track is judged under a '
                             'framing other than the one that produced it')
    parser.add_argument('--spans-csv', default=None,
                        help='CSV of session,group,animal,span_start,span_len: score only windows '
                             'starting inside each span (a targeted look at a clip bad stretch)')
    parser.add_argument('--val-stride', type=int, default=None,
                        help='window spacing; default is the run n_frames (non-overlapping)')
    args = parser.parse_args(argv)
    spans = None
    if args.spans_csv:
        import csv as _csv
        spans = {}
        with open(args.spans_csv) as f:
            for r in _csv.DictReader(f):
                lo = int(r['span_start'])
                spans[(r['session'], r['group'], str(r['animal']))] = (lo, lo + int(r['span_len']))
    table, _registry, _config = score_root(Path(args.run), args.data, args.split,
                                          args.device, args.limit,
                                          args.window_offset, args.val_stride, spans)
    report = rank(table, args.top)
    print(report)
    write_outputs(Path(args.out), table, Path(args.run), args.data, args.split, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
