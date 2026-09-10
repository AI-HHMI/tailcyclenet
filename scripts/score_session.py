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
    args = parser.parse_args(argv)
    table, _registry, _config = score_root(Path(args.run), args.data, args.split, args.device)
    report = rank(table, args.top)
    print(report)
    write_outputs(Path(args.out), table, Path(args.run), args.data, args.split, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
