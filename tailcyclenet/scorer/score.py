"""Score a dataset root's tracks with a trained scorer and write a worst-first QC report.

Body of `tailcyclenet score` (`tailcyclenet/__main__.py`) and of `scripts/score_session.py`
(`pixi run python scripts/score_session.py --run <scorer run> --data <root> --out qc/`) -- both
call this module's `main()` unchanged.

Writes `scores.pq` (per source frame and keypoint), `report.txt` (worst windows and keypoints
first) and `provenance.toml` under `--out`. Scores are relative: lower means more likely wrong.
To score a model's own predictions instead, pass `--scorer` to `tailcyclenet infer`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .contract import scorer_output_granularity
from .qc import rank, score_root, write_outputs


class _Deleted(argparse.Action):
    """A retired flag: refuse it by name and say why, rather than "unrecognized arguments"."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Raise SystemExit naming the flag.

        Inputs: the argparse Action protocol. Outputs: never returns.
        Side effects: raises SystemExit.
        """
        raise SystemExit(
            f'{option_string} is deleted: report 60 measured that offsetting the scorer\'s window '
            f'lattice does not make it catch more errors ("do not offset the scorer\'s windows")')


def main(argv=None):
    """Score a tracked root and write a worst-first QC report.

    Inputs: argv -- argument list, or None for `sys.argv`.
    Outputs: a process exit code.
    Side effects: writes `scores.pq`, `report.txt` and `provenance.toml` under `--out`; framewise
        runs also write raw `window_scores.pq` and canonicalize rows by source-frame ownership.
    """
    parser = argparse.ArgumentParser(prog='tailcyclenet score', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, help='a scorer run folder')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint filename; checkpoint_best.pth is validation-selected and '
                             'must be explicitly named')
    parser.add_argument('--data', required=True, help='the tracked root to score')
    parser.add_argument('--split', default='test', help='which split of --data to score')
    parser.add_argument('--session', default=None,
                        help='score one exact session id instead of the whole split')
    parser.add_argument('--start', type=int, default=None,
                        help='zero-based inclusive window-index start')
    parser.add_argument('--stop', type=int, default=None,
                        help='zero-based exclusive window-index stop')
    parser.add_argument('--out', required=True,
                        help='where QC artefacts go (frame mode also writes window_scores.pq)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--window-offset', action=_Deleted, help=argparse.SUPPRESS)
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
    coverage = []
    checkpoint_info = {}
    table, _registry, _config = score_root(
        Path(args.run), args.data, args.split, args.device, args.limit,
        val_stride=args.val_stride, spans=spans, coverage=coverage,
        checkpoint=args.checkpoint, checkpoint_info=checkpoint_info,
        session=args.session, start=args.start, stop=args.stop)
    report = rank(table, args.top)
    print(report)
    write_outputs(Path(args.out), table, Path(args.run), args.data, args.split, report, coverage,
                  checkpoint_file=checkpoint_info.get('checkpoint_file'),
                  checkpoint_iteration=checkpoint_info.get('checkpoint_iteration'),
                  output_granularity=scorer_output_granularity(_config), session=args.session,
                  start=args.start, stop=args.stop, val_stride=args.val_stride)
    return 0
