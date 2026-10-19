"""Pure flag refusals and resolution: everything checked before a checkpoint loads."""
from __future__ import annotations

from pathlib import Path


def refuse_selector_beside_file(path, selector, flag, selector_flag, default=None):
    """A checkpoint selector picks a file INSIDE a run folder; beside a file it means nothing.

    Pure (no checkpoint load), so the refusal lands before any model or video is touched.
    """
    if path is None:
        return
    path = Path(path)
    if path.is_file() and selector is not None and selector != default:
        raise SystemExit(f'{flag} {path} is already a checkpoint file; {selector_flag} '
                         f'{selector!r} selects a file inside a run folder and is refused here')


def check_frame_range(args) -> None:
    """Refusals on `--start-frame` / `--end-frame`; pure argument arithmetic, so it fires above
    both input branches and before the checkpoint loads.

    Resolves the ONE quantity the loop reads: `args.frame_start` / `args.frame_stop`, with
    `--max-frames N` folding into `frame_stop = N`. The group-length check lives in the group loop.
    Three refusals fire here, numbered 15-17 in the infer/adopt refusal ledger.
    """
    if args.max_frames and (args.start_frame or args.end_frame):
        raise SystemExit(
            f'--max-frames {args.max_frames} together with --start-frame {args.start_frame} / '
            f'--end-frame {args.end_frame} has two readings of equal force -- "N frames from the '
            'start" and "up to frame N, from the start" -- and no defensible precedence, so a '
            'rule here would make one of them silently wrong. --max-frames N IS --start-frame 0 '
            '--end-frame N; pick one spelling.')
    if args.start_frame < 0 or args.end_frame < 0:
        raise SystemExit(
            f'--start-frame {args.start_frame} / --end-frame {args.end_frame}: negative frame '
            'indices are not Python negative indexing here. "-1 means the last frame" and "-1 '
            'means one before the end" are both obvious and differ by one, which is the class of '
            'ambiguity that costs a day.')
    if args.end_frame and args.end_frame <= args.start_frame:
        raise SystemExit(
            f'--end-frame {args.end_frame} is not past --start-frame {args.start_frame}. The '
            'range is half-open [start, end), so this predicts nothing; an empty range is a typo, '
            'not a protocol.')
    args.frame_start = int(args.start_frame)
    args.frame_stop = int(args.end_frame or args.max_frames or 0)
