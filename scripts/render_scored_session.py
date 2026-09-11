#!/usr/bin/env python
"""Render a session's predictions COLOURED BY A SCORER'S QUALITY SCORE.

    pixi run python scripts/render_scored_session.py \
        --scores <qc dir or scores.pq> --data <session dir> \
        --group <gid> --start 88000 --end 89000 --out <dir>

`scripts/score_session.py` scores WINDOWS (per keypoint) and writes `scores.pq`; this draws those
scores back onto the pixels. The colour is the score of the window the frame belongs to, so the
mapping from frame to window is the scorer's own FRAMING, not a guess: a frame belongs to the last
window that contains it (the same seam rule the eval code uses -- a window's first frame is shared
with the previous window, and letting the EARLIER window win would leave the seam frames scored by
a window the lattice did not place there).

Scores are RELATIVE (`qc.py`'s own contract): the colour map is normalised over the rendered
range, a legend is burned in, and absolute numbers are printed rather than implied. Nothing here
writes to the dataset -- output is one mp4 per camera under `--out`.

The window LENGTH is not in `scores.pq`; it is read from the scorer run named in the `qc` folder's
`provenance.toml` (or passed with `--n-frames`), because getting it wrong would silently mis-assign
every frame.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt
from tailcyclenet.dataset import read_frames


def window_length(scores: Path, override: int | None) -> int:
    """Frames per scored window: `--n-frames`, else the scorer run's own `[data].n_frames`.

    Inputs: scores -- a `scores.pq` or the `qc` directory holding it; override -- the CLI figure.
    Outputs: the window length in frames.
    Side effects: reads `provenance.toml` and the scorer run's `config.toml`.
    """
    if override is not None:
        return int(override)
    prov = scores if scores.is_dir() else scores.parent
    prov = prov / 'provenance.toml'
    if not prov.exists():
        raise SystemExit(f'{prov}: missing, so the window length is unknown. Keep the qc '
                         'directory that score_session.py wrote, or pass --n-frames.')
    doc = tomllib.loads(prov.read_text())
    run = Path(doc['scorer_run'])
    config = run / 'config.toml'
    if not config.exists():
        raise SystemExit(f'{config}: missing, so the window length is unknown. Pass --n-frames.')
    n_frames = tomllib.loads(config.read_text())['data']['n_frames']
    print(f'window length {n_frames} frames (from {config})')
    return int(n_frames)


def load_scores(path: Path, session: str, group: str, animal: str) -> dict:
    """`{(window_start, keypoint): score}` for one (session, group, animal), plus its precision.

    Inputs: path -- a `scores.pq` or the directory holding it; the three keys to select rows.
    Outputs: `{'score': dict, 'precision': dict, 'starts': sorted array, 'starts_seen': set}`.
    Side effects: reads one parquet file.
    """
    import pandas as pd

    pq = path if path.is_file() else path / 'scores.pq'
    if not pq.exists():
        raise SystemExit(f'{pq}: not found')
    df = pd.read_parquet(pq)
    want = ((df.session == session) & (df.group == group) & (df.animal.astype(str) == animal))
    df = df[want]
    if df.empty:
        have = df.session.unique() if len(df) else []
        raise SystemExit(f'{pq}: no rows for {session}/{group}/{animal}. '
                         f'Sessions present: {list(have)[:3]}')
    return {
        'score': {(int(r.start), str(r.keypoint)): float(r.score) for r in df.itertuples()},
        'precision': {(int(r.start), str(r.keypoint)): float(r.precision) for r in df.itertuples()},
        'starts': np.unique(df.start.to_numpy()),
    }


# The colour scale: RED at the low end, GREEN at the high end, fixed so two clips are comparable.
SCORE_RANGE = (-0.5, 0.5)
BAD = (0, 0, 255)
GOOD = (0, 255, 0)
UNSCORED = (128, 128, 128)


def frame_to_window(starts: np.ndarray, n_frames: int, t: int) -> int | None:
    """The window a frame belongs to: the LAST one containing it, or None outside the scored set.

    Inputs: starts -- the scored window starts, ascending; n_frames -- the window length;
            t -- a frame index.
    Outputs: the chosen window start, or None when no scored window covers `t`.
    Side effects: none.
    """
    pos = int(np.searchsorted(starts, t, side='right'))
    if pos == 0:
        # `pos - 1` would be -1 and pick the LAST window, so a frame before the first scored
        # window would be coloured by a window from the end of the clip.
        return None
    start = int(starts[pos - 1])
    if t >= start + n_frames:
        return None
    return start


def _colour(score: float, lo: float, hi: float):
    """BGR for one score: RED at `lo`, GREEN at `hi`, linear between, CLAMPED outside.

    BGR (cv2's order), so `BAD` is (0, 0, 255) = red and `GOOD` is (0, 255, 0) = green.
    The scale is FIXED by default (`SCORE_RANGE`) rather than normalised per clip: scores are
    relative, so a per-clip ramp would repaint the same track differently in a different clip and
    two renders could not be compared. A score past either end is clamped to that end and reads as
    "at least this bad/good", which is what the legend states.
    """
    frac = 0.0 if hi <= lo else float(np.clip((score - lo) / (hi - lo), 0.0, 1.0))
    return tuple(int(round(a + (b - a) * frac)) for a, b in zip(BAD, GOOD))


def draw_scored(im: np.ndarray, lab: fmt.Labels, t: int, ci: int, names: list[str],
                skeleton: list[list[str]], scores: dict, start: int | None, lo: float, hi: float,
                note: str, stats: dict) -> np.ndarray:
    """One frame with every plotted keypoint coloured by its window's score.

    Inputs: im -- a BGR frame; lab -- the group's dense labels; t -- the frame; ci -- the camera
            index; names/skeleton -- the session's axis; scores -- `(start, keypoint)` -> score;
            start -- the covering window (None = unscored); lo/hi -- the colour range; note -- the
            first legend line; stats -- per-window (min, median, mean precision).
    Outputs: a new BGR frame with the overlay drawn.
    Side effects: none.
    """
    import cv2

    im = np.ascontiguousarray(im).copy()
    s = max(1, int(round(im.shape[1] / 1200)))
    ix = {n: i for i, n in enumerate(names)}
    a = 0
    pts = lab.points2d[a, t, :, ci] if lab.points2d is not None else None
    vis = lab.vis2d[a, t, :, ci] if lab.vis2d is not None else None

    for u, v in skeleton:
        if u not in ix or v not in ix or pts is None:
            continue
        p, q = pts[ix[u]], pts[ix[v]]
        if np.isfinite(p).all() and np.isfinite(q).all():
            cv2.line(im, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (200, 200, 200), s)

    for k in range(len(names)):
        if pts is None or not np.isfinite(pts[k]).all():
            continue
        if vis is not None and vis[k] == fmt.UNLABELED:
            continue
        score = scores.get((start, names[k])) if start is not None else None
        colour = _colour(score, lo, hi) if score is not None else UNSCORED
        centre = (int(pts[k][0]), int(pts[k][1]))
        r = 5 * s if score is not None else 3 * s
        cv2.circle(im, centre, r, colour, -1)
        cv2.circle(im, centre, r, (30, 30, 30), max(1, s // 2))

    for i, line in enumerate([note, stats.get(start, 'unscored: no window covers this frame')]):
        y = (26 + 26 * i) * s
        cv2.putText(im, line, (8 * s, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), 4 * s)
        cv2.putText(im, line, (8 * s, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255), s)
    return im


def window_stats(scores: dict, precisions: dict, lo: float, hi: float) -> dict:
    """One legend line per window: its worst and median score and mean precision.

    Inputs: scores/precisions -- `(window start, keypoint)` -> value; lo/hi -- the colour range,
            printed so the ramp is readable rather than implied.
    Outputs: `{window start: text}`.
    Side effects: none.
    """
    by_window: dict[int, list] = {}
    for (start, kpt), value in scores.items():
        by_window.setdefault(start, []).append((kpt, value))
    out = {}
    for start, rows in by_window.items():
        values = [v for _k, v in rows]
        prec = [precisions[(start, k)] for k, _v in rows if (start, k) in precisions]
        out[start] = (f'score {lo:.3f}..{hi:.3f} (colour)  window worst {min(values):.3f}'
                      f'  median {float(np.median(values)):.3f}'
                      f'  mean precision {float(np.mean(prec)) if prec else float("nan"):.3f}')
    return out


def main() -> int:
    """Render the span for every requested camera and print each path."""
    import cv2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', type=Path, required=True,
                    help='scores.pq, or the qc directory score_session.py wrote')
    ap.add_argument('--data', type=Path, required=True, help='the session directory')
    ap.add_argument('--group', required=True)
    ap.add_argument('--animal', default='det00')
    ap.add_argument('--start', type=int, required=True, help='first frame (inclusive)')
    ap.add_argument('--end', type=int, required=True, help='last frame (exclusive)')
    ap.add_argument('--cameras', default='', help='comma-separated; default is every camera')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-frames', type=int, default=None, help='window length; default from the '
                                                               'scorer run in provenance.toml')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--stride', type=int, default=1, help='write every Nth frame')
    ap.add_argument('--score-range', default='',
                    help=f'lo,hi for the colour map; default {SCORE_RANGE[0]},{SCORE_RANGE[1]} '
                         '(red at lo, green at hi, clamped outside)')
    args = ap.parse_args()

    n_frames = window_length(args.scores, args.n_frames)
    session = fmt.Session.load(args.data)
    group = session.groups[args.group]
    table = load_scores(args.scores, session.session_id, args.group, args.animal)
    starts = table['starts']
    scores = table['score']

    vals = [v for (st, _k), v in scores.items() if args.start <= st < args.end]
    if not vals:
        raise SystemExit(f'no scored windows start in [{args.start}, {args.end}); the scored '
                         f'windows are {starts.min()}..{starts.max()}')
    lo, hi = SCORE_RANGE
    if args.score_range:
        lo, hi = (float(x) for x in args.score_range.split(','))
    inside = sum(lo <= v <= hi for v in vals)
    print(f'{len(vals)} keypoint scores over {len(starts)} windows; this span runs '
          f'{np.min(vals):.4f}..{np.max(vals):.4f}, of which {inside} of {len(vals)} fall inside '
          f'the colour range {lo:g}..{hi:g} (red..green; the rest clamp to the end)')

    cameras = ([c.strip() for c in args.cameras.split(',')] if args.cameras
               else list(session.cam_names))
    args.out.mkdir(parents=True, exist_ok=True)
    stats = window_stats(scores, table['precision'], lo, hi)
    labels = session.labels(args.group)
    frames = list(range(args.start, args.end, args.stride))
    for cam in cameras:
        ci = session.cam_names.index(cam)
        path = args.out / f'{session.session_id}_{args.group}_{cam}_{args.start}_{args.end}.mp4'
        writer = None
        # Decode in bounded blocks: the span at native resolution is several GB if held at once.
        for lo_i in range(0, len(frames), 200):
            block = frames[lo_i:lo_i + 200]
            for t, im in zip(block, read_frames(group, cam, block)):
                if im is None:
                    continue
                win = frame_to_window(starts, n_frames, t)
                note = f'{args.animal} {cam}  frame {t}  {t / group.fps:.2f}s  window {win}'
                out_im = draw_scored(im, labels, t, ci, session.names, session.skeleton,
                                     scores, win, lo, hi, note, stats)
                h = int(round(out_im.shape[0] * args.width / out_im.shape[1]))
                out_im = cv2.resize(out_im, (args.width, h), interpolation=cv2.INTER_AREA)
                if writer is None:
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                                             float(group.fps) / args.stride,
                                             (out_im.shape[1], out_im.shape[0]))
                    if not writer.isOpened():
                        raise SystemExit(f'could not open {path}')
                writer.write(out_im)
        if writer is not None:
            writer.release()
            print(path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
