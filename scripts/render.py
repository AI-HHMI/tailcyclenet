#!/usr/bin/env python
"""Draw a prediction over the pixels it was made from. Offline, and model-free.

    pixi run python scripts/render.py --pred pred.npz --data <dataset> --out clips/

**IT USED TO BE `infer.py --render`, AND MOVING IT OUT IS NOT A TIDY-UP.** The window loop now
runs a group a BLOCK of windows at a time and writes each block out as it finishes, so there is no
moment at which a whole clip's `pred` exists in memory -- which is exactly what `render_group`
wants. Rendering inside the loop would mean either holding the block's frames until the encode
finished (splitting ownership of the one thing that must be evicted on schedule) or rendering
synchronously per block, which roughly doubles a run for a diagnostic.

What it buys, beyond the loop staying simple: you can render a prediction you already have,
without re-running inference. `tailcyclenet/render.py` itself is unchanged -- it always took a
session and an array, and the two now come from two places.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.format import sessions_for
from tailcyclenet.render import render_group


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True, type=Path, help='a prediction file from infer.py')
    ap.add_argument('--data', required=True, type=Path,
                    help='the dataset root or session directory the prediction was made from')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--cams', default='0', help='comma-separated camera indices')
    ap.add_argument('--zoom', type=int, default=0,
                    help='side in SOURCE px of a window following the prediction; 0 = whole frame')
    ap.add_argument('--groups', default='', help='comma-separated group ids; default all')
    args = ap.parse_args(argv)

    z = np.load(args.pred, allow_pickle=True)
    keys = [str(k) for k in z['__keys__']]
    want = set(args.groups.split(',')) if args.groups else None
    cams = [int(c) for c in args.cams.split(',') if c.strip() != '']

    _, sessions = sessions_for(args.data, args.split)
    by_id = {s.session_id: s for s in sessions}
    args.out.mkdir(parents=True, exist_ok=True)

    for key in keys:
        sid, gid = key.split('/', 1)
        if want and gid not in want:
            continue
        # A KEY WITH NO SESSION IS A MISMATCHED --data, not an empty render. Skipping silently is
        # how a run gets rendered against the wrong root and nobody notices.
        if sid not in by_id:
            raise SystemExit(f'{args.pred}: predicted session {sid!r} is not in {args.data}. '
                             f'Sessions present: {sorted(by_id)[:5]} ...')
        sess = by_id[sid]
        sess.preload()
        pred = z[f'{key}|pred']
        for ci in cams:
            out = args.out / f'{key.replace("/", "__")}__{sess.cam_names[ci]}.mp4'
            print(f'{key}: wrote {render_group(sess, gid, pred, out, cam=ci, zoom=args.zoom)}')


if __name__ == '__main__':
    main()
