#!/usr/bin/env python
"""Draw a prediction over the pixels it was made from. Offline, and model-free.

    pixi run python scripts/render.py --pred pred/ --out clips/

`--pred` is a prediction SESSION directory (not an npz -- an npz carries no provenance saying
where its pixels are); `--data` overrides a root that moved, checked against the prediction.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.infer.predictions import load_predictions
from tailcyclenet.render import render_group, resolve_camera, session_for_prediction


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser for scripts/render.py."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True, type=Path,
                    help='a prediction SESSION directory from infer.py. An .npz is refused -- '
                         'see this file\'s docstring for why.')
    ap.add_argument('--out', required=True, type=Path, help='output directory')
    ap.add_argument('--data', type=Path, default=None,
                    help='OVERRIDE for a root that moved since the run. Checked against the '
                         'prediction\'s own provenance, never trusted outright. The normal '
                         'invocation omits this entirely.')
    ap.add_argument('--split', default=None,
                    help='only meaningful with --data (the prediction\'s own source_split is '
                         'used otherwise); default test')
    ap.add_argument('--cams', default='0', help='comma-separated camera names or indices')
    ap.add_argument('--groups', default=None, help='comma-separated group ids; default all')
    ap.add_argument('--zoom', type=int, default=0,
                    help='side in SOURCE px of a window following the prediction; 0 = whole frame')
    ap.add_argument('--start-frame', type=int, default=None,
                    help='first SOURCE frame to draw. Default: the prediction\'s own recorded '
                         '--start-frame. Narrows the predicted range; cannot extend past it.')
    ap.add_argument('--end-frame', type=int, default=None,
                    help='one past the last SOURCE frame to draw, 0 = to the end. Default: the '
                         'prediction\'s own recorded --end-frame.')
    ap.add_argument('--draw', default='pred', choices=('pred', 'keypoints', 'both'),
                    help="'pred' (default): the prediction itself -- a 3D reprojection or the 2D "
                         "pose. 'keypoints': camera cam's OWN 2D head output (keypoints.pq), no "
                         "projection. 'both': the reprojection with the per-camera head overlaid "
                         'as thin crosses -- the two are different quantities and this is the '
                         'only place their disagreement is legible.')
    ap.add_argument('--boxes', action='store_true',
                    help='draw instances.pq boxes in each animal\'s own colour')
    ap.add_argument('--fps', type=float, default=15)
    ap.add_argument('--max-side', type=int, default=1600)
    return ap


def _frame_range(prov, n_total, args):
    """The [f0, f1) this call draws: the requested range intersected with the range this
    prediction actually covers (extending past it would draw guaranteed-NaN frames)."""
    pred_start = int(prov.get('frame_start', 0) or 0)
    # 0 = to the end.
    pred_stop = int(prov.get('frame_stop', 0) or 0) or n_total
    req_start = args.start_frame if args.start_frame is not None else pred_start
    req_stop = (pred_stop if args.end_frame is None
               else (n_total if args.end_frame == 0 else args.end_frame))
    f0 = max(0, pred_start, req_start)
    f1 = min(n_total, pred_stop, req_stop)
    return f0, f1


def main(argv=None):
    """Render a prediction session's groups over their source pixels.

    Inputs: argv -- CLI args (defaults to sys.argv[1:]).
    Side effects: writes mp4 clips under --out; prints per-group lines.
    """
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.pred.suffix == '.npz':
        raise SystemExit(
            f'--pred {args.pred}: an npz carries no provenance, so it cannot say which pixels '
            'it was made from -- rendering it would mean restating the source root by hand, '
            'which is the exact failure this tool exists to avoid. scripts/eval.py still scores '
            'this file directly; to look at it, re-run scripts/infer.py from the run folder it '
            'names and render the session that produces.')
    if not (args.pred / 'session.toml').exists():
        raise SystemExit(f'--pred {args.pred}: no session.toml -- this is not a prediction '
                         'session directory (docs/annotation_format.md).')

    with open(args.pred / 'session.toml', 'rb') as f:
        cfg = tomllib.load(f)
    prov = cfg.get('provenance', {})

    # The session first: it is the cheaper and more fundamental check, and reads no pixels.
    sess = session_for_prediction(args.pred, data=args.data, split=args.split)

    want_groups = [g.strip() for g in args.groups.split(',') if g.strip()] if args.groups else None
    preds, meta = load_predictions(args.pred, groups=want_groups)
    if not preds:
        raise SystemExit(f'--groups {args.groups!r}: none of the requested groups is in '
                         f'{args.pred}.')

    cams = [resolve_camera(sess, tok.strip()) for tok in args.cams.split(',') if tok.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    for key, out in preds.items():
        sid, gid = key.split('/', 1)
        if gid not in sess.groups:
            raise SystemExit(
                f'{key}: group {gid!r} was predicted but is not in the source session '
                f'{sess.session_id!r} ({sorted(sess.groups)[:5]}...). This looks like a '
                're-converted root; pass --data to point at the right one.')

        n_total = sess.groups[gid].n_frames
        f0, f1 = _frame_range(prov, n_total, args)
        if f1 <= f0:
            raise SystemExit(f'{key}: the requested range is empty after clamping to the '
                             f'predicted range and the group\'s {n_total} frame(s).')
        frames = np.arange(f0, f1)

        pred_arr = np.asarray(out['pred'])[:, f0:f1]
        p2 = np.asarray(out['pred2d'])[:, f0:f1] if 'pred2d' in out else None
        boxes = np.asarray(out['boxes'])[:, f0:f1] if args.boxes and 'boxes' in out else None

        finite = float(np.isfinite(pred_arr).all(-1).any(-1).mean()) if pred_arr.size else 0.0
        print(f'{key}: {f1 - f0} frame(s) [{f0},{f1}), {finite:.1%} of (animal, frame) finite')

        for ci in cams:
            cam_name = sess.cam_names[ci]
            out_path = args.out / f'{key.replace("/", "__")}__{cam_name}.mp4'
            # `draw_arr` must be 4D: `render_group` only auto-slices the camera axis on `overlay`
            # and `boxes`.
            draw_arr, overlay = pred_arr, None
            if args.draw in ('keypoints', 'both'):
                if p2 is None:
                    print(f'{key}/{cam_name}: --draw {args.draw} needs keypoints.pq, which this '
                         'prediction does not carry; falling back to --draw pred.')
                elif args.draw == 'keypoints':
                    draw_arr = p2[:, :, ci]
                else:
                    # (S,T,C,K,2); render_group slices camera `ci` itself.
                    overlay = p2
            render_group(sess, gid, draw_arr, out_path, cam=ci, zoom=args.zoom, boxes=boxes,
                        frames=frames, overlay=overlay, fps=args.fps, max_side=args.max_side)
            print(f'{key}: wrote {out_path}')


if __name__ == '__main__':
    main()
