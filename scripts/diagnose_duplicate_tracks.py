#!/usr/bin/env python3
"""Report row-level identity switches and the duplicate-track signature.

Example::

    pixi run python scripts/diagnose_duplicate_tracks.py pred/ --data DATA --split test

The output is JSON, so it can be retained beside an inference result or piped through ``jq``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tailcyclenet.eval import label_lookup
from tailcyclenet.identity_diagnostic import switch_diagnostics
from tailcyclenet.infer.predictions import load_predictions


def _scales(points):
    """Per-frame median labelled body extent, the one dimensionless reference for both modes.

    Every distance the diagnostic reports is divided by this, so the numbers mean "fraction of a
    body length" whether the labels are 2D pixels or 3D millimetres. The 3D path deliberately
    does NOT use the model's `cube_scale`: on the measured rig one cube unit is about a
    millimetre, so a `--near-scale` in cube units tested a sub-body-length fraction and called
    two rows on one animal (1-3 mm apart, ~0.01 body) "far" (report 53). Missing labels are
    filled only to make the scale probe well-defined; they never enter the switch matcher. The
    fill copies the nearest measured frame's scale, not a prediction.
    """
    points = np.asarray(points, float)
    with np.errstate(all='ignore'):
        span = np.nanmax(points, axis=2) - np.nanmin(points, axis=2)
        scale = np.nanmedian(np.linalg.norm(span, axis=-1), axis=0)
    good = np.isfinite(scale) & (scale > 0)
    if not good.any():
        raise ValueError('could not derive a finite body-extent scale from the labels')
    measured = np.flatnonzero(good)
    for frame in np.flatnonzero(~good):
        scale[frame] = scale[measured[np.argmin(np.abs(measured - frame))]]
    return scale


def build_parser():
    """Build the command-line parser for the JSON diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path, help='prediction session directory or legacy .npz')
    parser.add_argument('--data', type=Path, required=True, help='label dataset root')
    parser.add_argument('--split', default='test')
    parser.add_argument('--near-scale', type=float, default=1.0,
                        help='duplicate-signature threshold in labelled body extents')
    parser.add_argument('--bin-frames', type=int, default=100,
                        help='frame width of the switch histogram')
    return parser


def main(argv=None):
    """Load matching labels/predictions and print one JSON report."""
    args = build_parser().parse_args(argv)
    if args.near_scale < 0 or args.bin_frames <= 0:
        parser = build_parser()
        parser.error('--near-scale must be non-negative and --bin-frames must be positive')
    preds, _ = load_predictions(args.predictions)
    labels = label_lookup(args.data, args.split)
    report = {'prediction': str(args.predictions), 'data': str(args.data), 'split': args.split,
              'normalization': 'labelled body extent (both 2d px and 3d mm)', 'groups': {}}
    for key, out in sorted(preds.items()):
        if key not in labels:
            continue
        lab, sess = labels[key]
        true = lab.points3d if str(out['mode']) == '3d' else lab.points2d[..., 0, :]
        pred = np.asarray(out['pred'], float)
        scale = _scales(true)
        report['groups'][key] = switch_diagnostics(pred, true, scale=scale,
                                                    near_scale=args.near_scale,
                                                    bin_frames=args.bin_frames)
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
