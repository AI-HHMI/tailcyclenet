"""The association gate's unit: the body-unit gate (default) or the legacy pixel gate, and the body size.

`resolve_assoc_gate` decides, once per run (one source session per run), which gate `associate` uses:

1. `--assoc-res-max-px` given  -> the legacy PIXEL gate at that value (an explicit override).
2. `--assoc-res-max-body` given -> the body-unit gate at that value, even over a session's own pixel key.
3. session.toml declares `assoc_res_max_px` -> the legacy pixel gate at the session's value (logged: an
   explicit data key is honoured, never silently ignored; e.g. 3dpop's 2026-09-05 data edit).
4. otherwise -> the BODY-UNIT gate at `ASSOC_RES_MAX_BODY` (0.15).

The body size, in this order: session.toml `body_length` (world units, the calibration's), else a label-free
estimate -- the median 3D extent diagonal (max - min over keypoints, per row and frame) of the pose output from a
first pass over the session's first `FIRST_PASS_FRAMES` frames of its first selected group, run at the legacy
pixel gate (`estimate_body_size`, called by the driver). Measured within 1-4% of the GT extent on CMU
Panoptic, 3dpop and social-dannce (report 76, '### detr3d-diag'); the gate is flat over 0.10-0.20 body.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

from .associate import ASSOC_RES_MAX_BODY

FIRST_PASS_FRAMES = 300


def resolve_assoc_gate(args, sess):
    """-> dict(mode='px'|'body', px=float, body=float|None, why=str). Pure; prints nothing.

    `body` is the gate in body units (None in px mode); the body SIZE is resolved separately
    (`body_size_from_session` / `estimate_body_size`) because the label-free estimate needs the model.
    """
    cli_px = getattr(args, 'assoc_res_max_px', None)
    cli_body = getattr(args, 'assoc_res_max_body', None)
    if cli_px is not None and cli_body is not None:
        raise SystemExit('--assoc-res-max-px and --assoc-res-max-body are exactly-one-of: one is the '
                         'legacy pixel gate, the other the body-unit gate.')
    if cli_px is not None:
        return dict(mode='px', px=float(cli_px), body=None, why='--assoc-res-max-px')
    pixel_levers = [f for f, v in (('--assoc-res-max-sides', getattr(args, 'assoc_res_max_sides', None)),
                                   ('--assoc-score-order', getattr(args, 'assoc_score_order', False)))
                    if v not in (None, False)]
    if cli_body is None and getattr(sess, 'assoc_res_max_px_declared', False):
        return dict(mode='px', px=float(sess.assoc_res_max_px), body=None,
                    why='session.toml assoc_res_max_px')
    if pixel_levers:
        raise SystemExit(f'{" / ".join(pixel_levers)} are levers of the legacy PIXEL association gate; this '
                         'run uses the body-unit gate. Pass --assoc-res-max-px to run the pixel gate.')
    if cli_body is not None:
        return dict(mode='body', px=float(sess.assoc_res_max_px), body=float(cli_body),
                    why='--assoc-res-max-body')
    return dict(mode='body', px=float(sess.assoc_res_max_px), body=float(ASSOC_RES_MAX_BODY),
                why='default')


def body_size_from_session(sess):
    """session.toml `body_length` (world units) or None."""
    v = getattr(sess, 'body_length', None)
    return None if v is None else float(v)


def median_extent(pred):
    """Median over (row, frame) of the 3D extent diagonal ||max_k x - min_k x|| of `pred` (S, T, K, 3);
    rows / frames / keypoints that are NaN are ignored. NaN when nothing is finite."""
    p = np.asarray(pred, np.float64)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ext = np.nanmax(p, 2) - np.nanmin(p, 2)
            d = np.linalg.norm(ext, axis=-1)
            return float(np.nanmedian(d)) if np.isfinite(d).any() else float('nan')


def first_pass_config(args, cfg, n_frames, px):
    """The (args, cfg) of the label-free first pass: frames [0, min(FIRST_PASS_FRAMES, n_frames)) of one
    group, everything else as the run, association at the LEGACY pixel gate `px`."""
    stop = int(min(FIRST_PASS_FRAMES, n_frames))
    a = copy.copy(args)
    a.frame_start, a.frame_stop, a.max_frames = 0, stop, 0
    a._assoc_gate = dict(mode='px', px=float(px), body=None, why='first pass')
    a._body_size = None
    c = dataclasses.replace(cfg, frame_start=0, frame_stop=stop, max_frames=0)
    return a, c


def estimate_body_size(blocks):
    """Label-free body size from a first pass: `blocks` iterates the pass's (S, T, K, 3) predictions (the
    driver's own detect + `run_blocks` loop over `first_pass_config`); -> (body, frames_used)."""
    blocks = list(blocks)
    if not blocks:
        return float('nan'), 0
    pred = np.concatenate(blocks, axis=1)
    return median_extent(pred), int(pred.shape[1])
