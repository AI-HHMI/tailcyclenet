"""The cross-view association gate (body units) and the body size that scales it.

Association gates every residual as a WORLD distance divided by the body size, at `--assoc-res-max-body` (default
`ASSOC_RES_MAX_BODY` = 0.15; report 76, '### detr3d-diag'). No pixel gate is used anywhere.

The body size is `--body-length` (world units, the calibration's) when given; otherwise a label-free estimate from
`FIRST_PASS_WINDOWS` sparse pose windows spread evenly over the run's OWN frame range ([start, stop) after
--start-frame / --end-frame / --max-frames; never outside it), with no prior, its association gated in body units
at a PROVISIONAL body size from the detections alone (`associate.provisional_body_size`: box diagonal x depth /
focal over scale-free mutual-nearest pairs; x0.94-x1.36 of GT, inside the gate's flat band): the median over (row,
frame) of the predicted 3D extent diagonal. A range that holds no more than that many windows is used whole. No
finite pose -> the provisional (detection) body size, warned; nothing at all -> refused (pass --body-length).
Measured against the GT extent (offline, from full 30 px deploys; nopri): 4 windows land within 4.5% on CMU Panoptic
a1 / b1, 3dpop Seq59 / Seq29 and social-dannce soc3, 8.4% on social-dannce soc1 (6.4% with every frame); the gate
is flat over +-20-30% body error. A pose-free box-diagonal x depth / focal estimate was 3-28% off (pigeons), so
it is not used.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

from .associate import ASSOC_RES_MAX_BODY

FIRST_PASS_WINDOWS = 4


def resolve_assoc_gate(args):
    """-> dict(mode='body', body=float, why=str): `--assoc-res-max-body` if given, else the default. Pure."""
    v = getattr(args, 'assoc_res_max_body', None)
    if v is not None:
        return dict(mode='body', body=float(v), why='--assoc-res-max-body')
    return dict(mode='body', body=float(ASSOC_RES_MAX_BODY), why='default')


def median_extent(pred):
    """Median over (row, frame) of the 3D extent diagonal ||max_k x - min_k x|| of `pred` (S, T, K, 3);
    rows / frames / keypoints that are NaN are ignored. NaN when nothing is finite."""
    import warnings

    p = np.asarray(pred, np.float64)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ext = np.nanmax(p, 2) - np.nanmin(p, 2)
        d = np.linalg.norm(ext, axis=-1)
        return float(np.nanmedian(d)) if np.isfinite(d).any() else float('nan')


def first_pass_segments(start, stop, window, k=FIRST_PASS_WINDOWS):
    """The frame segments [s, e) of the sparse first pass inside the run's range [start, stop): `k` windows of
    `window` frames spread evenly (first starts at `start`, last ends at `stop`), or the whole range when it holds
    no more than `k` windows. Empty when the range is empty."""
    start, stop, window = int(start), int(stop), max(int(window), 2)
    if stop - start < 2:
        return []
    if stop - start <= k * window:
        return [(start, stop)]
    starts = np.round(np.linspace(start, stop - window, k)).astype(int)
    return [(int(s), int(s) + window) for s in starts]


def run_range(cfg, n_frames):
    """The run's own frame range [start, stop) in one group (`run_blocks`' rule: --end-frame, else --max-frames,
    else the group's end)."""
    stop = min(int(n_frames), int(cfg.frame_stop or cfg.max_frames or n_frames))
    return int(cfg.frame_start), stop


def first_pass_config(args, cfg, start, stop):
    """The (args, cfg) of one first-pass segment [start, stop): the body gate at the run's value with the
    PROVISIONAL body size (`_body_size = None` -> `associate_group` bootstraps it from the detections), no prior
    (`anchor = 'none'`), everything else as the run."""
    a = copy.copy(args)
    a.frame_start, a.frame_stop, a.max_frames = int(start), int(stop), 0
    gate = getattr(args, '_assoc_gate', None) or resolve_assoc_gate(args)
    a._assoc_gate = dict(mode='body', body=float(gate['body']), why='first pass (provisional body)')
    a._body_size = None
    c = dataclasses.replace(cfg, frame_start=int(start), frame_stop=int(stop), max_frames=0, anchor='none')
    return a, c


def estimate_body_size(blocks):
    """Label-free body size from the first pass: `blocks` iterates its (S, T, K, 3) predictions; ->
    (body, frames_used)."""
    blocks = [np.asarray(b) for b in blocks]
    if not blocks:
        return float('nan'), 0
    pred = np.concatenate(blocks, axis=1)
    return median_extent(pred), int(pred.shape[1])
