"""Diagnostics for duplicate-track identity switches.

This module is intentionally model-free.  It compares the row-level Hungarian matches used by
``tailcyclenet.metrics.mota`` with the prediction rows, so it can distinguish a genuine crossing
from two prediction rows sitting on one labelled animal.  Every reported distance is divided by
the labelled body extent (the per-frame median labelled span, in the labels' own units), so a
``near_scale`` threshold means "fraction of a body length" for 2D pixels and 3D millimetres
 alike -- the same reference the 2D path always used.  The model's ``cube_scale`` is NOT used:
on the measured 3D rig one cube unit is ~1 mm, so cube units turned a body-sized duplicate
signature (two rows ~2 mm apart on a ~200 mm animal) into a number that looked "far" (report 53).
"""
from __future__ import annotations

import warnings

import numpy as np

from .metrics import match_instances


def _mean_distance(first, second):
    """Mean Euclidean distance over keypoints finite in both instances."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    return float(np.linalg.norm(a[ok] - b[ok], axis=-1).mean()) if ok.any() else float('nan')


def _nearest_other_distance(points, row, frame):
    """Distance from one labelled instance to its nearest other labelled instance."""
    values = [_mean_distance(points[other, frame], points[row, frame])
              for other in range(points.shape[0]) if other != row]
    values = np.asarray(values, float)
    return float(np.nanmin(values)) if np.isfinite(values).any() else float('nan')


def _frame_scale(scale, frame):
    """Select one scalar body-extent scale for one frame."""
    value = float(scale if np.ndim(scale) == 0 else np.asarray(scale, float)[frame])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'scale must be finite and positive at frame {frame}, got {value}')
    return value


def switch_diagnostics(pred, true, scale, near_scale=1.0, bin_frames=100):
    """Return switch records and duplicate-signature summaries.

    ``pred`` and ``true`` are ``(instances, frames, keypoints, coordinates)`` arrays.  A switch is
    defined exactly as in the evaluator: the row matched to one GT instance changes between two
    successive frames.  Distances are divided by the supplied per-frame ``scale`` -- the labelled
    body extent, from ``diagnose_duplicate_tracks._scales`` -- before being reported, so the
    records are body-length fractions, comparable across rigs and dimensionalities. The record's
    ``nearest_gt_distance`` is the labelled animal's nearest *other* animal distance, while
    ``predicted_pair_distance`` is the distance between the old and new prediction rows. The
    latter is the useful duplicate signature: it is small when the Hungarian changed ownership
    between two rows that both sit on one animal.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if near_scale < 0 or bin_frames <= 0:
        raise ValueError('near_scale must be non-negative and bin_frames must be positive')
    if pred.ndim != 4 or true.ndim != 4:
        raise ValueError(f'expected pred/true with four dimensions, got {pred.shape}/{true.shape}')
    T = min(pred.shape[1], true.shape[1])
    pred, true = pred[:, :T], true[:, :T]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        span = np.nanmax(true, axis=2) - np.nanmin(true, axis=2)
        extent = float(np.nanmedian(np.linalg.norm(span, axis=-1))) if span.size else float('nan')
    radius = extent * 0.5 if np.isfinite(extent) and extent > 0 else np.inf
    matches = match_instances(pred, true, radius, 0.0, 'mean')
    switches = []
    last = {}
    for frame, pairs in enumerate(matches):
        for pred_ix, true_ix, _ in pairs:
            old = last.get(true_ix)
            if old is not None and old != pred_ix:
                switches.append({
                    'frame': int(frame),
                    'gt_row': int(true_ix),
                    'old_pred_row': int(old),
                    'new_pred_row': int(pred_ix),
                    'nearest_gt_distance': (_nearest_other_distance(true, true_ix, frame)
                                            / _frame_scale(scale, frame)),
                    'predicted_pair_distance': (_mean_distance(pred[old, frame], pred[pred_ix, frame])
                                                / _frame_scale(scale, frame)),
                })
            last[true_ix] = pred_ix

    bins = {}
    for item in switches:
        start = (item['frame'] // bin_frames) * bin_frames
        key = f'{start}-{start + bin_frames - 1}'
        bins[key] = bins.get(key, 0) + 1
    near = [item for item in switches
            if np.isfinite(item['predicted_pair_distance'])
            and item['predicted_pair_distance'] <= near_scale]
    values = np.asarray(scale, float)
    values = values[np.isfinite(values) & (values > 0)]
    return {
        'n_switches': len(switches),
        'switches': switches,
        'switches_by_frame_bin': bins,
        'near_coincident_count': len(near),
        'near_coincident_threshold': float(near_scale),
        'median_body_extent': (float(np.median(values))
                               if values.size and np.isfinite(extent) else float('nan')),
        'bin_frames': int(bin_frames),
    }
