"""Deployment-shaped detector quality over one whole clip (`deployment_score`).

Lives here, not in `detector/`, because it runs the deployment path (`infer.detect.detect_raw`,
`infer.associate.associate_group`, `infer.window.window_starts`), and nothing in `detector/`
may import `infer`.
"""
from __future__ import annotations

import numpy as np

from ..detector.boxes import NMS_CENTER_DIST
from ..detector.diagnostics import gt_crop_sides
from ..infer.associate import associate_group
from ..infer.detect import detect_raw
from ..infer.window import window_starts


def deployment_score(model, sess, gid, input_wh, device='cpu', top_k=24, max_animals=None,
                     det_score=0.01, track=True, link=False, min_views=2, max_move=1.25,
                     min_crop_dim=64, reduce=False, tile_scale=None, max_frames=0,
                     n_frames=24, overlap=4, min_box_frames=1, batch=16, iou_thresh=0.5,
                     center_dist_thresh=NMS_CENTER_DIST, box_source=None):
    """Deployment-shaped detector quality over ONE WHOLE CLIP, no frame sampling.

    `score_dataset` answers "what fraction of SAMPLED frames does the detector recall a box on",
    which does not predict the box path's real cost: current-generation checkpoints read val r@.5
    0.91-0.93 and IoU 0.79 while costing 43% of pose coverage and +25 px MPJPE. This runs the SAME
    functions deployment does -- `detect_raw` then `associate_group`, and `infer.window.window_starts`
    for the window rule `run_group` uses -- over a whole test group, and reports what a pose
    window actually gets:

        det_fill       fraction of (frame, camera) with >=1 box surviving `det_score` --
                       the DETECTION-ONLY ceiling, independent of row identity.
        slot_fill      fraction of (row, frame, camera) with a finite box AFTER association --
                       identity-bearing only under `track=True` (3D multiview) or `link=True`
                       (2D). Comparing it against `det_fill` separates "the detector missed the
                       animal" from "association dropped a detection it had".
        window_miss    fraction of (row, window) with fewer than `min_box_frames` finite boxes
                       anywhere in the window -- the EXACT quantity `infer.run_group` marks
                       `no box`, using its own `window_starts` rule.
        union_side_px  quantiles of each window's per-camera UNION box side (SOURCE px), the
                       crop-inflation number `--refine` exists to survive.
        gt_side_px     quantiles of the crop-RULE box side over this group's own labels, as an
                       UNPAIRED population comparison against `union_side_px`.

    `box_source` is this session's root's effective box source (`detect_raw`'s argument; None =
    the checkpoint's global one). Pass the per-root value deployment resolves, or a multi-root
    `extent` detector is scored under another root's crop pad.

    Returns a dict of floats/arrays; `n_gt` and `n_windows` say how much each rested on.
    """

    n_animals = max_animals or max(1, len(sess.labels(gid).animal_ids))
    raw = detect_raw(model, input_wh, sess, gid, top_k=max(top_k, n_animals), device=device,
                     batch=batch, score_thresh=det_score, reduce=reduce, max_frames=max_frames,
                     tile_scale=tile_scale, iou_thresh=iou_thresh,
                     center_dist_thresh=center_dist_thresh, box_source=box_source)
    r_box, r_sc = raw
    D, T, C = r_sc.shape
    det_fill = float(np.isfinite(r_sc[0]).mean()) if D else 0.0

    out, sc = associate_group(raw, sess, gid, n_animals, link=link, min_views=min_views,
                              track=track, max_move=max_move)
    S, T, C = sc.shape
    slot_fill = float(np.isfinite(sc).mean())

    starts = window_starts(T, n_frames, overlap)
    miss, union_sides = [], []
    for a in range(S):
        for st in starts:
            frames = np.arange(st, min(st + n_frames, T))
            bb = out[a][frames]
            n_ok = int(np.isfinite(bb).all(-1).sum())
            miss.append(n_ok < min_box_frames)
            if n_ok:
                for ci in range(C):
                    v = bb[:, ci][np.isfinite(bb[:, ci]).all(-1)]
                    if len(v):
                        x0, y0 = v[:, 0].min(), v[:, 1].min()
                        x1, y1 = v[:, 2].max(), v[:, 3].max()
                        union_sides.append(0.5 * ((x1 - x0) + (y1 - y0)))

    gt_sides = gt_crop_sides(sess, gid, min_crop_dim, max_frames=max_frames)
    q = (0.5, 0.9, 0.99)

    def _quant(a):
        """{p: quantile} at q=(0.5, 0.9, 0.99); NaN-filled when `a` is empty."""
        a = np.asarray(a, float)
        return {p: float(np.quantile(a, p)) for p in q} if a.size else {p: float('nan') for p in q}

    return {'det_fill': det_fill, 'slot_fill': slot_fill,
           'window_miss': float(np.mean(miss)) if miss else float('nan'),
           'n_windows': len(miss), 'n_gt': int(gt_sides.size),
           'union_side_px': _quant(union_sides), 'gt_side_px': _quant(gt_sides)}
