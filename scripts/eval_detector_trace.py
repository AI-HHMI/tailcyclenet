#!/usr/bin/env python
"""Attribute visible GT box misses to detector decode stages.

This is an opt-in, model-forwarding diagnostic. It keeps the detector's ordinary output path
unchanged while retaining the candidates that ``decode`` discards:

    pixi run python scripts/eval_detector_trace.py --run runs/det --data ROOT \
        --split test --out scratch/trace.json

The output is a raw-detector ledger, not a deployment metric: 2D linking, multiview association,
tracking, and window outcomes remain the separate counters printed by ``infer.py``. GT boxes are
crop-rule boxes in source pixels. Only animals with at least one finite projected/2D keypoint (or a
finite stored instance box) are counted; this deliberately does not call every NaN an FN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import crop_box_for_points
from tailcyclenet.detector import detect_raw, load_detector
from tailcyclenet.detector.assign import box_iou
from tailcyclenet.format import VISIBLE, load_datasets, sessions_for
from posetail.posetail.cube import project_points_torch


STAGES = ('all', 'score', 'nms', 'final')
REASONS = ('localization', 'score', 'nms', 'top_k', 'kept')


def _gt_boxes(sess, gid, frame, ci, box_source, min_crop_dim):
    """Return visible/in-view source-pixel GT crop boxes for one frame-camera."""
    lab = sess.labels(gid)
    if box_source == 'instances' and lab.boxes is not None:
        b = np.asarray(lab.boxes[:, frame, ci], dtype=np.float32)
        return b[np.isfinite(b).all(-1)]

    if sess.mode == '3d':
        cams = sess.cgroup(gid, frame)
        points = project_points_torch([cams[ci]], torch.as_tensor(lab.points3d[:, frame]))[0]
        points = points.detach().cpu().numpy()
        status = None
    else:
        points = np.asarray(lab.points2d[:, frame, :, ci], dtype=np.float32)
        status = None if lab.vis2d is None else np.asarray(lab.vis2d[:, frame, :, ci])

    out = []
    for s, pts in enumerate(points):
        # For 2D, a VISIBLE status is the only positive visibility claim. A 3D projection has no
        # per-camera status in some roots, so finite projected points are the conservative in-view
        # proxy; roots with explicit per-camera visibility use it instead.
        if status is not None and not np.any(status[s] == VISIBLE):
            continue
        finite = np.isfinite(pts).all(-1)
        if not finite.any():
            continue
        box = crop_box_for_points(torch.as_tensor(pts[finite]),
                                  torch.as_tensor(sess.rig.size(sess.cam_names[ci])),
                                  min_crop_dim)
        if box is not None:
            out.append(box.cpu().numpy().astype(np.float32))
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


def _matched(gt, pred, threshold):
    """Greedy one-to-one GT recall at IoU threshold."""
    if not len(gt) or not len(pred):
        return np.zeros(len(gt), dtype=bool)
    iou = box_iou(torch.as_tensor(gt), torch.as_tensor(pred)).numpy()
    pairs = [(float(iou[g, p]), g, p) for g in range(iou.shape[0])
             for p in range(iou.shape[1]) if iou[g, p] >= threshold]
    got_g, got_p = set(), set()
    for _, g, p in sorted(pairs, reverse=True):
        if g not in got_g and p not in got_p:
            got_g.add(g)
            got_p.add(p)
    out = np.zeros(len(gt), dtype=bool)
    out[list(got_g)] = True
    return out


def _attribute(gt, record, threshold):
    matches = {stage: _matched(gt, np.asarray(record[f'{stage}_boxes'], np.float32), threshold)
               for stage in STAGES}
    reasons = []
    for i in range(len(gt)):
        if not matches['all'][i]:
            reasons.append('localization')
        elif not matches['score'][i]:
            reasons.append('score')
        elif not matches['nms'][i]:
            reasons.append('nms')
        elif not matches['final'][i]:
            reasons.append('top_k')
        else:
            reasons.append('kept')
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--split', default='test')
    ap.add_argument('--boxes', default='keypoints', choices=('keypoints', 'instances'))
    ap.add_argument('--min-crop-dim', type=int, default=None)
    ap.add_argument('--score-thresh', type=float, default=0.01)
    ap.add_argument('--nms-iou', type=float, default=0.5)
    ap.add_argument('--nms-center-dist', type=float, default=0.5)
    ap.add_argument('--top-k', type=int, default=24)
    ap.add_argument('--max-frames', type=int, default=0,
                    help='prefix length when --start-frame is zero')
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--end-frame', type=int, default=0,
                    help='exclusive source-frame end; must be a detector batch boundary unless it '
                         'is the group end')
    ap.add_argument('--iou', type=float, default=0.5,
                    help='GT matching IoU threshold; default 0.5')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model, input_wh, _, min_crop_dim, reduce, box_source, tile_scale, _ = load_detector(
        args.run, device=device)
    if box_source != args.boxes:
        print(f'WARNING: detector was trained on {box_source!r}, scoring GT {args.boxes!r}')
    if (args.data / 'session.toml').exists():
        ds_name, sessions = sessions_for(args.data, args.split)
        session_items = [(ds_name, sess) for sess in sessions]
    else:
        session_items = [(ds.name, sess) for ds in load_datasets(args.data)
                         for sess in ds.sessions.get(args.split, [])]
    groups_out = {}
    for ds_name, sess in session_items:
        for gid, group in sess.groups.items():
                trace = []
                end = args.end_frame or args.max_frames or group.n_frames
                frames = (None if args.start_frame == 0 and not args.end_frame
                          else np.arange(args.start_frame, min(end, group.n_frames)))
                detect_raw(model, input_wh, sess, gid, top_k=args.top_k, device=device,
                           score_thresh=args.score_thresh, reduce=reduce,
                           max_frames=min(end, group.n_frames), tile_scale=tile_scale,
                           frames=frames, iou_thresh=args.nms_iou,
                           center_dist_thresh=args.nms_center_dist,
                           trace=trace, trace_detail=True)
                counts = {reason: 0 for reason in REASONS}
                by_camera = {}
                n_gt = 0
                for record in trace:
                    gt = _gt_boxes(sess, gid, int(record['frame']),
                                   sess.cam_names.index(record['camera']), args.boxes,
                                   args.min_crop_dim or min_crop_dim)
                    reasons = _attribute(gt, record, args.iou)
                    n_gt += len(reasons)
                    camera = by_camera.setdefault(record['camera'],
                                                  {'n_gt': 0, **{r: 0 for r in REASONS}})
                    camera['n_gt'] += len(reasons)
                    for reason in reasons:
                        counts[reason] += 1
                        camera[reason] += 1
                key = f'{sess.session_id}/{gid}'
                groups_out[key] = {'n_gt': n_gt, **counts, 'by_camera': by_camera,
                                   'n_records': len(trace)}
                print(f'{key}: n_gt={n_gt} ' + ' '.join(f'{k}={v}' for k, v in counts.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'run': str(args.run.resolve()), 'data': str(args.data.resolve()),
                                    'split': args.split, 'score_thresh': args.score_thresh,
                                    'nms_iou': args.nms_iou,
                                    'nms_center_dist': args.nms_center_dist,
                                    'gt_iou': args.iou, 'groups': groups_out}, indent=1) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
