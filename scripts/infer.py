#!/usr/bin/env python
"""Run a trained model. The only entry point that touches a checkpoint.

    # every group of a split, cropping from the labels (the GT-crop upper bound)
    pixi run python scripts/infer.py --run runs/w9 --data <dataset> --split test --out pred.npz

    # one session, query-free
    pixi run python scripts/infer.py --run runs/w9 --data <dataset>/test/<session> \\
        --anchor none --out pred.npz

    # crops from a detections file (the deployment number)
    pixi run python scripts/infer.py --run runs/w9 --data <dataset> --boxes dets.npz --out p.npz

A run folder carries its own config and keypoint registry, so `--run` is the whole model
specification and a config/checkpoint mismatch cannot happen.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.checkpoints import load_run
from tailcyclenet.format import Session, load_dataset
from tailcyclenet.infer import ANCHORS, InferConfig, run_group


def sessions_for(path: Path, split: str):
    """(dataset_name, [Session]) from either a session directory or a dataset root."""
    path = Path(path)
    if (path / 'session.toml').exists():
        return path.parent.parent.name, [Session.load(path)]
    ds = load_dataset(path)
    return ds.name, ds.sessions.get(split, [])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path, help='dataset root or one session dir')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--anchor', default='carry', choices=ANCHORS,
                    help="'labels' is an ORACLE, not a deployment number")
    ap.add_argument('--overlap', type=int, default=4)
    ap.add_argument('--n-frames', type=int, default=None, help='default: the run\'s own')
    ap.add_argument('--boxes', type=Path, default=None,
                    help='npz of crop points per group; default is to crop from the labels')
    ap.add_argument('--detector', type=Path, default=None,
                    help='a detector run folder. THE deployment path: boxes come from pixels, '
                         'not from labels.')
    ap.add_argument('--det-input-wh', type=int, nargs=2, default=None,
                    help='letterbox size the detector was TRAINED at. Only needed for a '
                         'posetail-pose checkpoint, which keeps it in a config file rather than '
                         'in the weights.')
    ap.add_argument('--det-score', type=float, default=0.99,
                    help='objectness floor for a detection. 0.99, not the 0.05 that reads like a '
                         'floor: the objectness is SATURATED -- 98.5%% of rat-city\'s boxes and '
                         '99.98%% of 3dpop\'s sit at exactly 1.0 -- so anything between 0.05 and 0.5 '
                         'moves 1-3%% of boxes and does nothing. At 0.99 it is worth MOTA +0.074 '
                         '[+0.009, +0.154] on 3dpop with MPJPE -2.11 mm and coverage +0.010 also '
                         'SIG, and +0.073 on rat-city. The gain is `fp_none`, which is what a score '
                         'threshold should remove. Its SIZE tracks how many boxes are actually '
                         'below it (6-8%% on those two, 0.9%% on calms21, where it reads +0.012 '
                         'n.s.), so a detector whose scores are not saturated wants this lowered -- '
                         'the per-group box coverage printed below is how that shows up.')
    ap.add_argument('--det-cache', type=Path, default=None,
                    help='npz of per-group detector boxes; read if it exists, written if not. '
                         'Detection is the expensive half of a run and depends on no model, so '
                         'several arms over one clip set should share ONE set of boxes -- which '
                         'makes them a matched comparison by construction instead of by trusting '
                         'the detector to be deterministic. The box source is recorded in it and '
                         'checked on load, so a cache from a different detector cannot be reused '
                         'silently.')
    ap.add_argument('--link-boxes', action='store_true',
                    help='follow one animal per instance row across frames (greedy IoU). Detector '
                         'rows are score-ordered and unlinked by default, which makes the window '
                         'crop the union over several different animals.')
    ap.add_argument('--min-views', type=int, default=2, choices=(1, 2),
                    help='3D only. 2 is cross-view association as it stands: every instance is '
                         'built from a camera PAIR, so an animal only one view saw is dropped from '
                         'the frame entirely. 1 also emits each leftover box as a single-view '
                         'instance -- coverage against precision, since a leftover box is exactly '
                         'one the geometry never corroborated. Whether the pose model can USE a '
                         'one-camera 3D window is the run\'s own `[data].prob_2d_only`: the shipped '
                         '3dpop and rat-city runs set 0.25, configs/w9.toml sets 0, and under 0 it '
                         'is an untrained input shape. Measured on 3dpop: no metric moves.')
    ap.add_argument('--dup-res-px', type=float, default=None,
                    help='only with --min-views 1: drop a leftover single-view box that reprojects '
                         'within this many pixels of an instance already accepted in that camera. '
                         'Those are second detections of an animal that is already in the output, '
                         'so they are fp_dup rather than new coverage. Default keeps them all, '
                         'which is the unconditional rule and carries the full FP risk.')
    ap.add_argument('--max-animals', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=0,
                    help='predict only the first N frames of each group. A PREFIX, not a sample: '
                         '`carry` needs the frames contiguous.')
    ap.add_argument('--render', type=Path, default=None,
                    help='also write <dir>/<session>__<group>__<cam>.mp4 of the prediction. A 3D '
                         'prediction is REPROJECTED into each rendered camera.')
    ap.add_argument('--render-cams', default='0',
                    help='comma-separated camera indices to render. A reprojection hides a depth '
                         'error along its own ray, so a 3D render wants more than one.')
    ap.add_argument('--render-zoom', type=int, default=0,
                    help='side, in SOURCE pixels, of a window that follows the prediction. 0 '
                         'draws the whole frame, which on a 3208x2200 rig makes a 100 px mouse '
                         'unreadable. A view only -- it changes nothing that was predicted.')
    ap.add_argument('--refine', action='store_true',
                    help='re-crop each window to the first pass\'s OWN prediction and predict '
                         'again, label-free. The prediction re-enters the crop rule as if it were '
                         'the labels, so the second pass sees the box a GT crop would have given -- '
                         'the only arm that beat every detector crop on 3dpop. Costs one extra '
                         'forward AND one extra decode per animal per window: the crop moves, so '
                         'neither the pixels nor the scene encode can be reused.')
    ap.add_argument('--vis-thresh', type=float, default=None,
                    help='withhold an (animal, frame) row whose MEDIAN `vis_pred` logit across '
                         'keypoints is below this. Measured against a rate-matched random rejection '
                         '-- the only honest control, since any rejection flatters a mean over '
                         'matched points -- it is worth MOTA +0.049 SIG on 3dpop at 7.3% of rows '
                         'and 0.601 -> 0.628 on rat-city at 14%, where the control gains nothing. '
                         'A LOGIT, and NOT PORTABLE: rat-city sits at a median of +2.7 and 3dpop at '
                         '+15.4, so pick it per dataset from the run\'s own `conf` field. Applies '
                         'to what is reported, never to the carried prompt.')
    ap.add_argument('--prior-vis-thresh', type=float, default=None,
                    help='drop a CARRIED keypoint from the prompt when the previous window\'s own '
                         '`vis_pred` LOGIT for it is below this. A logit, not a probability: 0.0 is '
                         'p = 0.5. Gates individual prompt keypoints, not rows -- it changes what '
                         'the model is told, never what it reports. Off by default.')
    ap.add_argument('--kpt-chunk', type=int, default=0,
                    help='decode keypoints in slices of this size, reusing one scene encode. '
                         'Lowers peak memory on large keypoint sets; the prediction is '
                         'unchanged. 0 = one pass.')
    ap.add_argument('--groups', default=None, help='comma-separated group ids to restrict to')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model, config, registry, ckpt = load_run(args.run, args.checkpoint, device=device)
    print(f'model: {ckpt.name}  ({registry.n_keypoints} keypoints)')

    trained_frames = int(config['data'].get('n_frames', 24))
    # LONGER THAN THE TRAINED WINDOW IS NOT A KNOB. `n_frames` sizes the temporal pos_embed the
    # checkpoint carries; asking for more frames than it was built for interpolates at best and
    # shape-errors deep inside the encoder at worst. Shorter is safe -- val/test already enumerate
    # fixed windows -- so only the ceiling is guarded.
    if args.n_frames and args.n_frames > trained_frames:
        raise SystemExit(f'--n-frames {args.n_frames} exceeds the run\'s trained window '
                         f'({trained_frames}). Shorter windows are fine; longer is not the same '
                         'model.')
    cfg = InferConfig(
        n_frames=args.n_frames or trained_frames,
        overlap=args.overlap, image_size=int(config['data'].get('image_size', 256)),
        min_crop_dim=int(config['data'].get('min_crop_dim', 64)),
        box_source=config['data'].get('box_source', 'keypoints'),
        anchor=args.anchor, max_animals=args.max_animals, max_frames=args.max_frames,
        kpt_chunk=args.kpt_chunk, prior_vis_thresh=args.prior_vis_thresh,
        vis_thresh=args.vis_thresh, refine=args.refine, device=device)
    if args.prior_vis_thresh is not None and args.anchor != 'carry':
        raise SystemExit('--prior-vis-thresh gates the CARRIED prompt, so it only means anything '
                         f'under --anchor carry; got {args.anchor!r}.')
    if cfg.box_source != 'keypoints':
        print(f'crops: box_source={cfg.box_source} (from the run config); a session with no '
              'instances.pq falls back to its keypoints')
    if args.anchor == 'labels':
        print('WARNING: --anchor labels seeds the model with GROUND TRUTH. This is an oracle '
              'upper bound, not a deployment number. Label it as such wherever you quote it.')

    boxes = dict(np.load(args.boxes, allow_pickle=True)) if args.boxes else {}
    det = det_wh = None
    if args.detector:
        from tailcyclenet.detector import detect_group, load_detector
        det, det_wh, det_ds, det_mcd, det_red, det_boxsrc = load_detector(
            args.detector, device, input_wh=args.det_input_wh)
        print(f'detector: {args.detector} ({det_wh[0]}x{det_wh[1]}, trained on {det_ds!r}, '
              f'boxes={det_boxsrc or "keypoints"})')
        # The detector regresses THE CROP RULE'S box, so its floor has to be the pose model's
        # floor. Same shapes and same losses if they differ, so nothing else would say so.
        if det_mcd != cfg.min_crop_dim:
            raise SystemExit(
                f'{args.detector}: trained at min_crop_dim={det_mcd}, but this run\'s '
                f'[data].min_crop_dim is {cfg.min_crop_dim}. The detector reproduces the crop '
                'rule; two floors are two rules.')
        # THE OTHER HALF OF THE SAME CONTRACT, and it is not a hard failure: a detector trained on
        # `instances` boxes is the best rat-city detector on record (recall 0.531 vs 0.429) while
        # every rat-city pose run was trained on keypoint-extent crops, so this arm is a legitimate
        # thing to run -- it just is not a detector-quality comparison against a keypoint-trained
        # one, because the crop source moved too (eval rule 4). The npz records which.
        if (det_boxsrc or 'keypoints') != cfg.box_source:
            print(f'WARNING: detector boxes are {det_boxsrc!r} but this run was trained on '
                  f'{cfg.box_source!r} crops. Two crop sources are two crop rules -- do not read '
                  'a delta against a run whose detector matched as a detector-quality result.')
    ds_name, sessions = sessions_for(args.data, args.split)
    want = set(args.groups.split(',')) if args.groups else None
    render_cams = [int(c) for c in args.render_cams.split(',') if c.strip() != '']

    # The box cache. `stamp` is every input the boxes depend on: reusing a cache written under a
    # different detector, threshold or animal count would quietly make one arm incomparable to
    # the next, which is the kind of mismatch that gets published (eval rule 4).
    #
    # ONLY THE OPTIONS THAT DIFFER FROM THEIR DEFAULTS, and sorted by name. A positional list of
    # every value invalidated every cache on disk each time a new flag was added to the end of it --
    # three times in one afternoon, twice in the middle of a sweep, each time refusing boxes that
    # were in fact exactly the right boxes. Under this form a new option carrying its default leaves
    # the stamp untouched, while any option a run actually set is still in it.
    #
    # `det_score` is UNCONDITIONAL, and that is the exception the rule needs: its default moved from
    # 0.05 to 0.99, so "equals the default" means different boxes before and after. Omitting it would
    # let a cache built under the old default be reused silently under the new one -- precisely the
    # mismatch this stamp exists to catch. Anything whose default may move belongs on this line.
    stamp = repr(sorted([('det_score', str(args.det_score))]
                        + [(k, str(getattr(args, k))) for k in
                           ('detector', 'max_animals', 'link_boxes', 'det_input_wh',
                            'max_frames', 'min_views', 'dup_res_px')
                           if getattr(args, k) != ap.get_default(k)]))
    det_cache, cache_dirty = {}, False
    if args.det_cache and args.det_cache.exists():
        loaded = dict(np.load(args.det_cache, allow_pickle=True))
        got = str(loaded.pop('__stamp__', ''))
        if got != stamp:
            raise SystemExit(f'{args.det_cache}: written for {got}\n'
                             f'{" " * len(str(args.det_cache))}  now running {stamp}\n'
                             'Delete it or point --det-cache somewhere else.')
        det_cache = loaded
        print(f'detector boxes: {sum(1 for k in det_cache if not k.endswith("|score"))} '
              f'cached group(s) from {args.det_cache}')

    results = {}
    for sess in sessions:
        sess.preload()
        for gid in sess.groups:
            if want and gid not in want:
                continue
            key = f'{sess.session_id}/{gid}'
            det_boxes = det_scores = None
            if det is not None:
                # Default to what the session actually holds, not to 1. `detect_group` caps
                # detections at this count, so a bare --detector run on a ten-animal dataset
                # used to return one animal per frame and read as a catastrophic miss rate
                # rather than as a missing flag.
                n_want = args.max_animals or max(1, len(sess.labels(gid).animal_ids))
                if key in det_cache:
                    det_boxes, det_scores = det_cache[key], det_cache.get(f'{key}|score')
                    print(f'{key}: up to {n_want} animal(s), boxes from --det-cache')
                else:
                    print(f'{key}: detecting up to {n_want} animal(s)'
                          f'{"" if args.max_animals else " (from the labels; set --max-animals)"}',
                          flush=True)
                    det_boxes, det_scores = detect_group(
                        det, det_wh, sess, gid, n_want, device=device,
                        score_thresh=args.det_score, link=args.link_boxes,
                        reduce=det_red, max_frames=args.max_frames,
                        min_views=args.min_views, dup_res_px=args.dup_res_px)
                    det_cache[key] = det_boxes
                    det_cache[f'{key}|score'] = det_scores
                    cache_dirty = True
                # HOW MUCH THE THRESHOLD LEFT. `--det-score` defaults to 0.99 because objectness is
                # saturated on every detector shipped here; a detector whose scores are NOT
                # saturated would lose most of its boxes to that, and this line is where that shows
                # up rather than downstream as an unexplained miss rate.
                filled = float(np.isfinite(det_boxes).all(-1).mean())
                print(f'{key}: boxes in {filled:.3f} of (animal, frame, camera) slots'
                      f'{"   <-- LOW: try a smaller --det-score" if filled < 0.25 else ""}')
            out = run_group(model, sess, gid, registry, ds_name, cfg,
                            box_points=boxes.get(key), boxes_stc=det_boxes)
            if det_scores is not None:
                # The objectness each crop was accepted on, beside the prediction it produced.
                # `--det-score` is then an offline sweep instead of a re-detection per threshold.
                out['det_score'] = det_scores[:, :out['pred'].shape[1]]
            results[key] = out
            print(f'{key}: {out["pred"].shape} '
                  f'{np.isfinite(out["pred"]).all(-1).mean():.3f} finite')
            if args.render is not None:
                from tailcyclenet.render import render_group
                for ci in render_cams:
                    cam_name = sess.cam_names[ci]
                    mp4 = render_group(sess, gid, out['pred'],
                                       args.render / f'{key.replace("/", "__")}__{cam_name}.mp4',
                                       cam=ci, zoom=args.render_zoom)
                    print(f'{key}: wrote {mp4}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, out in results.items():
        for field, value in out.items():
            flat[f'{key}|{field}'] = value
    flat['__keys__'] = np.asarray(list(results), object)
    flat['__run__'] = np.asarray(str(args.run))
    flat['__anchor__'] = np.asarray(cfg.anchor)
    flat['__boxes__'] = np.asarray(
        str(args.detector) if args.detector else
        (str(args.boxes) if args.boxes else 'labels'))
    # WHICH CROP RULE PRODUCED THESE PIXELS, in the file rather than in a shell history. The run's
    # own `[data].box_source` on the label path; the detector's own on the deployment path, which
    # is the one that can disagree with it.
    flat['__box_source__'] = np.asarray((det_boxsrc or 'keypoints') if args.detector
                                        else cfg.box_source)
    np.savez_compressed(args.out, **flat)
    print(f'wrote {args.out} ({len(results)} group(s))')

    # AFTER the prediction is on disk, never before: the cache is an optimisation for the next
    # run and a failure writing it must not lose the run that just paid for the detection.
    if args.det_cache and cache_dirty:
        args.det_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.det_cache, __stamp__=np.asarray(stamp), **det_cache)
        print(f'wrote {args.det_cache} ({len(det_cache)} group(s) of boxes)')


if __name__ == '__main__':
    main()
