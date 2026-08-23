"""The driver: one run over a dataset. Everything `run_group` is not.

Flag resolution and the refusals that must happen before a checkpoint loads, the group loop with
detection and association interleaved, and the prediction session written a block at a time. One
source session per run: `--out` names one session directory, and a session holds one calibration,
one mode and one keypoint axis.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..checkpoints import load_run, provenance
from ..dataset import LoaderConfig
from .predictions import SessionWriter, refuse_multi_session
from .window import InferConfig, run_blocks




def _box_provenance(args, det_tile, det_red, det_boxsrc):
    """Every input the detections depend on, recorded beside the numbers they produced.

    Unconditional and always complete (a provenance record has no cache-invalidation pressure),
    so two runs that look alike differ in their output file where a reader can see it. Keyed by
    the CLI name where it differs from `detect_raw`'s parameter name -- that is what a reader has
    to type to reproduce it. `tests/test_detector.py` walks `detect_raw`'s signature against this
    dict, so a new box-affecting parameter must land here too.
    """
    return {
        'detector': str(Path(args.detector).resolve()) if args.detector else '',
        'det_input_wh': str(args.det_input_wh or ''),
        'det_score': float(args.det_score),
        'det_nms_iou': float(getattr(args, 'det_nms_iou', 0.5)),
        'det_nms_center_dist': (float(args.det_nms_center_dist)
                               if getattr(args, 'det_nms_center_dist', None) is not None else 0.0),
        'det_top_k': int(args.det_top_k) if args.det_top_k else 0,
        'max_animals': int(args.max_animals),
        'max_frames': int(args.max_frames),
        # The frame range changes the detections (which frames, and the aligned lead-in).
        'frame_start': int(args.frame_start),
        'frame_stop': int(args.frame_stop),
        'tile_scale': float(det_tile) if det_tile else 0.0,
        'reduce': bool(det_red),
        # `det_box_source`, NOT `box_source`: this is the detector's training target, while
        # `[data].box_source` is the pose model's crop rule; they are allowed to disagree, which
        # is why both are recorded.
        'det_box_source': str(det_boxsrc or 'keypoints') if args.detector else '',
    }


_DET_BATCH = 16


def check_frame_range(args) -> None:
    """Refusals on `--start-frame` / `--end-frame`; pure argument arithmetic, so it fires above
    both input branches and before the checkpoint loads.

    Resolves the ONE quantity the loop reads: `args.frame_start` / `args.frame_stop`, with
    `--max-frames N` folding into `frame_stop = N`. The group-length check lives in the group loop.
    """
    if args.max_frames and (args.start_frame or args.end_frame):                     # 15
        raise SystemExit(
            f'--max-frames {args.max_frames} together with --start-frame {args.start_frame} / '
            f'--end-frame {args.end_frame} has two readings of equal force -- "N frames from the '
            'start" and "up to frame N, from the start" -- and no defensible precedence, so a '
            'rule here would make one of them silently wrong. --max-frames N IS --start-frame 0 '
            '--end-frame N; pick one spelling.')
    if args.start_frame < 0 or args.end_frame < 0:                                   # 16
        raise SystemExit(
            f'--start-frame {args.start_frame} / --end-frame {args.end_frame}: negative frame '
            'indices are not Python negative indexing here. "-1 means the last frame" and "-1 '
            'means one before the end" are both obvious and differ by one, which is the class of '
            'ambiguity that costs a day.')
    if args.end_frame and args.end_frame <= args.start_frame:                        # 17
        raise SystemExit(
            f'--end-frame {args.end_frame} is not past --start-frame {args.start_frame}. The '
            'range is half-open [start, end), so this predicts nothing; an empty range is a typo, '
            'not a protocol.')
    args.frame_start = int(args.start_frame)
    args.frame_stop = int(args.end_frame or args.max_frames or 0)


def _detector_boxes(det, det_wh, sess, gid, args, device, det_red, det_tile, n_det, n_want,
                    stats=None):
    """-> `boxes_for(store, lo, hi)`, detecting and associating on demand for one group.

    The detection cursor is not the block cursor: blocks are sized by free memory, and a
    budget-derived batch would change the input shape and move the boxes. Detection advances in
    its own globally `batch`-aligned runs and overshoots the block by up to `batch - 1` frames,
    which the buffer below holds. `associate_group` carries its state, so identity sees one
    continuous clip. Frames come from `store.read`, so the detector consumes the same decode the
    pose loop crops -- one pass over the video.
    """
    from tailcyclenet.detector import associate_group, detect_raw

    T = min(sess.groups[gid].n_frames, args.frame_stop or sess.groups[gid].n_frames)
    # One association state for the whole group: restarting every `_DET_BATCH` frames would break
    # identity every 16 frames. The cursor starts on a global `_DET_BATCH` boundary at or below
    # `frame_start`, and the lead-in is discarded: `detect_raw` asserts a slice starts on a
    # multiple of `batch`, since a short leading batch is an input SHAPE the whole-clip pass never
    # produces. So the boxes stay byte-identical to the whole-clip run's; do not relax the assert
    # or quantise `--start-frame` by 16. The cost is at most `batch - 1` extra frame-cameras, once
    # per group. Leaving the cursor at 0 would decode every frame from 0 to `frame_start` into a
    # store sized for the work.
    assoc_state, buf = {}, {}
    cursor = args.frame_start - args.frame_start % _DET_BATCH

    def boxes_for(store, lo, hi):
        nonlocal cursor
        while cursor < hi:
            end = min(cursor + _DET_BATCH, T)
            raw = detect_raw(det, det_wh, sess, gid, n_det, device=device,
                             score_thresh=args.det_score, reduce=det_red,
                             iou_thresh=getattr(args, 'det_nms_iou', 0.5),
                             center_dist_thresh=getattr(args, 'det_nms_center_dist', 0.5),
                             # `T`, the resolved STOP index -- not `args.max_frames` (0 whenever
                             # the range came in as --start-frame/--end-frame): it tells
                             # `detect_raw` where the clip ends, and its alignment assert accepts
                             # a short final slice only at that end.
                             max_frames=T, tile_scale=det_tile,
                             frames=np.arange(cursor, end),
                             read=lambda ci, cam, fr, pool=None, reduce=1: store.read(
                                 ci, cam, fr, pool=pool, reduce=reduce))
            b, s, k = associate_group(raw, sess, gid, n_want, link=args.link_boxes,
                                      min_views=args.min_views, track=args.track,
                                      max_move=args.max_move, stats=stats,
                                      pose_nms=args.pose_nms, state=assoc_state)
            for j, t in enumerate(range(cursor, end)):
                buf[t] = (b[:, j], s[:, j], None if k is None else k[:, j])
            if stats is not None:
                stats['filled'] = stats.get('filled', 0) + int(np.isfinite(b).all(-1).sum())
                stats['slots'] = stats.get('slots', 0) + int(np.isfinite(b).all(-1).size)
            cursor = end
        want = range(lo, min(hi, T))
        out = tuple(np.stack([buf[t][i] for t in want], 1) if buf[lo][i] is not None else None
                    for i in range(3))
        # The aligned lead-in below `frame_start` was detected to keep the batch partition and
        # associated so the tracker enters the range warm; it falls out here with every frame
        # below `lo`, which can never be asked for again (blocks advance monotonically).
        for t in [t for t in buf if t < lo]:
            del buf[t]
        return out

    return boxes_for


def run_dataset(args):
    """One inference run."""

    # Every pure-argument check before the checkpoint loads: a typo should not cost a multi-GB
    # load, and it makes them testable without a GPU.

    if args.anchor == 'labels':
        # An oracle prior and detector rows are incompatible: `run_group` seeds row `a` from LABEL
        # row `a`, but a detector row `a` is a score- or association-ordered slot that is not label
        # row `a` -- so the oracle would be handed a different animal's ground truth.
        if args.detector or args.boxes:
            raise SystemExit(
                '--anchor labels seeds row `a` from LABEL row `a`, but --detector/--boxes rows are '
                'score- or association-ordered and are not label rows. That is a different '
                "animal's ground truth, not an oracle. Use --anchor labels with the label crop "
                'path (no --detector, no --boxes), or --anchor carry / none with a box source.')
        print('WARNING: --anchor labels seeds the model with GROUND TRUTH. This is an oracle '
              'upper bound, not a deployment number. Label it as such wherever you quote it.')

    if args.oracle_corrupt:
        from tailcyclenet.infer import ORACLE_CORRUPTIONS
        kind = args.oracle_corrupt.split(':')[0]
        if kind not in ORACLE_CORRUPTIONS:
            raise SystemExit(f'--oracle-corrupt {args.oracle_corrupt!r}: kind must be one of '
                             f'{ORACLE_CORRUPTIONS} (off:<x> | stale:<n> | other | near | swap:<n>)')
        if kind in ('off', 'stale', 'swap') and ':' not in args.oracle_corrupt:
            example = f'{kind}:2' if kind == 'swap' else f'{kind}:0.5'
            raise SystemExit(f'--oracle-corrupt {kind} needs an amount, e.g. {example}')
        if args.anchor != 'labels':
            raise SystemExit('--oracle-corrupt breaks the ORACLE prior, so it only means anything '
                             f'under --anchor labels; got {args.anchor!r}.')
        print(f'*** DIAGNOSTIC: the oracle prior is deliberately corrupted '
              f'({args.oracle_corrupt}). This measures the echo coefficient and is not a '
              'prediction of anything. ***')

    if args.refine_px is not None:
        # `--refine` is tri-state (None = derive from mode), so only an explicit `--no-refine`
        # contradicts it.
        if args.refine is False:
            raise SystemExit('--refine-px sets the resolution of --refine\'s FIRST pass, and '
                             '--no-refine turns that pass off. Pick one.')
        # The structural floor: at patch_size 16 a 16 px input gives a 1x1 token grid and the
        # forward returns all-NaN with no exception. Two patches is the smallest input that can
        # carry a spatial relation; the measured floor is far above it.
        if args.refine_px < 32:
            raise SystemExit(f'--refine-px {args.refine_px} is below the structural floor of 32 '
                             '(about two patches). Below ~2 patches the forward returns all-NaN '
                             'with no exception. The measured floor is 96; 64 is already worse '
                             'than not refining at all.')

    # The frame range is not an input-format lever, so it is checked above the input branch,
    # before the checkpoint loads.
    check_frame_range(args)

    # The budget is resolved once, above both input branches, before anything allocates: the
    # `--videos` probe opens containers, and `memory.current` caches process-wide, so `--max-ram`
    # cannot be a ceiling on a phase that runs before it is read. Printed rather than inferred --
    # a run that does not say which budget it had cannot have its wall clock compared. Report a
    # peak as a fraction of this number.
    from tailcyclenet import memory as _memory
    _budget = _memory.current(override_gb=args.max_ram)
    print(f'ram: {_budget}')

    # The input: both branches contribute the same provenance KEYS with different values --
    # `SessionWriter` raises on a key given twice. The videos branch builds a
    # `format.VideoSession` in memory -- no staging directory, no symlink farm. Every pure
    # refusal fires above the checkpoint load and above any decode.
    if args.videos:
        from .. import adopt
        from ..format import Registry

        adopt.check_flags(args)
        reg = Registry.load(Path(args.run) / 'keypoint_registry.toml')
        vplan = adopt.plan(args.videos, args.calibration, args.cam_regex,
                          session_id=args.session_id, group_id=args.group_id)
        # `--dataset-name` does double duty here: with no staged directory tree, `ds_name` is
        # stated outright instead of recovered from the path.
        ds_name = adopt.dataset_name(reg, args.dataset_name)
        sess = adopt.build(vplan, names=reg.local_names(ds_name), units=args.units,
                           fps=args.fps, assoc_res_max_px=args.assoc_res_max_px,
                           trim=args.trim_to_shortest)
        src_prov = adopt.provenance_of(vplan)
        src_prov['source_split'] = ''
        if args.dump_session:
            adopt.dump(sess, args.dump_session)
    else:
        # One session per run, a pure-argument check above the checkpoint load: `sessions_for`
        # reads toml and opens no pixels.
        args.split = args.split or 'test'
        ds_name, sess = refuse_multi_session(args.data, args.split)
        src_prov = {'source': 'tailcyclenet infer',
                    'source_session': str(Path(args.data).resolve()),
                    'source_split': args.split,
                    'source_calibration': '', 'source_cam_regex': '',
                    'source_group_id': '', 'source_videos': []}

    device = args.device if torch.cuda.is_available() else 'cpu'
    over = ({'gridresid_offset': args.gridresid_offset} if args.gridresid_offset else None)
    model, config, registry, ckpt = load_run(args.run, args.checkpoint, device=device,
                                             model_overrides=over)
    print(f'model: {ckpt.name}  ({registry.n_keypoints} keypoints)  '
          f'query={config["model"].get("query", "prior")}  '
          f'gridresid_offset={config["model"]["gridresid_offset"]}')

    trained_frames = int(config['data'].get('n_frames', LoaderConfig.n_frames))
    # Longer than the trained window is not a knob: `n_frames` sizes the temporal pos_embed the
    # checkpoint carries. Shorter is safe -- val/test already enumerate fixed windows -- so only
    # the ceiling is guarded.
    if args.n_frames and args.n_frames > trained_frames:
        raise SystemExit(f'--n-frames {args.n_frames} exceeds the run\'s trained window '
                         f'({trained_frames}). Shorter windows are fine; longer is not the same '
                         'model.')
    trained_px = int(config['data'].get('image_size', LoaderConfig.image_size))
    # Larger than the trained input is not a knob either: `PadToSize` only pads UP, so a bigger
    # input reaches the 2D head's fixed canvas as an out-of-range position.
    if args.refine_px and args.refine_px > trained_px:
        raise SystemExit(f'--refine-px {args.refine_px} exceeds the run\'s image_size '
                         f'({trained_px}). A reduced first pass is the lever; a larger one is a '
                         'different model.')
    # The box deployment recipe is the default for a box model and inert otherwise. `--box-prompt
    # auto` resolves to `detector` when a detector/boxes file is given, never falls back to the GT
    # `labels` oracle on its own, and pulls crop_inflate -> 1.5, refine -> on, refine_px -> 128
    # unless the user set them; explicit flags always win. A box on a plain model resolves to none.
    model_is_box = config.get('model', {}).get('box_prompt', 'none') != 'none'
    box_prompt = args.box_prompt
    if box_prompt == 'auto':
        if model_is_box and (args.detector or args.boxes):
            box_prompt = 'detector'
        elif model_is_box:
            raise SystemExit(
                'this run is a box model ([model].box_prompt != "none") and --box-prompt auto '
                'must source the box from pixels, but no --detector or --boxes file was given. '
                'Pass --detector/--boxes (deployment), --box-prompt labels to EXPLICITLY use the '
                'GROUND-TRUTH oracle, or --box-prompt none to run the box model with the box '
                'withheld.')
        else:
            box_prompt = 'none'
    if box_prompt != 'none' and not model_is_box:
        print(f'--box-prompt {box_prompt}: this run is not a box model '
              '([model].box_prompt = "none"), so the box is ignored.')
        box_prompt = 'none'
    if box_prompt == 'labels':
        print('WARNING: --box-prompt labels seeds the box from GROUND TRUTH. This is an oracle '
              'upper bound, not a deployment number. Label it as such wherever you quote it.')
    box_on = box_prompt != 'none'
    crop_inflate = args.crop_inflate if args.crop_inflate is not None else (1.5 if box_on else 1.0)
    refine = args.refine if args.refine is not None else (True if box_on else None)
    refine_px = args.refine_px if args.refine_px is not None else (128 if box_on else None)
    if box_on:
        print(f'box deployment recipe: --box-prompt {box_prompt} '
              f'--crop-inflate {crop_inflate} --refine{"" if refine else " off"} '
              f'--refine-px {refine_px}')
    cfg = InferConfig(
        n_frames=args.n_frames or trained_frames,
        overlap=args.overlap, image_size=trained_px,
        min_crop_dim=int(config['data'].get('min_crop_dim', LoaderConfig.min_crop_dim)),
        box_source=config['data'].get('box_source', LoaderConfig.box_source),
        anchor=args.anchor, max_animals=args.max_animals, max_frames=args.max_frames,
        frame_start=args.frame_start, frame_stop=args.frame_stop,
        kpt_chunk=args.kpt_chunk,
        vis_thresh=args.vis_thresh, refine=refine, refine_px=refine_px,
        carry_source=args.carry_source, min_box_frames=args.min_box_frames,
        oracle_corrupt=args.oracle_corrupt, device=device,
        crop_source=args.crop_source,
        box_prompt=box_prompt, crop_inflate=crop_inflate,
        prefetch_windows=args.prefetch_windows)
    if cfg.box_source != 'keypoints':
        print(f'crops: box_source={cfg.box_source} (from the run config); a session with no '
              'instances.pq falls back to its keypoints')

    boxes = dict(np.load(args.boxes, allow_pickle=True)) if args.boxes else {}
    # Initialised here, not only inside the branch: all three reach `_box_provenance` below
    # unconditionally.
    det = det_wh = det_tile = det_boxsrc = None
    det_red = False
    if args.detector:
        from tailcyclenet.detector import load_detector
        det, det_wh, det_ds, det_mcd, det_red, det_boxsrc, det_tile, det_objq = load_detector(
            args.detector, device, input_wh=args.det_input_wh)
        # `--det-score` is not portable across detector generations, and this is the only place
        # that can tell: saturation is a property of the recipe, not the dataset. This is a
        # warning, never an automatic threshold -- the right value depends on whether coverage or
        # identity is the objective.
        if det_objq and args.det_score >= det_objq.get('q50', 0.0):
            print(f'WARNING: --det-score {args.det_score} is at or above this detector\'s MEDIAN '
                  f'objectness ({det_objq["q50"]:.4f}), so it discards at least half of the '
                  f'detections it offers. q01 {det_objq.get("q01", float("nan")):.4f} '
                  f'q10 {det_objq.get("q10", float("nan")):.4f} '
                  f'q90 {det_objq.get("q90", float("nan")):.4f}. This detector is NOT saturated; '
                  'sweep the threshold -- 0.97 maximises identity, 0.5 coverage.', flush=True)
        # A tile-trained detector deploys on the whole frame at its training scale, and `det_wh`
        # is its TILE size: `detect_group` derives the per-camera input from `det_tile`, so
        # `det_wh` is only a fallback here.
        print(f'detector: {args.detector} ({det_wh[0]}x{det_wh[1]}'
              + (f' TILE at scale {det_tile:g}, whole-frame input derived per camera'
                 if det_tile else '')
              + f', trained on {det_ds!r}, boxes={det_boxsrc or "keypoints"})')
        # The detector regresses the crop rule's box, so its floor must be the pose model's floor.
        if det_mcd != cfg.min_crop_dim:
            raise SystemExit(
                f'{args.detector}: trained at min_crop_dim={det_mcd}, but this run\'s '
                f'[data].min_crop_dim is {cfg.min_crop_dim}. The detector reproduces the crop '
                'rule; two floors are two rules.')
        # The other half of the same contract, and not a hard failure: a detector trained on
        # `instances` boxes is a legitimate arm, just not a detector-quality comparison against a
        # keypoint-trained one, because the crop source moved too. The npz records which.
        # A detector with no keypoint branch cannot serve `--crop-source keypoints`, and the
        # failure is SILENT (`run_group` takes `det_kpts_stc is not None` as the switch and would
        # quietly crop from the boxes); refused rather than warned.
        if args.crop_source == 'keypoints' and not int(getattr(det, 'n_keypoints', 0)):
            raise SystemExit(
                f'{args.detector}: has no keypoint branch, so --crop-source keypoints would '
                'silently fall back to cropping from the boxes and measure the arm it is being '
                'compared against. Train the detector with --keypoints, or use --crop-source '
                'boxes.')
        if (det_boxsrc or 'keypoints') != cfg.box_source:
            print(f'WARNING: detector boxes are {det_boxsrc!r} but this run was trained on '
                  f'{cfg.box_source!r} crops. Two crop sources are two crop rules -- do not read '
                  'a delta against a run whose detector matched as a detector-quality result.')
    # A registry is keyed by DATASET NAME, so deploying a run on a root it was not trained on --
    # the point of a shared keypoint vocabulary -- dies on the folder name alone without this.
    # Safe to override because it is CHECKED: `Registry.ids_for` aligns to the session's own
    # names and raises on any name the registry does not hold.
    if args.dataset_name and not args.videos:
        # The videos branch already consumed it, in `adopt.dataset_name`.
        print(f'registry: reading session keypoints as {args.dataset_name!r}, not {ds_name!r}')
        ds_name = args.dataset_name
    # The budget, re-resolved now that the weights are resident: `--max-ram` names the PROCESS,
    # and nothing so far has subtracted what torch, CUDA and the checkpoints already hold, so the
    # buffer shares could exceed the ceiling. Everything that sizes a buffer runs after this line.
    _budget = _memory.rebudget(override_gb=args.max_ram)
    print(f'ram: {_budget}')

    want = set(args.groups.split(',')) if args.groups else None

    # One session per run, checked before the loop and above the checkpoint load.
    gids = [g for g in sess.groups if not want or g in want]
    # A group shorter than `--start-frame` is skipped by name (a ragged root must still be
    # runnable), but a run that predicts NOTHING is a mistyped range, and an empty session with
    # exit 0 is the worst of both.
    if cfg.frame_start:
        short = [g for g in gids if sess.groups[g].n_frames <= cfg.frame_start]
        for g in short:
            print(f'{sess.session_id}/{g}: {sess.groups[g].n_frames} frames, which is at or below '
                  f'--start-frame {cfg.frame_start}; skipped.')
        gids = [g for g in gids if g not in set(short)]
        if not gids:
            raise SystemExit(
                f'--start-frame {cfg.frame_start} is past the end of every group '
                f'({ {g: sess.groups[g].n_frames for g in short} }), so this run would predict '
                'nothing. An empty session written with exit 0 is worse than a refusal.')
    writer = SessionWriter(args.out, sess, registry,
                           # A LIST OF PAIRS, not a dict: `SessionWriter` raises on a duplicate
                           # key, where a dict would silently keep the last one.
                           [*src_prov.items(),
                            *{'source_session_id': sess.session_id,
                            'source_dataset': ds_name,
                            # Absolute, like `source_session`: a relative path names nothing
                            # later. `checkpoint` is the resolved FILE -- the name alone does not
                            # say which folder it came from.
                            'run': str(Path(args.run).resolve()),
                            'checkpoint': str(Path(ckpt).resolve()),
                            'checkpoint_name': ckpt.name,
                            'anchor': cfg.anchor, 'carry_source': cfg.carry_source,
                            'n_frames': cfg.n_frames, 'overlap': cfg.overlap,
                            'frame_start': cfg.frame_start, 'frame_stop': cfg.frame_stop,
                            'refine': bool(cfg.refine), 'refine_px': cfg.refine_px or 0,
                            'crop_source': cfg.crop_source,
                            'boxes': (str(args.detector) if args.detector else
                                      (str(args.boxes) if args.boxes else 'labels')),
                            'box_source': ((det_boxsrc or 'keypoints') if args.detector
                                           else cfg.box_source),
                            'vis_thresh': float(cfg.vis_thresh) if cfg.vis_thresh else 0.0,
                            # The commit and the dirty flag: a config is not a provenance record.
                            **provenance()}.items(),
                            *_box_provenance(args, det_tile, det_red, det_boxsrc).items()],
                           gids)
    sess.preload()
    try:
        for gid in gids:
            key = f'{sess.session_id}/{gid}'
            # Initialised outside the branch, for the reason the comment above `det_tile` gives:
            # every non-detector box source reaches the `run_group` call below.
            boxes_for, det_stats, n_want = None, {}, 0
            if det is not None:
                # Default to what the session actually holds, not to 1: `associate_group` caps
                # rows at this count.
                n_want = args.max_animals or max(1, len(sess.labels(gid).animal_ids))
                # Detect at `top_k`, associate at `n_want` -- one number until the split, so a
                # row-count sweep no longer moves the detection budget.
                n_det = args.det_top_k or n_want
                print(f'{key}: detecting up to {n_det} per camera, {n_want} animal row(s)'
                      f'{"" if args.max_animals else " (from the labels; set --max-animals)"}',
                      flush=True)
                boxes_for = _detector_boxes(
                    det, det_wh, sess, gid, args, device, det_red, det_tile, n_det, n_want,
                    stats=det_stats)
            # A missing `--boxes` key is not an absent argument: `boxes.get(key)` would silently
            # fall back to cropping from the labels.
            if args.boxes and key not in boxes:
                raise SystemExit(
                    f'{args.boxes}: no entry for {key!r}. Falling back to the labels here would '
                    'quietly turn this into the GT-crop upper bound. Keys present: '
                    f'{sorted(k for k in boxes if not k.startswith("__"))[:5]} ...')
            # STREAMED: each block is written as it finishes, so nothing here is proportional
            # to the length of the clip.
            # `f0` is a SOURCE frame index, opening at `frame_start`; redundant by construction
            # (`f0 == blk['window_start'][0]`), which is asserted rather than trusted.
            f0, w0 = cfg.frame_start, 0
            n_frames = n_fin = n_pt = 0
            _t_group, _stats = time.time(), {}
            for blk in run_blocks(model, sess, gid, registry, ds_name, cfg,
                                  box_points=boxes.get(key), boxes_for=boxes_for, n_rows=n_want,
                                  stats=_stats):
                assert f0 == int(blk['window_start'][0]), (f0, int(blk['window_start'][0]))
                writer.write_block(gid, blk, f0, w0)
                f0 += blk['pred'].shape[1]
                w0 += blk['outcome'].shape[1]
                n_frames += blk['pred'].shape[1]
                n_fin += int(np.isfinite(blk['pred']).all(-1).sum())
                n_pt += int(np.isfinite(blk['pred']).all(-1).size)
            span = ('' if not (cfg.frame_start or cfg.frame_stop)
                    else f' [{cfg.frame_start}, {f0})')
            print(f'{key}: {n_frames} frames{span}, {n_fin / max(n_pt, 1):.3f} finite')
            # Decode's share, measured by this run: `decode_s` sums across threads, so it can
            # exceed the elapsed time; printed beside the wall clock rather than as a percentage.
            # The store's hit rate is the other half.
            _wall = time.time() - _t_group
            _dec, _h, _m = (_stats.get('decode_s', 0.0), _stats.get('decode_hits', 0),
                            _stats.get('decode_misses', 0))
            print(f'{key}: {_wall:.1f}s wall, {_dec:.1f}s of decode work across threads '
                  f'({_dec / max(_wall, 1e-9):.2f}x wall), store {_h}/{_h + _m} hits '
                  f'({_h / max(_h + _m, 1):.2f})')
            if det is not None:
                # Reported after the run, because detection happens block by block inside it: the
                # fire rate cannot be recovered from the output afterwards, so it is printed, not
                # derived.
                if args.pose_nms is not None:
                    # `.get(..., 0)`, both keys: `identity.pose_nms` returns before writing either
                    # when the detector has no keypoint branch.
                    print(f'{key}: pose-nms dropped {det_stats.get("nms_dropped", 0)} row(s) of '
                          f'{det_stats.get("nms_pairs", 0)} overlapping pair(s)'
                          + (' (no keypoint branch -- pose-nms is a no-op)'
                             if 'nms_pairs' not in det_stats else ''), flush=True)
                # How much the threshold left, accumulated over the blocks: a detector whose
                # scores are not saturated loses most of its boxes to a higher threshold, and this
                # is where it shows.
                _sl = det_stats.get('slots', 0)
                if _sl:
                    filled = det_stats.get('filled', 0) / _sl
                    print(f'{key}: boxes in {filled:.3f} of (animal, frame, camera) slots'
                          f'{"   <-- LOW: try a smaller --det-score" if filled < 0.25 else ""}')

    finally:
        writer.close()
    print(f'wrote {args.out} ({len(gids)} group(s))')

