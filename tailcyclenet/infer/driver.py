"""The DRIVER: one run over a dataset. Everything `run_group` is not.

Flag resolution and the refusals that must happen before a checkpoint loads, the group loop,
detection and association interleaved with it, and the prediction session written a block at a
time.

ONE SOURCE SESSION PER RUN, because `--out` names one session directory and a session holds one
calibration, one mode and one keypoint axis. `--data` therefore takes a session directory, which
`format.sessions_for` has always accepted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..checkpoints import load_run, provenance
from ..dataset import LoaderConfig
from .predictions import SessionWriter, refuse_multi_session
from .window import InferConfig, run_blocks




def _box_provenance(args, det_tile, det_red, det_boxsrc):
    """Every input the DETECTIONS depend on, recorded beside the numbers they produced.

    This is what is left of the `--det-cache` stamp, and it does a different job. The stamp
    compared two runs so a cache could be REFUSED; there is no cache, so nothing needs comparing --
    but "which detector, at which threshold, at which input size" is still the difference between
    two numbers that look alike (eval rule 4), and it belonged in the output file rather than in a
    shell history.

    UNCONDITIONAL, every key, always. The stamp recorded only the options that DIFFERED from their
    defaults, because a positional list invalidated every cache on disk each time a flag was added.
    A provenance record has no such pressure, and conditional membership is what made the stamp
    need five exceptions to its own rule.

    Keyed by the CLI name where it differs from `detect_raw`'s parameter name, because that is what
    a reader has to type to reproduce it. `tests/test_detector.py` walks `detect_raw`'s signature
    against this dict, so a new box-affecting parameter cannot be added without landing here.
    """
    return {
        'detector': str(Path(args.detector).resolve()) if args.detector else '',
        'det_input_wh': str(args.det_input_wh or ''),
        'det_score': float(args.det_score),
        'det_top_k': int(args.det_top_k) if args.det_top_k else 0,
        'max_animals': int(args.max_animals),
        'max_frames': int(args.max_frames),
        # THE FRAME RANGE CHANGES THE DETECTIONS -- which frames, and the aligned lead-in -- so
        # two runs differing in it must not be indistinguishable from their output.
        'frame_start': int(args.frame_start),
        'frame_stop': int(args.frame_stop),
        'tile_scale': float(det_tile) if det_tile else 0.0,
        'reduce': bool(det_red),
        # `det_box_source`, NOT `box_source`. They are two different facts and the names collided:
        # this one is the DETECTOR'S TRAINING TARGET (what its boxes are boxes OF), while the
        # run's own `[data].box_source` is the crop rule the pose model was trained on. They are
        # allowed to disagree -- the best rat-city detector is `instances`-trained while every
        # rat-city pose run is keypoint-trained -- which is exactly why both are recorded. Under
        # one name the splat order silently decided which survived.
        'det_box_source': str(det_boxsrc or 'keypoints') if args.detector else '',
    }


_DET_BATCH = 16


def check_frame_range(args) -> None:
    """Refusals 15-17 on `--start-frame` / `--end-frame`. PURE ARGUMENT ARITHMETIC, so it fires
    above both input branches and before the checkpoint loads.

    Resolves the ONE quantity the loop reads -- `args.frame_start` / `args.frame_stop`, with
    `--max-frames N` folding into `frame_stop = N`. Refusal 18 (a group shorter than the start)
    needs the group lengths and lives in the group loop.
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

    **THE DETECTION CURSOR IS NOT THE BLOCK CURSOR, and that is the whole design.** Blocks are
    sized by free memory, so if detection re-batched per block, every boundary would produce a
    short leading batch -- a different input SHAPE, and cuDNN picks convolution algorithms per
    shape (0.204 px of box, 1.69e-03 of objectness; dev/reports/38 §3.2). A budget-derived block
    size would then move the boxes, which is exactly what the rule "every knob the budget sizes is
    output-neutral" forbids. So detection advances in its OWN globally `batch`-aligned runs and
    overshoots the block by up to `batch - 1` frames, which the buffer below holds.

    `associate_group` carries its state, so the tracker and `link_rows` see one continuous clip
    however the frames were divided up.

    Frames come from `store.read`, so the detector consumes the same decode the pose loop is about
    to crop -- which is the other half of "one pass over the video".
    """
    from tailcyclenet.detector import associate_group, detect_raw

    T = min(sess.groups[gid].n_frames, args.frame_stop or sess.groups[gid].n_frames)
    # ONE association state for the whole group. Without it the tracker and `link_rows` would
    # restart every `_DET_BATCH` frames -- an identity break every 16 frames, which is far worse
    # than the block-boundary break `state=` was added for.
    #
    # **THE CURSOR STARTS ON A GLOBAL `_DET_BATCH` BOUNDARY AT OR BELOW `frame_start`, AND THE
    # LEAD-IN IS DISCARDED.** `detect_raw` ASSERTS that a slice starts on a multiple of `batch`,
    # because `_units` partitions on `range(0, T, batch)` and a short leading batch is an input
    # SHAPE the whole-clip pass never produces -- cuDNN picks convolution algorithms per shape,
    # worth 0.204 px of box and 1.7e-03 of objectness. So the boxes stay BYTE-IDENTICAL to the
    # whole-clip run's. Do NOT relax that assert, and do NOT round `--start-frame` to a multiple
    # of 16: the batch is an internal that no CLI value may be quantised by.
    #
    # It costs at most `batch - 1` extra frame-cameras of decode and detection, ONCE PER GROUP and
    # not once per block -- the cursor is created with this closure and persists across every
    # block, so the price does not scale with the clip.
    #
    # LEAVING IT AT 0 IS NOT MERELY WASTEFUL, IT IS A MEMORY FAILURE: the loop below reads through
    # `store.read`, so a cursor at 0 would decode every frame from 0 to `frame_start` into the
    # frame store -- and the store is sized from the WORK, about two windows, so a 300-frame
    # prefix is the block sizing being wrong by two orders of magnitude on every group's first
    # block.
    assoc_state, buf = {}, {}
    cursor = args.frame_start - args.frame_start % _DET_BATCH

    def boxes_for(store, lo, hi):
        nonlocal cursor
        while cursor < hi:
            end = min(cursor + _DET_BATCH, T)
            raw = detect_raw(det, det_wh, sess, gid, n_det, device=device,
                             score_thresh=args.det_score, reduce=det_red,
                             max_frames=args.max_frames, tile_scale=det_tile,
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
        # The aligned lead-in below `frame_start` was detected (it had to be, to keep the batch
        # partition), associated (so the tracker enters the range warm rather than cold), and is
        # written to nothing: it falls out here with every other frame below `lo`.
        # Frames below `lo` can never be asked for again -- blocks advance monotonically, exactly
        # as the frame store's own eviction argument goes.
        for t in [t for t in buf if t < lo]:
            del buf[t]
        return out

    return boxes_for


def run_dataset(args):
    """One inference run."""

    # EVERY PURE-ARGUMENT CHECK BEFORE THE CHECKPOINT LOADS. These cost nothing and a typo
    # should not cost a 5.6 GB load; it also makes them testable without a GPU.

    if args.anchor == 'labels':
        # AN ORACLE PRIOR AND DETECTOR ROWS ARE INCOMPATIBLE, not merely a bad idea. `run_group`
        # seeds row `a` from LABEL row `a`, and once boxes come from a detector row `a` is a
        # score-ordered or association-ordered slot that is not label row `a` for any `a` -- so the
        # arm that exists to be an upper bound was being handed a different animal's ground truth,
        # and read as a poor oracle rather than as a broken one.
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
        # `--refine` is tri-state now (None = derive from the session mode), so only an EXPLICIT
        # `--no-refine` is a contradiction. Leaving it unset is fine and means "wherever refine runs".
        if args.refine is False:
            raise SystemExit('--refine-px sets the resolution of --refine\'s FIRST pass, and '
                             '--no-refine turns that pass off. Pick one.')
        # THE STRUCTURAL FLOOR, and it is the spatial analogue of gotcha 1. At `patch_size` 16 a
        # 16 px input gives a 1x1 token grid and the forward returns ALL-NaN with no exception,
        # while `run_group` still marks every window `ok` -- only `coverage 0.0000` reveals it.
        # 8 px raises. Two patches is the smallest input that can carry a spatial relation at all;
        # the MEASURED floor is far above it (64 px is 3.2x worse than not refining, on both roots).
        if args.refine_px < 32:
            raise SystemExit(f'--refine-px {args.refine_px} is below the structural floor of 32 '
                             '(about two patches). Below ~2 patches the forward returns all-NaN '
                             'with no exception. The measured floor is 96; 64 is already worse '
                             'than not refining at all.')

    # THE FRAME RANGE IS NOT AN INPUT-FORMAT LEVER, so it is checked above the input branch:
    # refusals 15-17 are pure argument arithmetic and must fire before the checkpoint loads.
    check_frame_range(args)

    # THE BUDGET IS RESOLVED ONCE, HERE, ABOVE BOTH INPUT BRANCHES, AND BEFORE ANYTHING ALLOCATES.
    #
    # It sat BELOW the input branch until the `--videos` path put a real consumer above it: the
    # probe opens video containers, and `memory.current` caches process-wide, so during the probe
    # the override was not yet in effect and any consumer that asked would have got the HOST
    # figure. `--max-ram` cannot be a ceiling on a phase that runs before it is read.
    #
    # Printed rather than inferred: it depends on machine state, so a run that does not say which
    # budget it had cannot have its wall clock compared against another run's. Report a peak as a
    # FRACTION of this number -- an absolute peak on an unconstrained host is usually retained
    # allocator arena rather than the working set.
    from tailcyclenet import memory as _memory
    _budget = _memory.current(override_gb=args.max_ram)
    print(f'ram: {_budget}')

    # THE INPUT. Both branches contribute the same PROVENANCE KEYS with different values, which is
    # what keeps them honest: `SessionWriter` raises on a key given twice with two values.
    #
    # The videos branch builds a `format.VideoSession` IN MEMORY -- no staging directory, no
    # symlink farm, no second copy of the user's filenames on disk that can go stale. Every pure
    # refusal fires above the checkpoint load AND above any decode; `Registry.load` is a toml read,
    # so even the keypoint axis is settled before the weights.
    if args.videos:
        from .. import adopt
        from ..format import Registry

        adopt.check_flags(args)
        reg = Registry.load(Path(args.run) / 'keypoint_registry.toml')
        vplan = adopt.plan(args.videos, args.calibration, args.cam_regex,
                          session_id=args.session_id, group_id=args.group_id)
        # `--dataset-name` DOES DOUBLE DUTY HERE, and this also removes the directory-nesting
        # trick the on-disk path needs: with no staged `<stage>/<dataset>/test/<session>/`,
        # `ds_name` is STATED outright instead of recovered from `path.parent.parent.name`.
        ds_name = adopt.dataset_name(reg, args.dataset_name)
        sess = adopt.build(vplan, names=reg.local_names(ds_name), units=args.units,
                           fps=args.fps, assoc_res_max_px=args.assoc_res_max_px,
                           trim=args.trim_to_shortest)
        src_prov = adopt.provenance_of(vplan)
        src_prov['source_split'] = ''
        if args.dump_session:
            adopt.dump(sess, args.dump_session)
    else:
        # ONE SESSION PER RUN, AND IT IS A PURE-ARGUMENT CHECK -- so it belongs up here with the
        # others, above the checkpoint load. `sessions_for` reads toml and opens no pixels, so
        # this costs nothing and a mistyped `--data` does not cost a 5.6 GB load.
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
    # LONGER THAN THE TRAINED WINDOW IS NOT A KNOB. `n_frames` sizes the temporal pos_embed the
    # checkpoint carries; asking for more frames than it was built for interpolates at best and
    # shape-errors deep inside the encoder at worst. Shorter is safe -- val/test already enumerate
    # fixed windows -- so only the ceiling is guarded.
    if args.n_frames and args.n_frames > trained_frames:
        raise SystemExit(f'--n-frames {args.n_frames} exceeds the run\'s trained window '
                         f'({trained_frames}). Shorter windows are fine; longer is not the same '
                         'model.')
    trained_px = int(config['data'].get('image_size', LoaderConfig.image_size))
    # LARGER THAN THE TRAINED INPUT IS NOT A KNOB EITHER, for a different reason: `PadToSize` only
    # ever pads UP, so a bigger input is not padded, not resized, and reaches the 2D head's fixed
    # `image_size`-wide canvas as an out-of-range position. Smaller is what the compensations in
    # `model._input_extent` exist for.
    if args.refine_px and args.refine_px > trained_px:
        raise SystemExit(f'--refine-px {args.refine_px} exceeds the run\'s image_size '
                         f'({trained_px}). A reduced first pass is the lever; a larger one is a '
                         'different model.')
    # THE BOX DEPLOYMENT RECIPE (report 27) is the DEFAULT FOR A BOX MODEL and inert otherwise, so
    # a plain run is unchanged and a box run with a detector needs no flags. `--box-prompt auto`
    # resolves to `detector` when a detector/boxes file is given. It NEVER falls back to the GT
    # `labels` oracle on its own: a box model with no detector/boxes is an error, because silently
    # seeding a box from ground truth is eval rule 7's failure mode. `--box-prompt labels` is the
    # explicit opt-in for the oracle and warns. Auto also pulls crop_inflate -> 1.5, refine -> on,
    # refine_px -> 128 unless the user set them. Explicit flags always win. A box on a plain model
    # resolves to `none`.
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
        print(f'box deployment recipe (report 27): --box-prompt {box_prompt} '
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
    # `det_tile`, `det_red` and `det_boxsrc` are initialised HERE and not only inside the branch:
    # all three reach `_box_provenance` below unconditionally, and a NameError there would only
    # fire on the box-source paths that have no detector at all -- which is exactly how `det_kpts`
    # shipped broken.
    det = det_wh = det_tile = det_boxsrc = None
    det_red = False
    if args.detector:
        from tailcyclenet.detector import load_detector
        det, det_wh, det_ds, det_mcd, det_red, det_boxsrc, det_tile, det_objq = load_detector(
            args.detector, device, input_wh=args.det_input_wh)
        # `--det-score` IS NOT PORTABLE ACROSS DETECTOR GENERATIONS, and this is the only place
        # that can tell. Saturation is a property of the RECIPE, not the dataset: 0.99 was
        # measured on detectors whose objectness is saturated, but a tiled/masked one reads q01
        # 0.45-0.84 and loses two thirds of its detections to the same number -- coverage 0.703
        # against 0.986 at 0.50 (dev/reports/21 0b). The DEFAULT is now 0.5 (coverage-favouring)
        # and 0.97 maximises identity; this stays a warning rather than a refusal so the caller
        # can choose, and never an automatic threshold, because the right value depends on which
        # of coverage or identity is the objective.
        if det_objq and args.det_score >= det_objq.get('q50', 0.0):
            print(f'WARNING: --det-score {args.det_score} is at or above this detector\'s MEDIAN '
                  f'objectness ({det_objq["q50"]:.4f}), so it discards at least half of the '
                  f'detections it offers. q01 {det_objq.get("q01", float("nan")):.4f} '
                  f'q10 {det_objq.get("q10", float("nan")):.4f} '
                  f'q90 {det_objq.get("q90", float("nan")):.4f}. This detector is NOT saturated; '
                  'sweep the threshold -- 0.97 maximises identity, 0.5 coverage.', flush=True)
        # A TILE-TRAINED DETECTOR IS DEPLOYED ON THE WHOLE FRAME AT ITS TRAINING SCALE, and
        # `det_wh` is its TILE size. `detect_group` derives the per-camera input from `det_tile`
        # (frame sizes vary WITHIN a root -- rat-city-annotated ships 4696x2048 beside 4500x2050),
        # so `det_wh` is only a fallback here and is printed as the tile it is.
        print(f'detector: {args.detector} ({det_wh[0]}x{det_wh[1]}'
              + (f' TILE at scale {det_tile:g}, whole-frame input derived per camera'
                 if det_tile else '')
              + f', trained on {det_ds!r}, boxes={det_boxsrc or "keypoints"})')
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
        # A DETECTOR WITH NO KEYPOINT BRANCH CANNOT SERVE `--crop-source keypoints`, and the failure
        # is SILENT: `run_group` takes `det_kpts_stc is not None` as the switch, so a None does not
        # error -- it quietly crops from the boxes and reports the arm under the other arm's name.
        # Refused rather than warned, for the reason eval rule 4 exists: the two crop sources are
        # only comparable when they differ in exactly one lever.
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
    # the whole point of a shared keypoint vocabulary -- otherwise dies on the folder name alone.
    # `rat-city` and `rat-city-annotated` are the shipped instance: identical `names`, therefore
    # identical registry ids and the same embedding rows, and different directories. This is safe
    # to override because it is CHECKED rather than assumed: `Registry.ids_for` aligns to the
    # session's own `names` and raises on any name the registry does not hold, which is gotcha 4's
    # guard and the reason a per-dataset id vector is never applied positionally.
    if args.dataset_name and not args.videos:
        # The videos branch already consumed it, in `adopt.dataset_name`.
        print(f'registry: reading session keypoints as {args.dataset_name!r}, not {ds_name!r}')
        ds_name = args.dataset_name
    want = set(args.groups.split(',')) if args.groups else None

    # ONE SESSION PER RUN. Checked before the loop, and above the checkpoint load -- see the
    # `sessions_for` call, which reads toml and opens no pixels.
    gids = [g for g in sess.groups if not want or g in want]
    # REFUSAL 18: a group shorter than `--start-frame` is SKIPPED BY NAME, not refused -- a ragged
    # root with one short group must still be runnable. But a run that predicts NOTHING is a
    # mistyped range, and writing an empty session and exiting 0 is the worst of both.
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
                            # ABSOLUTE, like `source_session`: a relative path recorded from
                            # whatever directory the run happened to start in names nothing later.
                            # `checkpoint` is the resolved FILE, not just its name -- `_last` and
                            # `_best` differ by up to 13,600 iterations and the name alone does not
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
                            # THE COMMIT AND THE DIRTY FLAG. A config is not a
                            # provenance record -- gotcha 12 is what that cost.
                            **provenance()}.items(),
                            *_box_provenance(args, det_tile, det_red, det_boxsrc).items()],
                           gids)
    sess.preload()
    try:
        for gid in gids:
            key = f'{sess.session_id}/{gid}'
            # INITIALISED OUTSIDE THE BRANCH, for the reason the comment above `det_tile` gives:
            # every box source that is NOT a detector -- the GT-crop upper bound and the whole
            # `--boxes` path -- reaches the `run_group` call below, and binding this inside the
            # branch is how `det_kpts` once raised `UnboundLocalError` after paying for a
            # checkpoint load.
            boxes_for, det_stats, n_want = None, {}, 0
            if det is not None:
                # Default to what the session actually holds, not to 1. `associate_group` caps
                # rows at this count, so a bare --detector run on a ten-animal dataset
                # used to return one animal per frame and read as a catastrophic miss rate
                # rather than as a missing flag.
                n_want = args.max_animals or max(1, len(sess.labels(gid).animal_ids))
                # DETECT AT `top_k`, ASSOCIATE AT `n_want`. They were one number until the split,
                # and welding them meant a sweep over the row count also moved the detection budget
                # -- which is why `link_rows`' spare-rows finding could not be run end to end.
                n_det = args.det_top_k or n_want
                print(f'{key}: detecting up to {n_det} per camera, {n_want} animal row(s)'
                      f'{"" if args.max_animals else " (from the labels; set --max-animals)"}',
                      flush=True)
                boxes_for = _detector_boxes(
                    det, det_wh, sess, gid, args, device, det_red, det_tile, n_det, n_want,
                    stats=det_stats)
            # A MISSING `--boxes` KEY IS NOT AN ABSENT ARGUMENT. `boxes.get(key)` returning None
            # silently falls back to cropping from the LABELS, so a run whose keys did not match --
            # a different session naming, a stale npz, a typo in one group -- reported the GT-crop
            # oracle under a heading that said otherwise. Nothing in the output said which.
            if args.boxes and key not in boxes:
                raise SystemExit(
                    f'{args.boxes}: no entry for {key!r}. Falling back to the labels here would '
                    'quietly turn this into the GT-crop upper bound. Keys present: '
                    f'{sorted(k for k in boxes if not k.startswith("__"))[:5]} ...')
            # STREAMED: each block is written as it finishes, so nothing here is proportional
            # to the length of the clip.
            # `f0` IS A SOURCE FRAME INDEX, so it opens at `frame_start` rather than at 0 --
            # `write_block` then records source indices with no signature change. It is also
            # redundant by construction (`f0 == blk['window_start'][0]`), which is asserted rather
            # than trusted: a mis-set accumulator is exactly how a ranged prediction would come
            # back scattered against the wrong frames.
            f0, w0 = cfg.frame_start, 0
            n_frames = n_fin = n_pt = 0
            for blk in run_blocks(model, sess, gid, registry, ds_name, cfg,
                                  box_points=boxes.get(key), boxes_for=boxes_for, n_rows=n_want):
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
            if det is not None:
                # REPORTED AFTER THE RUN, because detection now happens block by block INSIDE it.
                # The fire rate is what a rate-matched random control has to be matched TO and
                # cannot be recovered from the output afterwards, so it is printed, not derived.
                if args.pose_nms is not None:
                    # `.get(..., 0)`, both keys: `identity.pose_nms` returns before writing EITHER
                    # when the detector has no keypoint branch -- the normal case for a 2D root,
                    # so an empty `det_stats` is not a bug signal.
                    print(f'{key}: pose-nms dropped {det_stats.get("nms_dropped", 0)} row(s) of '
                          f'{det_stats.get("nms_pairs", 0)} overlapping pair(s)'
                          + (' (no keypoint branch -- pose-nms is a no-op)'
                             if 'nms_pairs' not in det_stats else ''), flush=True)
                # HOW MUCH THE THRESHOLD LEFT, accumulated over the blocks rather than measured on
                # one whole-clip array. `--det-score` defaults to 0.5 (coverage-favouring); a
                # detector whose scores are NOT saturated loses most of its boxes to a higher
                # threshold, and this is where that shows up rather than downstream as an
                # unexplained miss rate.
                _sl = det_stats.get('slots', 0)
                if _sl:
                    filled = det_stats.get('filled', 0) / _sl
                    print(f'{key}: boxes in {filled:.3f} of (animal, frame, camera) slots'
                          f'{"   <-- LOW: try a smaller --det-score" if filled < 0.25 else ""}')

    finally:
        writer.close()
    print(f'wrote {args.out} ({len(gids)} group(s))')

