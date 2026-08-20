"""The DRIVER: one run over a dataset. Everything `run_group` is not.

Flag resolution and the refusals that must happen before a checkpoint loads, the session/group
loop, the `--det-cache` stamp and its read/checkpoint/write, association, the render pool, and the
flat-npz write. Lifted verbatim out of `scripts/infer.py`, where it could not be imported and so
could only be tested by reading the file as text.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..checkpoints import load_run
from ..dataset import LoaderConfig
from ..format import sessions_for
from .window import InferConfig, run_group




def run_dataset(args, ap):
    """One inference run. `ap` is the parser `args` came from -- the `--det-cache` stamp records
    only the options that DIFFER from their defaults and asks `ap.get_default` for them."""

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

    # THE BUDGET IS RESOLVED ONCE, HERE, BEFORE ANYTHING ALLOCATES, and printed rather than
    # inferred: it depends on machine state, so a run that does not say which budget it had cannot
    # have its wall clock compared against another run's. Report a peak as a FRACTION of this
    # number -- an absolute peak on an unconstrained host is retained allocator arena, not the
    # working set, and says almost nothing (this file's plan, dev/plans/...ram_budget.md).
    from tailcyclenet import memory as _memory
    _budget = _memory.current(override_gb=args.max_ram)
    print(f'ram: {_budget}')

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
    # `det_tile` and `det_red` are initialised HERE and not only inside the branch: both go into
    # the cache stamp unconditionally, and a NameError there would only fire on the box-source
    # paths that have no detector at all -- which is exactly how `det_kpts` shipped broken.
    det = det_wh = det_tile = None
    det_red = False
    if args.detector:
        from tailcyclenet.detector import associate_group, detect_raw, load_detector
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
        if (det_boxsrc or 'keypoints') != cfg.box_source:
            print(f'WARNING: detector boxes are {det_boxsrc!r} but this run was trained on '
                  f'{cfg.box_source!r} crops. Two crop sources are two crop rules -- do not read '
                  'a delta against a run whose detector matched as a detector-quality result.')
    ds_name, sessions = sessions_for(args.data, args.split)
    # A registry is keyed by DATASET NAME, so deploying a run on a root it was not trained on --
    # the whole point of a shared keypoint vocabulary -- otherwise dies on the folder name alone.
    # `rat-city` and `rat-city-annotated` are the shipped instance: identical `names`, therefore
    # identical registry ids and the same embedding rows, and different directories. This is safe
    # to override because it is CHECKED rather than assumed: `Registry.ids_for` aligns to the
    # session's own `names` and raises on any name the registry does not hold, which is gotcha 4's
    # guard and the reason a per-dataset id vector is never applied positionally.
    if args.dataset_name:
        print(f'registry: reading session keypoints as {args.dataset_name!r}, not {ds_name!r}')
        ds_name = args.dataset_name
    want = set(args.groups.split(',')) if args.groups else None
    render_cams = [int(c) for c in args.render_cams.split(',') if c.strip() != '']

    # The box cache. `stamp` is every input the boxes DEPEND ON: reusing a cache written under a
    # different detector, threshold or animal count would quietly make one arm incomparable to the
    # next (eval rule 4).
    #
    # ONLY THE OPTIONS THAT DIFFER FROM THEIR DEFAULTS, sorted by name -- a positional list of every
    # value invalidated every cache on disk each time a flag was added.
    #
    # THE UNCONDITIONAL KEYS ARE THE EXCEPTION THAT RULE NEEDS: anything whose default may MOVE, or
    # that comes from the CHECKPOINT rather than the command line, must be stamped always, or a
    # cache written under the old meaning is reused silently under the new one. That covers
    # `det_score`, `top_k`, `tile_scale`, `reduce` and `raw_rev`. `raw_rev` is the sharpest: a raw
    # cache and a pre-split associated one share shape, dtype and key names, so an old one read as
    # raw would be associated a SECOND time.
    #
    # THE CACHE HOLDS RAW DETECTIONS, SO THE ASSOCIATION OPTIONS LEAVE THE STAMP. `track`,
    # `link_boxes`, `max_animals`, `min_views` and `max_move` change only what happens after
    # detection, and `associate_group` re-runs every invocation -- microseconds per frame against
    # 44 ms of 4K decode. One cache serves every identity arm, matching them BY CONSTRUCTION.
    #
    # `top_k` is always present and its VALUE says which rule produced it (`--det-top-k` when set,
    # else the count `max_animals` implies). Conditional membership is what makes a stamp lie.
    from tailcyclenet.detector import RAW_REV
    top_k_stamp = (str(args.det_top_k) if args.det_top_k
                   else f'from-max-animals:{args.max_animals}')
    stamp = repr(sorted([('det_score', str(args.det_score)), ('raw_rev', str(RAW_REV)),
                         ('top_k', top_k_stamp),
                         ('tile_scale', str(det_tile)), ('reduce', str(det_red))]
                        + [(k, str(getattr(args, k))) for k in
                           ('detector', 'det_input_wh', 'max_frames')
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
        print(f'detector boxes: '
              f'{sum(1 for k in det_cache if not k.endswith(("|score", "|kpt")))} '
              f'cached group(s) from {args.det_cache}')

    results = {}
    # `renders` holds the futures still encoding; drained after the npz is written, so a slow encode
    # never delays the prediction reaching disk.
    render_pool = ThreadPoolExecutor(max_workers=1)
    renders = []
    for sess in sessions:
        sess.preload()
        for gid in sess.groups:
            if want and gid not in want:
                continue
            key = f'{sess.session_id}/{gid}'
            # ALL THREE INITIALISED TOGETHER, OUTSIDE THE BRANCH. `det_kpts` used to be bound
            # inside it, so every box source that is NOT a detector -- the GT-crop upper bound and
            # the whole `--boxes` path -- raised `UnboundLocalError` at the `run_group` call below,
            # after paying the checkpoint load. Same trap the comment above `det_tile` names.
            det_boxes = det_scores = det_kpts = None
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
                # WILL THE ANSWER ITSELF FIT? Checked HERE, before `detect_raw` allocates and
                # before hours of decode, because none of it is reachable by `--max-ram`: these
                # are `np.full` arrays proportional to the whole clip, not reusable buffers.
                #
                # A 200 fps hour is 720,000 frames, which is an ordinary recording. On a
                # 16-camera 24-keypoint rig that is 6.6 GB of detection arrays at top_k 2 and
                # 79.3 GB at top_k 24 -- so the failure is not exotic, it is what happens the
                # first time someone points this at a full session instead of a clip.
                _T_est = min(sess.groups[gid].n_frames, args.max_frames or sess.groups[gid].n_frames)
                _parts = _memory.result_array_gb(
                    _T_est, len(sess.rig), registry.n_keypoints, n_want, n_det,
                    det_kpts=bool(getattr(det, 'n_keypoints', 0)),
                    dims=3 if sess.mode == '3d' else 2)
                _need = sum(_parts.values())
                if _need > _budget.budget_gb:
                    _big = sorted(_parts.items(), key=lambda kv: -kv[1])[:3]
                    raise SystemExit(
                        f'{key}: this group would allocate {_need:.1f} GB of RESULT arrays for '
                        f'{_T_est:,} frames, against a {_budget.budget_gb:.1f} GB budget '
                        f'({_budget.source}).\n'
                        + ''.join(f'    {n:<18} {v:7.2f} GB\n' for n, v in _big)
                        + 'These scale with the LENGTH OF THE CLIP and are the answer itself, so '
                        'no --max-ram can shrink them. Either:\n'
                        f'  --max-frames N   score a prefix (N around '
                        f'{max(1, int(_T_est * _budget.budget_gb / max(_need, 1e-9) * 0.8)):,} '
                        'would fit here), then move the window and re-run; or\n'
                        '  --det-top-k K    lower the detection budget -- the keypoint array is '
                        '(top_k, T, C, K, 3) and is usually most of this; or\n'
                        '  split the recording into shorter groups at conversion time, which is '
                        'what every shipped root does.')
                if key in det_cache:
                    raw = (det_cache[key], det_cache.get(f'{key}|score'),
                           det_cache.get(f'{key}|kpt'))
                    # A CACHE WITHOUT KEYPOINTS CANNOT SERVE `--crop-source keypoints`, and the
                    # failure would be SILENT: `run_group` takes `det_kpts_stc is not None` as the
                    # switch, so a None here does not error, it quietly crops from the boxes and
                    # reports the arm under the other arm's name. That is the `--boxes`-key trap
                    # below, one flag over. Refused rather than warned, because the whole purpose of
                    # a shared cache is that two arms differ in exactly one lever.
                    if args.crop_source == 'keypoints' and raw[2] is None:
                        raise SystemExit(
                            f'{args.det_cache}: holds no keypoints for {key!r}, so --crop-source '
                            'keypoints would silently fall back to cropping from the boxes and '
                            'measure the arm it is being compared against. Delete it and re-detect '
                            '(a keypoint-trained detector fills this in), or drop --det-cache.')
                    # `flush` for the same reason the detecting branch has it: redirected to a log,
                    # stdout is block-buffered, so the CACHED path -- the fast one, which prints
                    # little else -- shows nothing for minutes and reads as a hung run.
                    print(f'{key}: up to {n_want} animal(s), raw boxes from --det-cache', flush=True)
                else:
                    print(f'{key}: detecting up to {n_det} per camera, {n_want} animal row(s)'
                          f'{"" if args.max_animals else " (from the labels; set --max-animals)"}',
                          flush=True)
                    _t_det = time.time()
                    raw = detect_raw(det, det_wh, sess, gid, n_det, device=device,
                                     score_thresh=args.det_score, reduce=det_red,
                                     max_frames=args.max_frames, tile_scale=det_tile)
                    _det_secs = time.time() - _t_det
                    # A keypoint-trained detector fills a third array, cached under its own key.
                    # This does NOT change what an old cache is allowed to satisfy: a box-only arm
                    # never looks at it, and a keypoint-crop arm is refused above rather than served
                    # boxes under the wrong name. Storing it is what lets the two crop sources share
                    # ONE box set and so differ in exactly one lever (eval rule 4) -- report 15 §6
                    # had to match its item-3 arms by configuration for want of this.
                    det_cache[key] = raw[0]
                    det_cache[f'{key}|score'] = raw[1]
                    if raw[2] is not None:
                        det_cache[f'{key}|kpt'] = raw[2]
                    cache_dirty = True
                    # CHECKPOINT AN EXPENSIVE GROUP IMMEDIATELY, rather than only at the end of the
                    # run. The end-of-run write below is still the one that matters for a short
                    # protocol, but it held rat-city's 57,594-frame group -- 3h18m of decode, 62 GB
                    # of JPEG -- in memory alone across a further ~3h pose pass, so any interruption
                    # lost the entire detection. That cache IS the artifact the long-clip benchmark
                    # exists to produce ("detect once, then every association arm is a CPU-minute"),
                    # and it did not survive the run that creates it.
                    #
                    # GATED ON THE TIME THE DETECTION ACTUALLY TOOK, not on a frame count: 60 s is
                    # "long enough that losing it would hurt", which is exactly the quantity in
                    # question. A 58-group protocol detects each group in seconds and so writes
                    # once, at the end, byte-identical to before; a single long group writes exactly
                    # once, immediately. Only a root with many SLOW groups pays repeated
                    # compression, and there it is buying back hours.
                    if args.det_cache and _det_secs > 60.0:
                        args.det_cache.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(args.det_cache, __stamp__=np.asarray(stamp),
                                            **det_cache)
                        print(f'{key}: detection took {_det_secs / 60:.1f} min -- checkpointed '
                              f'{args.det_cache}', flush=True)
                # THE ASSOCIATION HALF RUNS EVERY TIME, cached or not. It is microseconds per frame
                # against 44 ms of 4K decode, so recomputing it costs nothing measurable and buys
                # the property the cache exists for: two identity arms differ in exactly one lever
                # over byte-identical pixels.
                nms_stats = {}
                det_boxes, det_scores, det_kpts = associate_group(
                    raw, sess, gid, n_want, link=args.link_boxes, min_views=args.min_views,
                    track=args.track, max_move=args.max_move, stats=nms_stats,
                    pose_nms=args.pose_nms)
                # THE FIRE RATE IS THE NUMBER A RATE-MATCHED RANDOM CONTROL MUST BE MATCHED TO, and
                # it cannot be recovered afterwards from the npz -- a dropped row leaves no trace.
                # "Report the fire rate before the metric" means printed, not recoverable.
                if args.pose_nms is not None:
                    # `.get(..., 0)`, both keys: `identity.pose_nms` returns before writing EITHER
                    # key when the detector has no keypoint branch (`kpts is None`) -- a correct
                    # no-op, since the maDLC overlap it computes needs keypoints to exist at all.
                    # A keypoint-less detector is the NORMAL case for a 2D root (rat-city's own
                    # recipe omits --keypoints), so `nms_stats` being empty here is not a bug
                    # signal, and asserting `nms_pairs` unconditionally raised on every such run.
                    print(f'{key}: pose-nms dropped {nms_stats.get("nms_dropped", 0)} row(s) of '
                          f'{nms_stats.get("nms_pairs", 0)} overlapping pair(s)'
                          + (' (no keypoint branch -- pose-nms is a no-op)' if not nms_stats
                             else ''), flush=True)
                # HOW MUCH THE THRESHOLD LEFT. `--det-score` defaults to 0.5 (coverage-favouring);
                # a detector whose scores are NOT saturated would lose most of its boxes to a higher
                # threshold, and this line is where that shows up rather than downstream as an
                # unexplained miss rate.
                filled = float(np.isfinite(det_boxes).all(-1).mean())
                print(f'{key}: boxes in {filled:.3f} of (animal, frame, camera) slots'
                      f'{"   <-- LOW: try a smaller --det-score" if filled < 0.25 else ""}')
            # A MISSING `--boxes` KEY IS NOT AN ABSENT ARGUMENT. `boxes.get(key)` returning None
            # silently falls back to cropping from the LABELS, so a run whose keys did not match --
            # a different session naming, a stale npz, a typo in one group -- reported the GT-crop
            # oracle under a heading that said otherwise. Nothing in the output said which.
            if args.boxes and key not in boxes:
                raise SystemExit(
                    f'{args.boxes}: no entry for {key!r}. Falling back to the labels here would '
                    'quietly turn this into the GT-crop upper bound. Keys present: '
                    f'{sorted(k for k in boxes if not k.startswith("__"))[:5]} ...')
            out = run_group(model, sess, gid, registry, ds_name, cfg,
                            box_points=boxes.get(key), boxes_stc=det_boxes,
                            det_kpts_stc=det_kpts)
            if det_scores is not None:
                # The objectness each crop was accepted on, beside the prediction it produced.
                # `--det-score` is then an offline sweep instead of a re-detection per threshold.
                out['det_score'] = det_scores[:, :out['pred'].shape[1]]
            results[key] = out
            print(f'{key}: {out["pred"].shape} '
                  f'{np.isfinite(out["pred"]).all(-1).mean():.3f} finite')
            if args.render is not None:
                from tailcyclenet.render import render_group
                # RENDER ON A BACKGROUND THREAD so the loop can predict the next group while this
                # one encodes. A render of a 480-frame 4696x2048 clip is comparable in cost to the
                # inference that produced it, and it depends on nothing the loop mutates afterwards
                # -- `out['pred']` and `det_boxes` are finished arrays by here.
                #
                # ONE worker, not several. `_read_video`'s lock is now PER CONTAINER, so two
                # renders of two cameras would genuinely overlap their decodes -- but a render
                # holds a clip's worth of full frames and the encode, not the decode, is where a
                # render's time goes. The reason is memory now, not the lock.
                for ci in render_cams:
                    cam_name = sess.cam_names[ci]
                    # The per-frame boxes the crop rule was fed, in each row's own colour: a box
                    # with no skeleton in it is the disagreement worth seeing, and row `a` is not
                    # label row `a` once boxes come from a detector. `crop` is per WINDOW and the
                    # windows overlap, so it is not the array to draw here.
                    bx = (det_boxes[:, :out['pred'].shape[1], ci]
                          if det_boxes is not None else None)
                    renders.append((key, render_pool.submit(
                        render_group, sess, gid, out['pred'],
                        args.render / f'{key.replace("/", "__")}__{cam_name}.mp4',
                        cam=ci, zoom=args.render_zoom, boxes=bx)))
                # Report whatever has landed, without waiting for anything.
                for k, fut in [r for r in renders if r[1].done()]:
                    print(f'{k}: wrote {fut.result()}')
                    renders.remove((k, fut))

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
    # WHAT THE CROP WAS BUILT FROM, which `__box_source__` above does NOT say -- that one is the
    # detector's TRAINING target. Two arms differing only in `--crop-source` were previously
    # identical in their own provenance, so report 15 §6's pair could be told apart only by
    # filename. `--refine` rides here too, being the other re-crop lever, so a three-way comparison
    # is legible from the files alone.
    # `refine` is resolved per session from its mode, so read the RESOLVED flag off the results
    # rather than off `cfg`, which may still hold the tri-state None.
    did_refine = any(bool(r.get('refine')) for r in results.values())
    flat['__crop_source__'] = np.asarray(
        f'{cfg.crop_source}{"+refine" if did_refine else ""}'
        f'{f"@{cfg.refine_px}px" if did_refine and cfg.refine_px else ""}')
    np.savez_compressed(args.out, **flat)
    print(f'wrote {args.out} ({len(results)} group(s))')

    # THE PREDICTION IS ON DISK FIRST, then the renders are waited on. A render is a view and must
    # never be able to lose a run that has already paid for its inference.
    if renders:
        print(f'waiting on {len(renders)} render(s) still encoding', flush=True)
    for k, fut in renders:
        print(f'{k}: wrote {fut.result()}')
    render_pool.shutdown()

    # AFTER the prediction is on disk, never before: the cache is an optimisation for the next
    # run and a failure writing it must not lose the run that just paid for the detection.
    if args.det_cache and cache_dirty:
        args.det_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.det_cache, __stamp__=np.asarray(stamp), **det_cache)
        # COUNT GROUPS, NOT KEYS. Each group holds boxes plus `|score` plus `|kpt` where a
        # keypoint detector supplied them, so `len()` read 174 for 58 groups -- it was already
        # double-counting before the keypoint key existed, which is why 2x looked plausible.
        print(f'wrote {args.det_cache} '
              f'({sum(1 for k in det_cache if not k.endswith(("|score", "|kpt")))} '
              'group(s) of boxes)')


