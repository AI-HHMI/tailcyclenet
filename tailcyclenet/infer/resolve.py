"""Pure flag refusals and resolution: everything checked before a checkpoint loads."""
from __future__ import annotations

from pathlib import Path

from ..loader.config import LoaderConfig
from ..data.crop import crop_rule
from .window import InferConfig


def refuse_selector_beside_file(path, selector, flag, selector_flag, default=None):
    """A checkpoint selector picks a file INSIDE a run folder; beside a file it means nothing.

    Pure (no checkpoint load), so the refusal lands before any model or video is touched.
    """
    if path is None:
        return
    path = Path(path)
    if path.is_file() and selector is not None and selector != default:
        raise SystemExit(f'{flag} {path} is already a checkpoint file; {selector_flag} '
                         f'{selector!r} selects a file inside a run folder and is refused here')


def check_frame_range(args) -> None:
    """Refusals on `--start-frame` / `--end-frame`; pure argument arithmetic, so it fires above
    both input branches and before the checkpoint loads.

    Resolves the ONE quantity the loop reads: `args.frame_start` / `args.frame_stop`, with
    `--max-frames N` folding into `frame_stop = N`. The group-length check lives in the group loop.
    Three refusals fire here, numbered 15-17 in the infer/adopt refusal ledger.
    """
    if args.max_frames and (args.start_frame or args.end_frame):
        raise SystemExit(
            f'--max-frames {args.max_frames} together with --start-frame {args.start_frame} / '
            f'--end-frame {args.end_frame} has two readings of equal force -- "N frames from the '
            'start" and "up to frame N, from the start" -- and no defensible precedence, so a '
            'rule here would make one of them silently wrong. --max-frames N IS --start-frame 0 '
            '--end-frame N; pick one spelling.')
    if args.start_frame < 0 or args.end_frame < 0:
        raise SystemExit(
            f'--start-frame {args.start_frame} / --end-frame {args.end_frame}: negative frame '
            'indices are not Python negative indexing here. "-1 means the last frame" and "-1 '
            'means one before the end" are both obvious and differ by one, which is the class of '
            'ambiguity that costs a day.')
    if args.end_frame and args.end_frame <= args.start_frame:
        raise SystemExit(
            f'--end-frame {args.end_frame} is not past --start-frame {args.start_frame}. The '
            'range is half-open [start, end), so this predicts nothing; an empty range is a typo, '
            'not a protocol.')
    args.frame_start = int(args.start_frame)
    args.frame_stop = int(args.end_frame or args.max_frames or 0)


def refuse_before_load(args):
    """Every pure-argument refusal, run before any checkpoint loads (a typo should not cost a load).

    Inputs: args -- the parsed `infer` namespace. Outputs: None.
    Side effects: prints the oracle warnings; raises SystemExit on a contradictory flag set.
    """
    scorer_path = args.scorer
    scorer_checkpoint_name = args.scorer_checkpoint
    allow_pose_scorer_mismatch = bool(args.allow_pose_scorer_mismatch)
    if scorer_path is None and scorer_checkpoint_name is not None:
        raise SystemExit('--scorer-checkpoint requires --scorer')
    if scorer_path is None and allow_pose_scorer_mismatch:
        raise SystemExit('--allow-pose-scorer-mismatch requires --scorer')
    refuse_selector_beside_file(args.run, args.checkpoint, '--pose',
                                 '--pose-checkpoint')
    refuse_selector_beside_file(scorer_path, scorer_checkpoint_name, '--scorer',
                                 '--scorer-checkpoint')
    refuse_selector_beside_file(args.detector, args.detector_checkpoint,
                                 '--detector', '--detector-checkpoint', default='latest')
    if args.anchor == 'labels':
        if args.detector or args.boxes:
            raise SystemExit(
                '--anchor labels seeds row `a` from LABEL row `a`, but --detector/--boxes rows are '
                'score- or association-ordered and are not label rows. That is a different '
                "animal's ground truth, not an oracle. Use --anchor labels with the label crop "
                'path (no --detector, no --boxes), or --anchor carry / none with a box source.')
        print('WARNING: --anchor labels seeds the model with GROUND TRUTH. This is an oracle '
              'upper bound, not a deployment number. Label it as such wherever you quote it.')

    if args.oracle_corrupt:
        from .oracle import ORACLE_CORRUPTIONS
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

    if getattr(args, 'box_corrupt', None):
        from .oracle import parse_box_corrupt
        try:
            parse_box_corrupt(args.box_corrupt)
        except ValueError as e:
            raise SystemExit(f'--box-corrupt {args.box_corrupt!r}: {e}')
        if not args.detector:
            raise SystemExit('--box-corrupt breaks DETECTOR boxes; without --detector there are '
                             'none (--boxes and the labels give points, not boxes).')
        print(f'*** DIAGNOSTIC: the detector boxes are deliberately corrupted '
              f'({args.box_corrupt}). Not a deployment number. ***')

    if args.refine_px is not None:
        if args.refine is False:
            raise SystemExit('--refine-px sets the resolution of --refine\'s FIRST pass, and '
                             '--no-refine turns that pass off. Pick one.')
        if args.refine_px < 32:
            raise SystemExit(f'--refine-px {args.refine_px} is below the structural floor of 32 '
                             '(about two patches). Below ~2 patches the forward returns all-NaN '
                             'with no exception. The measured floor is 96; 64 is already worse '
                             'than not refining at all.')

    check_frame_range(args)


def refuse_out_dir(args):
    """Refuse an `--out` that is the source session or a non-empty directory.

    Inputs: args. Outputs: None. Side effects: raises SystemExit; nothing is deleted.
    """
    if not args.videos and args.data is not None and Path(args.data).resolve() == args.out.resolve():
        raise SystemExit(
            f'--out {args.out} names the SOURCE session itself (--data). Writing a prediction '
            "into the labels it is scored against is not a run -- point --out somewhere else.")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f'{args.out}: the output session directory is not empty. A run writes a block at a '
            'time and a table with no NEW rows is never created, so reusing a directory would '
            'silently mix an old table (or an old identity_events.pq) into the new session as '
            'if this run had written it. Point --out at a fresh directory; nothing is deleted '
            'automatically.')


def refuse_scorer_rig(scorer_path, sess):
    """Refuse a `--scorer` on a rig the scorer cannot score (moving or uncalibrated single-view 3D).

    Inputs: scorer_path (None = no scorer), sess. Outputs: None. Side effects: raises SystemExit.
    """
    if scorer_path is not None:
        moving = [name for name, is_moving in sess.rig.moving.items() if is_moving]
        if moving:
            raise SystemExit(f'{scorer_path}: scorer does not support moving-camera rigs; '
                             f'moving camera(s): {moving}')
        if (sess.mode == '3d' and len(sess.cam_names) == 1
                and not sess.rig.calibrated[sess.cam_names[0]]):
            raise SystemExit(f'{scorer_path}: scoring uncalibrated single-view 3D coordinates is '
                             'unsupported; use a calibrated rig or a 2D session.')


def trained_window(args, config):
    """The pose run's trained window -> (n_frames, image_size); refuses a longer `--n-frames`.

    Inputs: args, config -- the pose run's config. Side effects: raises SystemExit.
    """
    trained_frames = int(config['data'].get('n_frames', LoaderConfig.n_frames))
    if args.n_frames and args.n_frames > trained_frames:
        raise SystemExit(f'--n-frames {args.n_frames} exceeds the run\'s trained window '
                         f'({trained_frames}). Shorter windows are fine; longer is not the same '
                         'model.')
    trained_px = int(config['data'].get('image_size', LoaderConfig.image_size))
    return trained_frames, trained_px


def infer_config(args, config, sess, trained_frames, trained_px, device):
    """Resolve the box deployment recipe and build the run's `InferConfig`.

    `--box-prompt auto/firstonly` resolves to `detector` for a box model with a detector/boxes
    file and never falls back to the GT `labels` oracle; a box model then takes
    crop_inflate 1.5, refine on and refine_px 128 unless set.

    Inputs: args, config (pose run config), sess, trained_frames/trained_px (`trained_window`),
        device. Outputs: an `InferConfig`.
    Side effects: prints the resolved recipe and warnings; raises SystemExit on a refusal.
    """
    if args.refine_px and args.refine_px > trained_px:
        raise SystemExit(f'--refine-px {args.refine_px} exceeds the run\'s image_size '
                         f'({trained_px}). A reduced first pass is the lever; a larger one is a '
                         'different model.')
    model_is_box = config.get('model', {}).get('box_prompt', 'none') != 'none'
    box_prompt = args.box_prompt
    box_prompt_first_only = (box_prompt == 'firstonly')
    if box_prompt in ('auto', 'firstonly'):
        if model_is_box and (args.detector or args.boxes):
            box_prompt = 'detector'
        elif model_is_box:
            raise SystemExit(
                f'this run is a box model ([model].box_prompt != "none") and --box-prompt '
                f'{args.box_prompt} must source the box from pixels, but no --detector or '
                '--boxes file was given. Pass --detector/--boxes (deployment), --box-prompt '
                'labels to EXPLICITLY use the GROUND-TRUTH oracle, or --box-prompt none to run '
                'the box model with the box withheld.')
        else:
            box_prompt = 'none'
            box_prompt_first_only = False
    if box_prompt != 'none' and not model_is_box:
        print(f'--box-prompt {args.box_prompt}: this run is not a box model '
              '([model].box_prompt = "none"), so the box is ignored.')
        box_prompt = 'none'
        box_prompt_first_only = False
    if box_prompt == 'labels':
        if args.detector or args.boxes:
            raise SystemExit(
                '--box-prompt labels seeds the box from LABEL row `a`, but --detector/--boxes '
                'rows are score- or association-ordered and are not label rows. That is a '
                "different animal's ground truth, not an oracle -- the same reason --anchor "
                'labels refuses this combination. Use --box-prompt labels with the label crop '
                'path (no --detector, no --boxes), or --box-prompt detector/none with a box '
                'source.')
        print('WARNING: --box-prompt labels seeds the box from GROUND TRUTH. This is an oracle '
              'upper bound, not a deployment number. Label it as such wherever you quote it.')
    box_on = box_prompt != 'none'
    if box_prompt_first_only:
        print("box prompt: FIRSTONLY -- given at each animal's first window, withheld once a "
              'carry is available (owner-set default, 2026-09-05; report 55).')
    crop_inflate = args.crop_inflate if args.crop_inflate is not None else (1.5 if box_on else 1.0)
    refine = args.refine if args.refine is not None else (True if box_on else sess.mode == '3d')
    refine_px = args.refine_px if args.refine_px is not None else (128 if box_on else None)
    if box_on:
        print(f'box deployment recipe: --box-prompt {box_prompt} '
              f'--crop-inflate {crop_inflate} --refine{"" if refine else " off"} '
              f'--refine-px {refine_px}')
    cfg = InferConfig(
        n_frames=args.n_frames or trained_frames,
        overlap=args.overlap, image_size=trained_px,
        **dict(zip(('crop_pad_frac', 'min_crop_dim'), crop_rule(config['data']))),
        box_source=config['data'].get('box_source', LoaderConfig.box_source),
        anchor=args.anchor, max_animals=args.max_animals, max_frames=args.max_frames,
        frame_start=args.frame_start, frame_stop=args.frame_stop,
        kpt_chunk=args.kpt_chunk,
        vis_thresh=args.vis_thresh, refine=refine, refine_px=refine_px,
        carry_source=args.carry_source, min_box_frames=args.min_box_frames,
        oracle_corrupt=args.oracle_corrupt, device=device,
        box_prompt=box_prompt, box_prompt_first_only=box_prompt_first_only,
        crop_inflate=crop_inflate,
        prefetch_windows=args.prefetch_windows,
        capture_window_predictions=bool(args.window_predictions),
        refine_all_cams=bool(args.refine_all_cams),
        box_corrupt=args.box_corrupt or None,
        refine_consensus=bool(args.refine_consensus),
        refine_consensus_res=float(args.refine_consensus_res),
        refine_consensus_vis=float(args.refine_consensus_vis),
        crop_inflate_adaptive=bool(args.crop_inflate_adaptive),
        crop_inflate_max=float(args.crop_inflate_max),
        refine_carry_fallback=(sess.mode == '3d' if args.refine_carry_fallback is None
                               else bool(args.refine_carry_fallback)),
        refine_carry_dup_radius=float(args.refine_carry_dup_radius),
        refine_carry_scale_max=float(args.refine_carry_scale_max))
    if cfg.refine_carry_scale_max != 0 and cfg.refine_carry_scale_max < 1:
        raise SystemExit(f'--refine-carry-scale-max {cfg.refine_carry_scale_max} is below 1; it is '
                         'a factor either way (0 turns the scale gate off).')
    if cfg.refine_carry_dup_radius < 0:
        raise SystemExit(f'--refine-carry-dup-radius {cfg.refine_carry_dup_radius} is negative; '
                         '0 turns the duplicate-carry gate off.')
    if cfg.refine_carry_fallback and sess.mode != '3d':
        raise SystemExit('--refine-carry-fallback reprojects a carried 3D pose into each camera; '
                         'this is a 2D run.')
    if cfg.crop_inflate_adaptive and cfg.crop_inflate_max < 1.0:
        raise SystemExit(f'--crop-inflate-max {cfg.crop_inflate_max} is below 1.0; R3 never '
                         'shrinks a crop below its box.')
    if cfg.refine_consensus and not (cfg.refine and sess.mode == '3d'):
        raise SystemExit('--refine-consensus tests pass 1\'s cameras against each other, so it '
                         'needs 3D and --refine; this run is '
                         f'{sess.mode}{"" if cfg.refine else " with --no-refine"}.')
    if cfg.box_source != 'keypoints':
        print(f'crops: box_source={cfg.box_source} (from the run config); a session with no '
              'instances.pq falls back to its keypoints')
    return cfg


def select_groups(args, sess, cfg):
    """The groups this run predicts: `--groups`, minus groups at or below `--start-frame`.

    Inputs: args, sess, cfg. Outputs: a list of group ids.
    Side effects: prints each skipped group; raises SystemExit when nothing is left.
    """
    want = set(args.groups.split(',')) if args.groups else None

    gids = [g for g in sess.groups if not want or g in want]
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
    return gids
