"""Loading the networks a run deploys, and the compatibility rules between them and the source."""
from __future__ import annotations

from pathlib import Path

from ..checkpoints import load_run, peek_registry
from ..loader.config import LoaderConfig
from .predictions import refuse_multi_session


def share_pose_scorer_encoder(pose_model, scorer_model, *, allow_mismatch,
                               pose_run, scorer_run) -> bool:
    """Share the raw video backbone, refusing unequal weights without explicit opt-in."""
    from ..model import share_video_encoder_if_identical

    shared = share_video_encoder_if_identical(pose_model, scorer_model)
    if not shared and not allow_mismatch:
        raise SystemExit(f'{scorer_run}: raw video-encoder weights do not exactly match '
                         f'{pose_run}; refusing two different scoring/pose encoders. Pass '
                         '--allow-pose-scorer-mismatch to opt in to separate encoder passes.')
    return shared


def _dataset_family(name: str) -> str:
    """The leading `-`-token of a dataset name: 'rat-city-combined' and 'rat-city-tracked' are
    both 'rat'; 'johnson-mouse-combined-aug' and 'johnson-mouse-combined' are both 'johnson'.

    A detector-vs-session dataset check needs SOME notion of "the same dataset", but this repo
    routinely trains a detector and a pose run, and evaluates them, on differently-suffixed roots
    that are the same underlying species/rig (`-combined` / `-aug` / `-tracked` / `-annotated`
    variants -- see `scratch/phase15/run.sh`'s own `rat-city-tracked` session against a
    `rat-city-combined` detector). Exact string equality would refuse that ALREADY-WORKING,
    ALREADY-CHECKED pairing. The first token is a coarse but effective family key for every
    dataset name in this repo; it is not exact by construction, so it only guards the unambiguous
    cross-species accident (a calms21 detector on a rat-city session), not a same-family variant.
    """
    return str(name).split('-', 1)[0].strip().lower()


def detector_matches_source(trained_datasets, trained_on, ds_name):
    """Multi-root checkpoints require exact membership; legacy/single-root keeps family matching."""
    if len(trained_datasets) > 1:
        return ds_name in trained_datasets
    return bool(trained_on and ds_name
                and _dataset_family(trained_on) == _dataset_family(ds_name))


def detector_box_source(det, ds_name, default):
    """Effective detector target for this source root, with a unique-family fallback."""
    by_root = getattr(det, 'box_sources', {}) or {}
    if ds_name in by_root:
        return by_root[ds_name]
    family = _dataset_family(ds_name) if ds_name else ''
    matches = [value for name, value in by_root.items()
               if family and _dataset_family(name) == family]
    return matches[0] if len(matches) == 1 else default


def load_source(args):
    """The source session: `--videos` adopted in memory, or the one `--data` session.

    `--assoc-res-max-px` defaults to None so a `--data` session's own stored value survives
    unless given; under `--videos` an absent flag resolves to the format default.

    Inputs: args. Outputs: (ds_name, sess, src_prov) -- the registry key, the Session, and the
        source half of the run provenance (the same keys on both branches).
    Side effects: decodes container metadata (`--videos`); may write `--dump-session`; prints.
    """
    assoc_res_max_px = (args.assoc_res_max_px if args.assoc_res_max_px is not None else 30.0)

    if args.videos:
        from . import adopt
        adopt.check_flags(args)
        reg = peek_registry(args.run)
        vplan = adopt.plan(args.videos, args.calibration, args.cam_regex,
                          session_id=args.session_id, group_id=args.group_id)
        if args.units is None:
            if vplan.mode == '3d':
                raise SystemExit('--units is required for a 3D --videos run: calibration.toml '
                                 'has no units field, so state the calibration units explicitly.')
            units = 'px'
        else:
            units = args.units
        if getattr(vplan, 'mode', None) == '3d' and units == 'px':
            raise SystemExit('--units px is inconsistent with a 3D --videos run: pixel units '
                             'describe 2D image coordinates, not calibrated world coordinates.')
        ds_name = adopt.dataset_name(reg, args.dataset_name)
        sess = adopt.build(vplan, names=reg.local_names(ds_name), units=units,
                           fps=args.fps, assoc_res_max_px=assoc_res_max_px,
                           trim=args.trim_to_shortest)
        src_prov = adopt.provenance_of(vplan)
        src_prov['source_split'] = ''
        if args.dump_session:
            adopt.dump(sess, args.dump_session)
    else:
        args.split = args.split or 'test'
        ds_name, sess = refuse_multi_session(args.data, args.split)
        if args.assoc_res_max_px is not None:
            print(f'assoc_res_max_px: overriding the session\'s own '
                  f'{sess.assoc_res_max_px:g}px with {assoc_res_max_px:g}px')
            sess.assoc_res_max_px = float(assoc_res_max_px)
        src_prov = {'source': 'tailcyclenet infer',
                    'source_session': str(Path(args.data).resolve()),
                    'source_split': args.split,
                    'source_calibration': '', 'source_cam_regex': '',
                    'source_group_id': '', 'source_videos': []}
    return ds_name, sess, src_prov


def load_pose(args, device):
    """Load the pose run (`--run`/`--pose-checkpoint`) onto `device` at the requested speed.

    Inputs: args, device. Outputs: (model, config, registry, ckpt) as `load_run` returns them.
    Side effects: loads weights; prints the speed and model lines.
    """
    over = ({'gridresid_offset': args.gridresid_offset} if args.gridresid_offset else None)
    model, config, registry, ckpt = load_run(args.run, args.checkpoint, device=device,
                                             model_overrides=over)
    precision = args.precision
    camera_batch = args.camera_batch
    model.set_scene_speed(precision=precision, camera_batch=camera_batch)
    print(f'speed: precision={precision} camera_batch={camera_batch}')
    print(f'model: {ckpt.name}  ({registry.n_keypoints} keypoints)  '
          f'query={config["model"].get("query", "prior")}  '
          f'gridresid_offset={config["model"]["gridresid_offset"]}')
    return model, config, registry, ckpt


def load_scorer(args, model, registry, trained_frames, trained_px, device):
    """Load `--scorer` and refuse one that does not fit the pose run.

    Inputs: args, the pose model/registry and its trained window, device.
    Outputs: (scorer, scorer_ckpt, shared_encoder, scorer_prov); (None, None, False, {}) without
        `--scorer`.
    Side effects: loads weights; may share the pose video encoder with the scorer; prints.
    """
    scorer_path = args.scorer
    scorer_checkpoint_name = args.scorer_checkpoint
    allow_pose_scorer_mismatch = bool(args.allow_pose_scorer_mismatch)
    precision = args.precision
    camera_batch = args.camera_batch
    scorer = None
    scorer_ckpt = None
    scorer_config = None
    scorer_shared_encoder = False
    scorer_prov = {}
    if scorer_path is not None:
        from ..scorer.checkpoint import load_scorer_run
        scorer, scorer_config, scorer_registry, scorer_ckpt = load_scorer_run(
            scorer_path, checkpoint=scorer_checkpoint_name, device='cpu')
        if scorer.output_granularity != 'frame':
            raise SystemExit(f'{scorer_path}: integrated inference requires a framewise scorer; '
                             f'got output_granularity={scorer.output_granularity!r}')
        if tuple(scorer_registry.names) != tuple(registry.names):
            pose_names, score_names = set(registry.names), set(scorer_registry.names)
            raise SystemExit(f'{scorer_path}: scorer keypoint registry differs from pose registry; '
                             f'missing from scorer={sorted(pose_names - score_names)[:8]}, '
                             f'extra in scorer={sorted(score_names - pose_names)[:8]}. '
                             'Train/use a scorer with the pose run registry.')
        scorer_frames = int(scorer_config['data'].get('n_frames', LoaderConfig.n_frames))
        scorer_px = int(scorer_config['data'].get('image_size', LoaderConfig.image_size))
        infer_frames = int(args.n_frames or trained_frames)
        if scorer_frames != trained_frames or infer_frames != scorer_frames:
            raise SystemExit(f'{scorer_path}: scorer n_frames={scorer_frames} must equal pose '
                             f'training/inference n_frames={trained_frames}/{infer_frames}')
        if scorer_px != trained_px:
            raise SystemExit(f'{scorer_path}: scorer image_size={scorer_px} must equal pose '
                             f'image_size={trained_px}')
        scorer_shared_encoder = share_pose_scorer_encoder(
            model, scorer, allow_mismatch=allow_pose_scorer_mismatch,
            pose_run=args.run, scorer_run=scorer_path)
        if scorer_shared_encoder:
            print('scorer: raw video encoder weights exactly match pose; sharing one encoder')
        else:
            print('WARNING: pose/scorer raw video-encoder weights differ; running separate '
                  'encodes by explicit --allow-pose-scorer-mismatch opt-in')
        scorer = scorer.to(device).eval()
        scorer.set_scene_speed(precision=precision, camera_batch=camera_batch)
        scorer_prov = dict(getattr(scorer, 'run_provenance', None) or {})
    return scorer, scorer_ckpt, scorer_shared_encoder, scorer_prov


def load_deploy_detector(args, device, ds_name, cfg):
    """Load `--detector` and refuse one trained for another source or crop rule.

    Inputs: args, device, ds_name (the session's registry key), cfg.
    Outputs: (det, det_wh, det_red, det_tile, det_boxsrc); all None (det_red False) without
        `--detector`.
    Side effects: loads weights; records the resolved file on `args._detector_checkpoint`;
        prints the detector line and calibration warnings; raises SystemExit on a refusal.
    """
    det = det_wh = det_tile = det_boxsrc = None
    det_red = False
    if args.detector:
        from tailcyclenet.detector.checkpoint import load_detector, resolve_detector_checkpoint
        det_path = resolve_detector_checkpoint(args.detector, args.detector_checkpoint)
        args._detector_checkpoint = str(det_path.resolve())
        loaded = load_detector(det_path, device)
        det, det_wh, det_ds, det_mcd = (loaded.model, loaded.input_wh, loaded.dataset,
                                        loaded.min_crop_dim)
        det_red, det_boxsrc, det_tile, det_objq = (loaded.reduce, loaded.box_source,
                                                   loaded.tile_scale, loaded.obj_quantiles)
        if det_objq and args.det_score >= det_objq.get('q50', 0.0):
            print(f'WARNING: --det-score {args.det_score} is at or above this detector\'s MEDIAN '
                  f'objectness ({det_objq["q50"]:.4f}), so it discards at least half of the '
                  f'detections it offers. q01 {det_objq.get("q01", float("nan")):.4f} '
                  f'q10 {det_objq.get("q10", float("nan")):.4f} '
                  f'q90 {det_objq.get("q90", float("nan")):.4f}. This detector is NOT saturated; '
                  'sweep --det-score per checkpoint (the 0.01 default favours recall; raising '
                  'it was measured to cost coverage).', flush=True)
        trained_datasets = list(getattr(det, 'trained_datasets', []) or ([det_ds] if det_ds else []))
        det_boxsrc = detector_box_source(det, ds_name, det_boxsrc)
        trained_label = (trained_datasets[0] if len(trained_datasets) == 1 else
                         (', '.join(trained_datasets) if trained_datasets else det_ds))
        print(f'detector: {det_path} ({det_wh[0]}x{det_wh[1]}'
              + (f' TILE at scale {det_tile:g}, whole-frame input derived per camera'
                 if det_tile else '')
              + f', trained on {trained_label!r}, boxes={det_boxsrc or "keypoints"})')
        if ds_name and trained_datasets and not detector_matches_source(
                trained_datasets, det_ds, ds_name) and not args.allow_detector_transfer:
            if len(trained_datasets) > 1:
                raise SystemExit(
                    f'{args.detector}: multi-root detector was trained on {trained_datasets}, '
                    f'but this session is {ds_name!r}, which is not a member root. Pass '
                    '--allow-detector-transfer for an explicit, labelled transfer-evaluation run.')
            raise SystemExit(
                f'{args.detector}: trained on dataset {det_ds!r}, but this session is '
                f'{ds_name!r} -- different dataset families. Detectors are trained one per '
                'dataset -- see CLAUDE.md -- so a cross-dataset deploy risks domain-shift false '
                'negatives measured as though they were this dataset\'s own coverage. Pass '
                '--allow-detector-transfer for an explicit, labelled transfer-evaluation run.')
        if det_mcd != cfg.min_crop_dim:
            raise SystemExit(
                f'{args.detector}: trained at min_crop_dim={det_mcd}, but this run\'s '
                f'[data].min_crop_dim is {cfg.min_crop_dim}. The detector reproduces the crop '
                'rule; two floors are two rules.')
        if (det_boxsrc or 'keypoints') != cfg.box_source:
            print(f'WARNING: detector boxes are {det_boxsrc!r} but this run was trained on '
                  f'{cfg.box_source!r} crops. Two crop sources are two crop rules -- do not read '
                  'a delta against a run whose detector matched as a detector-quality result.')
    return det, det_wh, det_red, det_tile, det_boxsrc
