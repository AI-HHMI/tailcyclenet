"""Run provenance: the unconditional key sets every prediction session records."""
from __future__ import annotations

from pathlib import Path

from ..checkpoints import provenance
from ..detector.boxes import NMS_CENTER_DIST


def _center_dist(args) -> float:
    """The centre-distance NMS gate the detections were decoded with: the flag's value, the
    shared default when a caller's namespace never set it, and 0.0 for an explicit None -- the
    gate is off, which a 0.0 threshold is (every box is at distance >= 0)."""
    v = getattr(args, 'det_nms_center_dist', NMS_CENTER_DIST)
    return 0.0 if v is None else float(v)


def box_provenance(args, det_tile, det_red, det_boxsrc):
    """Every input the detections depend on, recorded beside the numbers they produced.

    Unconditional and always complete (a provenance record has no cache-invalidation pressure),
    so two runs that look alike differ in their output file where a reader can see it. Keyed by
    the CLI name where it differs from `detect_raw`'s parameter name -- that is what a reader has
    to type to reproduce it. `tests/infer/test_provenance.py` walks `detect_raw`'s signature against this
    dict, so a new box-affecting parameter must land here too. The frame range is recorded because
    it changes the detections (which frames, and the aligned lead-in); the detector's box target
    is recorded as `det_box_source`, NOT `box_source`, because the two are the detector's training
    target and the pose model's crop rule respectively -- they are allowed to disagree, which is
    why both are recorded.
    """
    return {
        'detector': str(Path(args.detector).resolve()) if args.detector else '',
        'detector_checkpoint': str(getattr(args, '_detector_checkpoint', '') or ''),
        'det_score': float(args.det_score),
        'det_nms_iou': float(args.det_nms_iou),
        'det_nms_center_dist': _center_dist(args),
        'det_top_k': int(args.det_top_k) if args.det_top_k else 0,
        'max_animals': int(args.max_animals),
        'max_frames': int(args.max_frames),
        'frame_start': int(args.frame_start),
        'frame_stop': int(args.frame_stop),
        'tile_scale': float(det_tile) if det_tile else 0.0,
        'reduce': bool(det_red),
        'det_box_source': str(det_boxsrc or 'keypoints') if args.detector else '',
    }


def identity_provenance(args):
    """Every lever that decides which ROW a detection lands in, recorded beside its numbers.

    The sibling of `box_provenance`, and it exists for the same reason one frame later in the
    pipeline: detection provenance answers "which boxes", this answers "whose". Without it two
    runs differing only in `--link-boxes` are distinguishable **only by their
    directory name**, which is how the two stored 2D suppression arms in `scratch/dupfollow/pred`
    ended up unmatched -- their `[provenance]` records the detector, the crop and the checkpoint,
    and says nothing about the identity mechanism whose effect they were built to measure. Eval
    rule 4 (match the controls) and rule 12 (a same-recipe replicate) both need these keys to be
    checkable from the artifact rather than assumed.

    Unconditional and always complete, exactly as `box_provenance` is: the same keys at every
    value, since conditional membership is what makes a record lie. `tests/infer/test_provenance.py`
    walks `associate_group`'s signature against this dict, so a new identity-affecting parameter
    must land here too.
    """
    return {
        'track': bool(args.track),
        'link_boxes': bool(args.link_boxes),
        'min_views': int(args.min_views),
        'max_move': float(args.max_move),
        'max_age': int(args.max_age),
        'identity_bridge': bool(args.identity_bridge),
        'duplicate_radius': float(args.duplicate_radius),
        'duplicate_persist': int(args.duplicate_persist),
        'assoc_res_max_sides': (float(args.assoc_res_max_sides)
                                if args.assoc_res_max_sides is not None else 0.0),
        'assoc_score_order': bool(args.assoc_score_order),
        'track_age_decay': (float(args.track_age_decay)
                            if args.track_age_decay is not None else -1.0),
        'track_affinity_2d': float(args.track_affinity_2d),
        'track_camera_claims': bool(args.track_camera_claims),
        'assoc_fast': bool(args.assoc_fast),
        'assoc_random_drop': float(args.assoc_random_drop),
    }


def run_provenance(args, sess, ds_name, ckpt, cfg, config, src_prov, scorer_run, det_tile, det_red,
                   det_boxsrc):
    """The run's provenance as a LIST OF PAIRS (`SessionWriter` raises on a key given twice).

    Inputs: args, the source (sess, ds_name, src_prov), the pose checkpoint/config, cfg,
        scorer_run = `load_scorer`'s tuple, and the detector facts `box_provenance` records.
    Outputs: a list of (key, value) pairs. Side effects: reads git state (`provenance()`).
    """
    scorer, scorer_ckpt, scorer_shared_encoder, scorer_prov = scorer_run
    scorer_path = args.scorer
    allow_pose_scorer_mismatch = bool(args.allow_pose_scorer_mismatch)
    precision = args.precision
    camera_batch = args.camera_batch
    return [*src_prov.items(),
            *{'source_session_id': sess.session_id,
            'source_dataset': ds_name,
            'run': str(Path(args.run).resolve()),
            'checkpoint': str(Path(ckpt).resolve()),
            'checkpoint_name': ckpt.name,
            'anchor': cfg.anchor, 'carry_source': cfg.carry_source,
            'window_predictions': bool(args.window_predictions),
            'n_frames': cfg.n_frames, 'overlap': cfg.overlap,
            'frame_start': cfg.frame_start, 'frame_stop': cfg.frame_stop,
            'refine': bool(cfg.refine), 'refine_px': cfg.refine_px or 0,
            'refine_all_cams': bool(cfg.refine_all_cams),
            'box_corrupt': str(cfg.box_corrupt or ''),
            'refine_consensus': bool(cfg.refine_consensus),
            'refine_consensus_res': float(cfg.refine_consensus_res),
            'refine_consensus_vis': float(cfg.refine_consensus_vis),
            'crop_inflate_adaptive': bool(cfg.crop_inflate_adaptive),
            'crop_inflate_max': float(cfg.crop_inflate_max),
            'precision': precision, 'camera_batch': bool(camera_batch),
            'boxes': (str(args.detector) if args.detector else
                      (str(args.boxes) if args.boxes else 'labels')),
            'box_source': ((det_boxsrc or 'keypoints') if args.detector
                           else cfg.box_source),
            'vis_thresh': float(cfg.vis_thresh) if cfg.vis_thresh else 0.0,
            'box_prompt': cfg.box_prompt,
            'box_prompt_requested': str(args.box_prompt),
            'box_prompt_first_only': bool(cfg.box_prompt_first_only),
            'crop_inflate': float(cfg.crop_inflate),
            'gridresid_offset': str(config['model']['gridresid_offset']),
            'gridresid_offset_override': str(args.gridresid_offset or ''),
            **provenance()}.items(),
           *([] if scorer is None else [
               ('scorer', str(Path(scorer_path).resolve())),
               ('scorer_checkpoint', str(Path(scorer_ckpt).resolve())),
               ('scorer_checkpoint_name', Path(scorer_ckpt).name),
               ('scorer_commit', str(scorer_prov.get('commit', ''))),
               ('scorer_dirty', bool(scorer_prov.get('dirty', False))),
               ('scorer_precision', precision),
               ('scorer_encoder_shared', bool(scorer_shared_encoder)),
               ('pose_scorer_mismatch_allowed',
                bool(allow_pose_scorer_mismatch))]),
            *box_provenance(args, det_tile, det_red, det_boxsrc).items(),
            *identity_provenance(args).items()]
