"""Run provenance: the unconditional key sets every prediction session records."""
from __future__ import annotations

from pathlib import Path


def box_provenance(args, det_tile, det_red, det_boxsrc):
    """Every input the detections depend on, recorded beside the numbers they produced.

    Unconditional and always complete (a provenance record has no cache-invalidation pressure),
    so two runs that look alike differ in their output file where a reader can see it. Keyed by
    the CLI name where it differs from `detect_raw`'s parameter name -- that is what a reader has
    to type to reproduce it. `tests/test_detector.py` walks `detect_raw`'s signature against this
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
        'det_nms_center_dist': (float(args.det_nms_center_dist)
                               if args.det_nms_center_dist is not None else 0.0),
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
    value, since conditional membership is what makes a record lie. `tests/test_detector.py`
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
    }
