"""Prediction provenance records every box- and identity-affecting option."""


def test_provenance_records_every_box_affecting_option():
    """Every `detect_raw` parameter that changes a detection is recorded, at every value
    (conditional membership is what makes a record lie)."""
    import argparse
    import inspect

    from tailcyclenet.infer.detect import detect_raw
    from tailcyclenet.infer.provenance import box_provenance

    args = argparse.Namespace(detector=None, det_score=0.5, det_top_k=0,
                              max_animals=0, max_frames=0, frame_start=0, frame_stop=0,
                              det_nms_iou=0.5, det_nms_center_dist=None)
    prov = box_provenance(args, None, False, None)

    # Everything `detect_raw` takes that can change the detections. The rest are plumbing --
    # `batch` is in there under protest: it is NOT inert but it is pinned rather
    # than exposed, so no run can differ in it. `input_wh` is the checkpoint's own, so
    # `detector_checkpoint` records it. `frames`/`read` are the block loop's plumbing and
    # change no pixel. See tests/test_memory.py.
    plumbing = {'det', 'session', 'gid', 'device', 'batch', 'frames', 'read', 'trace',
                'trace_detail', 'embed_out', 'input_wh'}
    params = set(inspect.signature(detect_raw).parameters) - plumbing
    # How each is spelled in the record, where the CLI name differs from the parameter name.
    alias = {'score_thresh': 'det_score', 'top_k': 'det_top_k',
             'iou_thresh': 'det_nms_iou', 'center_dist_thresh': 'det_nms_center_dist',
             'box_source': 'det_box_source'}
    missing = [p for p in sorted(params) if alias.get(p, p) not in prov]
    assert not missing, (
        f'these change the detections and are not recorded in the prediction: {missing}. Two runs '
        'differing in one of them would be indistinguishable from their output alone.')

    # UNCONDITIONAL, every key, always. The stamp recorded only the options that DIFFERED from
    # their defaults, because a positional list invalidated every cache on disk each time a flag
    # was added. Nothing here has that pressure, and conditional membership is what made the stamp
    # need five exceptions to its own rule.
    args2 = argparse.Namespace(detector='d.pt', det_score=0.97,
                               det_top_k=24, max_animals=2, max_frames=120,
                               frame_start=300, frame_stop=500, det_nms_iou=0.65,
                               det_nms_center_dist=0.25)
    assert set(box_provenance(args2, 1.0, True, 'instances')) == set(prov), \
        'the same keys at every value -- conditional membership is what makes a record lie'


def test_every_identity_lever_is_recorded_in_the_prediction():
    """Every `associate_group` parameter that changes which row a detection lands in is
    recorded, unconditionally: two stored 2D arms once differed only by directory name."""
    import inspect

    from tailcyclenet.infer.associate import associate_group
    from tailcyclenet.infer.cli import build_parser
    from tailcyclenet.infer.provenance import identity_provenance

    args = build_parser().parse_args(['--run', 'r', '--out', 'o'])
    prov = identity_provenance(args)

    # Everything `associate_group` takes that can change which row a detection lands in. The rest
    # is plumbing: `raw`/`session`/`gid` are the inputs themselves, `max_instances` is already
    # recorded as `max_animals` by `box_provenance`, `stats` is a diagnostic sink, `state` is
    # the block-boundary carry (recorded nowhere because it is derived, not chosen), and
    # `frame_base` is derived from `--start-frame`, already recorded as `frame_start` by
    # `box_provenance` -- it does not choose which row a detection lands in, only which
    # SOURCE frame the tracker's own step 0 is (A6a).
    plumbing = {'raw', 'session', 'gid', 'max_instances', 'stats', 'state', 'frame_base'}
    params = set(inspect.signature(associate_group).parameters) - plumbing
    # How each is spelled in the record, where the CLI name differs from the parameter name.
    alias = {'link': 'link_boxes'}
    missing = [p for p in sorted(params) if alias.get(p, p) not in prov]
    assert not missing, (
        f'these change which row a detection lands in and are not recorded in the prediction: '
        f'{missing}. Two runs differing in one of them would be indistinguishable from their '
        'output alone, which is exactly how the stored 2D suppression arms became unmatchable.')

    # UNCONDITIONAL, every key, always -- same rule as `box_provenance`: conditional membership
    # is what makes a record lie, because an absent key reads as "not used" and not as "unknown".
    args2 = build_parser().parse_args([
        '--run', 'r', '--out', 'o', '--no-track', '--no-link-boxes', '--min-views', '1',
        '--max-move', '2.0', '--max-age', '24', '--duplicate-radius', '0.9',
        '--duplicate-persist', '8', '--identity-bridge'])
    assert set(identity_provenance(args2)) == set(prov), \
        'the same keys at every value -- conditional membership is what makes a record lie'
    assert identity_provenance(args2)['max_age'] == 24
