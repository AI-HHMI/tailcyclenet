"""The body-unit association gate (`assoc_res_max_body`): world-unit residuals / body size, the gate's
resolution order, and the label-free body-size estimator."""
import argparse

import numpy as np
import pytest
import torch

from tests.helpers import _flat_rig, _session_3d


def _boxes_at(cg, points, half=20.0, shift=None):
    """Per camera, one square box per world point centred on its projection; `shift[c] = (dx, dy)` moves
    camera c's boxes by that many pixels."""
    from tailcyclenet.infer.track import _project

    out = []
    for c, cam in enumerate(cg):
        rows = []
        for p in points:
            u, v = _project(cam, np.asarray(p, np.float32).reshape(1, 3))[0].tolist()
            if shift and c in shift:
                u, v = u + shift[c][0], v + shift[c][1]
            rows.append([u - half, v - half, u + half, v + half])
        out.append(torch.tensor(rows, dtype=torch.float32))
    return out


def test_body_residual_is_a_world_distance_over_body_size():
    """Known geometry: cameras 900 units from the origin with f = 800 px, so a d-px centre offset in one view
    moves that view's ray ~d * 900 / 800 world units off the point. At body_size 1000 and the default gate
    0.15 (150 world units), an 80-unit offset still joins the group and a 250-unit one does not -- both far
    beyond what the legacy 30 px gate would admit."""
    from tailcyclenet.infer.associate import ASSOC_RES_MAX_BODY, associate

    assert ASSOC_RES_MAX_BODY == 0.15
    cg = _flat_rig()
    origin = [[0.0, 0.0, 0.0]]
    for world_off, joins in ((80.0, True), (250.0, False)):
        px = world_off * 800.0 / 900.0
        boxes = _boxes_at(cg, origin, shift={2: (px, 0.0)})
        g = associate(cg, boxes, max_res_body=ASSOC_RES_MAX_BODY, body_size=1000.0)
        assert len(g) == 1 and set(g[0]['boxes']) >= {0, 1}
        assert (2 in g[0]['boxes']) == joins, (world_off, sorted(g[0]['boxes']))
        assert g[0]['residual'] <= 0.15 * 1000.0, 'the refit passed the same world-unit gate'
        # the legacy pixel gate on the same boxes admits no third view at 30 px
        gp = associate(cg, boxes, max_res_px=30.0)
        assert 2 not in gp[0]['boxes']


def test_body_gate_separates_two_animals_and_scales_with_body_size():
    """Two animals 400 units apart: two groups at body 1000 (gate 150); one body unit is the scale, so the
    same boxes at body 10 (gate 1.5 units) still give two exact groups."""
    from tailcyclenet.infer.associate import associate

    cg = _flat_rig()
    pts = [[-200.0, 0.0, 0.0], [200.0, 0.0, 0.0]]
    boxes = _boxes_at(cg, pts)
    for body in (1000.0, 10.0):
        g = associate(cg, boxes, max_res_body=0.15, body_size=body)
        assert len(g) == 2 and all(len(x['boxes']) == 3 for x in g)
        got = sorted(float(x['point'][0]) for x in g)
        assert np.allclose(got, [-200.0, 200.0], atol=1e-2), got


def test_body_gate_refuses_a_missing_body_and_ports_score_order():
    """A body gate needs a positive body size; TR6a's score order is gate-independent and ranks the body path
    exactly as the pixel path: same groups on a scene where the two orders agree."""
    from tailcyclenet.infer.associate import associate

    cg = _flat_rig()
    boxes = _boxes_at(cg, [[-200.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match='positive body_size'):
        associate(cg, boxes, max_res_body=0.15, body_size=None)
    scores = [torch.tensor([0.9, 0.2]) for _ in boxes]
    a = associate(cg, boxes, max_res_body=0.15, body_size=1000.0)
    b = associate(cg, boxes, max_res_body=0.15, body_size=1000.0, scores_per_cam=scores, score_order=True)
    assert [sorted(g['members'].items()) for g in a] == [sorted(g['members'].items()) for g in b]
    with pytest.raises(ValueError, match='score_order needs scores'):
        associate(cg, boxes, max_res_body=0.15, body_size=1000.0, score_order=True)


def test_pixel_path_is_the_internal_default_of_associate():
    """`associate` without max_res_body is the internal pixel path, unchanged (the existing associate / track
    tests pin its behaviour)."""
    from tailcyclenet.infer.associate import associate

    cg = _flat_rig()
    boxes = _boxes_at(cg, [[0.0, 0.0, 0.0]])
    a = associate(cg, boxes, max_res_px=30.0)
    b = associate(cg, boxes, max_res_px=30.0, max_res_body=None, body_size=None)
    assert len(a) == len(b) == 1 and a[0]['residual'] == b[0]['residual']
    assert torch.equal(a[0]['point'], b[0]['point'])


def test_gate_is_the_cli_value_else_the_default():
    from tailcyclenet.infer.body import resolve_assoc_gate

    g = resolve_assoc_gate(argparse.Namespace(assoc_res_max_body=None))
    assert g == dict(mode='body', body=0.15, why='default')
    g = resolve_assoc_gate(argparse.Namespace(assoc_res_max_body=0.2))
    assert (g['mode'], g['body'], g['why']) == ('body', 0.2, '--assoc-res-max-body')


def test_removed_session_keys(tmp_path, capsys):
    """body_length RAISES by name; a leftover assoc_res_max_px is ignored with one warning per process (silently
    in a prediction session); neither is an attribute any more."""
    import toml

    from tailcyclenet.format import FormatError, Session
    from tailcyclenet.format import session as S

    root = tmp_path / 'ds' / 'test' / 's'
    _session_3d(root)
    cfg = toml.load(root / 'session.toml')
    cfg['body_length'] = 0.25
    (root / 'session.toml').write_text(toml.dumps(cfg))
    with pytest.raises(FormatError, match='body_length'):
        Session.load(root)
    del cfg['body_length']
    cfg['assoc_res_max_px'] = 30.0
    (root / 'session.toml').write_text(toml.dumps(cfg))
    S._WARNED_LEGACY.clear()
    s = Session.load(root)
    err = capsys.readouterr().err
    assert 'assoc_res_max_px` is ignored' in err and '2026-11-05' in err
    assert not hasattr(s, 'assoc_res_max_px') and not hasattr(s, 'body_length')
    Session.load(root)
    assert 'assoc_res_max_px' not in capsys.readouterr().err, 'warned once per process'


def test_first_pass_segments_stay_inside_the_run_range():
    """K windows spread over [start, stop), first at start and last ending at stop; a range holding no more
    than K windows is used whole; nothing outside the range is ever read."""
    from tailcyclenet.infer.body import FIRST_PASS_WINDOWS, first_pass_segments, run_range
    from tailcyclenet.infer.window import InferConfig

    assert FIRST_PASS_WINDOWS == 4
    seg = first_pass_segments(100, 1100, 12)
    assert len(seg) == 4 and seg[0] == (100, 112) and seg[-1] == (1088, 1100)
    assert all(100 <= s < e <= 1100 and e - s == 12 for s, e in seg)
    assert first_pass_segments(0, 40, 12) == [(0, 40)]
    assert first_pass_segments(5, 6, 12) == []
    assert run_range(InferConfig(frame_start=50, frame_stop=900), 1000) == (50, 900)
    assert run_range(InferConfig(max_frames=120), 1000) == (0, 120)
    assert run_range(InferConfig(), 70) == (0, 70)


def test_first_pass_body_estimator_on_a_tiny_fixture():
    """median over (row, frame) of the extent diagonal; NaN rows / frames / keypoints ignored."""
    from tailcyclenet.infer.body import estimate_body_size, first_pass_config, median_extent
    from tailcyclenet.infer.window import InferConfig

    S, T, K = 2, 5, 4
    pred = np.full((S, T, K, 3), np.nan)
    pred[0, :, 0] = 0.0
    pred[0, :, 1] = [2.0, 2.0, 1.0]
    pred[1, :, 0] = 10.0
    pred[1, :, 1] = [13.0, 14.0, 10.0]
    pred[1, 4] = np.nan
    assert median_extent(pred) == pytest.approx(3.0)
    assert np.isnan(median_extent(np.full((1, 2, 3, 3), np.nan)))
    body, used = estimate_body_size([pred[:, :3], pred[:, 3:]])
    assert body == pytest.approx(3.0) and used == T
    a, c = first_pass_config(argparse.Namespace(frame_start=0, frame_stop=0, max_frames=500), InferConfig(),
                             200, 212)
    assert (a.frame_start, a.frame_stop, a.max_frames) == (200, 212, 0)
    assert (c.frame_start, c.frame_stop, c.anchor) == (200, 212, 'none')
    assert a._assoc_gate == dict(mode='body', body=0.15, why='first pass (provisional body)')
    assert a._body_size is None, 'the first pass bootstraps its body size from the detections'


def test_cli_body_length_skips_the_first_pass(monkeypatch):
    """--body-length is the body size; no detection or pose pass runs."""
    from tailcyclenet.infer import driver

    def boom(*a, **k):
        raise AssertionError('the first pass must not run')

    monkeypatch.setattr(driver, 'detector_boxes', boom)
    monkeypatch.setattr(driver, 'run_blocks', boom)
    args = argparse.Namespace(body_length=1.6, _assoc_gate=dict(mode='body', px=30.0, body=0.15, why='default'),
                              _body_size=None, _body_size_source='')
    sess = argparse.Namespace(mode='3d', cam_names=['a', 'b'], units='m')
    driver._resolve_body_size(args, sess, ['g'], None, None, None, None, {}, object(), None, None, None, None,
                              'cpu', None, None)
    assert args._body_size == 1.6 and args._body_size_source == 'CLI --body-length'


def test_associate_group_threads_the_body_gate(tmp_path):
    """`associate_group` passes the body gate to the tracker and the per-frame path alike."""
    from tailcyclenet.format import Session
    from tailcyclenet.infer.associate import associate_group

    root = tmp_path / 'ds' / 'test' / 's'
    _session_3d(root)
    sess = Session.load(root)
    sess.preload()
    gid = sorted(sess.groups)[0]
    C = len(sess.cam_names)
    centre = np.array([24., 16., 40., 32.], np.float32)
    T = 3
    boxes = np.tile(centre, (1, T, C, 1))
    scores = np.ones((1, T, C), np.float32)
    for track in (True, False):
        st = {}
        b, _ = associate_group((boxes, scores), sess, gid, 1, track=track, state=st,
                               assoc_res_max_body=0.15, body_size=1e6)
        assert np.isfinite(b).all(), 'one animal seen in every camera is one row in every camera'
        if track:
            assert st['tracker'].max_res_body == 0.15 and st['tracker'].body_size == 1e6


def test_provisional_body_size_from_detections_alone():
    """Two animals of known size on the synthetic rig: boxes whose diagonal x depth / focal equals the size give
    that size back from mutual-nearest pairs (no pose, no pixel gate); no detections -> NaN."""
    from tailcyclenet.infer.associate import provisional_body_size

    cg = _flat_rig()
    pts = [[-200.0, 0.0, 0.0], [200.0, 0.0, 0.0]]
    half = 40.0
    boxes = _boxes_at(cg, pts, half=half)
    raw_box = np.stack([np.stack([b[i].numpy() for b in boxes]) for i in range(2)])[:, None]    # D, T=1, C, 4
    raw_sc = np.ones(raw_box.shape[:3], np.float32)
    est, n = provisional_body_size(raw_box, raw_sc, lambda t: cg, [0])
    # every view's box-implied size: diagonal (2 half sqrt 2 px) x that camera's depth of the point / focal 800
    depth = [float((c['ext'][:3, :3].double().numpy() @ np.asarray(p) + c['ext'][:3, 3].double().numpy())[2])
             for c in cg for p in pts]
    sizes = 2 * half * np.sqrt(2) * np.asarray(depth) / 800.0
    assert n >= 2 and sizes.min() * 0.99 <= est <= sizes.max() * 1.01
    est, n = provisional_body_size(raw_box, np.full_like(raw_sc, np.nan), lambda t: cg, [0])
    assert n == 0 and np.isnan(est)


def test_associate_group_bootstraps_the_body_size_and_has_no_pixel_default(tmp_path):
    """body_size None -> a provisional body from the detections, recorded in stats; associate itself has no
    default pixel gate."""
    from tailcyclenet.format import Session
    from tailcyclenet.infer.associate import associate, associate_group

    root = tmp_path / 'ds' / 'test' / 's'
    _session_3d(root)
    sess = Session.load(root)
    sess.preload()
    gid = sorted(sess.groups)[0]
    C = len(sess.cam_names)
    T = 3
    boxes = np.tile(np.array([24., 16., 40., 32.], np.float32), (1, T, C, 1))
    scores = np.ones((1, T, C), np.float32)
    st, stats = {}, {}
    b, _ = associate_group((boxes, scores), sess, gid, 1, state=st, stats=stats)
    assert stats['body_provisional'] > 0 and st['tracker'].body_size == stats['body_provisional']
    assert np.isfinite(b).all()
    with pytest.raises(ValueError, match='no default pixel gate'):
        associate(_flat_rig(), _boxes_at(_flat_rig(), [[0.0, 0.0, 0.0]]))


def test_triangulate_group_gates_in_body_units(tmp_path):
    """The GT triangulation's gate is the point-to-ray distance / the group's own body size."""
    import tailcyclenet.format as fmt
    from tests.format.test_format import _project_3d_labels

    assert fmt.VALIDATION_RES_MAX_BODY == 0.05
    root = tmp_path / 's'
    _session_3d(root)
    s = fmt.Session.load(root)
    lab = s.labels('g000')
    _project_3d_labels(s, lab)
    vis, gated, _ = fmt.triangulate_group(s.rig, lab)
    assert gated == 0 and vis > 0, 'exact projections pass the default gate'
    lab.points2d[0, 0, 0, 1:] += 30.0                 # two of three views pulled far off
    _, gated, _ = fmt.triangulate_group(s.rig, lab)
    assert gated >= 1 and lab.vis3d[0, 0, 0] == fmt.UNLABELED
