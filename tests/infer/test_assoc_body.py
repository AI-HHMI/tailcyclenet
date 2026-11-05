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


def test_body_gate_refuses_pixel_levers_and_a_missing_body():
    from tailcyclenet.infer.associate import associate

    cg = _flat_rig()
    boxes = _boxes_at(cg, [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match='TR5'):
        associate(cg, boxes, max_res_body=0.15, body_size=1.0, max_res_sides=0.2)
    with pytest.raises(ValueError, match='positive body_size'):
        associate(cg, boxes, max_res_body=0.15, body_size=None)


def test_pixel_path_is_the_default_of_associate():
    """`associate` without max_res_body is the legacy pixel path, unchanged (the existing associate /
    track tests pin its behaviour)."""
    from tailcyclenet.infer.associate import associate

    cg = _flat_rig()
    boxes = _boxes_at(cg, [[0.0, 0.0, 0.0]])
    a = associate(cg, boxes, max_res_px=30.0)
    b = associate(cg, boxes, max_res_px=30.0, max_res_body=None, body_size=None)
    assert len(a) == len(b) == 1 and a[0]['residual'] == b[0]['residual']
    assert torch.equal(a[0]['point'], b[0]['point'])


def _ns(**kw):
    base = dict(assoc_res_max_px=None, assoc_res_max_body=None, assoc_res_max_sides=None,
                assoc_score_order=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_gate_resolution_order_and_body_length_key(tmp_path):
    """CLI px > CLI body > a session's own assoc_res_max_px key > the body-unit default; session.toml
    body_length is read; the two CLI gates and the pixel-only levers under the body gate are refused."""
    import toml

    from tailcyclenet.format import Session
    from tailcyclenet.infer.body import body_size_from_session, resolve_assoc_gate

    root = tmp_path / 'ds' / 'test' / 's'
    _session_3d(root)
    sess = Session.load(root)
    assert not sess.assoc_res_max_px_declared and sess.body_length is None
    assert body_size_from_session(sess) is None
    g = resolve_assoc_gate(_ns(), sess)
    assert g['mode'] == 'body' and g['body'] == 0.15 and g['why'] == 'default'

    cfg = toml.load(root / 'session.toml')
    cfg['assoc_res_max_px'] = 30.0
    cfg['body_length'] = 0.25
    (root / 'session.toml').write_text(toml.dumps(cfg))
    sess = Session.load(root)
    assert sess.assoc_res_max_px_declared and sess.body_length == 0.25
    assert body_size_from_session(sess) == 0.25
    g = resolve_assoc_gate(_ns(), sess)
    assert (g['mode'], g['px'], g['why']) == ('px', 30.0, 'session.toml assoc_res_max_px')
    g = resolve_assoc_gate(_ns(assoc_res_max_body=0.2), sess)
    assert (g['mode'], g['body']) == ('body', 0.2)
    g = resolve_assoc_gate(_ns(assoc_res_max_px=45.0), sess)
    assert (g['mode'], g['px']) == ('px', 45.0)
    with pytest.raises(SystemExit):
        resolve_assoc_gate(_ns(assoc_res_max_px=45.0, assoc_res_max_body=0.2), sess)
    del cfg['assoc_res_max_px']
    (root / 'session.toml').write_text(toml.dumps(cfg))
    sess = Session.load(root)
    with pytest.raises(SystemExit, match='PIXEL'):
        resolve_assoc_gate(_ns(assoc_res_max_sides=0.19), sess)
    assert resolve_assoc_gate(_ns(assoc_res_max_sides=0.19, assoc_res_max_px=30.0), sess)['mode'] == 'px'


def test_write_session_round_trips_body_length(tmp_path):
    from tailcyclenet.format import Session, write_session

    root = tmp_path / 'ds' / 'test' / 's'
    _session_3d(root)
    s = Session.load(root)
    out = tmp_path / 'o' / 'test' / 's'
    write_session(out, mode=s.mode, units=s.units, label_source=s.label_source, names=s.names, rig=s.rig,
                  groups=s.groups, labels={g: s.labels(g) for g in s.groups}, body_length=1.7)
    back = Session.load(out)
    assert back.body_length == 1.7 and not back.assoc_res_max_px_declared


def test_first_pass_body_estimator_on_a_tiny_fixture():
    """median over (row, frame) of the extent diagonal; NaN rows / frames / keypoints ignored."""
    from tailcyclenet.infer.body import estimate_body_size, first_pass_config, median_extent
    from tailcyclenet.infer.window import InferConfig

    S, T, K = 2, 5, 4
    pred = np.full((S, T, K, 3), np.nan)
    # row 0: a unit-ish body whose extent diagonal is exactly 3 (2, 2, 1); row 1: diagonal 5 (3, 4, 0)
    pred[0, :, 0] = 0.0
    pred[0, :, 1] = [2.0, 2.0, 1.0]
    pred[1, :, 0] = 10.0
    pred[1, :, 1] = [13.0, 14.0, 10.0]
    pred[1, 4] = np.nan                      # one missing frame
    assert median_extent(pred) == pytest.approx(3.0)     # 5 x 3.0 and 4 x 5.0 -> median 3.0
    assert np.isnan(median_extent(np.full((1, 2, 3, 3), np.nan)))
    body, used = estimate_body_size([pred[:, :3], pred[:, 3:]])
    assert body == pytest.approx(3.0) and used == T
    a, c = first_pass_config(argparse.Namespace(frame_start=50, frame_stop=900, max_frames=0), InferConfig(
        frame_start=50, frame_stop=900), n_frames=1000, px=30.0)
    assert (a.frame_start, a.frame_stop, c.frame_start, c.frame_stop) == (0, 300, 0, 300)
    assert a._assoc_gate['mode'] == 'px' and a._assoc_gate['px'] == 30.0
    _, c = first_pass_config(argparse.Namespace(frame_start=0, frame_stop=0, max_frames=0), InferConfig(),
                             n_frames=120, px=30.0)
    assert c.frame_stop == 120


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
