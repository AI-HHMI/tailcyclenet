"""`scripts/render.py` end to end: a prediction SESSION directory in, an mp4 out; npz refused."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import tailcyclenet.render.render as render_cli
from tailcyclenet.format import Registry, load_dataset
from tailcyclenet.infer import cli as infer_cli
from tailcyclenet.model import build_model
from test_model import SMALL


def _run_and_predict(tmp_path, cf, mode='2d'):
    """A tiny checkpoint + a source root."""
    from tailcyclenet.checkpoints import save_checkpoint, save_run_meta

    root = tmp_path / 'ds'
    if mode == '2d':
        cf._session_2d(root / 'test' / 's')
    else:
        cf._session_3d(root / 'test' / 's')
    ds = load_dataset(root)
    registry = Registry.build([ds])
    model = build_model(SMALL, n_keypoints=registry.n_keypoints)
    run = tmp_path / 'run'
    config = {'model': SMALL,
              'data': {'image_size': 64, 'min_crop_dim': 16, 'n_frames': 4,
                       'box_source': 'keypoints'}}
    save_run_meta(run, config, registry)
    save_checkpoint(run, 0, model, torch.optim.SGD(model.parameters(), lr=0.0), config)
    return root, run


def _predict(run, data, out, split='test'):
    infer_cli.main(['--run', str(run), '--data', str(data), '--split', split, '--anchor', 'none',
                    '--device', 'cpu', '--overlap', '2', '--out', str(out)])
    return out


def _decode(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


@pytest.fixture(scope='module')
def pred_2d(tmp_path_factory):
    """One 2D prediction session shared by every CLI test that only reads it."""
    import conftest as cf

    tmp = tmp_path_factory.mktemp('pred2d')
    root, run = _run_and_predict(tmp, cf, mode='2d')
    return _predict(run, root / 'test' / 's', tmp / 'pred')


def _render_one(args, out):
    render_cli.main(args + ['--out', str(out)])
    mp4s = sorted(out.glob('*.mp4'))
    assert len(mp4s) == 1, f'expected one mp4, got {[p.name for p in mp4s]}'
    return _decode(mp4s[0])


def test_a_2d_prediction_renders_with_no_data(pred_2d, tmp_path):
    """The prediction says where its own pixels are -- no --data, no --split; a range only narrows."""
    assert len(_render_one(['--pred', str(pred_2d)], tmp_path / 'all')) == 4
    ranged = _render_one(['--pred', str(pred_2d), '--start-frame', '1', '--end-frame', '3'],
                         tmp_path / 'ranged')
    assert len(ranged) == 2


def test_a_3d_prediction_renders_with_no_data(tmp_path):
    """The projection path (pred.shape[-1] == 3) run from a session directory."""
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='3d')
    out = _predict(run, root / 'test' / 's', tmp_path / 'pred')
    frames = _render_one(['--pred', str(out)], tmp_path / 'clips')
    assert len(frames) == 4
    assert any(f.std() > 0 for f in frames), 'a blank render would pass the frame-count check too'


def test_render_refusals_on_a_real_prediction(pred_2d, tmp_path):
    """Bad --cams/--groups/range, and a --data whose session id is not the recorded source."""
    import conftest as cf

    cf._session_2d(tmp_path / 'ds2' / 'test' / 'other')
    for extra, match in [
            (['--cams', 'bogus'], 'neither a camera name nor a valid index'),
            (['--groups', 'nope'], 'none of the requested groups'),
            (['--start-frame', '10', '--end-frame', '12'], 'requested range is empty'),
            (['--data', str(tmp_path / 'ds2' / 'test' / 'other')], 'was made from session')]:
        with pytest.raises(SystemExit, match=match):
            render_cli.main(['--pred', str(pred_2d), '--out', str(tmp_path / 'c'), *extra])


def test_static_skew_projection_uses_full_intrinsics(tmp_path):
    """Static 3D rendering must not route a nonzero-skew camera through aniposelib's projection."""
    from aniposelib.cameras import Camera, CameraGroup
    from tailcyclenet import format as fmt
    from tailcyclenet.render.render import project

    K = np.array([[800.0, 40.0, 320.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]])
    cam = Camera(matrix=K, dist=np.zeros(5), rvec=np.zeros(3),
                 tvec=np.array([0.0, 0.0, 50.0]), name='skew')
    cam.set_size((640, 480))
    rig = fmt.Rig(CameraGroup([cam]), offset={'skew': (7.0, -4.0)},
                  moving={'skew': False}, calibrated={'skew': True})
    sess = fmt.Session(path=tmp_path / 's', mode='3d', units='mm', label_source='tracked',
                       names=['a', 'b'], rig=rig, groups={'g': fmt.Group('g', 1)})
    world = np.array([[[[1.0, 2.0, 10.0], [-2.0, 1.0, 20.0]]]], dtype=np.float64)
    got = project(sess, world, 0)

    camera_xyz = world + np.array([0.0, 0.0, 50.0])
    homogeneous = camera_xyz @ K.T
    expected = homogeneous[..., :2] / homogeneous[..., 2, None]
    expected -= np.array([7.0, -4.0])
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2e-4)


def test_videos_and_directory_sessions_render_the_identical_pixels(tmp_path):
    """A directory session and an `adopt.build` (--videos) session over the SAME mp4 render
    byte-identical frames."""
    import conftest as cf
    from tailcyclenet.infer import adopt
    from tailcyclenet import format as fmt
    from tailcyclenet.render.render import render_group

    W, H = 64, 48
    T = 5
    tmp_path = Path(tmp_path)
    rec = tmp_path / 'rec'
    # stem 'g' -> --videos derives group id 'g' (no regex: the whole stem) and session id 'rec'
    # (the videos' common parent directory name) -- matched by hand below on the directory side.
    src = cf._write_video(rec / 'g.mp4', 0, T, (W, H))
    rig = cf._rig([('cam0', W, H, False, False, 0)])
    names = ['nose', 'tail']
    K = len(names)

    dir_sess_path = tmp_path / 'diskA' / 'test' / 'rec'
    lab = fmt.empty_labels(1, T, K, 1, mode3d=False)
    fmt.write_session(dir_sess_path, mode='2d', units='px', label_source='tracked', names=names,
                      rig=rig, groups={'g': fmt.Group('g', T, fps=20.0)}, labels={'g': lab},
                      provenance={'source': 'synthetic'})
    (dir_sess_path / 'groups' / 'g').mkdir(parents=True, exist_ok=True)
    fmt.link(dir_sess_path / 'groups' / 'g' / 'cam0.mp4', src)
    dir_sess = fmt.Session.load(dir_sess_path)

    cal = tmp_path / 'calib.toml'
    fmt.dump_calibration(cal, rig)
    plan = adopt.plan([rec], cal, None)
    video_sess = adopt.build(plan, names=names, units='px', verbose=False)
    video_gid = next(iter(video_sess.groups))
    assert dir_sess.session_id == video_sess.session_id == 'rec'
    assert video_gid == 'g'

    pred = np.random.default_rng(0).uniform(5, min(W, H) - 5, size=(1, T, K, 2)).astype(np.float32)
    frames = np.arange(T)
    out_dir = tmp_path / 'a.mp4'
    out_video = tmp_path / 'b.mp4'
    render_group(dir_sess, 'g', pred.copy(), out_dir, cam=0, frames=frames, max_side=64, fps=5)
    render_group(video_sess, video_gid, pred.copy(), out_video, cam=0, frames=frames,
                max_side=64, fps=5)

    fa, fb = _decode(out_dir), _decode(out_video)
    assert len(fa) == len(fb) == T
    for a, b in zip(fa, fb):
        np.testing.assert_array_equal(a, b)


def test_render_group_frames_selects_the_requested_source_indices(monkeypatch, tmp_path):
    """`frames=` decides which SOURCE frames are decoded."""
    import conftest as cf
    from tailcyclenet import format as fmt
    from tailcyclenet.render.render import render_group

    W, H = 64, 48
    T = 8
    rig = cf._rig([('cam0', W, H, False, False, 0)])
    names = ['nose', 'tail']
    K = len(names)
    lab = fmt.empty_labels(1, T, K, 1, mode3d=False)
    sess_path = tmp_path / 'ds' / 'test' / 's'
    fmt.write_session(sess_path, mode='2d', units='px', label_source='tracked', names=names,
                      rig=rig, groups={'g': fmt.Group('g', T, fps=20.0)}, labels={'g': lab},
                      provenance={'source': 'synthetic'})
    cf._write_frames(sess_path / 'groups' / 'g', 'cam0', T, (W, H))
    sess = fmt.Session.load(sess_path)

    seen = []
    import tailcyclenet.render.render as render_mod
    real_read = render_mod.read_frames

    def spy(group, cam, frames_arg, **kw):
        seen.append(list(int(f) for f in frames_arg))
        return real_read(group, cam, frames_arg, **kw)

    monkeypatch.setattr(render_mod, 'read_frames', spy)

    pred = np.full((1, 3, K, 2), 20.0, np.float32)
    out = tmp_path / 'ranged.mp4'
    render_group(sess, 'g', pred, out, cam=0, frames=np.arange(2, 5), max_side=64, fps=5)

    assert seen and seen[0] == [2, 3, 4], f'read_frames was asked for {seen}, not [2, 3, 4]'
    assert len(_decode(out)) == 3


@pytest.mark.parametrize('pred, match', [('old.npz', 'carries no provenance'),
                                         ('nope', 'no session.toml')])
def test_a_non_session_pred_is_refused_by_name(tmp_path, pred, match):
    if pred.endswith('.npz'):
        (tmp_path / pred).write_bytes(b'')
    with pytest.raises(SystemExit, match=match):
        render_cli.main(['--pred', str(tmp_path / pred), '--out', str(tmp_path)])


def test_eval_still_scores_an_npz(tmp_path):
    """Render dropped npz; scoring (`load_predictions`) did not."""
    from tailcyclenet.format.prediction import load_predictions

    npz = tmp_path / 'p.npz'
    np.savez(npz, __keys__=np.array(['s/g000'], dtype=object),
             **{'s/g000|pred': np.zeros((1, 2, 2, 2), np.float32)},
             __run__='r', __anchor__='none', __boxes__='labels')
    preds, meta = load_predictions(npz)
    assert 's/g000' in preds
    assert meta['anchor'] == 'none'


def test_a_prediction_with_no_provenance_needs_data(tmp_path):
    """No source_session nor source_videos -- refused naming both, pointed at --data."""
    import toml

    pred = tmp_path / 'pred'
    pred.mkdir()
    (pred / 'session.toml').write_text(toml.dumps(
        {'mode': '2d', 'units': 'px', 'labels': 'tracked', 'names': ['nose'], 'provenance': {}}))
    with pytest.raises(SystemExit, match='neither source_session nor source_videos'):
        render_cli.main(['--pred', str(pred), '--out', str(tmp_path / 'c')])


def _two_group_session(path):
    import conftest as cf
    from tailcyclenet import format as fmt

    W, H = 64, 48
    T = 4
    rig = cf._rig([('cam0', W, H, False, False, 0)])
    names = list(cf.KPTS_2D)
    K = len(names)
    labs = {}
    for gid in ('g000', 'g001'):
        lab = fmt.empty_labels(1, T, K, 1, mode3d=False, animal_ids=['a01'])
        lab.vis2d[:] = fmt.VISIBLE
        lab.points2d[..., 0, :] = np.random.default_rng(0).uniform(5, 40, size=(1, T, K, 2))
        labs[gid] = lab
    groups = {gid: fmt.Group(gid, T, fps=20.0) for gid in labs}
    fmt.write_session(path, mode='2d', units='px', label_source='tracked', names=names,
                      rig=rig, groups=groups, labels=labs,
                      provenance={'source': 'synthetic', 'run': 'legacy-run',
                                  'checkpoint': 'legacy-checkpoint.pth'})
    for gid in labs:
        cf._write_frames(path / 'groups' / gid, 'cam0', T, (W, H))


def _as_marked_prediction(path):
    """Turn `_two_group_session`'s tables into a marked prediction session's."""
    import toml

    doc = toml.loads((path / 'session.toml').read_text())
    doc.update(prediction_session=True, complete=True)
    (path / 'session.toml').write_text(toml.dumps(doc))
    (path / 'keypoints.pq').rename(path / 'points2d.pq')


def test_load_predictions_groups_filter_matches_the_unfiltered_read(tmp_path):
    from tailcyclenet.format.prediction import load_predictions

    _two_group_session(tmp_path / 'sess')
    _as_marked_prediction(tmp_path / 'sess')

    full, _ = load_predictions(tmp_path / 'sess')
    assert sorted(full) == ['sess/g000', 'sess/g001']

    filtered, _ = load_predictions(tmp_path / 'sess', groups=['g000'])
    assert list(filtered) == ['sess/g000']
    np.testing.assert_array_equal(filtered['sess/g000']['pred'], full['sess/g000']['pred'])
    assert np.isfinite(full['sess/g000']['pred']).any()


def test_deleted_prediction_readers_are_refused_by_name(tmp_path):
    """Pre-marker and --independent-windows sessions lost their readers; reading names why."""
    import toml

    from tailcyclenet.format.prediction import load_predictions

    path = tmp_path / 'sess'
    _two_group_session(path)
    with pytest.raises(ValueError, match='pre-marker prediction session'):
        load_predictions(path)
    _as_marked_prediction(path)
    doc = toml.loads((path / 'session.toml').read_text())
    doc['provenance']['independent_windows'] = True
    (path / 'session.toml').write_text(toml.dumps(doc))
    with pytest.raises(ValueError, match='--independent-windows'):
        load_predictions(path)


def test_load_predictions_refuses_ambiguous_marker_free_session(tmp_path):
    import toml
    from tailcyclenet.format.prediction import load_predictions

    path = tmp_path / 'sess'
    _two_group_session(path)
    cfg = toml.loads((path / 'session.toml').read_text())
    cfg['provenance'].pop('run')
    cfg['provenance'].pop('checkpoint')
    (path / 'session.toml').write_text(toml.dumps(cfg))

    with pytest.raises(ValueError, match='refusing ambiguous annotation data'):
        load_predictions(path)


def test_skeleton_override_validates_pairs_and_keypoint_names(tmp_path):
    from tailcyclenet.render.render import load_skeleton_override

    good = tmp_path / 'skeleton.toml'
    good.write_text('skeleton = [["a", "b"], ["b", "c"]]\n')
    assert load_skeleton_override(good, ['a', 'b', 'c']) == [['a', 'b'], ['b', 'c']]

    missing = tmp_path / 'missing.toml'
    missing.write_text('names = ["a", "b"]\n')
    with pytest.raises(SystemExit, match='top-level `skeleton`'):
        load_skeleton_override(missing, ['a', 'b'])

    malformed = tmp_path / 'malformed.toml'
    malformed.write_text('skeleton = [["a"], ["a", 3]]\n')
    with pytest.raises(SystemExit, match=r'skeleton\[0\]'):
        load_skeleton_override(malformed, ['a', 'b'])

    unknown = tmp_path / 'unknown.toml'
    unknown.write_text('skeleton = [["a", "missing"]]\n')
    with pytest.raises(SystemExit, match='not in the prediction keypoint axis'):
        load_skeleton_override(unknown, ['a', 'b'])


def test_render_skeleton_override_replaces_resolved_session(monkeypatch, tmp_path):
    import types
    import tailcyclenet.render.render as render
    import tailcyclenet.format.prediction as predictions

    pred = tmp_path / 'pred'
    pred.mkdir()
    (pred / 'session.toml').write_text('names = ["a", "b"]\n[provenance]\n')
    skeleton = tmp_path / 'skeleton.toml'
    skeleton.write_text('skeleton = [["a", "b"]]\n')
    sess = types.SimpleNamespace(
        names=['a', 'b'], skeleton=[], session_id='s', cam_names=['cam0'],
        groups={'g': types.SimpleNamespace(n_frames=1)})
    monkeypatch.setattr(render, 'session_for_prediction', lambda *args, **kwargs: sess)
    monkeypatch.setattr(predictions, 'load_predictions',
                        lambda *args, **kwargs: ({'s/g': {'pred': np.zeros((1, 1, 2, 2), np.float32)}}, {}))
    seen = {}
    monkeypatch.setattr(render, 'render_group',
                        lambda session, *args, **kwargs: seen.setdefault('skeleton', session.skeleton))

    render.main(['--pred', str(pred), '--out', str(tmp_path / 'out'), '--skeleton', str(skeleton)])
    assert seen['skeleton'] == [['a', 'b']]


def test_render_writes_h264_yuv420p(tmp_path):
    """Rendered clips are H.264/yuv420p (web-playable), and an odd size is padded, not refused."""
    import av

    from tailcyclenet.render.h264 import H264Writer

    path = tmp_path / 'clip.mp4'
    rng = np.random.default_rng(0)
    with H264Writer(path, 12.5, (33, 21)) as w:
        for _ in range(5):
            w.write(rng.integers(0, 255, (21, 33, 3), dtype=np.uint8))
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        assert st.codec_context.name == 'h264'
        assert st.codec_context.pix_fmt == 'yuv420p'
        assert (st.codec_context.width, st.codec_context.height) == (34, 22)
        assert st.average_rate == Fraction(25, 2)
        assert sum(1 for _ in c.decode(st)) == 5
