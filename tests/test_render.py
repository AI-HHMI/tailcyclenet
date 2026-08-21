"""`scripts/render.py` on a real prediction session, from both `--data` and `--videos` runs.

Companion to `tests/test_infer.py`'s `test_render_writes_every_predicted_frame` /
`test_a_3d_render_uses_the_per_frame_camera_on_a_moving_rig`, which pin `render_group` itself
against `run_group`'s raw output. This file pins the CLI end to end: a prediction SESSION
directory in, an mp4 out, with no `--data` required and an npz refused outright.
See dev/plans/render_a_prediction_session.md.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from tailcyclenet.format import Registry, load_dataset
from tailcyclenet.model import build_model
from test_model import SMALL


def _load_script(name):
    """`scripts/{name}.py` as a module, without running main() -- test_infer.py's `cli` pattern."""
    spec = importlib.util.spec_from_file_location(
        f'tcn_{name}', Path(__file__).resolve().parent.parent / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def infer_cli():
    return _load_script('infer')


@pytest.fixture(scope='module')
def render_cli():
    return _load_script('render')


def _run_and_predict(tmp_path, cf, mode='2d', moving=False):
    """A tiny checkpoint + a prediction session, through the real CLI both ways."""
    from tailcyclenet.checkpoints import save_checkpoint, save_run_meta

    root = tmp_path / 'ds'
    if mode == '2d':
        cf._session_2d(root / 'test' / 's')
    else:
        cf._session_3d(root / 'test' / 's', moving=moving)
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


def _predict(infer_cli, monkeypatch, run, data, out, split='test', extra=()):
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(data),
                                      '--split', split, '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--out', str(out), *extra])
    infer_cli.main()
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


# ---------------------------------------------------------------------------------------------
# 1-2. rendering a real prediction session, in both modes, with no --data


def test_a_2d_prediction_renders_with_no_data(infer_cli, render_cli, monkeypatch, tmp_path):
    """The prediction says where its own pixels are -- no --data, no --split, nothing restated."""
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='2d')
    out = _predict(infer_cli, monkeypatch, run, root / 'test' / 's', tmp_path / 'pred')

    clips = tmp_path / 'clips'
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(out), '--out', str(clips)])
    render_cli.main()

    mp4s = sorted(clips.glob('*.mp4'))
    assert len(mp4s) == 1, f'expected one mp4, got {[p.name for p in mp4s]}'
    frames = _decode(mp4s[0])
    assert len(frames) == 4, f'wrote {len(frames)} frames for a 4-frame prediction'


def test_a_3d_prediction_renders_with_no_data(infer_cli, render_cli, monkeypatch, tmp_path):
    """The projection path (pred.shape[-1] == 3) run from a session DIRECTORY, not from
    `run_group`'s raw array -- `test_render_writes_every_predicted_frame` skips 3D entirely."""
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='3d')
    out = _predict(infer_cli, monkeypatch, run, root / 'test' / 's', tmp_path / 'pred')

    clips = tmp_path / 'clips'
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(out), '--out', str(clips)])
    render_cli.main()

    mp4s = sorted(clips.glob('*.mp4'))
    assert len(mp4s) == 1
    frames = _decode(mp4s[0])
    assert len(frames) == 4
    assert any(f.std() > 0 for f in frames), 'a blank render would pass the frame-count check too'


# ---------------------------------------------------------------------------------------------
# 3. byte identity: a directory session and a --videos session reading the SAME file


def test_videos_and_directory_sessions_render_the_identical_pixels(tmp_path):
    """THE ONE TEST THAT MAKES `--videos` REAL FOR RENDERING.

    Two `Session` objects -- one hand-authored on disk, one built by `adopt.build` from raw
    footage -- pointed at the literal SAME mp4, drawing the literal same prediction array, must
    decode to identical frames. Session id, group id and camera name are made to agree on
    purpose: `render_group` burns them into the caption, and a caption-text difference is not the
    risk this test exists to catch -- pixel decode agreement is.
    """
    import conftest as cf
    from tailcyclenet import adopt
    from tailcyclenet import format as fmt
    from tailcyclenet.render import render_group

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


# ---------------------------------------------------------------------------------------------
# 4. the frame range


def test_render_group_frames_selects_the_requested_source_indices(monkeypatch, tmp_path):
    """A unit test on `frames=`: it decides which SOURCE frames are decoded and what the caption
    reads, independent of the CLI."""
    import conftest as cf
    from tailcyclenet import format as fmt
    from tailcyclenet.render import render_group

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
    import tailcyclenet.render as render_mod
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


def test_the_cli_start_end_frame_narrows_the_predicted_range(infer_cli, render_cli, monkeypatch,
                                                              tmp_path):
    """A `--start-frame`/`--end-frame` run renders EXACTLY the frames it predicted with no flags
    at all, and an explicit flag can only narrow that further -- never extend into NaN frames."""
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='2d')
    out = _predict(infer_cli, monkeypatch, run, root / 'test' / 's', tmp_path / 'pred')

    clips = tmp_path / 'clips'
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(out), '--out', str(clips),
                                      '--start-frame', '1', '--end-frame', '3'])
    render_cli.main()
    mp4s = sorted(clips.glob('*.mp4'))
    assert len(mp4s) == 1
    assert len(_decode(mp4s[0])) == 2, '--start-frame 1 --end-frame 3 must draw exactly 2 frames'


# ---------------------------------------------------------------------------------------------
# 6. the refusals


def test_an_npz_is_refused_by_name(render_cli, monkeypatch, tmp_path):
    npz = tmp_path / 'old.npz'
    npz.write_bytes(b'')
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(npz), '--out', str(tmp_path)])
    with pytest.raises(SystemExit, match='carries no provenance'):
        render_cli.main()


def test_eval_still_scores_an_npz(tmp_path):
    """THE OTHER HALF OF THE SAME CLAIM: render dropped npz, scoring did not.

    `load_predictions` is the shared reader both `render.py` and `scripts/eval.py` used to call;
    this pins that `eval.py`'s caller still works on an npz after render.py stopped accepting one.
    """
    from tailcyclenet.infer.predictions import load_predictions

    npz = tmp_path / 'p.npz'
    np.savez(npz, __keys__=np.array(['s/g000'], dtype=object),
             **{'s/g000|pred': np.zeros((1, 2, 2, 2), np.float32)},
             __run__='r', __anchor__='none', __boxes__='labels')
    preds, meta = load_predictions(npz)
    assert 's/g000' in preds
    assert meta['anchor'] == 'none'


def test_not_a_directory_is_refused(render_cli, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(tmp_path / 'nope'),
                                      '--out', str(tmp_path)])
    with pytest.raises(SystemExit, match='no session.toml'):
        render_cli.main()


def test_an_unknown_camera_token_is_refused(infer_cli, render_cli, monkeypatch, tmp_path):
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='2d')
    out = _predict(infer_cli, monkeypatch, run, root / 'test' / 's', tmp_path / 'pred')
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(out), '--out', str(tmp_path / 'c'),
                                      '--cams', 'bogus'])
    with pytest.raises(SystemExit, match='neither a camera name nor a valid index'):
        render_cli.main()


def test_an_unknown_group_is_refused(infer_cli, render_cli, monkeypatch, tmp_path):
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='2d')
    out = _predict(infer_cli, monkeypatch, run, root / 'test' / 's', tmp_path / 'pred')
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(out), '--out', str(tmp_path / 'c'),
                                      '--groups', 'nope'])
    with pytest.raises(SystemExit, match='none of the requested groups'):
        render_cli.main()


def test_an_empty_range_is_refused(infer_cli, render_cli, monkeypatch, tmp_path):
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='2d')
    out = _predict(infer_cli, monkeypatch, run, root / 'test' / 's', tmp_path / 'pred')
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(out), '--out', str(tmp_path / 'c'),
                                      '--start-frame', '10', '--end-frame', '12'])
    with pytest.raises(SystemExit, match='requested range is empty'):
        render_cli.main()


def test_a_prediction_over_the_wrong_root_is_refused(infer_cli, render_cli, monkeypatch, tmp_path):
    """The recorded `source_session_id` must match the resolved session's own -- whether reached
    through provenance or through an explicit `--data` override."""
    import conftest as cf

    root, run = _run_and_predict(tmp_path, cf, mode='2d')
    out = _predict(infer_cli, monkeypatch, run, root / 'test' / 's', tmp_path / 'pred')

    other_root = tmp_path / 'ds2'
    cf._session_2d(other_root / 'test' / 'other')
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(out),
                                      '--data', str(other_root / 'test' / 'other'),
                                      '--out', str(tmp_path / 'clips')])
    with pytest.raises(SystemExit, match='was made from session'):
        render_cli.main()


def test_a_prediction_with_no_provenance_needs_data(render_cli, monkeypatch, tmp_path):
    """A hand-written or pre-provenance prediction has neither source_session nor source_videos
    -- refused naming both, pointed at --data."""
    import toml

    pred = tmp_path / 'pred'
    pred.mkdir()
    (pred / 'session.toml').write_text(toml.dumps(
        {'mode': '2d', 'units': 'px', 'labels': 'tracked', 'names': ['nose'], 'provenance': {}}))
    monkeypatch.setattr(sys, 'argv', ['render.py', '--pred', str(pred), '--out', str(tmp_path / 'c')])
    with pytest.raises(SystemExit, match='neither source_session nor source_videos'):
        render_cli.main()


# ---------------------------------------------------------------------------------------------
# 8. load_predictions(groups=...)


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
                      rig=rig, groups=groups, labels=labs, provenance={'source': 'synthetic'})
    for gid in labs:
        cf._write_frames(path / 'groups' / gid, 'cam0', T, (W, H))


def test_load_predictions_groups_filter_matches_the_unfiltered_read(tmp_path):
    from tailcyclenet.infer.predictions import load_predictions

    _two_group_session(tmp_path / 'sess')

    full, _ = load_predictions(tmp_path / 'sess')
    assert sorted(full) == ['sess/g000', 'sess/g001']

    filtered, _ = load_predictions(tmp_path / 'sess', groups=['g000'])
    assert list(filtered) == ['sess/g000']
    np.testing.assert_array_equal(filtered['sess/g000']['pred'], full['sess/g000']['pred'])
