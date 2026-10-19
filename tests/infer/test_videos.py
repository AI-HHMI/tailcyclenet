"""Inference straight off raw footage (`--videos`) through the CLI."""
import sys

import numpy as np
import pytest

from tailcyclenet.format import Registry
from tests.helpers import KPTS_3D, _rig, _save_run, _write_video

# --videos: raw footage instead of a session directory.

def _videos_fixture(tmp_path, T=8, wh=(64, 48)):
    """Two calibrated cameras, videos named `cam<i>_t.mp4`, plus a hand-authored session
    directory over THE SAME FILES. Everything a `--videos` run and its `--data` twin need."""
    from tailcyclenet import format as fmt

    W, H = wh
    rig = _rig([(str(i), W, H, True, False, i + 1) for i in range(2)])
    fmt.dump_calibration(tmp_path / 'calib.toml', rig)
    vids = {str(i): _write_video(tmp_path / 'rec' / f'cam{i}_t.mp4', i, T, wh)
            for i in range(2)}

    # The `--data` twin: the same videos, symlinked into a session directory.
    sdir = tmp_path / 'ds' / 'test' / 'rec'
    groups = {'t': fmt.Group('t', T, fps=20.0)}
    fmt.write_session(sdir, mode='3d', units='mm', label_source='tracked', names=KPTS_3D,
                      rig=rig, groups=groups,
                      labels={'t': fmt.empty_labels(0, T, len(KPTS_3D), 2, mode3d=True)})
    (sdir / 'groups' / 't').mkdir(parents=True)
    for cam, src in vids.items():
        fmt.link(sdir / 'groups' / 't' / f'{cam}.mp4', src)
    return rig, vids, sdir


def _videos_run(tmp_path, names):
    """A run folder whose registry holds exactly ONE dataset, so the default has to infer it."""
    registry = Registry(names=tuple(names), datasets=(('ds', tuple(range(len(names)))),))
    return _save_run(tmp_path, registry)[0], registry


def test_overlap_trace_flag_actually_reaches_the_capture_hook(cli, monkeypatch, tmp_path):
    """`--overlap-trace` must ARM the hook, not merely write a file: an all-zero census is the silent
    failure that once cost a GPU run."""
    import json


    rig, vids, sdir = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, KPTS_3D)
    pts = np.random.default_rng(0).uniform(-20, 20,
                                           size=(2, 8, len(KPTS_3D), 3)).astype(np.float32)
    np.savez(tmp_path / 'boxes.npz', **{'rec/t': pts})

    trace = tmp_path / 'ovl' / 'trace.npz'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--data', str(sdir), '--out', str(tmp_path / 'o'),
        '--run', str(run), '--boxes', str(tmp_path / 'boxes.npz'), '--anchor', 'carry',
        '--device', 'cpu', '--overlap', '2', '--dataset-name', 'ds',
        '--overlap-trace', str(trace)])
    cli.main()

    assert trace.exists(), '--overlap-trace must write its file'
    z = np.load(trace, allow_pickle=True)
    census = json.loads(str(z['census']))
    assert census, 'the census must name the group(s) that ran'
    seen = sum(c.get('pairs_seen', 0) for c in census.values())
    assert seen > 0, (
        f'the hook never fired: census {census}. A multi-window run at --overlap 2 has seams, so '
        'zero pairs means the flag did not reach `stats`, not that there was nothing to see.')
    assert len(z['dist']) == seen, 'the rows and the census must describe the same population'
    assert z['prev'].shape == z['new'].shape and z['prev'].shape[0] == seen


def test_the_videos_path_is_byte_identical_to_the_session_path(cli, monkeypatch, tmp_path):
    """`--videos` is an input path, not a second pipeline: same points3d.pq as the `--data` twin."""

    rig, vids, sdir = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, KPTS_3D)

    pts = np.random.default_rng(0).uniform(-20, 20,
                                           size=(2, 8, len(KPTS_3D), 3)).astype(np.float32)
    np.savez(tmp_path / 'boxes.npz', **{'rec/t': pts})

    common = ['--run', str(run), '--boxes', str(tmp_path / 'boxes.npz'), '--anchor', 'none',
              '--device', 'cpu', '--overlap', '2', '--dataset-name', 'ds', '--units', 'mm']
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--data', str(sdir),
                                      '--out', str(tmp_path / 'a')] + common)
    cli.main()
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--videos', str(tmp_path / 'rec'),
                                      '--calibration', str(tmp_path / 'calib.toml'),
                                      '--cam-regex', 'cam([0-9]+)_', '--group-id', 't',
                                      '--session-id', 'rec',
                                      '--out', str(tmp_path / 'b')] + common)
    cli.main()

    import pyarrow.parquet as pq
    cols = ['frame', 'animal_id', 'bodypart', 'x', 'y', 'z']

    def rows(d):
        t = pq.read_table(tmp_path / d / 'points3d.pq').select(cols).to_pydict()
        return sorted(zip(*(t[c] for c in cols)))

    a, b = rows('a'), rows('b')
    assert a and a == b, '--videos must be an input path, not a second pipeline'


def test_2d_videos_without_calibration_defaults_to_px_and_nominal_cam(cli, monkeypatch, tmp_path):
    """The raw single-view path derives mode from one video, defaults units to pixels, and
    records the converter-compatible nominal cam0 using the video's dimensions."""
    import tomllib

    from tailcyclenet.format import Session

    wh, T = (80, 48), 8
    video = _write_video(tmp_path / 'rec' / 'take.mp4', 0, T, wh)
    names = KPTS_3D
    run, _ = _videos_run(tmp_path, names)
    points = np.zeros((1, T, len(names), 2), np.float32)
    points[..., 0] = wh[0] / 2
    points[..., 1] = wh[1] / 2
    np.savez(tmp_path / 'boxes.npz', **{'rec/take': points})

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--run', str(run), '--videos', str(video), '--boxes', str(tmp_path / 'boxes.npz'),
        '--anchor', 'none', '--device', 'cpu', '--overlap', '2', '--dataset-name', 'ds',
        '--out', str(out)])
    cli.main()

    got = Session.load(out)
    assert got.mode == '2d' and got.units == 'px'
    assert got.cam_names == ['cam0'] and got.rig.size('cam0') == wh
    from tailcyclenet.infer import adopt
    back = adopt.session_from_prediction(out)
    assert back.cam_names == ['cam0'] and back.rig.size('cam0') == wh
    with open(out / 'session.toml', 'rb') as f:
        cfg = tomllib.load(f)
    assert cfg['units'] == 'px'


@pytest.mark.parametrize('units', [None, 'px'])
def test_3d_videos_without_valid_units_refuses_before_checkpoint_load(cli, monkeypatch, tmp_path,
                                                                     units):
    """A multiview calibration determines 3D mode, but cannot determine units: the operator
    must declare them explicitly rather than inheriting a misleading default."""

    rig, vids, _ = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, KPTS_3D)
    argv = [
        'infer.py', '--run', str(run), '--videos', str(tmp_path / 'rec'),
        '--calibration', str(tmp_path / 'calib.toml'), '--cam-regex', 'cam([0-9]+)_',
        '--group-id', 't', '--session-id', 'rec', '--boxes', str(tmp_path / 'missing.npz'),
        '--anchor', 'none', '--device', 'cpu', '--out', str(tmp_path / 'pred')]
    if units is not None:
        argv += ['--units', units]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit, match='--units'):
        cli.main()
    assert not (tmp_path / 'pred').exists(), 'the refusal must precede output/checkpoint work'


def test_the_cli_runs_end_to_end_from_videos(cli, monkeypatch, tmp_path):
    """Through `main()` from raw videos: the prediction validates, `source_*` provenance is verbatim,
    `--dump-session` writes a valid session, and `session_from_prediction` finds the pixels."""
    import tomllib

    from tailcyclenet.format import Session, validate_session
    from tailcyclenet.format.prediction import load_predictions

    rig, vids, sdir = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, KPTS_3D)
    pts = np.random.default_rng(1).uniform(-20, 20,
                                           size=(1, 8, len(KPTS_3D), 3)).astype(np.float32)
    np.savez(tmp_path / 'boxes.npz', **{'rec/t': pts})

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--run', str(run), '--videos', str(tmp_path / 'rec'),
        '--calibration', str(tmp_path / 'calib.toml'), '--cam-regex', 'cam([0-9]+)_',
        '--group-id', 't', '--session-id', 'rec', '--units', 'mm',
        '--boxes', str(tmp_path / 'boxes.npz'), '--anchor', 'none', '--device', 'cpu',
        '--overlap', '2', '--dump-session', str(tmp_path / 'dumped'), '--out', str(out)])
    cli.main()

    got = Session.load(out)
    assert got.mode == '3d' and got.units == 'mm' and got.names == list(KPTS_3D)
    errs = [e for e in validate_session(got) if '[rule 7]' not in e]
    assert not errs, f'a prediction session must be valid apart from its missing pixels: {errs}'

    with open(out / 'session.toml', 'rb') as f:
        prov = tomllib.load(f)['provenance']
    assert prov['source'] == 'tailcyclenet infer --videos'
    assert prov['source_session'] == ''
    assert prov['source_calibration'] == str(tmp_path / 'calib.toml')
    assert prov['source_cam_regex'] == 'cam([0-9]+)_'
    assert sorted(prov['source_videos']) == sorted(str(v) for v in vids.values())

    preds, _ = load_predictions(out)
    assert 'rec/t' in preds and np.isfinite(preds['rec/t']['pred']).any()

    assert not validate_session(Session.load(tmp_path / 'dumped'))

    from tailcyclenet.infer import adopt
    back = adopt.session_from_prediction(out)
    assert back.groups['t'].source('0')[1] == vids['0'].resolve()


@pytest.mark.parametrize('argv,expect', [
    ([], 'exactly one of --data'),
    (['--data', '/tmp/x', '--videos', '/tmp/y.mp4'], 'not allowed with'),
    (['--videos', '/tmp/y.mp4'], 'needs a box source'),
    (['--data', '/tmp/x', '--calibration', '/tmp/c.toml'], 'only means anything with --videos'),
])
def test_the_two_input_paths_are_exclusive_and_both_named(cli, monkeypatch, capsys, argv, expect):
    """`--data`/`--videos` are exactly-one-of and the refusal names both."""
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', '/tmp/r', '--out', '/tmp/o'] + argv)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    captured = capsys.readouterr().err
    err = captured or str(exc.value)
    assert expect in err, f'wanted {expect!r}, got: {err}'
