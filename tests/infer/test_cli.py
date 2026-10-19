"""The inference CLI: defaults, refusals before the checkpoint loads, and end-to-end runs."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet.format import Registry, load_dataset
from tests.helpers import REPO, SMALL, _detector_checkpoint, _save_run, _session_2d, _session_3d

# --- CLI defaults and provenance ----------------------------------------------------------


def test_the_tracker_is_the_default_and_help_renders(capsys):
    """`--track` is ON by default with `--no-track` to restore the memoryless pass; `--help`
    renders for both CLIs (a bare `%` in a help string kills argparse with a format TypeError)."""
    import inspect

    from tailcyclenet.detector.train import main as train_main
    from tailcyclenet.infer.associate import associate_group
    from tailcyclenet.infer.cli import build_parser

    sig = inspect.signature(associate_group)
    assert sig.parameters['track'].default is True
    assert sig.parameters['link'].default is False, 'link is opt-in at the function level'
    ap = build_parser()
    assert ap.get_default('track') is True
    opts = {s for a in ap._actions for s in a.option_strings}
    assert '--no-track' in opts and '--det-cache' not in opts
    assert '--det-score' in ap.format_help()

    with pytest.raises(SystemExit) as e:
        train_main(['--help'])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert all(f in out for f in ('--config', '--out', '--iters', '--device'))


def _cli_setup(tmp_path, kind='rat', name=None, T=4, model_cfg=SMALL, **kw):
    """One session under `<tmp>/<name>/test/s` plus a run folder; returns (session dir, run)."""

    root = tmp_path / (name or kind)
    if kind == 'rat':
        _session_2d(root / 'test' / 's', T=T)
    else:
        _session_3d(root / 'test' / 's', T=T, **kw)
    run, _ = _save_run(tmp_path, Registry.build([load_dataset(root)]), model_cfg)
    return root / 'test' / 's', run


def _no_load(monkeypatch):
    """Make `load_run` fail loudly, so a refusal is proven to fire before the checkpoint loads."""
    def boom(*a, **k):
        raise AssertionError('the refusal must fire before the checkpoint loads')
    monkeypatch.setattr('tailcyclenet.infer.loading.load_run', boom)


def test_parser_defaults_select_the_measured_path():
    """Speed/tracking defaults are the measured ones (bf16, camera batch, --track, --link-boxes)."""
    from tailcyclenet.infer.cli import build_parser

    parser = build_parser()
    assert parser.get_default('precision') == 'bf16'
    assert parser.get_default('units') is None, 'units are resolved only after deriving video mode'
    assert parser.get_default('camera_batch') is True
    assert parser.get_default('allow_pose_scorer_mismatch') is False
    assert parser.get_default('track') is True and parser.get_default('link_boxes') is True
    assert parser.get_default('max_age') == 8 and parser.get_default('max_move') == 1.25
    options = {option for action in parser._actions for option in action.option_strings}
    assert {'--camera-batch', '--no-camera-batch', '--allow-pose-scorer-mismatch'} <= options


def test_the_cli_runs_end_to_end_with_no_detector(cli, monkeypatch, tmp_path):
    """Through `main()` on the default box path: the output is a valid prediction session (no
    keypoints.pq, sidecar opt-in) and `[provenance]` names the run, checkpoint FILE and source by
    absolute path even from a relative `--run`."""
    import tomllib

    from tailcyclenet.format import Session, validate_session
    from tailcyclenet.format.prediction import load_predictions

    sdir, run = _cli_setup(tmp_path)
    out = tmp_path / 'pred'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', 'run', '--data', str(sdir),
                                      '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--out', str(out)])
    cli.main()

    for f in ('session.toml', 'calibration.toml', 'groups.pq', 'points2d.pq', 'windows.pq'):
        assert (out / f).exists(), f'{f} missing; wrote {sorted(p.name for p in out.iterdir())}'
    assert not (out / 'keypoints.pq').exists(), 'model estimates are not annotation observations'
    assert not (out / 'window_predictions.pq').exists(), 'sidecar export is opt-in'
    got = Session.load(out)
    assert got.label_source == 'tracked'
    errs = [e for e in validate_session(got) if '[rule 7]' not in e]
    assert not errs, f'a prediction session must be valid apart from its missing pixels: {errs}'

    preds, meta = load_predictions(out)
    assert preds['s/g000']['pred'].shape[1] == 4
    assert np.isfinite(preds['s/g000']['pred']).any(), 'the round trip lost every point'
    assert meta['anchor'] == 'none'

    with open(out / 'session.toml', 'rb') as f:
        prov = tomllib.load(f)['provenance']
    for k in ('run', 'checkpoint', 'source_session'):
        assert Path(prov[k]).is_absolute() and Path(prov[k]).exists(), (k, prov[k])
    assert Path(prov['source_session']).name == 's'
    assert Path(prov['checkpoint']).is_file(), 'checkpoint must be the FILE, not the run folder'
    assert prov['checkpoint_name'] == Path(prov['checkpoint']).name
    assert prov['box_source'] == 'keypoints'
    assert prov['det_box_source'] == '' and prov['detector'] == ''
    assert prov['precision'] == 'bf16' and prov['camera_batch'] is True


def test_the_cli_runs_from_a_lone_pose_checkpoint(cli, monkeypatch, tmp_path):
    """`--run` may be ONE checkpoint file with no run folder; the prediction is the folder's."""
    import shutil

    from tailcyclenet.format.prediction import load_predictions

    root = tmp_path / 'rat'
    _session_2d(root / 'test' / 's')
    run, path = _save_run(tmp_path, Registry.build([load_dataset(root)]))
    lone = tmp_path / 'elsewhere' / 'pose.pth'
    lone.parent.mkdir()
    shutil.copy(path, lone)

    def infer(where, out):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--pose', str(where),
                                          '--data', str(root / 'test' / 's'), '--anchor', 'none',
                                          '--device', 'cpu', '--overlap', '2', '--out', str(out)])
        cli.main()
        return load_predictions(out)

    ref, _ = infer(run, tmp_path / 'from-run')
    shutil.rmtree(run)
    got, meta = infer(lone, tmp_path / 'from-file')
    assert meta['run'] == str(lone.resolve())
    np.testing.assert_array_equal(got['s/g000']['pred'], ref['s/g000']['pred'])


@pytest.mark.parametrize('flag,selector_flag,value', [
    ('--pose', '--pose-checkpoint', 'checkpoint_best.pth'),
    ('--run', '--checkpoint', 'checkpoint_best.pth'),
    ('--scorer', '--scorer-checkpoint', 'checkpoint_best.pth'),
    ('--detector', '--detector-checkpoint', 'best'),
])
def test_a_checkpoint_selector_beside_a_file_is_refused_before_loading(
        cli, monkeypatch, tmp_path, flag, selector_flag, value):

    _session_2d(tmp_path / 'rat' / 'test' / 's')
    f = tmp_path / 'model.pth'
    torch.save({}, f)

    _no_load(monkeypatch)
    argv = {'--run': str(tmp_path / 'run'), flag: str(f), selector_flag: value}
    monkeypatch.setattr(sys, 'argv', ['infer.py', *[x for kv in argv.items() for x in kv],
                                      '--data', str(tmp_path / 'rat' / 'test' / 's'),
                                      '--device', 'cpu', '--out', str(tmp_path / 'o')])
    shown = {'--run': '--pose', '--checkpoint': '--pose-checkpoint'}
    with pytest.raises(SystemExit,
                       match=f'already a checkpoint file; {shown.get(selector_flag, selector_flag)}'):
        cli.main()


def test_pose_and_run_are_one_option():
    """`--pose`/`--pose-checkpoint` are the names; `--run`/`--checkpoint` stay as aliases."""
    from tailcyclenet.infer.cli import build_parser
    ap = build_parser()
    base = ['--data', 'd', '--out', 'o']
    new = ap.parse_args(['--pose', 'r', '--pose-checkpoint', 'c.pth', *base])
    old = ap.parse_args(['--run', 'r', '--checkpoint', 'c.pth', *base])
    assert (new.run, new.checkpoint) == (old.run, old.checkpoint) == (Path('r'), 'c.pth')


def test_session_shape_refusals_fire_before_the_checkpoint_loads(cli, monkeypatch, tmp_path):
    """`--data` must name ONE session; `--out` must be empty and must not be the source session."""

    _session_2d(tmp_path / 'two' / 'test' / 'a')
    _session_2d(tmp_path / 'two' / 'test' / 'b')
    sdir = tmp_path / 'two' / 'test' / 'a'
    stale = tmp_path / 'stale'
    stale.mkdir()
    (stale / 'stale.pq').write_bytes(b'x')
    _no_load(monkeypatch)
    for data, out, expect in [(tmp_path / 'two', tmp_path / 'o', 'ONE session directory'),
                              (sdir, stale, 'not empty'),
                              (sdir, sdir, 'SOURCE session')]:
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(tmp_path / 'nope'),
                                          '--data', str(data), '--split', 'test',
                                          '--device', 'cpu', '--out', str(out)])
        with pytest.raises(SystemExit, match=expect):
            cli.main()


def test_session_writer_finalizes_incomplete_output_when_preload_fails(cli, monkeypatch, tmp_path):
    """A failure mid-run still finalizes the output: `complete = false`, readable empty tables."""
    import tomllib

    import pyarrow.parquet as pq

    from tailcyclenet.format import Session

    sdir, run = _cli_setup(tmp_path)
    out = tmp_path / 'partial-prediction'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                      '--data', str(sdir), '--anchor', 'none',
                                      '--device', 'cpu', '--window-predictions', '--out', str(out)])

    def fail_preload(_self):
        raise RuntimeError('injected preload failure')

    monkeypatch.setattr(Session, 'preload', fail_preload)
    with pytest.raises(RuntimeError, match='injected preload failure'):
        cli.main()

    with (out / 'session.toml').open('rb') as f:
        assert tomllib.load(f)['complete'] is False
    assert pq.read_table(out / 'points2d.pq').num_rows == 0
    assert pq.read_table(out / 'window_predictions.pq').num_rows == 0


def _detector_ckpt(tmp_path, dataset, min_crop_dim=16, box_source='keypoints'):
    """A minimal but genuinely loadable detector checkpoint file."""
    from tailcyclenet.detector.yolox import YOLOXNano

    ckpt = _detector_checkpoint(YOLOXNano(version='hybrid'), input_wh=(64, 48), dataset=dataset,
                                   min_crop_dim=min_crop_dim, box_source=box_source)
    p = tmp_path / 'det.pth'
    torch.save(ckpt, p)
    return p


class _ReachedRunBlocks(Exception):
    """Raised by a `run_blocks` stub: the driver got past every refusal."""


def test_cli_detector_family_refusal_and_its_escape_hatch(cli, monkeypatch, tmp_path):
    """A detector from another dataset family is refused before `run_blocks`; a same-family variant
    (rat-city-tracked vs a rat-city-combined detector) and `--allow-detector-transfer` pass."""

    for root in ('other-species-tracked', 'rat-city-tracked'):
        _session_2d(tmp_path / root / 'test' / 's')
    registry = Registry.build([load_dataset(tmp_path / r)
                               for r in ('other-species-tracked', 'rat-city-tracked')])
    run, _ = _save_run(tmp_path, registry)
    det = _detector_ckpt(tmp_path, 'rat-city-combined')

    def boom(*a, **k):
        raise _ReachedRunBlocks()

    monkeypatch.setattr('tailcyclenet.infer.driver.run_blocks', boom)

    def main(root, *extra):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                          '--data', str(tmp_path / root / 'test' / 's'),
                                          '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                          '--overlap', '2', '--detector', str(det),
                                          '--out', str(tmp_path / f'pred-{root}-{len(extra)}'),
                                          *extra])
        cli.main()

    with pytest.raises(SystemExit, match='different dataset families'):
        main('other-species-tracked')
    with pytest.raises(_ReachedRunBlocks):
        main('rat-city-tracked')
    with pytest.raises(_ReachedRunBlocks):
        main('other-species-tracked', '--allow-detector-transfer')


@pytest.mark.parametrize('argv,expect', [
    (['--anchor', 'labels', '--detector', 'nope'], 'not label rows'),
    (['--anchor', 'labels', '--boxes', 'nope.npz'], 'not label rows'),
    (['--oracle-corrupt', 'nonsense'], 'kind must be one of'),
    (['--oracle-corrupt', 'off'], 'needs an amount'),
    (['--oracle-corrupt', 'off:0.5', '--anchor', 'carry'], 'only means anything'),
    (['--max-frames', '120', '--start-frame', '300'], 'two readings of equal force'),
    (['--start-frame', '-1'], 'negative frame indices'),
    (['--start-frame', '300', '--end-frame', '20'], 'not past --start-frame'),
])
def test_the_cli_refuses_incoherent_combinations_before_loading_anything(cli, monkeypatch,
                                                                        argv, expect):
    """Combinations that used to run and produce a number (worst: `--anchor labels` with a box
    source seeded detector row a from LABEL row a), plus the frame-range arithmetic refusals."""
    _no_load(monkeypatch)
    repo = REPO
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(repo / 'no_such_run'),
                                      '--data', str(repo), '--out', '/tmp/unused'] + argv)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert expect in str(e.value), f'wanted {expect!r}, got: {e.value}'


def test_deleted_inference_flags_raise_by_name(cli, monkeypatch):
    """Deleted flags (architecture cleanup 2026-10-19) raise naming themselves, not argparse noise."""
    repo = REPO
    for argv in (['--independent-windows'], ['--fill-from', 'prior'],
                 ['--det-input-wh', '416', '416']):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(repo / 'no_such_run'),
                                          '--data', str(repo), '--out', '/tmp/unused'] + argv)
        with pytest.raises(SystemExit, match=f'{argv[0]} is deleted'):
            cli.main()


def test_assoc_res_max_px_overrides_a_data_session_but_defaults_to_its_own(cli, monkeypatch,
                                                                           tmp_path):
    """An omitted `--assoc-res-max-px` keeps the session's own value (17.5, not the format's 30);
    given explicitly it overrides."""
    import tomllib

    import toml

    session_dir, run = _cli_setup(tmp_path)
    toml_path = session_dir / 'session.toml'
    cfg = toml.loads(toml_path.read_text())
    cfg['assoc_res_max_px'] = 17.5
    toml_path.write_text(toml.dumps(cfg))

    def _run(out, *extra):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                          '--data', str(session_dir), '--split', 'test',
                                          '--anchor', 'none', '--device', 'cpu',
                                          '--overlap', '2', '--out', str(out), *extra])
        cli.main()
        with open(out / 'session.toml', 'rb') as f:
            return tomllib.load(f)['assoc_res_max_px']

    assert _run(tmp_path / 'default') == 17.5, \
        "an omitted flag must not clobber the session's own value"
    assert _run(tmp_path / 'override', '--assoc-res-max-px', '12') == 12.0, \
        'given explicitly, it must override'


def test_a_range_writes_source_frame_indices(cli, monkeypatch, tmp_path):
    """`frame` is always the SOURCE index and groups.pq keeps the full length; `load_predictions`
    densifies NaN outside the range (rebasing to 0 was the `chunk_frames` bug)."""
    from tailcyclenet.format.prediction import load_predictions

    sdir, run = _cli_setup(tmp_path, T=16)

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                      '--data', str(sdir), '--anchor', 'none',
                                      '--device', 'cpu', '--overlap', '2',
                                      '--start-frame', '6', '--end-frame', '14',
                                      '--window-predictions', '--out', str(out)])
    cli.main()

    import pyarrow.parquet as pq
    gt = pq.read_table(out / 'groups.pq').to_pydict()
    assert int(gt['n_frames'][0]) == 16, 'groups.pq must keep the FULL group length'
    kp = pq.read_table(out / 'points2d.pq')
    fr = np.asarray(kp.column('frame').to_pylist())
    assert fr.min() >= 6 and fr.max() < 14, f'frames outside the range: {fr.min()}..{fr.max()}'
    sidecar = pq.read_table(out / 'window_predictions.pq')
    assert {'point2d', 'window'} <= set(sidecar['record_type'].to_pylist())
    assert all(r is None or 6 <= r < 14 for r in sidecar['frame'].to_pylist())

    preds, _ = load_predictions(out)
    p = preds['s/g000']['pred']
    assert p.shape[1] == 16, 'a ranged prediction must densify to the FULL group length'
    assert np.isfinite(p[:, 6:14]).any()
    assert not np.isfinite(p[:, :6]).any() and not np.isfinite(p[:, 14:]).any()


def test_a_range_past_every_group_is_refused(cli, monkeypatch, tmp_path):
    """A run whose range starts past every group is refused rather than writing an empty session."""
    sdir, run = _cli_setup(tmp_path)

    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                      '--data', str(sdir), '--anchor', 'none',
                                      '--device', 'cpu', '--overlap', '2',
                                      '--start-frame', '99', '--out', str(tmp_path / 'p')])
    with pytest.raises(SystemExit, match='past the end of every group'):
        cli.main()


def test_identity_events_reach_the_written_session(cli, monkeypatch, tmp_path):
    """A tracked 3D run with a real detector writes `identity_events.pq` through `cli.main()` (the
    wiring is what breaks) with at least one `born` and JSON `detail`."""
    import json

    import pyarrow.parquet as pq

    sdir, run = _cli_setup(tmp_path, 'mv', name='ds', T=8)
    det = _detector_ckpt(tmp_path, 'ds')

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--data', str(sdir), '--out', str(out),
        '--run', str(run), '--detector', str(det), '--device', 'cpu',
        '--split', 'test', '--overlap', '2', '--max-animals', '2', '--anchor', 'none'])
    cli.main()

    path = out / 'identity_events.pq'
    assert path.exists(), 'a tracked multiview run must write identity_events.pq'
    t = pq.read_table(path).to_pydict()
    assert t['event'], 'the table exists but is empty: the events never reached the writer'
    assert set(t) == {'group_id', 'frame', 'slot', 'event', 'detail'}, sorted(t)
    assert 'born' in set(t['event']), f"no birth recorded, events were {set(t['event'])}"
    assert all(f >= 0 for f in t['frame']) and all(s >= 0 for s in t['slot'])
    for d in t['detail']:
        json.loads(d)


def test_identity_events_are_absent_not_fatal_on_a_2d_run(cli, monkeypatch, tmp_path):
    """A 2D run has no tracker (`state['tracker'] = None`), so it writes no identity table and must
    not crash (the wiring once died with `'NoneType' object has no attribute 'events'`)."""
    sdir, run = _cli_setup(tmp_path, name='ds2d', T=8)

    out = tmp_path / 'pred2d'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--data', str(sdir), '--out', str(out),
        '--run', str(run), '--detector', str(_detector_ckpt(tmp_path, 'ds2d')),
        '--device', 'cpu', '--split', 'test', '--overlap', '2', '--max-animals', '2',
        '--anchor', 'none'])
    cli.main()

    assert (out / 'points2d.pq').exists(), 'the 2D run itself must still produce a prediction'
    assert not (out / 'identity_events.pq').exists(), \
        'a run with no tracker has no identity events; an empty table would imply it had some'


def test_the_2d_row_gate_masks_the_saved_prediction(cli, monkeypatch, tmp_path):
    """A gated 2D row must stay gated in the saved points2d.pq (it used to be resurrected there,
    since `load_predictions` reads it back as the primary prediction)."""
    from tailcyclenet.format.prediction import load_predictions

    sdir, run = _cli_setup(tmp_path)
    out = tmp_path / 'hi'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(sdir),
                                      '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--vis-thresh=1e9', '--out', str(out)])
    cli.main()
    preds, _ = load_predictions(out)
    assert not np.isfinite(preds['s/g000']['pred']).any()


def test_a_moving_rig_3d_prediction_carries_extrinsics_and_resolved_refine(cli, monkeypatch,
                                                                         tmp_path):
    """A moving-rig prediction writes the source's extrinsics.pq (else it fails rule 13), and a
    non-box 3D run records the RESOLVED `refine = true` (it once recorded `bool(None)`)."""
    import tomllib

    from tailcyclenet.format import Session, validate_session

    sdir, run = _cli_setup(tmp_path, 'mv', T=6, moving=True)
    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(sdir),
                                      '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--out', str(out)])
    cli.main()

    assert (out / 'extrinsics.pq').exists(), 'a moving-rig prediction must write extrinsics.pq'
    got = Session.load(out)
    np.testing.assert_array_equal(got.labels('g000').ext, Session.load(sdir).labels('g000').ext)
    errs = [e for e in validate_session(got) if '[rule 7]' not in e]
    assert not errs, f'a moving prediction session must validate apart from missing pixels: {errs}'

    with open(out / 'session.toml', 'rb') as f:
        prov = tomllib.load(f)['provenance']
    assert prov['refine'] is True, prov
    assert prov['box_prompt'] == 'none' and prov['box_prompt_requested'] == 'firstonly'
    assert isinstance(prov['crop_inflate'], float)
    assert 'gridresid_offset' in prov and 'gridresid_offset_override' in prov


@pytest.mark.parametrize('argv', [
    ['--box-prompt', 'labels', '--detector', 'nope'],
    ['--box-prompt', 'labels', '--boxes', 'nope.npz'],
])
def test_box_prompt_labels_refuses_a_detector_or_boxes_source(cli, monkeypatch, tmp_path, argv):
    """`--box-prompt labels` with a detector/`--boxes` source would feed label row a's box to a
    score-ordered detector row (another animal's ground truth) -- refused once the model is a box
    model."""
    sdir, run = _cli_setup(tmp_path, model_cfg={**SMALL, 'box_prompt': 'film'})
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(sdir),
                                      '--split', 'test', '--device', 'cpu',
                                      '--out', str(tmp_path / 'out')] + argv)
    with pytest.raises(SystemExit, match='not label rows'):
        cli.main()
