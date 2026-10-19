"""Regression tests for detector training/eval bugs fixed in the architecture cleanup."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _module(relpath, name):
    """Load a repo file as a fresh module (scripts are not importable as a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train(tmp_path, root, monkeypatch, *, batch_size, iters, eval_every, yolox='hybrid',
           input_wh='[48, 48]'):
    """Run `train_detector.main()` in-process on `root`; return the run folder."""
    out = tmp_path / 'run'
    cfg = tmp_path / 'config.toml'
    cfg.write_text(f"""
[data]
path = "{root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = {input_wh}
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
background_prob = 0.0
[model]
yolox = "{yolox}"
pretrained = ""
[training]
out = "{out}"
iters = {iters}
batch_size = {batch_size}
num_workers = 0
seed = 0
device = "cpu"
eval_every = {eval_every}
eval_batches = 1
""")
    mod = _module('tailcyclenet/train_detector.py', 'tcn_train_detector_fixes')
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    return out


@pytest.mark.timeout(300, method='thread')
def test_train_detector_runs_with_fewer_views_than_batch_size(tmp_path, monkeypatch):
    """detector B2: a train index smaller than `batch_size` used to yield zero batches under
    `drop_last=True`, so `while it < iters` spun forever at iteration 0."""
    from .conftest import _session_2d
    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)
    out = _train(tmp_path, root, monkeypatch, batch_size=16, iters=2, eval_every=2)
    ckpt = torch.load(out / 'detector_last.pth', map_location='cpu', weights_only=False)
    assert ckpt['iteration'] == 2


def test_train_detector_prints_the_iteration_detector_pth_holds(tmp_path, monkeypatch, capsys):
    """detector B7: on a tie the final `best:` line named the FIRST maximum while `detector.pth`
    is overwritten on `>=` and so holds the LAST one."""
    from .conftest import _session_2d
    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)
    out = _train(tmp_path, root, monkeypatch, batch_size=2, iters=3, eval_every=1)
    history = json.loads((out / 'metrics.json').read_text())
    saved = torch.load(out / 'detector.pth', map_location='cpu', weights_only=False)['iteration']
    best_score = max(h['selection_score'] for h in history)
    assert saved == max(h['iteration'] for h in history if h['selection_score'] == best_score)
    printed = re.search(r'best: it (\d+)', capsys.readouterr().out)
    assert printed and int(printed.group(1)) == saved


@pytest.mark.parametrize('key', ['input_wh', 'tile_wh'])
def test_convnext_config_refuses_an_input_side_not_divisible_by_32(tmp_path, key):
    """detector B3: convnext-t floors `side / stride`, so [120, 96] crashed in the first forward
    (PAFPN concat); the config now refuses it by name. 128 x 96 still loads."""
    from tailcyclenet.detector.config import load_detector_config
    cfg = tmp_path / 'c.toml'
    body = '[model]\nyolox = "convnext-t"\npretrained = ""\n[training]\nout = "{o}"\n'
    cfg.write_text(f'[data]\npath = "{tmp_path}"\n{key} = [120, 96]\n' + body.format(o=tmp_path))
    with pytest.raises(SystemExit, match=f'{key}.*divisible by 32'):
        load_detector_config(cfg)
    cfg.write_text(f'[data]\npath = "{tmp_path}"\n{key} = [128, 96]\n' + body.format(o=tmp_path))
    assert load_detector_config(cfg)['data'][key] == [128, 96]
    cfg.write_text(f'[data]\npath = "{tmp_path}"\n{key} = [120, 96]\n'
                   + body.replace('convnext-t', 'hybrid').format(o=tmp_path))
    assert load_detector_config(cfg)['data'][key] == [120, 96]


def test_load_detector_refuses_a_convnext_size_not_divisible_by_32(tmp_path):
    """detector B3: a convnext-t checkpoint recording an input side not divisible by 32 is
    refused at load, not by an opaque RuntimeError in `detect_raw`'s first forward."""
    from tailcyclenet.detector import YOLOXNano, load_detector
    model = YOLOXNano(version='convnext-t')
    for wh in ([128, 96], [120, 96]):
        torch.save({'kind': 'detector', 'input_wh': wh, 'norm': 'gn',
                    'yolox_version': 'convnext-t', 'model_state': model.state_dict()},
                   tmp_path / f'det{wh[0]}.pth')
    assert load_detector(tmp_path / 'det128.pth').input_wh == (128, 96)
    with pytest.raises(ValueError, match='divisible by 32'):
        load_detector(tmp_path / 'det120.pth')


def test_score_fixed_views_restores_augment(tmp_path):
    """detector B13: `score_fixed_views` set `ds.augment = False` and never restored it, so a
    train dataset scored mid-run silently stopped augmenting."""
    from .conftest import _session_2d
    from tailcyclenet.detector import BoxDataset, YOLOXNano
    from tailcyclenet.detector.evaluate import score_fixed_views
    _session_2d(tmp_path / 'r' / 'train' / 's0', T=4, S=2)
    ds = BoxDataset(tmp_path / 'r', 'train', input_wh=(64, 64), augment=True)
    rows = score_fixed_views(YOLOXNano(version='hybrid'), ds, [0, 1], 'cpu')
    assert len(rows) == 2
    assert ds.augment is True


def _saved_detector(tmp_path, **meta):
    """A fresh hybrid detector saved as a checkpoint dict with `meta` on top."""
    from tailcyclenet.detector import YOLOXNano
    path = tmp_path / 'det.pth'
    torch.save({'kind': 'detector', 'input_wh': [64, 64], 'norm': 'gn', 'yolox_version': 'hybrid',
                'model_state': YOLOXNano(version='hybrid').state_dict(), **meta},
               path)
    return path


def test_eval_detector_deploy_uses_the_roots_own_box_source(tmp_path, monkeypatch):
    """detector B10: `--deploy` called `detect_raw` without `box_source`, so a multi-root
    detector's per-root source (`box_sources`) was ignored for the checkpoint's global one."""
    import tailcyclenet.detector as det_pkg
    from .conftest import _session_2d
    root = tmp_path / 'myroot'
    _session_2d(root / 'test' / 's1', T=4, S=2)
    path = _saved_detector(tmp_path, box_source='keypoints', datasets=['other', 'myroot'],
                           box_sources={'other': 'keypoints', 'myroot': 'instances'})
    seen = []
    real = det_pkg.detect_raw

    def spy(*args, **kwargs):
        """Record the box source deployment_score forwards."""
        seen.append(kwargs.get('box_source'))
        return real(*args, **kwargs)

    monkeypatch.setattr(det_pkg, 'detect_raw', spy)
    mod = _module('scripts/eval_detector.py', 'tcn_eval_detector_fixes')
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector.py', '--run', str(path), '--data', str(root), '--split', 'test',
        '--deploy', '--no-track', '--device', 'cpu', '--n-frames', '2', '--overlap', '0'])
    mod.main()
    assert seen == ['instances']


def test_eval_detector_trace_uses_the_roots_own_box_source(tmp_path, monkeypatch):
    """detector B10: the trace script likewise passes the root's own `box_sources` entry."""
    from .conftest import _session_2d
    root = tmp_path / 'myroot'
    _session_2d(root / 'test' / 's1', T=4, S=2)
    path = _saved_detector(tmp_path, box_source='keypoints', datasets=['other', 'myroot'],
                           box_sources={'other': 'keypoints', 'myroot': 'instances'})
    mod = _module('scripts/eval_detector_trace.py', 'tcn_eval_detector_trace_fixes')
    seen = []
    real = mod.detect_raw

    def spy(*args, **kwargs):
        """Record the box source the trace script forwards."""
        seen.append(kwargs.get('box_source'))
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, 'detect_raw', spy)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector_trace.py', '--run', str(path), '--data', str(root), '--split', 'test',
        '--device', 'cpu', '--out', str(tmp_path / 'trace.json')])
    mod.main()
    assert seen and set(seen) == {'instances'}


def test_eval_detector_compare_scores_the_same_views(tmp_path, monkeypatch, capsys):
    """detector B9: under `--evalset`, `--compare` scored the default sampled views while `--run`
    scored the fixed subset, and a comma `--split` crashed it; the PAIRED delta compared
    different frames. Both arms now share one dataset/subset builder."""
    import json as _json
    from .conftest import _session_2d
    from tailcyclenet.format import Session
    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)
    _session_2d(root / 'test' / 's1', T=4, S=2)
    sess = Session.load(root / 'test' / 's1')
    gid = list(sess.groups)[0]
    evalset = tmp_path / 'evalset.json'
    evalset.write_text(_json.dumps({'frames': [
        {'session': sess.session_id, 'group': gid, 'frame': 1, 'split': 'test'}]}))
    path = _saved_detector(tmp_path)
    mod = _module('scripts/eval_detector.py', 'tcn_eval_detector_compare')
    calls = []
    real = mod.score_dataset

    def spy(model, ds, *args, **kwargs):
        """Record which views each scoring call covers."""
        subset = kwargs.get('subset_indices')
        idx = range(len(ds)) if subset is None else sorted(subset)
        calls.append(sorted((s.session_id, g, f, c) for s, g, f, c in (ds.index[i] for i in idx)))
        return real(model, ds, *args, **kwargs)

    monkeypatch.setattr(mod, 'score_dataset', spy)
    for extra in (['--evalset', str(evalset)], ['--split', 'train,test']):
        calls.clear()
        monkeypatch.setattr(sys, 'argv', [
            'eval_detector.py', '--run', str(path), '--compare', str(path), '--data', str(root),
            '--device', 'cpu', '--num-workers', '0', '--batches', '1', '--batch-size', '2',
            *extra])
        mod.main()
        assert calls[-1] == calls[-2]
        assert 'PAIRED' in capsys.readouterr().out


@pytest.mark.parametrize('flag', ['--pose-nms', '--crop-source'])
def test_deleted_infer_flags_raise_by_name(flag):
    """`--pose-nms` and `--crop-source` died with the detector keypoint branch; passing one exits
    naming it and why instead of argparse's bare "unrecognized arguments"."""
    from tailcyclenet.infer.cli import build_parser
    with pytest.raises(SystemExit, match=f'{flag} is deleted.*keypoint'):
        build_parser().parse_args(['--pose', 'r', '--data', 'd', '--out', 'o', flag, '0.5'])


def test_train_detector_weights_dir_raises_by_name():
    """`--weights-dir` cached the deleted COCO backbone."""
    from tailcyclenet import train_detector
    with pytest.raises(SystemExit, match='--weights-dir is deleted.*COCO'):
        train_detector.main(['--weights-dir', '/tmp/w'])
