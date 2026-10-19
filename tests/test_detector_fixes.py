"""Regression tests for detector training/eval bugs fixed in the architecture cleanup."""
from __future__ import annotations

import json
import re
import sys

import pytest
import torch

from scripts import eval_detector, eval_detector_trace
from tailcyclenet import train_detector

from .conftest import _session_2d


def test_train_detector_small_index_and_best_line(tmp_path, monkeypatch, capsys):
    """B2: an index smaller than `batch_size` yielded zero batches under `drop_last` (spun forever
    at iteration 0). B7: on a tie `best:` named the FIRST max while `detector.pth` holds the LAST."""
    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)              # 4 views < batch_size 6
    out = tmp_path / 'run'
    cfg = tmp_path / 'config.toml'
    cfg.write_text(f"""
[data]
path = "{root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [32, 32]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
background_prob = 0.0
[model]
yolox = "hybrid"
pretrained = ""
[training]
out = "{out}"
iters = 2
batch_size = 6
num_workers = 0
seed = 0
device = "cpu"
eval_every = 1
eval_batches = 1
""")
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    train_detector.main()
    last = torch.load(out / 'detector_last.pth', map_location='cpu', weights_only=False)
    assert last['iteration'] == 2
    history = json.loads((out / 'metrics.json').read_text())
    saved = torch.load(out / 'detector.pth', map_location='cpu', weights_only=False)['iteration']
    best_score = max(h['selection_score'] for h in history)
    assert saved == max(h['iteration'] for h in history if h['selection_score'] == best_score)
    printed = re.search(r'best: it (\d+)', capsys.readouterr().out)
    assert printed and int(printed.group(1)) == saved


def test_convnext_config_refuses_an_input_side_not_divisible_by_32(tmp_path):
    """B3: convnext-t floors `side / stride`, so [120, 96] crashed in the PAFPN concat; the config
    refuses it by name, for the input and the tile. 128 x 96 loads; hybrid has no such rule."""
    from tailcyclenet.detector.config import load_detector_config
    cfg = tmp_path / 'c.toml'
    body = '[model]\nyolox = "{y}"\npretrained = ""\n[training]\nout = "{o}"\n'
    for key in ('input_wh', 'tile_wh'):
        for side, yolox, ok in ((120, 'convnext-t', False), (128, 'convnext-t', True),
                                (120, 'hybrid', True)):
            cfg.write_text(f'[data]\npath = "{tmp_path}"\n{key} = [{side}, 96]\n'
                           + body.format(y=yolox, o=tmp_path))
            if ok:
                assert load_detector_config(cfg)['data'][key] == [side, 96]
            else:
                with pytest.raises(SystemExit, match=f'{key}.*divisible by 32'):
                    load_detector_config(cfg)


def test_load_detector_refuses_a_convnext_size_not_divisible_by_32(tmp_path):
    """B3: a convnext-t checkpoint with a side not divisible by 32 is refused at load, not by an
    opaque RuntimeError in `detect_raw`'s first forward."""
    from tailcyclenet.detector import YOLOXNano, load_detector
    model = YOLOXNano(version='convnext-t')
    for wh in ([128, 96], [120, 96]):
        torch.save({'kind': 'detector', 'input_wh': wh, 'norm': 'gn',
                    'yolox_version': 'convnext-t', 'model_state': model.state_dict()},
                   tmp_path / f'det{wh[0]}.pth')
    assert load_detector(tmp_path / 'det128.pth').input_wh == (128, 96)
    with pytest.raises(ValueError, match='divisible by 32'):
        load_detector(tmp_path / 'det120.pth')


def _saved_detector(tmp_path, **meta):
    """A fresh hybrid detector saved as a checkpoint dict with `meta` on top."""
    from tailcyclenet.detector import YOLOXNano
    path = tmp_path / 'det.pth'
    torch.save({'kind': 'detector', 'input_wh': [64, 64], 'norm': 'gn', 'yolox_version': 'hybrid',
                'model_state': YOLOXNano(version='hybrid').state_dict(), **meta},
               path)
    return path


def _multi_root_detector(tmp_path):
    """A test root `myroot` and a multi-root checkpoint whose per-root source for it differs."""
    root = tmp_path / 'myroot'
    _session_2d(root / 'test' / 's1', T=4, S=2)
    path = _saved_detector(tmp_path, box_source='keypoints', datasets=['other', 'myroot'],
                           box_sources={'other': 'keypoints', 'myroot': 'instances'})
    return root, path


def _spy_box_source(monkeypatch, owner):
    """Wrap `owner.detect_raw`, recording the `box_source` each call forwards."""
    seen, real = [], owner.detect_raw

    def spy(*args, **kwargs):
        """Record the box source, then detect as usual."""
        seen.append(kwargs.get('box_source'))
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, 'detect_raw', spy)
    return seen


def test_eval_detector_deploy_uses_the_roots_own_box_source(tmp_path, monkeypatch):
    """B10: `--deploy` called `detect_raw` without `box_source`, so a multi-root detector's
    per-root `box_sources` entry was ignored for the checkpoint's global one."""
    import tailcyclenet.detector as det_pkg
    root, path = _multi_root_detector(tmp_path)
    seen = _spy_box_source(monkeypatch, det_pkg)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector.py', '--run', str(path), '--data', str(root), '--split', 'test',
        '--deploy', '--no-track', '--device', 'cpu', '--n-frames', '2', '--overlap', '0'])
    eval_detector.main()
    assert seen == ['instances']


def test_eval_detector_trace_uses_the_roots_own_box_source(tmp_path, monkeypatch):
    """B10: the trace script likewise passes the root's own `box_sources` entry."""
    root, path = _multi_root_detector(tmp_path)
    seen = _spy_box_source(monkeypatch, eval_detector_trace)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector_trace.py', '--run', str(path), '--data', str(root), '--split', 'test',
        '--device', 'cpu', '--out', str(tmp_path / 'trace.json')])
    eval_detector_trace.main()
    assert seen and set(seen) == {'instances'}


def test_eval_detector_compare_scores_the_same_views(tmp_path, monkeypatch, capsys):
    """B9: a comma `--split` crashed `--compare`, and the PAIRED delta compared different frames.
    Both arms now share one dataset builder."""
    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)
    _session_2d(root / 'test' / 's1', T=4, S=2)
    path = _saved_detector(tmp_path)
    calls = []
    real = eval_detector.score_dataset

    def spy(model, ds, *args, **kwargs):
        """Record which views each scoring call covers."""
        subset = kwargs.get('subset_indices')
        idx = range(len(ds)) if subset is None else sorted(subset)
        calls.append(sorted((s.session_id, g, f, c) for s, g, f, c in (ds.index[i] for i in idx)))
        return real(model, ds, *args, **kwargs)

    monkeypatch.setattr(eval_detector, 'score_dataset', spy)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector.py', '--run', str(path), '--compare', str(path), '--data', str(root),
        '--device', 'cpu', '--num-workers', '0', '--batches', '1', '--batch-size', '2',
        '--split', 'train,test'])
    eval_detector.main()
    assert calls[-1] == calls[-2]
    assert 'PAIRED' in capsys.readouterr().out


def test_deleted_detector_flags_raise_by_name(monkeypatch):
    """Flags of the deleted keypoint branch / COCO backbone / report-73 sweep exit naming
    themselves and why, not with argparse's bare "unrecognized arguments"."""
    from tailcyclenet.infer.cli import build_parser
    with pytest.raises(SystemExit, match='--pose-nms is deleted.*keypoint'):
        build_parser().parse_args(['--pose', 'r', '--data', 'd', '--out', 'o', '--pose-nms', '1'])
    with pytest.raises(SystemExit, match='--weights-dir is deleted.*COCO'):
        train_detector.main(['--weights-dir', '/tmp/w'])
    monkeypatch.setattr(sys, 'argv', ['eval_detector.py', '--run', 'r', '--data', 'd',
                                      '--evalset', '1'])
    with pytest.raises(SystemExit, match='--evalset is deleted.*transfer'):
        eval_detector.main()
