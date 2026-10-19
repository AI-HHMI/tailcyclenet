"""`scripts/eval_detector.py` and `eval_detector_trace.py`: deploy, compare, refusals."""
from __future__ import annotations

import sys

import pytest
import torch

from scripts import eval_detector, eval_detector_trace
from tailcyclenet.detector import train as train_detector
from tests.helpers import _detector_checkpoint, _session_2d


def test_eval_detector_deploy_prints_the_scored_t_under_det_max_frames(tmp_path, monkeypatch,
                                                                       capsys):
    """`eval_detector.py --deploy` runs end to end on a saved checkpoint, and its T column is the
    `--det-max-frames` prefix actually scored, not the group's raw length (a 3dpop run printed
    T=899..2680 while scoring 120)."""
    from scripts import eval_detector

    root = tmp_path / 'root'
    _session_2d(root / 'test' / 's1', T=4, S=2)
    ckpt = tmp_path / 'detector.pth'
    torch.save(_detector_checkpoint(input_wh=[64, 64], min_crop_dim=8,
                                       box_source='keypoints'), ckpt)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector.py', '--run', str(ckpt), '--data', str(root), '--split', 'test',
        '--deploy', '--no-track', '--link-boxes', '--score-thresh', '0.001', '--device', 'cpu',
        '--n-frames', '2', '--overlap', '0', '--det-max-frames', '2'])
    eval_detector.main()
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if '/g' in ln and ln.strip()[0] != '=')
    assert line.split()[1] == '2', f'expected the truncated T=2 in {line!r}'


def _saved_detector(tmp_path, **meta):
    """A fresh hybrid detector saved as a checkpoint dict with `meta` on top."""
    from tailcyclenet.detector.yolox import YOLOXNano
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
    import tailcyclenet.evaluation.deployment as det_pkg
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
