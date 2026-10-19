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


def _train(tmp_path, root, monkeypatch, *, batch_size, iters, eval_every, yolox='trimmed',
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
