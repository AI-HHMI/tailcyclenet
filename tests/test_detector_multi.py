"""Multi-root detector training and explicit box-only refusal."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tailcyclenet.detector import load_detector
from tailcyclenet.infer.driver import (_detector_box_source, _detector_matches_source)
from tailcyclenet.format import load_datasets
from tailcyclenet.train_detector import input_wh_for_roots

REPO = Path(__file__).resolve().parent.parent


def _train_detector_module():
    spec = importlib.util.spec_from_file_location(
        'tcn_train_detector_multi', REPO / 'tailcyclenet' / 'train_detector.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path, root, out, *, keypoints=False):
    path = tmp_path / f'config-{"kpt" if keypoints else "box"}.toml'
    path.write_text(f'''
[data]
path = "{root}"
boxes = "keypoints"
boxes_by_dataset = {{ ratlike = "instances" }}
balance_datasets = true
input_wh = [64, 64]
min_box_px = 0
min_crop_dim = 16
val_frames_per_group = 4
augment = false
augment_strong = false
keypoints = {str(keypoints).lower()}
background_prob = 0.0
[model]
yolox = "trimmed"
pretrained = ""
[training]
out = "{out}"
iters = 1
batch_size = 2
num_workers = 0
seed = 0
device = "cpu"
eval_every = 1
eval_batches = 1
''')
    return path


def test_multi_root_training_records_root_scores_and_loads_checkpoint(tiny_root, tmp_path,
                                                                       monkeypatch):
    """A 2D + 3D folder trains once, validates only the root with val/, and packages metadata."""
    out = tmp_path / 'run'
    config = _config(tmp_path, tiny_root, out)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(config)])
    _train_detector_module().main()

    ckpt = torch.load(out / 'detector.pth', map_location='cpu', weights_only=False)
    assert ckpt['dataset'] == ''
    assert ckpt['datasets'] == ['mouselike', 'ratlike']
    assert ckpt['box_sources'] == {'mouselike': 'keypoints', 'ratlike': 'instances'}
    assert ckpt['balance_datasets'] is True
    assert ckpt['root_mix'] == pytest.approx({'mouselike': 0.5, 'ratlike': 0.5})
    assert ckpt['selection_metric'] == 'val_macro_r50'
    assert 'train_root_mouselike' in ckpt['eval']
    assert 'train_root_ratlike' in ckpt['eval']
    assert 'val_root_ratlike' in ckpt['eval']
    assert not any(key.startswith('val_root_mouselike') for key in ckpt['eval'])

    detector, _, trained_on, *_ = load_detector(out, device='cpu')
    assert trained_on == ''
    assert detector.trained_datasets == ['mouselike', 'ratlike']
    assert detector.box_sources == {'mouselike': 'keypoints', 'ratlike': 'instances'}
    history = json.loads((out / 'metrics.json').read_text())
    assert history[-1]['selection_metric'] == 'val_macro_r50'
    assert 'val_root_ratlike_r50' in history[-1]


def test_multi_root_auto_input_size_is_root_order_independent(tiny_root):
    roots = load_datasets(tiny_root, split='train')
    assert len(roots) == 2
    forward = input_wh_for_roots(roots, 'keypoints', min_box_px=0)
    reverse = input_wh_for_roots(list(reversed(roots)), 'keypoints', min_box_px=0)
    assert forward == reverse
    assert all(side % 32 == 0 for side in forward)


def test_multi_root_detector_inference_guard_and_box_source_resolution():
    assert _detector_matches_source(['allen-mouse-combined', 'johnson-mouse-combined-aug'], '',
                                    'allen-mouse-combined')
    assert not _detector_matches_source(['allen-mouse-combined', 'johnson-mouse-combined-aug'],
                                        '', 'allen-mouse-annotated')
    assert _detector_matches_source(['allen-mouse-combined'], 'allen-mouse-combined',
                                    'allen-mouse-annotated')

    class Detector:
        box_sources = {'allen-mouse-combined': 'instances',
                       'johnson-mouse-combined-aug': 'keypoints'}

    detector = Detector()
    assert _detector_box_source(detector, 'allen-mouse-combined', 'keypoints') == 'instances'
    assert _detector_box_source(detector, 'johnson-mouse-annotated', 'instances') == 'keypoints'
    assert _detector_box_source(detector, 'rat-city', 'keypoints') == 'keypoints'


def test_multi_root_training_explicitly_refuses_keypoints(tiny_root, tmp_path, monkeypatch):
    config = _config(tmp_path, tiny_root, tmp_path / 'run', keypoints=True)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(config)])
    with pytest.raises(SystemExit, match=r'multi-root detector training is box-only.*keypoints'):
        _train_detector_module().main()
