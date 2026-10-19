"""Detector training end to end, its optimizer, and the checkpoints it writes."""
from __future__ import annotations

import json
import re
import shutil
import sys

import pytest
import torch

from tailcyclenet.detector import train as train_detector
from tailcyclenet.detector.checkpoint import load_detector
from tailcyclenet.detector.train import input_wh_for_roots
from tailcyclenet.detector.yolox import YOLOXNano
from tailcyclenet.format import load_datasets
from tests.helpers import _session_2d, _write_config


def test_build_detector_optimizer_adamw_matches_param_count():
    from tailcyclenet.detector.train import build_detector_optimizer

    model = YOLOXNano(version='hybrid')
    opt, sched = build_detector_optimizer(
        model, {'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 5e-4, 'iters': 10},
        {'pretrained': ''})
    assert sum(p.numel() for g in opt.param_groups for p in g['params']) == \
        sum(p.numel() for p in model.parameters())
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


# --- train_detector end to end -----------------------------------------------------------


def test_train_detector_end_to_end_wires_every_lever(tmp_path, dense_root, monkeypatch):
    """A 2-iteration in-process run with the non-default levers on: every config value reaches
    `detector_loss` (box_weight was once a dead key; `max_pos_per_gt = 0` means uncapped), the run
    folder holds the artefacts, and the checkpoint is self-contained and reloads its architecture."""
    import tomllib

    import tailcyclenet.detector.train as td
    from tailcyclenet.detector.checkpoint import load_detector

    seen = []
    real_loss = td.detector_loss

    def spy(*a, **kw):
        seen.append(kw)
        return real_loss(*a, **kw)

    monkeypatch.setattr(td, 'detector_loss', spy)
    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
tile_wh = []
[model]
yolox = "hybrid"
p2 = true
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
box_weight = 3.0
max_pos_per_gt = 0
iou_aware_obj = true
iou_aware_warmup = 0
shared_head = false
fpn_upsample = "bilinear"
""")
    td.main(['--config', str(cfg)])
    assert len(seen) == 2
    kw = seen[0]
    assert kw['box_weight'] == 3.0 and kw['max_pos_per_gt'] is None
    assert kw['iou_aware'] is True and kw['iou_aware_warmup'] == 0
    assert (kw['assignment'], kw['box_loss_fn']) == ('tal', 'ciou')

    for name in ('detector_it000002.pth', 'detector_last.pth', 'detector.pth', 'metrics.json',
                 'config.toml', 'provenance.toml'):
        assert (out / name).exists(), name
    with open(out / 'config.toml', 'rb') as f:
        recorded = tomllib.load(f)
    assert recorded['data']['input_wh'] == [48, 48] and recorded['training']['iters'] == 2
    ld = load_detector(out / 'detector.pth')
    assert tuple(ld.input_wh) == (48, 48) and ld.min_crop_dim == 16
    assert ld.box_source == 'keypoints' and ld.tile_scale is None
    assert ld.model.p2 is True and hasattr(ld.model.head, 'obj_convs')
    assert ld.model.neck.fpn_upsample == 'bilinear'
    ckpt = torch.load(out / 'detector_it000002.pth', map_location='cpu', weights_only=False)
    assert ckpt['kind'] == 'detector' and ckpt['optimizer_kind'] == 'muon'
    assert ckpt['yolox_version'] == 'hybrid' and ckpt['config'] == recorded
    with open(out / 'provenance.toml', 'rb') as f:
        assert ckpt['provenance'] == tomllib.load(f)

    lone = tmp_path / 'elsewhere' / 'detector.pth'
    lone.parent.mkdir()
    shutil.copy(out / 'detector_it000002.pth', lone)
    model = load_detector(out / 'detector_it000002.pth').model
    shutil.rmtree(out)
    lone_model = load_detector(lone).model
    for k, v in model.state_dict().items():
        assert torch.equal(v, lone_model.state_dict()[k]), k


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


def test_checkpoint_holds_the_weights_that_were_scored(tmp_path, dense_root, monkeypatch):
    """Schedule-free's `opt.train()` swaps the scored `x` back to `y` in place after scoring, and
    the checkpoint used to be written after the swap: `model_state` must be the scored weights."""
    mod = train_detector
    seen = []
    real = mod.score_dataset

    def spy(model, *a, **k):
        """Snapshot the weights at scoring time, then score as usual."""
        seen.append({n: t.detach().clone() for n, t in model.state_dict().items()})
        return real(model, *a, **k)

    monkeypatch.setattr(mod, 'score_dataset', spy)
    out = tmp_path / 'run'
    cfg = tmp_path / 'config.toml'
    cfg.write_text(f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [32, 32]
min_box_px = 0
val_frames_per_group = 4
[model]
yolox = "hybrid"
[training]
out = "{out}"
iters = 2
batch_size = 2
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
warmup_steps = 0
""")
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    ckpt = torch.load(out / 'detector_it000002.pth', map_location='cpu', weights_only=False)
    assert ckpt['model_state_is'] == 'eval' and 'model_state_train' in ckpt
    assert ckpt['box_target'] == 'extent' and ckpt['antialias'] is True
    scored = seen[-1]
    for n, t in ckpt['model_state'].items():
        assert torch.equal(t, scored[n]), f'{n}: the checkpoint is not the scored weight'
    assert any(not torch.equal(t, ckpt['model_state_train'][n])
               for n, t in ckpt['model_state'].items()), 'x == y: the test checked nothing'

    from tailcyclenet.detector.checkpoint import load_detector
    det = load_detector(out, checkpoint='detector_it000002.pth').model
    assert det.box_target == 'extent' and det.antialias and det.min_crop_dim == 16


def _config(tmp_path, root, out):
    path = tmp_path / 'config-box.toml'
    path.write_text(f'''
[data]
path = "{root}"
boxes = "keypoints"
boxes_by_dataset = {{ ratlike = "instances" }}
balance_datasets = true
input_wh = [32, 32]
min_box_px = 0
min_crop_dim = 16
val_frames_per_group = 4
augment = false
augment_strong = false
background_prob = 0.0
[model]
yolox = "hybrid"
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
    train_detector.main()

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

    _ld = load_detector(out, device='cpu')
    detector, trained_on = _ld.model, _ld.dataset
    assert trained_on == ''
    assert detector.trained_datasets == ['mouselike', 'ratlike']
    assert detector.box_sources == {'mouselike': 'keypoints', 'ratlike': 'instances'}
    history = json.loads((out / 'metrics.json').read_text())
    assert history[-1]['selection_metric'] == 'val_macro_r50'
    assert 'val_root_ratlike_r50' in history[-1]


def test_multi_root_auto_input_size_is_root_order_independent(tiny_root):
    """The auto input size is a property of the root set, not of its order."""
    roots = load_datasets(tiny_root, split='train')
    assert len(roots) == 2
    forward = input_wh_for_roots(roots, 'keypoints', min_box_px=0)
    reverse = input_wh_for_roots(list(reversed(roots)), 'keypoints', min_box_px=0)
    assert forward == reverse
    assert all(side % 32 == 0 for side in forward)


def test_detector_pth_holds_the_best_checkpoint_not_the_last(tmp_path, monkeypatch):
    """`detector.pth` is the checkpoint with the best selection score (a tie goes to the later
    one); `detector_last.pth` is always the latest. Driven through `_evaluate_and_save` with a
    scripted val history, so the selection rule is exercised, not read off the source."""
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    history_r50 = {1: 0.3, 2: 0.7, 3: 0.5, 4: 0.7, 5: 0.6}
    split = {'r50': 0.0, 'r75': 0.0, 'iou': 0.0, 'fp': 0.0, 'mota': 0.0}

    def scripted(model, train, val, device, train_cfg, multi, selection_kind):
        return {'train': dict(split), 'val': {**split, 'r50': history_r50[it]}}, []

    monkeypatch.setattr(train_detector, '_score_splits', scripted)
    best, best_it, history = -float('inf'), None, []
    for it in history_r50:
        with torch.no_grad():
            model.weight.fill_(float(it))      # each iteration's weights are identifiable
        best, best_it = train_detector._evaluate_and_save(
            model, opt, object(), object(), 'cpu', {'balance_datasets': False}, {}, False,
            {'input_wh': [32, 32]}, it, tmp_path, history, best, best_it, None)
    assert (best, best_it) == (0.7, 4)
    kept = torch.load(tmp_path / 'detector.pth', weights_only=False)
    assert kept['iteration'] == 4 and kept['selection_score'] == 0.7
    assert float(kept['model_state']['weight'][0, 0]) == 4.0
    last = torch.load(tmp_path / 'detector_last.pth', weights_only=False)
    assert last['iteration'] == 5 and float(last['model_state']['weight'][0, 0]) == 5.0
    assert [h['iteration'] for h in history] == list(history_r50)
