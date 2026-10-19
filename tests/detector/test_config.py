"""Detector training config: the shipped recipe, overlays, and refusals by name."""
from __future__ import annotations

import re
import sys

import pytest

from tailcyclenet.detector.config import TRAINING_KEYS, load_detector_config
from tests.helpers import REPO, _write_config

# --- detector training config (configs/detector.toml + tailcyclenet/detector/config.py) ------

BARE = """
[data]
path = "/tmp/ds"
[model]
yolox = "hybrid"
[training]
out = "/tmp/run"
"""


def test_detector_config_bare_overlay_inherits_the_shipped_recipe(tmp_path):
    """The implicit base is `configs/detector.toml`: an absent key inherits its shipped value (not
    an old code default), and `--out`/`--iters`/`--device` land in [training]."""
    cfg = load_detector_config(_write_config(tmp_path, BARE))
    d, m, t = cfg['data'], cfg['model'], cfg['training']
    assert d['balance_datasets'] is True and d['boxes_by_dataset'] == {}
    assert d['annot_frac'] == 0.6
    assert 'wandb' not in cfg
    assert m['p2'] is True and m['pretrained'] == ''          # hybrid: no pretrained source
    assert (t['assignment'], t['box_loss'], t['tal_topk']) == ('tal', 'ciou', 20)
    assert 'assignment' in TRAINING_KEYS
    assert t['weight_decay'] == 0.01 and t['max_pos_per_gt'] == 10 and t['box_weight'] == 15.0
    assert t['iou_aware_obj'] is True and t['iou_aware_warmup'] == 2000
    assert (t['optimizer'], t['muon_momentum'], t['muon_lr_scale']) == ('muon', 0.95, 1.0)
    assert t['warmup_steps'] == 500 and (t['beta1'], t['beta2']) == (0.9, 0.95)
    assert t['shared_head'] is False and t['fpn_upsample'] == 'bilinear'

    overlay = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[training]
out = "/tmp/run-det"
""", 'overlay.toml')
    cfg = load_detector_config(overlay, out='/tmp/run-det', iters=7, device='cpu')
    d, m, t = cfg['data'], cfg['model'], cfg['training']
    assert d['boxes'] == 'instances' and d['min_crop_dim'] == 64 and d['min_box_px'] == 32
    assert d['max_input_px'] == 4 * 416 * 416 and d['val_frames_per_group'] == 8
    assert 'frames_per_group' not in d and 'use_regions' not in d and 'keypoints' not in d
    assert d['augment'] is True and d['augment_strong'] is True and d['rotate_deg'] == 180.0
    assert d['reduce'] is False and d['hflip'] is True
    assert d['input_wh'] is None and d['tile_wh'] is None
    assert d['tile_scale'] == 1.0 and d['tile_bg_per_frame'] == 1
    assert m['yolox'] == 'convnext-t' and m['pretrained'] == 'imagenet'
    assert (t['out'], t['iters'], t['device']) == ('/tmp/run-det', 7, 'cpu')
    assert t['batch_size'] == 8 and t['lr'] == 1e-3 and t['num_workers'] == 8 and t['seed'] == 23
    assert t['eval_every'] == 2000 and t['eval_batches'] == 25


_REFUSED = [  # (block, line, match): each is a decision the loader must refuse by name
    ('data', 'bogus = 1', 'unknown key'),
    ('model', 'bogus = 1', 'unknown key'),
    ('training', 'bogus = 1', 'unknown key'),
    ('data', 'boxes = "nope"', 'boxes'),
    ('model', 'yolox = "nope"', 'yolox'),
    ('model', 'pretrained = "/some/path.pth"', 'pretrained'),
    ('data', 'balance_datasets = "false"', 'balance_datasets'),
    ('data', 'boxes_by_dataset = []', 'boxes_by_dataset must be a table'),
    ('data', 'annot_frac = 1.5', 'annot_frac'),
    ('training', 'weight_decay = -0.1', 'weight_decay'),
    ('training', 'assignment = "bad"', 'assignment'),
    ('training', 'box_loss = "bad"', 'box_loss'),
    ('training', 'optimizer = "sgd"', 'optimizer'),
    ('training', 'fpn_upsample = "bicubic"', 'fpn_upsample'),
    ('wandb', 'mode = "invalid"', r'\[wandb\]\.mode'),
    (None, 'extends = "../detector.toml"', 'extends'),
]


def test_detector_config_refuses_bad_keys_and_values_by_name(tmp_path):
    for i, (block, line, match) in enumerate(_REFUSED):
        text = BARE + '[wandb]\n' if block == 'wandb' else BARE
        if block is None:
            text = line + '\n' + text
        elif line.split(' =')[0] + ' =' in text:
            text = re.sub(rf'^{line.split(" =")[0]} = .*$', line, text, flags=re.M)
        else:
            text = text.replace(f'[{block}]\n', f'[{block}]\n{line}\n', 1)
        with pytest.raises(SystemExit, match=match):
            load_detector_config(_write_config(tmp_path, text, f'bad{i}.toml'))


def test_detector_config_wandb_block_and_logging(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace

    from tailcyclenet.detector.train import (
        _init_detector_wandb,
        _log_detector_wandb,
        _wandb_eval_metrics,
    )

    cfg = load_detector_config(_write_config(tmp_path, BARE + """[wandb]
project_name = "tailcycle-detector"
path = "/tmp/wandb"
mode = "online"
"""))
    assert cfg['wandb'] == {'project_name': 'tailcycle-detector', 'path': '/tmp/wandb',
                            'mode': 'online'}

    scores = {'val_macro': {'r50': 0.75, 'n_gt': 12, 'nan': float('nan')},
              'val_root_facemap': {'r50': 0.5, 'label': 'ignored'}}
    assert _wandb_eval_metrics(scores, {'q50': 0.87}) == {
        'eval/val_macro/r50': 0.75, 'eval/val_macro/n_gt': 12.0,
        'eval/val_root_facemap/r50': 0.5, 'train/objectness_q50': 0.87}

    calls = {}
    fake = ModuleType('wandb')
    fake.run = SimpleNamespace(name='test-run')
    fake.init = lambda **kwargs: calls.update(init=kwargs)
    fake.log = lambda values, step: calls.update(log=(values, step))
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    config = {'wandb': {'project_name': 'tailcycle-detector', 'path': str(tmp_path),
                        'mode': 'online'}}
    wb = _init_detector_wandb(config, tmp_path / 'run')
    _log_detector_wandb(wb, {'train/loss': 1.25}, 50)
    assert calls['init']['project'] == 'tailcycle-detector'
    assert calls['init']['mode'] == 'online' and calls['init']['config'] is config
    assert calls['log'] == ({'train/loss': 1.25}, 50)


def test_load_detector_config_overrides_apply_before_the_required_check(tmp_path):
    """`--out` must rescue `out = ""` (the override used to land AFTER the required-check), and
    `--iters`/`--device` share that path."""
    cfg = _write_config(tmp_path, BARE.replace('"/tmp/run"', '""') + 'iters = 100\n')
    loaded = load_detector_config(cfg, out=tmp_path / 'run', iters=5, device='cuda:1')
    assert loaded['training']['out'] == str(tmp_path / 'run')
    assert loaded['training']['iters'] == 5 and loaded['training']['device'] == 'cuda:1'
    with pytest.raises(SystemExit, match='out.*required'):
        load_detector_config(cfg)


def test_detector_config_round_trips_through_the_run_folder(tmp_path):
    """The recorded config.toml (None values dropped) loads back to the same recipe."""
    from tailcyclenet.detector.config import load_detector_config

    src = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
boxes = "keypoints"
min_crop_dim = 64
input_wh = [96, 64]
tile_wh = []
[model]
yolox = "hybrid"
[training]
out = "/tmp/run"
iters = 2
""", 'src.toml')
    cfg = load_detector_config(src, out=str(tmp_path / 'run'), iters=2, device='cpu')
    import toml
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'config.toml').write_text(toml.dumps(cfg))
    again = load_detector_config(tmp_path / 'run' / 'config.toml')
    assert again['data']['input_wh'] == [96, 64] and again['data']['tile_wh'] is None
    assert again['data']['boxes'] == 'keypoints'
    assert again['training']['iters'] == 2
    assert again['model']['yolox'] == 'hybrid'


def test_per_root_detector_overlays_load_and_raise_val_frames_per_group(tmp_path):
    """Each shipped per-root overlay carries `[data].path` and raises `val_frames_per_group`
    past 8: each of those roots has ONE val group."""
    root = REPO / 'configs' / 'detector'
    want = {'rat-city.toml': 64, 'branson-fly.toml': 32, '3dpop.toml': 16}
    for name, expect in want.items():
        cfg = load_detector_config(root / name, out=tmp_path / name)
        assert cfg['data']['path'], f'{name}: [data].path must be set (no CLI override exists)'
        assert cfg['data']['val_frames_per_group'] == expect, name
        assert cfg['data']['val_frames_per_group'] > 8, \
            f'{name}: must exceed the shipped default of 8 or the overlay does nothing'


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


def _cfg(tmp_path, extra='', model='yolox = "hybrid"'):
    p = tmp_path / 'c.toml'
    p.write_text(f'[data]\npath = "/tmp/ds"\n{extra}\n[model]\n{model}\n'
                 '[training]\nout = "/tmp/run"\n')
    return p


def test_shipped_defaults_are_the_report_73_recipe(tmp_path):
    """The shipped recipe is report 73's; an overlay changing `yolox` never inherits a
    `pretrained` source that cannot load into it."""
    cfg = load_detector_config(_cfg(tmp_path, '', model=''))
    d, m = cfg['data'], cfg['model']
    assert (m['yolox'], m['pretrained']) == ('convnext-t', 'imagenet')
    assert d['input_norm'] == 'equalize' and d['invert_prob'] == 0.15
    assert d['scale_range'] == [0.55, 1.4] and d['background_prob'] == 0.25
    assert cfg['training']['iters'] == 8000 and d['boxes'] == 'instances'
    for model, want in (('yolox = "convnext-t"\npretrained = ""', ''), ('yolox = "hybrid"', '')):
        assert load_detector_config(_cfg(tmp_path, '', model=model))['model']['pretrained'] == want


@pytest.mark.parametrize('extra, model, match', [
    ('exposure_prob = 0.5', '', 'exposure_prob'),
    ('keypoints = false', '', "deleted key.*'keypoints'"),
    ('', 'yolox = "dinov2-s"', 'deleted'),
    ('', 'yolox = "tiny"', 'deleted.*v0.3.4'),
    ('', 'yolox = "hybrid"\npretrained = "coco"', "pretrained='coco' is deleted"),
    ('', 'yolox = "hybrid"\npretrained = "imagenet"', 'imagenet'),
    ('input_norm = "clahe"', '', 'input_norm'),
    ('scale_range = [1.2, 0.5]', '', 'scale_range')])
def test_deleted_and_mismatched_keys_raise_by_name(tmp_path, extra, model, match):
    """One case per family: deleted key, deleted architecture, deleted source, a source the
    architecture cannot load, and a nonsense value."""
    with pytest.raises(SystemExit, match=match):
        load_detector_config(_cfg(tmp_path, extra, model=model))


def test_save_every_defaults_to_eval_every_and_must_be_positive(tmp_path):
    """An absent `save_every` follows `eval_every`; a non-positive one is refused by name."""
    config = tmp_path / 'detector.toml'
    head = '[data]\npath = "dataset"\n[model]\nyolox = "hybrid"\n[training]\nout = "out"\n'
    config.write_text(head + 'eval_every = 13\n')
    assert load_detector_config(config)['training']['save_every'] == 13
    config.write_text(head + 'save_every = 0\n')
    with pytest.raises(SystemExit, match='save_every'):
        load_detector_config(config)
