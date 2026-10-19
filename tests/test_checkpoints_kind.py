"""The config merge and the run/checkpoint `kind` guards.

Neither raises when wrong; each yields a run that quietly means something else. The merge was one
level deep (an overlay key discarded its block's siblings); a checkpoint without `kind` defaulted
to `'pose'`, so a scorer loaded as a pose model.
"""
import tomllib
from pathlib import Path

import pytest
import torch

from tailcyclenet import checkpoints as ck
from tailcyclenet import config as cfgmod
from tailcyclenet.format import Registry
from tailcyclenet.pose import checkpoint as pck
from tailcyclenet.scorer.checkpoint import load_scorer_run


def _packaged(name):
    from importlib.resources import files
    return Path(str(files('tailcyclenet.configs').joinpath(name)))


# -- the merge ---------------------------------------------------------------------------

def test_an_overlay_merges_one_key_at_a_time_at_any_depth(tmp_path):
    """A key deep inside a block keeps its siblings (two and three levels down), sibling blocks
    survive, and a block new to the base is added whole."""
    base = tmp_path / 'base.toml'
    base.write_text("""
[training]
n_iterations = 10

[training.optimizer]
optimizer = "muon"
learning_rate = 1e-4
muon_schedulefree = true
warmup_steps = 500

[scorer.corruption]
const_offset_prob = 0.5

[scorer.corruption.mag_3d]
const_offset = 15.0
gradual_drift = 24.0
""")
    over = tmp_path / 'over.toml'
    over.write_text("""
[training.optimizer]
learning_rate = 2e-4

[scorer.corruption.mag_3d]
gradual_drift = 40.0

[extra]
pool_num_heads = 4
""")
    cfg = cfgmod.load_config(over, base=base)
    opt = cfg['training']['optimizer']
    assert opt == {'optimizer': 'muon', 'learning_rate': 2e-4, 'muon_schedulefree': True,
                   'warmup_steps': 500}
    assert cfg['training']['n_iterations'] == 10, 'a sibling BLOCK was dropped'
    corr = cfg['scorer']['corruption']
    assert corr['mag_3d'] == {'const_offset': 15.0, 'gradual_drift': 40.0}
    assert corr['const_offset_prob'] == 0.5
    assert cfg['extra'] == {'pool_num_heads': 4}


def test_the_shipped_configs_still_merge_to_the_same_dict():
    """No shipped config's meaning may change: every shipped overlay uses flat blocks, so the
    recursive and the old shallow merge must agree on all of them."""
    def shallow(base_cfg, over):
        out = dict(base_cfg)
        for block, o in over.items():
            if isinstance(o, dict) and isinstance(out.get(block), dict):
                out[block] = {**out[block], **o}
            else:
                out[block] = o
        return out

    checked = 0
    for overlay in ('detector/3dpop.toml', 'detector/branson-fly.toml', 'detector/rat-city.toml'):
        p = _packaged(overlay)
        if not p.exists():
            continue
        with open(p, 'rb') as f:
            over = tomllib.load(f)
        with open(cfgmod.DETECTOR_CONFIG, 'rb') as f:
            base_cfg = tomllib.load(f)
        assert cfgmod.deep_merge(base_cfg, over) == shallow(base_cfg, over), overlay
        checked += 1
    assert checked > 0, 'no shipped overlay was checked -- this test proves nothing'


def test_the_scorer_family_base_loads_and_its_kind_is_scorer():
    """`configs/scorer.toml` is a real scorer-family base with every prompt lever OFF (the scorer
    supplies no prior and no box)."""
    cfg = cfgmod.load_config(_packaged('scorer.toml'), base=_packaged('scorer.toml'))
    assert ck.run_kind(cfg) == 'scorer'
    assert cfg['model']['stride_length'] == cfg['data']['n_frames'] == 12
    assert cfg['model']['video_encoder_requires_grad'] is False
    assert set(cfg['scorer']['corruption']['mag_3d']) == {
        'const_offset', 'frame_noise', 'gradual_drift', 'sinusoid'}
    assert cfg['scorer']['min_valid_frames'] == 6
    assert cfg['data']['box_prompt'] == 'none'
    for k in ('prompt_dropout', 'prompt_noise_px', 'prompt_offset_px', 'prompt_swap_kpt_pairs',
              'prompt_swap_animal', 'box_prompt_dropout'):
        assert cfg['data'][k] == 0.0, k


# -- the kind guards ---------------------------------------------------------------------

def test_save_run_meta_stamps_the_kind_and_an_absent_kind_is_pose(tmp_path):
    """The writer decides the kind; every run folder written before the key existed IS pose, and
    the default leaves an existing [run] block's other keys alone."""
    assert ck.run_kind({}) == ck.run_kind({'run': {}}) == 'pose'
    assert ck.run_kind({'run': {'kind': 'scorer'}}) == 'scorer'
    reg = Registry(names=['a'], datasets={})
    ck.save_run_meta(tmp_path / 'r', {'model': {}, 'data': {}}, reg, kind='scorer')
    with open(tmp_path / 'r' / 'config.toml', 'rb') as f:
        assert ck.run_kind(tomllib.load(f)) == 'scorer'
    ck.save_run_meta(tmp_path / 'r2', {'run': {'note': 'x'}, 'model': {}, 'data': {}}, reg)
    with open(tmp_path / 'r2' / 'config.toml', 'rb') as f:
        cfg2 = tomllib.load(f)
    assert ck.run_kind(cfg2) == 'pose' and cfg2['run']['note'] == 'x'


def test_a_run_folder_of_the_other_kind_is_refused(tmp_path):
    body = '\n[model]\nimage_size = 64\n\n[data]\nimage_size = 64\n'
    (tmp_path / 'scorer-run').mkdir()
    (tmp_path / 'scorer-run' / 'config.toml').write_text('[run]\nkind = "scorer"\n' + body)
    (tmp_path / 'pose-run').mkdir()
    (tmp_path / 'pose-run' / 'config.toml').write_text(body)
    with pytest.raises(ValueError, match='scorer'):
        pck.load_run(tmp_path / 'scorer-run')
    with pytest.raises(ValueError, match='pose'):
        load_scorer_run(tmp_path / 'pose-run')


def test_a_checkpoint_carries_its_kind_and_an_unknown_kind_is_refused(tmp_path):
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    reg = Registry(names=['a', 'b'], datasets={})
    cfg = {'model': {}, 'data': {}}
    p = ck.save_checkpoint(tmp_path, 0, m, opt, cfg, registry=reg, kind='scorer')
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    assert ckpt['kind'] == 'scorer' and ckpt['keypoint_registry'] == reg.to_dict()
    p2 = ck.save_checkpoint(tmp_path, 0, m, opt, cfg, name='b', registry=reg)
    assert torch.load(p2, map_location='cpu', weights_only=False)['kind'] == 'pose'
    with pytest.raises(AssertionError, match='kind must be one of'):
        ck.save_checkpoint(tmp_path, 0, m, opt, cfg, kind='posed')
