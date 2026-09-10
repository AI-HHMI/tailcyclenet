"""The config merge and the run/checkpoint `kind` guards.

Both are load-bearing in the same way: neither produces an exception when it is wrong, it produces
a run that quietly means something else.

  * The merge was one level deep, so an overlay setting `[training.optimizer] learning_rate`
    replaced the whole `optimizer` sub-dict and silently discarded `muon_schedulefree`, `beta1`,
    `beta2` and both warmup keys. `_deep_merge` recurses; a block can no longer be replaced
    wholesale.
  * A checkpoint recorded no `kind` at all, and `_load_packaged_pose` defaulted a missing one to
    `'pose'`. A scorer IS a pose encoder plus heads, so a scorer checkpoint would have loaded as a
    pose model without raising.
"""
import tomllib
from pathlib import Path

import pytest
import torch

from tailcyclenet import checkpoints as ck
from tailcyclenet.format import Registry


def _packaged(name):
    from importlib.resources import files
    return Path(str(files('tailcyclenet.configs').joinpath(name)))


# -- the merge ---------------------------------------------------------------------------

def test_an_overlay_keeps_its_siblings_in_a_nested_block(tmp_path):
    """The bug this fixes: one key deep inside a block must not discard the rest of the block."""
    base = tmp_path / 'base.toml'
    base.write_text("""
[training]
n_iterations = 10

[training.optimizer]
optimizer = "muon"
learning_rate = 1e-4
muon_schedulefree = true
beta1 = 0.9
beta2 = 0.95
warmup_steps = 500
""")
    over = tmp_path / 'over.toml'
    over.write_text("""
[training.optimizer]
learning_rate = 2e-4
""")
    cfg = ck.load_config(over, base=base)
    opt = cfg['training']['optimizer']
    assert opt['learning_rate'] == 2e-4, 'the overlay did not win'
    # every sibling survives -- under the old shallow merge all five of these vanished
    assert opt['optimizer'] == 'muon'
    assert opt['muon_schedulefree'] is True
    assert opt['beta1'] == 0.9 and opt['beta2'] == 0.95
    assert opt['warmup_steps'] == 500
    assert cfg['training']['n_iterations'] == 10, 'a sibling BLOCK was dropped'


def test_a_three_deep_block_merges_one_key_at_a_time(tmp_path):
    """The scorer's case: `[scorer.corruption.mag_3d]` is three levels down."""
    base = tmp_path / 'base.toml'
    base.write_text("""
[scorer.corruption]
const_offset_prob = 0.5
frame_noise_prob = 0.5

[scorer.corruption.mag_3d]
const_offset = 15.0
frame_noise = 10.0
gradual_drift = 24.0
sinusoid = 16.0
""")
    over = tmp_path / 'over.toml'
    over.write_text("""
[scorer.corruption.mag_3d]
gradual_drift = 40.0
""")
    cfg = ck.load_config(over, base=base)
    corr = cfg['scorer']['corruption']
    assert corr['mag_3d']['gradual_drift'] == 40.0, 'the overlay did not win'
    assert corr['mag_3d']['const_offset'] == 15.0, 'a sibling magnitude was discarded'
    assert corr['mag_3d']['sinusoid'] == 16.0
    assert corr['const_offset_prob'] == 0.5, 'the parent block was replaced wholesale'


def test_an_overlay_that_did_not_exist_in_the_base_is_added_whole(tmp_path):
    """Merging must not refuse a NEW block -- only make existing ones merge instead of replace."""
    base = tmp_path / 'base.toml'
    base.write_text('[training]\nn_iterations = 10\n')
    over = tmp_path / 'over.toml'
    over.write_text('[scorer]\npool_num_heads = 4\n\n[scorer.corruption]\nconst_offset_prob = 1.0\n')
    cfg = ck.load_config(over, base=base)
    assert cfg['scorer']['pool_num_heads'] == 4
    assert cfg['scorer']['corruption']['const_offset_prob'] == 1.0


def test_the_shipped_configs_still_merge_to_the_same_dict(tmp_path):
    """The blast-radius check: no shipped config's MEANING may change.

    The deepening is only safe because every shipped overlay uses flat top-level blocks, so the
    recursive and shallow merges agree on all of them. Asserted rather than argued -- a future
    overlay with a nested block is exactly when this needs re-checking.
    """
    def shallow(base_cfg, over):
        out = dict(base_cfg)
        for block, o in over.items():
            if isinstance(o, dict) and isinstance(out.get(block), dict):
                out[block] = {**out[block], **o}
            else:
                out[block] = o
        return out

    checked = 0
    for overlay, base in (('detector/3dpop.toml', ck._DETECTOR_CONFIG),
                          ('detector/branson-fly.toml', ck._DETECTOR_CONFIG),
                          ('detector/rat-city.toml', ck._DETECTOR_CONFIG)):
        p = _packaged(overlay)
        if not p.exists():
            continue
        with open(p, 'rb') as f:
            over = tomllib.load(f)
        with open(base, 'rb') as f:
            base_cfg = tomllib.load(f)
        assert ck._deep_merge(base_cfg, over) == shallow(base_cfg, over), \
            f'{overlay}: the deepened merge changed this shipped config'
        checked += 1
    assert checked > 0, 'no shipped overlay was checked -- this test proves nothing'


def test_the_scorer_family_base_loads_and_its_kind_is_scorer():
    """`configs/scorer.toml` must be a real, loadable member of the scorer family."""
    cfg = ck.load_config(_packaged('scorer.toml'), base=_packaged('scorer.toml'))
    assert ck.run_kind(cfg) == 'scorer'
    assert cfg['model']['stride_length'] == cfg['data']['n_frames'] == 12
    assert cfg['model']['video_encoder_requires_grad'] is False
    # the nested corruption blocks survived the merge
    assert set(cfg['scorer']['corruption']['mag_3d']) == {
        'const_offset', 'frame_noise', 'gradual_drift', 'sinusoid'}
    assert cfg['scorer']['min_valid_frames'] == 6
    # the scorer supplies no prior and no box, so every prompt lever must be OFF here
    assert cfg['data']['box_prompt'] == 'none'
    for k in ('prompt_dropout', 'prompt_noise_px', 'prompt_offset_px', 'prompt_swap_kpt_pairs',
              'prompt_swap_animal', 'box_prompt_dropout'):
        assert cfg['data'][k] == 0.0, f'{k} must be 0 for the scorer, got {cfg["data"][k]}'


# -- the kind guards ---------------------------------------------------------------------

def test_run_kind_defaults_to_pose_for_a_config_without_the_key():
    """Every run folder written before this key existed IS a pose run."""
    assert ck.run_kind({}) == 'pose'
    assert ck.run_kind({'run': {}}) == 'pose'
    assert ck.run_kind({'run': {'kind': 'scorer'}}) == 'scorer'


def test_a_scorer_run_folder_is_refused_by_the_pose_loader(tmp_path):
    """`load_run` must refuse rather than build a pose model out of scorer weights."""
    run = tmp_path / 'scorer-run'
    run.mkdir()
    (run / 'config.toml').write_text("""
[run]
kind = "scorer"

[model]
image_size = 64

[data]
image_size = 64
""")
    with pytest.raises(ValueError, match='scorer'):
        ck.load_run(run)


def test_a_pose_run_folder_is_refused_by_the_scorer_loader(tmp_path):
    run = tmp_path / 'pose-run'
    run.mkdir()
    (run / 'config.toml').write_text("""
[model]
image_size = 64

[data]
image_size = 64
""")
    with pytest.raises(ValueError, match='pose'):
        ck.load_scorer_run(run)


def test_a_checkpoint_carries_its_kind(tmp_path):
    """`save_checkpoint` must record what the weights ARE.

    Before the `kind` parameter it wrote no such field, and `_load_packaged_pose` defaults a
    missing one to `'pose'` -- so a scorer checkpoint was loadable as a pose checkpoint, silently.
    """
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

        def state_dict(self, *a, **k):
            return super().state_dict(*a, **k)

    m = _M()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    reg = Registry(names=['a', 'b'], datasets={})
    run = tmp_path / 'r'
    run.mkdir()
    p = ck.save_checkpoint(run, 0, m, opt, {'model': {}, 'data': {}}, registry=reg, kind='scorer')
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    assert ckpt['kind'] == 'scorer', 'the kind was not recorded'
    assert ckpt['keypoint_registry'] == reg.to_dict()

    # and the default is the compatibility promise
    p2 = ck.save_checkpoint(run, 0, m, opt, {'model': {}, 'data': {}}, name='b',
                            registry=reg)
    assert torch.load(p2, map_location='cpu', weights_only=False)['kind'] == 'pose'


def test_an_unknown_kind_is_refused(tmp_path):
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    m = _M()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    run = tmp_path / 'r'
    run.mkdir()
    with pytest.raises(AssertionError, match='kind must be one of'):
        ck.save_checkpoint(run, 0, m, opt, {'model': {}, 'data': {}}, kind='posed')


def test_save_run_meta_stamps_the_kind_into_the_config(tmp_path):
    """The writer decides, so a family cannot forget to declare itself."""
    import tomllib as tl

    reg = Registry(names=['a'], datasets={})
    run = tmp_path / 'r'
    ck.save_run_meta(run, {'model': {}, 'data': {}}, reg, kind='scorer')
    with open(run / 'config.toml', 'rb') as f:
        cfg = tl.load(f)
    assert cfg['run']['kind'] == 'scorer'
    assert ck.run_kind(cfg) == 'scorer'
    # the default is pose, and it does not disturb an existing [run] block's other keys
    run2 = tmp_path / 'r2'
    ck.save_run_meta(run2, {'run': {'note': 'x'}, 'model': {}, 'data': {}}, reg)
    with open(run2 / 'config.toml', 'rb') as f:
        cfg2 = tl.load(f)
    assert ck.run_kind(cfg2) == 'pose' and cfg2['run']['note'] == 'x'
