"""Named optimizer-layout metadata used to make pose resumes state-safe."""
import copy

import torch

from tailcyclenet.checkpoints import save_checkpoint
from tailcyclenet.format import Registry
from tailcyclenet.optim import (
    build_muon,
    optimizer_layout_matches,
    optimizer_layout_matches_metadata,
    optimizer_metadata_from_checkpoint,
)
from tailcyclenet.unfreeze import replay_staged_unfreeze

from tests.test_optim import Tiny, _cfg, _train_module


def _staged_checkpoint(tmp_path):
    cfg = _cfg(muon_warmup_steps=10)
    fresh = {'decoder.mlps.0.weight'}
    model = Tiny(unfreeze_at=4, n_last=3)
    opt = build_muon(model, fresh, cfg)
    replay_staged_unfreeze(model, opt, cfg, 14, fresh=fresh)
    path = save_checkpoint(tmp_path / 'run', 14, model, opt, {'model': {}, 'data': {}},
                           registry=Registry(names=['a'], datasets={}), fresh_names=fresh)
    return path, model, opt, cfg, fresh


def test_optimizer_metadata_round_trip_and_source_is_unchanged(tmp_path):
    path, _, _, _, fresh = _staged_checkpoint(tmp_path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    original = copy.deepcopy(checkpoint)
    got_fresh, layout = optimizer_metadata_from_checkpoint(checkpoint)
    assert got_fresh == fresh
    assert layout['schema'] == 'tailcyclenet.optimizer-layout.v1'
    assert checkpoint.keys() == original.keys()
    assert checkpoint['optimizer_metadata'] == original['optimizer_metadata']


def test_staged_layout_records_three_muon_and_two_adamw_groups(tmp_path):
    path, _, _, _, _ = _staged_checkpoint(tmp_path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    layout = checkpoint['optimizer_metadata']['groups']
    assert len(layout['muon']) == 3
    assert len(layout['adamw']) == 2
    assert any('decoder.mlps.0.weight' in g['names'] for g in layout['muon'])


def test_resume_rebuild_matches_named_layout_and_lrs(tmp_path):
    path, _, _, cfg, fresh = _staged_checkpoint(tmp_path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    recovered, layout = optimizer_metadata_from_checkpoint(checkpoint)
    model = Tiny(unfreeze_at=4, n_last=3)
    opt = _train_module().build_optimizer(model, recovered, cfg, layout=layout)
    replay_staged_unfreeze(model, opt, cfg, 14, fresh=recovered, layout=layout)
    assert optimizer_layout_matches_metadata(opt, model, layout)
    assert optimizer_layout_matches(opt, checkpoint['optimizer_state'])


def test_old_checkpoint_without_metadata_uses_legacy_fallback():
    old = {'model_state': {}, 'optimizer_state': {'muon': {}, 'adam': {}}, 'iteration': 1}
    fresh, layout = optimizer_metadata_from_checkpoint(old)
    assert fresh == set()
    assert layout is None


def test_muon_warmup_state_round_trip_preserves_next_rate():
    cfg = _cfg(muon_warmup_steps=10)
    model = Tiny()
    opt = build_muon(model, set(), cfg)
    opt.train()
    for _ in range(3):
        opt.zero_grad()
        model.decoder.mlps[0](torch.randn(2, 8)).pow(2).sum().backward()
        opt.step()
    state = opt.state_dict()
    resumed = build_muon(Tiny(), set(), cfg)
    assert optimizer_layout_matches(resumed, state)
    resumed.load_state_dict(state)
    assert resumed._gstep == 3
    resumed.train()
    resumed.zero_grad()
    resumed._opts[0].param_groups[0]['params'][0].sum().backward()
    resumed.step()
    assert resumed.opt_muon.param_groups[0]['lr'] == 4e-5
