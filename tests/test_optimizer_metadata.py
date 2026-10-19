"""Named optimizer-layout metadata used to make pose resumes state-safe."""
import torch

from tailcyclenet.checkpoints import save_checkpoint
from tailcyclenet.format import Registry
from tailcyclenet.optim import build_muon, build_optimizer, check_resume
from tailcyclenet.unfreeze import apply_staged_unfreeze

from tests.test_optim import Tiny, _cfg


def _staged_checkpoint(tmp_path):
    cfg = _cfg(muon_warmup_steps=10)
    fresh = {'decoder.mlps.0.weight'}
    model = Tiny(unfreeze_at=4, n_last=3)
    opt = build_muon(model, fresh, cfg)
    apply_staged_unfreeze(model, opt, cfg, 14, fresh=fresh)
    path = save_checkpoint(tmp_path / 'run', 14, model, opt, {'model': {}, 'data': {}},
                           registry=Registry(names=['a'], datasets={}), fresh_names=fresh,
                           world_size=2)
    return path, model, opt, cfg, fresh


def test_optimizer_metadata_records_fresh_names_and_world_size(tmp_path):
    path, _, _, _, fresh = _staged_checkpoint(tmp_path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    layout = checkpoint['optimizer_metadata']
    assert set(layout['fresh_names']) == fresh
    assert layout['schema'] == 'tailcyclenet.optimizer-layout.v1'
    assert layout['world_size'] == 2


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
    layout = checkpoint['optimizer_metadata']
    recovered = set(layout['fresh_names'])
    model = Tiny(unfreeze_at=4, n_last=3)
    opt = build_optimizer(model, recovered, cfg)
    apply_staged_unfreeze(model, opt, cfg, 14, fresh=recovered)
    assert check_resume(opt, checkpoint['optimizer_state'], model, layout)
    other = Tiny(unfreeze_at=4, n_last=3)
    opt2 = build_optimizer(other, set(), cfg)
    apply_staged_unfreeze(other, opt2, cfg, 14, fresh=set())
    verdict = check_resume(opt2, checkpoint['optimizer_state'], other, layout)
    assert not verdict, 'a run routed with different fresh names must not resume'


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
    assert check_resume(resumed, state)
    resumed.load_state_dict(state)
    assert resumed._gstep == 3
    resumed.train()
    resumed.zero_grad()
    resumed._opts[0].param_groups[0]['params'][0].sum().backward()
    resumed.step()
    assert resumed.opt_muon.param_groups[0]['lr'] == 4e-5
