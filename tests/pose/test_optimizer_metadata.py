"""Named optimizer-layout metadata: what makes a pose resume state-safe."""
import torch

from tailcyclenet.checkpoints import save_checkpoint
from tailcyclenet.format import Registry
from tailcyclenet.pose.optim import build_muon, build_optimizer, check_resume
from tailcyclenet.pose.unfreeze import apply_staged_unfreeze
from tests.helpers import Tiny, _optim_cfg


def test_layout_metadata_is_recorded_and_gates_the_resume(tmp_path):
    """A staged checkpoint records its fresh names, world size and 3 Muon + 3 AdamW named groups;
    a rebuild routed with the same fresh names resumes, one routed differently does not."""
    cfg = _optim_cfg(muon_warmup_steps=10)
    fresh = {'decoder.mlps.0.weight'}
    model = Tiny(unfreeze_at=4, n_last=3)
    opt = build_muon(model, fresh, cfg)
    apply_staged_unfreeze(model, opt, cfg, 14, fresh=fresh)
    path = save_checkpoint(tmp_path / 'run', 14, model, opt, {'model': {}, 'data': {}},
                           registry=Registry(names=['a'], datasets={}), fresh_names=fresh,
                           world_size=2)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    layout = checkpoint['optimizer_metadata']
    assert layout['schema'] == 'tailcyclenet.optimizer-layout.v1'
    assert set(layout['fresh_names']) == fresh and layout['world_size'] == 2
    groups = layout['groups']
    assert len(groups['muon']) == 3 and len(groups['adamw']) == 3
    assert any('decoder.mlps.0.weight' in g['names'] for g in groups['muon'])

    for names, ok in ((set(layout['fresh_names']), True), (set(), False)):
        m = Tiny(unfreeze_at=4, n_last=3)
        o = build_optimizer(m, names, cfg)
        apply_staged_unfreeze(m, o, cfg, 14, fresh=names)
        assert bool(check_resume(o, checkpoint['optimizer_state'], m, layout)) is ok, names


def test_muon_warmup_state_round_trip_preserves_next_rate():
    cfg = _optim_cfg(muon_warmup_steps=10)
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
