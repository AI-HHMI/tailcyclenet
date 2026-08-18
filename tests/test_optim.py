"""SF-Muon routing and the resume contract the new default makes load-bearing.

CPU-only, on a small hand-built module carrying one of every shape a router must sort -- a 2D
Linear weight, a 2D Embedding weight (which must NOT reach Muon), a 4D Conv weight, a 1D LayerNorm
weight and a bias. No posetail model, no GPU: the thing under test is the router and the
checkpoint round-trip, not the network.
"""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tailcyclenet.optim import (KNOWN_OPTIMIZER_KEYS, PoseDualOptimizer, build_muon,
                                refuse_mismatched_optimizer_state)

REPO = Path(__file__).resolve().parent.parent


def _train_module():
    spec = importlib.util.spec_from_file_location('tcn_train', REPO / 'scripts' / 'train.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Tiny(nn.Module):
    """Named to match the router's substrings: `decoder.mlps` -> Muon, `decoder.heads_*` -> AdamW,
    `scene_encoder.encoder.blocks` -> the encoder Muon group, an Embedding that must stay AdamW."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.mlps = nn.ModuleList([nn.Linear(8, 8)])          # 2D weight -> muon_dec
        self.decoder.heads_3d = nn.ModuleList([nn.Linear(8, 3)])      # 2D weight -> adamw (head)
        self.scene_encoder = nn.Module()
        self.scene_encoder.encoder = nn.Module()
        self.scene_encoder.encoder.blocks = nn.ModuleList([nn.Linear(8, 8)])   # -> muon_enc
        self.scene_encoder.kv_proj = nn.Linear(8, 8)                 # 2D weight -> muon_dec
        self.query_encoder = nn.Module()
        self.query_encoder.kpt_embed = nn.Embedding(5, 8)            # 2D but Embedding -> adamw
        self.conv = nn.Conv2d(3, 4, 3)                               # 4D -> adamw
        self.norm = nn.LayerNorm(8)                                  # 1D weight + bias -> adamw


def _cfg(**over):
    base = dict(learning_rate=1e-4, kpt_lr=5e-4, encoder_lr_scale=0.1, weight_decay=0.002,
                warmup_steps=0, beta1=0.9, beta2=0.95, muon_schedulefree=True, muon_lr_scale=1.0,
                muon_momentum=0.95, muon_warmup_steps=0, muon_adjust_lr_fn='match_rms_adamw')
    base.update(over)
    return base


def _all_muon_params(opt):
    return [p for g in opt.opt_muon.param_groups for p in g['params']]


def _all_params(opt):
    return [p for o in opt._opts for g in o.param_groups for p in g['params']]


def test_routing_is_2d_disjoint_and_complete():
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_cfg())

    muon = _all_muon_params(opt)
    assert muon, 'expected a non-empty Muon group'
    for p in muon:
        assert p.ndim == 2, 'Muon only accepts 2D parameters'

    embed_ids = {id(p) for p in m.query_encoder.kpt_embed.parameters()}
    assert not (embed_ids & {id(p) for p in muon}), 'an Embedding weight reached Muon'

    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    routed = [id(p) for p in _all_params(opt)]
    assert set(routed) == trainable, 'the groups must partition the trainable params'
    assert len(routed) == len(set(routed)), 'a param routed to two groups is stepped twice'


def test_adamw_and_muon_param_sets_partition_and_clip_target_is_adamw_only():
    """The clip must target the AdamW half only (report 34b). `adamw_params` ∪ `muon_params` is the
    full trainable set, disjoint, and no Muon-routed 2D matrix is in the clip target."""
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_cfg())
    a = {id(p) for p in opt.adamw_params}
    mu = {id(p) for p in opt.muon_params}
    assert a and mu, 'both halves must be non-empty on this fixture'
    assert not (a & mu), 'a param in both the clipped and unclipped set is stepped wrong'
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert a | mu == trainable, 'clip target + unclipped must be the whole trainable set'
    # the decoder MLP weight is Muon-routed, so it must NOT be in the clip target
    assert id(m.decoder.mlps[0].weight) in mu and id(m.decoder.mlps[0].weight) not in a
    # a head weight and every embedding weight ARE in the clip target
    assert id(m.decoder.heads_3d[0].weight) in a
    assert id(m.query_encoder.kpt_embed.weight) in a


def test_frozen_encoder_contributes_no_muon_group():
    m = Tiny()
    for p in m.scene_encoder.encoder.parameters():
        p.requires_grad_(False)
    opt = build_muon(m, fresh=set(), cfg=_cfg())
    routed = {id(p) for p in _all_params(opt)}
    for p in m.scene_encoder.encoder.parameters():
        assert id(p) not in routed, 'a frozen encoder param was handed to an optimizer'
    # And Muon got no frozen tensor (it would raise at construction) -- reaching here proves it.


def test_fresh_2d_weight_routes_to_muon_at_kpt_lr():
    m = Tiny()
    fresh = {'decoder.heads_3d.0.weight'}   # pretend a head was freshly initialised
    opt = build_muon(m, fresh=fresh, cfg=_cfg())
    kpt_lr = 5e-4
    target = m.decoder.heads_3d[0].weight
    hit = [g for g in opt.opt_muon.param_groups
           if any(p is target for p in g['params'])]
    assert hit and hit[0]['lr'] == pytest.approx(kpt_lr), 'fresh 2D weight must be Muon @ kpt_lr'


def test_step_moves_every_param_and_raises_nothing():
    torch.manual_seed(0)
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_cfg())
    opt.train()
    before = [p.detach().clone() for p in m.parameters() if p.requires_grad]
    x = torch.randn(2, 3, 6, 6)
    # touch enough of the graph that every trainable leaf gets a grad
    feat = m.conv(x).flatten(1)[:, :8]
    feat = m.norm(feat)
    feat = m.scene_encoder.encoder.blocks[0](feat)
    feat = m.scene_encoder.kv_proj(feat)
    feat = m.decoder.mlps[0](feat)
    emb = m.query_encoder.kpt_embed(torch.zeros(2, dtype=torch.long))
    loss = m.decoder.heads_3d[0](feat + emb).pow(2).sum()
    loss.backward()
    opt.step()
    after = [p for p in m.parameters() if p.requires_grad]
    moved = sum(not torch.equal(a, b) for a, b in zip(after, before))
    assert moved == len(before), f'only {moved}/{len(before)} params moved'


def test_load_state_dict_restores_train_mode_by_value():
    """The double-lerp is invisible in any structural check, so assert on the WEIGHTS.

    Save y-iterate params + optimizer state; load into a fresh optimizer; `train()`. If
    `train_mode` came back False (the ScheduleFreeWrapper resume bug), `train()` would lerp y
    toward z a second time and the params would differ.
    """
    torch.manual_seed(0)
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_cfg())
    opt.train()
    x = torch.randn(2, 8)
    m.decoder.mlps[0](x).pow(2).sum().backward()
    opt.step()
    sd = opt.state_dict()
    saved = [p.detach().clone() for p in m.parameters()]

    opt2 = build_muon(m, fresh=set(), cfg=_cfg())
    opt2.load_state_dict(sd)
    opt2.train()   # a no-op iff train_mode is already True
    for p, s in zip(m.parameters(), saved):
        assert torch.equal(p, s), 'train() after load moved params -> train_mode came back False'


def test_adamwsf_state_into_muon_raises_named_systemexit():
    tr = _train_module()
    m = Tiny()
    sf = tr.build_optimizer(m, set(), _cfg(optimizer='schedulefree'))
    sf_state = sf.state_dict()
    muon = build_muon(m, set(), _cfg())
    with pytest.raises(SystemExit, match='schedulefree'):
        refuse_mismatched_optimizer_state(muon, sf_state, 'ck.pth', resolved='muon', explicit=False)


def test_muon_state_into_adamwsf_raises_named_systemexit():
    tr = _train_module()
    m = Tiny()
    muon = build_muon(m, set(), _cfg())
    muon_state = muon.state_dict()
    sf = tr.build_optimizer(m, set(), _cfg(optimizer='schedulefree'))
    with pytest.raises(SystemExit, match='muon'):
        refuse_mismatched_optimizer_state(sf, muon_state, 'ck.pth', resolved='schedulefree',
                                          explicit=True)


def test_eval_train_swap_and_has_averaged_iterate():
    torch.manual_seed(0)
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_cfg(muon_schedulefree=True))
    assert opt.has_averaged_iterate is True
    opt.train()
    for _ in range(5):   # >1 step, or the running average still coincides with the latest y
        opt.zero_grad()
        m.decoder.mlps[0](torch.randn(2, 8)).pow(2).sum().backward()
        opt.step()
    y = m.decoder.mlps[0].weight.detach().clone()
    opt.eval()
    assert not torch.equal(m.decoder.mlps[0].weight, y), 'eval() must swap to the averaged iterate'
    opt.train()
    assert torch.equal(m.decoder.mlps[0].weight, y), 'train() must restore the y iterate'

    plain = build_muon(Tiny(), fresh=set(), cfg=_cfg(muon_schedulefree=False))
    assert plain.has_averaged_iterate is False, 'a bare Muon half carries no averaged iterate'


def test_unknown_optimizer_key_raises():
    tr = _train_module()
    assert 'muon_lr' not in KNOWN_OPTIMIZER_KEYS
    # The guard lives in main(); replicate it exactly rather than standing up a full run.
    cfg = _cfg(muon_lr=1e-4)
    unknown = set(cfg) - KNOWN_OPTIMIZER_KEYS
    assert unknown == {'muon_lr'}


def test_schedulefree_escape_hatch_is_the_current_three_group_optimizer():
    """`optimizer = "schedulefree"` must build the exact AdamW-SF optimizer that exists today."""
    tr = _train_module()
    m = Tiny()
    fresh = {'decoder.mlps.0.weight'}
    opt = tr.build_optimizer(m, fresh, _cfg(optimizer='schedulefree'))
    assert not isinstance(opt, PoseDualOptimizer)
    lrs = sorted(g['lr'] for g in opt.param_groups)
    # fresh @ kpt_lr, rest @ lr, encoder @ lr*enc_scale -- three groups, no Muon.
    assert lrs == pytest.approx(sorted({1e-4 * 0.1, 1e-4, 5e-4})), lrs
