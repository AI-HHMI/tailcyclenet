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
                                refuse_group_count_mismatch, refuse_mismatched_optimizer_state)
from tailcyclenet.unfreeze import apply_staged_unfreeze, replay_staged_unfreeze

REPO = Path(__file__).resolve().parent.parent


def _train_module():
    spec = importlib.util.spec_from_file_location('tcn_train', REPO / 'scripts' / 'train.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TinyEncoder(nn.Module):
    """The shape `unfreeze.py` reads: `blocks`, `norms_block` and `hierarchical_layers`.

    Six blocks with the hierarchical taps at [1,3,4,5], so the last-N rule has something to select
    other than everything -- at N = 3 the trainable range is blocks 3..5, i.e. norms 1,2,3.
    """

    def __init__(self, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(depth)])
        self.hierarchical_layers = [1, 3, 4, 5]
        self.norms_block = nn.ModuleList([nn.LayerNorm(8) for _ in self.hierarchical_layers])


class Tiny(nn.Module):
    """Named to match the router's substrings: `decoder.mlps` -> Muon, `decoder.heads_*` -> AdamW,
    `scene_encoder.encoder.blocks` -> the encoder Muon group, an Embedding that must stay AdamW.

    It also stands in for `TrackerEncoder`'s staged-unfreeze contract: `unfreeze_video_encoder`
    has the same signature, gate and idempotence as upstream's, and re-freezes before unfreezing
    the last N exactly as `set_encoder_requires_grad` does -- which is what makes the norms
    extension order-dependent.
    """

    def __init__(self, unfreeze_at=None, n_last=3):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.mlps = nn.ModuleList([nn.Linear(8, 8)])          # 2D weight -> muon_dec
        self.decoder.heads_3d = nn.ModuleList([nn.Linear(8, 3)])      # 2D weight -> adamw (head)
        self.scene_encoder = nn.Module()
        self.scene_encoder.encoder = TinyEncoder()                    # blocks -> muon_enc
        self.scene_encoder.kv_proj = nn.Linear(8, 8)                 # 2D weight -> muon_dec
        self.query_encoder = nn.Module()
        self.query_encoder.kpt_embed = nn.Embedding(5, 8)            # 2D but Embedding -> adamw
        self.conv = nn.Conv2d(3, 4, 3)                               # 4D -> adamw
        self.norm = nn.LayerNorm(8)                                  # 1D weight + bias -> adamw
        self.video_encoder_unfreeze_iter = unfreeze_at
        self.video_encoder_finetune_last_n_layers = n_last
        self.video_encoder_requires_grad = unfreeze_at is None
        if unfreeze_at is not None:
            for p in self.scene_encoder.encoder.parameters():
                p.requires_grad_(False)

    def unfreeze_video_encoder(self, iteration):
        """Upstream's contract verbatim (tracker_encoder.py:262)."""
        if self.video_encoder_unfreeze_iter is None:
            return False
        if self.video_encoder_requires_grad:
            return False
        if iteration < self.video_encoder_unfreeze_iter:
            return False
        enc = self.scene_encoder.encoder
        for p in enc.parameters():
            p.requires_grad_(False)
        for blk in enc.blocks[-self.video_encoder_finetune_last_n_layers:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        self.video_encoder_requires_grad = True
        return True


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
    # touch enough of the graph that every trainable leaf gets a grad -- including EVERY encoder
    # block and hierarchical norm, which is the whole encoder here (unfreeze_at=None -> trainable)
    feat = m.conv(x).flatten(1)[:, :8]
    feat = m.norm(feat)
    for blk in m.scene_encoder.encoder.blocks:
        feat = blk(feat)
    for nrm in m.scene_encoder.encoder.norms_block:
        feat = nrm(feat)
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


# ---------------------------------------------------------------------------------------------
# the staged encoder unfreeze
# ---------------------------------------------------------------------------------------------

UNFREEZE_AT, N_LAST = 4, 3


def _staged(**over):
    """A model frozen at step 0 with the unfreeze scheduled, plus its optimizer."""
    m = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    cfg = _cfg(**over)
    return m, build_muon(m, fresh=set(), cfg=cfg), cfg


def _enc_tensors(m):
    return dict(m.scene_encoder.encoder.named_parameters())


def test_unfreeze_puts_every_new_param_in_exactly_one_group():
    """THE NO-OP THIS FIXES. `build_muon` filters `requires_grad`, so a param frozen at step 0 is
    in no group -- flipping the flag alone would give it gradients that nothing steps."""
    m, opt, cfg = _staged()
    routed = {id(p) for p in _all_params(opt)}
    assert not (routed & {id(p) for p in m.scene_encoder.encoder.parameters()}), \
        'a frozen encoder param was routed at build time'

    info = apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert info and info['blocks'] == [3, 4, 5], info

    routed = [id(p) for p in _all_params(opt)]
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert set(routed) == trainable, 'the groups must partition the trainable params after a fire'
    assert len(routed) == len(set(routed)), 'a param routed twice is stepped twice'
    a = {id(p) for p in opt.adamw_params}
    mu = {id(p) for p in opt.muon_params}
    assert not (a & mu) and a | mu == trainable


def test_unfreeze_extends_to_the_hierarchical_norms_in_range():
    """Upstream unfreezes blocks plus an `encoder.norm` VJEPA 2.1 does not have, so `norms_block`
    -- which feeds the decoder -- would stay frozen behind a trainable block. Taps are [1,3,4,5];
    at N = 3 the trainable range starts at block 3, so norms 1,2,3 qualify and norm 0 does not."""
    m, opt, cfg = _staged()
    info = apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert info['norms'] == [1, 2, 3], info['norms']
    norms = m.scene_encoder.encoder.norms_block
    assert not any(p.requires_grad for p in norms[0].parameters()), 'norm 0 is below the range'
    for i in (1, 2, 3):
        assert all(p.requires_grad for p in norms[i].parameters()), f'norm {i} should be trainable'


def test_unfreeze_is_gated_and_idempotent():
    m, opt, cfg = _staged()
    before = (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups))
    assert apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT - 1) is None, 'fired early'
    assert (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups)) == before
    assert apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT) is not None
    after = (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups))
    assert after == (before[0] + 1, before[1] + 1), after
    assert apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT + 1) is None, 'fired twice'
    assert (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups)) == after


def test_added_groups_start_a_fresh_schedule():
    """NEW GROUPS, NOT PRE-REGISTERED ONES. Both schedule-free impls advance `k`/`weight_sum` on
    every `step()` regardless of whether a param has a grad, and the averaging weight is
    `ckp1 = weight/weight_sum ~ 1/k` -- so a group that sat grad-less would fold its encoder into
    the averaged iterate at ~1/k and `model_state_eval` would hold a barely-moved encoder."""
    m, opt, cfg = _staged()
    opt.train()
    for _ in range(UNFREEZE_AT):
        opt.zero_grad()
        m.decoder.mlps[0](torch.randn(2, 8)).pow(2).sum().backward()
        opt.step()
    assert opt.opt_adam.param_groups[0]['k'] == UNFREEZE_AT, 'the existing group did step'
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    new = opt.opt_adam.param_groups[-1]
    assert new['k'] == 0 and new['weight_sum'] == 0.0, 'the added group must start its own average'


def test_added_adamw_group_is_in_phase_with_the_existing_ones():
    """`eval()`/`train()` read `train_mode` PER GROUP, so a group added out of phase is lerped the
    wrong way at the next checkpoint write. Asserted on VALUES, like the resume test."""
    m, opt, cfg = _staged()
    opt.train()
    for _ in range(UNFREEZE_AT):
        opt.zero_grad()
        m.decoder.mlps[0](torch.randn(2, 8)).pow(2).sum().backward()
        opt.step()
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert all(g['train_mode'] for g in opt.opt_adam.param_groups), 'a group is out of phase'
    before = [p.detach().clone() for p in m.parameters()]
    opt.eval()
    opt.train()
    for p, b in zip(m.parameters(), before):
        assert torch.equal(p, b), 'eval()+train() must round-trip every group'


def test_step_moves_the_unfrozen_blocks_and_leaves_the_rest():
    m, opt, cfg = _staged()
    opt.train()

    def _step():
        opt.zero_grad()
        enc = m.scene_encoder.encoder
        x = torch.randn(2, 8)
        for blk in enc.blocks:
            x = blk(x)
        for nrm in enc.norms_block:
            x = nrm(x)
        m.decoder.mlps[0](x).pow(2).sum().backward()
        opt.step()

    frozen_before = {k: v.detach().clone() for k, v in _enc_tensors(m).items()}
    for _ in range(UNFREEZE_AT):
        _step()
    for k, v in _enc_tensors(m).items():
        assert torch.equal(v, frozen_before[k]), f'{k} moved while the encoder was frozen'

    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    before = {k: v.detach().clone() for k, v in _enc_tensors(m).items()}
    for _ in range(3):
        _step()
    for k, v in _enc_tensors(m).items():
        moved = not torch.equal(v, before[k])
        should = dict(_enc_tensors(m))[k].requires_grad
        assert moved == should, f'{k}: moved={moved} but requires_grad={should}'


def test_add_muon_group_refuses_a_non_2d_param():
    """`torch.optim.Muon` validates ndim in `__init__` only; `add_param_group` does not re-check,
    so without this the slip surfaces as a shape error deep inside `step()`."""
    m, opt, _ = _staged()
    with pytest.raises(AssertionError, match='2D'):
        opt.add_muon_group([torch.zeros(4, requires_grad=True)], lr=1e-5, weight_decay=0.0)


def test_muon_base_lrs_stays_aligned_after_an_add():
    """`DualOptimizer.step` zips `_muon_base_lrs` against `opt_muon.param_groups` for the warmup
    rescale, and `zip` truncates SILENTLY -- an unappended group escapes the rescale unannounced."""
    m, opt, cfg = _staged(muon_warmup_steps=2)
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert len(opt._muon_base_lrs) == len(opt.opt_muon.param_groups)
    enc_lr = cfg['learning_rate'] * cfg['encoder_lr_scale']
    assert opt._muon_base_lrs[-1] == pytest.approx(enc_lr), 'the encoder group joins at the scale'


def test_resume_replays_the_unfreeze_and_round_trips_the_state():
    """A run resumed past its unfreeze must reach the layout a fresh run holds at that iteration.
    Groups match BY POSITION, so this asserts the replay produces the same order AND that
    `train()` after the load moves nothing (the ScheduleFreeWrapper `train_mode` bug)."""
    torch.manual_seed(0)
    m, opt, cfg = _staged()
    opt.train()
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    m.decoder.mlps[0](torch.randn(2, 8)).pow(2).sum().backward()
    opt.step()
    sd = opt.state_dict()
    saved = [p.detach().clone() for p in m.parameters()]

    m2 = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    opt2 = build_muon(m2, fresh=set(), cfg=cfg)
    replay_staged_unfreeze(m2, opt2, cfg, UNFREEZE_AT + 10)
    opt2.load_state_dict(sd)
    opt2.train()   # a no-op iff train_mode came back True
    for p, s in zip(m.parameters(), saved):
        assert torch.equal(p, s), 'train() after load moved params -> train_mode came back False'


def test_resume_without_replay_raises_a_named_group_count_error():
    """Torch's own message ('different number of parameter groups') is true and useless about the
    cause. This is the gridresid_offset rule for the group layout."""
    m, opt, cfg = _staged()
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    sd = opt.state_dict()
    fresh_opt = build_muon(Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST), fresh=set(), cfg=cfg)
    with pytest.raises(SystemExit, match='video_encoder_finetune_last_n_layers'):
        refuse_group_count_mismatch(fresh_opt, sd)
    with pytest.raises(SystemExit, match='video_encoder_finetune_last_n_layers'):
        fresh_opt.load_state_dict(sd)


def test_schedulefree_optimizer_takes_the_same_staged_path():
    tr = _train_module()
    m = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    cfg = _cfg(optimizer='schedulefree')
    opt = tr.build_optimizer(m, set(), cfg)
    assert not isinstance(opt, PoseDualOptimizer)
    before = len(opt.param_groups)
    info = apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert info and len(opt.param_groups) == before + 1, 'one encoder group, Muon routes collapse'
    enc_lr = cfg['learning_rate'] * cfg['encoder_lr_scale']
    assert opt.param_groups[-1]['lr'] == pytest.approx(enc_lr)
    assert opt.param_groups[-1]['k'] == 0
    routed = {id(p) for g in opt.param_groups for p in g['params']}
    assert routed == {id(p) for p in m.parameters() if p.requires_grad}


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
