"""SF-Muon routing, the staged encoder unfreeze, and the resume contract, on a tiny CPU module
carrying every shape the router sorts (2D Linear, 2D Embedding that must NOT reach Muon, 4D Conv,
1D norm, bias)."""
import pytest
import torch

from tailcyclenet.pose.optim import (
    PoseDualOptimizer,
    build_muon,
    build_optimizer,
    check_resume,
    state_kind,
)
from tailcyclenet.pose.unfreeze import apply_staged_unfreeze
from tests.helpers import N_LAST, UNFREEZE_AT, Tiny, _optim_cfg


def _all_muon_params(opt):
    return [p for g in opt.opt_muon.param_groups for p in g['params']]


def _all_params(opt):
    return [p for o in opt._opts for g in o.param_groups for p in g['params']]


def test_routing_is_2d_disjoint_complete_and_clips_only_adamw():
    """Muon gets only 2D non-Embedding params; groups partition the trainable set; the clip target
    is the AdamW half only (Muon orthogonalises its own gradients)."""
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_optim_cfg())

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

    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_optim_cfg())
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


def test_frozen_encoder_is_unrouted_and_fresh_2d_goes_to_muon_at_kpt_lr():
    """A frozen param is in no group (Muon would raise); a fresh 2D weight is Muon at kpt_lr."""
    m = Tiny()
    for p in m.scene_encoder.encoder.parameters():
        p.requires_grad_(False)
    opt = build_muon(m, fresh=set(), cfg=_optim_cfg())
    routed = {id(p) for p in _all_params(opt)}
    for p in m.scene_encoder.encoder.parameters():
        assert id(p) not in routed, 'a frozen encoder param was handed to an optimizer'
    # And Muon got no frozen tensor (it would raise at construction) -- reaching here proves it.

    m = Tiny()
    fresh = {'decoder.heads_3d.0.weight'}   # pretend a head was freshly initialised
    opt = build_muon(m, fresh=fresh, cfg=_optim_cfg())
    kpt_lr = 5e-4
    target = m.decoder.heads_3d[0].weight
    hit = [g for g in opt.opt_muon.param_groups
           if any(p is target for p in g['params'])]
    assert hit and hit[0]['lr'] == pytest.approx(kpt_lr), 'fresh 2D weight must be Muon @ kpt_lr'


def test_step_moves_every_param_and_raises_nothing():
    """One step moves every trainable param."""
    torch.manual_seed(0)
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_optim_cfg())
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


def test_optimizer_kind_mismatch_is_refused_by_name():
    """`state_kind` reads the writer (dual {'muon','adam'} vs plain); a kind mismatch fails
    `check_resume` naming the kind, both directions."""
    assert state_kind({'muon': {'param_groups': []}, 'adam': {'param_groups': []}}) == 'muon'
    assert state_kind({'param_groups': []}) == 'schedulefree'

    m = Tiny()
    sf = build_optimizer(m, set(), _optim_cfg(optimizer='schedulefree'))
    sf_state = sf.state_dict()
    muon = build_muon(m, set(), _optim_cfg())
    verdict = check_resume(muon, sf_state)
    assert not verdict and 'schedulefree' in verdict.reason
    with pytest.raises(SystemExit, match='schedulefree'):
        muon.load_state_dict(sf_state)

    m = Tiny()
    muon = build_muon(m, set(), _optim_cfg())
    muon_state = muon.state_dict()
    sf = build_optimizer(m, set(), _optim_cfg(optimizer='schedulefree'))
    verdict = check_resume(sf, muon_state)
    assert not verdict and '"muon"' in verdict.reason


def test_eval_train_swap_and_has_averaged_iterate():
    """eval() swaps to the averaged iterate and train() restores y; bare Muon has none."""
    torch.manual_seed(0)
    m = Tiny()
    opt = build_muon(m, fresh=set(), cfg=_optim_cfg(muon_schedulefree=True))
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

    plain = build_muon(Tiny(), fresh=set(), cfg=_optim_cfg(muon_schedulefree=False))
    assert plain.has_averaged_iterate is False, 'a bare Muon half carries no averaged iterate'


def _staged(**over):
    """A model frozen at step 0 with the unfreeze scheduled, plus its optimizer."""
    m = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    cfg = _optim_cfg(**over)
    return m, build_muon(m, fresh=set(), cfg=cfg), cfg


def _enc_tensors(m):
    return dict(m.scene_encoder.encoder.named_parameters())


def test_unfreeze_routes_every_new_param_once_including_hierarchical_norms():
    """A param frozen at step 0 is in no group, so the flag flip alone is a no-op; the fire adds it
    exactly once, and extends to norms_block taps in range (taps [1,3,4,5], N=3 -> norms 1-3)."""
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

    m, opt, cfg = _staged()
    info = apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert info['norms'] == [1, 2, 3], info['norms']
    norms = m.scene_encoder.encoder.norms_block
    assert not any(p.requires_grad for p in norms[0].parameters()), 'norm 0 is below the range'
    for i in (1, 2, 3):
        assert all(p.requires_grad for p in norms[i].parameters()), f'norm {i} should be trainable'


def test_unfreeze_is_gated_and_idempotent():
    """Fires at its iteration, once."""
    m, opt, cfg = _staged()
    before = (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups))
    assert apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT - 1) is None, 'fired early'
    assert (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups)) == before
    assert apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT) is not None
    after = (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups))
    assert after == (before[0] + 1, before[1] + 1), after
    assert apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT + 1) is None, 'fired twice'
    assert (len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups)) == after


def test_added_groups_start_a_fresh_schedule_in_phase():
    """Added groups are NEW (k=0; a pre-registered grad-less group would fold in at ~1/k) and share
    train_mode with the rest, so eval()+train() round-trips every value."""
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
    assert all(g['train_mode'] for g in opt.opt_adam.param_groups), 'a group is out of phase'
    before = [p.detach().clone() for p in m.parameters()]
    opt.eval()
    opt.train()
    for p, b in zip(m.parameters(), before):
        assert torch.equal(p, b), 'eval()+train() must round-trip every group'


def test_step_moves_the_unfrozen_blocks_and_leaves_the_rest():
    """Frozen tensors never move; after the fire exactly the requires_grad ones do."""
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


def test_muon_add_guards():
    """add_param_group skips Muon's 2D check (would fail deep in step()), and `_muon_base_lrs`
    must grow with the groups (zip truncates silently in the warmup rescale)."""
    m, opt, _ = _staged()
    with pytest.raises(AssertionError, match='2D'):
        opt.add_muon_group([torch.zeros(4, requires_grad=True)], lr=1e-5, weight_decay=0.0)
    m, opt, cfg = _staged(muon_warmup_steps=2)
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert len(opt._muon_base_lrs) == len(opt.opt_muon.param_groups)
    enc_lr = cfg['learning_rate'] * cfg['encoder_lr_scale']
    assert opt._muon_base_lrs[-1] == pytest.approx(enc_lr), 'the encoder group joins at the scale'


def test_resume_replays_the_unfreeze_and_round_trips_the_state():
    """Resumed past the unfreeze, the replay reproduces the group order and train() after load
    moves nothing (the ScheduleFreeWrapper train_mode bug)."""
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
    apply_staged_unfreeze(m2, opt2, cfg, UNFREEZE_AT + 10)
    opt2.load_state_dict(sd)
    opt2.train()   # a no-op iff train_mode came back True
    for p, s in zip(m.parameters(), saved):
        assert torch.equal(p, s), 'train() after load moved params -> train_mode came back False'


def test_resume_without_replay_and_pre_dual_states_are_refused_by_name():
    """Torch's group-count message is true and useless; ours names the key. Pre-`pose-dual.v1`
    states are refused (v0.3.4 is the last release that resumed them)."""
    m, opt, cfg = _staged()
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    sd = opt.state_dict()
    fresh_opt = build_muon(Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST), fresh=set(), cfg=cfg)
    assert 'video_encoder_finetune_last_n_layers' in check_resume(fresh_opt, sd).reason
    with pytest.raises(SystemExit, match='video_encoder_finetune_last_n_layers'):
        fresh_opt.load_state_dict(sd)

    m, opt, cfg = _staged()
    sd = opt.state_dict()
    sd.pop('_tailcyclenet')
    verdict = check_resume(opt, sd)
    assert not verdict and 'v0.3.4' in verdict.reason


def test_check_resume_compares_groups_by_position():
    """Groups load BY POSITION, so the per-group lr sequence must match on both halves."""
    torch.manual_seed(0)
    m, opt, cfg = _staged()
    apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    sd = opt.state_dict()
    assert check_resume(opt, sd)

    # The replay reproduces the layout a run at the same unfreeze point would have.
    m2 = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    opt2 = build_muon(m2, fresh=set(), cfg=cfg)
    apply_staged_unfreeze(m2, opt2, cfg, UNFREEZE_AT + 10)
    assert check_resume(opt2, sd)

    # A different lr is a different layout: loading would put the state on the wrong groups.
    m3 = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    cfg3 = {**cfg, 'learning_rate': 2e-4}
    opt3 = build_muon(m3, fresh=set(), cfg=cfg3)
    apply_staged_unfreeze(m3, opt3, cfg3, UNFREEZE_AT + 10)
    assert not check_resume(opt3, sd)

    # A run that never fired the unfreeze cannot resume one that did.
    m4 = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    opt4 = build_muon(m4, fresh=set(), cfg=cfg)
    assert not check_resume(opt4, sd)

    # Same rule on the plain AdamW-SF optimizer.
    opt5 = build_optimizer(Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST), set(),
                              {**cfg, 'optimizer': 'schedulefree'})
    assert check_resume(opt5, opt5.state_dict())
    opt6 = build_optimizer(Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST), set(),
                              {**cfg, 'optimizer': 'schedulefree', 'kpt_lr': 9e-4})
    assert not check_resume(opt6, opt5.state_dict())


def test_schedulefree_escape_hatch_three_groups_and_staged_path():
    """optimizer = "schedulefree" builds fresh@kpt_lr / rest@lr / encoder@lr*scale and takes the
    same staged unfreeze as one new encoder group."""
    m = Tiny()
    fresh = {'decoder.mlps.0.weight'}
    opt = build_optimizer(m, fresh, _optim_cfg(optimizer='schedulefree'))
    assert not isinstance(opt, PoseDualOptimizer)
    lrs = sorted(g['lr'] for g in opt.param_groups)
    # fresh @ kpt_lr, rest @ lr, encoder @ lr*enc_scale -- three groups, no Muon.
    assert lrs == pytest.approx(sorted({1e-4 * 0.1, 1e-4, 5e-4})), lrs

    m = Tiny(unfreeze_at=UNFREEZE_AT, n_last=N_LAST)
    cfg = _optim_cfg(optimizer='schedulefree')
    opt = build_optimizer(m, set(), cfg)
    assert not isinstance(opt, PoseDualOptimizer)
    before = len(opt.param_groups)
    info = apply_staged_unfreeze(m, opt, cfg, UNFREEZE_AT)
    assert info and len(opt.param_groups) == before + 1, 'one encoder group, Muon routes collapse'
    enc_lr = cfg['learning_rate'] * cfg['encoder_lr_scale']
    assert opt.param_groups[-1]['lr'] == pytest.approx(enc_lr)
    assert opt.param_groups[-1]['k'] == 0
    routed = {id(p) for g in opt.param_groups for p in g['params']}
    assert routed == {id(p) for p in m.parameters() if p.requires_grad}


# --- fresh non-matrix params at kpt_lr under Muon (model_train B5) ----------------------------

class _Fresh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.mlps = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        self.query_encoder = torch.nn.Module()
        self.query_encoder.kpt_embed = torch.nn.Embedding(5, 8)
        self.query_encoder.missing_qpos = torch.nn.Parameter(torch.zeros(8))
        self.head = torch.nn.Linear(8, 3)


_KPT_CFG = dict(learning_rate=1e-4, kpt_lr=5e-4, weight_decay=0.0)


def _lr_of(opt, p):
    for o in opt._opts if hasattr(opt, '_opts') else [opt]:
        for g in o.param_groups:
            if any(q is p for q in g['params']):
                return g['lr']
    raise AssertionError('param not routed')


@pytest.mark.parametrize('kind', ['muon', 'schedulefree'])
def test_identity_table_and_missing_tokens_train_at_kpt_lr(kind):
    m = _Fresh()
    opt = build_optimizer(m, {'query_encoder.missing_qpos'}, dict(_KPT_CFG, optimizer=kind))
    assert _lr_of(opt, m.query_encoder.kpt_embed.weight) == pytest.approx(5e-4)
    assert _lr_of(opt, m.query_encoder.missing_qpos) == pytest.approx(5e-4)
    assert _lr_of(opt, m.head.bias) == pytest.approx(1e-4)


def test_adamw_fresh_route_and_its_build_order():
    from tailcyclenet.pose.optim import ROUTES, group_lr, route_param

    m = _Fresh()
    embed = {id(m.query_encoder.kpt_embed.weight)}
    assert route_param('query_encoder.kpt_embed.weight', m.query_encoder.kpt_embed.weight,
                       set(), embed) == 'adamw_fresh'
    assert ROUTES == ('muon_dec', 'muon_fresh', 'muon_enc', 'adamw_fresh', 'adamw_base',
                      'adamw_enc')
    assert group_lr('adamw_fresh', _KPT_CFG) == pytest.approx(5e-4)
    opt = build_muon(m, {'query_encoder.missing_qpos'}, _KPT_CFG)
    assert [g['lr'] for g in opt.opt_adam.param_groups] == pytest.approx([5e-4, 1e-4])
