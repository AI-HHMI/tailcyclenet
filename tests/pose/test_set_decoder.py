"""The v7 2D MoE: E2 (the anchored set decoder) beside E1 (the box-FiLM decoder) on one encoder.

Tiny random models: shapes, gradient isolation, encode counts, the routing rule and the checkpoint
boundary -- never accuracy.
"""
import numpy as np
import pytest
import torch

from tailcyclenet.infer.moe import GroupRouter, hysteresis, in_contact
from tailcyclenet.loader.config import LoaderConfig, loader_config_from
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.loader.sampling import pose_collate
from tailcyclenet.pose import set_decoder as sdm
from tailcyclenet.pose.losses import PoseLoss
from tailcyclenet.pose.model import build_model
from tailcyclenet.pose.train import run_batch, to_device
from tests.helpers import SMALL, _session_2d

BOX = {**SMALL, 'box_prompt': 'film'}
MOE = {**BOX, 'set_decoder': True}


@pytest.fixture(scope='module')
def moe_batch(tmp_path_factory):
    root = tmp_path_factory.mktemp('moe') / 'rat'
    _session_2d(root / 'train' / 's', T=8)
    cfg = LoaderConfig(n_frames=4, image_size=64, min_crop_dim=16, prob_2d_only=0.0,
                       aug_prob=0.0, crop_jitter=0.0, prompt_dropout=0.0, box_prompt='film',
                       set_targets=True, hard_frac=0.5)
    ds = PoseDataset(root, 'train', cfg)
    ds.root = root
    torch.manual_seed(0)
    return ds, pose_collate([ds[(0, 0)]])


def _model(cfg=MOE, n=4, seed=0):
    torch.manual_seed(seed)
    return build_model(cfg, n_keypoints=n)


def test_set_targets_ride_the_batch(moe_batch):
    ds, b = moe_batch
    st = b.set_targets
    T, K = 4, ds.datasets[0].sessions['train'][0].n_keypoints
    assert st['all'].shape == (1, sdm.M, T, K, 2)
    assert st['anchors'].shape == (1, sdm.M, T, 4)
    assert st['prior'].shape == (1, sdm.M, K, 2) and st['prior_t'].shape == (1, sdm.M)
    assert st['ok'][0, :2].all() and not st['ok'][0, 2:].any()     # two animals, own first
    assert torch.isfinite(st['anchors'][0, 0]).all(-1).any()


def test_set_targets_is_not_a_config_key():
    with pytest.raises(SystemExit, match='derived'):
        loader_config_from({'set_targets': True})


def test_plain_items_are_unchanged(moe_batch):
    """No set decoder -> no 15th field (the plain loader)."""
    ds, _ = moe_batch
    from dataclasses import replace
    plain = PoseDataset(ds.root, 'train', replace(ds.cfg, set_targets=False, hard_frac=0.0))
    assert len(plain[(0, 0)]) == 14


def test_e2_forward_shapes_and_selection(moe_batch):
    _, b = moe_batch
    model = _model().eval()
    views, cg = to_device(b, 'cpu')
    K = b.kpt_ids.shape[1]
    out, pres, anch = sdm.set_forward(model, views, b.kpt_ids, cg, '2d',
                                      anchors=b.set_targets['anchors'][:, :sdm.NA],
                                      anchor_prior=b.set_targets['prior'][:, :sdm.NA],
                                      anchor_prompt_t=b.set_targets['prior_t'][:, :sdm.NA],
                                      input_size=64)
    assert out['coords_pred'].shape == (1, 4, sdm.M * K, 2)
    assert pres.shape == (1, 4, sdm.M) and anch.shape == (1, sdm.M, 4, 1, 4)
    one, q = sdm.select_and_slice(out, pres, anch, K, cg)
    assert q == 0 and one['coords_pred'].shape == (1, 4, K, 2)
    # no own anchor -> the box Hungarian / centre fallback still picks one query
    q2 = sdm.select_query(out['coords_pred'][0].reshape(4, sdm.M, K, 2), pres[0], (64, 64))
    assert 0 <= q2 < sdm.M


def test_train_step_logs_both_losses_on_one_encode(moe_batch):
    _, b = moe_batch
    model = _model().train()
    n = {'enc': 0}
    enc = model.encode_scene

    def counted(*a, **k):
        n['enc'] += 1
        return enc(*a, **k)
    model.encode_scene = counted
    loss, out = run_batch(model, PoseLoss(), b, 'cpu')
    assert n['enc'] == 1
    assert torch.isfinite(loss) and set(out['loss_parts']) == {'loss_e1', 'loss_e2'}
    assert out['coords_pred'].shape[2] == b.kpt_ids.shape[1]          # E1's output is returned
    assert abs(float(loss.detach()) - sum(out['loss_parts'].values())) < 1e-3 * max(1.0, abs(float(loss.detach())))


def test_e2_loss_reaches_no_encoder_and_no_e1_weight(moe_batch):
    """MX0e: E2 decodes DETACHED scene tokens, so its loss trains only E2."""
    _, b = moe_batch
    model = _model().train()
    views, cg = to_device(b, 'cpu')
    st = b.set_targets
    out = model(views, b.kpt_ids, cg, mode='2d', kpt_prior=b.kpt_prior, prompt_time=b.prompt_t,
                box_prompt=b.box_prompt,
                set_inputs={'route': 'train', 'anchors': st['anchors'][:, :sdm.NA],
                            'anchor_prior': st['prior'][:, :sdm.NA],
                            'anchor_prompt_t': st['prior_t'][:, :sdm.NA]})
    sdm.set_loss(PoseLoss(), model, out['set'], st, 'cpu', cg).backward()
    # `grad is None` (autograd never reached it), not merely zero: a tiny random model's encoder
    # gradient is exactly 0 even when attached (see the E1 test), so zero would prove nothing
    for name, p in model.named_parameters():
        if not name.startswith('set_decoder.'):
            assert p.grad is None, f'{name} was reached by the E2 loss'
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for n, p in model.named_parameters() if n.startswith('set_decoder.decoder.'))


def test_e1_loss_does_reach_the_encoder(moe_batch):
    _, b = moe_batch
    model = _model().train()
    views, cg = to_device(b, 'cpu')
    o1 = model(views, b.kpt_ids, cg, mode='2d', kpt_prior=b.kpt_prior, prompt_time=b.prompt_t,
               box_prompt=b.box_prompt)
    PoseLoss()(model, o1, coords_true=b.coords, vis_true=None, vis_true_cams=None, cgroup=cg,
               p2d=b.p2d, device='cpu').backward()
    enc = [p for n, p in model.named_parameters() if n.startswith('scene_encoder.') and p.requires_grad]
    assert enc and all(p.grad is not None for p in enc)      # attached (the value may be 0 here)


def test_deploy_mix_takes_each_frame_from_its_expert(moe_batch):
    _, b = moe_batch
    model = _model().eval()
    views, cg = to_device(b, 'cpu')
    common = dict(kpt_prior=b.kpt_prior, prompt_time=b.prompt_t, box_prompt=b.box_prompt)
    si = {'anchors': b.set_targets['anchors'][:, :sdm.NA]}
    n = {'enc': 0}
    enc = model.encode_scene

    def counted(*a, **k):
        n['enc'] += 1
        return enc(*a, **k)
    model.encode_scene = counted
    with torch.no_grad():
        e1 = model(views, b.kpt_ids, cg, mode='2d', **common)
        e2 = model(views, b.kpt_ids, cg, mode='2d', set_inputs={**si, 'route': np.ones(4, bool)}, **common)
        n['enc'] = 0
        mask = np.array([True, False, True, False])
        mixed = model(views, b.kpt_ids, cg, mode='2d', set_inputs={**si, 'route': mask}, **common)
        assert n['enc'] == 1
        none = model(views, b.kpt_ids, cg, mode='2d', set_inputs={**si, 'route': np.zeros(4, bool)},
                     **common)
    for t in range(4):
        want = e2 if mask[t] else e1
        assert torch.allclose(mixed['coords_pred'][:, t], want['coords_pred'][:, t], atol=1e-5)
    assert torch.equal(none['coords_pred'], e1['coords_pred'])


def test_e1_path_is_the_plain_box_film_model(moe_batch):
    """The MoE model's E1 IS the host: its E1-only forward equals a plain box-FiLM model loaded
    with the same (non-E2) weights, bit for bit."""
    _, b = moe_batch
    moe = _model().eval()
    plain = _model(BOX).eval()
    state = {k: v for k, v in moe.state_dict().items() if not k.startswith('set_decoder.')}
    plain.load_state_dict(state, strict=True)
    views, cg = to_device(b, 'cpu')
    kw = dict(kpt_prior=b.kpt_prior, prompt_time=b.prompt_t, box_prompt=b.box_prompt)
    with torch.no_grad():
        a = moe(views, b.kpt_ids, cg, mode='2d', **kw)
        c = plain(views, b.kpt_ids, cg, mode='2d', **kw)
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], c[k]), k


def test_no_set_decoder_builds_no_module():
    m = _model(BOX)
    assert m.set_decoder is None and not any(k.startswith('set_decoder.') for k in m.state_dict())


# ---------------------------------------------------------------------------------- the router
def _box(cx, cy, s=10.0):
    return np.array([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], np.float64)


def test_contact_rule():
    a = _box(0, 0)
    assert in_contact(a, [_box(5, 0)]) is True                 # IoU 0.33
    assert in_contact(a, [_box(7, 0)]) is True                 # centre 7 < 0.75 x 10
    assert in_contact(a, [_box(8, 0)]) is False                # centre 8 >= 7.5, IoU 0.11
    assert in_contact(np.full(4, np.nan), [_box(0, 0)]) is None


def test_hysteresis_enters_at_the_rule_and_exits_late():
    """Enter at centre < 7.5; stay while < 9.375 (1.25x); leave only after a 4-frame dwell and
    once clear; a frame without the slot's own box holds the state; the state crosses windows."""
    xs = [20, 8, 7, 8, 9, 9, 10, 10, 10, 8, 30, 30]
    bb = np.stack([np.stack([_box(0, 0) for _ in xs]), np.stack([_box(x, 0) for x in xs])])
    bb[0, 6] = np.nan                                          # no own box at frame 6 -> hold
    h = {}
    m1 = hysteresis(h, np.arange(0, 6), bb[:, 0:6], 0)
    m2 = hysteresis(h, np.arange(6, 12), bb[:, 6:12], 0)
    got = np.r_[m1, m2]
    #        20     8      7     8     9     9    nan   10    10    8     30     30
    want = [False, False, True, True, True, True, True, False, False, False, False, False]
    assert got.tolist() == want
    assert hysteresis(h, np.arange(0, 6), bb[:, 0:6], 0).tolist() == m1.tolist()   # idempotent


def test_dwell_holds_e2_for_four_frames():
    xs = [7, 30, 30, 30, 30, 30]
    bb = np.stack([np.stack([_box(0, 0) for _ in xs]), np.stack([_box(x, 0) for x in xs])])
    assert hysteresis({}, np.arange(6), bb, 0).tolist() == [True, True, True, True, False, False]


def test_one_keypoint_always_routes_to_e1(capsys):
    r = GroupRouter('contact', 1)
    assert r.route == 'e1'
    boxes = np.stack([np.stack([_box(10, 10)] * 4), np.stack([_box(12, 10)] * 4)])[:, :, None]
    assert r.inputs(0, np.arange(4), boxes, [torch.tensor([0, 0, 64, 64])], [torch.tensor(1.0)],
                    [{'size': torch.tensor([64, 64])}], 'cpu') is None
    assert GroupRouter('contact', 4).route == 'contact'


def test_router_anchors_own_slot_first_and_carries_other_rows():
    r = GroupRouter('contact', 3)
    boxes = np.stack([np.stack([_box(20, 20)] * 4), np.stack([_box(24, 20)] * 4),
                      np.stack([_box(200, 200)] * 4)])[:, :, None]          # slot 2 outside the crop
    r.record(1, np.arange(4), np.full((4, 3, 2), 22.0, np.float32))
    si = r.inputs(0, np.arange(4), boxes, [torch.tensor([10, 10, 42, 42])], [torch.tensor(2.0)],
                  [{'size': torch.tensor([64, 64])}], 'cpu')
    assert si['route'].all()
    a = si['anchors'][0, :, :, 0]
    np.testing.assert_allclose(a[0, 0].numpy(), (_box(20, 20) - 10) * 2)
    np.testing.assert_allclose(a[1, 0].numpy(), (_box(24, 20) - 10) * 2)
    assert not torch.isfinite(a[2:]).any()                                   # slot 2 not in crop
    np.testing.assert_allclose(si['anchor_prior'][0, 1].numpy(), 24.0)       # (22 - 10) * 2
    assert int(si['anchor_prompt_t'][0, 1, 0]) == 3


# ------------------------------------------------------------------------ the checkpoint boundary
def _to_scratch(state):
    """This repo's layout -> the scratch MX0e layout (the inverse of `map_scratch_state`)."""
    out = {}
    for k, v in state.items():
        if k.startswith('set_decoder.'):
            out[sdm.route_name(k)] = v
        elif k.startswith(('decoder.', 'query_encoder.')):
            out['e1.' + k] = v
        else:
            out[k] = v
    return out


def test_scratch_layout_maps_back_exactly():
    m = _model()
    state = m.state_dict()
    scratch = _to_scratch(state)
    assert sdm.is_scratch_layout(scratch) and not sdm.is_scratch_layout(state)
    back = sdm.map_scratch_state(scratch)
    assert set(back) == set(state)
    assert all(torch.equal(back[k], state[k]) for k in state)
    with pytest.raises(SystemExit, match='MXbb'):
        sdm.map_scratch_state({**scratch, 'e2b.decoder.x': torch.zeros(1)})


def test_scratch_run_folder_loads_into_the_ported_model(tmp_path):
    """A scratch MX0e run folder (config WITHOUT set_decoder, state in e1.* / dq_* layout) loads
    strictly into the ported model through `load_run`, nothing re-saved."""
    from tailcyclenet.pose.checkpoint import load_run
    from tests.helpers import _save_run
    src = _model()
    run, path = _save_run(tmp_path, _registry(), model_cfg=BOX)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ck['model_state'] = _to_scratch(src.state_dict())
    ck.pop('model_state_eval', None)
    torch.save(ck, path)
    model, config, _, _ = load_run(run)
    assert config['model']['set_decoder'] is True and model.set_decoder is not None
    got = model.state_dict()
    assert set(got) == set(src.state_dict())
    assert all(torch.equal(got[k], v) for k, v in src.state_dict().items())


def test_plain_box_film_run_folder_loads_strictly(tmp_path, capsys):
    from tailcyclenet.pose.checkpoint import load_run
    from tests.helpers import _save_run
    run, path = _save_run(tmp_path, _registry(), model_cfg=BOX)
    model, config, _, _ = load_run(run)
    assert model.set_decoder is None and 'set_decoder' not in config['model']
    ck = torch.load(path, map_location='cpu', weights_only=False)
    state = ck.get('model_state_eval') or ck['model_state']
    fresh = build_model(BOX, n_keypoints=4)
    fresh.load_state_dict(state, strict=True)
    assert 'scratch MX0e' not in capsys.readouterr().out


def _registry():
    from tailcyclenet.format import Registry
    return Registry(names=('nose', 'left_ear', 'right_ear', 'tail_base'),
                    datasets=(('rat', (0, 1, 2, 3)),))


def test_warm_start_fills_e2_from_the_host(tmp_path):
    from tailcyclenet.pose.checkpoint import warm_start
    base = _model(BOX, seed=1)
    p = tmp_path / 'base.pth'
    torch.save({'model_state': base.state_dict()}, p)
    m = _model(seed=2)
    fresh = warm_start(m, p, verbose=False)
    for part in ('query_encoder', 'decoder'):
        for k, v in getattr(m.set_decoder, part).state_dict().items():
            assert torch.equal(v, getattr(m, part).state_dict()[k]), (part, k)
    assert 'set_decoder.inst' in fresh and 'set_decoder.pres.0.weight' in fresh
    assert not any(k.startswith('set_decoder.decoder.') for k in fresh)


def test_e2_routes_under_mx0e_names():
    from tailcyclenet.pose.optim import embedding_ids, route_param
    m = _model()
    emb = embedding_ids(m)
    named = dict(m.named_parameters())
    for name, p in named.items():
        if not name.startswith('set_decoder.decoder.'):
            continue
        host = name[len('set_decoder.'):]
        if host in named:
            assert route_param(name, p, set(), emb) == route_param(host, named[host], set(), emb), name
    assert route_param('set_decoder.inst', named['set_decoder.inst'], set(), emb) == 'adamw_fresh'


def test_set_decoder_only_scratch_checkpoint_is_refused():
    """A crowd-detr set-model checkpoint (dq_* without e1.*) must not load as a box-FiLM model."""
    from tailcyclenet.pose.checkpoint import compat_state
    state = {k: v for k, v in _to_scratch(_model().state_dict()).items() if not k.startswith('e1.')}
    with pytest.raises(SystemExit, match='SET-DECODER-ONLY'):
        compat_state(state, {}, 'x')
