"""The v7 3D set decoder (PBM4, `[model].set_decoder = "only"`): E2 alone on the host decoder.

Tiny random models: shapes, the no-prior deploy path, the 3D loader targets, the scratch checkpoint
boundary, and that a model without it is untouched -- never accuracy.
"""
import numpy as np
import pytest
import torch

from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.loader.sampling import pose_collate
from tailcyclenet.pose import set_decoder as sdm
from tailcyclenet.pose.losses import PoseLoss
from tailcyclenet.pose.model import build_model
from tailcyclenet.pose.train import run_batch, to_device
from tests.helpers import KPTS_3D, SMALL, _session_3d_multi

BOX = {**SMALL, 'box_prompt': 'film'}
ONLY = {**BOX, 'set_decoder': 'only'}
K = len(KPTS_3D)


@pytest.fixture(scope='module')
def batch3(tmp_path_factory):
    root = tmp_path_factory.mktemp('set3d') / 'mice'
    _session_3d_multi(root / 'train' / 's', T=8, sep=5.0)
    cfg = LoaderConfig(n_frames=4, image_size=64, min_crop_dim=16, prob_2d_only=0.0,
                       aug_prob=0.0, crop_jitter=0.0, prompt_dropout=0.0, box_prompt='film',
                       set_targets='only', hard_frac=0.5, cams_to_sample=[3, 3],
                       mode_3d_frac=1.0)
    ds = PoseDataset(root, 'train', cfg)
    torch.manual_seed(0)
    np.random.seed(0)
    item = None
    for i in range(20):
        item = ds[(0, i)]
        if item is not None:
            break
    return ds, pose_collate([item])


def _model(cfg=ONLY, seed=0):
    torch.manual_seed(seed)
    return build_model(cfg, n_keypoints=K)


def _si(b):
    st = b.set_targets
    return {'anchors': st['anchors'][:, :sdm.NA], 'anchor_prior': st['prior'][:, :sdm.NA],
            'anchor_prompt_t': st['prior_t'][:, :sdm.NA]}


def test_3d_set_targets_shapes(batch3):
    _, b = batch3
    st = b.set_targets
    C = len(b.cgroup)
    assert b.sample_info['mode'] == '3d'
    assert st['all'].shape == (1, sdm.M_ONLY, 4, K, 3)
    assert st['anchors'].shape == (1, sdm.M_ONLY, 4, C, 4)
    assert st['prior'].shape == (1, sdm.M_ONLY, K, 3)
    assert st['ok'][0, :2].all() and not st['ok'][0, 2:].any()
    # row 0 is the item's own animal, in the item's own (gauge-rotated) world frame
    assert torch.allclose(st['all'][0, 0], b.coords[0], atol=1e-4, equal_nan=True)
    a0 = st['anchors'][0, 0]
    fin = torch.isfinite(a0).all(-1)
    assert fin.any()
    w, h = a0[..., 2] - a0[..., 0], a0[..., 3] - a0[..., 1]
    assert torch.allclose(w[fin], h[fin], rtol=0.25)        # squared (then +-10% edge jitter)


def test_square_pad_is_the_crop_rules_box():
    from tailcyclenet.loader.crowd import square_pad
    np.testing.assert_allclose(square_pad(np.array([0., 0., 20., 10.]), 0.05),
                               [-1., -6., 21., 16.])


def test_set_only_model_shares_the_host_decoder():
    m = _model()
    sd = m.set_decoder
    assert sd.shared and sd.M == sdm.M_ONLY and sd.decoder is m.decoder
    keys = [k for k in m.state_dict() if k.startswith('set_decoder.')]
    assert keys and not any(k.startswith(('set_decoder.decoder.', 'set_decoder.query_encoder.'))
                            for k in keys)


def test_3d_forward_shapes_and_query_zero(batch3):
    _, b = batch3
    m = _model().eval()
    views, cg = to_device(b, 'cpu')
    C = len(cg)
    with torch.no_grad():
        out, pres, anch = sdm.set_forward(m, views, b.kpt_ids, cg, '3d', input_size=64, **{
            k.replace('anchors', 'anchors'): v for k, v in _si(b).items()})
        assert out['coords_pred'].shape == (1, 4, sdm.M_ONLY * K, 3)
        assert pres.shape == (1, 4, sdm.M_ONLY) and anch.shape == (1, sdm.M_ONLY, 4, C, 4)
        one = m(views, b.kpt_ids, cg, mode='3d', set_inputs={'anchors': _si(b)['anchors']})
    assert one['coords_pred'].shape == (1, 4, K, 3)
    # 3D always selects query 0, even with no own anchor (world coords vs pixel boxes is refused)
    q = sdm.select_query(out['coords_pred'][0].reshape(4, sdm.M_ONLY, K, 3), pres[0], (64, 64),
                         np.zeros((2, 4, 4)), 1, None)
    assert q == 0


def test_no_prior_deploy_path(batch3):
    """`--anchor none`: no item prior -> every anchored query's prior is NaN; the box prompt never
    reaches E2; no set_inputs = no anchors (the validation pass)."""
    _, b = batch3
    m = _model().eval()
    views, cg = to_device(b, 'cpu')
    anch = {'anchors': _si(b)['anchors']}
    nan_pri = {**anch, 'anchor_prior': torch.full((1, sdm.NA, K, 3), float('nan')),
               'anchor_prompt_t': torch.zeros((1, sdm.NA), dtype=torch.long)}
    with torch.no_grad():
        a = m(views, b.kpt_ids, cg, mode='3d', set_inputs=anch)
        c = m(views, b.kpt_ids, cg, mode='3d', set_inputs=nan_pri, box_prompt=b.box_prompt)
        free = m(views, b.kpt_ids, cg, mode='3d')
        pri = m(views, b.kpt_ids, cg, mode='3d', set_inputs=_si(b))
    assert torch.equal(a['coords_pred'], c['coords_pred'])
    assert free['coords_pred'].shape == a['coords_pred'].shape
    assert not torch.equal(free['coords_pred'], a['coords_pred'])     # anchors are live
    assert torch.isfinite(_si(b)['anchor_prior']).any() and not torch.equal(
        pri['coords_pred'], a['coords_pred'])                          # priors are live


def test_set_only_train_step_trains_the_encoder_path(batch3):
    _, b = batch3
    m = _model().train()
    loss, out = run_batch(m, PoseLoss(), b, 'cpu')
    assert torch.isfinite(loss) and out['coords_pred'].shape[2] == K
    loss.backward()
    assert m.set_decoder.inst.grad is not None
    assert any(p.grad is not None for n, p in m.named_parameters() if n.startswith('decoder.'))
    assert any(p.grad is not None for n, p in m.named_parameters()
               if n.startswith('set_decoder.groups.'))


# ------------------------------------------------------------------------ the checkpoint boundary
def _to_scratch_only(state):
    """This repo's set-only layout -> crowd-detr's scratch PBM4 layout (the inverse map)."""
    out = {}
    for k, v in state.items():
        for old, new in sdm._SCRATCH_RENAMES[:4]:
            if k.startswith(new):
                out[old + k[len(new):]] = v
                break
        else:
            out[k] = v
    return out


def _registry():
    from tailcyclenet.format import Registry
    return Registry(names=tuple(KPTS_3D), datasets=(('mice', tuple(range(K))),))


def test_scratch_pbm4_round_trip_and_load_run(tmp_path, capsys):
    from tailcyclenet.pose.checkpoint import load_run
    from tests.helpers import _save_run
    src = _model()
    state = src.state_dict()
    scratch = _to_scratch_only(state)
    assert 'query_encoder.kpt_dq_inst' in scratch and 'dq_pres.0.weight' in scratch
    assert sdm.is_scratch_set_only(scratch) and not sdm.is_scratch_layout(scratch)
    back = sdm.map_scratch_set_only(scratch)
    assert set(back) == set(state) and all(torch.equal(back[k], state[k]) for k in state)
    run, path = _save_run(tmp_path, _registry(), model_cfg=BOX)     # the scratch run's config
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ck['model_state'] = scratch
    ck.pop('model_state_eval', None)
    torch.save(ck, path)
    model, config, _, _ = load_run(run)
    assert config['model']['set_decoder'] == 'only' and model.set_decoder.shared
    got = model.state_dict()
    assert set(got) == set(state) and all(torch.equal(got[k], v) for k, v in state.items())
    assert 'set-only PBM4' in capsys.readouterr().out


def test_non_pbm4_set_only_is_refused():
    from tailcyclenet.pose.checkpoint import compat_state
    scratch = _to_scratch_only(_model().state_dict())
    bad = {**scratch, 'query_encoder.kpt_dq_inst': torch.zeros(8, 64)}
    with pytest.raises(SystemExit, match='SET-DECODER-ONLY'):
        compat_state(bad, {}, 'x')
    with pytest.raises(SystemExit, match='SET-DECODER-ONLY'):
        compat_state({**scratch, 'dq_apool.weight': torch.zeros(1)}, {}, 'x')
    with pytest.raises(SystemExit, match='SET-DECODER-ONLY'):                 # sa 0: no GroupPose
        compat_state({k: v for k, v in scratch.items() if 'dq_groups' not in k}, {}, 'x')


def test_box_film_3d_checkpoint_is_untouched(tmp_path, batch3, capsys):
    """A 3D box-FiLM run (no set decoder) loads strictly, builds no E2, and its forward is the
    plain model's bit for bit."""
    from tailcyclenet.pose.checkpoint import load_run
    from tests.helpers import _save_run
    _, b = batch3
    run, path = _save_run(tmp_path, _registry(), model_cfg=BOX)
    model, config, _, _ = load_run(run)
    assert model.set_decoder is None and 'set_decoder' not in config['model']
    assert 'scratch' not in capsys.readouterr().out
    ck = torch.load(path, map_location='cpu', weights_only=False)
    plain = build_model(BOX, n_keypoints=K)
    plain.load_state_dict(ck.get('model_state_eval') or ck['model_state'], strict=True)
    views, cg = to_device(b, 'cpu')
    kw = dict(kpt_prior=b.kpt_prior, prompt_time=b.prompt_t, box_prompt=b.box_prompt)
    with torch.no_grad():
        x = model.eval()(views, b.kpt_ids, cg, mode='3d', **kw)
        y = plain.eval()(views, b.kpt_ids, cg, mode='3d', **kw)
    for k in x:
        if torch.is_tensor(x[k]):
            assert torch.equal(x[k], y[k]), k


def test_set_only_routes_under_pbm4_names():
    from tailcyclenet.pose.optim import embedding_ids, route_param
    m = _model()
    named = dict(m.named_parameters())
    emb = embedding_ids(m)
    assert route_param('set_decoder.inst', named['set_decoder.inst'], set(), emb) == 'adamw_fresh'
    w = 'set_decoder.groups.0.w.in_proj_weight'
    assert route_param(w, named[w], {w}, emb) == route_param(
        'decoder.dq_groups.0.w.in_proj_weight', named[w], {'decoder.dq_groups.0.w.in_proj_weight'},
        emb)


# --------------------------------------------------------------------------------- the deploy
def test_set_only_inputs_legacy_boxes_as_is_tight_boxes_squared():
    from tailcyclenet.infer.moe import set_only_inputs
    T = 2
    own = np.array([10., 10., 30., 20.])
    near = np.array([25., 10., 45., 30.])
    far = np.array([500., 500., 520., 520.])
    wb = np.stack([np.stack([np.stack([own] * 2)] * T), np.stack([np.stack([near] * 2)] * T),
                   np.stack([np.stack([far] * 2)] * T)])                     # S,T,C,4
    boxes = [torch.tensor([0, 0, 64, 64]), torch.tensor([5, 5, 69, 69])]
    scales = [torch.tensor(1.0), torch.tensor(2.0)]
    cg = [{'size': torch.tensor([64, 64])}, {'size': torch.tensor([128, 128])}]
    leg = set_only_inputs(0, wb, [0, 1], boxes, scales, cg, 'cpu', square=True, pad=0.05)
    a = leg['anchors'][0]
    assert a.shape == (sdm.NA, T, 2, 4)
    np.testing.assert_allclose(a[0, 0, 0].numpy(), own)
    np.testing.assert_allclose(a[0, 0, 1].numpy(), (own - 5) * 2)
    np.testing.assert_allclose(a[1, 0, 0].numpy(), near)
    assert not torch.isfinite(a[2:]).any()                                   # far slot not in crop
    tight = set_only_inputs(0, wb, [0, 1], boxes, scales, cg, 'cpu', square=False, pad=0.05)
    np.testing.assert_allclose(tight['anchors'][0, 0, 0, 0].numpy(), [9., 4., 31., 26.])


def test_resolve_refuses_what_a_set_only_model_cannot_do():
    from types import SimpleNamespace
    from tailcyclenet.infer.cli import build_parser
    from tailcyclenet.infer.resolve import infer_config
    cfg = {'model': {'set_decoder': 'only', 'box_prompt': 'film'}, 'data': {}}
    base = build_parser().parse_args(['--data', 'x', '--run', 'y', '--out', 'z'])
    s3, s2 = SimpleNamespace(mode='3d'), SimpleNamespace(mode='2d')

    def args(**kw):
        a = SimpleNamespace(**vars(base))
        a.__dict__.update(kw)
        return a
    with pytest.raises(SystemExit, match='3D sessions only'):
        infer_config(args(), cfg, s2, 4, 64, 'cpu')
    with pytest.raises(SystemExit, match='--anchor carry'):
        infer_config(args(anchor='carry'), cfg, s3, 4, 64, 'cpu')
    with pytest.raises(SystemExit, match='no E1'):
        infer_config(args(moe_route='e1'), cfg, s3, 4, 64, 'cpu')
