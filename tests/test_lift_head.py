"""The single-view lift head (report 83 §9) and the warm-start registry spelling fix."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet.format import Registry
from tailcyclenet.infer.cli import build_parser
from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.loader.sampling import pose_collate
from tailcyclenet.pose.checkpoint import _drop_untrained_lift_head
from tailcyclenet.pose.lift_head import LiftHead, lift_cameras, lift_loss
from tailcyclenet.pose.losses import PoseLoss
from tailcyclenet.pose.model import build_model
from tailcyclenet.pose.train import run_batch, to_device
from tests.helpers import SMALL, _session_3d


def test_spelled_as_maps_single_root_onto_prefixed_base():
    base = Registry(names=('a-nose', 'a-ear', 'b-nose'), datasets=(('a', (0, 1)), ('b', (2,))))
    mine = Registry(names=('nose', 'ear', 'tail'), datasets=(('a', (0, 1)), ('c', (2,))))
    assert mine.spelled_as(base) == ('a-nose', 'a-ear', 'tail')


def test_spelled_as_strips_prefix_for_unprefixed_base():
    base = Registry(names=('nose', 'ear'), datasets=(('a', (0, 1)),))
    mine = Registry(names=('a-nose', 'a-ear', 'b-x'), datasets=(('a', (0, 1)), ('b', (2,))))
    assert mine.spelled_as(base) == ('nose', 'ear', 'b-x')
    assert base.spelled_as(base) == base.names


# ---------------------------------------------------------------------------------------------
# the single-view lift head
# ---------------------------------------------------------------------------------------------
CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                   crop_jitter=0.0, prompt_dropout=0.0)
SEEING_CAM = 1     # `_session_3d`'s camera 0 cannot see the animal (see tests/pose/test_model.py)
HEAD = dict(lift_head=True, lift_head_layers=2, lift_head_width=32)


@pytest.fixture(scope='module')
def batch():
    """One static-rig 3D window (3 cameras)."""
    root = Path(tempfile.mkdtemp())
    _session_3d(root / 'st' / 'train' / 's_static')
    ds = PoseDataset(root / 'st', 'train', CFG, train=False)
    return pose_collate([ds[0]])


def _pair(b):
    """(model without a head, the same weights + a head)."""
    n = int(b.kpt_ids.max()) + 1
    torch.manual_seed(0)
    plain = build_model(SMALL, n_keypoints=n).eval()
    torch.manual_seed(0)
    lifted = build_model({**SMALL, **HEAD}, n_keypoints=n).eval()
    missing, unexpected = lifted.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith('lift_head.') for k in missing)
    return plain, lifted


def _fwd(model, b, cams):
    return model([b.views[c] for c in cams], b.kpt_ids, [b.cgroup[c] for c in cams], mode='3d',
                 kpt_prior=b.kpt_prior, prompt_time=b.prompt_t)


def test_lift_head_shapes(batch):
    head = LiftHead(7, width=32, layers=2)
    n = torch.randn(5, 7, 2) * 0.1
    n[0, 3] = float('nan')
    out = head(n, torch.arange(7)[None].expand(5, 7))
    assert out.shape == (5, 7, 3) and torch.isfinite(out).all()
    p2 = torch.rand(3, 1, 4, batch.kpt_ids.shape[1], 2) * 40
    head = LiftHead(int(batch.kpt_ids.max()) + 1, width=32, layers=2)
    xc, xw = lift_cameras(head, p2, batch.kpt_ids, batch.cgroup)
    assert xc.shape == xw.shape == (3, 1, 4, batch.kpt_ids.shape[1], 3)
    E = batch.cgroup[1]['ext']
    back = torch.einsum('ij,btnj->btni', E[:3, :3], xw[1]) + E[:3, 3]
    torch.testing.assert_close(back, xc[1], rtol=1e-4, atol=1e-3)


def test_lift_loss_is_per_camera_and_matches_the_formula(batch):
    """Perfect camera-frame output -> 0; a known per-camera error gives the patch's formula."""
    cg = batch.cgroup
    X = torch.randn(1, 4, 6, 3) * 20
    X[0, 1, 2] = float('nan')
    E = torch.stack([c['ext'][:3] for c in cg])
    G = torch.einsum('cij,btnj->cbtni', E[..., :3], X) + E[:, None, None, None, :, 3]
    term, shape, root = lift_loss(G.clone(), X, cg)
    assert float(term) < 1e-3
    P = G.clone()
    P[1] = P[1] * 1.1                     # camera 1 alone: 10% too far, same shape scaled
    term, shape, root = lift_loss(P, X, cg)
    m = torch.isfinite(G).all(-1)
    cnt = m.sum(-1, keepdim=True)
    cgt = torch.where(m[..., None], G, 0.).sum(-2) / cnt
    cpr = torch.where(m[..., None], P, 0.).sum(-2) / cnt
    e = ((P - cpr[..., None, :]) - (G - cgt[..., None, :])).norm(dim=-1)[m]
    r = (torch.log(cpr[..., 2].clamp_min(1.)) - torch.log(cgt[..., 2].clamp_min(1.))).abs()
    torch.testing.assert_close(shape, e.mean())
    torch.testing.assert_close(root, r.mean())
    torch.testing.assert_close(term, shape / 10 + 10 * root)
    assert float(r[0].max()) < 1e-6 and float(r[2].max()) < 1e-6, 'other cameras are exact'


def test_pose_loss_lift_term_skips_2d_items(batch):
    fn = PoseLoss(lift_loss_weight=0.5)
    cam = torch.zeros(3, 1, 4, 6, 3)
    assert fn._lift_term({'lift_cam': cam}, torch.zeros(1, 4, 6, 2), batch.cgroup, None,
                         'cpu') == 0.0
    assert fn._lift_term({'lift_cam': cam}, torch.zeros(1, 4, 6, 3), batch.cgroup,
                         torch.zeros(1), 'cpu') == 0.0, 'a prob_2d_only item (p2d) skips it'
    assert fn._lift_term({}, torch.zeros(1, 4, 6, 3), batch.cgroup, None, 'cpu') == 0.0
    assert PoseLoss(lift_loss_weight=0.0)._lift_term(
        {'lift_cam': cam}, torch.zeros(1, 4, 6, 3), batch.cgroup, None, 'cpu') == 0.0


def test_training_forward_lifts_every_camera_and_detaches_the_2d(batch):
    _, model = _pair(batch)
    model.train()
    out = _fwd(model, batch, [0, 1, 2])
    K = batch.kpt_ids.shape[1]
    assert out['lift_cam'].shape == (3, 1, 4, K, 3)
    fn = PoseLoss(lift_loss_weight=0.5)
    term = fn._lift_term(out, batch.coords, batch.cgroup, None, 'cpu')
    assert torch.is_tensor(term) and torch.isfinite(term)
    g = torch.autograd.grad(term, [out['2d_pred'], model.lift_head.inp.weight],
                            allow_unused=True)
    assert g[0] is None, 'the head must not train the 2D'
    assert g[1] is not None and float(g[1].abs().sum()) > 0
    assert len(fn.loss_history['lift']) == 1


def test_multiview_eval_output_is_bit_identical(batch):
    plain, lifted = _pair(batch)
    with torch.no_grad():
        a, b = _fwd(plain, batch, [0, 1, 2]), _fwd(lifted, batch, [0, 1, 2])
    assert 'lift_cam' not in b and set(a) == set(b)
    for k, v in a.items():
        if torch.is_tensor(v):
            assert torch.equal(v, b[k]) or (torch.isnan(v).any() and torch.equal(
                torch.nan_to_num(v), torch.nan_to_num(b[k]))), k


def test_single_view_deploy_uses_the_head(batch):
    plain, lifted = _pair(batch)
    with torch.no_grad():
        a = _fwd(plain, batch, [SEEING_CAM])
        b = _fwd(lifted, batch, [SEEING_CAM])
        lifted.lift_deploy = False
        c = _fwd(lifted, batch, [SEEING_CAM])
    torch.testing.assert_close(b['coords_pred'], b['lift_world'][0])
    assert not torch.equal(b['coords_pred'], a['coords_pred'])
    assert torch.equal(c['coords_pred'], a['coords_pred']), '--no-lift-head = the stock output'
    assert torch.equal(b['2d_pred'], a['2d_pred'])


def test_old_checkpoint_loads_without_a_head(batch):
    n = int(batch.kpt_ids.max()) + 1
    assert build_model(SMALL, n_keypoints=n).lift_head is None, 'code default: no head'
    plain, lifted = _pair(batch)
    _drop_untrained_lift_head(lifted, plain.state_dict(), 'old.pth')
    assert lifted.lift_head is None
    with torch.no_grad():
        a, b = _fwd(plain, batch, [SEEING_CAM]), _fwd(lifted, batch, [SEEING_CAM])
    assert torch.equal(a['coords_pred'], b['coords_pred'])
    _, kept = _pair(batch)
    _drop_untrained_lift_head(kept, kept.state_dict(), 'new.pth')
    assert kept.lift_head is not None


def test_set_only_model_trains_the_head_per_query(tmp_path):
    """PBM4: the head lifts every query of the full E2 output against its matched target (the
    loss is `set_loss` alone), gets a gradient, and gives a one-camera eval forward its 3D."""
    from tailcyclenet.pose import set_decoder as sdm
    from tests.helpers import KPTS_3D, _session_3d_multi
    root = tmp_path / 'mice'
    _session_3d_multi(root / 'train' / 's', T=8, sep=5.0)
    cfg = LoaderConfig(n_frames=4, image_size=64, min_crop_dim=16, prob_2d_only=0.0,
                       aug_prob=0.0, crop_jitter=0.0, prompt_dropout=0.0, box_prompt='film',
                       set_targets='only', hard_frac=0.5, cams_to_sample=[3, 3],
                       mode_3d_frac=1.0)
    ds = PoseDataset(root, 'train', cfg)
    torch.manual_seed(0)
    np.random.seed(0)
    item = next(it for it in (ds[(0, i)] for i in range(20)) if it is not None)
    b = pose_collate([item])
    K = len(KPTS_3D)
    torch.manual_seed(0)
    m = build_model({**SMALL, **HEAD, 'box_prompt': 'film', 'set_decoder': 'only'},
                    n_keypoints=K).train()
    loss_fn = PoseLoss()
    loss, out = run_batch(m, loss_fn, b, 'cpu')
    assert torch.isfinite(loss) and 'lift_cam' not in out, 'the query-0 slice is not lifted'
    assert np.isfinite(loss_fn.loss_history['lift'][-1])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.lift_head.parameters())
    views, cg = to_device(b, 'cpu')
    m.eval()
    with torch.no_grad():
        one = m([views[0]], b.kpt_ids, [cg[0]], mode='3d',
                set_inputs={'anchors': b.set_targets['anchors'][:, :sdm.NA, :, :1]})
        m.lift_deploy = False
        off = m([views[0]], b.kpt_ids, [cg[0]], mode='3d',
                set_inputs={'anchors': b.set_targets['anchors'][:, :sdm.NA, :, :1]})
    assert one['lift_world'].shape == (1, 1, 4, K, 3)
    assert torch.equal(one['coords_pred'], one['lift_world'][0])
    assert 'lift_world' not in off


def test_cli_lift_head_switch():
    base = ['--run', 'r', '--out', 'o', '--data', 'd']
    assert build_parser().parse_args(base).lift_head is True
    assert build_parser().parse_args(base + ['--no-lift-head']).lift_head is False
