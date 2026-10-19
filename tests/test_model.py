"""The architecture: shape-and-identity checks that cannot be made any other way.

`model.py` and `query_encoder.py` produce numbers, not exceptions, when they are wrong (a moving
rig reprojected through the wrong axis, a chunk handed another chunk's keypoint ids). The model is
built tiny with random weights; nothing here is an accuracy test.
"""
import tempfile
import warnings
from pathlib import Path

import pytest
import torch

from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.loader.sampling import pose_collate
from tailcyclenet.pose.model import build_model
from tailcyclenet.pose.unfreeze import _norms_in_range, apply_staged_unfreeze

# Everything structural from the shipped configs, everything expensive at its floor.
SMALL = dict(
    query='prior', mode_3d='encoder', video_encoder_version='base',
    video_encoder_requires_grad=False, video_encoder_hierarchical=True,
    stride_length=4, image_size=64, query_patch_size=5, per_camera_cube_scale=True,
    max_freq=4, embedding_dim=32, use_volume_embedding=False, corr_radius=2,
    principal_point_embedding=True, intrinsic_embedding=True, metric_ray_translation=True,
    occlusion_embedding=True, latent_dim=64, n_heads=2, n_time_space_blocks=1,
    embedding_factor=2, use_camera_self_attention=True, use_temporal_self_attention=True,
    f_eff_scale=True, scene_encoder_proj=True, scene_proj_dim=64,
    scene_pos_embed_mode='ropepos', rope_base=100.0, time_embed_mode='fourier_rel',
    gridresid_offset='query',
    output_mode='gridresid', head_3d_grid_size=64, head_3d_grid_radius=1.8,
    log_3d_output=True, log_3d_eps=0.1, soft_argmax_threshold=60, grid_decode_space='warped',
)

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                   crop_jitter=0.0, prompt_dropout=0.0)

# The single-view tests need a camera that can see the animal: `_session_3d`'s camera 0 cannot,
# which leaves `cube_scale` NaN with no sibling camera to fill from.
SEEING_CAM = 1


def small(query_encoder='wide', **over):
    return {**SMALL, 'query_encoder': query_encoder, **over}


def _model(b, **over):
    return build_model(small(**over), n_keypoints=int(b.kpt_ids.max()) + 1).eval()


@pytest.fixture(scope='module')
def moving_batch():
    """One 3D window from a rig where camera 0 slides along x. See conftest._session_3d."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    root = Path(tempfile.mkdtemp())
    cf._session_3d(root / 'mv' / 'train' / 's_moving', moving=True)
    ds = PoseDataset(root / 'mv', 'train', CFG, train=False)
    return pose_collate([ds[0]])


def _spy_query_encoder(model, run):
    """Call `run(tag_setter)` with the query encoder's output recorded per tag."""
    qe = type(model.query_encoder)
    orig, seen, tag = qe.forward, {}, [None]

    def spy(self, *a, **k):
        out = orig(self, *a, **k)
        seen[tag[0]] = (self._query_ok.clone(), k['query_coords'].clone(), out.clone())
        return out

    qe.forward = spy
    try:
        run(lambda t: tag.__setitem__(0, t))
    finally:
        qe.forward = orig
    return seen


def test_moving_camera_forward(moving_batch):
    """A moving rig (gotcha 8) reaches a finite prediction, and its query projects per frame.

    Two silent bugs: a mixed static/moving rig failed to stack `ext`, and the (t n) query axis was
    projected against the batch axis, carrying one frame's projection everywhere.
    """
    from einops import rearrange
    from tracktail.model.cube import project_points_torch

    b = moving_batch
    assert any(c['ext'].ndim == 3 for c in b.cgroup), 'the fixture must carry a moving camera'
    assert len({c['ext'].ndim for c in b.cgroup}) == 1, 'every camera needs the same ext rank'

    T, N = CFG.n_frames, b.kpt_ids.shape[1]
    qc = rearrange(torch.zeros(1, T * N, 3), 'b (t n) r -> b t n r', t=T)
    p2d = rearrange(project_points_torch(b.cgroup, qc), 'cams b t n r -> cams b (t n) r')
    assert p2d.shape[1] == 1, 'the batch axis must survive; the old path replaced it with T'
    moving_cam = next(i for i, c in enumerate(b.cgroup) if c['ext'].ndim == 3)
    per_frame = p2d[moving_cam, 0].reshape(T, N, 2)[:, 0]
    assert not torch.allclose(per_frame[0], per_frame[-1]), 'per-frame ext did not reach it'

    model = _model(b)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', FutureWarning)
        with torch.no_grad():
            out = model(b.views, b.kpt_ids, b.cgroup, mode='3d',
                        kpt_prior=b.kpt_prior, prompt_time=b.prompt_t)
    assert not [w for w in caught if 'torch.backends.cuda.sdp_kernel()' in str(w.message)]
    p = out['coords_pred']
    assert p.shape == (1, CFG.n_frames, N, 3)
    assert torch.isfinite(p).all()


def test_kpt_chunk_matches_unchunked(moving_batch):
    """Chunked decoding is numerically identical, each chunk gets ITS ids, and no stash leaks.

    Without a per-chunk slice every chunk received the full-K id stash: an assert on a
    non-divisible chunk, wrong identities when it happened to divide.
    """
    b = moving_batch
    K = b.kpt_ids.shape[1]
    assert K >= 3, 'need enough keypoints for an uneven chunk'
    model = _model(b)
    seen = []
    decode = type(model)._decode_from_scene

    def spy(self, sf, vn, coords, *a, **k):
        out = decode(self, sf, vn, coords, *a, **k)
        seen.append(tuple(self.query_encoder._kpt_ids[0].tolist()))
        return out

    def run(chunk):
        seen.clear()
        with torch.no_grad():
            type(model)._decode_from_scene = spy
            try:
                out = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=b.kpt_prior,
                            prompt_time=b.prompt_t, kpt_chunk=chunk)['coords_pred']
            finally:
                type(model)._decode_from_scene = decode
        return out, list(seen)

    ids = tuple(b.kpt_ids[0].tolist())
    whole, chunks = run(None)
    assert chunks == [ids], 'unchunked must be one pass over every keypoint'
    got, chunks = run(2)                     # 2 does not divide 3: ends on a SHORT slice
    assert chunks == [ids[:2], ids[2:]], f'chunking did not engage or mis-sliced ids: {chunks}'
    torch.testing.assert_close(got, whole, equal_nan=True)
    # A leaked stash applies one forward's identities to the next, silently.
    assert model.query_encoder._kpt_ids is None and model.query_encoder._query_ok is None
    assert model._kpt_ids_all is None and model._kpt_cursor == 0


def test_share_scene_matches_encoding_every_time(moving_batch):
    """`share_scene` encodes once but must still re-run the decode (the prior moves `coords_q`)."""
    from tailcyclenet.pose.model import share_scene

    b = moving_batch
    model = _model(b)
    n_enc = [0]
    real = model.encode_scene

    def counted(*a, **k):
        n_enc[0] += 1
        return real(*a, **k)

    model.encode_scene = counted
    kw = dict(mode='3d', prompt_time=b.prompt_t)
    with torch.no_grad():
        free = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=None, **kw)['coords_pred']
        prior = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=b.kpt_prior, **kw)['coords_pred']
        assert n_enc[0] == 2, 'baseline should encode once per forward'
        n_enc[0] = 0
        with share_scene(model):
            s_free = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=None, **kw)['coords_pred']
            s_prior = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=b.kpt_prior,
                            **kw)['coords_pred']
        assert n_enc[0] == 1, f'scene encoded {n_enc[0]} times inside share_scene, want 1'
        assert model._shared_scene is None, 'the context must restore itself'

    torch.testing.assert_close(s_free, free, equal_nan=True)
    torch.testing.assert_close(s_prior, prior, equal_nan=True)
    assert not torch.allclose(free, prior, equal_nan=True), 'the two regimes must differ'


@pytest.mark.parametrize('n_cams', [1, 3])
def test_camera_batch_scene_encode_is_bit_identical(n_cams):
    """Batching cameras through the scene encoder is bit-identical to the per-camera loop."""
    torch.manual_seed(0)
    model = build_model(SMALL, n_keypoints=5).eval()
    views = [torch.randn(1, 4, 3, 64, 64) for _ in range(n_cams)]
    with torch.no_grad():
        model.set_scene_speed(precision='fp32', camera_batch=False)
        loop = model.encode_scene(views)
        model.set_scene_speed(precision='fp32', camera_batch=True)
        batched = model.encode_scene(views)
    torch.testing.assert_close(batched, loop, rtol=0, atol=0)


def test_split_scene_encoding_matches_tracktail():
    """`project_scene` after the local backbone/tail split equals tracktail's own scene output."""
    torch.manual_seed(0)
    model = build_model(SMALL, n_keypoints=5).eval()
    views = [torch.randn(1, 4, 3, 64, 64) for _ in range(3)]
    model.set_scene_speed(precision='fp32', camera_batch=True)
    with torch.no_grad():
        encoding = model.encode_scene_backbone(views)
        with model._scene_autocast(encoding.views):
            expected = model.scene_encoder(encoding.views)
        if encoding.camera_batched:
            expected = expected[0].reshape(encoding.n_cams, encoding.batch_size,
                                           *expected.shape[2:])
        actual = model.project_scene(encoding)
    torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)


def test_speed_path_runs_end_to_end(moving_batch):
    """bf16 + camera batching reaches the decoder; autocast ends at fp32 scene features."""
    b = moving_batch
    model = _model(b)
    with pytest.raises(ValueError, match='precision'):
        model.set_scene_speed(precision='not-a-dtype')
    model.set_scene_speed(precision='bf16', camera_batch=True)
    with torch.no_grad():
        assert model.encode_scene([torch.randn(1, 4, 3, 64, 64)]).dtype == torch.float32
        out = model(b.views, b.kpt_ids, b.cgroup, mode='3d',
                    kpt_prior=b.kpt_prior, prompt_time=b.prompt_t)
    assert out['coords_pred'].shape == (1, CFG.n_frames, b.kpt_ids.shape[1], 3)
    assert torch.isfinite(out['coords_pred']).all()


def test_the_prior_reaches_the_query_encoder_per_keypoint(moving_batch):
    """Withholding the prior changes what the encoder is told, and withholding ONE keypoint's
    prior moves ONLY that keypoint's fused query (a broadcast bug silenced all or none).

    Asserted at the encoder: an untrained grid head saturates, so `coords_pred` cannot tell.
    """
    b = moving_batch
    K = b.kpt_ids.shape[1]
    model = _model(b)
    dropped = b.kpt_prior.clone()
    dropped[:, 0] = float('nan')

    def run(tag):
        with torch.no_grad():
            for name, prior, pt in (('free', None, None), ('all', b.kpt_prior, b.prompt_t),
                                    ('one', dropped, b.prompt_t)):
                tag(name)
                model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior, prompt_time=pt)

    seen = _spy_query_encoder(model, run)
    (ok_free, q_free, e_free), (ok_all, q_all, e_all) = seen['free'], seen['all']
    assert not ok_free.any(), 'prior-free must mark every query unprompted'
    assert ok_all.any(), 'a finite prior must mark queries prompted'
    assert not torch.allclose(q_free, q_all), 'the prior must move the query position'
    assert not torch.allclose(e_free, e_all), 'the two regimes must reach the decoder apart'

    a, c = e_all, seen['one'][2]             # fused axis is (B, T*K), t-major
    per_kpt = [torch.allclose(a[:, k::K], c[:, k::K]) for k in range(K)]
    assert not per_kpt[0], 'withholding keypoint 0 did not change its own query'
    assert all(per_kpt[1:]), f'it also changed other keypoints: {per_kpt}'


def test_an_unprompted_keypoint_carries_no_query_time(moving_batch):
    """A `prompt_dropout` step (NaN prior, real `prompt_t`) is the deployment forward
    (`prompt_time=None`); with a real prior the query time must still matter (positive control).
    """
    b = moving_batch
    model = _model(b)
    K = b.kpt_ids.shape[1]
    dropped = torch.full_like(b.kpt_prior, float('nan'))

    def fwd(prior, pt):
        with torch.no_grad():
            return model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior,
                         prompt_time=pt)['coords_pred']

    t = lambda v: torch.full((1, K), v, dtype=torch.int32)   # noqa: E731
    torch.testing.assert_close(fwd(dropped, t(1)), fwd(None, None))
    assert not torch.allclose(fwd(b.kpt_prior, t(0)), fwd(b.kpt_prior, t(1))), \
        'query_time is inert here, so this test proves nothing'


# ----------------------------------------------------------------------------------------------
# configs this architecture does not implement
# ----------------------------------------------------------------------------------------------

def test_unsupported_config_is_rejected():
    """Each is accepted by the library and wrong here; an absent `output_mode` must not fall
    through to the library's 'direct'. Also the deleted `query_encoder = "pose"` and the two
    `image_size` keys that must agree. Construction only, no forward.
    """
    from tailcyclenet.config import check_image_size

    with pytest.raises(AssertionError, match='use_volume_embedding'):
        build_model({**SMALL, 'use_volume_embedding': True}, n_keypoints=3)
    for mode in ('direct', 'grid'):
        with pytest.raises(AssertionError, match='output_mode'):
            build_model({**SMALL, 'output_mode': mode}, n_keypoints=3)
    with pytest.raises(AssertionError, match='mode_3d'):
        build_model({**SMALL, 'mode_3d': 'tapnext'}, n_keypoints=3)
    with pytest.raises(SystemExit, match='pose'):
        build_model(small('pose'), n_keypoints=5)
    cfg = {k: v for k, v in SMALL.items() if k != 'output_mode'}
    assert build_model(cfg, n_keypoints=3).output_mode == 'gridresid'

    check_image_size({'model': {'image_size': 256}, 'data': {'image_size': 256}})
    check_image_size({'model': {'image_size': 256}})                 # absent -> nothing to check
    with pytest.raises(ValueError, match='image_size'):
        check_image_size({'model': {'image_size': 256}, 'data': {'image_size': 128}})


def test_checkpoint_resolution_prefers_latest_and_detects_hf_ids(tmp_path, monkeypatch):
    """The default is the highest training iteration, not `best` or lexical order; a two-part
    `org/name` that is not a local path is a Hugging Face id and downloads `model.pth`.
    """
    import huggingface_hub

    from tailcyclenet.checkpoints import (is_hf_repo_id, resolve_checkpoint,
                                          resolve_hf_checkpoint)

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path)
    for n in (1000, 2000):
        (tmp_path / f'checkpoint_{n:08d}.pth').touch()
    (tmp_path / 'checkpoint_best.pth').touch()
    (tmp_path / 'checkpoint_last.pth').touch()
    assert resolve_checkpoint(tmp_path).name == 'checkpoint_00002000.pth'
    assert resolve_checkpoint(tmp_path, 'checkpoint_best.pth').name == 'checkpoint_best.pth'
    only_last = tmp_path / 'only-last'
    only_last.mkdir()
    (only_last / 'checkpoint_last.pth').touch()
    assert resolve_checkpoint(only_last).name == 'checkpoint_last.pth'

    assert is_hf_repo_id('org/model') and is_hf_repo_id('user/repo-name')
    for s in ('/absolute/path', './relative/path', 'no-slash', 'too/many/slashes'):
        assert not is_hf_repo_id(s), s
    local = tmp_path / 'local' / 'checkpoint'
    local.mkdir(parents=True)
    assert not is_hf_repo_id(str(local))

    calls = []
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download',
                        lambda **kw: calls.append(kw) or '/tmp/model.pth')
    assert resolve_hf_checkpoint('org/model', revision='main') == Path('/tmp/model.pth')
    assert calls == [{'repo_id': 'org/model', 'filename': 'model.pth', 'revision': 'main'}]


def test_wide_inherits_the_pretrained_patch_cnn(tmp_path):
    """`wide`'s patch CNN is built at the PRETRAINED width and projected, so every tensor loads by
    name (built at the fusion width it would inherit 93k of 5.5M params and retrain the rest).
    """
    from tailcyclenet.pose.checkpoint import warm_start

    from tracktail.model.encoder_decoder import PatchProcessor

    model = build_model(small(), n_keypoints=5)
    embed_dim = model.query_encoder.patch_proj.in_features
    assert embed_dim != model.query_encoder.dim

    torch.manual_seed(1234)
    src_pp = PatchProcessor(in_channels=3, patch_size=model.query_encoder.patch_size,
                            embed_dim=embed_dim, conv_channels=[32, 64, 128])
    ckpt = tmp_path / 'base.pth'
    torch.save({'model_state': {f'query_encoder.patch_processor.{k}': v
                                for k, v in src_pp.state_dict().items()}}, ckpt)

    before = {k: v.clone() for k, v in model.query_encoder.patch_processor.state_dict().items()}
    fresh = warm_start(model, ckpt, verbose=False)
    assert not [n for n in fresh if 'patch_processor' in n], 'the patch CNN was left fresh'
    after, src = model.query_encoder.patch_processor.state_dict(), src_pp.state_dict()
    assert after.keys() == before.keys()
    for k in after:
        torch.testing.assert_close(after[k], src[k])
        assert not torch.allclose(after[k], before[k]), k
    for name in ('kpt_embed.weight', 'gate.0.weight', 'linear_qpos.weight', 'patch_proj.weight',
                 'missing_qpos', 'missing_patch'):
        assert f'query_encoder.{name}' in fresh, name


def test_wide_query_terms_follow_the_query_mode():
    """`qpos`/`patch` default to `query`; the two useless combinations stay unbuildable (a prior
    with no route into the encoder; constant terms under `query = "none"`).
    """
    prior = build_model(small(query='prior'), n_keypoints=5).query_encoder
    free = build_model(small(query='none'), n_keypoints=5).query_encoder
    assert prior.term_names() == ['kpt', 'query_time', 'target_time', 'gap', 'qpos', 'patch',
                                  'pp', 'intrinsic']
    assert free.term_names() == ['kpt', 'query_time', 'target_time', 'gap', 'pp', 'intrinsic']
    assert not hasattr(free, 'linear_qpos') and not hasattr(free, 'patch_processor')
    assert free.n_fusion_terms == 6 and prior.n_fusion_terms == 8

    j4 = build_model(small(query='prior', query_patch_embedding=False),
                     n_keypoints=5).query_encoder
    assert j4.term_names() == ['kpt', 'query_time', 'target_time', 'gap', 'qpos', 'pp', 'intrinsic']
    assert j4.n_fusion_terms == 7 and not hasattr(j4, 'patch_processor')

    with pytest.raises(AssertionError, match='no-op'):
        build_model(small(query='prior', query_pos_embedding=False,
                          query_patch_embedding=False), n_keypoints=5)
    with pytest.raises(AssertionError, match='dead gate inputs'):
        build_model(small(query='none', query_pos_embedding=True), n_keypoints=5)


def test_a_grown_registry_keeps_its_trained_keypoint_rows(tmp_path):
    """An appended registry copies the base's trained `kpt_embed` rows (a whole-tensor shape drop
    reset them to noise); a base registry whose length disagrees with the table is refused.
    """
    from tailcyclenet.pose.checkpoint import warm_start

    base = build_model(small(), n_keypoints=3)
    ckpt = tmp_path / 'base.pth'
    torch.save({'model_state': base.state_dict()}, ckpt)

    model = build_model(small(), n_keypoints=5)
    virgin = model.query_encoder.kpt_embed.weight.detach().clone()
    src = base.query_encoder.kpt_embed.weight.detach()
    assert not torch.allclose(virgin[:3], src), 'the fresh table must differ'

    fresh = warm_start(model, ckpt, verbose=False, base_names=('a', 'b', 'c'))
    got = model.query_encoder.kpt_embed.weight.detach()
    torch.testing.assert_close(got[:3], src)
    torch.testing.assert_close(got[3:], virgin[3:])
    assert not any('kpt_embed' in n for n in fresh)

    model2 = build_model(small(), n_keypoints=5)
    virgin2 = model2.query_encoder.kpt_embed.weight.detach().clone()
    warm_start(model2, ckpt, verbose=False, base_names=('a', 'b'))
    torch.testing.assert_close(model2.query_encoder.kpt_embed.weight.detach(), virgin2)


# ----------------------------------------------------------------------------------------------
# the 3D output: gridresid_offset and the anchors
# ----------------------------------------------------------------------------------------------

def test_gridresid_offset_switches_the_anchor(moving_batch):
    """Query-free, "query" returns the triangulation and NaNs `anchor_local` (never pops `grid`:
    the depth CE and `f_eff` gate on it); "triangulated" adds a residual. An absent key RAISES
    (gotcha 11: both values load the same tensors).
    """
    b = moving_batch
    n_kpt = int(b.kpt_ids.max()) + 1
    outs = {}
    for off in ('query', 'triangulated'):
        m = build_model(small(query='none', gridresid_offset=off), n_keypoints=n_kpt).eval()
        with torch.no_grad():
            outs[off] = m(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=None,
                          prompt_time=None)

    q, t = outs['query'], outs['triangulated']
    torch.testing.assert_close(q['coords_pred'], q['3d_pred_triangulate'], equal_nan=True)
    assert not torch.allclose(t['coords_pred'], t['3d_pred_triangulate'], equal_nan=True)
    assert torch.isfinite(t['grid']['anchor_local']).all()
    assert 'grid' in q and not torch.isfinite(q['grid']['anchor_local']).any()
    for k in ('logits_depth', 'f_eff', 'cube_scale'):
        assert torch.isfinite(q['grid'][k]).all(), f'{k} must survive the query-free path'

    with pytest.raises(AssertionError, match='gridresid_offset'):
        build_model(small(gridresid_offset='nonsense'), n_keypoints=n_kpt)
    cfg = small()
    cfg.pop('gridresid_offset')
    with pytest.raises(KeyError, match='gridresid_offset'):
        build_model(cfg, n_keypoints=n_kpt)


def test_query_free_keeps_the_depth_ce_and_drops_only_the_3d_ce(moving_batch):
    """NaN-ing `anchor_local` zeroes only the query-anchored 3D CE; the depth CE (weight 1.5)
    fires prompted or not."""
    from tracktail.model.losses import TotalLoss

    b = moving_batch
    model = _model(b)
    for prompted in (True, False):
        out = model(b.views, b.kpt_ids, b.cgroup, mode='3d',
                    kpt_prior=b.kpt_prior.clone() if prompted else None,
                    prompt_time=b.prompt_t if prompted else None)
        loss_fn = TotalLoss(coords_softmax_3d_weight=0.4, depth_softmax_weight=1.5,
                            per_camera_cube_scale=True)
        loss_fn(model, out, coords_true=b.coords, vis_true=None, vis_true_cams=None,
                cgroup=b.cgroup, p2d=None, device='cpu')
        assert loss_fn.loss_history['depth_softmax_loss'][-1] > 0
        grid_ce = loss_fn.loss_history['3d_softmax_loss'][-1]
        assert (grid_ce > 0) if prompted else (grid_ce == 0), (prompted, grid_ce)


def test_prior_points_are_query_anchored_and_others_triangulated(moving_batch):
    """Per-KEYPOINT selection: a prompted keypoint uses the residual, an unprompted one does not."""
    b = moving_batch
    model = _model(b)
    prior = b.kpt_prior.clone()
    assert torch.isfinite(prior).all(), 'fixture must start fully prompted'
    prior[:, 1:] = float('nan')
    with torch.no_grad():
        out = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior,
                    prompt_time=b.prompt_t)
    tri, pred = out['3d_pred_triangulate'], out['coords_pred']
    assert not torch.allclose(pred[:, :, 0], tri[:, :, 0])
    torch.testing.assert_close(pred[:, :, 1:], tri[:, :, 1:], equal_nan=True)
    assert 'grid' in out


def test_direct_head_gets_no_gradient_at_unprompted_points(moving_batch):
    """The loss gate (multiview): the DETACHED triangulation makes the direct term constant at
    unprompted points, so it contributes zero gradient there. Checked over every parameter.
    """
    b = moving_batch
    model = _model(b)
    prior = b.kpt_prior.clone()
    prior[:, 1:] = float('nan')
    out = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior,
                prompt_time=b.prompt_t)
    direct = out['3d_pred_cams_direct']
    params = [p for p in model.parameters() if p.requires_grad]

    def n_reached(k):
        gs = torch.autograd.grad(direct[:, :, :, k].sum(), params,
                                 retain_graph=True, allow_unused=True)
        return sum(g is not None and bool(g.abs().sum() > 0) for g in gs)

    assert n_reached(0) > 0, 'the prompted keypoint must still reach the direct head'
    for k in range(1, b.kpt_ids.shape[1]):
        assert n_reached(k) == 0, f'unprompted keypoint {k} still reaches {n_reached(k)} params'


def test_single_view_predicts_the_mean_ray_point(moving_batch):
    """3D single-view anchors on the back-projected rays instead of dropping the step, and the
    anchor is that camera's ray point -- not the library's `3d_pred_rays` weighted SUM.
    """
    b = moving_batch
    model = _model(b, query='none')
    with torch.no_grad():
        out = model([b.views[SEEING_CAM]], b.kpt_ids, [b.cgroup[SEEING_CAM]], mode='3d',
                    kpt_prior=None, prompt_time=None)
    assert out.get('3d_pred_triangulate') is None, 'one camera cannot triangulate'
    assert 'loss_kpt_mask' not in out
    assert torch.isfinite(out['coords_pred']).all()
    assert 'grid' in out, 'the depth CE must survive the single-view path'
    torch.testing.assert_close(out['coords_pred'], out['3d_pred_cams_rays'][0])


def test_uncalibrated_singleview_uses_finite_unit_anchor(moving_batch):
    """A nominal one-camera inference has no metric scale, but must not emit NaN XYZ."""
    from tailcyclenet.pose.anchor import scene_center

    b = moving_batch
    cam = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in
           b.cgroup[SEEING_CAM].items()}
    w, h = map(int, cam['size'].tolist())
    f = float(max(w, h))
    cam['mat'] = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float32)
    cam['dist'] = torch.zeros_like(cam['dist'])
    cam['offset'] = torch.zeros_like(cam['offset'])
    cam['ext'] = torch.eye(4, dtype=torch.float32)
    cam['center'] = torch.zeros(3, dtype=torch.float32)
    cam['calibrated'] = False
    assert torch.linalg.norm(scene_center([cam])) < 1e-6

    model = _model(b, query='none', metric_ray_translation=True)
    with pytest.warns(RuntimeWarning, match='NOT metric'):
        with torch.no_grad():
            out = model([b.views[SEEING_CAM]], b.kpt_ids, [cam], mode='3d',
                        kpt_prior=None, prompt_time=None)
    scale = out['grid']['cube_scale']
    assert torch.isfinite(scale).all() and (scale > 0).all()
    for k in ('depth_pred', '3d_pred_cams_rays', 'coords_pred'):
        assert torch.isfinite(out[k]).all(), k


def test_single_view_substitutes_the_detached_rays(moving_batch):
    """The single-view loss gate, by value: the substituted anchor equals the rays point and
    carries no gradient (else the direct head reopens at every unprompted keypoint).
    """
    b = moving_batch
    model = _model(b)
    prior = b.kpt_prior.clone()
    prior[:, 1:] = float('nan')
    out = model([b.views[SEEING_CAM]], b.kpt_ids, [b.cgroup[SEEING_CAM]], mode='3d',
                kpt_prior=prior, prompt_time=b.prompt_t)
    rays = out['3d_pred_cams_rays']
    unprompted = out['3d_pred_cams_direct'][0, :, :, 1:]
    torch.testing.assert_close(unprompted, rays[0][:, :, 1:].detach())

    def grad_to_rays(t):
        g, = torch.autograd.grad(t.sum(), rays, retain_graph=True, allow_unused=True)
        return 0.0 if g is None else float(g.abs().sum())

    assert grad_to_rays(out['3d_pred_rays']) > 0, 'positive control: rays are in the graph'
    assert grad_to_rays(unprompted) == 0.0, 'the substituted anchor must be detached'


def test_a_degenerate_triangulation_needs_a_detach_not_a_where():
    """Library canary: `linalg.solve`'s backward NaNs every parameter through `torch.where` on a
    degenerate point; only detaching saves the step (the guard in model.py)."""
    def grads(detach):
        p = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x = torch.linalg.solve(torch.stack([p, p * float('nan')]), torch.ones(2, 2))
        assert not torch.isfinite(x).all(), 'the probe must actually be degenerate'
        if detach:
            x = x.detach()
        y = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        (y.sum() + p.sum()).backward()
        return p.grad

    assert torch.isnan(grads(detach=False)).all(), 'if `where` masks it, the guard is dead code'
    assert torch.isfinite(grads(detach=True)).all()


# ---------------------------------------------------------------------------------------------
# the staged encoder unfreeze (ViT depth 12, hierarchical taps [2,5,8,11])
# ---------------------------------------------------------------------------------------------

def _staged_model(n_last=4, at=3):
    return build_model(small(video_encoder_requires_grad=at,
                             video_encoder_finetune_last_n_layers=n_last), n_keypoints=3)


def test_staged_unfreeze_selects_the_last_n_blocks_and_their_norms():
    """Fires once at `at`, unfreezes blocks 8..11 for N=4, plus the `norms_block` taps in that
    range (upstream's `encoder.norm` does not exist on VJEPA 2.1)."""
    m = _staged_model(n_last=4, at=3)
    enc = m.scene_encoder.encoder
    assert len(enc.blocks) == 12 and enc.hierarchical_layers == [2, 5, 8, 11], 'fixture moved'
    for it in (0, 2):
        assert m.unfreeze_video_encoder(it) is False, f'fired early at {it}'
        assert not any(p.requires_grad for p in enc.parameters())
    assert m.unfreeze_video_encoder(3) is True
    for i, blk in enumerate(enc.blocks):
        assert all(p.requires_grad for p in blk.parameters()) == (i >= 8), f'block {i}'
    assert m.unfreeze_video_encoder(4) is False, 'not idempotent'

    assert _norms_in_range(enc, 4) == [2, 3]
    assert _norms_in_range(enc, 2) == [3]
    assert _norms_in_range(enc, 12) == [0, 1, 2, 3]
    m = _staged_model(n_last=4, at=0)
    enc = m.scene_encoder.encoder
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.0)
    info = apply_staged_unfreeze(m, opt, {'learning_rate': 1e-4, 'encoder_lr_scale': 0.1}, 0)
    assert info['norms'] == [2, 3], info
    for i, nrm in enumerate(enc.norms_block):
        assert all(p.requires_grad for p in nrm.parameters()) == (i >= 2), f'norms_block[{i}]'


def test_gradients_reach_the_unfrozen_blocks_only(moving_batch):
    """Finite grads on the trainable blocks, None on the frozen ones, through the checkpointed
    backward of a partially-frozen encoder."""
    b = moving_batch
    m = _staged_model(n_last=4, at=0)
    assert m.unfreeze_video_encoder(0) is True
    m(b.views, b.kpt_ids, b.cgroup, mode='3d')['coords_pred'].nansum().backward()
    for i, blk in enumerate(m.scene_encoder.encoder.blocks):
        grads = [p.grad for p in blk.parameters()]
        if i >= 8:
            assert all(g is not None and torch.isfinite(g).all() for g in grads), f'block {i}'
        else:
            assert all(g is None for g in grads), f'frozen block {i} received a gradient'


def test_2d_forward_shapes_match_pose_loss_expectations(tiny_root):
    """A real R == 2 forward's `vis_pred_2d` pairs with the collate's `vis_2d` in `PoseLoss` and
    runs a finite backward into the query encoder."""
    from tailcyclenet.pose.losses import PoseLoss

    ds = PoseDataset(tiny_root / 'ratlike', 'train', CFG, train=False)
    b = next((it for it in (pose_collate([ds[i]]) for i in range(len(ds)))
              if it.vis_2d is not None), None)
    assert b is not None, 'ratlike must yield at least one item with a real vis_2d target'

    model = build_model(small(), n_keypoints=int(b.kpt_ids.max()) + 1)
    out = model(b.views, b.kpt_ids, b.cgroup, mode='2d', kpt_prior=b.kpt_prior,
                prompt_time=b.prompt_t)
    assert out['vis_pred_2d'].shape == (1, 1, b.vis_2d.shape[1], b.vis_2d.shape[2])

    loss_fn = PoseLoss(vis_loss_2d_weight=5.0)
    total = loss_fn.forward(model, out, b.coords, None, None, vis_2d_true=b.vis_2d, p2d=b.p2d,
                            cgroup=b.cgroup, device='cpu')
    assert torch.isfinite(total)
    assert torch.isfinite(torch.tensor(loss_fn.loss_history['vis_loss_2d'][-1]))
    total.backward()
    grads = [p.grad for p in model.query_encoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).any() for g in grads)
