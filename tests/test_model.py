"""The architecture: the two things that cannot be checked any other way.

`model.py` and `query_encoder.py` produce numbers, not exceptions, when they are wrong. A moving
rig silently reprojected the query through the wrong axis, and a chunked forward would silently
hand one chunk's queries another chunk's keypoint identities. Neither shows up in a loss curve,
so both are pinned here.

The model is built at a deliberately small size -- ViT-base is the smallest scene encoder
posetail offers, and everything downstream of it is shrunk to the minimum that still exercises
every branch. This is a SHAPE-AND-IDENTITY test, not an accuracy test; the weights are random.
"""
import tempfile
from pathlib import Path

import pytest
import torch

from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate
from tailcyclenet.model import build_model

# Everything structural from configs/w9.toml, everything expensive at its floor.
SMALL = dict(
    query='prior', mode_3d='encoder', video_encoder_version='base',
    video_encoder_requires_grad=False, video_encoder_hierarchical=True,
    stride_length=4, image_size=64, query_patch_size=5, per_camera_cube_scale=True,
    max_freq=4, embedding_dim=32, use_volume_embedding=False, corr_radius=2,
    principal_point_embedding=True, intrinsic_embedding=True, metric_ray_translation=True,
    occlusion_embedding=True, latent_dim=64, n_heads=2, n_time_space_blocks=1,
    embedding_factor=2, use_camera_self_attention=True, use_temporal_self_attention=True,
    f_eff_scale=True, scene_encoder_proj=True, cross_attn_dim=64, scene_proj_dim=64,
    scene_pos_embed_mode='ropepos', rope_base=100.0, time_embed_mode='fourier_rel',
    output_mode='gridresid', head_3d_grid_size=64, head_3d_grid_radius=1.8,
    log_3d_output=True, log_3d_eps=0.1, soft_argmax_threshold=60, grid_decode_space='warped',
)

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                   crop_jitter=0.0, prompt_dropout=0.0)


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


def test_moving_camera_forward(moving_batch):
    """A moving rig must reach a finite prediction. It could not before.

    Two failures were in the way, and each was silent in a different way:

    1. `_decode_from_scene` stacks `cam['ext']` across cameras (tracker_encoder.py:623), so a
       MIXED rig -- one moving camera, two static -- was a stack error. `Session.cgroup` now
       gives every camera the per-frame form; `labels()` back-fills a static camera's rows with
       its own constant pose, so the geometry is unchanged.
    2. `PoseQueryEncoder` projected the flattened (t n) query axis directly. `project_cam` aligns
       a (T,4,4) extrinsic against axis -3, which is the BATCH axis there -- so the projection
       came back with the batch axis silently replaced by T, and the visibility term raised
       inside einops. Both branches exist in the stock QueryEncoder and were lost in the port.
    """
    b = moving_batch
    assert any(c['ext'].ndim == 3 for c in b.cgroup), 'the fixture must carry a moving camera'
    assert len({c['ext'].ndim for c in b.cgroup}) == 1, 'every camera needs the same ext rank'

    model = build_model(SMALL, n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    with torch.no_grad():
        out = model(b.views, b.kpt_ids, b.cgroup, mode='3d',
                    kpt_prior=b.kpt_prior, prompt_time=b.prompt_t)
    p = out['coords_pred']
    assert p.shape == (1, CFG.n_frames, b.kpt_ids.shape[1], 3)
    assert torch.isfinite(p).all()


def test_kpt_chunk_matches_unchunked(moving_batch):
    """Chunked decoding must be numerically identical, not merely well-shaped.

    `_forward_window` slices the query set and calls `_decode_from_scene` once per chunk. The
    keypoint ids and the query-validity mask live on a stash the library knows nothing about, so
    without a per-chunk slice every chunk would receive the FULL-K stash: `_tile_to_query_axis`
    asserts on a non-divisible chunk, and hands back the wrong identities whenever it happens to
    divide. Neither failure is visible in a prediction that still has the right shape -- which is
    why this compares values.
    """
    b = moving_batch
    K = b.kpt_ids.shape[1]
    assert K >= 3, 'need enough keypoints for an uneven chunk'
    model = build_model(SMALL, n_keypoints=int(b.kpt_ids.max()) + 1).eval()

    # Record what each chunk was actually handed. Without this the test would still pass if the
    # library silently declined to chunk at all -- comparing a single pass against itself.
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

    # 2 does not divide 3: the chunk loop ends on a SHORT slice, which is where an offset that is
    # assumed rather than tracked goes wrong.
    got, chunks = run(2)
    assert chunks == [ids[:2], ids[2:]], f'chunking did not engage or mis-sliced ids: {chunks}'
    torch.testing.assert_close(got, whole, equal_nan=True)

    got, chunks = run(1)
    assert chunks == [(i,) for i in ids], f'chunking did not engage or mis-sliced ids: {chunks}'
    torch.testing.assert_close(got, whole, equal_nan=True)


def test_kpt_chunk_leaves_no_stash_behind(moving_batch):
    """A leaked stash applies one forward's identities to the next, silently."""
    b = moving_batch
    model = build_model(SMALL, n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    with torch.no_grad():
        model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=b.kpt_prior,
              prompt_time=b.prompt_t, kpt_chunk=2)
    assert model.query_encoder._kpt_ids is None and model.query_encoder._query_ok is None
    assert model._kpt_ids_all is None and model._kpt_cursor == 0


def test_the_prior_reaches_the_query_encoder(moving_batch):
    """Withholding the prior must actually change what the encoder is told.

    This is what the periodic val eval depends on: `val/*` runs prior-free because the loader's
    `kpt_prior` is ground truth (evaluation rule 7), and that is only a meaningful gate if the
    prior was reaching the model in the first place. Asserted at the ENCODER, not at
    `coords_pred`: an untrained model's grid head saturates to the grid centre for every query,
    so comparing predictions would pass whether or not the prior was plumbed through.
    """
    b = moving_batch
    model = build_model(SMALL, n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    seen = {}
    qe = type(model.query_encoder)
    orig = qe.forward

    def spy(self, *a, **k):
        out = orig(self, *a, **k)
        seen[spy.tag] = (self._query_ok.clone(), k['query_coords'].clone(), out.clone())
        return out

    qe.forward = spy
    try:
        with torch.no_grad():
            spy.tag = 'free'
            model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=None, prompt_time=None)
            spy.tag = 'prompted'
            model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=b.kpt_prior,
                  prompt_time=b.prompt_t)
    finally:
        qe.forward = orig

    ok_free, q_free, e_free = seen['free']
    ok_prompt, q_prompt, e_prompt = seen['prompted']
    assert not ok_free.any(), 'prior-free must mark every query unprompted'
    assert ok_prompt.any(), 'a finite prior must mark queries prompted'
    assert not torch.allclose(q_free, q_prompt), 'the prior must move the query position'
    assert not torch.allclose(e_free, e_prompt), 'the two regimes must reach the decoder apart'


# ----------------------------------------------------------------------------------------------
# configs this architecture does not implement
# ----------------------------------------------------------------------------------------------

def test_unsupported_config_is_rejected():
    """Every one of these is accepted by the library and wrong here. Construction only, no forward.

    `output_mode` is the dangerous one: the library DEFAULTS to 'direct', so a config that merely
    omits the key builds a model whose predictions `_reanchor_per_frame` misdescribes -- and
    nothing raises.
    """
    with pytest.raises(AssertionError, match='use_volume_embedding'):
        build_model({**SMALL, 'use_volume_embedding': True}, n_keypoints=3)

    for mode in ('direct', 'residual', 'grid', 'resdirect', 'gridnorm'):
        with pytest.raises(AssertionError, match='output_mode'):
            build_model({**SMALL, 'output_mode': mode}, n_keypoints=3)

    with pytest.raises(AssertionError, match='mode_3d'):
        build_model({**SMALL, 'mode_3d': 'tapnext'}, n_keypoints=3)

    # Omitting the key must NOT fall through to the library's 'direct'.
    cfg = {k: v for k, v in SMALL.items() if k != 'output_mode'}
    assert build_model(cfg, n_keypoints=3).output_mode == 'gridresid'


def test_mismatched_image_size_is_rejected():
    """Two keys named image_size that must agree, and nothing else notices when they do not."""
    from tailcyclenet.checkpoints import check_image_size

    check_image_size({'model': {'image_size': 256}, 'data': {'image_size': 256}})
    check_image_size({'model': {'image_size': 256}})                 # absent -> nothing to check
    with pytest.raises(ValueError, match='image_size'):
        check_image_size({'model': {'image_size': 256}, 'data': {'image_size': 128}})


def test_moving_camera_query_projects_per_frame(moving_batch):
    """The query anchor must project to a DIFFERENT pixel each frame on a moving camera.

    This is the part a shape check cannot see: the pre-fix path produced a well-formed tensor
    that carried one frame's projection everywhere. Guarding the shape alone would pass.
    """
    from einops import rearrange
    from posetail.posetail.cube import project_points_torch

    b = moving_batch
    T, N = CFG.n_frames, b.kpt_ids.shape[1]
    q = torch.zeros(1, T * N, 3)                       # one shared world point, t-major (t n)
    qc = rearrange(q, 'b (t n) r -> b t n r', t=T)
    p2d = rearrange(project_points_torch(b.cgroup, qc), 'cams b t n r -> cams b (t n) r')

    assert p2d.shape[1] == 1, 'the batch axis must survive; the old path replaced it with T'
    moving_cam = next(i for i, c in enumerate(b.cgroup) if c['ext'].ndim == 3)
    per_frame = p2d[moving_cam, 0].reshape(T, N, 2)[:, 0]
    assert not torch.allclose(per_frame[0], per_frame[-1]), \
        'per-frame extrinsics did not reach the projection'
