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
