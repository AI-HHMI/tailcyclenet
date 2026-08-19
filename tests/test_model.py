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
from tailcyclenet.unfreeze import _norms_in_range, apply_staged_unfreeze

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
    gridresid_offset='query',
    output_mode='gridresid', head_3d_grid_size=64, head_3d_grid_radius=1.8,
    log_3d_output=True, log_3d_eps=0.1, soft_argmax_threshold=60, grid_decode_space='warped',
)

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                   crop_jitter=0.0, prompt_dropout=0.0)

# Both query encoders, everywhere it is structural. `wide` is a hand-written forward rather than a
# `QueryEncoder` subclass, so the moving-rig reshape and the per-chunk id slice are reimplemented
# there and would break silently -- which is the whole reason this file exists.
ENCODERS = ('pose', 'wide')

# The 3D single-view tests slice ONE camera out of the fixture rig, and it has to be one that can
# actually see the animal. `conftest._session_3d`'s camera 0 cannot: none of its three synthetic
# keypoints project inside its 64x48 crop. `get_camera_scale` then leaves `sensitivity` NaN and
# `fill_nan_with_batch_median` (cube.py:466) has no sibling camera to fill from, so `cube_scale`
# comes back NaN and the whole forward follows. That is a real -- if narrow -- edge case (the sole
# camera loses the animal), not the path these tests are about.
SEEING_CAM = 1


def small(query_encoder='pose', **over):
    return {**SMALL, 'query_encoder': query_encoder, **over}


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


@pytest.mark.parametrize('enc', ENCODERS)
def test_moving_camera_forward(moving_batch, enc):
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

    model = build_model(small(enc), n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    with torch.no_grad():
        out = model(b.views, b.kpt_ids, b.cgroup, mode='3d',
                    kpt_prior=b.kpt_prior, prompt_time=b.prompt_t)
    p = out['coords_pred']
    assert p.shape == (1, CFG.n_frames, b.kpt_ids.shape[1], 3)
    assert torch.isfinite(p).all()


@pytest.mark.parametrize('enc', ENCODERS)
def test_kpt_chunk_matches_unchunked(moving_batch, enc):
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
    model = build_model(small(enc), n_keypoints=int(b.kpt_ids.max()) + 1).eval()

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


def test_share_scene_matches_encoding_every_time(moving_batch):
    """Sharing the scene encode must change nothing about the output.

    The val loop runs two forwards per window over identical pixels -- prior-free, then re-queried
    at the model's own frame-0 prediction -- and the encoder is frozen, so `share_scene` encodes
    once. It is the whole basis of the optimisation that the second forward's DECODE still runs:
    `cube_scale` / `scene_center` / `scene_radius` come from `coords_q`, which the prior changes.
    A version that wrongly reused the decode too would still produce finite, well-shaped output.

    Also asserts the encode really is called once -- otherwise this compares a pass against itself.
    """
    from tailcyclenet.model import share_scene

    b = moving_batch
    model = build_model(SMALL, n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    n_enc = [0]
    real = model.scene_encoder.forward

    def counted(*a, **k):
        n_enc[0] += 1
        return real(*a, **k)

    model.scene_encoder.forward = counted
    kw = dict(mode='3d', prompt_time=b.prompt_t)
    with torch.no_grad():
        free = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=None, **kw)['coords_pred']
        prior = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=b.kpt_prior, **kw)['coords_pred']
        assert n_enc[0] == 2, 'baseline should encode once per forward'

        n_enc[0] = 0
        with share_scene(model):
            s_free = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=None, **kw)['coords_pred']
            s_prior = model(b.views, b.kpt_ids, b.cgroup, kpt_prior=b.kpt_prior, **kw)['coords_pred']
        assert n_enc[0] == 1, f'scene encoded {n_enc[0]} times inside share_scene, want 1'
        # and the context restores itself, so training does not silently keep a stale encode
        assert model._shared_scene is None
        model(b.views, b.kpt_ids, b.cgroup, kpt_prior=None, **kw)
        assert n_enc[0] == 2

    torch.testing.assert_close(s_free, free, equal_nan=True)
    torch.testing.assert_close(s_prior, prior, equal_nan=True)
    # the two regimes must genuinely differ, or "unchanged" is a vacuous claim
    assert not torch.allclose(free, prior, equal_nan=True)


def test_kpt_chunk_leaves_no_stash_behind(moving_batch):
    """A leaked stash applies one forward's identities to the next, silently."""
    b = moving_batch
    model = build_model(SMALL, n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    with torch.no_grad():
        model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=b.kpt_prior,
              prompt_time=b.prompt_t, kpt_chunk=2)
    assert model.query_encoder._kpt_ids is None and model.query_encoder._query_ok is None
    assert model._kpt_ids_all is None and model._kpt_cursor == 0


@pytest.mark.parametrize('enc', ENCODERS)
def test_the_prior_reaches_the_query_encoder(moving_batch, enc):
    """Withholding the prior must actually change what the encoder is told.

    This is what the periodic val eval depends on: `val/*` runs prior-free because the loader's
    `kpt_prior` is ground truth (evaluation rule 7), and that is only a meaningful gate if the
    prior was reaching the model in the first place. Asserted at the ENCODER, not at
    `coords_pred`: an untrained model's grid head saturates to the grid centre for every query,
    so comparing predictions would pass whether or not the prior was plumbed through.
    """
    b = moving_batch
    model = build_model(small(enc), n_keypoints=int(b.kpt_ids.max()) + 1).eval()
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


def test_resolve_checkpoint_prefers_last(tmp_path):
    """`last` is the default by name, not by sort order, and old numbered folders still resolve."""
    from tailcyclenet.checkpoints import resolve_checkpoint

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path)

    # An external base-checkpoint folder: numbered names, newest by name.
    for n in (1000, 2000):
        (tmp_path / f'checkpoint_{n:08d}.pth').touch()
    assert resolve_checkpoint(tmp_path).name == 'checkpoint_00002000.pth'

    # A run folder: `last` is the default, and no sort order gets a say in it.
    (tmp_path / 'checkpoint_best.pth').touch()
    (tmp_path / 'checkpoint_last.pth').touch()
    assert resolve_checkpoint(tmp_path).name == 'checkpoint_last.pth'
    assert resolve_checkpoint(tmp_path, 'checkpoint_best.pth').name == 'checkpoint_best.pth'


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


@pytest.mark.parametrize('enc', ENCODERS)
def test_missing_tokens_fire_per_keypoint(moving_batch, enc):
    """Withholding ONE keypoint's prior must move ONLY that keypoint's fused query.

    This is the behavioural form of posetail-pose's defect 0.0, which was live for its whole
    W4/W5 series: the validity mask is one flag per KEYPOINT, `(B, N)`, but the axis the
    position-derived terms live on is `(B, T*N)`. Its shape-mismatch branch broadcast keypoint 0's
    validity over every keypoint, so a single withheld prior either silenced all of them or none.

    A string grep cannot check this -- the comment explaining the bug quotes the code that caused
    it -- so the check has to be a differential forward. `wide` reimplements the forward by hand,
    which is exactly how such a bug comes back.
    """
    b = moving_batch
    K = b.kpt_ids.shape[1]
    model = build_model(small(enc), n_keypoints=int(b.kpt_ids.max()) + 1).eval()

    qe = type(model.query_encoder)
    orig, seen = qe.forward, {}

    def spy(self, *a, **k):
        out = orig(self, *a, **k)
        seen[spy.tag] = out.clone()
        return out

    def run(tag, prior):
        spy.tag = tag
        with torch.no_grad():
            model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior,
                  prompt_time=b.prompt_t)

    qe.forward = spy
    try:
        run('all', b.kpt_prior)
        dropped = b.kpt_prior.clone()
        dropped[:, 0] = float('nan')          # withhold keypoint 0 and nothing else
        run('one', dropped)
    finally:
        qe.forward = orig

    # The fused query axis is (B, T*K), t-major -- so keypoint k is every k-th column.
    a, c = seen['all'], seen['one']
    per_kpt = [torch.allclose(a[:, k::K], c[:, k::K]) for k in range(K)]
    assert not per_kpt[0], 'withholding keypoint 0 did not change its own query'
    assert all(per_kpt[1:]), f'it also changed other keypoints: {per_kpt}'


def test_wide_inherits_the_pretrained_patch_cnn(tmp_path):
    """`wide`'s patch CNN must LOAD, not re-initialise. Nothing else would show if it did not.

    `PatchProcessor`'s convs are independent of `embed_dim` (~93k params) but its MLP is not
    (~5.4M at 256). Building it at `wide`'s fusion width would inherit 93k of 5.5M and silently
    retrain 98% of the patch CNN from noise -- with no symptom other than a worse number months
    later. Building it at the pretrained width and projecting instead makes every tensor load by
    name at matching shape, which is why `warm_start` needs no special case for it at all.
    """
    from tailcyclenet.checkpoints import warm_start

    base = build_model(small('pose'), n_keypoints=5)
    ckpt = tmp_path / 'base.pth'
    torch.save({'model_state': base.state_dict()}, ckpt)

    model = build_model(small('wide'), n_keypoints=5)
    before = {k: v.clone() for k, v in model.query_encoder.patch_processor.state_dict().items()}
    fresh = warm_start(model, ckpt, verbose=False)

    patch = [n for n in fresh if 'patch_processor' in n]
    assert not patch, f'the patch CNN was left fresh instead of loaded: {patch}'
    after = model.query_encoder.patch_processor.state_dict()
    src = base.query_encoder.patch_processor.state_dict()
    assert after.keys() == before.keys()
    for k in after:
        torch.testing.assert_close(after[k], src[k])       # came from the checkpoint...
        assert not torch.allclose(after[k], before[k]), k  # ...and is not the fresh init

    # Everything genuinely new to this encoder MUST be fresh -- it inherits no fusion behaviour.
    for name in ('kpt_embed.weight', 'gate.0.weight', 'linear_qpos.weight', 'patch_proj.weight',
                 'missing_qpos', 'missing_patch'):
        assert f'query_encoder.{name}' in fresh, name


def test_wide_query_terms_follow_the_query_mode():
    """`qpos` and `patch` DEFAULT to `query`, and the two useless combinations stay unbuildable.

    Under `query = "none"` the prior is never read, so `_query_ok` is all-False for the whole run
    and `_sub_unprompted` swaps both terms for their no-query token on every query -- two constant
    vectors feeding two dead gate inputs. And with both absent `wide` ignores `query_coords`
    entirely, so `query = "prior"` could not reach the encoder: that is the combination six
    posetail-pose configs shipped as anchored arms whose anchor was a literal no-op.
    """
    prior = build_model(small('wide', query='prior'), n_keypoints=5).query_encoder
    free = build_model(small('wide', query='none'), n_keypoints=5).query_encoder

    assert prior.term_names() == ['kpt', 'query_time', 'target_time', 'gap', 'qpos', 'patch',
                                  'pp', 'intrinsic']
    # 6 terms, no qpos, no patch: this IS golden's j3 encoder.
    assert free.term_names() == ['kpt', 'query_time', 'target_time', 'gap', 'pp', 'intrinsic']
    assert not hasattr(free, 'linear_qpos') and not hasattr(free, 'patch_processor')
    assert free.n_fusion_terms == 6 and prior.n_fusion_terms == 8

    # Overriding the pair is allowed, and pos-without-patch is the `j4_prior` recipe.
    j4 = build_model(small('wide', query='prior', query_patch_embedding=False),
                     n_keypoints=5).query_encoder
    assert j4.term_names() == ['kpt', 'query_time', 'target_time', 'gap', 'qpos', 'pp', 'intrinsic']
    assert j4.n_fusion_terms == 7 and not hasattr(j4, 'patch_processor')

    # The trap stays unrepresentable: a declared prior with no route into the encoder.
    with pytest.raises(AssertionError, match='no-op'):
        build_model(small('wide', query='prior', query_pos_embedding=False,
                          query_patch_embedding=False), n_keypoints=5)
    # As does paying for a term that is constant all run.
    with pytest.raises(AssertionError, match='dead gate inputs'):
        build_model(small('wide', query='none', query_pos_embedding=True), n_keypoints=5)
    # And naming them on the encoder that cannot read them.
    with pytest.raises(AssertionError, match='only apply to'):
        build_model(small('pose', query='prior', query_pos_embedding=True), n_keypoints=5)


def test_item_dropout_reproduces_the_deployment_geometry(moving_batch):
    """A fully-dropped window must put the scene scalars exactly where deployment puts them.

    This is what per-ITEM dropout buys and per-keypoint dropout cannot. The decode is per-keypoint
    independent -- there is no attention across the query axis -- but `scene_center`,
    `scene_radius` and `cube_scale` are derived from the WHOLE `coords_q` set
    (`tracker_encoder.py:318,356`) and scale the depth and 3D outputs. Leave even a few keypoints
    prompted and those scalars are partly GT-derived, which is a condition deployment never meets:
    on allen, a per-keypoint draw at p = 0.5 lands `scene_center` 13.6 mm off (0 of 200 draws
    within 1 mm), where a per-item drop lands it at exactly 0.
    """
    from tailcyclenet.model import scene_center

    b = moving_batch
    K = b.kpt_ids.shape[1]
    q0 = scene_center(b.cgroup).view(1, 3).expand(K, 3)
    prior = b.kpt_prior[0]
    assert torch.isfinite(prior).any(), 'the fixture must carry a real prior to withhold'

    def centre(mask):                      # mask True -> that keypoint keeps its prior
        return torch.nanmean(torch.where(mask[:, None], prior, q0).float(), 0)

    deployed = centre(torch.zeros(K, dtype=torch.bool))
    torch.testing.assert_close(deployed, q0[0])                    # all dropped == query-free
    partial = centre(torch.tensor([True] + [False] * (K - 1)))     # ONE keypoint kept
    assert not torch.allclose(partial, deployed), (
        'a partially-prompted window must NOT share the query-free scene centre -- if it does, '
        'this test can no longer tell per-item from per-keypoint dropout')


@pytest.mark.parametrize('enc', ENCODERS)
def test_query_free_prediction_is_the_triangulation(moving_batch, enc):
    """With no prior anywhere, gridresid must not operate: the prediction IS the triangulation.

    A gridresid residual is an offset from the query point. Query-free, that point is the derived
    scene centre -- identical for every keypoint and carrying no information -- so the residual
    would have to explain the whole centre-to-keypoint offset from a fixed anchor.
    """
    b = moving_batch
    model = build_model(small(enc, query='none', gridresid_offset='query'),
                        n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    with torch.no_grad():
        out = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=None, prompt_time=None)

    tri = out['3d_pred_triangulate']
    torch.testing.assert_close(out['coords_pred'], tri, equal_nan=True)
    # The 3D grid CE has nothing to supervise, so its ANCHOR goes non-finite and
    # `grid_softmax_loss` (losses.py:45-52) drops the target. The `grid` dict itself must stay:
    # `losses.py:680` gates `depth_softmax` (weight 1.5) on `'grid' in outputs` too, and
    # `losses.py:458` reads `f_eff` out of it to normalise the depth Huber.
    assert not torch.isfinite(out['grid']['anchor_local']).any()
    assert torch.isfinite(out['grid']['logits_depth']).all()
    assert out['grid']['f_eff'] is not None


@pytest.mark.parametrize('prompted', [True, False], ids=['prompted', 'query_free'])
def test_query_free_keeps_the_depth_ce_and_drops_only_the_3d_ce(moving_batch, prompted):
    """The whole point of NaN-ing `anchor_local` instead of popping `grid`.

    `losses.py:680` gates BOTH `coords_softmax_3d` (weight 0.4) and `depth_softmax` (weight 1.5 --
    the largest CE term in w9.toml) on `'grid' in outputs`. Only the first is query-anchored; the
    depth target (`losses.py:769`) is `log(depths_true / (cube_scale * f_eff * sdep))`, which has
    nothing to do with the query. Popping `grid` therefore switched off the heavier of the two on
    every fully-unprompted step -- ~half of all steps at `prompt_dropout = 0.5`, and never on a
    `none` arm, which routes through `_reanchor_per_frame` and keeps `grid` alive.
    """
    from posetail.posetail.losses import TotalLoss

    b = moving_batch
    model = build_model(small('wide', query='prior', gridresid_offset='query'),
                        n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    prior = b.kpt_prior.clone() if prompted else None
    out = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior,
                prompt_time=b.prompt_t if prompted else None)

    loss_fn = TotalLoss(coords_softmax_3d_weight=0.4, depth_softmax_weight=1.5,
                        per_camera_cube_scale=True)
    loss_fn(model, out, coords_true=b.coords, vis_true=None, vis_true_cams=None,
            cgroup=b.cgroup, p2d=None, device='cpu')

    depth_ce = loss_fn.loss_history['depth_softmax_loss'][-1]
    grid_ce = loss_fn.loss_history['3d_softmax_loss'][-1]
    assert depth_ce > 0, 'the depth CE is query-independent and must fire either way'
    if prompted:
        assert grid_ce > 0
    else:
        assert grid_ce == 0, 'a non-finite anchor must zero the 3D CE, via losses.py:45-52'


def test_prior_points_are_query_anchored_and_others_triangulated(moving_batch):
    """Per-KEYPOINT selection: a prompted keypoint uses the residual, an unprompted one does not."""
    b = moving_batch
    model = build_model(small('wide', query='prior', gridresid_offset='query'),
                        n_keypoints=int(b.kpt_ids.max()) + 1).eval()

    prior = b.kpt_prior.clone()
    assert torch.isfinite(prior).all(), 'fixture must start fully prompted'
    prior[:, 1:] = float('nan')                      # keypoint 0 prompted, the rest not
    with torch.no_grad():
        out = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior,
                    prompt_time=b.prompt_t)

    tri, pred = out['3d_pred_triangulate'], out['coords_pred']
    assert not torch.allclose(pred[:, :, 0], tri[:, :, 0]), \
        'the prompted keypoint must use the query-anchored residual, not the triangulation'
    torch.testing.assert_close(pred[:, :, 1:], tri[:, :, 1:], equal_nan=True)
    assert 'grid' in out, 'a partially prompted window still has a CE target to supervise'


def test_direct_head_gets_no_gradient_at_unprompted_points(moving_batch):
    """The loss gate. Substituting the DETACHED triangulation makes the direct term constant at
    unprompted points, so `coords_loss_direct*` contributes exactly zero gradient there -- which
    is how the direct head is supervised on query points only without forking posetail's loss.

    Multiview only. The single-view path substitutes a different anchor and is covered by
    `test_single_view_substitutes_the_detached_rays`, which checks it by VALUE: at one camera this
    tiny model's soft-argmax saturates so hard that every parameter reads zero gradient, including
    for a prompted keypoint, so the positive control below cannot be established there and the
    negative one would pass vacuously.
    """
    b = moving_batch
    model = build_model(small('wide', query='prior', gridresid_offset='query'),
                        n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    prior = b.kpt_prior.clone()
    prior[:, 1:] = float('nan')
    out = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=prior,
                prompt_time=b.prompt_t)

    direct = out['3d_pred_cams_direct']
    params = [p for p in model.parameters() if p.requires_grad]

    def n_reached(k):
        """How many parameters keypoint k's direct output carries gradient to."""
        gs = torch.autograd.grad(direct[:, :, :, k].sum(), params,
                                 retain_graph=True, allow_unused=True)
        return sum(g is not None and bool(g.abs().sum() > 0) for g in gs)

    # Every parameter, not one chosen by hand: in this deliberately tiny model the soft-argmax
    # saturates, so only the final 3D head carries gradient even for a prompted point. Naming a
    # specific tensor would make the test pass for the wrong reason.
    assert n_reached(0) > 0, 'the prompted keypoint must still reach the direct head'
    for k in range(1, b.kpt_ids.shape[1]):
        assert n_reached(k) == 0, f'unprompted keypoint {k} still reaches {n_reached(k)} params'


def test_single_view_predicts_from_rays_instead_of_dropping_the_step(moving_batch):
    """3D single-view has no triangulation, so the back-projected rays are the anchor.

    This used to hand a `loss_kpt_mask` up and let `run_batch` NaN the entire 3D target, which
    made the whole STEP non-finite whenever no keypoint had a prior. `cams_to_sample = [1, 8]`
    draws one camera on 1/8 of 3D items, so at `prompt_dropout = 0.5` that binned ~6% of every
    `prior` arm's gradient updates while the matched `none` arm kept all of its own.
    """
    b = moving_batch
    model = build_model(small('wide', query='none', gridresid_offset='query'),
                        n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    with torch.no_grad():
        out = model([b.views[SEEING_CAM]], b.kpt_ids, [b.cgroup[SEEING_CAM]], mode='3d',
                    kpt_prior=None, prompt_time=None)

    assert out.get('3d_pred_triangulate') is None, 'one camera cannot triangulate'
    assert 'loss_kpt_mask' not in out, 'the mask is retired -- the rays are a real prediction'
    assert torch.isfinite(out['coords_pred']).all(), 'a finite prediction, not a dropped step'
    assert 'grid' in out, 'the depth CE (weight 1.5) must survive the single-view path'


def test_rays_fallback_is_a_mean_not_the_library_weighted_sum(moving_batch):
    """At one camera the prediction must BE that camera's ray point.

    `3d_pred_rays` is `sum_c sigmoid(conf_c) * X_c` with no division (`tracker_encoder.py:637`),
    so at one camera it lands a factor `sigmoid(conf)` -- about half -- of the way from the world
    origin to the animal. Substituting it verbatim would have made the single-view path finite and
    wrong, which is worse than skipping the step.
    """
    b = moving_batch
    model = build_model(small('wide', query='none', gridresid_offset='query'),
                        n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    with torch.no_grad():
        out = model([b.views[SEEING_CAM]], b.kpt_ids, [b.cgroup[SEEING_CAM]], mode='3d',
                    kpt_prior=None, prompt_time=None)
    torch.testing.assert_close(out['coords_pred'], out['3d_pred_cams_rays'][0])


def test_single_view_substitutes_the_detached_rays(moving_batch):
    """The loss gate on the single-view path, checked by value rather than by gradient.

    The substituted anchor must be bit-identical to the rays point and carry no grad_fn, which is
    what makes `coords_loss_direct*` constant -- and therefore zero-gradient -- at unprompted
    points. If the detach were dropped, the direct head would silently reopen on every unprompted
    keypoint of every single-camera step.
    """
    b = moving_batch
    model = build_model(small('wide', query='prior', gridresid_offset='query'),
                        n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    prior = b.kpt_prior.clone()
    prior[:, 1:] = float('nan')                      # keypoint 0 prompted, the rest not
    out = model([b.views[SEEING_CAM]], b.kpt_ids, [b.cgroup[SEEING_CAM]], mode='3d',
                kpt_prior=prior, prompt_time=b.prompt_t)

    rays = out['3d_pred_cams_rays']
    unprompted = out['3d_pred_cams_direct'][0, :, :, 1:]
    torch.testing.assert_close(unprompted, rays[0][:, :, 1:].detach())

    def grad_to_rays(t):
        g, = torch.autograd.grad(t.sum(), rays, retain_graph=True, allow_unused=True)
        return 0.0 if g is None else float(g.abs().sum())

    # Positive control: the rays tensor IS in the graph and gradient does flow through the einsum
    # that reduces it. Without this the negative below would pass for the wrong reason. Checking
    # `requires_grad` would not work either -- `torch.where` propagates it from the OTHER branch.
    assert grad_to_rays(out['3d_pred_rays']) > 0
    assert grad_to_rays(unprompted) == 0.0, 'the substituted anchor must be detached'


@pytest.mark.parametrize('enc', ENCODERS)
def test_gridresid_offset_switches_the_anchor(moving_batch, enc):
    """`gridresid_offset` picks what the residual is measured FROM, and the two must differ.

    "triangulated" recovers the residual and re-adds it to EACH frame's own triangulation for
    every keypoint -- posetail-pose's `_reanchor_per_frame` (its model.py:756, applied
    unconditionally), measured 2.07 -> 1.37 mm within-session. "query" keeps the library's native
    query anchor, but only where a real prior supplied one.

    Query-free the two are maximally far apart: "triangulated" still produces a residual on top of
    the triangulation, while "query" has no valid anchor anywhere and returns the triangulation
    itself. If these ever compare equal the switch is not wired to anything.
    """
    b = moving_batch
    n_kpt = int(b.kpt_ids.max()) + 1
    outs = {}
    for off in ('query', 'triangulated'):
        m = build_model(small(enc, query='none', gridresid_offset=off), n_keypoints=n_kpt).eval()
        torch.manual_seed(0)
        with torch.no_grad():
            outs[off] = m(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=None,
                          prompt_time=None)

    q, t = outs['query'], outs['triangulated']
    torch.testing.assert_close(q['coords_pred'], q['3d_pred_triangulate'], equal_nan=True)
    assert not torch.allclose(t['coords_pred'], t['3d_pred_triangulate'], equal_nan=True), \
        '"triangulated" must add a residual on top of the triangulation, not return it'
    # "triangulated" keeps a finite CE target (re-based onto the new anchor); "query" has no valid
    # anchor query-free, so it NaNs `anchor_local` -- which is the library's own off switch for the
    # 3D CE alone. It must NOT drop the `grid` dict: `depth_softmax` (weight 1.5) and `f_eff` live
    # behind the same `'grid' in outputs` gate (losses.py:458,680) and are query-independent.
    assert torch.isfinite(t['grid']['anchor_local']).all()
    assert 'grid' in q and not torch.isfinite(q['grid']['anchor_local']).any()
    for k in ('logits_depth', 'f_eff', 'cube_scale'):
        assert torch.isfinite(q['grid'][k]).all(), f'{k} must survive the query-free path'

    with pytest.raises(AssertionError, match='gridresid_offset'):
        build_model(small(enc, gridresid_offset='nonsense'), n_keypoints=n_kpt)

    # AND AN ABSENT KEY IS AN ERROR, not `'query'`. Both values load the same tensors, so a
    # checkpoint trained under one and built under the other is wrong without raising -- which is
    # exactly what happened to every keyless pre-`bcbfbc1` 3D run.
    cfg = small(enc)
    cfg.pop('gridresid_offset')
    with pytest.raises(KeyError, match='gridresid_offset'):
        build_model(cfg, n_keypoints=n_kpt)


def test_a_degenerate_triangulation_does_not_nan_the_whole_step():
    """`torch.where` DOES NOT STOP THE NaN, and that is not obvious.

    `_query_anchored` guards the degenerate solve with
    `torch.where(isfinite(tri), tri, rays)`, which routes zero gradient into the bad entries. But
    the NaN is not carried through the forward -- it is CREATED in the backward:
    `triangulate_simple_batch_reg` ends in `torch.linalg.solve` (cube.py:299), whose backward
    solves against its own NaN factorisation and returns NaN regardless of the incoming gradient.
    Param grads are summed over points, so ONE degenerate point NaNs every parameter and the whole
    step is dropped by `train.py`'s grad-norm guard -- counted as an unattributed `skipped`.

    `nan_to_num` on the forward value does not help, for the same reason. Detaching does.
    """
    def grads(detach):
        p = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        A = torch.stack([p, p * float('nan')])          # element 1: what a NaN box centre feeds it
        x = torch.linalg.solve(A, torch.ones(2, 2))
        assert not torch.isfinite(x).all(), 'the probe must actually be degenerate'
        if detach and not torch.isfinite(x).all():
            x = x.detach()
        y = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        (y.sum() + p.sum()).backward()                  # + the other heads, which must keep going
        return p.grad

    assert torch.isnan(grads(detach=False)).all(), \
        'if `where` alone ever starts masking the backward, the guard in model.py is dead code'
    assert torch.isfinite(grads(detach=True)).all(), 'the other heads must still get gradient'


def test_the_triangulation_repair_leaves_the_forward_value_alone():
    """The detach above must be invisible to every number this repo has published.

    It only changes what the backward does on a step that was being thrown away anyway, so the
    repaired tensor -- which the loss, the metric and `--anchor carry`'s seed all read -- has to
    be bit-identical with and without it.
    """
    from tailcyclenet.model import _rays_fallback

    tri = torch.tensor([[1.0, 2.0, 3.0], [float('nan'), 5.0, 6.0]], requires_grad=True)
    fallback = torch.zeros(2, 3)
    plain = torch.where(torch.isfinite(tri), tri, fallback)
    guarded = torch.where(torch.isfinite(tri.detach()), tri.detach(), fallback)
    assert torch.equal(torch.nan_to_num(plain, nan=-1.0), torch.nan_to_num(guarded, nan=-1.0))
    assert callable(_rays_fallback)


@pytest.mark.parametrize('enc', ENCODERS)
def test_a_grown_registry_keeps_its_trained_keypoint_rows(tmp_path, enc):
    """The one workflow the registry file exists for, and it used to reset the whole table.

    `Registry.build(base=)` APPENDS -- it raises rather than renumbering -- so a run that adds a
    dataset gets `kpt_embed.weight` at (n0+k, d) where the checkpoint has (n0, d).
    `_filter_shape_mismatch` drops a changed shape WHOLE, so every trained identity row went back
    to noise and retrained at `kpt_lr`, while CLAUDE.md promised the opposite. It is camouflaged
    because under `wide` most of `query_encoder.*` is expected to be dropped anyway.
    """
    from tailcyclenet.checkpoints import warm_start

    base = build_model(small(enc), n_keypoints=3)
    ckpt = tmp_path / f'base_{enc}.pth'
    torch.save({'model_state': base.state_dict()}, ckpt)

    model = build_model(small(enc), n_keypoints=5)
    virgin = model.query_encoder.kpt_embed.weight.detach().clone()
    src = base.query_encoder.kpt_embed.weight.detach()
    assert not torch.allclose(virgin[:3], src), 'the fresh table must differ, or this proves nothing'

    fresh = warm_start(model, ckpt, verbose=False, base_names=('a', 'b', 'c'))
    got = model.query_encoder.kpt_embed.weight.detach()
    torch.testing.assert_close(got[:3], src)               # the trained ids survived...
    torch.testing.assert_close(got[3:], virgin[3:])        # ...and the new ones are still fresh
    assert not any('kpt_embed' in n for n in fresh), \
        'the table must no longer be reported as dropped or left fresh'

    # NEGATIVE CONTROL: a base registry that does not match the checkpoint's row count is not that
    # registry's table, and copying it would point each row at a different body part. Refused.
    model2 = build_model(small(enc), n_keypoints=5)
    virgin2 = model2.query_encoder.kpt_embed.weight.detach().clone()
    warm_start(model2, ckpt, verbose=False, base_names=('a', 'b'))
    torch.testing.assert_close(model2.query_encoder.kpt_embed.weight.detach(), virgin2)


@pytest.mark.parametrize('enc', ENCODERS)
def test_an_unprompted_keypoint_carries_no_query_time(moving_batch, enc):
    """A "query-free" TRAINING step must be the same forward as the query-free DEPLOYMENT step.

    It was not. `prompt_dropout` NaNs `kpt_prior` and leaves `prompt_t` at each keypoint's first
    labelled frame (>0 on 19.5% of rat-city windows), while `val`, `self_prompt` and `run_group`
    all pass `prompt_time=None`, which zeroes it. `embed_query_time` and `embed_gap` are
    UNCONDITIONAL fusion terms -- unlike `patch`/`qpos` they carry no learned no-query token -- so
    at `prompt_dropout = 0.4` roughly 40% of steps trained a forward no deployment path produces.

    It is also eval rule 7's shape: WHEN a keypoint was first labelled is GT-derived, and it was
    reaching the model at a keypoint with no prior at all.
    """
    b = moving_batch
    model = build_model(small(enc, query='prior'), n_keypoints=int(b.kpt_ids.max()) + 1).eval()
    K = b.kpt_ids.shape[1]
    dropped = torch.full_like(b.kpt_prior, float('nan'))

    with torch.no_grad():
        deployed = model(b.views, b.kpt_ids, b.cgroup, mode='3d')['coords_pred']
        # What a dropout step used to send: no prior, but a real per-keypoint query time.
        late = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=dropped,
                     prompt_time=torch.full((1, K), 1, dtype=torch.int32))['coords_pred']
    torch.testing.assert_close(late, deployed)

    # POSITIVE CONTROL: with a REAL prior the query time must still matter, or the assertion above
    # passes vacuously on a tiny model whose time terms happen to be inert.
    assert torch.isfinite(b.kpt_prior).any(), 'the fixture must carry a real prior'
    with torch.no_grad():
        at0 = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=b.kpt_prior,
                    prompt_time=torch.zeros((1, K), dtype=torch.int32))['coords_pred']
        at1 = model(b.views, b.kpt_ids, b.cgroup, mode='3d', kpt_prior=b.kpt_prior,
                    prompt_time=torch.ones((1, K), dtype=torch.int32))['coords_pred']
    assert not torch.allclose(at0, at1), 'query_time is inert here, so this test proves nothing'


# ---------------------------------------------------------------------------------------------
# the staged encoder unfreeze, on a REAL ViT
# ---------------------------------------------------------------------------------------------
# ViT-base is depth 12 with hierarchical taps at [2,5,8,11], so `video_encoder_finetune_last_n_
# layers = 4` gives a trainable range of blocks 8..11 -- selecting SOME blocks, not all, which a
# hand-built fixture cannot check for us. The gate, the idempotence and the block selection are
# posetail 0.3.5's (`TrackerEncoder.unfreeze_video_encoder`); what is pinned here is that this
# repo gets what upstream promises, plus the norms extension upstream does not do.

def _staged_model(n_last=4, at=3):
    return build_model(small('wide', video_encoder_requires_grad=at,
                             video_encoder_finetune_last_n_layers=n_last), n_keypoints=3)


def test_staged_unfreeze_selects_exactly_the_last_n_blocks():
    m = _staged_model(n_last=4, at=3)
    enc = m.scene_encoder.encoder
    assert len(enc.blocks) == 12 and enc.hierarchical_layers == [2, 5, 8, 11], 'fixture moved'

    for it in (0, 1, 2):
        assert m.unfreeze_video_encoder(it) is False, f'fired early at {it}'
        assert not any(p.requires_grad for p in enc.parameters()), 'encoder trainable before time'

    assert m.unfreeze_video_encoder(3) is True
    for i, blk in enumerate(enc.blocks):
        want = i >= 8
        got = all(p.requires_grad for p in blk.parameters())
        assert got == want, f'block {i}: trainable={got}, expected {want}'
    assert m.unfreeze_video_encoder(4) is False, 'not idempotent'


def test_staged_unfreeze_norms_extension_tracks_the_block_range():
    """Upstream unfreezes `blocks[-N:]` plus an `encoder.norm` VJEPA 2.1 has no attribute for, so
    `norms_block` -- applied at the hierarchical taps and feeding the decoder -- would stay frozen
    behind a trainable block. Taps [2,5,8,11]: N=4 -> range starts at 8, so norms 2,3; N=2 -> 3."""
    m = _staged_model(n_last=4, at=0)
    enc = m.scene_encoder.encoder
    assert _norms_in_range(enc, 4) == [2, 3]
    assert _norms_in_range(enc, 2) == [3]
    assert _norms_in_range(enc, 12) == [0, 1, 2, 3], 'the whole encoder unfreezes every norm'

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.0)
    info = apply_staged_unfreeze(m, opt, {'learning_rate': 1e-4, 'encoder_lr_scale': 0.1}, 0)
    assert info['norms'] == [2, 3], info
    for i, nrm in enumerate(enc.norms_block):
        want = i >= 2
        got = all(p.requires_grad for p in nrm.parameters())
        assert got == want, f'norms_block[{i}]: trainable={got}, expected {want}'


def test_gradients_reach_the_unfrozen_blocks_only(moving_batch):
    """A real forward/backward after the unfreeze: finite grads on the trainable blocks, None on
    the frozen ones. `SceneRepresentation` turns activation checkpointing on unconditionally, so
    this also pins that the checkpointed backward works through a partially-frozen encoder."""
    b = moving_batch
    m = _staged_model(n_last=4, at=0)
    assert m.unfreeze_video_encoder(0) is True
    m(b.views, b.kpt_ids, b.cgroup, mode='3d')['coords_pred'].nansum().backward()
    enc = m.scene_encoder.encoder
    for i, blk in enumerate(enc.blocks):
        grads = [p.grad for p in blk.parameters()]
        if i >= 8:
            assert all(g is not None and torch.isfinite(g).all() for g in grads), \
                f'block {i} is trainable but got no finite gradient'
        else:
            assert all(g is None for g in grads), f'frozen block {i} received a gradient'
