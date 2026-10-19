"""The scorer model: the things that produce numbers instead of exceptions when wrong.

Keypoint ids must tile the (t, k) query axis t-major with the caller's GLOBAL ids; a fully-missing
keypoint must still score finitely; a moving rig must be refused (the rays assume a static rig).
"""
from pathlib import Path

import pytest
import torch

from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.pose.query_encoder import _tile_to_query_axis
from tailcyclenet.scorer import build_scorer
from tailcyclenet.scorer.model import FrameCameraPooling
from tailcyclenet.pose.model import build_model, share_video_encoder_if_identical

from .test_model import SMALL

SCORER_KW = dict(pool_num_heads=8, score_hidden=64, use_precision=True)


def _scorer(n_keypoints, **over):
    """A small scorer. `video_encoder_pretrained=False` -- a test must not touch the network."""
    cfg = {**SMALL, 'box_prompt': 'film', 'video_encoder_pretrained': False, **over}
    cfg.pop('query_encoder', None)
    return build_scorer(cfg, n_keypoints, **SCORER_KW)


@pytest.fixture(scope='module')
def scorer_batch(tmp_path_factory):
    """One real 3D window: (model, views, coords, cgroup, K)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    tmp = tmp_path_factory.mktemp('scorer')
    C._session_3d(tmp / 'mouselike' / 'train' / 'sess_c')
    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    ds = PoseDataset(tmp / 'mouselike', 'train', cfg, train=False)
    item = ds[0]
    views = [v[None] for v in item[0]]
    coords = item[1][None]
    # The loader's OWN global registry ids -- the same tensor the pose model is given. Using
    # `arange(K)` here would make the identity test below unable to tell the two apart.
    kpt_ids = item[10][None]
    model = _scorer(ds.registry.n_keypoints, stride_length=4).eval()
    return model, views, coords, item[4], ds.registry.n_keypoints, kpt_ids



def test_video_encoder_weights_are_shared_only_when_exactly_equal():
    pose = build_model(SMALL, n_keypoints=5).eval()
    scorer = _scorer(5, stride_length=4).eval()
    scorer.scene_encoder.encoder.load_state_dict(pose.scene_encoder.encoder.state_dict())
    assert share_video_encoder_if_identical(pose, scorer)
    assert scorer.scene_encoder.encoder is pose.scene_encoder.encoder

    other = _scorer(5, stride_length=4).eval()
    with torch.no_grad():
        other.scene_encoder.encoder.blocks[0].norm1.weight[0].add_(1)
    assert not share_video_encoder_if_identical(pose, other)
    assert other.scene_encoder.encoder is not pose.scene_encoder.encoder


def _fully_observed(coords):
    """`coords` with every NaN filled, so a test's own NaN pattern is the only missingness."""
    return torch.nan_to_num(coords.clone(), nan=0.0)


def _forward_capturing(model, module, views, coords, cgroup, kpt_ids):
    """(scores, precision, kwargs `module` was called with) for one no-grad forward."""
    captured = {}
    h = module.register_forward_pre_hook(
        lambda mod, args, kwargs: captured.update(kwargs, _x=args[0] if args else None),
        with_kwargs=True)
    try:
        with torch.no_grad():
            scores, precision = model(views, coords, cgroup, kpt_ids)
    finally:
        h.remove()
    return scores, precision, captured


def test_kpt_ids_tile_the_query_axis_t_major():
    """Slot `t*K + k` carries `kpt_ids[k]`; k-major would score each point as another."""
    ids = torch.tensor([[2, 0, 3]])
    assert torch.equal(_tile_to_query_axis(ids, 6, '_kpt_ids'),
                       torch.tensor([[2, 0, 3, 2, 0, 3]]))


def test_the_scorer_uses_the_caller_global_ids_not_arange(scorer_batch):
    """The encoder must receive the caller's registry ids verbatim, sliced per `kpt_chunk`.

    A session may subset/reorder the registry (gotcha 4); `arange(K)` would keep every shape right.
    """
    model, views, coords, cgroup, K, loader_ids = scorer_batch
    assert K >= 3
    # A subset of a bigger registry, out of order and NOT a dense range -- the shape a session
    # whose `names` reorder/subset the registry actually produces.
    big = K + 7
    ids = torch.tensor([[5, 2, 9, 0, 7, 1, 3, 4, 6, 8, 10, 11, 12, 13][:K]])
    assert ids.max() < big and not torch.equal(ids, torch.arange(K)[None]), 'fixture must differ'
    big_model = _scorer(big, stride_length=4).eval()

    seen = []
    orig = big_model.query_encoder.forward

    def spy(*args, **kwargs):
        seen.append(big_model.query_encoder._kpt_ids.clone())
        return orig(*args, **kwargs)

    big_model.query_encoder.forward = spy
    try:
        with torch.no_grad():
            big_model(views, coords, cgroup, ids)
    finally:
        big_model.query_encoder.forward = orig
    assert len(seen) == 1
    assert torch.equal(seen[0], ids), 'the query encoder did not receive the caller\'s ids'

    # Chunked: each slice must carry ITS OWN ids, not the first slice's.
    seen.clear()
    big_model.query_encoder.forward = spy
    try:
        with torch.no_grad():
            big_model(views, coords, cgroup, ids, kpt_chunk=1)
    finally:
        big_model.query_encoder.forward = orig
    assert len(seen) == K, f'expected {K} chunk forwards, got {len(seen)}'
    for i, got in enumerate(seen):
        assert got.shape == (1, 1), f'chunk {i} got ids of shape {tuple(got.shape)}'
        assert int(got[0, 0]) == int(ids[0, i]), f'chunk {i} carried the wrong id'

    # And a dense `arange(K)` would give a DIFFERENT result -- so this test can tell them apart.
    with torch.no_grad():
        a, _ = big_model(views, coords, cgroup, ids)
        b, _ = big_model(views, coords, cgroup, torch.arange(K)[None])
    assert not torch.allclose(a, b), 'permuted ids and arange gave the same scores'


def test_refusals_by_name(scorer_batch):
    """Out-of-registry ids, a moving rig, and an absent `gridresid_offset` raise, never guess."""
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    bad = torch.arange(K)[None].clone()
    bad[0, -1] = K + 5
    with pytest.raises(ValueError, match='outside this scorer'), torch.no_grad():
        model(views, coords, cgroup, bad)

    moving = [dict(c) for c in cgroup]
    e = moving[0]['ext']
    moving[0]['ext'] = e[None].expand(coords.shape[1], *e.shape).contiguous()
    with pytest.raises(ValueError, match='MOVING rig'), torch.no_grad():
        model(views, coords, moving, kpt_ids)

    cfg = {k: v for k, v in SMALL.items() if k != 'gridresid_offset'}
    with pytest.raises(KeyError, match='gridresid_offset'):
        build_scorer({**cfg, 'box_prompt': 'film'}, 4, **SCORER_KW)


# -- A.7: missing points ----------------------------------------------------------------

def test_the_pooling_mask_drops_exactly_the_missing_slots(scorer_batch):
    """The `attn_pool` mask must be `~valid` over (t, camera); a leak teaches missing == bad."""
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    c = _fully_observed(coords)
    c[:, 1, 2] = float('nan')                      # exactly one (frame, keypoint) slot
    _s, _p, kw = _forward_capturing(model, model.attn_pool, views, c, cgroup, kpt_ids)
    mask = kw['key_padding_mask']

    T, n_cams = coords.shape[1], len(cgroup)
    assert mask.shape == (1, K, T, n_cams)
    valid = torch.isfinite(c).all(-1)                                  # [b, t, k]
    want = (~valid).permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, n_cams)
    assert torch.equal(mask, want), 'the pooling mask does not match the observed slots'
    assert int(mask.sum()) == n_cams, f'expected exactly {n_cams} dropped slots'


def test_a_fully_missing_keypoint_is_force_unmasked_and_scores_finitely(scorer_batch):
    """Softmax over an all-masked key set is NaN, so only a FULLY missing point is un-dropped."""
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    c = _fully_observed(coords)
    c[:, :, 1] = float('nan')                      # keypoint 1, every frame
    c[:, 0, 2] = float('nan')                      # plus one PARTIAL slot on keypoint 2
    scores, precision, kw = _forward_capturing(model, model.attn_pool, views, c, cgroup, kpt_ids)
    mask = kw['key_padding_mask']
    assert not bool(mask[0, 1].any()), 'the force-unmask did not fire for a fully missing point'
    assert bool(mask[0, 2, 0].all()), 'a partial point was un-dropped (the guard is too wide)'
    assert not bool(mask[0, 2, 1:].any())
    assert scores.shape == (1, K)
    assert torch.isfinite(scores).all() and torch.isfinite(precision).all()


# -- hygiene -----------------------------------------------------------------------------

def test_stashes_are_cleared_on_success_and_when_the_forward_raises(scorer_batch):
    """A leaked `_kpt_ids`/`_query_ok`/`_box_prompt` stash silently applies to the next window."""
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    enc = model.query_encoder

    def cleared():
        return enc._kpt_ids is None and enc._query_ok is None and enc._box_prompt is None

    with torch.no_grad():
        model(views, coords, cgroup, kpt_ids)
    assert cleared(), 'a stash leaked out of a successful forward'

    def boom(mod, args, kwargs):
        raise RuntimeError('injected decoder failure')

    h = model.decoder.register_forward_pre_hook(boom, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match='injected'), torch.no_grad():
            model(views, coords, cgroup, kpt_ids)
    finally:
        h.remove()
    assert cleared(), 'a stash leaked out of a raising forward'


def test_kpt_chunk_is_exact(scorer_batch, frame_model):
    """Scene scalars/mask/NaN fill are computed over full K before slicing, so chunks are exact
    -- for both the sequence and the frame readout."""
    sequence, views, coords, cgroup, K, kpt_ids = scorer_batch
    for model in (sequence, frame_model):
        with torch.no_grad():
            whole_s, whole_p = model(views, coords, cgroup, kpt_ids)
            chunk_s, chunk_p = model(views, coords, cgroup, kpt_ids, kpt_chunk=1)
        assert whole_s.shape == chunk_s.shape
        assert torch.allclose(whole_s, chunk_s, atol=1e-5), (whole_s - chunk_s).abs().max()
        assert torch.allclose(whole_p, chunk_p, atol=1e-5)


def test_a_2d_window_scores_single_camera(tmp_path_factory):
    """The 2D path: one camera, pixel coords, `mode_idx` 0, and the 2D head untouched."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    tmp = tmp_path_factory.mktemp('scorer2d')
    C._session_2d(tmp / 'ratlike' / 'train' / 'sess_a')
    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                       crop_jitter=0.0, prompt_dropout=0.0)
    ds = PoseDataset(tmp / 'ratlike', 'train', cfg, train=False)
    item = ds[0]
    views = [v[None] for v in item[0]]
    coords = item[1][None]
    cgroup = item[4]
    assert coords.shape[-1] == 2 and len(cgroup) == 1
    model = _scorer(ds.registry.n_keypoints, stride_length=4).eval()
    with torch.no_grad():
        scores, precision = model(views, coords, cgroup, item[10][None])
    assert scores.shape == (1, ds.registry.n_keypoints)
    assert torch.isfinite(scores).all()


# -- explicit framewise output ---------------------------------------------------------------

@pytest.fixture(scope='module')
def frame_model(scorer_batch):
    return _scorer(scorer_batch[4], stride_length=4, output_granularity='frame').eval()


def test_frame_mode_keeps_time_and_masks_per_frame_keypoint(scorer_batch, frame_model):
    """One readout per frame, cameras pooled within a frame, no temporal embedding; a missing
    (frame, keypoint) is force-unmasked alone without touching that keypoint's other frames."""
    _sequence, views, coords, cgroup, K, kpt_ids = scorer_batch
    model = frame_model
    B, T, _K, _R = coords.shape
    scores, precision, kw = _forward_capturing(model, model.attn_pool.pool, views, coords,
                                               cgroup, kpt_ids)
    assert scores.shape == precision.shape == (B, T, K)
    assert torch.isfinite(scores).all() and torch.isfinite(precision).all()
    assert not model.attn_pool.use_time_embedding
    assert not hasattr(model.attn_pool.pool, 'time_embed')
    assert kw['_x'].shape[:4] == (B * T, 1, K, len(cgroup))
    assert kw['key_padding_mask'].shape == (B * T, K, 1, len(cgroup))

    c = _fully_observed(coords)
    c[:, 1, 2] = float('nan')
    scores, precision, kw = _forward_capturing(model, model.attn_pool, views, c, cgroup, kpt_ids)
    mask = kw['key_padding_mask']
    assert mask.shape == (1, T, K, len(cgroup))
    assert not bool(mask.sum())
    assert torch.isfinite(scores).all() and torch.isfinite(precision).all()


def test_frame_mode_triplet_keeps_structured_axes(scorer_batch, frame_model):
    """Only frame mode exposes [B,T,K,member], unlike the legacy flattened sequence API."""
    sequence, views, coords, cgroup, K, kpt_ids = scorer_batch
    trip = {
        'good': (views, coords, cgroup),
        'bad': (views, coords + 0.25, cgroup),
        'anchor': (views, coords, cgroup),
        'kpt_ids': kpt_ids,
        'anchor_label': 1.0,
        'occlusion': None,
    }
    with torch.no_grad():
        fs, fp, fl = frame_model.score_triplet(trip)
        ss, sp, sl = sequence.score_triplet(trip)
    assert fs.shape == fp.shape == fl.shape == (1, coords.shape[1], K, 3)
    assert ss.shape == sp.shape == sl.shape == (1, K, 3)
    assert torch.equal(fl[..., 0], torch.ones_like(fl[..., 0]))
    assert torch.equal(fl[..., 1], -torch.ones_like(fl[..., 1]))


def test_frame_camera_pooling_forces_only_all_missing_rows_finite():
    """The local camera pool's mask conversion keeps valid rows and repairs all-missing rows."""
    x = torch.randn(2, 3, 4, 2, 8)
    mask = torch.zeros(2, 3, 4, 2, dtype=torch.bool)
    mask[0, 1, 2] = True
    pool = FrameCameraPooling(8, num_heads=2).eval()
    captured = {}
    h = pool.pool.register_forward_pre_hook(
        lambda mod, args, kwargs: captured.update(kwargs), with_kwargs=True)
    try:
        with torch.no_grad():
            out = pool(x, mask)
    finally:
        h.remove()
    assert out.shape == (2, 3, 4, 8) and torch.isfinite(out).all()
    got = captured['key_padding_mask']
    assert got.shape == (2 * 3, 4, 1, 2)
    # [batch=0, frame=1, keypoint=2] is the sole repaired row.
    assert not bool(got[1, 2].any())
    assert not bool(got[[i for i in range(6) if i != 1]].any())
