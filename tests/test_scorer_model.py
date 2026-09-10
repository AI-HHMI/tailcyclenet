"""The scorer model: the things that produce numbers instead of exceptions when wrong.

Three of these are the compatibility facts the whole port rests on and none of them is visible in
a loss curve:

  * the keypoint ids must tile the (t, k) query axis **t-major**, or every point is scored with
    another point's identity;
  * a fully-missing keypoint must still score FINITELY (the pooling softmax over all-`-inf` is
    NaN), and the pooling mask must drop exactly the missing (t, camera) slots;
  * a moving rig must be REFUSED, because `_scene_scalars` and the ray construction treat the rig
    as static and would silently mis-project.
"""
from pathlib import Path

import pytest
import torch

from tailcyclenet.dataset import LoaderConfig, PoseDataset
from tailcyclenet.query_encoder import _tile_to_query_axis
from tailcyclenet.scorer import build_scorer

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



def _fully_observed(coords):
    """`coords` with every NaN filled, so a test's own NaN pattern is the only missingness.

    The fixtures carry incidental missing slots (that is what a partially-labelled window IS), and
    a mask test that assumed a particular slot was observed silently tested the wrong thing when it
    was not. Filling first makes the premise explicit.
    """
    return torch.nan_to_num(coords.clone(), nan=0.0)

# -- A.2: the id tiling is t-major ------------------------------------------------------

def test_kpt_ids_tile_the_query_axis_t_major():
    """Slot `t*K + k` must carry id `kpt_ids[k]`.

    `WideQueryEncoder` tiles (B, K) -> (B, T*K), and the scorer builds its query axis as `b (t k)`
    with `k` moving fastest. `repeat` gives that; `repeat_interleave` would give the k-major
    ordering, under which every keypoint is scored with another keypoint's learned identity vector.
    """
    ids = torch.tensor([[2, 0, 3]])
    assert torch.equal(_tile_to_query_axis(ids, 6, '_kpt_ids'),
                       torch.tensor([[2, 0, 3, 2, 0, 3]]))
    assert not torch.equal(_tile_to_query_axis(ids, 6, '_kpt_ids'),
                           torch.repeat_interleave(ids, 2, dim=1))


def test_the_scorer_uses_the_caller_global_ids_not_arange(scorer_batch):
    """The ids the encoder receives must be the caller's GLOBAL registry ids, verbatim.

    A session may reorder or SUBSET the registry's names (CLAUDE.md gotcha 4), so session position
    0 is not registry id 0 and the ids need not be contiguous. Generating `arange(K)` would score
    every keypoint under some other body part's identity vector while every shape stayed correct.
    Built with a registry LARGER than the session uses and a non-contiguous, permuted id set, so a
    dense range could not pass by accident; and the ids are sliced when `kpt_chunk` splits the axis.
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


def test_an_out_of_registry_keypoint_id_is_refused_by_name(scorer_batch):
    """Scoring a root whose names were not in the training registry must be refused, not indexed.

    The scorer is conditioned on the registry (`kpt_embed`), so a calms21 scorer has no row for an
    allen or 3dpop keypoint. Refused where the caller still knows which root it came from.
    """
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    bad = torch.arange(K)[None].clone()
    bad[0, -1] = K + 5
    with pytest.raises(ValueError, match='outside this scorer'):
        with torch.no_grad():
            model(views, coords, cgroup, bad)


# -- A.7: missing points ----------------------------------------------------------------

def test_a_fully_missing_keypoint_still_scores_finitely(scorer_batch):
    """The all-masked force-unmask guard.

    A keypoint with no coordinate anywhere masks every (t, camera) slot, and attention's softmax
    over an all-masked key set is a NaN. `min_valid_frames` is a training-side guarantee, so this
    must hold for a caller that breaks it.
    """
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    c = _fully_observed(coords)
    c[:, :, 0] = float('nan')                      # one keypoint, every frame
    with torch.no_grad():
        scores, precision = model(views, c, cgroup, kpt_ids)
    assert torch.isfinite(scores).all(), 'all-missing keypoint produced a non-finite score'
    assert torch.isfinite(precision).all()
    assert scores.shape == (1, K)

    # Do NOT assert the other keypoints are unchanged. The coupling is the FULL-K scene scalars
    # (`cube_scale`/`scene_center`/`scene_radius` are computed over all K before any masking), not
    # any cross-point attention -- the decoder documents none.
    assert torch.isfinite(scores[0, 1:]).all()


def _mask_for(model, views, coords, cgroup, kpt_ids):
    captured = {}
    h = model.attn_pool.register_forward_pre_hook(
        lambda mod, args, kwargs: captured.update(kwargs), with_kwargs=True)
    try:
        with torch.no_grad():
            model(views, coords, cgroup, kpt_ids)
    finally:
        h.remove()
    return captured['key_padding_mask']


def test_the_pooling_mask_drops_exactly_the_missing_slots(scorer_batch):
    """The mask handed to `attn_pool` must be `~valid` over (t, camera), True == drop.

    A mask that is transposed or widened silently pools the wrong slots -- and missingness is
    supposed to be ORTHOGONAL to track quality, so a leak here teaches the scorer that a dropped
    point is a bad track.
    """
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    c = _fully_observed(coords)
    c[:, 1, 2] = float('nan')                      # exactly one (frame, keypoint) slot
    mask = _mask_for(model, views, c, cgroup, kpt_ids)

    T, n_cams = coords.shape[1], len(cgroup)
    assert mask.shape == (1, K, T, n_cams)
    valid = torch.isfinite(c).all(-1)                                  # [b, t, k]
    want = (~valid).permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, n_cams)
    assert torch.equal(mask, want), 'the pooling mask does not match the observed slots'
    assert bool(mask[0, 2, 1, :].all()), 'the missing slot was NOT dropped'
    assert not bool(mask[0, 2, 0, :].any()), 'an observed slot was dropped'
    assert int(mask.sum()) == n_cams, f'expected exactly {n_cams} dropped slots'


def test_the_force_unmask_only_fires_for_a_fully_masked_point(scorer_batch):
    """A fully-missing keypoint is un-dropped wholesale; nothing else may be un-dropped."""
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    c = _fully_observed(coords)
    c[:, :, 1] = float('nan')                      # keypoint 1, every frame
    c[:, 0, 2] = float('nan')                      # plus one PARTIAL slot on keypoint 2
    mask = _mask_for(model, views, c, cgroup, kpt_ids)

    T = coords.shape[1]
    assert not bool(mask[0, 1].any()), 'the force-unmask did not fire for a fully missing point'
    # the PARTIAL point keeps its own drops -- the guard must be per-point, not blanket
    assert bool(mask[0, 2, 0].all()), 'a partial point was un-dropped (the guard is too wide)'
    assert not bool(mask[0, 2, 1:].any()), \
        f'the partial point lost slots it should have kept (T={T})'


# -- A.8: moving rigs --------------------------------------------------------------------

def test_a_moving_rig_is_refused_by_name(scorer_batch):
    """`_scene_scalars` and the rays index `cam['center']`/`cam['ext']` as static.

    Built from a real static group with per-frame extrinsics spliced in, so this is the same
    condition the query encoder uses to call a rig moving.
    """
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    moving = [dict(c) for c in cgroup]
    e = moving[0]['ext']
    moving[0]['ext'] = e[None].expand(coords.shape[1], *e.shape).contiguous()
    with pytest.raises(ValueError, match='MOVING rig'):
        with torch.no_grad():
            model(views, coords, moving, kpt_ids)


# -- hygiene -----------------------------------------------------------------------------

def test_no_stash_leaks_out_of_a_forward(scorer_batch):
    """`_kpt_ids`/`_query_ok`/`_box_prompt` must be cleared even on the success path.

    A leaked stash applies ONE window's ids and query-validity to the NEXT forward. At
    `batch_size` 1 that is a silent wrong answer, not a shape error -- which is exactly why
    `PoseTrackerEncoder._forward` clears them in a `finally`.
    """
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    with torch.no_grad():
        model(views, coords, cgroup, kpt_ids)
    assert model.query_encoder._kpt_ids is None
    assert model.query_encoder._query_ok is None
    assert model.query_encoder._box_prompt is None


def test_stashes_are_cleared_even_when_the_forward_raises(scorer_batch):
    """The `finally` must clear the stashes on the EXCEPTION path too.

    This is the failure the `finally` exists for: a leaked stash applies one window's keypoint ids
    to the next window's forward, which at `batch_size` 1 is a silent wrong answer rather than a
    shape error.
    """
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    ids = torch.arange(K)[None]

    def boom(mod, args, kwargs):
        raise RuntimeError('injected decoder failure')

    h = model.decoder.register_forward_pre_hook(boom, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match='injected'):
            with torch.no_grad():
                model(views, coords, cgroup, ids)
    finally:
        h.remove()
    assert model.query_encoder._kpt_ids is None, 'the id stash leaked out of a raising forward'
    assert model.query_encoder._query_ok is None
    assert model.query_encoder._box_prompt is None

    # and the model still works afterwards
    with torch.no_grad():
        scores, _ = model(views, coords, cgroup, ids)
    assert torch.isfinite(scores).all()


def test_kpt_chunk_is_exact(scorer_batch):
    """Chunking the per-point work must be numerically IDENTICAL to one pass.

    The scene scalars, the validity mask and the NaN fill are all computed over the full K before
    any slicing, which is what makes the slice exact -- so this is the assertion that the
    ordering of the two is right.
    """
    model, views, coords, cgroup, K, kpt_ids = scorer_batch
    with torch.no_grad():
        whole_s, whole_p = model(views, coords, cgroup, kpt_ids)
        chunk_s, chunk_p = model(views, coords, cgroup, kpt_ids, kpt_chunk=1)
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


def test_an_absent_gridresid_offset_is_still_a_refusal_not_a_default():
    """`build_scorer` must inherit `build_model`'s refusals rather than re-implement them.

    Built directly from a config with the key REMOVED -- `_scorer` re-adds it from `SMALL`, so
    routing through that helper would not actually test the absence.
    """
    cfg = {k: v for k, v in SMALL.items() if k != 'gridresid_offset'}
    with pytest.raises(KeyError, match='gridresid_offset'):
        build_scorer({**cfg, 'box_prompt': 'film'}, 4, **SCORER_KW)
