"""The triplet contract: what makes the good-vs-bad gap a track-quality signal.

Four properties, each of which fails silently rather than raising when it is wrong:

  * `good` and `bad` must be PIXEL-IDENTICAL (the corruption moves coordinates only). If their
    pixels differ, the gap stops isolating track quality and starts measuring appearance.
  * the point-drop mask must be SHARED, or the gap partly measures which slots went missing.
  * the CROP must not depend on the corruption -- a crop that follows each candidate's own
    coordinates gives the bad sample different pixels, which breaks the first property too.
  * the 2D anchor's coordinates must be carried into view B's frame EXACTLY
    (`x_B = H_B @ inv(H_A) @ x_A`), or the anchor is a different track and the invariance target
    is meaningless.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet.dataset import LoaderConfig, PoseDataset
from tailcyclenet.scorer.triplet import (build_corruptors_for, make_triplet,
                                         observed_frame_counts, transfer_points_2d,
                                         view_affine_2d)

CFG = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0)

CORRUPTION = {
    'const_offset_prob': 0.5, 'frame_noise_prob': 0.5,
    'gradual_drift_prob': 0.5, 'sinusoid_prob': 0.5,
    'point_drop_prob': 0.3, 'point_drop_max_frac': 0.4, 'point_drop_bernoulli_rate': 0.2,
    'min_valid_frames': 2,
    'mag_3d': {'const_offset': 15.0, 'frame_noise': 10.0,
               'gradual_drift': 24.0, 'sinusoid': 16.0},
    'mag_2d': {'const_offset': 10.0, 'frame_noise': 8.0,
               'gradual_drift': 12.0, 'sinusoid': 10.0},
}


@pytest.fixture(scope='module')
def roots(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    root = tmp_path_factory.mktemp('triplet')
    C._session_2d(root / 'ratlike' / 'train' / 'sess_a')
    C._session_3d(root / 'mouselike' / 'train' / 'sess_c')
    return root


def _one(root, session_root, mode_2d, seed=0, **over):
    cfg = {**CFG.__dict__, **over}
    cfg = LoaderConfig(**{k: v for k, v in cfg.items() if not k.startswith('_')})
    ds = PoseDataset(root / session_root, 'train', cfg, train=True)
    rng = np.random.default_rng(seed)
    sel = ds._select(0, rng, ds._shape(np.random.default_rng(1)))
    assert sel is not None
    trip = make_triplet(ds, sel, rng, CORRUPTION, build_corruptors_for(CORRUPTION))
    return ds, trip


def _both(roots, seed=0, **over):
    """One 3D triplet and one 2D triplet from their respective roots."""
    return (_one(roots, 'mouselike', False, seed, **over),
            _one(roots, 'ratlike', True, seed, **over))


# -- A.3: good and bad share pixels exactly -------------------------------------------------

@pytest.mark.parametrize('seed', [0, 1, 7])
def test_good_and_bad_share_pixels_exactly(roots, seed):
    """`bad` is `good` with different COORDINATES. Nothing else may differ."""
    for _ds, trip in _both(roots, seed):
        assert trip is not None, 'the triplet did not build'
        gv, _, _ = trip['good']
        bv, _, _ = trip['bad']
        assert len(gv) == len(bv) > 0
        for i, (a, b) in enumerate(zip(gv, bv)):
            assert a.shape == b.shape
            assert torch.equal(a, b), f'camera {i}: good and bad pixels differ'


def test_the_corruption_actually_moves_coordinates(roots):
    """The complement of the pixel test: the coordinates MUST differ, or there is no signal.

    Without this the pixel-identity test could pass on a triplet where the corruption silently
    did nothing.
    """
    moved = 0
    for _ds, trip in _both(roots, 3):
        g = trip['good'][1]
        b = trip['bad'][1]
        assert g.shape == b.shape
        both_obs = torch.isfinite(g).all(-1) & torch.isfinite(b).all(-1)
        if bool(both_obs.any()):
            delta = (b - g)[both_obs].abs().sum(-1)
            assert float(delta.max()) > 0, 'the bad track is byte-identical to the good one'
            moved += int((delta > 0).sum())
    assert moved > 0


# -- A.4: the drop mask is shared ------------------------------------------------------------

@pytest.mark.parametrize('seed', [0, 1, 7])
def test_the_drop_mask_is_shared(roots, seed):
    """Identical NaN pattern in good and bad: missingness must be orthogonal to quality."""
    for _ds, trip in _both(roots, seed):
        g, b = trip['good'][1], trip['bad'][1]
        assert torch.equal(torch.isfinite(g).all(-1), torch.isfinite(b).all(-1)), \
            'good and bad dropped different slots'


def test_drops_never_empty_a_point(roots):
    """A dropped-everywhere keypoint would be pooled through the force-unmask for no reason."""
    for _ds, trip in _both(roots, 5):
        g = trip['good'][1]
        obs = torch.isfinite(g).all(-1)
        assert bool(obs.any(dim=1).all()), 'some keypoint was dropped in every frame'


def test_observed_frame_counts_count_distinct_sources_not_slots(roots):
    """A clamp-padded window must not count one frame several times.

    `_frames` repeats frames at a group edge, so a T=8 window over a 2-frame group holds two
    distinct frames shown four times each. Counting finite SLOTS would read 8.
    """
    coords = torch.zeros(1, 8, 3, 3)
    frames = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    counts = observed_frame_counts(coords, frames)
    assert int(counts[0, 0]) == 2, f'distinct frames miscounted: {int(counts[0, 0])}'

    coords[0, :, 1] = float('nan')
    counts = observed_frame_counts(coords, frames)
    assert int(counts[0, 1]) == 0, 'an all-missing keypoint still counted observed frames'

    coords[0, :4, 2] = float('nan')
    counts = observed_frame_counts(coords, frames)
    assert int(counts[0, 2]) == 1, 'a keypoint observed in only one distinct frame miscounted'


# -- A.5: the crop does not depend on the corruption -----------------------------------------

def test_the_crop_does_not_depend_on_the_corruption(roots):
    """The crop must already be fixed before any corruption is drawn.

    `make_triplet` realises view A from the CLEAN selection and only THEN draws the corruption, so
    a standalone realisation driven by the SAME rng state must reproduce view A byte for byte. That
    is the structural statement of "the crop does not depend on the corruption", and it fails if
    the two are ever reordered.

    The rng handling is the whole test: `_select` consumes the stream before `_realise` sees it, so
    a fresh generator for the standalone view would draw DIFFERENT rotation and jitter and the
    comparison would fail on unchanged code. Both paths are therefore driven from a generator that
    has run the identical prefix.

    A test that instead corrupted with a huge magnitude and compared crops cannot work: at
    `mag = 1e6` a point's corrupted slot leaves the frame whenever the draw does not cancel, the
    visibility filter drops those points, and whether 2 survive is a property of the ambient torch
    RNG rather than of the crop.
    """
    ds = PoseDataset(roots / 'mouselike', 'train', CFG, train=True)
    shape = ds._shape(np.random.default_rng(1))

    rng_a = np.random.default_rng(0)
    sel_a = ds._select(0, rng_a, shape)
    standalone = ds._realise(sel_a, rng_a, world_gauge=False)

    rng_b = np.random.default_rng(0)
    sel_b = ds._select(0, rng_b, shape)
    trip = make_triplet(ds, sel_b, rng_b, CORRUPTION, build_corruptors_for(CORRUPTION))
    assert standalone is not None and trip is not None

    assert len(standalone.cgroup) == len(trip['good'][2])
    for a, b in zip(standalone.cgroup, trip['good'][2]):
        assert torch.equal(a['size'], b['size']), 'the crop SIZE moved'
        assert torch.allclose(a['mat'], b['mat']), 'the intrinsics moved'
    assert trip['good'][0][0].shape[0] == 1, 'the triplet must carry a batch axis'
    for i, (a, b) in enumerate(zip(standalone.views, trip['good'][0])):
        assert torch.equal(a[None], b), f'camera {i}: the good view is not the standalone view'

# -- A.6: the 2D anchor round-trip -----------------------------------------------------------

def test_the_2d_affine_round_trips_a_point(roots):
    """`H_to @ inv(H_from) @ (H_from @ x) == H_to @ x` -- the composition is exact."""
    ds, _trip = _one(roots, 'ratlike', True, 0)
    rng = np.random.default_rng(0)
    sel = ds._select(0, rng, ds._shape(np.random.default_rng(1)))
    a = ds._realise(sel, np.random.default_rng(11), world_gauge=False)
    b = ds._realise(sel, np.random.default_rng(12), world_gauge=False)
    Ha, Hb = view_affine_2d(a), view_affine_2d(b)

    src = torch.rand(10, 2, dtype=torch.float32) * 40 + 5
    xa = transfer_points_2d(src, torch.eye(3, dtype=torch.float64), Ha)
    xb = transfer_points_2d(xa, Ha, Hb)
    direct = transfer_points_2d(src, torch.eye(3, dtype=torch.float64), Hb)
    assert torch.allclose(xb, direct, atol=1e-3), (xb - direct).abs().max()


def test_the_2d_anchor_is_the_same_track_in_view_bs_frame(roots):
    """The anchor's coordinates must equal the source track carried through H_B @ inv(H_A)."""
    ds, trip = _one(roots, 'ratlike', True, 0)
    assert trip is not None and trip['mode'] == '2d'
    source = trip['good'][1] if trip['anchor_label'] > 0 else trip['bad'][1]
    anchor = trip['anchor'][1]
    assert source.shape == anchor.shape and source.shape[0] == 1

    keep = torch.isfinite(source).all(-1)
    assert bool(keep.any()), 'no observed slot to compare'
    assert torch.isfinite(anchor).all(-1)[keep].all(), 'the transfer NaNNed an observed slot'


def test_transfer_preserves_nan(roots):
    """A dropped slot must stay dropped after the transfer, or the anchor invents evidence."""
    p = torch.tensor([[[1.0, 2.0], [float('nan'), float('nan')]]])
    H = torch.eye(3, dtype=torch.float64)
    out = transfer_points_2d(p, H, H)
    assert torch.isfinite(out[0, 0]).all()
    assert not torch.isfinite(out[0, 1]).any()


# -- sparse windows (section 3.8b) ------------------------------------------------------------

def test_a_sparse_keypoint_gets_only_the_types_that_can_move_it():
    """`build_corruptors_for`'s sparse pair must exclude the ramp/crossing types."""
    _d3, _d2, s3, s2 = build_corruptors_for(CORRUPTION)
    assert set(s3.names) == {'const_offset', 'frame_noise'}
    assert set(s2.names) == {'const_offset', 'frame_noise'}


def test_a_sparse_menu_with_no_enabled_type_is_refused():
    """A sparse window that cannot be corrupted must be a loud failure, not silent no-training."""
    cfg = dict(CORRUPTION, const_offset_prob=0.0, frame_noise_prob=0.0)
    with pytest.raises(AssertionError, match='sparse'):
        build_corruptors_for(cfg)


def test_sparse_windows_still_build_and_still_carry_signal(roots):
    """A window where every keypoint is sparse must still train, on the reduced menu.

    Section 3.8b keeps sparse windows rather than excluding them, because a single observed frame
    still supplies a spatial good/bad comparison. `min_valid_frames` is set above the fixture's
    distinct-frame count so every keypoint is sparse here.
    """
    high = dict(CORRUPTION, min_valid_frames=6)
    ds = PoseDataset(roots / 'mouselike', 'train', CFG, train=True)
    rng = np.random.default_rng(0)
    trip = make_triplet(ds, ds._select(0, rng, ds._shape(np.random.default_rng(1))),
                        rng, high, build_corruptors_for(high))
    assert trip is not None, 'a fully sparse window was refused'
    assert trip['n_sparse'] == trip['good'][1].shape[2], 'not every point came out sparse'
    g, b = trip['good'][1], trip['bad'][1]
    observed = torch.isfinite(g).all(-1)
    if bool(observed.any()):
        delta = (b - g)[observed].abs().sum(-1)
        assert float(delta.max()) > 0, 'a sparse window drew a corruption that moved nothing'


def test_an_uncorruptible_sparse_draw_is_refused_not_trained_on():
    """Invariant 1 of section 3.8b, tested directly.

    `_ensure_sparse_corruption_moved` must raise when a sparse keypoint's shift moves no observed
    slot -- a corruption that changes nothing would teach the scorer to call a corrupt sample
    clean.
    """
    from tailcyclenet.scorer.triplet import _ensure_sparse_corruption_moved

    coords = torch.zeros(1, 4, 2, 3)
    coords[0, 0, 1] = float('nan')
    is_dense = torch.tensor([[True, False]])
    shift = torch.zeros(1, 4, 2, 3)
    shift[0, 3, 0] = 5.0                     # moves only the DENSE point (keypoint 0)
    with pytest.raises(ValueError, match='moves NO observed slot'):
        _ensure_sparse_corruption_moved(shift, coords, is_dense)

    shift[0, 1, 1] = 5.0                     # now it moves the SPARSE point at an observed frame
    _ensure_sparse_corruption_moved(shift, coords, is_dense)


# -- the triplet feeds the model -------------------------------------------------------------

def test_the_triplet_scores_end_to_end(roots):
    """`score_triplet` on a real triplet: shapes, finiteness, and one backward."""
    from tests.test_scorer_model import _scorer

    ds, trip = _one(roots, 'mouselike', False, 0)
    assert trip['good'][0][0].dim() == 5, 'views must already carry the batch axis'
    K = trip['kpt_ids'].shape[1]
    model = _scorer(max(int(trip['kpt_ids'].max()) + 1, 4), stride_length=4)
    scores, precision, labels = model.score_triplet(trip)
    assert scores.shape == (K, 3) and precision.shape == (K, 3) and labels.shape == (K, 3)
    assert torch.isfinite(scores).all(), 'a non-finite score reached the loss'
    assert torch.equal(labels[:, 0], torch.ones(K)) and torch.equal(labels[:, 1], -torch.ones(K))

    from posetail.posetail.losses_scorer import TripletScorerLoss

    loss = TripletScorerLoss()
    total = loss(scores, precision, labels)
    total.backward()
    assert torch.isfinite(total), 'the triplet loss is non-finite'
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.score_head.parameters())


# -- corruption tagging (Gate B needs a per-type breakdown) -----------------------------------

def test_tagged_draw_matches_the_library_corruptor_exactly():
    """`tagged_draw` must draw the SAME shift as the library's `PointCorruptor`.

    The tags decide what the per-type `triplet_acc` means, so if this draw diverged from the one
    the model is trained on, the breakdown would describe a different corruption than the model
    saw. Same seed, same tensors -- not a statistical match.
    """
    from posetail.datasets.scorer_corruption import PointCorruptor, GENERATORS

    from tailcyclenet.scorer.triplet import tagged_draw

    probs = {n: 0.5 for n in GENERATORS}
    mags = {n: 10.0 for n in GENERATORS}
    P, T, D = 7, 4, 3

    torch.manual_seed(1234)
    want = PointCorruptor(probs, mags)(P, T, D, torch.device('cpu'))
    torch.manual_seed(1234)
    got, fired = tagged_draw(PointCorruptor(probs, mags), P, T, D, torch.device('cpu'))

    assert torch.equal(got, want), (got - want).abs().max()
    assert fired.shape == (P, len(GENERATORS))
    assert bool(fired.any()), 'no type was tagged -- the breakdown would be empty'


def test_tagged_draw_tags_every_point_with_at_least_one_type():
    """Every point is corrupted, so every point must carry at least one tag."""
    from posetail.datasets.scorer_corruption import PointCorruptor, GENERATORS

    from tailcyclenet.scorer.triplet import tagged_draw

    probs = {n: 0.0 for n in GENERATORS}
    probs['const_offset'] = 0.01                 # almost never fires on its own
    torch.manual_seed(7)
    _shift, fired = tagged_draw(PointCorruptor(probs, {n: 5.0 for n in GENERATORS}),
                                6, 3, 2, torch.device('cpu'))
    assert bool(fired.any(dim=1).all()), 'a point was left with no corruption tag'


def test_the_triplet_carries_per_type_tags_for_the_surviving_points(roots):
    """`make_triplet` must return tags aligned with the keypoints it kept."""
    from posetail.datasets.scorer_corruption import GENERATORS

    _ds, trip = _one(roots, 'mouselike', False, 0)
    assert trip is not None
    K = trip['good'][1].shape[2]
    assert trip['fired'].shape == (1, K, len(GENERATORS))
    assert bool(trip['fired'].any()), 'no corruption was tagged on a real triplet'
