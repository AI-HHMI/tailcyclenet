"""The triplet contract: what makes the good-vs-bad gap a track-quality signal.

Each fails silently: good/bad must share pixels and the drop mask, the crop must not depend on the
corruption, and the 2D anchor must be carried into view B exactly (`H_B @ inv(H_A)`).
"""
import numpy as np
import pytest
import torch

from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.scorer.corruption import build_corruptors_for, draw_shift, observed_frame_counts
from tailcyclenet.scorer.masks import displacement_masks
from tailcyclenet.scorer.triplet import make_triplet
from tailcyclenet.scorer.views import transfer_points_2d, view_affine_2d
from tests.helpers import _scorer, _session_2d, _session_3d

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

    root = tmp_path_factory.mktemp('triplet')
    _session_2d(root / 'ratlike' / 'train' / 'sess_a')
    _session_3d(root / 'mouselike' / 'train' / 'sess_c')
    return root


def _one(root, session_root, mode_2d, seed=0, **over):
    cfg = {**CFG.__dict__, **over}
    cfg = LoaderConfig(**{k: v for k, v in cfg.items() if not k.startswith('_')})
    ds = PoseDataset(root / session_root, 'train', cfg, train=True)
    rng = np.random.default_rng(seed)
    sel = ds.select(0, rng, ds.shape(np.random.default_rng(1)))
    assert sel is not None
    trip = make_triplet(ds, sel, rng, CORRUPTION, build_corruptors_for(CORRUPTION))
    return ds, trip


def _both(roots, seed=0, **over):
    """One 3D triplet and one 2D triplet from their respective roots."""
    return (_one(roots, 'mouselike', False, seed, **over),
            _one(roots, 'ratlike', True, seed, **over))


def test_prob_2d_only_builds_a_2d_scorer_triplet(roots):
    _ds, trip = _one(roots, 'mouselike', False, seed=0, prob_2d_only=1.0)
    assert trip is not None
    assert trip['mode'] == '2d'
    assert trip['good'][1].shape[-1] == 2
    assert len(trip['good'][0]) == 1


def test_good_and_bad_differ_only_in_coordinates(roots):
    """Same pixels, same drop mask, no keypoint dropped everywhere -- but the coordinates move."""
    moved = 0
    for seed in (0, 3):
        for _ds, trip in _both(roots, seed):
            assert trip is not None, 'the triplet did not build'
            gv, bv = trip['good'][0], trip['bad'][0]
            assert len(gv) == len(bv) > 0
            for i, (a, b) in enumerate(zip(gv, bv)):
                assert torch.equal(a, b), f'camera {i}: good and bad pixels differ'
            g, b = trip['good'][1], trip['bad'][1]
            obs = torch.isfinite(g).all(-1)
            assert torch.equal(obs, torch.isfinite(b).all(-1)), 'good/bad dropped different slots'
            assert bool(obs.any(dim=1).all()), 'some keypoint was dropped in every frame'
            if bool(obs.any()):
                moved += int(((b - g)[obs].abs().sum(-1) > 0).sum())
    assert moved > 0, 'the corruption moved nothing'


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


def test_the_crop_does_not_depend_on_the_corruption(roots):
    """A standalone realisation on the same rng prefix must reproduce view A byte for byte.

    `make_triplet` realises view A from the CLEAN selection before drawing the corruption.
    """
    ds = PoseDataset(roots / 'mouselike', 'train', CFG, train=True)
    shape = ds.shape(np.random.default_rng(1))

    rng_a = np.random.default_rng(0)
    sel_a = ds.select(0, rng_a, shape)
    standalone = ds.realise(sel_a, rng_a, world_gauge=False)

    torch.manual_seed(0)
    rng_b = np.random.default_rng(0)
    sel_b = ds.select(0, rng_b, shape)
    trip = make_triplet(ds, sel_b, rng_b, CORRUPTION, build_corruptors_for(CORRUPTION))
    assert standalone is not None, 'the standalone realisation failed'
    assert trip is not None, ('the triplet did not build for this seed. The corruption draws come '
                              'from the ambient torch RNG, so this test seeds it -- a build '
                              'failure here means the visibility filter rejected every keypoint')

    assert len(standalone.cgroup) == len(trip['good'][2])
    for a, b in zip(standalone.cgroup, trip['good'][2]):
        assert torch.equal(a['size'], b['size']), 'the crop SIZE moved'
        assert torch.allclose(a['mat'], b['mat']), 'the intrinsics moved'
    assert trip['good'][0][0].shape[0] == 1, 'the triplet must carry a batch axis'
    for i, (a, b) in enumerate(zip(standalone.views, trip['good'][0])):
        assert torch.equal(a[None], b), f'camera {i}: the good view is not the standalone view'


def test_the_2d_transfer_round_trips_and_preserves_nan(roots):
    """`H_B @ inv(H_A)` composes exactly, a dropped slot stays dropped, and the anchor keeps every
    observed slot of its source track."""
    ds, trip = _one(roots, 'ratlike', True, 0)
    rng = np.random.default_rng(0)
    sel = ds.select(0, rng, ds.shape(np.random.default_rng(1)))
    a = ds.realise(sel, np.random.default_rng(11), world_gauge=False)
    b = ds.realise(sel, np.random.default_rng(12), world_gauge=False)
    Ha, Hb = view_affine_2d(a), view_affine_2d(b)
    eye = torch.eye(3, dtype=torch.float64)

    src = torch.rand(10, 2, dtype=torch.float32) * 40 + 5
    xb = transfer_points_2d(transfer_points_2d(src, eye, Ha), Ha, Hb)
    direct = transfer_points_2d(src, eye, Hb)
    assert torch.allclose(xb, direct, atol=1e-3), (xb - direct).abs().max()

    out = transfer_points_2d(torch.tensor([[[1.0, 2.0], [float('nan'), float('nan')]]]), eye, eye)
    assert torch.isfinite(out[0, 0]).all() and not torch.isfinite(out[0, 1]).any()

    assert trip is not None and trip['mode'] == '2d'
    source = trip['good'][1] if trip['anchor_label'] > 0 else trip['bad'][1]
    anchor = trip['anchor'][1]
    assert source.shape == anchor.shape
    keep = torch.isfinite(source).all(-1)
    assert bool(keep.any()) and torch.isfinite(anchor).all(-1)[keep].all()


# -- sparse windows (section 3.8b) ------------------------------------------------------------

def test_the_sparse_menu_is_const_offset_forced_plus_frame_noise():
    """Sparse keypoints get only types that can move them, with `const_offset` FORCED (gated at
    0.5 it left ~83% of late-observed 3dpop keypoints uncorrupted); a zero forced magnitude raises."""
    _d3, _d2, s3, s2 = build_corruptors_for(CORRUPTION)
    assert set(s3.names) == set(s2.names) == {'const_offset', 'frame_noise'}
    cfg = dict(CORRUPTION, const_offset_prob=0.01, frame_noise_prob=0.0)
    assert build_corruptors_for(cfg)[2].probs['const_offset'] == 1.0
    cfg = dict(CORRUPTION, mag_3d={k: 0.0 for k in CORRUPTION['mag_3d']})
    with pytest.raises(ValueError, match='const_offset'):
        build_corruptors_for(cfg)


def test_sparse_windows_still_build_and_still_carry_signal(roots):
    """A fully sparse window (min_valid_frames above the fixture's span) trains on the reduced menu."""
    high = dict(CORRUPTION, min_valid_frames=6)
    ds = PoseDataset(roots / 'mouselike', 'train', CFG, train=True)
    rng = np.random.default_rng(0)
    trip = make_triplet(ds, ds.select(0, rng, ds.shape(np.random.default_rng(1))),
                        rng, high, build_corruptors_for(high))
    assert trip is not None, 'a fully sparse window was refused'
    assert trip['n_sparse'] == trip['good'][1].shape[2], 'not every point came out sparse'
    g, b = trip['good'][1], trip['bad'][1]
    observed = torch.isfinite(g).all(-1)
    if bool(observed.any()):
        delta = (b - g)[observed].abs().sum(-1)
        assert float(delta.max()) > 0, 'a sparse window drew a corruption that moved nothing'


def test_sparse_stuck_detects_a_corruption_that_moves_no_observed_slot():
    """A shift that moves only never-observed slots is a corruption in name only."""
    from tailcyclenet.scorer.corruption import _sparse_stuck

    coords = torch.zeros(1, 4, 2, 3)
    coords[0, 0, 1] = float('nan')
    is_dense = torch.tensor([[True, False]])
    shift = torch.zeros(1, 4, 2, 3)
    shift[0, 3, 0] = 5.0                     # moves only the DENSE point (keypoint 0)
    stuck = _sparse_stuck(shift, coords, is_dense)
    assert bool(stuck[0, 1]), 'a sparse keypoint moved nowhere was not flagged'
    assert not bool(stuck[0, 0]), 'the dense keypoint was flagged'

    shift[0, 1, 1] = 5.0                     # now it moves the sparse point at an observed frame
    assert not bool(_sparse_stuck(shift, coords, is_dense).any())


def test_the_forced_sparse_menu_moves_every_sparse_keypoint(roots):
    """Regression for the first 3dpop probe: with `const_offset` forced no sparse keypoint is stuck."""
    from tailcyclenet.scorer.corruption import _sparse_stuck, draw_sparse_moved

    ds = PoseDataset(roots / 'mouselike', 'train', CFG, train=True)
    _d3, _d2, sparse_3d, _s2 = build_corruptors_for(CORRUPTION)
    checked = 0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        sel = ds.select(0, rng, ds.shape(np.random.default_rng(1)))
        if sel is None:
            continue
        view = ds.realise(sel, rng, world_gauge=False)
        if view is None:
            continue
        coords = view.coords[None]
        counts = observed_frame_counts(coords, sel.frames)
        alive = counts[0] > 0
        if int(alive.sum()) < 2:
            continue
        c = coords[:, :, alive]
        is_dense = (counts[:, alive] >= 6)[0]
        if bool(is_dense.all()):
            is_dense = torch.zeros_like(is_dense)      # force the sparse path for the test
        shift, _fired = draw_sparse_moved(sparse_3d, c, is_dense)
        assert not bool(_sparse_stuck(shift, c, is_dense).any()), \
            f'seed {seed}: a sparse keypoint was left uncorrupted'
        checked += 1
    assert checked > 0, 'no window exercised the sparse path'


# -- the triplet feeds the model -------------------------------------------------------------

def test_the_triplet_scores_end_to_end(roots):
    """`score_triplet` on a real triplet: shapes, finiteness, per-type tags, and one backward."""
    from tracktail.datasets.scorer_corruption import GENERATORS

    ds, trip = _one(roots, 'mouselike', False, 0)
    assert trip['good'][0][0].dim() == 5, 'views must already carry the batch axis'
    K = trip['kpt_ids'].shape[1]
    assert trip['fired'].shape == (1, trip['good'][1].shape[2], len(GENERATORS))
    assert bool(trip['fired'].any()), 'no corruption was tagged on a real triplet'
    model = _scorer(max(int(trip['kpt_ids'].max()) + 1, 4), stride_length=4)
    scores, precision, labels = model.score_triplet(trip)
    assert scores.shape == (1, K, 3) and precision.shape == (1, K, 3)
    assert labels.shape == (1, K, 3)
    assert torch.isfinite(scores).all(), 'a non-finite score reached the loss'
    assert torch.equal(labels[0, :, 0], torch.ones(K))
    assert torch.equal(labels[0, :, 1], -torch.ones(K))

    from tracktail.model.losses_scorer import TripletScorerLoss

    from tailcyclenet.scorer.model import flatten_sequence_triplet

    loss = TripletScorerLoss()
    scores, precision, labels = flatten_sequence_triplet(scores, precision, labels)
    total = loss(scores, precision, labels)
    total.backward()
    assert torch.isfinite(total), 'the triplet loss is non-finite'
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.score_head.parameters())


# -- corruption tagging (Gate B needs a per-type breakdown) -----------------------------------

def test_tagged_draw_matches_the_library_corruptor_and_tags_every_point():
    """Same seed -> the same shift as `PointCorruptor` (the tags describe what the model saw), and
    every point carries at least one tag even when no type fires on its own."""
    from tracktail.datasets.scorer_corruption import GENERATORS, PointCorruptor

    from tailcyclenet.scorer.corruption import tagged_draw

    probs = {n: 0.5 for n in GENERATORS}
    mags = {n: 10.0 for n in GENERATORS}
    torch.manual_seed(1234)
    want = PointCorruptor(probs, mags)(7, 4, 3, torch.device('cpu'))
    torch.manual_seed(1234)
    got, fired = tagged_draw(PointCorruptor(probs, mags), 7, 4, 3, torch.device('cpu'))
    assert torch.equal(got, want), (got - want).abs().max()
    assert fired.shape == (7, len(GENERATORS))

    probs = {n: 0.0 for n in GENERATORS}
    probs['const_offset'] = 0.01
    torch.manual_seed(7)
    _shift, fired = tagged_draw(PointCorruptor(probs, {n: 5.0 for n in GENERATORS}),
                                6, 3, 2, torch.device('cpu'))
    assert bool(fired.any(dim=1).all()), 'a point was left with no corruption tag'


def test_frame_triplet_masks_are_post_gate_and_source_weighted(roots):
    """Frame mode exposes far/near masks, partial segments, and duplicate-safe weights."""
    cfg = dict(CORRUPTION, output_granularity='frame', min_valid_frames=0,
               const_offset_prob=1.0, frame_noise_prob=0.0,
               gradual_drift_prob=0.0, sinusoid_prob=0.0,
               min_corrupt_px=1.0, max_clean_px=0.1, segment_prob=1.0,
               segment_count=1, segment_len_frames=[2, 2], full_window_share=0.0,
               reference_gate='source_far')
    for root_name, mode in (('mouselike', '3d'), ('ratlike', '2d')):
        ds = PoseDataset(roots / root_name, 'train', CFG, train=True)
        rng = np.random.default_rng(0)
        sel = ds.select(0, rng, ds.shape(np.random.default_rng(1)))
        trip = make_triplet(ds, sel, rng, cfg, build_corruptors_for(cfg))
        assert trip is not None and trip['mode'] == mode
        shape = (1, trip['good'][1].shape[1], trip['good'][1].shape[2])
        for key in ('observed_mask', 'anchor_observed_mask', 'in_view_mask', 'moved_mask',
                    'far_mask', 'near_mask', 'ambiguous_mask', 'active_mask',
                    'source_frame_weight'):
            assert trip[key].shape == shape, key
        assert torch.equal(
            trip['active_mask'],
            trip['far_mask'] & trip['observed_mask'] & trip['in_view_mask']
            & trip['anchor_observed_mask'])
        assert torch.equal(
            trip['corruption_type_mask'],
            trip['fired_frame'] & trip['far_mask'][..., None])
        # The forced segment is shorter than T, so at least one valid keypoint has a clean run.
        assert bool((~trip['far_mask'] & trip['observed_mask']).any())
        frames = trip['frames'].cpu().tolist()
        for ki in range(shape[2]):
            for frame in set(frames):
                slots = [i for i, value in enumerate(frames) if value == frame]
                eligible = (trip['observed_mask'][0, slots, ki]
                            & trip['in_view_mask'][0, slots, ki]
                            & trip['anchor_observed_mask'][0, slots, ki])
                if bool(eligible.any()):
                    assert torch.isclose(trip['source_frame_weight'][0, slots, ki][eligible].sum(),
                                         torch.tensor(1.0))


def test_frame_corruption_gathers_one_shift_for_duplicate_source_frames():
    _dense_3d, dense_2d, _sparse_3d, _sparse_2d = build_corruptors_for(CORRUPTION)
    coords = torch.zeros((1, 4, 1, 2))
    frames = np.array([0, 1, 1, 2], dtype=np.int64)
    torch.manual_seed(5)
    shift, _fired = draw_shift(
        dense_2d, coords, frames=frames,
        cfg={'output_granularity': 'frame', 'segment_prob': 0.0})
    assert torch.equal(shift[:, 1], shift[:, 2])
    assert shift.shape == coords.shape


def test_distance_states_exclude_near_and_ambiguous_rows():
    camera = [{'size': (100, 100)}]
    good = torch.tensor([[[[50.0, 50.0], [30.0, 30.0]]]])
    anchor = good.clone()
    # K0 is near (0.2 px), K1 is ambiguous (0.75 px), and the final K0 row is far.
    bad = torch.tensor([[[[50.2, 50.0], [30.75, 30.0]]]])
    out = displacement_masks(good, bad, anchor, camera, '2d',
                             {'min_corrupt_px': 1.0, 'max_clean_px': 0.5})
    assert bool(out['near_mask'][0, 0, 0])
    assert bool(out['ambiguous_mask'][0, 0, 1])
    assert not bool(out['active_mask'].any())
    bad[0, 0, 0] = torch.tensor([52.0, 50.0])
    out = displacement_masks(good, bad, anchor, camera, '2d',
                             {'min_corrupt_px': 1.0, 'max_clean_px': 0.5})
    assert bool(out['far_mask'][0, 0, 0]) and bool(out['active_mask'][0, 0, 0])
