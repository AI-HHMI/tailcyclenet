"""Detector sampling: chunk shuffle, cohort and multi-root weights, collation."""
import numpy as np
import pytest
import torch

from tailcyclenet.detector.dataset import BoxDataset
from tailcyclenet.detector.sampling import ChunkShuffle, CohortSampler, box_collate
from tests.helpers import KPTS_3D, _rig, _write_frames


def test_chunk_shuffle_is_a_permutation_that_keeps_locality():
    """Every index exactly once, few videos at a time: `_reader` caches per worker, so a wider
    draw re-opens containers (486 ms/batch vs 40)."""
    chunk, mix, n_pools = 512, 4, 3
    n = chunk * mix * n_pools                     # exact multiple, so pools are position-aligned
    s = ChunkShuffle(n, chunk=chunk, mix=mix, seed=0)
    order = list(iter(s))
    assert sorted(order) == list(range(n)), 'must visit every index exactly once'
    assert order != list(range(n)), 'must actually shuffle'
    for i in range(0, n, chunk * mix):            # one pool = at most `mix` distinct videos
        assert len({j // chunk for j in order[i:i + chunk * mix]}) == mix
    assert list(iter(s)) != order, 'a second epoch must reshuffle'


def _two_cohort_root(tmp_path, n_annot_frames=2, n_tracked_frames=30):
    """A root with ONE `annotated` session and ONE `tracked` session, deliberately lopsided.

    The shape rat-city-combined actually has: a handful of annotated stills against one long
    tracked clip, so an unweighted draw and a weighted one cannot agree by accident.
    """
    from tailcyclenet import format as fmt

    W = H = 48
    root = tmp_path / 'two_cohort'
    for name, source, n_lab, T in (('a', 'annotated', n_annot_frames, n_annot_frames),
                                   ('t', 'tracked', n_tracked_frames, n_tracked_frames)):
        path = root / 'train' / name
        lab = fmt.empty_labels(1, T, len(KPTS_3D), 1, mode3d=False)
        frames = list(range(n_lab))
        lab.vis2d[0, frames, :, 0] = fmt.VISIBLE
        lab.points2d[0, frames, :, 0] = (
            10.0 + 3.0 * np.arange(len(frames), dtype=np.float32))[:, None, None] % (W - 10)
        fmt.write_session(path, mode='2d', units='px', label_source=source, names=KPTS_3D,
                          rig=_rig([('cam0', W, H, False, False, 0)]),
                          groups={'g0': fmt.Group('g0', T)}, labels={'g0': lab})
        _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return root


def _multi_detector_roots(tmp_path):
    """Two colliding roots; the larger one has four tracked sessions instead of one."""
    import shutil

    root_dir = tmp_path / 'roots'
    root_dir.mkdir()
    small = _two_cohort_root(tmp_path / 'small_src', n_annot_frames=2, n_tracked_frames=4)
    large = _two_cohort_root(tmp_path / 'large_src', n_annot_frames=2, n_tracked_frames=20)
    shutil.copytree(small, root_dir / 'alpha')
    shutil.copytree(large, root_dir / 'zeta')
    for i in range(3):
        shutil.copytree(root_dir / 'zeta' / 'train' / 't',
                        root_dir / 'zeta' / 'train' / f't_extra_{i}')
    return root_dir


def test_multi_root_weights_balance_roots_pool_and_validate_names(tmp_path):
    """Balanced: equal root shares with each root's cohort split intact; pooled: shares follow
    the combined population; `alpha` composes; an unknown `boxes_by_dataset` root is refused."""
    root = _multi_detector_roots(tmp_path)
    ds = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=0)
    weights = ds.train_weights(annot_frac=0.25, balance_datasets=True)
    assert ds.root_names == ['alpha', 'zeta']
    assert ds.root_mix(weights) == pytest.approx({'alpha': 0.5, 'zeta': 0.5})
    for cohorts in ds.root_cohort_mix(weights).values():
        assert cohorts == pytest.approx({'annotated': 0.25, 'tracked': 0.75})

    pooled = ds.train_weights(annot_frac=0.25, balance_datasets=False)
    assert pooled == pytest.approx(ds.default_train_weights(0.25, balance_datasets=False))
    assert ds.root_mix(pooled)['zeta'] > 0.65
    assert ds.cohort_mix(pooled) == pytest.approx({'annotated': 0.25, 'tracked': 0.75})

    alpha = ds.train_weights(annot_frac=None, alpha=0.5, balance_datasets=True)
    assert ds.root_mix(alpha) == pytest.approx({'alpha': 0.5, 'zeta': 0.5})
    single = BoxDataset(root / 'alpha', 'train', input_wh=(64, 64), max_frames_per_group=0)
    np.testing.assert_array_equal(single.train_weights(0.5, 0.5, True),
                                  single.default_train_weights(0.5) * single.alpha_weights(0.5))

    with pytest.raises(ValueError, match='unknown dataset roots'):
        BoxDataset(root, 'train', boxes_by_dataset={'typo': 'instances'})


def test_annot_frac_is_inert_on_a_single_cohort_split(dense_root):
    """The key must not affect an all-`tracked` split (3dpop/calms21/branson-fly) -- calms21 is
    the one train-time video root, where decode locality still costs."""
    ds = BoxDataset(dense_root, 'train', input_wh=(64, 64))
    assert len(ds.cohort_mix()) == 1, 'fixture must be single-cohort for this test to mean anything'
    base = ds.default_train_weights(None)
    for frac in (0.0, 0.25, 0.5, 1.0):
        w = ds.default_train_weights(frac)
        np.testing.assert_allclose(w, base, err_msg=f'annot_frac={frac} must be inert on a '
                                   'single-cohort split')


def test_annot_frac_sets_the_cohort_share_uniformly_within_a_cohort(tmp_path):
    """The mix becomes the configured number, and within a cohort a group's weight does not scale
    with its surviving frame count (the same accident one level down)."""
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    assert ds.cohort_mix()['annotated'] < 0.2, 'fixture must be lopsided or this proves nothing'
    for frac in (0.25, 0.75):
        mix = ds.cohort_mix(ds.default_train_weights(frac))
        assert mix['annotated'] == pytest.approx(frac)
        assert mix['tracked'] == pytest.approx(1.0 - frac)
    w = ds.default_train_weights(0.5)
    src = np.array([s.label_source for s, _, _, _ in ds.index])
    for cohort in ('annotated', 'tracked'):
        assert len(np.unique(w[src == cohort])) == 1
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match='annot_frac'):
            ds.default_train_weights(bad)

    # The weights are only a claim until the sampler is drawn from.
    s = CohortSampler(w, num_samples=20000, seed=0)
    first = list(iter(s))
    assert len(first) == len(s) and min(first) >= 0 and max(first) < len(ds)
    assert (src[np.array(first)] == 'annotated').mean() == pytest.approx(0.5, abs=0.02)
    assert list(iter(s)) != first, 'a second epoch must redraw'


def _two_group_root(tmp_path, n1=4, n2=32, label_source='tracked'):
    """Two SINGLE-COHORT sessions with different frame counts, isolated from `annot_frac`."""
    from tailcyclenet import format as fmt

    W = H = 48
    root = tmp_path / 'two_group'
    for name, T in (('a', n1), ('b', n2)):
        path = root / 'train' / name
        lab = fmt.empty_labels(1, T, len(KPTS_3D), 1, mode3d=False)
        lab.vis2d[0, :, :, 0] = fmt.VISIBLE
        lab.points2d[0, :, :, 0] = (
            10.0 + 3.0 * np.arange(T, dtype=np.float32))[:, None, None] % (W - 10)
        fmt.write_session(path, mode='2d', units='px', label_source=label_source, names=KPTS_3D,
                          rig=_rig([('cam0', W, H, False, False, 0)]),
                          groups={'g0': fmt.Group('g0', T)}, labels={'g0': lab})
        _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return root


def test_max_frames_per_group_zero_drops_the_cap(tmp_path):
    """B1a: 0 means every labelled frame is indexed; the default cap (40) is unchanged."""
    root = _two_group_root(tmp_path, n1=4, n2=32)
    capped = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=8, seed=0)
    uncapped = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=0, seed=0)
    assert len(capped) == 4 + 8       # group 'a' (4 frames) under the cap, 'b' (32) truncated to 8
    assert len(uncapped) == 4 + 32    # nothing truncated
    # 40 (the shipped default) must be UNCHANGED from the pre-B1a behaviour.
    default = BoxDataset(root, 'train', input_wh=(64, 64), seed=0)
    assert len(default) == 4 + 32     # both groups are under the default cap of 40


def test_alpha_weights_interpolate_frame_and_group_uniform(tmp_path):
    """alpha=None is no weighting, 1.0 frame-uniform, 0.0 equal group totals; on uniform group
    sizes (3dpop's shape) every alpha is a free no-op."""
    ds = BoxDataset(_two_group_root(tmp_path / 'a', n1=4, n2=32), 'train', input_wh=(64, 64),
                    max_frames_per_group=0)
    assert ds.alpha_weights(None) is None
    w = ds.alpha_weights(1.0)
    assert np.allclose(w, w[0])
    w = ds.alpha_weights(0.0)
    keys = np.array([f'{s.session_id}/{g}' for s, g, _, _ in ds.index])
    totals = [float(w[keys == k].sum()) for k in np.unique(keys)]
    assert totals[0] == pytest.approx(totals[1])

    even = BoxDataset(_two_group_root(tmp_path / 'b', n1=16, n2=16), 'train', input_wh=(64, 64),
                      max_frames_per_group=0)
    for alpha in (0.0, 0.5):
        assert len(np.unique(np.round(even.alpha_weights(alpha), 10))) == 1


def test_collate_pads_uneven_animal_counts():
    a = {'x': torch.zeros(3, 8, 8), 'boxes': torch.zeros(2, 4)}
    b = {'x': torch.zeros(3, 8, 8), 'boxes': torch.zeros(5, 4)}
    got = box_collate([a, b])
    assert got['x'].shape == (2, 3, 8, 8) and got['boxes'].shape == (2, 5, 4)
    assert torch.isnan(got['boxes'][0, 2:]).all()
