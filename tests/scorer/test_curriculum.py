"""The corruption curriculum: off is the old draw, on is one global multiplier per step."""
import math

import pytest
import torch

from tailcyclenet import config as cfgmod
from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.loader.sampling import StepSampler
from tailcyclenet.scorer import train as scorer_train
from tailcyclenet.scorer.config import validate_scorer_config
from tailcyclenet.scorer.contract import scorer_contract
from tailcyclenet.scorer.corruption import build_corruptors_for, draw_shift
from tailcyclenet.scorer.curriculum import Curriculum, parse_curriculum, scale_corruptors
from tailcyclenet.scorer.dataset import ScorerDataset
from tests.helpers import REPO, _session_2d, _session_3d

_CORR = {'const_offset_prob': 1.0, 'frame_noise_prob': 0.0, 'gradual_drift_prob': 0.0,
         'sinusoid_prob': 0.0, 'point_drop_prob': 0.0, 'min_valid_frames': 2,
         'mag_3d': {'const_offset': 15.0}, 'mag_2d': {'const_offset': 10.0}}
_LOADER = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0)
_LINEAR = {'schedule': 'linear', 'start_scale': 1.0, 'end_scale': 0.25, 'warmup_iters': 100,
           'ramp_iters': 1000}
_ALL = ('const_offset', 'frame_noise', 'gradual_drift', 'sinusoid')


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    """A one-root dataset with train and val 3D sessions."""
    r = tmp_path_factory.mktemp('curriculum') / 'ds'
    for i in range(2):
        _session_3d(r / 'train' / f'sess_train_{i}')
        _session_3d(r / 'val' / f'sess_val_{i}')
    return r


def _shipped(curriculum=None):
    cfg = cfgmod.load_config(str(REPO / 'configs/scorer.toml'), base=cfgmod.SCORER_CONFIG)
    if curriculum is not None:
        cfg['scorer']['corruption'] = {**cfg['scorer']['corruption'], 'curriculum': curriculum}
    return cfg


def test_absent_curriculum_is_the_identity_and_the_old_objects():
    cur = parse_curriculum({})
    assert cur == Curriculum() and not cur.enabled and cur.scale(10 ** 6) == 1.0
    quad = build_corruptors_for({**_CORR, 'frame_noise_prob': 0.5})
    assert scale_corruptors(quad, 1.0) is quad
    # the shipped config carries no curriculum, so its contract has no curriculum key
    assert 'corruption_curriculum' not in scorer_contract(_shipped())
    assert 'corruption_curriculum' in scorer_contract(_shipped(_LINEAR))


def test_default_off_items_are_byte_identical(root):
    """Val items are seeded per index, so the draw is reproducible: absent == explicit none ==
    val_scale 1.0, tensor for tensor; the curriculum key never reaches make_triplet's cfg."""
    base = PoseDataset(root, 'val', _LOADER, train=False)
    off = ScorerDataset(base, _CORR)
    assert 'curriculum' not in off.cfg and off.corruptors_at(off.corruption_scale()) is off.corruptors
    for variant in ({'schedule': 'none'}, {'schedule': 'none', 'val_scale': 1.0}):
        on = ScorerDataset(base, {**_CORR, 'curriculum': variant})
        assert on.cfg == off.cfg
        a, b = off[0], on[0]
        for key, value in a.items():
            if torch.is_tensor(value):
                assert torch.equal(torch.nan_to_num(value), torch.nan_to_num(b[key])), key
        assert a['corruption_scale'] == b['corruption_scale'] == 1.0


def test_val_scale_multiplies_the_item_shift(root):
    base = PoseDataset(root, 'val', _LOADER, train=False)
    full = ScorerDataset(base, _CORR)[0]
    half = ScorerDataset(base, {**_CORR, 'curriculum': {'val_scale': 0.5}})[0]
    assert half['corruption_scale'] == 0.5
    assert torch.allclose(half['shift'], 0.5 * full['shift'], atol=1e-5)


@pytest.mark.parametrize('name', _ALL)
def test_scale_reaches_every_magnitude_bearing_type(name):
    """Each generator alone: the same RNG draw at scale s is s x the unscaled draw, for the dense
    and the sparse corruptors, in 3D and 2D."""
    cfg = {f'{n}_prob': (1.0 if n == name else 0.0) for n in _ALL}
    cfg.update(const_offset_prob=1.0 if name == 'const_offset' else 0.0,
               mag_3d={n: 9.0 for n in _ALL}, mag_2d={n: 7.0 for n in _ALL})
    quad = build_corruptors_for(cfg)
    scaled = scale_corruptors(quad, 0.3)
    coords = torch.zeros(1, 8, 5, 3)
    for c, s in zip(quad, scaled):
        assert s.probs == c.probs and s.names == c.names
        assert all(math.isclose(s.mags[k], 0.3 * v) for k, v in c.mags.items())
        torch.manual_seed(7)
        a, _ = draw_shift(c, coords)
        torch.manual_seed(7)
        b, _ = draw_shift(s, coords)
        assert a.abs().max() > 0
        assert torch.allclose(b, 0.3 * a, atol=1e-5)


@pytest.mark.parametrize('schedule', ['linear', 'cosine'])
def test_scale_is_monotone_and_hits_its_ends(schedule):
    cur = parse_curriculum({'curriculum': {**_LINEAR, 'schedule': schedule}})
    xs = [cur.scale(i) for i in range(0, 1500, 7)]
    assert all(b <= a + 1e-12 for a, b in zip(xs, xs[1:]))
    assert cur.scale(0) == cur.scale(100) == 1.0 and cur.scale(1100) == cur.scale(10 ** 6) == 0.25
    assert math.isclose(cur.scale(600), 0.625)


def _train_ds(root, world=1):
    base = PoseDataset(root, 'train', _LOADER)
    return ScorerDataset(base, {**_CORR, 'curriculum': _LINEAR}, iterations_per_step=world)


def test_train_scale_follows_the_ordinal_and_is_refused_without_one(root):
    ds = _train_ds(root, world=4)
    assert ds.corruption_scale(150) == ds.curriculum.scale(600)
    assert ds[(150, 0)]['corruption_scale'] == ds.curriculum.scale(600)
    with pytest.raises(RuntimeError, match='ordinal'):
        ds[0]


def test_two_datasets_at_one_step_get_one_scale(tmp_path):
    """A folder of roots, one 3D and one 2D: items from DIFFERENT datasets (and dimensionalities)
    at the same step carry the same scale -- one global multiplier, no per-dataset schedule."""
    _session_3d(tmp_path / 'dsA' / 'train' / 'sess')
    _session_2d(tmp_path / 'dsB' / 'train' / 'sess')
    ds = ScorerDataset(PoseDataset(tmp_path, 'train', _LOADER),
                       {**_CORR, 'curriculum': _LINEAR})
    for step in (0, 500, 5000):
        seen = {}
        for draw in range(40):
            trip = ds[(step, draw % len(ds))]
            names = {ds.base.registry.names[i].split('-')[0]
                     for i in trip['kpt_ids'].reshape(-1).tolist()}
            assert len(names) == 1
            seen.setdefault((names.pop(), trip['mode']), set()).add(trip['corruption_scale'])
        assert {k[0] for k in seen} == {'dsA', 'dsB'}, seen
        assert {k[1] for k in seen} == {'2d', '3d'}, seen
        assert set().union(*seen.values()) == {ds.curriculum.scale(step)}


def test_resume_continues_the_ordinal_and_the_scale(root):
    """The curriculum sampler starts at the first local step, so a resume at iteration N sees the
    scales an uninterrupted run sees from N on (world 1 and world 2)."""
    for world in (1, 2):
        ds = _train_ds(root, world)
        cfg = {'data': {'num_workers': 0}}
        n_target = 2000
        full, _ = scorer_train._loaders(ds, None, cfg, 23, world=world,
                                        num_samples=-(-n_target // world))
        assert isinstance(full.sampler, StepSampler)
        whole = [ds.corruption_scale(o) for o, _ in full.sampler]
        start_iter = 640
        start = -(-start_iter // world)
        resumed, _ = scorer_train._loaders(ds, None, cfg, 23, world=world,
                                           num_samples=-(-n_target // world) - start, start=start)
        tail = [ds.corruption_scale(o) for o, _ in resumed.sampler]
        assert tail == whole[start:]
        assert tail[0] == ds.curriculum.scale(start * world)


def test_off_loaders_are_unchanged(root):
    ds = ScorerDataset(PoseDataset(root, 'train', _LOADER), _CORR)
    one, _ = scorer_train._loaders(ds, None, {'data': {'num_workers': 0}}, 23, world=1,
                                   num_samples=5)
    assert not isinstance(one.sampler, StepSampler)


@pytest.mark.parametrize('bad, match', [
    ({'schedule': 'linear', 'ramp_iters': 10, 'end_scal': 0.5}, 'unknown key'),
    ({'schedule': 'adaptive'}, 'adaptive'),
    ({'schedule': 'step'}, 'not one of'),
    ({'schedule': 'linear'}, 'ramp_iters'),
    ({'schedule': 'linear', 'ramp_iters': 10, 'end_scale': 0.0}, 'positive'),
    ({'schedule': 'linear', 'ramp_iters': 10, 'warmup_iters': -1}, 'warmup_iters'),
    ({'end_scale': 0.5}, 'schedule = "none"'),
])
def test_config_validation_refuses_bad_curricula(bad, match):
    with pytest.raises(SystemExit, match=match):
        validate_scorer_config(_shipped(bad))


def test_config_validation_accepts_a_curriculum():
    assert validate_scorer_config(_shipped(_LINEAR)) == 'frame'
