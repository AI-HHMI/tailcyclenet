"""The scorer trainer's silent-wrong-number sites: warm-start registry, val camera count, val RNG
determinism, and scoring val on the averaged (schedule-free) iterate a consumer actually loads.
"""
import sys
from pathlib import Path

import numpy as np

import pytest
import torch

from tailcyclenet import checkpoints as ck
from tailcyclenet.pose import checkpoint as pck
from tailcyclenet import config as cfgmod
from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.scorer.dataset import ScorerDataset

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def scorer_root(tmp_path_factory):
    """A root with the same 3D session in train/ and val/ (3 cameras each)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    root = tmp_path_factory.mktemp('scorer_train') / 'ds'
    for i in range(3):
        C._session_3d(root / 'train' / f'sess_train_{i}')
        C._session_3d(root / 'val' / f'sess_val_{i}')
    test_session = root / 'test' / 'must_not_be_loaded'
    C._session_3d(test_session)
    (test_session / 'calibration.toml').unlink()
    return root


_CORR = {'const_offset_prob': 1.0, 'frame_noise_prob': 0.0, 'gradual_drift_prob': 0.0,
         'sinusoid_prob': 0.0, 'point_drop_prob': 0.0, 'min_valid_frames': 2,
         'mag_3d': {'const_offset': 15.0}, 'mag_2d': {'const_offset': 10.0}}
_LOADER = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0)


def test_build_datasets_skips_test_and_scores_val_at_val_cams_to_sample(scorer_root):
    """Test metadata is never parsed, and val reads `val_cams_to_sample` (not the train range),
    checked on the BUILT items."""
    from tailcyclenet.scorer.train import build_datasets

    cfg = cfgmod.load_config(str(REPO / 'configs/scorer.toml'), base=cfgmod.SCORER_CONFIG)
    for fixed in (2, 3):
        cfg['data'] = {**cfg['data'], 'path': str(scorer_root), 'num_workers': 0,
                       'val_num_workers': 0, 'cams_to_sample': [1, 3],
                       'val_cams_to_sample': fixed, 'n_frames': 4, 'image_size': 64,
                       'prob_2d_only': 0.0, 'aug_prob': 0.0, 'per_image_aug_prob': 0.0,
                       'grayscale_prob': 0.0, 'crop_jitter': 0.0}
        train_ds, val_ds, _reg = build_datasets(cfg, None)
        assert {s.split for ds in train_ds.base.datasets for s in ds.sessions['train']} == {'train'}
        assert {s.split for ds in val_ds.base.datasets for s in ds.sessions['val']} == {'val'}
        assert {len(val_ds[i]['good'][0]) for i in range(len(val_ds))} == {fixed}


def test_val_retries_never_use_an_entropy_seeded_rng(scorer_root, monkeypatch):
    """A failed val build is SKIPPED, not swapped, and draws from seeded RNGs only (the metric
    names a fixed window); train retries on an entropy seed so workers do not replay each other."""
    seeds: list = []
    real = np.random.default_rng

    def spy(seed=None):
        seeds.append(seed)
        return real(seed)

    monkeypatch.setattr(np.random, 'default_rng', spy)

    def build(split: str):
        base = PoseDataset(scorer_root, split, _LOADER, train=(split == 'train'))
        original = base.select

        def select(idx, rng, shape=None):
            return None if idx == 0 else original(idx, rng, shape)

        monkeypatch.setattr(base, 'select', select)
        return ScorerDataset(base, _CORR)

    val = build('val')
    seeds.clear()
    assert val[0] is None, 'window 0 always fails, so val must skip it rather than swap'
    assert val[1] is not None
    assert seeds and all(s is not None for s in seeds), f'val used an entropy seed: {seeds}'

    train = build('train')
    seeds.clear()
    assert train[0] is not None
    assert any(s is None for s in seeds)


def test_a_val_item_leaves_the_trainers_torch_rng_untouched(scorer_root):
    """With `val_num_workers = 0` val items run in the trainer; their per-item seed must not leak."""
    val = ScorerDataset(PoseDataset(scorer_root, 'val', _LOADER, train=False), _CORR)
    torch.manual_seed(1234)
    expected = torch.rand(8)
    torch.manual_seed(1234)
    first = val[0]
    assert torch.equal(torch.rand(8), expected), 'the val item reseeded the global torch RNG'
    torch.manual_seed(999)
    second = val[0]
    assert first is not None and second is not None
    for key, value in first.items():
        if torch.is_tensor(value):
            assert torch.allclose(value, second[key], rtol=0, atol=0, equal_nan=True), key


# -- val is scored on the averaged iterate, and restores both modes --------------------------

class _FakeOptimizer:
    """Records the eval/train transitions `evaluate` makes (the schedule-free swap)."""

    def __init__(self):
        self.calls: list[str] = []

    def eval(self):
        self.calls.append('eval')

    def train(self):
        self.calls.append('train')


class _FakeModel:
    """The three model methods `evaluate` touches."""

    def __init__(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    def score_triplet(self, trip):
        return torch.zeros(1, 3), torch.ones(1, 3), torch.zeros(1, 3)


def _fake_loader():
    """One 'triplet' carrying only what `evaluate` reads."""
    return [{'fired': [torch.zeros(1)]}]


def _frame_trip(in_view):
    return {
        'active_mask': torch.tensor([[[True], [False]]]),
        'observed_mask': torch.ones((1, 2, 1), dtype=torch.bool),
        'in_view_mask': torch.tensor(in_view).reshape(1, 2, 1),
        'anchor_observed_mask': torch.ones((1, 2, 1), dtype=torch.bool),
        'far_mask': torch.tensor([[[True], [False]]]),
        'near_mask': torch.tensor([[[False], [True]]]),
        'ambiguous_mask': torch.zeros((1, 2, 1), dtype=torch.bool),
        'corruption_type_mask': torch.zeros((1, 2, 1, 4), dtype=torch.bool),
        'source_frame_weight': torch.ones((1, 2, 1)),
        'anchor_label': 1.0,
    }


def test_evaluate_frame_metrics_use_active_rows_and_do_not_mutate_masks(monkeypatch):
    """Frame val metrics read active rows + pointwise labels; the localization count must not
    erase out-of-view slots from `observed_mask` (which pinned out_of_view_fraction at 0)."""
    from tailcyclenet.scorer import evaluate as st
    from tailcyclenet.scorer.losses import FrameTripletScorerLoss

    class FrameModel(_FakeModel):
        def score_triplet(self, _trip):
            # Far frame: bad is low; near frame: bad stays clean. Anchor is clean.
            scores = torch.tensor([[[[1.0, -1.0, 1.0]], [[1.0, 0.5, 1.0]]]])
            labels = torch.tensor([[[[1.0, -1.0, 1.0]], [[1.0, -1.0, 1.0]]]])
            return scores, torch.ones_like(scores), labels

    monkeypatch.setattr(st, 'triplet_to_device', lambda value, device: value)
    trip = _frame_trip([True, True])
    trip['corruption_type_mask'][0, 0, 0, 0] = True
    model = FrameModel()
    loss = FrameTripletScorerLoss(score_reg_weight=0.0)
    out = st.evaluate(model, [trip], loss, 'cpu', 1, output_granularity='frame')
    assert out['val/active_triplet_acc'] == 1.0
    assert out['val/active_fraction'] == 0.5
    assert out['val/pointwise_auroc'] == 1.0
    assert out['val/localization'] == 1.0
    assert out['val/acc_const_offset'] == 1.0
    assert model.mode == 'train'

    trip = _frame_trip([True, False])
    out = st.evaluate(model, [trip], loss, 'cpu', 1, output_granularity='frame')
    assert out['val/out_of_view_fraction'] == 0.5
    assert bool(trip['observed_mask'].all())


# -- the averaged iterate is what val MEASURES, numerically -----------------------------------

class _TinyScorer(torch.nn.Module):
    """A model whose triplet scores are a function of ONE parameter, so the raw iterate and the
    schedule-free average are directly comparable."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.head = torch.nn.Linear(1, 3)

    def score_triplet(self, trip):
        s = self.w.reshape(1)
        scores = torch.cat([s + 1.0, s - 1.0, s * 0.0]).reshape(1, 3)
        return scores, torch.ones_like(scores), torch.ones_like(scores)


_SF_CFG = {'optimizer': 'schedulefree', 'learning_rate': 1e-2, 'kpt_lr': 1e-2,
           'encoder_lr_scale': 1.0, 'weight_decay': 0.0, 'warmup_steps': 0,
           'beta1': 0.9, 'beta2': 0.95}


def _in_train_mode(opt) -> bool:
    """AdamWScheduleFree records the mode per param group; every group must be back in train."""
    return all(g['train_mode'] for g in opt.param_groups)


def test_validation_measures_the_averaged_iterate_and_it_round_trips(tmp_path, monkeypatch):
    """The number `checkpoint_best` is chosen on must be the one a consumer can reproduce.

    Raw and averaged weights are made to disagree on purpose; the test then checks that
    `evaluate(..., optimizer)` measures the averaged prediction, that `save_checkpoint` +
    `model_state_eval` reproduces exactly that prediction, and that both normal completion and a
    raised error leave the model and the optimizer in TRAIN mode.
    """
    from tailcyclenet.scorer import evaluate as st
    from tailcyclenet.pose.optim import build_optimizer

    monkeypatch.setattr(st, 'triplet_to_device', lambda trip, device: trip)
    monkeypatch.setattr(st, '_per_type_accuracy', lambda scores, fired: {'acc': 0.0})
    seen: dict = {}

    def loss_fn(scores, *rest):
        seen['scores'] = scores.detach().clone()

    model = _TinyScorer()
    opt = build_optimizer(model, set(), _SF_CFG)
    opt.train()                                         # a fresh AdamWScheduleFree is in EVAL mode
    for _ in range(5):                                  # the average must LAG the iterate
        opt.zero_grad()
        ((model.w - 3.0) ** 2).sum().backward()
        opt.step()
    assert _in_train_mode(opt)

    st.evaluate(model, _fake_loader(), loss_fn, 'cpu', 1)
    raw = seen['scores']
    st.evaluate(model, _fake_loader(), loss_fn, 'cpu', 1, opt)
    averaged = seen['scores']
    assert not torch.allclose(raw, averaged), 'the averaged iterate must be a different model'
    assert _in_train_mode(opt), 'val must leave the optimizer in train mode'
    assert model.training, 'val must leave the model in train mode'

    run = tmp_path / 'run'
    ck.save_checkpoint(run, 5, model, opt, {'seed': 0}, name='last', kind='scorer')
    saved = torch.load(run / 'checkpoints' / 'checkpoint_last.pth', map_location='cpu',
                       weights_only=False)
    assert not torch.allclose(saved['model_state']['w'], saved['model_state_eval']['w']), \
        'the checkpoint must carry both iterates'
    reloaded = _TinyScorer()
    reloaded.load_state_dict(saved['model_state_eval'])
    st.evaluate(reloaded, _fake_loader(), loss_fn, 'cpu', 1)
    assert torch.allclose(seen['scores'], averaged, atol=1e-6), \
        'model_state_eval must reproduce the val measurement'
    assert reloaded.training

    def boom(*_a):
        raise RuntimeError('mid-val failure')

    with pytest.raises(RuntimeError, match='mid-val failure'):
        st.evaluate(model, _fake_loader(), boom, 'cpu', 1, opt)
    assert _in_train_mode(opt), 'a raised val must still restore the optimizer'
    assert model.training, 'a raised val must still restore the model'


def test_a_failing_optimizer_restore_still_puts_the_model_back_in_train_mode(monkeypatch):
    """The cleanup is nested: the model's restore must not depend on the optimizer's succeeding."""
    from tailcyclenet.scorer import evaluate as st

    class _RudeOptimizer(_FakeOptimizer):
        def train(self):
            self.calls.append('train')
            raise RuntimeError('the optimizer failed to restore')

    monkeypatch.setattr(st, 'triplet_to_device', lambda trip, device: trip)
    monkeypatch.setattr(st, '_per_type_accuracy', lambda scores, fired: {'acc': 0.0})
    model, opt = _FakeModel(), _RudeOptimizer()
    with pytest.raises(RuntimeError, match='failed to restore'):
        st.evaluate(model, _fake_loader(), lambda *a: None, 'cpu', 1, opt)
    assert opt.calls == ['eval', 'train']
    assert model.mode == 'train', 'the model must be restored even when the optimizer is not'


def test_an_all_rejected_epoch_raises_even_though_rejections_advance_the_step():
    """A skipped triplet still advances `step`, so the guard must count optimizer steps."""
    from tailcyclenet.scorer.train import require_progress

    with pytest.raises(RuntimeError, match='all samples were rejected'):
        require_progress(stepped_this_epoch=0, stepped_this_run=0, step=50, local_target=100)
    with pytest.raises(RuntimeError, match='all samples were rejected'):
        require_progress(stepped_this_epoch=0, stepped_this_run=0, step=100, local_target=100)
    require_progress(stepped_this_epoch=3, stepped_this_run=3, step=50, local_target=100)
    require_progress(stepped_this_epoch=0, stepped_this_run=7, step=100, local_target=100)


# -- the warm start must pass the SOURCE registry, not the grown one -------------------------

class _KptModel(torch.nn.Module):
    """A module whose only interesting tensor is the identity table."""

    def __init__(self, n: int, d: int = 4):
        super().__init__()
        self.query_encoder = torch.nn.Module()
        self.query_encoder.kpt_embed = torch.nn.Embedding(n, d)
        torch.nn.init.normal_(self.query_encoder.kpt_embed.weight, std=0.02)


def test_warm_start_names_is_the_source_registry_and_copies_only_its_rows(tmp_path):
    """The grown registry (source + appended) has the wrong length, so `warm_start` refuses it and
    leaves EVERY row fresh; the source registry copies its rows and leaves the appended ones."""
    from tailcyclenet.format import Registry
    from tailcyclenet.scorer.train import warm_start_names

    source = Registry(names=('a', 'b', 'c'), datasets=(('src', (0, 1, 2)),))
    grown = Registry(names=('a', 'b', 'c', 'd', 'e'),
                     datasets=(('src', (0, 1, 2)), ('tgt', (3, 4))))
    assert warm_start_names(source) == ('a', 'b', 'c')
    assert warm_start_names(None) is None, 'no source registry means no copy is attempted'

    name = 'query_encoder.kpt_embed.weight'
    model = _KptModel(3)
    ckpt = tmp_path / 'source.pth'
    torch.save({'model_state': {name: model.query_encoder.kpt_embed.weight.detach().clone()}},
               ckpt)
    wide = _KptModel(5)
    before = wide.query_encoder.kpt_embed.weight.detach().clone()
    fresh = pck.warm_start(wide, ckpt, verbose=False, base_names=warm_start_names(grown))
    assert torch.equal(wide.query_encoder.kpt_embed.weight.detach(), before)
    assert name in fresh, 'the whole table is left fresh'
    fresh = pck.warm_start(wide, ckpt, verbose=False, base_names=warm_start_names(source))
    got = wide.query_encoder.kpt_embed.weight.detach()
    assert torch.equal(got[:3], model.query_encoder.kpt_embed.weight.detach())
    assert torch.equal(got[3:], before[3:]), 'the appended rows must be untouched by the copy'
    assert name not in fresh


class _Routed(torch.nn.Module):
    """The two shapes the routing rule distinguishes: an identity table and a 2D matrix."""

    def __init__(self, n: int = 5, d: int = 4):
        super().__init__()
        self.query_encoder = torch.nn.Module()
        self.query_encoder.kpt_embed = torch.nn.Embedding(n, d)
        self.score_head = torch.nn.Linear(d, 1)
        self.scene_encoder = torch.nn.Module()          # a Muon-routable 2D matrix, so the
        self.scene_encoder.kv_proj = torch.nn.Linear(d, d)   # muon half is never empty


def _route_of(opt, param):
    """(<half>, group index, lr) for one parameter, or None.

    The Muon recipe returns a `DualOptimizer` (a Muon half and an AdamW-SF half); the
    schedule-free recipe returns the bare `AdamWScheduleFree`. Both are inspected here rather
    than assumed, because the question is which group actually holds the parameter.
    """
    subs = [(name, sub) for name, sub in (('muon', getattr(opt, 'opt_muon', None)),
                                          ('adamw', getattr(opt, 'opt_adam', None)))
            if sub is not None] or [('single', opt)]
    for which, sub in subs:
        for gi, group in enumerate(sub.param_groups):
            if any(p is param for p in group['params']):
                return which, gi, float(group['lr'])
    return None


def test_widening_the_identity_table_does_not_move_it_between_optimizer_groups():
    """A successful widening removes `kpt_embed.weight` from `fresh`; the routing must not care.

    It does not: the table is an `nn.Embedding`, so Muon excludes it however `fresh` is spelled,
    and the schedule-free path pins it by NAME (`query_encoder.kpt_`). A change here would
    silently retune the identity table the moment the warm start starts working.
    """
    from tailcyclenet.pose.optim import build_optimizer

    for kind in ('muon', 'schedulefree'):
        cfg = {'optimizer': kind, 'learning_rate': 1e-4, 'kpt_lr': 5e-4, 'encoder_lr_scale': 0.1,
               'weight_decay': 0.002, 'warmup_steps': 0, 'muon_schedulefree': True,
               'muon_momentum': 0.95, 'muon_warmup_steps': 0, 'muon_adjust_lr_fn': 'match_rms_adamw'}
        name = 'query_encoder.kpt_embed.weight'
        routes = []
        for fresh in ({name}, set()):
            model = _Routed()
            opt = build_optimizer(model, fresh, cfg)
            routes.append(_route_of(opt, model.query_encoder.kpt_embed.weight))
        assert routes[0] == routes[1], f'the table moved between groups: {routes}'
        assert routes[0] is not None, 'the identity table must be in an optimizer group'
        if kind == 'muon':
            assert routes[0][0] == 'adamw', (
                'each row of the identity table is a keypoint; Muon orthogonalises matrices and must '
                'not touch it')
        else:
            assert routes[0][2] == pytest.approx(5e-4), (
                'the schedule-free path pins the table to kpt_lr by NAME (query_encoder.kpt_)')
