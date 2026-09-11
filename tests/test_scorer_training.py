"""The scorer trainer's four repair sites: what a run LOADS, what val it SCORES, and whether the
two are reproducible and comparable at all.

Each of these was a silent wrong number rather than a crash:

  * a warm start that passed the grown registry to `warm_start` refused the identity-table copy,
    so EVERY source row was reinitialised, not just the appended ones;
  * `val_cams_to_sample` had no reader on the scorer path, so val drew its camera count from the
    train range and averaged windows of different difficulty;
  * the val retry index came from an unseeded RNG, so which window a metric described depended on
    how many earlier windows happened to fail to build;
  * val (and therefore `checkpoint_best`) was scored on the RAW schedule-free iterate while
    `load_scorer_run` returns the AVERAGED one, so the selection number described weights no
    consumer would ever get.
"""
import sys
from pathlib import Path

import numpy as np

import pytest
import torch

from tailcyclenet import checkpoints as ck
from tailcyclenet.dataset import LoaderConfig, PoseDataset
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
    return root


# -- val is scored at a FIXED camera count ---------------------------------------------------

def test_val_uses_the_fixed_val_camera_count_not_the_train_range(scorer_root):
    """`val_cams_to_sample` must reach the val Dataset.

    The pose trainer replaces `cams_to_sample` for val; the scorer did not, so every val window
    drew its own count from `[1, 8]` and `val/triplet_acc` averaged windows of different
    difficulty. The count is read off the BUILT items, not off the config helper.
    """
    from tailcyclenet.scorer.train import build_datasets

    cfg = ck.load_config(str(REPO / 'configs/scorer.toml'), base=ck._SCORER_CONFIG)
    def val_counts_for(fixed: int) -> set[int]:
        cfg['data'] = {**cfg['data'], 'path': str(scorer_root), 'num_workers': 0,
                       'cams_to_sample': [1, 3], 'val_cams_to_sample': fixed,
                       'n_frames': 4, 'image_size': 64, 'prob_2d_only': 0.0,
                       'aug_prob': 0.0, 'per_image_aug_prob': 0.0, 'grayscale_prob': 0.0,
                       'crop_jitter': 0.0}
        _train, val_ds, _reg = build_datasets(cfg, None)
        assert val_ds is not None
        return {len(val_ds[i]['good'][0]) for i in range(len(val_ds))}

    assert val_counts_for(2) == {2}, 'val must use val_cams_to_sample, not cams_to_sample'
    assert val_counts_for(3) == {3}, 'the val count must actually follow the key'


# -- the val retry chain is a function of (seed, idx) ----------------------------------------

def test_val_retry_picks_the_same_window_every_time(scorer_root, monkeypatch):
    """A failed build on val must not send the metric to an entropy-chosen window.

    `_streams` keys val on `(seed, idx)` so a number is comparable across checkpoints; the retry
    index was drawn from `np.random.default_rng()` with no seed, which broke exactly that. The
    chain of window indices a failing item walks is recorded and compared between two fresh
    datasets.
    """
    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                       crop_jitter=0.0)
    corr = {'const_offset_prob': 1.0, 'frame_noise_prob': 0.0, 'gradual_drift_prob': 0.0,
            'sinusoid_prob': 0.0, 'point_drop_prob': 0.0, 'min_valid_frames': 2,
            'mag_3d': {'const_offset': 15.0}, 'mag_2d': {'const_offset': 10.0}}

    def chain():
        """The base indices tried while building window 0, with window 0 forced to fail."""
        base = PoseDataset(scorer_root, 'val', cfg)
        real = base._select
        tried: list[int] = []

        def select(idx, rng, shape=None):
            tried.append(idx)
            return None if idx == 0 else real(idx, rng, shape)

        monkeypatch.setattr(base, '_select', select)
        ds = ScorerDataset(base, corr)
        assert ds[0] is not None
        return tried

    first = chain()
    monkeypatch.undo()
    second = chain()
    assert first[0] == 0, 'the first attempt must be the requested window'
    assert len(first) > 1, 'the failing window must have been retried'
    assert first == second, f'val retry is not reproducible: {first} vs {second}'


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


def test_evaluate_scores_the_averaged_iterate_and_restores_train_mode(monkeypatch):
    """`load_scorer_run` hands a consumer `model_state_eval`, so val must score that iterate."""
    from tailcyclenet.scorer import train as st

    monkeypatch.setattr(st, 'triplet_to_device', lambda trip, device: trip)
    monkeypatch.setattr(st, '_per_type_accuracy', lambda scores, fired: {'acc': 0.5})
    opt, model = _FakeOptimizer(), _FakeModel()
    out = st.evaluate(model, _fake_loader(), lambda *a: None, 'cpu', 1, opt)
    assert opt.calls == ['eval', 'train']
    assert model.mode == 'train'
    assert out['val/n_scored'] == 1.0


def test_evaluate_restores_train_mode_when_the_pass_raises(monkeypatch):
    """An exception mid-val must not leave the run training an eval-mode model."""
    from tailcyclenet.scorer import train as st

    def boom(*a):
        raise RuntimeError('mid-val failure')

    monkeypatch.setattr(st, 'triplet_to_device', lambda trip, device: trip)
    opt, model = _FakeOptimizer(), _FakeModel()
    with pytest.raises(RuntimeError, match='mid-val failure'):
        st.evaluate(model, _fake_loader(), boom, 'cpu', 1, opt)
    assert opt.calls == ['eval', 'train'], 'the optimizer swap must be undone'
    assert model.mode == 'train'


# -- the warm start must pass the SOURCE registry, not the grown one -------------------------

class _KptModel(torch.nn.Module):
    """A module whose only interesting tensor is the identity table."""

    def __init__(self, n: int, d: int = 4):
        super().__init__()
        self.query_encoder = torch.nn.Module()
        self.query_encoder.kpt_embed = torch.nn.Embedding(n, d)
        torch.nn.init.normal_(self.query_encoder.kpt_embed.weight, std=0.02)


def test_grown_registry_is_refused_and_the_source_registry_copies(tmp_path):
    """The mechanism the scorer call site depends on.

    `warm_start`'s length check is the ONLY thing keeping a row on its own keypoint, so it must
    refuse a name list that is not the checkpoint table's registry -- and it must copy when it is.
    """
    source = _KptModel(3)
    ckpt = tmp_path / 'source.pth'
    torch.save({'model_state': {'query_encoder.kpt_embed.weight':
                                source.query_encoder.kpt_embed.weight.detach().clone()}}, ckpt)

    grown = _KptModel(5)
    before = grown.query_encoder.kpt_embed.weight.detach().clone()
    fresh = ck.warm_start(grown, ckpt, verbose=False,
                          base_names=('a', 'b', 'c', 'd', 'e'))      # the grown registry
    after = grown.query_encoder.kpt_embed.weight.detach().clone()
    assert torch.equal(before, after), 'a mismatched name list must not touch the table'
    assert 'query_encoder.kpt_embed.weight' in fresh, 'the whole table is left fresh'

    grown = _KptModel(5)
    before = grown.query_encoder.kpt_embed.weight.detach().clone()
    fresh = ck.warm_start(grown, ckpt, verbose=False, base_names=('a', 'b', 'c'))
    got = grown.query_encoder.kpt_embed.weight.detach()
    assert torch.equal(got[:3], source.query_encoder.kpt_embed.weight.detach()), \
        'the source rows must be copied exactly'
    assert torch.equal(got[3:], before[3:]), 'the appended rows must be untouched by the copy'
    assert 'query_encoder.kpt_embed.weight' not in fresh


def test_warm_start_names_returns_the_source_registry_not_the_grown_one():
    """The choice the scorer's warm-start call makes, and the consequence of getting it wrong.

    `warm_start` copies the identity table row-for-row only when the name list it is given has
    the table's length, so the grown registry (source + appended) refuses the copy and
    reinitialises EVERY row. The test drives the real helper and the real `warm_start`; it does
    not read the call site's source text.
    """
    from tailcyclenet.format import Registry
    from tailcyclenet.scorer.train import warm_start_names

    source = Registry(names=('a', 'b', 'c'), datasets=(('src', (0, 1, 2)),))
    grown = Registry(names=('a', 'b', 'c', 'd', 'e'),
                     datasets=(('src', (0, 1, 2)), ('tgt', (3, 4))))
    assert warm_start_names(source) == ('a', 'b', 'c')
    assert warm_start_names(None) is None, 'no source registry means no copy is attempted'

    model = _KptModel(3)
    ckpt = Path(__file__).parent / '_wkpt_tmp.pth'
    try:
        torch.save({'model_state': {'query_encoder.kpt_embed.weight':
                                    model.query_encoder.kpt_embed.weight.detach().clone()}}, ckpt)
        wide = _KptModel(5)
        before = wide.query_encoder.kpt_embed.weight.detach().clone()
        ck.warm_start(wide, ckpt, verbose=False, base_names=warm_start_names(grown))
        assert torch.equal(wide.query_encoder.kpt_embed.weight.detach(), before), \
            'the grown registry must leave the table alone (it is longer than the table)'
        ck.warm_start(wide, ckpt, verbose=False, base_names=warm_start_names(source))
        assert torch.equal(wide.query_encoder.kpt_embed.weight.detach()[:3],
                           model.query_encoder.kpt_embed.weight.detach()), \
            'the source registry must copy its rows'
    finally:
        ckpt.unlink(missing_ok=True)


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


@pytest.mark.parametrize('kind', ['muon', 'schedulefree'])
def test_widening_the_identity_table_does_not_move_it_between_optimizer_groups(kind):
    """A successful widening removes `kpt_embed.weight` from `fresh`; the routing must not care.

    It does not: the table is an `nn.Embedding`, so Muon excludes it however `fresh` is spelled,
    and the schedule-free path pins it by NAME (`query_encoder.kpt_`). A change here would
    silently retune the identity table the moment the warm start starts working.
    """
    from tailcyclenet.train import build_optimizer

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
    from tailcyclenet.scorer import train as st
    from tailcyclenet.train import build_optimizer

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


def test_val_retries_never_use_an_entropy_seeded_rng(scorer_root, monkeypatch):
    """Directly: on val every RNG is seeded; on train the retry keeps its entropy seed.

    The earlier assertion compares two chains, which would pass for ANY deterministic rule. This
    one records the seeds `np.random.default_rng` was actually called with while a val retry ran,
    and requires all of them to be seeded -- the property the fix restores.
    """
    from tailcyclenet.scorer.dataset import ScorerDataset

    cfg = LoaderConfig(n_frames=4, image_size=64, prob_2d_only=0.0, aug_prob=0.0,
                       crop_jitter=0.0)
    corr = {'const_offset_prob': 1.0, 'frame_noise_prob': 0.0, 'gradual_drift_prob': 0.0,
            'sinusoid_prob': 0.0, 'point_drop_prob': 0.0, 'min_valid_frames': 2,
            'mag_3d': {'const_offset': 15.0}, 'mag_2d': {'const_offset': 10.0}}
    seeds: list = []
    real = np.random.default_rng

    def spy(seed=None):
        seeds.append(seed)
        return real(seed)

    monkeypatch.setattr(np.random, 'default_rng', spy)

    def build(split: str):
        base = PoseDataset(scorer_root, split, cfg, train=(split == 'train'))
        original = base._select

        def select(idx, rng, shape=None):
            return None if idx == 0 else original(idx, rng, shape)

        monkeypatch.setattr(base, '_select', select)
        return ScorerDataset(base, corr)

    val = build('val')
    seeds.clear()
    assert val[0] is not None
    assert seeds, 'the val build must have drawn at least one RNG'
    assert all(s is not None for s in seeds), f'val used an entropy-seeded RNG: {seeds}'

    train = build('train')
    seeds.clear()
    assert train[0] is not None
    assert any(s is None for s in seeds), (
        'train must keep drawing its retry index from an entropy seed, or workers replay each '
        'other')


def test_a_failing_optimizer_restore_still_puts_the_model_back_in_train_mode(monkeypatch):
    """The cleanup is nested: the model's restore must not depend on the optimizer's succeeding."""
    from tailcyclenet.scorer import train as st

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
