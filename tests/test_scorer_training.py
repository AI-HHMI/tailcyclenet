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


def test_the_scorer_warm_start_passes_the_checkpoints_own_registry():
    """The scorer's warm-start call must name the checkpoint's registry.

    Pins the call site: `base_reg` is the registry read from the SOURCE run folder, and it is what
    `warm_start` must be given. A regression here reinitialises every source row silently.
    """
    src = (REPO / 'tailcyclenet' / 'scorer' / 'train.py').read_text()
    assert 'base_names=tuple(registry.names)' not in src, \
        'the grown registry must not be passed as the identity table base'
    assert 'base_names=tuple(base_reg.names) if base_reg else None' in src
