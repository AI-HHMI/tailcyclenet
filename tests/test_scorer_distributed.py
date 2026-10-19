"""Focused contracts for the scorer's distributed launch helpers."""
import torch
from pathlib import Path
import sys

from tailcyclenet import distributed as dist_utils
from tailcyclenet.scorer import dataset as scorer_dataset
from tailcyclenet.scorer import train as scorer_train
from tailcyclenet.dataset import LoaderConfig, PoseDataset, StepSampler, shard_sessions


class _D(torch.utils.data.Dataset):
    def __len__(self): return 17
    def __getitem__(self, index): return index


def test_scorer_loaders_are_rank_seeded_with_separate_val_workers():
    cfg = {'data': {'num_workers': 0}}
    a, _ = scorer_train._loaders(_D(), None, cfg, 23, world=2, rank=0, num_samples=5)
    b, _ = scorer_train._loaders(_D(), None, cfg, 23, world=2, rank=1, num_samples=5)
    assert isinstance(a.sampler, StepSampler)
    assert len(a) == len(b) == 5
    assert list(a.sampler) != list(b.sampler), 'ranks must draw distinct streams'
    assert not a.pin_memory
    one, val = scorer_train._loaders(
        _D(), _D(), {'data': {'num_workers': 0, 'val_num_workers': 0, 'worker_cv_threads': 0}},
        23, world=1, num_samples=1)
    assert one.pin_memory
    assert one.num_workers == val.num_workers == 0
    assert one.prefetch_factor is None and val.prefetch_factor is None

    cfg = {'data': {'num_workers': 2, 'val_num_workers': 1, 'prefetch_factor': 1,
                    'worker_cv_threads': 2}}
    train, val = scorer_train._loaders(_D(), _D(), cfg, 23, world=4, rank=0, num_samples=5)
    assert (train.num_workers, val.num_workers) == (2, 1)
    assert train.prefetch_factor == val.prefetch_factor == 1
    assert not train.pin_memory and not val.pin_memory
    assert train.persistent_workers and val.persistent_workers


def test_scorer_shape_draws_are_shared_by_ddp_ordinal():
    from types import SimpleNamespace

    wrapped = object.__new__(scorer_dataset.ScorerDataset)
    wrapped.base = SimpleNamespace(seed=23, train=True,
                                  cfg=LoaderConfig(cams_to_sample=[1, 8]))
    _, shape0, _ = wrapped._streams(17, ordinal=41)
    _, shape1, _ = wrapped._streams(93, ordinal=41)
    assert PoseDataset.shape(wrapped.base, shape0) == PoseDataset.shape(wrapped.base, shape1)


def test_scorer_absolute_rates_scale_only_with_world():
    cfg = {'learning_rate': 1e-4, 'kpt_lr': 5e-4, 'encoder_lr_scale': .1}
    got = dist_utils.scale_optimizer_cfg(cfg, 4)
    assert got['learning_rate'] == 2e-4
    assert got['kpt_lr'] == 1e-3
    assert got['encoder_lr_scale'] == cfg['encoder_lr_scale']


def test_sequence_scorer_mode_is_weights_only_for_frame_request():
    """A legacy scorer checkpoint may seed the framewise heads, but cannot resume its optimizer."""
    ckpt = {'kind': 'scorer', 'output_granularity': 'sequence',
            'config': {'scorer': {'output_granularity': 'sequence'}}}
    assert scorer_train.scorer_checkpoint_granularity(ckpt) == 'sequence'
    assert scorer_train.scorer_checkpoint_mode_mismatch('frame', ckpt)
    assert not scorer_train.scorer_checkpoint_mode_mismatch('sequence', ckpt)


def test_launch_requests_ddp_for_multiple_cpu_devices(monkeypatch):
    import lightning.fabric as fabric_module
    from types import SimpleNamespace

    seen = {}
    class FakeFabric:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.world_size = 4
            self.global_rank = 0
            self.is_global_zero = True
        def launch(self): pass
        def print(self, *_args, **_kwargs): pass

    monkeypatch.setattr(fabric_module, 'Fabric', FakeFabric)
    scorer_train.launch(SimpleNamespace(devices=4, device='cpu', precision='32-true',
                                        strategy=None))
    assert seen['accelerator'] == 'cpu'
    assert seen['devices'] == 4
    assert seen['strategy'] == 'ddp_find_unused_parameters_true'


def test_session_sharding_is_disjoint_and_an_empty_val_shard_is_allowed(tmp_path):
    sessions = list(range(11))
    shards = [shard_sessions(sessions, rank, 4) for rank in range(4)]
    assert sorted(x for shard in shards for x in shard) == sessions
    assert shard_sessions(sessions) == sessions

    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as C

    root = tmp_path / 'ds'
    for i in range(3):
        C._session_3d(root / 'val' / f'session_{i}')
    cfg = LoaderConfig(n_frames=4, image_size=64, val_cams_to_sample=2,
                       prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0)
    empty = PoseDataset(root, 'val', cfg, rank=3, world_size=4)
    assert len(empty) == 0
    assert empty.registry.n_keypoints > 0
