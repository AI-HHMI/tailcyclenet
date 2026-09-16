"""Focused contracts for the scorer's distributed launch helpers."""
import torch

from tailcyclenet import distributed as dist_utils
from tailcyclenet.scorer import train as scorer_train
from tailcyclenet.dataset import StepSampler


def test_scorer_rank_sampler_is_replacement_and_rank_seeded():
    class D(torch.utils.data.Dataset):
        def __len__(self): return 17
        def __getitem__(self, index): return index
    cfg = {'data': {'num_workers': 0}}
    a, _ = scorer_train._loaders(D(), None, cfg, 23, world=2, rank=0, num_samples=5)
    b, _ = scorer_train._loaders(D(), None, cfg, 23, world=2, rank=1, num_samples=5)
    assert isinstance(a.sampler, StepSampler)
    # Both streams have a fixed local length and distinct rank-seeded draws.
    assert len(a) == len(b) == 5
    assert list(a.sampler) != list(b.sampler)


def test_scorer_absolute_rates_scale_only_with_world():
    cfg = {'learning_rate': 1e-4, 'kpt_lr': 5e-4, 'encoder_lr_scale': .1}
    got = dist_utils.scale_optimizer_cfg(cfg, 4)
    assert got['learning_rate'] == 2e-4
    assert got['kpt_lr'] == 1e-3
    assert got['encoder_lr_scale'] == cfg['encoder_lr_scale']


def test_single_process_eval_gather_is_identity():
    metrics = {'val/n_scored': 3.0, 'val/active_triplet_acc': .75}
    assert scorer_train._gather_eval(None, metrics) == metrics


def test_sequence_scorer_mode_is_weights_only_for_frame_request():
    # A legacy scorer checkpoint may seed the framewise heads, but cannot resume its optimizer.
    ckpt = {'kind': 'scorer', 'output_granularity': 'sequence',
            'config': {'scorer': {'output_granularity': 'sequence'}}}
    assert scorer_train.scorer_checkpoint_granularity(ckpt) == 'sequence'
    assert scorer_train.scorer_checkpoint_granularity(ckpt) != 'frame'
    # The run path's mode-mismatch branch intentionally chooses warm_start rather than resume.
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
