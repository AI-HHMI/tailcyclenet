"""The training loop's guards: the branches around the model and the loader.

A non-finite loss is the one that matters: rare, only on data nobody can construct on demand, and
getting it wrong kills a 72-hour cluster job in its first two minutes.
"""
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import tailcyclenet.pose.evaluate as ev
import tailcyclenet.pose.train as tr


def _batch(mode='3d'):
    """The attributes `run_batch` touches, and nothing else."""
    R = 3 if mode == '3d' else 2
    return SimpleNamespace(
        views=[torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)],
        cgroup=[{'size': torch.tensor([4, 4])}],
        sample_info={'mode': mode},
        kpt_ids=torch.zeros(1, 2, dtype=torch.long),
        kpt_prior=torch.zeros(1, 2, R),
        prompt_t=torch.zeros(1, 2, dtype=torch.int32),
        coords=torch.zeros(1, 2, 2, R),
        vis=None, vis_2d=None, p2d=None)


class _History:
    def __init__(self, values):
        self.loss_history = defaultdict(list, values)


def test_non_finite_loss_becomes_a_skipped_step_and_other_errors_propagate():
    """tracktail RAISES on a poisoned sub-loss (it killed two sweep jobs); `run_batch` turns that
    into a graph-free NaN the loop skips. Any other ValueError is a real bug and must propagate.
    """
    model = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 3)}   # noqa: E731

    def poisoned(*a, **k):
        raise ValueError("sub-loss(es) went non-finite while attached to the autograd graph: "
                         "['coords_loss', 'smoothness_loss_3d']")

    loss, out = tr.run_batch(model, poisoned, _batch(), 'cpu')
    assert not torch.isfinite(loss) and loss.grad_fn is None and out is not None

    def broken(*a, **k):
        raise ValueError('shape mismatch: this is a real bug, not a bad step')

    with pytest.raises(ValueError, match='real bug'):
        tr.run_batch(model, broken, _batch(), 'cpu')


def test_run_batch_routes_2d_visibility_on_its_own_wire():
    """`batch.vis_2d` is tracktail's `vis_true_cams` in 3D but `PoseLoss`'s own `vis_2d_true` in
    2D, which `TotalLoss.forward` never sees."""
    model = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 2),   # noqa: E731
                             'vis_pred_2d': torch.zeros(1, 1, 2, 2)}
    seen = []

    def spy(model_, out, coords_true, vis_true, vis_true_cams, vis_2d_true=None, **kw):
        seen.append(dict(vis_true=vis_true, vis_true_cams=vis_true_cams,
                         vis_2d_true=vis_2d_true))
        return torch.tensor(0.0)

    batch_2d = _batch('2d')
    batch_2d.vis_2d, batch_2d.p2d = torch.ones(1, 2, 2, 1, 1), torch.zeros(1, 1, 2, 2, 2)
    tr.run_batch(model, spy, batch_2d, 'cpu')
    assert seen[-1]['vis_true'] is None and seen[-1]['vis_true_cams'] is None
    assert seen[-1]['vis_2d_true'] is not None

    batch_3d = _batch()
    batch_3d.vis_2d = torch.ones(1, 2, 2, 1, 1)
    tr.run_batch(model, spy, batch_3d, 'cpu')
    assert seen[-1]['vis_true_cams'] is not None and seen[-1]['vis_2d_true'] is None


def test_tune_smoothness_clamps_short_windows_and_divides_out_the_stride():
    """T = 2 degrades the order to a first difference (not disabled, not a raise) and restores it
    per batch; `torch.diff` is undivided, so the weight is re-derived per stride, idempotently.
    """
    from tracktail.model.losses import TotalLoss

    loss_fn = TotalLoss(smoothness_loss_3d_weight=0.5, smoothness_loss_2d_weight=0.5,
                        smoothness_loss_order=4)
    pred = torch.zeros(1, 2, 3, 3)
    tr._tune_smoothness(loss_fn, 2)
    assert loss_fn.smoothness_loss_3d.order == 1
    assert torch.isfinite(loss_fn.smoothness_loss_3d(pred, pred, torch.ones(1, 2, 3, 1),
                                                     time_dim=1))
    tr._tune_smoothness(loss_fn, 24)
    assert loss_fn.smoothness_loss_3d.order == 4, 'the clamp is per batch, not sticky'

    loss_fn = TotalLoss(smoothness_loss_3d_weight=0.5, smoothness_loss_2d_weight=0.5,
                        smoothness_loss_order=1)
    vis = torch.ones(1, 4, 3, 1)

    def excess(stride):
        tr._tune_smoothness(loss_fn, 4, stride)
        t = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1) * stride
        pred = t.expand(1, 4, 3, 3).contiguous()           # a ramp at this stride's rate
        return float(loss_fn.smoothness_loss_3d(pred, torch.zeros_like(pred), vis, time_dim=1))

    base = excess(1)
    assert base > 0, 'the probe must actually reach the hinge'
    assert abs(excess(4) - base) < 1e-6
    tr._tune_smoothness(loss_fn, 4, 1)
    assert loss_fn.smoothness_loss_3d.weight == 0.5


def test_smoothness_drain_and_logging():
    """Finite components reach W&B, zero is real data, NaN/inf are dropped; a drain empties the
    history IN PLACE (PoseLoss appends lazily to the same defaultdict), so a skipped batch never
    leaks into the next interval."""
    class FakeWandb:
        calls = []

        def log(self, values, step):
            self.calls.append((values, step))

    history = _History({'smoothness_3d_loss': [1.0, 3.0, float('nan'), float('inf')],
                        'smoothness_2d_loss': [0.0, float('nan')]})
    mapping_id = id(history.loss_history)
    values = ev.drain_smoothness(history)
    assert values == {'smoothness_3d_loss': (4.0, 2), 'smoothness_2d_loss': (0.0, 1)}
    assert history.loss_history == {'smoothness_3d_loss': [], 'smoothness_2d_loss': []}
    history.loss_history['vis_loss_2d'].append(3.0)
    assert id(history.loss_history) == mapping_id

    wb = FakeWandb()
    tr.log(wb, {'train/loss': 2.0, **ev.smoothness_log_values(values)}, step=20)
    assert wb.calls == [({'train/loss': 2.0, 'train/smoothness_3d_loss': 2.0,
                          'train/smoothness_2d_loss': 0.0}, 20)]
    nxt = ev.drain_smoothness(_History({'smoothness_3d_loss': [4.0],
                                         'smoothness_2d_loss': [float('nan')]}))
    assert ev.smoothness_log_values(nxt) == {'train/smoothness_3d_loss': 4.0}


def test_train_metrics_match_validation_names_drop_nonfinite_and_keep_zero(monkeypatch):
    """A training forward exposes the scalar val metrics; per-mode means keep a real 0.0."""
    batch = _batch()
    out = {'coords_pred': torch.zeros_like(batch.coords),
           'vis_pred': torch.zeros(*batch.coords.shape[:-1], 1)}
    values = ev.train_metrics(out, batch)
    assert set(values) == set(ev.TRAIN_METRIC_KEYS)
    assert all(isinstance(v, float) and v == v for v in values.values())
    assert ev.train_metrics({'coords_pred': out['coords_pred']}, batch) == {}

    from tracktail.model import eval_metrics
    monkeypatch.setattr(eval_metrics, 'get_eval_metrics',
                        lambda **kwargs: {'mpjpe': 0.0, 'mte': float('nan'),
                                          'jaccard_1': float('inf')})
    assert ev.train_metrics(out, batch) == {'mpjpe': 0.0}

    stats = {mode: {key: [0.0, 0] for key in ev.TRAIN_METRIC_KEYS}
             for mode in ev.TRAIN_METRIC_MODES}
    stats['3d']['mpjpe'] = [0.0, 2]
    stats['2d']['mpjpe'] = [6.0, 2]
    logged = ev.metric_log_values(stats)
    assert logged['train/3d/mpjpe'] == 0.0 and logged['train/2d/mpjpe'] == 3.0
    assert 'train/3d/mte' not in logged


def test_metric_reductions_are_one_fixed_shape_collective_on_every_rank():
    """A rank with no local samples must still join the same-shaped all_reduce, or DDP deadlocks;
    the smoothness value is count-weighted across ranks."""
    class FakeFabric:
        world_size, device = 2, 'cpu'

        def __init__(self, result=None):
            self.calls, self.numel, self.result = 0, None, result

        def all_reduce(self, tensor, reduce_op):
            assert reduce_op == 'sum'
            self.calls += 1
            self.numel = tensor.numel()
            if self.result is None:
                return tensor
            out = torch.zeros_like(tensor)
            out[:4] = torch.tensor(self.result, dtype=tensor.dtype)
            return out

    full = {mode: {key: [0.0, 0] for key in ev.TRAIN_METRIC_KEYS}
            for mode in ev.TRAIN_METRIC_MODES}
    full['3d']['mpjpe'] = [1.0, 1]
    f_full, f_empty = FakeFabric(), FakeFabric()
    ev.metric_log_values(full, f_full)
    ev.metric_log_values({mode: {} for mode in ev.TRAIN_METRIC_MODES}, f_empty)
    expected = 2 * len(ev.TRAIN_METRIC_KEYS) * len(ev.TRAIN_METRIC_MODES)
    assert f_full.calls == f_empty.calls == 1 and f_full.numel == f_empty.numel == expected

    empty = {'smoothness_3d_loss': (0.0, 0), 'smoothness_2d_loss': (0.0, 0)}
    fabric = FakeFabric((10.0, 4.0, 6.0, 2.0))
    assert ev.smoothness_log_values(empty, fabric) == {
        'train/smoothness_3d_loss': 2.5, 'train/smoothness_2d_loss': 3.0}
    assert fabric.calls == 1
    assert ev.smoothness_log_values(empty) == {}


def test_a_checkpoint_round_trips_enough_to_resume_from(tmp_path):
    """`optimizer_state` was written and read by nothing, so a relaunch restarted at 0. The
    schedule-free `z` survives, and a resume takes the RAW weights, not the averaged eval ones.
    """
    from schedulefree import AdamWScheduleFree

    from tailcyclenet.checkpoints import save_checkpoint, stamped_config
    from tailcyclenet.format import Registry

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
    opt.train()
    for _ in range(5):          # at step 1 the averaged and raw iterates coincide
        opt.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
    config = {'model': {}, 'data': {'image_size': 64}, 'training': {'seed': 23}}
    registry = Registry(names=('nose',), datasets=(('ds', (0,)),))
    save_checkpoint(tmp_path / 'run', 1234, model, opt, config, registry=registry)

    ck = torch.load(tmp_path / 'run' / 'checkpoints' / 'checkpoint_last.pth',
                    map_location='cpu', weights_only=False)
    assert int(ck['iteration']) == 1234
    assert ck['config'] == stamped_config(config, 'pose')
    assert ck['keypoint_registry'] == registry.to_dict()
    assert not torch.equal(ck['model_state']['weight'], ck['model_state_eval']['weight'])

    fresh = torch.nn.Linear(4, 4)
    fresh_opt = AdamWScheduleFree(fresh.parameters(), lr=1e-3)
    fresh.load_state_dict(ck['model_state'])
    fresh_opt.load_state_dict(ck['optimizer_state'])
    assert torch.equal(fresh.weight, ck['model_state']['weight'])
    assert any('z' in s for s in fresh_opt.state.values()), 'the schedule-free z is lost'


# -- config guards -----------------------------------------------------------------------------

def test_load_config_layers_over_its_family_base(tmp_path):
    """A pose config gets `configs/base.toml` automatically; an explicit base (the detector's)
    replaces it, inheriting none of the pose blocks."""
    from tailcyclenet.config import load_config

    overlay = tmp_path / 'overlay.toml'
    overlay.write_text('[data]\npath = "/some/root"\n')
    cfg = load_config(overlay)
    assert cfg['data']['path'] == '/some/root'
    assert cfg['model']['query'] == 'prior'
    assert cfg['training']['optimizer']['optimizer'] == 'muon'
    assert cfg['data']['grayscale_prob'] == 0.2

    base = tmp_path / 'det_base.toml'
    base.write_text('[data]\nboxes = "instances"\n')
    p = tmp_path / 'det.toml'
    p.write_text('[training]\niters = 4\n')
    assert load_config(p, base=base) == {'data': {'boxes': 'instances'}, 'training': {'iters': 4}}


def test_load_config_extends_is_deleted_and_raises(tmp_path):
    from tailcyclenet.config import load_config

    p = tmp_path / 'old.toml'
    p.write_text('extends = "base.toml"\n[data]\npath = "/x"\n')
    with pytest.raises(SystemExit, match='extends'):
        load_config(p)


def test_det_score_has_one_default():
    """`detect_raw` and the CLI must share one `--det-score` default (two meant two detectors)."""
    import inspect

    from tailcyclenet.infer.cli import build_parser
    from tailcyclenet.infer.detect import detect_raw

    sig = inspect.signature(detect_raw).parameters['score_thresh'].default
    assert sig == build_parser().get_default('det_score')
