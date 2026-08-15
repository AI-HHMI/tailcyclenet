"""The training loop's guards. Not the model, not the loader -- the two branches around them.

A non-finite loss is the one that matters: it is rare, it only fires on data nobody can construct
on demand, and getting it wrong costs a 72-hour cluster job that dies in the first two minutes.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _train_module():
    """Import scripts/train.py without running main()."""
    spec = importlib.util.spec_from_file_location('tcn_train', REPO / 'scripts' / 'train.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch():
    """The attributes `run_batch` touches, and nothing else."""
    return SimpleNamespace(
        views=[torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)],
        cgroup=[{'size': torch.tensor([4, 4])}],
        sample_info={'mode': '3d'},
        kpt_ids=torch.zeros(1, 2, dtype=torch.long),
        kpt_prior=torch.zeros(1, 2, 3),
        prompt_t=torch.zeros(1, 2, dtype=torch.int32),
        coords=torch.zeros(1, 2, 2, 3),
        vis=None, vis_2d=None, p2d=None)


def test_non_finite_loss_becomes_a_skipped_step():
    """posetail RAISES on a poisoned sub-loss; the loop needs a NaN it can skip.

    `TotalLoss.forward` raising (`losses.py:873`) is right for its own purpose -- a NaN term
    silently dropped from the total still returns NaN gradients -- but it killed two 60k-iteration
    sweep jobs in their first 100 seconds over what is an intermittent bad step. `run_batch`
    converts it into the signal the loop's own `torch.isfinite(loss)` guard already reads.
    """
    tr = _train_module()
    model = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 3)}   # noqa: E731

    def poisoned(*a, **k):
        raise ValueError("sub-loss(es) went non-finite while attached to the autograd graph: "
                         "['coords_loss', 'smoothness_loss_3d']")

    loss, out = tr.run_batch(model, poisoned, _batch(), 'cpu')
    assert not torch.isfinite(loss), 'a non-finite sub-loss must come back as NaN, not raise'
    assert loss.grad_fn is None, 'the skipped step must carry no graph into the next iteration'
    assert out is not None


def test_other_value_errors_still_propagate():
    """Only the non-finite case is converted. A real bug must not be swallowed as a skipped step:
    a run that silently skips every iteration looks exactly like a run that trained."""
    tr = _train_module()
    model = lambda *a, **k: {'coords_pred': torch.zeros(1, 2, 2, 3)}   # noqa: E731

    def broken(*a, **k):
        raise ValueError('shape mismatch: this is a real bug, not a bad step')

    with pytest.raises(ValueError, match='real bug'):
        tr.run_batch(model, broken, _batch(), 'cpu')


def test_short_window_survives_the_smoothness_loss():
    """T = 2 must not raise inside `SmoothnessLoss`, and must not be silently disabled either.

    `SmoothnessLoss.forward` narrows by `T - k` (`posetail/losses.py:1146`), so a window shorter
    than `order + 1` frames raises on a negative length -- and `torch.diff(n=k)` is undefined
    there anyway. Both weights are 0.5, so the `weight == 0` early return does not cover it. Now
    that the loader sizes T to the labelled span, T = 2 is the COMMON case on annotated sessions:
    without the clamp every one of those steps would die.
    """
    from posetail.posetail.losses import TotalLoss

    mod = _train_module()
    loss_fn = TotalLoss(smoothness_loss_3d_weight=0.5, smoothness_loss_2d_weight=0.5,
                        smoothness_loss_order=4)
    assert loss_fn.smoothness_loss_3d.order == 4

    # The real call raises before the clamp is applied...
    pred = torch.zeros(1, 2, 3, 3)
    with pytest.raises(RuntimeError):
        loss_fn.smoothness_loss_3d(pred, pred, torch.ones(1, 2, 3, 1), time_dim=1)

    # ...and does not after it. Degraded to a first difference, not disabled.
    mod._tune_smoothness(loss_fn, 2)
    assert loss_fn.smoothness_loss_3d.order == 1
    assert torch.isfinite(loss_fn.smoothness_loss_3d(pred, pred, torch.ones(1, 2, 3, 1),
                                                     time_dim=1))

    # A full-length window gets the configured order back -- the clamp is per batch, not sticky.
    mod._tune_smoothness(loss_fn, 24)
    assert loss_fn.smoothness_loss_3d.order == 4


def test_stride_is_divided_out_of_the_smoothness_weight():
    """`torch.diff` is UNDIVIDED, so a stride-s window's k-th difference is s^k times a stride-1
    one and the term's effective weight against every other loss would ride on a per-item draw.

    Checked against the loss's own output rather than the attribute, because that is the thing
    that has to be stride-invariant. A pure ramp is the right probe: its 1st difference is exactly
    `s` per step, so an unnormalised term reads s times larger at stride s.
    """
    from posetail.posetail.losses import TotalLoss

    mod = _train_module()
    loss_fn = TotalLoss(smoothness_loss_3d_weight=0.5, smoothness_loss_2d_weight=0.5,
                        smoothness_loss_order=1)
    vis = torch.ones(1, 4, 3, 1)

    def excess(stride):
        mod._tune_smoothness(loss_fn, 4, stride)
        t = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1) * stride
        pred = t.expand(1, 4, 3, 3).contiguous()           # a ramp at this stride's rate
        return float(loss_fn.smoothness_loss_3d(pred, torch.zeros_like(pred), vis, time_dim=1))

    base = excess(1)
    assert base > 0, 'the probe must actually reach the hinge, or the test proves nothing'
    for s in (2, 4):
        assert abs(excess(s) - base) < 1e-6, f'stride {s} changed the term by {excess(s) / base:.1f}x'

    # And it is idempotent -- re-derived from the configured weight each call, never compounded.
    mod._tune_smoothness(loss_fn, 4, 1)
    assert loss_fn.smoothness_loss_3d.weight == 0.5


def test_a_checkpoint_round_trips_enough_to_resume_from(tmp_path):
    """`optimizer_state` was written from day one and read by NOTHING.

    So a relaunch into the same --out began again at iteration 0 and overwrote both ~5.6 GB files
    -- `checkpoint_last.pth` at the first boundary, and `checkpoint_best.pth` as soon as any val
    beat `saved_mpjpe = inf`. On a preemptible job that is the whole run.

    Pins the part that can be silently wrong: that the schedule-free optimizer's own state (the
    `z` iterate) survives the round trip, and that resuming takes the RAW weights rather than the
    averaged eval ones. The `it = start_it` arithmetic is not what breaks.
    """
    from schedulefree import AdamWScheduleFree

    from tailcyclenet.checkpoints import save_checkpoint

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = AdamWScheduleFree(model.parameters(), lr=1e-3)
    opt.train()
    # SEVERAL steps: schedule-free's averaged iterate coincides with the raw one at step 1, so a
    # single step cannot tell "resume from the raw weights" from "resume from the eval weights".
    for _ in range(5):
        opt.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
    save_checkpoint(tmp_path / 'run', 1234, model, opt, {'model': {}})

    ck = torch.load(tmp_path / 'run' / 'checkpoints' / 'checkpoint_last.pth',
                    map_location='cpu', weights_only=False)
    assert int(ck['iteration']) == 1234, 'the iteration is what a resume restarts at'
    # The two iterates are genuinely different, or "resume from the raw one" is a distinction
    # without a difference and this test would pass on the wrong weights.
    assert not torch.equal(ck['model_state']['weight'], ck['model_state_eval']['weight'])

    fresh = torch.nn.Linear(4, 4)
    fresh_opt = AdamWScheduleFree(fresh.parameters(), lr=1e-3)
    fresh.load_state_dict(ck['model_state'])
    fresh_opt.load_state_dict(ck['optimizer_state'])
    assert torch.equal(fresh.weight, ck['model_state']['weight'])
    assert fresh_opt.state, 'no optimizer state came back, so the schedule-free z is lost'
    assert any('z' in s for s in fresh_opt.state.values()), \
        'the z iterate is the state that makes this a resume rather than a restart'
