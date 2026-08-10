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
