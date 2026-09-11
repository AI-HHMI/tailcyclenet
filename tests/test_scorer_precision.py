"""Regression tests for the scorer precision-collapse attractor."""
import math

import torch

from posetail.posetail.losses_scorer import TripletScorerLoss


def _equal_score_loss(precision_reg_weight, p, margin=0.25):
    """Loss at the equal-score collapsed point for one clean/corrupt/anchor triplet."""
    scores = torch.zeros((1, 3), dtype=torch.float32)
    precisions = torch.full((1, 3), p, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[1.0, -1.0, 1.0]])
    loss = TripletScorerLoss(margin=margin,
                             precision_reg_weight=precision_reg_weight,
                             score_reg_weight=0.0)(scores, precisions, labels)
    return loss, precisions


def test_old_recipe_has_the_observed_interior_collapse_minimum():
    """The historical 0.15/0.25 recipe is pinned to explain the observed plateau."""
    loss, _ = _equal_score_loss(0.15, 0.60)
    expected = 0.25 * 0.60 - 0.15 * math.log(0.60)
    assert math.isclose(float(loss.detach()), expected, rel_tol=1e-6)
    at_low, _ = _equal_score_loss(0.15, 0.50)
    at_high, _ = _equal_score_loss(0.15, 0.70)
    assert float(loss.detach()) < float(at_low.detach())
    assert float(loss.detach()) < float(at_high.detach())


def test_experimental_higher_precision_weight_pushes_precision_up():
    """With lambda above the margin, the collapsed point is not an interior minimum."""
    at_collapsed, _ = _equal_score_loss(0.30, 0.60)
    at_high, _ = _equal_score_loss(0.30, 0.99)
    assert float(at_high.detach()) < float(at_collapsed.detach())

    _, precision = _equal_score_loss(0.30, 0.60)
    loss = TripletScorerLoss(margin=0.25, precision_reg_weight=0.30,
                             score_reg_weight=0.0)(
        torch.zeros((1, 3)), precision, torch.tensor([[1.0, -1.0, 1.0]]))
    loss.backward()
    assert float(precision.grad.mean()) < 0.0, 'gradient descent must increase precision'


def test_shipped_recipe_is_the_unchanged_control():
    """The shipped recipe stays at the historical value; 0.30 remains an explicit overlay.

    `precision_reg_weight = 0.30` removes the interior `p = lambda/margin` minimum, so precision
    climbs toward 1 MECHANICALLY -- that is arithmetic, not evidence that ranking recovered. The
    measurements: the first 0.30 arm (154271160) reached precision ~0.9 with `score_gap ~0.0025`
    and was then found to run on misaligned videos, so it settled nothing. The corrected-axis arm
    (154272291, `scratch/scorer-qdmouse4m-fluo-separate-cameras/config-precision03-v2.toml`) sat
    at loss 0.25 / gap ~6e-06 through iteration ~3900 while the 0.15 control left the plateau at
    iteration 6000 (loss 0.2266 -> 0.2198, gap 1.6e-02). The control's escape that late is also
    why the two are not yet separable at that iteration -- so the honest statement is that 0.30
    has shown no ranking recovery, not that it prevents one.

    Decision (owner, 2026-09-11): the default stays 0.15.
    """
    import tomllib

    with open('configs/scorer.toml', 'rb') as f:
        config = tomllib.load(f)
    assert config['scorer']['precision_reg_weight'] == 0.15
    assert config['scorer']['triplet_margin'] == 0.25
