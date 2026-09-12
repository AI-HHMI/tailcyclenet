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
    climbs toward 1 MECHANICALLY -- arithmetic, not evidence about ranking. What the two arms
    actually measured on the CORRECTED camera axis, one run each:

    ==========================  =======================  ==========================
    arm                         escaped the tie state    best val/triplet_acc
    ==========================  =======================  ==========================
    0.15 (shipped control)      iteration 6000           0.8241 at iteration 7400
    0.30 (explicit overlay)     iteration 5060           0.8704 at iteration 9000
    ==========================  =======================  ==========================

    BOTH arms recovered, so 0.30 is neither a fix (0.15 recovers without it) nor a blocker. The
    0.046 gap in best val is NOT evidence of a benefit: a 9-window val is 108 triplets, i.e.
    resolution 1/108 = 0.0093 and roughly 1 sigma = 0.048, and those 108 comparisons are
    correlated, so the true uncertainty is wider than that -- and `checkpoint_best` is selection
    noise on top. One seed per arm.

    Decision (owner, 2026-09-11): the default stays 0.15, the historical control; 0.30 remains an
    explicit overlay for anyone re-measuring the lever.
    """
    import tomllib

    with open('configs/scorer.toml', 'rb') as f:
        config = tomllib.load(f)
    assert config['scorer']['precision_reg_weight'] == 0.15
    assert config['scorer']['triplet_margin'] == 0.25
