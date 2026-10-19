"""Regression tests for the scorer precision-collapse attractor."""
import math

import torch

from tracktail.model.losses_scorer import TripletScorerLoss


def _equal_score_loss(precision_reg_weight, p, margin=0.25):
    """Loss at the equal-score collapsed point for one clean/corrupt/anchor triplet."""
    scores = torch.zeros((1, 3), dtype=torch.float32)
    precisions = torch.full((1, 3), p, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[1.0, -1.0, 1.0]])
    loss = TripletScorerLoss(margin=margin,
                             precision_reg_weight=precision_reg_weight,
                             score_reg_weight=0.0)(scores, precisions, labels)
    return loss, precisions


def test_precision_weight_sets_the_collapse_minimum():
    """At 0.15/0.25 the equal-score point has the observed interior minimum at p = 0.6; with the
    weight above the margin it is gone and gradient descent raises precision."""
    loss, _ = _equal_score_loss(0.15, 0.60)
    expected = 0.25 * 0.60 - 0.15 * math.log(0.60)
    assert math.isclose(float(loss.detach()), expected, rel_tol=1e-6)
    for p in (0.50, 0.70):
        assert float(loss.detach()) < float(_equal_score_loss(0.15, p)[0].detach())

    at_collapsed, precision = _equal_score_loss(0.30, 0.60)
    assert float(_equal_score_loss(0.30, 0.99)[0].detach()) < float(at_collapsed.detach())
    at_collapsed.backward()
    assert float(precision.grad.mean()) < 0.0, 'gradient descent must increase precision'


def test_shipped_recipe_is_the_unchanged_control():
    """Owner decision 2026-09-11: 0.15 stays the default; 0.30 is an explicit overlay (both arms
    recovered from the tie state and the val gap was within noise)."""
    import tomllib

    with open('configs/scorer.toml', 'rb') as f:
        config = tomllib.load(f)
    assert config['scorer']['precision_reg_weight'] == 0.15
    assert config['scorer']['triplet_margin'] == 0.25
