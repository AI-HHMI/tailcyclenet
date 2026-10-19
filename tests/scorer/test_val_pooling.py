"""Scorer val metrics pool raw accumulators across ranks, so they do not depend on the world
size (scorer 7)."""
import types

import numpy as np
import pytest
import torch.distributed as dist

from tailcyclenet.scorer import evaluate as ev


def _state(correct, weight, skipped=0):
    return {'per_type': {}, 'type_totals': {'drift': [weight, correct]},
            'pointwise_chunks': [(np.array([0.9, 0.1]), np.array([1.0, 0.0]),
                                  np.array([weight, weight]))],
            'pointwise_type_chunks': {}, 'localization': [1, 2],
            'view_counts': {'out_of_view': 1, 'observed': 4, 'reference_rejected': 0,
                            'base_far': 0},
            'frame_totals': {'active_weight': weight, 'active_correct': correct,
                             'active_gap': 0.5 * weight, 'valid_weight': 2 * weight},
            'rejection_draws': skipped, 'seen': 1, 'skipped': skipped}


def _fake_gather(states, monkeypatch):
    monkeypatch.setattr(dist, 'all_gather_object',
                        lambda out, obj: out.__setitem__(slice(None), states))
    return types.SimpleNamespace(world_size=len(states))


def test_pooled_ratios_are_weighted_by_the_raw_totals(monkeypatch):
    a, b = _state(correct=9.0, weight=10.0), _state(correct=0.0, weight=90.0, skipped=1)
    fabric = _fake_gather([a, b], monkeypatch)
    out = ev._finalize(ev.gather_state(fabric, a), 'frame')
    assert out['val/active_triplet_acc'] == pytest.approx(9.0 / 100.0)
    assert out['val/acc_drift'] == pytest.approx(9.0 / 100.0)
    assert out['val/active_fraction'] == pytest.approx(0.5)
    assert out['val/localization'] == pytest.approx(0.5)
    assert out['val/n_scored'] == 2 and out['val/n_skipped'] == 1
    assert out['val/rejection_rate'] == pytest.approx(1 / 3)


def test_history_pools_counts_before_the_ratio(monkeypatch):
    from collections import defaultdict

    from tailcyclenet.scorer.losses import FrameTripletScorerLoss

    class Loss:
        def __init__(self, hits, weight):
            self.loss_history = {'scorer_loss': [1.0]}
            self._totals = defaultdict(float, active_weight=weight, active_hits=hits,
                                       valid_weight=weight)

        collapse_history = FrameTripletScorerLoss.collapse_history

    r0, r1 = Loss(9.0, 10.0), Loss(0.0, 90.0)
    fabric = _fake_gather([(r0.loss_history, dict(r0._totals)),
                           (r1.loss_history, dict(r1._totals))], monkeypatch)
    out = ev.gather_history(fabric, r0)
    assert out['val/active_triplet_acc'] == pytest.approx(0.09)
    assert out['val/scorer_loss'] == pytest.approx(1.0)
