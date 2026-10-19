"""Unit tests for the masked framewise scorer loss."""
import torch
from tracktail.model.losses_scorer import TripletScorerLoss

from tailcyclenet.scorer.losses import FrameTripletScorerLoss, signed_pointwise_targets


def _masks(shape):
    ones = torch.ones(shape, dtype=torch.bool)
    zeros = torch.zeros(shape, dtype=torch.bool)
    return dict(active_mask=ones, observed_mask=ones, in_view_mask=ones,
                anchor_observed_mask=ones, far_mask=ones, near_mask=zeros,
                ambiguous_mask=zeros)


def test_all_active_t1_matches_installed_triplet_loss():
    torch.manual_seed(4)
    scores = torch.randn(2, 1, 5, 3)
    precision = torch.sigmoid(torch.randn_like(scores))
    labels = torch.tensor([1.0, -1.0, 1.0]).expand_as(scores)
    kwargs = _masks((2, 1, 5))
    frame = FrameTripletScorerLoss(margin=0.25, precision_reg_weight=0.15,
                                   score_reg_weight=0.001)
    got = frame(scores, precision, labels, **kwargs)
    legacy = TripletScorerLoss(margin=0.25, precision_reg_weight=0.15,
                               score_reg_weight=0.001)
    want = legacy(scores.reshape(-1, 3), precision.reshape(-1, 3), labels.reshape(-1, 3))
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_empty_and_all_inactive_masks_are_finite_and_count_nothing():
    """No active row must give a finite loss and gradient, not a 0/0, and zero active counts."""
    scores = torch.randn(1, 3, 2, 3, requires_grad=True)
    labels = torch.tensor([1.0, -1.0, 1.0]).expand_as(scores)
    masks = _masks((1, 3, 2))
    for key in ('active_mask', 'far_mask'):
        masks[key] = torch.zeros_like(masks[key])
    masks['ambiguous_mask'] = torch.ones_like(masks['ambiguous_mask'])
    for kwargs in (dict(precision_reg_weight=0.0, score_reg_weight=0.0,
                        inactive_consistency_weight=1.0, anchor_consistency_weight=1.0),
                   dict(precision_reg_weight=0.15, score_reg_weight=0.001)):
        scores.grad = None
        loss = FrameTripletScorerLoss(margin=0.25, **kwargs)
        value = loss(scores, torch.ones_like(scores), labels, **masks)
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(scores.grad).all()
        assert loss.last_counts['n_active_rows'] == 0
        assert loss.last_counts['active_weight'] == 0.0


def test_signed_targets_follow_the_distance_state():
    """Far bad is negative, near bad is clean (+1); the anchor follows its source either way."""
    labels = torch.tensor([[[[1.0, -1.0, 1.0], [1.0, -1.0, 1.0]]]])
    observed = torch.ones((1, 1, 2), dtype=torch.bool)
    far = torch.tensor([[[True, False]]])
    targets, mask = signed_pointwise_targets(labels, observed, observed, observed, far, ~far,
                                             ambiguous_mask=torch.zeros_like(far))
    assert torch.equal(targets[0, 0, :, 0], torch.ones(2))
    assert torch.equal(targets[0, 0, :, 1], torch.tensor([-1.0, 1.0]))
    assert bool(mask.all())

    ones = torch.ones((1, 1, 1), dtype=torch.bool)
    labels = torch.tensor([[[[1.0, -1.0, -1.0]]]])
    for far, want in ((ones, [1.0, -1.0, -1.0]), (~ones, [1.0, 1.0, 1.0])):
        targets, mask = signed_pointwise_targets(labels, ones, ones, ones, far, ~far,
                                                 ambiguous_mask=torch.zeros_like(ones))
        assert targets[0, 0, 0].tolist() == want
        assert bool(mask.all())


def test_pointwise_term_scale_learns_and_label_smoothing_is_finite():
    scores = torch.tensor([[[[1.0, -1.0, 1.0]]]], requires_grad=True)
    loss = FrameTripletScorerLoss(margin=0.25, precision_reg_weight=0.0,
                                  score_reg_weight=0.0, pointwise_weight=1.0)
    loss(scores, torch.ones_like(scores), scores.detach().sign(), **_masks((1, 1, 1))).backward()
    assert float(loss.pointwise_scale.detach()) > 0
    assert loss.pointwise_log_scale.grad is not None
    assert torch.isfinite(loss.pointwise_log_scale.grad)

    scores = torch.randn(1, 2, 1, 3, requires_grad=True)
    labels = torch.tensor([1.0, -1.0, 1.0]).expand_as(scores)
    loss = FrameTripletScorerLoss(pointwise_weight=0.5, score_reg_weight=0.0,
                                  pointwise_label_smoothing=0.2)
    value = loss(scores, torch.ones_like(scores), labels, **_masks((1, 2, 1)))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(scores.grad).all()


def test_zero_aggregate_score_gap_is_not_reported_as_nan():
    scores = torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]])
    labels = torch.tensor([[[[1.0, -1.0, 1.0]], [[1.0, -1.0, 1.0]]]])
    loss = FrameTripletScorerLoss(precision_reg_weight=0.0, score_reg_weight=0.0)
    loss(scores, torch.ones_like(scores), labels, **_masks((1, 2, 1)))
    assert loss.collapse_history()['active_score_gap'] == 0.0


def test_duplicate_source_weights_do_not_change_a_repeated_row():
    scores = torch.tensor([[[[2.0, 0.0, 1.0]], [[2.0, 0.0, 1.0]]]])
    precision = torch.ones_like(scores)
    labels = torch.tensor([[[[1.0, -1.0, 1.0]], [[1.0, -1.0, 1.0]]]])
    masks = _masks((1, 2, 1))
    repeated = FrameTripletScorerLoss(margin=0.25, precision_reg_weight=0.15,
                                      score_reg_weight=0.001)
    weighted = repeated(scores, precision, labels,
                        source_frame_weight=torch.tensor([[[0.5], [0.5]]]), **masks)
    single = FrameTripletScorerLoss(margin=0.25, precision_reg_weight=0.15,
                                    score_reg_weight=0.001)
    one = single(scores[:, :1], precision[:, :1], labels[:, :1],
                 source_frame_weight=torch.ones((1, 1, 1)), **_masks((1, 1, 1)))
    assert torch.allclose(weighted, one, atol=1e-6, rtol=1e-6)
