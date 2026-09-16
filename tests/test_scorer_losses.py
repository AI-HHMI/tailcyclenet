"""Unit tests for the masked framewise scorer loss."""

import torch

from posetail.posetail.losses_scorer import TripletScorerLoss

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


def test_empty_mask_terms_have_finite_zero_backward():
    scores = torch.randn(1, 3, 2, 3, requires_grad=True)
    precision = torch.ones_like(scores)
    labels = torch.tensor([1.0, -1.0, 1.0]).expand_as(scores)
    shape = (1, 3, 2)
    zeros = torch.zeros(shape, dtype=torch.bool)
    ones = torch.ones(shape, dtype=torch.bool)
    loss = FrameTripletScorerLoss(margin=0.25, precision_reg_weight=0.0,
                                  score_reg_weight=0.0, inactive_consistency_weight=1.0,
                                  anchor_consistency_weight=1.0)
    value = loss(scores, precision, labels, active_mask=zeros, observed_mask=ones,
                 in_view_mask=ones, anchor_observed_mask=ones, far_mask=zeros,
                 near_mask=zeros, ambiguous_mask=ones)
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(scores.grad).all()


def test_signed_targets_keep_far_bad_negative_and_near_bad_positive():
    labels = torch.tensor([[[[1.0, -1.0, 1.0], [1.0, -1.0, 1.0]]]])
    observed = torch.ones((1, 1, 2), dtype=torch.bool)
    far = torch.tensor([[[True, False]]])
    near = torch.tensor([[[False, True]]])
    targets, mask = signed_pointwise_targets(
        labels, observed, observed, observed, far, near,
        ambiguous_mask=torch.zeros_like(far))
    assert torch.equal(targets[0, 0, :, 0], torch.ones(2))
    assert torch.equal(targets[0, 0, :, 1], torch.tensor([-1.0, 1.0]))
    assert bool(mask.all())


def test_all_inactive_rows_are_finite_and_not_counted_as_hits():
    scores = torch.zeros((1, 2, 1, 3), requires_grad=True)
    masks = _masks((1, 2, 1))
    masks['active_mask'].zero_()
    masks['far_mask'].zero_()
    masks['ambiguous_mask'].fill_(True)
    loss = FrameTripletScorerLoss(margin=0.25, precision_reg_weight=0.15,
                                  score_reg_weight=0.001)
    value = loss(scores, torch.ones_like(scores),
                 torch.tensor([1.0, -1.0, 1.0]).expand_as(scores), **masks)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(scores.grad).all()
    assert loss.last_counts['n_active_rows'] == 0
    assert loss.last_counts['active_weight'] == 0.0


def test_anchor_label_follows_its_source_for_far_and_near_slots():
    shape = (1, 1, 1)
    ones = torch.ones(shape, dtype=torch.bool)
    far = ones.clone()
    near = torch.zeros_like(ones)
    labels = torch.tensor([[[[1.0, -1.0, -1.0]]]])
    targets, mask = signed_pointwise_targets(labels, ones, ones, ones, far, near,
                                             ambiguous_mask=torch.zeros_like(ones))
    assert targets[0, 0, 0].tolist() == [1.0, -1.0, -1.0]
    assert bool(mask.all())
    targets, mask = signed_pointwise_targets(labels, ones, ones, ones, ~far, ones,
                                             ambiguous_mask=torch.zeros_like(ones))
    assert targets[0, 0, 0].tolist() == [1.0, 1.0, 1.0]
    assert bool(mask.all())


def test_pointwise_scale_is_positive_and_receives_gradient():
    scores = torch.tensor([[[[1.0, -1.0, 1.0]]]], requires_grad=True)
    precision = torch.ones_like(scores)
    labels = torch.tensor([[[[1.0, -1.0, 1.0]]]])
    masks = _masks((1, 1, 1))
    loss = FrameTripletScorerLoss(margin=0.25, precision_reg_weight=0.0,
                                  score_reg_weight=0.0, pointwise_weight=1.0)
    value = loss(scores, precision, labels, **masks)
    value.backward()
    assert float(loss.pointwise_scale) > 0
    assert loss.pointwise_log_scale.grad is not None
    assert torch.isfinite(loss.pointwise_log_scale.grad)


def test_pointwise_label_smoothing_is_finite():
    scores = torch.randn(1, 2, 1, 3, requires_grad=True)
    labels = torch.tensor([1.0, -1.0, 1.0]).expand_as(scores)
    loss = FrameTripletScorerLoss(pointwise_weight=0.5, score_reg_weight=0.0,
                                  pointwise_label_smoothing=0.2)
    value = loss(scores, torch.ones_like(scores), labels,
                 **_masks((1, 2, 1)))
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
