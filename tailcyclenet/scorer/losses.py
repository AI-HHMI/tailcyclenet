"""Losses for framewise synthetic track-quality triplets.

The installed :class:`posetail...TripletScorerLoss` remains the sequence-mode loss.  This module
owns the frame-mode contract instead of monkey-patching the installed package: masks are built
before rows are flattened, duplicate source frames are weighted once, and clean/corrupt labels
are optionally given a zero-centred signed pointwise term.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


_MASK_KEYS = (
    'active_mask', 'observed_mask', 'anchor_observed_mask', 'in_view_mask', 'far_mask',
    'near_mask', 'ambiguous_mask', 'reference_far_mask', 'reference_rejected_mask',
    'source_frame_weight', 'corruption_type_mask', 'fired_frame', 'fired', 'frames',
    'good_reference_distance_px', 'bad_reference_distance_px',
)


def _as_tensor(value, device, dtype=None):
    """Convert worker metadata without assuming it was already moved to the model device."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _shape_mask(value, shape, device, *, name, fill=True):
    """Broadcast a ``[T,K]``/``[B,T,K]`` metadata mask to the score shape."""
    b, t, k = shape
    if value is None:
        return torch.full(shape, bool(fill), dtype=torch.bool, device=device)
    value = _as_tensor(value, device, torch.bool)
    if value.ndim == 0:
        return value.expand(shape)
    if tuple(value.shape) == (t, k):
        value = value.unsqueeze(0)
    if tuple(value.shape) != shape:
        raise ValueError(f'{name} must have shape [B,T,K]={shape} or [T,K], got {tuple(value.shape)}')
    return value


def _shape_float(value, shape, device, *, name, fill=1.0):
    """Broadcast a source-frame weight tensor to ``[B,T,K]``."""
    if value is None:
        return torch.full(shape, fill, dtype=torch.float32, device=device)
    value = _as_tensor(value, device, torch.float32)
    b, t, k = shape
    if value.ndim == 0:
        return value.expand(shape)
    if tuple(value.shape) == (t, k):
        value = value.unsqueeze(0)
    if tuple(value.shape) != shape:
        raise ValueError(f'{name} must have shape [B,T,K]={shape} or [T,K], got {tuple(value.shape)}')
    if not torch.isfinite(value).all() or bool((value < 0).any()):
        raise ValueError(f'{name} must contain finite non-negative weights')
    return value


def _weighted_mean(values, mask, weights=None, *, reduction='mean'):
    """A differentiable weighted reduction, returning zero for an empty mask."""
    if values.shape != mask.shape:
        raise ValueError(f'values/mask shapes differ: {tuple(values.shape)} vs {tuple(mask.shape)}')
    w = torch.ones_like(values) if weights is None else weights.to(values.dtype)
    use = mask.to(values.dtype) * w
    numerator = (values * use).sum()
    if reduction == 'sum':
        return numerator, use.sum()
    denominator = use.sum()
    safe_denominator = denominator.clamp_min(1.0)
    zero = values.sum() * 0.0
    return torch.where(denominator > 0, numerator / safe_denominator, zero), denominator


def _triplet_values(scores, precisions, labels):
    """Installed posetail triplet algebra on an already-selected ``[N,3]`` row set."""
    if scores.numel() == 0:
        zero = scores.sum() * 0.0
        return zero, torch.empty(0, device=scores.device), torch.empty(0, device=scores.device)
    example_type = labels.sum(dim=-1, keepdim=True)
    close_mask = labels == example_type
    if not bool((close_mask.sum(-1) == 2).all()):
        raise ValueError('frame triplet labels must contain two same-label members and one odd member')
    close_scores = scores[close_mask].reshape(-1, 2)
    distant_scores = scores[~close_mask].reshape(-1, 1)
    close_prec = precisions[close_mask].reshape(-1, 2)
    distant_prec = precisions[~close_mask].reshape(-1, 1)
    dist_pos = torch.abs(close_scores[:, 0] - close_scores[:, 1])
    dist_neg = torch.min((distant_scores - close_scores) * example_type, dim=-1).values
    triplet = dist_pos + dist_neg
    all_prec = torch.cat([close_prec, distant_prec], dim=-1)
    triplet_precision = all_prec.prod(dim=-1).pow(1.0 / 3.0)
    return triplet, triplet_precision, example_type[:, 0]


def signed_pointwise_targets(labels, observed_mask, in_view_mask, anchor_observed_mask,
                             far_mask, near_mask, *, anchor_label=None,
                             ambiguous_mask=None, reference_rejected_mask=None,
                             reference_gate='source_far', good_reference_distance_px=None,
                             max_clean_px=0.0):
    """Construct member-specific signed targets and masks for the direct framewise term.

    Returns ``(targets, mask)`` with shape ``[B,T,K,3]``.  Columns are good, bad, anchor; targets
    are +1 for clean and -1 for corrupt.  A bad member is negative only on final-far rows and is
    positive on confidently near rows.  Ambiguous, missing, out-of-view and reference-rejected
    rows carry no target.  ``labels[...,2]`` supplies the anchor source sign when
    ``anchor_label`` is not given.
    """
    shape = observed_mask.shape
    if labels.shape[:3] != shape or labels.shape[-1] != 3:
        raise ValueError('labels and pointwise masks must align as [B,T,K,3] and [B,T,K]')
    observed_mask = observed_mask.bool()
    in_view_mask = in_view_mask.bool()
    anchor_observed_mask = anchor_observed_mask.bool()
    far_mask = far_mask.bool()
    near_mask = near_mask.bool()
    ambiguous_mask = (torch.zeros_like(observed_mask) if ambiguous_mask is None
                      else ambiguous_mask.bool())
    rejected = (torch.zeros_like(observed_mask) if reference_rejected_mask is None
                else reference_rejected_mask.bool())
    member = observed_mask & in_view_mask & ~rejected & ~ambiguous_mask

    independent = str(reference_gate) == 'independent_far'
    if independent:
        if good_reference_distance_px is None:
            raise ValueError(
                'independent_far pointwise labels require good_reference_distance_px metadata')
        good_reference_distance_px = good_reference_distance_px.to(observed_mask.device)
        good_clean = member & torch.isfinite(good_reference_distance_px)
        good_clean &= good_reference_distance_px <= float(max_clean_px)
    else:
        good_clean = member

    targets = torch.zeros((*shape, 3), dtype=torch.float32, device=labels.device)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    targets[..., 0] = 1.0
    mask[..., 0] = good_clean

    clean_gate = good_clean if independent else torch.ones_like(member)
    bad_valid = member & ~ambiguous_mask & clean_gate & (far_mask | near_mask)
    targets[..., 1] = torch.where(far_mask, -torch.ones_like(far_mask, dtype=torch.float32),
                                  torch.ones_like(far_mask, dtype=torch.float32))
    mask[..., 1] = bad_valid

    anc_valid = anchor_observed_mask & in_view_mask & ~rejected & ~ambiguous_mask
    if anchor_label is None:
        anchor_is_bad = labels[..., 2] < 0
    else:
        a = torch.as_tensor(anchor_label, device=labels.device, dtype=labels.dtype)
        anchor_is_bad = (a < 0).expand(shape)
    anchor_bad_valid = anc_valid & ~ambiguous_mask & clean_gate & (far_mask | near_mask)
    targets[..., 2] = torch.where(
        anchor_is_bad,
        torch.where(far_mask, -torch.ones_like(far_mask, dtype=torch.float32),
                    torch.ones_like(far_mask, dtype=torch.float32)),
        torch.ones_like(far_mask, dtype=torch.float32))
    mask[..., 2] = torch.where(anchor_is_bad, anchor_bad_valid, anc_valid)
    return targets, mask


class FrameTripletScorerLoss(nn.Module):
    """Masked ranking and signed pointwise loss for framewise scorer outputs.

    Parameters mirror the installed sequence loss where applicable.  ``forward`` accepts structured
    ``[B,T,K,3]`` scores/precisions/labels and a ``triplet=`` dictionary from
    :func:`tailcyclenet.scorer.triplet.make_triplet`.  Every mask/weight can instead be supplied as
    an explicit keyword.  The small compatibility convenience of accepting ``[T,K,3]`` and
    ``[N,3]`` is retained for focused tests and batch-one callers.

    Ranking uses only ``active_mask`` (final far, observed, in-view, anchor-observed).  Precision
    and score regularizers use valid/evaluable rows.  ``pointwise_weight`` adds a signed logistic
    term on raw scores with a learned positive scale ``softplus(pointwise_log_scale)`` and no
    additive offset.  A pointwise run forces the score regularizer off because both terms otherwise
    anchor the same score level in opposite ways.
    """

    def __init__(self, margin=0.5, precision_reg_weight=0.01, score_reg_weight=0.0,
                 reduction='mean', pointwise_weight=0.0, pointwise_balance=True,
                 pointwise_label_smoothing=0.0, inactive_consistency_weight=0.0,
                 anchor_consistency_weight=0.0, triplet_margin=None, max_clean_px=0.0):
        """Initialize loss weights and the fresh positive pointwise calibration scale."""
        super().__init__()
        if triplet_margin is not None:
            margin = triplet_margin
        if reduction not in ('mean', 'sum'):
            raise ValueError(f'unsupported reduction {reduction!r}; use mean or sum')
        if float(pointwise_weight) < 0:
            raise ValueError('pointwise_weight must be non-negative')
        if not 0 <= float(pointwise_label_smoothing) < 1:
            raise ValueError('pointwise_label_smoothing must be in [0,1)')
        if float(pointwise_balance) not in (0.0, 1.0) and not isinstance(pointwise_balance, bool):
            raise ValueError('pointwise_balance must be boolean')
        self.margin = float(margin)
        self.precision_reg_weight = float(precision_reg_weight)
        self.requested_score_reg_weight = float(score_reg_weight)
        self.score_reg_weight = 0.0 if float(pointwise_weight) > 0 else float(score_reg_weight)
        self.reduction = reduction
        self.pointwise_weight = float(pointwise_weight)
        self.pointwise_balance = bool(pointwise_balance)
        self.pointwise_label_smoothing = float(pointwise_label_smoothing)
        self.inactive_consistency_weight = float(inactive_consistency_weight)
        self.anchor_consistency_weight = float(anchor_consistency_weight)
        self.max_clean_px = float(max_clean_px)
        self.pointwise_log_scale = nn.Parameter(torch.zeros(()))
        self.loss_history = defaultdict(list)
        self._totals = defaultdict(float)
        self.last_counts = {}

    @property
    def pointwise_scale(self):
        """Positive scale used by the signed pointwise logits."""
        return F.softplus(self.pointwise_log_scale)

    def _metadata(self, triplet, name, explicit):
        """Resolve an explicit metadata value before the triplet dictionary value."""
        if explicit is not None:
            return explicit
        if triplet is None:
            return None
        return triplet.get(name)

    def _prepare_shape(self, scores, precisions, labels):
        """Validate member axes and normalize supported structured score shapes."""
        if scores.shape != precisions.shape or scores.shape != labels.shape:
            raise ValueError('scores, precisions and labels must have identical shapes')
        if scores.shape[-1] != 3:
            raise ValueError(f'last score axis must be triplet members of length 3, got {scores.shape}')
        original = scores.ndim
        if original == 4:
            return scores, precisions, labels, False
        if original == 3:
            return scores.unsqueeze(0), precisions.unsqueeze(0), labels.unsqueeze(0), True
        if original == 2:
            return scores.reshape(1, scores.shape[0], 1, 3), precisions.reshape(
                1, precisions.shape[0], 1, 3), labels.reshape(1, labels.shape[0], 1, 3), True
        raise ValueError('framewise scorer loss expects [B,T,K,3], [T,K,3] or [N,3]')

    def _source_weights(self, triplet, explicit, shape, device, eligible):
        """Resolve duplicate-source-frame weights, deriving them from source ids when needed."""
        value = self._metadata(triplet, 'source_frame_weight', explicit)
        if value is not None:
            return _shape_float(value, shape, device, name='source_frame_weight')
        frames = self._metadata(triplet, 'frames', None)
        if frames is None:
            return torch.ones(shape, dtype=torch.float32, device=device)
        from .triplet import source_frame_weights
        return source_frame_weights(frames, eligible=eligible, device=device)

    def _pointwise_loss(self, scores, labels, *, observed, in_view, anchor_observed,
                        far, near, ambiguous, rejected, reference_gate,
                        good_reference, source_weight, anchor_label):
        """Compute balanced signed logistic supervision and its diagnostic counts."""
        targets, target_mask = signed_pointwise_targets(
            labels, observed, in_view, anchor_observed, far, near,
            anchor_label=anchor_label, ambiguous_mask=ambiguous,
            reference_rejected_mask=rejected, reference_gate=reference_gate,
            good_reference_distance_px=good_reference, max_clean_px=self.max_clean_px)
        signed_targets = targets * (1.0 - self.pointwise_label_smoothing)
        losses = F.softplus(-signed_targets * self.pointwise_scale * scores)
        weights = source_weight[..., None].expand_as(losses)
        total = scores.sum() * 0.0
        n_rows = 0.0
        pos_mass = neg_mass = 0.0
        for bi in range(scores.shape[0]):
            m = target_mask[bi]
            if self.pointwise_balance:
                pos = m & (targets[bi] > 0)
                neg = m & (targets[bi] < 0)
                terms = []
                for cls in (pos, neg):
                    mass = (weights[bi] * cls).sum()
                    if bool(mass > 0):
                        terms.append((losses[bi] * weights[bi] * cls).sum() / mass)
                    if cls is pos:
                        pos_mass += float(mass.detach())
                    else:
                        neg_mass += float(mass.detach())
                if terms:
                    total = total + torch.stack(terms).mean()
                    n_rows += float(m.sum().detach())
            else:
                value, mass = _weighted_mean(losses[bi], m, weights[bi], reduction='mean')
                total = total + value
                n_rows += float(m.sum().detach())
                pos_mass += float((weights[bi] * (m & (targets[bi] > 0))).sum().detach())
                neg_mass += float((weights[bi] * (m & (targets[bi] < 0))).sum().detach())
        denom = max(scores.shape[0], 1)
        return total / denom, target_mask, {'n_pointwise_rows': n_rows,
                                            'pointwise_pos_mass': pos_mass,
                                            'pointwise_neg_mass': neg_mass}

    def forward(self, scores, precisions, labels, triplet=None, *, active_mask=None,
                observed_mask=None, anchor_observed_mask=None, in_view_mask=None,
                far_mask=None, near_mask=None, ambiguous_mask=None,
                reference_far_mask=None, reference_rejected_mask=None,
                source_frame_weight=None, corruption_type_mask=None, fired=None,
                frames=None, corrupted_keypoint=None, anchor_label=None):
        """Compute the masked frame loss and update count-based history.

        Explicit metadata kwargs override values in ``triplet``.  The key masks are all
        ``[B,T,K]`` (or ``[T,K]``); ``corruption_type_mask`` is ``[B,T,K,G]`` and is used only for
        per-generator diagnostics.  ``active_mask`` is required semantically for frame mode but
        defaults to all rows when no triplet metadata is supplied, which makes the loss useful in
        isolated algebra tests.
        """
        scores, precisions, labels, unbatched = self._prepare_shape(scores, precisions, labels)
        device = scores.device
        b, t, k, _ = scores.shape
        shape = (b, t, k)
        trip_meta = triplet
        if frames is not None:
            if trip_meta is None:
                trip_meta = {'frames': frames}
            else:
                trip_meta = {**trip_meta, 'frames': frames}
        observed = _shape_mask(self._metadata(trip_meta, 'observed_mask', observed_mask), shape,
                               device, name='observed_mask')
        anchor_observed = _shape_mask(
            self._metadata(trip_meta, 'anchor_observed_mask', anchor_observed_mask), shape,
            device, name='anchor_observed_mask')
        in_view = _shape_mask(self._metadata(trip_meta, 'in_view_mask', in_view_mask), shape,
                              device, name='in_view_mask')
        far = _shape_mask(self._metadata(trip_meta, 'far_mask', far_mask), shape, device,
                          name='far_mask', fill=False)
        near = _shape_mask(self._metadata(trip_meta, 'near_mask', near_mask), shape, device,
                           name='near_mask', fill=False)
        ambiguous = _shape_mask(
            self._metadata(trip_meta, 'ambiguous_mask', ambiguous_mask), shape, device,
            name='ambiguous_mask', fill=False)
        rejected = _shape_mask(
            self._metadata(trip_meta, 'reference_rejected_mask', reference_rejected_mask), shape,
            device, name='reference_rejected_mask', fill=False)
        given_active = self._metadata(trip_meta, 'active_mask', active_mask)
        if given_active is None:
            active = far & observed & in_view & anchor_observed
            if not bool(far.any()) and triplet is None and active_mask is None:
                active = torch.ones(shape, dtype=torch.bool, device=device)
        else:
            active = _shape_mask(given_active, shape, device, name='active_mask', fill=False)
            active &= far & observed & in_view & anchor_observed
        valid = observed & in_view & anchor_observed & ~ambiguous & ~rejected
        eligible = valid
        weights = self._source_weights(
            trip_meta, source_frame_weight, shape, device, eligible)
        if anchor_label is None:
            anchor_label = self._metadata(trip_meta, 'anchor_label', None)
        reference_gate = str((trip_meta or {}).get('reference_gate', 'source_far'))
        good_reference = self._metadata(trip_meta, 'good_reference_distance_px', None)
        if good_reference is not None:
            good_reference = _as_tensor(good_reference, device, torch.float32)
            if good_reference.ndim == 2:
                good_reference = good_reference.unsqueeze(0)
            if tuple(good_reference.shape) != shape:
                raise ValueError(
                    f'good_reference_distance_px must be {shape}, got {tuple(good_reference.shape)}')

        flat_scores = scores.reshape(-1, 3)
        flat_prec = precisions.reshape(-1, 3)
        flat_labels = labels.reshape(-1, 3)
        flat_active = active.reshape(-1)
        flat_weights = weights.reshape(-1)
        if bool(flat_active.any()):
            selected_scores = flat_scores[flat_active]
            selected_prec = flat_prec[flat_active]
            selected_labels = flat_labels[flat_active]
            raw_triplet, triplet_precision, _ = _triplet_values(
                selected_scores, selected_prec, selected_labels)
            raw_triplet = raw_triplet + self.margin
            per_row = F.relu(raw_triplet) * triplet_precision
            rank_value, active_weight = _weighted_mean(
                per_row, torch.ones_like(per_row, dtype=torch.bool),
                flat_weights[flat_active], reduction=self.reduction)
            rank_loss = rank_value
            active_hits = ((selected_scores[:, 0] > selected_scores[:, 1]).float()
                           * flat_weights[flat_active]).sum()
            score_gap_num = ((selected_scores[:, 0] - selected_scores[:, 1])
                             * flat_weights[flat_active]).sum()
        else:
            rank_loss = scores.sum() * 0.0
            active_weight = flat_weights[flat_active].sum()
            active_hits = scores.sum() * 0.0
            score_gap_num = scores.sum() * 0.0

        valid_member = valid[..., None].expand_as(scores)
        reg_weights = weights[..., None].expand_as(scores)
        precision_values = -torch.log(precisions.clamp_min(1e-6))
        precision_reg, _ = _weighted_mean(
            precision_values, valid_member, reg_weights, reduction='mean')
        precision_reg = precision_reg * self.precision_reg_weight
        score_reg, _ = _weighted_mean(scores.pow(2), valid_member, reg_weights, reduction='mean')
        score_reg = score_reg * self.score_reg_weight

        corrupted = self._metadata(trip_meta, 'corrupted_keypoint', corrupted_keypoint)
        if corrupted is None:
            corrupted = far.any(dim=1)
        else:
            corrupted = _as_tensor(corrupted, device, torch.bool)
            if corrupted.ndim == 1:
                corrupted = corrupted.unsqueeze(0)
            if tuple(corrupted.shape) != (b, k):
                raise ValueError(
                    f'corrupted_keypoint must have shape {(b, k)} or {(k,)}, got '
                    f'{tuple(corrupted.shape)}')
        consistency = valid & near & corrupted[:, None, :]
        inactive_values = torch.abs(scores[..., 1] - scores[..., 0])
        inactive_loss, _ = _weighted_mean(
            inactive_values, consistency, weights, reduction='mean')
        inactive_loss = inactive_loss * self.inactive_consistency_weight

        anchor_source_bad = (labels[..., 2] < 0)
        source_score = torch.where(anchor_source_bad, scores[..., 1], scores[..., 0])
        anchor_values = torch.abs(scores[..., 2] - source_score)
        anchor_loss, _ = _weighted_mean(anchor_values, valid, weights, reduction='mean')
        anchor_loss = anchor_loss * self.anchor_consistency_weight

        pointwise_loss = scores.sum() * 0.0
        point_mask = torch.zeros((*shape, 3), dtype=torch.bool, device=device)
        point_counts = {'n_pointwise_rows': 0.0, 'pointwise_pos_mass': 0.0,
                        'pointwise_neg_mass': 0.0}
        if self.pointwise_weight > 0:
            pointwise_loss, point_mask, point_counts = self._pointwise_loss(
                scores, labels, observed=observed, in_view=in_view,
                anchor_observed=anchor_observed, far=far, near=near,
                ambiguous=ambiguous, rejected=rejected,
                reference_gate=reference_gate, good_reference=good_reference,
                source_weight=weights, anchor_label=anchor_label)
            pointwise_loss = pointwise_loss * self.pointwise_weight

        total = rank_loss + precision_reg + score_reg + inactive_loss + anchor_loss + pointwise_loss
        if unbatched:
            total = total + scores.sum() * 0.0

        with torch.no_grad():
            active_rows = int(flat_active.sum())
            valid_rows = int(valid.sum())
            active_weight_value = float(active_weight.detach())
            valid_weight_value = float(weights[valid].sum().detach())
            hit_value = float(active_hits.detach())
            gap_value = float(score_gap_num.detach())
            type_mask = self._metadata(trip_meta, 'corruption_type_mask', corruption_type_mask)
            if type_mask is None:
                type_mask = self._metadata(trip_meta, 'fired_frame', fired)
            if type_mask is None:
                type_mask = self._metadata(trip_meta, 'fired', None)
            if type_mask is not None:
                type_mask = _as_tensor(type_mask, device, torch.bool)
                if type_mask.ndim == 3:
                    type_mask = type_mask[:, None].expand(b, t, k, type_mask.shape[-1])
                elif type_mask.ndim == 2:
                    type_mask = type_mask[None, None].expand(b, t, k, type_mask.shape[-1])
                if tuple(type_mask.shape[:3]) != shape:
                    raise ValueError(
                        f'corruption_type_mask must align with {shape}, got {tuple(type_mask.shape)}')
                for gi in range(type_mask.shape[-1]):
                    tm = type_mask[..., gi] & active
                    tw = weights[tm]
                    hit = ((scores[..., 0] > scores[..., 1])[tm].float() * tw).sum()
                    self._totals[f'type_{gi}_hits'] += float(hit)
                    self._totals[f'type_{gi}_weight'] += float(tw.sum())
                    self._totals[f'type_{gi}_rows'] += int(tm.sum())
            self.loss_history['scorer_loss'].append(float(total.detach()))
            self.loss_history['triplet_loss'].append(float(rank_loss.detach()))
            self.loss_history['precision_reg'].append(float(precision_reg.detach()))
            self.loss_history['score_reg'].append(float(score_reg.detach()))
            self.loss_history['inactive_consistency'].append(float(inactive_loss.detach()))
            self.loss_history['anchor_consistency'].append(float(anchor_loss.detach()))
            self.loss_history['pointwise_loss'].append(float(pointwise_loss.detach()))
            self.loss_history['active_triplet_acc'].append(
                hit_value / active_weight_value if active_weight_value else float('nan'))
            self.loss_history['active_score_gap'].append(
                gap_value / active_weight_value if active_weight_value else float('nan'))
            self.loss_history['active_fraction'].append(
                active_weight_value / valid_weight_value if valid_weight_value else float('nan'))
            self.loss_history['n_active_rows'].append(float(active_rows))
            self.loss_history['n_active_source_rows'].append(active_weight_value)
            self.loss_history['n_valid_rows'].append(float(valid_rows))
            self.loss_history['n_valid_source_rows'].append(valid_weight_value)
            self.loss_history['n_precision_rows'].append(valid_weight_value * 3.0)
            self.loss_history['n_score_rows'].append(valid_weight_value * 3.0)
            self.loss_history['n_consistency_rows'].append(float(consistency.sum()))
            self.loss_history['n_duplicate_weighted_rows'].append(
                float((weights < 1).logical_and(valid).sum()))
            self.loss_history['n_pointwise_rows'].append(point_counts['n_pointwise_rows'])
            self.loss_history['pointwise_pos_mass'].append(point_counts['pointwise_pos_mass'])
            self.loss_history['pointwise_neg_mass'].append(point_counts['pointwise_neg_mass'])
            valid_denom = weights[valid].sum()
            self.loss_history['score_good'].append(
                float((scores[..., 0][valid] * weights[valid]).sum() / valid_denom)
                if bool(valid_denom > 0) else float('nan'))
            self.loss_history['score_bad'].append(
                float((scores[..., 1][valid] * weights[valid]).sum() / valid_denom)
                if bool(valid_denom > 0) else float('nan'))
            self.loss_history['mean_precision'].append(
                float((precisions[valid] * weights[valid][..., None]).sum()
                      / (valid_denom * precisions.shape[-1]))
                if bool(valid_denom > 0) else float('nan'))
            self._totals['active_hits'] += hit_value
            self._totals['active_weight'] += active_weight_value
            self._totals['active_gap_num'] += gap_value
            self._totals['active_rows'] += active_rows
            self._totals['valid_rows'] += valid_rows
            self._totals['valid_weight'] += valid_weight_value
            self._totals['consistency_rows'] += int(consistency.sum())
            self._totals['duplicate_weighted_rows'] += int((weights < 1).logical_and(valid).sum())
            self.last_counts = {
                'n_active_rows': active_rows, 'n_active_source_rows': active_weight_value,
                'n_valid_rows': valid_rows, 'n_valid_source_rows': valid_weight_value,
                'active_weight': active_weight_value, 'valid_weight': valid_weight_value,
                'active_fraction': (active_weight_value / valid_weight_value
                                    if valid_weight_value else 0.0),
                'n_consistency_rows': int(consistency.sum()),
                'n_pointwise_rows': point_counts['n_pointwise_rows'],
                'pointwise_pos_mass': point_counts['pointwise_pos_mass'],
                'pointwise_neg_mass': point_counts['pointwise_neg_mass'],
                'n_duplicate_weighted_rows': int((weights < 1).logical_and(valid).sum()),
            }
            self.loss_history['triplet_acc'].append(
                hit_value / active_weight_value if active_weight_value else float('nan'))
            self.loss_history['score_gap'].append(
                gap_value / active_weight_value if active_weight_value else float('nan'))
        return total

    def collapse_history(self, prefix=''):
        """Collapse history with count-based active metrics rather than averaging window ratios."""
        out = {}
        if self.loss_history:
            for name, values in self.loss_history.items():
                if values and name not in ('active_triplet_acc', 'triplet_acc',
                                            'active_score_gap', 'score_gap', 'active_fraction'):
                    finite = [v for v in values if v == v]
                    if finite:
                        out[f'{prefix}{name}'] = float(sum(finite) / len(finite))
        aw = self._totals.get('active_weight', 0.0)
        vw = self._totals.get('valid_weight', 0.0)
        ah = self._totals.get('active_hits', 0.0)
        ag = self._totals.get('active_gap_num', 0.0)
        if aw:
            acc = ah / aw
            gap = ag / aw
        else:
            acc = gap = float('nan')
        out[f'{prefix}active_triplet_acc'] = float(acc)
        out[f'{prefix}triplet_acc'] = float(acc)
        out[f'{prefix}active_score_gap'] = float(gap)
        out[f'{prefix}score_gap'] = float(gap)
        out[f'{prefix}n_active_rows'] = float(self._totals.get('active_rows', 0.0))
        out[f'{prefix}n_active_source_rows'] = float(aw)
        out[f'{prefix}n_valid_rows'] = float(self._totals.get('valid_rows', 0.0))
        out[f'{prefix}n_valid_source_rows'] = float(vw)
        out[f'{prefix}n_precision_rows'] = float(vw * 3.0)
        out[f'{prefix}n_score_rows'] = float(vw * 3.0)
        out[f'{prefix}active_weight'] = float(aw)
        out[f'{prefix}valid_weight'] = float(vw)
        out[f'{prefix}active_fraction'] = float(aw / vw) if vw else 0.0
        out[f'{prefix}n_consistency_rows'] = float(self._totals.get('consistency_rows', 0.0))
        out[f'{prefix}n_duplicate_weighted_rows'] = float(
            self._totals.get('duplicate_weighted_rows', 0.0))
        for key, value in self._totals.items():
            if key.startswith('type_') and key.endswith('_rows'):
                out[f'{prefix}n_{key}'] = float(value)
            elif key.startswith('type_') and key.endswith('_weight'):
                gi = key[len('type_'):-len('_weight')]
                out[f'{prefix}type_{gi}_acc'] = float(
                    self._totals.get(f'type_{gi}_hits', 0.0) / value) if value else float('nan')
        return out

    def reset_history(self):
        """Drop per-window history and count accumulators."""
        self.loss_history = defaultdict(list)
        self._totals = defaultdict(float)
        self.last_counts = {}


__all__ = ['FrameTripletScorerLoss', 'signed_pointwise_targets']
