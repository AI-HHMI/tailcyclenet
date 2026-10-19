"""The scorer val pass and its metric reducers (triplet accuracy, pointwise AUROC/AP, localisation)."""
from __future__ import annotations

import numpy as np
import torch

from tracktail.datasets.scorer_corruption import GENERATORS

from .dataset import triplet_to_device
from .model import flatten_sequence_triplet


def scorer_loss(loss_fn, scores, precision, labels, trip=None, output_granularity='sequence'):
    """Call a scorer loss with the right shape and frame metadata.

    Sequence mode is intentionally a three-positional-argument call, preserving the installed loss
    algebra and all historical tests.  Frame mode keeps ``[B,T,K,3]`` until the local loss applies
    its active/duplicate masks; flattening here would make the frame labels impossible to audit.
    """
    if output_granularity == 'sequence':
        if scores.ndim == 3:
            scores, precision, labels = flatten_sequence_triplet(scores, precision, labels)
        return loss_fn(scores, precision, labels)
    if trip is None:
        raise ValueError('framewise scorer loss requires the triplet masks')
    return loss_fn(scores, precision, labels, triplet=trip)


def _per_type_accuracy(scores: torch.Tensor, fired: torch.Tensor,
                       active_mask: torch.Tensor | None = None) -> dict:
    """`P(good > bad)` broken out by corruption type at the active granularity.

    Legacy inputs are ``[K,3]``/``[K,G]``.  Framewise inputs are ``[B,T,K,3]`` (or unbatched
    ``[T,K,3]``) and ``[B,T,K,G]``.  No inactive/missing frame is allowed into a type denominator.
    """
    if scores.ndim == 2:
        correct = scores[:, 0] > scores[:, 1]
        type_mask = fired.bool()
        if active_mask is not None:
            type_mask = type_mask & active_mask.reshape(-1, 1).to(type_mask.device)
    else:
        correct = scores[..., 0] > scores[..., 1]
        type_mask = fired.bool()
        if active_mask is not None:
            am = active_mask.bool()
            if am.ndim == 2:
                am = am[None]
            type_mask = type_mask & am[..., None].to(type_mask.device)
        correct = correct.to(type_mask.device)
    out = {}
    for i, name in enumerate(GENERATORS):
        mask = type_mask[..., i]
        if bool(mask.any()):
            out[f'val/acc_{name}'] = float(correct[mask].float().mean())
    return out


def frame_pointwise_arrays(scores: torch.Tensor, labels: torch.Tensor,
                             trip: dict, type_index: int | None = None):
    """Return signed labels/scores/weights, optionally restricted to one corruption type."""
    from .losses import signed_pointwise_targets

    shape = scores.shape[:-1]
    device = scores.device
    def mask(name, fill):
        """Read one frame mask and broadcast its default to the score lattice."""
        value = trip.get(name)
        if value is None:
            return torch.full(shape, fill, dtype=torch.bool, device=device)
        return value.to(device=device, dtype=torch.bool)

    observed = mask('observed_mask', True)
    in_view = mask('in_view_mask', True)
    anchor = mask('anchor_observed_mask', True)
    far = mask('far_mask', False)
    near = mask('near_mask', False)
    ambiguous = mask('ambiguous_mask', False)
    rejected = mask('reference_rejected_mask', False)
    targets, target_mask = signed_pointwise_targets(
        labels, observed, in_view, anchor, far, near,
        anchor_label=trip.get('anchor_label'), ambiguous_mask=ambiguous,
        reference_rejected_mask=rejected)
    if type_index is not None:
        type_mask = trip.get('corruption_type_mask', trip.get('fired_frame'))
        if type_mask is None:
            return (np.empty(0), np.empty(0), np.empty(0))
        type_mask = type_mask.to(device=device, dtype=torch.bool)
        if type_mask.ndim == 3:
            type_mask = type_mask[:, None]
        target_mask &= type_mask[..., type_index, None]
    weight = trip.get('source_frame_weight')
    if weight is None:
        weight = torch.ones(shape, dtype=scores.dtype, device=device)
    else:
        weight = weight.to(device=device, dtype=scores.dtype)
    weights = weight[..., None].expand_as(target_mask)
    return (scores[target_mask].detach().cpu().numpy(),
            targets[target_mask].detach().cpu().numpy(),
            weights[target_mask].detach().cpu().numpy())


def weighted_binary_metrics(scores, targets, weights):
    """Weighted AUROC/AP for +/-1 synthetic labels, with a finite empty contract."""
    if len(scores) == 0:
        return float('nan'), float('nan')
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(targets) > 0
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores, positive, weights = scores[finite], positive[finite], weights[finite]
    pos = float(weights[positive].sum())
    neg = float(weights[~positive].sum())
    if not pos or not neg:
        return float('nan'), float('nan')
    order = np.argsort(scores, kind='stable')
    s, y, w = scores[order], positive[order], weights[order]
    auc_num = 0.0
    neg_before = 0.0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        pw = float(w[i:j][y[i:j]].sum())
        nw = float(w[i:j][~y[i:j]].sum())
        auc_num += pw * (neg_before + 0.5 * nw)
        neg_before += nw
        i = j
    auc = auc_num / (pos * neg)
    order = np.argsort(-scores, kind='stable')
    y, w = positive[order], weights[order]
    seen, ap_num = 0.0, 0.0
    for i in range(len(y)):
        seen += float(w[i])
        if y[i]:
            ap_num += float(w[i]) * (float((w[:i + 1] * y[:i + 1]).sum()) / seen)
    return float(auc), float(ap_num / pos)


def _frame_localization_counts(scores: torch.Tensor, trip: dict):
    """Count whether each corrupted keypoint's lowest bad frame lies in its far interval."""
    far = trip.get('far_mask')
    if far is None:
        return 0, 0
    far = far.to(device=scores.device, dtype=torch.bool)
    valid = (trip.get('observed_mask', torch.ones_like(far)).to(dtype=torch.bool)
             & trip.get('in_view_mask', torch.ones_like(far)).to(dtype=torch.bool)
             & trip.get('anchor_observed_mask', torch.ones_like(far)).to(dtype=torch.bool)
             & ~trip.get('ambiguous_mask', torch.zeros_like(far)).to(dtype=torch.bool)
             & ~trip.get('reference_rejected_mask', torch.zeros_like(far)).to(dtype=torch.bool))
    good = bad = 0
    for bi in range(scores.shape[0]):
        for ki in range(scores.shape[2]):
            candidate = valid[bi, :, ki]
            if bool(far[bi, :, ki].any()) and bool(candidate.any()):
                frame_scores = scores[bi, :, ki, 1].masked_fill(~candidate, torch.inf)
                ix = int(frame_scores.argmin())
                good += int(bool(far[bi, ix, ki]))
                bad += 1
    return good, bad


def evaluate(model, loader, loss_fn, device, max_batches: int,
             optimizer=None, output_granularity: str = 'sequence') -> dict:
    """The held-out synthetic pass, with count-based frame metrics in frame mode."""
    model.eval()
    averaged = optimizer is not None and hasattr(optimizer, 'eval')
    per_type: dict[str, list] = {}
    type_totals: dict[str, list[float]] = {}
    pointwise_chunks = []
    pointwise_type_chunks: dict[str, list] = {}
    localization = [0, 0]
    view_counts = {'out_of_view': 0, 'observed': 0, 'reference_rejected': 0,
                   'base_far': 0}
    frame_totals = {'active_weight': 0.0, 'active_correct': 0.0, 'active_gap': 0.0,
                    'valid_weight': 0.0}
    rejection_draws = 0
    seen = 0
    try:
        if averaged:
            optimizer.eval()
        with torch.no_grad():
            for trip in loader:
                if trip is None:
                    continue
                rejection_draws += int(trip.get('rejected_draws', 0))
                trip = triplet_to_device(trip, device)
                scores, precision, labels = model.score_triplet(trip)
                scorer_loss(loss_fn, scores, precision, labels, trip, output_granularity)
                if output_granularity == 'frame':
                    fired = trip.get('corruption_type_mask', trip.get('fired_frame'))
                    active = trip.get('active_mask')
                    if fired is not None and active is not None:
                        active = active.bool()
                        correct = scores[..., 0] > scores[..., 1]
                        gap = scores[..., 0] - scores[..., 1]
                        weight = trip.get('source_frame_weight')
                        if weight is None or tuple(weight.shape) != tuple(active.shape):
                            weight = torch.ones_like(active, dtype=scores.dtype)
                        else:
                            weight = weight.to(dtype=scores.dtype)
                        observed = trip.get('observed_mask', torch.ones_like(active))
                        in_view = trip.get('in_view_mask', torch.ones_like(active))
                        anchor = trip.get('anchor_observed_mask', torch.ones_like(active))
                        ambiguous = trip.get('ambiguous_mask', torch.zeros_like(active))
                        rejected = trip.get('reference_rejected_mask',
                                             torch.zeros_like(active))
                        valid = (observed.bool() & in_view.bool() & anchor.bool()
                                 & ~ambiguous.bool() & ~rejected.bool())
                        frame_totals['active_weight'] += float(weight[active].sum())
                        frame_totals['active_correct'] += float(weight[active & correct].sum())
                        frame_totals['active_gap'] += float((weight * gap)[active].sum())
                        frame_totals['valid_weight'] += float(weight[valid].sum())
                        for i, name in enumerate(GENERATORS):
                            mask = active & fired[..., i].bool()
                            if name not in type_totals:
                                type_totals[name] = [0.0, 0.0]
                            type_totals[name][0] += float(weight[mask].sum())
                            type_totals[name][1] += float((weight[mask]
                                                            * correct[mask].to(weight.dtype)).sum())
                        pointwise_chunks.append(frame_pointwise_arrays(scores, labels, trip))
                        for gi, name in enumerate(GENERATORS):
                            pointwise_type_chunks.setdefault(name, []).append(
                                frame_pointwise_arrays(scores, labels, trip, gi))
                        loc_hit, loc_n = _frame_localization_counts(scores, trip)
                        localization[0] += loc_hit
                        localization[1] += loc_n
                        observed = trip.get('observed_mask', torch.ones_like(active)).bool()
                        view_counts['observed'] += int(observed.sum())
                        view_counts['out_of_view'] += int((observed & ~trip.get(
                            'in_view_mask', torch.ones_like(active)).bool()).sum())
                        view_counts['reference_rejected'] += int(trip.get(
                            'reference_rejected_mask', torch.zeros_like(active)).bool().sum())
                        view_counts['base_far'] += int(trip.get(
                            'base_far_mask', trip.get('far_mask', torch.zeros_like(active))).bool().sum())
                else:
                    fired = trip['fired']
                    if scores.ndim == 2:
                        fired = fired[0]
                    for k, v in _per_type_accuracy(scores, fired).items():
                        per_type.setdefault(k, []).append(v)
                seen += 1
                if seen >= max_batches:
                    break
    finally:
        try:
            if averaged:
                optimizer.train()
        finally:
            model.train()
    out = {k: float(np.mean(v)) for k, v in per_type.items()}
    if output_granularity == 'frame':
        active_weight = frame_totals['active_weight']
        valid_weight = frame_totals['valid_weight']
        if active_weight:
            out['val/n_active_rows'] = active_weight
            out['val/active_triplet_acc'] = frame_totals['active_correct'] / active_weight
            out['val/active_score_gap'] = frame_totals['active_gap'] / active_weight
        if valid_weight:
            out['val/active_fraction'] = active_weight / valid_weight
        for name, (denom, hits) in type_totals.items():
            if denom:
                out[f'val/acc_{name}'] = hits / denom
        if pointwise_chunks:
            point_scores = np.concatenate([x[0] for x in pointwise_chunks])
            point_targets = np.concatenate([x[1] for x in pointwise_chunks])
            point_weights = np.concatenate([x[2] for x in pointwise_chunks])
            auc, ap = weighted_binary_metrics(point_scores, point_targets, point_weights)
            out['val/pointwise_auroc'] = auc
            out['val/pointwise_ap'] = ap
        for name, chunks in pointwise_type_chunks.items():
            typed = [chunk for chunk in chunks if len(chunk[0])]
            if not typed:
                continue
            typed_scores = np.concatenate([x[0] for x in typed])
            typed_targets = np.concatenate([x[1] for x in typed])
            typed_weights = np.concatenate([x[2] for x in typed])
            type_auc, type_ap = weighted_binary_metrics(
                typed_scores, typed_targets, typed_weights)
            out[f'val/pointwise_auroc_{name}'] = type_auc
            out[f'val/pointwise_ap_{name}'] = type_ap
        if localization[1]:
            out['val/localization'] = localization[0] / localization[1]
        if view_counts['observed']:
            out['val/out_of_view_fraction'] = (view_counts['out_of_view']
                                               / view_counts['observed'])
        if view_counts['base_far']:
            out['val/reference_rejected_fraction'] = (
                view_counts['reference_rejected'] / view_counts['base_far'])
    out['val/n_scored'] = float(seen)
    out['val/n_rejected_draws'] = float(rejection_draws)
    out['val/rejection_rate'] = float(rejection_draws / (rejection_draws + seen)
                                      if rejection_draws + seen else 0.0)
    return out


def gather_eval(fabric, metrics: dict) -> dict:
    """Reduce rank-local validation dictionaries to one deterministic world result."""
    if fabric is None or fabric.world_size <= 1:
        return metrics
    import torch.distributed as dist
    gathered = [None] * fabric.world_size
    dist.all_gather_object(gathered, metrics)
    gathered = [m for m in gathered if isinstance(m, dict)]
    if not gathered:
        return {}
    out = {}
    keys = set().union(*(m for m in gathered))
    for key in keys:
        vals = [float(m[key]) for m in gathered
                if key in m and np.isfinite(m[key])]
        if not vals:
            continue
        if key in {'val/n_scored', 'val/n_rejected_draws', 'val/n_active_rows'}:
            out[key] = float(sum(vals))
            continue
        if key == 'val/rejection_rate':
            n = sum(float(m.get('val/n_scored', 0.0)) for m in gathered)
            r = sum(float(m.get('val/n_rejected_draws', 0.0)) for m in gathered)
            out[key] = r / (r + n) if r + n else 0.0
            continue
        if key == 'val/active_fraction':
            active = sum(float(m.get('val/n_active_rows', 0.0)) for m in gathered)
            valid = sum(float(m.get('val/n_active_rows', 0.0)) /
                        float(m[key]) for m in gathered if m.get(key, 0.0))
            out[key] = active / valid if valid else float('nan')
            continue
        weights = [float(m.get('val/n_active_rows', m.get('val/n_scored', 0.0)))
                   for m in gathered if key in m and np.isfinite(m[key])]
        den = sum(weights)
        out[key] = (sum(v * w for v, w in zip(vals, weights)) / den
                    if den else float(np.mean(vals)))
    return out
