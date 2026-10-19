"""The pose val pass and the train/val metric reducers.

`train/{2d,3d}/*` uses the batch's GT prompt; `val/*` is prompt-free and `val_self/*` re-queries
at the model's own frame-0 prediction -- do not plot them on one axis.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import distributed as dist_utils
from .model import share_scene
from .prompt import self_prompt


def brief(m: dict) -> str:
    """The interesting metric keys of a val dict, space-joined for a one-line print."""
    keys = ('mpjpe', 'mte', 'delta_x_avg', 'survival_rate', 'n_nonfinite')
    return ' '.join(f'{k}={m[k]:.4g}' for k in keys if k in m)


def to_device(batch, device):
    """Move a batch's views and cgroup onto `device`, non-blocking where safe."""
    views = [v.to(device, non_blocking=True) for v in batch.views]
    cgroup = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
              for c in batch.cgroup]
    return views, cgroup


SMOOTHNESS_KEYS = ('smoothness_3d_loss', 'smoothness_2d_loss')
# Scalar outputs shared with validation; delta_x_1..16 are per-batch arrays and are omitted by
# the same scalar filter. The visibility-based 2D values are diagnostics alongside the optional
# 2D visibility-loss head.
TRAIN_METRIC_KEYS = (
    'mte', 'delta_x_avg', 'occlusion_acc', 'avg_jaccard', 'survival_rate', 'mpjpe',
    'jaccard_1', 'jaccard_2', 'jaccard_4', 'jaccard_8', 'jaccard_16',
)
TRAIN_METRIC_MODES = ('2d', '3d')


@torch.no_grad()
def train_metrics(out, batch):
    """Return finite pose metrics for one accepted training forward.

    These are the same metrics as validation, but the training forward uses the batch's
    ground-truth prompt.  They are therefore an easier, prompt-conditioned diagnostic and
    must not be compared directly with ``val/`` or ``val_self/``.  Metrics are kept separate
    by mode because 2D values are pixels while 3D values are world units.
    """
    if 'vis_pred' not in out or 'coords_pred' not in out:
        return {}
    from tracktail.model.eval_metrics import get_eval_metrics, get_vis_true

    coords = batch.coords
    vis_true = batch.vis if batch.vis is not None else get_vis_true(coords)
    metrics = get_eval_metrics(vis_pred=out['vis_pred'], vis_true=vis_true,
                               coords_pred=out['coords_pred'], coords_true=coords, prefix='')
    return {k: float(v) for k, v in metrics.items()
            if k in TRAIN_METRIC_KEYS and np.ndim(v) == 0 and np.isfinite(v)}


def _clear_loss_history(loss_fn):
    """Clear recorded loss samples in place, preserving ``defaultdict`` lazy keys."""
    for samples in loss_fn.loss_history.values():
        samples.clear()


def drain_smoothness(loss_fn):
    """Drain one batch's smoothness samples as ``name -> (finite sum, finite count)``.

    The training loop calls this after every batch.  It can therefore discard a skipped batch
    before adding anything to the print interval, while the in-place clear prevents history growth
    without replacing ``TotalLoss.loss_history`` with a plain dict.
    """
    stats = {}
    for name in SMOOTHNESS_KEYS:
        samples = loss_fn.loss_history.get(name, ())
        finite = [float(value) for value in samples if np.isfinite(value)]
        if finite:
            stats[name] = (float(np.sum(finite)), len(finite))
    _clear_loss_history(loss_fn)
    return stats


def _interval_means(stats, keys, prefix, fabric=None):
    """Reduce finite sum/count pairs and return count-weighted interval means.

    The reduction is one fixed-size collective so every rank participates even when its local
    interval has no samples.  A metric is omitted only when its global count is zero; genuine
    zero-valued terms are retained.
    """
    keys = tuple(keys)
    if fabric is not None and fabric.world_size > 1:
        local = []
        for key in keys:
            total, count = stats.get(key, (0.0, 0))
            local.extend((total, float(count)))
        reduced = fabric.all_reduce(
            torch.tensor(local, dtype=torch.float64, device=fabric.device), reduce_op='sum')
        stats = {key: (float(reduced[2 * i]), int(round(float(reduced[2 * i + 1]))))
                 for i, key in enumerate(keys)}
    values = {}
    for key in keys:
        total, count = stats.get(key, (0.0, 0))
        if count:
            values[f'{prefix}{key}'] = total / count
    return values


def smoothness_log_values(stats, fabric=None):
    """Reduce smoothness samples and return interval means for the W&B payload."""
    return _interval_means(stats, SMOOTHNESS_KEYS, 'train/', fabric)


def metric_log_values(stats, fabric=None):
    """Return mode-separated training metric interval means for W&B."""
    flat = {f'{mode}/{key}': pair for mode in TRAIN_METRIC_MODES
            for key, pair in stats.get(mode, {}).items()}
    return _interval_means(flat,
                           tuple(f'{mode}/{key}' for mode in TRAIN_METRIC_MODES
                                 for key in TRAIN_METRIC_KEYS),
                           'train/', fabric)


@torch.no_grad()
def evaluate(model, batches, optimizer, device, fabric=None):
    """Val metrics on a FIXED set of windows, both regimes from one pass -> (prior_free, self).

    Batches are this rank's shard; the per-window dicts are gathered before reducing. `model` is
    the unwrapped module (no DDP hooks in a no-grad forward). Prior-free is the honest number
    (GT-derived priors are gated off); the self-prompted pass re-queries at the model's own
    frame-0 prediction and is the deployable regime. `optimizer.eval()` swaps in the averaged
    iterate; the `finally` puts `y` back.
    """
    from tracktail.model.eval_metrics import get_eval_metrics, get_vis_true

    model.eval()
    if hasattr(optimizer, 'eval'):
        optimizer.eval()
    free, prompted = [], []
    try:
        for batch in batches:
            views, cgroup = to_device(batch, device)
            kpt_ids, coords = batch.kpt_ids.to(device), batch.coords.to(device)
            mode = batch.sample_info['mode']
            vis_true = (batch.vis.to(device) if batch.vis is not None else get_vis_true(coords))

            def score(out):
                """The scalar eval metrics of one forward, as a plain dict."""
                m = get_eval_metrics(vis_pred=out['vis_pred'], vis_true=vis_true,
                                     coords_pred=out['coords_pred'], coords_true=coords, prefix='')
                return {k: float(v) for k, v in m.items() if np.ndim(v) == 0}

            with share_scene(model):
                bx = getattr(batch, 'box_prompt', None)
                bx = None if bx is None else bx.to(device)
                out = model(views, kpt_ids, cgroup, mode=mode, kpt_prior=None, prompt_time=None,
                            box_prompt=bx)
                free.append(score(out))
                prompted.append(score(self_prompt(model, views, kpt_ids, cgroup, mode, out,
                                                  box_prompt=bx)))
    finally:
        model.train()
        if hasattr(optimizer, 'train'):
            optimizer.train()
    return (_reduce(dist_utils.gather_metrics(fabric, free)),
            _reduce(dist_utils.gather_metrics(fabric, prompted)))


def _reduce(mets):
    """nanmean over windows, with the non-finite count alongside."""
    if not mets:
        return {}
    skipped = sum(not all(np.isfinite(v) for v in m.values()) for m in mets)
    out = {k: float(np.nanmean([m[k] for m in mets])) for k in mets[0]}
    out['n_batches'] = len(mets)
    out['n_nonfinite'] = skipped
    return out
