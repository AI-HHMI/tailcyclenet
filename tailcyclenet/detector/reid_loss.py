"""Open-set ReID supervision for the detector's per-anchor embedding head (`embed_dim`).

The embedding head (report 55, commit `65074fe`) emits a raw D-vector per anchor. This module
turns those into the thing §5.3.1 actually consumes -- one vector per detected animal, pooled
over the anchors that box OWNS -- and supervises them with a supervised-contrastive loss whose
positives come from the OWNER-APPROVED source of identity: two independently-augmented views of
the SAME (session, frame, camera) item, which `BoxDataset.__getitem__` produces on demand
(report 55's "single-frame trick": fresh `default_rng(None)` per call at `data.py:862`), plus
cross-camera views of the same frame. NO cross-session identity, NO registry -- the label only
has to be consistent WITHIN a batch, and within a batch the same physical animal always carries
the same (group, animal_row) key, which is exactly what two draws of one index give.

This file is deliberately PURE: no wiring, no sampler, no training-loop changes. It is the loss
and pooling half only, so it can be unit-tested in isolation before any of the pipeline that
feeds it exists.
"""
from __future__ import annotations

import torch

from .assign import assign

# Default supervised-contrastive temperature (Khosla et al., SupCon).
TEMPERATURE = 0.1


def pool_embeddings_per_box(embed, anchors, gt_boxes):
    """Mean-pool per-anchor embeddings over the anchors `assign` gives each GT box.

    Inputs:
        embed -- (A, D) per-anchor embedding vectors, one row per anchor in `anchors`' order.
        anchors -- (A, 3) of (cx, cy, stride).
        gt_boxes -- (G, 4) xyxy; NaN rows (no animal in this view) yield an all-NaN pooled row.
    Outputs:
        (G, D) pooled vectors, one per GT box -- the embedding of the animal IN that box, pooled
        over the SAME anchors the box loss supervises, so identity and box describe the same
        physical object.
    Notes:
        A row with no positive anchor is all-NaN -- the same convention a NaN GT box sends the
        rest of the detector. Straight mean pooling, no learned attention: the head itself must
        concentrate identity info on the anchors it owns; attention here would only add a second
        thing that can be wrong.
    """
    G, D = gt_boxes.shape[0], embed.shape[-1]
    out = torch.zeros((G, D), dtype=embed.dtype, device=embed.device)
    pos, gix = assign(anchors, gt_boxes)
    if not pos.numel():
        return torch.full((G, D), float('nan'), dtype=embed.dtype, device=embed.device)
    vals = embed[pos]
    count = torch.zeros(G, dtype=embed.dtype, device=embed.device)
    out.index_add_(0, gix, vals)
    count.index_add_(0, gix, torch.ones_like(gix, dtype=embed.dtype))
    have = count > 0
    out[have] = out[have] / count[have, None]
    out[~have] = float('nan')
    return out


def contrastive_loss(vectors, labels, temperature=TEMPERATURE):
    """Supervised contrastive loss over pooled per-box embeddings.

    Inputs:
        vectors -- (N, D) per-box embeddings. Not required to be pre-normalised: normalisation
            is applied here because the head emits raw vectors and the norm is not a learnt
            degree of freedom the loss should care about.
        labels -- (N,) int: rows with the SAME label are positives, different labels negatives.
            `-1` marks a row with NO valid identity (no animal / no positive anchor) and excludes
            it from both the anchor and the positive set.
        temperature -- the contrastive temperature.
    Outputs:
        Scalar (mean over anchors that have at least one positive). Zero when there is nothing
        to contrast (fewer than two valid rows, or no positive pair present) -- so an
        accidentally-identity-free batch costs nothing rather than NaN-ing the run.
    """
    keep = labels >= 0
    if keep.sum() < 2:
        return torch.zeros((), dtype=vectors.dtype, device=vectors.device)
    v = vectors[keep] / vectors[keep].norm(dim=1, keepdim=True).clamp_min(1e-8)
    lab = labels[keep]
    N = v.shape[0]
    sim = v @ v.T / temperature
    eye = torch.eye(N, dtype=torch.bool, device=v.device)
    pos = (lab[:, None] == lab[None, :]) & ~eye
    if not pos.any():
        return torch.zeros((), dtype=vectors.dtype, device=v.device)
    sim_masked = sim.masked_fill(eye, float('-inf'))
    row_max = sim_masked.max(dim=1, keepdim=True).values.clamp_min(-1e9)
    denom = torch.logsumexp(sim_masked - row_max, dim=1) + row_max[:, 0]
    with torch.no_grad():
        anchor_ok = pos.any(dim=1)
    safe_sim = sim - denom[:, None]
    num = (safe_sim * pos.to(safe_sim.dtype)).sum(dim=1) / pos.sum(dim=1).clamp_min(1.0)
    loss = -(num[anchor_ok]).mean()
    return loss
