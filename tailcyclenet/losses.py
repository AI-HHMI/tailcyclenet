"""Repo-local loss additions on top of posetail's `TotalLoss`.

posetail 0.3.5 is pinned from PyPI and `../posetail-next` is read-only reference, so a term the
library does not carry has to live here rather than upstream. Only one exists today: per-camera
2D visibility supervision.

`TotalLoss.forward` branches on `coords_true.shape[-1]` (`losses.py:368`). The `R == 2` block
computes `coords_loss_2d`, `coords_softmax_2d` and the two smoothness terms; both visibility BCE
terms live inside the `R == 3` `else`. So a plain `TotalLoss` never supervises visibility on a 2D
session at all -- the `vis_pred` / `vis_pred_2d` heads still exist in the model's output dict, but
nothing trains them, which is why `--vis-thresh` was documented as "cannot work" on a 2D root. See
`dev/plans/status_consistency_and_occlusion.md` for the survey behind this file.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from posetail.posetail.losses import TotalLoss


def masked_bce_with_logits(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE against a THREE-STATE target: 1 visible, 0 occluded, NaN not assessed.

    Exactly `BCELossVis._compute_loss`'s idiom (posetail `losses.py:974-1004`), duplicated rather
    than imported because that method is private and the mask-THEN-substitute order is the whole
    point: a NaN target must never reach `binary_cross_entropy_with_logits`, even multiplied by a
    zero mask afterwards -- the backward of a masked-out NaN element is still NaN
    (`0.0 * (sigmoid(x) - NaN) = NaN`), which poisons every upstream parameter's gradient while
    the forward loss still looks finite and the printed curve falls normally. Measured on the
    library's own version of this hazard: 36 of 40 training steps silently skipped.

    Returns a detached NaN, not a graph node, when nothing in `target` was assessed -- the same
    "intentionally inert" shape `TotalLoss._nan()` uses, so the caller's own poison check
    (`requires_grad and not isfinite`) can tell an off switch from a real failure.
    """
    valid = torch.isfinite(target)
    if not valid.any():
        return torch.tensor(float('nan'), device=pred.device)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(pred, safe_target, reduction='none')
    valid_f = valid.float()
    return (loss * valid_f).sum() / valid_f.sum()


class PoseLoss(TotalLoss):
    """`TotalLoss` plus a 2D-only visibility term posetail does not carry.

    `vis_loss_2d_weight` defaults to 0.0, so an absent key in `[training.losses]` reproduces
    today's arm bit-for-bit (`tests/test_losses.py` pins this) and every run on record stays
    comparable. `scripts/train.py` builds this class in place of `TotalLoss`; every other
    `[training.losses]` key reaches `TotalLoss.__init__` unchanged, so an unknown key still
    raises `TypeError` -- the same guard the bare class gave, not a weaker one.
    """

    def __init__(self, vis_loss_2d_weight: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.vis_loss_2d_weight = vis_loss_2d_weight

    def forward(self, model, outputs, coords_true, vis_true, vis_true_cams,
               vis_2d_true=None, cgroup=None, p2d=None, device=None):
        """Identical to `TotalLoss.forward`, plus one term.

        `vis_2d_true` is THIS repo's own field and must never be conflated with posetail's
        `vis_true_cams`: that one only exists inside `TotalLoss`'s R == 3 branch and RAISES if
        supplied without `vis_true` (both-or-neither, `losses.py:358`) -- exactly the state every
        2D batch is in, since there is no 3D layer to populate `vis_true` from. `run_batch` keeps
        the two on separate wires for exactly this reason: `vis_true`/`vis_true_cams` always
        travel together (both None on a 2D batch), and `vis_2d_true` travels alone.
        """
        total = super().forward(model, outputs, coords_true, vis_true, vis_true_cams,
                                cgroup=cgroup, p2d=p2d, device=device)

        vis_loss_2d = torch.tensor(float('nan'), device=device)
        if self.vis_loss_2d_weight != 0 and vis_2d_true is not None \
                and 'vis_pred_2d' in outputs:
            # outputs['vis_pred_2d'] is (cams, b, t, n) (tracker_encoder.py's result_dict).
            # vis_2d_true is (b, t, n, cams, 1) off the collate (custom_collate's trailing [...,
            # None]). Move the target to cams-first and give the prediction a matching trailing
            # singleton -- the exact shapes TotalLoss's own R == 3 branch compares
            # (losses.py:547-552), so the two are simply stacked differently, not resqueezed.
            target = vis_2d_true.permute(3, 0, 1, 2, 4)             # -> (cams,b,t,n,1)
            pred = outputs['vis_pred_2d'][..., None]                # -> (cams,b,t,n,1)
            # UNCONDITIONAL, not gated on `isfinite` -- gating here would mean a genuinely
            # poisoned result (finite target, non-finite PREDICTION, e.g. an upstream numerical
            # blowup) silently reverts to the detached "nothing to supervise" placeholder instead
            # of ever reaching the poison check below, which is the exact failure mode that check
            # exists to catch. `masked_bce_with_logits` itself already guards the one thing that
            # is legitimately supposed to produce NaN here (a NaN TARGET), and returns a detached
            # leaf for it -- so `vis_loss_2d.requires_grad` is the correct discriminator between
            # "nothing assessed" (False) and "something is actually wrong" (True).
            vis_loss_2d = self.vis_loss_2d_weight * masked_bce_with_logits(pred, target)

        # THE SAME POISON CHECK `TotalLoss.forward` applies to its own stack
        # (`losses.py:914-921`): a term that goes non-finite while still attached to the graph
        # would otherwise be silently dropped from `total` below and still return NaN gradients
        # to every upstream parameter. `run_batch` converts this exact message into a skipped
        # step -- it must say 'non-finite' or that conversion misses it.
        if vis_loss_2d.requires_grad and not torch.isfinite(vis_loss_2d):
            raise ValueError(
                "sub-loss(es) went non-finite while attached to the autograd graph: "
                "['vis_loss_2d']. This would be dropped from the forward total and would still "
                'return NaN gradients to every upstream parameter. The usual cause is a '
                'non-finite TARGET reaching a loss that does not mask its inputs.')

        self.loss_history['vis_loss_2d'].append(vis_loss_2d.item())
        if torch.isfinite(vis_loss_2d):
            total = total + vis_loss_2d
        return total
