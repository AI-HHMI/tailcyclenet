"""Repo-local loss additions on top of posetail's `TotalLoss`.

`TotalLoss` puts both of its visibility BCE terms inside the R == 3 branch, so it never
supervises visibility on a 2D session -- this adds that term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from posetail.posetail.losses import TotalLoss


def masked_bce_with_logits(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE against a THREE-STATE target: 1 visible, 0 occluded, NaN not assessed. The mask must
    come BEFORE the substitute -- a NaN target multiplied by a zero mask still NaNs the backward
    (`0.0 * (sigmoid(x) - NaN)`). Returns a detached NaN when nothing was assessed, so the
    caller's poison check can tell an off switch from a real failure.
    """
    valid = torch.isfinite(target)
    if not valid.any():
        return torch.tensor(float('nan'), device=pred.device)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(pred, safe_target, reduction='none')
    valid_f = valid.float()
    return (loss * valid_f).sum() / valid_f.sum()


class PoseLoss(TotalLoss):
    """`TotalLoss` plus a 2D-only visibility term posetail does not carry. `vis_loss_2d_weight`
    defaults to 0.0, so an absent key is bit-identical to every run on record; other keys reach
    `TotalLoss` unchanged, so an unknown key still raises `TypeError`.
    """

    def __init__(self, vis_loss_2d_weight: float = 0.0, **kwargs):
        """Set the 2D visibility BCE weight; everything else reaches `TotalLoss` unchanged.

        Inputs: vis_loss_2d_weight -- weight of the 2D visibility term; 0.0 (the default)
                disables it, bit-identical to a run without the key.
        """
        super().__init__(**kwargs)
        self.vis_loss_2d_weight = vis_loss_2d_weight

    def forward(self, model, outputs, coords_true, vis_true, vis_true_cams,
               vis_2d_true=None, cgroup=None, p2d=None, device=None):
        """Identical to `TotalLoss.forward`, plus one term. `vis_2d_true` is THIS repo's own
        field, kept on a separate wire from `vis_true_cams` (which RAISES without `vis_true` --
        exactly the state every 2D batch is in).
        """
        total = super().forward(model, outputs, coords_true, vis_true, vis_true_cams,
                                cgroup=cgroup, p2d=p2d, device=device)

        vis_loss_2d = torch.tensor(float('nan'), device=device)
        if self.vis_loss_2d_weight != 0 and vis_2d_true is not None \
                and 'vis_pred_2d' in outputs:
            # outputs['vis_pred_2d'] is (cams,b,t,n); vis_2d_true is (b,t,n,cams,1). Move the
            # target to cams-first and give the prediction a matching trailing singleton.
            # target -> (cams,b,t,n,1); pred -> (cams,b,t,n,1).
            target = vis_2d_true.permute(3, 0, 1, 2, 4)
            pred = outputs['vis_pred_2d'][..., None]
            # UNCONDITIONAL: gating on `isfinite` would let a genuinely poisoned prediction
            # silently revert to the detached placeholder instead of reaching the poison check.
            vis_loss_2d = self.vis_loss_2d_weight * masked_bce_with_logits(pred, target)

        # A non-finite term still attached to the graph would return NaN gradients to every
        # upstream parameter; `run_batch` converts this exact message into a skipped step.
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
