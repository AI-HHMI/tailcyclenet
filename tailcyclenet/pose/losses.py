"""Repo-local loss additions on top of tracktail's `TotalLoss`.

`TotalLoss` puts both of its visibility BCE terms inside the R == 3 branch, so it never
supervises visibility on a 2D session -- this adds that term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from tracktail.model.losses import TotalLoss


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
    """`TotalLoss` plus a 2D-only visibility term tracktail does not carry. `vis_loss_2d_weight`
    defaults to 0.1, a conservative nonzero weight; other keys reach `TotalLoss` unchanged, so an
    unknown key still raises `TypeError`.
    """

    def __init__(self, vis_loss_2d_weight: float = 0.1, lift_loss_weight: float = 0.5,
                 **kwargs):
        """Set the 2D visibility BCE weight; everything else reaches `TotalLoss` unchanged.

        Inputs: vis_loss_2d_weight -- weight of the 2D visibility term; defaults to 0.1.
                lift_loss_weight -- weight of the single-view lift head's per-camera loss
                    (`lift_head.lift_loss`, report 83 §9); inert without the head's output.

        `gamma` and `feature_loss_weight` default to 0 here, not to the library's 0.8 / 0.5: both
        gate on outputs the TrackerEncoder never emits (`coords_pred_iters`,
        `feature_planes_levels`), so the library defaults are inert and only made `TotalLoss`
        warn on every run. An old config that still states either one is passed through.
        """
        kwargs.setdefault('gamma', 0.0)
        kwargs.setdefault('feature_loss_weight', 0.0)
        super().__init__(**kwargs)
        self.vis_loss_2d_weight = vis_loss_2d_weight
        self.lift_loss_weight = lift_loss_weight

    def forward(self, model, outputs, coords_true, vis_true, vis_true_cams,
               vis_2d_true=None, cgroup=None, p2d=None, device=None):
        """Identical to `TotalLoss.forward`, plus one term. `vis_2d_true` is THIS repo's own
        field, kept on a separate wire from `vis_true_cams` (which RAISES without `vis_true` --
        exactly the state every 2D batch is in).

        The target `(b,t,n,cams,1)` is moved to cams-first and the prediction `(cams,b,t,n)` is
        given a matching trailing singleton, so both end up `(cams,b,t,n,1)`. The BCE is applied
        UNCONDITIONALLY: gating on `isfinite` would let a genuinely poisoned prediction silently
        revert to the detached placeholder instead of reaching the poison check. A non-finite
        term still attached to the graph would return NaN gradients to every upstream parameter;
        `run_batch` converts this exact message into a skipped step.
        """
        total = super().forward(model, outputs, coords_true, vis_true, vis_true_cams,
                                cgroup=cgroup, p2d=p2d, device=device)

        vis_loss_2d = torch.tensor(float('nan'), device=device)
        if self.vis_loss_2d_weight != 0 and vis_2d_true is not None \
                and 'vis_pred_2d' in outputs:
            target = vis_2d_true.permute(3, 0, 1, 2, 4)
            pred = outputs['vis_pred_2d'][..., None]
            vis_loss_2d = self.vis_loss_2d_weight * masked_bce_with_logits(pred, target)

        if vis_loss_2d.requires_grad and not torch.isfinite(vis_loss_2d):
            raise ValueError(
                "sub-loss(es) went non-finite while attached to the autograd graph: "
                "['vis_loss_2d']. This would be dropped from the forward total and would still "
                'return NaN gradients to every upstream parameter. The usual cause is a '
                'non-finite TARGET reaching a loss that does not mask its inputs.')

        self.loss_history['vis_loss_2d'].append(vis_loss_2d.item())
        if torch.isfinite(vis_loss_2d):
            total = total + vis_loss_2d
        return total + self._lift_term(outputs, coords_true, cgroup, p2d, device)

    def _lift_term(self, outputs, coords_true, cgroup, p2d, device):
        """The lift head's per-camera loss on a 3D item (`lift_head.lift_loss` x weight); 0 on a
        2D item, a `prob_2d_only` item (`p2d` given) or a forward without the head's output. A
        set-only output (`lift_groups` = M) regroups its (B,T,M*K,3) targets per query."""
        if (self.lift_loss_weight == 0 or 'lift_cam' not in outputs or p2d is not None
                or coords_true.shape[-1] != 3):
            return 0.0
        from .lift_head import lift_loss
        M = outputs.get('lift_groups')
        if M is not None:
            B, T, MK, _ = coords_true.shape
            coords_true = (coords_true.reshape(B, T, M, MK // M, 3).permute(0, 2, 1, 3, 4)
                           .reshape(B * M, T, MK // M, 3))
        term, shape, root = lift_loss(outputs['lift_cam'], coords_true, cgroup)
        term = self.lift_loss_weight * term
        self.loss_history['lift'].append(term.item())
        self.loss_history['lift_shape_mm'].append(shape.item())
        self.loss_history['lift_root'].append(root.item())
        return term if torch.isfinite(term) else 0.0
