"""`PoseLoss`'s repo-local 2D visibility term; `TotalLoss.forward` is stubbed to isolate it."""
import pytest
import torch
from tracktail.model.losses import BCELossVis, TotalLoss

from tailcyclenet.losses import PoseLoss, masked_bce_with_logits


# ----------------------------------------------------------------------------------------------
# masked_bce_with_logits
# ----------------------------------------------------------------------------------------------

def test_masked_bce_matches_the_library_idiom():
    """Duplicated from `BCELossVis._compute_loss` on purpose -- it is private, so this guards
    against the two drifting apart rather than an import."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 1, 1)
    target = torch.randint(0, 2, (2, 3, 4, 1, 1)).float()
    target[0, 0, 0] = float('nan')
    got = masked_bce_with_logits(pred, target)
    want = BCELossVis(weight=1.0)._compute_loss(pred, target)
    assert torch.allclose(got, want)


def test_masked_bce_nan_handling():
    """Nothing assessed -> a detached NaN (inert, not a graph node); a NaN target is masked
    before the BCE so the backward stays finite."""
    out = masked_bce_with_logits(torch.randn(2, 3, 1), torch.full((2, 3, 1), float('nan')))
    assert torch.isnan(out) and not out.requires_grad
    pred = torch.zeros(2, 4, 1, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 1)).float()
    target[0, 1] = float('nan')
    loss = masked_bce_with_logits(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


# ----------------------------------------------------------------------------------------------
# PoseLoss
# ----------------------------------------------------------------------------------------------

def _patch_total_forward(monkeypatch, value):
    """Stub `TotalLoss.forward` on the CLASS (`super().forward` resolves via the MRO)."""
    calls = []

    def fake(self, model, outputs, coords_true, vis_true, vis_true_cams,
            cgroup=None, p2d=None, device=None):
        calls.append((vis_true, vis_true_cams))
        return torch.as_tensor(value)

    monkeypatch.setattr(TotalLoss, 'forward', fake)
    return calls


def test_pose_loss_is_inert_at_zero_weight_or_without_a_2d_target(monkeypatch):
    """Weight 0 = bare `TotalLoss`; no target (every 3D batch) = no term at any weight."""
    calls = _patch_total_forward(monkeypatch, 0.5)
    got = PoseLoss(vis_loss_2d_weight=0.0).forward(
        None, {'vis_pred_2d': torch.zeros(1, 1, 2, 3)}, torch.zeros(1, 2, 3, 2), None, None,
        vis_2d_true=torch.ones(1, 2, 3, 1, 1), device='cpu')
    assert torch.equal(got, torch.tensor(0.5)) and len(calls) == 1
    got = PoseLoss(vis_loss_2d_weight=5.0).forward(None, {}, torch.zeros(1, 2, 3, 2), None, None,
                                                   vis_2d_true=None, device='cpu')
    assert got.item() == pytest.approx(0.5)


def test_pose_loss_adds_the_term_when_weighted(monkeypatch):
    _patch_total_forward(monkeypatch, 0.0)
    outputs = {'vis_pred_2d': torch.zeros(1, 1, 2, 3)}        # sigmoid(0) = 0.5 everywhere
    target = torch.ones(1, 2, 3, 1, 1)                        # every entry a real positive

    loss_fn = PoseLoss(vis_loss_2d_weight=5.0)
    got = loss_fn.forward(None, outputs, torch.zeros(1, 2, 3, 2), None, None,
                          vis_2d_true=target, device='cpu')
    assert got.item() > 0, 'a nonzero weight with a real target must move the total off 0'
    assert loss_fn.loss_history['vis_loss_2d'][-1] == pytest.approx(got.item())


def test_pose_loss_raises_on_a_genuinely_poisoned_term(monkeypatch):
    """A non-finite PREDICTION raises 'non-finite', which `run_batch` turns into a skipped step."""
    _patch_total_forward(monkeypatch, 0.0)
    pred = torch.full((1, 1, 2, 3), float('inf'), requires_grad=True)
    target = torch.full((1, 2, 3, 1, 1), float('nan'))
    target[0, 0, 0, 0, 0] = 1.0             # one assessed entry, so the term attaches to the graph

    loss_fn = PoseLoss(vis_loss_2d_weight=1.0)
    with pytest.raises(ValueError, match='non-finite'):
        loss_fn.forward(None, {'vis_pred_2d': pred}, torch.zeros(1, 2, 3, 2), None, None,
                        vis_2d_true=target, device='cpu')


def test_pose_loss_rejects_unknown_keys_and_warns_only_on_inert_weights(capsys):
    """An unknown `[training.losses]` key raises; the defaults carry no inert weight (gamma and
    feature_loss_weight used to print '[warn] ... silently inert' on every run)."""
    with pytest.raises(TypeError):
        PoseLoss(not_a_real_key=1.0)
    PoseLoss()._warn_unconsumed_weights(object(), {})
    assert '[warn]' not in capsys.readouterr().out
    PoseLoss(gamma=0.8)._warn_unconsumed_weights(object(), {})
    assert 'gamma' in capsys.readouterr().out
