"""tailcyclenet/losses.py -- `PoseLoss`, the repo-local 2D visibility term.

Scoped to the ONE thing this module adds. `TotalLoss`'s own machinery -- coords, 3D visibility,
smoothness -- is posetail's, exercised by its own suite and by the smoothness tests in
tests/test_train.py; not duplicated here. `TotalLoss.forward` itself is stubbed out in most of
these so each test isolates the new code from the library's, which is large enough to need a
real, shape-correct `outputs` dict to run at all.
"""
import pytest
import torch
from posetail.posetail.losses import BCELossVis, TotalLoss

from tailcyclenet.losses import PoseLoss, masked_bce_with_logits


# ----------------------------------------------------------------------------------------------
# masked_bce_with_logits
# ----------------------------------------------------------------------------------------------

def test_masked_bce_matches_the_library_idiom():
    """Duplicated from `BCELossVis._compute_loss` on purpose -- it is a private method, so this
    is the guard against the two drifting apart rather than an import."""
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 1, 1)
    target = torch.randint(0, 2, (2, 3, 4, 1, 1)).float()
    target[0, 0, 0] = float('nan')
    got = masked_bce_with_logits(pred, target)
    want = BCELossVis(weight=1.0)._compute_loss(pred, target)
    assert torch.allclose(got, want)


def test_masked_bce_returns_a_detached_nan_when_nothing_assessed():
    pred = torch.randn(2, 3, 1)
    target = torch.full((2, 3, 1), float('nan'))
    out = masked_bce_with_logits(pred, target)
    assert torch.isnan(out)
    assert not out.requires_grad, 'an inert term must be a detached leaf, not a graph node'


def test_masked_bce_gradient_survives_a_nan_target():
    """The same hazard `test_gradients_survive_unassessed_visibility` pins for the library term:
    a NaN reaching `binary_cross_entropy_with_logits` unmasked poisons the backward even though
    the forward loss still looks finite."""
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
    """Replace `TotalLoss.forward` so `PoseLoss.forward`'s `super().forward(...)` call is a stub.

    Patched on the CLASS, not the instance: `super().forward` resolves against the MRO at call
    time, so this redirects it without needing to build a real, shape-correct `outputs` dict for
    the library's own (large) forward.
    """
    calls = []

    def fake(self, model, outputs, coords_true, vis_true, vis_true_cams,
            cgroup=None, p2d=None, device=None):
        calls.append((vis_true, vis_true_cams))
        return torch.as_tensor(value)

    monkeypatch.setattr(TotalLoss, 'forward', fake)
    return calls


def test_pose_loss_at_default_weight_is_bit_identical_to_totalloss(monkeypatch):
    """An absent `vis_loss_2d_weight` key must reproduce today's arm exactly -- every run on
    record used bare `TotalLoss`, and this is the guarantee that a config without the key still
    does."""
    calls = _patch_total_forward(monkeypatch, 0.5)
    plain = PoseLoss()                      # vis_loss_2d_weight defaults to 0.0
    got = plain.forward(None, {'vis_pred_2d': torch.zeros(1, 1, 2, 3)},
                        torch.zeros(1, 2, 3, 2), None, None,
                        vis_2d_true=torch.ones(1, 2, 3, 1, 1), device='cpu')
    assert torch.equal(got, torch.tensor(0.5)), 'weight 0 must add nothing to the library total'
    assert len(calls) == 1, 'the library forward must still run, exactly once'


def test_pose_loss_is_inert_without_a_2d_target(monkeypatch):
    """No target, whatever the weight, means no term -- this is what lets a 3D batch (which never
    supplies `vis_2d_true`) share this class safely."""
    _patch_total_forward(monkeypatch, 0.25)
    loss_fn = PoseLoss(vis_loss_2d_weight=5.0)
    got = loss_fn.forward(None, {}, torch.zeros(1, 2, 3, 2), None, None,
                          vis_2d_true=None, device='cpu')
    assert got.item() == pytest.approx(0.25)


def test_pose_loss_adds_the_term_when_weighted(monkeypatch):
    _patch_total_forward(monkeypatch, 0.0)
    outputs = {'vis_pred_2d': torch.zeros(1, 1, 2, 3)}        # sigmoid(0) = 0.5 everywhere
    target = torch.ones(1, 2, 3, 1, 1)                        # every entry a real positive

    loss_fn = PoseLoss(vis_loss_2d_weight=5.0)
    got = loss_fn.forward(None, outputs, torch.zeros(1, 2, 3, 2), None, None,
                          vis_2d_true=target, device='cpu')
    assert got.item() > 0, 'a nonzero weight with a real target must move the total off 0'
    assert loss_fn.loss_history['vis_loss_2d'][-1] == pytest.approx(got.item())


def test_pose_loss_shapes_the_target_camera_first_to_match_the_prediction():
    """`vis_pred_2d` is (cams,b,t,n); `vis_2d_true` off the collate is (b,t,n,cams,1). A shape
    mistake here would either crash or silently broadcast the wrong axis together -- this pins
    the correct pairing by making ONE camera's target the opposite of the other's and checking
    the loss lands where it should be near-zero (camera 0, target agrees with a confident
    prediction) rather than near-max (camera 1, target disagrees)."""
    C = 2
    pred = torch.zeros(C, 1, 1, 1)                # cams,b,t,n
    pred[0] = 8.0                                  # confident "visible"
    pred[1] = 8.0
    target = torch.ones(1, 1, 1, C, 1)             # b,t,n,cams,1
    target[..., 1, :] = 0.0                        # camera 1's true label DISAGREES

    lo = masked_bce_with_logits(pred[0:1, ..., None], target.permute(3, 0, 1, 2, 4)[0:1])
    hi = masked_bce_with_logits(pred[1:2, ..., None], target.permute(3, 0, 1, 2, 4)[1:2])
    assert lo.item() < 0.01
    assert hi.item() > 5.0


def test_pose_loss_raises_on_a_genuinely_poisoned_term(monkeypatch):
    """A non-finite PREDICTION (not a masked-out target) must reach the same poison check
    `TotalLoss.forward` applies to its own stack (`losses.py:914-921`), and `run_batch` depends
    on the message containing 'non-finite' to convert this into a skipped step rather than a
    dead run."""
    _patch_total_forward(monkeypatch, 0.0)
    pred = torch.full((1, 1, 2, 3), float('inf'), requires_grad=True)
    target = torch.full((1, 2, 3, 1, 1), float('nan'))
    target[0, 0, 0, 0, 0] = 1.0             # one assessed entry, so the term attaches to the graph

    loss_fn = PoseLoss(vis_loss_2d_weight=1.0)
    with pytest.raises(ValueError, match='non-finite'):
        loss_fn.forward(None, {'vis_pred_2d': pred}, torch.zeros(1, 2, 3, 2), None, None,
                        vis_2d_true=target, device='cpu')


def test_pose_loss_rejects_an_unknown_training_losses_key():
    """The same guard a bare `TotalLoss(**losses)` gave: an unknown `[training.losses]` key must
    raise rather than silently training at a default."""
    with pytest.raises(TypeError):
        PoseLoss(not_a_real_key=1.0)
