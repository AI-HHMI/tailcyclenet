"""Open-set ReID supervision (dev/plans/identity_bridge_and_reid.md §5).

Pure-function tests for `tailcyclenet/detector/reid_loss.py`: the per-box pooling and the
supervised-contrastive loss. The plan's design (owner-approved, report 55): positives come from
two independently-augmented views of the SAME frame item, so the loss must pull same-label rows
together (same animal, either view) and push different-label rows apart -- WITHIN a batch only,
no registry, no cross-session identity.
"""
import torch
import pytest

from tailcyclenet.detector.assign import assign as _assign
from tailcyclenet.detector.reid_loss import contrastive_loss, pool_embeddings_per_box


def _grid_anchors(n=8, stride=8):
    """(n*n, 3) anchors centred on an n x n grid at `stride` px spacing."""
    cy, cx = torch.meshgrid(torch.arange(n) * stride + stride / 2,
                            torch.arange(n) * stride + stride / 2, indexing='ij')
    return torch.stack([cx.reshape(-1), cy.reshape(-1),
                        torch.full((n * n,), float(stride))], -1)


def test_contrastive_pulls_same_label_and_pushes_different():
    """Same-label identical vectors must cost less than same-label orthogonal ones when a
    different-label competitor is present -- the whole point of the loss.
    """
    # 3 rows: 0 and 1 share a label but are ORTHOGONAL (bad); row 2 is a different label but
    # IDENTICAL to 0 (a hard negative). A good embedding is impossible here, so the loss must be
    # clearly positive -- and much larger than the well-separated version.
    hard = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    easy = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lab = torch.tensor([0, 0, 1])
    assert float(contrastive_loss(hard, lab)) > float(contrastive_loss(easy, lab)) + 1.0


def test_contrastive_identical_pair_is_ln3_not_ln4():
    """Four all-identical vectors, two labels: each anchor has ONE same-label positive and TWO
    different-label negatives at equal similarity, so SupCon costs exactly ln(3) -- NOT ln(4),
    which would mean the denominator wrongly includes the anchor itself.
    """
    v = torch.zeros(4, 8)
    loss = float(contrastive_loss(v, torch.tensor([0, 0, 1, 1])))
    assert loss == pytest.approx(1.0986122886681098, rel=1e-4)  # ln 3


def test_contrastive_zero_when_no_contrast_material():
    """Fewer than two valid rows, or no positive pair present, must cost zero -- an
    accidentally-identity-free batch costs nothing rather than NaN-ing a run.
    """
    assert float(contrastive_loss(torch.zeros(1, 4), torch.tensor([0]))) == 0.0
    assert float(contrastive_loss(torch.zeros(3, 4), torch.tensor([0, -1, 2]))) == 0.0


def test_contrastive_gradients_flow_and_push_the_right_way():
    """The loss must pull a same-label vector toward its positive: the gradient on the
    orthogonal same-label row points along the anchor's direction.
    """
    v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                     dtype=torch.float, requires_grad=True)
    loss = contrastive_loss(v, torch.tensor([0, 0, 1]))
    (grad,) = torch.autograd.grad(loss, v)
    assert torch.isfinite(grad).all()
    assert grad[1, 0] < 0, 'row 1 must be pulled toward +x (toward row 0, its positive)'


def test_pool_embeddings_averages_the_anchors_a_box_owns():
    """Pooled vector must be the mean of the anchors `assign` gives that GT box -- so identity
    and box describe the same physical object.
    """
    anchors = _grid_anchors()
    embed = torch.zeros(anchors.shape[0], 2)
    in_box0 = (anchors[:, 0] < 24) & (anchors[:, 1] < 24)
    in_box1 = (anchors[:, 0] >= 32) & (anchors[:, 1] >= 32)
    embed[in_box0] = torch.tensor([1.0, 0.0])
    embed[in_box1] = torch.tensor([0.0, 1.0])
    gt = torch.tensor([[0.0, 0.0, 24.0, 24.0], [32.0, 32.0, 64.0, 64.0]])
    pooled = pool_embeddings_per_box(embed, anchors, gt)
    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(pooled[1], torch.tensor([0.0, 1.0]))


def test_pool_embeddings_nan_box_yields_nan_row():
    """A GT box with no animal in this view must come out all-NaN, matching the loader's own
    NaN-box convention so the mask machinery downstream treats it as absent.
    """
    anchors = _grid_anchors()
    embed = torch.ones(anchors.shape[0], 2)
    gt = torch.full((2, 4), float('nan'))
    gt[0] = torch.tensor([0.0, 0.0, 24.0, 24.0])
    pooled = pool_embeddings_per_box(embed, anchors, gt)
    assert torch.isfinite(pooled[0]).all()
    assert torch.isnan(pooled[1]).all()


def test_pool_embeddings_matches_manual_mean_over_assign():
    """Cross-check against `assign` directly: the pooled value must equal the mean of embed over
    exactly the anchors assign returns for that box -- not a subtly different set.
    """
    anchors = _grid_anchors()
    rng = torch.Generator().manual_seed(0)
    embed = torch.rand(anchors.shape[0], 3, generator=rng)
    gt = torch.tensor([[4.0, 4.0, 40.0, 40.0], [36.0, 36.0, 60.0, 60.0]])
    pooled = pool_embeddings_per_box(embed, anchors, gt)
    for g in range(2):
        pos, gix = _assign(anchors, gt[g:g + 1])
        manual = embed[pos].mean(0)
        assert torch.allclose(pooled[g], manual), f'box {g} pooling disagrees with assign'


def test_pool_embeddings_and_contrastive_compose_end_to_end():
    """Two views of the same two animals: pooled embeddings from a trivial perfect head must
    give a small loss (same animal across views pulled together, different animals apart).
    """
    anchors = _grid_anchors()
    # view A: animal 0 in the top-left box, animal 1 bottom-right; view B: the same animals
    # shifted a little (an augmentation) but still inside the same boxes.
    gt_a = torch.tensor([[0.0, 0.0, 24.0, 24.0], [32.0, 32.0, 64.0, 64.0]])
    gt_b = torch.tensor([[2.0, 2.0, 26.0, 26.0], [34.0, 34.0, 62.0, 62.0]])
    embed = torch.zeros(anchors.shape[0], 2)
    in_tl = (anchors[:, 0] < 28) & (anchors[:, 1] < 28)
    in_br = (anchors[:, 0] >= 30) & (anchors[:, 1] >= 30)
    embed[in_tl] = torch.tensor([1.0, 0.0])
    embed[in_br] = torch.tensor([0.0, 1.0])
    va = pool_embeddings_per_box(embed, anchors, gt_a)
    vb = pool_embeddings_per_box(embed, anchors, gt_b)
    views = torch.cat([va, vb])                      # [A0, A1, B0, B1]
    labels = torch.tensor([0, 1, 0, 1])              # same animal across views shares a label
    assert float(contrastive_loss(views, labels)) < 0.05
