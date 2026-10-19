"""The detector network: shapes, anchor order, normalisation, and the backbone contract."""
from __future__ import annotations

import torch

from tailcyclenet.detector.yolox import YOLOXNano


def test_forward_shapes_and_anchor_order():
    m = YOLOXNano(version='hybrid')
    x = torch.zeros(2, 3, 128, 160)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0], \
        'anchor_points must match forward()s flattening order exactly'
    assert boxes.shape == (2, anchors.shape[0], 4)
    assert (boxes[..., 2] >= boxes[..., 0]).all() and (boxes[..., 3] >= boxes[..., 1]).all()


# --- model and checkpoint ----------------------------------------------------------------


def test_norm_groups_always_divides_the_channel_count():
    """A GroupNorm whose count does not divide its channels is a constructor error."""
    from tailcyclenet.detector.yolox import norm_groups
    for c in list(range(1, 200)) + [24, 48, 96, 192]:
        g = norm_groups(c)
        assert 1 <= g <= c and c % g == 0


def test_groupnorm_has_no_running_statistics_and_no_batch_dependence():
    """Train and inference are the SAME computation (BatchNorm's statistics are what would make
    train-on-tiles / infer-on-frames unsafe), and one item's output ignores the rest of the batch
    (so a resolution sweep that changes the batch measures resolution, not normalisation)."""
    torch.manual_seed(0)
    m = YOLOXNano(version='hybrid')
    assert list(m.named_buffers()) == []
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())
    x = torch.rand(3, 3, 96, 128)
    m.train()
    a = m(x)[0]
    m.eval()
    with torch.no_grad():
        torch.testing.assert_close(a, m(x)[0])
        torch.testing.assert_close(m(x[1:2])[0], m(x)[0][1:2], rtol=1e-4, atol=1e-5)


def test_p2_true_adds_a_stride_4_level():
    """`p2=True` adds a stride-4 level whose anchors match the forward's flattening exactly."""
    base = YOLOXNano(version='hybrid')
    m = YOLOXNano(version='hybrid', p2=True)
    assert m.STRIDES == (4, 8, 16, 32)
    assert base.STRIDES == (8, 16, 32), 'the OTHER instance must be unaffected'
    x = torch.zeros(1, 3, 128, 160)
    obj, boxes = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0]
    assert set(anchors[:, 2].tolist()) == {4.0, 8.0, 16.0, 32.0}
    n_base = sum(p.numel() for p in base.parameters())
    n_p2 = sum(p.numel() for p in m.parameters())
    assert n_p2 > n_base


def test_hybrid_backbone_shape_contract_and_finite_grads():
    from tailcyclenet.detector.hybrid import HybridBackbone

    bb = HybridBackbone(p2=True)
    H, W = 128, 192
    feats = bb(torch.rand(1, 3, H, W))
    assert len(feats) == len(bb.out_channels) == 4
    for f, s, c in zip(feats, (4, 8, 16, 32), bb.out_channels):
        assert f.shape == (1, c, H // s, W // s)
    sum(f.float().sum() for f in feats).backward()
    for name, p in bb.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), f'{name}: non-finite grad'


def test_gap_levers_build_the_head_and_neck_they_name():
    """`shared_head=False` adds a separate objectness tower; `fpn_upsample` defaults to nearest at
    the constructor (the shipped config sets bilinear)."""
    shared = YOLOXNano(version='hybrid', shared_head=True)
    unshared = YOLOXNano(version='hybrid', shared_head=False)
    assert not hasattr(shared.head, 'obj_convs') and hasattr(unshared.head, 'obj_convs')
    assert sum(p.numel() for p in unshared.parameters()) > \
        sum(p.numel() for p in shared.parameters())
    assert shared.neck.fpn_upsample == 'nearest'
    model = YOLOXNano(version='hybrid', fpn_upsample='bilinear')
    obj, boxes = model(torch.rand(1, 3, 64, 64))
    assert torch.isfinite(obj).all() and torch.isfinite(boxes).all()


def test_convnext_meets_the_backbone_contract():
    """Strides 4/8/16/32 at exactly H//s x W//s; the ImageNet trunk is what gets the backbone LR
    scale. Architecture only -- no weight download in CI."""
    from tailcyclenet.detector.train import _pretrained_params
    from tailcyclenet.detector.yolox import YOLOXNano
    m = YOLOXNano(version='convnext-t', p2=True)
    feats = m.backbone(torch.rand(1, 3, 96, 160))
    assert [tuple(f.shape[2:]) for f in feats] == [(96 // s, 160 // s) for s in (4, 8, 16, 32)]
    assert [f.shape[1] for f in feats] == list(m.backbone.out_channels)
    trunk = {id(p) for p in _pretrained_params(m)}
    assert trunk == {id(p) for p in m.backbone.features.parameters()}
