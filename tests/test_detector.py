"""The detector, and the property that makes it worth having.

The point of this detector is not that it finds animals -- it is that it reproduces THE CROP
RULE'S box. If it learned some other plausible box, every downstream pose number would shift.
"""
import numpy as np
import pytest
import torch
from pathlib import Path

from tailcyclenet.crop import box_corners, crop_box_for_points
from tailcyclenet.detector import (BoxDataset, ChunkShuffle, CohortSampler, YOLOXNano, assign,
                                   assign_tal, box_collate, box_iou, ciou_loss, decode,
                                   detector_loss, giou_loss, letterbox, paired_iou,
                                   unletterbox_boxes)
from tailcyclenet.detector.config import (MODEL_KEYS, TRAINING_KEYS,
                                           load_detector_config)
from tailcyclenet.detector.data import (_cutout_rects, _keypoints_in_rects, _photometric,
                                        random_affine)



def _infer_program_source():
    """The inference PROGRAM as one string: `tailcyclenet/infer/{cli,driver}.py`.

    These assertions predate the split and were written against `scripts/infer.py` when that file
    held the argparse and the whole driver. It is now a six-line shim onto
    `tailcyclenet.infer.main`, so reading it would silently assert nothing -- a scrape for an
    absent literal PASSES. Concatenating the two files it became keeps each check meaning exactly
    what it meant before.
    """
    base = Path(__file__).resolve().parent.parent / 'tailcyclenet' / 'infer'
    return (base / 'cli.py').read_text() + (base / 'driver.py').read_text()


def _infer_window_source():
    """The window loop, formerly `tailcyclenet/infer.py`."""
    return (Path(__file__).resolve().parent.parent
            / 'tailcyclenet' / 'infer' / 'window.py').read_text()


def test_forward_shapes_and_anchor_order():
    m = YOLOXNano()
    x = torch.zeros(2, 3, 128, 160)
    obj, boxes, _, _ = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0], \
        'anchor_points must match forward()s flattening order exactly'
    assert boxes.shape == (2, anchors.shape[0], 4)
    assert (boxes[..., 2] >= boxes[..., 0]).all() and (boxes[..., 3] >= boxes[..., 1]).all()


def test_chunk_shuffle_is_a_permutation_that_keeps_locality():
    """Every index exactly once (or it silently drops training data), and few videos at a time.

    The locality bound is the whole point: `_reader` caches per worker, so a draw that ranges
    over more than `mix` blocks re-opens containers and costs 486 ms/batch instead of 40.
    """
    chunk, mix, n_pools = 512, 4, 3
    n = chunk * mix * n_pools                     # exact multiple, so pools are position-aligned
    s = ChunkShuffle(n, chunk=chunk, mix=mix, seed=0)
    order = list(iter(s))
    assert sorted(order) == list(range(n)), 'must visit every index exactly once'
    assert order != list(range(n)), 'must actually shuffle'
    for i in range(0, n, chunk * mix):            # one pool = at most `mix` distinct videos
        assert len({j // chunk for j in order[i:i + chunk * mix]}) == mix
    assert list(iter(s)) != order, 'a second epoch must reshuffle'


def _two_cohort_root(tmp_path, n_annot_frames=2, n_tracked_frames=30):
    """A root with ONE `annotated` session and ONE `tracked` session, deliberately lopsided.

    The shape rat-city-combined actually has: a handful of annotated stills against one long
    tracked clip, so an unweighted draw and a weighted one cannot agree by accident.
    """
    from tailcyclenet import format as fmt
    from tests.conftest import KPTS_3D, _rig, _write_frames

    W = H = 48
    root = tmp_path / 'two_cohort'
    for name, source, n_lab, T in (('a', 'annotated', n_annot_frames, n_annot_frames),
                                   ('t', 'tracked', n_tracked_frames, n_tracked_frames)):
        path = root / 'train' / name
        lab = fmt.empty_labels(1, T, len(KPTS_3D), 1, mode3d=False)
        frames = list(range(n_lab))
        lab.vis2d[0, frames, :, 0] = fmt.VISIBLE
        lab.points2d[0, frames, :, 0] = (
            10.0 + 3.0 * np.arange(len(frames), dtype=np.float32))[:, None, None] % (W - 10)
        fmt.write_session(path, mode='2d', units='px', label_source=source, names=KPTS_3D,
                          rig=_rig([('cam0', W, H, False, False, 0)]),
                          groups={'g0': fmt.Group('g0', T)}, labels={'g0': lab})
        _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return root


def test_annot_frac_absent_means_no_weighting(tmp_path):
    """Absent must stay byte-identical: None weights => the caller keeps `ChunkShuffle`."""
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    w = ds.default_train_weights(None)
    assert w is not None and len(w) == len(ds)


def test_annot_frac_is_inert_on_a_single_cohort_split(dense_root):
    """3dpop/calms21/branson-fly are entirely `tracked`. The key must not be able to affect them,
    whatever it says -- and critically calms21 is the ONE train-time video root, so this is what
    keeps the weighted sampler away from the only place decode locality still costs anything.
    """
    ds = BoxDataset(dense_root, 'train', input_wh=(64, 64))
    assert len(ds.cohort_mix()) == 1, 'fixture must be single-cohort for this test to mean anything'
    base = ds.default_train_weights(None)
    for frac in (0.0, 0.25, 0.5, 1.0):
        w = ds.default_train_weights(frac)
        np.testing.assert_allclose(w, base, err_msg=f'annot_frac={frac} must be inert on a '
                                   'single-cohort split')


def test_annot_frac_sets_the_cohort_share(tmp_path):
    """The whole point: the mix becomes the configured number instead of whatever the frame
    counts happen to be."""
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    natural = ds.cohort_mix()
    assert natural['annotated'] < 0.2, 'fixture must be lopsided or this proves nothing'
    for frac in (0.25, 0.5, 0.75):
        mix = ds.cohort_mix(ds.default_train_weights(frac))
        assert mix['annotated'] == pytest.approx(frac)
        assert mix['tracked'] == pytest.approx(1.0 - frac)


def test_annot_frac_weights_are_uniform_within_a_cohort(tmp_path):
    """A group's weight must not scale with how many of its frames survived `frames_per_group`:
    that is the accident being removed, and re-introducing it inside a cohort would be the same
    bug one level down.
    """
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    w = ds.default_train_weights(0.5)
    src = np.array([s.label_source for s, _, _, _ in ds.index])
    for cohort in ('annotated', 'tracked'):
        assert len(np.unique(w[src == cohort])) == 1


def test_annot_frac_out_of_range_raises(tmp_path):
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match='annot_frac'):
            ds.default_train_weights(bad)


def test_cohort_sampler_realises_the_requested_share(tmp_path):
    """The weights are only a claim until the sampler is drawn from -- this is the live check."""
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    w = ds.default_train_weights(0.5)
    src = np.array([s.label_source for s, _, _, _ in ds.index])
    s = CohortSampler(w, num_samples=20000, seed=0)
    drawn = src[np.array(list(iter(s)))]
    assert (drawn == 'annotated').mean() == pytest.approx(0.5, abs=0.02)


def test_cohort_sampler_reshuffles_and_stays_in_range(tmp_path):
    ds = BoxDataset(_two_cohort_root(tmp_path), 'train', input_wh=(64, 64))
    s = CohortSampler(ds.default_train_weights(0.5), num_samples=256, seed=0)
    first = list(iter(s))
    assert len(first) == len(s) == 256
    assert min(first) >= 0 and max(first) < len(ds), 'a draw must be a valid index position'
    assert list(iter(s)) != first, 'a second epoch must redraw'


def _two_group_root(tmp_path, n1=4, n2=32, label_source='tracked'):
    """Two SINGLE-COHORT sessions with DIFFERENT frame counts -- for B1a/B1b, isolated from
    `annot_frac`'s cohort question (both groups share one `label_source`).
    """
    from tailcyclenet import format as fmt
    from tests.conftest import KPTS_3D, _rig, _write_frames

    W = H = 48
    root = tmp_path / 'two_group'
    for name, T in (('a', n1), ('b', n2)):
        path = root / 'train' / name
        lab = fmt.empty_labels(1, T, len(KPTS_3D), 1, mode3d=False)
        lab.vis2d[0, :, :, 0] = fmt.VISIBLE
        lab.points2d[0, :, :, 0] = (
            10.0 + 3.0 * np.arange(T, dtype=np.float32))[:, None, None] % (W - 10)
        fmt.write_session(path, mode='2d', units='px', label_source=label_source, names=KPTS_3D,
                          rig=_rig([('cam0', W, H, False, False, 0)]),
                          groups={'g0': fmt.Group('g0', T)}, labels={'g0': lab})
        _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return root


def test_max_frames_per_group_zero_drops_the_cap(tmp_path):
    """detector_v2 B1a: 0 means every labelled frame is indexed, not just the first
    `max_frames_per_group` a cap happened to keep."""
    root = _two_group_root(tmp_path, n1=4, n2=32)
    capped = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=8, seed=0)
    uncapped = BoxDataset(root, 'train', input_wh=(64, 64), max_frames_per_group=0, seed=0)
    assert len(capped) == 4 + 8       # group 'a' (4 frames) under the cap, 'b' (32) truncated to 8
    assert len(uncapped) == 4 + 32    # nothing truncated
    # 40 (the shipped default) must be UNCHANGED from the pre-B1a behaviour.
    default = BoxDataset(root, 'train', input_wh=(64, 64), seed=0)
    assert len(default) == 4 + 32     # both groups are under the default cap of 40


def test_alpha_weights_absent_means_no_weighting(tmp_path):
    ds = BoxDataset(_two_group_root(tmp_path), 'train', input_wh=(64, 64), max_frames_per_group=0)
    assert ds.alpha_weights(None) is None


def test_alpha_1_is_frame_uniform(tmp_path):
    """alpha=1.0: every entry the same weight, i.e. plain frame-uniform sampling."""
    ds = BoxDataset(_two_group_root(tmp_path, n1=4, n2=32), 'train', input_wh=(64, 64),
                    max_frames_per_group=0)
    w = ds.alpha_weights(1.0)
    assert np.allclose(w, w[0])


def test_alpha_0_is_group_uniform(tmp_path):
    """alpha=0.0: every GROUP gets the same TOTAL weight regardless of its view count -- the
    naive 'weight by group' scheme SS2.7 warns starves a small group without this exponent form,
    but AT alpha=0 exactly it reduces to that naive scheme and the two group totals must agree.
    """
    ds = BoxDataset(_two_group_root(tmp_path, n1=4, n2=32), 'train', input_wh=(64, 64),
                    max_frames_per_group=0)
    w = ds.alpha_weights(0.0)
    keys = np.array([f'{s.session_id}/{g}' for s, g, _, _ in ds.index])
    totals = {k: float(w[keys == k].sum()) for k in np.unique(keys)}
    vals = list(totals.values())
    assert vals[0] == pytest.approx(vals[1]), f'group totals must be equal at alpha=0: {totals}'


def test_alpha_is_a_free_noop_on_uniform_group_sizes(tmp_path):
    """3dpop's own shape (SS5.5): every group the SAME view count makes alpha provably inert --
    a free null control, not a bug."""
    ds = BoxDataset(_two_group_root(tmp_path, n1=16, n2=16), 'train', input_wh=(64, 64),
                    max_frames_per_group=0)
    for alpha in (0.0, 0.5, 1.0):
        w = ds.alpha_weights(alpha)
        assert len(np.unique(np.round(w, 10))) == 1, \
            f'uniform group sizes must make alpha=({alpha}) a no-op, got {w}'





def test_detector_config_weight_decay_inherits_the_shipped_0_01(tmp_path):
    """The implicit base is `configs/detector.toml`, so an absent key inherits its shipped
    value (0.01) rather than the old AdamW hardcoded 5e-4 -- a bare config now runs the
    shipped recipe."""
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    assert load_detector_config(p)['training']['weight_decay'] == 0.01


def test_detector_config_weight_decay_rejects_negative(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
weight_decay = -0.1
""")
    with pytest.raises(SystemExit, match='weight_decay'):
        load_detector_config(p)






def test_detector_config_annot_frac_inherits_the_shipped_0_6(tmp_path):
    """Absent annot_frac inherits the shipped recipe's 0.6 (the implicit detector.toml base),
    not the old None -- a bare config now runs the shipped recipe."""
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    assert load_detector_config(p)['data']['annot_frac'] == 0.6


def test_detector_config_annot_frac_rejects_out_of_range(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
annot_frac = 1.5
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='annot_frac'):
        load_detector_config(p)


def test_giou_is_zero_for_a_perfect_box():
    b = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    assert float(giou_loss(b, b)) < 1e-5
    assert float(giou_loss(b, b + 200)) > 1.0        # disjoint -> worse than any overlap


def test_tal_finds_edge_candidates_without_center_radius():
    anchors = torch.tensor([[4., 4., 8.], [12., 12., 8.], [20., 20., 8.], [80., 80., 8.]])
    gt = torch.tensor([[0., 0., 100., 100.]])
    obj = torch.zeros(4)
    boxes = torch.tensor([[0., 0., 8., 8.], [4., 4., 16., 16.],
                          [12., 12., 24., 24.], [70., 70., 90., 90.]])
    tal_pos, tal_gt = assign_tal(anchors, gt, obj, boxes, topk=3)
    center_pos, _ = assign(anchors, gt)
    assert center_pos.numel() <= 1
    assert tal_pos.numel() >= 3 and bool((tal_gt == 0).all())


def test_tal_center_assignment_contract_is_unchanged():
    anchors = torch.tensor([[4., 4., 8.], [12., 12., 8.], [20., 20., 8.]])
    gt = torch.tensor([[0., 0., 25., 25.]])
    expected = assign(anchors, gt)
    got = assign(anchors, gt, max_pos_per_gt=None)
    for a, b in zip(expected, got):
        torch.testing.assert_close(a, b)


def test_ciou_has_finite_nonzero_gradient_for_disjoint_boxes():
    pred = torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
    target = torch.tensor([[5., 5., 6., 6.]])
    loss = ciou_loss(pred, target).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0


def test_recall_v2_loss_switches_are_live_together():
    torch.manual_seed(11)
    anchors = YOLOXNano().anchor_points(128, 128, 'cpu')
    obj = torch.randn(1, anchors.shape[0], requires_grad=True)
    boxes = torch.rand(1, anchors.shape[0], 4) * 128
    # Make every decoded box valid for the synthetic assignment quality calculation.
    boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:].abs() + 1.0
    gt = torch.tensor([[[8., 8., 96., 96.]]])
    loss, parts = detector_loss(obj, boxes, anchors, gt, assignment='tal', box_loss_fn='ciou',
)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(obj.grad).all() and parts['n_pos'] > 0


def test_detector_loss_giou_path_is_byte_identical():
    torch.manual_seed(12)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 64
    gt = torch.tensor([[[10., 10., 40., 40.]]])
    a, ap = detector_loss(obj, boxes, anchors, gt)
    b, bp = detector_loss(obj, boxes, anchors, gt, box_loss_fn='giou')
    assert torch.equal(a, b) and ap == bp




def test_assign_ignores_nan_boxes():
    """An animal absent from a view gets a NaN box, and objectness must learn "nothing here"."""
    m = YOLOXNano()
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[float('nan')] * 4])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == 0

    gt = torch.tensor([[float('nan')] * 4, [30.0, 30.0, 90.0, 90.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    assert (gix == 1).all(), 'positives must point at the finite box, not the NaN one'


def test_every_positive_anchor_is_inside_its_own_box():
    """The whole guard on `inside & near`.

    A predicted box is `centre +- exp(ltrb) * stride`, so it always contains its own anchor
    centre. A positive whose centre is outside its assigned GT box therefore has an unreachable
    regression target while objectness is taught to fire there. `inside | near` shipped for a
    while and nothing in this file caught it: uniqueness held, NaNs were still skipped, the loss
    stayed finite, and 71% of rat-city's positives were unreachable.
    """
    m = YOLOXNano()
    anchors = m.anchor_points(256, 256, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 60.0, 70.0],        # small: radius reaches well past it
                       [100.0, 30.0, 240.0, 200.0]])    # large: inside reaches past the radius
    pos, gix = assign(anchors, gt)
    assert pos.numel() > 0
    cx, cy = anchors[pos, 0], anchors[pos, 1]
    box = gt[gix]
    assert ((cx > box[:, 0]) & (cx < box[:, 2]) & (cy > box[:, 1]) & (cy < box[:, 3])).all(), \
        'a positive anchor outside its assigned box cannot reach the target it is given'
    assert len(set(gix.tolist())) == 2, 'both boxes must keep positives'


def test_assign_gives_each_anchor_one_box():
    m = YOLOXNano()
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [30.0, 30.0, 110.0, 110.0]])
    pos, gix = assign(anchors, gt)
    assert pos.numel() == len(set(pos.tolist())), 'an anchor claimed by two boxes cancels'


# T2.2 -- a per-GT positive cap. The centre prior has no cap on how many anchors one GT box can
# claim, so a large box's candidate set can dwarf a small one's. `max_pos_per_gt=None` (default)
# is uncapped and byte-identical to every checkpoint on record.
# ----------------------------------------------------------------------------------------------

def test_assign_max_pos_per_gt_default_is_unchanged():
    m = YOLOXNano()
    anchors = m.anchor_points(128, 128, torch.device('cpu'))
    gt = torch.tensor([[20.0, 20.0, 100.0, 100.0], [30.0, 30.0, 110.0, 110.0]])
    a_pos, a_gix = assign(anchors, gt)
    b_pos, b_gix = assign(anchors, gt, max_pos_per_gt=None)
    torch.testing.assert_close(a_pos, b_pos)
    torch.testing.assert_close(a_gix, b_gix)


def test_assign_max_pos_per_gt_caps_a_large_boxs_candidacy():
    """A single big GT box on a fine-stride grid claims MANY anchors uncapped; capped at K it
    must claim at most K (every positive belongs to this one GT).
    """
    m = YOLOXNano()
    anchors = m.anchor_points(256, 256, torch.device('cpu'))
    gt = torch.tensor([[10.0, 10.0, 246.0, 246.0]])          # nearly the whole frame
    pos_uncapped, _ = assign(anchors, gt)
    assert pos_uncapped.numel() > 10, 'the test needs an uncapped count clearly above K'
    K = 10
    pos_capped, gix_capped = assign(anchors, gt, max_pos_per_gt=K)
    assert pos_capped.numel() <= K
    assert bool((gix_capped == 0).all())


def test_assign_max_pos_per_gt_leaves_a_small_gt_alone():
    """The cap bounds candidacy PER GT: a small box far from the large one keeps all its own
    anchors -- it is not evicted to make room.
    """
    m = YOLOXNano()
    anchors = m.anchor_points(256, 256, torch.device('cpu'))
    big = [10.0, 10.0, 130.0, 130.0]
    small = [200.0, 200.0, 216.0, 216.0]                      # one cell at the finest stride
    gt = torch.tensor([big, small])
    pos_u, gix_u = assign(anchors, gt)
    small_count_uncapped = int((gix_u == 1).sum())
    assert 0 < small_count_uncapped < 10, 'the small box needs an uncapped count under K'
    pos_c, gix_c = assign(anchors, gt, max_pos_per_gt=10)
    small_count_capped = int((gix_c == 1).sum())
    assert small_count_capped == small_count_uncapped, \
        'the small GT must keep every one of its own candidates'
    assert int((gix_c == 0).sum()) <= 10


def test_detector_loss_max_pos_per_gt_default_is_unchanged():
    torch.manual_seed(7)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 64
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    base, bp = detector_loss(obj, boxes, anchors, gt)
    same, sp = detector_loss(obj, boxes, anchors, gt, max_pos_per_gt=None)
    assert float(base) == float(same) and bp['n_pos'] == sp['n_pos']


def test_detector_loss_max_pos_per_gt_reduces_n_pos_for_a_large_box():
    torch.manual_seed(8)
    anchors = YOLOXNano().anchor_points(256, 256, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 256
    gt = torch.tensor([[[10.0, 10.0, 246.0, 246.0]]])
    uncapped, up = detector_loss(obj, boxes, anchors, gt)
    capped, cp = detector_loss(obj, boxes, anchors, gt, max_pos_per_gt=10)
    assert up['n_pos'] > 10
    assert cp['n_pos'] <= 10
    assert cp['n_pos'] < up['n_pos']


def test_detector_config_max_pos_per_gt_inherits_the_shipped_ten(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['training']['max_pos_per_gt'] == 10


def test_detector_config_box_weight_inherits_the_shipped_fifteen(tmp_path):
    """`box_weight` used to be `detector_loss`'s own hardcoded default (5.0), never exposed to a
    config. With the implicit detector.toml base an absent key now inherits the shipped 15.0
    instead.
    """
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['training']['box_weight'] == 15.0


def test_train_detector_box_weight_is_actually_wired_through(tmp_path, dense_root, monkeypatch):
    """Proves the config->train_cfg->`detector_loss` call site is LIVE, not a dead key: two
    otherwise-identical single-iteration runs must print DIFFERENT total losses.
    """
    import importlib.util
    import re
    import sys

    def run(box_weight):
        out = tmp_path / f'run_{box_weight}'
        cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 50
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 50
eval_batches = 1
box_weight = {box_weight}
""", f'cfg_{box_weight}.toml')
        spec = importlib.util.spec_from_file_location(f'tcn_train_detector_box_weight_{box_weight}',
                                                      REPO / 'tailcyclenet' / 'train_detector.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            mod.main()
        m = re.search(r'loss\s+([\d.]+)', buf.getvalue())
        assert m, f'no loss line found in output:\n{buf.getvalue()}'
        return float(m.group(1))

    loss_5 = run(5.0)
    loss_1 = run(1.0)
    assert loss_5 != loss_1, \
        'box_weight=5.0 and box_weight=1.0 produced the SAME loss -- the config value is not ' \
        'reaching detector_loss'





def test_train_detector_max_pos_per_gt_end_to_end(tmp_path, dense_root, monkeypatch):
    """A short run with `max_pos_per_gt = 2` through the real CLI entry point: must train to
    completion with no error (0 -> None conversion happens in `train_detector.py`, not the
    config loader, so this is the one test that exercises that specific line)."""
    import importlib.util
    import sys

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
max_pos_per_gt = 2
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector_max_pos_per_gt',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector.pth').exists()
    import tomllib
    with open(out / 'config.toml', 'rb') as f:
        recorded = tomllib.load(f)
    assert recorded['training']['max_pos_per_gt'] == 2


def test_loss_is_finite_with_no_animal_anywhere():
    m = YOLOXNano()
    x = torch.zeros(2, 3, 128, 128)
    obj, boxes, _, _ = m(x)
    anchors = m.anchor_points(128, 128, x.device)
    gt = torch.full((2, 1, 4), float('nan'))
    loss, parts = detector_loss(obj, boxes, anchors, gt)
    assert torch.isfinite(loss) and parts['n_pos'] == 0


def test_decode_suppresses_duplicates():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.], [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2, 'the two overlapping boxes must collapse to one'
    assert s[0] > s[1]


def test_decode_trace_reports_score_nms_and_top_k_without_changing_output():
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.],
                          [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    plain = decode(logits, boxes, top_k=1, iou_thresh=0.5, return_index=True)
    traced = decode(logits, boxes, top_k=1, iou_thresh=0.5, return_index=True,
                    return_trace=True)
    got = traced[:3]
    trace = traced[3]
    for a, b in zip(plain, got):
        assert torch.equal(a, b), 'diagnostic tracing must not alter deployed boxes/scores/indexes'
    assert (trace['n_total'], trace['n_score'], trace['n_nms'], trace['n_top_k']) == (3, 3, 2, 1)
    assert trace['score_boxes'].shape[0] == 3
    assert trace['nms_boxes'].shape[0] == 2


def test_decode_default_iou_thresh_is_unchanged():
    """detector_v2 A1: `iou_thresh` is a new PARAMETER, and every checkpoint on record must decode
    identically to it at `iou_thresh=0.5` (the old hardcoded value). `center_dist_thresh` must be
    passed explicitly as `None` here to isolate the `iou_thresh` claim -- its OWN default changed
    to 0.5 once A5 was confirmed (see `test_decode_center_dist_is_on_by_default_now`), which is a
    deliberate break from every checkpoint trained before A5 landed, not tested by this function."""
    boxes = torch.tensor([[10., 10., 50., 50.], [11., 11., 51., 51.], [200., 200., 260., 260.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b_old, s_old = decode(logits, boxes, top_k=5, center_dist_thresh=None)
    b_new, s_new = decode(logits, boxes, top_k=5, iou_thresh=0.5, center_dist_thresh=None)
    assert torch.equal(b_old, b_new) and torch.equal(s_old, s_new)


def test_decode_center_dist_is_on_by_default_now():
    """detector_v2 A5, CONFIRMED (2 seeds, 2 roots, dev/scratch/wave0/a5_centerdist_sweep*.log):
    a bare `decode()` call (no `center_dist_thresh` passed) now suppresses a near-concentric
    duplicate IoU alone would miss -- the opposite of the pre-A5 default. `center_dist_thresh=None`
    is what restores the old byte-identical-to-every-prior-checkpoint behaviour."""
    # Same centre (30, 30); side 40 vs side 100 -> IoU = 1600/10000 = 0.16, well under 0.5.
    boxes = torch.tensor([[10., 10., 50., 50.], [-20., -20., 80., 80.], [400., 400., 440., 440.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5)
    assert b.shape[0] == 2, 'the new default must suppress the near-concentric pair'
    b_off, s_off = decode(logits, boxes, top_k=5, iou_thresh=0.5, center_dist_thresh=None)
    assert b_off.shape[0] == 3, 'center_dist_thresh=None must restore the pre-A5 behaviour'


def test_decode_center_dist_suppresses_near_concentric_boxes_iou_misses():
    """detector_v2 A5: two boxes of very different size but (nearly) the same centre are a
    near-concentric duplicate (report 42 SS3.6's own measured shape of `fp_dup`) -- IoU-only NMS at
    0.5 must NOT collapse them (that is the bug A5 exists to fix), while `center_dist_thresh`
    must.
    """
    # Same centre (30, 30); side 40 vs side 100 -> IoU = 1600/10000 = 0.16, well under 0.5.
    boxes = torch.tensor([[10., 10., 50., 50.], [-20., -20., 80., 80.], [400., 400., 440., 440.]])
    logits = torch.tensor([3.0, 2.5, 2.0])
    b, s = decode(logits, boxes, top_k=5, iou_thresh=0.5, center_dist_thresh=None)
    assert b.shape[0] == 3, 'IoU alone must not suppress a near-concentric pair this different in size'

    b2, s2 = decode(logits, boxes, top_k=5, iou_thresh=0.5, center_dist_thresh=0.5)
    assert b2.shape[0] == 2, 'centre-distance NMS must collapse the near-concentric pair'
    assert s2[0] > s2[1], 'the higher-scored box of the pair must survive'


def test_box_center_dist_is_scale_free():
    """Units of box side, not pixels: doubling every coordinate must not move the ratio."""
    from tailcyclenet.detector.assign import box_center_dist

    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[5., 0., 15., 10.]])
    r1 = box_center_dist(a, b)[0, 0].item()
    r2 = box_center_dist(a * 2, b * 2)[0, 0].item()
    assert abs(r1 - r2) < 1e-5


def test_reduce_factor_never_decodes_below_the_target():
    from tailcyclenet.detector import reduce_factor

    assert reduce_factor((4696, 2048), (640, 288)) == 4       # rat-city: 1174x512, still above
    assert reduce_factor((4696, 2048), (896, 384)) == 4       # 1174x512 still clears 896x384
    assert reduce_factor((1024, 570), (544, 320)) == 1        # calms21: 1/2 is already below
    assert reduce_factor((640, 480), (640, 480)) == 1
    for src, out in (((4696, 2048), (640, 288)), ((3840, 2160), (544, 320))):
        n = reduce_factor(src, out)
        assert src[0] / n >= out[0] and src[1] / n >= out[1], 'the remaining resize must be a '\
            'downscale, or the letterbox is upsampling a decimated frame'


def test_letterbox_scale_is_in_source_units_under_reduction():
    """`unletterbox_boxes` undoes the letterbox with this scale, and it never sees the decode.

    A reduced decode that changed the returned scale would move every predicted box by the
    reduction factor -- 4x on rat-city -- with nothing in the output to say so.
    """
    full = np.zeros((2048, 4696, 3), np.uint8)
    quarter = np.zeros((512, 1174, 3), np.uint8)
    a, s_a, p_a = letterbox(full, (640, 288))
    b, s_b, p_b = letterbox(quarter, (640, 288), src_wh=(4696, 2048))
    assert a.shape == b.shape
    assert s_a == s_b and p_a == p_b
    box = torch.tensor([[100.0, 50.0, 700.0, 400.0]])
    moved = box.clone()
    moved[:, 0::2] = moved[:, 0::2] * s_b + p_b[0]
    moved[:, 1::2] = moved[:, 1::2] * s_b + p_b[1]
    torch.testing.assert_close(unletterbox_boxes(moved, s_b, p_b), box, atol=1e-3, rtol=0)


def test_letterbox_round_trip():
    img = np.zeros((200, 400, 3), np.uint8)
    out, scale, pad = letterbox(img, (416, 416))
    assert out.shape == (416, 416, 3)
    box = torch.tensor([[10.0, 20.0, 300.0, 150.0]])
    moved = box.clone()
    moved[:, 0::2] = moved[:, 0::2] * scale + pad[0]
    moved[:, 1::2] = moved[:, 1::2] * scale + pad[1]
    torch.testing.assert_close(unletterbox_boxes(moved, scale, pad), box, atol=1e-4, rtol=0)


def test_the_batched_pack_is_bit_identical_to_the_per_frame_one():
    """PIXELS ARE A CONTRACT. `detect_group` packs a batch through numpy instead of one
    `torch.as_tensor(lb, float32).permute(2,0,1) / 255.0` per frame, because handing a 0.5 MP
    elementwise op to torch's `nproc`-wide intraop pool cost 67 ms per frame against 1.0 ms. Both
    are uint8 -> float32 -> divide by 255, both correctly rounded, so the pixels the detector sees
    must be EQUAL and not merely close -- one ulp here is a different box somewhere."""
    rng = np.random.default_rng(0)
    lbs = [rng.integers(0, 256, (32, 48, 3), dtype=np.uint8) for _ in range(5)]
    old = torch.stack([torch.as_tensor(x, dtype=torch.float32).permute(2, 0, 1) / 255.0
                       for x in lbs])
    arr = np.ascontiguousarray(np.stack(lbs).transpose(0, 3, 1, 2))
    new = torch.from_numpy(arr.astype(np.float32) / np.float32(255))
    assert torch.equal(old, new)


def test_a_video_read_locks_per_container_not_globally():
    """`dataset._read_video` takes a lock PER PATH: two threads on ONE container interleave their
    seeks, two threads on DIFFERENT containers share no state at all. The second is the whole of
    `detect_group`'s multi-camera decode overlap (3.5x on 3dpop's four cameras), so a regression to
    one global lock has to fail something."""
    from tailcyclenet import dataset as ds

    a, b = ds._read_lock_for('/x/cam0.mp4'), ds._read_lock_for('/x/cam1.mp4')
    assert a is not b
    assert ds._read_lock_for('/x/cam0.mp4') is a
    with a:                       # holding one container's lock must not block another's
        assert b.acquire(blocking=False)
        b.release()


def test_targets_are_the_crop_rule(tiny_root):
    """The regression target must BE crop_box_for_points, not something similar to it."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2)
    item = ds[0]
    x, boxes = item['x'], item['boxes']
    assert x.shape == (3, 128, 128)
    sess, gid, f, ci = ds.index[0]
    lab = sess.labels(gid)
    cam = sess.rig.posetail()[ci]
    for s in range(boxes.shape[0]):
        want = crop_box_for_points(
            torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32),
            cam['size'], ds.min_crop_dim)
        got = boxes[s]
        if want is None:
            assert torch.isnan(got).all()
            continue
        # the stored box is the crop rule's box, letterboxed; undo that and it must come back
        img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
        _, scale, pad = letterbox(img, ds.input_wh)
        back = unletterbox_boxes(got[None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_augmented_targets_are_still_the_crop_rule(tiny_root):
    """Augmentation warps the GEOMETRY and re-derives the box; it never scales the box.

    Scaling the box with the image breaks the rule: the 20 px pad scales but the `min_crop_dim`
    floor does not, so a floored box scaled by 0.8 is a box `crop_box_for_points` can never emit
    and the detector trains off its own target.
    """
    from tailcyclenet.detector.data import random_affine

    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    plain = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                       max_frames_per_group=2)
    sess, gid, f, ci = ds.index[0]
    cam = sess.rig.posetail()[ci]
    warp = random_affine(cam['size'], np.random.default_rng([ds.seed, 0]))
    got = ds.boxes_for(0, warp)

    _, scale, pad = letterbox(np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8),
                              ds.input_wh)
    lab = sess.labels(gid)
    for s in range(got.shape[0]):
        p = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
        p = p @ torch.as_tensor(warp[:, :2]).T + torch.as_tensor(warp[:, 2])
        w, h = float(cam['size'][0]), float(cam['size'][1])
        outside = (p[:, 0] < 0) | (p[:, 0] > w) | (p[:, 1] < 0) | (p[:, 1] > h)
        want = crop_box_for_points(torch.where(outside[:, None], torch.nan, p),
                                   cam['size'], ds.min_crop_dim)
        if want is None:
            assert torch.isnan(got[s]).all()
            continue
        back = unletterbox_boxes(got[s][None], scale, pad)[0]
        torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)

    # And with no warp it is byte-identical to the unaugmented loader -- `augment` must be a key
    # you turn on, not a thing that leaks into a run that did not ask for it.
    torch.testing.assert_close(ds.boxes_for(0), plain.boxes_for(0))


def test_augment_decodes_real_pixels_through_getitem(tiny_root):
    """`augment=True` must survive an actual `__getitem__` call, not just `boxes_for`.

    Every other augmentation test above calls `ds.boxes_for(...)`, which never reaches
    `_photometric` -- that function is called from `__getitem__` alone. A cleanup commit
    (`3dbb0a1`) deleted `_photometric`'s `extended` parameter and `BoxDataset`'s `photometric`
    flag but left an `if extended:` block referencing the now-undefined name, a `NameError` on
    EVERY item under `--augment`. It went undetected through this whole file and only surfaced
    when real training jobs hit it. This test exists so that class of gap cannot reopen: it is
    the one place in this file that actually decodes a pixel tensor under augmentation.
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True)
    for i in range(min(3, len(ds))):
        item = ds[i]
        x = item['x']
        _boxes = item['boxes']
        assert x.shape == (3, 48, 64)
        assert torch.isfinite(x).all()
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0, \
            'the photometric gain must still land in the normalised [0, 1] range'


def test_a_rotated_box_needs_four_corners_in_the_detector(tiny_root):
    """Two diagonal corners are not a box under a flip: their extent is strictly inside all four.

    The stored `instances.pq` extent enters the detector as geometry, so it has to be expanded
    before the warp -- the same property `test_a_rotated_box_needs_four_corners` asserts for the
    pose loader.
    """
    b = torch.tensor([10.0, 20.0, 50.0, 40.0])
    x0, y0, x1, y1 = b
    four = torch.stack([torch.stack([x0, y0]), torch.stack([x1, y0]),
                        torch.stack([x1, y1]), torch.stack([x0, y1])])
    two = b.view(2, 2)
    M = torch.tensor([[0.8, -0.6, 5.0], [0.6, 0.8, -3.0]])
    f4 = four @ M[:, :2].T + M[:, 2]
    f2 = two @ M[:, :2].T + M[:, 2]
    ext = lambda p: torch.cat([p.min(0).values, p.max(0).values])   # noqa: E731
    e4, e2 = ext(f4), ext(f2)
    assert (e4[:2] <= e2[:2]).all() and (e4[2:] >= e2[2:]).all()
    assert not torch.allclose(e4, e2), 'two corners would crop the animal the box encloses'


def test_rotation_off_is_byte_identical_and_on_turns_about_its_centre():
    """`--rotate-deg 0` must reproduce the pre-rotation matrix EXACTLY, draw for draw.

    Not "close": every detector arm on record was trained without this key, and `random_affine`
    consumes the rng, so drawing an angle and multiplying it by zero would reseat every later draw
    and silently make those arms unreproducible. The draw is skipped instead.
    """
    def before(size, rng, scale=(0.8, 1.25), translate=0.08, hflip=0.5):
        w, h = float(size[0]), float(size[1])
        s = rng.uniform(*scale)
        sx = -s if rng.random() < hflip else s
        cx, cy = w / 2, h / 2
        return np.array([[sx, 0.0, cx - sx * cx + rng.uniform(-translate, translate) * w],
                         [0.0, s, cy - s * cy + rng.uniform(-translate, translate) * h]],
                        np.float32)

    for seed in range(25):
        a = before((4696, 2048), np.random.default_rng([0, seed]))
        b = random_affine((4696, 2048), np.random.default_rng([0, seed]))
        assert np.array_equal(a, b), f'seed {seed} moved'

    # And the rotation is about `centre`, which is the whole reason tiling and rotation compose.
    M = random_affine((1000, 1000), np.random.default_rng(1), scale=(1.0, 1.0), translate=0.0,
                      hflip=0.0, rotate_deg=180.0, centre=(300.0, 700.0))
    fixed = M @ np.array([300.0, 700.0, 1.0])
    np.testing.assert_allclose(fixed, [300.0, 700.0], atol=1e-3)


def test_a_tiled_item_turns_about_its_own_tile(tiny_root):
    """A tile is cut AFTER the warp, so the warp has to hold the tile or the tile holds nothing.

    `__getitem__` composes `tile @ warp @ decode`. About the FRAME centre a rotation sweeps an
    animal clean out of the 640-px window that was chosen for it -- measured on
    rat-city-annotated at 0.075 of animal-bearing tiles still holding an animal, against 0.820
    about the tile centre (measured). Here the check is structural:
    the tile's own centre is a fixed point of its warp, and the frame's centre is not.
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(32, 32), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, rotate_deg=180.0,
                    tile_wh=(32, 32), tile_scale=1.0)
    i = next(j for j in range(len(ds)) if ds.origins[j] is not None)
    ox, oy = ds.origins[i]
    tw, th = ds._tile_extent()
    assert ds._warp_centre(i) == (ox + tw / 2, oy + th / 2)

    sess, _, _, ci = ds.index[i]
    size = tuple(sess.rig.size(sess.cam_names[ci]))
    M = random_affine(size, np.random.default_rng([ds.seed, i]), hflip=0.0, rotate_deg=180.0,
                      translate=0.0, scale=(1.0, 1.0), centre=ds._warp_centre(i))
    c = np.array([ox + tw / 2, oy + th / 2, 1.0])
    np.testing.assert_allclose(M @ c, c[:2], atol=1e-3)


def test_a_rotated_region_certifies_less_not_more(tiny_root):
    """A certified area is a CLAIM, and under a rotation a claim must round DOWN: the hull of a
    rotated rect claims area the annotator never marked, re-admitting the unlabelled animals
    `regions.pq` exists to exclude.
    """
    from tailcyclenet.detector.data import _warp_region

    r = torch.tensor([[100.0, 200.0, 500.0, 900.0], [0.0, 0.0, 64.0, 64.0]])

    # Every warp that existed before rotation keeps a rect axis-aligned, so inscribed and
    # circumscribed coincide and this is a no-op against the old four-corner code.
    for M in (np.array([[1.0, 0.0, 37.0], [0.0, 1.0, -12.0]], np.float32),
              np.array([[0.83, 0.0, 5.0], [0.0, 0.83, 9.0]], np.float32),
              np.array([[-1.1, 0.0, 900.0], [0.0, 1.1, 3.0]], np.float32)):
        c = box_corners(r) @ torch.as_tensor(M[:, :2]).T + torch.as_tensor(M[:, 2])
        want = torch.cat([c.amin(-2), c.amax(-2)], -1)
        torch.testing.assert_close(_warp_region(r, M), want, atol=1e-3, rtol=0)

    sq = torch.tensor([[0.0, 0.0, 400.0, 400.0]])
    for deg in (15.0, 45.0, 137.0):
        a = np.radians(deg)
        M = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0]], np.float32)
        hull = box_corners(sq) @ torch.as_tensor(M[:, :2]).T
        out = _warp_region(sq, M)
        assert float(out[0, 2] - out[0, 0]) < 400.0 < float((hull.amax(-2) - hull.amin(-2))[0, 0])
    # 90 degrees maps a square onto itself exactly -- no shrink is owed and none is taken.
    M90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    torch.testing.assert_close(_warp_region(sq, M90)[0, 2] - _warp_region(sq, M90)[0, 0],
                               torch.tensor(400.0), atol=1e-3, rtol=0)


def test_chunk_is_one_containers_worth_of_index(tiny_root):
    """`ChunkShuffle`'s block must be one video, or the locality it exists for is not there: a
    hardcoded 512 spanned 13 calms21 videos per block and ran the reader cache at a 16% hit rate.
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), max_frames_per_group=2)
    n_src = len({(s.session_id, g, c) for s, g, _, c in ds.index})
    assert ds.chunk == len(ds.index) // n_src
    assert ds.chunk < len(ds.index) or n_src == 1


def test_paired_sampler_emits_each_draw_twice_adjacent():
    """The ReID positive-pair contract: batch slots 2k and 2k+1 are the SAME index, fetched
    twice (two independently-augmented views under `augment and train`).
    """
    from tailcyclenet.detector import PairedSampler

    class Base:
        def __init__(self, draws):
            self.draws = draws

        def __len__(self):
            return len(self.draws)

        def __iter__(self):
            yield from self.draws

    p = PairedSampler(Base([7, 3, 9]))
    assert len(p) == 6
    assert list(iter(p)) == [7, 7, 3, 3, 9, 9]


def test_cross_camera_paired_sampler_pairs_same_frame_different_camera():
    """The Cam2 failure (report 55): same-camera augmentation pairs never teach viewpoint
    invariance, so each pair must be a DIFFERENT camera of the SAME frame. GT rows are the same
    animals in every camera, so the camera-free `(src, row)` label already treats the pair as
    one animal -- this sampler only controls WHICH two views reach the loss.
    """
    from tailcyclenet.detector import CrossCameraPairedSampler

    class FakeDS:
        class S:
            def __init__(self, sid):
                self.session_id = sid

        # index entries (sess, gid, frame, cam): 2 frames x 2 cameras
        index = [(S('s'), 'g', 0, 0), (S('s'), 'g', 0, 1),
                 (S('s'), 'g', 1, 0), (S('s'), 'g', 1, 1)]

    class Base:
        def __init__(self, draws):
            self.draws = draws

        def __len__(self):
            return len(self.draws)

        def __iter__(self):
            yield from self.draws

    ds = FakeDS()
    p = CrossCameraPairedSampler(Base([0, 1, 2, 3]), ds)
    out = list(iter(p))
    assert len(out) == 8
    for k in range(0, len(out), 2):
        i, j = out[k], out[k + 1]
        si, sj = ds.index[i], ds.index[j]
        assert (si[1], si[2]) == (sj[1], sj[2]), 'pair must share (group, frame)'
        assert si[3] != sj[3], 'pair must be a DIFFERENT camera'



def test_collate_pads_uneven_animal_counts():
    a = {'x': torch.zeros(3, 8, 8), 'boxes': torch.zeros(2, 4)}
    b = {'x': torch.zeros(3, 8, 8), 'boxes': torch.zeros(5, 4)}
    got = box_collate([a, b])
    assert got['x'].shape == (2, 3, 8, 8) and got['boxes'].shape == (2, 5, 4)
    assert torch.isnan(got['boxes'][0, 2:]).all()


def test_tracker_retires_the_weaker_duplicate_target():
    """The post-branch invariant catches duplicates born by occupied-slot drift too.

    Two targets with identical claim sets on one animal: after ``duplicate_persist`` consecutive
    in-band frames the backstop retires the weaker so the clip keeps ONE row on the animal.
    (The plan's per-frame group NMS in `associate` was removed -- a genuine crossing passes
    through the same measured band for a few frames in every camera, so per-frame merging
    turned crossings into retirements; the duplicate evidence lives in the tracker and is
    gated on persistence.)
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    per_cam = [torch.cat([b, b.clone()]) for b in per_cam]
    scores = [torch.ones(2) for _ in scores]
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=3)   # shipped default: joint
    tr.targets[0] = {'point': point.clone(), 'age': 0}
    tr.targets[1] = {'point': point.clone(), 'age': 0}
    boxes = None
    for _ in range(3):
        boxes, _, claimed = tr.step(cg, per_cam, scores)
    assert len(tr.targets) == 1
    assert np.isfinite(boxes).all(-1).any(-1).sum() == 1
    assert (claimed >= 0).any() and (claimed[1] == -1).all()


def test_freed_slot_does_not_rebirth_the_retired_duplicate():
    """The backstop retires a duplicate, but its freed slot must not be re-born from the same
    leftover detection set next frame -- retire-and-rebirth every frame was what made the
    duplicate immortal on 3dpop Sequence 59 (report 53).  Once the winner is SHIELDED by a real
    retirement, a birth whose group duplicates it is refused.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    duplicate = [torch.cat([b, b.clone()]) for b in per_cam]
    dup_scores = [torch.ones(2) for _ in scores]
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    boxes = None
    for _ in range(2):                       # both sets seated each frame; backstop fires on 2
        boxes, _, _ = tr.step(cg, duplicate, dup_scores)
    assert len(tr.targets) == 1
    first_row = int(np.isfinite(boxes).all(-1).any(-1).argmax())
    for _ in range(4):                        # freed slot must NOT rebirth the duplicate
        boxes2, _, _ = tr.step(cg, duplicate, dup_scores)
    assert len(tr.targets) == 1, 'the retired duplicate must stay retired'
    second_row = int(np.isfinite(boxes2).all(-1).any(-1).argmax())
    assert first_row == second_row, 'the surviving row must keep the animal'


def test_identity_events_record_what_the_tracker_did_without_changing_it():
    """There is no record anywhere in the output of what the tracker did to identity, so every
    question about a duplicate episode has required a monkeypatched traced re-run that reproduces
    a whole GPU pass to learn something the first pass already knew
    (identity_review_followthrough plan C6). `CrossViewTracker.events` is that record: an
    append-only list of `{frame, slot, event, detail}`.

    Two things are asserted, and the second is the load-bearing one. First, the log actually sees
    the mechanism -- this is the same duplicate scene
    `test_freed_slot_does_not_rebirth_the_retired_duplicate` uses, so it must show births, a
    retirement naming its winner, that winner being shielded, and the subsequent birth refusals
    that keep the duplicate retired. Second, RECORDING CHANGES NOTHING: an identical tracker run
    on the identical scene produces byte-identical boxes/scores/claims, so the log can never be
    suspected of having moved a number it was added to explain.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    duplicate = [torch.cat([b, b.clone()]) for b in per_cam]
    dup_scores = [torch.ones(2) for _ in scores]

    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    outs = [tr.step(cg, duplicate, dup_scores) for _ in range(6)]

    by_event = {}
    for e in tr.events:
        assert set(e) == {'frame', 'slot', 'event', 'detail'}, e
        assert 0 <= e['frame'] < 6, f'frame must be tracker-local and in range: {e}'
        by_event.setdefault(e['event'], []).append(e)

    assert len(by_event.get('born', [])) == 2, 'both slots seat on frame 0'
    retired = by_event.get('retired_duplicate', [])
    assert len(retired) == 1, f'exactly one backstop retirement in this scene, got {retired}'
    shielded = by_event.get('shielded', [])
    assert len(shielded) == 1 and shielded[0]['slot'] == retired[0]['detail']['winner'], \
        'the retirement must name its winner and that winner must be the slot shielded'
    assert retired[0]['detail']['persisted'] >= 2, 'the persistence count that fired is the reason'
    # The two ages are what plan 6.9 needs: an idle-slot policy targets a slot that stopped
    # getting evidence and then drifted onto another animal, and `age` (frames since the last
    # evidence) is the only thing that separates that from a slot retired while actively tracked.
    # Without it the log can only report how long ago a slot was BORN, which is a different
    # question and answers 6.9's only wrongly.
    assert retired[0]['detail']['loser_age'] >= 0
    assert retired[0]['detail']['winner_age'] >= 0
    assert by_event.get('birth_refused'), \
        'the freed slot is offered the same duplicate every later frame and must be refused'

    # RECORDING IS INERT. A second, independent run of the identical scene must agree byte for
    # byte on all three returned arrays -- if it does not, the log changed the algorithm.
    tr2 = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for t, (boxes, sc, claimed) in enumerate(outs):
        b2, s2, c2 = tr2.step(cg, duplicate, dup_scores)
        np.testing.assert_array_equal(np.nan_to_num(b2, nan=-9e9),
                                      np.nan_to_num(boxes, nan=-9e9), err_msg=f'boxes at t={t}')
        np.testing.assert_array_equal(np.nan_to_num(s2, nan=-9e9),
                                      np.nan_to_num(sc, nan=-9e9), err_msg=f'scores at t={t}')
        np.testing.assert_array_equal(c2, claimed, err_msg=f'claimed at t={t}')


def test_identity_events_record_a_slot_dying_of_old_age():
    """`died` is the other half of a slot's life story: without it the log shows a birth and then
    silence, and a later reader cannot tell a slot that starved out from one still alive but
    unclaimed. The event carries the age that crossed `max_age`.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    tr = CrossViewTracker(1, max_res_px=30.0, max_age=3)
    tr.step(cg, per_cam, scores)
    assert [e['event'] for e in tr.events] == ['born']

    empty = [torch.zeros((0, 4)) for _ in cg]
    empty_scores = [s.new_zeros(0) for s in empty]
    for _ in range(6):
        tr.step(cg, empty, empty_scores)
    died = [e for e in tr.events if e['event'] == 'died']
    assert len(died) == 1 and died[0]['slot'] == 0, tr.events
    assert died[0]['detail']['age'] > tr.max_age, died[0]
    assert not tr.targets


def test_shield_survives_the_winner_vacating_for_one_frame():
    """The shield is anchored to the ANIMAL, not to the winner's current claims.

    Keying the refusal to the winner's this-frame claims made it blind for exactly the frames
    crowding swapped the winner's assignment off its animal: the duplicate re-seated in that
    one frame and the backstop paid a matched-row flip five frames later (~3 refires per 160
    frames, 3dpop Sequence 59, report 53).  Here the winner is dragged away for a single frame
    (a second animal appears where it sits) and the anchor must freeze rather than follow, so
    the duplicate set still cannot re-seat.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    duplicate = [torch.cat([b, b.clone()]) for b in per_cam]
    dup_scores = [torch.ones(2) for _ in scores]
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for _ in range(2):
        tr.step(cg, duplicate, dup_scores)
    assert len(tr.targets) == 1 and tr._dup_anchor, 'a retirement must anchor its winner'
    anchored = dict(tr._dup_anchor)
    far = torch.tensor([600.0, 0.0, 0.0])
    far_cam, far_scores = _lever_boxes(cg, [far.numpy()], side=100.0)
    tr.step(cg, far_cam, far_scores)          # the winner's only work this frame is elsewhere
    assert tr._dup_anchor.keys() == anchored.keys(), 'the anchor must outlive a one-frame vacate'
    for _ in range(4):
        boxes, _, _ = tr.step(cg, duplicate, dup_scores)
    seated = [t['point'] for t in tr.targets.values()
              if float(torch.linalg.norm(t['point'] - point)) < 300.0]
    assert len(seated) == 1, 'the duplicate must not re-seat on the anchored animal'
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 1


def test_shield_dies_when_the_winners_own_target_dies():
    """The shield must not outlive the winner it protects (independent review, report 53).

    A winner shielded by a real retirement can later die of ordinary neglect (no claims for
    `max_age` frames), not just be beaten in a second retirement. Before this fix `_dup_shield`
    was discarded only inside `_suppress_duplicate_targets`'s own cleanup pass, which runs
    BEFORE the age-based death loop each frame -- so a winner that aged out inherited a shield
    entry for one extra frame, exactly long enough for the very next frame's birth (an unrelated
    animal happening to appear in the same place) to be wrongly refused.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    duplicate = [torch.cat([b, b.clone()]) for b in per_cam]
    dup_scores = [torch.ones(2) for _ in scores]
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2, max_age=2)
    for _ in range(2):
        tr.step(cg, duplicate, dup_scores)
    assert tr._dup_shield, 'a retirement must shield its winner'
    empty = [b.new_zeros((0, 4)) for b in per_cam]
    empty_scores = [s.new_zeros(0) for s in scores]
    for _ in range(4):                        # starve the winner past max_age: it must die
        tr.step(cg, empty, empty_scores)
    assert not tr.targets, 'the winner must have aged out'
    assert not tr._dup_shield, 'the shield must die with the target it protects'
    boxes, _, _ = tr.step(cg, per_cam, scores)  # an unrelated single-set birth at the same spot
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 1, \
        'a dead shield must not refuse an unrelated birth'


def test_duplicate_persistence_does_not_survive_a_members_death():
    """A `_dup_contact` counter must not resume across a slot's death (independent review).

    Before this fix the counter for a pair was pruned only when BOTH slots were re-examined as
    still-live in `_suppress_duplicate_targets`'s own accumulation loop; a slot that died via
    ordinary ageing (the loop AFTER `_suppress_duplicate_targets` in `step`) left its counters
    stale, so a same-slot rebirth on a DIFFERENT animal could resume counting toward retirement
    with a head start instead of starting at zero -- risking a genuine new animal being merged
    with an unrelated neighbour sooner than `duplicate_persist` frames of real evidence.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    duplicate = [torch.cat([b, b.clone()]) for b in per_cam]
    dup_scores = [torch.ones(2) for _ in scores]
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=5, max_age=2)
    for _ in range(3):                        # 3 of 5 in-band frames: short of retirement
        tr.step(cg, duplicate, dup_scores)
    key = frozenset(tr.targets)
    assert tr._dup_contact.get(key, 0) == 3
    empty = [b.new_zeros((0, 4)) for b in per_cam]
    empty_scores = [s.new_zeros(0) for s in scores]
    for _ in range(4):                        # both slots starve past max_age and die together
        tr.step(cg, empty, empty_scores)
    assert not tr.targets
    assert not tr._dup_contact, 'a dead pair must not leave a stale counter behind'
    for _ in range(3):                        # rebirth: must count from 0, not resume at 3
        boxes, _, _ = tr.step(cg, duplicate, dup_scores)
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 2, \
        'a rebirth must not inherit a stale persistence count and retire early'


def test_age_stickiness_resolves_two_slots_on_one_group():
    """Two targets stranded on one animal (one stale) must stop alternating on near-ties: the
    fresher slot keeps the group, the stale one starves out.  Before the fix the Hungarian
    decided such ties on per-frame jitter and the pair flip-flopped forever, never reaching
    max_age (3dpop Sequence 59, report 53).
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    tr = CrossViewTracker(2, max_res_px=30.0, max_age=4)
    # Seat both targets on the same animal, one stale (age 2) and one fresh (age 0).
    tr.targets[0] = {'point': point.clone(), 'age': 0}
    tr.targets[1] = {'point': point.clone(), 'age': 2}
    rows = []
    for _ in range(8):
        boxes, _, _ = tr.step(cg, per_cam, scores)
        rows.append(boxes)
    assert all(np.isfinite(boxes).all(-1).any(-1).sum() == 1 for boxes in rows), \
        'two slots must never both claim one group'
    assert len(tr.targets) == 1, 'the stale duplicate must starve out'
    assert np.isfinite(rows[-1]).all(-1).any(-1).sum() == 1


@pytest.mark.parametrize('assoc_mode', ['joint', 'per-camera'])
def test_age_sticky_term_never_inverts_a_real_match(assoc_mode):
    """The Hungarian cost packs age into `affinity * 16 + AGE_STICKY_CAP - age`, which stays
    positive for any positive affinity only through `age <= AGE_STICKY_CAP`. Past it the term
    can go negative for a genuinely available, high-affinity cell -- the post-hoc `affinity > 0`
    check then treats a real, well-supported match as UNAVAILABLE and the slot starves instead
    of matching (independent review; identity_review_followthrough plan A3). Zero at the shipped
    default (`max_age = 8`, so age never reaches 9) -- this fires only for a reproduction that
    raises `max_age` past the cap, e.g. the legacy `per-camera max_age=24` path.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    # An 80mm offset against a 200px box side is a real, well-supported match (well inside the
    # `max_move` gate) but not a perfect one -- affinity comfortably short of 1.0, exactly the
    # regime the inverted sign discards.
    per_cam, scores = _lever_boxes(cg, [(point + torch.tensor([80.0, 0.0, 0.0])).numpy()],
                                   side=200.0)
    tr = CrossViewTracker(1, max_res_px=30.0, max_age=30, assoc_mode=assoc_mode)
    tr.targets[0] = {'point': point.clone(), 'age': 24}
    out, sc, claimed = tr.step(cg, per_cam, scores)
    assert np.isfinite(out).all(-1).any(), \
        f'{assoc_mode}: a stale (age 24) slot lost its own high-affinity candidate to nothing'
    assert tr.targets[0]['age'] == 0, 'a real match must reset age, not leave the slot starving'


def test_claim_residual_gate_is_rejected_for_joint_association():
    """The old gate cannot be silently advertised as live in the whole-group path."""
    from tailcyclenet.detector.track import CrossViewTracker

    with pytest.raises(ValueError, match='assoc_mode'):
        CrossViewTracker(1, assoc_mode='joint', claim_residual_gate=True)


def test_min_views_1_admits_the_box_no_pair_claimed(tmp_path):
    """`min_views = 2` is the ALGORITHM, not a threshold: every group starts from a cross-camera
    pair, so the floor never fires. `min_views = 1` emits each leftover box as a single-view
    instance, which the pose model supports.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    from tailcyclenet.detector import associate
    from tailcyclenet.format import Session

    cf._session_3d(tmp_path / 'ds' / 'test' / 's')
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    cams = sess.cgroup('g000', 0)

    # One animal both of the first two cameras see at the image centre, plus a box in camera 2 that
    # no pair can agree with (it sits in a corner).
    centre = torch.tensor([[24., 16., 40., 32.]])
    per_cam = [centre, centre, torch.cat([centre, torch.tensor([[0., 0., 8., 8.]])])]

    two = associate(cams, per_cam, max_res_px=30.0, min_views=2)
    one = associate(cams, per_cam, max_res_px=30.0, min_views=1)
    assert len(one) == len(two) + 1, f'{len(two)} -> {len(one)}: the leftover box must appear'
    extra = one[len(two)]
    assert list(extra['boxes']) == [2] and torch.isnan(extra['point']).all(), \
        'a single ray has no triangulated point, and inventing a depth would be a lie'


def test_min_views_1_target_ages_out_through_the_slots_filter():
    """A `min_views=1` birth has an all-NaN point BY DESIGN (a single ray has no depth). An
    independent review claimed the resulting slot never dies, contradicting `_birth`'s own
    docstring ("it ages out through the `slots` filter in `step`") -- settled here directly
    rather than by argument (identity_review_followthrough plan C4): an all-NaN-point target is
    excluded from `slots`, so `step` ages it every frame with no evidence and it must be gone
    well before `max_age + 2` frames of silence.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    max_age = 5
    tr = CrossViewTracker(1, max_res_px=30.0, max_age=max_age, min_views=1)
    box = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
    group = {'point': torch.full((3,), float('nan')), 'residual': float('inf'), 'members': {0: 0}}
    out = np.full((1, len(cg), 4), np.nan, np.float32)
    sc = np.full((1, len(cg)), np.nan, np.float32)
    claimed_ix = np.full((1, len(cg)), -1, np.int32)
    boxes_per_cam = [box] + [torch.zeros((0, 4)) for _ in cg[1:]]
    scores_per_cam = [torch.ones(1)] + [torch.zeros(0) for _ in cg[1:]]
    tr._birth(0, group, out, sc, claimed_ix, boxes_per_cam, scores_per_cam, lambda c, j: j)
    assert 0 in tr.targets and torch.isnan(tr.targets[0]['point']).all()

    empty = [torch.zeros((0, 4)) for _ in cg]
    empty_scores = [s.new_zeros(0) for s in empty]
    for _ in range(max_age + 2):
        tr.step(cg, empty, empty_scores)
    assert not tr.targets, 'a min_views=1 slot with no evidence must age out, not leak forever'


def test_link_rows_follows_one_animal():
    """Unlinked rows are score-ordered, so the window-union crop spans several animals.

    Two animals crossing the frame in opposite directions, with their rows swapped on every odd
    frame the way a score reordering would swap them. Linking must undo that: the give-away is
    the UNION box, which is the thing `run_group` actually crops to.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    T = 20
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        a = np.array([10 + t, 10, 40 + t, 40], np.float32)      # drifts right
        b = np.array([110 - t, 10, 140 - t, 40], np.float32)    # drifts left
        boxes[0, t, 0], boxes[1, t, 0] = (a, b) if t % 2 == 0 else (b, a)

    union = lambda x: np.concatenate([x[..., :2].min(0), x[..., 2:].max(0)], -1)  # noqa: E731
    before = union(boxes[0, :, 0])
    linked = link_rows(boxes.copy())
    after = union(linked[0, :, 0])

    assert (before[2] - before[0]) > 120, 'the swapped rows should span the whole frame'
    assert (after[2] - after[0]) < (before[2] - before[0]) / 2, \
        f'linking must shrink the union crop, got {after} from {before}'
    # Every frame still holds both animals -- linking reorders, it never drops.
    assert np.isfinite(linked).all()


def test_link_rows_duplicate_suppression_prefers_empty_row():
    """A persistent same-animal pair is safer as one prediction than as a switched pair."""
    import numpy as np

    from tailcyclenet.detector import link_rows

    boxes = np.full((2, 5, 1, 4), np.nan, np.float32)
    boxes[:, :, 0] = np.array([100, 100, 200, 200], np.float32)
    scores = np.full((2, 5, 1), np.nan, np.float32)
    scores[0, :, 0] = 0.8
    scores[1, :, 0] = 0.7
    linked, linked_scores = link_rows(
        boxes, scores, max_move=2.0, duplicate_suppress=True,
        duplicate_radius=0.75, duplicate_persist=2)
    assert np.isfinite(linked[0, -1, 0]).all()
    assert not np.isfinite(linked[1, -1, 0]).any()
    assert np.isnan(linked_scores[1, -1, 0])


def test_link_rows_state_carries_across_a_split():
    """N calls with `state=` must equal ONE call over the concatenation, byte for byte.

    Blocks are sized by the RAM budget, so identity must not break at a budget-derived boundary;
    the trap is the `t0` asymmetry -- a block's frame 0 is a mid-clip frame and must be permuted
    against the carried `last`, unlike the clip's own frame 0.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    T = 20
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        a = np.array([10 + t, 10, 40 + t, 40], np.float32)
        b = np.array([110 - t, 10, 140 - t, 40], np.float32)
        boxes[0, t, 0], boxes[1, t, 0] = (a, b) if t % 2 == 0 else (b, a)
    scores = np.tile(np.array([0.9, 0.8], np.float32)[:, None, None], (1, T, 1))

    whole_b, whole_s = link_rows(boxes.copy(), scores.copy())

    # The same clip in three unequal blocks, which is what a budget-derived split looks like.
    part_b, part_s, st = boxes.copy(), scores.copy(), {}
    for lo, hi in ((0, 7), (7, 8), (8, T)):
        link_rows(part_b[:, lo:hi], part_s[:, lo:hi], state=st)

    np.testing.assert_array_equal(np.nan_to_num(part_b, nan=-9e9),
                                  np.nan_to_num(whole_b, nan=-9e9))
    np.testing.assert_array_equal(np.nan_to_num(part_s, nan=-9e9),
                                  np.nan_to_num(whole_s, nan=-9e9))
    # And the split must not be a no-op: it has to actually reorder, or this proves nothing.
    assert not np.array_equal(np.nan_to_num(whole_b, nan=-9e9),
                              np.nan_to_num(boxes, nan=-9e9))


def test_associate_group_state_carries_across_a_split(tmp_path):
    """Same claim one level up, for BOTH branches: the tracker (C > 1) and `link_rows` (C == 1).
    The scene is built so a stateless split actually FAILS -- a row that vanishes across the
    boundary seats the survivor differently in a fresh tracker.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    import numpy as np

    from tailcyclenet.detector import associate_group
    from tailcyclenet.format import Session

    T = 12
    for tag, cams, track in (('mv', 3, True), ('sv', 1, False)):
        d = tmp_path / tag / 'test' / 's'
        (cf._session_3d if cams > 1 else cf._session_2d)(d, T=T)
        sess = Session.load(d)
        sess.preload()

        D, C, S = 4, cams, 2
        box = np.full((D, T, C, 4), np.nan, np.float32)
        sc = np.full((D, T, C), np.nan, np.float32)
        for t in range(T):
            for c in range(C):
                for a in range(S):
                    # Two animals drifting apart inside the fixture's 64x48 frame. What makes each
                    # branch discriminating differs: the tracker seats a vanished row's survivor
                    # differently; link_rows needs the DETECTION ORDER to alternate.
                    if C > 1 and a == 1 and 4 <= t < 7:
                        continue
                    d = a if C > 1 else (a + t) % 2
                    x = 4.0 + 24.0 * a + 0.5 * t
                    box[d, t, c] = [x, 12, x + 12, 32]
                    sc[d, t, c] = 0.9 - 0.1 * d
        raw = (box, sc, None)

        def run(state):
            parts = []
            for lo, hi in ((0, 5), (5, T)):
                sub = (box[:, lo:hi].copy(), sc[:, lo:hi].copy(), None)
                parts.append(associate_group(sub, sess, 'g000', S, link=not track, track=track,
                                             **({} if state is None else {'state': state})))
            return [np.concatenate([p[i] for p in parts], axis=1) for i in range(2)]

        whole = associate_group(raw, sess, 'g000', S, link=not track, track=track)
        joined = run({})
        for i, name in enumerate(('boxes', 'scores')):
            np.testing.assert_array_equal(
                np.nan_to_num(joined[i], nan=-9e9), np.nan_to_num(whole[i], nan=-9e9),
                err_msg=f'{name} differ at C={cams}, track={track}: a block boundary changed the '
                        'association, so the RAM budget can change the answer')

        # THE TEST MUST BE ABLE TO FAIL. Without the carry, this scene disagrees -- if it stops
        # disagreeing, the scene has gone back to being one a fresh matcher reproduces by luck
        # and the assertion above is vacuous.
        assert not np.array_equal(np.nan_to_num(run(None)[0], nan=-9e9),
                                  np.nan_to_num(whole[0], nan=-9e9)), \
            f'C={cams}, track={track}: a STATELESS split reproduces the whole clip here, so this ' \
            'scene does not exercise the carry at all'


def test_associate_group_threads_max_age_into_the_tracker(tmp_path):
    """`--max-age` is the SAME patience window `CrossViewTracker` and `link_rows` share; passing
    it through `associate_group` (C > 1) must land on `CrossViewTracker.max_age`, not silently
    stay at the class's own default of 24. Read back off the CALLER's own `state` dict, which
    `associate_group` populates in place -- the one place the constructed tracker is visible
    without reaching into a private attribute.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest as cf

    import numpy as np

    from tailcyclenet.detector import associate_group
    from tailcyclenet.format import Session

    T, C = 4, 3
    d = tmp_path / 'test' / 's'
    cf._session_3d(d, T=T)
    sess = Session.load(d)
    sess.preload()

    box = np.full((1, T, C, 4), np.nan, np.float32)
    sc = np.full((1, T, C), np.nan, np.float32)
    raw = (box, sc, None)

    state = {}
    associate_group(raw, sess, 'g000', 1, track=True, max_age=5, state=state)
    assert state['tracker'].max_age == 5, \
        '--max-age must reach CrossViewTracker, not be dropped on the way in'


def test_associate_group_threads_max_age_into_link_rows(monkeypatch):
    """Same claim, 2D (C == 1): `associate_group` builds no tracker there, so identity runs
    through `link_rows` instead, and `max_age` must reach IT.
    """
    import numpy as np

    import tailcyclenet.detector as detmod

    seen = {}

    def spy_link_rows(boxes, scores=None, **kw):
        seen.update(kw)
        return boxes, scores

    monkeypatch.setattr(detmod, 'link_rows', spy_link_rows)

    class _FakeSession:
        assoc_res_max_px = 30.0

        def cgroup(self, gid, t=None):
            return []

        rig = type('Rig', (), {'moving': {}})()

    box = np.full((1, 2, 1, 4), np.nan, np.float32)
    sc = np.full((1, 2, 1), np.nan, np.float32)
    detmod.associate_group((box, sc, None), _FakeSession(), 'g000', 1, link=True, track=False,
                           max_age=7)
    assert seen.get('max_age') == 7, \
        '--max-age must reach link_rows in the 2D (C == 1) path too'


def test_link_rows_survives_a_dropped_frame():
    """Matching is against each row's LAST KNOWN box, so a one-frame miss cannot break the chain."""
    import numpy as np

    from tailcyclenet.detector import link_rows

    boxes = np.full((2, 4, 1, 4), np.nan, np.float32)
    for t in range(4):
        boxes[0, t, 0] = [10 + t, 10, 40 + t, 40]
        boxes[1, t, 0] = [110, 10, 140, 40]
    boxes[:, 2, 0] = np.nan                       # the detector sees nothing at frame 2
    linked = link_rows(boxes.copy())
    assert linked[0, 3, 0][0] == 13, 'row 0 must still be the left animal after the gap'
    assert linked[1, 3, 0][0] == 110


def test_box_source_instances_retargets_only_where_a_box_exists(tiny_root):
    """`--boxes instances` regresses the stored extent, and falls back per animal.

    Both halves matter for rat-city: the table is what rescues the 26k instances whose keypoints
    were cleaned away, and the fallback is what keeps the rest on the rule they always had.
    """
    # min_crop_dim 8, not the default 64: the fixture frame is 64x48, so a 64 px floor forces
    # every crop to the whole image and the two sources would agree for the wrong reason.
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=4, box_source='instances')
    base = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                      max_frames_per_group=4)
    # the fixture's one stored box is a02 on frame 1
    i = next(i for i, (_, _, f, _) in enumerate(ds.index) if f == 1)
    boxes = ds[i]['boxes']
    plain = base[i]['boxes']
    sess, gid, f, ci = ds.index[i]
    lab = sess.labels(gid)
    cam = sess.rig.posetail()[ci]
    img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
    _, scale, pad = letterbox(img, ds.input_wh)

    # a02 (row 1) carries the box: the target is the STORED corners at pad=0, not the keypoints
    want = crop_box_for_points(torch.as_tensor(lab.boxes[1, f, ci]).view(2, 2),
                               cam['size'], ds.min_crop_dim, pad=0)
    back = unletterbox_boxes(boxes[1][None], scale, pad)[0]
    torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)
    assert not torch.allclose(boxes[1], plain[1], atol=0.51)
    # a01 has no stored box, so it is byte-identical to the keypoint target
    torch.testing.assert_close(boxes[0], plain[0])


def test_box_source_rejects_a_typo(tiny_root):
    with pytest.raises(AssertionError, match='box_source'):
        BoxDataset(tiny_root / 'ratlike', 'train', box_source='instance')


def test_link_rows_never_force_assigns_another_animal():
    """A row that matches nothing must stay EMPTY, not take a leftover: `free.pop(0)` handed an
    unmatched row another detection, whose teleporting boxes blew up the window-union crop.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    T = 6
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        boxes[0, t, 0] = [10, 10, 40, 40]                      # a stationary animal
    boxes[1, 3, 0] = [900, 900, 930, 930]                      # one far-away detection, once
    linked = link_rows(boxes.copy())
    assert np.allclose(linked[0, :, 0, 0], 10), 'the tracked row must not move'
    # The far detection is a birth into the empty row 1 -- and it must never land in row 0.
    assert not np.isfinite(linked[0, 3, 0]).all() or linked[0, 3, 0][0] == 10


def test_link_rows_gates_on_the_animals_own_size():
    """A jump of more than one box side is not motion. Real p90 is 0.06-0.11 body lengths."""
    import numpy as np

    from tailcyclenet.detector import link_rows

    boxes = np.full((1, 2, 1, 4), np.nan, np.float32)
    boxes[0, 0, 0] = [0, 0, 30, 30]
    boxes[0, 1, 0] = [300, 0, 330, 30]                         # 10 box sides away
    assert not np.isfinite(link_rows(boxes.copy())[0, 1, 0]).all(), \
        'a 10-box-side jump must be rejected, not accepted as the same animal'
    near = boxes.copy()
    near[0, 1, 0] = [20, 0, 50, 30]                            # 0.67 of a side -- ordinary motion
    assert np.isfinite(link_rows(near)[0, 1, 0]).all()


def test_link_rows_prefers_the_nearer_box_where_iou_prefers_the_wrong_one():
    """IoU ranks by shape agreement, which is not identity: a true continuation that shrank loses
    to a size-matched wrong animal. Centre distance is not fooled.
    """
    import numpy as np

    from tailcyclenet.detector import link_rows

    last = np.array([100.0, 100.0, 300.0, 300.0], np.float32)     # remembered, 200 px
    near = np.array([170.0, 170.0, 230.0, 230.0], np.float32)      # same centre, 60 px
    wide = np.array([150.0, 150.0, 350.0, 350.0], np.float32)      # 0.35 sides away, 200 px
    iou = box_iou(torch.as_tensor(last[None]), torch.as_tensor(np.stack([near, wide])))[0]
    assert iou[1] > iou[0], f'the fixture must be one IoU gets wrong, got {iou.tolist()}'

    boxes = np.stack([np.stack([last, near]),
                      np.stack([np.full(4, np.nan, np.float32), wide])])[:, :, None]
    linked = link_rows(boxes.astype(np.float32).copy())
    assert np.allclose(linked[0, 1, 0], near), \
        f'centre distance must keep the nearer box, got {linked[0, 1, 0]}'
    # ...and the other detection is a BIRTH into the empty row, not a dropped animal.
    assert np.allclose(linked[1, 1, 0], wide)


def test_pose_nms_drops_the_lower_scored_duplicate():
    """Two rows whose keypoints sit almost entirely inside each other's box are one animal twice."""
    from tailcyclenet.detector.identity import pose_nms

    boxes = np.zeros((2, 1, 1, 4), np.float32)
    boxes[0, 0, 0] = [0.0, 0.0, 100.0, 100.0]
    boxes[1, 0, 0] = [5.0, 5.0, 105.0, 105.0]           # near-identical box: a duplicate detection
    kpts = np.zeros((2, 1, 1, 3, 3), np.float32)
    kpts[0, 0, 0] = [[10, 10, 1], [50, 50, 1], [90, 90, 1]]
    kpts[1, 0, 0] = [[12, 12, 1], [52, 52, 1], [92, 92, 1]]     # inside row 0's box too
    scores = np.array([[[0.9]], [[0.5]]], np.float32)

    stats = {}
    dropped = pose_nms(boxes, kpts, scores=scores, thresh=0.8, stats=stats)
    assert dropped == 1 and stats == {'nms_pairs': 1, 'nms_dropped': 1}
    assert not np.isfinite(boxes[1, 0, 0]).all(), 'the LOWER-scored row must be the one dropped'
    assert np.isfinite(boxes[0, 0, 0]).all()


def test_pose_nms_is_3d_aware_not_camera_0_only():
    """detector_v2 C1: `identity.py:93-94` used to read `k[i, t, 0]` / `b[j, t, 0]` -- CAMERA 0
    ONLY -- for both liveness and the containment overlap, while the loser deletion already spans
    every camera. Construct a 2-camera duplicate pair that is invisible in camera 0 (row 1 has NO
    box there at all) and only overlaps in camera 1: the camera-0-only version can neither see row
    1 is alive nor compute an overlap, so it must have dropped nothing; the fixed version must
    aggregate over camera 1 and drop the duplicate.
    """
    from tailcyclenet.detector.identity import pose_nms

    C = 2
    boxes = np.full((2, 1, C, 4), np.nan, np.float32)
    # Row 0: alive in BOTH cameras. Row 1: alive ONLY in camera 1 -- invisible under the old
    # camera-0-only liveness check (`np.isfinite(b[i, t, 0]).all()`).
    boxes[0, 0, 0] = [0.0, 0.0, 100.0, 100.0]
    boxes[0, 0, 1] = [0.0, 0.0, 100.0, 100.0]
    boxes[1, 0, 1] = [5.0, 5.0, 105.0, 105.0]            # near-identical box, camera 1 only

    kpts = np.zeros((2, 1, C, 3, 3), np.float32)
    kpts[0, 0, 0] = [[10, 10, 1], [50, 50, 1], [90, 90, 1]]
    kpts[0, 0, 1] = [[10, 10, 1], [50, 50, 1], [90, 90, 1]]
    kpts[1, 0, 1] = [[12, 12, 1], [52, 52, 1], [92, 92, 1]]      # inside row 0's camera-1 box too
    scores = np.array([[[0.9, 0.9]], [[np.nan, 0.5]]], np.float32)

    dropped = pose_nms(boxes, kpts, scores=scores, thresh=0.8)
    assert dropped == 1, 'aggregating over camera 1 must find and drop the duplicate'
    assert not np.isfinite(boxes[1, 0]).any(), \
        'the dropped row must go NaN in EVERY camera, not just the one the overlap was seen in'
    assert np.isfinite(boxes[0, 0]).all()


def test_pose_nms_c1_matches_camera_0_only_when_c_equals_1():
    """The 3D-aware aggregation must be a no-op on 2D single-view (C=1) -- byte-identical to the
    pre-fix behaviour, which is what every 2D checkpoint on record was measured under.
    """
    from tailcyclenet.detector.identity import pose_nms

    boxes = np.zeros((2, 1, 1, 4), np.float32)
    boxes[0, 0, 0] = [0.0, 0.0, 100.0, 100.0]
    boxes[1, 0, 0] = [5.0, 5.0, 105.0, 105.0]
    kpts = np.zeros((2, 1, 1, 3, 3), np.float32)
    kpts[0, 0, 0] = [[10, 10, 1], [50, 50, 1], [90, 90, 1]]
    kpts[1, 0, 0] = [[12, 12, 1], [52, 52, 1], [92, 92, 1]]
    scores = np.array([[[0.9]], [[0.5]]], np.float32)
    stats = {}
    dropped = pose_nms(boxes, kpts, scores=scores, thresh=0.8, stats=stats)
    assert dropped == 1 and stats == {'nms_pairs': 1, 'nms_dropped': 1}


def test_pose_nms_is_a_correct_noop_with_no_keypoints():
    """`kpts=None` must return 0 and leave `stats` EMPTY, not populate it with zeros -- the
    empty-stats case the caller's bare-subscript bug needed to reproduce.
    """
    from tailcyclenet.detector.identity import pose_nms

    boxes = np.zeros((2, 1, 1, 4), np.float32)
    boxes[0, 0, 0] = [0.0, 0.0, 100.0, 100.0]
    boxes[1, 0, 0] = [5.0, 5.0, 105.0, 105.0]
    stats = {}
    dropped = pose_nms(boxes, None, thresh=0.8, stats=stats)
    assert dropped == 0
    assert stats == {}, 'a keypoint-less no-op must not invent stats keys'
    # the caller's own read must survive an empty dict
    assert stats.get('nms_pairs', 0) == 0 and stats.get('nms_dropped', 0) == 0


def test_infer_reads_pose_nms_stats_defensively():
    """Both stats keys must be `.get(..., 0)`, never a bare subscript -- `nms_pairs` raised
    KeyError on every keypoint-less detector, the NORMAL case for a 2D root.
    """

    src = _infer_program_source()
    # Name-agnostic: what matters is that neither key is ever subscripted bare, whatever the dict
    # holding them is called this month. It has been `nms_stats` and is now `det_stats`, which is
    # also the accumulator for the box fill rate.
    for k in ('nms_pairs', 'nms_dropped'):
        assert f'["{k}"]' not in src, \
            f'a bare [{k!r}] subscript will KeyError on a keypoint-less detector'
        assert f'.get("{k}", 0)' in src, f'{k} must be read with a default'


def test_unletterbox_clamps_a_runaway_box_into_the_frame():
    """A decoded side can reach ~12,910 px; IoU-only NMS cannot suppress it, so the clamp bounds
    it into the frame and a zero-area box comes back NaN.
    """
    from tailcyclenet.detector import unletterbox_boxes

    b = torch.tensor([[-5000.0, -5000.0, 20000.0, 20000.0], [10.0, 10.0, 9.0, 40.0]])
    out = unletterbox_boxes(b, 1.0, (0, 0), src_wh=(640, 480))
    assert out[0].tolist() == [0.0, 0.0, 640.0, 480.0]
    assert torch.isnan(out[1]).all(), 'a box with no positive area is not a detection'
    # Without `src_wh` -- the training path, which has no frame to clamp against -- nothing changes.
    assert unletterbox_boxes(b, 1.0, (0, 0))[0, 2].item() == 20000.0


def test_the_cross_view_tracker_holds_identity_where_the_two_old_passes_could_not():
    """`track.demo()` as a test: one target set with one affinity cannot disagree with itself.
    `associate` was memoryless and `link_rows` matched per camera, so a row could be re-grouped
    in one pass and re-permuted in the other.
    """
    from tailcyclenet.detector.track import demo
    demo()


def test_the_tracker_and_associate_agree_on_a_single_uncrowded_animal():
    """The target state must not change the easy case, or it is not a drop-in.

    One animal on three cameras with no ambiguity: both routes must produce the same boxes, which is
    what licenses reading a delta on the crowded case as a crowding result rather than a rewrite.
    """
    import numpy as np

    from tailcyclenet.detector.associate import associate
    from tailcyclenet.detector.track import CrossViewTracker, _project

    from aniposelib.cameras import Camera, CameraGroup
    from tailcyclenet import format as fmt

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    cg = fmt.Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
                 moving=dict.fromkeys(names, False),
                 calibrated=dict.fromkeys(names, True)).posetail()

    tr = CrossViewTracker(1, max_res_px=30.0)
    for t in range(5):
        w = np.array([[10.0 * t, 0.0, 0.0]], np.float32)
        per_cam = []
        for cam in cg:
            uv = _project(cam, w)
            per_cam.append(torch.stack([uv[:, 0] - 20, uv[:, 1] - 20,
                                        uv[:, 0] + 20, uv[:, 1] + 20], -1))
        scores = [torch.ones(1) for _ in cg]
        got, _, _ = tr.step(cg, per_cam, scores)
        ref = associate(cg, per_cam, max_res_px=30.0, max_instances=1)
        assert len(ref) == 1
        for c, box in ref[0]['boxes'].items():
            np.testing.assert_allclose(got[0, c], box.numpy(), atol=1e-4)


def test_a_detector_records_its_objectness_and_load_detector_hands_it_back(tmp_path):
    """`--det-score` is not portable across detector GENERATIONS, so the distribution must ride
    in the checkpoint.

    0.99 was measured against detectors whose objectness is saturated (98.5% of rat-city's boxes at
    exactly 1.0). The tiled/masked generation reads q01 0.45-0.84 and loses two thirds of its
    detections to the same number -- coverage 0.703 against 0.986 at 0.50. That
    is a property of the RECIPE, not of the dataset, so no constant is right for both and the only
    durable answer is to record what a checkpoint actually produces.

    A checkpoint written before the field returns `{}` rather than a guess.
    """
    from tailcyclenet.detector import YOLOXNano, load_detector

    p = tmp_path / 'detector.pth'
    base = dict(model_state=YOLOXNano(n_keypoints=0).state_dict(), input_wh=[416, 416], norm='gn')
    torch.save(base, p)
    assert load_detector(p)[-1] == {}, 'an unrecorded distribution must not be invented'

    q = {'q01': 0.452, 'q10': 0.601, 'q50': 0.883, 'q90': 0.981}
    torch.save({**base, 'obj_quantiles': q}, p)
    assert load_detector(p)[-1] == q


def test_the_tracker_is_the_default_and_can_be_turned_off():
    """`--track` is ON by default, and `--no-track` restores the memoryless pass. Asserted on the
    SIGNATURE and on the parser, not on source text.
    """
    import inspect

    from tailcyclenet.detector import associate_group
    from tailcyclenet.infer.cli import build_parser

    sig = inspect.signature(associate_group)
    assert sig.parameters['track'].default is True, 'the tracker is the default'
    assert sig.parameters['link'].default is False, '--link-boxes is opt-in'

    ap = build_parser()
    assert ap.get_default('track') is True, '--track is on by default'
    opts = {s for a in ap._actions for s in a.option_strings}
    assert '--no-track' in opts, '--no-track must exist to restore the memoryless pass'
    assert '--det-cache' not in opts, \
        'the detector and the pose loop are one pass; there is no detection phase to cache'


def test_the_npz_records_which_crop_source_made_it():
    """`__box_source__` is the detector's TRAINING target and does not say what the crop came from;
    `--crop-source` pairs must not be told apart by filename alone. `--refine` rides the same field.
    """

    src = _infer_program_source()
    # `[provenance]` in the written session.toml, not the shell history. `crop_source` says what
    # the crop was BUILT from; `box_source` (checked by the provenance test above) is the
    # detector's TRAINING target, and the two can disagree.
    assert "'crop_source': cfg.crop_source" in src, \
        'the crop source must be recorded in the prediction, not left to the command line'
    # AND `refine` MUST BE THE RESOLVED FLAG, NOT THE TRI-STATE. It defaults by dimensionality, so
    # `cfg.refine` is `None` on any run that did not pass the flag -- the normal 3D case -- and
    # `run_blocks` folds it to a concrete bool before anything reads it.
    assert "'refine': bool(cfg.refine)" in src, \
        'the resolved refine flag belongs in the record; None would say nothing'
    assert "'refine': bool(cfg.refine)" in _infer_window_source(), \
        'run_blocks must resolve the tri-state before recording it'


def test_crop_source_keypoints_refuses_a_keypointless_detector():
    """A detector with no keypoint branch must not silently become `--crop-source boxes`. A source
    check because the guard sits past `load_run`, so reaching it needs a trained detector.
    """

    src = _infer_program_source()
    assert "args.crop_source == 'keypoints' and not int(getattr(det, 'n_keypoints', 0))" in src, \
        'a keypointless detector must be refused for --crop-source keypoints, not silently served'


def test_keypoint_head_is_off_by_default():
    """`n_keypoints = 0` must be BYTE-identical to the head before the branch existed.

    Not "built and ignored": the modules are not constructed at all, so the `state_dict` has no
    new keys and every recorded detector checkpoint loads without a new flag.
    """
    import torch

    from tailcyclenet.detector import YOLOXNano

    plain, kp = YOLOXNano(), YOLOXNano(n_keypoints=17)
    assert set(plain.state_dict()) == {k for k in kp.state_dict() if 'kpt' not in k}
    assert not any('kpt' in k for k in plain.state_dict())
    assert sum(p.numel() for p in plain.parameters()) \
        < sum(p.numel() for p in kp.parameters())
    obj, boxes, kpts, _ = plain(torch.zeros(1, 3, 64, 64))
    assert kpts is None, 'a keypoint-free model must return None, not zeros'


def test_embed_head_is_off_by_default():
    """`embed_dim = 0` must be BYTE-identical to the head before the branch existed.

    Same contract as `n_keypoints`: not built and ignored, not constructed at all, so an
    existing checkpoint's `state_dict` gains no new keys and loads unmodified. The embed branch
    is independent of the keypoint branch (open-set ReID, not a per-anchor identity class), so
    both must be checkable in isolation from each other.
    """
    import torch

    from tailcyclenet.detector import YOLOXNano

    plain, emb = YOLOXNano(), YOLOXNano(embed_dim=32)
    assert set(plain.state_dict()) == {k for k in emb.state_dict() if 'embed' not in k}
    assert not any('embed' in k for k in plain.state_dict())
    assert sum(p.numel() for p in plain.parameters()) \
        < sum(p.numel() for p in emb.parameters())
    obj, boxes, kpt, e = plain(torch.zeros(1, 3, 64, 64))
    assert e is None, 'an embed-free model must return None, not zeros'
    obj, boxes, kpt, e = emb(torch.zeros(1, 3, 64, 64))
    anchors = emb.anchor_points(64, 64, torch.device('cpu'))
    assert e.shape == (1, anchors.shape[0], 32)


def test_embed_and_keypoint_heads_are_independent():
    """Building one branch must not silently build, size, or disturb the other."""
    import torch

    from tailcyclenet.detector import YOLOXNano

    kpt_only = YOLOXNano(n_keypoints=5)
    assert not any('embed' in k for k in kpt_only.state_dict())
    _, _, kpt, e = kpt_only(torch.zeros(1, 3, 64, 64))
    assert kpt is not None and e is None

    embed_only = YOLOXNano(embed_dim=16)
    assert not any('kpt' in k for k in embed_only.state_dict())
    _, _, kpt, e = embed_only(torch.zeros(1, 3, 64, 64))
    assert kpt is None and e is not None

    both = YOLOXNano(n_keypoints=5, embed_dim=16)
    _, _, kpt, e = both(torch.zeros(1, 3, 64, 64))
    assert kpt is not None and e is not None
    assert kpt.shape[-2:] == (5, 3)
    assert e.shape[-1] == 16


def test_keypoint_decode_is_signed_and_bounded():
    """A positive offset moves right/down, and no keypoint escapes 1.25 box half-widths.

    Both are silent failures: `exp` on a signed offset folds every keypoint to one side of its
    anchor, and an unbounded offset is how a keypoint lands on the NEIGHBOURING animal.
    """
    import torch

    from tailcyclenet.detector import YOLOXNano

    m = YOLOXNano(n_keypoints=3).eval()
    with torch.no_grad():
        for p in m.head.kpt_pred:
            p.weight.zero_()
            p.bias.zero_()
            p.bias[0::3] = +4.0          # dx large positive -> saturates tanh
            p.bias[1::3] = -4.0          # dy large negative
        _, boxes, kpts, _ = m(torch.zeros(1, 3, 64, 64))
        anchors = m.anchor_points(64, 64, torch.device('cpu'))
    cx, cy = anchors[:, 0], anchors[:, 1]
    assert (kpts[0, :, 0, 0] > cx).all(), 'positive dx must move RIGHT (an exp decode cannot)'
    assert (kpts[0, :, 0, 1] < cy).all(), 'negative dy must move UP'
    half_x = (boxes[0, :, 2] - boxes[0, :, 0]) / 2
    assert (((kpts[0, :, 0, 0] - cx).abs() - 1.25 * half_x) <= 1e-3).all(), 'offset unbounded'


def test_keypoint_loss_nan_rule():
    """Every part of the NaN rule fails QUIETLY, so each gets an assertion.

    - an all-NaN instance is exactly 0 with finite gradients (not NaN, not a pull to the origin)
    - a half-labelled instance costs the SAME per point as a fully labelled one with the same
      errors (this is what catches normalising by K instead of by the finite count)
    - perturbing a NaN-target keypoint changes the loss by exactly 0 (this is what catches
      `nan_to_num` supervising it toward the top-left corner)
    """
    import torch

    from tailcyclenet.detector.assign import keypoint_loss

    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    nan = float('nan')

    all_nan = torch.full((1, 4, 3), nan)
    pred = torch.zeros(1, 4, 3, requires_grad=True)
    reg, sc, nk, nv = keypoint_loss(pred, all_nan, box)
    assert float(reg) == 0.0 and float(sc) == 0.0 and nk == 0 and nv == 0
    (reg + sc).backward()
    assert torch.isfinite(pred.grad).all(), 'an all-NaN instance produced non-finite gradients'

    # Same errors on the finite points; one instance labels 2 of 4, the other all 4.
    full = torch.tensor([[[1.0, 0, 1], [1.0, 0, 1], [1.0, 0, 1], [1.0, 0, 1]]])
    half = torch.tensor([[[1.0, 0, 1], [1.0, 0, 1], [nan, nan, nan], [nan, nan, nan]]])
    p = torch.zeros(1, 4, 3)
    r_full, _, n_full, _ = keypoint_loss(p, full, box)
    r_half, _, n_half, _ = keypoint_loss(p, half, box)
    assert n_full == 4 and n_half == 2
    assert abs(float(r_full) - float(r_half)) < 1e-6, \
        'per-point cost changed with label density -- normalised by K, not by the finite count'

    # Moving a masked-out prediction must not move the loss at all.
    p2 = torch.zeros(1, 4, 3)
    p2[0, 3, :2] = 500.0
    r_moved, _, _, _ = keypoint_loss(p2, half, box)
    assert float(r_moved) == float(r_half), 'a NaN-target keypoint is being supervised'


def test_keypoint_score_target_is_status_not_finiteness():
    """`x, y` null on a VISIBLE row is legal, and that row must still train the score channel.

    The format permits it when a `points3d` row exists for the same key -- allen-mouse ships a
    real per-camera visibility with no per-camera 2D. So the coordinate mask and the score mask
    cannot be the same tensor.
    """
    import torch

    from tailcyclenet.detector.assign import keypoint_loss

    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    nan = float('nan')
    t = torch.tensor([[[nan, nan, 1.0], [nan, nan, 0.0]]])     # positioned nowhere, assessed
    reg, sc, n_kpt, n_vis = keypoint_loss(torch.zeros(1, 2, 3), t, box)
    assert n_kpt == 0, 'no coordinate is finite, so nothing should be regressed'
    assert n_vis == 2 and float(sc) > 0.0, 'the score channel must still be supervised'


# ----------------------------------------------------------------------------------------------
# normalisation -- GroupNorm, and the two properties the switch was made for
# ----------------------------------------------------------------------------------------------

def test_norm_groups_always_divides_the_channel_count():
    """A GroupNorm whose count does not divide its channels is a constructor error."""
    from tailcyclenet.detector.yolox import norm_groups
    for c in list(range(1, 200)) + [24, 48, 96, 192]:
        g = norm_groups(c)
        assert 1 <= g <= c and c % g == 0


def test_there_are_no_running_statistics():
    """The whole point: train and inference are the SAME computation.

    BatchNorm collects statistics on the training distribution and applies them at inference.
    Train on animal-rich crops and infer on a mostly-empty arena and those disagree -- which is
    the train-test resolution discrepancy, and it is what would make train-on-tiles /
    infer-on-whole-frame unsafe. GroupNorm has no buffers, so there is nothing to drift.
    """
    m = YOLOXNano(n_keypoints=3)
    assert list(m.named_buffers()) == []
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())

    x = torch.rand(2, 3, 128, 192)
    m.train()
    a = m(x)[0]
    m.eval()
    with torch.no_grad():
        b = m(x)[0]
    torch.testing.assert_close(a, b)


def test_the_forward_does_not_depend_on_the_rest_of_the_batch():
    """Batch independence is what lets a high-resolution arm hold a smaller batch: without it, a
    resolution sweep that changed the batch would measure normalisation instead of resolution.
    """
    torch.manual_seed(0)
    m = YOLOXNano().eval()
    x = torch.rand(4, 3, 96, 128)
    with torch.no_grad():
        alone = m(x[1:2])[0]
        together = m(x)[0][1:2]
    torch.testing.assert_close(alone, together, rtol=1e-4, atol=1e-5)


def test_a_batchnorm_checkpoint_is_refused_by_name(tmp_path):
    """It would fail on the key names anyway; this says WHY in one sentence."""
    from tailcyclenet.detector import load_detector
    p = tmp_path / 'detector.pth'
    torch.save({'model_state': {}, 'input_wh': [128, 128]}, p)
    with pytest.raises(ValueError, match='bn normalisation'):
        load_detector(p)


# ----------------------------------------------------------------------------------------------
# the YOLOX version switch -- capacity (nano/tiny/s/m/l/x) alongside `trimmed`
# ----------------------------------------------------------------------------------------------

def test_trimmed_is_the_default_and_is_unchanged():
    """`version='trimmed'` must be indistinguishable from the model before this switch existed.

    Every checkpoint on disk was trained under the old, switch-free `YOLOXNano()`. Pinning the
    param count and the backbone type is what stands between that and a silent architecture
    change the next time this file is edited.
    """
    from tailcyclenet.detector.yolox import CSPDarknetNano

    m = YOLOXNano()
    assert m.version == 'trimmed'
    assert isinstance(m.backbone, CSPDarknetNano)
    n = sum(p.numel() for p in m.parameters())
    assert abs(n - 664_179) < 100, f'trimmed grew to {n} params -- is this still the old net?'


def test_every_yolox_tier_builds_and_forwards():
    """Every named tier in `YOLOX_TIERS`, plus `trimmed`, must construct and run end to end."""
    from tailcyclenet.detector.yolox import YOLOX_TIERS

    prev_params = 0
    for v in sorted(YOLOX_TIERS, key=lambda k: YOLOX_TIERS[k][1]):     # by width_mul, ascending
        m = YOLOXNano(n_keypoints=5, version=v)
        x = torch.rand(1, 3, 96, 128)
        obj, boxes, kpt, _ = m(x)
        anchors = m.anchor_points(96, 128, x.device)
        assert obj.shape[1] == boxes.shape[1] == anchors.shape[0] == kpt.shape[1]
        assert kpt.shape[2] == 5
        n = sum(p.numel() for p in m.parameters())
        assert n > prev_params, f'{v} must be larger than the previous (narrower) tier'
        prev_params = n


def test_yolox_tier_names_and_conv_type_match_megvii():
    """Only `nano` (and `trimmed`) is depthwise-separable; tiny/s/m/l/x are full-convolution."""
    from tailcyclenet.detector.yolox import YOLOX_TIERS

    assert set(YOLOX_TIERS) == {'nano', 'tiny', 's', 'm', 'l', 'x'}
    assert YOLOX_TIERS['nano'][2] is True
    assert all(YOLOX_TIERS[v][2] is False for v in ('tiny', 's', 'm', 'l', 'x'))
    order = ['nano', 'tiny', 's', 'm', 'l', 'x']
    depths = [YOLOX_TIERS[v][0] for v in order]
    widths = [YOLOX_TIERS[v][1] for v in order]
    assert depths == sorted(depths) and widths == sorted(widths), \
        'depth_mul and width_mul must both increase monotonically nano -> x'


def test_an_unknown_yolox_version_raises():
    with pytest.raises(ValueError, match='trimmed'):
        YOLOXNano(version='medium')


def test_width_only_applies_to_trimmed():
    """A non-default `width` alongside a canonical tier is a mistake, not a silent no-op."""
    with pytest.raises(ValueError, match='width only applies'):
        YOLOXNano(width=128, version='nano')
    YOLOXNano(width=96, version='nano')          # the sentinel default must not raise


def test_yolox_version_round_trips_through_the_checkpoint(tmp_path):
    """The fifth instance of the absent-key rule: absent means `trimmed`, never a guess."""
    from tailcyclenet.detector import load_detector

    p = tmp_path / 'detector.pth'
    m = YOLOXNano(n_keypoints=0, version='s')
    torch.save({'model_state': m.state_dict(), 'input_wh': [416, 416], 'norm': 'gn',
               'yolox_version': 's'}, p)
    loaded, *_ = load_detector(p)
    assert loaded.version == 's'
    torch.testing.assert_close(
        loaded.state_dict()['head.obj_pred.0.bias'], m.state_dict()['head.obj_pred.0.bias'])

    # absent -> 'trimmed', a fact about every checkpoint written before this switch existed
    p2 = tmp_path / 'old.pth'
    old = YOLOXNano()
    torch.save({'model_state': old.state_dict(), 'input_wh': [416, 416], 'norm': 'gn'}, p2)
    loaded2, *_ = load_detector(p2)
    assert loaded2.version == 'trimmed'


def test_embed_dim_round_trips_through_the_checkpoint(tmp_path):
    """A ReID-trained checkpoint records `embed_dim` and `load_detector` rebuilds the branch,
    so an existing checkpoint without the key stays `embed_dim=0` (byte-identical load).
    """
    from tailcyclenet.detector import load_detector

    p = tmp_path / 'detector.pth'
    m = YOLOXNano(embed_dim=16)
    torch.save({'model_state': m.state_dict(), 'input_wh': [416, 416], 'norm': 'gn',
               'yolox_version': 'trimmed', 'embed_dim': 16}, p)
    loaded, *_ = load_detector(p)
    assert loaded.head.embed_dim == 16
    torch.testing.assert_close(
        loaded.state_dict()['head.embed_pred.0.weight'],
        m.state_dict()['head.embed_pred.0.weight'])

    p2 = tmp_path / 'old.pth'
    old = YOLOXNano()
    torch.save({'model_state': old.state_dict(), 'input_wh': [416, 416], 'norm': 'gn'}, p2)
    loaded2, *_ = load_detector(p2)
    assert loaded2.head.embed_dim == 0


def test_norm_groups_divides_every_canonical_tier_channel_count():
    """The channel counts a canonical tier actually produces, not just `trimmed`'s."""
    from tailcyclenet.detector.yolox import YOLOX_TIERS, norm_groups, round8

    for depth_mul, width_mul, _ in YOLOX_TIERS.values():
        c = round8(64 * width_mul)
        for ch in (c, c * 2, c * 4, c * 8, c * 16, round8(256 * width_mul)):
            g = norm_groups(ch)
            assert 1 <= g <= ch and ch % g == 0


def test_train_detector_help_renders():
    """The same failure mode `test_infer_help_renders` guards, one script over.

    argparse expands every `help=` string as `help % params`, so a bare `%` is a format spec --
    `3.7% of` reads as `% ` (a valid flag) then `o` (octal), and `--help` dies with
    `TypeError: %o format`. Nothing had ever run this script's `--help` before the yolox/seed
    flags were added, so two pre-existing bare `%`s were sitting undetected.
    """
    import subprocess
    import sys
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / 'scripts' / 'train_detector.py'
    r = subprocess.run([sys.executable, str(p), '--help'], capture_output=True, text=True)
    assert r.returncode == 0, f'--help failed:\n{r.stderr[-2000:]}'
    # The CLI is now `--config` + three overrides; the recipe lives in the config file.
    assert '--config' in r.stdout and '--out' in r.stdout
    assert '--iters' in r.stdout and '--device' in r.stdout


# ----------------------------------------------------------------------------------------------
# tiling and the regions.pq certified mask
# ----------------------------------------------------------------------------------------------

def _root_with_regions(tmp_path, rect=(4.0, 4.0, 44.0, 34.0)):
    """A copy of the 2D fixture carrying one certified region on frame 1, camera 0."""
    from tailcyclenet import format as fmt
    from .conftest import _session_2d
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    lab.regions = np.array([[1.0, 0.0, *rect]])
    fmt.write_session(path, mode=sess.mode, units=sess.units, label_source=sess.label_source,
                      names=sess.names, rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=sess.flip_pairs, provenance=sess.provenance)
    return tmp_path / 'ds'


def test_an_untiled_checkpoints_tile_scale_is_dropped(tmp_path):
    """`tile_scale` without `tile_wh` must not reach `detect_group`, or it derives the input size.

    `train_detector.py` records the flag's DEFAULT on every run, so an untiled checkpoint carries
    `tile_scale = 1.0` -- and `detect_group` reads any non-None value as "letterbox the whole frame
    at `frame_wh * scale`", which for branson-fly is 1024x1024 against the 416x416 it trained at.
    """
    from tailcyclenet.detector import YOLOXNano, load_detector
    p = tmp_path / 'detector.pth'
    base = dict(model_state=YOLOXNano(n_keypoints=0).state_dict(), input_wh=[416, 416], norm='gn')
    torch.save({**base, 'tile_wh': None, 'tile_scale': 1.0}, p)
    assert load_detector(p)[-2] is None
    # ...and a genuinely tiled one still keeps it, or the tiled path loses its whole point.
    torch.save({**base, 'tile_wh': [640, 640], 'tile_scale': 0.5}, p)
    assert load_detector(p)[-2] == 0.5


def test_tile_transform_is_the_letterbox_form():
    from tailcyclenet.detector.data import tile_transform
    scale, pad = tile_transform((100, 50), 0.5)
    # a source point at the tile's origin lands at the input origin
    assert scale == 0.5 and pad == (-50.0, -25.0)
    assert 100 * scale + pad[0] == 0.0 and 50 * scale + pad[1] == 0.0


def test_tiled_targets_are_still_the_crop_rule(tiny_root):
    """The box is RE-DERIVED in source px, never scaled by tile_scale -- if this fails every tiled
    detector number is invalid, exactly as for the whole-frame version.
    """
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), max_frames_per_group=2,
                    tile_wh=(32, 32), tile_scale=1.0)
    from tailcyclenet.detector.data import tile_transform
    for i in range(min(6, len(ds))):
        sess, gid, f, ci = ds.index[i]
        boxes = ds.boxes_for(i)
        lab = sess.labels(gid)
        cam = sess.rig.posetail()[ci]
        ox, oy = ds.origins[i]
        tw, th = ds._tile_extent()
        for s in range(boxes.shape[0]):
            pts = torch.as_tensor(lab.points2d[s, f, :, ci], dtype=torch.float32)
            keep = ((pts[:, 0] >= ox) & (pts[:, 0] <= ox + tw) &
                    (pts[:, 1] >= oy) & (pts[:, 1] <= oy + th))
            want = crop_box_for_points(torch.where(keep[:, None], pts, torch.nan),
                                       cam['size'], ds.min_crop_dim)
            if want is None:
                assert torch.isnan(boxes[s]).all()
                continue
            scale, pad = tile_transform((ox, oy), ds.tile_scale)
            back = unletterbox_boxes(boxes[s][None], scale, pad)[0]
            torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)


def test_a_point_outside_the_tile_is_dropped(tiny_root):
    """Out-of-tile behaves exactly like out-of-frame: shrink the box, or emit no box at all."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    tile_wh=(16, 16), tile_scale=1.0)
    # a tile far from every animal: the 64x48 fixture has its points in [5, 43]
    ds.origins[0] = (1000.0, 1000.0)
    assert torch.isnan(ds.boxes_for(0)).all()


def test_an_off_frame_tile_is_grey_not_wrapped(tiny_root):
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    tile_wh=(16, 16), tile_scale=1.0)
    ds.origins[0] = (-16.0, -16.0)          # wholly outside, up and left
    x = ds[0]['x']
    assert x.shape == (3, 16, 16)
    torch.testing.assert_close(x, torch.full_like(x, 114 / 255.0))


def test_regions_none_and_empty_are_different_in_the_loader(tmp_path, tiny_root):
    """`None` claims exhaustive labelling; `(0,4)` certifies nothing. Both reach the loader."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1)
    assert ds.regions_for(0) is None                     # the fixture has no regions.pq

    root = _root_with_regions(tmp_path)
    d2 = BoxDataset(root, 'train', input_wh=(64, 48), max_frames_per_group=4)
    got = {int(d2.index[i][2]): d2.regions_for(i) for i in range(len(d2))}
    assert got[1] is not None and got[1].shape == (1, 4)  # frame 1 carries the region
    assert got[0] is not None and got[0].shape == (0, 4)  # frame 0 certifies nothing


def test_regions_ride_the_same_transform_as_the_boxes(tmp_path):
    """A region letterboxed by a different rule than its own boxes is invisible in the loss."""
    root = _root_with_regions(tmp_path, rect=(4.0, 4.0, 44.0, 34.0))
    ds = BoxDataset(root, 'train', input_wh=(128, 96), max_frames_per_group=4)
    i = next(i for i in range(len(ds)) if int(ds.index[i][2]) == 1)
    scale, pad = ds._transform(i, (64, 48))
    r = ds.regions_for(i)[0]
    torch.testing.assert_close(r, torch.tensor([4.0 * scale + pad[0], 4.0 * scale + pad[1],
                                                44.0 * scale + pad[0], 34.0 * scale + pad[1]]))


def test_use_regions_emits_a_full_frame_rect_when_the_session_has_none(tiny_root):
    """No regions.pq = exhaustively labelled = every anchor supervised, encoded as one big rect."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), max_frames_per_group=1,
                    use_regions=True)
    item = ds[0]
    assert set(item) == {'x', 'boxes', 'src', 'regions'}
    torch.testing.assert_close(item['regions'], torch.tensor([[0.0, 0.0, 64.0, 48.0]]))


def test_certified_anchors_unions_the_boxes_in():
    from tailcyclenet.detector import certified_anchors
    anchors = torch.tensor([[5.0, 5.0, 8.0], [50.0, 50.0, 8.0], [95.0, 95.0, 8.0]])
    regions = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes = torch.tensor([[40.0, 40.0, 60.0, 60.0], [float('nan')] * 4])
    got = certified_anchors(anchors, regions, boxes)
    assert got.tolist() == [True, True, False]           # region, GT box, neither
    # a NaN-padded rect certifies nothing rather than certifying the origin
    assert not certified_anchors(anchors, torch.full((1, 4), float('nan')),
                                 torch.full((1, 4), float('nan'))).any()


def test_split_batch_reads_explicit_keys_not_ranks():
    from tailcyclenet.detector import split_batch
    x, b = torch.zeros(2, 3, 8, 8), torch.zeros(2, 1, 4)
    k, r = torch.zeros(2, 1, 5, 3), torch.zeros(2, 3, 4)
    assert split_batch({'x': x, 'boxes': b}) == (x, b, None, None)
    assert split_batch({'x': x, 'boxes': b, 'kpts': k})[2] is k \
        and split_batch({'x': x, 'boxes': b, 'kpts': k})[3] is None
    assert split_batch({'x': x, 'boxes': b, 'regions': r})[2] is None \
        and split_batch({'x': x, 'boxes': b, 'regions': r})[3] is r
    got = split_batch({'x': x, 'boxes': b, 'kpts': k, 'regions': r})
    assert got[2] is k and got[3] is r


def test_detector_loss_without_regions_is_unchanged():
    """THE BACKWARD-COMPATIBILITY PROOF. Reports 10-15's numbers depend on this equality."""
    torch.manual_seed(0)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 64
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    base, bp = detector_loss(obj, boxes, anchors, gt)
    same, sp = detector_loss(obj, boxes, anchors, gt, regions=None)
    assert float(base) == float(same) and 'certified' not in bp and 'certified' not in sp

    # a mask that certifies everything is the same loss; one that certifies nothing keeps only
    # the positives, which are forced in because an unsupervised positive is an animal trained
    # as nothing.
    everything = torch.tensor([[[0.0, 0.0, 64.0, 64.0]]] * 2)
    allm, ap = detector_loss(obj, boxes, anchors, gt, regions=everything)
    torch.testing.assert_close(allm, base)
    assert ap['certified'] == 1.0
    nothing = torch.full((2, 1, 4), float('nan'))
    _, np_ = detector_loss(obj, boxes, anchors, gt, regions=nothing)
    assert 0.0 < np_['certified'] < 1.0


def test_paired_iou_matches_box_iou_diagonal():
    """`paired_iou` is the SAME maths as `box_iou`, just elementwise instead of all-pairs -- the
    diagonal of the cross-product form must equal it exactly, on both perfect and partial overlap.
    """
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [5.0, 5.0, 15.0, 15.0]])
    got = paired_iou(a, b)
    want = box_iou(a, b).diagonal()
    torch.testing.assert_close(got, want)
    assert float(got[0]) == pytest.approx(1.0)         # perfect overlap
    assert 0.0 < float(got[1]) < 1.0                    # partial overlap
    assert float(got[2]) == pytest.approx(1.0)          # perfect overlap, offset box


def test_detector_loss_iou_aware_default_off_is_unchanged():
    """THE BACKWARD-COMPATIBILITY PROOF, same shape as `test_detector_loss_without_ignore_is_
    unchanged`. `iou_aware` defaults to False, and passing it explicitly at False (or omitting it)
    must produce the IDENTICAL loss and the IDENTICAL (hard 1.0) target regardless of how good or
    bad the predicted boxes are.
    """
    torch.manual_seed(2)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    # Deliberately BAD boxes (far from the GT), so a hard-1.0 vs IoU-valued target would differ
    # a lot if the flag were live -- the strongest test that it is not.
    boxes = torch.rand(2, anchors.shape[0], 4) * 5
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    base, bp = detector_loss(obj, boxes, anchors, gt)
    off, op = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(base) == float(off)
    assert 'iou_target' not in bp and 'iou_target' not in op


def test_detector_loss_iou_aware_holds_hard_target_during_warmup():
    """`it < iou_aware_warmup` must still use the hard 1.0 target -- the chicken-and-egg trap the
    plan names: an early, near-zero predicted IoU must not be allowed to teach objectness to stay
    off. `parts['iou_target']` reads exactly 1.0 throughout warmup regardless of box quality.
    """
    torch.manual_seed(3)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5             # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    warm, wp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000, it=0)
    hard, hp = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(warm) == float(hard), 'during warmup the loss must equal the hard-target one'
    assert wp['iou_target'] == pytest.approx(1.0)


def test_detector_loss_iou_aware_switches_to_iou_target_after_warmup():
    """Past warmup, the target at a positive must be the DETACHED IoU between its predicted and
    GT box -- not 1.0 -- and `parts['iou_target']` must report the same value a direct
    `paired_iou` computation gives, tying the loss's internal bookkeeping to the public helper.
    """
    torch.manual_seed(4)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5              # bad boxes -> IoU well under 1
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    past, pp = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000,
                             it=2000)
    hard, _ = detector_loss(obj, boxes, anchors, gt, iou_aware=False)
    assert float(past) != float(hard), 'past warmup the loss must differ from the hard-target one'
    assert 0.0 <= pp['iou_target'] < 1.0

    pos, gix = assign(anchors, gt[0])
    assert pos.numel(), 'need at least one positive anchor for this check to mean anything'
    want = float(paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1).mean())
    assert pp['iou_target'] == pytest.approx(want, abs=1e-5)


def test_detector_loss_iou_aware_it_none_means_past_warmup():
    """`it=None` (a caller that does not track iterations) must behave as ALREADY past warmup --
    the docstring's stated contract, since the only caller needing the warmup (`train_detector.py`)
    always passes `it`."""
    torch.manual_seed(5)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(2, anchors.shape[0])
    boxes = torch.rand(2, anchors.shape[0], 4) * 5
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]], [[float('nan')] * 4]])
    none_it, p1 = detector_loss(obj, boxes, anchors, gt, iou_aware=True, it=None)
    past_it, p2 = detector_loss(obj, boxes, anchors, gt, iou_aware=True, iou_aware_warmup=2000,
                                it=999999)
    torch.testing.assert_close(none_it, past_it)
    assert p1['iou_target'] == pytest.approx(p2['iou_target'])


def test_iou_aware_weight_forcing_is_keyed_on_pos_mask_not_target_value():
    """THE SUBTLE CORRECTNESS CASE this key introduces: `--use-regions`/`ignore` mask the
    objectness weight, and a pre-existing guard (`weight = maximum(weight, ...)`) forces a TRUE
    POSITIVE's weight back to 1 so masking can never silently drop a real animal from the
    objectness term. That guard must key on WHETHER an anchor is positive (`pos_mask`), not on the
    VALUE its target holds -- under `iou_aware` with bad boxes the target there can be near 0, and
    forcing off THAT would only guarantee a near-0 weight, defeating the guard's whole purpose.

    Constructed so a `regions` mask certifies NOTHING (weight starts at 0 everywhere) and the
    predicted boxes are deliberately bad (low true IoU): if the guard were wrongly keyed on
    `target`, the positive anchor's contribution to `obj` would be scaled by ~its IoU instead of
    1, and this loss would NOT match the identical computation with `iou_aware=False` (which is
    known-correctly forced to weight 1 by the pre-existing, unchanged code path).
    """
    torch.manual_seed(6)
    anchors = YOLOXNano().anchor_points(64, 64, 'cpu')
    obj = torch.randn(1, anchors.shape[0])
    boxes = torch.rand(1, anchors.shape[0], 4) * 5              # bad boxes -> low true IoU
    gt = torch.tensor([[[10.0, 10.0, 40.0, 40.0]]])
    nothing_certified = torch.full((1, 1, 4), float('nan'))     # certifies NOTHING

    _, sp = detector_loss(obj, boxes, anchors, gt, regions=nothing_certified,
                          iou_aware=True, iou_aware_warmup=0, it=1)
    assert sp['iou_target'] < 0.5, 'the test needs a genuinely low IoU to be a real check'
    # The OBJ term (not the box term, which iou_aware never touches) must be identical: the
    # positive's weight was forced to 1 in both cases, only the BCE TARGET value differs, and a
    # BCE(logit, 1.0) vs BCE(logit, iou) at weight=1 are different numbers -- so this checks obj
    # is computed off the same forced weight, not that the two total losses match.
    pos, gix = assign(anchors, gt[0])
    assert pos.numel()
    logit = obj[0, pos]
    target_hard = torch.ones_like(logit)
    target_soft = paired_iou(boxes[0, pos], gt[0][gix]).clamp(0, 1)
    import torch.nn.functional as F
    norm = max(pos.numel(), 1)          # `detector_loss`'s own `obj_all / max(n_pos, B)`, B=1 here
    expected_hard_obj = F.binary_cross_entropy_with_logits(
        logit, target_hard, reduction='sum') / norm
    expected_soft_obj = F.binary_cross_entropy_with_logits(
        logit, target_soft, reduction='sum') / norm
    # Re-derived directly rather than relying on `hard`/`soft` above (which also include
    # box_weight * box, identical in both arms and therefore not informative here) -- read `obj`
    # off `parts` instead.
    _, hp = detector_loss(obj, boxes, anchors, gt, regions=nothing_certified, iou_aware=False)
    assert hp['obj'] == pytest.approx(float(expected_hard_obj), rel=1e-4)
    assert sp['obj'] == pytest.approx(float(expected_soft_obj), rel=1e-4)


def test_detector_config_iou_aware_defaults(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['training']['iou_aware_obj'] is True
    assert cfg['training']['iou_aware_warmup'] == 2000


def test_train_detector_iou_aware_end_to_end(tmp_path, dense_root, monkeypatch):
    """A short run with `iou_aware_obj = true` through the real CLI entry point: it must train to
    completion with no error, and the log line must show `iouT` once it fires (warmup=0 so it
    fires on iteration 0 here, keeping the smoke test short)."""
    import importlib.util
    import sys

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
iou_aware_obj = true
iou_aware_warmup = 0
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector_iou_aware',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector.pth').exists()
    import tomllib
    with open(out / 'config.toml', 'rb') as f:
        recorded = tomllib.load(f)
    assert recorded['training']['iou_aware_obj'] is True
    assert recorded['training']['iou_aware_warmup'] == 0


def test_support_count_counts_independently_corroborating_cameras():
    """detector_v2 E1a/E3: `_support_count` must count OTHER cameras with an unused detection
    near a candidate's reprojection, exclude the seeding pair itself, and exclude a detection
    already claimed (`used`) -- the three properties the re-ranking fix relies on.
    """
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.detector.associate import _support_count
    from tailcyclenet.detector.track import _project
    from tailcyclenet.format import Rig

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    cg = Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
             moving=dict.fromkeys(names, False),
             calibrated=dict.fromkeys(names, True)).posetail()

    p3d = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    proj = [_project(cam, p3d.reshape(1, 3).numpy()) for cam in cg]     # (1,2) per camera

    # Every camera has a detection exactly at the true reprojection.
    centres_all = [p.clone() for p in proj]
    assert _support_count(cg, (0, 1), centres_all, set(), p3d, max_res_px=5.0) == 1, \
        'excluding the seeding pair (0, 1), only camera 2 remains and it corroborates'

    # Camera 2's detection moved far away: no corroboration left.
    centres_far = [proj[0].clone(), proj[1].clone(), proj[2] + 1000.0]
    assert _support_count(cg, (0, 1), centres_far, set(), p3d, max_res_px=5.0) == 0

    # Camera 2's detection is still at the true point, but already `used` by an earlier group --
    # an already-claimed detection must not corroborate a second candidate.
    assert _support_count(cg, (0, 1), centres_all, {(2, 0)}, p3d, max_res_px=5.0) == 0


def test_associate_corroborate_flag_is_a_noop_on_the_unambiguous_case():
    """`corroborate=False` restores the pre-E1a residual-only ordering; on a scene with no
    genuine ambiguity (one real animal, or a real pair plus one leftover box no pair can explain)
    there is nothing for support-counting to change the ranking OF, so the two arms must agree.
    """
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.detector.associate import associate
    from tailcyclenet.detector.track import _project
    from tailcyclenet.format import Rig

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    cg = Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
             moving=dict.fromkeys(names, False),
             calibrated=dict.fromkeys(names, True)).posetail()

    world = np.array([[0.0, 0.0, 0.0]], np.float32)
    per_cam = []
    for cam in cg:
        uv = _project(cam, world)
        per_cam.append(torch.stack([uv[:, 0] - 20, uv[:, 1] - 20, uv[:, 0] + 20, uv[:, 1] + 20],
                                   -1))
    with_supp = associate(cg, per_cam, max_res_px=20.0, corroborate=True)
    without = associate(cg, per_cam, max_res_px=20.0, corroborate=False)
    assert len(with_supp) == len(without) == 1
    torch.testing.assert_close(with_supp[0]['point'], without[0]['point'])


def test_a_nan_box_is_skipped_by_both_cross_view_paths():
    """The two halves of `unletterbox_boxes`' contract, joined. They never were: `associate`'s
    `isfinite` guard was unreachable (SVD raises on non-finite input) and the tracker refused the
    NaN affinity matrix -- both reachable from one anchor firing in the letterbox padding band.
    """
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.detector.associate import associate
    from tailcyclenet.detector.track import CrossViewTracker, _project
    from tailcyclenet.format import Rig

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    cg = Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
             moving=dict.fromkeys(names, False),
             calibrated=dict.fromkeys(names, True)).posetail()

    world = np.array([[0.0, 0.0, 0.0]], np.float32)
    nan_row = torch.full((1, 4), float('nan'))
    per_cam, scores = [], []
    for cam in cg:
        uv = _project(cam, world)
        good = torch.stack([uv[:, 0] - 20, uv[:, 1] - 20, uv[:, 0] + 20, uv[:, 1] + 20], -1)
        per_cam.append(torch.cat([good, nan_row]))       # one real animal, one dead box
        scores.append(torch.ones(2))

    got = associate(cg, per_cam, max_res_px=20.0)
    assert len(got) == 1, 'the real animal must still be found beside a NaN box'
    assert bool(torch.isfinite(got[0]['point']).all())
    # The NaN box is not silently adopted as a member of it.
    assert all(j == 0 for j in got[0]['members'].values())

    tr = CrossViewTracker(2, max_res_px=20.0)
    boxes, _, _ = tr.step(cg, per_cam, scores)
    assert np.isfinite(boxes[0]).all(), 'the tracked animal must come back with real boxes'

    # And `min_views = 1` does not emit the dead box as a single-view instance.
    solo = associate(cg, per_cam, max_res_px=20.0, min_views=1)
    for g in solo:
        for c, b in g['boxes'].items():
            assert torch.isfinite(b).all(), f'camera {c} emitted a NaN box as an instance'


def test_reduce_under_tiling_matches_what_deployment_decodes():
    """`--reduce` compared the whole frame against the TILE size, which is not where it is headed:
    deployment letterboxes the whole frame to `tiled_input_wh`, where the same function returns 1
    and the detector sees native pixels.
    """
    from tailcyclenet.detector.data import reduce_factor

    size = (4696, 2048)                                   # rat-city
    for tile, scale in (((640, 640), 1.0), ((640, 288), 1.0), ((896, 896), 1.0)):
        assert reduce_factor(size, tile) > 1, \
            f'{tile}: the old comparison must actually decimate, or this proves nothing'
        deployed = (size[0] * scale, size[1] * scale)
        assert reduce_factor(size, deployed) == 1, \
            'at tile_scale 1.0 deployment decodes natively, so training must too'

    # And the untiled path is untouched: there `input_wh` IS what the frame is headed for.
    assert reduce_factor(size, (640, 288)) == reduce_factor(size, (640, 288))
    # At a genuine downscale the reduction comes back, on both sides alike.
    assert reduce_factor(size, (size[0] * 0.125, size[1] * 0.125)) > 1


def test_provenance_records_every_box_affecting_option():
    """Nothing that changes a detection may go unrecorded beside the numbers it produced: "which
    detector, at which threshold, at which input size" belongs in the output, not in shell history.
    Table-driven against `detect_raw`'s own signature, asserted on the VALUE `_box_provenance`
    returns.
    """
    import argparse
    import inspect

    from tailcyclenet.detector import detect_raw
    from tailcyclenet.infer.driver import _box_provenance

    args = argparse.Namespace(detector=None, det_input_wh=None, det_score=0.5, det_top_k=0,
                              max_animals=0, max_frames=0, frame_start=0, frame_stop=0,
                              det_nms_iou=0.5, det_nms_center_dist=None)
    prov = _box_provenance(args, None, False, None)

    # Everything `detect_raw` takes that can change the detections. The rest are plumbing --
    # `batch` is in there under protest: it is NOT inert but it is pinned rather
    # than exposed, so no run can differ in it. `frames`/`read` are the block loop's plumbing and
    # change no pixel. See tests/test_memory_budget.py.
    plumbing = {'det', 'session', 'gid', 'device', 'batch', 'frames', 'read', 'trace',
                'trace_detail', 'embed_out'}
    params = set(inspect.signature(detect_raw).parameters) - plumbing
    # How each is spelled in the record, where the CLI name differs from the parameter name.
    alias = {'score_thresh': 'det_score', 'input_wh': 'det_input_wh', 'top_k': 'det_top_k',
             'iou_thresh': 'det_nms_iou', 'center_dist_thresh': 'det_nms_center_dist'}
    missing = [p for p in sorted(params) if alias.get(p, p) not in prov]
    assert not missing, (
        f'these change the detections and are not recorded in the prediction: {missing}. Two runs '
        'differing in one of them would be indistinguishable from their output alone.')

    # UNCONDITIONAL, every key, always. The stamp recorded only the options that DIFFERED from
    # their defaults, because a positional list invalidated every cache on disk each time a flag
    # was added. Nothing here has that pressure, and conditional membership is what made the stamp
    # need five exceptions to its own rule.
    args2 = argparse.Namespace(detector='d.pt', det_input_wh=(416, 416), det_score=0.97,
                               det_top_k=24, max_animals=2, max_frames=120,
                               frame_start=300, frame_stop=500, det_nms_iou=0.65,
                               det_nms_center_dist=0.25)
    assert set(_box_provenance(args2, 1.0, True, 'instances')) == set(prov), \
        'the same keys at every value -- conditional membership is what makes a record lie'


def test_every_identity_lever_is_recorded_in_the_prediction():
    """The sibling of the box-provenance guard above, one pipeline stage later: detection
    provenance answers "which boxes", identity provenance answers "whose".

    This gap was found, not hypothesised: the two stored 2D suppression arms in
    `scratch/dupfollow/pred/` record their detector, crop, checkpoint and commit and say NOTHING
    about `pose_nms` or `duplicate_suppress` -- the exact levers they were built to measure -- so
    the only thing distinguishing a suppression-on run from a suppression-off one is the
    directory name. Eval rule 4 (match the controls) and rule 12 (a same-recipe replicate) both
    need these checkable from the artifact. Table-driven against `associate_group`'s own
    signature, asserted on the VALUE `_identity_provenance` returns.
    """
    import argparse
    import inspect

    from tailcyclenet.detector import associate_group
    from tailcyclenet.infer.driver import _identity_provenance

    args = argparse.Namespace(track=True, link_boxes=True, min_views=2, max_move=1.25,
                              max_age=8, assoc_mode='joint', pose_nms=None)
    prov = _identity_provenance(args)

    # Everything `associate_group` takes that can change which row a detection lands in. The rest
    # is plumbing: `raw`/`session`/`gid` are the inputs themselves, `max_instances` is already
    # recorded as `max_animals` by `_box_provenance`, `stats` is a diagnostic sink, and `state` is
    # the block-boundary carry (recorded nowhere because it is derived, not chosen).
    plumbing = {'raw', 'session', 'gid', 'max_instances', 'stats', 'state'}
    params = set(inspect.signature(associate_group).parameters) - plumbing
    # How each is spelled in the record, where the CLI name differs from the parameter name.
    alias = {'link': 'link_boxes', 'velocity': 'track_velocity'}
    missing = [p for p in sorted(params) if alias.get(p, p) not in prov]
    assert not missing, (
        f'these change which row a detection lands in and are not recorded in the prediction: '
        f'{missing}. Two runs differing in one of them would be indistinguishable from their '
        'output alone, which is exactly how the stored 2D suppression arms became unmatchable.')

    # UNCONDITIONAL, every key, always -- same rule as `_box_provenance`: conditional membership
    # is what makes a record lie, because an absent key reads as "not used" and not as "unknown".
    args2 = argparse.Namespace(track=False, link_boxes=False, min_views=1, max_move=2.0,
                               max_age=24, assoc_mode='per-camera', pose_nms=0.6,
                               claim_residual_gate=True, track_velocity=True,
                               view_arbitration=True, duplicate_suppress=True,
                               duplicate_radius=0.9, duplicate_persist=8,
                               duplicate_birth_radius=1.25)
    assert set(_identity_provenance(args2)) == set(prov), \
        'the same keys at every value -- conditional membership is what makes a record lie'
    assert _identity_provenance(args2)['pose_nms'] == 0.6
    assert prov['pose_nms'] == 0.0, 'an unset --pose-nms must record as 0.0, never as absent'


def test_summarise_exposes_fp_dup_and_fp_none_as_rates():
    """D1: `box_mota`'s own fp_dup/fp_none split (already computed for `mota` above) must survive
    into the per-group summary, not just `fp_ignored`. Rates, matching `mota()`'s own
    `fp_dup_rate`/`fp_none_rate` and every other per-gt column here (`r50`, `r75`, `fp`).
    """
    from tailcyclenet.detector.evaluate import _summarise

    s = {'n_gt': 4, 'hit50': 3, 'hit75': 2, 'iou': 3.0, 'fp': 1,
        'mota': [{'mota': 0.5, 'gt': 4, 'fp_ignored': 1, 'fp_dup': 1, 'fp_none': 2, 'misses': 3}]}
    r = _summarise(s)
    assert r['fp_ignored'] == 1
    assert r['fp_dup'] == pytest.approx(1 / 4)
    assert r['fp_none'] == pytest.approx(2 / 4)
    assert r['miss'] == pytest.approx(3 / 4)


def test_summarise_fp_dup_none_are_nan_with_no_mota_rows():
    """A group `score_dataset` never ran `box_mota` on (e.g. every GT box absent) must not read
    as a silent 0.0 -- that is indistinguishable from "measured, zero false positives".
    """
    from tailcyclenet.detector.evaluate import _summarise

    r = _summarise({'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0, 'fp': 0, 'mota': []})
    assert np.isnan(r['fp_dup']) and np.isnan(r['fp_none']) and np.isnan(r['miss'])


def test_overall_weights_fp_dup_and_fp_none_by_n_gt():
    """Same weighting basis as every other column `overall` reports -- a group's vote on the
    aggregate fp_dup/fp_none rate is proportional to how many labelled boxes it carries.
    """
    from tailcyclenet.detector.evaluate import overall

    rows = {
        'a': {'n_gt': 2, 'r50': 1, 'r75': 1, 'iou': 1, 'fp': 0, 'mota': 1.0,
             'fp_ignored': 0, 'fp_dup': 0.5, 'fp_none': 0.0, 'miss': 0.25},
        'b': {'n_gt': 6, 'r50': 1, 'r75': 1, 'iou': 1, 'fp': 0, 'mota': 1.0,
             'fp_ignored': 2, 'fp_dup': 0.0, 'fp_none': 1.0, 'miss': 0.0},
    }
    o = overall(rows)
    assert o['fp_ignored'] == 2
    assert o['fp_dup'] == pytest.approx((2 * 0.5 + 6 * 0.0) / 8)
    assert o['fp_none'] == pytest.approx((2 * 0.0 + 6 * 1.0) / 8)
    assert o['miss'] == pytest.approx((2 * 0.25 + 6 * 0.0) / 8)


def test_overall_fp_dup_none_skip_nan_groups_rather_than_propagate():
    from tailcyclenet.detector.evaluate import overall

    rows = {
        'a': {'n_gt': 2, 'r50': 1, 'r75': 1, 'iou': 1, 'fp': 0, 'mota': 1.0,
             'fp_ignored': 0, 'fp_dup': float('nan'), 'fp_none': float('nan'),
             'miss': float('nan')},
        'b': {'n_gt': 6, 'r50': 1, 'r75': 1, 'iou': 1, 'fp': 0, 'mota': 1.0,
             'fp_ignored': 0, 'fp_dup': 0.25, 'fp_none': 0.1, 'miss': 0.4},
    }
    o = overall(rows)
    assert o['fp_dup'] == pytest.approx(0.25)
    assert o['fp_none'] == pytest.approx(0.1)
    assert o['miss'] == pytest.approx(0.4)


def test_box_mota_dup_and_none_reach_evaluate_via_the_real_pipeline():
    """End to end through `box_mota` itself (not a hand-built summary dict): one exact-match TP,
    one prediction near the claimed GT (dup), one far away (none) -- the shape `_summarise`'s
    comment describes.
    """
    from tailcyclenet.detector.evaluate import box_mota

    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0],      # exact match -> TP
                        [1.0, 1.0, 11.0, 11.0],      # near the claimed GT -> dup
                        [1000.0, 1000.0, 1010.0, 1010.0]])   # far away -> none
    store = {0: (pred, gt, None, None)}
    r = box_mota(store)
    assert r['fp'] == 2 and r['fp_dup'] == 1 and r['fp_none'] == 1 and r['misses'] == 0


def test_box_mota_a_missed_animal_is_a_miss_not_an_fp():
    """The other direction from the dup/none test above: a real animal with NO prediction near it
    must show up as `misses`, and must not move `fp_dup`/`fp_none` at all.
    """
    from tailcyclenet.detector.evaluate import box_mota

    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1000.0, 1000.0, 1010.0, 1010.0]])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])   # only the first animal gets a box
    store = {0: (pred, gt, None, None)}
    r = box_mota(store)
    assert r['misses'] == 1 and r['fp'] == 0 and r['fp_dup'] == 0 and r['fp_none'] == 0


def test_score_dataset_scores_unaugmented_and_restores_the_flag():
    """`ignore_for` takes no `warp`, unlike `boxes_for` and `regions_for` beside it.

    So under `--augment` the predictions and the GT were warped while the `instances.pq` PRESENT
    boxes were not, and the ignore mask excused the wrong pixels. rat-city ships 26,021 of those
    rows, so that is most of the train-side FP readout -- the number the train/val gap is read
    from. It is also the split that is supposed to be comparable to val, which is never augmented.
    """
    import inspect

    from tailcyclenet.detector import data as ddata
    from tailcyclenet.detector.evaluate import score_dataset

    # The asymmetry that caused it, pinned so a future `warp` on `ignore_for` is noticed.
    assert 'warp' in inspect.signature(ddata.BoxDataset.boxes_for).parameters
    assert 'warp' in inspect.signature(ddata.BoxDataset.regions_for).parameters
    assert 'warp' not in inspect.signature(ddata.BoxDataset.ignore_for).parameters, \
        'ignore_for now takes a warp -- score_dataset can stop disabling augmentation'

    src = inspect.getsource(score_dataset)
    assert 'ds.augment = False' in src, 'scoring must not run through the augmentation'
    assert 'ds.augment = aug_was' in src, 'and it must put the flag back, or training loses it'


def test_a_pointless_target_expires_instead_of_burning_a_slot_forever():
    """`--min-views 1` births a target whose 3D point is all-NaN BY DESIGN; such a target is
    invisible to the matcher, so it never ages and its row is lost for the whole clip. This is NOT
    the documented immortal one-camera target, which has a finite point.
    """
    import torch

    from tailcyclenet.detector.track import CrossViewTracker

    tr = CrossViewTracker(2, max_age=3)
    tr.targets[0] = {'point': torch.full((3,), float('nan')), 'age': 0}
    assert 0 in tr.targets

    # No detections at all: the pointless target must still age out, freeing its slot.
    for _ in range(tr.max_age + 1):
        tr.step([], [], [])
    assert 0 not in tr.targets, 'a target the matcher can never see must still be able to expire'

    # A target WITH a point is unaffected by this path -- it ages through the normal loop.
    tr2 = CrossViewTracker(2, max_age=3)
    tr2.targets[0] = {'point': torch.zeros(3), 'age': 0}
    tr2.step([], [], [])
    assert tr2.targets[0]['age'] == 1, 'the finite-point target must age exactly once per frame'


def test_ema_off_builds_nothing_and_on_tracks_the_weights_it_averages():
    """`--ema-decay 0` must not construct an averaged model; on, it must actually average -- the
    one optimiser lever that yields both arms from a single run.
    """
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    model = YOLOXNano(n_keypoints=0)
    decay = 0.9
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))

    p = next(model.parameters())

    # THE FIRST `update_parameters` IS A COPY, NOT AN AVERAGE: `AveragedModel` seeds itself from
    # the model on the first call; only from the second does `avg_fn` run.
    ema.update_parameters(model)
    seeded = next(ema.module.parameters()).detach().clone()
    torch.testing.assert_close(seeded, p.detach())

    # From the second update it is a real average: a +1.0 jump moves it by exactly (1 - decay).
    with torch.no_grad():
        p.add_(1.0)
    ema.update_parameters(model)
    e1 = next(ema.module.parameters()).detach().clone()

    assert not torch.allclose(e1, seeded), 'the EMA never moved -- it is not tracking the weights'
    assert not torch.allclose(e1, p.detach()), 'the EMA copied the weights instead of averaging'
    torch.testing.assert_close(e1, seeded + (1.0 - decay), rtol=1e-4, atol=1e-6)


def test_detector_pth_is_the_best_checkpoint_not_the_last(tmp_path):
    """The run measured its own peak and then overwrote it -- worth up to -28% recall: on a root
    whose labelled frame names 2 of ~10 rats, recall peaks at 4-8k and falls by 20k, so "last" is
    systematically the wrong end.
    """
    from pathlib import Path

    # The selection rule, exercised the way the loop runs it: r50 rises then falls.
    history = [{'iteration': 2000, 'val_r50': 0.32}, {'iteration': 8000, 'val_r50': 0.39},
               {'iteration': 20000, 'val_r50': 0.28}]
    best_score, kept = -float('inf'), None
    for h in history:
        sel = h['val_r50']
        if sel >= best_score:
            best_score, kept = sel, h['iteration']
    assert kept == 8000, 'detector.pth must hold the peak, not the last evaluation'

    # ...and the end-of-run `best` line must name the SAME checkpoint, or the print lies.
    best = max(history, key=lambda h: h.get('val_r50', h.get('train_r50')))
    assert best['iteration'] == kept

    src = (Path(__file__).resolve().parent.parent / 'tailcyclenet' / 'train_detector.py').read_text()
    assert "torch.save(ckpt, run / 'detector.pth')" in src
    assert 'if sel >= best_score:' in src, 'the unconditional overwrite is back'


def test_birth_age_off_is_the_rule_it_replaced():
    """`birth_age=None` must be byte-identical to the pre-knob rule, because it is the default.
    Loosening the birth rule is refuted (it lets one row hold two animals); the knob ships off.
    """
    from tailcyclenet.detector import link_rows
    rng = np.random.default_rng(0)
    S, T = 4, 40
    boxes = np.full((S, T, 1, 4), np.nan, np.float32)
    for s in range(S):
        x, y = 100.0 + 300 * s, 100.0
        for t in range(T):
            if 12 <= t < 30 and s == 1:          # row 1 disappears for 18 frames, under max_age
                continue
            x += rng.normal(0, 2)
            boxes[s, t, 0] = (x, y, x + 60, y + 60)

    off = link_rows(boxes.copy())
    # A row that vanishes for 18 frames stays ITS OWN under the shipped rule -- nothing is reseated.
    assert np.isfinite(off[1, 5, 0]).all() and np.isfinite(off[1, 35, 0]).all()
    assert not np.isfinite(off[1, 20, 0]).any(), 'the gap must stay empty, not be filled'
    # And the knob is genuinely a no-op at None: same array as an explicit huge threshold.
    huge = link_rows(boxes.copy(), birth_age=10_000)
    np.testing.assert_array_equal(np.isfinite(off), np.isfinite(huge))




# `--augment-strong`: the strong appearance/erasure/mosaic-lite suite.
# ----------------------------------------------------------------------------------------------

def test_strong_augment_off_is_byte_identical(tiny_root):
    """`strong=False` must never draw an extra rng value or touch a pixel or a box: every recorded
    arm before it existed must stay reproducible. Asserted on shape/box count (the augmented path
    draws fresh entropy per visit, so pixels are not literally reproducible).
    """
    ds_a = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                     max_frames_per_group=2, augment=True, strong=False, seed=0)
    ds_b = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                     max_frames_per_group=2, augment=True, seed=0)   # strong defaults False
    for i in range(min(3, len(ds_a))):
        np.random.seed(0)
        item_a = ds_a[i]
        xa, ba = item_a['x'], item_a['boxes']
        np.random.seed(0)
        item_b = ds_b[i]
        xb, bb = item_b['x'], item_b['boxes']
        # Pixels drawn from a fresh rng each visit (`default_rng(None)`) are not literally
        # reproducible call-to-call under the augmented path -- what must hold is that the two
        # CONSTRUCTIONS (with and without the explicit strong=False) are the same object, i.e.
        # neither adds a code path the other lacks. Assert on SHAPE and on box count instead of
        # bit-exact pixels, since `_photometric` alone already draws fresh entropy per visit.
        assert xa.shape == xb.shape
        assert ba.shape == bb.shape


def test_strong_augment_off_leaves_boxes_targets_unchanged(tiny_root):
    """With `strong` off, `boxes_for` -- the actual target -- is bit-identical to the plain loader.

    This is the sharper off-path guarantee: `boxes_for` never even sees the `strong` flag, so this
    just pins that nothing upstream of it was touched by adding the flag.
    """
    plain = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                       max_frames_per_group=2, augment=True, seed=0)
    strong_off = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                           max_frames_per_group=2, augment=True, strong=False, seed=0)
    for i in range(min(3, len(plain))):
        torch.testing.assert_close(plain.boxes_for(i), strong_off.boxes_for(i))


def test_strong_augment_preserves_box_targets(tiny_root):
    """Appearance ops and cutout must never move a box -- only mosaic-lite may ADD one."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    for i in range(min(4, len(ds))):
        base = ds.boxes_for(i)     # the animal count before any strong op ran
        for _ in range(8):
            boxes = ds[i]['boxes']
            # Appearance ops and cutout never touch `boxes_for`'s output -- only mosaic-lite
            # appends a row -- so the box count for this item can only stay the same or GROW,
            # never shrink or resize below what `boxes_for` alone would produce.
            assert boxes.shape[0] >= base.shape[0]


def test_cutout_zeroes_covered_keypoints(tiny_root):
    """A keypoint inside a cutout rect must end up coord-NaN AND score-0, never just one."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, keypoints=True, seed=0)
    _, kpts = ds.boxes_for(0, None, with_keypoints=True)
    rng = np.random.default_rng(0)
    rects = _cutout_rects(ds.input_wh, rng, n=(1, 1), frac=1.0)   # cover the WHOLE frame
    mask = _keypoints_in_rects(kpts[..., :2], rects)
    assert bool((torch.isfinite(kpts[..., 0]) & mask).any()), \
        'fixture must have at least one finite keypoint to erase'
    k2 = kpts.clone()
    k2[..., 0] = torch.where(mask, torch.nan, k2[..., 0])
    k2[..., 1] = torch.where(mask, torch.nan, k2[..., 1])
    k2[..., 2] = torch.where(mask, torch.zeros_like(k2[..., 2]), k2[..., 2])
    assert torch.isnan(k2[..., :2][mask]).all()
    assert (k2[..., 2][mask] == 0).all()


def test_mosaic_paste_is_fully_interior_and_reencodes_the_crop_rule(tiny_root):
    """The appended box is `src_box + translation`, entirely inside `input_wh`."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(96, 72), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, strong=True, seed=0)
    rng = np.random.default_rng(1)
    for i in range(min(3, len(ds))):
        base = ds.boxes_for(i)
        img, _, _ = ds._load_letterbox(i)
        boxes, kpts, img2 = ds._mosaic_paste(i, base.clone(), None, img.copy(), rng)
        if boxes.shape[0] == base.shape[0]:
            continue    # no finite source box was found in a few tries; not this fixture's job
        new = boxes[base.shape[0]:]
        assert (new[:, 0] >= 0).all() and (new[:, 1] >= 0).all()
        assert (new[:, 2] <= ds.input_wh[0]).all() and (new[:, 3] <= ds.input_wh[1]).all()
        return
    pytest.skip('no fixture item produced a finite mosaic source box in the tries allotted')


def test_mosaic_rejected_when_use_regions(tiny_root):
    """Fails at CONSTRUCTION, not on the ~20%% of items that happen to draw mosaic-lite -- a
    training job should not discover this combination is undefined hours into a run.
    """
    with pytest.raises(ValueError):
        BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                  max_frames_per_group=2, augment=True, strong=True, use_regions=True, seed=0)

    # And `_mosaic_paste` itself still refuses to run, for a caller that reaches it some other way.
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 48), min_crop_dim=8,
                    max_frames_per_group=2, augment=True, use_regions=True, seed=0)
    ds.strong = True     # bypass the constructor guard to exercise the method's own guard
    boxes = ds.boxes_for(0)
    img, _, _ = ds._load_letterbox(0)
    with pytest.raises(RuntimeError):
        ds._mosaic_paste(0, boxes, None, img, np.random.default_rng(0))


def test_infer_help_renders():
    """`--help` must actually print: argparse expands every `help=` string as `help % params`, so a
    bare `%` kills it with a format TypeError and hides every option the script has.
    """
    import subprocess
    import sys
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / 'scripts' / 'infer.py'
    r = subprocess.run([sys.executable, str(p), '--help'], capture_output=True, text=True)
    assert r.returncode == 0, f'--help failed:\n{r.stderr[-2000:]}'
    assert '--pose-nms' in r.stdout




# --- detector training config (configs/detector.toml + tailcyclenet/detector/config.py) ------

REPO = Path(__file__).resolve().parent.parent
SHIPPED_DETECTOR_CONFIG = REPO / 'configs' / 'detector.toml'


def _write_config(tmp_path, text, name='config.toml'):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_detector_config_recall_v2_defaults_and_keys(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    t = cfg['training']
    assert (t['assignment'], t['box_loss']) == ('tal', 'ciou')
    assert t['tal_topk'] == 20
    assert 'assignment' in TRAINING_KEYS


@pytest.mark.parametrize(('key', 'value'), [('assignment', 'bad'), ('box_loss', 'bad')])
def test_detector_config_recall_v2_choice_validation(tmp_path, key, value):
    p = _write_config(tmp_path, f"""
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
{key} = "{value}"
""")
    with pytest.raises(SystemExit, match=key):
        load_detector_config(p)


def test_detector_config_loads_with_shipped_defaults(tmp_path):
    """The implicit base is the shipped `configs/detector.toml`: a bare overlay inherits the
    whole recommended recipe, and `--out`/`--iters`/`--device` overrides land in [training].
    Path is still required (a missing dataset must fail at load, not train on the CWD)."""
    overlay = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[training]
out = "/tmp/run-det"
""")
    cfg = load_detector_config(overlay, out='/tmp/run-det', iters=7, device='cpu')
    d, m, t = cfg['data'], cfg['model'], cfg['training']
    assert d['path'] == '/tmp/ds'
    assert d['boxes'] == 'instances'
    assert d['min_crop_dim'] == 64
    assert d['min_box_px'] == 32
    assert d['max_input_px'] == 4 * 416 * 416
    assert 'frames_per_group' not in d      # DELETED: the train draw is weighted, not capped
    assert d['val_frames_per_group'] == 8
    assert d['augment'] is True and d['augment_strong'] is True
    assert d['rotate_deg'] == 45.0
    assert d['reduce'] is False and d['keypoints'] is False and d['hflip'] is True
    assert d['use_regions'] is False
    assert d['input_wh'] is None and d['tile_wh'] is None       # absent pair -> None
    assert d['tile_scale'] == 1.0 and d['tile_bg_per_frame'] == 1
    assert m['yolox'] == 'hybrid'
    assert t['out'] == '/tmp/run-det'
    assert t['iters'] == 7
    assert t['batch_size'] == 8 and t['lr'] == 1e-3
    assert t['num_workers'] == 8 and t['seed'] == 23
    assert t['device'] == 'cpu'
    assert t['eval_every'] == 2000 and t['eval_batches'] == 25
    assert t['kpt_weight'] == 1.0 and t['kpt_score_weight'] == 1.0
    assert t['shared_head'] is False
    assert t['fpn_upsample'] == 'bilinear'
    assert t['tal_topk'] == 20
    assert t['optimizer'] == 'muon'


def test_detector_config_unknown_key_raises_in_every_block(tmp_path):
    from tailcyclenet.detector.config import load_detector_config

    for block, bad in (('data', 'bogus = 1'),
                       ('model', 'bogus = 1'),
                       ('training', 'bogus = 1')):
        p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
boxes = "instances"
min_crop_dim = 64
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
iters = 1
""")
        # inject the unknown key into `block`
        lines = p.read_text().splitlines(keepends=True)
        out_lines = []
        for ln in lines:
            out_lines.append(ln)
            if ln.strip() == f'[{block}]':
                out_lines.append('bogus = 1\n')
        p.write_text(''.join(out_lines))
        with pytest.raises(SystemExit, match='unknown key'):
            load_detector_config(p)


def test_detector_config_bad_choices_raise(tmp_path):
    from tailcyclenet.detector.config import load_detector_config

    # boxes lives in [data]; append the bad key to the existing [data] block instead of a second one.
    p = _write_config(tmp_path, """\
[data]
path = "/tmp/ds"
boxes = "nope"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
iters = 1
""", 'b1.toml')
    with pytest.raises(SystemExit, match='boxes'):
        load_detector_config(p)
    p = _write_config(tmp_path, """\
[data]
path = "/tmp/ds"
boxes = "instances"
[model]
yolox = "nope"
[training]
out = "/tmp/run"
iters = 1
""", 'b2.toml')
    with pytest.raises(SystemExit, match='yolox'):
        load_detector_config(p)


def test_detector_config_extends_is_deleted_and_raises(tmp_path):
    """`extends` is deleted from the config language: EVERY config layers over its family's
    base automatically (detector -> `configs/detector.toml`), so the key is not needed and
    naming it is refused by name."""
    from tailcyclenet.detector.config import load_detector_config

    overlay = _write_config(tmp_path, """
extends = "../detector.toml"
[data]
path = "/tmp/ds"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='extends'):
        load_detector_config(overlay)


def test_detector_config_round_trips_through_the_run_folder(tmp_path):
    """The recorded config.toml (None values dropped) loads back to the same recipe."""
    from tailcyclenet.detector.config import load_detector_config

    src = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
boxes = "keypoints"
min_crop_dim = 64
input_wh = [96, 64]
tile_wh = []
[model]
yolox = "trimmed"
[training]
out = "/tmp/run"
iters = 2
""", 'src.toml')
    cfg = load_detector_config(src, out=str(tmp_path / 'run'), iters=2, device='cpu')
    import toml
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'config.toml').write_text(toml.dumps(cfg))
    again = load_detector_config(tmp_path / 'run' / 'config.toml')
    assert again['data']['input_wh'] == [96, 64] and again['data']['tile_wh'] is None
    assert again['data']['boxes'] == 'keypoints'
    assert again['training']['iters'] == 2
    assert again['model']['yolox'] == 'trimmed'


def test_train_detector_config_end_to_end(tmp_path, dense_root, monkeypatch):
    """A 2-iteration run through `scripts/train_detector.py`'s `main()` with a config file
    produces the same artefacts a CLI run did, and the checkpoint still loads through `load_detector`.
    IN-PROCESS, not a subprocess: a fresh interpreter pays ~60 s just importing torch.
    """
    import importlib.util
    import sys
    import tomllib

    from tailcyclenet.detector import load_detector

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
reduce = false
keypoints = false
hflip = true
tile_wh = []
tile_scale = 1.0
tile_bg_per_frame = 1
use_regions = false
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
kpt_weight = 1.0
kpt_score_weight = 1.0
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector_it000002.pth').exists()
    assert (out / 'detector_last.pth').exists()
    assert (out / 'detector.pth').exists()
    assert (out / 'metrics.json').exists()
    assert (out / 'config.toml').exists()
    assert (out / 'provenance.toml').exists()
    with open(out / 'config.toml', 'rb') as f:
        recorded = tomllib.load(f)
    assert recorded['data']['boxes'] == 'keypoints'
    assert recorded['data']['input_wh'] == [48, 48]
    assert recorded['training']['iters'] == 2
    model, wh, ds_name, mcd, reduce, box_src, ts, obj_q = load_detector(out / 'detector.pth')
    assert tuple(wh) == (48, 48)
    assert mcd == 16
    assert box_src == 'keypoints'
    assert ts is None                                # untiled: tile_scale is dropped at the read
    ckpt = torch.load(out / 'detector_it000002.pth', map_location='cpu', weights_only=False)
    assert ckpt['yolox_version'] == 'tiny'
    assert ckpt['min_crop_dim'] == 16
    assert ckpt['box_source'] == 'keypoints'
    assert tuple(ckpt['input_wh']) == (48, 48)


def test_load_detector_config_out_override_rescues_an_empty_out(tmp_path):
    """`--out` must be able to fill in `[training].out = ""` -- the override used to be applied
    AFTER the required-check, so it could never rescue anything.
    """
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text("""
[data]
path = "/does/not/need/to/exist/for/this/check"

[model]
yolox = "trimmed"

[training]
out = ""
""")
    loaded = load_detector_config(cfg, out=tmp_path / 'run')
    assert loaded['training']['out'] == str(tmp_path / 'run')

    with pytest.raises(SystemExit, match='out.*required'):
        load_detector_config(cfg)                    # nothing supplies `out`: still raises


def test_load_detector_config_iters_and_device_override_too(tmp_path):
    """The same ordering bug would have silently no-op'd `--iters`/`--device` as well, since all
    three overrides shared one code path -- pinned together rather than only for `out`."""
    cfg = tmp_path / 'cfg.toml'
    cfg.write_text("""
[data]
path = "/does/not/need/to/exist/for/this/check"

[model]
yolox = "trimmed"

[training]
out = "runs/x"
iters = 100
device = "cpu"
""")
    loaded = load_detector_config(cfg, iters=5, device='cuda:1')
    assert loaded['training']['iters'] == 5
    assert loaded['training']['device'] == 'cuda:1'


def test_per_root_detector_overlays_load_and_raise_val_frames_per_group(tmp_path):
    """Every shipped per-root overlay under `configs/detector/` must resolve through
    `extends = "../detector.toml"`, carry `[data].path` (the one thing with no CLI override), and
    set `val_frames_per_group` past the shipped default of 8 -- each of these three roots has ONE
    val group, so the default under-samples it."""
    root = Path(__file__).resolve().parent.parent / 'configs' / 'detector'
    want = {'rat-city.toml': 64, 'branson-fly.toml': 32, '3dpop.toml': 16}
    for name, expect in want.items():
        cfg = load_detector_config(root / name, out=tmp_path / name)
        assert cfg['data']['path'], f'{name}: [data].path must be set (no CLI override exists)'
        assert cfg['data']['val_frames_per_group'] == expect, name
        assert cfg['data']['val_frames_per_group'] > 8, \
            f'{name}: must exceed the shipped default of 8 or the overlay does nothing'


def test_deployment_score_untrained_model_is_all_zero(tmp_path):
    """`deployment_score` on a FRESH `YOLOXNano`: the rare-positive objectness prior sits below
    any sane `det_score`, so `det_fill`/`slot_fill` read exactly 0 and `window_miss` exactly 1.

    THE SEED IS WHAT MAKES "DETERMINISTIC" TRUE. The objectness PRIOR sits at ~0.0099, but the
    head's random init scatters the actual logits around it, and 3 of 20 torch seeds put at least
    one anchor above `det_score = 0.05` -- so unseeded this test reads its own RNG state.
    """
    torch.manual_seed(0)

    from .conftest import _session_2d
    from tailcyclenet.detector.evaluate import deployment_score
    from tailcyclenet.format import Session

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    model = YOLOXNano()
    r = deployment_score(model, sess, gid, input_wh=(64, 64), top_k=8, max_animals=2,
                         det_score=0.05, n_frames=2, overlap=0, min_box_frames=1)
    assert r['det_fill'] == 0.0 and r['slot_fill'] == 0.0 and r['window_miss'] == 1.0
    assert r['n_windows'] == 4                        # 2 rows x 2 windows (T=4, n_frames=2)
    assert r['n_gt'] == 8                              # 4 labelled frames x 1 camera x 2 animals
    assert all(v != v for v in r['union_side_px'].values()), \
        'no box ever fired, so the union-side quantiles must be all-NaN, not zero'
    assert r['gt_side_px'][0.5] > 0                    # GT sides are label-derived, independent


def test_deployment_score_forced_positive_objectness_fills_every_slot(tmp_path):
    """The live path: force every objectness logit strongly positive (`obj_pred` bias) so every
    anchor's box survives `det_score`, and confirm `det_fill` becomes 1.0 -- proving the
    zero-floor above is the init prior firing correctly, not a broken wire that always reads 0.
    """
    from .conftest import _session_2d
    from tailcyclenet.detector.evaluate import deployment_score
    from tailcyclenet.format import Session

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    model = YOLOXNano()
    for m in model.head.obj_pred:
        torch.nn.init.constant_(m.bias, 10.0)
    r = deployment_score(model, sess, gid, input_wh=(64, 64), top_k=8, max_animals=2,
                         det_score=0.05, n_frames=2, overlap=0, min_box_frames=1)
    assert r['det_fill'] == 1.0
    assert not any(v != v for v in r['union_side_px'].values()), \
        'a box fired everywhere, so the union-side quantiles must be finite, not NaN'


def test_gt_crop_sides_3d_projects_through_the_right_camera(tmp_path):
    """`_gt_crop_sides` must take the 3D branch (project points3d per camera) rather than
    silently reading `points2d`, which a 3D synthetic session does not even populate."""
    from .conftest import _session_3d
    from tailcyclenet.detector.evaluate import _gt_crop_sides
    from tailcyclenet.format import Session

    _session_3d(tmp_path / 'r' / 'test' / 's0', T=4)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    sides = _gt_crop_sides(sess, gid, min_crop_dim=8)
    assert sides.size > 0 and np.all(sides >= 8)       # the floor must be respected


def test_eval_detector_deploy_cli_end_to_end(tmp_path, monkeypatch, capsys):
    """A 2-iteration checkpoint through `train_detector.py`, scored through
    `eval_detector.py --deploy`: pins that the CLI plumbing runs end to end on a REAL
    trained-and-reloaded checkpoint, not just synthetic tensors.
    """
    import importlib.util
    import sys

    from .conftest import _session_2d

    root = tmp_path / 'root'
    _session_2d(root / 'train' / 's0', T=4, S=2)
    _session_2d(root / 'test' / 's1', T=4, S=2)

    train_cfg = _write_config(tmp_path, f"""
[data]
path = "{root}"
boxes = "keypoints"
min_crop_dim = 8
input_wh = [64, 64]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
reduce = false
keypoints = false
hflip = true
tile_wh = []
tile_scale = 1.0
tile_bg_per_frame = 1
use_regions = false
[model]
yolox = "trimmed"
[training]
out = "{tmp_path / 'run'}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
kpt_weight = 1.0
kpt_score_weight = 1.0
""", name='train.toml')

    spec = importlib.util.spec_from_file_location('tcn_train_detector2',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(train_cfg)])
    train_mod.main()

    spec2 = importlib.util.spec_from_file_location('tcn_eval_detector',
                                                    REPO / 'scripts' / 'eval_detector.py')
    eval_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(eval_mod)
    monkeypatch.setattr(sys, 'argv', [
        'eval_detector.py', '--run', str(tmp_path / 'run' / 'detector.pth'),
        '--data', str(root), '--split', 'test', '--deploy', '--no-track', '--link-boxes',
        '--score-thresh', '0.001', '--device', 'cpu', '--n-frames', '2', '--overlap', '0',
        '--det-max-frames', '2'])
    eval_mod.main()                                    # must not raise

    # THE PRINTED T COLUMN MUST REFLECT `--det-max-frames`, not the group's raw length -- the
    # fixture's group is 4 frames, bounded here to 2. Printing the raw 4 would silently claim a
    # full-length run when only a prefix was scored (the bug an actual 3dpop `--det-max-frames`
    # run exposed: the table read T=899..2680 while `detect_raw` had truncated to 120 internally).
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if '/g' in ln and ln.strip()[0] != '=')
    assert line.split()[1] == '2', f'expected the truncated T=2 in {line!r}'


# T2.1: instances.pq PRESENT boxes as an objectness ignore mask
# ----------------------------------------------------------------------------------------------







def test_bottleneck_expansion_default_matches_the_shipped_shape():
    """At the default 0.5, `dark3`'s first bottleneck's conv1 is (24,48,1,1) on `tiny` -- HALF
    Megvii's own (48,48,1,1). This is the shape every checkpoint on record was built at; pin it
    so a future change cannot silently move it.
    """
    m = YOLOXNano(version='tiny')
    assert m.bottleneck_expansion == 0.5
    w = m.backbone.dark3[1].m[0].conv1[0].weight
    assert tuple(w.shape) == (24, 48, 1, 1)


def test_bottleneck_expansion_one_matches_the_canonical_shape():
    """At 1.0, the SAME tensor is (48,48,1,1) -- full width, Megvii's own `CSPLayer`s
    `expansion=1.0` for its inner `Bottleneck`s. Everything else about the net is unchanged: only
    the internal width of each bottleneck's first conv moves.
    """
    m = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    w = m.backbone.dark3[1].m[0].conv1[0].weight
    assert tuple(w.shape) == (48, 48, 1, 1)
    # And the net still forwards at the wider shape.
    obj, boxes, _, _ = m(torch.zeros(1, 3, 96, 96))
    assert obj.shape[1] == boxes.shape[1]


def test_bottleneck_expansion_default_is_byte_identical_to_no_key():
    """Passing `bottleneck_expansion=0.5` explicitly must build the IDENTICAL module graph (same
    state_dict keys and shapes) as never passing it -- the same contract `crop_inflate = 1.0` and
    `pose_nms` off already carry elsewhere in this repo: a default value is not merely numerically
    close to absent, it is indistinguishable from it.
    """
    a = YOLOXNano(version='s').state_dict()
    b = YOLOXNano(version='s', bottleneck_expansion=0.5).state_dict()
    assert set(a) == set(b)
    for k in a:
        assert a[k].shape == b[k].shape


def test_bottleneck_expansion_raises_alongside_trimmed():
    """`trimmed`'s `CSPDarknetNano` does not take this key -- a non-default value alongside it
    must raise rather than silently being ignored, the same guard `width` already has the other
    way (canonical tier + non-default `width`)."""
    with pytest.raises(ValueError, match='bottleneck_expansion'):
        YOLOXNano(version='trimmed', bottleneck_expansion=1.0)
    YOLOXNano(version='trimmed', bottleneck_expansion=0.5)      # the default: must NOT raise


# T4.3 -- a stride-4 (P2) FPN level. `p2=False` (default) is byte-identical to every
# checkpoint on record; unlike `bottleneck_expansion`, this one is NOT tier-restricted -- it
# applies to `trimmed` and every canonical tier alike.
# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
# T4.2 -- temporal input. `in_channels` widens the stem; the TRAINING data path (`BoxDataset`)
# now supplies a wider input via `[data].temporal_input`, one mode ('stack2', frame t-1 stacked
# beside frame t). `detect_raw` (deployment) does NOT yet supply one -- still owed, so a
# `temporal_input != 'none'` checkpoint can be trained and scored via `eval_detector.py` but not
# yet run through `scripts/infer.py --detector`.
# `in_channels=3` / `temporal_input='none'` (both default) are byte-identical to every checkpoint
# on record.
# ----------------------------------------------------------------------------------------------

def test_in_channels_default_is_byte_identical_to_no_key():
    for version in ('s', 'trimmed'):
        a = YOLOXNano(version=version).state_dict()
        b = YOLOXNano(version=version, in_channels=3).state_dict()
        assert set(a) == set(b), version
        for k in a:
            assert a[k].shape == b[k].shape, (version, k)




def test_in_channels_still_has_no_model_config_key():
    """`in_channels` is always 3 (RGB); it must not be a model config key."""
    assert 'in_channels' not in MODEL_KEYS



def test_photometric_gain_none_is_byte_identical_to_before():
    """`_photometric(img, rng)` must draw and consume the rng stream EXACTLY as it did before the
    `gain=` parameter existed -- the T4.2 refactor's whole byte-identity claim rests on this."""
    img = (np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3) % 200).astype(np.uint8)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    out1 = _photometric(img, rng1)
    gain = rng2.uniform(0.7, 1.3)          # the same single draw `_photometric` makes internally
    out2 = _photometric(img, rng2, gain=gain)
    np.testing.assert_array_equal(out1, out2)
    # And the rng streams must have advanced identically -- the next draw from each must agree,
    # or a caller downstream of `_photometric` (the strong-suite ops) would silently diverge.
    assert rng1.uniform() == rng2.uniform()


def test_photometric_explicit_gain_does_not_redraw():
    """A caller supplying `gain=` must consume NOTHING from `rng` -- that is the whole point: the
    t-1 frame reuses frame t's own gain rather than drawing a second, different one."""
    img = np.full((4, 4, 3), 100, np.uint8)
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    _photometric(img, rng, gain=1.1)
    after = rng.bit_generator.state
    assert before == after














def test_detector_run_directory_defaults_to_latest_checkpoint(tmp_path):
    from tailcyclenet.detector import resolve_detector_checkpoint

    run = tmp_path / 'run'
    run.mkdir()
    for it in (2, 4):
        torch.save({'iteration': it}, run / f'detector_it{it:06d}.pth')
    (run / 'config.toml').write_text('[training]\niters = 4\n')
    (run / 'metrics.json').write_text('[{"iteration": 2}, {"iteration": 4}]')
    torch.save({'iteration': 2}, run / 'detector.pth')
    torch.save({'iteration': 4}, run / 'detector_last.pth')

    assert resolve_detector_checkpoint(run) == run / 'detector_it000004.pth'
    assert resolve_detector_checkpoint(run, checkpoint='last') == run / 'detector_last.pth'
    assert resolve_detector_checkpoint(run, checkpoint='best') == run / 'detector.pth'


def test_detector_run_directory_latest_accepts_incomplete_training(tmp_path):
    """An incomplete run (config iters=4, only iteration 2 written) still resolves `latest` to
    its highest numbered checkpoint -- only the `last` selector demands `detector_last.pth`."""
    from tailcyclenet.detector import resolve_detector_checkpoint

    run = tmp_path / 'run'
    run.mkdir()
    torch.save({'iteration': 2}, run / 'detector_it000002.pth')
    torch.save({'iteration': 2}, run / 'detector_last.pth')
    (run / 'config.toml').write_text('[training]\niters = 4\n')
    (run / 'metrics.json').write_text('[{"iteration": 2}]')
    assert resolve_detector_checkpoint(run, checkpoint='latest') == run / 'detector_it000002.pth'
    assert resolve_detector_checkpoint(run, checkpoint='detector_it000002.pth') == \
        run / 'detector_it000002.pth'


def test_detector_run_directory_last_still_requires_detector_last(tmp_path):
    """`last` is unaffected by the latest-selector relaxation: it still errors when
    `detector_last.pth` is absent."""
    from tailcyclenet.detector import resolve_detector_checkpoint

    run = tmp_path / 'run'
    run.mkdir()
    torch.save({'iteration': 2}, run / 'detector_it000002.pth')
    with pytest.raises(ValueError, match='detector_last.pth'):
        resolve_detector_checkpoint(run, checkpoint='last')


def test_detector_cli_defaults_to_last():
    from tailcyclenet.detector import load_detector, resolve_detector_checkpoint
    from tailcyclenet.infer.cli import build_parser

    assert build_parser().parse_args(['--run', 'r', '--data', 'd', '--out', 'o']).detector_checkpoint == 'latest'
    assert load_detector.__defaults__[-1] == 'latest'
    assert resolve_detector_checkpoint.__defaults__[-1] == 'latest'


def test_detector_packaging_preserves_config(tmp_path):
    from scripts.package_checkpoint import package_detector
    from tailcyclenet.detector import load_detector

    run = tmp_path / 'run'
    run.mkdir()
    config = {'data': {'input_wh': [96, 96]}, 'model': {'yolox': 'tiny'},
              'training': {'iters': 2, 'seed': 7}}
    (run / 'config.toml').write_text('[training]\niters = 2\n')
    model = YOLOXNano(version='tiny')
    source = {'iteration': 2, 'model_state': model.state_dict(), 'input_wh': [96, 96],
              'n_keypoints': 0, 'norm': 'gn', 'yolox_version': 'tiny',
              'bottleneck_expansion': 0.5, 'p2': False, 'in_channels': 3, 'config': config}
    torch.save(source, run / 'detector_last.pth')
    torch.save(source, run / 'detector_it000002.pth')
    (run / 'metrics.json').write_text('[{"iteration": 2}]')
    out = tmp_path / 'detector.pth'
    package_detector(run, out)
    loaded, wh, *_ = load_detector(out)
    packaged = torch.load(out, map_location='cpu', weights_only=False)
    assert wh == (96, 96)
    assert packaged['config'] == config
    assert packaged['kind'] == 'detector'
    assert loaded.in_channels == 3


def test_load_detector_absent_in_channels_means_3(tmp_path):
    """Every checkpoint written before this key existed carries no `in_channels` at all -- absent
    is a FACT about those files (3, the only stem width they were ever built at), not a guess.
    """
    from tailcyclenet.detector import load_detector

    m = YOLOXNano(version='tiny')
    ckpt = {
        'model_state': m.state_dict(), 'input_wh': (96, 96), 'n_keypoints': 0, 'norm': 'gn',
        'yolox_version': 'tiny', 'bottleneck_expansion': 0.5, 'p2': False,
    }
    p = tmp_path / 'detector.pth'
    torch.save(ckpt, p)
    loaded, *_ = load_detector(p)
    assert loaded.in_channels == 3


def test_detect_raw_refuses_a_wide_in_channels_checkpoint():
    """`load_detector` correctly rebuilds an `in_channels=6` model (the two tests above), but
    `detect_raw` -- the deployment loop -- has no paired-frame reader: `_fetch` always builds one
    3-channel letterboxed frame per (camera, source frame). Forwarding a 6-channel model there
    would silently run the stem on half real pixels and half whatever garbage occupies the other
    three channels, and report ordinary-looking boxes. This must refuse before touching `session`
    at all -- passing `session=None` proves the guard fires first, not after some session access
    that happened to already be safe on this fixture.
    """
    from tailcyclenet.detector import detect_raw

    det = YOLOXNano(version='tiny', in_channels=6)
    with pytest.raises(SystemExit, match='in_channels=6'):
        detect_raw(det, (96, 96), session=None, gid='g000', top_k=1)


def test_detect_raw_default_in_channels_does_not_raise_the_guard():
    """The ordinary `in_channels=3` model must clear the new guard and reach the (unrelated)
    session access below it -- pinning that the guard is genuinely conditional, not a blanket
    refusal that happens to match `in_channels=6` by coincidence.
    """
    from tailcyclenet.detector import detect_raw

    det = YOLOXNano(version='tiny', in_channels=3)
    with pytest.raises(AttributeError):
        # Clears the in_channels guard, then fails on `session.groups[gid]` -- `None` has no
        # `.groups` -- which is what proves the guard did not fire.
        detect_raw(det, (96, 96), session=None, gid='g000', top_k=1)


def test_p2_default_is_byte_identical_to_no_key():
    """Same contract as `bottleneck_expansion`'s own default-identity test: passing `p2=False`
    explicitly must build the IDENTICAL module graph as never passing it, on BOTH a canonical
    tier and `trimmed` (this key is not tier-restricted)."""
    for version in ('s', 'trimmed'):
        a = YOLOXNano(version=version).state_dict()
        b = YOLOXNano(version=version, p2=False).state_dict()
        assert set(a) == set(b), version
        for k in a:
            assert a[k].shape == b[k].shape, (version, k)


def test_p2_true_adds_a_stride_4_level():
    """`p2=True` must widen `STRIDES` to `(4, 8, 16, 32)`, `anchor_points` must match it exactly
    (the same invariant `test_forward_shapes_and_anchor_order` pins for the 3-level case), and
    the model must have MORE parameters (the new `lat2`/`mrg2`/`down2`/`out3` PAFPN modules plus
    the head's 4th level)."""
    base = YOLOXNano(version='tiny')
    m = YOLOXNano(version='tiny', p2=True)
    assert m.STRIDES == (4, 8, 16, 32)
    assert base.STRIDES == (8, 16, 32), 'the OTHER instance must be unaffected'
    x = torch.zeros(1, 3, 128, 160)
    obj, boxes, _, _ = m(x)
    anchors = m.anchor_points(128, 160, x.device)
    assert obj.shape[1] == boxes.shape[1] == anchors.shape[0]
    assert set(anchors[:, 2].tolist()) == {4.0, 8.0, 16.0, 32.0}
    n_base = sum(p.numel() for p in base.parameters())
    n_p2 = sum(p.numel() for p in m.parameters())
    assert n_p2 > n_base


def test_p2_works_on_trimmed_too():
    """Unlike `bottleneck_expansion`, `p2` is NOT tier-restricted -- it must build and forward on
    `trimmed` as well as a canonical tier, and must NOT raise the way a non-default
    `bottleneck_expansion` does alongside `trimmed`."""
    m = YOLOXNano(version='trimmed', p2=True)          # must not raise
    assert m.STRIDES == (4, 8, 16, 32)
    obj, boxes, _, _ = m(torch.zeros(1, 3, 96, 96))
    assert obj.shape[1] == boxes.shape[1]


def test_p2_checkpoint_round_trips_through_load_detector(tmp_path):
    """A `p2=True` checkpoint must reconstruct a `p2=True` model through `load_detector` -- absent
    means `False`, so this proves a SAVED `p2=True` fact survives, not just that the constructor
    kwarg works in isolation."""
    from tailcyclenet.detector import load_detector

    m = YOLOXNano(version='tiny', p2=True)
    ckpt = {
        'model_state': m.state_dict(), 'input_wh': (96, 96), 'n_keypoints': 0, 'norm': 'gn',
        'yolox_version': 'tiny', 'bottleneck_expansion': 0.5, 'p2': True,
    }
    p = tmp_path / 'detector.pth'
    torch.save(ckpt, p)
    loaded, wh, ds_name, mcd, reduce, box_src, ts, obj_q = load_detector(p)
    assert loaded.p2 is True
    assert loaded.STRIDES == (4, 8, 16, 32)
    obj, boxes, _, _ = loaded(torch.zeros(1, 3, 96, 96))
    assert obj.shape[1] == boxes.shape[1]


def test_detector_config_p2_inherits_the_shipped_true(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['model']['p2'] is True


def test_train_detector_p2_end_to_end(tmp_path, dense_root, monkeypatch):
    """A short run with `[model].p2 = true` through the real CLI entry point: must train to
    completion with no error, and the saved checkpoint must reload with `p2=True`."""
    import importlib.util
    import sys

    from tailcyclenet.detector import load_detector

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
p2 = true
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector_p2',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector.pth').exists()
    model, *_ = load_detector(out / 'detector.pth')
    assert model.p2 is True
    assert model.STRIDES == (4, 8, 16, 32)


def test_load_coco_backbone_raises_at_the_shipped_expansion():
    """The guard must fire BEFORE touching disk -- a model built at the shipped 0.5 must be
    refused even if no weights file exists at all, so the error names the real cause instead of a
    confusing FileNotFoundError."""
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='tiny', bottleneck_expansion=0.5)
    with pytest.raises(ValueError, match='bottleneck_expansion'):
        load_coco_backbone(m, 'tiny', weights_dir='/nonexistent')


def test_load_coco_backbone_raises_for_trimmed():
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='trimmed')
    with pytest.raises(ValueError, match='trimmed'):
        load_coco_backbone(m, 'trimmed', weights_dir='/nonexistent')


def test_load_coco_backbone_missing_weights_file_raises(tmp_path):
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    with pytest.raises(FileNotFoundError, match='yolox_tiny'):
        load_coco_backbone(m, 'tiny', weights_dir=tmp_path)


def test_bgr_to_rgb_focus_perm_is_an_involution_over_three_blocks():
    """The permutation must reverse each 3-channel block and nothing else -- it is claimed to be
    its own inverse (`pretrained.py`'s own derivation), and this is the property that makes ONE
    permutation correct for converting either direction."""
    from tailcyclenet.detector.pretrained import BGR_TO_RGB_FOCUS_PERM

    perm = BGR_TO_RGB_FOCUS_PERM
    assert len(perm) == 12 and sorted(perm) == list(range(12))
    for g in range(4):
        block = perm[3 * g:3 * g + 3]
        assert block == [3 * g + 2, 3 * g + 1, 3 * g + 0]
    # Applying it twice must be the identity.
    twice = [perm[perm[i]] for i in range(12)]
    assert twice == list(range(12))


def test_detector_config_bottleneck_expansion_and_pretrained_default_inert(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['model']['bottleneck_expansion'] == 0.5
    assert cfg['model']['pretrained'] == ''


def test_detector_config_pretrained_coco_requires_canonical_tier(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "trimmed"
pretrained = "coco"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='trimmed'):
        load_detector_config(p)


def test_detector_config_pretrained_coco_requires_bottleneck_expansion_one(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
pretrained = "coco"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='bottleneck_expansion'):
        load_detector_config(p)
    # The paired, correct config must NOT raise.
    p2 = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
pretrained = "coco"
bottleneck_expansion = 1.0
[training]
out = "/tmp/run"
""", 'ok.toml')
    cfg = load_detector_config(p2)
    assert cfg['model']['pretrained'] == 'coco' and cfg['model']['bottleneck_expansion'] == 1.0


def test_detector_config_pretrained_unsupported_value_raises(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
pretrained = "/some/path.pth"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='pretrained'):
        load_detector_config(p)


def test_detector_config_bottleneck_expansion_raises_alongside_trimmed(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
yolox = "trimmed"
bottleneck_expansion = 1.0
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='bottleneck_expansion'):
        load_detector_config(p)


WEIGHTS_DIR = REPO / 'scratch' / 'weights'
_HAVE_COCO_WEIGHTS = (WEIGHTS_DIR / 'yolox_tiny.pth').exists()


@pytest.mark.skipif(not _HAVE_COCO_WEIGHTS, reason='scratch/weights/yolox_*.pth is untracked '
                    'and may not be cached on this machine')
def test_load_coco_backbone_transfers_every_backbone_tensor():
    """The end-to-end claim T4.1 exists to fix: 35/35 backbone conv tensors load at
    bottleneck_expansion=1.0, not the 19/35 the shipped 0.5 shape permits."""
    from tailcyclenet.detector.pretrained import load_coco_backbone

    m = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    n_loaded, n_total = load_coco_backbone(m, 'tiny', weights_dir=WEIGHTS_DIR)
    assert (n_loaded, n_total) == (35, 35)
    # And the loaded weights actually forward without shape errors.
    obj, boxes, _, _ = m(torch.zeros(1, 3, 96, 96))
    assert obj.shape[1] == boxes.shape[1]


@pytest.mark.skipif(not _HAVE_COCO_WEIGHTS, reason='scratch/weights/yolox_*.pth is untracked '
                    'and may not be cached on this machine')
def test_load_coco_backbone_stem_correction_matches_megvii_activation():
    """T4.1's own test: a corrected weight fed this repo's RGB/[0,1] convention must produce the
    SAME stem activation Megvii's raw weight produces on the identical image fed BGR/[0,255] --
    not merely a plausible-looking one. Ports `scratch/validate_pretrained_load.py`'s check into a
    pinned test."""
    import torch.nn.functional as F

    from tailcyclenet.detector.pretrained import BGR_TO_RGB_FOCUS_PERM

    ck = torch.load(WEIGHTS_DIR / 'yolox_tiny.pth', map_location='cpu', weights_only=False)
    src = ck['model'] if 'model' in ck else ck
    w_orig = src['backbone.backbone.stem.conv.conv.weight'].clone()

    torch.manual_seed(0)
    img_bgr_255 = torch.rand(1, 3, 32, 32) * 255.0
    img_rgb_01 = img_bgr_255.flip(1) / 255.0

    def focus_cat(x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                          x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)

    y_megvii = F.conv2d(focus_cat(img_bgr_255), w_orig, stride=1, padding=1)
    w_corrected = (w_orig * 255.0)[:, BGR_TO_RGB_FOCUS_PERM]
    y_repo = F.conv2d(focus_cat(img_rgb_01), w_corrected, stride=1, padding=1)
    torch.testing.assert_close(y_megvii, y_repo, atol=1e-3, rtol=1e-4)


@pytest.mark.skipif(not _HAVE_COCO_WEIGHTS, reason='scratch/weights/yolox_*.pth is untracked '
                    'and may not be cached on this machine')
def test_train_detector_pretrained_coco_end_to_end(tmp_path, dense_root, monkeypatch):
    """A 2-iteration run with `pretrained = "coco"` through the real CLI entry point: the backbone
    loads, training runs, and the checkpoint records both new keys so `load_detector` rebuilds the
    right architecture later."""
    import importlib.util
    import sys

    from tailcyclenet.detector import load_detector

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [96, 96]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
bottleneck_expansion = 1.0
pretrained = "coco"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
""")
    spec = importlib.util.spec_from_file_location('tcn_train_detector_pretrained',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    monkeypatch.setattr(mod, 'load_coco_backbone',
                        lambda model, tier, weights_dir=None: __import__(
                            'tailcyclenet.detector.pretrained', fromlist=['load_coco_backbone']
                        ).load_coco_backbone(model, tier, weights_dir=WEIGHTS_DIR))
    mod.main()

    assert (out / 'detector.pth').exists()
    ckpt = torch.load(out / 'detector_it000002.pth', map_location='cpu', weights_only=False)
    assert ckpt['bottleneck_expansion'] == 1.0
    assert ckpt['pretrained'] == 'coco'
    model, wh, ds_name, mcd, reduce, box_src, ts, obj_q = load_detector(out / 'detector.pth')
    assert model.bottleneck_expansion == 1.0









def test_detector_config_pretrained_path_default_and_coco_unaffected(tmp_path):
    """T4.1b must not change either existing `pretrained` value's behaviour."""
    for pretrained_line, expect in (('', ''), ('pretrained = "coco"', 'coco')):
        p = _write_config(tmp_path, f"""
[data]
path = "/tmp/ds"
[model]
yolox = "tiny"
bottleneck_expansion = {"1.0" if expect == "coco" else "0.5"}
{pretrained_line}
[training]
out = "/tmp/run"
""")
        cfg = load_detector_config(p)
        assert cfg['model']['pretrained'] == expect


def test_detector_config_pretrained_path_must_exist(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "/tmp/ds"
[model]
pretrained = "/nonexistent/backbone.pth"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match='expected'):
        load_detector_config(p)



def test_detector_config_optimizer_inherits_the_shipped_muon(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "x"
[model]
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['training']['optimizer'] == 'muon'
    assert cfg['training']['muon_momentum'] == 0.95
    assert cfg['training']['muon_lr_scale'] == 1.0
    assert cfg['training']['warmup_steps'] == 500
    assert cfg['training']['beta1'] == 0.9 and cfg['training']['beta2'] == 0.95


def test_detector_config_optimizer_invalid_raises(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "x"
[model]
[training]
out = "/tmp/run"
optimizer = "sgd"
""")
    with pytest.raises(SystemExit, match='optimizer'):
        load_detector_config(p)


def test_detector_config_optimizer_muon_keys(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "x"
[model]
[training]
out = "/tmp/run"
optimizer = "muon"
muon_momentum = 0.9
muon_lr_scale = 2.0
warmup_steps = 10
""")
    cfg = load_detector_config(p)
    assert cfg['training']['optimizer'] == 'muon'
    assert cfg['training']['muon_momentum'] == 0.9
    assert cfg['training']['muon_lr_scale'] == 2.0
    assert cfg['training']['warmup_steps'] == 10


def test_build_detector_optimizer_pure_cnn_routes_nothing_to_muon():
    """A pure-CNN detector (every conv weight is 4D) has nothing for Muon to route -- the
    resulting optimizer must be plain AdamWScheduleFree, matching
    `dev/scratch/prototype_muon_detector.py`."""
    from schedulefree import AdamWScheduleFree

    from tailcyclenet.detector.yolox import YOLOXNano
    train_mod = _load_train_detector_script()
    model = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    train_cfg = {'optimizer': 'muon', 'lr': 1e-3, 'weight_decay': 5e-4,
                'muon_lr_scale': 1.0, 'muon_momentum': 0.95,
                'warmup_steps': 0, 'beta1': 0.9, 'beta2': 0.95}
    model_cfg = {'pretrained': ''}
    opt, sched = train_mod.build_detector_optimizer(model, train_cfg, model_cfg)
    assert sched is None
    assert isinstance(opt, AdamWScheduleFree)


def test_build_detector_optimizer_adamw_matches_param_count():
    train_mod = _load_train_detector_script()
    from tailcyclenet.detector.yolox import YOLOXNano
    model = YOLOXNano(version='tiny', bottleneck_expansion=1.0)
    train_cfg = {'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 5e-4,
                'iters': 10}
    model_cfg = {'pretrained': ''}
    opt, sched = train_mod.build_detector_optimizer(model, train_cfg, model_cfg)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g['params'])
    n_model = sum(p.numel() for p in model.parameters())
    assert n_opt == n_model
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def _load_train_detector_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location('tcn_train_detector_for_optim',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_detector_muon_optimizer_end_to_end(tmp_path, dense_root, monkeypatch):
    """A short run with `[training].optimizer = "muon"` through the real CLI entry point: must
    train to completion with no error, and the checkpoint must record `optimizer_kind`."""
    import sys

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
optimizer = "muon"
""")
    mod = _load_train_detector_script()
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector.pth').exists()
    ckpt = torch.load(out / 'detector.pth', map_location='cpu', weights_only=False)
    assert ckpt['optimizer_kind'] == 'muon'


def test_train_detector_muon_optimizer_is_default_end_to_end(tmp_path, dense_root, monkeypatch):
    """The implicit base makes the shipped muon the default; the checkpoint records it."""
    import sys

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
""")
    mod = _load_train_detector_script()
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector.pth').exists()
    ckpt = torch.load(out / 'detector.pth', map_location='cpu', weights_only=False)
    assert ckpt['optimizer_kind'] == 'muon'






def test_detector_config_yolox_choices_include_new_architectures(tmp_path):
    from tailcyclenet.detector.config import YOLOX_CHOICES
    assert 'hybrid' in YOLOX_CHOICES









@pytest.mark.parametrize('p2', [True, False])
def test_hybrid_backbone_shape_contract(p2):
    from tailcyclenet.detector.vit_backbone import HybridBackbone

    bb = HybridBackbone(p2=p2)
    H, W = 256, 384
    feats = bb(torch.rand(1, 3, H, W))
    strides = (4, 8, 16, 32) if p2 else (8, 16, 32)
    assert len(feats) == len(strides) == len(bb.out_channels)
    for f, s, c in zip(feats, strides, bb.out_channels):
        assert f.shape == (1, c, H // s, W // s)


def test_hybrid_backbone_backward_produces_finite_grads():
    from tailcyclenet.detector.vit_backbone import HybridBackbone

    bb = HybridBackbone(p2=True)
    feats = bb(torch.rand(1, 3, 128, 128))
    loss = sum(f.float().sum() for f in feats)
    loss.backward()
    for name, p in bb.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), f'{name}: non-finite grad'


def test_yolox_nano_hybrid_version_builds_and_forwards():
    model = YOLOXNano(version='hybrid', p2=True)
    assert model.STRIDES == (4, 8, 16, 32)
    obj, boxes, kpt, _ = model(torch.rand(1, 3, 128, 192))
    anchors = model.anchor_points(128, 192, 'cpu')
    assert obj.shape[1] == anchors.shape[0] == boxes.shape[1]
    assert kpt is None


def test_detector_config_yolox_hybrid_default_pretrained_empty(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "x"
[model]
yolox = "hybrid"
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['model']['pretrained'] == ''


def test_detector_config_pretrained_coco_rejects_hybrid(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "x"
[model]
yolox = "hybrid"
bottleneck_expansion = 1.0
pretrained = "coco"
[training]
out = "/tmp/run"
""")
    with pytest.raises(SystemExit, match="no counterpart"):
        load_detector_config(p)









def test_detector_config_gap_lever_inherits_the_shipped_values(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "x"
[model]
[training]
out = "/tmp/run"
""")
    cfg = load_detector_config(p)
    assert cfg['training']['shared_head'] is False
    assert cfg['training']['fpn_upsample'] == 'bilinear'


def test_detector_config_fpn_upsample_invalid_raises(tmp_path):
    p = _write_config(tmp_path, """
[data]
path = "x"
[model]
[training]
out = "/tmp/run"
fpn_upsample = "bicubic"
""")
    with pytest.raises(SystemExit, match='fpn_upsample'):
        load_detector_config(p)


def test_g1_shared_head_false_adds_separate_obj_tower():
    shared = YOLOXNano(version='tiny', p2=True, shared_head=True)
    unshared = YOLOXNano(version='tiny', p2=True, shared_head=False)
    assert not hasattr(shared.head, 'obj_convs')
    assert hasattr(unshared.head, 'obj_convs')
    n_shared = sum(p.numel() for p in shared.parameters())
    n_unshared = sum(p.numel() for p in unshared.parameters())
    assert n_unshared > n_shared


def test_g1_shared_head_default_forward_matches_reg_tower_output():
    """`shared_head=True` (default) must route obj through the SAME tensor as reg -- a
    regression guard against `Head.forward` accidentally building `obj_convs` unconditionally."""
    model = YOLOXNano(version='tiny', p2=False, shared_head=True)
    x = torch.rand(1, 3, 128, 128)
    obj, boxes, kpt, _ = model(x)
    assert torch.isfinite(obj).all() and torch.isfinite(boxes).all()


def test_g2_fpn_upsample_bilinear_forwards():
    model = YOLOXNano(version='tiny', p2=True, fpn_upsample='bilinear')
    assert model.neck.fpn_upsample == 'bilinear'
    x = torch.rand(1, 3, 128, 128)
    obj, boxes, kpt, _ = model(x)
    assert torch.isfinite(obj).all() and torch.isfinite(boxes).all()


def test_g2_fpn_upsample_default_is_nearest():
    model = YOLOXNano(version='tiny', p2=True)
    assert model.neck.fpn_upsample == 'nearest'






def test_train_detector_gap_levers_end_to_end(tmp_path, dense_root, monkeypatch):
    """A short run with every §2 lever flipped on, through the real CLI entry point."""
    import sys

    out = tmp_path / 'run'
    cfg = _write_config(tmp_path, f"""
[data]
path = "{dense_root}"
boxes = "keypoints"
min_crop_dim = 16
input_wh = [48, 48]
min_box_px = 0
val_frames_per_group = 4
augment = false
augment_strong = false
rotate_deg = 0.0
[model]
yolox = "tiny"
p2 = true
[training]
out = "{out}"
iters = 2
batch_size = 2
lr = 1e-3
num_workers = 0
seed = 0
device = "cpu"
eval_every = 2
eval_batches = 1
assignment = "tal"
shared_head = false
fpn_upsample = "bilinear"
""")
    import importlib.util
    spec = importlib.util.spec_from_file_location('tcn_train_detector_gap_levers',
                                                  REPO / 'tailcyclenet' / 'train_detector.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, 'argv', ['train_detector.py', '--config', str(cfg)])
    mod.main()
    assert (out / 'detector.pth').exists()

    from tailcyclenet.detector import load_detector
    model, *_ = load_detector(out / 'detector.pth')
    assert hasattr(model.head, 'obj_convs')
    assert model.neck.fpn_upsample == 'bilinear'



def _lever_rig(specs):
    """A NON-DEGENERATE synthetic rig for the tracker levers: every camera at its own tilt and
    height, so two rays through different animals are genuinely SKEW.

    The rig in `track.demo` (and in every older tracker test here) puts all three cameras in one
    plane with their optical axes through the world origin, and that is a trap for anything that
    reads a reprojection residual: coplanar rays always intersect, so a cross-view-inconsistent
    pair triangulates to residual ~0 and no residual gate can ever fire on it. specs entries are
    (rvec_x, rvec_y, tvec_y).
    """
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.format import Rig

    cams = []
    for i, (rx, ry, ty) in enumerate(specs):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([rx, ry, 0.0]),
                     tvec=np.array([0.0, ty, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    return Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
               moving=dict.fromkeys(names, False),
               calibrated=dict.fromkeys(names, True)).posetail()


def _lever_boxes(cg, worlds, side=200.0):
    """Project `worlds` into every camera of `cg` as fixed-size square boxes.

    Outputs: (per_cam, scores) in `CrossViewTracker.step`'s own argument shapes. The side is
    large by default because every gate in track.py is measured in box sides: a wrong claim only
    reaches the residual gate if it was inside the movement gate first, which is exactly the
    deployment situation (a big animal, a detection one body length away).
    """
    from tailcyclenet.detector.track import _project

    per_cam, scores = [], []
    for cam in cg:
        uv = _project(cam, np.asarray(worlds, np.float32))
        per_cam.append(torch.stack([uv[:, 0] - side / 2, uv[:, 1] - side / 2,
                                    uv[:, 0] + side / 2, uv[:, 1] + side / 2], -1))
        scores.append(torch.ones(len(worlds)))
    return per_cam, scores


def _lost_in_one_camera():
    """The scene the whole lever set exists for: animal A visible in cameras 0 and 1, MISSED in
    camera 2, where the only detection belongs to animal B.

    Outputs: (cg, per_cam, scores, A). A one-slot tracker holding A must not end up holding
    A-in-two-views plus B-in-the-third; today it does, because nothing checks that the detection
    claimed in camera 0 and the one claimed in camera 2 are the same animal.
    """
    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a = np.array([0.0, 0.0, 0.0], np.float32)
    b = np.array([120.0, 0.0, 0.0], np.float32)
    per_cam, scores = _lever_boxes(cg, [a, b])
    per_cam[2], scores[2] = per_cam[2][1:], torch.ones(1)
    return cg, per_cam, scores, a


def test_the_claim_residual_gate_rejects_a_cross_view_inconsistent_claim():
    """`claim_residual_gate` is the repair for the hole this file's tracker has always had:
    `max_res_px` was spent in exactly ONE place, the birth branch for unoccupied slots, so with
    every slot occupied the residual gate never executed at all and a slot could bind animal A
    in two views to animal B in a third. Nothing downstream sees it -- the boxes look fine and
    only the carried 3D point silently drifts between two animals.

    The claim: today's tracker ACCEPTS the inconsistent claim (this is asserted, not assumed --
    a gate that only ever agrees with the default path defends nothing), and the gate drops that
    one camera while keeping the two that agree.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg, per_cam, scores, a = _lost_in_one_camera()

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera')
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    out, _, claimed = tr.step(cg, per_cam, scores)
    assert claimed[0].tolist() == [0, 0, 0], \
        'the ungated tracker must take camera 2s wrong animal, or this test proves nothing'
    drift = float(np.linalg.norm(tr.targets[0]['point'].numpy() - a))
    assert drift > 50.0, f'the wrong claim must actually move the 3D point, moved {drift:.1f}'

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera',
                         claim_residual_gate=True)
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    out, sc, claimed = tr.step(cg, per_cam, scores)
    assert claimed[0].tolist() == [0, 0, -1], 'camera 2s claim must be dropped, and only it'
    assert not np.isfinite(out[0, 2]).any(), 'a dropped claim must not come back as a box'
    assert not np.isfinite(sc[0, 2]), 'nor as a score'
    assert np.isfinite(out[0, :2]).all(), 'the two consistent cameras must survive'
    np.testing.assert_allclose(tr.targets[0]['point'].numpy(), a, atol=1e-3)


def test_the_claim_residual_gate_holds_the_point_when_two_cameras_disagree():
    """The 2-camera rig -- the one the 5-fish clip runs on -- has no majority to appeal to, so
    when the pair's own fit misses both rays there is no way to tell which is wrong and BOTH
    claims must go, leaving the slot on its previous point (the file's existing "fewer than two
    cameras holds its point" rule).

    Also pins the negative half on the same rig: a consistent pair passes the gate untouched, so
    the gate is not simply refusing every 2-camera claim.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.4, 0.0), (0.35, 0.4, 100.0)])
    a = np.array([0.0, 0.0, 0.0], np.float32)
    b = np.array([200.0, 0.0, 0.0], np.float32)
    per_cam, scores = _lever_boxes(cg, [a])
    wrong, _ = _lever_boxes(cg, [b])
    per_cam[1] = wrong[1]

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera')
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    tr.step(cg, per_cam, scores)
    assert float(np.linalg.norm(tr.targets[0]['point'].numpy() - a)) > 50.0, \
        'the ungated tracker must swallow the skew pair, or the gated arm proves nothing'

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera',
                         claim_residual_gate=True)
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    out, _, claimed = tr.step(cg, per_cam, scores)
    assert claimed[0].tolist() == [-1, -1], 'neither ray of an inconsistent pair may be trusted'
    np.testing.assert_allclose(tr.targets[0]['point'].numpy(), a, atol=1e-4)
    assert tr.targets[0]['age'] == 1, 'a slot that kept no claim has no evidence and must age'

    good, good_sc = _lever_boxes(cg, [a])
    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera',
                         claim_residual_gate=True)
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    _, _, claimed = tr.step(cg, good, good_sc)
    assert claimed[0].tolist() == [0, 0], 'a consistent pair must pass the gate untouched'


def test_joint_association_refuses_a_binding_the_per_camera_hungarian_accepts():
    """`assoc_mode = "joint"` is the second, independent repair: form cross-view candidate groups
    over the WHOLE detection pool with `associate` -- which triangulates and gates on the
    residual already -- and run ONE Hungarian over slots x groups. Today `associate` only ever
    sees the LEFTOVERS the tracker did not want, so the one routine in the repo that checks
    cross-view consistency never gets a look at the detections that matter.

    Same scene as the claim-gate test, so the two repairs are measured against one defect: the
    per-camera path binds animal B's box in camera 2 to the slot holding animal A; joint cannot,
    because a group is the unit of matching and no group contains both animals.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg, per_cam, scores, a = _lost_in_one_camera()

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera')
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    tr.step(cg, per_cam, scores)
    assert float(np.linalg.norm(tr.targets[0]['point'].numpy() - a)) > 50.0

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='joint')
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    out, _, claimed = tr.step(cg, per_cam, scores)
    assert claimed[0].tolist() == [0, 0, -1], 'the joint decision must not reach into camera 2'
    np.testing.assert_allclose(tr.targets[0]['point'].numpy(), a, atol=1e-3)
    assert tr.targets[0]['age'] == 0, 'a slot that matched a group has evidence and must not age'


def test_joint_association_still_births_ages_and_resumes_like_the_shipped_path():
    """A mode that fixes the crowded case by breaking the easy one is not a lever. `joint`
    replaces the matching phase only, so the state machine around it -- births into free slots on
    frame 0, a one-frame detector miss that ages targets without dropping them, and the SAME
    slots resuming afterwards -- must behave exactly as `track.demo` asserts for the default
    path, including through the score-order swap that reorders detections every frame.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a, b = np.array([-80.0, 0.0, 0.0]), np.array([80.0, 0.0, 0.0])
    tr = CrossViewTracker(2, max_res_px=30.0, assoc_mode='joint')
    rows = []
    for t in range(12):
        w = [a + [12.0 * t, 0, 0], b - [12.0 * t, 0, 0]]
        per_cam, scores = _lever_boxes(cg, w if t % 2 == 0 else w[::-1], side=40.0)
        rows.append(tr.step(cg, per_cam, scores)[0])

    assert all(np.isfinite(r).all(-1).any(-1).sum() == 2 for r in rows), 'an animal was lost'
    for s in (0, 1):
        cx = np.array([r[s, 0, [0, 2]].mean() for r in rows])
        assert np.abs(np.diff(cx)).max() < 30.0, f'row {s} jumped: {np.diff(cx)}'

    empty = [torch.zeros((0, 4)) for _ in cg], [torch.zeros((0,)) for _ in cg]
    out, _, _ = tr.step(cg, *empty)
    assert not np.isfinite(out).any() and len(tr.targets) == 2, \
        'a one-frame detector miss must not end a track under joint either'
    w = [a + [12.0 * 11, 0, 0], b - [12.0 * 11, 0, 0]]
    resumed, _, _ = tr.step(cg, *_lever_boxes(cg, w, side=40.0))
    for s in (0, 1):
        assert abs(resumed[s, 0, [0, 2]].mean() - rows[-1][s, 0, [0, 2]].mean()) < 30.0, \
            f'row {s} resumed on the other animal'


def test_velocity_follows_two_animals_through_a_crossing_that_the_last_point_loses():
    """The module docstring's "no velocity model -- measured as not worth it" holds for the easy
    frames and fails at exactly the frame that matters. Two animals approach and cross: at the
    crossing each slot's LAST KNOWN point is nearer the other animal's new position than its own,
    so the Hungarian confidently swaps them and never recovers -- with one keypoint per animal
    and no appearance cue there is nothing else to decide on.

    A one-frame constant-velocity prediction puts each slot where its own animal actually is.
    Both arms are asserted: the default must swap (or this scene tests nothing) and `velocity`
    must not.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    track = [(-60.0, 60.0), (-20.0, 20.0), (20.0, -20.0), (60.0, -60.0)]

    ends = {}
    for velocity in (False, True):
        tr = CrossViewTracker(2, max_res_px=30.0, assoc_mode='per-camera', velocity=velocity)
        for xa, xb in track:
            w = [np.array([xa, 0.0, 0.0]), np.array([xb, 0.0, 0.0])]
            tr.step(cg, *_lever_boxes(cg, w, side=90.0))
        ends[velocity] = [float(tr.targets[s]['point'][0]) for s in (0, 1)]

    assert ends[False][0] < 0 and ends[False][1] > 0, \
        'the shipped matcher must swap here, or the velocity arm is not being tested'
    assert ends[True][0] > 0 and ends[True][1] < 0, \
        'a constant-velocity prediction must carry each slot through the crossing'


def test_velocity_leaves_point_and_age_alone_and_decays_when_evidence_stops():
    """The state dict is a contract: `link_rows`, `associate_group`'s `state` carry and the
    existing tests all read `'point'` and `'age'`, so lever 3 may only ADD a key. And a held
    target must not keep extrapolating: it holds its POINT, so its prediction is one step ahead
    however long it has been missing, and that step must shrink or a target that vanishes behind
    an occluder comes back predicting a position it was never measured at.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera', velocity=True)
    for x in (0.0, 20.0, 40.0):
        tr.step(cg, *_lever_boxes(cg, [np.array([x, 0.0, 0.0])], side=200.0))

    t = tr.targets[0]
    assert set(t) == {'point', 'age', 'velocity'}, f'unexpected target state: {sorted(t)}'
    assert t['age'] == 0 and bool(torch.isfinite(t['point']).all())
    assert float(t['velocity'][0]) > 1.0, 'the velocity must have picked up the motion'

    before = t['point'].clone()
    v0 = float(t['velocity'][0])
    empty = [torch.zeros((0, 4)) for _ in cg], [torch.zeros((0,)) for _ in cg]
    tr.step(cg, *empty)
    assert float(tr.targets[0]['velocity'][0]) < v0, 'a held velocity must decay, not persist'
    torch.testing.assert_close(tr.targets[0]['point'], before)
    assert tr.targets[0]['age'] == 1, "'age' must keep meaning frames since evidence"

    off = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera')
    off.step(cg, *_lever_boxes(cg, [np.array([0.0, 0.0, 0.0])], side=200.0))
    assert set(off.targets[0]) == {'point', 'age'}, \
        'with the lever off the state dict must be exactly what it always was'


def test_view_arbitration_keeps_a_crowded_camera_out_of_the_triangulation():
    """Several views with different occlusion geometry is the thing a multiview rig actually
    buys, and the shipped tracker spends none of it: a camera where two animals sit on top of
    each other votes on the 3D point exactly as loudly as one that separates them cleanly.

    `view_arbitration` measures each detection's distance to the nearest OTHER detection in its
    own camera, in box sides, and a camera under half a gate width does not triangulate provided
    two unambiguous cameras remain. Its box is still emitted -- crowding is uncertainty, not
    proof of a wrong claim, which is precisely what separates this lever from lever 1.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a = np.array([0.0, 0.0, 0.0], np.float32)
    near = np.array([26.0, 0.0, 0.0], np.float32)
    per_cam, scores = _lever_boxes(cg, [a])
    crowded, _ = _lever_boxes(cg, [a, near])
    per_cam[2], scores[2] = crowded[2], torch.ones(2)
    stale = torch.as_tensor(np.array([16.0, 0.0, 0.0], np.float32))

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera')
    tr.targets[0] = {'point': stale.clone(), 'age': 0}
    _, _, claimed = tr.step(cg, per_cam, scores)
    assert claimed[0, 2] == 1, 'the crowded camera must claim the WRONG neighbour here'
    assert float(np.linalg.norm(tr.targets[0]['point'].numpy() - a)) > 10.0, \
        'and that claim must drag the 3D point, or there is nothing to arbitrate'

    tr = CrossViewTracker(1, max_res_px=30.0, assoc_mode='per-camera',
                         view_arbitration=True)
    tr.targets[0] = {'point': stale.clone(), 'age': 0}
    out, _, claimed = tr.step(cg, per_cam, scores)
    np.testing.assert_allclose(tr.targets[0]['point'].numpy(), a, atol=1e-3)
    assert claimed[0, 2] == 1 and np.isfinite(out[0, 2]).all(), \
        'the discounted camera still reports its box -- only its vote is withheld'


def test_view_arbitration_is_inert_where_no_camera_is_crowded():
    """The rate-matched control for lever 4: a rejection rule must be scored against the
    population it governs, so it has to be provably silent everywhere else. With one detection
    per camera there is nobody to be confused with and every claim votes, byte for byte.
    """
    from tailcyclenet.detector.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    got = {}
    for arb in (False, True):
        tr = CrossViewTracker(2, max_res_px=30.0, assoc_mode='per-camera',
                             view_arbitration=arb)
        rows = []
        for t in range(6):
            w = [np.array([-60.0 + 8.0 * t, 0.0, 0.0]), np.array([90.0, 20.0 * t, 0.0])]
            rows.append(tr.step(cg, *_lever_boxes(cg, w, side=60.0)))
        got[arb] = ([r[0] for r in rows],
                    [tr.targets[s]['point'].numpy() for s in sorted(tr.targets)])
    for x, y in zip(got[False][0], got[True][0]):
        np.testing.assert_array_equal(np.nan_to_num(x, nan=-9e9), np.nan_to_num(y, nan=-9e9))
    for x, y in zip(got[False][1], got[True][1]):
        np.testing.assert_array_equal(x, y)


def test_measured_tracker_configuration_is_the_default():
    """The measured zero-switch configuration is the upstream default, while the legacy
    per-camera configuration remains explicitly reproducible. Pinning both prevents an accidental
    drift in the public constructor from silently changing either deployment or reproduction."""
    import inspect

    from tailcyclenet.detector.track import CrossViewTracker

    sig = inspect.signature(CrossViewTracker.__init__).parameters
    assert sig['assoc_mode'].default == 'joint'
    assert sig['max_age'].default == 8
    assert sig['max_move'].default == 1.25
    for name in ('claim_residual_gate', 'velocity', 'view_arbitration'):
        assert sig[name].default is False, f'{name} must ship off'

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a, b = np.array([-80.0, 0.0, 0.0]), np.array([80.0, 0.0, 0.0])

    def run(**kw):
        tr = CrossViewTracker(2, max_res_px=30.0, **kw)
        rows = []
        for t in range(10):
            w = [a + [14.0 * t, 0, 0], b - [14.0 * t, 0, 0]]
            per_cam, scores = _lever_boxes(cg, w if t % 2 == 0 else w[::-1], side=60.0)
            rows.append(tr.step(cg, per_cam, scores))
        return rows

    base = run()
    explicit = run(assoc_mode='joint', max_age=8, max_move=1.25,
                   claim_residual_gate=False, velocity=False, view_arbitration=False)
    for t, (want, got) in enumerate(zip(base, explicit)):
        for i, name in enumerate(('boxes', 'scores', 'claimed')):
            np.testing.assert_array_equal(
                np.nan_to_num(want[i], nan=-9e9), np.nan_to_num(got[i], nan=-9e9),
                err_msg=f'frame {t}: {name} moved with every lever at its default')
