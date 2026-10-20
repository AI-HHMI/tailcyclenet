"""The deployment prior: box prompts, the carried prior, and oracle corruptions."""
import numpy as np
import torch

from tailcyclenet.infer.window import InferConfig
from tests.helpers import _infer_cfg


def test_deploy_box_prompt_lands_inside_the_crop_2d():
    """`deploy_box_prompt` maps an animal's source points into the crop frame and boxes them,
    2D single-camera -- the path this function has always covered, unchanged in result."""
    from tailcyclenet.infer.prior import deploy_box_prompt
    source = np.array([[[[110., 110.], [130., 150.]]]])            # (S=1,T=1,K=2,2) source px
    boxes = [torch.tensor([100, 100, 200, 200], dtype=torch.int32)]
    scales = [2.0]
    cam = {'size': torch.tensor([200, 200], dtype=torch.int32)}
    out = deploy_box_prompt('2d', source, None, np.array([0]), 0, [0], boxes, scales, [cam],
                             'cpu')
    assert out.shape == (1, 1, 1, 4)
    x0, y0, x1, y1 = out[0, 0, 0].tolist()
    # source (110,110)->(20,20) and (130,150)->(60,100) in crop px; the box must bound them
    assert x0 <= 20 and y0 <= 20 and x1 >= 60 and y1 >= 100


def test_deploy_box_prompt_3d_labels_matches_the_training_computation(mv):
    """3D `labels` box prompt is byte-identical to `compute_box_prompt` (deploy/train parity)."""
    from tailcyclenet.infer.prior import deploy_box_prompt
    from tailcyclenet.pose import box_prompt as bpmod

    _, sess, _, _ = mv
    frames = np.arange(4)
    cgroup = sess.cgroup('g000', frames)
    lab = sess.labels('g000')
    world = lab.points3d                                # (S,T,K,3)

    got = deploy_box_prompt('3d', world, None, frames, 0, list(range(len(cgroup))), None, None,
                             cgroup, 'cpu')
    want = bpmod.compute_box_prompt(torch.as_tensor(world[0]), cgroup, '3d')[None]
    torch.testing.assert_close(got, want.to(got.dtype))


def test_box_prompt_detector_per_camera_degradation():
    """A camera with no finite detector box yields a NaN column (the no-box token), not an exception."""
    from tailcyclenet.infer.prior import deploy_box_prompt

    T = 3
    cgroup = [dict(size=torch.tensor([64, 48], dtype=torch.int32)) for _ in range(2)]
    boxes = [torch.tensor([0, 0, 64, 48], dtype=torch.int32) for _ in range(2)]
    scales = [1.0, 1.0]
    # (S=1,T,C=2,4) source px: camera 0 has a real box every frame, camera 1 is all-NaN.
    boxes_stc = np.full((1, T, 2, 4), np.nan, np.float32)
    boxes_stc[0, :, 0] = [10., 10., 30., 30.]
    frames = np.arange(T)

    got = deploy_box_prompt('3d', None, boxes_stc, frames, 0, [0, 1], boxes, scales, cgroup,
                             'cpu')
    assert got.shape == (1, T, 2, 4)
    assert torch.isfinite(got[:, :, 0]).all()           # camera 0: real box every frame
    assert torch.isnan(got[:, :, 1]).all()              # camera 1: nothing finite -> NaN column


def test_box_prompt_column_order_matches_use():
    """Column i of the box prompt is built from `use[i]`'s own box/scale/cgroup, not a neighbour's."""
    from tailcyclenet.infer.prior import deploy_box_prompt
    from tailcyclenet.pose import box_prompt as bpmod

    T = 2
    cgroup = [dict(size=torch.tensor([64, 48], dtype=torch.int32)),
              dict(size=torch.tensor([32, 96], dtype=torch.int32))]
    boxes = [torch.tensor([0, 0, 64, 48], dtype=torch.int32),
             torch.tensor([100, 100, 132, 196], dtype=torch.int32)]
    scales = [1.0, 2.0]
    # use = [1, 0]: camera index 1 (source cam 1) is requested FIRST, camera 0 second.
    boxes_stc = np.full((1, T, 2, 4), np.nan, np.float32)
    boxes_stc[0, :, 0] = [10., 10., 30., 30.]           # source cam 0's own detector box
    boxes_stc[0, :, 1] = [110., 104., 126., 140.]       # source cam 1's own detector box
    frames = np.arange(T)

    got = deploy_box_prompt('3d', None, boxes_stc, frames, 0, [1, 0],
                             [boxes[1], boxes[0]], [scales[1], scales[0]],
                             [cgroup[1], cgroup[0]], 'cpu')
    assert got.shape == (1, T, 2, 4)

    from tailcyclenet.data import crop as cropmod

    def expected(src_ci, origin, scale, size):
        db = torch.as_tensor(boxes_stc[0, 0, src_ci], dtype=torch.float32)
        corners = cropmod.box_corners(db)
        cf = (corners - torch.as_tensor(origin, dtype=torch.float32)) * float(scale)
        return cropmod.crop_box_for_points(cf, torch.as_tensor(size), bpmod.BOX_PROMPT_MIN_DIM,
                                           bpmod.BOX_PROMPT_PAD)

    # column 0 (use[0] = source cam 1): its OWN box/scale/cgroup, not source cam 0's
    want0 = expected(1, [100, 100], 2.0, [32, 96])
    torch.testing.assert_close(got[0, 0, 0], want0.to(got.dtype))
    # column 1 (use[1] = source cam 0): its OWN box/scale/cgroup
    want1 = expected(0, [0, 0], 1.0, [64, 48])
    torch.testing.assert_close(got[0, 0, 1], want1.to(got.dtype))
    # and swapped indices must NOT coincide -- the failure mode this guards against
    assert not torch.allclose(got[0, 0, 0], want1.to(got.dtype))


def test_give_box_this_window_compensates_for_a_stale_carry_not_just_a_missing_one():
    """The first-only gate gives the box back when `build_prior` DECLINED a carry (stale/mismatched),
    not only when there was none."""
    from tailcyclenet.infer.prior import give_box_this_window

    cfg = _infer_cfg(anchor='carry', box_prompt='detector', box_prompt_first_only=True)
    fake_carry = (torch.zeros(2, 2), 0)

    assert give_box_this_window(cfg, None, None), \
        'no carry at all (the true first window) must still give the box'
    assert not give_box_this_window(cfg, fake_carry, torch.zeros(2, 2)), \
        'a carry that WAS used as this window\'s prior must still withhold the box'
    assert give_box_this_window(cfg, fake_carry, None), \
        'a carry that existed but was declined (stale/shape-mismatched) must give the box back'

    off = _infer_cfg(anchor='carry', box_prompt='detector', box_prompt_first_only=False)
    assert give_box_this_window(off, fake_carry, None), \
        'box_prompt_first_only=False must give the box every window regardless of carry/prior'
    assert not give_box_this_window(_infer_cfg(anchor='carry', box_prompt='none'), None, None), \
        'box_prompt="none" must never give the box'


def test_carried_prior_is_bounds_masked_and_dated():
    """The carried prior is NaN outside the crop, dated to its frame in the window, and retired (not
    clamped) when it predates the window or is staler than the overlap."""
    from tailcyclenet.infer.prior import build_prior

    K, size = 4, torch.tensor([100, 100], dtype=torch.int32)
    cgroup = [{'size': size}]
    cfg = InferConfig(anchor='carry')
    frames = np.arange(20, 28)
    boxes = [(0, 0, 100, 100)]

    pose = torch.tensor([[10.0, 10.0], [90.0, 90.0], [-5.0, 10.0], [10.0, 400.0]])
    prior, qt = build_prior(cfg, (pose, 23), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                             cgroup)

    assert torch.isfinite(prior[0, :2]).all(), 'in-crop keypoints must survive'
    assert torch.isnan(prior[0, 2:]).all(), 'out-of-crop keypoints must become NaN'
    assert qt.shape == (1, K)
    assert (qt == 3).all(), f'prompt frame 23 is index 3 of a window starting at 20, got {qt}'

    assert build_prior(cfg, (pose, 17), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                        cgroup) == (None, None)
    _, ordinary = build_prior(cfg, (pose, 20), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                               cgroup)
    assert (ordinary == 0).all()
    _, late = build_prior(cfg, (pose, 999), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                           cgroup)
    assert (late == len(frames) - 1).all()

    assert build_prior(cfg, (pose, 2), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                        cgroup) == (None, None)


def test_oracle_corrupt_near_and_swap():
    """`near` picks the nearest ELIGIBLE row (no-op if none); `swap:<n>` permutes n keypoint pairs
    of the chosen row without changing which row/frame it came from."""
    from tailcyclenet.infer.oracle import corrupt_prior

    K, size = 2, torch.tensor([100, 100], dtype=torch.int32)
    cgroup = [{'size': size}]
    frames = np.arange(0, 4)
    boxes = [(0, 0, 100, 100)]
    scales = [1.0]

    src = np.zeros((3, 4, K, 2), dtype=np.float32)
    src[0, 0] = [[10.0, 10.0], [20.0, 20.0]]
    src[1, 0] = [[500.0, 500.0], [600.0, 600.0]]
    src[2, 0] = [[15.0, 15.0], [25.0, 25.0]]

    cfg = InferConfig(anchor='labels', oracle_corrupt='near')
    p, qt = corrupt_prior(cfg, src, 0, 3, frames, boxes, '2d', cgroup, scales)
    assert torch.allclose(p, torch.as_tensor(src[2, 0])), \
        'near must pick the ELIGIBLE row, never the out-of-bounds one'
    assert qt == 0

    # Nothing eligible -> a no-op, the target's own row comes back unchanged.
    src2 = src.copy()
    src2[2, 0] = [[500.0, 500.0], [600.0, 600.0]]     # make row 2 ineligible too
    p2, _ = corrupt_prior(cfg, src2, 0, 3, frames, boxes, '2d', cgroup, scales)
    assert torch.allclose(p2, torch.as_tensor(src2[0, 0])), 'no eligible row must be a no-op'

    src = np.zeros((1, 4, 4, 2), dtype=np.float32)
    src[0, 0] = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    p, qt = corrupt_prior(InferConfig(anchor='labels', oracle_corrupt='swap:1'), src, 0, 1,
                           frames, boxes, '2d', cgroup)
    assert qt == 0
    base = torch.as_tensor(src[0, 0])
    assert sorted(p.tolist()) == sorted(base.tolist()), 'the keypoint SET must be unchanged'
    assert int((p != base).any(-1).sum()) == 2, 'swap:1 is exactly one pair'


# --- `--box-corrupt` (a diagnostic for the refine robustness levers) ---

def _two_animal_boxes():
    """(S=2,T=3,C=2,4): animal 0 near the origin, animal 1 to its right, both in both cameras."""
    bb = np.zeros((2, 3, 2, 4), np.float32)
    bb[0, :, :] = [10, 10, 30, 30]
    bb[1, :, :] = [100, 10, 120, 30]
    return bb


def test_box_corrupt_kinds_do_what_they_say():
    """drop NaNs the camera; shift moves the centre by `sides` box sides; shrink scales the side
    about the centre; swap takes the other row's boxes; p = 0 is the identity."""
    from tailcyclenet.infer.oracle import corrupt_boxes, parse_box_corrupt
    bb = _two_animal_boxes()
    assert np.array_equal(corrupt_boxes(bb, parse_box_corrupt('drop:0,swap:0'), 'g', 0), bb)
    out = corrupt_boxes(bb, parse_box_corrupt('drop:1'), 'g', 0)
    assert np.isnan(out).all()
    out = corrupt_boxes(bb, parse_box_corrupt('shift:1:0.5'), 'g', 0)
    move = (out[..., :2] + out[..., 2:]) / 2 - (bb[..., :2] + bb[..., 2:]) / 2
    np.testing.assert_allclose(np.linalg.norm(move, axis=-1), 10.0, rtol=1e-5)   # 0.5 x 20 px
    np.testing.assert_allclose(out[..., 2:] - out[..., :2], bb[..., 2:] - bb[..., :2], atol=1e-4)
    out = corrupt_boxes(bb, parse_box_corrupt('shrink:1:0.6'), 'g', 0)
    np.testing.assert_allclose(out[..., 2] - out[..., 0], 12.0, rtol=1e-6)
    np.testing.assert_allclose(out[..., 0] + out[..., 2], bb[..., 0] + bb[..., 2])
    out = corrupt_boxes(bb, parse_box_corrupt('swap:1'), 'g', 0)
    assert np.array_equal(out[0], bb[1]) and np.array_equal(out[1], bb[0])
    one = corrupt_boxes(bb[:1], parse_box_corrupt('swap:1'), 'g', 0)
    assert np.array_equal(one, bb[:1]), 'swap with no other row is a no-op'


def test_box_corrupt_is_deterministic_and_per_animal_window_camera():
    """The same (group, window, animal, camera) draws the same corruption on every call, a
    different window draws independently, and the whole window's frames move together."""
    from tailcyclenet.infer.oracle import corrupt_boxes, parse_box_corrupt
    bb = np.tile(_two_animal_boxes(), (8, 1, 1, 1))[:16]          # 16 rows, plenty of draws
    spec = parse_box_corrupt('drop:0.5')
    a, b = corrupt_boxes(bb, spec, 'g', 0), corrupt_boxes(bb, spec, 'g', 0)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    dropped = np.isnan(a[..., 0])                                  # (S,T,C)
    assert (dropped == dropped[:, :1]).all(), 'one decision per (animal, window, camera)'
    assert 0 < dropped[:, 0].mean() < 1
    other = np.isnan(corrupt_boxes(bb, spec, 'g', 12)[..., 0])
    assert not np.array_equal(other, dropped), 'another window draws again'


def test_box_corrupt_spec_is_parsed_strictly():
    """A malformed spec is refused by name rather than read as something else."""
    import pytest
    from tailcyclenet.infer.oracle import parse_box_corrupt
    assert parse_box_corrupt('drop:0.25,shift:0.25:0.5') == (('drop', 0.25, None),
                                                            ('shift', 0.25, 0.5))
    for bad in ('blur:0.1', 'shift:0.2', 'drop:1.5', 'shrink:0.2:1.5', 'swap:0.1:2'):
        with pytest.raises(ValueError):
            parse_box_corrupt(bad)
