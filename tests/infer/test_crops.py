"""Crop views: the hoisted decode gives the same pixels as cropping at decode."""
import numpy as np
import torch

from tests.helpers import SKEW_RIG3, _skew_rig


def test_the_hoisted_decode_gives_the_same_pixels_as_cropping_at_decode(rat):
    """One decode per (camera, frame): cropping the decoded frame equals cropping at decode, bit for bit."""
    from tailcyclenet.data.pixels import read_frames
    from tailcyclenet.infer.crops import crop_views

    _, sess, _, _ = rat
    group, cam = sess.groups['g000'], sess.cam_names[0]
    frames = [0, 1, 2, 1]                       # a repeat, which `read_frames` dedupes
    box = torch.tensor([3, 5, 40, 44], dtype=torch.int32)
    target = [32, 32]

    at_decode = np.asarray(read_frames(group, cam, frames, crop_coords=box, target_size=target))
    hoisted = crop_views(read_frames(group, cam, frames), box, target)
    np.testing.assert_array_equal(hoisted[0].numpy(), at_decode)


# --- Phase 8 levers (each a switch whose default is the shipped behaviour) ---

def test_refine_every_camera_keeps_drops_and_adds_per_camera():
    """TR1: a kept camera's box is the shipped refine's box; a pass-1 camera whose refined box
    misses its pass-1 box is dropped alone; a camera pass 1 never used is added when the pose
    reprojects into its frame, and not when it falls outside."""
    from tailcyclenet.data import crop as cropmod
    from tailcyclenet.infer.crops import refine_every_camera

    cg = _skew_rig(SKEW_RIG3)
    pts = torch.tensor([[[0.0, 0.0, 0.0], [20.0, 10.0, 5.0], [-15.0, 5.0, 0.0]]] * 2)
    _, shipped = cropmod.crop_to_points_3d(cg, pts, 16)
    far = torch.tensor([600, 400, 640, 480], dtype=torch.int32)
    use2, b2, _ = refine_every_camera(pts, [0, 1], [shipped[0], far], [0, 1, 2], cg, 16)
    assert use2 == [0, 2]
    assert torch.equal(b2[0], shipped[0]) and torch.equal(b2[1], shipped[2])
    use3, _, _ = refine_every_camera(pts + torch.tensor([5000.0, 0.0, 0.0]), [0], [shipped[0]],
                                     [0, 1, 2], cg, 16)
    assert 1 not in use3 and 2 not in use3


def test_camera_consensus_flags_the_disagreeing_camera_and_never_on_nan():
    """R1: a camera whose 2D head describes another body (residual) or calls the animal unseen
    (visibility) is an outlier; the others are not; a camera with nothing finite is left alone."""
    from tracktail.model.cube import project_points_torch
    from tailcyclenet.infer.crops import camera_consensus

    cg = _skew_rig(SKEW_RIG3)
    pts = torch.tensor([[[0.0, 0.0, 0.0], [20.0, 10.0, 5.0], [-15.0, 5.0, 0.0]]] * 2)
    p2 = project_points_torch(cg, pts).numpy().copy()             # (C,T,K,2) agreeing heads
    p2[1] += 200.0                                                 # camera 1 saw another animal
    v2 = np.full(p2.shape[:-1], 3.0)
    v2[2] = -2.0                                                   # camera 2 says "not here"
    boxes = [torch.tensor([0, 0, 100, 100])] * 3
    res, vis, bad = camera_consensus(pts, p2, v2, [0, 1, 2], boxes, cg, 0.5, 0.0)
    assert bad.tolist() == [False, True, True]
    assert res[0] < 1e-3 and res[1] > 2.0 and vis[2] == -2.0
    p2[0] = np.nan
    _, _, bad = camera_consensus(pts, p2, None, [0, 1, 2], boxes, cg, 0.5, 0.0)
    assert bad.tolist() == [False, True, False]


def test_refine_every_camera_recrops_an_outlier_whose_box_was_elsewhere():
    """R1's recovery: an outlier's pass-1 box is not evidence, so a refined box that misses it is
    kept when the pose is in frame (the TR1 overlap test would have dropped it)."""
    from tailcyclenet.data import crop as cropmod
    from tailcyclenet.infer.crops import refine_every_camera

    cg = _skew_rig(SKEW_RIG3)
    pts = torch.tensor([[[0.0, 0.0, 0.0], [20.0, 10.0, 5.0], [-15.0, 5.0, 0.0]]] * 2)
    _, shipped = cropmod.crop_to_points_3d(cg, pts, 16)
    far = torch.tensor([600, 400, 640, 480], dtype=torch.int32)
    use, b, _ = refine_every_camera(pts, [0, 1], [shipped[0], far], [0, 1], cg, 16)
    assert use == [0]
    use, b, _ = refine_every_camera(pts, [0, 1], [shipped[0], far], [0, 1], cg, 16, outliers={1})
    assert use == [0, 1] and torch.equal(b[1], shipped[1])


def test_adaptive_inflation_widens_until_a_neighbour_and_falls_back_alone():
    """R3: alone -> the shipped factor; a far neighbour -> the ceiling; a near one -> the largest
    factor that does not touch it; already touching -> 1.0."""
    from tailcyclenet.data import crop as cropmod
    from tailcyclenet.infer.crops import adaptive_inflation, overlaps
    box, size = torch.tensor([100, 100, 140, 140], dtype=torch.int32), (1000, 1000)
    nan = np.full(4, np.nan)
    assert adaptive_inflation(box, size, [], 2.0, 1.5) == 1.5
    assert adaptive_inflation(box, size, [nan], 2.0, 1.5) == 1.5
    assert adaptive_inflation(box, size, [np.array([800, 800, 840, 840])], 2.0, 1.5) == 2.0
    near = np.array([146, 100, 186, 140])                          # 6 px to the right
    f = adaptive_inflation(box, size, [near], 2.0, 1.5)
    assert 1.0 < f < 1.5
    assert not overlaps(cropmod.inflate_box(box, size, f), near)
    assert overlaps(cropmod.inflate_box(box, size, f + 0.05), near)
    assert adaptive_inflation(box, size, [np.array([130, 100, 170, 140])], 2.0, 1.5) == 1.0
