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
