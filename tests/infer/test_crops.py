"""Crop views: the hoisted decode gives the same pixels as cropping at decode."""
import numpy as np
import torch


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
