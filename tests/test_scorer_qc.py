"""The QC path's device contract: a window must be BATCHED and MOVED before the forward.

`score_root` took `--device` all the way to `load_scorer_run` and then fed the model CPU tensors
from the loader, so every non-CPU scoring run died in `conv3d` with "Input type (torch.FloatTensor)
and weight type (torch.cuda.FloatTensor) should be the same". Nothing caught it because the one
caller on record scored on CPU, where the missing transfer is a no-op.

The shapes matter as much as the device. `views` and `coords` gain a batch axis at the call site
while `kpt_ids` does not -- batching the ids early made them `[1, 1, K]` and tripped `score`'s own
`(B, K)` assertion. Both halves are asserted here, on CPU, so no GPU is needed to hold the line.
"""
import torch

from tailcyclenet.scorer.qc import _to_device


def _window():
    """One loader-shaped window: a per-camera view list, coords, a camera dict and keypoint ids."""
    views = [torch.zeros(4, 8, 8, 3, dtype=torch.uint8), torch.zeros(4, 6, 6, 3, dtype=torch.uint8)]
    coords = torch.zeros(4, 5, 3)
    cgroup = [{'mat': torch.eye(3), 'offset': torch.zeros(2), 'name': 'cam0', 'n_frames': 4},
              {'mat': torch.eye(3), 'offset': torch.zeros(2), 'name': 'cam1', 'n_frames': 4}]
    return views, coords, cgroup, torch.arange(5)


def test_views_and_coords_get_a_batch_axis_and_kpt_ids_do_not():
    """The batching split is the contract `score` asserts against."""
    views, coords, cgroup, kpt_ids = _to_device(*_window(), 'cpu')
    assert views[0].shape == (1, 4, 8, 8, 3)
    assert views[1].shape == (1, 4, 6, 6, 3)
    assert coords.shape == (1, 4, 5, 3)
    assert kpt_ids.shape == (5,)


def test_everything_lands_on_the_requested_device():
    """The camera tensors travel too -- they feed the decoder's geometry, not just the pixels."""
    views, coords, cgroup, kpt_ids = _to_device(*_window(), 'cpu')
    assert all(v.device.type == 'cpu' for v in views)
    assert coords.device.type == 'cpu'
    assert kpt_ids.device.type == 'cpu'
    assert all(t.device.type == 'cpu' for cam in cgroup for t in cam.values()
               if torch.is_tensor(t))


def test_non_tensor_camera_entries_survive():
    """A camera dict carries labels and counts alongside its tensors; those must pass through."""
    views, coords, cgroup, kpt_ids = _to_device(*_window(), 'cpu')
    assert [c['name'] for c in cgroup] == ['cam0', 'cam1']
    assert [c['n_frames'] for c in cgroup] == [4, 4]
    assert len(cgroup) == len(views)


def test_the_move_is_real_and_not_a_noop_that_happens_to_type_check():
    """Targeting 'meta' proves the transfer happens: nothing was already on it."""
    views, coords, _cgroup, _kpt_ids = _to_device(*_window(), 'meta')
    assert views[0].device.type == 'meta'
    assert coords.device.type == 'meta'
