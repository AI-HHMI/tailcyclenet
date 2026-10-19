"""A rotated 3D camera only zeroes visibility it had asserted (data B1)."""
import torch

from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tests.conftest import _session_3d


def _unlabelled_slot_targets(root, rot_prob):
    cfg = LoaderConfig(n_frames=4, prob_2d_only=0.0, aug_prob=0.0, aug_rotation_prob=rot_prob,
                       aug_rotation_deg=5.0, crop_jitter=0.0, prompt_dropout=0.0, min_crop_dim=8,
                       cams_to_sample=0, box_source='keypoints')
    ds = PoseDataset(root, 'train', cfg)
    vals = []
    for _ in range(20):
        it = ds[0]
        for t, f in enumerate(it[3].tolist()):
            if f == 1:
                vals.append(it[7][t, 2])
    return vals


def test_rotation_keeps_unassessed_visibility_nan(tmp_path):
    root = tmp_path / 'ds'
    _session_3d(root / 'train' / 's')
    plain = _unlabelled_slot_targets(root, 0.0)
    rotated = _unlabelled_slot_targets(root, 1.0)
    assert plain and rotated
    # keypoint 2 at frame 1 is UNLABELED in every camera: no assessment, rotated or not
    assert all(torch.isnan(v).all() for v in plain)
    assert all(torch.isnan(v).all() for v in rotated)
