"""A val/test window that fails to build is skipped, never swapped for another (data B7)."""
from tailcyclenet.loader.config import LoaderConfig
from tailcyclenet.loader.dataset import PoseDataset
from tailcyclenet.loader.sampling import pose_collate
from tests.conftest import _session_3d

CFG = LoaderConfig(n_frames=4, prob_2d_only=0.0, aug_prob=0.0, crop_jitter=0.0,
                   prompt_dropout=0.0, min_crop_dim=8, cams_to_sample=0, box_source='keypoints')


def _failing_first(root, split, monkeypatch):
    ds = PoseDataset(root, split, CFG)
    real = ds._item
    calls = []

    def item(idx, rng, shape):
        calls.append(idx)
        return None if idx == 0 else real(idx, rng, shape)

    monkeypatch.setattr(ds, '_item', item)
    return ds, calls


def test_a_failed_val_window_is_skipped(tmp_path, monkeypatch):
    root = tmp_path / 'ds'
    _session_3d(root / 'val' / 's')
    ds, calls = _failing_first(root, 'val', monkeypatch)
    assert ds[0] is None and calls == [0]
    assert pose_collate([ds[0]]) is None


def test_a_failed_train_window_is_still_replaced(tmp_path, monkeypatch):
    root = tmp_path / 'ds'
    _session_3d(root / 'train' / 's')
    ds, calls = _failing_first(root, 'train', monkeypatch)
    if len(ds) < 2:
        return
    for _ in range(20):
        calls.clear()
        if ds[0] is not None and len(calls) > 1:
            return
    raise AssertionError('train never retried another window')
