"""The detector-recipe bump: FPs on GT-less groups count, mosaic pastes into empty space,
`[training].seed` seeds augmentation, one NMS centre distance (detector B1/B4/B5/B15)."""
import argparse

import numpy as np
import torch

from tailcyclenet.detector import boxes as boxes_mod
from tailcyclenet.detector.dataset import BoxDataset, _overlaps_any
from tailcyclenet.detector.evaluate import _summarise, overall
from tailcyclenet.infer.provenance import _center_dist
from tests.conftest import _session_3d


def test_overall_counts_false_positives_on_a_group_with_no_labelled_box():
    with_gt = _summarise({'n_gt': 10, 'hit50': 10, 'hit75': 10, 'iou': 9.0, 'fp': 0, 'mota': []})
    empty = _summarise({'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0.0, 'fp': 50, 'mota': []})
    assert overall({'a': with_gt, 'b': empty})['fp'] == 5.0
    assert overall({'a': with_gt})['fp'] == 0.0


def test_mosaic_paste_never_covers_the_items_own_boxes():
    class Stub:
        input_wh, box_target = (100, 100), 'crop'
        _mosaic_paste = BoxDataset._mosaic_paste

        def __len__(self):
            return 2

        def _load_letterbox(self, j):
            return np.full((100, 100, 3), 255, np.uint8), torch.tensor([[10., 10., 50., 50.]])

    own = torch.tensor([[0., 0., 55., 100.]])
    rng = np.random.default_rng(0)
    pasted = 0
    for _ in range(200):
        out, img = Stub()._mosaic_paste(0, own.clone(), np.zeros((100, 100, 3), np.uint8), rng)
        if len(out) > 1:
            pasted += 1
            assert not _overlaps_any(tuple(out[1].tolist()), own)
            assert not img[:, :55].any(), 'the paste covered the real animal'
    assert pasted > 100


def test_training_seed_fixes_the_augmentation_stream(tmp_path):
    root = tmp_path / 'ds'
    _session_3d(root / 'train' / 's')

    def first_views(seed):
        ds = BoxDataset(root, 'train', input_wh=(64, 64), augment=True, strong=True, seed=seed,
                        max_frames_per_group=0)
        return [ds[i % len(ds)]['x'] for i in range(4)]

    a, b, c = first_views(5), first_views(5), first_views(6)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_one_nms_centre_distance_and_its_provenance():
    assert boxes_mod.NMS_CENTER_DIST == 0.3
    assert not hasattr(boxes_mod, 'NMS_CENTER_DIST_SELECT')
    assert not hasattr(boxes_mod, 'NMS_CENTER_DIST_DEPLOY')
    assert _center_dist(argparse.Namespace()) == 0.3
    assert _center_dist(argparse.Namespace(det_nms_center_dist=0.25)) == 0.25
