"""Detector box scoring: `box_mota` and the fp_dup / fp_none split."""
import numpy as np
import pytest
import torch

# --- scoring --------------------------------------------------------------------------------


def test_summarise_and_overall_report_fp_dup_and_fp_none_as_rates():
    """D1: `box_mota`'s fp_dup/fp_none survive into the per-group summary as per-gt rates (NaN,
    not 0.0, where nothing was scored) and `overall` weights them by n_gt, skipping NaN groups."""
    from tailcyclenet.detector.evaluate import _summarise, overall

    s = {'n_gt': 4, 'hit50': 3, 'hit75': 2, 'iou': 3.0, 'fp': 1,
         'mota': [{'mota': 0.5, 'gt': 4, 'fp_ignored': 1, 'fp_dup': 1, 'fp_none': 2, 'misses': 3}]}
    r = _summarise(s)
    assert r['fp_ignored'] == 1
    assert (r['fp_dup'], r['fp_none'], r['miss']) == pytest.approx((1 / 4, 2 / 4, 3 / 4))
    r = _summarise({'n_gt': 0, 'hit50': 0, 'hit75': 0, 'iou': 0, 'fp': 0, 'mota': []})
    assert np.isnan(r['fp_dup']) and np.isnan(r['fp_none']) and np.isnan(r['miss'])

    row = {'r50': 1, 'r75': 1, 'iou': 1, 'fp': 0, 'mota': 1.0}
    o = overall({'a': {**row, 'n_gt': 2, 'fp_ignored': 0, 'fp_dup': 0.5, 'fp_none': 0.0,
                       'miss': 0.25},
                 'b': {**row, 'n_gt': 6, 'fp_ignored': 2, 'fp_dup': 0.0, 'fp_none': 1.0,
                       'miss': 0.0},
                 'c': {**row, 'n_gt': 2, 'fp_ignored': 0, 'fp_dup': float('nan'),
                       'fp_none': float('nan'), 'miss': float('nan')}})
    assert o['fp_ignored'] == 2
    assert o['fp_dup'] == pytest.approx((2 * 0.5) / 8)
    assert o['fp_none'] == pytest.approx((6 * 1.0) / 8)
    assert o['miss'] == pytest.approx((2 * 0.25) / 8)


def test_box_mota_splits_fp_into_dup_and_none_and_counts_misses():
    from tailcyclenet.detector.evaluate import box_mota

    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0],             # exact match -> TP
                         [1.0, 1.0, 11.0, 11.0],             # near the claimed GT -> dup
                         [1000.0, 1000.0, 1010.0, 1010.0]])  # far away -> none
    r = box_mota({0: (pred, gt, None, None)})
    assert r['fp'] == 2 and r['fp_dup'] == 1 and r['fp_none'] == 1 and r['misses'] == 0

    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1000.0, 1000.0, 1010.0, 1010.0]])
    r = box_mota({0: (pred[:1], gt, None, None)})
    assert r['misses'] == 1 and r['fp'] == 0 and r['fp_dup'] == 0 and r['fp_none'] == 0


def test_score_dataset_scores_unaugmented_and_restores_augmentation(tiny_root, monkeypatch):
    """`score_dataset` turns the dataset's augmentation OFF for every item it scores and puts
    the caller's setting (and the model's train mode) back afterwards: augmented scoring would
    compare warped predictions against unwarped ignore boxes."""
    from tailcyclenet.detector.dataset import BoxDataset
    from tailcyclenet.detector.evaluate import score_dataset
    from tailcyclenet.detector.yolox import YOLOXNano

    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(64, 64), min_crop_dim=8,
                    augment=True)
    seen = []
    real = BoxDataset.__getitem__

    def spy(self, i):
        seen.append(self.augment)
        return real(self, i)

    monkeypatch.setattr(BoxDataset, '__getitem__', spy)
    model = YOLOXNano(version='hybrid').train()
    with torch.no_grad():
        score_dataset(model, ds, 'cpu', batch_size=2, batches=1, num_workers=0)
    assert seen and not any(seen), f'scored items saw augment={seen}'
    assert ds.augment is True and model.training
