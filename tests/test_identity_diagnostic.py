import numpy as np

from tailcyclenet.identity_diagnostic import switch_diagnostics


def test_switch_diagnostic_reports_duplicate_signature():
    true = np.full((2, 3, 2, 3), np.nan, np.float32)
    true[0, :, 0], true[0, :, 1] = (0, 0, 0), (0, 1, 0)
    true[1, :, 0], true[1, :, 1] = (100, 0, 0), (100, 1, 0)
    pred = np.full((2, 3, 2, 3), np.nan, np.float32)
    pred[0, 0], pred[1, 0] = true[0, 0], true[1, 0]
    pred[0, 1], pred[1, 1] = true[0, 1], true[0, 1]
    pred[0, 2], pred[1, 2] = ((1, 0, 0), (1, 1, 0)), ((0, 0, 0), (0, 1, 0))

    report = switch_diagnostics(pred, true, scale=10.0, near_scale=0.2, bin_frames=2)
    assert report['n_switches'] == 1
    item = report['switches'][0]
    assert item['frame'] == 2
    assert item['gt_row'] == 0
    assert item['old_pred_row'] == 0 and item['new_pred_row'] == 1
    assert item['nearest_gt_distance'] == 10.0
    assert item['predicted_pair_distance'] == 0.1
    assert report['near_coincident_count'] == 1
    assert report['switches_by_frame_bin'] == {'2-3': 1}
