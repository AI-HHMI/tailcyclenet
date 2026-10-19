"""`deployment_score`: the detector-through-association score."""
import numpy as np
import torch

from tailcyclenet.detector.yolox import YOLOXNano
from tests.helpers import _session_2d, _session_3d


def test_deployment_score_reads_zero_untrained_and_one_when_forced(tmp_path):
    """Seeded fresh model: the rare-positive objectness prior sits below `det_score`, so fills
    read exactly 0 (union sides NaN); forcing every objectness logit positive fills every slot --
    the zero is the prior firing, not a broken wire."""
    torch.manual_seed(0)
    from tailcyclenet.detector.diagnostics import gt_crop_sides
    from tailcyclenet.evaluation.deployment import deployment_score
    from tailcyclenet.format import Session

    _session_2d(tmp_path / 'r' / 'test' / 's0', T=4, S=2)
    sess = Session.load(tmp_path / 'r' / 'test' / 's0')
    sess.preload()
    gid = list(sess.groups)[0]
    model = YOLOXNano(version='hybrid')
    kw = dict(input_wh=(64, 64), top_k=8, max_animals=2, n_frames=2, overlap=0, min_box_frames=1)
    r = deployment_score(model, sess, gid, det_score=0.5, **kw)
    assert r['det_fill'] == 0.0 and r['slot_fill'] == 0.0 and r['window_miss'] == 1.0
    assert r['n_windows'] == 4 and r['n_gt'] == 8
    assert all(v != v for v in r['union_side_px'].values())
    assert r['gt_side_px'][0.5] > 0

    for m in model.head.obj_pred:
        torch.nn.init.constant_(m.bias, 10.0)
    r = deployment_score(model, sess, gid, det_score=0.05, **kw)
    assert r['det_fill'] == 1.0
    assert not any(v != v for v in r['union_side_px'].values())

    # 3D: `gt_crop_sides` projects points3d per camera (a 3D session has no points2d).
    _session_3d(tmp_path / 'r3' / 'test' / 's0', T=4)
    s3 = Session.load(tmp_path / 'r3' / 'test' / 's0')
    s3.preload()
    sides = gt_crop_sides(s3, list(s3.groups)[0], min_crop_dim=8)
    assert sides.size > 0 and np.all(sides >= 8)
