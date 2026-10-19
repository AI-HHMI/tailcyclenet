"""The detector's box-source and eligibility rules, on purpose-built pixel-free sessions.

Not conftest's shared fixture: that one pins the POSE loader's per-view fallback.
"""
from __future__ import annotations

import numpy as np
import torch
from aniposelib.cameras import CameraGroup

from tailcyclenet import format as fmt
from tailcyclenet.detector.data import BoxDataset

NAMES = ['a', 'b']
W, H = 64, 48
BOX = [10.0, 10.0, 30.0, 30.0]


def _root(tmp_path, *, vis, points, instance=None, boxes=None, name='root'):
    """A one-camera 2D root carrying exactly the tables given."""
    rig = fmt.Rig(cgroup=CameraGroup([fmt.nominal_camera('cam0', (W, H))]),
                  offset={'cam0': (0.0, 0.0)}, moving={'cam0': False}, calibrated={'cam0': False})
    S, T, K, C = vis.shape
    lab = fmt.empty_labels(S, T, K, C, mode3d=False, animal_ids=[f'a{i}' for i in range(S)])
    lab.vis2d = vis.astype(np.int8)
    lab.points2d = points.astype(np.float32)
    lab.instance = instance
    lab.boxes = boxes
    path = tmp_path / name / 'train' / 'sess'
    fmt.write_session(path, mode='2d', units='px', label_source='annotated', names=NAMES, rig=rig,
                      groups={'g000': fmt.Group('g000', T, fps=40.0)}, labels={'g000': lab},
                      provenance={'source': 'synthetic'})
    return tmp_path / name


def _tables(T, *, kp_frames=(0,), box_frames=()):
    """(vis, points, instance, boxes) with keypoints on `kp_frames` and boxes on `box_frames`."""
    S, K, C = 1, len(NAMES), 1
    vis = np.full((S, T, K, C), fmt.UNLABELED, np.int8)
    points = np.full((S, T, K, C, 2), np.nan, np.float32)
    for f in kp_frames:
        vis[0, f, :, 0] = fmt.VISIBLE
        points[0, f, :, 0] = [12.0, 12.0]
    inst = None if box_frames is None else np.full((S, T, C), fmt.INST_NONE, np.int8)
    boxes = None if box_frames is None else np.full((S, T, C, 4), np.nan, np.float32)
    if box_frames is not None:
        for f in box_frames:
            inst[0, f, 0] = fmt.INST_LABELED
            boxes[0, f, 0] = BOX
    return vis, points, inst, boxes


def _index(root, **kw):
    ds = BoxDataset(root, 'train', input_wh=(64, 64), min_crop_dim=8, **kw)
    return ds, {(g, f) for _, g, f, _ in ds.index}


def test_absent_table_falls_back_to_keypoints(tmp_path):
    """No `instances.pq`: the keypoint rule indexes the labelled frames, and a `missing`-only
    frame (a visibility judgement with no coordinates) cannot anchor a box."""
    vis, points, inst, boxes = _tables(4, kp_frames=(0, 2), box_frames=None)
    vis[0, 1, :, 0] = fmt.MISSING              # assessed, placed nowhere
    root = _root(tmp_path, vis=vis, points=points, instance=inst, boxes=boxes)
    ds, idx = _index(root, box_source='instances')
    assert {f for _, f in idx} == {0, 2}
    assert all(torch.isfinite(ds.boxes_for(i)).all() for i in range(len(ds)))


def test_frame_with_no_stored_box_is_not_an_item(tmp_path):
    """A table that exists indexes exactly the frames it describes: it never falls back to their
    keypoints, and a labeled box needs no keypoints."""
    vis, points, inst, boxes = _tables(4, kp_frames=(0, 1, 2), box_frames=(1, 3))
    root = _root(tmp_path, vis=vis, points=points, instance=inst, boxes=boxes)
    ds, idx = _index(root, box_source='instances')
    assert {f for _, f in idx} == {1, 3}
    assert all(torch.isfinite(ds.boxes_for(i)[0]).all() for i in range(len(ds)))
    # the same root under the keypoint source indexes every labelled frame
    _, base = _index(root, box_source='keypoints')
    assert {f for _, f in base} == {0, 1, 2}


def test_an_animal_the_table_omits_gets_no_box(tmp_path):
    """Two animals, one stored box: the other gets NO target, not a keypoint-derived one."""
    vis, points, inst, boxes = _tables(1, kp_frames=(0,), box_frames=(0,))
    vis = np.repeat(vis, 2, axis=0)
    points = np.repeat(points, 2, axis=0)
    inst = np.repeat(inst, 2, axis=0)
    boxes = np.repeat(boxes, 2, axis=0)
    inst[1, 0, 0] = fmt.INST_NONE              # a01 has keypoints but no instance row
    boxes[1, 0, 0] = np.nan
    root = _root(tmp_path, vis=vis, points=points, instance=inst, boxes=boxes)
    ds, _ = _index(root, box_source='instances')
    got = ds.boxes_for(0)
    assert torch.isfinite(got[0]).all(), 'the described animal keeps its stored box'
    assert not torch.isfinite(got[1]).any(), 'the omitted animal must have no target'


def test_multi_root_box_source_override_is_resolved_per_session(tmp_path):
    """`boxes_by_dataset` overrides the shared source for its root only."""
    vis, points, inst, boxes = _tables(1, kp_frames=(0,), box_frames=(0,))
    root_dir = tmp_path / 'multi'
    _root(root_dir, vis=vis, points=points, instance=inst, boxes=boxes, name='alpha')
    _root(root_dir, vis=vis, points=points, instance=inst, boxes=boxes, name='zeta')
    ds = BoxDataset(root_dir, 'train', input_wh=(64, 64), min_crop_dim=8,
                    box_source='keypoints', boxes_by_dataset={'zeta': 'instances'},
                    max_frames_per_group=0)
    by_root = {}
    for i, (sess, _, _, _) in enumerate(ds.index):
        by_root[ds.session_dataset[sess.path]] = ds.boxes_for(i)[0]
    assert set(by_root) == {'alpha', 'zeta'}
    torch.testing.assert_close(by_root['zeta'], torch.tensor([10.0, 18.0, 30.0, 38.0]))
    assert not torch.allclose(by_root['alpha'], by_root['zeta'])

