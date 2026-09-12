"""The detector's box-source and eligibility rules, on purpose-built sessions.

These do NOT use `tests/conftest.py`'s shared fixture: that one is load-bearing for the POSE
loader's per-view fallback (`tests/test_dataset.py::test_box_source_falls_back_per_view`), and a
detector rule change must not be smuggled in by editing it.

No pixels are written. Neither `BoxDataset.__init__` (which reads tables and builds an index) nor
`boxes_for` decodes an image, so a root with no `groups/` is enough to pin both rules.
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
    """No `instances.pq` at all: the keypoint rule indexes the labelled frames, as it always did."""
    vis, points, inst, boxes = _tables(4, kp_frames=(0, 2), box_frames=None)
    root = _root(tmp_path, vis=vis, points=points, instance=inst, boxes=boxes)
    ds, idx = _index(root, box_source='instances')
    assert {f for _, f in idx} == {0, 2}
    assert all(torch.isfinite(ds.boxes_for(i)).all() for i in range(len(ds)))


def test_frame_with_no_stored_box_is_not_an_item(tmp_path):
    """A table that exists and says nothing about a frame must not fall back to its keypoints."""
    vis, points, inst, boxes = _tables(4, kp_frames=(0, 1, 2, 3), box_frames=(1,))
    root = _root(tmp_path, vis=vis, points=points, instance=inst, boxes=boxes)
    ds, idx = _index(root, box_source='instances')
    assert {f for _, f in idx} == {1}, 'only the frame the table describes is trainable'
    # the same root under the keypoint source still indexes every labelled frame
    _, base = _index(root, box_source='keypoints')
    assert {f for _, f in base} == {0, 1, 2, 3}


def test_missing_only_keypoints_is_not_a_target(tmp_path):
    """A `missing` row is a visibility judgement with no coordinates: it cannot anchor a box."""
    vis, points, inst, boxes = _tables(2, kp_frames=(0,), box_frames=None)
    vis[0, 1, :, 0] = fmt.MISSING              # assessed, placed nowhere
    root = _root(tmp_path, vis=vis, points=points, instance=inst, boxes=boxes)
    _, idx = _index(root, box_source='keypoints')
    assert {f for _, f in idx} == {0}


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


def test_boxes_with_no_keypoints_are_unreachable_by_the_index(tmp_path):
    """KNOWN GAP, pinned so it cannot be mistaken for intended behaviour.

    `instances.pq` holds real extents for animals whose keypoints the source removed -- 7,891 such
    frames exist in `3dpop`. `_has_target` would accept one, because it reads the selected source,
    but the candidate frames still come from keypoint visibility alone, so the item is never
    offered. Deriving eligibility from the selected source is what closes this; it is deliberately
    NOT done here because it changes the training set behind every existing 3dpop detector number.
    """
    vis, points, inst, boxes = _tables(1, kp_frames=(), box_frames=(0,))
    root = _root(tmp_path, vis=vis, points=points, instance=inst, boxes=boxes)
    with np.testing.assert_raises(ValueError):
        BoxDataset(root, 'train', input_wh=(64, 64), min_crop_dim=8, box_source='instances')
