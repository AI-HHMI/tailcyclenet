"""The detector, and the property that makes it worth having.

The point of this detector is not that it finds animals -- it is that it reproduces THE CROP
RULE'S box. If it learned some other plausible box, every downstream pose number would shift.
"""
from __future__ import annotations

import numpy as np
import torch
from aniposelib.cameras import CameraGroup

from tailcyclenet import format as fmt
from tailcyclenet.data.crop import crop_box_for_points
from tailcyclenet.detector.dataset import BoxDataset
from tailcyclenet.detector.letterbox import letterbox, unletterbox_boxes
from tailcyclenet.detector.sampling import box_collate, split_batch
from tests.helpers import KPTS_2D, _rig, _session_2d, _write_frames


def _box_only_session(path, group_ids=('video_ix0', 'video_ix10')):
    from tailcyclenet import format as fmt

    T = 2
    rig = _rig([('cam0', 64, 48, False, False, 0)])
    groups = {gid: fmt.Group(gid, T) for gid in group_ids}
    labels = {}
    for gid in group_ids:
        lab = fmt.empty_labels(2, T, len(KPTS_2D), 1, mode3d=False,
                               animal_ids=['labeled', 'present'])
        lab.points2d = lab.vis2d = None
        lab.instance = np.full((2, T, 1), fmt.INST_NONE, np.int8)
        lab.boxes = np.full((2, T, 1, 4), np.nan, np.float32)
        lab.instance[0, :, 0] = fmt.INST_LABELED
        lab.boxes[0, :, 0] = [8, 7, 28, 29]
        lab.instance[1, :, 0] = fmt.INST_PRESENT
        lab.boxes[1, :, 0] = [35, 8, 55, 30]
        labels[gid] = lab
    fmt.write_session(path, mode='2d', units='px', label_source='annotated',
                      names=KPTS_2D, rig=rig, groups=groups, labels=labels)
    for gid in group_ids:
        _write_frames(path / 'groups' / gid, 'cam0', T, (64, 48))


# --- box sources --------------------------------------------------------------------------


def test_box_source_instances_uses_only_labeled_stored_boxes(tmp_path):
    """Stored labeled corners are targets, not the crop rule over keypoints."""
    from tailcyclenet import format as fmt

    root = tmp_path / 'ds'
    path = root / 'train' / 's'
    _session_2d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    lab.instance[1, 1, 0] = fmt.INST_LABELED  # one stored positive alongside keypoints-only a01
    fmt.write_session(path, mode='2d', units='px', label_source='annotated', names=sess.names,
                      rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=sess.flip_pairs)

    ds = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=4, box_source='instances')
    base = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8,
                      max_frames_per_group=4)
    assert len(ds) == 1
    sess, gid, frame, cam_ix = ds.index[0]
    boxes = ds[0]['boxes']
    plain_index = next(i for i, (_, _, ff, _) in enumerate(base.index) if ff == frame)
    plain = base[plain_index]['boxes']
    lab = sess.labels(gid)
    cam = sess.rig.tracktail()[cam_ix]
    img = np.zeros((int(cam['size'][1]), int(cam['size'][0]), 3), np.uint8)
    _, scale, pad = letterbox(img, ds.input_wh)
    want = crop_box_for_points(torch.as_tensor(lab.boxes[1, frame, cam_ix]).view(2, 2),
                               cam['size'], ds.min_crop_dim, pad=0)
    back = unletterbox_boxes(boxes[1][None], scale, pad)[0]
    torch.testing.assert_close(back, want.float(), atol=0.51, rtol=0)
    assert not torch.isfinite(boxes[0]).any(), 'no stored box means no target, despite keypoints'
    assert torch.isfinite(plain[0]).all(), 'the keypoint source would produce a crop box'


def test_box_source_instances_trains_present_boxes_without_negatives(tiny_root):
    """A present box is a positive, and it makes the frame's background unreliable."""
    ds = BoxDataset(tiny_root / 'ratlike', 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=0, box_source='instances')
    # every other frame has only labeled rows without stored boxes: no target, no negatives
    assert [f for _, _, f, _ in ds.index] == [1]
    item = ds[0]
    assert not torch.isfinite(item['boxes'][0]).any()
    assert torch.isfinite(item['boxes'][1]).all(), 'present boxes are positive targets'
    assert item['negative_supervision'] is False


def test_negative_supervision_follows_instance_statuses(tmp_path):
    """Negatives need labeled/absent evidence and no present row, with or without a box."""
    from tailcyclenet import format as fmt

    T = 5
    path = tmp_path / 'ds' / 'train' / 'boxes'
    rig = _rig([('cam0', 64, 48, False, False, 0)])
    lab = fmt.empty_labels(2, T, len(KPTS_2D), 1, mode3d=False, animal_ids=['a', 'b'])
    lab.points2d = lab.vis2d = None
    lab.instance = np.full((2, T, 1), fmt.INST_NONE, np.int8)
    lab.boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    lab.instance[0, 0, 0] = fmt.INST_LABELED           # labeled box: positives + negatives
    lab.boxes[0, 0, 0] = [8, 7, 28, 29]
    lab.instance[0, 1, 0] = fmt.INST_ABSENT            # absent-only: all-background example
    lab.instance[0, 2, 0] = fmt.INST_LABELED           # labeled box beside a boxless present
    lab.boxes[0, 2, 0] = [8, 7, 28, 29]
    lab.instance[1, 2, 0] = fmt.INST_PRESENT
    lab.instance[0, 3, 0] = fmt.INST_PRESENT           # boxless present only: nothing to train
    # frame 4 has no instance row at all: no claim, not a negative
    fmt.write_session(path, mode='2d', units='px', label_source='annotated',
                      names=KPTS_2D, rig=rig, groups={'g': fmt.Group('g', T)},
                      labels={'g': lab})
    _write_frames(path / 'groups' / 'g', 'cam0', T, (64, 48))

    ds = BoxDataset(path.parent.parent, 'train', input_wh=(128, 128), min_crop_dim=8,
                    max_frames_per_group=0)
    got = {f: ds[i]['negative_supervision'] for i, (_, _, f, _) in enumerate(ds.index)}
    assert got == {0: True, 1: True, 2: False}
    assert not torch.isfinite(ds[1]['boxes']).any(), 'absent rows are not positives'
    batch = box_collate([ds[0], ds[1], ds[2]])
    assert batch['negative_supervision'].tolist() == [True, True, False]


def test_box_only_sessions_fall_back_to_labeled_instance_boxes(tmp_path):
    """A session with no keypoint tables trains its stored boxes, and indexes beside a pose one."""
    root = tmp_path / 'ds'
    path = root / 'train' / 'boxes_only'
    _box_only_session(path)
    assert not (path / 'keypoints.pq').exists()
    ds = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8)
    assert len(ds) == 4
    assert {item[1] for item in ds.index} == {'video_ix0', 'video_ix10'}
    assert len(ds.box_only_fallback_sessions) == 1
    for i in range(len(ds)):
        boxes = ds.boxes_for(i)
        assert torch.isfinite(boxes[0]).all()
        assert torch.isfinite(boxes[1]).all(), 'present boxes are positive targets'

    batch = box_collate([ds[0], ds[1]])
    assert batch['negative_supervision'].tolist() == [False, False]
    assert split_batch(batch)[2] is batch['negative_supervision']

    _session_2d(root / 'train' / 'pose')
    mixed = BoxDataset(root, 'train', input_wh=(128, 128), min_crop_dim=8)
    assert {sess.session_id for sess, _, _, _ in mixed.index} == {'boxes_only', 'pose'}
    assert mixed.box_only_fallback_sessions == {path}

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
