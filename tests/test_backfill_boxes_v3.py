"""The 3D multi-camera path of `scripts/backfill_boxes_v3.py`.

A 3D root stores no per-camera 2D -- 3dpop ships `points3d.pq` alone -- so its boxes are
PROJECTED per camera. Two failures there would be silent rather than loud: the camera axis
collapsing, so one view's box is written under every camera name, and the labelled/present status
being read positionally off an animal axis that `convert_v4` has already short-rowed. The second
one flips a real label into an eval ignore region, or asserts a label that has no keypoints.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tailcyclenet import format as fmt
from tailcyclenet.crop import crop_box_for_points

from .conftest import _session_3d

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def bf():
    spec = importlib.util.spec_from_file_location(
        'tcn_backfill_v3', REPO / 'scripts' / 'backfill_boxes_v3.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Args:
    nms_iou, nms_kpt_frac, check, dry_run = 0.7, 0.15, 50, False


@pytest.fixture
def backfilled(bf, tmp_path):
    """A 3-camera 3D session plus a v3 npz holding ONE MORE animal than the session does.

    The extra row is the point: it is 3dpop's five deleted pigeons in miniature, an animal v4
    NaN'd out of existence that v3 still has in full. It must come back as `present`.
    """
    dst = tmp_path / 'ds' / 'train' / 's'
    lab = _session_3d(dst, T=4)
    S, T, K = lab.points3d.shape[:3]

    src = tmp_path / 'v3' / 's'
    (src / 'g000').mkdir(parents=True)
    rng = np.random.default_rng(3)
    ghost = rng.uniform(-50, 50, (1, T, K, 3))
    pose = np.concatenate([lab.points3d.astype(np.float64), ghost])
    # v3 is the UNCLEANED source, so it has a point where the session's labels have a hole.
    pose[np.isnan(pose)] = rng.uniform(-50, 50, int(np.isnan(pose).sum()))
    np.savez(src / 'g000' / 'pose3d.npz', pose=pose, ids=np.array(['m1', 'ghost']))

    assert S == 1
    bf.convert_session(dst, src, _Args())
    return fmt.Session.load(dst).labels('g000'), dst


def test_one_row_per_animal_frame_camera(backfilled):
    lab, _ = backfilled
    assert lab.instance.shape == (2, 4, 3)              # the ghost widened S from 1 to 2
    assert (lab.instance != fmt.INST_NONE).all()
    assert np.isfinite(lab.boxes).all()


def test_status_follows_the_labels_by_animal_id_not_row_index(backfilled):
    lab, _ = backfilled
    g, m = lab.animal_ids.index('ghost'), lab.animal_ids.index('m1')
    # 'ghost' sorts BEFORE 'm1' in the animal vocab, so a positional read would tag row 0
    # `labeled` -- the exact off-by-one `abef250` fixed on rat-city.
    assert g < m
    assert (lab.instance[g] == fmt.INST_PRESENT).all()
    assert (lab.instance[m] == fmt.INST_LABELED).all()


def test_each_camera_gets_its_own_projection(backfilled):
    """Three calibrated cameras at different poses cannot agree on a box unless the axis collapsed."""
    lab, _ = backfilled
    b = lab.boxes[0, 0]                                 # (C,4)
    assert not np.allclose(b[0], b[1]) and not np.allclose(b[1], b[2])


def test_stored_corners_reproduce_the_crop_rule(bf, backfilled):
    """What every consumer assumes: re-entering at `pad=0` on the two stored corners gives the
    same int32 box the 20 px keypoint rule gives on the points they came from."""
    from posetail.posetail.cube import project_points_torch
    lab, dst = backfilled
    sess = fmt.Session.load(dst)
    m = lab.animal_ids.index('m1')      # frame 0 is fully labelled, so v3 and the labels agree
    for ci, cam in enumerate(sess.cam_names):
        wh = torch.tensor(sess.rig.size(cam), dtype=torch.int32)
        p2d = project_points_torch([sess.cgroup('g000')[ci]],
                                   torch.as_tensor(lab.points3d[m, 0]))[0]
        want = crop_box_for_points(p2d, wh)
        got = crop_box_for_points(torch.as_tensor(lab.boxes[m, 0, ci]).view(2, 2), wh, pad=0)
        assert torch.equal(want, got), f'{cam}: {want.tolist()} vs {got.tolist()}'


def test_the_2d_branch_still_reads_stored_pixels(bf, tmp_path):
    """The 2D path must not have grown a projection: it has stored pixels and one camera, and its
    npz carries no `ids`, so the animal names come from the v4 row-index scheme."""
    from .conftest import _session_2d
    dst = tmp_path / 'ds2' / 'train' / 's'
    lab = _session_2d(dst, T=4, S=2)                          # animal_ids a01, a02
    src = tmp_path / 'v3b' / 's'
    (src / 'g000').mkdir(parents=True)
    pose = np.nan_to_num(lab.points2d[..., 0, :].astype(np.float64), nan=12.0)
    np.savez(src / 'g000' / 'pose2d.npz', pose=pose)          # no `ids` -> a{i:02d}
    bf.convert_session(dst, src, _Args())

    out = fmt.Session.load(dst).labels('g000')
    assert out.instance.shape == (3, 4, 1)                    # a00 u {a01, a02}
    assert out.animal_ids == ['a00', 'a01', 'a02']
    ix = {a: i for i, a in enumerate(out.animal_ids)}
    assert (out.instance[ix['a00']] == fmt.INST_PRESENT).all()   # v3 row 0, not in the labels
    assert (out.instance[ix['a01']] == fmt.INST_LABELED).all()   # v3 row 1 AND labelled
    assert (out.instance[ix['a02']] == fmt.INST_NONE).all()      # labelled, but v3 has no row 2
