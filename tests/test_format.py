"""docs/annotation_format.md, checked.

The load path is the inverse of the write path, so a round-trip that survives the parquet
encoding is the strongest single statement this file can make: it covers the status enum, the
dictionary codes, the animal/camera/bodypart vocabularies and every array shape at once.
"""
import numpy as np
import pytest

from tailcyclenet import format as fmt

from .conftest import KPTS_2D, KPTS_3D, _session_2d, _session_3d


def test_roundtrip_2d(tiny_root):
    """Every dense array comes back byte-identical, including the three status codes."""
    sess = fmt.Session.load(tiny_root / 'ratlike' / 'train' / 'sess_a')
    assert sess.mode == '2d' and sess.units == 'px'
    assert sess.names == KPTS_2D
    assert sess.cam_names == ['cam0']
    assert not sess.rig.calibrated['cam0']     # a 2D camera may omit its intrinsics

    lab = sess.labels('g000')
    assert lab.animal_ids == ['a01', 'a02']
    assert lab.points2d.shape == (2, 4, 4, 1, 2)
    assert lab.vis2d.shape == (2, 4, 4, 1)
    assert lab.points3d is None and lab.vis3d is None

    assert lab.vis2d[0, 0, 1, 0] == fmt.MISSING
    assert lab.vis2d[1, 2, 3, 0] == fmt.UNLABELED
    assert np.isnan(lab.points2d[0, 0, 1, 0]).all()
    assert np.isfinite(lab.points2d[lab.vis2d == fmt.VISIBLE]).all()

    # the ignore region survives
    assert lab.instance[1, 1, 0] == fmt.INST_PRESENT
    assert lab.instance[0, 1, 0] == fmt.INST_LABELED
    np.testing.assert_allclose(lab.boxes[1, 1, 0], [10, 10, 30, 30])


def test_roundtrip_3d(tiny_root):
    """The 3D layer is first-class, and per-camera visibility needs no 2D position."""
    sess = fmt.Session.load(tiny_root / 'mouselike' / 'train' / 'sess_c')
    lab = sess.labels('g000')
    assert lab.points3d.shape == (1, 4, 3, 3)
    assert lab.vis3d[0, 1, 2] == fmt.MISSING
    assert np.isnan(lab.points3d[0, 1, 2]).all()
    assert np.isfinite(lab.points3d[lab.vis3d == fmt.VISIBLE]).all()

    # rule 10 exemption: visible in a camera, position lives in the 3D layer
    assert lab.vis2d.shape == (1, 4, 3, 3)
    assert (lab.vis2d[0, :, 0, 2] == fmt.MISSING).all()
    assert (lab.vis2d[0, :, 1, 2] == fmt.VISIBLE).all()
    assert np.isnan(lab.points2d).all()


def test_vis_is_an_int8_compare(tiny_root):
    """status -> 0/1 visibility must not require touching strings."""
    sess = fmt.Session.load(tiny_root / 'ratlike' / 'train' / 'sess_a')
    vis = sess.labels('g000').vis2d
    assert vis.dtype == np.int8
    assert ((vis == fmt.VISIBLE).sum()) == 2 * 4 * 4 - 2


def test_moving_camera(tiny_root):
    """extrinsics.pq gives (C,T,4,4); static cameras in the same session broadcast to constant."""
    sess = fmt.Session.load(tiny_root / 'mouselike' / 'train' / 'sess_moving')
    assert [sess.rig.moving[n] for n in sess.cam_names] == [True, False, False]
    ext = sess.labels('g000').ext
    assert ext.shape == (3, 4, 4, 4)
    np.testing.assert_allclose(ext[0, :, 0, 3], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(ext[1], ext[1, 0])         # static camera is constant over time


def test_pixels_and_validation(dataset_2d, dataset_3d):
    assert fmt.validate_dataset(dataset_2d) == []
    assert fmt.validate_dataset(dataset_3d) == []
    g = dataset_2d.sessions['train'][0].groups['g000']
    kind, path = g.pixels('cam0')
    assert kind == 'frames'
    assert [p.name for p in g.frame_paths('cam0')] == [f'{i:06d}.png' for i in range(4)]


def test_discovery_dataset_vs_collection(tiny_root):
    """The presence of a train/ directory is the whole rule."""
    one = fmt.load_datasets(tiny_root / 'ratlike')
    assert [d.name for d in one] == ['ratlike']
    assert set(one[0].sessions) == {'train', 'val'}

    many = fmt.load_datasets(tiny_root)
    assert [d.name for d in many] == ['mouselike', 'ratlike']


# ----------------------------------------------------------------------------------------------
# the registry
# ----------------------------------------------------------------------------------------------

def test_registry_single_dataset_does_not_prefix(dataset_2d):
    reg = fmt.Registry.build([dataset_2d])
    assert list(reg.names) == KPTS_2D
    np.testing.assert_array_equal(reg.ids_for('ratlike'), [0, 1, 2, 3])


def test_registry_prefixes_across_datasets(dataset_2d, dataset_3d):
    reg = fmt.Registry.build([dataset_2d, dataset_3d])
    assert reg.n_keypoints == len(KPTS_2D) + len(KPTS_3D)
    assert 'ratlike-nose' in reg.names and 'mouselike-nose' in reg.names
    # disjoint id blocks: a shared bare name must not collide across datasets
    assert set(reg.ids_for('ratlike')).isdisjoint(set(reg.ids_for('mouselike')))


def test_registry_is_append_only(dataset_2d, dataset_3d, tmp_path):
    """Old ids survive so the embedding rows behind them survive warm start."""
    first = fmt.Registry.build([dataset_2d, dataset_3d])
    p = tmp_path / 'keypoint_registry.toml'
    first.save(p)
    assert fmt.Registry.load(p) == first

    grown = fmt.Registry.build([dataset_2d, dataset_3d], base=first)
    np.testing.assert_array_equal(grown.ids_for('ratlike'), first.ids_for('ratlike'))
    np.testing.assert_array_equal(grown.ids_for('mouselike'), first.ids_for('mouselike'))
    assert grown.names[:first.n_keypoints] == first.names


# ----------------------------------------------------------------------------------------------
# the validator -- one test per rule that can actually fire
# ----------------------------------------------------------------------------------------------

def _rule(errs, n):
    """Matches both '[rule N]' and '[rule N WARNING]'."""
    return [e for e in errs if f'[rule {n}]' in e or f'[rule {n} ' in e]


def test_rule_3_cross_session_names_must_agree(tmp_path):
    _session_2d(tmp_path / 'ds' / 'train' / 'a')
    _session_2d(tmp_path / 'ds' / 'train' / 'b')
    cfg = tmp_path / 'ds' / 'train' / 'b' / 'session.toml'
    cfg.write_text(cfg.read_text().replace('"tail_base"', '"tailbase"'))
    errs = fmt.validate_dataset(fmt.load_dataset(tmp_path / 'ds'), check_images=False)
    assert _rule(errs, 3)


def test_rule_5_3d_needs_two_calibrated_cameras(tmp_path):
    _session_3d(tmp_path / 'ds' / 'train' / 'a')
    calib = tmp_path / 'ds' / 'train' / 'a' / 'calibration.toml'
    # strip camera 1 and 2 -> mode=3d with a single camera
    text = calib.read_text().split('[cam_1]')[0]
    calib.write_text(text)
    errs = fmt.validate_session(fmt.Session.load(tmp_path / 'ds' / 'train' / 'a'),
                                check_images=False)
    assert _rule(errs, 5)


def test_rule_7_frame_count_must_match_n_frames(tmp_path):
    _session_2d(tmp_path / 'ds' / 'train' / 'a')
    (tmp_path / 'ds' / 'train' / 'a' / 'groups' / 'g000' / 'cam0' / '000003.png').unlink()
    errs = fmt.validate_session(fmt.Session.load(tmp_path / 'ds' / 'train' / 'a'))
    assert _rule(errs, 7)


def test_rule_8_image_size_must_match_calibration(tmp_path):
    _session_2d(tmp_path / 'ds' / 'train' / 'a')
    calib = tmp_path / 'ds' / 'train' / 'a' / 'calibration.toml'
    calib.write_text(calib.read_text().replace('size = [ 64, 48,]',
                                               'size = [ 32, 48,]'))
    errs = fmt.validate_session(fmt.Session.load(tmp_path / 'ds' / 'train' / 'a'))
    assert _rule(errs, 8)


def test_rule_13_extrinsics_require_moving_true(tmp_path):
    _session_3d(tmp_path / 'ds' / 'train' / 'a', moving=True)
    calib = tmp_path / 'ds' / 'train' / 'a' / 'calibration.toml'
    calib.write_text(calib.read_text().replace('moving = true', 'moving = false'))
    errs = fmt.validate_session(fmt.Session.load(tmp_path / 'ds' / 'train' / 'a'),
                                check_images=False)
    assert _rule(errs, 13)


def test_rule_14_warns_when_a_session_spans_splits(tmp_path):
    """rat-city does this by construction, so it warns rather than failing."""
    _session_2d(tmp_path / 'ds' / 'train' / 'same')
    _session_2d(tmp_path / 'ds' / 'test' / 'same')
    errs = fmt.validate_dataset(fmt.load_dataset(tmp_path / 'ds'), check_images=False)
    assert _rule(errs, 14) and 'WARNING' in _rule(errs, 14)[0]


def test_unknown_bodypart_is_an_error(tmp_path):
    _session_2d(tmp_path / 'ds' / 'train' / 'a')
    cfg = tmp_path / 'ds' / 'train' / 'a' / 'session.toml'
    cfg.write_text(cfg.read_text().replace('"nose"', '"snout"'))
    sess = fmt.Session.load(tmp_path / 'ds' / 'train' / 'a')
    with pytest.raises(fmt.FormatError, match='unknown bodypart'):
        sess.labels('g000')


def test_animal_count_may_vary_between_groups(tmp_path):
    """A parquet dictionary is per FILE, not per group.

    branson-fly sessions hold 5..10 flies depending on the trial, so a session's `animal_id`
    dictionary names animals that any individual group has never seen. Reading one group must
    not trip over the others' animals.
    """
    from aniposelib.cameras import CameraGroup

    W = H = 32
    rig = fmt.Rig(CameraGroup([fmt.nominal_camera('cam0', (W, H))]),
                  offset={'cam0': (0.0, 0.0)}, moving={'cam0': False},
                  calibrated={'cam0': False})
    groups, labels = {}, {}
    for gid, n_animals in (('g_small', 2), ('g_big', 5)):
        lab = fmt.empty_labels(n_animals, 2, len(KPTS_2D), 1, mode3d=False)
        lab.vis2d[:] = fmt.VISIBLE
        lab.points2d[..., 0, :] = 10.0
        groups[gid] = fmt.Group(gid, 2)
        labels[gid] = lab

    path = tmp_path / 'ds' / 'train' / 's'
    fmt.write_session(path, mode='2d', units='px', names=KPTS_2D, rig=rig,
                      groups=groups, labels=labels)
    sess = fmt.Session.load(path)
    assert sess.labels('g_small').n_animals == 2
    assert sess.labels('g_big').n_animals == 5
