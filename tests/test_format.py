"""docs/annotation_format.md, checked.

The load path is the inverse of the write path, so a round-trip that survives the parquet
encoding is the strongest single statement this file can make: it covers the status enum, the
dictionary codes, the animal/camera/bodypart vocabularies and every array shape at once.
"""
import numpy as np
import pytest

from tailcyclenet import format as fmt

from .conftest import KPTS_2D, KPTS_3D, _session_2d, _session_3d


def _project_3d_labels(sess, lab):
    """Populate per-camera positions from the known 3D fixture."""
    S, T, K = lab.vis3d.shape
    C = len(sess.rig)
    lab.points2d = np.full((S, T, K, C, 2), np.nan, np.float32)
    for c, cam in enumerate(sess.rig.cameras):
        xy = cam.project(lab.points3d.reshape(-1, 3)).detach().cpu().numpy()
        xy = xy.reshape(S, T, K, 2)
        positioned = np.isin(lab.vis2d[..., c], fmt.POSITIONED)
        lab.points2d[..., c, :] = np.where(positioned[..., None], xy, np.nan)


def test_roundtrip_2d(tiny_root):
    """Every dense array comes back byte-identical, including the three status codes."""
    sess = fmt.Session.load(tiny_root / 'ratlike' / 'train' / 'sess_a')
    assert sess.mode == '2d' and sess.units == 'px'
    assert sess.names == KPTS_2D
    assert sess.cam_names == ['cam0']
    assert not sess.rig.calibrated['cam0']     # a 2D camera may omit its intrinsics
    assert sess.cgroup('g000')[0]['calibrated'] is False

    lab = sess.labels('g000')
    assert lab.animal_ids == ['a01', 'a02']
    assert lab.points2d.shape == (2, 4, 4, 1, 2)
    assert lab.vis2d.shape == (2, 4, 4, 1)
    assert lab.points3d is None and lab.vis3d is None

    assert lab.vis2d[0, 0, 1, 0] == fmt.MISSING
    assert lab.vis2d[1, 2, 3, 0] == fmt.UNLABELED
    assert np.isnan(lab.points2d[0, 0, 1, 0]).all()
    assert np.isfinite(lab.points2d[lab.vis2d == fmt.VISIBLE]).all()

    # the instance statuses and the one stored box survive
    assert (lab.instance[:, :, 0] == fmt.INST_PRESENT).all()
    np.testing.assert_allclose(lab.boxes[1, 1, 0], [10, 10, 30, 30])
    assert np.isnan(lab.boxes[0]).all()


def test_prediction_session_is_not_triangulated_as_annotation(tmp_path):
    path = tmp_path / 's'
    _session_3d(path)
    (path / 'points3d.pq').unlink()
    sess = fmt.Session.load(path)
    sess.prediction_session = True
    assert sess.labels('g000').points3d is None


def test_keypoints_only_moving_rig_refuses_static_triangulation(tmp_path):
    path = tmp_path / 's'
    _session_3d(path, moving=True)
    (path / 'points3d.pq').unlink()
    sess = fmt.Session.load(path)
    with pytest.raises(fmt.FormatError, match='moving cameras'):
        sess.labels('g000')


def test_triangulate_group_applies_residual_gate(tmp_path):
    path = tmp_path / 's'
    _session_3d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    _project_3d_labels(sess, lab)
    # Perturb one camera so the least-squares triangulation has nonzero reprojection residual.
    lab.points2d[0, 0, 0, 0, 0] += 20
    visible, gated, _missing = fmt.triangulate_group(sess.rig, lab, gate=0.01)
    assert gated >= 1
    assert lab.vis3d[0, 0, 0] == fmt.UNLABELED


def test_triangulate_group_single_view_is_not_a_3d_label(tmp_path):
    path = tmp_path / 's'
    _session_3d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    _project_3d_labels(sess, lab)
    lab.points2d[0, 0, 0, 1:] = np.nan
    lab.vis2d[0, 0, 0, 1:] = fmt.MISSING
    fmt.triangulate_group(sess.rig, lab, gate=30)
    assert lab.vis3d[0, 0, 0] == fmt.UNLABELED
    assert np.isnan(lab.points3d[0, 0, 0]).all()


def test_triangulate_group_all_assessed_missing_stays_missing(tmp_path):
    path = tmp_path / 's'
    _session_3d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    _project_3d_labels(sess, lab)
    lab.points2d[0, 0, 0] = np.nan
    lab.vis2d[0, 0, 0] = fmt.MISSING
    fmt.triangulate_group(sess.rig, lab, gate=30)
    assert lab.vis3d[0, 0, 0] == fmt.MISSING
    assert np.isnan(lab.points3d[0, 0, 0]).all()


@pytest.mark.parametrize(
    ('distortions', 'expected'),
    [([], np.zeros(5)), ([0.1, -0.2], [0.1, -0.2, 0.0, 0.0, 0.0])],
)
def test_short_distortions_load_padded(distortions, expected):
    """Short calibration vectors are zero-padded for projection consumers."""
    doc = {
        'cam0': {
            'name': 'cam0',
            'size': [640, 480],
            'matrix': [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            'distortions': distortions,
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, 0.0, 1.0],
        },
    }
    rig = fmt.rig_from_doc(doc, '<memory>')
    np.testing.assert_array_equal(
        rig.cgroup.cameras[0].get_distortions().detach().cpu().numpy(), expected,
    )
    assert doc['cam0']['distortions'] == distortions


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


def test_has_visibility_assessment(tiny_root, tracked_no_assessment_root):
    """The session-level gate `dataset.py` reads: `annotated` with a real `missing` row -> True;
    `tracked` with 100% `visible` and zero `missing` -> False; `tracked` WITH real per-camera
    `missing` rows -> True.
    """
    ann = fmt.Session.load(tiny_root / 'ratlike' / 'train' / 'sess_a')
    assert ann.label_source == 'annotated'
    assert ann.has_visibility_assessment is True

    tracked_dense = fmt.Session.load(tracked_no_assessment_root / 'train' / 's')
    assert tracked_dense.label_source == 'tracked'
    assert tracked_dense.has_visibility_assessment is False

    tracked_assessed = fmt.Session.load(tiny_root / 'mouselike' / 'train' / 'sess_c')
    assert tracked_assessed.label_source == 'tracked'
    assert tracked_assessed.has_visibility_assessment is True


def test_preload_caches_visibility_before_dropping_tables(tiny_root, tracked_no_assessment_root):
    """Preloading must not make the first selection reread parquet visibility tables."""
    cases = [
        (tiny_root / 'ratlike' / 'train' / 'sess_a', True),
        (tiny_root / 'mouselike' / 'train' / 'sess_c', True),
        (tracked_no_assessment_root / 'train' / 's', False),
    ]
    for path, expected in cases:
        sess = fmt.Session.load(path)
        sess.preload()
        assert '_tables' not in sess.__dict__
        assert sess.has_visibility_assessment is expected
        assert '_tables' not in sess.__dict__


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
    np.testing.assert_array_equal(reg.ids_for_dataset('ratlike'), [0, 1, 2, 3])


def test_registry_prefixes_across_datasets(dataset_2d, dataset_3d):
    reg = fmt.Registry.build([dataset_2d, dataset_3d])
    assert reg.n_keypoints == len(KPTS_2D) + len(KPTS_3D)
    assert 'ratlike-nose' in reg.names and 'mouselike-nose' in reg.names
    # disjoint id blocks: a shared bare name must not collide across datasets
    assert set(reg.ids_for_dataset('ratlike')).isdisjoint(set(reg.ids_for_dataset('mouselike')))
    # ... and a session still asks in its own BARE names, so the prefix has to come back off
    assert reg.local_names('ratlike') == KPTS_2D
    np.testing.assert_array_equal(reg.ids_for('ratlike', KPTS_2D[::-1]),
                                  reg.ids_for_dataset('ratlike')[::-1])


def test_registry_is_append_only(dataset_2d, dataset_3d, tmp_path):
    """Old ids survive so the embedding rows behind them survive warm start."""
    first = fmt.Registry.build([dataset_2d, dataset_3d])
    p = tmp_path / 'keypoint_registry.toml'
    first.save(p)
    assert fmt.Registry.load(p) == first

    grown = fmt.Registry.build([dataset_2d, dataset_3d], base=first)
    for name in ('ratlike', 'mouselike'):
        np.testing.assert_array_equal(grown.ids_for_dataset(name), first.ids_for_dataset(name))
    assert grown.names[:first.n_keypoints] == first.names


def test_a_base_that_grew_past_one_dataset_keeps_its_own_dataset(dataset_2d, dataset_3d):
    """A registry may only have ONE dataset when it is first written, and grow later.

    That is the shape every allen scorer is in: the pose checkpoint contributed
    `allen-mouse-combined` and the scorer run added `allen-mouse-combined-tracked`, both recorded
    with the BARE keypoint names because neither build had more than one dataset. The next build
    taking that registry as its base therefore has `prefix = True`, so a target root the base
    already names came back under `allen-mouse-combined-tracked-nose` -- 47 NEW identities -- and
    `build` then raised `keypoint ids changed against the base registry`. The scorer QC path
    (`score_session.py`) could not score the very root the scorer was trained on.
    """
    base = fmt.Registry.build([dataset_2d])
    grown = fmt.Registry.build([dataset_3d], base=base)
    assert len(grown.datasets) == 2, 'the fixture must reproduce a base with two datasets'

    again = fmt.Registry.build([dataset_3d], base=grown)      # raised before the fix
    assert again.names == grown.names
    np.testing.assert_array_equal(again.ids_for_dataset('mouselike'),
                                  grown.ids_for_dataset('mouselike'))


def _rewrite_names(path, names):
    """Restate a session's keypoint axis. Legal: the parquet tables are keyed by NAME."""
    import tomllib

    import toml
    cfg = path / 'session.toml'
    with open(cfg, 'rb') as f:
        doc = tomllib.load(f)
    doc['names'] = list(names)
    cfg.write_text(toml.dumps(doc))


def _drop_keypoint(path, name):
    """A session that never labelled one keypoint: off the axis AND out of the tables."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    _rewrite_names(path, [n for n in KPTS_2D if n != name])
    t = pq.read_table(path / 'keypoints.pq')
    keep = pc.not_equal(t.column('bodypart').cast(pa.string()), name)
    pq.write_table(t.filter(keep), path / 'keypoints.pq', compression='zstd')


def test_ids_follow_each_sessions_own_keypoint_axis(tmp_path):
    """A session may reorder the root's keypoints, or carry only some of them -- both used to be
    silent relabels (same length, different order, `nose` coordinates training the `left_ear`
    embedding row, invisible in the loss curve). `Registry.ids_for` now resolves by name.
    """
    root = tmp_path / 'ds'
    lab_a = _session_2d(root / 'train' / 'a')
    lab_b = _session_2d(root / 'train' / 'b')
    _session_2d(root / 'train' / 'c')
    _rewrite_names(root / 'train' / 'b', KPTS_2D[::-1])
    _drop_keypoint(root / 'train' / 'c', 'left_ear')

    ds = fmt.load_dataset(root)
    a, b, c = (ds.sessions['train'][i] for i in range(3))
    assert ds.names == KPTS_2D                        # the union, in load order
    assert b.names == KPTS_2D[::-1] and c.n_keypoints == len(KPTS_2D) - 1

    reg = fmt.Registry.build([ds])
    for s in (a, b, c):
        ids = reg.ids_for('ds', s.names)
        assert [reg.names[i] for i in ids] == s.names, s.path

    # the id permutation and the DATA permutation agree: same name -> same coordinates
    for k, name in enumerate(b.names):
        np.testing.assert_allclose(b.labels('g000').points2d[..., k, 0, :],
                                   lab_b.points2d[..., KPTS_2D.index(name), 0, :])
    np.testing.assert_allclose(lab_a.points2d, a.labels('g000').points2d)

    # reordering and subsetting are reported, not fatal
    warns = _rule(fmt.validate_dataset(ds, check_images=False), 3)
    assert len(warns) == 2 and all('WARNING' in w for w in warns)

    # a name the root does not have is still an error, and it says which
    with pytest.raises(fmt.FormatError, match='snoot'):
        reg.ids_for('ds', ['snoot'] + KPTS_2D[1:])


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


def _mark_prediction(path, complete=None):
    cfg = path / 'session.toml'
    text = cfg.read_text()
    fields = 'prediction_session = true\n'
    if complete is not None:
        fields += f'complete = {str(complete).lower()}\n'
    cfg.write_text(text.replace('[provenance]', fields + '\n[provenance]'))


def _prediction_points2d(path, rows=1):
    """Write a minimal prediction-only point table for format-layer tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = {
        'group_id': ['g000'] * rows, 'frame': [0] * rows,
        'animal_id': [f'pred{i}' for i in range(rows)], 'camera': ['cam0'] * rows,
        'bodypart': ['tail_base'] * rows, 'x': [1.0] * rows, 'y': [2.0] * rows,
        'visibility_logit': [0.0] * rows, 'visibility_probability': [0.5] * rows,
        'confidence_logit': [0.0] * rows, 'confidence_probability': [0.5] * rows,
        'slot': list(range(rows)), 'window': [0] * rows, 'window_start': [0] * rows,
        'window_stop': [2] * rows,
    }
    pq.write_table(pa.table(data), path / 'points2d.pq')


def test_prediction_points2d_validates_but_is_not_training_labels(tmp_path):
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    (path / 'keypoints.pq').unlink()
    _mark_prediction(path, complete=False)
    _prediction_points2d(path)

    sess = fmt.Session.load(path)
    assert sess.prediction_session is True and sess.complete is False
    assert sess._tables['points2d'] is not None
    assert not fmt.validate_session(sess, check_images=False)
    # Prediction rows cannot add IDs or annotations to the dense training-label view.
    lab = sess.labels('g000')
    assert 'pred0' not in lab.animal_ids and lab.points2d is None


def test_points2d_requires_marker_and_unique_prediction_keys(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    _prediction_points2d(path)
    assert any('prediction-only' in e for e in fmt.validate_session(
        fmt.Session.load(path), check_images=False))

    _mark_prediction(path)
    t = pq.read_table(path / 'points2d.pq')
    row = t.slice(0, 1)
    pq.write_table(pa.concat_tables([row, row]), path / 'points2d.pq')
    errs = fmt.validate_session(fmt.Session.load(path), check_images=False)
    assert any('points2d.pq has duplicate keys' in e for e in errs)


def test_prediction_points3d_missing_status_may_keep_coordinates(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'ds' / 'train' / 'a'
    _session_3d(path)
    table_path = path / 'points3d.pq'
    t = pq.read_table(table_path)
    status_i = t.column_names.index('status')
    t = t.set_column(status_i, 'status', pa.array(['missing'] * len(t)).dictionary_encode())
    pq.write_table(t, table_path, compression='zstd')

    # This remains invalid annotation data; the explicit marker enables the prediction exception.
    sess = fmt.Session.load(path)
    assert any('missing/unlabeled row carrying coordinates' in e for e in
               fmt.validate_session(sess, check_images=False))
    _mark_prediction(path)
    assert not any('missing/unlabeled row carrying coordinates' in e for e in
                   fmt.validate_session(fmt.Session.load(path), check_images=False))


def test_rule_6_no_label_table_is_required(tmp_path):
    """Boxes alone are a detection session; no label table at all is an unlabelled clip."""
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    (path / 'keypoints.pq').unlink()
    assert (path / 'instances.pq').exists()
    sess = fmt.Session.load(path)
    assert not fmt.validate_session(sess, check_images=False)
    lab = sess.labels('g000')
    assert lab.points2d is None or not np.isfinite(lab.points2d).any()
    assert (lab.instance == fmt.INST_PRESENT).any()

    (path / 'instances.pq').unlink()
    sess = fmt.Session.load(path)
    assert not fmt.validate_session(sess, check_images=False)
    sess.labels('g000')


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
    """A parquet dictionary is per FILE, not per group: a session's `animal_id` dictionary names
    animals that any individual group has never seen. Reading one group must not trip over the
    others' animals.
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
    fmt.write_session(path, mode='2d', units='px', label_source='tracked', names=KPTS_2D,
                      rig=rig, groups=groups, labels=labels)
    sess = fmt.Session.load(path)
    assert sess.labels('g_small').n_animals == 2
    assert sess.labels('g_big').n_animals == 5


def test_labels_key_is_required_and_closed(tmp_path):
    """§4 / decision 6: `labels` is required, and its vocabulary is two values so a typo fails
    loudly. Checked on BOTH seams -- `Session.load` for data on disk, `write_session` for data
    being produced.
    """
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    cfg = path / 'session.toml'
    good = cfg.read_text()
    assert 'labels = "annotated"' in good, 'the fixture should declare its label source'

    cfg.write_text(good.replace('labels = "annotated"\n', ''))
    with pytest.raises(fmt.FormatError, match="missing 'labels'"):
        fmt.Session.load(path)

    cfg.write_text(good.replace('"annotated"', '"traked"'))
    with pytest.raises(fmt.FormatError, match='labels must be one of'):
        fmt.Session.load(path)

    cfg.write_text(good)
    assert fmt.Session.load(path).label_source == 'annotated'

    with pytest.raises(fmt.FormatError, match='label_source must be one of'):
        _session_2d(tmp_path / 'ds' / 'train' / 'b', label_source='human')


def test_label_source_does_not_shadow_the_labels_method(tiny_root):
    """`Session.labels(gid)` stays callable: session.toml spells the key `labels` while `Session`
    exposes it as `label_source`, so a `labels` FIELD would shadow the method on every instance.
    """
    sess = fmt.Session.load(tiny_root / 'ratlike' / 'train' / 'sess_a')
    assert sess.label_source in fmt.LABEL_SOURCES
    assert sess.labels('g000').n_animals == 2


# ----------------------------------------------------------------------------------------------
# rule 11: a `labeled` instance row carries a box. `regions.pq` is no longer part of the format.
# ----------------------------------------------------------------------------------------------

def _rewrite(path, edit):
    """Re-emit a session written by `_session_2d` after `edit(lab)` mutates its labels."""
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    edit(lab)
    fmt.write_session(path, mode=sess.mode, units=sess.units, label_source=sess.label_source,
                      names=sess.names, rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=(sess.flip_pairs if sess.flip_pairs_declared else None),
                      provenance=sess.provenance)
    return fmt.Session.load(path)


def test_write_session_refuses_a_labeled_row_without_a_box(tmp_path):
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)

    def boxless(lab):
        lab.instance[0, 0, 0] = fmt.INST_LABELED        # the fixture stores no box for a01

    with pytest.raises(fmt.FormatError, match='rule 11'):
        _rewrite(path, boxless)

    def empty_box(lab):
        lab.instance[0, 0, 0] = fmt.INST_LABELED
        lab.boxes[0, 0, 0] = [10.0, 10.0, 10.0, 20.0]  # x1 == x0: empty, equivalent to no box

    with pytest.raises(fmt.FormatError, match='rule 11'):
        _rewrite(path, empty_box)

    def boxed(lab):
        lab.instance[0, 0, 0] = fmt.INST_LABELED
        lab.boxes[0, 0, 0] = [10.0, 10.0, 20.0, 20.0]

    sess = _rewrite(path, boxed)
    assert sess.labels('g000').instance[0, 0, 0] == fmt.INST_LABELED
    assert not _rule(fmt.validate_session(sess, check_images=False), 11)


def test_rule_11_flags_a_labeled_row_without_a_box_on_disk(tmp_path):
    """A table written by some other tool is caught by validation, not only by the writer."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    t = pq.read_table(path / 'instances.pq')
    i = t.column_names.index('status')
    status = pa.array(['labeled'] * len(t)).dictionary_encode()
    pq.write_table(t.set_column(i, 'status', status), path / 'instances.pq')
    errs = fmt.validate_session(fmt.Session.load(path), check_images=False)
    assert _rule(errs, 11) and 'box' in _rule(errs, 11)[0]


def test_a_stale_regions_file_is_a_validation_error(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    pq.write_table(pa.table({'group_id': ['g000']}), path / 'regions.pq')
    errs = fmt.validate_session(fmt.Session.load(path), check_images=False)
    assert any('regions.pq is no longer part of the format' in e for e in errs)


def test_table_writer_chunks_match_one_shot(tmp_path):
    """A table written in chunks must read back exactly as one written at once.

    This is what lets inference stream a prediction to disk: the rows arrive a block at a time and
    no array is ever proportional to the clip. It is only usable if the file is indistinguishable
    from what a converter would have written.
    """
    import numpy as np
    import pyarrow.parquet as pq

    from tailcyclenet.format import DICT_COLS, TableWriter, _codes, write_table

    n = 500
    rng = np.random.default_rng(0)
    rows = {'group_id': np.array([f'g{i % 3:03d}' for i in range(n)], dtype=object),
            'frame': np.arange(n, dtype=np.int32),
            'animal_id': np.array(['a', 'b'] * (n // 2), dtype=object),
            'status': np.array(['visible', 'missing'] * (n // 2), dtype=object),
            'x': rng.normal(size=n)}
    rows['x'][::7] = np.nan                       # nulls must survive chunking

    write_table(tmp_path / 'one.pq', rows, DICT_COLS)
    # Chunks small enough to force several row groups, and uneven so no boundary is special.
    with TableWriter(tmp_path / 'many.pq', DICT_COLS, chunk_rows=64) as w:
        for lo, hi in ((0, 33), (33, 200), (200, 201), (201, n)):
            w.write({k: v[lo:hi] for k, v in rows.items()})

    a, b = (pq.read_table(tmp_path / f) for f in ('one.pq', 'many.pq'))
    assert a.column_names == b.column_names
    assert a.to_pydict() == b.to_pydict(), 'chunked rows must equal one-shot rows'
    assert pq.ParquetFile(tmp_path / 'many.pq').num_row_groups > 1, 'the chunks must be row groups'
    # The dictionary columns must still READ BACK as dictionaries, or `_codes` breaks.
    for col in ('group_id', 'animal_id', 'status'):
        codes, vals = _codes(b, col)
        assert len(codes) == n and vals


def test_table_writer_skips_empty_chunks(tmp_path):
    """A block that produced no rows must not write an empty row group, or emit a schema of nulls."""
    import numpy as np
    import pyarrow.parquet as pq

    from tailcyclenet.format import DICT_COLS, TableWriter

    with TableWriter(tmp_path / 't.pq', DICT_COLS) as w:
        w.write({'group_id': np.array([], dtype=object), 'frame': np.array([], np.int32)})
        w.write({'group_id': np.array(['g0'], dtype=object), 'frame': np.array([3], np.int32)})
    t = pq.read_table(tmp_path / 't.pq')
    assert t.num_rows == 1 and t.column('frame').to_pylist() == [3]


# A SESSION WITH NO DIRECTORY -- the two format additions `--videos` rests on.

def _video_session(tmp_path, T=6, cams=('cam0', 'cam1'), where='vids'):
    """A `VideoSession` whose `path` does not exist and whose pixels are loose mp4s."""
    from .conftest import _rig, _write_video

    W, H = 64, 48
    d = tmp_path / where
    src = {c: _write_video(d / f'{c}.mp4', i, T, (W, H)) for i, c in enumerate(cams)}
    rig = _rig([(c, W, H, True, False, i + 1) for i, c in enumerate(cams)])
    K = len(KPTS_3D)
    groups = {'g0': fmt.video_group('g0', T, src, fps=20.0)}
    empty = {'g0': fmt.empty_labels(0, T, K, len(cams), mode3d=True)}
    sess = fmt.VideoSession(path=tmp_path / 'no_such_dir' / 's', mode='3d', units='mm',
                            label_source='tracked', names=KPTS_3D, rig=rig, groups=groups,
                            empty=empty)
    for g in groups.values():
        g.session = sess
    return sess


def test_a_video_group_decodes_with_no_session_directory(tmp_path):
    """THE INVARIANT THE WHOLE --videos DESIGN RESTS ON.

    `group.source` is the ONLY filesystem entry point in the decode path, and it is a cache over
    `_src`. `format.video_group` pre-fills it, so `pixels()` is never called, `dir` is never
    dereferenced and `session.path` -- which does not exist here -- is never read. If a refactor
    makes `read_frames` reach for a directory again, this fails immediately instead of the videos
    path failing in a user's hands.

    Asserted on VALUES, not shapes: the fixture's frames are solid colours keyed on
    (camera, frame), so a decode that is off by one frame or serving the wrong camera fails.
    """
    from tailcyclenet.dataset import read_frames

    from .conftest import _video_colour

    sess = _video_session(tmp_path)
    assert not sess.path.exists(), 'the point is that there is no directory'
    g = sess.groups['g0']
    for ci, cam in enumerate(sess.cam_names):
        imgs = read_frames(g, cam, np.asarray([0, 2, 5]))
        assert len(imgs) == 3
        for im, t in zip(imgs, (0, 2, 5)):
            assert im.shape == (48, 64, 3)
            got = im.reshape(-1, 3).mean(0)
            want = np.asarray(_video_colour(ci, t), float)
            assert np.abs(got - want).max() < 12, \
                f'{cam} frame {t}: decoded {got}, expected {want} (wrong frame or wrong camera)'
    # And it really did go through the pre-filled cache rather than the directory.
    assert g.source('cam0')[0] == 'video'
    with pytest.raises(fmt.FormatError):
        g.pixels('cam0')                      # there IS no groups/ directory to find


def test_a_video_session_never_reads_a_table(tmp_path):
    """`_table` is overridden to return None as a GUARANTEE, not as an accident of a missing file.

    `path` is a LABEL, not a location. Without the override a session whose `path` happened to
    collide with a real directory would silently adopt its parquet -- so this puts a real
    `keypoints.pq` exactly there and asserts the empty arrays survive.
    """
    sess = _video_session(tmp_path)
    sess.path.mkdir(parents=True)
    real = _session_2d(tmp_path / 'real')
    (sess.path / 'keypoints.pq').write_bytes((tmp_path / 'real' / 'keypoints.pq').read_bytes())
    assert real is not None

    assert sess._table('keypoints') is None
    assert all(v is None for v in sess._tables.values())
    lab = sess.labels('g0')
    assert lab.animal_ids == [] and lab.points3d.shape == (0, 6, len(KPTS_3D), 3)
    # `preload()` pops `_tables`, which is a cached_property -- anything that reads it afterwards
    # recomputes it, and a recomputed one would go to disk. An override cannot be popped.
    sess.preload()
    assert sess.labels('g0') is lab or sess.labels('g0').points3d.shape == lab.points3d.shape
    assert sess._table('points3d') is None
