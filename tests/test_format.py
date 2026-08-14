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
    """A session may reorder the root's keypoints, or carry only some of them.

    Both used to be silent relabels: the session scatters its rows through its OWN `names`
    (`_kpt_vocab`), while the id vector came from the FIRST session's list. Same length,
    different order, `nose` coordinates training the `left_ear` embedding row -- gotcha #4, and
    invisible in the loss curve. `Registry.ids_for` now resolves by name.
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
    fmt.write_session(path, mode='2d', units='px', label_source='tracked', names=KPTS_2D,
                      rig=rig, groups=groups, labels=labels)
    sess = fmt.Session.load(path)
    assert sess.labels('g_small').n_animals == 2
    assert sess.labels('g_big').n_animals == 5


def test_labels_key_is_required_and_closed(tmp_path):
    """§4 / decision 6: `labels` is required, and its vocabulary is two values.

    The point of a closed vocabulary is that a typo fails loudly. An open one would make
    `labels = "traked"` a third source that every consumer silently weights on its own.

    It is checked on BOTH seams -- `Session.load` for data already on disk, `write_session` for
    data being produced -- because the loader is what protects a training run and the writer is
    what stops a converter shipping 259 files that need backfilling again.
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
    """`Session.labels(gid)` stays callable.

    session.toml spells the key `labels`; `Session` exposes it as `label_source`. That mismatch
    is deliberate and this is what it buys -- a `labels` FIELD would shadow the method on every
    instance, and ~35 call sites would fail with "'str' object is not callable" at run time.
    """
    sess = fmt.Session.load(tiny_root / 'ratlike' / 'train' / 'sess_a')
    assert sess.label_source in fmt.LABEL_SOURCES
    assert sess.labels('g000').n_animals == 2


# ----------------------------------------------------------------------------------------------
# regions.pq -- §9b. The file's ABSENCE is a claim, so None and empty are different answers.
# ----------------------------------------------------------------------------------------------

def _rewrite_with_regions(path, regions):
    """Re-emit a session written by `_session_2d`, this time carrying `regions`."""
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    lab.regions = regions
    fmt.write_session(path, mode=sess.mode, units=sess.units, label_source=sess.label_source,
                      names=sess.names, rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=sess.flip_pairs, provenance=sess.provenance)
    return fmt.Session.load(path)


def test_regions_absent_means_exhaustively_labelled(tmp_path):
    """Every session written before regions.pq existed keeps reading as fully labelled."""
    _session_2d(tmp_path / 'ds' / 'train' / 'a')
    sess = fmt.Session.load(tmp_path / 'ds' / 'train' / 'a')
    assert not (sess.path / 'regions.pq').exists()
    assert sess.labels('g000').regions is None


def test_regions_roundtrip(tmp_path):
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    want = np.array([[0, 0, 1.0, 2.0, 30.0, 20.0],
                     [0, 0, 5.0, 5.0, 15.0, 15.0],      # two rects on one frame, overlapping
                     [2, 0, 8.0, 9.0, 40.0, 30.0]])
    got = _rewrite_with_regions(path, want).labels('g000').regions
    np.testing.assert_allclose(got, want)
    assert not fmt.validate_session(fmt.Session.load(path), check_images=False)


def test_regions_empty_is_not_the_same_as_absent(tmp_path):
    """The whole semantic: a certified-nothing group must not read as fully labelled.

    This is the rat-city-annotated `test/` split, where APT's GT mode records no ROIs at all --
    a MISSING regions.pq there would claim those frames are exhaustive, which they are not.
    """
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    sess = _rewrite_with_regions(path, np.zeros((0, 6)))
    assert (path / 'regions.pq').exists()
    regions = sess.labels('g000').regions
    assert regions is not None and regions.shape == (0, 6)


def test_rule_15_rejects_an_empty_rectangle(tmp_path):
    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    sess = _rewrite_with_regions(path, np.array([[0, 0, 30.0, 20.0, 30.0, 25.0]]))   # x1 == x0
    assert _rule(fmt.validate_session(sess, check_images=False), 15)


def test_rule_15_rejects_an_unknown_status(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'ds' / 'train' / 'a'
    _session_2d(path)
    _rewrite_with_regions(path, np.array([[0, 0, 1.0, 2.0, 30.0, 20.0]]))
    t = pq.read_table(path / 'regions.pq')
    i = t.column_names.index('status')
    pq.write_table(t.set_column(i, 'status', pa.array(['present']).dictionary_encode()),
                   path / 'regions.pq')
    errs = fmt.validate_session(fmt.Session.load(path), check_images=False)
    assert _rule(errs, 15) and 'present' in _rule(errs, 15)[0]
