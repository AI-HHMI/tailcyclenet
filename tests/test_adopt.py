"""`--videos`: raw footage plus an anipose calibration, adopted as a session IN MEMORY.

The acceptance test for the construction is here, not at runtime: `validate_session` cannot
certify a `VideoSession`, so the in-memory session is compared against the written one once, in CI.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from tailcyclenet.infer import adopt
from tailcyclenet import format as fmt

from .conftest import KPTS_3D, _rig, _write_video


def _calib(tmp_path, names, wh=(64, 48), calibrated=True, moving=False):
    W, H = wh
    rig = _rig([(n, W, H, calibrated, moving, i + 1) for i, n in enumerate(names)])
    fmt.dump_calibration(tmp_path / 'calib.toml', rig)
    return tmp_path / 'calib.toml'


def _touch(d: Path, *names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b'')
    return d


def _no_probe(monkeypatch):
    """Assert nothing decodes: every refusal fires above the checkpoint load and above any decode."""
    calls = []

    def boom(path):
        calls.append(path)
        raise AssertionError(f'a pure refusal decoded {path}')

    monkeypatch.setattr(adopt, '_probe', boom)
    return calls


# the naming rule


@pytest.mark.parametrize('stem,rx,cam,gid', [
    ('cam0_trial3', r'cam([0-9]+)_', '0', 'trial3'),      # anipose: the CAMERA IS THE CAPTURE GROUP
    ('cam0_trial3', r'cam[0-9]+', 'cam0', '_trial3'),     # no group: the whole match, sep kept
    ('Cam2005325', r'Cam[0-9]+', 'Cam2005325', ''),       # johnson's raw rig dump
    ('anything_at_all', None, '', 'anything_at_all'),     # no regex: the stem is the group
])
def test_the_anipose_naming_rule(stem, rx, cam, gid):
    """Both the camera name and the group id come from one regex; either can be silently wrong."""
    assert adopt.parse_name(stem, rx) == (cam, gid)


def test_one_camera_needs_no_regex(tmp_path, monkeypatch):
    """A one-camera rig needs no regex (2D single view); any larger rig requires one."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['cam0'], calibrated=False)
    _touch(tmp_path / 'rec', 'whatever.mp4', 'other.mp4')
    p = adopt.plan([tmp_path / 'rec'], cal, None)
    assert p.mode == '2d'
    assert sorted(p.videos) == ['other', 'whatever']
    assert all(list(v) == ['cam0'] for v in p.videos.values())
    with pytest.raises(SystemExit, match='--cam-regex is required'):
        adopt.plan([tmp_path / 'rec'], _calib(tmp_path, ['0', '1']), None)


def test_uncalibrated_single_camera_uses_nominal_camera_from_video_size(tmp_path):
    """A 2D raw recording has enough information for a nominal cam0; its geometry is built from
    the decoder's own (width, height), not an invented calibration matrix."""
    wh = (80, 48)
    video = _write_video(tmp_path / 'rec' / 'take.mp4', 0, 4, wh)
    p = adopt.plan([video], None, None)
    sess = adopt.build(p, names=KPTS_3D, units='px', verbose=False)
    assert sess.mode == '2d' and sess.units == 'px'
    assert sess.cam_names == ['cam0']
    assert sess.rig.size('cam0') == wh
    assert sess.rig.offset['cam0'] == (0.0, 0.0)
    assert sess.rig.moving['cam0'] is False
    assert sess.rig.calibrated['cam0'] is False
    np.testing.assert_allclose(
        sess.rig.by_name('cam0').get_camera_matrix().detach().cpu().numpy(),
        fmt.nominal_camera('cam0', wh).get_camera_matrix().detach().cpu().numpy())


def test_the_empty_remainder_rule_is_about_disagreement(tmp_path, monkeypatch):
    """Many groups share one session; an all-empty group id is one group, but `Cam1.mp4` beside
    `Cam1_t3.mp4` is an ambiguity and refused."""
    _no_probe(monkeypatch)
    p = adopt.plan([_touch(tmp_path / 'rec', *[f'cam{i}_{g}.mp4' for g in 'abc' for i in (0, 1)])],
                   _calib(tmp_path, ['0', '1']), r'cam([0-9]+)_')
    assert sorted(p.videos) == ['a', 'b', 'c']
    assert p.session_id == 'rec' and p.group_id == ''

    cal = _calib(tmp_path, ['Cam1', 'Cam2'])
    _touch(tmp_path / 'raw', 'Cam1.mp4', 'Cam2.mp4')
    p = adopt.plan([tmp_path / 'raw'], cal, r'Cam[0-9]+', group_id='rec42')
    assert list(p.videos) == ['rec42'] and sorted(p.videos['rec42']) == ['Cam1', 'Cam2']
    assert p.group_id == 'rec42'
    # and the default is the session id, not ''
    assert list(adopt.plan([tmp_path / 'raw'], cal, r'Cam[0-9]+').videos) == ['raw']

    _touch(tmp_path / 'mix', 'Cam1.mp4', 'Cam2.mp4', 'Cam1_t3.mp4', 'Cam2_t3.mp4')
    with pytest.raises(SystemExit, match='some filenames with a group id and some with nothing'):
        adopt.plan([tmp_path / 'mix'], cal, r'Cam[0-9]+')


# refusals 1-6: pure over the filenames


def test_refusal_1_no_video_matches_the_regex(tmp_path, monkeypatch):
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'])
    _touch(tmp_path / 'rec', 'a.mp4', 'b.mp4')
    with pytest.raises(SystemExit) as e:
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert 'matched none' in str(e.value) and 're.search' in str(e.value)


def test_refusal_2_no_parsed_camera_is_in_the_calibration(tmp_path, monkeypatch, capsys):
    """The most common error: `cam([0-9]+)_` yields '0' while the calibration names it 'cam0'. None
    matched refuses; some matched skips the rest by name."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['cam0', 'cam1'])
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4')
    with pytest.raises(SystemExit) as e:
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert 'CAPTURE GROUP' in str(e.value)
    # ...and the group-less superset the message points at actually works. The group id is `_t`,
    # not `t`: a group-less pattern does not consume the separator (see the naming-rule table).
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam[0-9]+')
    assert sorted(p.videos['_t']) == ['cam0', 'cam1']

    cal = _calib(tmp_path, ['Cam1', 'Cam2'])
    _touch(tmp_path / 'raw', 'Cam1.mp4', 'Cam2.mp4', 'Cam9.mp4')
    p = adopt.plan([tmp_path / 'raw'], cal, r'Cam[0-9]+')
    assert p.skipped == ('Cam9',)
    assert 'Cam9' in capsys.readouterr().out
    assert sorted(p.videos['raw']) == ['Cam1', 'Cam2']


def test_refusal_3_two_videos_on_one_group_camera(tmp_path, monkeypatch):
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'])
    d = _touch(tmp_path / 'a', 'cam0_t.mp4', 'cam1_t.mp4')
    _touch(tmp_path / 'b', 'cam0_t.avi')
    with pytest.raises(SystemExit, match='two videos land on group'):
        adopt.plan([d, tmp_path / 'b'], cal, r'cam([0-9]+)_')


def test_refusal_4_a_group_missing_a_camera(tmp_path, monkeypatch):
    """The rig is the calibration's: a group with a hole would triangulate against another recording."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1', '2'])
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4', 'cam2_t.mp4', 'cam0_u.mp4',
           'cam1_u.mp4')
    with pytest.raises(SystemExit) as e:
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert "group 'u'" in str(e.value) and "['2']" in str(e.value)


def test_refusal_5_an_extension_nothing_can_open(tmp_path, monkeypatch):
    """Only an explicitly named file: a directory expansion already filtered."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['cam0'], calibrated=False)
    _touch(tmp_path / 'rec', 'a.mkv')
    with pytest.raises(SystemExit, match='is not one of'):
        adopt.plan([tmp_path / 'rec' / 'a.mkv'], cal, None)
    # ...and a directory holding only that is refusal 1's empty list, not a crash.
    with pytest.raises(SystemExit, match='matched no file'):
        adopt.plan([tmp_path / 'rec'], cal, None)


def test_refusal_6_three_d_with_an_uncalibrated_camera(tmp_path, monkeypatch):
    """`load_calibration` silently nominalises a block with no matrix; this refuses it instead."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'], calibrated=False)
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4')
    with pytest.raises(SystemExit, match='no matrix/rotation/translation'):
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')


def test_a_moving_rig_is_refused(tmp_path, monkeypatch):
    """`extrinsics.pq` is per-frame geometry no filename can supply; refuse here, not inside `cgroup()`."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'], moving=True)
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4')
    with pytest.raises(SystemExit, match='moving = true'):
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')


# refusals 7-11: flags that mean something on a labelled session and nothing here


def _args(**kw):
    base = dict(anchor='carry', box_prompt='auto', detector=None, boxes='b.npz',
                max_animals=2, split=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize('kw,expect', [
    ({'anchor': 'labels'}, 'has no labels'),
    ({'box_prompt': 'labels'}, 'has no labels'),
    ({'boxes': None, 'detector': None}, 'needs a box source'),
    ({'detector': 'd', 'max_animals': 0}, 'needs --max-animals'),
    ({'split': 'test'}, 'inert here'),
])
def test_refusals_7_to_11(kw, expect, monkeypatch):
    """Pure argparse arithmetic, so each fires before the checkpoint loads or anything decodes."""
    _no_probe(monkeypatch)
    with pytest.raises(SystemExit) as e:
        adopt.check_flags(_args(**kw))
    assert expect in str(e.value), f'wanted {expect!r}, got: {e.value}'


def test_the_flags_that_are_fine_pass(monkeypatch):
    _no_probe(monkeypatch)
    adopt.check_flags(_args())
    adopt.check_flags(_args(detector='d', boxes=None))
    # `--boxes` states the row count in its own first axis, so nothing has to be guessed there.
    adopt.check_flags(_args(boxes='b.npz', max_animals=0))


def test_the_registry_entry_is_a_default_plus_a_refusal(capsys):
    """A label-less session takes its keypoint names from the run's own registry; which entry is the question."""
    one = fmt.Registry(names=('a', 'b'), datasets=(('only', (0, 1)),))
    assert adopt.dataset_name(one, None) == 'only'
    assert "as 'only'" in capsys.readouterr().out
    two = fmt.Registry(names=('a', 'b'), datasets=(('x', (0,)), ('y', (1,))))
    with pytest.raises(SystemExit, match='Pass --dataset-name'):
        adopt.dataset_name(two, None)
    assert adopt.dataset_name(two, 'y') == 'y'
    with pytest.raises(SystemExit, match='not in this run'):
        adopt.dataset_name(two, 'z')


# provenance and the self-checking reconstruction


def _fake_probe(n=6, wh=(64, 48), fps=20.0):
    return lambda path: (n, wh, fps)


def test_the_three_source_keys_reconstruct_the_session(tmp_path):
    """`plan()` is pure over filenames, so calibration + regex + file list (through TOML) re-derive
    the map exactly; an underscore group id cannot collide in a flat list."""
    import tomllib

    import toml

    cal = _calib(tmp_path, ['0', '1'])
    # A group id with an underscore -- the composite-key collision this design refuses to risk.
    _touch(tmp_path / 'rec', 'cam0_a_b.mp4', 'cam1_a_b.mp4', 'cam0_c.mp4', 'cam1_c.mp4')
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert sorted(p.videos) == ['a_b', 'c']

    prov = adopt.provenance_of(p)
    got = tomllib.loads(toml.dumps({'provenance': prov}))['provenance']
    assert got == prov, 'provenance must survive a TOML round trip unchanged'
    assert got['source_session'] == '', 'no directory: an absent root, not a stale one'
    assert got['source_cam_regex'] == r'cam([0-9]+)_'
    assert len(got['source_videos']) == 4 and all(Path(v).is_absolute()
                                                  for v in got['source_videos'])

    back = adopt.plan(got['source_videos'], got['source_calibration'],
                      got['source_cam_regex'] or None,
                      session_id=p.session_id, group_id=got['source_group_id'] or None)
    assert back.videos == p.videos and back.rig.names == p.rig.names


def _pred_session(tmp_path, p, dirname, groups, fps=True):
    """A minimal prediction session over plan `p` with these `groups.pq` rows."""
    import toml

    pred = tmp_path / dirname
    pred.mkdir()
    prov = adopt.provenance_of(p)
    prov['source_session_id'] = p.session_id
    (pred / 'session.toml').write_text(toml.dumps(
        {'mode': '3d', 'units': 'mm', 'labels': 'tracked', 'names': list(KPTS_3D),
         'provenance': prov}))
    fmt.dump_calibration(pred / 'calibration.toml', p.rig)
    cols = {'group_id': np.array(groups, dtype=object),
            'n_frames': np.array([6] * len(groups), np.int32)}
    if fps:
        cols['fps'] = np.array([20.0] * len(groups), np.float32)
    fmt.write_table(pred / 'groups.pq', cols, dict_cols=())
    return pred


def test_a_renamed_video_is_caught_at_reconstruction(tmp_path):
    """`session_from_prediction` re-derives the session and raises on a renamed, moved or added
    video instead of rendering over the wrong pixels."""
    cal = _calib(tmp_path, ['0', '1'])
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4')
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    sess = adopt.build(p, names=KPTS_3D, probe=_fake_probe(), verbose=False)

    good = _pred_session(tmp_path, p, 'pred', ['t'])
    back = adopt.session_from_prediction(good)
    assert list(back.groups) == ['t'] and back.groups['t'].n_frames == 6
    assert back.groups['t'].source('0') == sess.groups['t'].source('0')
    with pytest.raises(fmt.FormatError, match='renamed, moved or added'):
        adopt.session_from_prediction(_pred_session(tmp_path, p, 'wrong', ['somethingelse']))
    (tmp_path / 'rec' / 'cam1_t.mp4').rename(tmp_path / 'rec' / 'cam1_u.mp4')
    with pytest.raises(SystemExit, match='no such file'):
        adopt.session_from_prediction(good)


def test_a_prediction_groups_table_without_fps_reconstructs(tmp_path):
    """`fps` is optional in groups.pq; its fallback was sized by the COLUMN count (3 groups, 2
    columns raised IndexError)."""
    import math

    gids = ['t', 'u', 'v']
    _touch(tmp_path / 'rec', *[f'cam{c}_{g}.mp4' for g in gids for c in (0, 1)])
    p = adopt.plan([tmp_path / 'rec'], _calib(tmp_path, ['0', '1']), r'cam([0-9]+)_')
    back = adopt.session_from_prediction(_pred_session(tmp_path, p, 'pred', gids, fps=False))
    assert sorted(back.groups) == gids
    assert all(math.isnan(g.fps) for g in back.groups.values())


# with pixels


def _two_cam(tmp_path, T=6, wh=(64, 48), lens=None):
    cal = _calib(tmp_path, ['0', '1'], wh=wh)
    for i in (0, 1):
        n = T if lens is None else lens[i]
        _write_video(tmp_path / 'rec' / f'cam{i}_t.mp4', i, n, wh)
    return cal


def test_frame_counts_come_from_the_decoder(tmp_path):
    """`n_frames` comes from the decoder (every index in [0, T) decodes), not container metadata."""
    from tailcyclenet.data import video
    from tailcyclenet.data.pixels import read_frames

    cal = _two_cam(tmp_path, T=7)
    sess = adopt.build(adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_'),
                       names=KPTS_3D, verbose=False)
    g = sess.groups['t']
    reader = video.open_reader(str(tmp_path / 'rec' / 'cam0_t.mp4'))
    assert g.n_frames == len(reader)
    reader.close()
    for cam in sess.cam_names:
        imgs = read_frames(g, cam, np.arange(g.n_frames))
        assert len(imgs) == g.n_frames and all(im is not None for im in imgs)


def test_length_mismatch_is_refused_and_trims_only_on_request(tmp_path, capsys):
    """A ragged group is refused (`--trim-to-shortest` opts in); a size mismatch with the calibration
    is refused (it projects wrong with no symptom)."""
    cal = _two_cam(tmp_path, lens=(6, 9))
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    with pytest.raises(SystemExit, match='disagree on length'):
        adopt.build(p, names=KPTS_3D, verbose=False)
    sess = adopt.build(p, names=KPTS_3D, trim=True, verbose=False)
    assert sess.groups['t'].n_frames == 6
    assert 'trimmed to 6' in capsys.readouterr().out

    (tmp_path / 'small').mkdir()
    cal = _calib(tmp_path / 'small', ['0', '1'], wh=(64, 48))
    for i in (0, 1):
        _write_video(tmp_path / 'small' / 'rec' / f'cam{i}_t.mp4', i, 6, (32, 24))
    p = adopt.plan([tmp_path / 'small' / 'rec'], cal, r'cam([0-9]+)_')
    with pytest.raises(SystemExit) as e:
        adopt.build(p, names=KPTS_3D, verbose=False)
    assert '32x24' in str(e.value) and '64x48' in str(e.value)


def test_a_camera_block_with_no_size_is_filled_from_the_pixels(tmp_path, capsys):
    """A size derived from the pixels cannot be wrong about the pixels; the fill is printed."""
    import tomllib

    import toml

    cal = _two_cam(tmp_path)
    with open(cal, 'rb') as f:
        doc = tomllib.load(f)
    doc['cam_1'].pop('size')
    cal.write_text(toml.dumps(doc))
    with pytest.raises(fmt.FormatError, match='has no size'):
        fmt.load_calibration(cal)

    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert p.need_size == ('1',)
    sess = adopt.build(p, names=KPTS_3D, verbose=True)
    assert sess.rig.size('1') == (64, 48)
    assert "carried no `size`" in capsys.readouterr().out


def test_the_in_memory_session_matches_the_written_one(tmp_path):
    """The acceptance test `validate_session` cannot run on a `VideoSession`: the in-memory session,
    dumped with symlinked pixels, validates and agrees field by field."""
    cal = _two_cam(tmp_path, T=6)
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    mem = adopt.build(p, names=KPTS_3D, units='mm', assoc_res_max_px=17.5, verbose=False)

    out = tmp_path / 'written'
    adopt.dump(mem, out)
    disk = fmt.Session.load(out)

    errs = fmt.validate_session(disk)
    assert not errs, f'the written form must be valid under every rule: {errs}'

    assert (disk.mode, disk.units, disk.label_source) == (mem.mode, mem.units, mem.label_source)
    assert disk.names == mem.names and disk.cam_names == mem.cam_names
    assert disk.assoc_res_max_px == mem.assoc_res_max_px
    for cam in mem.cam_names:
        np.testing.assert_allclose(
            disk.rig.by_name(cam).get_camera_matrix().detach().cpu().numpy(),
            mem.rig.by_name(cam).get_camera_matrix().detach().cpu().numpy())
        np.testing.assert_allclose(
            disk.rig.by_name(cam).get_extrinsics_mat().detach().cpu().numpy(),
            mem.rig.by_name(cam).get_extrinsics_mat().detach().cpu().numpy())
        assert disk.rig.size(cam) == mem.rig.size(cam)
    assert sorted(disk.groups) == sorted(mem.groups)
    for gid in mem.groups:
        assert disk.groups[gid].n_frames == mem.groups[gid].n_frames
        assert disk.groups[gid].fps == pytest.approx(mem.groups[gid].fps)
        a, b = disk.labels(gid), mem.labels(gid)
        assert a.animal_ids == b.animal_ids == []
        for field_ in ('points3d', 'vis3d', 'points2d', 'vis2d'):
            x, y = getattr(a, field_), getattr(b, field_)
            assert (x is None) == (y is None)
            if x is not None:
                assert x.shape == y.shape and x.dtype == y.dtype
    # And the written form's pixels are the SAME files, reached a different way.
    for gid in mem.groups:
        for cam in mem.cam_names:
            assert (Path(disk.groups[gid].pixels(cam)[1]).resolve()
                    == Path(mem.groups[gid].source(cam)[1]).resolve())


