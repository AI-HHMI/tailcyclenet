"""`--videos`: raw footage plus an anipose calibration, adopted as a session IN MEMORY.

Split the way the module is: the naming rule and every refusal that does not need pixels run in
milliseconds with no video fixture at all, and the handful that genuinely need containers are
grouped at the bottom.

**THE ACCEPTANCE TEST FOR THE CONSTRUCTION IS HERE, NOT AT RUNTIME.** `validate_session` cannot
certify a `VideoSession` -- it resolves pixels through `Group.pixels()` and reads tables off
`path`, neither of which exists -- so `test_the_in_memory_session_matches_the_written_one` builds
the same plan both ways and compares. The format's own validator still certifies it; it just does
it once, in CI, instead of on every run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from tailcyclenet import adopt
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
    """Assert nothing decodes. Every refusal below fires above the checkpoint load AND above any
    decode, which is what makes them testable with no video fixture."""
    calls = []

    def boom(path):
        calls.append(path)
        raise AssertionError(f'a pure refusal decoded {path}')

    monkeypatch.setattr(adopt, '_probe', boom)
    return calls


# ---------------------------------------------------------------------------------------------
# the naming rule


@pytest.mark.parametrize('stem,rx,cam,gid', [
    # anipose's own documented patterns, and its own convention: the CAMERA IS THE CAPTURE GROUP.
    ('cam0_trial3', r'cam([0-9]+)_', '0', 'trial3'),
    ('cam12_trial3', r'cam([0-9]+)_', '12', 'trial3'),
    ('vid_A', r'_([A-Z])$', 'A', 'vid'),
    ('session1-camB', r'-cam([A-Z])', 'B', 'session1'),
    # SUPERSET 1: no capture group means the WHOLE match. anipose raises IndexError here, and
    # this is what johnson's `Cam2005325.mp4` beside a `Cam2005325` calibration entry needs --
    # the capture-group convention would name it '2005325' and refusal 2 would fire on all 16.
    #
    # AND THE SEPARATOR IS NOT CONSUMED, which is the price of the superset and is anipose's own
    # `re.sub` rule verbatim: 'cam[0-9]+' matches 'cam0' and leaves '_trial3', where
    # 'cam([0-9]+)_' matches the underscore too and leaves 'trial3'. Both are stable group ids and
    # neither is wrong; they are simply different, which is exactly why this asserts VALUES.
    ('cam0_trial3', r'cam[0-9]+', 'cam0', '_trial3'),
    ('cam0_trial3', r'cam[0-9]+_', 'cam0_', 'trial3'),
    ('Cam2005325', r'Cam[0-9]+', 'Cam2005325', ''),
    # SUPERSET 2: no regex at all -- the whole stem is the group, the caller names the camera.
    ('anything_at_all', None, '', 'anything_at_all'),
])
def test_the_anipose_naming_rule(stem, rx, cam, gid):
    """Asserted on VALUES, both halves, not on shapes -- the `p`-column lesson from
    `test_convert_apt_lbl.py`. Both the camera name and the group id are derived from one regex
    and either can be silently wrong."""
    assert adopt.parse_name(stem, rx) == (cam, gid)


def test_one_camera_needs_no_regex(tmp_path, monkeypatch):
    """The 2D single-view case, which is most of the intended traffic. Demanding a regex to select
    from a set of one is ceremony."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['cam0'], calibrated=False)
    _touch(tmp_path / 'rec', 'whatever.mp4', 'other.mp4')
    p = adopt.plan([tmp_path / 'rec'], cal, None)
    assert p.mode == '2d'
    assert sorted(p.videos) == ['other', 'whatever']
    assert all(list(v) == ['cam0'] for v in p.videos.values())


def test_multiple_groups_in_one_session_is_free(tmp_path, monkeypatch):
    """A session holds one calibration, one mode and one keypoint axis, and every video in one
    invocation shares all three by construction. Twelve trials of a four-camera rig is ONE session
    of twelve groups, and that is the invocation this feature is for."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'])
    _touch(tmp_path / 'rec', *[f'cam{i}_{g}.mp4' for g in ('a', 'b', 'c') for i in (0, 1)])
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert sorted(p.videos) == ['a', 'b', 'c']
    assert p.session_id == 'rec' and p.group_id == ''


def test_the_empty_remainder_rule_is_about_disagreement(tmp_path, monkeypatch):
    """`Cam2005325.mp4` under `Cam[0-9]+` leaves '': the whole stem IS the camera name, because a
    raw multi-camera recording of ONE session has nothing else to put in the filename. That is the
    shape of every raw rig dump, and an earlier "an empty group id raises" rule refused the first
    real input this feature had.

    But `cam0.mp4` beside `cam0_trial3.mp4` is a genuine ambiguity -- one of them is mis-named or
    the regex is wrong -- and merging the bare one into a group called '' is the silent merge the
    rule was reaching for.
    """
    _no_probe(monkeypatch)
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


# ---------------------------------------------------------------------------------------------
# refusals 1-6: pure over the filenames


def test_refusal_1_no_video_matches_the_regex(tmp_path, monkeypatch):
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'])
    _touch(tmp_path / 'rec', 'a.mp4', 'b.mp4')
    with pytest.raises(SystemExit) as e:
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert 'matched none' in str(e.value) and 're.search' in str(e.value)


def test_refusal_2_no_parsed_camera_is_in_the_calibration(tmp_path, monkeypatch):
    """THE COLLISION THAT WILL BE THE MOST COMMON ERROR BY A WIDE MARGIN: `cam([0-9]+)_` yields
    '0' and the calibration names it 'cam0'. The message has to say "the camera name is the
    CAPTURE GROUP" or it reads as "no cameras matched"."""
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


def test_refusal_2_some_matched_skips_rather_than_refusing(tmp_path, monkeypatch, capsys):
    """REVISED AGAINST REAL DATA. `mouse_2_validate` ships 17 mp4s against 16 calibrated cameras:
    a camera with video and no geometry can never join any group, so refusing the whole run is
    wrong -- but silently dropping pixels is worse. None matched -> refuse; some -> SKIP the
    unmatched, PRINTING each by name."""
    _no_probe(monkeypatch)
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
    """The rig is the CALIBRATION'S, so a group with a hole would triangulate against a camera
    whose pixels are a different recording -- and `video_group` would otherwise raise against a
    nonexistent directory, which is loud for the wrong reason."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1', '2'])
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4', 'cam2_t.mp4', 'cam0_u.mp4',
           'cam1_u.mp4')
    with pytest.raises(SystemExit) as e:
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    assert "group 'u'" in str(e.value) and "['2']" in str(e.value)


def test_refusal_5_an_extension_nothing_can_open(tmp_path, monkeypatch):
    """Only for an EXPLICITLY named file -- a directory expansion already filtered, so an
    unopenable container here is something the user typed."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['cam0'], calibrated=False)
    _touch(tmp_path / 'rec', 'a.mkv')
    with pytest.raises(SystemExit, match='is not one of'):
        adopt.plan([tmp_path / 'rec' / 'a.mkv'], cal, None)
    # ...and a directory holding only that is refusal 1's empty list, not a crash.
    with pytest.raises(SystemExit, match='matched no file'):
        adopt.plan([tmp_path / 'rec'], cal, None)


def test_refusal_6_three_d_with_an_uncalibrated_camera(tmp_path, monkeypatch):
    """LOAD-BEARING, not a convenience: `load_calibration` silently turns a block with no `matrix`
    into a `nominal_camera`, and with `validate_session` out of the loop nothing downstream would
    ever say so."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'], calibrated=False)
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4')
    with pytest.raises(SystemExit, match='no matrix/rotation/translation'):
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')


def test_a_moving_rig_is_refused(tmp_path, monkeypatch):
    """`extrinsics.pq` is per-frame geometry no filename can supply. Refused HERE rather than left
    to fail: `labels()` is overridden on this path, so a moving camera sails past the check that
    lives there and blows up inside `cgroup()` instead, much later and much worse."""
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'], moving=True)
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4')
    with pytest.raises(SystemExit, match='moving = true'):
        adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')


def test_the_regex_is_required_when_the_rig_is_not_a_singleton(tmp_path, monkeypatch):
    _no_probe(monkeypatch)
    cal = _calib(tmp_path, ['0', '1'])
    _touch(tmp_path / 'rec', 'a.mp4', 'b.mp4')
    with pytest.raises(SystemExit, match='--cam-regex is required'):
        adopt.plan([tmp_path / 'rec'], cal, None)


# ---------------------------------------------------------------------------------------------
# refusals 7-11: the flags that mean something on a labelled session and nothing here


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
    """Every one is pure argparse arithmetic, so it fires before the checkpoint loads AND before
    anything decodes -- which the probe guard pins."""
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
    """A session with no labels still needs `names`, and there is no data to derive them from --
    so they come from the RUN's own registry. Which entry is the open question."""
    one = fmt.Registry(names=('a', 'b'), datasets=(('only', (0, 1)),))
    assert adopt.dataset_name(one, None) == 'only'
    assert "as 'only'" in capsys.readouterr().out
    two = fmt.Registry(names=('a', 'b'), datasets=(('x', (0,)), ('y', (1,))))
    with pytest.raises(SystemExit, match='Pass --dataset-name'):
        adopt.dataset_name(two, None)
    assert adopt.dataset_name(two, 'y') == 'y'
    with pytest.raises(SystemExit, match='not in this run'):
        adopt.dataset_name(two, 'z')


# ---------------------------------------------------------------------------------------------
# provenance and the self-checking reconstruction


def _fake_probe(n=6, wh=(64, 48), fps=20.0):
    return lambda path: (n, wh, fps)


def test_the_three_source_keys_reconstruct_the_session(tmp_path):
    """`plan()` IS PURE OVER THE FILENAMES, so calibration + regex + file list re-derives the
    group -> camera map exactly. Recording a structure a pure function already computes is a
    second copy that can disagree with the first -- and a per-(group, camera) key would collide
    (group `a_b` camera `c` against group `a` camera `b_c`), which `SessionWriter` only catches
    when the two values differ, i.e. not reliably. A flat list has no key to collide.

    Round-tripped through TOML on purpose: `source_videos` is the first non-scalar provenance
    value in this repo, and the regex is full of metacharacters that must survive quoting.
    """
    import tomllib

    import toml

    cal = _calib(tmp_path, ['0', '1'])
    # A group id WITH AN UNDERSCORE -- the composite-key collision this design refuses to risk.
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


def test_a_renamed_video_is_caught_at_reconstruction(tmp_path):
    """THE SELF-CHECK, and it is the reason the derived map is not stored.

    A prediction is self-describing about the pixels it was made from, with no second directory to
    go stale against it -- so a video renamed, moved or added since the run RAISES naming the
    difference instead of rendering a prediction over the wrong pixels.
    """
    import toml

    def _pred(dirname, groups):
        pred = tmp_path / dirname
        pred.mkdir()
        prov = adopt.provenance_of(p)
        prov['source_session_id'] = p.session_id
        (pred / 'session.toml').write_text(toml.dumps(
            {'mode': '3d', 'units': 'mm', 'labels': 'tracked', 'names': list(KPTS_3D),
             'provenance': prov}))
        fmt.dump_calibration(pred / 'calibration.toml', p.rig)
        fmt.write_table(pred / 'groups.pq', {
            'group_id': np.array(groups, dtype=object),
            'n_frames': np.array([6] * len(groups), np.int32),
            'fps': np.array([20.0] * len(groups), np.float32)}, dict_cols=())
        return pred

    cal = _calib(tmp_path, ['0', '1'])
    _touch(tmp_path / 'rec', 'cam0_t.mp4', 'cam1_t.mp4')
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    sess = adopt.build(p, names=KPTS_3D, probe=_fake_probe(), verbose=False)

    good = _pred('pred', ['t'])
    back = adopt.session_from_prediction(good)
    assert list(back.groups) == ['t'] and back.groups['t'].n_frames == 6
    assert back.groups['t'].source('0') == sess.groups['t'].source('0')

    # (a) THE GROUPS DISAGREE -- the self-check the stored map could not do. The prediction was
    # written over a group these videos no longer derive.
    with pytest.raises(fmt.FormatError, match='renamed, moved or added'):
        adopt.session_from_prediction(_pred('pred_wrong_group', ['somethingelse']))

    # (b) A VIDEO MOVED. `provenance_of` records the RESOLVED file list, so the reconstruction
    # replays exactly those paths and a missing one is refused by name rather than dropped --
    # which is the whole reason the list is resolved and expanded before it is recorded.
    (tmp_path / 'rec' / 'cam1_t.mp4').rename(tmp_path / 'rec' / 'cam1_u.mp4')
    with pytest.raises(SystemExit, match='no such file'):
        adopt.session_from_prediction(good)


# ---------------------------------------------------------------------------------------------
# with pixels


def _two_cam(tmp_path, T=6, wh=(64, 48), lens=None):
    cal = _calib(tmp_path, ['0', '1'], wh=wh)
    for i in (0, 1):
        n = T if lens is None else lens[i]
        _write_video(tmp_path / 'rec' / f'cam{i}_t.mp4', i, n, wh)
    return cal


def test_frame_counts_come_from_the_decoder(tmp_path):
    """`Group.n_frames` is a PROMISE that every index in [0, T) decodes, and `dataset._read_video`
    fulfils it through `decord.VideoReader.get_batch`. A container-metadata count that disagrees
    with decord's own index -- routine on a VFR or truncated file -- would turn into a hard
    failure deep inside the window loop, after the checkpoint has loaded."""
    from decord import VideoReader

    from tailcyclenet.dataset import read_frames

    cal = _two_cam(tmp_path, T=7)
    sess = adopt.build(adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_'),
                       names=KPTS_3D, verbose=False)
    g = sess.groups['t']
    assert g.n_frames == len(VideoReader(str(tmp_path / 'rec' / 'cam0_t.mp4'), num_threads=1))
    for cam in sess.cam_names:
        imgs = read_frames(g, cam, np.arange(g.n_frames))
        assert len(imgs) == g.n_frames and all(im is not None for im in imgs)


def test_length_mismatch_is_refused_and_trims_only_on_request(tmp_path, capsys):
    """A one-frame offset is usually a dropped trigger and usually harmless; a 40,000-frame offset
    is two different recordings sharing a group id, and both look identical to a min()."""
    cal = _two_cam(tmp_path, lens=(6, 9))
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    with pytest.raises(SystemExit, match='disagree on length'):
        adopt.build(p, names=KPTS_3D, verbose=False)
    sess = adopt.build(p, names=KPTS_3D, trim=True, verbose=False)
    assert sess.groups['t'].n_frames == 6
    assert 'trimmed to 6' in capsys.readouterr().out


def test_size_mismatch_is_refused(tmp_path):
    """`matrix` and `distortions` are in SENSOR pixels and `size` is the image on disk, so a
    calibration made at twice the deployment resolution projects to the wrong place with no
    symptom other than being wrong. This is the one `validate_session` rule 8 would have caught
    for an image root and never checks for video -- not a regression, a check the format never
    had."""
    cal = _calib(tmp_path, ['0', '1'], wh=(64, 48))
    for i in (0, 1):
        _write_video(tmp_path / 'rec' / f'cam{i}_t.mp4', i, 6, (32, 24))
    p = adopt.plan([tmp_path / 'rec'], cal, r'cam([0-9]+)_')
    with pytest.raises(SystemExit) as e:
        adopt.build(p, names=KPTS_3D, verbose=False)
    assert '32x24' in str(e.value) and '64x48' in str(e.value)


def test_a_camera_block_with_no_size_is_filled_from_the_pixels(tmp_path, capsys):
    """The one place this path may be generous: a size derived from the pixels cannot be wrong
    about the pixels. `load_calibration` raises on such a block today, so the fill is real work
    and it is PRINTED."""
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
    """THE REPLACEMENT FOR `validate_session` AS THE ACCEPTANCE TEST.

    `validate_session` resolves pixels through `Group.pixels()` and reads tables off `path`, and a
    `VideoSession` has neither -- so it cannot certify one at runtime. Instead: build the session
    in memory, write the SAME plan out through `format.write_session` with the pixels symlinked,
    and assert (a) `validate_session` on the written one returns [] under ALL rules with no
    exemption, and (b) the two agree field for field. The format's own validator still certifies
    the construction; it just does it once, here, instead of on every run.
    """
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
