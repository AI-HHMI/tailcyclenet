"""The post-hoc identity bridge (`tailcyclenet/infer/bridge.py`).

The bridge is opt-in and trades misses for identity switches, so the properties that matter most
are the ones that bound the damage: it must be a no-op when there is nothing to repair, it must
only ever permute rows inside a declared component, and it must refuse rather than guess.
"""
import numpy as np
import pandas as pd
import pytest

from tailcyclenet.infer import bridge


def _windows(n_windows, stride, gid='g'):
    """One window table: window `i` starts at `i * stride`."""
    return pd.DataFrame({'group_id': [gid] * n_windows,
                         'window': list(range(n_windows)),
                         'frame': [i * stride for i in range(n_windows)]})


def _events(rows, gid='g'):
    return pd.DataFrame({'group_id': [gid] * len(rows),
                         'frame': [r[0] for r in rows],
                         'slot': [r[1] for r in rows],
                         'event': [r[2] for r in rows],
                         'detail': [''] * len(rows)})


def _two_animals_crossing(n_frames=240, k=3):
    """Two rows on straight, well-separated tracks, plus a third far away and unrelated."""
    pred = np.full((3, n_frames, k, 3), np.nan, np.float32)
    t = np.arange(n_frames, dtype=np.float32)
    for j in range(k):
        pred[0, :, j, 0] = t          # row 0 moves +x
        pred[0, :, j, 1] = 0.0 + j
        pred[1, :, j, 0] = -t         # row 1 moves -x
        pred[1, :, j, 1] = 100.0 + j
        pred[2, :, j, 0] = 500.0      # row 2 parked, never involved
        pred[2, :, j, 1] = 500.0 + j
    pred[..., 2] = 0.0
    return pred


def test_config_refuses_nonsense_by_name():
    for kw in ({'recovery_windows': 0}, {'horizon_steps': 0}, {'horizon_stride': 0},
               {'min_margin': -1}, {'missing_cost': 0}, {'pre_event_windows': -1},
               {'max_event_gap': -1}):
        with pytest.raises(ValueError):
            bridge.BridgeConfig(**kw).validate()
    bridge.BridgeConfig().validate()


def test_owned_segments_are_last_write_wins_and_contiguous():
    """A later window overwrites the overlap, so ownership is the stride, not the window."""
    seg = bridge.owned_segments(_windows(4, 10), 'g', n_frames=100, window_length=25)
    assert [len(seg[w]) for w in (0, 1, 2)] == [10, 10, 10]
    assert seg[0][0] == 0 and seg[0][-1] == 9
    # the final window keeps everything to the end of the clip
    assert seg[3][0] == 30 and seg[3][-1] == 54
    for frames in seg.values():
        assert np.all(np.diff(frames) == 1)


def test_episodes_gap_join_and_drop_single_slot():
    ev = _events([(10, 2, 'retired_duplicate'), (12, 3, 'shielded'),
                  (400, 5, 'retired_duplicate'), (401, 6, 'shielded'),
                  (800, 7, 'retired_duplicate')])
    eps = bridge.episodes(ev, 'g', max_gap=64)
    assert [e['component'] for e in eps] == [(2, 3), (5, 6)]  # the lone slot-7 episode is dropped
    assert eps[0]['first_frame'] == 10 and eps[0]['last_frame'] == 12


def test_episodes_ignore_births_and_other_groups():
    ev = _events([(10, 2, 'born'), (11, 3, 'birth_refused'), (12, 4, 'died')])
    assert bridge.episodes(ev, 'g', 64) == []
    ev2 = _events([(10, 2, 'retired_duplicate'), (11, 3, 'shielded')], gid='other')
    assert bridge.episodes(ev2, 'g', 64) == []


def test_no_event_log_is_a_byte_exact_no_op():
    """Output neutrality: a session with no episodes must come back untouched."""
    rows = {'pred': _two_animals_crossing(), 'group_id': 'g'}
    before = rows['pred'].copy()
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), _events([]), 'g',
                                         n_frames=240, window_length=10,
                                         cfg=bridge.BridgeConfig())
    assert decisions == []
    assert np.array_equal(out['pred'], before, equal_nan=True)


def test_identity_mapping_is_skipped_rather_than_quarantined():
    """The measured miss saving: an episode whose best map is the identity repairs nothing.

    Two rows on clean, separate tracks produce a duplicate event but no actual exchange, so the
    bridge must leave every frame alone instead of spending coverage on a no-op permutation.
    """
    rows = {'pred': _two_animals_crossing(), 'group_id': 'g'}
    before = rows['pred'].copy()
    ev = _events([(100, 0, 'retired_duplicate'), (100, 1, 'shielded')])
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                         bridge.BridgeConfig())
    assert len(decisions) == 1 and decisions[0].get('identity') is True
    assert np.array_equal(out['pred'], before, equal_nan=True)


def test_a_real_swap_is_repaired_and_only_inside_the_component():
    """After an induced row exchange the bridge must undo it, and never touch row 2."""
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred, 'group_id': 'g'}
    row2_before = pred[2].copy()
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                         bridge.BridgeConfig())
    assert len(decisions) == 1
    plan = decisions[0]
    assert not plan.get('skipped') and not plan.get('refused')
    assert plan['mapping'] == [1, 0], 'the exchange must be undone, not re-applied'
    # row 2 is outside the component and must be byte-identical
    assert np.array_equal(out['pred'][2], row2_before, equal_nan=True)
    # far past the episode, row 0 is back on its own +x track
    assert out['pred'][0, 230, 0, 0] == pytest.approx(230.0)


def test_quarantine_nans_the_component_before_release():
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred, 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                         bridge.BridgeConfig())
    plan = decisions[0]
    quarantined = [w for w in plan['windows'] if w < plan['release']]
    assert quarantined, 'a repaired episode must quarantine something'
    frame = 10 * quarantined[0]
    assert np.isnan(out['pred'][0, frame]).all() and np.isnan(out['pred'][1, frame]).all()
    assert not np.isnan(out['pred'][2, frame]).any(), 'row 2 is not in the component'


def test_every_row_field_takes_the_same_permutation():
    """Pose from one animal with boxes from another would be worse than the swap it repairs."""
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    boxes = np.zeros((3, 240, 2, 4), np.float32)
    boxes[0], boxes[1], boxes[2] = 10.0, 20.0, 30.0
    rows = {'pred': pred, 'boxes': boxes, 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                         bridge.BridgeConfig())
    assert decisions[0]['mapping'] == [1, 0]
    assert out['boxes'][0, 230, 0, 0] == pytest.approx(20.0), 'boxes must follow the pose'
    assert out['boxes'][2, 230, 0, 0] == pytest.approx(30.0), 'row 2 untouched'


def test_an_unbridgeable_episode_is_left_completely_alone():
    """An episode with no clean anchor before it must not be half-repaired."""
    rows = {'pred': _two_animals_crossing(), 'group_id': 'g'}
    before = rows['pred'].copy()
    ev = _events([(1, 0, 'retired_duplicate'), (1, 1, 'shielded')])
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                         bridge.BridgeConfig())
    assert decisions[0].get('skipped') is True
    assert np.array_equal(out['pred'], before, equal_nan=True)


def test_an_impossible_margin_refuses_and_quarantines_instead_of_guessing():
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred, 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])
    cfg = bridge.BridgeConfig(min_margin=1e12)  # nothing can clear this
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10, cfg)
    plan = decisions[0]
    assert plan['refused'] is True and plan['mapping'] is None
    assert np.isnan(out['pred'][0, 10 * plan['windows'][0]]).all()


def _nearly_parallel_tracks(n_frames=240, k=3):
    """Two rows moving in almost the same direction -- an AMBIGUOUS case, unlike the cleanly
    separated opposite-direction crossing every other test in this file uses. A swap here scores
    a real but LOW margin, the shape that damaged a perfectly clean clip in dev/reports/57.
    """
    pred = np.full((2, n_frames, k, 3), np.nan, np.float32)
    t = np.arange(n_frames, dtype=np.float32)
    for j in range(k):
        pred[0, :, j, 0] = t
        pred[0, :, j, 1] = 0.0 + j
        pred[1, :, j, 0] = t * 1.02 + 5.0
        pred[1, :, j, 1] = 2.0 + j
    pred[..., 2] = 0.0
    return pred


def test_a_low_margin_release_is_refused_at_the_shipped_default_but_not_at_the_old_one():
    """dev/reports/57: min_margin=1.0 (report 54's arm) released a real permutation on a
    PERFECTLY CLEAN 3dpop clip (Sequence42) at margin 43.08, turning idsw 0 into 2 -- CLAUDE.md's
    own kill condition for the whole mechanism. min_margin=60.0 (the shipped default since) must
    refuse this shape of low-margin release; the OLD default must still apply it, so this test
    would have caught the regression before it shipped.
    """
    pred = _nearly_parallel_tracks()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred, 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])

    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                         bridge.BridgeConfig())
    plan = decisions[0]
    assert plan['refused'] is True, 'the shipped default must refuse a low-margin release'
    assert plan['mapping'] is None
    assert np.isnan(out['pred'][0, 10 * plan['windows'][0]]).all(), 'refused means quarantined'

    pred_old = _nearly_parallel_tracks()
    pred_old[0, swap_at:], pred_old[1, swap_at:] = tail[1], tail[0]
    rows_old = {'pred': pred_old, 'group_id': 'g'}
    _, decisions_old = bridge.bridge_group(rows_old, _windows(24, 10), ev, 'g', 240, 10,
                                           bridge.BridgeConfig(min_margin=1.0))
    plan_old = decisions_old[0]
    assert not plan_old.get('refused'), 'the OLD default applied this release -- reproduces the bug'
    assert plan_old['mapping'] is not None


def test_backward_agrees_is_a_diagnostic_that_never_changes_the_arrays():
    """plan §7.1 / report 64: an independent backward confirmation, stored on the decision dict
    only -- it must never affect which frames are quarantined or how they are permuted.
    """
    pred = _two_animals_crossing(n_frames=400)
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred.copy(), 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])

    out, decisions = bridge.bridge_group(rows, _windows(40, 10), ev, 'g', 400, 10,
                                         bridge.BridgeConfig())
    plan = decisions[0]
    assert 'backward_agrees' in plan
    assert plan['backward_agrees'] in (True, False, None)
    # an involution (2-row swap) is its own inverse, so a real repair with room for a backward
    # anchor must agree with itself here
    assert plan['backward_agrees'] is True

    # identity and refused outcomes carry the key too (unconditional membership), always None
    out_id, dec_id = bridge.bridge_group(
        {'pred': _two_animals_crossing(), 'group_id': 'g'}, _windows(24, 10),
        _events([(100, 0, 'retired_duplicate'), (100, 1, 'shielded')]), 'g', 240, 10,
        bridge.BridgeConfig())
    assert dec_id[0]['identity'] is True and dec_id[0]['backward_agrees'] is None


def test_all_candidates_are_recorded_sorted_by_margin_and_change_nothing():
    """plan §7 item 2 / report 73: every horizon-invariant candidate is logged, not just the
    winner -- a pure diagnostic, sorted descending by margin, that must never change WHICH
    candidate wins or any array output.
    """
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred.copy(), 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])

    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                         bridge.BridgeConfig())
    plan = decisions[0]
    assert 'all_candidates' in plan
    assert len(plan['all_candidates']) >= 1
    margins = [c['mean_margin'] for c in plan['all_candidates']]
    assert margins == sorted(margins, reverse=True), 'must be sorted descending by margin'
    winner = plan['all_candidates'][0]
    assert winner['release'] == plan['release']
    assert winner['mapping'] == plan['mapping']

    # identity and refused outcomes carry the field too (unconditional membership)
    _, dec_id = bridge.bridge_group(
        {'pred': _two_animals_crossing(), 'group_id': 'g'}, _windows(24, 10),
        _events([(100, 0, 'retired_duplicate'), (100, 1, 'shielded')]), 'g', 240, 10,
        bridge.BridgeConfig())
    assert dec_id[0]['identity'] is True and 'all_candidates' in dec_id[0]


def test_rewrite_tables_deletes_quarantine_and_relabels_the_release(tmp_path):
    """A permutation IS an `animal_id` relabel and a quarantine IS a row deletion.

    Rebuilding the tables from arrays would drop `conf2d`, which does not survive a
    `load_predictions` round trip, so the bridge edits rows instead. This pins that contract.
    """
    from tailcyclenet.format import DICT_COLS, write_table
    frames = np.array([0, 0, 5, 5, 20, 20], np.int32)
    animals = np.array(['a0', 'a1', 'a0', 'a1', 'a0', 'a1'], dtype=object)
    write_table(tmp_path / 'points3d.pq', {
        'group_id': np.array(['g'] * 6, dtype=object), 'frame': frames,
        'animal_id': animals, 'bodypart': np.array(['n'] * 6, dtype=object),
        'status': np.array(['visible'] * 6, dtype=object),
        'x': np.arange(6, dtype=np.float32), 'y': np.zeros(6, np.float32),
        'z': np.zeros(6, np.float32)}, dict_cols=DICT_COLS)
    segments = {0: np.arange(0, 10), 1: np.arange(10, 30)}
    plan = {'component': [0, 1], 'windows': [0, 1], 'release': 1, 'mapping': [1, 0],
            'refused': False}
    bridge.rewrite_tables(tmp_path, 'g', ['a0', 'a1'], segments, [plan], n_frames=30)
    out = pd.read_parquet(tmp_path / 'points3d.pq')
    # window 0 (frames 0..9) is quarantined, so its rows are gone entirely
    assert set(out.frame.tolist()) == {20}
    # window 1 onward is released under [1, 0], so the two ids are exchanged
    got = dict(zip(out.animal_id.astype(str), out.x))
    assert got['a0'] == pytest.approx(5.0) and got['a1'] == pytest.approx(4.0)


def test_rewrite_tables_leaves_other_groups_byte_exact(tmp_path):
    from tailcyclenet.format import DICT_COLS, write_table
    write_table(tmp_path / 'points3d.pq', {
        'group_id': np.array(['g', 'g', 'other', 'other'], dtype=object),
        'frame': np.array([0, 0, 0, 0], np.int32),
        'animal_id': np.array(['a0', 'a1', 'a0', 'a1'], dtype=object),
        'bodypart': np.array(['n'] * 4, dtype=object),
        'status': np.array(['visible'] * 4, dtype=object),
        'x': np.arange(4, dtype=np.float32), 'y': np.zeros(4, np.float32),
        'z': np.zeros(4, np.float32)}, dict_cols=DICT_COLS)
    plan = {'component': [0, 1], 'windows': [0], 'release': None, 'mapping': None,
            'refused': True}
    bridge.rewrite_tables(tmp_path, 'g', ['a0', 'a1'], {0: np.arange(0, 10)}, [plan], 10)
    out = pd.read_parquet(tmp_path / 'points3d.pq')
    assert out.group_id.astype(str).tolist() == ['other', 'other']
    assert out.x.tolist() == [2.0, 3.0]


def test_provenance_lists_every_lever_unconditionally():
    """A partial record lies: an absent key reads as 'not used' when it means 'unknown'."""
    keys = {k for k, _ in bridge.config_provenance(bridge.BridgeConfig())}
    assert keys == {f'bridge_{f}' for f in bridge.BridgeConfig.__dataclass_fields__}


def test_interpolate_fills_the_quarantine_with_a_linear_ramp_between_true_anchors():
    """Section 2.3: with `interpolate=True` and a real release, the quarantine gap is filled
    instead of NaN'd, and on rows whose true trajectory IS linear (this fixture), interpolation
    must recover it almost exactly -- the strongest test a linear interpolator can pass.
    """
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred, 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])
    cfg = bridge.BridgeConfig(interpolate=True)
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10, cfg)
    plan = decisions[0]
    quarantined = [w for w in plan['windows'] if w < plan['release']]
    assert quarantined, 'a repaired episode must still have something to fill'
    frame = 10 * quarantined[0]
    assert not np.isnan(out['pred'][0, frame]).any(), 'interpolate=True must not leave NaN here'
    # row 0's true track is x=frame, y=0 -- a linear interpolator on a linear track is exact
    # up to the boundary anchors' own frame spacing.
    assert out['pred'][0, frame, 0, 0] == pytest.approx(float(frame), abs=2.0)
    assert out['pred'][0, frame, 0, 1] == pytest.approx(0.0, abs=2.0)
    # row 2, outside the component, must still be byte-untouched
    assert not np.isnan(out['pred'][2, frame]).any()


def test_interpolate_defaults_off_and_is_byte_identical_to_before():
    """The flag must be additive: unset, behaviour is exactly the pre-2.3 NaN quarantine."""
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])

    rows_a = {'pred': pred.copy(), 'group_id': 'g'}
    out_a, _ = bridge.bridge_group(rows_a, _windows(24, 10), ev, 'g', 240, 10,
                                   bridge.BridgeConfig())
    rows_b = {'pred': pred.copy(), 'group_id': 'g'}
    out_b, _ = bridge.bridge_group(rows_b, _windows(24, 10), ev, 'g', 240, 10,
                                   bridge.BridgeConfig(interpolate=False))
    assert np.array_equal(out_a['pred'], out_b['pred'], equal_nan=True)


def test_interpolate_without_a_release_still_falls_back_to_nan():
    """A refused release has no right-hand anchor, so interpolate=True must NOT invent one."""
    pred = _two_animals_crossing()
    swap_at = 120
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    rows = {'pred': pred, 'group_id': 'g'}
    ev = _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])
    cfg = bridge.BridgeConfig(min_margin=1e12, interpolate=True)  # nothing can clear this
    out, decisions = bridge.bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10, cfg)
    plan = decisions[0]
    assert plan['refused'] is True and plan['release'] is None
    assert np.isnan(out['pred'][0, 10 * plan['windows'][0]]).all()


def test_boundary_fill_map_matches_one_to_one_by_centre_distance():
    """The fill map is a Hungarian on 3D centroids at the boundary frame, one-to-one."""
    pred = np.zeros((2, 4, 3, 3))
    pred[0, :, :, 0] = 0.0
    pred[1, :, :, 0] = 100.0
    fill_pred = np.zeros((2, 4, 3, 3))
    fill_pred[0, :, :, 0] = 90.0
    fill_pred[1, :, :, 0] = 10.0
    m = bridge.boundary_fill_map(pred, fill_pred, 2, (0, 1))
    assert m == {0: 1, 1: 0}


def test_boundary_fill_map_refuses_a_boundary_it_cannot_compute():
    """A boundary the map cannot be computed at is a refusal, not a guess (section 6.4)."""
    pred = np.zeros((2, 4, 3, 3))
    pred[1, :, :] = np.nan
    fill_pred = np.zeros((2, 4, 3, 3))
    assert bridge.boundary_fill_map(pred, fill_pred, 2, (0, 1)) is None
    assert bridge.boundary_fill_map(pred, fill_pred, 99, (0,)) is None
    assert bridge.boundary_fill_map(pred, fill_pred, -1, (0,)) is None


def test_fill_keeps_a_quarantined_row_with_the_fill_pass_measurements(tmp_path):
    """A quarantined row with a fill observation is KEPT and its measurement columns are the
    fill pass's own values wholesale; a row with no fill observation stays dropped, exactly as
    the quarantine would leave it (section 6.4). Identity columns are never replaced."""
    from tailcyclenet.format import DICT_COLS, write_table
    frames = np.array([0, 0, 5, 5, 20, 20], np.int32)
    animals = np.array(['a0', 'a1', 'a0', 'a1', 'a0', 'a1'], dtype=object)
    write_table(tmp_path / 'points3d.pq', {
        'group_id': np.array(['g'] * 6, dtype=object), 'frame': frames,
        'animal_id': animals, 'bodypart': np.array(['n'] * 6, dtype=object),
        'status': np.array(['visible'] * 6, dtype=object),
        'x': np.arange(6, dtype=np.float32), 'y': np.zeros(6, np.float32),
        'z': np.zeros(6, np.float32)}, dict_cols=DICT_COLS)
    fill_table = pd.DataFrame({
        'group_id': np.array(['g'] * 4, dtype=object),
        'frame': np.array([0, 0, 5, 20], np.int32),
        'animal_id': np.array(['f0', 'f1', 'f0', 'f0'], dtype=object),
        'bodypart': np.array(['n'] * 4, dtype=object),
        'status': np.array(['visible', 'projected', 'visible', 'visible'], dtype=object),
        'x': np.array([90.0, 91.0, 92.0, 93.0], np.float32), 'y': np.zeros(4, np.float32),
        'z': np.zeros(4, np.float32)})
    segments = {0: np.arange(0, 10), 1: np.arange(10, 30)}
    plan = {'component': [0, 1], 'windows': [0, 1], 'release': 1, 'mapping': [1, 0],
            'refused': False, 'fill_map': {0: 0, 1: 1}}
    fill = {'tables': {'points3d': fill_table}, 'animal_ids': ['f0', 'f1']}
    bridge.rewrite_tables(tmp_path, 'g', ['a0', 'a1'], segments, [plan], n_frames=30, fill=fill)
    out = pd.read_parquet(tmp_path / 'points3d.pq')
    got = {(str(r.animal_id), int(r.frame)): (float(r.x), str(r.status))
           for r in out.itertuples()}
    # both quarantined a0 rows observed by fill slot f0: kept with the fill's measurements
    assert got[('a0', 0)][0] == pytest.approx(90.0)
    assert got[('a0', 5)][0] == pytest.approx(92.0)
    # the (0, a1) row is observed by fill slot f1: kept, and the fill's STATUS replaces the
    # standard's own -- status is a measurement column the fill borrows wholesale
    assert got[('a1', 0)] == (pytest.approx(91.0), 'projected')
    # the (5, a1) row has no fill observation: it stays dropped
    assert ('a1', 5) not in got
    # released rows keep the standard session's own relabel and are never filled
    assert got[('a0', 20)][0] == pytest.approx(5.0) and got[('a1', 20)][0] == pytest.approx(4.0)


def test_fill_defaults_off_and_is_byte_identical_to_before(tmp_path):
    """rewrite_tables without `fill` is byte-identical to the pre-fill contract."""
    from tailcyclenet.format import DICT_COLS, write_table
    frames = np.array([0, 0, 5, 5, 20, 20], np.int32)
    animals = np.array(['a0', 'a1', 'a0', 'a1', 'a0', 'a1'], dtype=object)
    write_table(tmp_path / 'points3d.pq', {
        'group_id': np.array(['g'] * 6, dtype=object), 'frame': frames,
        'animal_id': animals, 'bodypart': np.array(['n'] * 6, dtype=object),
        'status': np.array(['visible'] * 6, dtype=object),
        'x': np.arange(6, dtype=np.float32), 'y': np.zeros(6, np.float32),
        'z': np.zeros(6, np.float32)}, dict_cols=DICT_COLS)
    plan = {'component': [0, 1], 'windows': [0, 1], 'release': 1, 'mapping': [1, 0],
            'refused': False}
    bridge.rewrite_tables(tmp_path, 'g', ['a0', 'a1'], {0: np.arange(0, 10), 1: np.arange(10, 30)},
                          [plan], n_frames=30)
    out = pd.read_parquet(tmp_path / 'points3d.pq')
    assert set(out.frame.tolist()) == {20}
