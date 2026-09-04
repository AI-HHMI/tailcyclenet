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
