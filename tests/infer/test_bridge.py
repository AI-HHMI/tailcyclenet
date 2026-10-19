"""The post-hoc identity bridge (`tailcyclenet/infer/bridge.py`).

The bridge is opt-in and trades misses for identity switches, so the properties that matter most
are the ones that bound the damage: it must be a no-op when there is nothing to repair, it must
only ever permute rows inside a declared component, and it must refuse rather than guess.
"""
import numpy as np
import polars as pl
import pytest

from tailcyclenet.infer import bridge


def _windows(n_windows, stride, gid='g'):
    """One window table: window `i` starts at `i * stride`."""
    return pl.DataFrame({'group_id': [gid] * n_windows,
                         'window': list(range(n_windows)),
                         'frame': [i * stride for i in range(n_windows)]})


def _events(rows, gid='g'):
    return pl.DataFrame({'group_id': [gid] * len(rows),
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


def _bridge_group(rows, windows, events, gid, n_frames, window_length, cfg):
    """Plan every episode with the production planner and apply it to the row ARRAYS.

    `bridge_session` applies the same plans to the tables (quarantine = row deletion, release =
    relabel); this array form makes the planner's consequences directly assertable. Episodes are
    applied latest first, so an earlier repair's persistence is laid down last.
    """
    cfg.validate()
    segments = bridge.owned_segments(windows, gid, n_frames, window_length)
    decisions = []
    for ep in sorted(bridge.episodes(events, gid, cfg.max_event_gap),
                     key=lambda e: e['first_frame'], reverse=True):
        plan = bridge.plan_episode(rows['pred'], segments, ep, cfg)
        if plan is None:
            decisions.append({'component': list(ep['component']), 'skipped': True})
            continue
        if plan.get('windows') or plan.get('release') is not None:
            quarantine, released = bridge._plan_frames(segments, plan, n_frames)
            component = np.asarray(plan['component'])
            for key, value in list(rows.items()):
                if not isinstance(value, np.ndarray) or value.ndim < 2:
                    continue
                changed = value.copy()
                if len(released):
                    original = value[:, released]
                    for logical, local in zip(plan['component'], plan['mapping']):
                        changed[logical, released] = original[local]
                if len(quarantine):
                    changed[component[:, None], quarantine] = np.nan
                rows = {**rows, key: changed}
        decisions.append(plan)
    return rows, list(reversed(decisions))


def _swapped(pred=None, swap_at=120):
    """`pred` (default: the crossing) with rows 0 and 1 exchanged from `swap_at` on, plus the
    duplicate-episode events the tracker would log there."""
    pred = _two_animals_crossing() if pred is None else pred
    tail = pred[:, swap_at:].copy()
    pred[0, swap_at:], pred[1, swap_at:] = tail[1], tail[0]
    return pred, _events([(swap_at, 0, 'retired_duplicate'), (swap_at, 1, 'shielded')])


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
    assert seg[3][0] == 30 and seg[3][-1] == 54, 'the final window keeps the rest of the clip'
    for frames in seg.values():
        assert np.all(np.diff(frames) == 1)


def test_episodes_gap_join_drop_single_slot_and_ignore_other_events():
    """Duplicate events join within `max_gap`; a one-slot episode, births/deaths and other groups'
    events make no episode."""
    ev = _events([(10, 2, 'retired_duplicate'), (12, 3, 'shielded'),
                  (400, 5, 'retired_duplicate'), (401, 6, 'shielded'),
                  (800, 7, 'retired_duplicate')])
    eps = bridge.episodes(ev, 'g', max_gap=64)
    assert [e['component'] for e in eps] == [(2, 3), (5, 6)]
    assert eps[0]['first_frame'] == 10 and eps[0]['last_frame'] == 12
    assert bridge.episodes(_events([(10, 2, 'born'), (11, 3, 'birth_refused'), (12, 4, 'died')]),
                           'g', 64) == []
    assert bridge.episodes(_events([(10, 2, 'retired_duplicate'), (11, 3, 'shielded')],
                                   gid='other'), 'g', 64) == []


@pytest.mark.parametrize('case', ['no events', 'identity', 'unbridgeable'])
def test_nothing_to_repair_is_a_byte_exact_no_op(case):
    """No episode, an episode whose best map is the IDENTITY (skipped, not quarantined -- the
    measured miss saving), or one with no clean anchor before it: the arrays come back untouched."""
    rows = {'pred': _two_animals_crossing(), 'group_id': 'g'}
    before = rows['pred'].copy()
    ev = {'no events': _events([]),
          'identity': _events([(100, 0, 'retired_duplicate'), (100, 1, 'shielded')]),
          'unbridgeable': _events([(1, 0, 'retired_duplicate'), (1, 1, 'shielded')])}[case]
    out, decisions = _bridge_group(rows, _windows(24, 10), ev, 'g', 240, 10,
                                   bridge.BridgeConfig())
    assert np.array_equal(out['pred'], before, equal_nan=True)
    if case == 'no events':
        assert decisions == []
    elif case == 'identity':
        assert len(decisions) == 1 and decisions[0].get('identity') is True
    else:
        assert decisions[0].get('skipped') is True


def test_a_real_swap_is_repaired_quarantined_and_permuted_in_every_field():
    """An induced exchange is undone ([1, 0]); the component is NaN before the release; boxes
    follow the pose; row 2 (outside the component) is never touched."""
    pred, ev = _swapped()
    row2_before = pred[2].copy()
    boxes = np.zeros((3, 240, 2, 4), np.float32)
    boxes[0], boxes[1], boxes[2] = 10.0, 20.0, 30.0
    out, decisions = _bridge_group({'pred': pred, 'boxes': boxes, 'group_id': 'g'},
                                   _windows(24, 10), ev, 'g', 240, 10, bridge.BridgeConfig())
    assert len(decisions) == 1
    plan = decisions[0]
    assert not plan.get('skipped') and not plan.get('refused')
    assert plan['mapping'] == [1, 0], 'the exchange must be undone, not re-applied'
    assert np.array_equal(out['pred'][2], row2_before, equal_nan=True)
    assert out['pred'][0, 230, 0, 0] == pytest.approx(230.0)
    assert out['boxes'][0, 230, 0, 0] == pytest.approx(20.0), 'boxes must follow the pose'
    assert out['boxes'][2, 230, 0, 0] == pytest.approx(30.0)

    quarantined = [w for w in plan['windows'] if w < plan['release']]
    assert quarantined, 'a repaired episode must quarantine something'
    frame = 10 * quarantined[0]
    assert np.isnan(out['pred'][0, frame]).all() and np.isnan(out['pred'][1, frame]).all()
    assert not np.isnan(out['pred'][2, frame]).any()


def _nearly_parallel_tracks(n_frames=240, k=3):
    """Two rows moving in almost the same direction: a swap here scores a real but LOW margin,
    the shape that damaged a perfectly clean clip."""
    pred = np.full((2, n_frames, k, 3), np.nan, np.float32)
    t = np.arange(n_frames, dtype=np.float32)
    for j in range(k):
        pred[0, :, j, 0] = t
        pred[0, :, j, 1] = 0.0 + j
        pred[1, :, j, 0] = t * 1.02 + 5.0
        pred[1, :, j, 1] = 2.0 + j
    pred[..., 2] = 0.0
    return pred


def test_a_low_margin_release_is_refused_and_quarantined_instead_of_guessed():
    """A margin-43 release turned idsw 0 into 2 on a clean clip: the shipped `min_margin` refuses
    (and quarantines) it while the old 1.0 applied it; an impossible margin refuses a clean swap."""
    pred, ev = _swapped(_nearly_parallel_tracks())
    out, decisions = _bridge_group({'pred': pred.copy(), 'group_id': 'g'}, _windows(24, 10), ev,
                                   'g', 240, 10, bridge.BridgeConfig())
    plan = decisions[0]
    assert plan['refused'] is True and plan['mapping'] is None
    assert np.isnan(out['pred'][0, 10 * plan['windows'][0]]).all(), 'refused means quarantined'
    _, old = _bridge_group({'pred': pred.copy(), 'group_id': 'g'}, _windows(24, 10), ev, 'g',
                           240, 10, bridge.BridgeConfig(min_margin=1.0))
    assert not old[0].get('refused') and old[0]['mapping'] is not None

    pred, ev = _swapped()
    _, decisions = _bridge_group({'pred': pred, 'group_id': 'g'}, _windows(24, 10), ev, 'g',
                                 240, 10, bridge.BridgeConfig(min_margin=1e12))
    assert decisions[0]['refused'] is True and decisions[0]['mapping'] is None


def test_backward_agrees_and_all_candidates_are_diagnostics_on_every_decision():
    """`backward_agrees` and `all_candidates` (sorted by margin, winner first) ride on the
    decision dict only, unconditionally (identity outcomes too), and never pick the winner."""
    pred, ev = _swapped(_two_animals_crossing(n_frames=400))
    _, decisions = _bridge_group({'pred': pred, 'group_id': 'g'}, _windows(40, 10), ev, 'g', 400,
                                 10, bridge.BridgeConfig())
    plan = decisions[0]
    assert plan['backward_agrees'] is True, 'a 2-row swap is an involution: it must self-agree'
    margins = [c['mean_margin'] for c in plan['all_candidates']]
    assert margins and margins == sorted(margins, reverse=True)
    assert plan['all_candidates'][0]['release'] == plan['release']
    assert plan['all_candidates'][0]['mapping'] == plan['mapping']

    _, dec_id = _bridge_group(
        {'pred': _two_animals_crossing(), 'group_id': 'g'}, _windows(24, 10),
        _events([(100, 0, 'retired_duplicate'), (100, 1, 'shielded')]), 'g', 240, 10,
        bridge.BridgeConfig())
    assert dec_id[0]['identity'] is True and dec_id[0]['backward_agrees'] is None
    assert 'all_candidates' in dec_id[0]


def test_rewrite_tables_deletes_quarantine_relabels_release_and_spares_other_groups(tmp_path):
    """A quarantine IS a row deletion and a permutation IS an `animal_id` relabel (rebuilding from
    arrays would drop `conf2d`); another group's rows stay byte-exact."""
    from tailcyclenet.format import DICT_COLS, write_table
    write_table(tmp_path / 'points3d.pq', {
        'group_id': np.array(['g'] * 6 + ['other'] * 2, dtype=object),
        'frame': np.array([0, 0, 5, 5, 20, 20, 0, 0], np.int32),
        'animal_id': np.array(['a0', 'a1'] * 4, dtype=object),
        'bodypart': np.array(['n'] * 8, dtype=object),
        'status': np.array(['visible'] * 8, dtype=object),
        'x': np.arange(8, dtype=np.float32), 'y': np.zeros(8, np.float32),
        'z': np.zeros(8, np.float32)}, dict_cols=DICT_COLS)
    plan = {'component': [0, 1], 'windows': [0, 1], 'release': 1, 'mapping': [1, 0],
            'refused': False}
    bridge.rewrite_tables(tmp_path, 'g', ['a0', 'a1'], {0: np.arange(0, 10), 1: np.arange(10, 30)},
                          [plan], n_frames=30)
    out = pl.read_parquet(tmp_path / 'points3d.pq')
    g = out.filter(pl.col('group_id').cast(pl.String) == 'g')
    assert set(g['frame'].to_list()) == {20}
    got = dict(zip(g['animal_id'].cast(pl.String).to_list(), g['x'].to_list()))
    assert got['a0'] == pytest.approx(5.0) and got['a1'] == pytest.approx(4.0)
    other = out.filter(pl.col('group_id').cast(pl.String) == 'other')
    assert other['x'].to_list() == [6.0, 7.0]
    assert other['animal_id'].cast(pl.String).to_list() == ['a0', 'a1']


def test_bridge_provenance_lists_every_lever_and_merges_into_session_toml(tmp_path):
    """Every lever is recorded unconditionally (an absent key reads as 'not used'), and is ADDED
    to the already-frozen session.toml without disturbing the run's own keys."""
    import tomllib

    import toml

    want = {f'bridge_{f}' for f in bridge.BridgeConfig.__dataclass_fields__}
    assert {k for k, _ in bridge.config_provenance(bridge.BridgeConfig())} == want
    cfg_path = tmp_path / 'session.toml'
    cfg_path.write_text(toml.dumps({
        'mode': '3d', 'units': 'mm', 'labels': 'tracked', 'names': ['a', 'b'],
        'provenance': {'run': 'r', 'checkpoint': 'c', 'anchor': 'carry'}}))
    bridge._write_bridge_provenance(tmp_path, bridge.BridgeConfig())
    with open(cfg_path, 'rb') as f:
        doc = tomllib.load(f)
    prov = doc['provenance']
    assert prov['run'] == 'r' and prov['checkpoint'] == 'c' and prov['anchor'] == 'carry'
    assert want <= set(prov) and prov['bridge_min_margin'] == bridge.BridgeConfig().min_margin
    assert doc['mode'] == '3d' and doc['names'] == ['a', 'b']


def _bridgeable_session(tmp_path):
    """A complete prediction session header, a points3d table with `slot`, and windows.pq."""
    import toml

    from tailcyclenet.format import DICT_COLS, write_table

    (tmp_path / 'session.toml').write_text(toml.dumps({
        'mode': '3d', 'names': ['n'], 'prediction_session': True, 'complete': True,
        'provenance': {'run': 'r', 'checkpoint': 'c'}}))
    frames = np.array([0, 0, 5, 5, 20, 20], np.int32)
    write_table(tmp_path / 'points3d.pq', {
        'group_id': np.array(['g'] * 6, dtype=object), 'frame': frames,
        'animal_id': np.array(['a0', 'a1'] * 3, dtype=object),
        'bodypart': np.array(['n'] * 6, dtype=object),
        'status': np.array(['visible'] * 6, dtype=object),
        'x': np.arange(6, dtype=np.float32), 'y': np.zeros(6, np.float32),
        'z': np.zeros(6, np.float32), 'slot': np.array([0, 1] * 3, np.int32)},
        dict_cols=DICT_COLS)
    write_table(tmp_path / 'windows.pq', {
        'group_id': np.array(['g', 'g'], dtype=object),
        'animal_id': np.array(['a0', 'a1'], dtype=object),
        'slot': np.array([0, 1], np.int32), 'window': np.array([0, 0], np.int32),
        'frame': np.array([0, 0], np.int32)}, dict_cols=DICT_COLS)
    return tmp_path


def test_a_release_moves_the_slot_with_the_animal_id(tmp_path):
    """A permutation relabels `animal_id`; `slot` (the animal's row) must follow it, or the two
    columns disagree on every released row."""
    path = _bridgeable_session(tmp_path)
    plan = {'component': [0, 1], 'windows': [0, 1], 'release': 1, 'mapping': [1, 0],
            'refused': False}
    bridge.rewrite_tables(path, 'g', ['a0', 'a1'], {0: np.arange(0, 10), 1: np.arange(10, 30)},
                          [plan], n_frames=30)
    out = pl.read_parquet(path / 'points3d.pq')
    pairs = set(zip(out['animal_id'].cast(pl.String).to_list(), out['slot'].to_list()))
    assert pairs == {('a0', 0), ('a1', 1)}
    got = dict(zip(out['animal_id'].cast(pl.String).to_list(), out['x'].to_list()))
    assert got['a0'] == pytest.approx(5.0)


def test_bridge_session_marks_the_session_incomplete_while_it_edits(tmp_path, monkeypatch):
    """The bridge edits a session the writer already marked complete. A crash part-way must
    leave it `complete = false` (refused by every loader); success restores `complete = true`
    and records the bridge provenance, even when there is nothing to bridge."""
    import tomllib

    path = _bridgeable_session(tmp_path)

    def read():
        """The session header as written."""
        with open(path / 'session.toml', 'rb') as f:
            return tomllib.load(f)

    monkeypatch.setattr(bridge, 'load_predictions', lambda p: ({}, {}))
    assert bridge.bridge_session(path, bridge.BridgeConfig(), 10) == {}
    doc = read()
    assert doc['complete'] is True and 'bridge_min_margin' in doc['provenance']

    def boom(*a, **k):
        """Fail the way a part-way bridge crash would."""
        raise RuntimeError('crash mid-bridge')

    monkeypatch.setattr(bridge, '_bridge_loaded', boom)
    with pytest.raises(RuntimeError, match='mid-bridge'):
        bridge.bridge_session(path, bridge.BridgeConfig(), 10)
    assert read()['complete'] is False
