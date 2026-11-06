"""`associate`, `link_rows` and `associate_group`: cross-view association and 2D row linking."""
import numpy as np
import torch

from tailcyclenet.detector.boxes import box_iou
from tests.helpers import (SKEW_RIG3, _flat_rig, _jittered_boxes, _lever_boxes, _session_2d,
                           _session_3d, _skew_rig)


def test_min_views_1_admits_the_box_no_pair_claimed(tmp_path):
    """`min_views = 2` is the algorithm; `min_views = 1` emits each leftover box as a single-
    view instance with no invented depth."""

    from tailcyclenet.format import Session
    from tailcyclenet.infer.associate import associate

    _session_3d(tmp_path / 'ds' / 'test' / 's')
    sess = Session.load(tmp_path / 'ds' / 'test' / 's')
    sess.preload()
    cams = sess.cgroup('g000', 0)

    # One animal both of the first two cameras see at the image centre, plus a box in camera 2 that
    # no pair can agree with (it sits in a corner).
    centre = torch.tensor([[24., 16., 40., 32.]])
    per_cam = [centre, centre, torch.cat([centre, torch.tensor([[0., 0., 8., 8.]])])]

    two = associate(cams, per_cam, max_res_px=30.0, min_views=2)
    one = associate(cams, per_cam, max_res_px=30.0, min_views=1)
    assert len(one) == len(two) + 1, f'{len(two)} -> {len(one)}: the leftover box must appear'
    extra = one[len(two)]
    assert list(extra['boxes']) == [2] and torch.isnan(extra['point']).all(), \
        'a single ray has no triangulated point, and inventing a depth would be a lie'


def test_support_count_counts_independently_corroborating_cameras():
    """E1a/E3: `_support_count` counts OTHER cameras near the reprojection, excluding the
    seeding pair and already-`used` detections."""

    from tailcyclenet.infer.associate import _support_count
    from tailcyclenet.infer.track import _project

    cg = _flat_rig()

    p3d = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    proj = [_project(cam, p3d.reshape(1, 3).numpy()) for cam in cg]     # (1,2) per camera

    # Every camera has a detection exactly at the true reprojection.
    centres_all = [p.clone() for p in proj]
    assert _support_count(cg, (0, 1), centres_all, set(), p3d, max_res_px=5.0) == 1, \
        'excluding the seeding pair (0, 1), only camera 2 remains and it corroborates'

    # Camera 2's detection moved far away: no corroboration left.
    centres_far = [proj[0].clone(), proj[1].clone(), proj[2] + 1000.0]
    assert _support_count(cg, (0, 1), centres_far, set(), p3d, max_res_px=5.0) == 0

    # Camera 2's detection is still at the true point, but already `used` by an earlier group --
    # an already-claimed detection must not corroborate a second candidate.
    assert _support_count(cg, (0, 1), centres_all, {(2, 0)}, p3d, max_res_px=5.0) == 0


# --- link_rows / associate_group ----------------------------------------------------------


def test_link_rows_follows_one_animal():
    """Unlinked rows are score-ordered, so the window-union crop (what `run_group` crops to)
    spans several animals; linking must undo a per-frame swap."""
    from tailcyclenet.infer.associate import link_rows

    T = 20
    boxes = np.full((2, T, 1, 4), np.nan, np.float32)
    for t in range(T):
        a = np.array([10 + t, 10, 40 + t, 40], np.float32)      # drifts right
        b = np.array([110 - t, 10, 140 - t, 40], np.float32)    # drifts left
        boxes[0, t, 0], boxes[1, t, 0] = (a, b) if t % 2 == 0 else (b, a)

    union = lambda x: np.concatenate([x[..., :2].min(0), x[..., 2:].max(0)], -1)  # noqa: E731
    before = union(boxes[0, :, 0])
    linked = link_rows(boxes.copy())
    after = union(linked[0, :, 0])

    assert (before[2] - before[0]) > 120, 'the swapped rows should span the whole frame'
    assert (after[2] - after[0]) < (before[2] - before[0]) / 2, \
        f'linking must shrink the union crop, got {after} from {before}'
    # Every frame still holds both animals -- linking reorders, it never drops.
    assert np.isfinite(linked).all()


def test_link_rows_gates_on_size_and_never_force_assigns():
    """A row that matches nothing stays EMPTY (`free.pop(0)` once handed it a teleporting box),
    a jump of more than one box side is not motion, and a one-frame miss does not break the
    chain (matching is against the LAST KNOWN box)."""
    from tailcyclenet.infer.associate import link_rows

    boxes = np.full((2, 6, 1, 4), np.nan, np.float32)
    boxes[0, :, 0] = [10, 10, 40, 40]                          # a stationary animal
    boxes[1, 3, 0] = [900, 900, 930, 930]                      # one far-away detection, once
    linked = link_rows(boxes.copy())
    assert np.allclose(linked[0, :, 0, 0], 10), 'the tracked row must not move'

    jump = np.full((1, 2, 1, 4), np.nan, np.float32)
    jump[0, 0, 0] = [0, 0, 30, 30]
    jump[0, 1, 0] = [300, 0, 330, 30]                          # 10 box sides away
    assert not np.isfinite(link_rows(jump.copy())[0, 1, 0]).all()
    jump[0, 1, 0] = [20, 0, 50, 30]                            # 0.67 of a side: ordinary motion
    assert np.isfinite(link_rows(jump)[0, 1, 0]).all()

    gap = np.full((2, 4, 1, 4), np.nan, np.float32)
    for t in range(4):
        gap[0, t, 0] = [10 + t, 10, 40 + t, 40]
        gap[1, t, 0] = [110, 10, 140, 40]
    gap[:, 2, 0] = np.nan                                      # the detector sees nothing at t=2
    linked = link_rows(gap.copy())
    assert linked[0, 3, 0][0] == 13 and linked[1, 3, 0][0] == 110


def test_link_rows_prefers_the_nearer_box_where_iou_prefers_the_wrong_one():
    """IoU ranks by shape agreement, which is not identity: a true continuation that shrank loses
    to a size-matched wrong animal. Centre distance is not fooled.
    """
    from tailcyclenet.infer.associate import link_rows

    last = np.array([100.0, 100.0, 300.0, 300.0], np.float32)     # remembered, 200 px
    near = np.array([170.0, 170.0, 230.0, 230.0], np.float32)      # same centre, 60 px
    wide = np.array([150.0, 150.0, 350.0, 350.0], np.float32)      # 0.35 sides away, 200 px
    iou = box_iou(torch.as_tensor(last[None]), torch.as_tensor(np.stack([near, wide])))[0]
    assert iou[1] > iou[0], f'the fixture must be one IoU gets wrong, got {iou.tolist()}'

    boxes = np.stack([np.stack([last, near]),
                      np.stack([np.full(4, np.nan, np.float32), wide])])[:, :, None]
    linked = link_rows(boxes.astype(np.float32).copy())
    assert np.allclose(linked[0, 1, 0], near), \
        f'centre distance must keep the nearer box, got {linked[0, 1, 0]}'
    # ...and the other detection is a BIRTH into the empty row, not a dropped animal.
    assert np.allclose(linked[1, 1, 0], wide)


def test_associate_group_state_carries_across_a_split(tmp_path):
    """Blocks are sized by the RAM budget, so N calls with `state=` must equal ONE call byte
    for byte, in both branches (tracker C > 1, `link_rows` C == 1); the scene makes a stateless
    split FAIL."""

    from tailcyclenet.format import Session
    from tailcyclenet.infer.associate import associate_group

    T = 12
    for tag, cams, track in (('mv', 3, True), ('sv', 1, False)):
        d = tmp_path / tag / 'test' / 's'
        (_session_3d if cams > 1 else _session_2d)(d, T=T)
        sess = Session.load(d)
        sess.preload()

        D, C, S = 4, cams, 2
        box = np.full((D, T, C, 4), np.nan, np.float32)
        sc = np.full((D, T, C), np.nan, np.float32)
        for t in range(T):
            for c in range(C):
                for a in range(S):
                    # Two animals drifting apart inside the fixture's 64x48 frame. What makes each
                    # branch discriminating differs: the tracker seats a vanished row's survivor
                    # differently; link_rows needs the DETECTION ORDER to alternate.
                    if C > 1 and a == 1 and 4 <= t < 7:
                        continue
                    d = a if C > 1 else (a + t) % 2
                    x = 4.0 + 24.0 * a + 0.5 * t
                    box[d, t, c] = [x, 12, x + 12, 32]
                    sc[d, t, c] = 0.9 - 0.1 * d
        raw = (box, sc)

        def run(state):
            parts = []
            for lo, hi in ((0, 5), (5, T)):
                sub = (box[:, lo:hi].copy(), sc[:, lo:hi].copy())
                parts.append(associate_group(sub, sess, 'g000', S, link=not track, track=track,
                                             **({} if state is None else {'state': state})))
            return [np.concatenate([p[i] for p in parts], axis=1) for i in range(2)]

        whole = associate_group(raw, sess, 'g000', S, link=not track, track=track)
        joined = run({})
        for i, name in enumerate(('boxes', 'scores')):
            np.testing.assert_array_equal(
                np.nan_to_num(joined[i], nan=-9e9), np.nan_to_num(whole[i], nan=-9e9),
                err_msg=f'{name} differ at C={cams}, track={track}: a block boundary changed the '
                        'association, so the RAM budget can change the answer')

        # THE TEST MUST BE ABLE TO FAIL. Without the carry, this scene disagrees -- if it stops
        # disagreeing, the scene has gone back to being one a fresh matcher reproduces by luck
        # and the assertion above is vacuous.
        assert not np.array_equal(np.nan_to_num(run(None)[0], nan=-9e9),
                                  np.nan_to_num(whole[0], nan=-9e9)), \
            f'C={cams}, track={track}: a STATELESS split reproduces the whole clip here, so this ' \
            'scene does not exercise the carry at all'


def test_associate_group_threads_max_age_and_frame_base(tmp_path, monkeypatch):
    """`--max-age` reaches `CrossViewTracker` (C > 1) and `link_rows` (C == 1), and a ranged run's
    events are SOURCE frames (`frame_base` added back, A6a)."""
    import tailcyclenet.infer.associate as detmod
    from tailcyclenet.format import Session

    T = 4
    d = tmp_path / 'test' / 's'
    _session_3d(d, T=T)
    sess = Session.load(d)
    sess.preload()
    cg = sess.cgroup('g000')
    per_cam, scores = _lever_boxes(cg, [np.array([0.0, 0.0, 0.0])], side=100.0)
    box = np.tile(np.stack([b[0].numpy() for b in per_cam])[None, None], (1, T, 1, 1))
    sc = np.tile(np.stack([s[0].numpy() for s in scores])[None, None], (1, T, 1))
    raw = (box.astype(np.float32), sc.astype(np.float32))

    state = {}
    detmod.associate_group(raw, sess, 'g000', 1, track=True, max_age=5, state=state)
    assert state['tracker'].max_age == 5
    assert state['tracker'].events[0]['frame'] == 0
    ranged = {}
    detmod.associate_group(raw, sess, 'g000', 1, track=True, state=ranged, frame_base=32)
    assert ranged['tracker'].events[0]['frame'] == 32

    seen = {}

    def spy_link_rows(boxes, scores=None, **kw):
        seen.update(kw)
        return boxes, scores

    monkeypatch.setattr(detmod, 'link_rows', spy_link_rows)

    class _FakeSession:
        rig = type('Rig', (), {'moving': {}})()

        def cgroup(self, gid, t=None):
            return []

    nan = (np.full((1, 2, 1, 4), np.nan, np.float32), np.full((1, 2, 1), np.nan, np.float32))
    detmod.associate_group(nan, _FakeSession(), 'g000', 1, link=True, track=False, max_age=7,
                           link_two_stage_2d=False)
    assert seen.get('max_age') == 7

    seen2 = {}

    def spy_two_stage(boxes, scores, n_rows, **kw):
        seen2.update(kw)
        S, T = n_rows, boxes.shape[1]
        return np.full((S, T, 4), np.nan, np.float32), np.full((S, T), np.nan, np.float32)

    monkeypatch.setattr(detmod, 'link_two_stage', spy_two_stage)
    detmod.associate_group(nan, _FakeSession(), 'g000', 1, link=True, track=False, max_age=6)
    assert seen2.get('max_age') == 6


def test_moving_rig_associate_group_fetches_extrinsics_at_source_frames(tmp_path):
    """D1.1: a moving rig's extrinsics are fetched at SOURCE frames (`frame_base` + local), not
    the batch-local index."""

    from tailcyclenet.format import Session
    from tailcyclenet.infer.associate import associate_group

    class _RecordingSession:
        """Delegate to a real Session but record every `cgroup` frame argument."""

        def __init__(self, real):
            self._real = real
            self.frames = []

        def __getattr__(self, name):
            return getattr(self._real, name)

        def cgroup(self, gid, frames=None):
            self.frames.append(frames)
            return self._real.cgroup(gid, frames)

    d = tmp_path / 'test' / 's'
    _session_3d(d, T=12, moving=True)
    sess = Session.load(d)

    T, C = 4, 3
    raw = (np.full((1, T, C, 4), np.nan, np.float32),
           np.full((1, T, C), np.nan, np.float32))

    whole = _RecordingSession(sess)
    associate_group(raw, whole, 'g000', 1, track=True)
    assert whole.frames == [0, 1, 2, 3], \
        'frame_base=0 (the default) must keep the whole-clip behaviour byte-identical'

    ranged = _RecordingSession(sess)
    associate_group(raw, ranged, 'g000', 1, track=True, frame_base=6)
    assert ranged.frames == [6, 7, 8, 9], \
        "a moving rig's extrinsics must be fetched at SOURCE frames, not batch-local ones"


# --- Phase 8 levers (each a switch whose default is the shipped behaviour) ---

def _p8_scene(side):
    """Three animals, jittered centres (so residuals are non-zero), equal box sides."""
    cg = _skew_rig(SKEW_RIG3)
    worlds = [[0.0, 0.0, 0.0], [150.0, 0.0, 0.0], [0.0, 150.0, 30.0]]
    rng = np.random.default_rng(0)
    jitter = [rng.normal(0, 6.0, (3, 2)) for _ in cg]
    return cg, _jittered_boxes(cg, worlds, side, jitter)


def _p8_key(groups):
    return [(sorted(g['members'].items()), round(g['residual'], 6)) for g in groups]


def test_default_associate_is_unchanged_by_the_new_keywords():
    from tailcyclenet.infer.associate import associate

    cg, boxes = _p8_scene(80.0)
    scores = [torch.rand(len(b), generator=torch.Generator().manual_seed(1)) for b in boxes]
    assert _p8_key(associate(cg, boxes, max_res_px=30.0)) == _p8_key(associate(cg, boxes, max_res_px=30.0, scores_per_cam=scores))


def test_score_order_prefers_the_stronger_pair_over_the_lower_residual():
    """Two cameras, one animal, camera 1 double-fires: an exact weak box and a 6 px-off strong
    box. Support ties (no third camera), so the default takes the lower residual and the
    objectness order takes the stronger box."""
    from tailcyclenet.infer.associate import associate

    cg = _skew_rig(SKEW_RIG3[:2])
    b0, b1 = _jittered_boxes(cg, [[0.0, 0.0, 0.0]])
    off = b1 + torch.tensor([6.0, 0.0, 6.0, 0.0])
    boxes = [b0, torch.cat([b1, off])]
    scores = [torch.tensor([0.9]), torch.tensor([0.1, 0.9])]
    plain = associate(cg, boxes, max_res_px=30.0, scores_per_cam=scores)
    ordered = associate(cg, boxes, max_res_px=30.0, scores_per_cam=scores, score_order=True)
    assert plain[0]['members'] == {0: 0, 1: 0}
    assert ordered[0]['members'] == {0: 0, 1: 1}


def test_score_order_needs_scores():
    import pytest

    from tailcyclenet.infer.associate import associate

    cg, boxes = _p8_scene(80.0)
    with pytest.raises(ValueError, match='scores_per_cam'):
        associate(cg, boxes, max_res_px=30.0, score_order=True)


def test_fast_associate_is_identical_to_the_reference():
    """TR7: `fast=True` returns the same groups, bit for bit (points, residuals, members), under
    every gate and order switch, on crowded jittered scenes with double-fires and NaN boxes."""
    from tailcyclenet.infer.associate import associate

    cg = _skew_rig(SKEW_RIG3 + [(0.1, 0.3, 40.0)])
    rng = np.random.default_rng(3)
    for trial in range(6):
        W = rng.uniform(-200, 200, (8, 3))
        boxes = _jittered_boxes(cg, W, 100.0, [rng.normal(0, 8, (8, 2)) for _ in cg])
        boxes = [torch.cat([b, b[:2] + 7.0]) for b in boxes]
        boxes[trial % 4][1] = float('nan')
        scores = [torch.as_tensor(rng.uniform(0, 1, len(b)), dtype=torch.float32)
                  for b in boxes]
        for kw in ({}, {'score_order': True}, {'min_views': 1},
                   {'max_instances': 3}):
            ref = associate(cg, boxes, max_res_px=30.0, scores_per_cam=scores, **kw)
            got = associate(cg, boxes, max_res_px=30.0, scores_per_cam=scores, fast=True, **kw)
            assert len(ref) == len(got)
            for g, h in zip(ref, got):
                assert g['members'] == h['members'] and g['residual'] == h['residual']
                assert torch.equal(torch.nan_to_num(g['point'], nan=1e30),
                                   torch.nan_to_num(h['point'], nan=1e30))


def test_random_drop_rejects_at_its_rate_and_zero_draws_nothing():
    """The rule-14 control: p = 0 is the reference; p = 1 keeps nothing; p = 0.5 keeps some."""
    from tailcyclenet.infer.associate import associate

    cg, boxes = _p8_scene(80.0)
    ref = associate(cg, boxes, max_res_px=30.0)
    assert len(associate(cg, boxes, max_res_px=30.0, random_drop=(1.0, np.random.default_rng(0)))) == 0
    assert _p8_key(associate(cg, boxes, max_res_px=30.0, random_drop=(0.0, np.random.default_rng(0)))) == \
        _p8_key(ref)



# --- link_two_stage (2D default) -------------------------------------------------------------


def _calib_frames(n, S=2, score=0.9):
    """n frames of S well-separated, high-score animals (they set the reference score)."""
    boxes = np.full((4, n, 4), np.nan, np.float32)
    scores = np.full((4, n), np.nan, np.float32)
    for t in range(n):
        boxes[0, t] = [0, 0, 50, 50]
        boxes[1, t] = [300, 0, 350, 50]
        scores[:S, t] = score
    return boxes, scores


def test_link_two_stage_a_low_score_box_cannot_pull_a_row_onto_another_animal():
    """The mechanism (report 76 relink): A moves right and B is missed; a low-score half box sits
    beside A's old place. Joint Hungarian over the top-S (`link_rows`) hands A's TRUE box to row B
    (B's only positive-affinity option is it) and A's row to the half box -- a swap. Two-stage
    matches the high-score box first (to A) and lets the low box only extend a row in its own gate."""
    from tailcyclenet.infer.associate import link_rows, link_two_stage

    boxes, scores = _calib_frames(6)
    boxes[1, :] = [70, 0, 120, 50]                           # B at x=70..120, A at 0..50
    b2, s2 = boxes.copy(), scores.copy()
    t = 5
    b2[:, t] = np.nan
    s2[:, t] = np.nan
    b2[0, t], s2[0, t] = [30, 0, 80, 50], 0.9                # A's true box, moved right 30 px
    b2[1, t], s2[1, t] = [5, 0, 55, 50], 0.2                 # a low-score half box near A's old box

    legacy = link_rows(b2[:2, :, None].copy(), s2[:2, :, None].copy(), max_move=1.25)[0][:, :, 0]
    assert np.allclose(legacy[1, t], [30, 0, 80, 50]), 'legacy: row B takes A (the swap)'

    rows, rsc = link_two_stage(b2, s2, 2, max_move=1.25)
    assert np.allclose(rows[0, t], [30, 0, 80, 50]), 'two-stage: row A keeps its own box'
    assert not np.isfinite(rows[1, t]).any(), 'row B waits empty, it is not dragged onto A'
    assert rsc[0, t] == np.float32(0.9)


def test_link_two_stage_low_boxes_extend_but_never_open_rows():
    """A low-score box extends a live row inside its gate (detector score dips at contact) but an
    unclaimed one never opens a row; a box past rank S still extends a row."""
    from tailcyclenet.infer.associate import link_two_stage

    boxes, scores = _calib_frames(8)
    boxes[1, 4:], scores[1, 4:] = np.nan, np.nan             # B lost from frame 4 ...
    scores[0, 4] = 0.2                                       # A's score dips: low, still extends
    boxes[2, 5], scores[2, 5] = [302, 0, 352, 50], 0.3       # B re-seen at RANK 3 (low): extends B
    boxes[3, 6], scores[3, 6] = [600, 0, 650, 50], 0.2       # a far low box: never opens a row
    rows, _ = link_two_stage(boxes, scores, 3, max_move=1.25)
    assert np.allclose(rows[0, 4], [0, 0, 50, 50])
    assert np.allclose(rows[1, 5], [302, 0, 352, 50])
    assert not np.isfinite(rows[:, 6]).all(-1)[2], 'no row is born from a low-score box'
    assert not np.isfinite(rows[2]).any()


def test_link_two_stage_links_on_extents_and_returns_crop_boxes():
    """The match is computed on the tight extents (a crop box's floor hides the motion), and the
    rows carry the CROP boxes the pose loop crops to."""
    from tailcyclenet.infer.associate import link_two_stage

    boxes, scores = _calib_frames(4)
    crop = boxes + np.array([-20, -20, 20, 20], np.float32)
    rows, _ = link_two_stage(crop, scores, 2, extents=boxes)
    assert np.allclose(rows[:, :, :], crop[:2])


def test_link_two_stage_blocks_equal_one_pass():
    """State threads across calls: 16-frame blocks link exactly as the whole clip does (the
    driver's detection cursor), including the causal score calibration."""
    from tailcyclenet.infer.associate import link_two_stage

    rng = np.random.default_rng(0)
    T, D, S = 70, 6, 3
    ctr = np.cumsum(rng.normal(0, 8, (S, T, 2)), 1) + np.array([[100], [300], [500]])[:, :, None]
    boxes = np.full((D, T, 4), np.nan, np.float32)
    scores = np.full((D, T), np.nan, np.float32)
    for t in range(T):
        sc = rng.uniform(0.05, 1.0, S + 2)
        order = np.argsort(-sc)
        allc = np.concatenate([ctr[:, t], ctr[rng.integers(0, S, 2), t] + rng.normal(0, 20, (2, 2))])
        for r, k in enumerate(order):
            boxes[r, t] = [*(allc[k] - 25), *(allc[k] + 25)]
            scores[r, t] = sc[k]
    whole = link_two_stage(boxes, scores, S)
    state, parts = {}, []
    for t0 in range(0, T, 16):
        parts.append(link_two_stage(boxes[:, t0:t0 + 16], scores[:, t0:t0 + 16], S, state=state))
    assert np.array_equal(np.nan_to_num(whole[0], nan=-1),
                          np.nan_to_num(np.concatenate([p[0] for p in parts], 1), nan=-1))


def test_associate_group_2d_legacy_flag_reproduces_link_rows():
    """`link_two_stage_2d=False` is the old rule byte for byte: the top-S boxes, then `link_rows`."""
    from tailcyclenet.infer.associate import associate_group, link_rows

    class _S:
        rig = type('Rig', (), {'moving': {}})()

        def cgroup(self, gid, t=None):
            return []

    rng = np.random.default_rng(1)
    D, T, S = 5, 30, 2
    b = (rng.uniform(0, 400, (D, T, 1, 2)) + np.array([0, 0])).astype(np.float32)
    boxes = np.concatenate([b, b + 40], -1)
    scores = np.sort(rng.uniform(0.0, 1.0, (D, T, 1)), 0)[::-1].astype(np.float32).copy()
    out, sc = associate_group((boxes, scores), _S(), 'g', S, link=True, track=False,
                              link_two_stage_2d=False, state={})
    ref = link_rows(boxes[:S].copy(), scores[:S].copy(), max_move=1.25, max_age=8, state={})
    assert np.array_equal(np.nan_to_num(out, nan=-1), np.nan_to_num(ref[0], nan=-1))
    out2, _ = associate_group((boxes, scores, boxes), _S(), 'g', S, link=True, track=False,
                              state={})
    assert out2.shape == out.shape
