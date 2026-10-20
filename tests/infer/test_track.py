"""`CrossViewTracker`: slots, ages, the duplicate backstop and identity events."""
import numpy as np
import torch

from tests.helpers import SKEW_RIG3, _flat_rig, _jittered_boxes, _lever_boxes, _skew_rig

# --- cross-view tracker ------------------------------------------------------------------


def _lever_rig(specs):
    """A NON-DEGENERATE rig: every camera at its own tilt and height, so rays through different
    animals are SKEW (a coplanar rig triangulates any pair to residual ~0). specs entries are
    (rvec_x, rvec_y, tvec_y)."""
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.format import Rig

    cams = []
    for i, (rx, ry, ty) in enumerate(specs):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([rx, ry, 0.0]),
                     tvec=np.array([0.0, ty, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    return Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
               moving=dict.fromkeys(names, False),
               calibrated=dict.fromkeys(names, True)).tracktail()


def _lost_in_one_camera():
    """Animal A seen in cameras 0 and 1, MISSED in camera 2 where the only detection is animal
    B. Outputs: (cg, per_cam, scores, A)."""
    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a = np.array([0.0, 0.0, 0.0], np.float32)
    b = np.array([120.0, 0.0, 0.0], np.float32)
    per_cam, scores = _lever_boxes(cg, [a, b])
    per_cam[2], scores[2] = per_cam[2][1:], torch.ones(1)
    return cg, per_cam, scores, a


def _dup_scene():
    """One animal detected twice in every camera (two identical box sets)."""
    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    duplicate = [torch.cat([b, b.clone()]) for b in per_cam]
    return cg, point, per_cam, scores, duplicate, [torch.ones(2) for _ in scores]


def test_duplicate_backstop_retires_shields_and_logs_without_changing_output():
    """After `duplicate_persist` in-band frames the backstop retires the weaker duplicate, its
    freed slot does not re-birth it (the immortal duplicate of 3dpop Seq59, report 53), and
    `tracker.events` records born/retired/shielded/birth_refused -- recording changes nothing."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, _, _, _, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    outs = [tr.step(cg, duplicate, dup_scores) for _ in range(6)]
    assert len(tr.targets) == 1, 'the retired duplicate must stay retired'
    rows = [int(np.isfinite(b).all(-1).any(-1).sum()) for b, _, _ in outs[1:]]
    assert rows == [1] * 5, 'one row on the animal from the retirement on'
    first = int(np.isfinite(outs[1][0]).all(-1).any(-1).argmax())
    assert int(np.isfinite(outs[-1][0]).all(-1).any(-1).argmax()) == first, \
        'the surviving row must keep the animal'

    by_event = {}
    for e in tr.events:
        assert set(e) == {'frame', 'slot', 'event', 'detail'}, e
        assert 0 <= e['frame'] < 6, e
        by_event.setdefault(e['event'], []).append(e)
    assert len(by_event.get('born', [])) == 2
    retired = by_event.get('retired_duplicate', [])
    assert len(retired) == 1
    shielded = by_event.get('shielded', [])
    assert len(shielded) == 1 and shielded[0]['slot'] == retired[0]['detail']['winner']
    assert retired[0]['detail']['persisted'] >= 2
    assert retired[0]['detail']['loser_age'] >= 0 and retired[0]['detail']['winner_age'] >= 0
    assert by_event.get('birth_refused')

    tr2 = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for t, (boxes, sc, claimed) in enumerate(outs):
        b2, s2, c2 = tr2.step(cg, duplicate, dup_scores)
        np.testing.assert_array_equal(np.nan_to_num(b2, nan=-9e9),
                                      np.nan_to_num(boxes, nan=-9e9), err_msg=f'boxes at t={t}')
        np.testing.assert_array_equal(np.nan_to_num(s2, nan=-9e9),
                                      np.nan_to_num(sc, nan=-9e9), err_msg=f'scores at t={t}')
        np.testing.assert_array_equal(c2, claimed, err_msg=f'claimed at t={t}')


def test_identity_events_record_a_slot_dying_of_old_age():
    """`died` carries the age that crossed `max_age`: without it a starved slot reads like one
    still alive but unclaimed."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    tr = CrossViewTracker(1, max_res_px=30.0, max_age=3)
    tr.step(cg, per_cam, scores)
    assert [e['event'] for e in tr.events] == ['born']

    empty = [torch.zeros((0, 4)) for _ in cg]
    empty_scores = [s.new_zeros(0) for s in empty]
    for _ in range(6):
        tr.step(cg, empty, empty_scores)
    died = [e for e in tr.events if e['event'] == 'died']
    assert len(died) == 1 and died[0]['slot'] == 0, tr.events
    assert died[0]['detail']['age'] > tr.max_age, died[0]
    assert not tr.targets


def test_shield_survives_the_winner_vacating_for_one_frame():
    """The shield is anchored to the ANIMAL, not the winner's this-frame claims: a one-frame
    vacate let the duplicate re-seat (~3 refires per 160 frames, 3dpop Seq59)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, point, per_cam, scores, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for _ in range(2):
        tr.step(cg, duplicate, dup_scores)
    assert len(tr.targets) == 1 and tr._dup_anchor, 'a retirement must anchor its winner'
    anchored = dict(tr._dup_anchor)
    far = torch.tensor([600.0, 0.0, 0.0])
    far_cam, far_scores = _lever_boxes(cg, [far.numpy()], side=100.0)
    tr.step(cg, far_cam, far_scores)          # the winner's only work this frame is elsewhere
    assert tr._dup_anchor.keys() == anchored.keys(), 'the anchor must outlive a one-frame vacate'
    for _ in range(4):
        boxes, _, _ = tr.step(cg, duplicate, dup_scores)
    seated = [t['point'] for t in tr.targets.values()
              if float(torch.linalg.norm(t['point'] - point)) < 300.0]
    assert len(seated) == 1, 'the duplicate must not re-seat on the anchored animal'
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 1


def test_shield_dies_when_the_winners_own_target_dies():
    """A winner that ages out must take its shield with it, or the next frame's unrelated birth
    at that spot is wrongly refused."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, point, per_cam, scores, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2, max_age=2)
    for _ in range(2):
        tr.step(cg, duplicate, dup_scores)
    assert tr._dup_shield, 'a retirement must shield its winner'
    empty = [b.new_zeros((0, 4)) for b in per_cam]
    empty_scores = [s.new_zeros(0) for s in scores]
    for _ in range(4):                        # starve the winner past max_age: it must die
        tr.step(cg, empty, empty_scores)
    assert not tr.targets, 'the winner must have aged out'
    assert not tr._dup_shield, 'the shield must die with the target it protects'
    boxes, _, _ = tr.step(cg, per_cam, scores)  # an unrelated single-set birth at the same spot
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 1, \
        'a dead shield must not refuse an unrelated birth'


def test_duplicate_persistence_does_not_survive_a_members_death():
    """A `_dup_contact` counter must not resume across a slot's death, or a rebirth on a new
    animal retires early."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, point, per_cam, scores, duplicate, dup_scores = _dup_scene()
    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=5, max_age=2)
    for _ in range(3):                        # 3 of 5 in-band frames: short of retirement
        tr.step(cg, duplicate, dup_scores)
    key = frozenset(tr.targets)
    assert tr._dup_contact.get(key, 0) == 3
    empty = [b.new_zeros((0, 4)) for b in per_cam]
    empty_scores = [s.new_zeros(0) for s in scores]
    for _ in range(4):                        # both slots starve past max_age and die together
        tr.step(cg, empty, empty_scores)
    assert not tr.targets
    assert not tr._dup_contact, 'a dead pair must not leave a stale counter behind'
    for _ in range(3):                        # rebirth: must count from 0, not resume at 3
        boxes, _, _ = tr.step(cg, duplicate, dup_scores)
    assert int(np.isfinite(boxes).all(-1).any(-1).sum()) == 2, \
        'a rebirth must not inherit a stale persistence count and retire early'


def test_refused_duplicate_does_not_consume_the_free_slot():
    """A6b: a refused duplicate birth must retry the SAME free slot, or a genuine group behind
    it is crowded out of the frame."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    pointA = np.array([0.0, 0.0, 0.0])
    pointB = np.array([250.0, 0.0, 0.0])    # far from A: a distinct animal, not a duplicate
    per_cam, scores = _lever_boxes(cg, [pointA, pointB], side=100.0)
    dupA = [torch.cat([b[:1], b[:1].clone()]) for b in per_cam]      # A twice per camera
    dup_scores = [torch.ones(2) for _ in scores]
    genuine = [b[1:2] for b in per_cam]                              # B, one box per camera

    tr = CrossViewTracker(2, max_res_px=30.0, duplicate_persist=2)
    for _ in range(2):                       # both slots seat at A; the backstop retires one
        boxes, _, _ = tr.step(cg, dupA, dup_scores)
    assert len(tr.targets) == 1, 'the backstop must retire exactly one slot here'
    survivor = int(np.isfinite(boxes).all(-1).any(-1).argmax())

    frames23 = [torch.cat([d, g]) for d, g in zip(dupA, genuine)]    # [A, A, B] per camera
    boxes3, _, _ = tr.step(cg, frames23, [torch.ones(3) for _ in scores])
    other = 1 - survivor
    assert np.isfinite(boxes3[other]).all(-1).any(), \
        'the genuine birth must seat in the freed slot -- the refusal must not consume it'
    assert any(e['event'] == 'birth_refused' for e in tr.events), \
        'the duplicate refusals must still be recorded'


def test_age_stickiness_resolves_two_slots_on_one_group():
    """Two slots stranded on one animal must not flip-flop on near-ties: the fresher keeps the
    group, the stale one starves (3dpop Seq59)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    per_cam, scores = _lever_boxes(cg, [point.numpy()], side=100.0)
    tr = CrossViewTracker(2, max_res_px=30.0, max_age=4)
    # Seat both targets on the same animal, one stale (age 2) and one fresh (age 0).
    tr.targets[0] = {'point': point.clone(), 'age': 0}
    tr.targets[1] = {'point': point.clone(), 'age': 2}
    rows = []
    for _ in range(8):
        boxes, _, _ = tr.step(cg, per_cam, scores)
        rows.append(boxes)
    assert all(np.isfinite(boxes).all(-1).any(-1).sum() == 1 for boxes in rows), \
        'two slots must never both claim one group'
    assert len(tr.targets) == 1, 'the stale duplicate must starve out'
    assert np.isfinite(rows[-1]).all(-1).any(-1).sum() == 1


def test_age_sticky_term_never_inverts_a_real_match():
    """Past `AGE_STICKY_CAP` the packed cost `affinity * 16 + CAP - age` went negative and a
    real high-affinity match read as unavailable (only when `max_age` > the cap)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    point = torch.tensor([0.0, 0.0, 0.0])
    # An 80mm offset against a 200px box side is a real, well-supported match (well inside the
    # `max_move` gate) but not a perfect one -- affinity comfortably short of 1.0, exactly the
    # regime the inverted sign discards.
    per_cam, scores = _lever_boxes(cg, [(point + torch.tensor([80.0, 0.0, 0.0])).numpy()],
                                   side=200.0)
    tr = CrossViewTracker(1, max_res_px=30.0, max_age=30)
    tr.targets[0] = {'point': point.clone(), 'age': 24}
    out, sc, claimed = tr.step(cg, per_cam, scores)
    assert np.isfinite(out).all(-1).any(), \
        'a stale (age 24) slot lost its own high-affinity candidate to nothing'
    assert tr.targets[0]['age'] == 0, 'a real match must reset age, not leave the slot starving'


def test_a_pointless_target_expires_instead_of_burning_a_slot_forever():
    """A `--min-views 1` target has an all-NaN point the matcher never sees; it must still age
    out."""
    import torch

    from tailcyclenet.infer.track import CrossViewTracker

    tr = CrossViewTracker(2, max_age=3)
    tr.targets[0] = {'point': torch.full((3,), float('nan')), 'age': 0}
    assert 0 in tr.targets

    # No detections at all: the pointless target must still age out, freeing its slot.
    for _ in range(tr.max_age + 1):
        tr.step([], [], [])
    assert 0 not in tr.targets, 'a target the matcher can never see must still be able to expire'

    # A target WITH a point is unaffected by this path -- it ages through the normal loop.
    tr2 = CrossViewTracker(2, max_age=3)
    tr2.targets[0] = {'point': torch.zeros(3), 'age': 0}
    tr2.step([], [], [])
    assert tr2.targets[0]['age'] == 1, 'the finite-point target must age exactly once per frame'


def test_min_views_1_single_view_birth_holds_no_slot():
    """A single-view group (NaN point by design) is written but never seated: seating it
    rotated one animal through every slot and then lost it (T1)."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    box = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
    boxes_per_cam = [box] + [torch.zeros((0, 4)) for _ in cg[1:]]
    scores_per_cam = [torch.ones(1)] + [torch.zeros(0) for _ in cg[1:]]
    tr = CrossViewTracker(4, max_res_px=30.0, max_age=5, min_views=1)
    for _ in range(12):
        out, _, claimed = tr.step(cg, boxes_per_cam, scores_per_cam)
        rows = [s for s in range(4) if np.isfinite(out[s]).all(-1).any()]
        assert rows == [0]
        torch.testing.assert_close(torch.as_tensor(out[0, 0]), box[0])
        assert claimed[0, 0] == 0
        assert not tr.targets
    assert not [e for e in tr.events if e['event'] == 'born']


def test_joint_association_refuses_a_binding_the_per_camera_hungarian_accepts():
    """Joint association matches whole cross-view groups, so a slot holding A cannot bind B's
    box in the camera that missed A."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg, per_cam, scores, a = _lost_in_one_camera()

    tr = CrossViewTracker(1, max_res_px=30.0)
    tr.targets[0] = {'point': torch.as_tensor(a), 'age': 0}
    out, _, claimed = tr.step(cg, per_cam, scores)
    assert claimed[0].tolist() == [0, 0, -1], 'the joint decision must not reach into camera 2'
    np.testing.assert_allclose(tr.targets[0]['point'].numpy(), a, atol=1e-3)
    assert tr.targets[0]['age'] == 0, 'a slot that matched a group has evidence and must not age'


def test_joint_association_still_births_ages_and_resumes_like_the_shipped_path():
    """Two crossing animals with the score order swapped every frame: births on frame 0, no row
    jumps, a one-frame miss ages without dropping, the SAME slots resume."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _lever_rig([(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)])
    a, b = np.array([-80.0, 0.0, 0.0]), np.array([80.0, 0.0, 0.0])
    tr = CrossViewTracker(2, max_res_px=30.0)
    rows = []
    for t in range(12):
        w = [a + [12.0 * t, 0, 0], b - [12.0 * t, 0, 0]]
        per_cam, scores = _lever_boxes(cg, w if t % 2 == 0 else w[::-1], side=40.0)
        rows.append(tr.step(cg, per_cam, scores)[0])

    assert all(np.isfinite(r).all(-1).any(-1).sum() == 2 for r in rows), 'an animal was lost'
    for s in (0, 1):
        cx = np.array([r[s, 0, [0, 2]].mean() for r in rows])
        assert np.abs(np.diff(cx)).max() < 30.0, f'row {s} jumped: {np.diff(cx)}'

    empty = [torch.zeros((0, 4)) for _ in cg], [torch.zeros((0,)) for _ in cg]
    out, _, _ = tr.step(cg, *empty)
    assert not np.isfinite(out).any() and len(tr.targets) == 2, \
        'a one-frame detector miss must not end a track under joint either'
    w = [a + [12.0 * 11, 0, 0], b - [12.0 * 11, 0, 0]]
    resumed, _, _ = tr.step(cg, *_lever_boxes(cg, w, side=40.0))
    for s in (0, 1):
        assert abs(resumed[s, 0, [0, 2]].mean() - rows[-1][s, 0, [0, 2]].mean()) < 30.0, \
            f'row {s} resumed on the other animal'


def test_the_tracker_and_associate_agree_on_a_single_uncrowded_animal():
    """On an unambiguous single animal the tracker must reproduce `associate`, so a crowded-
    case delta is a crowding result."""
    import numpy as np

    from tailcyclenet.infer.associate import associate
    from tailcyclenet.infer.track import CrossViewTracker, _project


    cg = _flat_rig()

    tr = CrossViewTracker(1, max_res_px=30.0)
    for t in range(5):
        w = np.array([[10.0 * t, 0.0, 0.0]], np.float32)
        per_cam = []
        for cam in cg:
            uv = _project(cam, w)
            per_cam.append(torch.stack([uv[:, 0] - 20, uv[:, 1] - 20,
                                        uv[:, 0] + 20, uv[:, 1] + 20], -1))
        scores = [torch.ones(1) for _ in cg]
        got, _, _ = tr.step(cg, per_cam, scores)
        ref = associate(cg, per_cam, max_res_px=30.0, max_instances=1)
        assert len(ref) == 1
        for c, box in ref[0]['boxes'].items():
            np.testing.assert_allclose(got[0, c], box.numpy(), atol=1e-4)


def test_a_nan_box_is_skipped_by_both_cross_view_paths():
    """A NaN box (an anchor in the letterbox padding band) must be skipped by `associate`, the
    tracker, and `min_views = 1`."""

    from tailcyclenet.infer.associate import associate
    from tailcyclenet.infer.track import CrossViewTracker, _project

    cg = _flat_rig()

    world = np.array([[0.0, 0.0, 0.0]], np.float32)
    nan_row = torch.full((1, 4), float('nan'))
    per_cam, scores = [], []
    for cam in cg:
        uv = _project(cam, world)
        good = torch.stack([uv[:, 0] - 20, uv[:, 1] - 20, uv[:, 0] + 20, uv[:, 1] + 20], -1)
        per_cam.append(torch.cat([good, nan_row]))       # one real animal, one dead box
        scores.append(torch.ones(2))

    got = associate(cg, per_cam, max_res_px=20.0)
    assert len(got) == 1, 'the real animal must still be found beside a NaN box'
    assert bool(torch.isfinite(got[0]['point']).all())
    # The NaN box is not silently adopted as a member of it.
    assert all(j == 0 for j in got[0]['members'].values())

    tr = CrossViewTracker(2, max_res_px=20.0)
    boxes, _, _ = tr.step(cg, per_cam, scores)
    assert np.isfinite(boxes[0]).all(), 'the tracked animal must come back with real boxes'

    # And `min_views = 1` does not emit the dead box as a single-view instance.
    solo = associate(cg, per_cam, max_res_px=20.0, min_views=1)
    for g in solo:
        for c, b in g['boxes'].items():
            assert torch.isfinite(b).all(), f'camera {c} emitted a NaN box as an instance'


def test_measured_tracker_configuration_is_the_default():
    """The measured zero-switch configuration is what the constructor ships."""
    import inspect

    from tailcyclenet.infer.track import CrossViewTracker

    sig = inspect.signature(CrossViewTracker.__init__).parameters
    assert sig['max_age'].default == 8 and sig['max_move'].default == 1.25


def test_the_tracker_projects_correctly_on_a_moving_rig(mv):
    """On a moving rig `cgroup(gid, t)` with a scalar t gives a (4,4) ext, which `track._project`
    relies on (a (T,4,4) ext would project target i through frame i's pose)."""
    from tailcyclenet.infer.track import _project

    _, sess, _, _ = mv
    gid = list(sess.groups)[0]
    per_frame = sess.cgroup(gid, 0)
    assert all(c['ext'].ndim == 2 for c in per_frame), \
        'a scalar frame must give a (4,4) ext, or the projection axis alignment below is wrong'
    assert sess.cgroup(gid)[0]['ext'].ndim == 3, 'the whole-group form must still be per-frame'
    for n in (1, 3):
        out = _project(per_frame[0], np.zeros((n, 3), np.float32))
        assert out.shape == (n, 2)


# --- Phase 8 levers (each a switch whose default is the shipped behaviour) ---

def _p8_seat(tr, points_ages):
    """Seat targets directly: {slot: (world point, age)}."""
    for s, (p, age) in points_ages.items():
        tr.targets[s] = {'point': torch.tensor(p, dtype=torch.float32), 'age': age}


def test_age_decay_lets_a_close_stale_slot_keep_its_animal():
    """T2's repro: a slot ~0.05 side from the animal but 6 frames stale loses it to a fresh slot
    ~0.4 side away under the additive age term; a small multiplicative decay keeps it."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _skew_rig(SKEW_RIG3)
    boxes = _jittered_boxes(cg, [[0.0, 0.0, 0.0]])
    scores = [torch.ones(1) for _ in cg]
    picks = {}
    for decay in (None, 0.05):
        tr = CrossViewTracker(2, max_res_px=30.0, age_decay=decay)
        _p8_seat(tr, {0: ([5.0, 0.0, 0.0], 6), 1: ([45.0, 0.0, 0.0], 0)})
        out, _, _ = tr.step(cg, boxes, scores)
        picks[decay] = [s for s in range(2) if np.isfinite(out[s]).all()]
    assert picks[None] == [1]
    assert picks[0.05] == [0]


def test_camera_claims_take_a_single_view_detection():
    """TR2: an animal visible in one camera only forms no `associate` group, so the shipped
    tracker leaves its live target unclaimed and ageing; per-camera claiming takes the box and
    resets the age while holding the point."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _skew_rig(SKEW_RIG3)
    full = _jittered_boxes(cg, [[0.0, 0.0, 0.0]])
    boxes = [full[0], full[0][:0], full[0][:0]]
    scores = [torch.ones(1), torch.ones(0), torch.ones(0)]
    for claims in (False, True):
        tr = CrossViewTracker(1, max_res_px=30.0, camera_claims=claims)
        _p8_seat(tr, {0: ([3.0, 0.0, 0.0], 2)})
        out, _, ix = tr.step(cg, boxes, scores)
        if claims:
            assert ix[0, 0] == 0 and tr.targets[0]['age'] == 0
            assert torch.equal(tr.targets[0]['point'], torch.tensor([3.0, 0.0, 0.0]))
        else:
            assert not np.isfinite(out).any() and tr.targets[0]['age'] == 3


def test_affinity_2d_prefers_the_slot_that_continues_its_own_camera_tracks():
    """TR3: two slots whose 3D points sit equally far from a group; the one whose last claimed
    boxes coincide with the group's detections wins once A_2D carries weight."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _skew_rig(SKEW_RIG3)
    boxes = _jittered_boxes(cg, [[0.0, 0.0, 0.0]])
    scores = [torch.ones(1) for _ in cg]
    picks = {}
    for w2 in (0.0, 0.5):
        tr = CrossViewTracker(2, max_res_px=30.0, affinity_2d=w2)
        _p8_seat(tr, {0: ([-20.0, 0.0, 0.0], 0), 1: ([20.0, 0.0, 0.0], 0)})
        tr._t = 0
        tr._cam_history = {1: {c: (boxes[c][0].numpy(), 0) for c in range(3)},
                           0: {c: ((boxes[c][0] + 60.0).numpy(), 0) for c in range(3)}}
        out, _, _ = tr.step(cg, boxes, scores)
        picks[w2] = [s for s in range(2) if np.isfinite(out[s]).all()]
    assert picks[0.5] == [1]
