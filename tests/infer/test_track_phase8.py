"""Phase 8 `--track` levers (TR5 scale-free residual gate, TR6a objectness order, ...).

Every lever is a switch whose default is the shipped behaviour; each test pins the lever's own
semantics on a synthetic non-degenerate rig.
"""
import numpy as np
import torch


def _rig(specs):
    """A rig whose cameras sit at their own tilt and height, so rays through different points
    are skew. specs entries are (rvec_x, rvec_y, tvec_y)."""
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


RIG3 = [(0.0, -0.5, 0.0), (0.3, 0.0, 120.0), (-0.2, 0.5, -80.0)]


def _boxes(cg, worlds, side=100.0, jitter=None):
    """Square boxes around each world point's projection, per camera (+ optional pixel jitter)."""
    from tailcyclenet.infer.track import _project

    out = []
    for c, cam in enumerate(cg):
        uv = _project(cam, np.asarray(worlds, np.float32))
        if jitter is not None:
            uv = uv + torch.as_tensor(jitter[c], dtype=torch.float32)
        out.append(torch.stack([uv[:, 0] - side / 2, uv[:, 1] - side / 2,
                                uv[:, 0] + side / 2, uv[:, 1] + side / 2], -1))
    return out


def _scene(side):
    """Three animals, jittered centres (so residuals are non-zero), equal box sides."""
    cg = _rig(RIG3)
    worlds = [[0.0, 0.0, 0.0], [150.0, 0.0, 0.0], [0.0, 150.0, 30.0]]
    rng = np.random.default_rng(0)
    jitter = [rng.normal(0, 6.0, (3, 2)) for _ in cg]
    return cg, _boxes(cg, worlds, side, jitter)


def _key(groups):
    return [(sorted(g['members'].items()), round(g['residual'], 6)) for g in groups]


def test_sides_gate_equals_pixel_gate_when_every_box_has_one_side():
    """With every box the same side s, a gate of g sides IS a gate of g*s pixels: the same
    groups, in the same order, with residuals scaled by 1/s."""
    from tailcyclenet.infer.associate import associate

    side = 80.0
    cg, boxes = _scene(side)
    for px in (5.0, 12.0, 30.0):
        a = associate(cg, boxes, max_res_px=px)
        b = associate(cg, boxes, max_res_px=999.0, max_res_sides=px / side)
        assert [sorted(g['members'].items()) for g in a] == \
            [sorted(g['members'].items()) for g in b]
        np.testing.assert_allclose([g['residual'] / side for g in a],
                                   [g['residual'] for g in b], rtol=1e-5)


def test_sides_gate_is_scale_free():
    """A pixel gate that accepts a small-box rig rejects the same geometry at twice the image
    scale; the box-side gate reads both the same."""
    from tailcyclenet.infer.associate import associate

    cg, small = _scene(60.0)
    n_px = len(associate(cg, small, max_res_px=10.0))
    n_sides = len(associate(cg, small, max_res_sides=10.0 / 60.0))
    assert n_px == n_sides
    # half the box side at the same residual: the side gate tightens, the pixel gate cannot see it
    _, tiny = _scene(30.0)
    assert len(associate(cg, tiny, max_res_px=10.0)) == n_px
    assert len(associate(cg, tiny, max_res_sides=10.0 / 60.0)) <= n_px


def test_default_associate_is_unchanged_by_the_new_keywords():
    from tailcyclenet.infer.associate import associate

    cg, boxes = _scene(80.0)
    scores = [torch.rand(len(b), generator=torch.Generator().manual_seed(1)) for b in boxes]
    assert _key(associate(cg, boxes)) == _key(associate(cg, boxes, scores_per_cam=scores))


def test_score_order_prefers_the_stronger_pair_over_the_lower_residual():
    """Two cameras, one animal, camera 1 double-fires: an exact weak box and a 6 px-off strong
    box. Support ties (no third camera), so the default takes the lower residual and the
    objectness order takes the stronger box."""
    from tailcyclenet.infer.associate import associate

    cg = _rig(RIG3[:2])
    b0, b1 = _boxes(cg, [[0.0, 0.0, 0.0]])
    off = b1 + torch.tensor([6.0, 0.0, 6.0, 0.0])
    boxes = [b0, torch.cat([b1, off])]
    scores = [torch.tensor([0.9]), torch.tensor([0.1, 0.9])]
    plain = associate(cg, boxes, scores_per_cam=scores)
    ordered = associate(cg, boxes, scores_per_cam=scores, score_order=True)
    assert plain[0]['members'] == {0: 0, 1: 0}
    assert ordered[0]['members'] == {0: 0, 1: 1}


def test_score_order_needs_scores():
    import pytest

    from tailcyclenet.infer.associate import associate

    cg, boxes = _scene(80.0)
    with pytest.raises(ValueError, match='scores_per_cam'):
        associate(cg, boxes, score_order=True)


def test_refine_every_camera_keeps_drops_and_adds_per_camera():
    """TR1: a kept camera's box is the shipped refine's box; a pass-1 camera whose refined box
    misses its pass-1 box is dropped alone; a camera pass 1 never used is added when the pose
    reprojects into its frame, and not when it falls outside."""
    from tailcyclenet.data import crop as cropmod
    from tailcyclenet.infer.crops import refine_every_camera

    cg = _rig(RIG3)
    pts = torch.tensor([[[0.0, 0.0, 0.0], [20.0, 10.0, 5.0], [-15.0, 5.0, 0.0]]] * 2)
    _, shipped = cropmod.crop_to_points_3d(cg, pts, 16)
    far = torch.tensor([600, 400, 640, 480], dtype=torch.int32)
    use2, b2, _ = refine_every_camera(pts, [0, 1], [shipped[0], far], [0, 1, 2], cg, 16)
    assert use2 == [0, 2]
    assert torch.equal(b2[0], shipped[0]) and torch.equal(b2[1], shipped[2])
    use3, _, _ = refine_every_camera(pts + torch.tensor([5000.0, 0.0, 0.0]), [0], [shipped[0]],
                                     [0, 1, 2], cg, 16)
    assert 1 not in use3 and 2 not in use3


def _seat(tr, points_ages):
    """Seat targets directly: {slot: (world point, age)}."""
    for s, (p, age) in points_ages.items():
        tr.targets[s] = {'point': torch.tensor(p, dtype=torch.float32), 'age': age}


def test_age_decay_lets_a_close_stale_slot_keep_its_animal():
    """T2's repro: a slot ~0.05 side from the animal but 6 frames stale loses it to a fresh slot
    ~0.4 side away under the additive age term; a small multiplicative decay keeps it."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _rig(RIG3)
    boxes = _boxes(cg, [[0.0, 0.0, 0.0]])
    scores = [torch.ones(1) for _ in cg]
    picks = {}
    for decay in (None, 0.05):
        tr = CrossViewTracker(2, max_res_px=30.0, age_decay=decay)
        _seat(tr, {0: ([5.0, 0.0, 0.0], 6), 1: ([45.0, 0.0, 0.0], 0)})
        out, _, _ = tr.step(cg, boxes, scores)
        picks[decay] = [s for s in range(2) if np.isfinite(out[s]).all()]
    assert picks[None] == [1]
    assert picks[0.05] == [0]


def test_camera_claims_take_a_single_view_detection():
    """TR2: an animal visible in one camera only forms no `associate` group, so the shipped
    tracker leaves its live target unclaimed and ageing; per-camera claiming takes the box and
    resets the age while holding the point."""
    from tailcyclenet.infer.track import CrossViewTracker

    cg = _rig(RIG3)
    full = _boxes(cg, [[0.0, 0.0, 0.0]])
    boxes = [full[0], full[0][:0], full[0][:0]]
    scores = [torch.ones(1), torch.ones(0), torch.ones(0)]
    for claims in (False, True):
        tr = CrossViewTracker(1, max_res_px=30.0, camera_claims=claims)
        _seat(tr, {0: ([3.0, 0.0, 0.0], 2)})
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

    cg = _rig(RIG3)
    boxes = _boxes(cg, [[0.0, 0.0, 0.0]])
    scores = [torch.ones(1) for _ in cg]
    picks = {}
    for w2 in (0.0, 0.5):
        tr = CrossViewTracker(2, max_res_px=30.0, affinity_2d=w2)
        _seat(tr, {0: ([-20.0, 0.0, 0.0], 0), 1: ([20.0, 0.0, 0.0], 0)})
        tr._t = 0
        tr._cam_history = {1: {c: (boxes[c][0].numpy(), 0) for c in range(3)},
                           0: {c: ((boxes[c][0] + 60.0).numpy(), 0) for c in range(3)}}
        out, _, _ = tr.step(cg, boxes, scores)
        picks[w2] = [s for s in range(2) if np.isfinite(out[s]).all()]
    assert picks[0.5] == [1]
