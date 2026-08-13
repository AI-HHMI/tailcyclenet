"""ONE cross-view target set. Replaces `associate` + `link_rows` on a multi-camera rig.

`dev/reports/12_crossview_tracking.md` §1 names the problem: the pipeline held **three identity
mechanisms that never talked to each other** --

1. `associate()`  -- per frame, memoryless, box CENTRES only, greedy pairwise triangulation.
2. `link_rows()`  -- per frame, Hungarian on box IoU against last-known, per camera independently.
3. `carried[a]` in `infer.run_group` -- a real 3D POSE per row, fed to the model as a prompt and
   never read by 1 or 2.

Every artifact in RC2 and RC4 of `dev/reports/13_deployment_path.md` is a consequence of that split:
a row that teleports because its own last-known box was matched by IoU in one camera while the
cross-view grouping was rebuilt from scratch, an instance accepted on a two-view residual that two
rays can always satisfy, a real animal starved of a slot because a greedy pass consumed its box.

**One state, one affinity, one Hungarian.** A TARGET is a 3D point with a slot. Each frame every
target reprojects into every camera and is matched, per camera, against that camera's detections;
whatever nobody claims goes to `associate` as a BIRTH, which is the one place a memoryless pairwise
search is the right algorithm. A target that claimed two or more cameras re-triangulates; one that
claimed fewer HOLDS its point (report 12 R2: no velocity model -- it measured as not worth it).

Two things this buys, both measured in report 12 on the real rig:

- **Coverage.** §2.1: at 15 px box jitter `associate()` leaves **17.2%** of the boxes it was offered
  unclaimed (accuracy 0.932, boxes used 0.828) where point-to-ray target matching claims all of them
  at accuracy 0.979. Every unclaimed box is a `miss`, and 3dpop's miss term is 0.108.
- **A capability, not just accuracy.** §2.2: `associate()` is O(C^2) in cameras -- 222 ms/frame at
  C = 4, **4.1 s at C = 16, 13.7 s at C = 28** -- against 0.5-0.9 ms for target matching, and
  `max_instances` does not bound it because the whole candidate list is built before the greedy loop
  can break. johnson-mouse is a 16-camera rig; any multi-animal 16-camera rig is unrunnable today.
  Births still pay O(C^2), but only over the boxes nothing claimed, which after the first frame is
  nearly none.

**THE AFFINITY IS IN PIXELS, over the detection's own box side**, and that is a deliberate
simplification of report 12 eq 4's world-space point-to-ray distance. The two are the same test --
a point-to-ray distance is its reprojection error times depth over focal length -- but the pixel form
needs no `alpha_3d` normalisation constant, and dividing by the box side puts it in ANIMAL-SIZE units
for free, so one gate serves a 30 px fly and a 250 px rat. It is also the same gate, in the same
units, that `link_rows` uses, rather than a second threshold with its own calibration.

Not here, deliberately:

- **R3** (association inside the window loop, consuming `carried`) and **R5-proper** (the
  keypoint-in-box term of eq 2) both need the POSE, which does not exist inside `detect_group` --
  that pass finishes before any pose is computed. They are a structural change to `run_group`, and
  they were also unsafe before `--carry-source triangulate`, because report 12 R3 assumes `carried`
  does not accumulate error and RC1 shows it did. Next stage.
- **R6** (a camera with no box contributes its last box at a decayed weight) is SUBSUMED: what
  persists here is the target's 3D point, which reprojects into every camera whether or not that
  camera saw anything, so there is no per-camera dropout to decay.
- Appearance features, velocity, cross-track arbitration. Report 12 §2.3, §4, and report 11's
  `fp_none` ~ 10x `fp_dup` on 3dpop.
"""
from __future__ import annotations

import numpy as np
import torch

from posetail.posetail.cube import project_points_torch

from .associate import _centres, _triangulate, associate


def _sides(boxes):
    """Mean side of each box, the length every distance here is measured in."""
    return 0.5 * ((boxes[:, 2] - boxes[:, 0]) + (boxes[:, 3] - boxes[:, 1]))


def _project(cam, points):
    """(n,3) world -> (n,2) pixels in one camera. Batched: this is the per-frame inner loop."""
    p = torch.as_tensor(np.asarray(points, np.float32)).reshape(-1, 1, 3)
    return project_points_torch([cam], p)[0, :, 0]                      # (n,2)


class CrossViewTracker:
    """Stateful across frames. One instance per group; `step` once per frame, in order.

    `max_move` is in box sides, exactly as in `link_rows`: real consecutive-frame box-centre
    displacement is p90 0.06-0.11 body lengths on all three shipped multi-animal roots, so one full
    side has 10-16x headroom and rejects essentially nothing legitimate.
    """

    def __init__(self, n_slots, max_res_px=30.0, max_move=1.0, max_age=24, min_views=2,
                 dup_res_px=None):
        self.n = int(n_slots)
        self.max_res_px = float(max_res_px)
        self.max_move = float(max_move)
        self.max_age = int(max_age)
        self.min_views = int(min_views)
        self.dup_res_px = dup_res_px
        # slot -> {'point': (3,) float32, 'age': int, 'stale': int, 'sides': {cam: float}}
        self.targets = {}

    # -- one frame ------------------------------------------------------------------------
    def step(self, cgroup, boxes_per_cam, scores_per_cam):
        """Match, update, birth, retire. Returns (boxes (S,C,4), scores (S,C)) as numpy.

        `boxes_per_cam` is a list of (n_c, 4) tensors in each camera's own pixels, and
        `scores_per_cam` the matching (n_c,) objectness -- the same pair `associate` takes.
        """
        from scipy.optimize import linear_sum_assignment

        C = len(cgroup)
        out = np.full((self.n, C, 4), np.nan, np.float32)
        sc = np.full((self.n, C), np.nan, np.float32)
        centres = [_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
        sides = [_sides(b) if b.numel() else b.new_zeros((0,)) for b in boxes_per_cam]

        slots = [s for s, t in sorted(self.targets.items())
                 if bool(torch.isfinite(t['point']).all())]
        pts = (torch.stack([self.targets[s]['point'] for s in slots]) if slots else None)
        claimed = {c: set() for c in range(C)}
        got = {s: {} for s in slots}                    # slot -> {cam: det index}

        for c in range(C):
            n_det = centres[c].shape[0]
            if not slots or not n_det:
                continue
            proj = _project(cgroup[c], pts)                                  # (n_t,2)
            d = torch.linalg.norm(proj[:, None] - centres[c][None], dim=-1)  # (n_t,n_det)
            # NORMALISE BY THE MEAN OF BOTH SIDES, as `link_rows` does. Dividing by the DETECTION's
            # side alone lets a spuriously large box buy itself a proportionally large gate, so the
            # boxes most likely to be wrong are the ones most likely to be admitted. A target that
            # has never claimed this camera has no side to average, and falls back to the detection's.
            det_side = sides[c][None]                                        # (1,n_det)
            prev = torch.tensor([self.targets[s]['sides'].get(c, float('nan')) for s in slots],
                                dtype=det_side.dtype)[:, None]               # (n_t,1)
            side = torch.where(torch.isfinite(prev), 0.5 * (prev + det_side), det_side)
            gap = (d / (self.max_move * side.clamp_min(1e-6))).numpy()
            # THE GATE IS THE ALGORITHM, not a tie-break. A pair beyond one box side is not the same
            # animal, so it must be unavailable to Hungarian rather than merely expensive -- an
            # optimum over an all-bad cost matrix is an arbitrary permutation, which is exactly how
            # `link_rows` used to swap two animals that never touched.
            affinity = np.clip(1.0 - gap, 0.0, None)
            if not affinity.any():
                continue
            ri, ci = linear_sum_assignment(-affinity)
            for i, j in zip(ri, ci):
                if affinity[i, j] > 0:
                    got[slots[i]][c] = int(j)
                    claimed[c].add(int(j))

        # -- update: re-triangulate from what this target actually claimed, else hold -------
        for s in slots:
            cams = tuple(sorted(got[s]))
            fresh = False
            if len(cams) >= 2:
                p = torch.stack([centres[c][got[s][c]] for c in cams])
                new = _triangulate(cgroup, cams, p)
                if bool(torch.isfinite(new).all()):
                    self.targets[s]['point'] = new
                    fresh = True
            for c, j in got[s].items():
                out[s, c] = boxes_per_cam[c][j].numpy()
                sc[s, c] = float(scores_per_cam[c][j])
                self.targets[s]['sides'][c] = float(sides[c][j])
            # AGE COUNTS FRAMES WITH NO EVIDENCE AT ALL. One camera is evidence: it cannot move the
            # point, but it says the animal is still there, which is what expiry is about.
            self.targets[s]['age'] = 0 if got[s] else self.targets[s]['age'] + 1
            # STALE COUNTS FRAMES THE POINT DID NOT MOVE, and it is a SECOND counter because `age`
            # cannot see this: a target claiming exactly one camera every frame resets `age` forever
            # while `len(cams) >= 2` above never fires, so it never re-triangulates and never
            # expires -- an immortal target emitting one box per frame off a frozen point. That is a
            # one-camera 3D window downstream (an untrained input shape at `prob_2d_only = 0`) and a
            # garbage pose. Same `max_age`: a point nobody could re-derive for a window is not a track.
            self.targets[s]['stale'] = 0 if fresh else self.targets[s]['stale'] + 1

        # -- births: whatever nobody claimed, through the memoryless pairwise search --------
        free = [s for s in range(self.n) if s not in self.targets]
        if free:
            keep = [[j for j in range(centres[c].shape[0]) if j not in claimed[c]]
                    for c in range(C)]
            if any(keep):
                left = [boxes_per_cam[c][keep[c]] if keep[c]
                        else boxes_per_cam[c].new_zeros((0, 4)) for c in range(C)]
                born = associate(cgroup, left, max_res_px=self.max_res_px,
                                 min_views=self.min_views, max_instances=len(free),
                                 dup_res_px=self.dup_res_px)
                for s, g in zip(free, born):
                    self.targets[s] = {'point': g['point'], 'age': 0, 'stale': 0, 'sides': {}}
                    for c, j in g['members'].items():
                        det = keep[c][j]
                        out[s, c] = boxes_per_cam[c][det].numpy()
                        sc[s, c] = float(scores_per_cam[c][det])
                        self.targets[s]['sides'][c] = float(sides[c][det])

        # -- retire: a slot with no evidence for a window is free for whoever is there now ---
        for s in [s for s, t in self.targets.items()
                  if t['age'] > self.max_age or t['stale'] > self.max_age]:
            del self.targets[s]
        return out, sc


def demo():
    """Two synthetic animals on a three-camera rig: the properties that must hold.

    `assert`-based and dependency-free so this file can be checked without the test suite:
        pixi run python -m tailcyclenet.detector.track
    """
    from aniposelib.cameras import Camera, CameraGroup

    from ..format import Rig

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    cg = Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
             moving=dict.fromkeys(names, False),
             calibrated=dict.fromkeys(names, True)).posetail()

    def boxes_at(worlds, side=40.0):
        per_cam, scores = [], []
        for cam in cg:
            uv = _project(cam, np.asarray(worlds, np.float32))
            per_cam.append(torch.stack([uv[:, 0] - side / 2, uv[:, 1] - side / 2,
                                        uv[:, 0] + side / 2, uv[:, 1] + side / 2], -1))
            scores.append(torch.ones(len(worlds)))
        return per_cam, scores

    a, b = np.array([-80.0, 0.0, 0.0]), np.array([80.0, 0.0, 0.0])
    tr = CrossViewTracker(2, max_res_px=30.0)
    rows = []
    for t in range(12):
        # They cross: A walks right, B walks left, and the SCORE ORDER swaps every frame the way
        # `decode` reorders them. A row that follows one animal must be immune to that.
        w = [a + [12.0 * t, 0, 0], b - [12.0 * t, 0, 0]]
        per_cam, scores = boxes_at(w if t % 2 == 0 else w[::-1])
        rows.append(tr.step(cg, per_cam, scores)[0])

    # 1. Both slots are filled in every frame -- births on frame 0, matches after.
    assert all(np.isfinite(r).all(-1).any(-1).sum() == 2 for r in rows), 'an animal was lost'
    # 2. Each row's own box moves smoothly: a swapped row would jump the full separation.
    for s in (0, 1):
        cx = np.array([r[s, 0, [0, 2]].mean() for r in rows])
        assert np.abs(np.diff(cx)).max() < 30.0, f'row {s} jumped: {np.diff(cx)}'
    # 3. A frame with no detections at all ages the targets and returns nothing, without dropping
    #    them -- a one-frame detector miss must not end a track.
    empty = [torch.zeros((0, 4)) for _ in cg], [torch.zeros((0,)) for _ in cg]
    out, _ = tr.step(cg, *empty)
    assert not np.isfinite(out).any() and len(tr.targets) == 2
    # 4. ...and they resume in the SAME slots afterwards.
    w = [a + [12.0 * 11, 0, 0], b - [12.0 * 11, 0, 0]]
    resumed, _ = tr.step(cg, *boxes_at(w))
    assert np.isfinite(resumed).all(-1).any(-1).sum() == 2
    for s in (0, 1):
        assert abs(resumed[s, 0, [0, 2]].mean() - rows[-1][s, 0, [0, 2]].mean()) < 30.0
    # 5. A target that claims exactly ONE camera every frame must still expire. `age` cannot see
    #    this -- one camera resets it -- so this is what the second `stale` counter is for.
    tr2 = CrossViewTracker(2, max_res_px=30.0, max_age=4)
    per_cam, scores = boxes_at(w)
    tr2.step(cg, per_cam, scores)
    assert len(tr2.targets) == 2, 'births failed'
    one_cam = ([per_cam[0]] + [torch.zeros((0, 4)) for _ in cg[1:]],
               [scores[0]] + [torch.zeros((0,)) for _ in cg[1:]])
    for _ in range(6):
        out1, _ = tr2.step(cg, *one_cam)
    assert not tr2.targets, 'a single-camera target never expired'
    # 6. The gate normalises by the MEAN of the target's last side and the detection's, NOT by the
    #    detection's alone -- otherwise a spuriously large box buys itself a proportionally large
    #    gate, so the boxes most likely to be wrong are the ones most likely to be admitted.
    #    Born at side 40, then offered a box inflated 4x: the gate is (40+160)/2 = 100 px, so a
    #    130 px jump must be REFUSED. The detection-side-only form gated at 160 and took it.
    for px, want in ((60.0, True), (130.0, False)):
        tr3 = CrossViewTracker(1, max_res_px=30.0)
        tr3.step(cg, *boxes_at([a]))                            # birth records side 40
        moved, sc3 = boxes_at([a + [px * 900.0 / 800.0, 0, 0]], side=160.0)
        got3, _ = tr3.step(cg, moved, sc3)
        assert bool(np.isfinite(got3[0]).any()) == want, f'{px} px jump: gate followed the detection'
    print('track.demo: ok')


if __name__ == '__main__':
    demo()
