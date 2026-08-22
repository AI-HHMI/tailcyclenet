"""ONE cross-view target set. Replaces `associate` + `link_rows` on a multi-camera rig.

The old pipeline held three identity mechanisms that never talked to each other -- per-frame
memoryless `associate()`, per-camera `link_rows()`, and the pose `carried` prompt -- and every
artifact of the old deployment path is a consequence of that split: a row that teleports because
its own last-known box was matched by IoU in one camera while the cross-view grouping was rebuilt
from scratch, a real animal starved of a slot because a greedy pass consumed its box.

**One state, one affinity, one Hungarian.** A TARGET is a 3D point with a slot. Each frame every
target reprojects into every camera and is matched, per camera, against that camera's detections;
whatever nobody claims goes to `associate` as a BIRTH, which is the one place a memoryless
pairwise search is the right algorithm. A target that claimed two or more cameras re-triangulates;
one that claimed fewer HOLDS its point (no velocity model -- measured as not worth it).

**THE AFFINITY IS IN PIXELS, over the detection's own box side**, a deliberate simplification of
world-space point-to-ray distance: a point-to-ray distance is its reprojection error times depth
over focal length, but the pixel form needs no `alpha_3d` normalisation constant, and dividing by
the box side puts it in ANIMAL-SIZE units for free, so one gate serves a 30 px fly and a 250 px
rat. It is also the same gate, in the same units, that `link_rows` uses.
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
    displacement is p90 0.06-0.11 body lengths on the shipped multi-animal roots, so one full
    side has 10-16x headroom and rejects essentially nothing legitimate.
    """

    def __init__(self, n_slots, max_res_px=30.0, max_move=1.0, max_age=24, min_views=2):
        self.n = int(n_slots)
        self.max_res_px = float(max_res_px)
        self.max_move = float(max_move)
        self.max_age = int(max_age)
        self.min_views = int(min_views)
        # slot -> {'point': (3,) float32 tensor, 'age': int}
        self.targets = {}

    def step(self, cgroup, boxes_per_cam, scores_per_cam):
        """Match, update, birth, retire. -> (boxes (S,C,4), scores (S,C), claimed (S,C)) numpy.

        `claimed[s, c]` is the DETECTION INDEX slot `s` took in camera `c`, or -1. It is returned
        rather than recomputed because any per-detection quantity a caller wants to carry along
        (the keypoint branch's output, for one) has to follow the SAME assignment the boxes did,
        and matching boxes back to detections afterwards is ambiguous wherever two overlap.

        `boxes_per_cam` is a list of (n_c, 4) tensors in each camera's own pixels, and
        `scores_per_cam` the matching (n_c,) objectness -- the same pair `associate` takes.
        """
        from scipy.optimize import linear_sum_assignment

        C = len(cgroup)
        out = np.full((self.n, C, 4), np.nan, np.float32)
        sc = np.full((self.n, C), np.nan, np.float32)
        claimed_ix = np.full((self.n, C), -1, np.int32)
        centres = [_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
        sides = [_sides(b) if b.numel() else b.new_zeros((0,)) for b in boxes_per_cam]
        slots = [s for s, t in sorted(self.targets.items())
                 if bool(torch.isfinite(t['point']).all())]
        # A TARGET WITH NO 3D POINT CANNOT BE MATCHED, SO IT MUST STILL BE ABLE TO EXPIRE. It is
        # filtered out of `slots` above, so the update loop never touches its `age`, `retire` never
        # fires, and `free` below excludes it because it is still in `self.targets` -- the slot is
        # dead for the rest of the clip. `--min-views 1` creates exactly this: a single-view
        # instance's `point` is all-NaN by design (`associate`), and one such birth on frame 0
        # permanently costs a row.
        #
        # This is NOT the documented immortal one-camera target. That one HAS a finite point, is
        # in `slots`, and its age is maintained; retiring it was tried and measured worse
        # (+2.72 mm MPJPE). This one is invisible to the matcher entirely.
        for s, t in self.targets.items():
            if s not in slots:
                t['age'] += 1
        pts = (torch.stack([self.targets[s]['point'] for s in slots]) if slots else None)
        claimed = {c: set() for c in range(C)}
        got = {s: {} for s in slots}                    # slot -> {cam: det index}

        for c in range(C):
            n_det = centres[c].shape[0]
            if not slots or not n_det:
                continue
            proj = _project(cgroup[c], pts)                                  # (n_t,2)
            d = torch.linalg.norm(proj[:, None] - centres[c][None], dim=-1)  # (n_t,n_det)
            # THE DETECTION'S OWN SIDE, not the mean of it and the target's last claimed side.
            # `link_rows` uses the mean and making these two consistent was TRIED AND MEASURED
            # WORSE: +1.71 mm MPJPE and -0.038 MOTA, paired over 131,887 points on the two
            # ten-bird clips. The paths are not analogous -- `link_rows` averages two boxes seen in
            # the SAME frame, while a target's remembered side is carried forward, so one oversized
            # box gives it an oversized gate for life. Box slots filled RISE while pose coverage
            # FALLS: the extra boxes are wrong.
            side = sides[c][None].clamp_min(1e-6)
            gap = (d / (self.max_move * side)).numpy()
            # THE GATE IS THE ALGORITHM, not a tie-break. A pair beyond one box side is not the same
            # animal, so it must be unavailable to Hungarian rather than merely expensive -- an
            # optimum over an all-bad cost matrix is an arbitrary permutation.
            # A NaN BOX IS UNAVAILABLE, NOT UNRANKABLE. `unletterbox_boxes` returns NaN for a box
            # with no area, which makes `gap` NaN, which `clip` leaves NaN -- and `affinity.any()`
            # is True for NaN, so `linear_sum_assignment` raised `matrix contains invalid numeric
            # entries` and killed the clip. Zero is what the gate already means: unavailable.
            affinity = np.nan_to_num(np.clip(1.0 - gap, 0.0, None), nan=0.0)
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
            if len(cams) >= 2:
                p = torch.stack([centres[c][got[s][c]] for c in cams])
                new = _triangulate(cgroup, cams, p)
                if bool(torch.isfinite(new).all()):
                    self.targets[s]['point'] = new
                # THE KEYPOINT SET, TRIANGULATED THE SAME WAY, so the cues have a 3D thing to
                # reproject rather than one camera's 2D opinion. HELD, not cleared, when a frame
                # fails to produce one: the same rule the point follows (no velocity model), and a
                # shape is a slower-changing quantity than a position. Only keypoints valid in
                # EVERY claimed camera can be triangulated, which is why this is per keypoint
                # rather than all-or-nothing.
            for c, j in got[s].items():
                out[s, c] = boxes_per_cam[c][j].numpy()
                sc[s, c] = float(scores_per_cam[c][j])
                claimed_ix[s, c] = j
            # AGE COUNTS FRAMES WITH NO EVIDENCE AT ALL. One camera is evidence: it cannot move the
            # point, but it says the animal is still there, which is what expiry is about.
            #
            # SO A TARGET CLAIMING EXACTLY ONE CAMERA NEVER EXPIRES AND NEVER UPDATES ITS 3D POINT,
            # since `len(cams) >= 2` above never fires. That reads like a bug and a second counter
            # retiring it on frames-since-re-triangulation was TRIED AND MEASURED WORSE: +2.72 mm
            # MPJPE and miss +0.021, paired over 132,006 points on the same clips. The frozen point
            # costs nothing, because `out[s, c]` below is written from the CLAIMED DETECTION and
            # never from the reprojection -- so a one-camera target is still emitting a real box for
            # a real animal, and expiring it hands its slot to a spurious birth. Leave it immortal.
            self.targets[s]['age'] = 0 if got[s] else self.targets[s]['age'] + 1

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
)
                for s, g in zip(free, born):
                    self.targets[s] = {'point': g['point'], 'age': 0}
                    for c, j in g['members'].items():
                        det = keep[c][j]
                        out[s, c] = boxes_per_cam[c][det].numpy()
                        sc[s, c] = float(scores_per_cam[c][det])
                        claimed_ix[s, c] = det

        # -- retire: a slot with no evidence for a window is free for whoever is there now ---
        for s in [s for s, t in self.targets.items() if t['age'] > self.max_age]:
            del self.targets[s]
        return out, sc, claimed_ix


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
    out, _, _ = tr.step(cg, *empty)
    assert not np.isfinite(out).any() and len(tr.targets) == 2
    # 4. ...and they resume in the SAME slots afterwards.
    w = [a + [12.0 * 11, 0, 0], b - [12.0 * 11, 0, 0]]
    resumed, _, _ = tr.step(cg, *boxes_at(w))
    assert np.isfinite(resumed).all(-1).any(-1).sum() == 2
    for s in (0, 1):
        assert abs(resumed[s, 0, [0, 2]].mean() - rows[-1][s, 0, [0, 2]].mean()) < 30.0
    print('track.demo: ok')


if __name__ == '__main__':
    demo()
