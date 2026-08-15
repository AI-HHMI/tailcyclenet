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
                 dup_res_px=None, axis_veto_deg=None, kpt_affinity=None, random_veto=None,
                 seed=0, kpt_centre=False):
        self.n = int(n_slots)
        self.max_res_px = float(max_res_px)
        self.max_move = float(max_move)
        self.max_age = int(max_age)
        self.min_views = int(min_views)
        self.dup_res_px = dup_res_px
        # THE KEYPOINT CUES, ALL DEFAULT-OFF AND ALL VETOES. Each may only REMOVE an edge the centre
        # gate already accepted; none touches the affinity's value, so `max_move`'s calibration in
        # box sides stands untouched and a wrong cue costs a missed match rather than a wrong one.
        # A target or detection with no usable keypoints ABSTAINS -- see `identity`.
        self.axis_veto_deg = None if axis_veto_deg is None else float(axis_veto_deg)
        self.kpt_affinity = None if kpt_affinity is None else float(kpt_affinity)
        # THE RATE-MATCHED CONTROL, and it is not optional when a veto is quoted. Any rejection
        # flatters a mean over matched points, so a veto's number means nothing without the same
        # number of edges removed at random (`--vis-thresh`'s lesson, CLAUDE.md).
        self.random_veto = None if random_veto is None else float(random_veto)
        # ITEM 4, AND IT IS NOT A VETO. It moves the affinity's POINT from the box centre to the
        # keypoint centroid and leaves the candidate pair set exactly as it was -- the one shape
        # report 19 §4 says can win on a matcher §3 measured as starved of candidates. Gated on §7:
        # a typical animal's per-keypoint error is ~58% independent, so a centroid over K averages
        # that part down, where a common-mode shift is what the box centre already carries.
        self.kpt_centre = bool(kpt_centre)
        # WHICH CUES NEED THE TARGET'S TRIANGULATED KEYPOINT SET. Only these two reproject it; the
        # rest read the detection's own 2D. Computing it regardless made every arm pay K
        # triangulations per target per frame for nothing.
        self._wants_kpts = self.axis_veto_deg is not None or self.kpt_affinity is not None
        self._rng = np.random.default_rng(seed)
        self.vetoed = {'axis': 0, 'kpt': 0, 'random': 0, 'eligible': 0}
        # slot -> {'point': (3,) float32 tensor, 'age': int, 'kpts': (K,3) tensor or None}
        self.targets = {}

    def _veto(self, kind, n=1):
        self.vetoed[kind] += int(n)

    @staticmethod
    def _triangulate_kpts(cgroup, cams, members, kpts_per_cam):
        """(K,3) from one target's claimed detections, per keypoint. None if nothing triangulated.

        PER KEYPOINT, because a detector emits a confidence per keypoint and a bird's beak may be
        visible in two cameras while its tail is visible in three. Requiring all K in all cameras
        would throw the whole set away for one missing point, on the roots where occlusion is
        exactly the problem being solved.
        """
        per_cam = [kpts_per_cam[c] for c in cams]
        if any(k is None for k in per_cam):
            return None
        xy = torch.stack([torch.as_tensor(np.asarray(k[members[c]], np.float32))[:, :2]
                          for c, k in zip(cams, per_cam)])                       # (n_cam, K, 2)
        K = xy.shape[1]
        out = torch.full((K, 3), float('nan'))
        ok = torch.isfinite(xy).all(-1)                                          # (n_cam, K)
        for k in range(K):
            use = tuple(c for i, c in enumerate(cams) if bool(ok[i, k]))
            if len(use) < 2:
                continue
            p = xy[[i for i, c in enumerate(cams) if c in use], k]
            tri = _triangulate(cgroup, use, p)
            if bool(torch.isfinite(tri).all()):
                out[k] = tri
        return out if bool(torch.isfinite(out).all(-1).any()) else None

    def _apply_vetoes(self, affinity, cam, slots, det_kpts, det_boxes):
        """Zero the entries the keypoint cues reject. Returns a NEW array; `affinity` is untouched.

        Each cue reads the TARGET'S HELD KEYPOINT SET reprojected into this camera against the
        DETECTION'S OWN keypoints, so both sides are measured in the same pixels and the comparison
        needs no calibration constant. A target that has never claimed two cameras holds no keypoint
        set and abstains, exactly as a detection with too few valid points does.
        """
        from . import identity as idy

        out = affinity.copy()
        live = np.argwhere(out > 0)
        if live.size == 0:
            return out
        self._veto('eligible', len(live))

        # Reproject each target's held (K,3) into this camera ONCE, not once per pair.
        proj = {}
        for i, s in enumerate(slots):
            kp = self.targets[s].get('kpts')
            if kp is None:
                continue
            ok = torch.isfinite(kp).all(-1)
            if int(ok.sum()) < idy.MIN_PTS:
                continue
            xy = np.full((kp.shape[0], 2), np.nan, np.float32)
            xy[ok.numpy()] = _project(cam, kp[ok]).numpy()
            proj[i] = xy

        dk = None if det_kpts is None else np.asarray(det_kpts, float)
        for i, j in live:
            tk = proj.get(int(i))
            # THE RANDOM CONTROL NEEDS NO KEYPOINTS, and must not be skipped when a keypoint cue
            # would have abstained. It was: an early `continue` on `tk is None` guarded all three
            # cues together, so once the target keypoint set stopped being built for control arms
            # the control silently stopped firing -- a rate-matched control that rejects nothing is
            # not a control, and every veto number quoted against it would have been meaningless.
            have_kpts = tk is not None and dk is not None and int(j) < dk.shape[0]
            d = dk[int(j)] if have_kpts else None
            if self.axis_veto_deg is not None and have_kpts:
                gap = idy.angle_gap(idy.body_axis(tk), idy.body_axis(d))
                # NaN IS AN ABSTENTION, NOT A REJECTION. `gap > thresh` is False for NaN, which is
                # the behaviour wanted -- but writing it as `not (gap <= thresh)` would invert that
                # silently, so it is spelled positively and this comment is why.
                if np.isfinite(gap) and gap > self.axis_veto_deg:
                    out[i, j] = 0.0
                    self._veto('axis')
                    continue
            if self.kpt_affinity is not None and have_kpts:
                frac = idy.kpt_in_box_frac(tk, det_boxes[int(j)].numpy())
                if np.isfinite(frac) and frac < self.kpt_affinity:
                    out[i, j] = 0.0
                    self._veto('kpt')
                    continue
            if self.random_veto and self._rng.random() < self.random_veto:
                out[i, j] = 0.0
                self._veto('random')
        return out

    # -- one frame ------------------------------------------------------------------------
    def step(self, cgroup, boxes_per_cam, scores_per_cam, kpts_per_cam=None):
        """Match, update, birth, retire. -> (boxes (S,C,4), scores (S,C), claimed (S,C)) numpy.

        `claimed[s, c]` is the DETECTION INDEX slot `s` took in camera `c`, or -1. It is returned
        rather than recomputed because any per-detection quantity a caller wants to carry along
        (the keypoint branch's output, for one) has to follow the SAME assignment the boxes did,
        and matching boxes back to detections afterwards is ambiguous wherever two overlap.

        `boxes_per_cam` is a list of (n_c, 4) tensors in each camera's own pixels, and
        `scores_per_cam` the matching (n_c,) objectness -- the same pair `associate` takes.

        `kpts_per_cam` is the optional matching list of (n_c, K, 3) detector keypoints in the same
        pixels. Supplying it is what lets the cues above fire; without it every cue abstains and
        this is byte-identical to the centroid-only tracker.
        """
        from scipy.optimize import linear_sum_assignment

        C = len(cgroup)
        out = np.full((self.n, C, 4), np.nan, np.float32)
        sc = np.full((self.n, C), np.nan, np.float32)
        claimed_ix = np.full((self.n, C), -1, np.int32)
        centres = [_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
        sides = [_sides(b) if b.numel() else b.new_zeros((0,)) for b in boxes_per_cam]
        # THE POINT MOVES; THE GATE, THE SIDES AND THE PAIR SET DO NOT. `sides` stays the BOX side,
        # because `max_move` is calibrated in box sides and a keypoint-extent denominator would be a
        # second, uncalibrated lever riding along (eval rule 4). Per detection, and it falls back to
        # the box centre wherever there are too few keypoints, so no pair can disappear.
        if self.kpt_centre and kpts_per_cam is not None:
            from . import identity as idy
            for c, kk in enumerate(kpts_per_cam):
                if kk is None or not centres[c].numel():
                    continue
                k = np.asarray(kk, float)
                pts = np.stack([idy.centroid(k[i], box=boxes_per_cam[c][i].numpy())
                                for i in range(min(len(k), centres[c].shape[0]))])
                ok = np.isfinite(pts).all(-1)
                centres[c][:len(pts)][ok] = torch.as_tensor(pts[ok], dtype=centres[c].dtype)

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
            # WORSE: +1.71 mm MPJPE [+0.27, +3.15] and -0.038 MOTA [-0.070, -0.006], paired over
            # 131,887 points on the two ten-bird clips. The paths are not analogous -- `link_rows`
            # averages two boxes seen in the SAME frame, while a target's remembered side is
            # carried forward, so one oversized box gives it an oversized gate for life. Box slots
            # filled RISE (0.7605 -> 0.8075) while pose coverage FALLS: the extra boxes are wrong.
            side = sides[c][None].clamp_min(1e-6)
            gap = (d / (self.max_move * side)).numpy()
            # THE GATE IS THE ALGORITHM, not a tie-break. A pair beyond one box side is not the same
            # animal, so it must be unavailable to Hungarian rather than merely expensive -- an
            # optimum over an all-bad cost matrix is an arbitrary permutation, which is exactly how
            # `link_rows` used to swap two animals that never touched.
            # A NaN BOX IS UNAVAILABLE, NOT UNRANKABLE. `unletterbox_boxes` returns NaN for a box
            # with no area, which makes `gap` NaN, which `clip` leaves NaN -- and `affinity.any()`
            # is True for NaN, so `linear_sum_assignment` raised `matrix contains invalid numeric
            # entries` and killed the clip. Zero is what the gate already means: unavailable.
            affinity = np.nan_to_num(np.clip(1.0 - gap, 0.0, None), nan=0.0)
            # THE VETOES, APPLIED TO THE AFFINITY BEFORE THE HUNGARIAN AND NEVER TO ITS VALUE.
            # Zeroing an entry is what the gate already means -- unavailable -- so a veto is exactly
            # a narrower gate and cannot reorder the pairs it leaves alone. Applied before matching
            # rather than after, because rejecting a pair the Hungarian already chose leaves that
            # detection unclaimed while a legal alternative went unconsidered.
            if kpts_per_cam is not None and affinity.any():
                affinity = self._apply_vetoes(affinity, cgroup[c], slots, kpts_per_cam[c],
                                              boxes_per_cam[c])
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
                # AND THE KEYPOINT SET, TRIANGULATED THE SAME WAY, so the cues have a 3D thing to
                # reproject rather than one camera's 2D opinion. HELD, not cleared, when a frame
                # fails to produce one: the same rule the point follows (report 12 R2 -- no velocity
                # model), and a shape is a slower-changing quantity than a position, so a held set is
                # a better prior than none. Only keypoints valid in EVERY claimed camera can be
                # triangulated, which is why this is per keypoint rather than all-or-nothing.
                # ONLY WHEN A CUE ACTUALLY READS IT. This is per keypoint over K, so it is K
                # triangulations per target per frame -- 10 targets x 17 keypoints x 2,680 frames is
                # 455,600 per 3dpop clip, and it ran on EVERY arm the moment the cache carried
                # keypoints, including arms that read none of them. `kpt_centre` is deliberately not
                # in this test: it uses each DETECTION's own 2D keypoints and never the target's
                # triangulated set.
                if self._wants_kpts and kpts_per_cam is not None:
                    kk = self._triangulate_kpts(cgroup, cams, got[s], kpts_per_cam)
                    if kk is not None:
                        self.targets[s]['kpts'] = kk
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
            # MPJPE [+0.51, +4.94], miss +0.021, `fp_none` +0.022, paired over 132,006 points on
            # the same clips. The frozen point costs nothing, because `out[s, c]` below is written
            # from the CLAIMED DETECTION and never from the reprojection -- so a one-camera target
            # is still emitting a real box for a real animal, and expiring it hands its slot to a
            # spurious birth (the slot starvation in dev/reports/12). Leave it immortal.
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
                                 dup_res_px=self.dup_res_px)
                for s, g in zip(free, born):
                    self.targets[s] = {'point': g['point'], 'age': 0, 'kpts': None}
                    for c, j in g['members'].items():
                        det = keep[c][j]
                        out[s, c] = boxes_per_cam[c][det].numpy()
                        sc[s, c] = float(scores_per_cam[c][det])
                        claimed_ix[s, c] = det
                    # A NEWBORN GETS ITS KEYPOINT SET IMMEDIATELY where it claimed two cameras --
                    # otherwise every target abstains on its first frame after birth, and on a
                    # crowded clip births are frequent enough for that to be most of the cue's
                    # opportunities. `members` here indexes the LEFTOVER list, so it maps back
                    # through `keep` exactly as the boxes above do.
                    if self._wants_kpts and kpts_per_cam is not None and len(g['members']) >= 2:
                        cams = tuple(sorted(g['members']))
                        mem = {c: keep[c][g['members'][c]] for c in cams}
                        self.targets[s]['kpts'] = self._triangulate_kpts(
                            cgroup, cams, mem, kpts_per_cam)

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
