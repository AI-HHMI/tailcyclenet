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
one that claimed fewer HOLDS its point (no velocity model by default -- measured as not worth
it, until a crossing with no appearance cue; lever 3 below).

**THE AFFINITY IS IN PIXELS, over the detection's own box side**, a deliberate simplification of
world-space point-to-ray distance: a point-to-ray distance is its reprojection error times depth
over focal length, but the pixel form needs no `alpha_3d` normalisation constant, and dividing by
the box side puts it in ANIMAL-SIZE units for free, so one gate serves a 30 px fly and a 250 px
rat. It is also the same gate, in the same units, that `link_rows` uses.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from posetail.posetail.cube import project_points_torch

from .associate import _centres, _residual, _triangulate, associate

# Measured on 3dpop Sequence 59 frames 1000-1400 (dev/reports/53): a seated duplicate track's
# triangulated points project 0.32-0.73 box sides apart in every camera it claims (per-frame
# min gap >= 0.32), while a genuine distinct-animal contact sat at 0.82+ max.  0.75 clears the
# whole measured duplicate band; 0.25 clears its observed floor while still leaving the
# sub-0.25 near-concentric case to the detector's own centre-distance NMS.  These gates are used
# by the TRACKER's duplicate evidence only (backstop + birth refusal) -- the plan's per-frame
# group NMS in `associate` was REMOVED because a genuine crossing passes through the same band
# for a few frames in every camera, so merging on per-frame geometry turned crossings into
# forced retirements (measured: idsw 28 -> 34, miss 8 -> 11 on the same clip).
DUPLICATE_RADIUS = 0.75
DUPLICATE_MIN_GAP = 0.25


def _box_side(box):
    """Mean side of one finite xyxy box, in pixels."""
    return 0.5 * (float(box[2] - box[0]) + float(box[3] - box[1]))


def _groups_are_duplicate(cgroup, first, second, radius=DUPLICATE_RADIUS):
    """Whether two finite cross-view claim sets are the same animal.

    The comparison stays in the tracker's one unit system: the distance between the two
    triangulated points is measured after projection and divided by the boxes' own pixel side.
    Requiring the test in every SHARED camera rejects two animals that happen to be close in
    one view but separate in another; at least two shared cameras must agree.  The two sets are
    not required to be equal: on the measured clip the duplicate pair spends most frames in
    unequal claim states (one row 4 cameras, the other 3 -- or one row a decode ghost with no
    fresh claims at all), and a strict set-equality gate meant the persistence counter only
    ever saw the rare all-equal frames and never accumulated (report 53).
    """
    p, q = first.get('point'), second.get('point')
    if p is None or q is None or not bool(torch.isfinite(p).all()) \
            or not bool(torch.isfinite(q).all()):
        return False
    first_cams, second_cams = set(first.get('boxes', {})), set(second.get('boxes', {}))
    shared = sorted(first_cams & second_cams)
    if len(shared) < 2:
        return False
    proj = project_points_torch([cgroup[c] for c in shared], torch.stack([p, q]).reshape(2, 1, 3))
    gaps, identical = [], True
    for i, c in enumerate(shared):
        a, b = first['boxes'][c], second['boxes'][c]
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return False
        identical &= bool(torch.equal(a, b))
        side = max(min(_box_side(a), _box_side(b)), 1e-6)
        gaps.append(float(torch.linalg.norm(proj[i, 0] - proj[i, 1])) / side)
    return (max(gaps) <= radius and
            (identical or min(gaps) >= DUPLICATE_MIN_GAP))

# THE DEFECT THE FOUR OPT-IN LEVERS BELOW ADDRESS. The matching phase runs one INDEPENDENT
# Hungarian per camera and nothing afterwards checks that the detection a slot claimed in camera
# 0 and the one it claimed in camera 1 are the same animal: whatever was claimed is triangulated
# and accepted if merely finite. `max_res_px` is spent in exactly one place -- the birth branch
# for unoccupied slots -- so with every slot occupied the residual gate never executes at all
# (sweeping `--assoc-res-max-px` 15/20/40/50 produced byte-identical output at baseline). A slot
# can therefore bind animal A in one view to animal B in another, and the only symptom is a 3D
# point that silently drifts between the two.
#
# The measurement that says this matters: on a 5-fish 2-camera clip (846 frames, ONE keypoint per
# animal, so geometry is the only identity cue) the tracker reads MPJPE 0.291 mm / MOTA 0.962
# while `--no-track` -- the memoryless per-frame `associate`, which DOES group across views and
# DOES gate on a residual -- reads 0.081 mm / 0.997. The memoryless observation model wins
# decisively, which is what licenses spending code on getting it back into the tracker.
#
# THE MEASURED CONFIGURATION IS NOW THE DEFAULT: `joint`, max-age 8, and max-move 1.25. On the
# 3dzef 5-fish clip it has zero identity switches, full coverage, 0.100 mm MPJPE, and 0.309 mm
# p99. The old per-camera path remains explicitly selectable for reproduction
# (`assoc_mode='per-camera'`, max-age 24, max-move 1.0). Claim gating, velocity, and view arbitration remain
# opt-in because they did not improve this two-camera clip; each is independently toggleable so
# a different rig can measure its own trade-off.


def _sides(boxes):
    """Mean side of each box, the length every distance here is measured in."""
    return 0.5 * ((boxes[:, 2] - boxes[:, 0]) + (boxes[:, 3] - boxes[:, 1]))


def _project(cam, points):
    """(n,3) world -> (n,2) pixels in one camera. Batched: this is the per-frame inner loop."""
    p = torch.as_tensor(np.asarray(points, np.float32)).reshape(-1, 1, 3)
    return project_points_torch([cam], p)[0, :, 0]


def _crowd_weights(centres, sides, max_move):
    """Per-camera, per-detection ambiguity weight -- lever 4's one measurement of a view.

    Inputs: centres -- list of (n_c,2) detection centres; sides -- list of (n_c,) box sides;
            max_move -- the same gate width every other distance in this file is divided by.
    Outputs: list of (n_c,) float32 arrays, `clip(d_nearest_other / (max_move * side), 0, 1)`.
    Side effects: none.

    THE RULE: a detection is ambiguous exactly to the extent that ANOTHER detection in the same
    camera is near it, measured in that detection's own box sides -- the file's one unit system.
    1.0 means the nearest rival is a full gate width away, so nothing in this view could be
    confused with it; 0.0 means a twin sits on top of it and this camera's evidence about which
    animal is which is worth nothing. A camera holding a single detection scores 1.0: there is
    no one to confuse it with. A NaN centre is not a rival (it is `unletterbox_boxes`' "no
    detection here"), so its distance is infinite rather than undefined.

    This is the cue that having several views with different occlusion geometry actually buys:
    two animals overlapping in one camera are usually well separated in another, and the old code
    let the crowded camera vote on the 3D point exactly as loudly as the clean one.
    """
    out = []
    for cen, side in zip(centres, sides):
        n = int(cen.shape[0])
        if n < 2:
            out.append(np.ones(n, np.float32))
            continue
        d = torch.linalg.norm(cen[:, None] - cen[None], dim=-1).numpy().astype(np.float64)
        d = np.where(np.isnan(d), np.inf, d)
        np.fill_diagonal(d, np.inf)
        w = d.min(1) / np.maximum(max_move * side.numpy().astype(np.float64), 1e-6)
        out.append(np.clip(w, 0.0, 1.0).astype(np.float32))
    return out


class CrossViewTracker:
    """Stateful across frames. One instance per group; `step` once per frame, in order.

    `max_move` is in box sides, exactly as in `link_rows`: real consecutive-frame box-centre
    displacement is p90 0.06-0.11 body lengths on the shipped multi-animal roots, so one full
    side has 10-16x headroom and rejects essentially nothing legitimate.
    """

    def __init__(self, n_slots, max_res_px=30.0, max_move=1.25, max_age=8, min_views=2,
                 assoc_mode='joint', claim_residual_gate=False, velocity=False,
                 view_arbitration=False, duplicate_radius=DUPLICATE_RADIUS,
                 duplicate_persist=5):
        """Create an empty tracker with `n_slots` rows.

        Inputs: n_slots -- number of animal rows (slots).
                max_res_px -- max reprojection residual (px) for a birth's triangulation, and
                    (lever 1 only) for an individual per-camera claim.
                max_move -- per-frame box-centre displacement gate, in box sides; 1.25 is the
                    measured default.
                max_age -- frames without evidence before a slot is retired; 8 is the measured
                    default.
                min_views -- minimum cameras a birth must be seen in.
                assoc_mode -- 'joint' (measured default: cross-view candidate groups, then ONE
                    Hungarian) or 'per-camera' (the legacy independent-Hungarian path).
                claim_residual_gate -- lever 1: drop a per-camera claim that disagrees with the
                    slot's own other claims by more than `max_res_px`.
                velocity -- lever 3: match against a constant-velocity prediction.
                view_arbitration -- lever 4: discount a camera whose detections are crowded.
                duplicate_radius -- cross-view duplicate radius in box-side units; 0.75 is the
                    scale-free default (measured band, report 53).
                duplicate_persist -- consecutive in-band frames before a seated duplicate pair
                    is retired; 5 measured on 3dpop Sequence 59, where the seated duplicate's
                    in-band runs were 5-35+ frames and every genuine contact stayed under 4.

        `self.targets` maps slot -> {'point': (3,) float32 tensor, 'age': int}, plus
        'velocity' -- a (3,) tensor -- under lever 3 only. 'point' and 'age' keep their meaning
        in every mode: the last TRIANGULATED point and frames since the last evidence. The
        measured 3dzef defaults are `assoc_mode='joint'`, `max_age=8`, `max_move=1.25`.

        `vel_blend` / `vel_decay` / `ambiguous` are plain attributes, sweepable without a
        constructor argument, in the manner of `soft_argmax_threshold`.
        """
        assert assoc_mode in ('per-camera', 'joint'), f'unknown assoc_mode {assoc_mode!r}'
        if claim_residual_gate and assoc_mode == 'joint':
            raise ValueError('claim_residual_gate is only defined for assoc_mode=\'per-camera\'; '
                             'joint association already residual-gates complete groups')
        self.n = int(n_slots)
        self.max_res_px = float(max_res_px)
        self.max_move = float(max_move)
        self.max_age = int(max_age)
        self.min_views = int(min_views)
        self.assoc_mode = str(assoc_mode)
        self.claim_residual_gate = bool(claim_residual_gate)
        self.velocity = bool(velocity)
        self.view_arbitration = bool(view_arbitration)
        if duplicate_radius < 0:
            raise ValueError('duplicate_radius must be non-negative')
        self.duplicate_radius = float(duplicate_radius)
        self.duplicate_persist = int(duplicate_persist)
        self._residuals = {}
        self._dup_contact = {}
        self._dup_shield = set()
        self._dup_anchor = {}
        self._last_claims = {}
        self.vel_blend = 0.5
        self.vel_decay = 0.5
        self.ambiguous = 0.5
        self.targets = {}

    def _predict(self, s):
        """Where slot `s` is expected to be THIS frame -- lever 3's whole surface.

        Inputs: s -- slot id, present in `self.targets`.
        Outputs: a (3,) tensor: the remembered point, or point + velocity under lever 3.
        Side effects: none.

        The module docstring's "no velocity model -- measured as not worth it" was measured on
        roots where the animals rarely swap sides within one box side; it is NOT true of a clip
        where two animals cross, because at the crossing the last known point is equidistant from
        both and the Hungarian is deciding on nothing. A one-frame constant-velocity extrapolation
        is the only cue left when appearance is unavailable (one keypoint per animal, identical
        fish), so this is the lever aimed squarely at the crossing, not at the easy frames.
        A target that has never re-triangulated has no velocity and predicts its own point.
        """
        t = self.targets[s]
        v = t.get('velocity')
        if not self.velocity or v is None:
            return t['point']
        return t['point'] + v

    def _advance(self, s, new):
        """Accept a fresh triangulation for slot `s`.

        Inputs: s -- slot id; new -- (3,) finite world point.
        Outputs: None.
        Side effects: sets `targets[s]['point']`, and under lever 3 blends the velocity.

        The velocity is an EMA of the per-frame displacement at `vel_blend`, not the raw
        difference: the raw difference is one triangulation's noise against another's and would
        make the prediction noisier than the point it corrects.
        """
        t = self.targets[s]
        if self.velocity:
            v, step = t.get('velocity'), new - t['point']
            t['velocity'] = (step if v is None
                             else self.vel_blend * v + (1.0 - self.vel_blend) * step)
        t['point'] = new

    def _decay(self, updated):
        """Age the velocity of every target that did not re-triangulate this frame.

        Inputs: updated -- the set of slots `_advance` was called on.
        Outputs: None.
        Side effects: scales the stored velocities by `vel_decay`.

        A held target keeps its POINT, so its prediction is one step ahead however long it has
        been missing; the honest thing is for that step to shrink. Decay rather than hold, or a
        target that vanishes behind an occluder for a second comes back predicting a position it
        was never measured at and steals whichever animal happens to be there.
        """
        if not self.velocity:
            return
        for s, t in self.targets.items():
            if s not in updated and t.get('velocity') is not None:
                t['velocity'] = t['velocity'] * self.vel_decay

    def _trim(self, cgroup, centres, got_s):
        """Lever 1: drop the per-camera claims that disagree with the rest. -> (kept, dropped).

        Inputs: cgroup -- camera dicts; centres -- per-camera (n,2) detection centres;
                got_s -- {camera: detection index} one slot claimed this frame.
        Outputs: two camera tuples: the ones that survive, and the ones dropped.
        Side effects: none.

        THE OBVIOUS RULE DOES NOT WORK and the failure is not subtle: triangulating everything
        and reprojecting condemns the majority, because `_triangulate` minimises DLT ALGEBRAIC
        error -- on three claims where two rays intersect exactly and the third is 32 px off, the
        fit slides down the odd ray until IT reads 0.00 px and the two honest cameras read 11.41
        px each. Leave-one-out is no better: with one bad claim in three, every leave-one-out set
        of the honest cameras is itself contaminated.

        So the rule is a consensus, which is what `associate` already does one level up: SEED
        FROM EVERY PAIR of claims, count how many of the slot's other claims land within
        `max_res_px` of that pair's point, and keep the largest consistent set, breaking ties on
        the mean residual (which is what separates a pair that really intersects from two skew
        rays the DLT split the difference between). The full set is checked first and exits
        immediately when it is already consistent, so the common case costs ONE triangulation.
        If no pair supports two cameras, every claim is dropped and the slot holds its point.

        A slot claiming fewer than two cameras is untouched: with one ray there is nothing to be
        inconsistent with, and retiring one-camera targets was measured worse (+2.72 mm).
        """
        cams = tuple(sorted(got_s))
        if not self.claim_residual_gate or len(cams) < 2:
            return cams, ()
        best, best_key = (), None
        for seed in [cams] + list(itertools.combinations(cams, 2)):
            res = self._claim_residuals(cgroup, centres, got_s, cams, seed)
            if res is None:
                continue
            inl = tuple(c for c in cams if res[c] <= self.max_res_px)
            key = (len(inl), -sum(res[c] for c in inl) / max(len(inl), 1))
            if len(inl) >= 2 and (best_key is None or key > best_key):
                best, best_key = inl, key
            if len(inl) == len(cams):
                break
        return best, tuple(c for c in cams if c not in best)

    def _claim_residuals(self, cgroup, centres, got_s, cams, seed):
        """Reproject the point `seed`'s claims triangulate to, into every camera in `cams`.

        Inputs: cgroup; centres; got_s -- {camera: detection index}; cams -- every camera to
            score; seed -- the cameras the point is triangulated FROM.
        Outputs: {camera: float pixels}, or None when the triangulation is non-finite (that seed
            simply does not vote).
        Side effects: none.
        """
        pt = {c: centres[c][got_s[c]].reshape(1, 2) for c in cams}
        p3d = _triangulate(cgroup, seed, torch.cat([pt[c] for c in seed]))
        if not bool(torch.isfinite(p3d).all()):
            return None
        return {c: _residual(cgroup, (c,), pt[c], p3d) for c in cams}

    def _voters(self, kept, got_s, weights):
        """Lever 4, per-camera mode: which of a slot's claims may vote on its 3D point.

        Inputs: kept -- the cameras surviving `_trim`; got_s -- {camera: detection index};
                weights -- `_crowd_weights` output, or None when the lever is off.
        Outputs: a camera tuple, a subset of `kept`.
        Side effects: none.

        THE RULE: a camera whose claimed detection sits closer than `ambiguous` (half a gate
        width) to another detection in the SAME camera does not triangulate, provided at least
        two unambiguous cameras remain; otherwise there is nothing better to fall back on and
        every claim votes as before. Its box is still emitted -- crowding is uncertainty, not
        proof of a wrong claim, and that box is still the best crop that camera has to offer,
        so the discount is spent where a mistake is permanent (the carried 3D point) and not
        where it costs a frame of pose. That asymmetry is the whole difference between this
        lever and lever 1, which has a residual PROVING the claim wrong and therefore withholds
        the box too.
        """
        if weights is None or len(kept) < 2:
            return kept
        good = tuple(c for c in kept if float(weights[c][got_s[c]]) >= self.ambiguous)
        return good if len(good) >= 2 else kept

    def step(self, cgroup, boxes_per_cam, scores_per_cam):
        """Match, update, birth, retire. -> (boxes (S,C,4), scores (S,C), claimed (S,C)) numpy.

        Inputs:
            cgroup -- posetail camera dicts for this frame.
            boxes_per_cam -- list of (n_c, 4) tensors in each camera's own pixels.
            scores_per_cam -- matching (n_c,) objectness.
        Outputs:
            (boxes, scores, claimed): `claimed[s, c]` is the DETECTION INDEX slot `s` took in
            camera `c`, or -1 -- returned rather than recomputed so any per-detection quantity
            (keypoints) follows the same assignment.
        Side effects:
            Mutates `self.targets`: points re-triangulated from what each target claimed,
            ages advanced, births and retirements applied.
        Notes:
            A target with no 3D point cannot be matched but must still expire; it is filtered
            out of `slots`, so its age is never touched and its row goes dead for the clip
            (`--min-views 1` creates exactly this). The gate is the algorithm, not a tie-break:
            a pair beyond one box side is not the same animal, so it must be unavailable to
            Hungarian rather than merely expensive; a NaN box is unavailable, not unrankable
            (NaN affinity made `linear_sum_assignment` raise). The affinity uses the detection's
            own side (not the mean with the target's remembered side -- measured worse). A
            one-camera target never expires or updates its 3D point; retiring it was measured
            worse (+2.72 mm MPJPE) because output boxes come from the claimed detection, never
            the reprojection. Everything above `_per_camera` / `_joint` is SHARED, including
            which slots are matchable and how the unmatchable ones age, so the four levers can
            only change which detections a slot ends up holding -- never the state machine.
        """
        C = len(cgroup)
        out = np.full((self.n, C, 4), np.nan, np.float32)
        sc = np.full((self.n, C), np.nan, np.float32)
        claimed_ix = np.full((self.n, C), -1, np.int32)
        centres = [_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
        sides = [_sides(b) if b.numel() else b.new_zeros((0,)) for b in boxes_per_cam]
        slots = [s for s, t in sorted(self.targets.items())
                 if bool(torch.isfinite(t['point']).all())]
        for s, t in self.targets.items():
            if s not in slots:
                t['age'] += 1
        pts = (torch.stack([self._predict(s) for s in slots]) if slots else None)
        weights = (_crowd_weights(centres, sides, self.max_move)
                   if self.view_arbitration else None)
        branch = self._joint if self.assoc_mode == 'joint' else self._per_camera
        updated = branch(cgroup, boxes_per_cam, scores_per_cam, centres, sides, slots, pts,
                         weights, out, sc, claimed_ix)
        self._suppress_duplicate_targets(cgroup, out, sc, claimed_ix)
        self._decay(updated)
        for s in [s for s, t in self.targets.items() if t['age'] > self.max_age]:
            del self.targets[s]
            self._residuals.pop(s, None)
            self._last_claims.pop(s, None)
            self._dup_anchor.pop(s, None)
        for s in list(self._last_claims):
            if s not in self.targets:
                del self._last_claims[s]
        for s, t in self.targets.items():
            claimed = {c: out[s, c].copy() for c in range(out.shape[1])
                       if np.isfinite(out[s, c]).all()}
            if claimed:
                self._last_claims[s] = claimed
        return out, sc, claimed_ix

    def _duplicate_evidence(self, cgroup, out, first, second):
        """The duplicate predicate on two targets' claim evidence.

        This-frame claims win where they exist; a camera the target did not claim this frame is
        filled from its last claimed box (kept while the target lives, so at most `max_age`
        frames old).  Without the fill, a duplicate pair whose roles alternate -- one row fully
        boxed while the other is a decode ghost, then the reverse -- would reset the persistence
        counter on every role change and never accumulate the five in-band frames the backstop
        needs (measured 3dpop Sequence 59, report 53).  A target's remembered claim sits on its
        own track, so the fill cannot fabricate agreement with another animal.
        """
        a = {'point': self.targets[first]['point'], 'boxes': self._claim_evidence(out, first)}
        b = {'point': self.targets[second]['point'], 'boxes': self._claim_evidence(out, second)}
        return _groups_are_duplicate(cgroup, a, b, self.duplicate_radius)

    def _claim_evidence(self, out, slot):
        """One target's duplicate evidence: this frame's claims, filled from its last claims."""
        claimed = {c: torch.as_tensor(out[slot, c]) for c in range(out.shape[1])
                   if np.isfinite(out[slot, c]).all()}
        for c, box in self._last_claims.get(slot, {}).items():
            claimed.setdefault(c, torch.as_tensor(box))
        return claimed

    def _refresh_anchors(self, cgroup, out):
        """Follow each shield's animal, or freeze the anchor while its winner is elsewhere.

        A shield exists to stop the retired duplicate's detection set from re-seating. Keying it
        to the winner's CURRENT claims made it blind for exactly the frames crowding swapped the
        winner's Hungarian assignment onto another animal -- the duplicate re-seated in that one
        frame and the backstop paid a matched-row flip five frames later (report 53, ~3 refires
        per 160 frames). The anchor is the animal, not the slot: it tracks the winner while the
        winner stays on it, freezes when the winner departs, and expires after `max_age` frozen
        frames or when the winner's target dies.
        """
        for s in list(self._dup_anchor):
            if s not in self.targets or not bool(torch.isfinite(self.targets[s]['point']).all()):
                del self._dup_anchor[s]
                continue
            anchor = self._dup_anchor[s]
            current = {'point': self.targets[s]['point'], 'boxes': self._claim_evidence(out, s)}
            if len(current['boxes']) >= 2 and _groups_are_duplicate(
                    cgroup, anchor, current, self.duplicate_radius):
                self._dup_anchor[s] = {'point': current['point'].clone(),
                                       'boxes': dict(current['boxes']), 'stale': 0}
                continue
            anchor['stale'] += 1
            if anchor['stale'] > self.max_age:
                del self._dup_anchor[s]

    def _birth_duplicates_target(self, cgroup, group, out):
        """Whether a newborn group re-creates a duplicate the backstop already retired.

        The backstop retires a seated duplicate, but without this its freed slot is re-born from
        the same leftover detection set on the next frame -- retire-and-rebirth every frame was
        what made the duplicate immortal (3dpop Sequence 59, report 53).  This refuses that
        birth, but ONLY against a shield anchor: a group is refused when it duplicates the
        animal that won a REAL duplicate retirement (``duplicate_persist`` consecutive in-band
        frames).  A bare per-frame duplicate test is not enough -- a genuine crossing is
        geometrically a duplicate for a few frames -- so an unanchored seated target never
        blocks a birth.
        """
        if not bool(torch.isfinite(group.get('point', torch.tensor(float('nan')))).all()):
            return False
        for s, t in self.targets.items():
            if s not in self._dup_shield or not bool(torch.isfinite(t['point']).all()):
                continue
            claimed = self._claim_evidence(out, s)
            if len(claimed) < 2:
                continue
            seated = {'point': t['point'], 'boxes': claimed}
            if _groups_are_duplicate(cgroup, seated, group, self.duplicate_radius):
                return True
        for anchor in self._dup_anchor.values():
            if len(anchor['boxes']) < 2:
                continue
            if _groups_are_duplicate(cgroup, anchor, group, self.duplicate_radius):
                return True
        return False

    def _suppress_duplicate_targets(self, cgroup, out, sc, claimed_ix):
        """Retire the weaker of two targets that have been duplicates for enough frames.

        Deliberately after `_joint`/`_per_camera`: both branches may produce a target pair
        before their different matching logic is visible here, and a pair is only judged on
        the claims either target holds (this frame's, or its last claimed boxes while it
        lives) -- so per-frame geometry alone cannot tell a seated duplicate from a genuine
        crossing.  The discriminator is persistence: a duplicate stays within the measured
        band (0.32-0.73 box side in every shared camera) for run lengths of 5-35+ consecutive
        frames on the measured clip, while real contacts never exceeded 4 (report 53).
        ``duplicate_persist`` (5) is that gate: the counter accumulates only while the pair is
        in-band and resets on any gap.  The output row is cleared together with the target, so
        a retired duplicate cannot still become a false positive in this frame; its slot is
        available for a real birth on the next frame, and the winner's ANIMAL is anchored so
        the same leftover set cannot re-seat it (see `_refresh_anchors` and
        `_birth_duplicates_target`).  A time-expiring shield was measured and removed: the
        detector's double-fire pauses for up to ~50 frames, so a 40-frame expiry let the
        duplicate re-seat at the next return and restarted the retire-and-rebirth cycle
        (report 53).
        """
        live = sorted(s for s, t in self.targets.items()
                      if bool(torch.isfinite(t['point']).all()))
        for ix, first in enumerate(live):
            for second in live[ix + 1:]:
                key = frozenset((first, second))
                if self._duplicate_evidence(cgroup, out, first, second):
                    self._dup_contact[key] = self._dup_contact.get(key, 0) + 1
                elif key in self._dup_contact:
                    del self._dup_contact[key]
        for s in [s for s in self._dup_shield if s not in self.targets]:
            self._dup_shield.discard(s)
        self._refresh_anchors(cgroup, out)
        for key, count in sorted(self._dup_contact.items(), key=lambda kv: (kv[0],)):
            if count < self.duplicate_persist:
                continue
            first, second = tuple(key)
            if first not in self.targets or second not in self.targets:
                continue
            if not self._duplicate_evidence(cgroup, out, first, second):
                continue

            def strength(s):
                """Return a deterministic strength key for one duplicate target."""
                return (int(np.isfinite(out[s]).all(-1).sum()),
                        -float(self._residuals.get(s, float('inf'))),
                        -int(self.targets[s].get('age', 0)), -int(s))
            loser = min((first, second), key=strength)
            winner = first if loser == second else second
            out[loser] = np.nan
            sc[loser] = np.nan
            claimed_ix[loser] = -1
            del self.targets[loser]
            self._residuals.pop(loser, None)
            self._dup_shield.add(winner)
            self._dup_anchor[winner] = {'point': self.targets[winner]['point'].clone(),
                                        'boxes': self._claim_evidence(out, winner), 'stale': 0}
            for k in [k for k in self._dup_contact if loser in k or winner in k]:
                del self._dup_contact[k]

    def _per_camera(self, cgroup, boxes_per_cam, scores_per_cam, centres, sides, slots, pts,
                    weights, out, sc, claimed_ix):
        """The shipped matching phase: one INDEPENDENT Hungarian per camera. -> updated slots.

        Inputs: the frame's cameras, boxes, scores, precomputed centres/sides, the matchable
            `slots` with their predicted points `pts`, `weights` from `_crowd_weights` or None,
            and the three output arrays, written in place.
        Outputs: the set of slots whose 3D point was re-triangulated (for `_decay`).
        Side effects: mutates `self.targets` (points, ages, births) and the output arrays.

        With both levers off this is byte-for-byte the historical path: `_trim` returns every
        claim, `_voters` returns every kept camera, and the slot triangulates over exactly what
        it claimed. Lever 1 releases a dropped detection back into the leftover pool the birth
        branch draws from -- it was decided NOT to be this animal, so it is a candidate to be
        another one, and holding it hostage would turn one bad claim into a missed animal.

        A dropped claim leaves the slot's box, score and `claimed_ix` empty for that camera:
        `claimed_ix` is what the keypoints follow, so a claim that is not trusted enough to
        triangulate is not trusted enough to crop from either. Near-tied assignments are
        decided by age exactly as in `_joint` (1/16 of the affinity range per fresh slot;
        report 53): two slots stranded on one animal stop alternating and the loser starves.
        """
        from scipy.optimize import linear_sum_assignment

        C = len(cgroup)
        claimed = {c: set() for c in range(C)}
        got = {s: {} for s in slots}
        updated = set()

        for c in range(C):
            n_det = centres[c].shape[0]
            if not slots or not n_det:
                continue
            proj = _project(cgroup[c], pts)
            d = torch.linalg.norm(proj[:, None] - centres[c][None], dim=-1)
            side = sides[c][None].clamp_min(1e-6)
            gap = (d / (self.max_move * side)).numpy()
            affinity = np.nan_to_num(np.clip(1.0 - gap, 0.0, None), nan=0.0)
            if not affinity.any():
                continue
            affinity = np.where(
                affinity > 0,
                affinity * 16.0 + 9.0 - np.array([[float(self.targets[s].get('age', 0))
                                                   for s in slots]]).T,
                0.0)
            ri, ci = linear_sum_assignment(-affinity)
            for i, j in zip(ri, ci):
                if affinity[i, j] > 0:
                    got[slots[i]][c] = int(j)
                    claimed[c].add(int(j))

        for s in slots:
            kept, dropped = self._trim(cgroup, centres, got[s])
            for c in dropped:
                claimed[c].discard(got[s].pop(c))
            cams = self._voters(kept, got[s], weights)
            if len(cams) >= 2:
                p = torch.stack([centres[c][got[s][c]] for c in cams])
                new = _triangulate(cgroup, cams, p)
                if bool(torch.isfinite(new).all()):
                    self._advance(s, new)
                    self._residuals[s] = _residual(cgroup, cams,
                                                   torch.stack([centres[c][got[s][c]] for c in cams]),
                                                   new)
                    updated.add(s)
            for c, j in got[s].items():
                out[s, c] = boxes_per_cam[c][j].numpy()
                sc[s, c] = float(scores_per_cam[c][j])
                claimed_ix[s, c] = j
            self.targets[s]['age'] = 0 if got[s] else self.targets[s]['age'] + 1

        free = [s for s in range(self.n) if s not in self.targets]
        if free:
            keep = [[j for j in range(centres[c].shape[0]) if j not in claimed[c]]
                    for c in range(C)]
            if any(keep):
                left = [boxes_per_cam[c][keep[c]] if keep[c]
                        else boxes_per_cam[c].new_zeros((0, 4)) for c in range(C)]
                born = associate(cgroup, left, max_res_px=self.max_res_px,
                                 min_views=self.min_views, max_instances=len(free))
                for s, g in zip(free, born):
                    if self._birth_duplicates_target(cgroup, g, out):
                        continue
                    self._birth(s, g, out, sc, claimed_ix, boxes_per_cam, scores_per_cam,
                                lambda c, j, k=keep: k[c][j])
        return updated

    def _birth(self, s, g, out, sc, claimed_ix, boxes_per_cam, scores_per_cam, remap):
        """Seat one `associate` group in free slot `s`.

        Inputs: s -- a slot with no target; g -- an `associate` group; the three output arrays;
            the frame's boxes and scores; remap -- (camera, group-local index) -> raw detection
            index, since the per-camera path hands `associate` a filtered leftover pool.
        Outputs: None.
        Side effects: creates `self.targets[s]` and writes its row.

        A `min_views = 1` group has an all-NaN point BY DESIGN; it is seated anyway, exactly as
        before, and ages out through the `slots` filter in `step`.
        """
        self.targets[s] = {'point': g['point'], 'age': 0}
        self._residuals[s] = float(g.get('residual', float('inf')))
        if self.velocity:
            self.targets[s]['velocity'] = torch.zeros(3)
        for c, j in g['members'].items():
            det = remap(c, j)
            out[s, c] = boxes_per_cam[c][det].numpy()
            sc[s, c] = float(scores_per_cam[c][det])
            claimed_ix[s, c] = det

    def _group_affinity(self, proj, centres, sides, weights, i, g):
        """Lever 2's cost: how well slot `i`'s prediction explains candidate group `g`.

        Inputs: proj -- per-camera (n_slots,2) projections of every slot's predicted point;
            centres / sides -- per-camera detection centres and box sides; weights --
            `_crowd_weights` or None; i -- the slot's row in `proj`; g -- one `associate` group.
        Outputs: a float affinity in [0, 1]; 0 means UNAVAILABLE to the Hungarian.
        Side effects: none.

        Deliberately NOT the 3D distance between the slot's point and the group's triangulated
        point, even though that is the obvious quantity: a millimetre gate is a second unit
        system, un-normalised by animal size, and this file's whole affinity design exists to
        avoid exactly that. So the slot point is projected into every camera the group holds and
        compared against that group's own detection centre in box sides -- the same number the
        per-camera path computes, just averaged over the group instead of decided per camera.

        The average is over cameras, weighted by lever 4's ambiguity when it is on, so a crowded
        view contributes proportionally less to the decision than a view that separates the
        animals cleanly. One camera beyond the gate therefore does not by itself veto the match,
        but it does drag the mean; a pair whose weighted mean gap exceeds one box side scores 0
        and is UNAVAILABLE, not merely expensive -- the gate is the algorithm, as ever.
        """
        num, den = 0.0, 0.0
        for c, j in g['members'].items():
            side = max(float(sides[c][j]), 1e-6)
            d = float(torch.linalg.norm(proj[c][i] - centres[c][j]))
            w = 1.0 if weights is None else float(weights[c][j])
            num += w * d / (self.max_move * side)
            den += w
        if den <= 0.0 or not np.isfinite(num):
            return 0.0
        return float(np.clip(1.0 - num / den, 0.0, None))

    def _joint(self, cgroup, boxes_per_cam, scores_per_cam, centres, sides, slots, pts,
               weights, out, sc, claimed_ix):
        """Lever 2: cross-view groups first, then ONE Hungarian over slots x groups.

        Inputs and outputs: as `_per_camera`.
        Side effects: mutates `self.targets` and the output arrays.

        The shipped path consumes detections per camera and only the LEFTOVERS ever reach
        `associate`, so the one routine in the repo that checks a group's cross-view consistency
        sees only what the tracker did not want. Here `associate` runs first, over the WHOLE
        pool: every group it returns has already been triangulated and had its reprojection
        residual gated at `max_res_px`, honouring `min_views`. Identity is then a choice among
        objects that are internally consistent by construction, which is precisely the property
        `--no-track` has and the tracker threw away (0.081 mm vs 0.291 mm on the fish clip).

        A slot that matches nothing ages exactly as before. A group no slot wanted is a BIRTH
        into a free slot, in `associate`'s own support-then-residual order, so the strongest
        unclaimed evidence seats first. Groups are matched, not detections, so no slot can hold
        animal A in one camera and animal B in another: the hole this whole file is about.

        The affinity is lexicographically scaled against the slot's own age: a slot that
        matched last frame (age 0) wins any near-tie, so two slots whose stale points both sit
        on one animal (their points both project onto the frame's single detection set for it)
        stop flip-flopping and the loser starves to max_age. Measured 3dpop Sequence 59
        (report 53): without this, one escaped duplicate birth stranded TWO slots on a
        stationary pigeon; the Hungarian decided on per-frame jitter and the pair alternated
        every 1-13 frames forever, never reaching max_age, reading as 28 identity switches.
        1/16 of the affinity range (~0.08 box side at max_move 1.25) is far below a genuine
        contender's margin but far above detection-centre jitter, so only near-ties are
        decided by age; the +9 keeps every real affinity positive so an unavailable (zero)
        cell can never displace a weak-but-real match.
        """
        from scipy.optimize import linear_sum_assignment

        C = len(cgroup)
        groups = associate(cgroup, boxes_per_cam, max_res_px=self.max_res_px,
                           min_views=self.min_views)
        proj = {c: (_project(cgroup[c], pts) if slots else None) for c in range(C)}
        ages = {s: float(self.targets[s].get('age', 0)) for s in slots}
        aff = np.zeros((len(slots), len(groups)), np.float64)
        for i in range(len(slots)):
            for k, g in enumerate(groups):
                aff[i, k] = self._group_affinity(proj, centres, sides, weights, i, g)
        if aff.any():
            aff = np.where(aff > 0, aff * 16.0 + 9.0 - np.array([[ages[s] for s in slots]]).T, 0.0)
        updated, taken, matched = set(), set(), {}
        if aff.size and aff.any():
            ri, ci = linear_sum_assignment(-aff)
            for i, k in zip(ri, ci):
                if aff[i, k] > 0:
                    matched[slots[i]] = k
                    taken.add(k)

        for s in slots:
            if s not in matched:
                self.targets[s]['age'] += 1
                continue
            g = groups[matched[s]]
            if bool(torch.isfinite(g['point']).all()):
                self._advance(s, g['point'])
                self._residuals[s] = float(g.get('residual', float('inf')))
                updated.add(s)
            for c, j in g['members'].items():
                out[s, c] = boxes_per_cam[c][j].numpy()
                sc[s, c] = float(scores_per_cam[c][j])
                claimed_ix[s, c] = j
            self.targets[s]['age'] = 0

        free = [s for s in range(self.n) if s not in self.targets]
        for s, k in zip(free, [k for k in range(len(groups)) if k not in taken]):
            g = groups[k]
            if self._birth_duplicates_target(cgroup, g, out):
                continue
            self._birth(s, g, out, sc, claimed_ix, boxes_per_cam, scores_per_cam,
                        lambda c, j: j)
        return updated


def demo():
    """Two synthetic animals on a three-camera rig: the properties that must hold.

    `assert`-based and dependency-free so this file can be checked without the test suite:
        pixi run python -m tailcyclenet.detector.track

    The animals CROSS: A walks right, B walks left, and the SCORE ORDER swaps every frame the
    way `decode` reorders them -- a row that follows one animal must be immune to that. The
    checks, in order: (1) both slots stay filled in every frame (births on frame 0, matches
    after); (2) each row's own box moves smoothly -- a swapped row would jump the full
    separation; (3) a frame with no detections at all ages the targets and returns nothing
    without dropping them -- a one-frame detector miss must not end a track; (4) they resume in
    the SAME slots afterwards.
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
        """Project `worlds` into every camera as fixed-size square boxes.

        Outputs: (per_cam, scores): per_cam is a list of (n,4) xyxy tensors and scores
        a list of (n,) ones.
        """
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
        w = [a + [12.0 * t, 0, 0], b - [12.0 * t, 0, 0]]
        per_cam, scores = boxes_at(w if t % 2 == 0 else w[::-1])
        rows.append(tr.step(cg, per_cam, scores)[0])

    assert all(np.isfinite(r).all(-1).any(-1).sum() == 2 for r in rows), 'an animal was lost'
    for s in (0, 1):
        cx = np.array([r[s, 0, [0, 2]].mean() for r in rows])
        assert np.abs(np.diff(cx)).max() < 30.0, f'row {s} jumped: {np.diff(cx)}'
    empty = [torch.zeros((0, 4)) for _ in cg], [torch.zeros((0,)) for _ in cg]
    out, _, _ = tr.step(cg, *empty)
    assert not np.isfinite(out).any() and len(tr.targets) == 2
    w = [a + [12.0 * 11, 0, 0], b - [12.0 * 11, 0, 0]]
    resumed, _, _ = tr.step(cg, *boxes_at(w))
    assert np.isfinite(resumed).all(-1).any(-1).sum() == 2
    for s in (0, 1):
        assert abs(resumed[s, 0, [0, 2]].mean() - rows[-1][s, 0, [0, 2]].mean()) < 30.0
    print('track.demo: ok')


if __name__ == '__main__':
    demo()
