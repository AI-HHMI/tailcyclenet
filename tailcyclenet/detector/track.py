"""ONE cross-view target set. Replaces `associate` + `link_rows` on a multi-camera rig.

The old pipeline held three identity mechanisms that never talked to each other -- per-frame
memoryless `associate()`, per-camera `link_rows()`, and the pose `carried` prompt -- and every
artifact of the old deployment path is a consequence of that split: a row that teleports because
its own last-known box was matched by IoU in one camera while the cross-view grouping was rebuilt
from scratch, a real animal starved of a slot because a greedy pass consumed its box.

**One state, one affinity, one Hungarian.** A TARGET is a 3D point with a slot. Each frame
`associate` first groups the WHOLE detection pool across cameras (triangulated, residual-gated at
`max_res_px`, `min_views` honoured); then one Hungarian matches targets to those GROUPS, scoring a
target by its point's reprojection against each group's own detection centres (`_joint`). A group
no target takes is a BIRTH into a free slot. A matched group with a triangulated point updates the
target's point; otherwise the target HOLDS its point (no velocity model -- measured not worth it,
report 53/54). Matching is O(slots x groups) after `associate`'s own O(C^2) pairwise search, so
the tracker inherits that cost; it does not claim boxes per camera that `associate` left
unclaimed. This is not the per-camera matcher of Chen et al. 2020, which an earlier version of
this file implemented (deleted, reports 53/54).

**THE AFFINITY IS IN PIXELS, over the detection's own box side**, a deliberate simplification of
world-space point-to-ray distance: a point-to-ray distance is its reprojection error times depth
over focal length, but the pixel form needs no `alpha_3d` normalisation constant, and dividing by
the box side puts it in ANIMAL-SIZE units for free, so one gate serves a 30 px fly and a 250 px
rat. It is also the same gate, in the same units, that `link_rows` uses.
"""
from __future__ import annotations


import numpy as np
import torch

from tracktail.model.cube import project_points_torch

from .associate import associate
from .boxes import centres as box_centres
from .boxes import sides as box_sides

# On a crowded multiview clip, a seated duplicate track's
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


# The Hungarian cost below packs age into `affinity * 16 + AGE_STICKY_CAP - age` so a fresher
# slot wins a near-tie (report 53). That ordering holds only while `AGE_STICKY_CAP - age >= 0`;
# past it the term can go negative for a genuinely available, high-affinity cell, which the
# post-hoc `affinity > 0` check then reads as UNAVAILABLE -- a real match loses to nothing
# (independent review, identity_review_followthrough plan A3). The shipped default `max_age=8`
# never reaches age 9, so clamping the age USED IN THIS TERM at the cap changes nothing there;
# it only stops ages above it (an old `max_age=24` reproduction, or any future config raising
# `max_age`) from inverting the sign. Do not raise the cap without re-deriving `* 16.0` beside it
# -- they are the same invariant.
AGE_STICKY_CAP = 9.0


def _age_term(age):
    """The `9 - age` half of the sticky-tie cost, clamped so it can never go negative."""
    return AGE_STICKY_CAP - min(float(age), AGE_STICKY_CAP)


def _groups_are_duplicate(cgroup, first, second, radius=DUPLICATE_RADIUS, min_shared=2):
    """Whether two finite cross-view claim sets are the same animal.

    The comparison stays in the tracker's one unit system: the distance between the two
    triangulated points is measured after projection and divided by the boxes' own pixel side.
    Requiring the test in every SHARED camera rejects two animals that happen to be close in
    one view but separate in another; at least two shared cameras must agree by default.  The
    two sets are not required to be equal: on the measured clip the duplicate pair spends most
    frames in unequal claim states (one row 4 cameras, the other 3 -- or one row a decode ghost
    with no fresh claims at all), and a strict set-equality gate meant the persistence counter
    only ever saw the rare all-equal frames and never accumulated (report 53).

    ``min_shared=1`` is a SEPARATE, laxer mode used only for the birth refusal against a
    shielded anchor (report 53 follow-up): a birth whose leftover group only overlaps the
    anchor's memory in ONE camera (the double-fire's second box momentarily invisible in every
    other view) was passing this test unrefused at 2-camera agreement, letting the freed slot
    re-seat and drift back onto the same animal over subsequent frames -- the observed refire
    cycle. A false-positive refusal here only costs one birth's coverage; the retirement
    predicate that can actually MERGE two crossing animals keeps the 2-camera floor.
    """
    p, q = first.get('point'), second.get('point')
    if p is None or q is None or not bool(torch.isfinite(p).all()) \
            or not bool(torch.isfinite(q).all()):
        return False
    first_cams, second_cams = set(first.get('boxes', {})), set(second.get('boxes', {}))
    shared = sorted(first_cams & second_cams)
    if len(shared) < min_shared:
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

# HISTORICAL CONTEXT (the legacy `assoc_mode='per-camera'` path, deleted -- reports 53/54 measured
# `joint` strictly better and no consumer of `per-camera` survived it). That path ran one
# INDEPENDENT Hungarian per camera and nothing afterwards checked that the detection a slot
# claimed in camera 0 and the one it claimed in camera 1 were the same animal: whatever was
# claimed was triangulated and accepted if merely finite. `max_res_px` was spent in exactly one
# place there -- the birth branch for unoccupied slots -- so with every slot occupied the residual
# gate never executed at all under `per-camera`. Under `joint` (`_joint` below) this does not
# apply: `associate()` runs over the WHOLE detection pool every frame, so `max_res_px` gates every
# candidate group in every frame, not only births.
#
# The measurement that motivated `joint`: on a 5-fish 2-camera clip (846 frames, ONE keypoint per
# animal, so geometry is the only identity cue) the OLD tracker read MPJPE 0.291 mm / MOTA 0.962
# while `--no-track` -- the memoryless per-frame `associate`, which DOES group across views and
# DOES gate on a residual -- read 0.081 mm / 0.997. The memoryless observation model won
# decisively, which is what licensed spending code on getting it back into the tracker.
#
# THE MEASURED CONFIGURATION IS THE ONLY ONE NOW: `joint`, max-age 8, and max-move 1.25. On the
# 3dzef 5-fish clip it has zero identity switches, full coverage, 0.100 mm MPJPE, and 0.309 mm
# p99.


def _project(cam, points):
    """(n,3) world -> (n,2) pixels in one camera. Batched: this is the per-frame inner loop."""
    p = torch.as_tensor(np.asarray(points, np.float32)).reshape(-1, 1, 3)
    return project_points_torch([cam], p)[0, :, 0]


class CrossViewTracker:
    """Stateful across frames. One instance per group; `step` once per frame, in order.

    `max_move` is in box sides, exactly as in `link_rows`: real consecutive-frame box-centre
    displacement is p90 0.06-0.11 body lengths on the shipped multi-animal roots, so one full
    side has 10-16x headroom and rejects essentially nothing legitimate.

    `self.events` is a per-slot identity-EVENT log, always recorded: an append-only list of
    `{frame, slot, event, detail}` over `born`, `died`, `retired_duplicate`, `shielded` and
    `birth_refused`. Without it every question about a duplicate episode needs a monkeypatched
    traced re-run reproducing a whole GPU pass to learn what the first pass already knew. Always
    on because a few dict appends per frame are negligible behind the video decode this sits
    after, and an opt-in diagnostic is never on when the surprise happens. `frame` is
    TRACKER-LOCAL -- 0 at construction, +1 per `step()` -- which is the absolute frame within
    the group whenever one tracker lives for the whole group, as the `state` dict
    `associate_group` threads across block boundaries ensures. Recording cannot change `step()`'s
    returned arrays or `self.targets`; a byte-identity test asserts it. Writing the log into a
    prediction session (`identity_events.pq`) is a separate step, NOT built here.
    """

    def __init__(self, n_slots, max_res_px=30.0, max_move=1.25, max_age=8, min_views=2,
                 duplicate_radius=DUPLICATE_RADIUS, duplicate_persist=5, frame_base=0):
        """Create an empty tracker with `n_slots` rows.

        Inputs: n_slots -- number of animal rows (slots).
                max_res_px -- max reprojection residual (px) of every `associate` group,
                    births and matches alike.
                max_move -- per-frame box-centre displacement gate, in box sides; 1.25 is the
                    measured default.
                max_age -- frames without evidence before a slot is retired; 8 is the measured
                    default.
                min_views -- minimum cameras an `associate` group is built from (2, or 1 for
                    single-view rows, which have no point and are never kept as targets).
                duplicate_radius -- cross-view duplicate radius in box-side units; 0.75 is the
                    scale-free default (measured band, report 53).
                duplicate_persist -- consecutive in-band frames (5-35+ measured, contact <=4)
                    before a seated duplicate pair is retired; 5 is the measured default.
                frame_base -- the SOURCE frame of this tracker's own step 0; 0 (default) is
                    byte-identical to every run on record. A ranged run's tracker is first built
                    on an aligned lead-in batch below `--start-frame`, so its step count alone is
                    not the source frame -- events must add this back, or a bridge/consumer that
                    reads them as source-absolute silently misreads a ranged run.

        `self.targets` maps slot -> {'point': (3,) float32 tensor, 'age': int}. 'point' and 'age'
        keep their meaning throughout: the last TRIANGULATED point and frames since the last
        evidence. The measured 3dzef defaults are `max_age=8`, `max_move=1.25`.
        """
        self.n = int(n_slots)
        self._frame_base = int(frame_base)
        self.max_res_px = float(max_res_px)
        self.max_move = float(max_move)
        self.max_age = int(max_age)
        self.min_views = int(min_views)
        if duplicate_radius < 0:
            raise ValueError('duplicate_radius must be non-negative')
        self.duplicate_radius = float(duplicate_radius)
        self.duplicate_persist = int(duplicate_persist)
        self._residuals = {}
        self._dup_contact = {}
        self._dup_shield = set()
        self._dup_anchor = {}
        self._last_claims = {}
        self.targets = {}
        self.events = []
        self._t = -1

    def step(self, cgroup, boxes_per_cam, scores_per_cam):
        """Match, update, birth, retire. -> (boxes (S,C,4), scores (S,C), claimed (S,C)) numpy.

        Inputs:
            cgroup -- tracktail camera dicts for this frame.
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
            Every target holds a finite 3D point (a single-view `--min-views 1` row is written
            for its frame but never seated as a target, see `_birth`); a target without one
            would be left out of `slots`, aged here and retired at `max_age` like any other.
            The gate is the algorithm, not a tie-break:
            a pair beyond one box side is not the same animal, so it must be unavailable to
            Hungarian rather than merely expensive; a NaN box is unavailable, not unrankable
            (NaN affinity made `linear_sum_assignment` raise). The affinity uses the detection's
            own side (not the mean with the target's remembered side -- measured worse). A
            target matched to a group without a triangulated point keeps its 3D point (its age
            resets, so it does not expire); retiring it was measured worse (+2.72 mm MPJPE)
            because output boxes come from the claimed detection, never the reprojection.
        """
        self._t += 1
        C = len(cgroup)
        out = np.full((self.n, C, 4), np.nan, np.float32)
        sc = np.full((self.n, C), np.nan, np.float32)
        claimed_ix = np.full((self.n, C), -1, np.int32)
        centres = [box_centres(b) if b.numel() else b.new_zeros((0, 2)) for b in boxes_per_cam]
        sides = [box_sides(b) if b.numel() else b.new_zeros((0,)) for b in boxes_per_cam]
        slots = [s for s, t in sorted(self.targets.items())
                 if bool(torch.isfinite(t['point']).all())]
        for s, t in self.targets.items():
            if s not in slots:
                t['age'] += 1
        pts = (torch.stack([self.targets[s]['point'] for s in slots]) if slots else None)
        self._joint(cgroup, boxes_per_cam, scores_per_cam, centres, sides, slots, pts,
                   out, sc, claimed_ix)
        self._suppress_duplicate_targets(cgroup, out, sc, claimed_ix)
        for s in [s for s, t in self.targets.items() if t['age'] > self.max_age]:
            self.events.append({'frame': self._t + self._frame_base, 'slot': s, 'event': 'died',
                                'detail': {'age': self.targets[s]['age']}})
            del self.targets[s]
            self._residuals.pop(s, None)
            self._last_claims.pop(s, None)
            self._dup_anchor.pop(s, None)
            self._dup_shield.discard(s)
            for k in [k for k in self._dup_contact if s in k]:
                del self._dup_contact[k]
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
        blocks a birth.  The test uses ``min_shared=1``: a traced refire (report 53 follow-up)
        showed the double-fire's second box sometimes visible in only ONE camera at the moment
        of rebirth, which slipped through the 2-camera floor unrefused and then drifted back
        onto the same animal through ordinary tracking.  A false refusal here costs one birth's
        coverage, not an identity, so the laxer floor is safe only in this direction.
        """
        if not bool(torch.isfinite(group.get('point', torch.tensor(float('nan')))).all()):
            return False
        for s, t in self.targets.items():
            if s not in self._dup_shield or not bool(torch.isfinite(t['point']).all()):
                continue
            claimed = self._claim_evidence(out, s)
            if len(claimed) < 1:
                continue
            seated = {'point': t['point'], 'boxes': claimed}
            if _groups_are_duplicate(cgroup, seated, group, self.duplicate_radius,
                                     min_shared=1):
                return True
        for anchor in self._dup_anchor.values():
            if len(anchor['boxes']) < 1:
                continue
            if _groups_are_duplicate(cgroup, anchor, group, self.duplicate_radius,
                                     min_shared=1):
                return True
        return False

    def _suppress_duplicate_targets(self, cgroup, out, sc, claimed_ix):
        """Retire the weaker of two targets that have been duplicates for enough frames.

        Deliberately after `_joint`: matching can leave two targets on one animal (each matched
        to a different group of its detections), and a pair is only judged on
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
        for key in [k for k in self._dup_contact if not set(k) <= self.targets.keys()]:
            del self._dup_contact[key]
        for s in [s for s in self._dup_shield if s not in self.targets]:
            self._dup_shield.discard(s)
        self._refresh_anchors(cgroup, out)
        for key, count in sorted(self._dup_contact.items(), key=lambda kv: sorted(kv[0])):
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
            self.events.append({'frame': self._t + self._frame_base, 'slot': loser,
                                'event': 'retired_duplicate',
                                'detail': {'winner': winner, 'persisted': int(count),
                                           'loser_age': int(self.targets[loser].get('age', 0)),
                                           'winner_age': int(self.targets[winner].get('age', 0))}})
            self.events.append({'frame': self._t + self._frame_base, 'slot': winner,
                                'event': 'shielded',
                                'detail': {'loser': loser}})
            out[loser] = np.nan
            sc[loser] = np.nan
            claimed_ix[loser] = -1
            del self.targets[loser]
            self._residuals.pop(loser, None)
            self._dup_shield.discard(loser)
            self._dup_shield.add(winner)
            self._dup_anchor[winner] = {'point': self.targets[winner]['point'].clone(),
                                        'boxes': self._claim_evidence(out, winner), 'stale': 0}
            for k in [k for k in self._dup_contact if loser in k or winner in k]:
                del self._dup_contact[k]

    def _birth(self, s, g, out, sc, claimed_ix, boxes_per_cam, scores_per_cam):
        """Seat one `associate` group in free slot `s`.

        Inputs: s -- a slot with no target; g -- an `associate` group; the three output arrays;
            the frame's boxes and scores (`associate` saw the whole pool, so a group's member
            index IS the raw detection index).
        Outputs: None.
        Side effects: writes slot `s`'s row; creates `self.targets[s]` when `g` has a point.

        A `min_views = 1` single-view group has an all-NaN point BY DESIGN: no target can ever
        match it, so seating one would hold its slot dead for `max_age` frames while the same
        detection re-births into the next free slot every frame (one animal rotating through
        every slot). Its row is written for this frame only and the slot stays free, so the
        next frame re-seats it in the same lowest free slot.
        """
        if not bool(torch.isfinite(g['point']).all()):
            for c, j in g['members'].items():
                out[s, c] = boxes_per_cam[c][j].numpy()
                sc[s, c] = float(scores_per_cam[c][j])
                claimed_ix[s, c] = j
            return
        self.targets[s] = {'point': g['point'], 'age': 0}
        self._residuals[s] = float(g.get('residual', float('inf')))
        self.events.append({'frame': self._t + self._frame_base, 'slot': s, 'event': 'born',
                            'detail': {'cameras': sorted(g['members']),
                                       'residual': float(g.get('residual', float('inf')))}})
        for c, j in g['members'].items():
            out[s, c] = boxes_per_cam[c][j].numpy()
            sc[s, c] = float(scores_per_cam[c][j])
            claimed_ix[s, c] = j

    def _group_affinity(self, proj, centres, sides, i, g):
        """Lever 2's cost: how well slot `i`'s prediction explains candidate group `g`.

        Inputs: proj -- per-camera (n_slots,2) projections of every slot's predicted point;
            centres / sides -- per-camera detection centres and box sides; i -- the slot's row
            in `proj`; g -- one `associate` group.
        Outputs: a float affinity in [0, 1]; 0 means UNAVAILABLE to the Hungarian.
        Side effects: none.

        Deliberately NOT the 3D distance between the slot's point and the group's triangulated
        point, even though that is the obvious quantity: a millimetre gate is a second unit
        system, un-normalised by animal size, and this file's whole affinity design exists to
        avoid exactly that. So the slot point is projected into every camera the group holds and
        compared against that group's own detection centre in box sides, averaged over the
        cameras the group holds.

        The average is over cameras: one camera beyond the gate therefore does not by itself veto
        the match, but it does drag the mean; a pair whose mean gap exceeds one box side scores 0
        and is UNAVAILABLE, not merely expensive -- the gate is the algorithm, as ever.
        """
        num, den = 0.0, 0
        for c, j in g['members'].items():
            side = max(float(sides[c][j]), 1e-6)
            d = float(torch.linalg.norm(proj[c][i] - centres[c][j]))
            num += d / (self.max_move * side)
            den += 1
        if den <= 0 or not np.isfinite(num):
            return 0.0
        return float(np.clip(1.0 - num / den, 0.0, None))

    def _joint(self, cgroup, boxes_per_cam, scores_per_cam, centres, sides, slots, pts,
               out, sc, claimed_ix):
        """Cross-view groups first, then ONE Hungarian over slots x groups.

        Inputs: as `step`, plus per-camera detection `centres`/`sides`, the finite-point `slots`
            and their stacked `pts` (None when empty), and the output arrays to write in place.
        Outputs: None.
        Side effects: mutates `self.targets` and the output arrays.

        `associate` runs first, over the WHOLE detection pool: every group it returns has already
        been triangulated and had its reprojection residual gated at `max_res_px`, honouring
        `min_views`. Identity is then a choice among objects that are internally consistent by
        construction, which is the property the memoryless `--no-track` path has (0.081 mm vs
        0.291 mm for the deleted per-camera matcher on the fish clip). A detection `associate`
        leaves ungrouped is not claimed by any target.

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
                aff[i, k] = self._group_affinity(proj, centres, sides, i, g)
        if aff.any():
            aff = np.where(aff > 0,
                          aff * 16.0 + np.array([[_age_term(ages[s]) for s in slots]]).T, 0.0)
        taken, matched = set(), {}
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
                self.targets[s]['point'] = g['point']
                self._residuals[s] = float(g.get('residual', float('inf')))
            for c, j in g['members'].items():
                out[s, c] = boxes_per_cam[c][j].numpy()
                sc[s, c] = float(scores_per_cam[c][j])
                claimed_ix[s, c] = j
            self.targets[s]['age'] = 0

        free = [s for s in range(self.n) if s not in self.targets]
        unclaimed = [k for k in range(len(groups)) if k not in taken]
        gi = 0
        for s in free:
            while gi < len(unclaimed):
                k = unclaimed[gi]
                gi += 1
                g = groups[k]
                if self._birth_duplicates_target(cgroup, g, out):
                    self.events.append({'frame': self._t + self._frame_base, 'slot': s,
                                        'event': 'birth_refused',
                                        'detail': {'cameras': sorted(g.get('members', {}))}})
                    continue
                self._birth(s, g, out, sc, claimed_ix, boxes_per_cam, scores_per_cam)
                break
