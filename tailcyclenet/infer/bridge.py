"""Post-hoc identity bridge: repair a slot permutation across a hard interaction.

THE PROBLEM. `CrossViewTracker`'s duplicate backstop retires a duplicate slot but cannot repair
the row PERMUTATION a hard interaction leaves behind: after two animals touch, slot 2 may carry
the animal slot 3 carried before, and every later frame inherits the swap. That is one `idsw` per
affected animal per episode, and report 54 measured it as the LAST remaining source on the only
two 3dpop clips that have any.

THE MECHANISM, `stable -> quarantine -> release`. Masking alone was measured and does NOT bridge
(a GT-informed upper bound reached idsw 0 only by holding 675 frames back, at +1,423 misses --
the rows never return to their old identity on their own). So: gap-join the tracker's own
`retired_duplicate`/`shielded` rows (`identity_events.pq`) into EPISODES whose slots are the
COMPONENT, the only rows that may be permuted; QUARANTINE the component as NaN from
`pre_event_windows` before the first event through the last, which is where the misses come
from; then RELEASE one one-to-one mapping scored by RECIPROCAL constant-velocity motion.

WHY HORIZON INVARIANCE AND NOT JUST CONFIDENCE. A single fixed-horizon argmax over the
best-vs-second-best margin is off-by-one fragile: on Sequence 59, moving the horizon by ONE
window moves the argmax from window 169 to 175 and flips the result between 0 and 2 switches.
The margin grows with elapsed motion rather than identity certainty, so "most confident" is not
"correct"; requiring agreement across horizons FIRST is stable to the horizon settings themselves.

WHAT THIS IS NOT. Not a solved problem. Measured (report 54): Sequence 59 16 -> 2 and Sequence 29
9 -> 4, label-free, at +552 and +218 misses. A better release EXISTS on Seq59 (window 169 gives
idsw 0, verified frame by frame) and FOUR label-free signals -- single-horizon margin, horizon
invariance, box shape, anchor invariance -- all failed to find it without labels. So this ships
OPT-IN, off by default, and any `idsw` from it is quoted with miss and coverage (eval rule 6).
Deliberately a POST-PASS, not a window-loop step: the release needs evidence from after the
episode, and a post-pass re-runs on stored output without paying for inference again.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .predictions import load_predictions

# Every per-row field a permutation must carry, so a repaired session stays self-consistent.
ROW_FIELDS = ('pred', 'pred2d', 'boxes', 'conf', 'box_agree', 'kpt_agree')

# The events that mark a hard interaction. Both are written by the duplicate backstop: the loser
# is `retired_duplicate`, the winner is `shielded`. A `born`/`birth_refused` row is churn, not an
# interaction, and including it would widen every episode to no purpose.
EPISODE_EVENTS = ('retired_duplicate', 'shielded')


@dataclass(frozen=True)
class BridgeConfig:
    """Tuning for one bridge pass. Defaults are the measured label-free arm from report 54.

    `max_event_gap` joins events into one episode (frames). `pre_event_windows` extends the
    quarantine backwards, since the interaction usually starts before the backstop notices.
    `recovery_windows` bounds how far past the episode a release may be chosen.
    `horizon_steps`/`horizon_stride` define the independent horizons a mapping must agree across.
    `min_margin` refuses a release whose mean margin is at or below it -- refusing leaves the
    component quarantined rather than guessing. `missing_cost` is what an unobservable pairing
    pays, in the same units as the pose (mm in 3D, px in 2D).

    `min_margin = 60.0` (raised from report 54's 1.0): on a 10-clip 3dpop Wave 1 population
    (dev/reports/57), 1.0 released a real permutation on Sequence42 at margin 43.08 on a clip
    whose baseline was PERFECTLY CLEAN, turning idsw 0 into 2 -- CLAUDE.md's own kill condition
    for the whole mechanism (damaging a clean clip). Every known-good outcome in the same
    population sits far above that (Sequence29's real repair: 140.21/220.86; Sequence10 and
    Sequence53's correct identity resolutions: 82.99/435.51; Sequence58's release: 1379.84).
    60.0 sits strictly between the one bad case and every good one, refuses Sequence42 at ZERO
    added miss cost, and changes nothing else measured so far.

    `interpolate` (plan section 2.3, default OFF): once a release mapping is known, the
    pre-episode and post-episode tracks are both identified, so the quarantined gap CAN be linearly
    interpolated instead of NaN'd -- converting misses into (possibly wrong) detections. Default
    OFF is a deliberate position, not an oversight: interpolated pose during a fight is likely
    wrong in a way indistinguishable downstream from a real observation, and this repo's culture
    is that a confident wrong answer is worse than an absence. A caller whose downstream metric is
    trajectory continuity rather than per-frame pose accuracy may want it on; MPJPE on the
    interpolated frames should be reported separately from the clip's normal error whenever it is.
    """

    max_event_gap: int = 64
    pre_event_windows: int = 1
    recovery_windows: int = 8
    horizon_steps: int = 4
    horizon_stride: int = 4
    min_margin: float = 60.0
    missing_cost: float = 200.0
    interpolate: bool = False

    def validate(self) -> None:
        """Refuse a nonsensical config by name rather than producing quiet nonsense."""
        if self.max_event_gap < 0:
            raise ValueError('max_event_gap must be >= 0')
        if self.pre_event_windows < 0:
            raise ValueError('pre_event_windows must be >= 0')
        if self.recovery_windows < 1:
            raise ValueError('recovery_windows must be >= 1')
        if self.horizon_steps < 1 or self.horizon_stride < 1:
            raise ValueError('horizon_steps and horizon_stride must be >= 1')
        if self.min_margin < 0:
            raise ValueError('min_margin must be >= 0')
        if not self.missing_cost > 0:
            raise ValueError('missing_cost must be > 0')


def owned_segments(windows: pd.DataFrame, gid: str, n_frames: int,
                   window_length: int) -> dict[int, np.ndarray]:
    """Frames each window OWNS after the seam rule, which is last-write-wins.

    A window writes `window_length` frames from its start, and a later window overwrites the
    overlap, so ownership is not the window's own extent -- it is what survives. Every downstream
    decision is per owned segment, because that is the unit the stored output actually contains.
    """
    cols = ['group_id', 'window', 'frame']
    missing = [c for c in cols if c not in windows.columns]
    if missing:
        raise ValueError(f'windows.pq lacks columns {missing}')
    mine = windows[windows.group_id.astype(str) == str(gid)][cols].drop_duplicates()
    mine = mine.sort_values('window')
    if mine.empty:
        raise ValueError(f'windows.pq has no rows for group {gid!r}')
    if mine['window'].duplicated().any():
        raise ValueError('one window id has more than one start frame')
    owner = np.full(int(n_frames), -1, np.int32)
    for row in mine.itertuples(index=False):
        start = int(row.frame)
        owner[start:min(start + int(window_length), int(n_frames))] = int(row.window)
    out = {int(w): np.flatnonzero(owner == w) for w in mine['window']}
    return {w: f for w, f in out.items() if len(f)}


def episodes(events: pd.DataFrame, gid: str, max_gap: int) -> list[dict]:
    """Gap-join this group's duplicate events into episodes, each with its own component.

    An episode is the unit a single permutation repair applies to. Two events closer than
    `max_gap` frames are one interaction; further apart they are two, and get separate repairs.
    """
    if events is None or events.empty:
        return []
    keep = events[(events.group_id.astype(str) == str(gid))
                  & events.event.isin(EPISODE_EVENTS)]
    if keep.empty:
        return []
    keep = keep.sort_values('frame')
    out, current, last = [], [], None
    for row in keep.itertuples(index=False):
        frame = int(row.frame)
        if last is not None and frame - last > int(max_gap):
            out.append(current)
            current = []
        current.append((frame, int(row.slot)))
        last = frame
    out.append(current)
    made = []
    for ep in out:
        component = tuple(sorted({slot for _, slot in ep}))
        if len(component) < 2:
            continue
        made.append({'component': component, 'first_frame': ep[0][0], 'last_frame': ep[-1][0]})
    return made


def _pose_cost(a: np.ndarray, b: np.ndarray, missing_cost: float) -> float:
    """Mean keypoint displacement over positions finite in BOTH, else the missing cost."""
    keep = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    if not keep.any():
        return float(missing_cost)
    return float(np.linalg.norm(a[keep] - b[keep], axis=-1).mean())


def _extrapolate(pred: np.ndarray, row: int, frames: np.ndarray, query: np.ndarray,
                 forward: bool) -> np.ndarray:
    """Constant-velocity extrapolation from a row's own last (or first) two observations.

    Deliberately constant-velocity and nothing richer: the interval being bridged is short, and a
    model with more parameters would fit the very ambiguity this is trying to resolve.
    """
    valid = np.isfinite(pred[row, frames]).all(-1).any(-1)
    if not valid.any():
        return np.full((len(query), pred.shape[2], pred.shape[3]), np.nan, pred.dtype)
    seen = frames[valid]
    end = int(seen[-1] if forward else seen[0])
    pose = pred[row, end]
    if len(seen) >= 2:
        other = int(seen[-2] if forward else seen[1])
        velocity = (pose - pred[row, other]) / max(1, abs(end - other))
        if not forward:
            velocity = -velocity
    else:
        velocity = np.zeros_like(pose)
    return pose[None] + (query[:, None, None] - end) * velocity[None]


def solve_mapping(pred: np.ndarray, component: tuple[int, ...], anchor: np.ndarray,
                  post: np.ndarray, missing_cost: float):
    """Best one-to-one map from logical rows to observed rows, by RECIPROCAL motion.

    Scored both directions -- pre-event extrapolated forward onto the post interval, and
    post-event extrapolated backward onto the anchor -- because a one-directional score is
    systematically biased toward whichever side has the longer clean run. Returns the best
    permutation, its cost, and the margin to the runner-up; the margin is what a release
    decision is allowed to trust.
    """
    scored = []
    for state in itertools.permutations(component):
        cost = 0.0
        for logical, local in zip(component, state):
            forward = _extrapolate(pred, logical, anchor, post, forward=True)
            backward = _extrapolate(pred, local, post, anchor, forward=False)
            cost += 0.5 * (_pose_cost(pred[local, post], forward, missing_cost)
                           + _pose_cost(pred[logical, anchor], backward, missing_cost))
        scored.append((cost, state))
    scored.sort(key=lambda x: (x[0], x[1]))
    best, state = scored[0]
    second = scored[1][0] if len(scored) > 1 else float('inf')
    return state, best, second - best


def _backward_agrees(pred: np.ndarray, segments: dict[int, np.ndarray], order: list[int],
                     component: tuple[int, ...], mapping: tuple[int, ...], release: int,
                     start: int, missing_cost: float, span: int = 8) -> bool | None:
    """Independent confirmation (plan §7.1, report 64): re-solve the SAME identity question
    anchored on a clean segment WELL AFTER the release (trusted because the forward pass already
    resolved everything up to there), resolving BACKWARD toward the pre-episode segment, and
    check the composed permutation (forward then backward) returns every row to itself.

    DIAGNOSTIC ONLY -- never gates a release; the caller stores this for later analysis. `None`
    means there was no room for a backward anchor (near the end of the clip), not disagreement.

    An involution (a 2-row swap, alone or beside fixed points) is its own inverse, so agreement
    there means `bwd_state == mapping` exactly; a longer cycle's true inverse is a DIFFERENT
    permutation from the forward one, and report 64 confirmed the composition check (not a raw
    equality) is what correctly distinguishes real agreement from coincidental restatement.
    """
    anchor_ws = [w for w in order if release + span <= w < release + span + 4]
    if not anchor_ws or start - 1 not in segments:
        return None
    bwd_anchor = np.concatenate([segments[w] for w in anchor_ws])
    bwd_post = segments[start - 1]
    bwd_state, _, _ = solve_mapping(pred, component, bwd_anchor, bwd_post, missing_cost)
    fwd_map = dict(zip(component, mapping))
    bwd_map = dict(zip(component, bwd_state))
    return all(bwd_map[fwd_map[c]] == c for c in component)


def plan_episode(pred: np.ndarray, segments: dict[int, np.ndarray], episode: dict,
                 cfg: BridgeConfig) -> dict | None:
    """Decide one episode's quarantine range, release window, and mapping. Reads no labels.

    Returns None when the episode cannot be bridged at all (no anchor before it, or no room
    after it) -- an un-bridgeable episode is left completely untouched rather than half-repaired.

    A release is trusted only under HORIZON INVARIANCE: the same mapping under every horizon, or
    the candidate is skipped. Among survivors the largest mean margin wins. A single fixed
    horizon is off-by-one fragile (moving it one window flips Sequence 59 between 0 and 2
    switches), which is why agreement is required before confidence is consulted at all.

    Every horizon-invariant candidate (not just the winner) is recorded on the returned plan as
    `all_candidates`, sorted by margin -- a pure diagnostic, gates nothing, same rule
    `backward_agrees` follows. Report 73: on the one clip this repo has independently verified
    frame-by-frame (Sequence 59), the CORRECT release is the runner-up by margin, not absent
    from the candidate set -- so the ranking rule, not candidate generation, is where the
    remaining error lives, and this field is what a future selection-rule attempt would need.

    Three outcomes, and the two non-repairing ones are deliberate:
    * `identity` -- the best map is the identity, so no rows were exchanged and there is nothing
      to repair. Quarantining anyway would spend coverage for no identity gain; skipping these
      cut Sequence 59 from 629 misses to 563 at unchanged `idsw`, since four of its six episodes
      resolve to the identity. A duplicate the backstop already handled correctly is this case,
      and it is the common one.
    * `refused` -- no horizon-invariant mapping cleared `min_margin`. The component stays
      quarantined and NO permutation is guessed, which is the whole point of having a margin.
    """
    if not segments:
        return None
    component = episode['component']
    if any(c >= pred.shape[0] for c in component):
        return None
    order = sorted(segments)
    lo, hi = order[0], order[-1]

    def owner(frame: int) -> int | None:
        """The owned window containing this frame, or None if no window kept it."""
        for w in order:
            f = segments[w]
            if f[0] <= frame <= f[-1]:
                return w
        return None

    first_w, last_w = owner(episode['first_frame']), owner(episode['last_frame'])
    if first_w is None or last_w is None:
        return None
    start = max(lo, first_w - cfg.pre_event_windows)
    if start - 1 < lo:
        return None
    stop = min(hi, last_w + cfg.recovery_windows) + 1
    candidates = [w for w in order if last_w < w < stop]
    if not candidates:
        return None
    anchor = segments[start - 1]
    horizons = sorted({min(hi + 1, stop + k * cfg.horizon_stride)
                       for k in range(cfg.horizon_steps)})
    best = None
    all_candidates = []
    for w in candidates:
        seen, margins = set(), []
        for h in horizons:
            frames = [segments[q] for q in order if w <= q < h]
            if not frames:
                continue
            state, _, margin = solve_mapping(pred, component, anchor,
                                             np.concatenate(frames), cfg.missing_cost)
            seen.add(state)
            margins.append(margin)
        if len(seen) != 1 or len(margins) != len(horizons):
            continue
        mean_margin = float(np.mean(margins))
        all_candidates.append({'release': w, 'mapping': list(next(iter(seen))),
                               'mean_margin': mean_margin})
        if mean_margin <= cfg.min_margin:
            continue
        if best is None or mean_margin > best['mean_margin']:
            best = {'release': w, 'mapping': next(iter(seen)), 'mean_margin': mean_margin}
    all_candidates.sort(key=lambda c: c['mean_margin'], reverse=True)
    windows = [w for w in order if start <= w < stop]
    if best is not None and tuple(best['mapping']) == tuple(component):
        return {'component': list(component), 'windows': [], 'release': None, 'mapping': None,
                'mean_margin': best['mean_margin'], 'refused': False, 'identity': True,
                'backward_agrees': None, 'all_candidates': all_candidates,
                'event_frames': [episode['first_frame'], episode['last_frame']]}
    if best is None:
        return {'component': list(component), 'windows': windows, 'release': None,
                'mapping': None, 'mean_margin': None, 'refused': True, 'backward_agrees': None,
                'all_candidates': all_candidates,
                'event_frames': [episode['first_frame'], episode['last_frame']]}
    backward_agrees = _backward_agrees(pred, segments, order, component, tuple(best['mapping']),
                                       best['release'], start, cfg.missing_cost)
    return {'component': list(component), 'windows': windows, 'release': best['release'],
            'mapping': list(best['mapping']), 'mean_margin': best['mean_margin'],
            'refused': False, 'backward_agrees': backward_agrees, 'all_candidates': all_candidates,
            'event_frames': [episode['first_frame'], episode['last_frame']]}


def _is_noop(plan: dict) -> bool:
    """An episode that touches nothing: no windows to quarantine and no mapping to apply."""
    return not plan.get('windows') and plan.get('release') is None


def _interpolate_gap(changed: np.ndarray, component: np.ndarray,
                     quarantine: np.ndarray) -> np.ndarray:
    """Section 2.3: linearly interpolate COMPONENT rows across `quarantine`'s frame span,
    in place, between the frame just before it and the frame just after it.

    Must run AFTER the release permutation has already written `changed`'s post-gap frames, so
    the right-hand anchor is the CORRECTLY IDENTIFIED position, not the pre-repair one -- an
    interpolation from the wrong anchor would connect two different animals' trajectories.

    Falls back to NaN (this frame's ordinary quarantine behaviour) wherever an anchor is missing:
    off the edge of the clip (no `lo`/`hi` frame exists), or the anchor itself is non-finite
    (the animal was not observed right at the boundary either) -- two real endpoints are
    required, not one, and per-ELEMENT (not per-row), so a partially-visible keypoint still
    interpolates the keypoints that DO have both anchors.

    The CALLER must gate this on `plan['release'] is not None` -- an in-bounds `hi` frame is not
    by itself proof of a verified anchor: a REFUSED episode still quarantines only up to
    `recovery_windows`, and the raw prediction just past that boundary is whatever the tracker
    originally wrote (possibly still the very swap this episode exists because of), never
    validated by a release decision. Interpolating toward that would invent a good-looking but
    wrong trajectory, worse than the NaN it would otherwise be.
    """
    if not len(quarantine):
        return changed
    lo, hi = int(quarantine.min()) - 1, int(quarantine.max()) + 1
    if lo < 0 or hi >= changed.shape[1]:
        changed[component[:, None], quarantine] = np.nan
        return changed
    left = changed[component, lo]
    right = changed[component, hi]
    alpha = (quarantine.astype(np.float64) - lo) / (hi - lo)
    a = alpha.reshape((1, -1) + (1,) * (left.ndim - 1))
    interp = left[:, None, ...] * (1.0 - a) + right[:, None, ...] * a
    valid = np.broadcast_to(np.isfinite(left[:, None, ...]) & np.isfinite(right[:, None, ...]),
                            interp.shape)
    changed[component[:, None], quarantine] = np.where(valid, interp, np.nan)
    return changed


def apply_plan(rows: dict, segments: dict[int, np.ndarray], plan: dict,
               n_frames: int, interpolate: bool = False) -> dict:
    """Apply one episode's plan to the row arrays: fill the quarantine, permute from the release.

    The permutation is applied to EVERY per-row field together, so a bridged session cannot end
    up with pose from one animal and boxes from another. Rows outside the component are
    byte-unchanged, and so is every frame before the quarantine.

    The release PERSISTS to the end of the clip, not just to the end of the bridged range: a
    resolved identity holds until something else changes it, and stopping at the range edge would
    re-introduce the very swap this just repaired, one window later.

    `interpolate` (section 2.3, default OFF at the caller) fills the quarantine by linear
    interpolation between its boundary frames instead of NaN-ing it; see `_interpolate_gap`.
    """
    component = np.asarray(plan['component'])
    quarantined = [w for w in plan['windows']
                   if plan['release'] is None or w < plan['release']]
    quarantine = (np.concatenate([segments[w] for w in quarantined])
                  if quarantined else np.empty(0, int))
    if plan['release'] is None:
        released = np.empty(0, int)
    else:
        tail = [segments[w] for w in plan['windows'] if w >= plan['release']]
        last = int(np.concatenate(tail)[-1]) if tail else -1
        released = (np.concatenate(tail + [np.arange(last + 1, n_frames, dtype=int)])
                    if tail else released)
    out = {}
    for key, value in rows.items():
        if key not in ROW_FIELDS or not isinstance(value, np.ndarray) or not value.size:
            out[key] = value
            continue
        changed = value.copy()
        if len(released):
            original = value[:, released]
            for logical, local in zip(plan['component'], plan['mapping']):
                changed[logical, released] = original[local]
        if len(quarantine):
            if interpolate and plan['release'] is not None:
                changed = _interpolate_gap(changed, component, quarantine)
            else:
                changed[component[:, None], quarantine] = np.nan
        out[key] = changed
    return out


def bridge_group(rows: dict, windows: pd.DataFrame, events: pd.DataFrame, gid: str,
                 n_frames: int, window_length: int, cfg: BridgeConfig) -> tuple[dict, list[dict]]:
    """Bridge every episode in one group, latest first. Returns the rows and the decisions.

    Episodes are applied LATEST FIRST so an earlier episode's quarantine and persistence cannot
    be overwritten by a later one's tail: each repair's persistence runs to the end of the clip,
    so they must be laid down in reverse temporal order to compose correctly.
    """
    cfg.validate()
    pred = rows.get('pred')
    if pred is None or not isinstance(pred, np.ndarray) or pred.ndim != 4:
        return rows, []
    segments = owned_segments(windows, gid, n_frames, window_length)
    found = episodes(events, gid, cfg.max_event_gap)
    decisions = []
    for ep in sorted(found, key=lambda e: e['first_frame'], reverse=True):
        plan = plan_episode(rows['pred'], segments, ep, cfg)
        if plan is None:
            decisions.append({'component': list(ep['component']), 'skipped': True,
                              'event_frames': [ep['first_frame'], ep['last_frame']]})
            continue
        if not _is_noop(plan):
            rows = apply_plan(rows, segments, plan, n_frames, interpolate=cfg.interpolate)
        decisions.append(plan)
    return rows, list(reversed(decisions))


def bridge_predictions(predictions: dict, path: Path, cfg: BridgeConfig,
                       window_length: int) -> dict:
    """Bridge a whole loaded prediction set in place-of-copy. Returns per-group decisions.

    `predictions` is mutated to hold the bridged rows, matching `load_predictions`' own shape so
    a caller can score or write it with no further translation. A session with no event log has
    no episode to bridge, which is a no-op rather than an error.
    """
    cfg.validate()
    path = Path(path)
    wpath, epath = path / 'windows.pq', path / 'identity_events.pq'
    if not wpath.exists():
        raise FileNotFoundError(
            f'{wpath} is required: the bridge decides per OWNED window segment, and without the '
            'window table it cannot tell which frames each window actually kept.')
    windows = pd.read_parquet(wpath)
    events = pd.read_parquet(epath) if epath.exists() else None
    if events is None or events.empty:
        return {gid: [] for gid in predictions}
    out = {}
    for gid, rows in predictions.items():
        pred = rows.get('pred')
        n_frames = int(pred.shape[1]) if isinstance(pred, np.ndarray) and pred.ndim == 4 else 0
        if not n_frames:
            out[gid] = []
            continue
        key = str(rows.get('group_id', gid))
        new_rows, decisions = bridge_group(rows, windows, events, key, n_frames,
                                           window_length, cfg)
        predictions[gid] = new_rows
        out[gid] = decisions
    return out


def _plan_frames(segments: dict[int, np.ndarray], plan: dict, n_frames: int):
    """The (quarantined, released) frame index arrays this plan implies."""
    quarantined = [w for w in plan['windows']
                   if plan['release'] is None or w < plan['release']]
    quarantine = (np.concatenate([segments[w] for w in quarantined])
                  if quarantined else np.empty(0, int))
    if plan['release'] is None:
        return quarantine, np.empty(0, int)
    tail = [segments[w] for w in plan['windows'] if w >= plan['release']]
    if not tail:
        return quarantine, np.empty(0, int)
    last = int(np.concatenate(tail)[-1])
    return quarantine, np.concatenate(tail + [np.arange(last + 1, n_frames, dtype=int)])


def boundary_fill_map(pred: np.ndarray, fill_pred: np.ndarray, frame: int,
                      component: tuple[int, ...]) -> dict[int, int] | None:
    """Map standard slots onto fill slots at one boundary frame, one-to-one by centre distance.

    `pred` is the standard session's (S, T, N, 3) pose array and `fill_pred` the fill pass's, both
    over the same clip. At `frame` the standard's component slots are still themselves -- the
    quarantine starts after -- so their centroids are the identification the map needs; the fill
    pass's rows are matched one-to-one by 3D centre distance (Hungarian on the rectangular
    matrix, each standard slot to its nearest distinct fill row). Returns {standard_slot:
    fill_slot}, or None when any centroid is missing or `frame` is outside the clip: a boundary
    the map cannot be computed at is a boundary the fill refuses rather than guesses at.
    """
    from scipy.optimize import linear_sum_assignment
    if frame < 0 or frame >= pred.shape[1] or frame >= fill_pred.shape[1]:
        return None
    rows = [int(i) for i in component]
    std = np.array([np.nanmean(pred[i, frame], axis=0) if np.isfinite(pred[i, frame]).any()
                    else np.full(pred.shape[-1], np.nan) for i in rows])
    fill_rows = [j for j in range(fill_pred.shape[0])
                 if np.isfinite(fill_pred[j, frame]).any()]
    if not fill_rows or not np.isfinite(std).all():
        return None
    fill_c = np.array([np.nanmean(fill_pred[j, frame], axis=0) for j in fill_rows])
    dist = np.linalg.norm(std[:, None, :] - fill_c[None, :, :], axis=-1)
    si, fj = linear_sum_assignment(dist)
    return {rows[a]: fill_rows[b] for a, b in zip(si, fj)}


# Columns a fill row BORROWS wholesale; everything else ((group_id, frame, animal_id) plus the
# structural bodypart/camera) stays the standard session's own.
_FILL_MEAS_COLS = frozenset({'status', 'x', 'y', 'z', 'score', 'score_logit',
                             'box_agree', 'x0', 'y0', 'x1', 'y1'})


def _fill_slice(ftable: pd.DataFrame, fill_name: str, table: pd.DataFrame, sel: np.ndarray,
                match_cols: list[str], replace_cols: list[str]) -> tuple[np.ndarray, dict]:
    """Fill values for the selected standard rows from the fill session's rows for `fill_name`.

    Matches on `match_cols` (frame plus the structural bodypart/camera), returns (got, vals):
    `got` is a bool array aligned to `sel`, True where the fill session has a row for the mapped
    fill animal at the same frame and bodypart/camera; `vals` maps each measurement column to an
    array aligned to the WHOLE table (undefined where not filled), so the caller assigns
    wholesale. A frame the fill pass did not observe stays for the quarantine to drop.
    """
    n = int(sel.sum())
    got = np.zeros(n, bool)
    if n == 0:
        return got, {}
    sub = ftable[ftable['animal_id'].astype(str) == str(fill_name)]
    if sub.empty:
        return got, {}
    keys = pd.DataFrame({'_row': np.flatnonzero(sel),
                         'frame': table['frame'].astype(np.int64).to_numpy()[sel]})
    for c in match_cols:
        if c != 'frame':
            keys[c] = table[c].to_numpy()[sel]
    merged = keys.merge(sub[match_cols + replace_cols], on=match_cols, how='left',
                        indicator=True).sort_values('_row')
    hit = (merged['_merge'] == 'both').to_numpy()
    got = hit
    vals = {c: merged[c].to_numpy()[hit] for c in replace_cols}
    return got, vals


def _fill_session(fill_dir: Path, fill_preds: dict) -> dict:
    """Per-group slice of the fill pass's session: its pred array, animal ids, and tables.

    `fill_preds` is `load_predictions(fill_dir)`'s own shape. The tables are read once and
    filtered to the group, so a lookup can never adopt another group's row (a prediction session
    may hold many groups, and the animal names repeat across them). A group the fill pass did not
    produce is absent from the result, which is a refusal at the caller, not a guess.
    """
    out = {}
    for gid, rows in fill_preds.items():
        key = str(rows.get('group_id', gid))
        tables = {}
        for stem in ('points3d', 'keypoints', 'instances'):
            f = fill_dir / f'{stem}.pq'
            if not f.exists():
                continue
            t = pd.read_parquet(f)
            t = t[t['group_id'].astype(str) == key]
            if not t.empty:
                tables[stem] = t
        out[key] = {'pred': rows['pred'], 'animal_ids': rows['animal_ids'], 'tables': tables}
    return out


def rewrite_tables(path: Path, gid: str, animal_ids, segments: dict[int, np.ndarray],
                   plans: list[dict], n_frames: int, fill: dict | None = None) -> dict[str, int]:
    """Apply plans to the stored parquet tables as row deletions and `animal_id` relabels.

    A permutation IS a relabel and a quarantine IS a deletion, so the tables are edited at row
    level rather than rebuilt from arrays. That matters: `write_block` needs `conf2d`, which does
    not survive a `load_predictions` round trip, so rebuilding would silently drop the per-camera
    scores. Editing rows preserves every column this module does not explicitly touch.

    Only rows of THIS group are considered, so a multi-group session keeps the others byte-exact.
    Plans are applied latest-first for the same reason `bridge_group` does: each release persists
    to the end of the clip.

    `fill` (plan section 6.4, default None): this group's slice of the fill pass's session --
    {'pred': array, 'animal_ids': array, 'tables': {stem: DataFrame}} -- with each real plan
    carrying a 'fill_map' ({standard slot: fill slot}) computed at the quarantine boundary. When
    set, a quarantined row with a fill observation is KEPT and its measurement columns are
    replaced by the mapped fill row's values wholesale (the fill pass's own observation of the
    mapped animal); a row with no fill observation stays dropped, exactly as the quarantine would
    leave it. (animal_id, frame, group_id) and the structural bodypart/camera are never replaced
    -- the row keeps the standard session's identity and only borrows the fill pass's
    measurements.
    """
    from ..format import DICT_COLS, write_table
    counts = {}
    ftables = (fill or {}).get('tables', {})
    fill_ids = (fill or {}).get('animal_ids')
    for stem in ('points3d', 'keypoints', 'instances'):
        f = path / f'{stem}.pq'
        if not f.exists():
            continue
        table = pd.read_parquet(f)
        if not {'animal_id', 'frame', 'group_id'}.issubset(table.columns):
            continue
        gcol = table['group_id'].astype(str).to_numpy()
        frame = table['frame'].astype(np.int64).to_numpy()
        animal = table['animal_id'].astype(str).to_numpy().astype(object)
        mine = gcol == str(gid)
        drop = np.zeros(len(frame), bool)
        new_animal = animal.copy()
        cols_np = {c: np.array(table[c].to_numpy()) for c in table.columns}
        match_cols = [c for c in table.columns
                      if c not in _FILL_MEAS_COLS and c not in ('group_id', 'animal_id')]
        replace_cols = [c for c in table.columns if c in _FILL_MEAS_COLS]
        for plan in sorted(plans, key=lambda p: (p['windows'] or [0])[0], reverse=True):
            if plan.get('skipped') or plan.get('identity'):
                continue
            names = [str(animal_ids[i]) for i in plan['component']]
            quarantine, released = _plan_frames(segments, plan, n_frames)
            fmap = plan.get('fill_map')
            if len(quarantine):
                inq = np.isin(frame, quarantine) & np.isin(animal, names) & mine
                here = inq.copy()
                if fmap and fill_ids is not None and ftables.get(stem) is not None:
                    ftable = ftables[stem]
                    for i in plan['component']:
                        if i not in fmap:
                            continue
                        sel = here & (animal == str(animal_ids[i]))
                        if not sel.any():
                            continue
                        got, vals = _fill_slice(ftable, str(fill_ids[fmap[i]]), table, sel,
                                                match_cols, replace_cols)
                        hit = np.flatnonzero(sel)[got]
                        here[hit] = False
                        for c, v in vals.items():
                            cols_np[c][hit] = v
                drop |= here
            if len(released) and plan['mapping'] is not None:
                inr = np.isin(frame, released) & mine
                source = new_animal.copy()
                for logical, local in zip(plan['component'], plan['mapping']):
                    sel = inr & (source == str(animal_ids[local]))
                    new_animal[sel] = str(animal_ids[logical])
        keep = ~drop
        out = {c: cols_np[c][keep] for c in table.columns}
        out['animal_id'] = new_animal[keep]
        write_table(f, out, dict_cols=DICT_COLS)
        counts[stem] = int(drop.sum())
    return counts


def bridge_session(path: Path, cfg: BridgeConfig, window_length: int,
                   fill_dir: Path | None = None) -> dict:
    """Bridge a written prediction session IN PLACE, editing its parquet tables. Reads no labels.

    This is what `--identity-bridge` runs after the writer closes. It re-reads the session it just
    wrote, decides per episode from the stored pose, and edits the tables. Re-running it on an
    already-bridged session is NOT a no-op and is not supported -- the events that defined the
    episodes are still in the log, but the rows they describe have already moved.

    `fill_dir` (plan section 6.4, default None): a fill pass's prediction session -- built by
    running `infer.py` again over the same clip with `--anchor carry --box-prompt none` -- whose
    quarantined frames fill this session's instead of being dropped. The fill session is bridged
    IN PLACE first (its own episodes repaired, so the row assignment the boundary map needs is
    stable through the quarantine), then read; each real plan's quarantine boundary frame maps
    this session's component slots onto the fill pass's rows one-to-one by centre distance, and a
    quarantined row the fill pass observed is kept with the fill pass's measurements. A session
    lacking the group, a boundary the map cannot be computed at, or a frame the fill pass did not
    observe is a refusal, not a guess: those rows stay dropped exactly as the quarantine leaves
    them.
    """
    cfg.validate()
    path = Path(path)
    if fill_dir is not None:
        fill_dir = Path(fill_dir)
        if not (fill_dir / 'windows.pq').exists():
            raise FileNotFoundError(
                f'{fill_dir} is required to carry windows.pq: the fill maps per OWNED window '
                'segment, and without the window table it cannot be bridged first.')
        bridge_session(fill_dir, cfg, window_length)
    predictions, _ = load_predictions(path)
    windows = pd.read_parquet(path / 'windows.pq')
    epath = path / 'identity_events.pq'
    if not epath.exists():
        return {}
    events = pd.read_parquet(epath)
    if events.empty:
        return {}
    fill = None
    if fill_dir is not None:
        fill_preds, _ = load_predictions(fill_dir)
        fill = _fill_session(fill_dir, fill_preds)
    out = {}
    for key, rows in predictions.items():
        pred = rows.get('pred')
        if not isinstance(pred, np.ndarray) or pred.ndim != 4:
            continue
        gid = str(rows.get('group_id', key))
        n_frames = int(pred.shape[1])
        segments = owned_segments(windows, gid, n_frames, window_length)
        plans = []
        for ep in sorted(episodes(events, gid, cfg.max_event_gap),
                         key=lambda e: e['first_frame'], reverse=True):
            plan = plan_episode(pred, segments, ep, cfg)
            plans.append(plan if plan is not None else
                         {'component': list(ep['component']), 'skipped': True, 'windows': [],
                          'event_frames': [ep['first_frame'], ep['last_frame']]})
        real = [p for p in plans if not p.get('skipped') and not p.get('identity')]
        if real:
            this_fill = fill.get(gid) if fill is not None else None
            if this_fill is not None:
                for p in real:
                    qframes, _ = _plan_frames(segments, p, n_frames)
                    p['fill_map'] = (boundary_fill_map(pred, this_fill['pred'],
                                                       int(qframes[0]) - 1, tuple(p['component']))
                                     if len(qframes) and int(qframes[0]) > 0 else None)
            rewrite_tables(path, gid, rows['animal_ids'], segments, real, n_frames,
                           fill=this_fill)
        out[key] = list(reversed(plans))
    return out


def summarise(decisions: dict) -> str:
    """One human line per bridged episode, for a run's own stdout."""
    lines = []
    for gid, plans in decisions.items():
        for p in plans:
            if p.get('skipped'):
                lines.append(f'{gid}: component {p["component"]} not bridgeable, left untouched')
            elif p.get('identity'):
                lines.append(f'{gid}: component {p["component"]} maps to itself, nothing to '
                             f'repair (margin {p["mean_margin"]:.2f})')
            elif p.get('refused'):
                lines.append(f'{gid}: component {p["component"]} REFUSED '
                             f'(no horizon-invariant mapping), quarantined '
                             f'{len(p["windows"])} window(s)')
            else:
                lines.append(f'{gid}: component {p["component"]} -> {p["mapping"]} '
                             f'released at window {p["release"]} '
                             f'(margin {p["mean_margin"]:.2f})')
    return '\n'.join(lines) if lines else 'identity bridge: no episode to bridge'


def config_provenance(cfg: BridgeConfig) -> list[tuple[str, float]]:
    """The bridge's own levers, for the prediction's provenance block.

    Unconditional membership, the same rule `_box_provenance`/`_identity_provenance` follow: an
    absent key would read as "not used" when it means "unknown", and that is what makes a
    partial record lie.
    """
    return [(f'bridge_{k}', float(v)) for k, v in asdict(cfg).items()]
