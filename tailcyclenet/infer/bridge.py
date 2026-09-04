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
    """

    max_event_gap: int = 64
    pre_event_windows: int = 1
    recovery_windows: int = 8
    horizon_steps: int = 4
    horizon_stride: int = 4
    min_margin: float = 1.0
    missing_cost: float = 200.0

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


def plan_episode(pred: np.ndarray, segments: dict[int, np.ndarray], episode: dict,
                 cfg: BridgeConfig) -> dict | None:
    """Decide one episode's quarantine range, release window, and mapping. Reads no labels.

    Returns None when the episode cannot be bridged at all (no anchor before it, or no room
    after it) -- an un-bridgeable episode is left completely untouched rather than half-repaired.

    A release is trusted only under HORIZON INVARIANCE: the same mapping under every horizon, or
    the candidate is skipped. Among survivors the largest mean margin wins. A single fixed
    horizon is off-by-one fragile (moving it one window flips Sequence 59 between 0 and 2
    switches), which is why agreement is required before confidence is consulted at all.

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
        if mean_margin <= cfg.min_margin:
            continue
        if best is None or mean_margin > best['mean_margin']:
            best = {'release': w, 'mapping': next(iter(seen)), 'mean_margin': mean_margin}
    windows = [w for w in order if start <= w < stop]
    if best is not None and tuple(best['mapping']) == tuple(component):
        return {'component': list(component), 'windows': [], 'release': None, 'mapping': None,
                'mean_margin': best['mean_margin'], 'refused': False, 'identity': True,
                'event_frames': [episode['first_frame'], episode['last_frame']]}
    if best is None:
        return {'component': list(component), 'windows': windows, 'release': None,
                'mapping': None, 'mean_margin': None, 'refused': True,
                'event_frames': [episode['first_frame'], episode['last_frame']]}
    return {'component': list(component), 'windows': windows, 'release': best['release'],
            'mapping': list(best['mapping']), 'mean_margin': best['mean_margin'],
            'refused': False,
            'event_frames': [episode['first_frame'], episode['last_frame']]}


def _is_noop(plan: dict) -> bool:
    """An episode that touches nothing: no windows to quarantine and no mapping to apply."""
    return not plan.get('windows') and plan.get('release') is None


def apply_plan(rows: dict, segments: dict[int, np.ndarray], plan: dict,
               n_frames: int) -> dict:
    """Apply one episode's plan to the row arrays: NaN the quarantine, permute from the release.

    The permutation is applied to EVERY per-row field together, so a bridged session cannot end
    up with pose from one animal and boxes from another. Rows outside the component are
    byte-unchanged, and so is every frame before the quarantine.

    The release PERSISTS to the end of the clip, not just to the end of the bridged range: a
    resolved identity holds until something else changes it, and stopping at the range edge would
    re-introduce the very swap this just repaired, one window later.
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
        if len(quarantine):
            changed[component[:, None], quarantine] = np.nan
        if len(released):
            original = value[:, released]
            for logical, local in zip(plan['component'], plan['mapping']):
                changed[logical, released] = original[local]
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
            rows = apply_plan(rows, segments, plan, n_frames)
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


def rewrite_tables(path: Path, gid: str, animal_ids, segments: dict[int, np.ndarray],
                   plans: list[dict], n_frames: int) -> dict[str, int]:
    """Apply plans to the stored parquet tables as row deletions and `animal_id` relabels.

    A permutation IS a relabel and a quarantine IS a deletion, so the tables are edited at row
    level rather than rebuilt from arrays. That matters: `write_block` needs `conf2d`, which does
    not survive a `load_predictions` round trip, so rebuilding would silently drop the per-camera
    scores. Editing rows preserves every column this module does not explicitly touch.

    Only rows of THIS group are considered, so a multi-group session keeps the others byte-exact.
    Plans are applied latest-first for the same reason `bridge_group` does: each release persists
    to the end of the clip.
    """
    from ..format import DICT_COLS, write_table
    counts = {}
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
        for plan in sorted(plans, key=lambda p: (p['windows'] or [0])[0], reverse=True):
            if plan.get('skipped') or plan.get('identity'):
                continue
            names = [str(animal_ids[i]) for i in plan['component']]
            quarantine, released = _plan_frames(segments, plan, n_frames)
            if len(quarantine):
                inq = np.isin(frame, quarantine) & np.isin(animal, names) & mine
                drop |= inq
            if len(released) and plan['mapping'] is not None:
                inr = np.isin(frame, released) & mine
                source = new_animal.copy()
                for logical, local in zip(plan['component'], plan['mapping']):
                    sel = inr & (source == str(animal_ids[local]))
                    new_animal[sel] = str(animal_ids[logical])
        keep = ~drop
        out = {c: table[c].to_numpy()[keep] for c in table.columns}
        out['animal_id'] = new_animal[keep]
        write_table(f, out, dict_cols=DICT_COLS)
        counts[stem] = int(drop.sum())
    return counts


def bridge_session(path: Path, cfg: BridgeConfig, window_length: int) -> dict:
    """Bridge a written prediction session IN PLACE, editing its parquet tables. Reads no labels.

    This is what `--identity-bridge` runs after the writer closes. It re-reads the session it just
    wrote, decides per episode from the stored pose, and edits the tables. Re-running it on an
    already-bridged session is NOT a no-op and is not supported -- the events that defined the
    episodes are still in the log, but the rows they describe have already moved.
    """
    cfg.validate()
    path = Path(path)
    predictions, _ = load_predictions(path)
    windows = pd.read_parquet(path / 'windows.pq')
    epath = path / 'identity_events.pq'
    if not epath.exists():
        return {}
    events = pd.read_parquet(epath)
    if events.empty:
        return {}
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
            rewrite_tables(path, gid, rows['animal_ids'], segments, real, n_frames)
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
