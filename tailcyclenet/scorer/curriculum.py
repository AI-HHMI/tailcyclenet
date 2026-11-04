"""The corruption CURRICULUM: a multiplier on every corruption magnitude, as a function of iteration.

A fixed corruption distribution gets "solved": on univ-rat-v2 the val `active_triplet_acc` peaked
near 0.957 at 40-50k and drifted to ~0.93 while the train loss kept falling, because most draws
are large and easy. The curriculum shrinks the magnitudes as training goes so the scorer keeps
learning distinctions at the size of real pose-model errors.

`scale(iteration)` is a PURE function of the global iteration, so the scale is reproduced by a
resume and agrees across ranks with no IPC: the train sampler hands each item its step ordinal
(`StepSampler`), and the worker derives the scale from it. That is why an ADAPTIVE schedule (a
scale stepped by a running train accuracy) is refused rather than implemented -- it would need a
collective decision broadcast into persistent loader workers and a scale history in the checkpoint.

The scale multiplies `mag_3d`/`mag_2d` (const_offset offsets, frame_noise sigmas, gradual_drift and
sinusoid amplitudes). Every tracktail generator is LINEAR in its magnitude and draws the same RNG
numbers whatever the magnitude, so a scaled draw is exactly `scale x` the unscaled one. It does
NOT scale `min_corrupt_px`/`max_clean_px` (the far/clean gate: what counts as corrupt at all) nor
the point-drop menu (missingness, not a magnitude).

An absent `[scorer.corruption.curriculum]` table (or `schedule = "none"` with `val_scale = 1`) is
byte-identical to a run before this module existed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tracktail.datasets.scorer_corruption import PointCorruptor

CURRICULUM_KEYS = ('schedule', 'start_scale', 'end_scale', 'warmup_iters', 'ramp_iters',
                   'val_scale')
SCHEDULES = ('none', 'linear', 'cosine')


@dataclass(frozen=True)
class Curriculum:
    """A magnitude schedule: `start_scale` for `warmup_iters`, then `linear`/`cosine` to
    `end_scale` over `ramp_iters`, then `end_scale`. `val_scale` is the FIXED scale of the val
    split's corruption (never scheduled: `checkpoint_best` must be selected on a fixed target)."""
    schedule: str = 'none'
    start_scale: float = 1.0
    end_scale: float = 1.0
    warmup_iters: int = 0
    ramp_iters: int = 0
    val_scale: float = 1.0

    @property
    def enabled(self) -> bool:
        """Whether the train scale varies with iteration (and so needs the step ordinal)."""
        return self.schedule != 'none'

    def scale(self, iteration: int) -> float:
        """The train magnitude multiplier at global `iteration` (monotone between the ends)."""
        if not self.enabled:
            return 1.0
        f = (float(iteration) - self.warmup_iters) / float(self.ramp_iters)
        f = min(max(f, 0.0), 1.0)
        if self.schedule == 'cosine':
            f = 0.5 * (1.0 - math.cos(math.pi * f))
        return float(self.start_scale + (self.end_scale - self.start_scale) * f)

    def contract(self) -> dict:
        """The schedule as a plain dict (for the checkpoint contract and the provenance)."""
        return {k: getattr(self, k) for k in CURRICULUM_KEYS}


def parse_curriculum(corr: dict | None) -> Curriculum:
    """Validate `[scorer.corruption.curriculum]` -> a `Curriculum`; absent -> the identity.

    Inputs: corr -- the `[scorer.corruption]` block (the `curriculum` sub-table is read).
    Outputs: a `Curriculum`.
    Side effects: none, or raises SystemExit naming an unknown key or a bad value.
    """
    raw = dict((corr or {}).get('curriculum') or {})
    where = '[scorer.corruption.curriculum]'
    unknown = set(raw) - set(CURRICULUM_KEYS)
    if unknown:
        raise SystemExit(f'{where}: unknown key(s) {sorted(unknown)}. Nothing reads them. Known '
                         f'keys: {list(CURRICULUM_KEYS)}')
    schedule = str(raw.get('schedule', 'none')).lower()
    if schedule == 'adaptive':
        raise SystemExit(
            f'{where}.schedule = "adaptive" is not implemented: the scale is a pure function of '
            'the iteration so workers, ranks and a resume agree without IPC. Use "linear" or '
            '"cosine".')
    if schedule not in SCHEDULES:
        raise SystemExit(f'{where}.schedule = {schedule!r} is not one of {SCHEDULES}')
    try:
        cur = Curriculum(schedule=schedule,
                         start_scale=float(raw.get('start_scale', 1.0)),
                         end_scale=float(raw.get('end_scale', 1.0)),
                         warmup_iters=int(raw.get('warmup_iters', 0)),
                         ramp_iters=int(raw.get('ramp_iters', 0)),
                         val_scale=float(raw.get('val_scale', 1.0)))
    except (TypeError, ValueError) as e:
        raise SystemExit(f'{where}: {e}') from None
    for name in ('start_scale', 'end_scale', 'val_scale'):
        value = getattr(cur, name)
        if not (math.isfinite(value) and value > 0.0):
            raise SystemExit(f'{where}.{name} must be a finite positive multiplier, got {value}')
    if cur.warmup_iters < 0:
        raise SystemExit(f'{where}.warmup_iters must be >= 0, got {cur.warmup_iters}')
    if cur.enabled and cur.ramp_iters < 1:
        raise SystemExit(f'{where}.ramp_iters must be >= 1 for schedule {schedule!r}')
    if not cur.enabled and any(k in raw for k in ('start_scale', 'end_scale', 'warmup_iters',
                                                   'ramp_iters')):
        raise SystemExit(f'{where}: schedule = "none" reads no start/end/warmup/ramp key; set '
                         'schedule = "linear" or "cosine", or delete them')
    return cur


def scale_corruptors(corruptors, scale: float):
    """The (dense_3d, dense_2d, sparse_3d, sparse_2d) quadruple with every magnitude x `scale`.

    `scale == 1.0` returns the SAME objects (the default-off path draws from the unchanged
    corruptors). Probabilities and the type menu are untouched.
    """
    if float(scale) == 1.0:
        return corruptors
    return tuple(PointCorruptor(dict(c.probs), {k: float(v) * float(scale)
                                                 for k, v in c.mags.items()})
                 for c in corruptors)
