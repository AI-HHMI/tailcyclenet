"""ONE host-RAM budget, read once per process, for every buffer this repo sizes by hand.

The budget is the smallest of the cgroup limit (walked up EVERY ancestor), LSF's limits,
`MemAvailable` and `MemTotal` -- never `SC_PHYS_PAGES`, which is the MACHINE's memory. This
module must not decode, stat, or open any data path; it reads /proc, /sys and the environment.

Everything the budget controls must be OUTPUT-NEUTRAL: the budget depends on machine state, so
only wall clock -- never a predicted number -- may move with it.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

GB = float(1 << 30)

# What fraction of the headroom this repo's sized buffers may occupy. The rest is the model, the
# CUDA host-side allocations, the interpreter and glibc's own slack -- none of which is on this
# budget. Measured under a cgroup cap; 0.85 spends the grant without touching the cap.
DEFAULT_FRACTION = 0.85

# THE PARTITION, AND IT MUST SUM BELOW 1. Three consumers hold large buffers and each is sized
# independently, so a fraction that is generous alone overcommits in company. Readers saturate
# at `n_cams` (bounded); the store is the term that decides whether a run is possible at all;
# and the detector's letterbox buffers are live WHILE the store holds the block.
FRACTION_READERS = 0.35
FRACTION_DETECT = 0.10
FRACTION_STORE = 0.55

ENV_MAX_RAM = 'TAILCYCLENET_MAX_RAM_GB'

# A v1 cgroup writes a huge sentinel rather than a word for "unlimited".
_V1_UNLIMITED = 1 << 62


@dataclass(frozen=True)
class Budget:
    """`limit_gb` is the hard cap, `available_gb` the headroom now, `budget_gb` what we may use."""

    limit_gb: float
    available_gb: float
    budget_gb: float
    source: str
    # Did a human state this number, or did we infer it from the machine? An inferred budget is
    # "what is lying around" and is sized from the WORK; a stated one is a grant to spend.
    stated: bool = False
    # What the process already holds that no buffer share can touch -- torch, CUDA, the weights.
    # 0.0 until `rebudget` MEASURES it (it depends on the workload, not a constant).
    floor_gb: float = 0.0

    def share(self, fraction: float) -> float:
        """Bytes for one consumer's slice of the budget."""
        return max(0.0, self.budget_gb * float(fraction) * GB)

    def __str__(self) -> str:
        return (f'buffers {self.budget_gb:.1f} GB, cap {self.limit_gb:.1f} GB '
                f'({self.available_gb:.1f} GB free, {self.source})')


def _meminfo() -> dict[str, float]:
    """`/proc/meminfo` in bytes. Empty on a host that does not have it (macOS, some containers)."""
    out: dict[str, float] = {}
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return out
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        parts = rest.split()
        if parts:
            try:
                out[key] = float(parts[0]) * 1024.0
            except ValueError:
                pass
    return out


def _cgroup() -> tuple[float | None, float]:
    """(limit_bytes or None, current_bytes) -- the limit MINIMISED OVER EVERY ANCESTOR, because
    an LSF job may sit in a scope that declares no limit while the slice above it caps the job.
    """
    try:
        rel = Path('/proc/self/cgroup').read_text().strip().splitlines()[-1].split(':')[-1]
    except (OSError, IndexError):
        return None, 0.0
    base = Path('/sys/fs/cgroup')
    parts = [p for p in rel.split('/') if p]
    limit: float | None = None
    current = 0.0
    for i in range(len(parts) + 1):
        d = base.joinpath(*parts[:i])
        for name in ('memory.max', 'memory.limit_in_bytes'):
            try:
                raw = (d / name).read_text().strip()
            except OSError:
                continue
            if raw in ('max', ''):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if val >= _V1_UNLIMITED:
                continue
            limit = val if limit is None else min(limit, val)
        if i == len(parts):
            for name in ('memory.current', 'memory.usage_in_bytes'):
                try:
                    current = float((d / name).read_text().strip())
                    break
                except (OSError, ValueError):
                    continue
    return limit, current


def _lsf_limit() -> float | None:
    """LSF's own cap, in bytes. `LSB_CG_MEMLIMIT` is hex bytes; `LSB_MEMLIMIT` is decimal KB.
    Kept because the scheduler can enforce a number the cgroup does not carry.
    """
    raw = os.environ.get('LSB_CG_MEMLIMIT')
    if raw:
        try:
            return float(int(raw.strip(), 16))
        except ValueError:
            pass
    raw = os.environ.get('LSB_MEMLIMIT')
    if raw:
        try:
            return float(int(raw.strip())) * 1024.0
        except ValueError:
            pass
    return None


def host_budget(override_gb: float | None = None,
                fraction: float = DEFAULT_FRACTION) -> Budget:
    """Resolve the budget. `override_gb` beats the environment, which beats the derivation."""
    mi = _meminfo()
    total = mi.get('MemTotal')
    if total is None:
        try:
            total = float(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, AttributeError):
            total = 8.0 * GB                    # a floor, so a weird host degrades rather than dies
    cg_limit, cg_current = _cgroup()
    lsf = _lsf_limit()

    limit = total
    source = 'MemTotal'
    for name, val in (('cgroup', cg_limit), ('LSF', lsf)):
        if val is not None and val < limit:
            limit, source = val, name

    # HEADROOM, NOT THE CAP. Inside a cgroup what is left is `limit - current`; outside one it is
    # `MemAvailable`, which counts reclaimable page cache and so is the honest number. Take the
    # smaller: a cgroup can be nearly empty on a host that is nearly full.
    avail = mi.get('MemAvailable', limit)
    if cg_limit is not None:
        avail = min(avail, max(0.0, cg_limit - cg_current))
    avail = min(avail, limit)

    # `--max-ram N` IS A CEILING ON THE PROCESS, NOT AN ALLOWANCE FOR THE BUFFERS, so the same
    # `fraction` applies to it as to a derived budget.
    stated = False
    if override_gb is not None and override_gb > 0:
        budget = fraction * float(override_gb) * GB
        source = f'--max-ram {override_gb:g}'
        stated = True
    elif os.environ.get(ENV_MAX_RAM):
        try:
            budget = fraction * float(os.environ[ENV_MAX_RAM]) * GB
            source = f'{ENV_MAX_RAM}={os.environ[ENV_MAX_RAM]}'
            stated = True
        except ValueError:
            budget = fraction * min(limit, avail)
    else:
        budget = fraction * min(limit, avail)

    # A budget above the hard cap is a request to be OOM-killed. Clamp, and say so.
    if budget > limit:
        source += f' (clamped to the {limit / GB:.1f} GB cap)'
        budget = limit
    return Budget(limit_gb=limit / GB, available_gb=avail / GB,
                  budget_gb=max(budget, 0.0) / GB, source=source, stated=stated)


_cached: Budget | None = None


def current(override_gb: float | None = None,
            fraction: float = DEFAULT_FRACTION) -> Budget:
    """The process-wide budget, resolved ONCE, so two groups of one run are sized identically --
    a budget that shrank mid-run would be reproducible in neither wall clock nor peak.
    """
    global _cached
    if _cached is None or override_gb is not None:
        _cached = host_budget(override_gb, fraction)
    return _cached


def reset() -> None:
    """Drop the cached budget. For tests, and for a process that changes its own limits."""
    global _cached
    _cached = None


def rebudget(override_gb: float | None = None,
             fraction: float = DEFAULT_FRACTION) -> Budget:
    """Re-resolve the budget with THIS PROCESS'S OWN FLOOR SUBTRACTED. Call once, after the
    weights are loaded and before any buffer is sized.

    The floor (torch, CUDA, checkpoints) sat outside every budget; it is measured as the current
    RSS rather than assumed, since it depends on the workload. The budget is then the smaller of
    the fraction and `limit - floor`.
    """
    global _cached
    base = host_budget(override_gb, fraction)
    floor = rss_gb()
    if not (floor == floor) or not base.stated:
        # An INFERRED budget is already "what was lying around" and already contains the floor.
        _cached = base
        return base
    ceiling = base.budget_gb / fraction              # the figure the caller actually named
    _cached = replace(base, budget_gb=max(0.0, min(base.budget_gb, ceiling - floor)),
                      floor_gb=floor,
                      source=f'{base.source} minus a {floor:.1f} GB process floor')
    return _cached


def result_array_gb(n_frames: int, n_cams: int, n_kpts: int, n_animals: int,
                    top_k: int, det_kpts: bool, dims: int = 3) -> dict[str, float]:
    """GB of RESULT arrays a group will allocate up front, by name. Nothing to do with pixels.

    THE TERM THE RAM BUDGET DOES NOT COVER: these are whole-clip `np.full` allocations made
    before any decode, so no budget can shrink them. Callers use this to fail loudly BEFORE the
    work rather than be OOM-killed hours into it.
    """
    f4 = 4.0
    T, C, K, S, D = (max(int(n_frames), 0), max(int(n_cams), 1), max(int(n_kpts), 1),
                     max(int(n_animals), 1), max(int(top_k), 1))
    out = {
        'detect boxes': D * T * C * 4 * f4,
        'detect scores': D * T * C * f4,
        'detect keypoints': (D * T * C * K * 3 * f4) if det_kpts else 0.0,
        'pred': S * T * K * dims * f4,
        'conf': S * T * K * f4,
        'box_agree': S * T * C * f4,
    }
    return {k: v / GB for k, v in out.items() if v}


_libc = None
_TRIM_OFF = os.environ.get('TAILCYCLENET_NO_MALLOC_TRIM')


def trim() -> None:
    """Hand glibc's free heap back to the OS. Safe to call often; a no-op off glibc.

    Without it, `--max-ram` bounds the buffers and not the process -- `free()` returns blocks to
    the arena, not the kernel. Call at a LOOP BOUNDARY, never per allocation.
    `TAILCYCLENET_NO_MALLOC_TRIM=1` disables it, for measuring what it is worth.
    """
    global _libc
    if _TRIM_OFF:
        return
    if _libc is None:
        import ctypes
        try:
            _libc = ctypes.CDLL('libc.so.6')
            _libc.malloc_trim.argtypes = [ctypes.c_size_t]
        except (OSError, AttributeError):
            _libc = False
    if _libc is not False:
        try:
            _libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass


def rss_gb() -> float:
    """This process's CURRENT resident set, in GB. `/proc` only -- opens no data path."""
    return _proc_status_gb('VmRSS')


def peak_gb() -> float:
    """This process's HIGH-WATER resident set (`VmHWM`), in GB."""
    return _proc_status_gb('VmHWM')


def _proc_status_gb(key: str) -> float:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return float('nan')


_peak_warned = False


def check_peak(phase: str, budget: 'Budget | None' = None) -> float:
    """WARN ONCE if the process has blown through its own STATED ceiling. -> the peak in GB.

    Diagnosis, not enforcement: consumers size themselves from the budget and nothing checks the
    total, so an outside-partition allocation is invisible until the node OOMs. Only a STATED
    budget is a promise an inferred one is not; once per process, at loop boundaries.
    """
    global _peak_warned
    b = current() if budget is None else budget
    peak = peak_gb()
    if _peak_warned or not b.stated or not (peak == peak):
        return peak
    # Against the STATED figure, not the buffer share: `--max-ram` is a ceiling on the PROCESS.
    ceiling = b.budget_gb / DEFAULT_FRACTION
    if peak > ceiling:
        _peak_warned = True
        warnings.warn(
            f'RSS peaked at {peak:.1f} GB during {phase}, above the {ceiling:.1f} GB this run was '
            f'given ({b.source}). Every buffer this module sizes is derived from that figure, so '
            'something is allocating outside the budget -- report it with the phase named here. '
            'Under a cgroup cap this is an OOM kill, not a warning.', stacklevel=2)
    return peak


def fits(budget_bytes: float, per_unit_bytes: float, want: int, floor: int = 1) -> int:
    """How many units of `per_unit_bytes` fit in `budget_bytes`, capped at `want`, never below
    `floor`. NEVER ABOVE `want`: this module may shrink a buffer and may never grow one, so a
    bigger machine does not silently become a different recipe.
    """
    if per_unit_bytes <= 0:
        return max(floor, want)
    return max(floor, min(int(want), int(budget_bytes // per_unit_bytes)))
