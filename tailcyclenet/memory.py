"""ONE host-RAM budget, read once per process, for every buffer this repo sizes by hand.

Inference on a video root was sized by nothing at all, and it OOMs. Measured at the shipped
defaults, scoring **120 frames** of one johnson session (16 cameras, 3208x2200): **122,911 MB**,
one megabyte under the L4 queue's 122,880 MB cap. Three multipliers stacked -- `detect_raw` held
`2 x batch x C` full frames as float32, `_reader_cache_size` sized the decord cache off the
MACHINE's memory rather than the job's, and the allocator never gave any of it back.

**THE CONSTRAINT IS NOT `MemTotal` AND NEVER WAS.** `os.sysconf('SC_PHYS_PAGES')` is what the host
has, not what this process may use. Under `systemd-run --user --scope -p MemoryMax=16G` it still
reports 503 GB here -- a 20x error, and that is exactly the LSF case this repo dies in. Three
things bound a job and the smallest wins:

- **the cgroup limit**, v2 `memory.max` or v1 `memory.limit_in_bytes`, minimised over EVERY
  ancestor -- a limit set on a parent slice binds a child that declares none;
- **LSF's own `LSB_CG_MEMLIMIT`/`LSB_MEMLIMIT`**, which on some configurations is tighter than the
  cgroup and is the number the scheduler actually kills on;
- **`MemAvailable`**, because a 503 GB host with 288 GB already resident by another user has
  215 GB, and `MemTotal` cannot see that.

THIS MODULE MUST NOT DECODE, STAT, OR OPEN ANY DATA PATH (gotcha 10). It reads `/proc/meminfo`,
`/sys/fs/cgroup` and the environment -- virtual files and parsed strings, no container, no CUDA,
nothing that can be inherited across a fork. `dataset._reader_cache_size` keeps taking `ram_gb`
as a parameter so it stays pure and testable on any host; this module is what the CALLER consults.

**EVERYTHING THE BUDGET CONTROLS MUST BE OUTPUT-NEUTRAL, and that is a hard invariant, not a
hope.** The budget depends on machine state, so two runs of one command can resolve it
differently -- which is only acceptable because every knob downstream of it changes wall clock and
never a predicted number: the decord cache size changes how often a container is reopened, the
detector batch changes how many frames ride in one forward (asserted byte-identical), and
`_CAM_DECODE` changes how many cameras decode at once. If a knob is ever added here that CAN move
a number, it does not belong on this budget -- it belongs in a config, where it is recorded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

GB = float(1 << 30)

# What fraction of the headroom this repo's sized buffers may occupy. The rest is the model, the
# CUDA host-side allocations, the npz accumulation, the interpreter and glibc's own slack -- none
# of which is on this budget. A starting point corrected by measurement, not a result.
DEFAULT_FRACTION = 0.6

# THE PARTITION, AND IT MUST SUM BELOW 1. Three consumers hold large buffers at the same time and
# each is sized independently, so a fraction that is generous on its own overcommits in company.
# `FRACTION_READERS` is applied by `dataset._reader_cache_size` to `min(limit, available)`
# DIRECTLY, not to `budget_gb` -- it keeps the 0.5 it always had so that an unconstrained host
# resolves to exactly the cache size it resolved to before this module existed, and no shipped
# measurement moves. The other two are fractions OF `budget_gb`, i.e. of `DEFAULT_FRACTION` of the
# headroom, so their combined footprint is 0.6 * 0.5 = 0.30 of headroom against readers' 0.50.
FRACTION_READERS = 0.5
FRACTION_DETECT = 0.3
FRACTION_CROPS = 0.2

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

    def share(self, fraction: float) -> float:
        """Bytes for one consumer's slice of the budget."""
        return max(0.0, self.budget_gb * float(fraction) * GB)

    def __str__(self) -> str:
        return (f'budget {self.budget_gb:.1f} GB of {self.limit_gb:.1f} GB '
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
    """(limit_bytes or None, current_bytes) -- the limit MINIMISED OVER EVERY ANCESTOR.

    A cgroup inherits its ancestors' limits without restating them: an LSF job may sit in a scope
    whose own `memory.max` is `max` while the slice above it caps the whole job. Reading only the
    leaf reports "unlimited" for a process that is very much limited, which is the failure this
    walk exists to prevent.
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

    Kept because this repo's jobs die on LSF specifically, and on some configurations the
    scheduler enforces a number the cgroup does not carry.
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
    # `MemAvailable`, which counts reclaimable page cache and so is the honest number rather than
    # `MemFree`. Take the smaller: a cgroup can be nearly empty on a host that is nearly full.
    avail = mi.get('MemAvailable', limit)
    if cg_limit is not None:
        avail = min(avail, max(0.0, cg_limit - cg_current))
    avail = min(avail, limit)

    if override_gb is not None and override_gb > 0:
        budget = float(override_gb) * GB
        source = f'--max-ram {override_gb:g}'
    elif os.environ.get(ENV_MAX_RAM):
        try:
            budget = float(os.environ[ENV_MAX_RAM]) * GB
            source = f'{ENV_MAX_RAM}={os.environ[ENV_MAX_RAM]}'
        except ValueError:
            budget = fraction * min(limit, avail)
    else:
        budget = fraction * min(limit, avail)

    # A budget above the hard cap is a request to be OOM-killed. Clamp, and say so.
    if budget > limit:
        source += f' (clamped to the {limit / GB:.1f} GB cap)'
        budget = limit
    return Budget(limit_gb=limit / GB, available_gb=avail / GB,
                  budget_gb=max(budget, 0.0) / GB, source=source)


_cached: Budget | None = None


def current(override_gb: float | None = None,
            fraction: float = DEFAULT_FRACTION) -> Budget:
    """The process-wide budget, resolved ONCE.

    Resolved once rather than per call so that two groups of one run are sized identically: an
    inference pass that shrank its own buffers halfway through because another user's job started
    would be reproducible in neither wall clock nor peak, and "the peak as a fraction of the
    budget" -- the only honest way to report this -- would have no single denominator.
    """
    global _cached
    if _cached is None or override_gb is not None:
        _cached = host_budget(override_gb, fraction)
    return _cached


def reset() -> None:
    """Drop the cached budget. For tests, and for a process that changes its own limits."""
    global _cached
    _cached = None


def fits(budget_bytes: float, per_unit_bytes: float, want: int, floor: int = 1) -> int:
    """How many units of `per_unit_bytes` fit in `budget_bytes`, capped at `want`, never below
    `floor`.

    **NEVER ABOVE `want`.** Every caller passes the value it uses today as `want`, so a bigger
    budget can only leave a configuration where it already was -- this module may shrink a buffer
    and may never grow one. A machine with more memory does not silently become a different
    recipe, and no existing measurement is invalidated by installing this.
    """
    if per_unit_bytes <= 0:
        return max(floor, want)
    return max(floor, min(int(want), int(budget_bytes // per_unit_bytes)))
