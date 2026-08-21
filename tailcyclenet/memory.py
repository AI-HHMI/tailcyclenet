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
import warnings
from dataclasses import dataclass
from pathlib import Path

GB = float(1 << 30)

# What fraction of the headroom this repo's sized buffers may occupy. The rest is the model, the
# CUDA host-side allocations, the interpreter and glibc's own slack -- none of which is on this
# budget.
#
# **MEASURED UNDER A CGROUP CAP, which is the only measurement that answers the question this flag
# exists for** (dev/reports/38 §3.1: an unconstrained RSS peak is retained arena, not the working
# set). johnson, 120 frames x 16 cameras of 3208x2200, detector-to-pose, run inside
# `systemd-run --user --scope -p MemoryMax=<flag>G -p MemorySwapMax=0`:
#
#     fraction   --max-ram 16        --max-ram 24        times the cap was hit
#     0.60       10.96 GB / 63.1 s   11.88 GB / 61.1 s   0
#     0.75       12.00 GB / 58.5 s   13.30 GB / 59.9 s   0
#     0.85       11.89 GB / 57.8 s   17.36 GB / 59.4 s   0
#     0.95       11.90 GB / 58.3 s   17.30 GB / 60.6 s   0
#
# 0.85 is the KNEE: it spends the grant (17.4 of 24 GB, where 0.6 spent 11.9) and is slightly
# faster because the bigger block turns the detection lookahead on, while 0.95 buys nothing over
# it and leaves no slack for the terms that are not on this budget. `memory.events max` reads 0
# throughout, so the cap is never touched at any of these.
#
# **THE FLOOR IS 10.86 GB AND IT IS NOT ON THIS BUDGET EITHER.** Importing torch, initialising CUDA
# and loading the pose and detector checkpoints costs that much before a frame is decoded -- so
# `--max-ram` below ~12 is unusable at any fraction, and 8 and 10 are OOM-killed during startup
# rather than refused. That is why 0.6 looked safe at 16: the store was not the binding term
# there, the floor was.
#
# These are ONE workload's numbers. The three shares below are a ceiling this clip does not reach;
# a wider rig or a longer window could, so re-measure under a cap before assuming 0.85 travels.
DEFAULT_FRACTION = 0.85

# THE PARTITION, AND IT MUST SUM BELOW 1. Three consumers hold large buffers and each is sized
# independently, so a fraction that is generous alone overcommits in company.
#
# **WEIGHTED TOWARD FRAMES, NOT READERS, BECAUSE THE TWO COSTS HAVE DIFFERENT SHAPES.** Holding
# `n` open decord readers costs `~0.035/megapixel * n^2` GB -- quadratic, measured, see
# `dataset._reader_cache_size` -- while reopening one costs 41.5 ms against the ~300 ms it takes to
# decode the window anyway. The frame cache is LINEAR in what it holds and removes reader pressure
# outright, since a cached frame needs no reader at all. So a GB spent on frames buys strictly more
# than a GB spent on readers, and the split moved 0.50/0.20 -> 0.25/0.45 to say so.
#
# The quadratic law also means readers self-limit: 0.25 of a large budget still clamps at the rig's
# camera count, so an unconstrained host is unchanged, while a small budget now affords MORE
# readers than the old linear price allowed (5 rather than 2 for johnson at 16 GB).
#
# **THE SPLIT MOVED WHEN DETECTION AND POSE BECAME ONE PASS, AND THE OLD COMMENT HERE WAS FALSE.**
# It read "`FRACTION_DETECT` overlaps neither of the others in time -- detection finishes before
# the pose loop starts -- so it is deliberately the loosest of the three". That was true of a
# separate `detect_raw` pass over the whole group. It is not true now: the detector's letterbox
# buffers are live WHILE the store holds the block, so the two are simultaneous and the detector's
# share had to come down. Its own buffers are transient (`cams_flight x batch x 2` frames per
# unit) and it no longer decodes anything the store is not already holding, so 0.10 is enough.
#
# **`FRACTION_READERS` DID NOT MOVE, AND THE OBVIOUS ARGUMENT FOR MOVING IT IS WRONG.** "A stored
# frame needs no reader" is true of the number of READS and false of the number of open
# CONTAINERS: every block still touches every camera, so the reader cycle is the rig, exactly as
# before. Dropping this to 0.05 takes johnson from 16 readers to 10 on an unconstrained host, and
# report 38 measured 16 at 61 s against 11 at 149 s -- a 2.4x regression bought for memory the
# store does not need. The store takes what is left instead.
#
# The store gets the rest because it is the term that decides whether a run is possible at all --
# it must hold a whole window of full frames across the refine pass, and if it cannot,
# `run_blocks` refuses rather than degrading.
#
# **`FRACTION_READERS` MOVED 0.25 -> 0.35, AND A FRACTION IS THE WRONG SHAPE FOR THIS CONSUMER.**
# The reader cache is BOUNDED: it saturates at `n_cams` and cannot use another byte beyond that,
# unlike the store, which can always hold more frames. So this fraction is a CEILING that the
# reader need runs into, not an allocation it spends -- on a 16-camera 3208x2200 rig the whole rig
# costs 6 GB (`dataset._READER_GB_PER_MP`, measured on PyAV) and anything above that is left for
# nobody.
#
# The move is worth **5.1x on decode** and **15% end to end**, because the access pattern is a
# CYCLE and a cache one camera short of the rig takes almost no hits (report 40 §4): at 0.25 this
# rig needed `--max-ram 32` to reach its own camera count, and at 0.35 it reaches it at **24**.
#
# **THE STORE PAYS FOR IT, AND THE PRICE IS SMALLER THAN THE ARITHMETIC SUGGESTS.** `run_blocks`
# REFUSES when one window of frames does not fit, so taking share from the store raises the
# smallest `--max-ram` a rig can run at. Measured on johnson (16 x 3208x2200, one window 4.07 GB):
# **7.4 -> 8.1 GB**, not the ~15 -> ~17 a fixed-pipelining calculation predicts -- because at a
# tight budget `_pipeline_det` turns OFF FIRST and the store then gets the WHOLE share rather than
# half of it. That is `run_blocks`'s own stated design -- the LOOKAHEAD degrades, not the ceiling
# and not the floor -- absorbing the change.
#
# **AND BOTH FIGURES SIT BELOW THE 10.86 GB PROCESS FLOOR, so the regression is INERT**: a run at
# `--max-ram 8` is OOM-killed during startup by torch and the checkpoints whatever these fractions
# say. **Anything that lowers `FRACTION_STORE` further must re-check that floor on the widest rig
# in use** -- the margin is only inert while the store floor stays under the process floor, and a
# wider rig moves the store floor up.
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
    # DID A HUMAN STATE THIS NUMBER, or did we infer it from the machine? The two are different
    # permissions, not different values. An inferred budget is "what is lying around", and
    # spending it is how a 120-frame clip came to hold 40 GB of frames to do 10 GB of work -- so
    # the inferred case is sized from the WORK. `--max-ram 24` is a grant: the caller has said 24
    # is theirs to use, and being frugal with it buys them nothing they asked for.
    stated: bool = False

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

    # `--max-ram N` IS A CEILING ON THE PROCESS, NOT AN ALLOWANCE FOR THE BUFFERS, so the same
    # `fraction` applies to it as to a derived budget. The three shares below sum to 1.0 of
    # `budget_gb`, and everything outside them -- the model, CUDA's host-side allocations, the
    # result arrays, the interpreter, glibc's slack -- is real memory this number never sees.
    # Spending the full stated figure on buffers is how `--max-ram 16` peaked at 16.8 GB and
    # `--max-ram 8` at 10.5 GB: the flag was honoured exactly and the promise was still broken.
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


def result_array_gb(n_frames: int, n_cams: int, n_kpts: int, n_animals: int,
                    top_k: int, det_kpts: bool, dims: int = 3) -> dict[str, float]:
    """GB of RESULT arrays a group will allocate up front, by name. Nothing to do with pixels.

    **THIS IS THE TERM THE RAM BUDGET DOES NOT COVER, AND ON A LONG CLIP IT IS THE ONE THAT KILLS
    THE RUN.** Everything else in this module sizes buffers that are reused window after window;
    these are `np.full` allocations proportional to the WHOLE CLIP, made before a single frame is
    decoded, and no budget can shrink them because they are the answer being computed.

    A 200 fps hour -- 720,000 frames, which is an ordinary recording, not a pathological one --
    on a 16-camera 24-keypoint rig:

        detect_raw, top_k 2    6.6 GB   (6.2 of it keypoints)
        detect_raw, top_k 24  79.3 GB   (74.2 of it keypoints)
        run_group,  1 animal   0.4 GB
        run_group,  4 animals  1.6 GB

    `kpt` dominates because it is the only array left with BOTH a camera and a keypoint axis:
    `(top_k, T, C, K, 3)` float32. At top_k 24 it is 93% of the detection footprint, and it is
    allocated whenever the detector merely HAS a keypoint branch, whether or not anything reads it.
    `run_group`'s share fell 4x when `pred_tri` and `kpt_agree` left the output -- `kpt_agree` was
    the only pose-side array with a camera AND a keypoint axis, so it carried the same factor of K.

    Callers use this to fail loudly BEFORE the work rather than to be OOM-killed hours into it.
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

    **WITHOUT THIS, `--max-ram` BOUNDS THE BUFFERS AND NOT THE PROCESS, which is not what anyone
    means by a memory limit.** Measured on johnson (16 cameras, 3208x2200, 120 frames): at
    `--max-ram 8` the live buffers are a few GB and RSS still reached **123.4 GB**, because
    `free()` returns a block to the arena rather than to the kernel and nothing on an idle host
    ever forces the arena to shrink. The same run under a 24 GB cgroup sits at 10.5 GB -- the
    working set was always small; only the retention was large.

    So the flag has to do something the allocator would not do on its own. This is that something,
    and it is why a peak on an unconstrained host is a statement about glibc rather than about
    this repo.

    Cheap where it matters: `malloc_trim` walks the free lists and `madvise`s what it can, which is
    microseconds against the tens of milliseconds of decode that produced the garbage. Call it at a
    LOOP BOUNDARY (a finished window, a finished detection unit), never per allocation.

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

    **THE BUDGET WAS PURELY ADVISORY UNTIL THIS EXISTED, AND THAT IS HOW A `--max-ram 24` RUN
    REACHED 456 GB WITH NOTHING IN ITS OUTPUT SAYING SO.** Every consumer in this module sizes
    itself from the budget and no one ever checked the TOTAL, so a consumer that sits outside the
    partition -- or an allocation nobody attributed to a fraction at all -- is invisible until the
    node runs out of memory. On a shared host that is somebody else's job dying, not ours.

    **DIAGNOSIS, NOT ENFORCEMENT, AND DELIBERATELY SO.** It names the phase and gets out of the
    way. Killing the run would be worse than the disease: the peak may be retained arena rather
    than working set (see `trim`), the offending allocation is not always ours, and a run that is
    merely over budget is not a run that is WRONG. What it buys is that the next 456 GB arrives as
    a line of output naming the phase it grew in, instead of as a stopwatch and a dead node.

    **ONLY ON A STATED BUDGET.** An inferred budget is "what was lying around" (see `host_budget`),
    so exceeding it is not a broken promise -- it is the host being busier than it was at startup.
    A `--max-ram` or `TAILCYCLENET_MAX_RAM_GB` figure IS a promise.

    Once per process: this is called at loop boundaries, and a warning per block is noise that
    trains the reader to skip it.
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
    `floor`.

    **NEVER ABOVE `want`.** Every caller passes the value it uses today as `want`, so a bigger
    budget can only leave a configuration where it already was -- this module may shrink a buffer
    and may never grow one. A machine with more memory does not silently become a different
    recipe, and no existing measurement is invalidated by installing this.
    """
    if per_unit_bytes <= 0:
        return max(floor, want)
    return max(floor, min(int(want), int(budget_bytes // per_unit_bytes)))
