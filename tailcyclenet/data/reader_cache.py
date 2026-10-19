"""THE video reader cache: at most `size` open readers per process, sized from the RAM budget.

An OPEN reader retains memory (~0.053 GB/MP on PyAV, linear), so the cache size is a MEMORY BUDGET,
not a hint, sized per process from `--max-ram` (the only knob). It is built at the first video read,
because the size is a function of the rig. Sizing MUST NOT decode, stat or open anything (gotcha 10:
opening a reader in the parent deadlocks the forked loader workers).

NOT AN LRU: every window touches all `C` cameras in the same order, a cycle, and with capacity
`k < C` LRU always evicts the entry needed next (hit rate zero). Random eviction pays roughly the
missing fraction; pinning the first `k` would never serve a second group. Below the camera count
the slowdown is a threshold (5.1x on a 16-camera rig), so it warns. Measurements:
dev/reports/38_inference_ram_budget.md, dev/reports/40_pyav_read_path.md.

ONE LOCK PER CONTAINER around the decode (`read_lock_for`): readers are stateful (`get_batch`
seeks), so two threads on one container would interleave seeks; different files share no state and
decode concurrently (PyAV releases the GIL).
"""
from __future__ import annotations

import os
import random
import threading
import warnings

import torch

from .. import memory as _memory

_READER_GB_PER_MP = 0.06


def reader_cache_ram_gb(n_cams: int, wh, workers: int | None = None,
                        procs: int | None = None) -> float:
    """The `--max-ram` that would hold a reader for EVERY camera of this rig.

    The inverse of `reader_cache_size` (possible because the price is linear); undoes
    `FRACTION_READERS` and `DEFAULT_FRACTION` because `--max-ram` names the PROCESS, not the buffers.
    The process floor is on top: the ceiling that yields `need` GB of buffers is
    `max(need/fraction, need + floor)`. Nudged up by 1e-6 because `reader_cache_size` ends in `int()`
    and the exact figure would truncate one reader short of the rig.
    """
    k = _READER_GB_PER_MP * max(int(wh[0]) * int(wh[1]) / 1e6, 1e-3)
    share = max(workers or 1, 1) * max(int(procs or 1), 1)
    readers_gb = max(int(n_cams), 1) * k * share
    need = readers_gb / _memory.FRACTION_READERS
    floor = _memory.current().floor_gb
    exact = max(need / _memory.DEFAULT_FRACTION, need + floor)
    return exact * (1 + 1e-6)


def reader_cache_size(n_cams: int, wh, workers: int | None, ram_gb: float | None = None,
                       procs: int | None = None) -> int:
    """How many open readers this process may hold.

    Pure given `ram_gb`: reads the environment, `/proc`, `/sys` and a camera count, never a video.
    A single streaming process wants the WHOLE rig; a loader worker wants 4 (`ChunkShuffle.mix`).
    RAM is the constraint, per PROCESS, multiplied by the workers and by `procs` ranks. The budget is
    the job's (`memory.current()`), and the per-reader price is rounded UP so the estimate errs toward
    a smaller cache. A set `TAILCYCLENET_READER_CACHE` (deleted) raises.
    """
    if os.environ.get('TAILCYCLENET_READER_CACHE'):
        raise SystemExit('TAILCYCLENET_READER_CACHE is deleted: the reader cache is sized from '
                         'the RAM budget only. Unset it and use --max-ram / '
                         'TAILCYCLENET_MAX_RAM_GB instead.')
    if ram_gb is None:
        ram_gb = _memory.current().budget_gb
    if procs is None:
        procs = int(os.environ.get('TAILCYCLENET_LOCAL_WORLD_SIZE', '1') or 1)
    want = 4 if workers else max(int(n_cams), 4)
    k = _READER_GB_PER_MP * max(int(wh[0]) * int(wh[1]) / 1e6, 1e-3)
    share = max(workers or 1, 1) * max(int(procs), 1)
    budget_bytes = max(0.0, ram_gb * _memory.FRACTION_READERS) * _memory.GB
    per_reader_bytes = k * share * _memory.GB
    return _memory.fits(budget_bytes, per_reader_bytes, want, floor=1)


_readers = None
_cache_lock = threading.Lock()


class ReaderCache:
    """At most `maxsize` open readers, evicting a RANDOM entry on overflow (module docstring)."""

    def __init__(self, maxsize: int):
        """A cache of at most `maxsize` open readers. Seeded so eviction does not vary run to run."""
        self.maxsize = max(1, int(maxsize))
        self.hits = self.misses = 0
        self._d: dict[str, object] = {}
        self._rng = random.Random(0)

    def __call__(self, path: str):
        """The open reader for `path`, opening it and evicting randomly when full.

        Inputs: path -- the video file path.
        Outputs: the open reader object.
        Side effects: opens a container and may drop an existing reader's last reference --
        dropping the last reference frees the decoder's frame pool, which is why the cache is a
        memory budget and not a handle count.
        """
        got = self._d.get(path)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        if len(self._d) >= self.maxsize:
            del self._d[self._rng.choice(list(self._d))]
        from . import video

        got = self._d[path] = video.open_reader(path)
        return got


def reader(path: str, group, cam: str):
    """The cached `video.open_reader` for `path`, sized on first use from the rig `group` belongs to.

    FIRST VIDEO READ WINS: a process that later walks a wider rig keeps the size it picked, because
    resizing means dropping live readers (the memory clamp bounds it either way). `_cache_lock` covers
    both the late build and the lookup, since a rig's cameras decode concurrently; the DECODE is not
    under it. Warns, naming the `--max-ram` that would hold the rig, when the cache is below the camera
    count in a non-worker process.
    """
    global _readers
    with _cache_lock:
        if _readers is None:
            info = torch.utils.data.get_worker_info()
            rig = group.session.rig
            size = reader_cache_size(
                len(rig), rig.size(cam), None if info is None else info.num_workers)
            if info is None and size < len(rig):
                why = (f'--max-ram {reader_cache_ram_gb(len(rig), rig.size(cam)):.0f} would hold '
                       f'all {len(rig)}. Raise --max-ram / TAILCYCLENET_MAX_RAM_GB, or accept the '
                       'slowdown.')
                warnings.warn(
                    f'video reader cache is {size} for a {len(rig)}-camera rig: every window '
                    f'touches all {len(rig)} cameras, so this misses on nearly every call and '
                    f'reopens up to {len(rig) - size} container(s) per window. {why}',
                    stacklevel=2)
            _readers = ReaderCache(size)
        return _readers(path)


_lock_lock = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def read_lock_for(path: str) -> threading.Lock:
    """The one lock guarding reads of a given container path."""
    with _lock_lock:
        lk = _path_locks.get(path)
        if lk is None:
            lk = _path_locks[path] = threading.Lock()
        return lk
