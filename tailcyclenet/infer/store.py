"""THE one place inference decodes a frame.

Every consumer -- the detector, refine pass 1, refine pass 2, and the next window's overlap --
wants the same (camera, source frame) pixels. The store is not a "nice if it hits" cache: the
block loop sizes itself so every frame a block needs FITS, eviction is by window position (a
frame belongs to the last window containing it), and a miss is a bug in the block sizing -- which
is why there is no capacity check on insert. It opens, stats and decodes nothing at construction,
so it is safe to create before any fork.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..dataset import read_frames


class FrameStore:
    """`(camera index, source frame) -> full decoded frame`, decoded once, dropped by window.

    `read` mirrors `dataset.read_frames(group, cam_name, frames, pool=)` so both callers are
    drop-ins whether the pixels come from disk or from here.
    """

    def __init__(self, group, cam_names):
        """Bind the group and camera names this store decodes for; opens and decodes nothing.

        `_busy` holds the in-flight decode claims (see `read`); `_lock` guards both dicts, held
        only around bookkeeping so decodes still overlap. `decode_s` is wall seconds spent inside
        `read_frames`, summed across threads -- NOT a duration, because decodes overlap, so it is
        a measure of decode WORK, bounded by `cam_decode` when reported against wall clock.
        """
        self.group = group
        self.cam_names = list(cam_names)
        self._f: dict[tuple[int, int], np.ndarray] = {}
        self._busy: dict[tuple[int, int], threading.Event] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        self.decode_s = 0.0

    def read(self, ci, cam_name=None, frames=(), pool=None, reduce=1):
        """The frames for one camera, decoding only what is not already held.

        `reduce` is NOT stored or served from the store: a reduced decode is libjpeg DCT
        decimation, different pixels from a full decode, so serving a full frame in its place
        would run the detector off its sampling distribution. It goes straight to `read_frames`.
        The misses are CLAIMED before decoding: detection for the next block overlaps this
        block's decodes on the seam, and a claim stops both from decoding the same frames --
        whoever claims a key decodes it, everyone else waits on the event.
        """
        name = cam_name if cam_name is not None else self.cam_names[ci]
        want = [int(t) for t in frames]
        if reduce != 1:
            return read_frames(self.group, name, np.asarray(want), reduce=reduce, pool=pool)
        with self._lock:
            need, waits = [], []
            for t in want:
                if (ci, t) in self._f:
                    self.hits += 1
                elif (ci, t) in self._busy:
                    waits.append((ci, t))
                else:
                    self._busy[ci, t] = threading.Event()
                    need.append(t)
            self.misses += len(need)
        try:
            if need:
                t0 = time.perf_counter()
                got = read_frames(self.group, name, np.asarray(need), pool=pool)
                dt = time.perf_counter() - t0
                with self._lock:
                    self.decode_s += dt
                    for t, im in zip(need, got):
                        self._f[ci, t] = im
        finally:
            with self._lock:
                for t in need:
                    self._busy.pop((ci, t)).set()
        for k in waits:
            ev = self._busy.get(k)
            if ev is not None:
                ev.wait()
        with self._lock:
            return [self._f.get((ci, t)) for t in want]

    def evict_below(self, t):
        """Drop every frame before `t`. Exact, not heuristic -- see the module docstring."""
        with self._lock:
            for k in [k for k in self._f if k[1] < t]:
                del self._f[k]

    def clear(self):
        """Drop every held frame."""
        with self._lock:
            self._f.clear()

    def __len__(self):
        """The number of frames currently held."""
        with self._lock:
            return len(self._f)
