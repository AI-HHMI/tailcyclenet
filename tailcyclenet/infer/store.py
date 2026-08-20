"""THE ONE PLACE INFERENCE DECODES A FRAME.

Every consumer -- the detector, refine pass 1, refine pass 2, and the next window's overlap --
wants the same `(camera, source frame)` pixels, and before this they each fetched their own. A
frame sat in three windows at the shipped `n_frames = 12 --overlap 8`, `--refine` asked for it
twice per window, and `detect_raw` had already decoded it once on its own pass: **seven decodes of
every frame-camera**, against a 44 ms 4K decode and a 0.86 ms forward (dev/reports/38 §5, 14).

So the store is not a cache in the "nice if it hits" sense. The block loop sizes itself so that
every frame a block needs FITS, and `run_blocks` evicts by window position -- exactly, since
`starts` is monotone and a frame belongs to the last window containing it (eval rule 11). A miss
is therefore a bug in the block sizing, not a cost to absorb, which is why there is no capacity
check on insert: the graceful degradation this replaces (`_frame_cap`, insert-what-fits) existed
because the old cache had to survive a budget too small to hold a window, and that case is now a
refusal with an arithmetic message instead of a silent 3x slowdown.

**It opens, stats and decodes nothing at construction** (gotcha 10): building one from a `Group`
touches no data path, so it is safe to create before any fork.
"""
from __future__ import annotations

import numpy as np

from ..dataset import read_frames


class FrameStore:
    """`(camera index, source frame) -> full decoded frame`, decoded once, dropped by window.

    `read` mirrors `dataset.read_frames(group, cam_name, frames, pool=)` on purpose -- it is a
    drop-in for both callers, `window.decode_crops` and `detect_raw(read=)`, so neither needs to
    know whether the pixels came from disk or from here.
    """

    def __init__(self, group, cam_names):
        self.group = group
        self.cam_names = list(cam_names)
        self._f: dict[tuple[int, int], np.ndarray] = {}
        self.hits = self.misses = 0

    def read(self, ci, cam_name=None, frames=(), pool=None, reduce=1):
        """The frames for one camera, decoding only what is not already held.

        `reduce` IS NOT STORED AND NOT SERVED FROM THE STORE. A reduced decode is libjpeg DCT
        decimation -- different pixels from a full decode downscaled -- and it is what the detector
        was trained on where it fires, so serving a full frame in its place would run the detector
        off its own sampling distribution, silently. It goes straight to `read_frames` and the
        result is not retained. No shipped detector sets it (`configs/detector.toml` ships
        `reduce = false`), so no shipped root pays the second decode.
        """
        name = cam_name if cam_name is not None else self.cam_names[ci]
        want = [int(t) for t in frames]
        if reduce != 1:
            return read_frames(self.group, name, np.asarray(want), reduce=reduce, pool=pool)
        need = [t for t in want if (ci, t) not in self._f]
        self.hits += len(want) - len(need)
        self.misses += len(need)
        if need:
            got = read_frames(self.group, name, np.asarray(need), pool=pool)
            for t, im in zip(need, got):
                self._f[ci, t] = im
        return [self._f.get((ci, t)) for t in want]

    def evict_below(self, t):
        """Drop every frame before `t`. Exact, not heuristic -- see the module docstring."""
        for k in [k for k in self._f if k[1] < t]:
            del self._f[k]

    def clear(self):
        self._f.clear()

    def __len__(self):
        return len(self._f)
