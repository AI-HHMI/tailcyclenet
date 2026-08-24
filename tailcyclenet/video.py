"""THE ONE PLACE A VIDEO CONTAINER IS OPENED AND DECODED.

PyAV provides bounded-memory, frame-accurate decoding. Frame accuracy is by construction: seek to
 the preceding KEYFRAME and decode forward counting frames -- OpenCV's `CAP_PROP_POS_FRAMES` is
 documented 8 to -3 frames off on MP4/AVC1, silently.
"""
from __future__ import annotations

import os

import numpy as np

# libav's threading modes.
THREAD_TYPES = ('AUTO', 'FRAME', 'SLICE', 'NONE')
# Decode forward instead of re-seeking when the decoder is within this many frames: a seek lands
# on the preceding keyframe, so re-seeking would throw away a partly-decoded GOP.
_FORWARD_LIMIT = 256


class PyAVReader:
    """Frame-accurate random access over one container, with bounded memory.

    `get_batch` accepts an arbitrary list of indices and returns them in the ORDER ASKED FOR,
    including repeats.
    """

    def __init__(self, path: str):
        """Open one video container for frame-accurate random access.

        Inputs: path -- path to the video file.
        Side effects: opens the container and sets libav's threading mode from
        TAILCYCLENET_PYAV_THREADS.

        `thread_type` is NOT settable once the codec is open, which a reader CACHE guarantees on
        every hit after the first -- so it is set here, once, and never in `get_batch`. PyAV
        threads WITHIN a container, competing with the window loop's cross-container concurrency
        for the same cores; it is a PyAV ENUM, not a count. The rate is `guessed_rate`, NEVER
        `average_rate`: the index arithmetic is rate times pts, and the derived average drifts to
        an off-by-one frame when the declared duration is not exactly frames x period.
        """
        import av

        self.path = str(path)
        self._c = av.open(self.path)
        self._st = self._c.streams.video[0]
        _tt = os.environ.get('TAILCYCLENET_PYAV_THREADS', 'AUTO').strip().upper()
        if _tt not in THREAD_TYPES:
            raise ValueError(f'TAILCYCLENET_PYAV_THREADS={_tt!r} is not one of {THREAD_TYPES}. '
                             'It names libav\'s threading MODE, not a thread count.')
        self._st.thread_type = _tt
        self._tb = self._st.time_base
        self._rate = self._st.guessed_rate or self._st.average_rate
        self._pos = None
        self._iter = None

    # -- the facts `adopt._probe` needs -------------------------------------------------
    def __len__(self) -> int:
        """Frame count: from the header, or derived from duration x rate when absent.

        A header-less container derives from duration x rate rather than guessing; `n_frames` is
        a promise that every index in [0, T) decodes, so erring low is the safe direction.
        """
        n = int(self._st.frames or 0)
        if n > 0:
            return n
        dur = self._st.duration or (self._c.duration and self._c.duration / 1e6 / float(self._tb))
        return int(float(dur) * float(self._tb) * float(self._rate)) if dur else 0

    @property
    def fps(self) -> float:
        """The container's frame rate (guessed, not the drifting average)."""
        return float(self._rate)

    def frame_shape(self):
        """(height, width, 3) of decoded frames."""
        return (int(self._st.codec_context.height), int(self._st.codec_context.width), 3)

    # -- decoding -----------------------------------------------------------------------
    def _index_of(self, frame) -> int:
        """The frame index a decoded frame's pts corresponds to."""
        return int(round(float(frame.pts * self._tb) * float(self._rate)))

    def _seek(self, idx: int):
        """Seek to the keyframe at/before `idx` and restart decoding there.

        Inputs: idx -- target frame index.
        Side effects: replaces the decoder iterator and clears the position marker.
        """
        self._c.seek(int(round(idx / float(self._rate) / float(self._tb))),
                     stream=self._st, backward=True, any_frame=False)
        self._iter = self._c.decode(self._st)
        self._pos = None

    def get_batch(self, indices) -> np.ndarray:
        """Decode and return the frames at `indices`, in the order asked for.

        Inputs: indices -- iterable of frame indices; repeats are honoured.
        Outputs: (N, H, W, 3) uint8 array, one row per requested index.
        Side effects: advances the decoder, seeking once when the request is far ahead.

        The decoder continues rather than re-seeks when it is already close enough -- the loop
        walks the clip forwards, so consecutive calls are usually a short hop apart. Missing
        frames get ONE retry from an explicit seek (a container whose first keyframe sits after
        the requested index, or a decoder that skipped a damaged frame). The result preserves the
        ORDER ASKED FOR, including repeats -- `read_frames` relies on this when a clamp-padded
        window repeats its last frame.
        """
        want = [int(i) for i in indices]
        if not want:
            return np.empty((0, *self.frame_shape()), np.uint8)
        need = sorted(set(want))
        if not (self._iter is not None and self._pos is not None
                and self._pos <= need[0] <= self._pos + _FORWARD_LIMIT):
            self._seek(need[0])
        got: dict[int, np.ndarray] = {}
        target = set(need)
        last = need[-1]
        while True:
            try:
                frame = next(self._iter)
            except StopIteration:
                break
            idx = self._index_of(frame)
            self._pos = idx + 1
            if idx in target:
                got[idx] = frame.to_ndarray(format='rgb24')
            if idx >= last:
                break
        missing = [i for i in need if i not in got]
        if missing:
            self._seek(missing[0])
            for frame in self._iter:
                idx = self._index_of(frame)
                self._pos = idx + 1
                if idx in target and idx not in got:
                    got[idx] = frame.to_ndarray(format='rgb24')
                if idx >= last:
                    break
            missing = [i for i in need if i not in got]
        if missing:
            raise RuntimeError(
                f'{self.path}: frames {missing[:5]} did not decode (asked for '
                f'{need[0]}..{need[-1]} of {len(self)}). A frame index that does not decode is a '
                'broken promise about n_frames, not a pixel to substitute.')
        return np.asarray([got[i] for i in want])

    def close(self):
        """Close the underlying container; idempotent and swallows errors."""
        try:
            self._c.close()
        except Exception:
            pass

    def __del__(self):
        """Best-effort close on garbage collection."""
        self.close()


def open_reader(path: str):
    """Open one bounded-memory reader over one container."""
    return PyAVReader(path)
