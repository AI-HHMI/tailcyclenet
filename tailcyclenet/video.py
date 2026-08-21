"""THE ONE PLACE A VIDEO CONTAINER IS OPENED AND DECODED.

`dataset._read_video` and `adopt._probe` both used `decord` directly. **decord is unmaintained
(0.6.0, 2021) and its buffering is what made a long container unusable**: dmlc/decord#80 reports
that it loads the whole file into memory, and #197 is our exact symptom -- "RAM usage builds up
beyond 30 GB and the process gets killed" on a `get_batch` of a 4K video. A `--videos` run over
sixteen 21 GB containers peaked at **456 GB under `--max-ram 24`** and had to be killed at 9 GB
free on a shared node; 21 GB x 16 = 336 GB is the right order, and it explains why isolated
micro-benchmarks of `get_batch` were not reproducible -- how much of the file is resident depends
on how far the decoder has been driven.

**THE SWAP IS OUTPUT-NEUTRAL, AND THAT IS MEASURED RATHER THAN HOPED.** PyAV is BIT-IDENTICAL to
decord on every video root this repo has, sampled at BOTH ENDS of each container because the seek
path is what differs between backends:

    johnson raw + cut   h264   3208x2200   263,798 f   frames 120,060+     bit-identical
    3dpop               mpeg4  3840x2160       899 f   0-7, 886-893        bit-identical
    calms21             mpeg4  1024x570     23,810 f   0-7, 23,797-23,804  bit-identical

so no recorded number moves. `scratch/backend_parity_roots.py` is that check.

**FRAME ACCURACY IS BY CONSTRUCTION, NOT BY LUCK, AND THAT IS WHY THIS IS NOT OpenCV.**
`cv2.VideoCapture` + `CAP_PROP_POS_FRAMES` also passed on this container, but opencv#9053
documents it landing **8 to -3 frames off on MP4/AVC1**, and one root behaving is not evidence the
next will -- an off-by-one frame is a different picture of a moving animal and it is SILENT. The
recipe here seeks to the preceding KEYFRAME and then decodes forward COUNTING FRAMES, so the index
arithmetic is ours rather than the container's.

`TAILCYCLENET_VIDEO_BACKEND=decord` restores the old reader exactly, for bisecting.
"""
from __future__ import annotations

import os

import numpy as np

BACKENDS = ('pyav', 'decord')
# How far ahead it is worth DECODING to reach a wanted frame rather than seeking again. A seek
# lands on the preceding keyframe and this container's GOP is 180, so a second seek costs up to a
# GOP of decode anyway; below that, decoding forward is strictly cheaper than re-seeking.
_FORWARD_LIMIT = 256


def backend_name() -> str:
    name = os.environ.get('TAILCYCLENET_VIDEO_BACKEND', 'pyav').strip().lower()
    if name not in BACKENDS:
        raise ValueError(f'TAILCYCLENET_VIDEO_BACKEND={name!r} is not one of {BACKENDS}')
    return name


class PyAVReader:
    """Frame-accurate random access over one container, with bounded memory.

    `get_batch` mirrors `decord.VideoReader.get_batch` -- an arbitrary list of indices, returned
    in the ORDER ASKED FOR -- so it is a drop-in for the one call site that used it.
    """

    def __init__(self, path: str):
        import av

        self.path = str(path)
        self._c = av.open(self.path)
        self._st = self._c.streams.video[0]
        # `thread_type` is NOT settable once the codec is open, which a reader CACHE guarantees on
        # every hit after the first -- so it is set here, once, and never in `get_batch`.
        self._st.thread_type = 'AUTO'
        self._tb = self._st.time_base
        self._rate = self._st.average_rate
        self._pos = None                    # next frame index the decoder would yield, if known
        self._iter = None

    # -- the facts `adopt._probe` needs -------------------------------------------------
    def __len__(self) -> int:
        n = int(self._st.frames or 0)
        if n > 0:
            return n
        # A container with no frame count in its header. Derive from duration x rate rather than
        # guessing; `n_frames` is a PROMISE that every index in [0, T) decodes, so erring low is
        # the safe direction.
        dur = self._st.duration or (self._c.duration and self._c.duration / 1e6 / float(self._tb))
        return int(float(dur) * float(self._tb) * float(self._rate)) if dur else 0

    @property
    def fps(self) -> float:
        return float(self._rate)

    def frame_shape(self):
        return (int(self._st.codec_context.height), int(self._st.codec_context.width), 3)

    # -- decoding -----------------------------------------------------------------------
    def _index_of(self, frame) -> int:
        return int(round(float(frame.pts * self._tb) * float(self._rate)))

    def _seek(self, idx: int):
        self._c.seek(int(round(idx / float(self._rate) / float(self._tb))),
                     stream=self._st, backward=True, any_frame=False)
        self._iter = self._c.decode(self._st)
        self._pos = None

    def get_batch(self, indices) -> np.ndarray:
        want = [int(i) for i in indices]
        if not want:
            return np.empty((0, *self.frame_shape()), np.uint8)
        need = sorted(set(want))
        # CONTINUE RATHER THAN RE-SEEK where the decoder is already close enough. The window loop
        # walks a clip forwards, so consecutive calls are usually a short hop apart and a seek
        # would throw away a partly-decoded GOP.
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
            # One retry from an explicit seek: a container whose first keyframe sits after the
            # requested index, or a decoder that skipped a damaged frame.
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
        # In the ORDER ASKED FOR, repeats included -- decord's contract, and `read_frames` relies
        # on it when a clamp-padded window repeats its last frame.
        return np.asarray([got[i] for i in want])

    def close(self):
        try:
            self._c.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


class DecordReader:
    """The old path, kept for bisecting via `TAILCYCLENET_VIDEO_BACKEND=decord`."""

    def __init__(self, path: str):
        from decord import VideoReader

        self.path = str(path)
        self._vr = VideoReader(self.path, num_threads=1)

    def __len__(self):
        return len(self._vr)

    @property
    def fps(self):
        return float(self._vr.get_avg_fps())

    def frame_shape(self):
        return tuple(self._vr[0].shape)

    def get_batch(self, indices) -> np.ndarray:
        return self._vr.get_batch([int(i) for i in indices]).asnumpy()

    def close(self):
        self._vr = None


def open_reader(path: str):
    """One reader over one container. `num_threads=1` on decord for the reason it always was."""
    return PyAVReader(path) if backend_name() == 'pyav' else DecordReader(path)
