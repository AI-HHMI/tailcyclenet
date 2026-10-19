"""`H264Writer`: the one MP4 writer for rendered clips (H.264, yuv420p, +faststart)."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np


class H264Writer:
    """An MP4 writer for rendered clips: H.264 (libx264), yuv420p, `+faststart`.

    A drop-in for the `cv2.VideoWriter` calls the renderers used to make (`write(bgr)`,
    `release()`), whose `mp4v` output is MPEG-4 Part 2 -- which browsers and chat clients do not
    play. yuv420p needs even dimensions, so an odd-sized frame is edge-padded by one pixel rather
    than cropped: nothing drawn is lost. The first frame fixes the size when `size` is None.
    """

    def __init__(self, path, fps, size=None, crf: int = 20, preset: str = 'veryfast'):
        """Open `path` for writing.

        Inputs: path -- output .mp4 path.
                fps -- frame rate (float or Fraction; a float is rationalised).
                size -- (width, height); None = take it from the first frame.
                crf, preset -- libx264 quality / speed.
        Side effects: creates the file.
        """
        import av

        self.path = Path(path)
        self.rate = Fraction(fps).limit_denominator(1001) if not isinstance(fps, Fraction) else fps
        if self.rate <= 0:
            raise ValueError(f'{self.path}: fps must be positive, got {fps!r}')
        self._options = {'crf': str(int(crf)), 'preset': str(preset)}
        self._container = av.open(str(self.path), mode='w', format='mp4',
                                  options={'movflags': '+faststart'})
        self._stream = None
        self._size = None if size is None else (int(size[0]), int(size[1]))
        self.count = 0

    def _open_stream(self, w: int, h: int) -> None:
        """Add the H.264 stream at the even-padded size."""
        self._size = (w, h)
        st = self._container.add_stream('libx264', rate=self.rate)
        st.width, st.height = w + (w & 1), h + (h & 1)
        st.pix_fmt = 'yuv420p'
        st.options = dict(self._options)
        self._stream = st

    def write(self, bgr: np.ndarray) -> None:
        """Encode one (H, W, 3) uint8 BGR frame."""
        import av

        if self._container is None:
            raise RuntimeError(f'{self.path}: write after release')
        img = np.ascontiguousarray(bgr, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f'{self.path}: expected an (H, W, 3) BGR frame, got {img.shape}')
        h, w = img.shape[:2]
        if self._stream is None:
            self._open_stream(*(self._size or (w, h)))
        if (w, h) != self._size:
            raise ValueError(f'{self.path}: frame is {w}x{h}, stream is '
                             f'{self._size[0]}x{self._size[1]}')
        if (w | h) & 1:
            img = np.pad(img, ((0, h & 1), (0, w & 1), (0, 0)), mode='edge')
        frame = av.VideoFrame.from_ndarray(img, format='bgr24').reformat(format='yuv420p')
        frame.pts = self.count
        frame.time_base = 1 / self.rate
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.count += 1

    def release(self) -> None:
        """Flush the encoder and close the file. Idempotent."""
        if self._container is None:
            return
        try:
            if self._stream is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None

    def __enter__(self):
        """Context-manager entry."""
        return self

    def __exit__(self, *exc):
        """Release on exit."""
        self.release()
