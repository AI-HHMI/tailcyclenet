"""THE ONE PLACE A VIDEO CONTAINER IS OPENED AND DECODED.

PyAV provides bounded-memory, frame-accurate decoding. Frame accuracy is by construction: seek to
 the preceding KEYFRAME and decode forward counting frames -- OpenCV's `CAP_PROP_POS_FRAMES` is
 documented 8 to -3 frames off on MP4/AVC1, silently.

**A FRAME INDEX IS A RANK IN THE SORTED PRESENTATION-TIMESTAMP TABLE, NOT `pts x rate`.** Frame
`k` is the k-th frame in display order, whatever the time base, the pts grid or the frame spacing.
Rate arithmetic is wrong in both directions: `guessed_rate` is read off the time base, so a 20 fps
camera writing onto a 1/30 grid (pts 0,2,3,5,6,...) had every third index never decode; and the
derived `average_rate` drifts to an off-by-one part way through a `-vsync vfr` remux
(`tests/test_video.py` pins both). The table is built ONCE per file per process:

- **MP4/MOV: from the container's own sample index** (`stream.index_entries`), which libavformat
  already parsed from the header at open -- no packet data is read, so a 600,000-frame file costs
  ~0.1 s, not a pass over its 20+ GB. The index holds DECODE timestamps, so it is only trusted
  after a probe of the head and tail packets shows their presentation timestamps are members of
  the (shifted) table -- B-frame reordering with a constant delay passes, anything else falls
  through.
- **Everything else** (MKV/WebM cues index only keyframes, a failed probe, a count that disagrees
  with the header): **a demux-only packet scan** -- reads the whole file but decodes nothing, at
  disk speed -- **cached on disk** under `TAILCYCLENET_FRAME_TABLE_CACHE` (default
  `~/.cache/tailcyclenet/frame-tables`, `off` disables), keyed on (real path, size, mtime), so each
  file is scanned once, ever, not once per reader open.

Every decoded frame's pts is checked for EXACT membership in the table, so a table that is wrong
anywhere in the file is caught at the first frame it mislabels: a header-built table is then
replaced by a scan, a scanned one raises. A container whose packets carry no timestamps at all
keeps the old rate arithmetic (`table_source == 'rate'`).
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

# libav's threading modes.
THREAD_TYPES = ('AUTO', 'FRAME', 'SLICE', 'NONE')
# Decode forward instead of re-seeking when the decoder is within this many frames: a seek lands
# on the preceding keyframe, so re-seeking would throw away a partly-decoded GOP.
_FORWARD_LIMIT = 256
# Packets read at each end of the file to validate a header-built table.
_PROBE_PACKETS = 300
# In-process table cache bound, in frames across all files (8 bytes each).
_MEM_LIMIT_FRAMES = 32_000_000
_CACHE_ENV = 'TAILCYCLENET_FRAME_TABLE_CACHE'
# Bump when the on-disk table layout or its meaning changes.
_CACHE_VERSION = 1


class FrameTable(NamedTuple):
    """The frame index of one video stream.

    pts    -- sorted int64 presentation timestamps (stream time base), one per frame; frame k is
              the frame whose pts is pts[k].
    source -- 'index' (container sample index), 'scan' (demux pass) or 'cache' (a stored scan).
    """
    pts: np.ndarray
    source: str


class _TableMismatch(Exception):
    """A decoded frame's pts is not in the table."""


_TABLES: OrderedDict = OrderedDict()
_TABLES_LOCK = threading.Lock()


def _file_key(path: str, stream_index: int) -> tuple:
    """(real path, size, mtime_ns, stream): a table is valid exactly as long as the file is."""
    st = os.stat(path)
    return (os.path.realpath(path), st.st_size, st.st_mtime_ns, stream_index)


def _remember(key, table: FrameTable) -> None:
    """Store a table in the in-process LRU, evicting the oldest past `_MEM_LIMIT_FRAMES`."""
    with _TABLES_LOCK:
        _TABLES[key] = table
        _TABLES.move_to_end(key)
        total = sum(len(t.pts) for t in _TABLES.values())
        while total > _MEM_LIMIT_FRAMES and len(_TABLES) > 1:
            _, old = _TABLES.popitem(last=False)
            total -= len(old.pts)


def _recall(key) -> FrameTable | None:
    """The in-process table for `key`, or None."""
    with _TABLES_LOCK:
        got = _TABLES.get(key)
        if got is not None:
            _TABLES.move_to_end(key)
        return got


def _cache_dir() -> Path | None:
    """The on-disk table directory from TAILCYCLENET_FRAME_TABLE_CACHE, or None when off."""
    v = os.environ.get(_CACHE_ENV, '').strip()
    if v.lower() in ('off', '0', 'none', 'false'):
        return None
    return Path(v).expanduser() if v else Path.home() / '.cache' / 'tailcyclenet' / 'frame-tables'


def _cache_file(key) -> Path | None:
    """The on-disk table path for `key`, or None when the cache is off."""
    d = _cache_dir()
    if d is None:
        return None
    h = hashlib.sha1(repr((_CACHE_VERSION,) + key).encode()).hexdigest()
    return d / f'{Path(key[0]).name}.{h[:16]}.npy'


def _load_cached(key) -> np.ndarray | None:
    """A stored scanned table for `key`, or None if absent/unreadable."""
    f = _cache_file(key)
    if f is None or not f.is_file():
        return None
    try:
        pts = np.load(f, allow_pickle=False)
    except Exception:
        return None
    if pts.dtype != np.int64 or pts.ndim != 1:
        return None
    return pts


def _store_cached(key, pts: np.ndarray) -> None:
    """Atomic write (tmp + rename); a cache that cannot be written is not an error."""
    f = _cache_file(key)
    if f is None:
        return
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f'{f.name}.{os.getpid()}.{threading.get_ident()}.tmp')
        with open(tmp, 'wb') as fh:
            np.save(fh, pts, allow_pickle=False)
        os.replace(tmp, f)
    except OSError:
        pass


def _packet_ts(pkt):
    """A packet's presentation timestamp, or None. Discarded packets (edit lists) yield None."""
    if getattr(pkt, 'is_discard', False):
        return None
    return pkt.pts


def scan_table(path: str, stream_index: int = 0) -> np.ndarray | None:
    """Demux every packet of one video stream (no decode) and return its sorted pts, or None.

    Inputs: path -- video file; stream_index -- index among the container's VIDEO streams.
    Outputs: sorted unique int64 pts, or None when any packet carries no timestamp. The
    demuxer's empty flush packet and edit-list discards are skipped.
    Cost: one sequential read of the whole file.
    """
    import av

    out = []
    with av.open(path) as c:
        st = c.streams.video[stream_index]
        for pkt in c.demux(st):
            if pkt.size == 0 and pkt.pts is None and pkt.dts is None:
                continue
            if getattr(pkt, 'is_discard', False):
                continue
            if pkt.pts is None:
                return None
            out.append(pkt.pts)
    pts = np.unique(np.asarray(out, np.int64))
    return pts


def _index_table(container, stream) -> np.ndarray | None:
    """Presentation timestamps from the container's own sample index, validated, or None.

    The index is libavformat's (decode-order) sample table, complete for MP4/MOV and built at
    open from the header. Accepted only when (a) it lists as many samples as the header's frame
    count, and (b) the presentation timestamps of the first and last `_PROBE_PACKETS` packets are
    all members of the table shifted by one constant (0 unless B-frames delay presentation).
    """
    entries = getattr(stream, 'index_entries', None)
    if entries is None:
        return None
    n_hdr = int(stream.frames or 0)
    try:
        n_all = len(entries)
    except TypeError:
        return None
    if n_all == 0 or n_hdr <= 0 or n_all != n_hdr:
        return None
    ts = np.fromiter((e.timestamp for e in entries if not e.is_discard), np.int64)
    if len(ts) == 0:
        return None
    ts = np.unique(ts)

    def probe():
        """The pts of up to `_PROBE_PACKETS` packets from the current position."""
        got = []
        for pkt in container.demux(stream):
            p = _packet_ts(pkt)
            if p is not None:
                got.append(p)
            if len(got) >= _PROBE_PACKETS:
                break
        return np.asarray(got, np.int64)

    try:
        container.seek(int(ts[0]), stream=stream, backward=True, any_frame=False)
        head = probe()
        tail_at = ts[max(0, len(ts) - _PROBE_PACKETS)]
        container.seek(int(tail_at), stream=stream, backward=True, any_frame=False)
        tail = probe()
    except Exception:
        return None
    if len(head) == 0:
        return None
    shift = int(head.min() - ts[0])
    pts = ts + shift
    for probe_pts in (head, tail):
        k = np.searchsorted(pts, probe_pts)
        ok = (k < len(pts)) & (pts[np.minimum(k, len(pts) - 1)] == probe_pts)
        if not ok.all():
            return None
    return pts


def frame_table(path: str, stream_index: int = 0, container=None, *,
                rescan: bool = False) -> FrameTable | None:
    """The frame table of one video stream, from the cheapest source that is verifiably right.

    Inputs: path -- video file; stream_index -- which video stream; container -- an already-open
    PyAV container on `path` (reused for the header route, left at an arbitrary position);
    rescan -- skip the header route and the caches and demux the file afresh.
    Outputs: a FrameTable, or None when the packets carry no timestamps.
    Side effects: fills the in-process cache and, for a scan, the on-disk cache. The scan runs
    under `_scan_lock` and re-checks the disk cache after acquiring it, so a file another process
    scanned while this one waited is read, not rescanned.
    """
    key = _file_key(path, stream_index)
    if not rescan:
        got = _recall(key)
        if got is not None:
            return got
        if container is not None:
            pts = _index_table(container, container.streams.video[stream_index])
            if pts is not None:
                table = FrameTable(pts, 'index')
                _remember(key, table)
                return table
        pts = _load_cached(key)
        if pts is not None:
            table = FrameTable(pts, 'cache')
            _remember(key, table)
            return table
    with _scan_lock(key):
        pts = None if rescan else _load_cached(key)
        source = 'cache'
        if pts is None:
            pts = scan_table(path, stream_index)
            if pts is None:
                return None
            _store_cached(key, pts)
            source = 'scan'
    table = FrameTable(pts, source)
    _remember(key, table)
    return table


class _scan_lock:
    """An exclusive flock beside the cache file, so N processes opening one long file for the
    first time scan it ONCE and the rest read the stored table. A no-op without a cache."""

    def __init__(self, key):
        """Resolve the lock path for `key` (None when the cache is off)."""
        self._f = _cache_file(key)
        self._fh = None

    def __enter__(self):
        """Block until this process holds the lock; a lock that cannot be taken is skipped."""
        if self._f is None:
            return self
        try:
            import fcntl

            self._f.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._f.with_name(self._f.name + '.lock'), 'a')
            fcntl.flock(self._fh, fcntl.LOCK_EX)
        except (OSError, ImportError):
            self._fh = None
        return self

    def __exit__(self, *exc):
        """Release and close the lock file."""
        if self._fh is not None:
            try:
                import fcntl

                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
        return False


class PyAVReader:
    """Frame-accurate random access over one container, with bounded memory.

    `get_batch` accepts an arbitrary list of indices and returns them in the ORDER ASKED FOR,
    including repeats.
    """

    def __init__(self, path: str):
        """Open one video container for frame-accurate random access.

        Inputs: path -- path to the video file.
        Side effects: opens the container, sets libav's threading mode from
        TAILCYCLENET_PYAV_THREADS, and builds (or recalls) the frame table.

        `thread_type` is NOT settable once the codec is open, which a reader CACHE guarantees on
        every hit after the first -- so it is set here, once, and never in `get_batch`. PyAV
        threads WITHIN a container, competing with the window loop's cross-container concurrency
        for the same cores; it is a PyAV ENUM, not a count.
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
        self._set_table(frame_table(self.path, 0, self._c))

    def _set_table(self, table: FrameTable | None):
        """Install a frame table and invalidate the decoder position."""
        self._table = table
        self._tab = None if table is None else table.pts
        self._pos = None
        self._iter = None

    @property
    def table_source(self) -> str:
        """Where the frame index came from: 'index', 'scan', 'cache', or 'rate' (no timestamps)."""
        return 'rate' if self._table is None else self._table.source

    # -- the facts `adopt._probe` needs -------------------------------------------------
    def __len__(self) -> int:
        """Frame count: the table's length, which is exactly the set of indices that exist.

        Without a table, from the header or derived from duration x rate; `n_frames` is a promise
        that every index in [0, T) decodes, so erring low is the safe direction.
        """
        if self._tab is not None:
            return int(len(self._tab))
        n = int(self._st.frames or 0)
        if n > 0:
            return n
        dur = self._st.duration or (self._c.duration and self._c.duration / 1e6 / float(self._tb))
        return int(float(dur) * float(self._tb) * float(self._rate)) if dur else 0

    @property
    def fps(self) -> float:
        """Frames per second: the declared rate when the timestamps agree with it (to 0.1%), else
        the rate the timestamps actually carry -- (frames - 1) / (last pts - first pts)."""
        declared = float(self._rate) if self._rate else 0.0
        tab = self._tab
        if tab is None or len(tab) < 2 or tab[-1] == tab[0]:
            return declared
        actual = float(Fraction(len(tab) - 1) / (Fraction(int(tab[-1] - tab[0])) * self._tb))
        if declared > 0 and abs(actual / declared - 1.0) < 1e-3:
            return declared
        return actual

    def frame_times(self) -> np.ndarray:
        """Presentation time in seconds of every frame index, float64 (stream clock, not zeroed).

        This is what to sync long recordings on: a dropped frame is a gap here, invisible in the
        index."""
        if self._tab is None:
            return np.arange(len(self), dtype=np.float64) / float(self._rate)
        return self._tab.astype(np.float64) * float(self._tb)

    def frame_shape(self):
        """(height, width, 3) of decoded frames."""
        return (int(self._st.codec_context.height), int(self._st.codec_context.width), 3)

    # -- decoding -----------------------------------------------------------------------
    def _index_of(self, frame) -> int:
        """The frame index a decoded frame corresponds to: its pts's rank in the table."""
        p = frame.pts if frame.pts is not None else frame.dts
        if self._tab is None:
            return int(round(float(p * self._tb) * float(self._rate)))
        if p is None:
            raise _TableMismatch('a decoded frame carries no timestamp')
        k = int(np.searchsorted(self._tab, p))
        if k >= len(self._tab) or self._tab[k] != p:
            raise _TableMismatch(f'decoded pts {p} is not in the frame table')
        return k

    def _seek_ts(self, ts: int):
        """Seek to the keyframe at/before stream timestamp `ts` and restart decoding there."""
        self._c.seek(int(ts), stream=self._st, backward=True, any_frame=False)
        self._iter = self._c.decode(self._st)
        self._pos = None

    def _seek(self, idx: int):
        """Seek to the keyframe at/before frame `idx`."""
        if self._tab is None:
            self._seek_ts(int(round(idx / float(self._rate) / float(self._tb))))
        else:
            self._seek_ts(int(self._tab[min(max(idx, 0), len(self._tab) - 1)]))

    def _decode_until(self, target: set, got: dict, last: int):
        """Decode forward, keeping frames in `target`, until index `last`. Returns the first pts
        decoded (the keyframe a seek landed on), or None."""
        first = None
        for frame in self._iter:
            if first is None:
                first = frame.pts
            idx = self._index_of(frame)
            self._pos = idx + 1
            if idx in target and idx not in got:
                got[idx] = frame.to_ndarray(format='rgb24')
            if idx >= last:
                break
        return first

    def get_batch(self, indices) -> np.ndarray:
        """Decode and return the frames at `indices`, in the order asked for.

        Inputs: indices -- iterable of frame indices; repeats are honoured.
        Outputs: (N, H, W, 3) uint8 array, one row per requested index.
        Side effects: advances the decoder, seeking once when the request is far ahead.

        The decoder continues rather than re-seeks when it is already close enough -- the loop
        walks the clip forwards, so consecutive calls are usually a short hop apart. Indices more
        than `_FORWARD_LIMIT` apart WITHIN one call are fetched by separate seeks. Missing frames
        are retried from progressively earlier keyframes (a decoder that skipped a damaged or
        open-GOP leading frame). The result preserves the
        ORDER ASKED FOR, including repeats -- `read_frames` relies on this when a clamp-padded
        window repeats its last frame.

        A decoded frame whose pts is not in the table means the table is wrong: a table built
        from the container index is replaced by a packet scan and the batch retried; a scanned
        table raises.
        """
        try:
            return self._get_batch(indices)
        except _TableMismatch as e:
            if self._table is None or self._table.source == 'scan':
                raise RuntimeError(f'{self.path}: {e} even after a full packet scan; the decoder '
                                   'and the demuxer disagree about timestamps.') from None
            print(f'video: {self.path}: {e} ({self._table.source} table); rebuilding the frame '
                  'table from a packet scan', file=sys.stderr, flush=True)
            self._set_table(frame_table(self.path, 0, rescan=True))
            return self._get_batch(indices)

    def _get_batch(self, indices) -> np.ndarray:
        """`get_batch` without the table-mismatch recovery.

        The sorted request is split into RUNS wherever two consecutive indices are more than
        `_FORWARD_LIMIT` apart, and each run is fetched with its own seek-or-continue decision --
        so scattered indices on a long video cost one GOP each, not a decode of everything
        between the first and the last.
        """
        want = [int(i) for i in indices]
        if not want:
            return np.empty((0, *self.frame_shape()), np.uint8)
        need = sorted(set(want))
        n = len(self)
        if need[0] < 0 or need[-1] >= n:
            raise RuntimeError(f'{self.path}: frame indices {need[0]}..{need[-1]} are outside '
                               f'[0, {n}).')
        runs, cur = [], [need[0]]
        for i in need[1:]:
            if i - cur[-1] > _FORWARD_LIMIT:
                runs.append(cur)
                cur = [i]
            else:
                cur.append(i)
        runs.append(cur)
        got: dict[int, np.ndarray] = {}
        for run in runs:
            self._get_run(run, got)
        return np.asarray([got[i] for i in want])

    def _get_run(self, need: list[int], got: dict):
        """Decode one sorted run of nearby indices into `got`, seeking only when the decoder is
        not already within `_FORWARD_LIMIT` before the run's first index.

        Missing frames are retried from progressively earlier keyframes: each seek backs off
        further from where the failed decode began, at most three times.
        """
        if not (self._iter is not None and self._pos is not None
                and self._pos <= need[0] <= self._pos + _FORWARD_LIMIT):
            self._seek(need[0])
        target = set(need)
        last = need[-1]
        first = self._decode_until(target, got, last)
        missing = [i for i in need if i not in got]
        if missing and self._tab is None:
            self._seek(missing[0])
            self._decode_until(target, got, last)
            missing = [i for i in need if i not in got]
        elif missing:
            step = int(np.median(np.diff(self._tab))) if len(self._tab) > 1 else 1
            start = int(self._tab[missing[0]])
            origin = start if first is None else min(start, int(first))
            for k in range(3):
                self._seek_ts(origin - 1 - step * 8 * 4 ** k)
                self._decode_until(target, got, last)
                missing = [i for i in need if i not in got]
                if not missing:
                    break
        if missing:
            raise RuntimeError(
                f'{self.path}: frames {missing[:5]} did not decode (asked for '
                f'{need[0]}..{need[-1]} of {len(self)}). A frame index that does not decode is a '
                'broken promise about n_frames, not a pixel to substitute.')

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
