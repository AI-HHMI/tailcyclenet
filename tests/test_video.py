"""THE VIDEO READ BACKEND, and the contract a replacement has to honour exactly.

`tailcyclenet/video.py` exists because PyAV provides bounded-memory, frame-accurate decoding: a
16-camera `--videos` run over 21 GB recordings peaked at 456 GB under the old backend. The
contract is pinned HERE against the synthetic fixture so it keeps being true.

`get_batch` returns frames in the ORDER ASKED FOR, including REPEATS -- and `dataset.read_frames`
leans on both: `_frames` clamp-pads a window that runs past the end of its group, so one index
legitimately occupies several positions.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from tailcyclenet import video

from .conftest import _video_colour, _write_video


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp('vid')
    return str(_write_video(d / 'c.mp4', 2, 24, (64, 48)))


def test_reader_reports_the_facts_the_probe_reads(clip):
    """`n_frames` is a PROMISE that every index in [0, T) decodes."""
    r = video.open_reader(clip)
    assert len(r) == 24
    assert r.frame_shape() == (48, 64, 3)
    assert r.fps == pytest.approx(20.0, abs=0.01)
    r.close()


@pytest.mark.parametrize('idx', [
    [0, 1, 2, 3],                 # contiguous
    [5, 5, 5, 6, 7, 7],           # A CLAMP-PADDED WINDOW: repeats, in position order
    [9, 2, 7, 2, 0],              # out of order, with a repeat
    [23],                         # the last frame
    [0, 23],                      # both ends, forcing a seek
])
def test_get_batch_honours_order_and_repeats(clip, idx):
    """Asserted on VALUES against the fixture's own (camera, frame) colours, not just on shapes --
    a backend that returned sorted-unique frames would pass a shape check on the first case and
    silently corrupt the other four."""
    r = video.open_reader(clip)
    try:
        got = r.get_batch(idx)
        assert got.shape == (len(idx), 48, 64, 3)
        for pos, want_i in enumerate(idx):
            mean = got[pos].reshape(-1, 3).mean(0)
            want = np.asarray(_video_colour(2, want_i), float)
            assert np.abs(mean - want).max() < 12, (
                f'position {pos} should be frame {want_i}; decoded {mean} wanted {want}')
    finally:
        r.close()


def test_a_vfr_remux_is_indexed_by_the_declared_rate_not_the_average(tmp_path):
    """A RATE OFF BY A PART IN 400 IS AN OFF-BY-ONE FRAME PART WAY THROUGH THE CLIP: `average_rate`
    is a derived average that drifts on a `-vsync vfr` remux, and under it the ordinals eventually
    SKIP -- the decoder then cannot produce that index, and every frame after is mislabelled +1.
    So this asserts the COLOURS, not merely that nothing raised (the same reason OpenCV was rejected).
    """
    import subprocess

    from .conftest import _write_video

    src = _write_video(tmp_path / 'src.mp4', 1, 400, (64, 48), fps=200.0)
    dst = tmp_path / 'clip.mp4'
    keep, n_out = 4, 100
    rc = subprocess.run(
        ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', str(src),
         '-vf', f"select='not(mod(n\\,{keep}))'", '-vsync', 'vfr',
         '-frames:v', str(n_out), '-an', '-c:v', 'libx264', '-crf', '16',
         '-preset', 'veryfast', '-pix_fmt', 'yuv420p', str(dst)]).returncode
    if rc != 0 or not dst.exists():
        pytest.skip('ffmpeg could not produce the vfr remux')

    import av
    with av.open(str(dst)) as c:
        st = c.streams.video[0]
        # THE PRECONDITION. If a future ffmpeg stops producing this shape the test is no longer
        # exercising the bug, and saying so is better than passing vacuously.
        if st.average_rate == st.guessed_rate:
            pytest.skip('this ffmpeg wrote a container whose two rates agree')

    r = video.PyAVReader(str(dst))
    try:
        n = len(r)
        # EVERY index in [0, n) decodes -- `n_frames` is a promise, and under `average_rate` one
        # of these indices did not exist at all (frame 2000 of the real allen clip).
        got = r.get_batch(list(range(n)))
        assert got.shape[0] == n
        # AND each one is the frame it claims to be. Output frame i is source frame keep*i, so a
        # +1 ordinal shift anywhere shows up as the wrong colour rather than as a clean array.
        for i in (0, n // 2, n - 2, n - 1):
            mean = got[i].reshape(-1, 3).mean(0)
            want = np.asarray(_video_colour(1, keep * i), float)
            assert np.abs(mean - want).max() < 12, (
                f'frame {i} decoded {mean}, wanted {want} -- the ordinal drifted')
    finally:
        r.close()


def test_read_frames_uses_the_pyav_backend(tmp_path):
    """THE REAL CALL PATH, not the reader in isolation: `read_frames` adds a dedupe, the
    clamp-pad's per-position copies and an optional warp on top."""
    import conftest as cf
    from tailcyclenet import format as fmt

    W, H, T = 64, 48, 12
    src = cf._write_video(tmp_path / 'rec' / 'cam0.mp4', 0, T, (W, H))
    g = fmt.video_group('g', T, {'cam0': src})
    rig = cf._rig([('cam0', W, H, False, False, 0)])
    sess = fmt.VideoSession(path=tmp_path / 'nope', mode='2d', units='px',
                            label_source='tracked', names=['a'], rig=rig, groups={'g': g},
                            empty={'g': fmt.empty_labels(0, T, 1, 1, mode3d=False)})
    g.session = sess

    want = np.asarray([2, 2, 3, 7, 1])
    import tailcyclenet.dataset as ds
    ds._readers = None
    out = np.asarray(ds.read_frames(g, 'cam0', want))
    ds._readers = None
    assert out.shape == (len(want), H, W, 3)



# ---------------------------------------------------------------------------------------------
# the frame table: index k = the k-th pts in display order


def _ffmpeg(*args):
    import subprocess

    rc = subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', *args]).returncode
    if rc != 0:
        pytest.skip('ffmpeg could not produce the fixture')


def _assert_colours(got, idx, cam, scale=1):
    for pos, i in enumerate(idx):
        mean = got[pos].reshape(-1, 3).mean(0)
        want = np.asarray(_video_colour(cam, scale * i), float)
        assert np.abs(mean - want).max() < 12, f'position {pos} should be frame {i}; got {mean}'


@pytest.fixture
def fresh_tables(tmp_path, monkeypatch):
    """An empty in-process table cache and a private on-disk one."""
    monkeypatch.setenv('TAILCYCLENET_FRAME_TABLE_CACHE', str(tmp_path / 'tables'))
    video._TABLES.clear()
    yield tmp_path / 'tables'
    video._TABLES.clear()


def test_a_20fps_camera_on_a_1_30_grid_decodes_every_index(tmp_path, fresh_tables):
    """THE OUTDOOR-PORTABLE CAMERAS: 20 fps written onto a 1/30 time base, pts 0,2,3,5,6,...
    `pts x guessed_rate(30)` skipped every third index ("frames [1, 4, 7, ...] did not decode");
    a rank in the pts table cannot."""
    import av

    src = _write_video(tmp_path / 'src.mp4', 1, 60, (64, 48))
    dst = tmp_path / 'grid.mp4'
    _ffmpeg('-i', str(src), '-vf', 'settb=1/30,setpts=ceil(N*1.5)', '-fps_mode', 'passthrough',
            '-video_track_timescale', '30', '-c:v', 'libx264', '-bf', '0', '-g', '10',
            '-crf', '10', '-pix_fmt', 'yuv420p', str(dst))
    with av.open(str(dst)) as c:
        pts = [p.pts for p in c.demux(c.streams.video[0]) if p.pts is not None]
    if sorted(pts)[:4] != [0, 2, 3, 5]:
        pytest.skip(f'this ffmpeg did not write the 1/30 grid (pts {sorted(pts)[:4]})')

    r = video.PyAVReader(str(dst))
    try:
        assert r.table_source == 'index'      # read off the MP4 header, no scan
        assert len(r) == 60
        assert r.fps == pytest.approx(20.0, rel=0.01)
        assert r.frame_times()[:4] == pytest.approx([0, 2 / 30, 3 / 30, 5 / 30])
        idx = list(range(60))
        _assert_colours(r.get_batch(idx), idx, 1)
        idx = [59, 1, 1, 31, 4, 0]            # seeks, repeats, out of order
        _assert_colours(r.get_batch(idx), idx, 1)
    finally:
        r.close()


def test_b_frames_are_indexed_in_display_order(tmp_path, fresh_tables):
    """The MP4 sample index holds DECODE timestamps; with B-frames the header route must shift
    them onto presentation time (validated against the head and tail packets) or fall back."""
    src = _write_video(tmp_path / 'src.mp4', 2, 90, (64, 48))
    dst = tmp_path / 'bf.mp4'
    _ffmpeg('-i', str(src), '-c:v', 'libx264', '-bf', '3', '-g', '12', '-crf', '10',
            '-pix_fmt', 'yuv420p', str(dst))
    r = video.PyAVReader(str(dst))
    try:
        assert r.table_source in ('index', 'scan')
        assert len(r) == 90
        idx = [0, 1, 2, 3, 50, 49, 89, 88, 13, 13, 70]
        _assert_colours(r.get_batch(idx), idx, 2)
    finally:
        r.close()


def test_mkv_falls_back_to_a_packet_scan_that_is_cached_on_disk(tmp_path, fresh_tables):
    """Matroska cues index only keyframes, so the header route refuses and the file is demuxed
    once -- then the NEXT process (here: a cleared in-process cache) reads the stored table."""
    src = _write_video(tmp_path / 'src.mp4', 0, 40, (64, 48))
    dst = tmp_path / 'clip.mkv'
    _ffmpeg('-i', str(src), '-c:v', 'libx264', '-bf', '2', '-g', '8', '-crf', '10',
            '-pix_fmt', 'yuv420p', str(dst))
    r = video.PyAVReader(str(dst))
    assert r.table_source == 'scan'
    idx = [39, 0, 20, 21, 21]
    _assert_colours(r.get_batch(idx), idx, 0)
    r.close()
    assert len(list(fresh_tables.glob('*.npy'))) == 1

    video._TABLES.clear()
    r = video.PyAVReader(str(dst))
    try:
        assert r.table_source == 'cache'
        assert len(r) == 40
        _assert_colours(r.get_batch(idx), idx, 0)
    finally:
        r.close()


def test_a_wrong_header_table_is_replaced_by_a_scan(tmp_path, fresh_tables, monkeypatch):
    """Every decoded pts must be a MEMBER of the table, so a header table that is wrong past the
    probed ends is caught at the first frame it would mislabel, and rebuilt -- never trusted."""
    src = _write_video(tmp_path / 'src.mp4', 1, 60, (64, 48))
    dst = tmp_path / 'clip.mp4'
    _ffmpeg('-i', str(src), '-c:v', 'libx264', '-bf', '0', '-g', '10', '-crf', '10',
            '-pix_fmt', 'yuv420p', str(dst))
    real = video._index_table

    def corrupt(container, stream):
        pts = real(container, stream)
        pts = pts.copy()
        pts[30:] += 1                           # a table off by one tick from frame 30 on
        return pts

    monkeypatch.setattr(video, '_index_table', corrupt)
    r = video.PyAVReader(str(dst))
    try:
        assert r.table_source == 'index'
        idx = [0, 1, 45, 46]
        _assert_colours(r.get_batch(idx), idx, 1)
        assert r.table_source == 'scan'
    finally:
        r.close()


def test_scattered_indices_in_one_call_are_fetched_by_separate_seeks(tmp_path, fresh_tables,
                                                                    monkeypatch):
    """A batch spanning a long file must not decode everything between its ends: indices more
    than `_FORWARD_LIMIT` apart are separate runs, each seeking on its own."""
    src = _write_video(tmp_path / 'src.mp4', 1, 400, (64, 48))
    monkeypatch.setattr(video, '_FORWARD_LIMIT', 20)
    r = video.PyAVReader(str(src))
    decoded = []
    real = r._decode_until

    def counting(target, got, last):
        before = r._pos
        out = real(target, got, last)
        decoded.append((before, r._pos))
        return out

    r._decode_until = counting
    try:
        idx = [390, 5, 200, 201, 5]
        _assert_colours(r.get_batch(idx), idx, 1)
        assert len(decoded) == 3                 # three runs: [5], [200, 201], [390]
        assert r._pos <= 391
    finally:
        r.close()
