"""THE VIDEO READ BACKEND, and the contract a replacement has to honour exactly.

`tailcyclenet/video.py` exists because decord loads whole containers into memory: a 16-camera
`--videos` run over 21 GB recordings peaked at 456 GB under `--max-ram 24`. PyAV replaced it, and
the ONLY thing that makes that swap legitimate is that it is output-neutral -- pinned HERE against
the synthetic fixture so it keeps being true.

**decord's `get_batch` CONTRACT IS NOT JUST "give me these frames".** It returns them in the ORDER
ASKED FOR, including REPEATS -- and `dataset.read_frames` leans on both: `_frames` clamp-pads a
window that runs past the end of its group, so one index legitimately occupies several positions.
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


def _both(clip):
    import os
    out = {}
    for name in ('pyav', 'decord'):
        os.environ['TAILCYCLENET_VIDEO_BACKEND'] = name
        out[name] = video.open_reader(clip)
    os.environ.pop('TAILCYCLENET_VIDEO_BACKEND', None)
    return out


def test_the_default_backend_is_pyav_and_decord_is_reachable(monkeypatch):
    """decord stays REACHABLE rather than deleted: it is what every recorded number was produced
    with, so bisecting a suspected pixel change needs it."""
    monkeypatch.delenv('TAILCYCLENET_VIDEO_BACKEND', raising=False)
    assert video.backend_name() == 'pyav'
    monkeypatch.setenv('TAILCYCLENET_VIDEO_BACKEND', 'decord')
    assert video.backend_name() == 'decord'
    monkeypatch.setenv('TAILCYCLENET_VIDEO_BACKEND', 'nonsense')
    with pytest.raises(ValueError, match='not one of'):
        video.backend_name()


def test_the_two_backends_agree_on_the_facts_the_probe_reads(clip):
    """`n_frames` is a PROMISE that every index in [0, T) decodes, so the two must not disagree
    about T -- `adopt.build` writes it into `groups.pq` and the window loop indexes against it."""
    rs = _both(clip)
    assert len(rs['pyav']) == len(rs['decord']) == 24
    assert rs['pyav'].frame_shape() == rs['decord'].frame_shape() == (48, 64, 3)
    assert rs['pyav'].fps == pytest.approx(rs['decord'].fps, abs=0.01)
    for r in rs.values():
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
    rs = _both(clip)
    try:
        for name, r in rs.items():
            got = r.get_batch(idx)
            assert got.shape == (len(idx), 48, 64, 3), f'{name}: {got.shape}'
            for pos, want_i in enumerate(idx):
                mean = got[pos].reshape(-1, 3).mean(0)
                want = np.asarray(_video_colour(2, want_i), float)
                assert np.abs(mean - want).max() < 12, (
                    f'{name}: position {pos} should be frame {want_i}; decoded {mean} '
                    f'wanted {want}')
        a, b = rs['pyav'].get_batch(idx), rs['decord'].get_batch(idx)
        assert np.array_equal(a, b), 'the backends must be BIT-IDENTICAL, or numbers move'
    finally:
        for r in rs.values():
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


def test_read_frames_is_identical_under_either_backend(tmp_path, monkeypatch):
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
    out = {}
    for name in ('pyav', 'decord'):
        monkeypatch.setenv('TAILCYCLENET_VIDEO_BACKEND', name)
        import tailcyclenet.dataset as ds
        ds._readers = None                      # the cache is built on first use, per backend
        out[name] = np.asarray(ds.read_frames(g, 'cam0', want))
    ds._readers = None
    assert out['pyav'].shape == (len(want), H, W, 3)
    assert np.array_equal(out['pyav'], out['decord'])
