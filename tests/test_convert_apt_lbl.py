"""The pure decode/placement logic of `scripts/convert_apt_lbl.py`.

Everything here is a thing that would be silently wrong rather than loud: the `p` reshape, the
1-based offsets, APT's two coordinate sentinels, and where the label sits in its window.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def cv():
    spec = importlib.util.spec_from_file_location(
        'tcn_convert_apt', REPO / 'scripts' / 'convert_apt_lbl.py')
    mod = importlib.util.module_from_spec(spec)
    # In sys.modules BEFORE exec: the module defines a @dataclass, and dataclasses resolves
    # string annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(p, frm, tgt, npts=4):
    return SimpleNamespace(p=np.asarray(p, float), frm=np.asarray(frm), tgt=np.asarray(tgt),
                           npts=np.asarray(npts))


def test_p_is_x_then_y_and_one_based(cv):
    """`p` column is [x1..x4, y1..y4]; frm/tgt/coords all lose their MATLAB 1."""
    e = _entry([[10], [20], [30], [40], [50], [60], [70], [80]], [7], [3])
    frm, tgt, xy = cv.movie_labels(e)
    assert frm.tolist() == [6] and tgt.tolist() == [2]
    # x's lead, so keypoint 0 is (10, 50) in MATLAB and (9, 49) here.
    assert xy.shape == (1, 4, 2)
    assert xy[0].tolist() == [[9, 49], [19, 59], [29, 69], [39, 79]]


def test_interleaved_reshape_would_be_caught(cv):
    """The interleaved (npts, 2) reading gives different points -- the axis order is load-bearing.

    Both readings produce plausible coordinates, which is why this is asserted on values. On the
    real project the interleaved one puts 28.25% of points outside the frame against 0.01%.
    """
    e = _entry([[10], [20], [30], [40], [50], [60], [70], [80]], [1], [1])
    _, _, xy = cv.movie_labels(e)
    interleaved = np.asarray(e.p, float).reshape(4, 2, 1)[..., 0] - 1.0
    assert not np.array_equal(xy[0], interleaved)


def test_empty_movie(cv):
    frm, tgt, xy = cv.movie_labels(SimpleNamespace(frm=np.zeros(0)))
    assert frm.size == tgt.size == 0 and xy.shape == (0, 4, 2)


def test_sentinels_map_to_visible_missing_and_no_row(cv):
    """NaN -> no row, Inf -> missing with no coords, finite -> visible with coords."""
    fmt = importlib.import_module('tailcyclenet.format')
    xy = np.array([[[1.0, 2.0], [np.nan, np.nan], [np.inf, np.inf], [5.0, 6.0]]])
    job = SimpleNamespace(frm=np.array([10]), tgt=np.array([0]), xy=xy, wh=(100, 100))
    lab = cv.group_labels(job, frame=10, start=4, n=9, K=4)
    v = lab.vis2d[0, 6, :, 0]                     # label sits at 10 - 4 = 6
    assert v.tolist() == [fmt.VISIBLE, fmt.UNLABELED, fmt.MISSING, fmt.VISIBLE]
    assert np.isfinite(lab.points2d[0, 6, [0, 3], 0]).all()
    assert not np.isfinite(lab.points2d[0, 6, [1, 2], 0]).any()
    assert lab.instance[0, 6, 0] == fmt.INST_LABELED
    # No other frame in the window says anything.
    assert (np.delete(lab.vis2d[0, :, :, 0], 6, axis=0) == fmt.UNLABELED).all()


def test_box_is_the_padded_extent_of_positioned_points(cv):
    """Inf and NaN points must not enter the box, and it clips to the frame and truncates."""
    xy = np.array([[[50.0, 60.0], [np.inf, np.inf], [10.0, 10.0], [np.nan, np.nan]]])
    job = SimpleNamespace(frm=np.array([0]), tgt=np.array([0]), xy=xy, wh=(100, 100))
    lab = cv.group_labels(job, frame=0, start=0, n=4, K=4)
    # extent of (50,60) and (10,10) padded by 20, clipped into 100x100
    assert lab.boxes[0, 0, 0].tolist() == [0.0, 0.0, 70.0, 80.0]
    assert np.isnan(cv.padded_extent(np.full((4, 2), np.nan), (100, 100))).all()


@pytest.mark.parametrize('frame,n_source,want', [
    (1000, 5000, (968, 65)),      # interior: label centered at 32
    (5, 5000, (0, 65)),           # near the start: clamped, label at 5
    (4990, 5000, (4935, 65)),     # near the end: clamped, label at 55
    (10, 40, (0, 40)),            # movie shorter than the context
])
def test_window_placement(cv, frame, n_source, want):
    start, n = cv.window_start(frame, n_source, 65)
    assert (start, n) == want
    assert 0 <= frame - start < n            # the labelled frame is inside its own window
    assert 0 <= start and start + n <= n_source


def test_split_and_session_id(cv):
    root = '/nrs/branson/RatCity/ConcatVideos'
    assert cv.split_of(f'{root}/train/cohort5/cohort5_20250812_1241/movie.avi', False) == 'val'
    assert cv.split_of(f'{root}/train/cohort7/cohort7_20251209_1659/movie.avi', False) == 'train'
    assert cv.split_of(f'{root}/test/cohort1vs2/cohort1vs2_20240731_1800/movie.avi', True) == 'test'
    # the merged movie dated inside cohort5's range goes to val with it
    m = f'{root}/train/original/merged_video_all_keyframes_20250819_111300_to_20250819_112300_APT.avi'
    assert cv.split_of(m, False) == 'val'
    assert cv.split_of(f'{root}/train/original/merged_video_all_keyframes.avi', False) == 'train'

    assert cv.session_id(f'{root}/train/cohort7/cohort7_20251209_1659/movie.avi') \
        == 'cohort7_20251209_1659'
    assert cv.session_id(f'{root}/train/original/merged_video_all_keyframes_005.avi') \
        == 'merged_video_all_keyframes_005'
