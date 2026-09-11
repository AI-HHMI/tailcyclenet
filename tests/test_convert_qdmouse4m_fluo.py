"""The camera-axis map of `scripts/convert_qdmouse4m_fluo.py`.

The bug this pins was silent in both directions: the derived dataset VALIDATED (every camera had
a calibration block and a video) and only the 2D labels were attached to the wrong physical view,
so the failure was a spatially offset overlay rather than an error. The converter expanded the
per-camera arrays as a block copy while the rig INTERLEAVES `<view>` and `<view>_fluo`, so
destination camera `j` received the labels of source camera `j % 6`.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def conv():
    spec = importlib.util.spec_from_file_location(
        'tcn_convert_qdmouse4m_fluo', REPO / 'scripts' / 'convert_qdmouse4m_fluo.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SRC = ['a', 'b', 'c']
DST = ['a', 'a_fluo', 'b', 'b_fluo', 'c', 'c_fluo']


def test_interleaved_rig_maps_each_camera_to_its_own_view(conv):
    """The interleaved order must give [0, 0, 1, 1, 2, 2] -- not [0, 1, 2, 0, 1, 2]."""
    assert conv.camera_index_map(SRC, DST) == [0, 0, 1, 1, 2, 2]


def test_a_block_copy_would_have_been_the_bug(conv):
    """Guard the shape of the mistake: the map is NOT the identity, so a naive
    `concatenate([a, a], axis=camera)` cannot produce the correct array."""
    take = conv.camera_index_map(SRC, DST)
    assert take != list(range(len(DST)))


def test_unknown_view_is_loud(conv):
    """A camera with no source view must raise rather than silently pick a neighbour's labels."""
    with pytest.raises(RuntimeError, match='no source camera'):
        conv.camera_index_map(SRC, ['a', 'zzz_fluo'])


def test_taking_the_map_puts_each_camera_on_its_own_projection(conv):
    """End to end on arrays: after `take`, slot j's labels are view j//2's, and the two slots
    of one view are identical (a channel of the same camera sees the same animal)."""
    take = conv.camera_index_map(SRC, DST)
    src = np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3)     # camera axis last
    dst = np.take(src, take, axis=3)
    assert dst.tolist() == [[[[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]]]]
