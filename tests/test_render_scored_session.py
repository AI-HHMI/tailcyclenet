"""The frame -> window rule of `scripts/render_scored_session.py`.

A render colours each frame by the score of "its" window, so the assignment rule IS the rendering's
correctness: two windows overlap on their seam frames, and picking the earlier one would attribute
a frame to a window the scorer's lattice did not place it in (eval's rule is the LAST window
containing the frame -- `docs`/`CLAUDE.md` evaluation rule 11).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _mod():
    """Import the renderer without running main()."""
    spec = importlib.util.spec_from_file_location(
        'tcn_render_scored', REPO / 'scripts' / 'render_scored_session.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


STARTS = np.array([0, 12, 24], dtype=np.int64)


def test_a_seam_frame_belongs_to_the_later_window():
    """Frame 12 starts window 12 and is also the last frame of window 0; the later one wins."""
    assert _mod().frame_to_window(STARTS, 12, 12) == 12
    assert _mod().frame_to_window(STARTS, 12, 23) == 12
    assert _mod().frame_to_window(STARTS, 12, 24) == 24


def test_interior_frames_and_the_first_frame():
    """Frame 0 belongs to window 0; frame 11 (still inside it) does too."""
    assert _mod().frame_to_window(STARTS, 12, 0) == 0
    assert _mod().frame_to_window(STARTS, 12, 11) == 0


def test_a_frame_nobody_scored_is_unscored_rather_than_extrapolated():
    """Before the first start and past the last window's end there is no window, so no colour.

    A render that smeared the nearest score over these frames would show quality where none was
    measured.
    """
    assert _mod().frame_to_window(np.array([88008, 88020]), 12, 88000) is None
    assert _mod().frame_to_window(np.array([88008, 88020]), 12, 88032) is None
    assert _mod().frame_to_window(np.array([], dtype=np.int64), 12, 5) is None
