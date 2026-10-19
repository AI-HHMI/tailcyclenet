"""`scripts/render_scored_session.py`: a frame is coloured by the LAST window containing it
(evaluation rule 11), on a fixed colour scale, for the selected animal."""

import numpy as np

from scripts import render_scored_session
from tailcyclenet.infer.window import last_window_start

STARTS = np.array([0, 12, 24], dtype=np.int64)


def test_the_seam_rule_and_unscored_frames():
    """A seam frame belongs to the later window; a frame no window covers is unscored, not
    extrapolated (a smeared score would show quality where none was measured)."""
    assert [last_window_start(STARTS, 12, f) for f in (0, 11, 12, 23, 24)] == [0, 0, 12, 12, 24]
    assert last_window_start(np.array([88008, 88020]), 12, 88000) is None
    assert last_window_start(np.array([88008, 88020]), 12, 88032) is None
    assert last_window_start(np.array([], dtype=np.int64), 12, 5) is None


def test_the_colour_ramp_runs_red_to_green_and_clamps():
    """Red->green, monotone, clamped, on a FIXED scale so two renders compare."""
    mod = render_scored_session
    lo, hi = mod.SCORE_RANGE
    assert mod._colour(lo, lo, hi) == mod.BAD == (0, 0, 255)      # red
    assert mod._colour(hi, lo, hi) == mod.GOOD == (0, 255, 0)     # green
    # Outside the range the colour is the nearer END, not an extrapolation.
    assert mod._colour(lo - 100, lo, hi) == mod.BAD
    assert mod._colour(hi + 100, lo, hi) == mod.GOOD

    # Monotone: green rises and red falls as the score rises. None of red/green/dead-grey is
    # confusable with the unscored grey used where no window covers a frame.
    greens = [mod._colour(lo + (hi - lo) * f, lo, hi)[1] for f in (0, 0.25, 0.5, 0.75, 1)]
    reds = [mod._colour(lo + (hi - lo) * f, lo, hi)[2] for f in (0, 0.25, 0.5, 0.75, 1)]
    assert greens == sorted(greens) and reds == sorted(reds, reverse=True)
    assert mod.UNSCORED == (128, 128, 128) and mod.UNSCORED != mod._colour(0.0, lo, hi)


def test_the_selected_animal_is_drawn_not_animal_zero():
    """`--animal` filters the scores, so the skeleton drawn under them must be that animal's."""
    from types import SimpleNamespace

    mod = render_scored_session
    points2d = np.full((2, 1, 1, 1, 2), np.nan)
    points2d[1, 0, 0, 0] = (100.0, 100.0)
    lab = SimpleNamespace(animal_ids=['det00', 'det01'], points2d=points2d, vis2d=None)
    a = mod.animal_index(lab, 'det01')
    assert a == 1
    im = np.zeros((200, 200, 3), dtype=np.uint8)
    legend = {0: ''}
    out = mod.draw_scored(im, lab, a, 0, 0, ['nose'], [], {(0, 'nose'): 1.0}, 0, 0.0, 1.0, '', legend)
    assert out[100, 100].any(), 'animal 1 owns the only finite point; it must be drawn'
    blank = mod.draw_scored(im, lab, 0, 0, 0, ['nose'], [], {(0, 'nose'): 1.0}, 0, 0.0, 1.0, '', legend)
    assert not blank[100, 100].any()


def test_a_scores_miss_names_the_sessions_that_are_present(tmp_path):
    """The hint must be read before the filter empties the table."""
    import polars as pl
    import pytest

    pl.DataFrame({'session': ['other'], 'group': ['g'], 'animal': ['det00'], 'start': [0],
                  'keypoint': ['nose'], 'score': [1.0], 'precision': [1.0]}).write_parquet(
        tmp_path / 'scores.pq')
    with pytest.raises(SystemExit, match="Sessions present: \\['other'\\]"):
        render_scored_session.load_scores(tmp_path, 'sess', 'g', 'det00')
