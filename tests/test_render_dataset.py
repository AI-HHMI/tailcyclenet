"""`scripts/render_dataset.py` -- one end-to-end check that the overlays land.

The drawing itself is cv2 calls; what is worth pinning is the indexing around them, since a
render that silently draws frame 0's labels onto frame 3, or a region on the wrong camera, looks
plausible and is exactly what the script exists to catch.
"""
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from tailcyclenet import format as fmt

from .conftest import _session_2d

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location(
        'tcn_render', REPO / 'scripts' / 'render_dataset.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _session_with_a_region(path):
    _session_2d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    lab.regions = np.array([[1.0, 0.0, 4.0, 4.0, 40.0, 30.0]])      # frame 1, cam 0
    # `_session_2d` leaves every `labeled` instance's box NaN, so give the crop path something
    # to crop -- a boxless labelled instance is legal (§9, decision 10) and yields no crop.
    lab.boxes[0, :, 0] = [8.0, 6.0, 34.0, 28.0]
    fmt.write_session(path, mode=sess.mode, units=sess.units, label_source=sess.label_source,
                      names=sess.names, rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=sess.flip_pairs, provenance=sess.provenance)
    return fmt.Session.load(path)


def test_renders_a_sheet_a_crop_and_a_video(tmp_path):
    r = _mod()
    sess = _session_with_a_region(tmp_path / 'ds' / 'train' / 'a')
    out = tmp_path / 'out'
    out.mkdir()
    args = Namespace(width=200, video=True, fps=10.0, crops=True)
    stat = r.render_group(sess, 'g000', out, 'stem', args)

    assert stat['sheets'] == 1 and stat['videos'] == 1 and stat['crops'] > 0
    # _session_2d assesses every frame, so the sheet is the MIDDLE labelled frame, not frame 0
    assert stat['label_frames'] == 4
    assert (out / 'stem_f2.jpg').exists() and (out / 'stem.mp4').exists()
    assert list(out.glob('stem_f2_a0*.jpg'))


def test_the_region_is_drawn_on_its_own_frame_and_nowhere_else(tmp_path):
    """A region on frame 1 must not appear on frame 0 -- the failure a render cannot show you."""
    r = _mod()
    sess = _session_with_a_region(tmp_path / 'ds' / 'train' / 'a')
    from tailcyclenet.dataset import read_frames

    lab = sess.labels('g000')
    ims = read_frames(sess.groups['g000'], 'cam0', [0, 1])

    def cyan(t):
        drawn = r.draw(ims[t], lab, t, 0, sess.names, sess.skeleton)
        b, g, rd = drawn[..., 0], drawn[..., 1], drawn[..., 2]
        return int(((b > 200) & (g > 200) & (rd < 60)).sum())

    assert cyan(1) > 0 and cyan(0) == 0
