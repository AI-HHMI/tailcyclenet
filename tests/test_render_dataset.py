"""`scripts/render_dataset.py`: the indexing around the cv2 calls (frame 0's labels drawn on
frame 3 would look plausible)."""
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

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


def _session_with_boxes(path):
    _session_2d(path)
    sess = fmt.Session.load(path)
    lab = sess.labels('g000')
    # `_session_2d` stores no box for a01, so give it one (and make it `labeled`, which rule 11
    # allows only with a box) so the crop path has something to crop.
    lab.boxes[0, :, 0] = [8.0, 6.0, 34.0, 28.0]
    lab.instance[0, :, 0] = fmt.INST_LABELED
    fmt.write_session(path, mode=sess.mode, units=sess.units, label_source=sess.label_source,
                      names=sess.names, rig=sess.rig, groups=sess.groups, labels={'g000': lab},
                      flip_pairs=(sess.flip_pairs if sess.flip_pairs_declared else None),
                      provenance=sess.provenance)
    return fmt.Session.load(path)


def test_renders_a_sheet_a_crop_and_a_chunked_video(tmp_path, monkeypatch):
    """The sheet is the MIDDLE labelled frame, crops land, and `--video` decodes in chunks of at
    most `VIDEO_CHUNK` (whole-group decode OOMed on a 23,810-frame group) yet writes every frame."""
    from tailcyclenet import video

    r = _mod()
    sess = _session_with_boxes(tmp_path / 'ds' / 'train' / 'a')
    out = tmp_path / 'out'
    out.mkdir()
    asked = []
    real = r.read_frames
    monkeypatch.setattr(r, 'read_frames', lambda g, c, f, **kw: asked.append(len(f)) or real(g, c, f, **kw))
    monkeypatch.setattr(r, 'VIDEO_CHUNK', 3)
    args = Namespace(width=200, video=True, fps=10.0, crops=True)
    stat = r.render_group(sess, 'g000', out, 'stem', args)

    assert stat['sheets'] == 1 and stat['videos'] == 1 and stat['crops'] > 0
    assert stat['label_frames'] == 4
    assert (out / 'stem_f2.jpg').exists() and list(out.glob('stem_f2_a0*.jpg'))
    assert asked and max(asked) <= 3
    reader = video.open_reader(str(out / 'stem.mp4'))
    assert len(reader) == sess.groups['g000'].n_frames
    reader.close()
