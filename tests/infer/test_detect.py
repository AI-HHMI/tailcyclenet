"""`detect_raw`: batch-aligned slices and tight (never squared) boxes from an extent head."""
from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import _session_2d, _session_3d


def _detect_session(tmp_path, T):
    """A 2D session of T frames, preloaded (no model: the detector tests stub detection)."""
    from tailcyclenet.format import Session

    _session_2d(tmp_path / 'rat' / 'test' / 's', T=T)
    sess = Session.load(tmp_path / 'rat' / 'test' / 's')
    sess.preload()
    return sess


@pytest.mark.parametrize('start,stop', [(37, 53), (37, 64)])
def test_the_detection_slice_is_batch_aligned_and_reaches_the_range_end(tmp_path, start, stop):
    """The detection cursor opens on a `DET_BATCH` boundary BELOW `--start-frame` (never rounding
    the user's value), and `detect_raw` is told the RANGE end, so a ragged final slice (53) is
    accepted as well as an exact one (64). The driver once passed `max_frames=0` and failed 272 s
    into a 1000-frame run."""
    import tailcyclenet.infer.associate as assocmod
    import tailcyclenet.infer.detect as detmod
    from tailcyclenet.infer.cli import build_parser

    sess = _detect_session(tmp_path, T=80)
    seen = []

    def fake_detect_raw(det, wh, session, gid, top_k, **kw):
        fr = np.asarray(kw['frames'])
        T_clip = min(session.groups[gid].n_frames, kw['max_frames'] or session.groups[gid].n_frames)
        batch = detmod.DET_BATCH
        assert int(fr[0]) % batch == 0 and (len(fr) % batch == 0 or int(fr[-1]) == T_clip - 1), (
            f'slice {fr[0]}..{fr[-1]} of T_clip={T_clip} would be refused by detect_raw')
        seen.append((int(fr[0]), int(fr[-1])))
        C, D = len(session.rig), max(1, top_k)
        return (np.full((D, len(fr), C, 4), np.nan, np.float32),
                np.full((D, len(fr), C), np.nan, np.float32))

    class _Store:
        def read(self, ci, cam, fr, pool=None, reduce=1):
            return [None] * len(fr)

    old = (detmod.detect_raw, assocmod.associate_group)
    detmod.detect_raw = fake_detect_raw
    assocmod.associate_group = lambda raw, s, g, n, **kw: (raw[0][:n], raw[1][:n])
    try:
        args = build_parser().parse_args(['--run', 'r', '--out', 'o', '--det-score', '0.5',
                                          '--max-move', '1.0'])
        args.frame_start, args.frame_stop = start, stop
        boxes_for = detmod.detector_boxes(None, (64, 64), sess, 'g000', args, 'cpu', False,
                                           None, 2, 2)
        boxes_for(_Store(), start, stop)
    finally:
        detmod.detect_raw, assocmod.associate_group = old
    assert seen[0][0] == 32, f'the cursor must open on a multiple of 16, got {seen[0][0]}'
    assert seen[-1][1] == stop - 1, f'detection must reach the range end: {seen[-1]}'


@pytest.fixture(scope='module')
def det_scene(tmp_path_factory):
    """A 3-camera session and an untrained YOLOX-Nano (random weights: only the bytes matter)."""
    from tailcyclenet.detector.yolox import YOLOXNano
    from tailcyclenet.format import sessions_for

    root = tmp_path_factory.mktemp('det')
    _session_3d(root / 'ds' / 'test' / 's', T=4)
    _, sessions = sessions_for(root / 'ds', 'test')
    sess = sessions[0]
    sess.preload()
    return YOLOXNano(version='hybrid').eval(), sess, next(iter(sess.groups))


def test_detect_raw_returns_the_heads_own_boxes_for_either_target(det_scene):
    """`detect_raw` never re-squares: a `tight` head's boxes and a legacy `square` head's leave
    it as the same unletterboxed source-pixel boxes (the target only picks the window path)."""
    from tailcyclenet.infer.detect import detect_raw, legacy_square_head

    det, sess, gid = det_scene
    assert det.box_target == 'tight' and not legacy_square_head(det)
    got, sc = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4, score_thresh=0.0)
    det.box_target = 'square'
    try:
        assert legacy_square_head(det)
        sq, sc2 = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4, score_thresh=0.0)
    finally:
        det.box_target = 'tight'
    assert np.array_equal(got, sq, equal_nan=True) and np.array_equal(sc, sc2, equal_nan=True)
    ok = np.isfinite(got).all(-1)
    assert ok.any() and np.array_equal(ok, np.isfinite(sc))
    for ci, cam in enumerate(sess.cam_names):
        W, H = sess.rig.size(cam)
        b = got[:, :, ci][ok[:, :, ci]]
        assert (b[:, 2:] > b[:, :2]).all()
        assert (b[:, [0, 2]] >= 0).all() and (b[:, [0, 2]] <= W).all()
        assert (b[:, [1, 3]] >= 0).all() and (b[:, [1, 3]] <= H).all()


def test_old_box_target_names_read_as_square_and_tight(tmp_path, capsys):
    """'crop' / 'extent' (or absent) are READ as 'square' / 'tight' at the config and checkpoint
    boundary, with a one-line note for a config; nothing raises."""
    import torch

    from tailcyclenet.detector.checkpoint import load_detector
    from tailcyclenet.detector.dataset import canonical_box_target
    from tests.helpers import _detector_checkpoint

    assert canonical_box_target('extent', where='[data]') == 'tight'
    assert 'read as' in capsys.readouterr().out
    assert canonical_box_target('crop') == 'square' and canonical_box_target('tight') == 'tight'
    for old, new in (('crop', 'square'), ('extent', 'tight'), (None, 'square')):
        ck = _detector_checkpoint()
        if old is None:
            ck.pop('box_target')
        else:
            ck['box_target'] = old
        torch.save(ck, tmp_path / 'd.pth')
        assert load_detector(tmp_path / 'd.pth', 'cpu').model.box_target == new


def test_det_chunk_is_the_batch_for_small_inputs_and_splits_large_ones():
    """`det_chunk_for` keeps the whole batch whenever it fits (every existing root stays byte-identical) and splits
    only an input too large for the device; non-CUDA devices always forward the whole batch."""
    from tailcyclenet.infer.detect import DET_BATCH, det_chunk, det_chunk_for

    L4, A100 = 23.7e9, 80e9
    for wh in ((416, 416), (640, 640), (1920, 1080), (1280, 1024)):
        assert det_chunk_for(DET_BATCH, wh[0] * wh[1], L4) == DET_BATCH
    assert det_chunk_for(DET_BATCH, 2240 * 2240, A100) == DET_BATCH
    small = det_chunk_for(DET_BATCH, 2240 * 2240, L4)
    assert 1 <= small < DET_BATCH
    assert det_chunk_for(DET_BATCH, 2240 * 2240, L4) == small
    assert det_chunk_for(DET_BATCH, 10 ** 9, L4) == 1
    assert det_chunk(DET_BATCH, 10 ** 9, 'cpu') == DET_BATCH


def test_chunked_detection_matches_one_forward_per_batch(det_scene):
    """Splitting the model call into chunks leaves the detections unchanged within float tolerance. NOT bit-exact,
    even on CPU with one thread: conv kernels are chosen per batch shape (measured ~1e-6 relative on this model), so
    `det_chunk` keeps the whole batch whenever it fits and every existing root stays byte-identical."""
    from tailcyclenet.infer.detect import detect_raw

    det, sess, gid = det_scene
    whole = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4, score_thresh=0.0)
    for ck in (1, 3):
        got = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4, score_thresh=0.0, chunk=ck)
        np.testing.assert_allclose(got[0], whole[0], rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(got[1], whole[1], rtol=1e-4, atol=1e-5)
