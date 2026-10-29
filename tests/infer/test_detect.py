"""`detect_raw`: batch-aligned slices and the crop rule applied to an extent head."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tailcyclenet.data.crop import crop_boxes_from_extents
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


def test_detect_raw_applies_the_crop_rule_to_an_extent_head(det_scene):
    """An `extent` checkpoint's boxes leave `detect_raw` as crop-rule boxes in SOURCE pixels,
    padded by the session's own box source; a `crop` checkpoint's are untouched."""
    from tailcyclenet.infer.detect import detect_raw

    det, sess, gid = det_scene
    det.box_target = 'crop'
    raw, sc = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4,
                            score_thresh=0.0)
    det.box_target, det.min_crop_dim = 'extent', 16
    try:
        for source, frac in (('keypoints', 0.05), ('instances', 0.0)):
            got, sc2 = detect_raw(det, (64, 64), sess, gid, 3, device='cpu', batch=4,
                                     score_thresh=0.0, box_source=source)
            assert np.array_equal(sc, sc2, equal_nan=True)
            assert np.isfinite(raw).any()
            for ci, cam in enumerate(sess.cam_names):
                size = sess.rig.size(cam)
                want = crop_boxes_from_extents(torch.as_tensor(raw[:, :, ci].reshape(-1, 4)),
                                               size, 16, 0.0, frac).numpy()
                assert np.array_equal(got[:, :, ci].reshape(-1, 4), want, equal_nan=True)
    finally:
        det.box_target = 'crop'
