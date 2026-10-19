"""`render_group` and `project` on window-loop predictions."""
import numpy as np
import torch

from tailcyclenet.infer.window import run_group
from tests.helpers import _infer_cfg


def test_render_writes_every_predicted_frame(rat, tmp_path):
    """The mp4 holds exactly the predicted frames."""
    import cv2

    from tailcyclenet.render.render import render_group

    model, sess, registry, name = rat
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none', max_frames=3))
    path = tmp_path / 'clip.mp4'
    render_group(sess, 'g000', out['pred'], path, max_side=64, fps=5)

    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 3, f'wrote {n} frames for a 3-frame prediction'


def test_a_3d_render_uses_the_per_frame_camera_on_a_moving_rig(mv):
    """`render.project` uses the per-frame extrinsic on a moving rig (matching project_points_torch),
    and reads extrinsic `frames[t]` at column t for a non-identity `frames`."""
    from tailcyclenet.render.render import project

    model, sess, registry, name = mv
    cam_name = sess.cam_names[0]
    assert sess.rig.moving.get(cam_name), 'the fixture must actually be moving'

    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'))
    pred = out['pred']
    frames = np.arange(pred.shape[1])

    per_frame = project(sess, pred, 0, 'g000', frames)
    nominal = project(sess, pred, 0)                  # what it used to do
    assert per_frame.shape == pred.shape[:-1] + (2,)

    ok = np.isfinite(per_frame) & np.isfinite(nominal)
    assert ok.any(), 'need a finite projection to compare'
    assert not np.allclose(per_frame[ok], nominal[ok]), \
        'the two must differ on a moving rig, or the per-frame extrinsic is not being used'

    from tracktail.model.cube import project_points_torch
    cams = sess.cgroup('g000', frames)
    want = project_points_torch([cams[0]], torch.as_tensor(pred[0], dtype=torch.float32))[0]
    np.testing.assert_allclose(per_frame[0], want.numpy(), rtol=1e-5, atol=1e-4)

    shifted_frames = np.roll(np.arange(pred.shape[1]), 1)      # a permutation, still in bounds
    shifted = project(sess, pred, 0, 'g000', shifted_frames)
    ok2 = np.isfinite(shifted) & np.isfinite(per_frame)
    assert ok2.any()
    assert not np.allclose(shifted[ok2], per_frame[ok2]), \
        'a permuted frames array must change the projection, or the offset never reached cgroup'
    cams_shifted = sess.cgroup('g000', shifted_frames)
    want_shifted = project_points_torch([cams_shifted[0]],
                                        torch.as_tensor(pred[0], dtype=torch.float32))[0]
    np.testing.assert_allclose(shifted[0], want_shifted.numpy(), rtol=1e-5, atol=1e-4)
