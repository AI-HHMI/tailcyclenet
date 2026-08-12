"""THE window loop, on the paths that used to raise.

`run_group` had no coverage at all, and two of its failures were IndexErrors reachable from the
ordinary deployment command line rather than from anything exotic. The model here is random and
tiny -- these assert that the loop RUNS and returns the right shapes and identities, never that
the numbers are good.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from tailcyclenet.format import Registry, load_dataset
from tailcyclenet.infer import InferConfig, run_group
from tailcyclenet.model import build_model
from test_model import SMALL


@pytest.fixture(scope='module', params=['rat', 'mv'])
def scene(request):
    """A 2D multi-animal session and a 3D session on a moving rig."""
    import conftest as cf

    root = Path(tempfile.mkdtemp())
    if request.param == 'rat':
        cf._session_2d(root / 'rat' / 'test' / 's')          # T=4, 2 animals, 1 camera
    else:
        cf._session_3d(root / 'mv' / 'test' / 's', moving=True)   # T=4, 1 animal, moving
    ds = load_dataset(root / request.param)
    registry = Registry.build([ds])
    sess = ds.sessions['test'][0]
    sess.preload()
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()
    return model, sess, registry, ds.name


def _cfg(**kw):
    kw.setdefault('overlap', 2)
    return InferConfig(n_frames=4, image_size=64, min_crop_dim=16, device='cpu', **kw)


@pytest.mark.parametrize('anchor', ['none', 'carry', 'self', 'labels'])
def test_every_anchor_runs(scene, anchor):
    model, sess, registry, name = scene
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor=anchor))
    R = 3 if sess.mode == '3d' else 2
    assert out['pred'].shape == (len(sess.labels('g000').animal_ids), 4, sess.n_keypoints, R)
    assert len(out['animal_ids']) == out['pred'].shape[0]


def test_more_detector_rows_than_label_rows(scene):
    """A DETECTOR ROW IS NOT A LABEL ROW.

    `S` comes from the box source. On the deployment path that is the detector, which can offer
    more animals than the session ever labelled -- and `src[a]` was evaluated for every one of
    them, on every anchor mode including `none`. Plain IndexError.
    """
    model, sess, registry, name = scene
    lab = sess.labels('g000')
    w, h = sess.rig.size(sess.cam_names[0])
    boxes = np.zeros((5, 4, len(sess.rig), 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h

    for anchor in ('none', 'carry', 'labels'):
        out = run_group(model, sess, 'g000', registry, name, _cfg(anchor=anchor),
                        boxes_stc=boxes)
        assert out['pred'].shape[0] == 5
        # the id list must match the prediction row count, and name the surplus rows honestly
        assert len(out['animal_ids']) == 5
        assert list(out['animal_ids'][:len(lab.animal_ids)]) == list(lab.animal_ids)
        assert all(str(i).startswith('det') for i in out['animal_ids'][len(lab.animal_ids):])


def test_a_camera_without_a_box_does_not_drop_the_animal(scene):
    """Use the cameras that saw it, not all or nothing.

    Cross-view association leaves an unmatched camera NaN, which is the normal outcome when a
    detector misses one view. Requiring a box in every camera discarded the animal for the whole
    window even when the others had it -- measured as coverage 0.000 where it should be 1.000 on
    a three-camera rig whose first view went unmatched. The model is trained on camera subsets
    already, so a subset is a supported input, not a degenerate one.
    """
    model, sess, registry, name = scene
    C = len(sess.rig)
    if C < 2:
        pytest.skip('needs a multi-camera rig')
    w, h = sess.rig.size(sess.cam_names[0])
    boxes = np.zeros((1, 4, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h
    boxes[:, :, 0] = np.nan                       # camera 0 saw nothing

    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_stc=boxes)
    assert np.isfinite(out['pred']).all(-1).any(), 'the surviving cameras must still predict'


def test_group_shorter_than_the_overlap(scene):
    """`p[-overlap]` ran off the front of a window with fewer frames than the step."""
    model, sess, registry, name = scene
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', overlap=8))
    assert out['pred'].shape[1] == 4


def test_kpt_chunk_matches_unchunked_end_to_end(scene):
    """Chunking is a memory knob, so it must not move the prediction by so much as an ulp."""
    model, sess, registry, name = scene
    whole = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))['pred']
    chunked = run_group(model, sess, 'g000', registry, name,
                        _cfg(anchor='none', kpt_chunk=2))['pred']
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-5)


def test_carry_requires_overlap(scene):
    model, sess, registry, name = scene
    with pytest.raises(ValueError, match='overlap'):
        run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', overlap=0))


def test_carried_prior_is_bounds_masked_and_dated():
    """The two defects in the deployed prompt, both of which were silent.

    `prompt_time = 0` is right for interior windows and wrong on the last window of a group, which
    `_window_starts` pulls back to `n_frames - T`; and a carried keypoint that left the new crop
    was handed in as a confident prior instead of as "I was not told".
    """
    from tailcyclenet.infer import _build_prior

    K, size = 4, torch.tensor([100, 100], dtype=torch.int32)
    cgroup = [{'size': size}]
    cfg = InferConfig(anchor='carry')
    frames = np.arange(20, 28)
    boxes = [(0, 0, 100, 100)]

    # Two keypoints inside the crop, two outside it.
    pose = torch.tensor([[10.0, 10.0], [90.0, 90.0], [-5.0, 10.0], [10.0, 400.0]])
    prior, qt = _build_prior(cfg, (pose, 23), None, frames, boxes, [1.0], '2d', K, 2, cgroup)

    assert torch.isfinite(prior[0, :2]).all(), 'in-crop keypoints must survive'
    assert torch.isnan(prior[0, 2:]).all(), 'out-of-crop keypoints must become NaN'
    assert qt.shape == (1, K)
    assert (qt == 3).all(), f'prompt frame 23 is index 3 of a window starting at 20, got {qt}'

    # A prompt from before the window cannot be expressed and clamps into range rather than
    # indexing off the front of it.
    _, early = _build_prior(cfg, (pose, 2), None, frames, boxes, [1.0], '2d', K, 2, cgroup)
    assert (early == 0).all()
    _, late = _build_prior(cfg, (pose, 999), None, frames, boxes, [1.0], '2d', K, 2, cgroup)
    assert (late == len(frames) - 1).all()


def test_max_frames_takes_a_prefix(scene):
    """A PREFIX, not a sample: `carry` needs the frames contiguous, and the rat-city protocol is
    frames 0-479 of one trial. `n_frames` in the result must describe what was actually predicted,
    or the render and the eval disagree about which clip this is."""
    model, sess, registry, name = scene
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', max_frames=3))
    assert out['pred'].shape[1] == 3
    assert out['n_frames'] == 3
    # 0 and a ceiling above the group both mean "the whole group".
    for m in (0, 99):
        assert run_group(model, sess, 'g000', registry, name,
                         _cfg(anchor='none', max_frames=m))['pred'].shape[1] == 4


def test_render_writes_every_predicted_frame(scene):
    """The mp4 must hold exactly the frames the prediction covers -- a render one frame short of
    its npz is how a clip and its numbers drift apart."""
    import cv2

    from tailcyclenet.render import render_group

    model, sess, registry, name = scene
    if sess.mode != '2d':
        pytest.skip('3D sessions are not drawn yet')
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', max_frames=3))
    path = Path(tempfile.mkdtemp()) / 'clip.mp4'
    render_group(sess, 'g000', out['pred'], path, max_side=64, fps=5)

    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 3, f'wrote {n} frames for a 3-frame prediction'
