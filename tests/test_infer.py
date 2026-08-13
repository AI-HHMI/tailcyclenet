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


@pytest.fixture(scope='module')
def cli():
    """`scripts/infer.py` as a module, without running main(). Same pattern as test_train.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tcn_infer', Path(__file__).resolve().parent.parent / 'scripts' / 'infer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    And EVERY row is a detector row there, not just the surplus ones: they are score-ordered or
    association-ordered, so lending the first few the labels' own ids claims an identity nothing
    established.
    """
    model, sess, registry, name = scene
    w, h = sess.rig.size(sess.cam_names[0])
    boxes = np.zeros((5, 4, len(sess.rig), 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h

    for anchor in ('none', 'carry', 'labels'):
        out = run_group(model, sess, 'g000', registry, name, _cfg(anchor=anchor),
                        boxes_stc=boxes)
        assert out['pred'].shape[0] == 5
        # the id list must match the prediction row count, and name every row honestly
        assert len(out['animal_ids']) == 5
        assert all(str(i).startswith('det') for i in out['animal_ids'])
    # ...while the LABEL path keeps the labels' ids, which are identity there.
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))
    assert list(out['animal_ids']) == list(sess.labels('g000').animal_ids)


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


def test_the_window_union_covers_the_window_and_stays_in_the_image(scene):
    """The union box, which is DELIBERATELY not re-entered into the crop rule.

    Squaring a union of crop-rule boxes was measured at 3dpop +3.06 mm MPJPE / -0.032 MOTA (both
    SIG) because it grows the p90 box area by 82%; `run_group` carries the numbers. What the box
    must still be: the union over the window's frames, int32, and inside the image -- an off-frame
    box gives a negative `cam['offset']` and breaks `project_cam` far downstream.
    """
    model, sess, registry, name = scene
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    for t in range(4):
        boxes[0, t, :] = [10 + t, 12, 14 + t, 16]         # drifts right across the window

    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_stc=boxes)
    crop = out['crop'][0, 0]                              # (C,4), the first window's box per cam
    for ci in range(C):
        x0, y0, x1, y1 = crop[ci]
        assert (x0, y0, x1, y1) == (10.0, 12.0, 17.0, 16.0), \
            f'cam {ci}: {crop[ci]} is not the union over t = 0..3'
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h


def test_the_hoisted_decode_gives_the_same_pixels_as_cropping_at_decode(scene):
    """ONE DECODE PER (CAMERA, FRAME), shared by every animal in the window.

    The crop differs per animal and the decode does not, so the decode moved out of the animal
    loop and the crop became an affine on the already-decoded frame. That is only a speed change if
    it is bit-identical to what `load_image` produced when it did both at once.
    """
    from tailcyclenet.dataset import read_frames
    from tailcyclenet.infer import _crop_views

    _, sess, _, _ = scene
    group, cam = sess.groups['g000'], sess.cam_names[0]
    frames = [0, 1, 2, 1]                       # a repeat, which `read_frames` dedupes
    box = torch.tensor([3, 5, 40, 44], dtype=torch.int32)
    target = [32, 32]

    at_decode = np.asarray(read_frames(group, cam, frames, crop_coords=box, target_size=target))
    hoisted = _crop_views(read_frames(group, cam, frames), box, target)
    np.testing.assert_array_equal(hoisted[0].numpy(), at_decode)


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
    prior, qt = _build_prior(cfg, (pose, 23), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                             cgroup)

    assert torch.isfinite(prior[0, :2]).all(), 'in-crop keypoints must survive'
    assert torch.isnan(prior[0, 2:]).all(), 'out-of-crop keypoints must become NaN'
    assert qt.shape == (1, K)
    assert (qt == 3).all(), f'prompt frame 23 is index 3 of a window starting at 20, got {qt}'

    # A prompt from just before the window -- inside the overlap, so the ordinary case -- cannot be
    # expressed as a frame index and clamps into range rather than indexing off the front.
    _, early = _build_prior(cfg, (pose, 17), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                            cgroup)
    assert (early == 0).all()
    _, late = _build_prior(cfg, (pose, 999), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                           cgroup)
    assert (late == len(frames) - 1).all()

    # A KEYPOINT THE MODEL ITSELF DOUBTED, gated in the same currency (NaN = "I was not told").
    gated = InferConfig(anchor='carry', prior_vis_thresh=0.0)
    vis = torch.tensor([2.0, -1.0, 3.0, -5.0])
    pri, _ = _build_prior(gated, (pose, 23, vis), None, 0, 0, frames, boxes, [1.0], '2d', K,
                          2, cgroup)
    assert torch.isfinite(pri[0, 0]).all(), 'a confident in-crop keypoint must survive'
    assert torch.isnan(pri[0, 1]).all(), 'a keypoint below the logit threshold must be dropped'
    # ...and the default keeps it, so an unconfigured run is byte-identical.
    keep, _ = _build_prior(cfg, (pose, 23, vis), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                           cgroup)
    assert torch.isfinite(keep[0, 1]).all()

    # STALER THAN THE OVERLAP IS NOT A PRIOR. `carried` is only written by a window that predicted,
    # so an animal the box source lost for a few windows keeps offering a pose from before this
    # one, and the clamp above would present it as this window's first frame.
    assert _build_prior(cfg, (pose, 2), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                        cgroup) == (None, None)


def test_refine_recrops_to_its_own_prediction_and_keeps_the_coverage(scene):
    """The second pass must be crop-rule boxes around the FIRST pass's prediction, and an animal
    whose refined crop fails must keep the box it already had -- a bad prediction cannot be allowed
    to cost coverage a loose box was already giving."""
    model, sess, registry, name = scene
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h          # the whole frame: refinement must shrink it

    plain = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_stc=boxes)
    ref = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', refine=True),
                    boxes_stc=boxes)

    assert (ref['outcome'] == 0).all(), 'refinement must not drop an animal'
    got = np.isfinite(ref['pred']).all(-1)
    assert got.sum() >= np.isfinite(plain['pred']).all(-1).sum(), 'coverage must not fall'
    # `crop` KEEPS THE FIRST-PASS BOX -- it is the only record of what the box source offered, and
    # every coverage and crop-inflation number in reports 08 and 11 is computed from it. The refined
    # box is a second field, so both are readable rather than one overwriting the other.
    np.testing.assert_array_equal(np.nan_to_num(ref['crop'], nan=-1),
                                  np.nan_to_num(plain['crop'], nan=-1))
    cw = ref['crop_refined'][0, 0, :, 2] - ref['crop_refined'][0, 0, :, 0]
    ch = ref['crop_refined'][0, 0, :, 3] - ref['crop_refined'][0, 0, :, 1]
    assert (cw <= w).all() and (ch <= h).all()
    assert ((cw < w) | (ch < h)).any(), 'a whole-frame box should have been refined smaller'
    # And a refined box must OVERLAP the box it came from, or it is not a refinement of it -- a pose
    # that wandered squares into a box somewhere else entirely, at 2x the pose compute.
    for ci in range(C):
        b2, b1 = ref['crop_refined'][0, 0, ci], plain['crop'][0, 0, ci]
        if np.isfinite(b2).all() and np.isfinite(b1).all():
            assert min(b2[2], b1[2]) > max(b2[0], b1[0])
            assert min(b2[3], b1[3]) > max(b2[1], b1[1])


def test_the_row_gate_withholds_rows_but_not_the_carried_prompt(scene):
    """`--vis-thresh` gates what is REPORTED. Gating the prompt too would be a second lever in one
    flag, and it is `--prior-vis-thresh` that owns that one.

    A threshold above every logit must empty the prediction; one below every logit must leave it
    byte-identical, so an unconfigured run is unchanged.
    """
    model, sess, registry, name = scene
    base = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry'))
    lo = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', vis_thresh=-1e9))
    hi = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', vis_thresh=1e9))

    np.testing.assert_array_equal(np.isnan(lo['pred']), np.isnan(base['pred']))
    assert np.isnan(hi['pred']).all(), 'every row is below an infinite threshold'
    # The gate ran per window, and `carried` was taken from the ungated pose, so the windows after
    # the first still predicted -- which is what `outcome` all-ok proves.
    assert (hi['outcome'] == 0).all(), 'the gate must not abort a window'


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


def test_2d_is_bit_identical_under_either_carry_source(scene):
    """THE FREE INVARIANCE CHECK on de-looping the carry.

    In 2D there is nothing to triangulate at one camera and the grid head decodes ABSOLUTE pixel
    bins, so `coords_pred` is not built on top of the query and `carry_source` has nothing to
    change. Every 2D root -- calms21, rat-city, branson-fly -- must therefore come out unchanged,
    which is what confines the disambiguation risk of this change to the 3D roots.
    """
    model, sess, registry, name = scene
    if sess.mode != '2d':
        pytest.skip('2D only, by construction')
    torch.manual_seed(0)
    a = run_group(model, sess, 'g000', registry, name,
                  _cfg(anchor='carry', carry_source='triangulate'))
    torch.manual_seed(0)
    b = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', carry_source='pred'))
    np.testing.assert_array_equal(np.nan_to_num(a['pred'], nan=-9e9),
                                  np.nan_to_num(b['pred'], nan=-9e9))


def test_3d_carry_feeds_back_the_anchor_free_estimate(scene):
    """...and in 3D it must actually differ, or the de-loop is not wired to anything.

    `pred_tri` is the tensor `carry` now hands the next window: re-derived from each window's own
    pixels rather than `prior + residual`, which is a loop with gain. It also rides in the npz
    because nothing had ever scored the triangulation against the labels.
    """
    model, sess, registry, name = scene
    if sess.mode != '3d':
        pytest.skip('3D only')
    out = run_group(model, sess, 'g000', registry, name,
                    _cfg(anchor='carry', carry_source='triangulate'))
    assert 'pred_tri' in out and out['pred_tri'].shape == out['pred'].shape
    # A single-camera 3D window has no triangulation, and the fixture rig has three cameras, so
    # here it must be present -- and it is a different estimate from the reported one.
    assert np.isfinite(out['pred_tri']).any(), 'the anchor-free estimate must be recorded'
    assert not np.array_equal(np.nan_to_num(out['pred_tri'], nan=-9e9),
                              np.nan_to_num(out['pred'], nan=-9e9))


def test_seam_blend_averages_the_overlap_and_changes_only_the_overlap(scene):
    """The window seam is a discontinuity: measured at 3.46x the interior per-frame displacement on
    3dpop and 2.33x on johnson-mouse, because `last` reports each overlap frame from the window that
    saw it with the LEAST left-context.

    `blend` must (a) leave frames only one window decoded exactly as they were, and (b) move the
    overlap frames, or it is averaging nothing.
    """
    model, sess, registry, name = scene
    # n_frames=2, overlap=1 -- NOT `_cfg`'s defaults. The fixture groups are 4 frames long, so at
    # n_frames=4 `_window_starts` returns a SINGLE window and there is no overlap to blend: an earlier
    # version of this test guarded that with `pytest.skip` and therefore never ran on either fixture.
    # 2 is the floor (gotcha 1: T = 1 hits posetail's gT = T // tubelet = 0).
    NF, OV = 2, 1
    kw = dict(anchor='none', n_frames=NF, overlap=OV)
    last = run_group(model, sess, 'g000', registry, name,
                     InferConfig(image_size=64, min_crop_dim=16, device='cpu', seam='last', **kw))
    blend = run_group(model, sess, 'g000', registry, name,
                      InferConfig(image_size=64, min_crop_dim=16, device='cpu', seam='blend', **kw))
    assert blend['pred'].shape == last['pred'].shape
    # Coverage cannot fall: a frame one window decoded is that window's value, nan-aware.
    assert (np.isfinite(blend['pred']).all(-1).sum()
            >= np.isfinite(last['pred']).all(-1).sum())
    starts = last['window_start']
    assert len(starts) >= 2, f'the fixture must produce several windows, got {starts}'
    T = last['pred'].shape[1]
    covered = np.zeros(T, int)
    for s0 in starts:
        covered[s0:min(s0 + NF, T)] += 1
    assert (covered > 1).any(), 'the fixture must actually overlap'
    solo = covered == 1
    if solo.any():
        np.testing.assert_allclose(blend['pred'][:, solo], last['pred'][:, solo],
                                   rtol=1e-5, atol=1e-5)
    # ...and the overlap frames must actually MOVE, or nothing is being averaged.
    over = covered > 1
    ok = np.isfinite(blend['pred'][:, over]).all(-1) & np.isfinite(last['pred'][:, over]).all(-1)
    assert ok.any(), 'need a finite overlap frame to compare'
    assert not np.allclose(blend['pred'][:, over][ok], last['pred'][:, over][ok])


@pytest.mark.parametrize('argv,expect', [
    (['--anchor', 'labels', '--detector', 'nope'], 'not label rows'),
    (['--anchor', 'labels', '--boxes', 'nope.npz'], 'not label rows'),
    (['--oracle-corrupt', 'nonsense'], 'kind must be one of'),
    (['--oracle-corrupt', 'off'], 'needs an amount'),
    (['--oracle-corrupt', 'off:0.5', '--anchor', 'carry'], 'only means anything'),
    (['--prior-vis-thresh', '1.0', '--anchor', 'none'], 'only means anything'),
])
def test_the_cli_refuses_incoherent_combinations_before_loading_anything(cli, monkeypatch,
                                                                        argv, expect):
    """These are the combinations that used to run and produce a number instead of an error.

    The worst was `--anchor labels` with a box source: `run_group` seeds row `a` from LABEL row `a`,
    and detector rows are score- or association-ordered, so the arm whose whole purpose is to be an
    upper bound was being handed a DIFFERENT animal's ground truth.

    Checked through the real entry point -- the script's own `main`, its own argparse, its own
    guards -- but IN PROCESS rather than through six forks. Each fork re-imported
    torch/torchvision/posetail for 13.7 s to reach an argument check, which was 60% of the whole
    suite. `--run` still points at nothing, and that is what pins the property this test is for: a
    guard that moved after `load_run` raises FileNotFoundError instead of a SystemExit carrying
    `expect`, so it still fails here.
    """
    repo = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(repo / 'no_such_run'),
                                      '--data', str(repo), '--out', '/tmp/unused.npz'] + argv)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert expect in str(e.value), f'wanted {expect!r}, got: {e.value}'


def test_seam_blend_accumulates_a_repeated_frame_index():
    """`arr[[0, 0]] += x` applies only the LAST write, and `run_group` produces a repeated frame index.

    A group shorter than two frames is padded by repeating its index (`np.clip(np.arange(start,
    start + 2), ...)`), because T = 1 hits posetail's `gT = T // tubelet = 0` bug. Every other write in
    the loop is an assignment, where a repeat is harmless; the blend accumulator is the one place it is
    not, and numpy's fancy-index `+=` fails silently there rather than raising.
    """
    sum_ = np.zeros((1, 3, 1, 2))
    cnt = np.zeros((1, 3, 1), int)
    frames = np.array([0, 0])                      # what a one-frame group produces
    p = np.array([[[1.0, 1.0]], [[3.0, 3.0]]])     # two decodes of the same frame
    fin = np.isfinite(p).all(-1)
    np.add.at(sum_, (0, frames), np.where(fin[..., None], p, 0.0))
    np.add.at(cnt, (0, frames), fin)
    assert cnt[0, 0, 0] == 2, 'both decodes of the repeated frame must be counted'
    assert sum_[0, 0, 0, 0] == 4.0
    # The mean is then 2.0. A plain `+=` would have given count 1 and sum 3.0, i.e. it would have
    # thrown one decode away and still produced a plausible-looking number.
    naive_s = np.zeros_like(sum_); naive_c = np.zeros_like(cnt)
    naive_s[0, frames] += np.where(fin[..., None], p, 0.0)
    naive_c[0, frames] += fin
    assert naive_c[0, 0, 0] == 1 and naive_s[0, 0, 0, 0] == 3.0


def test_the_tracker_projects_correctly_on_a_moving_rig():
    """Gotcha 9's class: `project_points_torch` aligns a (T,4,4) extrinsic against axis -3 of the points,
    which for a (n_targets, 1, 3) reshape would be the TARGET axis, silently projecting target i through
    frame i's pose. `detect_group` avoids it by asking for one frame at a time -- `Session.cgroup(gid, t)`
    with a scalar returns a (4,4) ext even on a moving rig -- and that is the property pinned here,
    because it is a property of the CALLER and nothing in `track.py` would notice if it changed."""
    import conftest as cf
    from tailcyclenet.detector.track import _project
    from tailcyclenet.format import Session

    root = Path(tempfile.mkdtemp())
    cf._session_3d(root / 'mv' / 'test' / 's_moving', moving=True)
    sess = Session.load(root / 'mv' / 'test' / 's_moving')
    sess.preload()
    gid = list(sess.groups)[0]
    per_frame = sess.cgroup(gid, 0)
    assert all(c['ext'].ndim == 2 for c in per_frame), \
        'a scalar frame must give a (4,4) ext, or the projection axis alignment below is wrong'
    assert sess.cgroup(gid)[0]['ext'].ndim == 3, 'the whole-group form must still be per-frame'
    for n in (1, 3):
        out = _project(per_frame[0], np.zeros((n, 3), np.float32))
        assert out.shape == (n, 2)
