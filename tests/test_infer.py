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


@pytest.fixture(scope='module', params=['rat', 'mv'])
def multiwindow_scene(request):
    """The same two shapes as `scene`, but T=16 -- SEVERAL windows at n_frames=4/overlap=2

    (`_window_starts(16, 4, 2)` is 7 starts), which `scene`'s T=4 cannot exercise at all (one
    window only). This is what a prefetch-ahead pipeline needs to be tested against: with one
    window there is nothing to prefetch and `prefetch_windows` is a no-op by construction.
    """
    import conftest as cf

    root = Path(tempfile.mkdtemp())
    if request.param == 'rat':
        cf._session_2d(root / 'rat' / 'test' / 's', T=16)
    else:
        cf._session_3d(root / 'mv' / 'test' / 's', T=16, moving=True)
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


@pytest.mark.parametrize('anchor,refine', [('none', False), ('carry', False),
                                           ('none', True), ('carry', True)])
def test_prefetch_windows_is_bit_exact(multiwindow_scene, anchor, refine):
    """THE PIPELINE PROTOTYPE'S OWN CLAIM (dev/reports/31): decoding a future window ahead of
    the current one's forward must not move a single value, at any `prefetch_windows`.

    `_build_plans`/`decode_crops` depend only on the box source and window geometry, never on
    `carried` or any model output, so preparing window wi+1 while window wi forwards changes no
    pixel and no order -- `carried` is still read and written strictly in window order, on the
    main thread, inside `_process_window`. Checked under BOTH `refine` (the second decode pass)
    and `carry` (the prior path most likely to break under reordering), which is why this needs
    `multiwindow_scene` rather than `scene`: `scene`'s T=4 is one window and prefetching a
    single-window run is a no-op by construction.
    """
    model, sess, registry, name = multiwindow_scene
    base = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor=anchor, refine=refine, prefetch_windows=0))
    for pf in (1, 2, 5):
        got = run_group(model, sess, 'g000', registry, name,
                        _cfg(anchor=anchor, refine=refine, prefetch_windows=pf))
        assert set(got) == set(base)
        for k in base:
            va, vb = base[k], got[k]
            if isinstance(va, np.ndarray) and va.dtype.kind == 'f':
                assert np.array_equal(np.isnan(va), np.isnan(vb)), f'{k}: finite mask differs'
                fin = ~np.isnan(va)
                np.testing.assert_array_equal(va[fin], vb[fin], err_msg=f'{k} (prefetch={pf})')
            elif isinstance(va, np.ndarray):
                np.testing.assert_array_equal(va, vb, err_msg=f'{k} (prefetch={pf})')
            else:
                assert str(va) == str(vb), f'{k} (prefetch={pf})'


def test_prefetch_windows_default_matches_the_old_serial_loop(multiwindow_scene):
    """THE DEFAULT (`prefetch_windows = 1`) IS A NO-OP ON THE PREDICTION, not just a documented
    claim: a run with no new flag set must be indistinguishable from `prefetch_windows = 0`, the
    exact old code path where `run_group` never builds the prefetch pool at all.
    """
    model, sess, registry, name = multiwindow_scene
    default = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry'))   # default = 1
    old = run_group(model, sess, 'g000', registry, name,
                    _cfg(anchor='carry', prefetch_windows=0))
    for k in old:
        va, vb = old[k], default[k]
        if isinstance(va, np.ndarray) and va.dtype.kind == 'f':
            assert np.array_equal(np.isnan(va), np.isnan(vb))
            fin = ~np.isnan(va)
            np.testing.assert_array_equal(va[fin], vb[fin], err_msg=k)
        elif isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb, err_msg=k)
        else:
            assert str(va) == str(vb), k


def test_carry_requires_overlap(scene):
    model, sess, registry, name = scene
    with pytest.raises(ValueError, match='overlap'):
        run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', overlap=0))


def test_inflate_box_widens_about_its_centre():
    """The wide-crop deployment helper (report 27): 1.5x the side, same centre, clamped."""
    from tailcyclenet.infer import _inflate_box
    box = torch.tensor([100, 100, 140, 160], dtype=torch.int32)     # 40x60, centre (120,130)
    size = torch.tensor([1000, 1000])
    out = _inflate_box(box, size, 1.5)
    assert int((out[0] + out[2]) / 2) == 120 and int((out[1] + out[3]) / 2) == 130
    assert int(out[2] - out[0]) == 60 and int(out[3] - out[1]) == 90       # 1.5x
    # clamps to the image rather than going negative
    clamped = _inflate_box(torch.tensor([0, 0, 40, 40], dtype=torch.int32), size, 4.0)
    assert int(clamped[0]) == 0 and int(clamped[1]) == 0


def test_deploy_box_prompt_lands_inside_the_crop():
    """`_deploy_box_prompt` maps an animal's source points into the crop frame and boxes them."""
    from tailcyclenet.infer import _deploy_box_prompt
    source = torch.tensor([[[110., 110.], [130., 150.]]])          # (T=1,K=2,2) source px
    boxes = [torch.tensor([100, 100, 200, 200], dtype=torch.int32)]
    scales = [2.0]
    cam = {'size': torch.tensor([200, 200], dtype=torch.int32)}
    out = _deploy_box_prompt(source, boxes, scales, [cam], 'cpu')
    assert out.shape == (1, 1, 1, 4)
    x0, y0, x1, y1 = out[0, 0, 0].tolist()
    # source (110,110)->(20,20) and (130,150)->(60,100) in crop px; the box must bound them
    assert x0 <= 20 and y0 <= 20 and x1 >= 60 and y1 >= 100


def test_box_prompt_skipped_on_3d_and_multicam(scene):
    """Guard: box_prompt is wired for 2D single-camera only; on 3D / multi-camera it is SKIPPED
    with a warning (not an error), so a box model deploys gracefully on every root (report 27)."""
    model, sess, registry, name = scene
    if sess.mode != '3d':
        pytest.skip('needs the 3D scene')
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = run_group(model, sess, 'g000', registry, name,
                        _cfg(anchor='none', box_prompt='labels'))
    assert out['pred'].shape[0] >= 1                      # ran to completion, box skipped
    assert any('2D single-camera' in str(x.message) for x in w)


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

    # A PROMPT FROM BEFORE THE WINDOW IS RETIRED, not clamped. This used to assert the opposite
    # ("inside the overlap, so the ordinary case") and that reading was wrong: `j = len(frames) -
    # overlap` makes the carried frame the NEXT window's start exactly, so consecutive windows
    # give qt == 0 and a NEGATIVE qt happens only when a window was skipped. Clamping it to 0
    # presented a pose from a frame the model was never shown as this window's first frame, and in
    # 3D the bounds mask cannot catch that -- a stale pose still visible to two cameras passes.
    #
    # The old guard spent a budget of `overlap` frames, which meant it only fired when
    # `n_frames > 2 * overlap`: at the swept `--n-frames 24 --overlap 12` it never fired.
    assert _build_prior(cfg, (pose, 17), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                        cgroup) == (None, None)
    # ...and the ordinary case -- the carried frame IS this window's first -- is still qt 0.
    _, ordinary = _build_prior(cfg, (pose, 20), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                               cgroup)
    assert (ordinary == 0).all()
    _, late = _build_prior(cfg, (pose, 999), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                           cgroup)
    assert (late == len(frames) - 1).all()

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


def test_the_cli_runs_end_to_end_with_no_detector(cli, monkeypatch, tmp_path):
    """THE DEFAULT BOX PATH, THROUGH `main()`. Every other CLI test SystemExits inside argparse.

    That gap is not theoretical: `det_kpts` was initialised inside `if det is not None:` while its
    two siblings were initialised outside it, so every box source that is NOT a detector -- the
    GT-crop upper bound and the whole `--boxes` path -- raised `UnboundLocalError` at the
    `run_group` call, after paying the checkpoint load. Six CLI tests passed throughout.

    Asserts almost nothing about the numbers on purpose. What it pins is that every name in the
    group loop is bound on the path with no detector, which is the class of failure that shipped.
    """
    import conftest as cf
    from tailcyclenet.checkpoints import save_checkpoint, save_run_meta

    root = tmp_path / 'rat'
    cf._session_2d(root / 'test' / 's')
    ds = load_dataset(root)
    registry = Registry.build([ds])
    model = build_model(SMALL, n_keypoints=registry.n_keypoints)
    run = tmp_path / 'run'
    config = {'model': SMALL,
              'data': {'image_size': 64, 'min_crop_dim': 16, 'n_frames': 4, 'box_source': 'keypoints'}}
    save_run_meta(run, config, registry)
    save_checkpoint(run, 0, model, torch.optim.SGD(model.parameters(), lr=0.0), config)

    out = tmp_path / 'pred.npz'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(root),
                                      '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--out', str(out)])
    cli.main()
    assert out.exists()
    got = dict(np.load(out, allow_pickle=True))
    assert 's/g000|pred' in got, f'no prediction written; keys: {sorted(got)}'


@pytest.mark.parametrize('argv,expect', [
    (['--anchor', 'labels', '--detector', 'nope'], 'not label rows'),
    (['--anchor', 'labels', '--boxes', 'nope.npz'], 'not label rows'),
    (['--oracle-corrupt', 'nonsense'], 'kind must be one of'),
    (['--oracle-corrupt', 'off'], 'needs an amount'),
    (['--oracle-corrupt', 'off:0.5', '--anchor', 'carry'], 'only means anything'),
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


def test_instances_boxes_get_the_training_crop_rule_at_inference(scene):
    """`box_source = 'instances'` used two different crop rules on the two sides of the run.

    The loader routes a stored box through `_crop_source` -> `crop_box_for_points(..., pad=0)`,
    which SQUARES the extent and floors it at `min_crop_dim`. The window loop took the raw clamped
    union instead -- no squaring, no floor -- on the grounds that "a detector box IS a crop-rule
    box". True of a detector box, false of `instances.pq`: 96% of rat-city's stored boxes are
    non-square (aspect p50 1.737). After `_resize_camera` that puts the animal at a different
    scale on a different-aspect canvas than any crop the model was trained on -- gotcha 8's axis,
    and invisible in every number the run reports.

    The detector path must be UNCHANGED, because re-squaring an already-square union is measured
    +3.06 mm worse on 3dpop.
    """
    from tailcyclenet import crop as cropmod

    model, sess, registry, name = scene
    w, h = sess.rig.size(sess.cam_names[0])
    S, T, C = len(sess.labels('g000').animal_ids), 4, len(sess.rig)

    # A deliberately NON-SQUARE stored extent, which is the case the two rules disagree on.
    wide = np.tile(np.array([10.0, 10.0, 10.0 + 0.6 * w, 10.0 + 0.2 * h], np.float32),
                   (S, T, C, 1))
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_stc=wide)
    det_box = out['crop'][0, 0, 0]

    # The DETECTOR path keeps the raw union: it must NOT equal the squared rule here.
    assert det_box[2] - det_box[0] != det_box[3] - det_box[1], \
        'the detector union must stay non-square, or the measured +3.06 mm result is being undone'

    # ...and the `instances` path DOES go through the rule. conftest's 2D session ships one stored
    # box (animal 1, frame 1, 20x20 -- so `min_crop_dim` is what bites) and no others, which is
    # also the per-animal keypoint fallback.
    if sess.labels('g000').boxes is None:
        pytest.skip('this fixture carries no instances.pq')
    stored = sess.labels('g000').boxes
    a, t = [(i, j) for i in range(stored.shape[0]) for j in range(stored.shape[1])
            if np.isfinite(stored[i, j, 0]).all()][0]
    cfg = _cfg(anchor='none', box_source='instances')
    got = run_group(model, sess, 'g000', registry, name, cfg)

    # THE ACTUAL CLAIM: the box the window loop uses is the box the LOADER would have used, which
    # is `crop_box_for_points` over the stored corners at pad 0. Compared against the rule rather
    # than asserted square, because the rule also clamps into the frame -- on this 64x48 fixture
    # a squared box can come back non-square, and that is the rule's own behaviour, not a miss.
    x0, y0, x1, y1 = stored[a, t, 0]
    expect = cropmod.crop_box_for_points(
        torch.tensor([[x0, y0], [x1, y1]]), torch.tensor([int(w), int(h)]),
        cfg.min_crop_dim, pad=0)
    # `crop` is indexed by WINDOW, not by frame; this group is one window, and `t` is its only
    # finite stored frame, so the union over the window is that one box.
    np.testing.assert_array_equal(got['crop'][a, 0, 0], np.asarray(expect, np.float32))


def test_a_3d_render_uses_the_per_frame_camera_on_a_moving_rig(scene):
    """Gotcha 9's class, and the FIFTH camera-group builder to drop `moving_ext`.

    `render.project` took `Rig.by_name`, which carries `calibration.toml`'s single NOMINAL
    extrinsic; the per-frame ones exist only through `Session.cgroup(gid, frames)`. So `--render`
    on johnson-mouse -- or any `moving = true` session -- drew the skeleton somewhere the animal
    is not, which reads as a pose failure rather than as a render bug. `_fill_box_agreement` in
    the same package already did this correctly, so the two disagreed.

    `test_render_writes_every_predicted_frame` skips 3D entirely, and the `mv` fixture has been
    the moving one all along, so nothing here had ever executed this path.
    """
    from tailcyclenet.render import project

    model, sess, registry, name = scene
    if sess.mode != '3d':
        pytest.skip('needs the moving 3D rig')
    cam_name = sess.cam_names[0]
    assert sess.rig.moving.get(cam_name), 'the fixture must actually be moving'

    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))
    pred = out['pred']
    frames = np.arange(pred.shape[1])

    per_frame = project(sess, pred, 0, 'g000', frames)
    nominal = project(sess, pred, 0)                  # what it used to do
    assert per_frame.shape == pred.shape[:-1] + (2,)

    ok = np.isfinite(per_frame) & np.isfinite(nominal)
    assert ok.any(), 'need a finite projection to compare'
    assert not np.allclose(per_frame[ok], nominal[ok]), \
        'the two must differ on a moving rig, or the per-frame extrinsic is not being used'

    # AND IT MATCHES THE PROJECTION THE PIPELINE ALREADY TRUSTS -- `infer`'s own, per frame.
    from posetail.posetail.cube import project_points_torch
    cams = sess.cgroup('g000', frames)
    want = project_points_torch([cams[0]], torch.as_tensor(pred[0], dtype=torch.float32))[0]
    np.testing.assert_allclose(per_frame[0], want.numpy(), rtol=1e-5, atol=1e-4)




def test_refine_px_off_is_bit_identical(scene):
    """INERTNESS FIRST, and it must be bit-identical, not "within noise".

    `--refine-px` is default-None and everything below it is gated on `px < image_size`, so
    `--refine` alone must reproduce today's output exactly. `refine_px == image_size` is asserted
    separately because the model's gate is a strict `<`: the two have to meet at the boundary or
    the flag has an off-by-one regime nobody would ever run deliberately.
    """
    model, sess, registry, name = scene
    torch.manual_seed(0)
    a = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', refine=True))
    torch.manual_seed(0)
    b = run_group(model, sess, 'g000', registry, name,
                  _cfg(anchor='none', refine=True, refine_px=None))
    torch.manual_seed(0)
    c = run_group(model, sess, 'g000', registry, name,
                  _cfg(anchor='none', refine=True, refine_px=64))   # == _cfg's image_size
    for other, why in ((b, 'refine_px=None'), (c, 'refine_px == image_size')):
        for k in ('pred', 'crop', 'crop_refined', 'box_agree'):
            np.testing.assert_array_equal(np.nan_to_num(a[k], nan=-9e9),
                                          np.nan_to_num(other[k], nan=-9e9),
                                          err_msg=f'{why} moved {k}')


def test_a_smaller_input_is_compensated_at_all_four_sites(scene):
    """THE SEAM ITSELF, pinned without depending on what the weights predict.

    `image_size` stands for three things and only one -- the pixel extent of the input -- is wrong
    for a smaller crop. This asserts the wiring of all of it: the pad target and the two
    `self.image_size` reads inside the library's forward (`:661` gauge centre, `:697` gridresid
    normaliser) move to `px`; `undistort_points` (`:614`) is wrapped; the 2D decode comes back
    scaled by `px / decoder.image_size`; and every one of those globals is restored afterwards,
    including on the exception path.

    The MAGNITUDES are measured on real weights in `scratch/refine3d/RESULT.md` -- 45.2 mm on the
    triangulation, 1.3334 = 256/192 on the residual. A random fixture model cannot pin those; it
    can pin that each correction is applied exactly once, which is the part that rots.
    """
    from posetail.posetail import tracker_encoder as te

    from tailcyclenet.model import _input_extent

    model, sess, registry, name = scene
    pad, full, head = model.transform_norm.transforms[0], model.image_size, model.decoder.image_size
    stock_undistort = te.undistort_points
    seen = {}
    with _input_extent(model, 32):
        seen = dict(pad=pad.size, size=model.image_size, wrapped=te.undistort_points)
    assert seen == dict(pad=32, size=32, wrapped=seen['wrapped'])
    assert seen['wrapped'] is not stock_undistort, 'the :614 frame mismatch is not corrected'
    assert (pad.size, model.image_size, te.undistort_points) == (full, full, stock_undistort)

    with pytest.raises(RuntimeError):
        with _input_extent(model, 32):
            raise RuntimeError('boom')
    assert (pad.size, model.image_size, te.undistort_points) == (full, full, stock_undistort), \
        'a forward that raised must not leave the library monkeypatched for the next one'

    # ...and the 2D rescale, through the public `forward`, with `_forward` stubbed so the assertion
    # is about the factor and not about the weights.
    if sess.mode != '2d':
        return
    one = torch.ones(1, 2, 3, 2)
    model._forward = lambda *a, **kw: {'coords_pred': one.clone()}
    try:
        cg = [{'size': torch.tensor([32, 32])}]
        got = model.forward([torch.zeros(1)], torch.zeros(1, 3, dtype=torch.long), cg, '2d')
        torch.testing.assert_close(got['coords_pred'], one * (32 / head))
        cg = [{'size': torch.tensor([full, full])}]
        got = model.forward([torch.zeros(1)], torch.zeros(1, 3, dtype=torch.long), cg, '2d')
        torch.testing.assert_close(got['coords_pred'], one)   # the gate is a strict <
    finally:
        del model._forward


def test_a_rejected_refinement_falls_back_at_full_resolution(scene):
    """THE FLAGSHIP SILENT BUG, and the existing refine test cannot catch it.

    `test_refine_recrops_to_its_own_prediction_and_keeps_the_coverage` uses whole-frame boxes, so
    `_overlaps` always passes and the fallback branch never runs. Here the box is a corner far from
    where the random model predicts, so the refined box misses it and the plan falls back -- and
    the fallback used to re-append the PASS-1 plan verbatim, carrying its reduced-resolution
    camera. Pass 2 then ran at pass 1's resolution, on exactly the animal whose first pass had
    already gone wrong.
    """
    model, sess, registry, name = scene
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    boxes[..., 0], boxes[..., 1] = 0, 0
    boxes[..., 2], boxes[..., 3] = w // 8, h // 8      # a corner: refinement should be rejected

    seen = []
    inner = model.forward

    def spy(views, *a, **kw):
        seen.append(int(views[0].shape[-2]))
        return inner(views, *a, **kw)

    model.forward = spy
    try:
        run_group(model, sess, 'g000', registry, name,
                  _cfg(anchor='none', refine=True, refine_px=32), boxes_stc=boxes)
    finally:
        model.forward = inner

    assert 32 in seen, 'pass 1 must have run at the reduced resolution'
    assert 64 in seen, ('every SECOND pass must run at image_size, including the ones whose '
                        'refinement was rejected -- that fallback is the silent half')
