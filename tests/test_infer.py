"""THE window loop and the inference CLI.

The model is random and tiny: these assert that the loop RUNS and returns the right shapes,
identities and byte-identities, never that the numbers are good.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from tailcyclenet.format import Registry, load_dataset
from tailcyclenet.infer.window import InferConfig, run_group
from tailcyclenet.model import build_model
from test_model import SMALL


def _scene(tmp_path_factory, kind, T):
    """(model, session, registry, dataset name): `rat` = 2D, 2 animals, 1 camera; `mv` = 3D on a
    moving 3-camera rig. T=4 is ONE window at n_frames=4; T=16 is seven (overlap 2)."""
    import conftest as cf

    root = tmp_path_factory.mktemp(f'{kind}{T}')
    if kind == 'rat':
        cf._session_2d(root / 'rat' / 'test' / 's', T=T)
    else:
        cf._session_3d(root / 'mv' / 'test' / 's', T=T, moving=True)
    ds = load_dataset(root / kind)
    registry = Registry.build([ds])
    sess = ds.sessions['test'][0]
    sess.preload()
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()
    return model, sess, registry, ds.name


@pytest.fixture(scope='module')
def rat(tmp_path_factory):
    """2D multi-animal session, T=4 (one window)."""
    return _scene(tmp_path_factory, 'rat', 4)


@pytest.fixture(scope='module')
def mv(tmp_path_factory):
    """3D moving-rig session, T=4 (one window)."""
    return _scene(tmp_path_factory, 'mv', 4)


@pytest.fixture(scope='module')
def rat16(tmp_path_factory):
    """2D session, T=16 -- several windows, which prefetch/seam/carry tests need."""
    return _scene(tmp_path_factory, 'rat', 16)


@pytest.fixture(scope='module')
def mv16(tmp_path_factory):
    """3D moving-rig session, T=16."""
    return _scene(tmp_path_factory, 'mv', 16)


@pytest.fixture(scope='module', params=['rat', 'mv'])
def scene(request):
    """Both shapes, for the few invariants whose 2D and 3D code paths differ."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope='module')
def cli():
    """The inference CLI module (`scripts/infer.py` just calls its `main`)."""
    from tailcyclenet.infer import cli as mod
    return mod


def _boxes_for(boxes):
    """Adapt a whole-clip (S,T,C,4) array to `run_blocks`' `boxes_for` callback."""
    def f(store, lo, hi):
        return (boxes[:, lo:hi], None)
    return f


def _cfg(**kw):
    kw.setdefault('overlap', 2)
    return InferConfig(n_frames=4, image_size=64, min_crop_dim=16, device='cpu', **kw)


def _assert_same(a, b, msg=''):
    """Two run_group/merge_blocks outputs are identical in every key (NaN-mask aware)."""
    assert set(a) == set(b)
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) and va.dtype.kind == 'f':
            assert np.array_equal(np.isnan(va), np.isnan(vb)), f'{k}{msg}: finite mask differs'
            fin = ~np.isnan(va)
            np.testing.assert_array_equal(va[fin], vb[fin], err_msg=f'{k}{msg}')
        elif isinstance(va, np.ndarray):
            np.testing.assert_array_equal(va, vb, err_msg=f'{k}{msg}')
        else:
            assert str(va) == str(vb), f'{k}{msg}'


def _save_run(tmp_path, registry, model_cfg=SMALL):
    """A loadable run folder (config + registry + one checkpoint) for a random SMALL model."""
    from tailcyclenet.checkpoints import save_checkpoint, save_run_meta

    model = build_model(model_cfg, n_keypoints=registry.n_keypoints)
    run = tmp_path / 'run'
    config = {'model': model_cfg, 'data': {'image_size': 64, 'min_crop_dim': 16, 'n_frames': 4,
                                           'box_source': 'keypoints'}}
    save_run_meta(run, config, registry)
    path = save_checkpoint(run, 0, model, torch.optim.SGD(model.parameters(), lr=0.0), config,
                           registry=registry)
    return run, path


def _cli_setup(tmp_path, kind='rat', name=None, T=4, model_cfg=SMALL, **kw):
    """One session under `<tmp>/<name>/test/s` plus a run folder; returns (session dir, run)."""
    import conftest as cf

    root = tmp_path / (name or kind)
    if kind == 'rat':
        cf._session_2d(root / 'test' / 's', T=T)
    else:
        cf._session_3d(root / 'test' / 's', T=T, **kw)
    run, _ = _save_run(tmp_path, Registry.build([load_dataset(root)]), model_cfg)
    return root / 'test' / 's', run


def _no_load(monkeypatch):
    """Make `load_run` fail loudly, so a refusal is proven to fire before the checkpoint loads."""
    def boom(*a, **k):
        raise AssertionError('the refusal must fire before the checkpoint loads')
    monkeypatch.setattr('tailcyclenet.infer.driver.load_run', boom)


def test_parser_defaults_select_the_measured_path():
    """Speed/tracking defaults are the measured ones (bf16, camera batch, --track, --link-boxes)."""
    from tailcyclenet.infer.cli import build_parser

    parser = build_parser()
    assert parser.get_default('precision') == 'bf16'
    assert parser.get_default('units') is None, 'units are resolved only after deriving video mode'
    assert parser.get_default('camera_batch') is True
    assert parser.get_default('allow_pose_scorer_mismatch') is False
    assert parser.get_default('track') is True and parser.get_default('link_boxes') is True
    assert parser.get_default('max_age') == 8 and parser.get_default('max_move') == 1.25
    options = {option for action in parser._actions for option in action.option_strings}
    assert {'--camera-batch', '--no-camera-batch', '--allow-pose-scorer-mismatch'} <= options


def test_detector_inference_without_keypoint_table(tmp_path):
    """A deployment detector can drive pose inference without 2D pose labels."""
    import conftest as cf

    session_path = tmp_path / 'portable' / 'train' / 'session'
    cf._session_2d(session_path, T=4, S=2)
    (session_path / 'keypoints.pq').unlink()
    ds = load_dataset(session_path.parent.parent)
    sess = ds.sessions['train'][0]
    registry = Registry.build([ds])
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()
    boxes = np.zeros((1, 4, 1, 4), np.float32)
    boxes[..., 2:] = 32

    out = run_group(model, sess, 'g000', registry, ds.name,
                    _cfg(anchor='none', refine=False), boxes_for=_boxes_for(boxes), n_rows=1)

    assert out['pred'].shape == (1, 4, sess.n_keypoints, 2)
    assert out['animal_ids'] == ['det00']


def test_scorer_encoder_mismatch_requires_explicit_opt_in():
    from tests.test_scorer_model import _scorer
    from tailcyclenet.infer.driver import _share_pose_scorer_encoder

    pose = build_model(SMALL, n_keypoints=5).eval()
    scorer = _scorer(5, stride_length=4).eval()
    with pytest.raises(SystemExit, match='--allow-pose-scorer-mismatch'):
        _share_pose_scorer_encoder(pose, scorer, allow_mismatch=False,
                                   pose_run='pose-run', scorer_run='scorer-run')
    assert not _share_pose_scorer_encoder(pose, scorer, allow_mismatch=True,
                                          pose_run='pose-run', scorer_run='scorer-run')
    assert scorer.scene_encoder.encoder is not pose.scene_encoder.encoder


def test_integrated_scorer_shares_scene_and_preserves_pose(rat):
    model, sess, registry, name = rat
    from tests.test_scorer_model import _scorer
    from tailcyclenet.model import share_video_encoder_if_identical

    scorer = _scorer(registry.n_keypoints, stride_length=4,
                     output_granularity='frame').eval()
    scorer.scene_encoder.encoder.load_state_dict(model.scene_encoder.encoder.state_dict())
    assert share_video_encoder_if_identical(model, scorer)
    cfg = _cfg(anchor='none', refine=False)
    pose_only = run_group(model, sess, 'g000', registry, name, cfg)
    integrated = run_group(model, sess, 'g000', registry, name, cfg, scorer=scorer,
                           shared_scene_encoder=True)
    np.testing.assert_array_equal(pose_only['pred'], integrated['pred'])
    assert integrated['quality'].shape == pose_only['pred'].shape[:-1]
    assert integrated['quality_precision'].shape == integrated['quality'].shape
    assert np.isfinite(integrated['quality']).any()


@pytest.mark.parametrize('anchor', ['self', 'labels'])
def test_every_anchor_runs(scene, anchor):
    """`self`/`labels` run on both shapes (`none`/`carry` run in nearly every other test)."""
    model, sess, registry, name = scene
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor=anchor))
    R = 3 if sess.mode == '3d' else 2
    assert out['pred'].shape == (len(sess.labels('g000').animal_ids), 4, sess.n_keypoints, R)
    assert len(out['animal_ids']) == out['pred'].shape[0]


def test_more_detector_rows_than_label_rows(rat):
    """A detector row is not a label row: more rows than labels must not IndexError, and every
    detector row is named `det*` (the label path keeps label ids)."""
    model, sess, registry, name = rat
    w, h = sess.rig.size(sess.cam_names[0])
    boxes = np.zeros((5, 4, len(sess.rig), 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h

    for anchor in ('none', 'carry', 'labels'):
        out = run_group(model, sess, 'g000', registry, name, _cfg(anchor=anchor),
                        boxes_for=_boxes_for(boxes), n_rows=len(boxes))
        assert out['pred'].shape[0] == 5
        # the id list must match the prediction row count, and name every row honestly
        assert len(out['animal_ids']) == 5
        assert all(str(i).startswith('det') for i in out['animal_ids'])
    # ...while the LABEL path keeps the labels' ids, which are identity there.
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))
    assert list(out['animal_ids']) == list(sess.labels('g000').animal_ids)


def test_a_camera_without_a_box_does_not_drop_the_animal(mv):
    """A camera with no box leaves the animal predicted from the cameras that saw it (was coverage 0)."""
    model, sess, registry, name = mv
    C = len(sess.rig)
    w, h = sess.rig.size(sess.cam_names[0])
    boxes = np.zeros((1, 4, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h
    boxes[:, :, 0] = np.nan                       # camera 0 saw nothing

    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    assert np.isfinite(out['pred']).all(-1).any(), 'the surviving cameras must still predict'


def test_the_window_union_covers_the_window_and_stays_in_the_image(mv):
    """The union box (deliberately NOT re-squared, +3.06 mm) spans the window and stays in the image."""
    model, sess, registry, name = mv
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    for t in range(4):
        boxes[0, t, :] = [10 + t, 12, 14 + t, 16]         # drifts right across the window

    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    crop = out['crop'][0, 0]                              # (C,4), the first window's box per cam
    for ci in range(C):
        x0, y0, x1, y1 = crop[ci]
        assert (x0, y0, x1, y1) == (10.0, 12.0, 17.0, 16.0), \
            f'cam {ci}: {crop[ci]} is not the union over t = 0..3'
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h


def test_the_hoisted_decode_gives_the_same_pixels_as_cropping_at_decode(rat):
    """One decode per (camera, frame): cropping the decoded frame equals cropping at decode, bit for bit."""
    from tailcyclenet.data.pixels import read_frames
    from tailcyclenet.infer.window import _crop_views

    _, sess, _, _ = rat
    group, cam = sess.groups['g000'], sess.cam_names[0]
    frames = [0, 1, 2, 1]                       # a repeat, which `read_frames` dedupes
    box = torch.tensor([3, 5, 40, 44], dtype=torch.int32)
    target = [32, 32]

    at_decode = np.asarray(read_frames(group, cam, frames, crop_coords=box, target_size=target))
    hoisted = _crop_views(read_frames(group, cam, frames), box, target)
    np.testing.assert_array_equal(hoisted[0].numpy(), at_decode)


def test_group_shorter_than_the_overlap(rat):
    """`p[-overlap]` ran off the front of a window with fewer frames than the step."""
    model, sess, registry, name = rat
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', overlap=8))
    assert out['pred'].shape[1] == 4


def test_kpt_chunk_matches_unchunked_end_to_end(mv):
    """Chunking is a memory knob, so it must not move the prediction by so much as an ulp."""
    model, sess, registry, name = mv
    whole = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))['pred']
    chunked = run_group(model, sess, 'g000', registry, name,
                        _cfg(anchor='none', kpt_chunk=2))['pred']
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-5)


def test_prefetch_windows_is_bit_exact(rat16):
    """Prefetching future windows moves nothing: the default (1) and 3 equal the serial loop (0)."""
    model, sess, registry, name = rat16
    base = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor='carry', refine=True, prefetch_windows=0))
    _assert_same(base, run_group(model, sess, 'g000', registry, name,
                                 _cfg(anchor='carry', refine=True)), ' (default prefetch)')
    _assert_same(base, run_group(model, sess, 'g000', registry, name,
                                 _cfg(anchor='carry', refine=True, prefetch_windows=3)),
                 ' (prefetch=3)')


def test_overlap_trace_round_trips_both_estimates_and_its_census(tmp_path):
    """`--overlap-trace` writes exactly what the hook recorded: both estimates, the census (incl. what
    was NOT seen), the prior-OOB and model-confidence probes; an empty capture keeps its census."""
    import json

    from tailcyclenet.infer.driver import _write_overlap_trace

    rows = [{'animal': 0, 'frame': 12, 'pos': 0, 'dist': 1.5, 'n_shared': 3,
             'prev': np.arange(6, dtype=np.float32).reshape(3, 2),
             'new': np.arange(6, dtype=np.float32).reshape(3, 2) + 1.5},
            {'animal': 1, 'frame': 13, 'pos': 1, 'dist': 0.25, 'n_shared': 2,
             'prev': np.zeros((3, 2), np.float32),
             'new': np.full((3, 2), 0.25, np.float32)}]
    census = {'pairs_seen': 2, 'pairs_no_prev': 7, 'pairs_no_shared': 1,
              'block_boundary_seams': 3, 'block_boundary_frames': 12}
    out = tmp_path / 'sub' / 'ovl.npz'
    _write_overlap_trace(out, {"s/g000": (rows, census, [(0, 8, 2, 17)], [(0, 8, -0.5, -0.9, 3)])})

    z = np.load(out, allow_pickle=True)
    assert list(z['animal']) == [0, 1] and list(z['frame']) == [12, 13]
    assert list(z['pos']) == [0, 1] and list(z['n_shared']) == [3, 2]
    np.testing.assert_allclose(z['dist'], [1.5, 0.25])
    assert z['prev'].shape == (2, 3, 2) and z['new'].shape == (2, 3, 2)
    np.testing.assert_allclose(z['new'][0] - z['prev'][0], 1.5)
    got = json.loads(str(z['census']))['s/g000']
    assert got == census, got
    assert list(z['oob_animal']) == [0] and list(z['oob_frame']) == [8]
    assert list(z['oob_nan']) == [2] and list(z['oob_k']) == [17]
    assert list(z['conf_animal']) == [0] and list(z['conf_frame']) == [8]
    np.testing.assert_allclose(z['conf_mean'], [-0.5])
    np.testing.assert_allclose(z['conf_min'], [-0.9])

    empty = tmp_path / 'empty.npz'
    _write_overlap_trace(empty, {'s/g000': ([], {'pairs_seen': 0, 'pairs_no_prev': 4}, [], [])})
    ze = np.load(empty, allow_pickle=True)
    assert ze['dist'].size == 0 and ze['prev'].size == 0
    assert json.loads(str(ze['census']))['s/g000']['pairs_no_prev'] == 4


def test_capture_overlap_agreement_is_a_side_channel_with_an_honest_census(rat16):
    """The opt-in seam capture is off by default and never moves a prediction; when on it records
    both seam values (distance reproducible from them) and a census that agrees with its rows."""
    from tailcyclenet.infer.window import merge_blocks, run_blocks

    model, sess, registry, name = rat16
    cfg = _cfg(anchor='carry')
    stats_off = {}
    off = merge_blocks(run_blocks(model, sess, 'g000', registry, name, cfg, stats=stats_off))
    assert 'overlap_agreement' not in stats_off, 'the flag defaults off'
    stats = {'capture_overlap_agreement': True}
    on = merge_blocks(run_blocks(model, sess, 'g000', registry, name, cfg, stats=stats))
    _assert_same(off, on)

    rows = stats.get('overlap_agreement', [])
    assert rows, 'a multi-window run with overlap > 0 must record at least one seam comparison'
    for r in rows:
        assert isinstance(r['animal'], int) and isinstance(r['frame'], int)
        assert r['dist'] >= 0.0 and np.isfinite(r['dist'])
        assert 0 <= r['frame'] < 16
        assert r['prev'].shape == r['new'].shape and r['prev'].ndim == 2
        assert 0 <= r['pos'] < cfg.n_frames, f"pos {r['pos']} outside the window"
        assert 1 <= r['n_shared']
    r = rows[0]
    shared = np.isfinite(r['prev']).all(-1) & np.isfinite(r['new']).all(-1)
    d = np.linalg.norm(r['new'] - r['prev'], axis=-1)
    np.testing.assert_allclose(d[shared].mean(), r['dist'], rtol=1e-6)

    census = stats['overlap_census']
    for k in ('pairs_seen', 'pairs_no_prev', 'pairs_no_shared'):
        assert k in census and census[k] >= 0, census
    assert census['pairs_seen'] == len(rows) > 0


def test_refine_records_each_overlap_trace_probe_once(rat16):
    """`--refine` runs `forward` twice per (animal, window); the overlap-trace probes
    (`prior_oob`, `model_conf`) must record each (animal, window) once, not once per pass."""
    from tailcyclenet.infer.window import merge_blocks, run_blocks

    model, sess, registry, name = rat16
    counts = {}
    for refine in (False, True):
        stats = {'capture_overlap_agreement': True}
        merge_blocks(run_blocks(model, sess, 'g000', registry, name,
                                _cfg(anchor='carry', refine=refine), stats=stats))
        oob = [(a, f) for a, f, *_ in stats.get('prior_oob', [])]
        assert len(oob) == len(set(oob))
        counts[refine] = (len(oob), len(stats.get('model_conf', [])))
    assert counts[True][0] == counts[False][0] > 0
    if counts[False][1]:
        assert counts[True][1] == counts[False][1]


def test_carry_needs_overlap_but_self_does_not(rat):
    """`carry` at overlap 0 is refused; `self` carries nothing, so overlap 0 is valid for it."""
    model, sess, registry, name = rat
    with pytest.raises(ValueError, match='overlap'):
        run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', overlap=0))
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='self', overlap=0))
    assert np.isfinite(out['pred']).any()


def test_inflate_box_widens_about_its_centre():
    """THE ONE INFLATION RULE, shared with the wide-crop training regime
    (`dataset.LoaderConfig.crop_inflate`) so the model trains and deploys on the same geometry."""
    from tailcyclenet.crop import inflate_box as _inflate_box
    box = torch.tensor([100, 100, 140, 160], dtype=torch.int32)     # 40x60, centre (120,130)
    size = torch.tensor([1000, 1000])
    out = _inflate_box(box, size, 1.5)
    assert int((out[0] + out[2]) / 2) == 120 and int((out[1] + out[3]) / 2) == 130
    assert int(out[2] - out[0]) == 60 and int(out[3] - out[1]) == 90       # 1.5x
    # clamps to the image rather than going negative
    clamped = _inflate_box(torch.tensor([0, 0, 40, 40], dtype=torch.int32), size, 4.0)
    assert int(clamped[0]) == 0 and int(clamped[1]) == 0


def test_deploy_box_prompt_lands_inside_the_crop_2d():
    """`_deploy_box_prompt` maps an animal's source points into the crop frame and boxes them,
    2D single-camera -- the path this function has always covered, unchanged in result."""
    from tailcyclenet.infer.window import _deploy_box_prompt
    source = np.array([[[[110., 110.], [130., 150.]]]])            # (S=1,T=1,K=2,2) source px
    boxes = [torch.tensor([100, 100, 200, 200], dtype=torch.int32)]
    scales = [2.0]
    cam = {'size': torch.tensor([200, 200], dtype=torch.int32)}
    out = _deploy_box_prompt('2d', source, None, np.array([0]), 0, [0], boxes, scales, [cam],
                             'cpu')
    assert out.shape == (1, 1, 1, 4)
    x0, y0, x1, y1 = out[0, 0, 0].tolist()
    # source (110,110)->(20,20) and (130,150)->(60,100) in crop px; the box must bound them
    assert x0 <= 20 and y0 <= 20 and x1 >= 60 and y1 >= 100


def test_deploy_box_prompt_3d_labels_matches_the_training_computation(mv):
    """3D `labels` box prompt is byte-identical to `compute_box_prompt` (deploy/train parity)."""
    from tailcyclenet import box_prompt as bpmod
    from tailcyclenet.infer.window import _deploy_box_prompt

    _, sess, _, _ = mv
    frames = np.arange(4)
    cgroup = sess.cgroup('g000', frames)
    lab = sess.labels('g000')
    world = lab.points3d                                # (S,T,K,3)

    got = _deploy_box_prompt('3d', world, None, frames, 0, list(range(len(cgroup))), None, None,
                             cgroup, 'cpu')
    want = bpmod.compute_box_prompt(torch.as_tensor(world[0]), cgroup, '3d')[None]
    torch.testing.assert_close(got, want.to(got.dtype))


def test_box_prompt_live_on_3d_multicam(mv):
    """The box is live on 3D per camera: the encoder receives a (1,T,C,4) box, no warning."""
    model, sess, registry, name = mv
    import warnings

    seen = {}
    orig = model.query_encoder.forward

    def spy(*a, **kw):
        box = getattr(model.query_encoder, '_box_prompt', None)
        if box is not None:
            seen['shape'] = tuple(box.shape)
        return orig(*a, **kw)

    model.query_encoder.forward = spy
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = run_group(model, sess, 'g000', registry, name,
                            _cfg(anchor='none', box_prompt='labels'))
    finally:
        model.query_encoder.forward = orig
    assert out['pred'].shape[0] >= 1
    assert not any('skipped' in str(x.message) for x in w)
    assert 'shape' in seen and seen['shape'][2] == len(sess.rig)
    assert out['box_prompt_cams'][0, 0] >= 1


def test_box_prompt_detector_per_camera_degradation():
    """A camera with no finite detector box yields a NaN column (the no-box token), not an exception."""
    from tailcyclenet.infer.window import _deploy_box_prompt

    T = 3
    cgroup = [dict(size=torch.tensor([64, 48], dtype=torch.int32)) for _ in range(2)]
    boxes = [torch.tensor([0, 0, 64, 48], dtype=torch.int32) for _ in range(2)]
    scales = [1.0, 1.0]
    # (S=1,T,C=2,4) source px: camera 0 has a real box every frame, camera 1 is all-NaN.
    boxes_stc = np.full((1, T, 2, 4), np.nan, np.float32)
    boxes_stc[0, :, 0] = [10., 10., 30., 30.]
    frames = np.arange(T)

    got = _deploy_box_prompt('3d', None, boxes_stc, frames, 0, [0, 1], boxes, scales, cgroup,
                             'cpu')
    assert got.shape == (1, T, 2, 4)
    assert torch.isfinite(got[:, :, 0]).all()           # camera 0: real box every frame
    assert torch.isnan(got[:, :, 1]).all()              # camera 1: nothing finite -> NaN column


def test_box_prompt_column_order_matches_use():
    """Column i of the box prompt is built from `use[i]`'s own box/scale/cgroup, not a neighbour's."""
    from tailcyclenet import box_prompt as bpmod
    from tailcyclenet.infer.window import _deploy_box_prompt

    T = 2
    cgroup = [dict(size=torch.tensor([64, 48], dtype=torch.int32)),
              dict(size=torch.tensor([32, 96], dtype=torch.int32))]
    boxes = [torch.tensor([0, 0, 64, 48], dtype=torch.int32),
             torch.tensor([100, 100, 132, 196], dtype=torch.int32)]
    scales = [1.0, 2.0]
    # use = [1, 0]: camera index 1 (source cam 1) is requested FIRST, camera 0 second.
    boxes_stc = np.full((1, T, 2, 4), np.nan, np.float32)
    boxes_stc[0, :, 0] = [10., 10., 30., 30.]           # source cam 0's own detector box
    boxes_stc[0, :, 1] = [110., 104., 126., 140.]       # source cam 1's own detector box
    frames = np.arange(T)

    got = _deploy_box_prompt('3d', None, boxes_stc, frames, 0, [1, 0],
                             [boxes[1], boxes[0]], [scales[1], scales[0]],
                             [cgroup[1], cgroup[0]], 'cpu')
    assert got.shape == (1, T, 2, 4)

    from tailcyclenet import crop as cropmod

    def expected(src_ci, origin, scale, size):
        db = torch.as_tensor(boxes_stc[0, 0, src_ci], dtype=torch.float32)
        corners = cropmod.box_corners(db)
        cf = (corners - torch.as_tensor(origin, dtype=torch.float32)) * float(scale)
        return cropmod.crop_box_for_points(cf, torch.as_tensor(size), bpmod.BOX_PROMPT_MIN_DIM,
                                           bpmod.BOX_PROMPT_PAD)

    # column 0 (use[0] = source cam 1): its OWN box/scale/cgroup, not source cam 0's
    want0 = expected(1, [100, 100], 2.0, [32, 96])
    torch.testing.assert_close(got[0, 0, 0], want0.to(got.dtype))
    # column 1 (use[1] = source cam 0): its OWN box/scale/cgroup
    want1 = expected(0, [0, 0], 1.0, [64, 48])
    torch.testing.assert_close(got[0, 0, 1], want1.to(got.dtype))
    # and swapped indices must NOT coincide -- the failure mode this guards against
    assert not torch.allclose(got[0, 0, 0], want1.to(got.dtype))


def test_box_prompt_none_is_byte_identical_on_2d_and_3d(mv):
    """`box_prompt='none'` + `crop_inflate=1.0` is byte-identical to never mentioning the box."""
    model, sess, registry, name = mv
    base = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))
    off = run_group(model, sess, 'g000', registry, name,
                    _cfg(anchor='none', box_prompt='none', crop_inflate=1.0))
    np.testing.assert_array_equal(base['pred'], off['pred'])
    assert 'box_prompt_cams' not in off


def test_box_prompt_first_only_withholds_it_once_carried(rat16):
    """`first_only` gives the box at an animal's first window and withholds it once carry holds;
    off, it fires every window."""
    model, sess, registry, name = rat16
    cams = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor='carry', box_prompt='labels',
                          box_prompt_first_only=True))['box_prompt_cams']
    assert cams.shape[1] > 1
    assert (cams[:, 0] >= 1).any() and (cams[:, 1:] == -1).all()
    cams = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor='carry', box_prompt='labels',
                          box_prompt_first_only=False))['box_prompt_cams']
    assert (cams[:, 1:] >= 1).any()


def test_give_box_this_window_compensates_for_a_stale_carry_not_just_a_missing_one():
    """The first-only gate gives the box back when `_build_prior` DECLINED a carry (stale/mismatched),
    not only when there was none."""
    from tailcyclenet.infer.window import _give_box_this_window

    cfg = _cfg(anchor='carry', box_prompt='detector', box_prompt_first_only=True)
    fake_carry = (torch.zeros(2, 2), 0)

    assert _give_box_this_window(cfg, None, None), \
        'no carry at all (the true first window) must still give the box'
    assert not _give_box_this_window(cfg, fake_carry, torch.zeros(2, 2)), \
        'a carry that WAS used as this window\'s prior must still withhold the box'
    assert _give_box_this_window(cfg, fake_carry, None), \
        'a carry that existed but was declined (stale/shape-mismatched) must give the box back'

    off = _cfg(anchor='carry', box_prompt='detector', box_prompt_first_only=False)
    assert _give_box_this_window(off, fake_carry, None), \
        'box_prompt_first_only=False must give the box every window regardless of carry/prior'
    assert not _give_box_this_window(_cfg(anchor='carry', box_prompt='none'), None, None), \
        'box_prompt="none" must never give the box'


def test_box_prompt_wide_pass1_tight_pass2_geometry(rat):
    """Pass 1 is inflated about its centre by `crop_inflate`; pass 2 (from the prediction) is not."""
    model, sess, registry, name = rat
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    cx, cy = w / 2, h / 2
    side = min(w, h) / 4
    boxes[..., 0], boxes[..., 1] = cx - side / 2, cy - side / 2
    boxes[..., 2], boxes[..., 3] = cx + side / 2, cy + side / 2

    tight = run_group(model, sess, 'g000', registry, name,
                      _cfg(anchor='none', refine=True, crop_inflate=1.0), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    wide = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor='none', refine=True, crop_inflate=1.5), boxes_for=_boxes_for(boxes), n_rows=len(boxes))

    c_tight, c_wide = tight['crop'][0, 0], wide['crop'][0, 0]
    for ci in range(C):
        if not (np.isfinite(c_tight[ci]).all() and np.isfinite(c_wide[ci]).all()):
            continue
        wt = c_tight[ci, 2] - c_tight[ci, 0]
        ww = c_wide[ci, 2] - c_wide[ci, 0]
        assert ww > wt * 1.2, 'pass 1 must be visibly wider under crop_inflate = 1.5'
        cxt = (c_tight[ci, 0] + c_tight[ci, 2]) / 2
        cxw = (c_wide[ci, 0] + c_wide[ci, 2]) / 2
        assert abs(cxt - cxw) < wt, 'inflation is about the SAME centre, not a shifted one'

    r_tight, r_wide = tight['crop_refined'][0, 0], wide['crop_refined'][0, 0]
    for ci in range(C):
        if not (np.isfinite(r_tight[ci]).all() and np.isfinite(r_wide[ci]).all()):
            continue
        rt = r_tight[ci, 2] - r_tight[ci, 0]
        rw = r_wide[ci, 2] - r_wide[ci, 0]
        assert rw < rt * 1.2, 'pass 2 must not carry pass 1\'s inflation forward'


def test_box_prompt_detector_needs_a_box_source(rat):
    """`box_prompt='detector'` runs from detector boxes and reports them; without a source it raises."""
    model, sess, registry, name = rat
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, len(sess.rig), 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', box_prompt='detector'),
                    boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    assert (out['box_prompt_cams'][0] >= 1).any()
    with pytest.raises(ValueError, match='detector'):
        run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', box_prompt='detector'))


def test_box_prompt_detector_from_a_boxes_file_uses_its_points(mv):
    """`--boxes` points on a box model give the same prompt as `labels` from those points (used to raise)."""
    model, sess, registry, name = mv
    lab = sess.labels('g000')
    pts = lab.points3d if sess.mode == '3d' else lab.points2d[..., 0, :]
    got = run_group(model, sess, 'g000', registry, name,
                    _cfg(anchor='none', box_prompt='detector'), box_points=pts)
    want = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor='none', box_prompt='labels'))
    assert (got['box_prompt_cams'] >= 1).any()
    np.testing.assert_array_equal(got['box_prompt_cams'], want['box_prompt_cams'])
    np.testing.assert_array_equal(got['pred'], want['pred'])


def test_carried_prior_is_bounds_masked_and_dated():
    """The carried prior is NaN outside the crop, dated to its frame in the window, and retired (not
    clamped) when it predates the window or is staler than the overlap."""
    from tailcyclenet.infer.window import _build_prior

    K, size = 4, torch.tensor([100, 100], dtype=torch.int32)
    cgroup = [{'size': size}]
    cfg = InferConfig(anchor='carry')
    frames = np.arange(20, 28)
    boxes = [(0, 0, 100, 100)]

    pose = torch.tensor([[10.0, 10.0], [90.0, 90.0], [-5.0, 10.0], [10.0, 400.0]])
    prior, qt = _build_prior(cfg, (pose, 23), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                             cgroup)

    assert torch.isfinite(prior[0, :2]).all(), 'in-crop keypoints must survive'
    assert torch.isnan(prior[0, 2:]).all(), 'out-of-crop keypoints must become NaN'
    assert qt.shape == (1, K)
    assert (qt == 3).all(), f'prompt frame 23 is index 3 of a window starting at 20, got {qt}'

    assert _build_prior(cfg, (pose, 17), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                        cgroup) == (None, None)
    _, ordinary = _build_prior(cfg, (pose, 20), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                               cgroup)
    assert (ordinary == 0).all()
    _, late = _build_prior(cfg, (pose, 999), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                           cgroup)
    assert (late == len(frames) - 1).all()

    assert _build_prior(cfg, (pose, 2), None, 0, 0, frames, boxes, [1.0], '2d', K, 2,
                        cgroup) == (None, None)


def test_oracle_corrupt_near_and_swap():
    """`near` picks the nearest ELIGIBLE row (no-op if none); `swap:<n>` permutes n keypoint pairs
    of the chosen row without changing which row/frame it came from."""
    from tailcyclenet.infer.window import _corrupt_prior

    K, size = 2, torch.tensor([100, 100], dtype=torch.int32)
    cgroup = [{'size': size}]
    frames = np.arange(0, 4)
    boxes = [(0, 0, 100, 100)]
    scales = [1.0]

    src = np.zeros((3, 4, K, 2), dtype=np.float32)
    src[0, 0] = [[10.0, 10.0], [20.0, 20.0]]
    src[1, 0] = [[500.0, 500.0], [600.0, 600.0]]
    src[2, 0] = [[15.0, 15.0], [25.0, 25.0]]

    cfg = InferConfig(anchor='labels', oracle_corrupt='near')
    p, qt = _corrupt_prior(cfg, src, 0, 3, frames, boxes, '2d', cgroup, scales)
    assert torch.allclose(p, torch.as_tensor(src[2, 0])), \
        'near must pick the ELIGIBLE row, never the out-of-bounds one'
    assert qt == 0

    # Nothing eligible -> a no-op, the target's own row comes back unchanged.
    src2 = src.copy()
    src2[2, 0] = [[500.0, 500.0], [600.0, 600.0]]     # make row 2 ineligible too
    p2, _ = _corrupt_prior(cfg, src2, 0, 3, frames, boxes, '2d', cgroup, scales)
    assert torch.allclose(p2, torch.as_tensor(src2[0, 0])), 'no eligible row must be a no-op'

    src = np.zeros((1, 4, 4, 2), dtype=np.float32)
    src[0, 0] = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    p, qt = _corrupt_prior(InferConfig(anchor='labels', oracle_corrupt='swap:1'), src, 0, 1,
                           frames, boxes, '2d', cgroup)
    assert qt == 0
    base = torch.as_tensor(src[0, 0])
    assert sorted(p.tolist()) == sorted(base.tolist()), 'the keypoint SET must be unchanged'
    assert int((p != base).any(-1).sum()) == 2, 'swap:1 is exactly one pair'


def test_refine_recrops_to_its_own_prediction_and_keeps_the_coverage(mv):
    """Pass 2 re-crops around pass 1's prediction, never costs coverage, keeps `crop` as the pass-1
    box, and overlaps the box it came from."""
    model, sess, registry, name = mv
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h          # the whole frame: refinement must shrink it

    plain = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    ref = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', refine=True),
                    boxes_for=_boxes_for(boxes), n_rows=len(boxes))

    assert (ref['outcome'] == 0).all(), 'refinement must not drop an animal'
    got = np.isfinite(ref['pred']).all(-1)
    assert got.sum() >= np.isfinite(plain['pred']).all(-1).sum(), 'coverage must not fall'
    np.testing.assert_array_equal(np.nan_to_num(ref['crop'], nan=-1),
                                  np.nan_to_num(plain['crop'], nan=-1))
    cw = ref['crop_refined'][0, 0, :, 2] - ref['crop_refined'][0, 0, :, 0]
    ch = ref['crop_refined'][0, 0, :, 3] - ref['crop_refined'][0, 0, :, 1]
    assert (cw <= w).all() and (ch <= h).all()
    assert ((cw < w) | (ch < h)).any(), 'a whole-frame box should have been refined smaller'
    for ci in range(C):
        b2, b1 = ref['crop_refined'][0, 0, ci], plain['crop'][0, 0, ci]
        if np.isfinite(b2).all() and np.isfinite(b1).all():
            assert min(b2[2], b1[2]) > max(b2[0], b1[0])
            assert min(b2[3], b1[3]) > max(b2[1], b1[1])


def test_a_rejected_refinement_falls_back_at_full_resolution(rat):
    """A rejected refinement must fall back to a pass 2 at full `image_size`, not re-run pass 1's
    reduced-resolution plan (the corner box forces the fallback)."""
    model, sess, registry, name = rat
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
                  _cfg(anchor='none', refine=True, refine_px=32), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    finally:
        model.forward = inner

    assert 32 in seen, 'pass 1 must have run at the reduced resolution'
    assert 64 in seen, ('every SECOND pass must run at image_size, including the ones whose '
                        'refinement was rejected -- that fallback is the silent half')


def test_refine_px_off_is_bit_identical(mv):
    """`--refine` alone == `refine_px=None` == `refine_px=image_size`, bit for bit (strict `<` gate)."""
    model, sess, registry, name = mv
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


def test_a_smaller_input_is_compensated_at_all_four_sites(rat):
    """`input_size=` reaches the library forward as the canvas (long side) extent."""
    model, sess, registry, name = rat
    seen = {}

    def spy(*a, **kw):
        seen['input_size'] = kw.get('input_size')
        return {'coords_pred': torch.zeros(1)}

    model._forward = spy
    try:
        cg = [{'size': torch.tensor([32, 32])}]
        model.forward([torch.zeros(1)], torch.zeros(1, 3, dtype=torch.long), cg, '3d')
        assert seen['input_size'] == 32, f'input_size must be the canvas extent, got {seen}'
        cg = [{'size': torch.tensor([48, 64])}]
        model.forward([torch.zeros(1)], torch.zeros(1, 3, dtype=torch.long), cg, '3d')
        assert seen['input_size'] == 64
        cg = [{'size': torch.tensor([model.image_size, model.image_size])}]
        model.forward([torch.zeros(1)], torch.zeros(1, 3, dtype=torch.long), cg, '3d')
        assert seen['input_size'] == model.image_size
    finally:
        del model._forward


def test_the_row_gate_withholds_rows_but_not_the_carried_prompt(rat16):
    """`--vis-thresh` gates what is REPORTED, not the carried prompt: -inf is a no-op, +inf empties
    the rows but every window still runs."""
    model, sess, registry, name = rat16
    base = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry'))
    lo = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', vis_thresh=-1e9))
    hi = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', vis_thresh=1e9))

    np.testing.assert_array_equal(np.isnan(lo['pred']), np.isnan(base['pred']))
    assert np.isnan(hi['pred']).all(), 'every row is below an infinite threshold'
    assert (hi['outcome'] == 0).all(), 'the gate must not abort a window'


def test_max_frames_is_a_prefix_and_the_zero_start_range(rat16):
    """`--max-frames N` is a prefix and the same thing as `--end-frame N`; 0 or > T is the group."""
    model, sess, registry, name = rat16
    a = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', max_frames=9))
    b = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', frame_stop=9))
    np.testing.assert_array_equal(np.nan_to_num(a['pred'], nan=-9e9),
                                  np.nan_to_num(b['pred'], nan=-9e9))
    assert a['pred'].shape[1] == a['n_frames'] == 9
    assert a['frame_start'] == 0 and a['frame_stop'] == 9
    for m in (0, 99):
        assert run_group(model, sess, 'g000', registry, name,
                         _cfg(anchor='none', max_frames=m))['pred'].shape[1] == 16


def test_render_writes_every_predicted_frame(rat, tmp_path):
    """The mp4 holds exactly the predicted frames."""
    import cv2

    from tailcyclenet.render import render_group

    model, sess, registry, name = rat
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none', max_frames=3))
    path = tmp_path / 'clip.mp4'
    render_group(sess, 'g000', out['pred'], path, max_side=64, fps=5)

    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 3, f'wrote {n} frames for a 3-frame prediction'


def test_2d_is_bit_identical_under_either_carry_source(rat16):
    """2D has nothing to triangulate, so `carry_source` must not move a 2D prediction."""
    model, sess, registry, name = rat16
    torch.manual_seed(0)
    a = run_group(model, sess, 'g000', registry, name,
                  _cfg(anchor='carry', carry_source='triangulate'))
    torch.manual_seed(0)
    b = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', carry_source='pred'))
    np.testing.assert_array_equal(np.nan_to_num(a['pred'], nan=-9e9),
                                  np.nan_to_num(b['pred'], nan=-9e9))


def test_3d_carry_feeds_back_the_anchor_free_estimate(mv16):
    """...while in 3D `carry_source` must change the prediction, or the de-loop is wired to nothing."""
    model, sess, registry, name = mv16
    tri = run_group(model, sess, 'g000', registry, name,
                    _cfg(anchor='carry', carry_source='triangulate'))
    pred = run_group(model, sess, 'g000', registry, name, _cfg(anchor='carry', carry_source='pred'))
    assert len(tri['window_start']) > 1
    assert not np.array_equal(np.nan_to_num(tri['pred'], nan=-9e9),
                              np.nan_to_num(pred['pred'], nan=-9e9))


def test_instances_boxes_get_the_training_crop_rule_at_inference(rat):
    """`box_source='instances'` boxes go through the loader's crop rule (pad 0); detector boxes keep
    the raw union (re-squaring is measured worse)."""
    from tailcyclenet import crop as cropmod

    model, sess, registry, name = rat
    w, h = sess.rig.size(sess.cam_names[0])
    S, T, C = len(sess.labels('g000').animal_ids), 4, len(sess.rig)

    wide = np.tile(np.array([10.0, 10.0, 10.0 + 0.6 * w, 10.0 + 0.2 * h], np.float32),
                   (S, T, C, 1))
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'), boxes_for=_boxes_for(wide), n_rows=len(wide))
    det_box = out['crop'][0, 0, 0]

    assert det_box[2] - det_box[0] != det_box[3] - det_box[1], \
        'the detector union must stay non-square, or the measured +3.06 mm result is being undone'

    stored = sess.labels('g000').boxes
    a, t = [(i, j) for i in range(stored.shape[0]) for j in range(stored.shape[1])
            if np.isfinite(stored[i, j, 0]).all()][0]
    cfg = _cfg(anchor='none', box_source='instances')
    got = run_group(model, sess, 'g000', registry, name, cfg)

    x0, y0, x1, y1 = stored[a, t, 0]
    expect = cropmod.crop_box_for_points(
        torch.tensor([[x0, y0], [x1, y1]]), torch.tensor([int(w), int(h)]),
        cfg.min_crop_dim, pad=0)
    np.testing.assert_array_equal(got['crop'][a, 0, 0], np.asarray(expect, np.float32))


def test_a_3d_render_uses_the_per_frame_camera_on_a_moving_rig(mv):
    """`render.project` uses the per-frame extrinsic on a moving rig (matching project_points_torch),
    and reads extrinsic `frames[t]` at column t for a non-identity `frames`."""
    from tailcyclenet.render import project

    model, sess, registry, name = mv
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


def test_the_tracker_projects_correctly_on_a_moving_rig(mv):
    """On a moving rig `cgroup(gid, t)` with a scalar t gives a (4,4) ext, which `track._project`
    relies on (a (T,4,4) ext would project target i through frame i's pose)."""
    from tailcyclenet.detector.track import _project

    _, sess, _, _ = mv
    gid = list(sess.groups)[0]
    per_frame = sess.cgroup(gid, 0)
    assert all(c['ext'].ndim == 2 for c in per_frame), \
        'a scalar frame must give a (4,4) ext, or the projection axis alignment below is wrong'
    assert sess.cgroup(gid)[0]['ext'].ndim == 3, 'the whole-group form must still be per-frame'
    for n in (1, 3):
        out = _project(per_frame[0], np.zeros((n, 3), np.float32))
        assert out.shape == (n, 2)


def test_the_per_camera_2d_pose_at_camera_zero_is_the_prediction(rat):
    """In 2D `pred2d[:, :, 0]` IS `pred`, bit for bit (same un-crop expression; the exact inverse
    would move every published 2D number)."""
    model, sess, registry, name = rat
    out = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))
    C, K = len(sess.rig), sess.n_keypoints
    S, T = out['pred'].shape[0], out['pred'].shape[1]
    assert out['pred2d'].shape == (S, T, C, K, 2)
    assert out['conf2d'].shape == (S, T, C, K)
    assert np.isfinite(out['pred2d']).any(), 'the per-camera pose must actually be written'
    if sess.mode == '2d':
        np.testing.assert_array_equal(np.nan_to_num(out['pred2d'][:, :, 0], nan=-9e9),
                                      np.nan_to_num(out['pred'], nan=-9e9))


def test_session_writer_preserves_named_2d_signals_and_window_metadata(rat, tmp_path):
    """The new prediction table keeps visibility and model confidence as separate signals."""
    import pyarrow.parquet as pq

    _, sess, registry, _ = rat
    from tailcyclenet.infer.predictions import SessionWriter

    gid = next(iter(sess.groups))
    K = sess.n_keypoints
    C = len(sess.rig)
    pred2d = np.zeros((1, 2, C, K, 2), np.float32)
    visibility_logit = (np.arange(2 * C * K, dtype=np.float32).reshape(1, 2, C, K) / 10) - 1
    model_logit = (np.arange(2 * C * K, dtype=np.float32).reshape(1, 2, C, K) / 7) - 2
    blk = {
        'animal_ids': np.array(['a00'], object),
        'pred': np.zeros((1, 2, K, 2), np.float32),
        'conf': np.zeros((1, 2, K), np.float32),
        'pred2d': pred2d,
        'conf2d': visibility_logit,
        'model_conf2d': model_logit,
        'box_agree': np.full((1, 2, C), np.nan, np.float32),
        'outcome': np.zeros((1, 1), np.int8),
        'crop': np.zeros((1, 1, C, 4), np.float32),
        'outcome_names': ['ok'],
        'window_start': np.array([0], np.int32),
        'window_stop': np.array([2], np.int32),
        'owner_window': np.array([0, 0], np.int32),
    }
    out = tmp_path / 'pred'
    writer = SessionWriter(out, sess, registry, {'n_frames': 2}, [gid])
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)

    table = pq.read_table(out / 'points2d.pq')
    assert {'visibility_logit', 'visibility_probability', 'confidence_logit',
            'confidence_probability', 'slot', 'window', 'window_start', 'window_stop'} <= set(
                table.column_names)
    visibility = table['visibility_logit'].to_numpy()
    confidence = table['confidence_logit'].to_numpy()
    assert np.allclose(table['visibility_probability'].to_numpy(),
                       1 / (1 + np.exp(-visibility)))
    assert np.allclose(table['confidence_probability'].to_numpy(),
                       1 / (1 + np.exp(-confidence)))
    assert not np.allclose(visibility, confidence)

    no_model = dict(blk)
    no_model.pop('model_conf2d')
    out_no_model = tmp_path / 'pred-no-model-confidence'
    writer = SessionWriter(out_no_model, sess, registry, {'n_frames': 2}, [gid])
    writer.write_block(gid, no_model, 0, 0)
    writer.close(complete=True)
    missing = pq.read_table(out_no_model / 'points2d.pq')
    assert np.isnan(missing['confidence_probability'].to_numpy()).all()
    assert np.isnan(missing['confidence_logit'].to_numpy()).all()


def test_merge_blocks_stitches_model_2d_confidence():
    from tailcyclenet.infer.window import merge_blocks

    first = {'model_conf2d': np.array([[[[0.1]]]], np.float32)}
    second = {'model_conf2d': np.array([[[[0.9]]]], np.float32)}
    merged = merge_blocks([first, second])
    np.testing.assert_array_equal(merged['model_conf2d'],
                                  np.array([[[[0.1]], [[0.9]]]], np.float32))


def test_prediction_3d_missing_status_retains_best_guess_coordinates(mv, tmp_path):
    """A 3D `missing` prediction row keeps its best-guess coordinates and validates."""
    import pyarrow.parquet as pq

    _, sess, registry, _ = mv
    from tailcyclenet.infer.predictions import SessionWriter, load_predictions

    gid = next(iter(sess.groups))
    K, C, T = sess.n_keypoints, len(sess.rig), 2
    pattern = np.resize(np.array([-1.0, 0.0, 1.0], np.float32), K)
    logits = np.tile(pattern[None, None], (1, T, 1))
    pred = np.arange(T * K * 3, dtype=np.float32).reshape(1, T, K, 3) + 1
    blk = {
        'animal_ids': np.array(['a00'], object), 'pred': pred, 'conf': logits,
        'pred2d': np.full((1, T, C, K, 2), np.nan, np.float32),
        'conf2d': np.full((1, T, C, K), np.nan, np.float32),
        'box_agree': np.full((1, T, C), np.nan, np.float32),
        'outcome': np.zeros((1, 1), np.int8), 'crop': np.zeros((1, 1, C, 4), np.float32),
        'outcome_names': ['ok'], 'window_start': np.array([0], np.int32),
        'window_stop': np.array([T], np.int32), 'owner_window': np.zeros(T, np.int32),
    }
    out = tmp_path / 'pred3d'
    writer = SessionWriter(out, sess, registry,
                           {'n_frames': T, 'source_session_id': sess.session_id}, [gid])
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)

    table = pq.read_table(out / 'points3d.pq')
    rows = table.to_pylist()
    assert len(rows) == T * K
    for row in rows:
        assert row['status'] == ('visible' if row['score'] > 0.5 else 'missing')
        assert np.isfinite([row['x'], row['y'], row['z']]).all()
    zero_logit = next(row for row in rows if row['score_logit'] == 0.0)
    assert zero_logit['status'] == 'missing' and zero_logit['x'] is not None
    preds, _ = load_predictions(out)
    assert np.isfinite(preds[f'{sess.session_id}/{gid}']['pred'][:, :T]).all()
    from tailcyclenet.format import Session, validate_session
    errs = [e for e in validate_session(Session.load(out)) if '[rule 7]' not in e]
    assert not errs, f'prediction-only 3D rows should validate apart from missing pixels: {errs}'


def test_quality_columns_are_optional_and_align_with_pose_rows(scene, tmp_path):
    import pyarrow.parquet as pq
    from tailcyclenet.infer.predictions import SessionWriter

    _, sess, registry, _ = scene
    gid = next(iter(sess.groups))
    T, K, C = 2, sess.n_keypoints, len(sess.rig)
    R = 3 if sess.mode == '3d' else 2
    pred = np.arange(T * K * R, dtype=np.float32).reshape(1, T, K, R) + 1
    pred2d = (pred[:, :, None] if R == 2 else
              np.full((1, T, C, K, 2), np.nan, np.float32))
    quality = np.arange(T * K, dtype=np.float32).reshape(1, T, K) / 10
    precision = quality + 1
    blk = {
        'animal_ids': np.array(['a00'], object), 'pred': pred,
        'conf': np.ones((1, T, K), np.float32), 'pred2d': pred2d,
        'conf2d': np.ones((1, T, C, K), np.float32),
        'model_conf2d': np.ones((1, T, C, K), np.float32),
        'quality': quality, 'quality_precision': precision,
        'box_agree': np.full((1, T, C), np.nan, np.float32),
        'outcome': np.zeros((1, 1), np.int8), 'crop': np.zeros((1, 1, C, 4), np.float32),
        'outcome_names': ['ok'], 'window_start': np.array([0], np.int32),
        'window_stop': np.array([T], np.int32), 'owner_window': np.zeros(T, np.int32),
    }
    out = tmp_path / f'quality-{sess.mode}'
    writer = SessionWriter(out, sess, registry, {'n_frames': T}, [gid], quality_enabled=True)
    writer.write_block(gid, blk, 0, 0)
    writer.close(complete=True)
    table_name = 'points3d.pq' if R == 3 else 'points2d.pq'
    rows = pq.read_table(out / table_name).to_pylist()
    assert len(rows) == T * K
    got = {(row['frame'], row['bodypart']):
           (row['quality_score'], row['quality_precision']) for row in rows}
    for frame in range(T):
        for k, bodypart in enumerate(registry.names):
            assert got[(frame, bodypart)] == pytest.approx((quality[0, frame, k],
                                                             precision[0, frame, k]))


def test_window_prediction_sidecar_streams_typed_nullable_records(rat, tmp_path):
    import pyarrow.parquet as pq

    _, sess, registry, _ = rat
    from tailcyclenet.infer.predictions import SessionWriter

    gid = next(iter(sess.groups))
    out = tmp_path / 'pred-with-sidecar'
    writer = SessionWriter(out, sess, registry,
                           {'n_frames': 4, 'window_predictions': True}, [gid],
                           quality_enabled=True)
    writer.write_window_records([
        {'record_type': 'window', 'group_id': gid, 'frame': None, 'window': 1,
         'window_start': 2, 'window_stop': 4, 'slot': 0, 'animal_id': 'a00',
         'camera': sess.cam_names[0], 'bodypart': None, 'outcome': 'ok',
         'crop_x0': 1.0, 'crop_y0': 2.0, 'crop_x1': 3.0, 'crop_y1': 4.0,
         'crop_refined_x0': 5.0},
        {'record_type': 'point3d', 'group_id': gid, 'frame': 2, 'window': 1,
         'window_start': 2, 'window_stop': 4, 'slot': 0, 'animal_id': 'a00',
         'camera': None, 'bodypart': sess.names[0], 'x': 1.5, 'y': 2.5, 'z': 3.5,
         'status': 'missing', 'score': 0.4, 'score_logit': -0.4,
         'quality_score': 0.7, 'quality_precision': 0.8, 'gated': True},
    ])
    writer.close(complete=True)

    table = pq.read_table(out / 'window_predictions.pq')
    rows = table.to_pylist()
    assert len(rows) == 2
    assert rows[0]['frame'] is None and rows[0]['bodypart'] is None
    assert rows[0]['refined_x0'] == 5.0
    assert rows[1]['status'] == 'missing' and rows[1]['gated'] is True
    assert (rows[1]['x'], rows[1]['y'], rows[1]['z']) == (1.5, 2.5, 3.5)
    assert (rows[1]['quality_score'], rows[1]['quality_precision']) == pytest.approx((0.7, 0.8))
    assert rows[0]['quality_score'] is None
    assert table.schema.field('frame').nullable
    assert table.schema.field('bodypart').nullable


def test_uncalibrated_singleview_prediction_session_uses_normalized_units(tmp_path):
    """The finite single-view fallback must not inherit a 2D source's pixel units."""
    import tomllib
    import conftest as cf
    from tailcyclenet.format import Session
    from tailcyclenet.infer.predictions import SessionWriter

    source_path = tmp_path / 'r' / 'train' / 's'
    cf._session_2d(source_path, T=4)
    source = Session.load(source_path)
    source.mode = '3d'
    registry = Registry.build([load_dataset(source_path.parent.parent)])
    out = tmp_path / 'prediction'
    writer = SessionWriter(out, source, registry, {}, list(source.groups))
    writer.close(complete=True)
    cfg = tomllib.loads((out / 'session.toml').read_text())
    assert cfg['mode'] == '3d'
    assert cfg['units'] == 'normalized'


def test_a_duplicate_provenance_key_raises_rather_than_silently_winning(tmp_path):
    """Two provenance values under one name raise (the run's `box_source` was once silently
    overwritten by the detector's); the same value twice is fine."""
    import conftest as cf
    from tailcyclenet.format import Registry, Session
    from tailcyclenet.infer.predictions import SessionWriter

    cf._session_2d(tmp_path / 'r' / 'test' / 's')
    sess = Session.load(tmp_path / 'r' / 'test' / 's')
    sess.preload()
    reg = Registry.build([load_dataset(tmp_path / 'r')])

    with pytest.raises(ValueError, match='given twice'):
        SessionWriter(tmp_path / 'out', sess, reg,
                      [('box_source', 'keypoints'), ('box_source', 'instances')],
                      list(sess.groups))
    # ...and the same value twice is not a conflict.
    SessionWriter(tmp_path / 'ok', sess, reg,
                  [('box_source', 'keypoints'), ('box_source', 'keypoints')],
                  list(sess.groups)).close()


def test_the_cli_runs_end_to_end_with_no_detector(cli, monkeypatch, tmp_path):
    """Through `main()` on the default box path: the output is a valid prediction session (no
    keypoints.pq, sidecar opt-in) and `[provenance]` names the run, checkpoint FILE and source by
    absolute path even from a relative `--run`."""
    import tomllib

    from tailcyclenet.format import Session, validate_session
    from tailcyclenet.infer.predictions import load_predictions

    sdir, run = _cli_setup(tmp_path)
    out = tmp_path / 'pred'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', 'run', '--data', str(sdir),
                                      '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--out', str(out)])
    cli.main()

    for f in ('session.toml', 'calibration.toml', 'groups.pq', 'points2d.pq', 'windows.pq'):
        assert (out / f).exists(), f'{f} missing; wrote {sorted(p.name for p in out.iterdir())}'
    assert not (out / 'keypoints.pq').exists(), 'model estimates are not annotation observations'
    assert not (out / 'window_predictions.pq').exists(), 'sidecar export is opt-in'
    got = Session.load(out)
    assert got.label_source == 'tracked'
    errs = [e for e in validate_session(got) if '[rule 7]' not in e]
    assert not errs, f'a prediction session must be valid apart from its missing pixels: {errs}'

    preds, meta = load_predictions(out)
    assert preds['s/g000']['pred'].shape[1] == 4
    assert np.isfinite(preds['s/g000']['pred']).any(), 'the round trip lost every point'
    assert meta['anchor'] == 'none'

    with open(out / 'session.toml', 'rb') as f:
        prov = tomllib.load(f)['provenance']
    for k in ('run', 'checkpoint', 'source_session'):
        assert Path(prov[k]).is_absolute() and Path(prov[k]).exists(), (k, prov[k])
    assert Path(prov['source_session']).name == 's'
    assert Path(prov['checkpoint']).is_file(), 'checkpoint must be the FILE, not the run folder'
    assert prov['checkpoint_name'] == Path(prov['checkpoint']).name
    assert prov['box_source'] == 'keypoints'
    assert prov['det_box_source'] == '' and prov['detector'] == ''
    assert prov['precision'] == 'bf16' and prov['camera_batch'] is True


def test_the_cli_runs_from_a_lone_pose_checkpoint(cli, monkeypatch, tmp_path):
    """`--run` may be ONE checkpoint file with no run folder; the prediction is the folder's."""
    import shutil

    import conftest as cf
    from tailcyclenet.infer.predictions import load_predictions

    root = tmp_path / 'rat'
    cf._session_2d(root / 'test' / 's')
    run, path = _save_run(tmp_path, Registry.build([load_dataset(root)]))
    lone = tmp_path / 'elsewhere' / 'pose.pth'
    lone.parent.mkdir()
    shutil.copy(path, lone)

    def infer(where, out):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--pose', str(where),
                                          '--data', str(root / 'test' / 's'), '--anchor', 'none',
                                          '--device', 'cpu', '--overlap', '2', '--out', str(out)])
        cli.main()
        return load_predictions(out)

    ref, _ = infer(run, tmp_path / 'from-run')
    shutil.rmtree(run)
    got, meta = infer(lone, tmp_path / 'from-file')
    assert meta['run'] == str(lone.resolve())
    np.testing.assert_array_equal(got['s/g000']['pred'], ref['s/g000']['pred'])


@pytest.mark.parametrize('flag,selector_flag,value', [
    ('--pose', '--pose-checkpoint', 'checkpoint_best.pth'),
    ('--run', '--checkpoint', 'checkpoint_best.pth'),
    ('--scorer', '--scorer-checkpoint', 'checkpoint_best.pth'),
    ('--detector', '--detector-checkpoint', 'best'),
])
def test_a_checkpoint_selector_beside_a_file_is_refused_before_loading(
        cli, monkeypatch, tmp_path, flag, selector_flag, value):
    import conftest as cf

    cf._session_2d(tmp_path / 'rat' / 'test' / 's')
    f = tmp_path / 'model.pth'
    torch.save({}, f)

    _no_load(monkeypatch)
    argv = {'--run': str(tmp_path / 'run'), flag: str(f), selector_flag: value}
    monkeypatch.setattr(sys, 'argv', ['infer.py', *[x for kv in argv.items() for x in kv],
                                      '--data', str(tmp_path / 'rat' / 'test' / 's'),
                                      '--device', 'cpu', '--out', str(tmp_path / 'o')])
    shown = {'--run': '--pose', '--checkpoint': '--pose-checkpoint'}
    with pytest.raises(SystemExit,
                       match=f'already a checkpoint file; {shown.get(selector_flag, selector_flag)}'):
        cli.main()


def test_pose_and_run_are_one_option():
    """`--pose`/`--pose-checkpoint` are the names; `--run`/`--checkpoint` stay as aliases."""
    from tailcyclenet.infer.cli import build_parser
    ap = build_parser()
    base = ['--data', 'd', '--out', 'o']
    new = ap.parse_args(['--pose', 'r', '--pose-checkpoint', 'c.pth', *base])
    old = ap.parse_args(['--run', 'r', '--checkpoint', 'c.pth', *base])
    assert (new.run, new.checkpoint) == (old.run, old.checkpoint) == (Path('r'), 'c.pth')


def test_session_shape_refusals_fire_before_the_checkpoint_loads(cli, monkeypatch, tmp_path):
    """`--data` must name ONE session; `--out` must be empty and must not be the source session."""
    import conftest as cf

    cf._session_2d(tmp_path / 'two' / 'test' / 'a')
    cf._session_2d(tmp_path / 'two' / 'test' / 'b')
    sdir = tmp_path / 'two' / 'test' / 'a'
    stale = tmp_path / 'stale'
    stale.mkdir()
    (stale / 'stale.pq').write_bytes(b'x')
    _no_load(monkeypatch)
    for data, out, expect in [(tmp_path / 'two', tmp_path / 'o', 'ONE session directory'),
                              (sdir, stale, 'not empty'),
                              (sdir, sdir, 'SOURCE session')]:
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(tmp_path / 'nope'),
                                          '--data', str(data), '--split', 'test',
                                          '--device', 'cpu', '--out', str(out)])
        with pytest.raises(SystemExit, match=expect):
            cli.main()


def test_session_writer_finalizes_incomplete_output_when_preload_fails(cli, monkeypatch, tmp_path):
    """A failure mid-run still finalizes the output: `complete = false`, readable empty tables."""
    import tomllib

    import pyarrow.parquet as pq
    from tailcyclenet.format import Session

    sdir, run = _cli_setup(tmp_path)
    out = tmp_path / 'partial-prediction'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                      '--data', str(sdir), '--anchor', 'none',
                                      '--device', 'cpu', '--window-predictions', '--out', str(out)])

    def fail_preload(_self):
        raise RuntimeError('injected preload failure')

    monkeypatch.setattr(Session, 'preload', fail_preload)
    with pytest.raises(RuntimeError, match='injected preload failure'):
        cli.main()

    with (out / 'session.toml').open('rb') as f:
        assert tomllib.load(f)['complete'] is False
    assert pq.read_table(out / 'points2d.pq').num_rows == 0
    assert pq.read_table(out / 'window_predictions.pq').num_rows == 0


def _detector_ckpt(tmp_path, dataset, min_crop_dim=16, box_source='keypoints'):
    """A minimal but genuinely loadable detector checkpoint file."""
    import conftest as cf
    from tailcyclenet.detector.yolox import YOLOXNano

    ckpt = cf._detector_checkpoint(YOLOXNano(version='hybrid'), input_wh=(64, 48), dataset=dataset,
                                   min_crop_dim=min_crop_dim, box_source=box_source)
    p = tmp_path / 'det.pth'
    torch.save(ckpt, p)
    return p


class _ReachedRunBlocks(Exception):
    """Raised by a `run_blocks` stub: the driver got past every refusal."""


def test_cli_detector_family_refusal_and_its_escape_hatch(cli, monkeypatch, tmp_path):
    """A detector from another dataset family is refused before `run_blocks`; a same-family variant
    (rat-city-tracked vs a rat-city-combined detector) and `--allow-detector-transfer` pass."""
    import conftest as cf

    for root in ('other-species-tracked', 'rat-city-tracked'):
        cf._session_2d(tmp_path / root / 'test' / 's')
    registry = Registry.build([load_dataset(tmp_path / r)
                               for r in ('other-species-tracked', 'rat-city-tracked')])
    run, _ = _save_run(tmp_path, registry)
    det = _detector_ckpt(tmp_path, 'rat-city-combined')

    def boom(*a, **k):
        raise _ReachedRunBlocks()

    monkeypatch.setattr('tailcyclenet.infer.driver.run_blocks', boom)

    def main(root, *extra):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                          '--data', str(tmp_path / root / 'test' / 's'),
                                          '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                          '--overlap', '2', '--detector', str(det),
                                          '--out', str(tmp_path / f'pred-{root}-{len(extra)}'),
                                          *extra])
        cli.main()

    with pytest.raises(SystemExit, match='different dataset families'):
        main('other-species-tracked')
    with pytest.raises(_ReachedRunBlocks):
        main('rat-city-tracked')
    with pytest.raises(_ReachedRunBlocks):
        main('other-species-tracked', '--allow-detector-transfer')


@pytest.mark.parametrize('argv,expect', [
    (['--anchor', 'labels', '--detector', 'nope'], 'not label rows'),
    (['--anchor', 'labels', '--boxes', 'nope.npz'], 'not label rows'),
    (['--oracle-corrupt', 'nonsense'], 'kind must be one of'),
    (['--oracle-corrupt', 'off'], 'needs an amount'),
    (['--oracle-corrupt', 'off:0.5', '--anchor', 'carry'], 'only means anything'),
    (['--max-frames', '120', '--start-frame', '300'], 'two readings of equal force'),
    (['--start-frame', '-1'], 'negative frame indices'),
    (['--start-frame', '300', '--end-frame', '20'], 'not past --start-frame'),
])
def test_the_cli_refuses_incoherent_combinations_before_loading_anything(cli, monkeypatch,
                                                                        argv, expect):
    """Combinations that used to run and produce a number (worst: `--anchor labels` with a box
    source seeded detector row a from LABEL row a), plus the frame-range arithmetic refusals."""
    _no_load(monkeypatch)
    repo = Path(__file__).resolve().parent.parent
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(repo / 'no_such_run'),
                                      '--data', str(repo), '--out', '/tmp/unused'] + argv)
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert expect in str(e.value), f'wanted {expect!r}, got: {e.value}'


def test_deleted_inference_flags_raise_by_name(cli, monkeypatch):
    """Deleted flags (architecture cleanup 2026-10-19) raise naming themselves, not argparse noise."""
    repo = Path(__file__).resolve().parent.parent
    for argv in (['--independent-windows'], ['--fill-from', 'prior'],
                 ['--det-input-wh', '416', '416']):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(repo / 'no_such_run'),
                                          '--data', str(repo), '--out', '/tmp/unused'] + argv)
        with pytest.raises(SystemExit, match=f'{argv[0]} is deleted'):
            cli.main()


def test_assoc_res_max_px_overrides_a_data_session_but_defaults_to_its_own(cli, monkeypatch,
                                                                           tmp_path):
    """An omitted `--assoc-res-max-px` keeps the session's own value (17.5, not the format's 30);
    given explicitly it overrides."""
    import tomllib

    import toml

    session_dir, run = _cli_setup(tmp_path)
    toml_path = session_dir / 'session.toml'
    cfg = toml.loads(toml_path.read_text())
    cfg['assoc_res_max_px'] = 17.5
    toml_path.write_text(toml.dumps(cfg))

    def _run(out, *extra):
        monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                          '--data', str(session_dir), '--split', 'test',
                                          '--anchor', 'none', '--device', 'cpu',
                                          '--overlap', '2', '--out', str(out), *extra])
        cli.main()
        with open(out / 'session.toml', 'rb') as f:
            return tomllib.load(f)['assoc_res_max_px']

    assert _run(tmp_path / 'default') == 17.5, \
        "an omitted flag must not clobber the session's own value"
    assert _run(tmp_path / 'override', '--assoc-res-max-px', '12') == 12.0, \
        'given explicitly, it must override'


def test_a_range_is_byte_identical_where_the_window_partition_agrees(rat16):
    """Under `--anchor none` a range equals the whole-clip slice where the window partition agrees:
    outright to the clip end; mid-clip, up to the range's own (pulled-back) last window start."""
    model, sess, registry, name = rat16
    whole = run_group(model, sess, 'g000', registry, name, _cfg(anchor='none'))

    a = 6
    tail = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor='none', frame_start=a))
    assert list(tail['window_start']) == [w for w in whole['window_start'] if w >= a]
    np.testing.assert_array_equal(np.nan_to_num(tail['pred'], nan=-9e9),
                                  np.nan_to_num(whole['pred'][:, a:], nan=-9e9))

    b = 14
    part = run_group(model, sess, 'g000', registry, name,
                     _cfg(anchor='none', frame_start=a, frame_stop=b))
    assert part['pred'].shape[1] == b - a
    assert part['n_frames'] == b - a and part['frame_start'] == a and part['frame_stop'] == b
    assert int(part['window_start'][0]) == a
    assert int(part['window_start'][-1]) == b - 4, 'the last window must end exactly at --end-frame'
    last = int(part['window_start'][-1])
    np.testing.assert_array_equal(
        np.nan_to_num(part['pred'][:, :last - a], nan=-9e9),
        np.nan_to_num(whole['pred'][:, a:last], nan=-9e9))


def test_a_range_writes_source_frame_indices(cli, monkeypatch, tmp_path):
    """`frame` is always the SOURCE index and groups.pq keeps the full length; `load_predictions`
    densifies NaN outside the range (rebasing to 0 was the `chunk_frames` bug)."""
    from tailcyclenet.infer.predictions import load_predictions

    sdir, run = _cli_setup(tmp_path, T=16)

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                      '--data', str(sdir), '--anchor', 'none',
                                      '--device', 'cpu', '--overlap', '2',
                                      '--start-frame', '6', '--end-frame', '14',
                                      '--window-predictions', '--out', str(out)])
    cli.main()

    import pyarrow.parquet as pq
    gt = pq.read_table(out / 'groups.pq').to_pydict()
    assert int(gt['n_frames'][0]) == 16, 'groups.pq must keep the FULL group length'
    kp = pq.read_table(out / 'points2d.pq')
    fr = np.asarray(kp.column('frame').to_pylist())
    assert fr.min() >= 6 and fr.max() < 14, f'frames outside the range: {fr.min()}..{fr.max()}'
    sidecar = pq.read_table(out / 'window_predictions.pq')
    assert {'point2d', 'window'} <= set(sidecar['record_type'].to_pylist())
    assert all(r is None or 6 <= r < 14 for r in sidecar['frame'].to_pylist())

    preds, _ = load_predictions(out)
    p = preds['s/g000']['pred']
    assert p.shape[1] == 16, 'a ranged prediction must densify to the FULL group length'
    assert np.isfinite(p[:, 6:14]).any()
    assert not np.isfinite(p[:, :6]).any() and not np.isfinite(p[:, 14:]).any()


def _detect_session(tmp_path, T):
    """A 2D session of T frames, preloaded (no model: the detector tests stub detection)."""
    import conftest as cf
    from tailcyclenet.format import Session

    cf._session_2d(tmp_path / 'rat' / 'test' / 's', T=T)
    sess = Session.load(tmp_path / 'rat' / 'test' / 's')
    sess.preload()
    return sess


@pytest.mark.parametrize('start,stop', [(37, 53), (37, 64)])
def test_the_detection_slice_is_batch_aligned_and_reaches_the_range_end(tmp_path, start, stop):
    """The detection cursor opens on a `_DET_BATCH` boundary BELOW `--start-frame` (never rounding
    the user's value), and `detect_raw` is told the RANGE end, so a ragged final slice (53) is
    accepted as well as an exact one (64). The driver once passed `max_frames=0` and failed 272 s
    into a 1000-frame run."""
    import tailcyclenet.detector as detmod
    from tailcyclenet.infer import driver
    from tailcyclenet.infer.cli import build_parser

    sess = _detect_session(tmp_path, T=80)
    seen = []

    def fake_detect_raw(det, wh, session, gid, top_k, **kw):
        fr = np.asarray(kw['frames'])
        T_clip = min(session.groups[gid].n_frames, kw['max_frames'] or session.groups[gid].n_frames)
        batch = driver._DET_BATCH
        assert int(fr[0]) % batch == 0 and (len(fr) % batch == 0 or int(fr[-1]) == T_clip - 1), (
            f'slice {fr[0]}..{fr[-1]} of T_clip={T_clip} would be refused by detect_raw')
        seen.append((int(fr[0]), int(fr[-1])))
        C, D = len(session.rig), max(1, top_k)
        return (np.full((D, len(fr), C, 4), np.nan, np.float32),
                np.full((D, len(fr), C), np.nan, np.float32))

    class _Store:
        def read(self, ci, cam, fr, pool=None, reduce=1):
            return [None] * len(fr)

    old = (detmod.detect_raw, detmod.associate_group)
    detmod.detect_raw = fake_detect_raw
    detmod.associate_group = lambda raw, s, g, n, **kw: (raw[0][:n], raw[1][:n])
    try:
        args = build_parser().parse_args(['--run', 'r', '--out', 'o', '--det-score', '0.5',
                                          '--max-move', '1.0'])
        args.frame_start, args.frame_stop = start, stop
        boxes_for = driver._detector_boxes(None, (64, 64), sess, 'g000', args, 'cpu', False,
                                           None, 2, 2)
        boxes_for(_Store(), start, stop)
    finally:
        detmod.detect_raw, detmod.associate_group = old
    assert seen[0][0] == 32, f'the cursor must open on a multiple of 16, got {seen[0][0]}'
    assert seen[-1][1] == stop - 1, f'detection must reach the range end: {seen[-1]}'


def test_a_range_past_every_group_is_refused(cli, monkeypatch, tmp_path):
    """A run whose range starts past every group is refused rather than writing an empty session."""
    sdir, run = _cli_setup(tmp_path)

    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run),
                                      '--data', str(sdir), '--anchor', 'none',
                                      '--device', 'cpu', '--overlap', '2',
                                      '--start-frame', '99', '--out', str(tmp_path / 'p')])
    with pytest.raises(SystemExit, match='past the end of every group'):
        cli.main()


# --videos: raw footage instead of a session directory.

def _videos_fixture(tmp_path, T=8, wh=(64, 48)):
    """Two calibrated cameras, videos named `cam<i>_t.mp4`, plus a hand-authored session
    directory over THE SAME FILES. Everything a `--videos` run and its `--data` twin need."""
    import conftest as cf
    from tailcyclenet import format as fmt

    W, H = wh
    rig = cf._rig([(str(i), W, H, True, False, i + 1) for i in range(2)])
    fmt.dump_calibration(tmp_path / 'calib.toml', rig)
    vids = {str(i): cf._write_video(tmp_path / 'rec' / f'cam{i}_t.mp4', i, T, wh)
            for i in range(2)}

    # The `--data` twin: the same videos, symlinked into a session directory.
    sdir = tmp_path / 'ds' / 'test' / 'rec'
    groups = {'t': fmt.Group('t', T, fps=20.0)}
    fmt.write_session(sdir, mode='3d', units='mm', label_source='tracked', names=cf.KPTS_3D,
                      rig=rig, groups=groups,
                      labels={'t': fmt.empty_labels(0, T, len(cf.KPTS_3D), 2, mode3d=True)})
    (sdir / 'groups' / 't').mkdir(parents=True)
    for cam, src in vids.items():
        fmt.link(sdir / 'groups' / 't' / f'{cam}.mp4', src)
    return rig, vids, sdir


def _videos_run(tmp_path, names):
    """A run folder whose registry holds exactly ONE dataset, so the default has to infer it."""
    registry = Registry(names=tuple(names), datasets=(('ds', tuple(range(len(names)))),))
    return _save_run(tmp_path, registry)[0], registry


def test_identity_events_reach_the_written_session(cli, monkeypatch, tmp_path):
    """A tracked 3D run with a real detector writes `identity_events.pq` through `cli.main()` (the
    wiring is what breaks) with at least one `born` and JSON `detail`."""
    import json

    import pyarrow.parquet as pq

    sdir, run = _cli_setup(tmp_path, 'mv', name='ds', T=8)
    det = _detector_ckpt(tmp_path, 'ds')

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--data', str(sdir), '--out', str(out),
        '--run', str(run), '--detector', str(det), '--device', 'cpu',
        '--split', 'test', '--overlap', '2', '--max-animals', '2', '--anchor', 'none'])
    cli.main()

    path = out / 'identity_events.pq'
    assert path.exists(), 'a tracked multiview run must write identity_events.pq'
    t = pq.read_table(path).to_pydict()
    assert t['event'], 'the table exists but is empty: the events never reached the writer'
    assert set(t) == {'group_id', 'frame', 'slot', 'event', 'detail'}, sorted(t)
    assert 'born' in set(t['event']), f"no birth recorded, events were {set(t['event'])}"
    assert all(f >= 0 for f in t['frame']) and all(s >= 0 for s in t['slot'])
    for d in t['detail']:
        json.loads(d)


def test_identity_events_are_absent_not_fatal_on_a_2d_run(cli, monkeypatch, tmp_path):
    """A 2D run has no tracker (`state['tracker'] = None`), so it writes no identity table and must
    not crash (the wiring once died with `'NoneType' object has no attribute 'events'`)."""
    sdir, run = _cli_setup(tmp_path, name='ds2d', T=8)

    out = tmp_path / 'pred2d'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--data', str(sdir), '--out', str(out),
        '--run', str(run), '--detector', str(_detector_ckpt(tmp_path, 'ds2d')),
        '--device', 'cpu', '--split', 'test', '--overlap', '2', '--max-animals', '2',
        '--anchor', 'none'])
    cli.main()

    assert (out / 'points2d.pq').exists(), 'the 2D run itself must still produce a prediction'
    assert not (out / 'identity_events.pq').exists(), \
        'a run with no tracker has no identity events; an empty table would imply it had some'


def test_overlap_trace_flag_actually_reaches_the_capture_hook(cli, monkeypatch, tmp_path):
    """`--overlap-trace` must ARM the hook, not merely write a file: an all-zero census is the silent
    failure that once cost a GPU run."""
    import json

    import conftest as cf

    rig, vids, sdir = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, cf.KPTS_3D)
    pts = np.random.default_rng(0).uniform(-20, 20,
                                           size=(2, 8, len(cf.KPTS_3D), 3)).astype(np.float32)
    np.savez(tmp_path / 'boxes.npz', **{'rec/t': pts})

    trace = tmp_path / 'ovl' / 'trace.npz'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--data', str(sdir), '--out', str(tmp_path / 'o'),
        '--run', str(run), '--boxes', str(tmp_path / 'boxes.npz'), '--anchor', 'carry',
        '--device', 'cpu', '--overlap', '2', '--dataset-name', 'ds',
        '--overlap-trace', str(trace)])
    cli.main()

    assert trace.exists(), '--overlap-trace must write its file'
    z = np.load(trace, allow_pickle=True)
    census = json.loads(str(z['census']))
    assert census, 'the census must name the group(s) that ran'
    seen = sum(c.get('pairs_seen', 0) for c in census.values())
    assert seen > 0, (
        f'the hook never fired: census {census}. A multi-window run at --overlap 2 has seams, so '
        'zero pairs means the flag did not reach `stats`, not that there was nothing to see.')
    assert len(z['dist']) == seen, 'the rows and the census must describe the same population'
    assert z['prev'].shape == z['new'].shape and z['prev'].shape[0] == seen


def test_the_videos_path_is_byte_identical_to_the_session_path(cli, monkeypatch, tmp_path):
    """`--videos` is an input path, not a second pipeline: same points3d.pq as the `--data` twin."""
    import conftest as cf

    rig, vids, sdir = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, cf.KPTS_3D)

    pts = np.random.default_rng(0).uniform(-20, 20,
                                           size=(2, 8, len(cf.KPTS_3D), 3)).astype(np.float32)
    np.savez(tmp_path / 'boxes.npz', **{'rec/t': pts})

    common = ['--run', str(run), '--boxes', str(tmp_path / 'boxes.npz'), '--anchor', 'none',
              '--device', 'cpu', '--overlap', '2', '--dataset-name', 'ds', '--units', 'mm']
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--data', str(sdir),
                                      '--out', str(tmp_path / 'a')] + common)
    cli.main()
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--videos', str(tmp_path / 'rec'),
                                      '--calibration', str(tmp_path / 'calib.toml'),
                                      '--cam-regex', 'cam([0-9]+)_', '--group-id', 't',
                                      '--session-id', 'rec',
                                      '--out', str(tmp_path / 'b')] + common)
    cli.main()

    import pyarrow.parquet as pq
    cols = ['frame', 'animal_id', 'bodypart', 'x', 'y', 'z']

    def rows(d):
        t = pq.read_table(tmp_path / d / 'points3d.pq').select(cols).to_pydict()
        return sorted(zip(*(t[c] for c in cols)))

    a, b = rows('a'), rows('b')
    assert a and a == b, '--videos must be an input path, not a second pipeline'


def test_2d_videos_without_calibration_defaults_to_px_and_nominal_cam(cli, monkeypatch, tmp_path):
    """The raw single-view path derives mode from one video, defaults units to pixels, and
    records the converter-compatible nominal cam0 using the video's dimensions."""
    import tomllib

    import conftest as cf
    from tailcyclenet.format import Session

    wh, T = (80, 48), 8
    video = cf._write_video(tmp_path / 'rec' / 'take.mp4', 0, T, wh)
    names = cf.KPTS_3D
    run, _ = _videos_run(tmp_path, names)
    points = np.zeros((1, T, len(names), 2), np.float32)
    points[..., 0] = wh[0] / 2
    points[..., 1] = wh[1] / 2
    np.savez(tmp_path / 'boxes.npz', **{'rec/take': points})

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--run', str(run), '--videos', str(video), '--boxes', str(tmp_path / 'boxes.npz'),
        '--anchor', 'none', '--device', 'cpu', '--overlap', '2', '--dataset-name', 'ds',
        '--out', str(out)])
    cli.main()

    got = Session.load(out)
    assert got.mode == '2d' and got.units == 'px'
    assert got.cam_names == ['cam0'] and got.rig.size('cam0') == wh
    from tailcyclenet import adopt
    back = adopt.session_from_prediction(out)
    assert back.cam_names == ['cam0'] and back.rig.size('cam0') == wh
    with open(out / 'session.toml', 'rb') as f:
        cfg = tomllib.load(f)
    assert cfg['units'] == 'px'


@pytest.mark.parametrize('units', [None, 'px'])
def test_3d_videos_without_valid_units_refuses_before_checkpoint_load(cli, monkeypatch, tmp_path,
                                                                     units):
    """A multiview calibration determines 3D mode, but cannot determine units: the operator
    must declare them explicitly rather than inheriting a misleading default."""
    import conftest as cf

    rig, vids, _ = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, cf.KPTS_3D)
    argv = [
        'infer.py', '--run', str(run), '--videos', str(tmp_path / 'rec'),
        '--calibration', str(tmp_path / 'calib.toml'), '--cam-regex', 'cam([0-9]+)_',
        '--group-id', 't', '--session-id', 'rec', '--boxes', str(tmp_path / 'missing.npz'),
        '--anchor', 'none', '--device', 'cpu', '--out', str(tmp_path / 'pred')]
    if units is not None:
        argv += ['--units', units]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit, match='--units'):
        cli.main()
    assert not (tmp_path / 'pred').exists(), 'the refusal must precede output/checkpoint work'


def test_the_cli_runs_end_to_end_from_videos(cli, monkeypatch, tmp_path):
    """Through `main()` from raw videos: the prediction validates, `source_*` provenance is verbatim,
    `--dump-session` writes a valid session, and `session_from_prediction` finds the pixels."""
    import tomllib

    import conftest as cf
    from tailcyclenet.format import Session, validate_session
    from tailcyclenet.infer.predictions import load_predictions

    rig, vids, sdir = _videos_fixture(tmp_path)
    run, registry = _videos_run(tmp_path, cf.KPTS_3D)
    pts = np.random.default_rng(1).uniform(-20, 20,
                                           size=(1, 8, len(cf.KPTS_3D), 3)).astype(np.float32)
    np.savez(tmp_path / 'boxes.npz', **{'rec/t': pts})

    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', [
        'infer.py', '--run', str(run), '--videos', str(tmp_path / 'rec'),
        '--calibration', str(tmp_path / 'calib.toml'), '--cam-regex', 'cam([0-9]+)_',
        '--group-id', 't', '--session-id', 'rec', '--units', 'mm',
        '--boxes', str(tmp_path / 'boxes.npz'), '--anchor', 'none', '--device', 'cpu',
        '--overlap', '2', '--dump-session', str(tmp_path / 'dumped'), '--out', str(out)])
    cli.main()

    got = Session.load(out)
    assert got.mode == '3d' and got.units == 'mm' and got.names == list(cf.KPTS_3D)
    errs = [e for e in validate_session(got) if '[rule 7]' not in e]
    assert not errs, f'a prediction session must be valid apart from its missing pixels: {errs}'

    with open(out / 'session.toml', 'rb') as f:
        prov = tomllib.load(f)['provenance']
    assert prov['source'] == 'tailcyclenet infer --videos'
    assert prov['source_session'] == ''
    assert prov['source_calibration'] == str(tmp_path / 'calib.toml')
    assert prov['source_cam_regex'] == 'cam([0-9]+)_'
    assert sorted(prov['source_videos']) == sorted(str(v) for v in vids.values())

    preds, _ = load_predictions(out)
    assert 'rec/t' in preds and np.isfinite(preds['rec/t']['pred']).any()

    assert not validate_session(Session.load(tmp_path / 'dumped'))

    from tailcyclenet import adopt
    back = adopt.session_from_prediction(out)
    assert back.groups['t'].source('0')[1] == vids['0'].resolve()


@pytest.mark.parametrize('argv,expect', [
    ([], 'exactly one of --data'),
    (['--data', '/tmp/x', '--videos', '/tmp/y.mp4'], 'not allowed with'),
    (['--videos', '/tmp/y.mp4'], 'needs a box source'),
    (['--data', '/tmp/x', '--calibration', '/tmp/c.toml'], 'only means anything with --videos'),
])
def test_the_two_input_paths_are_exclusive_and_both_named(cli, monkeypatch, capsys, argv, expect):
    """`--data`/`--videos` are exactly-one-of and the refusal names both."""
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', '/tmp/r', '--out', '/tmp/o'] + argv)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    captured = capsys.readouterr().err
    err = captured or str(exc.value)
    assert expect in err, f'wanted {expect!r}, got: {err}'


def test_the_2d_row_gate_masks_the_saved_prediction(cli, monkeypatch, tmp_path):
    """A gated 2D row must stay gated in the saved points2d.pq (it used to be resurrected there,
    since `load_predictions` reads it back as the primary prediction)."""
    from tailcyclenet.infer.predictions import load_predictions

    sdir, run = _cli_setup(tmp_path)
    out = tmp_path / 'hi'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(sdir),
                                      '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--vis-thresh=1e9', '--out', str(out)])
    cli.main()
    preds, _ = load_predictions(out)
    assert not np.isfinite(preds['s/g000']['pred']).any()


def test_a_moving_rig_3d_prediction_carries_extrinsics_and_resolved_refine(cli, monkeypatch,
                                                                         tmp_path):
    """A moving-rig prediction writes the source's extrinsics.pq (else it fails rule 13), and a
    non-box 3D run records the RESOLVED `refine = true` (it once recorded `bool(None)`)."""
    import tomllib

    from tailcyclenet.format import Session, validate_session

    sdir, run = _cli_setup(tmp_path, 'mv', T=6, moving=True)
    out = tmp_path / 'pred'
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(sdir),
                                      '--split', 'test', '--anchor', 'none', '--device', 'cpu',
                                      '--overlap', '2', '--out', str(out)])
    cli.main()

    assert (out / 'extrinsics.pq').exists(), 'a moving-rig prediction must write extrinsics.pq'
    got = Session.load(out)
    np.testing.assert_array_equal(got.labels('g000').ext, Session.load(sdir).labels('g000').ext)
    errs = [e for e in validate_session(got) if '[rule 7]' not in e]
    assert not errs, f'a moving prediction session must validate apart from missing pixels: {errs}'

    with open(out / 'session.toml', 'rb') as f:
        prov = tomllib.load(f)['provenance']
    assert prov['refine'] is True, prov
    assert prov['box_prompt'] == 'none' and prov['box_prompt_requested'] == 'firstonly'
    assert isinstance(prov['crop_inflate'], float)
    assert 'gridresid_offset' in prov and 'gridresid_offset_override' in prov


@pytest.mark.parametrize('argv', [
    ['--box-prompt', 'labels', '--detector', 'nope'],
    ['--box-prompt', 'labels', '--boxes', 'nope.npz'],
])
def test_box_prompt_labels_refuses_a_detector_or_boxes_source(cli, monkeypatch, tmp_path, argv):
    """`--box-prompt labels` with a detector/`--boxes` source would feed label row a's box to a
    score-ordered detector row (another animal's ground truth) -- refused once the model is a box
    model."""
    sdir, run = _cli_setup(tmp_path, model_cfg={**SMALL, 'box_prompt': 'film'})
    monkeypatch.setattr(sys, 'argv', ['infer.py', '--run', str(run), '--data', str(sdir),
                                      '--split', 'test', '--device', 'cpu',
                                      '--out', str(tmp_path / 'out')] + argv)
    with pytest.raises(SystemExit, match='not label rows'):
        cli.main()


def test_det_trace_file_is_valid_json(tmp_path):
    """`--det-trace` writes JSON a reader can load: it used to end in a literal backslash-n."""
    import json

    from tailcyclenet.infer.driver import _write_det_trace

    path = tmp_path / 'sub' / 'trace.json'
    _write_det_trace(path, {'s/g000': [{'frame': 0, 'kept': 2}]})
    assert json.loads(path.read_text()) == {'s/g000': [{'frame': 0, 'kept': 2}]}
    assert path.read_text().endswith('}\n')
