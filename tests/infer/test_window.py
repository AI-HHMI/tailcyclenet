"""THE window loop.

The model is random and tiny: these assert that the loop RUNS and returns the right shapes,
identities and byte-identities, never that the numbers are good.
"""
import numpy as np
import pytest
import torch

from tailcyclenet.format import Registry, load_dataset
from tailcyclenet.infer.window import run_group
from tailcyclenet.pose.model import build_model
from tests.helpers import SMALL, _infer_cfg, _scene, _session_2d


@pytest.fixture(scope='module')
def rat16(tmp_path_factory):
    """2D session, T=16 -- several windows, which prefetch/seam/carry tests need."""
    return _scene(tmp_path_factory, 'rat', 16)


@pytest.fixture(scope='module')
def mv16(tmp_path_factory):
    """3D moving-rig session, T=16."""
    return _scene(tmp_path_factory, 'mv', 16)


def _boxes_for(boxes):
    """Adapt a whole-clip (S,T,C,4) array to `run_blocks`' `boxes_for` callback."""
    def f(store, lo, hi):
        return (boxes[:, lo:hi], None)
    return f


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


def test_detector_inference_without_keypoint_table(tmp_path):
    """A deployment detector can drive pose inference without 2D pose labels."""

    session_path = tmp_path / 'portable' / 'train' / 'session'
    _session_2d(session_path, T=4, S=2)
    (session_path / 'keypoints.pq').unlink()
    ds = load_dataset(session_path.parent.parent)
    sess = ds.sessions['train'][0]
    registry = Registry.build([ds])
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()
    boxes = np.zeros((1, 4, 1, 4), np.float32)
    boxes[..., 2:] = 32

    out = run_group(model, sess, 'g000', registry, ds.name,
                    _infer_cfg(anchor='none', refine=False), boxes_for=_boxes_for(boxes), n_rows=1)

    assert out['pred'].shape == (1, 4, sess.n_keypoints, 2)
    assert out['animal_ids'] == ['det00']


@pytest.mark.parametrize('anchor', ['self', 'labels'])
def test_every_anchor_runs(scene, anchor):
    """`self`/`labels` run on both shapes (`none`/`carry` run in nearly every other test)."""
    model, sess, registry, name = scene
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor=anchor))
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
        out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor=anchor),
                        boxes_for=_boxes_for(boxes), n_rows=len(boxes))
        assert out['pred'].shape[0] == 5
        # the id list must match the prediction row count, and name every row honestly
        assert len(out['animal_ids']) == 5
        assert all(str(i).startswith('det') for i in out['animal_ids'])
    # ...while the LABEL path keeps the labels' ids, which are identity there.
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'))
    assert list(out['animal_ids']) == list(sess.labels('g000').animal_ids)


def test_a_camera_without_a_box_does_not_drop_the_animal(mv):
    """A camera with no box leaves the animal predicted from the cameras that saw it (was coverage 0)."""
    model, sess, registry, name = mv
    C = len(sess.rig)
    w, h = sess.rig.size(sess.cam_names[0])
    boxes = np.zeros((1, 4, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h
    boxes[:, :, 0] = np.nan                       # camera 0 saw nothing

    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    assert np.isfinite(out['pred']).all(-1).any(), 'the surviving cameras must still predict'


def test_the_window_union_covers_the_window_and_stays_in_the_image(mv):
    """The union box (deliberately NOT re-squared, +3.06 mm) spans the window and stays in the image."""
    model, sess, registry, name = mv
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    for t in range(4):
        boxes[0, t, :] = [10 + t, 12, 14 + t, 16]         # drifts right across the window

    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    crop = out['crop'][0, 0]                              # (C,4), the first window's box per cam
    for ci in range(C):
        x0, y0, x1, y1 = crop[ci]
        assert (x0, y0, x1, y1) == (10.0, 12.0, 17.0, 16.0), \
            f'cam {ci}: {crop[ci]} is not the union over t = 0..3'
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h


def test_group_shorter_than_the_overlap(rat):
    """`p[-overlap]` ran off the front of a window with fewer frames than the step."""
    model, sess, registry, name = rat
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', overlap=8))
    assert out['pred'].shape[1] == 4


def test_kpt_chunk_matches_unchunked_end_to_end(mv):
    """Chunking is a memory knob, so it must not move the prediction by so much as an ulp."""
    model, sess, registry, name = mv
    whole = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'))['pred']
    chunked = run_group(model, sess, 'g000', registry, name,
                        _infer_cfg(anchor='none', kpt_chunk=2))['pred']
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-5)


def test_prefetch_windows_is_bit_exact(rat16):
    """Prefetching future windows moves nothing: the default (1) and 3 equal the serial loop (0)."""
    model, sess, registry, name = rat16
    base = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='carry', refine=True, prefetch_windows=0))
    _assert_same(base, run_group(model, sess, 'g000', registry, name,
                                 _infer_cfg(anchor='carry', refine=True)), ' (default prefetch)')
    _assert_same(base, run_group(model, sess, 'g000', registry, name,
                                 _infer_cfg(anchor='carry', refine=True, prefetch_windows=3)),
                 ' (prefetch=3)')


def test_capture_overlap_agreement_is_a_side_channel_with_an_honest_census(rat16):
    """The opt-in seam capture is off by default and never moves a prediction; when on it records
    both seam values (distance reproducible from them) and a census that agrees with its rows."""
    from tailcyclenet.infer.window import merge_blocks, run_blocks

    model, sess, registry, name = rat16
    cfg = _infer_cfg(anchor='carry')
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
                                _infer_cfg(anchor='carry', refine=refine), stats=stats))
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
        run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', overlap=0))
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='self', overlap=0))
    assert np.isfinite(out['pred']).any()


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
                            _infer_cfg(anchor='none', box_prompt='labels'))
    finally:
        model.query_encoder.forward = orig
    assert out['pred'].shape[0] >= 1
    assert not any('skipped' in str(x.message) for x in w)
    assert 'shape' in seen and seen['shape'][2] == len(sess.rig)
    assert out['box_prompt_cams'][0, 0] >= 1


def test_box_prompt_none_is_byte_identical_on_2d_and_3d(mv):
    """`box_prompt='none'` + `crop_inflate=1.0` is byte-identical to never mentioning the box."""
    model, sess, registry, name = mv
    base = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'))
    off = run_group(model, sess, 'g000', registry, name,
                    _infer_cfg(anchor='none', box_prompt='none', crop_inflate=1.0))
    np.testing.assert_array_equal(base['pred'], off['pred'])
    assert 'box_prompt_cams' not in off


def test_box_prompt_first_only_withholds_it_once_carried(rat16):
    """`first_only` gives the box at an animal's first window and withholds it once carry holds;
    off, it fires every window."""
    model, sess, registry, name = rat16
    cams = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='carry', box_prompt='labels',
                          box_prompt_first_only=True))['box_prompt_cams']
    assert cams.shape[1] > 1
    assert (cams[:, 0] >= 1).any() and (cams[:, 1:] == -1).all()
    cams = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='carry', box_prompt='labels',
                          box_prompt_first_only=False))['box_prompt_cams']
    assert (cams[:, 1:] >= 1).any()


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
                      _infer_cfg(anchor='none', refine=True, crop_inflate=1.0), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    wide = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='none', refine=True, crop_inflate=1.5), boxes_for=_boxes_for(boxes), n_rows=len(boxes))

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
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none', box_prompt='detector'),
                    boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    assert (out['box_prompt_cams'][0] >= 1).any()
    with pytest.raises(ValueError, match='detector'):
        run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none', box_prompt='detector'))


def test_box_prompt_detector_from_a_boxes_file_uses_its_points(mv):
    """`--boxes` points on a box model give the same prompt as `labels` from those points (used to raise)."""
    model, sess, registry, name = mv
    lab = sess.labels('g000')
    pts = lab.points3d if sess.mode == '3d' else lab.points2d[..., 0, :]
    got = run_group(model, sess, 'g000', registry, name,
                    _infer_cfg(anchor='none', box_prompt='detector'), box_points=pts)
    want = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='none', box_prompt='labels'))
    assert (got['box_prompt_cams'] >= 1).any()
    np.testing.assert_array_equal(got['box_prompt_cams'], want['box_prompt_cams'])
    np.testing.assert_array_equal(got['pred'], want['pred'])


def test_refine_recrops_to_its_own_prediction_and_keeps_the_coverage(mv):
    """Pass 2 re-crops around pass 1's prediction, never costs coverage, keeps `crop` as the pass-1
    box, and overlaps the box it came from."""
    model, sess, registry, name = mv
    C = len(sess.rig)
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((1, 4, C, 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h          # the whole frame: refinement must shrink it

    plain = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    ref = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none', refine=True),
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
                  _infer_cfg(anchor='none', refine=True, refine_px=32), boxes_for=_boxes_for(boxes), n_rows=len(boxes))
    finally:
        model.forward = inner

    assert 32 in seen, 'pass 1 must have run at the reduced resolution'
    assert 64 in seen, ('every SECOND pass must run at image_size, including the ones whose '
                        'refinement was rejected -- that fallback is the silent half')


def test_refine_px_off_is_bit_identical(mv):
    """`--refine` alone == `refine_px=None` == `refine_px=image_size`, bit for bit (strict `<` gate)."""
    model, sess, registry, name = mv
    torch.manual_seed(0)
    a = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none', refine=True))
    torch.manual_seed(0)
    b = run_group(model, sess, 'g000', registry, name,
                  _infer_cfg(anchor='none', refine=True, refine_px=None))
    torch.manual_seed(0)
    c = run_group(model, sess, 'g000', registry, name,
                  _infer_cfg(anchor='none', refine=True, refine_px=64))   # == _infer_cfg's image_size
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
    base = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry'))
    lo = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', vis_thresh=-1e9))
    hi = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', vis_thresh=1e9))

    np.testing.assert_array_equal(np.isnan(lo['pred']), np.isnan(base['pred']))
    assert np.isnan(hi['pred']).all(), 'every row is below an infinite threshold'
    assert (hi['outcome'] == 0).all(), 'the gate must not abort a window'


def test_max_frames_is_a_prefix_and_the_zero_start_range(rat16):
    """`--max-frames N` is a prefix and the same thing as `--end-frame N`; 0 or > T is the group."""
    model, sess, registry, name = rat16
    a = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none', max_frames=9))
    b = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none', frame_stop=9))
    np.testing.assert_array_equal(np.nan_to_num(a['pred'], nan=-9e9),
                                  np.nan_to_num(b['pred'], nan=-9e9))
    assert a['pred'].shape[1] == a['n_frames'] == 9
    assert a['frame_start'] == 0 and a['frame_stop'] == 9
    for m in (0, 99):
        assert run_group(model, sess, 'g000', registry, name,
                         _infer_cfg(anchor='none', max_frames=m))['pred'].shape[1] == 16


def test_2d_is_bit_identical_under_either_carry_source(rat16):
    """2D has nothing to triangulate, so `carry_source` must not move a 2D prediction."""
    model, sess, registry, name = rat16
    torch.manual_seed(0)
    a = run_group(model, sess, 'g000', registry, name,
                  _infer_cfg(anchor='carry', carry_source='triangulate'))
    torch.manual_seed(0)
    b = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', carry_source='pred'))
    np.testing.assert_array_equal(np.nan_to_num(a['pred'], nan=-9e9),
                                  np.nan_to_num(b['pred'], nan=-9e9))


def test_3d_carry_feeds_back_the_anchor_free_estimate(mv16):
    """...while in 3D `carry_source` must change the prediction, or the de-loop is wired to nothing."""
    model, sess, registry, name = mv16
    tri = run_group(model, sess, 'g000', registry, name,
                    _infer_cfg(anchor='carry', carry_source='triangulate'))
    pred = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', carry_source='pred'))
    assert len(tri['window_start']) > 1
    assert not np.array_equal(np.nan_to_num(tri['pred'], nan=-9e9),
                              np.nan_to_num(pred['pred'], nan=-9e9))


def test_instances_boxes_get_the_training_crop_rule_at_inference(rat):
    """`box_source='instances'` boxes go through the loader's crop rule (pad 0); detector boxes keep
    the raw union (re-squaring is measured worse)."""
    from tailcyclenet.data import crop as cropmod

    model, sess, registry, name = rat
    w, h = sess.rig.size(sess.cam_names[0])
    S, T, C = len(sess.labels('g000').animal_ids), 4, len(sess.rig)

    wide = np.tile(np.array([10.0, 10.0, 10.0 + 0.6 * w, 10.0 + 0.2 * h], np.float32),
                   (S, T, C, 1))
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'), boxes_for=_boxes_for(wide), n_rows=len(wide))
    det_box = out['crop'][0, 0, 0]

    assert det_box[2] - det_box[0] != det_box[3] - det_box[1], \
        'the detector union must stay non-square, or the measured +3.06 mm result is being undone'

    stored = sess.labels('g000').boxes
    a, t = [(i, j) for i in range(stored.shape[0]) for j in range(stored.shape[1])
            if np.isfinite(stored[i, j, 0]).all()][0]
    cfg = _infer_cfg(anchor='none', box_source='instances')
    got = run_group(model, sess, 'g000', registry, name, cfg)

    x0, y0, x1, y1 = stored[a, t, 0]
    expect = cropmod.crop_box_for_points(
        torch.tensor([[x0, y0], [x1, y1]]), torch.tensor([int(w), int(h)]),
        cfg.min_crop_dim, pad=0)
    np.testing.assert_array_equal(got['crop'][a, 0, 0], np.asarray(expect, np.float32))


def test_the_per_camera_2d_pose_at_camera_zero_is_the_prediction(rat):
    """In 2D `pred2d[:, :, 0]` IS `pred`, bit for bit (same un-crop expression; the exact inverse
    would move every published 2D number)."""
    model, sess, registry, name = rat
    out = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'))
    C, K = len(sess.rig), sess.n_keypoints
    S, T = out['pred'].shape[0], out['pred'].shape[1]
    assert out['pred2d'].shape == (S, T, C, K, 2)
    assert out['conf2d'].shape == (S, T, C, K)
    assert np.isfinite(out['pred2d']).any(), 'the per-camera pose must actually be written'
    if sess.mode == '2d':
        np.testing.assert_array_equal(np.nan_to_num(out['pred2d'][:, :, 0], nan=-9e9),
                                      np.nan_to_num(out['pred'], nan=-9e9))


def test_merge_blocks_stitches_model_2d_confidence():
    from tailcyclenet.infer.window import merge_blocks

    first = {'model_conf2d': np.array([[[[0.1]]]], np.float32)}
    second = {'model_conf2d': np.array([[[[0.9]]]], np.float32)}
    merged = merge_blocks([first, second])
    np.testing.assert_array_equal(merged['model_conf2d'],
                                  np.array([[[[0.1]], [[0.9]]]], np.float32))


def test_a_range_is_byte_identical_where_the_window_partition_agrees(rat16):
    """Under `--anchor none` a range equals the whole-clip slice where the window partition agrees:
    outright to the clip end; mid-clip, up to the range's own (pulled-back) last window start."""
    model, sess, registry, name = rat16
    whole = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='none'))

    a = 6
    tail = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='none', frame_start=a))
    assert list(tail['window_start']) == [w for w in whole['window_start'] if w >= a]
    np.testing.assert_array_equal(np.nan_to_num(tail['pred'], nan=-9e9),
                                  np.nan_to_num(whole['pred'][:, a:], nan=-9e9))

    b = 14
    part = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='none', frame_start=a, frame_stop=b))
    assert part['pred'].shape[1] == b - a
    assert part['n_frames'] == b - a and part['frame_start'] == a and part['frame_stop'] == b
    assert int(part['window_start'][0]) == a
    assert int(part['window_start'][-1]) == b - 4, 'the last window must end exactly at --end-frame'
    last = int(part['window_start'][-1])
    np.testing.assert_array_equal(
        np.nan_to_num(part['pred'][:, :last - a], nan=-9e9),
        np.nan_to_num(whole['pred'][:, a:last], nan=-9e9))


def test_vis_gate_is_per_camera_and_drops_3d_only_when_every_camera_is_gated():
    """One occluded camera withholds only its own 2D row; the 3D row needs every camera gated."""
    import numpy as np

    from tailcyclenet.infer.window import vis_gate

    v2 = np.array([[[5.0, 5.0], [-5.0, -5.0]],
                   [[-5.0, -5.0], [-5.0, -5.0]]])
    drop, cam_drop = vis_gate(v2, None, 0.0)
    assert cam_drop.tolist() == [[False, True], [True, True]]
    assert drop.tolist() == [False, True]
    drop, cam_drop = vis_gate(None, np.array([[1.0, 1.0], [-1.0, -1.0]]), 0.0)
    assert cam_drop is None and drop.tolist() == [False, True]


def test_box_corrupt_zero_is_byte_identical_and_one_drops_every_box(mv):
    """`--box-corrupt` at probability 0 changes nothing; `drop:1` leaves no camera to crop."""
    model, sess, registry, name = mv
    C = len(sess.rig)
    boxes = np.zeros((2, 4, C, 4), np.float32)
    boxes[0, ..., :] = [2, 2, 40, 40]
    boxes[1, ..., :] = [20, 20, 60, 60]
    kw = dict(boxes_for=_boxes_for(boxes), n_rows=2)
    torch.manual_seed(0)
    base = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', refine=True), **kw)
    torch.manual_seed(0)
    zero = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='carry', refine=True,
                                box_corrupt='drop:0,shift:0:1,shrink:0:0.5,swap:0'), **kw)
    _assert_same(base, zero, ' (box_corrupt at p=0)')
    dropped = run_group(model, sess, 'g000', registry, name,
                        _infer_cfg(anchor='carry', refine=True, box_corrupt='drop:1'), **kw)
    assert not np.isfinite(dropped['pred']).any()


def _whole_frame_boxes(sess, S=1, T=4):
    """(S,T,C,4) boxes covering every camera's whole frame."""
    w, h = (int(x) for x in sess.rig.size(sess.cam_names[0]))
    boxes = np.zeros((S, T, len(sess.rig), 4), np.float32)
    boxes[..., 2], boxes[..., 3] = w, h
    return boxes


def _count_forwards(model):
    """Wrap `model.forward` to count calls; returns (calls list, restore)."""
    calls, inner = [], model.forward

    def spy(views, *a, **kw):
        calls.append(len(views))
        return inner(views, *a, **kw)
    model.forward = spy

    def restore():
        model.forward = inner
    return calls, restore


def test_refine_consensus_is_inert_without_an_outlier_and_reruns_without_one(mv, monkeypatch):
    """R1: no outlier -> byte-identical to the shipped refine (plus the trace); an outlier camera
    costs exactly one extra pass-1 forward on the inliers, and pass 2 re-crops it from the
    consensus pose; with fewer than two inliers left the shipped refine runs untouched."""
    from tailcyclenet.infer import window as wmod
    model, sess, registry, name = mv
    boxes = _whole_frame_boxes(sess)
    kw = dict(boxes_for=_boxes_for(boxes), n_rows=1)
    torch.manual_seed(0)
    base = run_group(model, sess, 'g000', registry, name, _infer_cfg(anchor='carry', refine=True), **kw)
    real = wmod.camera_consensus

    def flag(cams):
        def fake(points, p2, v2, use, *a):
            res, vis, _ = real(points, p2, v2, use, *a)
            return res, vis, np.array([ci in cams for ci in use])
        return fake

    on = _infer_cfg(anchor='carry', refine=True, refine_consensus=True)
    monkeypatch.setattr(wmod, 'camera_consensus', flag(()))
    torch.manual_seed(0)
    none = run_group(model, sess, 'g000', registry, name, on, **kw)
    assert (none.pop('refine_outlier') == 0).all()
    _assert_same(base, none, ' (R1 with no outlier)')

    monkeypatch.setattr(wmod, 'camera_consensus', flag((0, 1)))
    torch.manual_seed(0)
    few = run_group(model, sess, 'g000', registry, name, on, **kw)
    assert set(np.unique(few.pop('refine_outlier')[..., :2])) == {3}
    _assert_same(base, few, ' (R1 with one inlier)')

    monkeypatch.setattr(wmod, 'camera_consensus', flag((0,)))
    calls, restore = _count_forwards(model)
    try:
        one = run_group(model, sess, 'g000', registry, name, on, **kw)
    finally:
        restore()
    n_win = one['outcome'].shape[1]
    assert calls.count(2) == n_win, 'one consensus forward per window, on the two inliers'
    assert len(calls) == 3 * n_win
    ro = one['refine_outlier'][0]
    assert set(ro[:, 0]) <= {1, 2} and (ro[:, 1:] == 0).all()


def test_refine_consensus_composes_with_refine_all_cams(mv, monkeypatch):
    """R2 = R1 + TR1: with no outlier the pair is exactly TR1; with one, a camera pass 1 had no
    box for is cropped from the CONSENSUS pose too (when in frame), which R1 alone never does."""
    from tailcyclenet.infer import window as wmod
    model, sess, registry, name = mv
    boxes = _whole_frame_boxes(sess)
    boxes[:, :, 2] = np.nan                                        # camera 2: no detector box
    kw = dict(boxes_for=_boxes_for(boxes), n_rows=1)
    real = wmod.camera_consensus

    def flag(cams):
        def fake(points, p2, v2, use, *a):
            res, vis, _ = real(points, p2, v2, use, *a)
            return res, vis, np.array([ci in cams for ci in use])
        return fake

    torch.manual_seed(0)
    tr1 = run_group(model, sess, 'g000', registry, name,
                    _infer_cfg(anchor='carry', refine=True, refine_all_cams=True), **kw)
    monkeypatch.setattr(wmod, 'camera_consensus', flag(()))
    torch.manual_seed(0)
    both = run_group(model, sess, 'g000', registry, name,
                     _infer_cfg(anchor='carry', refine=True, refine_all_cams=True,
                                refine_consensus=True), **kw)
    both.pop('refine_outlier')
    _assert_same(tr1, both, ' (R2 with no outlier)')

    monkeypatch.setattr(wmod, 'camera_consensus', flag((0,)))
    monkeypatch.setattr(wmod, '_CONSENSUS_MIN_INLIERS', 1)        # 3-camera rig, one box-less
    r1 = run_group(model, sess, 'g000', registry, name,
                   _infer_cfg(anchor='carry', refine=True, refine_consensus=True), **kw)
    r2 = run_group(model, sess, 'g000', registry, name,
                   _infer_cfg(anchor='carry', refine=True, refine_consensus=True,
                              refine_all_cams=True), **kw)
    assert (r1['refine_outlier'][..., 0] != 3).all() and (r2['refine_outlier'][..., 0] != 3).all()
    assert not np.isfinite(r1['crop_refined'][..., 2, :]).any(), 'R1 alone never adds a camera'
    assert np.isfinite(r2['crop_refined'][..., 2, :]).all(-1).any(), \
        'R2 adds the box-less camera from the consensus pose'
