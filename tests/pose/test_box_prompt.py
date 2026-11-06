"""The box prompt: the FiLM encoder, the data helper, and the build guards (loop wiring is in
`tests/infer/test_window.py`)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from tailcyclenet.data import crop as cropmod
from tailcyclenet.pose import box_prompt as bpmod
from tailcyclenet.pose.model import build_model
from tailcyclenet.pose.query_encoder import BoxFilmEncoder, WideQueryEncoder, _normalize_box
from tests.helpers import SMALL


def _wide_cfg(**kw):
    cfg = dict(SMALL)
    cfg['query'] = 'prior'
    cfg['query_encoder'] = 'wide'
    cfg.update(kw)
    return cfg


# -- build_model wiring ---------------------------------------------------------------------

def test_build_wiring_and_refusals():
    """film keeps the plain encoder's term count; the deleted 'term' raises by name."""
    plain = build_model(_wide_cfg(), n_keypoints=4).query_encoder
    film = build_model(_wide_cfg(box_prompt='film'), n_keypoints=4).query_encoder
    assert type(plain) is WideQueryEncoder and not hasattr(plain, 'film')
    assert isinstance(film, BoxFilmEncoder) and film.n_fusion_terms == plain.n_fusion_terms
    with pytest.raises(SystemExit, match='term'):
        build_model(_wide_cfg(box_prompt='term'), n_keypoints=4)
    with pytest.raises(AssertionError, match='box_prompt'):
        build_model(_wide_cfg(box_prompt='bogus'), n_keypoints=4)


# -- the data helper ------------------------------------------------------------------------

def _cam(size=(256, 256)):
    return {'size': torch.tensor(size, dtype=torch.int32)}


def test_tight_prompt_box_is_the_per_axis_rule():
    """Pad 10 per side and floor 32 PER AXIS (about the centre), clamped -- never squared."""
    size = torch.tensor([256, 256], dtype=torch.int32)
    got = bpmod.tight_prompt_box(torch.tensor([[12., 20.], [50., 30.]]), size)
    # x: 2..60 (38 >= 32 kept); y: 10..40 is 30 < 32 -> centre 25 +- 16
    assert torch.equal(got, torch.tensor([2., 9., 60., 41.]))
    assert got[2] - got[0] != got[3] - got[1]
    edge = bpmod.tight_prompt_box(torch.tensor([[2., 100.], [200., 250.]]), size)
    assert torch.equal(edge, torch.tensor([0., 90., 210., 256.]))      # clamped, not shifted
    assert bpmod.tight_prompt_box(torch.full((3, 2), float('nan')), size) is None


def test_compute_box_prompt_2d_is_the_tight_rule_and_prefers_the_stored_box():
    coords = torch.rand(4, 5, 2) * 200 + 20
    coords[0, 0] = float('nan')                       # a missing point must not break it
    box = bpmod.compute_box_prompt(coords, [_cam()], '2d')
    assert box.shape == (4, 1, 4)
    for t in range(4):
        assert torch.equal(box[t, 0], bpmod.tight_prompt_box(coords[t], _cam()['size']))
    inst = torch.full((1, 4, 4, 2), float('nan'))
    inst[0, 1] = cropmod.box_corners(torch.tensor([30., 40., 120., 90.]))
    got = bpmod.compute_box_prompt(coords, [_cam()], '2d', box_pts=inst)
    assert torch.equal(got[1, 0], torch.tensor([20., 30., 130., 100.]))  # the instance box
    assert torch.equal(got[0, 0], box[0, 0]) and torch.equal(got[2:], box[2:])  # else keypoints
    box = bpmod.compute_box_prompt(torch.full((2, 3, 2), float('nan')), [_cam()], '2d')
    assert torch.isnan(box).all()


def test_train_and_deploy_build_the_same_prompt_from_the_same_box():
    """The loader's prompt from a stored instance box == `deploy_box_prompt` from the detector's
    tight box of the same value, through the same crop origin and scale (2D and per camera 3D)."""
    from tailcyclenet.infer.prior import deploy_box_prompt

    T = 3
    src = np.array([[100., 80., 180., 120.], [104., 82., 186., 121.], [108., 84., 190., 125.]],
                   np.float32)                                         # tight, source px
    sizes = [torch.tensor([128, 128], dtype=torch.int32), torch.tensor([96, 128], dtype=torch.int32)]
    origins = [np.array([70., 40.], np.float32), np.array([60., 30.], np.float32)]
    scales = [torch.tensor([1.25, 1.25]), torch.tensor([0.75, 0.8])]
    crop_boxes = [torch.tensor([*o, *(o + 100)]) for o in origins]
    boxes_stc = np.tile(src[None, :, None], (1, 1, 2, 1))              # (S=1, T, C=2, 4)
    for mode, C in (('2d', 1), ('3d', 2)):
        cg = [dict(size=sizes[c]) for c in range(C)]
        pts = [(cropmod.box_corners(torch.as_tensor(src)) - torch.as_tensor(origins[c])) * scales[c]
               for c in range(C)]
        train = bpmod.compute_box_prompt(torch.full((T, 2, 2), float('nan')), cg, '2d',
                                         box_pts=pts) if mode == '2d' else torch.stack(
            [bpmod.compute_box_prompt(torch.full((T, 2, 2), float('nan')), [cg[c]], '2d',
                                      box_pts=[pts[c]])[:, 0] for c in range(C)], 1)
        deploy = deploy_box_prompt(mode, None, boxes_stc, np.arange(T), 0, list(range(C)),
                                   crop_boxes[:C], scales[:C], cg, 'cpu')[0]
        assert torch.isfinite(train).all()
        torch.testing.assert_close(deploy, train)


def test_box_prompt_geometry_key():
    """One key: only 'tight' trains; a run folder without it trained on the retired square box."""
    from tailcyclenet.loader.config import LoaderConfig

    assert LoaderConfig().box_prompt_geometry == 'tight'
    with pytest.raises(ValueError, match='square box prompt was retired'):
        LoaderConfig(box_prompt_geometry='square')
    assert bpmod.trained_geometry({'data': {}}) == 'square'
    assert bpmod.trained_geometry({'data': {'box_prompt_geometry': 'tight'}}) == 'tight'


def test_frames_mode_first_holds_frame_zero_and_jitter_preserves_nan():
    box = torch.arange(2 * 1 * 4, dtype=torch.float32).reshape(2, 1, 4)
    out = bpmod.apply_frames_mode(box, 'first')
    assert torch.equal(out[0], box[0]) and torch.equal(out[1], box[0])
    with pytest.raises(ValueError):
        bpmod.apply_frames_mode(box, 'bogus')
    nanbox = torch.tensor([[[10., 10., 50., 50.]], [[float('nan')] * 4]])
    j = bpmod.apply_jitter(nanbox, np.random.default_rng(0), 0.3, 0.3)
    assert torch.isfinite(j[0]).all() and torch.isnan(j[1]).all()
    n = _normalize_box(torch.tensor([[[0., 0., 100., 200.]]]), torch.tensor([[[200., 200.]]]))
    assert torch.allclose(n[0, 0], torch.tensor([-0.5, 0.0, 1.0, 2.0]))


# -- the encoders ---------------------------------------------------------------------------

def _enc_inputs(enc, T=2, K=5):
    views = [torch.rand(1, T, 3, 32, 32)]
    cg = [dict(size=torch.tensor([32, 32], dtype=torch.int32), offset=torch.zeros(2),
               mat=torch.eye(3), ext=torch.eye(4), dist=torch.zeros(5), center=torch.zeros(3))]
    qc = torch.rand(1, T * K, 2) * 32
    qt = torch.zeros(1, T * K, dtype=torch.int64)
    tt = torch.zeros(1, T * K, dtype=torch.int64)
    for t in range(T):
        tt[:, t * K:(t + 1) * K] = t
    enc._kpt_ids = torch.arange(K)[None].repeat(1, T)
    enc._query_ok = torch.zeros(1, T * K, dtype=torch.bool)
    return views, cg, qc, qt, tt, torch.ones(1, 1)


def _make(cls, **kw):
    return cls(dim=32, embed_dim=16, decoder_dim=16, n_frames=8, n_keypoints=5, max_freq=3,
               patch_size=9, query_pos_embedding=True, query_patch_embedding=True, **kw)


def test_film_is_a_no_op_at_init_and_diverges_once_trained():
    torch.manual_seed(0)
    enc = _make(BoxFilmEncoder).eval()
    views, cg, qc, qt, tt, cs = _enc_inputs(enc)
    enc._box_prompt = None
    out_none = enc(views, cg, qc, qt, tt, cs)
    enc._box_prompt = torch.tensor([[[[5., 5., 20., 20.]], [[8., 8., 18., 25.]]]])
    assert torch.allclose(out_none, enc(views, cg, qc, qt, tt, cs))    # zero-init -> no-op
    with torch.no_grad():
        nn.init.normal_(enc.film[-1].weight, std=0.2)
    assert not torch.allclose(out_none, enc(views, cg, qc, qt, tt, cs))


def test_none_and_nan_boxes_are_finite():
    """No box is finite and idempotent; a NaN box substitutes the missing token, not NaN."""
    torch.manual_seed(0)
    enc = _make(BoxFilmEncoder).eval()
    args = _enc_inputs(enc)
    enc._box_prompt = None
    a = enc(*args)
    assert torch.allclose(a, enc(*args)) and torch.isfinite(a).all()
    enc._box_prompt = torch.full((1, 2, 1, 4), float('nan'))
    assert torch.isfinite(enc(*args)).all()
