"""The box prompt: the two encoders, the data helper, and the build guards.

The inference-loop wiring is pinned by `test_infer.py`; this pins the encoder and the data side.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import box_prompt as bpmod
from tailcyclenet import crop as cropmod
from tailcyclenet.model import build_model
from tailcyclenet.query_encoder import (BoxFilmEncoder, WideQueryEncoder,
                                        _normalize_box)
from test_model import SMALL


def _wide_cfg(**kw):
    cfg = dict(SMALL)
    cfg['query'] = 'prior'
    cfg['query_encoder'] = 'wide'
    cfg.update(kw)
    return cfg


# -- build_model wiring ---------------------------------------------------------------------

def test_plain_wide_has_no_box_and_film_matches_term_count():
    plain = build_model(_wide_cfg(), n_keypoints=4).query_encoder
    film = build_model(_wide_cfg(box_prompt='film'), n_keypoints=4).query_encoder
    assert type(plain) is WideQueryEncoder and not hasattr(plain, 'film')
    assert isinstance(film, BoxFilmEncoder) and film.n_fusion_terms == plain.n_fusion_terms


def test_removed_box_prompt_term_raises_by_name():
    with pytest.raises(SystemExit, match='term'):
        build_model(_wide_cfg(box_prompt='term'), n_keypoints=4)


def test_unknown_box_prompt_raises():
    with pytest.raises(AssertionError, match='box_prompt'):
        build_model(_wide_cfg(box_prompt='bogus'), n_keypoints=4)


# -- the data helper ------------------------------------------------------------------------

def _cam(size=(256, 256)):
    return {'size': torch.tensor(size, dtype=torch.int32)}


def test_compute_box_prompt_2d_matches_the_crop_rule():
    coords = torch.rand(4, 5, 2) * 200 + 20
    coords[0, 0] = float('nan')                       # a missing point must not break it
    box = bpmod.compute_box_prompt(coords, [_cam()], '2d')
    assert box.shape == (4, 1, 4)
    for t in range(4):
        ref = cropmod.crop_box_for_points(coords[t], _cam()['size'],
                                          bpmod.BOX_PROMPT_MIN_DIM, bpmod.BOX_PROMPT_PAD)
        assert torch.allclose(box[t, 0], ref.float())


def test_compute_box_prompt_all_nan_frame_is_nan():
    box = bpmod.compute_box_prompt(torch.full((2, 3, 2), float('nan')), [_cam()], '2d')
    assert torch.isnan(box).all()


def test_frames_mode_first_holds_frame_zero_and_jitter_preserves_nan():
    box = torch.arange(2 * 1 * 4, dtype=torch.float32).reshape(2, 1, 4)
    out = bpmod.apply_frames_mode(box, 'first')
    assert torch.equal(out[0], box[0]) and torch.equal(out[1], box[0])
    with pytest.raises(ValueError):
        bpmod.apply_frames_mode(box, 'bogus')
    nanbox = torch.tensor([[[10., 10., 50., 50.]], [[float('nan')] * 4]])
    j = bpmod.apply_jitter(nanbox, np.random.default_rng(0), 0.3, 0.3)
    assert torch.isfinite(j[0]).all() and torch.isnan(j[1]).all()


def test_normalize_box_roundtrip():
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


@pytest.mark.parametrize('cls', [BoxFilmEncoder])
def test_none_box_is_finite_and_idempotent(cls):
    torch.manual_seed(0)
    enc = _make(cls).eval()
    args = _enc_inputs(enc)
    enc._box_prompt = None
    a = enc(*args)
    enc._box_prompt = None
    assert torch.allclose(a, enc(*args)) and torch.isfinite(a).all()


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


def test_nan_box_substitutes_the_missing_token_not_nan():
    for cls in (BoxFilmEncoder,):
        torch.manual_seed(0)
        enc = _make(cls).eval()
        views, cg, qc, qt, tt, cs = _enc_inputs(enc)
        enc._box_prompt = torch.full((1, 2, 1, 4), float('nan'))
        assert torch.isfinite(enc(views, cg, qc, qt, tt, cs)).all()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
