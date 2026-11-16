"""The tt carry witness: the memory-exact patch sampler and output-neutral slot chunking."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tailcyclenet.identity import carry


def test_fast_patches_equal_upstream():
    from tracktail.model.encoder_decoder import sample_patches
    g = torch.Generator().manual_seed(0)
    B, T, C, H, W, Z, P = 2, 5, 3, 24, 32, 7, 5
    images = torch.rand(B, T, C, H, W, generator=g)
    centers = torch.rand(B, Z, 2, generator=g) * torch.tensor([W + 6.0, H + 6.0]) - 3.0
    qt = torch.randint(0, T, (B, Z), generator=g)
    ref = sample_patches(images, centers, qt, P)
    got = carry.fast_sample_patches(images, centers, qt, P)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-6, rtol=0)


def test_fast_patches_scoped_to_decode():
    import tracktail.model.encoder_decoder as ED
    orig = ED.sample_patches
    with carry._fast_patches():
        assert ED.sample_patches is carry.fast_sample_patches
    assert ED.sample_patches is orig


class _Enc:
    def __init__(self, v):
        self.views, self.raw_features = [v], [v.mean((2, 3, 4)).reshape(1, -1, 1)]


def _toy_track(head, model, views, cgroup, q, qt, enc):
    """A per-batch-row tracker (as tracktail: queries mix only within a row): every point moves by
    its row's mean pixel times t, plus a within-row coupling through the row's query mean."""
    v = views[0].float()
    Bn, T = v.shape[:2]
    shift = v.reshape(Bn, T, -1).mean(-1)                       # (B, T)
    coup = q.mean(1, keepdim=True)                              # (B, 1, 2)
    tr = q[:, None] + shift[:, :, None, None] + 1e-3 * coup[:, None]
    return {'2d_pred': tr[None], 'vis_pred': torch.zeros(Bn, T, q.shape[1], 1)}


@pytest.mark.parametrize('chunk', [1, 2, 3, 8])
def test_chunked_equals_unchunked(monkeypatch, chunk):
    monkeypatch.setattr(carry, '_track', _toy_track)
    rng = np.random.default_rng(0)
    items, jobs = {}, []
    for a in range(5):
        v = torch.as_tensor(rng.integers(0, 255, (1, 4, 8, 8, 3)), dtype=torch.uint8)
        items[a] = dict(views=[v], enc=_Enc(v.float()), cg=[{'size': torch.tensor([8, 8])}])
        n = int(rng.integers(1, 6))
        qc = rng.random((n + 2, 2)).astype(np.float32) * 8
        jobs.append((a, np.arange(n), qc))

    def run(k):
        w = carry.CarryWitness.__new__(carry.CarryWitness)
        w.device, w.head, w.model, w.slot_chunk = 'cpu', None, None, k
        w.stats = dict(dec_calls=0, decode_slots_max=0, chunked_calls=0, chunk_min=0)
        return w._decode(items, jobs)
    ref, got = run(0), run(chunk)
    for a in ref:
        np.testing.assert_array_equal(ref[a][0], got[a][0])
        np.testing.assert_array_equal(ref[a][1], got[a][1])


def _witness(slot_chunk, stats=None):
    w = carry.CarryWitness.__new__(carry.CarryWitness)
    w.device, w.slot_chunk = 'cuda', slot_chunk
    w.stats = stats or dict(decode_slots_max=0, chunked_calls=0, chunk_min=0)
    return w


def _gjobs(n, n_queries, tokens=1536, frames=12):
    enc = SimpleNamespace(raw_features=[torch.zeros(1, tokens, 4096, dtype=torch.float32,
                                                    device='meta')])
    views = [torch.zeros(1, frames, 256, 256, 3, dtype=torch.uint8, device='meta')]
    items = {a: dict(enc=enc, views=views) for a in range(n)}
    return [(a, np.arange(n_queries), None) for a in range(n)], items


def test_decode_slot_bytes_matches_the_measured_decode():
    """Peak extra memory of `_track` per slot (fp32, 256 px, 12 frames, ViT-L), measured at B = 1..64:
    50.3 MiB at 20 queries and 108.8 MiB at 60. The model adds the concatenated tokens and views
    (24 + 2.25 MiB, which the measurement excluded), so the model sits at or just above the sum."""
    gj, items = _gjobs(1, 60)
    est = carry.decode_slot_bytes(items[0]['enc'], items[0]['views'], 60) / 2**20
    assert 135.0 <= est <= 1.05 * 135.0
    gj, items = _gjobs(1, 20)
    est = carry.decode_slot_bytes(items[0]['enc'], items[0]['views'], 20) / 2**20
    assert 76.5 <= est <= 1.05 * 76.5


def test_auto_slot_chunk_is_one_call_whenever_it_fits():
    assert carry.auto_slot_chunk(100, 100.0, 10_000.0) == (100, True)
    assert carry.auto_slot_chunk(100, 100.0, 5_000.0) == (50, False)
    assert carry.auto_slot_chunk(100, 100.0, 50.0) == (1, False)


def test_auto_decodes_all_slots_at_once_unless_the_gpu_is_short(monkeypatch, capsys):
    """Default (0) = every decoded slot in one call and `chunked_calls` stays 0; a budget below the
    estimate chunks at the largest K that fits, warns once, and records K."""
    gj, items = _gjobs(40, 60)
    per_slot = carry.decode_slot_bytes(items[0]['enc'], items[0]['views'], 60)
    monkeypatch.setattr(carry, 'gpu_budget_bytes', lambda device: 40 * per_slot)
    w = _witness(0)
    assert w._chunk_for(gj, items, 60) == 40
    assert (w.stats['chunked_calls'], w.stats['decode_slots_max']) == (0, 40)
    monkeypatch.setattr(carry, 'gpu_budget_bytes', lambda device: 16.5 * per_slot)
    assert w._chunk_for(gj, items, 60) == 16
    assert w._chunk_for(gj, items, 60) == 16
    assert (w.stats['chunked_calls'], w.stats['chunk_min']) == (2, 16)
    out = capsys.readouterr().out
    assert out.count('NOT identity-exact') == 1 and 'chunking at 16 slots' in out


def test_explicit_slot_chunk_wins_over_the_estimate(monkeypatch):
    gj, items = _gjobs(40, 60)
    monkeypatch.setattr(carry, 'gpu_budget_bytes', lambda device: 1e18)
    w = _witness(8)
    assert w._chunk_for(gj, items, 60) == 8
    assert w.stats['chunk_min'] == 8
    w = _witness(64)
    assert w._chunk_for(gj, items, 60) == 64
    assert w.stats['chunked_calls'] == 0


def test_auto_off_cuda_is_one_call():
    w = _witness(0)
    w.device = 'cpu'
    gj, items = _gjobs(5, 10)
    assert w._chunk_for(gj, items, 10) == 5


def test_slot_chunk_report_names_k_and_exactness():
    from tailcyclenet.identity.stage import slot_chunk_report
    assert slot_chunk_report(dict(slot_chunk=0, decode_slots_max=37)) == dict(
        requested=0, chunked_calls=0, exact=True, used=37)
    assert slot_chunk_report(dict(slot_chunk=0, chunked_calls=3, chunk_min=12,
                                  decode_slots_max=60)) == dict(
        requested=0, chunked_calls=3, exact=False, used=12)
