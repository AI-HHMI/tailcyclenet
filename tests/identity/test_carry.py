"""The tt carry witness: the memory-exact patch sampler and output-neutral slot chunking."""
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
        w.stats = dict(dec_calls=0)
        return w._decode(items, jobs)
    ref, got = run(0), run(chunk)
    for a in ref:
        np.testing.assert_array_equal(ref[a][0], got[a][0])
        np.testing.assert_array_equal(ref[a][1], got[a][1])
