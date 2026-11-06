"""OVC (infer/ovc.py): the seam fix's solver, chain bookkeeping and block-streaming."""
import numpy as np
import pytest

from tailcyclenet.infer import ovc


def _dense(D, lam):
    n = len(D) + 1
    A = lam * np.eye(n)
    b = np.zeros((n,) + D[0].shape)
    for w, d in enumerate(D):
        A[w, w] += 1
        A[w + 1, w + 1] += 1
        A[w, w + 1] -= 1
        A[w + 1, w] -= 1
        b[w] -= d
        b[w + 1] += d
    return np.linalg.solve(A, b.reshape(n, -1)).reshape(b.shape)


def test_solve_matches_dense_full_and_lookahead():
    rng = np.random.default_rng(0)
    D = [rng.normal(size=(5, 3)) for _ in range(20)]
    assert np.allclose(ovc.solve(D, 1.0, None), _dense(D, 1.0), atol=1e-12)
    C = ovc.solve(D, 1.0, 2)
    for w in range(21):
        m = min(w + 2, 20)
        assert np.allclose(C[w], _dense(D[:m], 1.0)[w], atol=1e-12)


def test_known_offsets_are_recovered_and_lookahead_is_close():
    # windows = truth + an independent offset o_w; D_w = o_w - o_{w+1}; c_w should move toward -(o_w - local mean)
    rng = np.random.default_rng(1)
    o = rng.normal(size=(200, 4, 2))
    D = [o[w] - o[w + 1] for w in range(199)]
    full = ovc.solve(D, 1.0, None)
    err_before = np.abs(o - o.mean(0)).mean()
    err_after = np.abs(o + full - (o + full).mean(0)).mean()
    assert err_after < 0.7 * err_before
    assert np.abs(ovc.solve(D, 1.0, 2) - full).max() < 0.1 * np.abs(full).max()
    assert np.abs(ovc.solve(D, 1.0, 6) - full).max() < 0.005 * np.abs(full).max()


def test_seam_offset_cut_and_nan():
    prev = np.zeros((4, 3, 2), np.float32)
    prev[:, :, 0] = [0, 10, 20]                       # extent 20
    new = prev - 1.0
    assert np.allclose(ovc.seam_offset(prev, new), 1.0)
    assert ovc.seam_offset(prev, prev - 10.0) is None   # 14 px > 0.25 x 20
    new2 = new.copy()
    new2[:, 1] = np.nan
    d = ovc.seam_offset(prev, new2)
    assert np.all(d[1] == 0) and np.allclose(d[0], 1.0)


def _blocks(o, src, sizes, chains, K=3):
    """Fake `_run_blocks` output: one slot, windows of stride 1 frame each owning one frame."""
    w = 0
    for n in sizes:
        pred = np.zeros((1, n, K, 2), np.float32)
        pred2d = np.zeros((1, n, 1, K, 2), np.float32)
        for t in range(n):
            pred[0, t] = src[w + t]
            pred2d[0, t, 0] = src[w + t]
        for t in range(n):
            chains.observe(0, w + t, None if w + t == 0 else (o[w + t - 1] - o[w + t]))
        yield {'pred': pred, 'pred2d': pred2d, 'ovc_src': np.arange(w, w + n)[None].astype(np.int32),
               'ovc_chains': chains, 'ovc_upto': w + n - 1}
        w += n


@pytest.mark.parametrize('sizes', [[30], [7, 7, 7, 9], [1, 2, 3, 24], [29, 1]])
def test_block_partition_is_one_pass(sizes):
    rng = np.random.default_rng(2)
    o = rng.normal(size=(30, 3, 2)).astype(np.float32)
    ref = np.concatenate([b['pred'] for b in ovc.stream(_blocks(o, o, [30], ovc.Chains()))], 1)
    got = np.concatenate([b['pred'] for b in ovc.stream(_blocks(o, o, sizes, ovc.Chains()))], 1)
    assert np.array_equal(ref, got)
    D = [o[w] - o[w + 1] for w in range(29)]
    assert np.allclose(ref[0], o + ovc.solve(D, ovc.LAM, ovc.LOOKAHEAD), atol=1e-5)
    g2 = np.concatenate([b['pred2d'][:, :, 0] for b in ovc.stream(_blocks(o, o, sizes, ovc.Chains()))], 1)
    assert np.array_equal(g2, got)


def test_chain_breaks_are_independent():
    ch = ovc.Chains()
    d = np.ones((2, 2))
    ch.observe(0, 0, None)
    ch.observe(0, 1, d)
    ch.observe(0, 2, None)          # cut: a new chain
    ch.observe(0, 3, -d)
    c = ch.corrections(0, final=True)
    assert np.allclose(c[(0, 0)], -c[(0, 1)]) and np.allclose(c[(0, 2)], -c[(0, 3)])
    assert np.allclose(c[(0, 0)], ovc.solve([d], 1.0, None)[0])
