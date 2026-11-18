"""The sync-free rewrites of the train step's host reads give the values the old reads gave."""
import numpy as np
import torch

from tailcyclenet import tracktail_compat as tc
from tailcyclenet.pose import set_decoder as sd


def _old_row_norm(g, f):
    g_ = g[f]
    return float((g_.amax(0) - g_.amin(0)).max()) * 2.0 if len(g_) >= 2 else 1.0


def test_row_norm_matches_the_masked_select_extent():
    gen = torch.Generator().manual_seed(0)
    for n_finite in (0, 1, 2, 7, 24):
        g = torch.randn(4, 6, 3, generator=gen) * 50
        f = torch.zeros(24, dtype=torch.bool)
        f[torch.randperm(24, generator=gen)[:n_finite]] = True
        f = f.reshape(4, 6)
        g = torch.where(f[:, :, None], g, torch.full_like(g, float('nan')))
        assert float(sd._row_norm(g, f)) == max(_old_row_norm(g, f), 1e-6)


def _old_hungarian(cp, gt, valid_rows, pm, norm):
    from scipy.optimize import linear_sum_assignment
    A = [i for i in range(gt.shape[0]) if bool(valid_rows[i])]
    matched = {}
    if A:
        Cm = np.zeros((cp.shape[1], len(A)))
        for j, i in enumerate(A):
            f = torch.isfinite(gt[i]).all(-1)
            d = (cp - gt[i][:, None]).abs().sum(-1)
            d = torch.where(f[:, None], d, torch.zeros_like(d)).sum((0, 2)) / f.sum().clamp(min=1)
            nm = norm if norm is not None else _old_row_norm(gt[i], f)
            Cm[:, j] = (sd.C_L1 * d / max(nm, 1e-6) + sd.C_PRES * (1 - pm)).cpu().numpy()
        r, c = linear_sum_assignment(Cm)
        for qi, j in zip(r, c):
            matched[int(qi)] = A[j]
    return matched


def test_hungarian_matches_the_per_column_readback():
    gen = torch.Generator().manual_seed(1)
    for norm in (None, 64.0):
        for _ in range(5):
            T, M, K, R, rows = 3, 5, 4, 3, 4
            cp = torch.randn(T, M, K, R, generator=gen)
            gt = torch.randn(rows, T, K, R, generator=gen)
            gt[0, 1, 2] = float('nan')
            pm = torch.rand(M, generator=gen)
            valid = [True, False, True, True]
            assert sd._hungarian(cp, gt, valid, pm, norm) == _old_hungarian(
                cp, gt, torch.tensor(valid), pm, norm)


def test_remembered_skew_answers_without_reading_the_matrix():
    mat = torch.eye(3)
    cam = {'mat': mat}
    assert tc.has_skew(cam) is False
    tc.remember_skew(mat, True)
    assert tc.has_skew(dict(cam)) is True
    assert tc.has_skew({'mat': mat.clone()}) is False
