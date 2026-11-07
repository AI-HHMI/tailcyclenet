"""Label-free identity witnesses read off one 2D video.

- tracktail carry labels (which base row each carried point set sits on) and the tt edge table on base tracklets;
- the pixel-ID column's quality: ``kappa_uf`` (per-frame label uniqueness vs random columns) and the C0 calibration
  (Platt on label-free self-calibration pairs; C0 = kappa_uf >= 0.8 AND pixel link accuracy >= base link accuracy).

Every float op mirrors the frozen research code so its outputs reproduce bit for bit.
"""

import re

import numpy as np
from matplotlib.path import Path as MPath
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import ConvexHull
from scipy.special import expit

from tailcyclenet.identity.graph import NMIN

TT_GMAX = 250
"""tt edge table: every node pair with gap 0..TT_GMAX."""
KAPPA_MIN = 0.8
"""C0's pixel-uniqueness floor."""


# ---------------------------------------------------------------- body length
def body_length(pred, notes=None):
    """graph body length in px. ``notes`` (the source session group's notes) carrying ``body_length_px=`` or
    ``box_px=`` wins (idtracker / UDMT roots: centroid rows, no keypoint extent); else the median over row-frames
    of the keypoint-extent diagonal."""
    if notes:
        m = re.search(r'body_length_px=([0-9.]+)', str(notes)) or re.search(r'box_px=([0-9.]+)', str(notes))
        if m:
            return float(m.group(1))
    ext = np.nanmax(pred, 2) - np.nanmin(pred, 2)
    return float(np.nanmedian(np.linalg.norm(ext, axis=-1)))


def carry_body_length(pred, bl):
    """the carry reader's body length: median max-axis keypoint extent over ~200 sampled frames when K > 1, else the
    graph body length ``bl``."""
    if pred.shape[2] <= 1:
        return float(bl)
    ext = []
    for r in range(pred.shape[0]):
        for t in range(0, pred.shape[1], max(1, pred.shape[1] // 200)):
            X = pred[r, t][np.isfinite(pred[r, t]).all(-1)]
            if len(X) >= 2:
                ext.append(np.ptp(X, 0).max())
    return float(np.median(ext)) if ext else 20.0


# ---------------------------------------------------------------- tracktail carry witness
def _inside(pts, ref, tol):
    """points within tol of a finite ref keypoint or inside the ref keypoints' convex hull -> (n,) bool."""
    ref = ref[np.isfinite(ref).all(-1)]
    if len(ref) == 0 or len(pts) == 0:
        return np.zeros(len(pts), bool)
    near = np.min(np.linalg.norm(pts[:, None] - ref[None], axis=-1), 1) <= tol
    if len(ref) >= 3:
        try:
            h = ConvexHull(ref)
            near |= MPath(ref[h.vertices]).contains_points(pts)
        except Exception:
            pass
    return near


def carry_labels(recs, pred, bl, fmin=0.5, ratio=2.0, vmin=0.0, tol=0.15):
    """tracktail carry records -> L (S, T) int32 carried id per base row-frame (-1 = none), W (S, T) float32 share.

    recs: list of dict(ws, nfr, ids={id: dict(tr (nfr, n, 2), vis (nfr, n))}); a frame is read once, from the window
    that owns it (window w owns [w, w_next)). Per frame, F[id, row] = fraction of the id's visible points inside the
    row's keypoint hull (+tol BL); Hungarian, kept iff F >= fmin and >= ratio x the best competitor."""
    S, T = pred.shape[:2]
    L = np.full((S, T), -1, np.int32)
    W = np.zeros((S, T), np.float32)
    rs = sorted(recs, key=lambda r: r['ws'])
    X = np.nanmean(pred, 2)
    for q, r in enumerate(rs):
        w0 = r['ws']
        w1 = rs[q + 1]['ws'] if q + 1 < len(rs) else w0 + r['nfr']
        ids = list(r['ids'])
        if not ids:
            continue
        for t in range(w0, min(w1, T, w0 + r['nfr'])):
            F = np.zeros((len(ids), S))
            for a, k in enumerate(ids):
                tr = r['ids'][k]['tr'][t - w0]
                vis = r['ids'][k]['vis'][t - w0].astype(float)
                ok = np.isfinite(tr).all(-1) & (vis >= vmin)
                if ok.sum() < 2:
                    continue
                pts = tr[ok]
                c = pts.mean(0)
                for s in range(S):
                    if not np.isfinite(X[s, t]).all() or np.linalg.norm(X[s, t] - c) > 2.5 * bl:
                        continue
                    F[a, s] = _inside(pts, pred[s, t], tol * bl).mean()
            if not F.any():
                continue
            ra, cs = linear_sum_assignment(-F)
            for a, s in zip(ra, cs):
                f = F[a, s]
                if f < fmin:
                    continue
                r2 = max(np.delete(F[a], s).max() if S > 1 else 0, np.delete(F[:, s], a).max() if len(ids) > 1 else 0)
                if f >= ratio * r2:
                    L[s, t] = ids[a]
                    W[s, t] = f
    return L, W


def tt_pairs(info, gmax=TT_GMAX):
    """every node pair (k1, k2) with 0 <= a2 - b1 - 1 <= gmax -> (m, 2) int64."""
    info = np.asarray(info)
    g = info[None, :, 1] - info[:, None, 2] - 1
    k1, k2 = np.nonzero((g >= 0) & (g <= gmax))
    return np.stack([k1, k2], 1).astype(np.int64)


def _node_ids(info, L, W, n=30, nmin=3):
    """per node: None (< nmin labelled frames) or {tail, head: (majority id, its share, mean W)} over the last / first n labelled frames."""
    out = []
    for s, a, b in info:
        l_all = L[s, a : b + 1]
        w_all = W[s, a : b + 1]
        mall = np.flatnonzero(l_all >= 0)
        if len(mall) < nmin:
            out.append(None)
            continue
        res = {}
        for nm in ('tail', 'head'):
            idx = np.arange(max(0, len(l_all) - n), len(l_all)) if nm == 'tail' else np.arange(0, min(len(l_all), n))
            idx = idx[l_all[idx] >= 0]
            if not len(idx):
                idx = mall[-n:] if nm == 'tail' else mall[:n]
            ll = l_all[idx]
            bc = np.bincount(ll)
            res[nm] = (int(bc.argmax()), bc.max() / len(ll), float(w_all[idx].mean()))
        out.append(res)
    return out


def tt_edges(info, L, W, events, gmax=TT_GMAX):
    """the tt edge table -> (E (m, 2), lo (m,), ok (m,)). Node id = majority carried id over k1's last / k2's first 30
    labelled frames (>= 3 labelled frames); same id -> +(2 + 2c), different -> -(2 + 2c), c = min over the two nodes
    of majority share x mean W. 'different' abstains when k2's id was born after k1 ended or k1's id was killed
    before k2 started (events: ('birth' | 'kill', id, frame, ...))."""
    info = np.asarray(info).astype(np.int64)
    E = tt_pairs(info, gmax)
    born, died = {}, {}
    for e in events:
        if e[0] == 'birth':
            born[int(e[1])] = int(e[2])
        elif e[0] == 'kill':
            died[int(e[1])] = int(e[2])
    nd = _node_ids(info, L, W)
    lo = np.zeros(len(E))
    ok = np.zeros(len(E), bool)
    for q, (k1, k2) in enumerate(E):
        A = nd[k1]
        B = nd[k2]
        if A is None or B is None:
            continue
        i1, s1, w1 = A['tail']
        i2, s2, w2 = B['head']
        c = min(s1 * w1, s2 * w2)
        if i1 == i2:
            lo[q] = 2 + 2 * c
            ok[q] = True
        else:
            if born.get(i2, -1) > info[k1, 2] or died.get(i1, 10**9) < info[k2, 1]:
                continue
            lo[q] = -(2 + 2 * c)
            ok[q] = True
    return E, lo, ok


# ---------------------------------------------------------------- pixel column quality
def kappa_uf(P, info, T):
    """per-frame label uniqueness of the node argmax over <= 2000 frames with >= 2 live tracklets, normalised
    against 3 Dirichlet-random columns (seeds 0..2): (u - u_rand) / (1 - u_rand). NaN with no such frame."""
    info = np.asarray(info)
    tid = -np.ones((int(info[:, 0].max()) + 1, T), int)
    for k, (s, a, b) in enumerate(info):
        tid[s, a : b + 1] = k
    fr = np.flatnonzero((tid >= 0).sum(0) >= 2)
    if not len(fr):
        return np.nan
    fr = fr[np.linspace(0, len(fr) - 1, min(2000, len(fr))).astype(int)]

    def uf(M):
        """mean per-frame fraction of distinct argmax labels among live tracklets."""
        lab = M.argmax(1)
        return float(np.mean([len(set(lab[tid[tid[:, t] >= 0, t]].tolist())) / (tid[:, t] >= 0).sum() for t in fr]))

    u = uf(P)
    un = np.mean([uf(np.random.default_rng(s).dirichlet(np.ones(P.shape[1]), size=len(P))) for s in range(3)])
    return (u - un) / max(1 - un, 1e-6)


class _Seg:
    """node posterior over an arbitrary frame range of one row (float16-stored F / W, as the calibration reads them)."""

    def __init__(self, F, W):
        """F (S, T, L), W (S, T) frame weights; both stored float16."""
        self.F = F.astype(np.float16)
        self.W = W.astype(np.float16)
        self.L = F.shape[-1]
        self.c = {}

    def get(self, r, a, b):
        """(mean distribution, n frames) of row r over [a, b] (frames with W > 0), cached; None if none."""
        k = (int(r), int(a), int(b))
        if k not in self.c:
            w = self.W[r, a : b + 1].astype(np.float32)
            m = w > 0
            self.c[k] = (self.F[r, a : b + 1][m].astype(np.float32).mean(0), int(m.sum())) if m.any() else None
        return self.c[k]


def _lo_of(P1, P2, L):
    """log LR(same) of two node distributions vs uniform."""
    q = np.clip((P1 * P2).sum(-1), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)) + np.log(max(L - 1, 1))


def _pair(sg, r1, a1, b1, r2, a2, b2):
    """(lo, n1, n2) of two row-frame ranges, None if either has no pixel frame."""
    A = sg.get(r1, a1, b1)
    B = sg.get(r2, a2, b2)
    if A is None or B is None:
        return None
    return float(_lo_of(A[0], B[0], sg.L)), A[1], B[1]


def link_table(sg, info, Ec, n):
    """pixel log LR on every candidate link whose nodes have >= NMIN pixel frames -> (ei, lo, same_row)."""
    rows = []
    for i, (k1, k2, sr, g) in enumerate(Ec):
        if n[k1] < NMIN or n[k2] < NMIN:
            continue
        s1, a1, b1 = info[k1]
        s2, a2, b2 = info[k2]
        f = _pair(sg, s1, a1, b1, s2, a2, b2)
        if f is not None:
            rows.append((i, f[0], int(sr)))
    a = np.array(rows, float).reshape(-1, 3)
    return a[:, 0].astype(int), a[:, 1], a[:, 2].astype(int)


def selfcal_table(sg, info, inf, n, Ec, seed=0, max_diff=6000, reps=2):
    """label-free calibration pairs -> (lo, y). SAME: one tracklet split at t with a gap g (drawn from the candidate
    links' gaps) removed. DIFF: two time-overlapping tracklets on different rows, A up to t, B from t + g + 1. GT is
    never read."""
    rng = np.random.default_rng(seed)
    gaps = Ec[:, 3] if len(Ec) else np.array([0])
    lo, y = [], []
    for k in np.flatnonzero(inf & (n >= 2 * NMIN + 2)):
        s, a, b = info[k]
        Lk = b - a + 1
        for _ in range(reps):
            g = int(min(rng.choice(gaps), max(0, Lk - 2 * NMIN - 2)))
            hi = b - g - NMIN
            t = int(rng.integers(a + NMIN - 1, hi)) if hi > a + NMIN - 1 else a + (Lk - g) // 2
            f = _pair(sg, s, a, t, s, t + g + 1, b)
            if f and f[1] >= NMIN and f[2] >= NMIN:
                lo.append(f[0])
                y.append(1.0)
    iks = np.flatnonzero(inf)
    Ii = info[iks]
    ii, jj, lo_, hi_ = [], [], [], []
    for i0 in range(0, len(iks), 512):
        Ib = Ii[i0 : i0 + 512]
        lb = np.maximum(Ib[:, None, 1], Ii[None, :, 1])
        hb = np.minimum(Ib[:, None, 2], Ii[None, :, 2])
        a_, b_ = np.nonzero((hb - lb >= 4) & (Ib[:, None, 0] != Ii[None, :, 0]))
        ii.append(a_ + i0)
        jj.append(b_)
        lo_.append(lb[a_, b_])
        hi_.append(hb[a_, b_])
    ii, jj, lo_, hi_ = (np.concatenate(x) if x else np.zeros(0, int) for x in (ii, jj, lo_, hi_))
    for p in rng.permutation(len(ii))[:max_diff]:
        i, j = iks[ii[p]], iks[jj[p]]
        g = int(rng.choice(gaps))
        t = int(rng.integers(lo_[p], hi_[p] + 1))
        s1, a1, b1 = info[i]
        s2, a2, b2 = info[j]
        if t + g + 1 > b2:
            g = 0
        if t + 1 > b2:
            continue
        f = _pair(sg, s1, a1, t, s2, t + g + 1, b2)
        if f and f[1] >= NMIN and f[2] >= NMIN:
            lo.append(f[0])
            y.append(0.0)
    return np.array(lo, float), np.array(y, float)


def _half_binomial(raw, y):
    """pointwise half-binomial loss and gradient, the branchy closed form of scikit-learn's CyHalfBinomialLoss
    (scalar libm exp / log1p, so it reproduces its fit bit for bit)."""
    from math import exp, log1p

    lv = np.empty(len(raw))
    gv = np.empty(len(raw))
    for i, (r, t) in enumerate(zip(raw.tolist(), y.tolist())):
        if r <= -37:
            e = exp(r)
            lv[i] = e - t * r
            gv[i] = e - t
        elif r <= -2:
            e = exp(r)
            lv[i] = log1p(e) - t * r
            gv[i] = ((1 - t) * e - t) / (1 + e)
        elif r <= 18:
            e = exp(-r)
            lv[i] = log1p(e) + (1 - t) * r
            gv[i] = ((1 - t) - t * e) / (1 + e)
        else:
            e = exp(-r)
            lv[i] = e + (1 - t) * r
            gv[i] = ((1 - t) - t * e) / (1 + e)
    return lv, gv


def platt(lo, y, C=1.0, max_iter=100, tol=1e-4):
    """class-balanced Platt scaling on clip(lo, -12, 12) -> fn(lo) -> P(same). The objective and L-BFGS-B settings
    of scikit-learn's LogisticRegression(C=1.0, lbfgs) with balanced sample weights; a single-class table falls
    back to sigmoid(lo)."""
    x = np.clip(lo, -12, 12)
    if len(np.unique(y)) < 2:
        return lambda d: 1 / (1 + np.exp(-np.clip(d, -12, 12)))
    sw = np.where(y == 1, 0.5 / max((y == 1).mean(), 1e-9), 0.5 / max((y == 0).mean(), 1e-9))
    X = x[:, None].astype(np.float64)
    sw_sum = float(np.sum(sw))
    l2 = 1.0 / (C * sw_sum)

    def f(coef):
        """(weighted mean loss + L2 penalty, gradient) at coef = (w, intercept)."""
        w = coef[:-1]
        raw = X @ w + coef[-1]
        lv, gv = _half_binomial(raw, y)
        lv *= sw
        gv *= sw
        loss = float(np.sum(lv) / sw_sum) + float(0.5 * l2 * (w @ w))
        gv /= sw_sum
        grad = np.empty_like(coef)
        grad[:-1] = X.T @ gv + l2 * w
        grad[-1] = np.sum(gv)
        return loss, grad

    r = minimize(
        f,
        np.zeros(2),
        method='L-BFGS-B',
        jac=True,
        options=dict(maxiter=max_iter, maxls=50, gtol=tol, ftol=64 * np.finfo(float).eps),
    )
    coef = r.x.copy()
    return lambda d: expit((np.clip(d, -12, 12)[:, None] @ coef[None, :1].T + coef[1:]).ravel())


def c0(F, W, info, inf, n, Ec, P, T):
    """the C0 calibration -> dict(kappa_uf, pix_acc_cal, base_acc_cal, C0, n_links, n_selfcal).

    pix_acc_cal / base_acc_cal = mean calibrated P(correct) over the candidate links of the pixel call (lo > 0 =
    same) / of the base call (same row = same). An empty link or self-calibration table (no evidence to calibrate)
    resolves to C0 False."""
    T = int(T)
    info = np.asarray(info)
    kap = kappa_uf(P, info, T) if len(info) else np.nan
    sg = _Seg(F, W)
    ei, lo, sr = link_table(sg, info, Ec, n)
    slo, sy = selfcal_table(sg, info, inf, n, Ec)
    out = dict(kappa_uf=float(kap), pix_acc_cal=None, base_acc_cal=None, C0=False, n_links=len(ei), n_selfcal=len(sy))
    if not len(ei) or not len(sy):
        return out
    p = np.clip(platt(slo, sy)(lo), 1e-6, 1 - 1e-6)
    pc = float(np.where(lo > 0, p, 1 - p).mean())
    bc = float(np.where(sr == 1, p, 1 - p).mean())
    out.update(pix_acc_cal=pc, base_acc_cal=bc, C0=bool(kap >= KAPPA_MIN and pc >= bc))
    return out
