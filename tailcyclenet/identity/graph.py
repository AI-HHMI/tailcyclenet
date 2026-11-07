"""TrackGraph solver: one graph per video over base tracklets, witness responses calibrated per video by
Dawid-Skene (label-free), max-weight path cover, a greedy stage-2 path merge, and the pixel-ID tracklet anchor.

Nodes are base tracklets ``info`` (n, 3) = (row, first frame, last frame) in base row-frame space. The only
configuration that ships is F2C0: the base column merged with the constant-velocity motion witness, the pixel-ID
label column, and the pixel anchor iff the C0 calibration allows it (``solve_f2c0``). Every float op mirrors the
frozen research solver so its labels reproduce bit for bit.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

LO_BINS = (-3.0, -1.0, 1.0, 3.0)
"""Response bins on a witness's log LR(same) vs uniform: D2 | D1 | abstain | S1 | S2."""
NMIN = 5
"""Pixel frames for a node to be informative."""
GMAX = 250
"""Max gap (frames) of a cross-row candidate link."""
EPS = 0.1
"""Path-cover entry / exit cost: lam = log post - log EPS."""
NEAR_TIE = 0.1
"""A tracklet whose pixel top-2 gap < NEAR_TIE is not an anchor class."""


def row_positions(pred):
    """(S, T, K, d) -> (S, T, d) centroid over finite keypoints."""
    return np.nanmean(pred.reshape(pred.shape[0], pred.shape[1], -1, pred.shape[-1]), 2)


def pix_frames(probs, S, T):
    """pixel-ID posteriors (Kens, S, T', N) -> per-frame distribution F (S, T, N) float32 (ensemble mean, 0 where
    absent) and frame weight W (S, T) float32 (1 where every class is finite)."""
    P = np.nanmean(probs.astype(np.float32), 0)[:, :T]
    W = np.isfinite(P).all(-1).astype(np.float32)
    F = np.zeros((S, T, P.shape[-1]), np.float32)
    F[:, : P.shape[1]] = np.nan_to_num(P)
    Wf = np.zeros((S, T), np.float32)
    Wf[:, : W.shape[1]] = W
    return F, Wf


def node_agg(F, Wt, info):
    """frame-level label witness (F (S, T, L), weight Wt (S, T), 0 = no vote) -> node distribution P (n, L) and
    vote count n (n,)."""
    info = np.asarray(info)
    L = F.shape[-1]
    P = np.full((len(info), L), 1.0 / L)
    n = np.zeros(len(info))
    for k, (s, a, b) in enumerate(info):
        w = Wt[s, a : b + 1]
        m = w > 0
        if m.any():
            f = F[s, a : b + 1][m]
            ww = w[m]
            P[k] = (f * ww[:, None]).sum(0) / ww.sum()
            n[k] = m.sum()
    return P, n


def build_edges(info, inf, gmax=GMAX):
    """CONTINUE candidates among informative nodes -> (m, 4) int64 (k1, k2, same_row, gap). k1 -> k2 iff k2 is the
    first informative node on its row starting after k1 ends, gap <= gmax, and k1 is the last informative node on
    its own row ending before k2 starts. The same-row consecutive pair is always a candidate (any gap), so keeping
    every same-row edge reproduces the base rows."""
    info = np.asarray(info)
    idx = np.flatnonzero(inf)
    rows = {}
    for k in idx:
        rows.setdefault(int(info[k, 0]), []).append(k)
    for r in rows:
        rows[r].sort(key=lambda k: info[k, 1])
    starts = {r: np.array([info[k, 1] for k in lst]) for r, lst in rows.items()}
    E = []
    for r, lst in rows.items():
        for i, k1 in enumerate(lst):
            b1 = info[k1, 2]
            nxt_same = lst[i + 1] if i + 1 < len(lst) else None
            for r2, l2 in rows.items():
                j = int(np.searchsorted(starts[r2], b1 + 1))
                if j >= len(l2):
                    continue
                k2 = l2[j]
                a2 = info[k2, 1]
                if r2 == r:
                    E.append((k1, k2, 1, a2 - b1 - 1))
                    continue
                if a2 - b1 - 1 > gmax:
                    continue
                if nxt_same is not None and info[nxt_same, 2] < a2:
                    continue
                E.append((k1, k2, 0, a2 - b1 - 1))
    return np.array(E, dtype=np.int64).reshape(-1, 4)


def label_response(P, n, E, nmin=NMIN, bins=LO_BINS):
    """graded response of a label witness per edge: 0 abstain, 1 D2, 2 D1, 3 S1, 4 S2 (from q = sum_c P1 P2 vs
    uniform) -> (resp, lo)."""
    L = P.shape[1]
    k1, k2 = E[:, 0], E[:, 1]
    ok = (n[k1] >= nmin) & (n[k2] >= nmin)
    q = np.clip((P[k1] * P[k2]).sum(1), 1e-6, 1 - 1e-6)
    lo = np.log(q / (1 - q)) + np.log(max(L - 1, 1))
    r = np.digitize(lo, bins)
    resp = np.select([r == 0, r == 1, r == 2, r == 3, r == 4], [1, 2, 0, 3, 4])
    resp[~ok] = 0
    return resp, lo


def lo_response(lo, ok, bins=LO_BINS):
    """an edge witness's log LR -> the same graded response as ``label_response``."""
    r = np.digitize(lo, bins)
    resp = np.select([r == 0, r == 1, r == 2, r == 3, r == 4], [1, 2, 0, 3, 4])
    resp[~ok] = 0
    return resp


def base_response(E):
    """the base rows as a binary witness: 2 = same (same-row edge), 1 = different."""
    return np.where(E[:, 2] == 1, 2, 1)


def motion_lo(X, info, E, bl, m=3, vmax_dt=10, sigma=0.5):
    """label-free constant-velocity witness: forward (k1 end) and backward (k2 start) extrapolation across the gap,
    distance in body lengths, softmax competition among k1's out- and k2's in-candidates -> lo (log LR same vs
    uniform), ok (>= 2 competitors and a finite distance)."""
    info = np.asarray(info)
    n = len(E)
    d = np.full(n, np.inf)

    def endstate(k, end):
        """(mean position, velocity) over a node's last (end) / first m finite frames; (None, None) if none."""
        s, a, b = info[k]
        x = X[s, a : b + 1]
        ok = np.isfinite(x).all(-1)
        x = x[ok]
        if not len(x):
            return None, None
        if end:
            p = x[-m:].mean(0)
            v = (x[-1] - x[max(0, len(x) - 6)]) / max(1, min(5, len(x) - 1))
        else:
            p = x[:m].mean(0)
            v = (x[min(len(x) - 1, 5)] - x[0]) / max(1, min(5, len(x) - 1))
        return p, v

    cache = {}
    for i, (k1, k2, sr, g) in enumerate(E):
        if (k1, 1) not in cache:
            cache[(k1, 1)] = endstate(k1, True)
        if (k2, 0) not in cache:
            cache[(k2, 0)] = endstate(k2, False)
        (p1, v1), (p2, v2) = cache[(k1, 1)], cache[(k2, 0)]
        if p1 is None or p2 is None:
            continue
        dt = min(g + 1, vmax_dt)
        ef = np.linalg.norm(p1 + v1 * dt - p2)
        eb = np.linalg.norm(p2 - v2 * dt - p1)
        d[i] = 0.5 * (ef + eb) / bl
    s = np.exp(-0.5 * (np.where(np.isfinite(d), d, 50) / sigma) ** 2) + 1e-12
    out = np.zeros(n)
    inn = np.zeros(n)
    cnt_o = np.zeros(n)
    cnt_i = np.zeros(n)
    for col, acc, cnt in ((0, out, cnt_o), (1, inn, cnt_i)):
        u, inv = np.unique(E[:, col], return_inverse=True)
        tot = np.bincount(inv, weights=s)
        acc[:] = s / tot[inv]
        cnt[:] = np.bincount(inv)[inv]
    q = np.clip(np.sqrt(out * inn), 1e-6, 1 - 1e-6)
    nc = np.maximum(np.maximum(cnt_o, cnt_i), 2)
    lo = np.log(q / (1 - q)) + np.log(nc - 1)
    ok = ((cnt_o >= 2) | (cnt_i >= 2)) & np.isfinite(d)
    return lo, ok


def base_motion_response(E, X, info, bl):
    """F2's merged base column: the motion witness's binned call (2 same / 1 different) where it reads, the base
    rows elsewhere."""
    lo, ok = motion_lo(X, info, E, bl)
    return np.where(ok, np.where(lo > 0, 2, 1), base_response(E))


def dawid_skene(R, ncat, iters=100, alpha=0.5):
    """R (n_edge, W) int responses (0 = abstain), ncat[w] non-abstain categories. Latent z in {D=0, S=1}, majority
    vote init. -> post (n,), ll (n,) log-odds, pi, theta list of (2, ncat+1) P(resp | z)."""
    n, W = R.shape
    s = np.zeros(n)
    t = np.zeros(n)
    for w in range(W):
        m = R[:, w] > 0
        hi = R[:, w] > ncat[w] / 2
        s += m & hi
        t += m
    post = np.where(t > 0, (s + 0.5) / (t + 1), 0.5)
    for _ in range(iters):
        pi = np.clip(post.mean(), 1e-4, 1 - 1e-4)
        th = []
        for w in range(W):
            M = np.full((2, ncat[w] + 1), alpha)
            M[:, 0] = 0
            for c in range(1, ncat[w] + 1):
                m = R[:, w] == c
                M[1, c] += post[m].sum()
                M[0, c] += (1 - post[m]).sum()
            M[:, 1:] /= M[:, 1:].sum(1, keepdims=True)
            th.append(M)
        ll = np.full(n, np.log(pi / (1 - pi)))
        for w in range(W):
            m = R[:, w] > 0
            ll[m] += np.log(th[w][1, R[m, w]]) - np.log(th[w][0, R[m, w]])
        new = 1 / (1 + np.exp(-ll))
        if np.abs(new - post).max() < 1e-6:
            post = new
            break
        post = new
    return post, ll, pi, th


def path_cover(E, lam, n_nodes):
    """max-weight one-in / one-out matching over edges with lam > 0 (time-ordered edges => paths) -> succ {k1: k2}."""
    pos = lam > 0
    if not pos.any():
        return {}
    Ep = E[pos]
    lp = lam[pos]
    srcs = np.unique(Ep[:, 0])
    dsts = np.unique(Ep[:, 1])
    si = {k: i for i, k in enumerate(srcs)}
    di = {k: i for i, k in enumerate(dsts)}
    C = np.zeros((len(srcs), len(dsts)))
    for (k1, k2), lv in zip(Ep[:, :2], lp):
        C[si[k1], di[k2]] = max(C[si[k1], di[k2]], lv)
    r, c = linear_sum_assignment(-C)
    return {int(srcs[i]): int(dsts[j]) for i, j in zip(r, c) if C[i, j] > 0}


def paths_of(info, succ, inf):
    """the solved paths (lists of node ids, head first) over informative nodes."""
    pred = {v: k for k, v in succ.items()}
    out = []
    for k in np.flatnonzero(inf):
        if k in pred:
            continue
        x = [int(k)]
        while x[-1] in succ:
            x.append(succ[x[-1]])
        out.append(x)
    return out


def merge_paths(info, succ, inf, LWl, ths, N):
    """stage 2: greedily link a path TAIL to a later path HEAD (disjoint in time, so exclusion holds) by the label
    witnesses' path-aggregated distributions and their stage-1 DS per-bin LLRs, prior -log(N - 1); best pair first
    while its score > 0. Scores are cached per pair and invalidated on merge (same choices as recomputing)."""
    info = np.asarray(info)
    succ = dict(succ)
    pred = {v: k for k, v in succ.items()}
    prior = -np.log(max(N - 1, 1))

    def chain(h):
        """the path from head h."""
        x = [h]
        while x[-1] in succ:
            x.append(succ[x[-1]])
        return x

    heads = [k for k in np.flatnonzero(inf) if k not in pred]
    paths = {h: chain(h) for h in heads}

    def agg(nodes):
        """per witness, the vote-weighted mean distribution and vote total over ``nodes``."""
        out = []
        for P, c in LWl:
            w = c[nodes]
            tot = w.sum()
            out.append(((P[nodes] * w[:, None]).sum(0) / tot, tot) if tot > 0 else (None, 0))
        return out

    A = {h: agg(p) for h, p in paths.items()}
    span = {h: (info[p[0], 1], info[p[-1], 2]) for h, p in paths.items()}

    def score(h1, h2):
        """log-odds that path h2 continues h1 (-inf when no witness reads)."""
        s = prior
        used = 0
        for (Pa, na), (Pb, nb), th in zip(A[h1], A[h2], ths):
            if Pa is None or Pb is None or na < NMIN or nb < NMIN:
                continue
            L = len(Pa)
            q = np.clip((Pa * Pb).sum(), 1e-6, 1 - 1e-6)
            lo = np.log(q / (1 - q)) + np.log(max(L - 1, 1))
            r = int(np.digitize(lo, LO_BINS))
            resp = [1, 2, 0, 3, 4][r]
            if resp == 0:
                continue
            s += np.log(th[1][resp]) - np.log(th[0][resp])
            used += 1
        return s if used >= 1 else -np.inf

    cache = {}
    while True:
        best = (0.0, None, None)
        for h1 in A:
            e1 = span[h1][1]
            for h2 in A:
                if h2 == h1 or span[h2][0] <= e1:
                    continue
                sc = cache.get((h1, h2))
                if sc is None:
                    sc = cache[(h1, h2)] = score(h1, h2)
                if sc > best[0]:
                    best = (sc, h1, h2)
        if best[1] is None:
            break
        _, h1, h2 = best
        t = paths[h1][-1]
        succ[int(t)] = int(paths[h2][0])
        paths[h1] = paths[h1] + paths[h2]
        del paths[h2]
        A[h1] = agg(paths[h1])
        del A[h2]
        span[h1] = (span[h1][0], span[h2][1])
        del span[h2]
        cache = {k: v for k, v in cache.items() if h1 not in k and h2 not in k}
    return succ


def tracklet_anchor(info, succ, inf, P, n, N, ratio=0.6, nmin=NMIN):
    """pixel-ID class per confident tracklet (most certain first, exclusive in time, abstain at top-2 ratio >= ratio);
    the solved paths fill the rest: an unassigned node takes the nearest assigned class on its own path (previous
    first) if that class is free over its frames, else a fresh id >= N shared with its contiguous unassigned run.
    -> pid (n_nodes,)."""
    info = np.asarray(info)
    T = int(info[:, 2].max()) + 1
    occ = np.zeros((N, T), bool)
    cls = -np.ones(len(info), int)
    cand = []
    for k in np.flatnonzero(inf):
        if n[k] < nmin:
            continue
        o = np.argsort(P[k])[::-1]
        if P[k, o[1]] / max(P[k, o[0]], 1e-9) >= ratio:
            continue
        cand.append((-P[k, o[0]], int(k), int(o[0])))
    for _, k, c in sorted(cand):
        a, b = info[k, 1], info[k, 2]
        if occ[c, a : b + 1].any():
            continue
        occ[c, a : b + 1] = True
        cls[k] = c
    pid = -np.ones(len(info), int)
    nxt = N
    for p in paths_of(info, succ, inf):
        lab = [cls[k] for k in p]
        for i, k in enumerate(p):
            if lab[i] >= 0:
                continue
            prev = next((lab[j] for j in range(i - 1, -1, -1) if lab[j] >= 0), -1)
            nx = next((lab[j] for j in range(i + 1, len(p)) if lab[j] >= 0), -1)
            a, b = info[k, 1], info[k, 2]
            done = False
            for c in ([prev] if prev >= 0 else []) + ([nx] if nx >= 0 and nx != prev else []):
                if c < N and not occ[c, a : b + 1].any():
                    occ[c, a : b + 1] = True
                    lab[i] = c
                    done = True
                    break
            if not done:
                if i > 0 and lab[i - 1] >= N:
                    lab[i] = lab[i - 1]
                else:
                    lab[i] = nxt
                    nxt += 1
        for k, lb in zip(p, lab):
            pid[k] = lb
    return pid


def _free(occ, lv, a, t, T):
    """whether label lv is unused over frames [a, t)."""
    return lv >= 0 and not occ.get(int(lv), np.zeros(T, bool))[a:t].any()


def paths_to_labels(info, succ, inf, S, T, valid, pid_override=None):
    """paths over informative nodes -> lab (S, T). Uninformative nodes / gaps on a row ('sandwich' fill): the label
    before if it equals the label after (or nothing follows) and is free, else the label before if free, else the
    label after if free, else a fresh id. -1 where ``valid`` is False."""
    info = np.asarray(info)
    pid = -np.ones(len(info), int)
    nxt = 0
    pred = {v: k for k, v in succ.items()}
    if pid_override is not None:
        pid = pid_override.copy()
        nxt = int(pid.max()) + 1
    for k in np.flatnonzero(inf):
        if k in pred or pid[k] >= 0:
            continue
        x = k
        while True:
            pid[x] = nxt
            if x not in succ:
                break
            x = succ[x]
        nxt += 1
    lab = np.full((S, T), -1)
    P = max(nxt, 1)
    for k in np.flatnonzero(inf):
        s, a, b = info[k]
        lab[s, a : b + 1] = pid[k]
    occ = {}
    for s in range(S):
        for lv in np.unique(lab[s][lab[s] >= 0]):
            occ.setdefault(int(lv), np.zeros(T, bool))[lab[s] == lv] = True
    fresh = P
    for s in range(S):
        t = 0
        last = -1
        while t < T:
            if lab[s, t] >= 0:
                last = lab[s, t]
                t += 1
                continue
            a = t
            while t < T and lab[s, t] < 0:
                t += 1
            nx = lab[s, t] if t < T else -1
            if last >= 0 and (last == nx or nx < 0) and _free(occ, last, a, t, T):
                L_ = last
            elif last >= 0 and _free(occ, last, a, t, T):
                L_ = last
            elif nx >= 0 and _free(occ, nx, a, t, T):
                L_ = nx
            else:
                L_ = fresh
                fresh += 1
            lab[s, a:t] = L_
            occ.setdefault(int(L_), np.zeros(T, bool))[a:t] = True
            last = L_
    lab[~valid] = -1
    return lab


def solve_f2c0(info, F, W, X, bl, valid, N, anchor, ds_iters=100, min_cov=0.05):
    """the F2C0 solve on the whole video -> (lab (S, T) int, dec dict).

    Columns: the base rows merged with the motion witness (F2) and the pixel-ID label column; informative node =
    >= NMIN pixel frames. ``anchor`` (= C0) turns on the pixel tracklet anchor (near-tie masked). lab values:
    0..N-1 = pixel-ID class (anchored paths), >= N = fresh ids, -1 = invalid row-frame."""
    info = np.asarray(info)
    S, T = valid.shape
    P, c = node_agg(F, W, info)
    inf = c >= NMIN
    E = build_edges(info, inf, GMAX)
    dec = dict(n_nodes=len(info), n_inf=int(inf.sum()), n_edges=len(E), anchored=False)
    if not len(E):
        return paths_to_labels(info, {}, inf, S, T, valid), dec
    R = np.stack([base_motion_response(E, X, info, bl), label_response(P, c, E, NMIN)[0]], 1)
    post, ll, pi, th = dawid_skene(R, [2, 4], ds_iters)
    nwit = int(((R[:, 1:] > 0).mean(0) >= min_cov).sum())
    lam = ll.copy()
    if nwit < 1:
        lam = np.where(E[:, 2] == 1, 10.0, -10.0)
    lam = -np.logaddexp(0, -lam) - np.log(EPS)
    succ = path_cover(E, lam, len(info))
    n1 = len(succ)
    pid = None
    if nwit >= 1:
        succ = merge_paths(info, succ, inf, [(P, c)], [th[1]], N)
        if anchor:
            s2 = np.sort(P, 1)
            gap = s2[:, -1] - (s2[:, -2] if P.shape[1] > 1 else 0)
            ca = np.where(gap < NEAR_TIE, 0, c)
            pid = tracklet_anchor(info, succ, inf, P, ca, N)
            dec['anchored'] = True
    dec.update(nwit=nwit, n_merge=len(succ) - n1)
    return paths_to_labels(info, succ, inf, S, T, valid, pid_override=pid), dec


def relabel(pred, lab):
    """pred (S, T, K, d), lab (S, T) (-1 = drop) -> (L, T, K, d) with output row = label (NaN where a label is
    absent). A label appears on at most one row per frame."""
    L = int(lab.max()) + 1 if (lab >= 0).any() else 0
    S, T = lab.shape
    out = np.full((max(L, 1), T) + pred.shape[2:], np.nan)
    for s in range(S):
        m = lab[s] >= 0
        tt = np.flatnonzero(m)
        assert not np.isfinite(out[lab[s, m], tt, 0, 0]).any(), 'label conflict'
        out[lab[s, m], tt] = pred[s, m]
    return out
