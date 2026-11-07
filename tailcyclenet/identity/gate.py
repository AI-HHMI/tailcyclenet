"""The frozen identity gate (PREREG8) and ``run_identity``, the one entry point.

    gate_v8 = C0 AND [ gate_v7  OR  ( p_mcn <= 0.01 AND p_null <= 0.05 ) ]

over tt-refereed links (tt edge table, |lo| >= 3, both nodes >= 5 pixel frames):
- gate_v7: n_ttSAME >= 10 and the pixel column's agreement on tt-SAME links (gate_ttS) > ``bar(n, N)``, the 99th
  percentile of random-control columns (frozen draws, ``random_draws.json``).
- p_mcn: one-sided sign (McNemar) test that the pixel column beats the base rows; b = #(pixel agrees with tt, base
  does not), c = the reverse.
- p_null: Monte-Carlo p of the pixel column's total tt agreement against this video's own random columns (random
  class, exclusive N-colouring, random chain; 100 draws each, seeds 0 / 1 / 2), max over the families.
When ON, the base rows are replaced by the F2C0 labels; OFF keeps the base rows.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from tailcyclenet.identity import graph as G
from tailcyclenet.identity import witness as W

A_MCN, A_NULL, NDRAW, NMIN, LO, BAR_NMIN, BAR_Q = 0.01, 0.05, 100, 5, 3.0, 10, 0.99
DRAWS = Path(__file__).with_name('random_draws.json')
_D = None


def bar(n, N):
    """smallest fraction b such that a random-control column reaches gate_ttS >= b with probability <= 1% at n
    tt-SAME links (mixture of Binomial(n, p_i) over the frozen random-control draws of the N = 2 / N >= 4 group)."""
    global _D
    if _D is None:
        _D = json.loads(DRAWS.read_text())['draws']
    ps = np.clip(np.asarray(_D['N=2' if N == 2 else 'N>=4']), 1e-4, 1 - 1e-4)
    for k in range(n + 1):
        if np.mean(binom.sf(k - 1, n, ps)) <= 1 - BAR_Q:
            return k / n
    return 1.0


def links(z):
    """the tt-refereed links -> (E (m, 2), tt says same (m,), base says same (m,))."""
    E = z['E'][:, :2].astype(int)
    lo = z['lo']
    ok = z['ok'].astype(bool)
    n = z['n']
    info = z['info']
    m = ok & (np.abs(lo) >= LO) & (n[E[:, 0]] >= NMIN) & (n[E[:, 1]] >= NMIN)
    return E[m], lo[m] >= LO, info[E[m, 0], 0] == info[E[m, 1], 0]


def stats(lab, z):
    """agreement of a node labelling ``lab`` with tt: nl links, agree, nS tt-SAME links, aS = agreement on them,
    McNemar b / c / one-sided p against the base rows."""
    E, tts, bs = links(z)
    ps = lab[E[:, 0]] == lab[E[:, 1]]
    pa = ps == tts
    ba = bs == tts
    b = int((pa & ~ba).sum())
    c = int((~pa & ba).sum())
    return dict(
        nl=len(E),
        agree=int(pa.sum()),
        nS=int(tts.sum()),
        aS=float(ps[tts].mean()) if tts.any() else np.nan,
        b=b,
        c=c,
        pm=float(binom.sf(b - 1, b + c, 0.5)) if b + c else 1.0,
    )


def excl_lab(info, N, rng):
    """random exclusive N-colouring of the nodes in start order (a class is free once its last node ended)."""
    order = np.argsort(info[:, 1], kind='stable')
    lab = np.full(len(info), -1)
    mx = np.full(N, -1)
    for k in order:
        s, a, b = info[k]
        free = np.flatnonzero(mx < a)
        c = rng.choice(free) if len(free) else rng.integers(N)
        lab[k] = c
        mx[c] = max(mx[c], b)
    return lab


def chain_lab(z, rng):
    """random chains: a path cover over the candidate links with random weights, then a random exclusive colouring
    of the paths."""
    info = z['info']
    E = z['Ec']
    n = len(info)
    N = int(z['N'])
    T = int(z['T'])
    succ = G.path_cover(E, rng.random(len(E)) + 0.01, n)
    paths = G.paths_of(info, succ, z['inf'])
    onp = np.zeros(n, bool)
    for p in paths:
        onp[p] = True
    paths += [[int(k)] for k in np.flatnonzero(~onp)]
    sp = np.array([[info[p, 1].min(), info[p, 2].max()] for p in paths])
    order = np.lexsort((rng.random(len(paths)), sp[:, 0]))
    occ = np.zeros((N, T), bool)
    cls = -np.ones(n, int)
    for j in order:
        fr = np.zeros(T, bool)
        for k in paths[j]:
            fr[info[k, 1] : info[k, 2] + 1] = True
        free = [c for c in range(N) if not (occ[c] & fr).any()]
        c = free[rng.integers(len(free))] if free else int(rng.integers(N))
        occ[c] |= fr
        cls[paths[j]] = c
    return cls


def null_agree(z, ndraw=NDRAW, seed=0):
    """tt agreement of ndraw random columns per family (rand / excl / chain, seeds seed + 0 / 1 / 2)."""
    N = int(z['N'])
    info = z['info']
    out = {}
    for fi, fam in enumerate(('rand', 'excl', 'chain')):
        rng = np.random.default_rng(seed + fi)
        out[fam] = np.array(
            [
                stats(
                    rng.integers(N, size=len(info))
                    if fam == 'rand'
                    else excl_lab(info, N, rng)
                    if fam == 'excl'
                    else chain_lab(z, rng),
                    z,
                )['agree']
                for _ in range(ndraw)
            ]
        )
    return out


def p_null(null, agree):
    """Monte-Carlo p of ``agree`` against each family's draws, max over families."""
    return max((1 + np.sum(v >= agree)) / (1 + len(v)) for v in null.values())


def gate_v8(z, C0):
    """-> dict(on, v7, p_mcn, p_null, nl, agree, nS, aS, b, c, bar). z: info, E / lo / ok (tt edge table),
    P / n (pixel node posterior / frame count), Ec / inf (candidate links), N, T. p_null is drawn only when the
    McNemar arm decides (McNemar passes and v7 does not)."""
    s = stats(z['P'].argmax(1), z)
    N = int(z['N'])
    br = bar(s['nS'], N) if s['nS'] > 0 else None
    v7 = s['nS'] >= BAR_NMIN and np.isfinite(s['aS']) and s['aS'] > br
    mcn = s['pm'] <= A_MCN
    pn = p_null(null_agree(z), s['agree']) if (mcn and not v7) else None
    on = bool(C0 and (v7 or (mcn and pn <= A_NULL)))
    return dict(on=on, v7=bool(v7), p_mcn=s['pm'], p_null=pn, bar=br, **s)


def run_identity(pred, pix, tt_recs, tt_events, bl, solve_off=False):
    """one video's identity decision and labels.

    pred      (S, T, K, 2) base prediction rows (S rows = the session's animal slots, NaN = absent).
    pix       the pixel-ID result: dict(probs (Kens, S, T', N) float16, info (n, 3) base tracklets, N).
    tt_recs   tracktail carry records (list of dict(ws, nfr, ids)); tt_events: ('birth' | 'kill', id, frame) list.
    bl        graph body length in px (``witness.body_length``).
    -> dict(on, arm ('v7' | 'mcn' | None), inputs {...}, labels (S, T) int or None). labels = the F2C0 labels when
    ON (and when ``solve_off``), None = keep the base rows. Label values: 0..N-1 pixel-ID classes, >= N fresh ids,
    -1 = row-frame dropped (any keypoint non-finite); write with ``graph.relabel``."""
    pred = np.asarray(pred, float)
    S, T = pred.shape[:2]
    out = dict(on=False, arm=None, labels=None, inputs=dict(S=S, T=T))
    if pix is None or int(pix['N']) != S or not len(pix['info']):
        out['inputs']['note'] = 'no pixel column'
        return out
    valid = np.isfinite(pred[..., 0]).all(-1)
    N = int(pix['N'])
    info = np.asarray(pix['info']).astype(np.int64)
    F, Wp = G.pix_frames(np.asarray(pix['probs']), S, T)
    P, n = G.node_agg(F, Wp, info)
    inf = n >= G.NMIN
    Ec = G.build_edges(info, inf, G.GMAX)
    L, Wt = W.carry_labels(tt_recs, pred, W.carry_body_length(pred, bl))
    E, lo, ok = W.tt_edges(info, L, Wt, tt_events)
    cal = W.c0(F, Wp, info, inf, n, Ec, P, T)
    z = dict(info=info, E=E, lo=lo, ok=ok, P=P, n=n, Ec=Ec, inf=inf, N=N, T=T)
    g = gate_v8(z, cal['C0'])
    out['inputs'].update(
        cal,
        N=N,
        n_nodes=len(info),
        gate_ttS=None if not np.isfinite(g['aS']) else g['aS'],
        n_ttSAME=g['nS'],
        bar=g['bar'],
        b=g['b'],
        c=g['c'],
        p_mcn=g['p_mcn'],
        p_null=g['p_null'],
        n_tt_links=g['nl'],
        tt_agree=g['agree'],
        gate_v7=bool(cal['C0'] and g['v7']),
    )
    out['on'] = g['on']
    out['arm'] = ('v7' if g['v7'] else 'mcn') if g['on'] else None
    out['tt'] = dict(L=L, W=Wt, E=E, lo=lo, ok=ok)
    if g['on'] or solve_off:
        lab, dec = G.solve_f2c0(info, F, Wp, G.row_positions(pred), bl, valid, N, anchor=cal['C0'])
        out['labels'] = lab
        out['inputs']['solve'] = dec
    return out
