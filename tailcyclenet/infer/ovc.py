"""OVC -- overlap-consensus window offsets: the seam fix (report 76, '### seam-jitter').

Without a carried prior every window is an independent draw, and what two windows disagree about on
the frames they share is a per-KEYPOINT window offset (pose shape), not a rigid shift and not frame
noise. The overlap estimates that the seam rule throws away measure exactly that disagreement, and
it contains no true motion. So, per (slot, keypoint) chain of consecutive windows, with
D_w = mean over the overlap frames of (window w's estimate - window w+1's estimate):

    min_c  sum_w |c_w - c_{w+1} + D_w|^2  +  LAM sum_w |c_w|^2

(the ridge stops the random-walk drift a pure difference fit has), and c_w is added to every row
window w wrote. Its influence decays by (2 + LAM - sqrt((2 + LAM)^2 - 4)) / 2 = 0.38 per window at
LAM = 1, so the system is solved with a fixed LOOK-AHEAD: c_w comes from the chain truncated at
window w + LOOKAHEAD (all of its past kept), which is exact, block-independent, and within ~0.01 mm
of the full solve at 2. A chain is CUT where the two windows disagree by more than CUT x the slot's
body extent (median over keypoints): an identity change must not be averaged -- the reason the
deleted `--seam blend` (an average of the covering windows) lost MOTA. Unlike that blend, OVC
moves every frame of a window by one offset solved jointly over the chain; it averages the windows'
independent offsets rather than two identities on the overlap frames.

LAM, CUT and LOOKAHEAD are fixed constants, not knobs (LAM chosen on social-dannce before any other
readout). The solve is a tridiagonal (Thomas) sweep: the forward elimination of rows < m does not
depend on where the chain is truncated, so each window costs O(LOOKAHEAD).
"""
from __future__ import annotations

import numpy as np

LAM = 1.0
CUT = 0.25
LOOKAHEAD = 2


def seam_offset(prev_tail: np.ndarray, new_head: np.ndarray, cut: float = CUT):
    """The overlap disagreement D (K, R) between two consecutive windows, or None if cut.

    Inputs: prev_tail -- the earlier window's estimates on the overlap frames (ov, K, R); new_head --
        the later window's estimates on the same frames (ov, K, R). Both float32 as the loop holds them.
    Outputs: D (K, R) float32 with NaN -> 0 (a keypoint neither window placed has no opinion), or None
        when the median keypoint disagreement exceeds `cut` x the earlier window's body extent (the
        max over axes of the median over frames of the keypoint bounding box) or nothing is finite.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dk = np.nanmean(prev_tail - new_head, 0)
        if not np.isfinite(prev_tail).any():
            return None
        ext = np.nanmedian(np.nanmax(prev_tail, 1) - np.nanmin(prev_tail, 1), 0).max()
        mag = np.nanmedian(np.linalg.norm(dk, axis=-1))
    if not np.isfinite(mag) or not np.isfinite(ext) or mag > cut * ext:
        return None
    return np.nan_to_num(dk)


def solve(D, lam: float = LAM, lookahead: int | None = LOOKAHEAD, which=None) -> np.ndarray:
    """Per-window corrections c (n, ...) for one chain of n = len(D) + 1 windows.

    Inputs: D -- the n - 1 seam disagreements, each (K, R) (or any common shape); lam -- the ridge;
        lookahead -- c_w from the chain truncated at window min(w + lookahead, n - 1); None = the full
        chain (the solve at the end of a group is the same thing: the truncation clips at the end).
        which -- window indices to solve for (default all; the others are left NaN).
    Outputs: float64 (n, *D[0].shape).

    Rows i < m of any truncation at window m are the full chain's rows (diag lam + (i > 0) + 1, rhs
    D_{i-1} - D_i), so the forward sweep is shared; only the last row (diag lam + 1, rhs D_{m-1}) and
    the back-substitution from m down to w depend on the truncation.
    """
    D = [np.asarray(d, np.float64) for d in D]
    if not D:
        raise ValueError('a chain of one window has no seam; its correction is zero')
    n = len(D) + 1
    shape = D[0].shape
    Df = np.stack([d.reshape(-1) for d in D])
    cp = np.empty(n)
    dp = np.empty((n, Df.shape[1]))
    for i in range(n - 1):
        a = lam + (1.0 if i > 0 else 0.0) + 1.0
        b = (Df[i - 1] if i > 0 else 0.0) - Df[i]
        den = a + (cp[i - 1] if i > 0 else 0.0)
        cp[i] = -1.0 / den
        dp[i] = (b + (dp[i - 1] if i > 0 else 0.0)) / den
    out = np.full((n, Df.shape[1]), np.nan)
    L = (n - 1) if lookahead is None else int(lookahead)
    for w in (range(n) if which is None else which):
        m = min(w + L, n - 1)
        if m == 0:
            out[w] = 0.0
            continue
        x = (Df[m - 1] + dp[m - 1]) / (lam + 1.0 + cp[m - 1])
        for i in range(m - 1, w - 1, -1):
            x = dp[i] - cp[i] * x
        out[w] = x
    return out.reshape((n,) + shape)


class Chains:
    """Streaming OVC state for one group: per chain key (slot[, camera]), the windows seen in order and
    the seam disagreements between them.

    `observe(key, wi, D)` records that window `wi` follows window `wi - 1` of the same key with
    disagreement D (None = cut / not consecutive -> a new chain starts at `wi`). `start(key, wi)`
    opens a window with no predecessor. `corrections(upto, final)` returns {(key, wi): c} for every
    window whose correction is determined: wi + LOOKAHEAD <= upto (the last window index processed),
    or all of them when `final`.
    """

    def __init__(self, lam: float = LAM, lookahead: int = LOOKAHEAD):
        """`chains`: key -> list of (window ids, seam disagreements); `done`: (key, window) solved."""
        self.lam, self.lookahead = lam, lookahead
        self.chains = {}
        self.done = set()

    def start(self, key, wi):
        """Open a new chain for `key` at window `wi`."""
        self.chains.setdefault(key, []).append(([int(wi)], []))

    def observe(self, key, wi, D):
        """Window `wi` of `key` follows `wi - 1` with disagreement D; None or a gap starts a new chain."""
        cs = self.chains.get(key)
        if D is None or not cs or cs[-1][0][-1] != int(wi) - 1:
            self.start(key, wi)
            return
        cs[-1][0].append(int(wi))
        cs[-1][1].append(D)

    def corrections(self, upto: int, final: bool = False):
        """{(key, wi): c or None} for each not-yet-returned window with wi + lookahead <= upto (all if final)."""
        out = {}
        for key, cs in self.chains.items():
            for wins, D in cs:
                todo = [j for j, wi in enumerate(wins) if (key, wi) not in self.done
                        and (final or wi + self.lookahead <= upto)]
                if not todo:
                    continue
                if not D:
                    for j in todo:
                        out[(key, wins[j])] = None
                        self.done.add((key, wins[j]))
                    continue
                C = solve(D, self.lam, self.lookahead, which=todo)
                for j in todo:
                    out[(key, wins[j])] = C[j]
                    self.done.add((key, wins[j]))
        return out


def _apply(blk, corr):
    """Add each window's correction to the rows it wrote (`pred`; in 2D also `pred2d`, which IS the
    per-camera pose there). In 3D the per-camera `pred2d` is left as predicted."""
    src = blk.pop('ovc_src')
    pred = blk['pred']
    two_d = pred.shape[-1] == 2
    n_moved = 0
    for a in range(src.shape[0]):
        for wi in np.unique(src[a]):
            if wi < 0:
                continue
            c = corr.get((a, int(wi)))
            if c is None:
                continue
            m = src[a] == wi
            pred[a, m] = (pred[a, m] + c[None]).astype(pred.dtype)
            if two_d:
                blk['pred2d'][a, m, 0] = (blk['pred2d'][a, m, 0] + c[None]).astype(pred.dtype)
            n_moved += int(m.sum())
    return n_moved


def stream(blocks, stats=None):
    """Apply OVC to a group's block stream. A block is held until every window it holds has
    `LOOKAHEAD` processed successors (or the group ends), so its rows are corrected exactly as a
    whole-clip solve truncated at window w + LOOKAHEAD would correct them -- block-independent.

    Inputs: blocks -- `_run_blocks`' generator (each block carries `ovc_src`, `ovc_chains`, `ovc_upto`);
        stats -- the loop's telemetry dict or None. Outputs: the same blocks, in order, corrected.
    """
    held, corr, chains = [], {}, None
    for blk in blocks:
        chains = blk.pop('ovc_chains')
        upto = blk.pop('ovc_upto')
        held.append((upto, blk))
        corr.update(chains.corrections(upto))
        while held and held[0][0] + chains.lookahead <= upto:
            _, b = held.pop(0)
            n = _apply(b, corr)
            if stats is not None:
                stats['ovc_rows'] = stats.get('ovc_rows', 0) + n
            yield b
    if chains is not None:
        corr.update(chains.corrections(0, final=True))
    for _, b in held:
        n = _apply(b, corr)
        if stats is not None:
            stats['ovc_rows'] = stats.get('ovc_rows', 0) + n
        yield b
