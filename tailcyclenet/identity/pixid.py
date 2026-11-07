"""Per-video pixel-identity network (TRex / idtracker.ai style), trained label-free on one session's crops.

Input: a crops dict (`identity.crops`). Rows are cut into tracklets at every uncertainty (row absent,
centroid jump > `jump` BL, another row within `cut` BL). A global segment is a maximal frame run where
exactly N tracklets coexist; its N tracklets are N distinct animals. The network (V118_3: three conv5
blocks 16/64/128 + FC100 + FC N) trains on the best segment (quality = shortest tracklet), then
accumulates segments whose tracklets it already labels as a confident permutation of N (one duplicate
repaired by giving the missing id to the less confident claimant; min mean softmax > 1.5/N; consistent with labels already given). A step is kept iff the
uniqueness (fraction of ~333 sampled frames whose rows' argmax ids are all distinct) stays >= 0.97 x the
best so far, else it is rolled back. Stops when nothing is accepted, after 60 steps, or at a plateau
(uniqueness >= max(0.9, (N-0.5)/N) (0.95 if N < 3) and a kept step's gain < `plateau`); then a final
step trains on everything accepted. Output: per-crop softmax over N for every ok (row, frame).

Crops are unrotated detector-box crops, so the network is made rotation-blind (s21's recipe): every input is
masked to the inscribed circle, training rotates each crop by a uniform 0-360 deg angle (plus gain +-15%,
offset +-0.1; no shift), and EVERY softmax (inference and the training-time uniqueness / accumulation checks)
is the mean over `TTA_ROT` = 8 rotations (0, 45, ..., 315 deg).

The shipped setting is one network (`K = 1`), uint8 crops resident on the device, bf16 autocast on CUDA,
`plateau = 0.005`. Training draws from torch's global RNG on the device; it runs inside
`torch.random.fork_rng`, so the caller's RNG state is untouched and a run depends only on `seed`.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


TTA_ROT = 8


def circle_mask(w, dev):
    """(w, w) float mask of the inscribed circle (1 inside)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, w), indexing='ij')
    return ((yy ** 2 + xx ** 2) <= 1).float().to(dev)


def rotate(x, th):
    """Rotate (B, 1, H, W) images by per-image angles `th` (B,) radians about the centre (bilinear, zero fill)."""
    z = torch.zeros_like(th)
    A = torch.stack([torch.stack([th.cos(), -th.sin(), z], 1), torch.stack([th.sin(), th.cos(), z], 1)], 1)
    return Fn.grid_sample(x, Fn.affine_grid(A, x.shape, align_corners=False), align_corners=False)


def prep(xb, circ, aug=False, th0=0.0):
    """uint8 crops -> circle-masked, per-crop standardised floats. `aug`: a uniform 0-360 deg rotation, gain +-15%,
    offset +-0.1 (torch's device RNG); else a fixed rotation by `th0` radians (none at 0)."""
    dev = circ.device
    x = xb.to(dev).float() / 255.
    B = x.shape[0]
    if aug:
        x = rotate(x, torch.rand(B, device=dev) * 2 * np.pi)
    elif th0:
        x = rotate(x, torch.full((B,), float(th0), device=dev))
    x = x * circ
    if aug:
        x = x * (1 + (torch.rand(B, 1, 1, 1, device=dev) - .5) * .3) + (torch.rand(B, 1, 1, 1, device=dev) - .5) * .2
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-3)


@torch.no_grad()
def tta_softmax(net, xb, circ, autocast=None):
    """Mean softmax of `net` over `TTA_ROT` rotations (0, 360/TTA_ROT, ... deg) of uint8 crops `xb`."""
    acc = 0
    for r in range(TTA_ROT):
        if autocast is None:
            lo = net(prep(xb, circ, th0=2 * np.pi * r / TTA_ROT))
        else:
            with autocast():
                lo = net(prep(xb, circ, th0=2 * np.pi * r / TTA_ROT))
        acc = acc + Fn.softmax(lo.float(), -1)
    return acc / TTA_ROT


class V118_3(nn.Module):
    """TRex's V118_3 identity network on a W x H gray crop."""

    def __init__(self, W, H, N, C=1):
        """W x H input with C channels, N identities."""
        super().__init__()

        def blk(i, o):
            """conv5 + BatchNorm + ReLU + maxpool 2 + Dropout2d 0.05."""
            return nn.Sequential(nn.Conv2d(i, o, 5, padding='same'), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.05))
        self.f = nn.Sequential(blk(C, 16), blk(16, 64), blk(64, 128))
        self.fc1 = nn.Linear(128 * (W // 8) * (H // 8), 100)
        self.ln = nn.LayerNorm(100)
        self.do = nn.Dropout(0.05)
        self.fc2 = nn.Linear(100, N)

    def forward(self, x):
        """(B,C,H,W) normalised crops -> (B,N) logits."""
        x = self.f(x).flatten(1)
        return self.fc2(self.do(Fn.relu(self.ln(self.fc1(x)))))


def tracklets(ctr, ok, bl, cut=0.5, jump=0.5, sep_train=0.75):
    """Cut rows into tracklets. Returns (tid (S,T) tracklet id or -1, info [(row, first, last)],
    nearest-other-row distance (S,T), train mask = nearest other row >= `sep_train` BL)."""
    S, T = ok.shape
    d = np.full((S, T), np.inf)
    for i in range(S):
        for j in range(S):
            if i != j:
                dd = np.linalg.norm(ctr[i] - ctr[j], axis=-1)
                dd[~(ok[i] & ok[j])] = np.inf
                d[i] = np.minimum(d[i], dd)
    good = ok & (d >= cut * bl)
    tid = np.full((S, T), -1)
    info = []
    for s in range(S):
        t = 0
        while t < T:
            if not good[s, t]:
                t += 1
                continue
            a = t
            while t + 1 < T and good[s, t + 1] and np.linalg.norm(ctr[s, t + 1] - ctr[s, t]) <= jump * bl:
                t += 1
            tid[s, a:t + 1] = len(info)
            info.append((s, a, t))
            t += 1
    return tid, info, d, (d >= sep_train * bl)


def global_segments(tid, N):
    """Maximal frame runs where exactly N tracklets coexist: [(first, last, sorted tracklet ids)]."""
    S, T = tid.shape
    segs = []
    cur = None
    a = 0
    for t in range(T + 1):
        key = None
        if t < T:
            ids = tid[:, t]
            ids = ids[ids >= 0]
            if len(ids) == N:
                key = tuple(sorted(ids.tolist()))
        if key != cur:
            if cur is not None:
                segs.append((a, t - 1, cur))
            cur = key
            a = t
    return segs


def train_pixid(c, device=None, seed=0, K=1, N=0, cut=0.5, jump=0.5, epochs=12, cap=3000, B=8, lr=1e-3,
                max_steps=60, bs=128, plateau=0.005, amp=True, log=print):
    """Train the per-video identity network on crops dict `c` and score every ok crop.

    `N` = 0: the most common number of ok rows per frame. Returns a dict: probs (K,S,T,N) float16
    (NaN off `ok`), tid, info (n,3), lab (tracklet label or -1), mind, bl, N, gid, log (json), uniq,
    member_uniq, t_train, t_inf, n_inf, T, K; or {'fail': True} when there is no global segment."""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    devices = [dev] if dev.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices, device_type='cuda'):
        return _train(c, dev, seed, K, N, cut, jump, epochs, cap, B, lr, max_steps, bs, plateau,
                      amp and dev.type == 'cuda', log)


def _train(c, dev, seed, K, N, cut, jump, epochs, cap, Bc, lr, max_steps, bs, plateau, amp, log):
    """`train_pixid`'s body (see the module docstring). Crops stay uint8 on `dev`; uniqueness is scored on
    <= 333 frames, evenly spread, where exactly N tracklet rows exist; each accumulation step ranks the
    top 4B candidates by local uniqueness on <= 30 of their frames, lowest first; rejected candidates
    are retried after a kept step."""
    t00 = time.time()
    X = c['crops']
    ok = c['ok']
    ctr = c['ctr']
    bl = float(c['bl'])
    gid = c['gid']
    S, T = ok.shape
    na = ok.sum(0)
    N = N or int(np.bincount(na[na > 0]).argmax())
    pure = 1.0 / N
    addf = 1.5
    tid, info, mind, sepok = tracklets(ctr, ok, bl, cut, jump)
    L = np.array([b - x + 1 for _, x, b in info])
    segs = global_segments(tid, N)
    log(f'{gid} S{S} T{T} N{N} bl{bl:.0f} tracklets {len(info)} global segs {len(segs)} dev {dev}')
    good_u = 0.95 if N < 3 else max(0.9, (N - 0.5) / N)
    Xt = torch.from_numpy(X.reshape(S * T, 1, X.shape[-2], X.shape[-1])).to(dev)
    Wpx = Xt.shape[-1]

    def idx_of_trk(k, train=True):
        """Flat (row * T + frame) crop indices of tracklet k; `train` keeps frames >= 0.75 BL from any other row."""
        s, x, b = info[k]
        t = np.arange(x, b + 1)
        if train:
            t = t[sepok[s, x:b + 1]]
        return s * T + t
    uf = np.flatnonzero((tid >= 0).sum(0) == N)
    uf = uf[np.linspace(0, len(uf) - 1, min(333, len(uf))).astype(int)] if len(uf) else uf
    urows = [np.flatnonzero(tid[:, t] >= 0) for t in uf]

    circ = circle_mask(Wpx, dev)
    nets = []
    for k in range(K):
        torch.manual_seed(seed * 100 + k)
        nets.append(V118_3(Wpx, Wpx, N).to(dev))
    opts = [torch.optim.Adam(list(n.parameters()), lr=lr * (bs / 128) ** 0.5) for n in nets]

    def autocast():
        """bf16 autocast when `amp`."""
        return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp)

    @torch.no_grad()
    def probs(idx, members=None):
        """(K, n, N) rotation-averaged softmax (`tta_softmax`) of crops `idx` per network, 8192 at a time."""
        out = []
        for n in (nets if members is None else [nets[m] for m in members]):
            n.eval()
            ps = []
            for i in range(0, len(idx), 8192):
                xb = Xt[torch.as_tensor(idx[i:i + 8192], device=Xt.device)]
                ps.append(tta_softmax(n, xb, circ, autocast).cpu())
            out.append(torch.cat(ps).numpy() if ps else np.zeros((0, N), np.float32))
        return np.stack(out)

    def uniqueness(members=None):
        """Fraction of the uniqueness frames whose rows' ensemble argmax ids are all distinct."""
        if not len(uf):
            return 0.
        flat = np.concatenate([r * T + t for r, t in zip(urows, uf)])
        am = probs(flat, members).mean(0).argmax(-1)
        o = 0
        good = 0
        for r in urows:
            ids = am[o:o + len(r)]
            o += len(r)
            good += len(set(ids.tolist())) == len(ids)
        return good / len(urows)

    def train(lab, n_epochs):
        """Train every network on labelled tracklets `lab` ({tracklet: class}); <= cap crops per class per
        epoch, numpy RNG seeded by seed + len(lab)."""
        idx = np.concatenate([idx_of_trk(k) for k in lab])
        y = np.concatenate([np.full(len(idx_of_trk(k)), c_) for k, c_ in lab.items()])
        rng = np.random.default_rng(seed + len(lab))
        for _ in range(n_epochs):
            sel = np.concatenate([rng.permutation(np.flatnonzero(y == c_))[:cap] for c_ in range(N)])
            sel = rng.permutation(sel)
            for n, o in zip(nets, opts):
                n.train()
                ss = rng.permutation(sel)
                for i in range(0, len(ss), bs):
                    b = ss[i:i + bs]
                    xb = prep(Xt[torch.as_tensor(idx[b], device=Xt.device)], circ, aug=True)
                    yb = torch.from_numpy(y[b]).to(dev)
                    with autocast():
                        lo = n(xb)
                    loss = Fn.cross_entropy(lo.float(), yb)
                    o.zero_grad()
                    loss.backward()
                    o.step()

    def tmean(k):
        """Mean ensemble softmax over all of tracklet k's crops."""
        return probs(idx_of_trk(k, train=False)).mean(0).mean(0)
    if not segs:
        log('no global segment')
        return dict(fail=True)
    q = [min(L[list(c_)]) for _, _, c_ in segs]
    order = np.argsort(q)[::-1]
    s0 = segs[order[0]]
    lab = {k: c_ for c_, k in enumerate(s0[2])}
    hist = [dict(step=0, seg=[int(s0[0]), int(s0[1])], q=int(q[order[0]]), n_lab=len(lab))]
    tt = time.time()
    train(lab, epochs * 2)
    best_u = uniqueness()
    hist[-1].update(u=best_u, t=time.time() - tt)
    log(f'init {hist[-1]}')
    rejected, hard, used = set(), set(), {int(order[0])}
    prev_u = best_u
    for step in range(1, max_steps + 1):
        cand = [int(i) for i in order if int(i) not in used and int(i) not in rejected and any(k not in lab for k in segs[i][2])]
        if not cand:
            break
        cand = cand[:4 * Bc]
        lu = []
        for i in cand:
            x0, x1, _ = segs[i]
            fr = np.linspace(x0, x1, min(30, x1 - x0 + 1)).astype(int)
            flat = np.concatenate([np.flatnonzero(tid[:, t] >= 0) * T + t for t in fr])
            am = probs(flat).mean(0).argmax(-1).reshape(len(fr), N)
            lu.append(np.mean([len(set(r)) == N for r in am]))
        cand = [c_ for _, c_ in sorted(zip(lu, cand), key=lambda z_: z_[0])]
        added = {}
        acc = []
        for i in cand:
            ks = segs[i][2]
            new = [k for k in ks if k not in lab and k not in added]
            if any(len(idx_of_trk(k)) <= 5 for k in new):
                rejected.add(i)
                hard.add(i)
                continue
            mp = {k: tmean(k) for k in ks}
            am = {k: int(mp[k].argmax()) for k in ks}
            mx = {k: float(mp[k].max()) for k in ks}
            ids = set(am.values())
            if len(ids) == N - 1 and min(mx.values()) > pure * addf:
                miss = (set(range(N)) - ids).pop()
                seen = {}
                for k in ks:
                    if am[k] in seen:
                        o_ = seen[am[k]]
                        loser = k if mx[k] < mx[o_] else o_
                        am[loser] = miss
                        break
                    seen[am[k]] = k
                ids = set(am.values())
            okc = len(ids) == N and min(mx.values()) > pure * addf
            okc = okc and all(am[k] == lab.get(k, am[k]) and am[k] == added.get(k, am[k]) for k in ks)
            if okc:
                for k in new:
                    added[k] = am[k]
                used.add(i)
                acc.append(i)
            else:
                rejected.add(i)
            if len(acc) >= Bc:
                break
        if not added:
            hist.append(dict(step=step, acc=0))
            log(f'step {step} nothing accepted')
            break
        snap = [{k_: v.clone() for k_, v in n.state_dict().items()} for n in nets]
        osnap = [o.state_dict() for o in opts]
        lab2 = dict(lab)
        lab2.update(added)
        tt = time.time()
        train(lab2, epochs)
        u = uniqueness()
        keep = u >= 0.97 * best_u
        if keep:
            lab = lab2
            best_u = max(best_u, u)
            rejected = set(hard)
        else:
            for n, sn in zip(nets, snap):
                n.load_state_dict(sn)
            for o, so in zip(opts, osnap):
                o.load_state_dict(so)
            for i in acc:
                used.discard(i)
                rejected.add(i)
        frac = sum(len(idx_of_trk(k, False)) for k in lab) / max(1, sum(len(idx_of_trk(k, False)) for k in range(len(info))))
        hist.append(dict(step=step, acc=len(acc), new=len(added), u=u, keep=bool(keep), best_u=best_u, frac_lab=frac, t=time.time() - tt))
        log(f'step {hist[-1]}')
        if plateau >= 0 and keep and best_u >= good_u and (u - prev_u) < plateau:
            log('plateau stop')
            break
        prev_u = u if keep else prev_u
    tt = time.time()
    train(lab, epochs)
    u_final = uniqueness()
    hist.append(dict(step='final', u=u_final, t=time.time() - tt))
    log(f'final {hist[-1]}')
    t_train = time.time() - t00
    allidx = np.flatnonzero(ok.reshape(-1))
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    ti = time.time()
    P = probs(allidx)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t_inf = time.time() - ti
    full = np.full((K, S * T, N), np.nan, np.float16)
    full[:, allidx] = P.astype(np.float16)
    member_u = [uniqueness([m]) for m in range(K)]
    labarr = np.full(len(info), -1)
    for k, c_ in lab.items():
        labarr[k] = c_
    return dict(probs=full.reshape(K, S, T, N), tid=tid, info=np.array(info), lab=labarr, mind=mind, bl=bl, N=N, gid=gid,
                log=json.dumps(hist), uniq=u_final, member_uniq=np.array(member_u), t_train=t_train, t_inf=t_inf,
                n_inf=len(allidx), T=T, K=K)


def save_pixid(path, r):
    """Write a `train_pixid` result as npz (``feat``/``half`` kept empty for the scratch readers' layout)."""
    if r.get('fail'):
        np.savez(path, fail=True)
        return
    np.savez(path, feat=np.zeros((0, 100), np.float32), half=-1, **r)
