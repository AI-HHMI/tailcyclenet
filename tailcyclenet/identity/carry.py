"""The tracktail carry witness: an identity point set carried window to window, recorded per window.

Runs INSIDE the pose loop on the pose model's own backbone tokens (no second encode): per 2D window it
seeds each isolated slot's identity with points (the slot's pose + body-interior combinations), decodes
them through the next window with the `tracktail-static` scene tail + point decoder, and carries them to
the next window start. An identity is therefore a continuous point track through contacts, independent
of the pose rows' own association -- the cross-gap witness the identity gate referees the pixel column
with (`identity.witness`). Records are written beside the prediction session (`tt.pkl`).

Rules (frozen with the identity stack, report 76 'TrackGraph v7' / PREREG8): a slot isolated by
>= 0.25 BL and claimed (>= 0.7 hull share, runner-up < 0.2) by exactly one identity re-seeds it; an
unclaimed isolated slot births a new identity; a slot claimed >= 0.5 by two identities kills both
(the witness never guesses which duplicate was right); an identity no slot holds >= 20% of for 4
windows is an orphan and dies. Only slots with another slot within 1 BL somewhere in the window are
decoded (gate 1.0); other points ride their uniquely-claiming slot's centroid displacement, or are
held when no slot claims them uniquely. Each point is decoded once, in its most central crop.

Memory: the decode over slots runs in chunks of `slot_chunk` slots, every chunk padded to the same
group-wide query count, so each slot sees exactly the inputs of the unchunked call (tracktail mixes
queries only within a batch row). `slot_chunk` >= the number of decoded slots IS the unchunked call.
`sample_patches` is swapped for a memory-exact equivalent during the decode only
(`fast_sample_patches`): upstream gathers a full frame per query, which is what runs out of memory at
60-100 animals.
"""
from __future__ import annotations

import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TT_REPO = 'ai-hhmi/tracktail-static'
TT_REVISION = '4b28bc1cc611e80c38861278ef94e729314f0a2d'
"""The frozen witness head: tracktail-static at HF main 2026-10-12."""
HEAD = 'S'
"""The record key the identity stack reads (one head)."""
N_INT = 16
"""Body-interior seed points per identity."""
GATE_BL = 1.0
"""Decode only slots with another slot within this many BL somewhere in the window."""
REFRESH_SEP = 0.25
"""A slot is isolated at >= this many BL from every other slot."""
REFRESH_FRAC = 0.7
"""Hull share an identity needs to re-seed from an isolated slot."""
TAIL = ('tail',)


def fast_sample_patches(images, centers, query_times, patch_size):
    """Memory-exact `tracktail.model.encoder_decoder.sample_patches`.

    Upstream gathers a full (C,H,W) frame per query before sampling a PxP patch; here one
    `grid_sample` per distinct query time samples all patch grids at once. Bilinear sampling is
    computed per output point, so the patches are identical.
    """
    B, T, C, H, W = images.shape
    Z = centers.shape[1]
    P = patch_size
    lin = torch.arange(P, dtype=centers.dtype, device=centers.device)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')
    offsets = torch.stack([grid_x - (P - 1) / 2.0, grid_y - (P - 1) / 2.0], dim=-1)
    px = centers[:, :, None, None, :] + offsets[None, None, :, :, :]
    scales = torch.tensor([W, H], dtype=images.dtype, device=images.device)
    grid = (2.0 * px + 1.0) / scales - 1.0
    qt = query_times.long()
    out = None
    for t in torch.unique(qt).tolist():
        res = F.grid_sample(images[:, t], grid.reshape(B, Z * P, P, 2), mode='bilinear',
                            padding_mode='zeros', align_corners=False)
        res = res.reshape(B, C, Z, P, P).permute(0, 2, 1, 3, 4)
        m = (qt == t)[:, :, None, None, None]
        out = torch.where(m, res, out if out is not None else torch.zeros_like(res))
    return out.contiguous()


@contextmanager
def _fast_patches():
    """Swap tracktail's module-level `sample_patches` for the decode only (the pose model's query
    encoder binds its own reference at import and is never affected)."""
    import tracktail.model.encoder_decoder as ED
    orig = ED.sample_patches
    ED.sample_patches = fast_sample_patches
    try:
        yield
    finally:
        ED.sample_patches = orig


def load_head(device):
    """The witness head: `tracktail-static` at the pinned revision with its video encoder removed
    (the backbone tokens are the pose model's)."""
    from tracktail.model.tracker_encoder import TrackerEncoder

    from ..checkpoints import resolve_hf_checkpoint
    ck = torch.load(resolve_hf_checkpoint(TT_REPO, TT_REVISION), map_location='cpu',
                    weights_only=False)
    m = TrackerEncoder(**{**ck['model_config'], 'video_encoder_pretrained': False})
    m.load_state_dict(ck['model_state'], strict=True)
    m.scene_encoder.encoder = None
    return m.to(device).eval()


def seed_weights(P0, names, rng, n_int):
    """Seed points as keypoint weights: every finite keypoint + `n_int` convex combinations of 3
    random finite non-tail keypoints. -> W (n, K)."""
    K = P0.shape[0]
    f = np.isfinite(P0).all(-1)
    body = [k for k in range(K) if f[k] and not any(s in names[k].lower() for s in TAIL)]
    if len(body) < 3:
        body = [k for k in range(K) if f[k]]
    W = [np.eye(K)[k] for k in range(K) if f[k]]
    m = min(3, len(body))
    for _ in range(n_int if m >= 2 else 0):
        tri = rng.choice(body, m, replace=False)
        w = rng.dirichlet([1] * m)
        v = np.zeros(K)
        v[tri] = w
        W.append(v)
    return np.array(W, np.float32)


def apply_weights(W, P):
    """W (n,K), P (T,K,R) -> (T,n,R); NaN where a used keypoint is missing."""
    out = np.einsum('nk,tkr->tnr', W, np.nan_to_num(P))
    bad = (W[None] > 0) & ~np.isfinite(P).all(-1)[:, None, :]
    out[bad.any(-1)] = np.nan
    return out


def _seed_pts(P0, names, rng, Sz):
    """P0 (K,2) crop px -> seed points (crop px). K = 1: the point + a disc of sd 4% of the crop."""
    fin = np.isfinite(P0).all(-1)
    if fin.sum() == 0:
        return np.zeros((0, 2), np.float32)
    if P0.shape[0] == 1:
        off = rng.normal(0, 0.04 * Sz, (N_INT, 2))
        q = np.concatenate([P0[:1], P0[:1] + off])
    else:
        Wb = seed_weights(np.where(fin[:, None], P0, np.nan), names, rng, N_INT)
        q = apply_weights(Wb, P0[None])[0]
    ok = np.isfinite(q).all(-1) & (q >= 0).all(-1) & (q < Sz).all(-1)
    return q[ok].astype(np.float32)


def _hull_frac(pts, ref, tol):
    """Per point: within `tol` of a finite `ref` keypoint, or inside their convex hull."""
    ref = ref[np.isfinite(ref).all(-1)]
    if len(pts) == 0 or len(ref) == 0:
        return np.zeros(len(pts), bool)
    near = np.min(np.linalg.norm(pts[:, None] - ref[None], axis=-1), 1) <= tol
    if len(ref) >= 3:
        try:
            from matplotlib.path import Path as MPath
            from scipy.spatial import ConvexHull
            h = ConvexHull(ref)
            near |= MPath(ref[h.vertices]).contains_points(pts)
        except Exception:
            pass
    return near


class _Host:
    """What `project_scene` needs: the witness head's scene tail + the pose model's autocast."""

    def __init__(self, head, model):
        """Borrow the head's scene tail and the model's autocast scope."""
        self.scene_encoder = head.scene_encoder
        self._scene_autocast = model._scene_autocast


def _track(head, model, views, cgroup, q, qt, enc):
    """Decode query points q (B,N,2 crop px, at query time qt) through one window."""
    from ..pose.scene import SceneEncodingMixin
    feats = SceneEncodingMixin.project_scene(_Host(head, model), enc)
    v = [x.float().div(255) if x.dtype == torch.uint8 else x for x in views]
    px = max(int(c['size'].max()) for c in cgroup)
    with _fast_patches():
        return head(v, q, cgroup, query_times=qt, input_size=px, scene_features=feats)


class CarryWitness:
    """The in-loop witness. `begin_group` / `observe` (one call per 2D slot forward, in window
    order) / `end_group`; `save` writes the records. The pose path is untouched: the witness only
    reads what the loop already computed (the backbone tokens via `tap`, views, coords)."""

    def __init__(self, model, device, slot_chunk=8):
        """Attach to `model` (sets its `backbone_tap`); the head loads at the first decode."""
        self.model, self.device, self.slot_chunk = model, device, int(slot_chunk)
        self.head = None
        self.rec, self.events, self.stats = [], {}, dict(windows=0, dec_calls=0, slot_windows=0,
                                                          slot_windows_dec=0)
        self.state, self.nid, self.ev = {}, 0, []
        self.gid, self.names, self.ws, self.items, self._last = None, [], None, {}, None
        model.backbone_tap = self.tap

    def tap(self, encoding):
        """`encode_scene_backbone`'s hook: keep the latest raw backbone encoding."""
        self._last = encoding

    def begin_group(self, gid, names):
        """Fresh identities for a new group (the window counter, and so the seed stream, runs on)."""
        self.gid, self.names, self.ws, self.items = gid, list(names), None, {}
        self.state, self.nid, self.ev = {}, 0, []

    def end_group(self):
        """Decode the group's last window and keep its events."""
        self._flush()
        self.ws = None
        self.events[self.gid] = list(self.ev)

    def observe(self, a, frames, box, scale, views, cgroup, psrc):
        """One slot's 2D forward: its crop box origin / scale, the views it saw, its source-px pose."""
        ws = int(frames[0])
        if self.ws is not None and ws != self.ws:
            self._flush()
            self._advance(ws)
        self.ws = ws
        self.items[int(a)] = dict(frames=np.asarray(frames).copy(), box=np.asarray(box, np.float32)[:2].copy(),
                                  scale=np.asarray(scale, np.float32).reshape(-1)[:2].copy(),
                                  enc=self._last, views=list(views), cg=cgroup, psrc=psrc)
        self._last = None

    def save(self, path):
        """Write `{rec: {S: [window records]}, heads, events: {gid: [...]}, stats}` as a pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(dict(rec={HEAD: self.rec}, heads=[HEAD], events=dict(self.events),
                             stats=dict(self.stats)), f)

    # ---- the per-window step
    def _flush(self):
        """Step the window just completed (loads the head once, at the first window)."""
        items, self.items = self.items, {}
        if not items:
            return
        self.stats['windows'] += 1
        if self.head is None:
            self.head = load_head(self.device)
        self._step(items)

    def _step(self, items):
        """One window: (1) claims, orphans, duplicate kills, refresh / birth; (2) decode each alive point
        once, in its most central gated crop, the rest riding or held; (3) record the window and keep
        each identity's track for the carry. `bl` = the median slot extent at the window start."""
        state, names = self.state, self.names
        rng = np.random.default_rng(self.stats['windows'])
        slots = sorted(items)
        ws = int(items[slots[0]]['frames'][0])
        nfr = len(items[slots[0]]['frames'])
        P0 = {a: items[a]['psrc'][0] for a in slots}
        ext = [np.ptp(p[np.isfinite(p).all(-1)], 0).max() for p in P0.values()
               if np.isfinite(p).all(-1).sum() >= 2]
        Szs = {a: float(items[a]['cg'][0]['size'].max()) for a in slots}
        bl = (float(np.median(ext)) if ext else
              float(np.median([Szs[a] / items[a]['scale'][0] for a in slots])) / 1.5)

        def slot_sep(a):
            """Nearest other slot at the window start, in BL."""
            d = np.inf
            A = P0[a][np.isfinite(P0[a]).all(-1)]
            for b in slots:
                if b == a:
                    continue
                B = P0[b][np.isfinite(P0[b]).all(-1)]
                if len(A) and len(B):
                    d = min(d, float(np.min(np.linalg.norm(A[:, None] - B[None], axis=-1))) / bl)
            return d
        claims = {a: {} for a in slots}
        for k, s in state.items():
            v = s['pts'][s['alive']]
            if not len(v):
                continue
            cv_ = v.mean(0)
            for a in slots:
                pa = P0[a][np.isfinite(P0[a]).all(-1)]
                if not len(pa) or np.min(np.linalg.norm(pa - cv_, axis=-1)) > 2.5 * bl:
                    continue
                f = _hull_frac(v, P0[a], 0.15 * bl).mean()
                if f > 0:
                    claims[a][k] = f
        for k in list(state):
            if max([claims[a].get(k, 0) for a in slots] + [0]) < 0.2:
                state[k]['orph'] = state[k].get('orph', 0) + 1
                if state[k]['orph'] > 3:
                    state.pop(k)
                    self.ev.append(('orphan', int(k), ws))
                    for b in slots:
                        claims[b].pop(k, None)
            else:
                state[k]['orph'] = 0
        for a in slots:
            dup = [k for k, f in claims[a].items() if f >= 0.5]
            if len(dup) >= 2 and slot_sep(a) >= REFRESH_SEP:
                for k in dup:
                    state.pop(k, None)
                    self.ev.append(('kill', int(k), ws))
                    for b in slots:
                        claims[b].pop(k, None)
        for a in slots:
            it = items[a]
            Sz = Szs[a]
            top = sorted(claims[a].items(), key=lambda x: -x[1])
            isolated = slot_sep(a) >= REFRESH_SEP
            if top and isolated and top[0][1] >= REFRESH_FRAC and (len(top) == 1 or top[1][1] < 0.2):
                k = top[0][0]
                if not any(claims[b].get(k, 0) > 0.2 for b in slots if b != a):
                    q = _seed_pts((P0[a] - it['box']) * it['scale'], names, rng, Sz)
                    if len(q) >= 2:
                        state[k] = dict(pts=q / it['scale'] + it['box'], alive=np.ones(len(q), bool),
                                        born=state[k]['born'], orph=0)
            elif not top and (isolated or not state):
                q = _seed_pts((P0[a] - it['box']) * it['scale'], names, rng, Sz)
                if len(q) >= 2:
                    k = self.nid
                    self.nid += 1
                    state[k] = dict(pts=q / it['scale'] + it['box'], alive=np.ones(len(q), bool), born=ws)
                    self.ev.append(('birth', int(k), ws))
        allk = [(k, n) for k, s in state.items() for n in np.flatnonzero(s['alive'])]

        def near(a):
            """Another slot within GATE_BL somewhere in the window."""
            if GATE_BL <= 0:
                return True
            A = items[a]['psrc']
            for b in slots:
                if b == a:
                    continue
                d = np.linalg.norm(A[:, :, None] - items[b]['psrc'][:, None], axis=-1)
                if np.isfinite(d).any() and np.nanmin(d) / bl < GATE_BL:
                    return True
            return False
        dec = [a for a in slots if items[a]['enc'] is not None and near(a)]
        jobs = []
        if allk:
            Q = np.array([state[k]['pts'][n] for k, n in allk])
            cen_best = np.full(len(Q), -np.inf)
            arg = np.full(len(Q), -1)
            QC = {}
            for a in dec:
                Sz = Szs[a]
                qc = (Q - items[a]['box']) * items[a]['scale']
                QC[a] = qc
                inside = np.isfinite(qc).all(-1) & (qc >= 0).all(-1) & (qc < Sz).all(-1)
                cen = np.where(inside, np.minimum(qc, Sz - qc).min(-1), -np.inf)
                upd = cen > cen_best
                cen_best[upd] = cen[upd]
                arg[upd] = a
            for a in dec:
                ix = np.flatnonzero(arg == a)
                if len(ix):
                    jobs.append((a, ix, QC[a]))
        outs = self._decode(items, jobs) if jobs else {}
        self.stats['slot_windows'] += len(slots)
        self.stats['slot_windows_dec'] += len(jobs)
        best = {}
        for a, ix, qc in jobs:
            it = items[a]
            Sz = Szs[a]
            tr, vis = outs[a]
            cen = np.minimum(qc[ix], Sz - qc[ix]).min(-1)
            for m, i in enumerate(ix):
                key = allk[i]
                if key not in best or cen[m] > best[key][0]:
                    best[key] = (cen[m], tr[:, m] / it['scale'] + it['box'], vis[:, m])
        for k, s in state.items():
            miss = [n for n in np.flatnonzero(s['alive']) if (k, n) not in best]
            if not miss:
                continue
            own = [a for a in slots if claims[a].get(k, 0) >= 0.5]
            hold = len(own) != 1
            if not hold:
                cen_t = np.nanmean(items[own[0]]['psrc'], 1)
                hold = not np.isfinite(cen_t).all()
            if hold:
                for n in miss:
                    best[(k, n)] = (-2.0, np.repeat(s['pts'][n][None], nfr, 0).astype(np.float32),
                                    np.zeros(nfr, np.float32))
                continue
            disp = cen_t - cen_t[0]
            for n in miss:
                best[(k, n)] = (-1.0, (s['pts'][n][None] + disp).astype(np.float32),
                                np.ones(len(disp), np.float32))
        rec = dict(gid=self.gid, ws=ws, nfr=nfr, ids={})
        for k, s in state.items():
            N = len(s['pts'])
            trk = np.full((nfr, N, 2), np.nan, np.float32)
            vv = np.zeros((nfr, N), np.float16)
            for n in range(N):
                b = best.get((k, n))
                if b is not None:
                    trk[:, n] = b[1]
                    vv[:, n] = b[2]
            rec['ids'][k] = dict(tr=trk, vis=vv)
            s['_trk'] = trk
        rec['psrc'] = {a: items[a]['psrc'].astype(np.float32) for a in slots}
        self.rec.append(rec)

    @torch.no_grad()
    def _decode(self, items, jobs):
        """Batched decode of every job, in chunks of `slot_chunk` slots padded to one query count."""
        from ..pose.scene import SceneEncoding
        dev, head, model = self.device, self.head, self.model
        outs, groups = {}, {}
        for jb in jobs:
            groups.setdefault(tuple(items[jb[0]]['views'][0].shape), []).append(jb)
        for gjobs in groups.values():
            if len(gjobs) > 1:
                nmax = max(len(ix) for _, ix, _ in gjobs)
                CK = self.slot_chunk if self.slot_chunk > 0 else len(gjobs)
                for c0 in range(0, len(gjobs), CK):
                    cjobs = gjobs[c0:c0 + CK]
                    B_ = len(cjobs)
                    q = np.zeros((B_, nmax, 2), np.float32)
                    for bi, (a, ix, qc) in enumerate(cjobs):
                        q[bi, :len(ix)] = qc[ix]
                        q[bi, len(ix):] = qc[ix][:1]
                    e0 = [items[a]['enc'] for a, _, _ in cjobs]
                    enc = SceneEncoding([torch.cat([e.views[0] for e in e0], 0)],
                                        [torch.cat([e.raw_features[0] for e in e0], 0)], 1, B_, False)
                    views = [torch.cat([items[a]['views'][0] for a, _, _ in cjobs], 0)]
                    o = _track(head, model, views, items[cjobs[0][0]]['cg'], torch.as_tensor(q, device=dev),
                               torch.zeros((B_, nmax), dtype=torch.int64, device=dev), enc)
                    tr_all = o['2d_pred'][0].float().cpu().numpy()
                    vis_all = torch.sigmoid(o['vis_pred'][..., 0]).float().cpu().numpy()
                    for bi, (a, ix, qc) in enumerate(cjobs):
                        outs[a] = (tr_all[bi, :, :len(ix)], vis_all[bi, :, :len(ix)])
                    del o, enc, views
                    self.stats['dec_calls'] += 1
            else:
                for a, ix, qc in gjobs:
                    it = items[a]
                    o = _track(head, model, it['views'], it['cg'],
                               torch.as_tensor(qc[ix], dtype=torch.float32, device=dev)[None],
                               torch.zeros((1, len(ix)), dtype=torch.int64, device=dev), it['enc'])
                    outs[a] = (o['2d_pred'][0, 0].float().cpu().numpy(),
                               torch.sigmoid(o['vis_pred'][0, ..., 0]).float().cpu().numpy())
                    self.stats['dec_calls'] += 1
        return outs

    def _advance(self, next_ws):
        """Move carried points to the next window start (index next_ws - ws of the last record)."""
        if not self.rec:
            return
        d = next_ws - self.rec[-1]['ws']
        for k, s in list(self.state.items()):
            trk = s.get('_trk')
            if trk is None:
                continue
            if 0 < d < trk.shape[0]:
                p = trk[d]
                ok = np.isfinite(p).all(-1)
                s['pts'] = np.where(ok[:, None], p, s['pts'])
                s['alive'] = s['alive'] & ok
            s['_trk'] = None
            if not s['alive'].any():
                del self.state[k]
