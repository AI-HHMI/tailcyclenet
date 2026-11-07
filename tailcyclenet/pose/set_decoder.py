"""E2, the anchored set decoder of the v7 2D pose model (MX0e; report 76 crowd-moe).

One network, two decoder experts on ONE scene encode: E1 is the host's own box-FiLM query encoder +
decoder (the stock pose model, unchanged), E2 is this module -- a DETR-style set decoder with `M`
animal queries, the first `NA` of them ANCHORED on a box (query 0 = the slot's own box, then the
nearest other animals'), GroupPose self-attention across queries, a per-query presence head, and a
known assignment for the anchored queries in its loss. E2 reads DETACHED scene tokens, so the shared
encoder trains on E1's loss alone (the MX0e arm; joint and E1-detached arms were measured worse).

The B1p recipe is fixed here, not configured (one architecture): M 8, NA 4, inst_std 0.5,
presence weight 0.2, Hungarian costs L1 1.0 / presence 0.2, box-only anchors (no ellipse pool).

E2 owns COPIES of the host's query encoder and tracktail `Decoder`. While an E2 forward is in flight
(`self._layout` set) three hooks on E2's OWN modules do the set-decoder work: a pre-hook on its
decoder adds the per-query instance + anchor-box embedding to the query tokens, a hook on each of
its cross-attentions adds the GroupPose attention, and a hook on the decoder keeps the latent for the
presence head. Nothing in the library or in E1 is touched.

`[model].set_decoder = "only"` is the v7 3D set decoder (PBM4, report 76 crowd-detrq / detr3d-multi):
the SAME module with no E1 beside it -- the host's own query encoder + decoder ARE E2's (shared, not
copied), the encoder trains on E2's loss, `M_ONLY` = 4 queries all anchored (known assignment,
no free query), per-camera anchors (the crop rule's square box: each side padded by
`[data].crop_pad_frac` x the longer side), no ellipse pool. Its deploy is `--anchor none` (no
priors) and query 0 is always the slot's row in 3D (world coords are never matched to pixel boxes).
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# M animal queries, the first NA anchored (query i <- the i-th box: own first, then nearest);
# INST_STD the instance-embedding init; W_PRES the presence BCE weight; C_PRES / C_L1 the Hungarian
# costs (L1 in crop px / HUNGARIAN_NORM); MF the anchor-box Fourier max frequency
M = 8
NA = 4
# the set-only (PBM4) recipe: 4 queries, every one anchored
M_ONLY = 4
INST_STD = 0.5
W_PRES = 0.2
C_PRES = 0.2
C_L1 = 1.0
MF = 10
HUNGARIAN_NORM = 256.0


class GroupAttn(nn.Module):
    """GroupPose self-attention: within-query (over K) + across-query (over M) per keypoint.
    Zero-initialised outputs, so a fresh module is a no-op on the warm-started decoder."""

    def __init__(self, D, heads):
        """Two zero-output attentions at width D."""
        super().__init__()
        self.w = nn.MultiheadAttention(D, heads, batch_first=True)
        self.a = nn.MultiheadAttention(D, heads, batch_first=True)
        for m in (self.w, self.a):
            nn.init.zeros_(m.out_proj.weight)
            nn.init.zeros_(m.out_proj.bias)

    def forward(self, xn, T, Mq, K):
        """xn (CB, T*Mq*K, D) the normed cross-attention query tokens -> their additive update."""
        CB, L, D = xn.shape
        x = xn.reshape(CB, T, Mq, K, D)
        xw = x.reshape(CB * T * Mq, K, D)
        w = self.w(xw, xw, xw, need_weights=False)[0].reshape(CB, T, Mq, K, D)
        xa = x.permute(0, 1, 3, 2, 4).reshape(CB * T * K, Mq, D)
        a = self.a(xa, xa, xa, need_weights=False)[0].reshape(CB, T, K, Mq, D).permute(0, 1, 3, 2, 4)
        return (w + a).reshape(CB, L, D)


def _fourier(x):
    """x concatenated with its Fourier features (min 0, max MF) on the last axis."""
    from tracktail.model.utils import get_fourier_encoding
    sh = x.shape
    fe = get_fourier_encoding(x.reshape(1, 1, -1, sh[-1]), min_freq=0, max_freq=MF)
    return torch.cat([x, fe.reshape(*sh[:-1], fe.shape[-1])], -1)


class SetDecoder(nn.Module):
    """E2. Built from the host AFTER the host's own query encoder exists (its copies share the
    architecture; the weights are loaded or warm-filled later, `fill_from_host`)."""

    def __init__(self, host, shared=False):
        """Copy the host's query encoder + decoder (or, `shared`, use the host's own: the set-only
        model -- plain attributes, not submodules, so the host owns and saves them), add E2's own
        parts, hook E2's decoder."""
        super().__init__()
        self.shared = bool(shared)
        self.M = M_ONLY if self.shared else M
        self.NA = NA
        if self.shared:
            object.__setattr__(self, 'query_encoder', host.query_encoder)
            object.__setattr__(self, 'decoder', host.decoder)
        else:
            self.query_encoder = copy.deepcopy(host.query_encoder)
            self.decoder = copy.deepcopy(host.decoder)
        D = self.decoder.embed_dim
        self.inst = nn.Parameter(torch.randn(self.M, D) * INST_STD)
        self.groups = nn.ModuleList([GroupAttn(D, 8) for _ in range(self.decoder.num_layers)])
        self.pres = nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1))
        self.abox = nn.Sequential(nn.Linear(8 * MF + 4, D), nn.GELU(), nn.Linear(D, D))
        self._layout = None
        self._latent = None
        self.decoder.register_forward_pre_hook(self._add_query_embedding, with_kwargs=True)
        self.decoder.register_forward_hook(self._keep_latent)
        for i, ca in enumerate(self.decoder.cross_attns):
            ca.register_forward_hook(self._group_hook(i))

    def fill_from_host(self, host):
        """The warm-start init: E2's query encoder / decoder start as the warm-started host's
        (nothing to do when they ARE the host's)."""
        if self.shared:
            return
        self.query_encoder.load_state_dict(host.query_encoder.state_dict(), strict=True)
        self.decoder.load_state_dict(host.decoder.state_dict(), strict=True)

    # ---- hooks (inert unless an E2 forward set `_layout`) ----
    def _add_query_embedding(self, mod, args, kwargs):
        """Decoder pre-hook: query tokens += instance + anchor-box embedding of their query."""
        lay = self._layout
        if lay is None:
            return None
        scene_features, query_embeds, *rest = args
        T, K, Mt = lay['T'], lay['K'], self.M
        B, T_, Kt, C, D = query_embeds.shape
        assert Kt == Mt * K and T_ == T, (query_embeds.shape, T, K)
        emb = self.inst[None, None, :, None]
        ab = lay['anch']
        if ab is not None:
            emb = emb + self._anchor_emb(ab, lay['sizes'])
        emb = emb.expand(B, T, Mt, C, D).repeat_interleave(K, 2)
        return (scene_features, query_embeds + emb.to(query_embeds.dtype), *rest), kwargs

    def _keep_latent(self, mod, args, output):
        """Decoder hook: keep the per-token latent the presence head reads."""
        if self._layout is not None:
            self._latent = output['latent']

    def _group_hook(self, i):
        """The hook for cross-attention layer i: its output += GroupPose attention."""
        def hook(mod, args, output):
            """Cross-attention hook (inert outside an E2 forward)."""
            lay = self._layout
            if lay is None:
                return None
            return output + self.groups[i](args[0], lay['T'], self.M, lay['K'])
        return hook

    def _anchor_emb(self, ab, sizes):
        """ab (B,M,T,C,4) crop px -> (B,T,M,C,D): box-Fourier MLP, zero where no anchor."""
        ok = torch.isfinite(ab).all(-1)
        bn = torch.nan_to_num(ab / torch.cat([sizes, sizes], -1)[None, None, None] * 2 - 1)
        e = self.abox(_fourier(bn))
        return (e * ok[..., None].to(e.dtype)).permute(0, 2, 1, 3, 4)


def _pad_anchors(a, Mq=M):
    """(B,na,T,[C,]4) -> (B,Mq,T,C,4) float, NaN beyond the first NA."""
    if a.dim() == 4:
        a = a[:, :, :, None]
    B, na = a.shape[:2]
    out = torch.full((B, Mq) + tuple(a.shape[2:]), float('nan'), device=a.device, dtype=torch.float32)
    out[:, :min(na, NA)] = a[:, :NA].float()
    return out


def _qboxes(coords):
    """(...,K,2) -> (...,4) keypoint extent, NaN with fewer than 2 finite points."""
    f = torch.isfinite(coords).all(-1)
    x, y = coords[..., 0], coords[..., 1]
    b = torch.stack([torch.where(f, x, torch.inf).amin(-1), torch.where(f, y, torch.inf).amin(-1),
                     torch.where(f, x, -torch.inf).amax(-1), torch.where(f, y, -torch.inf).amax(-1)], -1)
    b[f.sum(-1) < 2] = torch.nan
    return b


def _iou(a, b):
    """Elementwise IoU of xyxy boxes."""
    ix = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    iy = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    ar = lambda z: (z[..., 2] - z[..., 0]).clamp(min=0) * (z[..., 3] - z[..., 1]).clamp(min=0)  # noqa: E731
    inter = ix * iy
    return inter / (ar(a) + ar(b) - inter).clamp(min=1e-6)


def select_query(coords, pres, size, slot_boxes=None, slot=None, anch=None):
    """Which of the M queries is this slot's: query 0 when its own anchor exists anywhere in the
    window; else the one a box Hungarian (query skeleton box vs every slot's box) gives this slot;
    else the most central, present one. coords (T,M,K,R), pres (T,M) logits, anch (M,T,C,4).
    3D: always query 0 -- the slot's own anchored query; world coords are never compared with
    pixel boxes (detr3d-diag)."""
    T, Mq = coords.shape[:2]
    p = torch.sigmoid(pres).mean(0)
    if anch is not None and torch.isfinite(anch[0]).all(-1).any():
        return 0
    if coords.shape[-1] == 3:
        return 0
    qb = _qboxes(coords)
    qc = (qb[..., :2] + qb[..., 2:]) / 2
    diag = float(torch.as_tensor(size, dtype=torch.float32).norm())
    if slot_boxes is not None:
        sb = torch.as_tensor(slot_boxes, dtype=torch.float32, device=coords.device)
        fin = torch.isfinite(sb).all(-1)
        if fin[slot].any():
            slots = [s for s in range(sb.shape[0]) if fin[s].any()]
            C = np.zeros((Mq, len(slots)))
            for j, s in enumerate(slots):
                tt = fin[s]
                iou = _iou(qb[tt], sb[s, tt][:, None])
                sc = (sb[s, tt, :2] + sb[s, tt, 2:]) / 2
                dc = (qc[tt] - sc[:, None]).norm(dim=-1) / diag
                c = (1 - torch.nan_to_num(iou, nan=0.0)) + torch.nan_to_num(dc, nan=1.0)
                C[:, j] = c.mean(0).cpu().numpy() - 0.1 * p.cpu().numpy()
            r, cidx = linear_sum_assignment(C)
            hit = [i for i, j in zip(r, cidx) if j == slots.index(slot)]
            if hit:
                return int(hit[0])
    ctr = torch.as_tensor(size, dtype=torch.float32, device=coords.device) / 2
    d = torch.nan_to_num((qc - ctr).norm(dim=-1), nan=1e6).mean(0) / diag
    return int(torch.argmin(d - 0.5 * p))


def slice_keypoints(obj, Ktot, idx):
    """Every tensor's first axis of length `Ktot` (the M*K token axis) indexed by `idx`."""
    if torch.is_tensor(obj):
        for ax, s in enumerate(obj.shape):
            if s == Ktot:
                return obj.index_select(ax, idx.to(obj.device))
        return obj
    if isinstance(obj, dict):
        return {k: slice_keypoints(v, Ktot, idx) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(slice_keypoints(v, Ktot, idx) for v in obj)
    return obj


def mix_outputs(o2, o1, use2, T, K):
    """Per frame: E2's tensor where `use2[t]`, else E1's, on every output with adjacent (T, K)
    axes; anything else is E2's."""
    out = dict(o2)
    for k, v2 in o2.items():
        v1 = o1.get(k)
        if not (torch.is_tensor(v2) and torch.is_tensor(v1)) or v1.shape != v2.shape:
            continue
        sh = list(v2.shape)
        ax = next((i for i in range(len(sh) - 1) if sh[i] == T and sh[i + 1] == K), None)
        if ax is None:
            continue
        ms = [1] * len(sh)
        ms[ax] = T
        out[k] = torch.where(use2.view(ms).to(v2.device), v2, v1)
    return out


def _detach(x):
    """Every tensor in a nested container, detached."""
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_detach(v) for v in x)
    return x


@contextmanager
def _as_expert(host, sd):
    """Run the host's forward machinery with E2's query encoder and decoder in place of E1's."""
    mods = host._modules
    old = mods['query_encoder'], mods['decoder']
    mods['query_encoder'], mods['decoder'] = sd.query_encoder, sd.decoder
    try:
        yield
    finally:
        mods['query_encoder'], mods['decoder'] = old


def set_forward(host, views, kpt_ids, camera_group, mode, *, anchors, anchor_prior=None,
                anchor_prompt_t=None, kpt_prior=None, prompt_time=None, input_size=None):
    """One E2 forward over all M queries -> (full output over M*K tokens, presence logits
    (B,T,M), anchors (B,M,T,C,4) or None).

    anchors (B,na,T,[C,]4) crop px; anchor_prior (B,na,K,2) crop px + anchor_prompt_t (B,na[,K]):
    a position prior per anchored query. At deploy `kpt_prior`/`prompt_time` (the slot's own carry)
    replace anchor 0's prior. No box prompt: E2's box is its anchor. 3D: anchors (B,na,T,C,4)
    per camera, priors in world units."""
    sd = host.set_decoder
    M, NA = sd.M, sd.NA
    B, K = kpt_ids.shape
    dev = views[0].device
    T = int(views[0].shape[1])
    R = 2 if mode == '2d' else 3
    anch = None if anchors is None else _pad_anchors(anchors.to(dev), M)
    Kt = M * K
    prior = torch.full((B, Kt, R), float('nan'), device=dev)
    pt = torch.zeros((B, Kt), dtype=torch.long, device=dev)
    if anch is not None:
        if anchor_prior is None:
            pp = torch.full((B, NA, K, R), float('nan'), device=dev)
            ptt = torch.zeros((B, NA, K), dtype=torch.long, device=dev)
        else:
            pp = anchor_prior.to(dev).float().clone()
            ptt = anchor_prompt_t.to(dev).long()
            ptt = (ptt[..., None].expand(B, ptt.shape[1], K) if ptt.dim() == 2 else ptt).clone()
        if kpt_prior is not None:
            kp0 = kpt_prior.to(dev).float()
            if torch.isfinite(kp0).any():
                pp[:, 0] = kp0
                if prompt_time is not None:
                    ptt[:, 0] = prompt_time.to(dev).long()
        for i in range(min(NA, pp.shape[1])):
            prior[:, i * K:(i + 1) * K] = pp[:, i]
            pt[:, i * K:(i + 1) * K] = ptt[:, i].clamp(0, T - 1)
    sd._layout = dict(T=T, K=K, anch=anch,
                      sizes=torch.stack([c['size'][:2].to(dev).float() for c in camera_group]))
    try:
        with _as_expert(host, sd):
            out = host._forward(views, kpt_ids.repeat(1, M), camera_group, mode, kpt_prior=prior,
                                prompt_time=pt, kpt_chunk=None, box_prompt=None,
                                input_size=input_size)
        lat = sd._latent
    finally:
        sd._layout = None
        sd._latent = None
    lat = lat.reshape(B, T, M, K, *lat.shape[3:]).mean(dim=(3, 4))
    pres = sd.pres(lat)[..., 0]
    return out, pres, anch


def select_and_slice(out, pres, anch, K, camera_group, slot_boxes=None, slot=None):
    """E2's full output reduced to the one query that is this slot's (batch item 0 decides)."""
    coords = out['coords_pred']
    T = coords.shape[1]
    M = pres.shape[-1]
    size = camera_group[0]['size'][:2].float().cpu()
    q = select_query(coords[0, :, :M * K].detach().reshape(T, M, K, -1), pres[0].detach(), size,
                     slot_boxes, slot, None if anch is None else anch[0])
    return slice_keypoints(out, M * K, torch.arange(q * K, (q + 1) * K)), q


def moe_forward(host, views, kpt_ids, camera_group, mode, set_inputs, *, input_size=None,
                kpt_prior=None, prompt_time=None, box_prompt=None, kpt_chunk=None,
                scene_features=None):
    """The two-expert forward. `set_inputs['route']`:
      'train' -- ONE scene encode, E1 (stock, its own prior/box) then E2 on the DETACHED scene
                 tokens; returns E1's output with E2's full output under 'set'.
      a (T,) bool mask -- deploy: frames where True take E2's (selected-query) rows, the rest E1's;
                 an all-False mask runs E1 only, all-True E2 only, a mixed one both on one encode.
    """
    from .model import share_scene
    if host.set_decoder.shared:
        return only_forward(host, views, kpt_ids, camera_group, mode, set_inputs,
                            input_size=input_size, kpt_prior=kpt_prior, prompt_time=prompt_time,
                            scene_features=scene_features)
    route = set_inputs['route']
    K = int(kpt_ids.shape[1])
    e1 = dict(kpt_prior=kpt_prior, prompt_time=prompt_time, box_prompt=box_prompt,
              input_size=input_size)
    if isinstance(route, str) and route == 'train':
        with share_scene(host, scene_features=scene_features):
            out1 = host._forward(views, kpt_ids, camera_group, mode, kpt_chunk=kpt_chunk, **e1)
            sh = host._shared_scene
            if 'f' in sh:
                sh['f'] = _detach(sh['f'])
            out2, pres, anch = set_forward(
                host, views, kpt_ids, camera_group, mode, anchors=set_inputs['anchors'],
                anchor_prior=set_inputs.get('anchor_prior'),
                anchor_prompt_t=set_inputs.get('anchor_prompt_t'), input_size=input_size)
        out1 = dict(out1)
        out1['set'] = dict(out=out2, pres=pres, anch=anch, K=K)
        return out1
    use2 = torch.as_tensor(np.asarray(route, dtype=bool))
    T = int(views[0].shape[1])
    if not bool(use2.any()):
        return host._forward(views, kpt_ids, camera_group, mode, kpt_chunk=kpt_chunk,
                             scene_features=scene_features, **e1)

    def e2():
        """E2 over all queries, reduced to this slot's query."""
        out2, pres, anch = set_forward(
            host, views, kpt_ids, camera_group, mode, anchors=set_inputs['anchors'],
            anchor_prior=set_inputs.get('anchor_prior'),
            anchor_prompt_t=set_inputs.get('anchor_prompt_t'),
            kpt_prior=kpt_prior, prompt_time=prompt_time, input_size=input_size)
        return select_and_slice(out2, pres, anch, K, camera_group,
                                set_inputs.get('slot_boxes'), set_inputs.get('slot'))[0]
    if bool(use2[:T].all()):
        if scene_features is None:
            return e2()
        with share_scene(host, scene_features=scene_features):
            return e2()
    with share_scene(host, scene_features=scene_features):
        o2 = e2()
        o1 = host._forward(views, kpt_ids, camera_group, mode, kpt_chunk=kpt_chunk, **e1)
    return mix_outputs(o2, o1, use2[:T], T, K)


def only_forward(host, views, kpt_ids, camera_group, mode, set_inputs, *, input_size=None,
                 kpt_prior=None, prompt_time=None, scene_features=None):
    """The set-only (PBM4) model's EVERY forward: E2 over its M queries, reduced to this slot's
    query (3D: query 0). `set_inputs` None = no anchors and no priors (validation, as the PBM4
    runs validated); `route = 'train'` keeps the full output under 'set' for `set_loss`. The box
    prompt never reaches E2 (its box is its anchor), as in PBM4."""
    from .model import share_scene
    si = set_inputs or {}
    K = int(kpt_ids.shape[1])

    def e2():
        """E2 over all queries -> (full output, presence, anchors)."""
        return set_forward(host, views, kpt_ids, camera_group, mode, anchors=si.get('anchors'),
                           anchor_prior=si.get('anchor_prior'),
                           anchor_prompt_t=si.get('anchor_prompt_t'), kpt_prior=kpt_prior,
                           prompt_time=prompt_time, input_size=input_size)
    if scene_features is None:
        out2, pres, anch = e2()
    else:
        with share_scene(host, scene_features=scene_features):
            out2, pres, anch = e2()
    out = select_and_slice(out2, pres, anch, K, camera_group, si.get('slot_boxes'),
                           si.get('slot'))[0]
    if si.get('route') == 'train':
        out = dict(out)
        out['set'] = dict(out=out2, pres=pres, anch=anch, K=K)
    return out


# ------------------------------------------------------------------------------------ the loss
def _hungarian(cp, gt, valid_rows, pm, norm):
    """{query: row} minimising mean L1 / norm + presence cost over the valid GT rows. `norm` None
    (3D): each row's own world extent x 2."""
    A = [i for i in range(gt.shape[0]) if bool(valid_rows[i])]
    matched = {}
    if A:
        Mq = cp.shape[1]
        Cm = np.zeros((Mq, len(A)))
        for j, i in enumerate(A):
            f = torch.isfinite(gt[i]).all(-1)
            d = (cp - gt[i][:, None]).abs().sum(-1)
            d = torch.where(f[:, None], d, torch.zeros_like(d)).sum((0, 2)) / f.sum().clamp(min=1)
            nm = norm
            if nm is None:
                g_ = gt[i][f]
                nm = float((g_.amax(0) - g_.amin(0)).max()) * 2.0 if len(g_) >= 2 else 1.0
            Cm[:, j] = (C_L1 * d / max(nm, 1e-6) + C_PRES * (1 - pm)).cpu().numpy()
        r, c = linear_sum_assignment(Cm)
        for qi, j in zip(r, c):
            matched[int(qi)] = A[j]
    return matched


def set_loss(loss_fn, raw, set_out, targets, device, cgroup=None):
    """E2's loss: the stock `PoseLoss` on all M*K tokens against per-query targets, plus the
    presence BCE. Anchored queries with an anchor and a valid row get THAT row (known assignment);
    the free queries are Hungarian-matched to the remaining rows. Presence target: 1 where the
    matched row has >= 2 in-crop keypoints and was assessed, 0 for an unmatched query on frames where
    every animal was assessed, else unsupervised."""
    out, pres, K = set_out['out'], set_out['pres'], set_out['K']
    M = pres.shape[-1]
    allc = targets['all'].to(device).float()
    ass = targets['assessed'].to(device)
    okr = targets['ok'].to(device)
    allass = targets['all_assessed'].to(device)
    abox = targets['anchors'].to(device)
    coords = out['coords_pred']
    B, T, Kt, R = coords.shape
    tgt = torch.full_like(coords, float('nan'))
    pt = torch.full(pres.shape, float('nan'), device=device)
    nan_T = torch.full((T,), float('nan'), device=device)
    for b in range(B):
        gt = allc[b]
        n_in = torch.isfinite(gt).all(-1).sum(-1)
        valid = okr[b] & ((n_in >= 2).any(-1))
        cp = coords[b].detach().reshape(T, M, K, R)
        NA_ = min(NA, M)
        pm = torch.sigmoid(pres[b].detach()).mean(0)
        matched = {i: i for i in range(NA_)
                   if bool(valid[i]) and bool(torch.isfinite(abox[b, i]).all(-1).any())}
        rest = valid.clone()
        for i in matched.values():
            rest[i] = False
        free = _hungarian(cp[:, NA_:], gt, rest, pm[NA_:], HUNGARIAN_NORM if R == 2 else None)
        matched.update({NA_ + q: r for q, r in free.items()})
        for q in range(M):
            if q in matched:
                i = matched[q]
                tgt[b, :, q * K:(q + 1) * K] = gt[i]
                m_ = ass[b, i] & ((n_in[i] >= 2) | (n_in[i] == 0))
                pt[b, :, q] = torch.where(m_, (n_in[i] >= 2).float(), nan_T)
            else:
                pt[b, :, q] = torch.where(allass[b], torch.zeros(T, device=device), nan_T)
    loss = loss_fn(raw, out, coords_true=tgt, vis_true=None, vis_true_cams=None, vis_2d_true=None,
                   cgroup=cgroup, p2d=tgt[:, None] if R == 2 else None, device=device)
    fm = torch.isfinite(pt)
    if fm.any():
        loss = loss + W_PRES * F.binary_cross_entropy_with_logits(pres[fm], pt[fm])
    return loss


# ------------------------------------------------------------------- checkpoint compatibility
# The scratch MX layout (crowd/moe, 2026-11): the HOST's decoder/query_encoder were E2 (with the dq
# additions) and E1 lived under `e1.*`. Read at the load boundary; nothing is re-saved.
_SCRATCH_RENAMES = (('query_encoder.kpt_dq_inst', 'set_decoder.inst'),
                    ('decoder.dq_groups.', 'set_decoder.groups.'),
                    ('dq_pres.', 'set_decoder.pres.'),
                    ('dq_abox.', 'set_decoder.abox.'),
                    ('decoder.', 'set_decoder.decoder.'),
                    ('query_encoder.', 'set_decoder.query_encoder.'))
# scratch-name equivalent of a set_decoder parameter -- the optimizer routes by it (same groups as MX0e)
ROUTE_NAMES = tuple((new, old) for old, new in _SCRATCH_RENAMES)


def is_scratch_layout(state) -> bool:
    """A scratch MX checkpoint: E1 under `e1.*` and the dq presence head at the top level."""
    return any(k.startswith('e1.') for k in state) and any(k.startswith('dq_pres.') for k in state)


def map_scratch_state(state):
    """A scratch MX0e state dict in this repo's layout: e1.X -> X (E1 IS the host), the host's old
    dq/decoder/query_encoder -> set_decoder.*. An MXbb (`e2b.*`, dropped) checkpoint is refused."""
    if any(k.startswith('e2b.') for k in state):
        raise SystemExit('this checkpoint is a scratch MXbb (two box-FiLM experts, e2b.*): that arm '
                         'was DROPPED (owner, 2026-11-06) and is not loadable here.')
    out = {}
    for k, v in state.items():
        if k.startswith('e1.'):
            out[k[3:]] = v
            continue
        for old, new in _SCRATCH_RENAMES:
            if k.startswith(old):
                out[new + k[len(old):]] = v
                break
        else:
            out[k] = v
    return out


def is_scratch_set_only(state) -> bool:
    """A scratch crowd-detr SET-DECODER-ONLY checkpoint (the PB series): dq_* additions, no `e1.*`
    -- its host decoder / query encoder ARE the set decoder's."""
    return (any(k.startswith('dq_pres.') for k in state)
            and not any(k.startswith('e1.') for k in state))


def map_scratch_set_only(state, where=''):
    """A scratch set-only state dict in this repo's `set_decoder = "only"` layout: only E2's own
    additions are renamed (host decoder / query encoder stay where they are). Only the PBM4
    architecture (M = 4 instance queries, box-anchor MLP, GroupPose, no other dq module) maps;
    anything else is refused by name. NOT visible in the weights: the anchored count NA (an
    `MA<n>` arm trained NA = n < 4 on the same tensors) -- such a run would load as NA = 4."""
    inst = state.get('query_encoder.kpt_dq_inst')
    extra = sorted({k.split('.')[0] + '.' + k.split('.')[1] for k in state
                    if ('dq_' in k and not k.startswith(('dq_pres.', 'dq_abox.'))
                        and not k.startswith(('decoder.dq_groups.', 'query_encoder.kpt_dq_inst')))})
    parts = all(any(k.startswith(x) for k in state) for x in ('decoder.dq_groups.', 'dq_abox.'))
    if inst is None or inst.shape[0] != M_ONLY or extra or not parts:
        raise SystemExit(
            f'{where}: a scratch crowd-detr SET-DECODER-ONLY checkpoint that is not the PBM4 '
            f'architecture ({0 if inst is None else inst.shape[0]} instance queries, expected '
            f'{M_ONLY}; extra dq modules {extra}). Only PBM4 (the v7 3D set decoder, '
            'set_decoder = "only") is deployable here.')
    out = {}
    for k, v in state.items():
        for old, new in _SCRATCH_RENAMES[:4]:
            if k.startswith(old):
                out[new + k[len(old):]] = v
                break
        else:
            out[k] = v
    return out


def route_name(name: str) -> str:
    """The name the optimizer routes a `set_decoder.*` parameter under (MX0e's own names)."""
    for new, old in ROUTE_NAMES:
        if name.startswith(new):
            return old + name[len(new):]
    return name
