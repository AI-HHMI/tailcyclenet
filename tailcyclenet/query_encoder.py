"""The query encoder: what the model is told about each keypoint before it looks.

posetail's stock `QueryEncoder` describes a *point being tracked*: where it is, what it looks
like, how far away, whether it is visible. A pose model instead has K NAMED keypoints, and the
name is the thing that distinguishes them. So one fusion term becomes a learned per-keypoint
identity vector, and the appearance patch is kept alongside it -- the model gets to say both
*which* keypoint and *what is there*, which neither encoder alone could.

The position-derived terms carry a learned no-query token rather than falling back to the crop
centre: that fallback is indistinguishable from a real prior that happens to say "the crop
centre", so the model could not tell "I was not told" from "I was told this".

**Keypoint ids do not travel through the occlusion channel.** `TrackerEncoder.forward`'s only
per-keypoint channel is `occlusion`, and the stock encoder clamps `occlusion + 1` into `[0, 2]`,
so the two consumers can never share that tensor. The model stashes `_kpt_ids` on this module
directly instead.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat

from posetail.posetail.cube import project_points_torch
from posetail.posetail.encoder_decoder import PatchProcessor, sample_patches
from posetail.posetail.utils import get_fourier_encoding


def _tile_to_query_axis(x, n_query, what):
    """(B, N) per-keypoint -> (B, T*N) along the query axis, t-major.

    `_decode_from_scene` repeats the query set as `b n r -> b (t n) r`, so a per-keypoint tensor
    tiles that axis exactly. A non-divisible shape means the query set was chunked or reordered,
    and silently broadcasting one keypoint's value over all of them is how this went wrong
    before -- so it is an assertion, not a `repeat` that happens to work.
    """
    if x.shape[1] == n_query:
        return x
    n_rep, rem = divmod(n_query, x.shape[1])
    assert rem == 0, (
        f'{what} has {x.shape[1]} entries but the query axis is {n_query}; it must tile the '
        '(t n) axis exactly.')
    return repeat(x, 'b n -> b (t n)', t=n_rep)


def _sub_unprompted(owner, term, token):
    """Replace `term` with `token` on every query slot that had no real prior.

    ONE implementation for every position-derived term in BOTH encoders. A second copy for `pos`,
    `vis` or `qpos` would be the same mistake as having three window loops -- and it is the exact
    mistake posetail-pose made: the two worst defects in posetail-pose's query path, live for most of its history, were a mask derived from an always-finite tensor (so `missing_patch` was
    DEAD CODE and unprompted patches were sampled at the crop centre) and a `(B,N)` mask applied
    to a `(B,T*N)` axis (so keypoint 0's validity was broadcast over every keypoint). This repo
    fixes the first in `model.py` and the second in `_tile_to_query_axis` below -- but the second
    fix only holds if every substitution routes through HERE. A hand-written second copy in a new
    encoder reintroduces it, and no string grep would catch that.
    """
    ok = getattr(owner, '_query_ok', None)
    if ok is None or term is None:
        return term
    ok = _tile_to_query_axis(ok, term.shape[1], '_query_ok')
    m = rearrange(ok.to(term.dtype), 'b t -> b t 1 1')
    return m * term + (1.0 - m) * token.view(1, 1, 1, -1)


class WideQueryEncoder(nn.Module):
    """The query encoder: identity + time, fused at `latent_dim` instead of `embed_dim`.

    Six terms at 512 dims, inheriting almost nothing from the base tracker. 94% of the parameters
    are the fusion MLP; the identity table is the one mechanism distinguishing LH-wrist from
    RH-wrist, which is load-bearing in a query-free model.

    Two terms put the prior back:

    - `query_pos_embedding` -- where the query is. Without it this encoder ignores `query_coords`
      ENTIRELY, so a declared prior becomes a silent no-op.
    - `query_patch_embedding` -- WHAT IS AT the query, not just where.

    `PoseTrackerEncoder` builds both iff `query = "prior"`: under `query = "none"` the prior is
    never read, so both terms would be constant vectors feeding dead gate inputs. Both terms carry
    a learned no-query token, or an unprompted keypoint's query would be the derived scene point
    presented as a real answer.
    """

    def __init__(self, *, dim, embed_dim, decoder_dim, n_keypoints, n_frames, max_freq=10,
                 patch_size=9, time_embed_mode='fourier_rel', principal_point_embedding=False,
                 intrinsic_embedding=False, query_pos_embedding=True,
                 query_patch_embedding=True):
        super().__init__()
        self.dim = dim
        self.decoder_dim = decoder_dim
        self.max_freq = max_freq
        self.n_frames = n_frames
        self.n_keypoints = n_keypoints
        self.patch_size = int(patch_size)
        self.time_embed_mode = time_embed_mode
        self.principal_point_embedding = principal_point_embedding
        self.intrinsic_embedding = intrinsic_embedding
        self.query_pos_embedding = query_pos_embedding
        self.query_patch_embedding = query_patch_embedding

        self.kpt_embed = nn.Embedding(n_keypoints, dim)
        nn.init.normal_(self.kpt_embed.weight, std=0.02)
        self.kpt_norm = nn.LayerNorm(dim)

        self.time_norm = float(n_frames)
        if time_embed_mode == 'learned':
            self.t_query_embed = nn.Embedding(n_frames, dim)
            self.t_target_embed = nn.Embedding(n_frames, dim)
        else:
            self.linear_query_time = nn.Linear(2 * max_freq + 1, dim)
            self.linear_target_time = nn.Linear(2 * max_freq + 1, dim)
            self.linear_gap = nn.Linear(2 * max_freq + 1, dim)

        if principal_point_embedding:
            self.linear_pp = nn.Linear(4 * max_freq + 2, dim)
        if intrinsic_embedding:
            self.linear_intrinsic = nn.Linear(4 * max_freq + 2, dim)
        if query_pos_embedding:
            self.linear_qpos = nn.Linear(4 * max_freq + 2, dim)
            self.missing_qpos = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.missing_qpos, std=0.02)
        if query_patch_embedding:
            # AT THE PRETRAINED WIDTH, then projected. `PatchProcessor`'s convs are independent of
            # `embed_dim` (~93k params) but its MLP is not (~5.4M at 256): building it at `dim`
            # would inherit 93k of 5.5M and retrain 98% of the patch CNN from noise. Built at
            # `embed_dim` and named as the parent names it, every tensor loads by name at matching
            # shape and `warm_start` needs no special case at all. The adapter is 131k fresh.
            self.patch_processor = PatchProcessor(
                in_channels=3, patch_size=self.patch_size, embed_dim=embed_dim,
                conv_channels=[32, 64, 128])
            self.patch_proj = nn.Linear(embed_dim, dim)
            self.missing_patch = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.missing_patch, std=0.02)

        self.n_fusion_terms = len(self.term_names())
        self.gate = nn.Sequential(
            nn.Linear(dim * self.n_fusion_terms, self.n_fusion_terms), nn.Sigmoid())
        nn.init.normal_(self.gate[0].weight, std=0.01)
        nn.init.constant_(self.gate[0].bias, 0.0)

        self.fusion_norm = nn.LayerNorm(dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(dim * 4, decoder_dim))

    def term_names(self):
        """This class's fusion order. Must match the `terms` list built in `forward`."""
        t = ['kpt', 'query_time', 'target_time']
        if self.time_embed_mode == 'fourier_rel':
            t.append('gap')
        if self.query_pos_embedding:
            t.append('qpos')
        if self.query_patch_embedding:
            t.append('patch')
        if self.principal_point_embedding:
            t.append('pp')
        if self.intrinsic_embedding:
            t.append('intrinsic')
        return t

    def _fourier_time(self, times, linear):
        s = (times.to(torch.float32) / self.time_norm)[..., None, None]
        feat = torch.cat([s, get_fourier_encoding(s, min_freq=0, max_freq=self.max_freq)], dim=-1)
        return linear(feat)[..., 0, :]

    def _interp_time_embed(self, emb, times, n_frames):
        if n_frames == emb.num_embeddings:
            return emb(times)
        w = torch.nn.functional.interpolate(
            emb.weight.t().unsqueeze(0), size=n_frames,
            mode='linear', align_corners=False).squeeze(0).t()
        return torch.nn.functional.embedding(times, w)

    def forward(self, preprocessed_views, camera_group, query_coords, query_time, target_time,
                cube_scale, occlusion=None):
        """Byte-compatible with `QueryEncoder.forward`, so it drops into `_decode_from_scene`.

        Split into `_build_terms` (the fusion terms, up to the gate) and `_gate_and_fuse`, so a
        box-prompt subclass (`BoxFilmEncoder` / `BoxTermEncoder`) can extend the SAME term list
        rather than reimplementing this forward -- the one place the fusion terms are built.
        """
        terms, ctx = self._build_terms(preprocessed_views, camera_group, query_coords,
                                       query_time, target_time)
        return self._gate_and_fuse(terms, ctx)

    def _gate_and_fuse(self, terms, ctx):
        B, T_query, n_cams = ctx['B'], ctx['T_query'], ctx['n_cams']
        assert len(terms) == self.n_fusion_terms, \
            f'built {len(terms)} terms but n_fusion_terms is {self.n_fusion_terms}'
        stack = torch.stack([t.expand(B, T_query, n_cams, self.dim) for t in terms], dim=-2)
        weights = self.gate(rearrange(stack, 'b t c n d -> b t c (n d)'))
        combined = torch.einsum('btcn,btcnd->btcd', weights, stack)
        return self.fusion_mlp(self.fusion_norm(combined))

    def _build_terms(self, preprocessed_views, camera_group, query_coords, query_time,
                     target_time):
        """The fusion terms up to (not including) the gate. Returns `(terms, ctx)`; `ctx` carries
        what a box subclass needs (`B`, `T_query`, `n_cams`, `sizes`, `uniform`, `qpix`, `qt`).

        `occlusion` is unused: keypoint ids arrive on `self._kpt_ids`, stashed by
        `PoseTrackerEncoder.forward`. posetail-pose smuggled them through the occlusion channel
        because that was the only per-keypoint channel the library sliced for free; gotcha 5 freed
        it here.
        """
        B, T_query, coord_dim = query_coords.shape
        n_cams = len(preprocessed_views)
        is_2d = coord_dim == 2
        if is_2d:
            assert n_cams == 1, f'2D queries are single-camera; got {n_cams}'
        sizes = torch.stack([
            torch.tensor([v.shape[-1], v.shape[-2]], dtype=query_coords.dtype,
                         device=query_coords.device) for v in preprocessed_views])   # (cams, 2)

        # -- identity --------------------------------------------------------------------
        ids = getattr(self, '_kpt_ids', None)
        assert ids is not None, (
            'WideQueryEncoder needs keypoint ids; PoseTrackerEncoder.forward must set _kpt_ids.')
        ids = _tile_to_query_axis(ids, T_query, '_kpt_ids')
        assert int(ids.min()) >= 0 and int(ids.max()) < self.n_keypoints, \
            f'keypoint id out of range [0, {self.n_keypoints}): {int(ids.min())}..{int(ids.max())}'
        embed_kpt = repeat(self.kpt_norm(self.kpt_embed(ids)), 'b t d -> b t cams d', cams=n_cams)

        # -- time ------------------------------------------------------------------------
        embed_gap = None
        if self.time_embed_mode == 'learned':
            n_cur = preprocessed_views[0].shape[1]
            embed_query_time = repeat(self._interp_time_embed(self.t_query_embed, query_time,
                                                              n_cur), 'b t d -> b t c d', c=n_cams)
            embed_target_time = repeat(self._interp_time_embed(self.t_target_embed, target_time,
                                                               n_cur), 'b t d -> b t c d', c=n_cams)
        else:
            embed_query_time = repeat(self._fourier_time(query_time, self.linear_query_time),
                                      'b t d -> b t c d', c=n_cams)
            embed_target_time = repeat(self._fourier_time(target_time, self.linear_target_time),
                                       'b t d -> b t c d', c=n_cams)
            embed_gap = repeat(self._fourier_time(target_time - query_time, self.linear_gap),
                               'b t d -> b t c d', c=n_cams)

        terms = [embed_kpt, embed_query_time, embed_target_time]
        if embed_gap is not None:
            terms.append(embed_gap)

        # -- the query, in the pixel frame of THIS window's crop --------------------------
        # Computed once and shared by the two query terms. Both the moving-rig reshape and the
        # width-1 `uniform` collapse are ported from `PoseQueryEncoder`; posetail-pose's copy has
        # neither, so it projects a (t n)-flattened axis against a (T,4,4) extrinsic (gotcha 9)
        # and recomputes a constant term at full width on every query-free step.
        qpix, uniform = None, False
        if self.query_pos_embedding or self.query_patch_embedding:
            # A PER-FRAME OFFSET IS A MOVING CAMERA TOO: `ext` stays static while `offset` varies,
            # and every branch this flag guards -- the (b,t,n) reshape before projecting, and the
            # per-frame visibility -- is needed for exactly the same reason. Nothing in this repo
            # builds a (T,2) offset any more; the guard stays for a rig that has one.
            moving = not is_2d and any(c['ext'].ndim == 3 or c['offset'].ndim > 1
                                       for c in camera_group)
            T_clip = preprocessed_views[0].shape[1]
            uniform = (not moving and bool(torch.equal(
                query_coords, query_coords[:, :1].expand_as(query_coords))))
            qc = query_coords[:, :1] if uniform else query_coords
            if is_2d:
                qpix = rearrange(qc, 'b t r -> 1 b t r')
            elif moving:
                qc_btn = rearrange(qc, 'b (t n) r -> b t n r', t=T_clip)
                qpix = rearrange(project_points_torch(camera_group, qc_btn),
                                 'cams b t n r -> cams b (t n) r')
            else:
                qpix = project_points_torch(camera_group, qc)

        if self.query_pos_embedding:
            qp = rearrange(qpix, 'cams b t r -> b t cams r') / sizes * 2.0 - 1.0
            embed_qpos = self.linear_qpos(torch.cat(
                [qp, get_fourier_encoding(qp, min_freq=0, max_freq=self.max_freq)], dim=-1))
            if uniform:
                embed_qpos = embed_qpos.expand(B, T_query, n_cams, embed_qpos.shape[-1])
            terms.append(_sub_unprompted(self, embed_qpos, self.missing_qpos))

        if self.query_patch_embedding:
            # `sample_patches` builds its grid from `centers` and gathers frames with `query_time`,
            # so the two must agree on Z. Derive Z from the centres: this encoder is called per
            # keypoint-chunk at inference, so the query axis it receives is not always the full
            # one, and a mismatch surfaces as an opaque grid_sampler error rather than a name.
            cen = qpix
            Z = cen.shape[2]
            qt = query_time if query_time.dim() > 1 else query_time[:, None]
            if qt.shape[-1] != Z:
                qt = qt[..., :1].expand(-1, Z)
            patches = torch.stack([sample_patches(preprocessed_views[i], cen[i], qt,
                                                  self.patch_size) for i in range(n_cams)])
            embed_patch = rearrange(self.patch_processor(
                rearrange(patches, 'cams b t c p q -> (cams b t) c p q')),
                '(cams b t) d -> b t cams d', cams=n_cams, b=B, t=Z)
            embed_patch = self.patch_proj(embed_patch)
            if uniform:
                embed_patch = embed_patch.expand(B, T_query, n_cams, embed_patch.shape[-1])
            terms.append(_sub_unprompted(self, embed_patch, self.missing_patch))
        else:
            qt = None

        # -- rig ---------------------------------------------------------------------------
        if self.principal_point_embedding:
            ppt = torch.stack([(c["mat"][:2, 2] - c["offset"]).to(query_coords.dtype)
                               for c in camera_group])
            ppn = repeat(ppt / sizes * 2.0 - 1.0, 'cams r -> b t cams r', b=B, t=T_query)
            terms.append(self.linear_pp(torch.cat(
                [ppn, get_fourier_encoding(ppn, min_freq=0, max_freq=self.max_freq)], dim=-1)))
        if self.intrinsic_embedding:
            focal = torch.stack([torch.stack([c['mat'][0, 0], c['mat'][1, 1]]).to(
                query_coords.dtype) for c in camera_group])
            fn = repeat(focal / sizes, 'cams r -> b t cams r', b=B, t=T_query)
            terms.append(self.linear_intrinsic(torch.cat(
                [fn, get_fourier_encoding(fn, min_freq=0, max_freq=self.max_freq)], dim=-1)))

        ctx = dict(B=B, T_query=T_query, n_cams=n_cams, sizes=sizes, uniform=uniform,
                   qpix=qpix, qt=qt, is_2d=is_2d, target_time=target_time)
        return terms, ctx


# the box prompt: which-animal-occupies-this-box, as a NON-position input channel
#
# Report 27. Told which box the target animal occupies -- per frame, per camera -- a box-prompt
# model can be told which animal to return, WITHOUT the box being a position prior (it carries no
# per-keypoint position, only the animal's extent). Measured on crowded 2D calms21: at fixed
# weights the box removes -3.5 mm MPJPE held-out, and in the real window loop on WIDE crops it
# adds +0.26 MOTA and removes 11.7% fp_dup (the report-24 §9m wide-crop collapse). It SELECTS the
# named animal at own 0.94 on a shared/union crop against a no-box control's 0.00. It is
# ROOT-CONDITIONAL by keypoint density (sparse rat-city K=4 selects far less) and
# GEOMETRY-CONDITIONAL (redundant on a crop already centred on its target, decisive on a
# shared/off-centre one). Two mechanisms work: `film` (the winner) and `term`; a patch-channel
# variant was tried and its box conv never trains off zero in a warm start (report 27), so it is
# not ported.
#
# The box arrives as an INSTANCE ATTRIBUTE `self._box_prompt` (T-frame (B,T,C,4) crop-pixel box),
# stashed by `PoseTrackerEncoder._decode_from_scene` exactly like `_kpt_ids` -- the library's
# call into the query encoder has a fixed signature with no box slot.

BOX_MODES = ('film',)


def _normalize_box(box_prompt, sizes):
    """(...,C,4) xyxy crop pixels -> (cx,cy,w,h) normalised per camera, the same `/size*2-1` the
    position terms use, so the box lands in the distribution the fusion gate already expects."""
    x0, y0, x1, y1 = box_prompt.unbind(-1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    W, H = sizes[..., 0], sizes[..., 1]
    return torch.stack([cx / W * 2.0 - 1.0, cy / H * 2.0 - 1.0, w / W * 2.0, h / H * 2.0], -1)


def _gather_box_by_target_time(box_per_frame, target_time):
    """(B,T_clip,cams,D) per-FRAME box features gathered onto the (t n) query axis by the
    library's own `target_time` (B,T_query), the ACTUAL frame each query slot decodes. Safer than
    re-deriving the t-major tiling by hand, and it sidesteps the `uniform` collapse (target_time
    is never reduced to width 1 the way qpos/patch are)."""
    B, T_clip, C, D = box_per_frame.shape
    idx = target_time.to(torch.float32).round().long().clamp(0, T_clip - 1)      # (B,T_query)
    idx = repeat(idx, 'b t -> b t c d', c=C, d=D)
    return torch.gather(box_per_frame, 1, idx)


def _box_features(box_prompt, sizes, target_time):
    """(B,T_clip,cams,4) crop-pixel box -> ((B,T_query,cams,4) normalised, finite mask), gathered
    per query slot. A NaN frame (no box) yields ok=False, so the missing-box token stands in."""
    feat = _normalize_box(box_prompt, sizes[None, None])
    feat = _gather_box_by_target_time(feat, target_time)
    ok = torch.isfinite(feat).all(-1)
    return torch.nan_to_num(feat), ok


def _sub_missing(term, mask, token):
    """The box-shaped `_sub_unprompted`: `mask` is already at the query axis's width (from
    `_box_features`'s gather), so a plain masked blend with the learned no-box token. ONE routine
    for every box variant."""
    if term is None:
        return term
    m = mask.to(term.dtype)[..., None]
    return m * torch.nan_to_num(term) + (1.0 - m) * token.view(1, 1, 1, -1)


class BoxFilmEncoder(WideQueryEncoder):
    """`kpt = kpt * (1 + gamma(box)) + beta(box)` -- FiLM on the identity term. THE WINNER
    (report 27). Per-keypoint by construction, works query-free (the identity term is always
    built), MIPNet's lambda at the cheapest site -- one linear off the identity term, not buried
    in a frozen CNN. gamma/beta are ZERO-INITIALISED, so at init this is a bit-identical no-op
    (`test_box_prompt.py` pins it) and warm start is undisturbed; it only diverges once trained.
    """

    def __init__(self, *args, box_max_freq=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_max_freq = int(box_max_freq if box_max_freq is not None else self.max_freq)
        self.film = nn.Sequential(nn.Linear(8 * self.box_max_freq + 4, self.dim), nn.GELU(),
                                  nn.Linear(self.dim, 2 * self.dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.missing_film = nn.Parameter(torch.zeros(2 * self.dim))

    def forward(self, preprocessed_views, camera_group, query_coords, query_time, target_time,
                cube_scale, occlusion=None):
        terms, ctx = self._build_terms(preprocessed_views, camera_group, query_coords,
                                       query_time, target_time)
        B, T_query, n_cams = ctx['B'], ctx['T_query'], ctx['n_cams']
        box = getattr(self, '_box_prompt', None)
        if box is not None:
            feat, ok = _box_features(box, ctx['sizes'], target_time)
            gb = self.film(torch.cat(
                [feat, get_fourier_encoding(feat, min_freq=0, max_freq=self.box_max_freq)], -1))
            gb = _sub_missing(gb, ok, self.missing_film)
        else:
            gb = repeat(self.missing_film, 'd -> b t c d', b=B, t=T_query, c=n_cams)
        gamma, beta = gb[..., :self.dim], gb[..., self.dim:]
        terms[0] = terms[0] * (1.0 + gamma) + beta          # terms[0] is the identity term
        return self._gate_and_fuse(terms, ctx)


BOX_ENCODERS = {'film': BoxFilmEncoder}
