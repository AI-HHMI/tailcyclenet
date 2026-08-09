"""The query encoder: what the model is told about each keypoint before it looks.

posetail's stock `QueryEncoder` describes a *point being tracked*: where it is, what it looks
like, how far away, whether it is visible. A pose model instead has K NAMED keypoints, and the
name is the thing that distinguishes them. So one fusion term becomes a learned per-keypoint
identity vector, and the appearance patch is kept alongside it -- the model gets to say both
*which* keypoint and *what is there*, which neither encoder alone could.

The honest part (posetail-pose's `w9_honest`, here the only behaviour): four of the fusion terms
are computed FROM the query position -- `pos`, `patch`, `vis`, and `depth` in 3D. When a keypoint
has no prior, there is no query position, and the natural fallback is the crop centre. That
fallback is not neutral: it is indistinguishable from a real prior that happens to say "the crop
centre", so the model cannot tell "I was not told" from "I was told this". Each of those terms
therefore carries a learned no-query token instead.

The tokens are initialised to the value the unprompted case produces today -- `missing_pos` to
`linear_pos([0, fourier(0)])`, `missing_vis` to the in-bounds row of the pretrained `vis_embed`
-- so an untrained model starts at the pretrained behaviour rather than at noise, and any
measured difference is the model learning to use an honest token.

**Keypoint ids do not travel through the occlusion channel.** posetail-pose had to smuggle them
there because `TrackerEncoder.forward`'s only per-keypoint channel is `occlusion`, and the stock
encoder clamps `occlusion + 1` into `[0, 2]` -- so the two consumers could never share the
tensor, and mis-packing it was invisible. Here the model stashes `_kpt_ids` on this module
directly, which frees the occlusion channel for actual occlusion. Nothing reads it yet (the
occ term stays at all-unknown, exactly as `w9_honest` had it), but the format now carries real
per-camera visibility, so wiring it up is a change to one line rather than an architecture.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat

from posetail.posetail.cube import is_point_visible, project_points_torch
from posetail.posetail.encoder_decoder import QueryEncoder, sample_patches
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


class PoseQueryEncoder(QueryEncoder):
    """Constructed with the same kwargs `TrackerEncoder` passes its own query encoder."""

    def __init__(self, *args, n_keypoints, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_keypoints = n_keypoints

        # One learned vector per keypoint id. The ids are GLOBAL (see format.Registry), so this
        # table spans every dataset in the run and a later run that appends datasets keeps the
        # rows it already trained.
        self.kpt_embed = nn.Embedding(n_keypoints, self.embed_dim)
        nn.init.normal_(self.kpt_embed.weight, std=0.02)
        self.kpt_norm = nn.LayerNorm(self.embed_dim)

        # No-query tokens. `missing_depth` / `missing_intrinsic` already exist on the parent and
        # are PRETRAINED, so they are reused rather than shadowed.
        with torch.no_grad():
            z = torch.zeros(1, 1, 1, 2)
            pos0 = self.linear_pos(torch.cat(
                [z, get_fourier_encoding(z, min_freq=0, max_freq=self.max_freq)], dim=-1))
            self.missing_pos = nn.Parameter(pos0.reshape(-1).clone())
            # The crop centre is always in bounds, so the unprompted `vis` term today always
            # reports index 1 -- confidently "visible" about a query that does not exist.
            self.missing_vis = nn.Parameter(self.vis_embed.weight[1].detach().clone())
        self.missing_patch = nn.Parameter(torch.zeros(self.embed_dim))
        nn.init.normal_(self.missing_patch, std=0.02)

        # Identity is an extra term beside the ten the parent builds, so the gate is rebuilt.
        # It MUST stay a gate: `fusion_norm` and `fusion_mlp` are inherited from the base
        # checkpoint and were trained on `0.5 * sum(terms)`, because a stock gate outputs
        # sigmoid(0) = 0.5 at init and stays in [0, 1] forever. Dropping the Sigmoid once left
        # weights ranging -1.385..+1.244 and put the fusion output outside the distribution the
        # pretrained decoder consumes.
        self.n_fusion_terms += 1
        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim * self.n_fusion_terms, self.n_fusion_terms), nn.Sigmoid())
        nn.init.normal_(self.gate[0].weight, std=0.01)
        nn.init.constant_(self.gate[0].bias, 0.0)

    # -- term bookkeeping, used by the warm start ------------------------------------------
    def _tail_terms(self):
        t = ['gap'] if self.time_embed_mode == 'fourier_rel' else []
        t += ['pos', 'depth', 'vis']
        if self.principal_point_embedding:
            t.append('pp')
        if self.intrinsic_embedding:
            t.append('intrinsic')
        if self.occlusion_embedding:
            t.append('occ')
        return t

    def stock_term_names(self):
        """`QueryEncoder.forward`'s own term order."""
        return ['patch', 'query_time', 'target_time'] + self._tail_terms() + (
            ['volume'] if self.use_volume_embedding else [])

    def term_names(self):
        """This class's order. Must match the `terms` list built in `forward`."""
        return ['kpt', 'query_time', 'target_time', 'patch'] + self._tail_terms()

    def inflate_stock_gate(self, weight, bias):
        """The base checkpoint's N-term gate placed into this (N+1)-term one, term by term.

        Mapped BY NAME, not by index: the two orders differ (stock leads with `patch`, this leads
        with `kpt`), so deriving the correspondence from names makes a future reordering a
        KeyError instead of every term silently receiving the weights trained for another.

        The identity row and identity input block stay zero, so at init the identity term is
        gated at sigmoid(0) = 0.5 -- the stock init value -- and every inherited row reads
        nothing from it. The model starts at the pretrained fusion behaviour.
        """
        src, dst = self.stock_term_names(), self.term_names()
        assert len(dst) == self.n_fusion_terms, \
            f'term_names() gives {len(dst)} but n_fusion_terms is {self.n_fusion_terms}'
        D = self.embed_dim
        assert tuple(weight.shape) == (len(src), D * len(src)), \
            f'checkpoint gate {tuple(weight.shape)} is not ({len(src)}, {D * len(src)})'
        ix = {n: i for i, n in enumerate(dst)}
        missing = [n for n in src if n not in ix]
        assert not missing, f'stock terms with no slot here: {missing}'
        w = torch.zeros(len(dst), D * len(dst), dtype=weight.dtype)
        b = torch.zeros(len(dst), dtype=bias.dtype)
        for i, ni in enumerate(src):
            for j, nj in enumerate(src):
                w[ix[ni], ix[nj] * D:(ix[nj] + 1) * D] = weight[i, j * D:(j + 1) * D]
            b[ix[ni]] = bias[i]
        return w, b

    # -- the honest substitution ------------------------------------------------------------
    def _sub_unprompted(self, term, token):
        """Replace `term` with `token` on every query slot that had no real prior.

        One implementation for every position-derived term: a second copy for `pos` and `vis`
        would be the same mistake as having three window loops.
        """
        ok = getattr(self, '_query_ok', None)
        if ok is None or term is None:
            return term
        ok = _tile_to_query_axis(ok, term.shape[1], '_query_ok')
        m = rearrange(ok.to(term.dtype), 'b t -> b t 1 1')
        return m * term + (1.0 - m) * token.view(1, 1, 1, -1)

    def forward(self, preprocessed_views, camera_group, query_coords, query_time, target_time,
                cube_scale, occlusion=None):
        """Byte-compatible with `QueryEncoder.forward`. Returns [B, T_query, n_cams, decoder_dim].

        `occlusion` is the real occlusion channel and is currently always None/unknown; keypoint
        ids arrive on `self._kpt_ids`, stashed by `PoseTrackerEncoder.forward`.
        """
        B, T_query, coord_dim = query_coords.shape
        n_cams = len(preprocessed_views)
        assert coord_dim in (2, 3), coord_dim
        is_2d = coord_dim == 2
        if is_2d:
            assert n_cams == 1, f'2D queries are single-camera; got {n_cams}'

        sizes = torch.stack([
            torch.tensor([v.shape[-1], v.shape[-2]], dtype=query_coords.dtype,
                         device=query_coords.device) for v in preprocessed_views])   # (cams, 2)

        # When every keypoint shares one query -- the unprompted case, where they all sit at the
        # scene centre -- the position-derived terms are constant along a query axis that is
        # T*K long (1128 on a 24-frame 47-keypoint window). Compute at width 1 and expand.
        # Asserted rather than assumed, with a full-width fallback.
        uniform = bool(torch.equal(query_coords, query_coords[:, :1].expand_as(query_coords)))
        Tq = 1 if uniform else T_query
        qc = query_coords[:, :1] if uniform else query_coords

        if is_2d:
            p2d_full = rearrange(qc, 'b t r -> 1 b t r')
        else:
            p2d_full = project_points_torch(camera_group, qc)

        pp = rearrange(p2d_full, 'ncams b t r -> b t ncams r') / sizes
        pp = pp * 2.0 - 1.0
        embed_pos = self.linear_pos(
            torch.cat([pp, get_fourier_encoding(pp, min_freq=0, max_freq=self.max_freq)], dim=-1))

        if is_2d:
            # Depth and visibility have no 3D meaning here, and the PARENT's answers are used
            # rather than invented: `missing_depth` is pretrained, and visibility degrades to a
            # pixel bounds check with margin 2. Substituting anything else would feed the
            # pretrained fusion gate a term it never saw.
            embed_depth = repeat(self.missing_depth, 'd -> b t c d', b=B, t=Tq, c=n_cams)
            W, H = sizes[0, 0], sizes[0, 1]
            in_bounds = ((qc[..., 0] >= 2) & (qc[..., 0] < W - 2) &
                         (qc[..., 1] >= 2) & (qc[..., 1] < H - 2))
            embed_vis = self.vis_embed(rearrange(in_bounds, 'b t -> b t 1').to(torch.int32))
        else:
            # `center` is (3,) static or (T,3) moving; frame 0 anchors the depth term, matching
            # the library's own approximation (encoder_decoder.py:455).
            centers = torch.stack([c['center'][0] if c['center'].ndim == 2 else c['center']
                                   for c in camera_group]).to(query_coords.dtype)
            cs = rearrange(cube_scale, 'cams b -> b 1 cams')
            raw = torch.linalg.norm(rearrange(qc, 'b t r -> b t 1 r') - centers, dim=-1) / cs
            dr = rearrange(torch.log(raw + 1e-6) * self.depth_norm_scale, 'b t c -> b t c 1')
            embed_depth = self.linear_depth(torch.cat(
                [dr, get_fourier_encoding(dr, min_freq=0, max_freq=self.max_freq)], dim=-1))
            qflat = rearrange(qc, 'b t r -> (b t) r')
            visible = torch.stack([is_point_visible(c, qflat, margin=2) for c in camera_group])
            embed_vis = self.vis_embed(
                rearrange(visible, 'ncams (b t) -> b t ncams', b=B).to(torch.int32))

        embed_pp = embed_intrinsic = None
        if self.principal_point_embedding:
            ppt = torch.stack([(c['mat'][:2, 2] - c['offset']).to(query_coords.dtype)
                               for c in camera_group])
            ppn = repeat(ppt / sizes * 2.0 - 1.0, 'cams r -> b t cams r', b=B, t=Tq)
            embed_pp = self.linear_pp(torch.cat(
                [ppn, get_fourier_encoding(ppn, min_freq=0, max_freq=self.max_freq)], dim=-1))
        if self.intrinsic_embedding:
            if is_2d:
                # The 2D camera is a NOMINAL pinhole, so its focal length says nothing about the
                # real optics. The parent uses the pretrained token here; so do we.
                embed_intrinsic = repeat(self.missing_intrinsic, 'd -> b t c d',
                                         b=B, t=Tq, c=n_cams)
            else:
                focal = torch.stack([torch.stack([c['mat'][0, 0], c['mat'][1, 1]]).to(
                    query_coords.dtype) for c in camera_group])
                fn = repeat(focal / sizes, 'cams r -> b t cams r', b=B, t=Tq)
                embed_intrinsic = self.linear_intrinsic(torch.cat(
                    [fn, get_fourier_encoding(fn, min_freq=0, max_freq=self.max_freq)], dim=-1))

        if uniform:
            def widen(x):
                return None if x is None else x.expand(B, T_query, n_cams, x.shape[-1])
            embed_pos, embed_depth = widen(embed_pos), widen(embed_depth)
            embed_vis, embed_pp = widen(embed_vis), widen(embed_pp)
            embed_intrinsic = widen(embed_intrinsic)

        # Honest no-query substitution, after widening so every term is on the (t n) axis.
        embed_pos = self._sub_unprompted(embed_pos, self.missing_pos)
        embed_vis = self._sub_unprompted(embed_vis, self.missing_vis)
        if not is_2d:
            embed_depth = self._sub_unprompted(embed_depth, self.missing_depth)

        # -- identity --------------------------------------------------------------------
        ids = getattr(self, '_kpt_ids', None)
        assert ids is not None, (
            'PoseQueryEncoder needs keypoint ids; PoseTrackerEncoder.forward must set _kpt_ids.')
        ids = _tile_to_query_axis(ids, T_query, '_kpt_ids')
        assert int(ids.min()) >= 0 and int(ids.max()) < self.n_keypoints, \
            f'keypoint id out of range [0, {self.n_keypoints}): {int(ids.min())}..{int(ids.max())}'
        embed_kpt = repeat(self.kpt_norm(self.kpt_embed(ids)), 'b t d -> b t cams d', cams=n_cams)

        # -- appearance at the query -----------------------------------------------------
        # `sample_patches` gathers frames with `query_time` and builds its grid from `centers`,
        # so the two must agree on Z. Derive Z from the centres: this encoder is called per
        # keypoint-chunk at inference, so the query axis it receives is not always the full one,
        # and a mismatch surfaces as an opaque grid_sampler error rather than anything nameable.
        cen = p2d_full[:, :, :1] if uniform else p2d_full
        Z = cen.shape[2]
        qt = query_time if query_time.dim() > 1 else query_time[:, None]
        if qt.shape[-1] != Z:
            qt = qt[..., :1].expand(-1, Z)
        patches = torch.stack([sample_patches(preprocessed_views[i], cen[i], qt, self.patch_size)
                               for i in range(n_cams)])
        embed_patch = rearrange(self.patch_processor(
            rearrange(patches, 'cams b t c p q -> (cams b t) c p q')),
            '(cams b t) d -> b t cams d', cams=n_cams, b=B, t=Z)
        if uniform:
            embed_patch = embed_patch.expand(B, T_query, n_cams, embed_patch.shape[-1])
        embed_patch = self._sub_unprompted(embed_patch, self.missing_patch)

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

        embed_occ = None
        if self.occlusion_embedding:
            occ_idx = (torch.zeros((B, T_query, n_cams), dtype=torch.long,
                                   device=query_coords.device) if occlusion is None
                       else (occlusion.to(torch.long) + 1).clamp_(0, 2))
            embed_occ = self.occ_embed(occ_idx)

        # -- gated fusion; ORDER MUST MATCH term_names() ----------------------------------
        terms = [embed_kpt, embed_query_time, embed_target_time, embed_patch]
        if embed_gap is not None:
            terms.append(embed_gap)
        terms += [embed_pos, embed_depth, embed_vis]
        for t in (embed_pp, embed_intrinsic, embed_occ):
            if t is not None:
                terms.append(t)
        assert len(terms) == self.n_fusion_terms, \
            f'built {len(terms)} terms but n_fusion_terms is {self.n_fusion_terms}'

        stack = torch.stack([t.expand(B, T_query, n_cams, self.embed_dim) for t in terms], dim=-2)
        weights = self.gate(rearrange(stack, 'b t c n d -> b t c (n d)'))
        combined = torch.einsum('btcn,btcnd->btcd', weights, stack)
        return self.fusion_mlp(self.fusion_norm(combined))
