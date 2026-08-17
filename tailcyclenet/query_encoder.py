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

from posetail.posetail.encoder_decoder import PatchProcessor, QueryEncoder, sample_patches
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
    mistake posetail-pose made: `scripts/test_query_path.py`'s two worst defects, live for its
    entire W4/W5 series, were a mask derived from an always-finite tensor (so `missing_patch` was
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




class PoseQueryEncoder(QueryEncoder):
    """Constructed with the same kwargs `TrackerEncoder` passes its own query encoder."""

    def __init__(self, *args, n_keypoints, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_keypoints = n_keypoints
        # `forward` below builds no volume term and `term_names()` has no slot for one, so the
        # base checkpoint's volume gate row has nowhere to go. Caught here rather than as the
        # assertion it would otherwise become deep inside `inflate_stock_gate`, a warm start later.
        assert not self.use_volume_embedding, (
            'use_volume_embedding is not supported: this encoder builds no volume term, so the '
            "base checkpoint's volume gate row has no slot here. Set use_volume_embedding = false.")

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
        """See the module-level `_sub_unprompted`; this is the only route to it here."""
        return _sub_unprompted(self, term, token)

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

        # A moving camera's extrinsic is (T,4,4), and `project_cam` aligns it against axis -3 of
        # the points (cube.py:95-99) -- so the query axis has to be un-flattened to (t n) first.
        # Both this and the visibility branch below exist in the stock QueryEncoder and were lost
        # in the port (posetail-pose's kpt_query_encoder.py:361,392 dropped them too; it never
        # ran a moving rig through its own encoder).
        # A PER-FRAME OFFSET IS A MOVING CAMERA TOO: `ext` stays static while `offset` varies,
        # and every branch this flag guards -- the (b,t,n) reshape before projecting, and the
        # per-frame visibility -- is needed for exactly the same reason. Nothing in this repo
        # builds a (T,2) offset any more, but posetail >= 0.3.5 supports it end to end and the
        # encoder's guard stays for a rig that has one.
        moving = not is_2d and any(c['ext'].ndim == 3 or c['offset'].ndim > 1
                                   for c in camera_group)
        T_clip = preprocessed_views[0].shape[1]

        # When every keypoint shares one query -- the unprompted case, where they all sit at the
        # scene centre -- the position-derived terms are constant along a query axis that is
        # T*K long (1128 on a 24-frame 47-keypoint window). Compute at width 1 and expand.
        # Asserted rather than assumed, with a full-width fallback.
        #
        # NOT AVAILABLE ON A MOVING RIG. One shared query point still projects to a DIFFERENT
        # pixel every frame, so a width-1 axis cannot represent the answer -- and it cannot be
        # reshaped to (t n) either.
        uniform = (not moving
                   and bool(torch.equal(query_coords, query_coords[:, :1].expand_as(query_coords))))
        Tq = 1 if uniform else T_query
        qc = query_coords[:, :1] if uniform else query_coords

        if is_2d:
            p2d_full = rearrange(qc, 'b t r -> 1 b t r')
        elif moving:
            qc_btn = rearrange(qc, 'b (t n) r -> b t n r', t=T_clip)
            p2d_full = rearrange(project_points_torch(camera_group, qc_btn),
                                 'cams b t n r -> cams b (t n) r')
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
            if moving:
                # Per-frame: the anchor is in view for some frames of the window and not others.
                qc_btn = rearrange(qc, 'b (t n) r -> b t n r', t=T_clip)
                visible = torch.stack([is_point_visible(c, qc_btn, margin=2)
                                       for c in camera_group])          # (cams,b,t,n)
                visible = rearrange(visible, 'ncams b t n -> b (t n) ncams')
            else:
                qflat = rearrange(qc, 'b t r -> (b t) r')
                visible = torch.stack([is_point_visible(c, qflat, margin=2)
                                       for c in camera_group])
                visible = rearrange(visible, 'ncams (b t) -> b t ncams', b=B)
            embed_vis = self.vis_embed(visible.to(torch.int32))

        embed_pp = embed_intrinsic = None
        if self.principal_point_embedding:
            ppt = torch.stack([(c["mat"][:2, 2] - c["offset"]).to(query_coords.dtype)
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


class WideQueryEncoder(nn.Module):
    """The pre-port query encoder: identity + time, fused at `latent_dim` instead of `embed_dim`.

    `PoseQueryEncoder` above describes a keypoint with ten terms at 256 dims, 27 of its 30 tensors
    inherited from the base tracker. This one describes it with six at 512 and inherits almost
    nothing. Which wins is not obvious, and the record says this one does whenever no prior is
    supplied: query-free, `PoseQueryEncoder`'s four distinguishing terms -- `pos`, `patch`, `vis`,
    `depth` -- are all computed from the SAME derived scene point for every keypoint, so they are
    constants and it degenerates into a lower-capacity version of this. `wide` won on fly (3.639
    vs 3.873) and on 3dpop at every matched iteration; every `pose` win on record was anchored and
    those are retracted GT leaks.

    So the width is not the point -- 94% of the parameters here are the fusion MLP, and the
    identity table at 47x512 is twice the capacity for the one mechanism that distinguishes
    LH-wrist from RH-wrist, which is load-bearing in a query-free model.

    TWO TERMS put the prior back, and they are what make this more than a reproduction:

    - `query_pos_embedding` -- where the query is. Without it this encoder ignores `query_coords`
      ENTIRELY, which is how six posetail-pose configs declared an anchor, trained, and were
      reported as anchored arms while the anchor was a literal no-op.
    - `query_patch_embedding` -- WHAT IS AT the query, not just where. The stock point tracker
      scored 100% on the two-animal probe where the pose model scored 0.0-8.5% on identical crops
      with an identical scorer, and the difference between those two queries is exactly a patch
      sampled at the query position.

    THEY ARE NOT SEPARATELY CONFIGURABLE. `PoseTrackerEncoder` builds both iff `query = "prior"`,
    because under `query = "none"` the prior is never read: `_query_ok` is all-False for the whole
    run, `_sub_unprompted` substitutes both tokens on every query, and the terms are two constant
    vectors feeding two dead gate inputs. Both off is therefore what `query = "none"` MEANS here
    -- and that is exactly golden's `j3` encoder, which until now could not be built in this repo
    at all: scoring it needed posetail-pose's own package, env and weights.

    UNLIKE posetail-pose's copy, both terms carry a learned no-query token. Its source note says
    missing-query tokens cannot work here because there is no position-derived term to be honest
    about; that note predates these two terms, which are exactly that. Without the tokens an
    unprompted keypoint's query is the derived scene point presented as a real answer -- and with
    an item-level `prompt_dropout` that is now ~half of all training steps.
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


# ----------------------------------------------------------------------------------------------
# the box prompt: which-animal-occupies-this-box, as a NON-position input channel
# ----------------------------------------------------------------------------------------------
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

BOX_MODES = ('film', 'term')


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


class BoxTermEncoder(WideQueryEncoder):
    """The box as its own [cx,cy,w,h] Fourier fusion term + one new gate row. Cheapest; scene-level
    rather than per-keypoint, so slower to earn gate mass than `film` -- but by 10k it matches it
    (report 27). Warm start inflates the wide parent's gate by NAME into the +1 slot, box row zero,
    so the pretrained fusion behaviour is preserved rather than reset."""

    def __init__(self, *args, box_max_freq=None, **kwargs):
        super().__init__(*args, **kwargs)          # term_names() already counts 'box' -> gate sized
        self.box_max_freq = int(box_max_freq if box_max_freq is not None else self.max_freq)
        self.linear_box = nn.Linear(8 * self.box_max_freq + 4, self.dim)
        self.missing_box = nn.Parameter(torch.zeros(self.dim))
        nn.init.normal_(self.missing_box, std=0.02)

    def term_names(self):
        return super().term_names() + ['box']

    def stock_term_names(self):
        """The WIDE parent's term order -- what warm_start's inflate-print reports as the source."""
        return WideQueryEncoder.term_names(self)

    def inflate_stock_gate(self, weight, bias):
        """Place the wide parent's N-term gate into this (N+1)-term one BY NAME, box row zero, so
        warm start preserves the pretrained fusion behaviour instead of dropping the whole gate on
        a shape mismatch (report 27). Called by `warm_start` because it is named this and the
        shapes differ."""
        src, dst = WideQueryEncoder.term_names(self), self.term_names()
        D = self.dim
        assert tuple(weight.shape) == (len(src), D * len(src)), \
            f'checkpoint gate {tuple(weight.shape)} is not ({len(src)}, {D * len(src)})'
        ix = {n: i for i, n in enumerate(dst)}
        w = torch.zeros(len(dst), D * len(dst), dtype=weight.dtype)
        b = torch.zeros(len(dst), dtype=bias.dtype)
        for i, ni in enumerate(src):
            for j, nj in enumerate(src):
                w[ix[ni], ix[nj] * D:(ix[nj] + 1) * D] = weight[i, j * D:(j + 1) * D]
            b[ix[ni]] = bias[i]
        return w, b

    def forward(self, preprocessed_views, camera_group, query_coords, query_time, target_time,
                cube_scale, occlusion=None):
        terms, ctx = self._build_terms(preprocessed_views, camera_group, query_coords,
                                       query_time, target_time)
        B, T_query, n_cams = ctx['B'], ctx['T_query'], ctx['n_cams']
        box = getattr(self, '_box_prompt', None)
        if box is not None:
            feat, ok = _box_features(box, ctx['sizes'], target_time)
            embed_box = self.linear_box(torch.cat(
                [feat, get_fourier_encoding(feat, min_freq=0, max_freq=self.box_max_freq)], -1))
            embed_box = _sub_missing(embed_box, ok, self.missing_box)
        else:
            embed_box = repeat(self.missing_box, 'd -> b t c d', b=B, t=T_query, c=n_cams)
        terms.append(embed_box)
        return self._gate_and_fuse(terms, ctx)


BOX_ENCODERS = {'film': BoxFilmEncoder, 'term': BoxTermEncoder}
