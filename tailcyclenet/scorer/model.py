"""`PoseScorer`: the track-quality scorer, as a subclass of THIS repo's pose encoder.

Why not `class PoseScorer(ScorerEncoder, PoseTrackerEncoder)`: both bases define `encode_scene`
with incompatible signatures and incompatible semantics -- the reference takes RAW views and
normalises them itself, `PoseTrackerEncoder` (`model.py`) takes ALREADY-NORMALISED views and is the
only place `scene_precision`/`camera_batch` are honoured. Under the MRO one silently wins and the
other's behaviour is discarded, which produces numbers rather than an exception (the class of bug
`gridresid_offset` is named for). So: SINGLE inheritance, and the two concerns are separated by
name -- `_normalize_views` (ours) and `encode_scene` (inherited, untouched).

The structural difference from the tracker is **query = target**. The tracker repeats one query
position across all frames ("where is point n at every frame?"); the scorer feeds the whole
trajectory and evaluates each point in place at its own frame, then pools over (time, cameras) into
one latent per point and reads out a scalar quality score. Everything downstream of the query is
the same decoder, which is what lets a pose checkpoint warm-start this model.

Higher score = cleaner track. Scores are RELATIVE, never calibrated probabilities -- no cross-root
or cross-keypoint threshold is implied by any number this model produces.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat

from posetail.posetail.cube import (get_camera_scale, points_to_rays, project_points_torch)
from posetail.posetail.encoder_decoder import AttentionPooling

from ..model import PoseTrackerEncoder, build_model


def _refuse_moving_rig(camera_group):
    """The scorer is static-camera only, refused BY NAME where the cameras are in hand.

    `_scene_scalars` and the ray loop index `cam['center']` / `cam['ext']` as if a rig were fixed,
    so a moving rig would be silently mis-projected. This is checked at the INPUT boundary (not at
    model construction, which never sees a camera), and it is the same test the query encoder uses
    to decide a rig is moving.

    Inputs: camera_group -- a list of posetail camera dicts.
    Outputs: None, or raises ValueError naming the offending cameras.
    Side effects: none.
    """
    moving = [i for i, c in enumerate(camera_group)
              if c['ext'].ndim == 3 or c['offset'].ndim > 1]
    if moving:
        raise ValueError(
            f'the track-quality scorer does not support a MOVING rig (camera(s) {moving} carry '
            'per-frame extrinsics). `_scene_scalars` and the ray construction treat the rig as '
            'static, so the projection would be silently wrong rather than failing. Scoring a '
            'moving-rig session needs a moving-safe scoring path that does not exist yet.')


class PoseScorer(PoseTrackerEncoder):
    """The pose encoder plus an attention-pooling head and a score/precision readout.

    Adds, on top of `PoseTrackerEncoder`:
      * `attn_pool` -- pools the per-(time, camera) decoder latents of each point into one latent
        (permutation-invariant over cameras, time-embedded).
      * `missing_point` -- a learned token substituted for slots whose track coordinate is NaN, so
        a missing point never reaches the decoder as a NaN or as a fabricated position.
      * `score_feature`/`score_head` -- the scalar quality readout, its head built with NO bias
        (mirroring the reference's miss-alignment lineage).
      * `precision_head` -- a per-point confidence in (0,1) used to weight the triplet loss.

    The inherited pose heads (the 3D grid head, the 2D head bank, confidence/visibility) are NOT
    driven by this path -- `score()` never calls the parent forward. Whether they can be frozen
    and dropped from the optimizer is a QUESTION TO MEASURE, not to reason from the class
    hierarchy: `self.decoder` is shared, so a backward audit decides. `build_optimizer` skips
    `requires_grad=False` parameters in every branch, so freezing them is safe once measured.
    """

    def __init__(self, *args, pool_num_heads=8, score_hidden=64, use_precision=True, **kwargs):
        """`*args`/`**kwargs` go to `PoseTrackerEncoder` unchanged -- every encoder/decoder shape
        comes from there, so a warm start stays an exact load. The keyword arguments here are the
        scorer's own heads and are FRESH at warm start.

        `video_encoder_requires_grad = false` is the shipped default (the reference freezes the
        V-JEPA backbone): the staged unfreeze is a later arm, and freezing is what keeps good and
        bad sharing one scene encode honest without retaining activations for it.

        `missing_point` is sized to the query encoder's OUTPUT dim, because it replaces the fused
        query embedding before the decoder. It keeps missing slots finite so the decoder's
        temporal/camera self-attention cannot be poisoned by a NaN, and the slot still flows
        through pooling with its time embedding, so missingness informs the score rather than
        being hidden from it.

        Inputs: pool_num_heads / score_hidden / use_precision -- the new heads' sizes.
        Outputs: none.
        Side effects: builds parameters and prints the query-encoder summary via the parent.
        """
        super().__init__(*args, **kwargs)
        d = self.decoder.embed_dim

        self.attn_pool = AttentionPooling(d, num_heads=pool_num_heads, n_frames=self.S)

        self.missing_point = nn.Parameter(torch.zeros(self.query_encoder.decoder_dim))
        nn.init.normal_(self.missing_point, std=0.02)

        self.score_feature = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, score_hidden), nn.SiLU())
        self.score_head = nn.Linear(score_hidden, 1, bias=False)
        self.precision_head = nn.Linear(score_hidden, 1) if use_precision else None

    def _normalize_views(self, views):
        """uint8 (or float) views -> the `views_norm` list `encode_scene` expects.

        THE ONE PLACE THE uint8 DIVIDE HAPPENS on the scoring path. `PoseTrackerEncoder._forward`
        normally does it (and `TrackerEncoder.forward` then pads and normalises); the scoring path
        does not go through `_forward`, so it must reproduce exactly that sequence here --
        `/255`, then `b t h w c -> b t c h w`, then `pad_to_size` + ImageNet `normalize` (the two
        steps `transform_norm` composes). Doing it twice would apply the divide to already-scaled
        pixels; not doing it would feed the frozen backbone 0..255.

        Inputs: views -- list of [b,t,h,w,c] tensors, one per camera.
        Outputs: a list of [b,t,c,h,w] normalised float tensors, same length.
        Side effects: none (a uint8 input is converted to a new float tensor, not written through).
        """
        out = []
        for frames in views:
            if frames.dtype == torch.uint8:
                frames = frames.float().div_(255)
            frames = rearrange(frames, 'b t h w c -> b t c h w')
            out.append(self.transform_norm(frames))
        return out

    def _scene_scalars(self, coords_flat, camera_group, device, times=None):
        """`cube_scale`, `cube_scale_shared`, `f_eff`, `scene_center`, `scene_radius`.

        Mirrors `TrackerEncoder.forward` so the numbers match what the warm-started weights were
        trained under, with the query axis flattened over (t, k) instead of (t, n) -- same thing,
        a different name. Computed from the RAW NaN-carrying coords: `get_camera_scale` drops NaN
        via `is_point_visible` and `scene_center` uses `nanmean`, so missing slots cannot bias the
        scene scale or centroid.

        Inputs: coords_flat -- [B, T*K, R]; camera_group -- posetail camera dicts;
                device -- where to build constants; times -- per-query-slot frame index or None.
        Outputs: (cube_scale [n_cams,B], cube_scale_shared [B], f_eff [n_cams]|None,
                  scene_center [B,3]|None, scene_radius [B]|None).
        Side effects: none.
        """
        B, _, R = coords_flat.shape
        n_cams = len(camera_group)

        if R == 3:
            cube_scale = get_camera_scale(camera_group, coords_flat, times=times)
        else:
            cube_scale = torch.ones((n_cams, B), device=device)
        if not self.per_camera_cube_scale:
            med = torch.median(cube_scale, dim=0).values
            cube_scale = med[None, :].expand(n_cams, B).contiguous()

        f_eff = None
        if self.f_eff_scale:
            if R == 3:
                f_eff = torch.stack([
                    0.5 * (cam['mat'][0, 0] + cam['mat'][1, 1]) for cam in camera_group
                ]).to(device)
                if not self.per_camera_cube_scale:
                    f_eff = torch.full((n_cams,), torch.median(f_eff).item(), device=device)
            else:
                f_eff = torch.ones((n_cams,), device=device)

        cube_scale_shared = torch.median(cube_scale, dim=0).values

        scene_center = None
        scene_radius = None
        if self.metric_ray_translation:
            centers_w = torch.stack([cam['center'][0] if cam['center'].ndim == 2
                                     else cam['center'] for cam in camera_group])
            if R == 3:
                scene_center = torch.nanmean(coords_flat.to(torch.float32), dim=1)
                dist = (centers_w[:, None, :] - scene_center[None, :, :]).norm(dim=-1)
                scene_radius = torch.median(dist, dim=0).values
            else:
                scene_center = centers_w[0][None].expand(B, 3)
                scene_radius = torch.ones(B, device=device)

        return cube_scale, cube_scale_shared, f_eff, scene_center, scene_radius

    @staticmethod
    def _fill_nearest_valid(coords, valid):
        """Fill NaN slots with the point's nearest OBSERVED frame (forward-fill, then back-fill).

        Purely to keep projection and rays finite -- `missing_point` is what the decoder and
        pooling actually see for those slots, and a NaN in a masked key would still poison the
        decoder's self-attention. An all-missing point falls through as 0 after the caller's
        `nan_to_num`.

        Inputs: coords -- [b,t,k,R] with NaN where unobserved; valid -- [b,t,k] bool.
        Outputs: [b,t,k,R] with every slot finite.
        Side effects: none.
        """
        b, t, k, R = coords.shape
        idx = torch.arange(t, device=coords.device)[None, :, None]
        fwd = torch.where(valid, idx, torch.full_like(idx, -1)).cummax(dim=1).values
        bwd = torch.where(valid, idx, torch.full_like(idx, t)).flip(1).cummin(dim=1).values.flip(1)
        src = torch.where(fwd >= 0, fwd, bwd).clamp(0, t - 1)
        src = src.unsqueeze(-1).expand(b, t, k, R)
        return torch.gather(torch.nan_to_num(coords, nan=0.0), 1, src)

    def _kpt_ids_checked(self, kpt_ids, n_kpt):
        """Every id must address a real row of this model's identity table.

        The scorer is conditioned on the keypoint REGISTRY, so scoring a root whose names were not
        in the training registry is out of range -- refused by name rather than by an index error
        deep inside an embedding lookup.

        Inputs: kpt_ids -- an integer tensor of any shape; n_kpt -- this model's table size.
        Outputs: `kpt_ids` as a long tensor, unchanged in value.
        Side effects: none, or raises ValueError naming the offending id.
        """
        lo, hi = int(kpt_ids.min()), int(kpt_ids.max())
        if lo < 0 or hi >= n_kpt:
            bad = hi if hi >= n_kpt else lo
            raise ValueError(
                f'keypoint id {bad} is outside this scorer\'s registry [0, {n_kpt}). A scorer is '
                'conditioned on the keypoint registry it was trained on (a calms21 scorer has no '
                'row for an allen or 3dpop keypoint), so scoring a root with different keypoint '
                'names needs its own scorer.')
        return kpt_ids.long()

    def _scene_frame_pos(self, scene_features, views_norm):
        """Temporal-RoPE key positions for the scene tokens, or None when RoPE is off.

        The decoder REQUIRES these when `cross_attn_rope` is on. The token count is derived from
        the tokens the encoder ACTUALLY produced rather than recomputed from T, because at
        T < tubelet_size the recomputation is 0 and would silently yield zero scene tokens. It
        depends on the scene and the canvas, never on the keypoint slice, so it is computed once.

        Inputs: scene_features -- the encoded scene; views_norm -- the normalised views.
        Outputs: [n_tokens] float tensor, or None.
        Side effects: none.
        """
        if not self.cross_attn_rope:
            return None
        tub, ps = self.scene_encoder.tubelet_size, self.scene_encoder.patch_size
        H, W = views_norm[0].shape[-2], views_norm[0].shape[-1]
        gH, gW = H // ps, W // ps
        gT = scene_features.shape[-2] // (gH * gW)
        slot = torch.arange(gT * gH * gW, device=views_norm[0].device) // (gH * gW)
        return slot.float() * tub + (tub - 1) / 2.0

    def _score_slice(self, cf, valid_s, occ_s, k0, ctx):
        """Score one K-slice of the window: query, rays, decode, pool, read out.

        Split out so `score`'s chunked path and its single-pass path run the SAME code. The slice
        is exact because everything that couples keypoints -- the scene scalars, the validity mask
        and the NaN fill -- was computed over the full K before this is called.

        QUERY == TARGET: each `(t, k)` token lives at its own frame and is evaluated at that frame,
        which is the whole structural difference from the tracker. Missing slots are replaced by
        the learned `missing_point` token, and the pooling mask drops only OBSERVED `(t, camera)`
        slots -- missingness is orthogonal to track quality, so it must not shift the score. The
        mask is force-cleared for a fully missing point, because a softmax over all `-inf` is NaN
        and `min_valid_frames` is a training-side guarantee this function should not assume.

        Inputs: cf -- [B,T,Kc,R] coords for this slice; valid_s -- [B,T,Kc] bool;
                occ_s -- [B,Kc,n_cams] occlusion state or None; k0 -- the slice's offset into K;
                ctx -- the shared scene scalars, times, RoPE positions and slice-invariant sizes.
        Outputs: (scores [B,Kc], precision [B,Kc]).
        Side effects: sets `query_encoder._kpt_ids` / `_query_ok` for the duration of the call.
        """
        device = cf.device
        B, T, Kc, R = cf.shape
        n_cams = ctx['n_cams']
        cube_scale = ctx['cube_scale']
        cube_scale_shared = ctx['cube_scale_shared']
        scene_center, scene_radius = ctx['scene_center'], ctx['scene_radius']
        scene_features, views_norm = ctx['scene_features'], ctx['views_norm']
        scene_frame_pos = ctx['scene_frame_pos']

        self.query_encoder._kpt_ids = ctx['kpt_ids'][:, k0:k0 + Kc]
        self.query_encoder._query_ok = torch.ones((B, Kc), dtype=torch.bool, device=device)
        coords_flat = rearrange(cf, 'b t k r -> b (t k) r')

        query_coords = coords_flat
        query_time = repeat(torch.arange(T, device=device), 't -> b (t k)', b=B, k=Kc)
        target_time = query_time

        occlusion_rep = None
        if occ_s is not None:
            occlusion_rep = repeat(occ_s, 'b k c -> b (t k) c', t=T)

        query_embeds = self.query_encoder(
            views_norm, camera_group=ctx['camera_group'],
            query_coords=query_coords, query_time=query_time,
            target_time=target_time, cube_scale=cube_scale,
            occlusion=occlusion_rep)
        query_embeds = rearrange(query_embeds, 'b (t k) cams d -> b t k cams d', t=T, k=Kc)

        query_embeds = torch.where(
            (~valid_s)[..., None, None], self.missing_point.to(query_embeds.dtype), query_embeds)

        if R == 3:
            p2d_query = project_points_torch(ctx['camera_group'], query_coords)
            p2d_query = rearrange(p2d_query, 'cams b (t k) r -> cams b t k r', t=T, k=Kc)
        else:
            p2d_query = rearrange(query_coords, 'b (t k) r -> 1 b t k r', t=T, k=Kc)

        query_rays_per_cam = []
        for i in range(n_cams):
            rays_per_b = []
            for b in range(B):
                p2d_ib = rearrange(p2d_query[i, b], 't k r -> (t k) r')
                if self.metric_ray_translation:
                    rays_per_b.append(points_to_rays(
                        ctx['camera_group'][i], p2d_ib, cube_scale_shared[b],
                        scene_center=scene_center[b], scene_radius=scene_radius[b]))
                else:
                    rays_per_b.append(points_to_rays(
                        ctx['camera_group'][i], p2d_ib, cube_scale_shared[b]))
            query_rays_per_cam.append(torch.stack(rays_per_b, dim=0))
        query_rays = rearrange(torch.stack(query_rays_per_cam, dim=0),
                               'cams b (t k) d e -> b t k cams d e', t=T, k=Kc)

        mode_idx = torch.tensor([1 if R == 3 else 0], dtype=torch.long, device=device)
        latents = self.decoder(scene_features, query_embeds, query_rays, mode_idx,
                               scene_frame_pos=scene_frame_pos)['latent']

        pool_mask = repeat(~valid_s, 'b t k -> b k t cams', cams=latents.shape[3]).contiguous()
        all_masked = pool_mask.flatten(2).all(dim=-1)
        if all_masked.any():
            pool_mask[all_masked] = False

        pooled = self.attn_pool(latents, key_padding_mask=pool_mask)
        feats = self.score_feature(pooled)
        scores = self.score_head(feats)[..., 0]
        if self.precision_head is not None:
            precision = torch.sigmoid(self.precision_head(feats)[..., 0])
        else:
            precision = torch.ones_like(scores)
        return scores, precision

    def score(self, views_norm, scene_features, coords_full, camera_group, kpt_ids,
              kpt_chunk=None, occlusion=None):
        """Score one window's track. `coords_full`: [B, T, K, R] -> (scores, precision) [B, K].

        `kpt_ids`: [B, K] LONG -- the GLOBAL REGISTRY ids of the keypoints in `coords_full`'s own
        axis order, exactly as the loader hands them to the pose model. NOT `arange(K)`, and the
        difference is a silent wrong answer rather than an error: a session may reorder or subset
        the registry's names (CLAUDE.md gotcha 4), so session position 0 is not registry id 0.
        A dense range here would score every keypoint under some other body part's learned
        identity vector while every shape stayed correct.

        The per-point work runs over a K-slice when `kpt_chunk` is set; the scene scalars, the
        validity mask and the NaN fill are all computed over the FULL K first, so slicing is exact
        and only bounds peak memory.

        The query-encoder stashes are set immediately before the `try` and cleared in its
        `finally`: an exception anywhere after them must not leave one behind, or the NEXT window's
        forward silently runs with this window's ids. `_box_prompt` is never a real box --
        `box_prompt = "film"` keeps the model's `missing_film` token in place, so the warm start is
        an exact load and a future box-at-QC-time arm stays buildable.

        Inputs: views_norm -- normalised views; scene_features -- their encode; coords_full --
                [B,T,K,R]; camera_group -- posetail cameras; kpt_ids -- [B,K] global registry ids;
                kpt_chunk -- score K in slices of this size, or None; occlusion -- [B,K,n_cams].
        Outputs: (scores [B,K], precision [B,K]).
        Side effects: temporarily stashes `_kpt_ids`/`_query_ok`/`_box_prompt` on the query
            encoder, always clearing them; raises ValueError on a moving rig or a bad keypoint id.
        """
        device = coords_full.device
        B, T, K, R = coords_full.shape
        n_cams = len(camera_group)
        _refuse_moving_rig(camera_group)
        assert kpt_ids.shape == (B, K), \
            f'kpt_ids {tuple(kpt_ids.shape)} must be (B, K) = {(B, K)}'
        kpt_ids = self._kpt_ids_checked(kpt_ids.to(device), self.n_keypoints)

        valid = torch.isfinite(coords_full).all(dim=-1)
        coords_full = coords_full.to(torch.float32)

        coords_raw_flat = rearrange(coords_full, 'b t k r -> b (t k) r')
        times_full = repeat(torch.arange(T, device=device), 't -> b (t k)', b=B, k=K)
        cube_scale, cube_scale_shared, f_eff, scene_center, scene_radius = self._scene_scalars(
            coords_raw_flat, camera_group, device, times=times_full)

        coords_full = self._fill_nearest_valid(coords_full, valid)
        coords_full = torch.nan_to_num(coords_full, nan=0.0)

        ctx = {'n_cams': n_cams, 'kpt_ids': kpt_ids, 'camera_group': camera_group,
               'cube_scale': cube_scale, 'cube_scale_shared': cube_scale_shared,
               'scene_center': scene_center, 'scene_radius': scene_radius,
               'scene_features': scene_features, 'views_norm': views_norm,
               'scene_frame_pos': self._scene_frame_pos(scene_features, views_norm)}

        self.query_encoder._query_ok = torch.ones((B, K), dtype=torch.bool, device=device)
        self.query_encoder._box_prompt = None
        try:
            if kpt_chunk and K > kpt_chunk:
                s_parts, p_parts = [], []
                for k0 in range(0, K, kpt_chunk):
                    k1 = min(k0 + kpt_chunk, K)
                    occ = None if occlusion is None else occlusion[:, k0:k1]
                    s, p = self._score_slice(coords_full[:, :, k0:k1], valid[:, :, k0:k1], occ,
                                             k0, ctx)
                    s_parts.append(s)
                    p_parts.append(p)
                return torch.cat(s_parts, dim=1), torch.cat(p_parts, dim=1)
            return self._score_slice(coords_full, valid, occlusion, 0, ctx)
        finally:
            self.query_encoder._kpt_ids = None
            self.query_encoder._query_ok = None
            self.query_encoder._box_prompt = None

    def forward(self, views, coords, camera_group, kpt_ids, kpt_chunk=None, occlusion=None):
        """Single-sample inference path.

        Inputs: views -- list of [b,t,h,w,c] uint8 or float; coords -- [b,t,k,R]; camera_group --
                posetail cameras; kpt_ids -- [b,k] global registry ids (see `score`).
        Outputs: (scores [b,k], precision [b,k]).
        Side effects: none beyond `score`'s temporary stashes.
        """
        views_norm = self._normalize_views(views)
        scene_features = self.encode_scene(views_norm)
        return self.score(views_norm, scene_features, coords, camera_group, kpt_ids,
                          kpt_chunk=kpt_chunk, occlusion=occlusion)

    def score_triplet(self, trip):
        """Score a (good, bad, anchor) triplet in ONE forward pass.

        One entry/exit of the module's forward and one backward -- which is why this exists as a
        method rather than a loop at the call site, and what keeps a DDP reducer from seeing
        several forwards per iteration.

        `bad` shares `good`'s pixels by construction, so its scene is reused rather than encoded
        twice; the anchor owns its own pixels and is encoded separately.

        Inputs: trip -- the dict from `tailcyclenet.scorer.triplet.make_triplet`, carrying
                `good`/`bad`/`anchor` as `(views, coords, cgroup)`, `kpt_ids`, `anchor_label`,
                optional `occlusion`, and `reuse_scene_for_anchor`.
        Outputs: (scores [N,3], precision [N,3], labels [N,3]) with N = b*k and columns
            (good, bad, anchor).
        Side effects: none beyond `score`'s temporary stashes.
        """
        gv, gc, gcg = trip['good']
        _, bc, bcg = trip['bad']
        av, ac, acg = trip['anchor']
        occ = trip.get('occlusion')
        kpt_ids = trip['kpt_ids']

        gvn = self._normalize_views(gv)
        sf = self.encode_scene(gvn)
        good_s, good_p = self.score(gvn, sf, gc, gcg, kpt_ids, occlusion=occ)
        bad_s, bad_p = self.score(gvn, sf, bc, bcg, kpt_ids, occlusion=occ)
        if trip.get('reuse_scene_for_anchor'):
            avn, asf = gvn, sf
        else:
            avn = self._normalize_views(av)
            asf = self.encode_scene(avn)
        anc_s, anc_p = self.score(avn, asf, ac, acg, kpt_ids, occlusion=occ)

        scores = torch.stack([good_s, bad_s, anc_s], dim=-1).reshape(-1, 3)
        precision = torch.stack([good_p, bad_p, anc_p], dim=-1).reshape(-1, 3)
        labels = torch.tensor([1.0, -1.0, float(trip['anchor_label'])],
                              device=scores.device).expand_as(scores)
        return scores, precision, labels


def build_scorer(model_cfg: dict, n_keypoints: int, **scorer_kwargs) -> PoseScorer:
    """`[model]` -> a `PoseScorer`, with every `build_model` refusal shared rather than copied.

    Inputs: model_cfg -- the `[model]` block; n_keypoints -- the registry size;
            scorer_kwargs -- the scorer's own head settings (`pool_num_heads`, `score_hidden`,
            `use_precision`). These are FRESH parameters at warm start, so they are not part of
            the shape contract that must match a pose checkpoint.
    Outputs: a built `PoseScorer`.
    Side effects: prints the query-encoder summary via the parent constructor.
    """
    return build_model(model_cfg, n_keypoints, cls=PoseScorer, **scorer_kwargs)
