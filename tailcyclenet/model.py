"""The pose model: posetail's `TrackerEncoder` with a pose-shaped query and a per-frame anchor.

`TrackerEncoder` tracks arbitrary points given a query position. A pose model has K named
keypoints and, at deployment, no ground truth to query with. Two changes make it a pose
estimator, and `forward` is the only method overridden:

1. **The query is derived, not given.** In 3D every keypoint is queried at a query-free scene
   point back-projected from the cameras' own crop centres (`scene_center`); in 2D at the crop
   centre. A per-keypoint prior, when present, replaces it. Keypoint identity rides in a
   dedicated fusion term (see `query_encoder.py`), which is what tells the K queries apart.

2. **What the 3D residual is an offset FROM is a switch.** The library reconstructs
   `world = query + R @ residual` with one anchor for the whole window. That is right when the
   query is a genuine prior and wrong when it is the derived scene centre -- identical for every
   keypoint, so the residual would have to carry the whole centre-to-keypoint offset.

Two switches, and they are orthogonal:

`query` -- whether a prior is supplied:

- `"prior"`  -- a per-keypoint prior (`kpt_prior`), the previous window's pose at deployment,
   with a learned no-query token wherever a keypoint has none. This is posetail-pose's
   `w9_honest` with the instance-anchor machinery removed rather than defaulted off.
- `"none"`   -- query-free. No prior is read at all; every keypoint is queried at the derived
   point and told only its identity.

`gridresid_offset` -- what the residual is measured from:

- `"query"`         -- the library's NATIVE structure, kept per keypoint only where a real prior
   anchored it; every other keypoint falls back to that frame's own triangulation, and the direct
   head is supervised on query points only. See `_query_anchored`.
- `"triangulated"`  -- the residual is recovered and re-added to EACH frame's own triangulation,
   for every keypoint. Measured 2.07 -> 1.37 mm within-session in posetail-pose, where it was the
   largest single architectural effect; it exists to rescue the fixed scene-centre anchor, so it
   is the sensible pairing for `query = "none"`. See `_reanchor_per_frame`.
"""
from contextlib import contextmanager

import torch
from einops import einsum, repeat

from posetail.posetail.cube import (from_homogeneous, get_camera_scale, to_homogeneous,
                                    undistort_points)
from posetail.posetail.tracker_encoder import TrackerEncoder

from .query_encoder import PoseQueryEncoder, WideQueryEncoder

QUERY_MODES = ('prior', 'none')
QUERY_ENCODERS = ('pose', 'wide')
# What the gridresid residual is an offset FROM. See `_query_anchored` / `_reanchor_per_frame`.
GRIDRESID_OFFSETS = ('query', 'triangulated')


# ----------------------------------------------------------------------------------------------
# the query-free scene point
# ----------------------------------------------------------------------------------------------

def scene_center(camera_group):
    """The 3D point every camera's crop centre looks at. (3,) float32.

    The model needs a query position, and at deployment there is no ground truth to supply one.
    A stored world constant will not do either: the loader applies a random world rotation to
    points AND cameras every sample, so the world gauge changes per item and anything computed
    from a fixed world coordinate would be in the wrong frame.

    So derive it from the CAMERAS, which are rotated by that same gauge and are the only thing
    available at inference anyway: each camera's crop centre back-projects to a ray, and the
    point closest to all of them is the scene centre. Gauge-correct by construction, query-free,
    and identical in training and deployment -- at inference the crop comes from the detector,
    so the rays converge on whatever it boxed.

    Feeding this as `coords` to the stock forward makes the library derive every remaining
    scene scalar (`cube_scale`, `scene_radius`) correctly by itself.

    ON A MOVING RIG, every FRAME contributes its own ray. The query anchor is structurally one
    point per keypoint for the whole window, so the per-frame rays have to be reduced somehow --
    and taking frame 0 alone put the anchor where the rig started rather than where it looked,
    which for a rig that travels during the window is a different place. Solving over all
    (camera, frame) rays at once is the same least-squares problem with more rows.
    """
    origins, dirs = [], []
    for cam in camera_group:
        px = (cam['size'].to(torch.float64) / 2.0).reshape(1, 2)
        und = undistort_points(cam, px)
        ray_cam = torch.cat([und[0], und.new_ones(1)])
        ext = cam['ext'] if cam['ext'].ndim == 3 else cam['ext'][None]
        centre = cam['center'] if cam['center'].ndim == 2 else cam['center'][None]
        for t in range(ext.shape[0]):
            d = ext[t, :3, :3].to(torch.float64).t() @ ray_cam
            dirs.append(d / d.norm())
            origins.append(centre[t].to(torch.float64))
    origins, dirs = torch.stack(origins), torch.stack(dirs)

    # Least-squares point minimising distance to a set of lines:
    #   sum_i ||(I - d_i d_i^T)(x - c_i)||^2  ->  (sum_i P_i) x = sum_i P_i c_i
    eye = torch.eye(3, dtype=origins.dtype, device=origins.device)
    P = eye - dirs[:, :, None] * dirs[:, None, :]
    A = P.sum(0)
    b = torch.einsum('cij,cj->i', P, origins)
    # Ridge term keeps a near-degenerate rig from producing a wild centre; ~1e-6 of the trace,
    # so it never moves a good solve.
    A = A + eye * (1e-6 * torch.diagonal(A).abs().sum().clamp_min(1.0))
    return torch.linalg.solve(A, b).to(torch.float32)


# ----------------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------------

class PoseTrackerEncoder(TrackerEncoder):
    def __init__(self, *args, n_keypoints, query='prior', query_encoder='pose',
                 gridresid_offset='query', query_terms=None, **kwargs):
        assert query in QUERY_MODES, f'query must be one of {QUERY_MODES}, got {query!r}'
        assert query_encoder in QUERY_ENCODERS, \
            f'query_encoder must be one of {QUERY_ENCODERS}, got {query_encoder!r}'
        assert gridresid_offset in GRIDRESID_OFFSETS, \
            f'gridresid_offset must be one of {GRIDRESID_OFFSETS}, got {gridresid_offset!r}'
        super().__init__(*args, **kwargs)
        self.n_keypoints = n_keypoints
        self.query = query
        self.gridresid_offset = gridresid_offset
        # None -> encode per forward, the normal path. `share_scene` swaps in a dict.
        self._shared_scene = None

        # Replace the stock query encoder, built from the same kwargs the parent used so it is a
        # drop-in either way.
        old = self.query_encoder
        if query_encoder == 'pose':
            self.query_encoder = PoseQueryEncoder(
                embed_dim=old.embed_dim, decoder_dim=old.decoder_dim, n_frames=old.n_frames,
                corr_radius=old.corr_radius, max_freq=old.max_freq, patch_size=old.patch_size,
                use_volume_embedding=old.use_volume_embedding,
                principal_point_embedding=old.principal_point_embedding,
                intrinsic_embedding=old.intrinsic_embedding,
                occlusion_embedding=old.occlusion_embedding,
                time_embed_mode=old.time_embed_mode,
                n_keypoints=n_keypoints)
        else:
            # `wide`'s two query terms DEFAULT to following `query`. Under `query = "none"` the
            # prior is never read, so `_query_ok` is all-False for the whole run and
            # `_sub_unprompted` swaps both terms for their learned no-query token on every query --
            # two constant vectors and two dead gate inputs.
            #
            # `query_terms` overrides the pair, which is what the `j4_prior` recipe needs (pos but
            # no patch). The one combination that stays unrepresentable is the trap: with BOTH off,
            # `wide` ignores `query_coords` entirely, so a declared prior cannot reach the encoder
            # at all. That is how six posetail-pose configs declared an anchor, trained, and were
            # reported as anchored arms whose anchor was a literal no-op.
            terms = dict.fromkeys(('query_pos_embedding', 'query_patch_embedding'),
                                  query == 'prior')
            terms.update(query_terms or {})
            if query == 'prior':
                assert any(terms.values()), (
                    'query = "prior" with both query terms off: `wide` would ignore `query_coords` '
                    'entirely and the prior would be a no-op. Set at least one of '
                    'query_pos_embedding / query_patch_embedding, or use query = "none".')
            else:
                assert not any(terms.values()), (
                    f'query = "none" supplies no prior, so {[k for k, v in terms.items() if v]} '
                    'would be constant no-query tokens feeding dead gate inputs for the whole run.')
            self.query_encoder = WideQueryEncoder(
                dim=self.latent_dim, embed_dim=old.embed_dim, decoder_dim=old.decoder_dim,
                n_frames=old.n_frames, max_freq=old.max_freq, patch_size=old.patch_size,
                principal_point_embedding=old.principal_point_embedding,
                intrinsic_embedding=old.intrinsic_embedding,
                time_embed_mode=old.time_embed_mode, n_keypoints=n_keypoints, **terms)
            print(f'query encoder: wide, dim {self.latent_dim}, '
                  f'{self.query_encoder.n_fusion_terms} terms '
                  f'({", ".join(self.query_encoder.term_names())})')

    def _forward_window(self, views_norm, *args, **kwargs):
        """Reuse one scene encode across several decodes over the SAME pixels.

        Inside `share_scene`, `scene_features` is computed on the first call and reused after. Only
        the encode is shareable: the decode also needs `cube_scale` / `scene_center` /
        `scene_radius`, which `forward` derives from `coords_q` and which the prior changes -- so
        they arrive per call in `*args` and are passed through untouched.

        Outside the context, and whenever `kpt_chunk` is set, this delegates to the library
        verbatim. That is deliberate: training and chunked inference must not route through a
        reimplementation of a private method. Gotcha #2 -- `scene_features=` was dropped from
        `TrackerEncoder.forward` in 0.3.x, so this seam is the sanctioned way to share an encode.
        """
        if self._shared_scene is None or kwargs.get('kpt_chunk'):
            return super()._forward_window(views_norm, *args, **kwargs)
        if 'f' not in self._shared_scene:
            self._shared_scene['f'] = self.scene_encoder(views_norm)
        kwargs.pop('kpt_chunk', None)
        return self._decode_from_scene(self._shared_scene['f'], views_norm, *args, **kwargs)

    def _decode_from_scene(self, scene_features, views_norm, coords, *args, **kwargs):
        """Hand the query encoder the keypoint slice belonging to THIS chunk.

        `_forward_window` slices `coords[:, k0:k1]` when `kpt_chunk` is set and calls this once
        per chunk, in order, without passing the bounds down -- so the offset is a running
        cursor. `forward` asserts it lands exactly on K, which turns a future reordering into a
        failure rather than one chunk's queries silently wearing another chunk's identities.

        posetail-pose avoided half of this by routing ids through the `occlusion` channel, which
        the library slices for free (`posetail_pose/model.py:718`); gotcha #5 deliberately freed
        that channel here. Its `_query_ok` had this bug unfixed. Both halves are handled here.
        """
        k0, n = self._kpt_cursor, coords.shape[1]
        self.query_encoder._kpt_ids = self._kpt_ids_all[:, k0:k0 + n]
        self.query_encoder._query_ok = self._query_ok_all[:, k0:k0 + n]
        self._kpt_cursor = k0 + n
        return super()._decode_from_scene(scene_features, views_norm, coords, *args, **kwargs)

    def forward(self, views, kpt_ids, camera_group, mode, kpt_prior=None, prompt_time=None,
                kpt_chunk=None):
        """
        Args:
            views: list of (B,T,H,W,3) float32 in [0,1], one per camera
            kpt_ids: (B,K) long -- GLOBAL registry ids, not positions
            camera_group: list of posetail camera dicts, one per view
            mode: '2d' | '3d'. A property of the SAMPLED SESSION, not of the run -- one training
                run mixes both, so it cannot be a config field.
            kpt_prior: (B,K,R) or None. Per-keypoint prior; NaN where a keypoint has none.
                Ignored entirely when `query == "none"`.
            prompt_time: (B,K) int or None -- the frame each prior describes.
            kpt_chunk: decode the keypoints in slices of this size, reusing one scene encode.
                INFERENCE ONLY -- the library drops the loss-only `grid` tensors when chunking
                (`_CHUNK_SKIP`), so `_reanchor_ce_target` correctly no-ops. Ignored when
                `output_mode == 'gridnorm'`, whose per-camera gauge solve couples points.
        """
        assert mode in ('2d', '3d'), mode
        B, K = kpt_ids.shape
        n_cams = len(views)
        device = views[0].device
        # The loader hands over uint8 -- 4x fewer bytes to queue from the worker and to pin. The
        # library's `transform_norm` wants [0,1] floats, so the divide happens HERE, after the
        # transfer, where it is free. One seam for train, infer, eval and the tests; a no-op for
        # anything that already arrives as float.
        views = [v.float().div_(255) if v.dtype == torch.uint8 else v for v in views]
        prior = None if self.query == 'none' or kpt_prior is None else kpt_prior.to(device).float()

        if mode == '2d':
            # ONE camera, and the query is a PIXEL. The library asserts len(views) == 1 for R==2
            # and routes the forward through its separate 2D head bank (`mode_idx = 0`), which
            # exists at full size in the base checkpoint and so is pretrained, not fresh.
            assert n_cams == 1, f'2D mode is single-camera; got {n_cams}'
            size = camera_group[0]['size'].to(device).float()
            coords_q = (size * 0.5).view(1, 1, 2).expand(B, K, 2).contiguous()
        else:
            assert n_cams >= 1, 'need at least one camera'
            coords_q = scene_center(camera_group).to(device).view(1, 1, 3).expand(
                B, K, 3).contiguous()

        if prior is not None:
            assert prior.shape == (B, K, coords_q.shape[-1]), \
                f'kpt_prior {tuple(prior.shape)} != {(B, K, coords_q.shape[-1])}'
            coords_q = torch.where(torch.isfinite(prior), prior, coords_q).contiguous()

        # The query-validity mask comes from the PRIOR's own finiteness, not from `coords_q` --
        # which has already had absent priors replaced by the derived point. This is what the
        # no-query tokens key off.
        #
        # Stashed at FULL K here; `_decode_from_scene` below hands the query encoder the slice
        # belonging to the chunk being decoded.
        query_ok = (torch.isfinite(prior).all(-1) if prior is not None
                    else torch.zeros((B, K), dtype=torch.bool, device=device))
        self._query_ok_all = query_ok      # local copy survives the `finally` clear below
        self._kpt_ids_all = kpt_ids.to(device).long()
        self._kpt_cursor = 0

        # The frame the prompt describes. INT and CLAMPED: the library uses this as a
        # `torch.gather` index on the learnable-scale and gridnorm paths, so an out-of-range or
        # floating value is an index error rather than a soft degradation. A prompt from BEFORE
        # this window (deployment staleness) cannot be expressed and clamps to 0 -- the patch has
        # to be sampled from a frame that exists.
        if prompt_time is None or self.query == 'none':
            qt = torch.zeros((B, K), dtype=torch.int32, device=device)
        else:
            T_win = int(views[0].shape[1])
            qt = prompt_time.to(device).round().long().clamp_(0, T_win - 1).to(torch.int32)
            assert qt.shape == (B, K), f'prompt_time {tuple(qt.shape)} != {(B, K)}'

        try:
            out = super().forward(views, coords_q, camera_group, query_times=qt,
                                  occlusion=None, kpt_chunk=kpt_chunk)
            # AFTER the call, not in the `finally`: an assertion there would mask whatever real
            # exception got us out. Every keypoint must have been decoded exactly once.
            assert self._kpt_cursor == K, (
                f'{self._kpt_cursor} of {K} keypoints were decoded; _decode_from_scene is not '
                'being called once per chunk in order, so the id slices are unreliable')
        finally:
            # Always clear: a leaked stash would apply one item's ids to the next forward,
            # silently, and only on multi-call paths like eval and the windowed driver.
            self._query_ok_all = self._kpt_ids_all = None
            self._kpt_cursor = 0
            self.query_encoder._query_ok = None
            self.query_encoder._kpt_ids = None

        if mode == '2d':
            # No re-anchoring in 2D and nothing to re-anchor onto -- triangulation is None at
            # one camera, and the 2D grid head already decodes ABSOLUTE pixel bins. `coords_pred`
            # is rebound to the 2D prediction so every downstream consumer can say "the
            # prediction" without a mode branch. At R==2 it is in PIXELS, not mm.
            out = dict(out)
            out['coords_pred'] = out['2d_pred'][0]
            return out
        if self.gridresid_offset == 'query':
            return _query_anchored(out, query_ok)
        if out.get('3d_pred_triangulate') is None:
            # 3D single-view: nothing to re-anchor onto, so the library's query-anchored residual
            # stands. This is the path `prob_2d_only` trains.
            return out
        return _reanchor_per_frame(out, coords_q)


@contextmanager
def share_scene(model):
    """Encode the scene ONCE for every forward inside the block. The pixels must be identical.

    The val loop runs two forwards over one window -- prior-free, then re-queried at the model's own
    frame-0 prediction -- and the video encoder is frozen, so encoding twice is pure waste. It is
    also the bulk of the forward: johnson-mouse's 16-camera eval ran ~200 s per val step.

    The caller owns the "identical pixels" precondition, which is why the scope is one window rather
    than one eval: nothing here checks that `views_norm` matches between calls, because the tensors
    are large and comparing them would cost what the sharing saves.

    Only the encode is shared. `cube_scale`, `scene_center` and `scene_radius` are derived from
    `coords_q`, which the prior changes, so the decode still runs per forward.
    """
    prev = model._shared_scene
    model._shared_scene = {}
    try:
        yield model
    finally:
        model._shared_scene = prev


def _rays_fallback(out):
    """The conf-weighted MEAN of the per-camera ray points -- the honest version of `3d_pred_rays`.

    `3d_pred_rays` is a weighted SUM, not a mean: `conf_pred_2d` is an unnormalized sigmoid
    (`tracker_encoder.py:554`) and line 637 einsums it straight over the camera axis with no
    division. So the library's own ray point is off by a factor of `sum_c conf_c`, which is ~C/2 at
    C cameras and ~0.5 at one -- a single-camera "prediction" that lands halfway to the world
    origin. Nothing pins that factor either: `coords_loss_rays_weight = 0` ships and the only
    pressure is `coords_loss_rays_reproj` at 0.05/16, one weak term asked to satisfy every camera
    count in `cams_to_sample` at once.

    Used wherever a triangulation is missing or degenerate, so the scale error would otherwise be
    inherited by the substituted point.
    """
    w = torch.sigmoid(out['conf_pred_2d'])                          # (cams,b,t,n)
    num = einsum(out['3d_pred_cams_rays'], w, 'c b t n r, c b t n -> b t n r')
    return num / w.sum(0)[..., None].clamp_min(1e-6)


def _query_anchored(out, query_ok):
    """gridresid is an OFFSET FROM THE QUERY POINT, so honour it only where the query is real.

    The library reconstructs `world = query_world + R_{ray->world} @ residual`
    (`tracker_encoder.py:736`) with ONE anchor for the whole window. That is the right structure
    when the query is a genuine prior on the animal. It is the wrong one when the query is the
    derived scene centre, which is identical for every keypoint and carries no information --
    there the residual would have to explain the entire centre-to-keypoint offset from a fixed
    point, and it is not a pose prediction so much as a memorised mean.

    So the prediction is the query-anchored residual WHERE a prior exists, and each frame's own
    triangulation everywhere else. Substituting the DETACHED triangulation is also what gates the
    loss: `coords_loss_direct*` then compares a constant against the target at those points, which
    contributes exactly zero gradient, so the direct head is supervised on query points only --
    without forking posetail's `TotalLoss`.

    THIS REPLACES per-frame re-anchoring. That mechanism (residual recovered as
    `3d_pred_cams_direct - query`, re-added to every frame's triangulation) existed to rescue the
    fixed scene-centre anchor and measured 2.07 -> 1.37 mm within-session in posetail-pose. It is
    deliberately gone: with a real prior the native anchor is meaningful, and with no prior the
    prediction is now the triangulation outright rather than a residual on a meaningless anchor.
    """
    tri = out.get('3d_pred_triangulate')
    out = dict(out)
    if tri is None:
        # 3D SINGLE-VIEW: no triangulation exists, so the back-projected rays are the only anchor
        # there is. This USED to hand `loss_kpt_mask` up and let `run_batch` NaN the whole 3D
        # target, which killed the step outright whenever no keypoint had a prior -- and
        # `cams_to_sample = [1, 8]` draws exactly that on 1/8 of 3D items, so with
        # `prompt_dropout = 0.5` it silently binned ~6% of every `prior` arm's steps (measured:
        # johnson 6.20% excess skips over the matched `none` arm, predicted 6.25%).
        sub = _rays_fallback(out)
    else:
        # Repair FIRST, so the loss and the metric see ONE tensor and a degenerate solve cannot
        # silently reduce coverage. `get_mpjpe` credits a non-finite prediction as perfect (nansum
        # numerator, full denominator), which is exactly how a wrong comparison gets published.
        bad = ~torch.isfinite(tri).all(-1)
        # A NON-FINITE ENTRY POISONS THE SOLVE'S BACKWARD EVEN AT ZERO GRADIENT, and `torch.where`
        # cannot stop it. The `where` below routes 0 into the bad entries, but
        # `triangulate_simple_batch_reg`'s `torch.linalg.solve` (cube.py:299) then solves against
        # its own NaN factorisation and RETURNS NaN -- which reaches every upstream parameter and
        # gets the whole step thrown away by the grad-norm guard in `train.py`, counted as an
        # unattributed `skipped`. `nan_to_num` here does NOT help: the NaN is created in the
        # backward, downstream of anything applied to the forward value. Per-entry detach cannot
        # help either -- the graph node is the whole batched solve. So on the rare degenerate step
        # the window's triangulation supervision goes gradient-free and every other term keeps
        # training. The forward value is unchanged in every case.
        if not torch.isfinite(tri).all():
            tri = tri.detach()
        sub = torch.where(torch.isfinite(tri), tri, _rays_fallback(out))
        # NOT written back on the single-view path: this key drives `coords_loss_triangulate_reproj`
        # at weight 2.0, so pointing it at the rays would reweight the rays supervision by 40x.
        out['3d_pred_triangulate'] = sub
        # WHERE THE REPAIR ACTUALLY FIRED. The substitution is silent by design -- one tensor for the
        # loss and the metric -- but `--anchor carry` now SEEDS the next window from this tensor, and
        # a seed taken from `_rays_fallback` is a point no camera claimed rather than a triangulated
        # one. Recorded, not gated: the caller decides what a degenerate frame is worth.
        out['tri_degenerate'] = bad

    m = query_ok[None, :, None, :, None]                    # -> (cams, b, t, n, r)
    direct = torch.where(m, out['3d_pred_cams_direct'], sub.detach()[None])
    conf = torch.softmax(out['conf_3d'], dim=0)
    coords_pred = torch.einsum('cbtnr,cbtn->btnr', direct, conf)
    out['3d_pred_cams_direct'] = direct
    out['3d_pred_direct'] = coords_pred
    out['coords_pred'] = coords_pred

    if not bool(query_ok.any()) and out.get('grid') is not None:
        # Nothing to supervise the 3D grid CE with -- every point is triangulated. Kill THAT term
        # and nothing else. Popping `grid` (what this used to do) overshoots badly, because
        # `losses.py:680` gates on `'grid' in outputs` and TWO other things live behind it:
        #   - `depth_softmax` (losses.py:756-772), weight 1.5, the LARGEST CE term in w9.toml,
        #     whose target (`:769`) is `log(depths_true / (cube_scale * f_eff * sdep))` -- nothing
        #     to do with the query anchor.
        #   - `f_eff` itself (losses.py:458), so the depth regression Huber silently reswitches
        #     its normaliser mid-run and the arm trains two different depth losses in alternation.
        # With `prompt_dropout = 0.5` that fired on ~half the steps of every `prior` arm and never
        # on a `none` arm, contaminating the shipped sweep delta.
        #
        # A non-finite anchor is the library's own off switch: `anchor_local` is read at exactly
        # one place (`losses.py:738`, `target_3d = (p_raylocal - anchor_local) / denom_resid`), and
        # `grid_softmax_loss` (`:45-52`) drops non-finite targets and returns 0 when all are
        # dropped. In a PARTIALLY prompted window the CE stays on and masks itself the same way.
        grid = dict(out['grid'])
        grid['anchor_local'] = torch.full_like(grid['anchor_local'], float('nan'))
        out['grid'] = grid
    return out


def _reanchor_per_frame(out, anchor):
    """Re-anchor the 3D residual on EACH frame's own triangulation.

    The anchor is recoverable from the outputs -- `3d_pred_cams_direct = query + R @ residual` --
    so subtracting the query returns the world-space residual, which is then added to each
    frame's own (detached) triangulation. No change to posetail is needed.
    """
    src = out['3d_pred_triangulate']
    # Repair FIRST, so the loss and the metric see ONE tensor and a degenerate solve cannot
    # silently reduce coverage. `get_mpjpe` credits a non-finite prediction as perfect (nansum
    # numerator, full denominator), which is exactly how a wrong comparison gets published.
    bad = ~torch.isfinite(src).all(-1)
    if not torch.isfinite(src).all():
        src = src.detach()                    # see `_query_anchored`: the solve's backward NaNs
    src = torch.where(torch.isfinite(src), src, _rays_fallback(out))
    out = dict(out)
    out['tri_degenerate'] = bad                    # see `_query_anchored`

    residual = out['3d_pred_cams_direct'] - anchor[:, None, :, :][None]
    new_direct = src.detach()[None] + residual
    conf = torch.softmax(out['conf_3d'], dim=0)
    coords_pred = torch.einsum('cbtnr,cbtn->btnr', new_direct, conf)

    out['3d_pred_cams_direct'] = new_direct
    out['3d_pred_direct'] = coords_pred
    out['coords_pred'] = coords_pred
    out['3d_pred_triangulate'] = src
    return _reanchor_ce_target(out, src)


def _reanchor_ce_target(out, src):
    """Move the 3D cross-entropy target onto the SAME anchor the outputs now use.

    Without this, `grid['anchor_local']` still describes the query-anchored task, so with
    `coords_softmax_3d_weight = 0.4` roughly 40% of the 3D objective would train fixed-anchor
    propagation while every metric and reprojection term sees per-frame refinement -- two
    objectives pulling in different directions.

    float64 is mandatory, not defensive: the library does this einsum in float64 precisely so
    `(p_raylocal - anchor_local)` cancels exactly. In float32 the ray-local coordinates are large
    and nearly equal, and the difference annihilates.
    """
    grid = out.get('grid')
    if grid is None or grid.get('anchor_local') is None or grid.get('rays_c') is None:
        return out
    n_cams = grid['anchor_local'].shape[0]
    aw = repeat(src, 'b t n r -> cams b t n r', cams=n_cams)
    anchor_local = from_homogeneous(einsum(
        grid['rays_c'].to(torch.float64), to_homogeneous(aw.to(torch.float64)),
        'cams x r, cams b t n r -> cams b t n x'))
    grid = dict(grid)
    grid['anchor_local'] = anchor_local.to(grid['anchor_local'].dtype).detach()
    out['grid'] = grid
    return out


def build_model(model_cfg: dict, n_keypoints: int) -> PoseTrackerEncoder:
    """`[model]` splatted into the constructor, with this repo's two keys pulled out first.

    Two of the library's options are checked HERE rather than in the constructor, because both
    are wrong in a way that produces numbers instead of exceptions and the message needs to name
    the config key rather than surface later as bad accuracy.
    """
    cfg = dict(model_cfg)
    query = cfg.pop('query', 'prior')
    # `query_encoder` picks the MODULE, `query` picks whether a prior is supplied to it. `wide`'s
    # two query terms are DERIVED from `query`, not configured -- see PoseTrackerEncoder.__init__.
    # So `query_encoder = "wide"` with `query = "none"` is exactly golden's j3 encoder, with no
    # third key to get wrong.
    enc = cfg.pop('query_encoder', 'pose')
    # NO DEFAULT, DELIBERATELY. This key decides what the 3D residual is measured from, and the two
    # values are different architectures sharing one set of tensor shapes -- so a checkpoint trained
    # under one and loaded under the other produces numbers rather than an exception.
    #
    # That is not hypothetical. `runs/3dpop-prior` trained under unconditional per-frame
    # re-anchoring, finished nine hours before `bcbfbc1` replaced it with the query-anchored
    # residual, and has no `gridresid_offset` in its config -- so every later run of it inferred
    # `world = prior + residual` from weights that learned `world = tri_t + residual_t`. Under
    # `--anchor carry` that turns the prior into a static anchor the residual head never saw, and
    # the pose lags the animal until the bounds mask drops the prior and the fallback snaps it back
    # to the triangulation it was trained on. A default of `'query'` is what made that silent.
    #
    # A run folder from before this check has to be told which one it was, per
    # `load_run(model_overrides=...)` / `scripts/infer.py --gridresid-offset`.
    if 'gridresid_offset' not in cfg:
        raise KeyError(
            "[model].gridresid_offset is required, not defaulted: 'query' anchors the 3D residual "
            "on the query point for the whole window and 'triangulated' re-anchors it on each "
            'frame\'s own triangulation. Same shapes, same losses, different architecture -- so a '
            'checkpoint trained under one and loaded under the other is wrong without ever '
            'raising. Set it in the config, or for a run folder written before this key existed '
            'pass --gridresid-offset (scripts/infer.py) / load_run(model_overrides=...) to state '
            'which one those weights were trained with.')
    offset = cfg.pop('gridresid_offset')
    # `wide`'s two query terms DEFAULT to `query`, and setting one is the `j4_prior` recipe (pos,
    # no patch). Only `wide` reads them -- `pose` derives its ten terms from the query directly --
    # so naming one alongside `pose` is a silent no-op and must not be silent.
    query_terms = {k: bool(cfg.pop(k)) for k in
                   ('query_pos_embedding', 'query_patch_embedding') if k in cfg}
    assert not (query_terms and enc != 'wide'), (
        f'{sorted(query_terms)} only apply to query_encoder = "wide"; {enc!r} would ignore them.')
    cfg.pop('n_keypoints', None)          # derived from the registry, never configured

    # The library DEFAULTS to 'direct', so omitting the key is as dangerous as setting it wrong.
    # BOTH `gridresid_offset` paths need the output to be query-anchored in the first place, and
    # the library only adds `query_world` for 'residual' and 'gridresid' (tracker_encoder.py:670).
    # Under 'direct' or 'grid' the prediction never depended on the query, so `"query"` would gate
    # away a perfectly good absolute output at every unprompted point, and `"triangulated"` would
    # subtract the query from something it was never added to and re-add the difference to every
    # frame -- running the re-anchoring backwards, silently.
    output_mode = cfg.setdefault('output_mode', 'gridresid')
    assert output_mode == 'gridresid', (
        f'output_mode = {output_mode!r} is not supported: both gridresid_offset modes assume the '
        'library anchored the residual on the query, which it only does for "gridresid". Under '
        'an absolute mode "query" discards a valid prediction at every unprompted point and '
        '"triangulated" re-anchors a prediction that was never query-anchored. '
        'Set output_mode = "gridresid".')

    # Upstream this selects the CLASS (train_utils.py:439 -- 'encoder' -> TrackerEncoder,
    # 'tapnext' -> TrackerTapNext). Here it is stored and never read, so anything else would be
    # accepted and then quietly ignored -- and TrackerTapNext is not moving-cam-safe anyway.
    mode_3d = cfg.setdefault('mode_3d', 'encoder')
    assert mode_3d == 'encoder', (
        f'mode_3d = {mode_3d!r} is not supported: build_model always constructs a '
        'PoseTrackerEncoder, so any other value would be silently ignored rather than honoured.')

    return PoseTrackerEncoder(n_keypoints=n_keypoints, query=query, query_encoder=enc,
                              gridresid_offset=offset, query_terms=query_terms, **cfg)
