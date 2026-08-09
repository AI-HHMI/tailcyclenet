"""The pose model: posetail's `TrackerEncoder` with a pose-shaped query and a per-frame anchor.

`TrackerEncoder` tracks arbitrary points given a query position. A pose model has K named
keypoints and, at deployment, no ground truth to query with. Two changes make it a pose
estimator, and `forward` is the only method overridden:

1. **The query is derived, not given.** In 3D every keypoint is queried at a query-free scene
   point back-projected from the cameras' own crop centres (`scene_center`); in 2D at the crop
   centre. A per-keypoint prior, when present, replaces it. Keypoint identity rides in a
   dedicated fusion term (see `query_encoder.py`), which is what tells the K queries apart.

2. **The 3D residual is re-anchored on each frame's own triangulation.** The library anchors a
   gridresid residual on the query position and holds it for the whole window; with a scene-centre
   query that would make the residual explain the entire centre-to-keypoint offset from a fixed
   point. Measured within-session at 2.07 mm (frame-0 anchor) vs 1.37 mm (per-frame) -- the
   largest single architectural effect in the project it came from.

There is ONE architecture switch, `query`:

- `"prior"`  -- a per-keypoint prior (`kpt_prior`), the previous window's pose at deployment,
   with a learned no-query token wherever a keypoint has none. This is posetail-pose's
   `w9_honest` with the instance-anchor machinery removed rather than defaulted off.
- `"none"`   -- query-free. No prior is read at all; every keypoint is queried at the derived
   point and told only its identity.
"""
import torch
from einops import einsum, repeat

from posetail.posetail.cube import (from_homogeneous, get_camera_scale, to_homogeneous,
                                    undistort_points)
from posetail.posetail.tracker_encoder import TrackerEncoder

from .query_encoder import PoseQueryEncoder

QUERY_MODES = ('prior', 'none')


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
    def __init__(self, *args, n_keypoints, query='prior', **kwargs):
        assert query in QUERY_MODES, f'query must be one of {QUERY_MODES}, got {query!r}'
        super().__init__(*args, **kwargs)
        self.n_keypoints = n_keypoints
        self.query = query

        # Replace the stock query encoder with the pose one, built from the same kwargs the
        # parent used so it is a drop-in.
        old = self.query_encoder
        self.query_encoder = PoseQueryEncoder(
            embed_dim=old.embed_dim, decoder_dim=old.decoder_dim, n_frames=old.n_frames,
            corr_radius=old.corr_radius, max_freq=old.max_freq, patch_size=old.patch_size,
            use_volume_embedding=old.use_volume_embedding,
            principal_point_embedding=old.principal_point_embedding,
            intrinsic_embedding=old.intrinsic_embedding,
            occlusion_embedding=old.occlusion_embedding,
            time_embed_mode=old.time_embed_mode,
            n_keypoints=n_keypoints)

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
        self._query_ok_all = (torch.isfinite(prior).all(-1) if prior is not None
                              else torch.zeros((B, K), dtype=torch.bool, device=device))
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
        if out.get('3d_pred_triangulate') is None:
            # 3D single-view: no triangulation to re-anchor onto, so the library's query-anchored
            # residual stands. This is the path `prob_2d_only` trains.
            return out
        return _reanchor_per_frame(out, coords_q)


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
    src = torch.where(torch.isfinite(src), src, out['3d_pred_rays'])

    residual = out['3d_pred_cams_direct'] - anchor[:, None, :, :][None]
    new_direct = src.detach()[None] + residual
    conf = torch.softmax(out['conf_3d'], dim=0)
    coords_pred = torch.einsum('cbtnr,cbtn->btnr', new_direct, conf)

    out = dict(out)
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
    cfg.pop('n_keypoints', None)          # derived from the registry, never configured

    # The library DEFAULTS to 'direct', so omitting the key is as dangerous as setting it wrong.
    # `_reanchor_per_frame` recovers the world residual as (3d_pred_cams_direct - query), which
    # is only true where the library actually anchored the residual on the query
    # (tracker_encoder.py:670 -- 'residual' and 'gridresid' only). Under any other mode it
    # subtracts the query from something that was never added to it and adds the difference to
    # every frame's triangulation: the re-anchoring runs backwards, silently.
    output_mode = cfg.setdefault('output_mode', 'gridresid')
    assert output_mode == 'gridresid', (
        f'output_mode = {output_mode!r} is not supported. model._reanchor_per_frame recovers '
        'the residual as (3d_pred_cams_direct - query), which the library only makes true for '
        '"gridresid"; any other mode re-anchors a prediction that was never query-anchored and '
        'silently corrupts coords_pred. Set output_mode = "gridresid".')

    # Upstream this selects the CLASS (train_utils.py:439 -- 'encoder' -> TrackerEncoder,
    # 'tapnext' -> TrackerTapNext). Here it is stored and never read, so anything else would be
    # accepted and then quietly ignored -- and TrackerTapNext is not moving-cam-safe anyway.
    mode_3d = cfg.setdefault('mode_3d', 'encoder')
    assert mode_3d == 'encoder', (
        f'mode_3d = {mode_3d!r} is not supported: build_model always constructs a '
        'PoseTrackerEncoder, so any other value would be silently ignored rather than honoured.')

    return PoseTrackerEncoder(n_keypoints=n_keypoints, query=query, **cfg)
