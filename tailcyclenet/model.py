"""The pose model: posetail's `TrackerEncoder` with a pose-shaped query and a per-frame anchor.

(1) The query is DERIVED, not given: query-free keypoints are queried at a scene point
back-projected from the cameras' own crop centres (2D: the crop centre), replaced by a
per-keypoint prior when one is present (`query`: `"prior"` | `"none"`); identity rides in a
dedicated fusion term. (2) What the 3D residual is an offset FROM is a switch
(`gridresid_offset`): `"query"` keeps the native anchor where a real prior anchored it,
`"triangulated"` re-adds the residual to each frame's own triangulation.
"""
from contextlib import contextmanager
import threading

import torch
from einops import einsum, repeat

from posetail.posetail.cube import from_homogeneous, to_homogeneous, undistort_points
from posetail.posetail.tracker_encoder import TrackerEncoder

from .query_encoder import BOX_ENCODERS, BOX_MODES, WideQueryEncoder

QUERY_MODES = ('prior', 'none')
QUERY_ENCODERS = ('wide',)
# What the gridresid residual is an offset FROM. See `_query_anchored` / `_reanchor_per_frame`.
GRIDRESID_OFFSETS = ('query', 'triangulated')
SCENE_PRECISIONS = ('fp32', 'bf16', 'fp16')

# posetail==0.4.1 still uses the deprecated `torch.backends.cuda.sdp_kernel()` around its
# V-JEPA attention. Keep the dependency pinned, but replace that module's no-argument call with
# the equivalent current API while the scene encoder is running. The lock matters because both
# `torch` and the dependency's module globals are process-wide even though the replacement is
# scoped to this encoder call.
_SDPA_COMPAT_LOCK = threading.RLock()


@contextmanager
def _posetail_sdpa_compat():
    """Use the non-deprecated SDPA context for posetail's legacy attention calls.

    The dependency calls the old context with its defaults, which enable every available backend.
    Preserve that contract (including explicit flags for any future posetail caller) by translating
    the flags to the backend enum list expected by ``torch.nn.attention.sdpa_kernel``. Very old
    PyTorch releases without the new API retain the dependency's original behavior.
    """
    from posetail.posetail import vjepa2

    attention = getattr(torch.nn, 'attention', None)
    sdpa_kernel = getattr(attention, 'sdpa_kernel', None)
    if sdpa_kernel is None:
        yield
        return

    backend_enum = getattr(attention, 'SDPBackend', None)
    if backend_enum is None:
        yield
        return

    def compat_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True,
                      enable_cudnn=True):
        """Translate the legacy backend flags to the current enum-based API."""
        flags = (
            (enable_flash, 'FLASH_ATTENTION'),
            (enable_mem_efficient, 'EFFICIENT_ATTENTION'),
            (enable_math, 'MATH'),
            (enable_cudnn, 'CUDNN_ATTENTION'),
        )
        backends = [getattr(backend_enum, name) for enabled, name in flags
                    if enabled and hasattr(backend_enum, name)]
        return sdpa_kernel(backends)

    with _SDPA_COMPAT_LOCK:
        old_kernel = vjepa2.torch.backends.cuda.sdp_kernel
        vjepa2.torch.backends.cuda.sdp_kernel = compat_kernel
        try:
            yield
        finally:
            vjepa2.torch.backends.cuda.sdp_kernel = old_kernel


def _wrap_posetail_forward(module):
    """Run one posetail module under the SDPA compatibility shim."""
    original_forward = module.forward

    def forward(*args, **kwargs):
        """Call the original module forward with legacy SDPA translated."""
        with _posetail_sdpa_compat():
            return original_forward(*args, **kwargs)

    module.forward = forward


def _wrap_posetail_encoder(encoder):
    """Keep the SDPA shim active when activation checkpointing recomputes encoder blocks."""
    _wrap_posetail_forward(encoder)
    for block in encoder.blocks:
        _wrap_posetail_forward(block)


# the query-free scene point

def scene_center(camera_group):
    """The 3D point every camera's crop centre looks at. (3,) float32.

    Deployment has no ground truth, so derive it from the CAMERAS: each crop centre back-projects
    to a ray and the point closest to all of them is the scene centre -- gauge-correct by
    construction. On a MOVING rig every frame contributes its own ray: a moving crop gives every
    frame its own ray, the one place a per-frame offset must not be collapsed.

    The solve is a least-squares point minimising distance to a set of lines,
    (sum_i P_i) x = sum_i P_i c_i, with a ridge term that keeps a near-degenerate rig from
    producing a wild centre -- ~1e-6 of the trace, so it never moves a good solve.
    """
    origins, dirs = [], []
    for cam in camera_group:
        px = (cam['size'].to(torch.float64) / 2.0).reshape(1, 2)
        off = cam['offset']
        und = [undistort_points(dict(cam, offset=o), px)
               for o in (off if off.ndim == 2 else off[None])]
        ext = cam['ext'] if cam['ext'].ndim == 3 else cam['ext'][None]
        centre = cam['center'] if cam['center'].ndim == 2 else cam['center'][None]
        for t in range(max(ext.shape[0], len(und))):
            u = und[min(t, len(und) - 1)]
            ray_cam = torch.cat([u[0], u.new_ones(1)])
            d = ext[min(t, ext.shape[0] - 1), :3, :3].to(torch.float64).t() @ ray_cam
            dirs.append(d / d.norm())
            origins.append(centre[min(t, centre.shape[0] - 1)].to(torch.float64))
    origins, dirs = torch.stack(origins), torch.stack(dirs)

    eye = torch.eye(3, dtype=origins.dtype, device=origins.device)
    P = eye - dirs[:, :, None] * dirs[:, None, :]
    A = P.sum(0)
    b = torch.einsum('cij,cj->i', P, origins)
    A = A + eye * (1e-6 * torch.diagonal(A).abs().sum().clamp_min(1.0))
    return torch.linalg.solve(A, b).to(torch.float32)


# the model

class PoseTrackerEncoder(TrackerEncoder):
    def __init__(self, *args, n_keypoints, query='prior', query_encoder='wide',
                 gridresid_offset='query', query_terms=None, box_prompt='none', **kwargs):
        """Build the pose tracker: the stock `TrackerEncoder` with a pose-shaped query encoder.

        Inputs: n_keypoints -- size of the keypoint registry.
                query -- 'prior' (per-keypoint prior + missing-query tokens) or 'none'.
                query_encoder -- 'wide' (the only one left; others raise by name).
                gridresid_offset -- 'query' or 'triangulated' (see `_query_anchored`).
                query_terms -- overrides for the pair of prior terms.
                box_prompt -- 'none', or a `BOX_MODES` box-prompt encoder ('film').

        The stock query encoder is replaced with the wide one, built from the same kwargs the
        parent used so it is a drop-in either way. `_shared_scene` starts None (encode per
        forward, the normal path); `share_scene` swaps in a dict. The two query terms DEFAULT to
        following `query`; `query_terms` overrides the pair (pos without patch). Both off under
        "prior" means the encoder ignores `query_coords` entirely -- a declared prior is a silent
        no-op. A box prompt swaps in a `WideQueryEncoder` subclass consuming a per-frame animal
        box as a non-position channel, stashed as `_box_prompt` by `_decode_from_scene`.
        The inference levers `scene_precision`/`camera_batch` default to the historical training
        path; `scripts/infer.py` opts into the measured deployment defaults with
        `set_scene_speed()`.
        """
        assert query in QUERY_MODES, f'query must be one of {QUERY_MODES}, got {query!r}'
        assert query_encoder in QUERY_ENCODERS, \
            f'query_encoder must be one of {QUERY_ENCODERS}, got {query_encoder!r}'
        assert gridresid_offset in GRIDRESID_OFFSETS, \
            f'gridresid_offset must be one of {GRIDRESID_OFFSETS}, got {gridresid_offset!r}'
        assert box_prompt == 'none' or box_prompt in BOX_MODES, \
            f'box_prompt must be "none" or one of {BOX_MODES}, got {box_prompt!r}'
        super().__init__(*args, **kwargs)
        _wrap_posetail_encoder(self.scene_encoder.encoder)
        self.n_keypoints = n_keypoints
        self.query = query
        self.gridresid_offset = gridresid_offset
        self.box_prompt = box_prompt
        self._shared_scene = None
        self.scene_precision = 'fp32'
        self.camera_batch = False

        old = self.query_encoder
        terms = dict.fromkeys(('query_pos_embedding', 'query_patch_embedding'),
                              query == 'prior')
        terms.update(query_terms or {})
        if query == 'prior':
            assert any(terms.values()), (
                'query = "prior" with both query terms off: the encoder would ignore '
                '`query_coords` entirely and the prior would be a no-op. Set at least one of '
                'query_pos_embedding / query_patch_embedding, or use query = "none".')
        else:
            assert not any(terms.values()), (
                f'query = "none" supplies no prior, so {[k for k, v in terms.items() if v]} '
                'would be constant no-query tokens feeding dead gate inputs for the whole run.')
        enc_cls = BOX_ENCODERS.get(box_prompt, WideQueryEncoder)
        self.query_encoder = enc_cls(
            dim=self.latent_dim, embed_dim=old.embed_dim, decoder_dim=old.decoder_dim,
            n_frames=old.n_frames, max_freq=old.max_freq, patch_size=old.patch_size,
            principal_point_embedding=old.principal_point_embedding,
            intrinsic_embedding=old.intrinsic_embedding,
            time_embed_mode=old.time_embed_mode, n_keypoints=n_keypoints, **terms)
        print(f'query encoder: wide{"" if box_prompt == "none" else "+box:" + box_prompt}, '
              f'dim {self.latent_dim}, {self.query_encoder.n_fusion_terms} terms '
              f'({", ".join(self.query_encoder.term_names())})')

    def set_scene_speed(self, precision='fp32', camera_batch=False):
        """Configure the inference-only scene encoding optimizations.

        Precision autocast is deliberately scoped to the scene encoder in ``encode_scene``;
        decoding, camera geometry, triangulation and their intentional float64 solves remain on
        their existing paths. These attributes are runtime state rather than model config because
        they change no weights and are recorded on prediction-session provenance by the CLI.
        """
        if precision not in SCENE_PRECISIONS:
            raise ValueError(f'precision must be one of {SCENE_PRECISIONS}, got {precision!r}')
        self.scene_precision = precision
        self.camera_batch = bool(camera_batch)

    def encode_scene(self, views_norm):
        """Encode a window, optionally batching cameras and autocasting only this call.

        ``SceneRepresentation.forward`` is intentionally left untouched: its operations after
        the encoder are batch-agnostic, so camera inputs can be concatenated on the batch axis and
        the returned ``[1, C*B, N, D]`` tensor reshaped back to ``[C, B, N, D]``. Single-camera
        windows use the original list unchanged, making that case an exact no-op. Under an
        autocast precision the encoder's output is cast back to float32: the decoder and all
        geometry must see the historical FP32 boundary.
        """
        batched = self.camera_batch and len(views_norm) > 1
        if batched:
            ref_shape = views_norm[0].shape[1:]
            if not all(view.shape[1:] == ref_shape for view in views_norm):
                shapes = [tuple(view.shape[1:]) for view in views_norm]
                raise ValueError(
                    'camera batching requires every camera to have the same (T,C,H,W) shape; '
                    f'got {shapes}. Use --no-camera-batch while the upstream resize is fixed.')
            n_cams = len(views_norm)
            batch_size = views_norm[0].shape[0]
            views = [torch.cat(views_norm, dim=0)]
        else:
            views = views_norm

        if self.scene_precision == 'fp32':
            features = self.scene_encoder(views)
        else:
            dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[self.scene_precision]
            with torch.autocast(views[0].device.type, dtype=dtype):
                features = self.scene_encoder(views)
            features = features.float()

        if batched:
            features = features[0].reshape(n_cams, batch_size, *features.shape[2:])
        return features

    def _forward_window(self, views_norm, *args, **kwargs):
        """Route every scene encode through ``encode_scene``.

        Inside ``share_scene`` the first call computes ``scene_features`` and later calls reuse it;
        only the encode is shareable because decoding derives geometry from the prior-changed
        query. Keypoint chunking retains the upstream decode loop while sharing one encode.
        """
        if self._shared_scene is not None and not kwargs.get('kpt_chunk'):
            if 'f' not in self._shared_scene:
                self._shared_scene['f'] = self.encode_scene(views_norm)
            kwargs.pop('kpt_chunk', None)
            kwargs.pop('scene_features', None)
            return super()._forward_window(views_norm, *args, **kwargs,
                                           scene_features=self._shared_scene['f'])
        if kwargs.get('scene_features') is None:
            kwargs['scene_features'] = self.encode_scene(views_norm)
        return super()._forward_window(views_norm, *args, **kwargs)

    def _decode_from_scene(self, scene_features, views_norm, coords, *args, **kwargs):
        """Hand the query encoder the keypoint slice belonging to THIS chunk. Ids must not ride
        the `occlusion` channel (the stock encoder clamps it into [0, 2]), so the slices are
        done here, tracked by a running cursor `forward` asserts lands exactly on K. The box is
        PER FRAME, not per keypoint, so it is not sliced by the chunk cursor -- the encoder
        gathers it onto the query axis by target_time.
        """
        k0, n = self._kpt_cursor, coords.shape[1]
        self.query_encoder._kpt_ids = self._kpt_ids_all[:, k0:k0 + n]
        self.query_encoder._query_ok = self._query_ok_all[:, k0:k0 + n]
        self.query_encoder._box_prompt = getattr(self, '_box_prompt_all', None)
        self._kpt_cursor = k0 + n
        return super()._decode_from_scene(scene_features, views_norm, coords, *args, **kwargs)

    def forward(self, views, kpt_ids, camera_group, mode, **kw):
        """Run `_forward` at the input's actual pixel extent: `image_size` is baked into the
        weights (a pad target, a weight shape, and the input extent), and only the third is
        wrong for a smaller input -- 0.3.5 splits it out as `input_size=`.
        """
        px = max(int(c['size'].max()) for c in camera_group)
        return self._forward(views, kpt_ids, camera_group, mode, input_size=px, **kw)

    def _forward(self, views, kpt_ids, camera_group, mode, kpt_prior=None, prompt_time=None,
                 kpt_chunk=None, box_prompt=None, input_size=None):
        """The forward pass, with the derived query, the query-validity mask, and the 2D/3D
        branch post-processing.

        Args:
            views: list of (B,T,H,W,3) float32 in [0,1], one per camera
            kpt_ids: (B,K) long -- GLOBAL registry ids, not positions
            camera_group: list of posetail camera dicts, one per view
            mode: '2d' | '3d' -- a property of the SAMPLED session, not of the run (one run
                mixes both)
            kpt_prior: (B,K,R) or None; NaN where a keypoint has none. Ignored under
                `query == "none"`.
            prompt_time: (B,K) int or None -- the frame each prior describes
            kpt_chunk: decode in slices of this size, reusing one scene encode (INFERENCE ONLY)

        The loader hands over uint8 (4x fewer bytes to queue and pin); the divide to [0,1]
        happens here after the transfer, where it is free. In 2D there is ONE camera and the
        query is a PIXEL: the library asserts len(views) == 1 for R==2 and uses its 2D head bank.

        The query-validity mask comes from the PRIOR's own finiteness, not `coords_q` (what
        the no-query tokens key off); stashed at FULL K with the box prompt, both handed to
        `_decode_from_scene`. The prompt frame is INT and CLAMPED (a torch.gather index): a
        stale prompt clamps to 0; a KEYPOINT WITH NO PRIOR HAS NO QUERY TIME (time terms
        carry no no-query token), so `prompt_dropout` NaNs the prior but leaves `prompt_t` set.

        The kpt-cursor assertion runs after the call, not in the `finally`; the stash is
        always cleared (a leak would apply one item's ids to the next forward). In 2D the
        grid head decodes ABSOLUTE pixel bins (`coords_pred` in PIXELS); 3D single-view
        keeps the query-anchored residual (the `prob_2d_only` path).
        """
        assert mode in ('2d', '3d'), mode
        B, K = kpt_ids.shape
        n_cams = len(views)
        device = views[0].device
        views = [v.float().div_(255) if v.dtype == torch.uint8 else v for v in views]
        prior = None if self.query == 'none' or kpt_prior is None else kpt_prior.to(device).float()

        if mode == '2d':
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

        query_ok = (torch.isfinite(prior).all(-1) if prior is not None
                    else torch.zeros((B, K), dtype=torch.bool, device=device))
        self._query_ok_all = query_ok
        self._kpt_ids_all = kpt_ids.to(device).long()
        self._box_prompt_all = None if box_prompt is None else box_prompt.to(device).float()
        self._kpt_cursor = 0

        if prompt_time is None or self.query == 'none':
            qt = torch.zeros((B, K), dtype=torch.int64, device=device)
        else:
            T_win = int(views[0].shape[1])
            qt = prompt_time.to(device).round().long().clamp_(0, T_win - 1).to(torch.int64)
            assert qt.shape == (B, K), f'prompt_time {tuple(qt.shape)} != {(B, K)}'
            qt = torch.where(query_ok, qt, torch.zeros_like(qt))

        try:
            out = super().forward(views, coords_q, camera_group, query_times=qt,
                                  occlusion=None, kpt_chunk=kpt_chunk, input_size=input_size)
            assert self._kpt_cursor == K, (
                f'{self._kpt_cursor} of {K} keypoints were decoded; _decode_from_scene is not '
                'being called once per chunk in order, so the id slices are unreliable')
        finally:
            self._query_ok_all = self._kpt_ids_all = None
            self._kpt_cursor = 0
            self._box_prompt_all = None
            self.query_encoder._query_ok = None
            self.query_encoder._kpt_ids = None
            self.query_encoder._box_prompt = None

        if mode == '2d':
            out = dict(out)
            out['coords_pred'] = out['2d_pred'][0]
            return out
        if self.gridresid_offset == 'query':
            return _query_anchored(out, query_ok)
        if out.get('3d_pred_triangulate') is None:
            return out
        return _reanchor_per_frame(out, coords_q)


@contextmanager
def share_scene(model):
    """Encode the scene ONCE for every forward inside the block. The pixels must be identical
    (the caller owns that precondition, which is why the scope is one window). Only the encode
    is shared -- `cube_scale`/`scene_center`/`scene_radius` derive from `coords_q`, which the
    prior changes, so the decode still runs per forward.
    """
    prev = model._shared_scene
    model._shared_scene = {}
    try:
        yield model
    finally:
        model._shared_scene = prev


def _rays_fallback(out):
    """The conf-weighted MEAN of the per-camera ray points -- the honest version of `3d_pred_rays`.

    The library's is a weighted SUM with no division (an unnormalised sigmoid), so at one camera
    it lands about halfway from the world origin to the animal. Used wherever a triangulation is
    missing or degenerate.
    """
    w = torch.sigmoid(out['conf_pred_2d'])
    num = einsum(out['3d_pred_cams_rays'], w, 'c b t n r, c b t n -> b t n r')
    return num / w.sum(0)[..., None].clamp_min(1e-6)


def _query_anchored(out, query_ok):
    """gridresid is an OFFSET FROM THE QUERY POINT, so honour it only where the query is real.

    The library reconstructs `world = query_world + R @ residual` with ONE anchor per window;
    that is right for a genuine prior and wrong for the derived scene centre (identical for
    every keypoint, carries no information). So the prediction is query-anchored WHERE a prior
    exists and each frame's own triangulation elsewhere -- the DETACHED substitution is also the
    loss gate: the direct terms then see a constant at those points and supervise query points
    only.

    In 3D SINGLE-VIEW no triangulation exists, so the back-projected rays are the only anchor
    there is (this used to NaN the whole 3D target, killing the step whenever no keypoint had a
    prior -- which `cams_to_sample = [1, 8]` draws). The substitution repairs FIRST so the loss
    and the metric see ONE tensor. A non-finite entry also poisons the solve's BACKWARD (the NaN
    is created inside `torch.linalg.solve`'s factorisation, downstream of any forward-value fix),
    so on a degenerate step the triangulation supervision goes gradient-free; the forward value
    is unchanged. The repaired triangulation is not written back on the single-view path: that
    key drives `coords_loss_triangulate_reproj` at weight 2.0, and pointing it at the rays would
    reweight that supervision by 40x. Degeneracy is recorded, not gated: a seed taken from
    `_rays_fallback` is a point no camera claimed, and the caller decides what a degenerate frame
    is worth.

    When no keypoint has a valid query there is nothing to supervise the 3D grid CE with --
    every point is triangulated. That term is killed and nothing else: popping `grid` would also
    drop `depth_softmax` (weight 1.5, the largest CE term) and `f_eff` behind it, so the anchor
    is NaN'd in place -- a non-finite anchor is the library's own off switch.
    """
    tri = out.get('3d_pred_triangulate')
    out = dict(out)
    if tri is None:
        sub = _rays_fallback(out)
    else:
        bad = ~torch.isfinite(tri).all(-1)
        if not torch.isfinite(tri).all():
            tri = tri.detach()
        sub = torch.where(torch.isfinite(tri), tri, _rays_fallback(out))
        out['3d_pred_triangulate'] = sub
        out['tri_degenerate'] = bad

    m = query_ok[None, :, None, :, None]
    direct = torch.where(m, out['3d_pred_cams_direct'], sub.detach()[None])
    conf = torch.softmax(out['conf_3d'], dim=0)
    coords_pred = torch.einsum('cbtnr,cbtn->btnr', direct, conf)
    out['3d_pred_cams_direct'] = direct
    out['3d_pred_direct'] = coords_pred
    out['coords_pred'] = coords_pred

    if not bool(query_ok.any()) and out.get('grid') is not None:
        grid = dict(out['grid'])
        grid['anchor_local'] = torch.full_like(grid['anchor_local'], float('nan'))
        out['grid'] = grid
    return out


def _reanchor_per_frame(out, anchor):
    """Re-anchor the 3D residual on EACH frame's own triangulation. `3d_pred_cams_direct =
    query + R @ residual`, so subtracting the query recovers the world-space residual, re-added
    to each frame's own (detached) triangulation.

    The substitution repairs FIRST so the loss and the metric see ONE tensor -- see
    `_query_anchored` for the backward-poisoning reason the detach happens (the solve's backward
    NaNs).
    """
    src = out['3d_pred_triangulate']
    bad = ~torch.isfinite(src).all(-1)
    if not torch.isfinite(src).all():
        src = src.detach()
    src = torch.where(torch.isfinite(src), src, _rays_fallback(out))
    out = dict(out)
    out['tri_degenerate'] = bad

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
    """Move the 3D cross-entropy target onto the SAME anchor the outputs now use. Without this,
    `grid['anchor_local']` still describes the query-anchored task while every metric sees
    per-frame refinement. float64 is mandatory: the library's einsum cancels exactly only in
    float64, where the ray-local coordinates are large and nearly equal.
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
    """`[model]` splatted into the constructor, with this repo's two keys pulled out first. Two
    library options are checked here because both are wrong in a way that produces numbers
    instead of exceptions.

    The two query terms are derived from `query` -- see PoseTrackerEncoder.__init__.
    gridresid_offset has NO DEFAULT, DELIBERATELY: the two values are different architectures
    sharing one set of tensor shapes, so a checkpoint trained under one and loaded under the
    other produces numbers instead of an exception -- which is exactly what happened to one 3D
    run, silently. A pre-key run folder must be told which it was (load_run(model_overrides=...)
    / --gridresid-offset). The two query terms DEFAULT to `query`; setting one gives pos without
    patch. 'none' is a plain wide model, byte-identical to a config without the key.
    n_keypoints is derived from the registry, never configured. The library DEFAULTS to
    'direct', so omitting the key is as dangerous as setting it wrong: both gridresid_offset
    paths need the output query-anchored, which the library only does for 'residual' and
    'gridresid'. Upstream uses mode_3d to select the CLASS; here it is stored and never read, so
    anything else would be accepted and quietly ignored (and TrackerTapNext is not
    moving-cam-safe anyway). use_volume_embedding is consumed only by the stock QueryEncoder,
    which is no longer built -- refused rather than accepted-and-ignored, so it cannot read as a
    knob.
    """
    cfg = dict(model_cfg)
    query = cfg.pop('query', 'prior')
    enc = cfg.pop('query_encoder', 'wide')
    if enc == 'pose':
        raise SystemExit(
            'query_encoder = "pose" was removed: no shipped config selected it and `wide` won '
            'every unanchored arm on record. Set query_encoder = "wide" (or drop the key).')
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
    query_terms = {k: bool(cfg.pop(k)) for k in
                   ('query_pos_embedding', 'query_patch_embedding') if k in cfg}
    box_prompt = cfg.pop('box_prompt', 'none')
    if box_prompt == 'term':
        raise SystemExit(
            'box_prompt = "term" was removed: no shipped config selected it and `film` matched or '
            'beat it. Set box_prompt = "film" (or "none").')
    cfg.pop('n_keypoints', None)

    output_mode = cfg.setdefault('output_mode', 'gridresid')
    assert output_mode == 'gridresid', (
        f'output_mode = {output_mode!r} is not supported: both gridresid_offset modes assume the '
        'library anchored the residual on the query, which it only does for "gridresid". Under '
        'an absolute mode "query" discards a valid prediction at every unprompted point and '
        '"triangulated" re-anchors a prediction that was never query-anchored. '
        'Set output_mode = "gridresid".')

    mode_3d = cfg.setdefault('mode_3d', 'encoder')
    assert mode_3d == 'encoder', (
        f'mode_3d = {mode_3d!r} is not supported: build_model always constructs a '
        'PoseTrackerEncoder, so any other value would be silently ignored rather than honoured.')

    assert not cfg.get('use_volume_embedding'), (
        'use_volume_embedding is not supported: the encoder builds no volume term, so the value '
        'would be silently ignored. Set use_volume_embedding = false or drop the key.')

    return PoseTrackerEncoder(n_keypoints=n_keypoints, query=query, query_encoder=enc,
                              gridresid_offset=offset, query_terms=query_terms,
                              box_prompt=box_prompt, **cfg)
