"""The pose model: posetail's `TrackerEncoder` with a pose-shaped query and a per-frame anchor.

(1) The query is DERIVED, not given: query-free keypoints are queried at a scene point
back-projected from the cameras' own crop centres (2D: the crop centre), replaced by a
per-keypoint prior when one is present (`query`: `"prior"` | `"none"`); identity rides in a
dedicated fusion term. (2) What the 3D residual is an offset FROM is a switch
(`gridresid_offset`): `"query"` keeps the native anchor where a real prior anchored it,
`"triangulated"` re-adds the residual to each frame's own triangulation.
"""
from contextlib import contextmanager

import torch
from einops import einsum, repeat

from posetail.posetail.cube import from_homogeneous, to_homogeneous, undistort_points
from posetail.posetail.tracker_encoder import TrackerEncoder

from .query_encoder import BOX_ENCODERS, BOX_MODES, WideQueryEncoder

QUERY_MODES = ('prior', 'none')
QUERY_ENCODERS = ('wide',)
# What the gridresid residual is an offset FROM. See `_query_anchored` / `_reanchor_per_frame`.
GRIDRESID_OFFSETS = ('query', 'triangulated')


# the query-free scene point

def scene_center(camera_group):
    """The 3D point every camera's crop centre looks at. (3,) float32.

    Deployment has no ground truth, so derive it from the CAMERAS: each crop centre back-projects
    to a ray and the point closest to all of them is the scene centre -- gauge-correct by
    construction. On a MOVING rig every frame contributes its own ray.
    """
    origins, dirs = [], []
    for cam in camera_group:
        px = (cam['size'].to(torch.float64) / 2.0).reshape(1, 2)
        # A moving crop gives every frame its own ray -- the one place a per-frame offset must
        # not be collapsed.
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

    # Least-squares point minimising distance to a set of lines: (sum_i P_i) x = sum_i P_i c_i.
    eye = torch.eye(3, dtype=origins.dtype, device=origins.device)
    P = eye - dirs[:, :, None] * dirs[:, None, :]
    A = P.sum(0)
    b = torch.einsum('cij,cj->i', P, origins)
    # Ridge term keeps a near-degenerate rig from producing a wild centre; ~1e-6 of the trace,
    # so it never moves a good solve.
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
        """
        assert query in QUERY_MODES, f'query must be one of {QUERY_MODES}, got {query!r}'
        assert query_encoder in QUERY_ENCODERS, \
            f'query_encoder must be one of {QUERY_ENCODERS}, got {query_encoder!r}'
        assert gridresid_offset in GRIDRESID_OFFSETS, \
            f'gridresid_offset must be one of {GRIDRESID_OFFSETS}, got {gridresid_offset!r}'
        assert box_prompt == 'none' or box_prompt in BOX_MODES, \
            f'box_prompt must be "none" or one of {BOX_MODES}, got {box_prompt!r}'
        super().__init__(*args, **kwargs)
        self.n_keypoints = n_keypoints
        self.query = query
        self.gridresid_offset = gridresid_offset
        self.box_prompt = box_prompt
        # None -> encode per forward, the normal path. `share_scene` swaps in a dict.
        self._shared_scene = None

        # Replace the stock query encoder, built from the same kwargs the parent used so it is a
        # drop-in either way.
        old = self.query_encoder
        # The two query terms DEFAULT to following `query`; `query_terms` overrides the pair
        # (pos without patch). Both off under "prior" means the encoder ignores `query_coords`
        # entirely -- a declared prior is a silent no-op.
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
        # A box prompt swaps in a `WideQueryEncoder` subclass consuming a per-frame animal box
        # as a non-position channel, stashed as `_box_prompt` by `_decode_from_scene`.
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

    def _forward_window(self, views_norm, *args, **kwargs):
        """Reuse one scene encode across several decodes over the SAME pixels. Inside
        `share_scene` the first call computes `scene_features` and the rest pass it in; only the
        encode is shareable (the decode derives geometry from the prior-changed `coords_q`).
        """
        if self._shared_scene is None or kwargs.get('kpt_chunk'):
            return super()._forward_window(views_norm, *args, **kwargs)
        if 'f' not in self._shared_scene:
            self._shared_scene['f'] = self.scene_encoder(views_norm)
        kwargs.pop('kpt_chunk', None)
        # The upstream forward already passed None/None.
        kwargs.pop('scene_features', None)
        return super()._forward_window(views_norm, *args, **kwargs,
                                       scene_features=self._shared_scene['f'])

    def _decode_from_scene(self, scene_features, views_norm, coords, *args, **kwargs):
        """Hand the query encoder the keypoint slice belonging to THIS chunk. Ids must not ride
        the `occlusion` channel (the stock encoder clamps it into [0, 2]), so the slices are
        done here, tracked by a running cursor `forward` asserts lands exactly on K.
        """
        k0, n = self._kpt_cursor, coords.shape[1]
        self.query_encoder._kpt_ids = self._kpt_ids_all[:, k0:k0 + n]
        self.query_encoder._query_ok = self._query_ok_all[:, k0:k0 + n]
        # The box is PER FRAME, not per keypoint, so it is not sliced by the chunk cursor -- the
        # encoder gathers it onto the query axis by target_time.
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
        """
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
        """
        assert mode in ('2d', '3d'), mode
        B, K = kpt_ids.shape
        n_cams = len(views)
        device = views[0].device
        # The loader hands over uint8 -- 4x fewer bytes to queue from the worker and to pin --
        # and the divide to [0,1] happens HERE, after the transfer, where it is free. A no-op
        # for anything that already arrives as float.
        views = [v.float().div_(255) if v.dtype == torch.uint8 else v for v in views]
        prior = None if self.query == 'none' or kpt_prior is None else kpt_prior.to(device).float()

        if mode == '2d':
            # ONE camera, and the query is a PIXEL: the library asserts len(views) == 1 for R==2
            # and routes through its separate 2D head bank (mode_idx = 0), pretrained at full
            # size in the base checkpoint.
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

        # The query-validity mask comes from the PRIOR's own finiteness, not from `coords_q`
        # (absent priors are already replaced by the derived point) -- what the no-query tokens
        # key off. Stashed at FULL K; `_decode_from_scene` hands the encoder the chunk's slice.
        query_ok = (torch.isfinite(prior).all(-1) if prior is not None
                    else torch.zeros((B, K), dtype=torch.bool, device=device))
        # Local copy survives the `finally` clear below.
        self._query_ok_all = query_ok
        self._kpt_ids_all = kpt_ids.to(device).long()
        # The box prompt, stashed for `_decode_from_scene`; a plain model ignores it, so passing
        # one is harmless.
        self._box_prompt_all = None if box_prompt is None else box_prompt.to(device).float()
        self._kpt_cursor = 0

        # The frame the prompt describes. INT and CLAMPED: the library uses it as a
        # `torch.gather` index, so an out-of-range or floating value is an index error; a prompt
        # from BEFORE this window (deployment staleness) cannot be expressed and clamps to 0.
        if prompt_time is None or self.query == 'none':
            qt = torch.zeros((B, K), dtype=torch.int32, device=device)
        else:
            T_win = int(views[0].shape[1])
            qt = prompt_time.to(device).round().long().clamp_(0, T_win - 1).to(torch.int32)
            assert qt.shape == (B, K), f'prompt_time {tuple(qt.shape)} != {(B, K)}'
            # A KEYPOINT WITH NO PRIOR HAS NO QUERY TIME: the time terms carry no no-query
            # token, so an unprompted keypoint still reporting a frame index trains a forward no
            # deployment path produces. `prompt_dropout` NaNs the prior but leaves `prompt_t`
            # set -- exactly this case. Fixed here, at the seam every caller routes through.
            qt = torch.where(query_ok, qt, torch.zeros_like(qt))

        try:
            out = super().forward(views, coords_q, camera_group, query_times=qt,
                                  occlusion=None, kpt_chunk=kpt_chunk, input_size=input_size)
            # After the call, not in the `finally`: an assertion there would mask the real error.
            assert self._kpt_cursor == K, (
                f'{self._kpt_cursor} of {K} keypoints were decoded; _decode_from_scene is not '
                'being called once per chunk in order, so the id slices are unreliable')
        finally:
            # Always clear: a leaked stash would apply one item's ids to the next forward,
            # silently, on multi-call paths like eval and the windowed driver.
            self._query_ok_all = self._kpt_ids_all = None
            self._kpt_cursor = 0
            self._box_prompt_all = None
            self.query_encoder._query_ok = None
            self.query_encoder._kpt_ids = None
            self.query_encoder._box_prompt = None

        if mode == '2d':
            # No re-anchoring in 2D: triangulation is None at one camera and the 2D grid head
            # decodes ABSOLUTE pixel bins. `coords_pred` is rebound so every consumer can say
            # "the prediction" without a mode branch; at R==2 it is in PIXELS, not mm.
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
    # (cams, b, t, n)
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
    """
    tri = out.get('3d_pred_triangulate')
    out = dict(out)
    if tri is None:
        # 3D SINGLE-VIEW: no triangulation exists, so the back-projected rays are the only
        # anchor there is (this used to NaN the whole 3D target, killing the step whenever no
        # keypoint had a prior -- which `cams_to_sample = [1, 8]` draws).
        sub = _rays_fallback(out)
    else:
        # Repair FIRST so the loss and the metric see ONE tensor. A non-finite entry also
        # poisons the solve's BACKWARD (the NaN is created inside `torch.linalg.solve`'s
        # factorisation, downstream of any forward-value fix), so on a degenerate step the
        # triangulation supervision goes gradient-free; the forward value is unchanged.
        bad = ~torch.isfinite(tri).all(-1)
        if not torch.isfinite(tri).all():
            tri = tri.detach()
        sub = torch.where(torch.isfinite(tri), tri, _rays_fallback(out))
        # Not written back on the single-view path: that key drives
        # `coords_loss_triangulate_reproj` at weight 2.0, and pointing it at the rays would
        # reweight that supervision by 40x.
        out['3d_pred_triangulate'] = sub
        # Recorded, not gated: a seed taken from `_rays_fallback` is a point no camera claimed,
        # and the caller decides what a degenerate frame is worth.
        out['tri_degenerate'] = bad

    # query_ok -> (cams, b, t, n, r)
    m = query_ok[None, :, None, :, None]
    direct = torch.where(m, out['3d_pred_cams_direct'], sub.detach()[None])
    conf = torch.softmax(out['conf_3d'], dim=0)
    coords_pred = torch.einsum('cbtnr,cbtn->btnr', direct, conf)
    out['3d_pred_cams_direct'] = direct
    out['3d_pred_direct'] = coords_pred
    out['coords_pred'] = coords_pred

    if not bool(query_ok.any()) and out.get('grid') is not None:
        # Nothing to supervise the 3D grid CE with -- every point is triangulated. Kill THAT
        # term and nothing else: popping `grid` also drops `depth_softmax` (weight 1.5, the
        # largest CE term) and `f_eff` behind it. A non-finite anchor is the library's own off
        # switch.
        grid = dict(out['grid'])
        grid['anchor_local'] = torch.full_like(grid['anchor_local'], float('nan'))
        out['grid'] = grid
    return out


def _reanchor_per_frame(out, anchor):
    """Re-anchor the 3D residual on EACH frame's own triangulation. `3d_pred_cams_direct =
    query + R @ residual`, so subtracting the query recovers the world-space residual, re-added
    to each frame's own (detached) triangulation.
    """
    src = out['3d_pred_triangulate']
    # Repair FIRST so the loss and the metric see ONE tensor; see `_query_anchored` for the
    # backward-poisoning reason the detach happens.
    bad = ~torch.isfinite(src).all(-1)
    if not torch.isfinite(src).all():
        # See `_query_anchored`: the solve's backward NaNs.
        src = src.detach()
    src = torch.where(torch.isfinite(src), src, _rays_fallback(out))
    out = dict(out)
    # See `_query_anchored`.
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
    """
    cfg = dict(model_cfg)
    query = cfg.pop('query', 'prior')
    # The two query terms are derived from `query` -- see PoseTrackerEncoder.__init__.
    enc = cfg.pop('query_encoder', 'wide')
    if enc == 'pose':
        raise SystemExit(
            'query_encoder = "pose" was removed: no shipped config selected it and `wide` won '
            'every unanchored arm on record. Set query_encoder = "wide" (or drop the key).')
    # NO DEFAULT, DELIBERATELY: the two values are different architectures sharing one set of
    # tensor shapes, so a checkpoint trained under one and loaded under the other produces
    # numbers instead of an exception -- which is exactly what happened to one 3D run, silently.
    # A pre-key run folder must be told which it was (load_run(model_overrides=...) /
    # --gridresid-offset).
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
    # The two query terms DEFAULT to `query`; setting one gives pos without patch.
    query_terms = {k: bool(cfg.pop(k)) for k in
                   ('query_pos_embedding', 'query_patch_embedding') if k in cfg}
    # 'none' is a plain wide model, byte-identical to a config without the key.
    box_prompt = cfg.pop('box_prompt', 'none')
    if box_prompt == 'term':
        raise SystemExit(
            'box_prompt = "term" was removed: no shipped config selected it and `film` matched or '
            'beat it. Set box_prompt = "film" (or "none").')
    # Derived from the registry, never configured.
    cfg.pop('n_keypoints', None)

    # The library DEFAULTS to 'direct', so omitting the key is as dangerous as setting it
    # wrong: both gridresid_offset paths need the output query-anchored, which the library only
    # does for 'residual' and 'gridresid'.
    output_mode = cfg.setdefault('output_mode', 'gridresid')
    assert output_mode == 'gridresid', (
        f'output_mode = {output_mode!r} is not supported: both gridresid_offset modes assume the '
        'library anchored the residual on the query, which it only does for "gridresid". Under '
        'an absolute mode "query" discards a valid prediction at every unprompted point and '
        '"triangulated" re-anchors a prediction that was never query-anchored. '
        'Set output_mode = "gridresid".')

    # Upstream uses this to select the CLASS; here it is stored and never read, so anything else
    # would be accepted and quietly ignored (and TrackerTapNext is not moving-cam-safe anyway).
    mode_3d = cfg.setdefault('mode_3d', 'encoder')
    assert mode_3d == 'encoder', (
        f'mode_3d = {mode_3d!r} is not supported: build_model always constructs a '
        'PoseTrackerEncoder, so any other value would be silently ignored rather than honoured.')

    # Consumed only by the stock QueryEncoder, which is no longer built. Refused rather than
    # accepted-and-ignored, so it cannot read as a knob.
    assert not cfg.get('use_volume_embedding'), (
        'use_volume_embedding is not supported: the encoder builds no volume term, so the value '
        'would be silently ignored. Set use_volume_embedding = false or drop the key.')

    return PoseTrackerEncoder(n_keypoints=n_keypoints, query=query, query_encoder=enc,
                              gridresid_offset=offset, query_terms=query_terms,
                              box_prompt=box_prompt, **cfg)
