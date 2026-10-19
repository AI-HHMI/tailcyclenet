"""The pose model: tracktail's `TrackerEncoder` with a pose-shaped query and a per-frame anchor.

(1) The query is DERIVED, not given: query-free keypoints are queried at a scene point
back-projected from the cameras' own crop centres (2D: the crop centre), replaced by a
per-keypoint prior when one is present (`query`: `"prior"` | `"none"`); identity rides in a
dedicated fusion term. (2) What the 3D residual is an offset FROM is a switch
(`gridresid_offset`): `"query"` keeps the native anchor where a real prior anchored it,
`"triangulated"` re-adds the residual to each frame's own triangulation.
"""
from contextlib import contextmanager
import warnings

import torch

from tracktail.model.tracker_encoder import TrackerEncoder

from .anchor import query_anchored, reanchor_per_frame, scene_center, uncalibrated_singleview_anchor
from .query_encoder import BOX_ENCODERS, BOX_MODES, WideQueryEncoder
from .scene import SceneEncodingMixin

QUERY_MODES = ('prior', 'none')
QUERY_ENCODERS = ('wide',)
GRIDRESID_OFFSETS = ('query', 'triangulated')


class PoseTrackerEncoder(SceneEncodingMixin, TrackerEncoder):
    def __init__(self, *args, n_keypoints, query='prior', query_encoder='wide',
                 gridresid_offset='query', query_terms=None, box_prompt='none', **kwargs):
        """Build the pose tracker: the stock `TrackerEncoder` with a pose-shaped query encoder.

        Inputs: n_keypoints -- size of the keypoint registry.
                query -- 'prior' (per-keypoint prior + missing-query tokens) or 'none'.
                query_encoder -- 'wide' (the only one left; others raise by name).
                gridresid_offset -- 'query' or 'triangulated' (see `anchor.query_anchored`).
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

    def _forward_window(self, views_norm, *args, **kwargs):
        """Route scene encodes through the local split or reuse caller-provided features."""
        shared = self._shared_scene
        supplied = kwargs.get('scene_features')
        if shared is not None and (not kwargs.get('kpt_chunk') or 'f' in shared or
                                   supplied is not None):
            if 'f' not in shared:
                shared['f'] = supplied if supplied is not None else self.encode_scene(views_norm)
            kwargs.pop('kpt_chunk', None)
            kwargs.pop('scene_features', None)
            return super()._forward_window(views_norm, *args, **kwargs,
                                           scene_features=shared['f'])
        if supplied is None:
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
                 kpt_chunk=None, box_prompt=None, input_size=None, scene_features=None):
        """The forward pass, with the derived query, the query-validity mask, and the 2D/3D
        branch post-processing.

        Args:
            views: list of (B,T,H,W,3) float32 in [0,1], one per camera
            kpt_ids: (B,K) long -- GLOBAL registry ids, not positions
            camera_group: list of tracktail camera dicts, one per view
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
        grid head decodes ABSOLUTE pixel bins (`coords_pred` in PIXELS); a one-camera 3D
        sample keeps the query-anchored residual.
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
            center = scene_center(camera_group).to(device)
            cam = camera_group[0]
            if (n_cams == 1 and not self.training
                    and cam.get('calibrated', True) is False and cam['ext'].ndim == 2):
                center = uncalibrated_singleview_anchor(cam).to(device=device, dtype=torch.float32)
                if not getattr(self, '_warned_singleview_scale_fallback', False):
                    warnings.warn(
                        'uncalibrated single-view 3D inference: using a one-unit camera-ray '
                        'anchor to keep cube_scale finite; resulting XYZ is normalized and NOT '
                        'metric', RuntimeWarning, stacklevel=2)
                    self._warned_singleview_scale_fallback = True
            coords_q = center.view(1, 1, 3).expand(B, K, 3).contiguous()

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
                                  occlusion=None, kpt_chunk=kpt_chunk, input_size=input_size,
                                  scene_features=scene_features)
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
            return query_anchored(out, query_ok)
        if out.get('3d_pred_triangulate') is None:
            return out
        return reanchor_per_frame(out, coords_q)


@contextmanager
def share_scene(model, scene_features=None):
    """Reuse scene features across same-window forwards, optionally supplied by a joint caller.

    The pixels must be identical (the caller owns that precondition, hence one-window scope).
    Only the encode is shared -- geometry derives from `coords_q`, which the prior changes, so
    every forward still runs its own decoder.
    """
    prev = model._shared_scene
    model._shared_scene = {} if scene_features is None else {'f': scene_features}
    try:
        yield model
    finally:
        model._shared_scene = prev


def share_video_encoder_if_identical(pose_model, scorer_model):
    """Tie the raw scene encoders only when every checkpoint tensor is exactly equal.

    Models may be on different devices (the scorer is loaded on CPU to avoid a duplicate encoder
    peak on GPU); tensors are compared one at a time on CPU. On mismatch, leave both modules
    independent and return False so inference can encode each model separately.
    """
    pose_state = pose_model.scene_encoder.encoder.state_dict()
    scorer_state = scorer_model.scene_encoder.encoder.state_dict()
    if pose_state.keys() != scorer_state.keys():
        return False
    for key, pose_tensor in pose_state.items():
        scorer_tensor = scorer_state[key]
        if pose_tensor.shape != scorer_tensor.shape or pose_tensor.dtype != scorer_tensor.dtype:
            return False
        if pose_tensor.device != scorer_tensor.device:
            equal = torch.equal(pose_tensor.detach().cpu(), scorer_tensor.detach().cpu())
        else:
            equal = torch.equal(pose_tensor, scorer_tensor)
        if not equal:
            return False
    scorer_model.scene_encoder.encoder = pose_model.scene_encoder.encoder
    return True


def build_model(model_cfg: dict, n_keypoints: int, cls=PoseTrackerEncoder, **extra) -> PoseTrackerEncoder:
    """`[model]` splatted into the constructor, with this repo's keys pulled out and checked first.

    `cls`/`**extra` exist for `scorer.build_scorer` (a `PoseScorer` and its head kwargs), so every
    refusal below is shared rather than copied. `gridresid_offset` has NO DEFAULT: its two values
    are different architectures with identical tensor shapes, so loading under the wrong one
    produces numbers, not an exception (a pre-key run folder states it via
    `load_run(model_overrides=...)` / `--gridresid-offset`). `output_mode` must be `gridresid`
    (both offsets assume a query-anchored residual; the library default `direct` is not);
    `mode_3d` and `use_volume_embedding` would be silently ignored, so they are refused.
    `n_keypoints` is derived from the registry, never configured.
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

    return cls(n_keypoints=n_keypoints, query=query, query_encoder=enc,
               gridresid_offset=offset, query_terms=query_terms,
               box_prompt=box_prompt, **cfg, **extra)


