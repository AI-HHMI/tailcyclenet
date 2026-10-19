"""The split scene encoding inference uses: one raw backbone pass shared by pose and scorer.

`project_scene` mirrors tracktail's `SceneRepresentation.forward` tail through its private
positional-embedding helpers; `test_split_scene_encoding_matches_tracktail` guards it.
"""
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from einops import rearrange

SCENE_PRECISIONS = ('fp32', 'bf16', 'fp16')


@dataclass
class SceneEncoding:
    """Raw encoder tokens and the exact camera packing used to produce them."""

    views: list[torch.Tensor]
    raw_features: list[torch.Tensor]
    n_cams: int
    batch_size: int
    camera_batched: bool


class SceneEncodingMixin:
    """`TrackerEncoder.encode_scene` split into backbone + tail, with inference-only autocast.

    Mixed in BEFORE `TrackerEncoder` so its `encode_scene` wins; the host sets `scene_precision`
    and `camera_batch` in its constructor.
    """

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

    def _scene_autocast(self, views):
        """The inference autocast scope for both the V-JEPA backbone and its scene tail."""
        if self.scene_precision == 'fp32':
            return nullcontext()
        dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[self.scene_precision]
        return torch.autocast(views[0].device.type, dtype=dtype)

    def normalize_scene_views(self, views, input_size=None):
        """Apply the exact preprocessing `TrackerEncoder.forward` uses before scene encoding.

        Inputs are the inference window's `[B,T,H,W,C]` tensors, uint8 or float in [0,1].
        Outputs are normalized `[B,T,C,H,W]` tensors on the model device. This helper exists so
        pose and scorer can share one normalized pixel tensor and one raw encoder pass.
        """
        device = next(self.parameters()).device
        px = self.image_size if input_size is None else int(input_size)
        out = []
        for frames in views:
            frames = frames.to(device)
            if frames.dtype == torch.uint8:
                frames = frames.float().div_(255)
            frames = rearrange(frames, 'b t h w c -> b t c h w')
            out.append(self.normalize(self.pad_to_size(frames, px)))
        return out

    def encode_scene_backbone(self, views_norm):
        """Encode only the common raw video backbone, retaining camera-batch metadata.

        When camera batching is enabled, cameras are concatenated along B exactly as in the
        deployment path. The returned raw tokens stay in that packed shape so each model's
        positional/projection tail can run before the final `[C,B,N,D]` reshape.
        """
        camera_batched = self.camera_batch and len(views_norm) > 1
        if camera_batched:
            ref_shape = views_norm[0].shape[1:]
            if not all(view.shape[1:] == ref_shape for view in views_norm):
                shapes = [tuple(view.shape[1:]) for view in views_norm]
                raise ValueError(
                    'camera batching requires every camera to have the same (T,C,H,W) shape; '
                    f'got {shapes}. Use --no-camera-batch while the upstream resize is fixed.')
            n_cams = len(views_norm)
            batch_size = views_norm[0].shape[0]
            encoder_views = [torch.cat(views_norm, dim=0)]
        else:
            n_cams = len(views_norm)
            batch_size = views_norm[0].shape[0]
            encoder_views = views_norm

        raw = []
        with self._scene_autocast(encoder_views):
            for view in encoder_views:
                raw.append(self.scene_encoder.encoder(
                    rearrange(view, 'b t c h w -> b c t h w')))
        return SceneEncoding(encoder_views, raw, n_cams, batch_size, camera_batched)

    def project_scene(self, encoding: SceneEncoding):
        """Apply this checkpoint's own SceneRepresentation tail to raw encoder tokens.

        This mirrors the installed tracktail implementation using its own positional-embedding
        helpers and projection modules. In the camera-batched path, every tail operation precedes
        the `[C,B,N,D]` reshape, matching the historical `SceneRepresentation.forward` order.
        """
        scene = self.scene_encoder
        projected = []
        with self._scene_autocast(encoding.views):
            for view, feat in zip(encoding.views, encoding.raw_features):
                if scene.pos_embed_mode != 'none':
                    g_h = view.shape[3] // scene.patch_size
                    g_w = view.shape[4] // scene.patch_size
                    g_t = feat.shape[1] // (g_h * g_w)
                    if scene.pos_embed_mode == 'learned':
                        feat = feat + scene._pos_embed_for(g_t, g_h, g_w)
                    elif scene.pos_embed_mode == 'spatial':
                        feat = feat + scene._spatial_pos_embed_for(g_t, g_h, g_w)
                    else:
                        feat = feat + scene._sincos_pos_embed_for(
                            g_t, g_h, g_w, feat.device, feat.dtype)
                if scene.kv_proj is not None:
                    if scene.pre_norms is not None:
                        chunks = torch.chunk(feat, scene.n_proj_levels, dim=-1)
                        feat = torch.cat([norm(chunk) for norm, chunk in
                                          zip(scene.pre_norms, chunks)], dim=-1)
                    feat = scene.kv_proj(feat)
                    feat = scene.kv_norm(feat)
                projected.append(feat)
            features = torch.stack(projected)
            if encoding.camera_batched:
                features = features[0].reshape(
                    encoding.n_cams, encoding.batch_size, *features.shape[2:])
        return features.float()

    def encode_scene(self, views_norm):
        """Encode and project one window; autocast ends at the historical FP32 boundary."""
        return self.project_scene(self.encode_scene_backbone(views_norm))
