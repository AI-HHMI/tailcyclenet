"""Builders shared across the suite: synthetic sessions and rigs, tiny models and run folders.

Plain functions and constants only; pytest fixtures live in the `conftest.py` files.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from tailcyclenet import format as fmt
from tailcyclenet.format import Registry, load_dataset
from tailcyclenet.infer.window import InferConfig
from tailcyclenet.pose.model import build_model
from tailcyclenet.scorer import build_scorer

REPO = Path(__file__).resolve().parent.parent


KPTS_2D = ['nose', 'left_ear', 'right_ear', 'tail_base']
KPTS_3D = ['nose', 'neck', 'tail_base']

# Everything structural from the shipped configs, everything expensive at its floor.
SMALL = dict(
    query='prior', mode_3d='encoder', video_encoder_version='base',
    video_encoder_requires_grad=False, video_encoder_hierarchical=True,
    stride_length=4, image_size=64, query_patch_size=5, per_camera_cube_scale=True,
    max_freq=4, embedding_dim=32, use_volume_embedding=False, corr_radius=2,
    principal_point_embedding=True, intrinsic_embedding=True, metric_ray_translation=True,
    occlusion_embedding=True, latent_dim=64, n_heads=2, n_time_space_blocks=1,
    embedding_factor=2, use_camera_self_attention=True, use_temporal_self_attention=True,
    f_eff_scale=True, scene_encoder_proj=True, scene_proj_dim=64,
    scene_pos_embed_mode='ropepos', rope_base=100.0, time_embed_mode='fourier_rel',
    gridresid_offset='query',
    output_mode='gridresid', head_3d_grid_size=64, head_3d_grid_radius=1.8,
    log_3d_output=True, log_3d_eps=0.1, soft_argmax_threshold=60, grid_decode_space='warped',
)


# ---------------------------------------------------------------------------------------------
# the staged encoder unfreeze
# ---------------------------------------------------------------------------------------------

UNFREEZE_AT, N_LAST = 4, 3


SCORER_KW = dict(pool_num_heads=8, score_hidden=64, use_precision=True)


def _detector_checkpoint(model=None, **overrides):
    """A minimal but genuinely loadable detector checkpoint dict -- `load_detector` needs real
    tensors and every key it reads, not a stub. NOT written to disk (the caller decides whether
    to `torch.save` it as-is or merge in more overrides first, e.g. `{**_detector_checkpoint(), 'tile_wh': [640, 640]}`).

    Inputs: model -- a built `YOLOXNano`, or `None` for a fresh default one; overrides -- any
        checkpoint key, replacing the default.
    Outputs: a checkpoint dict with `model_state`/`input_wh`/`norm` at their shipped-recipe
        defaults, plus every override.
    Side effects: none.

    Deliberately malformed checkpoints (missing keys, an empty `model_state`, wrong dtypes -- the
    refusal tests) stay as explicit inline dict literals at their call sites: a shared factory
    would obscure exactly what each of those tests omits.
    """
    from tailcyclenet.detector.yolox import YOLOXNano

    m = model if model is not None else YOLOXNano(version='hybrid')
    return {'model_state': m.state_dict(), 'input_wh': [416, 416], 'norm': 'gn',
            'yolox_version': m.version, 'p2': m.p2, 'shared_head': m.head.shared_head,
            'fpn_upsample': m.neck.fpn_upsample, **overrides}


def _rig(specs):
    """specs: [(name, w, h, calibrated, moving, idx)] -> a Rig of aniposelib cameras."""
    from aniposelib.cameras import Camera, CameraGroup

    cams, offset, moving_d, calib_d = [], {}, {}, {}
    for name, w, h, calibrated, moving, idx in specs:
        f = float(max(w, h))
        if calibrated:
            cam = Camera(matrix=np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]),
                         dist=np.zeros(5), rvec=np.array([0.0, 0.1 * idx, 0.0]),
                         tvec=np.array([10.0 * idx, 0.0, 300.0]), name=name)
            cam.set_size((w, h))
        else:
            cam = fmt.nominal_camera(name, (w, h))
        cams.append(cam)
        offset[name], moving_d[name], calib_d[name] = (0.0, 0.0), moving, calibrated
    return fmt.Rig(CameraGroup(cams), offset=offset, moving=moving_d, calibrated=calib_d)


def _write_frames(group_dir, cam, n_frames, size):
    """Frames whose content identifies (cam, frame), so an off-by-one is visible in pixels."""
    d = group_dir / cam
    d.mkdir(parents=True, exist_ok=True)
    W, H = size
    y, x = np.mgrid[0:H, 0:W]
    for i in range(n_frames):
        px = np.stack([(x * 3 + i * 11) % 256, (y * 5 + i * 7) % 256,
                       np.full((H, W), (i * 37 + len(cam) * 3) % 256)], -1).astype(np.uint8)
        Image.fromarray(px).save(d / f'{i:06d}.png')


def _video_colour(cam_ix, i):
    """The RGB a `_write_video` frame should decode to. PyAV returns RGB; cv2 writes BGR."""
    return (7 * i) % 256, (200 - 10 * i) % 256, (30 * i + 11 * cam_ix) % 256


def _write_video(path, cam_ix, n_frames, size, fps=20.0):
    """A tiny mp4 whose every frame is a solid colour identifying (cam, frame)."""
    import cv2

    W, H = size
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (W, H))
    assert vw.isOpened(), f'cannot write {path}'
    for i in range(n_frames):
        r, g, b = _video_colour(cam_ix, i)
        vw.write(np.dstack([np.full((H, W), b, np.uint8), np.full((H, W), g, np.uint8),
                            np.full((H, W), r, np.uint8)]))
    vw.release()
    return path


def _session_2d(path, T=4, S=2, label_source='annotated'):
    """rat-city's shape: one uncalibrated camera, several animals, pixel labels."""
    W, H = 64, 48
    rig = _rig([('cam0', W, H, False, False, 0)])
    K = len(KPTS_2D)
    lab = fmt.empty_labels(S, T, K, 1, mode3d=False, animal_ids=['a01', 'a02'])
    rng = np.random.default_rng(0)
    lab.vis2d[:] = fmt.VISIBLE
    lab.points2d[..., 0, :] = rng.uniform(5, min(W, H) - 5, size=(S, T, K, 2)).astype(np.float32)
    # one explicitly-missing point and one never-assessed point, so both codes are exercised
    lab.vis2d[0, 0, 1, 0] = fmt.MISSING
    lab.points2d[0, 0, 1, 0] = np.nan
    lab.vis2d[1, 2, 3, 0] = fmt.UNLABELED
    lab.points2d[1, 2, 3, 0] = np.nan
    # a02 is a boxed `present` row on frame 1. a01 deliberately has NO stored box anywhere: that
    # is what `test_box_source_falls_back_per_view` pins for the pose loader, whose per-camera
    # keypoint fallback is unchanged. A boxless annotated animal cannot be `labeled` (rule 11), so
    # every row here is `present`: the fixture's frames are all incompletely labelled.
    lab.instance = np.full((S, T, 1), fmt.INST_NONE, np.int8)
    lab.boxes = np.full((S, T, 1, 4), np.nan, np.float32)
    lab.instance[:, :, 0] = fmt.INST_PRESENT
    lab.boxes[1, 1, 0] = [10.0, 10.0, 30.0, 30.0]

    groups = {'g000': fmt.Group('g000', T, fps=40.0, source_video='movie.avi')}
    fmt.write_session(path, mode='2d', units='px', label_source=label_source, names=KPTS_2D,
                      rig=rig, groups=groups,
                      labels={'g000': lab}, flip_pairs=[['left_ear', 'right_ear']],
                      provenance={'source': 'synthetic'})
    _write_frames(path / 'groups' / 'g000', 'cam0', T, (W, H))
    return lab


def _session_2d_tracked_dense(path, T=4, S=2):
    """`tracked`, every row `visible`, no assessment ever recorded -- the calms21 shape
    `has_visibility_assessment` must read as False."""
    W, H = 64, 48
    rig = _rig([('cam0', W, H, False, False, 0)])
    K = len(KPTS_2D)
    lab = fmt.empty_labels(S, T, K, 1, mode3d=False, animal_ids=['a01', 'a02'])
    rng = np.random.default_rng(3)
    lab.vis2d[:] = fmt.VISIBLE
    lab.points2d[..., 0, :] = rng.uniform(5, min(W, H) - 5, size=(S, T, K, 2)).astype(np.float32)
    groups = {'g000': fmt.Group('g000', T, fps=40.0, source_video='movie.avi')}
    fmt.write_session(path, mode='2d', units='px', label_source='tracked', names=KPTS_2D,
                      rig=rig, groups=groups, labels={'g000': lab},
                      provenance={'source': 'synthetic', 'visibility': 'none'})
    _write_frames(path / 'groups' / 'g000', 'cam0', T, (W, H))
    return lab


def _session_3d(path, T=4, moving=False, label_source='tracked', names=None,
                flip_pairs=None):
    """allen-mouse's shape: native 3D plus coordinate-free per-camera visibility rows."""
    W, H = 64, 48
    names = KPTS_3D if names is None else list(names)
    rig = _rig([(f'cam{i}', W, H, True, moving and i == 0, i + 1) for i in range(3)])
    K = len(names)
    lab = fmt.empty_labels(1, T, K, 3, mode3d=True, animal_ids=['m1'])
    rng = np.random.default_rng(1)
    lab.vis3d[:] = fmt.VISIBLE
    lab.points3d[:] = rng.uniform(-50, 50, size=(1, T, K, 3)).astype(np.float32)
    lab.vis3d[0, 1, 2] = fmt.MISSING
    lab.points3d[0, 1, 2] = np.nan
    # Per-camera visibility with no x,y: visible, assessed-but-occluded, and never assessed.
    lab.vis2d[:] = fmt.VISIBLE
    lab.vis2d[0, :, 0, 2] = fmt.MISSING
    lab.vis2d[0, 1, 2] = fmt.UNLABELED          # keypoint 2, frame 1: nobody looked
    if moving:
        lab.ext = np.tile(np.eye(4), (3, T, 1, 1))
        for t in range(T):
            lab.ext[0, t, 0, 3] = float(t)          # camera 0 slides along x
        for i, cam in enumerate(rig.cameras):
            if not rig.moving[cam.get_name()]:
                lab.ext[i] = cam.get_extrinsics_mat().detach().cpu().numpy()

    groups = {'g000': fmt.Group('g000', T, fps=200.0)}
    write_args = {} if flip_pairs is None else {'flip_pairs': flip_pairs}
    fmt.write_session(path, mode='3d', units='mm', label_source=label_source, names=names,
                      rig=rig, groups=groups, labels={'g000': lab},
                      provenance={'source': 'synthetic'}, **write_args)
    for name in rig.names:
        _write_frames(path / 'groups' / 'g000', name, T, (W, H))
    return lab


def _session_3d_multi(path, T=4, sep=5.0, label_source='tracked'):
    """`_session_3d`'s shape with a second animal (the first's pose shifted by `sep`), for the
    animal-swap prior corruption: small `sep` = eligible neighbour, large = ineligible."""
    W, H = 64, 48
    rig = _rig([(f'cam{i}', W, H, True, False, i + 1) for i in range(3)])
    K = len(KPTS_3D)
    lab = fmt.empty_labels(2, T, K, 3, mode3d=True, animal_ids=['m1', 'm2'])
    rng = np.random.default_rng(2)
    lab.vis3d[:] = fmt.VISIBLE
    base = rng.uniform(-50, 50, size=(T, K, 3)).astype(np.float32)
    lab.points3d[0] = base
    lab.points3d[1] = base + np.array([sep, 0.0, 0.0], dtype=np.float32)
    lab.vis2d[:] = fmt.VISIBLE

    groups = {'g000': fmt.Group('g000', T, fps=200.0)}
    fmt.write_session(path, mode='3d', units='mm', label_source=label_source, names=KPTS_3D,
                      rig=rig, groups=groups,
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    for name in rig.names:
        _write_frames(path / 'groups' / 'g000', name, T, (W, H))
    return lab


def _session_projected(path, T=4):
    """johnson-mouse's shape: per-camera 2D positions with no visibility assessment anywhere."""
    W, H = 64, 48
    rig = _rig([(f'cam{i}', W, H, True, False, i + 1) for i in range(3)])
    K = len(KPTS_3D)
    lab = fmt.empty_labels(1, T, K, 3, mode3d=True, animal_ids=['m1'])
    rng = np.random.default_rng(7)
    lab.vis3d[:] = fmt.VISIBLE
    lab.points3d[:] = rng.uniform(-20, 20, size=(1, T, K, 3)).astype(np.float32)
    lab.vis2d[:] = fmt.PROJECTED
    lab.points2d[:] = rng.uniform(5, 40, size=(1, T, K, 3, 2)).astype(np.float32)

    groups = {'g000': fmt.Group('g000', T, fps=100.0)}
    fmt.write_session(path, mode='3d', units='mm', label_source='tracked', names=KPTS_3D,
                      rig=rig, groups=groups,
                      labels={'g000': lab}, provenance={'source': 'synthetic'})
    for name in rig.names:
        _write_frames(path / 'groups' / 'g000', name, T, (W, H))
    return lab


def _session_centred_labels(path, T=24, labelled=(11, 12, 13)):
    """A group whose labels sit only in the middle -- the shape the old loader could not use."""
    W = H = 48
    K = len(KPTS_3D)
    rig = _rig([('cam0', W, H, False, False, 0)])
    lab = fmt.empty_labels(1, T, K, 1, mode3d=False)
    lab.vis2d[0, list(labelled), :, 0] = fmt.VISIBLE
    lab.points2d[0, list(labelled), :, 0] = (
        10.0 + 10.0 * np.arange(len(labelled), dtype=np.float32))[:, None, None] % (W - 10)
    groups = {'g0': fmt.Group('g0', T)}
    fmt.write_session(path, mode='2d', units='px', label_source='tracked', names=KPTS_3D,
                      rig=rig, groups=groups, labels={'g0': lab})
    _write_frames(path / 'groups' / 'g0', 'cam0', T, (W, H))
    return list(labelled)


def _write_config(tmp_path, text, name='config.toml'):
    p = tmp_path / name
    p.write_text(text)
    return p


def _lever_boxes(cg, worlds, side=200.0):
    """Project `worlds` into every camera of `cg` as square boxes, in `CrossViewTracker.step`'s
    argument shapes. Large by default: every gate in track.py is in box sides."""
    from tailcyclenet.infer.track import _project

    per_cam, scores = [], []
    for cam in cg:
        uv = _project(cam, np.asarray(worlds, np.float32))
        per_cam.append(torch.stack([uv[:, 0] - side / 2, uv[:, 1] - side / 2,
                                    uv[:, 0] + side / 2, uv[:, 1] + side / 2], -1))
        scores.append(torch.ones(len(worlds)))
    return per_cam, scores


def _flat_rig():
    """Three coplanar cameras with optical axes through the origin 
    (fine where no residual gate is read)."""
    from aniposelib.cameras import Camera, CameraGroup

    from tailcyclenet.format import Rig

    cams = []
    for i, ang in enumerate((-0.5, 0.0, 0.5)):
        cam = Camera(matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]]),
                     dist=np.zeros(5), rvec=np.array([0.0, ang, 0.0]),
                     tvec=np.array([0.0, 0.0, 900.0]), name=f'c{i}')
        cam.set_size((640, 480))
        cams.append(cam)
    names = [c.get_name() for c in cams]
    return Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in names},
               moving=dict.fromkeys(names, False),
               calibrated=dict.fromkeys(names, True)).tracktail()


def _scene(tmp_path_factory, kind, T):
    """(model, session, registry, dataset name): `rat` = 2D, 2 animals, 1 camera; `mv` = 3D on a
    moving 3-camera rig. T=4 is ONE window at n_frames=4; T=16 is seven (overlap 2)."""

    root = tmp_path_factory.mktemp(f'{kind}{T}')
    if kind == 'rat':
        _session_2d(root / 'rat' / 'test' / 's', T=T)
    else:
        _session_3d(root / 'mv' / 'test' / 's', T=T, moving=True)
    ds = load_dataset(root / kind)
    registry = Registry.build([ds])
    sess = ds.sessions['test'][0]
    sess.preload()
    model = build_model(SMALL, n_keypoints=registry.n_keypoints).eval()
    return model, sess, registry, ds.name


def _infer_cfg(**kw):
    kw.setdefault('overlap', 2)
    return InferConfig(n_frames=4, image_size=64, min_crop_dim=16, device='cpu', **kw)


def _save_run(tmp_path, registry, model_cfg=SMALL):
    """A loadable run folder (config + registry + one checkpoint) for a random SMALL model."""
    from tailcyclenet.checkpoints import save_checkpoint, save_run_meta

    model = build_model(model_cfg, n_keypoints=registry.n_keypoints)
    run = tmp_path / 'run'
    config = {'model': model_cfg, 'data': {'image_size': 64, 'min_crop_dim': 16, 'n_frames': 4,
                                           'box_source': 'keypoints'}}
    save_run_meta(run, config, registry)
    path = save_checkpoint(run, 0, model, torch.optim.SGD(model.parameters(), lr=0.0), config,
                           registry=registry)
    return run, path


class TinyEncoder(nn.Module):
    """The shape `unfreeze.py` reads: `blocks`, `norms_block` and `hierarchical_layers`.
    Six blocks with taps at [1,3,4,5], so the last-N rule selects a proper sub-range.
    """

    def __init__(self, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(depth)])
        self.hierarchical_layers = [1, 3, 4, 5]
        self.norms_block = nn.ModuleList([nn.LayerNorm(8) for _ in self.hierarchical_layers])


class Tiny(nn.Module):
    """Named to match the router's substrings: `decoder.mlps` -> Muon, `decoder.heads_*` -> AdamW,
    `scene_encoder.encoder.blocks` -> the encoder Muon group, an Embedding that must stay AdamW.
    Also stands in for `TrackerEncoder`'s staged-unfreeze contract, including re-freezing before
    unfreezing the last N.
    """

    def __init__(self, unfreeze_at=None, n_last=3):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.mlps = nn.ModuleList([nn.Linear(8, 8)])          # 2D weight -> muon_dec
        self.decoder.heads_3d = nn.ModuleList([nn.Linear(8, 3)])      # 2D weight -> adamw (head)
        self.scene_encoder = nn.Module()
        self.scene_encoder.encoder = TinyEncoder()                    # blocks -> muon_enc
        self.scene_encoder.kv_proj = nn.Linear(8, 8)                 # 2D weight -> muon_dec
        self.query_encoder = nn.Module()
        self.query_encoder.kpt_embed = nn.Embedding(5, 8)            # 2D but Embedding -> adamw
        self.conv = nn.Conv2d(3, 4, 3)                               # 4D -> adamw
        self.norm = nn.LayerNorm(8)                                  # 1D weight + bias -> adamw
        self.video_encoder_unfreeze_iter = unfreeze_at
        self.video_encoder_finetune_last_n_layers = n_last
        self.video_encoder_requires_grad = unfreeze_at is None
        if unfreeze_at is not None:
            for p in self.scene_encoder.encoder.parameters():
                p.requires_grad_(False)

    def unfreeze_video_encoder(self, iteration):
        """Upstream's contract verbatim (tracker_encoder.py:262)."""
        if self.video_encoder_unfreeze_iter is None:
            return False
        if self.video_encoder_requires_grad:
            return False
        if iteration < self.video_encoder_unfreeze_iter:
            return False
        enc = self.scene_encoder.encoder
        for p in enc.parameters():
            p.requires_grad_(False)
        for blk in enc.blocks[-self.video_encoder_finetune_last_n_layers:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        self.video_encoder_requires_grad = True
        return True


def _optim_cfg(**over):
    base = dict(learning_rate=1e-4, kpt_lr=5e-4, encoder_lr_scale=0.1, weight_decay=0.002,
                warmup_steps=0, beta1=0.9, beta2=0.95, muon_schedulefree=True, muon_lr_scale=1.0,
                muon_momentum=0.95, muon_warmup_steps=0, muon_adjust_lr_fn='match_rms_adamw')
    base.update(over)
    return base


def _scorer(n_keypoints, **over):
    """A small scorer. `video_encoder_pretrained=False` -- a test must not touch the network."""
    cfg = {**SMALL, 'box_prompt': 'film', 'video_encoder_pretrained': False, **over}
    cfg.pop('query_encoder', None)
    return build_scorer(cfg, n_keypoints, **SCORER_KW)
