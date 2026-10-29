"""`LoaderConfig`: everything the pose loader may vary, validated at construction, and
`loader_config_from`, the ONE place a `[data]` block becomes a loader config (pose training, scorer
training and scorer QC). Defaults here serve tests and library use; runs layer over `base.toml`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoaderConfig:
    """Everything the loader is allowed to vary. Deliberately short."""
    n_frames: int = 24
    # cameras are resized so max(W,H) == this
    image_size: int = 256
    # int, or a [low, high] pair drawn per item (tracktail's own `sample_cameras`); 0 means every
    # camera. The pretrained tracker this finetunes from was trained at [1, 8], so a fixed count
    # above that is out of distribution as well as slow.
    cams_to_sample: int | list = 0
    # the reference's [dataset.val] value
    val_cams_to_sample: int | list = 5
    # rate at which a training item uses the single-camera image-plane 2D path
    prob_2d_only: float = 0.25
    # semantic switch for 3D single-view sampling; legacy keeps 3D targets in one camera
    two_d_sampling: str = 'true-2d-hybrid-v1'
    # sample datasets uniformly, not proportionally
    balance_datasets: bool = True
    # in-plane rotation, per-camera appearance, cutout
    aug_prob: float = 0.4
    # in-plane rotation magnitude (180 = a full-circle draw); split out of `aug_prob`
    aug_rotation_deg: float = 180.0
    # None means "follow aug_prob"; set it to dial rotation without moving appearance jitter.
    aug_rotation_prob: float | None = None
    # train-only source-canvas reflection for native/forced 2D items; zero is byte/RNG-identical.
    flip_2d_prob: float = 0.0
    flip_2d_modes: list = field(default_factory=lambda: ['horizontal'])
    # per-FRAME appearance: motion blur, sensor noise
    per_image_aug_prob: float = 0.25
    # rate at which a train item drops colour entirely
    grayscale_prob: float = 0.2
    # rate at which a train item inverts colour (black <-> white)
    invert_prob: float = 0.0
    # box centre jitter, fraction of box size
    crop_jitter: float = 0.3
    # box scale jitter
    crop_jitter_scale: float = 0.3
    # THE crop rule's three parameters (data/crop.py): floor, pad px, pad cap as a fraction of the
    # extent's longer side (inf = no cap). Shipped 32 / 5 / 0.05 since 2026-10-29 (owner); the
    # legacy rule every earlier run trained under is 64 / 20 / inf -- pin it to continue one.
    min_crop_dim: int = 32
    crop_pad_px: float = 5.0
    crop_pad_frac: float = 0.05
    # What the crop rule bounds: `keypoints` (the labels) or `instances` (the `instances.pq`
    # box), for a root whose stored keypoints are too sparse to enclose the animal. INERT where
    # a root ships no table (falls back to keypoints per view).
    box_source: str = 'instances'
    # fraction of TRAINING STEPS that run fully query-free
    prompt_dropout: float = 0.4
    # sigma on the prior, in PIXELS (3D scales by cube_scale)
    prompt_noise_px: float = 0.0
    # sigma of a WHOLE-BODY offset, one vector per item (the shape of a deployment failure)
    prompt_offset_px: float = 0.0
    # max frames the prior may describe away from `prompt_t`
    prompt_stale_frames: int = 0
    # Per-KEYPOINT Bernoulli: a selected prior is REPLACED by another finite keypoint's ORIGINAL
    # position (uniform, independent per point; not a permutation). Models a `--anchor carry` swap.
    prompt_swap_kpt_pairs: float = 0.0
    # Per-KEYPOINT Bernoulli: the prior becomes the ONE picked neighbour animal's point, where that
    # point is usable (`prior_out_of_bounds`), so the configured rate bounds the presented one.
    prompt_swap_animal: float = 0.0
    # 0 -> non-overlapping windows for val/test
    val_stride: int = 0
    # Frame stride for a TRAIN window, drawn per item; repeat an entry to weight it.
    # Val/test are always 1.
    frame_strides: list = field(default_factory=lambda: [1])
    # Train sampling mix, a TWO-LEVEL draw: source first, then mode within that source. Either
    # level is skipped where a dataset offers no choice; None leaves that level alone (entries
    # keep their natural share).
    # P(a step comes from an `annotated` session)
    annot_frac: float | None = None
    # P(3d | source), i.e. applied WITHIN each source
    mode_3d_frac: float | None = None
    # Relative P(dataset root) on train, keyed by root NAME (absent = 1.0); applied on top of
    # `balance_datasets`'s uniform-over-roots draw. None/empty = uniform (byte-identical).
    dataset_weights: dict | None = None
    # 'none' | 'film': emit a per-frame animal box (`box_prompt.compute_box_prompt`); set by
    # `scripts/train.py` from `[model].box_prompt` so the two cannot disagree
    box_prompt: str = 'none'
    # 'all' (per frame) | 'first' (the starting frame's box)
    box_prompt_frames: str = 'all'
    # fraction of STEPS the box is withheld (no-box token)
    box_prompt_dropout: float = 0.0
    # of the items ALREADY box-dropped, the fraction that also get a CLEAN pose prior (no
    # dropout/offset/noise); 0.0 makes no extra rng call (byte-identical)
    pose_only_prob: float = 0.0
    # exposure bias: the deployed box is a DETECTOR box
    box_prompt_jitter: float = 0.0
    box_prompt_scale_jitter: float = 0.0

    def __post_init__(self):
        """Validate the opt-in 2D reflection configuration at construction time."""
        try:
            probability = float(self.flip_2d_prob)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'flip_2d_prob must be a number in [0, 1], got {self.flip_2d_prob!r}') from exc
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f'flip_2d_prob must be in [0, 1], got {self.flip_2d_prob!r}')
        if not isinstance(self.flip_2d_modes, (list, tuple)):
            raise ValueError(
                f'flip_2d_modes must be a list of "horizontal"/"vertical" values, '
                f'got {self.flip_2d_modes!r}')
        modes = list(self.flip_2d_modes)
        allowed = ('horizontal', 'vertical')
        unknown = [mode for mode in modes if not isinstance(mode, str) or mode not in allowed]
        if unknown:
            raise ValueError(
                f'flip_2d_modes must contain only {allowed}, got {unknown[0]!r}')
        if len(set(modes)) != len(modes):
            raise ValueError(f'flip_2d_modes must contain no duplicate entries, got {modes!r}')
        if probability > 0 and not modes:
            raise AssertionError(
                'flip_2d_modes must be non-empty when flip_2d_prob > 0; '
                'a positive probability has no legal mode to draw')
        self.flip_2d_prob = probability
        self.flip_2d_modes = modes
        self._check_values()

    def _check_values(self) -> None:
        """Refuse, at construction, the values the loader would otherwise first trip over inside a
        worker at the first item -- or, for `box_prompt`, silently read as ON."""
        from ..pose.box_prompt import BOX_PROMPT_FRAMES
        from ..pose.query_encoder import BOX_MODES

        sampling = ('true-2d-hybrid-v1', 'legacy-3d-single-view')
        if self.two_d_sampling not in sampling:
            raise ValueError(f'two_d_sampling must be one of {sampling}, '
                             f'got {self.two_d_sampling!r}')
        if self.box_prompt_frames not in BOX_PROMPT_FRAMES:
            raise ValueError(f'box_prompt_frames must be one of {BOX_PROMPT_FRAMES}, '
                             f'got {self.box_prompt_frames!r}')
        if self.box_prompt != 'none' and self.box_prompt not in BOX_MODES:
            raise ValueError(f"box_prompt must be 'none' or one of {BOX_MODES}, "
                             f'got {self.box_prompt!r}')
        for name in ('cams_to_sample', 'val_cams_to_sample', 'crop_inflate'):
            spec = getattr(self, name)
            if isinstance(spec, (list, tuple)):
                if len(spec) < 2 or float(spec[0]) > float(spec[1]):
                    raise ValueError(f'{name} must be a number or a [low, high] pair with '
                                     f'low <= high, got {spec!r}')
        p = self.aug_rotation_prob
        if p is not None and not 0.0 <= float(p) <= 1.0:
            raise ValueError(f'aug_rotation_prob must be None or in [0, 1], got {p!r}')
        self.crop_pad_px, self.crop_pad_frac = float(self.crop_pad_px), float(self.crop_pad_frac)
        if not (np.isfinite(self.crop_pad_px) and self.crop_pad_px >= 0):
            raise ValueError(f'crop_pad_px must be a finite number >= 0, got {self.crop_pad_px!r}')
        if not self.crop_pad_frac > 0:
            raise ValueError(f'crop_pad_frac must be > 0 (inf = no cap), got {self.crop_pad_frac!r}')

    # widen the crop-rule box about its centre (1.0 = inert). A float or a [low, high] pair drawn
    # per TRAIN item; val/test use the MIDPOINT so `checkpoint_best` reads one geometry
    crop_inflate: float | list = 1.0


LOADER_PLUMBING = ('path', 'num_workers')
SCORER_PLUMBING = LOADER_PLUMBING + ('val_num_workers', 'prefetch_factor', 'worker_cv_threads')


def loader_config_from(data_cfg: dict, *, extra=LOADER_PLUMBING) -> LoaderConfig:
    """`[data]` -> a `LoaderConfig`, refusing any key neither it nor the caller reads.

    An unknown key would otherwise be ignored and the run would report as an arm it is not. The
    ONE place pose training, scorer training and scorer QC turn a `[data]` block into a loader.

    Inputs: data_cfg -- the `[data]` block; extra -- keys the caller reads itself (plumbing such
            as `path`/`num_workers`), allowed but not passed to the dataclass.
    Outputs: a `LoaderConfig`.
    Side effects: none, or raises SystemExit naming the unknown keys.
    """
    fields = LoaderConfig.__dataclass_fields__
    known = set(fields) | set(extra)
    unknown = set(data_cfg) - known
    if unknown:
        raise SystemExit(
            f'[data]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at the defaults and report as the arm it is not. Known keys: {sorted(known)}')
    return LoaderConfig(**{k: v for k, v in data_cfg.items() if k in fields})
