"""The training loader: tailcycle-dataset on disk -> the batch posetail's model consumes.

Three sampling modes, decided per item:

- **3D multiview**   -- a `mode = "3d"` session, `cams_to_sample` cameras, targets in world mm
- **3D single-view** -- a 3D session shown ONE camera, targets still world mm, plus the 2D
  reprojection as `p2d`. Fired with `prob_2d_only`; this is exactly what posetail's own
  `prob_2d_only` does (`posetail_dataset.py:838-846, 971-975`) despite the name -- it subsets
  the cameras and adds `p2d`, and never converts `coords` to pixels. It is what teaches the
  model to recover metric 3D from a single view.
- **2D single-view** -- a `mode = "2d"` session: one camera, targets in crop pixels (R=2).

Mode is a property of the sampled SESSION, not of the run, so one `train/` may hold both, and
both head-bank slots (`mode_idx` 0 and 1) get gradient in one run.

Two rules that are not negotiable:

1. **Keypoints are never filtered.** The library's `filter_keypoints` drops keypoints seen by
   too few views, which shrinks N so array position stops equalling keypoint identity -- and
   nothing in the loss curve shows it. Every item here carries all K of its session's keypoints,
   in registry order.
2. **T >= 2, always.** posetail computes `gT = T // tubelet_size`, which is 0 at T=1 and yields a
   zero-length positional embedding (`encoder_decoder.py:748`). Short groups clamp-pad.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.datasets.posetail_dataset import custom_collate, rotate_camera_image_plane_3d
from posetail.posetail.cube import get_camera_scale, is_point_visible, project_points_torch

from . import crop as cropmod
from . import memory as _memory
from .crop import BOX_SOURCES
from .format import PROJECTED, UNLABELED, VISIBLE, Registry, load_datasets


@dataclass
class LoaderConfig:
    """Everything the loader is allowed to vary. Deliberately short."""
    n_frames: int = 24
    image_size: int = 256              # cameras are resized so max(W,H) == this
    # An int, or a [low, high] pair drawn per item -- posetail's own `sample_cameras`
    # (`posetail_dataset.py:1258`). 0 means every camera. The pretrained tracker this finetunes
    # from was trained at [1, 8] (`config_encoder_3d_finetuning_h100.toml:22`), so a fixed count
    # above that is out of distribution as well as slow: johnson-mouse's 16-camera sessions ran
    # at 2.9 s/it against branson's 0.25, and s/it tracks camera count almost linearly.
    cams_to_sample: int | list = 0
    val_cams_to_sample: int | list = 5  # the reference's [dataset.val] value
    prob_2d_only: float = 0.25         # rate at which a 3D session is shown a single camera
    balance_datasets: bool = True      # sample datasets uniformly, not proportionally
    aug_prob: float = 0.25             # in-plane rotation, per-camera appearance, cutout
    # The in-plane rotation's magnitude and rate, split out of `aug_prob` because they are the two
    # things a root actually wants to set. 180 is a FULL 360 draw, and on a wide frame it costs
    # nothing over the historic 45: `_rotated_rect_max_inscribed` is 90-degree PERIODIC, so the
    # border-free canvas [-45,45] lands on is the same one [-180,180] lands on (measured on
    # rat-city's 2.29:1 frame: mean retained area 0.416 vs 0.417, min 0.218 both). What the wider
    # draw buys is heading coverage -- the 2D rotation moves pixel coords directly with no Z-roll,
    # so +-45 shows the model a quarter of the circle and +-180 all of it.
    aug_rotation_deg: float = 45.0
    # None means "follow aug_prob", which is what every run before this key did. Set it to dial
    # rotation without moving appearance jitter and cutout with it: an overhead arena camera wants
    # every item rotated and does NOT want every item blurred and cut out.
    aug_rotation_prob: float | None = None
    per_image_aug_prob: float = 0.25   # per-FRAME appearance: motion blur, sensor noise
    grayscale_prob: float = 0.2        # rate at which a train item drops colour entirely
    crop_jitter: float = 0.3           # box centre jitter, fraction of box size
    crop_jitter_scale: float = 0.3     # box scale jitter
    min_crop_dim: int = 64
    # What the crop rule bounds: `keypoints` is the labels themselves, `instances` the
    # `instances.pq` box, for a root whose stored keypoints are too sparse to enclose the animal.
    # INERT where a root ships no table (falls back to keypoints per view, `crop._crop_source`).
    # A root whose table holds MARS/COCO boxes gets a DIFFERENT extent than a keypoint crop, so a
    # number on one is not comparable to a run trained on the other; it rides in the run config.
    box_source: str = 'instances'
    prompt_dropout: float = 0.4        # fraction of TRAINING STEPS that run fully query-free
    prompt_noise_px: float = 0.0       # sigma on the prior, in PIXELS (3D scales by cube_scale)
    # The two corruptions that have the SHAPE of a deployment failure, where the jitter above does
    # not. Both off by default: CLAUDE.md lists prompt corruption as deliberately deleted, and these
    # reopen it with a measurement behind them rather than by preference. See the prior block below.
    prompt_offset_px: float = 0.0       # sigma of a WHOLE-BODY offset, one vector per item
    prompt_stale_frames: int = 0        # max frames the prior may describe away from `prompt_t`
    # TWO MORE FAILURES WITH THE SHAPE OF A DEPLOYMENT ERROR (dev/plans/
    # prompt_prior_corruptions.md), both off by default and both aimed at `--anchor carry`
    # specifically: a carried pose that swapped two keypoints, or latched onto the wrong animal.
    #
    # A COUNT, NOT A PER-KEYPOINT RATE -- K ranges 4 (rat-city, 3dpop) to 47 (allen-mouse) across
    # roots trained jointly, so a per-keypoint probability would mean 12x more corruption on one
    # root than the other from one config line (the same quantisation CLAUDE.md already records
    # for --pose-nms: 'quote it as a COUNT'). The EXPECTED number of transposed pairs per prompted
    # item; floor(x) + Bernoulli(frac(x)) draws it, clamped to n_finite // 2. A TRANSPOSITION, not
    # a copy: p_k := p_j while p_j keeps its own value would duplicate one point and delete
    # another, moving scene_center/cube_scale for nothing -- a transposition leaves the prior SET
    # unchanged, so those whole-set scalars stay bit-identical and this may be drawn PER KEYPOINT
    # with none of prompt_dropout's per-item warning applying.
    prompt_swap_kpt_pairs: float = 0.0
    # P(per prompted ITEM) the WHOLE prior becomes an ELIGIBLE neighbour's pose instead of this
    # animal's own -- drawn PER ITEM, unlike the pairs above, because a wrong-animal prior moves
    # scene_center onto the wrong animal and that whole-set scalar is the point (the same
    # per-item/per-keypoint split prompt_dropout's own docstring argues for). 'Eligible' is
    # prior_out_of_bounds applied to the CANDIDATE, reused rather than re-derived: it is the same
    # rule --anchor carry already uses to decide a prior is usable, which also bounds how far a
    # neighbour can be and still be a reachable 3D residual target (head_3d_grid_radius). A
    # session with fewer than two animals never even draws this coin.
    prompt_swap_animal: float = 0.0
    # Fraction of keypoints that jump with the animal swap; 1.0 (default) is the WHOLE prior and is
    # INERT IN THE RNG STREAM -- no per-keypoint draw happens at 1.0, the same contract
    # crop_inflate = 1.0 already has. Below 1.0 is a DIFFERENT geometry, not a weaker one (it puts
    # scene_center between two animals rather than on the neighbour), so treat it as its own arm
    # rather than a dial on the 1.0 result.
    prompt_swap_animal_frac: float = 1.0
    val_stride: int = 0                # 0 -> non-overlapping windows for val/test
    # Frame stride for a TRAIN window, drawn per item -- posetail's `interval`
    # (`posetail_dataset.py:343-361`). [1] is consecutive frames, i.e. no augmentation. Repeat an
    # entry to weight it: [1, 1, 2, 4] draws stride 1 half the time. Val/test are always 1.
    frame_strides: list = field(default_factory=lambda: [1])
    # Train sampling mix, a TWO-LEVEL draw: source first, then mode within that source. Either
    # level is skipped where a dataset offers no choice -- an all-tracked root ignores
    # `annot_frac`, a single-mode root ignores `mode_3d_frac` -- so one setting serves every root.
    # None leaves that level alone (entries keep their natural share); both None is uniform, which
    # is what every run before this change did. See `_pool_weights` for why this is load-bearing.
    annot_frac: float | None = None    # P(a step comes from an `annotated` session)
    mode_3d_frac: float | None = None  # P(3d | source), i.e. applied WITHIN each source
    # THE BOX PROMPT (report 27), the DATA side. 'none' | 'film' | 'term' -- when not 'none' the
    # loader emits a per-frame animal box (`box_prompt.compute_box_prompt`) for a box-prompt model
    # to consume as a non-position channel. 'none' emits nothing, so a plain run is byte-identical
    # (the item has one fewer field and `pose_collate` adds no `box_prompt` key). Set by
    # `scripts/train.py` from `[model].box_prompt` so the two cannot disagree.
    box_prompt: str = 'none'
    box_prompt_frames: str = 'all'     # 'all' (per frame) | 'first' (the starting frame's box)
    box_prompt_dropout: float = 0.0    # fraction of STEPS the box is withheld (no-box token)
    box_prompt_jitter: float = 0.0     # exposure bias: the deployed box is a DETECTOR box
    box_prompt_scale_jitter: float = 0.0
    # WIDE-CROP TRAINING (report 27's endpoint-1 mechanism).
    # Widen the crop-rule box about its centre by this factor BEFORE the coords are shifted into
    # it, in both the 2D and 3D branches of `_item` -- so the animal sits off-centre in a wider
    # crop that includes more of any neighbour, and `compute_box_prompt` (run post-hoc on the
    # returned coords) reads the animal's TIGHT extent within that wider crop, making the box the
    # only non-centred cue for which animal. 1.0 is INERT and byte-identical to a config without
    # this key -- `_inflate_crop_box` is a no-op at 1.0, asserted by `tests/test_dataset.py`.
    #
    # A float, or a `[low, high]` pair drawn per TRAIN item -- the same `cams_to_sample` contract
    # (`_n_cams`), reused here as `_crop_inflate_draw` rather than a second ad hoc parser. `[0.9,
    # 1.5]` shows the model a range of context rather than one fixed geometry every step, which is
    # a diversity lever a single scalar cannot be. VAL/TEST NEVER DRAW: `_crop_inflate` returns the
    # range's MIDPOINT there (a scalar's own value if not a range), exactly as `crop_jitter` is
    # gated to train-only by `_jitter` -- `checkpoint_best` selection has to read the SAME crop
    # geometry every val pass, or a run's own val curve would be noise from the geometry draw
    # rather than the weights. Sweeps launched under a scalar (report 33's `sweep_box_wide`) are
    # unaffected: a plain float is not a range and is applied identically everywhere, as before.
    crop_inflate: float | list = 1.0


# pixels

def _rotate_2d(cam, coords, angle_deg):
    """`rotate_points_image_plane` WITHOUT its border-free inscribed crop. Same return shape.

    The library rotates the whole frame, expands the canvas so no pixel is lost, and then crops to
    the largest axis-aligned rectangle containing no black border. That last step is right for a
    consumer that uses the whole frame and catastrophic for one that crops around an animal: the
    inscribed rectangle of a 4696x2048 frame keeps a MEAN 0.416 of its area (0.218 at the worst
    angle), and an animal outside it is not skipped -- the `< 2 finite` guard runs BEFORE the
    rotation and `crop_box_for_points` clamps rather than returning None, so the item comes back
    with its labels silently NaN'd by the post-crop `_mask_outside`.

    MEASURED on rat-city-annotated at `aug_rotation_prob = 1.0`: finite label coordinates per item
    2.94 unrotated against 0.99 through the library, with 61% of rotated items carrying NO LABEL AT
    ALL against 1.3%. At the historic `aug_prob = 0.25` that is ~16% of items quietly unsupervised
    in every run on record; at the rotation rate an overhead arena camera wants it is two thirds.

    Keeping the full expanded canvas costs nothing here. The crop that follows is a ~256 px window
    around one animal, so it only meets the canvas edge for an animal that was at the frame edge,
    and there `_crop_affine` already renders BORDER_CONSTANT zeros -- which its docstring calls out
    as exactly what the old pad-safe crop buffer existed to produce. So this trades a black wedge
    on the rare edge crop for the two thirds of items above.

    THE 3D PATH IS DELIBERATELY LEFT ON THE LIBRARY HELPER. It has the same inscribed crop, but it
    recomputes `vis_2d` through `is_point_visible` right after rotating (`_item`), so a point off
    the canvas becomes INVISIBLE -- a label the loss handles -- instead of vanishing into NaN.

    THAT MITIGATION IS NARROWER THAN IT READS, and this docstring used to overstate it. It is
    gated on `vis_2d is not None`, so it never runs for 3dpop, branson-fly or johnson-mouse, whose
    rows are all `projected` and whose visibility is therefore withheld; and it never recomputed
    the 3D noisy-OR at all. An animal outside the inscribed rectangle then projected out of
    `cam['size']`, where `crop_box_for_points` CLAMPS rather than refusing -- so the item came back
    as a `min_crop_dim` corner crop of background carrying full-strength world targets, with
    `_item`'s `< 2 finite` guard already behind it. `_item` now REVERTS a rotation that costs a
    camera the animal, and takes the noisy-OR after the augmentation rather than before it.
    """
    import cv2

    w, h = cam['size'].tolist()
    center_x = float(cam['mat'][0, 2].item()) - float(cam['offset'][0].item())
    center_y = float(cam['mat'][1, 2].item()) - float(cam['offset'][1].item())
    M = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    rot = corners @ M[:, :2].T + M[:, 2]
    tx, ty = -rot.min(axis=0)
    M[0, 2] += tx
    M[1, 2] += ty
    cw = int(np.ceil(rot[:, 0].max() - rot[:, 0].min()))
    ch = int(np.ceil(rot[:, 1].max() - rot[:, 1].min()))

    out = dict(cam)
    # Principal point tracks the canvas expansion only -- there is no crop offset to subtract now.
    # ext/ext_inv/center stay untouched: the 2D coords ARE the target and moving them through the
    # same affine as the pixels is the whole rule; a Z-roll would double-apply it.
    out['mat'] = cam['mat'].clone()
    out['mat'][0, 2] = cam['mat'][0, 2] + tx
    out['mat'][1, 2] = cam['mat'][1, 2] + ty
    out['offset'] = cam['offset'].clone()
    out['size'] = torch.tensor([cw, ch], dtype=torch.int32, device=cam['size'].device)

    Mt = torch.as_tensor(M, dtype=coords.dtype, device=coords.device)
    return out, coords @ Mt[:, :2].T + Mt[:, 2], (M, (cw, ch))


def _apply_affine(pts, rotation):
    """Move (...,2) pixel points through a rotation's own 2x3, or pass them through untouched.

    Both rotation helpers return `(M_2x3, (w, h))` (`posetail_dataset.py:104,175`), and this is
    the same line `rotate_points_image_plane` applies to the coords -- shared so a stored box and
    the labels can never end up in different frames.

    A LIST of those is a PER-FRAME rotation, and then `pts` is indexed on its
    leading axis, which is time in every caller: `(T,K,2)` labels and `(T,4,2)` stored box corners
    alike. Same requirement as the single form -- whatever moves the pixels must move the points.
    """
    if pts is None or rotation is None:
        return pts
    if isinstance(rotation, list):
        return torch.stack([_apply_affine(pts[t], rotation[min(t, len(rotation) - 1)])
                            for t in range(pts.shape[0])])
    M = torch.as_tensor(rotation[0], dtype=pts.dtype)
    return pts @ M[:, :2].T + M[:, 2]


def prior_out_of_bounds(p, mode, cgroup):
    """Which keypoints of a prior are NOT usable as one. (K,) bool, in the MODEL's own frame.

    MOVED HERE FROM `infer.py` (dev/plans/prompt_prior_corruptions.md) so the loader's own prior
    corruptions -- the animal-swap jump, specifically -- can reuse the identical rule rather than
    re-deriving it: a second copy is exactly the failure mode this docstring already warns about
    (`carry` had it, `self` did not, and the two silently disagreed). `infer.py` re-imports this
    name from here; nothing about its behaviour changed.

    A PRIOR OUTSIDE THE CROP IS NOT A PRIOR. In 2D that is a point outside the crop rectangle; in
    3D it is a point no PAIR of cameras can see, since a point one camera sees is not
    reconstructible and so cannot be a position the model should trust. A MOVING camera answers per
    frame ((T,K) rather than (K,)) and the prior is one pose for the whole window, so a camera
    counts if it saw the point at any point during it.

    ONE COPY OF THE RULE, called from both prompted regimes. `carry` had it and `self` did not, so
    the two label-free regimes disagreed about what counts as a prior -- and `self` is the one the
    periodic val eval reports, so training and deployment were being scored under different rules.
    Masking this was worth MOTA +0.041, miss -0.032 SIG and idsw 24 -> 13 on rat-city under `carry`.
    """
    if mode == '2d':
        w, h = (float(x) for x in cgroup[0]['size'][:2])
        return (p[:, 0] < 0) | (p[:, 0] >= w) | (p[:, 1] < 0) | (p[:, 1] >= h)
    seen = []
    for c in cgroup:
        v = is_point_visible(c, p, margin=2)
        seen.append(v.any(0) if v.ndim > 1 else v)
    return torch.stack(seen).sum(0) < 2


def _rotate_camera_group_with_neighbours(cgroup, coords, others):
    """`PosetailDataset.rotate_camera_group`, optionally carrying a SECOND point set through the
    identical draw. Returns `(cgroup, coords, others)`; `others` is None in, None out.

    `others` is None on every item that does not draw `prompt_swap_animal` (the default), and then
    this is EXACTLY `PosetailDataset.rotate_camera_group(None, cgroup, coords)` -- no concatenation,
    no extra tensor, no extra draw, so a plain config is byte-identical (dev/plans/
    prompt_prior_corruptions.md Section 1, correction 6's underlying claim: the world rotation is
    drawn from the GLOBAL numpy RNG inside the library call, not from this loader's own `rng`, and
    is drawn exactly once here either way).

    When `others` (T, n_other, K, R) is given, it rides through the SAME `rotmat` as `coords` by
    being concatenated onto the keypoint axis for the one call and split back off after --
    `coords @ rotmat` is a per-point operation, so this changes nothing about `coords`' own output
    (no cross-talk between the concatenated rows) and needs no reimplementation of the rotation
    itself. Callers must NOT pass this combined tensor to `crop_to_points_3d`: the crop is the
    TARGET's own extent, and widening it with a neighbour would change the item's geometry, which
    is the one thing dev/plans/prompt_prior_corruptions.md Section 4.4 requires never happens.
    """
    from posetail.datasets.posetail_dataset import PosetailDataset

    if others is None:
        cgroup, coords = PosetailDataset.rotate_camera_group(None, cgroup, coords)
        return cgroup, coords, None
    T, K = coords.shape[0], coords.shape[1]
    combined = torch.cat([coords, others.reshape(T, -1, 3)], dim=1)
    cgroup, combined = PosetailDataset.rotate_camera_group(None, cgroup, combined)
    return cgroup, combined[:, :K], combined[:, K:].reshape(others.shape)


def _crop_affine(src_wh, crop_coords, target_size, rotation):
    """The one dst<-src affine for rotate -> crop -> resize. Returns (M_2x3, (w, h)) or None.

    None means all three are no-ops and the caller should not warp at all (the detector reads
    whole frames).

    Composing them is not a micro-optimisation. Done in sequence, the rotation warps the WHOLE
    frame and the crop then throws >95% of it away: 44 ms per frame on rat-city's 4696x2048
    against 0.2 ms for the composed warp, plus an expanded rotation canvas and a zero-filled crop
    buffer that both disappear here. It also resamples once instead of twice.

    The composition uses the CORNER convention, `x_dst = (x_src - x1) * sx`, not `cv2.resize`'s
    half-pixel one. That is deliberate rather than incidental: `crop.apply_crop` sets
    `cam['offset'] += x1` and `_resize_camera` scales `cam['mat']`, which is exactly this affine on
    continuous coordinates -- so the pixels now agree with the intrinsics that
    `project_points_torch` reprojects through, where the old `resize` was off by half a pixel.
    `test_the_fused_warp_agrees_with_the_camera` is what holds that.

    Out-of-source pixels arrive as BORDER_CONSTANT zeros, which is what the old pad-safe crop
    buffer existed to produce. One behavioural difference, and it is unreachable in practice: a box
    reaching outside the rotated canvas used to be zero-filled there, and now samples the source
    instead (real pixels the inscribed-rect canvas had excluded). `crop_box_for_points` clamps
    every box to the camera's own size, so the loader cannot produce one.
    """
    w, h = src_wh
    M = np.eye(3)
    if rotation is not None:
        M_rot, (w, h) = rotation
        M[:2] = M_rot
    box = (0, 0, w, h) if crop_coords is None else tuple(int(c) for c in crop_coords)
    x1, y1, x2, y2 = box
    tw, th = ((x2 - x1, y2 - y1) if target_size is None
              else (int(target_size[0]), int(target_size[1])))
    if rotation is None and box == (0, 0, w, h) and (tw, th) == (w, h):
        return None
    sx, sy = tw / (x2 - x1), th / (y2 - y1)
    A = np.array([[sx, 0.0, -sx * x1], [0.0, sy, -sy * y1]])
    return (A @ M).astype(np.float32), (tw, th)


def load_warps(path, specs, target_size=None, reduce=1):
    """ONE decode, N affines -> list of (H,W,3) uint8 RGB. All None if the file will not decode.

    `specs` is one `(crop_coords, rotation)` per output frame.

    DECODE COST IS PER FILE AND WARP COST IS PER OUTPUT, and fusing the two -- which is what one
    `load_image` call per output frame does -- charges the first at the rate of the second. That
    is invisible until a caller wants several DIFFERENT crops of the SAME frame: the fused form
    pays N full decodes for one picture, N x 27 ms on a 4696x2048 jpeg, dominating the item. It is
    the same waste `read_frames`' dedupe was written to prevent, arriving from the other side.

    BGR IS KEPT UNTIL AFTER THE WARP so the colour convert still runs on the SMALL output, which
    is the property the fused version had and the reason not to just decode-then-call-load_image.

    `reduce` in {1,2,4,8} decodes at 1/N via libjpeg's DCT-domain decimation -- a proper box
    filter, and cheaper than decoding full size. It is only valid when the caller wants the WHOLE
    frame: `crop_coords` are source pixels and would land in the wrong place at 1/N, so the two
    are mutually exclusive and this asserts rather than silently cropping the wrong region.
    See `detector.data.reduce_factor` for how N is chosen.
    """
    import cv2

    if reduce == 1:
        img = cv2.imread(path)
    else:
        assert all(c is None and r is None for c, r in specs), \
            'reduce decodes at 1/N, so source-pixel crop_coords/rotation would be misplaced'
        flag = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
        img = cv2.imread(path, flag)
    if img is None:
        return [None] * len(specs)
    src_wh = (img.shape[1], img.shape[0])
    out = []
    for crop_coords, rotation in specs:
        aff = _crop_affine(src_wh, crop_coords, target_size, rotation)
        warped = (img if aff is None
                  else cv2.warpAffine(img, aff[0], aff[1], flags=cv2.INTER_LINEAR))
        out.append(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return out


# AN OPEN DECORD READER COSTS ABOUT A GIGABYTE, SO THE CACHE SIZE IS A MEMORY BUDGET, NOT A HINT.
# A `VideoReader` is cheap until it decodes; the first seek allocates a frame pool sized by the
# video, and on calms21's 1024x570 mpeg4s that settles at ~1.0 GB each. Measured on one loader
# process walking calms21 train: 0.98 GB with the index built, 44 GB once 32 readers were live,
# back to 10.4 GB on `cache_clear()`. Times 8 workers that is 350 GB, which is how the calms21
# runs got OOM-killed on a 503 GB host -- twice, silently, as "DataLoader worker killed by
# signal: Killed".
#
# 4 is `ChunkShuffle.mix`, which is exactly how many containers a loader worker is inside at once
# -- and one past it the curve is flat while the memory is not. Measured on calms21 train, 600
# items through one process:
#
#   size 2 -> 2.57 GB, 48% hits, 52.8 ms/item      the pool has 4 videos; 2 of them thrash
#   size 4 -> 2.03 GB, 97% hits,  9.4 ms/item      <-- default
#   size 8 -> 6.62 GB, 97% hits,  9.8 ms/item      3x the memory, no hits, no speed
#   size 32 -> 44 GB (the shipped default), and 8 workers of that is what got OOM-killed
#
# A rig with more than 4 cameras is the other end of the trade: the pose loader touches every
# camera within one window, so a cache below the camera count misses on every call -- a 16-camera
# video rig at 4 re-opened 12 containers per window and ran detection 2.5x slower. Both ends are
# now derived per process by `_reader_cache_size`; `TAILCYCLENET_READER_CACHE` only overrides it.


def _reader_cache_size(n_cams: int, wh, workers: int | None, ram_gb: float | None = None,
                       procs: int | None = None) -> int:
    """How many open decord readers this process may hold.

    THIS FUNCTION MUST NOT DECODE, STAT, OR OPEN ANYTHING (gotcha 11). It reads the environment,
    `/proc` and `/sys` via `memory.current()`, a camera count and a frame size -- all of which are
    parsed strings or virtual files -- because opening a `VideoReader` in the parent to measure
    anything is what deadlocks the forked workers. It is pure given `ram_gb`, which is what makes
    the sizing testable on any host.

    Two axes, because one number cannot serve both callers:

    - **Role.** A single process streaming windows (inference, rendering, `detect_group`) wants the
      WHOLE rig, or it misses on every call. A loader worker wants 4 -- `ChunkShuffle.mix`, how
      many containers a worker is inside at once, where the hit curve above flattens and the memory
      does not. `workers is None` means the main process, which `get_worker_info` also returns for
      `num_workers = 0`; that genuinely is one process, so the divisor below is 1.
    - **Memory.** The count is a wish; RAM is the constraint, and it is per PROCESS while the
      workers multiply it. Half of physical memory, split across the workers, is the ceiling.
      **And there are now `procs` RANKS on this node**, each with its own worker pool, so the
      divisor is `workers x procs`: a 4-gpu job is four times as many decoding processes on the
      same host, and sizing as if it were one is how a job that fits on one gpu gets OOM-killed on
      four. `procs` comes from `TAILCYCLENET_LOCAL_WORLD_SIZE`, which `scripts/train.py` sets from
      `fabric.world_size` immediately after launch -- a parsed-environment fact, so this function
      still opens, stats and decodes nothing (gotcha 11). Absent means 1, i.e. today's numbers.

    `per_gb` is `1.0 + 0.25/megapixel`, a line through the only two points anyone has measured:
    calms21's 1024x570 settles at ~1.0 GB an entry and johnson's 3208x2200 at 2.56 GB (41 GB
    across 16). Both terms are rounded up from the fit, so the estimate errs toward a smaller
    cache -- and the clamp only ever binds inside workers, where small is the safe direction. The
    flat "~1 GB per entry" this replaces was wrong by 2.5x on the rig that needed it most.
    """
    env = os.environ.get('TAILCYCLENET_READER_CACHE')
    if env:
        return max(1, int(env))
    if ram_gb is None:
        # THE JOB'S MEMORY, NOT THE MACHINE'S. `SC_PHYS_PAGES` -- what this used to read -- is the
        # host's total: it cannot see a cgroup cap, an LSF limit, or 288 GB another user already
        # has resident. Under `MemoryMax=16G` it still reported 503 GB, a 20x error, and a 20x
        # over-estimate here is 16 open readers of a 3208x2200 rig (~37 GB) on a node that has 16.
        # `memory.current()` minimises over the cgroup ancestry, LSF's own variables, MemAvailable
        # and MemTotal. It opens no data path (gotcha 11).
        b = _memory.current()
        ram_gb = min(b.limit_gb, b.available_gb)
    if procs is None:
        procs = int(os.environ.get('TAILCYCLENET_LOCAL_WORLD_SIZE', '1') or 1)
    want = 4 if workers else max(int(n_cams), 4)
    per_gb = 1.0 + 0.25 * (int(wh[0]) * int(wh[1])) / 1e6
    share = max(workers or 1, 1) * max(int(procs), 1)
    return max(1, min(want, int(0.5 * ram_gb / share / per_gb)))


# Built at the FIRST video read, not at import: `lru_cache`'s maxsize is fixed at decoration and
# cannot be changed afterwards -- `cache_parameters()` hands back a copy and assigning `.maxsize`
# silently does nothing -- so the only way to size it from the data is to wrap it late.
_readers = None
_cache_lock = threading.Lock()


def _open_reader(path: str):
    """One `VideoReader` per file per process. Opening the container and building its frame index
    is not per-window work, but `read_frames` is called once per window per camera -- so a
    windowed pass over 3dpop's test videos paid it hundreds of times.

    `num_threads=1` because decord's default (0 = one decode context per core) costs 0.60 GB of
    RSS per open reader on a 128-core host against 0.24 GB at one thread. Single-threaded decode
    is 4.5 -> 7.3 ms/frame, which the loader never notices."""
    from decord import VideoReader

    return VideoReader(path, num_threads=1)


def _reader(path: str, group, cam: str):
    """The cached `_open_reader`, sized on first use from the rig this group belongs to.

    `group.session` is not checked: `group.source(cam)` reached us through `Group.dir`, which
    already dereferences `self.session.path`, so a session-less Group cannot get this far.

    FIRST VIDEO READ WINS. A process that later walks a root with a wider rig keeps the size it
    picked, because resizing means dropping live readers.
    # ponytail: first-rig sizing, revisit if one process ever streams two video roots of different
    # widths -- the memory clamp bounds it either way.

    `_cache_lock` covers BOTH the late build and the lookup, because the cameras of one rig are now
    decoded concurrently (`detector.detect_group`): two threads racing the `is None` test would
    build two caches and the first one's readers would leak, and two `lru_cache` misses on the same
    path would open the same 4K container twice. Opening is once per file and off the hot path, so
    holding the lock across it costs nothing measurable; the DECODE is not under it.
    """
    global _readers
    with _cache_lock:
        if _readers is None:
            info = torch.utils.data.get_worker_info()
            rig = group.session.rig
            size = _reader_cache_size(
                len(rig), rig.size(cam), None if info is None else info.num_workers)
            # THE DOCUMENTED CLIFF, SAID OUT LOUD. The pose loop touches every camera inside one
            # window, so a cache below the camera count misses on EVERY call -- a 16-camera video
            # rig at 4 re-opened 12 containers per window and ran 2.5x slower. That used to be
            # reachable only by setting `TAILCYCLENET_READER_CACHE` by hand; now a tight RAM budget
            # can produce it too, and a run that is silently 2.5x slower for a legitimate reason
            # should still say so rather than being diagnosed from a stopwatch.
            if info is None and size < len(rig):
                warnings.warn(
                    f'decord reader cache is {size} for a {len(rig)}-camera rig: every window '
                    f'touches all {len(rig)} cameras, so this misses on every call and re-opens '
                    f'{len(rig) - size} container(s) per window. The RAM budget would not hold '
                    f'more ({_memory.current()}). Raise --max-ram / TAILCYCLENET_MAX_RAM_GB, or '
                    'accept the slowdown.', stacklevel=2)
            _readers = lru_cache(maxsize=size)(_open_reader)
        return _readers(path)


# ONE LOCK PER CONTAINER AROUND THE DECODE, because `_readers` is a module-level cache of STATEFUL
# readers. `VideoReader.get_batch` seeks, so two threads reading the SAME container interleave their
# seeks and get each other's frames -- or crash inside decord. That did not matter while every caller
# was sequential; `scripts/infer.py` renders one group on a background thread while the loop predicts
# the next, and on a video-backed root both touch the same reader.
#
# IT USED TO BE ONE GLOBAL LOCK, WHICH ALSO SERIALISED DIFFERENT FILES -- and different files share
# no state at all, so that was a bound nothing needed. It cost the whole of the multi-camera win:
# 3dpop's four 3840x2160 cameras decode at ~62 ms/frame each on one thread and at ~18 ms/frame-camera
# when the four containers run concurrently (3.5x, measured; decord releases the GIL inside
# `get_batch`). A 16-camera rig is the same argument four times over. The per-path dict is guarded by
# `_lock_lock`, which is held only for the dict lookup and never across a decode.
_lock_lock = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def _read_lock_for(path: str) -> threading.Lock:
    with _lock_lock:
        lk = _path_locks.get(path)
        if lk is None:
            lk = _path_locks[path] = threading.Lock()
        return lk


def _read_video(path, group, cam, frames, crop_coords, target_size, rotation):
    """Frames from a video file. Only 3dpop's test split needs this."""
    import cv2

    key = str(path)
    # DECODE EACH DISTINCT INDEX ONCE, for the reason the image-directory path below decodes once:
    # a clamp-padded window repeats its last frame, and a synthetic-motion window is ONE index T
    # times under T different crops. Handing `get_batch` the repeats decodes them.
    want = [int(i) for i in frames]
    uniq = list(dict.fromkeys(want))
    with _read_lock_for(key):
        dec = _reader(key, group, cam).get_batch(uniq).asnumpy()          # decord gives RGB
    at = {i: dec[n] for n, i in enumerate(uniq)}
    src_wh = (dec.shape[2], dec.shape[1])
    # A (T,4) `crop_coords` -- or a list of rotations -- is a MOVING crop: one affine per frame
    # rather than one for the window. Same count of `warpAffine` calls either way; only the matrix
    # differs per frame.
    cc = None if crop_coords is None else np.asarray(crop_coords)
    per_box = cc is not None and cc.ndim == 2
    per_rot = isinstance(rotation, list)
    out, seen = [], set()
    for t, i in enumerate(want):
        aff = _crop_affine(src_wh,
                           cc[min(t, len(cc) - 1)] if per_box else crop_coords,
                           target_size,
                           rotation[min(t, len(rotation) - 1)] if per_rot else rotation)
        im = at[i]
        if aff is not None:
            out.append(cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR))
        else:
            # A repeat gets a COPY: `_augment`'s cutout writes in place, and now that the decode
            # is deduped the repeats would otherwise share one buffer. Same rule as `read_frames`.
            out.append(im.copy() if i in seen else im)
        seen.add(i)
    return out


def read_frames(group, cam, frames, crop_coords=None, target_size=None, rotation=None,
                pool: ThreadPoolExecutor | None = None, reduce=1):
    """(T,H,W,3) uint8 RGB for one camera, from an image directory or a video.

    `reduce` is honoured for image directories and IGNORED for video: decord has no
    decode-time decimation, so a video root silently returns full-size frames. The caller must
    therefore letterbox with `src_wh` and never assume the returned size.
    """
    kind, src, ext = group.source(cam)
    if kind == 'video':
        return _read_video(src, group, cam, frames, crop_coords, target_size, rotation)
    # Names are computed, not listed. Frame files are `%06d.<ext>` contiguous from 000000 by spec
    # (§12, enforced by `validate_session`), and listing the directory to select T of them cost
    # 0.90 s of a 1.06 s rat-city item -- its `cam0` holds 57,594 entries.
    #
    # DECODE EACH DISTINCT FRAME ONCE. `_frames` clamp-pads a window that runs past the end of its
    # group, and a group shorter than `n_frames` pads entirely -- 251 of johnson-mouse's 624 train
    # windows come from `n_frames = 1` groups, where all 24 indices are frame 0. Without the dedupe
    # that window decodes 24 copies of one image per camera (384 decodes for 16 distinct frames).
    want = [int(i) for i in frames]
    path = lambda i: os.path.join(src, f'{i:06d}{ext}')           # noqa: E731
    # PER-FRAME CROPS MOVE THE DEDUPE KEY. A per-frame box is a function of the POSITION in the
    # window, not of the source index -- and `_frames` clamp-pads a short window, so one index can
    # occupy several positions. Keying on the index alone would then serve every repeat the first
    # position's crop. The key becomes (index, box, rotation slot); with one box and one rotation
    # for the whole window those halves are constant and this is the old behaviour exactly.
    cc = None if crop_coords is None else np.asarray(crop_coords)
    per_box = cc is not None and cc.ndim == 2
    per_rot = isinstance(rotation, list)
    keys = [(w, tuple(int(v) for v in cc[min(t, len(cc) - 1)]) if per_box else None,
             min(t, len(rotation) - 1) if per_rot else None)
            for t, w in enumerate(want)]
    uniq = list(dict.fromkeys(keys))
    # BUT THE DEDUPE KEY IS NOT THE DECODE KEY, and conflating them is what made a synthetic-motion
    # window unaffordable: it is ONE source index under T distinct crops, so every key differs and
    # a fused decode-and-warp per key pays T full decodes for one picture. Group the keys by source
    # index and hand each group to `load_warps`, which decodes once and warps per output. With one
    # box for the window each index has exactly one key and this is a single fused call, as before.
    by_index = {}
    for k in uniq:
        by_index.setdefault(k[0], []).append(k)
    spec = lambda k: (list(k[1]) if per_box else crop_coords,          # noqa: E731
                      rotation[k[2]] if per_rot else rotation)
    batch = lambda i, ks: load_warps(path(i), [spec(k) for k in ks],   # noqa: E731
                                     target_size, reduce)
    if pool is None:
        res = {i: batch(i, ks) for i, ks in by_index.items()}
    else:
        fs = {i: pool.submit(batch, i, ks) for i, ks in by_index.items()}
        res = {i: f.result() for i, f in fs.items()}
    got = {k: im for i, ks in by_index.items() for k, im in zip(ks, res[i])}
    if any(v is None for v in got.values()):
        return [got[k] for k in keys]          # the caller checks for None and drops the item
    # A repeat gets a COPY, not the same array object. `_augment`'s cutout writes in place, so an
    # aliased list would have every repeat sharing one buffer. That is harmless today (imgaug
    # returns fresh arrays, and painting a constant rect twice is idempotent) but it is a trap not
    # worth leaving: the copy is ~20 us against the 27 ms decode it replaces.
    seen, out = set(), []
    for i in keys:
        out.append(got[i].copy() if i in seen else got[i])
        seen.add(i)
    return out


# appearance augmentation

def _crop_inflate(cfg, rng, train):
    """THIS item's crop-inflate factor. `cfg.crop_inflate` is a float, or a `[low, high]` pair --
    same contract as `_n_cams`/`cams_to_sample`, one continuous draw instead of `_n_cams`'
    integer one.

    TRAIN draws uniformly in `[low, high]` per item, so a wide-crop run shows the model a RANGE of
    context rather than one fixed geometry every step. VAL/TEST NEVER DRAW -- they get the range's
    MIDPOINT, for the same reason `_jitter` is gated to `self.train` alone: `checkpoint_best` is
    selected on the val curve, and a val geometry that moved step to step would put noise from the
    crop draw into a number that is supposed to isolate the weights. A plain scalar has no
    range to be a midpoint of and returns unchanged either way, so a scalar config -- everything
    shipped before this key existed a range -- is untouched.
    """
    spec = cfg.crop_inflate
    if not isinstance(spec, (list, tuple)):
        return float(spec)
    lo, hi = float(spec[0]), float(spec[1])
    return float(rng.uniform(lo, hi)) if train else (lo + hi) / 2.0


def _n_cams(spec, rng):
    """How many cameras this item shows. `spec` is an int, or a [low, high] pair drawn per item.

    The pair form is posetail's (`PosetailDataset.sample_cameras`, `posetail_dataset.py:1258-1266`)
    and it is what the pretrained tracker was finetuned with, at [1, 8]. Returning a number larger
    than the session has is fine -- the caller takes every camera in that case, exactly as the
    reference's `if len(cam_names) > num_cams_to_sample` guard does.
    """
    if isinstance(spec, (list, tuple)):
        lo, hi = int(spec[0]), int(spec[1])
        return int(rng.integers(lo, hi + 1))
    return int(spec)


def _even_span(span, ceiling):
    """Round a labelled span up to an even window length in [2, ceiling].

    EVEN because the scene encoder tokenises in tubelets of 2 (`vjepa2.py:103`,
    `gT = view.shape[1] // tubelet_size`), so an odd T silently drops a frame's worth of tokens.
    FLOOR OF 2 because T = 1 gives `gT = 0` and a zero-length pos_embed -- gotcha 1, the failure
    that cost the `memory` branch.
    """
    hi = max(2, int(ceiling) - int(ceiling) % 2)
    return min(max(2, int(span) + int(span) % 2), hi)


def _build_augmenters(cfg):
    """The two appearance pipelines, taken from the reference (`posetail_dataset.py:570-588`).

    The SPLIT is the point, not the list. `per_camera` is sampled once per camera and replayed
    frame by frame, so a camera's colour, gamma and focus hold steady down a clip: appearance is
    an identity cue for a tracker, and re-rolling hue every frame teaches that it is noise.
    `per_image` is resampled per frame, which is what sensor noise and motion blur actually are.

    Cost was last measured at 0.141 s for `per_camera` and 0.052 s for `per_image` on 24 crops of
    256x256 with every augmenter firing, but that predates dropping `DefocusBlur` -- which was
    4.2 ms/frame, the most expensive single entry -- so `per_camera` is now well under it. Watch
    `train/loader_wait_frac`; re-measure the same way before blaming augmentation for it.
    """
    import imgaug.augmenters as iaa

    p, q = cfg.aug_prob, cfg.per_image_aug_prob
    per_camera = iaa.Sequential([
        iaa.Sometimes(p, iaa.GammaContrast((0.6, 1.8))),
        iaa.Sometimes(p, iaa.AddToSaturation((-50, 30))),
        iaa.Sometimes(p, iaa.AddToHue((-10, 10))),
    ])
    per_image = iaa.Sequential([
        iaa.Sometimes(q, iaa.MotionBlur(k=(3, 5))),
        iaa.Sometimes(q, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
        iaa.Sometimes(q, iaa.Multiply((0.9, 1.1))),
    ])
    return per_camera, per_image


def _cutout_rects(rng, size, p2d, vis_2d, cnum):
    """Random-erasing rectangles for one camera, in crop pixels. Mutates `vis_2d` in place.

    A keypoint underneath a rect is no longer visible, and saying so is the whole point: without
    it the model is asked to report "visible" for a patch that has been painted over, which is the
    one label that is definitely wrong.

    `vis_2d` here is THREE-state -- NaN means "no one assessed this camera". Cutout overwrites
    NaN with 0, and that is right rather than an invention: the pixels are now literally covered,
    so "not visible" became a fact about the image we just produced.
    """
    w, h = int(size[0]), int(size[1])
    rects = []
    for _ in range(int(rng.integers(1, 4))):
        rw, rh = int(w * 0.15), int(h * 0.15)
        rx = int(rng.integers(0, max(w - rw, 1)))
        ry = int(rng.integers(0, max(h - rh, 1)))
        rects.append((rx, ry, rx + rw, ry + rh, rng.integers(0, 256, 3).tolist()))
        if vis_2d is not None:
            pts = p2d[cnum]                                    # (T,K,2), crop pixels
            inside = ((pts[..., 0] >= rx) & (pts[..., 0] <= rx + rw) &
                      (pts[..., 1] >= ry) & (pts[..., 1] <= ry + rh))
            vis_2d[:, :, cnum][inside] = 0
    return rects


# the dataset

@dataclass
class _Item:
    """One addressable training unit: an animal in a group, optionally at a fixed start."""
    ds: int
    session: object
    gid: str
    animal: int
    start: int = -1                 # -1 -> pick at random (train)


class PoseDataset(Dataset):
    def __init__(self, path, split: str, cfg: LoaderConfig, registry: Registry | None = None,
                 train: bool | None = None, seed: int = 23,
                 registry_base: Registry | None = None):
        # Gotcha #1, and the clamp-pad in `_frames` does NOT cover it: that pads a short GROUP up
        # to `cfg.n_frames`, which does nothing when `cfg.n_frames` is itself 1. A 1-frame window
        # gives posetail `gT = T // tubelet_size = 0` and a zero-length positional embedding.
        assert cfg.n_frames >= 2, (
            f'n_frames = {cfg.n_frames} is not usable: posetail computes gT = T // tubelet_size '
            '(encoder_decoder.py:748), which is 0 at T=1 and yields a zero-length pos_embed. '
            'Use n_frames >= 2; short groups are clamp-padded up to it.')
        # A typo here would silently mean `keypoints`, which is the old behaviour and so leaves
        # nothing to notice: same shapes, same losses, a run that just quietly ignored the boxes.
        assert cfg.box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {cfg.box_source!r}'
        self.cfg = cfg
        self.split = split
        self.train = (split == 'train') if train is None else train
        self.datasets = load_datasets(path)
        # `registry_base` makes the ids APPEND-ONLY against a run that already exists, so the
        # embedding rows behind them survive a warm start. Without it a second run over the same
        # datasets in a different order silently remaps every row of `kpt_embed` -- gotcha #4,
        # and invisible in the loss curve. `Registry.build` raises if an old id would move.
        self.registry = registry or Registry.build(self.datasets, registry_base)
        self.seed = seed
        # Appearance augmentation is train-only, and `None` is also the flag the pixel path reads.
        # Val must stay clean: a metric computed on augmented pixels is not comparable to the last
        # one, and `test_val_windows_are_deterministic` would fail outright.
        self._aug = _build_augmenters(cfg) if self.train and cfg.aug_prob > 0 else None

        # Scatter every session's parquet into dense arrays HERE, in the parent process, and drop
        # the tables. Forked workers then share the arrays copy-on-write instead of each holding
        # its own copy of a 44 MB table (12 workers x rat-city would be half a gigabyte).
        self.index: list[_Item] = []
        self.by_dataset: list[list[int]] = []
        # Ids are per SESSION, not per dataset: a session may declare the root's keypoints in a
        # different order or use only a subset of them, and its dense K axis follows its OWN
        # `names`. Resolved here so a root that cannot be mapped fails at construction rather
        # than in the middle of an epoch.
        self._kpt_ids: dict[Path, torch.Tensor] = {}
        boxed: list[tuple[str, int, int]] = []
        for di, ds in enumerate(self.datasets):
            mine, n_box, n_sess = [], 0, 0
            for sess in ds.sessions.get(split, []):
                sess.preload()
                n_sess += 1
                self._kpt_ids[sess.path] = torch.as_tensor(
                    self.registry.ids_for(ds.name, sess.names), dtype=torch.long)
                for gid, group in sess.groups.items():
                    lab = sess.labels(gid)
                    # THE TABLE `_item` WILL DEREFERENCE, CHECKED WHERE THE SESSION HAS A NAME.
                    # `_item` picks its target table off `sess.mode` alone, but the window is
                    # admitted below off whichever visibility table exists -- and the spec allows
                    # a `mode = "3d"` session carrying only `keypoints.pq` (§8, validation rule
                    # 12; rule 6 requires only ONE of the two tables). That combination reached
                    # `lab.points3d[a]` as `TypeError: 'NoneType' object is not subscriptable`,
                    # mid-epoch, uncaught -- only a `None` return is retried. Refused here
                    # instead, naming the session and the table, because the loader genuinely
                    # cannot build 3D targets out of 2D labels.
                    need = 'points3d' if sess.mode == '3d' else 'points2d'
                    if getattr(lab, need) is None:
                        raise ValueError(
                            f'{sess.path}: mode is {sess.mode!r} so every window needs '
                            f'{need}, and this session carries none. The format allows it '
                            '(one label table is enough) but training on it does not: there is '
                            'nothing to supervise the targets with.')
                    vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
                    if vis is None:
                        continue
                    for a in range(len(lab.animal_ids)):
                        starts = self._starts(vis, a, group.n_frames)
                        for st in starts:
                            mine.append(len(self.index))
                            self.index.append(_Item(di, sess, gid, a, st))
                n_box += (sess.path / 'instances.pq').exists()
            boxed.append((ds.name, n_box, n_sess))
            self.by_dataset.append(mine)
        if not self.index:
            raise ValueError(f'{path}: split {split!r} yielded no usable windows')
        if cfg.box_source == 'instances':
            # Which roots the switch actually reached. The keypoint fallback is per view and
            # silent by design, so a root that carries no `instances.pq` trains exactly as it did
            # before -- invisible in a loss curve, and the reason this is printed rather than
            # assumed.
            print(f'{split}: box_source=instances  ' + '  '.join(
                f'{n}/{t} {name}' + ('' if n == t else ' (keypoint fallback)')
                for name, n, t in boxed))
        # Sampling pools. A pool is a set of index positions plus an optional cumulative weight
        # array; `_pick` draws a pool, then an entry inside it. Balancing across datasets is the
        # only thing that makes more than one pool, and it is train-only -- val and test address
        # `self.index` directly so a window's identity stays tied to its index.
        multi = self.train and cfg.balance_datasets and len(self.datasets) > 1
        pools = self.by_dataset if multi else [list(range(len(self.index)))]
        self._pools = [(np.asarray(p, dtype=np.int64),
                        self._pool_weights(p) if self.train else None) for p in pools if p]

    # -- indexing ------------------------------------------------------------------------
    def _labelled_frames(self, vis, a):
        """Frames where this animal has at least one assessed keypoint."""
        v = vis[a]
        v = v.reshape(v.shape[0], -1) if v.ndim > 2 else v
        return np.flatnonzero((v != UNLABELED).any(-1))

    def _starts(self, vis, a, n_frames):
        """Window starts for one animal.

        Train indexes at animal granularity and picks the start inside `__getitem__`, so a
        57,594-frame rat-city group costs 12 index entries instead of 691,000. Val and test
        enumerate fixed windows so a metric is reproducible.

        THE FIRST FRAME OF A WINDOW NEED NOT BE LABELLED, on either path. The old v4 loader
        admitted a training window only if its first frame had a finite coordinate, so a group
        whose labels sat in the middle yielded zero windows -- and the natural annotation shape,
        a label with context on both sides, was silently unusable. Here the window is placed
        around the label instead of the label being required at the window's edge.

        Windows are also clamped into the group rather than running off the end: a start beyond
        `n_frames - T` would be clamp-padded with duplicates of the last frame while real context
        sat unused earlier in the group.
        """
        labelled = self._labelled_frames(vis, a)
        if labelled.size == 0:
            return []
        if self.train:
            return [-1]
        T = self.cfg.n_frames
        stride = self.cfg.val_stride or T
        lo, hi = int(labelled[0]), int(labelled[-1])
        limit = max(0, n_frames - T)
        # Start half a window before the first label so the label sits inside the window rather
        # than at frame 0 -- frame 0 is the one frame where per-frame anchoring contributes
        # nothing, so putting every label there would measure the wrong thing.
        first = int(np.clip(lo - T // 2, 0, limit))
        return sorted({min(s, limit) for s in range(first, hi + 1, stride)}) or [0]

    def __len__(self):
        return len(self.index)

    def _pool_weights(self, pool):
        """Cumulative per-entry weights for one pool, or None to sample it uniformly.

        WHY THIS EXISTS. `_starts` returns one index entry per (session, group, animal) on train,
        whatever the group's length, and the sampler draws entries uniformly -- so an entry IS a
        sampling weight, and it is decoupled from how much data sits behind it. On
        allen-mouse-combined that put 90.4% of steps on 63 per-camera 2D sessions holding 1,023
        labelled frames between them, and 3.9% on the tracked session holding 21,500: a 500-frame
        tracked clip is one group, so it buys exactly one entry, the same price a single
        hand-annotated still pays. `mode='2d'` routes to head bank 0 and fires ~3 of 15 loss terms,
        so the 3D bank that validation reads was getting under a tenth of the gradient. On
        johnson-mouse-combined the tracked source sat at 0.3%.

        THE DRAW IS TWO-LEVEL: source (`annot_frac`), then mode WITHIN that source
        (`mode_3d_frac`), then uniform over the entries in that cell. The levels are independent
        by construction rather than jointly fitted, so they can never be mutually infeasible --
        `mode_3d_frac` is a conditional P(3d | source), not a second marginal.

        A level with nothing to choose between is skipped, which is what lets one setting serve
        every root: an all-tracked dataset ignores `annot_frac`, a single-mode one ignores
        `mode_3d_frac`, and a root that is both -- 3dpop, branson-fly, rat-city -- is untouched.
        A level left at None keeps its cells' natural entry shares, so it is not a silent
        rebalance.

        Returned flattened: the product of the two levels is one cumulative array and one
        `searchsorted`, which is the same distribution as drawing twice for less work.
        """
        cfg = self.cfg
        if cfg.annot_frac is None and cfg.mode_3d_frac is None:
            return None
        for name, v in (('annot_frac', cfg.annot_frac), ('mode_3d_frac', cfg.mode_3d_frac)):
            if v is not None and not 0.0 <= float(v) <= 1.0:
                raise ValueError(f'{name} must be in [0, 1], got {v}')

        cells: dict[tuple[str, str], list[int]] = {}
        for j in pool:
            sess = self.index[j].session
            cells.setdefault((sess.label_source, sess.mode), []).append(j)

        def share(key, present, frac, n):
            """P(key) across `present`: the configured fraction, or the natural entry share."""
            if len(present) == 1:
                return 1.0
            if frac is None:
                return n[key] / sum(n.values())
            return float(frac) if key == present[0] else 1.0 - float(frac)

        srcs = [s for s in ('annotated', 'tracked') if any(k[0] == s for k in cells)]
        n_src = {s: sum(len(v) for k, v in cells.items() if k[0] == s) for s in srcs}
        pos = {j: i for i, j in enumerate(pool)}
        w = np.zeros(len(pool), dtype=np.float64)
        for s in srcs:
            p_src = share(s, srcs, cfg.annot_frac, n_src)
            modes = [m for m in ('3d', '2d') if (s, m) in cells]
            n_mode = {m: len(cells[(s, m)]) for m in modes}
            for m in modes:
                js = cells[(s, m)]
                w[[pos[j] for j in js]] = (p_src * share(m, modes, cfg.mode_3d_frac, n_mode)
                                           / len(js))
        if w.sum() <= 0:
            raise ValueError('sampling weights are all zero -- check annot_frac / mode_3d_frac')
        return np.cumsum(w / w.sum())

    def mix(self):
        """Realised share of train steps per (label_source, mode) cell. Reporting only.

        Printed at startup because the mix is the single easiest thing here to get wrong by
        accident and the hardest to see afterwards: it is invisible in the loss curve, and the
        arithmetic that produces it lives in three places (index construction, dataset balancing,
        the two fractions).
        """
        out: dict[str, float] = {}
        for p, cum in self._pools:
            wt = np.diff(cum, prepend=0.0) if cum is not None else np.full(len(p), 1.0 / len(p))
            for j, x in zip(p, wt / len(self._pools)):
                sess = self.index[j].session
                out[f'{sess.mode}-{sess.label_source}'] = (
                    out.get(f'{sess.mode}-{sess.label_source}', 0.0) + float(x))
        return dict(sorted(out.items()))

    def _pick(self, idx, rng):
        # Val and test address the index directly -- a window's identity is its index, which is
        # what `test_val_windows_are_deterministic` rests on.
        if not self.train:
            return self.index[idx]
        # Uniform over datasets, then weighted within: without the outer level, branson-fly's 194
        # groups would outvote allen-mouse's 45 by 4:1 for no reason anyone chose.
        pool, cum = (self._pools[rng.integers(len(self._pools))] if len(self._pools) > 1
                     else self._pools[0])
        if cum is None:
            # Byte-for-byte the previous behaviour, so an unconfigured run is unchanged: one pool
            # replays `idx` straight from the sampler, many pools draw uniformly inside the pool.
            return self.index[idx if len(self._pools) == 1 else pool[rng.integers(len(pool))]]
        return self.index[int(pool[np.searchsorted(cum, rng.random())])]

    # -- item ----------------------------------------------------------------------------
    def __getitem__(self, idx):
        """One window. `idx` is an index, or `(ordinal, index)` from `StepSampler`.

        THE ORDINAL IS WHAT MAKES THE ITEM'S *COST* THE SAME ON EVERY RANK, and that is worth a
        paragraph because it is the difference between 2.5x and 3.4x on 4 gpus. A DDP step costs
        the SLOWEST rank's item, so what decides multi-gpu efficiency is `E[max of N]`, not the
        mean. `cams_to_sample = [2, 8]` is drawn per item and the scene encoder runs once per
        camera, so four independent draws make almost every step a 7-or-8-camera step: measured on
        allen-mouse-combined, the tax is **1.57x at N = 4** (2.55x of a theoretical 4x), and
        synchronising the camera count alone takes it to **1.18x** (3.38x). Synchronising the
        sampling CELL as well adds nothing (1.18x), so it is deliberately left free -- the batch
        stays heterogeneous in data and becomes homogeneous only in cost.

        The ordinal is the item's position in the sampler stream, i.e. the global step, so a
        `shape_rng` keyed on it draws the same camera count on every rank while the window, the
        session and the augmentation stay independent. The MARGINAL distribution is untouched --
        `test_camera_count_distribution_is_unchanged` pins that -- so this is not a recipe change,
        and it applies at every world size so that a 4-gpu run is one lever off a 1-gpu one rather
        than two.

        DRAWN ONCE, OUTSIDE THE RETRY LOOP. A failed item re-picks `idx`, and if the shape were
        redrawn there a rank that retried would consume a draw its peers did not and the ranks
        would fall out of step for the rest of the run.
        """
        ordinal, idx = idx if isinstance(idx, tuple) else (None, idx)
        # Entropy-seeded on train so workers do not replay one another's augmentation;
        # index-seeded on val/test so a metric is reproducible.
        rng = np.random.default_rng(None if self.train else (self.seed, idx))
        shape_rng = (rng if ordinal is None
                     else np.random.default_rng((self.seed, 0x5AFE, int(ordinal))))
        shape = self._shape(shape_rng)
        for _ in range(8):
            out = self._item(idx, rng, shape)
            if out is not None:
                return out
            idx = int(rng.integers(len(self.index)))
        raise RuntimeError(f'{self.split}: 8 consecutive items failed to build')

    def _shape(self, rng) -> dict:
        """The cost-determining draws, made from a stream every rank shares. See `__getitem__`.

        Exactly the two draws that change how much WORK an item is and nothing else: the camera
        count (the encoder runs once per camera) and the single-view coin (which collapses a 3D
        item to one camera). The cell, the session, the window start, T and every augmentation
        stay on the item's own stream, because syncing them was measured to buy nothing and they
        are what makes a batch diverse.
        """
        return {'n_cams': _n_cams(self.cfg.cams_to_sample, rng),
                'single_view_draw': float(rng.random())}

    def _frames(self, item, lab, group, rng):
        """T frame indices, clamp-padded so T >= 2.

        ON TRAIN, T IS DERIVED FROM THE LABELS, and `cfg.n_frames` is only its ceiling. The
        annotated sessions carry ONE labelled frame per 65-frame group, so a fixed T = 24 encodes
        24 frames to supervise 1 -- and with `annot_frac = 0.4` that is 40% of all steps paying 12x
        for nothing. Sizing the window to the labelled span makes those steps T = 2.

        This does NOT mean every encoded frame is supervised: two frames is the floor (gotcha 1 --
        a single-frame window gives `gT = 0` and a zero-length pos_embed), so one label still
        carries an unsupervised partner, and labels that are not adjacent force every frame
        between them to be encoded to span the pair. The claim is only that no frame is encoded
        which is not needed to reach a label.

        ON TRAIN THE FRAMES MAY ALSO BE STRIDED, by `cfg.frame_strides` -- posetail's `interval`
        (`posetail_dataset.py:343-361`), which this loader dropped when it moved to picking the
        start inside `__getitem__`. A stride of s widens the window to s times the wall time for
        the same T, so the model meets motion at more than one time scale. Everything below is the
        derived-T rule re-expressed on a LATTICE of spacing s: a strided window through the anchor
        can only reach labels congruent to it mod s, so the span is measured in lattice steps and
        the start is snapped onto the lattice.

        VAL AND TEST ARE UNTOUCHED -- `_starts` enumerates fixed `cfg.n_frames` windows there at
        stride 1, and a metric whose window geometry moved would not be comparable across
        checkpoints.
        """
        T = self.cfg.n_frames
        vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
        labelled = self._labelled_frames(vis, item.animal)
        if labelled.size == 0:
            return None
        s = 1
        if item.start >= 0:
            start = item.start
        else:
            # A stride is admissible if the group holds the FLOOR window of two frames at it; T
            # is then capped by what the group actually has room for at that stride. Testing the
            # full ceiling instead would reject stride exactly where it is most useful -- an
            # allen group is 65 frames with T = 24, and a 24-wide stride-4 window needs 93.
            fit = [x for x in self.cfg.frame_strides if x <= group.n_frames - 1]
            s = int(fit[rng.integers(len(fit))]) if fit else 1
            # Anchor on a labelled frame, then place the window around it. The old v4 loader
            # required the window's FIRST frame to be labelled, which silently discarded any
            # group whose labels sat in the middle; here the window moves to the label.
            anchor = int(labelled[rng.integers(labelled.size)])
            # Cap T at what the group holds ON THIS ANCHOR'S LATTICE -- the offset eats into the
            # room, and measuring from frame 0 instead lets the last frames clamp onto the end,
            # which reads as a shorter window rather than as the error it is.
            T = min(T, (group.n_frames - 1 - anchor % s) // s + 1)
            # Shrink T to the labelled frames this window could actually reach. `near` is every
            # label some placement of a full-width window over the anchor would cover, so
            # spanning first..last of them is the most T ever has to be. Off-lattice labels are
            # unreachable at this stride, so they are dropped before the span is measured.
            near = labelled[(labelled > anchor - T * s) & (labelled < anchor + T * s)]
            near = near[(near - anchor) % s == 0]
            first, last = int(near[0]), int(near[-1])
            T = _even_span((last - first) // s + 1, T)
            if last - first > (T - 1) * s:
                # The span is wider than the ceiling allows; no placement covers it, so fall back
                # to the anchor and let the draw pick which end of the span it lands on.
                first = last = anchor
            # Bounds that COVER first..last rather than merely containing the anchor -- sizing T
            # to a span and then placing the window off it would pay for frames it never reads.
            span = (T - 1) * s
            lo = max(0, last - span)
            hi = min(first, max(0, group.n_frames - 1 - span))
            lo += (anchor - lo) % s                 # snap up onto the anchor's lattice
            start = int(lo + s * rng.integers(0, (hi - lo) // s + 1)) if hi > lo else lo
        f = np.clip(np.arange(start, start + T * s, s), 0, group.n_frames - 1)
        return f

    def _crop_pts(self, lab, a, frames, cam_ix):
        """One animal's stored boxes over a window, as points the crop rule can bound.

        (T, C, 4, 2) -- FOUR corners per box, not the two the table stores. An in-plane rotation
        turns the box into a rotated rectangle, and the extent of its two diagonal corners is
        strictly inside the extent of all four, so a two-corner version would crop the animal.

        None when the switch is off or the session has no `instances.pq` at all. An all-NaN
        result is legal and handled downstream: `crop._crop_source` falls back per view.
        """
        if self.cfg.box_source != 'instances' or lab.boxes is None:
            return None
        b = lab.boxes[a][frames][:, cam_ix]                        # (T,C,4) = x0,y0,x1,y1
        return torch.as_tensor(cropmod.box_corners(b), dtype=torch.float32)

    def _item(self, idx, rng, shape=None):
        shape = shape or self._shape(rng)
        item = self._pick(idx, rng)
        sess, group = item.session, item.session.groups[item.gid]
        lab = sess.labels(item.gid)
        frames = self._frames(item, lab, group, rng)
        if frames is None:
            return None
        a = item.animal
        K = sess.n_keypoints

        # THE ANIMAL-SWAP PRIOR CORRUPTION (`prompt_swap_animal`, dev/plans/
        # prompt_prior_corruptions.md), drawn HERE -- before any geometry, and before anything
        # else in `_item` touches `rng` -- so the coin is a single shared decision for whichever
        # branch runs below, and a session with fewer than two animals never draws it at all
        # (`lab.n_animals >= 2` short-circuits BEFORE `rng.random()`). Gated on `self.cfg.
        # prompt_swap_animal > 0` first for the same reason: a default config must draw nothing
        # extra and stay byte-identical to one that never mentions the key.
        want_swap_animal = (self.train and self.cfg.prompt_swap_animal > 0
                            and lab.n_animals >= 2
                            and rng.random() < self.cfg.prompt_swap_animal)

        cgroup = sess.cgroup(item.gid, frames)
        # ONE DRAW FOR THE WHOLE ITEM, not one per camera -- the wide-crop regime widens every
        # camera's box by the SAME factor (matching `_jitter`'s one-draw-per-item contract), so a
        # multi-camera 3D window is one consistent geometry rather than each view getting its own
        # unrelated inflation.
        inflate = _crop_inflate(self.cfg, rng, self.train)

        true_2d = sess.mode == '2d'
        single_view = (not true_2d and self.train
                       and self.cfg.prob_2d_only > 0
                       and shape['single_view_draw'] < self.cfg.prob_2d_only)

        # -- pick cameras ----------------------------------------------------------------
        if true_2d:
            cam_ix = [0]
        elif single_view:
            cam_ix = [int(rng.integers(len(cgroup)))]
        else:
            n = shape['n_cams']
            # sorted(), where the reference leaves the draw unsorted: camera order is arbitrary
            # either way, and a stable one makes a window comparable to itself across runs.
            cam_ix = (sorted(rng.choice(len(cgroup), n, replace=False)) if 0 < n < len(cgroup)
                      else list(range(len(cgroup))))
        cgroup = [cgroup[i] for i in cam_ix]
        cam_names = [sess.cam_names[i] for i in cam_ix]
        crop_pts = self._crop_pts(lab, a, frames, cam_ix)      # (T,C,4,2) source px, or None

        # -- targets and visibility ------------------------------------------------------
        if true_2d:
            coords = torch.as_tensor(lab.points2d[a][frames][:, :, 0], dtype=torch.float32)
            # `vis` (the 3D noisy-OR) has nothing to describe here -- there is no 3D layer at
            # R == 2. `vis_2d` is the per-camera (single-camera) target, built the same way the
            # 3D branch builds it below, and gated the same way: `sess.has_visibility_assessment`
            # withholds it for a `tracked` session that never recorded a negative (calms21,
            # rat-city-tracked, branson-fly) so the head is not trained toward "always visible"
            # from labels that assert nothing (see the cached_property's docstring).
            vis = vis_2d = None
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]             # (T,K,1), three-state
                vis_2d = torch.as_tensor(np.where(np.isin(v2, (UNLABELED, PROJECTED)), np.nan,
                                                  (v2 == VISIBLE).astype(np.float32)))
                if not torch.isfinite(vis_2d).any():
                    # Every assessed row in this WINDOW is `projected`/`unlabeled` even though
                    # the session as a whole carries real assessments elsewhere -- withhold
                    # exactly as the 3D branch does, for the same reason.
                    vis_2d = None
        else:
            coords = torch.as_tensor(lab.points3d[a][frames], dtype=torch.float32)
            if lab.vis2d is not None and sess.has_visibility_assessment:
                v2 = lab.vis2d[a][frames][:, :, cam_ix]            # (T,K,c), three-state
                # PER-CAMERA: three states, passed through as three states. NaN means "not
                # assessed", and posetail >= 0.3.2 masks it out of the visibility BCE so those
                # entries produce no gradient instead of being trained as "not visible". Under
                # 0.3.0 a NaN here silently returned NaN gradients for every parameter while the
                # loss curve looked healthy, and this had to be collapsed to two states.
                # `projected` joins UNLABELED on the NaN side: it is a POSITION with no
                # visibility claim (johnson-mouse labels all 24 keypoints in all 16 views,
                # including ones the body hides), so training BCE on it would teach the head
                # "always visible" from 1.4M points that assert nothing.
                vis_2d = torch.as_tensor(np.where(np.isin(v2, (UNLABELED, PROJECTED)), np.nan,
                                                  (v2 == VISIBLE).astype(np.float32)))
                # 3D NOISY-OR: bool, and two-state by construction -- the loss inverts it with
                # `~` to build its occluded-point target (`losses.py:440`), which no float can
                # satisfy. That is the right semantics anyway: this layer answers "is the point
                # reconstructible in 3D", and where no camera assessed it there is no 3D label
                # either, so `False` is a fact rather than a guess.
                vis = torch.as_tensor((v2 == VISIBLE).any(-1))
                if not torch.isfinite(vis_2d).any():
                    # Nothing in this window carries a visibility ASSESSMENT -- every row is
                    # `projected`. There is no noisy-OR to take: all-False would assert that no
                    # point is reconstructible in 3D, which is a stronger lie than all-True.
                    # Drop to the 3dpop path and let the loss derive both masks geometrically.
                    vis = vis_2d = None
            else:
                # No per-camera assessment (3dpop, or a `tracked` session with no `missing` row
                # anywhere -- `sess.has_visibility_assessment`): let the loss derive both masks
                # geometrically. vis and vis_2d are both-or-neither -- one without the other
                # dies inside einops.
                vis = vis_2d = None

        if torch.isfinite(coords).all(-1).sum() < 2:
            return None

        # -- geometry: rotate, crop, resize ----------------------------------------------
        # One draw of the rate for the whole item. `aug_rotation_prob = None` reproduces the old
        # expression exactly -- same value, same number of draws, same order -- so a run that does
        # not set these two keys is bit-identical to every run recorded before they existed.
        rot_p = (self.cfg.aug_prob if self.cfg.aug_rotation_prob is None
                 else self.cfg.aug_rotation_prob)
        rot_deg = self.cfg.aug_rotation_deg
        rotation_info = [None] * len(cgroup)
        # (T, n_other, K, R) in the SAME final frame `coords` ends up in, or None -- built inside
        # whichever branch below runs, and consumed once, after `prompt_t` exists, in the query
        # prior section. None whenever `want_swap_animal` is False, which is every item on a
        # default config: nothing here changes what either branch does today.
        neighbour_full = None
        if true_2d:
            cam = cgroup[0]
            coords = _mask_outside(coords, cam['size'])
            if vis_2d is not None:
                # A point OUTSIDE the frame is not visible in the pixels the model is about to
                # see -- the same argument the 3D path makes after its own rotation, below. Flip
                # only entries that were a real POSITIVE (`== 1`): an entry already 0 (assessed
                # occluded) or NaN (unassessed) needs no change, and `NaN == 1` is False so this
                # is safe without an extra isnan check.
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            cp = None if crop_pts is None else crop_pts[:, 0]
            if self.train and rng.random() < rot_p:
                cam, coords, rot = _rotate_2d(cam, coords,
                                              float(rng.uniform(-rot_deg, rot_deg)))
                rotation_info = [rot]
                cp = _apply_affine(cp, rot)
                # `_rotate_2d` keeps the WHOLE expanded canvas (its own docstring), so nothing
                # here can push a point outside `cam['size']` -- no vis_2d update from rotation.
            jit = self._jitter(rng)
            cam, box, coords = cropmod.crop_to_points_2d(cam, coords, self.cfg.min_crop_dim,
                                                         jit, crop_pts=cp,
                                                         inflate=inflate)
            if cam is None:
                return None
            cam, scale = _resize_camera(cam, self.cfg.image_size)
            coords = coords * scale
            coords = _mask_outside(coords, cam['size'])
            if vis_2d is not None:
                # THE MEANINGFUL UPDATE: a point outside the final crop is not visible in the
                # crop the model trains on, even though it was visible in the source frame.
                vis_2d[:, :, 0][~torch.isfinite(coords).all(-1) & (vis_2d[:, :, 0] == 1)] = 0
            if want_swap_animal:
                # THE SAME THREE STEPS `coords` JUST WENT THROUGH -- the in-plane rotation (if
                # any), the crop's own shift, the resize's own scale -- reapplied to every OTHER
                # animal's raw pose with the existing `_apply_affine` helper rather than a second
                # copy of the arithmetic. Deliberately NOT `_mask_outside`'d against the ORIGINAL
                # frame: the only masking this needs is the FINAL one, `prior_out_of_bounds`,
                # applied once against the finished crop in the query-prior section below.
                cand_ids = [i for i in range(lab.n_animals) if i != a]
                raw = torch.as_tensor(lab.points2d[cand_ids][:, frames][..., 0, :],
                                      dtype=torch.float32)          # (n_other,T,K,2)
                raw = raw.permute(1, 0, 2, 3)                        # (T,n_other,K,2)
                raw = _apply_affine(raw, rotation_info[0])
                neighbour_full = (raw - box[:2].to(raw.dtype)) * scale
            cgroup, boxes = [cam], [box]
            p2d = p2d_all = coords[None]
            R = 2
        else:
            # 3D path, one camera or several. Single-view differs ONLY in how many cameras are
            # shown -- the targets stay world-metric, which is the whole point.
            if self.train:
                rotated = []
                for cam in cgroup:
                    if rng.random() < rot_p:
                        cam_r, rot = rotate_camera_image_plane_3d(
                            cam, float(rng.uniform(-rot_deg, rot_deg)))
                        # THE INSCRIBED CROP CAN TAKE THE ANIMAL WITH IT. The library helper crops
                        # to the border-free rectangle -- ~0.416 of the frame -- which is the exact
                        # step `_rotate_2d` was hand-written to avoid. An animal outside it
                        # projects out of `cam['size']`, and `crop_box_for_points` CLAMPS rather
                        # than refusing, so the item came back as a `min_crop_dim` corner crop of
                        # BACKGROUND carrying full-strength world targets. The `< 2 finite` guard
                        # at the top of `_item` runs before this, so nothing caught it.
                        #
                        # Reverted rather than retried: a retry re-draws `idx` and perturbs the
                        # whole subsequent sampling stream, where reverting consumes the identical
                        # draws and changes only the item at hand.
                        #
                        # The `2 <=` half is load-bearing. A camera that never saw the animal --
                        # ordinary on a 16-camera rig -- keeps today's behaviour instead of having
                        # its augmentation suppressed for a reason that is nothing to do with the
                        # rotation.
                        if int(is_point_visible(cam_r, coords).sum()) < 2 <= int(
                                is_point_visible(cam, coords).sum()):
                            cam_r, rot = cam, None
                        rotated.append((cam_r, rot))
                    else:
                        rotated.append((cam, None))
                cgroup = [c for c, _ in rotated]
                rotation_info = [r for _, r in rotated]
                if vis_2d is not None:
                    for cnum, cam in enumerate(cgroup):
                        if rotation_info[cnum] is not None:
                            vis_2d[:, :, cnum][~is_point_visible(cam, coords)] = 0
            # OTHER ANIMALS' RAW WORLD POSES, threaded through the SAME two random world rotations
            # `coords` is about to go through -- and NOTHING else: the per-camera in-plane
            # rotation above only moves cameras, and `crop_to_points_3d` below is never handed
            # this tensor, so it cannot widen the crop or change `cube_scale`
            # (dev/plans/prompt_prior_corruptions.md Section 4.4). None when `want_swap_animal` is
            # False, which is the default -- see `_rotate_camera_group_with_neighbours`.
            others_raw = None
            if want_swap_animal:
                cand_ids = [i for i in range(lab.n_animals) if i != a]
                others_raw = torch.as_tensor(lab.points3d[cand_ids][:, frames],
                                             dtype=torch.float32).permute(1, 0, 2, 3)
            if self.train:
                # Random world rotation, applied to points AND cameras together, so the model
                # cannot learn a fixed world gauge. `_rotate_camera_group_with_neighbours` is
                # exactly this call when `others_raw` is None (every item on a default config).
                cgroup, coords, others_raw = _rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            # A stored box lives in SOURCE pixels, so it follows each camera's own in-plane
            # rotation -- the same affine `rotate_camera_image_plane_3d` hands back for the warp,
            # or one per frame under a slide.
            cp3 = None if crop_pts is None else [
                _apply_affine(crop_pts[:, i], rotation_info[i]) for i in range(len(cgroup))]
            jit = self._jitter(rng)
            cgroup, boxes = cropmod.crop_to_points_3d(cgroup, coords, self.cfg.min_crop_dim,
                                                      jit, crop_pts=cp3,
                                                      inflate=inflate)
            if cgroup is None:
                return None
            cgroup = [_resize_camera(c, self.cfg.image_size)[0] for c in cgroup]
            if self.train:
                # Random world rotation, applied to points AND cameras together, so the model
                # cannot learn a fixed world gauge. Second draw of the item (see the first call
                # above); `others_raw` rides through this one too, so it ends up in EXACTLY the
                # frame `coords` itself ends up in.
                cgroup, coords, others_raw = _rotate_camera_group_with_neighbours(
                    cgroup, coords, others_raw)
            if others_raw is not None:
                neighbour_full = others_raw
            # `p2d` is the reprojection AFTER crop/resize/rotate, so it lands in crop pixels.
            # Cutout needs it too, in the same frame -- projected once and shared, since a second
            # `project_points_torch` is a float64 reprojection of every point in the window.
            p2d_all = (project_points_torch(cgroup, coords)
                       if single_view or self._aug is not None else None)
            p2d = p2d_all if single_view else None
            R = 3

        # -- pixels ----------------------------------------------------------------------
        # Appearance augmentation runs HERE, on the final crops, not on the source frames: the
        # crops are ~256 px where a source frame can be 4696x2048, so it is the same augmentation
        # for a fraction of the work.
        gray = self._aug is not None and rng.random() < self.cfg.grayscale_prob
        # `read_frames` IGNORES `pool` outright on the video path -- `_read_video` decodes every
        # frame of a camera in one `get_batch()` call, no per-frame `pool.submit()` at all -- so
        # a video-backed group spawned and joined 16 OS threads that did NOTHING, every item. Pure
        # overhead removed on that basis alone; tried as a candidate explanation for calms21's
        # per-worker RSS plateau under spawn and it made no measurable difference on a real job
        # (dev/plans/video_loader_memory.md §2.3) -- worker thread count was unchanged with or
        # without it. A group's cameras are one kind or the other by the format spec, never
        # mixed, so checking the first is checking all of them.
        use_pool = group.source(cam_names[0])[0] != 'video'
        with (ThreadPoolExecutor(max_workers=16) if use_pool else nullcontext()) as pool:
            views = []
            for cnum, cam_name in enumerate(cam_names):
                imgs = read_frames(group, cam_name, frames, crop_coords=boxes[cnum],
                                   target_size=cgroup[cnum]['size'].tolist(),
                                   rotation=rotation_info[cnum], pool=pool)
                if any(im is None for im in imgs):
                    return None
                if self._aug is not None:
                    imgs = self._augment(imgs, cnum, cgroup[cnum]['size'], p2d_all, vis_2d,
                                         gray, rng)
                # UINT8, not float32/255. The model divides on device (`model.py`), where it is
                # free, and this is 4x fewer bytes to collate, queue and pin -- 33 MB instead of
                # 132 MB for a 7-camera window, so 12 workers x prefetch 2 hold ~0.8 GB of pinned
                # host memory rather than ~3.2 GB.
                views.append(torch.from_numpy(np.asarray(imgs)))

        # -- the query prior -------------------------------------------------------------
        # kpt_prior is the pose at the prompt frame: at training time the GT, at deployment the
        # previous window's own prediction. prompt_t is the first frame each keypoint is
        # labelled at, which is NOT always 0 -- it was frame > 0 on 19.5% of rat-city windows.
        finite = torch.isfinite(coords).all(-1)                    # (T,K)
        prompt_t = torch.where(finite.any(0), finite.float().argmax(0), torch.zeros(K).long())
        prompt_t = prompt_t.to(torch.int32)
        kpt_prior = coords[prompt_t, torch.arange(K)].clone()      # (K,R)
        kpt_prior[~finite.any(0)] = float('nan')
        # PER ITEM, NOT PER KEYPOINT. The reference draws one coin for the whole window
        # (`posetail_pose/model.py:619-621`, shape (B,1,1) broadcast over K), which is what makes
        # `prompt_dropout` the fraction of TRAINING STEPS that run fully query-free -- so one set
        # of weights serves both the first window of a clip and every window after it. Drawing
        # i.i.d. per keypoint instead put P(a fully unprompted window) at 0.4^47 ~ 1e-19: the
        # query-free forward, which is the path val and `best_mpjpe` score, was never trained.
        # It also left the prompted windows partially dense where `--anchor carry` supplies a
        # 100%-dense prior, so training and deployment disagreed on prompt density too.
        if self.train and rng.random() < self.cfg.prompt_dropout:
            kpt_prior[:] = float('nan')
        # EXPOSURE BIAS. The prior trains on exact GT and deploys as the model's own prediction;
        # an arm trained only on exact priors learns to trust a precision the carried signal does
        # not have. On rat-city, over the same 138 instance-windows, the (0, 0) control had carry
        # HURT by +1.815 mm while the (0.4, 5.7) recipe had it HELP by -0.749. NaN + noise is
        # still NaN, so a withheld prior stays withheld with no mask.
        #
        # SIGMA IS IN PIXELS, AND 3D CONVERTS. One scalar in the SESSION's own units cannot work:
        # `allen-mouse-combined` alone holds 63 px sessions beside 14 mm ones, so a single 1.0
        # meant a 1 px nudge on one and a 1 mm nudge on the other. `cube_scale` is world units per
        # pixel -- the same conversion `WeightedMAELoss` uses to enter the Huber in pixels
        # (`losses.py:1070-1080`) -- so scaling by it makes ONE setting mean the same visual
        # displacement on a 30 px fly and a 57,594-frame rat rig alike. Measured at the prior's own
        # position, through THIS window's cropped and resized cameras, so crop jitter is included.
        #
        # I.I.D. NOISE IS NOT THE FAILURE DEPLOYMENT PRODUCES, which is why the two extra
        # corruptions below exist. `--anchor carry` hands back the model's own previous output, and
        # its error is a WHOLE-BODY LAG -- every keypoint displaced the same way, growing window over
        # window -- or a pose describing a different frame than the model is told. Gaussian jitter
        # averages to zero over the keypoint set and so trains the model to trust the CENTROID of
        # the prior exactly, which is precisely the quantity a lag gets wrong. Measured with
        # `scripts/infer.py --oracle-corrupt`, which applies the same three failures at inference.
        #
        # ONE VECTOR FOR THE WHOLE POSE, drawn per item, in the same pixel units as the jitter.
        px = 1.0
        if R == 3 and (self.cfg.prompt_noise_px > 0 or self.cfg.prompt_offset_px > 0) \
                and bool(torch.isfinite(kpt_prior).any()):
            pts = kpt_prior[torch.isfinite(kpt_prior).all(-1)][None]     # (1,n,3)
            # A PROJECTION JACOBIAN IS OFFSET-INVARIANT -- a constant image-plane translation
            # has zero derivative -- and `pts` is one mean pose with no time axis. posetail
            # 0.3.5 collapses a per-frame (T,2) offset itself; there is none in this repo anyway.
            px = float(torch.nanmedian(get_camera_scale(cgroup, pts)))
            if not np.isfinite(px):
                px = 1.0
        # STALE: the pose from a DIFFERENT frame than `prompt_t` claims. What `carried` degrades into
        # when the box source loses an animal for a window or two; the resulting error is the
        # animal's own motion over that gap, correlated across keypoints exactly as a lag is.
        if self.train and self.cfg.prompt_stale_frames > 0 \
                and bool(torch.isfinite(kpt_prior).any()):
            d = int(rng.integers(0, int(self.cfg.prompt_stale_frames) + 1))
            if d:
                t_alt = torch.clamp(prompt_t.long() + d, max=coords.shape[0] - 1)
                alt = coords[t_alt, torch.arange(K)]
                # Only where the OTHER frame has a point too: a stale prior is a wrong position, not
                # a withdrawn one, and swapping in NaN would be `prompt_dropout` under another name.
                swap = torch.isfinite(alt).all(-1) & torch.isfinite(kpt_prior).all(-1)
                kpt_prior = torch.where(swap[:, None], alt, kpt_prior)
        # B: THE PRIOR JUMPS TO A NEARBY ANIMAL (`prompt_swap_animal`, dev/plans/
        # prompt_prior_corruptions.md). `neighbour_full` is None unless `want_swap_animal` drew
        # true at the top of `_item` -- so a default config, and every single-animal session,
        # never reaches this block at all.
        if neighbour_full is not None:
            mode_str = '2d' if R == 2 else '3d'
            # (T, n_other, K, R) -> this item's own PROMPT_T per keypoint, exactly how
            # `kpt_prior` itself was built above -- reused rather than re-derived.
            cand = neighbour_full.permute(0, 2, 1, 3)[prompt_t, torch.arange(K)]  # (K,n_other,R)
            oob = torch.stack([prior_out_of_bounds(cand[:, i], mode_str, cgroup)
                               for i in range(cand.shape[1])], dim=1)              # (K,n_other)
            eligible = (~oob).sum(0) >= 2
            if bool(eligible.any()):
                # UNIFORM among eligible neighbours: the nearest animal is already the modal
                # eligible one, so a nearest rule would add tie-breaking machinery for no
                # measured benefit (dev/plans/prompt_prior_corruptions.md Section 4.3).
                elig_idx = eligible.nonzero(as_tuple=True)[0]
                choice = int(elig_idx[int(rng.integers(len(elig_idx)))])
                neighbour_prior = cand[:, choice].clone()
                # THE BOUNDS MASK, APPLIED ONLY HERE. The neighbour survives as a WHOLE (>= 2
                # keypoints eligible), but an individual keypoint may still be out of bounds, and
                # that one becomes NaN exactly as a departed `carry` keypoint does.
                neighbour_prior[oob[:, choice]] = float('nan')
                frac = float(self.cfg.prompt_swap_animal_frac)
                # 1.0 DRAWS NOTHING -- the same inert-at-1.0 contract `crop_inflate` already has.
                jump = (torch.isfinite(neighbour_prior).all(-1) if frac >= 1.0 else
                       (torch.as_tensor(rng.random(K)) < frac)
                       & torch.isfinite(neighbour_prior).all(-1))
                kpt_prior = torch.where(jump[:, None], neighbour_prior, kpt_prior)
            # An INELIGIBLE draw (every other animal loses -- fewer than 2 keypoints survive the
            # bounds mask) is a NO-OP: `prompt_swap_animal`'s CONFIGURED rate and its PRESENTED
            # rate then differ, which is why an arm is measured with a loader audit rather than
            # assumed from the config.
        # A: THE PRIOR JUMPS TO OTHER KEYPOINTS OF THE SAME ANIMAL (`prompt_swap_kpt_pairs`). A
        # TRANSPOSITION of `kpt_prior`'s own finite entries -- the prior SET is therefore
        # unchanged and `px` above stays valid, unlike `prompt_dropout`'s per-item draw. `prompt_t`
        # is deliberately left untouched, the same precedent `prompt_stale_frames` sets just
        # above: staleness moves the POSE, never the CLAIMED frame.
        if self.train and self.cfg.prompt_swap_kpt_pairs > 0:
            finite_idx = torch.isfinite(kpt_prior).all(-1).nonzero(as_tuple=True)[0]
            if len(finite_idx) >= 2:
                x = float(self.cfg.prompt_swap_kpt_pairs)
                n_pairs = int(x) + int(rng.random() < (x - int(x)))
                n_pairs = min(n_pairs, len(finite_idx) // 2)
                if n_pairs:
                    perm = finite_idx[torch.from_numpy(rng.permutation(len(finite_idx)))]
                    kpt_prior = kpt_prior.clone()
                    for i in range(n_pairs):
                        j, k = int(perm[2 * i]), int(perm[2 * i + 1])
                        kpt_prior[j], kpt_prior[k] = kpt_prior[k].clone(), kpt_prior[j].clone()
        if self.train and self.cfg.prompt_offset_px > 0 and bool(torch.isfinite(kpt_prior).any()):
            kpt_prior += torch.as_tensor(
                rng.normal(0.0, float(self.cfg.prompt_offset_px) * px, (1, R)),
                dtype=kpt_prior.dtype)
        if self.train and self.cfg.prompt_noise_px > 0 and bool(torch.isfinite(kpt_prior).any()):
            kpt_prior += torch.as_tensor(
                rng.normal(0.0, float(self.cfg.prompt_noise_px) * px, kpt_prior.shape),
                dtype=kpt_prior.dtype)

        kpt_ids = self._kpt_ids[sess.path]      # aligned to THIS session's axis, not the root's
        query_times = torch.zeros(K, dtype=torch.int32)
        query_occlusion = torch.full((K, len(cgroup)), -1, dtype=torch.int64)
        # The stride is read back off `frames` rather than threaded out of `_frames`: `SmoothnessLoss`
        # has no notion of dt and its k-th difference grows like s^k, so `run_batch` has to undo
        # that or the term's effective weight would depend on a draw. MEDIAN, not `frames[1] -
        # frames[0]`: a window clipped at a group edge (`_frames`, np.clip) repeats its last frame,
        # and those zero gaps must not be read as stride 1.
        stride = max(1, int(np.median(np.diff(frames)))) if len(frames) > 1 else 1

        # THE 3D NOISY-OR IS TAKEN OVER THE *FINAL* vis_2d, and it used not to be. It was computed
        # from the raw statuses before any augmentation, while the rotation loop zeroes
        # `vis_2d[:, :, cnum]` for points the rotated camera no longer sees and `_cutout_rects`
        # zeroes more -- so a point could end up `vis_2d == 0` in EVERY shown camera while `vis`
        # still claimed True. `TotalLoss` takes a supplied `vis_true` verbatim (losses.py:421) and
        # weights `mae_loss_coords`, `coords_loss_direct` and `bce_loss_vis_3d` by it, so the 3D
        # coord and visibility heads were supervised at points the crop provably does not contain.
        # With `vis=None` the loss derives this geometrically and masks correctly -- i.e. supplying
        # visibility was WORSE than withholding it, which is the opposite of the intent.
        #
        # `== 1` is NaN-safe (NaN == 1 is False), and the result stays bool, which the loss
        # requires: it inverts this with `~` to build the occluded-point target.
        if vis is not None and vis_2d is not None:
            vis = (vis_2d == 1).any(-1)

        row = {'dataset': self.datasets[item.ds].name, 'session': sess.session_id,
               'group': item.gid, 'animal': lab.animal_ids[a], 'mode': '2d' if R == 2 else '3d',
               'single_view': single_view, 'start': int(frames[0]), 'cameras': cam_names,
               'stride': stride}

        out = [views, coords, vis, torch.as_tensor(frames), cgroup, row, query_times,
               vis_2d, p2d, query_occlusion, kpt_ids, kpt_prior, prompt_t]
        # THE BOX PROMPT (report 27), emitted only when a box model is training -- so a plain run
        # keeps its 13-field item and is byte-identical. The box is the target's extent in THIS
        # window's crop frame (`box_prompt.compute_box_prompt`), NOT a second copy of the crop
        # geometry -- it reads the coords the crop already produced.
        if self.cfg.box_prompt != 'none':
            from . import box_prompt as bpmod
            box = bpmod.compute_box_prompt(coords, cgroup, '2d' if R == 2 else '3d')
            box = bpmod.apply_frames_mode(box, self.cfg.box_prompt_frames)
            if self.train:
                if self.cfg.box_prompt_dropout > 0 and rng.random() < self.cfg.box_prompt_dropout:
                    box = torch.full_like(box, float('nan'))
                elif self.cfg.box_prompt_jitter > 0 or self.cfg.box_prompt_scale_jitter > 0:
                    box = bpmod.apply_jitter(box, rng, self.cfg.box_prompt_jitter,
                                             self.cfg.box_prompt_scale_jitter)
            out.append(box)
        return tuple(out)

    def _augment(self, imgs, cnum, size, p2d, vis_2d, gray, rng):
        """Appearance augmentation for one camera's T crops. `vis_2d` is mutated by cutout."""
        import cv2

        per_camera, per_image = self._aug
        # to_deterministic() freezes this camera's sampled parameters so every frame gets the
        # SAME gamma/hue -- see `_build_augmenters`. Only the apply is per-frame.
        cam_det = per_camera.to_deterministic()
        imgs = [per_image(image=cam_det(image=im)) for im in imgs]
        if rng.random() < self.cfg.aug_prob:
            for x1, y1, x2, y2, fill in _cutout_rects(rng, size, p2d, vis_2d, cnum):
                for im in imgs:
                    im[y1:y2, x1:x2] = fill
        if gray:
            imgs = [np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)] * 3, -1) for im in imgs]
        return imgs

    def _jitter(self, rng):
        if not self.train or self.cfg.crop_jitter <= 0:
            return None
        return cropmod.jitter_box(rng, self.cfg.crop_jitter, self.cfg.crop_jitter_scale)


def _mask_outside(coords, size):
    """A point outside the image is not a label. Drops it rather than clamping it to the edge."""
    w, h = float(size[0]), float(size[1])
    bad = ((coords[..., 0] < 0) | (coords[..., 0] >= w) |
           (coords[..., 1] < 0) | (coords[..., 1] >= h))
    coords = coords.clone()
    coords[bad] = float('nan')
    return coords


def _resize_camera(cam, target_res):
    """Scale a camera so its long side is `target_res`. Returns (cam, scale)."""
    cam = dict(cam)
    size = cam['size']
    scale = float(target_res) / float(max(size))
    # A ZERO SIDE IS NOT A SMALL CROP, IT IS THE WHOLE FRAME. An extreme-aspect box -- one
    # detector box clipped to a sliver against a frame edge, which `run_group`'s per-camera union
    # allows (it only guarantees x1 >= x0+1) -- rounds to e.g. [256, 0], and `cv2.warpAffine`
    # treats a 0 in `dsize` as "use the source size": it returns the FULL 4696x2048 frame, with no
    # exception, while the camera dict claims [256, 0]. Silent garbage rather than a clean failure.
    cam['size'] = torch.round(size * scale).to(torch.int32).clamp_min(1)
    cam['mat'] = cam['mat'] * scale
    cam['mat'][2, 2] = 1
    cam['offset'] = cam['offset'] * scale
    return cam, scale


class StepSampler(torch.utils.data.Sampler):
    """Draws indices with replacement and yields `(ordinal, index)`.

    The ordinal is the item's position in the stream, which is the global training step -- and it
    is the SAME on every rank while the index is not. `PoseDataset.__getitem__` keys its
    cost-determining draws on it so that every rank's step-k item costs the same, which is what a
    synchronous DDP step needs (it costs the slowest rank's item; see `__getitem__`).

    Replacement, and no `DistributedSampler`: an index entry here is one (session, group, animal)
    whatever the group's length, so it is a poor sampling weight -- `_pick` re-draws inside
    `__getitem__` and partitioning the index would partition the wrong thing. Independent streams
    per rank is the right construction, and it is what the single-gpu loop already did.
    """

    def __init__(self, n: int, num_samples: int, generator=None, start: int = 0):
        self.n, self.num_samples, self.generator, self.start = int(n), int(num_samples), generator, int(start)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        idx = torch.randint(0, self.n, (self.num_samples,), generator=self.generator)
        for k, i in enumerate(idx.tolist()):
            yield (self.start + k, i)


def worker_init(worker_id):
    """A DataLoader `worker_init_fn`. Four things, all per-worker and all easy to miss.

    1. **Pin cv2's thread pool to one thread.** OpenCV sizes it to the machine, and each worker
       already runs its own decode pool on top, so the nesting is pure contention.

    2. **Pin torch's intraop pool to one thread too.** A worker does small CPU-only work and never
       calls an ATen op that would use it. Hygiene, not a measured win.

    3. **Reseed imgaug.** It keeps its OWN global RNG, which fork copies and nothing else reseeds,
       so every worker's k-th `to_deterministic()` would draw the same gamma, hue and blur --
       appearance diversity silently divided by `num_workers`, invisible in the loss curve.

    4. **Force imgaug's hue/saturation LUT cache to build in THIS process.**
       `AddToHueAndSaturation.__init__` populates a CLASS attribute as a side effect. Under
       `spawn` the dataset is unpickled without `__init__` re-running, so the attribute is still
       None and the first hue augmentation crashes. A throwaway instance re-triggers it.

    numpy is deliberately NOT reseeded: torch already decorrelates it per worker.
    `tests/test_dataset.py::test_workers_do_not_share_a_random_stream` pins both halves.
    """
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        import imgaug
        import imgaug.augmenters as iaa
    except ImportError:
        return
    imgaug.random.seed(int(np.random.randint(2 ** 31)))
    iaa.AddToHue()


def pose_collate(batch):
    """posetail's collate for the first ten fields, plus this repo's three.

    `custom_collate` keeps only item 0's `cgroup` and asserts a batch does not mix 2D and 3D --
    which is why batch_size is structurally 1 PER RANK, and why the only batch dimension this repo
    has is the world (`--devices N`, `tailcyclenet.distributed`). The same reason covers K:
    sessions may carry different keypoint subsets, so a batch that mixed two of them would fail the
    `kpt_ids` stack below. Loud, and unreachable at batch_size 1.
    """
    batch = [b for b in batch if b is not None]
    out = custom_collate([b[:10] for b in batch])
    out['kpt_ids'] = torch.stack([b[10] for b in batch])
    out['kpt_prior'] = torch.stack([b[11] for b in batch])
    out['prompt_t'] = torch.stack([b[12] for b in batch])
    # THE BOX PROMPT (report 27), present only when a box model is training (`_item` appends a
    # 14th field then). Absent for a plain run, so nothing downstream sees it.
    if len(batch[0]) > 13:
        out['box_prompt'] = torch.stack([b[13] for b in batch])
    return out
