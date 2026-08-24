import json
import re
import tomllib
from pathlib import Path

import torch

from .assign import (assign, assign_tal, box_iou, certified_anchors, ciou_loss, decode,
                     detector_loss, giou_loss, paired_iou, quality_focal_loss)
from .associate import associate
from .data import (BoxDataset, ChunkShuffle, CohortSampler, TEMPORAL_INPUT_BY_CHANNELS,
                   TEMPORAL_INPUT_CHANNELS, TEMPORAL_INPUTS, box_collate, letterbox,
                   letterbox_transform, reduce_factor, split_batch, tile_transform,
                   unletterbox_boxes, unletterbox_keypoints)
from .pretrained import load_coco_backbone, load_pretrained_backbone
from .yolox import YOLOX_TIERS, YOLOXNano


def resolve_detector_checkpoint(path, checkpoint='latest'):
    """Resolve a detector file from a run directory.

    Directory deployment defaults to the highest complete ``detector_it*.pth`` checkpoint, not
    historical ``detector.pth`` (which is the validation-selected *best* checkpoint). ``best``
    remains an explicit compatibility selector; an explicit filename is also an override and is
    therefore not subjected to the latest-completeness check.
    """
    p = Path(path)
    if not p.is_dir():
        return p
    selector = str(checkpoint or 'latest')
    if selector == 'best':
        out = p / 'detector.pth'
        if not out.exists():
            raise ValueError(f'{p}: --detector-checkpoint best requested, but detector.pth is absent')
        return out
    if selector != 'latest':
        out = p / selector
        if not out.exists():
            raise ValueError(f'{p}: explicit detector checkpoint {selector!r} does not exist')
        return out

    files = []
    for candidate in p.glob('detector_it*.pth'):
        match = re.fullmatch(r'detector_it(\d+)\.pth', candidate.name)
        if match:
            files.append((int(match.group(1)), candidate))
    if not files:
        raise ValueError(
            f'{p}: latest detector checkpoint requested, but no detector_it*.pth files exist; '
            'pass --detector-checkpoint best or an explicit filename for a legacy run')
    files.sort(reverse=True)
    expected = None
    config_path = p / 'config.toml'
    if config_path.exists():
        with config_path.open('rb') as f:
            expected = tomllib.load(f).get('training', {}).get('iters')
        expected = None if expected is None else int(expected)
    metrics_last = None
    metrics_path = p / 'metrics.json'
    if metrics_path.exists():
        with metrics_path.open() as f:
            history = json.load(f)
        if history:
            metrics_last = int(history[-1]['iteration'])

    for iteration, candidate in files:
        ckpt = torch.load(candidate, map_location='cpu', weights_only=False)
        saved_iteration = ckpt.get('iteration')
        if saved_iteration is None or int(saved_iteration) != iteration:
            continue
        if expected is not None and iteration != expected:
            continue
        if metrics_last is not None and iteration != metrics_last:
            continue
        if expected is None and metrics_last is None:
            raise ValueError(
                f'{p}: cannot establish that {candidate.name} is complete because neither '
                'config.toml [training].iters nor metrics.json is present; pass '
                '--detector-checkpoint best or an explicit filename')
        return candidate

    details = []
    if expected is not None:
        details.append(f'config iters={expected}')
    if metrics_last is not None:
        details.append(f'metrics last iteration={metrics_last}')
    state = ', '.join(details) if details else 'missing completion metadata'
    raise ValueError(
        f'{p}: no complete latest detector checkpoint found ({state}); pass '
        '--detector-checkpoint best or an explicit filename to override')


__all__ = ['YOLOXNano', 'YOLOX_TIERS', 'BoxDataset', 'ChunkShuffle', 'CohortSampler',
           'box_collate', 'letterbox',
           'letterbox_transform', 'reduce_factor', 'split_batch', 'tile_transform',
           'unletterbox_boxes', 'unletterbox_keypoints', 'assign', 'assign_tal', 'box_iou',
           'certified_anchors', 'ciou_loss', 'decode', 'detector_loss', 'giou_loss',
           'quality_focal_loss', 'associate', 'TEMPORAL_INPUT_CHANNELS',
           'TEMPORAL_INPUTS', 'TEMPORAL_INPUT_BY_CHANNELS',
           'detect_raw', 'associate_group', 'link_rows', 'load_coco_backbone',
           'load_pretrained_backbone', 'paired_iou', 'resolve_detector_checkpoint']

# The `--det-cache` version constants lived here; detection and the pose loop are one pass now,
# so the detections' dependencies are recorded in the prediction's own provenance instead.


def load_detector(path, device='cpu', input_wh=None, checkpoint='latest'):
    """(model, input_wh, dataset_name, min_crop_dim, reduce, box_source, tile_scale, obj_q).

    The input size, min_crop_dim, box_source and tile_scale are recorded in the checkpoint because
    each is part of the weights: the letterbox a detector was trained under decides what an animal
    looks like to it, and a mismatch serves boxes from a different crop rule silently. `input_wh`
    supplies the size for checkpoints that predate the field. `yolox_version`,
    `bottleneck_expansion`, `p2` and `in_channels` similarly record the architecture the weights
    were shaped for (absent = the pre-key default), used only to build the right model internally.
    For a run directory, `checkpoint='latest'` selects and verifies the highest complete
    `detector_it*.pth`; `checkpoint='best'` explicitly selects historical `detector.pth`.
    """
    import torch
    p = resolve_detector_checkpoint(path, checkpoint=checkpoint)
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    wh = input_wh or ckpt.get('input_wh') or ckpt.get('det_input_wh')
    if wh is None:
        raise ValueError(f'{p}: no input_wh in the checkpoint -- a posetail-pose detector keeps '
                         'it in its dataset config. Pass --det-input-wh W H (rat-city 896 384, '
                         'branson-fly 416 416).')
    # A2.5: a ViT backbone's own DINOv2 patch embedding needs both dims divisible by 14, and the
    # coarsest FPN stride needs 32 -- LCM 224. `train_detector.py` already rounds to this at
    # training time, so this only bites an explicit `--det-input-wh` override (or a checkpoint
    # that predates the `input_wh` field).
    if str(ckpt.get('yolox_version', 'trimmed')).startswith('vit_'):
        wh = tuple(max(64, int(-(-v // 224)) * 224) for v in wh)
    # Absent `norm` means BatchNorm: the key only exists since the model became GroupNorm.
    norm = str(ckpt.get('norm', 'bn'))
    if norm != 'gn':
        raise ValueError(
            f'{p}: trained with {norm} normalisation; the model is GroupNorm now (there are no '
            'running statistics to load into). Retrain this detector -- see '
            '`tailcyclenet/detector/yolox.py:conv_norm_act` for why the switch was made.')
    model = YOLOXNano(n_keypoints=int(ckpt.get('n_keypoints', 0)),
                      version=str(ckpt.get('yolox_version', 'trimmed')),
                      bottleneck_expansion=float(ckpt.get('bottleneck_expansion', 0.5)),
                      p2=bool(ckpt.get('p2', False)),
                      in_channels=int(ckpt.get('in_channels', 3)),
                      head_depthwise=ckpt.get('head_depthwise', None),
                      pretrained=str(ckpt.get('pretrained', '') or ''),
                      shared_head=bool(ckpt.get('shared_head', True)),
                      fpn_upsample=str(ckpt.get('fpn_upsample', 'nearest') or 'nearest'),
                      p2_bottomup=bool(ckpt.get('p2_bottomup', False)))
    model.load_state_dict(ckpt['model_state'])
    ts = ckpt.get('tile_scale')
    if ckpt.get('tile_wh') is not None and ts is None:
        raise ValueError(
            f'{p}: trained on tiles ({ckpt["tile_wh"]}) but carries no `tile_scale`, so the '
            'deployment input size cannot be derived. `input_wh` here is the TILE size, not the '
            'whole-frame size -- running the frame at it is a scale shift, not a smaller input.')
    # `tile_scale` is meaningless without `tile_wh`, so drop it (untiled runs record 1.0).
    return (model.to(device).eval(), tuple(wh), str(ckpt.get('dataset', '')),
            int(ckpt.get('min_crop_dim', 64)), bool(ckpt.get('reduce', False)),
            str(ckpt.get('box_source', 'keypoints')),
            None if ts is None or ckpt.get('tile_wh') is None else float(ts),
            # The objectness distribution this checkpoint produces; `--det-score` is not portable
            # across detector generations, so callers warn rather than guess a threshold.
            dict(ckpt.get('obj_quantiles') or {}))


def tiled_input_wh(src_wh, tile_scale):
    """Whole-frame input size a tile-trained detector must be deployed at.

    The invariant is the animal's size in INPUT pixels (a convnet is not scale-invariant), so a
    tile-trained detector must see the whole frame at native scale. Rounded to a multiple of 32,
    the coarsest stride, as `train_detector.input_wh_for` rounds.
    """
    return tuple(max(64, int(round(float(v) * float(tile_scale) / 32) * 32)) for v in src_wh)


@torch.no_grad()
def detect_raw(det, input_wh, session, gid, top_k, device='cpu', batch=16, score_thresh=0.01,
               reduce=False, max_frames=0, tile_scale=None, frames=None, read=None,
               iou_thresh=0.5, center_dist_thresh=0.5, trace=None, trace_detail=False):
    """The DETECTION half: pixels -> per-camera detections, ranked by score, unassociated.

    -> (boxes (D,T,C,4), scores (D,T,C), kpts (D,T,C,K,3) or None) with `D = top_k`, where index
    `d` is the d-th highest-scoring detection IN THAT CAMERA AT THAT FRAME and means nothing across
    cameras or across frames. Rows become an animal axis in `associate_group`, not here.

    The split exists so every association arm shares one detection pass: detection is the
    decode-bound expensive half of a run, and identity levers change only what happens after it.

    `score_thresh` defaults to 0.01 -- NOT the 0.99 an older, saturated-near-1.0 objectness
    distribution (hard-1.0 target, pre-`iou_aware_obj`) would have tolerated, and lower than
    `decode`/`score_dataset`'s own 0.05 "as-trained" convention. Measured directly against the
    current default recipe (`iou_aware_obj=true`, COCO-pretrained): rat-city-combined's MOTA
    peaks AT 0.01 (0.795 vs 0.640 at 0.05 vs 0.747 at 0.0 -- 0.0 buys no MOTA over 0.01, just more
    false positives); allen-mouse-combined and 3dpop are BYTE-IDENTICAL between 0.01 and 0.05 (no
    detections score in that band). Deliberate choice: a false negative costs more than the extra
    false positives a looser floor invites, and those extra candidates still pass through NMS
    (IoU + centre-distance) and, in 3D multiview, `associate`/`CrossViewTracker`'s own
    reprojection-residual gate (`assoc_res_max_px`, default 30px) before they can become a kept
    detection -- this is not an unfiltered flood, it is more candidates for filters that already
    exist. Still sweep per checkpoint -- this is a measured DEFAULT, not a universal constant.

    `iou_thresh` / `center_dist_thresh` are `decode`'s own NMS knobs, threaded through so a caller
    (a CLI flag, a config key) can move them -- `decode`'s Python defaults are now 0.5 / 0.5
    (detector_v2 plan A1's `iou_thresh` unchanged; A5's `center_dist_thresh` CONFIRMED and made
    the default, a deliberate break from every checkpoint trained before this landed). Pass
    `center_dist_thresh=None` explicitly to restore the pre-A5 byte-identical behaviour.

    `max_frames` is the same PREFIX `infer.run_group` takes, so the two agree about the clip.

    `frames` detects a SLICE of the clip, so detection advances alongside the window loop. The
    returned arrays are `len(frames)` long and index `t` is a position in `frames`, not a source
    frame number. `None` is `range(T)`.

    A slice must START ON A GLOBAL `batch` BOUNDARY: `_units` partitions on `range(0, T, batch)`, so
    a misaligned slice forwards a short leading batch -- a different input shape, and cuDNN selects
    convolution algorithms per shape. Aligned slices are byte-identical to one whole-clip pass.

    `read` REPLACES THE DECODE with `(ci, cam_name, frames, pool) -> imgs`, so a caller that
    already holds these frames does not decode them a second time. `None` is `read_frames`.

    `trace`, when a list is supplied, receives one compact decode-stage record per
    (source-frame, camera): score survivors, post-NMS survivors, and final top-k survivors. With
    `trace_detail=True`, the record also carries source-pixel candidate boxes and scores for all,
    post-score, post-NMS, and final candidates, for an offline GT matcher. It is diagnostic-only;
    the default is `None`, and output arrays/return arity are unchanged.
    """
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from .. import memory as _memory
    from ..dataset import read_frames
    from .data import letterbox, reduce_factor, unletterbox_boxes, unletterbox_keypoints

    # `_fetch` below always builds a 3-channel `arr` -- one letterboxed RGB frame per (camera,
    # source frame). A `temporal_input='stack2'` checkpoint (`in_channels=6`) trains and
    # `load_detector` reconstructs the wider stem correctly (that half is real, see
    # `YOLOXNano.__init__`'s docstring), but this deployment loop has no paired-frame path to
    # fill the other 3 channels. Forwarding it here would silently feed zeros/garbage into half
    # the stem and report ordinary-looking boxes -- refuse instead of guessing.
    if int(getattr(det, 'in_channels', 3)) != 3:
        raise SystemExit(
            f'detect_raw: this checkpoint has in_channels={det.in_channels}, but detection '
            'always forwards one 3-channel frame at a time -- there is no paired-frame deployment '
            "path yet (see YOLOXNano's docstring, `NOT YET WIRED`). Retrain with the default "
            "[data].temporal_input='none', or wire a real deployment reader before using this "
            'checkpoint.')

    group = session.groups[gid]
    T_clip = min(group.n_frames, max_frames or group.n_frames)
    want = np.arange(T_clip) if frames is None else np.asarray(frames, np.int64)
    T = len(want)
    _read = read if read is not None else (
        lambda ci, cam_name, fr, pool=None, reduce=1: read_frames(group, cam_name, fr,
                                                                  reduce=reduce, pool=pool))
    C = len(session.rig)
    D = max(1, int(top_k))
    out = np.full((D, T, C, 4), np.nan, np.float32)
    sc = np.full((D, T, C), np.nan, np.float32)
    # (D,T,C,K,3) of (x, y, score_logit) in SOURCE pixels, or None when this detector has no
    # keypoint branch. Kept beside the boxes so a caller that indexes `[d, t, ci]` for a box
    # indexes the same way for its keypoints.
    K_det = int(getattr(det, 'n_keypoints', 0))
    kp = np.full((D, T, C, K_det, 3), np.nan, np.float32) if K_det else None

    # ONE THREAD PER CAMERA FOR THE DECODE, which is where this function's wall clock lives;
    # the containers share no state and decord releases the GIL, so they overlap ~3.5x. The
    # forward stays serial and in camera order -- it is ~1% of the time.
    def _fetch(job):
        ci, cam_name, src_frames = job
        # Same decode the detector was trained on: `BoxDataset` reduces at decode where the frame
        # is far above the letterbox target, and a detector fed differently-sampled pixels at
        # deployment is off its own training distribution.
        src = session.rig.size(cam_name)
        # Per camera: a tile-trained detector's input size is a function of the FRAME size.
        wh = input_wh if tile_scale is None else tiled_input_wh(src, tile_scale)
        r = reduce_factor(src, wh) if reduce else 1
        imgs = _read(ci, cam_name, src_frames, pool=frame_pool, reduce=r)
        # ONE numpy conversion for the whole batch, not one torch op per frame: a tiny elementwise
        # op through torch's intraop pool is measured ~60x slower than numpy's packed path.
        #
        # uint8 all the way to the device (the /255 happens there, as the pose loader does) --
        # 4x less host, PCIe and device memory than a host-side float32 copy. Bit-identical:
        # uint8 -> float32 is exact and the float32 divide by 255 is correctly rounded.
        n = len(imgs)
        metas, arr = [], None
        for i in range(n):
            lb, scale, pad = letterbox(imgs[i], wh, src_wh=src)
            if arr is None:
                arr = np.empty((n, 3, lb.shape[0], lb.shape[1]), np.uint8)
            arr[i] = lb.transpose(2, 0, 1)
            metas.append((scale, pad))
            imgs[i] = None                 # the decode is dead the moment it is letterboxed
        return ci, torch.from_numpy(arr), metas, src

    # WHAT IS BOUNDED IS THE NUMBER OF CAMERAS IN FLIGHT, AND NEVER `batch`. `batch` is not a
    # memory knob because it is not inert: cuDNN selects convolution algorithms per input shape, so
    # a different batch is a different reduction order -- two runs on two machines would produce
    # different boxes with nothing in either output saying so. Chunking CAMERAS is output-neutral
    # by construction: each camera is forwarded on its own and `out`/`sc`/`kp` are indexed by
    # `[.., ci]`; only how much of the decode overlaps changes.
    _per_frame_cam = 0
    for _cam in session.cam_names:
        _src = session.rig.size(_cam)
        _wh = input_wh if tile_scale is None else tiled_input_wh(_src, tile_scale)
        _per_frame_cam = max(_per_frame_cam,
                             int(_src[0]) * int(_src[1]) * 3 + int(_wh[0]) * int(_wh[1]) * 3)
    _budget = _memory.current().share(_memory.FRACTION_DETECT)
    cams_flight = _memory.fits(_budget, _per_frame_cam * max(1, int(batch)) * 2, want=max(1, C))

    # A 0-d TENSOR, NOT THE PYTHON INT 255: on CUDA `x / 255` takes a reciprocal-multiply fast path
    # that is off by 1 ULP on 156 of 256 byte values; dividing by a 0-d tensor is correctly rounded.
    # 1 ULP on the input perturbs every objectness score and can reorder NMS ties.
    _div255 = torch.tensor(255.0, device=device)

    pool = ThreadPoolExecutor(max_workers=cams_flight)
    # A SECOND POOL for image-directory roots: `read_frames` threads those over their FRAMES (it
    # ignores `pool` for video), so single-camera roots spend their time here. It must NOT be
    # `pool`: `_fetch` runs IN `pool` and waits on these futures, and a pool that waits on itself
    # deadlocks the moment both are full.
    frame_pool = ThreadPoolExecutor(max_workers=min(16, max(1, batch)))

    # A SLICE MUST LINE UP WITH THE WHOLE-CLIP BATCH PARTITION, or its forwards have shapes the
    # whole-clip pass never produces and cuDNN answers them differently (see the docstring).
    if frames is not None:
        _aligned = (int(want[0]) % batch == 0
                    and (T % batch == 0 or int(want[-1]) == T_clip - 1))
        assert _aligned and np.array_equal(want, np.arange(want[0], want[0] + T)), (
            f'detect_raw(frames=) must be a contiguous run starting on a multiple of batch '
            f'({batch}) and ending on one or at the clip end; got {want[0]}..{want[-1]} of '
            f'{T_clip}. `_units` partitions on range(0, T, batch), so a misaligned slice forwards '
            'a short leading batch -- a shape the whole-clip pass never sees, and cuDNN selects '
            'algorithms per shape.')

    # THE UNIT OF WORK IS (FRAME BATCH, CAMERA CHUNK), which is what makes the peak bounded while
    # every forward keeps its exact shape. Indices are LOCAL to `want`; `_submit` maps them back to
    # source frame numbers, the only place the two ever differ.
    _units = [(list(range(st, min(st + batch, T))), list(range(lo, min(lo + cams_flight, C))))
              for st in range(0, T, batch)
              for lo in range(0, C, cams_flight)]

    def _submit(u):
        """The next unit's decode, started BEFORE the current one's forwards are run."""
        if u >= len(_units):
            return None
        fr, cams = _units[u]
        src_fr = want[fr]
        return [pool.submit(_fetch, (ci, session.cam_names[ci], src_fr)) for ci in cams]

    try:
        # ONE UNIT OF LOOKAHEAD: submitting unit i+1's decode before unit i's forwards run overlaps
        # the ~120 ms of forward+NMS with the decoder threads. Changes no pixels and no order.
        pending = _submit(0)
        for _u in range(len(_units)):
            unit_ix, _ = _units[_u]
            fetched = [f.result() for f in pending]
            nxt = _submit(_u + 1)
            for ci, x, metas, src in fetched:
                # Indexed, not unpacked: the head's return arity grows with each optional branch.
                # The /255 happens here on the device, off the uint8 `_fetch` handed back.
                _o = det(x.to(device).float().div_(_div255))
                obj, boxes, kpts = _o[0], _o[1], _o[2]
                for j, t in enumerate(unit_ix):
                    decoded = decode(obj[j], boxes[j], top_k=D, score_thresh=score_thresh,
                                     iou_thresh=iou_thresh,
                                     center_dist_thresh=center_dist_thresh,
                                     return_index=True, return_trace=trace is not None)
                    if trace is None:
                        b, s, ix = decoded
                    else:
                        b, s, ix, dt = decoded
                        record = {'frame': int(want[t]), 'camera': session.cam_names[ci],
                                  'n_total': dt['n_total'], 'n_score': dt['n_score'],
                                  'n_nms': dt['n_nms'], 'n_top_k': dt['n_top_k']}
                        if trace_detail:
                            for stage, key in (('all', 'all'), ('score', 'score'),
                                               ('nms', 'nms')):
                                bx = unletterbox_boxes(dt[f'{key}_boxes'].cpu(), *metas[j],
                                                       src_wh=src)
                                record[f'{stage}_boxes'] = bx.numpy().tolist()
                                record[f'{stage}_scores'] = dt[f'{key}_scores'].cpu().numpy().tolist()
                            final_b = unletterbox_boxes(b.cpu(), *metas[j], src_wh=src)
                            record['final_boxes'] = final_b.numpy().tolist()
                            record['final_scores'] = s.cpu().numpy().tolist()
                        trace.append(record)
                    if not b.numel():
                        continue
                    n = min(D, b.shape[0])
                    out[:n, t, ci] = unletterbox_boxes(b.cpu(), *metas[j], src_wh=src)[:n].numpy()
                    sc[:n, t, ci] = s.cpu().numpy()[:n]
                    if kp is not None and kpts is not None:
                        # Same letterbox inverse the box goes through -- see `unletterbox_keypoints`.
                        k = unletterbox_keypoints(kpts[j, ix].cpu(), *metas[j], src_wh=src)
                        kp[:n, t, ci] = k[:n].numpy()
            # A UNIT'S FRAMES ARE DEAD HERE, so give the arena back rather than letting RSS ratchet
            # up -- see `memory.trim`. This is the loop that allocates and frees the most.
            del fetched
            _memory.trim()
            pending = nxt
    finally:
        frame_pool.shutdown()
        pool.shutdown()
    return out, sc, kp


def associate_group(raw, session, gid, max_instances, link=False, min_views=2,
                    track=True, max_move=1.0, stats=None, pose_nms=None, state=None):
    """The ASSOCIATION half: per-camera detections -> ONE ROW PER ANIMAL. Microseconds per frame.

    `raw` is `detect_raw`'s `(boxes, scores, kpts)`. Returns the same triple re-indexed so row `a`
    is one animal -- across cameras always, and across frames wherever a tracker or `link_rows` ran.

    2D / single camera: instances are the NMS survivors, ordered by score; the row index is the
    only identity there is and is NOT tracked, so row `a` at frame t and t+1 need not be the same
    animal. `link=True` adds the minimal tracker (`link_rows`); note the two levers do not overlap
    -- the tracker below is built when `track and C > 1`, and `link_rows` runs only when it was
    not.

    3D multiview: `track=True` (the DEFAULT) runs `track.CrossViewTracker` -- one cross-view target
    set carried across frames -- and `link_rows` is not run on top of it. `track=False` restores
    the memoryless per-frame `associate`.

    `pose_nms` is the one keypoint identity lever that survived measurement -- maDLC's
    instance-level NMS by keypoint containment. Six coarser cues spent as a veto, a permutation or
    a Hungarian cost were each built, measured and refuted; the ranking a K = 4 body axis supplies
    is too noisy to spend in any of those forms.

    `stats` collects `pose_nms`' fire count: a rejection rate is what its rate-matched random
    control has to be matched TO.

    `state` MAKES A SEQUENCE OF CALLS EQUAL TO ONE CALL OVER THE CONCATENATION: it holds the
    tracker and `link_rows`' `last`/`age`. `pose_nms` needs nothing (per-frame pass, `stats`
    accumulates). `state=None` builds a fresh tracker per call and is byte-identical to the
    version before this parameter existed.
    """
    import numpy as np
    import torch

    from . import identity as idy

    r_box, r_sc, r_kp = raw
    D, T, C = r_sc.shape
    S = max_instances or 1
    out = np.full((S, T, C, 4), np.nan, np.float32)
    sc = np.full((S, T, C), np.nan, np.float32)
    kp = None if r_kp is None else np.full((S, T, C) + r_kp.shape[3:], np.nan, np.float32)
    # Static rig: build once. Moving rig: `associate` triangulates per frame from (n,2) centres,
    # so it needs that frame's own (4,4) extrinsic -- built inside the loop below.
    moving = any(session.rig.moving.values())
    cgroup = None if moving else session.cgroup(gid)
    # ONE CROSS-VIEW TARGET SET instead of `associate` per frame plus `link_rows` after -- see
    # `track.py`. It subsumes both, so `link_rows` must not run on top of it.
    if state is None:
        state = {}
    if 'tracker' not in state:
        tracker = None
        if track and C > 1:
            from .track import CrossViewTracker
            tracker = CrossViewTracker(S, max_res_px=session.assoc_res_max_px,
                                       min_views=min_views,
                                       max_move=max_move)
        state['tracker'] = tracker
    tracker = state['tracker']

    def _cam(t, c):
        """This frame-camera's decoded detections as torch, plus their raw indices.

        LIVENESS IS THE SCORE, NOT THE BOX: `unletterbox_boxes` returns NaN for a detection clamped
        to no positive area, so filtering on the box would silently drop detections the unsplit
        pass kept, shifting every later index and with it the `claimed` gather.
        """
        ok = np.flatnonzero(np.isfinite(r_sc[:, t, c]))
        return torch.from_numpy(r_box[ok, t, c]), torch.from_numpy(r_sc[ok, t, c]), ok

    raw_offered = 0
    for t in range(T):
        per_cam = [_cam(t, c) for c in range(C)]
        raw_offered += sum(len(p[2]) for p in per_cam)
        if C == 1:
            b, s, ok = per_cam[0]
            n = min(S, len(ok))
            out[:n, t, 0] = b[:n].numpy()
            sc[:n, t, 0] = s[:n].numpy()
            if kp is not None:
                kp[:n, t, 0] = r_kp[ok[:n], t, 0]
            continue
        cams = session.cgroup(gid, t) if moving else cgroup
        if tracker is not None:
            out[:, t], sc[:, t], claimed = tracker.step(
                cams, [p[0] for p in per_cam], [p[1] for p in per_cam])
            if kp is not None:
                # `claimed[a, c]` is the DETECTION index that slot a took in camera c, or -1;
                # gathering by it is the only way the keypoints follow the same row assignment
                # the boxes did.
                for a in range(S):
                    for c in range(C):
                        d = int(claimed[a, c])
                        if d >= 0:
                            kp[a, t, c] = r_kp[per_cam[c][2][d], t, c]
            continue
        groups = associate(cams, [p[0] for p in per_cam],
                           max_res_px=session.assoc_res_max_px, max_instances=S,
                           min_views=min_views)
        for a, g in enumerate(groups[:S]):
            for c, box in g['boxes'].items():
                d = g['members'][c]
                out[a, t, c] = box.numpy()
                sc[a, t, c] = float(per_cam[c][1][d])
                if kp is not None:
                    kp[a, t, c] = r_kp[per_cam[c][2][d], t, c]
    pre_link = int(np.isfinite(out).all(-1).sum())
    if link and tracker is None:
        out, sc = link_rows(out, sc, max_move=max_move, extra=kp,
                            state=state.setdefault('link', {}))
    if stats is not None:
        stats['association_raw_offered'] = stats.get('association_raw_offered', 0) + raw_offered
        stats['association_pre_link'] = stats.get('association_pre_link', 0) + pre_link
    # LEAD 1, AFTER ASSOCIATION: drop a row that duplicates another row's animal, by maDLC's
    # keypoint-containment overlap rather than by IoU. It runs here, on the finished assignment,
    # because a duplicate is a property of the SEATED rows -- `decode`'s own per-box NMS cannot
    # see it.
    if pose_nms is not None and kp is not None:
        idy.pose_nms(out, kp, scores=sc, thresh=pose_nms, stats=stats)
    if stats is not None:
        stats['association_kept'] = stats.get('association_kept', 0) + \
            int(np.isfinite(out).all(-1).sum())

    return out, sc, kp


def link_rows(boxes, scores=None, max_move=1.0, max_age=24, birth_age=None, extra=None,
              state=None):
    """Reorder instance rows frame by frame so a row follows ONE animal. In place, returns both.

    WITHOUT THIS THE ROWS ARE NOT AN ANIMAL AXIS: `decode` orders by score, so row 0 at frame t
    and row 0 at frame t+1 are unrelated, and `run_group` crops each window to the union of its
    frames' boxes -- an unlinked union is many animals squeezed into one crop.

    Matching is against each row's LAST KNOWN box (not frame t-1, so a one-frame detector miss
    does not break the chain), expiring after `max_age` frames. The cost is CENTRE DISTANCE OVER
    THE MEAN BOX SIDE, gated at one side -- deliberately not IoU, which ranks by shape agreement
    (not identity) and is exactly zero under fast motion. The gate has 10-16x headroom over real
    motion.

    AN UNMATCHED ROW STAYS EMPTY rather than taking an arbitrary leftover: a force-assigned row
    TELEPORTS across the frame, and `run_group` then crops the window to the union of those
    positions -- measured at 1924x1924 against a 244 px rat. A detection nobody claimed may still
    START a row, but only an empty or expired one -- a birth, not a swap.

    THIS DROPS A THIRD OF rat-city's DETECTIONS, AND SPARE ROWS ARE THE FIX -- NOT `birth_age`.
    Relaxing eligibility (`birth_age`) buys coverage at exactly the price the strict rule exists
    to prevent: a row that changes animal mid-window spans both. Raising the ROW COUNT instead
    seats nearly everything and tightens the union, because no row has to hold two animals.
    `birth_age` defaults to None, off and byte-identical to the rule before it existed.

    `state` CARRIES THE MATCHER ACROSS A CALL BOUNDARY, so a clip processed in blocks links
    exactly as the whole clip does; without it, every block would restart identity from its own
    frame 0 -- a silent identity break at a boundary chosen by the RAM budget.

    THE ASYMMETRY IN `t0` IS THE WHOLE OF IT. Frame 0 of the CLIP is the only frame nothing is
    matched against: it seeds `last` and is left unpermuted. A block's own frame 0 is not that
    frame, so with carried state the loop starts at 0 and permutes it against the previous
    block's `last`, exactly as the whole-clip pass does mid-clip. `state=None` restores the
    original single-call behaviour byte for byte.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    S, T, C, _ = boxes.shape
    if state is None or 'last' not in state:
        last = boxes[:, 0].copy()                 # (S,C,4), each row's most recent known box
        age = np.zeros(S, int)                    # frames since this row was last seen
        t0 = 1                                    # frame 0 seeds `last` and is not permuted
    else:
        last, age, t0 = state['last'], state['age'], 0
    for t in range(t0, T):
        cur = boxes[:, t]
        cost = np.zeros((S, S), np.float32)
        for c in range(C):
            ok_p = np.isfinite(last[:, c]).all(-1)
            ok_c = np.isfinite(cur[:, c]).all(-1)
            if not (ok_p.any() and ok_c.any()):
                continue
            a, b = last[ok_p, c], cur[ok_c, c]
            ca = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], -1)
            cb = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], -1)
            d = np.linalg.norm(ca[:, None] - cb[None], axis=-1)
            sa = 0.5 * ((a[:, 2] - a[:, 0]) + (a[:, 3] - a[:, 1]))
            sb = 0.5 * ((b[:, 2] - b[:, 0]) + (b[:, 3] - b[:, 1]))
            side = 0.5 * (sa[:, None] + sb[None])
            # IN UNITS OF THE ANIMAL'S OWN SIZE, never pixels: rat-city's rats are ~250 px and
            # branson's flies ~30, so one pixel gate cannot serve both.
            gap = np.where(side > 0, d / (max_move * np.maximum(side, 1e-6)), np.inf)
            cost[np.ix_(ok_p, ok_c)] += np.clip(1.0 - gap, 0.0, None)
        rows, cols = linear_sum_assignment(-cost)
        taken = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] > 0}
        # A BIRTH, and only into a slot no live animal is using.
        claimed = set(taken.values())
        free_dets = [c for c in range(S)
                     if c not in claimed and np.isfinite(cur[c]).all(-1).any()]
        # `birth_age is None` is the shipped path, byte-identical to the rule before the knob
        # existed; the `sorted` branch only runs when a caller opts in, so the default cannot drift.
        open_rows = [r for r in range(S)
                     if r not in taken and not np.isfinite(last[r]).all(-1).any()]
        if birth_age is not None:
            # OCCUPIED is `age`, not `last`: `last` is retained for MATCHING. Oldest first, because
            # the longest-unseen row is the likeliest to be free rather than mid-blink.
            open_rows = sorted((r for r in range(S)
                                if r not in taken and (not np.isfinite(last[r]).all(-1).any()
                                                       or age[r] >= birth_age)),
                               key=lambda r: -age[r])
        for r, c in zip(open_rows, free_dets):
            taken[r] = c
        out = np.full_like(cur, np.nan)
        sc = None if scores is None else np.full_like(scores[:, t], np.nan)
        # `extra` (S,T,C,...) rides the SAME permutation. Anything indexed by row has to.
        ex = None if extra is None else np.full_like(extra[:, t], np.nan)
        for r, c in taken.items():
            out[r] = cur[c]
            if sc is not None:
                # Same assignment, or the score stops describing the box beside it.
                sc[r] = scores[:, t][c]
            if ex is not None:
                ex[r] = extra[:, t][c]
        boxes[:, t] = out
        if ex is not None:
            extra[:, t] = ex
        if sc is not None:
            scores[:, t] = sc
        seen = np.isfinite(boxes[:, t]).all(-1)
        last = np.where(seen[..., None], boxes[:, t], last)
        age = np.where(seen.any(-1), 0, age + 1)
        # EXPIRY IS A FORGET, not just a flag: a stale centre that stays in the cost matrix keeps
        # competing for the detection that belongs to whoever is there now.
        last[age > max_age] = np.nan
    if state is not None:
        state['last'], state['age'] = last, age
    return boxes if scores is None else (boxes, scores)
