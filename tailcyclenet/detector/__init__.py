import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .assign import assign, assign_tal, ciou_loss, decode, detector_loss, giou_loss
from .associate import associate
from .boxes import box_iou, paired_iou
from .data import (BoxDataset, ChunkShuffle, CohortSampler, box_collate, letterbox,
                   letterbox_transform, reduce_factor, split_batch, tile_transform,
                   unletterbox_boxes)
from .yolox import YOLOXNano


def resolve_detector_checkpoint(path, checkpoint='latest'):
    """Resolve a detector file from a run directory.

    Directory deployment defaults to the highest numbered checkpoint, not historical
    ``detector.pth`` (which is the validation-selected *best* checkpoint). ``last`` and ``best``
    remain explicit selectors; an explicit filename is also an override. ``latest`` only checks
    that the filename's iteration matches the checkpoint's own saved iteration -- it does not
    require the run's configured iteration count or metrics log to claim training completed, so
    a stopped run's highest numbered checkpoint still resolves. Only ``last`` requires
    ``detector_last.pth`` to exist.
    """
    p = Path(path)
    if not p.is_dir():
        return p
    selector = str(checkpoint or 'latest')
    if selector == 'last':
        out = p / 'detector_last.pth'
        if not out.exists():
            raise ValueError(f'{p}: --detector-checkpoint last requested, but detector_last.pth '
                             'is absent')
        return out
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
    for iteration, candidate in files:
        ckpt = torch.load(candidate, map_location='cpu', weights_only=False)
        saved_iteration = ckpt.get('iteration')
        if saved_iteration is not None and int(saved_iteration) == iteration:
            return candidate

    raise ValueError(f'{p}: no numbered detector checkpoint has matching iteration metadata')


__all__ = ['YOLOXNano', 'BoxDataset', 'ChunkShuffle', 'CohortSampler',
           'box_collate', 'letterbox',
           'letterbox_transform', 'reduce_factor', 'split_batch', 'tile_transform',
           'unletterbox_boxes', 'assign', 'assign_tal', 'box_iou',
           'ciou_loss', 'decode', 'detector_loss', 'giou_loss',
           'associate', 'LoadedDetector', 'load_detector',
           'detect_raw', 'associate_group', 'link_rows',
           'paired_iou', 'resolve_detector_checkpoint']

@dataclass
class LoadedDetector:
    """A detector checkpoint, built and ready: the model plus the facts it was trained under.

    `input_wh`, `min_crop_dim`, `box_source` and `tile_scale` are part of the weights: the
    letterbox a detector was trained under decides what an animal looks like to it, and a mismatch
    serves boxes from a different crop rule silently. `tile_scale` is None for an untiled run.
    `obj_quantiles` describe the objectness distribution this checkpoint produces (`--det-score`
    is not portable across detector generations). `path` is the resolved checkpoint file.
    """
    model: torch.nn.Module
    input_wh: tuple
    dataset: str
    min_crop_dim: int
    reduce: bool
    box_source: str
    tile_scale: float | None
    obj_quantiles: dict
    path: Path


def load_detector(path, device='cpu', checkpoint='latest'):
    """Build the detector a checkpoint (or run directory) describes -> `LoadedDetector`.

    For a run directory, `checkpoint='latest'` selects and verifies the highest complete
    `detector_it*.pth`; `'last'` selects `detector_last.pth`; `'best'` the historical
    `detector.pth`. A size the architecture cannot take is refused here, not in the first forward.
    `yolox_version`, `p2`, `shared_head` and `fpn_upsample` record the architecture the weights
    were shaped for. An absent `yolox_version` is a pre-key `trimmed` detector; it and the
    canonical tiers are deleted architectures (v0.3.4 is the last release that loads them), as is
    a keypoint branch (`n_keypoints > 0`) or a non-RGB stem (`in_channels != 3`).

    Absent `norm` means BatchNorm, which is refused (the model is GroupNorm). `tile_scale` is
    meaningless without `tile_wh`, so it is dropped (untiled runs record 1.0).
    `box_target`/`antialias`/`input_norm` (what the head regresses, how its input was resampled
    and normalised) ride on the model for `detect_raw`; absent = crop-rule boxes, bilinear
    shrinking, no normalisation.
    """
    p = resolve_detector_checkpoint(path, checkpoint=checkpoint)
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f'{p}: checkpoint must be a dictionary, got {type(ckpt).__name__}')
    if ckpt.get('kind', 'detector') != 'detector':
        raise ValueError(f"{p}: this is a {ckpt['kind']!r} checkpoint, not a detector one")
    wh = ckpt.get('input_wh') or ckpt.get('det_input_wh')
    if wh is None:
        raise ValueError(f'{p}: no input_wh in the checkpoint -- a posetail-pose detector keeps '
                         'it in its dataset config; it is not loadable here.')
    from .config import deleted_yolox_reason, input_size_error
    version = str(ckpt.get('yolox_version', 'trimmed'))
    why = deleted_yolox_reason(version)
    if why:
        raise ValueError(f'{p}: yolox={version!r} is a deleted architecture ({why}); retrain '
                         'this detector on the default (convnext-t).')
    if int(ckpt.get('n_keypoints', 0)):
        raise ValueError(f"{p}: trained with a keypoint branch (n_keypoints="
                         f"{ckpt['n_keypoints']}); the detector keypoint branch is deleted "
                         '(v0.3.4 is the last release that loads it).')
    if int(ckpt.get('in_channels', 3)) != 3:
        raise ValueError(f"{p}: in_channels={ckpt['in_channels']}; detection forwards one "
                         '3-channel frame at a time.')
    why = input_size_error(version, wh)
    if why:
        raise ValueError(f'{p}: input_wh {list(wh)}: {why}')
    norm = str(ckpt.get('norm', 'bn'))
    if norm != 'gn':
        raise ValueError(
            f'{p}: trained with {norm} normalisation; the model is GroupNorm now (there are no '
            'running statistics to load into). Retrain this detector -- see '
            '`tailcyclenet/detector/yolox.py:conv_norm_act` for why the switch was made.')
    model = YOLOXNano(version=version, p2=bool(ckpt.get('p2', False)),
                      shared_head=bool(ckpt.get('shared_head', True)),
                      fpn_upsample=str(ckpt.get('fpn_upsample', 'nearest') or 'nearest'))
    model.load_state_dict(ckpt['model_state'])
    trained_datasets = ckpt.get('datasets')
    if not trained_datasets:
        trained_datasets = [ckpt.get('dataset', '')] if ckpt.get('dataset') else []
    model.trained_datasets = [str(name) for name in trained_datasets]
    model.box_sources = {str(k): str(v) for k, v in (ckpt.get('box_sources') or {}).items()}
    model.box_target = str(ckpt.get('box_target', 'crop'))
    model.antialias = bool(ckpt.get('antialias', False))
    model.input_norm = str(ckpt.get('input_norm', 'none'))
    model.min_crop_dim = int(ckpt.get('min_crop_dim', 64))
    model.box_source = str(ckpt.get('box_source', 'keypoints'))
    ts = ckpt.get('tile_scale')
    if ckpt.get('tile_wh') is not None and ts is None:
        raise ValueError(
            f'{p}: trained on tiles ({ckpt["tile_wh"]}) but carries no `tile_scale`, so the '
            'deployment input size cannot be derived. `input_wh` here is the TILE size, not the '
            'whole-frame size -- running the frame at it is a scale shift, not a smaller input.')
    return LoadedDetector(
        model=model.to(device).eval(), input_wh=tuple(wh), dataset=str(ckpt.get('dataset', '')),
        min_crop_dim=int(ckpt.get('min_crop_dim', 64)), reduce=bool(ckpt.get('reduce', False)),
        box_source=str(ckpt.get('box_source', 'keypoints')),
        tile_scale=None if ts is None or ckpt.get('tile_wh') is None else float(ts),
        obj_quantiles=dict(ckpt.get('obj_quantiles') or {}), path=Path(p))


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
               iou_thresh=0.5, center_dist_thresh=0.5, trace=None, trace_detail=False,
               box_source=None):
    """The DETECTION half: pixels -> per-camera detections, ranked by score, unassociated.

    Inputs:
        det, input_wh, session, gid, top_k -- the detector, its input size, the session, the
            group, and `D`, the per-camera detection cap.
        score_thresh -- default 0.01 (measured, not universal): a looser floor trades a few
            extra false positives for materially fewer false negatives, and every survivor
            still passes the existing NMS and (3D) reprojection gates. Sweep per checkpoint.
        iou_thresh / center_dist_thresh -- NMS knobs threaded through so a caller can move
            them; `center_dist_thresh=None` restores the pre-A5 byte-identical NMS.
        max_frames -- the same PREFIX `infer.run_group` takes.
        frames -- detect a SLICE of the clip; arrays are `len(frames)` long and must START ON A
            GLOBAL `batch` BOUNDARY (aligned slices are byte-identical to one whole-clip pass).
        read -- replaces the decode with `(ci, cam_name, frames, pool) -> imgs`.
        trace / trace_detail -- optional decode-stage diagnostics; output unchanged.
        box_source -- this session's box source (pad 20 for 'keypoints', 0 for 'instances');
            None = the checkpoint's. An `extent` head's boxes get the crop rule HERE, in source
            px (`crop_boxes_from_extents`), so callers still receive crop-rule boxes.
    Outputs:
        (boxes (D,T,C,4), scores (D,T,C)): `d` is the d-th highest-scoring detection in that
        camera at that frame; rows become an animal axis in `associate_group`, not here.
    Side effects:
        Decode runs one thread per camera (where the wall clock lives; the forward is serial).
        Cameras in flight are bounded, never `batch` -- a different batch is a different cuDNN
        reduction order, so outputs would not be reproducible across runs. The /255 divisor is
        a 0-d tensor, not the int 255 (off by 1 ULP on CUDA).
    """
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from .. import memory as _memory
    from ..dataset import read_frames
    from .data import letterbox, reduce_factor, unletterbox_boxes
    from ..crop import crop_boxes_from_extents

    extent = getattr(det, 'box_target', 'crop') == 'extent'
    antialias = bool(getattr(det, 'antialias', False))
    input_norm = str(getattr(det, 'input_norm', 'none'))
    from .transfer_aug import normalize_input
    crop_pad = 0 if (box_source or getattr(det, 'box_source', 'keypoints')) == 'instances' else 20
    crop_mcd = int(getattr(det, 'min_crop_dim', 64))

    def _to_crop(bx, src):
        """Unletterboxed (N,4) boxes -> what callers consume: crop-rule boxes for an `extent`
        checkpoint, unchanged for a `crop` one (re-applying the rule would pad twice)."""
        return crop_boxes_from_extents(bx, src, crop_mcd, crop_pad) if extent else bx

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

    def _fetch(job):
        """Decode, reduce and letterbox one (camera, frame) job into a uint8 batch tensor.

        Inputs: job -- (ci, cam_name, src_frames) tuple from `_submit`.
        Outputs: (ci, arr (n,3,H,W) uint8, metas [(scale, pad)], src (W,H)).

        The decode is the SAME one the detector was trained on: `BoxDataset` reduces at decode
        where the frame is far above the letterbox target, and a detector fed differently-sampled
        pixels at deployment is off its own training distribution. For a tile-trained detector the
        input size is a function of the FRAME size. ONE numpy conversion serves the whole batch,
        not one torch op per frame (a tiny elementwise op through torch's intraop pool is measured
        ~60x slower than numpy's packed path), and the batch stays uint8 all the way to the device
        -- the /255 happens there, as the pose loader does, 4x less host, PCIe and device memory
        than a host-side float32 copy. Bit-identical: uint8 -> float32 is exact and the float32
        divide by 255 is correctly rounded. A source frame is dead the moment it is letterboxed,
        so it is released then.
        """
        ci, cam_name, src_frames = job
        src = session.rig.size(cam_name)
        wh = input_wh if tile_scale is None else tiled_input_wh(src, tile_scale)
        r = reduce_factor(src, wh) if reduce else 1
        imgs = _read(ci, cam_name, src_frames, pool=frame_pool, reduce=r)
        n = len(imgs)
        metas, arr = [], None
        for i in range(n):
            lb, scale, pad = letterbox(imgs[i], wh, src_wh=src, antialias=antialias)
            if input_norm != 'none':
                lb = normalize_input(lb, input_norm)
            if arr is None:
                arr = np.empty((n, 3, lb.shape[0], lb.shape[1]), np.uint8)
            arr[i] = lb.transpose(2, 0, 1)
            metas.append((scale, pad))
            imgs[i] = None
        return ci, torch.from_numpy(arr), metas, src

    _per_frame_cam = 0
    for _cam in session.cam_names:
        _src = session.rig.size(_cam)
        _wh = input_wh if tile_scale is None else tiled_input_wh(_src, tile_scale)
        _per_frame_cam = max(_per_frame_cam,
                             int(_src[0]) * int(_src[1]) * 3 + int(_wh[0]) * int(_wh[1]) * 3)
    _budget = _memory.current().share(_memory.FRACTION_DETECT)
    cams_flight = _memory.fits(_budget, _per_frame_cam * max(1, int(batch)) * 2, want=max(1, C))

    _div255 = torch.tensor(255.0, device=device)

    pool = ThreadPoolExecutor(max_workers=cams_flight)
    frame_pool = ThreadPoolExecutor(max_workers=min(16, max(1, batch)))

    if frames is not None:
        _aligned = (int(want[0]) % batch == 0
                    and (T % batch == 0 or int(want[-1]) == T_clip - 1))
        assert _aligned and np.array_equal(want, np.arange(want[0], want[0] + T)), (
            f'detect_raw(frames=) must be a contiguous run starting on a multiple of batch '
            f'({batch}) and ending on one or at the clip end; got {want[0]}..{want[-1]} of '
            f'{T_clip}. `_units` partitions on range(0, T, batch), so a misaligned slice forwards '
            'a short leading batch -- a shape the whole-clip pass never sees, and cuDNN selects '
            'algorithms per shape.')

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
        pending = _submit(0)
        for _u in range(len(_units)):
            unit_ix, _ = _units[_u]
            fetched = [f.result() for f in pending]
            nxt = _submit(_u + 1)
            for ci, x, metas, src in fetched:
                obj, boxes = det(x.to(device).float().div_(_div255))
                for j, t in enumerate(unit_ix):
                    decoded = decode(obj[j], boxes[j], top_k=D, score_thresh=score_thresh,
                                     iou_thresh=iou_thresh,
                                     center_dist_thresh=center_dist_thresh,
                                     return_trace=trace is not None)
                    if trace is None:
                        b, s = decoded
                    else:
                        b, s, dt = decoded
                        record = {'frame': int(want[t]), 'camera': session.cam_names[ci],
                                  'n_total': dt['n_total'], 'n_score': dt['n_score'],
                                  'n_nms': dt['n_nms'], 'n_top_k': dt['n_top_k']}
                        if trace_detail:
                            for stage, key in (('all', 'all'), ('score', 'score'),
                                               ('nms', 'nms')):
                                bx = _to_crop(unletterbox_boxes(dt[f'{key}_boxes'].cpu(),
                                                                *metas[j], src_wh=src), src)
                                record[f'{stage}_boxes'] = bx.numpy().tolist()
                                record[f'{stage}_scores'] = dt[f'{key}_scores'].cpu().numpy().tolist()
                            final_b = _to_crop(unletterbox_boxes(b.cpu(), *metas[j], src_wh=src),
                                               src)
                            record['final_boxes'] = final_b.numpy().tolist()
                            record['final_scores'] = s.cpu().numpy().tolist()
                        trace.append(record)
                    if not b.numel():
                        continue
                    n = min(D, b.shape[0])
                    out[:n, t, ci] = _to_crop(unletterbox_boxes(b.cpu(), *metas[j], src_wh=src),
                                              src)[:n].numpy()
                    sc[:n, t, ci] = s.cpu().numpy()[:n]
            del fetched
            _memory.trim()
            pending = nxt
    finally:
        frame_pool.shutdown()
        pool.shutdown()
    return out, sc


def associate_group(raw, session, gid, max_instances, link=False, min_views=2,
                    track=True, max_move=1.25, max_age=8, stats=None,
                    state=None, duplicate_radius=0.75, duplicate_persist=5, frame_base=0):
    """The ASSOCIATION half: per-camera detections -> ONE ROW PER ANIMAL. Microseconds per frame.

    Inputs:
        raw -- `detect_raw`'s (boxes, scores); session, gid, max_instances (row count S).
        track -- 3D multiview default: `CrossViewTracker`, one cross-view target set carried
            across frames; `False` restores the memoryless per-frame `associate`.
        link -- adds `link_rows` in 2D only (the tracker builds when `track and C > 1`).
        min_views / max_move -- passed through to the tracker / `associate`; max_move defaults
            to the measured 1.25 box-side gate.
        max_age -- the tracker's and `link_rows`' SHARED patience window: frames without
            evidence before a slot or row is retired. The measured default is 8.
        stats -- accumulates association counters when given.
        state -- makes calls equal to one concatenated call; `None` builds fresh state.
        duplicate_radius / duplicate_persist -- the 3D --track backstop's duplicate-retirement
            geometry; inert without a tracker (2D or track=False).
        frame_base -- the SOURCE frame of `raw`'s own frame 0, threaded to a freshly built
            tracker so its `identity_events` are source-absolute even on a ranged run whose
            tracker is first built on an aligned lead-in batch; inert once `state` already
            holds one (the tracker is built exactly once).
    Outputs:
        The same pair re-indexed so row `a` is one animal -- across cameras always, and across
        frames wherever a tracker or `link_rows` ran. In 2D the row index is the only identity.
    Notes:
        Cross-view geometry is built once for a static rig, per frame for a moving one.
    """
    import numpy as np
    import torch

    r_box, r_sc = raw
    D, T, C = r_sc.shape
    S = max_instances or 1
    out = np.full((S, T, C, 4), np.nan, np.float32)
    sc = np.full((S, T, C), np.nan, np.float32)
    moving = any(session.rig.moving.values())
    cgroup = None if moving else session.cgroup(gid)
    if state is None:
        state = {}
    if 'tracker' not in state:
        tracker = None
        if track and C > 1:
            from .track import CrossViewTracker
            tracker = CrossViewTracker(S, max_res_px=session.assoc_res_max_px,
                                       min_views=min_views,
                                       max_move=max_move, max_age=max_age,
                                       duplicate_radius=duplicate_radius,
                                       duplicate_persist=duplicate_persist,
                                       frame_base=frame_base)
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
            continue
        cams = session.cgroup(gid, frame_base + t) if moving else cgroup
        if tracker is not None:
            out[:, t], sc[:, t], _ = tracker.step(
                cams, [p[0] for p in per_cam], [p[1] for p in per_cam])
            continue
        groups = associate(cams, [p[0] for p in per_cam],
                           max_res_px=session.assoc_res_max_px, max_instances=S,
                           min_views=min_views)
        for a, g in enumerate(groups[:S]):
            for c, box in g['boxes'].items():
                d = g['members'][c]
                out[a, t, c] = box.numpy()
                sc[a, t, c] = float(per_cam[c][1][d])
    pre_link = int(np.isfinite(out).all(-1).sum())
    if link and tracker is None:
        out, sc = link_rows(out, sc, max_move=max_move, max_age=max_age,
                            state=state.setdefault('link', {}))
    if stats is not None:
        stats['association_raw_offered'] = stats.get('association_raw_offered', 0) + raw_offered
        stats['association_pre_link'] = stats.get('association_pre_link', 0) + pre_link
    if stats is not None:
        stats['association_kept'] = stats.get('association_kept', 0) + \
            int(np.isfinite(out).all(-1).sum())

    return out, sc


def link_rows(boxes, scores=None, max_move=1.0, max_age=24, state=None):
    """Reorder instance rows frame by frame so a row follows ONE animal. In place, returns both.

    Inputs:
        boxes (S,T,C,4), scores (S,T,C) -- `scores` rides the same permutation.
        max_move -- the identity gate, in units of the animal's own mean box side (never
            pixels), gated at one side.
        max_age -- frames without evidence before a row's last box is forgotten.
        state -- carries the matcher across a call boundary so a clip processed in blocks links
            exactly as the whole clip does.
    Outputs:
        boxes (and scores when given), reordered so row `a` follows one animal.
    Side effects:
        In place on `boxes` and `scores`. An unmatched row stays EMPTY rather than taking
        an arbitrary leftover (a force-assigned row teleports across the frame and widens the
        crop union); an unclaimed detection may still START a row, but only an empty or expired
        one.
    Notes:
        Matching is against each row's LAST KNOWN box (not frame t-1), so a one-frame
        detector miss does not break the chain. Frame 0 of the clip is the only frame
        nothing is matched against; with carried state a block's frame 0 is permuted
        against the previous block's `last`, exactly as the whole-clip pass does mid-clip.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from .boxes import centres as box_centres
    from .boxes import sides as box_sides

    S, T, C, _ = boxes.shape
    if state is None or 'last' not in state:
        last = boxes[:, 0].copy()
        age = np.zeros(S, int)
        t0 = 1
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
            d = np.linalg.norm(box_centres(a)[:, None] - box_centres(b)[None], axis=-1)
            sa, sb = box_sides(a), box_sides(b)
            side = 0.5 * (sa[:, None] + sb[None])
            gap = np.where(side > 0, d / (max_move * np.maximum(side, 1e-6)), np.inf)
            cost[np.ix_(ok_p, ok_c)] += np.clip(1.0 - gap, 0.0, None)
        rows, cols = linear_sum_assignment(-cost)
        taken = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] > 0}
        claimed = set(taken.values())
        free_dets = [c for c in range(S)
                     if c not in claimed and np.isfinite(cur[c]).all(-1).any()]
        open_rows = [r for r in range(S)
                     if r not in taken and not np.isfinite(last[r]).all(-1).any()]
        for r, c in zip(open_rows, free_dets):
            taken[r] = c
        out = np.full_like(cur, np.nan)
        sc = None if scores is None else np.full_like(scores[:, t], np.nan)
        for r, c in taken.items():
            out[r] = cur[c]
            if sc is not None:
                sc[r] = scores[:, t][c]
        boxes[:, t] = out
        if sc is not None:
            scores[:, t] = sc
        seen = np.isfinite(boxes[:, t]).all(-1)
        last = np.where(seen[..., None], boxes[:, t], last)
        age = np.where(seen.any(-1), 0, age + 1)
        last[age > max_age] = np.nan
    if state is not None:
        state['last'], state['age'] = last, age
    return boxes if scores is None else (boxes, scores)
