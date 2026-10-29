"""The DETECTION half of deployment: frames -> per-camera detections (`detect_raw`), and the
driver's detection cursor over a group (`detector_boxes`)."""
from __future__ import annotations

import numpy as np
import torch

from ..detector.boxes import NMS_CENTER_DIST
from ..detector.checkpoint import tiled_input_wh
from ..detector.decode import decode


@torch.no_grad()
def detect_raw(det, input_wh, session, gid, top_k, device='cpu', batch=16, score_thresh=0.01,
               reduce=False, max_frames=0, tile_scale=None, frames=None, read=None,
               iou_thresh=0.5, center_dist_thresh=NMS_CENTER_DIST, trace=None,
               trace_detail=False,
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
        box_source -- this session's box source: 'instances' extents are already padded (pad 0);
            'keypoints' get the checkpoint's own `crop_pad_frac`;
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
    from ..data.pixels import read_frames
    from ..detector.letterbox import letterbox, reduce_factor, unletterbox_boxes
    from ..data.crop import crop_boxes_from_extents

    extent = getattr(det, 'box_target', 'crop') == 'extent'
    antialias = bool(getattr(det, 'antialias', False))
    input_norm = str(getattr(det, 'input_norm', 'none'))
    from ..detector.augment import normalize_input
    crop_pad_frac = (0.0 if (box_source or getattr(det, 'box_source', 'keypoints')) == 'instances'
                     else float(getattr(det, 'crop_pad_frac', 0.05)))
    crop_mcd = int(getattr(det, 'min_crop_dim', 32))

    def _to_crop(bx, src):
        """Unletterboxed (N,4) boxes -> what callers consume: crop-rule boxes for an `extent`
        checkpoint, unchanged for a `crop` one (re-applying the rule would pad twice)."""
        return crop_boxes_from_extents(bx, src, crop_mcd, 0.0, crop_pad_frac) if extent else bx

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


DET_BATCH = 16


def detector_boxes(det, det_wh, sess, gid, args, device, det_red, det_tile, n_det, n_want,
                    stats=None, box_source=None):
    """-> `boxes_for(store, lo, hi)`, detecting and associating on demand for one group.

    The detection cursor is not the block cursor: blocks are sized by free memory, and a
    budget-derived batch would change the input shape and move the boxes. Detection advances in
    its own globally `batch`-aligned runs and overshoots the block by up to `batch - 1` frames,
    which the buffer below holds. `associate_group` carries its state, so identity sees one
    continuous clip. Frames come from `store.read`, so the detector consumes the same decode the
    pose loop crops -- one pass over the video.

    One association state serves the whole group: restarting every `DET_BATCH` frames would
    break identity every 16 frames. The cursor starts on a global `DET_BATCH` boundary at or
    below `frame_start`, and the lead-in is discarded -- `detect_raw` asserts a slice starts on a
    multiple of `batch`, since a short leading batch is an input SHAPE the whole-clip pass never
    produces, so the boxes stay byte-identical to the whole-clip run's (do not relax the assert
    or quantise `--start-frame` by 16). The cost is at most `batch - 1` extra frame-cameras,
    once per group; leaving the cursor at 0 would decode every frame from 0 to `frame_start`
    into a store sized for the work.
    """
    from .associate import associate_group

    T = min(sess.groups[gid].n_frames, args.frame_stop or sess.groups[gid].n_frames)
    assoc_state, buf = {}, {}
    cursor = args.frame_start - args.frame_start % DET_BATCH

    def boxes_for(store, lo, hi):
        """Boxes for the frames [lo, hi): detect/associate on demand, served from the buffer.

        Detection advances a group-wide cursor in `DET_BATCH` runs and overshoots `hi`; the
        results are buffered by source frame and sliced to the requested range on return.

        Every frame is consumed by this window; later windows may need only the suffix beginning
        at their start, so buffered raw detections before that next start are discarded.

        `max_frames=T` passes the resolved STOP index -- not `args.max_frames` (0 whenever the
        range came in as --start-frame/--end-frame): it tells `detect_raw` where the clip ends,
        and its alignment assert accepts a short final slice only at that end. The aligned lead-in
        below `frame_start` was detected to keep the batch partition and associated so the tracker
        enters the range warm; it falls out of the buffer with every frame below `lo`, which can
        never be asked for again (blocks advance monotonically).
        """
        nonlocal cursor
        while cursor < hi:
            end = min(cursor + DET_BATCH, T)
            raw = detect_raw(det, det_wh, sess, gid, n_det, device=device,
                             score_thresh=args.det_score, reduce=det_red,
                             iou_thresh=args.det_nms_iou,
                             center_dist_thresh=args.det_nms_center_dist,
                             max_frames=T, tile_scale=det_tile, box_source=box_source,
                             frames=np.arange(cursor, end),
                             trace=(stats.setdefault('decode_trace', [])
                                    if args.det_trace else None),
                             read=lambda ci, cam, fr, pool=None, reduce=1: store.read(
                                 ci, cam, fr, pool=pool, reduce=reduce))
            b, s = associate_group(raw, sess, gid, n_want, link=args.link_boxes,
                                   min_views=args.min_views, track=args.track,
                                   max_move=args.max_move,
                                   max_age=args.max_age, stats=stats,
                                   state=assoc_state,
                                   duplicate_radius=args.duplicate_radius,
                                   duplicate_persist=args.duplicate_persist,
                                   assoc_res_max_sides=args.assoc_res_max_sides,
                                   assoc_score_order=args.assoc_score_order,
                                   track_age_decay=args.track_age_decay,
                                   track_affinity_2d=args.track_affinity_2d,
                                   track_camera_claims=args.track_camera_claims,
                                   assoc_fast=args.assoc_fast,
                                   assoc_random_drop=args.assoc_random_drop,
                                   frame_base=cursor)
            for j, t in enumerate(range(cursor, end)):
                buf[t] = (b[:, j], s[:, j])
            if stats is not None:
                stats['filled'] = stats.get('filled', 0) + int(np.isfinite(b).all(-1).sum())
                stats['slots'] = stats.get('slots', 0) + int(np.isfinite(b).all(-1).size)
                if assoc_state.get('tracker') is not None:
                    stats['identity_events'] = assoc_state['tracker'].events
            cursor = end
        want = range(lo, min(hi, T))
        out = tuple(np.stack([buf[t][i] for t in want], 1) for i in range(2))
        for t in [t for t in buf if t < lo]:
            del buf[t]
        return out

    return boxes_for
