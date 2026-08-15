import torch

from .assign import (assign, box_iou, certified_anchors, decode, detector_loss,
                     giou_loss)
from .associate import associate
from .data import (BoxDataset, ChunkShuffle, box_collate, letterbox, letterbox_transform,
                   reduce_factor, split_batch, tile_transform, unletterbox_boxes,
                   unletterbox_keypoints)
from .yolox import YOLOXNano

__all__ = ['YOLOXNano', 'BoxDataset', 'ChunkShuffle', 'box_collate', 'letterbox',
           'letterbox_transform', 'reduce_factor', 'split_batch', 'tile_transform',
           'unletterbox_boxes', 'unletterbox_keypoints', 'assign', 'box_iou', 'certified_anchors',
           'decode', 'detector_loss', 'giou_loss', 'associate', 'LINK_REV', 'RAW_REV',
           'detect_raw', 'associate_group', 'detect_group', 'link_rows']

# BUMP THIS WHENEVER `link_rows` CHANGES WHAT IT EMITS. `--det-cache` stores boxes that have already
# been linked, so a cache written under an older rule is a different box set under an identical
# stamp -- exactly the silent mismatch the stamp exists to catch (`scripts/infer.py`). Rev 2 is the
# gated centre-distance matcher with births and expiry; rev 1 was ungated IoU with `free.pop(0)`.
# `birth_age` did NOT bump this: it defaults to None, which is byte-identical to rev 2, and it is
# unreachable from the CLI precisely because the sweep in `link_rows` refutes turning it on.
LINK_REV = 2

# BUMP THIS WHENEVER `detect_raw` CHANGES WHAT IT EMITS, and note that it is UNCONDITIONAL in the
# `--det-cache` stamp for the reason `det_score` and `track` are. A cache is now RAW -- per-camera
# detections, unassociated -- where every cache written before the split held boxes that had already
# been through `associate`/`track`/`link_rows`. The two are the same shape and the same dtype under
# what would otherwise be an identical stamp, so an old cache must be REFUSED rather than
# reinterpreted: reading associated boxes as raw ones would silently associate them twice.
RAW_REV = 1


def load_detector(path, device='cpu', input_wh=None):
    """(model, input_wh, dataset_name, min_crop_dim, reduce, box_source, tile_scale) from a folder.

    THE INPUT SIZE IS PART OF THE WEIGHTS, not a runtime choice: the letterbox the detector was
    trained under decides what an animal looks like to it, and a square 416 puts the median rat
    at 15.8 x 12.5 px where an aspect-matched 896x384 does not. So it is read from the checkpoint
    -- except that posetail-pose's own detectors predate that field entirely (they carry
    `dataset`, `epoch`, `eval`, `max_instances`, `strategy` and nothing else) and the size lives
    in a config file this repo does not have. `input_wh` supplies it for those; guessing a default
    would silently run a detector at a size it never saw.

    `min_crop_dim` rides along for the same reason at a smaller scale: it is the floor in the crop
    rule the detector exists to reproduce, so a pose run whose `[data].min_crop_dim` differs is
    being served boxes from a different rule -- silently, since the shapes and the losses are
    identical either way. 64 for a checkpoint predating the field; every shipped config says 64.

    `box_source` rides along for the third time on the same theme: it says whether the boxes this
    detector reproduces came from the keypoint extent or from `instances.pq`, which are two
    different crop rules. The caller checks it against the pose run's own -- as a warning, not a
    failure, because the best rat-city detector on record is `instances`-trained while every
    rat-city pose run is keypoint-trained, and running that pair is a legitimate arm as long as
    nobody reads its delta as detector quality.

    `tile_scale` is the fourth instance of the same theme and the most dangerous one. A TILE-TRAINED
    detector's `input_wh` is its TILE size, which is NOT its deployment input size: the invariant
    that makes train-on-tiles / infer-on-whole-frame work is the animal's size in INPUT pixels, so
    deployment must letterbox the whole frame at the same source->input scale, i.e. at
    `round(frame_wh * tile_scale)` -- per camera, because `rat-city-annotated` ships 4696x2048
    beside 4500x2050. Feeding a tile-trained detector its tile size on a whole frame is a 1/scale
    scale shift and is reported in the literature as producing near-zero precision.

    `None` means "not tile-trained, use `input_wh` as-is". A tiled run therefore has to record it,
    and `detect_group` derives the size from it. This is gotcha 12's shape: two values load the same
    tensors, and the silently-wrong one cost +23.1 mm MPJPE for weeks.
    """
    import torch
    from pathlib import Path
    p = Path(path)
    if p.is_dir():
        p = p / 'detector.pth'
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    wh = input_wh or ckpt.get('input_wh') or ckpt.get('det_input_wh')
    if wh is None:
        raise ValueError(f'{p}: no input_wh in the checkpoint -- a posetail-pose detector keeps '
                         'it in its dataset config. Pass --det-input-wh W H (rat-city 896 384, '
                         'branson-fly 416 416).')
    # `norm` absent means BatchNorm, and here that is a FACT about the file rather than gotcha
    # 12's assertion about weights nobody recorded: the key did not exist until the model became
    # GroupNorm, so every checkpoint without it is a BN one. The load would fail anyway -- BN
    # carries `running_mean` / `running_var` / `num_batches_tracked` that GN does not -- but it
    # fails with a wall of key names that says nothing about the cause.
    norm = str(ckpt.get('norm', 'bn'))
    if norm != 'gn':
        raise ValueError(
            f'{p}: trained with {norm} normalisation; the model is GroupNorm now (there are no '
            'running statistics to load into). Retrain this detector -- see '
            '`tailcyclenet/detector/yolox.py:conv_norm_act` for why the switch was made.')
    model = YOLOXNano(n_keypoints=int(ckpt.get('n_keypoints', 0)))
    model.load_state_dict(ckpt['model_state'])
    ts = ckpt.get('tile_scale')
    if ckpt.get('tile_wh') is not None and ts is None:
        raise ValueError(
            f'{p}: trained on tiles ({ckpt["tile_wh"]}) but carries no `tile_scale`, so the '
            'deployment input size cannot be derived. `input_wh` here is the TILE size, not the '
            'whole-frame size -- running the frame at it is a scale shift, not a smaller input.')
    # AND `tile_scale` IS MEANINGLESS WITHOUT `tile_wh`, so it is dropped here rather than trusted.
    # `train_detector.py` records the flag's default (1.0) on every run, tiled or not, so an
    # UNTILED checkpoint carries `tile_scale = 1.0` and `detect_group` would read that as "derive
    # the input size from the frame" and letterbox the whole frame at its native size -- the fly's
    # 1024x1024 against the 416x416 it trained at. That is the same 1/scale shift this field exists
    # to prevent, arriving through the branch that is supposed to be the safe one. Normalised at
    # the READ, not at the write, because every checkpoint already on disk has it.
    return (model.to(device).eval(), tuple(wh), str(ckpt.get('dataset', '')),
            int(ckpt.get('min_crop_dim', 64)), bool(ckpt.get('reduce', False)),
            str(ckpt.get('box_source', 'keypoints')),
            None if ts is None or ckpt.get('tile_wh') is None else float(ts))


def tiled_input_wh(src_wh, tile_scale):
    """The whole-frame input size a tile-trained detector must be deployed at.

    The invariant is the ANIMAL'S SIZE IN INPUT PIXELS, not the image size: a convnet is
    translation-equivariant, not scale-invariant. So a detector trained on native-scale tiles must
    see the whole frame at native scale too -- feeding it the tile size instead is a 1/scale shift
    and is reported in the literature as costing essentially all precision.

    Rounded to a multiple of 32, the coarsest stride, exactly as `train_detector.input_wh_for`
    rounds. That perturbs the scale by under one part in `src/32` (0.7% on rat-city's 4696) which is
    far inside the augmentation's own +-25%, and it keeps every feature map an integer size.
    """
    return tuple(max(64, int(round(float(v) * float(tile_scale) / 32) * 32)) for v in src_wh)


@torch.no_grad()
def detect_raw(det, input_wh, session, gid, top_k, device='cpu', batch=16, score_thresh=0.99,
               reduce=False, max_frames=0, tile_scale=None):
    """The DETECTION half: pixels -> per-camera detections, ranked by score, unassociated.

    -> (boxes (D,T,C,4), scores (D,T,C), kpts (D,T,C,K,3) or None) with `D = top_k`, where index
    `d` is the d-th highest-scoring detection IN THAT CAMERA AT THAT FRAME and means nothing across
    cameras or across frames. Rows become an animal axis in `associate_group`, not here.

    **THIS SPLIT EXISTS SO EVERY ASSOCIATION ARM SHARES ONE DETECTION PASS.** Detection is the
    expensive half of a run -- report 14 measures it as decode-bound, 44 ms per 4K frame-camera
    against a 0.86 ms forward -- and every identity lever changes only what happens after it. Before
    the split, an arm that moved `--track` or `--max-animals` had to re-detect, and two arms were
    matched only by trusting the detector to be deterministic (eval rule 4). Now they are matched by
    construction: one cache, byte-identical pixels, one lever.

    It also separates two levers `--max-animals` used to weld together. That flag set `decode`'s
    `top_k` AND the row count `S` at once, so the spare-rows sweep in `link_rows` could not be run
    end to end without also changing the detection budget. Detect once at `top_k = 24` and pass `S`
    to `associate_group` and the row count moves alone.

    `score_thresh` DEFAULTS TO 0.99, not to `decode`'s 0.05. The objectness is saturated -- 98.5% of
    rat-city's boxes and 99.98% of 3dpop's sit at exactly 1.0 -- so 0.05 through 0.5 are the same
    threshold in practice and the live range starts at 0.99, where dropping the bottom few percent is
    worth MOTA +0.074 [+0.009, +0.154] on 3dpop and +0.073 on rat-city, entirely out of `fp_none`.
    `decode` keeps 0.05 deliberately: it is also the training-time and detector-scoring primitive,
    and `eval_detector.py`'s numbers in dev/reports/10 are all at 0.05.

    It stays applied HERE rather than offline, even though a threshold can only ever remove boxes:
    `decode` keeps the top-`top_k` survivors ABOVE it, so a higher threshold can admit a box a lower
    one crowded out. The two are not the same set and the cache stamp records it.

    `max_frames` is the same PREFIX `infer.run_group` takes, and it has to be honoured here or the
    two disagree about the clip: rat-city's one test group is 57,594 frames and the protocol is its
    first 480, so detecting the whole group threw away 99.2% of the detection.
    """
    import numpy as np
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from ..dataset import read_frames
    from .data import letterbox, reduce_factor, unletterbox_boxes, unletterbox_keypoints

    group = session.groups[gid]
    T = min(group.n_frames, max_frames or group.n_frames)
    C = len(session.rig)
    D = max(1, int(top_k))
    out = np.full((D, T, C, 4), np.nan, np.float32)
    sc = np.full((D, T, C), np.nan, np.float32)
    # (D,T,C,K,3) of (x, y, score_logit) in SOURCE pixels, or None when this detector has no
    # keypoint branch. Kept beside the boxes rather than returned separately-shaped so a caller
    # that indexes `[d, t, ci]` for a box indexes the same way for its keypoints.
    K_det = int(getattr(det, 'n_keypoints', 0))
    kp = np.full((D, T, C, K_det, 3), np.nan, np.float32) if K_det else None

    # ONE THREAD PER CAMERA FOR THE DECODE, and it is where this function's wall clock lives:
    # profiled on 3dpop (four 3840x2160 cameras) the decode is 35-48 ms per frame-camera against a
    # 0.4 ms detector forward, 100x. decord releases the GIL inside `get_batch` and the four
    # containers share no state, so they overlap 3.5x -- but only since `dataset._read_video` took
    # a lock PER PATH instead of one global one. Sized to the rig, because that is how many
    # independent containers there are; more threads would contend on the same reader.
    # THE FORWARD STAYS SERIAL AND IN CAMERA ORDER: it is 1% of the time and moving it would put
    # two streams on one CUDA context for nothing.
    def _fetch(job):
        ci, cam_name, frames = job
        # THE SAME DECODE THE DETECTOR WAS TRAINED ON. `BoxDataset` reduces at decode where
        # the frame is far above the letterbox target, and a detector fed differently-sampled
        # pixels at deployment is being run off its own training distribution -- silently,
        # since nothing about the shapes or the scores would say so.
        src = session.rig.size(cam_name)
        # PER CAMERA, because a tile-trained detector's input size is a function of the FRAME size
        # and frame sizes vary within a root (rat-city-annotated: 4696x2048 beside 4500x2050). This
        # is the whole of "train on tiles, infer on the whole frame": one forward, no cross-tile
        # NMS, no seam handling -- only a different input size, derived rather than configured.
        wh = input_wh if tile_scale is None else tiled_input_wh(src, tile_scale)
        r = reduce_factor(src, wh) if reduce else 1
        imgs = read_frames(group, cam_name, frames, reduce=r, pool=frame_pool)
        lbs, metas = [], []
        for im in imgs:
            lb, scale, pad = letterbox(im, wh, src_wh=src)
            lbs.append(lb)
            metas.append((scale, pad))
        # ONE numpy CONVERSION FOR THE WHOLE BATCH, NOT ONE torch OP PER FRAME. Bit-identical --
        # uint8 -> float32 and a divide by 255 are both correctly rounded either way, checked in
        # `tests/test_detector.py` -- and 64x faster in wall clock: a 544x320 frame is 0.5 MP, so
        # `torch.as_tensor(...).permute(...) / 255.0` hands a tiny elementwise op to torch's
        # intraop pool, which is `nproc` wide (96 here). Measured at 67 ms PER FRAME against 1.0 ms
        # through numpy, and it was 62% of this function's wall clock and ~2,200% of one process's
        # CPU with every GPU idle. The pose loader already avoids this by keeping uint8 to the
        # device (`dataset.py`, "UINT8, not float32/255"); the detector was the one path left.
        arr = np.ascontiguousarray(np.stack(lbs).transpose(0, 3, 1, 2))
        return ci, torch.from_numpy(arr.astype(np.float32) / np.float32(255)), metas, src

    # A SECOND POOL, FOR THE OTHER HALF OF THE ROOTS. `read_frames` threads an image directory over
    # its FRAMES (it ignores `pool` for video, which decodes a batch in one `get_batch`), and that is
    # where rat-city and branson-fly spend their time -- 39 ms per `cv2.imread` of a 4696x2048 JPEG
    # (dev/reports/08), one frame at a time, on roots that are single-camera so the camera pool above
    # buys them nothing. cv2 releases the GIL. It must NOT be `pool`: `_fetch` runs IN `pool` and
    # waits on these futures, and a pool that waits on itself deadlocks the moment both are full.
    pool = ThreadPoolExecutor(max_workers=max(1, C))
    frame_pool = ThreadPoolExecutor(max_workers=min(16, max(1, batch)))

    def _submit(start):
        """The next batch's decode, started BEFORE the current one's forwards are run."""
        if start >= T:
            return None, None
        fr = list(range(start, min(start + batch, T)))
        return fr, [pool.submit(_fetch, (ci, cam, fr))
                    for ci, cam in enumerate(session.cam_names)]

    try:
        # ONE BATCH OF LOOKAHEAD. Decode is ~100% of this loop's wall clock once the pack is fixed,
        # but the ~120 ms per batch of forward + NMS is time the decoder threads spent idle: they
        # had nothing queued until the main thread came back round. Submitting batch i+1 first
        # overlaps the two. It changes no pixels and no order -- `_submit` returns the futures in
        # camera order and the forwards still run one camera at a time, in that order.
        frames, pending = _submit(0)
        while pending is not None:
            fetched = [f.result() for f in pending]
            nxt = _submit(frames[-1] + 1)
            for ci, x, metas, src in fetched:
                obj, boxes, kpts = det(x.to(device))
                for j, t in enumerate(frames):
                    b, s, ix = decode(obj[j], boxes[j], top_k=D, score_thresh=score_thresh,
                                      return_index=True)
                    if not b.numel():
                        continue
                    n = min(D, b.shape[0])
                    out[:n, t, ci] = unletterbox_boxes(b.cpu(), *metas[j], src_wh=src)[:n].numpy()
                    sc[:n, t, ci] = s.cpu().numpy()[:n]
                    if kp is not None and kpts is not None:
                        # THE SAME letterbox inverse the box goes through -- see
                        # `unletterbox_keypoints`, which is why it lives next to the box version.
                        k = unletterbox_keypoints(kpts[j, ix].cpu(), *metas[j], src_wh=src)
                        kp[:n, t, ci] = k[:n].numpy()
            frames, pending = nxt
    finally:
        frame_pool.shutdown()
        pool.shutdown()
    return out, sc, kp


def associate_group(raw, session, gid, max_instances, link=False, min_views=2, dup_res_px=None,
                    track=True, max_move=1.0):
    """The ASSOCIATION half: per-camera detections -> ONE ROW PER ANIMAL. Microseconds per frame.

    `raw` is `detect_raw`'s `(boxes, scores, kpts)`. Returns the same triple re-indexed so row `a`
    is one animal -- across cameras always, and across frames wherever a tracker or `link_rows` ran.

    2D / single camera: instances are the NMS survivors, ordered by score, and the row index is
    the only identity there is -- it is NOT tracked, so row `a` at frame t and frame t+1 need not
    be the same animal. Feeding these straight to the pose model is the honest deployment
    baseline for a single window and nothing more; a tracker belongs on top.

    `link=True` puts the smallest possible one there -- see `link_rows`. `scripts/infer.py` passes
    it ON by default; it stays off HERE because this function's contract is the honest untracked
    baseline and callers that want a tracker should say so. Note the two levers do not overlap:
    the tracker below is built when `track and C > 1`, and `link_rows` runs only when it was not,
    so in 2D single-view `link` is the whole of cross-frame identity.

    3D multiview: `track=True` (the DEFAULT) runs `track.CrossViewTracker` -- one cross-view target
    set carried across frames, so a row is one physical animal both within a frame and along the
    clip, and `link_rows` is not run on top of it. `track=False` restores the memoryless per-frame
    `associate`, whose rows are one animal within a frame and untracked across them; that is the
    arm every number before dev/reports/13 was measured on, so reproducing one needs it.
    """
    import numpy as np
    import torch

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
    tracker = None
    if track and C > 1:
        from .track import CrossViewTracker
        tracker = CrossViewTracker(S, max_res_px=session.assoc_res_max_px,
                                   min_views=min_views, dup_res_px=dup_res_px,
                                   max_move=max_move)

    def _cam(t, c):
        """This frame-camera's decoded detections as torch, plus their raw indices.

        LIVENESS IS THE SCORE, NOT THE BOX. Every decoded detection has a finite score by
        construction (it survived `score_thresh`), but `unletterbox_boxes` returns NaN for a
        detection clamped to no positive area -- so filtering on the box would silently DROP a
        detection the unsplit `detect_group` passed straight to `associate`, shifting every index
        after it and with it the `claimed` gather. Keying on the score keeps the list exactly the
        tensor `decode` returned, NaN boxes included.
        """
        ok = np.flatnonzero(np.isfinite(r_sc[:, t, c]))
        return torch.from_numpy(r_box[ok, t, c]), torch.from_numpy(r_sc[ok, t, c]), ok

    for t in range(T):
        per_cam = [_cam(t, c) for c in range(C)]
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
                # `claimed[a, c]` is the DETECTION index that slot a took in camera c, or -1.
                # Gathering by it is the only way the keypoints follow the same row assignment
                # the boxes did -- and it indexes the LIVE list, so it maps back through `ok`.
                for a in range(S):
                    for c in range(C):
                        d = int(claimed[a, c])
                        if d >= 0:
                            kp[a, t, c] = r_kp[per_cam[c][2][d], t, c]
            continue
        groups = associate(cams, [p[0] for p in per_cam],
                           max_res_px=session.assoc_res_max_px, max_instances=S,
                           min_views=min_views, dup_res_px=dup_res_px)
        for a, g in enumerate(groups[:S]):
            for c, box in g['boxes'].items():
                d = g['members'][c]
                out[a, t, c] = box.numpy()
                sc[a, t, c] = float(per_cam[c][1][d])
                if kp is not None:
                    kp[a, t, c] = r_kp[per_cam[c][2][d], t, c]
    if link and tracker is None:
        out, sc = link_rows(out, sc, max_move=max_move, extra=kp)
    return out, sc, kp


@torch.no_grad()
def detect_group(det, input_wh, session, gid, max_instances, device='cpu', batch=16,
                 score_thresh=0.99, link=False, reduce=False, max_frames=0, min_views=2,
                 dup_res_px=None, track=True, max_move=1.0, tile_scale=None, top_k=None):
    """Run the detector over every frame and camera of a group -> (boxes, scores).

    boxes (S,T,C,4), scores (S,T,C). The score is the objectness the box survived NMS on, and it
    is returned rather than dropped because `--det-score` is otherwise a re-detection per
    threshold: detection is the expensive half of a run, and a sweep over a threshold that only
    ever *removes* boxes can be done offline from what one pass already computed.

    THE COMPOSITION OF `detect_raw` AND `associate_group`, and nothing else. It is kept so no caller
    changed when the two were split, and a test pins the composition byte-identical. `top_k`
    defaults to `max_instances`, which is what this function did when the two were one.

    `score_thresh` DEFAULTS TO 0.99, not to `decode`'s 0.05. The objectness is saturated -- 98.5% of
    rat-city's boxes and 99.98% of 3dpop's sit at exactly 1.0 -- so 0.05 through 0.5 are the same
    threshold in practice and the live range starts at 0.99, where dropping the bottom few percent is
    worth MOTA +0.074 [+0.009, +0.154] on 3dpop and +0.073 on rat-city, entirely out of `fp_none`.
    `decode` keeps 0.05 deliberately: it is also the training-time and detector-scoring primitive,
    and `eval_detector.py`'s numbers in dev/reports/10 are all at 0.05.

    `max_frames` is the same PREFIX `infer.run_group` takes, and it has to be honoured here or the
    two disagree about the clip: rat-city's one test group is 57,594 frames and the protocol is its
    first 480, so detecting the whole group threw away 99.2% of the detection -- which is the
    expensive half of a run.

    2D / single camera: instances are the NMS survivors, ordered by score, and the row index is
    the only identity there is -- it is NOT tracked, so row `a` at frame t and frame t+1 need not
    be the same animal. Feeding these straight to the pose model is the honest deployment
    baseline for a single window and nothing more; a tracker belongs on top.

    `link=True` puts the smallest possible one there -- see `link_rows`. `scripts/infer.py` passes
    it ON by default; it stays off HERE because this function's contract is the honest untracked
    baseline and callers that want a tracker should say so. Note the two levers do not overlap:
    the tracker is built when `track and C > 1`, and `link_rows` runs only when it was not, so in
    2D single-view `link` is the whole of cross-frame identity.

    3D multiview: `track=True` (the DEFAULT) runs `track.CrossViewTracker` -- one cross-view target
    set carried across frames, so a row is one physical animal both within a frame and along the
    clip, and `link_rows` is not run on top of it. `track=False` restores the memoryless per-frame
    `associate`, whose rows are one animal within a frame and untracked across them; that is the
    arm every number before dev/reports/13 was measured on, so reproducing one needs it.
    """
    raw = detect_raw(det, input_wh, session, gid, top_k or max_instances or 1, device=device,
                     batch=batch, score_thresh=score_thresh, reduce=reduce, max_frames=max_frames,
                     tile_scale=tile_scale)
    out, sc, kp = associate_group(raw, session, gid, max_instances, link=link, min_views=min_views,
                                  dup_res_px=dup_res_px, track=track, max_move=max_move)
    return (out, sc) if kp is None else (out, sc, kp)


def link_rows(boxes, scores=None, max_move=1.0, max_age=24, birth_age=None, extra=None):
    """Reorder instance rows frame by frame so a row follows ONE animal. In place, returns both.

    WITHOUT THIS THE ROWS ARE NOT AN ANIMAL AXIS. `decode` orders by score, so row 0 at frame t
    and row 0 at frame t+1 are unrelated -- measured on branson-fly, the median IoU between a
    row's own consecutive boxes is 0.000 across ten near-identical flies. That matters because
    `infer.run_group` crops each window to the UNION of its frames' boxes, to stop an animal
    walking out of its own crop: fed unlinked rows, that union is 45x (branson-fly) / 59x
    (rat-city) the area of one animal and the pose model receives the whole arena squeezed into
    256 px.

    Matching is against each row's LAST KNOWN box, not against frame t-1, so a one-frame detector
    miss does not break the chain -- but that box EXPIRES after `max_age` frames, because a
    position more than a window old is not evidence about now, and an unexpiring one made a row
    permanently unavailable for the animal that actually appeared there.

    Three things this used to get wrong, and all three are visible in `scratch/phase3`'s renders:

    - **THE COST WAS IoU, WHICH IS NON-DISCRIMINATIVE IN EXACTLY THE CROWDED CASE.** Replaying
      calms21 frame 301 -> 302 from the box cache: IoU picks the WRONG mouse (row0-det1 0.512
      against row0-det0 0.233) because two touching 220 px mice overlap almost equally, while
      centre distance picks the right one (0.24 against 0.50 box sides). Hungarian-matching the
      pose to labels after that swap shows the error jump from 4-10 px to 60-82 px -- the user's
      "the points go haywire". IoU is also exactly ZERO under fast motion, where it cannot rank at
      all. So the cost is CENTRE DISTANCE OVER THE MEAN BOX SIDE, turned into an affinity that is
      positive only inside the gate.
    - **THERE WAS NO GATE.** The only test was `cost > 0`, i.e. any overlap whatsoever. Real motion
      is tiny -- consecutive-frame box-centre displacement is p90 0.06-0.11 body lengths on all
      three roots -- so a gate at ONE box side has 10-16x headroom and rejects essentially nothing
      legitimate. What it does reject is the 3.4-3.9% of 3dpop row transitions that jump more than
      a full body length (max 11) and rat-city's 8 jumps beyond two.
    - **AN UNMATCHED ROW WAS FORCE-ASSIGNED SOMEBODY ELSE'S DETECTION** (`free.pop(0)`, an
      arbitrary leftover). That is rat-city row 9, the user's "weird rat stretching across the
      whole frame where there is no rat": its per-frame boxes are normal size (170x173, 278x169)
      but TELEPORT -- x~3820 at t=0-10, x~1900 at t=12-14, back to 1937 at t=32, 3581 at t=42 --
      and `run_group` then crops the window to their union, 1924x1924 against a 244 px rat, 62x the
      area. Now an unmatched row stays EMPTY, which fixes the giant union crop at its source rather
      than bounding it downstream.

    A detection nobody claimed may still START a row, but only a row that is empty or expired --
    which is a birth, not a swap. Beyond that it is dropped: there is no row for it, and inventing
    one on top of a live animal is `fp_dup`.

    **THIS FUNCTION DROPS A THIRD OF rat-city's DETECTIONS AND `birth_age` IS THE FIX THAT DOES NOT
    WORK. THE FIX IS SPARE ROWS.** Both halves are measured, and the second is the useful one.

    The symptom, on rat-city's 500-frame clip where the detector fills 0.993 of slots
    (`scratch/phase11/probe_link.py`): 5,946 detections offered, 3,891 matched, **7 born, 2,048
    (34.4%) DROPPED** -- and 29% of the dropped were INSIDE the gate, so `max_move` is not what
    rejected them. They lost the Hungarian and had nowhere to go.

    The obvious reading is that eligibility is too strict: a row is open only when `last` is
    entirely non-finite, which needs `max_age = 24` frames of absence, one whole window. `birth_age`
    relaxes that -- a row unseen for that many frames is free even though `last` is kept for
    MATCHING. It is measured and it is REFUTED (`scratch/phase11/probe_birth_age.py`), because
    `run_group` crops a window to the UNION of a row's boxes and a row that changes animal
    mid-window spans both:

        birth_age   999(off)   8      4      2      1      0
        fill          0.652  0.716  0.730  0.764  0.816  0.993
        union p99       590   3044   3804   4090   4200   4367     px, against a 244 px rat

    Coverage is bought at exactly the price the strict rule exists to prevent, and `birth_age = 0`
    is `--no-link-boxes` (78.9 px at coverage 0.131 end to end). So it DEFAULTS TO None -- off,
    byte-identical to the rule before it existed -- and is kept only because the sweep above is
    worth more than the knob.

    **What actually works is giving births somewhere to go.** `--max-animals` sets the row count `S`
    from the LABEL count, so 12 rats get 12 rows and an unmatched detection can only be seated by
    evicting a live animal. With spare rows the STRICT rule seats nearly everything and the union
    gets TIGHTER, because no row has to hold two animals (`probe_spare_rows.py`):

        rows      12     18     24        (birth_age off throughout)
        fill/GT  0.652  0.914  1.042
        union p99  590    564    525      px

    Not the row count's fault either, strictly: it is that `S` is derived from how many animals the
    LABELS name, which is a statement about annotation and not about how many boxes a tracker needs.

    ponytail: still per-frame Hungarian on geometry alone. No appearance model, no velocity (report
    12 R2 measured that as not worth it), no re-identification after a long occlusion. `dev/reports/
    12_crossview_tracking.md` R1 is the target state -- ONE cross-view target set with one affinity,
    which deletes this function -- and this is the measurable interim that makes the renders usable
    and gives R1 a baseline to beat.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    S, T, C, _ = boxes.shape
    last = boxes[:, 0].copy()                     # (S,C,4), each row's most recent known box
    age = np.zeros(S, int)                        # frames since this row was last seen
    for t in range(1, T):
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
        # `birth_age is None` is the shipped path and is byte-identical to the rule before the knob
        # existed -- open means `last` has expired to all-NaN, and rows are paired in index order.
        # The `sorted` branch only runs when a caller opts in, so the default cannot drift.
        open_rows = [r for r in range(S)
                     if r not in taken and not np.isfinite(last[r]).all(-1).any()]
        if birth_age is not None:
            # OCCUPIED is `age`, not `last`: `last` is retained for MATCHING. Oldest first, because
            # `free_dets` pairs by position and the longest-unseen row is the likeliest to be free
            # rather than mid-blink.
            open_rows = sorted((r for r in range(S)
                                if r not in taken and (not np.isfinite(last[r]).all(-1).any()
                                                       or age[r] >= birth_age)),
                               key=lambda r: -age[r])
        for r, c in zip(open_rows, free_dets):
            taken[r] = c
        out = np.full_like(cur, np.nan)
        sc = None if scores is None else np.full_like(scores[:, t], np.nan)
        # `extra` (S,T,C,...) rides the SAME permutation. Anything indexed by row has to, or it
        # silently describes a different animal than the box beside it.
        ex = None if extra is None else np.full_like(extra[:, t], np.nan)
        for r, c in taken.items():
            out[r] = cur[c]
            if sc is not None:
                # The SAME assignment, or the score stops describing the box beside it.
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
    return boxes if scores is None else (boxes, scores)
