import torch

from .assign import (assign, box_iou, certified_anchors, decode, detector_loss,
                     giou_loss)
from .associate import associate
from .data import (BoxDataset, ChunkShuffle, box_collate, letterbox, letterbox_transform,
                   reduce_factor, split_batch, tile_transform, unletterbox_boxes,
                   unletterbox_keypoints)
from .yolox import YOLOX_TIERS, YOLOXNano

__all__ = ['YOLOXNano', 'YOLOX_TIERS', 'BoxDataset', 'ChunkShuffle', 'box_collate', 'letterbox',
           'letterbox_transform', 'reduce_factor', 'split_batch', 'tile_transform',
           'unletterbox_boxes', 'unletterbox_keypoints', 'assign', 'box_iou', 'certified_anchors',
           'decode', 'detector_loss', 'giou_loss', 'associate', 'LINK_REV', 'RAW_REV',
           'detect_raw', 'associate_group', 'link_rows']

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
    """(model, input_wh, dataset_name, min_crop_dim, reduce, box_source, tile_scale, obj_q).

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

    `yolox_version` is the fifth instance of the same theme: it says which architecture the
    weights were shaped for (`'trimmed'` -- the repo's bespoke net -- or a canonical tier name,
    see `tailcyclenet.detector.yolox.YOLOX_TIERS`), and absent means `'trimmed'` -- a fact about
    every checkpoint written before the capacity switch existed, not a guess about one that could
    have been anything. The return signature here is UNCHANGED: this is used only to build the
    right model internally, so no caller needs a new field to keep working.
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
    model = YOLOXNano(n_keypoints=int(ckpt.get('n_keypoints', 0)),
                      version=str(ckpt.get('yolox_version', 'trimmed')))
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
            None if ts is None or ckpt.get('tile_wh') is None else float(ts),
            # THE OBJECTNESS DISTRIBUTION THIS CHECKPOINT PRODUCES, or {} for one trained before
            # it was recorded. `--det-score` is not portable across detector GENERATIONS -- see
            # `scripts/infer.py`, which warns rather than guessing a threshold on the caller's
            # behalf, because the right value depends on whether coverage or identity is wanted.
            dict(ckpt.get('obj_quantiles') or {}))


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
def detect_raw(det, input_wh, session, gid, top_k, device='cpu', batch=16, score_thresh=0.5,
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
    from .. import memory as _memory
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
        # ONE numpy CONVERSION FOR THE WHOLE BATCH, NOT ONE torch OP PER FRAME, and it stays that
        # way: a 544x320 frame is 0.5 MP, so `torch.as_tensor(...).permute(...) / 255.0` hands a
        # tiny elementwise op to torch's intraop pool (`nproc` wide, 96 here) -- measured at 67 ms
        # PER FRAME against 1.0 ms through numpy, 62% of this function's wall clock at ~2,200% CPU
        # with every GPU idle. The batch is still packed once, in numpy.
        #
        # WHAT CHANGED IS THE DTYPE AND THE NUMBER OF LIVE COPIES. This used to build, per camera
        # per batch: `imgs` + `lbs` + `np.stack` + `ascontiguousarray` + a float32 `astype` -- five
        # buffers, the last at 4 bytes a pixel. On johnson (16 cameras, 3208x2200, whole-frame
        # input at `tile_scale = 1.0`) that is ~2.4 GB per camera per batch, x16 cameras x2 batches
        # in flight = ~76 GB of the measured 120 GB peak, against ~2.5 GB of live pixels.
        #
        # Now: letterbox straight into a preallocated uint8 (n,3,h,w) and DROP EACH SOURCE FRAME AS
        # IT IS CONSUMED, so `imgs` shrinks while `arr` fills instead of both standing at full
        # size. uint8 goes to the device and the `/255` happens THERE -- which is what the pose
        # loader has always done (`dataset.py`, "UINT8, not float32/255") and the detector was the
        # one path left. 4x less host memory, 4x less to copy over PCIe, 4x less device memory.
        #
        # BIT-IDENTICAL, and that is the whole reason this is allowed to be a memory change rather
        # than a numerical one: uint8 -> float32 is exact and the float32 divide by 255 is
        # correctly rounded, so the tensor the detector sees is the one it always saw. **This is
        # NOT free -- see `_DIV255` below, where getting it wrong costs 1 ULP and every cached
        # detection.** Pinned by `tests/test_detector_memory.py`.
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

    # WHAT IS BOUNDED IS THE NUMBER OF CAMERAS IN FLIGHT, AND NEVER `batch`. `2 x C x batch` full
    # frames is what OOMs -- every camera is fetched for every frame batch and one batch of
    # lookahead is live, which on johnson is 2 x 16 x 16 x 42.4 MB = 21.7 GB even after the dtype
    # fix above, on a node that may have 16 GB in total.
    #
    # **`batch` IS NOT AVAILABLE AS A MEMORY KNOB, BECAUSE IT IS NOT INERT.** It looks inert --
    # `tests/test_detector.py`'s `--det-cache` stamp guard classifies it as `plumbing`, i.e.
    # asserts it cannot change the detections and therefore need not be stamped -- and that
    # assertion is FALSE on a GPU. Measured on johnson's own detector, 12 frames, batch 16 against
    # batch 3: **boxes differ by 0.204 px, scores by 1.69e-03, keypoints by 0.447 px.** cuDNN
    # selects convolution algorithms per input SHAPE, so a different batch is a different
    # reduction order. It is not the decode: `read_frames` at batch 16 versus batch 3, and cached
    # against reopened, are byte-identical (checked directly on this root's video).
    #
    # Making `batch` depend on free memory would therefore mean two runs of one command on two
    # machines producing DIFFERENT BOXES and silently sharing a `--det-cache` -- precisely the
    # class of divergence the stamp exists to prevent, introduced underneath it by the thing meant
    # to be output-neutral. So `batch` is passed through untouched and the memory comes out of the
    # camera axis instead.
    #
    # CHUNKING CAMERAS IS OUTPUT-NEUTRAL BY CONSTRUCTION: each camera is forwarded on its own, with
    # its own `(batch, 3, h, w)` tensor, and `out`/`sc`/`kp` are indexed by `[.., ci]`. How many
    # cameras happen to be decoding at the same moment changes no forward's shape and no array's
    # contents -- only the wall clock, via how much of the decode overlaps.
    #
    # SIZED FROM PARSED TOML ONLY (`rig.size`), so this opens, stats and decodes nothing -- the
    # gotcha-10 rule that governs every other sizing decision in this repo.
    _per_frame_cam = 0
    for _cam in session.cam_names:
        _src = session.rig.size(_cam)
        _wh = input_wh if tile_scale is None else tiled_input_wh(_src, tile_scale)
        _per_frame_cam = max(_per_frame_cam,
                             int(_src[0]) * int(_src[1]) * 3 + int(_wh[0]) * int(_wh[1]) * 3)
    _budget = _memory.current().share(_memory.FRACTION_DETECT)
    cams_flight = _memory.fits(_budget, _per_frame_cam * max(1, int(batch)) * 2, want=max(1, C))

    # A 0-d TENSOR, NOT THE PYTHON INT 255, AND THE DIFFERENCE IS NOT COSMETIC.
    #
    # Moving the `/255` off the host is only allowed because it is bit-identical, and on CUDA
    # `x / 255` is NOT: dividing by a Python scalar takes a reciprocal-multiply fast path that is
    # off by 1 ULP on 156 of the 256 possible byte values. Dividing by a 0-d tensor is correctly
    # rounded and matches numpy exactly. Measured, both ways, in `tests/test_detector_memory.py`.
    #
    # 1 ULP on the INPUT is not a rounding curiosity here: it perturbs every objectness score,
    # which reorders NMS ties, which returns different boxes -- so it would silently invalidate
    # every `--det-cache` on disk and make no recorded detector number reproducible, without
    # changing a shape, a dtype or a `RAW_REV`. Exactly the class of silent divergence the cache
    # stamp exists to prevent, arriving underneath it.
    _div255 = torch.tensor(255.0, device=device)

    pool = ThreadPoolExecutor(max_workers=cams_flight)
    # A SECOND POOL, FOR THE OTHER HALF OF THE ROOTS. `read_frames` threads an image directory over
    # its FRAMES (it ignores `pool` for video, which decodes a batch in one `get_batch`), and that is
    # where rat-city and branson-fly spend their time -- 39 ms per `cv2.imread` of a 4696x2048 JPEG
    # (dev/reports/08), one frame at a time, on roots that are single-camera so the camera pool above
    # buys them nothing. cv2 releases the GIL. It must NOT be `pool`: `_fetch` runs IN `pool` and
    # waits on these futures, and a pool that waits on itself deadlocks the moment both are full.
    frame_pool = ThreadPoolExecutor(max_workers=min(16, max(1, batch)))

    # THE UNIT OF WORK IS (FRAME BATCH, CAMERA CHUNK), which is what makes the peak bounded while
    # every forward keeps its exact shape. With room, one chunk is the whole rig and this is the
    # loop it always was, unit for unit.
    _units = [(list(range(st, min(st + batch, T))), list(range(lo, min(lo + cams_flight, C))))
              for st in range(0, T, batch)
              for lo in range(0, C, cams_flight)]

    def _submit(u):
        """The next unit's decode, started BEFORE the current one's forwards are run."""
        if u >= len(_units):
            return None
        fr, cams = _units[u]
        return [pool.submit(_fetch, (ci, session.cam_names[ci], fr)) for ci in cams]

    try:
        # ONE UNIT OF LOOKAHEAD. Decode is ~100% of this loop's wall clock once the pack is fixed,
        # but the ~120 ms per batch of forward + NMS is time the decoder threads spent idle: they
        # had nothing queued until the main thread came back round. Submitting unit i+1 first
        # overlaps the two. It changes no pixels and no order -- `_submit` returns the futures in
        # camera order and the forwards still run one camera at a time, in that order.
        pending = _submit(0)
        for _u in range(len(_units)):
            frames, _ = _units[_u]
            fetched = [f.result() for f in pending]
            nxt = _submit(_u + 1)
            for ci, x, metas, src in fetched:
                # Indexed for the same reason `evaluate.py` is: the head's return arity grows
                # with each optional branch, and `detect_raw` needs only the first three.
                # THE `/255` HAPPENS HERE, ON THE DEVICE, off the uint8 `_fetch` handed back --
                # see the bit-identical note there. `float()` then `div_` in place, so the float32
                # copy exists only in device memory and only for this one camera. `_div255` is a
                # 0-d TENSOR on purpose; the scalar form is 1 ULP wrong on CUDA.
                _o = det(x.to(device).float().div_(_div255))
                obj, boxes, kpts = _o[0], _o[1], _o[2]
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
            pending = nxt
    finally:
        frame_pool.shutdown()
        pool.shutdown()
    return out, sc, kp


def associate_group(raw, session, gid, max_instances, link=False, min_views=2,
                    track=True, max_move=1.0, stats=None, pose_nms=None):
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

    `pose_nms` is the one keypoint identity lever that survived measurement -- maDLC's
    instance-level NMS by keypoint containment. Six coarse cues that spent the same signal as a
    VETO, a permutation or a Hungarian COST were built, measured on both roots and refuted; the
    ranking a K = 4 body axis supplies is too noisy to spend in any of those forms, and they are
    deleted rather than left default-off (dev/reports/19 and 21 §6/§6b).

    `stats` collects `pose_nms`' fire count: a rejection rate is what its rate-matched random
    control has to be matched TO, and it cannot be recovered afterwards from the output.
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
    tracker = None
    if track and C > 1:
        from .track import CrossViewTracker
        tracker = CrossViewTracker(S, max_res_px=session.assoc_res_max_px,
                                   min_views=min_views,
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
                           min_views=min_views)
        for a, g in enumerate(groups[:S]):
            for c, box in g['boxes'].items():
                d = g['members'][c]
                out[a, t, c] = box.numpy()
                sc[a, t, c] = float(per_cam[c][1][d])
                if kp is not None:
                    kp[a, t, c] = r_kp[per_cam[c][2][d], t, c]
    if link and tracker is None:
        out, sc = link_rows(out, sc, max_move=max_move, extra=kp)
    # LEAD 1, AFTER ASSOCIATION: drop a row that is a duplicate of another row's animal, by
    # maDLC's keypoint-containment overlap rather than by IoU. It runs here, on the finished
    # assignment, because a duplicate is a property of the SEATED rows -- `decode`'s own NMS is
    # per-box IoU before any row exists, and cannot see it.
    if pose_nms is not None and kp is not None:
        idy.pose_nms(out, kp, scores=sc, thresh=pose_nms, stats=stats)

    return out, sc, kp


def link_rows(boxes, scores=None, max_move=1.0, max_age=24, birth_age=None, extra=None):
    """Reorder instance rows frame by frame so a row follows ONE animal. In place, returns both.

    WITHOUT THIS THE ROWS ARE NOT AN ANIMAL AXIS. `decode` orders by score, so row 0 at frame t and
    row 0 at frame t+1 are unrelated. That matters because `infer.run_group` crops each window to
    the UNION of its frames' boxes, and fed unlinked rows that union is 45-59x the area of one
    animal -- the whole arena squeezed into 256 px.

    Matching is against each row's LAST KNOWN box, not against frame t-1, so a one-frame detector
    miss does not break the chain -- but that box EXPIRES after `max_age` frames, or a stale
    position makes a row permanently unavailable to the animal that actually appeared there.

    THE COST IS CENTRE DISTANCE OVER THE MEAN BOX SIDE, GATED AT ONE SIDE -- deliberately not IoU.
    IoU ranks by shape agreement, which is not identity: two touching mice overlap almost equally,
    and IoU is exactly ZERO under fast motion, where it cannot rank at all. The gate has 10-16x
    headroom over real motion (p90 centre displacement is 0.06-0.11 body lengths on every
    multi-animal root).

    AN UNMATCHED ROW STAYS EMPTY rather than taking an arbitrary leftover. A force-assigned row
    TELEPORTS across the frame, and `run_group` then crops the window to the union of those
    positions -- measured at 1924x1924 against a 244 px rat. Empty fixes it at the source.

    A detection nobody claimed may still START a row, but only one that is empty or expired --
    a birth, not a swap. Beyond that it is dropped; inventing a row on top of a live animal is
    `fp_dup`.

    THIS DROPS A THIRD OF rat-city's DETECTIONS, AND SPARE ROWS ARE THE FIX -- NOT `birth_age`.
    Relaxing eligibility (`birth_age`) buys coverage at exactly the price the strict rule exists to
    prevent: the union p99 grows from 590 px to 4367 against a 244 px rat, because a row that
    changes animal mid-window spans both. It defaults to None, off and byte-identical to the rule
    before it existed. Raising the ROW COUNT instead seats nearly everything AND tightens the union
    (p99 590 -> 525 at 12 -> 24 rows), because no row has to hold two animals.

    ponytail: still per-frame Hungarian on geometry alone. No appearance model, no velocity
    (measured as not worth it), no re-identification after a long occlusion. ONE cross-view target
    set with one affinity is the target state and deletes this function; this is the interim.
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
