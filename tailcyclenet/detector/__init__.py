import torch

from .assign import assign, box_iou, decode, detector_loss, giou_loss
from .associate import associate
from .data import (BoxDataset, ChunkShuffle, box_collate, letterbox, letterbox_transform,
                   reduce_factor, unletterbox_boxes)
from .yolox import YOLOXNano

__all__ = ['YOLOXNano', 'BoxDataset', 'ChunkShuffle', 'box_collate', 'letterbox',
           'letterbox_transform', 'reduce_factor', 'unletterbox_boxes', 'assign', 'box_iou',
           'decode', 'detector_loss', 'giou_loss', 'associate', 'LINK_REV']

# BUMP THIS WHENEVER `link_rows` CHANGES WHAT IT EMITS. `--det-cache` stores boxes that have already
# been linked, so a cache written under an older rule is a different box set under an identical
# stamp -- exactly the silent mismatch the stamp exists to catch (`scripts/infer.py`). Rev 2 is the
# gated centre-distance matcher with births and expiry; rev 1 was ungated IoU with `free.pop(0)`.
LINK_REV = 2


def load_detector(path, device='cpu', input_wh=None):
    """(model, input_wh, dataset_name, min_crop_dim, reduce, box_source) from a folder or a .pth.

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
    model = YOLOXNano()
    model.load_state_dict(ckpt['model_state'])
    return (model.to(device).eval(), tuple(wh), str(ckpt.get('dataset', '')),
            int(ckpt.get('min_crop_dim', 64)), bool(ckpt.get('reduce', False)),
            str(ckpt.get('box_source', 'keypoints')))


@torch.no_grad()
def detect_group(det, input_wh, session, gid, max_instances, device='cpu', batch=16,
                 score_thresh=0.99, link=False, reduce=False, max_frames=0, min_views=2,
                 dup_res_px=None, track=True, max_move=1.0):
    """Run the detector over every frame and camera of a group -> (boxes, scores).

    boxes (S,T,C,4), scores (S,T,C). The score is the objectness the box survived NMS on, and it
    is returned rather than dropped because `--det-score` is otherwise a re-detection per
    threshold: detection is the expensive half of a run, and a sweep over a threshold that only
    ever *removes* boxes can be done offline from what one pass already computed.

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
    from concurrent.futures import ThreadPoolExecutor
    from ..dataset import read_frames
    from .data import letterbox, reduce_factor, unletterbox_boxes

    group = session.groups[gid]
    T = min(group.n_frames, max_frames or group.n_frames)
    C = len(session.rig)
    S = max_instances or 1
    out = np.full((S, T, C, 4), np.nan, np.float32)
    sc = np.full((S, T, C), np.nan, np.float32)
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

    # ONE THREAD PER CAMERA FOR THE DECODE, and it is where this function's wall clock lives:
    # profiled on 3dpop (four 3840x2160 cameras) the decode is 35-48 ms per frame-camera against a
    # 0.4 ms detector forward, 100x. decord releases the GIL inside `get_batch` and the four
    # containers share no state, so they overlap 3.5x -- but only since `dataset._read_video` took
    # a lock PER PATH instead of one global one. Sized to the rig, because that is how many
    # independent containers there are; more threads would contend on the same reader.
    # THE FORWARD STAYS SERIAL AND IN CAMERA ORDER: it is 1% of the time and moving it would put
    # two streams on one CUDA context for nothing.
    def _fetch(ci_cam):
        ci, cam_name = ci_cam
        # THE SAME DECODE THE DETECTOR WAS TRAINED ON. `BoxDataset` reduces at decode where
        # the frame is far above the letterbox target, and a detector fed differently-sampled
        # pixels at deployment is being run off its own training distribution -- silently,
        # since nothing about the shapes or the scores would say so.
        src = session.rig.size(cam_name)
        r = reduce_factor(src, input_wh) if reduce else 1
        imgs = read_frames(group, cam_name, frames, reduce=r)
        lbs, metas = [], []
        for im in imgs:
            lb, scale, pad = letterbox(im, input_wh, src_wh=src)
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

    pool = ThreadPoolExecutor(max_workers=max(1, C)) if C > 1 else None
    try:
        for start in range(0, T, batch):
            frames = list(range(start, min(start + batch, T)))
            todo = list(enumerate(session.cam_names))
            fetched = list(map(_fetch, todo) if pool is None else pool.map(_fetch, todo))
            per_cam = []
            for ci, x, metas, src in fetched:
                obj, boxes = det(x.to(device))
                cam_frames = []
                for j in range(len(frames)):
                    b, s = decode(obj[j], boxes[j], top_k=S, score_thresh=score_thresh)
                    cam_frames.append((unletterbox_boxes(b.cpu(), *metas[j], src_wh=src)
                                       if b.numel() else b.cpu(), s.cpu()))
                per_cam.append(cam_frames)

            for j, t in enumerate(frames):
                if C == 1:
                    b, s = per_cam[0][j]
                    for a in range(min(S, b.shape[0])):
                        out[a, t, 0] = b[a].numpy()
                        sc[a, t, 0] = float(s[a])
                else:
                    cams = session.cgroup(gid, t) if moving else cgroup
                    if tracker is not None:
                        out[:, t], sc[:, t] = tracker.step(
                            cams, [per_cam[c][j][0] for c in range(C)],
                            [per_cam[c][j][1] for c in range(C)])
                        continue
                    groups = associate(cams, [per_cam[c][j][0] for c in range(C)],
                                       max_res_px=session.assoc_res_max_px, max_instances=S,
                                       min_views=min_views, dup_res_px=dup_res_px)
                    for a, g in enumerate(groups[:S]):
                        for c, box in g['boxes'].items():
                            out[a, t, c] = box.numpy()
                            sc[a, t, c] = float(per_cam[c][j][1][g['members'][c]])
    finally:
        if pool is not None:
            pool.shutdown()
    return link_rows(out, sc, max_move=max_move) if (link and tracker is None) else (out, sc)


def link_rows(boxes, scores=None, max_move=1.0, max_age=24):
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
        open_rows = [r for r in range(S)
                     if r not in taken and not np.isfinite(last[r]).all(-1).any()]
        for r, c in zip(open_rows, free_dets):
            taken[r] = c
        out = np.full_like(cur, np.nan)
        sc = None if scores is None else np.full_like(scores[:, t], np.nan)
        for r, c in taken.items():
            out[r] = cur[c]
            if sc is not None:
                # The SAME assignment, or the score stops describing the box beside it.
                sc[r] = scores[:, t][c]
        boxes[:, t] = out
        if sc is not None:
            scores[:, t] = sc
        seen = np.isfinite(boxes[:, t]).all(-1)
        last = np.where(seen[..., None], boxes[:, t], last)
        age = np.where(seen.any(-1), 0, age + 1)
        # EXPIRY IS A FORGET, not just a flag: a stale centre that stays in the cost matrix keeps
        # competing for the detection that belongs to whoever is there now.
        last[age > max_age] = np.nan
    return boxes if scores is None else (boxes, scores)
