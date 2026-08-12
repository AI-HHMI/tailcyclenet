import torch

from .assign import assign, box_iou, decode, detector_loss, giou_loss
from .associate import associate
from .data import (BoxDataset, ChunkShuffle, box_collate, letterbox, letterbox_transform,
                   reduce_factor, unletterbox_boxes)
from .yolox import YOLOXNano

__all__ = ['YOLOXNano', 'BoxDataset', 'ChunkShuffle', 'box_collate', 'letterbox',
           'letterbox_transform', 'reduce_factor', 'unletterbox_boxes', 'assign', 'box_iou',
           'decode', 'detector_loss', 'giou_loss', 'associate']


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
                 score_thresh=0.05, link=False, reduce=False, max_frames=0, min_views=2):
    """Run the detector over every frame and camera of a group -> (boxes, scores).

    boxes (S,T,C,4), scores (S,T,C). The score is the objectness the box survived NMS on, and it
    is returned rather than dropped because `--det-score` is otherwise a re-detection per
    threshold: detection is the expensive half of a run, and a sweep over a threshold that only
    ever *removes* boxes can be done offline from what one pass already computed.

    `max_frames` is the same PREFIX `infer.run_group` takes, and it has to be honoured here or the
    two disagree about the clip: rat-city's one test group is 57,594 frames and the protocol is its
    first 480, so detecting the whole group threw away 99.2% of the detection -- which is the
    expensive half of a run.

    2D / single camera: instances are the NMS survivors, ordered by score, and the row index is
    the only identity there is -- it is NOT tracked, so row `a` at frame t and frame t+1 need not
    be the same animal. Feeding these straight to the pose model is the honest deployment
    baseline for a single window and nothing more; a tracker belongs on top.

    `link=True` puts the smallest possible one there -- see `link_rows`. Off by default so the
    contract in the paragraph above stays the contract.

    3D multiview: rows come from cross-view association, so a row IS one physical animal within
    a frame, again untracked across frames.
    """
    import numpy as np
    import torch
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

    for start in range(0, T, batch):
        frames = list(range(start, min(start + batch, T)))
        per_cam = []
        for ci, cam_name in enumerate(session.cam_names):
            # THE SAME DECODE THE DETECTOR WAS TRAINED ON. `BoxDataset` reduces at decode where
            # the frame is far above the letterbox target, and a detector fed differently-sampled
            # pixels at deployment is being run off its own training distribution -- silently,
            # since nothing about the shapes or the scores would say so.
            src = session.rig.size(cam_name)
            r = reduce_factor(src, input_wh) if reduce else 1
            imgs = read_frames(group, cam_name, frames, reduce=r)
            packed, metas = [], []
            for im in imgs:
                lb, scale, pad = letterbox(im, input_wh, src_wh=src)
                packed.append(torch.as_tensor(lb, dtype=torch.float32).permute(2, 0, 1) / 255.0)
                metas.append((scale, pad))
            x = torch.stack(packed).to(device)
            obj, boxes = det(x)
            cam_frames = []
            for j in range(len(frames)):
                b, s = decode(obj[j], boxes[j], top_k=S, score_thresh=score_thresh)
                cam_frames.append((unletterbox_boxes(b.cpu(), *metas[j]) if b.numel()
                                   else b.cpu(), s.cpu()))
            per_cam.append(cam_frames)

        for j, t in enumerate(frames):
            if C == 1:
                b, s = per_cam[0][j]
                for a in range(min(S, b.shape[0])):
                    out[a, t, 0] = b[a].numpy()
                    sc[a, t, 0] = float(s[a])
            else:
                cams = session.cgroup(gid, t) if moving else cgroup
                groups = associate(cams, [per_cam[c][j][0] for c in range(C)],
                                   max_res_px=session.assoc_res_max_px, max_instances=S,
                                   min_views=min_views)
                for a, g in enumerate(groups[:S]):
                    for c, box in g['boxes'].items():
                        out[a, t, c] = box.numpy()
                        sc[a, t, c] = float(per_cam[c][j][1][g['members'][c]])
    return link_rows(out, sc) if link else (out, sc)


def link_rows(boxes, scores=None):
    """Reorder instance rows frame by frame so a row follows ONE animal. In place, returns both.

    WITHOUT THIS THE ROWS ARE NOT AN ANIMAL AXIS. `decode` orders by score, so row 0 at frame t
    and row 0 at frame t+1 are unrelated -- measured on branson-fly, the median IoU between a
    row's own consecutive boxes is 0.000 across ten near-identical flies. That matters because
    `infer.run_group` crops each window to the UNION of its frames' boxes, to stop an animal
    walking out of its own crop: fed unlinked rows, that union is 45x (branson-fly) / 59x
    (rat-city) the area of one animal and the pose model receives the whole arena squeezed into
    256 px.

    Matching is on IoU against each row's LAST KNOWN box, not against frame t-1, so a one-frame
    detector miss does not break the chain.

    ponytail: greedy per-frame Hungarian on IoU, nothing else. No births, no deaths, no
    re-identification after a long occlusion, no appearance model -- a row that drifts onto
    another animal stays there. posetail-pose needed six more flags on top of this
    (`--track-bridge`, `--rebirth`, `--birth-score`, `--dup-action`, `--pose-arbitrate`,
    `--max-age-frames`) to reach MOTA 0.69 on rat-city, and `reports/crowding.md` is the record of
    why. Upgrade path is that report, not another heuristic here.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    from .assign import box_iou as _iou

    S, T, C, _ = boxes.shape
    last = boxes[:, 0].copy()                     # (S,C,4), each row's most recent known box
    for t in range(1, T):
        cur = boxes[:, t]
        cost = np.zeros((S, S), np.float32)
        for c in range(C):
            ok_p = np.isfinite(last[:, c]).all(-1)
            ok_c = np.isfinite(cur[:, c]).all(-1)
            if not (ok_p.any() and ok_c.any()):
                continue
            iou = _iou(torch.as_tensor(last[ok_p, c]), torch.as_tensor(cur[ok_c, c])).numpy()
            cost[np.ix_(ok_p, ok_c)] += iou
        rows, cols = linear_sum_assignment(-cost)
        perm = np.arange(S)
        # An unmatched row keeps its own slot; only positive-overlap pairs are moved, or two
        # animals that never touch would be swapped by an all-zero cost matrix's arbitrary
        # optimum.
        taken = {}
        for r, c_ in zip(rows, cols):
            if cost[r, c_] > 0:
                taken[r] = c_
        free = [i for i in range(S) if i not in taken.values()]
        for r in range(S):
            perm[r] = taken[r] if r in taken else (free.pop(0) if free else r)
        boxes[:, t] = cur[perm]
        if scores is not None:
            # The SAME permutation, or the score stops describing the box beside it.
            scores[:, t] = scores[:, t][perm]
        seen = np.isfinite(boxes[:, t]).all(-1)
        last = np.where(seen[..., None], boxes[:, t], last)
    return boxes if scores is None else (boxes, scores)
