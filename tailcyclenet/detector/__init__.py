import torch

from .assign import assign, box_iou, decode, detector_loss, giou_loss
from .associate import associate
from .data import BoxDataset, box_collate, letterbox, unletterbox_boxes
from .yolox import YOLOXNano

__all__ = ['YOLOXNano', 'BoxDataset', 'box_collate', 'letterbox', 'unletterbox_boxes',
           'assign', 'box_iou', 'decode', 'detector_loss', 'giou_loss', 'associate']


def load_detector(path, device='cpu'):
    """(model, input_wh, dataset_name) from a detector run folder or a .pth."""
    import torch
    from pathlib import Path
    p = Path(path)
    if p.is_dir():
        p = p / 'detector.pth'
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    model = YOLOXNano()
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval(), tuple(ckpt['input_wh']), str(ckpt.get('dataset', ''))


@torch.no_grad()
def detect_group(det, input_wh, session, gid, max_instances, device='cpu', batch=16,
                 score_thresh=0.05):
    """Run the detector over every frame and camera of a group -> boxes (S,T,C,4).

    2D / single camera: instances are the NMS survivors, ordered by score, and the row index is
    the only identity there is -- it is NOT tracked, so row `a` at frame t and frame t+1 need not
    be the same animal. Feeding these straight to the pose model is the honest deployment
    baseline for a single window and nothing more; a tracker belongs on top.

    3D multiview: rows come from cross-view association, so a row IS one physical animal within
    a frame, again untracked across frames.
    """
    import numpy as np
    import torch
    from ..dataset import read_frames
    from .data import letterbox, unletterbox_boxes

    group = session.groups[gid]
    T, C = group.n_frames, len(session.rig)
    S = max_instances or 1
    out = np.full((S, T, C, 4), np.nan, np.float32)
    # Static rig: build once. Moving rig: `associate` triangulates per frame from (n,2) centres,
    # so it needs that frame's own (4,4) extrinsic -- built inside the loop below.
    moving = any(session.rig.moving.values())
    cgroup = None if moving else session.cgroup(gid)

    for start in range(0, T, batch):
        frames = list(range(start, min(start + batch, T)))
        per_cam = []
        for ci, cam_name in enumerate(session.cam_names):
            imgs = read_frames(group, cam_name, frames)
            packed, metas = [], []
            for im in imgs:
                lb, scale, pad = letterbox(im, input_wh)
                packed.append(torch.as_tensor(lb, dtype=torch.float32).permute(2, 0, 1) / 255.0)
                metas.append((scale, pad))
            x = torch.stack(packed).to(device)
            obj, boxes = det(x)
            cam_frames = []
            for j in range(len(frames)):
                b, s = decode(obj[j], boxes[j], top_k=S, score_thresh=score_thresh)
                cam_frames.append(unletterbox_boxes(b.cpu(), *metas[j]) if b.numel()
                                  else b.cpu())
            per_cam.append(cam_frames)

        for j, t in enumerate(frames):
            if C == 1:
                b = per_cam[0][j]
                for a in range(min(S, b.shape[0])):
                    out[a, t, 0] = b[a].numpy()
            else:
                cams = session.cgroup(gid, t) if moving else cgroup
                groups = associate(cams, [per_cam[c][j] for c in range(C)],
                                   max_res_px=session.assoc_res_max_px, max_instances=S)
                for a, g in enumerate(groups[:S]):
                    for c, box in g['boxes'].items():
                        out[a, t, c] = box.numpy()
    return out
