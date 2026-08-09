"""Box training data: labels -> the crop rule's box, letterboxed into the detector's input.

The target is `tailcyclenet.crop.crop_box_for_points` applied to the same points the pose loader
crops on. That is the point of the whole detector: it reproduces the crop the pose model was
trained on, so swapping a GT crop for a detector crop costs a fraction of a millimetre instead
of whatever an independently-plausible box rule would cost.

An animal with no finite point in a view gets a **NaN box**, not a dropped frame. Objectness
still has to learn "no animal here" for that view, and silently dropping those frames would
train a detector that has never seen an empty image.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.posetail.cube import project_points_torch

from ..crop import crop_box_for_points
from ..dataset import read_frames
from ..format import load_datasets


def letterbox(img, out_wh):
    """Resize preserving aspect ratio, pad with grey. Returns (img, scale, (padx, pady))."""
    import cv2
    H, W = img.shape[:2]
    ow, oh = out_wh
    s = min(ow / W, oh / H)
    nw, nh = int(round(W * s)), int(round(H * s))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((oh, ow, 3), 114, np.uint8)
    px, py = (ow - nw) // 2, (oh - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, s, (px, py)


def unletterbox_boxes(boxes, scale, pad):
    """Detector-input boxes -> source-image boxes."""
    out = boxes.clone().float()
    out[:, 0::2] = (out[:, 0::2] - pad[0]) / scale
    out[:, 1::2] = (out[:, 1::2] - pad[1]) / scale
    return out


class BoxDataset(Dataset):
    """One item = one camera view of one frame, with every animal's crop box in it.

    Deliberately per-view and per-frame rather than per-window: a box predictor has no temporal
    model, and giving it 24 near-identical frames as one item would just correlate the batch.
    """

    def __init__(self, path, split: str, input_wh=(416, 416), min_crop_dim=64,
                 max_frames_per_group: int = 40, seed: int = 0):
        self.datasets = load_datasets(path)
        if len(self.datasets) != 1:
            raise ValueError(
                f'{path}: the detector is trained per dataset (input size and box statistics are '
                f'dataset-specific); found {len(self.datasets)} dataset roots')
        self.ds = self.datasets[0]
        self.input_wh = tuple(input_wh)
        self.min_crop_dim = min_crop_dim
        self.train = split == 'train'
        rng = np.random.default_rng(seed)

        self.index = []
        for sess in self.ds.sessions.get(split, []):
            sess.preload()
            for gid, group in sess.groups.items():
                lab = sess.labels(gid)
                vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
                if vis is None:
                    continue
                v = vis.reshape(vis.shape[0], vis.shape[1], -1)
                frames = np.flatnonzero((v != -1).any((0, 2)))
                if frames.size > max_frames_per_group:
                    frames = rng.choice(frames, max_frames_per_group, replace=False)
                for f in sorted(frames):
                    for ci in range(len(sess.rig)):
                        self.index.append((sess, gid, int(f), ci))
        if not self.index:
            raise ValueError(f'{path}: split {split!r} has no labelled frames')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sess, gid, f, ci = self.index[i]
        group = sess.groups[gid]
        lab = sess.labels(gid)
        cam = sess.rig.posetail()[ci]

        if sess.mode == '3d':
            pts = torch.as_tensor(lab.points3d[:, f], dtype=torch.float32)      # (S,K,3)
            p2d = project_points_torch([cam], pts)[0]                            # (S,K,2)
        else:
            p2d = torch.as_tensor(lab.points2d[:, f, :, ci], dtype=torch.float32)

        boxes = []
        for s in range(p2d.shape[0]):
            box = crop_box_for_points(p2d[s], cam['size'], self.min_crop_dim)
            boxes.append(torch.full((4,), float('nan')) if box is None else box.float())
        boxes = torch.stack(boxes)

        img = read_frames(group, sess.cam_names[ci], [f])[0]
        if img is None:
            raise RuntimeError(f'{gid}/{sess.cam_names[ci]}: frame {f} unreadable')
        img, scale, pad = letterbox(img, self.input_wh)
        boxes[:, 0::2] = boxes[:, 0::2] * scale + pad[0]
        boxes[:, 1::2] = boxes[:, 1::2] * scale + pad[1]

        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return x, boxes


def box_collate(batch):
    """Pad the box axis with NaN so a batch can hold different animal counts."""
    xs = torch.stack([b[0] for b in batch])
    n = max(b[1].shape[0] for b in batch)
    boxes = torch.full((len(batch), n, 4), float('nan'))
    for i, (_, b) in enumerate(batch):
        boxes[i, :b.shape[0]] = b
    return xs, boxes
