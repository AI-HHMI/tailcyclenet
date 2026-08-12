"""Box training data: labels -> the crop rule's box, letterboxed into the detector's input.

The target is `tailcyclenet.crop.crop_box_for_points` applied to the same points the pose loader
crops on. That is the point of the whole detector: it reproduces the crop the pose model was
trained on, so swapping a GT crop for a detector crop costs a fraction of a millimetre instead
of whatever an independently-plausible box rule would cost.

An animal with no finite point in a view gets a **NaN box**, not a dropped frame. Objectness
still has to learn "no animal here" for that view, and silently dropping those frames would
train a detector that has never seen an empty image.

`box_source='instances'` takes the extent from `instances.pq` instead, for a dataset whose stored
keypoints are too sparse to bound the animal: rat-city's converter dropped noisy points, leaving
25,777 train instances with NO finite point at all -- each one a NaN box teaching "no animal here"
where a rat plainly is. It is opt-in rather than "use the table when it exists" because an
`instances.pq` box is not a crop box in general (spec §9 says so outright; johnson-mouse ships COCO
boxes and calms21 ships MARS ones), and defaulting to it would silently retarget those detectors.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from posetail.posetail.cube import project_points_torch

from ..crop import BOX_SOURCES, crop_box_for_points
from ..dataset import read_frames
from ..format import UNLABELED, load_datasets


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


def letterbox_transform(size, out_wh):
    """The (scale, (padx, pady)) `letterbox` would return for an image of `size` = (w, h).

    Same arithmetic, no pixels: the box target and the assignment diagnostics need the transform
    without paying rat-city's 39 ms JPEG decode to learn it.
    """
    w, h = float(size[0]), float(size[1])
    s = min(out_wh[0] / w, out_wh[1] / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    return s, ((out_wh[0] - nw) // 2, (out_wh[1] - nh) // 2)


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
                 max_frames_per_group: int = 40, seed: int = 0, box_source='keypoints'):
        assert box_source in BOX_SOURCES, \
            f'box_source must be one of {BOX_SOURCES}, got {box_source!r}'
        self.box_source = box_source
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
                frames = np.flatnonzero((v != UNLABELED).any((0, 2)))
                if frames.size > max_frames_per_group:
                    frames = rng.choice(frames, max_frames_per_group, replace=False)
                for f in sorted(frames):
                    for ci in range(len(sess.rig)):
                        self.index.append((sess, gid, int(f), ci))
        if not self.index:
            raise ValueError(f'{path}: split {split!r} has no labelled frames')
        # ONE CONTAINER'S WORTH OF INDEX POSITIONS, which is what `ChunkShuffle` needs a block to
        # be. Its old hardcoded 512 was set against rat-city, where the whole split is one group;
        # on calms21 a session contributes `max_frames_per_group` = 40 positions, so a 512-block
        # spanned 13 videos and a 4-block pool spanned 52. The reader cache then held 8 of 52 and
        # ran at a 16% hit rate -- and at the cache size that thrash used to need, ~1 GB of open
        # decord reader each, it OOM-killed the workers.
        n_src = len({(s.session_id, g, c) for s, g, _, c in self.index})   # (group, camera) = file
        self.chunk = max(1, len(self.index) // n_src)

    def __len__(self):
        return len(self.index)

    def boxes_for(self, i):
        """The letterboxed target boxes for item `i`, without decoding its image.

        Split out of `__getitem__` so `scripts/diag_assign.py` can read what the loss is actually
        assigned over a few hundred views without paying for the pixels -- and so there is one
        copy of the box rule rather than a second one in the diagnostic that can drift from it.
        """
        sess, gid, f, ci = self.index[i]
        lab = sess.labels(gid)
        # Frame-indexed, not whole-group: `pts` below is (S,K,3) whose axis -3 is the ANIMAL, so a
        # moving camera's (T,4,4) extrinsic would project animal `i` through frame `i`'s pose.
        cam = sess.cgroup(gid, f)[ci]

        if sess.mode == '3d':
            pts = torch.as_tensor(lab.points3d[:, f], dtype=torch.float32)      # (S,K,3)
            p2d = project_points_torch([cam], pts)[0]                            # (S,K,2)
        else:
            p2d = torch.as_tensor(lab.points2d[:, f, :, ci], dtype=torch.float32)

        boxes = []
        for s in range(p2d.shape[0]):
            # A stored box is an ALREADY-PADDED extent, so it re-enters the rule at pad 0 and
            # picks up only the min_crop_dim floor and the clamp. Per animal, not per session:
            # rat-city's boxes come from a tracker that loses animals, and where it did the
            # keypoints are still the best -- and only -- source left.
            src, pad = p2d[s], 20
            if self.box_source == 'instances' and lab.boxes is not None:
                b = torch.as_tensor(lab.boxes[s, f, ci], dtype=torch.float32)
                if torch.isfinite(b).all():
                    src, pad = b.view(2, 2), 0
            box = crop_box_for_points(src, cam['size'], self.min_crop_dim, pad)
            boxes.append(torch.full((4,), float('nan')) if box is None else box.float())
        boxes = torch.stack(boxes)
        scale, pad = letterbox_transform(cam['size'], self.input_wh)
        boxes[:, 0::2] = boxes[:, 0::2] * scale + pad[0]
        boxes[:, 1::2] = boxes[:, 1::2] * scale + pad[1]
        return boxes

    def __getitem__(self, i):
        sess, gid, f, ci = self.index[i]
        boxes = self.boxes_for(i)
        img = read_frames(sess.groups[gid], sess.cam_names[ci], [f])[0]
        if img is None:
            raise RuntimeError(f'{gid}/{sess.cam_names[ci]}: frame {f} unreadable')
        # The box transform is derived from the rig's recorded size, so a frame that disagrees
        # with it would put the boxes in a different letterbox than the pixels, silently.
        assert (img.shape[1], img.shape[0]) == tuple(sess.rig.size(sess.cam_names[ci])), \
            f'{gid}/{sess.cam_names[ci]} frame {f}: image is {img.shape[1]}x{img.shape[0]}, rig '\
            f'says {sess.rig.size(sess.cam_names[ci])}'
        img, _, _ = letterbox(img, self.input_wh)
        x = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return x, boxes


class ChunkShuffle(torch.utils.data.Sampler):
    """A shuffle that keeps a worker inside a few videos at a time.

    `BoxDataset.index` is built session-by-session, so a run of contiguous index positions stays
    inside one video. A plain `shuffle=True` therefore sends every item to a different container:
    on calms21's 63 one-mp4 sessions that thrashes `dataset._reader`'s cache and costs 486 ms per
    batch of 16, against a 16 ms GPU step. Shuffling BLOCKS instead, and pooling `mix` of them so
    a batch still spans several sessions, costs 40 ms -- 12x faster, and 7 sessions per batch so
    BatchNorm does not end up normalising one animal's lighting.

    PASS `chunk=dataset.chunk`. The default 512 was set against rat-city, whose whole split is one
    group; a dataset with 40 index positions per video gets 13 videos to a block and 52 to a pool,
    which is not locality at all. `BoxDataset` derives the right value from its own index.

    Image-directory datasets do not need this and are not harmed by it: their frames are separate
    files, so locality buys nothing and costs nothing.
    """

    def __init__(self, n, chunk=512, mix=4, seed=0):
        self.n, self.chunk, self.mix, self.seed = n, chunk, mix, seed
        self.epoch = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        starts = np.arange(0, self.n, self.chunk)
        rng.shuffle(starts)
        for i in range(0, len(starts), self.mix):
            pool = np.concatenate([np.arange(s, min(s + self.chunk, self.n))
                                   for s in starts[i:i + self.mix]])
            rng.shuffle(pool)
            yield from (int(j) for j in pool)


def box_collate(batch):
    """Pad the box axis with NaN so a batch can hold different animal counts."""
    xs = torch.stack([b[0] for b in batch])
    n = max(b[1].shape[0] for b in batch)
    boxes = torch.full((len(batch), n, 4), float('nan'))
    for i, (_, b) in enumerate(batch):
        boxes[i, :b.shape[0]] = b
    return xs, boxes
