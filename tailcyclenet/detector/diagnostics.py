"""Label-only detector diagnostics: which frames are labelled, and the crop-rule box sides they imply."""
from __future__ import annotations

import numpy as np
import torch


def labelled_frames(sess, gid):
    """(F,) int frame indices with any labelled keypoint in this group, or empty."""
    from ..format import UNLABELED
    lab = sess.labels(gid)
    vis = lab.vis3d if lab.vis3d is not None else lab.vis2d
    if vis is None:
        return np.zeros(0, int)
    v = vis.reshape(vis.shape[0], vis.shape[1], -1)
    return np.flatnonzero((v != UNLABELED).any((0, 2)))


def gt_crop_sides(sess, gid, min_crop_dim, max_frames=0, cap=200):
    """Crop-rule box side (SOURCE px) for every labelled (animal, frame[, camera]), a POPULATION
    -- not paired to any detector row. `deployment_score` compares this against the union-box
    side distribution the box path actually produces, unpaired; pairing a detector row to a GT
    identity needs a matcher (MOTA's), and this is meant to run before spending a GPU-hour on
    either half of the box path, not to replace `box_mota`.

    2D: `points2d[:, f, :, ci]` directly. 3D: `points3d[:, f]` projected through frame `f`'s own
    camera group, the same branch `BoxDataset._points_2d` takes (mirrored here rather than
    imported, because building a whole `BoxDataset` -- augmentation RNG, tiling, the multi-root
    refusal -- to read one projection is the wrong tool for a label-only pass).

    `cap` bounds the frames sampled per group (evenly spaced), so a 20,000-frame clip costs the
    same as a 500-frame one -- this is a population read, not a census.
    """
    from ..data.crop import crop_box_for_points
    from tracktail.model.cube import project_points_torch
    frames = labelled_frames(sess, gid)
    if max_frames:
        frames = frames[frames < max_frames]
    if frames.size > cap:
        frames = frames[np.linspace(0, frames.size - 1, cap).astype(int)]
    lab = sess.labels(gid)
    sides = []
    for f in frames.tolist():
        if sess.mode == '3d':
            cams = sess.cgroup(gid, int(f))
            pts3d = torch.as_tensor(lab.points3d[:, f], dtype=torch.float32)
            for ci, cam in enumerate(cams):
                p2d = project_points_torch([cam], pts3d)[0]
                w, h = sess.rig.size(sess.cam_names[ci])
                for s in range(p2d.shape[0]):
                    box = crop_box_for_points(p2d[s], torch.tensor([w, h]), min_crop_dim)
                    if box is not None:
                        sides.append(float(0.5 * ((box[2] - box[0]) + (box[3] - box[1]))))
        else:
            for ci, cam_name in enumerate(sess.cam_names):
                w, h = sess.rig.size(cam_name)
                p2d = torch.as_tensor(lab.points2d[:, f, :, ci], dtype=torch.float32)
                for s in range(p2d.shape[0]):
                    box = crop_box_for_points(p2d[s], torch.tensor([w, h]), min_crop_dim)
                    if box is not None:
                        sides.append(float(0.5 * ((box[2] - box[0]) + (box[3] - box[1]))))
    return np.array(sides)
