"""Training augmentation for the pose loader: the per-item geometry draws (camera count, crop
inflate, the 3D camera-group rotation carrying a neighbour) and the appearance pipelines (per-camera
colour replayed down the clip, per-frame sensor noise, cutout, grayscale, invert).
"""
from __future__ import annotations

import numpy as np
import torch


def rotate_camera_group_with_neighbours(cgroup, coords, others):
    """`TracktailDataset.rotate_camera_group`, optionally carrying a SECOND point set through the
    identical draw. Returns `(cgroup, coords, others)`; `others` is None in, None out.

    With `others` None (the default) this is EXACTLY the library call. With it, `others` is one
    neighbour's (T,K,R) pose and rides the same `rotmat` via a temporary concatenation on the
    keypoint axis, which changes nothing about `coords`' own output. Callers must NOT pass the
    combined tensor to `crop_to_points_3d`: the crop is the TARGET's extent, and a neighbour
    would change the item's geometry.
    """
    from tracktail.datasets.tracktail_dataset import TracktailDataset

    if others is None:
        cgroup, coords = TracktailDataset.rotate_camera_group(None, cgroup, coords)
        return cgroup, coords, None
    K = coords.shape[1]
    combined = torch.cat([coords, others], dim=1)
    cgroup, combined = TracktailDataset.rotate_camera_group(None, cgroup, combined)
    return cgroup, combined[:, :K], combined[:, K:]


def crop_inflate(cfg, rng, train):
    """THIS item's crop-inflate factor. `cfg.crop_inflate` is a float, or a `[low, high]` pair --
    same contract as `n_cams`/`cams_to_sample`, one continuous draw instead of an integer one.

    TRAIN draws uniformly in `[low, high]` per item, so a wide-crop run shows the model a RANGE of
    context rather than one fixed geometry every step. VAL/TEST NEVER DRAW -- they get the range's
    MIDPOINT, for the same reason `_jitter` is gated to `self.train` alone: `checkpoint_best` is
    selected on the val curve, and a val geometry that moved step to step would put noise from the
    crop draw into a number that is supposed to isolate the weights. A scalar returns unchanged.
    """
    spec = cfg.crop_inflate
    if not isinstance(spec, (list, tuple)):
        return float(spec)
    lo, hi = float(spec[0]), float(spec[1])
    return float(rng.uniform(lo, hi)) if train else (lo + hi) / 2.0


def n_cams(spec, rng):
    """How many cameras this item shows. `spec` is an int, or a [low, high] pair drawn per item.

    The pair form is tracktail's (`TracktailDataset.sample_cameras`)
    and it is what the pretrained tracker was finetuned with, at [1, 8]. Returning a number larger
    than the session has is fine -- the caller takes every camera in that case, exactly as the
    reference's `if len(cam_names) > num_cams_to_sample` guard does.
    """
    if isinstance(spec, (list, tuple)):
        lo, hi = int(spec[0]), int(spec[1])
        return int(rng.integers(lo, hi + 1))
    return int(spec)


def build_augmenters(cfg):
    """The two appearance pipelines, taken from the reference (`TracktailDataset`).

    The SPLIT is the point, not the list. `per_camera` is sampled once per camera and replayed
    frame by frame, so a camera's colour, gamma and focus hold steady down a clip: appearance is
    an identity cue for a tracker, and re-rolling hue every frame teaches that it is noise.
    `per_image` is resampled per frame, which is what sensor noise and motion blur actually are.
    Watch `train/loader_wait_frac` before blaming augmentation for loader cost.
    """
    import imgaug.augmenters as iaa

    p, q = cfg.aug_prob, cfg.per_image_aug_prob
    per_camera = iaa.Sequential([
        iaa.Sometimes(p, iaa.GammaContrast((0.6, 1.8))),
        iaa.Sometimes(p, iaa.AddToSaturation((-50, 30))),
        iaa.Sometimes(p, iaa.AddToHue((-10, 10))),
    ])
    per_image = iaa.Sequential([
        iaa.Sometimes(q, iaa.MotionBlur(k=(3, 5))),
        iaa.Sometimes(q, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
        iaa.Sometimes(q, iaa.Multiply((0.9, 1.1))),
    ])
    return per_camera, per_image


def cutout_rects(rng, size, p2d, vis_2d, cnum):
    """Random-erasing rectangles for one camera, in crop pixels. Mutates `vis_2d` in place.

    A keypoint underneath a rect is no longer visible, and saying so is the whole point: without
    it the model is asked to report "visible" for a patch that has been painted over, which is the
    one label that is definitely wrong.

    `vis_2d` here is THREE-state -- NaN means "no one assessed this camera". Cutout overwrites
    NaN with 0, and that is right rather than an invention: the pixels are now literally covered,
    so "not visible" became a fact about the image we just produced.
    """
    w, h = int(size[0]), int(size[1])
    rects = []
    for _ in range(int(rng.integers(1, 4))):
        rw, rh = int(w * 0.15), int(h * 0.15)
        rx = int(rng.integers(0, max(w - rw, 1)))
        ry = int(rng.integers(0, max(h - rh, 1)))
        rects.append((rx, ry, rx + rw, ry + rh, rng.integers(0, 256, 3).tolist()))
        if vis_2d is not None:
            pts = p2d[cnum]
            inside = ((pts[..., 0] >= rx) & (pts[..., 0] <= rx + rw) &
                      (pts[..., 1] >= ry) & (pts[..., 1] <= ry + rh))
            vis_2d[:, :, cnum][inside] = 0
    return rects


def augment_views(aug, cfg, imgs, cnum, size, p2d, vis_2d, gray, rng, invert=False):
    """Appearance augmentation for one camera's T crops. `vis_2d` is mutated by cutout.

    `to_deterministic()` freezes this camera's sampled parameters so every frame gets the
    SAME gamma/hue; only the apply is per-frame. `invert` is one item-level color-inversion
    decision, shared by every frame and camera in the item.
    """
    import cv2

    per_camera, per_image = aug
    cam_det = per_camera.to_deterministic()
    imgs = [per_image(image=cam_det(image=im)) for im in imgs]
    if rng.random() < cfg.aug_prob:
        for x1, y1, x2, y2, fill in cutout_rects(rng, size, p2d, vis_2d, cnum):
            for im in imgs:
                im[y1:y2, x1:x2] = fill
    if gray:
        imgs = [np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)] * 3, -1) for im in imgs]
    if invert:
        imgs = [255 - im for im in imgs]
    return imgs
