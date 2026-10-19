"""Pixels for one camera of one group: `read_frames` (image directory or video), `load_warps`, and
`crop_affine`, the one dst<-src affine every crop/rotate/resize in the repo renders through.

Video goes through `reader_cache` (one decode per distinct index, under a per-container lock);
image directories are decoded with OpenCV. Each DISTINCT frame index is decoded once and a repeat
gets a COPY, because `_augment`'s cutout writes in place.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import reader_cache


def crop_affine(src_wh, crop_coords, target_size, rotation):
    """The one dst<-src affine for rotate -> crop -> resize. Returns (M_2x3, (w, h)) or None.

    None means all three are no-ops and the caller should not warp at all. Composition is not a
    micro-optimisation: sequential rotate-then-crop warps the whole frame and resamples twice.
    The CORNER convention (`x_dst = (x_src - x1) * sx`) matches `crop.apply_crop`'s offset and
    `resize_camera`'s mat scale, so the pixels agree with the intrinsics `project_points_torch`
    reprojects through. Out-of-source pixels arrive as BORDER_CONSTANT zeros.
    """
    w, h = src_wh
    M = np.eye(3)
    if rotation is not None:
        M_rot, (w, h) = rotation
        M[:2] = M_rot
    box = (0, 0, w, h) if crop_coords is None else tuple(int(c) for c in crop_coords)
    x1, y1, x2, y2 = box
    tw, th = ((x2 - x1, y2 - y1) if target_size is None
              else (int(target_size[0]), int(target_size[1])))
    if rotation is None and box == (0, 0, w, h) and (tw, th) == (w, h):
        return None
    sx, sy = tw / (x2 - x1), th / (y2 - y1)
    A = np.array([[sx, 0.0, -sx * x1], [0.0, sy, -sy * y1]])
    return (A @ M).astype(np.float32), (tw, th)


def load_warps(path, specs, target_size=None, reduce=1):
    """ONE decode, N affines -> list of (H,W,3) uint8 RGB. All None if the file will not decode.

    Decode cost is per FILE and warp cost per OUTPUT, so one `load_image` call per output frame
    charges the first at the rate of the second -- invisible until a caller wants several crops of
    the same frame. BGR is kept until after the warp so the colour convert runs on the SMALL
    output. `reduce` in {1,2,4,8} decodes at 1/N via libjpeg DCT decimation, only valid for
    whole-frame callers (asserted).
    """
    import cv2

    if reduce == 1:
        img = cv2.imread(path)
    else:
        assert all(c is None and r is None for c, r in specs), \
            'reduce decodes at 1/N, so source-pixel crop_coords/rotation would be misplaced'
        flag = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
        img = cv2.imread(path, flag)
    if img is None:
        return [None] * len(specs)
    src_wh = (img.shape[1], img.shape[0])
    out = []
    for crop_coords, rotation in specs:
        aff = crop_affine(src_wh, crop_coords, target_size, rotation)
        warped = (img if aff is None
                  else cv2.warpAffine(img, aff[0], aff[1], flags=cv2.INTER_LINEAR))
        out.append(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return out


def _read_video(path, group, cam, frames, crop_coords, target_size, rotation):
    """Frames from a video file, under one crop box and one rotation for the whole window.

    Each DISTINCT index is decoded once: a clamp-padded window repeats its last frame, and handing
    `get_batch` the repeats would decode them. A repeat gets a COPY, not a shared buffer:
    `_augment`'s cutout writes in place, and the dedupe would otherwise make repeats alias one
    buffer (same rule as `read_frames`).
    """
    import cv2

    key = str(path)
    want = [int(i) for i in frames]
    uniq = list(dict.fromkeys(want))
    with reader_cache.read_lock_for(key):
        dec = reader_cache.reader(key, group, cam).get_batch(uniq)
    at = {i: dec[n] for n, i in enumerate(uniq)}
    src_wh = (dec.shape[2], dec.shape[1])
    aff = crop_affine(src_wh, crop_coords, target_size, rotation)
    out, seen = [], set()
    for i in want:
        im = at[i]
        if aff is not None:
            out.append(cv2.warpAffine(im, aff[0], aff[1], flags=cv2.INTER_LINEAR))
        else:
            out.append(im.copy() if i in seen else im)
        seen.add(i)
    return out


def decode_errors() -> tuple[type[BaseException], ...]:
    """What a video decode raises for a frame it cannot deliver: `PyAVReader`'s own RuntimeError
    (a frame that did not decode, an index past the table) and libav's errors."""
    import av.error
    return (RuntimeError, av.error.FFmpegError)


def warn_decode_failure(group, cam, err: BaseException) -> None:
    """Say, once per (group, camera), that a loader item was skipped because its pixels did not
    decode -- the image path's unreadable file returns None and is retried the same way."""
    key = (getattr(group, 'group_id', None), cam, id(getattr(group, 'session', None)))
    if key in _decode_warned:
        return
    _decode_warned.add(key)
    warnings.warn(f'group {key[0]!r} camera {cam!r}: a frame did not decode, so this item is '
                  f'skipped and another drawn ({err})', stacklevel=3)


_decode_warned: set = set()


def read_frames(group, cam, frames, crop_coords=None, target_size=None, rotation=None,
                pool: ThreadPoolExecutor | None = None, reduce=1):
    """(T,H,W,3) uint8 RGB for one camera, from an image directory or a video.

    `reduce` is honoured for image directories and IGNORED for video (no decode-time decimation), so a
    caller must letterbox with `src_wh` and never assume the returned size. Frame names are computed,
    not listed (`%06d.<ext>` contiguous from 000000, spec §12). One crop box and one rotation apply to
    the whole window; each distinct index is decoded once (a clamp-padded window repeats its last
    frame) and a repeat gets a COPY, not an alias. Unreadable image frames come back as None.
    """
    kind, src, ext = group.source(cam)
    if kind == 'video':
        return _read_video(src, group, cam, frames, crop_coords, target_size, rotation)
    want = [int(i) for i in frames]
    path = lambda i: os.path.join(src, f'{i:06d}{ext}')           # noqa: E731
    uniq = list(dict.fromkeys(want))
    batch = lambda i: load_warps(path(i), [(crop_coords, rotation)],   # noqa: E731
                                 target_size, reduce)[0]
    if pool is None:
        got = {i: batch(i) for i in uniq}
    else:
        fs = {i: pool.submit(batch, i) for i in uniq}
        got = {i: f.result() for i, f in fs.items()}
    if any(v is None for v in got.values()):
        return [got[i] for i in want]
    seen, out = set(), []
    for i in want:
        out.append(got[i].copy() if i in seen else got[i])
        seen.add(i)
    return out
