"""Draw a prediction over the pixels it was made from. `scripts/infer.py --render` is the caller.

Ported from `../posetail-pose/scripts/render_tracking.py` -- `PALETTE`, `draw_instance` and the
lazy `VideoWriter` are that file's, near verbatim. What is deliberately NOT ported is the rest of
it: `render_tracking.py` re-runs inference to get something to draw, and `render_clip_npz.py`'s
docstring is explicit that this is the trap, because the render then shows a different pipeline
from the one the numbers came from. So there is no model here and no window loop -- `render_group`
takes the array `run_group` just returned.

Colour is per ANIMAL ROW, so an identity swap on a stationary animal shows up as a colour change
rather than as a number in a table.
"""
from __future__ import annotations

import numpy as np

from .dataset import read_frames

# BGR. Twelve, which is exactly rat-city's rat count -- no two animals share a colour there.
PALETTE = [(60, 60, 255), (60, 220, 60), (255, 140, 40), (40, 220, 220), (230, 80, 230),
           (250, 250, 60), (150, 90, 255), (90, 200, 150), (200, 200, 200), (80, 140, 255),
           (170, 110, 60), (60, 170, 110)]

CHUNK = 32          # frames read per batch. 480 rat-city frames at once is 13.8 GB.


def draw_instance(img, p2d, colour, skel_ix, tid, radius=3, lw=1, font=0.5):
    """One instance: skeleton (where the session has one) and keypoints, in place."""
    import cv2

    ok = np.isfinite(p2d).all(-1)
    for a, b in skel_ix:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(np.int32(p2d[a])), tuple(np.int32(p2d[b])), colour, lw, cv2.LINE_AA)
    for k in np.flatnonzero(ok):
        c = tuple(np.int32(p2d[k]))
        # Dark halo under the coloured dot -- a marker on a pale rat is otherwise invisible.
        cv2.circle(img, c, radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, c, radius, colour, -1, cv2.LINE_AA)
    if ok.any():
        c = np.int32(p2d[ok].mean(0))
        cv2.putText(img, str(tid), (int(c[0]) + 6, int(c[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font, colour, max(1, lw), cv2.LINE_AA)
    return img


def project(session, pred, cam, gid=None, frames=None):
    """(S,T,K,3) world points -> (S,T,K,2) pixels in camera `cam`'s stored image.

    aniposelib owns the projection, including distortion; `offset` is the one thing it does not
    model -- `matrix` is in SENSOR coordinates and the image on disk starts at `offset`, so the
    subtraction is what puts the point on the pixel a viewer sees (see `format.Rig`).

    NaN passes through: `project_points` is elementwise, so an unpredicted keypoint stays NaN and
    `draw_instance` skips it.

    ON A MOVING RIG THIS NEEDS `gid`, and without it it drew the skeleton off the animal.
    `Rig.by_name` hands back the camera carrying `calibration.toml`'s single NOMINAL extrinsic;
    the per-frame ones exist only through `Session.cgroup(gid, frames)`. That is gotcha 9's class
    and this was the fifth builder to drop `moving_ext` -- and the symptom, a pose that does not
    sit on the animal, reads as a model failure rather than as a render bug. The static path is
    left exactly as it was, so every 2D and static-rig render is unchanged.
    """
    import torch

    name = session.cam_names[cam]
    if gid is not None and session.rig.moving.get(name):
        from posetail.posetail.cube import project_points_torch

        # `format_camera` folds `offset` into the dict, so this comes back in image pixels
        # already -- the same call `infer._fill_box_agreement` makes, for the same reason.
        cams = session.cgroup(gid, frames)
        S = pred.shape[0]
        # PER ANIMAL, because `project_points_torch` aligns the (T,4,4) extrinsic against axis -3.
        # Flattening (S,T) into one axis there would silently project animal i through frame i's
        # camera pose -- the same alignment trap `test_the_tracker_projects_correctly_on_a_moving_rig`
        # pins one module over.
        with torch.no_grad():
            p = torch.as_tensor(np.asarray(pred), dtype=torch.float32)
            xy = [project_points_torch([cams[cam]], p[s])[0].cpu().numpy() for s in range(S)]
        return np.stack(xy).astype(np.float32)
    obj = session.rig.by_name(name)
    with torch.no_grad():
        p = torch.as_tensor(np.asarray(pred).reshape(-1, 3),
                            dtype=obj.matrix.dtype, device=obj.matrix.device)
        xy = obj.project(p).cpu().numpy()
    xy = xy - np.asarray(session.rig.offset[name], np.float64)
    return xy.reshape(pred.shape[:-1] + (2,)).astype(np.float32)


def follow(pred, zoom, W, H, smooth=15):
    """(T,2) int32 top-left corners of a `zoom`-square window that follows the prediction.

    A johnson mouse is ~100 px on a 3208x2200 sensor: whole-frame at any sane bitrate, the
    skeleton is a red smudge and the render cannot answer the question it was made to answer.

    Carried forward through frames the prediction misses, so the view does not snap back to the
    image corner every time the detector drops the animal, and smoothed, because a window that
    tracks per-frame jitter makes the *background* move and nothing else.
    """
    T = pred.shape[1]
    c = np.nanmean(np.moveaxis(pred, 1, 0).reshape(T, -1, 2), axis=1)
    ok = np.isfinite(c).all(1)
    if not ok.any():
        c[:] = (W / 2, H / 2)
    else:
        t = np.arange(T)
        for i in range(2):
            c[:, i] = np.interp(t, t[ok], c[ok, i])
    pad = np.pad(c, ((smooth // 2, smooth // 2), (0, 0)), mode='edge')
    c = np.mean([pad[i:i + T] for i in range(smooth)], axis=0)
    return np.stack([np.clip(c[:, 0] - zoom / 2, 0, max(0, W - zoom)),
                     np.clip(c[:, 1] - zoom / 2, 0, max(0, H - zoom))], 1).astype(np.int32)


def render_group(session, gid, pred, out_path, cam=0, max_side=1600, fps=15, zoom=0,
                 boxes=None):
    """Predicted tracks over one group's frames -> an mp4 at `out_path`.

    `pred` is `run_group`'s own output: `(S,T,K,2)` in SOURCE pixels for a 2D session, or
    `(S,T,K,3)` world points for a 3D one, which get projected into camera `cam` here. Only its
    first `T` frames are drawn, so a `--max-frames` run renders exactly the clip it predicted.

    `zoom` is the side, in SOURCE pixels, of a window that follows the prediction; 0 draws the
    whole frame. It is a view, not a crop rule -- it does not affect anything that was predicted.

    `boxes` is an optional `(S,T,4)` of `[x0,y0,x1,y1]` in the same source pixels, drawn in each
    animal's own colour, so a render answers whether the box and the keypoints describe the same
    animal.

    A 3D render is a REPROJECTION, not a measurement: a depth error along camera `cam`'s ray is
    invisible in it. Draw more than one camera before believing a 3D pose.
    """
    import cv2

    assert pred.ndim == 4 and pred.shape[-1] in (2, 3), f'bad prediction shape {pred.shape}'
    if pred.shape[-1] == 3:
        pred = project(session, pred, cam, gid)
    group = session.groups[gid]
    cam_name = session.cam_names[cam]
    W, H = (int(x) for x in session.rig.size(cam_name))
    zoom = min(int(zoom), W, H)
    scale = min(1.0, max_side / max(W, H))
    corners = follow(pred, zoom, W, H) if zoom else None
    size = ((2 * zoom, 2 * zoom) if zoom else
            (int(round(W * scale)), int(round(H * scale))))

    ix = {n: i for i, n in enumerate(session.names)}
    skel_ix = [(ix[a], ix[b]) for a, b in session.skeleton if a in ix and b in ix]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'cv2 could not open {out_path} for writing')
    T = pred.shape[1]
    # `t` indexes a COLUMN of `pred`; `src[t]` is the frame that column was predicted from. They
    # are the same thing only when the whole group is rendered.
    src = np.arange(T)
    try:
        for lo in range(0, T, CHUNK):
            cols = np.arange(lo, min(lo + CHUNK, T))
            imgs = read_frames(group, cam_name, src[cols])
            for t, rgb in zip(cols, imgs):
                if rgb is None:
                    continue
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if corners is not None:
                    x0, y0 = (int(v) for v in corners[t])
                    img = cv2.resize(img[y0:y0 + zoom, x0:x0 + zoom], size,
                                     interpolation=cv2.INTER_LINEAR)
                    origin, s = (x0, y0), size[0] / zoom
                else:
                    if scale != 1.0:
                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    origin, s = (0, 0), scale
                drawn = 0
                for a in range(pred.shape[0]):
                    colour = PALETTE[a % len(PALETTE)]
                    # Before the keypoint test: a box with no keypoints behind it is exactly the
                    # disagreement worth seeing.
                    if boxes is not None and np.isfinite(boxes[a, t]).all():
                        b = (boxes[a, t].reshape(2, 2) - origin) * s
                        cv2.rectangle(img, tuple(np.int32(b[0])), tuple(np.int32(b[1])),
                                      colour, 1, cv2.LINE_AA)
                    p = (pred[a, t] - origin) * s
                    if not np.isfinite(p).all(-1).any():
                        continue
                    draw_instance(img, p, colour, skel_ix, a)
                    drawn += 1
                cv2.putText(img, f'{session.session_id} {gid} {cam_name}  frame {int(src[t])}  '
                                 f'{drawn} tracked',
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)
                writer.write(img)
    finally:
        writer.release()
    return out_path
