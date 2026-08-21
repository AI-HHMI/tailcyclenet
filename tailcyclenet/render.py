"""Draw a prediction over the pixels it was made from. `scripts/render.py` is the caller.

Ported from `../posetail-pose/scripts/render_tracking.py` -- `PALETTE`, `draw_instance` and the
lazy `VideoWriter` are that file's, near verbatim. What is deliberately NOT ported is the rest of
it: `render_tracking.py` re-runs inference to get something to draw, and `render_clip_npz.py`'s
docstring is explicit that this is the trap, because the render then shows a different pipeline
from the one the numbers came from. So there is no model here and no window loop -- `render_group`
takes the array `run_group`/a prediction session already holds.

Colour is per ANIMAL ROW, so an identity swap on a stationary animal shows up as a colour change
rather than as a number in a table.

`session_for_prediction` and `resolve_camera`, at the bottom, are the CLI's own two questions --
where are this prediction's pixels, and which camera did `--cams TOKEN` mean -- kept here rather
than in `scripts/render.py` because both need `Session`/`Rig` internals a thin CLI should not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import read_frames

# BGR. Twelve, which is exactly rat-city's rat count -- no two animals share a colour there.
PALETTE = [(60, 60, 255), (60, 220, 60), (255, 140, 40), (40, 220, 220), (230, 80, 230),
           (250, 250, 60), (150, 90, 255), (90, 200, 150), (200, 200, 200), (80, 140, 255),
           (170, 110, 60), (60, 170, 110)]

CHUNK = 32          # frames read per batch. 480 rat-city frames at once is 13.8 GB.


def draw_instance(img, p2d, colour, skel_ix, tid, radius=3, lw=1, font=0.5, marker='dot',
                  label=True):
    """One instance: skeleton (where the session has one) and keypoints, in place.

    `marker='cross'` and `label=False` are the OVERLAY style (`render_group`'s `overlay=`): a
    second prediction drawn over the first must be visually distinct with no halo and no second
    id text competing for the same six pixels, or the two readings are indistinguishable.
    """
    import cv2

    # isfinite alone lets a degenerate triangulation (a finite but ~1e20 px point) through, which
    # overflows the int32 cast below and crashes cv2 with a useless "wrong type" error. Bound to
    # the frame plus a margin generous enough to still draw a point just off-canvas.
    margin = 4 * max(img.shape[:2])
    ok = np.isfinite(p2d).all(-1) & (np.abs(p2d[..., 0]) < margin) & (np.abs(p2d[..., 1]) < margin)
    for a, b in skel_ix:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(np.int32(p2d[a])), tuple(np.int32(p2d[b])), colour, lw, cv2.LINE_AA)
    for k in np.flatnonzero(ok):
        c = tuple(np.int32(p2d[k]))
        if marker == 'cross':
            d = radius
            cv2.line(img, (c[0] - d, c[1] - d), (c[0] + d, c[1] + d), colour, lw, cv2.LINE_AA)
            cv2.line(img, (c[0] - d, c[1] + d), (c[0] + d, c[1] - d), colour, lw, cv2.LINE_AA)
        else:
            # Dark halo under the coloured dot -- a marker on a pale rat is otherwise invisible.
            cv2.circle(img, c, radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, c, radius, colour, -1, cv2.LINE_AA)
    if label and ok.any():
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
                 boxes=None, frames=None, overlay=None):
    """Predicted tracks over one group's frames -> an mp4 at `out_path`.

    `pred` is `run_group`'s own output (or a prediction session's `pred`/`pred2d`): `(S,T,K,2)` in
    SOURCE pixels for a 2D session, or `(S,T,K,3)` world points for a 3D one, which get projected
    into camera `cam` here. Only its first `T` frames are drawn, so a `--max-frames` run renders
    exactly the clip it predicted.

    `frames` is an optional `(T,)` array of SOURCE frame indices, aligned to `pred`'s columns --
    the caller's own slice, e.g. `np.arange(f0, f1)` for a `--start-frame`/`--end-frame` render.
    `None` keeps the old assumption, `np.arange(T)`, i.e. `pred`'s column `t` IS source frame `t`.
    It is what decides which pixels are decoded, what the burned-in frame number reads, and --
    on a MOVING rig -- which per-frame extrinsic `project` reads; passing the wrong one there
    draws the skeleton off the animal with no other symptom (gotcha 9's class).

    `zoom` is the side, in SOURCE pixels, of a window that follows the prediction; 0 draws the
    whole frame. It is a view, not a crop rule -- it does not affect anything that was predicted.

    `boxes` is an optional `(S,T,4)` of `[x0,y0,x1,y1]` in the same source pixels, OR `(S,T,C,4)`
    -- camera `cam`'s own boxes are then selected, so a caller holding `instances.pq`'s full
    per-camera array does not have to slice it first. Drawn in each animal's own colour, so a
    render answers whether the box and the keypoints describe the same animal.

    `overlay` is an optional SECOND `(S,T,K,2)` (or `(S,T,C,K,2)`, camera-sliced the same way) set
    of points, already in `cam`'s own pixel space -- drawn as thin, unlabelled crosses in the same
    animal's colour. The one designed use is the 3D reprojection against the per-camera 2D head's
    own prediction: they are different quantities and a render is the only place their
    disagreement is legible.

    A 3D render is a REPROJECTION, not a measurement: a depth error along camera `cam`'s ray is
    invisible in it. Draw more than one camera before believing a 3D pose.
    """
    import cv2

    assert pred.ndim == 4 and pred.shape[-1] in (2, 3), f'bad prediction shape {pred.shape}'
    T = pred.shape[1]
    # `t` indexes a COLUMN of `pred`; `src[t]` is the frame that column was predicted from. They
    # are the same thing only when `frames` is None, i.e. the whole group is rendered.
    src = np.asarray(frames) if frames is not None else np.arange(T)
    assert len(src) == T, f'frames has {len(src)} entries for a {T}-frame prediction'
    if pred.shape[-1] == 3:
        pred = project(session, pred, cam, gid, src)
    if boxes is not None:
        boxes = np.asarray(boxes)
        if boxes.ndim == 4:                  # (S,T,C,4) -> this camera's own boxes
            boxes = boxes[:, :, cam]
    if overlay is not None:
        overlay = np.asarray(overlay)
        if overlay.ndim == 5:                # (S,T,C,K,2) -> this camera's own overlay
            overlay = overlay[:, :, cam]
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
                    if overlay is not None:
                        po = (overlay[a, t] - origin) * s
                        if np.isfinite(po).all(-1).any():
                            draw_instance(img, po, colour, skel_ix, a, marker='cross', label=False)
                cv2.putText(img, f'{session.session_id} {gid} {cam_name}  frame {int(src[t])}  '
                                 f'{drawn} tracked',
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)
                writer.write(img)
    finally:
        writer.release()
    return out_path


def resolve_camera(session, token: str) -> int:
    """A `--cams` token -> a camera INDEX. NAME first, then index.

    `--cams 0` is fine on `cam0`/`cam1`; a johnson rig names its cameras `Cam2005325`, and
    `--cams Cam2005325` must work too -- `--cam-regex` is what named them that way in the first
    place, and a token that IS a camera name must never be silently re-read as a position.
    """
    names = list(session.cam_names)
    if token in names:
        return names.index(token)
    try:
        i = int(token)
    except ValueError:
        i = None
    if i is not None and 0 <= i < len(names):
        return i
    raise SystemExit(f'--cams {token!r} is neither a camera name nor a valid index. This '
                     f'session has {len(names)} camera(s): {names}.')


def session_for_prediction(pred, data=None, split=None):
    """The SOURCE session a prediction was made from: the pixels, the rig, the skeleton.

    THE PREDICTION SAYS WHERE ITS OWN PIXELS ARE. `session.toml`'s `[provenance]` carries either
    `source_videos` (a `--videos` run -- reconstructed by `adopt.session_from_prediction`, which
    re-derives the group/camera map from the recorded file list and CHECKS itself against the
    prediction's own `groups.pq`/`calibration.toml`, so nothing is probed and gotcha 10 is not
    reachable here) or `source_session` (a directory run -- `sessions_for` on that path and split).

    `data` is an OVERRIDE for a root that MOVED since the run, not the normal input, and it is
    CHECKED against the prediction rather than trusted: given, it replaces the lookup above but
    every guard below still runs, so a `--data` pointed at the wrong root still refuses.

    Four refusals, all before a single frame is decoded:

    - neither `source_session` nor `source_videos` and no `--data` -- this prediction predates
      provenance or was hand-written, and there is nothing to find its pixels with.
    - the resolved session's own id disagrees with the prediction's recorded `source_session_id`
      -- rendering it would draw the prediction over the wrong pixels.
    - the resolved session's `names` disagree with the prediction's own -- a reordered axis draws
      bones between the wrong keypoints with no other symptom (gotcha 4's family).
    - a predicted group is missing from the resolved session, or its `n_frames` disagrees -- the
      shape of a re-converted root.
    """
    import tomllib

    import pyarrow.parquet as pq

    from . import adopt
    from .format import sessions_for

    pred = Path(pred)
    with open(pred / 'session.toml', 'rb') as f:
        cfg = tomllib.load(f)
    prov = cfg.get('provenance', {})
    src_id = prov.get('source_session_id') or ''

    if data is not None:
        _, sessions = sessions_for(Path(data), split or 'test')
        if len(sessions) != 1:
            raise SystemExit(
                f'--data {data} covers {len(sessions)} session(s), and a prediction was made '
                'from exactly one. Point --data at a single session directory, or drop it and '
                "let the prediction's own provenance find its pixels.")
        sess = sessions[0]
    elif prov.get('source_videos'):
        sess = adopt.session_from_prediction(pred)
    elif prov.get('source_session'):
        _, sessions = sessions_for(Path(prov['source_session']),
                                   prov.get('source_split') or split or 'test')
        if len(sessions) != 1:
            raise SystemExit(
                f'{pred}: [provenance] source_session {prov["source_session"]!r} now covers '
                f'{len(sessions)} session(s); it named exactly one at run time. Pass --data to '
                'point at the pixels by hand.')
        sess = sessions[0]
    else:
        raise SystemExit(
            f'{pred}: [provenance] has neither source_session nor source_videos, so this '
            'prediction does not say where its pixels are. It predates provenance or was '
            'hand-written; pass --data to point at them.')

    if src_id and sess.session_id != src_id:
        raise SystemExit(
            f'{pred}: this prediction was made from session {src_id!r}, but the session found '
            f'now is {sess.session_id!r}. Rendering it would draw the prediction over the wrong '
            'pixels.')

    names = list(cfg.get('names') or [])
    if names and list(sess.names) != names:
        raise SystemExit(
            f"{pred}: the prediction's keypoint axis {names} does not match the source "
            f"session's {list(sess.names)}. A reordered axis draws bones between the wrong "
            'keypoints with no symptom other than being wrong.')

    gt = pq.read_table(pred / 'groups.pq').to_pydict()
    for gid, n in zip(gt['group_id'], gt['n_frames']):
        gid = str(gid)
        if gid not in sess.groups:
            raise SystemExit(
                f'{pred}: predicted group {gid!r} is not in session {sess.session_id!r} '
                f'({sorted(sess.groups)[:5]}...).')
        if sess.groups[gid].n_frames != int(n):
            raise SystemExit(
                f'{pred}: group {gid!r} was predicted at {int(n)} frames, but the session found '
                f'now has {sess.groups[gid].n_frames}. This looks like a re-converted root.')

    return sess
