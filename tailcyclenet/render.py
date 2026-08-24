"""Draw a prediction over the pixels it was made from. `scripts/render.py` is the caller.

No model and no window loop: `render_group` takes the array a prediction session already holds,
so the render shows the same pipeline the numbers came from. Colour is per ANIMAL ROW, so an
identity swap shows as a colour change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import read_frames

# BGR. Twelve, which is exactly rat-city's rat count -- no two animals share a colour there.
PALETTE = [(60, 60, 255), (60, 220, 60), (255, 140, 40), (40, 220, 220), (230, 80, 230),
           (250, 250, 60), (150, 90, 255), (90, 200, 150), (200, 200, 200), (80, 140, 255),
           (170, 110, 60), (60, 170, 110)]

# Frames read per batch. 480 rat-city frames at once is 13.8 GB.
CHUNK = 32


def draw_instance(img, p2d, colour, skel_ix, tid, radius=3, lw=1, font=0.5, marker='dot',
                  label=True):
    """One instance: skeleton (where the session has one) and keypoints, in place. `cross`/
    `label=False` is the overlay style: visually distinct, no halo, no competing id text.

    `isfinite` alone would let a degenerate triangulation (~1e20 px) through and overflow the
    int32 cast, so points are also bounded to the frame plus a margin. The dot marker draws a
    dark halo under it -- a marker on a pale rat is otherwise invisible.
    """
    import cv2

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
            cv2.circle(img, c, radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, c, radius, colour, -1, cv2.LINE_AA)
    if label and ok.any():
        c = np.int32(p2d[ok].mean(0))
        cv2.putText(img, str(tid), (int(c[0]) + 6, int(c[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font, colour, max(1, lw), cv2.LINE_AA)
    return img


def project(session, pred, cam, gid=None, frames=None):
    """(S,T,K,3) world points -> (S,T,K,2) pixels in camera `cam`'s stored image.

    aniposelib owns the projection; `offset` is subtracted because `matrix` is in SENSOR
    coordinates. ON A MOVING RIG this needs `gid`: the per-frame extrinsics exist only through
    `Session.cgroup(gid, frames)`, and without them the skeleton draws off the animal silently.
    `format_camera` folds `offset` into the dict, so the moving-rig path comes back in image
    pixels already -- the same call `infer._fill_box_agreement` makes, for the same reason. The
    projection is PER ANIMAL: `project_points_torch` aligns the (T,4,4) extrinsic against axis
    -3, so flattening (S,T) would project animal i through frame i's pose.
    """
    import torch

    name = session.cam_names[cam]
    if gid is not None and session.rig.moving.get(name):
        from posetail.posetail.cube import project_points_torch

        cams = session.cgroup(gid, frames)
        S = pred.shape[0]
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
    """(T,2) int32 top-left corners of a `zoom`-square window that follows the prediction,
    carried through missed frames and smoothed so the background does not jitter.
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

    `pred` is `run_group`'s own output (or a prediction session's `pred`/`pred2d`): `(S,T,K,2)`
    in SOURCE pixels for a 2D session, or `(S,T,K,3)` world points for a 3D one, projected into
    camera `cam` here; only its first `T` frames are drawn. `frames` optionally maps `pred`'s
    columns to SOURCE frame indices (a ranged render); `zoom` is a follow window in source
    pixels; `boxes` and `overlay` draw the animal's box and a second point set. A 3D render is a
    REPROJECTION, not a measurement -- draw more than one camera before believing a 3D pose.

    `t` indexes a COLUMN of `pred`; `src[t]` is the frame that column was predicted from -- the
    same thing only when `frames` is None, i.e. the whole group is rendered. `boxes` is
    (S,T,C,4) and `overlay` is (S,T,C,K,2), each reduced to this camera's own slice. A box with
    no keypoints behind it is exactly the disagreement worth seeing.
    """
    import cv2

    assert pred.ndim == 4 and pred.shape[-1] in (2, 3), f'bad prediction shape {pred.shape}'
    T = pred.shape[1]
    src = np.asarray(frames) if frames is not None else np.arange(T)
    assert len(src) == T, f'frames has {len(src)} entries for a {T}-frame prediction'
    if pred.shape[-1] == 3:
        pred = project(session, pred, cam, gid, src)
    if boxes is not None:
        boxes = np.asarray(boxes)
        if boxes.ndim == 4:
            boxes = boxes[:, :, cam]
    if overlay is not None:
        overlay = np.asarray(overlay)
        if overlay.ndim == 5:
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
    """A `--cams` token -> a camera INDEX. NAME first, then index: a token that IS a camera name
    must never be silently re-read as a position.
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

    `[provenance]` says where the pixels are (`source_videos` rebuilt by
    `adopt.session_from_prediction`, or `source_session`); `data` is an OVERRIDE for a moved
    root, checked rather than trusted. Refusals: no provenance, session id, `names` or a
    predicted group's `n_frames` disagreeing.
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
