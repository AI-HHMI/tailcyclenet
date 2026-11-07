"""Per-row identity crops from a prediction session, the input of the per-video pixel-ID network.

Three recipes, one output (`crops` (S,T,opx,opx) uint8 gray, `ok` (S,T), `ctr` (S,T,2) row centroid in
source px, `bl` body length in source px, plus `gid`, `ids`, `names`); every recipe is label-free:

- `keypoint_crops` (K >= 2 skeletons): centred on the row's keypoint mean, rotated so the body axis
  (tail base -> head keypoints) points up, side = `side` x body length, `out_px` px.
- `box_crops` (K = 1 without a mask head): the row's detector box centre, unrotated, side = `field`
  (negative = that multiple of the median box side), `out_px` px.
- `mask_crops` (K = 1 with a mask head): detector + `identity.maskhead` masks, each row takes the
  mask whose moment centroid is nearest its own centroid (Hungarian, <= 0.5 BL); centred on the
  mask centroid, rotated by the mask's second-moment axis (signed by its skew), spanning `side` BL.

Pixels: `keypoint_crops`/`box_crops` read libav's own gray conversion (`to_ndarray(format='gray')`),
which `data.video.PyAVReader` cannot produce (it decodes rgb24, and rgb24 -> gray differs from libav's
gray by up to 2 levels), so they decode here, sequentially, once per frame. `mask_crops` reads through
`data.video.open_reader` (rgb24 -> cv2 gray, the conversion the mask-crop recipe was built on). Frame
index = decode order (= `PyAVReader`'s presentation rank on any regular file).
"""
from __future__ import annotations

import tomllib
import warnings
from pathlib import Path

import cv2
import numpy as np

HEAD_NAMES = ('nose', 'neck', 'ear', 'head', 'snout')
VIDEO_SUFFIXES = ('.mp4', '.avi', '.mov', '.mkv')


def axis_parts(names):
    """(head keypoint indices, tail-base keypoint indices) of a skeleton, matched by name."""
    hd = [i for i, n in enumerate(names) if any(h in n.lower() for h in HEAD_NAMES)]
    tb = [i for i, n in enumerate(names) if 'tail_base' in n.lower() or n.lower() == 'tailbase']
    return hd, tb


def _session(pred_dir, source, group, stop=None):
    """(predictions of one group, session.toml, source session dir).

    `stop`: keep frames [0, stop) only -- every per-frame array is cut to it and a row with no table
    entry before `stop` is dropped, as if the session had been written for that range."""
    from ..format.prediction import load_predictions
    pred_dir = Path(pred_dir)
    meta = tomllib.load(open(pred_dir / 'session.toml', 'rb'))
    preds, _ = load_predictions(pred_dir, groups=None if group is None else [group])
    if not preds:
        raise ValueError(f'{pred_dir}: no group {group!r}')
    d = next(iter(preds.values()))
    if stop:
        import pandas as pd
        seen = set()
        for name in ('points3d', 'points2d', 'instances'):
            f = pred_dir / f'{name}.pq'
            if f.exists():
                t = pd.read_parquet(f, columns=['group_id', 'frame', 'animal_id'])
                seen |= set(t[(t.group_id.astype(str) == d['group_id']) & (t.frame < stop)].animal_id.astype(str))
        keep = np.array([str(a) in seen for a in d['animal_ids']])
        d = {k: (v[keep][:, :stop] if isinstance(v, np.ndarray) and v.ndim >= 2 else v) for k, v in d.items()}
        d['animal_ids'] = d['animal_ids'][keep]
    src = Path(source) if source is not None else Path(meta['provenance']['source_session'])
    return d, meta, src


def _media(src, gid):
    """('video', path) or ('images', camera dir) of a source group; the first camera."""
    gdir = Path(src) / 'groups' / gid
    if not gdir.exists():
        raise FileNotFoundError(f'{gdir}: source group has no pixels')
    vids = sorted(p for p in gdir.iterdir() if p.suffix in VIDEO_SUFFIXES)
    if vids:
        return 'video', vids[0]
    cams = sorted(p for p in gdir.iterdir() if p.is_dir())
    if not cams:
        raise FileNotFoundError(f'{gdir}: no video and no image directory')
    return 'images', cams[0]


def gray_frames(kind, path, n):
    """Yield (index, uint8 gray frame) for the first `n` frames: libav gray for a video, cv2 for images."""
    if kind == 'video':
        import av
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            st.thread_type = 'AUTO'
            for i, fr in enumerate(c.decode(st)):
                if i >= n:
                    break
                yield i, fr.to_ndarray(format='gray')
    else:
        ext = next(p.suffix for p in Path(path).iterdir())
        for i in range(n):
            yield i, cv2.imread(str(Path(path) / f'{i:06d}{ext}'), cv2.IMREAD_GRAYSCALE)


def _warp(img, cx, cy, side, th, opx, flags):
    """opx x opx crop of `img`: a `side` px square centred at (cx, cy), rotated by `th` (th + pi/2 = up)."""
    sc = opx / side
    c_, s_ = np.cos(th), np.sin(th)
    M = np.array([[c_, s_, 0], [-s_, c_, 0]], np.float64) * sc
    M[:, 2] = np.array([opx / 2, opx / 2]) - M[:, :2] @ np.array([cx, cy])
    return cv2.warpAffine(img, M, (opx, opx), flags=flags)


def _out(gid, crops, ok, ctr, bl, ids, names, **extra):
    """The crops dict every recipe returns (`ids` as str)."""
    return dict(gid=gid, crops=crops, ok=ok, ctr=ctr, bl=bl, ids=np.array([str(x) for x in ids]),
                names=np.array(names), **extra)


def save_crops(path, c):
    """Write a crops dict as ``{gid}__{field}`` npz (the layout `pixid.load_crops` reads)."""
    out = {f"{c['gid']}__{k}": v for k, v in c.items() if k not in ('gid', 'names')}
    np.savez(path, **out, names=c['names'])


def load_crops(path, group=None):
    """Read a crops npz back into a crops dict (`group` = None: the first group id, sorted)."""
    z = np.load(path)
    gid = group or sorted({k.split('__')[0] for k in z.files if '__' in k})[0]
    c = {k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(gid + '__')}
    c['bl'] = float(c['bl'])
    return dict(gid=gid, names=z['names'], **c)


# --------------------------------------------------------------------------- keypoint crops
def keypoint_crops(pred_dir, source=None, group=None, out_px=64, side=1.6, stop=None):
    """Axis-aligned keypoint crops of every row (K >= 2 skeletons with head and tail-base keypoints).

    Body length = 1.25 x median head-mean -> tail-base distance over rows with >= half their keypoints
    (the head-mean -> tail-base span is ~0.8 BL). A frame whose axis is undefined or shorter than 0.2 BL
    keeps the row's previous good angle (0 before the first); an all-NaN row has a NaN centroid.
    `stop` (every recipe): frames [0, stop) only, e.g. a ranged run's `frame_stop`. `ok` = >= max(2, ceil(K/2)) finite keypoints."""
    d, meta, src = _session(pred_dir, source, group, stop)
    names = list(meta['names'])
    gid = d['group_id']
    hd, tb = axis_parts(names)
    if not hd or not tb:
        raise ValueError(f'{pred_dir}: keypoint crops need head and tail-base keypoints, names {names}')
    P = d['pred'].astype(np.float32)
    S, T = P.shape[:2]
    fin = np.isfinite(P).all(-1)
    okr = fin.sum(-1) >= max(2, (P.shape[2] + 1) // 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ctr = np.nanmean(np.where(fin[..., None], P, np.nan), axis=2)
        H = np.nanmean(np.where(fin[..., hd, None], P[:, :, hd], np.nan), axis=2)
        Tb = np.nanmean(np.where(fin[..., tb, None], P[:, :, tb], np.nan), axis=2)
    v = H - Tb
    L = np.linalg.norm(v, axis=-1)
    bl = float(np.nanmedian(L[okr])) * 1.25
    ang = np.arctan2(v[..., 1], v[..., 0])
    good = np.isfinite(ang) & (L > 0.2 * bl)
    last = np.where(good, np.arange(T)[None], -1)
    last = np.maximum.accumulate(last, axis=1)
    ang = np.where(last >= 0, np.take_along_axis(ang, np.maximum(last, 0), axis=1), np.float32(0.0)).astype(ang.dtype)
    ok = okr & np.isfinite(ctr).all(-1)
    opx = out_px
    crops = np.zeros((S, T, opx, opx), np.uint8)
    for i, img in gray_frames(*_media(src, gid), T):
        for s in np.flatnonzero(ok[:, i]):
            crops[s, i] = _warp(img, *ctr[s, i], side * bl, ang[s, i] + np.pi / 2, opx, cv2.INTER_AREA)
    return _out(gid, crops, ok, ctr, bl, d['animal_ids'], names, ang=ang)


# --------------------------------------------------------------------------- box crops
def box_body_length(pred_dir, gid, stop=None):
    """Label-free body length of a prediction session's group, source px.

    K >= 2: median long side of the per-(frame, row) keypoint extents in points2d.pq; K = 1: median
    long side of the instances.pq boxes on the dense (frame, row) grid. `stop`: frames < stop only."""
    import pandas as pd
    pred_dir = Path(pred_dir)
    ins = pd.read_parquet(pred_dir / 'instances.pq')
    grp = pd.read_parquet(pred_dir / 'groups.pq')
    pts = pd.read_parquet(pred_dir / 'points2d.pq')
    T = int(grp[grp.group_id.astype(str) == gid].iloc[0].n_frames)
    if stop:
        T, ins, pts = min(T, stop), ins[ins.frame < stop], pts[pts.frame < stop]
    a = ins[ins.group_id.astype(str) == gid]
    ids = sorted(a.animal_id.astype(str).unique())
    im = {x: i for i, x in enumerate(ids)}
    box = np.full((T, len(ids), 4), np.nan, np.float32)
    box[a.frame.to_numpy(), a.animal_id.astype(str).map(im).to_numpy()] = a[['x0', 'y0', 'x1', 'y1']].to_numpy(np.float32)
    p = pts[pts.group_id.astype(str) == gid]
    if p.bodypart.nunique() >= 2:
        ex = p.groupby(['frame', 'animal_id'], observed=True).agg(x0=('x', 'min'), x1=('x', 'max'), y0=('y', 'min'), y1=('y', 'max'))
        return float(np.nanmedian(np.maximum(ex.x1 - ex.x0, ex.y1 - ex.y0)))
    return float(np.nanmedian(np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1])))


def box_crops(pred_dir, source=None, group=None, field=-1.0, out_px=44, stop=None):
    """Unrotated crops centred on each row's detector box (K = 1 roots without a mask head).

    `field` = crop side in source px; negative = that multiple of the median box side. `ctr` is the
    row's keypoint mean, `ok` = finite box with side > 1 px and a finite centroid."""
    d, meta, src = _session(pred_dir, source, group, stop)
    gid = d['group_id']
    P = d['pred'].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ctr = np.nanmean(P[..., :2].reshape(P.shape[0], P.shape[1], -1, 2), 2)
    bl = np.float32(box_body_length(pred_dir, gid, stop))
    B = d['boxes'][:, :, 0].astype(np.float64)
    S, T = B.shape[:2]
    side = np.maximum(B[..., 2] - B[..., 0], B[..., 3] - B[..., 1])
    okb = np.isfinite(B).all(-1) & (side > 1)
    bc = np.stack([(B[..., 0] + B[..., 2]) / 2, (B[..., 1] + B[..., 3]) / 2], -1)
    med = float(np.nanmedian(side[okb]))
    if field < 0:
        field = -field * med
    opx = out_px
    crops = np.zeros((S, T, opx, opx), np.uint8)
    for t, img in gray_frames(*_media(src, gid), T):
        for s in np.flatnonzero(okb[:, t]):
            crops[s, t] = _warp(img, *bc[s, t], field, 0.0, opx, cv2.INTER_AREA)
    ok = okb & np.isfinite(ctr).all(-1)
    return _out(gid, crops, ok, ctr, bl, d['animal_ids'], [str(n) for n in meta['names']])


# --------------------------------------------------------------------------- mask crops
def _moment_geom(mask):
    """(cx, cy, axis angle) of a binary mask; the axis is signed toward positive skew (the long thin
    side, away from the heavy end); None under 5 px."""
    mu = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if mu['m00'] < 5:
        return None
    cx, cy = mu['m10'] / mu['m00'], mu['m01'] / mu['m00']
    th = 0.5 * np.arctan2(2 * mu['mu11'], mu['mu20'] - mu['mu02'])
    ys, xs = np.nonzero(mask)
    p = (xs - cx) * np.cos(th) + (ys - cy) * np.sin(th)
    if (p ** 3).mean() < 0:
        th += np.pi
    return cx, cy, th


def _mask_geoms(qs, Minv, sw, sh, pad=4):
    """Source-px geometry of each mask's exclusive region (argmax == j and soft_j > 0.5).

    That region only exists where soft_j > 0.5, so every mask is warped into mask j's own ROI (bbox of
    q_j > 0.3 mapped to source px, + pad) instead of the whole frame."""
    out = []
    for j, q in enumerate(qs):
        ys, xs = np.nonzero(q > 0.3)
        if not len(xs):
            out.append(None)
            continue
        P = np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()], [xs.min(), ys.max() + 1], [xs.max() + 1, ys.max() + 1]], np.float64)
        Q = P @ Minv[:, :2].T + Minv[:, 2]
        x0, y0 = int(max(0, np.floor(Q[:, 0].min()) - pad)), int(max(0, np.floor(Q[:, 1].min()) - pad))
        x1, y1 = int(min(sw, np.ceil(Q[:, 0].max()) + pad)), int(min(sh, np.ceil(Q[:, 1].max()) + pad))
        if x1 <= x0 or y1 <= y0:
            out.append(None)
            continue
        Mr = Minv.copy()
        Mr[:, 2] -= np.array([x0, y0])
        soft = np.stack([cv2.warpAffine(qq, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR) for qq in qs])
        g = _moment_geom((soft.argmax(0) == j) & (soft[j] > 0.5))
        out.append(None if g is None else (g[0] + x0, g[1] + y0, g[2]))
    return out


def _frame_batches(kind, path, n, batch):
    """Yield (frame indices, [uint8 gray frames]) in batches over the first `n` frames. A video reads
    through `data.video.open_reader` (rgb24 -> cv2 gray); an image directory through `gray_frames`."""
    if kind == 'video':
        from ..data.video import open_reader
        reader = open_reader(str(path))
        try:
            for f0 in range(0, min(n, len(reader)), batch):
                fs = list(range(f0, min(n, len(reader), f0 + batch)))
                yield fs, [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in reader.get_batch(fs)]
        finally:
            reader.close()
        return
    buf = []
    for i, img in gray_frames(kind, path, n):
        buf.append(img)
        if len(buf) == batch:
            yield list(range(i + 1 - batch, i + 1)), buf
            buf = []
    if buf:
        yield list(range(n - len(buf), n)), buf


def _prep(img, in_wh):
    """Source RGB uint8 -> (letterboxed, equalized detector input, 2x3 affine source -> input)."""
    from ..detector.augment import normalize_input
    from ..detector.letterbox import letterbox
    lb, s, (px, py) = letterbox(img, in_wh, antialias=True)
    return normalize_input(lb, 'equalize'), np.array([[s, 0, px], [0, s, py]], np.float64)


def mask_body_length(pred_dir, stop=None):
    """Label-free body length for mask crops: median long side of instances.pq boxes / 0.9 (source px),
    every group (`stop`: frames < stop only)."""
    import pandas as pd
    ins = pd.read_parquet(Path(pred_dir) / 'instances.pq')
    if stop:
        ins = ins[ins.frame < stop]
    bx = ins[['x0', 'y0', 'x1', 'y1']].to_numpy(float)
    return float(np.nanmedian(np.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]))) / 0.9


def mask_crops(pred_dir, detector, mask_head, source=None, group=None, out_px=52, device='cuda',
               batch=16, stop=None, body_length=None, side=1.6):
    """Mask-aligned crops (K = 1 roots with a mask head); `detector` from `detector.checkpoint.load_detector`,
    `mask_head` from `identity.maskhead.load_maskhead` (both on `device`).

    The detector input is the deploy letterbox with antialiasing and `equalize` normalisation (what the
    mask heads were trained on). Body length (`body_length`, default `mask_body_length`) sets the 0.5 BL
    row-to-mask gate and, with `side` (crop side in BL), the crop scale out_px / (side x BL)."""
    from scipy.optimize import linear_sum_assignment
    from .maskhead import detect_masks
    d, meta, src = _session(pred_dir, source, group, stop)
    gid = d['group_id']
    P = d['pred'].astype(np.float32)
    S, T = P.shape[:2]
    fin = np.isfinite(P).all(-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ctr = np.nanmean(np.where(fin[..., None], P, np.nan), axis=2)
    BL = mask_body_length(pred_dir, stop) if body_length is None else float(body_length)
    opx = out_px
    k = (opx / side) / BL
    IN = detector.input_wh
    model = detector.model
    crops = np.zeros((S, T, opx, opx), np.uint8)
    ok = np.zeros((S, T), bool)

    def crop(img, cx, cy, th):
        """opx x opx crop at scale k, mask axis `th` rotated to point up; INTER_AREA when shrinking."""
        rot = th + np.pi / 2
        c_, s_ = np.cos(rot), np.sin(rot)
        M = np.array([[c_, s_, 0], [-s_, c_, 0]], np.float64) * k
        M[:, 2] = np.array([opx / 2, opx / 2]) - M[:, :2] @ np.array([cx, cy])
        return cv2.warpAffine(img, M, (opx, opx), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)

    for fs, frames in _frame_batches(*_media(src, gid), T, batch):
        ims, Ms = zip(*[_prep(np.repeat(fr[..., None], 3, -1), IN) for fr in frames])
        out = detect_masks(model, mask_head, list(ims), top_k=S, device=device)
        for f, fr, M, o in zip(fs, frames, Ms, out):
            if not len(o[2]):
                continue
            img = fr.astype(np.float32)
            sh, sw = fr.shape
            g = _mask_geoms([cv2.resize(q, IN, interpolation=cv2.INTER_LINEAR) for q in o[2]],
                            cv2.invertAffineTransform(M), sw, sh)
            Dm = np.full((S, len(g)), 1e6)
            for s in range(S):
                for j, gg in enumerate(g):
                    if np.isfinite(ctr[s, f]).all() and gg is not None:
                        Dm[s, j] = np.hypot(gg[0] - ctr[s, f, 0], gg[1] - ctr[s, f, 1])
            for s, j in zip(*linear_sum_assignment(Dm)):
                if Dm[s, j] > 0.5 * BL:
                    continue
                crops[s, f] = crop(img, *g[j]).astype(np.uint8)
                ok[s, f] = True
    return _out(gid, crops, ok, ctr[:, :T], np.float32(BL), d['animal_ids'], [str(n) for n in meta['names']])
