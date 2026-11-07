"""Per-row identity crops from a prediction session, the input of the per-video pixel-ID network.

One recipe for every skeleton (owner decision 2026-11-07): `box_crops`, the row's detector box centre,
unrotated, side = `field` (negative = that multiple of the median box side), `out_px` px. Output:
`crops` (S,T,opx,opx) uint8 gray, `ok` (S,T), `ctr` (S,T,2) row centroid in source px, `bl` body length
in source px, plus `gid`, `ids`, `names`; label-free. The scratch keypoint-aligned and mask-head recipes
are deliberately NOT in the repo (keypoint crops were the selection recipe for K >= 2 roots, so box
crops on K >= 2 are unmeasured by the v7 / v8 finals).

Pixels: the recipe reads libav's own gray conversion (`to_ndarray(format='gray')`),
which `data.video.PyAVReader` cannot produce (it decodes rgb24, and rgb24 -> gray differs from libav's
gray by up to 2 levels), so they decode here, sequentially, once per frame. Frame index = decode order
(= `PyAVReader`'s presentation rank on any regular file).
"""
from __future__ import annotations

import tomllib
import warnings
from pathlib import Path

import cv2
import numpy as np

VIDEO_SUFFIXES = ('.mp4', '.avi', '.mov', '.mkv')


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
    """Unrotated crops centred on each row's detector box (every root).

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
