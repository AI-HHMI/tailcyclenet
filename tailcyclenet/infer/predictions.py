"""A prediction IS a session in this repo's own format, written a block at a time.

The output is a `labels = "tracked"` session (long parquet tables, `status` as the visibility
channel, boxes in `instances.pq`), so `format.Session` reads it back and a prediction directory
is a first-class citizen of every tool here. `--out` names the session directory itself: a session
holds one calibration, one mode and one keypoint axis, so a run covering more than one source
session is refused rather than merged.

There are no pixels and no `groups/` -- the directory is not self-contained, and where the pixels
are is recorded in `[provenance]`, which `scripts/render.py` reads (an npz carries no provenance
to find its pixels with). Three columns are additive spec additions: `score` and `box_agree` on
`instances.pq`, and `score_logit` beside `score` on the keypoint tables -- the latter because
`sigmoid` rounds to exactly 1.0 in float32 at the logit medians this repo measures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..format import DICT_COLS, Session, TableWriter, dump_calibration, sessions_for, write_table

# The tables a prediction writes; `windows.pq` is per (animal, window, camera) diagnostics and
# `identity_events.pq` is the tracker's own record of what it did to identity -- both deliberately
# NOT spec tables.
_TABLES = ('points3d', 'keypoints', 'instances', 'windows', 'identity_events')


def _sigmoid(x):
    """The logistic sigmoid, computed in float64 so it does not saturate in float32."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


class SessionWriter:
    """One prediction session, appended a block at a time.

    The header (`session.toml`, `calibration.toml`, `groups.pq`) is written up front, so a run
    that dies half way leaves a directory that says what it was.
    """

    def __init__(self, out: Path, source: Session, registry, provenance: dict, groups):
        """Open a prediction session for writing: header first, parquet writers for every table.

        Inputs: out -- output session directory (created here).
                source -- the source session (calibration, mode, keypoint axis).
                registry -- keypoint registry; its `names` win over the session's.
                provenance -- dict or (key, value) pairs; duplicate keys with differing
                    values raise.
                groups -- group ids in run order, recorded in groups.pq.
        Side effects: writes session.toml, calibration.toml and groups.pq, and opens the
        per-table parquet writers. Provenance values are scalars and lists of strings
        (`source_videos` is the resolved file list); a duplicate key is a silent loss -- `dict`
        merges without complaint -- so the caller passes ITEMS and the collision is caught here.
        """
        self.out = Path(out)
        self.src = source
        self.names = list(registry.names) if hasattr(registry, 'names') else list(source.names)
        self.out.mkdir(parents=True, exist_ok=True)
        self._w = {t: TableWriter(self.out / f'{t}.pq', DICT_COLS) for t in _TABLES}

        if not isinstance(provenance, dict):
            seen = {}
            for k, v in provenance:
                if k in seen and seen[k] != v:
                    raise ValueError(
                        f'provenance key {k!r} given twice, as {seen[k]!r} and {v!r}. Two facts '
                        'under one name: rename one rather than letting the later win.')
                seen[k] = v
            provenance = seen

        import toml
        cfg = {'mode': source.mode, 'units': source.units, 'labels': 'tracked',
               'names': list(source.names),
               'assoc_res_max_px': float(source.assoc_res_max_px),
               'provenance': dict(provenance)}
        (self.out / 'session.toml').write_text(toml.dumps(cfg))
        dump_calibration(self.out / 'calibration.toml', source.rig)
        order = list(groups)
        write_table(self.out / 'groups.pq', {
            'group_id': np.array(order, dtype=object),
            'n_frames': np.array([source.groups[g].n_frames for g in order], np.int32),
            'fps': np.array([source.groups[g].fps for g in order], np.float32),
            'source_video': np.array([source.groups[g].source_video for g in order], dtype=object),
            'source_frame_start': np.array([source.groups[g].source_frame_start for g in order],
                                           np.int32),
            'source_frame_step': np.array([source.groups[g].source_frame_step for g in order],
                                          np.int32),
            'notes': np.array([source.groups[g].notes for g in order], dtype=object),
        }, dict_cols=())

    def write_block(self, gid: str, blk: dict, f0: int, w0: int) -> None:
        """One block's rows. `f0`/`w0` are its first frame and window in the WHOLE group.

        The spec's own sparsity rule -- "no row: not labelled" -- applies: a declined point
        writes nothing, which every consumer reads as absence. `keypoints.pq` holds the
        per-camera 2D pose: in 2D this is the prediction itself; in 3D it is the overlay a run
        used to discard. `instances.pq` holds the box, its objectness and the pose-to-box
        distance, per (animal, frame, camera) -- exactly this table's grain. `windows.pq` holds
        why a window produced nothing and what box it was given -- deliberately NOT a spec table.
        """
        ids = [str(x) for x in blk['animal_ids']]
        cams = self.src.cam_names
        kpts = list(self.src.names)
        pred, conf = np.asarray(blk['pred']), np.asarray(blk['conf'])
        S, T, K = pred.shape[0], pred.shape[1], pred.shape[2]
        if not (S and T and K):
            return
        a_ix, t_ix, k_ix = (x.ravel() for x in np.meshgrid(np.arange(S), np.arange(T),
                                                           np.arange(K), indexing='ij'))
        if pred.shape[-1] == 3:
            keep = np.isfinite(pred).all(-1).ravel()
            if keep.any():
                self._w['points3d'].write({
                    'group_id': np.array([gid] * int(keep.sum()), dtype=object),
                    'frame': (t_ix[keep] + f0).astype(np.int32),
                    'animal_id': np.array([ids[i] for i in a_ix[keep]], dtype=object),
                    'bodypart': np.array([kpts[i] for i in k_ix[keep]], dtype=object),
                    'status': np.array(['visible'] * int(keep.sum()), dtype=object),
                    'x': pred[..., 0].ravel()[keep].astype(np.float32),
                    'y': pred[..., 1].ravel()[keep].astype(np.float32),
                    'z': pred[..., 2].ravel()[keep].astype(np.float32),
                    'score': _sigmoid(conf.ravel()[keep]).astype(np.float32),
                    'score_logit': conf.ravel()[keep].astype(np.float32)})

        p2, c2 = np.asarray(blk['pred2d']), np.asarray(blk['conf2d'])
        C = p2.shape[2]
        a2, t2, c2i, k2 = (x.ravel() for x in np.meshgrid(
            np.arange(S), np.arange(T), np.arange(C), np.arange(K), indexing='ij'))
        keep2 = np.isfinite(p2).all(-1).ravel()
        if keep2.any():
            n = int(keep2.sum())
            self._w['keypoints'].write({
                'group_id': np.array([gid] * n, dtype=object),
                'frame': (t2[keep2] + f0).astype(np.int32),
                'animal_id': np.array([ids[i] for i in a2[keep2]], dtype=object),
                'camera': np.array([cams[i] for i in c2i[keep2]], dtype=object),
                'bodypart': np.array([kpts[i] for i in k2[keep2]], dtype=object),
                'status': np.array(['visible'] * n, dtype=object),
                'x': p2[..., 0].ravel()[keep2].astype(np.float32),
                'y': p2[..., 1].ravel()[keep2].astype(np.float32),
                'score': _sigmoid(c2.ravel()[keep2]).astype(np.float32),
                'score_logit': c2.ravel()[keep2].astype(np.float32)})

        ba = np.asarray(blk['box_agree'])
        det = blk.get('det_box')
        ai, ti, ci = (x.ravel() for x in np.meshgrid(np.arange(S), np.arange(T), np.arange(C),
                                                     indexing='ij'))
        have = np.isfinite(ba).ravel() if det is None else (
            np.isfinite(ba).ravel() | np.isfinite(np.asarray(det)).all(-1).ravel())
        if have.any():
            n = int(have.sum())
            rows = {'group_id': np.array([gid] * n, dtype=object),
                    'frame': (ti[have] + f0).astype(np.int32),
                    'animal_id': np.array([ids[i] for i in ai[have]], dtype=object),
                    'camera': np.array([cams[i] for i in ci[have]], dtype=object),
                    'status': np.array(['labeled'] * n, dtype=object),
                    'box_agree': ba.ravel()[have].astype(np.float32)}
            for j, q in enumerate(('x0', 'y0', 'x1', 'y1')):
                rows[q] = (np.full(n, np.nan, np.float32) if det is None
                           else np.asarray(det)[..., j].ravel()[have].astype(np.float32))
            ds = blk.get('det_score')
            rows['score'] = (np.full(n, np.nan, np.float32) if ds is None
                             else np.asarray(ds).ravel()[have].astype(np.float32))
            self._w['instances'].write(rows)

        oc, cr = np.asarray(blk['outcome']), np.asarray(blk['crop'])
        W = oc.shape[1]
        aw, ww, cw = (x.ravel() for x in np.meshgrid(np.arange(S), np.arange(W), np.arange(C),
                                                     indexing='ij'))
        names = list(blk['outcome_names'])
        rows = {'group_id': np.array([gid] * len(aw), dtype=object),
                'animal_id': np.array([ids[i] for i in aw], dtype=object),
                'camera': np.array([cams[i] for i in cw], dtype=object),
                'window': (ww + w0).astype(np.int32),
                'frame': np.asarray(blk['window_start'])[ww].astype(np.int32),
                'outcome': np.array([names[oc[a, w]] for a, w in zip(aw, ww)], dtype=object)}
        for j, q in enumerate(('x0', 'y0', 'x1', 'y1')):
            rows[q] = cr[..., j].ravel().astype(np.float32)
        rf = blk.get('crop_refined')
        for j, q in enumerate(('rx0', 'ry0', 'rx1', 'ry1')):
            rows[q] = (np.full(len(aw), np.nan, np.float32) if rf is None
                       else np.asarray(rf)[..., j].ravel().astype(np.float32))
        bp = blk.get('box_prompt_cams')
        rows['box_prompt_cams'] = (np.full(len(aw), -1, np.int32) if bp is None
                                   else np.asarray(bp)[aw, ww].astype(np.int32))
        self._w['windows'].write(rows)

    def write_identity_events(self, gid: str, events) -> None:
        """One group's tracker identity events -> `identity_events.pq`. Non-spec, like `windows`.

        Inputs: gid -- group id; events -- `CrossViewTracker.events`, a list of
            `{frame, slot, event, detail}`.
        Outputs: None.
        Side effects: appends to the `identity_events` writer; a no-op on an empty list.

        `detail` is a small per-event dict whose KEYS DIFFER BY EVENT TYPE (a retirement names
        its winner, a birth names its cameras), so it is stored as one JSON string column rather
        than exploded into sparse typed columns that would be null for most rows. The consumer is
        a diagnostic join, not a query engine, and a schema that changes shape per row is the
        thing parquet handles worst.

        Written once per group rather than per block: the event count is bounded by identity
        DECISIONS, not by clip length, so it does not reintroduce the proportionality the block
        loop exists to avoid. A pathological refire cycle is the one case that could grow it, and
        that is itself the bug such a log is for.
        """
        import json

        if not events:
            return
        n = len(events)
        self._w['identity_events'].write({
            'group_id': np.array([gid] * n, dtype=object),
            'frame': np.array([int(e['frame']) for e in events], np.int32),
            'slot': np.array([int(e['slot']) for e in events], np.int32),
            'event': np.array([str(e['event']) for e in events], dtype=object),
            'detail': np.array([json.dumps(e.get('detail', {}), sort_keys=True)
                                for e in events], dtype=object)})

    def close(self):
        """Close every open parquet writer."""
        for w in self._w.values():
            w.close()


def load_predictions(path, groups=None):
    """A prediction session or a legacy npz -> `({key: {field: ndarray}}, meta)`.

    The npz half is the archive -- every published number lives in one -- and scoring kept reading
    it even after rendering stopped. `groups`, given, restricts the read to those group ids (bare,
    not `session/group` keys); the session half scatters one group at a time, so a caller asking
    for one group never pays for the rest.
    """
    path = Path(path)
    return (_load_npz(path, groups) if path.suffix == '.npz' else _load_session(path, groups))


def _load_npz(path, groups=None):
    """A legacy npz archive -> ({key: {field: ndarray}}, meta); restricted to `groups` when given."""
    z = np.load(path, allow_pickle=True)
    keys = [str(k) for k in z['__keys__']]
    if groups is not None:
        want = set(groups)
        keys = [k for k in keys if k.split('/', 1)[1] in want]
    out = {}
    for key in keys:
        out[key] = {f.split('|', 1)[1]: z[f] for f in z.files if f.startswith(key + '|')}
    meta = {'run': str(z['__run__']), 'anchor': str(z['__anchor__']),
            'boxes': str(z['__boxes__']),
            'box_source': str(z['__box_source__']) if '__box_source__' in z.files else ''}
    return out, meta


def _load_session(path, groups=None):
    """Densify a prediction session back into the arrays `eval.py` reads.

    The scatter is `format.Session`'s own (the spec defines how long tables become dense arrays);
    this renames fields and adds the two non-spec tables. No `preload()` -- `Session.labels`
    caches per group, so scattering is already lazy.

    In 2D the prediction IS the per-camera pose at camera 0 (`coords_pred` is `2d_pred[0]`),
    which is what `keypoints.pq` holds; `pred2d` is transposed `(S,T,K,C,2) -> (S,T,C,K,2)`.
    `conf` is the LOGIT, from the additive `score_logit` column -- `sigmoid` cannot be inverted
    in float32 once it saturates, which it does at the medians this repo measures.
    """
    import tomllib

    sess = Session.load(Path(path))
    with open(Path(path) / 'session.toml', 'rb') as f:
        prov = tomllib.load(f).get('provenance', {})
    src_id = prov.get('source_session_id') or sess.session_id

    want = set(groups) if groups is not None else None
    out = {}
    for gid in sess.groups:
        if want is not None and gid not in want:
            continue
        lab = sess.labels(gid)
        d = {'mode': sess.mode, 'group_id': gid, 'session': src_id,
             'animal_ids': np.asarray(list(lab.animal_ids), object)}
        if sess.mode == '3d':
            d['pred'] = lab.points3d
        else:
            d['pred'] = lab.points2d[..., 0, :]
        if lab.points2d is not None:
            d['pred2d'] = np.moveaxis(lab.points2d, 3, 2)
        if lab.boxes is not None:
            d['boxes'] = lab.boxes
        d['conf'] = _score_logit(Path(path), gid, sess, lab, 'points3d' if sess.mode == '3d'
                                 else 'keypoints')
        ba = _instances_col(Path(path), gid, sess, lab, 'box_agree')
        if ba is not None:
            d['box_agree'] = ba
        out[f'{src_id}/{gid}'] = d
    meta = {'run': str(prov.get('run', '')), 'anchor': str(prov.get('anchor', '')),
            'boxes': str(prov.get('boxes', '')), 'box_source': str(prov.get('box_source', ''))}
    return out, meta


def _table(path: Path, stem: str):
    """`path/{stem}.pq` as a pyarrow table, or None if the file does not exist."""
    import pyarrow.parquet as pq
    f = path / f'{stem}.pq'
    return pq.read_table(f) if f.exists() else None


def _score_logit(path, gid, sess, lab, stem):
    """`(S,T,K)` of the visibility logit, scattered from the additive `score_logit` column.

    For the keypoints table there is one row per camera but `conf` is per keypoint, so only
    camera 0 is kept -- camera 0 is the 2D prediction's own head.
    """
    t = _table(path, stem)
    S, T, K = len(lab.animal_ids), sess.groups[gid].n_frames, len(sess.names)
    out = np.full((S, T, K), np.nan, np.float32)
    if t is None or 'score_logit' not in t.column_names or not len(t):
        return out
    import pyarrow.compute as pc
    t = t.filter(pc.equal(t.column('group_id'), gid))
    if not len(t):
        return out
    a = {v: i for i, v in enumerate(lab.animal_ids)}
    k = {v: i for i, v in enumerate(sess.names)}
    ai = np.array([a.get(str(v), -1) for v in t.column('animal_id').to_pylist()])
    ki = np.array([k.get(str(v), -1) for v in t.column('bodypart').to_pylist()])
    ti = np.asarray(t.column('frame').to_pylist(), int)
    sl = np.asarray(t.column('score_logit').to_pylist(), np.float32)
    ok = (ai >= 0) & (ki >= 0) & (ti < T)
    if stem == 'keypoints':
        c0 = str(sess.cam_names[0])
        ok &= np.array([str(v) == c0 for v in t.column('camera').to_pylist()])
    out[ai[ok], ti[ok], ki[ok]] = sl[ok]
    return out


def _instances_col(path, gid, sess, lab, col):
    """`(S,T,C)` of one additive `instances.pq` column, or None if it is not there."""
    t = _table(path, 'instances')
    if t is None or col not in t.column_names or not len(t):
        return None
    import pyarrow.compute as pc
    t = t.filter(pc.equal(t.column('group_id'), gid))
    S, T, C = len(lab.animal_ids), sess.groups[gid].n_frames, len(sess.rig)
    out = np.full((S, T, C), np.nan, np.float32)
    if not len(t):
        return out
    a = {v: i for i, v in enumerate(lab.animal_ids)}
    c = {v: i for i, v in enumerate(sess.cam_names)}
    ai = np.array([a.get(str(v), -1) for v in t.column('animal_id').to_pylist()])
    ci = np.array([c.get(str(v), -1) for v in t.column('camera').to_pylist()])
    ti = np.asarray(t.column('frame').to_pylist(), int)
    vv = np.asarray(t.column(col).to_pylist(), np.float32)
    ok = (ai >= 0) & (ci >= 0) & (ti < T)
    out[ai[ok], ti[ok], ci[ok]] = vv[ok]
    return out


def refuse_multi_session(data, split):
    """One session per run, checked before the checkpoint loads -> the single session.

    A session directory holds one calibration, one mode and one keypoint axis, so a prediction
    session cannot represent several at once.
    """
    ds_name, sessions = sessions_for(Path(data), split)
    if len(sessions) != 1:
        raise SystemExit(
            f'--data covers {len(sessions)} session(s) and --out is ONE session directory. A '
            'session holds one calibration, one mode and one keypoint axis, and these do not '
            'agree. Point --data at a single session directory (tailcyclenet reads one '
            f'directly), or run once per session. Sessions: '
            f'{[s.session_id for s in sessions][:5]}')
    return ds_name, sessions[0]
