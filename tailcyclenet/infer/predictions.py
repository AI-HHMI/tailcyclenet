"""Prediction sessions and their readers, written a block at a time.

The top-level marker distinguishes prediction-only `points2d.pq` from annotation tables; its
rows carry stable keypoint and camera names plus slot/window ownership. Primary 3D predictions,
per-camera 2D poses, detector instances and window diagnostics are separate typed tables. Optional
`window_predictions.pq` preserves each window's own pre-gate predictions for overlap analysis.
There are no pixels and no `groups/`; `[provenance]` identifies the source for rendering. A session
holds one calibration, mode and keypoint axis, so a run covering multiple source sessions is
refused rather than merged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..format import DICT_COLS, Session, TableWriter, dump_calibration, sessions_for, write_table

# The tables a prediction writes; `windows.pq` is per (animal, window, camera) diagnostics and
# `identity_events.pq` is the tracker's own record of what it did to identity -- both deliberately
# NOT spec tables.
_TABLES = ('points3d', 'points2d', 'instances', 'windows', 'identity_events')


def _sigmoid(x):
    """The logistic sigmoid, computed in float64 so it does not saturate in float32."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _frame_windows(blk: dict, f0: int, w0: int, T: int, group_frames: int,
                   default_length: int, frame_stop: int = 0):
    """Return global owner ordinals and their exact interval metadata for a primary block."""
    starts = np.asarray(blk.get('window_start', []), dtype=np.int64).reshape(-1)
    stops = np.asarray(blk.get('window_stop', []), dtype=np.int64).reshape(-1)
    W = len(starts)
    frames = np.arange(f0, f0 + T, dtype=np.int64)
    owner = blk.get('owner_window')
    if owner is None:
        if W:
            owner = w0 + np.maximum(0, np.searchsorted(starts, frames, side='right') - 1)
        else:
            owner = np.full(T, -1, np.int64)
    owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    if owner.shape != (T,):
        raise ValueError(f'owner_window must have shape {(T,)}, got {owner.shape}')
    local = owner - w0
    valid = (local >= 0) & (local < W)
    start_for_frame = np.full(T, -1, np.int32)
    stop_for_frame = np.full(T, -1, np.int32)
    if valid.any():
        start_for_frame[valid] = starts[local[valid]].astype(np.int32)
        if len(stops) == W:
            stop_for_frame[valid] = stops[local[valid]].astype(np.int32)
        else:
            ends = starts + int(default_length)
            ends = np.minimum(ends, int(group_frames))
            if frame_stop:
                ends = np.minimum(ends, int(frame_stop))
            stop_for_frame[valid] = ends[local[valid]].astype(np.int32)
    return owner.astype(np.int32), start_for_frame, stop_for_frame


def _row_ids(ids, slots, windows, independent: bool):
    """Resolve row IDs from fixed linked IDs or independent window/slot IDs."""
    if independent:
        return np.asarray([f'w{int(w):06d}_s{int(s):02d}' for s, w in zip(slots, windows)],
                          dtype=object)
    return np.asarray([ids[int(s)] for s in slots], dtype=object)


class _WindowPredictionWriter:
    """Typed, incrementally-written union schema for optional per-window prediction records."""

    STRING_FIELDS = ('record_type', 'group_id', 'animal_id', 'camera', 'bodypart', 'status',
                     'outcome')
    DICT_FIELDS = ('group_id', 'animal_id', 'camera', 'bodypart', 'status')
    INT_FIELDS = ('frame', 'window', 'window_start', 'window_stop', 'slot', 'box_prompt_cams')
    BOOL_FIELDS = ('gated',)
    FLOAT_FIELDS = (
        'x', 'y', 'z', 'triangulated_x', 'triangulated_y', 'triangulated_z', 'score',
        'score_logit', 'visibility_logit', 'visibility_probability', 'confidence_logit',
        'confidence_probability', 'det_score', 'box_agree', 'x0', 'y0', 'x1', 'y1',
        'crop_x0', 'crop_y0', 'crop_x1', 'crop_y1', 'refined_x0', 'refined_y0',
        'refined_x1', 'refined_y1')

    def __init__(self, path: Path):
        """Open a typed Parquet writer; status fields use a stable dictionary type."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = [pa.field(name, pa.string()) for name in self.STRING_FIELDS]
        fields.extend(pa.field(name, pa.int32()) for name in self.INT_FIELDS)
        fields.extend(pa.field(name, pa.float32()) for name in self.FLOAT_FIELDS)
        fields.extend(pa.field(name, pa.bool_()) for name in self.BOOL_FIELDS)
        fields = [pa.field(f.name, pa.dictionary(pa.int32(), pa.string())
                           if f.name in self.DICT_FIELDS else f.type) for f in fields]
        self.schema = pa.schema(fields, metadata={b'tailcyclenet.schema_version': b'1'})
        self._writer = pq.ParquetWriter(path, self.schema, compression='zstd')
        self._closed = False

    def write(self, records) -> None:
        """Normalize and append prediction records; empty batches are ignored."""
        import pyarrow as pa

        records = list(records)
        if not records:
            return
        normalized = []
        for record in records:
            row = dict(record)
            if 'record_type' not in row and 'type' in row:
                row['record_type'] = row.pop('type')
            if 'box' in row and row['box'] is not None:
                row.update(zip(('x0', 'y0', 'x1', 'y1'), row.pop('box')))
            if 'crop' in row and row['crop'] is not None:
                row.update(zip(('crop_x0', 'crop_y0', 'crop_x1', 'crop_y1'), row.pop('crop')))
            if 'crop_refined' in row and row['crop_refined'] is not None:
                row.update(zip(('refined_x0', 'refined_y0', 'refined_x1', 'refined_y1'),
                               row.pop('crop_refined')))
            for axis in ('x0', 'y0', 'x1', 'y1'):
                key = f'crop_refined_{axis}'
                if key in row:
                    row[f'refined_{axis}'] = row.pop(key)
            normalized.append(row)
        columns = {}
        for name in self.STRING_FIELDS:
            values = [row.get(name) for row in normalized]
            arr = pa.array(values, type=pa.string())
            columns[name] = arr.dictionary_encode() if name in self.DICT_FIELDS else arr
        for name in self.INT_FIELDS:
            columns[name] = pa.array([row.get(name) for row in normalized], type=pa.int32())
        for name in self.BOOL_FIELDS:
            columns[name] = pa.array([row.get(name) for row in normalized], type=pa.bool_())
        for name in self.FLOAT_FIELDS:
            values = [row.get(name) for row in normalized]
            values = [None if v is None or not np.isfinite(v) else float(v) for v in values]
            columns[name] = pa.array(values, type=pa.float32())
        self._writer.write_table(pa.table(columns, schema=self.schema))

    def close(self) -> None:
        """Close the underlying Parquet writer once."""
        if not self._closed:
            self._writer.close()
            self._closed = True


class SessionWriter:
    """One prediction session, appended a block at a time.

    The header (`session.toml`, `calibration.toml`, `groups.pq`, and `extrinsics.pq` when the
    source rig has a moving camera) is written up front, so a run that dies half way leaves a
    directory that says what it was.
    """

    def __init__(self, out: Path, source: Session, registry, provenance: dict, groups):
        """Open a prediction session for writing: header first, parquet writers for every table.

        Inputs: out -- output session directory (created here).
                source -- the source session (calibration, mode, keypoint axis).
                registry -- keypoint registry; its `names` win over the session's.
                provenance -- dict or (key, value) pairs; duplicate keys with differing
                    values raise.
                groups -- group ids in run order, recorded in groups.pq.
        Side effects: writes session.toml, calibration.toml, groups.pq and (D1.2, a moving
        source) extrinsics.pq, and opens the per-table parquet writers. Extrinsics are copied
        for the FULL group, not just the range this run predicted -- rule 13 needs every frame
        of every moving camera regardless, and `groups.pq` already keeps the group's full
        `n_frames` on a ranged run. Provenance values are scalars and lists of strings
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
        self.provenance = dict(provenance)
        self.independent_windows = bool(self.provenance.get('independent_windows', False))
        self.window_predictions = bool(self.provenance.get('window_predictions', False))
        self.window_length = int(self.provenance.get('n_frames', 0) or 0)
        self.frame_stop = int(self.provenance.get('frame_stop', 0) or 0)
        self._sidecar = (_WindowPredictionWriter(self.out / 'window_predictions.pq')
                         if self.window_predictions else None)

        import toml
        units = source.units
        if (source.mode == '3d' and len(source.cam_names) == 1
                and not source.rig.calibrated[source.cam_names[0]]
                and not source.rig.moving[source.cam_names[0]]):
            units = 'normalized'
        cfg = {'mode': source.mode, 'units': units, 'labels': 'tracked',
               'names': list(source.names), 'prediction_session': True, 'complete': False,
               'assoc_res_max_px': float(source.assoc_res_max_px),
               'provenance': dict(provenance)}
        self._session_cfg = cfg
        self._write_session_header(toml.dumps(cfg))
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
        if any(source.rig.moving.values()):
            ext_group, ext_frame, ext_camera, ext_vals = [], [], [], []
            for gid in order:
                lab = source.labels(gid)
                T = lab.ext.shape[1]
                for ci, name in enumerate(source.cam_names):
                    if not source.rig.moving[name]:
                        continue
                    ext_group.extend([gid] * T)
                    ext_frame.extend(range(T))
                    ext_camera.extend([name] * T)
                    ext_vals.extend(e.ravel().tolist() for e in lab.ext[ci])
            write_table(self.out / 'extrinsics.pq', {
                'group_id': np.array(ext_group, dtype=object),
                'frame': np.array(ext_frame, np.int32),
                'camera': np.array(ext_camera, dtype=object),
                'ext': ext_vals,
            }, dict_cols=('group_id', 'camera'))

    def write_block(self, gid: str, blk: dict, f0: int, w0: int) -> None:
        """One block's rows. `f0`/`w0` are its first frame and window in the WHOLE group.

        A declined primary point writes no row; independently captured window records preserve
        pre-gate predictions when requested. `points2d.pq` stores the per-camera pose, while
        `points3d.pq` stores world predictions and triangulation. `instances.pq` holds boxes,
        objectness and pose-to-box distance per (animal, frame, camera); `windows.pq` holds the
        primary outcome and crop diagnostics. The latter tables are deliberately non-spec.
        """
        ids = [str(x) for x in blk['animal_ids']]
        cams = self.src.cam_names
        kpts = list(self.src.names)
        pred, conf = np.asarray(blk['pred']), np.asarray(blk['conf'])
        S, T, K = pred.shape[0], pred.shape[1], pred.shape[2]
        if not (S and T and K):
            return
        owner, owner_start, owner_stop = _frame_windows(
            blk, f0, w0, T, self.src.groups[gid].n_frames, self.window_length, self.frame_stop)
        a_ix, t_ix, k_ix = (x.ravel() for x in np.meshgrid(np.arange(S), np.arange(T),
                                                           np.arange(K), indexing='ij'))
        if pred.shape[-1] == 3:
            keep = np.isfinite(pred).all(-1).ravel()
            if keep.any():
                self._w['points3d'].write({
                    'group_id': np.array([gid] * int(keep.sum()), dtype=object),
                    'frame': (t_ix[keep] + f0).astype(np.int32),
                    'animal_id': _row_ids(ids, a_ix[keep], owner[t_ix[keep]],
                                          self.independent_windows),
                    'bodypart': np.array([kpts[i] for i in k_ix[keep]], dtype=object),
                    'status': np.where(conf.ravel()[keep] > 0, 'visible', 'missing'),
                    'x': pred[..., 0].ravel()[keep].astype(np.float32),
                    'y': pred[..., 1].ravel()[keep].astype(np.float32),
                    'z': pred[..., 2].ravel()[keep].astype(np.float32),
                    'score': _sigmoid(conf.ravel()[keep]).astype(np.float32),
                    'score_logit': conf.ravel()[keep].astype(np.float32),
                    'slot': a_ix[keep].astype(np.int32),
                    'window': owner[t_ix[keep]].astype(np.int32),
                    'window_start': owner_start[t_ix[keep]].astype(np.int32),
                    'window_stop': owner_stop[t_ix[keep]].astype(np.int32),
                    'triangulated_x': (np.full(int(keep.sum()), np.nan, np.float32)
                                       if blk.get('triangulated') is None else
                                       np.asarray(blk['triangulated'])[..., 0].ravel()[keep].astype(np.float32)),
                    'triangulated_y': (np.full(int(keep.sum()), np.nan, np.float32)
                                       if blk.get('triangulated') is None else
                                       np.asarray(blk['triangulated'])[..., 1].ravel()[keep].astype(np.float32)),
                    'triangulated_z': (np.full(int(keep.sum()), np.nan, np.float32)
                                       if blk.get('triangulated') is None else
                                       np.asarray(blk['triangulated'])[..., 2].ravel()[keep].astype(np.float32))})

        p2, c2 = np.asarray(blk['pred2d']), np.asarray(blk['conf2d'])
        mc2 = np.asarray(blk.get('model_conf2d', np.full_like(c2, np.nan)))
        C = p2.shape[2]
        a2, t2, c2i, k2 = (x.ravel() for x in np.meshgrid(
            np.arange(S), np.arange(T), np.arange(C), np.arange(K), indexing='ij'))
        keep2 = np.isfinite(p2).all(-1).ravel()
        if keep2.any():
            n = int(keep2.sum())
            self._w['points2d'].write({
                'group_id': np.array([gid] * n, dtype=object),
                'frame': (t2[keep2] + f0).astype(np.int32),
                'animal_id': _row_ids(ids, a2[keep2], owner[t2[keep2]],
                                      self.independent_windows),
                'camera': np.array([cams[i] for i in c2i[keep2]], dtype=object),
                'bodypart': np.array([kpts[i] for i in k2[keep2]], dtype=object),
                'x': p2[..., 0].ravel()[keep2].astype(np.float32),
                'y': p2[..., 1].ravel()[keep2].astype(np.float32),
                'visibility_logit': c2.ravel()[keep2].astype(np.float32),
                'visibility_probability': _sigmoid(c2.ravel()[keep2]).astype(np.float32),
                'confidence_logit': mc2.ravel()[keep2].astype(np.float32),
                'confidence_probability': _sigmoid(mc2.ravel()[keep2]).astype(np.float32),
                'slot': a2[keep2].astype(np.int32),
                'window': owner[t2[keep2]].astype(np.int32),
                'window_start': owner_start[t2[keep2]].astype(np.int32),
                'window_stop': owner_stop[t2[keep2]].astype(np.int32)})

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
                    'animal_id': _row_ids(ids, ai[have], owner[ti[have]],
                                          self.independent_windows),
                    'camera': np.array([cams[i] for i in ci[have]], dtype=object),
                    'status': np.array(['labeled'] * n, dtype=object),
                    'box_agree': ba.ravel()[have].astype(np.float32),
                    'slot': ai[have].astype(np.int32),
                    'window': owner[ti[have]].astype(np.int32),
                    'window_start': owner_start[ti[have]].astype(np.int32),
                    'window_stop': owner_stop[ti[have]].astype(np.int32)}
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
        win_ids = (ww + w0).astype(np.int32)
        win_start = np.asarray(blk['window_start'], dtype=np.int32)
        win_stop = np.asarray(blk.get('window_stop', win_start + self.window_length),
                               dtype=np.int32)
        rows = {'group_id': np.array([gid] * len(aw), dtype=object),
                'animal_id': _row_ids(ids, aw, win_ids, self.independent_windows),
                'camera': np.array([cams[i] for i in cw], dtype=object),
                'slot': aw.astype(np.int32),
                'window': win_ids,
                'window_start': win_start[ww],
                'window_stop': win_stop[ww],
                'frame': win_start[ww],
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

    def write_window_records(self, records) -> None:
        """Append the current block's pre-gate, per-window rows when sidecar export is enabled."""
        if self._sidecar is not None:
            self._sidecar.write(records)

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

    def _write_session_header(self, text: str) -> None:
        """Atomically rewrite session metadata, leaving a readable completion marker."""
        path = self.out / 'session.toml'
        tmp = path.with_suffix('.toml.tmp')
        tmp.write_text(text)
        tmp.replace(path)

    def close(self, complete: bool = False):
        """Close writers and mark completion; create an empty typed points2d table if needed."""
        for w in self._w.values():
            w.close()
        if self._sidecar is not None:
            self._sidecar.close()
        if not (self.out / 'points2d.pq').exists():
            write_table(self.out / 'points2d.pq', {
                'group_id': np.array([], object), 'frame': np.array([], np.int32),
                'animal_id': np.array([], object), 'camera': np.array([], object),
                'bodypart': np.array([], object), 'x': np.array([], np.float32),
                'y': np.array([], np.float32),
                'visibility_logit': np.array([], np.float32),
                'visibility_probability': np.array([], np.float32),
                'confidence_logit': np.array([], np.float32),
                'confidence_probability': np.array([], np.float32),
                'slot': np.array([], np.int32), 'window': np.array([], np.int32),
                'window_start': np.array([], np.int32), 'window_stop': np.array([], np.int32),
            }, dict_cols=('group_id', 'animal_id', 'camera', 'bodypart'))
        self._session_cfg['complete'] = bool(complete)
        import toml
        self._write_session_header(toml.dumps(self._session_cfg))


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
    """Read marked prediction sessions or provenance-identified pre-marker outputs."""
    path = Path(path)
    sess = Session.load(path)
    with (path / 'session.toml').open('rb') as f:
        import tomllib
        cfg = tomllib.load(f)
    if sess.prediction_session:
        if sess.complete is False:
            raise ValueError(f'{path}: prediction session is incomplete (complete=false)')
        return _load_prediction_session(path, sess, cfg, groups)
    provenance = cfg.get('provenance', {})
    if provenance.get('run') and provenance.get('checkpoint'):
        return _load_legacy_session(path, groups)
    raise ValueError(f'{path}: not a marked prediction session; refusing ambiguous annotation data')


def _load_prediction_session(path, sess, cfg, groups=None):
    """Read primary prediction rows, slot-dense only when IDs are window-local."""
    import pyarrow.compute as pc

    provenance = cfg.get('provenance', {})
    independent = bool(provenance.get('independent_windows', False))
    p3_all = _table(path, 'points3d')
    p2_all = _table(path, 'points2d')
    inst_all = _table(path, 'instances')
    windows_all = _table(path, 'windows')
    wanted = set(groups) if groups is not None else None
    output = {}
    for gid, group in sess.groups.items():
        if wanted is not None and gid not in wanted:
            continue
        T, K, C = group.n_frames, len(sess.names), len(sess.cam_names)
        p3 = _filter_group(p3_all, gid, pc)
        p2 = _filter_group(p2_all, gid, pc)
        inst = _filter_group(inst_all, gid, pc)
        win = _filter_group(windows_all, gid, pc)
        if independent:
            slot_values = []
            for table in (p3, p2, inst, win):
                if table is not None and 'slot' in table.column_names and len(table):
                    slot_values.extend(_nullable_ints(table, 'slot').tolist())
            slot_values = [v for v in slot_values if v >= 0]
            S = max(slot_values, default=-1) + 1
            animal_ids = [f'slot_{i:02d}' for i in range(S)]
        else:
            lab = sess.labels(gid)
            animal_ids = list(lab.animal_ids)
            if not animal_ids:
                values = []
                for table in (p3, p2, inst):
                    if table is not None and len(table) and 'animal_id' in table.column_names:
                        values.extend(table.column('animal_id').to_pylist())
                animal_ids = sorted({str(v) for v in values if v is not None})
            S = len(animal_ids)
        id_index = {str(v): i for i, v in enumerate(animal_ids)}
        pred3 = np.full((S, T, K, 3), np.nan, np.float32)
        pred2 = np.full((S, T, C, K, 2), np.nan, np.float32)
        conf3 = np.full((S, T, K), np.nan, np.float32)
        conf2 = np.full((S, T, K), np.nan, np.float32)
        boxes = np.full((S, T, C, 4), np.nan, np.float32)
        box_agree = np.full((S, T, C), np.nan, np.float32)
        _scatter_prediction(p3, pred3, sess, gid, id_index, independent, ('x', 'y', 'z'))
        _scatter_prediction(p2, pred2, sess, gid, id_index, independent, ('x', 'y'),
                             camera_axis=True)
        _scatter_prediction(p3, conf3, sess, gid, id_index, independent, ('score_logit',))
        _scatter_prediction(p2, conf2, sess, gid, id_index, independent,
                             ('visibility_logit',), camera_axis=True, camera_only=0)
        _scatter_prediction(inst, boxes, sess, gid, id_index, independent,
                             ('x0', 'y0', 'x1', 'y1'), camera_axis=True)
        _scatter_prediction(inst, box_agree, sess, gid, id_index, independent,
                             ('box_agree',), camera_axis=True)
        d = {'mode': sess.mode, 'group_id': gid, 'session':
             provenance.get('source_session_id', sess.session_id),
             'animal_ids': np.asarray(animal_ids, object),
             'independent_windows': independent}
        if sess.mode == '3d':
            d['pred'] = pred3
            if p2 is not None:
                d['pred2d'] = pred2
            d['conf'] = conf3
        else:
            d['pred'] = pred2[:, :, 0]
            d['pred2d'] = pred2
            d['conf'] = conf2
        if inst is not None and len(inst):
            d['boxes'] = boxes
            if np.isfinite(box_agree).any():
                d['box_agree'] = box_agree
        output[f'{d["session"]}/{gid}'] = d
    meta = {'run': str(provenance.get('run', '')),
            'anchor': str(provenance.get('anchor', '')),
            'boxes': str(provenance.get('boxes', '')),
            'box_source': str(provenance.get('box_source', '')),
            'prediction_session': True, 'independent_windows': independent}
    return output, meta


def _filter_group(table, gid, pc):
    """Select rows belonging to one group, preserving absent or empty tables."""
    if table is None or not len(table):
        return table
    return table.filter(pc.equal(table.column('group_id'), gid))


def _nullable_ints(table, col):
    """Read an integer column, replacing null entries with -1."""
    return np.asarray([-1 if v is None else int(v) for v in table.column(col).to_pylist()],
                      dtype=np.int64)


def _scatter_prediction(table, out, sess, gid, id_index, independent, fields,
                        camera_axis=False, camera_only=None):
    """Scatter one prediction table by animal id or explicit slot; coordinates keep NaNs."""
    if table is None or not len(table):
        return
    fields = (fields,) if isinstance(fields, str) else tuple(fields)
    if len(fields) > 1 and fields not in (('x', 'y'), ('x', 'y', 'z'),
                                          ('x0', 'y0', 'x1', 'y1')):
        raise ValueError(f'unsupported prediction fields {fields!r}')
    S, T, K = out.shape[0], out.shape[1], len(sess.names)
    frame = _nullable_ints(table, 'frame')
    if independent:
        animal = _nullable_ints(table, 'slot')
    else:
        animal = np.asarray([id_index.get(str(v), -1)
                             for v in table.column('animal_id').to_pylist()], np.int64)
    body = (np.asarray([sess.names.index(str(v)) if str(v) in sess.names else -1
                        for v in table.column('bodypart').to_pylist()], np.int64)
            if 'bodypart' in table.column_names else np.zeros(len(table), np.int64))
    valid = (animal >= 0) & (animal < S) & (frame >= 0) & (frame < T)
    if 'bodypart' in table.column_names:
        valid &= (body >= 0) & (body < K)
    if camera_axis:
        cams = np.asarray([sess.cam_names.index(str(v)) if str(v) in sess.cam_names else -1
                           for v in table.column('camera').to_pylist()], np.int64)
        valid &= (cams >= 0) & (cams < len(sess.cam_names))
        if camera_only is not None:
            valid &= cams == camera_only
    values = np.stack([_nullable_floats(table, f) for f in fields], axis=-1)
    if camera_axis:
        if camera_only is not None and out.ndim == 3:
            out[animal[valid], frame[valid], body[valid]] = values[valid, 0]
        elif out.ndim == 5:
            out[animal[valid], frame[valid], cams[valid], body[valid]] = values[valid]
        elif out.ndim == 4:
            out[animal[valid], frame[valid], cams[valid]] = values[valid]
        elif out.ndim == 3:
            out[animal[valid], frame[valid], cams[valid]] = values[valid, 0]
        else:
            raise ValueError(f'cannot scatter camera rows into {out.shape}')
    elif out.ndim == 4:
        out[animal[valid], frame[valid], body[valid]] = values[valid]
    elif out.ndim == 3:
        out[animal[valid], frame[valid], body[valid]] = values[valid, 0]
    else:
        raise ValueError(f'cannot scatter non-camera rows into {out.shape}')


def _nullable_floats(table, col):
    """Read a float column as float32, using NaN for null or absent values."""
    if col not in table.column_names:
        return np.full(len(table), np.nan, np.float32)
    return np.asarray([np.nan if v is None else float(v)
                       for v in table.column(col).to_pylist()], dtype=np.float32)


def _load_legacy_session(path, groups=None):
    """Densify a provenance-identified pre-marker prediction session.

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
            d['pred'] = (lab.points3d if lab.points3d is not None
                         else np.full((len(lab.animal_ids), sess.groups[gid].n_frames,
                                       len(sess.names), 3), np.nan, np.float32))
        else:
            d['pred'] = (lab.points2d[..., 0, :] if lab.points2d is not None
                         else np.full((len(lab.animal_ids), sess.groups[gid].n_frames,
                                       len(sess.names), 2), np.nan, np.float32))
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
