"""`SessionWriter`: a prediction session written a block at a time.

The top-level marker distinguishes prediction-only `points2d.pq` from annotation tables; its
rows carry stable keypoint and camera names plus slot/window ownership. Primary 3D predictions,
per-camera 2D poses, detector instances and window diagnostics are separate typed tables. There
are no pixels and no `groups/`; `[provenance]` identifies the source for rendering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..format import DICT_COLS, Session, TableWriter, dump_calibration, write_table
from .window_records import WindowPredictionWriter


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


def _row_ids(ids, slots):
    """Each row's animal id: the linked id of its slot."""
    return np.asarray([ids[int(s)] for s in slots], dtype=object)


class SessionWriter:
    """One prediction session, appended a block at a time.

    The header (`session.toml`, `calibration.toml`, `groups.pq`, and `extrinsics.pq` when the
    source rig has a moving camera) is written up front, so a run that dies half way leaves a
    directory that says what it was.
    """

    def __init__(self, out: Path, source: Session, registry, provenance: dict, groups, *,
                 quality_enabled=False):
        """Open a prediction session for writing: header first, parquet writers for every table.

        Inputs: out -- output session directory (created here).
                source -- the source session (calibration, mode, keypoint axis).
                registry -- keypoint registry; not used for the tables: every table writes
                    the SOURCE session's `names`, the axis the predictions are indexed on.
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
        self.window_predictions = bool(self.provenance.get('window_predictions', False))
        self.quality_enabled = bool(quality_enabled)
        self.window_length = int(self.provenance.get('n_frames', 0) or 0)
        self.frame_stop = int(self.provenance.get('frame_stop', 0) or 0)
        self._sidecar = (WindowPredictionWriter(
            self.out / 'window_predictions.pq', quality_enabled=self.quality_enabled)
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
                    'animal_id': _row_ids(ids, a_ix[keep]),
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
                                       np.asarray(blk['triangulated'])[..., 2].ravel()[keep].astype(np.float32)),
                    **({} if blk.get('quality') is None else {
                        'quality_score': np.asarray(blk['quality']).ravel()[keep].astype(np.float32),
                        'quality_precision': np.asarray(blk['quality_precision'])
                        .ravel()[keep].astype(np.float32)})})

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
                'animal_id': _row_ids(ids, a2[keep2]),
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
                'window_stop': owner_stop[t2[keep2]].astype(np.int32),
                **({} if pred.shape[-1] != 2 or blk.get('quality') is None else {
                    'quality_score': np.broadcast_to(
                        np.asarray(blk['quality'])[:, :, None, :], p2.shape[:-1])
                    .ravel()[keep2].astype(np.float32),
                    'quality_precision': np.broadcast_to(
                        np.asarray(blk['quality_precision'])[:, :, None, :], p2.shape[:-1])
                    .ravel()[keep2].astype(np.float32)})})

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
                    'animal_id': _row_ids(ids, ai[have]),
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
                'animal_id': _row_ids(ids, aw),
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
        cc = blk.get('carry_crop')
        if cc is not None:
            rows['carry_crop'] = np.asarray(cc).ravel().astype(np.int32)
        iu = blk.get('crop_inflate_used')
        if iu is not None:
            rows['crop_inflate_used'] = np.asarray(iu).ravel().astype(np.float32)
        ro = blk.get('refine_outlier')
        if ro is not None:
            rows['refine_outlier'] = np.asarray(ro).ravel().astype(np.int32)
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
