"""THE prediction-session reader: a marked prediction session (or a legacy `.npz` archive) ->
dense per-group arrays. `evaluation`, `render` and `infer.bridge` read predictions only through
`load_predictions`; the writer (`infer`'s `SessionWriter`) is the other half of the contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .session import Session


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
    """Read a marked prediction session; a pre-marker or window-local one is refused by name."""
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
        raise ValueError(
            f'{path}: a pre-marker prediction session (no `prediction_session = true`). Its '
            'reader was deleted on 2026-10-19; the last release that reads it is v0.3.4.')
    raise ValueError(f'{path}: not a marked prediction session; refusing ambiguous annotation data')


def _load_prediction_session(path, sess, cfg, groups=None):
    """Read primary prediction rows into dense per-animal arrays."""
    import pyarrow.compute as pc

    provenance = cfg.get('provenance', {})
    if provenance.get('independent_windows', False):
        raise ValueError(
            f'{path}: written by --independent-windows (window-local animal ids), which was '
            'deleted on 2026-10-19; the last release that reads it is v0.3.4.')
    p3_all = _table(path, 'points3d')
    p2_all = _table(path, 'points2d')
    inst_all = _table(path, 'instances')
    wanted = set(groups) if groups is not None else None
    output = {}
    for gid, group in sess.groups.items():
        if wanted is not None and gid not in wanted:
            continue
        T, K, C = group.n_frames, len(sess.names), len(sess.cam_names)
        p3 = _filter_group(p3_all, gid, pc)
        p2 = _filter_group(p2_all, gid, pc)
        inst = _filter_group(inst_all, gid, pc)
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
        _scatter_prediction(p3, pred3, sess, gid, id_index, ('x', 'y', 'z'))
        _scatter_prediction(p2, pred2, sess, gid, id_index, ('x', 'y'),
                             camera_axis=True)
        _scatter_prediction(p3, conf3, sess, gid, id_index, ('score_logit',))
        _scatter_prediction(p2, conf2, sess, gid, id_index,
                             ('visibility_logit',), camera_axis=True, camera_only=0)
        _scatter_prediction(inst, boxes, sess, gid, id_index,
                             ('x0', 'y0', 'x1', 'y1'), camera_axis=True)
        _scatter_prediction(inst, box_agree, sess, gid, id_index,
                             ('box_agree',), camera_axis=True)
        d = {'mode': sess.mode, 'group_id': gid, 'session':
             provenance.get('source_session_id', sess.session_id),
             'animal_ids': np.asarray(animal_ids, object)}
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
            'prediction_session': True,
            'frame_start': int(provenance.get('frame_start', 0) or 0),
            'frame_stop': int(provenance.get('frame_stop', 0)
                              or provenance.get('max_frames', 0) or 0)}
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


def _scatter_prediction(table, out, sess, gid, id_index, fields,
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


def _table(path: Path, stem: str):
    """`path/{stem}.pq` as a pyarrow table, or None if the file does not exist."""
    import pyarrow.parquet as pq
    f = path / f'{stem}.pq'
    return pq.read_table(f) if f.exists() else None
