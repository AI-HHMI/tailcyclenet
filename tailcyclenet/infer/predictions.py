"""A prediction IS a session in this repo's own format, written a block at a time.

`docs/annotation_format.md` already describes exactly what a prediction is: a `labels = "tracked"`
session -- long parquet tables, one row per assessed thing, `status` as the visibility channel,
boxes in `instances.pq`. Writing predictions that way means `format.Session` reads them back,
MATLAB's `parquetread` reads them, and a prediction directory is a first-class citizen of every
tool here rather than a private npz schema nothing else understands.

**`--out` NAMES THE SESSION DIRECTORY ITSELF.** A session holds ONE calibration, one `mode` and one
keypoint axis, and sessions under a root need agree on none of them -- so a run covering more than
one source session is refused rather than merged. `format.sessions_for` already accepts a bare
session directory, so `--data <one session>` is the answer and not a workaround.

**NO PIXELS AND NO `groups/`.** The output is toml and parquet, a few MB beside the source's
hundreds of GB. The consequence is honest and worth knowing: the directory is NOT self-contained,
and `validate_session` reports one rule-7 error per (group, camera) because of it. Everything that
does not need pixels passes, which is what the test asserts.

**WHERE THE PIXELS ARE IS `[provenance]`, AND `scripts/render.py` NOW READS IT.**
`render.session_for_prediction` opens `source_session` (a directory input) or the
`source_videos`/`source_calibration`/`source_cam_regex`/`source_group_id` quadruple (a `--videos`
input), the latter via `adopt.session_from_prediction`, which CHECKS itself against `groups.pq`
and `calibration.toml`. `--data` is an OVERRIDE for a root that MOVED, not the normal input, and
is checked against `source_session_id` rather than trusted. `render.py` no longer reads an npz at
all -- an npz carries none of this, so every npz render was a render against a hand-restated root,
which is exactly the failure this closed. `predictions.py` itself reads only `source_session_id`,
to confirm whatever session `render.py` found is the one this prediction was written over.

THREE COLUMNS ARE ADDITIONS TO THE SPEC, all additive: `score` and `box_agree` on `instances.pq`
(the detector's objectness and the pose-to-box distance, both per (animal, frame, camera), exactly
that table's grain), and `score_logit` beside `score` on the two keypoint tables. The last one is
not redundancy: `score` is the spec's `[0, 1]`, so it is `sigmoid(vis_pred)`, and `vis_pred` runs
to a median of +15.4 on one root, where `sigmoid` rounds to exactly 1.0 in float32 and cannot be
inverted. `--vis-thresh` and `eval.py`'s confusion matrix both read the logit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..format import DICT_COLS, Session, TableWriter, dump_calibration, sessions_for, write_table

# The tables a prediction writes. `windows.pq` is NOT a spec table -- it is per (animal, window,
# camera) diagnostics, named so it cannot be mistaken for one.
_TABLES = ('points3d', 'keypoints', 'instances', 'windows')


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


class SessionWriter:
    """One prediction session, appended a block at a time.

    The header -- `session.toml`, `calibration.toml`, `groups.pq` -- is written UP FRONT, before
    any inference runs, so a run that dies half way leaves a directory that says what it was
    rather than a pile of anonymous parquet.
    """

    def __init__(self, out: Path, source: Session, registry, provenance: dict, groups):
        self.out = Path(out)
        self.src = source
        self.names = list(registry.names) if hasattr(registry, 'names') else list(source.names)
        self.out.mkdir(parents=True, exist_ok=True)
        self._w = {t: TableWriter(self.out / f'{t}.pq', DICT_COLS) for t in _TABLES}

        # THE PROVENANCE CONTRACT IS SCALARS AND LISTS OF STRINGS. Everything here was a scalar
        # until `source_videos` (the resolved `--videos` file list, one entry per file); a TOML
        # array of strings round-trips through `toml.dumps`/`tomllib` unchanged and the `!=` below
        # compares lists correctly, but it is a first and the next caller should not have to
        # discover which by experiment.
        #
        # A DUPLICATE PROVENANCE KEY IS A SILENT LOSS, and it has already happened once: the
        # caller builds this dict by splatting `_box_provenance` over its own literals, and a name
        # in both meant the splat order decided which fact survived -- `box_source` (the run's
        # crop rule) was overwritten by the detector's training target, which is a different
        # statement entirely. `dict` merges without complaint, so the caller passes ITEMS and this
        # is where the collision is caught.
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
        """One block's rows. `f0`/`w0` are its first frame and window in the WHOLE group."""
        ids = [str(x) for x in blk['animal_ids']]
        cams = self.src.cam_names
        kpts = list(self.src.names)
        pred, conf = np.asarray(blk['pred']), np.asarray(blk['conf'])
        S, T, K = pred.shape[0], pred.shape[1], pred.shape[2]
        if not (S and T and K):
            return
        a_ix, t_ix, k_ix = (x.ravel() for x in np.meshgrid(np.arange(S), np.arange(T),
                                                           np.arange(K), indexing='ij'))
        # THE SPEC'S OWN SPARSITY RULE: "No row -- not labelled." A prediction that declined a
        # point writes nothing, which is what every consumer here already reads as absence. On the
        # deployment path coverage runs ~0.57, so this is most of the file.
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

        # THE PER-CAMERA 2D POSE. In 2D this is the prediction itself (`coords_pred` IS
        # `2d_pred[0]`); in 3D it is the overlay a run used to discard.
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

        # `instances.pq`: the box, its objectness and the pose-to-box distance, all per
        # (animal, frame, camera) -- exactly this table's grain, and all describing THIS box.
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

        # `windows.pq`: why a window produced nothing, and what box it was given. Per
        # (animal, window, camera) -- deliberately NOT a spec table.
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

    def close(self):
        for w in self._w.values():
            w.close()


def load_predictions(path, groups=None):
    """A prediction session or a legacy npz -> `({key: {field: ndarray}}, meta)`.

    THE NPZ HALF IS NOT LEGACY DEBT, it is the archive: every number this repo has published lives
    in one, and dropping the reader would make them unscoreable. It is the same argument that kept
    `eval.py` able to read them at all -- `scripts/render.py` is what stopped reading it (an npz
    carries no provenance to find its pixels with); scoring did not.

    `groups`, given, restricts the read to those group ids (bare, not `session/group` keys).
    `eval.py` leaves it `None` and gets every group, unchanged. A RENDER wants this: the session
    half calls `Session.preload()` no longer -- it scatters one group at a time -- so a caller
    asking for one group of a many-hundred-group clip never pays for the rest.
    """
    path = Path(path)
    return (_load_npz(path, groups) if path.suffix == '.npz' else _load_session(path, groups))


def _load_npz(path, groups=None):
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

    **THE SCATTER IS `format.Session`'s OWN**, not a second implementation: §12 of the spec already
    defines how long tables become `(S,T,K,3)` dense arrays, `Labels` already returns exactly that,
    and it is already tested. This function renames those fields and adds the two tables the spec
    does not define.

    **NO `preload()`.** `Session.labels` caches PER GROUP, so scattering is already lazy; the old
    `preload()` call scattered every group up front regardless of `groups`, which is the whole of
    what made a render of one group out of a many-hundred-group clip cost the other 499.
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
            # 2D: the prediction IS the per-camera pose at camera 0 (`coords_pred` is
            # `2d_pred[0]`), which is what `keypoints.pq` holds.
            d['pred'] = lab.points2d[..., 0, :]
        if lab.points2d is not None:
            d['pred2d'] = np.moveaxis(lab.points2d, 3, 2)          # (S,T,K,C,2) -> (S,T,C,K,2)
        if lab.boxes is not None:
            d['boxes'] = lab.boxes
        # `conf` is the LOGIT, from the additive `score_logit` column -- `sigmoid` cannot be
        # inverted in float32 once it saturates, which it does at the medians this repo measures.
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
    import pyarrow.parquet as pq
    f = path / f'{stem}.pq'
    return pq.read_table(f) if f.exists() else None


def _score_logit(path, gid, sess, lab, stem):
    """`(S,T,K)` of the visibility logit, scattered from the additive `score_logit` column."""
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
        # One row per camera; `conf` is per keypoint. Camera 0 is the 2D prediction's own head.
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
    """One session per run, checked BEFORE the checkpoint loads. -> the single session.

    A session directory holds one calibration, one `mode` and one keypoint axis; sessions under a
    root need agree on none of them, so a prediction session cannot represent several at once.
    `sessions_for` already accepts a bare session directory, which is the whole answer.
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
