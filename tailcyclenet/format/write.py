"""Write a session to disk (`write_session`, which refuses what the spec forbids) and `link` pixels."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import FormatError, INST_NONE, INST_STATUS, KPT_STATUS, LABEL_SOURCES, UNLABELED
from .calibration import Rig, dump_calibration
from .session import Group, Labels, boxless_labeled
from .tables import DICT_COLS, invert_enum, write_table


def write_session(path: Path, *, mode: str, units: str, label_source: str, names: list[str],
                  rig: Rig, groups: dict[str, Group], labels: dict[str, Labels],
                  skeleton=(), flip_pairs=None, provenance=None,
                  assoc_res_max_px: float | None = None) -> None:
    """Write session.toml, calibration.toml and the label tables. Pixels are the caller's job.

    A row is emitted only where a status says a determination was made, so sparse hand
    annotation and dense tracking are the same code path. `label_source` becomes the `labels`
    key of session.toml and is required, not defaulted.

    Rule 6 requires the mode's own tables to EXIST, so `points3d.pq`/`keypoints.pq` are emitted
    when the corresponding array is present, not when any row is labelled.

    Rule 11 is refused here rather than written and caught later: a `labeled` instance row must
    carry a finite, non-empty box.
    """
    import toml

    if label_source not in LABEL_SOURCES:
        raise FormatError(f'{path}: label_source must be one of {LABEL_SOURCES}, '
                          f'got {label_source!r}')
    for gid, lab in labels.items():
        if lab is None or lab.instance is None:
            continue
        n = boxless_labeled(lab.instance, lab.boxes)
        if n:
            raise FormatError(f'{path}: group {gid!r} has {n} labeled instance row(s) without a '
                              f'finite, non-empty box (rule 11); write `present` for an animal '
                              f'whose box was not annotated')
    path.mkdir(parents=True, exist_ok=True)
    cfg = {'mode': mode, 'units': units, 'labels': label_source, 'names': list(names)}
    if skeleton:
        cfg['skeleton'] = [list(p) for p in skeleton]
    if flip_pairs is not None:
        cfg['flip_pairs'] = [list(p) for p in flip_pairs]
    if assoc_res_max_px is not None:
        cfg['assoc_res_max_px'] = float(assoc_res_max_px)
    cfg['provenance'] = dict(provenance or {})
    (path / 'session.toml').write_text(toml.dumps(cfg))
    dump_calibration(path / 'calibration.toml', rig)

    order = list(groups)
    write_table(path / 'groups.pq', {
        'group_id': np.array(order, dtype=object),
        'n_frames': np.array([groups[g].n_frames for g in order], np.int32),
        'fps': np.array([groups[g].fps for g in order], np.float32),
        'source_video': np.array([groups[g].source_video for g in order], dtype=object),
        'source_frame_start': np.array([groups[g].source_frame_start for g in order], np.int32),
        'source_frame_step': np.array([groups[g].source_frame_step for g in order], np.int32),
        'notes': np.array([groups[g].notes for g in order], dtype=object),
    }, dict_cols=())

    cam_names = rig.names
    kpt = invert_enum(KPT_STATUS)
    inst = invert_enum(INST_STATUS)
    tables: dict[str, dict[str, list]] = {
        'points3d': {c: [] for c in
                     ('group_id', 'frame', 'animal_id', 'bodypart', 'status', 'x', 'y', 'z')},
        'keypoints': {c: [] for c in
                      ('group_id', 'frame', 'animal_id', 'camera', 'bodypart', 'status', 'x', 'y')},
        'instances': {c: [] for c in
                      ('group_id', 'frame', 'animal_id', 'camera', 'x0', 'y0', 'x1', 'y1',
                       'status')},
        'extrinsics': {c: [] for c in ('group_id', 'frame', 'camera', 'ext')},
    }
    emit_points3d = any(lab is not None and lab.vis3d is not None for lab in labels.values())
    emit_keypoints = any(lab is not None and lab.vis2d is not None for lab in labels.values())

    def push(name, **cols):
        """Append one row's columns to a table's buffers."""
        for k, v in cols.items():
            tables[name][k].extend(v)

    for gid in order:
        lab = labels.get(gid)
        if lab is None:
            continue
        aid = np.asarray(lab.animal_ids, dtype=object)

        if lab.vis3d is not None:
            s, t, k = np.nonzero(lab.vis3d != UNLABELED)
            xyz = lab.points3d[s, t, k]
            push('points3d', group_id=[gid] * len(s), frame=t.astype(np.int32),
                 animal_id=aid[s], bodypart=np.asarray(names, dtype=object)[k],
                 status=[kpt[v] for v in lab.vis3d[s, t, k]],
                 x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])

        if lab.vis2d is not None:
            s, t, k, c = np.nonzero(lab.vis2d != UNLABELED)
            xy = lab.points2d[s, t, k, c]
            push('keypoints', group_id=[gid] * len(s), frame=t.astype(np.int32),
                 animal_id=aid[s], camera=np.asarray(cam_names, dtype=object)[c],
                 bodypart=np.asarray(names, dtype=object)[k],
                 status=[kpt[v] for v in lab.vis2d[s, t, k, c]], x=xy[:, 0], y=xy[:, 1])

        if lab.instance is not None:
            s, t, c = np.nonzero(lab.instance != INST_NONE)
            box = lab.boxes[s, t, c]
            push('instances', group_id=[gid] * len(s), frame=t.astype(np.int32),
                 animal_id=aid[s], camera=np.asarray(cam_names, dtype=object)[c],
                 x0=box[:, 0], y0=box[:, 1], x1=box[:, 2], y1=box[:, 3],
                 status=[inst[v] for v in lab.instance[s, t, c]])

        if lab.ext is not None:
            for ci, name in enumerate(cam_names):
                if not rig.moving[name]:
                    continue
                T = lab.ext.shape[1]
                push('extrinsics', group_id=[gid] * T, frame=np.arange(T, dtype=np.int32),
                     camera=[name] * T, ext=[e.ravel().tolist() for e in lab.ext[ci]])

    for stem, cols in tables.items():
        if not cols['group_id'] and not (stem == 'points3d' and emit_points3d) \
                and not (stem == 'keypoints' and emit_keypoints):
            continue
        write_table(path / f'{stem}.pq',
                    {k: np.asarray(v, dtype=object if k in DICT_COLS else None)
                     if k != 'ext' else v for k, v in cols.items()},
                    dict_cols=DICT_COLS)


def link(dst: Path, src: Path) -> None:
    """Replace `dst` with a symlink to `src`. Every converter links pixels rather than copying."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)
