"""`validate_session` / `validate_dataset`: check a session or root against the spec, returning the
list of violations (docs/annotation_format.md rules)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from . import FormatError, VIDEO_EXTS
from .discovery import Dataset
from .session import Session, nonempty_box
from .tables import column_codes, column_floats


def _rows_without_3d(tk, t3, sel: np.ndarray) -> int:
    """How many `sel` rows of keypoints.pq have no points3d.pq row with the same
    (group_id, frame, animal_id, bodypart) key (rule 10's coordinate-free exemption)."""
    import pyarrow.compute as pc

    cols = ('group_id', 'frame', 'animal_id', 'bodypart')

    def keyed(table, mask=None) -> pa.Table:
        """The key columns of `table` (rows `mask`), dictionary-decoded so they can be joined."""
        table = getattr(table, 'table', table)
        if mask is not None:
            table = table.filter(pa.array(mask))
        out = {}
        for c in cols:
            arr = table.column(c)
            if pa.types.is_dictionary(arr.type):
                arr = pc.cast(arr, arr.type.value_type)
            out[c] = pc.cast(arr, pa.int64()) if c == 'frame' else pc.cast(arr, pa.string())
        return pa.table(out)

    if any(c not in t3.column_names for c in cols):
        return int(sel.sum())
    left = keyed(tk, sel)
    right = keyed(t3).group_by(list(cols)).aggregate([])
    return len(left.join(right, list(cols), join_type='left anti'))


def _check_video(path: Path, n_frames: int, size, where: str, bad) -> None:
    """Rules 7/8 for a video: its decodable frame count is `n_frames` and its frame size is the
    calibration `size`. Opens the container (never decodes a frame)."""
    from ..data import video
    try:
        reader = video.open_reader(str(path))
    except Exception as e:
        bad(7, f'{where}: cannot open {path.name}: {e}')
        return
    try:
        n = len(reader)
        h, w, _ = reader.frame_shape()
    finally:
        reader.close()
    if n != n_frames:
        bad(7, f'{where}: video has {n} frames, n_frames={n_frames}')
    if size and (w, h) != tuple(size):
        bad(8, f'{where}: video frame is {(w, h)}, calibration size is {tuple(size)}')


def _check_names(sess: Session, bad) -> None:
    """Rule 2: names are unique strings; skeleton/flip_pairs name known keypoints; flip is an involution."""
    if not all(isinstance(name, str) for name in sess.names):
        bad(2, 'names must contain only strings')
    if len({name for name in sess.names if isinstance(name, str)}) != sum(
            isinstance(name, str) for name in sess.names):
        bad(2, 'names has duplicates')
    known = {name for name in sess.names if isinstance(name, str)}
    valid_flip_pairs = []
    for kind, pairs in (('skeleton', sess.skeleton), ('flip_pairs', sess.flip_pairs)):
        for i, pair in enumerate(pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                bad(2, f'{kind}[{i}] must be a [name, name] pair')
                continue
            if not all(isinstance(n, str) for n in pair):
                bad(2, f'{kind}[{i}] names must be strings')
                continue
            for n in pair:
                if n not in known:
                    bad(2, f'{kind} names unknown keypoint {n!r}')
            if kind == 'flip_pairs':
                valid_flip_pairs.append((pair[0], pair[1], i))
    flip: dict[str, str] = {}
    seen_pairs: set[frozenset[str]] = set()
    for a, b, i in valid_flip_pairs:
        if a == b:
            bad(2, f'flip_pairs has a self-pair at {a!r}')
            continue
        if a not in known or b not in known:
            continue
        pair_key = frozenset((a, b))
        if pair_key in seen_pairs:
            bad(2, f'flip_pairs lists the pair {a!r}, {b!r} more than once')
        seen_pairs.add(pair_key)
        for x, y in ((a, b), (b, a)):
            if flip.setdefault(x, y) != y:
                bad(2, f'flip_pairs is not an involution: {x!r} maps to both '
                       f'{flip[x]!r} and {y!r}')


def _check_rig(sess: Session, bad) -> None:
    """Rules 4/5: every camera is named and sized; 3D needs >= 2 calibrated cameras, 2D exactly one."""
    for name in sess.cam_names:
        if not name:
            bad(4, 'a camera has an empty name')
        if not sess.rig.size(name):
            bad(4, f'camera {name!r} has no size')
    if sess.mode == '3d':
        if len(sess.rig) < 2:
            bad(5, f'mode=3d with {len(sess.rig)} camera(s)')
        for name in sess.cam_names:
            if not sess.rig.calibrated[name]:
                bad(5, f'mode=3d but camera {name!r} has no matrix/rotation/translation')
    elif len(sess.rig) != 1:
        bad(5, f'mode=2d with {len(sess.rig)} cameras')


def _check_keys(sess: Session, bad) -> None:
    """Rules 9/6: no missing key columns or duplicate keys; every label-table `group_id` is in groups.pq."""
    keys = {'keypoints': ('group_id', 'frame', 'animal_id', 'camera', 'bodypart'),
            'points3d': ('group_id', 'frame', 'animal_id', 'bodypart'),
            'instances': ('group_id', 'frame', 'animal_id', 'camera'),
            'extrinsics': ('group_id', 'frame', 'camera')}
    for stem, cols in keys.items():
        table = sess._tables[stem]
        if table is None:
            continue
        missing = [c for c in cols if c not in table.column_names]
        if missing:
            bad(9, f'{stem}.pq missing columns {missing}')
            continue
        n = len(table)
        if n and len(table.select(cols).group_by(list(cols)).aggregate([])) != n:
            bad(9, f'{stem}.pq has duplicate keys')

    for stem in ('keypoints', 'points3d', 'instances', 'extrinsics'):
        table = sess._tables[stem]
        if table is None or not len(table) or 'group_id' not in table.column_names:
            continue
        codes, values = column_codes(table, 'group_id')
        unknown = sorted({values[c] for c in np.unique(codes)} - set(sess.groups))
        if unknown:
            bad(6, f'{stem}.pq has {len(unknown)} group_id(s) not in groups.pq, e.g. '
                   f'{unknown[0]!r}; their rows are never read')


def _check_points2d(sess: Session, bad) -> None:
    """Rule 9: `points2d.pq` is prediction-only, keyed, and names only known groups/cameras/bodyparts/frames."""
    t2p = sess._tables['points2d']
    if t2p is not None:
        if not sess.prediction_session:
            bad(9, 'points2d.pq is prediction-only; set prediction_session = true')
        else:
            required = ('group_id', 'frame', 'animal_id', 'camera', 'bodypart', 'x', 'y',
                        'visibility_logit', 'visibility_probability', 'confidence_logit',
                        'confidence_probability', 'slot', 'window', 'window_start',
                        'window_stop')
            missing = [c for c in required if c not in t2p.column_names]
            if missing:
                bad(9, f'points2d.pq missing columns {missing}')
            else:
                point_keys = ('group_id', 'frame', 'animal_id', 'camera', 'bodypart')
                n = len(t2p)
                for key in point_keys:
                    if t2p.column(key).null_count:
                        bad(9, f'points2d.pq has null {key} values')
                if n and len(t2p.select(point_keys).group_by(list(point_keys)).aggregate([])) != n:
                    bad(9, 'points2d.pq has duplicate keys')
                groups = set(sess.groups)
                gvals = t2p.column('group_id').to_pylist()
                frames = t2p.column('frame').to_pylist()
                cameras = t2p.column('camera').to_pylist()
                bodyparts = t2p.column('bodypart').to_pylist()
                animals = t2p.column('animal_id').to_pylist()
                unknown_group = next((g for g in gvals if g not in groups), None)
                if unknown_group is not None:
                    bad(9, f'points2d.pq has unknown group_id {unknown_group!r}')
                unknown_camera = next((c for c in cameras if c not in sess._cam_vocab), None)
                if unknown_camera is not None:
                    bad(9, f'points2d.pq has unknown camera {unknown_camera!r}')
                unknown_bodypart = next((k for k in bodyparts if k not in sess._kpt_vocab), None)
                if unknown_bodypart is not None:
                    bad(9, f'points2d.pq has unknown bodypart {unknown_bodypart!r}')
                bad_frame = next(((g, f) for g, f in zip(gvals, frames)
                                  if g in groups and (f is None or f < 0 or
                                                      f >= sess.groups[g].n_frames)), None)
                if bad_frame is not None:
                    bad(9, f'points2d.pq has frame {bad_frame[1]!r} outside group '
                        f'{bad_frame[0]!r}')
                if any(a is None for a in animals):
                    bad(9, 'points2d.pq has a null animal_id')


def _check_coords(sess: Session, bad) -> None:
    """Rule 10: a positioned row carries its coordinates (a coordinate-free keypoints row only with a
    points3d row for the same key); a missing/unlabeled row carries none; points3d is never `projected`.
    """
    positioned = ('visible', 'projected')
    t3 = sess._tables['points3d']
    if t3 is not None and len(t3):
        st, vals = column_codes(t3, 'status')
        used = {vals[c] for c in np.unique(st)}
        if 'projected' in used:
            bad(10, 'points3d.pq has a status=projected row; the 3D layer is visible | missing | '
                    'unlabeled (projected is a per-camera 2D state)')
        vis = np.isin(st, [i for i, v in enumerate(vals) if v in positioned])
        xyz = np.stack([column_floats(t3, c, len(t3)) for c in 'xyz'], -1)
        if vis.any() and not np.isfinite(xyz[vis]).all():
            bad(10, 'points3d.pq has a visible row without x,y,z')
        if (~vis).any():
            positioned_status = np.isin(st, [i for i, v in enumerate(vals) if v in positioned])
            prediction_missing = (sess.prediction_session
                                  & np.isin(st, [i for i, v in enumerate(vals)
                                                 if v == 'missing']))
            forbidden = ~positioned_status & ~prediction_missing
            if np.any(forbidden & np.isfinite(xyz).any(-1)):
                bad(10, 'points3d.pq has a missing/unlabeled row carrying coordinates')

    tk = sess._tables['keypoints']
    if tk is not None and len(tk):
        st, vals = column_codes(tk, 'status')
        vis = np.isin(st, [i for i, v in enumerate(vals) if v in positioned])
        xy = np.stack([column_floats(tk, c, len(tk)) for c in 'xy'], -1)
        free = vis & ~np.isfinite(xy).all(-1)
        if free.any() and t3 is None:
            bad(10, 'keypoints.pq has a visible row without x,y and there is no 3D layer')
        elif free.any():
            n = _rows_without_3d(tk, t3, free)
            if n:
                bad(10, f'keypoints.pq has {n} positioned row(s) without x,y and no points3d row '
                        'for the same (group_id, frame, animal_id, bodypart)')
        if (~vis).any() and np.isfinite(xy[~vis]).any():
            bad(10, 'keypoints.pq has a missing/unlabeled row carrying coordinates')


def _check_instances(sess: Session, bad) -> None:
    """Rules 6/11: no stale `regions.pq`; a `labeled` instance row has a finite, non-empty box
    (prediction sessions exempt).
    """
    if (sess.path / 'regions.pq').exists():
        bad(6, 'regions.pq is no longer part of the format; mark an incompletely labelled '
               'camera-frame with a `present` instances.pq row instead')

    ti = sess._tables['instances']
    if ti is not None and len(ti) and not sess.prediction_session:
        st, vals = column_codes(ti, 'status')
        n = len(ti)
        labeled = np.isin(st, [i for i, v in enumerate(vals) if v == 'labeled'])
        box = np.stack([column_floats(ti, q, n) for q in ('x0', 'y0', 'x1', 'y1')], -1)
        m = int((labeled & ~nonempty_box(box)).sum())
        if m:
            bad(11, f'instances.pq has {m} labeled row(s) without a finite, non-empty box')


def _check_extrinsics(sess: Session, bad) -> None:
    """Rule 13: extrinsics only for moving cameras, and every frame of every moving camera (a missing
    frame would read as a pose at the world origin).
    """
    te = sess._tables['extrinsics']
    moving = {n for n in sess.cam_names if sess.rig.moving[n]}
    if te is not None and len(te):
        _, cams = column_codes(te, 'camera')
        for name in cams:
            if name not in moving:
                bad(13, f'extrinsics.pq names camera {name!r}, which is not moving=true')
        gcodes, gvals = column_codes(te, 'group_id')
        frames = te.column('frame').combine_chunks().to_numpy(zero_copy_only=False)
        ccodes, cvals = column_codes(te, 'camera')
        for gid, g in (sess.groups.items() if moving else ()):
            if gid not in gvals:
                bad(13, f'group {gid!r}: cameras {sorted(moving)} are moving=true but '
                        f'extrinsics.pq has no rows for it')
                continue
            sel = gcodes == gvals.index(gid)
            for name in sorted(moving):
                if name not in cvals:
                    bad(13, f'group {gid!r}: no extrinsics for moving camera {name!r}')
                    continue
                n = len(np.unique(frames[sel & (ccodes == cvals.index(name))]))
                if n != g.n_frames:
                    bad(13, f'group {gid!r} camera {name!r}: extrinsics for {n} of '
                            f'{g.n_frames} frames')
    elif moving:
        bad(13, f'cameras {sorted(moving)} are moving=true but extrinsics.pq is absent/empty')


def _check_groups(sess: Session, bad, errs: list[str], check_images: bool) -> None:
    """Rules 4/6/7/8 per group: no undeclared camera dirs/videos, pixels present, contiguous and the
    right size (video frame count/size with `check_images`); `sess.labels(gid)` must load.
    """
    cams = set(sess.cam_names)
    for gid, g in sess.groups.items():
        if g.dir.is_dir():
            extra = sorted(e.name for e in g.dir.iterdir()
                           if (e.is_dir() and e.name not in cams)
                           or (e.suffix in VIDEO_EXTS and e.stem not in cams))
            if extra:
                bad(4, f'group {gid!r}: camera dirs/videos not in calibration.toml: {extra}')
        for cname in sess.cam_names:
            try:
                kind, p = g.pixels(cname)
            except FormatError as e:
                bad(7, str(e))
                continue
            if kind != 'frames':
                if check_images:
                    _check_video(p, g.n_frames, sess.rig.size(cname),
                                 f'group {gid!r} camera {cname!r}', bad)
                continue
            files = g.frame_paths(cname)
            if len(files) != g.n_frames:
                bad(7, f'group {gid!r} camera {cname!r}: {len(files)} files, '
                       f'n_frames={g.n_frames}')
            exts = {f.suffix for f in files}
            if len(exts) > 1:
                bad(7, f'group {gid!r} camera {cname!r}: mixed extensions {sorted(exts)}')
            if files and [f.stem for f in files] != [f'{i:06d}' for i in range(len(files))]:
                bad(7, f'group {gid!r} camera {cname!r}: names are not contiguous %06d')
            if check_images and files:
                from PIL import Image
                with Image.open(files[0]) as im:
                    want = sess.rig.size(cname)
                    if tuple(im.size) != want:
                        bad(8, f'group {gid!r} camera {cname!r}: image is {im.size}, '
                               f'calibration size is {want}')
        try:
            sess.labels(gid)
        except FormatError as e:
            errs.append(str(e))


def validate_session(sess: Session, check_images: bool = True) -> list[str]:
    """Return a list of rule violations. Empty means the session is valid.

    The checks follow docs/annotation_format.md §11, one helper per rule group (each names its
    rules), in a fixed order so the message list is stable. Label tables are optional;
    `points2d.pq` is accepted only for marked prediction sessions.
    """
    errs: list[str] = []
    here = str(sess.path)

    def bad(rule: int, msg: str) -> None:
        """Record one rule violation against this session."""
        errs.append(f'{here}: [rule {rule}] {msg}')

    _check_names(sess, bad)
    _check_rig(sess, bad)
    _check_keys(sess, bad)
    _check_points2d(sess, bad)
    _check_coords(sess, bad)
    _check_instances(sess, bad)
    _check_extrinsics(sess, bad)
    _check_groups(sess, bad, errs, check_images)
    return errs


def validate_dataset(ds: Dataset, check_images: bool = True) -> list[str]:
    """Validate every session in a dataset, plus the cross-session rules.

    Inputs: ds -- the dataset to validate.
            check_images -- also verify each frame image's size against calibration.
    Outputs: a list of rule violations; empty means the dataset is valid.

    The cross-session checks are rule 3 (cross-session agreement on names -- a WARNING, not an
    error: `Registry.ids_for` resolves a session's axis BY NAME, but a "missing" keypoint is
    usually a typo, not a decision) and rule 14 (a leak: a session folder name used in more
    than one split).
    """
    errs: list[str] = []
    sessions = ds.all_sessions()
    if not sessions:
        return [f'{ds.root}: no sessions']
    axis = ds.names
    for s in sessions:
        if s.names == axis:
            continue
        missing = [n for n in axis if n not in set(s.names)]
        what = (f'is missing {len(missing)} of the root\'s {len(axis)} keypoints ({missing})'
                if missing else 'declares the root\'s keypoints in a different order')
        errs.append(f'{s.path}: [rule 3 WARNING] {what}; resolved by name')
    seen: dict[str, str] = {}
    for split, group in ds.sessions.items():
        for s in group:
            if s.session_id in seen and seen[s.session_id] != split:
                errs.append(f'{ds.root}: [rule 14 WARNING] session {s.session_id!r} appears in '
                            f'both {seen[s.session_id]!r} and {split!r}')
            seen[s.session_id] = split
    for s in sessions:
        errs.extend(validate_session(s, check_images=check_images))
    return errs
