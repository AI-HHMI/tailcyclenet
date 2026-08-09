"""Read, write and validate the `tailcycle-dataset` format.

`docs/annotation_format.md` is the spec; this module implements it. When the two disagree, the
document wins.

The read path turns tidy-long parquet into dense arrays ONCE per group, in whatever process calls
it, so forked dataloader workers share the pages copy-on-write. `status` is the visibility
channel and is dictionary-encoded in parquet, which means "is this point visible" is an int8
compare on the dictionary codes rather than a string comparison -- the reason the format uses
parquet at all.

Calibration is written in aniposelib's `CameraGroup.dump` layout (so anipose can read it) but
parsed here directly, because this format adds four per-camera keys (`type`, `offset`,
`image_size`, `moving`) and going through aniposelib's loader would make their survival depend on
its tolerance for unknown keys.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Visibility codes, shared by points3d and keypoints. -1 doubles as "no row".
UNLABELED, MISSING, VISIBLE = -1, 0, 1
# Instance codes. -1 doubles as "no row" == no determination.
INST_NONE, INST_ABSENT, INST_PRESENT, INST_LABELED = -1, 0, 1, 2

KPT_STATUS = {'unlabeled': UNLABELED, 'missing': MISSING, 'visible': VISIBLE}
INST_STATUS = {'absent': INST_ABSENT, 'present': INST_PRESENT, 'labeled': INST_LABELED}
IMAGE_EXTS = ('.png', '.jpg')
VIDEO_EXTS = ('.mp4', '.avi')
SPLITS = ('train', 'val', 'test')


class FormatError(Exception):
    """A dataset on disk does not satisfy docs/annotation_format.md."""


# --------------------------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------------------------

@dataclass
class Camera:
    """One camera. `size` is the SENSOR; `image_size` is what is on disk at `offset` inside it."""
    name: str
    type: str                       # 'pinhole' | 'fisheye'
    size: tuple[int, int]           # sensor (W, H) -- what `matrix`/`dist` describe
    offset: tuple[float, float]     # origin of the stored image inside the sensor frame
    image_size: tuple[int, int]     # (W, H) of the stored image
    moving: bool = False
    matrix: np.ndarray | None = None       # (3,3); None -> nominal pinhole (2D sessions)
    dist: np.ndarray | None = None         # (5,)
    rvec: np.ndarray | None = None         # (3,) Rodrigues, world -> cam
    tvec: np.ndarray | None = None         # (3,)

    @property
    def calibrated(self) -> bool:
        return self.matrix is not None and self.rvec is not None and self.tvec is not None

    def to_aniposelib(self):
        """An aniposelib Camera. For an uncalibrated 2D camera, a nominal pinhole.

        The nominal focal length is max(W, H) so normalised coords land in ~[-0.5, 0.5]; the
        value is irrelevant because the intrinsic embedding uses its missing-intrinsic token.
        `set_size` must be the real image so `p2d / size` is pixel-normalised.
        """
        from aniposelib.cameras import Camera as ACamera

        if self.calibrated:
            mat, dist, rvec, tvec = self.matrix, self.dist, self.rvec, self.tvec
        else:
            w, h = self.image_size
            f = float(max(w, h))
            mat = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
            dist = np.zeros(5)
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        cam = ACamera(matrix=mat, dist=dist if dist is not None else np.zeros(5),
                      rvec=rvec, tvec=tvec, name=self.name)
        cam.set_size(tuple(int(v) for v in self.size))
        return cam


def _ext_to_rt(ext) -> tuple[np.ndarray, np.ndarray]:
    """4x4 world->cam -> (rvec, tvec)."""
    import cv2
    ext = np.asarray(ext, dtype=np.float64).reshape(4, 4)
    rvec = cv2.Rodrigues(ext[:3, :3])[0].ravel()
    return rvec, ext[:3, 3].copy()


def _rt_to_ext(rvec, tvec) -> np.ndarray:
    import cv2
    ext = np.eye(4)
    ext[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    ext[:3, 3] = np.asarray(tvec, dtype=np.float64)
    return ext


def load_calibration(path: Path) -> list[Camera]:
    """Parse a calibration.toml. Camera order is the file's section order."""
    with open(path, 'rb') as f:
        doc = tomllib.load(f)
    cams = []
    for key, block in doc.items():
        if key == 'metadata' or not isinstance(block, dict):
            continue
        name = block.get('name', key)
        if 'size' not in block:
            raise FormatError(f'{path}: camera {name!r} has no size')
        arr = lambda k: np.asarray(block[k], dtype=np.float64) if k in block else None
        cams.append(Camera(
            name=name,
            type=block.get('type', 'pinhole'),
            size=tuple(int(v) for v in block['size']),
            offset=tuple(float(v) for v in block.get('offset', (0.0, 0.0))),
            image_size=tuple(int(v) for v in block.get('image_size', block['size'])),
            moving=bool(block.get('moving', False)),
            matrix=arr('matrix'),
            dist=None if arr('distortions') is None else arr('distortions').ravel(),
            rvec=None if arr('rotation') is None else arr('rotation').ravel(),
            tvec=None if arr('translation') is None else arr('translation').ravel(),
        ))
    if not cams:
        raise FormatError(f'{path}: no camera sections')
    return cams


def dump_calibration(path: Path, cams: list[Camera]) -> None:
    """Write calibration.toml in aniposelib's layout plus this format's four added keys."""
    import toml

    doc = {}
    for i, cam in enumerate(cams):
        block = {
            'name': cam.name,
            'type': cam.type,
            'size': [int(v) for v in cam.size],
            'offset': [float(v) for v in cam.offset],
            'image_size': [int(v) for v in cam.image_size],
            'moving': bool(cam.moving),
        }
        if cam.calibrated:
            block['matrix'] = np.asarray(cam.matrix, dtype=float).tolist()
            block['distortions'] = np.asarray(
                cam.dist if cam.dist is not None else np.zeros(5), dtype=float).ravel().tolist()
            block['rotation'] = np.asarray(cam.rvec, dtype=float).ravel().tolist()
            block['translation'] = np.asarray(cam.tvec, dtype=float).ravel().tolist()
        doc[f'cam_{i}'] = block
    doc['metadata'] = {}
    path.write_text(toml.dumps(doc))


# --------------------------------------------------------------------------------------------
# parquet helpers
# --------------------------------------------------------------------------------------------

def _codes(table: pa.Table, col: str) -> tuple[np.ndarray, list[str]]:
    """(int32 codes, distinct values) for a string column. Dictionary-encodes if it is not."""
    arr = table.column(col).combine_chunks()
    if not pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_encode()
    return arr.indices.to_numpy(zero_copy_only=False).astype(np.int32), arr.dictionary.to_pylist()


def _remap(codes: np.ndarray, values: list[str], vocab: dict[str, int],
           what: str, where: str) -> np.ndarray:
    """Translate dictionary codes into `vocab` indices. Unknown value is a FormatError."""
    lut = np.empty(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        if v not in vocab:
            raise FormatError(f'{where}: unknown {what} {v!r}')
        lut[i] = vocab[v]
    return lut[codes]


def _floats(table: pa.Table, col: str, n: int) -> np.ndarray:
    """A float64 column as numpy, nulls as NaN. Missing column -> all NaN."""
    if col not in table.column_names:
        return np.full(n, np.nan)
    return table.column(col).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)


def write_table(path: Path, rows: dict[str, np.ndarray], dict_cols: tuple[str, ...]) -> None:
    """Write a tidy-long table, dictionary-encoding the columns the spec says are dictionaries."""
    arrays, names = [], []
    for name, values in rows.items():
        arr = pa.array(values)
        if name in dict_cols:
            arr = arr.dictionary_encode()
        arrays.append(arr)
        names.append(name)
    pq.write_table(pa.table(arrays, names=names), path, compression='zstd')


DICT_COLS = ('group_id', 'animal_id', 'camera', 'bodypart', 'status')


def _inv(enum: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in enum.items()}


# --------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------

@dataclass
class Labels:
    """One group's labels as dense arrays. See docs/annotation_format.md §12."""
    animal_ids: list[str]
    points3d: np.ndarray | None      # (S,T,K,3) float32, NaN where not visible
    vis3d: np.ndarray | None         # (S,T,K) int8
    points2d: np.ndarray | None      # (S,T,K,C,2) float32
    vis2d: np.ndarray | None         # (S,T,K,C) int8
    boxes: np.ndarray | None         # (S,T,C,4) float32
    instance: np.ndarray | None      # (S,T,C) int8
    ext: np.ndarray | None = None    # (C,T,4,4) float64, only when a camera is moving

    @property
    def n_animals(self) -> int:
        return len(self.animal_ids)


def _scatter(table, gid, T, kpt_vocab, cam_vocab, animals, where, per_camera, n_extra):
    """Common scatter: pick this group's rows and index them into dense arrays.

    Returns (rows_index, a, f, k, c, status) with `a` already remapped to 0..S-1 and `c` None
    when the table has no camera column.
    """
    gcodes, gvals = _codes(table, 'group_id')
    try:
        want = gvals.index(gid)
    except ValueError:
        return None
    sel = np.flatnonzero(gcodes == want)
    if sel.size == 0:
        return None

    frame = table.column('frame').combine_chunks().to_numpy(zero_copy_only=False)[sel].astype(np.int64)
    if frame.min() < 0 or frame.max() >= T:
        raise FormatError(f'{where}: group {gid!r} has a frame outside [0, {T})')

    acodes, avals = _codes(table, 'animal_id')
    a = _remap(acodes, avals, animals, 'animal_id', where)[sel]

    k = None
    if 'bodypart' in table.column_names:
        kcodes, kvals = _codes(table, 'bodypart')
        k = _remap(kcodes, kvals, kpt_vocab, 'bodypart', where)[sel]

    c = None
    if per_camera:
        ccodes, cvals = _codes(table, 'camera')
        c = _remap(ccodes, cvals, cam_vocab, 'camera', where)[sel]

    scodes, svals = _codes(table, 'status')
    enum = INST_STATUS if n_extra == 4 else KPT_STATUS
    status = _remap(scodes, svals, enum, 'status', where)[sel]

    return sel, a, frame, k, c, status


def _animal_vocab(tables: list[pa.Table | None], gid: str) -> list[str]:
    """Distinct animal_ids appearing in this group, sorted. Defines S and the row order."""
    found: set[str] = set()
    for table in tables:
        if table is None:
            continue
        gcodes, gvals = _codes(table, 'group_id')
        if gid not in gvals:
            continue
        sel = gcodes == gvals.index(gid)
        acodes, avals = _codes(table, 'animal_id')
        found.update(np.asarray(avals, dtype=object)[np.unique(acodes[sel])].tolist())
    return sorted(found)


# --------------------------------------------------------------------------------------------
# groups and sessions
# --------------------------------------------------------------------------------------------

@dataclass
class Group:
    group_id: str
    n_frames: int
    fps: float = float('nan')
    source_video: str = ''
    source_frame_start: int = 0
    source_frame_step: int = 1
    notes: str = ''
    session: 'Session' = field(default=None, repr=False, compare=False)

    @property
    def dir(self) -> Path:
        return self.session.path / 'groups' / self.group_id

    def pixels(self, cam: str) -> tuple[str, Path]:
        """('frames', dir) or ('video', file) for one camera. Raises if neither or both exist."""
        d = self.dir / cam
        vids = [self.dir / (cam + e) for e in VIDEO_EXTS]
        vids = [v for v in vids if v.exists()]
        if d.is_dir() and vids:
            raise FormatError(f'{self.dir}: camera {cam!r} has both a frame dir and a video')
        if d.is_dir():
            return 'frames', d
        if vids:
            return 'video', vids[0]
        raise FormatError(f'{self.dir}: camera {cam!r} has neither a frame dir nor a video')

    def frame_paths(self, cam: str) -> list[Path]:
        kind, p = self.pixels(cam)
        if kind != 'frames':
            raise FormatError(f'{p}: is a video, not a frame directory')
        files = sorted(f for f in p.iterdir() if f.suffix in IMAGE_EXTS)
        return files

    def labels(self) -> Labels:
        return self.session.labels(self.group_id)


@dataclass
class Session:
    path: Path
    mode: str                       # '2d' | '3d'
    units: str
    names: list[str]                # keypoint names, THE authority for the keypoint axis
    cameras: list[Camera]
    groups: dict[str, Group]
    skeleton: list[list[str]] = field(default_factory=list)
    flip_pairs: list[list[str]] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    assoc_res_max_px: float = 30.0

    @property
    def session_id(self) -> str:
        return self.path.name

    @property
    def split(self) -> str:
        return self.path.parent.name

    @property
    def cam_names(self) -> list[str]:
        return [c.name for c in self.cameras]

    @property
    def n_keypoints(self) -> int:
        return len(self.names)

    @cached_property
    def _kpt_vocab(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    @cached_property
    def _cam_vocab(self) -> dict[str, int]:
        return {c.name: i for i, c in enumerate(self.cameras)}

    def _table(self, stem: str) -> pa.Table | None:
        p = self.path / f'{stem}.pq'
        return pq.read_table(p) if p.exists() else None

    @cached_property
    def _tables(self) -> dict[str, pa.Table | None]:
        return {s: self._table(s) for s in ('keypoints', 'points3d', 'instances', 'extrinsics')}

    @classmethod
    def load(cls, path: Path) -> 'Session':
        path = Path(path)
        cfg_path = path / 'session.toml'
        if not cfg_path.exists():
            raise FormatError(f'{path}: no session.toml')
        with open(cfg_path, 'rb') as f:
            cfg = tomllib.load(f)
        for key in ('mode', 'units', 'names'):
            if key not in cfg:
                raise FormatError(f'{cfg_path}: missing {key!r}')
        if cfg['mode'] not in ('2d', '3d'):
            raise FormatError(f'{cfg_path}: mode must be "2d" or "3d", got {cfg["mode"]!r}')
        if not cfg['names']:
            raise FormatError(f'{cfg_path}: names is empty')

        cams = load_calibration(path / 'calibration.toml')

        gt = pq.read_table(path / 'groups.pq').to_pydict()
        groups: dict[str, Group] = {}
        for i, gid in enumerate(gt['group_id']):
            groups[gid] = Group(
                group_id=gid,
                n_frames=int(gt['n_frames'][i]),
                fps=float(gt.get('fps', [float('nan')] * len(gt['group_id']))[i] or float('nan')),
                source_video=gt.get('source_video', [''] * len(gt['group_id']))[i] or '',
                source_frame_start=int(gt.get('source_frame_start',
                                              [0] * len(gt['group_id']))[i] or 0),
                source_frame_step=int(gt.get('source_frame_step',
                                             [1] * len(gt['group_id']))[i] or 1),
                notes=gt.get('notes', [''] * len(gt['group_id']))[i] or '',
            )

        sess = cls(
            path=path,
            mode=cfg['mode'],
            units=cfg['units'],
            names=list(cfg['names']),
            cameras=cams,
            groups=groups,
            skeleton=[list(p) for p in cfg.get('skeleton', [])],
            flip_pairs=[list(p) for p in cfg.get('flip_pairs', [])],
            provenance=dict(cfg.get('provenance', {})),
            assoc_res_max_px=float(cfg.get('assoc_res_max_px', 30.0)),
        )
        for g in groups.values():
            g.session = sess
        return sess

    def labels(self, gid: str) -> Labels:
        """Scatter one group's rows into dense arrays. See docs/annotation_format.md §12."""
        group = self.groups[gid]
        T, K, C = group.n_frames, len(self.names), len(self.cameras)
        t = self._tables
        animals = _animal_vocab([t['keypoints'], t['points3d'], t['instances']], gid)
        avocab = {a: i for i, a in enumerate(animals)}
        S = len(animals)
        where = f'{self.path}:{gid}'

        points3d = vis3d = points2d = vis2d = boxes = instance = None

        if t['points3d'] is not None:
            r = _scatter(t['points3d'], gid, T, self._kpt_vocab, self._cam_vocab, avocab,
                         f'{where}/points3d', per_camera=False, n_extra=3)
            points3d = np.full((S, T, K, 3), np.nan, np.float32)
            vis3d = np.full((S, T, K), UNLABELED, np.int8)
            if r is not None:
                sel, a, f, k, _, status = r
                vis3d[a, f, k] = status
                m = status == VISIBLE
                n = len(t['points3d'])
                xyz = np.stack([_floats(t['points3d'], c, n)[sel] for c in 'xyz'], -1)
                points3d[a[m], f[m], k[m]] = xyz[m]

        if t['keypoints'] is not None:
            r = _scatter(t['keypoints'], gid, T, self._kpt_vocab, self._cam_vocab, avocab,
                         f'{where}/keypoints', per_camera=True, n_extra=2)
            points2d = np.full((S, T, K, C, 2), np.nan, np.float32)
            vis2d = np.full((S, T, K, C), UNLABELED, np.int8)
            if r is not None:
                sel, a, f, k, c, status = r
                vis2d[a, f, k, c] = status
                m = status == VISIBLE
                n = len(t['keypoints'])
                xy = np.stack([_floats(t['keypoints'], q, n)[sel] for q in 'xy'], -1)
                points2d[a[m], f[m], k[m], c[m]] = xy[m]

        if t['instances'] is not None:
            r = _scatter(t['instances'], gid, T, self._kpt_vocab, self._cam_vocab, avocab,
                         f'{where}/instances', per_camera=True, n_extra=4)
            boxes = np.full((S, T, C, 4), np.nan, np.float32)
            instance = np.full((S, T, C), INST_NONE, np.int8)
            if r is not None:
                sel, a, f, _, c, status = r
                instance[a, f, c] = status
                n = len(t['instances'])
                box = np.stack([_floats(t['instances'], q, n)[sel]
                                for q in ('x0', 'y0', 'x1', 'y1')], -1)
                boxes[a, f, c] = box

        ext = None
        if t['extrinsics'] is not None and any(c.moving for c in self.cameras):
            tab = t['extrinsics']
            gcodes, gvals = _codes(tab, 'group_id')
            if gid in gvals:
                sel = np.flatnonzero(gcodes == gvals.index(gid))
                ccodes, cvals = _codes(tab, 'camera')
                c = _remap(ccodes, cvals, self._cam_vocab, 'camera', where)[sel]
                f = tab.column('frame').combine_chunks().to_numpy(
                    zero_copy_only=False)[sel].astype(np.int64)
                raw = np.stack(tab.column('ext').combine_chunks().to_numpy(
                    zero_copy_only=False)[sel]).astype(np.float64)
                ext = np.tile(np.eye(4), (C, T, 1, 1))
                ext[c, f] = raw.reshape(-1, 4, 4)
                for i, cam in enumerate(self.cameras):
                    if not cam.moving and cam.calibrated:
                        ext[i] = _rt_to_ext(cam.rvec, cam.tvec)

        return Labels(animal_ids=animals, points3d=points3d, vis3d=vis3d, points2d=points2d,
                      vis2d=vis2d, boxes=boxes, instance=instance, ext=ext)


# --------------------------------------------------------------------------------------------
# writing -- the exact inverse of Session.labels, so a round-trip test is meaningful
# --------------------------------------------------------------------------------------------

def write_session(path: Path, *, mode: str, units: str, names: list[str], cameras: list[Camera],
                  groups: dict[str, Group], labels: dict[str, Labels],
                  skeleton=(), flip_pairs=(), provenance=None,
                  assoc_res_max_px: float | None = None) -> None:
    """Write session.toml, calibration.toml and the label tables. Pixels are the caller's job.

    `labels` maps group_id -> dense arrays in exactly the layout `Session.labels` returns; a row
    is emitted only where the status array says a determination was made, which is what makes
    sparse hand annotation and dense tracking the same code path.
    """
    import toml

    path.mkdir(parents=True, exist_ok=True)
    cfg = {'mode': mode, 'units': units, 'names': list(names)}
    if skeleton:
        cfg['skeleton'] = [list(p) for p in skeleton]
    if flip_pairs:
        cfg['flip_pairs'] = [list(p) for p in flip_pairs]
    if assoc_res_max_px is not None:
        cfg['assoc_res_max_px'] = float(assoc_res_max_px)
    cfg['provenance'] = dict(provenance or {})
    (path / 'session.toml').write_text(toml.dumps(cfg))
    dump_calibration(path / 'calibration.toml', cameras)

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

    cam_names = [c.name for c in cameras]
    kpt = _inv(KPT_STATUS)
    inst = _inv(INST_STATUS)
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

    def push(name, **cols):
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
            for ci, cam in enumerate(cameras):
                if not cam.moving:
                    continue
                T = lab.ext.shape[1]
                push('extrinsics', group_id=[gid] * T, frame=np.arange(T, dtype=np.int32),
                     camera=[cam.name] * T, ext=[e.ravel().tolist() for e in lab.ext[ci]])

    for stem, cols in tables.items():
        if not cols['group_id']:
            continue
        write_table(path / f'{stem}.pq',
                    {k: np.asarray(v, dtype=object if k in DICT_COLS else None)
                     if k != 'ext' else v for k, v in cols.items()},
                    dict_cols=DICT_COLS)


def empty_labels(n_animals: int, T: int, K: int, C: int, *, mode3d: bool,
                 animal_ids: list[str] | None = None) -> Labels:
    """All-unlabeled dense arrays, ready to fill and hand to `write_session`."""
    S = n_animals
    return Labels(
        animal_ids=animal_ids or [f'a{i:02d}' for i in range(S)],
        points3d=np.full((S, T, K, 3), np.nan, np.float32) if mode3d else None,
        vis3d=np.full((S, T, K), UNLABELED, np.int8) if mode3d else None,
        points2d=np.full((S, T, K, C, 2), np.nan, np.float32),
        vis2d=np.full((S, T, K, C), UNLABELED, np.int8),
        boxes=None, instance=None,
    )


# --------------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------------

@dataclass
class Dataset:
    name: str
    root: Path
    sessions: dict[str, list[Session]]      # split -> sessions

    @property
    def names(self) -> list[str]:
        for group in self.sessions.values():
            if group:
                return group[0].names
        raise FormatError(f'{self.root}: no sessions')

    def all_sessions(self) -> list[Session]:
        return [s for group in self.sessions.values() for s in group]


def _is_dataset_root(path: Path) -> bool:
    return (path / 'train').is_dir() or any((path / s).is_dir() for s in SPLITS)


def load_dataset(root: Path) -> Dataset:
    root = Path(root)
    sessions: dict[str, list[Session]] = {}
    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            continue
        sessions[split] = [Session.load(p) for p in sorted(d.iterdir())
                           if (p / 'session.toml').exists()]
    if not sessions:
        raise FormatError(f'{root}: no train/val/test directory')
    return Dataset(name=root.name, root=root, sessions=sessions)


def load_datasets(path: Path) -> list[Dataset]:
    """One dataset root, or a folder whose children are dataset roots.

    This is the whole of the "point at a dataset OR at a folder of datasets" rule: the presence
    of a `train/` directory is what distinguishes them.
    """
    path = Path(path)
    if _is_dataset_root(path):
        return [load_dataset(path)]
    children = sorted(p for p in path.iterdir() if p.is_dir() and _is_dataset_root(p))
    if not children:
        raise FormatError(f'{path}: neither a dataset root nor a folder of dataset roots')
    return [load_dataset(p) for p in children]


# --------------------------------------------------------------------------------------------
# keypoint registry
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Registry:
    """Global keypoint identity across one or more datasets.

    A single dataset keeps its names as-is. A collection prefixes them with the dataset folder
    name (`rat-city-nose`), which is what lets one embedding table serve every dataset at once.

    Ids are append-only against a `base`: a later run that is handed an existing registry keeps
    every old id, so the learned embedding rows behind them survive warm start. That is the whole
    reason this is a file in the run folder and not something recomputed from a directory
    listing.
    """
    names: tuple[str, ...]
    datasets: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def n_keypoints(self) -> int:
        return len(self.names)

    def ids_for(self, dataset: str) -> np.ndarray:
        for name, ids in self.datasets:
            if name == dataset:
                return np.asarray(ids, dtype=np.int64)
        raise KeyError(f'{dataset!r} is not in this registry: {[d for d, _ in self.datasets]}')

    @classmethod
    def build(cls, datasets: list[Dataset], base: 'Registry | None' = None) -> 'Registry':
        prefix = len(datasets) > 1 or (base is not None and len(base.datasets) > 1)
        names = list(base.names) if base else []
        index = {n: i for i, n in enumerate(names)}
        out = dict(base.datasets) if base else {}
        for ds in datasets:
            ids = []
            for local in ds.names:
                full = f'{ds.name}-{local}' if prefix else local
                if full not in index:
                    index[full] = len(names)
                    names.append(full)
                ids.append(index[full])
            if ds.name in out and tuple(ids) != tuple(out[ds.name]):
                raise FormatError(
                    f'{ds.name}: keypoint ids changed against the base registry '
                    f'({out[ds.name]} -> {tuple(ids)}); names must be append-only')
            out[ds.name] = tuple(ids)
        return cls(names=tuple(names), datasets=tuple(sorted(out.items())))

    def save(self, path: Path) -> None:
        import toml
        path.write_text(toml.dumps({
            'names': list(self.names),
            'datasets': {name: list(ids) for name, ids in self.datasets},
        }))

    @classmethod
    def load(cls, path: Path) -> 'Registry':
        with open(path, 'rb') as f:
            doc = tomllib.load(f)
        return cls(names=tuple(doc['names']),
                   datasets=tuple(sorted((k, tuple(v)) for k, v in doc['datasets'].items())))


# --------------------------------------------------------------------------------------------
# validation -- docs/annotation_format.md §11
# --------------------------------------------------------------------------------------------

def validate_session(sess: Session, check_images: bool = True) -> list[str]:
    """Return a list of rule violations. Empty means the session is valid."""
    errs: list[str] = []
    here = str(sess.path)

    def bad(rule: int, msg: str) -> None:
        errs.append(f'{here}: [rule {rule}] {msg}')

    # 2. names unique; skeleton/flip_pairs reference known names; flip is an involution
    if len(set(sess.names)) != len(sess.names):
        bad(2, 'names has duplicates')
    known = set(sess.names)
    for pair in sess.skeleton + sess.flip_pairs:
        for n in pair:
            if n not in known:
                bad(2, f'skeleton/flip_pairs names unknown keypoint {n!r}')
    # each pair is listed ONCE; the involution is the symmetric closure, so the only way to
    # break it is to name the same keypoint in two pairs with different partners.
    flip: dict[str, str] = {}
    for a, b in sess.flip_pairs:
        for x, y in ((a, b), (b, a)):
            if flip.setdefault(x, y) != y:
                bad(2, f'flip_pairs is not an involution: {x!r} maps to both '
                       f'{flip[x]!r} and {y!r}')
        if a == b:
            bad(2, f'flip_pairs has a self-pair at {a!r}')

    # 4/5. cameras and calibration
    for cam in sess.cameras:
        if cam.type not in ('pinhole', 'fisheye'):
            bad(4, f'camera {cam.name!r} has type {cam.type!r}')
    if sess.mode == '3d':
        if len(sess.cameras) < 2:
            bad(5, f'mode=3d with {len(sess.cameras)} camera(s)')
        for cam in sess.cameras:
            if not cam.calibrated:
                bad(5, f'mode=3d but camera {cam.name!r} has no matrix/rotation/translation')
    elif len(sess.cameras) != 1:
        bad(5, f'mode=2d with {len(sess.cameras)} cameras')

    if not (sess.path / 'keypoints.pq').exists() and not (sess.path / 'points3d.pq').exists():
        bad(6, 'neither keypoints.pq nor points3d.pq exists')

    # 9. no duplicate keys
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

    # 10. a visible row carries its coordinates
    t3 = sess._tables['points3d']
    if t3 is not None and len(t3):
        st, vals = _codes(t3, 'status')
        vis = np.isin(st, [i for i, v in enumerate(vals) if v == 'visible'])
        xyz = np.stack([_floats(t3, c, len(t3)) for c in 'xyz'], -1)
        if vis.any() and not np.isfinite(xyz[vis]).all():
            bad(10, 'points3d.pq has a visible row without x,y,z')
        if (~vis).any() and np.isfinite(xyz[~vis]).any():
            bad(10, 'points3d.pq has a non-visible row carrying coordinates')

    tk = sess._tables['keypoints']
    if tk is not None and len(tk):
        st, vals = _codes(tk, 'status')
        vis = np.isin(st, [i for i, v in enumerate(vals) if v == 'visible'])
        xy = np.stack([_floats(tk, c, len(tk)) for c in 'xy'], -1)
        if vis.any() and not np.isfinite(xy[vis]).all() and t3 is None:
            bad(10, 'keypoints.pq has a visible row without x,y and there is no 3D layer')
        if (~vis).any() and np.isfinite(xy[~vis]).any():
            bad(10, 'keypoints.pq has a non-visible row carrying coordinates')

    # 13. extrinsics only for cameras declared moving
    te = sess._tables['extrinsics']
    moving = {c.name for c in sess.cameras if c.moving}
    if te is not None and len(te):
        _, cams = _codes(te, 'camera')
        for name in cams:
            if name not in moving:
                bad(13, f'extrinsics.pq names camera {name!r}, which is not moving=true')
    elif moving:
        bad(13, f'cameras {sorted(moving)} are moving=true but extrinsics.pq is absent/empty')

    # 6/7/8. per group: frames in range, pixels present and the right shape
    for gid, g in sess.groups.items():
        for cam in sess.cameras:
            try:
                kind, p = g.pixels(cam.name)
            except FormatError as e:
                bad(7, str(e))
                continue
            if kind != 'frames':
                continue
            files = g.frame_paths(cam.name)
            if len(files) != g.n_frames:
                bad(7, f'group {gid!r} camera {cam.name!r}: {len(files)} files, '
                       f'n_frames={g.n_frames}')
            exts = {f.suffix for f in files}
            if len(exts) > 1:
                bad(7, f'group {gid!r} camera {cam.name!r}: mixed extensions {sorted(exts)}')
            if files and [f.stem for f in files[:len(files)]] != [
                    f'{i:06d}' for i in range(len(files))]:
                bad(7, f'group {gid!r} camera {cam.name!r}: names are not contiguous %06d')
            if check_images and files:
                from PIL import Image
                with Image.open(files[0]) as im:
                    if tuple(im.size) != tuple(cam.image_size):
                        bad(8, f'group {gid!r} camera {cam.name!r}: image is {im.size}, '
                               f'image_size is {tuple(cam.image_size)}')
        try:
            sess.labels(gid)          # raises on unknown bodypart/camera/animal or bad frame
        except FormatError as e:
            errs.append(str(e))
    return errs


def validate_dataset(ds: Dataset, check_images: bool = True) -> list[str]:
    errs: list[str] = []
    sessions = ds.all_sessions()
    if not sessions:
        return [f'{ds.root}: no sessions']
    # 3. cross-session agreement on names
    ref = sessions[0]
    for s in sessions[1:]:
        if s.names != ref.names:
            errs.append(f'{s.path}: [rule 3] names differ from {ref.path}')
    # 14. leak: a session folder name used in more than one split
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
