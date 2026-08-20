"""Read, write and validate the `tailcycle-dataset` format.

`docs/annotation_format.md` is the spec; this module implements it. When the two disagree, the
document wins.

The read path turns tidy-long parquet into dense arrays ONCE per group, in whatever process calls
it, so forked dataloader workers share the pages copy-on-write. `status` is the visibility
channel and is dictionary-encoded in parquet, which means "is this point visible" is an int8
compare on the dictionary codes rather than a string comparison -- the reason the format uses
parquet at all.

Camera geometry is aniposelib's job -- `Rig` wraps a `CameraGroup` and adds only the two facts
aniposelib has no field for (`offset`, `moving`). calibration.toml is aniposelib's own layout, so
anipose can read it directly.
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
# PROJECTED is a position with NO visibility claim: the annotator placed the point in this view
# but never judged whether it was actually seen there. Consumers must not train visibility on it.
UNLABELED, MISSING, VISIBLE, PROJECTED = -1, 0, 1, 2
# The statuses that carry coordinates.
POSITIONED = (VISIBLE, PROJECTED)
# Instance codes. -1 doubles as "no row" == no determination.
INST_NONE, INST_ABSENT, INST_PRESENT, INST_LABELED = -1, 0, 1, 2

# Region codes, for the `regions.pq` table. A region is a statement about PIXELS, not about an
# animal, which is why it cannot ride in `instances.pq`: that table is keyed by `animal_id`, and
# its `present` status is an IGNORE region -- the opposite polarity to "this area is fully
# labelled, so absence of a label here IS evidence of absence".
REGION_COMPLETE = 0

KPT_STATUS = {'unlabeled': UNLABELED, 'missing': MISSING, 'visible': VISIBLE,
              'projected': PROJECTED}
INST_STATUS = {'absent': INST_ABSENT, 'present': INST_PRESENT, 'labeled': INST_LABELED}
REGION_STATUS = {'labelled_complete': REGION_COMPLETE}
IMAGE_EXTS = ('.png', '.jpg')
VIDEO_EXTS = ('.mp4', '.avi')
SPLITS = ('train', 'val', 'test')
# session.toml `labels` (§4, decision 6): who produced the labels. Closed vocabulary, so a typo is
# a FormatError rather than a silent third category. Exposed on Session as `label_source`, NOT as
# `labels` -- `Session.labels(gid)` is the method that returns a group's label arrays, and a
# dataclass field of that name would shadow it on every instance.
LABEL_SOURCES = ('annotated', 'tracked')


class FormatError(Exception):
    """A dataset on disk does not satisfy docs/annotation_format.md."""


# cameras

def nominal_camera(name: str, size, dist=None):
    """An aniposelib Camera for an uncalibrated single view (a 2D session).

    Focal length max(W, H) so normalised coords land in ~[-0.5, 0.5]; the value is irrelevant
    because posetail's intrinsic embedding uses its missing-intrinsic token. `set_size` must be
    the real image so `p2d / size` is pixel-normalised.
    """
    from aniposelib.cameras import Camera

    w, h = int(size[0]), int(size[1])
    f = float(max(w, h))
    cam = Camera(matrix=np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]]),
                 dist=np.zeros(5) if dist is None else np.asarray(dist, float).ravel(),
                 rvec=np.zeros(3), tvec=np.zeros(3), name=name)
    cam.set_size((w, h))
    return cam


@dataclass
class Rig:
    """An aniposelib `CameraGroup` plus the three per-camera facts it does not model.

    aniposelib owns the geometry -- intrinsics, distortion, extrinsics, projection,
    triangulation. This adds only what the format needs on top and aniposelib has no field for:

    - `offset`: origin of the stored image inside the sensor frame. `matrix` and `distortions`
      are in SENSOR coordinates and `size` is the image ON DISK, so projecting is "apply matrix,
      then subtract offset" -- posetail's convention (`cube.project_cam` subtracts
      `cam['offset']` after the intrinsic matmul; `undistort_points` adds it back).
    - `moving`: whether per-frame extrinsics live in `extrinsics.pq`.
    - `calibrated`: whether the file actually carried a matrix, or `nominal_camera` invented one.

    `fisheye` is aniposelib's own per-camera flag and is what `CameraGroup.from_dicts` keys the
    `FisheyeCamera` subclass off, so it is not duplicated here -- `cam_type` reads it back.
    """
    cgroup: object                              # aniposelib.cameras.CameraGroup
    offset: dict[str, tuple[float, float]]
    moving: dict[str, bool]
    calibrated: dict[str, bool]

    @property
    def cameras(self):
        return self.cgroup.cameras

    @property
    def names(self) -> list[str]:
        return [c.get_name() for c in self.cgroup.cameras]

    def __len__(self) -> int:
        return len(self.cgroup.cameras)

    def size(self, name: str) -> tuple[int, int]:
        return tuple(int(v) for v in self.by_name(name).get_size())

    def by_name(self, name: str):
        for c in self.cgroup.cameras:
            if c.get_name() == name:
                return c
        raise KeyError(f'no camera named {name!r}')

    def cam_type(self, name: str) -> str:
        from aniposelib.cameras import FisheyeCamera
        return 'fisheye' if isinstance(self.by_name(name), FisheyeCamera) else 'pinhole'

    def posetail(self, device='cpu', moving_ext: dict | None = None) -> list[dict]:
        """posetail's camera dicts, DETACHED.

        On the pytorch branch aniposelib's intrinsics and extrinsics are `nn.Parameter`s, so
        `format_camera` hands back tensors with `requires_grad=True`. Left attached, every
        projection in the loss would build an autograd graph through the calibration and
        gradients would flow into camera intrinsics we are not training.
        """
        import torch
        from posetail.posetail.train_utils import format_camera

        out = []
        with torch.no_grad():
            for cam in self.cgroup.cameras:
                n = cam.get_name()
                d = format_camera(cam, {n: self.offset[n]}, self.cam_type(n), device,
                                  ext_override=None if moving_ext is None else moving_ext.get(n))
                out.append({k: (v.detach() if torch.is_tensor(v) else v) for k, v in d.items()})
        return out


def load_calibration(path: Path) -> Rig:
    """Parse a calibration.toml into a Rig. Camera order is the file's section order."""
    from aniposelib.cameras import CameraGroup

    with open(path, 'rb') as f:
        doc = tomllib.load(f)

    blocks = [b for k, b in doc.items() if k != 'metadata' and isinstance(b, dict)]
    if not blocks:
        raise FormatError(f'{path}: no camera sections')

    cams, offset, moving, calibrated = [], {}, {}, {}
    for block in blocks:
        name = str(block.get('name', ''))
        if 'size' not in block:
            raise FormatError(f'{path}: camera {name!r} has no size')
        if 'matrix' in block:
            cams.append(CameraGroup.from_dicts([block]).cameras[0])
        else:
            cams.append(nominal_camera(name, block['size'], block.get('distortions')))
        offset[name] = tuple(float(v) for v in block.get('offset', (0.0, 0.0)))
        moving[name] = bool(block.get('moving', False))
        calibrated[name] = 'matrix' in block
    return Rig(cgroup=CameraGroup(cams), offset=offset, moving=moving, calibrated=calibrated)


def dump_calibration(path: Path, rig: Rig) -> None:
    """Write calibration.toml: aniposelib's own dicts plus `offset` / `moving`.

    `fisheye` comes from aniposelib's `get_dict`; an uncalibrated camera writes only what it
    really knows (name, size, offset) so that reading it back rebuilds the nominal camera rather
    than resurrecting an invented matrix as if it were a calibration.
    """
    import toml

    doc = {}
    for i, cam in enumerate(rig.cgroup.cameras):
        name = cam.get_name()
        if rig.calibrated[name]:
            block = cam.get_dict()
        else:
            block = {'name': name, 'size': [int(v) for v in cam.get_size()]}
        block['offset'] = [float(v) for v in rig.offset[name]]
        block['moving'] = bool(rig.moving[name])
        doc[f'cam_{i}'] = block
    doc['metadata'] = getattr(rig.cgroup, 'metadata', {}) or {}
    path.write_text(toml.dumps(doc))


# parquet helpers

def _codes(table: pa.Table, col: str) -> tuple[np.ndarray, list[str]]:
    """(int32 codes, distinct values) for a string column. Dictionary-encodes if it is not."""
    arr = table.column(col).combine_chunks()
    if not pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_encode()
    return arr.indices.to_numpy(zero_copy_only=False).astype(np.int32), arr.dictionary.to_pylist()


def _remap(codes: np.ndarray, values: list[str], vocab: dict[str, int],
           what: str, where: str, sel: np.ndarray | None = None) -> np.ndarray:
    """Translate dictionary codes into `vocab` indices, restricted to `sel` rows.

    The restriction matters: a parquet dictionary is per FILE, not per group, so a session whose
    groups hold different numbers of animals has `animal_id` values in the dictionary that the
    group being read has never heard of. Validating the whole dictionary rejected every
    branson-fly session, where the fly count varies 5..10 between trials.
    """
    lut = np.array([vocab.get(v, -1) for v in values], dtype=np.int32)
    codes = codes if sel is None else codes[sel]
    out = lut[codes]
    if (out < 0).any():
        unknown = sorted({values[c] for c in np.unique(codes[out < 0])})
        raise FormatError(f'{where}: unknown {what} {unknown[0]!r}')
    return out


def _floats(table: pa.Table, col: str, n: int) -> np.ndarray:
    """A float64 column as numpy, nulls as NaN. Missing column -> all NaN."""
    if col not in table.column_names:
        return np.full(n, np.nan)
    return table.column(col).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)


def _arrow(rows: dict[str, np.ndarray], dict_cols: tuple[str, ...]) -> pa.Table:
    """One chunk of tidy-long rows as an arrow table. THE definition of "how a row becomes parquet".

    Non-finite floats become NULL, not NaN. The spec says a `missing` row's `x,y` are empty, and a
    null is what "empty" means in parquet -- it costs a validity bit instead of four bytes, and any
    other reader sees the absence rather than a sentinel it has to know about.
    """
    arrays, names = [], []
    for name, values in rows.items():
        values = np.asarray(values) if not isinstance(values, list) else values
        if isinstance(values, np.ndarray) and values.dtype == object and values.size == 0:
            # An empty object array infers pyarrow's `null` type, and a dictionary of nulls is not
            # the schema any reader expects. Only a zero-row table gets here -- an empty
            # regions.pq, which is a meaningful thing to write (§9b).
            arr = pa.array([], pa.string())
        elif isinstance(values, np.ndarray) and values.dtype.kind == 'f':
            arr = pa.array(values, mask=~np.isfinite(values))
        else:
            arr = pa.array(values)
        if name in dict_cols:
            arr = arr.dictionary_encode()
        arrays.append(arr)
        names.append(name)
    return pa.table(arrays, names=names)


class TableWriter:
    """A tidy-long table written in CHUNKS, so a producer need never hold the whole thing.

    `write_table` below is this with one chunk, and every converter still calls that -- a converter
    has its rows in memory anyway. What needs chunks is INFERENCE: a prediction over a 720,000-frame
    clip is tens of millions of rows and the whole point of the streaming loop is that no array is
    proportional to the clip.

    **THE DICTIONARY COLUMNS FIX THEIR SCHEMA ON THE FIRST CHUNK, and that is not a detail.**
    `format._codes` reads `DICT_COLS` back as a `DictionaryArray`, so they must be written as
    `dictionary<int32, string>` -- a plain string column with parquet's `use_dictionary=True` has
    the same bytes on disk but round-trips as plain strings, and every reader in this file would
    break. Later chunks are CAST to the first chunk's schema; pyarrow unifies the per-chunk
    dictionaries itself.

    ROWS ARE BUFFERED to `chunk_rows` before a row group is emitted. Writing one row group per
    call would give a long clip tens of thousands of tiny row groups -- metadata bloat, and a file
    no reader can scan efficiently -- because a block at a tight budget can be a few dozen rows.
    """

    def __init__(self, path: Path, dict_cols: tuple[str, ...] = (), chunk_rows: int = 250_000):
        self.path, self.dict_cols, self.chunk_rows = Path(path), dict_cols, int(chunk_rows)
        self._w = None
        self._buf: list[pa.Table] = []
        self._n = 0

    def write(self, rows: dict[str, np.ndarray]) -> None:
        t = _arrow(rows, self.dict_cols)
        if not len(t):
            return
        self._buf.append(t)
        self._n += len(t)
        if self._n >= self.chunk_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        t = pa.concat_tables(self._buf)
        if self._w is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._w = pq.ParquetWriter(self.path, t.schema, compression='zstd')
        else:
            t = t.cast(self._w.schema)
        self._w.write_table(t)
        self._buf, self._n = [], 0

    def close(self) -> None:
        self._flush()
        if self._w is not None:
            self._w.close()
            self._w = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_table(path: Path, rows: dict[str, np.ndarray], dict_cols: tuple[str, ...]) -> None:
    """Write a tidy-long table in one shot. See `TableWriter` for the chunked form."""
    pq.write_table(_arrow(rows, dict_cols), path, compression='zstd')


DICT_COLS = ('group_id', 'animal_id', 'camera', 'bodypart', 'status')


def _inv(enum: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in enum.items()}


# labels

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
    # (M,6) float64 [frame, camera, x0, y0, x1, y1]. None IFF the session has no regions.pq, which
    # claims exhaustive labelling; an empty (0,6) says the file exists and certifies nothing in
    # this group. Collapsing the two reads an uncertified group as fully labelled -- the exact
    # inversion the table exists to prevent (§9b).
    regions: np.ndarray | None = None

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
    a = _remap(acodes, avals, animals, 'animal_id', where, sel)

    k = None
    if 'bodypart' in table.column_names:
        kcodes, kvals = _codes(table, 'bodypart')
        k = _remap(kcodes, kvals, kpt_vocab, 'bodypart', where, sel)

    c = None
    if per_camera:
        ccodes, cvals = _codes(table, 'camera')
        c = _remap(ccodes, cvals, cam_vocab, 'camera', where, sel)

    scodes, svals = _codes(table, 'status')
    enum = INST_STATUS if n_extra == 4 else KPT_STATUS
    status = _remap(scodes, svals, enum, 'status', where, sel)

    return sel, a, frame, k, c, status


def _regions(table: pa.Table, gid: str, T: int, cam_vocab: dict[str, int],
             where: str) -> np.ndarray:
    """One group's `regions.pq` rows as (M,6) [frame, camera, x0, y0, x1, y1].

    Not `_scatter`: a region has no `animal_id` and no dense shape to scatter into -- it is a
    list of rectangles, and the number of them per frame is unbounded.
    """
    empty = np.zeros((0, 6), np.float64)
    gcodes, gvals = _codes(table, 'group_id')
    if gid not in gvals:
        return empty
    sel = np.flatnonzero(gcodes == gvals.index(gid))
    if sel.size == 0:
        return empty

    frame = table.column('frame').combine_chunks().to_numpy(
        zero_copy_only=False)[sel].astype(np.int64)
    if frame.min() < 0 or frame.max() >= T:
        raise FormatError(f'{where}: group {gid!r} has a frame outside [0, {T})')
    c = _remap(*_codes(table, 'camera'), cam_vocab, 'camera', where, sel)
    _remap(*_codes(table, 'status'), REGION_STATUS, 'status', where, sel)

    n = len(table)
    box = np.stack([_floats(table, q, n)[sel] for q in ('x0', 'y0', 'x1', 'y1')], -1)
    return np.concatenate([frame[:, None].astype(np.float64),
                           c[:, None].astype(np.float64), box], axis=1)


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


# groups and sessions

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
    _src: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def dir(self) -> Path:
        return self.session.path / 'groups' / self.group_id

    def source(self, cam: str) -> tuple[str, Path, str]:
        """('frames', dir, ext) or ('video', file, '') for one camera. Cached.

        `pixels` costs three stats and the extension needs one more, and neither answer changes
        during a run -- but the loader asks once per camera per item, so both were being paid
        168 times per allen-mouse window.

        The extension is all a caller needs to compute any frame's path: §12 of the spec
        guarantees an image dir holds exactly `%06d.<ext>` contiguous from `000000` with one
        extension per directory, and `validate_session` enforces it. Listing the directory
        instead cost 0.90 s of a 1.06 s rat-city item, whose `cam0` holds 57,594 entries.
        """
        if cam not in self._src:
            kind, p = self.pixels(cam)
            ext = ''
            if kind == 'frames':
                ext = next((e for e in IMAGE_EXTS if (p / f'{0:06d}{e}').exists()), '')
                if not ext:
                    raise FormatError(f'{p}: no 000000{{{",".join(IMAGE_EXTS)}}} -- frame names '
                                      'must be contiguous %06d (see docs/annotation_format.md)')
            self._src[cam] = (kind, p, ext)
        return self._src[cam]

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
    label_source: str               # session.toml `labels`: 'annotated' | 'tracked'
    names: list[str]                # keypoint names, THE authority for the keypoint axis
    rig: Rig
    groups: dict[str, Group]
    skeleton: list[list[str]] = field(default_factory=list)
    flip_pairs: list[list[str]] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    assoc_res_max_px: float = 30.0
    _label_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def session_id(self) -> str:
        return self.path.name

    @property
    def split(self) -> str:
        return self.path.parent.name

    @property
    def cameras(self):
        return self.rig.cameras

    @property
    def cam_names(self) -> list[str]:
        return self.rig.names

    @property
    def n_keypoints(self) -> int:
        return len(self.names)

    @cached_property
    def has_visibility_assessment(self) -> bool:
        """Whether `keypoints.pq` ever records a real occlusion judgement (`status == missing`).

        A `tracked` session whose table is 100% `visible` recorded no negative at all: calms21's
        MARS pipeline emits all 7 keypoints unconditionally (`[provenance].visibility` already
        disclaims it as not an assessment), and rat-city-tracked and branson-fly are the same
        shape. The per-window NaN-masking that handles an all-`projected` session (dataset.py)
        cannot catch this case the same way -- every row here IS `visible`, i.e. finite, not NaN
        -- so the loader gates on this SESSION-LEVEL fact instead: no `missing` row anywhere in
        the table means no assessment happened anywhere in it, and the visibility target must be
        withheld rather than trained as "always visible" from labels nobody wrote as a judgement.

        An `annotated` session is exempt even at zero `missing` rows -- a hand-labelled root with
        no occluded points on record is still an assessment, just one with no negatives (yet).
        The gate only fires on `labels == "tracked"`.
        """
        if self.label_source != 'tracked':
            return True
        t = self._tables['keypoints']
        if t is None:
            return True
        codes, vocab = _codes(t, 'status')
        if 'missing' not in vocab:
            return False
        return bool((codes == vocab.index('missing')).any())

    @cached_property
    def _kpt_vocab(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    @cached_property
    def _cam_vocab(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.rig.names)}

    def _table(self, stem: str) -> pa.Table | None:
        p = self.path / f'{stem}.pq'
        return pq.read_table(p) if p.exists() else None

    @cached_property
    def _tables(self) -> dict[str, pa.Table | None]:
        return {s: self._table(s)
                for s in ('keypoints', 'points3d', 'instances', 'regions', 'extrinsics')}

    @classmethod
    def load(cls, path: Path) -> 'Session':
        path = Path(path)
        cfg_path = path / 'session.toml'
        if not cfg_path.exists():
            raise FormatError(f'{path}: no session.toml')
        with open(cfg_path, 'rb') as f:
            cfg = tomllib.load(f)
        for key in ('mode', 'units', 'labels', 'names'):
            if key not in cfg:
                raise FormatError(f'{cfg_path}: missing {key!r}')
        if cfg['mode'] not in ('2d', '3d'):
            raise FormatError(f'{cfg_path}: mode must be "2d" or "3d", got {cfg["mode"]!r}')
        if cfg['labels'] not in LABEL_SOURCES:
            raise FormatError(f'{cfg_path}: labels must be one of {LABEL_SOURCES}, '
                              f'got {cfg["labels"]!r}')
        if not cfg['names']:
            raise FormatError(f'{cfg_path}: names is empty')

        rig = load_calibration(path / 'calibration.toml')

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
            label_source=cfg['labels'],
            names=list(cfg['names']),
            rig=rig,
            groups=groups,
            skeleton=[list(p) for p in cfg.get('skeleton', [])],
            flip_pairs=[list(p) for p in cfg.get('flip_pairs', [])],
            provenance=dict(cfg.get('provenance', {})),
            assoc_res_max_px=float(cfg.get('assoc_res_max_px', 30.0)),
        )
        for g in groups.values():
            g.session = sess
        return sess

    def preload(self) -> None:
        """Scatter every group now and drop the parquet tables.

        Call this in the PARENT process before forking dataloader workers: the dense arrays are
        then shared copy-on-write, where lazy per-worker scattering would give 12 workers their
        own copy of a 44 MB table apiece.
        """
        for gid in self.groups:
            self.labels(gid)
        self.__dict__.pop('_tables', None)

    def cgroup(self, gid: str, frames=None) -> list[dict]:
        """posetail cameras for a group, carrying per-frame extrinsics where a camera moves.

        THE one place a camera group is built. There were five, and four of them silently dropped
        `moving_ext` and used the static extrinsic instead.

        `frames`:
          - None       -> the whole group; a moving camera's `ext` is (T,4,4)
          - a sequence -> (T_win,4,4) aligned to that window
          - an INT     -> static (4,4) cameras at that one frame

        The int form is what the per-frame consumers need. `project_cam` aligns a (T,4,4)
        extrinsic against axis -3 of the points (cube.py:95-99), and cross-view association and
        the box loader pass (S,K,3) whose axis -3 is the ANIMAL, not time -- so handing them a
        (T,4,4) camera silently projects animal `i` through frame `i`'s pose.
        """
        if not any(self.rig.moving.values()):
            return self.rig.posetail()

        import torch
        ext = self.labels(gid).ext                      # (C,T,4,4), coverage already checked
        sel = slice(None) if frames is None else frames
        # EVERY camera gets the per-frame form, not just the moving ones. `_decode_from_scene`
        # stacks `cam['ext']` across cameras (tracker_encoder.py:623), so a mixed rig with one
        # (T,4,4) and two (4,4) cameras is a stack error. `labels()` already back-fills a static
        # camera's rows with its own constant pose, so this is the same geometry either way.
        moving_ext = {n: torch.as_tensor(ext[i][sel], dtype=torch.float)
                      for i, n in enumerate(self.cam_names)}
        return self.rig.posetail(moving_ext=moving_ext)

    def labels(self, gid: str) -> Labels:
        """Scatter one group's rows into dense arrays. See docs/annotation_format.md §12."""
        if gid in self._label_cache:
            return self._label_cache[gid]
        group = self.groups[gid]
        T, K, C = group.n_frames, len(self.names), len(self.rig)
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
                m = np.isin(status, POSITIONED)
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
                m = np.isin(status, POSITIONED)
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

        regions = None
        if t['regions'] is not None:
            regions = _regions(t['regions'], gid, T, self._cam_vocab, f'{where}/regions')

        ext = None
        moving = [n for n in self.rig.names if self.rig.moving[n]]
        if moving:
            # A gap here is NOT benign: the array is pre-filled with eye(4), which is a
            # perfectly valid-looking extrinsic that puts the camera at the world origin. So the
            # coverage is checked HERE rather than only in validate_session -- every consumer
            # goes through labels(), and validation is opt-in.
            tab = t['extrinsics']
            gvals = _codes(tab, 'group_id')[1] if tab is not None else []
            if tab is None or gid not in gvals:
                raise FormatError(
                    f'{where}: cameras {moving} are moving=true but extrinsics.pq has no rows '
                    f'for this group')
            gcodes, _ = _codes(tab, 'group_id')
            sel = np.flatnonzero(gcodes == gvals.index(gid))
            ccodes, cvals = _codes(tab, 'camera')
            c = _remap(ccodes, cvals, self._cam_vocab, 'camera', where, sel)
            f = tab.column('frame').combine_chunks().to_numpy(
                zero_copy_only=False)[sel].astype(np.int64)
            if f.size and (f.min() < 0 or f.max() >= T):
                raise FormatError(f'{where}: extrinsics has a frame outside [0, {T})')
            seen = np.zeros((C, T), bool)
            seen[c, f] = True
            for i, name in enumerate(self.rig.names):
                if self.rig.moving[name] and not seen[i].all():
                    raise FormatError(
                        f'{where}: moving camera {name!r} has extrinsics for '
                        f'{int(seen[i].sum())} of {T} frames; a gap would silently become the '
                        f'identity pose')
            raw = np.stack(tab.column('ext').combine_chunks().to_numpy(
                zero_copy_only=False)[sel]).astype(np.float64)
            ext = np.tile(np.eye(4), (C, T, 1, 1))
            ext[c, f] = raw.reshape(-1, 4, 4)
            for i, cam in enumerate(self.rig.cameras):
                if not self.rig.moving[cam.get_name()]:
                    ext[i] = cam.get_extrinsics_mat().detach().cpu().numpy()

        out = Labels(animal_ids=animals, points3d=points3d, vis3d=vis3d, points2d=points2d,
                     vis2d=vis2d, boxes=boxes, instance=instance, ext=ext, regions=regions)
        self._label_cache[gid] = out
        return out


# writing -- the exact inverse of Session.labels, so a round-trip test is meaningful

def write_session(path: Path, *, mode: str, units: str, label_source: str, names: list[str],
                  rig: Rig, groups: dict[str, Group], labels: dict[str, Labels],
                  skeleton=(), flip_pairs=(), provenance=None,
                  assoc_res_max_px: float | None = None) -> None:
    """Write session.toml, calibration.toml and the label tables. Pixels are the caller's job.

    `labels` maps group_id -> dense arrays in exactly the layout `Session.labels` returns; a row
    is emitted only where the status array says a determination was made, which is what makes
    sparse hand annotation and dense tracking the same code path.

    `label_source` is written as the `labels` KEY of session.toml -- the two names differ because
    this function's `labels` argument is already the label tables. It is required and not
    defaulted on purpose: a converter has to state whether a human or a machine produced these
    points, and a default would let the next converter quietly inherit someone else's answer.
    """
    import toml

    if label_source not in LABEL_SOURCES:
        raise FormatError(f'{path}: label_source must be one of {LABEL_SOURCES}, '
                          f'got {label_source!r}')
    path.mkdir(parents=True, exist_ok=True)
    cfg = {'mode': mode, 'units': units, 'labels': label_source, 'names': list(names)}
    if skeleton:
        cfg['skeleton'] = [list(p) for p in skeleton]
    if flip_pairs:
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
        'regions': {c: [] for c in
                    ('group_id', 'frame', 'camera', 'x0', 'y0', 'x1', 'y1', 'status')},
        'extrinsics': {c: [] for c in ('group_id', 'frame', 'camera', 'ext')},
    }
    # A session that certifies nothing anywhere still writes an EMPTY regions.pq if any group
    # said `regions is not None` -- the file's absence is the claim "exhaustively labelled" (§9b),
    # so the row count cannot be what decides whether it exists.
    emit_regions = any(lab is not None and lab.regions is not None for lab in labels.values())
    # A session with no labelled row anywhere (an inference-only clip built on `empty_labels`)
    # writes zero rows into the mode's own table -- but rule 6 requires that table to EXIST, so
    # the empty-file-is-a-claim precedent above applies here too, keyed on the array being
    # PRESENT rather than on any row being labelled.
    emit_points3d = any(lab is not None and lab.vis3d is not None for lab in labels.values())
    emit_keypoints = any(lab is not None and lab.vis2d is not None for lab in labels.values())

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

        if lab.regions is not None and len(lab.regions):
            r = np.asarray(lab.regions, np.float64).reshape(-1, 6)
            push('regions', group_id=[gid] * len(r), frame=r[:, 0].astype(np.int32),
                 camera=np.asarray(cam_names, dtype=object)[r[:, 1].astype(np.int64)],
                 x0=r[:, 2], y0=r[:, 3], x1=r[:, 4], y1=r[:, 5],
                 status=['labelled_complete'] * len(r))

        if lab.ext is not None:
            for ci, name in enumerate(cam_names):
                if not rig.moving[name]:
                    continue
                T = lab.ext.shape[1]
                push('extrinsics', group_id=[gid] * T, frame=np.arange(T, dtype=np.int32),
                     camera=[name] * T, ext=[e.ravel().tolist() for e in lab.ext[ci]])

    for stem, cols in tables.items():
        if not cols['group_id'] and not (stem == 'regions' and emit_regions) \
                and not (stem == 'points3d' and emit_points3d) \
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


# discovery

@dataclass
class Dataset:
    name: str
    root: Path
    sessions: dict[str, list[Session]]      # split -> sessions

    @property
    def names(self) -> list[str]:
        """The root's keypoint axis: the UNION of its sessions' names, in load order.

        A session may declare the same names in a different order, or only a subset of them --
        `allen-mouse-combined` holds 80 hand-annotated sessions in anatomical order beside a
        tracked one in name-sorted order. Ids are handed out against THIS list and then remapped
        per session by `Registry.ids_for`, which is what keeps a session's dense K axis (built
        from its own `_kpt_vocab`) attached to the right embedding rows.

        Order is deterministic -- `SPLITS` is a fixed tuple and `load_dataset` sorts `iterdir()`
        -- and for a root whose sessions all agree this is exactly the first session's list, so
        no existing registry's ids move.
        """
        names: list[str] = []
        seen: set[str] = set()
        for sess in self.all_sessions():
            for n in sess.names:
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        if not names:
            raise FormatError(f'{self.root}: no sessions')
        return names

    def all_sessions(self) -> list[Session]:
        return [s for group in self.sessions.values() for s in group]


def _is_dataset_root(path: Path) -> bool:
    return any((path / s).is_dir() for s in SPLITS)


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


def sessions_for(path: Path, split: str) -> tuple[str, list['Session']]:
    """(dataset_name, sessions) from either ONE session directory or a dataset root.

    A path is a session iff it holds `session.toml`; that is the format's own marker, and it is
    the one rule both `scripts/infer.py` and `scripts/eval.py` need.
    """
    path = Path(path)
    if (path / 'session.toml').exists():
        return path.parent.parent.name, [Session.load(path)]
    ds = load_dataset(path)
    return ds.name, ds.sessions.get(split, [])


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


# keypoint registry

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

    def ids_for_dataset(self, dataset: str) -> np.ndarray:
        """Global ids in the DATASET's own name order, i.e. aligned to `Dataset.names`."""
        for name, ids in self.datasets:
            if name == dataset:
                return np.asarray(ids, dtype=np.int64)
        raise KeyError(f'{dataset!r} is not in this registry: {[d for d, _ in self.datasets]}')

    def local_names(self, dataset: str) -> list[str]:
        """That dataset's own keypoint names, un-prefixed, in registry-id order."""
        got = [self.names[i] for i in self.ids_for_dataset(dataset)]
        p = f'{dataset}-'
        return [n[len(p):] for n in got] if all(n.startswith(p) for n in got) else got

    def ids_for(self, dataset: str, names) -> np.ndarray:
        """Global ids ALIGNED TO `names` -- the keypoint axis the caller actually holds.

        `names` is required and not defaulted. A per-dataset id vector applied to a session that
        declares the same names in a different order is a silent relabel: the session scatters
        its rows through its own `_kpt_vocab`, so `nose` coordinates would train the `L-ear`
        embedding row and nothing downstream would notice (gotcha #4). A session may carry any
        SUBSET of the dataset's names, in any order; a name the dataset does not have is an
        error, loudly.
        """
        ids = self.ids_for_dataset(dataset)
        local = self.local_names(dataset)
        names = list(names)
        if names == local:
            return ids
        ix = {n: i for i, n in enumerate(local)}
        unknown = [n for n in names if n not in ix]
        if unknown:
            raise FormatError(
                f'{dataset}: keypoints {unknown} are not in this registry. The registry holds '
                f'{len(local)} names for this dataset; a session may reorder them or use a '
                f'subset, but not invent one. Retrain, or fix the session\'s `names`.')
        return ids[[ix[n] for n in names]]

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


# validation -- docs/annotation_format.md §11

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

    # 10. a positioned row (visible or projected) carries its coordinates
    positioned = ('visible', 'projected')
    t3 = sess._tables['points3d']
    if t3 is not None and len(t3):
        st, vals = _codes(t3, 'status')
        vis = np.isin(st, [i for i, v in enumerate(vals) if v in positioned])
        xyz = np.stack([_floats(t3, c, len(t3)) for c in 'xyz'], -1)
        if vis.any() and not np.isfinite(xyz[vis]).all():
            bad(10, 'points3d.pq has a visible row without x,y,z')
        if (~vis).any() and np.isfinite(xyz[~vis]).any():
            bad(10, 'points3d.pq has a missing/unlabeled row carrying coordinates')

    tk = sess._tables['keypoints']
    if tk is not None and len(tk):
        st, vals = _codes(tk, 'status')
        vis = np.isin(st, [i for i, v in enumerate(vals) if v in positioned])
        xy = np.stack([_floats(tk, c, len(tk)) for c in 'xy'], -1)
        if vis.any() and not np.isfinite(xy[vis]).all() and t3 is None:
            bad(10, 'keypoints.pq has a visible row without x,y and there is no 3D layer')
        if (~vis).any() and np.isfinite(xy[~vis]).any():
            bad(10, 'keypoints.pq has a missing/unlabeled row carrying coordinates')

    # 15. regions: known status, non-empty rectangles. Camera and frame are checked by labels().
    tr = sess._tables['regions']
    if tr is not None and len(tr):
        _, vals = _codes(tr, 'status')
        unknown = sorted(set(vals) - set(REGION_STATUS))
        if unknown:
            bad(15, f'regions.pq has unknown status {unknown[0]!r}')
        n = len(tr)
        x0, y0, x1, y1 = (_floats(tr, q, n) for q in ('x0', 'y0', 'x1', 'y1'))
        empty = ~((x1 > x0) & (y1 > y0))
        if empty.any():
            bad(15, f'regions.pq has {int(empty.sum())} empty rectangle(s) (x1<=x0 or y1<=y0); a '
                    f'certificate covering nothing is a converter bug, not a no-op')

    # 13. extrinsics only for cameras declared moving, and EVERY frame of every moving camera.
    # A missing frame is not a gap: labels() pre-fills eye(4), so it reads as a real pose at the
    # world origin. Reported here as well as raised in labels() so a bulk validate lists every
    # bad session instead of stopping at the first.
    te = sess._tables['extrinsics']
    moving = {n for n in sess.cam_names if sess.rig.moving[n]}
    if te is not None and len(te):
        _, cams = _codes(te, 'camera')
        for name in cams:
            if name not in moving:
                bad(13, f'extrinsics.pq names camera {name!r}, which is not moving=true')
        gcodes, gvals = _codes(te, 'group_id')
        frames = te.column('frame').combine_chunks().to_numpy(zero_copy_only=False)
        ccodes, cvals = _codes(te, 'camera')
        for gid, g in sess.groups.items():
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

    # 6/7/8. per group: frames in range, pixels present and the right shape
    for gid, g in sess.groups.items():
        for cname in sess.cam_names:
            try:
                kind, p = g.pixels(cname)
            except FormatError as e:
                bad(7, str(e))
                continue
            if kind != 'frames':
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
            sess.labels(gid)          # raises on unknown bodypart/camera/animal or bad frame
        except FormatError as e:
            errs.append(str(e))
    return errs


def validate_dataset(ds: Dataset, check_images: bool = True) -> list[str]:
    errs: list[str] = []
    sessions = ds.all_sessions()
    if not sessions:
        return [f'{ds.root}: no sessions']
    # 3. cross-session agreement on names. A WARNING, not an error: `Registry.ids_for` resolves a
    # session's axis BY NAME, so a reordering or a subset is handled rather than mislabelled. It
    # is still worth saying out loud -- a "missing" keypoint is usually a typo, not a decision.
    axis = ds.names
    for s in sessions:
        if s.names == axis:
            continue
        missing = [n for n in axis if n not in set(s.names)]
        what = (f'is missing {len(missing)} of the root\'s {len(axis)} keypoints ({missing})'
                if missing else 'declares the root\'s keypoints in a different order')
        errs.append(f'{s.path}: [rule 3 WARNING] {what}; resolved by name')
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
