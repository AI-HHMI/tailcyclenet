"""In-memory sessions: `Labels` (dense per-group label arrays), `Group`, `Session`, the in-memory
`VideoSession`, `empty_labels`, and `triangulate_group`."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import FormatError, IMAGE_EXTS, INST_LABELED, INST_NONE, INST_STATUS, KPT_STATUS, LABEL_SOURCES, MISSING, POSITIONED, UNLABELED, VIDEO_EXTS, VISIBLE
from .calibration import Rig, load_calibration
from .tables import Columns, column_codes, column_floats, column_ints, group_rows, remap_codes


@dataclass
class Labels:
    """One group's labels as dense arrays. See docs/annotation_format.md §12."""
    animal_ids: list[str]
    # (S,T,K,3) float32, NaN where not visible
    points3d: np.ndarray | None
    # (S,T,K) int8
    vis3d: np.ndarray | None
    # (S,T,K,C,2) float32
    points2d: np.ndarray | None
    # (S,T,K,C) int8
    vis2d: np.ndarray | None
    # (S,T,C,4) float32
    boxes: np.ndarray | None
    # (S,T,C) int8
    instance: np.ndarray | None
    # (C,T,4,4) float64, only when a camera is moving
    ext: np.ndarray | None = None

    @property
    def n_animals(self) -> int:
        """Number of animals in this group's labels."""
        return len(self.animal_ids)


def _scatter(table, gid, T, kpt_vocab, cam_vocab, animals, where, per_camera, n_extra):
    """Common scatter: pick this group's rows and index them into dense arrays.

    Returns (rows_index, a, f, k, c, status) with `a` already remapped to 0..S-1 and `c` None
    when the table has no camera column.
    """
    sel = group_rows(table, gid)
    if sel is None:
        return None

    frame = column_ints(table, 'frame')[sel]
    if frame.min() < 0 or frame.max() >= T:
        raise FormatError(f'{where}: group {gid!r} has a frame outside [0, {T})')

    acodes, avals = column_codes(table, 'animal_id')
    a = remap_codes(acodes, avals, animals, 'animal_id', where, sel)

    k = None
    if 'bodypart' in table.column_names:
        kcodes, kvals = column_codes(table, 'bodypart')
        k = remap_codes(kcodes, kvals, kpt_vocab, 'bodypart', where, sel)

    c = None
    if per_camera:
        ccodes, cvals = column_codes(table, 'camera')
        c = remap_codes(ccodes, cvals, cam_vocab, 'camera', where, sel)

    scodes, svals = column_codes(table, 'status')
    enum = INST_STATUS if n_extra == 4 else KPT_STATUS
    status = remap_codes(scodes, svals, enum, 'status', where, sel)

    return sel, a, frame, k, c, status


def nonempty_box(box: np.ndarray) -> np.ndarray:
    """(...,4) xyxy -> (...) bool: finite and `x1 > x0`, `y1 > y0` (spec §9 box encoding)."""
    box = np.asarray(box, np.float64)
    with np.errstate(invalid='ignore'):
        return (np.isfinite(box).all(-1) & (box[..., 2] > box[..., 0])
                & (box[..., 3] > box[..., 1]))


def boxless_labeled(instance: np.ndarray, boxes: np.ndarray | None) -> int:
    """Count `labeled` cells of a dense `(S,T,C)` instance array with no usable box (rule 11)."""
    labeled = np.asarray(instance) == INST_LABELED
    if boxes is None:
        return int(labeled.sum())
    return int((labeled & ~nonempty_box(boxes)).sum())


def _animal_vocab(tables: list[pa.Table | None], gid: str) -> list[str]:
    """Distinct animal_ids appearing in this group, sorted. Defines S and the row order."""
    found: set[str] = set()
    for table in tables:
        if table is None:
            continue
        sel = group_rows(table, gid)
        if sel is None:
            continue
        acodes, avals = column_codes(table, 'animal_id')
        found.update(np.asarray(avals, dtype=object)[np.unique(acodes[sel])].tolist())
    return sorted(found)


def triangulate_group(rig: Rig, lab: Labels, gate: float) -> tuple[int, int, int]:
    """Derive 3D labels from per-camera positions; return (visible, gated, missing).

    Stored 2D coordinates are in image pixels, so camera offsets are restored before
    triangulation. `visible` and `projected` both carry positions; `missing` does not.
    A point with at least two views is accepted when its median reprojection residual is
    within `gate`; failures and single-view points remain unlabelled. Zero-view points are
    missing only when every camera was assessed.

    Moving-camera rigs are refused: callers must provide frame-specific camera geometry rather
    than silently triangulating against the static calibration.
    """
    if lab.points2d is None or lab.vis2d is None:
        raise ValueError('triangulation requires per-camera 2D labels and visibility')
    if any(rig.moving.values()):
        raise FormatError('cannot triangulate keypoints-only 3D session with moving cameras; '
                          'frame-specific extrinsics are required')
    import torch

    S, T, K, C, _ = lab.points2d.shape
    off = np.asarray([rig.offset[n] for n in rig.names], dtype=np.float64)
    p2 = lab.points2d.astype(np.float64) + off.reshape(1, 1, 1, C, 2)
    lab.points3d = np.full((S, T, K, 3), np.nan, np.float32)
    lab.vis3d = np.full((S, T, K), UNLABELED, np.int8)

    nviews = np.isfinite(p2).all(-1).sum(-1)
    idx = np.flatnonzero(nviews.reshape(-1) >= 2)
    flat_vis, flat_p3 = lab.vis3d.reshape(-1), lab.points3d.reshape(-1, 3)
    n_visible = n_gated = 0
    if idx.size:
        xy = np.ascontiguousarray(p2.reshape(-1, C, 2)[idx].transpose(1, 0, 2))
        with torch.no_grad():
            xyz = np.asarray(rig.cgroup.triangulate(xy, progress=False))
            reproj = np.asarray(rig.cgroup.reprojection_error(xyz, xy, mean=False))
        residual = np.linalg.norm(reproj, axis=-1)
        seen = np.isfinite(xy).all(-1)
        median = np.nanmedian(np.where(seen, residual, np.nan), axis=0)
        good = np.isfinite(xyz).all(-1) & (median <= gate)
        flat_vis[idx[good]] = VISIBLE
        flat_p3[idx[good]] = xyz[good].astype(np.float32)
        n_visible, n_gated = int(good.sum()), int((~good).sum())

    assessed_all = (lab.vis2d != UNLABELED).all(-1)
    missing = ((nviews == 0) & assessed_all).reshape(-1)
    flat_vis[missing] = MISSING
    return n_visible, n_gated, int(missing.sum())


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
        """The group's pixels directory under the session."""
        return self.session.path / 'groups' / self.group_id

    def source(self, cam: str) -> tuple[str, Path, str]:
        """('frames', dir, ext) or ('video', file, '') for one camera. Cached.

        The extension is all a caller needs: spec \u00a712 guarantees `%06d.<ext>` contiguous from
        000000 with one extension per directory.
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
        """Sorted frame image paths for one camera; raises if it is a video."""
        kind, p = self.pixels(cam)
        if kind != 'frames':
            raise FormatError(f'{p}: is a video, not a frame directory')
        files = sorted(f for f in p.iterdir() if f.suffix in IMAGE_EXTS)
        return files

    def labels(self) -> Labels:
        """This group's dense labels, from the session's cache."""
        return self.session.labels(self.group_id)


@dataclass
class Session:
    path: Path
    # '2d' | '3d'
    mode: str
    units: str
    # session.toml `labels`: 'annotated' | 'tracked'
    label_source: str
    # keypoint names, THE authority for the keypoint axis
    names: list[str]
    rig: Rig
    groups: dict[str, Group]
    skeleton: list[list[str]] = field(default_factory=list)
    flip_pairs: list[list[str]] = field(default_factory=list)
    # Whether session.toml explicitly declared ``flip_pairs`` (including an empty list).
    flip_pairs_declared: bool = False
    provenance: dict = field(default_factory=dict)
    assoc_res_max_px: float = 30.0
    # Prediction outputs have narrowly different coordinate/status semantics. Missing on legacy
    # annotation sessions means False; complete is optional for pre-marker sessions.
    prediction_session: bool = False
    complete: bool | None = None
    _label_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def session_id(self) -> str:
        """The session folder's name."""
        return self.path.name

    @property
    def split(self) -> str:
        """The parent folder's name: 'train', 'val' or 'test'."""
        return self.path.parent.name

    @property
    def cameras(self):
        """The aniposelib cameras of the rig."""
        return self.rig.cameras

    @property
    def cam_names(self) -> list[str]:
        """Camera names in rig order."""
        return self.rig.names

    @property
    def n_keypoints(self) -> int:
        """Number of keypoints on the session's axis."""
        return len(self.names)

    @cached_property
    def has_visibility_assessment(self) -> bool:
        """Whether `keypoints.pq` ever records a real occlusion judgement (`status == missing`).

        A `tracked` session whose table is 100% `visible` recorded no negative anywhere, so the
        visibility target must be withheld rather than trained as "always visible"; `annotated`
        sessions are exempt even at zero `missing` rows.
        """
        if self.label_source != 'tracked':
            return True
        t = self._tables['keypoints']
        if t is None:
            return True
        codes, vocab = column_codes(t, 'status')
        if 'missing' not in vocab:
            return False
        return bool((codes == vocab.index('missing')).any())

    @cached_property
    def _kpt_vocab(self) -> dict[str, int]:
        """keypoint name -> global axis index."""
        return {n: i for i, n in enumerate(self.names)}

    @cached_property
    def _cam_vocab(self) -> dict[str, int]:
        """camera name -> rig index."""
        return {n: i for i, n in enumerate(self.rig.names)}

    def _table(self, stem: str) -> pa.Table | None:
        """Read one `{stem}.pq` table, or None when the file does not exist."""
        p = self.path / f'{stem}.pq'
        return pq.read_table(p) if p.exists() else None

    @cached_property
    def _tables(self) -> dict[str, pa.Table | None]:
        """Label and prediction tables keyed by stem; missing files are None."""
        return {s: self._table(s)
                for s in ('keypoints', 'points2d', 'points3d', 'instances', 'extrinsics')}

    @classmethod
    def load(cls, path: Path) -> 'Session':
        """Load a Session from a session directory, validating its session.toml.

        Inputs: path -- the session directory.
        Outputs: a populated Session whose groups are wired back to it.
        Side effects: raises FormatError when session.toml or calibration.toml is invalid.
        """
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
            flip_pairs_declared='flip_pairs' in cfg,
            provenance=dict(cfg.get('provenance', {})),
            assoc_res_max_px=float(cfg.get('assoc_res_max_px', 30.0)),
            prediction_session=bool(cfg.get('prediction_session', False)),
            complete=cfg.get('complete'),
        )
        for g in groups.values():
            g.session = sess
        return sess

    @cached_property
    def _columns(self) -> dict[str, Columns | None]:
        """`_tables` with memoised column decodes, for `labels()`. Dropped with `_tables`."""
        return {s: None if t is None else Columns(t) for s, t in self._tables.items()}

    def preload(self) -> None:
        """Scatter every group now and drop the parquet tables.

        Call in the PARENT process before forking dataloader workers: the dense arrays are then
        shared copy-on-write instead of each worker scattering its own copy. Visibility assessment
        is cached while parquet tables are resident, before those tables are dropped, so sampled
        windows do not trigger a reread.
        """
        for gid in self.groups:
            self.labels(gid)
        self.has_visibility_assessment
        self.__dict__.pop('_tables', None)
        self.__dict__.pop('_columns', None)

    def cgroup(self, gid: str, frames=None) -> list[dict]:
        """tracktail cameras for a group, carrying per-frame extrinsics where a camera moves.

        The one place a camera group is built. `frames`: None -> whole group (moving `ext` is
        (T,4,4)); a sequence -> (T_win,4,4) aligned to that window; an INT -> static (4,4) at one
        frame. Per-frame consumers need the int form: `project_cam` aligns `ext` against axis -3,
        which is the ANIMAL for box-loader callers, not time. `ext` is (C,T,4,4) and its coverage
        is already checked. Every camera gets the per-frame form, not just the moving ones:
        `_decode_from_scene` stacks `cam['ext']` across cameras, so a mixed rig is a stack error.
        """
        if not any(self.rig.moving.values()):
            return self.rig.tracktail()

        import torch
        ext = self.labels(gid).ext
        sel = slice(None) if frames is None else frames
        moving_ext = {n: torch.as_tensor(ext[i][sel], dtype=torch.float)
                      for i, n in enumerate(self.cam_names)}
        return self.rig.tracktail(moving_ext=moving_ext)

    def labels(self, gid: str) -> Labels:
        """Scatter one group's rows into dense arrays. See docs/annotation_format.md §12.

        A gap in a moving camera's extrinsics is NOT benign: the array is pre-filled with
        eye(4), a valid-looking extrinsic at the world origin -- so coverage is checked here,
        not only in `validate_session`, since every consumer goes through `labels()`.
        """
        if gid in self._label_cache:
            return self._label_cache[gid]
        group = self.groups[gid]
        T, K, C = group.n_frames, len(self.names), len(self.rig)
        t = self._columns
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
                xyz = np.stack([column_floats(t['points3d'], c, n)[sel] for c in 'xyz'], -1)
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
                xy = np.stack([column_floats(t['keypoints'], q, n)[sel] for q in 'xy'], -1)
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
                box = np.stack([column_floats(t['instances'], q, n)[sel]
                                for q in ('x0', 'y0', 'x1', 'y1')], -1)
                boxes[a, f, c] = box

        ext = None
        moving = [n for n in self.rig.names if self.rig.moving[n]]
        if moving:
            tab = t['extrinsics']
            gvals = column_codes(tab, 'group_id')[1] if tab is not None else []
            if tab is None or gid not in gvals:
                raise FormatError(
                    f'{where}: cameras {moving} are moving=true but extrinsics.pq has no rows '
                    f'for this group')
            gcodes, _ = column_codes(tab, 'group_id')
            sel = np.flatnonzero(gcodes == gvals.index(gid))
            ccodes, cvals = column_codes(tab, 'camera')
            c = remap_codes(ccodes, cvals, self._cam_vocab, 'camera', where, sel)
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
                     vis2d=vis2d, boxes=boxes, instance=instance, ext=ext)
        if (self.mode == '3d' and not self.prediction_session and out.points3d is None
                and out.points2d is not None):
            triangulate_group(self.rig, out, self.assoc_res_max_px)
        self._label_cache[gid] = out
        return out


def video_group(group_id: str, n_frames: int, sources: dict[str, Path], *,
                fps: float = float('nan'), **kw) -> Group:
    """A Group whose pixels are NAMED FILES rather than a `groups/<gid>/` directory.

    Pre-filling `_src` virtualises the pixels: `Group.source` is the only entry point
    `pixels.read_frames` calls, so `pixels()`, `dir` and `session.path` are never reached.
    Every camera must be present (a missing key falls through to `pixels()` and raises for the
    wrong reason, which `adopt.plan` catches earlier). A function, not "the caller pokes `_src`".
    """
    g = Group(group_id=str(group_id), n_frames=int(n_frames), fps=float(fps), **kw)
    g._src = {str(cam): ('video', Path(p), '') for cam, p in sources.items()}
    return g


@dataclass
class VideoSession(Session):
    """A Session with no directory: pixels are named videos, and there are no label tables.

    `path` is a label, not a location -- nothing may be read from it, so `_table` returns None
    rather than silently adopting a colliding directory's parquet. `labels` is overridden rather
    than pre-seeded because `preload()` pops `_tables`, which would recompute it to disk.
    """
    empty: dict[str, Labels] = field(default_factory=dict, repr=False, compare=False)

    def _table(self, stem: str) -> pa.Table | None:
        """No directory exists, so no table is ever read: always None."""
        return None

    def labels(self, gid: str) -> Labels:
        """The pre-seeded empty labels for a group."""
        return self.empty[gid]


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
