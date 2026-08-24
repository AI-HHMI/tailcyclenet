#!/usr/bin/env python
"""Convert an APT (Animal Part Tracker) `.lbl` project into a tailcycle dataset.

    pixi run python scripts/convert_apt_lbl.py --lbl <project.lbl> --out ... --validate

The `.lbl` is a POSIX tar holding a MATLAB-v5 `label_file.lbl`; `labels{i}.p` is x-block-then-
y-block; NaN = unlabeled, Inf = fully occluded. Four conversion decisions are recorded in
`[provenance]`: occ==1 points are written `visible` WITH coordinates (label density over an
honest visibility channel); `tail` -> `tail_base`; `labelsRoi` -> `regions.pq`, written even when
empty (absence would claim exhaustive labelling); one group per labelled frame, label centered.
Pixels are COPIED from the MJPEG AVI (`ffmpeg -c:v copy -bsf:v mjpeg2jpeg`), verified frame-exact.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt, video

CAM = 'cam0'
PAD = 20                    # the crop rule's pad; instances.pq stores the PADDED extent.
APT_NAMES = ['nose', 'left_ear', 'right_ear', 'tail']
RENAME = {'tail': 'tail_base'}
LBL_MEMBER = 'label_file.lbl'
ALIGN_TOL = 1.0             # mean |diff| vs PyAV's decode: ~0.016 right, 2.9+ for a neighbour.


# reading the .lbl

def read_lbl(path: Path, tmp: Path) -> dict:
    """The APT project's variables. Accepts the tar wrapper or a bare MATLAB file."""
    import scipy.io as sio

    src = path
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            member = tf.getmember(LBL_MEMBER)
            tf.extract(member, tmp, filter='data')
        src = tmp / LBL_MEMBER
    want = ['labels', 'labelsGT', 'labelsRoi', 'labelsRoiGT', 'movieFilesAll', 'movieFilesAllGT',
            'movieInfoAll', 'movieInfoAllGT', 'skelNames', 'skeletonEdges', 'cfg', 'projname',
            'VERSION']
    return sio.loadmat(str(src), variable_names=want, struct_as_record=False, squeeze_me=True)


def _strs(v) -> list[str]:
    return [str(x) for x in np.ravel(v)]


def check_axis(d: dict) -> None:
    """Assert the keypoint axis against the name list.

    `skelNames` and `cfg.LabelPointNames` are two independent statements of the same order;
    a transposed axis is invisible in a loss curve.
    """
    if _strs(d['skelNames']) != APT_NAMES:
        raise SystemExit(f'skelNames is {_strs(d["skelNames"])}, expected {APT_NAMES}')
    if _strs(d['cfg'].LabelPointNames) != APT_NAMES:
        raise SystemExit(f'cfg.LabelPointNames is {_strs(d["cfg"].LabelPointNames)}, '
                         f'expected {APT_NAMES}')
    for key, want in (('NumLabelPoints', len(APT_NAMES)), ('NumViews', 1)):
        got = int(np.ravel(getattr(d['cfg'], key))[0])
        if got != want:
            raise SystemExit(f'cfg.{key} is {got}, expected {want}')


def movie_labels(entry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One `labels{i}` cell -> (frm, tgt, xy), all 0-based; Inf kept as the occluded sentinel.

    `occ` is deliberately not returned: APT's occluded-with-a-position points are written
    `visible` (see the module docstring), so the tag changes nothing downstream and returning it
    would imply it does.
    """
    if np.size(getattr(entry, 'frm', [])) == 0:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros((0, len(APT_NAMES), 2)))
    frm = np.ravel(entry.frm).astype(int) - 1
    tgt = np.ravel(entry.tgt).astype(int) - 1
    npts = int(np.ravel(entry.npts)[0])
    # (2*npts, n) -> (2, npts, n) -> (n, npts, 2): a column is a BLOCK of x's then a block of
    # y's, not interleaved pairs -- the interleaved reading puts ~28% of points out of frame.
    xy = np.asarray(entry.p, float).reshape(2, npts, frm.size) - 1.0
    return frm, tgt, np.transpose(xy, (2, 1, 0))


def movie_regions(entry) -> tuple[np.ndarray, np.ndarray]:
    """One `labelsRoi{i}` cell -> (frames, (n,4) [x0,y0,x1,y1]), both 0-based.

    APT's "Label Box" certifies an area as COMPLETELY LABELLED (not an animal box -- it has no
    target index). Axis-alignment is asserted: a rotated ROI would silently become its bounding
    box and certify area the annotator did not.
    """
    v = getattr(entry, 'verts', None)
    if v is None or np.size(v) == 0:
        return np.zeros(0, int), np.zeros((0, 4))
    v = np.asarray(v, float)
    if v.ndim == 2:
        v = v[:, :, None]
    v = v - 1.0                                  # 1-based, like `p` and `frm`
    f = np.ravel(entry.f).astype(int) - 1
    lo, hi = v.min(0), v.max(0)                  # (2,n) each
    corner_on_edge = (np.isclose(v, lo[None]) | np.isclose(v, hi[None])).all()
    if not corner_on_edge:
        raise SystemExit('labelsRoi holds a rotated rectangle; the format stores axis-aligned '
                         'regions and squaring it would certify area nobody marked')
    return f, np.stack([lo[0], lo[1], hi[0], hi[1]], -1)


# what becomes a session, and which split it lands in

_C5_DATE = re.compile(r'/original/.*_20250819_')


def split_of(movie: str, is_gt: bool) -> str:
    """Cross-cohort: cohort5 is held out for val, APT's own GT set is test.

    The dated `original/*_20250819_*` movie joins val -- it sits inside cohort5's date range, so
    leaving it in train would break the cross-cohort claim.
    """
    if is_gt:
        return 'test'
    return 'val' if ('/cohort5/' in movie or _C5_DATE.search(movie)) else 'train'


def session_id(movie: str) -> str:
    """One session per movie, named for the recording.

    Frame size varies across this project, and calibration is a session property -- bucketing by
    movie makes that a non-issue by construction.
    """
    p = Path(movie)
    return p.parent.name if p.stem == 'movie' else p.stem


@dataclass
class Job:
    """One movie: its pixels, its geometry, and the labelled frames to cut groups around."""
    split: str
    session: str
    movie: str
    wh: tuple[int, int]
    fps: float
    n_source: int
    is_gt: bool
    apt_index: int
    frm: np.ndarray
    tgt: np.ndarray
    xy: np.ndarray
    roi_f: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    roi_rect: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    frames: list[int] = field(default_factory=list)


def build_jobs(d: dict) -> list[Job]:
    jobs = []
    for is_gt in (False, True):
        movies = _strs(d['movieFilesAllGT' if is_gt else 'movieFilesAll'])
        labels = np.ravel(d['labelsGT' if is_gt else 'labels'])
        info = np.ravel(d['movieInfoAllGT' if is_gt else 'movieInfoAll'])
        rois = np.ravel(d['labelsRoiGT' if is_gt else 'labelsRoi'])
        for i, (movie, entry) in enumerate(zip(movies, labels)):
            frm, tgt, xy = movie_labels(entry)
            if frm.size == 0:
                continue
            nfo = info[i].info
            roi_f, roi_rect = movie_regions(rois[i]) if i < rois.size else \
                (np.zeros(0, int), np.zeros((0, 4)))
            jobs.append(Job(
                split=split_of(movie, is_gt), session=session_id(movie), movie=movie,
                wh=(int(np.ravel(nfo.Width)[0]), int(np.ravel(nfo.Height)[0])),
                fps=float(np.ravel(nfo.FrameRate)[0]),
                n_source=int(np.ravel(info[i].nframes)[0]),
                is_gt=is_gt, apt_index=i, frm=frm, tgt=tgt, xy=xy,
                roi_f=roi_f, roi_rect=roi_rect,
                frames=sorted(set(frm.tolist()))))
    seen: dict[str, str] = {}
    for j in jobs:
        if seen.setdefault(j.session, j.split) != j.split:
            raise SystemExit(f'session {j.session!r} would land in two splits '
                             f'({seen[j.session]} and {j.split}) -- format rule 14')
    return jobs


# pixels

_readers: dict[str, object] = {}
_readers_lock = threading.Lock()


def reader(movie: str):
    """One PyAV reader per movie, reused. Opening one costs 2-5 s on a 70k-frame AVI."""
    with _readers_lock:
        if movie not in _readers:
            _readers[movie] = video.open_reader(movie)
        return _readers[movie]


def _rgb(path: Path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f'{path}: unreadable')
    return im[:, :, ::-1]


def _diff(a, b) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def copy_window(movie: str, start: int, n: int, fps: float, out: Path) -> int:
    """Lift `n` stored JPEGs out of the MJPEG AVI without re-encoding. Returns files written."""
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{start / fps:.6f}', '-i', movie,
           '-frames:v', str(n), '-c:v', 'copy', '-bsf:v', 'mjpeg2jpeg',
           '-start_number', '0', str(out / '%06d.jpg')]
    subprocess.run(cmd, check=True, capture_output=True)
    return len(list(out.glob('*.jpg')))


def decode_window(movie: str, start: int, n: int, out: Path) -> int:
    """Fallback: decode with PyAV and re-encode. Index-exact by construction, and lossy."""
    import cv2
    vr = reader(movie)
    n = min(n, len(vr) - start)
    imgs = vr.get_batch(list(range(start, start + n)))
    for i, im in enumerate(imgs):
        cv2.imwrite(str(out / f'{i:06d}.jpg'), im[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    return n


def write_window(movie: str, start: int, n: int, fps: float, out: Path, mode: str) -> tuple[int, str]:
    """`n` frames from `start` into `out`, verified to actually START at `start`.

    The alignment check is the whole reason this function exists rather than a bare ffmpeg call:
    `-ss` is a TIME seek, so nothing but a pixel comparison stands between it and a silently
    misaligned window -- and a window off by one frame is a mislabelled window.
    """
    for p in out.glob('*.jpg'):
        p.unlink()
    if mode == 'copy':
        got = copy_window(movie, start, n, fps, out)
        if got and _diff(_rgb(out / '000000.jpg'),
                         reader(movie).get_batch([start])[0]) < ALIGN_TOL:
            return got, 'copy'
        for p in out.glob('*.jpg'):
            p.unlink()
    return decode_window(movie, start, n, out), 'decode'


def check_seek(movie: str, start: int, fps: float, tmp: Path) -> str:
    """Once per movie: is the time seek frame-EXACT, or merely close?

    Demands `start` be the argmin over its neighbours, which no off-by-one survives; `-ss` maps
    time to frame linearly, so once per movie is enough.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob('*.jpg'):
        p.unlink()
    if not copy_window(movie, start, 1, fps, tmp):
        return 'ffmpeg produced no frame'
    got = _rgb(tmp / '000000.jpg')
    vr = reader(movie)
    cand = [f for f in (start - 1, start, start + 1) if 0 <= f < len(vr)]
    diffs = {f: _diff(got, vr.get_batch([f])[0]) for f in cand}
    # Tied, not strictly best: stalled recordings hold byte-identical consecutive frames, and it
    # does not matter which of them came back. Tied-for-minimum still rejects every real off-by-one.
    if diffs[start] > min(diffs.values()) + 1e-6:
        return f'-ss lands on frame {min(diffs, key=diffs.get)}, not {start} (diffs {diffs})'
    return ''


def check_contiguity(movie: str, starts: list[int]) -> str:
    """Is this movie really contiguous video, or a montage of unrelated frames?

    The test is that the frame difference GROWS with temporal distance, not that it is small:
    sensor noise floors all lags equally (so a fixed ratio threshold flags real video), while a
    montage's shuffled frames read the same at every lag.
    """
    vr = reader(movie)
    lags = (1, 10, 100)
    acc = {g: [] for g in lags}
    for s in starts:
        if s + max(lags) >= len(vr):
            continue
        im = vr.get_batch([s, s + 1, s + 10, s + 100]).asnumpy()
        for g, j in zip(lags, (1, 2, 3)):
            acc[g].append(_diff(im[0], im[j]))
    med = {g: float(np.median(v)) for g, v in acc.items() if v}
    if len(med) < len(lags):
        return ''
    if not med[1] < med[10] < med[100]:
        return (f'frame difference does not grow with temporal distance '
                f'(+1 {med[1]:.2f}, +10 {med[10]:.2f}, +100 {med[100]:.2f}) -- '
                f'is this contiguous video?')
    return ''


# labels

def padded_extent(pts: np.ndarray, wh) -> np.ndarray:
    """(K,2) source px -> [x0,y0,x1,y1], the padded extent of the finite points, or NaN.

    Steps 1-2 of the crop rule (not step 3), as the tracked root stores it: a consumer re-enters
    the real rule at `pad=0` and reproduces it.
    """
    finite = np.isfinite(pts).all(-1)
    if not finite.any():
        return np.full(4, np.nan, np.float32)
    p, wh = pts[finite], np.asarray(wh, np.float32)
    lo = np.clip(p.min(0) - PAD, 0, wh)
    hi = np.clip(p.max(0) + PAD, 0, wh)
    # int32 truncation, as the rule does it -- re-truncating the float is then a no-op.
    return np.concatenate([lo, hi]).astype(np.int32).astype(np.float32)


def window_start(frame: int, n_source: int, context: int) -> tuple[int, int]:
    """(start, n) for a group centered on `frame`, clamped into the movie.

    Centering matters: frame 0 is the one frame where per-frame anchoring contributes nothing.
    """
    n = min(context, n_source)
    return int(np.clip(frame - n // 2, 0, max(0, n_source - n))), n


def group_labels(job: Job, frame: int, start: int, n: int, K: int) -> fmt.Labels:
    """The dense arrays for one group: one labelled frame at `frame - start`, the rest unlabeled.

    NaN -> no row (an unlabeled skip, not an assessment); Inf -> `missing` with null coordinates.
    """
    rows = np.flatnonzero(job.frm == frame)
    slots = job.tgt[rows]
    lab = fmt.empty_labels(int(slots.max()) + 1, n, K, 1, mode3d=False)
    lab.boxes = np.full((len(lab.animal_ids), n, 1, 4), np.nan, np.float32)
    lab.instance = np.full((len(lab.animal_ids), n, 1), fmt.INST_NONE, np.int8)
    lf = frame - start
    for r, a in zip(rows, slots):
        pts = job.xy[r]                                             # (K, 2)
        occluded = np.isinf(pts).any(-1)
        positioned = np.isfinite(pts).all(-1)
        lab.vis2d[a, lf, occluded, 0] = fmt.MISSING
        lab.vis2d[a, lf, positioned, 0] = fmt.VISIBLE
        lab.points2d[a, lf, positioned, 0] = pts[positioned].astype(np.float32)
        if (occluded | positioned).any():
            lab.instance[a, lf, 0] = fmt.INST_LABELED
            lab.boxes[a, lf, 0] = padded_extent(pts, job.wh)
    return lab


def group_regions(job: Job, frame: int, start: int, wh) -> tuple[np.ndarray, int]:
    """Region rows for one group: the ROIs on ITS OWN labelled frame, and no others.

    A Label Box certifies one source frame; the claim is only sound in the group carrying that
    frame's labels. Returns ((M,6) in `Labels.regions` layout, n_clipped_away).
    """
    r = job.roi_rect[np.flatnonzero(job.roi_f == frame)]
    out = np.zeros((len(r), 6))
    out[:, 0] = frame - start                    # group-local frame index
    out[:, 1] = 0                                # the single camera
    out[:, 2:] = np.clip(r, 0.0, np.tile(np.asarray(wh, float), 2))
    keep = (out[:, 4] > out[:, 2]) & (out[:, 5] > out[:, 3])
    return out[keep], int((~keep).sum())


# one movie -> one session

def convert_movie(job: Job, out: Path, context: int, mode: str, args) -> dict:
    names = [RENAME.get(nm, nm) for nm in APT_NAMES]
    dst = out / job.split / job.session
    if dst.exists() and args.clean:
        shutil.rmtree(dst)
    stat = {'session': job.session, 'split': job.split, 'groups': 0, 'frames': 0,
            'decode_fallback': 0, 'decode_movie': False, 'skipped': 0, 'regions': 0,
            'roi_off_label': int((~np.isin(job.roi_f, job.frames)).sum()), 'warnings': []}

    starts = {f: window_start(f, job.n_source, context) for f in job.frames}

    if not args.dry_run and not args.labels_only:
        with tempfile.TemporaryDirectory() as td:
            err = check_seek(job.movie, starts[job.frames[0]][0], job.fps, Path(td))
        if err:
            if mode == 'copy':
                stat['warnings'].append(f'seek check failed, decoding instead: {err}')
                mode, stat['decode_movie'] = 'decode', True
            else:
                stat['warnings'].append(f'seek check: {err}')
        err = check_contiguity(job.movie, [s for s, _ in list(starts.values())[:3]])
        if err:
            stat['warnings'].append(err)

    groups, labels, extract = {}, {}, None
    # `--labels-only`: the extracted frames are already correct, so a tables-only change must not
    # re-run ffmpeg. The groups on disk stay the authority for `n_frames`/`source_frame_start`.
    reuse = args.labels_only and not args.dry_run
    if reuse:
        if not (dst / 'session.toml').exists():
            raise SystemExit(f'{dst}: --labels-only, but this session has not been converted yet')
        old = fmt.Session.load(dst)
        extract = old.provenance.get('extract')      # how the pixels on disk were really made
        for gid, g in old.groups.items():
            groups[gid] = fmt.Group(gid, g.n_frames, fps=g.fps, source_video=g.source_video,
                                    source_frame_start=g.source_frame_start,
                                    source_frame_step=g.source_frame_step, notes=g.notes)
            labels[gid] = group_labels(job, int(gid[1:]), g.source_frame_start, g.n_frames,
                                       len(names))
            stat['groups'] += 1
            stat['frames'] += g.n_frames

    for f in ([] if reuse else job.frames):
        (start, n), gid = starts[f], f'f{f:06d}'
        if args.dry_run:
            groups[gid], stat['groups'], stat['frames'] = None, stat['groups'] + 1, stat['frames'] + n
            continue
        cdir = dst / 'groups' / gid / CAM
        cdir.mkdir(parents=True, exist_ok=True)
        got, used = write_window(job.movie, start, n, job.fps, cdir, mode)
        stat['decode_fallback'] += used == 'decode' and mode == 'copy'
        # The project file's `nframes` is a claim about the container; truncate to what came out,
        # and drop the group if the labelled frame is past the end or fewer than two frames.
        if got < 2 or f - start >= got:
            shutil.rmtree(dst / 'groups' / gid)
            stat['skipped'] += 1
            stat['warnings'].append(f'{gid}: {got} frames extracted from {start}, dropped')
            continue
        groups[gid] = fmt.Group(gid, got, fps=job.fps, source_video=job.movie,
                                source_frame_start=start, source_frame_step=1)
        labels[gid] = group_labels(job, f, start, got, len(names))
        stat['groups'] += 1
        stat['frames'] += got

    if args.dry_run or not groups:
        return stat

    # Every group gets a `regions` array, empty or not: absence is the format's claim of
    # exhaustive labelling, and a label-sparse root is the opposite. The GT sessions carry no
    # ROIs, so they get an empty regions.pq rather than a missing one.
    clipped = 0
    for gid, g in groups.items():
        labels[gid].regions, n = group_regions(job, int(gid[1:]), g.source_frame_start, job.wh)
        stat['regions'] += len(labels[gid].regions)
        clipped += n
    if clipped:
        stat['warnings'].append(f'{clipped} Label Box(es) clipped away as empty')

    cam = fmt.nominal_camera(CAM, job.wh)
    from aniposelib.cameras import CameraGroup
    rig = fmt.Rig(CameraGroup([cam]), offset={CAM: (0.0, 0.0)},
                  moving={CAM: False}, calibrated={CAM: False})
    edges = np.asarray(np.atleast_2d(args.skeleton_edges), int) - 1
    fmt.write_session(
        dst, mode='2d', units='px', label_source='annotated', names=names, rig=rig,
        groups=groups, labels=labels,
        skeleton=[[names[a], names[b]] for a, b in edges],
        flip_pairs=[[names[1], names[2]]],
        provenance={
            'source': str(args.lbl),
            'annotator': '',
            'annotator_tool': f'APT (Animal Part Tracker), project {args.projname} '
                              f'VERSION {args.version}',
            'converter': 'scripts/convert_apt_lbl.py',
            'apt_movie': job.movie,
            'apt_movie_index': job.apt_index,
            'apt_gt_set': job.is_gt,
            'keypoint_renames': ', '.join(f'{k} -> {v}' for k, v in RENAME.items()),
            'occluded_as': 'visible',
            'occluded_note': "APT occ==1 (occluded, position placed by the annotator) is written "
                             "`visible` WITH coordinates -- 22% of point-slots project-wide. This "
                             "root's visibility channel therefore asserts visible for occluded "
                             "points; do not use a vis head trained on it as a row gate without a "
                             "rate-matched random control. Inf -> missing, NaN -> no row.",
            'animal_id_source': 'APT per-frame target slot (tgt-1); one labelled frame per group, '
                                'so no cross-frame identity is asserted',
            'context_frames': context,
            'extract': extract or ('ffmpeg -c:v copy -bsf:v mjpeg2jpeg (lossless)'
                                   if mode == 'copy' else 'PyAV decode + JPEG q95'),
            'regions_source': "APT labelsRoi (Label Box), axis-aligned, kept only on each group's "
                              'own labelled frame -- a certificate is sound only where that '
                              "frame's labels live. An empty regions.pq means nothing in the "
                              'session is certified, which is NOT the same as no file at all.',
        })
    return stat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lbl', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--context', type=int, default=65,
                    help='frames per group, label centered (default 65, allen-mouse\'s shape)')
    ap.add_argument('--extract', choices=('copy', 'decode'), default='copy')
    ap.add_argument('--jobs', type=int, default=1, help='movies in parallel')
    ap.add_argument('--max-groups', type=int, default=0, help='0 = all')
    ap.add_argument('--only', default='', help='substring of the session id -- re-run one session')
    ap.add_argument('--labels-only', action='store_true',
                    help='rewrite the tables against the frames already on disk -- no ffmpeg. '
                         'The existing groups.pq is the authority for n_frames and '
                         'source_frame_start, so a truncated or dropped group stays as extracted.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--clean', action='store_true', help='remove each session dir first')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--no-image-check', action='store_true')
    args = ap.parse_args()
    if args.context < 2:
        raise SystemExit('--context must be >= 2 (T=1 gives a zero-length pos_embed)')

    with tempfile.TemporaryDirectory() as td:
        d = read_lbl(args.lbl, Path(td))
    check_axis(d)
    args.projname = str(np.ravel(d['projname'])[0]) if 'projname' in d else '?'
    args.version = str(np.ravel(d['VERSION'])[0]) if 'VERSION' in d else '?'
    args.skeleton_edges = d['skeletonEdges']

    jobs = build_jobs(d)
    if args.only:
        jobs = [j for j in jobs if args.only in j.session]
        if not jobs:
            raise SystemExit(f'--only {args.only!r} matched no session')
    if args.max_groups:
        budget = args.max_groups
        for j in jobs:
            j.frames = j.frames[:budget]
            budget -= len(j.frames)
        jobs = [j for j in jobs if j.frames]

    # Over the kept frames, not the whole row set: `--max-groups` must report the conversion done.
    def n_inst(j):
        return int(np.isin(j.frm, j.frames).sum())

    print(f'{args.projname} v{args.version}: {len(jobs)} labelled movies, '
          f'{sum(len(j.frames) for j in jobs)} groups, '
          f'{sum(n_inst(j) for j in jobs)} instances')
    for split in fmt.SPLITS:
        mine = [j for j in jobs if j.split == split]
        print(f'  {split:5s} {len(mine):3d} movies  {sum(len(j.frames) for j in mine):5d} groups  '
              f'{sum(n_inst(j) for j in mine):5d} instances  '
              f'{sum(j.roi_f.size for j in mine):4d} label boxes  '
              f'sizes {sorted({j.wh for j in mine})}')

    stats = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for stat in ex.map(lambda j: convert_movie(j, args.out, args.context, args.extract, args),
                           jobs):
            stats.append(stat)
            for w in stat['warnings']:
                print(f'  [warn] {stat["session"]}: {w}')
            print(f'  {stat["split"]:5s} {stat["session"]:42s} '
                  f'{stat["groups"]:4d} groups {stat["frames"]:6d} frames'
                  + (f' {stat["regions"]:3d} regions' if stat['regions'] else '')
                  + (' RE-ENCODED' if stat['decode_movie'] else '')
                  + (f' {stat["decode_fallback"]} fallback' if stat['decode_fallback'] else '')
                  + (f' {stat["skipped"]} skipped' if stat['skipped'] else ''))

    # Both fallback routes reported: a movie-wide `decode` switch is correct but re-encoded.
    reenc = [s for s in stats if s['decode_movie']]
    # ROIs on no group's labelled frame are dropped (`group_regions`) -- a real loss, reported.
    off = sum(s['roi_off_label'] for s in stats)
    print(f'\nregions: {sum(s["regions"] for s in stats)} written, '
          f'{off} label boxes dropped (on a frame no group is centered on)')
    print(f'total: {sum(s["groups"] for s in stats)} groups, '
          f'{sum(s["frames"] for s in stats)} frames, '
          f'{sum(s["decode_fallback"] for s in stats)} per-group decode fallbacks, '
          f'{len(reenc)} movies re-encoded ({sum(s["frames"] for s in reenc)} frames), '
          f'{sum(s["skipped"] for s in stats)} groups skipped')
    if args.dry_run:
        return 0

    if args.validate:
        errs = fmt.validate_dataset(fmt.load_dataset(args.out),
                                   check_images=not args.no_image_check)
        hard = [e for e in errs if 'WARNING' not in e]
        for e in errs:
            print(f'  {e}')
        print(f'validate: {len(hard)} hard, {len(errs) - len(hard)} warnings')
        return 1 if hard else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
