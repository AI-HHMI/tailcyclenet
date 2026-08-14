#!/usr/bin/env python
"""Convert an APT (Animal Part Tracker) `.lbl` project into a tailcycle dataset.

    pixi run python scripts/convert_apt_lbl.py \
        --lbl /groups/branson/bransonlab/manan/ratCity_round13_mjpegqv5_bounded.lbl \
        --out .../tailcycle-datasets/rat-city-annotated --validate

Built for `RatCityFullSizeRT` (round 13): 2,607 hand-labelled (frame, animal) instances over
1,067 frames in 48 movies, plus APT's own 240-instance / 20-frame ground-truth set. Written as a
2D single-view `labels = "annotated"` root beside the tracked `rat-city`.

FACTS ABOUT THE SOURCE THAT THE CODE DEPENDS ON, all measured rather than assumed:

- The `.lbl` is a **POSIX tar**; the MATLAB file is its `label_file.lbl` member.
- That member is **MATLAB v5, not v7.3**, so `scipy.io.loadmat` reads it and h5py cannot. Two of
  its variables are MCOS opaque objects, which is why `whosmat` dies and why `variable_names=` is
  not just an optimisation.
- `labels{i}.p` is `(2*npts, n)`, decoded as `reshape(npts, 2, n)` -- four x's then four y's per
  column (MATLAB `s.p(:,end+1) = xy(:)` with `xy` npts-by-2, column-major). `frm`, `tgt` and the
  coordinates are all **1-based**.
- The coordinate sentinels are APT's: **NaN = unlabeled, Inf = fully occluded** with no position
  (`Labels.getUnlabeledValue` / `getFullyOccValue`). Read with the correct `p` layout these agree
  perfectly with the `occ` tag -- all 171 Inf slots carry `occ == 1` and all 220 NaN slots carry
  `occ == 0`, zero off-diagonal -- which is independent corroboration of the reshape below.
- The movies are **MJPEG**, all-intra, 40 fps, which is what makes the lossless extraction below
  possible; 45 are 4696x2048 and the five `original/merged_video_all_keyframes*` are 4500x2050.
- Those `merged_*` movies are **real contiguous video** -- "all_keyframes" is the encoding, not a
  montage. Consecutive-frame mean |diff| reads 1.4-2.9 against 3.8-5.0 at +100 frames and
  5.0-7.4 at +1000. `_contiguity_ok` re-checks it per movie rather than trusting this note,
  because a scene cut inside a context window is corrupt data that nothing downstream reveals.

FOUR CONVERSION DECISIONS, each recorded in `[provenance]` on disk:

1. `occ == 1` (occluded, but the annotator placed the point) is written **`visible`, with its
   coordinates** -- 2,500 of 11,388 point-slots, 22.0%, which is 22.7% of the visible rows. That
   is a deliberate choice of label density over an honest visibility channel, and it means THIS
   ROOT'S PER-CAMERA VISIBILITY
   CHANNEL IS THE calms21 FAILURE MODE: a `vis_pred` head trained on it learns against a target
   that calls occluded points visible, so it must never be used as a row gate without a
   rate-matched random control. `Inf` still becomes `missing` and `NaN` still becomes no row at
   all, so the negatives that do exist are real.
2. APT's `tail` is renamed **`tail_base`** to match the tracked `rat-city` root, so the two share
   registry ids -- and therefore keypoint embedding rows -- instead of splitting the keypoint in
   two.
3. APT's `labelsRoi` ("Label Box") becomes **`regions.pq`**, and every session writes one even
   when it has none. A Label Box marks an area the annotator certified as COMPLETELY LABELLED,
   which is the only thing standing between this root and teaching ~9 rats per frame to a
   detector as background: a labelled frame here names a median of 2 rats where the tracked root
   finds a median of 11. An ROI is kept only on the group whose OWN labelled frame it sits on
   (`group_regions` says why), so the 140 of 636 that sit elsewhere are dropped and counted. The
   `test/` sessions get an EMPTY regions.pq -- APT's GT mode records no ROIs -- which certifies
   nothing, where a missing file would certify everything.
4. One group per labelled frame, `--context` frames wide, **label centered**, which is
   allen-mouse's annotated shape. One labelled frame per group is also what makes APT's `tgt`
   safe to use as `animal_id`: `tgt` is a per-frame slot, not a tracked identity, and a group
   holding a single labelled frame asserts no identity across frames.

THE PIXELS ARE COPIED, NOT RE-ENCODED. `ffmpeg -ss start/fps -c:v copy -bsf:v mjpeg2jpeg` lifts
each stored JPEG out of the AVI byte-for-byte (the bitstream filter supplies the headers AVI
omits). Verified frame-exact against decord: the extracted frame reads mean |diff| 0.016 against
decord's decode of the same index -- its own YUV->RGB rounding -- and 2.9+ against either
neighbour. `--extract decode` forces the re-encoding path, and a failed per-group alignment check
falls back to it automatically rather than shipping a misaligned window.

They are written as an image DIRECTORY rather than a symlinked video on purpose: format rules 7
and 8 (contiguous %06d, count == n_frames, dims == camera size) are then checked for free, and a
root of image dirs cannot trip gotcha 11's video-fork deadlock in the training loader.
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

from tailcyclenet import format as fmt

CAM = 'cam0'
PAD = 20                    # THE crop rule's pad, as `backfill_boxes_v3.py` stores it: what goes
                            # in `instances.pq` is the PADDED EXTENT (steps 1-2 of
                            # `crop_box_for_points`), not the squared crop box, so `min_crop_dim`
                            # stays a consumer parameter.
APT_NAMES = ['nose', 'left_ear', 'right_ear', 'tail']
RENAME = {'tail': 'tail_base'}
LBL_MEMBER = 'label_file.lbl'
ALIGN_TOL = 1.0             # mean |diff| between the extracted frame and decord's own decode of
                            # the same index. 0.016 measured for the right frame, 2.9+ for a
                            # neighbour, so this sits ~180x below the discriminating gap.


# --------------------------------------------------------------------------------------------
# reading the .lbl
# --------------------------------------------------------------------------------------------

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
    """Assert the keypoint axis against the name list -- format spec 13's first gotcha.

    Not a formality: allen-mouse's converter transposed 16 of 47 keypoints by trusting a column
    order, and the failure is invisible in a loss curve. `skelNames` and `cfg.LabelPointNames`
    are two independent statements of the same order in the project file, so both are checked.
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
    # (2*npts, n) -> (2, npts, n) -> (n, npts, 2). A column is [x1..xK, y1..yK] -- one BLOCK of
    # x's then one of y's, not interleaved pairs -- so the coordinate axis leads and npts is
    # second. Reading it interleaved is the failure this converter is most exposed to, because
    # both readings produce plausible-looking numbers: measured on this project, the interleaved
    # reading puts 28.25% of points outside the frame against 0.01% for this one.
    xy = np.asarray(entry.p, float).reshape(2, npts, frm.size) - 1.0
    return frm, tgt, np.transpose(xy, (2, 1, 0))


def movie_regions(entry) -> tuple[np.ndarray, np.ndarray]:
    """One `labelsRoi{i}` cell -> (frames, (n,4) [x0,y0,x1,y1]), both 0-based.

    APT's "Label Box": a rectangle the annotator marked as COMPLETELY LABELLED, which its own
    GUI help describes as "teaching the classifier what a negative label is". It is not an animal
    box and could not be one -- `LabelROI` has no target index field at all.

    `verts` is (4,2,n) -- four corners, (x,y), per ROI -- and `squeeze_me` flattens it to (4,2)
    when a movie has exactly one. Every ROI in this project is axis-aligned; that is ASSERTED
    rather than assumed, because a rotated one would silently become its bounding box, and a
    bounding box certifies area the annotator did not.
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


# --------------------------------------------------------------------------------------------
# what becomes a session, and which split it lands in
# --------------------------------------------------------------------------------------------

_C5_DATE = re.compile(r'/original/.*_20250819_')


def split_of(movie: str, is_gt: bool) -> str:
    """Cross-cohort: cohort5 is held out for val, APT's own GT set is test.

    The `original/*_20250819_*` movie goes to val WITH cohort5. It is dated inside cohort5's
    range (cohort5_20250812 .. cohort5_20250822), so it is almost certainly the same rats, and
    leaving it in train would make the cross-cohort claim false. The other four `original/`
    movies are undated, so the val axis is honestly "cross-cohort modulo four undated movies" --
    eval rule 1 is about not overclaiming exactly this.
    """
    if is_gt:
        return 'test'
    return 'val' if ('/cohort5/' in movie or _C5_DATE.search(movie)) else 'train'


def session_id(movie: str) -> str:
    """One session per movie, named for the recording.

    Per movie rather than per cohort because calibration is a session property and the frame size
    is not constant across this project (4696x2048 vs 4500x2050); bucketing by movie makes that
    a non-issue by construction instead of a rule-8 failure.
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


# --------------------------------------------------------------------------------------------
# pixels
# --------------------------------------------------------------------------------------------

_readers: dict[str, object] = {}
_readers_lock = threading.Lock()


def reader(movie: str):
    """One decord `VideoReader` per movie, reused. Opening one costs 2-5 s on a 70k-frame AVI."""
    with _readers_lock:
        if movie not in _readers:
            from decord import VideoReader
            _readers[movie] = VideoReader(movie, num_threads=1)
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
    """Fallback: decode with decord and re-encode. Index-exact by construction, and lossy."""
    import cv2
    vr = reader(movie)
    n = min(n, len(vr) - start)
    imgs = vr.get_batch(list(range(start, start + n))).asnumpy()
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
                         reader(movie).get_batch([start]).asnumpy()[0]) < ALIGN_TOL:
            return got, 'copy'
        for p in out.glob('*.jpg'):
            p.unlink()
    return decode_window(movie, start, n, out), 'decode'


def check_seek(movie: str, start: int, fps: float, tmp: Path) -> str:
    """Once per movie: is the time seek frame-EXACT, or merely close?

    The per-group check above only asks whether the extracted frame is near decord's decode of
    `start`, which a nearly-static scene could pass while off by one. This one demands that
    `start` be the ARGMIN over its neighbours, which no off-by-one survives. Once per movie is
    enough: `-ss` maps time to frame linearly, so if it lands for one group it lands for all.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob('*.jpg'):
        p.unlink()
    if not copy_window(movie, start, 1, fps, tmp):
        return 'ffmpeg produced no frame'
    got = _rgb(tmp / '000000.jpg')
    vr = reader(movie)
    cand = [f for f in (start - 1, start, start + 1) if 0 <= f < len(vr)]
    diffs = {f: _diff(got, vr.get_batch([f]).asnumpy()[0]) for f in cand}
    # TIED, not strictly best. These recordings contain stalls where consecutive frames are
    # byte-identical -- cohort1vs2_20240806_1030 has frames 11, 12 and 13 equal -- so an argmin
    # picks the first of them and a strict test reads that as a seek failure. It is not: when the
    # frames are identical it does not matter which one came back. Demanding only that `start` be
    # tied for the minimum still rejects every real off-by-one, where the gap is ~180x this
    # epsilon.
    if diffs[start] > min(diffs.values()) + 1e-6:
        return f'-ss lands on frame {min(diffs, key=diffs.get)}, not {start} (diffs {diffs})'
    return ''


def check_contiguity(movie: str, starts: list[int]) -> str:
    """Is this movie really contiguous video, or a montage of unrelated frames?

    A context window straddling a scene cut is corrupt training data that no downstream metric
    would expose, and `merged_video_all_keyframes*` is named exactly like a montage.

    THE TEST IS THAT THE DIFFERENCE GROWS WITH TEMPORAL DISTANCE, not that the consecutive
    difference is small. Sensor noise puts a floor under all three statistics, so the consec/+100
    RATIO never approaches zero even on plainly contiguous video -- measured 0.38-0.66 across
    these movies, which is why a 0.6 threshold flagged four of them spuriously. Monotone growth is
    insensitive to that floor because the floor adds about equally to every lag, and it is what a
    montage cannot produce: on shuffled frames all three lags read the same.
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


# --------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------

def padded_extent(pts: np.ndarray, wh) -> np.ndarray:
    """(K,2) source px -> [x0,y0,x1,y1], the padded extent of the finite points, or NaN.

    Steps 1-2 of `crop.crop_box_for_points` and not step 3, exactly as `backfill_boxes_v3.py`
    stores it for the tracked root: min/max over the two stored corners is min/max over the
    points they came from, so a consumer re-enters the real rule at `pad=0` and reproduces it.
    """
    finite = np.isfinite(pts).all(-1)
    if not finite.any():
        return np.full(4, np.nan, np.float32)
    p, wh = pts[finite], np.asarray(wh, np.float32)
    lo = np.clip(p.min(0) - PAD, 0, wh)
    hi = np.clip(p.max(0) + PAD, 0, wh)
    # int32 truncation, as the rule does it, so reading the float back and truncating is a no-op
    # rather than a second and different rounding.
    return np.concatenate([lo, hi]).astype(np.int32).astype(np.float32)


def window_start(frame: int, n_source: int, context: int) -> tuple[int, int]:
    """(start, n) for a group centered on `frame`, clamped into the movie.

    Centering is not cosmetic: frame 0 is the one frame where per-frame anchoring contributes
    nothing, so a label parked there measures the wrong thing (spec 14). Clamping rather than
    padding keeps every frame a real frame.
    """
    n = min(context, n_source)
    return int(np.clip(frame - n // 2, 0, max(0, n_source - n))), n


def group_labels(job: Job, frame: int, start: int, n: int, K: int) -> fmt.Labels:
    """The dense arrays for one group: one labelled frame at `frame - start`, the rest unlabeled.

    NaN becomes no row at all rather than a `missing` row -- `missing` claims a human looked and
    judged occlusion, and APT's NaN is the opposite claim (spec 13). Inf is APT's fully-occluded
    sentinel, which IS that claim, so it becomes `missing` with null coordinates.
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

    A Label Box certifies that one SOURCE frame is completely labelled, and that claim is only
    sound in the group carrying that frame's LABELS. Every group here is 65 frames wide around a
    single labelled frame, so a context frame's ROI would certify an area whose animals were
    converted into a different group -- i.e. it would assert the area empty. This is why the ROIs
    are keyed against `frame` rather than against the window.

    Returns ((M,6) in `Labels.regions` layout, n_clipped_away).
    """
    r = job.roi_rect[np.flatnonzero(job.roi_f == frame)]
    out = np.zeros((len(r), 6))
    out[:, 0] = frame - start                    # group-local frame index
    out[:, 1] = 0                                # the single camera
    out[:, 2:] = np.clip(r, 0.0, np.tile(np.asarray(wh, float), 2))
    keep = (out[:, 4] > out[:, 2]) & (out[:, 5] > out[:, 3])
    return out[keep], int((~keep).sum())


# --------------------------------------------------------------------------------------------
# one movie -> one session
# --------------------------------------------------------------------------------------------

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
    # `--labels-only`: the 70,655 extracted frames are lossless copies and already correct, so a
    # change to the TABLES must not re-run 80 minutes of ffmpeg over them. The groups on disk are
    # the authority for `n_frames` and `source_frame_start` -- re-deriving them would silently
    # disagree with the pixels wherever a group was truncated or dropped at extraction time.
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
        # `nframes` from the project file is a claim about the container; the pixels are the fact.
        # Truncate to what came out rather than padding -- spec 13 -- and drop the group if the
        # labelled frame itself is past the end or gotcha 1's two-frame floor is not met.
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

    # EVERY group gets a `regions` array, empty or not, so every session here writes a regions.pq.
    # Its absence is the format's claim of exhaustive labelling (§9b), and this project is the
    # opposite: a labelled frame names a median of 2 rats where the tracker finds 11. The GT
    # sessions in `test/` carry no ROIs at all -- APT builds them with `LabelROI.new()` -- so they
    # get an EMPTY regions.pq, which certifies nothing, rather than a missing one, which would
    # certify everything.
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
                                   if mode == 'copy' else 'decord decode + JPEG q95'),
            'regions_source': "APT labelsRoi (Label Box), axis-aligned, kept only on each group's "
                              'own labelled frame -- a certificate is sound only where that '
                              "frame's labels live. An empty regions.pq means nothing in the "
                              'session is certified, which is NOT the same as no file at all.',
        })
    return stat


# --------------------------------------------------------------------------------------------

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
        raise SystemExit('--context must be >= 2 (gotcha 1: T=1 gives a zero-length pos_embed)')

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

    # Over the KEPT frames, not the movie's whole row set -- otherwise `--max-groups` reports the
    # instance count of a conversion it did not do.
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

    # Both fallback routes are reported. A whole movie switched to `decode` is still correct data
    # -- decord indexing is exact by construction -- but it is re-encoded rather than copied, and
    # counting only the per-group route would leave that invisible.
    reenc = [s for s in stats if s['decode_movie']]
    # A Label Box on a frame that is no group's LABELLED frame is dropped -- see `group_regions`.
    # Reported because it is a real loss of certified negative area, not a rounding detail.
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
