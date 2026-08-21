"""ADOPT A FOLDER OF VIDEOS PLUS AN ANIPOSE CALIBRATION AS A SESSION, IN MEMORY.

`--data` is a session directory in `docs/annotation_format.md`. A user with a folder of videos and
an anipose calibration had to write a converter before the model would look at their pixels, and
there is no converter for unlabelled footage because every shipped converter exists to carry
LABELS across.

**THE SESSION IS BUILT IN MEMORY. NOTHING IS WRITTEN.** `plan` turns filenames into a
group -> camera -> path map; `build` probes each video for its length and frame size and returns a
`format.VideoSession`. No staging directory, no symlink farm, no second copy of the user's
filenames on disk that can go stale.

This is affordable because the decode path has exactly one filesystem entry point:
`dataset.read_frames` opens with `kind, src, ext = group.source(cam)`, and `Group.source` is a
cache over `Group._src`. `format.video_group` pre-fills it, so `Group.pixels()` is never called,
`Group.dir` is never dereferenced, and `session.path` is never read.

**THE COST IS REAL AND IS NOT HIDDEN: `validate_session` cannot be the acceptance test any more**,
because it resolves pixels through `Group.pixels()` and reads tables off `path`. The check moves
into `tests/test_adopt.py`, which builds the same plan both ways and compares the in-memory
session against the on-disk one `format.write_session` produces -- once, in CI, instead of on
every run.

**GOTCHA 10 IS LIVE HERE AND THE ANSWER IS A SCOPE RULE, NOT A WORKAROUND.** `build` opens video
containers in the PARENT process, which is the fork deadlock if that process later forks
dataloader workers. Inference does not fork -- the window loop decodes in-process behind a thread
pool -- so this module is safe from `scripts/infer.py` AND NOWHERE ELSE. Never call it from
`scripts/train.py`.

Pure/impure is split down the middle on purpose: `plan` and `check_flags` fire every refusal that
does not need pixels, so they run before the checkpoint loads and before anything decodes, and
they are testable with no video fixture at all.
"""
from __future__ import annotations

import re
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import format as fmt

# How many videos to probe at once. It is I/O, so it parallelises -- but 16 concurrent decord
# opens is exactly the quadratic reader cost (~59 GB at 16 on a 3208x2200 rig, see
# `dataset._reader_cache_size`), so the pool is bounded and every reader is dropped immediately.
PROBE_WORKERS = 4


@dataclass(frozen=True)
class VideoPlan:
    """Everything derivable from the FILENAMES plus the calibration. No pixels were opened."""
    session_id: str
    mode: str                                   # '3d' | '2d'
    rig: fmt.Rig
    videos: dict[str, dict[str, Path]]          # group_id -> camera -> absolute video
    # The three CLI inputs, kept so `provenance_of` records exactly what a reconstruction needs.
    files: tuple[Path, ...] = ()                # the RESOLVED --videos list, after dir expansion
    calibration: Path | None = None
    cam_regex: str | None = None
    group_id: str = ''                          # only meaningful when every remainder was empty
    # Cameras whose calibration block carried no `size` and must be filled from the pixels (14).
    need_size: tuple[str, ...] = ()
    skipped: tuple[str, ...] = field(default=(), compare=False)   # parsed, uncalibrated, dropped


def _die(msg: str):
    raise SystemExit(msg)


# ---------------------------------------------------------------------------------------------
# the naming rule -- anipose's own (anipose/common.py), with two documented supersets


def parse_name(stem: str, cam_regex: str | None) -> tuple[str, str]:
    """(camera, group_id) for one video basename, under anipose's `[triangulation] cam_regex`.

        basename = Path(video).stem
        camera   = re.search(cam_regex, basename).group(1).strip()
        group_id = re.sub(cam_regex, '', basename).strip()

    So `cam0_trial3.mp4` under `cam([0-9]+)_` is camera `0`, group `trial3`, and `cam1_trial3.mp4`
    joins it as the same group's second camera. **THE CAMERA NAME IS THE CAPTURE GROUP**, so it is
    `0` and not `cam0` -- anipose's convention, and the number-one thing a user gets wrong.

    Two deviations, both supersets, both the same shape as `crop.py` being a documented superset of
    posetail's crop rule:

    - **No capture group means the whole match.** anipose does `match.groups()[0]` and raises
      `IndexError` on a group-less pattern; here `--cam-regex 'Cam[0-9]+'` works and names the
      camera `Cam2005325`. NOT A NICETY: johnson's `mouse_2_validate` ships `Cam2005325.mp4`
      beside a calibration whose camera is named `Cam2005325`, so the capture-group convention
      would name it `2005325` and refusal 2 would fire on all 16 cameras.
    - **A one-camera calibration needs no regex.** `cam_regex is None` -> the whole stem is the
      group id and the caller supplies the camera. Demanding a regex to select from a set of one
      is ceremony, and 2D single-view is most of the intended traffic.

    Returns ('', stem) when there is no regex; the caller names the camera.
    """
    if not cam_regex:
        return '', stem.strip()
    m = re.search(cam_regex, stem)
    if m is None:
        return '', ''
    cam = (m.group(1) if m.re.groups else m.group(0)).strip()
    return cam, re.sub(cam_regex, '', stem).strip()


def _expand(videos) -> list[Path]:
    """Files and/or directories -> the resolved, sorted file list. A directory expands to its
    `VIDEO_EXTS` children, sorted, NOT recursively.

    RESOLVED AND EXPANDED BEFORE IT IS RECORDED (`provenance_of`), so a `--videos rec/` that later
    grows a file does not reconstruct differently.
    """
    out: list[Path] = []
    for v in videos:
        p = Path(v)
        if p.is_dir():
            out.extend(sorted(q.resolve() for q in p.iterdir()
                              if q.is_file() and q.suffix.lower() in fmt.VIDEO_EXTS))
        else:
            # Refusal 5, and only for an EXPLICITLY named file: a directory expansion already
            # filtered, so an unopenable container here is something the user typed.
            if p.suffix.lower() not in fmt.VIDEO_EXTS:
                _die(f'--videos {p}: extension {p.suffix!r} is not one of {fmt.VIDEO_EXTS}. '
                     'Nothing downstream distinguishes a container it cannot open from one it '
                     'was never given.')
            # NAMED AND ABSENT IS A REFUSAL, not a silent drop. It is also what catches a video
            # RENAMED OR MOVED since a run: `provenance_of` records the resolved file list, so
            # `session_from_prediction` replays exactly these paths, and a missing one there means
            # the prediction can no longer be rendered over the pixels it was made from.
            if not p.exists():
                _die(f'--videos {p}: no such file.')
            out.append(p.resolve())
    # dict.fromkeys, not set: the ORDER is the sorted expansion above and must not become a hash
    # order, because `source_videos` is recorded and compared.
    return list(dict.fromkeys(out))


def _common_parent(files: list[Path]) -> Path:
    parents = {f.parent for f in files}
    if len(parents) == 1:
        return parents.pop()
    import os
    return Path(os.path.commonpath([str(f.parent) for f in files]))


def plan(videos, calibration, cam_regex=None, *, session_id=None, group_id=None) -> VideoPlan:
    """PURE GIVEN THE FILESYSTEM'S NAMES. Refusals 1-6. Opens no video, decodes nothing,
    loads no checkpoint.

    `calibration` is an **aniposelib-layout** toml -- what `format.load_calibration` reads, which
    is also what this repo writes and what anipose itself writes. It is NOT a converter for any
    other calibration format; see `--calibration`'s help.
    """
    files = _expand(videos)
    if not files:
        _die('--videos matched no file. Pass video files, or a directory holding them '
             f'(expanded non-recursively over {fmt.VIDEO_EXTS}).')

    cal = Path(calibration).resolve()
    if not cal.exists():
        _die(f'--calibration {cal}: no such file.')
    with open(cal, 'rb') as f:
        doc = tomllib.load(f)
    # REFUSAL 14, and it is the one place this path may be generous: a size derived from the
    # pixels cannot be wrong about the pixels. `load_calibration` raises on a block with no `size`,
    # so the document is patched here and `build` fills it in -- and PRINTS that it did.
    need_size = []
    for k, block in doc.items():
        if k == 'metadata' or not isinstance(block, dict):
            continue
        if 'size' not in block:
            need_size.append(str(block.get('name', k)))
            block['size'] = [0, 0]
    rig = fmt.rig_from_doc(doc, str(cal))

    # NO MOVING RIGS. `extrinsics.pq` is per-frame geometry no filename can supply. Refused here
    # rather than left to fail: `labels()` is overridden on this path, so a moving camera would
    # sail past the check that lives there and blow up inside `cgroup()` instead, much later and
    # much worse.
    moving = [n for n in rig.names if rig.moving[n]]
    if moving:
        _die(f'{cal}: cameras {moving} declare moving = true, and --videos cannot supply '
             'per-frame extrinsics -- there is no extrinsics.pq and no filename that could carry '
             'one. Convert the recording to a session directory if the rig really moves.')

    mode = '3d' if len(rig) > 1 else '2d'
    if mode == '3d':
        # REFUSAL 6. `load_calibration` silently turns a block with no `matrix` into a
        # `nominal_camera`, and with `validate_session` out of the loop nothing downstream would
        # ever say so -- which is what makes this load-bearing rather than a convenience.
        bad = [n for n in rig.names if not rig.calibrated[n]]
        if bad:
            _die(f'{cal}: {len(rig)} cameras means 3D, but {bad} carry no matrix/rotation/'
                 'translation, so load_calibration invented a nominal camera for each. A '
                 'triangulation against an invented camera is silently wrong. Calibrate them, or '
                 'run one camera at a time.')

    if cam_regex is None and len(rig) != 1:
        _die(f'--cam-regex is required: the calibration names {len(rig)} cameras '
             f'({rig.names[:4]}...) and nothing else says which video is which. It is anipose\'s '
             "own `[triangulation] cam_regex`, e.g. 'cam([0-9]+)_'. Only a ONE-camera "
             'calibration may omit it.')

    # THE REGEX, over the basenames.
    parsed: list[tuple[Path, str, str]] = []
    for f in files:
        cam, gid = parse_name(f.stem, cam_regex)
        if cam_regex is None:
            cam = rig.names[0]
        parsed.append((f, cam, gid))

    matched_regex = [p for p in parsed if p[1]]
    if not matched_regex:
        _die(f'--cam-regex {cam_regex!r} matched none of the {len(files)} video(s). '
             f'First basenames: {[f.stem for f in files[:3]]}. `re.search` is used, so the '
             'pattern need not anchor.')

    # REFUSAL 2, all-versus-some. `mouse_2_validate` ships 17 mp4s against 16 calibrated cameras:
    # a camera with video and no geometry can never join any group, so refusing the whole run is
    # wrong -- but silently dropping pixels is worse.
    known = set(rig.names)
    keep = [p for p in matched_regex if p[1] in known]
    if not keep:
        seen = sorted({p[1] for p in matched_regex})
        _die(f'--cam-regex {cam_regex!r} parsed camera name(s) {seen[:6]} and the calibration '
             f'names {rig.names[:6]}. THE CAMERA NAME IS THE CAPTURE GROUP, so '
             "'cam([0-9]+)_' yields '0' and not 'cam0'. Either rename the calibration's "
             'cameras, or drop the capture group -- a group-less pattern matches the whole '
             "string (e.g. 'cam[0-9]+' -> 'cam0').")
    skipped = tuple(sorted({p[1] for p in matched_regex if p[1] not in known}))
    for name in skipped:
        print(f'--videos: camera {name!r} has video but no calibration block; skipping its '
              f'{sum(1 for p in matched_regex if p[1] == name)} file(s).')

    # THE EMPTY GROUP ID IS ABOUT DISAGREEMENT, NOT ABOUT EMPTINESS. `Cam2005325.mp4` under
    # `Cam[0-9]+` leaves '': the whole stem IS the camera name, because a raw multi-camera
    # recording of ONE session has nothing else to put in the filename. That is the shape of every
    # raw rig dump. But `cam0.mp4` beside `cam0_trial3.mp4` is a genuine ambiguity.
    empty = [p for p in keep if not p[2]]
    if empty and len(empty) != len(keep):
        _die('--cam-regex leaves some filenames with a group id and some with nothing: '
             f'empty for {[p[0].name for p in empty[:3]]}, non-empty for '
             f'{[p[0].name for p in keep if p[2]][:3]}. One of those is mis-named or the regex is '
             'wrong; merging the bare ones into a group called \'\' would hide it.')

    one_group = bool(empty)
    sid = str(session_id) if session_id else _common_parent(files).name
    gid_default = str(group_id) if group_id else sid

    out: dict[str, dict[str, Path]] = {}
    where: dict[tuple[str, str], Path] = {}
    for f, cam, gid in keep:
        gid = gid_default if one_group else gid
        if (gid, cam) in where:                                   # REFUSAL 3
            _die(f'two videos land on group {gid!r} camera {cam!r}: {where[gid, cam]} and {f}. '
                 'Silently keeping one is how a whole trial goes missing.')
        where[gid, cam] = f
        out.setdefault(gid, {})[cam] = f

    for gid in sorted(out):                                       # REFUSAL 4
        miss = [n for n in rig.names if n not in out[gid]]
        if miss:
            _die(f'group {gid!r} has no video for calibrated camera(s) {miss}. The rig is the '
                 "calibration's, so a group with a hole would triangulate against a camera whose "
                 'pixels are a different recording. Write a reduced calibration if the rig really '
                 'is smaller -- there is no --allow-missing-cameras.')

    return VideoPlan(session_id=sid, mode=mode, rig=rig,
                     videos={g: dict(sorted(out[g].items())) for g in sorted(out)},
                     files=tuple(files), calibration=cal, cam_regex=cam_regex,
                     group_id=gid_default if one_group else '',
                     need_size=tuple(need_size), skipped=skipped)


# ---------------------------------------------------------------------------------------------
# the flag refusals that need no filesystem at all


def check_flags(args) -> None:
    """Refusals 7-11: the flags that mean something on a labelled session and nothing here.

    Every one is pure argparse arithmetic, so it fires before the checkpoint loads AND before
    anything decodes -- which `tests/test_adopt.py` pins by asserting the probe was never called.
    """
    if args.anchor == 'labels':                                                   # 7
        _die('--anchor labels seeds the model with GROUND TRUTH, and --videos has no labels: the '
             'label array is S = 0, so the "oracle" would be an oracle over nothing. Use '
             '--anchor carry (deployment) or --anchor none.')
    if args.box_prompt == 'labels':                                               # 8
        _die('--box-prompt labels seeds the box from GROUND TRUTH, and --videos has no labels. '
             'Use --box-prompt detector (with --detector/--boxes) or --box-prompt none.')
    if not args.detector and not args.boxes:                                      # 9
        _die('--videos needs a box source: the default crop comes from the LABELS, and here they '
             'are an S = 0 array, so every window would abort `no points` and the run would '
             'report coverage 0.000 with nothing saying why. Pass --detector <run folder> or '
             '--boxes <npz>.')
    # 10, AND IT IS THE DETECTOR PATH'S REFUSAL SPECIFICALLY. `run_dataset` falls back to
    # `max(1, len(sess.labels(gid).animal_ids))`, which is 1 for any footage -- the
    # catastrophic-miss-rate failure the driver already warns about, here with no labels to make
    # it visible. A `--boxes` npz states the row count in its own first axis, so there is nothing
    # to guess and nothing to refuse.
    if args.detector and not args.max_animals:                                    # 10
        _die('--videos --detector needs --max-animals: the row count otherwise falls back to the '
             "session's own animal count, which is 0 here and clamps to 1 -- a catastrophic miss "
             'rate on any multi-animal footage, with no labels to make it visible. The animal '
             'count is a fact no file on this path carries, so the operator states it.')
    # `--split` DEFAULTS TO None IN THE PARSER, not to 'test', precisely so this can tell whether
    # it was passed. The directory path resolves `args.split or 'test'`.
    if getattr(args, 'split', None) is not None:                                  # 11
        _die('--split selects among a dataset root\'s sessions and --videos has no root, so it is '
             'inert here. Silently ignoring it would let you believe you selected something.')


def dataset_name(registry, explicit=None) -> str:
    """WHICH REGISTRY ENTRY TO READ THIS FOOTAGE'S KEYPOINTS AS.

    A session with no labels still needs `names`, and there is no data to derive them from -- so
    they come from the run's own registry. Which ENTRY is the open question, and the answer is a
    default plus a refusal: `--dataset-name` wins; otherwise a registry holding exactly one
    dataset is used and PRINTED; otherwise refuse, listing them. A multi-root run has several
    keypoint vocabularies and nothing on this path can choose between them.
    """
    have = [d for d, _ in registry.datasets]
    if explicit:
        if explicit not in have:
            _die(f'--dataset-name {explicit!r} is not in this run\'s keypoint registry. It holds '
                 f'{have}.')
        return explicit
    if len(have) == 1:
        print(f'registry: one dataset, reading these videos\' keypoints as {have[0]!r}')
        return have[0]
    _die(f'this run\'s keypoint registry holds {len(have)} datasets ({have}) and --videos carries '
         'no directory name to pick one from. Pass --dataset-name.')


# ---------------------------------------------------------------------------------------------
# the probe


def _probe(path: Path) -> tuple[int, tuple[int, int], float]:
    """(n_frames, (w, h), fps) for one video. OPENS ONE READER AND DROPS IT.

    **THE FRAME COUNT COMES FROM DECORD, NOT FROM FFPROBE OR `cv2.CAP_PROP_FRAME_COUNT`.**
    `Group.n_frames` is a promise that every index in `[0, T)` decodes, and `dataset._read_video`
    fulfils it through `decord.VideoReader.get_batch`. A container-metadata count that disagrees
    with decord's own index -- routine on a variable-frame-rate or truncated file -- turns into a
    hard failure deep inside the window loop, after the checkpoint has loaded. Taking the count
    from the same reader the loader uses makes it consistent by construction.
    `scripts/convert_calms21.py` already does exactly this.

    **AND IT DOES NOT GO THROUGH `dataset._reader`.** That cache is process-wide, sized on FIRST
    USE from whatever rig asked first, and the cost of `n` open readers is quadratic in `n`. A
    probe that populated it would fix its size before the run's own RAM budget is resolved and
    hold 16 containers open for the sake of 32 integers. Open, read three facts, drop; reopening
    is 41.5 ms, and the open also WARMS THE PAGE CACHE the run then reads through.
    """
    from decord import VideoReader

    vr = VideoReader(str(path), num_threads=1)
    try:
        n = len(vr)
        shape = vr[0].shape
        fps = float(vr.get_avg_fps())
    finally:
        del vr
    return int(n), (int(shape[1]), int(shape[0])), fps


def build(plan: VideoPlan, *, names, units='mm', fps=None, assoc_res_max_px=30.0,
          trim=False, probe=_probe, verbose=True) -> fmt.VideoSession:
    """Probe every video, then construct the session IN MEMORY. Refusals 12-14.

    GOTCHA 10: this opens video containers in the calling process. Safe from `scripts/infer.py`,
    which never forks; never call it from `scripts/train.py`.
    """
    jobs = [(gid, cam, p) for gid, cams in plan.videos.items() for cam, p in cams.items()]
    got: dict[tuple[str, str], tuple] = {}
    # A silent pause before the first box looks like a hang -- 40 s per cold open x 16 cameras is
    # ~11 minutes on a node that has never seen the recording. So it is parallel AND it says so.
    if verbose:
        print(f'--videos: probing {len(jobs)} video(s) for length and frame size '
              f'({PROBE_WORKERS} at a time)...', flush=True)
    with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
        futs = [(gid, cam, pool.submit(probe, p)) for gid, cam, p in jobs]
        for gid, cam, fu in futs:
            n, wh, f = fu.result()
            got[gid, cam] = (n, wh, f)
            if verbose:
                print(f'  {gid}/{cam}: {n} frames, {wh[0]}x{wh[1]}, {f:g} fps', flush=True)

    # REFUSAL 14, printed. A size derived from the pixels cannot be wrong about the pixels.
    for name in plan.need_size:
        wh = next((v[1] for (g, c), v in got.items() if c == name), None)
        if wh is None:
            continue
        plan.rig.by_name(name).set_size(wh)
        if verbose:
            print(f'--calibration: camera {name!r} carried no `size`; filled from its own first '
                  f'frame as {wh[0]}x{wh[1]}.')

    # REFUSAL 13. `matrix` and `distortions` are in SENSOR pixels and `size` is the image on disk,
    # so a calibration made at twice the deployment resolution projects to the wrong place with no
    # symptom other than being wrong. This is the one `validate_session` rule 8 would have caught
    # for an image root and never checks for video, so it is a check the format never had.
    for (gid, cam), (n, wh, f) in sorted(got.items()):
        want = plan.rig.size(cam)
        if tuple(wh) != tuple(want):
            _die(f'{gid}/{cam}: the video is {wh[0]}x{wh[1]} but the calibration says '
                 f'{want[0]}x{want[1]}. `matrix` and `distortions` are in SENSOR pixels and '
                 '`size` is the image on disk, so a mismatched pair projects to the wrong place '
                 'with no symptom other than being wrong.')

    groups: dict[str, fmt.Group] = {}
    empty: dict[str, fmt.Labels] = {}
    K, C = len(names), len(plan.rig)
    for gid, cams in plan.videos.items():
        lens = {cam: got[gid, cam][0] for cam in cams}
        if len(set(lens.values())) > 1:                                        # REFUSAL 12
            if not trim:
                _die(f'group {gid!r}: its cameras disagree on length -- {lens}. A one-frame '
                     'offset is usually a dropped trigger and usually harmless; a 40,000-frame '
                     'offset is two different recordings sharing a group id, and both look '
                     'identical to a min(). Pass --trim-to-shortest to take the min anyway.')
            n = min(lens.values())
            print(f'--trim-to-shortest: group {gid!r} trimmed to {n} frames, dropping '
                  f'{ {c: v - n for c, v in lens.items() if v != n} }')
        else:
            n = next(iter(lens.values()))
        got_fps = [got[gid, cam][2] for cam in cams]
        f = float(fps) if fps else (got_fps[0] if got_fps else float('nan'))
        # `source_video` is the spec's own ONE string per group. Right and spec-native for a
        # single-camera rig; left empty for a multi-camera one rather than picking arbitrarily.
        sv = str(next(iter(cams.values()))) if len(cams) == 1 else ''
        groups[gid] = fmt.video_group(gid, n, cams, fps=f, source_video=sv)
        empty[gid] = fmt.empty_labels(0, n, K, C, mode3d=(plan.mode == '3d'))

    sess = fmt.VideoSession(
        # `path` IS A LABEL, NOT A LOCATION -- see VideoSession. The common parent is what makes
        # `session_id` right and error strings name something a human recognises.
        path=_common_parent(list(plan.files)) / plan.session_id,
        mode=plan.mode, units=str(units), label_source='tracked', names=list(names),
        rig=plan.rig, groups=groups, assoc_res_max_px=float(assoc_res_max_px),
        provenance={'source': 'tailcyclenet infer --videos'}, empty=empty)
    for g in groups.values():
        g.session = sess
    if verbose:
        print(f'--videos: session {sess.session_id!r}, mode {sess.mode}, units {sess.units!r} '
              f'(a DECLARATION about the calibration, not a measurement), {len(groups)} group(s), '
              f'{len(plan.rig)} camera(s)')
    return sess


# ---------------------------------------------------------------------------------------------
# provenance, and the reconstruction that checks itself


def provenance_of(plan: VideoPlan) -> dict:
    """The CLI INPUTS, and that is the whole record.

    `source_videos` is the resolved `--videos` list -- the files themselves, absolute, one entry
    each, after directory expansion. NOT the glob as typed, and **not the derived
    `group -> camera -> path` map**: `plan()` is pure over the filenames, so calibration + regex +
    file list re-derives that map exactly, and recording a structure a pure function already
    computes is a second copy that can disagree with the first.

    It also kills a naming problem outright. A per-(group, camera) key like `video_<gid>_<cam>`
    collides -- group `a_b` camera `c` against group `a` camera `b_c` -- and `SessionWriter` only
    catches a collision when the two values differ, i.e. not reliably. A flat list has no key to
    collide.

    `source_videos` IS THE FIRST NON-SCALAR PROVENANCE VALUE IN THIS REPO. A TOML array of strings
    round-trips through `toml.dumps`/`tomllib` unchanged and `SessionWriter`'s duplicate-key check
    compares with `!=`, which works on lists.
    """
    return {
        'source': 'tailcyclenet infer --videos',
        # DELIBERATELY EMPTY: there is no directory, and writing the path of one that does not
        # exist reads as a stale root rather than as an absent one.
        'source_session': '',
        'source_calibration': str(plan.calibration),
        'source_cam_regex': str(plan.cam_regex or ''),
        # A fourth CLI input, and it is load-bearing exactly where the regex leaves every
        # remainder empty -- without it the reconstruction below cannot name the group.
        'source_group_id': str(plan.group_id),
        'source_videos': [str(p) for p in plan.files],
    }


def session_from_prediction(pred_dir) -> fmt.VideoSession:
    """Rebuild the identical `VideoSession` from a prediction session's own provenance.

    `names`, `mode`, `units` and every group's `n_frames` come from the PREDICTION -- so no video
    is probed and gotcha 10 is not reachable from this path at all.

    **AND THE RECONSTRUCTION CHECKS ITSELF, which a stored group -> camera map could not.** The
    re-derived group ids and camera names must equal the prediction's own `groups.pq` and
    `calibration.toml`; a mismatch means a video was renamed, moved or added since the run, and it
    raises naming the difference instead of rendering a prediction over the wrong pixels.
    """
    pred_dir = Path(pred_dir)
    with open(pred_dir / 'session.toml', 'rb') as f:
        cfg = tomllib.load(f)
    prov = cfg.get('provenance', {})
    files = list(prov.get('source_videos') or [])
    if not files:
        raise fmt.FormatError(
            f'{pred_dir}: [provenance] has no `source_videos`, so this prediction was not made '
            'from a --videos run and there is nothing to reconstruct.')

    p = plan(files, prov['source_calibration'], prov.get('source_cam_regex') or None,
             session_id=prov.get('source_session_id') or None,
             group_id=prov.get('source_group_id') or None)

    import pyarrow.parquet as pq
    gt = pq.read_table(pred_dir / 'groups.pq').to_pydict()
    want_groups = [str(g) for g in gt['group_id']]
    if sorted(p.videos) != sorted(want_groups):
        raise fmt.FormatError(
            f'{pred_dir}: the videos named in [provenance] now derive groups '
            f'{sorted(p.videos)}, but this prediction was written over {sorted(want_groups)}. A '
            'video was renamed, moved or added since the run.')
    want_cams = fmt.load_calibration(pred_dir / 'calibration.toml').names
    if list(p.rig.names) != list(want_cams):
        raise fmt.FormatError(
            f'{pred_dir}: the calibration named in [provenance] now has cameras {p.rig.names}, '
            f'but this prediction was written over {want_cams}.')

    names = list(cfg['names'])
    K, C = len(names), len(p.rig)
    groups, empty = {}, {}
    for i, gid in enumerate(want_groups):
        n = int(gt['n_frames'][i])
        groups[gid] = fmt.video_group(gid, n, p.videos[gid],
                                      fps=float(gt.get('fps', [float('nan')] * len(gt))[i]
                                                or float('nan')))
        empty[gid] = fmt.empty_labels(0, n, K, C, mode3d=(cfg['mode'] == '3d'))
    sess = fmt.VideoSession(
        path=_common_parent([Path(f) for f in files]) / p.session_id,
        mode=cfg['mode'], units=cfg['units'], label_source='tracked', names=names,
        rig=p.rig, groups=groups,
        assoc_res_max_px=float(cfg.get('assoc_res_max_px', 30.0)),
        provenance=dict(prov), empty=empty)
    for g in groups.values():
        g.session = sess
    return sess


def dump(sess: fmt.VideoSession, out: Path) -> None:
    """**DEBUG ONLY.** Write the constructed session out through `format.write_session`, symlinking
    the pixels, so a user can point `validate_session` and their own eyes at what the flags
    produced. Not the mechanism, and on no default path.
    """
    out = Path(out)
    fmt.write_session(out, mode=sess.mode, units=sess.units, label_source=sess.label_source,
                      names=sess.names, rig=sess.rig, groups=sess.groups,
                      labels={g: sess.labels(g) for g in sess.groups},
                      provenance=dict(sess.provenance),
                      assoc_res_max_px=sess.assoc_res_max_px)
    for gid, g in sess.groups.items():
        d = out / 'groups' / gid
        d.mkdir(parents=True, exist_ok=True)
        for cam in sess.cam_names:
            _, src, _ = g.source(cam)
            fmt.link(d / f'{cam}{Path(src).suffix}', Path(src))
    print(f'--dump-session: wrote {out} (pixels are symlinks). This is a DEBUG artefact; the '
          'run itself uses the in-memory session.', file=sys.stderr)
