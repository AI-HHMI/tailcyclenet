"""ADOPT A FOLDER OF VIDEOS PLUS AN ANIPOSE CALIBRATION AS A SESSION, IN MEMORY.

`plan` turns filenames into a group -> camera -> path map; `build` probes each video and returns
a `format.VideoSession`, with nothing staged (`Group.source` is the one filesystem entry point
and is pre-filled). The cost: `validate_session` cannot be the acceptance test, so
`tests/test_adopt.py` compares the in-memory session against the on-disk one in CI.

`build` opens video containers in the PARENT process (the fork deadlock), so this module is safe
from `scripts/infer.py` -- which never forks -- and NOWHERE ELSE.
"""
from __future__ import annotations

import re
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import format as fmt

# How many videos to probe at once. It is I/O, so it parallelises; bounded by the READER share
# of the RAM budget (an un-budgeted consumer is how the next one gets added), and cheap -- an
# open reader retains little when trimmed after each step.
PROBE_WORKERS_MAX = 4
# Rounded UP so the estimate errs toward a smaller pool -- the safe direction, since a miss costs
# a reopen and an over-estimate costs memory.
_PROBE_READER_BYTES = 0.05 * (1 << 30)


def probe_workers(n_videos: int, budget=None) -> int:
    """How many containers the probe may hold open at once. Never above `PROBE_WORKERS_MAX`, never
    below 1, and bounded by the reader share of the RAM budget."""
    from . import memory

    b = memory.current() if budget is None else budget
    return memory.fits(b.share(memory.FRACTION_READERS), _PROBE_READER_BYTES,
                       want=max(1, min(PROBE_WORKERS_MAX, int(n_videos))))


@dataclass(frozen=True)
class VideoPlan:
    """Everything derivable from the FILENAMES plus the calibration. No pixels were opened."""
    session_id: str
    # '3d' | '2d'
    mode: str
    rig: fmt.Rig
    # group_id -> camera -> absolute video
    videos: dict[str, dict[str, Path]]
    # The three CLI inputs, kept so `provenance_of` records exactly what a reconstruction needs.
    # The RESOLVED --videos list, after dir expansion.
    files: tuple[Path, ...] = ()
    calibration: Path | None = None
    cam_regex: str | None = None
    # Only meaningful when every remainder was empty.
    group_id: str = ''
    # Cameras whose calibration block carried no `size` and must be filled from the pixels (14).
    need_size: tuple[str, ...] = ()
    # Parsed, uncalibrated, dropped.
    skipped: tuple[str, ...] = field(default=(), compare=False)


def _die(msg: str):
    """Exit the process with a refusal message. The one way this module says no."""
    raise SystemExit(msg)


# ---------------------------------------------------------------------------------------------
# the naming rule -- anipose's own (anipose/common.py), with two documented supersets


def parse_name(stem: str, cam_regex: str | None) -> tuple[str, str]:
    """(camera, group_id) for one video basename, under anipose's `[triangulation] cam_regex`:
    camera = search(rx, stem).group(1), group_id = sub(rx, '', stem). THE CAMERA NAME IS THE
    CAPTURE GROUP (`cam([0-9]+)_` yields `0`, not `cam0`). Supersets: no capture group matches
    the whole stem, and a one-camera calibration needs no regex (returns ('', stem)).
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
    `VIDEO_EXTS` children, sorted, NOT recursively. Resolved BEFORE it is recorded, so a
    `--videos rec/` that later grows a file does not reconstruct differently.

    An explicitly named file with a foreign extension is a refusal (Refusal 5) -- a directory
    expansion already filtered. Named and absent is also a refusal: the recorded file list is
    replayed on render, so a missing file means the prediction lost its pixels. The result is
    deduplicated with `dict.fromkeys`, not a set: the order is the sorted expansion and must
    survive recording.
    """
    out: list[Path] = []
    for v in videos:
        p = Path(v)
        if p.is_dir():
            out.extend(sorted(q.resolve() for q in p.iterdir()
                              if q.is_file() and q.suffix.lower() in fmt.VIDEO_EXTS))
        else:
            if p.suffix.lower() not in fmt.VIDEO_EXTS:
                _die(f'--videos {p}: extension {p.suffix!r} is not one of {fmt.VIDEO_EXTS}. '
                     'Nothing downstream distinguishes a container it cannot open from one it '
                     'was never given.')
            if not p.exists():
                _die(f'--videos {p}: no such file.')
            out.append(p.resolve())
    return list(dict.fromkeys(out))


def _common_parent(files: list[Path]) -> Path:
    """The one directory every video's parent shares -- the session root for naming.

    Inputs: files -- resolved, sorted video paths (at least one).
    Outputs: Path -- the common parent directory.
    """
    parents = {f.parent for f in files}
    if len(parents) == 1:
        return parents.pop()
    import os
    return Path(os.path.commonpath([str(f.parent) for f in files]))


def plan(videos, calibration, cam_regex=None, *, session_id=None, group_id=None) -> VideoPlan:
    """PURE GIVEN THE FILESYSTEM'S NAMES. Refusals 1-6. Opens no video, decodes nothing, loads
    no checkpoint. `calibration` is an aniposelib-layout toml -- what `format.load_calibration`
    reads and what anipose itself writes.

    A camera block with no `size` is patched here (Refusal 14, and the one place this path may
    be generous) and `build` fills it in -- and PRINTS that it did: a size derived from the
    pixels cannot be wrong about the pixels, and `load_calibration` raises on a block with no
    `size`, so the document must be patched before the Rig is built. NO MOVING RIGS:
    `extrinsics.pq` is per-frame geometry no filename can supply, and `labels()` is overridden
    on this path so nothing downstream would catch it. Refusal 6: `load_calibration` silently
    turns a matrix-less block into a nominal camera, and with `validate_session` out of the
    loop nothing downstream would say so. The camera-name filter is REFUSAL 2, all-versus-some:
    `mouse_2_validate` ships 17 mp4s against 16 calibrated cameras -- a camera with video and
    no geometry can never join any group, so refusing the whole run is wrong, but silently
    dropping pixels is worse, so unknown cameras are skipped and printed. An empty group id is
    about DISAGREEMENT, not emptiness: all empty is one group (a raw rig dump); some empty and
    some not is a genuine ambiguity (Refusals 3 and 4 cover a duplicate (group, camera) pair
    and a group missing a calibrated camera's video).
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
    need_size = []
    for k, block in doc.items():
        if k == 'metadata' or not isinstance(block, dict):
            continue
        if 'size' not in block:
            need_size.append(str(block.get('name', k)))
            block['size'] = [0, 0]
    rig = fmt.rig_from_doc(doc, str(cal))

    moving = [n for n in rig.names if rig.moving[n]]
    if moving:
        _die(f'{cal}: cameras {moving} declare moving = true, and --videos cannot supply '
             'per-frame extrinsics -- there is no extrinsics.pq and no filename that could carry '
             'one. Convert the recording to a session directory if the rig really moves.')

    mode = '3d' if len(rig) > 1 else '2d'
    if mode == '3d':
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
        if (gid, cam) in where:
            _die(f'two videos land on group {gid!r} camera {cam!r}: {where[gid, cam]} and {f}. '
                 'Silently keeping one is how a whole trial goes missing.')
        where[gid, cam] = f
        out.setdefault(gid, {})[cam] = f

    for gid in sorted(out):
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
    """Refusals 7-11: the flags that mean something on a labelled session and nothing here. Pure
    argparse arithmetic, so it fires before the checkpoint loads and before anything decodes.

    The numbered checks fire in order and are refusals 7-11. Refusal 10 concerns the DETECTOR
    path specifically: the fallback animal count is 1 for any footage, a catastrophic miss rate
    with no labels to make it visible; a `--boxes` npz states its row count in its own first
    axis, so there is nothing to refuse. `--split` defaults to None so this can tell whether it
    was PASSED; the directory path resolves `args.split or 'test'`.
    """
    if args.anchor == 'labels':
        _die('--anchor labels seeds the model with GROUND TRUTH, and --videos has no labels: the '
             'label array is S = 0, so the "oracle" would be an oracle over nothing. Use '
             '--anchor carry (deployment) or --anchor none.')
    if args.box_prompt == 'labels':
        _die('--box-prompt labels seeds the box from GROUND TRUTH, and --videos has no labels. '
             'Use --box-prompt detector (with --detector/--boxes) or --box-prompt none.')
    if not args.detector and not args.boxes:
        _die('--videos needs a box source: the default crop comes from the LABELS, and here they '
             'are an S = 0 array, so every window would abort `no points` and the run would '
             'report coverage 0.000 with nothing saying why. Pass --detector <run folder> or '
             '--boxes <npz>.')
    if args.detector and not args.max_animals:
        _die('--videos --detector needs --max-animals: the row count otherwise falls back to the '
             "session's own animal count, which is 0 here and clamps to 1 -- a catastrophic miss "
             'rate on any multi-animal footage, with no labels to make it visible. The animal '
             'count is a fact no file on this path carries, so the operator states it.')
    if getattr(args, 'split', None) is not None:
        _die('--split selects among a dataset root\'s sessions and --videos has no root, so it is '
             'inert here. Silently ignoring it would let you believe you selected something.')


def dataset_name(registry, explicit=None) -> str:
    """WHICH REGISTRY ENTRY TO READ THIS FOOTAGE'S KEYPOINTS AS. A session with no labels still
    needs `names`, and there is no data to derive them from -- so `--dataset-name` wins, a
    registry holding exactly one dataset is used and PRINTED, otherwise refuse.
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

    The count comes from the SAME reader the loader uses (a metadata count that disagrees with
    the decoder fails inside the window loop), and NOT from `dataset._reader`, whose size would
    be fixed before the run's own budget is resolved. The open also warms the page cache.
    """
    from . import video

    vr = video.open_reader(str(path))
    try:
        n = len(vr)
        h, w = vr.frame_shape()[:2]
        fps = float(vr.fps)
    finally:
        vr.close()
        del vr
    return int(n), (int(w), int(h)), fps


def build(plan: VideoPlan, *, names, units='mm', fps=None, assoc_res_max_px=30.0,
          trim=False, probe=_probe, verbose=True) -> fmt.VideoSession:
    """Probe every video, then construct the session IN MEMORY. Refusals 12-14.

    Opens video containers in the calling process (the fork deadlock): safe from
    `scripts/infer.py`, which never forks; never call it from `scripts/train.py`.

    The probe runs in PARALLEL and prints progress: a silent pause before the first box looks
    like a hang. A camera in `plan.need_size` is filled from its own first frame and the fill
    is printed (Refusal 14 -- a size derived from the pixels cannot be wrong about the pixels).
    Refusal 13 rejects a calibration at the wrong resolution: `matrix` and `distortions` are in
    SENSOR pixels and `size` is the image on disk, so a mismatched pair projects to the wrong
    place with no symptom other than being wrong. Refusal 12 rejects a group whose cameras
    disagree on length unless `--trim-to-shortest` opts in. `source_video` is the spec's one
    string per group; left empty for multi-camera rather than picking arbitrarily. `path` is a
    LABEL, not a location -- the common parent is what makes `session_id` right and error
    strings name something a human recognises. The probe is where the memory goes, so the
    process is trimmed and peak-checked afterwards.
    """
    jobs = [(gid, cam, p) for gid, cams in plan.videos.items() for cam, p in cams.items()]
    got: dict[tuple[str, str], tuple] = {}
    workers = probe_workers(len(jobs))
    if verbose:
        print(f'--videos: probing {len(jobs)} video(s) for length and frame size '
              f'({workers} at a time)...', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [(gid, cam, pool.submit(probe, p)) for gid, cam, p in jobs]
        for gid, cam, fu in futs:
            n, wh, f = fu.result()
            got[gid, cam] = (n, wh, f)
            if verbose:
                print(f'  {gid}/{cam}: {n} frames, {wh[0]}x{wh[1]}, {f:g} fps', flush=True)

    for name in plan.need_size:
        wh = next((v[1] for (g, c), v in got.items() if c == name), None)
        if wh is None:
            continue
        plan.rig.by_name(name).set_size(wh)
        if verbose:
            print(f'--calibration: camera {name!r} carried no `size`; filled from its own first '
                  f'frame as {wh[0]}x{wh[1]}.')

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
        if len(set(lens.values())) > 1:
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
        sv = str(next(iter(cams.values()))) if len(cams) == 1 else ''
        groups[gid] = fmt.video_group(gid, n, cams, fps=f, source_video=sv)
        empty[gid] = fmt.empty_labels(0, n, K, C, mode3d=(plan.mode == '3d'))

    sess = fmt.VideoSession(
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
    from . import memory
    memory.trim()
    memory.check_peak('the --videos probe')
    return sess


# ---------------------------------------------------------------------------------------------
# provenance, and the reconstruction that checks itself


def provenance_of(plan: VideoPlan) -> dict:
    """The CLI INPUTS, and that is the whole record. `source_videos` is the RESOLVED file list --
    not the derived map, which `plan()` re-derives exactly from the list (and a per-(group,
    camera) key would collide). `source_session` is deliberately empty: there is no directory,
    and a stale-looking path reads as a root. `source_group_id` is load-bearing exactly where
    the regex leaves every remainder empty -- without it the reconstruction cannot name the
    group.
    """
    return {
        'source': 'tailcyclenet infer --videos',
        'source_session': '',
        'source_calibration': str(plan.calibration),
        'source_cam_regex': str(plan.cam_regex or ''),
        'source_group_id': str(plan.group_id),
        'source_videos': [str(p) for p in plan.files],
    }


def session_from_prediction(pred_dir) -> fmt.VideoSession:
    """Rebuild the identical `VideoSession` from a prediction session's own provenance.

    `names`, `mode`, `units` and `n_frames` come from the PREDICTION, so no video is probed. The
    reconstruction CHECKS ITSELF against the prediction's own `groups.pq` and `calibration.toml`
    -- a mismatch means a video was renamed, moved or added since the run.
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
    """**DEBUG ONLY.** Write the constructed session through `format.write_session`, symlinking
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
