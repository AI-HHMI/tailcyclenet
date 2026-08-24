#!/usr/bin/env python
"""Convert QDMouse4M into the tailcycle-dataset format.

The source is a collection of six-camera, single-mouse sessions.  This converter selects
non-overlapping 100-frame windows by 3D movement, writes dense pose tables, and makes exact
100-frame refonly MP4 clips for each selected group.

Examples:
    pixi run python scripts/convert_qdmouse4m.py --dry-run
    pixi run python scripts/convert_qdmouse4m.py --validate
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tailcyclenet import format as fmt

SOURCE = Path('/groups/karashchuk/karashchuklab/animal-datasets/qdmouse4m')
OUTPUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/qdmouse4m')
WINDOW = 100
TRAIN_GROUPS = 1000
REPROJECTION_THRESHOLD_PX = 10.0
VAL_SESSION = 'session_20251030132542-867148'
TEST_SESSION = 'session_20251108104601-195491'

KEYPOINTS = [
    'ear_L', 'ear_R', 'back_top', 'back_middle', 'back_bottom', 'tail_base',
    'tail_middle', 'tail_tip', 'forepaw_L', 'forepaw_R', 'hindpaw_L', 'hindpaw_R',
]
SKELETON = [
    ('ear_L', 'back_top'), ('ear_R', 'back_top'), ('back_top', 'back_middle'),
    ('back_middle', 'back_bottom'), ('back_bottom', 'tail_base'),
    ('tail_base', 'tail_middle'), ('tail_middle', 'tail_tip'),
    ('back_middle', 'forepaw_L'), ('back_middle', 'forepaw_R'),
    ('back_bottom', 'hindpaw_L'), ('back_bottom', 'hindpaw_R'),
]
FLIP_PAIRS = [('ear_L', 'ear_R'), ('forepaw_L', 'forepaw_R'), ('hindpaw_L', 'hindpaw_R')]


def session_dirs() -> list[Path]:
    """All source session directories, sorted."""
    return sorted(p for p in (SOURCE / 'sessions').iterdir() if p.is_dir())


def load_npz(session: Path) -> dict[str, np.ndarray]:
    """Load and sanity-check a session's keypoint annotations.

    Inputs: session -- a source session directory.
    Outputs: dict of npz arrays, verified: frame_index contiguous from zero,
             keypoint axis == KEYPOINTS, and six cameras.
    Side effects: raises RuntimeError on any violated invariant.
    """
    path = session / 'labels' / 'keypoint_annotations.npz'
    if not path.exists():
        raise RuntimeError(f'{session}: missing {path.name}')
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k].copy() for k in data.files}
    expected = np.arange(len(out['frame_index']))
    if not np.array_equal(out['frame_index'], expected):
        raise RuntimeError(f'{session}: frame_index is not contiguous from zero')
    if list(out['keypoint_names']) != KEYPOINTS:
        raise RuntimeError(f'{session}: unexpected keypoint axis {list(out["keypoint_names"])}')
    if out['keypoints_2d_px'].shape[1] != 6:
        raise RuntimeError(f'{session}: expected six cameras')
    return out


def movement_scores(data: dict[str, np.ndarray]) -> np.ndarray:
    """Mean 3D displacement per transition, ignoring invalid keypoint pairs."""
    xyz = np.asarray(data['keypoints_3d'], dtype=np.float64)
    valid = np.asarray(data['valid_3d'], dtype=bool) & np.isfinite(xyz).all(axis=-1)
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    d[~pair_valid] = np.nan
    finite = np.isfinite(d)
    count = finite.sum(axis=1)
    score = np.divide(np.nansum(d, axis=1), count,
                      out=np.full(d.shape[0], np.nan), where=count > 0)
    return score


def candidate_windows(data: dict[str, np.ndarray]) -> list[tuple[float, int]]:
    """Return (mean speed, start) for valid, non-jump 100-frame windows.

    A source-marked jump is not genuine animal movement, so any window touching one is excluded.
    """
    n = len(data['frame_index'])
    if n < WINDOW:
        return []
    speed = movement_scores(data)
    jumps = np.asarray(data.get('invalid_jump_timepoint', np.zeros(n, bool)), dtype=bool)
    prefix_jumps = np.concatenate(([0], np.cumsum(jumps, dtype=np.int64)))
    candidates: list[tuple[float, int]] = []
    for start in range(n - WINDOW + 1):
        vals = speed[start:start + WINDOW - 1]
        jump_count = prefix_jumps[start + WINDOW] - prefix_jumps[start]
        if jump_count or not np.isfinite(vals).any():
            continue
        score = float(np.nanmean(vals))
        if np.isfinite(score):
            candidates.append((score, start))
    return sorted(candidates, key=lambda x: (-x[0], x[1]))


def choose_windows(data: dict[str, np.ndarray], count: int) -> list[tuple[float, int]]:
    """Choose the highest-scoring non-overlapping windows in a session."""
    chosen: list[tuple[float, int]] = []
    for score, start in candidate_windows(data):
        if all(start + WINDOW <= old_start or old_start + WINDOW <= start
               for _, old_start in chosen):
            chosen.append((score, start))
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise RuntimeError(f'could only select {len(chosen)} of {count} windows')
    return sorted(chosen, key=lambda x: x[1])


def allocate_train_counts(sessions: list[Path], total: int) -> dict[str, int]:
    """Allocate the exact train count proportionally to available session frames."""
    lengths = {s.name: len(load_npz(s)['frame_index']) for s in sessions}
    raw = {name: total * n / sum(lengths.values()) for name, n in lengths.items()}
    counts = {name: int(v) for name, v in raw.items()}
    for name in sorted(raw, key=lambda x: (-(raw[x] - counts[x]), x))[:total - sum(counts.values())]:
        counts[name] += 1
    return counts


def source_cameras(session: Path) -> list[str]:
    """The session's camera names, from the npz view_names.

    Inputs: session -- a source session directory.
    Outputs: list of camera name strings.
    """
    with np.load(session / 'labels' / 'keypoint_annotations.npz', allow_pickle=True) as data:
        return [str(v) for v in data['view_names']]


def source_fps(video: Path) -> Fraction:
    """The source video's frame rate as an exact Fraction.

    Inputs: video -- a refonly MP4 path.
    Outputs: Fraction frame rate (44/1 fallback when the container omits it).
    """
    with av.open(str(video)) as container:
        rate = container.streams.video[0].average_rate
    if rate is None:
        return Fraction(44, 1)
    return Fraction(rate)


class ClipWriter:
    """A libx264 MP4 being built, one encoded frame at a time."""

    def __init__(self, path: Path, width: int, height: int, rate: Fraction):
        """Open an MP4 for writing with a fixed size and frame rate.

        Inputs: path -- output path.
                width, height -- frame size in pixels.
                rate -- output frame rate.
        Side effects: opens the output container and adds an H.264 stream.
        """
        self.path = path
        self.container = av.open(str(path), mode='w')
        self.stream = self.container.add_stream('libx264', rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'preset': 'veryfast', 'crf': '18'}
        self.rate = rate
        self.count = 0

    def add(self, frame: av.VideoFrame) -> None:
        """Encode one frame onto the clip with a fresh local PTS.

        Inputs: frame -- the source frame (reformatted to yuv420p).
        Side effects: encodes and muxes the frame; increments the PTS counter.

        Local PTS is set explicitly so the new clip does not carry the source clip's timestamps.
        """
        frame = frame.reformat(format='yuv420p')
        frame.pts = self.count
        frame.time_base = Fraction(self.rate.denominator, self.rate.numerator)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self) -> None:
        """Flush the encoder and close the container.

        Side effects: closes the file; raises RuntimeError if fewer than WINDOW
                      frames were encoded.
        """
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        if self.count != WINDOW:
            raise RuntimeError(f'{self.path}: encoded {self.count} frames, expected {WINDOW}')


def write_video_clips(source_session: Path, output_session: Path,
                      groups: list[tuple[str, int]], cameras: list[str], rate: Fraction) -> None:
    """Decode each refonly source once per camera and encode selected clips exactly."""
    starts = sorted((start, gid) for gid, start in groups)
    for camera in cameras:
        source = source_session / 'videos' / 'refonly' / f'{camera}.mp4'
        if not source.exists():
            raise RuntimeError(f'{source_session}: missing refonly video {source}')
        out_by_start = {start: output_session / 'groups' / gid / f'{camera}.mp4'
                        for start, gid in starts}
        max_end = starts[-1][0] + WINDOW
        active: ClipWriter | None = None
        next_start = 0
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for frame_index, frame in enumerate(container.decode(stream)):
                if next_start < len(starts) and frame_index == starts[next_start][0]:
                    path = out_by_start[starts[next_start][0]]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    active = ClipWriter(path, frame.width, frame.height, rate)
                    next_start += 1
                if active is not None:
                    active.add(frame)
                    if active.count == WINDOW:
                        active.close()
                        active = None
                if frame_index + 1 >= max_end:
                    break
        if active is not None:
            active.close()
        if next_start != len(starts):
            raise RuntimeError(f'{source}: reached frame {frame_index}, missing clips '
                               f'{starts[next_start:]}')


def make_labels(data: dict[str, np.ndarray], start: int, rig: fmt.Rig) -> fmt.Labels:
    """Build labels from consistent 3D, projecting accepted points back into every camera.

    The source 2D coordinates are used only as an observation for the reprojection gate. A
    finite but sub-threshold 2D coordinate is not an occlusion: it is omitted as UNLABELED.
    Likewise, source-side 3D interpolation and any point whose 3D-to-2D residual exceeds the
    threshold are omitted. Thus this dataset contains only VISIBLE 3D and PROJECTED 2D rows.

    Source 2D is (T, camera, keypoint, xy) and is transposed to the internal (T, keypoint,
    camera, xy). 3D is projected into stored-image pixels via a harmless finite placeholder on
    invalid rows, masked before the result is consumed. A 3D point is shared by all cameras: if
    any trustworthy observed camera disagrees, the shared 3D point and all of its projected
    camera rows are omitted rather than mixing targets.
    """
    import torch

    k = len(KEYPOINTS)
    t_global = np.arange(start, start + WINDOW)
    source_xy = np.asarray(data['keypoints_2d_px'][start:start + WINDOW], dtype=np.float32)
    source_xy = np.transpose(source_xy, (0, 2, 1, 3))
    source_present = np.isfinite(source_xy).all(axis=-1)
    source_valid2d = np.asarray(data['valid_2d_conf0p5'][start:start + WINDOW], dtype=bool)
    source_valid2d = np.transpose(source_valid2d, (0, 2, 1)) & source_present

    xyz = np.asarray(data['keypoints_3d'][start:start + WINDOW], dtype=np.float32)
    valid3d = np.asarray(data['valid_3d'][start:start + WINDOW], dtype=bool)
    valid3d &= np.isfinite(xyz).all(axis=-1)
    interp_frames = np.asarray(data.get('naninterp_v1_applied_frames', []), dtype=np.int64)
    interpolated = np.isin(t_global, interp_frames)
    valid3d &= ~interpolated[:, None]

    projected = np.full((WINDOW, k, 6, 2), np.nan, dtype=np.float32)
    for ci, camera in enumerate(rig.names):
        cam = rig.by_name(camera)
        safe_xyz = np.nan_to_num(xyz, nan=0.0).reshape(-1, 3)
        with torch.no_grad():
            p = cam.project(torch.as_tensor(safe_xyz, dtype=cam.matrix.dtype)).cpu().numpy()
        p = p.reshape(WINDOW, k, 2) - np.asarray(rig.offset[camera], dtype=np.float64)
        projected[:, :, ci] = p.astype(np.float32)

    pair = source_valid2d & valid3d[:, :, None]
    residual = np.linalg.norm(projected - source_xy, axis=-1)
    bad_pair = pair & (~np.isfinite(residual) | (residual > REPROJECTION_THRESHOLD_PX))
    bad_3d = bad_pair.any(axis=-1)
    accepted3d = valid3d & ~bad_3d
    accepted2d = pair & ~bad_3d[:, :, None] & ~bad_pair

    points3d = np.full((1, WINDOW, k, 3), np.nan, dtype=np.float32)
    vis3d = np.full((1, WINDOW, k), fmt.UNLABELED, dtype=np.int8)
    points3d[0] = xyz
    vis3d[0, accepted3d] = fmt.VISIBLE

    points2d = np.full((1, WINDOW, k, 6, 2), np.nan, dtype=np.float32)
    vis2d = np.full((1, WINDOW, k, 6), fmt.UNLABELED, dtype=np.int8)
    points2d[0] = projected
    vis2d[0, accepted2d] = fmt.PROJECTED

    return fmt.Labels(
        animal_ids=['a0'], points3d=points3d, vis3d=vis3d,
        points2d=points2d, vis2d=vis2d, boxes=None, instance=None,
    )


def write_session_data(source_session: Path, output_session: Path,
                       selected: list[tuple[str, float, int]], data: dict[str, np.ndarray],
                       rig: fmt.Rig, cameras: list[str], rate: Fraction,
                       write_videos: bool = True) -> None:
    """Write one session's tables, then the exact refonly clips for its groups.

    Inputs: source_session -- source directory (videos/refonly).
            output_session -- destination session directory.
            selected -- (group_id, mean_3d_speed, start) per group.
            data -- the session's loaded npz arrays.
            rig -- calibrated camera rig.
            cameras -- camera names.
            rate -- frame rate for the output clips.
            write_videos -- when false, tables only (--repair-labels mode).
    Side effects: writes the session (tables + groups/*.mp4 clips).
    """
    groups = {}
    labels = {}
    for gid, score, start in selected:
        groups[gid] = fmt.Group(
            group_id=gid, n_frames=WINDOW, fps=float(rate),
            source_video=f'{source_session.name}/videos/refonly',
            source_frame_start=start, source_frame_step=1,
            notes=f'mean_3d_speed={score:.6f}',
        )
        labels[gid] = make_labels(data, start, rig)

    fmt.write_session(
        output_session, mode='3d', units='mm', label_source='tracked', names=KEYPOINTS,
        rig=rig, groups=groups, labels=labels, skeleton=SKELETON, flip_pairs=FLIP_PAIRS,
        provenance={
            'source': 'qdmouse4m',
            'annotator': '',
            'annotator_tool': 'five-model consensus + EKS + manual jump review',
            'created': '2025-11-10',
        },
    )
    if write_videos:
        write_video_clips(source_session, output_session,
                          [(gid, start) for gid, _, start in selected], cameras, rate)


def validate_output() -> None:
    """Validate the generated dataset; raise on any error.

    Side effects: prints a summary; raises RuntimeError listing the first 50 errors.
    """
    ds = fmt.load_dataset(OUTPUT)
    errors = fmt.validate_dataset(ds, check_images=False)
    if errors:
        raise RuntimeError('validation failed:\n' + '\n'.join(errors[:50]))
    print(f'validated {len(ds.all_sessions())} sessions')


def plan() -> tuple[dict[str, list[tuple[str, float, int]]], dict]:
    """Select windows for every session and build the run manifest.

    Inputs: none (reads SOURCE sessions on disk).
    Outputs: (selections, manifest) where selections maps session name to
             (group_id, score, start) tuples.
    Side effects: prints a per-session selection summary; raises if a held-out
                  session is absent.
    """
    sessions = session_dirs()
    names = {s.name for s in sessions}
    if VAL_SESSION not in names or TEST_SESSION not in names:
        raise RuntimeError('held-out session is absent from source')
    train_sessions = [s for s in sessions if s.name not in {VAL_SESSION, TEST_SESSION}]
    counts = allocate_train_counts(train_sessions, TRAIN_GROUPS)
    selections: dict[str, list[tuple[str, float, int]]] = {}
    for session in sessions:
        data = load_npz(session)
        if session.name == VAL_SESSION:
            want = 1
            split = 'val'
        elif session.name == TEST_SESSION:
            want = 5
            split = 'test'
        else:
            want = counts[session.name]
            split = 'train'
        chosen = choose_windows(data, want)
        selections[session.name] = [
            (f'g{i:04d}', score, start) for i, (score, start) in enumerate(chosen)
        ]
        print(f'{split:5s} {session.name}: {len(chosen):4d} groups, '
              f'frames {chosen[0][1]}..{chosen[-1][1] + WINDOW - 1}, '
              f'speed {min(x[0] for x in chosen):.4f}..{max(x[0] for x in chosen):.4f}')
    manifest = {
        'source': str(SOURCE), 'output': str(OUTPUT), 'window_frames': WINDOW,
        'video_variant': 'refonly', 'train_frames': TRAIN_GROUPS * WINDOW,
        'val_frames': WINDOW, 'test_frames': 5 * WINDOW,
        'val_session': VAL_SESSION, 'test_session': TEST_SESSION,
        'selections': {
            name: [{'group_id': gid, 'mean_3d_speed': score, 'source_frame_start': start}
                   for gid, score, start in groups]
            for name, groups in selections.items()
        },
    }
    return selections, manifest


def main() -> None:
    """Convert QDMouse4M into the tailcycle dataset; 0 on success.

    Inputs: argv (via argparse): --dry-run, --validate, --overwrite,
            --repair-labels.
    Side effects: writes the dataset under OUTPUT; prints progress.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='select and print groups only')
    parser.add_argument('--validate', action='store_true', help='validate the generated dataset')
    parser.add_argument('--overwrite', action='store_true', help='remove an existing output first')
    parser.add_argument('--repair-labels', action='store_true',
                        help='rewrite pose tables in the existing output without re-encoding videos')
    args = parser.parse_args()

    if args.repair_labels:
        if not (OUTPUT / 'selection.json').exists():
            raise SystemExit(f'{OUTPUT}/selection.json is required for --repair-labels')
        manifest = json.loads((OUTPUT / 'selection.json').read_text())
        for session in session_dirs():
            split = ('val' if session.name == VAL_SESSION else
                     'test' if session.name == TEST_SESSION else 'train')
            output_session = OUTPUT / split / session.name
            if not output_session.exists():
                raise SystemExit(f'missing generated session {output_session}')
            selected = [(str(row['group_id']), float(row['mean_3d_speed']),
                         int(row['source_frame_start']))
                        for row in manifest['selections'][session.name]]
            rig = fmt.load_calibration(session / 'calibrations' / 'camera_calibration.toml')
            data = load_npz(session)
            cameras = [str(v) for v in data['view_names']]
            rate = source_fps(session / 'videos' / 'refonly' / f'{cameras[0]}.mp4')
            write_session_data(session, output_session, selected, data, rig, cameras, rate,
                               write_videos=False)
            print(f'repaired labels {output_session}')
        (OUTPUT / 'label_policy.json').write_text(json.dumps({
            'two_d_source': 'projection_of_3d',
            'reprojection_threshold_px': REPROJECTION_THRESHOLD_PX,
            'source_2d_gate': 'valid_2d_conf0p5',
            'interpolated_3d': 'unlabeled',
            'invalid_reprojection': 'unlabeled',
            'statuses_emitted': ['visible', 'projected'],
        }, indent=2) + '\n')
        if args.validate:
            validate_output()
        return

    if OUTPUT.exists():
        if not args.overwrite:
            raise SystemExit(f'{OUTPUT} exists; use --overwrite to replace it')
        shutil.rmtree(OUTPUT)

    selections, manifest = plan()
    if args.dry_run:
        return

    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for session in session_dirs():
        split = 'val' if session.name == VAL_SESSION else 'test' if session.name == TEST_SESSION else 'train'
        output_session = OUTPUT / split / session.name
        rig = fmt.load_calibration(session / 'calibrations' / 'camera_calibration.toml')
        data = load_npz(session)
        cameras = [str(v) for v in data['view_names']]
        if cameras != rig.names:
            raise RuntimeError(f'{session}: npz cameras {cameras} disagree with calibration {rig.names}')
        rate = source_fps(session / 'videos' / 'refonly' / f'{cameras[0]}.mp4')
        write_session_data(session, output_session, selections[session.name], data, rig, cameras, rate)
        print(f'wrote {output_session}')

    if args.validate:
        validate_output()


if __name__ == '__main__':
    main()
