#!/usr/bin/env python
"""Convert the minimaze SLEAP labels into tailcycle 2D sessions.

    pixi run python scripts/convert_minimaze.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import av
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from tailcyclenet import format as fmt
from tailcyclenet.video import PyAVReader

SRC = Path('/groups/voigts/voigtslab/alison_temp/minimaze_calibration_test_data/sleap/minimaze.v004.slp')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/comrie-minimaze')
WINDOW = 24


def _paths_and_labels(path: Path):
    """Read SLEAP tables, using skeleton-0 node order for the point axis."""
    with h5py.File(path, 'r') as f:
        videos = [json.loads(raw.decode())['backend']['filename']
                  for raw in f['videos_json'][:]]
        videos = [p.replace('/mnt/v/', '/groups/voigts/voigtslab/', 1) for p in videos]
        frames = f['frames'][:]
        instances = f['instances'][:]
        points = f['points'][:]
        metadata = json.loads(f['metadata'].attrs['json'])

    used_skeletons = sorted(set(int(x) for x in instances['skeleton']))
    if used_skeletons != [0]:
        raise RuntimeError(f'expected only skeleton 0 in use, got {used_skeletons}')
    skeleton = metadata['skeletons'][0]
    node_names = [metadata['nodes'][int(node['id'])]['name'] for node in skeleton['nodes']]
    edges = [[metadata['nodes'][int(edge['source'])]['name'],
              metadata['nodes'][int(edge['target'])]['name']]
             for edge in skeleton['links']]
    if len(node_names) != 8 or len(set(node_names)) != len(node_names):
        raise RuntimeError(f'unexpected skeleton nodes: {node_names}')

    labels = defaultdict(list)
    excluded = []
    for frame in frames:
        vi = int(frame['video'])
        source = videos[vi]
        a0, a1 = int(frame['instance_id_start']), int(frame['instance_id_end'])
        if a1 - a0 != 1:
            raise RuntimeError(f'{path}: frame {int(frame["frame_id"])} has {a1-a0} instances')
        inst = instances[a0]
        p0, p1 = int(inst['point_id_start']), int(inst['point_id_end'])
        xy = np.stack([points[p0:p1]['x'], points[p0:p1]['y']], axis=-1)
        vis = points[p0:p1]['visible'].astype(bool)
        source_frame = int(frame['frame_idx'])
        if xy.shape != (len(node_names), 2) or not np.isfinite(xy).all():
            raise RuntimeError(f'{path}: non-finite or wrong-shaped points at frame {int(frame["frame_id"])}')
        if not vis.any():
            excluded.append({'source_video': source, 'source_frame': source_frame,
                             'reason': 'all 8 points visible=false; SLEAP placeholder instance, '
                                      'not a label'})
            continue
        labels[source].append({'source_frame': source_frame, 'xy': xy, 'visible': vis})
    if len(excluded) != 4:
        raise RuntimeError(f'expected 4 all-invisible SLEAP placeholders, found {len(excluded)}')
    return labels, node_names, edges, excluded


def _probe(video: Path) -> tuple[int, int, int, float]:
    """Return frame count, width, height and guessed frame rate for one source video."""
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        n = int(stream.frames)
        width, height = int(stream.codec_context.width), int(stream.codec_context.height)
        fps = float(stream.guessed_rate or stream.average_rate)
    if n < WINDOW:
        raise RuntimeError(f'{video}: only {n} frames, cannot make {WINDOW}-frame groups')
    return n, width, height, fps


def _clusters(frames: list[int]) -> list[list[int]]:
    """Greedily pack neighboring labels whose inclusive span fits in one group."""
    out: list[list[int]] = []
    for frame in sorted(frames):
        if not out or frame - out[-1][0] >= WINDOW:
            out.append([frame])
        else:
            out[-1].append(frame)
    return out


def _window(cluster: list[int], n_video: int) -> tuple[int, int]:
    """Return a clamped source-video window centered on a label cluster."""
    center = (cluster[0] + cluster[-1]) // 2
    start = min(max(0, center - WINDOW // 2), n_video - WINDOW)
    return int(start), WINDOW


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Atomically write one decoded RGB frame, compacting grayscale frames to L PNG."""
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
        image = Image.fromarray(rgb[..., 0], mode='L')
    else:
        image = Image.fromarray(rgb, mode='RGB')
    tmp = path.parent / f'.{path.name}.tmp'
    image.save(tmp, format='PNG', compress_level=1)
    os.replace(tmp, path)


def _valid_png(path: Path, width: int, height: int) -> bool:
    """Return whether an existing cache PNG decodes with the expected dimensions and mode."""
    try:
        with Image.open(path) as image:
            if image.size != (width, height) or image.mode not in {'L', 'RGB'}:
                return False
            image.load()
        return True
    except (OSError, ValueError):
        return False


def _extract(video: Path, indices: set[int], cache: Path, width: int, height: int) -> None:
    """Extract only requested frames, seeking to each contiguous requested range with PyAV."""
    cache.mkdir(parents=True, exist_ok=True)
    valid = {i for i in indices if _valid_png(cache / f'{i:06d}.png', width, height)}
    missing = indices - valid
    if not missing:
        return

    ordered = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index != previous + 1:
            ranges.append((start, previous))
            start = index
        previous = index
    ranges.append((start, previous))

    reader = PyAVReader(str(video))
    if reader.frame_shape()[:2] != (height, width):
        reader.close()
        raise RuntimeError(f'{video}: dimensions changed while extracting')
    try:
        with tqdm(total=len(missing), desc=f'extract {video.name}', unit='frame',
                  dynamic_ncols=True) as progress:
            for start, end in ranges:
                needed = [i for i in range(start, end + 1) if i in missing]
                if not needed:
                    continue
                frames = reader.get_batch(range(start, end + 1))
                for index, rgb in zip(range(start, end + 1), frames):
                    if index in missing:
                        _write_png(cache / f'{index:06d}.png', rgb)
                        valid.add(index)
                        progress.update(1)
    finally:
        reader.close()
    missing = sorted(indices - valid)
    if missing:
        raise RuntimeError(f'{video}: failed to decode requested frames {missing[:10]}')


def _link_frames(session: Path, gid: str, start: int, cache: Path) -> None:
    """Link one group's local frame names to its source-indexed cache entries."""
    target = session / 'groups' / gid / 'cam0'
    target.mkdir(parents=True, exist_ok=True)
    for local in range(WINDOW):
        src = cache / f'{start + local:06d}.png'
        if not src.is_file():
            raise RuntimeError(f'missing extracted frame {src}')
        dst = target / f'{local:06d}.png'
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(Path(os.path.relpath(src, target)))


def _make_labels(group_anchors: list[dict], start: int, names: list[str]) -> fmt.Labels:
    """Build sparse one-animal 2D labels at the cluster's source-frame offsets."""
    K = len(names)
    points = np.full((1, WINDOW, K, 1, 2), np.nan, dtype=np.float32)
    status = np.full((1, WINDOW, K, 1), fmt.UNLABELED, dtype=np.int8)
    for row in group_anchors:
        local = row['source_frame'] - start
        visible = row['visible']
        points[0, local, visible, 0] = row['xy'][visible]
        status[0, local, visible, 0] = fmt.VISIBLE
        status[0, local, ~visible, 0] = fmt.MISSING
    return fmt.Labels(animal_ids=['animal0'], points3d=None, vis3d=None,
                      points2d=points, vis2d=status, boxes=None, instance=None, regions=None)


def convert(src: Path, out: Path, clean: bool = False) -> None:
    """Convert the SLP into six physical-video sessions under a tailcycle dataset root."""
    labels_by_video, names, edges, excluded = _paths_and_labels(src)
    if out.exists():
        if clean:
            shutil.rmtree(out)
        elif not (out / '_frame_cache').is_dir():
            raise RuntimeError(f'{out} exists; pass --clean to replace it')
        else:
            shutil.rmtree(out / 'train', ignore_errors=True)
            (out / 'excluded_frames.pq').unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    total_groups = total_labels = 0
    for video_i, (video_text, rows) in enumerate(sorted(labels_by_video.items())):
        video = Path(video_text)
        if not video.is_file():
            raise RuntimeError(f'SLP video does not exist after prefix replacement: {video}')
        n_video, width, height, fps = _probe(video)
        rows_by_frame = {int(r['source_frame']): r for r in rows}
        if len(rows_by_frame) != len(rows):
            raise RuntimeError(f'{video}: duplicate labeled source frame')
        clusters = _clusters(sorted(rows_by_frame))
        windows = [(_window(cluster, n_video), cluster) for cluster in clusters]
        session_id = f'video{video_i:02d}_{video.stem}'
        session_path = out / 'train' / session_id
        cache = out / '_frame_cache' / session_id
        required = {start + local for (start, _), _ in windows for local in range(WINDOW)}
        cache.mkdir(parents=True, exist_ok=True)
        for cached in cache.iterdir():
            if cached.suffix == '.png' and int(cached.stem) not in required:
                cached.unlink()
        print(f'{session_id}: {len(rows)} labels -> {len(windows)} groups; '
              f'{len(required)} unique context frames; source has {n_video} frames at {fps:g} Hz')
        _extract(video, required, cache, width, height)

        rig = fmt.Rig(cgroup=__import__('aniposelib.cameras', fromlist=['CameraGroup']).CameraGroup([
            fmt.nominal_camera('cam0', (width, height))]), offset={'cam0': (0.0, 0.0)},
            moving={'cam0': False}, calibrated={'cam0': False})
        groups = {}
        group_labels = {}
        for gi, ((start, length), cluster) in enumerate(windows):
            gid = f'g{video_i:02d}_{gi:04d}_{cluster[0]:09d}_{cluster[-1]:09d}'
            groups[gid] = fmt.Group(
                gid, length, fps=fps, source_video=str(video), source_frame_start=start,
                source_frame_step=1,
                notes=f'24 frames of context; labeled source frames {cluster[0]}..{cluster[-1]}')
            group_labels[gid] = _make_labels([rows_by_frame[f] for f in cluster], start, names)
            _link_frames(session_path, gid, start, cache)
        fmt.write_session(
            session_path, mode='2d', units='px', label_source='annotated', names=names, rig=rig,
            groups=groups, labels=group_labels, skeleton=edges, flip_pairs=[['earL', 'earR']],
            provenance={
                'source': str(src), 'source_slp': str(src), 'source_video': str(video),
                'converter': 'scripts/convert_minimaze.py', 'annotator': 'Comrie/Voigts Lab',
                'annotator_tool': 'SLEAP', 'created': '2026-03-09',
                'split_note': 'all retained labels are under train; source supplied no split',
                'coordinate_note': 'SLEAP visible=true points are visible; visible=false points are '
                                   'missing assessments and their stale coordinates are omitted',
                'excluded_frames_note': 'four all-invisible SLEAP placeholder instances were excluded; '
                                        'see root excluded_frames.pq',
                'window_note': 'labels greedily clustered when their inclusive span fits in 24 frames; '
                               'each group contains 24 source-video frames of context',
                'video_path_rewrite': '/mnt/v/ replaced with /groups/voigts/voigtslab/',
            })
        total_groups += len(groups)
        total_labels += len(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(excluded), out / 'excluded_frames.pq')
    print(f'converted {total_labels} labeled frames into {total_groups} groups at {out}; '
          f'excluded {len(excluded)} placeholder frames')


def main() -> None:
    """Parse conversion paths and build the requested dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, default=SRC)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    convert(args.src, args.out, args.clean)


if __name__ == '__main__':
    main()
