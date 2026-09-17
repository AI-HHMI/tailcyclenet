#!/usr/bin/env python
"""Clean the processed QDMouse4M dataset using scorer quality and pose jumps.

The cleaner keeps the source sessions and camera calibration, removes low-quality label rows
(as no-label/UNLABELED cells), and splits a parent group at coherent 3D body-centroid jumps. Child
pixel clips are cut from the processed group-local videos, never from raw full-session videos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tailcyclenet import format as fmt

SOURCE = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/qdmouse4m')
SCORES = Path('/groups/karashchuk/home/karashchukl/projects/tailcycle/tailcyclenet/'
              'scratch/qdmouse-full-score-25399/merged/scores.pq')
OUTPUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
              'tailcycle-datasets/qdmouse4m-cleaned')
JUMP_THRESHOLD_MM = 8.0
SCORE_THRESHOLD = 0.1
SCORE_WINDOW = 12
CHECKPOINT_ITERATION = 25399
CHECKPOINT_SHA256 = 'f263fb124a258a6e7dfbe6971be1cb6ebc0ef6177a33ba03e797fbba7f21c411'

MANIFEST_COLUMNS = [
    'split', 'session', 'parent_group', 'child_group', 'parent_n_frames',
    'child_local_start', 'child_local_end', 'child_n_frames', 'source_frame_start',
    'source_frame_end', 'jump_cuts_in_parent', 'reason', 'n_unlabeled_3d',
    'n_unlabeled_2d', 'dropped_frames',
]
MANIFEST_NULL_VALUES = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN',
                       '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN',
                       'None', 'n/a', 'nan', 'null']
MANIFEST_SCHEMA = {
    'split': pl.String, 'session': pl.String, 'parent_group': pl.String,
    'child_group': pl.String, 'parent_n_frames': pl.Int64,
    'child_local_start': pl.Int64, 'child_local_end': pl.Int64, 'child_n_frames': pl.Int64,
    'source_frame_start': pl.Int64, 'source_frame_end': pl.Int64,
    'jump_cuts_in_parent': pl.String, 'reason': pl.String,
    'n_unlabeled_3d': pl.Int64, 'n_unlabeled_2d': pl.Int64, 'dropped_frames': pl.Int64,
}


def write_manifest(manifest: list[dict[str, object]] | pl.DataFrame, path: Path) -> None:
    """Write the stable 15-column cleaning manifest as a minimally quoted TSV.

    Python csv minimal quoting keeps empty jump-cut fields blank rather than quoting them;
    None renders as an empty field.
    """
    if isinstance(manifest, pl.DataFrame):
        frame = manifest
    elif manifest:
        frame = pl.DataFrame(manifest)
    else:
        frame = pl.DataFrame([], schema=MANIFEST_SCHEMA)
    if set(frame.columns) != set(MANIFEST_COLUMNS):
        raise RuntimeError(f'cleaning manifest columns differ: {frame.columns}')
    integer_columns = [name for name, dtype in MANIFEST_SCHEMA.items() if dtype == pl.Int64]
    for name in integer_columns:
        values = frame[name].to_list()
        invalid = [value for value in values
                   if isinstance(value, (float, np.floating)) and
                   (not np.isnan(value) and
                    (not np.isfinite(value) or float(value) != int(value)))]
        if invalid:
            raise RuntimeError(f'cleaning manifest {name} contains non-integer values')
    float_columns = [name for name, dtype in frame.schema.items()
                     if dtype in (pl.Float32, pl.Float64)]
    if float_columns:
        frame = frame.with_columns(*(pl.col(name).fill_nan(None) for name in float_columns))
    frame = frame.select(MANIFEST_COLUMNS).cast(MANIFEST_SCHEMA, strict=True)
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(frame.iter_rows())


def read_manifest(path: Path) -> pl.DataFrame:
    """Read and validate a cleaning manifest without changing its column contract."""
    frame = pl.read_csv(path, separator='\t', null_values=MANIFEST_NULL_VALUES,
                        schema_overrides=MANIFEST_SCHEMA)
    if frame.columns != MANIFEST_COLUMNS:
        raise RuntimeError(f'{path}: expected cleaning manifest columns {MANIFEST_COLUMNS}, '
                           f'got {frame.columns}')
    return frame


def sha256(path: Path) -> str:
    """Return a file's SHA256 digest."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_rate(stream) -> float:
    """Return a video stream's exact average rate as a float."""
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        raise RuntimeError(f'video has no frame rate: {stream}')
    return float(rate)


def jump_cuts(labels: fmt.Labels, names: list[str], threshold_mm: float) -> list[int]:
    """Return local target-frame indices where a non-tail 3D centroid jumps.

    A returned index ``t`` denotes the transition ``t-1 -> t``. Splitting at ``t`` therefore
    keeps every source frame while ensuring that transition is a child boundary. A transition is
    considered only when both centroids have finite, positioned points.
    """
    if labels.points3d is None or labels.vis3d is None:
        return []
    non_tail = [i for i, name in enumerate(names) if not name.startswith('tail_')]
    if not non_tail:
        return []
    xyz = np.asarray(labels.points3d)
    positioned = np.isin(labels.vis3d, fmt.POSITIONED)
    cuts: set[int] = set()
    for animal in range(xyz.shape[0]):
        centroids = np.full((xyz.shape[1], 3), np.nan, dtype=np.float64)
        for frame in range(xyz.shape[1]):
            valid = positioned[animal, frame, non_tail]
            values = xyz[animal, frame, non_tail][valid]
            if len(values):
                finite = np.isfinite(values).all(axis=1)
                if finite.any():
                    centroids[frame] = values[finite].mean(axis=0)
        displacement = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        cuts.update((np.flatnonzero(np.isfinite(displacement) &
                                    (displacement > threshold_mm)) + 1).tolist())
    return sorted(cuts)


def segments(n_frames: int, cuts: list[int]) -> list[tuple[int, int]]:
    """Convert target transition indices into half-open child frame intervals."""
    valid = sorted({int(c) for c in cuts if 0 < int(c) < n_frames})
    bounds = [0, *valid, n_frames]
    return list(zip(bounds[:-1], bounds[1:]))


def score_masks(scores: pl.DataFrame, session: str, group: str, names: list[str],
                animals: list[str], n_frames: int, *, threshold: float = SCORE_THRESHOLD,
                window_frames: int = SCORE_WINDOW) -> dict[str, np.ndarray]:
    """Build per-animal ``(frame,keypoint)`` low-score masks using any covering window.

    Missing score rows do not mark a point low. This means an entirely unscored frame remains as
    it was; the caller records the score source and policy in the dataset provenance.
    """
    cols = ['session', 'group', 'animal', 'start', 'keypoint']
    subset = scores.filter((pl.col('session') == session) & (pl.col('group') == group))
    if subset.is_empty():
        return {}
    if subset.select(pl.struct(cols).is_duplicated().any()).item():
        raise RuntimeError(f'duplicate scorer rows for {session}/{group}')
    kpt_index = {name: i for i, name in enumerate(names)}
    unknown = sorted(set(subset.get_column('keypoint').to_list()) - set(kpt_index))
    if unknown:
        raise RuntimeError(f'{session}/{group}: scores contain unknown keypoints {unknown}')
    out: dict[str, np.ndarray] = {}
    for animal_group in subset.group_by('animal', maintain_order=True):
        (animal,), adf = animal_group
        by_start: dict[int, np.ndarray] = {}
        for start_group in adf.sort('start').group_by('start', maintain_order=True):
            (start,), sdf = start_group
            values = np.full(len(names), np.nan, dtype=np.float64)
            for row in sdf.iter_rows(named=True):
                values[kpt_index[row['keypoint']]] = float(row['score'])
            by_start[int(start)] = values
        low = np.zeros((n_frames, len(names)), dtype=bool)
        starts = sorted(by_start)
        for frame in range(n_frames):
            covering = [s for s in starts if s <= frame < s + window_frames]
            for start in covering:
                values = by_start[start]
                low[frame] |= np.isfinite(values) & (values < threshold)
        out[str(animal)] = low
    missing_animals = sorted(set(animals) - set(out))
    if missing_animals:
        raise RuntimeError(f'{session}/{group}: scores missing animals {missing_animals}')
    return out


def has_labels(labels: fmt.Labels) -> bool:
    """Whether the loader's target visibility array contains any labelled cell."""
    vis = labels.vis3d if labels.vis3d is not None else labels.vis2d
    return vis is not None and bool(np.any(vis != fmt.UNLABELED))


def clean_labels(labels: fmt.Labels, names: list[str], masks: dict[str, np.ndarray]) -> fmt.Labels:
    """Copy labels and turn low-score cells into no-label cells with null coordinates."""
    out = fmt.Labels(
        animal_ids=list(labels.animal_ids),
        points3d=None if labels.points3d is None else np.array(labels.points3d, copy=True),
        vis3d=None if labels.vis3d is None else np.array(labels.vis3d, copy=True),
        points2d=None if labels.points2d is None else np.array(labels.points2d, copy=True),
        vis2d=None if labels.vis2d is None else np.array(labels.vis2d, copy=True),
        boxes=None if labels.boxes is None else np.array(labels.boxes, copy=True),
        instance=None if labels.instance is None else np.array(labels.instance, copy=True),
        ext=None if labels.ext is None else np.array(labels.ext, copy=True),
        regions=None if labels.regions is None else np.array(labels.regions, copy=True),
    )
    for ai, animal in enumerate(out.animal_ids):
        low = masks.get(animal)
        if low is None:
            continue
        if out.vis3d is not None:
            out.vis3d[ai] = np.where(low, fmt.UNLABELED, out.vis3d[ai])
            out.points3d[ai] = np.where(low[..., None], np.nan, out.points3d[ai])
        if out.vis2d is not None:
            out.vis2d[ai] = np.where(low[..., None], fmt.UNLABELED, out.vis2d[ai])
            out.points2d[ai] = np.where(low[..., None, None], np.nan, out.points2d[ai])
    return out


def slice_labels(labels: fmt.Labels, start: int, end: int) -> fmt.Labels:
    """Return a copied label slice with local frame coordinates and regions adjusted."""
    regions = None
    if labels.regions is not None:
        regions = np.array(labels.regions, copy=True)
        keep = (regions[:, 0] >= start) & (regions[:, 0] < end)
        regions = regions[keep]
        regions[:, 0] -= start
    return fmt.Labels(
        animal_ids=list(labels.animal_ids),
        points3d=None if labels.points3d is None else labels.points3d[:, start:end].copy(),
        vis3d=None if labels.vis3d is None else labels.vis3d[:, start:end].copy(),
        points2d=None if labels.points2d is None else labels.points2d[:, start:end].copy(),
        vis2d=None if labels.vis2d is None else labels.vis2d[:, start:end].copy(),
        boxes=None if labels.boxes is None else labels.boxes[:, start:end].copy(),
        instance=None if labels.instance is None else labels.instance[:, start:end].copy(),
        ext=None if labels.ext is None else labels.ext[:, start:end].copy(),
        regions=regions,
    )


def unlabeled_counts(before: fmt.Labels, after: fmt.Labels) -> tuple[int, int]:
    """Count 3D and 2D cells changed from a determination to no-label."""
    n3 = 0
    n2 = 0
    if before.vis3d is not None:
        n3 = int(((before.vis3d != fmt.UNLABELED) & (after.vis3d == fmt.UNLABELED)).sum())
    if before.vis2d is not None:
        n2 = int(((before.vis2d != fmt.UNLABELED) & (after.vis2d == fmt.UNLABELED)).sum())
    return n3, n2


def cut_clip(source: Path, target: Path, start: int, end: int, expected_fps: float) -> None:
    """Decode a group-local source interval and encode it as a lossless H.264 MP4."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    tmp.unlink(missing_ok=True)
    try:
        with av.open(str(source), mode='r') as inp:
            stream = next((s for s in inp.streams if s.type == 'video'), None)
            if stream is None:
                raise RuntimeError(f'no video stream: {source}')
            rate = source_rate(stream)
            if abs(rate - expected_fps) > 1e-4:
                raise RuntimeError(f'fps mismatch {source}: source={rate}, group={expected_fps}')
            wanted = list(range(start, end))
            if not wanted:
                raise RuntimeError(f'empty clip requested: {target}')
            time_base = stream.time_base
            if time_base is None:
                raise RuntimeError(f'video has no time base: {source}')
            seek_ts = int(wanted[0] / rate / float(time_base))
            inp.seek(max(0, seek_ts), stream=stream, any_frame=False, backward=True)
            selected: dict[int, av.VideoFrame] = {}
            for frame in inp.decode(stream):
                if frame.pts is None:
                    raise RuntimeError(f'missing PTS while decoding {source}')
                index = int(round(float(frame.pts * time_base * rate)))
                if index in wanted:
                    selected[index] = frame
                    if len(selected) == len(wanted):
                        break
                if index > wanted[-1]:
                    break
            if len(selected) != len(wanted):
                missing = [i for i in wanted if i not in selected]
                raise RuntimeError(f'{source}: missing source frames {missing[:5]}')
            with av.open(str(tmp), mode='w', format='mp4') as out:
                enc = out.add_stream('libx264', rate=stream.average_rate or stream.base_rate)
                enc.width, enc.height = stream.width, stream.height
                enc.pix_fmt = stream.pix_fmt or 'yuv420p'
                enc.options = {'crf': '0', 'preset': 'ultrafast'}
                from fractions import Fraction
                enc_tb = Fraction(1, 1) / (stream.average_rate or stream.base_rate)
                for j, index in enumerate(wanted):
                    frame = selected[index]
                    frame.pts = j
                    frame.time_base = enc_tb
                    for packet in enc.encode(frame):
                        out.mux(packet)
                for packet in enc.encode():
                    out.mux(packet)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _cut_worker(task: tuple[str, str, int, int, float]) -> str:
    """Pool worker for one camera/child clip."""
    source, target, start, end, fps = task
    cut_clip(Path(source), Path(target), start, end, fps)
    return target


def link_group_pixels(src_group: fmt.Group, dst_dir: Path, cameras: list[str]) -> None:
    """Link an unsplit child to the absolute processed group-local source clips."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        kind, source = src_group.pixels(camera)
        if kind != 'video':
            raise RuntimeError(f'expected video source for {src_group.group_id}/{camera}, got {kind}')
        fmt.link(dst_dir / source.name, source.resolve())


def write_root_provenance(root: Path, source: Path, scores: Path, *, jump_threshold: float,
                          score_threshold: float, score_window: int, code_commit: str,
                          code_dirty: bool, checkpoint_iteration: int,
                          checkpoint_sha256: str, scorer_code_commit: str | None,
                          scorer_code_dirty: bool | None) -> None:
    """Write the root-level provenance record for the cleaning operation."""
    import toml
    doc = {
        'kind': 'qdmouse4m-cleaned',
        'source_root': str(source),
        'score_table': str(scores),
        'score_table_sha256': sha256(scores),
        'scorer_checkpoint_iteration': checkpoint_iteration,
        'scorer_checkpoint_sha256': checkpoint_sha256,
        'jump_metric': 'non-tail 3D body-centroid displacement per frame',
        'jump_threshold_mm_per_frame': float(jump_threshold),
        'score_threshold': float(score_threshold),
        'score_mapping': 'any covering window',
        'score_window_frames': int(score_window),
        'short_segments': 'kept',
        'code_commit': code_commit,
        'code_dirty': bool(code_dirty),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    if scorer_code_commit is not None:
        doc['scorer_code_commit'] = scorer_code_commit
    if scorer_code_dirty is not None:
        doc['scorer_code_dirty'] = scorer_code_dirty
    (root / 'provenance.toml').write_text(toml.dumps(doc))


def build(source: Path, scores_path: Path, output: Path, *, jump_threshold: float,
          score_threshold: float, score_window: int, workers: int, overwrite: bool,
          checkpoint_iteration: int, checkpoint_sha256: str,
          scores_sha256: str | None, scorer_code_commit: str | None,
          scorer_code_dirty: bool | None, write_videos: bool = True) -> dict[str, int]:
    """Build the cleaned dataset atomically in a sibling staging directory."""
    if output.exists() and not overwrite:
        raise RuntimeError(f'{output} exists; pass --overwrite to replace it')
    source_ds = fmt.load_dataset(source)
    scores = pl.read_parquet(scores_path)
    if scores_sha256 is not None:
        actual_scores_sha256 = sha256(scores_path)
        if actual_scores_sha256 != scores_sha256:
            raise RuntimeError(
                f'{scores_path}: SHA256 {actual_scores_sha256} != expected {scores_sha256}')
    required = {'session', 'group', 'animal', 'start', 'keypoint', 'score'}
    missing = required - set(scores.columns)
    if missing:
        raise RuntimeError(f'{scores_path}: missing score columns {sorted(missing)}')
    finite = scores.select(pl.col('score').is_finite().fill_null(False).all()).item()
    if not finite:
        raise RuntimeError(f'{scores_path}: non-finite scores')
    score_keys = ['session', 'group', 'animal', 'start', 'keypoint']
    if scores.select(pl.struct(score_keys).is_duplicated().any()).item():
        raise RuntimeError(f'{scores_path}: duplicate score keys')
    stage = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    tasks: list[tuple[str, str, int, int, float]] = []
    counts = {'sessions': 0, 'groups': 0, 'children': 0, 'dropped_no_labels': 0,
              'dropped_frames': 0, 'split_parents': 0, 'jump_transitions': 0,
              'symlinks': 0, 'cut_clips': 0, 'unlabeled_3d': 0, 'unlabeled_2d': 0}
    try:
        for split in fmt.SPLITS:
            for src_session in source_ds.sessions.get(split, []):
                out_session = stage / split / src_session.session_id
                out_session.mkdir(parents=True, exist_ok=True)
                counts['sessions'] += 1
                session_labels: dict[str, fmt.Labels] = {}
                out_groups: dict[str, fmt.Group] = {}
                child_pixels: list[tuple[str, fmt.Group, int, int, bool]] = []
                visibility_before = src_session.has_visibility_assessment
                for parent_id, src_group in src_session.groups.items():
                    original = src_session.labels(parent_id)
                    masks = score_masks(scores, src_session.session_id, parent_id,
                                        src_session.names, original.animal_ids, src_group.n_frames,
                                        threshold=score_threshold, window_frames=score_window)
                    cleaned = clean_labels(original, src_session.names, masks)
                    cuts = jump_cuts(original, src_session.names, jump_threshold)
                    counts['jump_transitions'] += len(cuts)
                    spans = segments(src_group.n_frames, cuts)
                    if len(spans) > 1:
                        counts['split_parents'] += 1
                    for child_no, (start, end) in enumerate(spans):
                        child_id = parent_id if len(spans) == 1 else f'{parent_id}_s{child_no}'
                        child = slice_labels(cleaned, start, end)
                        before_child = slice_labels(original, start, end)
                        n3, n2 = unlabeled_counts(before_child, child)
                        counts['unlabeled_3d'] += n3
                        counts['unlabeled_2d'] += n2
                        raw_start = src_group.source_frame_start + start * src_group.source_frame_step
                        raw_end = src_group.source_frame_start + (end - 1) * src_group.source_frame_step
                        note = src_group.notes
                        suffix = f'; parent_group={parent_id}; cleaned_score_lt={score_threshold:g}'
                        if len(spans) > 1:
                            note += f'; split_at_centroid_gt={jump_threshold:g}mm'
                        reason = 'unsplit' if len(spans) == 1 else 'jump_split'
                        if not has_labels(child):
                            counts['dropped_no_labels'] += 1
                            counts['dropped_frames'] += end - start
                            manifest.append({
                                'split': split, 'session': src_session.session_id,
                                'parent_group': parent_id, 'child_group': child_id,
                                'parent_n_frames': src_group.n_frames,
                                'child_local_start': start, 'child_local_end': end,
                                'child_n_frames': end - start,
                                'source_frame_start': raw_start, 'source_frame_end': raw_end,
                                'jump_cuts_in_parent': ','.join(map(str, cuts)),
                                'reason': 'dropped_no_labels', 'n_unlabeled_3d': n3,
                                'n_unlabeled_2d': n2, 'dropped_frames': end - start,
                            })
                            continue
                        out_groups[child_id] = fmt.Group(
                            group_id=child_id, n_frames=end - start, fps=src_group.fps,
                            source_video=src_group.source_video,
                            source_frame_start=raw_start,
                            source_frame_step=src_group.source_frame_step,
                            notes=(note + suffix).strip('; '),
                        )
                        session_labels[child_id] = child
                        child_pixels.append((child_id, src_group, start, end, len(spans) == 1))
                        manifest.append({
                            'split': split, 'session': src_session.session_id,
                            'parent_group': parent_id, 'child_group': child_id,
                            'parent_n_frames': src_group.n_frames,
                            'child_local_start': start, 'child_local_end': end,
                            'child_n_frames': end - start,
                            'source_frame_start': raw_start, 'source_frame_end': raw_end,
                            'jump_cuts_in_parent': ','.join(map(str, cuts)),
                            'reason': reason, 'n_unlabeled_3d': n3, 'n_unlabeled_2d': n2,
                            'dropped_frames': 0,
                        })
                if visibility_before != src_session.has_visibility_assessment:
                    raise RuntimeError(f'{src_session.path}: visibility assessment changed in source')
                prov = dict(src_session.provenance)
                prov.update({'source': str(src_session.path), 'cleaning': 'qdmouse4m-cleaned',
                             'cleaning_score_threshold': float(score_threshold),
                             'cleaning_jump_threshold_mm_per_frame': float(jump_threshold),
                             'cleaning_score_mapping': 'any covering window'})
                fmt.write_session(
                    out_session, mode=src_session.mode, units=src_session.units,
                    label_source=src_session.label_source, names=src_session.names, rig=src_session.rig,
                    groups=out_groups, labels=session_labels, skeleton=src_session.skeleton,
                    flip_pairs=(src_session.flip_pairs if src_session.flip_pairs_declared else None),
                    provenance=prov,
                    assoc_res_max_px=src_session.assoc_res_max_px,
                )
                for child_id, src_group, start, end, unsplit in child_pixels:
                    dst_dir = out_session / 'groups' / child_id
                    if unsplit:
                        link_group_pixels(src_group, dst_dir, src_session.cam_names)
                        counts['symlinks'] += len(src_session.cam_names)
                    else:
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        for camera in src_session.cam_names:
                            kind, src_video = src_group.pixels(camera)
                            if kind != 'video':
                                raise RuntimeError(f'expected video source for {src_group.group_id}/{camera}')
                            target = dst_dir / f'{camera}.mp4'
                            tasks.append((str(src_video.resolve()), str(target), start, end,
                                          float(src_group.fps)))
                counts['groups'] += len(out_groups)
                counts['children'] += len(out_groups)
        if write_videos and tasks:
            from multiprocessing import get_context
            with get_context('spawn').Pool(processes=max(1, workers)) as pool:
                for _ in pool.imap_unordered(_cut_worker, tasks):
                    counts['cut_clips'] += 1
        elif not write_videos and tasks:
            counts['cut_clips'] = 0
        code = subprocess_run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent.parent)
        dirty = bool(subprocess_run(['git', 'status', '--short'], cwd=Path(__file__).resolve().parent.parent).strip())
        write_root_provenance(
            stage, source, scores_path, jump_threshold=jump_threshold,
            score_threshold=score_threshold, score_window=score_window,
            code_commit=code, code_dirty=dirty,
            checkpoint_iteration=checkpoint_iteration,
            checkpoint_sha256=checkpoint_sha256,
            scorer_code_commit=scorer_code_commit,
            scorer_code_dirty=scorer_code_dirty,
        )
        write_manifest(manifest, stage / 'cleaning_manifest.tsv')
        (stage / 'cleaning_summary.json').write_text(json.dumps(counts, indent=2) + '\n')
        if output.exists():
            if not overwrite:
                raise RuntimeError(f'{output} appeared during build')
            shutil.rmtree(output)
        stage.rename(output)
        return counts
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prune_unlabeled(root: Path) -> dict[str, int]:
    """Remove existing child groups whose loader target has no labelled cells."""
    import toml

    root = Path(root).resolve()
    ds = fmt.load_dataset(root)
    manifest_path = root / 'cleaning_manifest.tsv'
    if not manifest_path.exists():
        raise RuntimeError(f'{root}: missing cleaning_manifest.tsv')
    manifest = read_manifest(manifest_path)

    drops: list[tuple[str, fmt.Session, str]] = []
    before_rows: dict[tuple[str, str, str], int] = {}
    for split in fmt.SPLITS:
        for session in ds.sessions.get(split, []):
            for stem in ('points3d', 'keypoints', 'instances', 'regions', 'extrinsics'):
                table_path = session.path / f'{stem}.pq'
                if table_path.exists():
                    before_rows[(split, session.session_id, stem)] = pq.read_metadata(
                        table_path).num_rows
            for gid in session.groups:
                labels = session.labels(gid)
                if not has_labels(labels):
                    for stem in ('points3d', 'keypoints', 'instances', 'regions', 'extrinsics'):
                        table_path = session.path / f'{stem}.pq'
                        if table_path.exists():
                            group_ids = pq.read_table(table_path, columns=['group_id'])
                            n = group_ids.column('group_id').to_pylist().count(gid)
                            if n:
                                raise RuntimeError(
                                    f'{session.path}/{gid}: empty labels but {n} {stem} rows')
                    drops.append((split, session, gid))

    drop_keys = {(split, session.session_id, gid) for split, session, gid in drops}
    manifest_keys = set(zip(manifest['split'].to_list(), manifest['session'].to_list(),
                            manifest['child_group'].to_list()))
    if not drop_keys <= manifest_keys:
        raise RuntimeError('unlabeled groups are missing from cleaning_manifest.tsv')
    for split, session, gid in drops:
        mask = ((pl.col('split') == split) & (pl.col('session') == session.session_id) &
                (pl.col('child_group') == gid))
        matching = manifest.filter(mask)
        if len(matching) != 1:
            raise RuntimeError(f'{split}/{session.session_id}/{gid}: manifest row count != 1')
        if split != 'train' and matching['reason'][0] != 'dropped_no_labels':
            raise RuntimeError(f'{split}/{session.session_id}/{gid}: unexpected non-train drop')

    by_session: dict[Path, list[str]] = {}
    for _, session, gid in drops:
        by_session.setdefault(session.path, []).append(gid)
    for session_path, gids in by_session.items():
        session = next(s for s in ds.all_sessions() if s.path == session_path)
        remaining_groups = {k: v for k, v in session.groups.items() if k not in gids}
        if not remaining_groups:
            raise RuntimeError(f'{session.path}: pruning would leave no groups')
        labels = {k: session.labels(k) for k in remaining_groups}
        visibility_before = session.has_visibility_assessment
        fmt.write_session(
            session.path, mode=session.mode, units=session.units,
            label_source=session.label_source, names=session.names, rig=session.rig,
            groups=remaining_groups, labels=labels, skeleton=session.skeleton,
            flip_pairs=(session.flip_pairs if session.flip_pairs_declared else None),
            provenance=session.provenance,
            assoc_res_max_px=session.assoc_res_max_px,
        )
        reloaded = fmt.Session.load(session.path)
        if reloaded.has_visibility_assessment != visibility_before:
            raise RuntimeError(f'{session.path}: visibility assessment changed while pruning')
        groups_root = (session.path / 'groups').resolve()
        for gid in gids:
            group_dir = session.path / 'groups' / gid
            if group_dir.exists() or group_dir.is_symlink():
                if group_dir.parent.resolve() != groups_root:
                    raise RuntimeError(f'refusing unsafe group path: {group_dir}')
                if group_dir.is_symlink():
                    group_dir.unlink()
                else:
                    shutil.rmtree(group_dir)

    for split, session, gid in drops:
        mask = ((pl.col('split') == split) & (pl.col('session') == session.session_id) &
                (pl.col('child_group') == gid))
        manifest = manifest.with_columns(
            pl.when(mask).then(pl.lit('dropped_no_labels')).otherwise(pl.col('reason'))
              .alias('reason'),
            pl.when(mask).then(pl.col('child_n_frames')).otherwise(pl.col('dropped_frames'))
              .alias('dropped_frames'),
        )
    write_manifest(manifest, manifest_path)

    after_ds = fmt.load_dataset(root)
    after_rows: dict[tuple[str, str, str], int] = {}
    phantom = []
    for split in fmt.SPLITS:
        for session in after_ds.sessions.get(split, []):
            for gid in session.groups:
                if not (session.path / 'groups' / gid).exists():
                    phantom.append(f'{split}/{session.session_id}/{gid}')
            for stem in ('points3d', 'keypoints', 'instances', 'regions', 'extrinsics'):
                table_path = session.path / f'{stem}.pq'
                if table_path.exists():
                    after_rows[(split, session.session_id, stem)] = pq.read_metadata(
                        table_path).num_rows
    if phantom:
        raise RuntimeError(f'groups.pq has missing pixel directories: {phantom[:5]}')
    if before_rows != after_rows:
        raise RuntimeError(f'label table row counts changed: before={before_rows}, after={after_rows}')
    kept_manifest = manifest.filter(pl.col('reason').is_null() |
                                    (pl.col('reason') != 'dropped_no_labels'))
    if len(kept_manifest) != sum(len(s.groups) for s in after_ds.all_sessions()):
        raise RuntimeError('manifest kept-group count differs from groups.pq')

    all_dropped = manifest.filter(pl.col('reason') == 'dropped_no_labels')
    summary_path = root / 'cleaning_summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({
        'children': int(sum(len(s.groups) for s in after_ds.all_sessions())),
        'dropped_no_labels': int(len(all_dropped)),
        'dropped_frames': int(all_dropped['child_n_frames'].sum()),
        'dropped_no_labels_by_split': {
            row['split']: int(row['len'])
            for row in all_dropped.group_by('split').len().sort('split').iter_rows(named=True)},
    })
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')

    provenance_path = root / 'provenance.toml'
    provenance = toml.load(provenance_path) if provenance_path.exists() else {}
    repo = Path(__file__).resolve().parent.parent
    provenance['prune_runs'] = int(provenance.get('prune_runs', 0)) + 1
    provenance['prune'] = {
        'predicate': 'loader target visibility has no cell != UNLABELED',
        'dropped_this_run': len(drops),
        'dropped_total': len(all_dropped),
        'dropped_frames_total': int(all_dropped['child_n_frames'].sum()),
        'code_commit': subprocess_run(['git', 'rev-parse', 'HEAD'], cwd=repo),
        'code_dirty': bool(subprocess_run(['git', 'status', '--short'], cwd=repo)),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    provenance_path.write_text(toml.dumps(provenance))
    return {'dropped_this_run': len(drops), 'dropped_total': len(all_dropped),
            'dropped_frames_total': int(all_dropped.child_n_frames.sum())}


def subprocess_run(args: list[str], cwd: Path) -> str:
    """Run a small provenance command and return stripped stdout."""
    import subprocess
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def validate(root: Path, check_images: bool = False) -> None:
    """Validate a generated root and raise with the first violations."""
    ds = fmt.load_dataset(root)
    errors = fmt.validate_dataset(ds, check_images=check_images)
    if errors:
        raise RuntimeError('validation failed:\n' + '\n'.join(errors[:50]))
    print(f'validated {len(ds.all_sessions())} sessions', flush=True)


def main() -> None:
    """Build or validate the cleaned dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune-unlabeled', type=Path, metavar='ROOT')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--scores', type=Path, default=SCORES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--jump-threshold-mm', type=float, default=JUMP_THRESHOLD_MM)
    parser.add_argument('--score-threshold', type=float, default=SCORE_THRESHOLD)
    parser.add_argument('--score-window', type=int, default=SCORE_WINDOW)
    parser.add_argument('--scores-sha256')
    parser.add_argument('--checkpoint-iteration', type=int, default=CHECKPOINT_ITERATION)
    parser.add_argument('--checkpoint-sha256', default=CHECKPOINT_SHA256)
    parser.add_argument('--scorer-code-commit')
    parser.add_argument('--scorer-code-dirty', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-videos', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--validate-images', action='store_true')
    args = parser.parse_args()
    if args.prune_unlabeled is not None:
        if args.validate or args.overwrite or args.no_videos or args.validate_images:
            raise SystemExit('--prune-unlabeled cannot be combined with build/validate flags')
        print(prune_unlabeled(args.prune_unlabeled), flush=True)
        return
    if args.score_window < 1 or args.jump_threshold_mm <= 0:
        raise SystemExit('thresholds must be positive')
    if args.validate and not args.output.exists():
        raise SystemExit(f'missing output: {args.output}')
    if not args.validate:
        print(build(args.source, args.scores, args.output,
                    jump_threshold=args.jump_threshold_mm, score_threshold=args.score_threshold,
                    score_window=args.score_window, workers=args.workers,
                    overwrite=args.overwrite,
                    checkpoint_iteration=args.checkpoint_iteration,
                    checkpoint_sha256=args.checkpoint_sha256,
                    scores_sha256=args.scores_sha256,
                    scorer_code_commit=args.scorer_code_commit,
                    scorer_code_dirty=args.scorer_code_dirty,
                    write_videos=not args.no_videos), flush=True)
    if args.validate or not args.no_videos:
        validate(args.output, check_images=args.validate_images)


if __name__ == '__main__':
    main()
