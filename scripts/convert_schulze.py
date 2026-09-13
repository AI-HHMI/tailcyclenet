#!/usr/bin/env python
"""Convert Schulze SLEAP human-label exports to tailcycle 2D datasets.

Labels come from ``ground_truth_points.csv``; SLEAP and machine-annotated artifacts are not read.
Each group contains consecutive stored-video frames around nearby anchors, with labels only at
those anchors. Groups never cross discontinuities detected from the source predictions, and short
runs produce proportionally shorter groups. Groups share a resumable lossless PNG cache through
relative symlinks.

No ``regions.pq`` is written. Complete anchors receive derived ``instances.pq`` boxes, while
partial six-fish anchors retain keypoints but no instance row. Finite placed occlusions are
``projected``; explicit coordinate-free rows are ``missing``. Six-fish identities are inferred
only within each group and recorded with match diagnostics in ``identity_map.pq``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from aniposelib.cameras import CameraGroup

from tailcyclenet import format as fmt
from tailcyclenet.crop import crop_box_for_points

SRC_ROOT = Path('/nrs/schulze/Users/Lisanne/Data/sleap_DC_tracking')
DATASETS = {
    'lili_1fish_260831': {
        'out_name': 'schulze-1fish',
        'video': 'rec_20260705-142756_lossless_30fps.mp4',
        'fps': 240.0,
        'n_animals': 1,
    },
    'lili_6fish_260831': {
        'out_name': 'schulze-6fish',
        'video': '260209_120Hz_1536px2_20cm_resocialization_all_12.avi',
        'fps': 120.0,
        'n_animals': 6,
    },
}
WINDOW = 31
HALF = WINDOW // 2
# Source file carrying per-frame model positions, read ONLY to segment the timeline.
PREDICTIONS = 'MACHINE_ANNOTATED_predictions.slp'
# A fish cannot move this many body lengths between two CONSECUTIVE stored frames. Chosen at the
# CENTRE of the observed gap between real motion and discontinuity, not at a round number: over the
# 2,399 consecutive pairs of `lili_1fish_260831` the largest real frame-to-frame motion is 7.9 px
# and the smallest discontinuity is 26.0 px, so any threshold in 0.05-0.15 body lengths yields the
# IDENTICAL break set. Above that band discontinuities start being missed (0.20 loses 2, 0.25 loses
# 3, 0.50 loses 11); below it nothing changes.
RUN_JUMP_BODY = 0.10


def canonical_node(name: str) -> str:
    """Use one cross-dataset name for the source's ``hd``/``hc`` head-center typo."""
    return 'hc' if name == 'hd' else name


def read_labels(src: Path) -> tuple[list[str], dict[int, list[dict]], dict[int, int]]:
    """Read and strictly validate the CSV export."""
    rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    names: list[str] = []
    seen_names: set[str] = set()
    path = src / 'ground_truth_points.csv'
    with path.open(newline='') as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            try:
                frame = int(row['frame_idx'])
                instance = int(row['instance_idx'])
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f'{path}:{line}: invalid frame/instance') from exc
            node = canonical_node(row.get('node', ''))
            if not node:
                raise RuntimeError(f'{path}:{line}: empty node name')
            if node not in seen_names:
                seen_names.add(node)
                names.append(node)
            vis = row.get('visible', '')
            if vis not in {'0', '1'}:
                raise RuntimeError(f'{path}:{line}: visible must be 0 or 1, got {vis!r}')
            sx, sy = row.get('x', '').strip(), row.get('y', '').strip()
            if (not sx) != (not sy):
                raise RuntimeError(f'{path}:{line}: only one coordinate is present')
            if sx:
                try:
                    x, y = float(sx), float(sy)
                except ValueError as exc:
                    raise RuntimeError(f'{path}:{line}: invalid coordinate') from exc
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise RuntimeError(f'{path}:{line}: non-finite coordinate')
                if vis != '1':
                    raise RuntimeError(f'{path}:{line}: finite coordinate with visible=0')
            elif vis != '0':
                raise RuntimeError(f'{path}:{line}: missing coordinate with visible=1')
            rows_by_frame[frame].append({
                'instance': instance, 'node': node, 'x': float(sx) if sx else None,
                'y': float(sy) if sy else None, 'visible': vis,
            })

    if not rows_by_frame:
        raise RuntimeError(f'{path}: no labels')
    count_by_frame = {
        frame: len({int(r['instance']) for r in rows}) for frame, rows in rows_by_frame.items()}
    K = len(names)
    for frame, rows in rows_by_frame.items():
        keys = [(int(r['instance']), r['node']) for r in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f'{path}: duplicate (instance,node) in frame {frame}')
        for instance in {int(r['instance']) for r in rows}:
            n = sum(int(r['instance']) == instance for r in rows)
            if n != K:
                raise RuntimeError(f'{path}: frame {frame}, instance {instance}: '
                                   f'{n} nodes, expected {K}')
    return names, rows_by_frame, count_by_frame


def read_manifest(src: Path, frames: set[int], expected: int) -> dict[int, int]:
    """Cross-check CSV frames against the package's auditable frame manifest."""
    path = src / 'frame_manifest.csv'
    got: dict[int, int] = {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            got[int(row['frame_idx'])] = int(row['n_user_instances'])
    if frames != {f for f, n in got.items() if n > 0}:
        raise RuntimeError(f'{path}: user-labeled frame set disagrees with ground_truth_points.csv')
    for frame in frames:
        if got[frame] != expected and expected == 1:
            raise RuntimeError(f'{path}: frame {frame} says {got[frame]} user instances, '
                               'expected one for the 1-fish package')
    return got


def load_skeleton(src: Path, names: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """Load the source's name-preserving edges and symmetry pairs."""
    import json
    doc = json.loads((src / 'skeleton.json').read_text())
    source_names = [canonical_node(str(n)) for n in doc.get('nodes', [])]
    if source_names != names:
        raise RuntimeError(f'{src}/skeleton.json: node order disagrees with CSV')
    edges = [[canonical_node(str(a)), canonical_node(str(b))]
             for a, b in doc.get('edges', [])]
    flips = [[canonical_node(str(n)) for n in pair] for pair in doc.get('symmetries', [])]
    return edges, flips


def make_rig(width: int, height: int):
    """Make the nominal one-camera rig required for an uncalibrated 2D session."""
    cam = fmt.nominal_camera('cam0', (width, height))
    return fmt.Rig(cgroup=CameraGroup([cam]), offset={'cam0': (0.0, 0.0)},
                   moving={'cam0': False}, calibrated={'cam0': False})


def detect_runs(src: Path, n_video: int, extent_px: float | None) -> list[tuple[int, int]]:
    """Contiguous stored-frame runs of one recording, as inclusive ``(first, last)`` pairs.

    THE FIX for a teleporting fish. `lili_1fish_260831` is not temporally contiguous: the camera
    wrote 2,400 of 6,228 acquired frames, and in this copy the loss is not the six clean breakouts
    the source README describes. Measured on the source's own per-frame predictions, stored frames
    after ~1471 alternate between ~63-frame runs and isolated 1-2 frame runs, and the fish's
    apparent position jumps 100-305 px across each boundary against a median frame-to-frame motion
    of 0.6 px. A 31-frame window that straddles one of those boundaries shows the model a fish that
    teleports between consecutive frames -- physically impossible at 240 Hz, and the `--overlap`
    smoothness prior is then trained against a discontinuity.

    Detection is a threshold on the fish's own displacement between consecutive stored frames, in
    units of BODY LENGTH so it is independent of resolution: > `RUN_JUMP_BODY` body lengths between
    consecutive stored frames is not motion. The positions come from the source's
    ``MACHINE_ANNOTATED_predictions.slp`` when it ships one, and that is a DELIBERATE, NARROW use:
    only the gross temporal discontinuity is read from it, never a coordinate and never an
    identity. It is the only signal in the package that covers all 2,400 frames -- the human labels
    are too sparse to bracket every boundary (the boundary this fix was written for, 2369 -> 2370,
    sits in the unlabelled context of its own group, so no pair of anchors brackets it).

    A recording with no such file, or one whose frames are contiguous, yields a single run and every
    downstream group keeps the full 31-frame window -- so this is a no-op on `lili_6fish_260831`,
    whose README documents no dropouts.
    """
    import h5py

    path = src / PREDICTIONS
    if not path.is_file() or not extent_px:
        return [(0, n_video - 1)]
    with h5py.File(path, 'r') as f:
        frames, instances, points = f['frames'][:], f['instances'][:], f['pred_points'][:]
    centre = np.full((n_video, 2), np.nan)
    for row in frames:
        index = int(row['frame_idx'])
        if not 0 <= index < n_video:
            continue
        sel = instances[(instances['instance_id'] >= row['instance_id_start']) &
                        (instances['instance_id'] < row['instance_id_end'])]
        if not len(sel):
            continue
        xy = points[int(sel[0]['point_id_start']):int(sel[0]['point_id_end'])]
        ok = np.isfinite(xy['x']) & np.isfinite(xy['y'])
        if ok.any():
            centre[index] = (xy['x'][ok].mean(), xy['y'][ok].mean())
    jump = np.linalg.norm(np.diff(centre, axis=0), axis=1)
    breaks = np.flatnonzero(jump > RUN_JUMP_BODY * extent_px)
    starts = [0] + [int(b) + 1 for b in breaks]
    ends = [int(b) for b in breaks] + [n_video - 1]
    return [(s, e) for s, e in zip(starts, ends) if e >= s]


def label_extent_px(rows_by_frame: dict[int, list[dict]]) -> float:
    """Mean longest side of the labelled fish, in px: the body length the jump test is scaled by."""
    spans = []
    for rows in rows_by_frame.values():
        xy = np.array([(r['x'], r['y']) for r in rows if r['x'] is not None])
        if len(xy):
            spans.append(float(np.ptp(xy, axis=0).max()))
    return float(np.mean(spans)) if spans else 0.0


def anchor_groups(frames: list[int]) -> list[list[int]]:
    """Greedily combine sorted anchors while each group's span remains below 30 frames.

    Callers pass the anchors of ONE run: a group must never span a temporal discontinuity, so the
    run is the outer boundary and this is only the inner packing rule.
    """
    groups: list[list[int]] = []
    for frame in frames:
        if not groups or frame - groups[-1][0] >= 30:
            groups.append([frame])
        else:
            groups[-1].append(frame)
    return groups


def window_span(anchors: list[int], run: tuple[int, int]) -> tuple[int, int]:
    """The ``(start, n_frames)`` of a window holding `anchors` INSIDE one run.

    The window is `WINDOW` frames wherever the run allows it, centred on the anchors and clamped to
    the run -- never to the recording, which is what let it straddle a discontinuity before. A run
    shorter than `WINDOW` yields a group the length of that run, which the format permits
    (`n_frames` is per group) and the train loader handles (T is derived, floor 2).
    """
    first, last = run
    length = min(WINDOW, last - first + 1)
    if length < 2:
        raise RuntimeError(f'run {run} is too short to hold a window')
    center = (anchors[0] + anchors[-1]) // 2
    start = min(max(center - HALF, first), last - length + 1)
    return start, length


def infer_identity_map(anchors: list[int], rows_by_frame: dict[int, list[dict]], group_id: str,
                       expected: int) -> tuple[dict[tuple[int, int], str], list[dict]]:
    """Associate ONE group's own anchor frames into group-local six-fish IDs.

    The source's ``instance_idx`` is frame-local and carries no identity. Identity is therefore
    inferred, and it is inferred ONLY across the anchors of a single group -- a span of at most
    :data:`WINDOW` - 1 frames by construction -- so there is no long-gap regime in which stale
    positions can be matched to a moved animal. Nothing links two groups: the same
    ``g..._animal00`` label in two groups is two different fish, and no consumer may read it as a
    trajectory.

    The assignment is a complete one-to-one Hungarian between the previous anchor's known
    positions and this anchor's detected instances, so an anchor with `expected` instances is
    given exactly the `expected` row slots in the same order as its predecessor. A partial anchor
    (the source has some) matches only as many slots as it has instances, which is the honest
    reading: two labelled fish among six cannot say which four the others are.

    `match_distance_px` and `competitor_margin_px` are recorded per row so a consumer can
    quarantine ambiguous anchors rather than trusting every inferred label equally. The first
    anchor establishes deterministic position-ordered slots; unavailable prior positions are
    excluded from Hungarian matches, and unclaimed instances receive new slots rather than being
    dropped.
    """
    def centres(frame: int) -> dict[int, np.ndarray]:
        """Return mean finite coordinates grouped by source instance for one frame."""
        by_instance: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in rows_by_frame.get(frame, []):
            if row['x'] is not None:
                by_instance[int(row['instance'])].append((row['x'], row['y']))
        return {instance: np.mean(points, axis=0) for instance, points in by_instance.items()}

    mapping: dict[tuple[int, int], str] = {}
    records: list[dict] = []
    slots: list[str] = []
    last: dict[str, tuple[int, np.ndarray]] = {}
    for frame in anchors:
        current = centres(frame)
        instances = sorted(current)
        if not instances:
            continue
        if not slots:
            ordered = sorted(instances, key=lambda i: tuple(current[i]))
            slots = [f'{group_id}_animal{i:02d}' for i in range(len(ordered))]
            if len(slots) > expected:
                raise RuntimeError(f'{group_id}): frame {frame}: {len(slots)} instances > {expected}')
            assignments = [(instance, slots[k], None, None) for k, instance in enumerate(ordered)]
        else:
            track_ids = [s for s in slots]
            track_pos = np.stack([last[s][1] if s in last else np.full(2, np.nan)
                                  for s in track_ids])
            inst_pos = np.stack([current[i] for i in instances])
            cost = np.linalg.norm(track_pos[:, None] - inst_pos[None, :], axis=2)
            blocked = ~np.isfinite(cost)
            cost = np.where(blocked, 1e12, cost)
            rr, cc = linear_sum_assignment(cost)
            assignments = []
            claimed: set[int] = set()
            for r, c in zip(rr, cc):
                if blocked[r, c]:
                    continue
                others = np.delete(cost[:, c], r)
                margin = (float(others.min()) - float(cost[r, c])) if others.size else None
                assignments.append((instances[c], track_ids[r], float(cost[r, c]), margin))
                claimed.add(instances[c])
            for instance in instances:
                if instance in claimed:
                    continue
                slot = f'{group_id}_animal{len(slots):02d}'
                slots.append(slot)
                assignments.append((instance, slot, None, None))
        for instance, slot, distance, margin in assignments:
            mapping[(frame, instance)] = slot
            records.append({'group_id': group_id, 'source_frame': frame,
                            'source_instance_idx': instance, 'animal_id': slot,
                            'match_distance_px': distance, 'competitor_margin_px': margin,
                            'previous_frame': last[slot][0] if slot in last else None})
            last[slot] = (frame, current[instance])
    return mapping, records


def build_labels(anchors: list[int], rows_by_frame: dict[int, list[dict]], names: list[str],
                 expected: int, start: int, width: int, height: int, group_id: str,
                 identity_map: dict[tuple[int, int], str] | None = None,
                 n_frames: int = WINDOW, box_pad: int = 2) -> tuple[fmt.Labels, int]:
    """Build one group's labels over `n_frames` stored frames, at its anchors only.

    `regions.pq` is deliberately NOT emitted. Its absence is the format's claim of exhaustive
    labelling (spec S9b), and `instances.pq` now carries the machinery that makes that claim
    safe: a `labeled` row is written ONLY for an anchor frame where every expected animal is
    present and labelled. An anchor that labelled 1/2/4 of 6 fish gets NO rows at all.

    That one omission does two things, and it is why the fix is an omitted row rather than
    dropped keypoints:

    * DETECTOR training drops the frame. `BoxDataset._has_target` requires a `labeled` row with a
      finite box under `box_source='instances'`, and `lab.instance` is present-but-all-`INST_NONE`
      here, so a partial anchor never becomes pure background -- its 4 unlabelled fish are no
      longer taught as negatives.
    * POSE training is untouched. `crop._crop_source` falls back PER CAMERA to the keypoint crop
      rule when the stored box is all-NaN, so a partial anchor keeps every one of its keypoints
      and is cropped exactly as it was before `instances.pq` existed.

    Boxes are the crop rule's own PADDED extent (the configured `box_pad`, 2 by default), stored
    so that reading them back with pad 0 reproduces the identical box -- the convention
    `scripts/convert_apt_lbl.py` uses. They
    are DERIVED from the labelled keypoints, not an independent human box, which spec S9 permits
    (a box need not match any particular crop rule) and which is the only honest option here: the
    source DISCARDED its predictions (`EXPORT_REPORT.txt`) and ships no coordinates for an
    unlabelled animal.
    """
    all_instances = sorted({
        int(row['instance']) for frame in anchors for row in rows_by_frame[frame]})
    if expected == 1:
        animal_ids = ['fish0']
    else:
        if identity_map is None:
            raise RuntimeError('six-fish labels require a per-group identity map')
        animal_ids = sorted({identity_map[(frame, int(row['instance']))]
                             for frame in anchors for row in rows_by_frame[frame]},
                            key=lambda slot: int(slot.rsplit('animal', 1)[1]))
    if expected == 1 and all_instances != [0]:
        raise RuntimeError(f'one-fish source has unexpected instance ids {all_instances}')
    ai = {aid: i for i, aid in enumerate(animal_ids)}
    ki = {name: i for i, name in enumerate(names)}
    S, K = len(animal_ids), len(names)
    points = np.full((S, n_frames, K, 1, 2), np.nan, dtype=np.float32)
    status = np.full((S, n_frames, K, 1), fmt.UNLABELED, dtype=np.int8)
    boxes = np.full((S, n_frames, 1, 4), np.nan, dtype=np.float32)
    instance = np.full((S, n_frames, 1), fmt.INST_NONE, dtype=np.int8)
    size_wh = torch.as_tensor((float(width), float(height)))
    complete = 0
    for frame in anchors:
        local = frame - start
        instance_ids = sorted({int(row['instance']) for row in rows_by_frame[frame]})
        if len(instance_ids) > expected:
            raise RuntimeError(f'frame {frame}: {len(instance_ids)} instances, expected at most {expected}')
        full_frame = len(instance_ids) == expected
        if full_frame:
            complete += 1
        for row in rows_by_frame[frame]:
            instance_id = int(row['instance'])
            aid = ('fish0' if expected == 1 else
                   identity_map[(frame, instance_id)])
            a, k = ai[aid], ki[row['node']]
            if row['x'] is None:
                status[a, local, k, 0] = fmt.MISSING
            else:
                status[a, local, k, 0] = fmt.PROJECTED
                points[a, local, k, 0] = (row['x'], row['y'])
        if not full_frame:
            continue
        by_animal: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in rows_by_frame[frame]:
            if row['x'] is not None:
                aid = ('fish0' if expected == 1 else
                       identity_map[(frame, int(row['instance']))])
                by_animal[ai[aid]].append((row['x'], row['y']))
        for a in range(S):
            pts = by_animal.get(a)
            box = (None if not pts else
                   crop_box_for_points(torch.as_tensor(pts, dtype=torch.float32), size_wh, 64, box_pad))
            if box is None:
                continue
            instance[a, local, 0] = fmt.INST_LABELED
            boxes[a, local, 0] = box.numpy()
    return fmt.Labels(animal_ids=animal_ids, points3d=None, vis3d=None,
                      points2d=points, vis2d=status, boxes=boxes, instance=instance,
                      regions=None), complete


def _valid_png(path: Path, width: int, height: int) -> bool:
    """Fully decode an existing cache entry and check its dimensions and pixel mode."""
    try:
        with Image.open(path) as image:
            if image.size != (width, height) or image.mode not in {'L', 'RGB'}:
                return False
            image.load()
        return True
    except (OSError, ValueError):
        return False


def validate_cache(cache: Path, indices: set[int], width: int, height: int) -> set[int]:
    """Validate unique cache files in bounded parallel workers before resuming."""
    paths = sorted((i, cache / f'{i:06d}.png') for i in indices)
    valid: set[int] = set()
    with ThreadPoolExecutor(max_workers=4) as workers, tqdm(
            total=len(paths), desc=f'check {cache.parent.name}', unit='PNG',
            dynamic_ncols=True) as progress:
        for index, ok in zip((i for i, _ in paths), workers.map(
                lambda pair: _valid_png(pair[1], width, height), paths)):
            progress.update(1)
            if ok:
                valid.add(index)
    return valid


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Encode one decoded frame atomically; called by a bounded writer pool."""
    if np.array_equal(rgb[..., 0], rgb[..., 1]) \
            and np.array_equal(rgb[..., 0], rgb[..., 2]):
        image = Image.fromarray(rgb[..., 0], mode='L')
    else:
        image = Image.fromarray(rgb, mode='RGB')
    tmp = path.parent / f'.{path.name}.tmp'
    image.save(tmp, format='PNG', compress_level=1)
    os.replace(tmp, path)


def extract_cache(video: Path, indices: set[int], cache: Path,
                  width: int, height: int) -> tuple[int, int]:
    """Decode once, reusing valid cache entries and writing each new frame atomically.

    The decoded RGB array is converted once. Equal RGB channels are stored as one-channel PNG;
    the tailcycle image reader expands it back to identical RGB channels. No immediate reopen is
    needed here because final validation checks representative source/output pixels and every
    cache entry's dimensions.
    """
    cache.mkdir(parents=True, exist_ok=True)
    existing = validate_cache(cache, indices, width, height)
    missing = indices - existing
    if not missing:
        return len(existing), 0
    max_index = max(indices)
    decoded = 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        got_width, got_height = int(stream.codec_context.width), int(stream.codec_context.height)
        if (got_width, got_height) != (width, height):
            raise RuntimeError('video dimensions changed between probes')
        total = min(int(stream.frames) or max_index + 1, max_index + 1)
        pending = []
        with ThreadPoolExecutor(max_workers=4) as writers, tqdm(
                total=total, desc=f'extract {video.name}', unit='frame',
                dynamic_ncols=True, initial=0) as progress:
            for index, frame in enumerate(container.decode(stream)):
                progress.update(1)
                decoded += 1
                if index in missing:
                    rgb = frame.to_ndarray(format='rgb24')
                    path = cache / f'{index:06d}.png'
                    pending.append((index, writers.submit(_write_png, path, rgb)))
                    if len(pending) >= 4:
                        done_index, future = pending.pop(0)
                        future.result()
                        existing.add(done_index)
                if index >= max_index:
                    break
            for done_index, future in pending:
                future.result()
                existing.add(done_index)
    missing = sorted(indices - existing)
    if missing:
        raise RuntimeError(f'{video}: requested frames did not decode: {missing[:8]}')
    return len(existing), decoded


def link_group_frames(session: Path, gid: str, start: int, n_frames: int, cache: Path) -> None:
    """Link local group frame names to the shared source-indexed cache."""
    local_dir = session / 'groups' / gid / 'cam0'
    local_dir.mkdir(parents=True, exist_ok=True)
    for local in range(n_frames):
        source = cache / f'{start + local:06d}.png'
        if not source.is_file():
            raise RuntimeError(f'{source}: cache frame missing')
        destination = local_dir / f'{local:06d}.png'
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(Path(os.path.relpath(source, local_dir)))


def convert_one(src: Path, out: Path, cfg: dict, clean: bool, resume: bool,
                box_pad: int = 2) -> None:
    """Convert one source package into one staged tailcycle dataset root.

    Anchors are packed independently inside contiguous runs; one-frame runs are recorded as
    excluded because the upstream loader cannot train on a one-frame window. Rebuilds quarantine
    stale group directories outside the dataset rather than deleting their withdrawn pixels.
    """
    if out.exists():
        if clean and not resume:
            shutil.rmtree(out)
        elif not resume:
            raise RuntimeError(f'{out} exists; pass --clean only for a staging directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    names, rows_by_frame, count_by_frame = read_labels(src)
    manifest = read_manifest(src, set(rows_by_frame), cfg['n_animals'])
    for frame, count in count_by_frame.items():
        if manifest[frame] != count:
            raise RuntimeError(f'{src}/frame_manifest.csv: frame {frame} says '
                               f'{manifest[frame]} user instances, CSV has {count}')
    edges, flips = load_skeleton(src, names)
    video = src / cfg['video']
    if not video.is_file():
        raise RuntimeError(f'{video}: source video is missing')
    with av.open(str(video)) as probe:
        stream = probe.streams.video[0]
        n_video = int(stream.frames)
        width, height = int(stream.codec_context.width), int(stream.codec_context.height)
    frames = sorted(rows_by_frame)
    runs = detect_runs(src, n_video, label_extent_px(rows_by_frame))

    def run_of(frame: int) -> tuple[int, int]:
        """Return the contiguous run containing a labelled source frame."""
        for run in runs:
            if run[0] <= frame <= run[1]:
                return run
        raise RuntimeError(f'{frame}: no run contains this labelled frame')

    windows: list[tuple[list[int], tuple[int, int], int, int]] = []
    dropped: list[int] = []
    for run in runs:
        in_run = [f for f in frames if run[0] <= f <= run[1]]
        if not in_run:
            continue
        if run[1] - run[0] + 1 < 2:
            dropped.extend(in_run)
            continue
        for cluster in anchor_groups(in_run):
            start, length = window_span(cluster, run)
            windows.append((cluster, run, start, length))
    required = {start + local for _, _, start, length in windows for local in range(length)}
    cache = out / '_frame_cache' / 'cam0'
    extracted, decoded = extract_cache(video, required, cache, width, height)

    session = out / 'train' / src.name
    groups: dict[str, fmt.Group] = {}
    labels: dict[str, fmt.Labels] = {}
    identity_records: list[dict] = []
    complete = 0
    for anchors, run, start, length in tqdm(windows, desc=f'link {src.name}', unit='group',
                                            dynamic_ncols=True):
        gid = f'g{anchors[0]:06d}_{anchors[-1]:06d}'
        if gid in groups:
            raise RuntimeError(f'duplicate generated group id {gid!r}')
        identity_map, records = (infer_identity_map(anchors, rows_by_frame, gid, cfg['n_animals'])
                                 if cfg['n_animals'] > 1 else (None, []))
        identity_records.extend(records)
        lab, n_complete = build_labels(anchors, rows_by_frame, names, cfg['n_animals'],
                                       start, width, height, gid, identity_map, length,
                                       box_pad=box_pad)
        groups[gid] = fmt.Group(
            gid, length, fps=cfg['fps'], source_video=str(video), source_frame_start=start,
            source_frame_step=1,
            notes=(f'{length} stored-video frames inside contiguous run {run[0]}-{run[1]}; '
                   'labels at nearby anchors only' +
                   ('; six-fish instance_idx is frame-local and has no identity ground truth'
                    if cfg['n_animals'] > 1 else '')))
        labels[gid] = lab
        complete += n_complete
        link_group_frames(session, gid, start, length, cache)

    rig = make_rig(width, height)
    session.mkdir(parents=True, exist_ok=True)
    stale = sorted(d.name for d in (session / 'groups').iterdir()
                   if d.is_dir() and d.name not in groups)
    if stale:
        quarantine = out.parent / f'.{out.name}.stale' / src.name
        for gid in stale:
            target = quarantine / gid
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(session / 'groups' / gid), str(target))
        print(f'{out.name}: quarantined {len(stale)} withdrawn group(s) -> {quarantine}')
    fmt.write_session(
        session, mode='2d', units='px', label_source='annotated', names=names, rig=rig,
        groups=groups, labels=labels, skeleton=edges, flip_pairs=flips,
        provenance={
            'source': str(src), 'source_csv': str(src / 'ground_truth_points.csv'),
            'source_slp': str(src / 'ground_truth_user_only.slp'),
            'source_readme': str(src / 'README.md'), 'annotator': 'Schulze Lab',
            'annotator_tool': 'SLEAP human labels', 'converter': 'scripts/convert_schulze.py',
            'coordinate_note': 'source coordinates are pixels; finite placed occluded points are projected',
            'keypoint_note': 'source hd (head centre) is canonicalized to hc to match the six-fish session',
            'window_note': 'windows stay inside one contiguous run; nearby anchors with span <30 combined',
            'runs_note': (f'{len(runs)} contiguous runs detected from the fish\'s own displacement '
                          f'between stored frames (> {RUN_JUMP_BODY:g} body lengths per frame, '
                          f'positions from {PREDICTIONS}); runs: {runs}'),
            'segmentation_note': ('run boundaries are HEURISTIC exclusions of a prediction-'
                                  'displacement outlier, not hardware timestamps: the package ships '
                                  'none. They mark where a window must not reach, which is all they '
                                  'are used for; nothing here asserts the acquisition timing'),
            'identity_note': ('six-fish source has no identity ground truth; animal IDs are '
                              'GROUP-LOCAL (a per-group one-to-one match over that group\'s own '
                              'anchors, span <= 31 frames). The same _animalNN in two groups is '
                              'two different fish and MUST NOT be read as a trajectory. '
                              'identity_map.pq carries the per-row match distance and '
                              'competitor margin.'
                              if cfg['n_animals'] > 1 else 'one fish; identity association is not applicable'),
            'regions_note': ('no regions.pq is written; per spec S9b its absence asserts exhaustive '
                             'labelling everywhere'),
            'instances_note': (f'instances.pq carries the crop rule\'s own padded extent '
                               f'(pad={box_pad} px, min_crop_dim=64) as a `labeled` row, '
                               'written ONLY on anchors where every expected animal is present '
                               'and labelled. Partial anchors have NO row, which drops them from '
                               'detector training while pose falls back per camera to the keypoint '
                               'crop rule and keeps every keypoint. Boxes are DERIVED from the '
                               'labelled keypoints, not annotated.'),
            'split_note': 'source supplied no train/val/test split; all labeled frames are under train',
            'timing_note': ('1-fish stored frames are NOT temporally contiguous; each window is confined '
                            'to one contiguous run so no window shows the fish teleporting across a '
                            'discontinuity'),
            'excluded': ('MACHINE_ANNOTATED_* files are machine output and were not used as labels; '
                         f'{PREDICTIONS} was read ONLY to locate temporal discontinuities, never for '
                         'a coordinate or an identity'),
        })
    if identity_records:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(identity_records), out / 'identity_map.pq')
    if dropped:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist([
            {'source_frame': int(f), 'reason': 'isolated in a 1-frame run; no window fits'}
            for f in sorted(dropped)]), out / 'excluded_anchors.pq')
    print(f'{out.name}: {len(groups)} groups (window lengths '
          f'{sorted({g.n_frames for g in groups.values()})}), {len(rows_by_frame)} labeled anchors, '
          f'{len(dropped)} dropped in degenerate runs, '
          f'{extracted} cached PNGs ({decoded} decoded this run), '
          f'{sum(len(r) for r in rows_by_frame.values())} point rows, '
          f'{complete} anchors carrying a full fish count, {width}x{height}, {cfg["fps"]:g} Hz')


def main() -> None:
    """Parse conversion options and convert each selected source dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, default=SRC_ROOT)
    parser.add_argument('--out-parent', type=Path,
                        default=Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets'))
    parser.add_argument('--only', choices=sorted(DATASETS), action='append')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='reuse valid cache PNGs and safely rebuild the session')
    parser.add_argument('--box-pad', type=int, default=2,
                        help='keypoint-extent padding in pixels for derived instances boxes '
                             '(published Schulze roots carry pad 2 as of 2026-09-12; 20 is the '
                             'pose crop rule\'s own pad (tailcyclenet/crop.py))')
    args = parser.parse_args()
    if args.box_pad < 0:
        parser.error('--box-pad must be non-negative')
    for source_name in args.only or sorted(DATASETS):
        cfg = DATASETS[source_name]
        convert_one(args.src / source_name, args.out_parent / cfg['out_name'], cfg,
                    args.clean, args.resume, box_pad=args.box_pad)


if __name__ == '__main__':
    main()
