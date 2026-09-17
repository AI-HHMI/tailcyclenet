#!/usr/bin/env python
"""Add fluorescence camera views to an existing cleaned QDMouse dataset."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tailcyclenet import format as fmt
from scripts.clean_qdmouse4m import MANIFEST_COLUMNS, _cut_worker, read_manifest
from scripts.convert_qdmouse4m_fluo import camera_index_map

CLEAN = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
             'tailcycle-datasets/qdmouse4m-cleaned')
FLUO = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
            'tailcycle-datasets/qdmouse4m-fluo-separate-cameras')
OUTPUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/'
              'tailcycle-datasets/qdmouse4m-fluo-cleaned')



def copy_group(group: fmt.Group) -> fmt.Group:
    """Copy group metadata without retaining the source session object."""
    return fmt.Group(
        group_id=group.group_id, n_frames=group.n_frames, fps=group.fps,
        source_video=group.source_video, source_frame_start=group.source_frame_start,
        source_frame_step=group.source_frame_step, notes=group.notes,
    )



def map_regions(regions, take: list[int]) -> np.ndarray | None:
    """Duplicate region rows onto the destination camera axis by source camera index."""
    if regions is None:
        return None
    base = np.asarray(regions, dtype=np.float64)
    rows = []
    for dst, src in enumerate(take):
        selected = base[:, 1] == src
        if selected.any():
            block = base[selected].copy()
            block[:, 1] = dst
            rows.append(block)
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 6), dtype=np.float64)



def map_labels(labels: fmt.Labels, take: list[int]) -> fmt.Labels:
    """Widen per-camera arrays; 3D arrays and camera-independent fields are unchanged."""
    return fmt.Labels(
        animal_ids=list(labels.animal_ids),
        points3d=None if labels.points3d is None else np.array(labels.points3d, copy=True),
        vis3d=None if labels.vis3d is None else np.array(labels.vis3d, copy=True),
        points2d=None if labels.points2d is None else np.take(labels.points2d, take, axis=3),
        vis2d=None if labels.vis2d is None else np.take(labels.vis2d, take, axis=3),
        boxes=None if labels.boxes is None else np.take(labels.boxes, take, axis=2),
        instance=None if labels.instance is None else np.take(labels.instance, take, axis=2),
        ext=None if labels.ext is None else np.take(labels.ext, take, axis=0),
        regions=map_regions(labels.regions, take),
    )



def assert_calibration_matches(clean: fmt.Session, fluo: fmt.Session) -> None:
    """Ensure each reference camera retained its original calibration by name."""
    expected = [name for camera in clean.cam_names for name in (camera, f'{camera}_fluo')]
    if fluo.cam_names != expected:
        raise RuntimeError(f'{clean.path}: fluo camera order changed: {fluo.cam_names} vs {expected}')

    def arr(value):
        """Convert an aniposelib parameter to a numeric array."""
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return np.asarray(value, dtype=np.float64)

    for name in clean.cam_names:
        if clean.rig.size(name) != fluo.rig.size(name):
            raise RuntimeError(f'{clean.path}: camera size changed for {name}')
        if clean.rig.offset[name] != fluo.rig.offset[name]:
            raise RuntimeError(f'{clean.path}: camera offset changed for {name}')
        if clean.rig.moving[name] != fluo.rig.moving[name]:
            raise RuntimeError(f'{clean.path}: moving flag changed for {name}')
        if clean.rig.calibrated[name] != fluo.rig.calibrated[name]:
            raise RuntimeError(f'{clean.path}: calibrated flag changed for {name}')
        clean_camera = clean.rig.by_name(name)
        fluo_camera = fluo.rig.by_name(name)
        for getter in ('get_camera_matrix', 'get_distortions', 'get_rotation', 'get_translation'):
            if not np.allclose(arr(getattr(clean_camera, getter)()),
                               arr(getattr(fluo_camera, getter)()), atol=1e-6):
                raise RuntimeError(f'{clean.path}: {getter} differs for {name}')



def add_cameras(clean_root: Path, fluo_root: Path, output: Path, workers: int,
                overwrite: bool = False) -> dict[str, int]:
    """Build a 12-camera sibling from cleaned labels and the validated fluo videos."""
    clean_root, fluo_root, output = (Path(p).resolve() for p in (clean_root, fluo_root, output))
    if output.exists() and not overwrite:
        raise RuntimeError(f'{output} exists; pass --overwrite to replace it')
    clean_ds = fmt.load_dataset(clean_root)
    fluo_ds = fmt.load_dataset(fluo_root)
    stage = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    manifest_path = clean_root / 'cleaning_manifest.tsv'
    manifest = read_manifest(manifest_path)
    if manifest.columns != MANIFEST_COLUMNS:
        raise RuntimeError(f'{manifest_path}: unexpected cleaning manifest columns')
    kept = manifest.filter(pl.col('reason').is_null() |
                           (pl.col('reason') != 'dropped_no_labels'))
    if len(kept) != sum(len(s.groups) for s in clean_ds.all_sessions()):
        raise RuntimeError('cleaning manifest does not match cleaned groups')
    tasks = []
    counts = {'sessions': 0, 'groups': 0, 'symlinks': 0, 'cut_clips': 0}
    try:
        for split in fmt.SPLITS:
            clean_sessions = {s.session_id: s for s in clean_ds.sessions.get(split, [])}
            fluo_sessions = {s.session_id: s for s in fluo_ds.sessions.get(split, [])}
            if clean_sessions.keys() != fluo_sessions.keys():
                raise RuntimeError(f'{split}: cleaned/fluo session sets differ')
            for session_id, clean in clean_sessions.items():
                fluo = fluo_sessions[session_id]
                assert_calibration_matches(clean, fluo)
                rows = kept.filter((pl.col('split') == split) &
                                   (pl.col('session') == session_id))
                if set(rows['child_group'].to_list()) != set(clean.groups):
                    raise RuntimeError(f'{clean.path}: manifest children != groups.pq')
                missing = set(rows['parent_group'].cast(pl.String).to_list()) - set(fluo.groups)
                if missing:
                    raise RuntimeError(f'{clean.path}: fluo root lacks parent groups {sorted(missing)}')
                for row in rows.iter_rows(named=True):
                    parent = fluo.groups[str(row['parent_group'])]
                    child = clean.groups[row['child_group']]
                    if (int(row['child_local_end']) > parent.n_frames or
                            abs(float(parent.fps) - float(child.fps)) > 1e-6):
                        raise RuntimeError(f"{clean.path}/{row['child_group']}: child span or fps "
                                           f"disagrees with fluo parent {row['parent_group']}")
                if rows['child_group'].is_duplicated().any():
                    raise RuntimeError(f'{clean.path}: manifest has duplicate child groups')
                rows_by_child = {row['child_group']: row for row in rows.iter_rows(named=True)}
                take = camera_index_map(clean.cam_names, fluo.cam_names)
                out_session = stage / split / session_id
                out_session.mkdir(parents=True, exist_ok=True)
                groups = {gid: copy_group(group) for gid, group in clean.groups.items()}
                labels = {gid: map_labels(clean.labels(gid), take) for gid in clean.groups}
                visibility_before = clean.has_visibility_assessment
                provenance = dict(clean.provenance)
                provenance.update({
                    'source': str(clean.path),
                    'camera_modalities': {
                        'reference': list(clean.cam_names),
                        'fluorescence': [f'{name}_fluo' for name in clean.cam_names],
                    },
                    'fluorescence_calibration': 'copied associated reference camera geometry',
                    'camera_label_source': {'dst': list(fluo.cam_names), 'src_index': take},
                })
                fmt.write_session(
                    out_session, mode=clean.mode, units=clean.units,
                    label_source=clean.label_source, names=clean.names, rig=fluo.rig,
                    groups=groups, labels=labels, skeleton=clean.skeleton,
                    flip_pairs=(clean.flip_pairs if clean.flip_pairs_declared else None),
                    provenance=provenance,
                    assoc_res_max_px=clean.assoc_res_max_px,
                )
                reloaded = fmt.Session.load(out_session)
                if reloaded.has_visibility_assessment != visibility_before:
                    raise RuntimeError(f'{clean.path}: visibility assessment changed')
                for gid, group in clean.groups.items():
                    if gid not in rows_by_child:
                        raise RuntimeError(f'{split}/{session_id}/{gid}: manifest row missing')
                    row = rows_by_child[gid]
                    child_dir = out_session / 'groups' / gid
                    child_dir.mkdir(parents=True, exist_ok=True)
                    for camera in clean.cam_names:
                        src = (clean.path / 'groups' / gid / f'{camera}.mp4').resolve()
                        fmt.link(child_dir / f'{camera}.mp4', src)
                        counts['symlinks'] += 1
                    for camera in clean.cam_names:
                        fluo_name = f'{camera}_fluo'
                        src_group = fluo.groups[str(row['parent_group'])]
                        src = src_group.dir / f'{fluo_name}.mp4'
                        if row['reason'] == 'unsplit':
                            fmt.link(child_dir / f'{fluo_name}.mp4', src.resolve())
                            counts['symlinks'] += 1
                        else:
                            if not src.exists():
                                raise RuntimeError(f"{fluo.path}/{row['parent_group']}: missing {src}")
                            target = child_dir / f'{fluo_name}.mp4'
                            tasks.append((str(src.resolve()), str(target),
                                          int(row['child_local_start']), int(row['child_local_end']),
                                          float(group.fps)))
                counts['groups'] += len(groups)
                counts['sessions'] += 1
        if tasks:
            from multiprocessing import get_context
            with get_context('spawn').Pool(max(1, workers)) as pool:
                for _ in pool.imap_unordered(_cut_worker, tasks):
                    counts['cut_clips'] += 1
        shutil.copy2(clean_root / 'cleaning_manifest.tsv', stage / 'cleaning_manifest.tsv')
        shutil.copy2(clean_root / 'cleaning_summary.json', stage / 'cleaning_summary.json')
        import toml
        provenance = toml.load(clean_root / 'provenance.toml')
        provenance['fluo_cameras'] = {
            'cleaned_source_root': str(clean_root),
            'fluo_source_root': str(fluo_root),
            'mapping': 'by camera name; fluorescence labels are paired reference labels',
            'calibration': 'copied from qdmouse4m-fluo-separate-cameras',
            'created_utc': datetime.now(timezone.utc).isoformat(),
        }
        (stage / 'provenance.toml').write_text(toml.dumps(provenance))
        if output.exists():
            if not overwrite:
                raise RuntimeError(f'{output} appeared during build')
            shutil.rmtree(output)
        stage.rename(output)
        return counts
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise



def main() -> None:
    """Add six fluorescence views to the existing cleaned dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=Path, default=CLEAN)
    parser.add_argument('--fluo', type=Path, default=FLUO)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(add_cameras(args.clean, args.fluo, args.output, args.workers, args.overwrite), flush=True)


if __name__ == '__main__':
    main()
