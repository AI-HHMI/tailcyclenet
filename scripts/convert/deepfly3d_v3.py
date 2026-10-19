#!/usr/bin/env python3
"""Convert native ``deeperfly`` HDF5 results into tailcycle 3-D sessions.

The native result is kept as the authority for both the reconstructed points and the
bundle-adjusted camera rig.  Native coordinates have no metric scale, so each trial is
scaled using the mean length of the two front coxa segments (lf 0--1 and rf 19--20),
whose reference length is 0.456 mm.  This is a mechanical conversion, not a calibration
validation or an accuracy claim; the audit is recorded in the manifests and provenance.

Typical use::

    pixi run python scripts/convert/deepfly3d_v3.py inventory \\
        --source /path/ramdya-fly --outputs /path/deeperfly-results \\
        --report deepfly3d-v3-inventory.json
    pixi run python scripts/convert/deepfly3d_v3.py convert \\
        --source /path/ramdya-fly --outputs /path/deeperfly-results --out deepfly3d-v3

``--outputs`` may be a result root containing one ``results.h5`` below each trial, or
one HDF5 file when converting a single trial.  Source images are never copied.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tailcyclenet import format as fmt
from _common import link_numbered_frames, sha256

# The condition names and family split rule are shared conventions; v3 has its own HDF5 and
# pixel path handling.
from deepfly3d_common import SOURCE_CONDITIONS, family_id, split_map
from deepfly3d_h5 import (CAMERA_ORDERING, FRONT_COXA_PAIRS, N_CAMERAS, N_KEYPOINTS, NATIVE_NAMES,
                          REFERENCE_COXA_MM, NativeResult, camera_files, load_result,
                          scale_from_front_coxa, valid_2d)
from deepfly3d_rig import audit, calibration_fingerprint, calibration_rig, skeleton_for


POLICY_VERSION = 'deepfly3d-v3-native-h5-scale-front-coxa-v1'


@dataclass(frozen=True)
class Record:
    """One extracted source trial and its native result."""

    condition: str
    archive: str
    images: Path
    result: Path | None
    family: str
    reason: str | None = None

    @property
    def session_id(self) -> str:
        """Collision-proof target session id."""
        return f'{self.condition}__{self.archive}'


def _archive_for(record: Record) -> Path | None:
    """Find the source ZIP alongside an extracted trial, if one exists."""
    root = record.images.parent.parent
    candidates = [root / f'{record.archive}.zip', root.parent / f'{record.archive}.zip']
    return next((p for p in candidates if p.is_file()), None)


def _candidate_result_paths(outputs: Path) -> list[Path]:
    """List native result files below a result root, in deterministic order."""
    if outputs.is_file():
        return [outputs] if outputs.name == 'results.h5' else []
    if not outputs.is_dir():
        return []
    return sorted(p for p in outputs.rglob('results.h5') if p.is_file())


def _result_matches(result: Path, record: Record) -> bool:
    """Match an HDF5 result to a source trial by path, then native footage metadata."""
    names = {record.archive, record.archive.removesuffix('_behData_images'),
             record.archive.removesuffix('_images')}
    parts = set(result.parts)
    if result.parent.name in names and record.condition in parts:
        return True
    if result.parent.name in names:
        return True
    try:
        with h5py.File(result, 'r') as f:
            raw = f.get('pose2d')
            raw = raw.attrs.get('footage', '') if raw is not None else ''
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            return str(record.images) in str(raw) or record.archive in str(raw)
    except (OSError, KeyError, ValueError, TypeError, EOFError):
        return False


def _find_result(outputs: Path, record: Record, paths: list[Path]) -> tuple[Path | None, str | None]:
    """Resolve exactly one native result for a trial without quadratic HDF5 metadata reads."""
    if outputs.is_file() and len(paths) == 1:
        return paths[0], None
    names = {record.archive, record.archive.removesuffix('_behData_images'),
             record.archive.removesuffix('_images')}
    matched = [p for p in paths if p.parent.name in names and record.condition in p.parts]
    if not matched:
        matched = [p for p in paths if p.parent.name in names]
    if not matched:
        matched = [p for p in paths if _result_matches(p, record)]
    if not matched:
        return None, 'missing_result'
    if len(matched) != 1:
        return None, 'ambiguous_result:' + ','.join(str(p) for p in matched[:3])
    return matched[0], None


def discover(source: Path, outputs: Path) -> list[Record]:
    """Discover extracted source trials and associate native ``results.h5`` files."""
    source, outputs = Path(source), Path(outputs)
    if not source.is_dir():
        raise SystemExit(f'missing source directory: {source}')

    known_conditions = [c for c in SOURCE_CONDITIONS if (source / c / 'extracted').is_dir()]
    extra_conditions = sorted(p.name for p in source.iterdir()
                              if p.is_dir() and (p / 'extracted').is_dir()
                              and p.name not in SOURCE_CONDITIONS)
    conditions = known_conditions + extra_conditions
    if not conditions:
        raise SystemExit(f'{source}: no condition/extracted directories')
    candidates: list[tuple[str, str, Path, str]] = []
    for condition in conditions:
        extracted = source / condition / 'extracted'
        for d in sorted(extracted.iterdir()):
            if d.is_dir() and d.name.endswith('_behData_images'):
                try:
                    fam = family_id(condition, d.name)
                    reason = None
                except ValueError as exc:
                    fam = f'{condition}__invalid__{d.name}'
                    reason = 'invalid_family_id:' + str(exc)
                candidates.append((condition, d.name, d / 'images', fam, reason))
    paths = _candidate_result_paths(outputs)
    records: list[Record] = []
    direct = outputs.is_file()
    for condition, archive, images, fam, reason in candidates:
        if reason is None and not images.is_dir():
            reason = 'missing_images'
        result = None
        if reason is None:
            if direct:
                if len(candidates) != 1:
                    reason = 'direct_result_requires_one_source_trial'
                else:
                    result = paths[0] if paths else None
                    reason = None if result else 'missing_result'
            else:
                result, reason = _find_result(
                    outputs, Record(condition, archive, images, None, fam), paths)
        records.append(Record(condition, archive, images, result, fam, reason))
    return records


def _source_files_for(result: NativeResult, images: Path) -> list[list[Path]]:
    """Resolve source sequences and verify representative dimensions against native metadata.

    The extraction/worker checks already establish the complete contiguous 900-frame layout.
    Opening and decoding all 1.24 million JPEGs again on the shared filesystem made inventory
    needlessly take hours, so inspect the first and last frame of each camera here; the generic
    session validator repeats the first-frame calibration-size check after writing.
    """
    from PIL import Image
    out = []
    for c in range(N_CAMERAS):
        files = camera_files(images, c, result.n_frames)
        check_frames = sorted({0, len(files) - 1})
        for frame in check_frames:
            path = files[frame]
            try:
                with Image.open(path) as im:
                    got = tuple(int(v) for v in im.size)
                    im.verify()
            except (OSError, ValueError, SyntaxError) as exc:
                raise ValueError(f'camera {c} frame {frame} is not a readable image: {path}') from exc
            if got != result.image_sizes[c]:
                raise ValueError(f'camera {c} frame {frame} source image is {got}, '
                                 f'HDF5 metadata is {result.image_sizes[c]}')
        out.append(files)
    return out


def _append_scores(path: Path, scores: np.ndarray) -> None:
    """Add native detector confidence to keypoints.pq (the generic Labels API has no score)."""
    table = pq.read_table(path)
    vals = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(table) != len(vals):
        raise RuntimeError(f'{path}: score rows {len(vals)} != keypoint rows {len(table)}')
    table = table.append_column('score', pa.array(vals, type=pa.float32()))
    tmp = path.with_name('.' + path.name + '.score.tmp')
    pq.write_table(table, tmp, compression='zstd')
    os.replace(tmp, path)


def _fsync_tree(path: Path) -> None:
    """Best-effort fsync for a completed session directory before its rename."""
    for name in ('session.toml', 'calibration.toml', 'groups.pq', 'keypoints.pq',
                 'points3d.pq'):
        p = path / name
        if p.exists():
            with p.open('rb') as f:
                os.fsync(f.fileno())
    try:
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


def git_commit() -> str:
    """Return the converter checkout commit, or an explicit unknown marker."""
    try:
        return subprocess.run(('git', 'rev-parse', 'HEAD'), check=True, capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def _source_hashes(record: Record) -> dict[str, Any]:
    """Stable input identity used by idempotency checks without rereading 137 GiB of ZIPs."""
    archive = _archive_for(record)
    names = (sorted(p.name for p in record.images.iterdir() if p.name.startswith('camera_'))
             if record.images.is_dir() else [])
    return {
        'result_sha256': sha256(record.result) if record.result else None,
        'source_archive': str(archive) if archive else None,
        'source_archive_size': int(archive.stat().st_size) if archive else None,
        'source_images': str(record.images),
        'source_image_count': len(names),
        'source_image_names_sha256': hashlib.sha256('\\n'.join(names).encode()).hexdigest(),
    }


def convert_record(record: Record, split: str, root: Path, *, force: bool = False,
                   max_reprojection_px: float | None = None) -> dict:
    """Convert one source trial atomically and validate its resulting session."""
    if record.reason:
        return {'condition': record.condition, 'archive': record.archive, 'split': split,
                'status': 'quarantined', 'reason': record.reason}
    assert record.result is not None
    result = load_result(record.result, source_images=record.images)
    source_files = _source_files_for(result, record.images)
    observed = valid_2d(result)
    support = observed.sum(axis=0)
    scale, scale_meta = scale_from_front_coxa(result.points3d)
    audit_meta = audit(result, max_reprojection_px=max_reprojection_px)
    session_id = record.session_id
    session_dir = root / split / session_id
    if session_dir.exists():
        if not force:
            raise SystemExit(f'output exists (use --force to replace): {session_dir}')
        shutil.move(str(session_dir), str(root / f'.quarantine-{session_id}-{time.time_ns()}'))
    temp = root / split / f'.{session_id}.tmp'
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    farms: list[Path] = []
    try:
        names = result.names
        skeleton, flip_pairs = skeleton_for(result)

        points2d = np.transpose(result.points2d, (1, 2, 0, 3))[None].astype(np.float32)
        vis2d = np.full((1, result.n_frames, N_KEYPOINTS, N_CAMERAS), fmt.MISSING, np.int8)
        vis2d[0][np.transpose(observed, (1, 2, 0))] = fmt.VISIBLE
        points3d = (result.points3d * scale)[None].astype(np.float32)
        vis3d = np.full((1, result.n_frames, N_KEYPOINTS), fmt.UNLABELED, np.int8)
        valid3 = np.isfinite(result.points3d).all(axis=-1) & (support >= 2)
        vis3d[0][valid3] = fmt.PROJECTED

        points3d[0][~np.isfinite(result.points3d).all(axis=-1)] = np.nan
    
        group = fmt.Group(
            group_id='g000', n_frames=result.n_frames, source_video=f'{record.condition}/{record.archive}',
            source_frame_start=0, source_frame_step=1,
            notes='native deeperfly tracked 3D; scale and calibration are provisional_unvalidated',
        )
        labels = fmt.Labels(
            animal_ids=['fly0'], points3d=points3d, vis3d=vis3d,
            points2d=points2d, vis2d=vis2d, boxes=None, instance=None,
        )
        provenance = {
            'source': 'native deeperfly results.h5',
            'annotator': '',
            'annotator_tool': 'deeperfly native HDF5 result',
        'converter_git_commit': git_commit(),
            'created': date.today().isoformat(),
            'status': 'provisional_unvalidated',
            'condition': record.condition,
            'source_archive': record.archive,
            'source_images': str(record.images),
            'source_result': str(record.result),
            'source_result_sha256': sha256(record.result),
            'source_archive_size': (_archive_for(record).stat().st_size
                                     if _archive_for(record) else None),
            'native_camera_order': ','.join(result.camera_names),
        'calibration_fingerprint': calibration_fingerprint(result, scale),
            'source_camera_policy': 'HDF5 camera index c maps to source camera_c image sequence',
            'point3d_source': 'triangulation/points3d',
            'point2d_source': result.points2d_source + ' (stored x,y pixels)',
            'confidence_source': 'pose2d/conf; copied to keypoints.pq score without thresholding',
            'point2d_status_policy': 'visible iff selected native per-view point is finite; missing iff NaN; confidence never thresholded',
            'point3d_status_policy': 'projected iff finite and at least two finite 2D predictions; no independent 3D visibility assessment',
            'scale_policy': '0.456 mm divided by mean native lengths of front coxa segments 0-1 and 19-20',
            'scale_reference_mm': REFERENCE_COXA_MM,
            'scale_native_front_coxa_0_1': scale_meta['0-1'],
            'scale_native_front_coxa_19_20': scale_meta['19-20'],
            'scale_native_mean': scale_meta['combined'],
            'scale_to_mm': scale,
            'audit_status': 'provisional_unvalidated',
            'audit_min_2d_support': audit_meta['min_2d_support'],
            'audit_overall_median_reprojection_px': audit_meta['overall_median_reprojection_px'],
            'audit_overall_p95_reprojection_px': audit_meta['overall_p95_reprojection_px'],
            'audit_overall_max_reprojection_px': audit_meta['overall_max_reprojection_px'],
            'audit_positive_depth_fraction': audit_meta['overall_positive_depth_fraction'],
            'frame_policy': 'all native results.h5 frames; no implicit trim',
        }
        rig = calibration_rig(result, scale)
        fmt.write_session(temp, mode='3d', units='mm', label_source='tracked', names=names,
                          rig=rig, groups={'g000': group}, labels={'g000': labels},
                          skeleton=skeleton, flip_pairs=flip_pairs, provenance=provenance)

        _append_scores(temp / 'keypoints.pq', result.confidence.transpose(1, 2, 0).reshape(-1))
    
        for c, cname in enumerate(result.camera_names):
            farm = root / '.pixel_views' / f'{session_id}__{cname}'
            farms.append(farm)
            link_numbered_frames(farm, source_files[c])
            target = temp / 'groups' / 'g000' / cname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(farm.resolve())
        _fsync_tree(temp)
        loaded = fmt.Session.load(temp)
        errors = fmt.validate_session(loaded, check_images=True)
        if errors:
            raise RuntimeError('\n'.join(errors[:20]))
        temp.rename(session_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        for farm in farms:

            shutil.rmtree(farm, ignore_errors=True)
        raise
    return {
        'condition': record.condition, 'archive': record.archive, 'split': split,
        'session': str(session_dir.relative_to(root)), 'status': 'converted',
        'source_result': str(record.result), 'source_result_sha256': sha256(record.result),
        'source_archive_size': (_archive_for(record).stat().st_size
                                 if _archive_for(record) else None),
        'source_camera_names': result.camera_names,
        'source_frame_counts': [len(files) for files in source_files],
        'pixel_view_dirs': {name: str(root / '.pixel_views' / f'{session_id}__{name}')
                            for name in result.camera_names},
        'calibration_fingerprint': calibration_fingerprint(result, scale),
        'n_frames': result.n_frames,
        'scale_reference_mm': REFERENCE_COXA_MM,
        'scale_native_front_coxa_0_1': scale_meta['0-1'],
        'scale_native_front_coxa_19_20': scale_meta['19-20'],
        'scale_native_mean': scale_meta['combined'],
        'scale_to_mm': scale,
        'keypoint_rows': int(observed.sum()),
        'point3d_rows': int(valid3.sum()),
        'audit': audit_meta,
    }


def inspect_record(record: Record, *, max_reprojection_px: float | None = None) -> dict:
    """Inventory one record without writing a session."""
    out: dict[str, Any] = {
        'condition': record.condition, 'archive': record.archive,
        'family': record.family, 'images': str(record.images),
        'result': str(record.result) if record.result else None,
        'status': 'quarantined' if record.reason else 'ready',
        'reason': record.reason,
    }
    if record.reason:
        return out
    try:
        assert record.result is not None
        result = load_result(record.result, source_images=record.images)
        files = _source_files_for(result, record.images)
        scale, scale_meta = scale_from_front_coxa(result.points3d)
        a = audit(result, max_reprojection_px=max_reprojection_px)
        out.update({
            'status': 'ready', 'result_sha256': sha256(record.result),
            'n_frames': result.n_frames, 'camera_names': result.camera_names,
            'image_sizes': result.image_sizes, 'source_frame_counts': [len(x) for x in files],
            'scale': scale_meta, 'audit': a,
        })
    except (AssertionError, OSError, ValueError, RuntimeError, KeyError, TypeError,
            EOFError, OverflowError) as exc:
        out['status'] = 'quarantined'
        out['reason'] = 'invalid_result:' + str(exc)
    return out


@contextlib.contextmanager
def root_lock(root: Path):
    """Serialize conversions targeting one root."""
    import fcntl
    lock = root.parent / f'.{root.name}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict) -> None:
    """Write one JSON manifest and atomically replace its previous version."""
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def orphan_results(records: list[Record], outputs: Path) -> list[str]:
    """Return result files not associated with an extracted trial."""
    known = {str(r.result.resolve()) for r in records if r.result is not None}
    return [str(p) for p in _candidate_result_paths(Path(outputs)) if str(p.resolve()) not in known]


def _policy_signature(records: list[Record], splits: dict[str, str], extras: list[str] = (),
                      max_reprojection_px: float | None = None) -> dict:
    """Stable conversion inputs used to make reruns no-ops or explicit refusals."""
    return {
        'policy_version': POLICY_VERSION,
        'units': 'mm',
        'scale_reference_mm': REFERENCE_COXA_MM,
        'scale_front_coxa_pairs': [list(p) for p in FRONT_COXA_PAIRS],
        'axis': NATIVE_NAMES,
        'splits': splits,
        'records': {f'{r.condition}/{r.archive}': _source_hashes(r) for r in records},
        'orphan_results': list(extras),
        'max_reprojection_px': max_reprojection_px,
    }


def _same_policy(root: Path, policy: dict) -> bool:
    """Return whether a previous root is identical and its sessions still validate."""
    p = root / 'conversion_manifest.json'
    if not p.exists():
        return False
    try:
        old = json.loads(p.read_text())
        if old.get('input_signature') != policy:
            return False
        for entry in old.get('records', []):
            if entry.get('status') != 'converted':
                continue
            rel = entry.get('session')
            if not rel:
                return False
            session = root / rel
            if not (session / 'session.toml').exists():
                return False
            loaded = fmt.Session.load(session)
            if fmt.validate_session(loaded, check_images=False):
                return False
        return True
    except (OSError, ValueError, fmt.FormatError):
        return False


def inventory_command(args) -> int:
    """Run discovery and schema/audit checks, writing a JSON inventory."""
    records = discover(Path(args.source), Path(args.outputs))
    report = [inspect_record(r, max_reprojection_px=args.max_reprojection_px) for r in records]
    extras = orphan_results(records, Path(args.outputs))
    payload = {
        'dataset': 'deepfly3d-v3', 'converter_policy': POLICY_VERSION,
        'source_root': str(args.source), 'outputs_root': str(args.outputs),
        'expected_source_sessions': len(records), 'orphan_results': extras, 'records': report,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, payload)
    bad = [r for r in report if r['status'] != 'ready']
    bad.extend({'status': 'quarantined'} for _ in extras)
    print(f'inventory: {len(records)} trials; {len(bad)} with issues; report={report_path}')
    return 1 if bad and args.strict else 0


def convert_command(args) -> int:
    """Convert all ready trials, quarantining malformed trials in the manifest."""
    source, outputs, root = Path(args.source), Path(args.outputs), Path(args.out)
    records = discover(source, outputs)
    splits = split_map(records)
    extras = orphan_results(records, outputs)
    policy = _policy_signature(records, splits, extras, args.max_reprojection_px)
    root.parent.mkdir(parents=True, exist_ok=True)
    with root_lock(root):
        if root.exists():
            if _same_policy(root, policy):
                print(f'conversion is already complete and identical: {root}')
                return 0
            if any(root.iterdir()) and not args.force:
                raise SystemExit(f'output root is non-empty or differs from previous conversion '
                                 f'(use --force to quarantine it): {root}')
            if args.force and any(root.iterdir()):
                old = root.with_name(f'{root.name}.quarantine-{time.time_ns()}')
                shutil.move(str(root), str(old))
        root.mkdir(parents=True, exist_ok=True)
        for split in fmt.SPLITS:
            split_dir = root / split
            split_dir.mkdir(exist_ok=True)
            for stale in split_dir.glob('.*.tmp'):
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
        for stale in root.glob('.*.tmp'):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        (root / '.pixel_views').mkdir(exist_ok=True)
        _atomic_json(root / 'split_manifest.json', {
            'dataset': 'deepfly3d-v3',
            'policy': 'family grouped; condition + acquisition date + FlyN stays in one split',
            'source_family_split': 'deepfly3d_common.split_map (v2 family policy)',
            'assignment': splits,
        })
        inventory = []
        for record in records:
            split = splits[record.family]
            try:
                inventory.append(convert_record(record, split, root, force=False,
                                                 max_reprojection_px=args.max_reprojection_px))
            except (AssertionError, OSError, ValueError, RuntimeError, EOFError,
                    OverflowError) as exc:
                inventory.append({'condition': record.condition, 'archive': record.archive,
                                  'split': split, 'status': 'quarantined',
                                  'reason': 'conversion_failed:' + str(exc),
                                  'source_result': str(record.result) if record.result else None})
        manifest = {
            'dataset': 'deepfly3d-v3', 'format': 'tailcycle-dataset; docs/annotation_format.md',
            'mode': '3d', 'units': 'mm', 'labels': 'tracked',
            'status': 'provisional_unvalidated', 'converter_policy': POLICY_VERSION,
            'source_root': str(source), 'outputs_root': str(outputs),
            'created': date.today().isoformat(), 'converter_git_commit': git_commit(),
            'keypoint_names': NATIVE_NAMES,
            'keypoint_names_sha256': hashlib.sha256('\n'.join(NATIVE_NAMES).encode()).hexdigest(),
            'camera_order': list(CAMERA_ORDERING),
            'camera_policy': 'native HDF5 camera order; index c maps to source camera_c',
            'calibration_policy': 'native bundle_adjustment camera geometry; translations scaled to mm',
            'scale_policy': '0.456 mm / mean native front coxa lengths (0-1, 19-20)',
            'scale_reference_mm': REFERENCE_COXA_MM,
            'pixel_policy': 'numbered per-camera symlink farms target original extracted images',
            'keypoint_status_policy': 'finite selected 2D prediction=visible; NaN=missing (native pathway plan)',
            'point3d_status_policy': 'finite triangulated point with >=2 views=projected; no independent visibility',
            'confidence_policy': 'raw pose2d/conf retained as non-spec score; never used as a threshold',
            'split_counts': {s: sum(x.get('status') == 'converted' and x.get('split') == s
                                    for x in inventory) for s in fmt.SPLITS},
            'expected_source_sessions': len(records),
            'converted_sessions': sum(x.get('status') == 'converted' for x in inventory),
            'quarantined_source_sessions': sum(x.get('status') == 'quarantined' for x in inventory),
            'orphan_results': extras, 'input_signature': policy, 'records': inventory,
        }
        _atomic_json(root / 'conversion_manifest.json', manifest)
    print(f'converted {manifest["converted_sessions"]} native 3-D sessions; '
          f'quarantined {manifest["quarantined_source_sessions"]}')
    print('WARNING: units are scaled provisional mm; calibration remains provisional_unvalidated',
          file=sys.stderr)
    if records and not manifest['converted_sessions']:
        print(f'ERROR: every one of {len(records)} trials was quarantined; see '
              f'{root / "conversion_manifest.json"}', file=sys.stderr)
        return 1
    return 0


def main() -> int:
    """Dispatch inventory or conversion."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory', help='discover and audit native HDF5 results')
    inv.add_argument('--source', required=True, type=Path)
    inv.add_argument('--outputs', required=True, type=Path)
    inv.add_argument('--report', required=True, type=Path)
    inv.add_argument('--max-reprojection-px', type=float, default=None,
                     help='optional hard audit gate; default records residuals without rejecting')
    inv.add_argument('--strict', action='store_true')
    inv.set_defaults(func=inventory_command)
    conv = sub.add_parser('convert', help='write tailcycle mode=3d sessions')
    conv.add_argument('--source', required=True, type=Path)
    conv.add_argument('--outputs', required=True, type=Path)
    conv.add_argument('--out', required=True, type=Path)
    conv.add_argument('--force', action='store_true',
                      help='quarantine an existing root before replacing it')
    conv.add_argument('--workers', type=int, default=1,
                      help='reserved for compatibility; conversion is serialized per root')
    conv.add_argument('--max-reprojection-px', type=float, default=None,
                      help='optional hard audit gate; default records residuals without rejecting')
    conv.set_defaults(func=convert_command)
    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
