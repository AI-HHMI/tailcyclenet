#!/usr/bin/env python
"""Convert the ORIGINAL QDMouse4M into full-session tailcycle roots (one group per session).

Every source session becomes ONE group covering every frame of the session; its videos are
SYMLINKED to the source mp4s (never re-encoded). ``--variant ref`` links ``videos/refonly`` under
the reference camera names (consistent with ``qdmouse4m-ref-cleaned-v2``); ``--variant fluo``
links ``videos/fluoonly`` under ``<view>_fluo`` names with the reference geometry copied
(consistent with ``qdmouse4m-fluo-cleaned-v2``). Labels, status policy, units, skeleton and
flip pairs are ``qdmouse4m.py``'s own (imported, not reimplemented); splits follow
``qdmouse4m-ref-cleaned-v2``.

    pixi run --frozen python scripts/convert/qdmouse4m_full.py --variant both [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tomllib
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

import qdmouse4m as qd
from tailcyclenet import format as fmt
from tailcyclenet.data.video import frame_table

SOURCE = Path('/groups/karashchuk/karashchuklab/animal-datasets/mouse/qdmouse4m')
ROOT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
SPLIT_REFERENCE = ROOT / 'qdmouse4m-ref-cleaned-v2'
OUTPUT_NAMES = {'ref': 'qdmouse4m-ref-full', 'fluo': 'qdmouse4m-fluo-full'}
VIDEO_DIR = {'ref': 'refonly', 'fluo': 'fluoonly'}
FLUO_SUFFIX = '_fluo'
GROUP_ID = 'g0000'


def reference_splits(reference: Path = SPLIT_REFERENCE) -> dict[str, str]:
    """Session name -> split, read from the reference root's split directories."""
    out: dict[str, str] = {}
    for split in fmt.SPLITS:
        d = reference / split
        if not d.is_dir():
            continue
        for s in d.iterdir():
            if s.is_dir():
                if s.name in out:
                    raise RuntimeError(f'{reference}: {s.name} is in two splits')
                out[s.name] = split
    return out


def video_frames(path: Path) -> int:
    """Frame count of one video from the PyAV frame table (refuses a table-less file)."""
    table = frame_table(str(path))
    if table is None:
        raise RuntimeError(f'{path}: no frame table (packets carry no timestamps)')
    return int(len(table.pts))


def video_rate(path: Path) -> Fraction:
    """The stream's exact average rate (``qdmouse4m.source_fps``)."""
    return qd.source_fps(path)


def variant_rig(calibration: Path, variant: str) -> fmt.Rig:
    """The session rig: the source calibration, cameras renamed ``<view>_fluo`` for fluo."""
    if variant == 'ref':
        return fmt.load_calibration(calibration)
    with calibration.open('rb') as f:
        doc = tomllib.load(f)
    out = {}
    for key, block in doc.items():
        if key == 'metadata' or not isinstance(block, dict):
            continue
        block = dict(block)
        block['name'] = f'{block["name"]}{FLUO_SUFFIX}'
        out[key] = block
    out['metadata'] = doc.get('metadata', {})
    return fmt.rig_from_doc(out, str(calibration))


def full_labels(data: dict[str, np.ndarray], rig: fmt.Rig, n: int) -> fmt.Labels:
    """``qdmouse4m.make_labels`` over the whole session (its window length is a module global)."""
    old = qd.WINDOW
    qd.WINDOW = n
    try:
        return qd.make_labels(data, 0, rig)
    finally:
        qd.WINDOW = old


def inspect_session(session: Path, variants: list[str]) -> dict:
    """Load the npz and check T == frame_index length == every video's frame-table length."""
    data = qd.load_npz(session)
    n = len(data['frame_index'])
    for key in ('keypoints_2d_px', 'keypoints_3d', 'valid_3d', 'valid_2d_conf0p5'):
        if data[key].shape[0] != n:
            raise RuntimeError(f'{session}: {key} has T={data[key].shape[0]}, frame_index {n}')
    cameras = [str(v) for v in data['view_names']]
    ref_rig = fmt.load_calibration(session / 'calibrations' / 'camera_calibration.toml')
    if cameras != ref_rig.names:
        raise RuntimeError(f'{session}: npz cameras {cameras} disagree with calibration {ref_rig.names}')
    counts, rates = {}, {}
    for variant in variants:
        for cam in cameras:
            p = session / 'videos' / VIDEO_DIR[variant] / f'{cam}.mp4'
            if not p.exists():
                raise RuntimeError(f'{session}: missing {p}')
            counts[(variant, cam)] = video_frames(p)
            rates[(variant, cam)] = video_rate(p)
    bad = {f'{v}/{c}': k for (v, c), k in counts.items() if k != n}
    if bad:
        raise RuntimeError(f'{session.name}: npz T={n} but video frame counts {bad}')
    for variant in variants:
        if len({rates[(variant, c)] for c in cameras}) != 1:
            raise RuntimeError(f'{session.name}: {variant} cameras disagree on rate')
    return {'data': data, 'n': n, 'cameras': cameras,
            'rate': {v: rates[(v, cameras[0])] for v in variants}}


def write_variant_session(session: Path, info: dict, variant: str, out_session: Path) -> int:
    """Write one session of one variant (tables + symlinked videos); returns symlink count."""
    rig = variant_rig(session / 'calibrations' / 'camera_calibration.toml', variant)
    n, rate = info['n'], info['rate'][variant]
    group = fmt.Group(group_id=GROUP_ID, n_frames=n, fps=float(rate),
                      source_video=f'{session.name}/videos/{VIDEO_DIR[variant]}',
                      source_frame_start=0, source_frame_step=1,
                      notes='full session')
    labels = full_labels(info['data'], rig, n)
    prov = {
        'source': str(session),
        'annotator': '',
        'annotator_tool': 'five-model consensus + EKS + manual jump review',
        'created': '2025-11-10',
        'label_policy': 'qdmouse4m.make_labels (full session)',
    }
    if variant == 'fluo':
        prov['fluorescence_calibration'] = 'copied associated reference camera geometry'
        prov['camera_modalities'] = {'fluorescence': list(rig.names)}
    fmt.write_session(out_session, mode='3d', units='mm', label_source='tracked',
                      names=qd.KEYPOINTS, rig=rig, groups={GROUP_ID: group},
                      labels={GROUP_ID: labels}, skeleton=qd.SKELETON,
                      flip_pairs=qd.FLIP_PAIRS, provenance=prov)
    d = out_session / 'groups' / GROUP_ID
    d.mkdir(parents=True, exist_ok=True)
    for cam, name in zip(info['cameras'], rig.names):
        src = (session / 'videos' / VIDEO_DIR[variant] / f'{cam}.mp4').resolve(strict=True)
        fmt.link(d / f'{name}.mp4', src)
    return len(rig.names)


def git(*args: str) -> str:
    """Run a git command in this repo and return stripped stdout."""
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def convert(variants: list[str], source: Path = SOURCE, overwrite: bool = False,
            sessions: list[str] | None = None, out_root: Path = ROOT) -> dict:
    """Build every requested variant in staging dirs, then rename each into place."""
    outputs = {v: out_root / OUTPUT_NAMES[v] for v in variants}
    for v, out in outputs.items():
        if out.exists() and not overwrite:
            raise SystemExit(f'{out} exists; pass --overwrite to replace it')
    splits = reference_splits()
    session_dirs = sorted(p for p in (source / 'sessions').iterdir() if p.is_dir())
    if sessions:
        session_dirs = [p for p in session_dirs if p.name in sessions]
    not_in_reference = [p.name for p in session_dirs if p.name not in splits]
    stages = {v: out.with_name(f'.{out.name}.tmp-{os.getpid()}') for v, out in outputs.items()}
    summary = {v: {'sessions': {}, 'frames': {}, 'symlinks': 0} for v in variants}
    try:
        for stage in stages.values():
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
        for session in session_dirs:
            split = splits.get(session.name, 'train')
            info = inspect_session(session, variants)
            for v in variants:
                summary[v]['symlinks'] += write_variant_session(
                    session, info, v, stages[v] / split / session.name)
                summary[v]['sessions'][split] = summary[v]['sessions'].get(split, 0) + 1
                summary[v]['frames'][split] = summary[v]['frames'].get(split, 0) + info['n']
            print(f'{split:5s} {session.name}: {info["n"]} frames', flush=True)
        import toml
        for v in variants:
            doc = {
                'kind': f'qdmouse4m-{v}-full',
                'source_root': str(source),
                'video_variant': VIDEO_DIR[v],
                'grouping': 'one group per session covering every frame; videos symlinked',
                'label_converter': 'scripts/convert/qdmouse4m.py make_labels',
                'reprojection_threshold_px': qd.REPROJECTION_THRESHOLD_PX,
                'split_reference': str(SPLIT_REFERENCE),
                'not_in_split_reference': not_in_reference,
                'code_commit': git('rev-parse', 'HEAD'),
                'code_dirty': bool(git('status', '--short')),
                'created_utc': datetime.now(timezone.utc).isoformat(),
            }
            if v == 'fluo':
                doc['cameras'] = ('<view>_fluo; calibration = copied reference geometry; '
                                  'labels = the reference projections')
            (stages[v] / 'provenance.toml').write_text(toml.dumps(doc))
            (stages[v] / 'label_policy.json').write_text(json.dumps({
                'two_d_source': 'projection_of_3d',
                'reprojection_threshold_px': qd.REPROJECTION_THRESHOLD_PX,
                'source_2d_gate': 'valid_2d_conf0p5',
                'interpolated_3d': 'unlabeled',
                'invalid_reprojection': 'unlabeled',
                'statuses_emitted': ['visible', 'projected'],
            }, indent=2) + '\n')
            (stages[v] / 'summary.json').write_text(json.dumps(summary[v], indent=2) + '\n')
        for v, out in outputs.items():
            if out.exists():
                shutil.rmtree(out)
            stages[v].rename(out)
    except BaseException:
        for stage in stages.values():
            shutil.rmtree(stage, ignore_errors=True)
        raise
    return {'summary': summary, 'not_in_split_reference': not_in_reference}


def validate(root: Path) -> None:
    """Validate a written root (no image check); raise on any error."""
    errors = fmt.validate_dataset(fmt.load_dataset(root), check_images=False)
    if errors:
        raise RuntimeError(f'{root}: validation failed:\n' + '\n'.join(errors[:50]))
    print(f'validated {root}', flush=True)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['ref', 'fluo', 'both'], default='both')
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--sessions', nargs='*', help='restrict to these session names (debugging)')
    p.add_argument('--out-root', type=Path, default=ROOT, help='parent folder of the output roots')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    variants = ['ref', 'fluo'] if a.variant == 'both' else [a.variant]
    print(json.dumps(convert(variants, a.source, a.overwrite, a.sessions, a.out_root), indent=2), flush=True)
    if a.validate:
        for v in variants:
            validate(a.out_root / OUTPUT_NAMES[v])


if __name__ == '__main__':
    main()
