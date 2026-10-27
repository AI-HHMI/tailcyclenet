#!/usr/bin/env python
"""Derive a fluorescence-only QDMouse root from the combined (reference + fluo) cleaned root.

Keeps the ``<view>_fluo`` cameras of every session, in combined-rig order, with their copied
reference calibration, per-camera labels and videos (symlinked to their resolved files). 3D
labels, groups and the cleaning manifest are unchanged: the cleaning is the reference root's.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tailcyclenet import format as fmt

ROOT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
COMBINED = ROOT / 'qdmouse4m-combined-cleaned-v2'
OUTPUT = ROOT / 'qdmouse4m-fluo-cleaned-v2'
SUFFIX = '_fluo'


def fluo_rig(calibration: Path) -> fmt.Rig:
    """The combined calibration restricted to its fluorescence blocks, renumbered cam_0.."""
    with calibration.open('rb') as f:
        doc = tomllib.load(f)
    blocks = [b for k, b in doc.items()
              if k != 'metadata' and isinstance(b, dict) and str(b.get('name', '')).endswith(SUFFIX)]
    if not blocks:
        raise RuntimeError(f'{calibration}: no {SUFFIX} cameras')
    out = {f'cam_{i}': b for i, b in enumerate(blocks)}
    out['metadata'] = doc.get('metadata', {})
    return fmt.rig_from_doc(out, str(calibration))


def convert(combined: Path, output: Path, overwrite: bool = False) -> dict[str, int]:
    """Build the fluo-only sibling in a staging dir and rename it into place."""
    combined, output = combined.resolve(), Path(output).absolute()
    if output.exists() and not overwrite:
        raise RuntimeError(f'{output} exists; pass --overwrite to replace it')
    ds = fmt.load_dataset(combined)
    stage = output.with_name(f'.{output.name}.tmp-{os.getpid()}')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    counts = {'sessions': 0, 'groups': 0, 'symlinks': 0}
    try:
        for split in fmt.SPLITS:
            for s in ds.sessions.get(split, []):
                rig = fluo_rig(s.path / 'calibration.toml')
                take = [s.cam_names.index(n) for n in rig.names]

                def pick(arr, axis):
                    """Keep only the fluorescence cameras on one label array's camera axis."""
                    return None if arr is None else np.take(np.asarray(arr), take, axis=axis)

                groups = {gid: fmt.Group(group_id=g.group_id, n_frames=g.n_frames, fps=g.fps,
                                         source_video=g.source_video,
                                         source_frame_start=g.source_frame_start,
                                         source_frame_step=g.source_frame_step, notes=g.notes)
                          for gid, g in s.groups.items()}
                labels = {}
                for gid in s.groups:
                    lab = s.labels(gid)
                    labels[gid] = fmt.Labels(
                        animal_ids=list(lab.animal_ids),
                        points3d=lab.points3d, vis3d=lab.vis3d,
                        points2d=pick(lab.points2d, 3), vis2d=pick(lab.vis2d, 3),
                        boxes=pick(lab.boxes, 2), instance=pick(lab.instance, 2),
                        ext=pick(lab.ext, 0))
                prov = dict(s.provenance)
                prov.pop('camera_label_source', None)
                prov['source'] = str(s.path)
                prov['camera_modalities'] = {'fluorescence': list(rig.names)}
                out_session = stage / split / s.session_id
                fmt.write_session(
                    out_session, mode=s.mode, units=s.units, label_source=s.label_source,
                    names=s.names, rig=rig, groups=groups, labels=labels, skeleton=s.skeleton,
                    flip_pairs=(s.flip_pairs if s.flip_pairs_declared else None),
                    provenance=prov, assoc_res_max_px=s.assoc_res_max_px)
                if fmt.Session.load(out_session).has_visibility_assessment != s.has_visibility_assessment:
                    raise RuntimeError(f'{s.path}: visibility assessment changed')
                for gid in s.groups:
                    d = out_session / 'groups' / gid
                    d.mkdir(parents=True, exist_ok=True)
                    for cam in rig.names:
                        src = (s.path / 'groups' / gid / f'{cam}.mp4').resolve(strict=True)
                        fmt.link(d / f'{cam}.mp4', src)
                        counts['symlinks'] += 1
                counts['groups'] += len(groups)
                counts['sessions'] += 1
        for name in ('cleaning_manifest.tsv', 'cleaning_summary.json'):
            shutil.copy2(combined / name, stage / name)
        import toml
        prov = toml.load(combined / 'provenance.toml')
        prov['fluo_only'] = {
            'combined_source_root': str(combined),
            'cameras': 'only <view>_fluo cameras kept; calibration = copied reference geometry',
            'created_utc': datetime.now(timezone.utc).isoformat(),
        }
        (stage / 'provenance.toml').write_text(toml.dumps(prov))
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
        return counts
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser()
    p.add_argument('--combined', type=Path, default=COMBINED)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    print(convert(a.combined, a.output, a.overwrite), flush=True)


if __name__ == '__main__':
    main()
