#!/usr/bin/env python
"""Split QDMouse4M reference/fluorescence sidecars into explicit cameras.

The processed qdmouse4m-fluo root stores ``<view>.mp4`` and
``<view>_fluo.mp4`` under one six-camera calibration.  This converter derives a
normal tailcycle dataset whose camera axis has one reference and one
fluorescence camera per calibrated view.  Fluorescence has no independent
calibration in the source, so its calibration block is an exact copy of the
associated reference block; this is the source's channel-separated-view
semantics, not a newly estimated geometry.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from copy import deepcopy

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tailcyclenet import format as fmt

SOURCE = Path("/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/qdmouse4m-fluo")
OUTPUT = Path("/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/qdmouse4m-fluo-separate-cameras")


def split_cameras(calibration: Path) -> tuple[fmt.Rig, list[str], dict[str, str]]:
    """Duplicate each calibrated view as ``<view>_fluo``.

    Returns the expanded rig, its names, and the reference-to-fluorescence
    mapping.  The source calibration is parsed, never modified in place.
    """
    with calibration.open("rb") as f:
        doc = tomllib.load(f)
    expanded = {}
    mapping = {}
    i = 0
    for key, block in doc.items():
        if key == "metadata" or not isinstance(block, dict):
            continue
        ref = str(block.get("name", ""))
        if not ref:
            raise RuntimeError(f"{calibration}: camera block {key!r} has no name")
        fluo = f"{ref}_fluo"
        if fluo in mapping or ref in mapping:
            raise RuntimeError(f"{calibration}: duplicate camera mapping for {ref!r}")
        a = deepcopy(block)
        b = deepcopy(block)
        a["name"] = ref
        b["name"] = fluo
        expanded[f"cam_{i}"] = a
        expanded[f"cam_{i + 1}"] = b
        mapping[ref] = fluo
        i += 2
    if not mapping:
        raise RuntimeError(f"{calibration}: no camera blocks")
    expanded["metadata"] = deepcopy(doc.get("metadata", {}))
    rig = fmt.rig_from_doc(expanded, str(calibration))
    return rig, rig.names, mapping


def camera_index_map(src_names: list[str], dst_names: list[str],
                     suffix: str = "_fluo") -> list[int]:
    """For each destination camera, the source camera whose labels it carries.

    The expanded rig INTERLEAVES ``<view>`` and ``<view>_fluo``, so expanding an array as
    ``concatenate([a, a], axis=camera)`` -- a block copy -- would hand destination camera ``j``
    the labels of source camera ``j``, a different physical view.  Every destination camera is
    a channel of exactly one source camera; this is the map that says which, and it is derived
    from NAMES so it stays correct whatever order ``split_cameras`` grows.
    """
    out = []
    for name in dst_names:
        base = name[: -len(suffix)] if name.endswith(suffix) else name
        if base not in src_names:
            raise RuntimeError(f"destination camera {name!r} has no source camera {base!r}")
        out.append(src_names.index(base))
    return out


def convert_session(source: Path, output: Path) -> dict[str, str]:
    """Convert one session through ``Session.load``/``write_session`` APIs."""
    old = fmt.Session.load(source)
    rig, _, mapping = split_cameras(source / "calibration.toml")
    take = camera_index_map(old.cam_names, rig.names)

    def pick(arr, axis: int):
        """Map one source label array onto the expanded camera axis by NAME."""
        return None if arr is None else np.take(np.asarray(arr), take, axis=axis)

    groups = {
        gid: fmt.Group(group_id=g.group_id, n_frames=g.n_frames, fps=g.fps,
                       source_video=g.source_video, source_frame_start=g.source_frame_start,
                       source_frame_step=g.source_frame_step, notes=g.notes)
        for gid, g in old.groups.items()
    }
    labels = {}
    for gid in old.groups:
        lab = old.labels(gid)
        if lab.points2d is None or lab.vis2d is None:
            raise RuntimeError(f"{source}/{gid}: expected source per-camera 2D labels")
        # Existing rows are PROJECTED projections of the 3D layer.  Duplicating
        # them for the channel-separated stream preserves that status rather
        # than asserting fluorescence visibility observations.
        regions = None
        if lab.regions is not None:
            base = np.asarray(lab.regions, dtype=np.float64)
            rows = []
            for j, src in enumerate(take):
                sel = base[:, 1] == src
                if not sel.any():
                    continue
                block = base[sel].copy()
                block[:, 1] = j
                rows.append(block)
            regions = np.concatenate(rows, axis=0) if rows else np.zeros((0, 6))
        labels[gid] = fmt.Labels(
            animal_ids=list(lab.animal_ids),
            points3d=lab.points3d,
            vis3d=lab.vis3d,
            points2d=pick(lab.points2d, 3),
            vis2d=pick(lab.vis2d, 3),
            boxes=pick(lab.boxes, 2),
            instance=pick(lab.instance, 2),
            ext=pick(lab.ext, 0),
            regions=regions,
        )
    provenance = dict(old.provenance)
    provenance.update({
        "source": "qdmouse4m-fluo (separate reference/fluorescence cameras)",
        "camera_modalities": {"reference": list(mapping), "fluorescence": list(mapping.values())},
        "fluorescence_calibration": "copied associated reference camera geometry; no independent source calibration",
        "camera_label_source": {"dst": list(rig.names),
                               "src_index": take},
    })
    fmt.write_session(
        output, mode=old.mode, units=old.units, label_source=old.label_source,
        names=old.names, rig=rig, groups=groups, labels=labels,
        skeleton=old.skeleton, flip_pairs=old.flip_pairs, provenance=provenance,
        assoc_res_max_px=old.assoc_res_max_px,
    )
    for gid, g in old.groups.items():
        dest = output / "groups" / gid
        dest.mkdir(parents=True, exist_ok=True)
        for ref, fluo in mapping.items():
            for cam, suffix in ((ref, ".mp4"), (fluo, ".mp4")):
                src_name = f"{ref}.mp4" if cam == ref else f"{ref}_fluo.mp4"
                src = g.dir / src_name
                if not src.exists():
                    raise RuntimeError(f"{source}/{gid}: missing {src_name}")
                (dest / f"{cam}{suffix}").symlink_to(src)
    return mapping


def convert(*, source=SOURCE, output=OUTPUT, overwrite=False, validate=False) -> None:
    """Convert all train/val/test sessions without touching ``source``."""
    source, output = Path(source), Path(output)
    if output.exists():
        if not overwrite:
            raise SystemExit(f"{output} exists; use --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    # Keep converter manifests alongside the derived root, updating only the
    # manifest's output pointer; label policy is unchanged.
    for stem in ("selection.json", "label_policy.json"):
        src_manifest = source / stem
        if src_manifest.exists():
            if stem == "selection.json":
                manifest = json.loads(src_manifest.read_text())
                manifest["output"] = str(output)
                (output / stem).write_text(json.dumps(manifest, indent=2) + "\n")
            else:
                shutil.copy2(src_manifest, output / stem)
    for split in ("train", "val", "test"):
        src_split = source / split
        if not src_split.exists():
            continue
        for src in sorted(p for p in src_split.iterdir() if p.is_dir()):
            dst = output / split / src.name
            mapping = convert_session(src, dst)
            print(f"wrote {dst} ({len(mapping)} reference + {len(mapping)} fluorescence cameras)")
    if validate:
        ds = fmt.load_dataset(output)
        errors = fmt.validate_dataset(ds, check_images=False)
        if errors:
            raise RuntimeError("validation failed:\n" + "\n".join(errors[:50]))
        print(f"validated {len(ds.all_sessions())} sessions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    convert(source=args.source, output=args.output, overwrite=args.overwrite, validate=args.validate)


if __name__ == "__main__":
    main()
