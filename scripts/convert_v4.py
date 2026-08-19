#!/usr/bin/env python
"""Convert `posetail-finetuning-v4` into the tailcycle-dataset format.

    pixi run python scripts/convert_v4.py --dataset allen-mouse --validate
    pixi run python scripts/convert_v4.py --dataset all

One v4 trial becomes one group. **Pixels are symlinked, never copied**: 475 GB of frames live in
`posetail-finetuning-v3` (v4 is a thin npz overlay whose `img`/`vid`/`metadata.yaml` are already
symlinks into it), and one symlink per camera per group is ~900 links across all four datasets
rather than 550,000.

A v4 session becomes one session when its trials share camera metadata (allen-mouse: byte-
identical across all 45) and one session PER TRIAL when they do not (3dpop: calibration is
per-sequence). That rule is data-driven, not a per-dataset special case -- calibration is a
session property in this format, so trials that disagree about it are not one session.

Keypoint names come from configs/datasets/<name>.toml, copied from posetail-pose. They are
authoritative: rat-city and branson-fly carry no `keypoints` array in their npz at all, so the
provenance comment in those files IS the verification.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

V4 = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v4')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
SPECS = Path(__file__).resolve().parent.parent / 'configs' / 'datasets'
DATASETS = ('rat-city', 'allen-mouse', '3dpop', 'branson-fly', 'johnson-mouse')

# v4 cuts one recording into several trials named `<recording>_ix<N>`, where N is the source frame
# of the trial's frame 0. Nothing else encodes that, and `source_frame_start = 0` on a trial that
# starts at 9000 is a false statement about the provenance.
_IX = re.compile(r'_ix(\d+)$')


# the allen column-sort repair

def column_sort_perm(names):
    """perm[i] = the slot holding `names[i]` when a pose array is stored COLUMN-sorted.

    allen's preprocessing writes `pose` from `df[sorted(coord_columns)]` but `keypoints` from
    `np.unique(bare_names)`, and those orders differ wherever a name is a prefix of another:
    '-' is 0x2D and '_' is 0x5F, so 'LF-index-base_x' < 'LF-index_x' as columns while
    'LF-index' < 'LF-index-base' as names. That transposes all 8 `X` / `X-base` pairs -- 16 of
    47 keypoints -- and nothing downstream notices.

    Lifted from posetail-pose `dataset_spec.column_sort_perm`. Returns None when the ordering is
    already name-sorted.
    """
    cols = sorted(f'{n}_{a}' for n in names for a in 'xyz')
    order, seen = [], []
    for c in cols:
        base = c.rsplit('_', 1)[0]
        if base not in seen:
            seen.append(base)
            order.append(base)
    perm = np.array([order.index(n) for n in names], dtype=np.int64)
    return None if (perm == np.arange(len(names))).all() else perm


# v4 reading

def load_spec(name: str) -> dict:
    with open(SPECS / f'{name}.toml', 'rb') as f:
        return tomllib.load(f)


def out_dir(name: str, out_root: Path) -> Path:
    """`out_name` distinguishes a tracked root from the hand-annotated root of the same rig."""
    return out_root / load_spec(name).get('out_name', name)


def read_metadata(trial: Path) -> dict | None:
    p = trial / 'metadata.yaml'
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def rig_from_metadata(meta: dict) -> fmt.Rig:
    """v4's inline calibration -> Rig. 3dpop nests `distortion_matrices` one level deeper."""
    from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera
    from posetail.datasets.utils import disassemble_extrinsics

    intr, extr = meta['intrinsic_matrices'], meta['extrinsic_matrices']
    dist, wid, hei = meta['distortion_matrices'], meta['camera_widths'], meta['camera_heights']
    offsets = meta.get('offset_dict', {})
    cls = FisheyeCamera if meta.get('cam_type') == 'fisheye' else Camera

    names = list(intr)
    names = sorted(names, key=int) if all(n.isdigit() for n in names) else sorted(names)

    cams, offset, moving, calibrated = [], {}, {}, {}
    for n in names:
        rvec, tvec = disassemble_extrinsics(np.asarray(extr[n], dtype=np.float64))
        cam = cls(matrix=np.asarray(intr[n], dtype=np.float64).reshape(3, 3),
                  dist=np.asarray(dist[n], dtype=np.float64).ravel()[:5],
                  rvec=np.asarray(rvec, dtype=np.float64).ravel(),
                  tvec=np.asarray(tvec, dtype=np.float64).ravel(), name=n)
        cam.set_size((int(wid[n]), int(hei[n])))
        cams.append(cam)
        offset[n] = tuple(float(v) for v in offsets.get(n, (0.0, 0.0))[:2])
        moving[n], calibrated[n] = False, True
    return fmt.Rig(CameraGroup(cams), offset=offset, moving=moving, calibrated=calibrated)


def rig_from_pixels(trial: Path, spec: dict) -> fmt.Rig:
    """No metadata.yaml (rat-city, branson-fly): one uncalibrated camera, size from the images."""
    from aniposelib.cameras import CameraGroup
    from PIL import Image

    cam_dirs = sorted(p for p in (trial / 'img').iterdir() if p.is_dir())
    if len(cam_dirs) != 1:
        raise RuntimeError(f'{trial}: expected 1 camera without metadata, found {len(cam_dirs)}')
    name = cam_dirs[0].name
    wh = spec.get('image_wh')
    if wh is None:
        with Image.open(sorted(cam_dirs[0].iterdir())[0]) as im:
            wh = im.size
    cam = fmt.nominal_camera(name, wh)
    return fmt.Rig(CameraGroup([cam]), offset={name: (0.0, 0.0)},
                   moving={name: False}, calibrated={name: False})


def rig_key(rig: fmt.Rig) -> str:
    """Identity of a camera rig, for deciding whether trials belong to one session."""
    def num(t):
        return None if t is None else np.round(np.asarray(
            t.detach().cpu() if hasattr(t, 'detach') else t, dtype=np.float64), 9).tolist()
    return repr([(c.get_name(), rig.cam_type(c.get_name()), tuple(c.get_size()),
                  rig.offset[c.get_name()], num(c.get_camera_matrix()),
                  num(c.get_rotation()), num(c.get_translation()))
                 for c in rig.cameras])


def trial_pixels(trial: Path, cam_names: list[str]) -> dict[str, tuple[str, Path]]:
    """{cam: ('frames'|'video', resolved source path)}. Resolved so the link does not chain."""
    out = {}
    for cam in cam_names:
        d = trial / 'img' / cam
        if d.is_dir():
            out[cam] = ('frames', d.resolve())
            continue
        vids = [trial / 'vid' / (cam + e) for e in fmt.VIDEO_EXTS]
        vids = [v for v in vids if v.exists()]
        if vids:
            out[cam] = ('video', vids[0].resolve())
            continue
        raise RuntimeError(f'{trial}: camera {cam!r} has neither img/ nor vid/')
    return out


def n_pixel_frames(kind: str, src: Path, meta: dict | None) -> int:
    if kind == 'frames':
        return sum(1 for f in src.iterdir() if f.suffix in fmt.IMAGE_EXTS)
    if meta is None or 'num_frames' not in meta:
        raise RuntimeError(f'{src}: video group with no num_frames in metadata.yaml')
    return int(meta['num_frames'])


def build_labels(trial: Path, spec: dict, rig: fmt.Rig, T: int) -> fmt.Labels:
    """v4 npz -> dense label arrays.

    v4 encodes "no label" as NaN and nothing else -- there is no assessed-but-occluded state. So
    a finite point becomes `visible` and a NaN becomes NO ROW (`unlabeled`), never `missing`:
    claiming a point was assessed and judged occluded would be inventing an annotation that was
    never made. allen-mouse's per-camera `vis` array IS a real assessment, and is the one place
    `missing` is written.
    """
    mode3d = spec['mode'] == '3d'
    # allow_pickle: johnson-mouse writes `keypoints` as dtype('O') where allen writes '<U14'.
    npz = np.load(trial / ('pose3d.npz' if mode3d else 'pose2d.npz'), allow_pickle=True)
    pose = npz['pose']
    names = list(spec['names'])
    K, C = len(names), len(rig)

    # The one check that makes a keypoint-axis mistake (gotcha 4, spec 13) impossible to carry
    # forward. Unconditional: it has nothing to do with the allen column-sort repair below, and a
    # dataset that ships names and is never compared against the spec is exactly the silent failure.
    if 'keypoints' in npz:
        stored = [str(s) for s in npz['keypoints']]
        assert stored == names, (
            f'{trial}: npz keypoints disagree with the spec:\n  npz  {stored}\n  spec {names}')

    if spec.get('npz_column_sorted'):
        perm = column_sort_perm(names)
        if perm is not None:
            assert 'keypoints' in npz, f'{trial}: npz_column_sorted but no keypoints array'
            pose = pose[:, :, perm]

    if pose.shape[2] != K:
        raise RuntimeError(f'{trial}: pose has {pose.shape[2]} keypoints, spec says {K}')
    pose = pose[:, :T]
    S = pose.shape[0]

    ids = [str(v) for v in npz['ids']] if 'ids' in npz else [f'a{i:02d}' for i in range(S)]
    finite = np.isfinite(pose).all(-1)                       # (S,T,K)

    lab = fmt.empty_labels(S, T, K, C, mode3d=mode3d, animal_ids=ids)
    if mode3d:
        lab.vis3d[finite] = fmt.VISIBLE
        lab.points3d[finite] = pose[finite].astype(np.float32)
        if 'vis' in npz:
            # A real per-camera assessment: visible / occluded. No 2D position is recorded --
            # the position lives in the 3D layer (spec §7, the rule 10 exemption).
            vis = npz['vis'][:, :T]
            if vis.shape[-1] != C:
                raise RuntimeError(f'{trial}: vis has {vis.shape[-1]} cameras, expected {C}')
            assessed = finite[..., None] & np.ones(C, bool)
            lab.vis2d[assessed] = np.where(vis[assessed], fmt.VISIBLE, fmt.MISSING)
    else:
        lab.vis2d[finite, 0] = fmt.VISIBLE
        lab.points2d[finite, 0] = pose[finite].astype(np.float32)

    # Drop animals with no label anywhere in this group: S is per-group in this format, and an
    # all-NaN row is a slot the tracker never filled, not an animal.
    keep = np.flatnonzero((lab.vis3d if mode3d else lab.vis2d).reshape(S, -1).max(1) != fmt.UNLABELED)
    if len(keep) != S:
        lab = fmt.Labels(
            animal_ids=[ids[i] for i in keep],
            points3d=None if lab.points3d is None else lab.points3d[keep],
            vis3d=None if lab.vis3d is None else lab.vis3d[keep],
            points2d=lab.points2d[keep], vis2d=lab.vis2d[keep],
            boxes=None, instance=None)
    return lab


# conversion

def convert_dataset(name: str, src_root: Path, out_root: Path, max_groups: int | None,
                    dry_run: bool) -> None:
    spec = load_spec(name)
    src = src_root / name
    out = out_dir(name, out_root)
    print(f'== {name}  ({spec["mode"]}, {len(spec["names"])} keypoints)')

    for split in fmt.SPLITS:
        sdir = src / split
        if not sdir.is_dir():
            continue
        for v4_session in sorted(p for p in sdir.iterdir() if p.is_dir()):
            trials = sorted(p for p in v4_session.iterdir() if p.is_dir())
            if max_groups:
                trials = trials[:max_groups]
            if not trials:
                continue

            metas = [read_metadata(t) for t in trials]
            rigs = [rig_from_metadata(m) if m else rig_from_pixels(t, spec)
                    for t, m in zip(trials, metas)]
            one_session = len({rig_key(r) for r in rigs}) == 1

            if one_session:
                buckets = [(v4_session.name, trials, metas, rigs[0])]
            else:
                # Calibration is a session property; trials that disagree are separate sessions.
                buckets = [(f'{v4_session.name}__{t.name}', [t], [m], r)
                           for t, m, r in zip(trials, metas, rigs)]

            for session_id, ts, ms, rig in buckets:
                dst = out / split / session_id
                groups, labels, empty = {}, {}, []
                for t, m in zip(ts, ms):
                    gid = t.name
                    pix = trial_pixels(t, rig.names)
                    T = min(n_pixel_frames(*pix[n], m) for n in rig.names)
                    lab = build_labels(t, spec, rig, T)
                    n_pose = np.load(t / ('pose3d.npz' if spec['mode'] == '3d'
                                          else 'pose2d.npz'),
                                     allow_pickle=True)['pose'].shape[1]
                    if n_pose != T:
                        print(f'   ! {session_id}/{gid}: pose has {n_pose} frames, pixels have '
                              f'{T}; truncated to {T}')
                    if not lab.animal_ids:
                        # v4's score-cleaning can NaN out an entire trial (3dpop val Pigeon01
                        # Sequence49 is 16 frames of nothing). A group with no labels carries no
                        # supervision, so drop it -- loudly, never silently.
                        empty.append(gid)
                        continue
                    ix = _IX.search(gid)
                    groups[gid] = fmt.Group(
                        gid, T,
                        fps=float(m['fps']) if m and 'fps' in m else float('nan'),
                        source_video=str(pix[rig.names[0]][1]),
                        source_frame_start=int(ix.group(1)) if ix else 0, source_frame_step=1)
                    labels[gid] = lab
                    if not dry_run:
                        gdir = dst / 'groups' / gid
                        gdir.mkdir(parents=True, exist_ok=True)
                        for cam, (kind, s) in pix.items():
                            fmt.link(gdir / (cam if kind == 'frames' else cam + s.suffix), s)

                note = f'  [dropped {len(empty)} all-NaN group(s)]' if empty else ''
                print(f'   {split}/{session_id}: {len(groups)} group(s), '
                      f'{sum(g.n_frames for g in groups.values())} frames, '
                      f'{len(rig)} cam(s){note}')
                if dry_run or not groups:
                    if not groups:
                        print(f'   ! {split}/{session_id}: no labelled groups, session not written')
                    continue
                fmt.write_session(
                    dst, mode=spec['mode'], units=spec['units'], label_source='tracked',
                    names=list(spec['names']), rig=rig, groups=groups, labels=labels,
                    skeleton=spec.get('skeleton', []), flip_pairs=spec.get('flip_pairs', []),
                    assoc_res_max_px=spec.get('assoc_res_max_px'),
                    provenance={
                        'source': f'posetail-finetuning-v4/{name}/{split}/{v4_session.name}',
                        'annotator': '', 'annotator_tool': 'scripts/convert_v4.py',
                        'names_provisional': bool(spec.get('names_provisional', False)),
                        'animal_id_source': 'npz ids' if len(ts) and 'ids' in np.load(
                            ts[0] / ('pose3d.npz' if spec['mode'] == '3d' else 'pose2d.npz'),
                            allow_pickle=True) else 'row index',
                    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='all', choices=('all',) + DATASETS, nargs='+')
    ap.add_argument('--src', type=Path, default=V4)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--max-groups', type=int, default=None,
                    help='cap groups per v4 session; for smoke tests')
    ap.add_argument('--dry-run', action='store_true', help='report shapes, write nothing')
    ap.add_argument('--validate', action='store_true', help='validate after writing')
    ap.add_argument('--no-image-check', action='store_true',
                    help='skip opening one image per group during validation')
    ap.add_argument('--clean', action='store_true', help='remove the output dataset dir first')
    args = ap.parse_args()

    names = DATASETS if 'all' in args.dataset else tuple(args.dataset)
    for name in names:
        if args.clean and out_dir(name, args.out).exists() and not args.dry_run:
            shutil.rmtree(out_dir(name, args.out))
        convert_dataset(name, args.src, args.out, args.max_groups, args.dry_run)

    if args.validate and not args.dry_run:
        bad = 0
        for name in names:
            errs = fmt.validate_dataset(fmt.load_dataset(out_dir(name, args.out)),
                                        check_images=not args.no_image_check)
            hard = [e for e in errs if 'WARNING' not in e]
            for e in errs:
                print(('  WARN ' if 'WARNING' in e else '  FAIL ') + e)
            print(f'{name}: {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')
            bad += len(hard)
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
