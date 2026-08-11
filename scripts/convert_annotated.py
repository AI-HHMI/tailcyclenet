#!/usr/bin/env python
"""Convert the anivia `motor-observatory-frames-posetail-*` export into the tailcycle format.

    pixi run python scripts/convert_annotated.py --validate

The export is already spec-SHAPED -- session.toml, calibration.toml, per-group frame dirs -- but
is not readable by `tailcyclenet.format`. Nine things differ, and each is closed here:

1. tables are CSV, not parquet
2. there is no split level; `--val-root` picks the split
3. `calibration.toml` `size` is the SENSOR (1440x1080) while the JPEGs are the crop; the truth
   lives in `session.toml` `[cameras.<c>] image_wh` / `offset`, which is where validation rule 8
   and the 16.9-vs-2.38 mm offset lesson (spec S5) both bite
4. `[cameras.*]` belongs in calibration.toml, not session.toml
5. `groups.csv` carries `label_frames`, which the spec deleted on purpose (S6)
6. three of the 27 sessions order `names` with the RF-* and LH-* blocks swapped; rule 3 wants one
   order per root, and the tables are name-indexed so rewriting the list moves nothing
7. there is no 3D layer at all -- see `triangulate_group`
8. `--twod-root`'s cameras were not properly synced, so it is exported per camera as 2D sessions
9. pixels are symlinked, one link per camera per group

A group with no `visible` row anywhere is dropped, loudly (spec S13): it carries no supervision,
and `dataset._labelled_frames` counts a `missing` row as labelled, so such a group yields index
entries that `_item` rejects forever.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/allen-mouse-training/'
           'motor-observatory-frames-posetail-2026-08-06')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/'
           'allen-mouse-annotated')
VAL_ROOT = 'motor-observatory-frames-2025-02-13-eval'
TWOD_ROOT = 'motor-observatory-frames-2024-05-12'


# ----------------------------------------------------------------------------------------------
# reading the export
# ----------------------------------------------------------------------------------------------

def rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def read_rig(sdir: Path, cfg: dict) -> fmt.Rig:
    """calibration.toml + session.toml's `[cameras.*]` -> a Rig whose `size` is the IMAGE.

    The file's `size` is the sensor the intrinsics were calibrated against; the pixels on disk are
    a crop of it. Keeping the sensor size would fail rule 8 and, worse, would make every consumer
    project into the wrong place.
    """
    rig = fmt.load_calibration(sdir / 'calibration.toml')
    cams = cfg.get('cameras', {})
    for name in rig.names:
        c = cams.get(name)
        if c is None:
            raise SystemExit(f'{sdir}: camera {name!r} is in calibration.toml but not session.toml')
        rig.by_name(name).set_size(tuple(int(v) for v in c['image_wh']))
        rig.offset[name] = tuple(float(v) for v in c['offset'])
        rig.moving[name] = False
    return rig


def read_session(sdir: Path, names: list[str]) -> tuple:
    """(cfg, rig, groups, labels) with labels keyed by group id, in the CANONICAL name order."""
    with open(sdir / 'session.toml', 'rb') as f:
        cfg = tomllib.load(f)
    rig = read_rig(sdir, cfg)
    if sorted(cfg['names']) != sorted(names):
        raise SystemExit(f'{sdir}: keypoint names differ from the canonical set, not just in order')

    ki = {n: i for i, n in enumerate(names)}
    ci = {n: i for i, n in enumerate(rig.names)}
    K, C = len(names), len(rig)

    groups = {}
    for r in rows(sdir / 'groups.csv'):
        gid = r['group_id']
        groups[gid] = fmt.Group(
            gid, int(r['n_frames']),
            fps=float(r['fps'] or 'nan'), source_video=r.get('source_video') or '',
            source_frame_start=int(r.get('source_frame_start') or 0),
            source_frame_step=int(r.get('source_frame_step') or 1),
            notes=r.get('notes') or '')
    kpt = defaultdict(list)
    for r in rows(sdir / 'keypoints.csv'):
        kpt[r['group_id']].append(r)
    inst = defaultdict(list)
    if (sdir / 'instances.csv').exists():
        for r in rows(sdir / 'instances.csv'):
            inst[r['group_id']].append(r)

    labels = {}
    for gid, g in groups.items():
        animals = sorted({r['animal_id'] for r in kpt[gid]} | {r['animal_id'] for r in inst[gid]})
        if not animals:
            continue
        ai = {a: i for i, a in enumerate(animals)}
        S, T = len(animals), g.n_frames
        lab = fmt.empty_labels(S, T, K, C, mode3d=False, animal_ids=animals)
        lab.boxes = np.full((S, T, C, 4), np.nan, np.float32)
        lab.instance = np.full((S, T, C), fmt.INST_NONE, np.int8)
        for r in kpt[gid]:
            a, f, c, k = ai[r['animal_id']], int(r['frame']), ci[r['camera']], ki[r['bodypart']]
            lab.vis2d[a, f, k, c] = fmt.KPT_STATUS[r['status']]
            if r['status'] == 'visible':
                lab.points2d[a, f, k, c] = (float(r['x']), float(r['y']))
        for r in inst[gid]:
            a, f, c = ai[r['animal_id']], int(r['frame']), ci[r['camera']]
            lab.instance[a, f, c] = fmt.INST_STATUS[r['status']]
            if r['x0']:
                lab.boxes[a, f, c] = [float(r[q]) for q in ('x0', 'y0', 'x1', 'y1')]
        labels[gid] = lab
    return cfg, rig, groups, labels


# ----------------------------------------------------------------------------------------------
# the 3D layer
# ----------------------------------------------------------------------------------------------

def triangulate_group(rig: fmt.Rig, lab: fmt.Labels, gate: float) -> tuple[int, int, int]:
    """Fill `lab.points3d` / `lab.vis3d` from the per-camera 2D. Returns (visible, gated, missing).

    The 2D lives in stored-image pixels and the calibration in sensor pixels, so `offset` goes
    back on before triangulating (spec S5) and the stored 2D is never touched.

      >= 2 views, median reprojection residual <= gate  -> visible + xyz
      >= 2 views over the gate, or exactly 1 view       -> unlabeled (no row)
      0 views visible and EVERY camera assessed it      -> missing, the honest 3D occlusion
      anything else                                     -> no row
    """
    import torch

    S, T, K, C, _ = lab.points2d.shape
    off = np.array([rig.offset[n] for n in rig.names], np.float64)          # (C,2)
    p2 = lab.points2d.astype(np.float64) + off
    lab.points3d = np.full((S, T, K, 3), np.nan, np.float32)
    lab.vis3d = np.full((S, T, K), fmt.UNLABELED, np.int8)

    nvis = np.isfinite(p2).all(-1).sum(-1)                                  # (S,T,K)
    idx = np.flatnonzero(nvis.reshape(-1) >= 2)
    flat_vis, flat_p3 = lab.vis3d.reshape(-1), lab.points3d.reshape(-1, 3)

    n_vis = n_gated = 0
    if idx.size:
        X = np.ascontiguousarray(p2.reshape(-1, C, 2)[idx].transpose(1, 0, 2))   # (C,n,2)
        with torch.no_grad():
            # aniposelib on the pytorch branch holds intrinsics as nn.Parameters, so without
            # no_grad both calls die on "Can't call numpy() on Tensor that requires grad".
            p3 = np.asarray(rig.cgroup.triangulate(X, progress=False))
            rep = np.asarray(rig.cgroup.reprojection_error(p3, X, mean=False))
        err = np.linalg.norm(rep, axis=-1)                                  # (C,n)
        seen = np.isfinite(X).all(-1)
        res = np.nanmedian(np.where(seen, err, np.nan), axis=0)
        good = np.isfinite(p3).all(-1) & (res <= gate)
        flat_vis[idx[good]] = fmt.VISIBLE
        flat_p3[idx[good]] = p3[good].astype(np.float32)
        n_vis, n_gated = int(good.sum()), int((~good).sum())

    assessed_all = (lab.vis2d != fmt.UNLABELED).all(-1)                     # (S,T,K)
    miss = ((nvis == 0) & assessed_all).reshape(-1)
    flat_vis[miss] = fmt.MISSING
    return n_vis, n_gated, int(miss.sum())


# ----------------------------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------------------------

def link(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def drop_unlabelled(groups: dict, labels: dict, where: str) -> list[str]:
    """Groups with no `visible` row carry no supervision. Drop them, never silently."""
    dropped = []
    for gid in list(groups):
        lab = labels.get(gid)
        if lab is None or not (lab.vis2d == fmt.VISIBLE).any():
            dropped.append(gid)
            groups.pop(gid)
            labels.pop(gid, None)
    if dropped:
        print(f'   ! {where}: dropped {len(dropped)} group(s) with no visible label: '
              f'{" ".join(dropped)}')
    return dropped


def created_date(cfg: dict, root: str) -> str:
    """`created` is "02-13-eval" in one root. Fall back to the date in the root folder name."""
    got = str(cfg.get('provenance', {}).get('created', ''))
    if len(got) == 10 and got[4] == got[7] == '-':
        return got
    parts = root.split('-')
    return '-'.join(parts[3:6]) if len(parts) >= 6 else ''


def write_3d(dst: Path, cfg: dict, rig, groups, labels, names, root, sid, gate) -> dict:
    stats = {'visible': 0, 'gated': 0, 'missing': 0}
    for gid, lab in labels.items():
        v, g, m = triangulate_group(rig, lab, gate)
        stats['visible'] += v
        stats['gated'] += g
        stats['missing'] += m
    fmt.write_session(
        dst, mode='3d', units='mm', label_source='annotated', names=names, rig=rig,
        groups=groups, labels=labels,
        skeleton=cfg.get('skeleton', []), flip_pairs=cfg.get('flip_pairs', []),
        provenance={
            'source': f'{SRC.name}/{root}/{sid}',
            'annotator': cfg.get('provenance', {}).get('annotator', ''),
            'annotator_tool': cfg.get('provenance', {}).get('annotator_tool', 'anivia'),
            'created': created_date(cfg, root),
            'converter': 'scripts/convert_annotated.py',
            'points3d_source': f'triangulated (aniposelib) from the per-camera 2D, '
                               f'residual gate {gate:g} px',
        })
    return stats


def sub_rig(rig: fmt.Rig, cam: str) -> fmt.Rig:
    from aniposelib.cameras import CameraGroup
    return fmt.Rig(CameraGroup([rig.by_name(cam)]), offset={cam: rig.offset[cam]},
                   moving={cam: False}, calibrated={cam: rig.calibrated[cam]})


def slice_camera(lab: fmt.Labels, c: int) -> fmt.Labels:
    return fmt.Labels(
        animal_ids=list(lab.animal_ids), points3d=None, vis3d=None,
        points2d=lab.points2d[:, :, :, c:c + 1].copy(), vis2d=lab.vis2d[:, :, :, c:c + 1].copy(),
        boxes=lab.boxes[:, :, c:c + 1].copy(), instance=lab.instance[:, :, c:c + 1].copy())


def write_2d(out_split: Path, cfg, rig, groups, labels, names, root, sid, pixels, dry) -> int:
    """One session PER CAMERA: rule 5 says mode=2d is exactly one camera.

    Real intrinsics and the real offset are kept. They are legal in a 2D session, cost nothing,
    and let the same session be read later as 3D single-view; the 2D training path fixes
    cam_ix = [0] and never looks at them. What is NOT kept is the cross-camera geometry: the
    cameras in this root were not properly synced, so frame i of two views is not one instant and
    triangulating them would be inventing a 3D label.
    """
    n = 0
    for c, cam in enumerate(rig.names):
        gsub = dict(groups)
        lsub = {gid: slice_camera(lab, c) for gid, lab in labels.items()}
        drop_unlabelled(gsub, lsub, f'{sid}__{cam}')
        if not gsub:
            print(f'   ! {sid}__{cam}: no labelled groups, session not written')
            continue
        dst = out_split / f'{sid}__{cam}'
        if not dry:
            write_pixels(dst, gsub, pixels, [cam])
            fmt.write_session(
                dst, mode='2d', units='px', label_source='annotated', names=names,
                rig=sub_rig(rig, cam), groups=gsub, labels=lsub,
                skeleton=cfg.get('skeleton', []), flip_pairs=cfg.get('flip_pairs', []),
                provenance={
                    'source': f'{SRC.name}/{root}/{sid}',
                    'annotator': cfg.get('provenance', {}).get('annotator', ''),
                    'annotator_tool': cfg.get('provenance', {}).get('annotator_tool', 'anivia'),
                    'created': created_date(cfg, root),
                    'converter': 'scripts/convert_annotated.py',
                    'camera': cam,
                    'mode_note': 'the cameras in this root were not properly synced, so the '
                                 'session is exported per camera as 2D',
                })
        n += 1
    return n


def write_pixels(dst: Path, groups: dict, pixels: Path, cams: list[str]) -> None:
    for gid in groups:
        gdir = dst / 'groups' / gid
        gdir.mkdir(parents=True, exist_ok=True)
        for cam in cams:
            link(gdir / cam, (pixels / gid / cam).resolve())


# ----------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------

def keypoint_rows(sdir: Path, cams: list[str] | None, keep: set[str]) -> set:
    """The source's own keypoint rows as comparable tuples, for the round-trip check."""
    out = set()
    for r in rows(sdir / 'keypoints.csv'):
        if r['group_id'] not in keep or (cams is not None and r['camera'] not in cams):
            continue
        xy = ((np.float32(r['x']), np.float32(r['y'])) if r['status'] == 'visible' else (None, None))
        out.add((r['group_id'], int(r['frame']), r['animal_id'], r['camera'], r['bodypart'],
                 r['status'], *xy))
    return out


def written_rows(path: Path) -> set:
    sess = fmt.Session.load(path)
    inv = {v: k for k, v in fmt.KPT_STATUS.items()}
    out = set()
    for gid in sess.groups:
        lab = sess.labels(gid)
        s, t, k, c = np.nonzero(lab.vis2d != fmt.UNLABELED)
        xy = lab.points2d[s, t, k, c]
        for i in range(len(s)):
            st = inv[int(lab.vis2d[s[i], t[i], k[i], c[i]])]
            pos = (xy[i][0], xy[i][1]) if st == 'visible' else (None, None)
            out.add((gid, int(t[i]), lab.animal_ids[s[i]], sess.cam_names[c[i]],
                     sess.names[k[i]], st, *pos))
    return out


BONES = [('nose', 'L-ear'), ('L-ear', 'R-ear'), ('LF-wrist', 'LF-index-base')]


def anatomy(root: Path) -> None:
    """Bone lengths, not aggregates: a keypoint-axis mistake moves exactly these (spec S13)."""
    got = defaultdict(list)
    for split in fmt.SPLITS:
        for sdir in sorted((root / split).glob('*')) if (root / split).is_dir() else []:
            sess = fmt.Session.load(sdir)
            if sess.mode != '3d':
                continue
            ki = {n: i for i, n in enumerate(sess.names)}
            for gid in sess.groups:
                p3 = sess.labels(gid).points3d
                for a, b in BONES:
                    d = np.linalg.norm(p3[..., ki[a], :] - p3[..., ki[b], :], axis=-1)
                    got[(a, b)].extend(d[np.isfinite(d)].tolist())
    for (a, b), v in got.items():
        print(f'   {a:10s}-{b:14s} median {np.median(v):6.2f} mm   n={len(v)}')


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def canonical_names(src: Path) -> list[str]:
    """The majority `names` order. Rule 3 wants one order per root; three sessions disagree."""
    counts = defaultdict(int)
    for sdir in sorted(src.glob('*/*/session.toml')):
        with open(sdir, 'rb') as f:
            counts[tuple(tomllib.load(f)['names'])] += 1
    order = max(counts.items(), key=lambda kv: kv[1])[0]
    if len(counts) > 1:
        print(f'!  {len(counts)} keypoint orders in the export; writing the majority order '
              f'({counts[order]} of {sum(counts.values())} sessions) everywhere')
    return list(order)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--val-root', default=VAL_ROOT, help='export root that becomes val/')
    ap.add_argument('--twod-root', default=TWOD_ROOT,
                    help='export root exported per camera as 2D (unsynced cameras)')
    ap.add_argument('--max-reproj-px', type=float, default=20.0,
                    help='a triangulated point over this median residual is not written')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--clean', action='store_true', help='remove the output root first')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--no-check', action='store_true', help='skip the row-level round trip')
    ap.add_argument('--no-image-check', action='store_true')
    args = ap.parse_args()

    if args.clean and args.out.exists() and not args.dry_run:
        shutil.rmtree(args.out)

    names = canonical_names(args.src)
    stats = defaultdict(int)
    checks: list[tuple[Path, Path, list[str] | None, set]] = []

    for root in sorted(p.name for p in args.src.iterdir() if p.is_dir()):
        split = 'val' if root == args.val_root else 'train'
        two_d = root == args.twod_root
        print(f'== {root}  -> {split}/  ({"2d per camera" if two_d else "3d"})')
        for sdir in sorted(p for p in (args.src / root).iterdir() if p.is_dir()):
            sid = sdir.name
            cfg, rig, groups, labels = read_session(sdir, names)
            dropped = drop_unlabelled(groups, labels, sid) if not two_d else []
            if not two_d and not groups:
                print(f'   ! {sid}: no labelled groups, session not written')
                stats['sessions_dropped'] += 1
                continue

            if two_d:
                n = write_2d(args.out / split, cfg, rig, groups, labels, names, root, sid,
                             sdir / 'groups', args.dry_run)
                stats['sessions_2d'] += n
                stats['groups'] += n * len(groups)
                print(f'   {split}/{sid}: {n} 2D session(s) x up to {len(groups)} group(s)')
                for cam in rig.names:
                    d = args.out / split / f'{sid}__{cam}'
                    if d.exists():
                        checks.append((d, sdir, [cam], set()))
                continue

            dst = args.out / split / sid
            if not args.dry_run:
                write_pixels(dst, groups, sdir / 'groups', rig.names)
                s = write_3d(dst, cfg, rig, groups, labels, names, root, sid, args.max_reproj_px)
                for k, v in s.items():
                    stats[f'3d_{k}'] += v
                print(f'   {split}/{sid}: {len(groups)} group(s), {len(rig)} cams, '
                      f'3D {s["visible"]} visible / {s["gated"]} gated / {s["missing"]} missing')
                checks.append((dst, sdir, None, set()))
            stats['sessions_3d'] += 1
            stats['groups'] += len(groups)
            stats['groups_dropped'] += len(dropped)

    print(f'\n{dict(stats)}')
    if args.dry_run:
        return

    bad = 0
    if not args.no_check:
        print('\n-- round trip (every source row, status and x,y)')
        for dst, sdir, cams, _ in checks:
            keep = {p.name for p in (dst / 'groups').iterdir()}
            want, got = keypoint_rows(sdir, cams, keep), written_rows(dst)
            if want != got:
                bad += 1
                print(f'   FAIL {dst}: {len(want - got)} row(s) lost, {len(got - want)} invented')
        print(f'   {len(checks)} session(s) checked, {bad} mismatch(es)')

    print('\n-- anatomy of the triangulated 3D')
    anatomy(args.out)

    if args.validate:
        print('\n-- validation')
        errs = fmt.validate_dataset(fmt.load_dataset(args.out),
                                    check_images=not args.no_image_check)
        hard = [e for e in errs if 'WARNING' not in e]
        for e in errs:
            print(('   WARN ' if 'WARNING' in e else '   FAIL ') + e)
        print(f'   {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')
        bad += len(hard)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
