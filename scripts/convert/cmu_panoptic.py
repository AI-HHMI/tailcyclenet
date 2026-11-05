#!/usr/bin/env python
"""Convert CMU Panoptic Studio multi-person sessions (HD cameras + hdPose3d_stage1_coco19) to one tailcycle 3D root.

Source (raw store): animal-datasets/human/cmupanoptic/<session>/ (calibration_<session>.json, hdVideos/hd_00_XX.mp4,
hdPose3d_stage1_coco19/body3DScene_<hd frame>.json). Licence: CMU Panoptic Studio terms -- research / NON-COMMERCIAL use
only (http://domedb.perception.cs.cmu.edu/); written into every session's provenance.

- Cameras: 8 of the 31 HD cameras, the SAME 8 in every session (CAMS): the 8-subset of the 28 eligible HD cameras that
  maximises the minimum pairwise angle between viewing directions (camera centre minus rig centroid,
  170221_haggling_b1): 3 upper (elevation 32-34 deg), 1 mid (4 deg), 4 lower (-20..-21 deg); azimuths
  -130/-114/-58/-41/22/89/107/160 deg; minimum pairwise separation 54 deg. Eligible = present in every candidate
  session (00_20 / 00_21 are missing from 160422_haggling1) and passing a projection check: the share of projected
  joints (score > 0.2) on background-subtracted foreground, minus the same with poses from 900 frames later, is
  0.19-0.64 for every camera-session pair except 00_09 on 170221_haggling_m1 (0.03) and 170221_haggling_b1 (0.23 at
  half resolution, ~0 at full): 00_09 excluded. Calibration: the session json (OpenCV K, 5 dist, R, t; cm -> mm).
- Points: `joints19` (coco19 order, x y z score per joint, cm -> mm) -> `points3d.pq` `visible` rows where score > SCORE_MIN
  (0.1, the Panoptic toolbox's own drawing threshold), `score` = the source score. A joint at or below it (incl. the
  source's -1 = not reconstructed) gets NO row: the score is a reconstruction confidence, not a visibility assessment,
  so nothing is written `missing`. `labels = "tracked"` (machine reconstruction from 480 VGA + 31 HD views). No per-camera
  2D (`keypoints.pq` empty): 2D comes from projection.
- Identity: the source `id` is a TRACK id, re-issued whenever the reconstruction re-acquires a person (haggling sessions
  carry 18-24 ids for 3 people: one triple per ~70 s game, plus re-acquisitions). A group is therefore a maximal run of
  hd frames with a CONSTANT, non-empty set of ids, trimmed by MARGIN (15) frames at each end (entry / re-acquisition
  latency), kept if >= MIN_FRAMES (300, 10 s) remain; `animal_id` = `p<id>`. Every person in a group has rows on every
  frame and the dome holds no one else, so the session is exhaustive (no `instances.pq`).
- Pixels: each group's frames [a, b) of every chosen camera, decoded in order and re-encoded (H.264 crf 17, GOP 24, yuv420p) as
  the group's own `<cam>.mp4`; frame counts asserted. Pose file index = hd video frame index. Native 29.97 Hz.
Splits are by SESSION and RECORDING DAY (SPLITS): no day, hence no participant group, spans two splits.
No unit tests (converter rule); check with --validate, scripts/dataset_tools/check_status.py and a render.

Usage: pixi run python scripts/convert/cmu_panoptic.py --jobs 16 --validate
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from tailcyclenet import format as fmt
from _common import validate_root

RAW = Path('/groups/karashchuk/karashchuklab/animal-datasets/human/cmupanoptic')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/cmu-panoptic-8cam')
CAMS = ['00_02', '00_06', '00_14', '00_15', '00_18', '00_19', '00_24', '00_25']
SPLITS = {'train': ['170221_haggling_b1', '170221_haggling_m1', '170224_haggling_a2', '170224_haggling_b1'],
          'val': ['170228_haggling_a1'],
          'test': ['170404_haggling_a1', '170407_haggling_a1', '160422_ultimatum1']}
NAMES = ['neck', 'nose', 'body_center', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_hip', 'l_knee', 'l_ankle', 'r_shoulder',
         'r_elbow', 'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye', 'l_ear', 'r_eye', 'r_ear']
SKELETON = [(0, 1), (0, 3), (3, 4), (4, 5), (0, 2), (2, 6), (6, 7), (7, 8), (2, 12), (12, 13), (13, 14), (0, 9), (9, 10),
            (10, 11), (1, 15), (15, 16), (1, 17), (17, 18)]
FLIP = [('l_shoulder', 'r_shoulder'), ('l_elbow', 'r_elbow'), ('l_wrist', 'r_wrist'), ('l_hip', 'r_hip'),
        ('l_knee', 'r_knee'), ('l_ankle', 'r_ankle'), ('l_eye', 'r_eye'), ('l_ear', 'r_ear')]
SCORE_MIN = 0.1
MARGIN = 15
MIN_FRAMES = 300
CRF = 17
GOP = 24
DEC_THREADS = 2
FPS = 30000 / 1001


def pose_dir(sess: str) -> Path:
    """The session's coco19 folder (some sessions nest the json files under `hd/`)."""
    d = RAW / sess / 'hdPose3d_stage1_coco19'
    return d / 'hd' if (d / 'hd').is_dir() else d


def read_poses(sess: str) -> dict[int, dict[int, np.ndarray]]:
    """{hd frame: {track id: (19, 4) x y z score in cm}} for every pose file of a session."""
    out = {}
    for f in sorted(pose_dir(sess).glob('body3DScene_*.json')):
        bodies = json.loads(f.read_text())['bodies']
        out[int(f.stem.split('_')[1])] = {int(b['id']): np.asarray(b['joints19'], np.float64).reshape(19, 4)
                                          for b in bodies}
    return out


def spans(poses: dict) -> list[tuple[int, int, tuple[int, ...]]]:
    """Maximal runs of consecutive hd frames with one constant non-empty id set, trimmed by MARGIN, >= MIN_FRAMES.

    Returns [(start, end, ids)], half-open source frames."""
    runs, cur = [], None
    for t in sorted(poses):
        ids = tuple(sorted(poses[t]))
        if cur and ids == cur[2] and t == cur[1]:
            cur[1] = t + 1
        else:
            if cur:
                runs.append(tuple(cur))
            cur = [t, t + 1, ids] if ids else None
    if cur:
        runs.append(tuple(cur))
    return [(a + MARGIN, b - MARGIN, ids) for a, b, ids in runs if ids and b - a - 2 * MARGIN >= MIN_FRAMES]


def build_rig(sess: str) -> fmt.Rig:
    """The CAMS of one session's calibration json as an aniposelib CameraGroup, translation cm -> mm."""
    import cv2
    from aniposelib.cameras import Camera, CameraGroup
    cal = {c['name']: c for c in json.loads((RAW / sess / f'calibration_{sess}.json').read_text())['cameras']
           if c['type'] == 'hd'}
    cams = []
    for n in CAMS:
        c = cal[n]
        cam = Camera(matrix=np.asarray(c['K'], np.float64), dist=np.asarray(c['distCoef'], np.float64)[:5],
                     rvec=cv2.Rodrigues(np.asarray(c['R'], np.float64))[0].ravel(),
                     tvec=np.asarray(c['t'], np.float64).ravel() * 10.0, name=n)
        cam.set_size(tuple(c['resolution']))
        cams.append(cam)
    return fmt.Rig(CameraGroup(cams), offset={n: (0.0, 0.0) for n in CAMS}, moving={n: False for n in CAMS},
                   calibrated={n: True for n in CAMS})


def write_clips(src: Path, clips: list[tuple[Path, int, int]]) -> None:
    """Decode `src` once in order and write each (dst, a, b) clip of source frames [a, b) (H.264 crf CRF)."""
    import av
    clips = sorted(clips, key=lambda c: c[1])
    with av.open(str(src)) as ci:
        si = ci.streams.video[0]
        si.thread_type = 'AUTO'
        si.codec_context.thread_count = DEC_THREADS
        k, out, so, n = 0, None, None, 0
        for i, f in enumerate(ci.decode(si)):
            while k < len(clips) and i >= clips[k][2]:
                for pk in so.encode():
                    out.mux(pk)
                out.close()
                if n != clips[k][2] - clips[k][1]:
                    raise RuntimeError(f'{clips[k][0]}: {n} frames, expected {clips[k][2] - clips[k][1]}')
                k, out = k + 1, None
            if k == len(clips):
                break
            if i < clips[k][1]:
                continue
            if out is None:
                clips[k][0].parent.mkdir(parents=True, exist_ok=True)
                out = av.open(str(clips[k][0]), 'w')
                so = out.add_stream('libx264', rate=si.average_rate)
                so.width, so.height, so.pix_fmt = si.codec_context.width, si.codec_context.height, 'yuv420p'
                so.options = {'crf': str(CRF), 'preset': 'veryfast'}
                so.codec_context.gop_size = GOP
                so.thread_count = 4
                n = 0
            f = f.reformat(format='yuv420p') if f.format.name != 'yuv420p' else f
            f.pts, f.time_base = n, 1 / si.average_rate
            for pk in so.encode(f):
                out.mux(pk)
            n += 1
        if out is not None:
            for pk in so.encode():
                out.mux(pk)
            out.close()
            if n != clips[k][2] - clips[k][1]:
                raise RuntimeError(f'{clips[k][0]}: video ended after {n} of {clips[k][2] - clips[k][1]} frames')
            k += 1
    if k != len(clips):
        raise RuntimeError(f'{src}: video ended before clip {k}')


def plan(sess: str) -> dict:
    """Read one session's poses and calibration; returns its groups, labels, rig and pixel jobs (nothing written)."""
    poses = read_poses(sess)
    groups, labels = {}, {}
    for k, (a, b, ids) in enumerate(spans(poses)):
        gid = f'g{k:02d}'
        lab = fmt.empty_labels(len(ids), b - a, len(NAMES), len(CAMS), mode3d=True, animal_ids=[f'p{i:02d}' for i in ids])
        for t in range(a, b):
            for s, i in enumerate(ids):
                j = poses[t][i]
                ok = j[:, 3] > SCORE_MIN
                lab.points3d[s, t - a, ok] = j[ok, :3] * 10.0
                lab.vis3d[s, t - a, ok] = fmt.VISIBLE
        groups[gid] = fmt.Group(gid, b - a, fps=FPS, source_video=f'{sess}/hdVideos', source_frame_start=a,
                                notes=f'hd frames [{a}, {b}); track ids {list(ids)}; cameras {",".join(CAMS)} re-encoded '
                                      f'H.264 crf {CRF} GOP {GOP}')
        labels[gid] = lab
    return {'groups': groups, 'labels': labels, 'rig': build_rig(sess), 'n_pose_frames': len(poses),
            'n_ids': len({i for b in poses.values() for i in b})}


def convert(split: str, sess: str, out: Path, dry: bool, jobs: int) -> dict:
    """Write `<out>/<split>/<sess>/` (tables, calibration, re-encoded pixels); returns its counts."""
    p = plan(sess)
    lab = p['labels']
    st = {'session': sess, 'split': split, 'groups': len(p['groups']),
          'frames': int(sum(g.n_frames for g in p['groups'].values())),
          'pose_frames': p['n_pose_frames'], 'track_ids': p['n_ids'],
          'persons_per_group': sorted({len(v.animal_ids) for v in lab.values()}),
          'visible_frac': float(np.mean(np.concatenate([(v.vis3d == fmt.VISIBLE).ravel() for v in lab.values()])))}
    print(json.dumps(st), flush=True)
    if dry:
        return st
    root = out / split / sess
    prov = {'source': f'CMU Panoptic Studio {sess} (hdPose3d_stage1_coco19, HD cameras {",".join(CAMS)})',
            'annotator': '', 'annotator_tool': 'Panoptic Studio 3D reconstruction (480 VGA + 31 HD views)',
            'converter': 'scripts/convert/cmu_panoptic.py', 'created': date.today().isoformat(),
            'licence': 'CMU Panoptic Studio: research / non-commercial use only'}
    fmt.write_session(root, mode='3d', units='mm', label_source='tracked', names=NAMES, rig=p['rig'],
                      groups=p['groups'], labels=lab, skeleton=[(NAMES[i], NAMES[j]) for i, j in SKELETON],
                      flip_pairs=FLIP, provenance=prov)
    todo = [(RAW / sess / 'hdVideos' / f'hd_{c}.mp4',
             [(root / 'groups' / g / f'{c}.mp4', g_.source_frame_start, g_.source_frame_start + g_.n_frames)
              for g, g_ in p['groups'].items()]) for c in CAMS]
    with ProcessPoolExecutor(max(1, min(jobs, len(todo)))) as ex:
        list(ex.map(write_clips, *zip(*todo)))
    return st


def main() -> int:
    """Parse the CLI, convert the requested sessions, write the README, optionally validate; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', help='subset of session names')
    ap.add_argument('--sessions', nargs='*', metavar='SPLIT=SESSION',
                    help='replace SPLITS (for a separate root, e.g. --out .../cmu-panoptic-8cam-dev '
                         '--sessions val=170228_haggling_b1)')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--jobs', type=int, default=4,
                    help='cameras re-encoded in parallel per session; threads = jobs x (4 encode + DEC_THREADS decode)')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--no-image-check', action='store_true')
    a = ap.parse_args()
    if a.clean and a.out.exists() and not a.dry_run:
        shutil.rmtree(a.out)
    splits = SPLITS
    if a.sessions:
        splits = {}
        for kv in a.sessions:
            sp, s = kv.split('=')
            if sp not in ('train', 'val', 'test'):
                raise SystemExit(f'--sessions {kv}: split must be train/val/test')
            splits.setdefault(sp, []).append(s)
    stats = [convert(sp, s, a.out, a.dry_run, a.jobs) for sp, ss in splits.items() for s in ss
             if not a.only or s in a.only]
    if not a.dry_run and not a.only:
        rows = '\n'.join(f'| {s["split"]} | {s["session"]} | {s["groups"]} | {s["frames"]} | {s["persons_per_group"]} | '
                         f'{s["track_ids"]} | {s["visible_frac"]:.3f} |' for s in stats)
        (a.out / 'README.md').write_text(
            '# cmu-panoptic-8cam\n\nCMU Panoptic Studio multi-person 3D (coco19), 8 HD cameras, written by '
            '`scripts/convert/cmu_panoptic.py` (see its docstring for the camera choice, the group rule and the status '
            'rule). Licence: research / non-commercial only.\n\nCameras: ' + ', '.join(CAMS) + '\n\nSplit by session '
            'and recording day:\n\n| split | session | groups | frames | persons/group | source track ids | visible '
            'joint frac |\n|---|---|---|---|---|---|---|\n' + rows + '\n')
        (a.out / 'stats.json').write_text(json.dumps(stats, indent=1))
    if a.validate and not a.dry_run:
        return 1 if validate_root(a.out, check_images=not a.no_image_check) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
