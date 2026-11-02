#!/usr/bin/env python
"""Convert the idtracker.ai 2026 benchmark (human-validated trajectories) to tailcycle 2D roots.

Source (raw store): animal-datasets/mouse/idtrackerai-mice/<video>/ and animal-datasets/fish/idtrackerai-zebrafish/<video>/,
each holding `<video>.avi` (rawvideo) + `validated.npy` (idtracker.ai v5.2.5 trajectories after full human validation with the
idtracker.ai Validator; Torrents et al., eLife 2026). See the raw folders' README.md.

Output: one 2D session per video under `<out>/test/<video>/`, one group `g0` = the whole recording.
- Pixels: the rawvideo .avi (10-16 GB) is re-encoded ONCE to H.264 gray (`groups/g0/cam0.mp4`, crf 12; the sources are grayscale palette video and some have odd heights, which 4:2:0 cannot encode; every frame kept,
  frame count asserted against the source and against the trajectory length). This is a transform, so it is a file, not a link.
- Labels: ONE keypoint, `centroid` (the idtracker.ai blob centroid, validated identity), written `projected` (a centroid is
  a position, not a landmark anyone judged visible) ONLY on frames where the animal is an INDIVIDUAL blob: frames covered by
  an individual fragment whose `groundtruth_identity` (list_of_fragments.json, computed by the authors from the Validator
  output) is that animal. Inside CROSSINGS (merged blobs) the source trajectory is a straight-line interpolation between the
  entry and exit positions, not a measurement (checked on mice_2_1: constant per-frame displacement across the crossing), so
  those frames get NO row -- identity is asserted before and after each crossing, never a position inside it. This is the
  paper's own "accuracy without crossings" convention. Each kept position is cross-checked against its fragment's start
  position: a few fragments next to crossings sit up to ~0.1 BL off (validation edits; counted and printed);
  a fragment whose start is > 0.25 BL from its own validated trajectory or nearer another animal's (the fragment's
  groundtruth_identity and validated.npy DISAGREE about who it is -- seen on mice_2_2, 26 short fragments / 228 frames
  around a few crossings, and mice_4_1, 2 fragments / 3 frames) gets NO rows; counts are printed and go in the README.
  `labels = "annotated"`: identities are human-validated in every frame (the source is a tracker output corrected by a human
  with the Validator, which the authors use as ground truth).
- No `instances.pq`: the source has every animal in every frame (exhaustive), and it has no boxes.
- `groups.pq` notes carry `body_length_px=<source body_length>` (idtracker's own BL, full-resolution px) for scoring radii.
- No unit tests (converter rule); check with --validate, scripts/dataset_tools/check_status.py and a render.

Usage:
  pixi run python scripts/convert/idtrackerai.py --species mice --validate
  pixi run python scripts/convert/idtrackerai.py --species zebrafish --validate
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

from tailcyclenet import format as fmt
from _common import validate_root

RAW = Path('/groups/karashchuk/karashchuklab/animal-datasets')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
SPECIES = {
    'mice': (RAW / 'mouse/idtrackerai-mice', OUT / 'idtrackerai-mice',
             ['mice_2_1', 'mice_2_2', 'mice_2_3', 'mice_2_4', 'mice_4_1', 'mice_4_2']),
    'zebrafish': (RAW / 'fish/idtrackerai-zebrafish', OUT / 'idtrackerai-zebrafish',
                  ['zebrafish_2', 'zebrafish_5', 'zebrafish_7', 'zebrafish_8', 'zebrafish_10_1', 'zebrafish_10_2',
                   'zebrafish_10_3', 'zebrafish_10_4', 'zebrafish_20']),
}


def n_video_frames(path: Path) -> int:
    """Decoded frame count (PyAV frame table, the reader the loader uses)."""
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        return sum(1 for p in c.demux(st) if p.size)


def encode(src: Path, dst: Path, n_expected: int) -> None:
    """Re-encode the rawvideo to H.264 keeping every frame; refuse a frame-count mismatch."""
    if dst.exists() and n_video_frames(dst) == n_expected:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.tmp.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-map', '0:v:0', '-fps_mode', 'passthrough',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'gray', '-g', '50',
                    str(tmp)], check=True)
    n = n_video_frames(tmp)
    if n != n_expected:
        raise RuntimeError(f'{tmp}: {n} frames after encoding, expected {n_expected}')
    tmp.rename(dst)


def video_labels(raw: Path, name: str) -> dict:
    """Build one video's `centroid` labels (individual-blob frames only) without writing anything.

    Inputs: the raw species folder and the video name. `validated.npy` trajectories are (T, N, 2) px. A fragment whose
    groundtruth identity and the validated trajectory disagree about who it is gets no rows (counted).
    Outputs: a dict with `lab` (fmt.Labels), `group` (fmt.Group), `rig`, `provenance`, `avi` (source video path),
    `traj` ((T, N, 2) validated trajectories, every frame), `session` (session.json), `body_length`.
    Side effects: prints the per-video counts. Shared by `convert_video` and `blob_split.py` (the split roots).
    """
    src = raw / name
    d = np.load(src / 'validated.npy', allow_pickle=True).item()
    traj = np.asarray(d['trajectories'], np.float64)
    T, N, _ = traj.shape
    sj = json.loads((src / 'session.json').read_text())
    W, H = int(sj['original_width']), int(sj['original_height'])
    avi = src / f'{name}.avi'
    n_src = n_video_frames(avi)
    if n_src != T:
        raise RuntimeError(f'{name}: video has {n_src} frames, trajectories {T} -- refusing to guess the alignment')
    fps, bl = float(d['frames_per_second']), float(d['body_length'])
    finite = np.isfinite(traj).all(-1)
    lf = json.loads((src / 'list_of_fragments.json').read_text())['fragments']
    rr = float(sj.get('resolution_reduction', 1.0))
    labels_ = [int(v) for v in d['identities_labels']]
    indiv = np.zeros((T, N), bool)
    n_corr = n_off = n_dis = dis_frames = 0
    for f in lf:
        g = int(f.get('groundtruth_identity') or 0)
        if not f['is_an_individual'] or g <= 0:
            continue
        a = labels_.index(g)
        n_corr += int(f.get('identity') != g)
        p0 = np.asarray(f['start_position']) / rr
        dist = np.linalg.norm(p0 - traj[f['start_frame']], axis=-1)
        n_off += int(dist[a] > 2.0)
        if dist[a] > 0.25 * bl or np.nanargmin(dist) != a:
            n_dis += 1
            dis_frames += f['end_frame'] - f['start_frame']
            continue
        indiv[f['start_frame']:f['end_frame'], a] = True
    finite &= indiv
    print(f'{name}: T={T} N={N} {W}x{H} fps={fps} BL={bl:.1f}px individual(labelled)={finite.mean():.4f} '
          f'crossing/unlabelled={1 - finite.mean():.4f} fragments corrected by the Validator={n_corr} '
          f'fragment starts moved >2px by validation={n_off} '
          f'DISAGREEING fragments dropped={n_dis} ({dis_frames} animal-frames)')
    rig = fmt.nominal_rig('cam0', (W, H))
    ids = [f'id{int(v):02d}' for v in d['identities_labels']]
    lab = fmt.empty_labels(N, T, 1, 1, mode3d=False, animal_ids=ids)
    lab.points2d[:, :, 0, 0, :] = np.transpose(traj, (1, 0, 2)).astype(np.float32)
    lab.vis2d[:, :, 0, 0] = np.where(finite.T, fmt.PROJECTED, fmt.UNLABELED)
    lab.points2d[lab.vis2d == fmt.UNLABELED] = np.nan
    g = fmt.Group('g0', T, fps=fps, source_video=f'{name}.avi', notes=f'body_length_px={bl:.2f}; crossing_frac={1 - finite.mean():.4f}; no rows inside crossings; dropped_disagreeing_animal_frames={dis_frames}')
    prov = {'source': f'idtracker.ai 2026 benchmark ({name}); validated.npy (idtracker.ai v{d["version"]} '
                      'trajectories, fully human-validated with the idtracker.ai Validator)',
            'annotator': '', 'annotator_tool': 'idtracker.ai Validator',
            'converter': 'scripts/convert/idtrackerai.py',
            'created': date.today().isoformat(), 'licence': 'none stated (academic use)'}
    return {'lab': lab, 'group': g, 'rig': rig, 'provenance': prov, 'avi': avi, 'traj': traj, 'session': sj,
            'body_length': bl}


def convert_video(raw: Path, out: Path, name: str, dry: bool) -> None:
    """Write one video's 2D session: `centroid` rows on individual-blob frames only, plus the re-encoded pixels.

    Inputs: the raw species folder, the output root, the video name, and `dry` (report only).
    Side effects: writes `<out>/test/<name>/` and its mp4.
    """
    v = video_labels(raw, name)
    if dry:
        return
    sess = out / 'test' / name
    fmt.write_session(sess, mode='2d', units='px', label_source='annotated', names=['centroid'], rig=v['rig'],
                      groups={'g0': v['group']}, labels={'g0': v['lab']}, flip_pairs=[], provenance=v['provenance'])
    encode(v['avi'], sess / 'groups' / 'g0' / 'cam0.mp4', v['group'].n_frames)


def main() -> int:
    """Parse the CLI, convert the requested videos, optionally validate the root; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--species', choices=sorted(SPECIES), required=True)
    ap.add_argument('--only', nargs='*', help='subset of video names')
    ap.add_argument('--out', type=Path, default=None, help='default: tailcycle-datasets/idtrackerai-<species>')
    ap.add_argument('--clean', action='store_true', help='remove the output first')
    ap.add_argument('--dry-run', action='store_true', help='report, write nothing')
    ap.add_argument('--validate', action='store_true', help='validate after writing')
    ap.add_argument('--no-image-check', action='store_true')
    a = ap.parse_args()
    raw, out, names = SPECIES[a.species]
    out = a.out or out
    if a.clean and out.exists() and not a.dry_run:
        shutil.rmtree(out)
    for name in (a.only or names):
        convert_video(raw, out, name, a.dry_run)
    if a.validate and not a.dry_run:
        return 1 if validate_root(out, check_images=not a.no_image_check) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
