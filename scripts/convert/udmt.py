#!/usr/bin/env python
"""Convert the UDMT multi-animal recordings + manual identity GT to one tailcycle 2D root.

Source (raw store): animal-datasets/mouse/udmt-multianimal/ (Zenodo 14580256 videos, Zenodo 20355567 `Annotation/`,
CC BY 4.0; see that folder's README.md). The C. elegans videos of the same record live in animal-datasets/worm/udmt-celegans/
and convert to their own root `udmt-celegans` (`--dataset udmt-celegans`); their annotations are in the same archive.
Zenodo 20355567 has NO annotation for the two Betta splendens (fish_2) videos, so they are downloaded
(animal-datasets/fish/udmt-betta/) but not converted.

GT format (MOT): `frame, id, x, y, w, h, 1, -1, -1, -1`, 1-based annotation index, fixed 65x65 px boxes placed around each
animal by a human, SPARSE: annotation k sits on video frame (k-1)*STEP with STEP = 300 (mice), 150 (flies), 600 (94 Hz
mice, C. elegans: 150), per the record description, plus a per-video OFFSET. Offset 0 was checked visually on mice_7
annotation 50 (boxes sit on the mice at (k-1)*300; at (k-1)*300+299 they do not). celegans_7's video is 1,014 frames
shorter than celegans_6 while both annotation sequences hold 98 checkpoints at step 150: its video starts 1,014 frames
into the annotated sequence (offset -1014, = 14,569 - 13,555 frames). Measured: the share of clicked points within 6 px of
a dark-foreground pixel over annotations 8-98 is 0.31-0.49 per quarter at offset 0 and 0.69-0.78 at -1014, flat on
-1030..-975 (worms are slow at 10 Hz), the same plateau shape celegans_6 shows around its own offset 0; annotations 1-7
fall before the first video frame and are dropped.

Output `<out>/test/<session>/`, one session per video (sizes differ), one group `g0` = the whole video (symlinked mp4):
- `instances.pq`: a `labeled` row with the source box for every annotated animal-frame (every annotated frame lists every
  animal, asserted), so an annotated camera-frame is complete; frames without annotation have no rows (no determination).
- `keypoints.pq`: ONE keypoint `centroid` = the box centre, `projected` (a position, no visibility claim).
- `labels = "annotated"`.
No unit tests (converter rule); check with --validate, scripts/dataset_tools/check_status.py and a render.

Usage: pixi run python scripts/convert/udmt.py --validate
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tailcyclenet import format as fmt
from _common import validate_root

RAW = Path('/groups/karashchuk/karashchuklab/animal-datasets/mouse/udmt-multianimal')
WORM = Path('/groups/karashchuk/karashchuklab/animal-datasets/worm/udmt-celegans')
TC = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')
OUT = TC / 'udmt-multianimal'
# session id -> (annotation sequence, video file (relative to RAW, or absolute), step, species[, offset])
SESSIONS = {
    'mice3': ('3-bbnc-mice-gt-7min', 'mice_3_652x636_67Hz_6min57s.mp4', 300, 'mouse'),
    'mice5': ('5-bbnc-mice-gt-7min', 'mice_5_652x636_66Hz_8min4s.mp4', 300, 'mouse'),
    'mice5_94hz': ('5-mice-full-94hz-15min-g2', 'mice_5_416x444_94Hz_11min30s.mp4', 600, 'mouse'),
    'mice7': ('7-bbnc-mice-gt-7min', 'mice_7_664x630_67Hz_7min21s.mp4', 300, 'mouse'),
    'mice10': ('10-mice-1207-full', 'mice_10_600x686_40Hz_6min.mp4', 300, 'mouse'),
    'whitemice5': ('5-white-mice-full-gt-5min', 'whitemice_5_704x642_70Hz_4min13s.mp4', 300, 'mouse'),
    'rat1mice2': ('1-rat-2-mice-2-full-gt-2min', 'rat1&mice2_664x630_68Hz_2min6s.mp4', 300, 'rat+mouse'),
    'flies12': ('12-bbnc-flies-fullres-50hz-gt-4min', 'drosophila_12_592x592_55Hz_3min37s.mp4', 150, 'fly'),
    'flies17': ('17-bbnc-flies-50hz-gt-10min', 'drosophila_17_560x624_54Hz_8min35s.mp4', 150, 'fly'),
    'flies18': ('18-bbnc-flies-fullres-50hz-gt-5min', 'drosophila_18_560x624_54Hz_4min32s.mp4', 150, 'fly'),
    'flies18_illum': ('18-bbnc-flies-50hz-illumination-change-gt-2min',
                      'drosophila_18_illumination_change_560x624_54Hz_2min10s.mp4', 150, 'fly'),
    'flies23': ('23-bbnc-flies-50hz-gt-10min', 'drosophila_23_560x624_54Hz_9min59s.mp4', 150, 'fly'),
    'celegans6': ('6-celegans-10hz-g1-gt-25min', WORM / 'celegans_6_614x612_10Hz_24min16s.mp4', 150, 'C. elegans'),
    'celegans7': ('7-celegans-10hz-g1-gt-25min', WORM / 'celegans_7_614x612_10Hz_22min35s.mp4', 150, 'C. elegans', -1014),
}
# output root -> its sessions (one root per species family; `udmt-multianimal` is the original mice + flies root)
DATASETS = {'udmt-multianimal': [k for k in SESSIONS if not k.startswith('celegans')],
            'udmt-celegans': ['celegans6', 'celegans7']}


def probe(path: Path) -> tuple[int, float, int, int]:
    """Frame count from the PyAV frame table, fps, width, height."""
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        n = sum(1 for p in c.demux(st) if p.size)
        return n, float(st.average_rate), st.codec_context.width, st.codec_context.height


def session_labels(sid: str) -> dict:
    """Build one UDMT session's labels (source 65 px boxes + box-centre `centroid` on the annotated frames).

    Inputs: the session id (key of SESSIONS). Annotations past the last video frame are dropped and listed in the group
    notes. Outputs: a dict with `lab`, `group`, `rig`, `provenance`, `video` (source mp4 path), `frames` (the annotated
    video frames, sorted). Side effects: prints the per-session counts. Shared by `convert` and `blob_split.py`.
    """
    seq, video, step, species, *rest = SESSIONS[sid]
    offset = rest[0] if rest else 0
    gt = pd.read_csv(RAW / 'Annotation' / seq / 'gt' / 'gt.txt', header=None)
    vid = RAW / video
    video = vid.name
    T, fps, W, H = probe(vid)
    ids = sorted(gt[1].unique())
    N = len(ids)
    per = gt.groupby(0)[1].nunique()
    if (per != N).any():
        raise RuntimeError(f'{sid}: annotation frames with fewer than {N} animals: {per[per != N].index.tolist()[:10]}')
    if gt.duplicated([0, 1]).any():
        raise RuntimeError(f'{sid}: duplicate (frame, id)')
    frames = (gt[0].to_numpy() - 1) * step + offset
    keep = (frames >= 0) & (frames < T)
    dropped = sorted(set(gt[0][~keep]))
    print(f'{sid}: T={T} fps={fps:.2f} {W}x{H} N={N} annotated={gt[0].nunique()} step={step} offset={offset} '
          f'dropped(outside the video)={dropped} box={sorted(set(gt[4]) | set(gt[5]))}')
    gt, frames = gt[keep], frames[keep]
    lab = fmt.empty_labels(N, T, 1, 1, mode3d=False, animal_ids=[f'id{int(i):02d}' for i in ids])
    lab.boxes = np.full((N, T, 1, 4), np.nan, np.float32)
    lab.instance = np.full((N, T, 1), fmt.INST_NONE, np.int8)
    s = np.searchsorted(ids, gt[1].to_numpy())
    x0, y0 = gt[2].to_numpy(np.float32), gt[3].to_numpy(np.float32)
    w, h = gt[4].to_numpy(np.float32), gt[5].to_numpy(np.float32)
    lab.boxes[s, frames, 0] = np.stack([x0, y0, x0 + w, y0 + h], -1)
    lab.instance[s, frames, 0] = fmt.INST_LABELED
    lab.points2d[s, frames, 0, 0] = np.stack([x0 + w / 2, y0 + h / 2], -1)
    lab.vis2d[s, frames, 0, 0] = fmt.PROJECTED
    notes = (f'species={species}; sparse GT every {step} frames'
             + (f' from video frame {offset} (annotation k -> frame (k-1)*{step}{offset:+d})' if offset else '')
             + f'; box_px={int(w[0])}'
             + (f'; dropped annotation index {dropped} (outside the video)' if dropped else ''))
    g = fmt.Group('g0', T, fps=fps, source_video=video, notes=notes)
    prov = {'source': f'UDMT (Zenodo 14580256 video {video}; Zenodo 20355567 Annotation/{seq})',
            'annotator': '', 'annotator_tool': 'manual MOT boxes (UDMT authors)',
            'converter': 'scripts/convert/udmt.py', 'created': date.today().isoformat(), 'licence': 'CC BY 4.0'}
    return {'lab': lab, 'group': g, 'rig': fmt.nominal_rig('cam0', (W, H)), 'provenance': prov, 'video': vid,
            'frames': np.unique(frames)}


def convert(sid: str, out: Path, dry: bool) -> None:
    """Write one UDMT session: labeled boxes + `centroid` rows on the annotated frames, and link the video.

    Inputs: the session id (key of SESSIONS), the output root, `dry` (report only). Side effects: writes
    `<out>/test/<sid>/`.
    """
    v = session_labels(sid)
    if dry:
        return
    sess = out / 'test' / sid
    fmt.write_session(sess, mode='2d', units='px', label_source='annotated', names=['centroid'], rig=v['rig'],
                      groups={'g0': v['group']}, labels={'g0': v['lab']}, flip_pairs=[], provenance=v['provenance'])
    (sess / 'groups' / 'g0').mkdir(parents=True, exist_ok=True)
    fmt.link(sess / 'groups' / 'g0' / 'cam0.mp4', v['video'])


def main() -> int:
    """Parse the CLI, convert the requested sessions, optionally validate the root; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=sorted(DATASETS), default='udmt-multianimal')
    ap.add_argument('--only', nargs='*', choices=sorted(SESSIONS))
    ap.add_argument('--out', type=Path, default=None, help='default: tailcycle-datasets/<dataset>')
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--no-image-check', action='store_true')
    a = ap.parse_args()
    a.out = a.out or TC / a.dataset
    if a.clean and a.out.exists() and not a.dry_run:
        shutil.rmtree(a.out)
    for sid in (a.only or DATASETS[a.dataset]):
        convert(sid, a.out, a.dry_run)
    if a.validate and not a.dry_run:
        return 1 if validate_root(a.out, check_images=not a.no_image_check) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
