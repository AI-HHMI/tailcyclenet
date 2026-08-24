#!/usr/bin/env python
"""Convert CalMS21 task 1 into the tailcycle-dataset format.

    pixi run python scripts/convert_calms21.py --dry-run

Only task 1 ships pixels; one video = one session = one group (2 mice, 7 MARS keypoints),
pixels symlinked. Keypoints come from the official 2021 json; `task1_MARS_features/` (a later
re-run whose keypoints differ by ~2 px) is used only for bbox/fps/vid_name.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt, video

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/CalMS21')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/calms21')

# readme order; mouse 0 = resident (black), mouse 1 = intruder (white).
NAMES = ['nose', 'left_ear', 'right_ear', 'neck', 'left_hip', 'right_hip', 'tail_base']
FLIP = [['left_ear', 'right_ear'], ['left_hip', 'right_hip']]
SKELETON = [['nose', 'left_ear'], ['nose', 'right_ear'], ['left_ear', 'neck'],
            ['right_ear', 'neck'], ['neck', 'left_hip'], ['neck', 'right_hip'],
            ['left_hip', 'tail_base'], ['right_hip', 'tail_base']]
ANIMALS = ['resident', 'intruder']
W, H = 1024, 570
# Last 7 of the 70 train videos, by sorted name.
N_VAL = 7


def rig() -> fmt.Rig:
    """One uncalibrated 1024x570 camera. Same shape as convert_v4.rig_from_pixels."""
    from aniposelib.cameras import CameraGroup
    cam = fmt.nominal_camera('cam0', (W, H))
    return fmt.Rig(CameraGroup([cam]), offset={'cam0': (0.0, 0.0)},
                   moving={'cam0': False}, calibrated={'cam0': False})


def build(seq: dict, npz, T: int) -> tuple[fmt.Labels, float]:
    """One CalMS21 sequence -> dense labels. Returns (labels, fraction of kpts inside their box).

    `keypoints` is (T, mouse, xy, part); the format wants (S, T, K, C, 2).

    Every status is `visible`, which is only defensible because a non-finite check guards it:
    MARS has no occlusion channel, so nothing here is an assessment (kept `visible`, not
    `projected`, because a tracked session with zero `missing` rows withholds its visibility
    target anyway). Boxes are [x0,y0,x1,y1] normalised by (W,H), checked rather than assumed
    because they come from a different MARS run than the keypoints.
    """
    kp = np.asarray(seq['keypoints'], np.float32)
    assert kp.shape == (T, 2, 2, len(NAMES)), f'keypoints {kp.shape}, expected {(T, 2, 2, 7)}'
    assert np.isfinite(kp).all(), 'keypoints carry non-finite values'

    lab = fmt.empty_labels(2, T, len(NAMES), 1, mode3d=False, animal_ids=ANIMALS)
    lab.points2d[:, :, :, 0, :] = kp.transpose(1, 0, 3, 2)
    lab.vis2d[:] = fmt.VISIBLE

    box = np.asarray(npz['bbox'], np.float32).transpose(0, 2, 1)
    box[..., 0::2] *= W
    box[..., 1::2] *= H
    lab.boxes = np.ascontiguousarray(box[:, :, None, :])
    lab.instance = np.full((2, T, 1), fmt.INST_LABELED, np.int8)

    p = lab.points2d[:, :, :, 0, :]
    lo, hi = box[:, :, None, :2], box[:, :, None, 2:]
    inside = float(((p >= lo - 1.0) & (p <= hi + 1.0)).all(-1).mean())
    return lab, inside


def convert(out_root: Path, dry_run: bool, max_seqs: int | None) -> None:
    """Convert both CalMS21 task-1 splits: one session per video.

    Inputs: out_root -- output root; sessions land at out_root/<split>/<stem>.
            dry_run -- report shapes, write nothing.
            max_seqs -- cap sequences per split, or None.
    Side effects: writes sessions + symlinked mp4s; prints per-sequence lines.

    The labelled-frame count is asserted to equal the video frame count (they agree on all 89
    sequences, so a disagreement is a broken file). Two npz files carry a broken fps (a
    timestamp regression) and fall back to the container's nominal rate. The 0.90 inside-box
    floor catches a swapped bbox axis order, which reads near 0% against a 97.5% worst case.
    """
    r = rig()
    val_stems: set[str] = set()

    for src_split in ('train', 'test'):
        path = SRC / 'task1_classic_classification' / f'calms21_task1_{src_split}.json'
        print(f'== reading {path.name} ({path.stat().st_size / 1e9:.2f} GB)')
        blob = json.load(open(path))
        assert len(blob) == 1, f'expected one annotator group, got {list(blob)}'
        data = blob[next(iter(blob))]
        stems = sorted(k.split('/')[-1] for k in data)
        if src_split == 'train':
            val_stems = set(stems[-N_VAL:])
        if max_seqs:
            stems = stems[:max_seqs]

        for stem in stems:
            split = 'val' if stem in val_stems else src_split
            seq = data[f'task1/{src_split}/{stem}']
            mp4 = (SRC / 'task1_videos_mp4' / src_split / f'{stem}.mp4').resolve()
            npz = np.load(SRC / 'task1_MARS_features' / 'calms21_task1_features' / src_split /
                          f'{stem}.npz', allow_pickle=True)

            vr = video.open_reader(str(mp4))
            T = len(vr)
            n_lab = len(seq['keypoints'])
            assert n_lab == T, f'{stem}: {n_lab} labelled frames, {T} video frames'

            lab, inside = build(seq, npz, T)
            fps = float(npz['fps'])
            if not 25.0 <= fps <= 35.0:
                fps = float(vr.fps)
                print(f'   ! {stem}: npz fps is {float(npz["fps"]):.2f}; using the container\'s '
                      f'nominal {fps:.2f}')
            group = fmt.Group(stem, T, fps=fps, source_video=str(mp4), source_frame_start=0,
                              source_frame_step=1, notes=str(npz['vid_name']))
            vr.close()
            print(f'   {split}/{stem}: {T} frames @ {fps:.2f} fps, '
                  f'{2 * T * len(NAMES)} kpt rows, {inside * 100:.2f}% of kpts inside their box')
            if inside < 0.90:
                print(f'   ! {stem}: only {inside * 100:.2f}% of keypoints fall inside their '
                      'box -- check the bbox axis order')
            if dry_run:
                continue

            dst = out_root / split / stem
            (dst / 'groups' / stem).mkdir(parents=True, exist_ok=True)
            fmt.link(dst / 'groups' / stem / 'cam0.mp4', mp4)
            fmt.write_session(
                dst, mode='2d', units='px', label_source='tracked', names=NAMES, rig=r,
                groups={stem: group}, labels={stem: lab},
                skeleton=SKELETON, flip_pairs=FLIP,
                provenance={
                    'source': f'CalMS21/task1_classic_classification/calms21_task1_{src_split}',
                    'annotator': '', 'annotator_tool': 'scripts/convert_calms21.py',
                    'pose_tool': 'MARS (Segalin et al 2020), stacked hourglass',
                    'boxes': 'CalMS21/task1_MARS_features (a LATER MARS re-run; its keypoints '
                             'differ from the json by ~2 px mean, only bbox/fps/vid_name used)',
                    'visibility': 'asserted (MARS emits no occlusion channel -- every point is '
                                  'written visible, so this root supervises position only and '
                                  'its visibility target asserts nothing anyone observed)',
                    'dropped': 'per-frame behaviour annotations and MARS confidence scores; both '
                               'recoverable from the source json',
                    'animal_id_source': 'readme: mouse 0 = resident (black), 1 = intruder (white)',
                })
            del data[f'task1/{src_split}/{stem}']
        del blob, data


def main() -> None:
    """Convert CalMS21 task 1 into the tailcycle dataset; exit 0/1.

    Inputs: argv (via argparse): --out, --max-seqs, --dry-run, --validate,
            --clean.
    Side effects: writes the dataset under --out; prints progress and validation.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--max-seqs', type=int, default=None, help='cap sequences per split')
    ap.add_argument('--dry-run', action='store_true', help='report shapes, write nothing')
    ap.add_argument('--validate', action='store_true', help='validate after writing')
    ap.add_argument('--clean', action='store_true', help='remove the output dir first')
    args = ap.parse_args()

    if args.clean and args.out.exists() and not args.dry_run:
        shutil.rmtree(args.out)
    convert(args.out, args.dry_run, args.max_seqs)

    if args.validate and not args.dry_run:
        errs = fmt.validate_dataset(fmt.load_dataset(args.out))
        hard = [e for e in errs if 'WARNING' not in e]
        for e in errs:
            print(('  WARN ' if 'WARNING' in e else '  FAIL ') + e)
        print(f'calms21: {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')
        sys.exit(1 if hard else 0)


if __name__ == '__main__':
    main()
