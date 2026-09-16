#!/usr/bin/env python
"""Convert the COCO-style APTv2 release to a Tailcycle 2D dataset."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tailcyclenet import format as fmt

NAMES = [
    'left_eye', 'right_eye', 'nose', 'neck', 'root_of_tail',
    'left_shoulder', 'left_elbow', 'left_front_paw',
    'right_shoulder', 'right_elbow', 'right_front_paw',
    'left_hip', 'left_knee', 'left_back_paw',
    'right_hip', 'right_knee', 'right_back_paw',
]
SKELETON = [
    [NAMES[a - 1], NAMES[b - 1]]
    for a, b in [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6],
                 [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12],
                 [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
]
FLIP_PAIRS = [
    ['left_eye', 'right_eye'], ['left_shoulder', 'right_shoulder'],
    ['left_elbow', 'right_elbow'], ['left_front_paw', 'right_front_paw'],
    ['left_hip', 'right_hip'], ['left_knee', 'right_knee'],
    ['left_back_paw', 'right_back_paw'],
]
SPLIT_FILES = {
    'train': 'train_annotations.json',
    'val': 'val_annotations.json',
    'test': 'test_annotations.json',
}


def load_json(path: Path) -> dict:
    """Load a JSON document from disk."""
    with path.open() as f:
        return json.load(f)


def decode_keypoints(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Decode COCO triplets into Tailcycle coordinates and visibility codes."""
    arr = np.asarray(values, dtype=float).reshape(len(NAMES), 3)
    xy = np.full((len(NAMES), 2), np.nan, np.float32)
    status = np.full(len(NAMES), fmt.UNLABELED, np.int8)
    visible = arr[:, 2] == 2
    xy[visible] = arr[visible, :2].astype(np.float32)
    status[visible] = fmt.VISIBLE
    return xy, status


def bbox_xywh_to_xyxy(box: list[float]) -> np.ndarray:
    """Convert a COCO [x, y, width, height] box to Tailcycle coordinates."""
    x, y, w, h = map(float, box)
    return np.asarray([x, y, x + w, y + h], np.float32)


def dedupe_annotations(annotations: list[dict], images: dict[int, dict]) -> tuple[list[dict], int]:
    """Keep the most-visible record for duplicate (video, frame, track) keys."""
    chosen: dict[tuple[int, int, int], dict] = {}
    for a in annotations:
        image = images[int(a['image_id'])]
        key = (int(a['video_id']), int(Path(image['file_name']).stem), int(a['track_id']))
        score = sum(int(v) == 2 for v in a['keypoints'][2::3])
        old = chosen.get(key)
        if old is None or score > sum(int(v) == 2 for v in old['keypoints'][2::3]):
            chosen[key] = a
    return list(chosen.values()), len(annotations) - len(chosen)


def frame_name(frame: int) -> str:
    """Map an APTv2 four-digit source frame to the required six-digit Tailcycle name."""
    return f'{frame:06d}.jpg'


def preflight(src: Path, docs: dict[str, dict]) -> tuple[dict[int, dict], dict[int, list[dict]], dict[int, Path], int]:
    """Check the shared COCO image index and return image/video lookup tables.

    Verify all source clip directories and their six-digit output mapping. A few release JSONs
    repeat one (video, frame, track) record; prefer the record with actual keypoints (the
    duplicate is commonly an all-zero COCO placeholder), retaining one representable animal row
    rather than creating duplicate Tailcycle keys.
    """
    base = docs['train']
    cats = base['categories']
    if len(cats) != 30:
        raise SystemExit(f'expected 30 categories, got {len(cats)}')
    for c in cats:
        if c.get('keypoints') != NAMES:
            raise SystemExit(f"category {c['name']!r} has a different keypoint axis")
        if c.get('skeleton') != [[a + 1, b + 1] for a, b in
                                 [(NAMES.index(x), NAMES.index(y)) for x, y in SKELETON]]:
            raise SystemExit(f"category {c['name']!r} has a different skeleton")
    images = {int(i['id']): i for i in base['images']}
    if len(images) != len(base['images']):
        raise SystemExit('duplicate image ids')
    for split, d in docs.items():
        if d['images'] != base['images']:
            raise SystemExit(f'{split}: image index differs from train_annotations.json')

    by_video: dict[int, list[dict]] = defaultdict(list)
    for im in images.values():
        parts = Path(im['file_name']).parts
        if len(parts) != 4 or parts[-1].split('.')[0] != f'{int(parts[-1].split(".")[0]):04d}':
            raise SystemExit(f"unexpected APTv2 image path {im['file_name']!r}")
        if (im['width'], im['height']) != (1920, 1080):
            raise SystemExit(f"unexpected image size for {im['file_name']}: {im['width']}x{im['height']}")
        by_video[int(im['video_id'])].append(im)
    video_dirs: dict[int, Path] = {}
    for vid, ims in by_video.items():
        ims.sort(key=lambda x: int(Path(x['file_name']).stem))
        nums = [int(Path(x['file_name']).stem) for x in ims]
        if nums != list(range(len(ims))):
            raise SystemExit(f'video {vid}: frame indices are not contiguous: {nums[:10]}')
        rel = Path(ims[0]['file_name'])
        d = src / 'data' / rel.parts[0] / rel.parts[1] / rel.parts[2]
        if not d.is_dir():
            raise SystemExit(f'video {vid}: missing source directory {d}')
        files = sorted(d.glob('*.jpg'))
        if len(files) != len(ims):
            raise SystemExit(f'video {vid}: {len(files)} source files, {len(ims)} image records')
        if [int(p.stem) for p in files] != nums:
            raise SystemExit(f'video {vid}: source files do not match image records')
        video_dirs[vid] = d

    deduped = 0
    for split, d in docs.items():
        d['annotations'], n_removed = dedupe_annotations(d['annotations'], images)
        deduped += n_removed

    vis = defaultdict(int)
    crowd = defaultdict(int)
    ann_count = 0
    for split, d in docs.items():
        for a in d['annotations']:
            kp = a['keypoints']
            if len(kp) != 3 * len(NAMES):
                raise SystemExit(f"annotation {a['id']}: wrong keypoint length")
            for value in kp[2::3]:
                vis[value] += 1
            crowd[int(a['is_crowd'])] += 1
            if int(a['image_id']) not in images or int(a['video_id']) != int(images[int(a['image_id'])]['video_id']):
                raise SystemExit(f"{split}: annotation {a['id']} has inconsistent image/video id")
            x, y, w, h = map(float, a['bbox'])
            if not np.isfinite([x, y, w, h]).all() or not (w > 0 and h > 0):
                raise SystemExit(f"annotation {a['id']}: invalid COCO bbox {a['bbox']}")
            ann_count += 1
    if set(vis) - {0, 2}:
        raise SystemExit(f'unexpected COCO visibility values: {dict(vis)}')
    if crowd.get(1, 0):
        raise SystemExit(f"is_crowd=1 annotations are unsupported ({crowd[1]})")
    seen_keys: set[tuple[int, int, int]] = set()
    for d in docs.values():
        for a in d['annotations']:
            key = (int(a['video_id']), int(Path(images[int(a['image_id'])]['file_name']).stem),
                   int(a['track_id']))
            if key in seen_keys:
                raise SystemExit(f'duplicate annotation for video/frame/track {key}')
            seen_keys.add(key)
    print(f'preflight: {len(images)} images, {len(video_dirs)} clips, {ann_count} annotations '
          f'({deduped} duplicate records removed); visibility={dict(vis)}, crowd={dict(crowd)}')
    return images, by_video, video_dirs, deduped


def source_parts(im: dict) -> tuple[str, str]:
    """Return the difficulty and species components of an image path."""
    p = Path(im['file_name'])
    return p.parts[0], p.parts[1]


def convert_session(dst: Path, split: str, difficulty: str, species: str,
                    groups_data: dict[int, list[dict]], images: dict[int, dict],
                    images_by_video: dict[int, list[dict]], video_dirs: dict[int, Path],
                    source: Path, annotation_file: str, duplicate_records: int,
                    clean: bool) -> tuple[int, int, int]:
    """Convert one split, difficulty, and species partition to a Tailcycle session."""
    if dst.exists():
        if not clean:
            raise SystemExit(f'{dst} exists; pass --clean to replace it')
        shutil.rmtree(dst)
    groups: dict[str, fmt.Group] = {}
    labels: dict[str, fmt.Labels] = {}
    for vid in sorted(groups_data):
        anns = groups_data[vid]
        ims = images_by_video[vid]
        T = len(ims)
        tracks = sorted({int(a['track_id']) for a in anns})
        aids = [f'{track:02d}' for track in tracks]
        ai = {track: i for i, track in enumerate(tracks)}
        gid = f'v{vid:07d}'
        groups[gid] = fmt.Group(gid, T, fps=float('nan'),
                                source_video=str(video_dirs[vid]),
                                source_frame_start=0, source_frame_step=1)
        lab = fmt.empty_labels(len(aids), T, len(NAMES), 1, mode3d=False,
                               animal_ids=aids)
        lab.boxes = np.full((len(aids), T, 1, 4), np.nan, np.float32)
        lab.instance = np.full((len(aids), T, 1), fmt.INST_NONE, np.int8)
        lab.regions = np.zeros((0, 6), np.float64)
        for a in anns:
            track = int(a['track_id'])
            frame = int(Path(images[int(a['image_id'])]['file_name']).stem)
            s = ai[track]
            lab.boxes[s, frame, 0] = bbox_xywh_to_xyxy(a['bbox'])
            lab.instance[s, frame, 0] = fmt.INST_LABELED
            xy, status = decode_keypoints(a['keypoints'])
            positioned = status == fmt.VISIBLE
            lab.points2d[s, frame, positioned, 0] = xy[positioned]
            lab.vis2d[s, frame, :, 0] = status
        labels[gid] = lab
        out_group = dst / 'groups' / gid
        out_group.mkdir(parents=True, exist_ok=True)
        cam = out_group / 'cam0'
        cam.mkdir()
        for i in range(T):
            (cam / frame_name(i)).symlink_to(video_dirs[vid] / f'{i:04d}.jpg')

    from aniposelib.cameras import CameraGroup
    cam = fmt.nominal_camera('cam0', (1920, 1080))
    rig = fmt.Rig(CameraGroup([cam]), offset={'cam0': (0.0, 0.0)},
                  moving={'cam0': False}, calibrated={'cam0': False})
    fmt.write_session(
        dst, mode='2d', units='px', label_source='annotated', names=NAMES, rig=rig,
        groups=groups, labels=labels, skeleton=SKELETON, flip_pairs=FLIP_PAIRS,
        provenance={
            'source': str(source),
            'annotation_file': annotation_file,
            'difficulty': difficulty,
            'species': species,
            'converter': 'scripts/convert_aptv2.py',
            'duplicate_records': f'{duplicate_records} duplicate (video, frame, track) records '
                                 'removed; kept the record with the most visible keypoints',
            'identity_source': 'APTv2 track_id, valid within each video clip only',
            'visibility_source': 'COCO keypoint visibility: v=2 -> visible; v=0 -> no row; '
                                 'APTv2 has no v=1 occluded-with-position state',
            'box_source': 'COCO bbox [x,y,w,h], converted to [x0,y0,x1,y1]',
            'regions_note': 'empty regions.pq: APTv2 does not certify exhaustive labelling',
        })
    return len(groups), sum(g.n_frames for g in groups.values()), sum(len(x) for x in groups_data.values())


def main() -> None:
    """Parse converter arguments and write the requested APTv2 dataset."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--no-image-check', action='store_true')
    args = ap.parse_args()
    docs = {split: load_json(args.src / 'annotations' / fn)
            for split, fn in SPLIT_FILES.items()}
    images, images_by_video, video_dirs, duplicate_records = preflight(args.src, docs)
    if args.out.exists() and args.clean and not args.dry_run:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    total = defaultdict(int)
    for split, d in docs.items():
        by_video: dict[int, list[dict]] = defaultdict(list)
        for a in d['annotations']:
            by_video[int(a['video_id'])].append(a)
        buckets: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(dict)
        for vid, anns in by_video.items():
            difficulty, species = source_parts(images[int(anns[0]['image_id'])])
            if any(source_parts(images[int(a['image_id'])]) != (difficulty, species) for a in anns):
                raise SystemExit(f'video {vid}: annotations span source species/difficulty')
            buckets[(difficulty, species)][vid] = anns
        for (difficulty, species), groups_data in sorted(buckets.items()):
            sid = f'{split}__{difficulty}__{species}'
            dst = args.out / split / sid
            ng = nf = na = 0
            if args.dry_run:
                ng = len(groups_data)
                nf = sum(len(images_by_video[vid]) for vid in groups_data)
                na = sum(map(len, groups_data.values()))
            else:
                ng, nf, na = convert_session(dst, split, difficulty, species, groups_data,
                                              images, images_by_video, video_dirs, args.src,
                                              SPLIT_FILES[split], duplicate_records, args.clean)
            total['groups'] += ng
            total['frames'] += nf
            total['annotations'] += na
            print(f'{split:5s} {sid:32s} {ng:4d} groups {nf:6d} frames {na:6d} annotations')
    print(f'total: {total["groups"]} groups, {total["frames"]} frames, '
          f'{total["annotations"]} annotations')
    if args.validate and not args.dry_run:
        errs = fmt.validate_dataset(fmt.load_dataset(args.out),
                                    check_images=not args.no_image_check)
        hard = [e for e in errs if 'WARNING' not in e]
        for e in errs:
            print(e)
        print(f'validate: {len(hard)} hard, {len(errs) - len(hard)} warnings')
        raise SystemExit(1 if hard else 0)


if __name__ == '__main__':
    main()
