#!/usr/bin/env python
"""Write `instances.pq` for a tailcycle dataset, from the UNCLEANED v3 keypoints.

    pixi run python scripts/backfill_boxes_v3.py --dataset rat-city --dry-run
    pixi run python scripts/backfill_boxes_v3.py --dataset rat-city

`convert_v4.py` reads `posetail-finetuning-v4`, whose score-cleaning NaNs out individual noisy
keypoints. That cleaning is wanted for pose supervision and ruinous for anything that BOUNDS the
animal, because the crop rule is a min/max over whatever survived. On rat-city's train group:

    v4  13% of instances have 0 points, 15% have 1, 16% have 2, 25% have 3, 31% have 4
    v3  all-or-nothing -- 4 points or none

so 25,777 train instances hold no finite point at all and 105,853 hold exactly one. The first
group becomes a NaN box, which `detector/data.py` correctly turns into an objectness target of
"no animal here" on a frame where a rat plainly is; the second collapses to a `min_crop_dim`
square around one keypoint. v3 is the same recording, same frames (v4's `img` is a symlink into
v3), same tracker slots. So: boxes from v3, keypoints stay on v4.

WHAT IS STORED IS THE PADDED EXTENT, NOT THE CROP BOX -- steps 1-2 of `crop_box_for_points` and
not step 3. Keeping `min_crop_dim` out of the file leaves it a consumer parameter, and costs
nothing: min/max over the two stored corners is min/max over the points they came from, so
`crop_box_for_points(corners, size, d, pad=0)` reproduces the full rule bit-for-bit. `--check`
asserts exactly that against the raw points before anything is written.

ROW ALIGNMENT is by name, not by position. `convert_v4.py:213` names animals `a{row:02d}` over
the ORIGINAL npz row index, so v3 row i is `a{i:02d}` even where v4's all-NaN filter dropped a
row (it dropped `a04` from rat-city val). Such an animal comes back here as a `present` row --
an ignore region with a box and no keypoints, which is what §9 is for and what
`scripts/eval.py:186` already reads for MOTA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt
from tailcyclenet.crop import crop_box_for_points

V3 = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v3')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets')

PAD = 20        # THE crop rule's pad. Not a knob: a consumer re-enters at pad=0 assuming it.


def padded_extent(pose, size):
    """(S,T,K,2) source px -> (S,T,4) int-valued [x0,y0,x1,y1], NaN where nothing is finite.

    Vectorised rather than a `crop_box_for_points` call per instance: rat-city's train group is
    691,128 instances and the per-instance loop ran for minutes. `--check` is what keeps the two
    honest -- it re-derives a sample through the real function and demands equality.
    """
    finite = np.isfinite(pose).all(-1)                                   # (S,T,K)
    any_ = finite.any(-1)
    p = np.where(finite[..., None], pose, np.nan)
    wh = np.asarray(size, np.float32)
    # An instance with no finite point is all-NaN through nanmin and garbage through the int32
    # cast; both are overwritten below. Warnings off so a real one would still be visible.
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lo = np.clip(np.nanmin(p, axis=2) - PAD, 0, wh)
            hi = np.clip(np.nanmax(p, axis=2) + PAD, 0, wh)
        # int32 truncation, exactly as the rule does it, so a consumer reading the float back and
        # truncating again is a no-op rather than a second, different rounding.
        box = np.concatenate([lo, hi], -1).astype(np.int32).astype(np.float32)
    box[~any_] = np.nan
    return box


def suppress_duplicates(box, pose, iou_thresh, kpt_frac):
    """Drop the second of any pair that is the SAME ANIMAL twice. Returns a (S,T) keep mask.

    Two conditions, both required. Box IoU alone is wrong here and measurably so: of the 327
    same-frame pairs in rat-city train above IoU 0.5, only ~5% have near-identical keypoints --
    the rest are two rats genuinely huddled, and plain NMS would delete real animals. Zero pairs
    anywhere in the dataset exceed IoU 0.9. So the keypoints decide, and the box only nominates.

    The survivor is the one with more surviving v4 keypoints (so `labeled` outranks `present`),
    tie-broken on the lower row index.
    """
    S, T = box.shape[:2]
    keep = np.isfinite(box).all(-1)
    rank = np.isfinite(pose).all(-1).sum(-1)                             # (S,T) v4 support
    area = np.prod(box[..., 2:] - box[..., :2], -1)
    # Size from the RAW keypoint extent, not the stored box: the box carries 2 x PAD on each
    # axis, which on a 206x173 rat inflates the diagonal by 26% and quietly loosens the gate by
    # the same factor. "0.15 body diagonals" has to mean the body.
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            diag = np.linalg.norm(np.nanmax(pose, 2) - np.nanmin(pose, 2), axis=-1)
    n = 0
    for i in range(S):
        for j in range(i + 1, S):
            m = np.flatnonzero(keep[i] & keep[j])
            if not m.size:
                continue
            lo = np.maximum(box[i, m, :2], box[j, m, :2])
            hi = np.minimum(box[i, m, 2:], box[j, m, 2:])
            inter = np.prod(np.clip(hi - lo, 0, None), -1)
            iou = inter / np.maximum(area[i, m] + area[j, m] - inter, 1e-6)
            d = np.linalg.norm(pose[i, m] - pose[j, m], axis=-1)
            with np.errstate(invalid='ignore'):
                same = np.nanmean(d, -1) < kpt_frac * np.minimum(diag[i, m], diag[j, m])
            hit = m[(iou > iou_thresh) & np.nan_to_num(same, nan=False).astype(bool)]
            if not hit.size:
                continue
            drop = np.where(rank[j, hit] > rank[i, hit], i, j)
            keep[drop, hit] = False
            n += hit.size
    return keep, n


def labelled_keys(sess, gid):
    """{(animal_id, camera)} x frame -> was this instance annotated? As a (n_animals, T) bool.

    Read off the dense arrays rather than the parquet: `Session.labels` has already scattered
    them and this is one `!= UNLABELED` reduce over an axis.
    """
    lab = sess.labels(gid)
    vis = lab.vis2d if lab.vis2d is not None else None
    if vis is None:                                  # 3D-only session: per-camera is a projection
        v3 = lab.vis3d
        return None if v3 is None else (v3 != fmt.UNLABELED).any(-1)     # (S,T)
    return (vis != fmt.UNLABELED).any(2)                                  # (S,T,C)


def convert_session(dst: Path, src: Path, args) -> dict:
    sess = fmt.Session.load(dst)
    if len(sess.rig) != 1:
        raise SystemExit(f'{dst}: this backfill assumes one camera, found {len(sess.rig)}')
    cam = sess.cam_names[0]
    size = tuple(int(x) for x in sess.rig.size(cam))
    cols = {c: [] for c in ('group_id', 'frame', 'animal_id', 'camera',
                            'x0', 'y0', 'x1', 'y1', 'status')}
    stats = {'rows': 0, 'labeled': 0, 'present': 0, 'suppressed': 0, 'new_animals': set()}

    for gid, group in sess.groups.items():
        npz = src / gid / 'pose2d.npz'
        if not npz.exists():
            raise SystemExit(f'{npz}: no v3 pose for group {gid!r}')
        pose = np.load(npz)['pose'].astype(np.float32)                    # (S,T,K,2)
        T = group.n_frames
        if pose.shape[1] < T:
            raise SystemExit(f'{npz}: {pose.shape[1]} frames, session says {T}')
        pose = pose[:, :T]
        if pose.shape[2] != len(sess.names):
            raise SystemExit(f'{npz}: {pose.shape[2]} keypoints, session says {len(sess.names)}')

        box = padded_extent(pose, size)                                   # (S,T,4)
        keep, n_sup = suppress_duplicates(box, pose, args.nms_iou, args.nms_kpt_frac)
        check_round_trip(pose, box, size, args.check, gid)

        # `a{row:02d}` is the v4 converter's own id scheme, and the only thing that makes v3 row i
        # and this session's animal i the same rat. Reproducing it here rather than reading
        # `lab.animal_ids` is deliberate: that list is missing exactly the rows v4 dropped, which
        # are the rows this table exists to restore.
        aids = [f'a{i:02d}' for i in range(pose.shape[0])]
        known = set(sess.labels(gid).animal_ids)
        stats['new_animals'] |= {a for a in aids if a not in known}

        lab_mask = labelled_keys(sess, gid)                               # (S,T[,C]) or None
        s, t = np.nonzero(keep)
        b = box[s, t]
        annotated = np.zeros(len(s), bool)
        if lab_mask is not None:
            m = lab_mask[..., 0] if lab_mask.ndim == 3 else lab_mask
            ok = s < m.shape[0]
            annotated[ok] = m[s[ok], t[ok]]

        cols['group_id'].extend([gid] * len(s))
        cols['frame'].extend(t.astype(np.int32))
        cols['animal_id'].extend(np.asarray(aids, dtype=object)[s])
        cols['camera'].extend([cam] * len(s))
        for i, q in enumerate(('x0', 'y0', 'x1', 'y1')):
            cols[q].extend(b[:, i])
        cols['status'].extend(np.where(annotated, 'labeled', 'present'))

        wh = b[:, 2:] - b[:, :2]
        huge = int((wh[:, 0] > size[0] / 2).sum())
        stats['rows'] += len(s)
        stats['labeled'] += int(annotated.sum())
        stats['present'] += int((~annotated).sum())
        stats['suppressed'] += n_sup
        print(f'    {gid}: {len(s)} rows  labeled={int(annotated.sum())} '
              f'present={int((~annotated).sum())}  suppressed={n_sup}  '
              f'median box {np.median(wh[:, 0]):.0f}x{np.median(wh[:, 1]):.0f}px  '
              f'wider-than-half-frame={huge}')

    if not args.dry_run:
        out = {k: (np.asarray(v, np.float32) if k in ('x0', 'y0', 'x1', 'y1')
                   else np.asarray(v, np.int32) if k == 'frame'
                   else np.asarray(v, dtype=object)) for k, v in cols.items()}
        fmt.write_table(dst / 'instances.pq', out, fmt.DICT_COLS)
    return stats


def check_round_trip(pose, box, size, n, gid):
    """The claim the storage format rests on, asserted on a sample of real instances.

    Bounding the two stored corners at `pad=0` must give the SAME int32 box as bounding the raw
    points at the default pad. If it ever does not, every consumer of this table is cropping
    somewhere the pose model was never trained.
    """
    if n <= 0:
        return
    s, t = np.nonzero(np.isfinite(box).all(-1))
    if not s.size:
        return
    step = max(1, s.size // n)
    wh = torch.tensor(size, dtype=torch.int32)
    for i in range(0, s.size, step):
        si, ti = int(s[i]), int(t[i])
        want = crop_box_for_points(torch.as_tensor(pose[si, ti]), wh)
        got = crop_box_for_points(torch.as_tensor(box[si, ti]).view(2, 2), wh, pad=0)
        assert torch.equal(want, got), (
            f'{gid}: stored extent does not reproduce the crop rule at animal {si} frame {ti}: '
            f'points -> {want.tolist()}, stored corners -> {got.tolist()}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='rat-city')
    ap.add_argument('--src', type=Path, default=V3, help='the UNCLEANED source root')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--dry-run', action='store_true', help='report everything, write nothing')
    ap.add_argument('--nms-iou', type=float, default=0.7,
                    help='box IoU above which a pair is a duplicate CANDIDATE')
    ap.add_argument('--nms-kpt-frac', type=float, default=0.15,
                    help='and mean keypoint distance below this fraction of the smaller box '
                         'diagonal for it to actually be suppressed')
    ap.add_argument('--check', type=int, default=200,
                    help='instances per group to re-derive through crop_box_for_points; 0 = off')
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()

    root = args.out / args.dataset
    total = {'rows': 0, 'labeled': 0, 'present': 0, 'suppressed': 0}
    new_animals = set()
    for split in fmt.SPLITS:
        d = root / split
        if not d.is_dir():
            continue
        for dst in sorted(p for p in d.iterdir() if p.is_dir()):
            src = args.src / args.dataset / split / dst.name
            if not src.is_dir():
                raise SystemExit(f'{src}: no matching v3 session for {split}/{dst.name}')
            print(f'  {split}/{dst.name}')
            st = convert_session(dst, src, args)
            new_animals |= st.pop('new_animals')
            for k, v in st.items():
                total[k] += v

    verb = 'would write' if args.dry_run else 'wrote'
    print(f'{verb} {total["rows"]} rows  labeled={total["labeled"]} present={total["present"]}  '
          f'suppressed={total["suppressed"]}')
    if new_animals:
        # Not an error: an animal v4 NaN'd out entirely is exactly the ignore region §9 wants.
        # Printed because it CHANGES THE ANIMAL AXIS -- `_animal_vocab` unions the tables, so `S`
        # grows for that group and a downstream shape assumption would fail far from here.
        print(f'note: {len(new_animals)} animal id(s) appear only in instances.pq '
              f'({sorted(new_animals)}) -- present-but-unannotated, and they widen S')

    if args.validate and not args.dry_run:
        errs = fmt.validate_dataset(fmt.load_dataset(root), check_images=False)
        hard = [e for e in errs if 'WARNING' not in e]
        for e in errs:
            print(('  WARN ' if 'WARNING' in e else '  FAIL ') + e)
        print(f'validate: {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')
        return 1 if hard else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
