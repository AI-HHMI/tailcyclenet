#!/usr/bin/env python
"""Convert `johnson-mouse/merge_aug` into the tailcycle-dataset format.

    pixi run python scripts/convert_johnson_aug.py --dry-run
    pixi run python scripts/convert_johnson_aug.py --clean

`merge_aug` is `merge` with 13 photometric/compositing augmentation variants baked in as extra
IMAGES. Everything reusable -- the calibration reader, the robust triangulation, the run cutter,
the reprojection check -- is IMPORTED from `convert_johnson.py`; this script adds only what the
augmented layout needs. `convert_johnson.py` is not modified.

Five source facts decide the shape of this script, all measured:

- **An augmented image's keypoints are byte-identical to its base image's** (280,002 identical, 0
  differing, across all 13 variants; boxes too). So this root adds PIXELS AND NOT LABELS: 8.4x the
  frames over the same 2,965 poses. It is a training root only -- a bootstrap or `--vs` over it
  resamples the same poses many times. `assert_same_2d` turns that finding into a check rather
  than a comment, and the 3D layer is triangulated ONCE per (trial, source frame) and reused.

- **An aug frameset's 16 views are a random per-camera MIX** of augmented and original images:
  `Frame_104__shadow_00` uses augmented pixels for 5 of 16 cameras and the originals for the rest.
  Symlinks therefore follow the JSON per camera, never a filename rule.

- **The originals under `merge_aug/train/` are 53,250 DANGLING symlinks** into an unmounted
  `/mnt/nvme2`. `resolve_image` falls back to `merge/train/`, which holds them as real files, and
  raises naming both paths if neither has it. The output root has zero dangling links.

- **`mirror_00` is a ghost**: 21,554 images on disk whose keypoints genuinely DIFFER from their
  base (it is a real geometric augmentation), referenced by ZERO framesets. Being JSON-driven
  excludes it automatically -- the same ghost-jpg lesson `convert_johnson.py` carries.

- **Trial `2026_08_07_16_45_33` cannot be converted**: it is new in `merge_aug` (154 framesets, 11
  source frames), has no `calib_params/2026_08_07*` anywhere on disk, and its originals are
  dangling with no `merge/` counterpart. It is skipped LOUDLY, by trial name.

`val/` comes from `merge`, UNAUGMENTED: `merge_aug/annotations/instances_val.json` is a dangling
symlink and `merge_aug/val` symlinks to `../merge/val`, so there is no augmented val to convert --
and an unaugmented val is what model selection wants anyway.

Group ids are variant-suffixed: `000104` for an original run, `000104__shadow_00` for a variant's.
Runs are cut per (trial, variant), because a variant covers a SUBSET of the trial's frames, so its
runs differ. **Group ids therefore do not pair 1:1 across variants**; key on (trial, source frame)
via `source_frame_start`/`source_frame_step` if you want "the same clip under 13 augmentations".
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tailcyclenet import format as fmt
from convert_johnson import (build_labels, build_rig, check_reprojection, modal_step,
                             read_split, runs, triangulate_robust)

SRC = Path('/groups/karashchuk/karashchuklab/animal-datasets/johnson-mouse/merge_aug')
FALLBACK = Path('/groups/karashchuk/karashchuklab/animal-datasets/johnson-mouse/merge')
OUT = Path('/groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/'
           'johnson-mouse-annotated-aug')

# `Frame_<N>` or `Frame_<N>__<variant>`. Anything else raises: a new suffix convention must not be
# absorbed silently as part of the frame number or as a trial of its own.
_KEY = re.compile(r'^(?P<trial>[^/]+)/Frame_(?P<frame>\d+)(?:__(?P<variant>[A-Za-z0-9_]+))?$')

AUG_NOTE = ('13 photometric/compositing variants; an augmented image carries keypoints IDENTICAL '
            'to its source frame, so this root adds pixels and not labels -- training only, '
            'never eval')


# ----------------------------------------------------------------------------------------------
# the augmented frameset grammar
# ----------------------------------------------------------------------------------------------

def split_key(key: str) -> tuple[str, int, str]:
    """`<trial>/Frame_<N>[__<variant>]` -> (trial, source frame, variant or '')."""
    m = _KEY.match(key)
    if m is None:
        raise RuntimeError(f'frameset key {key!r} does not match <trial>/Frame_<N>[__<variant>]')
    return m['trial'], int(m['frame']), m['variant'] or ''


def resolve_image(rel: str, src: Path, fallback: Path) -> Path:
    """`<trial>/<cam>/<file>.jpg` -> a real file, preferring the augmented root.

    53,250 of `merge_aug/train`'s entries are symlinks into an unmounted /mnt/nvme2. `Path.exists`
    already follows the link, so a dangling one fails this test and falls through to `merge`,
    which holds the same originals as real files.
    """
    a = src / 'train' / rel
    if a.exists():
        return a.resolve()
    b = fallback / 'train' / rel
    if b.exists():
        return b.resolve()
    raise RuntimeError(f'image {rel!r} resolves nowhere: {a} is missing or dangling, '
                       f'and {b} does not exist')


# ----------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------

def read_aug(src: Path) -> dict:
    """The augmented train JSON, indexed by (trial, variant) -> {source frame: {camera: image id}}.

    Everything is driven from `framesets`; the `images` list is only a lookup. That is what keeps
    `mirror_00`'s 21,554 unreferenced images out, and it is the reason a filesystem walk is never
    used here.
    """
    with open(src / 'annotations' / 'instances_train.json') as f:
        d = json.load(f)

    imgs = {im['id']: im for im in d['images']}
    anns: dict[int, dict] = {}
    for a in d['annotations']:
        if a['image_id'] in anns:
            raise RuntimeError(f'image {a["image_id"]} has >1 annotation')
        anns[a['image_id']] = a

    # (trial, variant) -> {source frame: {camera: image_id}}
    clips: dict[tuple[str, str], dict[int, dict[str, int]]] = defaultdict(dict)
    for key, fs in d['framesets'].items():
        trial, frame, variant = split_key(key)
        views = {}
        for iid in fs['frames']:
            parts = imgs[iid]['file_name'].split('/')
            if parts[0] != trial:
                raise RuntimeError(f'frameset {key} holds an image from {parts[0]}')
            views[parts[1]] = iid
        clips[(trial, variant)][frame] = views

    d['_imgs'], d['_anns'], d['_clips'] = imgs, anns, clips
    return d


# ----------------------------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------------------------

def fill_2d(d: dict, run: list[int], views: dict[int, dict[str, int]], cams: list[str],
            K: int) -> fmt.Labels:
    """The 2D layer for one group: keypoints, boxes and the per-camera instance rows.

    Same rules as `convert_johnson.build_labels`: an unannotated image leaves every cell
    UNLABELED rather than `present`, and the status is PROJECTED because the source flags
    1,235,334 points "visible" against 18 "not" across 16 views of a mouse on a dome.
    """
    C, T = len(cams), len(run)
    lab = fmt.empty_labels(1, T, K, C, mode3d=True, animal_ids=['a00'])
    lab.boxes = np.full((1, T, C, 4), np.nan, np.float32)
    lab.instance = np.full((1, T, C), fmt.INST_NONE, np.int8)

    for t, frame in enumerate(run):
        for ci, cam in enumerate(cams):
            iid = views[frame].get(cam)
            if iid is None:
                continue
            a = d['_anns'].get(iid)
            if a is None:                      # image exists, nobody labelled it
                continue
            kp = np.asarray(a['keypoints'], dtype=np.float64).reshape(K, 3)
            seen = kp[:, 2] > 0
            lab.vis2d[0, t, seen, ci] = fmt.PROJECTED
            lab.points2d[0, t, seen, ci] = kp[seen, :2].astype(np.float32)

            x, y, w, h = (float(v) for v in a['bbox'])
            lab.boxes[0, t, ci] = (x, y, x + w, y + h)
            lab.instance[0, t, ci] = fmt.INST_LABELED
    return lab


def fit_3d(rig: fmt.Rig, lab: fmt.Labels, reject_px: float) -> int:
    """Triangulate this group's own 2D in place. Returns the rejected-observation count."""
    S, T, K, C, _ = lab.points2d.shape
    p2d = np.moveaxis(lab.points2d[0], 2, 0).reshape(C, T * K, 2).astype(np.float64)
    p3d, rejected = triangulate_robust(rig.cgroup, p2d, reject_px)
    p3d = p3d.reshape(T, K, 3)
    ok = np.isfinite(p3d).all(-1)
    lab.vis3d[0][ok] = fmt.VISIBLE
    lab.points3d[0][ok] = p3d[ok].astype(np.float32)
    apply_rejections(lab, rejected.reshape(C, T, K).transpose(1, 2, 0))
    return int(rejected.sum())


def apply_rejections(lab: fmt.Labels, bad: np.ndarray) -> None:
    """Drop outlier 2D from keypoints.pq as NO ROW -- `missing` would claim someone judged it.

    A view that loses every observation also loses its box: the source box is the keypoint hull,
    so it carries the same displacement, and rule 11 wants a `labeled` instance to still have
    keypoint rows.
    """
    lab.vis2d[0][bad] = fmt.UNLABELED
    lab.points2d[0][bad] = np.nan
    dead = (lab.vis2d[0] == fmt.UNLABELED).all(1) & (lab.instance[0] != fmt.INST_NONE)
    lab.instance[0][dead] = fmt.INST_NONE
    lab.boxes[0][dead] = np.nan


class Solved:
    """Per-(trial, source frame) 3D, solved once from the ORIGINAL frameset and reused.

    Reuse is only legal because an augmented image's keypoints are identical to its base's, so
    `carry` ASSERTS that before copying. A variant that ever diverges is triangulated on its own
    and counted -- the finding is a check, not a comment. Reuse is also ~9x fewer robust DLT fits
    (2,965 instead of 27,817).
    """

    def __init__(self) -> None:
        self.p3d: dict[tuple[str, int], np.ndarray] = {}       # (K,3)
        self.rejected: dict[tuple[str, int], np.ndarray] = {}  # (K,C) bool
        self.pre: dict[tuple[str, int], np.ndarray] = {}       # (K,C,2), before rejection
        self.diverged = 0

    def store(self, trial: str, run: list[int], lab: fmt.Labels, bad: np.ndarray,
              p2d_pre: np.ndarray) -> None:
        for t, frame in enumerate(run):
            self.p3d[(trial, frame)] = lab.points3d[0, t].copy()
            self.rejected[(trial, frame)] = bad[t].copy()
            self.pre[(trial, frame)] = p2d_pre[t].copy()

    def carry(self, trial: str, run: list[int], lab: fmt.Labels) -> bool:
        """Copy the solved 3D onto this variant group, iff every frame's 2D matches."""
        keys = [(trial, f) for f in run]
        if any(k not in self.p3d for k in keys):
            return False
        for t, k in enumerate(keys):
            if not _same_2d(lab.points2d[0, t], self.pre[k]):
                self.diverged += 1
                return False
        bad = np.stack([self.rejected[k] for k in keys])          # (T,K,C)
        for t, k in enumerate(keys):
            lab.points3d[0, t] = self.p3d[k]
        ok = np.isfinite(lab.points3d[0]).all(-1)
        lab.vis3d[0][ok] = fmt.VISIBLE
        apply_rejections(lab, bad)
        return True


def _same_2d(a: np.ndarray, b: np.ndarray) -> bool:
    """NaN-aware exact equality: these are copies of the same floats, not a numeric comparison."""
    return bool((np.isnan(a) == np.isnan(b)).all() and np.array_equal(a[~np.isnan(a)],
                                                                      b[~np.isnan(b)]))


# ----------------------------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------------------------

def convert_train(src: Path, fallback: Path, out: Path, max_gap: int, reject_px: float,
                  only: list[str] | None, max_groups: int | None, dry_run: bool) -> dict:
    d = read_aug(src)
    names = list(d['keypoint_names'])
    K = len(names)
    skeleton = [[j['keypointA'], j['keypointB']] for j in d['skeleton']]
    flip_pairs = [[n, n[:-1] + 'R'] for n in names if n.endswith('L') and n[:-1] + 'R' in names]

    trials = sorted({t for t, _ in d['_clips']})
    stats = {'sessions': 0, 'groups': 0, 'frames': 0, 'links': 0, 'rejected': 0,
             'reused': 0, 'solved': 0, 'skipped_trials': 0, 'skipped_framesets': 0}

    for trial in trials:
        if only and trial not in only:
            continue
        variants = sorted({v for t, v in d['_clips'] if t == trial})
        n_fs = sum(len(d['_clips'][(trial, v)]) for v in variants)

        cal = d['calibrations'].get(trial)
        if cal is None or not all((src / p).exists() or (fallback / p).exists()
                                  for p in cal.values()):
            print(f'   ! {trial}: no calibration on disk -- SKIPPED, '
                  f'{n_fs} frameset(s) dropped ({len(variants)} variant(s))')
            stats['skipped_trials'] += 1
            stats['skipped_framesets'] += n_fs
            continue
        cal_root = src if all((src / p).exists() for p in cal.values()) else fallback

        # An UNKNOWN camera raises -- that is a frameset and a calibration disagreeing about what
        # the rig IS. A frameset holding a SUBSET is DROPPED, loudly: rule 7 wants every camera
        # dir to hold exactly `n_frames` contiguous files, so a 15-of-16 frameset cannot be
        # written at all, and the alternatives are inventing a pixel or failing validation.
        # `merge` has all 16 everywhere, which is why `convert_johnson.py` could raise on any
        # disagreement; merge_aug has exactly ONE such frameset in 27,817
        # (2026_02_26_13_29_50/Frame_4431, missing Cam2006052), and refusing a whole trial over it
        # is not a conversion rule either.
        cams = sorted(cal)
        short: list[tuple[str, int]] = []
        for v in variants:
            for f, views in sorted(d['_clips'][(trial, v)].items()):
                unknown = [c for c in sorted(views) if c not in cal]
                if unknown:
                    raise RuntimeError(f'{trial}/{f} [{v or "original"}]: camera(s) {unknown} '
                                       f'are not in the calibration {cams}')
                if len(views) != len(cams):
                    short.append((v, f))
        for v, f in short:
            missing = [c for c in cams if c not in d['_clips'][(trial, v)][f]]
            print(f'   ! {trial}/Frame_{f} [{v or "original"}]: missing {missing} -- '
                  f'frameset DROPPED (rule 7 needs every camera)')
            del d['_clips'][(trial, v)][f]
            stats['skipped_framesets'] += 1
        variants = [v for v in variants if d['_clips'][(trial, v)]]
        first = d['_clips'][(trial, '')]
        f0 = sorted(first)[0]
        sizes = {c: (d['_imgs'][first[f0][c]]['width'], d['_imgs'][first[f0][c]]['height'])
                 for c in cams}
        rig = build_rig(cal_root, cal, sizes)

        solved = Solved()
        groups: dict[str, fmt.Group] = {}
        labels: dict[str, fmt.Labels] = {}
        links: list[tuple[Path, Path]] = []

        # The ORIGINAL variant first: it is what every other variant carries its 3D from.
        for variant in [''] + [v for v in variants if v]:
            clip = d['_clips'][(trial, variant)]
            frames = sorted(clip)
            clips = runs(frames, max_gap)
            if max_groups:
                clips = clips[:max_groups]
            suffix = f'__{variant}' if variant else ''

            for run in clips:
                gid = f'{run[0]:06d}{suffix}'
                lab = fill_2d(d, run, clip, cams, K)
                pre = lab.points2d[0].copy()                                  # (T,K,C,2)
                if variant and solved.carry(trial, run, lab):
                    stats['reused'] += 1
                else:
                    n = fit_3d(rig, lab, reject_px)
                    stats['rejected'] += n
                    stats['solved'] += 1
                    if not variant:
                        bad = np.isnan(lab.points2d[0]).any(-1) & ~np.isnan(pre).any(-1)
                        solved.store(trial, run, lab, bad, pre)

                groups[gid] = fmt.Group(gid, len(run), fps=float('nan'),
                                        source_video=f'{src.name}/train/{trial}',
                                        source_frame_start=run[0],
                                        source_frame_step=modal_step(run),
                                        notes=variant)
                labels[gid] = lab
                stats['frames'] += len(run)
                for cam in cams:
                    for t, frame in enumerate(run):
                        rel = d['_imgs'][clip[frame][cam]]['file_name']
                        links.append((out / 'train' / trial / 'groups' / gid / cam /
                                      f'{t:06d}.jpg', resolve_image(rel, src, fallback)))

        n_orig = sum(1 for g in groups.values() if not g.notes)
        print(f'   train/{trial}: {len(groups)} group(s) '
              f'({n_orig} original + {len(groups) - n_orig} augmented), '
              f'{sum(g.n_frames for g in groups.values())} frames, {len(cams)} cams, '
              f'{len(links)} link(s)')
        stats['sessions'] += 1
        stats['groups'] += len(groups)
        stats['links'] += len(links)
        if solved.diverged:
            print(f'     ! {solved.diverged} augmented group(s) whose 2D differs from the '
                  f'original -- triangulated independently')
        if dry_run:
            continue

        for dst, srcp in links:
            dst.parent.mkdir(parents=True, exist_ok=True)
            fmt.link(dst, srcp)
        fmt.write_session(
            out / 'train' / trial, mode='3d', units='mm', label_source='annotated', names=names,
            rig=rig, groups=groups, labels=labels, skeleton=skeleton, flip_pairs=flip_pairs,
            provenance={
                'source': f'johnson-mouse/merge_aug/train/{trial}',
                'annotator': '', 'annotator_tool': 'scripts/convert_johnson_aug.py',
                'points3d_source': 'DLT triangulation of the 16-view 2D labels (derived, not '
                                   'annotated); an augmented group reuses its source frame\'s '
                                   'solution, asserted equal on the 2D',
                'animal_id_source': 'single animal per session',
                'augmentation': AUG_NOTE,
            })
    return stats


def convert_val(fallback: Path, out: Path, max_gap: int, reject_px: float, dry_run: bool) -> None:
    """val/ from `merge`, UNAUGMENTED -- merge_aug ships no readable val of its own.

    `convert_johnson`'s own `read_split` and `build_labels` do the work, so this is the verified
    val path rather than a second implementation of it. `convert_johnson.convert` itself is not
    called because it writes BOTH splits, and re-converting merge's 51,888-link train half to
    throw it away is not a cheap way to get one split.
    """
    print('== val/ from johnson-mouse/merge (unaugmented)')
    d = read_split(fallback, 'val')
    names = list(d['keypoint_names'])
    K = len(names)
    skeleton = [[j['keypointA'], j['keypointB']] for j in d['skeleton']]
    flip_pairs = [[n, n[:-1] + 'R'] for n in names if n.endswith('L') and n[:-1] + 'R' in names]

    for trial in sorted(d['_framesets']):
        fsets = d['_framesets'][trial]
        frames = sorted(fsets)
        cams = sorted(d['calibrations'][trial])
        for f in frames:
            if sorted(fsets[f]) != cams:
                raise RuntimeError(f'val/{trial}/{f}: cameras {sorted(fsets[f])} disagree with '
                                   f'calibration {cams}')
        sizes = {c: (d['_imgs'][fsets[frames[0]][c]]['width'],
                     d['_imgs'][fsets[frames[0]][c]]['height']) for c in cams}
        rig = build_rig(fallback, d['calibrations'][trial], sizes)

        dst = out / 'val' / trial
        groups, labels, gated = {}, {}, 0
        for run in runs(frames, max_gap):
            gid = f'{run[0]:06d}'
            groups[gid] = fmt.Group(gid, len(run), fps=float('nan'),
                                    source_video=str(fallback / 'val' / trial),
                                    source_frame_start=run[0], source_frame_step=modal_step(run))
            labels[gid], n = build_labels(d, run, fsets, rig, K, reject_px)
            gated += n
            if dry_run:
                continue
            for cam in rig.names:
                cdir = dst / 'groups' / gid / cam
                cdir.mkdir(parents=True, exist_ok=True)
                for t, frame in enumerate(run):
                    s = (fallback / 'val' / d['_imgs'][fsets[frame][cam]]['file_name']).resolve()
                    fmt.link(cdir / f'{t:06d}.jpg', s)
        note = f', {gated} outlier 2D observation(s) rejected' if gated else ''
        print(f'   val/{trial}: {len(groups)} group(s), '
              f'{sum(g.n_frames for g in groups.values())} frames, {len(rig)} cams{note}')
        if dry_run:
            continue
        fmt.write_session(
            dst, mode='3d', units='mm', label_source='annotated', names=names, rig=rig,
            groups=groups, labels=labels, skeleton=skeleton, flip_pairs=flip_pairs,
            provenance={
                'source': f'johnson-mouse/merge/val/{trial}',
                'annotator': '', 'annotator_tool': 'scripts/convert_johnson_aug.py',
                'points3d_source': 'DLT triangulation of the 16-view 2D labels (derived, not '
                                   'annotated)',
                'animal_id_source': 'single animal per session',
                'augmentation': 'none -- val is the unaugmented merge split; merge_aug ships no '
                                'readable val (its instances_val.json is a dangling symlink into '
                                'an unmounted /mnt/nvme2)',
            })


# ----------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------

def check_aug_equals_original(out: Path, limit: int) -> int:
    """Re-read from disk: an augmented group must equal its source frames' original group.

    This is the one check that catches a variant/frame misalignment, which is the failure mode
    the variant-suffixed layout is exposed to.

    **The source frame index is read off the SYMLINK TARGET**, not reconstructed as
    `source_frame_start + t * source_frame_step`. `modal_step` is what its name says -- a run cut
    at `--max-gap 8` legitimately holds gaps of 8 among steps of 4, so the arithmetic form
    mis-indexes the middle of most runs, and this check reported 22 false failures before it read
    the links instead. Reading the target also verifies the pixels' own mapping, which the
    arithmetic form never touched.
    """
    print(f'\n== augmented groups vs their originals (up to {limit} per session)')
    bad = 0
    for sdir in sorted((out / 'train').iterdir()):
        sess = fmt.Session.load(sdir)
        orig: dict[int, tuple] = {}
        for gid, g in sess.groups.items():
            if g.notes:
                continue
            lab = sess.labels(gid)
            for t, f in enumerate(_source_frames(sdir, gid, g.n_frames)):
                orig[f] = (lab.points2d[0, t], lab.vis2d[0, t], lab.points3d[0, t])
        n = miss = 0
        for gid, g in sess.groups.items():
            if not g.notes or n >= limit:
                continue
            lab = sess.labels(gid)
            for t, f in enumerate(_source_frames(sdir, gid, g.n_frames)):
                if f not in orig:
                    miss += 1
                    continue
                p2, v2, p3 = orig[f]
                if not (_same_2d(lab.points2d[0, t], p2)
                        and np.array_equal(lab.vis2d[0, t], v2)
                        and _same_2d(lab.points3d[0, t], p3)):
                    bad += 1
                    print(f'   FAIL {sdir.name}/{gid} frame {f}: differs from the original')
            n += 1
        note = f', {miss} frame(s) with no original in this root' if miss else ''
        print(f'   {sdir.name}: {n} augmented group(s) checked{note}')
    return bad


_FRAME = re.compile(r'Frame_(\d+)')


def _source_frames(sdir: Path, gid: str, n_frames: int) -> list[int]:
    """The source frame index at each position of a group, off the written symlinks."""
    cdir = sorted((sdir / 'groups' / gid).iterdir())[0]
    out = []
    for t in range(n_frames):
        target = (cdir / f'{t:06d}.jpg').readlink().name
        m = _FRAME.search(target)
        if m is None:
            raise RuntimeError(f'{cdir}/{t:06d}.jpg -> {target}: no Frame_<N> in the target')
        out.append(int(m.group(1)))
    return out


def check_no_dangling(out: Path) -> int:
    n = sum(1 for p in out.rglob('*.jpg') if p.is_symlink() and not p.exists())
    print(f'\n== dangling symlinks in the output: {n}')
    return n


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--fallback', type=Path, default=FALLBACK,
                    help='root holding the originals merge_aug symlinks to (default merge/)')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--max-gap', type=int, default=8,
                    help='source-frame gap above which a group is cut (default 8)')
    ap.add_argument('--trials', nargs='+', default=None, help='convert only these trials')
    ap.add_argument('--max-groups', type=int, default=None, help='cap groups per (trial, variant)')
    ap.add_argument('--no-val', action='store_true', help='skip the val split')
    ap.add_argument('--dry-run', action='store_true', help='report counts, write nothing')
    ap.add_argument('--no-check', action='store_true', help='skip the reprojection check')
    ap.add_argument('--no-image-check', action='store_true')
    ap.add_argument('--max-reproj-px', type=float, default=2.0)
    ap.add_argument('--reject-px', type=float, default=20.0)
    ap.add_argument('--aug-check-limit', type=int, default=20,
                    help='augmented groups per session to re-read and compare (default 20)')
    ap.add_argument('--clean', action='store_true', help='remove the output dir first')
    args = ap.parse_args()

    if args.clean and args.out.exists() and not args.dry_run:
        shutil.rmtree(args.out)

    print('== train/ from johnson-mouse/merge_aug')
    stats = convert_train(args.src, args.fallback, args.out, args.max_gap, args.reject_px,
                          args.trials, args.max_groups, args.dry_run)
    print(f'\n{stats}')
    if not args.no_val:
        convert_val(args.fallback, args.out, args.max_gap, args.reject_px, args.dry_run)
    if args.dry_run:
        return

    errs = fmt.validate_dataset(fmt.load_dataset(args.out),
                                check_images=not args.no_image_check)
    hard = [e for e in errs if 'WARNING' not in e]
    for e in errs:
        print(('  WARN ' if 'WARNING' in e else '  FAIL ') + e)
    print(f'validate: {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')

    bad = check_aug_equals_original(args.out, args.aug_check_limit)
    bad += check_no_dangling(args.out)
    if not args.no_check:
        bad += check_reprojection(args.out, args.max_reproj_px)
    sys.exit(1 if hard or bad else 0)


if __name__ == '__main__':
    main()
