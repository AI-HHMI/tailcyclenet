"""Convert MultiFly (Robie & Branson, Fly Disco) into a tailcycle dataset with FULL source frames.

Source: https://research.janelia.org/bransonlab/multifly/ (multifly_v_1_0.zip), COCO JSONs of
192x192 rotated crops, one fly per crop. `crop_params.csv` (from reconstruct_crops.py next to the
download) maps every crop to its source UFMF frame and crop geometry; that rule reproduces all
10,895 PNGs byte-exactly, so the inverse map below is exact up to the annotation itself:
  crop (u, v), pixel-centre 0-based  ->  source (x0 + c*du - s*dv, y0 + s*du + c*dv)
  du, dv = u - 95.5, v - 95.5;  c, s = cos, sin(theta + pi/2);  (x0, y0) = trx (x, y) - 1

Checked against APT's .trk tracking: 0.5-0.7 px median per keypoint (its L/R order is swapped).
Layout: one session per (split, source movie). Each labelled frame gets a centred window of
[f-16, f+16] (33 frames; shifted inside the movie at its ends); windows that overlap or touch are
merged into one group, so a group holds as many labelled frames as its context allows. Pixels are
decoded from the UFMF and written as lossless PNG. Every labelled keypoint is `visible` (the
source's v=1 "occluded, placed" points included, by decision). Every labelled fly gets a `present`
instances.pq row -- not all flies in a frame are labelled -- boxed by the 192x192 crop window
(axis-aligned, centred on the crop centre). `animal_id` is the trx target (`fly##`).

Splits: train/test as published. val = whole train groups (so disjoint frames) holding ~5% of each
train movie's labels, seeded (--val-frac, --seed); val shares movies and flies with train.
`labels_index.csv` at the root has one row per labelled fly-frame: its split/session/group/frame/
animal, the source movie frame, the COCO annotation/image it came from, and the test set's
`behavior`/`condition` tags (empty for train/val, which the source does not tag).

A re-run reuses already-written PNG groups (moved, not re-decoded) when the group plan matches.
The inter/intra-annotator relabels (test_inter/test_intra) are NOT converted: they label the same
frames as test by another annotator, which the format puts in a separate root.

Usage:
  pixi run python scripts/convert_multifly.py \
      --src /groups/karashchuk/karashchuklab/animal-datasets/robie-multifly \
      --out /groups/karashchuk/karashchuklab/animal-datasets-processed/tailcycle-datasets/robie-multifly \
      --jobs 16 [--only <substr>] [--dry-run] [--validate]
"""
import argparse
import csv
import datetime
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tailcyclenet import format as fmt  # noqa: E402

APT_DEEPNET = '/groups/branson/bransonlab/apt/repo/prod/deepnet'
URL = 'https://research.janelia.org/bransonlab/multifly/multifly_v_1_0.zip'
SPLITS = {'train': 'train_annotations.json', 'test': 'test_annotations.json'}
CAM = 'cam0'
HALF = 16
STAGE = '.staging'
RENAME = {'right_mid_fitib': 'right_mid_fetib'}
CROP = 192
CTR = (CROP - 1) / 2
FLIP = [('right_eye', 'left_eye'), ('right_thorax', 'left_thorax'),
        ('right_mid_fe', 'left_mid_fe'), ('right_mid_fetib', 'left_mid_fetib'),
        ('right_front_tar', 'left_front_tar'), ('right_mid_tar', 'left_mid_tar'),
        ('right_back_tar', 'left_back_tar'), ('right_mid_wing', 'left_mid_wing'),
        ('right_outer_wing', 'left_outer_wing')]


def crop_to_full(kp: np.ndarray, x0: float, y0: float, theta: float) -> np.ndarray:
    """(K,2) crop coords -> (K,2) source-frame coords, both 0-based pixel centres."""
    c, s = np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)
    du, dv = kp[:, 0] - CTR, kp[:, 1] - CTR
    return np.stack([x0 + c * du - s * dv, y0 + s * du + c * dv], 1)


def make_groups(frames, n_src: int) -> list[list]:
    """[start, stop, [labelled frames]] per group; each label centred in >= 2*HALF+1 frames."""
    groups = []
    for f in sorted(set(frames)):
        s, e = f - HALF, f + HALF + 1
        if s < 0:
            s, e = 0, e - s
        if e > n_src:
            s, e = max(0, s - (e - n_src)), n_src
        if groups and s <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], e)
            groups[-1][2].append(f)
        else:
            groups.append([s, e, [f]])
    return groups


def load_jobs(src: Path) -> tuple[list[dict], list[str], list]:
    """One job per (split, movie): its annotations with full-frame keypoints."""
    data = src / 'fly_bubble_data_20241024'
    params = {(r['json'], r['image']): r for r in csv.DictReader(open(src / 'crop_params.csv'))}
    if not all(r['exact'] == 'True' for r in params.values()):
        raise SystemExit('crop_params.csv has non-exact rows; re-run reconstruct_crops.py')
    jobs, names, skeleton = {}, None, None
    for split, fname in SPLITS.items():
        d = json.loads((data / fname).read_text())
        cat = d['categories'][0]
        if names is None:
            names, skeleton = [RENAME.get(n, n) for n in cat['keypoints']], cat['skeleton']
        elif [RENAME.get(n, n) for n in cat['keypoints']] != names:
            raise SystemExit(f'{fname}: keypoint list differs from train')
        imgs = {im['id']: im for im in d['images']}
        beh, cond = d['info'].get('behaviors', {}), d['info'].get('conditions', {})
        for a in d['annotations']:
            p = params[(fname, imgs[a['image_id']]['file_name'])]
            kp = np.asarray(a['keypoints'], float).reshape(-1, 3)
            if len(kp) != len(names) or (kp[:, 2] == 0).any():
                raise SystemExit(f'{fname} ann {a["id"]}: unexpected keypoint count/visibility')
            x0, y0, th = float(p['x0']), float(p['y0']), float(p['theta_rad'])
            j = jobs.setdefault((split, p['movie_path']), {
                'split': split, 'movie': p['movie_path'],
                'session': Path(p['movie_path']).parent.name, 'anns': []})
            j['anns'].append({'frame': int(p['ufmf_index']), 'tgt': int(a['tgt']),
                              'xy': crop_to_full(kp[:, :2], x0, y0, th),
                              'occluded': int((kp[:, 2] == 1).sum()),
                              'box': (x0 - CROP / 2, y0 - CROP / 2, x0 + CROP / 2, y0 + CROP / 2),
                              'image': imgs[a['image_id']]['file_name'], 'json': fname,
                              'ann_id': a['id'], 'coco_frm': int(a['frm']),
                              'behavior': beh.get(str(a.get('behavior')), ''),
                              'condition': cond.get(str(a.get('condition')), '')})
    return sorted(jobs.values(), key=lambda j: (j['split'], j['session'])), names, skeleton


def n_source_frames(movie: str) -> int:
    """Frame count from the UFMF index (no decode)."""
    sys.path.insert(0, APT_DEEPNET)
    import logging
    logging.disable(logging.DEBUG)
    import ufmf
    r = ufmf.FlyMovieEmulator(movie, mode='rb', is_ok_to_write_regenerated_index=False,
                              allow_no_such_frame_errors=True)
    n = r.get_n_frames()
    r.close()
    return n


def plan_sessions(jobs: list[dict], val_frac: float, seed: int) -> list[dict]:
    """Group each (split, movie) job; move whole train groups to `val` until it holds
    `val_frac` of that movie's train labels. Whole groups, so val frames never overlap train."""
    rng = np.random.default_rng(seed)
    out = []
    for j in jobs:
        plan = make_groups([a['frame'] for a in j['anns']], n_source_frames(j['movie']))
        n_lab = [sum(s <= a['frame'] < e for a in j['anns']) for s, e, _ in plan]
        val = set()
        if j['split'] == 'train' and val_frac > 0:
            target, got = val_frac * len(j['anns']), 0
            for i in rng.permutation(len(plan)):
                if got >= target:
                    break
                if got + n_lab[i] > 1.2 * target:
                    continue
                val.add(int(i))
                got += n_lab[i]
            if not val:
                val.add(int(np.argmin(n_lab)))
        for split, idx in ((j['split'], [i for i in range(len(plan)) if i not in val]),
                           ('val', sorted(val))):
            if not idx:
                continue
            groups = [plan[i] for i in idx]
            anns = [a for a in j['anns'] if any(s <= a['frame'] < e for s, e, _ in groups)]
            out.append({**j, 'split': split, 'source_split': j['split'], 'plan': groups,
                        'anns': anns})
    return sorted(out, key=lambda j: (j['split'], j['session']))


def convert(job: dict, out: Path, names: list[str], skeleton: list, dry_run: bool) -> dict:
    """Write one session: PNG frames + tables. Returns a stats dict."""
    sys.path.insert(0, APT_DEEPNET)
    import logging
    logging.disable(logging.DEBUG)
    import cv2
    import scipy.io as sio
    import ufmf
    from aniposelib.cameras import CameraGroup

    movie = job['movie']
    trx = sio.loadmat(str(Path(movie).parent / 'registered_trx.mat'), squeeze_me=True,
                      struct_as_record=False)['trx']
    fps = float(np.atleast_1d(trx)[0].fps)
    reader = ufmf.FlyMovieEmulator(movie, mode='rb', is_ok_to_write_regenerated_index=False,
                                   allow_no_such_frame_errors=True)
    plan = job['plan']
    dst = out / job['split'] / job['session']
    staged = [out / STAGE / sp / job['session'] / 'groups'
              for sp in dict.fromkeys((job['split'], job['source_split']))]
    stat = {'split': job['split'], 'session': job['session'], 'groups': len(plan),
            'frames': sum(e - s for s, e, _ in plan), 'labels': len(job['anns']),
            'labelled_frames': sum(len(f) for _, _, f in plan), 'reused': 0}
    if dry_run:
        reader.close()
        return stat
    if dst.exists():
        shutil.rmtree(dst)
    K, groups, labels, wh = len(names), {}, {}, None
    for start, stop, lframes in plan:
        gid = f'f{start:06d}'
        n = stop - start
        cdir = dst / 'groups' / gid / CAM
        old = next((p / gid / CAM for p in staged
                    if len(list((p / gid / CAM).glob('*.png'))) == n), None)
        if old is not None:
            cdir.parent.mkdir(parents=True)
            old.rename(cdir)
            stat['reused'] += 1
            im = cv2.imread(str(cdir / '000000.png'), cv2.IMREAD_GRAYSCALE)
            wh = (im.shape[1], im.shape[0])
        else:
            cdir.mkdir(parents=True)
            for i in range(n):
                frame, _ = reader.get_frame(start + i)
                wh = (frame.shape[1], frame.shape[0])
                cv2.imwrite(str(cdir / f'{i:06d}.png'), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        mine = [a for a in job['anns'] if start <= a['frame'] < stop]
        tgts = sorted({a['tgt'] for a in mine})
        lab = fmt.empty_labels(len(tgts), n, K, 1, mode3d=False,
                               animal_ids=[f'fly{t:02d}' for t in tgts])
        lab.boxes = np.full((len(tgts), n, 1, 4), np.nan, np.float32)
        lab.instance = np.full((len(tgts), n, 1), fmt.INST_NONE, np.int8)
        for a in mine:
            s, f = tgts.index(a['tgt']), a['frame'] - start
            lab.points2d[s, f, :, 0] = a['xy'].astype(np.float32)
            lab.vis2d[s, f, :, 0] = fmt.VISIBLE
            lab.boxes[s, f, 0] = np.asarray(a['box'], np.float32)
            lab.instance[s, f, 0] = fmt.INST_PRESENT
        groups[gid] = fmt.Group(gid, n, fps=fps, source_video=movie, source_frame_start=start,
                                source_frame_step=1,
                                notes=f'{len(lframes)} labelled frame(s), {len(mine)} labelled fly-frames')
        labels[gid] = lab
    reader.close()
    rig = fmt.Rig(CameraGroup([fmt.nominal_camera(CAM, wh)]), offset={CAM: (0.0, 0.0)},
                  moving={CAM: False}, calibrated={CAM: False})
    occ = sum(a['occluded'] for a in job['anns'])
    fmt.write_session(
        dst, mode='2d', units='px', label_source='annotated', names=names, rig=rig,
        groups=groups, labels=labels,
        skeleton=[[names[a - 1], names[b - 1]] for a, b in skeleton],
        flip_pairs=[list(p) for p in FLIP],
        provenance={
            'source': URL,
            'source_json': SPLITS[job['source_split']],
            'split_note': ('val = whole groups held out of the source TRAIN set (~5% of each '
                           'train movie\'s labels, seeded); same flies/movie as train, disjoint '
                           'frames' if job['split'] == 'val' else f'source {job["split"]} split'),
            'source_movie': movie,
            'annotator': '',
            'annotator_tool': 'APT (Animal Part Tracker); MultiFly v1.0 COCO export (Robie & Branson)',
            'created': datetime.date.today().isoformat(),
            'converter': 'scripts/convert_multifly.py',
            'crop_rule': 'UFMF index = COCO frm - 1; centre = registered_trx (x-1, y-1) of tgt; '
                         'rotation theta + 90 deg, scale 1, 192x192, bilinear; byte-exact on all '
                         '10,895 source crops (reconstruct_crops.py)',
            'keypoint_transform': 'crop px -> full frame by the inverse crop rotation; median '
                                  '0.5-0.7 px vs APT .trk (whose bilateral order is L/R swapped '
                                  'relative to the COCO names; COCO names kept)',
            'occluded_as': 'visible',
            'occluded_note': f'COCO v=1 (occluded, position placed) written visible WITH '
                             f'coordinates: {occ} of {K * len(job["anns"])} point-slots here '
                             f'(~2.3% source-wide). Do not judge a vis head on this root.',
            'instances': 'every labelled fly is `present` (not all flies in a frame are labelled), '
                         'boxed by the axis-aligned 192x192 crop window centred on the crop centre',
            'keypoint_renames': ', '.join(f'{k} -> {v}' for k, v in RENAME.items()),
            'animal_id_source': 'registered_trx target index (tgt), fly##',
            'context': f'each labelled frame centred in [f-{HALF}, f+{HALF}]; overlapping or '
                       f'touching windows merged',
            'pixels': 'UFMF decoded with APT ufmf.FlyMovieEmulator, written as lossless PNG',
            'excluded': 'test_inter/test_intra relabels (another annotator; separate root per spec)',
        })
    return stat


def write_index(path: Path, jobs: list[dict]) -> None:
    """One row per labelled fly-frame: where it lives here, where it came from, its test tags."""
    cols = ['split', 'session', 'group_id', 'frame', 'animal_id', 'source_movie',
            'source_frame', 'coco_json', 'coco_annotation_id', 'coco_image', 'coco_frm',
            'coco_tgt', 'behavior', 'condition']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for j in jobs:
            for s, e, _ in j['plan']:
                for a in sorted((a for a in j['anns'] if s <= a['frame'] < e),
                                key=lambda a: (a['frame'], a['tgt'])):
                    w.writerow([j['split'], j['session'], f'f{s:06d}', a['frame'] - s,
                                f'fly{a["tgt"]:02d}', j['movie'], a['frame'], a['json'],
                                a['ann_id'], a['image'], a['coco_frm'], a['tgt'],
                                a['behavior'], a['condition']])


def main() -> int:
    """Convert every (split, movie) session; 0 on success, 1 when --validate finds errors."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--only', default='')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--val-frac', type=float, default=0.05,
                    help='fraction of each train movie\'s labels moved to val, by whole group')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    jobs, names, skeleton = load_jobs(args.src)
    if args.only:
        jobs = [j for j in jobs if args.only in j['session']]
    jobs = plan_sessions(jobs, args.val_frac, args.seed)
    for sp in ('train', 'val', 'test'):
        mine = [j for j in jobs if j['split'] == sp]
        print(f'{sp:5s} {len(mine):3d} sessions {sum(len(j["plan"]) for j in mine):5d} groups '
              f'{sum(len(j["anns"]) for j in mine):5d} labelled fly-frames')
    if not args.dry_run:
        for j in {(sp, j['session']) for j in jobs for sp in (j['split'], j['source_split'])}:
            src, dst = args.out / j[0] / j[1], args.out / STAGE / j[0] / j[1]
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
    stats = []
    with ProcessPoolExecutor(max(1, args.jobs)) as ex:
        futs = [ex.submit(convert, j, args.out, names, skeleton, args.dry_run) for j in jobs]
        for fu in futs:
            st = fu.result()
            stats.append(st)
            print(f'  {st["split"]:5s} {st["session"]:62s} {st["groups"]:4d} groups '
                  f'{st["frames"]:6d} frames {st["labels"]:5d} labels '
                  f'({st["reused"]} groups reused)', flush=True)
    print(f'total: {sum(s["groups"] for s in stats)} groups, {sum(s["frames"] for s in stats)} '
          f'frames, {sum(s["labels"] for s in stats)} labelled fly-frames, '
          f'{sum(s["reused"] for s in stats)} groups reused')
    if not args.dry_run:
        shutil.rmtree(args.out / STAGE, ignore_errors=True)
        if not args.only:
            write_index(args.out / 'labels_index.csv', jobs)
    if args.validate and not args.dry_run:
        bad = 0
        for st in stats:
            sess = fmt.Session.load(args.out / st['split'] / st['session'])
            errs = fmt.validate_session(sess)
            bad += bool(errs)
            for e in errs[:5]:
                print(f'  [invalid] {st["session"]}: {e}')
        print(f'validate: {len(stats) - bad}/{len(stats)} sessions clean')
        return int(bad > 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
