#!/usr/bin/env python
"""Render a tailcycle dataset so a human can see what is actually in it.

    pixi run python scripts/render_dataset.py --data <root|session> --out <outdir>

Three views per sampled group: the sheet (labelled frame, all overlays), per-animal crops at
NATIVE resolution (the only view a keypoint's position is checkable in), and optionally a
video of the whole group with the frame index burned in. `regions.pq` is drawn in CYAN: it marks
the area the annotator certified as completely labelled, not an animal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt
from tailcyclenet.dataset import read_frames

# BGR (cv2's order). Cyan for regions, instance-status colours, fixed per-keypoint palette.
REGION = (255, 255, 0)
INST_COLOR = {fmt.INST_LABELED: (80, 220, 80), fmt.INST_PRESENT: (60, 200, 255),
              fmt.INST_ABSENT: (150, 150, 150)}
PALETTE = [(60, 60, 255), (60, 200, 255), (60, 255, 60), (255, 200, 60), (255, 60, 200),
           (200, 60, 255), (255, 255, 60), (60, 255, 200), (160, 160, 255), (255, 160, 160)]


def _scale(wh) -> int:
    """Line/marker size for a frame this wide. A 2 px line is invisible on a 4696 px frame."""
    return max(1, int(round(wh[0] / 900)))


def draw(im: np.ndarray, lab: fmt.Labels, t: int, ci: int, names: list[str],
         skeleton: list[list[str]]) -> np.ndarray:
    """Every overlay for one (frame, camera), on a COPY at native resolution.

    A keypoint may be POSITIONED yet carry no coordinates -- a coordinate-free visibility
    observation (§7); those are skipped.
    """
    import cv2

    im = np.ascontiguousarray(im[:, :, ::-1]).copy()
    s = _scale((im.shape[1], im.shape[0]))
    ix = {n: i for i, n in enumerate(names)}

    if lab.regions is not None and len(lab.regions):
        for r in lab.regions[(lab.regions[:, 0] == t) & (lab.regions[:, 1] == ci)]:
            p0, p1 = (int(r[2]), int(r[3])), (int(r[4]), int(r[5]))
            cv2.rectangle(im, p0, p1, REGION, 3 * s)
            cv2.putText(im, 'labelled_complete', (p0[0] + 4 * s, p0[1] + 14 * s),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, REGION, s)

    for a in range(lab.vis2d.shape[0] if lab.vis2d is not None else 0):
        if lab.instance is not None and lab.instance[a, t, ci] != fmt.INST_NONE:
            box, st = lab.boxes[a, t, ci], int(lab.instance[a, t, ci])
            if np.isfinite(box).all():
                p0 = (int(box[0]), int(box[1]))
                cv2.rectangle(im, p0, (int(box[2]), int(box[3])), INST_COLOR[st], 2 * s)
                cv2.putText(im, lab.animal_ids[a], (p0[0], p0[1] - 4 * s),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, INST_COLOR[st], s)

        vis, pts = lab.vis2d[a, t, :, ci], lab.points2d[a, t, :, ci]
        for k in np.flatnonzero(np.isin(vis, fmt.POSITIONED)):
            if not np.isfinite(pts[k]).all():
                continue
            cv2.circle(im, (int(pts[k][0]), int(pts[k][1])), 3 * s, PALETTE[k % len(PALETTE)], -1)
        for u, v in skeleton:
            if u not in ix or v not in ix:
                continue
            p, q = pts[ix[u]], pts[ix[v]]
            if np.isfinite(p).all() and np.isfinite(q).all():
                cv2.line(im, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (255, 255, 255), s)
    return im


def label_frames(lab: fmt.Labels) -> list[int]:
    """Frames carrying any assessment at all. Empty for a pure context group."""
    if lab.vis2d is None:
        return []
    return np.flatnonzero((lab.vis2d != fmt.UNLABELED).any((0, 2, 3))).tolist()


def fit(im: np.ndarray, width: int) -> np.ndarray:
    """Downscale a frame to at most `width` pixels wide, preserving aspect.

    Inputs: im -- a BGR frame.
            width -- maximum output width.
    Outputs: the resized frame (unchanged when already at or below width).
    """
    import cv2
    if im.shape[1] <= width:
        return im
    h = int(round(im.shape[0] * width / im.shape[1]))
    return cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)


def render_group(sess: fmt.Session, gid: str, out: Path, stem: str, args) -> dict:
    """Render one group: sheets, per-animal crops, and optionally a video.

    Inputs: sess -- the session holding the group.
            gid -- the group id.
            out -- output directory.
            stem -- the file-name stem for this group's renders.
            args -- parsed CLI args (width, crops, video, fps).
    Outputs: per-kind render counts for the group.
    Side effects: writes jpg/mp4 files under out.

    Per-animal crops are cut from the already-drawn image so they match the sheet at native
    resolution.
    """
    import cv2

    group = sess.groups[gid]
    lab = sess.labels(gid)
    frames = label_frames(lab)
    t = frames[len(frames) // 2] if frames else group.n_frames // 2
    stat = {'sheets': 0, 'crops': 0, 'videos': 0, 'label_frames': len(frames)}

    for ci, cam in enumerate(sess.cam_names):
        im = read_frames(group, cam, [t])[0]
        if im is None:
            print(f'  [warn] {stem}/{cam}: frame {t} unreadable')
            continue
        drawn = draw(im, lab, t, ci, sess.names, sess.skeleton)
        tag = f'{stem}_{cam}' if len(sess.cam_names) > 1 else stem
        cv2.imwrite(str(out / f'{tag}_f{t}.jpg'), fit(drawn, args.width),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        stat['sheets'] += 1

        if args.crops and lab.instance is not None:
            H, W = drawn.shape[:2]
            for a in np.flatnonzero(lab.instance[:, t, ci] == fmt.INST_LABELED):
                box = lab.boxes[a, t, ci]
                if not np.isfinite(box).all():
                    continue
                pad = int(0.25 * max(box[2] - box[0], box[3] - box[1]))
                x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
                x1, y1 = min(W, int(box[2]) + pad), min(H, int(box[3]) + pad)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                cv2.imwrite(str(out / f'{tag}_f{t}_{lab.animal_ids[a]}.jpg'),
                            drawn[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                stat['crops'] += 1

        if args.video:
            ims = read_frames(group, cam, list(range(group.n_frames)))
            if any(i is None for i in ims):
                print(f'  [warn] {stem}/{cam}: unreadable frames, no video')
                continue
            first = fit(draw(ims[0], lab, 0, ci, sess.names, sess.skeleton), args.width)
            vw = cv2.VideoWriter(str(out / f'{tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps, (first.shape[1], first.shape[0]))
            for i, raw in enumerate(ims):
                fr = fit(draw(raw, lab, i, ci, sess.names, sess.skeleton), args.width)
                mark = '  <-- LABELLED' if i in frames else ''
                cv2.putText(fr, f'{gid} frame {i}/{group.n_frames - 1}'
                                f' (source {group.source_frame_start + i}){mark}',
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if mark else (255, 255, 255), 2)
                vw.write(fr)
            vw.release()
            stat['videos'] += 1
    return stat


def main() -> int:
    """Render sampled groups across a dataset; 0 on success.

    Inputs: argv (via argparse): --data, --out, --n, --split, --session,
            --group, --width, --video, --fps, --no-crops, --seed.
    Outputs: process exit code.
    Side effects: writes renders under --out; prints a per-group summary.

    Labelled groups are preferred when sampling -- a random draw would be dominated by whichever
    unlabelled ones happen to be sampled.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True,
                    help='a dataset root, a folder of roots, or one session directory')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=3, help='groups per session (0 = all)')
    ap.add_argument('--split', default='', help='only this split')
    ap.add_argument('--session', default='', help='substring of the session id')
    ap.add_argument('--group', default='', help='substring of the group id')
    ap.add_argument('--width', type=int, default=1800, help='max width of a rendered frame')
    ap.add_argument('--video', action='store_true', help='also write one mp4 per group')
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--no-crops', dest='crops', action='store_false',
                    help='skip the per-animal native-resolution crops')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if (args.data / 'session.toml').exists():
        sessions = [fmt.Session.load(args.data)]
    else:
        sessions = [s for ds in fmt.load_datasets(args.data) for s in ds.all_sessions()]
    if args.split:
        sessions = [s for s in sessions if s.split == args.split]
    if args.session:
        sessions = [s for s in sessions if args.session in s.session_id]
    if not sessions:
        raise SystemExit('no sessions matched')

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total = {'sheets': 0, 'crops': 0, 'videos': 0}
    for s in sessions:
        gids = [g for g in s.groups if args.group in g]
        if args.n and len(gids) > args.n:
            gids = [gids[i] for i in sorted(rng.choice(len(gids), args.n, replace=False))]
        for gid in gids:
            stem = f'{s.split}_{s.session_id}_{gid}'
            stat = render_group(s, gid, args.out, stem, args)
            for k in total:
                total[k] += stat[k]
            print(f'  {stem}: {stat["sheets"]} sheet(s), {stat["crops"]} crop(s), '
                  f'{stat["videos"]} video(s), {stat["label_frames"]} labelled frame(s)')
    print(f'\n{len(sessions)} session(s) -> {args.out}: '
          f'{total["sheets"]} sheets, {total["crops"]} crops, {total["videos"]} videos')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
