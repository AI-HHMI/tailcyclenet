"""Re-cut a video-backed dataset root into short JPEG groups around its labelled frames.

A root whose groups are whole recordings (`groups/<gid>/<cam>.mp4`, tens of thousands of frames,
a few dozen labelled) costs the full video on disk. This writes a new root in which every source
group becomes several short groups, each an image directory per camera
(`groups/<gid>/<cam>/000000.jpg ...`), and nothing else changes: `session.toml` and
`calibration.toml` are copied byte-for-byte, every label table keeps every row and column, only
`group_id` and `frame` are rewritten to address the new groups (spec section 6: a parquet `frame`
indexes the group's own pixels; `source_frame_start` records where frame 0 came from).

Grouping rule: each labelled frame f gets a `--context`-frame window [f - pre, f - pre + context)
(pre = context // 2), shifted inside [0, n_frames). Windows that overlap, or whose gap is at most
`--merge-gap` frames, are merged into one group, so a group holds as many labels as the spacing
allows and is never shorter than `--context` (unless the source group itself is).

Usage:
    pixi run python scripts/compress_video_groups.py --src <root> --dst <new root> [--workers 32]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LABEL_TABLES = ('keypoints.pq', 'points3d.pq', 'points2d.pq', 'instances.pq', 'extrinsics.pq')
CHUNK = 64


def plan_groups(frames, n_frames, context, merge_gap):
    """Return [(start, end)] half-open windows covering every labelled frame."""
    pre = context // 2
    out = []
    for f in sorted(set(int(x) for x in frames)):
        a = max(0, min(f - pre, n_frames - context))
        b = min(n_frames, a + context)
        if out and a <= out[-1][1] + merge_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def _labelled_frames(sess_dir, gid):
    """Every frame of group `gid` that any label table of `sess_dir` has a row for."""
    frames = set()
    for t in LABEL_TABLES:
        p = sess_dir / t
        if p.exists():
            tab = pq.read_table(p, columns=['group_id', 'frame'])
            m = pc.equal(tab['group_id'], gid)
            frames |= set(tab.filter(m)['frame'].to_pylist())
    return frames


def _video_for(gdir, cam):
    """The video file of camera `cam` in group directory `gdir`."""
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        p = gdir / f'{cam}{ext}'
        if p.exists():
            return p
    raise FileNotFoundError(f'{gdir}: no video for camera {cam}')


def extract(video, out_dirs, windows, quality):
    """Decode `windows` [(start, end)] of `video` into JPEG dirs `out_dirs` (one per window)."""
    import cv2

    from tailcyclenet.data.video import open_reader

    r = open_reader(str(video))
    n_written = 0
    try:
        for (a, b), od in zip(windows, out_dirs):
            od = Path(od)
            if od.is_dir() and len(list(od.glob('*.jpg'))) == b - a:
                continue
            tmp = od.with_name(od.name + '.tmp')
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            for c0 in range(a, b, CHUNK):
                idx = list(range(c0, min(b, c0 + CHUNK)))
                imgs = r.get_batch(idx)
                for i, im in zip(idx, imgs):
                    ok = cv2.imwrite(str(tmp / f'{i - a:06d}.jpg'),
                                     cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not ok:
                        raise RuntimeError(f'failed to write {tmp}/{i - a:06d}.jpg')
                    n_written += 1
            shutil.rmtree(od, ignore_errors=True)
            tmp.rename(od)
    finally:
        r.close()
    return str(video), n_written


def rewrite_tables(sess_dir, out_dir, plan):
    """Rewrite groups.pq and every label table for the new groups. `plan`: gid -> [(new, a, b)]."""
    g = pq.read_table(sess_dir / 'groups.pq')
    rows = g.to_pylist()
    new_rows = []
    for row in rows:
        start0 = row.get('source_frame_start') or 0
        step = row.get('source_frame_step') or 1
        for new, a, b in plan[row['group_id']]:
            nr = dict(row)
            nr['group_id'] = new
            nr['n_frames'] = b - a
            if 'source_frame_start' in nr or 'source_frame_step' in nr:
                nr['source_frame_start'] = start0 + a * step
            note = f'frames [{a}, {b}) of source group {row["group_id"]}'
            nr['notes'] = f'{row["notes"]}; {note}' if row.get('notes') else note
            new_rows.append(nr)
    schema = g.schema
    for col in ('source_frame_start', 'notes'):
        if col not in schema.names:
            schema = schema.append(pa.field(col, pa.int32() if col != 'notes' else pa.string()))
    pq.write_table(pa.Table.from_pylist(new_rows, schema=schema), out_dir / 'groups.pq')

    counts = {}
    for t in LABEL_TABLES:
        p = sess_dir / t
        if not p.exists():
            continue
        tab = pq.read_table(p)
        gids = tab['group_id'].to_pylist()
        frames = np.asarray(tab['frame'].to_pylist(), dtype=np.int64)
        new_g, new_f = [], np.empty_like(frames)
        for i, (gid, f) in enumerate(zip(gids, frames)):
            hit = [(new, a) for new, a, b in plan[gid] if a <= f < b]
            if len(hit) != 1:
                raise RuntimeError(f'{p}: row {i} ({gid}, frame {f}) falls in {len(hit)} groups')
            new_g.append(hit[0][0])
            new_f[i] = f - hit[0][1]
        gi, fi = tab.schema.get_field_index('group_id'), tab.schema.get_field_index('frame')
        tab = tab.set_column(gi, tab.schema.field(gi), pa.array(new_g, tab.schema.field(gi).type))
        tab = tab.set_column(fi, tab.schema.field(fi), pa.array(new_f, tab.schema.field(fi).type))
        pq.write_table(tab, out_dir / t)
        counts[t] = tab.num_rows
    return counts


def main():
    """Plan the new groups for every session under --src and write them to --dst."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--src', required=True, type=Path)
    ap.add_argument('--dst', required=True, type=Path)
    ap.add_argument('--context', type=int, default=32, help='minimum group length (frames)')
    ap.add_argument('--merge-gap', type=int, default=32,
                    help='merge windows separated by at most this many unlabelled frames')
    ap.add_argument('--quality', type=int, default=95, help='JPEG quality')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    sessions = sorted(p.parent for p in args.src.glob('*/*/session.toml'))
    if not sessions:
        sys.exit(f'{args.src}: no <split>/<session>/session.toml found')
    jobs, tables = [], []
    for sd in sessions:
        od = args.dst / sd.relative_to(args.src)
        groups = pq.read_table(sd / 'groups.pq').to_pylist()
        plan = {}
        for row in groups:
            gid, n = row['group_id'], int(row['n_frames'])
            frames = _labelled_frames(sd, gid)
            if not frames:
                print(f'{sd.name}/{gid}: no labels, dropped', flush=True)
                plan[gid] = []
                continue
            if max(frames) >= n or min(frames) < 0:
                sys.exit(f'{sd}/{gid}: label frame outside [0, {n})')
            wins = plan_groups(frames, n, args.context, args.merge_gap)
            plan[gid] = [(f'{gid}_{a:06d}', a, b) for a, b in wins]
            gdir = sd / 'groups' / gid
            cams = sorted(p.stem for p in gdir.iterdir()
                          if p.is_file() and not p.name.startswith('.')
                          and p.suffix in ('.mp4', '.avi', '.mov', '.mkv'))
            for cam in cams:
                outs = [od / 'groups' / new / cam for new, _, _ in plan[gid]]
                jobs.append((_video_for(gdir, cam), outs, wins))
            lens = [b - a for a, b in wins]
            print(f'{sd.name}/{gid}: {len(frames)} labelled frames -> {len(wins)} groups, '
                  f'{sum(lens)} frames (min {min(lens)}, max {max(lens)}) x {len(cams)} cams',
                  flush=True)
        tables.append((sd, od, plan))
    print(f'{len(jobs)} camera videos, '
          f'{sum(sum(b - a for a, b in w) for _, _, w in jobs)} images to write', flush=True)
    if args.dry_run:
        return

    for sd, od, plan in tables:
        od.mkdir(parents=True, exist_ok=True)
        for f in sd.iterdir():
            if f.is_file() and f.suffix == '.toml':
                shutil.copy2(f, od / f.name)
        counts = rewrite_tables(sd, od, plan)
        print(f'{od}: tables {counts}', flush=True)

    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(extract, v, o, w, args.quality) for v, o, w in jobs]
        for fu in as_completed(futs):
            v, n = fu.result()
            done += 1
            print(f'[{done}/{len(jobs)}] {v}: {n} images', flush=True)


if __name__ == '__main__':
    main()
