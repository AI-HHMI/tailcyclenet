#!/usr/bin/env python
"""Create deterministic, session-stratified fixed detector evaluation frame lists."""
import argparse
import json
from pathlib import Path

import numpy as np

from tailcyclenet.detector.evaluate import _labelled_frames
from tailcyclenet.format import load_datasets


def make_evalset(data, splits, count=12, seed=0):
    """Select labelled frames round-robin across sessions and groups."""
    roots = load_datasets(data)
    candidates = []
    for split in splits:
        for root in roots:
            for sess in root.sessions.get(split, []):
                for gid in sess.groups:
                    frames = _labelled_frames(sess, gid)
                    if len(frames):
                        candidates.extend((sess, gid, int(frame), split) for frame in frames)
    rng = np.random.default_rng(seed)
    by_session = {}
    for row in candidates:
        by_session.setdefault(row[0].session_id, []).append(row)
    session_ids = sorted(by_session)
    for sid, rows in by_session.items():
        by_group = {}
        for row in rows:
            by_group.setdefault(row[1], []).append(row)
        group_ids = sorted(by_group)
        rng.shuffle(group_ids)
        for group_rows in by_group.values():
            rng.shuffle(group_rows)
        rows[:] = [by_group[gid][round_index] for round_index in range(
            max(map(len, by_group.values()))) for gid in group_ids
            if round_index < len(by_group[gid])]
    selected = []
    cursor = {s: 0 for s in session_ids}
    while len(selected) < count and session_ids:
        moved = False
        for sid in session_ids:
            rows = by_session[sid]
            while cursor[sid] < len(rows):
                sess, gid, frame, split = rows[cursor[sid]]
                cursor[sid] += 1
                selected.append({'session': sess.session_id, 'group': gid,
                                 'frame': frame, 'split': split})
                moved = True
                break
            if len(selected) >= count:
                break
        if not moved:
            break
    return {'frames': selected}


def main():
    """Build and report one fixed detector eval set."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--split', nargs='+', default=['val'])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--count', type=int, default=12)
    args = ap.parse_args()
    result = make_evalset(args.data, args.split, args.count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    roots = load_datasets(args.data)
    lookup = {s.session_id: s for r in roots for s in r.all_sessions()}
    views = sum(len(lookup[x['session']].cam_names) for x in result['frames'])
    no_visible = 0
    for item in result['frames']:
        sess = lookup[item['session']]
        lab = sess.labels(item['group'])
        for ci in range(len(sess.cam_names)):
            if lab.points2d is not None:
                points = lab.points2d[:, item['frame'], :, ci]
                width, height = sess.rig.size(sess.cam_names[ci])
                visible = (np.isfinite(points).all(-1) & (points[..., 0] >= 0) &
                           (points[..., 0] < width) & (points[..., 1] >= 0) &
                           (points[..., 1] < height)).any()
            else:
                visible = lab.vis3d is not None and np.isfinite(
                    lab.points3d[:, item['frame']]).all(-1).any()
            no_visible += not bool(visible)
    print(f'{len(result["frames"])} frames x cameras = {views} views; {no_visible} views with no visible animal')


if __name__ == '__main__':
    main()
