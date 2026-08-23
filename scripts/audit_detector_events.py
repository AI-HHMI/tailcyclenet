#!/usr/bin/env python
"""Make source-pixel contact sheets for manual detector-event auditing.

The input JSONL comes from ``eval_detector_trace.py --events-out``.  This tool deliberately does
not infer visibility: every panel is emitted with a blank audit label for a human to mark as
``visible``, ``partially_visible``, ``not_visible``, or ``unscorable``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tailcyclenet.format import load_datasets, sessions_for
from tailcyclenet.video import open_reader

LABELS = ('visible', 'partially_visible', 'not_visible', 'unscorable')
REASONS = ('localization', 'score', 'nms', 'top_k', 'kept')


def _sessions(path: Path, split: str):
    if (path / 'session.toml').exists():
        _, sessions = sessions_for(path, split)
        return {s.session_id: s for s in sessions}
    return {s.session_id: s for ds in load_datasets(path)
            for s in ds.sessions.get(split, [])}


def _read_frame(group, camera, frame, readers):
    kind, source, _ = group.source(camera)
    if kind == 'video':
        key = str(source)
        reader = readers.setdefault(key, open_reader(key))
        return reader.get_batch([int(frame)])[0]
    path = source / f'{int(frame):06d}{group.source(camera)[2]}'
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def _panel(frame, box, title, size):
    image = Image.fromarray(frame)
    x0, y0, x1, y1 = [float(v) for v in box]
    side = max(x1 - x0, y1 - y0, 32.0) * 2.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = max(0, min(image.width - side, cx - side / 2))
    top = max(0, min(image.height - side, cy - side / 2))
    right, bottom = min(image.width, left + side), min(image.height, top + side)
    crop = image.crop((left, top, right, bottom)).resize((size, size), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(crop)
    scale_x, scale_y = size / (right - left), size / (bottom - top)
    rect = tuple((v - o) * scale for v, o, scale in
                 zip((x0, y0, x1, y1), (left, top, left, top), (scale_x, scale_y,
                                                                   scale_x, scale_y)))
    draw.rectangle(rect, outline='red', width=3)
    draw.rectangle((0, 0, size, 28), fill='black')
    draw.text((4, 5), title, fill='white')
    return crop


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--events', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--labels-out', required=True, type=Path)
    ap.add_argument('--per-reason', type=int, default=12)
    ap.add_argument('--columns', type=int, default=4)
    ap.add_argument('--panel-size', type=int, default=320)
    args = ap.parse_args()

    by_reason = defaultdict(list)
    with args.events.open() as f:
        for line in f:
            event = json.loads(line)
            if event['reason'] in REASONS and len(by_reason[event['reason']]) < args.per_reason:
                by_reason[event['reason']].append(event)
    sessions = _sessions(args.data, args.split)
    readers = {}
    selected = [event for reason in REASONS for event in by_reason[reason]]
    panels = []
    labels = []
    try:
        for n, event in enumerate(selected):
            sess = sessions[event['session']]
            group = sess.groups[event['group']]
            frame = _read_frame(group, event['camera'], event['frame'], readers)
            title = f"{event['reason']} f{event['frame']} {event['camera']}"
            panels.append(_panel(frame, event['gt_box'], title, args.panel_size))
            labels.append({**event, 'audit_label': None, 'allowed_labels': LABELS,
                           'panel_index': n})
    finally:
        for reader in readers.values():
            reader.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(panels) + args.columns - 1) // args.columns
    sheet = Image.new('RGB', (args.columns * args.panel_size, rows * args.panel_size), 'gray')
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % args.columns) * args.panel_size,
                            (i // args.columns) * args.panel_size))
    sheet.save(args.out)
    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    args.labels_out.write_text(''.join(json.dumps(row) + '\n' for row in labels))
    print(f'wrote {args.out} and {args.labels_out} ({len(labels)} panels)')


if __name__ == '__main__':
    main()
