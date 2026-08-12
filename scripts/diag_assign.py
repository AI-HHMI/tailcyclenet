#!/usr/bin/env python
"""What the centre-prior assignment actually hands the loss, on a real dataset. No model.

    pixi run python scripts/diag_assign.py --data <root> --boxes instances

Three numbers per operator, over `--n` sampled views:

- **pos/img** -- how much supervision there is at all.
- **outside** -- the fraction of positives whose anchor centre is NOT inside the box it was
  assigned. A box is `centre +- exp(ltrb) * stride`, so those targets are unreachable while
  objectness is taught to fire there. **Under `&` this must be 0.0%.**
- **starved** -- the fraction of GT boxes left with no positive at all. Tightening the rule can
  only take positives away, so this is the risk `&` carries; posetail-pose covers it with a
  sub-cell nearest-anchor fallback we do not have. Measured 0.0% on every shipped dataset at
  every input size -- **if this stops being 0, that fallback is what is missing.**

Cheap enough to re-run after any change to the crop rule, the box source or the input size, all
three of which move it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import BOX_SOURCES
from tailcyclenet.detector import BoxDataset, YOLOXNano, assign
from tailcyclenet.detector.assign import CENTER_RADIUS
from tailcyclenet.format import load_datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_detector import default_input_wh                          # noqa: E402


def stats(anchors, gt, op):
    """(n_pos, n_outside, n_starved, n_gt) for one image under `op` in {'and', 'or'}."""
    keep = torch.isfinite(gt).all(-1)
    if not keep.any():
        return 0, 0, 0, 0
    g = gt[keep]
    cx, cy, stride = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    inside = ((cx[:, None] > g[None, :, 0]) & (cx[:, None] < g[None, :, 2]) &
              (cy[:, None] > g[None, :, 1]) & (cy[:, None] < g[None, :, 3]))
    gcx, gcy = (g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2
    r = CENTER_RADIUS * stride[:, None]
    near = ((cx[:, None] - gcx[None]).abs() < r) & ((cy[:, None] - gcy[None]).abs() < r)
    ok = (inside & near) if op == 'and' else (inside | near)
    d = (cx[:, None] - gcx[None]) ** 2 + (cy[:, None] - gcy[None]) ** 2
    d = torch.where(ok, d, torch.full_like(d, float('inf')))
    best = d.argmin(1)
    pos = torch.nonzero(torch.isfinite(d.min(1).values), as_tuple=True)[0]
    gix = best[pos]
    outside = int((~inside[pos, gix]).sum())
    starved = int(g.shape[0] - len(set(gix.tolist())))
    return pos.numel(), outside, starved, g.shape[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--split', default='train')
    ap.add_argument('--boxes', default='keypoints', choices=BOX_SOURCES)
    ap.add_argument('--input-wh', type=int, nargs=2, default=None)
    ap.add_argument('--n', type=int, default=200, help='views to sample')
    ap.add_argument('--frames-per-group', type=int, default=8)
    args = ap.parse_args()

    roots = load_datasets(args.data)
    wh = tuple(args.input_wh) if args.input_wh else default_input_wh(roots[0])
    ds = BoxDataset(args.data, args.split, input_wh=wh, box_source=args.boxes,
                    max_frames_per_group=args.frames_per_group)
    anchors = YOLOXNano().anchor_points(wh[1], wh[0], torch.device('cpu'))
    ix = np.random.default_rng(0).choice(len(ds), min(args.n, len(ds)), replace=False)

    # The BOXES are all this needs, and decoding the image is 95% of the cost -- so read the
    # target the same way `BoxDataset` does and skip `__getitem__` entirely.
    boxes = []
    for i in ix:
        boxes.append(ds.boxes_for(int(i)))
    print(f'{args.data.name}/{args.split}  {wh[0]}x{wh[1]}  boxes={args.boxes}  '
          f'{len(boxes)} views')
    side = torch.cat([(b[:, 2:] - b[:, :2]).flatten() for b in boxes])
    print(f'median box side {float(np.nanmedian(side.numpy())):.1f} px\n')
    print(f'{"op":>4s} {"pos/img":>9s} {"outside":>9s} {"starved":>9s}')
    for op in ('or', 'and'):
        p = o = s = n = 0
        for b in boxes:
            dp, do, ds_, dn = stats(anchors, b, op)
            p, o, s, n = p + dp, o + do, s + ds_, n + dn
        print(f'{op:>4s} {p / len(boxes):9.1f} {o / max(p, 1):8.1%} {s / max(n, 1):8.1%}')

    # `assign` itself, so this cannot drift from what the loss actually sees.
    p = o = 0
    for b in boxes:
        pos, gix = assign(anchors, b)
        p += pos.numel()
        if pos.numel():
            cx, cy = anchors[pos, 0], anchors[pos, 1]
            box = b[gix]                 # `assign` returns ORIGINAL gt rows, not finite-subset ones
            o += int((~((cx > box[:, 0]) & (cx < box[:, 2]) &
                        (cy > box[:, 1]) & (cy < box[:, 3]))).sum())
    print(f'\nassign() as shipped: {p / len(boxes):.1f} pos/img, {o / max(p, 1):.1%} outside')


if __name__ == '__main__':
    main()
