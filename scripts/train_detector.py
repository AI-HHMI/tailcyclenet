#!/usr/bin/env python
"""Train the box predictor. ONE DETECTOR PER DATASET.

    pixi run python scripts/train_detector.py --data <dataset root> --out runs/det-rat-city

The input size is dataset-specific and it matters more than it looks. rat-city's frames are
4696x2048 (2.29:1); letterboxed into a square 416 that scales by min(416/2048, 416/4696) = 0.089,
so the frame becomes 416x181 in a 416x416 canvas -- 56% black padding -- and the median rat
arrives at 15.8 x 12.5 px. YOLOX pools at strides 8/16/32, so that animal is ~2 x 1.6 cells at
the finest level and absent from the other two: two thirds of the FPN cannot represent it.
Measured against branson-fly (same detector, same 416, but a square 1024x1024 frame) the median
fly arrives at 26.5 x 28.1 px and reaches AP50 0.985 where rat-city sits near 0.50.

So `--input-wh` defaults to an aspect-matched size rather than a square, and training the
detector across datasets is not offered: one letterbox cannot serve both.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.detector import (BoxDataset, YOLOXNano, box_collate, box_iou, decode,
                                   detector_loss)


def default_input_wh(dataset, target_px=416 * 416):
    """An input size matched to the frame's aspect ratio, at roughly a square-416 pixel budget."""
    sess = next(iter(next(iter(dataset.sessions.values()))))
    w, h = sess.rig.size(sess.cam_names[0])
    ar = w / h
    ow = int(round((target_px * ar) ** 0.5 / 32) * 32)
    oh = int(round((target_px / ar) ** 0.5 / 32) * 32)
    return max(ow, 64), max(oh, 64)


@torch.no_grad()
def evaluate(model, loader, device, iou_thresh=0.5, limit=50):
    """AP50-ish: recall at IoU 0.5 with one box per animal. Enough to see training work."""
    model.eval()
    hits = total = 0
    for i, (x, gt) in enumerate(loader):
        if i >= limit:
            break
        x = x.to(device)
        obj, boxes = model(x)
        anchors = model.anchor_points(x.shape[-2], x.shape[-1], device)
        for b in range(x.shape[0]):
            g = gt[b][torch.isfinite(gt[b]).all(-1)]
            if not g.numel():
                continue
            pred, _ = decode(obj[b], boxes[b], top_k=max(1, g.shape[0]))
            total += g.shape[0]
            if pred.numel():
                hits += int((box_iou(g.to(device), pred).max(1).values >= iou_thresh).sum())
    model.train()
    return hits / total if total else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, type=Path, help='ONE dataset root')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--input-wh', type=int, nargs=2, default=None)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--frames-per-group', type=int, default=40)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    probe = BoxDataset(args.data, 'train', input_wh=(64, 64), max_frames_per_group=1)
    wh = tuple(args.input_wh) if args.input_wh else default_input_wh(probe.ds)
    print(f'input {wh[0]}x{wh[1]}  (frame '
          f'{probe.ds.all_sessions()[0].rig.size(probe.ds.all_sessions()[0].cam_names[0])})')

    train = BoxDataset(args.data, 'train', input_wh=wh,
                       max_frames_per_group=args.frames_per_group)
    print(f'train: {len(train)} views')
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=box_collate, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = None
    try:
        val = BoxDataset(args.data, 'val', input_wh=wh, max_frames_per_group=8)
        val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size,
                                                 num_workers=2, collate_fn=box_collate)
        print(f'val:   {len(val)} views')
    except ValueError as e:
        print(f'val:   none ({e})')

    model = YOLOXNano().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'YOLOX-Nano: {n / 1e6:.2f}M params')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    args.out.mkdir(parents=True, exist_ok=True)
    it, t0, running = 0, time.time(), []
    model.train()
    while it < args.iters:
        for x, gt in loader:
            if it >= args.iters:
                break
            x, gt = x.to(device), gt.to(device)
            obj, boxes = model(x)
            anchors = model.anchor_points(x.shape[-2], x.shape[-1], device)
            loss, parts = detector_loss(obj, boxes, anchors, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            running.append(float(loss))
            it += 1
            if it % 50 == 0:
                print(f'{it:7d}/{args.iters}  loss {np.mean(running):7.4f}  '
                      f'obj {parts["obj"]:6.3f}  box {parts["box"]:6.3f}  '
                      f'pos {parts["n_pos"]:4d}  {(time.time() - t0) / 50:5.3f}s/it', flush=True)
                running, t0 = [], time.time()
            if it % 2000 == 0 or it == args.iters:
                torch.save({'iteration': it, 'model_state': model.state_dict(),
                            'input_wh': wh, 'dataset': train.ds.name},
                           args.out / 'detector.pth')
                if val_loader is not None:
                    print(f'   recall@IoU0.5 = {evaluate(model, val_loader, device):.4f}')
    print(f'done: {it} iterations -> {args.out / "detector.pth"}')


if __name__ == '__main__':
    main()
