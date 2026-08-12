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

`--boxes instances` regresses the dataset's own `instances.pq` extent instead of the keypoint
extent. rat-city wants it: its converter dropped noisy points, so 26k train instances carry no
finite keypoint at all and would otherwise be trained as "no animal here".

Every checkpoint is written as its own `detector_it<n>.pth` WITH its scores inside, plus a
`metrics.json` of the whole history, and both splits are scored each time. A single rolling
`detector.pth` carrying no score cannot be selected on -- johnson peaked at val recall 0.871 and
shipped 0.706 -- and the TRAIN score is what says whether a dataset's problem is generalisation
(a train/val gap) or capacity to fit at all (no gap, low absolute recall). Scoring is
`detector.evaluate`, the same code `scripts/eval_detector.py` reports with.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.crop import BOX_SOURCES
from tailcyclenet.dataset import worker_init
from tailcyclenet.format import load_datasets
from tailcyclenet.detector import BoxDataset, ChunkShuffle, YOLOXNano, box_collate, detector_loss
from tailcyclenet.detector.evaluate import overall, score_dataset


def default_input_wh(dataset, target_px=416 * 416):
    """An input size matched to the frame's aspect ratio, at roughly a square-416 pixel budget."""
    sess = next(iter(next(iter(dataset.sessions.values()))))
    w, h = sess.rig.size(sess.cam_names[0])
    ar = w / h
    ow = int(round((target_px * ar) ** 0.5 / 32) * 32)
    oh = int(round((target_px / ar) ** 0.5 / 32) * 32)
    return max(ow, 64), max(oh, 64)


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
    ap.add_argument('--min-crop-dim', type=int, default=64,
                    help='MUST equal the pose run\'s [data].min_crop_dim -- it is the same crop '
                         'rule. Stored in the checkpoint, and scripts/infer.py refuses a mismatch.')
    ap.add_argument('--boxes', default='keypoints', choices=BOX_SOURCES,
                    help='what the regression target bounds. `instances` needs a dataset whose '
                         'instances.pq boxes ARE crop extents -- rat-city\'s are; johnson-mouse '
                         'ships COCO boxes and calms21 MARS ones, which are not.')
    ap.add_argument('--val-frames-per-group', type=int, default=8,
                    help='A DATASET WITH ONE GROUP GETS ONE GROUP\'S WORTH OF VAL. rat-city and '
                         'branson-fly each hold a single val group, so the default 8 makes the '
                         'recall readout 8 images; raise it to the group\'s labelled length.')
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--eval-batches', type=int, default=25,
                    help='batches per split at each checkpoint. TRAIN is scored too: the '
                         'train/val gap is what says whether a dataset needs augmentation '
                         '(a gap) or resolution (no gap, low absolute recall).')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    # Just the camera size, so just the discovery -- building a BoxDataset here scattered every
    # session's parquet into dense arrays to read two integers.
    roots = load_datasets(args.data)
    if len(roots) != 1:
        raise SystemExit(f'{args.data}: the detector is trained per dataset; found {len(roots)}')
    probe_sess = roots[0].all_sessions()[0]
    wh = tuple(args.input_wh) if args.input_wh else default_input_wh(roots[0])
    print(f'input {wh[0]}x{wh[1]}  (frame {probe_sess.rig.size(probe_sess.cam_names[0])})')

    train = BoxDataset(args.data, 'train', input_wh=wh, box_source=args.boxes,
                       min_crop_dim=args.min_crop_dim,
                       max_frames_per_group=args.frames_per_group)
    print(f'train: {len(train)} views')
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, sampler=ChunkShuffle(len(train), chunk=train.chunk),
        num_workers=args.num_workers,
        collate_fn=box_collate, drop_last=True, persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init)
    val = None
    try:
        val = BoxDataset(args.data, 'val', input_wh=wh, box_source=args.boxes,
                         min_crop_dim=args.min_crop_dim,
                         max_frames_per_group=args.val_frames_per_group)
        print(f'val:   {len(val)} views')
    except ValueError as e:
        print(f'val:   none ({e})')

    model = YOLOXNano().to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'YOLOX-Nano: {n / 1e6:.2f}M params')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
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
            running.append(float(loss.detach()))
            it += 1
            if it % 50 == 0:
                print(f'{it:7d}/{args.iters}  loss {np.mean(running):7.4f}  '
                      f'obj {parts["obj"]:6.3f}  box {parts["box"]:6.3f}  '
                      f'pos {parts["n_pos"]:4d}  {(time.time() - t0) / 50:5.3f}s/it', flush=True)
                running, t0 = [], time.time()
            if it % args.eval_every == 0 or it == args.iters:
                # Both splits, EVERY checkpoint, and the score stored beside the weights. A
                # rolling `detector.pth` with no score cannot be selected on: johnson peaked at
                # val recall 0.871 and shipped 0.706, branson peaked 0.885 and shipped 0.833.
                scores = {}
                for name, ds in (('train', train), ('val', val)):
                    if ds is None:
                        continue
                    scores[name] = overall(score_dataset(
                        model, ds, device, batch_size=args.batch_size,
                        batches=args.eval_batches, num_workers=2))
                ckpt = {'iteration': it, 'model_state': model.state_dict(), 'input_wh': wh,
                        'dataset': train.ds.name, 'box_source': args.boxes,
                        'min_crop_dim': args.min_crop_dim, 'eval': scores}
                torch.save(ckpt, args.out / f'detector_it{it:06d}.pth')
                torch.save(ckpt, args.out / 'detector.pth')
                history.append({'iteration': it, **{f'{k}_{m}': v[m] for k, v in scores.items()
                                                    for m in ('r50', 'r75', 'iou', 'fp', 'mota')}})
                (args.out / 'metrics.json').write_text(json.dumps(history, indent=1))
                for name, s in scores.items():
                    print(f'   {name:5s} r@.5 {s["r50"]:.4f}  r@.75 {s["r75"]:.4f}  '
                          f'IoU {s["iou"]:.4f}  fp {s["fp"]:.3f}  MOTA {s["mota"]:.3f}',
                          flush=True)
                t0 = time.time()               # evaluation is not part of the s/it readout
    best = max(history, key=lambda h: h.get('val_r50', h['train_r50'])) if history else None
    print(f'done: {it} iterations -> {args.out}')
    if best:
        print(f'best: it {best["iteration"]}  ' +
              '  '.join(f'{k} {v:.4f}' for k, v in best.items() if k != 'iteration'))


if __name__ == '__main__':
    main()
