#!/usr/bin/env python
"""Pretrain ONE detector backbone across SEVERAL dataset roots. T4.1b, dev/plans/detector_accuracy.md.

    pixi run python scripts/pretrain_detector_backbone.py --config configs/pretrain_backbone.toml

T4.1's own COCO backbone is out-of-domain (web photos, 80 classes, upright objects); this repo
already holds ~100k labelled frames across six roots, and the only thing stopping a SHARED
backbone across them is `BoxDataset`'s single-root refusal -- an argument about INPUT SIZE (one
letterbox cannot serve two roots whose frames differ in size and aspect ratio), not about the
backbone, which is a plain conv stack that has no notion of "which root". This script is the
missing piece: build one `BoxDataset` PER ROOT at THAT root's own `input_wh`, forward each
through ONE SHARED model, and step. A batch is always ONE root (never mixed), so no image in a
step needs another root's letterbox.

**Why this is safe:** GroupNorm has no running statistics to collect on one root and misapply to
another (the whole reason `conv_norm_act` switched off BatchNorm -- see `yolox.py`). A step drawn
from root A updates the SAME weights a step from root B will use next, with no cross-root state
riding along in a buffer.

**What is pretrained, and what is not:** the BACKBONE ONLY (`model.backbone`), the same scope
`load_coco_backbone` uses and for the same reason -- the neck is NOT shared across roots (each
root's own `input_wh` gives it a different feature-map size, but that is irrelevant here: the
issue is that this repo unifies all three FPN levels to one width, so there is nothing tier- or
root-specific about the neck to begin with; it is simply never saved). `n_keypoints=0` always --
different roots disagree on `K` and keypoint identity, and the backbone this script exists to
produce has no notion of either.

**Round-robin, not concatenation.** Each step draws the NEXT root in a fixed cycle (not randomly
by root, not one root run to exhaustion before the next starts) -- `_endless` wraps each root's
own `DataLoader` iterator so it restarts on exhaustion, and the cycle order is fixed by config
order so a run is reproducible from its own file. A root with fewer labelled frames simply
restarts its own loader more often; no root's epoch length constrains another's.

Output: `backbone.pth` (and periodic `backbone_it<n>.pth` snapshots) at `[training].out`, holding
`backbone_state` (`model.backbone.state_dict()`), `yolox_version`, `bottleneck_expansion`,
`in_channels` and `roots` (the paths trained on, for provenance) -- read by
`tailcyclenet.detector.pretrained.load_pretrained_backbone`, and named `[model].pretrained = "<
this file's path>"` in a normal `configs/detector.toml`-style fine-tune config.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import toml

from tailcyclenet.checkpoints import provenance
from tailcyclenet.dataset import worker_init
from tailcyclenet.detector import BoxDataset, ChunkShuffle, YOLOXNano, box_collate, detector_loss
from tailcyclenet.detector.yolox import YOLOX_TIERS

REPO = Path(__file__).resolve().parent.parent

ROOT_KEYS = frozenset({'path', 'boxes', 'frames_per_group', 'input_wh'})
MODEL_KEYS = frozenset({'yolox', 'bottleneck_expansion'})
TRAINING_KEYS = frozenset({'out', 'iters', 'batch_size', 'lr', 'num_workers', 'seed', 'device',
                           'snapshot_every'})


def _raise_unknown(block, cfg, known):
    unknown = set(cfg) - known
    if unknown:
        raise SystemExit(f'[{block}]: unknown key(s) {sorted(unknown)}. Known: {sorted(known)}')


def load_pretrain_config(path):
    """Load + validate. Returns `(roots, model, training)`, all resolved (no more `.get`)."""
    raw = toml.loads(Path(path).read_text())
    roots_raw = raw.get('roots')
    if not roots_raw:
        raise SystemExit(f'{path}: needs at least one [[roots]] table -- this script pretrains '
                         'ACROSS roots, so one is a config error, not a degenerate case of one.')
    roots = []
    for i, r in enumerate(roots_raw):
        _raise_unknown(f'roots[{i}]', r, ROOT_KEYS)
        if not r.get('path'):
            raise SystemExit(f'roots[{i}]: path is required.')
        roots.append({
            'path': r['path'],
            'boxes': str(r.get('boxes', 'keypoints')),
            'frames_per_group': int(r.get('frames_per_group', 40)),
            'input_wh': tuple(r['input_wh']) if r.get('input_wh') else None,
        })

    model = dict(raw.get('model', {}))
    _raise_unknown('model', model, MODEL_KEYS)
    model = {'yolox': str(model.get('yolox', 'tiny')),
             'bottleneck_expansion': float(model.get('bottleneck_expansion', 0.5))}
    if model['yolox'] == 'trimmed' and model['bottleneck_expansion'] != 0.5:
        raise SystemExit("[model].bottleneck_expansion != 0.5 needs a canonical yolox tier -- "
                         "'trimmed' does not take it.")
    if model['yolox'] != 'trimmed' and model['yolox'] not in YOLOX_TIERS:
        raise SystemExit(f"[model].yolox must be 'trimmed' or one of {sorted(YOLOX_TIERS)}, "
                         f"got {model['yolox']!r}.")

    train = dict(raw.get('training', {}))
    _raise_unknown('training', train, TRAINING_KEYS)
    if not train.get('out'):
        raise SystemExit('[training].out is required.')
    train = {'out': str(train['out']), 'iters': int(train.get('iters', 20000)),
             'batch_size': int(train.get('batch_size', 16)),
             'lr': float(train.get('lr', 1e-3)),
             'num_workers': int(train.get('num_workers', 8)),
             'seed': int(train.get('seed', 0)),
             'device': str(train.get('device', 'cuda:0')),
             'snapshot_every': int(train.get('snapshot_every', 2000))}
    return roots, model, train


def _endless(loader):
    """Cycle a DataLoader forever -- one root's epoch length never bounds another's."""
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    roots_cfg, model_cfg, train_cfg = load_pretrain_config(args.config)
    if args.out is not None:
        train_cfg['out'] = str(args.out)
    if args.iters is not None:
        train_cfg['iters'] = int(args.iters)
    if args.device is not None:
        train_cfg['device'] = str(args.device)

    torch.manual_seed(train_cfg['seed'])
    np.random.seed(train_cfg['seed'])
    device = train_cfg['device'] if torch.cuda.is_available() else 'cpu'
    run = Path(train_cfg['out'])
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.toml').write_text(toml.dumps({'roots': roots_cfg, 'model': model_cfg,
                                                 'training': train_cfg}))
    prov = provenance()
    if prov:
        (run / 'provenance.toml').write_text(toml.dumps(prov))

    # PER-ROOT input size: reuses train_detector.py's own sizing rule rather than a second copy of
    # it, imported by file path the same way tests/test_detector.py already reuses this script.
    import importlib.util
    spec = importlib.util.spec_from_file_location('tcn_train_detector_for_pretrain',
                                                  REPO / 'scripts' / 'train_detector.py')
    train_detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_detector)

    from tailcyclenet.format import load_datasets

    loaders, names = [], []
    for r in roots_cfg:
        ds_root = load_datasets(r['path'])
        if len(ds_root) != 1:
            raise SystemExit(f'{r["path"]}: expected one dataset root, found {len(ds_root)}')
        wh = r['input_wh'] or train_detector.input_wh_for(r['path'], ds_root[0], r['boxes'])
        train = BoxDataset(r['path'], 'train', input_wh=wh, box_source=r['boxes'],
                           max_frames_per_group=r['frames_per_group'], augment=True,
                           seed=train_cfg['seed'])
        print(f'{Path(r["path"]).name}: {len(train)} views at {wh[0]}x{wh[1]}', flush=True)
        loader = torch.utils.data.DataLoader(
            train, batch_size=train_cfg['batch_size'],
            sampler=ChunkShuffle(len(train), chunk=train.chunk, seed=train_cfg['seed']),
            num_workers=train_cfg['num_workers'], collate_fn=box_collate, drop_last=True,
            persistent_workers=train_cfg['num_workers'] > 0, worker_init_fn=worker_init)
        loaders.append(_endless(loader))
        names.append(Path(r['path']).name)

    model = YOLOXNano(n_keypoints=0, version=model_cfg['yolox'],
                      bottleneck_expansion=model_cfg['bottleneck_expansion']).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'YOLOX [{model_cfg["yolox"]}] backbone pretrain: {n / 1e6:.2f}M params '
          f'(bottleneck_expansion={model_cfg["bottleneck_expansion"]:g}) across {len(loaders)} '
          f'root(s): {", ".join(names)}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg['iters'])

    def _save(it):
        ck = {'iteration': it, 'backbone_state': model.backbone.state_dict(),
             'yolox_version': model_cfg['yolox'],
             'bottleneck_expansion': model_cfg['bottleneck_expansion'],
             'in_channels': model.in_channels,
             'roots': [r['path'] for r in roots_cfg]}
        torch.save(ck, run / f'backbone_it{it:06d}.pth')
        torch.save(ck, run / 'backbone.pth')

    it, t0, running = 0, time.time(), []
    model.train()
    # ROUND-ROBIN: one batch from EACH root per "super-step", in fixed config order, not a random
    # root per step -- a fixed order is what makes a run's own root-visitation schedule
    # reproducible from the config alone, and every root gets equal wall-clock share regardless
    # of size (a root with fewer frames just restarts its own loader more often via `_endless`).
    while it < train_cfg['iters']:
        for loader in loaders:
            if it >= train_cfg['iters']:
                break
            batch = next(loader)
            x, gt = batch[0].to(device), batch[1].to(device)
            obj, boxes, _ = model(x)
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
                print(f'{it:7d}/{train_cfg["iters"]}  loss {np.mean(running):7.4f}  '
                      f'obj {parts["obj"]:6.3f}  box {parts["box"]:6.3f}  '
                      f'pos {parts["n_pos"]:4d}  {(time.time() - t0) / 50:5.3f}s/it', flush=True)
                running, t0 = [], time.time()
            if it % train_cfg['snapshot_every'] == 0 or it == train_cfg['iters']:
                _save(it)
    print(f'done: {train_cfg["iters"]} iterations -> {run}')


if __name__ == '__main__':
    main()
