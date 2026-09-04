#!/usr/bin/env python
"""Train the crop-level ReID net (`tailcyclenet.detector.crop_reid`) -- the pivot from the dense
per-anchor embedding head (report 55, `dev/reports/55_identity_bridge_and_reid.md`).

Not part of the detector: this net's only input is one animal's crop, so it trains standalone,
independent of `train_detector.py` and its Muon/box/objectness machinery. Simple Adam, a small
from-scratch CNN, one PK-sampled contrastive loss -- deliberately the least architecture that can
answer whether crop-level appearance separates individuals better than the dense head did.
"""
import argparse
import time
from pathlib import Path

import toml
import torch

from tailcyclenet.checkpoints import provenance
from tailcyclenet.detector import BoxDataset, CropReidDataset, CropReidNet, PKSampler, contrastive_loss


def parse_args():
    """CLI for the crop-level ReID net's own standalone training loop."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='dataset root (one root, train split)')
    p.add_argument('--out', required=True, help='output run folder')
    p.add_argument('--iters', type=int, default=8000)
    p.add_argument('--embed-dim', type=int, default=32)
    p.add_argument('--crop-wh', default='96,96')
    p.add_argument('--input-wh', default='736,448', help='BoxDataset frame size the crop is cut '
                   'from before re-letterboxing -- NOT the crop net\'s own input size')
    p.add_argument('--box-source', default='instances', choices=['keypoints', 'instances'])
    p.add_argument('--p', type=int, default=8, help='identities per batch')
    p.add_argument('--k', type=int, default=4, help='crops per identity per batch')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--no-augment', action='store_true', help='disable crop flip/brightness jitter')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _wh(s):
    """'W,H' -> (int(W), int(H))."""
    w, h = s.split(',')
    return int(w), int(h)


def main():
    """Build the crop dataset + PK sampler, train `CropReidNet` with plain Adam + `contrastive_
    loss`, and checkpoint periodically plus at the end.
    """
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crop_wh = _wh(args.crop_wh)
    input_wh = _wh(args.input_wh)

    torch.manual_seed(args.seed)
    boxes_ds = BoxDataset(args.data, 'train', input_wh=input_wh, box_source=args.box_source)
    crop_ds = CropReidDataset(boxes_ds, crop_wh=crop_wh, augment=not args.no_augment)
    print(f'crop dataset: {len(crop_ds)} crops, {crop_ds.n_labels} identities')

    sampler = PKSampler(crop_ds, p=args.p, k=args.k, seed=None)
    batch_size = sampler.p * sampler.k
    loader = torch.utils.data.DataLoader(crop_ds, batch_size=batch_size, sampler=sampler,
                                         num_workers=args.num_workers, drop_last=False)

    net = CropReidNet(embed_dim=args.embed_dim).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    (out / 'config.toml').write_text(toml.dumps(vars(args)))
    (out / 'provenance.toml').write_text(toml.dumps(provenance()))

    def save(it):
        """Write `checkpoint_last.pth`: model weights plus the two facts a loader needs back
        (`embed_dim`, `crop_wh`) since this net carries no registry of its own.
        """
        torch.save({'model_state': net.state_dict(), 'embed_dim': args.embed_dim,
                   'crop_wh': list(crop_wh), 'iteration': it},
                  out / 'checkpoint_last.pth')

    it = 0
    t0 = time.time()
    while it < args.iters:
        for crops, labels in loader:
            crops = crops.to(args.device)
            labels = labels.to(args.device)
            vectors = net(crops)
            loss = contrastive_loss(vectors, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            if it % args.log_every == 0 or it == 1:
                dt = time.time() - t0
                print(f'{it}/{args.iters}  loss {float(loss.detach()):.4f}  {dt / max(it, 1):.3f}s/it')
            if it % args.checkpoint_every == 0:
                save(it)
            if it >= args.iters:
                break
    save(it)
    print(f'done: {it} iterations -> {out}')


if __name__ == '__main__':
    main()
