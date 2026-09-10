"""Report the phase-3.5 cell contract: which sources train, on which corruption menu, and why.

Section 3.8b splits a training window by SOURCE and DENSITY. A dense window (at least
`min_valid_frames` distinct observed frames per keypoint) gets all four reference corruption types;
a sparse window gets only the types that can move an OBSERVED slot, with `const_offset` forced.
The plan's phase-3.5 exit is that BOTH sources contribute, that nothing is silently dropped or
starved, and that losses and gradients are finite on both cells.

3dpop and calms21 are 100% `tracked`, so their annotated cell is empty before any filtering and the
lever is structurally inert there. allen-mouse-combined is where it is exercised.

Usage:
    pixi run python scripts/report_scorer_cells.py --config configs/scorer-allen-mouse-combined.toml \\
        [--split train] [--windows 400] [--grad] [--device cpu]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.checkpoints import load_config, _SCORER_CONFIG  # noqa: E402
from tailcyclenet.dataset import PoseDataset  # noqa: E402
from tailcyclenet.scorer.dataset import ScorerDataset, triplet_to_device  # noqa: E402
from tailcyclenet.scorer.model import build_scorer  # noqa: E402
from tailcyclenet.scorer.train import (corruption_config, loader_config,  # noqa: E402
                                       scorer_kwargs)
from tailcyclenet.scorer.triplet import GENERATORS, make_triplet  # noqa: E402


def _open(config, split):
    """A `ScorerDataset` for one split, built from the split's own registry.

    Inputs: config -- a loaded scorer config; split -- 'train' or 'val'.
    Outputs: the ScorerDataset.
    Side effects: reads the dataset root; decodes video on item access.
    """
    base = PoseDataset(config['data']['path'], split,
                       loader_config(config['data'], config['model']))
    return ScorerDataset(base, corruption_config(config))


def _walk(config, split, n_windows):
    """Walk the dataset directly, so every rejection names its own stage.

    Inputs: config -- a loaded scorer config; split -- 'train' or 'val'; n_windows -- how many
            item indices to attempt.
    Outputs: (per-cell Counter of accepted triplets, Counter of rejection reasons, Counter of
        fired corruption types per cell, Counter of dense/sparse keypoints per cell).
    Side effects: decodes video frames.
    """
    ds = _open(config, split)
    base = ds.base
    corruptors = ds.corruptors
    cfg = ds.cfg
    accepted = collections.Counter()
    reasons = collections.Counter()
    fired = collections.defaultdict(collections.Counter)
    density = collections.defaultdict(collections.Counter)
    for i in range(min(n_windows, len(base))):
        rng = np.random.default_rng(None if base.train else (base.seed, i))
        shape_rng = rng if base.train else np.random.default_rng((base.seed, 0x5AFE, i))
        cell_of = base.index[i].session
        sel = base._select(i, rng, base._shape(shape_rng))
        if sel is None:
            reasons[(cell_of.label_source, cell_of.mode, 'select returned None')] += 1
            continue
        view = base._realise(sel, rng, world_gauge=False)
        if view is None:
            reasons[(sel.sess.label_source, sel.sess.mode, 'realise returned None')] += 1
            continue
        trip = make_triplet(base, sel, rng, cfg, corruptors)
        cell = (sel.sess.label_source, sel.sess.mode)
        if trip is None:
            reasons[(*cell, 'triplet rejected')] += 1
            continue
        accepted[cell] += 1
        density[cell]['dense'] += trip['n_dense']
        density[cell]['sparse'] += trip['n_sparse']
        names = trip['fired'].sum((0, 1)).tolist()
        for j, name in enumerate(GENERATORS):
            if names[j] > 0:
                fired[cell][name] += names[j]
    return accepted, reasons, fired, density


def _grad_check(config, device):
    """One forward/backward per cell, reporting whether loss and every gradient stay finite.

    Inputs: config -- a loaded scorer config; device -- torch device string.
    Outputs: dict (source, mode) -> (loss float, finite bool, n_tensors, n_nonfinite).
    Side effects: builds a model, runs a backward pass, mutates model gradients.
    """
    ds = _open(config, 'train')
    model = build_scorer({**config['model'], 'video_encoder_pretrained': False},
                         ds.base.registry.n_keypoints, **scorer_kwargs(config)).to(device)
    loss_fn = _loss(config)
    out = {}
    wanted = {}
    for i in range(len(ds)):
        sel_cell = (ds.base.index[i].session.label_source, ds.base.index[i].session.mode)
        if sel_cell not in wanted:
            wanted[sel_cell] = i
        if len(wanted) == len({(s.label_source, s.mode) for s in
                               (x.session for x in ds.base.index)}):
            break
    for cell, i in sorted(wanted.items()):
        rng = np.random.default_rng(0)
        ds.base.train = True
        sel = ds.base._select(i, rng, ds.base._shape(rng))
        if sel is None:
            continue
        trip = make_triplet(ds.base, sel, rng, ds.cfg, ds.corruptors)
        if trip is None:
            continue
        trip = triplet_to_device(trip, device)
        model.zero_grad(set_to_none=True)
        scores, precision, labels = model.score_triplet(trip)
        loss = loss_fn(scores, precision, labels)
        loss.backward()
        n_bad = sum(int((~torch.isfinite(p.grad).all()).sum()) for p in model.parameters()
                    if p.grad is not None)
        n_t = sum(1 for p in model.parameters() if p.grad is not None)
        out[cell] = (float(loss.detach()), bool(torch.isfinite(loss)), n_t, n_bad)
    return out


def _loss(config):
    """A `TripletScorerLoss` built from the config's [scorer] block.

    Inputs: config -- a loaded scorer config.
    Outputs: the loss module.
    Side effects: none.
    """
    from posetail.posetail.losses_scorer import TripletScorerLoss
    s = config['scorer']
    return TripletScorerLoss(margin=float(s.get('triplet_margin', 0.25)),
                             precision_reg_weight=float(s.get('precision_reg_weight', 0.01)),
                             score_reg_weight=float(s.get('score_reg_weight', 0.0)))


def main(argv=None) -> int:
    """Entry point for the phase-3.5 cell report.

    Inputs: argv -- argument list, defaulting to sys.argv.
    Outputs: process exit code.
    Side effects: prints the report; decodes video; runs one backward pass per cell under --grad.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--windows', type=int, default=400)
    ap.add_argument('--grad', action='store_true')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)

    config = load_config(args.config, base=_SCORER_CONFIG)
    ds = _open(config, args.split)
    print(f'split={args.split}  windows={len(ds)}  K={ds.base.registry.n_keypoints}')
    print(f'requested mix: {ds.mix()}')
    print(f'annot_frac={config["data"].get("annot_frac")}  '
          f'min_valid_frames={config["scorer"]["min_valid_frames"]}')

    accepted, reasons, fired, density = _walk(config, args.split, args.windows)
    total = sum(accepted.values())
    print(f'\nrealised cells over {args.windows} draws ({total} accepted):')
    for cell, n in accepted.most_common():
        d = density[cell]
        print(f'  {cell[0]:>9}/{cell[1]}  {n:5d} triplets  '
              f'({100.0 * n / max(total, 1):5.1f}%)  dense kpts {d["dense"]}, sparse {d["sparse"]}')
    if reasons:
        print('\nrejections, by cell and stage:')
        for (src, mode, why), n in reasons.most_common():
            print(f'  {src:>9}/{mode}  {n:5d}  {why}')
    else:
        print('\nrejections: none')
    print('\nrealised corruption menu, per cell (points actually perturbed):')
    for cell in sorted(fired, key=lambda c: -sum(fired[c].values())):
        tot = sum(fired[cell].values())
        parts = ', '.join(f'{k}={v / tot:.3f}' for k, v in fired[cell].most_common())
        print(f'  {cell[0]:>9}/{cell[1]}  n={int(tot)}  {parts}')
        if cell[0] == 'annotated':
            types = {k for k in fired[cell]}
            if 'gradual_drift' in types or 'sinusoid' in types:
                print('    NOTE: a sparse annotated window drew a TEMPORAL type; check whether '
                      'it moved an observed slot.')

    if args.grad:
        print('\nloss and gradient finiteness, one backward per cell:')
        for cell, (loss, finite, n_t, n_bad) in sorted(_grad_check(config, args.device).items()):
            print(f'  {cell[0]:>9}/{cell[1]}  loss={loss:.5g} finite={finite}  '
                  f'tensors with grad={n_t}, non-finite grad entries={n_bad}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
