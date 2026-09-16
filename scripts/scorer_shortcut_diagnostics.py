"""Gate C: the shortcut diagnostics for a trained scorer (plan section 7.3).

A high synthetic `triplet_acc` can mean less than it looks. Two controls separate "the scorer
compares the image to the coordinates" from "the scorer is judging pose plausibility from the
coordinates alone":

- **coordinate-only** -- the pixels are replaced by a constant (zeros). If `triplet_acc` stays
  near the real arm, the pixels are not being used. `frame_noise` and `sinusoid` are the exposed
  types: they make a trajectory jittery in a way no real tracker output is, and that is visible in
  the coordinates with no image at all.
- **mismatched-video** -- the track keeps its coordinates but gets a DIFFERENT window's pixels. A
  scorer that compares image to coordinates should call the mismatched pair bad; one that does not
  will score it like the original.

These are DIAGNOSTICS, not a pass/fail bar, and neither is a trained coordinate-only baseline
(which would be the rigorous version and is out of scope). Read the gap with both directions in
mind: a small gap does not prove the corruption design is the whole problem, and a large one can
partly reflect out-of-distribution inputs rather than genuine pixel-grounding.

    pixi run python scripts/scorer_shortcut_diagnostics.py --run <scorers>/scorer-3dpop \\
        --config configs/scorer-3dpop.toml --split val --batches 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.checkpoints import load_config, load_scorer_run, _SCORER_CONFIG  # noqa: E402
from tailcyclenet.dataset import PoseDataset  # noqa: E402
from tailcyclenet.scorer.dataset import ScorerDataset, triplet_to_device  # noqa: E402
from tailcyclenet.scorer.dataset import scorer_collate  # noqa: E402
from tailcyclenet.scorer.model import flatten_sequence_triplet  # noqa: E402
from tailcyclenet.scorer.train import (corruption_config, loader_config,  # noqa: E402
                                       _frame_pointwise_arrays, _weighted_binary_metrics,
                                       scorer_output_granularity)
from tailcyclenet.scorer.triplet import GENERATORS, seed_worker  # noqa: E402


def _shapes_match(a, b):
    """Whether two per-camera view lists can stand in for one another.

    Inputs: a / b -- lists of per-camera uint8 frame tensors.
    Outputs: True when the lists are the same length and every pair has one shape.
    Side effects: none.
    """
    return len(a) == len(b) and all(x.shape == y.shape for x, y in zip(a, b))


def _swap_views(trip, other_views):
    """The triplet with every member's pixels replaced by `other_views`.

    Inputs: trip -- a triplet dict; other_views -- the replacement view list for one camera set.
    Outputs: a new triplet dict, or None when the shapes do not match.
    Side effects: none.
    """
    if other_views is None:
        return None
    target = trip['good'][0]
    if len(other_views) != len(target):
        return None
    for a, b in zip(other_views, target):
        if a.shape != b.shape:
            return None
    swapped = dict(trip)
    for name in ('good', 'bad', 'anchor'):
        views, coords, cgroup = trip[name]
        swapped[name] = ([v.detach() for v in other_views], coords, cgroup)
    return swapped


def _arms(model, loader, device, max_batches, output_granularity='sequence'):
    """Real, coordinate-only and mismatched-video `triplet_acc`, overall and per type.

    Inputs: model -- a trained scorer in eval mode; loader -- the val loader; device -- where to
            run; max_batches -- how many triplets to score.
    Outputs: {arm: {'acc': float, 'n': int, 'per_type': {name: [n, correct]}}}, plus the number of
        mismatched draws that had to be skipped for shape reasons under the key 'skipped'.
    Side effects: decodes video frames through the loader.
    """
    arms = {a: {'hits': 0.0, 'weight': 0.0, 'n': 0, 'per_type': {},
                'pointwise': [], 'pointwise_types': {}}
            for a in ('real', 'coordinate_only', 'mismatched')}
    skipped = 0
    reservoir: list = []
    seen = 0
    with torch.no_grad():
        for trip in loader:
            if trip is None:
                continue
            trip = triplet_to_device(trip, device)
            fired = (trip.get('corruption_type_mask') if output_granularity == 'frame'
                     else trip['fired'][0])
            active = trip.get('active_mask') if output_granularity == 'frame' else None
            donor = next((r for r in reservoir if _shapes_match(r, trip['good'][0])), None)
            variants = {'real': trip,
                        'coordinate_only': _swap_views(
                            trip, [torch.zeros_like(v) for v in trip['good'][0]])}
            variants['mismatched'] = _swap_views(trip, donor)
            if variants['mismatched'] is None:
                skipped += 1
            for arm, t in variants.items():
                if t is None:
                    continue
                scores, _precision, labels = model.score_triplet(t)
                if output_granularity == 'sequence':
                    scores, _precision, labels = flatten_sequence_triplet(
                        scores, _precision, labels)
                    correct_for_type = scores[:, 0] > scores[:, 1]
                    metric_mask = torch.ones_like(correct_for_type, dtype=torch.bool)
                    weights_for_type = torch.ones_like(correct_for_type, dtype=scores.dtype)
                else:
                    correct_for_type = scores[..., 0] > scores[..., 1]
                    metric_mask = (active.to(correct_for_type.device).bool()
                                   if active is not None
                                   else torch.ones_like(correct_for_type, dtype=torch.bool))
                    weights_for_type = trip.get('source_frame_weight')
                    if weights_for_type is None:
                        weights_for_type = torch.ones_like(correct_for_type, dtype=scores.dtype)
                    else:
                        weights_for_type = weights_for_type.to(
                            device=correct_for_type.device, dtype=scores.dtype)
                if not bool(metric_mask.any()):
                    continue
                metric_weight = weights_for_type[metric_mask]
                arms[arm]['hits'] += float(
                    (metric_weight * correct_for_type[metric_mask].to(metric_weight.dtype)).sum())
                arms[arm]['weight'] += float(metric_weight.sum())
                arms[arm]['n'] += 1
                type_mask = fired.to(correct_for_type.device).bool()
                if output_granularity == 'frame':
                    arms[arm]['pointwise'].append(
                        _frame_pointwise_arrays(scores, labels, t))
                    for gi, name in enumerate(GENERATORS):
                        arms[arm]['pointwise_types'].setdefault(name, []).append(
                            _frame_pointwise_arrays(scores, labels, t, gi))
                if scores.ndim == 2 and type_mask.ndim == 3:
                    type_mask = type_mask[0]
                if output_granularity == 'frame' and active is not None:
                    type_mask = type_mask & active[..., None].to(type_mask.device).bool()
                for i, name in enumerate(GENERATORS):
                    mask = type_mask[..., i]
                    if not bool(mask.any()):
                        continue
                    slot = arms[arm]['per_type'].setdefault(name, [0.0, 0.0])
                    slot[0] += float(weights_for_type[mask].sum())
                    slot[1] += float((weights_for_type[mask]
                                      * correct_for_type[mask].to(weights_for_type.dtype)).sum())
            reservoir.append([v.detach() for v in trip['good'][0]])
            if len(reservoir) > 64:
                reservoir.pop(0)
            seen += 1
            if seen >= max_batches:
                break
    out = {}
    for arm, d in arms.items():
        pointwise = [chunk for chunk in d['pointwise'] if len(chunk[0])]
        if pointwise:
            point_scores = np.concatenate([chunk[0] for chunk in pointwise])
            point_targets = np.concatenate([chunk[1] for chunk in pointwise])
            point_weights = np.concatenate([chunk[2] for chunk in pointwise])
            point_auc, point_ap = _weighted_binary_metrics(
                point_scores, point_targets, point_weights)
        else:
            point_auc = point_ap = float('nan')
        type_pointwise = {}
        for name, chunks in d['pointwise_types'].items():
            typed = [chunk for chunk in chunks if len(chunk[0])]
            if not typed:
                continue
            typed_scores = np.concatenate([chunk[0] for chunk in typed])
            typed_targets = np.concatenate([chunk[1] for chunk in typed])
            typed_weights = np.concatenate([chunk[2] for chunk in typed])
            type_pointwise[name] = _weighted_binary_metrics(
                typed_scores, typed_targets, typed_weights)
        out[arm] = {
            'n': d['n'],
            'acc': d['hits'] / d['weight'] if d['weight'] else float('nan'),
            'per_type': {k: v[1] / max(v[0], 1) for k, v in d['per_type'].items()},
            'pointwise_auroc': point_auc,
            'pointwise_ap': point_ap,
            'pointwise_per_type': type_pointwise,
        }
    out['skipped'] = skipped
    return out


def main(argv=None) -> int:
    """Entry point for the Gate C shortcut diagnostics.

    Inputs: argv -- argument list, defaulting to sys.argv.
    Outputs: process exit code.
    Side effects: prints the three-arm table; decodes video frames.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a scorer run folder')
    ap.add_argument('--config', required=True, help='the config the run was trained with')
    ap.add_argument('--split', default='val')
    ap.add_argument('--batches', type=int, default=40)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)

    model, _rcfg, registry, ckpt = load_scorer_run(Path(args.run), device=args.device)
    model.eval()
    output_granularity = scorer_output_granularity(_rcfg)
    config = load_config(args.config, base=_SCORER_CONFIG)
    config['scorer'] = {**config.get('scorer', {}),
                        'output_granularity': output_granularity}
    config['scorer']['corruption'] = {
        **config['scorer'].get('corruption', {}),
        'output_granularity': output_granularity,
    }
    ds = ScorerDataset(
        PoseDataset(config['data']['path'], args.split,
                    loader_config(config['data'], config['model']), registry=registry),
        corruption_config(config))
    nw = int(config['data'].get('num_workers', 8))
    val_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=scorer_collate,
                            num_workers=nw, prefetch_factor=2 if nw else None,
                            persistent_workers=bool(nw), pin_memory=True, worker_init_fn=seed_worker)

    print(f'run {args.run}  checkpoint {ckpt.name}  split {args.split}  '
          f'val windows {len(ds)}  K={registry.n_keypoints}')
    res = _arms(model, val_loader, args.device, args.batches, output_granularity)
    real = res['real']['acc']
    print(f'\n{"arm":>17}  {"triplet_acc":>11}  {"n":>4}   gap vs real')
    for arm in ('real', 'coordinate_only', 'mismatched'):
        a = res[arm]
        gap = '' if arm == 'real' else f'{a["acc"] - real:+.3f}'
        print(f'{arm:>17}  {a["acc"]:>11.4f}  {a["n"]:>4}   {gap}')
    print(f'\nmismatched draws skipped for shape mismatch: {res["skipped"]}')
    if output_granularity == 'frame':
        print('\npointwise signed metrics (real / coordinate-only / mismatched):')
        for metric in ('pointwise_auroc', 'pointwise_ap'):
            values = [res[arm][metric] for arm in ('real', 'coordinate_only', 'mismatched')]
            cells = '  '.join('   n/a  ' if not np.isfinite(v) else f'{v:7.4f}'
                              for v in values)
            print(f'  {metric:>20}  {cells}')
    print('\nper corruption type (real / coordinate-only / mismatched):')
    for name in GENERATORS:
        row = [res[a]['per_type'].get(name) for a in
               ('real', 'coordinate_only', 'mismatched')]
        cells = '  '.join('   n/a  ' if v is None else f'{v:7.4f}' for v in row)
        print(f'  {name:>15}  {cells}')
    print('\nA coordinate-only accuracy near the real one means the pixels are not being used; a')
    print('mismatched accuracy near the real one means the scorer does not compare image to track.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
