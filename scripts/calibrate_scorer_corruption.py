#!/usr/bin/env python
"""Measure what a corruption does, before trusting a long run to it.

Section 3.4's gate: a corrupted point that lands OUTSIDE the view is trivially detectable -- the
query projects out of bounds, the patch sampler grid-samples padding, and the scorer learns "out of
frame implies bad" without ever comparing pixels to coordinates. The reference's magnitudes were
tuned for 128 generic points on a different data distribution, so they are a starting point here,
not a default.

Reports, per corruption type: how often it fires, the realised displacement in PIXELS (projected
through the cameras, so 2D and 3D are on one scale), and the fraction of corrupted slots that leave
every camera's view. Needs no model -- the geometry depends only on the data -- so it is cheap on a
real root.

    pixi run python scripts/calibrate_scorer_corruption.py --config configs/scorer-3dpop.toml \
        --windows 200
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posetail.datasets.scorer_corruption import GENERATORS
from posetail.posetail.cube import is_point_visible, project_points_torch

from tailcyclenet.checkpoints import _SCORER_CONFIG, load_config
from tailcyclenet.dataset import PoseDataset
from tailcyclenet.scorer.dataset import ScorerDataset, scorer_collate
from tailcyclenet.scorer.train import corruption_config, loader_config


class _View:
    """The two fields `_pixels` / `_out_of_view` need, read off the triplet itself.

    Re-realising the selection would draw a DIFFERENT view and measure the corruption against
    cameras it was never built with -- and would decode the video a second time. The triplet
    already carries the view's own cameras, so the measurement uses those.
    """

    def __init__(self, r, cgroup):
        """Inputs: r -- 2 or 3; cgroup -- the view's cameras. Outputs: none. Side effects: none."""
        self.r = r
        self.cgroup = cgroup


def _pixels(view, coords):
    """`coords` [t,k,R] for ONE member -> [n_cams,t,k,2] pixels, NaN preserved.

    For 3D this projects through the realised cameras, which is the frame the model actually sees;
    for 2D the coordinates already ARE those pixels. One scale for both dimensionalities.

    Inputs: view -- the realised `View` whose cameras project them; coords -- [t,k,R].
    Outputs: [n_cams,t,k,2] float tensor.
    Side effects: none.
    """
    if view.r == 2:
        return coords[None]
    return project_points_torch(view.cgroup, coords)


def _out_of_view(view, coords):
    """[t,k] bool: the point is visible in NO camera, so the crop could never have held it.

    Inputs: view -- the realised `View`; coords -- [t,k,R].
    Outputs: [t,k] bool.
    Side effects: none.
    """
    t, k, _ = coords.shape
    if view.r == 2:
        size = view.cgroup[0]['size']
        c = torch.nan_to_num(coords, nan=-1e9)
        w, h = float(size[0]), float(size[1])
        inside = (c[..., 0] >= 0) & (c[..., 0] < w) & (c[..., 1] >= 0) & (c[..., 1] < h)
        return ~inside
    flat = coords.reshape(-1, 3)
    vis = torch.stack([is_point_visible(cam, flat) for cam in view.cgroup])
    return ~vis.any(0).reshape(t, k)


def measure(config_path, windows: int, seed: int) -> dict:
    """Build triplets and accumulate the per-type geometry statistics.

    Inputs: config_path -- a scorer config (only [data] and [scorer.corruption] are read);
            windows -- how many triplets to build; seed -- item-stream seed.
    Outputs: {type_name: {'n', 'disp_px' (median), 'disp_px_p90', 'off_view_frac'}}.
    Side effects: reads the dataset root and decodes video.
    """
    config = load_config(config_path, base=_SCORER_CONFIG)
    lc = loader_config(config['data'], config['model'])
    base = PoseDataset(config['data']['path'], 'train', lc, train=True)
    ds = ScorerDataset(base, corruption_config(config))

    stats = defaultdict(lambda: {'n': 0, 'disp': [], 'off': 0})
    built = 0
    for i in range(len(ds)):
        if built >= windows:
            break
        trip = ds[i]
        if trip is None:
            continue
        built += 1
        fired = trip['fired'][0]
        if trip['reuse_scene_for_anchor']:
            continue
        good = trip['good'][1][0]
        bad = trip['bad'][1][0]
        observed = torch.isfinite(good).all(-1) & torch.isfinite(bad).all(-1)

        view = _View(r=2 if trip['mode'] == '2d' else 3, cgroup=trip['good'][2])
        pg = _pixels(view, good)
        pb = _pixels(view, bad)
        disp = (pb - pg).norm(dim=-1).nanmedian(dim=0).values
        off = _out_of_view(view, bad)

        for ti, name in enumerate(GENERATORS):
            mask = fired[:, ti] & observed
            count = int(mask.sum())
            if not count:
                continue
            stats[name]['n'] += count
            stats[name]['disp'].append(disp[mask])
            stats[name]['off'] += int((off & fired[:, ti])[observed].sum())
    return stats


def _report(stats: dict) -> None:
    """Print the per-type table.

    Inputs: stats -- `measure`'s output.
    Outputs: none.
    Side effects: writes to stdout.
    """
    print(f'{"type":<15}{"n":>8}{"median px":>12}{"p90 px":>10}{"off-view":>12}')
    for name in GENERATORS:
        s = stats.get(name)
        if not s or not s['n']:
            print(f'{name:<15}{0:>8}')
            continue
        d = torch.cat(s['disp']).float()
        off = s['off'] / max(s['n'], 1)
        print(f'{name:<15}{s["n"]:>8}{float(d.median()):>12.2f}'
              f'{float(d.quantile(0.9)):>10.2f}{off:>11.1%}')


def main(argv=None) -> int:
    """CLI entry point.

    Inputs: argv -- argument list, or None for `sys.argv`.
    Outputs: a process exit code.
    Side effects: prints the measurement table.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--windows', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    _report(measure(args.config, args.windows, args.seed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
