#!/usr/bin/env python
"""Finetune a posetail tracker into a pose estimator.

    pixi run python scripts/train.py --config configs/w9.toml
    pixi run python scripts/train.py --config configs/w9.toml --iters 200   # smoke test

The config is the source of truth for everything that defines a run; the CLI carries only
overrides and one-offs. The run folder is the output, and it holds the config, the keypoint
registry and the checkpoints -- so eval and inference take only `--run <folder>`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posetail.posetail.losses import TotalLoss

from tailcyclenet.checkpoints import (check_image_size, resolve_checkpoint, save_checkpoint,
                                      save_run_meta, warm_start)
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate, worker_init
from tailcyclenet.format import Registry
from tailcyclenet.model import build_model


def build_optimizer(model, fresh: set[str], cfg: dict):
    """Three parameter groups.

    Freshly initialised params -- the keypoint identity table, the no-query tokens, the rebuilt
    fusion gate -- start from noise and need a much higher rate than weights that arrive from
    ~1M pretraining steps. The video encoder, when unfrozen, needs a much lower one.
    """
    lr = float(cfg['learning_rate'])
    kpt_lr = float(cfg.get('kpt_lr', lr))
    enc_scale = float(cfg.get('encoder_lr_scale', 1.0))
    groups = {'fresh': [], 'encoder': [], 'rest': []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in fresh or name.startswith('query_encoder.kpt_'):
            groups['fresh'].append(p)
        elif name.startswith('scene_encoder.encoder.'):
            groups['encoder'].append(p)
        else:
            groups['rest'].append(p)

    from schedulefree import AdamWScheduleFree
    param_groups = [
        {'params': groups['fresh'], 'lr': kpt_lr},
        {'params': groups['encoder'], 'lr': lr * enc_scale},
        {'params': groups['rest'], 'lr': lr},
    ]
    opt = AdamWScheduleFree(
        [g for g in param_groups if g['params']], lr=lr,
        weight_decay=float(cfg.get('weight_decay', 0.0)),
        warmup_steps=int(cfg.get('warmup_steps', 0)),
        betas=(float(cfg.get('beta1', 0.9)), float(cfg.get('beta2', 0.95))))
    print(f'optimizer: {len(groups["fresh"])} fresh @ {kpt_lr:g}, '
          f'{len(groups["encoder"])} encoder @ {lr * enc_scale:g}, '
          f'{len(groups["rest"])} rest @ {lr:g}')
    return opt


def init_wandb(config: dict, run: Path, disabled: bool = False):
    """wandb from the existing `[wandb]` block, or None. No wrapper, no logger abstraction.

    THE RUN FOLDER STAYS `--out`. Both reference implementations put the run folder inside
    `wandb.run.dir`, which is how eval ends up needing a wandb path to find a checkpoint. Here
    `--run <folder>` is the whole model specification (`checkpoints.load_run`), so wandb gets
    `dir` for its own bookkeeping and the run folder is left alone.
    """
    cfg = config.get('wandb')
    if disabled or not cfg:
        return None
    try:
        import wandb
    except ImportError:
        print('wandb: not installed, skipping')
        return None
    Path(cfg.get('path', '.')).mkdir(parents=True, exist_ok=True)
    # <timestamp>-<run folder name>. The folder name alone is not unique -- runs/a/w9 and
    # runs/b/w9 both read `w9` in the UI, and wandb only guarantees uniqueness on the run ID --
    # and re-running into the same --out is the normal way to resume here. `%Y%m%d_%H%M%S` is
    # wandb's own directory convention (`run-20260727_113457-ukt14i7c`), so it sorts
    # lexicographically and carries no spaces or colons.
    name = f'{time.strftime("%Y%m%d_%H%M%S")}-{run.name}'
    wandb.init(project=cfg.get('project_name', 'tailcyclenet'), dir=cfg.get('path'),
               mode=cfg.get('mode', 'offline'), name=name, config=config)
    print(f'wandb: {cfg.get("mode", "offline")} | {wandb.run.name}')
    return wandb


def log(wb, values: dict, step: int) -> None:
    if wb is not None:
        wb.log(values, step=step)


def _brief(m: dict) -> str:
    keys = ('mpjpe', 'mte', 'delta_x_avg', 'survival_rate', 'n_nonfinite')
    return ' '.join(f'{k}={m[k]:.4g}' for k in keys if k in m)


def base_registry(run: Path, checkpoint_path):
    """An existing keypoint registry to append to, or None.

    Looked for in the run folder first (resuming into the same `--out`), then beside the
    warm-start checkpoint -- `Path(<run>/checkpoints).parent` is the run folder, so one lookup
    covers both that layout and a bare checkpoint directory.

    Without this every run renumbers the keypoints from scratch, and warm-starting from a run
    whose datasets were discovered in a different order quietly points each embedding row at a
    different body part. `Registry.build` raises rather than renumbering, so a genuine conflict
    is loud.
    """
    candidates = [run / 'keypoint_registry.toml']
    if checkpoint_path:
        candidates.append(Path(checkpoint_path).parent / 'keypoint_registry.toml')
    for p in candidates:
        if p.exists():
            print(f'keypoint registry: appending to {p}')
            return Registry.load(p)
    return None


def timed(loader, acc):
    """Yield from `loader`, accumulating seconds spent BLOCKED on the next batch into `acc[0]`.

    This is the starvation canary, and it is the only honest way to answer "is the loader
    bottlenecking the GPU" -- `sec_per_it` folds data wait and compute together, so a loader that
    is fine and a loader that is starving look identical in it.

    What it measures is genuine: the step ahead of it ends in `float(loss.detach())`, which syncs
    the GPU, so any time `next()` blocks is time the GPU spent idle waiting for pixels. Wrapping
    the iterator rather than timing inside the loop body keeps it correct across the `continue`
    paths that skip a non-finite step.
    """
    it = iter(loader)
    while True:
        t = time.time()
        try:
            batch = next(it)
        except StopIteration:
            return
        acc[0] += time.time() - t
        yield batch


def to_device(batch, device):
    views = [v.to(device, non_blocking=True) for v in batch.views]
    cgroup = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
              for c in batch.cgroup]
    return views, cgroup


def run_batch(model, loss_fn, batch, device):
    """One forward + loss. A non-finite sub-loss comes back as NaN, never as an exception.

    posetail's `TotalLoss.forward` RAISES on a poisoned sub-loss (`losses.py:873`). That is right
    for its own purpose -- a NaN term silently dropped from the total still returns NaN gradients --
    but it turns one intermittent bad step into a dead run, and this loop already has the handler
    for exactly this case (`torch.isfinite(loss)` in the training loop). So the raise is converted
    into the signal that handler reads: a detached NaN, which skips the step, runs no backward, lets
    nothing NaN reach the optimizer, and shows up in `skipped N` and `train/skipped_frac`.

    NOT a root-cause fix, and the difference matters. The NaN is real and is not understood:
    measured on 3dpop's `query = "none"` arm, the whole forward goes NaN while the parameters,
    `scene_center` and every target are finite, and it is intermittent (step 16 and step 31 on two
    runs of one config). `scratch/sweep/probe.py` is the instrument for chasing it.

    WATCH `skipped`. A few isolated steps is the case this handles. A rising fraction means the
    model is sitting in a NaN state and the run is dead while still printing -- kill it.
    """
    views, cgroup = to_device(batch, device)
    mode = batch.sample_info['mode']
    out = model(views, batch.kpt_ids.to(device), cgroup, mode=mode,
                kpt_prior=batch.kpt_prior.to(device), prompt_time=batch.prompt_t.to(device))
    try:
        loss = loss_fn(
            model, out, coords_true=batch.coords.to(device),
            vis_true=None if batch.vis is None else batch.vis.to(device),
            vis_true_cams=None if batch.vis_2d is None else batch.vis_2d.to(device),
            cgroup=cgroup, p2d=None if batch.p2d is None else batch.p2d.to(device), device=device)
    except ValueError as e:
        if 'non-finite' not in str(e):
            raise
        print(f'  non-finite loss -> skipping step: {e}', flush=True)
        loss = torch.tensor(float('nan'), device=device)
    return loss, out


@torch.no_grad()
def evaluate(model, batches, optimizer, device):
    """Val metrics on a FIXED set of windows. BOTH regimes, from ONE pass. Returns (prior_free,
    self_prompted).

    The prior-free number is the honest one: the loader's `kpt_prior` is ground truth (evaluation
    rule 7), so it is gated off -- letting it through inflates every number, and in the project this
    descends from it inflated every anchored number ever published.

    But prior-free is a structurally DIFFERENT forward from training on a `query = "prior"` arm:
    with no prior every query sits at the derived scene centre, `uniform` flips True in the query
    encoder, and the patch term collapses from K independent patches to one broadcast patch. So
    checkpoint selection on that number alone judges an arm by a path it is not trained on. Hence
    the second, deployable regime: predict prior-free, then re-query at the model's OWN frame-0
    prediction. Label-free, so no gate reopens, and it is the same construction
    `run_windowed(anchor='self')` uses -- literally, via `infer.self_prompt`.

    ONE pass over the batches, not two. This used to be two calls, and the second recomputed a
    bit-identical prior-free forward before self-prompting from it -- 3 full forwards per window
    where 2 will do. That cost real time: johnson-mouse's eval ran ~200 s against a 2.9 s/it step,
    +1.0 s/it amortized at `val_freq = 200`, and for allen more than half the projected wall time of
    a 60k run was eval rather than training. `share_scene` then reuses the frozen encoder across the
    two remaining forwards, which see identical pixels.

    `optimizer.eval()` swaps the schedule-free averaged iterate `x` into the parameters IN PLACE,
    which is the weight that gets deployed; the `finally` puts `y` back even if a batch raises.
    """
    from posetail.posetail.eval_metrics import get_eval_metrics, get_vis_true

    from tailcyclenet.infer import self_prompt
    from tailcyclenet.model import share_scene

    model.eval()
    if hasattr(optimizer, 'eval'):
        optimizer.eval()
    free, prompted = [], []
    try:
        for batch in batches:
            views, cgroup = to_device(batch, device)
            kpt_ids, coords = batch.kpt_ids.to(device), batch.coords.to(device)
            mode = batch.sample_info['mode']
            vis_true = (batch.vis.to(device) if batch.vis is not None else get_vis_true(coords))

            def score(out):
                m = get_eval_metrics(vis_pred=out['vis_pred'], vis_true=vis_true,
                                     coords_pred=out['coords_pred'], coords_true=coords, prefix='')
                return {k: float(v) for k, v in m.items() if np.ndim(v) == 0}

            with share_scene(model):
                out = model(views, kpt_ids, cgroup, mode=mode, kpt_prior=None, prompt_time=None)
                free.append(score(out))
                prompted.append(score(self_prompt(model, views, kpt_ids, cgroup, mode, out)))
    finally:
        model.train()
        if hasattr(optimizer, 'train'):
            optimizer.train()
    return _reduce(free), _reduce(prompted)


def _reduce(mets):
    """nanmean over windows, with the non-finite count alongside.

    A val curve quietly averaging over half its batches is the same trap as a train curve that
    does, so the skip count is reported rather than absorbed.
    """
    if not mets:
        return {}
    skipped = sum(not all(np.isfinite(v) for v in m.values()) for m in mets)
    out = {k: float(np.nanmean([m[k] for m in mets])) for k in mets[0]}
    out['n_batches'] = len(mets)
    out['n_nonfinite'] = skipped
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--iters', type=int, default=None, help='override n_iterations')
    ap.add_argument('--out', type=Path, default=None, help='override the run folder')
    ap.add_argument('--data', type=Path, default=None, help='override [data].path')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--no-warm-start', action='store_true')
    ap.add_argument('--no-wandb', action='store_true', help='ignore the [wandb] config block')
    args = ap.parse_args()

    with open(args.config, 'rb') as f:
        config = tomllib.load(f)
    check_image_size(config)
    data_cfg, train_cfg = config['data'], config['training']
    if args.data:
        data_cfg['path'] = str(args.data)
    if args.iters:
        train_cfg['n_iterations'] = args.iters
    run = Path(args.out or train_cfg['out'])
    n_iter = int(train_cfg['n_iterations'])
    device = args.device if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(int(train_cfg.get('seed', 0)))
    np.random.seed(int(train_cfg.get('seed', 0)))

    # -- data ------------------------------------------------------------------------------
    lc = LoaderConfig(**{k: v for k, v in data_cfg.items()
                         if k in LoaderConfig.__dataclass_fields__})
    train_ds = PoseDataset(data_cfg['path'], 'train', lc,
                           registry_base=base_registry(run, train_cfg.get('checkpoint_path')))
    registry = train_ds.registry
    print(f'train: {len(train_ds)} windows across {len(train_ds.datasets)} dataset(s), '
          f'{registry.n_keypoints} keypoints')
    val_ds = None
    try:
        # Val gets its OWN camera count, like the reference's separate [dataset.val] block. It
        # matters more than it looks: johnson's val sessions carry 16 cameras, so a 20-window
        # eval at full width cost ~200 s every 200 iterations -- +1.0 s/it amortized onto a
        # 2.9 s/it step. Everything else about the val loader is deliberately shared.
        val_lc = replace(lc, cams_to_sample=lc.val_cams_to_sample)
        val_ds = PoseDataset(data_cfg['path'], 'val', val_lc, registry=registry)
        print(f'val:   {len(val_ds)} windows, {lc.val_cams_to_sample} camera(s) each')
    except (ValueError, KeyError) as e:
        print(f'val:   none ({e})')

    nw = args.num_workers if args.num_workers is not None else int(data_cfg.get('num_workers', 8))
    # ONE epoch for the whole run, not one per pass over the index. rat-city's train index is 12
    # windows, so `shuffle=True` exhausted and reset the iterator every 12 steps and drained the
    # prefetch queue each time -- 26% of its wall clock, and visible as a `wait` cycle with period
    # lcm(print_freq, 12). Sampling with replacement costs nothing here: `_starts` returns a single
    # -1 per animal on train and `_item` re-picks the window start inside `__getitem__`, so which
    # index came up carries no information anyway.
    #
    # prefetch_factor 4, not the default 2: an item is a burst of T decodes (24 of 4696x2048 on
    # rat-city, ~0.9 CPU-seconds) so arrivals are lumpy, and a deeper queue is worth another 13% on
    # that arm. Affordable only because views are uint8 -- 12 workers x 4 is ~230 MB there, ~1.8 GB
    # on johnson's widest windows.
    #
    # batch_size is structurally 1: posetail's collate keeps only item 0's camera group, and the
    # model takes one camera group per batch. This is also why there is no DDP.
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, num_workers=nw, collate_fn=pose_collate,
        sampler=torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=n_iter),
        persistent_workers=nw > 0, pin_memory=True, drop_last=True,
        prefetch_factor=4 if nw > 0 else None, worker_init_fn=worker_init)

    # THE SAME WINDOWS EVERY TIME. `PoseDataset` seeds val items by index, so materialising the
    # first `val_batches` of them once makes every evaluation comparable to the last -- which is
    # the only thing that lets a val curve mean anything across iterations.
    val_freq = int(train_cfg.get('val_freq', 0))
    val_batches = []
    if val_ds is not None and val_freq:
        val_batches = [pose_collate([val_ds[i]])
                       for i in range(min(int(train_cfg.get('val_batches', 20)), len(val_ds)))]
        print(f'val:   {len(val_batches)} fixed window(s) every {val_freq} iterations')

    # -- model -----------------------------------------------------------------------------
    model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    fresh: set[str] = set()
    if not args.no_warm_start and train_cfg.get('checkpoint_path'):
        fresh = warm_start(model, resolve_checkpoint(Path(train_cfg['checkpoint_path'])))
    if train_cfg.get('freeze_encoder', True):
        n = 0
        for name, p in model.named_parameters():
            if name.startswith('scene_encoder.encoder.'):
                p.requires_grad_(False)
                n += 1
        print(f'encoder frozen: {n} tensors')
    model = model.to(device)

    opt = build_optimizer(model, fresh, config['training']['optimizer'])
    loss_fn = TotalLoss(**config['training']['losses'])
    save_run_meta(run, config, registry)
    print(f'run folder: {run.resolve()}')
    wb = init_wandb(config, run, disabled=args.no_wandb)
    log_path = run / 'log.jsonl'                  # the numbers survive without wandb

    def record(rec):
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    # -- loop ------------------------------------------------------------------------------
    max_grad = float(train_cfg.get('max_grad_norm', 0)) or None
    print_freq = int(train_cfg.get('print_freq', 20))
    ckpt_freq = int(train_cfg.get('checkpoint_freq', 1000))
    # `best` is written at checkpoint boundaries using THAT iteration's val, so a val has to land
    # on one. Loud rather than silent: misaligned, no best checkpoint would ever be written and
    # nothing else would say so.
    if val_batches and val_freq and ckpt_freq % val_freq:
        print(f'WARNING: checkpoint_freq {ckpt_freq} is not a multiple of val_freq {val_freq}, so '
              'no evaluation lands on a checkpoint boundary and checkpoint_best.pth will never be '
              'written. Pick freqs that divide.')
    model.train()
    opt.train()
    it, skipped, t0, running, clipped = 0, 0, time.time(), [], []
    # best_mpjpe/best_iter: the best val at ANY val step -- a number, used for plateau detection.
    # saved_mpjpe: the metric of the file currently on disk as `checkpoint_best.pth`. The two are
    # different on purpose; only the second one costs a 3.15 GB write. `latest` is the most recent
    # val and the iteration it came from, so the checkpoint block can tell a fresh number from a
    # stale one.
    best_mpjpe, best_iter, saved_mpjpe = float('inf'), 0, float('inf')
    latest = (float('inf'), -1)
    waited, evalled, ckpted = [0.0], [0.0], [0.0]
    while it < n_iter:
        for batch in timed(loader, waited):
            if it >= n_iter:
                break
            loss, _ = run_batch(model, loss_fn, batch, device)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad or 1e9)
            if not torch.isfinite(gn):
                # A non-finite gradient is counted, not hidden: a run that silently skips half
                # its steps looks like a run that trained.
                skipped += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            opt.step()
            running.append(float(loss.detach()))
            clipped.append(float(max_grad is not None and float(gn) > max_grad))
            it += 1

            if it % print_freq == 0:
                wall = time.time() - t0
                # `s/it` is the TRAINING step, with eval and checkpointing taken out. Left in, the
                # print after a val step read 4.9 s/it on allen and 12.1 on johnson and meant
                # nothing -- and what it hid was not noise: eval was the larger half of allen's
                # projected wall time, and a checkpoint is 3.15 GB written on every val improvement.
                elapsed = max(wall - evalled[0] - ckpted[0], 1e-9)
                dt = elapsed / print_freq
                # Fraction of that step time the GPU spent BLOCKED on pixels. Near zero means the
                # loader is keeping up and no amount of loader work will speed this run up;
                # anything large is the signal to look at `dataset.py` rather than the model.
                wait_frac = waited[0] / elapsed
                eval_frac = evalled[0] / wall if wall > 0 else 0.0
                print(f'{it:7d}/{n_iter}  loss {np.mean(running):8.4f}  '
                      f'{dt:5.2f}s/it  wait {wait_frac:4.0%}  eval {eval_frac:4.0%}  '
                      f'skipped {skipped}  '
                      f'[{batch.sample_info["dataset"]}/{batch.sample_info["mode"]}'
                      f'{"/1cam" if batch.sample_info["single_view"] else ""}]', flush=True)
                # `clipped` is the running fraction of steps hitting max_grad_norm. At
                # batch_size 1 it has measured ~50%, which makes it the most informative
                # training-health number here -- and it was being computed and thrown away.
                log(wb, {'train/loss': float(np.mean(running)), 'train/grad_norm': float(gn),
                         'train/clipped': float(np.mean(clipped[-200:])) if clipped else 0.0,
                         'train/sec_per_it': dt, 'train/loader_wait_frac': wait_frac,
                         'train/eval_frac': eval_frac,
                         'train/ckpt_frac': ckpted[0] / wall if wall > 0 else 0.0,
                         'train/skipped_frac': skipped / max(it, 1)}, it)
                running, t0 = [], time.time()
                waited[0] = evalled[0] = ckpted[0] = 0.0

            if val_batches and it % val_freq == 0:
                t_val = time.time()
                m, ms = evaluate(model, val_batches, opt, device)
                evalled[0] += time.time() - t_val
                print(f'  EVAL  ({time.time() - t_val:.0f}s)  prior-free {_brief(m)}\n'
                      f'        self-prompted {_brief(ms)}', flush=True)
                log(wb, {f'val/{k}': v for k, v in m.items()}, it)
                log(wb, {f'val_self/{k}': v for k, v in ms.items()}, it)
                # Tracked on EVERY val because it is just a number; the checkpoint it would justify
                # is written at `checkpoint_freq` instead (below). Writing on every improvement
                # meant a 3.15 GB file every `val_freq` iterations early in a run, which showed up
                # as multi-second stalls in `s/it` that had nothing to do with training.
                if np.isfinite(m.get('mpjpe', np.nan)):
                    latest = (m['mpjpe'], it)
                    if m['mpjpe'] < best_mpjpe:
                        best_mpjpe, best_iter = m['mpjpe'], it
                # Plateaus in this project break late -- one arm stalled 2600 iterations and then
                # improved -- so the stopping rule reads a number instead of an eyeball.
                log(wb, {'val/no_new_best_span': it - best_iter}, it)
                if wb is not None:
                    wb.run.summary['best_mpjpe'] = best_mpjpe
                    wb.run.summary['best_iter'] = best_iter
                record({'iter': it, 'val': m, 'val_self': ms,
                        'best_mpjpe': best_mpjpe, 'best_iter': best_iter})

            if it % ckpt_freq == 0 or it == n_iter:
                t_ck = time.time()
                p = save_checkpoint(run, it, model, opt, config)
                # `best` is decided HERE, not at every val: these are the only iterations whose
                # weights are written at all, so they are the only ones that can honestly be
                # labelled best. The comparison is against the metric of the file already on disk,
                # and it uses THIS iteration's val -- an earlier, better val describes weights that
                # no longer exist.
                if latest[1] == it and latest[0] < saved_mpjpe:
                    saved_mpjpe = latest[0]
                    save_checkpoint(run, it, model, opt, config, name='best')
                    print(f'  new best: mpjpe {saved_mpjpe:.4g} -> checkpoint_best.pth')
                ckpted[0] += time.time() - t_ck
                print(f'saved {p} ({time.time() - t_ck:.0f}s)')
    print(f'done: {it} iterations, {skipped} skipped')
    record({'iter': it, 'done': True, 'skipped': skipped,
            'best_mpjpe': best_mpjpe, 'best_iter': best_iter})
    if wb is not None:
        wb.finish()


if __name__ == '__main__':
    main()
