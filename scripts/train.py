#!/usr/bin/env python
"""Finetune a posetail tracker into a pose estimator.

    pixi run python scripts/train.py --config configs/3d.toml
    pixi run python scripts/train.py --config configs/3d.toml --iters 200   # smoke test

The config is the source of truth for everything that defines a run; the CLI carries only
overrides and one-offs. The run folder is the output, and it holds the config, the keypoint
registry and the checkpoints -- so eval and inference take only `--run <folder>`.
"""
# E402: the tailcyclenet imports below must follow this file's sys.path.insert.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# The default `file_descriptor` sharing strategy keeps a fd open per worker-produced tensor
# until BOTH ends are done with it. Tried as one candidate explanation for calms21's per-worker
# RSS plateau under `multiprocessing_context='spawn'` (below) and made no measurable difference
# on a real job (dev/plans/video_loader_memory.md §2.3) -- kept anyway, since it is the standard
# mitigation for this class of shared-memory-handle accounting and costs nothing to leave in.
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posetail.posetail.losses import TotalLoss

from tailcyclenet.checkpoints import (check_image_size, load_config, resolve_checkpoint, save_checkpoint,
                                      save_run_meta, warm_start)
from tailcyclenet.dataset import LoaderConfig, PoseDataset, pose_collate, worker_init
from tailcyclenet.format import Registry
from tailcyclenet.model import build_model
from tailcyclenet.unfreeze import (apply_norms_extension, apply_staged_unfreeze,
                                   replay_staged_unfreeze, trainable_encoder_params)


def build_optimizer(model, fresh: set[str], cfg: dict):
    """The run's optimizer, selected by `[training.optimizer].optimizer` (config key only).

    `"muon"` (the shipped value, AND what an ABSENT key means) runs `torch.optim.Muon` on the 2D
    transformer matrices with AdamW-schedule-free on everything else -- see `tailcyclenet.optim`.
    `"schedulefree"` restores the previous AdamW-SF recipe below, and is what an existing run
    folder needs to resume, since its checkpoint carries AdamW-SF optimizer state.

    Freshly initialised params -- the keypoint identity table, the no-query tokens, the rebuilt
    fusion gate -- start from noise and need a much higher rate than weights that arrive from
    ~1M pretraining steps. The video encoder, when unfrozen, needs a much lower one.
    """
    kind = str(cfg.get('optimizer', 'muon'))
    if kind == 'muon':
        from tailcyclenet.optim import build_muon
        return build_muon(model, fresh, cfg)
    if kind != 'schedulefree':
        raise SystemExit(f'[training.optimizer].optimizer = {kind!r} is not one of '
                         '"muon" | "schedulefree".')
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
    # <timestamp>-<run folder name>. The folder name alone is not unique -- runs/a/exp and
    # runs/b/exp both read `exp` in the UI, and wandb only guarantees uniqueness on the run ID --
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


def _gpu_peak_gb(device, reset: bool = False) -> float:
    """Peak allocator bytes since the last reset, in GB. 0.0 on CPU.

    `max_memory_allocated`, not `memory_allocated`: whether a recipe FITS is a property of the
    peak, and the peak is where an OOM happens. It is also not the same as `nvidia-smi`, which
    reports the caching allocator's reserved pool -- the reserved figure is what the driver holds,
    the allocated figure is what the run actually needs, and the gap is reusable.
    """
    if not (isinstance(device, str) and device.startswith('cuda')) or not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return peak


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


def _tune_smoothness(loss_fn, T, stride=1):
    """Hold the smoothness order below the window length and undo the stride. Mutates in place.

    `SmoothnessLoss.forward` builds its stencil mask with `valid.narrow(time_dim, 0, T - k)`
    (`posetail/losses.py:1146`), so a window shorter than `k + 1` frames RAISES on a negative
    narrow length -- and `torch.diff(n=k)` is undefined there anyway. posetail 0.3.5 clamps
    `order` to `T - 1` inside the loss itself; this per-batch rule is the same clamp applied
    eagerly, and also undoes the stride (below). Both weights are 0.5, so the `weight == 0` early
    return does not cover it.

    Now that the loader sizes T to the labelled span, T = 2 is the common case on annotated
    sessions and k = 4 would kill every one of those steps. Clamping degrades rather than
    disables: at T = 2 the penalty becomes a first difference, which is a real smoothness term.
    And on the windows this actually fires for it changes nothing measurable -- the stencil
    requires all k+1 frames valid, so a single labelled frame gives an empty mask and the loss
    returns 0 either way.

    STRIDE. `SmoothnessLoss` uses `torch.diff`, an UNDIVIDED difference with no notion of dt, so on
    a stride-s lattice its k-th difference grows like `s^k` -- 256x at k = 4, s = 4. `excess` is
    homogeneous of degree 1 in that, so the term's magnitude, and therefore its effective weight
    against every other loss, would depend on a per-item draw. Dividing by `s^k` puts it back in
    per-frame units. Folded into `weight` because `forward` divides `excess` by `scale`, and
    `scale` is TotalLoss's to set.

    What this does NOT fix, because nothing can: the HINGE is scale-invariant --
    `clamp(|d_pred| - 1.5|d_true|, 0) > 0` is unchanged by rescaling both sides -- and striding
    genuinely loosens it. The threshold tracks the true trajectory's k-th derivative, which grows
    like `s^k`, while the per-frame jitter the hinge exists to catch is white and s-independent. So
    at large s the term stops firing on jitter. That is a real change of meaning, not a bug to
    normalise away; declare it when reporting a `frame_strides` arm. (CLAUDE.md used to state the
    effective weight RISES with stride. It is the magnitude that rose; the hinge got looser.)
    """
    for name in ('smoothness_loss_3d', 'smoothness_loss_2d'):
        sl = getattr(loss_fn, name, None)      # a stub loss in the tests has neither
        if sl is None:
            continue
        # Latch the configured order and weight on first use rather than at construction: one place
        # holds this rule, so there is no second site to keep in sync with `TotalLoss(**losses)`.
        # Latching also makes both mutations idempotent -- they are re-derived from the configured
        # values every batch, never compounded onto the previous batch's.
        if not hasattr(sl, '_configured_order'):
            sl._configured_order, sl._configured_weight = sl.order, sl.weight
        sl.order = min(sl._configured_order, max(1, T - 1))
        sl.weight = sl._configured_weight / float(stride) ** sl.order


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
    runs of one config).

    WATCH `skipped`. A few isolated steps is the case this handles. A rising fraction means the
    model is sitting in a NaN state and the run is dead while still printing -- kill it.
    """
    views, cgroup = to_device(batch, device)
    mode = batch.sample_info['mode']
    _tune_smoothness(loss_fn, int(views[0].shape[1]), batch.sample_info.get('stride', 1))
    box = getattr(batch, 'box_prompt', None)
    out = model(views, batch.kpt_ids.to(device), cgroup, mode=mode,
                kpt_prior=batch.kpt_prior.to(device), prompt_time=batch.prompt_t.to(device),
                box_prompt=None if box is None else box.to(device))
    coords_true = batch.coords.to(device)
    try:
        loss = loss_fn(
            model, out, coords_true=coords_true,
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
                bx = getattr(batch, 'box_prompt', None)
                bx = None if bx is None else bx.to(device)
                out = model(views, kpt_ids, cgroup, mode=mode, kpt_prior=None, prompt_time=None,
                            box_prompt=bx)
                free.append(score(out))
                prompted.append(score(self_prompt(model, views, kpt_ids, cgroup, mode, out,
                                                  box_prompt=bx)))
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
    ap.add_argument('--no-resume', action='store_true',
                    help='start from iteration 0 even if the run folder holds a checkpoint_last')
    ap.add_argument('--no-wandb', action='store_true', help='ignore the [wandb] config block')
    ap.add_argument('--no-checkpoints', action='store_true',
                    help='write no checkpoints at all, including the final one. For probes '
                         '(memory, throughput) whose weights are worthless and whose files are '
                         '~5.6 GB each -- checkpoint_freq alone cannot express this, because the '
                         'last iteration always writes.')
    args = ap.parse_args()

    config = load_config(args.config)
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
    # AN UNKNOWN [data] KEY IS A TYPO, NOT A COMMENT. The filter is needed -- `path` and
    # `num_workers` live in this block and are not LoaderConfig fields -- but it also swallowed
    # `prompt_offset` for `prompt_offset_px`, `frame_stride` for `frame_strides`, and any renamed
    # key: the arm trained at the DEFAULT, `config.toml` in the run folder recorded the key nobody
    # read, and nothing in the log said so. That is an arm silently reporting its own control
    # (eval rule 4). `build_model` splats [model] and raises on an unknown key; this is the same
    # guard for [data], which had none.
    known = set(LoaderConfig.__dataclass_fields__) | {'path', 'num_workers'}
    unknown = set(data_cfg) - known
    if unknown:
        raise SystemExit(
            f'[data]: unknown key(s) {sorted(unknown)}. Nothing reads them, so this run would '
            f'train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(known)}')
    lc = LoaderConfig(**{k: v for k, v in data_cfg.items()
                         if k in LoaderConfig.__dataclass_fields__})
    # THE LOADER EMITS A BOX ONLY FOR A BOX MODEL, and the mode comes from [model].box_prompt so
    # the two cannot disagree (report 27). A plain run leaves this 'none' and its items keep their
    # 13-field shape, byte-identical.
    lc = replace(lc, box_prompt=config['model'].get('box_prompt', 'none'))
    # Hoisted out of the call because `warm_start` needs it too: it is the evidence that the
    # checkpoint's keypoint table is a PREFIX of this run's, and therefore that its trained rows
    # can be carried over instead of reset.
    base_reg = base_registry(run, train_cfg.get('checkpoint_path'))
    train_ds = PoseDataset(data_cfg['path'], 'train', lc, registry_base=base_reg)
    registry = train_ds.registry
    print(f'train: {len(train_ds)} windows across {len(train_ds.datasets)} dataset(s), '
          f'{registry.n_keypoints} keypoints')
    # The sampling mix, not the window count. An index entry is one (session, group, animal)
    # whatever the group's length, so window counts say nothing about what the model will
    # actually see -- allen-mouse-combined reads 1108 windows either way, while the share of
    # steps reaching the 3D head bank moves from 9.6% to 88%. Invisible in the loss curve.
    print('train: mix ' + '  '.join(f'{k}={v:.1%}' for k, v in train_ds.mix().items()))
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
    # `multiprocessing_context='spawn'`: `persistent_workers` forks on the loader's first
    # `next()`, which is AFTER `model.to(device)` -- so the default `fork` start method hands
    # every worker a live CUDA context it never asked for and never uses. Measured on calms21:
    # each worker then independently grows its OWN heap by several GB within the first handful
    # of items (private, not COW-shared -- confirmed via smaps), an LSF `TERM_MEMLIMIT` in well
    # under ten minutes that no CUDA-free or checkpoint-free reproduction of the loader alone
    # ever showed. `spawn` starts each worker as a fresh interpreter that never touches CUDA at
    # all (dev/plans/video_loader_memory.md). Costs a few seconds of import time once, at start.
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, num_workers=nw, collate_fn=pose_collate,
        sampler=torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=n_iter),
        persistent_workers=nw > 0, pin_memory=True, drop_last=True,
        prefetch_factor=4 if nw > 0 else None, worker_init_fn=worker_init,
        multiprocessing_context='spawn' if nw > 0 else None)

    # THE SAME WINDOWS EVERY TIME. `PoseDataset` seeds val items by index, so materialising a
    # fixed set of them once makes every evaluation comparable to the last -- which is the only
    # thing that lets a val curve mean anything across iterations.
    #
    # SPREAD ACROSS THE INDEX, not the first N. The index is ordered dataset -> session -> group,
    # so a prefix is a prefix of the SESSION list: allen's first 20 of 65 val windows were 3 from
    # `behavior_750095` and 17 from `behavior_750096`, and the tracked `mouse1` session -- the one
    # this project's reference numbers were measured on -- contributed 2 windows and never
    # entered the metric at all. Checkpoint selection ran off that number. `linspace` costs
    # nothing and keeps every val session represented.
    val_freq = int(train_cfg.get('val_freq', 0))
    val_batches = []
    if val_ds is not None and val_freq:
        n_val = min(int(train_cfg.get('val_batches', 20)), len(val_ds))
        idxs = np.unique(np.linspace(0, len(val_ds) - 1, n_val).round().astype(int))
        # THROUGH A CHILD, NOT IN THIS PROCESS, and that is a hang not a preference. These windows
        # used to be decoded here, in the parent, which is BEFORE the train loader above forks its
        # workers (`persistent_workers` forks on the first `next()`, not at construction). On a
        # video-backed root that initialises decord in the parent, and the forked workers then
        # deadlock in a futex while holding an open container -- 0% GPU, zero worker CPU, no error,
        # forever. Measured on calms21: hangs at every reader-cache size, and does not hang when the
        # parent decodes nothing. 3dpop is video-backed too and survives it, which is the only
        # reason the five-dataset sweep never hit this.
        #
        # One worker, no shuffle, `Subset` in `idxs` order, so `val_batches` is the same list in the
        # same order it always was -- and it is byte-identical, not merely equivalent, because
        # `PoseDataset` seeds val items by index rather than by RNG draw.
        val_batches = list(torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_ds, [int(i) for i in idxs]), batch_size=1,
            num_workers=1, collate_fn=pose_collate, worker_init_fn=worker_init))
        seen = Counter(val_ds.index[int(i)].session.session_id for i in idxs)
        print(f'val:   {len(val_batches)} fixed window(s) every {val_freq} iterations, '
              f'over {len(seen)}/{len({it.session.session_id for it in val_ds.index})} session(s)')
        for sid, n in sorted(seen.items()):
            print(f'         {n:3d}  {sid}')

    # -- model -----------------------------------------------------------------------------
    model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    fresh: set[str] = set()
    if not args.no_warm_start and train_cfg.get('checkpoint_path'):
        fresh = warm_start(model, resolve_checkpoint(Path(train_cfg['checkpoint_path'])),
                           base_names=base_reg.names if base_reg else None)
    # THE ENCODER'S GRADIENT STATE IS `[model].video_encoder_requires_grad` AND NOTHING ELSE.
    # `[training].freeze_encoder` used to re-freeze the encoder here, after `build_model` had
    # already honoured the model key -- two sources of truth for one fact, and the loop's one
    # would have silently won over the staged unfreeze. Same rule as `per_camera_cube_scale`.
    # `[training]` has no unknown-key guard, so a config still naming it is refused by name
    # rather than ignored.
    if 'freeze_encoder' in train_cfg:
        raise SystemExit(
            '[training].freeze_encoder was removed: it re-froze the encoder AFTER build_model, so '
            'it silently overrode [model].video_encoder_requires_grad and would defeat the staged '
            'unfreeze. Set [model].video_encoder_requires_grad instead -- false (never train the '
            'encoder), true (from step 0), or an ITERATION to unfreeze at -- with '
            '[model].video_encoder_finetune_last_n_layers for how many trailing blocks.')
    enc_grad = config['model'].get('video_encoder_requires_grad', False)
    n_last = config['model'].get('video_encoder_finetune_last_n_layers')
    n_frozen = sum(1 for n, p in model.named_parameters()
                   if n.startswith('scene_encoder.encoder.') and not p.requires_grad)
    if isinstance(enc_grad, bool):
        # `= true` unfreezes inside the CONSTRUCTOR, which never reaches `apply_staged_unfreeze`,
        # so the norms extension has to be applied here or `true` + n_last would train a different
        # parameter set than an int + the same n_last. Before `build_optimizer`, so the norms are
        # routed into a group like any other trainable tensor.
        ext = apply_norms_extension(model)
        n_frozen = sum(1 for n, p in model.named_parameters()
                       if n.startswith('scene_encoder.encoder.') and not p.requires_grad)
        if not enc_grad:
            print(f'encoder: frozen for the whole run ({n_frozen} frozen tensors)')
        else:
            print(f'encoder: trainable from step 0, last '
                  f'{n_last if n_last is not None else "all"} block(s)'
                  f'{f" + norms {ext}" if ext else ""} ({n_frozen} frozen tensors)')
    else:
        print(f'encoder: frozen now, last {n_last if n_last is not None else "all"} block(s) '
              f'unfreeze at iteration {int(enc_grad)} ({n_frozen} frozen tensors)')
    model = model.to(device)

    # AN UNKNOWN [training.optimizer] KEY IS A TYPO, NOT A COMMENT -- the same guard `[data]` and
    # `build_model` already apply. This block was splatted unguarded, so `muon_lr` for
    # `muon_lr_scale` would train at the default and the run folder would record the key nobody
    # read (eval rule 4). This is what makes "config key only" safe as the whole selection surface.
    opt_cfg = config['training']['optimizer']
    from tailcyclenet.optim import KNOWN_OPTIMIZER_KEYS
    opt_unknown = set(opt_cfg) - KNOWN_OPTIMIZER_KEYS
    if opt_unknown:
        raise SystemExit(
            f'[training.optimizer]: unknown key(s) {sorted(opt_unknown)}. Nothing reads them, so '
            f'this run would train at the defaults and report as the arm it is not. Known keys: '
            f'{sorted(KNOWN_OPTIMIZER_KEYS)}')
    # READ THE CHECKPOINT BEFORE BUILDING THE OPTIMIZER, because `start_it` decides whether the
    # staged unfreeze has already fired for the run being continued -- and that changes the param
    # GROUPS the state below is keyed against by position. One 5.6 GB read, not two; only the read
    # moves, the load ORDER below is unchanged.
    start_it, ck, resumed = 0, None, run / 'checkpoints' / 'checkpoint_last.pth'
    if resumed.exists() and not args.no_resume:
        ck = torch.load(resumed, map_location=device, weights_only=False)
        start_it = int(ck['iteration'])
    opt = build_optimizer(model, fresh, opt_cfg)
    # `per_camera_cube_scale` is a GAUGE, and the model and the loss have to agree on it. It lives
    # in `[model]`, and `TotalLoss` defaults it False (`losses.py:98`) -- so leaving it out of
    # `[training.losses]` had the loss take the median over cameras while the model kept its own
    # per camera. Derived rather than duplicated: two copies of one gauge is a config away from
    # silently disagreeing again.
    losses = dict(config['training']['losses'])
    losses.setdefault('per_camera_cube_scale',
                      bool(config['model'].get('per_camera_cube_scale', False)))
    loss_fn = TotalLoss(**losses)

    # RESUME, because the jobs are preemptible and re-running into the same --out is the
    # documented way to restart one (`init_wandb`, above). This used to begin again at iteration 0
    # AND destroy what the previous run had earned: `checkpoint_last.pth` is overwritten at the
    # first boundary, and `saved_mpjpe = inf` replaces `checkpoint_best.pth` with whatever the
    # fresh run reaches first. Both files are ~5.6 GB and a 60k run has no other copy.
    # `optimizer_state` has been written since day one and read by nothing; this is that read.
    #
    # `model_state`, not `model_state_eval`: the raw iterate is what training continues from, and
    # the averaged one is for evaluating. Warm start still runs above -- wasteful, but it is what
    # fixes the `fresh` set and therefore the optimizer's parameter GROUPS, which the state below
    # is keyed against by position.
    #
    # NOT restored: the sampler position. The sampler draws with replacement, so window order
    # carries no contract, and a resumed run is simply not bit-identical to an uninterrupted one
    # -- inside the ~0.041 mm run-to-run floor either way.
    if ck is not None:
        model.load_state_dict(ck['model_state'])
        # REPLAY THE STAGED UNFREEZE BEFORE LOADING OPTIMIZER STATE. The optimizer above was
        # built in the FROZEN layout whatever `start_it` is, so replaying the adds here reaches
        # the same group ORDER a fresh run would hold at this iteration -- and `load_state_dict`
        # matches groups by position, so order is the whole contract.
        info = replay_staged_unfreeze(model, opt, opt_cfg, start_it, fresh=fresh)
        if info:
            print(f'resume: replayed the encoder unfreeze (blocks {info["blocks"]}, '
                  f'norms {info["norms"]}, {info["n_params"]:,} params)')
        # REFUSE, by name, an optimizer state written by a DIFFERENT optimizer than this config
        # builds. The model tensors above loaded either way; only the state's own shape can tell
        # you that the run is being resumed under the wrong optimizer -- and making `muon` the
        # default means every AdamW-SF run folder now hits this. Loud rather than a KeyError deep
        # in `load_state_dict`. This is the gridresid_offset rule for optimizer state.
        from tailcyclenet.optim import refuse_mismatched_optimizer_state
        refuse_mismatched_optimizer_state(
            opt, ck['optimizer_state'], resumed,
            resolved=str(config['training']['optimizer'].get('optimizer', 'muon')),
            explicit='optimizer' in config['training']['optimizer'])
        opt.load_state_dict(ck['optimizer_state'])
        print(f'resuming {resumed} at iteration {start_it} of {n_iter} '
              '(--no-resume to start over, which OVERWRITES both checkpoints)')
    elif resumed.exists():
        print(f'--no-resume: {resumed} and any checkpoint_best.pth beside it WILL be overwritten')

    # RECORD THE RESOLVED `box_source`, not just the one the toml happened to name. Its default
    # moved from 'keypoints' to 'instances', and `scripts/infer.py` reads the run config with
    # `.get('box_source', 'keypoints')` -- which is RIGHT for every run trained before the move and
    # WRONG for a run that takes the new default silently. Writing it makes the run folder say what
    # it trained as, which is the whole job of a run folder; gotcha 12 is what the alternative cost.
    config.setdefault('data', {})['box_source'] = train_ds.cfg.box_source
    # RECORD THE RESOLVED OPTIMIZER too, for the same reason: an absent key means "muon", and a run
    # folder must be able to say what it trained as. A folder that resolved to muon by default and a
    # folder that will only resume as "schedulefree" must not read identically.
    opt_cfg['optimizer'] = str(opt_cfg.get('optimizer', 'muon'))
    save_run_meta(run, config, registry)
    print(f'run folder: {run.resolve()}')
    wb = init_wandb(config, run, disabled=args.no_wandb)
    log_path = run / 'log.jsonl'                  # the numbers survive without wandb

    def record(rec):
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    # The sampling mix goes in the LOG, not just stdout. It is not recoverable from `config.toml`
    # after the fact -- `annot_frac` and `mode_3d_frac` are conditionals whose realised shares
    # depend on which levels each dataset actually has, so a root with no 2D session skips the
    # mode level entirely and its whole annotated share lands on 3D. Reconstructing that by hand
    # is how a 9.6% figure (the uniform-over-entries baseline) got read as the run's real 88%.
    record({'iter': 0, 'mix': train_ds.mix(), 'n_windows': len(train_ds),
            'n_keypoints': registry.n_keypoints})

    # -- loop ------------------------------------------------------------------------------
    max_grad = float(train_cfg.get('max_grad_norm', 0)) or None
    # CLIP ONLY THE AdamW-ROUTED PARAMS ON A MUON RUN. Muon orthogonalizes its own gradients
    # inside `step()` (Newton-Schulz), so their RAW norm is not a step size -- clipping it is
    # meaningless. Worse, on this model the Muon-routed 2D matrices carry ~95% of the global grad
    # norm, so a single shared `clip_grad_norm_(model.parameters())` computes a ~0.05 coefficient
    # and applies it to the freshly-initialised heads/embeddings/norms too, throttling exactly the
    # AdamW half that needs to learn fastest (report 34b: measured 8-20x on the AdamW grads, and a
    # 3-7x higher reported grad_norm than the schedulefree arm at matched steps). So the clip
    # targets `opt.adamw_params`; Muon's half is left to its own orthogonalisation. For a
    # schedule-free (non-dual) optimizer this is every trainable param, exactly as before.
    def _clip_targets():
        """RECOMPUTED AFTER AN UNFREEZE, not captured once. A stale list leaves the newly-trainable
        encoder AdamW tensors unclipped and the new Muon tensors unchecked for non-finiteness --
        and report 34b's whole point is that WHICH half is clipped is not incidental."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        return getattr(opt, 'adamw_params', trainable), getattr(opt, 'muon_params', [])

    clip_params, unclipped_params = _clip_targets()
    # The Muon half is not clipped, but its gradient still must not be allowed to go non-finite
    # into `step()` -- the all-params clip used to be what caught that. `muon_params` is empty for
    # a schedule-free optimizer, so this guard is inert there.
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
    it, skipped, t0, running, clipped = start_it, 0, time.time(), [], []
    # best_mpjpe/best_iter: the best val at ANY val step -- a number, used for plateau detection.
    # saved_mpjpe: the metric of the file currently on disk as `checkpoint_best.pth`. The two are
    # different on purpose; only the second one costs a 3.15 GB write. `latest` is the most recent
    # val and the iteration it came from, so the checkpoint block can tell a fresh number from a
    # stale one.
    best_mpjpe, best_iter, saved_mpjpe = float('inf'), start_it, float('inf')
    # ON RESUME, `saved_mpjpe` DESCRIBES A FILE THAT STILL EXISTS. Left at inf it would be beaten
    # by the first val the resumed run produces, which is how a good `checkpoint_best.pth` gets
    # replaced by a worse one. Read back from the log rather than from the checkpoint, because it
    # is the metric of the `best` file and the `best` file does not record it.
    if start_it and log_path.exists():
        prev = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
        marks = [r for r in prev if 'saved_mpjpe' in r]
        if marks:
            saved_mpjpe = float(marks[-1]['saved_mpjpe'])
            best_mpjpe, best_iter = saved_mpjpe, int(marks[-1]['iter'])
            print(f'resuming: checkpoint_best.pth holds mpjpe {saved_mpjpe:.4g} '
                  f'from iteration {best_iter}; it is only replaced by something better')
        else:
            print('resuming: no saved_mpjpe in log.jsonl, so checkpoint_best.pth is unattributed '
                  'and the first val of this run will replace it')
    # The train loader's persistent workers fork/spawn on its first `next()`, i.e. inside this
    # loop -- AFTER build_model, warm_start's multi-GB `torch.load`, and the optimizer have all
    # run and freed their scratch memory in this process. Tried as a candidate fix for calms21's
    # per-worker RSS plateau and made no measurable difference on a real job
    # (dev/plans/video_loader_memory.md §2.3) -- kept anyway, since a `malloc_trim` here is a
    # cheap, harmless attempt to hand a fresh worker the smallest possible parent to inherit.
    ctypes.CDLL('libc.so.6').malloc_trim(0)

    latest = (float('inf'), -1)
    waited, evalled, ckpted = [0.0], [0.0], [0.0]
    while it < n_iter:
        for batch in timed(loader, waited):
            if it >= n_iter:
                break
            # THE STAGED UNFREEZE. Upstream owns the gate, the idempotence and the block
            # selection (`TrackerEncoder.unfreeze_video_encoder`); `apply_staged_unfreeze` adds
            # the newly-trainable tensors to the optimizer, which is the half a `requires_grad`
            # flip alone cannot do -- a frozen param is in no param group, so it would receive
            # gradients that nothing steps.
            unfroze = apply_staged_unfreeze(model, opt, opt_cfg, it, fresh=fresh)
            if unfroze:
                clip_params, unclipped_params = _clip_targets()
                print(f'  UNFROZE the video encoder at iteration {it}: blocks '
                      f'{unfroze["blocks"]} of {unfroze["n_blocks"]}, norms {unfroze["norms"]}, '
                      f'{unfroze["n_tensors"]} tensors / {unfroze["n_params"]:,} params '
                      f'(+{unfroze["muon_groups"]} muon, +{unfroze["adamw_groups"]} adamw '
                      f'group(s))', flush=True)
                log(wb, {'train/encoder_unfrozen_at': it}, it)
                record({'iter': it, 'unfroze_encoder': unfroze})
            loss, _ = run_batch(model, loss_fn, batch, device)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(clip_params, max_grad or 1e9)
            # `gn` now measures the AdamW half only; the Muon half is checked for finiteness
            # separately (its grads are stepped unclipped, so a NaN there would otherwise slip in).
            mgrads = [p.grad for p in unclipped_params if p.grad is not None]
            mgn = torch.nn.utils.get_total_norm(mgrads) if mgrads else gn.new_zeros(())
            if not (torch.isfinite(gn) and torch.isfinite(mgn)):
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
                # Peak READ here and reset in the `log` call below -- one read per window, so the
                # printed number and the logged one are the same peak rather than two windows'.
                peak_gb = _gpu_peak_gb(device)
                print(f'{it:7d}/{n_iter}  loss {np.mean(running):8.4f}  '
                      f'{dt:5.2f}s/it  wait {wait_frac:4.0%}  eval {eval_frac:4.0%}  '
                      f'peak {peak_gb:5.1f}G  skipped {skipped}  '
                      f'[{batch.sample_info["dataset"]}/{batch.sample_info["mode"]}'
                      f'{"/1cam" if batch.sample_info["single_view"] else ""}]', flush=True)
                # `clipped` is the running fraction of steps hitting max_grad_norm. At
                # batch_size 1 it has measured ~50%, which makes it the most informative
                # training-health number here -- and it was being computed and thrown away.
                # `train/grad_norm` is the CLIPPED (AdamW) half; `train/grad_norm_muon` is the
                # unclipped Muon half, which reads several-fold higher by construction (it is a raw
                # pre-orthogonalisation norm, not a step size -- report 34b). 0 for a schedulefree run.
                log(wb, {'train/loss': float(np.mean(running)), 'train/grad_norm': float(gn),
                         'train/grad_norm_muon': float(mgn),
                         # The iteration as a plain METRIC, not just the wandb step: a resumed run
                         # and a fresh one carry different step offsets, so panels compared across
                         # runs need an x-axis that says which iteration a point is.
                         'train/iteration': it,
                         'train/encoder_trainable_params': trainable_encoder_params(model),
                         # PEAK GPU BYTES SINCE THE LAST PRINT, not the current allocation. The
                         # staged unfreeze adds optimizer state AND the backward activations of N
                         # blocks, and an L4 is 22.5 GB -- so whether a recipe fits is a property
                         # of the PEAK, which nothing here logged. Reset each window so a spike is
                         # attributable to the iterations it happened in rather than to the run.
                         'train/gpu_peak_gb': _gpu_peak_gb(device, reset=True),
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

            if (it % ckpt_freq == 0 or it == n_iter) and not args.no_checkpoints:
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
                    # WHAT THE `best` FILE HOLDS, in the log, because the file itself does not
                    # record it -- and a resumed run that cannot read this replaces a good `best`
                    # with its own first val.
                    record({'iter': it, 'saved_mpjpe': saved_mpjpe})
                ckpted[0] += time.time() - t_ck
                print(f'saved {p} ({time.time() - t_ck:.0f}s)')
    print(f'done: {it} iterations, {skipped} skipped')
    record({'iter': it, 'done': True, 'skipped': skipped,
            'best_mpjpe': best_mpjpe, 'best_iter': best_iter})
    if wb is not None:
        wb.finish()


if __name__ == '__main__':
    main()
