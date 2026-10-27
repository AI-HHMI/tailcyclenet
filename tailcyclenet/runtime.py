"""What every trainer shares: the Fabric launcher, the loader-wait timer and the wandb helpers.

Pose (`train.py`) and scorer (`scorer/train.py`) training both call these; one copy each, so a
launcher fix reaches both families.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

from . import distributed as dist_utils


def set_tf32(on: bool = True) -> bool:
    """Allow (or forbid) TF32 tensor-core matmuls and convolutions; returns `on`.

    PyTorch leaves matmul TF32 off by default, so under `32-true` an A100/H100 runs full-FP32
    GEMMs. Every trainer turns it on (since 2026-10-27); `--no-tf32` restores strict FP32.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)
    return bool(on)


def launch(args):
    """The Fabric, launched from `args.device/devices/strategy/precision/no_tf32`. Everything after this
    call runs once PER RANK.

    `--devices -1` means every visible GPU (one process on CPU) and resolves to that count
    before the strategy is chosen, so a one-GPU box takes the exact `--devices 1` path (strategy
    `auto`, no wrapper). Above one device the strategy is `ddp_find_unused_parameters_true`,
    because 2D and 3D items drive different parameters. `--device cuda:N` selects that GPU at
    one device and is refused beside `--devices > 1`, where the launcher places every rank;
    `--device cpu` stays CPU even when CUDA is available. `16-mixed` is refused by name.
    Fabric re-executes the script per rank, so everything before this call runs on every rank --
    keep it video-free (gotcha 10). Non-zero ranks TAG their stdout rather than silencing it.
    """
    from lightning.fabric import Fabric

    cpu = str(args.device).startswith('cpu') or not torch.cuda.is_available()
    devices = int(args.devices)
    if devices == -1:
        devices = 1 if cpu else torch.cuda.device_count()
    if devices < 1:
        raise SystemExit(f'--devices {args.devices}: a positive count, or -1 for every visible gpu')
    if args.precision == '16-mixed':
        raise SystemExit(
            '--precision 16-mixed needs a GradScaler applied to the optimizer, and this loop steps\n'
            'its own optimizer rather than handing it to Fabric (PoseDualOptimizer is not an\n'
            'Optimizer subclass, and at 32-true/bf16-mixed a Fabric optimizer wrapper adds\n'
            'nothing). Use --precision 32-true, or bf16-mixed.')
    if devices != 1 and not cpu and str(args.device) != 'cuda:0':
        raise SystemExit(
            f'--device {args.device} names ONE gpu and --devices {args.devices} asks for several. '
            'At --devices != 1 the launcher places every rank, so --device would be silently '
            'ignored -- pick one (CUDA_VISIBLE_DEVICES selects WHICH gpus).')
    tf32 = set_tf32(not getattr(args, 'no_tf32', False))
    strategy = args.strategy or ('auto' if devices == 1 else 'ddp_find_unused_parameters_true')
    if cpu:
        accelerator, dev_arg = 'cpu', devices
    elif devices == 1:
        accelerator = 'gpu'
        dev_arg = [int(str(args.device).rsplit(':', 1)[-1])] if ':' in str(args.device) else 1
    else:
        accelerator, dev_arg = 'gpu', devices
    fabric = Fabric(accelerator=accelerator, devices=dev_arg, strategy=strategy,
                    precision=args.precision)
    fabric.launch()
    if not fabric.is_global_zero:
        sys.stdout = dist_utils.RankPrefix(sys.stdout, fabric.global_rank)
    os.environ['TAILCYCLENET_LOCAL_WORLD_SIZE'] = str(fabric.world_size)
    if fabric.world_size > 1:
        fabric.print(f'distributed: {fabric.world_size} ranks, strategy {strategy!r}, '
                     f'precision {args.precision}. n_iterations/val_freq/checkpoint_freq/'
                     f'print_freq and the unfreeze iteration are TOTALS across ranks; the '
                     f'learning rate is scaled by sqrt({fabric.world_size}) = '
                     f'{fabric.world_size ** 0.5:.3f}')
    fabric.print(f'tf32: {"on" if tf32 else "off (--no-tf32)"}')
    if args.precision == 'bf16-mixed':
        fabric.print('--precision bf16-mixed is UNMEASURED on this repo\'s roots: every number '
                     'on record was measured at 32-true.')
    return fabric


def timed(loader, acc):
    """Yield from `loader`, accumulating seconds blocked on `next()` into `acc[0]` -- the loader
    starvation canary. Wrapping the iterator stays correct across `continue` paths."""
    it = iter(loader)
    while True:
        t = time.time()
        try:
            batch = next(it)
        except StopIteration:
            return
        acc[0] += time.time() - t
        yield batch


def init_wandb(config: dict, run: Path, disabled: bool = False):
    """wandb from the existing `[wandb]` block, or None. The run folder stays `--out`.

    The run name carries a timestamp because the folder alone is not unique across resumes.
    `[wandb].path` (wandb's log directory) defaults to the run folder itself.
    """
    cfg = config.get('wandb')
    if disabled or not cfg:
        return None
    try:
        import wandb
    except ImportError:
        print('wandb: not installed, skipping')
        return None
    log_dir = Path(cfg.get('path') or run)
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f'{time.strftime("%Y%m%d_%H%M%S")}-{run.name}'
    wandb.init(project=cfg.get('project_name', 'tailcyclenet'), dir=str(log_dir),
               mode=cfg.get('mode', 'offline'), name=name, config=config)
    print(f'wandb: {cfg.get("mode", "offline")} | {wandb.run.name}')
    return wandb


def log(wb, values: dict, step: int) -> None:
    """Log `values` at `step` if wandb is active."""
    if wb is not None:
        wb.log(values, step=step)


def gpu_peak_gb(device, reset: bool = False) -> float:
    """Peak allocator bytes since the last reset, in GB (0.0 on CPU); max, not current.

    `device` may be a `torch.device` (the loop passes `fabric.device`) or a `str` (tests and
    older callers), so it is compared as a string.
    """
    if str(device).split(':')[0] != 'cuda' or not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return peak
