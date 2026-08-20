"""The world-size axis: one item per rank per step, gradients averaged by DDP.

`batch_size` is structurally 1 -- posetail's collate keeps only item 0's camera group and the
model takes one camera group per batch -- so the only batch dimension available to this repo is
the WORLD. `--devices N` runs N ranks of the existing loop and lets DDP average their gradients,
which makes the effective batch N without touching `pose_collate` or the one-camera-group
contract.

Everything here is a pure function of `(fabric, tensors)` or of plain ints, so the arithmetic that
decides what a config key MEANS on N GPUs is testable in one process. `scripts/train.py` holds the
loop; this holds the rules.

THE ONE RULE THAT IS NOT OBVIOUS: **DDP takes its parameter headcount at wrap time.**
`_build_params_for_reducer` (torch/nn/parallel/distributed.py:1338) filters `requires_grad` when
the module is wrapped and nothing re-checks afterwards, so a tensor that was frozen at
`fabric.setup_module` is NEVER all-reduced no matter what `requires_grad` becomes later. This repo
ships the staged encoder unfreeze (8 blocks at iteration 10,000) and `tailcyclenet.unfreeze` makes
those tensors genuinely step, so the loop re-wraps the module when the unfreeze fires and
`check_ranks_agree` is what keeps proving that the re-wrap happened.
"""
from __future__ import annotations

import math

import torch


def ceil_div(x: int, world: int) -> int:
    """`ceil(x / world)`, with no floor. The resume position, which may legitimately be 0."""
    return -(-int(x) // int(world))


def per_rank(x: int, world: int) -> int:
    """A TOTAL-ACROSS-GPUS count as a per-rank one. `total_to_per_gpu`, ported.

    `n_iterations`, `val_freq`, `checkpoint_freq` and `print_freq` are totals: `n_iterations =
    60000` is 60,000 SAMPLES whatever the GPU count, so an N-GPU run does 60,000/N optimizer steps
    and finishes about N times sooner. Every shipped config keeps its meaning across GPU counts,
    and at `world = 1` this is the identity.

    Floor 1, because a frequency that rounds to 0 would make `step % freq` fire on every step (or
    divide by zero), and a run with more GPUs than iterations still has to do one.
    """
    return max(1, ceil_div(x, world))


def scale_optimizer_cfg(cfg: dict, world: int) -> dict:
    """A COPY of `[training.optimizer]` with the ABSOLUTE rates multiplied by `sqrt(world)`.

    The effective batch grows 1 -> N, so the rate grows with its square root, as in the reference.
    Only `learning_rate` and `kpt_lr` are absolute; `encoder_lr_scale`, `muon_lr_scale` and the
    rest are MULTIPLIERS on those, so every ratio in the recipe is preserved and a multi-GPU run
    is one lever off its single-GPU control rather than several.

    A COPY, and the copy is what `build_optimizer`, `apply_staged_unfreeze` AND
    `replay_staged_unfreeze` all receive -- `optim.group_lr` reads this dict when the staged
    unfreeze adds its groups, so a copy that reached only the build would hand the encoder an
    unscaled rate at iteration 10,000 and nothing would say so. The run folder's `config.toml`
    keeps the CONFIGURED value (scaling it there would re-scale on every resume); the effective
    numbers are recorded in `provenance.toml`.
    """
    if int(world) <= 1:
        return dict(cfg)
    f = math.sqrt(int(world))
    out = dict(cfg)
    out['learning_rate'] = float(cfg['learning_rate']) * f
    if 'kpt_lr' in cfg:
        out['kpt_lr'] = float(cfg['kpt_lr']) * f
    return out


def all_ranks_finite(fabric, ok: bool) -> bool:
    """True iff EVERY rank passed True. A skipped step has to be a collective decision.

    `run_batch` returns NaN for a poisoned loss and the loop skips the step -- but under DDP a
    rank that skips its backward while the others run theirs leaves them waiting in the gradient
    all-reduce forever. So the finiteness flag is reduced with MIN first and every rank skips
    together, which also keeps `skipped` and `train/skipped_frac` meaning the same thing on every
    rank.
    """
    if fabric is None or fabric.world_size <= 1:
        return bool(ok)
    t = torch.tensor(1.0 if ok else 0.0, device=fabric.device)
    return bool(float(fabric.all_reduce(t, reduce_op='min')) > 0.0)


def all_ranks_mean(fabric, value: float) -> float:
    """The mean of a scalar across ranks. `train/loss` is a batch statistic, not rank 0's draw."""
    if fabric is None or fabric.world_size <= 1:
        return float(value)
    t = torch.tensor(float(value), device=fabric.device)
    return float(fabric.all_reduce(t, reduce_op='mean'))


def gather_metrics(fabric, mets: list[dict]) -> list[dict]:
    """Every rank's per-window metric dicts, concatenated, on every rank.

    Validation is SHARDED (`idxs[rank::world]`) because it was the larger half of one root's
    projected wall time. Gathering the per-window dicts and reducing them with the loop's own
    `_reduce` gives the nanmean over the SAME window set a single-GPU run would produce -- so the
    val curve is not a different metric on more GPUs -- and it leaves every rank holding the same
    number, which is what lets `latest`/`best_mpjpe` agree while only rank 0 writes files.

    A rank whose shard is empty still takes part: `all_gather_object` is collective, and skipping
    it on the short rank is a hang.
    """
    if fabric is None or fabric.world_size <= 1:
        return list(mets)
    import torch.distributed as dist

    out: list = [None] * fabric.world_size
    dist.all_gather_object(out, list(mets))
    return [m for part in out for m in part]


def check_registry(fabric, names) -> None:
    """Raise unless every rank built the SAME keypoint registry.

    Each rank discovers datasets from the filesystem independently. A listing that ordered
    differently on one rank would give that rank's embedding table different rows for the same
    body parts -- the hazard `warm_start`'s row-copy refusal exists for, now with N chances to
    happen and nothing downstream that would notice.
    """
    if fabric is None or fabric.world_size <= 1:
        return
    ref = fabric.broadcast(tuple(names), src=0)
    if tuple(names) != tuple(ref):
        raise SystemExit(
            f'[rank {fabric.global_rank}] built a different keypoint registry than rank 0: '
            f'{len(names)} name(s) vs {len(ref)}, first difference at '
            f'{next((i for i, (a, b) in enumerate(zip(names, ref)) if a != b), min(len(names), len(ref)))}. '
            'Keypoint ids are embedding rows, so two orderings mean two different models. Check '
            'that every rank sees the same [data].path.')


def param_signature(model) -> torch.Tensor:
    """A small float64 fingerprint of the trainable parameters: per tensor, sum and sum of squares.

    Two scalars per tensor rather than the tensors themselves, because this is compared across
    ranks at every checkpoint boundary and the weights are gigabytes. Any drift in any tensor moves
    its sum of squares; `named_parameters()` order is deterministic and identical on every rank.

    `dtype=torch.float64` ON THE REDUCTION, never `.double()` on the tensor. The latter materialises
    a full float64 COPY of every parameter -- 8 bytes a param, transiently, at the exact moment a
    checkpoint is also being written -- to produce two scalars. `sum(dtype=)` and `vector_norm(...,
    dtype=)` accumulate in float64 straight off the fp32 storage and allocate nothing but the
    scalar. Same numbers, and the reason this is affordable at all.
    """
    vals = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        d = p.detach()
        vals.append(torch.stack([d.sum(dtype=torch.float64),
                                 torch.linalg.vector_norm(d, 2, dtype=torch.float64).square()]))
    if not vals:
        return torch.zeros(0, dtype=torch.float64)
    return torch.stack(vals).flatten()


def check_ranks_agree(fabric, model, tol: float = 1e-9) -> float:
    """Raise if the ranks' weights have drifted apart. Returns the worst relative difference.

    THIS IS THE GUARD FOR THE DDP HEADCOUNT RULE at the top of this module. Every rank steps
    identical averaged gradients, so the weights should be bit-identical; a tensor that stopped
    being all-reduced -- because it was frozen when the module was wrapped and the unfreeze
    re-wrap did not fire -- drifts immediately and silently, and the loss curve does not care.

    **`tol` IS SET FROM THE SEPARATION BETWEEN TWO MEASURED POPULATIONS, NOT FROM TASTE.**

      * a correct run reads **exactly 0.0**, every time -- six clean 2-rank runs (with and without
        checkpoint writes, with and without the staged unfreeze, 1200 iterations each) at every
        checkpoint boundary. Not "small": zero. NCCL hands every rank bitwise-identical reduced
        gradients and the optimizer step is deterministic given them.
      * the one bug this has ever caught -- the unfreeze re-wrap not firing, negative-controlled --
        reads **2.9e-07 to 4.9e-07**, on two cluster jobs and two local ones.

    So the two populations are separated by seven orders of magnitude and `tol` can sit anywhere
    between. 1e-9 is a thousand times looser than exact equality, which is real headroom for a
    noise floor nobody has observed yet, and still ~300x below the smallest real failure on record.
    **Do not raise it into the 1e-3 range**: that is 2,000x ABOVE the bug's signature, so the guard
    would pass the exact failure it exists for while still looking like it was doing something. A
    first version of this made that mistake in the other direction (normalising by the largest
    signature entry, `tol = 1e-5`) and waved a real divergence through at 5.6e-10.

    A drift in the 1e-7 range is NOT harmless-because-small: it is not a rounding wobble, it is
    two ranks training two different models, and it GROWS -- 2.9e-07 at one boundary, 4.9e-07 by
    the next. The number is small only because it is caught early.

    **IT COSTS NOTHING WORTH A KNOB, MEASURED.** On the real 3D model: 25.8 ms at 315 trainable
    tensors / 47.9M params, 38.1 ms at 411 / 148.7M (8 encoder blocks unfrozen). It fires only at
    checkpoint boundaries -- `checkpoint_freq = 1000` -- beside a ~5.6 GB checkpoint write, so it
    is 0.005-0.008% of a 1000-step window at 0.5 s/it and a smaller share of the write it sits
    next to. There is deliberately no frequency setting: a knob whose right value is "every time"
    is one more way for a run to report as the arm it is not.

    Collective on purpose: the verdict itself is all-reduced so every rank raises together rather
    than one rank raising and the rest hanging in the next reduction.
    """
    if fabric is None or fabric.world_size <= 1:
        return 0.0
    # ON THE CPU, and that is not tidiness. Fabric's `broadcast` pickles through
    # `broadcast_object_list`, so a CUDA tensor arrives on the device index it was SENT from --
    # rank 0's `cuda:0` lands on rank 1 as `cuda:0` while its own signature is on `cuda:1`, and
    # the subtraction below raises a device mismatch. One tiny transfer, once per checkpoint.
    sig = param_signature(model).cpu()
    ref = fabric.broadcast(sig, src=0)
    diff = signature_drift(sig, ref.cpu())
    worst = float(fabric.all_reduce(torch.tensor(diff, device=fabric.device), reduce_op="max"))
    # `NaN > tol` is False, so a poisoned parameter would slip through the comparison below as
    # agreement. Checked by name instead.
    if worst > tol or not math.isfinite(worst):
        raise SystemExit(
            f'THE RANKS HOLD DIFFERENT WEIGHTS: worst relative parameter drift {worst:.3e} > '
            f'{tol:g}, where a correct run reads exactly 0. DDP registers parameters when the '
            'module is WRAPPED and never re-checks (torch/nn/parallel/distributed.py:1338), so a '
            'tensor unfrozen after the wrap is never all-reduced and each rank trains its own '
            'copy. Check that the staged encoder unfreeze re-wrapped the module (look for '
            '"re-wrapped DDP" in this log), and that no rank-dependent state reached the weights.')
    return worst


def signature_drift(sig: torch.Tensor, ref: torch.Tensor) -> float:
    """The worst PER-ENTRY relative difference. Separated out to be testable without a process
    group.

    Per entry, not normalised by the largest entry of the whole signature: one tensor's sum of
    squares can be six orders of magnitude larger than another's, so a global normaliser measures
    the biggest tensor and calls everything else agreement. That is exactly how the first version
    of this rated a real divergence at 5.6e-10 and passed it.
    """
    if sig.numel() == 0:
        return 0.0
    return float(((sig - ref).abs() / ref.abs().clamp_min(1e-12)).max())


class RankPrefix:
    """A stdout wrapper tagging every line with its rank. Installed on the non-zero ranks.

    NOTHING IS SUPPRESSED. The setup phase prints from places this loop does not own -- warm start
    naming every dropped tensor, `PoseDataset` naming the sampling mix, `build_muon` naming each
    param group -- and those are exactly the lines you need when ONE rank disagrees. Silencing the
    other ranks would hide the only evidence of that; tagging keeps it attributable. The loop's
    own high-volume lines go through `fabric.print` and appear once.
    """

    def __init__(self, stream, rank: int):
        self._stream, self._tag, self._at_line_start = stream, f'[rank {rank}] ', True

    def write(self, text: str) -> int:
        if not text:
            return 0
        out = []
        for piece in text.splitlines(keepends=True):
            if self._at_line_start:
                out.append(self._tag)
            out.append(piece)
            self._at_line_start = piece.endswith('\n')
        self._stream.write(''.join(out))
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()
