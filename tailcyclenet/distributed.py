"""The world-size axis: one item per rank per step, gradients averaged by DDP.

`batch_size` is structurally 1, so the WORLD is the batch dimension; everything here is a pure
function of (fabric, tensors) or plain ints, so the arithmetic is testable in one process.

DDP takes its parameter headcount at wrap time and never re-checks: a tensor frozen at
`fabric.setup_module` is never all-reduced later. The staged unfreeze re-wraps the module when
it fires, and `check_ranks_agree` proves the re-wrap happened.
"""
from __future__ import annotations

import math

import torch


def ceil_div(x: int, world: int) -> int:
    """`ceil(x / world)`, with no floor. The resume position, which may legitimately be 0."""
    return -(-int(x) // int(world))


def per_rank(x: int, world: int) -> int:
    """A TOTAL-ACROSS-GPUS count as a per-rank one. Floor 1: a frequency that rounds to 0 would
    fire on every step (or divide by zero).
    """
    return max(1, ceil_div(x, world))


def scale_optimizer_cfg(cfg: dict, world: int) -> dict:
    """A COPY of `[training.optimizer]` with the ABSOLUTE rates (`learning_rate`, `kpt_lr`)
    multiplied by `sqrt(world)`. Multipliers are untouched, so every ratio in the recipe is
    preserved. Every consumer (build and the unfreeze paths) must get the same copy, or the
    encoder would arrive at an unscaled rate mid-run.
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
    """True iff EVERY rank passed True. A skipped step has to be a collective decision: a rank
    that skips its backward while others run theirs leaves them waiting in the all-reduce.
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
    """Every rank's per-window metric dicts, concatenated, on every rank. Val is sharded, so
    gathering gives the nanmean over the same window set a single-GPU run would produce; a rank
    with an empty shard still takes part -- the gather is collective.
    """
    if fabric is None or fabric.world_size <= 1:
        return list(mets)
    import torch.distributed as dist

    out: list = [None] * fabric.world_size
    dist.all_gather_object(out, list(mets))
    return [m for part in out for m in part]


def check_registry(fabric, names) -> None:
    """Raise unless every rank built the SAME keypoint registry -- embedding rows are keypoint
    ids, and each rank discovers datasets independently.
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
    """A small float64 fingerprint of the trainable parameters: per tensor, sum and sum of
    squares. `dtype=torch.float64` on the REDUCTION only -- `.double()` would materialise a full
    float64 copy of every parameter at checkpoint time.
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

    THE GUARD FOR THE DDP HEADCOUNT RULE: weights should be bit-identical, and a tensor that
    stopped being all-reduced drifts silently. `tol` sits between two measured populations
    (correct runs read exactly 0.0, the one real bug ~3e-07), so it must not be loosened toward
    1e-3. Collective on purpose, so every rank raises together. The signature is compared ON
    THE CPU: fabric's `broadcast` pickles CUDA tensors to the device index they were sent from,
    so a GPU signature would land on the wrong device on every non-zero rank. `NaN > tol` is
    False, so a poisoned parameter would slip through as agreement -- hence the explicit finite
    check.
    """
    if fabric is None or fabric.world_size <= 1:
        return 0.0
    sig = param_signature(model).cpu()
    ref = fabric.broadcast(sig, src=0)
    diff = signature_drift(sig, ref.cpu())
    worst = float(fabric.all_reduce(torch.tensor(diff, device=fabric.device), reduce_op="max"))
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
    """The worst PER-ENTRY relative difference, separated out to be testable without a process
    group. Per entry, not per whole signature: a global normaliser measures the biggest tensor
    and calls everything else agreement.
    """
    if sig.numel() == 0:
        return 0.0
    return float(((sig - ref).abs() / ref.abs().clamp_min(1e-12)).max())


class RankPrefix:
    """A stdout wrapper tagging every line with its rank, installed on the non-zero ranks.

    NOTHING IS SUPPRESSED -- setup-phase prints from code this loop does not own are exactly the
    lines you need when one rank disagrees.
    """

    def __init__(self, stream, rank: int):
        """Wrap `stream` so every line written through this object carries `[rank N]`.

        Inputs: stream -- the text stream to wrap; rank -- this process's rank.
        """
        self._stream, self._tag, self._at_line_start = stream, f'[rank {rank}] ', True

    def write(self, text: str) -> int:
        """Write `text` to the wrapped stream, tagging each new line with the rank prefix.

        Inputs: text -- the string to write.
        Outputs: int -- the number of characters accepted (len(text)).
        Side effects: writes to the wrapped stream.
        """
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
        """Flush the wrapped stream."""
        self._stream.flush()

    def isatty(self) -> bool:
        """Whether the wrapped stream is a terminal."""
        return self._stream.isatty()
