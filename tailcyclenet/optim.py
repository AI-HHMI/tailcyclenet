"""SF-Muon for the pose network, ported from ../posetail-next (train.py + posetail/muon.py).

`torch.optim.Muon` orthogonalizes the momentum of 2D hidden matrices; it RAISES on any non-2D
parameter, so it must be paired with an AdamW-family optimizer for everything else (biases, norms,
heads, embeddings). The library ships `DualOptimizer`, which presents the pair as one optimizer to
the training loop. This module subclasses it for the three things THIS repo's checkpointing needs
that the reference (Lightning fabric + a different save path) did not:

  * `has_averaged_iterate` -- whether BOTH halves carry a schedule-free averaged `x`, so
    `save_checkpoint` knows whether `model_state_eval` is a real averaged weight.
  * a `load_state_dict` that restores `ScheduleFreeWrapper.train_mode` -- a real resume bug (see
    `PoseDualOptimizer.load_state_dict`).
  * a named refusal when a run folder's optimizer state does not match the optimizer this config
    builds -- the cost of making `muon` the default an absent key resolves to.

No DDP: `batch_size` is structurally 1 here and there is no world-size LR scaling, so the
reference's `sqrt(world_size)` / `total_to_per_gpu` machinery is deliberately not ported.
"""
from __future__ import annotations

from pathlib import Path

import torch

from posetail.posetail.muon import DualOptimizer

# The one place [training.optimizer] keys are enumerated. `scripts/train.py` guards this block
# against a typo the same way it guards [data] and `build_model` guards [model]: an unknown key
# would otherwise train at the default and the run folder would record the key nobody read -- an
# arm silently reporting its own control (eval rule 4). `muon_lr` for `muon_lr_scale` is exactly
# the class of mistake this catches.
KNOWN_OPTIMIZER_KEYS = frozenset({
    'optimizer',
    'learning_rate', 'kpt_lr', 'encoder_lr_scale', 'weight_decay', 'warmup_steps',
    'beta1', 'beta2',
    'muon_schedulefree', 'muon_lr_scale', 'muon_momentum', 'muon_warmup_steps',
    'muon_adjust_lr_fn',
})


class PoseDualOptimizer(DualOptimizer):
    """`DualOptimizer` with the three hooks this repo's `save_checkpoint`/resume path needs."""

    @property
    def adamw_params(self):
        """The AdamW-routed params -- heads, embeddings, norms, biases. The ONLY half that should
        be gradient-clipped: Muon orthogonalizes its own grads inside `step()`, so their raw norm
        is not a step size (report 34b)."""
        return [p for g in self.opt_adam.param_groups for p in g['params']]

    @property
    def muon_params(self):
        """The Muon-routed 2D matrices. `opt_muon.param_groups` delegates through the
        `ScheduleFreeWrapper` to the base Muon under `muon_schedulefree = true`."""
        return [p for g in self.opt_muon.param_groups for p in g['params']]

    @property
    def has_averaged_iterate(self) -> bool:
        """True iff BOTH inner optimizers maintain a schedule-free averaged iterate.

        Under `muon_schedulefree = true` the Muon half is a `ScheduleFreeWrapper` and the rest is
        `AdamWScheduleFree`; both expose `eval()`/`train()` and both hold an `x`. Under
        `muon_schedulefree = false` the Muon half is a bare `torch.optim.Muon` with no averaged
        iterate, so `model_state_eval` would be only half-averaged -- this reports that, and
        `save_checkpoint` then writes no eval weight (and `load_run` falls back with its message).
        """
        return all(hasattr(o, 'eval') and hasattr(o, 'train') for o in self._opts)

    def load_state_dict(self, sd):
        """Delegate, then force `ScheduleFreeWrapper.train_mode = True`.

        THIS FIXES A REAL RESUME BUG. `AdamWScheduleFree` keeps `train_mode` inside its
        `param_groups`, so it round-trips through `state_dict`. `ScheduleFreeWrapper` keeps it as a
        plain attribute that does NOT, so after `load_state_dict` it is back at its constructed
        `False`. `save_checkpoint` writes `model_state` at the **y** iterate (train mode), and the
        training loop then calls `opt.train()` (train.py) -- which, believing the params are at the
        averaged **x**, lerps y toward z a SECOND time. A resumed Muon run would silently restart
        from a corrupted weight. Restored explicitly here so the resume is exact.
        """
        from schedulefree import ScheduleFreeWrapper
        _refuse_state_shape(self, sd)
        super().load_state_dict(sd)
        for o in self._opts:
            if isinstance(o, ScheduleFreeWrapper):
                o.train_mode = True


def build_muon(model, fresh: set[str], cfg: dict) -> PoseDualOptimizer:
    """Route the model's parameters into Muon (2D transformer matrices) + AdamW-SF (the rest).

    The routing is by NAME and MODULE TYPE, not by `ndim == 2` alone, because three kinds of 2D
    tensor must not reach Muon:

      * `nn.Embedding` weights are `(rows, dim)` and 2D, but each row is a keypoint identity (or a
        time index). Orthogonalizing mixes rows that point at different body parts -- the same
        hazard `warm_start`'s row-copy refusal exists for. Every `nn.Embedding` param is collected
        by `id` and excluded (the technique the reference uses for its scene ids).
      * the output heads (`decoder.heads_*`) stay on AdamW, as in the reference; only the
        transformer MLPs (`decoder.mlps`) and attention projections go to Muon.
      * frozen params are filtered first, so `freeze_encoder = true` leaves the encoder Muon group
        empty rather than handing Muon a frozen tensor.

    `torch.optim.Muon` raises at construction on any non-2D param, so a routing slip is loud on
    the run's first line rather than a silent wrong-optimizer arm.
    """
    from torch.optim import Muon as TorchMuon
    from schedulefree import AdamWScheduleFree, ScheduleFreeWrapper

    lr = float(cfg['learning_rate'])
    kpt_lr = float(cfg.get('kpt_lr', lr))
    enc_scale = float(cfg.get('encoder_lr_scale', 1.0))
    wd = float(cfg.get('weight_decay', 0.0))
    muon_scale = float(cfg.get('muon_lr_scale', 1.0))
    momentum = float(cfg.get('muon_momentum', 0.95))
    adj = str(cfg.get('muon_adjust_lr_fn', 'match_rms_adamw'))
    sf = bool(cfg.get('muon_schedulefree', True))
    warmup = int(cfg.get('warmup_steps', 0))
    muon_warmup = int(cfg.get('muon_warmup_steps', 0))
    betas = (float(cfg.get('beta1', 0.9)), float(cfg.get('beta2', 0.95)))

    embed_ids = {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding)
                 for p in m.parameters()}
    dec_substr = ('decoder.cross_attns', 'decoder.mlps', 'decoder.camera_attns',
                  'decoder.temporal_attns')
    muon_dec, muon_enc, muon_fresh, adamw_base, adamw_enc = [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is2d = (p.ndim == 2 and name.endswith('.weight') and id(p) not in embed_ids)
        is_fresh = name in fresh or name.startswith('query_encoder.kpt_')
        if is2d and is_fresh:
            muon_fresh.append(p)                       # NEW vs the reference: fresh 2D -> Muon@kpt_lr
        elif is2d and 'scene_encoder.encoder.blocks' in name:
            muon_enc.append(p)
        elif is2d and ('scene_encoder.kv_proj' in name or any(s in name for s in dec_substr)):
            muon_dec.append(p)
        elif name.startswith('scene_encoder.encoder.'):
            adamw_enc.append(p)
        else:
            adamw_base.append(p)

    muon_groups = [{'params': muon_dec, 'lr': lr * muon_scale, 'weight_decay': wd}]
    if muon_fresh:
        muon_groups.append({'params': muon_fresh, 'lr': kpt_lr, 'weight_decay': wd})
    if muon_enc:
        muon_groups.append({'params': muon_enc, 'lr': lr * enc_scale, 'weight_decay': wd})
    adamw_groups = [{'params': adamw_base, 'lr': lr, 'weight_decay': wd}]
    if adamw_enc:
        adamw_groups.append({'params': adamw_enc, 'lr': lr * enc_scale, 'weight_decay': wd})

    opt_adam = AdamWScheduleFree([g for g in adamw_groups if g['params']], lr=lr, weight_decay=wd,
                                 warmup_steps=warmup, betas=betas)
    if sf:
        # ScheduleFreeWrapper is NOT an Optimizer subclass, so it wraps a constructed Muon; the
        # decoupled weight decay is applied at the y point by the wrapper (weight_decay_at_y), the
        # per-group value drives Muon's own decoupled decay -- mirroring the reference's arm.
        base_muon = TorchMuon([g for g in muon_groups if g['params']], lr=lr, weight_decay=0.0,
                              momentum=momentum, adjust_lr_fn=adj)
        opt_muon = ScheduleFreeWrapper(base_muon, momentum=0.9, weight_decay_at_y=wd)
    else:
        opt_muon = TorchMuon([g for g in muon_groups if g['params']], lr=lr, weight_decay=wd,
                             momentum=momentum, adjust_lr_fn=adj)

    print(f'optimizer: muon (schedulefree={sf}, adjust_lr_fn={adj!r}) | '
          f'muon_dec {len(muon_dec)} @ {lr * muon_scale:g}, '
          f'muon_fresh {len(muon_fresh)} @ {kpt_lr:g}, '
          f'muon_enc {len(muon_enc)} @ {lr * enc_scale:g} | '
          f'adamw_base {len(adamw_base)} @ {lr:g}, adamw_enc {len(adamw_enc)} @ {lr * enc_scale:g}')
    return PoseDualOptimizer(opt_muon, opt_adam, muon_warmup_steps=muon_warmup)


def _is_dual_state(state) -> bool:
    """A `PoseDualOptimizer.state_dict()` is `{'muon': ..., 'adam': ...}`; an AdamW-SF one is not."""
    return isinstance(state, dict) and 'muon' in state and 'adam' in state


def _refuse_state_shape(opt, state) -> None:
    """Raise if the state dict's shape does not match the optimizer kind. Path-free fallback."""
    if isinstance(opt, PoseDualOptimizer) and not _is_dual_state(state):
        raise SystemExit(
            'optimizer state is AdamW-schedule-free but this config builds a Muon optimizer. '
            'Set optimizer = "schedulefree" in the config to resume, or --no-resume to start over.')
    if not isinstance(opt, PoseDualOptimizer) and _is_dual_state(state):
        raise SystemExit(
            'optimizer state is Muon but this config builds an AdamW-schedule-free optimizer. '
            'Set optimizer = "muon" in the config to resume, or --no-resume to start over.')


def refuse_mismatched_optimizer_state(opt, state, path, resolved: str, explicit: bool) -> None:
    """Refuse to load an optimizer state that was written by a DIFFERENT optimizer, by name.

    This is the `gridresid_offset` rule applied to optimizer state: the MODEL tensors load either
    way, so nothing but an explicit check on the state's own shape can tell you the run is being
    resumed under the wrong optimizer. It fires in BOTH directions -- an AdamW-SF checkpoint under
    a Muon config (the case the new default creates for every old run folder) and a Muon checkpoint
    under `optimizer = "schedulefree"`.
    """
    dual_opt = isinstance(opt, PoseDualOptimizer)
    dual_state = _is_dual_state(state)
    if dual_opt == dual_state:
        return
    path = Path(path)
    held = 'Muon' if dual_state else 'AdamW-schedule-free'
    builds = 'a Muon' if dual_opt else 'an AdamW-schedule-free'
    why_default = (' (`[training.optimizer].optimizer` is absent, which now means "muon")'
                   if dual_opt and not explicit else f' (optimizer = {resolved!r})')
    fix = 'schedulefree' if dual_opt else 'muon'
    raise SystemExit(
        f'{path} holds {held} optimizer state, but this config builds {builds} optimizer'
        f'{why_default}. The model tensors load either way, so nothing else can tell you. Set '
        f'optimizer = "{fix}" in the config to resume this run as it was trained, or --no-resume '
        f'to start over (which OVERWRITES both checkpoints).')
