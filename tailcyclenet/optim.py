"""SF-Muon for the pose network, ported from ../posetail-next.

Muon orthogonalizes 2D hidden matrices and RAISES on non-2D params, so it is paired with
AdamW-SF for the rest. This module adds what this repo's checkpointing needs: `has_averaged_iterate`, a `load_state_dict` that restores `train_mode` (a resume bug), and a named refusal on optimizer-state mismatch. World-size scaling lives in `tailcyclenet.distributed`.
"""
from __future__ import annotations

from pathlib import Path

import torch

from posetail.posetail.muon import DualOptimizer

# The one place [training.optimizer] keys are enumerated; an unknown key would otherwise train
# at the default and the run folder would record the key nobody read.
KNOWN_OPTIMIZER_KEYS = frozenset({
    'optimizer',
    'learning_rate', 'kpt_lr', 'encoder_lr_scale', 'weight_decay', 'warmup_steps',
    'beta1', 'beta2',
    'muon_schedulefree', 'muon_lr_scale', 'muon_momentum', 'muon_warmup_steps',
    'muon_adjust_lr_fn',
})

# The five routes a parameter can take. Order is the order the groups are BUILT in, which is the
# order `load_state_dict` matches them by -- so a resumed run must reconstruct it exactly.
ROUTES = ('muon_dec', 'muon_fresh', 'muon_enc', 'adamw_base', 'adamw_enc')

_DEC_SUBSTR = ('decoder.cross_attns', 'decoder.mlps', 'decoder.camera_attns',
               'decoder.temporal_attns')


def embedding_ids(model) -> set[int]:
    """`id()` of every `nn.Embedding` param. Each row is a keypoint (or time) identity, so
    orthogonalizing the table mixes body parts.
    """
    return {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding)
            for p in m.parameters()}


def route_param(name: str, p, fresh: set[str], embed_ids: set[int]) -> str:
    """Which of `ROUTES` this parameter belongs to -- the single routing rule, shared by build
    time and the staged unfreeze so the two cannot drift.
    """
    is2d = (p.ndim == 2 and name.endswith('.weight') and id(p) not in embed_ids)
    is_fresh = name in fresh or name.startswith('query_encoder.kpt_')
    if is2d and is_fresh:
        # NEW vs the reference: fresh 2D -> Muon @ kpt_lr.
        return 'muon_fresh'
    if is2d and 'scene_encoder.encoder.blocks' in name:
        return 'muon_enc'
    if is2d and ('scene_encoder.kv_proj' in name or any(s in name for s in _DEC_SUBSTR)):
        return 'muon_dec'
    if name.startswith('scene_encoder.encoder.'):
        return 'adamw_enc'
    return 'adamw_base'


class PoseDualOptimizer(DualOptimizer):
    """`DualOptimizer` with the hooks this repo's `save_checkpoint`/resume/unfreeze paths need."""

    def add_muon_group(self, params, lr: float, weight_decay: float) -> None:
        """Add a NEW Muon group at unfreeze time. See `add_adamw_group` for why it must be new.

        Muon validates ndim == 2 only in `__init__`, so the shape is asserted here;
        `_muon_base_lrs` must grow with the group or `zip` silently skips its warmup rescale.
        """
        params = list(params)
        if not params:
            return
        for p in params:
            assert p.ndim == 2, (
                f'Muon only accepts 2D parameters; got {tuple(p.shape)}. add_param_group does not '
                're-validate, so this would surface inside step() instead.')
        self.opt_muon.add_param_group({'params': params, 'lr': lr, 'weight_decay': weight_decay})
        self._muon_base_lrs.append(lr)

    def add_adamw_group(self, params, lr: float, weight_decay: float) -> None:
        """Add a NEW AdamW-SF group at unfreeze time, in phase with the existing ones.

        NEW, not pre-registered: schedule-free advances `k` every step, so a grad-less group
        would fold its encoder into the averaged iterate at ~1/k -- the deployed weight would
        hold a barely-moved encoder. `train_mode` is copied from group 0 (read per group).
        """
        params = list(params)
        if not params:
            return
        self.opt_adam.add_param_group({'params': params, 'lr': lr, 'weight_decay': weight_decay})
        self.opt_adam.param_groups[-1]['train_mode'] = self.opt_adam.param_groups[0]['train_mode']

    @property
    def adamw_params(self):
        """The AdamW-routed params -- heads, embeddings, norms, biases. The ONLY half that should
        be gradient-clipped: Muon orthogonalizes its own grads inside `step()`.
        """
        return [p for g in self.opt_adam.param_groups for p in g['params']]

    @property
    def muon_params(self):
        """The Muon-routed 2D matrices. `opt_muon.param_groups` delegates through the
        `ScheduleFreeWrapper` to the base Muon under `muon_schedulefree = true`."""
        return [p for g in self.opt_muon.param_groups for p in g['params']]

    @property
    def has_averaged_iterate(self) -> bool:
        """True iff BOTH inner optimizers maintain a schedule-free averaged iterate. Under
        `muon_schedulefree = false` the Muon half is bare `torch.optim.Muon`, so
        `model_state_eval` would be only half-averaged -- `save_checkpoint` then writes no eval
        weight and `load_run` falls back.
        """
        return all(hasattr(o, 'eval') and hasattr(o, 'train') for o in self._opts)

    def load_state_dict(self, sd):
        """Delegate, then force `ScheduleFreeWrapper.train_mode = True`.

        The wrapper keeps `train_mode` as an attribute that does NOT round-trip, so after load
        the params sit at the averaged x while the loop calls `opt.train()` -- lerping y toward z
        a second time.
        """
        from schedulefree import ScheduleFreeWrapper
        _refuse_state_shape(self, sd)
        refuse_group_count_mismatch(self, sd)
        super().load_state_dict(sd)
        for o in self._opts:
            if isinstance(o, ScheduleFreeWrapper):
                o.train_mode = True


def build_muon(model, fresh: set[str], cfg: dict) -> PoseDualOptimizer:
    """Route the model's parameters into Muon (2D transformer matrices) + AdamW-SF (the rest).

    Routing is by NAME and MODULE TYPE, not ndim alone: `nn.Embedding` weights are 2D but each
    row is a keypoint identity; the output heads stay on AdamW; frozen params are filtered first
    (a frozen param is in no group, which is why the staged unfreeze must add new groups).
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

    embed_ids = embedding_ids(model)
    groups = {k: [] for k in ROUTES}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups[route_param(name, p, fresh, embed_ids)].append(p)
    muon_dec, muon_enc, muon_fresh = (groups['muon_dec'], groups['muon_enc'],
                                      groups['muon_fresh'])
    adamw_base, adamw_enc = groups['adamw_base'], groups['adamw_enc']

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
        # The wrapper applies decoupled weight decay at the y point; the per-group value drives
        # Muon's own decay -- mirroring the reference's arm.
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


def group_lr(route: str, cfg: dict) -> float:
    """The LR a route's group is built at. One place, so the unfreeze cannot pick a different one."""
    lr = float(cfg['learning_rate'])
    if route == 'muon_fresh':
        return float(cfg.get('kpt_lr', lr))
    if route in ('muon_enc', 'adamw_enc'):
        return lr * float(cfg.get('encoder_lr_scale', 1.0))
    if route == 'muon_dec':
        return lr * float(cfg.get('muon_lr_scale', 1.0))
    return lr


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


def _group_counts(opt) -> tuple[int, int]:
    """The (Muon, AdamW) param-group counts of the CURRENT optimizer."""
    if isinstance(opt, PoseDualOptimizer):
        return len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups)
    return 0, len(opt.param_groups)


def _state_group_counts(state) -> tuple[int, int]:
    """The (Muon, AdamW) param-group counts recorded in a saved state dict."""
    if _is_dual_state(state):
        return len(state['muon']['param_groups']), len(state['adam']['param_groups'])
    return 0, len(state['param_groups'])


def refuse_group_count_mismatch(opt, state) -> None:
    """Raise a NAMED error when the saved state has a different number of param groups. The
    staged unfreeze ADDS groups mid-run, so a resumed run must replay the unfreeze before
    loading -- torch's own message would not name the config key that caused it.
    """
    have, want = _group_counts(opt), _state_group_counts(state)
    if have == want:
        return
    raise SystemExit(
        f'optimizer state holds {want[0]} Muon + {want[1]} AdamW param group(s) but this run built '
        f'{have[0]} + {have[1]}. The group layout changes when the staged encoder unfreeze fires, '
        f'so a resumed run must reach the same layout as the run it continues. Check that '
        f'[model].video_encoder_requires_grad and [model].video_encoder_finetune_last_n_layers '
        f'still hold the values this run folder was trained with, or --no-resume to start over '
        f'(which OVERWRITES both checkpoints).')


def refuse_mismatched_optimizer_state(opt, state, path, resolved: str, explicit: bool) -> None:
    """Refuse to load an optimizer state written by a DIFFERENT optimizer, by name. The model
    tensors load either way, so only an explicit shape check can tell you the resume is wrong.
    Fires in both directions.
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
