"""The pose optimizers: SF-Muon (the default) and the AdamW-SF escape hatch, plus the resume check.

Muon orthogonalizes 2D hidden matrices and RAISES on non-2D params, so it is paired with AdamW-SF
for the rest. Beyond upstream: `has_averaged_iterate`, a `load_state_dict` that restores
`train_mode`, name-keyed layout metadata, and `check_resume`, the one verdict on whether a saved
optimizer state belongs to the optimizer this config builds. World-size scaling lives in
`tailcyclenet.distributed`.
"""
from __future__ import annotations

import torch

from tracktail.model.muon import DualOptimizer

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

# Checkpoint metadata is deliberately versioned: optimizer state matches groups by position,
# so a name-level layout is needed before state loading can be trusted.
OPTIMIZER_LAYOUT_SCHEMA = 'tailcyclenet.optimizer-layout.v1'
DUAL_STATE_SCHEMA = 'tailcyclenet.pose-dual.v1'

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
    time and the staged unfreeze so the two cannot drift. NEW vs the reference: fresh 2D
    matrices go to Muon at `kpt_lr`.
    """
    is2d = (p.ndim == 2 and name.endswith('.weight') and id(p) not in embed_ids)
    is_fresh = name in fresh or name.startswith('query_encoder.kpt_')
    if is2d and is_fresh:
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

    def state_dict(self):
        """Persist Muon's scheduler state, which the upstream DualOptimizer omits."""
        state = super().state_dict()
        state['_tailcyclenet'] = {
            'schema': DUAL_STATE_SCHEMA,
            'muon_warmup_steps': int(self.muon_warmup_steps),
            'gstep': int(self._gstep),
            'muon_base_lrs': [float(lr) for lr in self._muon_base_lrs],
        }
        return state

    def load_state_dict(self, sd):
        """Restore upstream state and the Muon warmup step without mutating ``sd``, after
        `check_resume` has accepted it (a refusal is a `SystemExit` naming why)."""
        from schedulefree import ScheduleFreeWrapper
        verdict = check_resume(self, sd)
        if not verdict:
            raise SystemExit(f'optimizer state does not resume: {verdict.reason}')
        super().load_state_dict({k: v for k, v in sd.items() if k != '_tailcyclenet'})
        self._gstep = int(sd['_tailcyclenet'].get('gstep', 0))
        for o in self._opts:
            if isinstance(o, ScheduleFreeWrapper):
                o.train_mode = True


def build_muon(model, fresh: set[str], cfg: dict) -> PoseDualOptimizer:
    """Route the model's parameters into Muon (2D transformer matrices) + AdamW-SF (the rest).

    Routing is by NAME and MODULE TYPE, not ndim alone: `nn.Embedding` weights are 2D but each
    row is a keypoint identity; the output heads stay on AdamW; frozen params are filtered first
    (a frozen param is in no group, which is why the staged unfreeze must add new groups). The
    schedule-free wrapper applies decoupled weight decay at the y point; the per-group value
    drives Muon's own decay -- mirroring the reference's arm.
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


def optimizer_metadata(model, optimizer, fresh_names: set[str] | None = None,
                       world_size: int | None = None) -> dict:
    """Stable, name-based optimizer layout metadata for a pose checkpoint, with the world size
    whose sqrt-scaled rates the groups carry (None: not recorded)."""
    names = {id(p): name for name, p in model.named_parameters()}

    def groups_doc(groups, lrs=None):
        """Describe parameter groups by stable names and effective rates."""
        return [{'lr': float(group['lr'] if lrs is None else lrs[i]),
                 'names': [names[id(p)] for p in group['params']]}
                for i, group in enumerate(groups)]

    if isinstance(optimizer, PoseDualOptimizer):
        groups = {'muon': groups_doc(optimizer.opt_muon.param_groups, optimizer._muon_base_lrs),
                  'adamw': groups_doc(optimizer.opt_adam.param_groups)}
        kind = 'muon'
    else:
        groups = {'muon': [], 'adamw': groups_doc(optimizer.param_groups)}
        kind = 'schedulefree'
    out = {'schema': OPTIMIZER_LAYOUT_SCHEMA, 'optimizer': kind,
           'fresh_names': sorted(str(n) for n in (fresh_names or set())),
           'groups': groups}
    if world_size is not None:
        out['world_size'] = int(world_size)
    return out


def build_optimizer(model, fresh: set[str], cfg: dict):
    """The run's optimizer, selected by `[training.optimizer].optimizer`; an absent key is "muon".

    `"schedulefree"` is the AdamW-SF recipe a resume needs. Fresh params (identity table,
    no-query tokens, fusion gate) get `kpt_lr`; a trainable encoder gets a lower one. A resume
    passes the checkpoint's own recorded `fresh_names` (see `start.plan_start`).
    """
    kind = str(cfg.get('optimizer', 'muon'))
    if kind == 'muon':
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


class Verdict:
    """Whether a saved optimizer state can be loaded into this run's optimizer; falsy with the
    reason when it cannot."""

    def __init__(self, ok: bool, reason: str = ''):
        """`ok` is the verdict, `reason` the human sentence naming what differs."""
        self.ok, self.reason = bool(ok), str(reason)

    def __bool__(self):
        """Truthy iff the state resumes."""
        return self.ok

    def __repr__(self):
        """`Verdict(ok)` or `Verdict(refused: reason)`."""
        return 'Verdict(ok)' if self.ok else f'Verdict(refused: {self.reason})'


def _is_dual_state(state) -> bool:
    """A `PoseDualOptimizer.state_dict()` is `{'muon': ..., 'adam': ...}`; an AdamW-SF one is not."""
    return isinstance(state, dict) and 'muon' in state and 'adam' in state


def state_kind(state) -> str:
    """The optimizer kind an `optimizer_state` was written by: `'muon'` or `'schedulefree'`."""
    return 'muon' if _is_dual_state(state) else 'schedulefree'


def _lrs(groups) -> list[float]:
    """Group learning rates in their stored order."""
    return [float(g['lr']) for g in groups]


def _named_layout(opt, model) -> tuple[str, dict]:
    """(kind, groups) of the CURRENT optimizer in `optimizer_metadata`'s name-keyed form."""
    names = {id(p): name for name, p in model.named_parameters()}

    def current(groups, lrs):
        """One `{lr, names}` entry per group."""
        return [{'lr': float(lr), 'names': [names.get(id(p)) for p in g['params']]}
                for g, lr in zip(groups, lrs)]

    if isinstance(opt, PoseDualOptimizer):
        return 'muon', {'muon': current(opt.opt_muon.param_groups, opt._muon_base_lrs),
                        'adamw': current(opt.opt_adam.param_groups,
                                         _lrs(opt.opt_adam.param_groups))}
    return 'schedulefree', {'muon': [], 'adamw': current(opt.param_groups, _lrs(opt.param_groups))}


def check_resume(opt, state, model=None, metadata: dict | None = None) -> Verdict:
    """Whether `state` (an `optimizer_state`) belongs to `opt`, the optimizer this config built --
    the one resume-compatibility check, replacing seven overlapping predicates.

    In order: the optimizer KIND (Muon's dual `{'muon', 'adam'}` state vs AdamW-SF's plain one);
    the param-group COUNT (the staged unfreeze adds groups, so a resume must replay it first);
    for Muon the `pose-dual.v1` schema, `muon_warmup_steps` and the base rates; the AdamW rates;
    and, given `model` and the checkpoint's `optimizer_metadata`, the name-keyed layout. Groups
    are matched BY POSITION on load, so a rate or membership difference is a refusal, not a
    warning. A state from before `pose-dual.v1` is refused: v0.3.4 is the last release that
    resumes it (a warm start from it still works).
    """
    if not isinstance(state, dict):
        return Verdict(False, 'the checkpoint holds no optimizer state')
    dual_opt, dual_state = isinstance(opt, PoseDualOptimizer), _is_dual_state(state)
    if dual_opt != dual_state:
        held = 'Muon' if dual_state else 'AdamW-schedule-free'
        builds = 'Muon' if dual_opt else 'AdamW-schedule-free'
        fix = 'muon' if dual_state else 'schedulefree'
        return Verdict(False, f'it holds {held} optimizer state but this config builds a {builds} '
                              f'optimizer; set [training.optimizer].optimizer = "{fix}" to resume '
                              'it as it was trained')
    have = ((len(opt.opt_muon.param_groups), len(opt.opt_adam.param_groups)) if dual_opt
            else (0, len(opt.param_groups)))
    want = ((len(state['muon']['param_groups']), len(state['adam']['param_groups'])) if dual_state
            else (0, len(state['param_groups'])))
    if have != want:
        return Verdict(False, (
            f'it holds {want[0]} Muon + {want[1]} AdamW param group(s) but this run built '
            f'{have[0]} + {have[1]}. The group layout changes when the staged encoder unfreeze '
            'fires, so check that [model].video_encoder_requires_grad and '
            '[model].video_encoder_finetune_last_n_layers still hold the values it was trained '
            'with'))
    if dual_opt:
        extra = state.get('_tailcyclenet')
        if not isinstance(extra, dict) or extra.get('schema') != DUAL_STATE_SCHEMA:
            return Verdict(False, f'its Muon state predates {DUAL_STATE_SCHEMA}; v0.3.4 is the '
                                  'last release that resumes it')
        if int(extra.get('muon_warmup_steps', -1)) != int(opt.muon_warmup_steps):
            return Verdict(False, f'muon_warmup_steps was {extra.get("muon_warmup_steps")}, this '
                                  f'config has {opt.muon_warmup_steps}')
        saved = [float(x) for x in extra.get('muon_base_lrs', [])]
        if saved != [float(x) for x in opt._muon_base_lrs]:
            return Verdict(False, f'the Muon group learning rates were {saved}, this run builds '
                                  f'{[float(x) for x in opt._muon_base_lrs]}')
        adam_have, adam_want = _lrs(opt.opt_adam.param_groups), _lrs(state['adam']['param_groups'])
    else:
        adam_have, adam_want = _lrs(opt.param_groups), _lrs(state['param_groups'])
    if adam_have != adam_want:
        return Verdict(False, f'the AdamW group learning rates were {adam_want}, this run builds '
                              f'{adam_have}')
    if metadata is not None and model is not None:
        if metadata.get('schema') != OPTIMIZER_LAYOUT_SCHEMA:
            return Verdict(False, f'its optimizer metadata is not {OPTIMIZER_LAYOUT_SCHEMA}')
        kind, groups = _named_layout(opt, model)
        if kind != metadata.get('optimizer') or groups != metadata.get('groups'):
            return Verdict(False, 'its named param-group layout (which parameter sits in which '
                                  'group, at which rate) differs from the one this run builds')
    return Verdict(True)
