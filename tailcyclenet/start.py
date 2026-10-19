"""Where a pose training run starts: fresh, warm-started, or resumed -- decided once, up front.

`plan_start` reads the run folder and `[training].checkpoint_path` and returns a `StartPlan`
without building anything (no Fabric, no model). `build_start` then builds the model and
optimizer the plan names. A resume is decided BEFORE the staged unfreeze is replayed, so a
checkpoint that turns out not to resume falls back to a clean warm start (a fresh model, a fresh
optimizer, the configured unfreeze still ahead) rather than to a half-replayed one.

The rules, in order:
- the run folder's own `checkpoints/checkpoint_last.pth` (unless `--no-resume`) RESUMES, or is
  refused by name: a resume that cannot continue the run it continues is an error, never a
  silent restart. That includes a different world size (the rates are sqrt(world)-scaled and
  the iteration-to-sample mapping changes); a checkpoint that records no world size predates
  the record and is allowed, with a note.
- `[training].checkpoint_path` (a `.pth`, a run folder, its `checkpoints/`, or a Hugging Face
  repo id) RESUMES when it is a full pose training state whose optimizer kind, layout metadata
  and world size match; anything else WARM-STARTS from its weights at iteration 0, saying why.
- otherwise the run starts fresh.
The keypoint registry a run appends to is the checkpoint's own (embedded, else the run folder it
sits in), so identity rows keep their body parts; a keypoint table with no registry is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from .checkpoints import (enclosing_run, full_training_state, is_hf_repo_id, prior_provenance,
                          resolve_checkpoint, resolve_hf_checkpoint, warm_start)
from .format import Registry
from .optim import OPTIMIZER_LAYOUT_SCHEMA, build_optimizer, check_resume, state_kind
from .unfreeze import apply_norms_extension, apply_staged_unfreeze

MODES = ('fresh', 'warm', 'resume_own', 'resume_ckpt')

_OVERWRITES = '--no-resume to start over (which OVERWRITES both checkpoints)'


@dataclass(frozen=True)
class StartPlan:
    """How a run starts. `path` is the checkpoint resumed or warm-started from; `fresh` and
    `layout` are the resumed checkpoint's recorded fresh-parameter names and optimizer metadata;
    `registry` is the keypoint registry to append to; `state` the loaded checkpoint (resume only).
    """

    mode: str
    path: Path | None = None
    start_it: int = 0
    fresh: frozenset = frozenset()
    registry: Registry | None = None
    reason: str = ''
    layout: dict | None = field(default=None, compare=False, repr=False)
    state: dict | None = field(default=None, compare=False, repr=False)

    @property
    def resumes(self) -> bool:
        """Whether the plan continues a checkpoint's optimizer and iteration."""
        return self.mode in ('resume_own', 'resume_ckpt')

    def as_warm(self, reason: str) -> 'StartPlan':
        """The same checkpoint as a warm start: its weights, iteration 0, a fresh optimizer."""
        return replace(self, mode='warm', start_it=0, fresh=frozenset(), layout=None, state=None,
                       reason=reason)


def load_checkpoint(path: Path) -> dict:
    """A checkpoint on the CPU, memory-mapped where the file format allows it (a training
    checkpoint is several GB and every rank reads it)."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:
        return torch.load(path, map_location='cpu', weights_only=False)


def resolve_start_checkpoint(ref, revision: str | None = None) -> Path:
    """`[training].checkpoint_path` as a file: a `.pth`, the latest checkpoint of a run folder
    (or of its `checkpoints/`), or a Hugging Face repo id's `model.pth`."""
    if is_hf_repo_id(ref):
        return resolve_hf_checkpoint(ref, revision=revision)
    ref = Path(ref)
    return ref if ref.is_file() else resolve_checkpoint(ref)


def checkpoint_registry(path: Path, ck: dict) -> Registry | None:
    """The keypoint registry a checkpoint's identity rows are keyed by: embedded in the file,
    else the run folder it sits in, else None."""
    doc = ck.get('keypoint_registry') if isinstance(ck, dict) else None
    if isinstance(doc, dict):
        return Registry.from_dict(doc)
    run = enclosing_run(Path(path))
    if run is not None and (run / 'keypoint_registry.toml').is_file():
        return Registry.load(run / 'keypoint_registry.toml')
    return None


def _has_keypoint_table(ck: dict) -> bool:
    """Whether a checkpoint's weights carry a per-keypoint identity table."""
    state = ck.get('model_state_eval') or ck.get('model_state') if isinstance(ck, dict) else None
    return isinstance(state, dict) and any(k.endswith('kpt_embed.weight') for k in state)


def _refusal(ck: dict, opt_kind: str, world: int) -> tuple[str | None, int | None]:
    """(why `ck` cannot resume under this config and world size, or None; its recorded world
    size, or None when it records none)."""
    if not full_training_state(ck):
        return 'it holds weights only (no optimizer_state and iteration)', None
    if ck.get('kind', 'pose') != 'pose':
        return f'it is a {ck.get("kind")!r} checkpoint, not a pose one', None
    held = state_kind(ck['optimizer_state'])
    if held != opt_kind:
        return (f'it holds {held} optimizer state and this config builds {opt_kind} '
                f'([training.optimizer].optimizer = "{held}" resumes it as it was trained)'), None
    meta = ck.get('optimizer_metadata')
    if (not isinstance(meta, dict) or meta.get('schema') != OPTIMIZER_LAYOUT_SCHEMA
            or not isinstance(meta.get('fresh_names'), list)):
        return (f'it predates {OPTIMIZER_LAYOUT_SCHEMA} metadata; v0.3.4 is the last release '
                'that resumes it'), None
    saved = meta.get('world_size')
    if saved is not None and int(saved) != int(world):
        return (f'it was trained on {int(saved)} rank(s) and this launch has {world}: the '
                'learning rates are scaled by sqrt(world) and the iteration-to-sample mapping '
                'changes, so the continued run would not be the run it continues'), int(saved)
    return None, None if saved is None else int(saved)


def _resume_plan(mode: str, path: Path, ck: dict, registry: Registry | None) -> StartPlan:
    """A resume of `ck`, carrying its recorded fresh names and layout."""
    meta = ck['optimizer_metadata']
    return StartPlan(mode, Path(path), int(ck['iteration']), frozenset(meta['fresh_names']),
                     registry, layout=meta, state=ck)


def plan_start(run: Path, checkpoint_path=None, *, revision: str | None = None,
               no_resume: bool = False, no_warm_start: bool = False, opt_kind: str = 'muon',
               world: int = 1, load=load_checkpoint) -> StartPlan:
    """Decide how the run in `run` starts (module docstring). Builds nothing; reads the
    checkpoint(s) with `load`, and downloads a Hugging Face base only when no own-folder resume
    takes precedence. Raises `SystemExit` for an own-folder checkpoint that cannot resume, and
    for a warm-start checkpoint whose keypoint table has no registry.
    """
    run = Path(run)
    own = run / 'checkpoints' / 'checkpoint_last.pth'
    if own.exists() and not no_resume:
        ck = load(own)
        why, saved_world = _refusal(ck, str(opt_kind), world)
        if why is not None:
            hint = 'Restore the configuration it was trained with, or'
            if saved_world is not None:
                devices = str(prior_provenance(run).get('devices', saved_world))
                devices = str(saved_world) if devices == '-1' else devices
                hint = f'Relaunch with --devices {devices} (how it was trained), or'
            raise SystemExit(f'{own}: this run folder does not resume -- {why}. {hint} '
                             f'{_OVERWRITES}.')
        if saved_world is None:
            print(f'{own}: records no world size (written before it was stored), so resuming '
                  f'on {world} rank(s) is not checked against the run it continues')
        registry = checkpoint_registry(own, ck)
        if registry is None and (run / 'keypoint_registry.toml').is_file():
            registry = Registry.load(run / 'keypoint_registry.toml')
        return _resume_plan('resume_own', own, ck, registry)
    if no_warm_start or not checkpoint_path:
        return StartPlan('fresh')
    path = resolve_start_checkpoint(checkpoint_path, revision)
    ck = load(path)
    registry = checkpoint_registry(path, ck)
    if registry is None and _has_keypoint_table(ck):
        raise SystemExit(
            f'checkpoint_path: {path} carries a keypoint identity table but no keypoint registry '
            '(none embedded, and no keypoint_registry.toml in a run folder around it), so which '
            'body part each row belongs to cannot be read back. Point at its run folder, or a '
            'checkpoint written with its registry.')
    why = '--no-resume' if no_resume else _refusal(ck, str(opt_kind), world)[0]
    if why is None:
        return _resume_plan('resume_ckpt', path, ck, registry)
    return StartPlan('warm', Path(path), registry=registry, reason=why)


@dataclass
class Started:
    """What `build_start` built: the model (on its device), its optimizer, the fresh-parameter
    names the optimizer was routed with, and the plan it carried out (`state` dropped)."""

    model: torch.nn.Module
    optimizer: object
    fresh: set
    plan: StartPlan


def _describe_encoder(model, model_cfg: dict) -> None:
    """Apply the norms extension to a build-time-trainable encoder and say what trains when."""
    enc_grad = model_cfg.get('video_encoder_requires_grad', False)
    n_last = model_cfg.get('video_encoder_finetune_last_n_layers')
    blocks = n_last if n_last is not None else 'all'

    def n_frozen():
        """Frozen tensors inside the video encoder."""
        return sum(1 for n, p in model.named_parameters()
                   if n.startswith('scene_encoder.encoder.') and not p.requires_grad)

    if not isinstance(enc_grad, bool):
        print(f'encoder: frozen now, last {blocks} block(s) unfreeze at iteration '
              f'{int(enc_grad)} ({n_frozen()} frozen tensors)')
        return
    ext = apply_norms_extension(model)
    if not enc_grad:
        print(f'encoder: frozen for the whole run ({n_frozen()} frozen tensors)')
    else:
        print(f'encoder: trainable from step 0, last {blocks} block(s)'
              f'{f" + norms {ext}" if ext else ""} ({n_frozen()} frozen tensors)')


def build_start(plan: StartPlan, make_model, opt_cfg: dict, model_cfg: dict | None = None,
                device='cpu') -> Started:
    """Build the model and optimizer `plan` names.

    `make_model(pretrained)` returns a new model; `pretrained` is False whenever a checkpoint
    will overwrite the backbone (skipping the VJEPA2 download). `opt_cfg` is the world-scaled
    optimizer config, the ONE dict the build, the resume replay and the live unfreeze all read.
    A resume loads the raw weights, replays the staged unfreeze to the checkpoint's iteration,
    then asks `check_resume`; a refusal is a `SystemExit` for the run's own folder and a clean
    warm start (a NEW model) for a `checkpoint_path`.
    """
    model = make_model(plan.mode == 'fresh')
    fresh = set(plan.fresh)
    if plan.resumes:
        try:
            model.load_state_dict(plan.state['model_state'])
        except RuntimeError as e:
            why = (f'its weights do not load into this model ({type(e).__name__}: '
                   f'{str(e).splitlines()[0]})')
            return _refused(plan, why, make_model, opt_cfg, model_cfg, device)
    elif plan.mode == 'warm':
        fresh = warm_start(model, plan.path,
                           base_names=plan.registry.names if plan.registry else None)
    _describe_encoder(model, model_cfg or {})
    model = model.to(device)
    opt = build_optimizer(model, fresh, opt_cfg)
    if plan.resumes:
        info = apply_staged_unfreeze(model, opt, opt_cfg, plan.start_it, fresh=fresh)
        verdict = check_resume(opt, plan.state['optimizer_state'], model, plan.layout)
        if not verdict:
            return _refused(plan, verdict.reason, make_model, opt_cfg, model_cfg, device)
        if info:
            print(f'resume: replayed the encoder unfreeze (blocks {info["blocks"]}, '
                  f'norms {info["norms"]}, {info["n_params"]:,} params)')
        opt.load_state_dict(plan.state['optimizer_state'])
        print(f'resuming {plan.path} at iteration {plan.start_it} ({_OVERWRITES})')
    return Started(model, opt, fresh, replace(plan, state=None))


def _refused(plan, why, make_model, opt_cfg, model_cfg, device) -> Started:
    """A resume `build_start` could not carry out: an error for the run's own folder, a warm
    start from the same checkpoint otherwise."""
    if plan.mode == 'resume_own':
        raise SystemExit(f'{plan.path}: this run folder does not resume -- {why}. Restore the '
                         f'configuration it was trained with, or {_OVERWRITES}.')
    print(f'checkpoint_path: {plan.path} does not resume -- {why}; warm-starting from its '
          'weights at iteration 0')
    return build_start(plan.as_warm(why), make_model, opt_cfg, model_cfg, device)
