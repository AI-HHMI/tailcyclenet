"""Training speedups (`[training.speed]`): scoped bf16 autocast, per-block torch.compile, and the
encoder's activation checkpointing.

A port of tracktail 0.6.2's `tracktail.model.speed` to this repo's model, which tracktail's own
helper does not cover: the v7 MoE carries a SECOND decoder (E2, `set_decoder.SetDecoder`, a copy
of the host's), and the staged unfreeze changes which encoder blocks need a backward mid-run.
With the section absent training runs as if it said (measured on L4, report 88):

    [training.speed]
    enabled = true
    autocast_bf16 = ["encoder", "decoder"]   # bf16 inside these parts, fp32 handed back
    compile = ["encoder"]                    # torch.compile per V-JEPA block
    encoder_activation_checkpointing = true  # trainable blocks only

`encoder_activation_checkpointing` checkpoints only a block that will have a backward (trainable
parameters or an input that requires grad): the 16 always-frozen blocks, and every block before
the staged unfreeze, build no graph, so checkpointing them only cost the Python wrapper (the
library flag checkpoints every block). The answer is re-read per call, so the unfreeze turns it on
for the blocks it releases. `false` keeps every activation (+~4 GB on social-dannce unfrozen).
`enabled = false` restores fp32 eager (the activation-checkpointing key still applies). Every
patch is on the module INSTANCE (`forward` attribute, `Module.compile`), so parameter names and
`state_dict` keys are unchanged and a checkpoint loads either way; inference never calls this and
keeps its own `set_scene_speed` levers. bf16 is used only on CUDA with bf16 support and compile
only on CUDA; otherwise those parts are skipped by name and training runs fp32 eager.

Parts: `encoder` = `scene_encoder.encoder` (the V-JEPA ViT, the dominant cost); `decoder` = the
camera / temporal / cross-attention and MLP blocks of E1's decoder AND E2's (their residual stream,
heads, losses and camera geometry stay fp32). tracktail measured bf16 on kv_proj / query / heads
as an accuracy cost (+11% train loss), so those are not offered.
"""
from __future__ import annotations

import functools

import torch
from torch.utils.checkpoint import checkpoint

AUTOCAST_PARTS = ('encoder', 'decoder')
COMPILE_PARTS = ('encoder',)
DEFAULTS = {
    'enabled': True,
    'autocast_bf16': ['encoder', 'decoder'],
    'compile': ['encoder'],
    'encoder_activation_checkpointing': True,
}
_OFF = {'autocast_bf16': [], 'compile': []}
RECOMPILE_LIMIT = 64
_DEC_BLOCKS = ('camera_attns', 'temporal_attns', 'cross_attns', 'mlps')


def speed_options(train_cfg: dict) -> dict:
    """The effective options: DEFAULTS overridden by `[training.speed]`; unknown keys or parts
    raise (a typo would train at the defaults and report as the arm it is not)."""
    section = dict(train_cfg.get('speed') or {})
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f'[training.speed]: unknown key(s) {sorted(unknown)}; known: '
                         f'{sorted(DEFAULTS)}')
    opts = dict(DEFAULTS)
    if not section.get('enabled', True):
        opts.update(_OFF)
    opts.update(section)
    for key, allowed in (('autocast_bf16', AUTOCAST_PARTS), ('compile', COMPILE_PARTS)):
        opts[key] = list(opts[key] or [])
        bad = set(opts[key]) - set(allowed)
        if bad:
            raise SystemExit(f'[training.speed].{key}: unknown part(s) {sorted(bad)}; known: '
                             f'{list(allowed)}')
    return opts


def _to_float(x):
    """Every half-precision tensor in a nested container, as float32."""
    if torch.is_tensor(x):
        return x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
    if isinstance(x, (tuple, list)):
        return type(x)(_to_float(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_float(v) for k, v in x.items()}
    return x


def _autocast_forward(orig_forward, *args, **kwargs):
    """`orig_forward` under bf16 autocast, its outputs handed back as float32."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = orig_forward(*args, **kwargs)
    return _to_float(out)


def _wrap_autocast(module) -> None:
    """Patch the instance's forward (hooks and parameter names are untouched)."""
    module.forward = functools.partial(_autocast_forward, module.forward)


def _needs_backward(blk, x) -> bool:
    """Whether running `blk` on `x` builds an autograd graph worth checkpointing."""
    if not torch.is_grad_enabled():
        return False
    if torch.is_tensor(x) and x.requires_grad:
        return True
    return any(p.requires_grad for p in blk.parameters())


def _block_forward(blk, fwd, ckpt, x, *args, **kwargs):
    """One encoder block: checkpointed iff `ckpt` and it needs a backward."""
    if ckpt and _needs_backward(blk, x):
        return checkpoint(fwd, x, *args, use_reentrant=False, **kwargs)
    return fwd(x, *args, **kwargs)


def patch_encoder_blocks(encoder, compile_blocks: bool, ckpt: bool) -> None:
    """Per-block compile and selective checkpointing, OUTSIDE the compiled region (checkpoint wraps
    the compiled forward, so dynamo never traces the requires-grad branch). The library's own
    every-block flag is turned off: this function owns checkpointing from here.

    Dynamo caches per CODE object and all blocks share `Block.forward`, so frozen vs trainable
    blocks (a `requires_grad` guard) times the camera-count shapes exceed the default limit of 8
    and every later variant silently ran eager; the limit is raised to `RECOMPILE_LIMIT`."""
    encoder.use_activation_checkpointing = False
    if compile_blocks:
        import torch._dynamo.config as dynamo_config
        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, RECOMPILE_LIMIT)
    for blk in encoder.blocks:
        fwd = torch.compile(blk.forward) if compile_blocks else blk.forward
        blk.forward = functools.partial(_block_forward, blk, fwd, ckpt)


def decoder_blocks(model) -> list:
    """The attention / MLP blocks of E1's decoder and, on a MoE, E2's own copy (a set-only
    model's E2 decoder IS the host's, so it is not listed twice)."""
    decoders = [model.decoder]
    sd = getattr(model, 'set_decoder', None)
    if sd is not None and sd.decoder is not model.decoder:
        decoders.append(sd.decoder)
    mods = []
    for dec in decoders:
        for name in _DEC_BLOCKS:
            mods.extend(list(getattr(dec, name, None) or []))
    return mods


def apply_training_speedups(model, train_cfg: dict, device_type: str) -> dict:
    """Apply the effective `[training.speed]` options to the UNWRAPPED pose model in place, before
    DDP wrapping -> a description dict (logged and recorded in provenance).

    Under bf16 the cuDNN SDPA backend is turned off: it re-plans a graph for every new shape (the
    camera count changes every step), so flash attention is used instead (tracktail's finding).
    """
    opts = speed_options(train_cfg)
    ckpt = bool(opts['encoder_activation_checkpointing'])
    info = {'encoder_activation_checkpointing': 'trainable blocks' if ckpt else False}
    cuda = device_type == 'cuda' and torch.cuda.is_available()
    ac, comp = set(opts['autocast_bf16']), set(opts['compile'])
    if ac and not (cuda and torch.cuda.is_bf16_supported()):
        info['autocast_bf16_skipped'] = f'no bf16 on {device_type}'
        ac = set()
    if comp and not cuda:
        info['compile_skipped'] = f'no torch.compile on {device_type}'
        comp = set()
    if ac:
        torch.backends.cuda.enable_cudnn_sdp(False)
    patch_encoder_blocks(model.scene_encoder.encoder, 'encoder' in comp, ckpt)
    if 'encoder' in ac:
        _wrap_autocast(model.scene_encoder.encoder)
    if 'decoder' in ac:
        for m in decoder_blocks(model):
            _wrap_autocast(m)
    info['autocast_bf16'] = sorted(ac)
    info['compile'] = sorted(comp)
    return info
