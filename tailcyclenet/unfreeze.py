"""Staged encoder unfreeze: finetune the last N ViT blocks from iteration M.

The model side is upstream's (`video_encoder_requires_grad` as bool or int). This adds the two
things upstream has no opinion about: the optimizer (a param frozen at build is in NO group, so
newly-trainable tensors are added as new groups via `optim.route_param`), and the hierarchical
output norms (upstream leaves `norms_block` frozen behind the trainable blocks).
"""
from __future__ import annotations

from .optim import PoseDualOptimizer, embedding_ids, group_lr, route_param


def _norms_in_range(encoder, n_last: int) -> list[int]:
    """Indices of `norms_block` whose hierarchical layer falls inside the last `n_last` blocks.
    `getattr`-guarded: an encoder variant without them is skipped, not crashed.
    """
    blocks = getattr(encoder, 'blocks', None)
    norms = getattr(encoder, 'norms_block', None)
    layers = getattr(encoder, 'hierarchical_layers', None)
    if blocks is None or norms is None or layers is None:
        return []
    first = len(blocks) - n_last
    return [i for i, layer in enumerate(layers) if i < len(norms) and layer >= first]


def apply_norms_extension(model) -> list[int]:
    """Apply the norms extension to a model whose encoder is ALREADY trainable at build time --
    `true` unfreezes inside the constructor and never reaches `apply_staged_unfreeze`, so without
    this it would train a different parameter set than an int + the same `n_last`. Called before
    the optimizer is built.
    """
    if not getattr(model, 'video_encoder_requires_grad', False):
        return []
    n_last = getattr(model, 'video_encoder_finetune_last_n_layers', None)
    if n_last is None:
        return []                      # the whole encoder is trainable; every norm already is
    encoder = model.scene_encoder.encoder
    norms = _norms_in_range(encoder, int(n_last))
    for i in norms:
        for p in encoder.norms_block[i].parameters():
            p.requires_grad_(True)
    return norms


def apply_staged_unfreeze(model, opt, opt_cfg: dict, iteration: int,
                          fresh: set[str] | None = None) -> dict | None:
    """Fire the staged unfreeze if `iteration` reaches it, and tell the optimizer about it.
    Returns None when nothing fired; idempotent by delegation (upstream flips the flag on the
    first fire and returns False forever after).
    """
    if not hasattr(model, 'unfreeze_video_encoder'):
        return None
    if not model.unfreeze_video_encoder(int(iteration)):
        return None

    encoder = model.scene_encoder.encoder
    n_last = getattr(model, 'video_encoder_finetune_last_n_layers', None)
    n_blocks = len(getattr(encoder, 'blocks', []))
    n_last = n_blocks if n_last is None else int(n_last)

    # THE NORMS EXTENSION, applied after upstream has set `requires_grad` -- upstream re-freezes
    # everything first, so doing this earlier would be undone.
    norms = _norms_in_range(encoder, n_last)
    for i in norms:
        for p in encoder.norms_block[i].parameters():
            p.requires_grad_(True)

    # Route every tensor now trainable and not already held; membership is by `id`, so a param
    # the optimizer already holds is never added twice.
    held = {id(p) for g in opt.param_groups for p in g['params']}
    embed_ids = embedding_ids(model)
    added: dict[str, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in held:
            continue
        added.setdefault(route_param(name, p, fresh or set(), embed_ids), []).append(p)

    wd = float(opt_cfg.get('weight_decay', 0.0))
    n_muon = n_adamw = 0
    if isinstance(opt, PoseDualOptimizer):
        for route, params in sorted(added.items()):
            lr = group_lr(route, opt_cfg)
            if route.startswith('muon'):
                opt.add_muon_group(params, lr, wd)
                n_muon += 1
            else:
                opt.add_adamw_group(params, lr, wd)
                n_adamw += 1
    else:
        # `optimizer = "schedulefree"`: ONE AdamW-SF optimizer, so the Muon routes collapse onto
        # it merged BY LR, or the encoder would arrive as two groups at one identical rate.
        by_lr: dict[float, list] = {}
        for route, params in sorted(added.items()):
            by_lr.setdefault(group_lr(route, opt_cfg), []).extend(params)
        for lr, params in sorted(by_lr.items()):
            opt.add_param_group({'params': params, 'lr': lr, 'weight_decay': wd})
            # Only AdamW-SF carries a per-group `train_mode`; `.get` keeps this usable from a
            # test or probe that builds a plain optimizer.
            if 'train_mode' in opt.param_groups[0]:
                opt.param_groups[-1]['train_mode'] = opt.param_groups[0]['train_mode']
            n_adamw += 1

    tensors = [p for ps in added.values() for p in ps]
    return {'iteration': int(iteration), 'n_last_blocks': n_last, 'n_blocks': n_blocks,
            'blocks': list(range(max(n_blocks - n_last, 0), n_blocks)), 'norms': norms,
            'n_tensors': len(tensors), 'n_params': sum(p.numel() for p in tensors),
            'muon_groups': n_muon, 'adamw_groups': n_adamw}


def replay_staged_unfreeze(model, opt, opt_cfg: dict, start_it: int,
                           fresh: set[str] | None = None) -> dict | None:
    """Reach, at resume, the group layout a fresh run would have at `start_it`. The optimizer is
    always built in the frozen layout and the unfreeze replayed on top: `load_state_dict`
    matches groups BY POSITION, so only the same sequence of adds guarantees the same order.
    """
    return apply_staged_unfreeze(model, opt, opt_cfg, start_it, fresh=fresh)


def trainable_encoder_params(model) -> int:
    """Trainable params inside the video encoder; logged every print so the unfreeze is a
    visible step in the dashboard rather than one line of stdout.
    """
    enc = getattr(getattr(model, 'scene_encoder', None), 'encoder', None)
    if enc is None:
        return 0
    return sum(p.numel() for p in enc.parameters() if p.requires_grad)
