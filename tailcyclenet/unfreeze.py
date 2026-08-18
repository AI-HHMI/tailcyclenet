"""Staged encoder unfreeze: finetune the last N ViT blocks from iteration M.

THE MODEL SIDE IS ENTIRELY UPSTREAM'S. posetail 0.3.5's `TrackerEncoder` takes
`video_encoder_requires_grad` as a bool OR an int (the iteration to switch on at) and
`video_encoder_finetune_last_n_layers`, and `TrackerEncoder.unfreeze_video_encoder(iteration)`
does the gate, the idempotence and the last-N block selection. `build_model` splats `[model]`, so
both keys reach the constructor with no code here. This module adds only the two things upstream
has no opinion about:

1. **The optimizer.** `build_muon` filters `requires_grad` at construction, so a param frozen at
   step 0 is in NO param group -- forever. Flipping `requires_grad` at 5,000 would hand those
   tensors gradients that nothing steps, and the run would report as a finetuning arm while
   finetuning nothing. The newly-trainable params are added as NEW groups, routed by
   `optim.route_param` so build-time and unfreeze-time cannot disagree.

2. **The hierarchical output norms.** Upstream unfreezes `encoder.blocks[-N:]` plus an
   `encoder.norm` that the VJEPA 2.1 `VisionTransformer` does not have -- so `norms_block`, which
   is applied at `hierarchical_layers` (`[5,11,17,23]` at depth 24) and feeds the decoder, would
   stay frozen while the block feeding IT trains. `norms_block[i]` is unfrozen iff its layer is
   inside the trainable range. A deliberate, repo-local divergence from the reference, and
   separately measurable by dropping it.

Called from the training loop every iteration (cheap: one int compare once unfrozen) and again on
resume to replay the layout a fresh run would have reached -- see `scripts/train.py`.
"""
from __future__ import annotations

from .optim import PoseDualOptimizer, embedding_ids, group_lr, route_param


def _norms_in_range(encoder, n_last: int) -> list[int]:
    """Indices of `norms_block` whose hierarchical layer falls inside the last `n_last` blocks.

    `getattr`-guarded on both attributes: an encoder variant without them is skipped, not crashed.
    At depth 24 with n_last = 4 the trainable range is layers 20..23 and `hierarchical_layers` is
    `[5,11,17,23]`, so this is `[3]` -- which is also the final norm the non-hierarchical path
    uses.
    """
    blocks = getattr(encoder, 'blocks', None)
    norms = getattr(encoder, 'norms_block', None)
    layers = getattr(encoder, 'hierarchical_layers', None)
    if blocks is None or norms is None or layers is None:
        return []
    first = len(blocks) - n_last
    return [i for i, layer in enumerate(layers) if i < len(norms) and layer >= first]


def apply_norms_extension(model) -> list[int]:
    """Apply the norms extension to a model whose encoder is ALREADY trainable at build time.

    `video_encoder_requires_grad = true` unfreezes inside the constructor
    (`SceneRepresentation.__init__` -> `set_encoder_requires_grad`), which never reaches
    `apply_staged_unfreeze` -- so without this, `true` + `n_last` would train a DIFFERENT parameter
    set than an int + the same `n_last`: same blocks, but the hierarchical norms frozen. Two ways
    to say "train the last N" that disagree is exactly the class of silent divergence this repo
    refuses elsewhere.

    Called from `scripts/train.py` BEFORE the optimizer is built, so the norms are routed into a
    group like any other trainable tensor. Returns the indices unfrozen (empty when the encoder is
    frozen, when no last-N is set, or when the variant has no `norms_block`).
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

    Returns None when nothing fired (not scheduled, not yet, or already unfrozen -- all three are
    upstream's `unfreeze_video_encoder` answering), else a dict describing what was unfrozen.

    Idempotent by delegation: upstream flips `video_encoder_requires_grad` to True on the first
    fire and returns False forever after, so this adds each group exactly once.
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
    # everything and then unfreezes the last N blocks, so doing this first would be undone.
    norms = _norms_in_range(encoder, n_last)
    for i in norms:
        for p in encoder.norms_block[i].parameters():
            p.requires_grad_(True)

    # Route every tensor that is now trainable and was not in a group before. Membership is by
    # `id`, so a param the optimizer already holds (any non-encoder param) is never added twice.
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
        # it -- merged BY LR rather than added per route, or the encoder would arrive as two
        # groups at one identical learning rate, which `build_optimizer` would never have built.
        # Same fresh-group reasoning as `PoseDualOptimizer.add_adamw_group`.
        by_lr: dict[float, list] = {}
        for route, params in sorted(added.items()):
            by_lr.setdefault(group_lr(route, opt_cfg), []).extend(params)
        for lr, params in sorted(by_lr.items()):
            opt.add_param_group({'params': params, 'lr': lr, 'weight_decay': wd})
            # Only AdamW-SF carries a per-group `train_mode`; a plain optimizer has none, and
            # `.get` keeps this usable from a test or a probe that builds one.
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
    """Reach, at resume, the group layout a fresh run would have at `start_it`.

    The optimizer is always BUILT in the frozen layout and the unfreeze replayed on top, rather
    than built already-unfrozen: `load_state_dict` matches groups BY POSITION, and only replaying
    the same sequence of adds guarantees the same order.
    """
    return apply_staged_unfreeze(model, opt, opt_cfg, start_it, fresh=fresh)


def trainable_encoder_params(model) -> int:
    """Trainable params inside the video encoder. 0 before the unfreeze; logged every print so the
    transition is a visible step in the dashboard rather than one line of stdout."""
    enc = getattr(getattr(model, 'scene_encoder', None), 'encoder', None)
    if enc is None:
        return 0
    return sum(p.numel() for p in enc.parameters() if p.requires_grad)
