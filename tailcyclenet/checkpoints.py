"""Run folders, warm start, save and load.

A run folder holds the config, the keypoint registry and the checkpoints; every consumer takes
only `--run <folder>`. Schedule-free training keeps two iterates -- `model_state` (raw, resume)
and `model_state_eval` (averaged, evaluate) -- so both are saved explicitly.
"""
from __future__ import annotations

import tomllib
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

from posetail.posetail.train_utils import (_convert_cross_attn, _filter_shape_mismatch,
                                           _interp_res_params)

from .format import Registry
from .model import build_model

# Names `posetail.posetail.encoder_decoder.SceneRepresentation.__init__` resolves in its own
# module namespace to build the video backbone; see `skip_video_encoder_download`.
_VJEPA_BUILDER_NAMES = ('vjepa2_1_vit_base_384', 'vjepa2_1_vit_large_384',
                        'vjepa2_1_vit_giant_384', 'vjepa2_1_vit_gigantic_384')


@contextmanager
def skip_video_encoder_download():
    """Build the next `TrackerEncoder` without fetching VJEPA2 weights over the network.

    `SceneRepresentation.__init__` (posetail.posetail.encoder_decoder) always calls
    `vjepa2_1_vit_{base,large,giant,gigantic}_384()` with no arguments, so every
    `TrackerEncoder()` construction downloads the multi-GB backbone checkpoint from
    dl.fbaipublicfiles.com -- even here, where the caller (`load_run`, `_load_packaged_pose`,
    a warm start, a training resume) is about to overwrite EVERY one of those tensors with its
    own checkpoint's `model_state`/`model_state_eval` via `load_state_dict`. The download is
    also the one place this repo's inference path silently needs internet access, which a
    deployment host need not have (see `dev/plans/infer_from_videos_and_calibration.md`).

    The four builders (`posetail.posetail.vjepa2`) already take `pretrained: bool` for exactly
    this; `SceneRepresentation` just never forwards it upward. This substitutes which
    already-public builder function `encoder_decoder`'s own module namespace resolves to, for
    the scope of ONE `build_model()` call, then restores it -- it drives an option the library
    itself defines but never exposes, rather than reproducing behaviour posetail is missing (the
    no-monkeypatch invariant in `tailcyclenet/__init__.py` is about the latter: workarounds for
    library bugs, deleted once 0.4.1 landed them upstream).

    Never wrap a `build_model()` call that will NOT immediately load a full checkpoint --
    training from scratch has nothing else to supply those weights, and skipping the download
    there would silently start the encoder from noise instead of VJEPA2.
    """
    from posetail.posetail import encoder_decoder as _ed
    from posetail.posetail import vjepa2 as _vj

    saved = {name: getattr(_ed, name) for name in _VJEPA_BUILDER_NAMES}
    try:
        for name in _VJEPA_BUILDER_NAMES:
            setattr(_ed, name, partial(getattr(_vj, name), pretrained=False))
        yield
    finally:
        for name, fn in saved.items():
            setattr(_ed, name, fn)


_BASE_CONFIG = Path(__file__).resolve().parent.parent / 'configs' / 'base.toml'


def load_config(path, base: Path | None = None) -> dict:
    """A run config layered over a base file -- the pose default is the repo's
    `configs/base.toml`; the detector loader passes `configs/detector.toml`.

    EVERY config layers over its family's base automatically: the `extends` key is deleted and
    RAISES by name, and the overlay IS the whole difference. The merge is per BLOCK, key by
    key, so an overlay need not restate the base's blocks.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    if 'extends' in cfg:
        raise SystemExit(
            f'{path.name}: `extends` is deleted -- every config layers over '
            f'`{(base or _BASE_CONFIG).name}` automatically, so the key is not needed. '
            'Delete the line and the recipe is unchanged.')
    with open(base or _BASE_CONFIG, 'rb') as f:
        base_cfg = tomllib.load(f)
    for block, over in cfg.items():
        if isinstance(over, dict) and isinstance(base_cfg.get(block), dict):
            base_cfg[block].update(over)
        else:
            base_cfg[block] = over
    return base_cfg


def check_image_size(config: dict) -> None:
    """`[model].image_size` and `[data].image_size` must agree; nothing else notices if they do
    not. The loader resizes crops to the data value while the weights bake the model value into
    the decode arithmetic -- both silent otherwise. `--refine-px` is the only thing that may walk
    past this, because `model.PoseTrackerEncoder.forward` compensates for it.
    """
    model_px = config.get('model', {}).get('image_size')
    data_px = config.get('data', {}).get('image_size')
    if model_px is None or data_px is None or int(model_px) == int(data_px):
        return
    raise ValueError(
        f'[model].image_size = {model_px} but [data].image_size = {data_px}. These must agree: '
        f'the loader resizes crops to {data_px} while the model decodes as if they were '
        f'{model_px}, shifting 2D predictions by {(int(model_px) - int(data_px)) // 2} px and '
        f'scaling the 3D residual by {int(model_px) / int(data_px):g}.')


def is_hf_repo_id(value: str) -> bool:
    """Whether ``value`` is a Hugging Face repo id rather than a local path.

    Hub repo ids are ``namespace/name``. Existing paths always win, so a relative local
    checkpoint directory can still be used when it exists on disk.
    """
    value = str(value)
    return (value.count('/') == 1 and not value.startswith(('/', '.'))
            and not Path(value).exists())


def resolve_hf_checkpoint(repo_id: str, revision: str | None = None) -> Path:
    """Download a packaged posetail checkpoint and return its local cached path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename='model.pth', revision=revision))


def resolve_checkpoint(folder: Path, checkpoint: str | None = None):
    """Resolve the latest training checkpoint, or an explicit checkpoint name.

    The validation-selected ``checkpoint_best.pth`` is never implicit. Numbered checkpoints are
    preferred because they identify the highest training iteration; a ``checkpoint_last.pth`` is
    the fallback for runs that only maintain the rolling last file.
    """
    folder = Path(folder)
    if checkpoint:
        p = folder / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    files = sorted(folder.glob('checkpoint_*.pth'))
    numbered = sorted((int(p.stem.split('_')[-1]), p) for p in files
                      if p.stem.split('_')[-1].isdigit())
    if numbered:
        got = numbered[-1][1]
        if (folder / 'checkpoint_last.pth').exists():
            print(f'{folder}: using latest numbered checkpoint {got.name}')
        return got
    last = folder / 'checkpoint_last.pth'
    if last.exists():
        return last
    if not files:
        raise FileNotFoundError(f'{folder}: no checkpoint_*.pth')
    raise FileNotFoundError(f'{folder}: no numbered checkpoint or checkpoint_last.pth; '
                            'checkpoint_best.pth is validation-selected and must be explicit')


def provenance() -> dict:
    """The commit this source tree is at, and whether it was dirty. Best effort; empty when git
    is unavailable (an installed copy, a tarball). It is a record, not a gate.
    """
    import subprocess
    root = Path(__file__).resolve().parent.parent
    try:
        run = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        if run.returncode:
            return {}
        commit = run.stdout.strip()
        st = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                            capture_output=True, text=True, timeout=10)
        return {'commit': commit, 'dirty': bool(st.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {}


def prior_provenance(run: Path) -> dict:
    """The `provenance.toml` a previous run left in this folder, or {}. Read BEFORE it is
    rewritten: the resume path needs the previous world size to know what the rates meant.
    """
    p = Path(run) / 'provenance.toml'
    if not p.exists():
        return {}
    try:
        with open(p, 'rb') as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def save_run_meta(run: Path, config: dict, registry: Registry,
                  extra: dict | None = None) -> None:
    """`config.toml`, the keypoint registry and `provenance.toml`. `extra` joins the provenance
    with how the run was launched (world size, effective rates) -- a config cannot state these.
    """
    import toml
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.toml').write_text(toml.dumps(config))
    registry.save(run / 'keypoint_registry.toml')
    prov = {**provenance(), **(extra or {})}
    if prov:
        (run / 'provenance.toml').write_text(toml.dumps(prov))


def full_training_state(ck: dict) -> bool:
    """Whether `ck` is a full training checkpoint -- raw weights, optimizer state and the
    iteration -- rather than a packaged pose checkpoint (weights only). `save_checkpoint` and
    the reference repo's train loop write the former; `package_checkpoint.py` writes the latter.
    Only a full training checkpoint can be resumed from; anything else is a warm start.
    """
    return (isinstance(ck, dict)
            and isinstance(ck.get('model_state'), dict)
            and isinstance(ck.get('optimizer_state'), dict)
            and 'iteration' in ck)


def save_checkpoint(run: Path, iteration: int, model, optimizer, config: dict,
                    name: str = 'last', write: bool = True,
                    registry: Registry | None = None) -> Path | None:
    """Save both schedule-free iterates to `checkpoint_<name>.pth`, overwriting.

    `model_state` is the raw training weight (resume); `model_state_eval` is the averaged weight
    (evaluate), captured by toggling the optimizer into eval mode and back. Only `last` and
    `best` are ever written; the write renames a sibling temp file into place.

    `write = False` runs the eval/train toggle but skips the clone and disk write -- correctness,
    not an optimisation: the float32 toggle round trip is not bit-exact, so every rank must pay
    it (only rank 0 writes) or rank 0's weights drift, which `check_ranks_agree` exists to catch.
    A DualOptimizer exposes eval()/train() even when its Muon half has NO averaged iterate
    (`muon_schedulefree = false`), in which case `model_state_eval` would be half-averaged;
    `has_averaged_iterate` reports whether both halves carry an `x`. The eval/train toggle is
    UNCONDITIONAL -- every rank pays it -- for the same bit-exactness reason.
    """
    state = None
    if write:
        ckpt_dir = run / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    eval_state = None
    averaged = getattr(optimizer, 'has_averaged_iterate',
                       hasattr(optimizer, 'eval') and hasattr(optimizer, 'train'))
    if averaged:
        optimizer.eval()
        if write:
            eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer.train()
    if not write:
        return None
    path = ckpt_dir / f'checkpoint_{name}.pth'
    tmp = path.with_suffix('.tmp')
    torch.save({'iteration': iteration, 'model_state': state,
                'model_state_eval': eval_state,
                'optimizer_state': optimizer.state_dict(),
                'config': config,
                'model_config': config.get('model'),
                'keypoint_registry': None if registry is None else registry.to_dict()}, tmp)
    tmp.replace(path)
    return path


def _load_packaged_pose(path: Path, device='cpu', model_overrides: dict | None = None):
    """Load a self-contained pose checkpoint, whether packaged or copied from training."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f'{path}: checkpoint must be a dictionary, got {type(ckpt).__name__}')
    if ckpt.get('kind', 'pose') != 'pose':
        raise ValueError(f'{path}: expected a pose checkpoint, got kind={ckpt.get("kind")!r}')
    registry_doc = ckpt.get('keypoint_registry')
    if not isinstance(registry_doc, dict):
        raise ValueError(f'{path}: pose checkpoint has no embedded keypoint_registry; '
                         'use a checkpoint written after registry embedding or a run folder')
    registry = Registry.from_dict(registry_doc)

    config = ckpt.get('config')
    if not isinstance(config, dict):
        config = {'model': ckpt.get('model_config'), 'data': ckpt.get('data_config', {})}
    if not isinstance(config.get('model'), dict):
        raise ValueError(f'{path}: pose checkpoint has no dictionary model config')
    config = dict(config)
    check_image_size(config)
    if model_overrides:
        config['model'] = {**config.get('model', {}), **model_overrides}
        print(f'load_run: [model] OVERRIDDEN {model_overrides} -- this is an assertion about what '
              'the checkpoint was trained with, not something read from it')

    state = ckpt.get('model_state_eval') or ckpt.get('model_state')
    if not isinstance(state, dict):
        raise ValueError(f'{path}: pose checkpoint has no model_state dictionary')
    source = ckpt.get('source_run', '?')
    selected = ckpt.get('source_checkpoint', path.name)
    print(f'packaged pose checkpoint: {selected} from {source} '
          f'(iteration {ckpt.get("iteration", "?")})')
    with skip_video_encoder_download():
        model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    missing, unexpected = model.load_state_dict(state, strict=False)
    _report('load_run', missing, unexpected, [])
    return model.to(device).eval(), config, registry, path


def peek_registry(run: Path) -> Registry:
    """Read the keypoint registry from a run folder or a self-contained pose checkpoint."""
    run = Path(run)
    if run.is_file():
        ckpt = torch.load(run, map_location='cpu', weights_only=False)
        if not isinstance(ckpt, dict) or not isinstance(ckpt.get('keypoint_registry'), dict):
            raise ValueError(f'{run}: pose checkpoint has no embedded keypoint_registry')
        return Registry.from_dict(ckpt['keypoint_registry'])
    return Registry.load(run / 'keypoint_registry.toml')


def load_run(run: Path, checkpoint: str | None = None, device='cpu',
             model_overrides: dict | None = None):
    """(model, config, registry, checkpoint_path) from a run folder or pose checkpoint."""
    run = Path(run)
    if run.is_file():
        if checkpoint is not None:
            raise ValueError(f'{run}: --checkpoint selects a file inside a run folder, but --run '
                             'already names a checkpoint file')
        return _load_packaged_pose(run, device=device, model_overrides=model_overrides)
    with open(run / 'config.toml', 'rb') as f:
        config = tomllib.load(f)
    check_image_size(config)
    if model_overrides:
        config['model'] = {**config.get('model', {}), **model_overrides}
        print(f'load_run: [model] OVERRIDDEN {model_overrides} -- this is an assertion about what '
              'the checkpoint was trained with, not something read from it')
    prov = run / 'provenance.toml'
    if prov.exists():
        with open(prov, 'rb') as f:
            p = tomllib.load(f)
        print(f'run provenance: {p.get("commit", "?")[:12]}'
              f'{" +DIRTY" if p.get("dirty") else ""}')
    else:
        print(f'{run}: no provenance.toml -- this run predates commit recording, so which '
              'architecture the weights were trained under cannot be read back from the folder')
    registry = Registry.load(run / 'keypoint_registry.toml')
    path = resolve_checkpoint(run / 'checkpoints', checkpoint)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    with skip_video_encoder_download():
        model = build_model(config['model'], n_keypoints=registry.n_keypoints)
    state = ckpt.get('model_state_eval')
    if state is None:
        print(f'{path.name}: no model_state_eval; falling back to the raw training weights')
        state = ckpt['model_state']
    missing, unexpected = model.load_state_dict(state, strict=False)
    _report('load_run', missing, unexpected, [])
    return model.to(device).eval(), config, registry, path


def warm_start(model, checkpoint_path: Path, verbose: bool = True,
               base_names: tuple[str, ...] | None = None) -> set[str]:
    """Load the base tracker into a pose model. Returns the names of the params left fresh.

    Base migrations run first and every `strict=False` drop is named. A GROWN REGISTRY KEEPS ITS
    ROWS: the checkpoint's (n0, d) identity table is copied into the first n0 rows of this
    model's (n, d) one -- refused if `base_names`'s length does not match n0, because a
    mis-applied copy points each row at a different body part. `_interp_res_params` returns
    (dict, BOOL): `interpolated` is the flag, not a key list.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    state = dict(ckpt.get('model_state_eval') or ckpt['model_state'])

    state = _convert_cross_attn(state, model)
    state, interpolated = _interp_res_params(state, model)
    if interpolated and verbose:
        print('warm start: resolution-coupled tensors checked; see any res-interp lines above')

    msd = model.state_dict()
    for k, v in list(state.items()):
        if not k.endswith('kpt_embed.weight') or k not in msd:
            continue
        n0, n = v.shape[0], msd[k].shape[0]
        if not (n0 < n and v.shape[1:] == msd[k].shape[1:]):
            continue
        if base_names is not None and len(base_names) != n0:
            if verbose:
                print(f'warm start: {k} is {n0} rows but the base registry names '
                      f'{len(base_names)}, so it is NOT that registry\'s table -- the rows are '
                      'left fresh rather than copied onto the wrong keypoints')
            continue
        grown = msd[k].clone()
        grown[:n0] = v
        state[k] = grown
        if verbose:
            print(f'warm start: {k} widened {n0} -> {n} rows, {n0} preserved')

    state, dropped = _filter_shape_mismatch(state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        _report('warm start', missing, unexpected, dropped)
    return set(missing) | set(dropped)


def _report(what, missing, unexpected, dropped):
    """Name everything. A silent drop here is a whole training run spent on the wrong weights."""
    def head(xs, n=8):
        """First `n` names joined, with `… (+k more)` appended when more remain."""
        xs = list(xs)
        return ', '.join(xs[:n]) + (f' … (+{len(xs) - n} more)' if len(xs) > n else '')

    if dropped:
        print(f'{what}: {len(dropped)} tensor(s) dropped on a SHAPE MISMATCH: {head(dropped)}')
    if missing:
        print(f'{what}: {len(missing)} param(s) left at fresh init: {head(missing)}')
    if unexpected:
        print(f'{what}: {len(unexpected)} checkpoint key(s) UNUSED: {head(unexpected)}')
    if not (dropped or missing or unexpected):
        print(f'{what}: exact match, nothing fresh and nothing discarded')
