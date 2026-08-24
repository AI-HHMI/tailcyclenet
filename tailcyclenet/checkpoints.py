"""Run folders, warm start, save and load.

A run folder holds the config, the keypoint registry and the checkpoints; every consumer takes
only `--run <folder>`. Schedule-free training keeps two iterates -- `model_state` (raw, resume)
and `model_state_eval` (averaged, evaluate) -- so both are saved explicitly.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import torch

from posetail.posetail.train_utils import (_convert_cross_attn, _filter_shape_mismatch,
                                           _interp_res_params)

from .format import Registry
from .model import build_model


def load_config(path) -> dict:
    """A run config, with `extends = "<file>"` resolved once against the config's own directory.

    One level deep, so `configs/2d.toml` vs `configs/3d.toml` show their whole difference in one
    file. The merge is per BLOCK, key by key, so an overlay need not restate the base's blocks.
    """
    import tomllib

    path = Path(path)
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    base_name = cfg.pop('extends', None)
    if base_name is None:
        return cfg
    with open(path.parent / base_name, 'rb') as f:
        base = tomllib.load(f)
    if 'extends' in base:
        raise SystemExit(f'{base_name}: `extends` is one level deep; {path.name} extends it and '
                         'it may not extend anything further.')
    for block, over in cfg.items():
        if isinstance(over, dict) and isinstance(base.get(block), dict):
            base[block].update(over)
        else:
            base[block] = over
    return base


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
    """An explicit name, else `checkpoint_last.pth`, else the newest by name.

    `last` is the default because it is the weight a resume continues from; pass
    `checkpoint='checkpoint_best.pth'` for the best-val one.
    """
    folder = Path(folder)
    if checkpoint:
        p = folder / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    last = folder / 'checkpoint_last.pth'
    if last.exists():
        return last
    files = sorted(folder.glob('checkpoint_*.pth'))
    if not files:
        raise FileNotFoundError(f'{folder}: no checkpoint_*.pth')
    # HIGHEST ITERATION, NOT LAST BY NAME: 'b' > '0', so a folder holding
    # `checkpoint_00060000.pth` beside `checkpoint_best.pth` must return the numeric one.
    numbered = sorted((int(p.stem.split('_')[-1]), p) for p in files if p.stem.split('_')[-1].isdigit())
    got = numbered[-1][1] if numbered else files[-1]
    print(f'{folder}: no checkpoint_last.pth, using {got.name} of '
          f'{[p.name for p in files]}')
    return got


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


def save_checkpoint(run: Path, iteration: int, model, optimizer, config: dict,
                    name: str = 'last', write: bool = True) -> Path | None:
    """Save both schedule-free iterates to `checkpoint_<name>.pth`, overwriting.

    `model_state` is the raw training weight (resume); `model_state_eval` is the averaged weight
    (evaluate), captured by toggling the optimizer into eval mode and back. Only `last` and
    `best` are ever written; the write renames a sibling temp file into place.

    `write = False` runs the eval/train toggle but skips the clone and disk write -- correctness,
    not an optimisation: the float32 toggle round trip is not bit-exact, so every rank must pay
    it (only rank 0 writes) or rank 0's weights drift, which `check_ranks_agree` exists to catch.
    """
    state = None
    if write:
        ckpt_dir = run / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    eval_state = None
    # A DualOptimizer exposes eval()/train() even when its Muon half has NO averaged iterate
    # (`muon_schedulefree = false`), in which case `model_state_eval` would be half-averaged.
    # `has_averaged_iterate` reports whether both halves carry an `x`.
    averaged = getattr(optimizer, 'has_averaged_iterate',
                       hasattr(optimizer, 'eval') and hasattr(optimizer, 'train'))
    if averaged:
        optimizer.eval()                    # UNCONDITIONAL -- every rank pays the toggle
        if write:
            eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        optimizer.train()                   # UNCONDITIONAL, same reason
    if not write:
        return None
    path = ckpt_dir / f'checkpoint_{name}.pth'
    tmp = path.with_suffix('.tmp')
    torch.save({'iteration': iteration, 'model_state': state,
                'model_state_eval': eval_state,
                'optimizer_state': optimizer.state_dict(), 'model_config': config.get('model')},
               tmp)
    tmp.replace(path)
    return path


def load_run(run: Path, checkpoint: str | None = None, device='cpu',
             model_overrides: dict | None = None):
    """(model, config, registry, checkpoint_path) from a run folder. Nothing else is needed.

    `model_overrides` patches `[model]` before build, for a run folder written before a key
    existed; it is echoed loudly because it is an assertion about weights nobody recorded.
    """
    run = Path(run)
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
    mis-applied copy points each row at a different body part.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    state = dict(ckpt.get('model_state_eval') or ckpt['model_state'])

    state = _convert_cross_attn(state, model)
    # Returns (dict, BOOL); `interpolated` is the flag, not a key list.
    state, interpolated = _interp_res_params(state, model)
    if interpolated and verbose:
        print('warm start: resolution-coupled tensors checked; see any res-interp lines above')

    # A GROWN REGISTRY KEEPS ITS ROWS: the checkpoint's (n0, d) identity table is exactly the
    # first n0 rows of this model's (n, d) one (see the docstring for the refusal rule).
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
