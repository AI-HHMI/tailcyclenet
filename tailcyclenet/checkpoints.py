"""Run folders, warm start, save and load.

A run folder holds the config, the keypoint registry and the checkpoints; every checkpoint ALSO
embeds the config, registry and provenance, so a consumer takes a run folder or one `.pth` file
(pose, scorer, detector alike). Schedule-free training keeps two iterates -- `model_state` (raw, resume)
and `model_state_eval` (averaged, evaluate) -- so both are saved explicitly.
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import torch

from tracktail.model.train_utils import (_convert_cross_attn, _filter_shape_mismatch,
                                           _interp_res_params)

from .config import check_image_size
from .format import Registry
from .pose.model import build_model

# What a checkpoint or run folder is a checkpoint OF. A run folder's kind lives in its config as
# `[run] kind`; a checkpoint's lives in the file. Both are written by `save_run_meta` /
# `save_checkpoint` rather than declared in the config file, so a family cannot forget to say.
KINDS = ('pose', 'detector', 'scorer')

def is_hf_repo_id(value: str) -> bool:
    """Whether ``value`` is a Hugging Face repo id rather than a local path.

    Hub repo ids are ``namespace/name``. Existing paths always win, so a relative local
    checkpoint directory can still be used when it exists on disk.
    """
    value = str(value)
    return (value.count('/') == 1 and not value.startswith(('/', '.'))
            and not Path(value).exists())


def resolve_hf_checkpoint(repo_id: str, revision: str | None = None) -> Path:
    """Download a packaged tracktail checkpoint and return its local cached path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename='model.pth', revision=revision))


def resolve_checkpoint(folder: Path, checkpoint: str | None = None):
    """Resolve the latest training checkpoint, or an explicit checkpoint name.

    The validation-selected ``checkpoint_best.pth`` is never implicit. Numbered checkpoints are
    preferred because they identify the highest training iteration; a ``checkpoint_last.pth`` is
    the fallback for runs that only maintain the rolling last file.
    """
    folder = Path(folder)
    if (folder / 'checkpoints').is_dir() and not any(folder.glob('checkpoint_*.pth')):
        folder = folder / 'checkpoints'
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
                  extra: dict | None = None, kind: str = 'pose') -> None:
    """`config.toml`, the keypoint registry and `provenance.toml`. `extra` joins the provenance
    with how the run was launched (world size, effective rates) -- a config cannot state these.

    `kind` is stamped into the config as `[run] kind` by the WRITER, not declared by the config
    file: a family that forgot the key would otherwise produce a run folder indistinguishable
    from a pose run. `load_run` guards on it.
    """
    import toml
    assert kind in KINDS, f'kind must be one of {KINDS}, got {kind!r}'
    run.mkdir(parents=True, exist_ok=True)
    (run / 'config.toml').write_text(toml.dumps(stamped_config(config, kind)))
    registry.save(run / 'keypoint_registry.toml')
    prov = {**provenance(), **(extra or {})}
    if prov:
        (run / 'provenance.toml').write_text(toml.dumps(prov))


def stamped_config(config: dict, kind: str) -> dict:
    """The config exactly as `save_run_meta` writes it: a deep copy with `[run] kind` stamped.

    A checkpoint embeds THIS, not the caller's dict, so a lone `.pth` carries the same config the
    run folder's `config.toml` does (including the family guard) and a later in-place mutation of
    the training config cannot reach a checkpoint already written.
    """
    assert kind in KINDS, f'kind must be one of {KINDS}, got {kind!r}'
    config = copy.deepcopy(config)
    config['run'] = {**config.get('run', {}), 'kind': kind}
    return config


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
                    registry: Registry | None = None, kind: str = 'pose',
                    extra: dict | None = None,
                    fresh_names: set[str] | None = None,
                    world_size: int | None = None) -> Path | None:
    """Save both schedule-free iterates to `checkpoint_<name>.pth`, overwriting.

    `model_state` is the raw training weight (resume); `model_state_eval` is the averaged weight
    (evaluate), captured by toggling the optimizer into eval mode and back. Only `last` and
    `best` are ever written; the write renames a sibling temp file into place.

    Pose checkpoints also store optimizer metadata keyed by parameter name because state loading
    matches optimizer groups by position, with `world_size` (the rates are sqrt(world)-scaled, so
    a resume on another world size is refused by `start.plan_start`). `extra` joins the payload
    (the scorer's contract fields).

    `write = False` runs the eval/train toggle but skips the clone and disk write -- correctness,
    not an optimisation: the float32 toggle round trip is not bit-exact, so every rank must pay
    it (only rank 0 writes) or rank 0's weights drift, which `check_ranks_agree` exists to catch.
    A DualOptimizer exposes eval()/train() even when its Muon half has NO averaged iterate
    (`muon_schedulefree = false`), in which case `model_state_eval` would be half-averaged;
    `has_averaged_iterate` reports whether both halves carry an `x`. The eval/train toggle is
    UNCONDITIONAL -- every rank pays it -- for the same bit-exactness reason.

    `kind` records WHAT these weights are. It is written explicitly rather than left implicit
    because `_load_packaged_pose` DEFAULTED to `'pose'` for a missing key: before this parameter,
    a scorer checkpoint would have loaded as a pose checkpoint without raising, and the two share
    every encoder/decoder tensor name. Defaulting to `'pose'` keeps every existing file's meaning.
    """
    assert kind in KINDS, f'kind must be one of {KINDS}, got {kind!r}'
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
    payload = {'kind': kind, 'iteration': iteration, 'model_state': state,
               'model_state_eval': eval_state,
               'optimizer_state': optimizer.state_dict(),
               'config': stamped_config(config, kind),
               'keypoint_registry': None if registry is None else registry.to_dict(),
               'provenance': prior_provenance(run) or provenance()}
    if kind == 'pose':
        from .optim import optimizer_metadata
        payload['optimizer_metadata'] = optimizer_metadata(model, optimizer, fresh_names,
                                                           world_size)
    payload.update(extra or {})
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def enclosing_run(path: Path) -> Path | None:
    """The run folder a checkpoint file sits in, or None: `<run>/checkpoints/x.pth` (pose,
    scorer, and a W&B `files/checkpoints/` mirror) or `<run>/x.pth` (detector)."""
    for d in (path.parent.parent, path.parent) if path.parent.name == 'checkpoints' else \
            (path.parent,):
        if (d / 'config.toml').is_file():
            return d
    return None


def file_bundle(path: Path, ckpt: dict, kind: str) -> tuple[dict, Registry, dict]:
    """(config, registry, provenance) for a checkpoint FILE, without a run folder.

    The checkpoint's own embedded config/registry/provenance win: they were written beside these
    weights. A file written before one was embedded falls back to the run folder it still sits
    in (`enclosing_run`); a lone file missing either is refused by name rather than guessed.
    """
    run = enclosing_run(path)
    config = ckpt.get('config')
    if not isinstance(config, dict) and run is not None:
        with open(run / 'config.toml', 'rb') as f:
            config = tomllib.load(f)
    if not isinstance(config, dict) or not isinstance(config.get('model'), dict):
        raise ValueError(f'{path}: {kind} checkpoint has no embedded config and sits in no run '
                         'folder; pass the run folder or a checkpoint written with its config')
    if 'kind' in config.get('run', {}):
        require_kind(config, kind, path)
    registry_doc = ckpt.get('keypoint_registry')
    if isinstance(registry_doc, dict):
        registry = Registry.from_dict(registry_doc)
    elif run is not None and (run / 'keypoint_registry.toml').is_file():
        registry = Registry.load(run / 'keypoint_registry.toml')
    else:
        raise ValueError(f'{path}: {kind} checkpoint has no embedded keypoint_registry and sits '
                         'in no run folder; pass the run folder or a checkpoint written with '
                         'its registry')
    prov = ckpt.get('provenance')
    if not isinstance(prov, dict):
        prov = prior_provenance(run) if run is not None else {}
    return dict(config), registry, prov


def print_provenance(prov: dict, where) -> None:
    """One line naming the commit the weights were trained at, or that it is unknown."""
    if prov:
        print(f'run provenance: {str(prov.get("commit", "?"))[:12]}'
              f'{" +DIRTY" if prov.get("dirty") else ""}')
    else:
        print(f'{where}: no provenance recorded -- which commit trained these weights cannot be '
              'read back')


def _load_packaged_pose(path: Path, device='cpu', model_overrides: dict | None = None):
    """Load a self-contained pose checkpoint, whether packaged or copied from training."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f'{path}: checkpoint must be a dictionary, got {type(ckpt).__name__}')
    if ckpt.get('kind', 'pose') != 'pose':
        require_ckpt_kind(ckpt, 'pose', path)
    config, registry, prov = file_bundle(path, ckpt, 'pose')
    print_provenance(prov, path)
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
    model = build_model({**config['model'], 'video_encoder_pretrained': False},
                        n_keypoints=registry.n_keypoints)
    missing, unexpected = model.load_state_dict(state, strict=False)
    report_load('load_run', missing, unexpected, [])
    model.run_provenance = prov
    return model.to(device).eval(), config, registry, path


def peek_registry(run: Path) -> Registry:
    """Read the keypoint registry from a run folder or a self-contained pose checkpoint."""
    run = Path(run)
    if run.is_file():
        ckpt = torch.load(run, map_location='cpu', weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f'{run}: checkpoint must be a dictionary')
        return file_bundle(run, ckpt, str(ckpt.get('kind', 'pose')))[1]
    return Registry.load(run / 'keypoint_registry.toml')


def run_kind(config: dict) -> str:
    """The `[run] kind` a run folder's config carries. An absent key means `'pose'`: every run
    folder written before this key existed is a pose run, and the detector family has its own
    writer, so nothing shipped changes meaning.
    """
    return config.get('run', {}).get('kind', 'pose')


def require_kind(config: dict, want: str, where) -> None:
    """Refuse a run folder of the wrong family, BY NAME.

    The families share almost every tensor name -- a scorer IS a pose encoder plus heads -- so
    loading one as another produces a model rather than an exception. This is the same failure
    mode `gridresid_offset` is named for, so it is refused rather than warned about.
    """
    got = run_kind(config)
    if got != want:
        raise ValueError(
            f'{where}: this is a {got!r} run folder, not a {want!r} one. The two families share '
            'their encoder and decoder tensor names, so loading one as the other would build a '
            f'model with the wrong weights instead of raising. Point at a {want!r} run.')


def load_run(run: Path, checkpoint: str | None = None, device='cpu',
             model_overrides: dict | None = None):
    """(model, config, registry, checkpoint_path) from a pose run folder or pose checkpoint."""
    run = Path(run)
    if run.is_file():
        if checkpoint is not None:
            raise ValueError(f'{run}: --checkpoint selects a file inside a run folder, but --run '
                             'already names a checkpoint file')
        return _load_packaged_pose(run, device=device, model_overrides=model_overrides)
    with open(run / 'config.toml', 'rb') as f:
        config = tomllib.load(f)
    require_kind(config, 'pose', run)
    check_image_size(config)
    if model_overrides:
        config['model'] = {**config.get('model', {}), **model_overrides}
        print(f'load_run: [model] OVERRIDDEN {model_overrides} -- this is an assertion about what '
              'the checkpoint was trained with, not something read from it')
    prov = prior_provenance(run)
    if prov:
        print_provenance(prov, run)
    else:
        print(f'{run}: no provenance.toml -- this run predates commit recording, so which '
              'architecture the weights were trained under cannot be read back from the folder')
    registry = Registry.load(run / 'keypoint_registry.toml')
    path = resolve_checkpoint(run / 'checkpoints', checkpoint)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    model = build_model({**config['model'], 'video_encoder_pretrained': False},
                        n_keypoints=registry.n_keypoints)
    state = ckpt.get('model_state_eval')
    if state is None:
        print(f'{path.name}: no model_state_eval; falling back to the raw training weights')
        state = ckpt['model_state']
    missing, unexpected = model.load_state_dict(state, strict=False)
    report_load('load_run', missing, unexpected, [])
    model.run_provenance = prov
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
        report_load('warm start', missing, unexpected, dropped)
    return set(missing) | set(dropped)


def report_load(what, missing, unexpected, dropped):
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


def require_ckpt_kind(ckpt: dict, want: str, where) -> None:
    """A checkpoint's recorded kind. `_load_packaged_pose` reads `ckpt.get('kind', 'pose')`, which
    is why a scorer checkpoint written before this key existed would load as a pose one -- the
    default is a compatibility promise about OLD files, not a licence to write new ones without it.
    """
    got = ckpt.get('kind', 'pose')
    if got != want:
        raise ValueError(
            f'{where}: this is a {got!r} checkpoint, not a {want!r} one. They share almost every '
            'tensor name, so loading one as the other builds a model rather than raising.')
