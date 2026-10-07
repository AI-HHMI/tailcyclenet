#!/usr/bin/env python3
"""Package pose, scorer or detector checkpoints as portable, self-describing ``.pth`` files.

Every training checkpoint already embeds its config, keypoint registry and provenance, so any one
of them can be passed to ``scripts/infer.py`` directly. Packaging only strips the training state
(optimizer, raw iterate) and records where the file came from.

Examples:

    pixi run python scripts/package_checkpoint.py --kind pose \
        --run runs/pose --out packaged/pose.pth
    pixi run python scripts/package_checkpoint.py --kind scorer \
        --run runs/scorer --out packaged/scorer.pth
    pixi run python scripts/package_checkpoint.py --kind detector \
        --run runs/detector --out packaged/detector.pth

Both commands select the LATEST training checkpoint by default. Pass ``--checkpoint best`` only
when the validation-selected checkpoint is explicitly wanted.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import tomllib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet.checkpoints import resolve_checkpoint  # noqa: E402
from tailcyclenet.detector import resolve_detector_checkpoint  # noqa: E402
from tailcyclenet.format import Registry  # noqa: E402


def _config_path(run: Path) -> Path:
    """Find a config in either this repo's run folder or a W&B run mirror."""
    for path in (run / 'config.toml', run / 'files' / 'config.toml'):
        if path.is_file():
            return path
    raise FileNotFoundError(f'{run}: no config.toml or files/config.toml')


def _pose_run_files(run: Path) -> tuple[Path, Path, Path]:
    """Return ``(root, config, checkpoints)`` for a pose run."""
    config = _config_path(run)
    root = config.parent
    checkpoints = root / 'checkpoints'
    if not checkpoints.is_dir():
        raise FileNotFoundError(f'{root}: no checkpoints/ directory')
    registry = root / 'keypoint_registry.toml'
    if not registry.is_file():
        raise FileNotFoundError(f'{root}: no keypoint_registry.toml')
    return root, config, checkpoints


def _detector_run_dir(run: Path) -> tuple[Path, Path]:
    """Return ``(root, config)`` for a detector run or W&B run mirror."""
    config = _config_path(run)
    root = config.parent
    if not any(root.glob('detector_*.pth')):
        raise FileNotFoundError(f'{root}: no detector_*.pth files')
    return root, config


def _read_config(path: Path) -> dict[str, Any]:
    """Read and validate a run's effective TOML configuration."""
    with path.open('rb') as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError(f'{path}: config must be a TOML table')
    return config


def _load_dict(path: Path) -> dict[str, Any]:
    """Load a checkpoint dictionary on CPU without retaining unrelated device state."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f'{path}: checkpoint must be a dictionary, got '
                         f'{type(checkpoint).__name__}')
    return checkpoint


def _source_run(config: dict[str, Any], root: Path) -> str:
    """Choose the recorded W&B run id, falling back to the source folder name."""
    return str(config.get('wandb', {}).get('run_id') or root.name)


def package_pose(run: str | Path, output: str | Path,
                 checkpoint: str | None = None, kind: str = 'pose') -> Path:
    """Write an inference-only pose (or scorer) checkpoint and return its output path.

    A scorer is a pose encoder plus heads, so it packages the same way; its checkpoint-level
    output-granularity and semantic-contract metadata are carried over so `load_scorer_run` runs
    the same checks on the packaged file as on the run folder.
    """
    run = Path(run)
    output = Path(output).expanduser().resolve()
    root, config_path, checkpoints = _pose_run_files(run)
    config = _read_config(config_path)
    registry = Registry.load(root / 'keypoint_registry.toml')
    selector = {'last': 'checkpoint_last.pth', 'best': 'checkpoint_best.pth'}.get(checkpoint,
                                                               checkpoint)
    source_path = resolve_checkpoint(checkpoints, selector)
    source = _load_dict(source_path)

    eval_state = source.get('model_state_eval')
    if not isinstance(eval_state, dict) or not eval_state:
        raise ValueError(
            f'{source_path} does not contain model_state_eval; refusing to package raw '
            'training weights as inference weights')
    model_config = config.get('model')
    data_config = config.get('data')
    if not isinstance(model_config, dict) or not isinstance(data_config, dict):
        raise ValueError(f'{config_path}: pose config must contain [model] and [data] tables')

    if source.get('kind', 'pose') != kind:
        raise ValueError(f"{source_path} is a {source.get('kind', 'pose')!r} checkpoint, not a "
                         f'{kind!r} one')
    embedded = copy.deepcopy(source.get('config') or config)
    prov = source.get('provenance')
    if not isinstance(prov, dict) and (root / 'provenance.toml').is_file():
        prov = _read_config(root / 'provenance.toml')
    packaged = {
        'format_version': 1,
        'kind': kind,
        'model_state': {name: value.cpu() if isinstance(value, torch.Tensor) else value
                        for name, value in eval_state.items()},
        'config': embedded,
        'provenance': prov or {},
        'model_config': copy.deepcopy(model_config),
        'data_config': copy.deepcopy(data_config),
        'keypoint_registry': registry.to_dict(),
        'iteration': int(source.get('iteration', 0)),
        'source_run': _source_run(config, root),
        'source_checkpoint': source_path.name,
    }
    if kind == 'scorer':
        for key in ('output_granularity', 'scorer_output_granularity', 'scorer_metadata',
                    'scorer_selection_metric', 'scorer_selection_value'):
            if key in source:
                packaged[key] = copy.deepcopy(source[key])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packaged, output)
    print(f'packaged {kind} iteration {packaged["iteration"]} from {source_path} -> {output}')
    return output


def package_detector(run: str | Path, output: str | Path,
                     checkpoint: str | None = 'latest') -> Path:
    """Write a self-describing detector checkpoint and return its output path."""
    run = Path(run)
    output = Path(output).expanduser().resolve()
    root, config_path = _detector_run_dir(run)
    config = _read_config(config_path)
    source_path = resolve_detector_checkpoint(root, checkpoint=checkpoint)
    packaged = _load_dict(source_path)
    packaged = dict(packaged)
    packaged['format_version'] = 1
    packaged['kind'] = 'detector'
    packaged['config'] = copy.deepcopy(packaged.get('config') or config)
    if not isinstance(packaged.get('provenance'), dict) and (root / 'provenance.toml').is_file():
        packaged['provenance'] = _read_config(root / 'provenance.toml')
    packaged['source_run'] = _source_run(config, root)
    packaged['source_checkpoint'] = source_path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packaged, output)
    print(f'packaged detector iteration {packaged.get("iteration", "?")} '
          f'from {source_path} -> {output}')
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit pose/scorer/detector packaging CLI."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', required=True, choices=('pose', 'scorer', 'detector'))
    parser.add_argument('--run', required=True, type=Path,
                        help='tailcyclenet run folder or a W&B run folder')
    parser.add_argument('--out', required=True, type=Path, help='destination .pth path')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint filename or selector. Defaults to the LAST checkpoint: '
                             'the latest numbered checkpoint for every kind.')
    return parser


def main() -> None:
    """Parse arguments and package the requested model kind."""
    args = build_parser().parse_args()
    if args.kind in ('pose', 'scorer'):
        package_pose(args.run, args.out, args.checkpoint, kind=args.kind)
    else:
        package_detector(args.run, args.out, args.checkpoint or 'latest')


if __name__ == '__main__':
    main()
