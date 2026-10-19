"""Helpers shared by the converters in this folder.

Each converter imports these as ``from _common import ...``; running ``python scripts/convert/x.py``
puts this folder on ``sys.path``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from tailcyclenet import format as fmt


def add_output_args(ap: argparse.ArgumentParser, default_out: Path | None = None, *,
                    dry_run: bool = True, validate: bool = True,
                    image_check: bool = False) -> None:
    """Add the shared ``--out``/``--clean``/``--dry-run``/``--validate`` options.

    Inputs: the parser; ``default_out`` (None makes ``--out`` required); which optional flags to
    add (``image_check`` adds ``--no-image-check``).
    Side effects: registers the options on ``ap``.
    """
    if default_out is None:
        ap.add_argument('--out', type=Path, required=True)
    else:
        ap.add_argument('--out', type=Path, default=default_out)
    ap.add_argument('--clean', action='store_true', help='remove the output first')
    if dry_run:
        ap.add_argument('--dry-run', action='store_true', help='report, write nothing')
    if validate:
        ap.add_argument('--validate', action='store_true', help='validate after writing')
    if image_check:
        ap.add_argument('--no-image-check', action='store_true',
                        help='skip opening one image per group during validation')


def validate_root(root: Path, label: str = 'validate', check_images: bool = True) -> int:
    """Validate a written dataset root and print every finding.

    Inputs: the root; a label for the summary line; whether to open one image per group.
    Outputs: the number of hard errors (warnings are printed but not counted).
    """
    errs = fmt.validate_dataset(fmt.load_dataset(root), check_images=check_images)
    hard = [e for e in errs if 'WARNING' not in e]
    for e in errs:
        print(('  WARN ' if 'WARNING' in e else '  FAIL ') + e)
    print(f'{label}: {len(hard)} error(s), {len(errs) - len(hard)} warning(s)')
    return len(hard)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Hash one file without loading it all into memory."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Atomically write one decoded RGB frame, compacting a grayscale frame to an L PNG."""
    from PIL import Image
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
        image = Image.fromarray(rgb[..., 0], mode='L')
    else:
        image = Image.fromarray(rgb, mode='RGB')
    tmp = path.parent / f'.{path.name}.tmp'
    image.save(tmp, format='PNG', compress_level=1)
    os.replace(tmp, path)


def valid_png(path: Path, width: int, height: int) -> bool:
    """Fully decode an existing PNG and check its dimensions and pixel mode (L or RGB)."""
    from PIL import Image
    try:
        with Image.open(path) as image:
            if image.size != (width, height) or image.mode not in {'L', 'RGB'}:
                return False
            image.load()
        return True
    except (OSError, ValueError):
        return False


def link_numbered_frames(view_dir: Path, sources: list[Path], *, relative: bool = False,
                         replace: bool = False) -> None:
    """Symlink ``sources`` into ``view_dir`` as ``000000<suffix>``, ``000001<suffix>``, ...

    Inputs: the camera's pixel directory; the source files in frame order; ``relative`` writes
    links relative to ``view_dir``; ``replace`` overwrites an existing entry, otherwise an entry
    that does not already point at its source raises.
    Side effects: creates ``view_dir`` and the links. A directory that did not exist skips the
    per-frame existence check (a fresh farm is empty by construction).
    """
    fresh = not view_dir.exists()
    view_dir.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(sources):
        dst = view_dir / f'{i:06d}{source.suffix.lower()}'
        target = Path(os.path.relpath(source, view_dir)) if relative else source
        if fresh:
            os.symlink(os.fspath(target), os.fspath(dst))
            continue
        if dst.exists() or dst.is_symlink():
            if replace:
                dst.unlink()
            elif not dst.is_symlink() or dst.resolve() != source.resolve():
                raise RuntimeError(f'pixel link collision: {dst}')
            else:
                continue
        dst.symlink_to(target)
