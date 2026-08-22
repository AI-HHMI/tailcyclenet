#!/usr/bin/env python
"""Build a `-combined` dataset root: one symlink per session of each source root.

    pixi run python scripts/combine_roots.py --out <base>/rat-city-combined <src>...

Holds no pixels and no tables -- symlinks only, each session's split preserved as the parent
directory. A colliding session name gets its source root's folder name suffixed on BOTH sides,
detected root-wide (a session may span splits). Keypoint name sets are verified across all
sources before anything is written. Re-running is a no-op; anything unexpected is refused unless
`--force` (symlinks only).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tailcyclenet import format as fmt


def sessions(root: Path) -> dict[str, list[Path]]:
    """{split: [session dir, ...]} for one dataset root, in `load_dataset` order."""
    out = {s: sorted(p for p in (root / s).iterdir() if (p / 'session.toml').exists())
           for s in fmt.SPLITS if (root / s).is_dir()}
    out = {k: v for k, v in out.items() if v}
    if not out:
        raise SystemExit(f'{root}: no train/val/test directory with sessions in it')
    return out


def plan(roots: list[Path]) -> dict[str, dict[str, Path]]:
    """{split: {link name: source session dir}} -- the whole collision rule, see the docstring."""
    per_root = [sessions(r) for r in roots]
    seen: Counter[str] = Counter()
    for s in per_root:
        seen.update({p.name for ss in s.values() for p in ss})   # a SET: one vote per root
    collide = {n for n, c in seen.items() if c > 1}

    out: dict[str, dict[str, Path]] = {s: {} for s in fmt.SPLITS}
    for root, s in zip(roots, per_root):
        for split, ss in s.items():
            for p in ss:
                name = f'{p.name}__{root.name}' if p.name in collide else p.name
                if name in out[split]:
                    raise SystemExit(f'{split}/{name}: two sources claim this name')
                out[split][name] = p
    return {s: links for s, links in out.items() if links}


def check_names(roots: list[Path]) -> list[str]:
    """The union axis, or refuse: a keypoint-axis disagreement is invisible downstream."""
    ref: tuple[Path, list[str]] | None = None
    for root in roots:
        for s in fmt.load_dataset(root).all_sessions():
            if ref is None:
                ref = (s.path, list(s.names))
            elif set(s.names) != set(ref[1]):
                raise SystemExit(f'keypoint names disagree, refusing to build:\n'
                                 f'  {ref[0]}: {ref[1]}\n  {s.path}: {s.names}')
            elif list(s.names) != ref[1]:
                print(f'note: {s.path} reorders the keypoint axis (legal, resolved by name)')
    assert ref is not None
    return ref[1]


def build(out_root: Path, links_by_split: dict[str, dict[str, Path]], force: bool) -> tuple[int, int]:
    """(links written, stale links removed). Idempotent; never deletes anything but a symlink."""
    out_root.mkdir(parents=True, exist_ok=True)
    extra = sorted(set(os.listdir(out_root)) - set(links_by_split))
    if extra:
        raise SystemExit(f'{out_root}: unexpected entries {extra} -- remove them by hand')

    made = removed = 0
    for split, links in links_by_split.items():
        d = out_root / split
        d.mkdir(exist_ok=True)
        for stale in sorted(set(os.listdir(d)) - set(links)):
            p = d / stale
            if not force:
                raise SystemExit(f'{p}: not in the plan; pass --force to remove it')
            if not p.is_symlink():
                raise SystemExit(f'{p}: not a symlink -- refusing to delete real data')
            p.unlink()
            removed += 1
        for name, target in links.items():
            # RELATIVE, like the hand-built roots: the whole datasets tree stays movable.
            rel = os.path.relpath(target, d)
            p = d / name
            if p.is_symlink():
                if os.readlink(p) == rel:
                    continue
                if not force:
                    raise SystemExit(f'{p} -> {os.readlink(p)}: points elsewhere; pass --force')
                p.unlink()
                removed += 1
            elif p.exists():
                raise SystemExit(f'{p}: exists and is not a symlink -- refusing to clobber')
            p.symlink_to(rel, target_is_directory=True)
            made += 1
    return made, removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('roots', nargs='+', type=Path, help='source dataset roots')
    ap.add_argument('--out', type=Path, required=True, help='output root, e.g. rat-city-combined')
    ap.add_argument('--force', action='store_true',
                    help='replace/remove stale SYMLINKS in an existing output root')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    roots = [r.resolve() for r in args.roots]
    if len(roots) < 2:
        raise SystemExit('give at least two source roots')
    if len({r.name for r in roots}) != len(roots):
        raise SystemExit('source roots must have distinct folder names (the collision suffix)')

    names = check_names(roots)
    links_by_split = plan(roots)
    for split, links in links_by_split.items():
        per_root = Counter(str(p.parent.parent.name) for p in links.values())
        print(f'{split}: {len(links)} sessions  ' + '  '.join(f'{k}={v}' for k, v in per_root.items()))
    suffixed = [n for links in links_by_split.values() for n in links if '__' in n]
    print(f'keypoints: {names}')
    print(f'collisions disambiguated: {len(suffixed)}' +
          (f' ({", ".join(sorted(set(suffixed)))})' if suffixed else ''))
    if args.dry_run:
        return 0

    made, removed = build(args.out.resolve(), links_by_split, args.force)
    print(f'{args.out}: {made} symlinks written, {removed} stale removed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
