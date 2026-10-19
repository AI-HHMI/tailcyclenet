"""Find sessions on disk: a dataset root (`Dataset`), a folder of roots, and the split filter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from . import FormatError, SPLITS
from .session import Session


@dataclass
class Dataset:
    name: str
    root: Path
    # split -> sessions
    sessions: dict[str, list[Session]]

    @property
    def names(self) -> list[str]:
        """The root's keypoint axis: the UNION of its sessions' names, in load order.

        A session may reorder or subset the root's names; ids are handed out against this list
        and remapped per session by `Registry.ids_for`. Order is deterministic, and equals the
        first session's list when all sessions agree, so no existing registry's ids move.
        """
        names: list[str] = []
        seen: set[str] = set()
        for sess in self.all_sessions():
            for n in sess.names:
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        if not names:
            raise FormatError(f'{self.root}: no sessions')
        return names

    def all_sessions(self) -> list[Session]:
        """Every session across every split, in load order."""
        return [s for group in self.sessions.values() for s in group]


def _is_dataset_root(path: Path) -> bool:
    """True when the directory holds at least one train/val/test split."""
    return any((path / s).is_dir() for s in SPLITS)


def load_dataset(root: Path, split: str | None = None) -> Dataset:
    """Load sessions under a dataset root, optionally restricted to one split.

    Inputs: root -- the dataset root directory; split -- one requested split, or ``None`` for
        every split (the format/discovery behavior).
    Outputs: a Dataset; raises FormatError when the requested split directory does not exist.
    """
    root = Path(root)
    if split is not None and split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}, got {split!r}')
    requested = (split,) if split is not None else SPLITS
    sessions: dict[str, list[Session]] = {}
    for name in requested:
        d = root / name
        if not d.is_dir():
            continue
        sessions[name] = [Session.load(p) for p in sorted(d.iterdir())
                          if (p / 'session.toml').exists()]
    if not sessions:
        wanted = f'{split}/' if split is not None else 'train/val/test'
        raise FormatError(f'{root}: no {wanted} directory')
    return Dataset(name=root.name, root=root, sessions=sessions)


def sessions_for(path: Path, split: str) -> tuple[str, list['Session']]:
    """(dataset_name, sessions) from either ONE session directory or a dataset root.

    A path is a session iff it holds `session.toml`; that is the format's own marker, and it is
    the one rule both `scripts/infer.py` and `scripts/eval.py` need.
    """
    path = Path(path)
    if (path / 'session.toml').exists():
        return path.parent.parent.name, [Session.load(path)]
    ds = load_dataset(path)
    return ds.name, ds.sessions.get(split, [])


def load_datasets(path: Path, split: str | None = None) -> list[Dataset]:
    """One dataset root, or a folder whose children are dataset roots.

    With ``split`` set, only sessions in that split are discovered. This is used by training
    loaders so another split is not even parsed into ``Session`` objects; omitting it preserves
    the format/discovery behavior that callers such as validation and rendering need.

    This is the whole of the "point at a dataset OR at a folder of datasets" rule: the presence
    of a `train/` directory is what distinguishes them.
    """
    path = Path(path)
    if split is not None and split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}, got {split!r}')
    if _is_dataset_root(path):
        return [load_dataset(path, split=split)]
    children = sorted(
        p for p in path.iterdir()
        if p.is_dir() and _is_dataset_root(p)
        and (split is None or (p / split).is_dir()))
    if not children:
        suffix = f' with a {split}/ split' if split is not None else ''
        raise FormatError(f'{path}: neither a dataset root nor a folder of dataset roots{suffix}')
    return [load_dataset(p, split=split) for p in children]
