"""`Registry`: the append-only keypoint-name -> embedding-id map shared by training and inference."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import FormatError
from .discovery import Dataset


@dataclass(frozen=True)
class Registry:
    """Global keypoint identity across one or more datasets.

    A collection prefixes names with the dataset folder name so one embedding table serves every
    dataset. Ids are append-only against a `base`: a later run keeps every old id, and the
    learned embedding rows behind them survive warm start.
    """
    names: tuple[str, ...]
    datasets: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def n_keypoints(self) -> int:
        """Number of global keypoint ids."""
        return len(self.names)

    def ids_for_dataset(self, dataset: str) -> np.ndarray:
        """Global ids in the DATASET's own name order, i.e. aligned to `Dataset.names`."""
        for name, ids in self.datasets:
            if name == dataset:
                return np.asarray(ids, dtype=np.int64)
        raise KeyError(f'{dataset!r} is not in this registry: {[d for d, _ in self.datasets]}')

    def local_names(self, dataset: str) -> list[str]:
        """That dataset's own keypoint names, un-prefixed, in registry-id order."""
        got = [self.names[i] for i in self.ids_for_dataset(dataset)]
        p = f'{dataset}-'
        return [n[len(p):] for n in got] if all(n.startswith(p) for n in got) else got

    def ids_for(self, dataset: str, names) -> np.ndarray:
        """Global ids ALIGNED TO `names` -- the keypoint axis the caller actually holds.

        `names` is required, not defaulted: a per-dataset id vector applied to a session that
        declares the same names in a different order is a silent relabel. A session may use any
        SUBSET, in any order; an unknown name is an error, loudly.
        """
        ids = self.ids_for_dataset(dataset)
        local = self.local_names(dataset)
        names = list(names)
        if names == local:
            return ids
        ix = {n: i for i, n in enumerate(local)}
        unknown = [n for n in names if n not in ix]
        if unknown:
            raise FormatError(
                f'{dataset}: keypoints {unknown} are not in this registry. The registry holds '
                f'{len(local)} names for this dataset; a session may reorder them or use a '
                f'subset, but not invent one. Retrain, or fix the session\'s `names`.')
        return ids[[ix[n] for n in names]]

    @staticmethod
    def _base_spelling(base: 'Registry | None', ds: 'Dataset') -> str | None:
        """How `base` spells this dataset's keypoints: 'prefixed', 'unprefixed', or None.

        Inputs: base -- the registry being appended to, or None; ds -- the dataset to look up.
        Outputs: the spelling the base recorded, or None when the base does not name the dataset.
        Side effects: none.
        """
        recorded = dict(base.datasets).get(ds.name) if base is not None else None
        if not recorded:
            return None
        if all(base.names[i].startswith(f'{ds.name}-') for i in recorded):
            return 'prefixed'
        return 'unprefixed'

    @classmethod
    def build(cls, datasets: list[Dataset], base: 'Registry | None' = None) -> 'Registry':
        """Build a registry covering `datasets`, appending to `base` when given.

        Inputs: datasets -- the datasets to cover.
                base -- an existing registry whose ids must be preserved.
        Outputs: a Registry whose ids are append-only against `base`.
        Side effects: raises FormatError if an existing id would move.

        A dataset already named by `base` keeps the spelling recorded when it was first added.
        This remains true even after `base` grows, preventing duplicate identities and preserving
        append-only ids.
        """
        prefix = len(datasets) > 1 or (base is not None and len(base.datasets) > 1)
        names = list(base.names) if base else []
        index = {n: i for i, n in enumerate(names)}
        out = dict(base.datasets) if base else {}
        for ds in datasets:
            spelling = cls._base_spelling(base, ds)
            ds_prefix = prefix if spelling is None else spelling == 'prefixed'
            ids = []
            for local in ds.names:
                full = f'{ds.name}-{local}' if ds_prefix else local
                if full not in index:
                    index[full] = len(names)
                    names.append(full)
                ids.append(index[full])
            if ds.name in out and tuple(ids) != tuple(out[ds.name]):
                raise FormatError(
                    f'{ds.name}: keypoint ids changed against the base registry '
                    f'({out[ds.name]} -> {tuple(ids)}); names must be append-only')
            out[ds.name] = tuple(ids)
        return cls(names=tuple(names), datasets=tuple(sorted(out.items())))

    def to_dict(self) -> dict:
        """Return the registry's portable, TOML-compatible representation."""
        return {'names': list(self.names),
                'datasets': {name: list(ids) for name, ids in self.datasets}}

    @classmethod
    def from_dict(cls, doc: dict) -> 'Registry':
        """Build a registry from its portable dictionary representation."""
        return cls(names=tuple(doc['names']),
                   datasets=tuple(sorted((k, tuple(v)) for k, v in doc['datasets'].items())))

    def save(self, path: Path) -> None:
        """Write the registry to a TOML file."""
        import toml
        path.write_text(toml.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> 'Registry':
        """Read a registry back from a TOML file."""
        with open(path, 'rb') as f:
            return cls.from_dict(tomllib.load(f))
