"""The DeepFly3D source conventions shared across converter generations.

The condition names and the family-grouped split rule were fixed by the retired v1/v2 converters
(deleted 2026-10-19; recover them from git history); `convert_deepfly3d_v3.py` keeps using both so
its splits stay the v2 family policy.
"""
from __future__ import annotations

import re

SOURCE_CONDITIONS = (
    'aDN-GAL4_Control',
    'MDN-GAL4_Control',
    'aDN-GAL4_UAS-CsChrimson',
    'MDN-GAL4_UAS-CsChrimson',
)


def family_id(condition: str, archive: str) -> str:
    """Derive the stable split-group identifier from a recording archive name."""
    m = re.match(r'^(\d+)_.*?_Fly(\d+)_', archive)
    if not m:
        raise ValueError(f'cannot derive fly family from {archive!r}')
    return f'{condition}__{m.group(1)}__Fly{m.group(2)}'


def split_map(records) -> dict[str, str]:
    """Assign each recording family deterministically to train, val, or test."""
    families = sorted({r.family for r in records})
    result = {}
    for i, family in enumerate(families):
        result[family] = ('train', 'val', 'test')[2 if i % 10 == 9 else 1 if i % 10 == 8 else 0]
    return result
