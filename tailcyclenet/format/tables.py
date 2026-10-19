"""Parquet column helpers (dictionary codes, ints, floats, per-group row lookup, a memoising
`Columns` view) and the streaming `TableWriter` / `write_table` used by every table writer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import FormatError


class Columns:
    """A parquet table whose decoded columns are memoised.

    `Session.labels` scatters one group at a time, and every group used to re-decode every
    column of the WHOLE session table -- O(groups x rows), ~30 s on a 350-group session. This
    decodes each column once and indexes rows by group once. Callers must not mutate what the
    helpers return (all of them slice with `[sel]`, which copies).
    """

    def __init__(self, table: pa.Table):
        """Wrap `table`; nothing is decoded until asked for."""
        self.table = table
        self.column_names = table.column_names
        self._memo: dict = {}

    def __len__(self) -> int:
        """Row count of the wrapped table."""
        return len(self.table)

    def column(self, col: str):
        """The wrapped table's raw column."""
        return self.table.column(col)

    def memo(self, key, fn):
        """`fn()` the first time `key` is asked for, the stored value after."""
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]


def group_rows(table, gid: str) -> np.ndarray | None:
    """Ascending row indices of group `gid`, or None when the table has no such group."""
    if isinstance(table, Columns):
        def build():
            """group id -> its ascending row indices, from one stable argsort of the codes."""
            codes, values = column_codes(table, 'group_id')
            order = np.argsort(codes, kind='stable')
            bounds = np.searchsorted(codes[order], np.arange(len(values) + 1))
            return {v: order[bounds[i]:bounds[i + 1]] for i, v in enumerate(values)}
        rows = table.memo(('rows',), build).get(gid)
        return rows if rows is not None and rows.size else None
    gcodes, gvals = column_codes(table, 'group_id')
    if gid not in gvals:
        return None
    sel = np.flatnonzero(gcodes == gvals.index(gid))
    return sel if sel.size else None


def column_ints(table, col: str) -> np.ndarray:
    """An integer column as int64 numpy."""
    if isinstance(table, Columns):
        return table.memo(('ints', col), lambda: column_ints(table.table, col))
    return table.column(col).combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)


def column_codes(table: pa.Table, col: str) -> tuple[np.ndarray, list[str]]:
    """(int32 codes, distinct values) for a string column. Dictionary-encodes if it is not."""
    if isinstance(table, Columns):
        return table.memo(('codes', col), lambda: column_codes(table.table, col))
    arr = table.column(col).combine_chunks()
    if not pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_encode()
    return arr.indices.to_numpy(zero_copy_only=False).astype(np.int32), arr.dictionary.to_pylist()


def remap_codes(codes: np.ndarray, values: list[str], vocab: dict[str, int],
           what: str, where: str, sel: np.ndarray | None = None) -> np.ndarray:
    """Translate dictionary codes into `vocab` indices, restricted to `sel` rows.

    A parquet dictionary is per FILE, not per group, so values other groups use must not be
    validated here.
    """
    lut = np.array([vocab.get(v, -1) for v in values], dtype=np.int32)
    codes = codes if sel is None else codes[sel]
    out = lut[codes]
    if (out < 0).any():
        unknown = sorted({values[c] for c in np.unique(codes[out < 0])})
        raise FormatError(f'{where}: unknown {what} {unknown[0]!r}')
    return out


def column_floats(table: pa.Table, col: str, n: int) -> np.ndarray:
    """A float64 column as numpy, nulls as NaN. Missing column -> all NaN."""
    if col not in table.column_names:
        return np.full(n, np.nan)
    if isinstance(table, Columns):
        return table.memo(('floats', col), lambda: column_floats(table.table, col, n))
    return table.column(col).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)


def _arrow(rows: dict[str, np.ndarray], dict_cols: tuple[str, ...]) -> pa.Table:
    """One chunk of tidy-long rows as an arrow table. Non-finite floats become NULL, which is
    what "empty" means in parquet for a `missing` row's x,y. An empty object array would
    otherwise infer pyarrow's `null` type; only a zero-row table gets here (an empty mode
    table, which rule 6 requires to exist).
    """
    arrays, names = [], []
    for name, values in rows.items():
        values = np.asarray(values) if not isinstance(values, list) else values
        if isinstance(values, np.ndarray) and values.dtype == object and values.size == 0:
            arr = pa.array([], pa.string())
        elif isinstance(values, np.ndarray) and values.dtype.kind == 'f':
            arr = pa.array(values, mask=~np.isfinite(values))
        else:
            arr = pa.array(values)
        if name in dict_cols:
            arr = arr.dictionary_encode()
        arrays.append(arr)
        names.append(name)
    return pa.table(arrays, names=names)


class TableWriter:
    """A tidy-long table written in chunks, so a producer need never hold the whole thing.

    Dictionary columns fix their schema on the FIRST chunk (readers here expect `DictionaryArray`),
    and rows buffer to `chunk_rows` so a long clip does not produce thousands of tiny row groups.
    """

    def __init__(self, path: Path, dict_cols: tuple[str, ...] = (), chunk_rows: int = 250_000):
        """Set the output path, the dictionary-encoded columns and the chunk size."""
        self.path, self.dict_cols, self.chunk_rows = Path(path), dict_cols, int(chunk_rows)
        self._w = None
        self._buf: list[pa.Table] = []
        self._n = 0

    def write(self, rows: dict[str, np.ndarray]) -> None:
        """Buffer one chunk of rows, flushing once `chunk_rows` is reached."""
        t = _arrow(rows, self.dict_cols)
        if not len(t):
            return
        self._buf.append(t)
        self._n += len(t)
        if self._n >= self.chunk_rows:
            self._flush()

    def _flush(self) -> None:
        """Write the buffered chunks, fixing the file's schema on the first flush."""
        if not self._buf:
            return
        t = pa.concat_tables(self._buf)
        if self._w is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._w = pq.ParquetWriter(self.path, t.schema, compression='zstd')
        else:
            t = t.cast(self._w.schema)
        self._w.write_table(t)
        self._buf, self._n = [], 0

    def close(self) -> None:
        """Flush and close the underlying parquet writer."""
        self._flush()
        if self._w is not None:
            self._w.close()
            self._w = None

    def __enter__(self):
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc):
        """Close the writer when the `with` block exits."""
        self.close()


def write_table(path: Path, rows: dict[str, np.ndarray], dict_cols: tuple[str, ...]) -> None:
    """Write a tidy-long table in one shot. See `TableWriter` for the chunked form."""
    pq.write_table(_arrow(rows, dict_cols), path, compression='zstd')


DICT_COLS = ('group_id', 'animal_id', 'camera', 'bodypart', 'status')


def invert_enum(enum: dict[str, int]) -> dict[int, str]:
    """Invert an enum dict so int codes map back to their names."""
    return {v: k for k, v in enum.items()}
