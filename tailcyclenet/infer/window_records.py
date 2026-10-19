"""Per-window prediction records (`--window-predictions`): the record fields and their writer."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def logit_probability(value):
    """Stable scalar logit-to-probability conversion for optional window records."""
    return float(1.0 / (1.0 + np.exp(-np.clip(float(value), -80.0, 80.0))))


def window_record_base(gid, window, start, stop, slot, animal_id, frame, bodypart,
                        record_type, camera=None):
    """Common long-form sidecar keys; absent dimensions are explicit nulls."""
    return {'record_type': record_type, 'group_id': str(gid), 'frame': frame,
            'window': int(window), 'window_start': int(start), 'window_stop': int(stop),
            'slot': int(slot), 'animal_id': str(animal_id), 'camera': camera,
            'bodypart': bodypart}


class WindowPredictionWriter:
    """Typed, incrementally-written union schema for optional per-window prediction records."""

    STRING_FIELDS = ('record_type', 'group_id', 'animal_id', 'camera', 'bodypart', 'status',
                     'outcome')
    DICT_FIELDS = ('group_id', 'animal_id', 'camera', 'bodypart', 'status')
    INT_FIELDS = ('frame', 'window', 'window_start', 'window_stop', 'slot', 'box_prompt_cams')
    BOOL_FIELDS = ('gated',)
    FLOAT_FIELDS = (
        'x', 'y', 'z', 'triangulated_x', 'triangulated_y', 'triangulated_z', 'score',
        'score_logit', 'visibility_logit', 'visibility_probability', 'confidence_logit',
        'confidence_probability', 'box_agree', 'x0', 'y0', 'x1', 'y1',
        'crop_x0', 'crop_y0', 'crop_x1', 'crop_y1', 'refined_x0', 'refined_y0',
        'refined_x1', 'refined_y1')
    QUALITY_FIELDS = ('quality_score', 'quality_precision')

    def __init__(self, path: Path, *, quality_enabled=False):
        """Open a typed Parquet writer; status fields use a stable dictionary type."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = [pa.field(name, pa.string()) for name in self.STRING_FIELDS]
        fields.extend(pa.field(name, pa.int32()) for name in self.INT_FIELDS)
        self.float_fields = (self.FLOAT_FIELDS + self.QUALITY_FIELDS
                             if quality_enabled else self.FLOAT_FIELDS)
        fields.extend(pa.field(name, pa.float32()) for name in self.float_fields)
        fields.extend(pa.field(name, pa.bool_()) for name in self.BOOL_FIELDS)
        fields = [pa.field(f.name, pa.dictionary(pa.int32(), pa.string())
                           if f.name in self.DICT_FIELDS else f.type) for f in fields]
        self.schema = pa.schema(fields, metadata={b'tailcyclenet.schema_version': b'1'})
        self._writer = pq.ParquetWriter(path, self.schema, compression='zstd')
        self._closed = False

    def write(self, records) -> None:
        """Normalize and append prediction records; empty batches are ignored."""
        import pyarrow as pa

        records = list(records)
        if not records:
            return
        normalized = []
        for record in records:
            row = dict(record)
            for axis in ('x0', 'y0', 'x1', 'y1'):
                key = f'crop_refined_{axis}'
                if key in row:
                    row[f'refined_{axis}'] = row.pop(key)
            normalized.append(row)
        columns = {}
        for name in self.STRING_FIELDS:
            values = [row.get(name) for row in normalized]
            arr = pa.array(values, type=pa.string())
            columns[name] = arr.dictionary_encode() if name in self.DICT_FIELDS else arr
        for name in self.INT_FIELDS:
            columns[name] = pa.array([row.get(name) for row in normalized], type=pa.int32())
        for name in self.BOOL_FIELDS:
            columns[name] = pa.array([row.get(name) for row in normalized], type=pa.bool_())
        for name in self.float_fields:
            values = [row.get(name) for row in normalized]
            values = [None if v is None or not np.isfinite(v) else float(v) for v in values]
            columns[name] = pa.array(values, type=pa.float32())
        self._writer.write_table(pa.table(columns, schema=self.schema))

    def close(self) -> None:
        """Close the underlying Parquet writer once."""
        if not self._closed:
            self._writer.close()
            self._closed = True
