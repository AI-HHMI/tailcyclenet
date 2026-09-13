"""Phase-0 behavior locks for the pandas-to-Polars table migration.

The tests deliberately import both table engines inside each test.  Pandas is the current
reference; the Polars expressions show the explicit null, ordering, and row-preservation
choices that a port must make rather than relying on backend defaults.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _backends() -> tuple[Any, Any]:
    """Load optional table engines only for tests which need them."""
    pandas = pytest.importorskip("pandas")
    polars = pytest.importorskip("polars")
    return pandas, polars


def _canonical(value: Any) -> Any:
    """Use one representation for pandas NaN, Polars null, and scalar extension values."""
    if value is None or value.__class__.__name__ == "NAType":
        return None
    try:
        if bool(math.isnan(value)):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if item is not None and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _canonical(item())
        except (TypeError, ValueError):
            pass
    return value


def _records(frame: Any, *, pandas: bool) -> list[dict[str, Any]]:
    rows = frame.to_dict(orient="records") if pandas else frame.to_dicts()
    return [{str(key): _canonical(value) for key, value in row.items()} for row in rows]


def test_nan_and_null_are_distinct_until_a_policy_is_chosen():
    pandas, polars = _backends()
    pandas_frame = pandas.DataFrame({"value": [1.0, float("nan"), None]})
    polars_frame = polars.DataFrame({"value": [1.0, float("nan"), None]})

    # pandas' floating column collapses None and NaN into one missing value.  Polars keeps
    # IEEE NaN and Arrow null separate, so a port must say which predicate it intends.
    assert pandas_frame["value"].isna().tolist() == [False, True, True]
    assert polars_frame["value"].is_null().to_list() == [False, False, True]
    assert polars_frame["value"].is_nan().fill_null(False).to_list() == [False, True, False]
    assert polars_frame.select(
        (polars.col("value").is_null() | polars.col("value").is_nan()).alias("missing")
    )["missing"].to_list() == [False, True, True]


def test_null_last_sort_is_stable_and_explicit_ties_are_preserved():
    pandas, polars = _backends()
    rows = [
        {"row_id": 0, "score": 0.2}, {"row_id": 1, "score": None},
        {"row_id": 2, "score": 0.2}, {"row_id": 3, "score": 0.1},
        {"row_id": 4, "score": None}, {"row_id": 5, "score": 0.2},
    ]
    pandas_frame = pandas.DataFrame(rows)
    polars_frame = polars.DataFrame(rows, schema_overrides={"score": polars.Float64})
    expected = [3, 0, 2, 5, 1, 4]
    assert pandas_frame.sort_values("score", na_position="last", kind="stable")["row_id"].tolist() == expected
    assert polars_frame.sort("score", nulls_last=True, maintain_order=True)["row_id"].to_list() == expected
    assert [row for row in expected if row in {0, 2, 5}] == [0, 2, 5]

    # This is the known QC tie: without secondary keys, pandas' stable null bucket follows
    # grouped-key order while Polars' null bucket follows hash/group encounter order.
    rank_rows = []
    for group, keypoint, score in [
        ("g0", "tail", None), ("g2", "nose", float("nan")),
        ("g0", "paw", float("nan")), ("g1", "tail", None),
    ]:
        rank_rows.append({"group": group, "keypoint": keypoint, "score": score})
    p_rank = (pandas.DataFrame(rank_rows).groupby(["group", "keypoint"], sort=True, dropna=True)
              .agg(median=("score", "median")).reset_index()
              .sort_values("median", na_position="last", kind="stable"))
    l_rank = (polars.DataFrame(rank_rows, schema_overrides={"score": polars.Float64})
              .group_by(["group", "keypoint"], maintain_order=True)
              .agg(polars.col("score").median().alias("median"))
              .sort("median", nulls_last=True, maintain_order=True))
    assert [(row["group"], row["keypoint"]) for row in _records(p_rank, pandas=True)] == [
        ("g0", "paw"), ("g0", "tail"), ("g1", "tail"), ("g2", "nose")
    ]
    assert [(row["group"], row["keypoint"]) for row in _records(l_rank, pandas=False)] != [
        ("g0", "paw"), ("g0", "tail"), ("g1", "tail"), ("g2", "nose")
    ]

    # Secondary keys are the portable fix.  Nulls are sent to an explicit +inf sort bucket,
    # while the original aggregate remains null in the result.
    p_fixed = p_rank.sort_values(["median", "group", "keypoint"], na_position="last", kind="stable")
    l_fixed = (l_rank.with_columns(
        polars.col("median").fill_nan(None).fill_null(float("inf")).alias("_sort_metric")
    ).sort(["_sort_metric", "group", "keypoint"], maintain_order=True).drop("_sort_metric"))
    assert _records(p_fixed, pandas=True) == _records(l_fixed, pandas=False)


def test_group_key_order_is_not_an_accidental_groupby_default():
    pandas, polars = _backends()
    rows = [{"dataset": "d", "group": "b", "n": 1},
            {"dataset": "d", "group": "a", "n": 2},
            {"dataset": "d", "group": "b", "n": 3},
            {"dataset": "d", "group": "c", "n": 4}]
    p = pandas.DataFrame(rows).groupby(["dataset", "group"], sort=True, dropna=True).size().reset_index(name="n")
    polars_frame = (polars.DataFrame(rows).group_by(["dataset", "group"], maintain_order=True)
                    .agg(polars.len().alias("n")).sort(["dataset", "group"], maintain_order=True))
    assert _records(p, pandas=True) == _records(polars_frame, pandas=False)
    assert p["group"].tolist() == ["a", "b", "c"]


def test_median_nunique_and_isin_have_explicit_missing_policies():
    pandas, polars = _backends()
    values = [1.0, float("nan"), 1.0, 3.0, None]
    p = pandas.Series(values, name="value")
    polars_frame = polars.DataFrame({"value": values}).with_columns(
        polars.col("value").fill_nan(None).alias("value")
    )
    assert p.median() == 1.0
    assert polars_frame["value"].median() == 1.0
    assert p.nunique(dropna=True) == 2
    assert p.nunique(dropna=False) == 3
    assert polars_frame["value"].drop_nulls().n_unique() == 2
    assert polars_frame["value"].n_unique() == 3

    assert p.isin([1.0, 3.0]).tolist() == [True, False, True, True, False]
    assert polars_frame.select(polars.col("value").is_in([1.0, 3.0]).fill_null(False))["value"].to_list() == [
        True, False, True, True, False
    ]
    # pandas' float column treats both spellings of missing as NaN for membership.  Polars needs
    # that compatibility policy stated explicitly; plain is_in would preserve null as unknown.
    assert p.isin([float("nan")]).tolist() == [False, True, False, False, True]
    missing_match = (polars.col("value").is_nan().fill_null(False)
                     | polars.col("value").is_null()).alias("match")
    assert polars_frame.select(missing_match)["match"].to_list() == [False, True, False, False, True]


def test_drop_duplicates_keeps_the_first_row_after_nan_normalization():
    pandas, polars = _backends()
    rows = [{"row_id": 0, "key": 1.0}, {"row_id": 1, "key": float("nan")},
            {"row_id": 2, "key": float("nan")}, {"row_id": 3, "key": 2.0},
            {"row_id": 4, "key": None}]
    p = pandas.DataFrame(rows).drop_duplicates(subset=["key"], keep="first")
    polars_frame = (polars.DataFrame(rows, schema_overrides={"key": polars.Float64})
                    .with_columns(polars.col("key").fill_nan(None))
                    .unique(subset=["key"], keep="first", maintain_order=True))
    assert p["row_id"].tolist() == [0, 1, 3]
    assert polars_frame["row_id"].to_list() == [0, 1, 3]


def test_string_casts_preserve_identifier_spelling_and_nullable_values():
    pandas, polars = _backends()
    ids = [0, 12, 305]
    assert pandas.Series(ids).astype(str).tolist() == ["0", "12", "305"]
    assert polars.DataFrame({"animal_id": ids}).with_columns(
        polars.col("animal_id").cast(polars.String)
    )["animal_id"].to_list() == ["0", "12", "305"]

    p_nullable = pandas.Series(["a", None], dtype="string")
    l_nullable = polars.DataFrame({"animal_id": ["a", None]}).with_columns(
        polars.col("animal_id").cast(polars.String)
    )
    assert [_canonical(value) for value in p_nullable.tolist()] == ["a", None]
    assert l_nullable["animal_id"].to_list() == ["a", None]


def test_empty_tables_keep_columns_and_have_backend_neutral_empty_checks():
    pandas, polars = _backends()
    p = pandas.DataFrame({"group_id": pandas.Series(dtype="string"),
                          "frame": pandas.Series(dtype="int64")})
    polars_frame = polars.DataFrame(schema={"group_id": polars.String, "frame": polars.Int64})
    assert p.empty and polars_frame.is_empty()
    assert list(p.columns) == polars_frame.columns == ["group_id", "frame"]
    assert str(p.dtypes["group_id"]) == "string"
    assert polars_frame.schema == {"group_id": polars.String, "frame": polars.Int64}


def test_left_join_preserves_left_order_and_duplicate_matches():
    pandas, polars = _backends()
    left_rows = [{"key": 2, "left": "two"}, {"key": 1, "left": "one"}, {"key": 3, "left": "three"}]
    right_rows = [{"key": 1, "right": "first"}, {"key": 1, "right": "second"}, {"key": 4, "right": "unused"}]
    p = pandas.DataFrame(left_rows).merge(pandas.DataFrame(right_rows), how="left", on="key", sort=False)
    polars_frame = polars.DataFrame(left_rows).join(polars.DataFrame(right_rows), on="key", how="left", maintain_order="left")
    expected = [{"key": 2, "left": "two", "right": None},
                {"key": 1, "left": "one", "right": "first"},
                {"key": 1, "left": "one", "right": "second"},
                {"key": 3, "left": "three", "right": None}]
    assert _records(p, pandas=True) == expected
    assert _records(polars_frame, pandas=False) == expected


def test_csv_round_trip_has_no_index_and_parquet_schema_metadata_is_observable(tmp_path: Path):
    pandas, polars = _backends()
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    rows = [{"animal_id": "a", "score": 1.25}, {"animal_id": "b", "score": 2.5}]
    p = pandas.DataFrame(rows)
    polars_frame = polars.DataFrame(rows)

    pandas_csv = tmp_path / "pandas.csv"
    polars_csv = tmp_path / "polars.csv"
    p.to_csv(pandas_csv, index=False)
    polars_frame.write_csv(polars_csv)
    assert pandas_csv.read_text() == polars_csv.read_text() == "animal_id,score\na,1.25\nb,2.5\n"
    assert _records(pandas.read_csv(pandas_csv), pandas=True) == rows
    assert _records(polars.read_csv(polars_csv), pandas=False) == rows

    pandas_parquet = tmp_path / "pandas.pq"
    polars_parquet = tmp_path / "polars.pq"
    p.to_parquet(pandas_parquet, engine="pyarrow", compression="zstd", index=False)
    polars_frame.write_parquet(polars_parquet, compression="zstd")
    for path in (pandas_parquet, polars_parquet):
        schema = pyarrow_parquet.read_schema(path)
        assert list(schema.names) == ["animal_id", "score"]
        assert pyarrow_parquet.ParquetFile(path).metadata.num_rows == 2
        assert str(pyarrow_parquet.ParquetFile(path).metadata.row_group(0).column(0).compression).lower() == "zstd"
    pandas_metadata = pyarrow_parquet.read_schema(pandas_parquet).metadata or {}
    polars_metadata = pyarrow_parquet.read_schema(polars_parquet).metadata or {}
    assert b"pandas" in pandas_metadata
    assert b"pandas" not in polars_metadata
    assert _records(pandas.read_parquet(polars_parquet), pandas=True) == rows
    assert _records(polars.read_parquet(pandas_parquet), pandas=False) == rows


def test_target_modules_do_not_load_pandas_in_a_fresh_subprocess():
    """Table-only modules must not import pandas in a fresh interpreter."""
    for module in ("tailcyclenet.infer.bridge", "tailcyclenet.scorer.qc"):
        code = f"import sys; import {module}; print('pandas' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False", module
