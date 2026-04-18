from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ins_pricing.utils.paths import coerce_dataset_types, load_dataset


def test_load_dataset_projects_columns_and_dtype_map(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
            "c": ["x", "y", "z"],
        }
    )
    df.to_csv(path, index=False)

    out = load_dataset(
        path,
        data_format="csv",
        dtype_map={"a": "int32", "b": "float32", "missing": "float32"},
        usecols=["a", "c"],
        low_memory=False,
    )

    assert list(out.columns) == ["a", "c"]
    assert str(out["a"].dtype) in {"int32", "int64"}


def test_coerce_dataset_types_is_in_place_and_downcasts_numeric():
    df = pd.DataFrame(
        {
            "num_i": np.array([1, 2, 3], dtype=np.int64),
            "num_f": np.array([1.0, np.nan, 3.5], dtype=np.float64),
            "cat": ["a", None, "c"],
        }
    )

    out = coerce_dataset_types(df)

    assert out is df
    assert out["num_i"].dtype == np.float32
    assert out["num_f"].dtype == np.float32
    assert out["num_f"].isna().sum() == 0
    assert out["cat"].tolist() == ["a", "<NA>", "c"]


def test_coerce_dataset_types_avoids_settingwithcopy_warning_for_sliced_frame():
    base = pd.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "cat": ["a", None, "c", "d"],
        }
    )
    sliced = base[base["num"] > 2]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = coerce_dataset_types(sliced)

    swc_warnings = [
        w for w in caught if "settingwithcopywarning" in type(w.message).__name__.lower()
    ]
    assert not swc_warnings
    assert out["num"].dtype == np.float32
    assert out["cat"].tolist() == ["c", "d"]


def test_load_dataset_with_chunksize_returns_iterator(tmp_path):
    path = tmp_path / "sample_chunks.csv"
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [0.1, 0.2, 0.3, 0.4],
        }
    )
    df.to_csv(path, index=False)

    chunks = load_dataset(
        path,
        data_format="csv",
        usecols=["a", "b"],
        low_memory=False,
        chunksize=2,
    )

    assert not isinstance(chunks, pd.DataFrame)
    first = next(iter(chunks))
    assert list(first.columns) == ["a", "b"]
    assert len(first) == 2
