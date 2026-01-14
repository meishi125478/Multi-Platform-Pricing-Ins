from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _dtype_matches(actual: np.dtype, expected) -> bool:
    if callable(expected):
        return bool(expected(actual))
    if isinstance(expected, (list, tuple, set)):
        return any(_dtype_matches(actual, item) for item in expected)
    try:
        expected_dtype = np.dtype(expected)
    except Exception:
        return False
    if pd.api.types.is_categorical_dtype(actual) and expected_dtype == np.dtype("category"):
        return True
    if pd.api.types.is_string_dtype(actual) and expected_dtype.kind in {"U", "S", "O"}:
        return True
    if np.issubdtype(actual, expected_dtype):
        return True
    return pd.api.types.is_dtype_equal(actual, expected_dtype)


def validate_schema(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    dtypes: Optional[Dict[str, object]] = None,
    *,
    raise_on_error: bool = True,
) -> Dict[str, object]:
    """Validate required columns and optional dtypes."""
    required = list(required_cols)
    missing = [col for col in required if col not in df.columns]
    dtype_mismatch: Dict[str, Dict[str, str]] = {}
    if dtypes:
        for col, expected in dtypes.items():
            if col not in df.columns:
                continue
            actual = df[col].dtype
            if not _dtype_matches(actual, expected):
                dtype_mismatch[col] = {
                    "expected": str(expected),
                    "actual": str(actual),
                }

    ok = not missing and not dtype_mismatch
    result = {"ok": ok, "missing": missing, "dtype_mismatch": dtype_mismatch}
    if raise_on_error and not ok:
        raise ValueError(f"Schema validation failed: {result}")
    return result


def profile_columns(
    df: pd.DataFrame, cols: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Basic column profiling for missing/uniques and numeric stats."""
    columns = list(cols) if cols is not None else list(df.columns)
    rows = []
    for col in columns:
        series = df[col]
        n = len(series)
        missing_ratio = float(series.isna().mean()) if n else 0.0
        nunique = int(series.nunique(dropna=True))
        unique_ratio = float(nunique / n) if n else 0.0
        entry = {
            "column": col,
            "dtype": str(series.dtype),
            "missing_ratio": missing_ratio,
            "n_unique": nunique,
            "unique_ratio": unique_ratio,
        }
        if pd.api.types.is_numeric_dtype(series):
            entry.update(
                {
                    "min": float(series.min(skipna=True)),
                    "max": float(series.max(skipna=True)),
                    "mean": float(series.mean(skipna=True)),
                }
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def detect_leakage(
    df: pd.DataFrame,
    target_col: str,
    *,
    exclude_cols: Optional[Iterable[str]] = None,
    corr_threshold: float = 0.995,
) -> pd.DataFrame:
    """Detect simple leakage via identical columns or very high correlation."""
    if target_col not in df.columns:
        raise ValueError("target_col not found.")
    exclude = set(exclude_cols or [])
    exclude.add(target_col)
    target = df[target_col]
    results = []
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        reason = None
        score = None
        if series.equals(target):
            reason = "identical"
            score = 1.0
        elif pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
            corr = series.corr(target)
            if pd.notna(corr) and abs(corr) >= corr_threshold:
                reason = "high_corr"
                score = float(corr)
        if reason:
            results.append({"feature": col, "reason": reason, "score": score})
    return pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)
