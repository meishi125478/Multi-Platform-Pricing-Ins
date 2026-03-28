from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def clean_column_name(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    return name.replace("\ufeff", "").strip()


def build_sparse_onehot_encoder(*, drop_first: bool) -> OneHotEncoder:
    kwargs: Dict[str, Any] = {
        "handle_unknown": "ignore",
        "dtype": np.float32,
        "drop": "first" if drop_first else None,
    }
    try:
        params = inspect.signature(OneHotEncoder).parameters
    except (TypeError, ValueError):
        params = {}
    if "sparse_output" in params:
        kwargs["sparse_output"] = True
    else:
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)


def normalize_required_columns(
    df: pd.DataFrame,
    required: List[Optional[str]],
    *,
    df_label: str,
) -> None:
    required_names = [r for r in required if isinstance(r, str) and r.strip()]
    if not required_names:
        return

    mapping: Dict[Any, Any] = {}
    existing = set(df.columns)
    for col in df.columns:
        cleaned = clean_column_name(col)
        if cleaned != col and cleaned not in existing:
            mapping[col] = cleaned
    if mapping:
        df.rename(columns=mapping, inplace=True)

    existing = set(df.columns)
    for req in required_names:
        if req in existing:
            continue
        candidates = [
            col
            for col in df.columns
            if isinstance(col, str) and clean_column_name(col).lower() == req.lower()
        ]
        if len(candidates) == 1 and req not in existing:
            df.rename(columns={candidates[0]: req}, inplace=True)
            existing = set(df.columns)
        elif len(candidates) > 1:
            raise KeyError(
                f"{df_label} has multiple columns matching required {req!r} "
                f"(case/space-insensitive): {candidates}"
            )


__all__ = [
    "build_sparse_onehot_encoder",
    "clean_column_name",
    "normalize_required_columns",
]
