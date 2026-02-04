"""Feature inference helpers shared across training and production."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd


def infer_factor_and_cate_list(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    resp_nme: str,
    weight_nme: str,
    *,
    binary_resp_nme: Optional[str] = None,
    factor_nmes: Optional[List[str]] = None,
    cate_list: Optional[List[str]] = None,
    infer_categorical_max_unique: int = 50,
    infer_categorical_max_ratio: float = 0.05,
) -> Tuple[List[str], List[str]]:
    """Infer factor_nmes/cate_list when feature names are not provided."""
    excluded = {resp_nme, weight_nme}
    if binary_resp_nme:
        excluded.add(binary_resp_nme)

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    if factor_nmes is None:
        factors = [c for c in common_cols if c not in excluded]
    else:
        factors = [c for c in factor_nmes if c in common_cols and c not in excluded]

    if cate_list is not None:
        cats = [c for c in cate_list if c in factors]
        return factors, cats

    n_rows = max(1, len(train_df))
    cats: List[str] = []
    for col in factors:
        s = train_df[col]
        if (
            pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ):
            cats.append(col)
            continue
        if pd.api.types.is_integer_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if nunique <= infer_categorical_max_unique or (nunique / n_rows) <= infer_categorical_max_ratio:
                cats.append(col)

    return factors, cats
