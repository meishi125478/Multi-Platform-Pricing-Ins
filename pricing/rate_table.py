from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from ins_pricing.exceptions import DataValidationError
from ins_pricing.utils.numerics import safe_divide
from ins_pricing.utils.validation import (
    validate_dataframe_not_empty,
    validate_required_columns,
)


def compute_base_rate(
    df: pd.DataFrame,
    *,
    loss_col: str,
    exposure_col: str,
    weight_col: Optional[str] = None,
) -> float:
    """Compute base rate as loss / exposure."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("df must be a pandas DataFrame.")
    validate_dataframe_not_empty(df, df_name="df")
    required_cols = [loss_col, exposure_col]
    if weight_col:
        required_cols.append(weight_col)
    validate_required_columns(df, required_cols, df_name="df")

    loss = df[loss_col].to_numpy(dtype=float, copy=False)
    exposure = df[exposure_col].to_numpy(dtype=float, copy=False)
    if weight_col and weight_col in df.columns:
        weight = df[weight_col].to_numpy(dtype=float, copy=False)
        loss = loss * weight
        exposure = exposure * weight
    total_loss = float(np.sum(loss))
    total_exposure = float(np.sum(exposure))
    return safe_divide(total_loss, total_exposure, default=np.nan)


def apply_factor_tables(
    df: pd.DataFrame,
    factor_tables: Dict[str, pd.DataFrame],
    *,
    default_relativity: float = 1.0,
) -> np.ndarray:
    """Apply factor relativities and return a multiplicative factor."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("df must be a pandas DataFrame.")
    validate_dataframe_not_empty(df, df_name="df")

    multiplier = np.ones(len(df), dtype=float)
    for factor, table in factor_tables.items():
        if factor not in df.columns:
            raise ValueError(f"Missing factor column: {factor}")
        if "level" not in table.columns or "relativity" not in table.columns:
            raise ValueError("Factor table must include 'level' and 'relativity'.")
        mapping = table.set_index("level")["relativity"]
        rel = df[factor].map(mapping).fillna(default_relativity).to_numpy(dtype=float)
        multiplier *= rel
    return multiplier


def rate_premium(
    df: pd.DataFrame,
    *,
    exposure_col: str,
    base_rate: float,
    factor_tables: Dict[str, pd.DataFrame],
    default_relativity: float = 1.0,
) -> np.ndarray:
    """Compute premium using base rate and factor tables."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("df must be a pandas DataFrame.")
    validate_dataframe_not_empty(df, df_name="df")
    validate_required_columns(df, [exposure_col], df_name="df")

    exposure = df[exposure_col].to_numpy(dtype=float, copy=False)
    factors = apply_factor_tables(
        df, factor_tables, default_relativity=default_relativity
    )
    return exposure * float(base_rate) * factors


@dataclass
class RateTable:
    base_rate: float
    factor_tables: Dict[str, pd.DataFrame]
    default_relativity: float = 1.0

    def score(self, df: pd.DataFrame, *, exposure_col: str) -> np.ndarray:
        return rate_premium(
            df,
            exposure_col=exposure_col,
            base_rate=self.base_rate,
            factor_tables=self.factor_tables,
            default_relativity=self.default_relativity,
        )


def generate_rate_table(
    factors: Dict[str, pd.DataFrame],
    *,
    base_rate: float,
) -> pd.DataFrame:
    """Generate multi-dimensional cartesian rate table from factor tables."""
    pieces = []
    for name, table in factors.items():
        if "relativity" not in table.columns:
            raise ValueError(f"Factor table '{name}' must include 'relativity'.")
        level_cols = [col for col in table.columns if col != "relativity"]
        if not level_cols:
            raise ValueError(f"Factor table '{name}' must include a level column.")
        level_col = level_cols[0]
        slim = table[[level_col, "relativity"]].copy()
        slim = slim.rename(columns={"relativity": f"{name}__rel"})
        pieces.append(slim)

    if not pieces:
        return pd.DataFrame({"rate": [float(base_rate)]})

    table = pieces[0].copy()
    for piece in pieces[1:]:
        table = table.assign(_k=1).merge(piece.assign(_k=1), on="_k", how="inner").drop(columns="_k")

    rel_cols = [c for c in table.columns if c.endswith("__rel")]
    rel_product = np.ones(len(table), dtype=float)
    for col in rel_cols:
        rel_product *= pd.to_numeric(table[col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    table["rate"] = float(base_rate) * rel_product
    return table.drop(columns=rel_cols)


def lookup_rate(
    rate_table: pd.DataFrame,
    *,
    characteristics: Dict[str, object],
) -> float:
    """Lookup a rate from a generated rate table by exact match."""
    if "rate" not in rate_table.columns:
        raise ValueError("rate_table must contain a 'rate' column.")
    mask = pd.Series(True, index=rate_table.index)
    for key, value in characteristics.items():
        if key not in rate_table.columns:
            raise ValueError(f"Characteristic '{key}' not present in rate_table.")
        mask &= rate_table[key] == value
    matched = rate_table.loc[mask, "rate"]
    if matched.empty:
        raise KeyError("No matching rate found for characteristics.")
    return float(matched.iloc[0])
