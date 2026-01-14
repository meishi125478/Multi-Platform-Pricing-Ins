from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_base_rate(
    df: pd.DataFrame,
    *,
    loss_col: str,
    exposure_col: str,
    weight_col: Optional[str] = None,
) -> float:
    """Compute base rate as loss / exposure."""
    loss = df[loss_col].to_numpy(dtype=float, copy=False)
    exposure = df[exposure_col].to_numpy(dtype=float, copy=False)
    if weight_col and weight_col in df.columns:
        weight = df[weight_col].to_numpy(dtype=float, copy=False)
        loss = loss * weight
        exposure = exposure * weight
    total_exposure = float(np.sum(exposure))
    if total_exposure <= 0:
        return 0.0
    return float(np.sum(loss) / total_exposure)


def apply_factor_tables(
    df: pd.DataFrame,
    factor_tables: Dict[str, pd.DataFrame],
    *,
    default_relativity: float = 1.0,
) -> np.ndarray:
    """Apply factor relativities and return a multiplicative factor."""
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
