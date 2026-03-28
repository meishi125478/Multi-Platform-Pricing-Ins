from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from ins_pricing.exceptions import DataValidationError
from ins_pricing.utils.validation import (
    validate_dataframe_not_empty,
    validate_required_columns,
)


def compute_exposure(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    *,
    unit: str = "year",
    time_unit: Optional[str] = None,
    inclusive: bool = False,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = None,
) -> pd.Series | pd.DataFrame:
    """Compute exposure from start/end date columns.

    Compatibility behavior:
    - If ``time_unit`` is provided, returns a DataFrame copy with ``exposure`` column.
    - Otherwise returns a Series (legacy lightweight utility behavior).
    """

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("df must be a pandas DataFrame.")
    validate_dataframe_not_empty(df, df_name="df")
    validate_required_columns(df, [start_col, end_col], df_name="df")

    resolved_unit = (time_unit or unit or "year").strip().lower()
    aliases = {
        "day": "day",
        "days": "day",
        "month": "month",
        "months": "month",
        "year": "year",
        "years": "year",
    }
    resolved_unit = aliases.get(resolved_unit, resolved_unit)

    start = pd.to_datetime(df[start_col])
    end = pd.to_datetime(df[end_col])
    delta_days = (end - start).dt.days.astype(float)
    if inclusive:
        delta_days = delta_days + 1.0
    if resolved_unit == "day":
        exposure = delta_days
    elif resolved_unit == "month":
        exposure = delta_days / 30.0
    elif resolved_unit == "year":
        exposure = delta_days / 365.25
    else:
        raise ValueError("unit must be one of: day, month, year.")

    exposure = exposure.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip_min is not None:
        exposure = exposure.clip(lower=clip_min)
    if clip_max is not None:
        exposure = exposure.clip(upper=clip_max)
    if time_unit is not None:
        out = df.copy()
        out["exposure"] = exposure.to_numpy(dtype=float)
        return out
    return exposure


def aggregate_policy_level(
    df: pd.DataFrame,
    policy_keys: Iterable[str],
    *,
    exposure_col: str,
    claim_count_col: Optional[str] = None,
    claim_amount_col: Optional[str] = None,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate event-level rows to policy-level records."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("df must be a pandas DataFrame.")
    validate_dataframe_not_empty(df, df_name="df")
    required_cols = list(policy_keys) + [exposure_col]
    if claim_count_col:
        required_cols.append(claim_count_col)
    if claim_amount_col:
        required_cols.append(claim_amount_col)
    if weight_col:
        required_cols.append(weight_col)
    validate_required_columns(df, required_cols, df_name="df")

    agg = {exposure_col: "sum"}
    if claim_count_col:
        agg[claim_count_col] = "sum"
    if claim_amount_col:
        agg[claim_amount_col] = "sum"
    if weight_col:
        agg[weight_col] = "sum"
    grouped = df.groupby(list(policy_keys), dropna=False).agg(agg).reset_index()
    return grouped


def build_frequency_severity(
    df: pd.DataFrame,
    *,
    exposure_col: str,
    claim_count_col: str,
    claim_amount_col: str,
    zero_severity: float = 0.0,
) -> pd.DataFrame:
    """Compute frequency, severity and pure premium from counts and losses."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("df must be a pandas DataFrame.")
    validate_dataframe_not_empty(df, df_name="df")
    validate_required_columns(
        df,
        [exposure_col, claim_count_col, claim_amount_col],
        df_name="df",
    )

    exposure = df[exposure_col].to_numpy(dtype=float, copy=False)
    counts = df[claim_count_col].to_numpy(dtype=float, copy=False)
    amounts = df[claim_amount_col].to_numpy(dtype=float, copy=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        frequency = np.where(exposure > 0, counts / exposure, 0.0)
        severity = np.where(counts > 0, amounts / counts, zero_severity)
        pure_premium = frequency * severity

    out = df.copy()
    out["frequency"] = frequency
    out["severity"] = severity
    out["pure_premium"] = pure_premium
    return out
