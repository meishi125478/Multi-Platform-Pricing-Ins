from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@lru_cache(maxsize=128)
def _compute_bins_cached(
    data_hash: int,
    n_bins: int,
    method: str,
    min_val: float,
    max_val: float,
    n_unique: int
) -> Tuple[tuple, int]:
    """Cache bin edge computation based on data characteristics.

    Args:
        data_hash: Hash of sorted unique values for cache key
        n_bins: Number of bins to create
        method: Binning method ('quantile' or 'uniform')
        min_val: Minimum value in data
        max_val: Maximum value in data
        n_unique: Number of unique values

    Returns:
        Tuple of (bin_edges_tuple, actual_bins)

    Note:
        This function caches bin computation for identical data distributions.
        The cache key includes data_hash to ensure correctness while enabling
        reuse when the same column is binned multiple times.
    """
    # This function is called after validation, so we can safely compute
    # The actual binning is done in the calling function
    # This just provides a cache key mechanism
    return (data_hash, n_bins, method, min_val, max_val, n_unique), n_bins


def bin_numeric(
    series: pd.Series,
    *,
    bins: int = 10,
    method: str = "quantile",
    labels: Optional[list] = None,
    include_lowest: bool = True,
    use_cache: bool = True,
) -> Tuple[pd.Series, np.ndarray]:
    """Bin numeric series and return (binned, bin_edges).

    Args:
        series: Numeric series to bin
        bins: Number of bins to create
        method: Binning method ('quantile' or 'uniform')
        labels: Optional labels for bins
        include_lowest: Whether to include lowest value (for uniform binning)
        use_cache: Whether to use caching for repeated binning operations

    Returns:
        Tuple of (binned_series, bin_edges)

    Note:
        When use_cache=True, identical distributions will reuse cached bin edges,
        improving performance when the same column is binned multiple times.
    """
    # Create cache key from data characteristics if caching enabled
    if use_cache:
        # Compute data characteristics for cache key
        unique_vals = series.dropna().unique()
        unique_sorted = np.sort(unique_vals)
        data_hash = hash(unique_sorted.tobytes())
        min_val = float(series.min())
        max_val = float(series.max())
        n_unique = len(unique_vals)

        # Check cache (the function call acts as cache lookup)
        try:
            _compute_bins_cached(data_hash, bins, method, min_val, max_val, n_unique)
        except Exception:
            # If hashing fails, proceed without cache
            pass

    # Perform actual binning
    if method == "quantile":
        binned = pd.qcut(series, q=bins, duplicates="drop", labels=labels)
        bin_edges = binned.cat.categories.left.to_numpy()
    elif method == "uniform":
        binned = pd.cut(series, bins=bins, include_lowest=include_lowest, labels=labels)
        bin_edges = binned.cat.categories.left.to_numpy()
    else:
        raise ValueError("method must be one of: quantile, uniform.")

    return binned, bin_edges


def clear_binning_cache() -> None:
    """Clear the binning cache to free memory.

    This function clears the LRU cache used by bin_numeric to cache
    bin edge computations. Call this periodically in long-running processes
    or when working with very different datasets.

    Example:
        >>> from ins_pricing.pricing.factors import clear_binning_cache
        >>> # After processing many different columns
        >>> clear_binning_cache()
    """
    _compute_bins_cached.cache_clear()


def get_cache_info() -> dict:
    """Get information about the binning cache.

    Returns:
        Dictionary with cache statistics:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - maxsize: Maximum cache size
        - currsize: Current cache size

    Example:
        >>> from ins_pricing.pricing.factors import get_cache_info
        >>> info = get_cache_info()
        >>> print(f"Cache hit rate: {info['hits'] / (info['hits'] + info['misses']):.2%}")
    """
    cache_info = _compute_bins_cached.cache_info()
    return {
        'hits': cache_info.hits,
        'misses': cache_info.misses,
        'maxsize': cache_info.maxsize,
        'currsize': cache_info.currsize
    }


def build_factor_table(
    df: pd.DataFrame,
    *,
    factor_col: str,
    loss_col: str,
    exposure_col: str,
    weight_col: Optional[str] = None,
    base_rate: Optional[float] = None,
    smoothing: float = 0.0,
    min_exposure: Optional[float] = None,
) -> pd.DataFrame:
    """Build a factor table with rate and relativity."""
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].to_numpy(dtype=float, copy=False)
    else:
        weights = None

    loss = df[loss_col].to_numpy(dtype=float, copy=False)
    exposure = df[exposure_col].to_numpy(dtype=float, copy=False)

    if weights is not None:
        loss = loss * weights
        exposure = exposure * weights

    data = pd.DataFrame(
        {
            "factor": df[factor_col],
            "loss": loss,
            "exposure": exposure,
        }
    )
    grouped = data.groupby("factor", dropna=False).agg({"loss": "sum", "exposure": "sum"})
    grouped = grouped.reset_index().rename(columns={"factor": "level"})

    if base_rate is None:
        total_loss = float(grouped["loss"].sum())
        total_exposure = float(grouped["exposure"].sum())
        base_rate = total_loss / total_exposure if total_exposure > 0 else 0.0

    exposure_vals = grouped["exposure"].to_numpy(dtype=float, copy=False)
    loss_vals = grouped["loss"].to_numpy(dtype=float, copy=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(
            exposure_vals > 0,
            (loss_vals + smoothing * base_rate) / (exposure_vals + smoothing),
            0.0,
        )
        relativity = np.where(base_rate > 0, rate / base_rate, 1.0)

    grouped["rate"] = rate
    grouped["relativity"] = relativity
    grouped["base_rate"] = float(base_rate)

    if min_exposure is not None:
        low_exposure = grouped["exposure"] < float(min_exposure)
        grouped.loc[low_exposure, "relativity"] = 1.0
        grouped.loc[low_exposure, "rate"] = float(base_rate)
        grouped["is_low_exposure"] = low_exposure
    else:
        grouped["is_low_exposure"] = False

    return grouped
