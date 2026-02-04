from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd


_BIN_CACHE_MAXSIZE = 128
_BIN_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_BIN_CACHE_HITS = 0
_BIN_CACHE_MISSES = 0


def _cache_key(series: pd.Series, n_bins: int, method: str) -> Optional[tuple]:
    try:
        values = series.dropna().to_numpy(dtype=float, copy=False)
        if values.size == 0:
            return None
        values = np.sort(values)
        digest = hashlib.blake2b(values.tobytes(), digest_size=16).hexdigest()
        return (digest, int(values.size), int(n_bins), str(method))
    except Exception:
        return None


def _cache_get(key: tuple) -> Optional[np.ndarray]:
    global _BIN_CACHE_HITS, _BIN_CACHE_MISSES
    if key in _BIN_CACHE:
        _BIN_CACHE_HITS += 1
        _BIN_CACHE.move_to_end(key)
        return _BIN_CACHE[key].copy()
    _BIN_CACHE_MISSES += 1
    return None


def _cache_set(key: tuple, edges: np.ndarray) -> None:
    _BIN_CACHE[key] = np.asarray(edges, dtype=float)
    _BIN_CACHE.move_to_end(key)
    if len(_BIN_CACHE) > _BIN_CACHE_MAXSIZE:
        _BIN_CACHE.popitem(last=False)


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
    cache_key = _cache_key(series, bins, method) if use_cache else None
    bin_edges_full: Optional[np.ndarray] = None

    if cache_key is not None:
        bin_edges_full = _cache_get(cache_key)

    if bin_edges_full is not None:
        binned = pd.cut(series, bins=bin_edges_full, include_lowest=True, labels=labels)
        return binned, np.asarray(bin_edges_full[:-1], dtype=float)

    # Perform actual binning
    if method == "quantile":
        binned, bin_edges_full = pd.qcut(
            series,
            q=bins,
            duplicates="drop",
            labels=labels,
            retbins=True,
        )
    elif method == "uniform":
        binned, bin_edges_full = pd.cut(
            series,
            bins=bins,
            include_lowest=include_lowest,
            labels=labels,
            retbins=True,
        )
    else:
        raise ValueError("method must be one of: quantile, uniform.")

    if cache_key is not None and bin_edges_full is not None:
        _cache_set(cache_key, np.asarray(bin_edges_full, dtype=float))

    return binned, np.asarray(bin_edges_full[:-1], dtype=float)


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
    global _BIN_CACHE_HITS, _BIN_CACHE_MISSES
    _BIN_CACHE.clear()
    _BIN_CACHE_HITS = 0
    _BIN_CACHE_MISSES = 0


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
    return {
        "hits": _BIN_CACHE_HITS,
        "misses": _BIN_CACHE_MISSES,
        "maxsize": _BIN_CACHE_MAXSIZE,
        "currsize": len(_BIN_CACHE),
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
