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
    method_norm = str(method).strip().lower()
    if method_norm == "equal_width":
        method_norm = "uniform"

    cache_key = _cache_key(series, bins, method_norm) if use_cache else None
    bin_edges_full: Optional[np.ndarray] = None

    if cache_key is not None:
        bin_edges_full = _cache_get(cache_key)

    if bin_edges_full is not None:
        binned = pd.cut(series, bins=bin_edges_full, include_lowest=True, labels=labels)
        return binned, np.asarray(bin_edges_full[:-1], dtype=float)

    # Perform actual binning
    if method_norm == "quantile":
        binned, bin_edges_full = pd.qcut(
            series,
            q=bins,
            duplicates="drop",
            labels=labels,
            retbins=True,
        )
    elif method_norm == "uniform":
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
    method: Optional[str] = None,
    n_bins: int = 10,
    weight_col: Optional[str] = None,
    base_rate: Optional[float] = None,
    smoothing: float = 0.0,
    min_exposure: Optional[float] = None,
) -> pd.DataFrame:
    """Build a factor table with rate and relativity.

    Compatibility:
    - Supports legacy ``method`` and ``n_bins`` parameters.
    - Numeric methods: ``quantile`` and ``equal_width``/``uniform``.
    - Categorical method: ``categorical``.
    """
    if weight_col and weight_col in df.columns:
        weights = df[weight_col].to_numpy(dtype=float, copy=False)
    else:
        weights = None

    loss = df[loss_col].to_numpy(dtype=float, copy=False)
    exposure = df[exposure_col].to_numpy(dtype=float, copy=False)

    if weights is not None:
        loss = loss * weights
        exposure = exposure * weights

    series = df[factor_col]
    method_name = (method or "").strip().lower()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if not method_name:
        method_name = "quantile" if is_numeric else "categorical"
    if method_name == "equal_width":
        method_name = "uniform"

    if method_name in {"quantile", "uniform"}:
        if not is_numeric:
            raise ValueError(f"Method '{method_name}' requires numeric factor column.")
        bin_col = f"{factor_col}_bin"
        binned, _ = bin_numeric(series, bins=int(max(1, n_bins)), method=method_name)
        data = pd.DataFrame(
            {
                bin_col: binned,
                "loss": loss,
                "exposure": exposure,
            }
        )
        grouped = data.groupby(bin_col, dropna=False).agg(
            loss=("loss", "sum"),
            exposure=("exposure", "sum"),
            claim_count=("loss", "size"),
        )
        grouped = grouped.reset_index()
        grouped["level"] = grouped[bin_col]
        if len(grouped) > 0 and pd.api.types.is_interval_dtype(grouped[bin_col]):
            grouped["bin_left"] = grouped[bin_col].apply(lambda x: float(x.left) if pd.notna(x) else np.nan)
            grouped["bin_right"] = grouped[bin_col].apply(lambda x: float(x.right) if pd.notna(x) else np.nan)
    elif method_name == "categorical":
        data = pd.DataFrame(
            {
                factor_col: series,
                "loss": loss,
                "exposure": exposure,
            }
        )
        grouped = data.groupby(factor_col, dropna=False).agg(
            loss=("loss", "sum"),
            exposure=("exposure", "sum"),
            claim_count=("loss", "size"),
        )
        grouped = grouped.reset_index()
        grouped["level"] = grouped[factor_col]
    else:
        raise ValueError("method must be one of: quantile, equal_width, uniform, categorical.")

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
    if "claim_count" not in grouped.columns:
        grouped["claim_count"] = 0
    grouped["claim_count"] = grouped["claim_count"].astype(int)

    if min_exposure is not None:
        low_exposure = grouped["exposure"] < float(min_exposure)
        grouped.loc[low_exposure, "relativity"] = 1.0
        grouped.loc[low_exposure, "rate"] = float(base_rate)
        grouped["is_low_exposure"] = low_exposure
    else:
        grouped["is_low_exposure"] = False

    return grouped


def apply_credibility_smoothing(
    factor_table: pd.DataFrame,
    *,
    base_relativity: float = 1.0,
    exposure_col: str = "exposure",
    credibility_k: float = 100.0,
) -> pd.DataFrame:
    """Shrink low-exposure relativities toward the base relativity."""
    out = factor_table.copy()
    if "relativity" not in out.columns:
        raise ValueError("factor_table must include 'relativity'.")
    exposure = pd.to_numeric(out.get(exposure_col, 0.0), errors="coerce").fillna(0.0)
    exposure = exposure.clip(lower=0.0)
    denom = exposure + float(max(credibility_k, 1e-9))
    credibility = np.where(denom > 0, exposure / denom, 0.0)
    out["relativity"] = (
        credibility * pd.to_numeric(out["relativity"], errors="coerce").fillna(base_relativity)
        + (1.0 - credibility) * float(base_relativity)
    )
    return out


def apply_neighbor_smoothing(
    factor_table: pd.DataFrame,
    *,
    window: int = 3,
) -> pd.DataFrame:
    """Smooth relativities by rolling mean over neighboring bins/levels."""
    if window <= 1:
        return factor_table.copy()
    out = factor_table.copy()
    if "relativity" not in out.columns:
        raise ValueError("factor_table must include 'relativity'.")
    rel = pd.to_numeric(out["relativity"], errors="coerce").fillna(1.0)
    out["relativity"] = rel.rolling(window=window, center=True, min_periods=1).mean()
    return out


def apply_factors(
    df: pd.DataFrame,
    factor_table: pd.DataFrame,
    *,
    factor_col: str,
    default_relativity: float = 1.0,
) -> pd.DataFrame:
    """Apply a single factor table to a DataFrame and append relativity column."""
    out = df.copy()
    rel_col = f"{factor_col}_relativity"

    bin_col = f"{factor_col}_bin"
    if bin_col in factor_table.columns:
        mapping = factor_table.set_index(bin_col)["relativity"]
        if {"bin_left", "bin_right"}.issubset(factor_table.columns):
            left = pd.to_numeric(factor_table["bin_left"], errors="coerce").dropna().to_numpy()
            right = pd.to_numeric(factor_table["bin_right"], errors="coerce").dropna().to_numpy()
            if left.size > 0 and right.size > 0:
                edges = np.unique(np.concatenate([left, right]))
                edges.sort()
                if edges.size >= 2:
                    binned = pd.cut(pd.to_numeric(out[factor_col], errors="coerce"), bins=edges, include_lowest=True)
                    out[rel_col] = binned.map(mapping).fillna(float(default_relativity)).to_numpy(dtype=float)
                    return out
        raw_series = out[factor_col]
        out[rel_col] = raw_series.map(mapping).fillna(float(default_relativity)).to_numpy(dtype=float)
        return out

    lookup_col = factor_col if factor_col in factor_table.columns else "level"
    if lookup_col not in factor_table.columns:
        raise ValueError(f"factor_table must contain '{factor_col}' or 'level' column.")
    mapping = factor_table.set_index(lookup_col)["relativity"]
    out[rel_col] = out[factor_col].map(mapping).fillna(float(default_relativity)).to_numpy(dtype=float)
    return out
