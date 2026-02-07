"""Metric utilities for model evaluation and drift monitoring.

This module consolidates metric computation used across:
- pricing/monitoring.py: PSI for feature drift
- production/drift.py: PSI wrapper for production monitoring
- modelling/bayesopt/: Model evaluation metrics

Example:
    >>> from ins_pricing.utils import psi_report, MetricFactory
    >>> # PSI for drift monitoring
    >>> report = psi_report(expected_df, actual_df, features=["age", "region"])
    >>> # Model evaluation
    >>> metric = MetricFactory(task_type="regression", tweedie_power=1.5)
    >>> score = metric.compute(y_true, y_pred, sample_weight)
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency
    roc_auc_score = None


# =============================================================================
# PSI (Population Stability Index) Calculations
# =============================================================================


def psi_numeric(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = 10,
    strategy: str = "quantile",
    eps: float = 1e-6,
) -> float:
    """Calculate PSI for numeric features.

    Args:
        expected: Expected/baseline distribution
        actual: Actual/current distribution
        bins: Number of bins for discretization
        strategy: Binning strategy ('quantile' or 'uniform')
        eps: Small value to avoid log(0)

    Returns:
        PSI value (0 = identical, >0.25 = significant drift)
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size == 0 or actual.size == 0:
        return 0.0

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(expected, quantiles)
        bin_edges = np.unique(bin_edges)
    elif strategy == "uniform":
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
    else:
        raise ValueError("strategy must be one of: quantile, uniform.")

    if bin_edges.size < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.clip(exp_pct, eps, 1.0)
    act_pct = np.clip(act_pct, eps, 1.0)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_categorical(
    expected: Iterable,
    actual: Iterable,
    *,
    eps: float = 1e-6,
) -> float:
    """Calculate PSI for categorical features.

    Args:
        expected: Expected/baseline distribution
        actual: Actual/current distribution
        eps: Small value to avoid log(0)

    Returns:
        PSI value (0 = identical, >0.25 = significant drift)
    """
    expected = pd.Series(expected)
    actual = pd.Series(actual)
    categories = pd.Index(expected.dropna().unique()).union(actual.dropna().unique())

    if categories.empty:
        return 0.0

    exp_counts = expected.value_counts().reindex(categories, fill_value=0)
    act_counts = actual.value_counts().reindex(categories, fill_value=0)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.clip(exp_pct.to_numpy(dtype=float), eps, 1.0)
    act_pct = np.clip(act_pct.to_numpy(dtype=float), eps, 1.0)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Calculate PSI, automatically detecting numeric vs categorical.

    Args:
        expected: Expected/baseline distribution
        actual: Actual/current distribution
        bins: Number of bins for numeric features
        strategy: Binning strategy for numeric features

    Returns:
        PSI value
    """
    if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(actual):
        return psi_numeric(expected, actual, bins=bins, strategy=strategy)
    return psi_categorical(expected, actual)


def psi_report(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    *,
    features: Optional[Iterable[str]] = None,
    bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Generate a PSI report for multiple features.

    Args:
        expected_df: Expected/baseline DataFrame
        actual_df: Actual/current DataFrame
        features: List of features to analyze (defaults to all columns)
        bins: Number of bins for numeric features
        strategy: Binning strategy for numeric features

    Returns:
        DataFrame with columns ['feature', 'psi'], sorted by PSI descending
    """
    feats = list(features) if features is not None else list(expected_df.columns)
    rows: List[dict] = []

    for feat in feats:
        if feat not in expected_df.columns or feat not in actual_df.columns:
            continue
        psi = population_stability_index(
            expected_df[feat].to_numpy(),
            actual_df[feat].to_numpy(),
            bins=bins,
            strategy=strategy,
        )
        rows.append({"feature": feat, "psi": psi})

    return pd.DataFrame(rows).sort_values(by="psi", ascending=False).reset_index(drop=True)


# =============================================================================
# Model Evaluation Metrics
# =============================================================================


def _to_numpy(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    return out.reshape(-1)


def _align(y_true, y_pred, sample_weight=None):
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if sample_weight is None:
        return y_t, y_p, None
    w = _to_numpy(sample_weight)
    if w.shape[0] != y_t.shape[0]:
        raise ValueError("sample_weight must have the same length as y_true.")
    return y_t, y_p, w


def _weighted_mean(values: np.ndarray, weight: Optional[np.ndarray]) -> float:
    if weight is None:
        return float(np.mean(values))
    total = float(np.sum(weight))
    if total <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weight) / total)


def mse(y_true, y_pred, sample_weight=None) -> float:
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    err = (y_t - y_p) ** 2
    return _weighted_mean(err, w)


def rmse(y_true, y_pred, sample_weight=None) -> float:
    return float(np.sqrt(mse(y_true, y_pred, sample_weight)))


def mae(y_true, y_pred, sample_weight=None) -> float:
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    err = np.abs(y_t - y_p)
    return _weighted_mean(err, w)


def mape(y_true, y_pred, sample_weight=None, eps: float = 1e-8) -> float:
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    denom = np.maximum(np.abs(y_t), eps)
    err = np.abs((y_t - y_p) / denom)
    return _weighted_mean(err, w)


def r2_score(y_true, y_pred, sample_weight=None) -> float:
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    if w is None:
        y_mean = float(np.mean(y_t))
        sse = float(np.sum((y_t - y_p) ** 2))
        sst = float(np.sum((y_t - y_mean) ** 2))
    else:
        w_sum = float(np.sum(w))
        y_mean = float(np.sum(w * y_t) / w_sum) if w_sum > 0 else float(np.mean(y_t))
        sse = float(np.sum(w * (y_t - y_p) ** 2))
        sst = float(np.sum(w * (y_t - y_mean) ** 2))
    if sst <= 0:
        return 0.0
    return 1.0 - sse / sst


def logloss(y_true, y_pred, sample_weight=None, eps: float = 1e-8) -> float:
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    p = np.clip(y_p, eps, 1 - eps)
    loss = -(y_t * np.log(p) + (1 - y_t) * np.log(1 - p))
    return _weighted_mean(loss, w)


def tweedie_deviance(
    y_true,
    y_pred,
    sample_weight=None,
    *,
    power: float = 1.5,
    eps: float = 1e-8,
) -> float:
    if power < 0:
        raise ValueError("power must be >= 0.")
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    y_p = np.clip(y_p, eps, None)
    y_t_safe = np.clip(y_t, eps, None)

    if power == 0:
        dev = (y_t - y_p) ** 2
    elif power == 1:
        dev = 2 * (y_t_safe * np.log(y_t_safe / y_p) - (y_t_safe - y_p))
    elif power == 2:
        ratio = y_t_safe / y_p
        dev = 2 * ((ratio - 1) - np.log(ratio))
    else:
        term1 = np.power(y_t_safe, 2 - power) / ((1 - power) * (2 - power))
        term2 = y_t_safe * np.power(y_p, 1 - power) / (1 - power)
        term3 = np.power(y_p, 2 - power) / (2 - power)
        dev = 2 * (term1 - term2 + term3)
    return _weighted_mean(dev, w)


def poisson_deviance(y_true, y_pred, sample_weight=None, eps: float = 1e-8) -> float:
    return tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=1.0, eps=eps
    )


def gamma_deviance(y_true, y_pred, sample_weight=None, eps: float = 1e-8) -> float:
    return tweedie_deviance(
        y_true, y_pred, sample_weight=sample_weight, power=2.0, eps=eps
    )


def auc_score(y_true, y_pred, sample_weight=None) -> float:
    if roc_auc_score is None:
        raise RuntimeError("auc requires scikit-learn.")
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    return float(roc_auc_score(y_t, y_p, sample_weight=w))


def resolve_metric(metric, *, task_type: Optional[str] = None, higher_is_better: Optional[bool] = None):
    if callable(metric):
        if higher_is_better is None:
            raise ValueError("higher_is_better must be provided for custom metric.")
        return metric, bool(higher_is_better), getattr(metric, "__name__", "custom")

    name = str(metric or "auto").lower()
    if name == "auto":
        name = "logloss" if task_type == "classification" else "rmse"

    mapping = {
        "rmse": (rmse, False),
        "mae": (mae, False),
        "mape": (mape, False),
        "r2": (r2_score, True),
        "logloss": (logloss, False),
        "poisson": (poisson_deviance, False),
        "gamma": (gamma_deviance, False),
        "tweedie": (tweedie_deviance, False),
        "auc": (auc_score, True),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported metric: {metric}")
    fn, hib = mapping[name]
    if higher_is_better is not None:
        hib = bool(higher_is_better)
    return fn, hib, name


class MetricFactory:
    """Factory for computing evaluation metrics consistently across all trainers.

    This class centralizes metric computation logic that was previously duplicated
    across FTTrainer, ResNetTrainer, GNNTrainer, XGBTrainer, and GLMTrainer.

    Example:
        >>> factory = MetricFactory(task_type='regression', tweedie_power=1.5)
        >>> score = factory.compute(y_true, y_pred, sample_weight)
    """

    def __init__(
        self,
        task_type: str = "regression",
        tweedie_power: float = 1.5,
        loss_name: str = "tweedie",
        clip_min: float = 1e-8,
        clip_max: float = 1 - 1e-8,
    ):
        """Initialize the metric factory.

        Args:
            task_type: Either 'regression' or 'classification'
            tweedie_power: Power parameter for Tweedie deviance (1.0-2.0)
            loss_name: Regression loss name ('tweedie', 'poisson', 'gamma', 'mse', 'mae')
            clip_min: Minimum value for clipping predictions
            clip_max: Maximum value for clipping predictions (for classification)
        """
        self.task_type = task_type
        self.tweedie_power = tweedie_power
        self.loss_name = loss_name
        self.clip_min = clip_min
        self.clip_max = clip_max

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the appropriate metric based on task type.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights

        Returns:
            Computed metric value (lower is better)
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        if self.task_type == "classification":
            y_pred_clipped = np.clip(y_pred, self.clip_min, self.clip_max)
            return float(logloss(y_true, y_pred_clipped, sample_weight=sample_weight))

        loss_name = str(self.loss_name or "tweedie").strip().lower()
        if loss_name in {"mse", "mae"}:
            y_t, y_p, w = _align(y_true, y_pred, sample_weight)
            if loss_name == "mse":
                err = (y_t - y_p) ** 2
                return _weighted_mean(err, w)
            err = np.abs(y_t - y_p)
            return _weighted_mean(err, w)

        y_pred_safe = np.maximum(y_pred, self.clip_min)
        power = self.tweedie_power
        if loss_name == "poisson":
            power = 1.0
        elif loss_name == "gamma":
            power = 2.0
        return float(
            tweedie_deviance(
                y_true,
                y_pred_safe,
                sample_weight=sample_weight,
                power=power,
            )
        )

    def update_power(self, power: float) -> None:
        """Update the Tweedie power parameter.

        Args:
            power: New power value (1.0-2.0)
        """
        self.tweedie_power = power
