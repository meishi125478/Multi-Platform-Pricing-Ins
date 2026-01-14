"""Metric utilities for model evaluation and drift monitoring.

This module consolidates metric computation used across:
- pricing/monitoring.py: PSI for feature drift
- production/drift.py: PSI wrapper for production monitoring
- modelling/core/bayesopt/: Model evaluation metrics

Example:
    >>> from ins_pricing.utils import psi_report, MetricFactory
    >>> # PSI for drift monitoring
    >>> report = psi_report(expected_df, actual_df, features=["age", "region"])
    >>> # Model evaluation
    >>> metric = MetricFactory(task_type="regression", tweedie_power=1.5)
    >>> score = metric.compute(y_true, y_pred, sample_weight)
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import log_loss, mean_tweedie_deviance
except ImportError:
    log_loss = None
    mean_tweedie_deviance = None


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
        clip_min: float = 1e-8,
        clip_max: float = 1 - 1e-8,
    ):
        """Initialize the metric factory.

        Args:
            task_type: Either 'regression' or 'classification'
            tweedie_power: Power parameter for Tweedie deviance (1.0-2.0)
            clip_min: Minimum value for clipping predictions
            clip_max: Maximum value for clipping predictions (for classification)
        """
        self.task_type = task_type
        self.tweedie_power = tweedie_power
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
        if log_loss is None or mean_tweedie_deviance is None:
            raise ImportError("sklearn is required for metric computation")

        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        if self.task_type == "classification":
            y_pred_clipped = np.clip(y_pred, self.clip_min, self.clip_max)
            return float(log_loss(y_true, y_pred_clipped, sample_weight=sample_weight))

        # Regression: use Tweedie deviance
        y_pred_safe = np.maximum(y_pred, self.clip_min)
        return float(
            mean_tweedie_deviance(
                y_true,
                y_pred_safe,
                sample_weight=sample_weight,
                power=self.tweedie_power,
            )
        )

    def update_power(self, power: float) -> None:
        """Update the Tweedie power parameter.

        Args:
            power: New power value (1.0-2.0)
        """
        self.tweedie_power = power
