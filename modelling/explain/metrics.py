"""Thin wrappers for shared metric utilities."""

from __future__ import annotations

from ins_pricing.utils.metrics import (
    auc_score,
    gamma_deviance,
    logloss,
    mae,
    mape,
    poisson_deviance,
    r2_score,
    resolve_metric,
    rmse,
    tweedie_deviance,
)

__all__ = [
    "rmse",
    "mae",
    "mape",
    "r2_score",
    "logloss",
    "tweedie_deviance",
    "poisson_deviance",
    "gamma_deviance",
    "auc_score",
    "resolve_metric",
]
