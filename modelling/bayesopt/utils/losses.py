"""Backward-compatible re-exports for loss utilities."""

from __future__ import annotations

from ins_pricing.utils.losses import (
    CLASSIFICATION_LOSSES,
    CLASSIFICATION_DISTRIBUTION_TO_LOSS,
    DISTRIBUTION_ALIASES,
    LOSS_ALIASES,
    REGRESSION_LOSSES,
    REGRESSION_DISTRIBUTION_TO_LOSS,
    infer_loss_name_from_model_name,
    loss_requires_positive,
    normalize_distribution_name,
    normalize_loss_name,
    regression_loss,
    resolve_effective_loss_name,
    resolve_loss_from_distribution,
    resolve_tweedie_power,
    resolve_xgb_objective,
)

__all__ = [
    "LOSS_ALIASES",
    "DISTRIBUTION_ALIASES",
    "REGRESSION_LOSSES",
    "CLASSIFICATION_LOSSES",
    "REGRESSION_DISTRIBUTION_TO_LOSS",
    "CLASSIFICATION_DISTRIBUTION_TO_LOSS",
    "normalize_loss_name",
    "normalize_distribution_name",
    "infer_loss_name_from_model_name",
    "resolve_effective_loss_name",
    "resolve_loss_from_distribution",
    "resolve_tweedie_power",
    "resolve_xgb_objective",
    "regression_loss",
    "loss_requires_positive",
]
