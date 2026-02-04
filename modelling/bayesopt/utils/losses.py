"""Backward-compatible re-exports for loss utilities."""

from __future__ import annotations

from ins_pricing.utils.losses import (
    CLASSIFICATION_LOSSES,
    LOSS_ALIASES,
    REGRESSION_LOSSES,
    infer_loss_name_from_model_name,
    loss_requires_positive,
    normalize_loss_name,
    regression_loss,
    resolve_tweedie_power,
    resolve_xgb_objective,
)

__all__ = [
    "LOSS_ALIASES",
    "REGRESSION_LOSSES",
    "CLASSIFICATION_LOSSES",
    "normalize_loss_name",
    "infer_loss_name_from_model_name",
    "resolve_tweedie_power",
    "resolve_xgb_objective",
    "regression_loss",
    "loss_requires_positive",
]
