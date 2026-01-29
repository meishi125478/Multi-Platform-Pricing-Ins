"""Loss selection and regression loss utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ....explain.metrics import (
    gamma_deviance,
    poisson_deviance,
    tweedie_deviance,
)

LOSS_ALIASES = {
    "poisson_deviance": "poisson",
    "gamma_deviance": "gamma",
    "tweedie_deviance": "tweedie",
    "l2": "mse",
    "l1": "mae",
    "absolute": "mae",
    "gaussian": "mse",
    "normal": "mse",
}

REGRESSION_LOSSES = {"tweedie", "poisson", "gamma", "mse", "mae"}
CLASSIFICATION_LOSSES = {"logloss", "bce"}


def normalize_loss_name(loss_name: Optional[str], task_type: str) -> str:
    """Normalize the loss name and validate against supported values."""
    name = str(loss_name or "auto").strip().lower()
    if not name or name == "auto":
        return "auto"
    name = LOSS_ALIASES.get(name, name)
    if task_type == "classification":
        if name not in CLASSIFICATION_LOSSES:
            raise ValueError(
                f"Unsupported classification loss '{loss_name}'. "
                f"Supported: {sorted(CLASSIFICATION_LOSSES)}"
            )
    else:
        if name not in REGRESSION_LOSSES:
            raise ValueError(
                f"Unsupported regression loss '{loss_name}'. "
                f"Supported: {sorted(REGRESSION_LOSSES)}"
            )
    return name


def infer_loss_name_from_model_name(model_name: str) -> str:
    """Preserve legacy heuristic for loss selection based on model name."""
    name = str(model_name or "")
    if "f" in name:
        return "poisson"
    if "s" in name:
        return "gamma"
    return "tweedie"


def resolve_tweedie_power(loss_name: str, default: float = 1.5) -> Optional[float]:
    """Resolve Tweedie power based on loss name."""
    if loss_name == "poisson":
        return 1.0
    if loss_name == "gamma":
        return 2.0
    if loss_name == "tweedie":
        return float(default)
    return None


def resolve_xgb_objective(loss_name: str) -> str:
    """Map regression loss name to XGBoost objective."""
    name = loss_name if loss_name != "auto" else "tweedie"
    mapping = {
        "tweedie": "reg:tweedie",
        "poisson": "count:poisson",
        "gamma": "reg:gamma",
        "mse": "reg:squarederror",
        "mae": "reg:absoluteerror",
    }
    return mapping.get(name, "reg:tweedie")


def regression_loss(
    y_true,
    y_pred,
    sample_weight=None,
    *,
    loss_name: str,
    tweedie_power: Optional[float] = 1.5,
    eps: float = 1e-8,
) -> float:
    """Compute weighted regression loss based on configured loss name."""
    name = normalize_loss_name(loss_name, task_type="regression")
    if name == "auto":
        name = "tweedie"

    y_t = np.asarray(y_true, dtype=float).reshape(-1)
    y_p = np.asarray(y_pred, dtype=float).reshape(-1)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    if name == "mse":
        err = (y_t - y_p) ** 2
        return _weighted_mean(err, w)
    if name == "mae":
        err = np.abs(y_t - y_p)
        return _weighted_mean(err, w)
    if name == "poisson":
        return poisson_deviance(y_t, y_p, sample_weight=w, eps=eps)
    if name == "gamma":
        return gamma_deviance(y_t, y_p, sample_weight=w, eps=eps)

    power = 1.5 if tweedie_power is None else float(tweedie_power)
    return tweedie_deviance(y_t, y_p, sample_weight=w, power=power, eps=eps)


def loss_requires_positive(loss_name: str) -> bool:
    """Return True if the loss requires positive predictions."""
    return loss_name in {"tweedie", "poisson", "gamma"}


def _weighted_mean(values: np.ndarray, weight: Optional[np.ndarray]) -> float:
    if weight is None:
        return float(np.mean(values))
    total = float(np.sum(weight))
    if total <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weight) / total)
