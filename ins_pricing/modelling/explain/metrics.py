from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


def _to_numpy(arr) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    return out.reshape(-1)


def _align(y_true, y_pred, sample_weight=None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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


def rmse(y_true, y_pred, sample_weight=None) -> float:
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    err = (y_t - y_p) ** 2
    return float(np.sqrt(_weighted_mean(err, w)))


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
        if w_sum <= 0:
            y_mean = float(np.mean(y_t))
        else:
            y_mean = float(np.sum(w * y_t) / w_sum)
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
    """Tweedie deviance (power=1 -> Poisson, power=2 -> Gamma, power=0 -> Normal)."""
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
        y_true,
        y_pred,
        sample_weight=sample_weight,
        power=1.0,
        eps=eps,
    )


def gamma_deviance(y_true, y_pred, sample_weight=None, eps: float = 1e-8) -> float:
    return tweedie_deviance(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        power=2.0,
        eps=eps,
    )


def auc_score(y_true, y_pred, sample_weight=None) -> float:
    if roc_auc_score is None:
        raise RuntimeError("auc requires scikit-learn.")
    y_t, y_p, w = _align(y_true, y_pred, sample_weight)
    return float(roc_auc_score(y_t, y_p, sample_weight=w))


def resolve_metric(
    metric: str | Callable,
    *,
    task_type: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
) -> Tuple[Callable, bool, str]:
    if callable(metric):
        if higher_is_better is None:
            raise ValueError("higher_is_better must be provided for custom metric.")
        return metric, bool(higher_is_better), getattr(metric, "__name__", "custom")

    name = str(metric or "auto").lower()
    if name == "auto":
        if task_type == "classification":
            name = "logloss"
        else:
            name = "rmse"

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
