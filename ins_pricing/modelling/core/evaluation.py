from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from ...production.monitoring import (
    classification_metrics,
    regression_metrics,
)


@dataclass
class CalibrationResult:
    method: str
    calibrator: Any

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        return self.calibrator.transform(scores)


def select_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "f1",
    min_positive_rate: Optional[float] = None,
    grid: int = 99,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    thresholds = np.linspace(0.01, 0.99, max(2, int(grid)))
    best = {"threshold": 0.5, "score": -1.0}
    for thr in thresholds:
        pred_label = (y_pred >= thr).astype(float)
        pos_rate = float(np.mean(pred_label))
        if min_positive_rate is not None and pos_rate < float(min_positive_rate):
            continue
        metrics = classification_metrics(y_true, y_pred, threshold=float(thr))
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        score = f1 if metric == "f1" else metrics.get(metric, f1)
        if score > best["score"]:
            best = {"threshold": float(thr), "score": float(score)}
    return best


def calibrate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str = "sigmoid",
) -> CalibrationResult:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    method = str(method or "sigmoid").strip().lower()
    if method in {"platt", "sigmoid", "logistic"}:
        model = LogisticRegression(max_iter=200)
        model.fit(y_pred.reshape(-1, 1), y_true)
        return CalibrationResult(method="sigmoid", calibrator=model)
    if method in {"isotonic"}:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(y_pred, y_true)
        return CalibrationResult(method="isotonic", calibrator=model)
    raise ValueError(f"Unsupported calibration method: {method}")


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    weight: Optional[np.ndarray] = None,
    n_samples: int = 200,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if weight is not None:
        weight = np.asarray(weight).reshape(-1)
    n = len(y_true)
    stats = []
    for _ in range(max(1, int(n_samples))):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        w_t = weight[idx] if weight is not None else None
        stats.append(float(metric_fn(y_t, y_p, w_t)))
    stats = np.asarray(stats, dtype=float)
    alpha = (1.0 - float(ci)) / 2.0
    low = float(np.quantile(stats, alpha))
    high = float(np.quantile(stats, 1.0 - alpha))
    return {"mean": float(np.mean(stats)), "low": low, "high": high}


def metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_type: str,
    weight: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if str(task_type) == "classification":
        return classification_metrics(y_true, y_pred, threshold=threshold)
    return regression_metrics(y_true, y_pred, weight=weight)
