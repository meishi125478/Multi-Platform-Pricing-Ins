from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ins_pricing.utils.metrics import mae as _mae
from ins_pricing.utils.metrics import mape as _mape
from ins_pricing.utils.metrics import mse as _mse
from ins_pricing.utils.metrics import r2_score as _r2_score
from ins_pricing.utils.numerics import safe_divide


def _to_1d(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _align_inputs(
    actual,
    predicted,
    weights=None,
    *,
    check_non_negative_weights: bool = True,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    y_true = _to_1d(actual, name="actual")
    y_pred = _to_1d(predicted, name="predicted")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("actual and predicted must have the same length.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("actual and predicted must not contain NaN/Inf.")

    if weights is None:
        return y_true, y_pred, None

    w = _to_1d(weights, name="weights")
    if w.shape[0] != y_true.shape[0]:
        raise ValueError("weights must have the same length as actual.")
    if not np.isfinite(w).all():
        raise ValueError("weights must not contain NaN/Inf.")
    if check_non_negative_weights and (w < 0).any():
        raise ValueError("weights must be non-negative.")
    return y_true, y_pred, w


def weighted_mse(actual, predicted, weights=None) -> float:
    y_true, y_pred, w = _align_inputs(actual, predicted, weights)
    return float(_mse(y_true, y_pred, sample_weight=w))


def weighted_mae(actual, predicted, weights=None) -> float:
    y_true, y_pred, w = _align_inputs(actual, predicted, weights)
    return float(_mae(y_true, y_pred, sample_weight=w))


def weighted_r2(actual, predicted, weights=None) -> float:
    y_true, y_pred, w = _align_inputs(actual, predicted, weights)
    return float(_r2_score(y_true, y_pred, sample_weight=w))


def mape(actual, predicted, eps: float = 1e-12) -> float:
    y_true, y_pred, _ = _align_inputs(actual, predicted, None)
    if (np.abs(y_true) <= eps).any():
        raise ValueError("MAPE is undefined when actual contains zeros.")
    return float(_mape(y_true, y_pred, eps=eps))


def accuracy(actual, predicted_class) -> float:
    y_true, y_pred, _ = _align_inputs(actual, predicted_class, None, check_non_negative_weights=False)
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    return float(np.mean(y_true == y_pred))


def precision_recall(actual, predicted_class) -> tuple[float, float]:
    y_true, y_pred, _ = _align_inputs(actual, predicted_class, None, check_non_negative_weights=False)
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def f1_score(actual, predicted_class) -> float:
    precision, recall = precision_recall(actual, predicted_class)
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def roc_auc(actual, predicted_proba) -> float:
    y_true, score, _ = _align_inputs(actual, predicted_proba, None, check_non_negative_weights=False)
    y_true = (y_true > 0.5).astype(int)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def confusion_matrix(actual, predicted_class) -> np.ndarray:
    y_true, y_pred, _ = _align_inputs(actual, predicted_class, None, check_non_negative_weights=False)
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def loss_ratio(claims, premiums, exposure=None) -> float:
    y_claims, y_premiums, w = _align_inputs(claims, premiums, exposure)
    if w is not None:
        y_claims = y_claims * w
        y_premiums = y_premiums * w
    numer = float(np.sum(y_claims))
    denom = float(np.sum(y_premiums))
    return safe_divide(numer, denom, default=np.nan)


def gini_coefficient(actual, predicted) -> float:
    y_true, y_pred, _ = _align_inputs(actual, predicted, None)
    if float(np.sum(y_true)) <= 0:
        return 0.0
    order = np.argsort(y_pred)
    sorted_actual = y_true[order]
    n = sorted_actual.shape[0]
    numerator = 2.0 * float(np.sum((np.arange(1, n + 1)) * sorted_actual))
    denominator = float(n * np.sum(sorted_actual))
    if denominator <= 0:
        return 0.0
    gini = numerator / denominator - (n + 1.0) / n
    return float(np.clip(gini, -1.0, 1.0))


def lift_at_percentile(actual, predicted, *, percentile: float = 20.0) -> float:
    y_true, y_pred, _ = _align_inputs(actual, predicted, None)
    if percentile <= 0 or percentile > 100:
        raise ValueError("percentile must be in (0, 100].")
    n = y_true.shape[0]
    k = max(1, int(np.ceil(n * float(percentile) / 100.0)))
    top_idx = np.argsort(y_pred)[-k:]
    overall_mean = float(np.mean(y_true))
    if overall_mean == 0:
        return 0.0
    top_mean = float(np.mean(y_true[top_idx]))
    return float(top_mean / overall_mean)


def generate_scoring_report(
    *,
    actual,
    predicted,
    task_type: str = "regression",
    weights=None,
    predicted_proba=None,
) -> dict:
    task = str(task_type).strip().lower()
    if task == "classification":
        report = {
            "accuracy": accuracy(actual, predicted),
        }
        precision, recall = precision_recall(actual, predicted)
        report["precision"] = precision
        report["recall"] = recall
        report["f1"] = f1_score(actual, predicted)
        score_input = predicted if predicted_proba is None else predicted_proba
        report["roc_auc"] = roc_auc(actual, score_input)
        return report

    report = {
        "mse": weighted_mse(actual, predicted, weights),
        "mae": weighted_mae(actual, predicted, weights),
        "r2": weighted_r2(actual, predicted, weights),
    }
    try:
        report["mape"] = mape(actual, predicted)
    except ValueError:
        report["mape"] = float("nan")
    return report


def save_report(report: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


def batch_score(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    data: pd.DataFrame,
    *,
    output_col: str = "prediction",
    batch_size: int = 10000,
    output_path: Optional[str | Path] = None,
    keep_input: bool = True,
) -> pd.DataFrame:
    """Batch scoring for large datasets."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    n_rows = len(data)
    prediction = np.empty(n_rows, dtype=float)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        chunk = data.iloc[start:end]
        pred = np.asarray(predict_fn(chunk)).reshape(-1)
        if pred.shape[0] != (end - start):
            raise ValueError("predict_fn output length must match batch size.")
        prediction[start:end] = pred
    result = data.copy() if keep_input else pd.DataFrame(index=data.index)
    result[output_col] = prediction
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in {".parquet", ".pq"}:
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
    return result
