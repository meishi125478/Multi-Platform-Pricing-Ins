from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    if denom == 0:
        return default
    return numer / denom


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if weight is not None:
        weight = np.asarray(weight, dtype=float).reshape(-1)
        if weight.shape[0] != y_true.shape[0]:
            raise ValueError("weight length must match y_true.")
    err = y_true - y_pred
    if weight is None:
        mse = float(np.mean(err ** 2))
        mae = float(np.mean(np.abs(err)))
    else:
        w_sum = float(np.sum(weight))
        mse = float(np.sum(weight * (err ** 2)) / max(w_sum, 1.0))
        mae = float(np.sum(weight * np.abs(err)) / max(w_sum, 1.0))
    rmse = float(np.sqrt(mse))
    denom = float(np.mean(y_true)) if np.mean(y_true) != 0 else 1.0
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-9, None)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = 1.0 - _safe_div(ss_res, ss_tot, default=0.0)
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def loss_ratio(
    actual_loss: np.ndarray,
    predicted_premium: np.ndarray,
    *,
    weight: Optional[np.ndarray] = None,
) -> float:
    actual_loss = np.asarray(actual_loss, dtype=float).reshape(-1)
    predicted_premium = np.asarray(predicted_premium, dtype=float).reshape(-1)
    if weight is not None:
        weight = np.asarray(weight, dtype=float).reshape(-1)
        actual_loss = actual_loss * weight
        predicted_premium = predicted_premium * weight
    return _safe_div(float(np.sum(actual_loss)), float(np.sum(predicted_premium)), default=0.0)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    pred_label = (y_pred >= threshold).astype(float)
    acc = float(np.mean(pred_label == y_true))
    precision = _safe_div(float(np.sum((pred_label == 1) & (y_true == 1))),
                          float(np.sum(pred_label == 1)), default=0.0)
    recall = _safe_div(float(np.sum((pred_label == 1) & (y_true == 1))),
                       float(np.sum(y_true == 1)), default=0.0)
    return {"accuracy": acc, "precision": precision, "recall": recall}


def metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_type: str = "regression",
    weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if task_type == "classification":
        metrics = classification_metrics(y_true, y_pred)
    else:
        metrics = regression_metrics(y_true, y_pred, weight=weight)
    return metrics


def group_metrics(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    group_cols: Iterable[str],
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    group_cols = list(group_cols)
    work = df[group_cols].copy()
    y_true = df[actual_col].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    err = y_true - y_pred
    work["_y_true"] = y_true
    work["_y_pred"] = y_pred
    work["_err"] = err
    work["_abs_err"] = np.abs(err)
    work["_err_sq"] = err ** 2
    work["_abs_ratio"] = work["_abs_err"] / np.clip(np.abs(work["_y_true"]), 1e-9, None)
    work["_y_true_sq"] = work["_y_true"] ** 2

    if weight_col:
        w = df[weight_col].to_numpy(dtype=float)
        work["_w"] = w
        work["_w_err_sq"] = w * work["_err_sq"]
        work["_w_abs_err"] = w * work["_abs_err"]

    grouped = work.groupby(group_cols, dropna=False)
    count = grouped["_y_true"].count().replace(0, 1.0)
    sum_y = grouped["_y_true"].sum()
    sum_y2 = grouped["_y_true_sq"].sum()
    ss_tot = sum_y2 - (sum_y ** 2) / count
    ss_tot = ss_tot.clip(lower=0.0)
    ss_res = grouped["_err_sq"].sum()
    r2 = 1.0 - (ss_res / ss_tot.replace(0.0, np.nan))
    r2 = r2.fillna(0.0)

    mape = grouped["_abs_ratio"].mean()
    if weight_col:
        sum_w = grouped["_w"].sum().replace(0, 1.0)
        mse = grouped["_w_err_sq"].sum() / sum_w
        mae = grouped["_w_abs_err"].sum() / sum_w
    else:
        mse = grouped["_err_sq"].sum() / count
        mae = grouped["_abs_err"].sum() / count

    rmse = np.sqrt(mse)
    result = pd.DataFrame({
        "rmse": rmse.astype(float),
        "mae": mae.astype(float),
        "mape": mape.astype(float),
        "r2": r2.astype(float),
    })
    return result.reset_index()
