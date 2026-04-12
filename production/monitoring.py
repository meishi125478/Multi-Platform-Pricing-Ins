from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ins_pricing.exceptions import DataValidationError
from ins_pricing.production.scoring import (
    weighted_mae,
    accuracy,
    precision_recall,
    f1_score,
    roc_auc,
    loss_ratio as scoring_loss_ratio,
    generate_scoring_report,
)
from ins_pricing.utils.metrics import population_stability_index, psi_categorical

DEFAULT_DRIFT_ALERT_THRESHOLD = 0.25
HIGH_SEVERITY_PSI_THRESHOLD = 0.30


def calculate_psi(expected, actual, *, buckets: int = 10) -> float:
    return float(
        population_stability_index(
            np.asarray(expected),
            np.asarray(actual),
            bins=int(max(2, buckets)),
            strategy="quantile",
        )
    )


def categorical_drift(expected, actual) -> float:
    return float(psi_categorical(expected, actual))


def ks_test(expected, actual) -> tuple[float, float]:
    x = np.sort(np.asarray(expected, dtype=float).reshape(-1))
    y = np.sort(np.asarray(actual, dtype=float).reshape(-1))
    if x.size == 0 or y.size == 0:
        return 0.0, 1.0
    z = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, z, side="right") / x.size
    cdf_y = np.searchsorted(y, z, side="right") / y.size
    d = float(np.max(np.abs(cdf_x - cdf_y)))
    n_eff = x.size * y.size / (x.size + y.size)
    p = float(2.0 * np.exp(-2.0 * (d * np.sqrt(max(n_eff, 1e-12))) ** 2))
    p = float(np.clip(p, 0.0, 1.0))
    return d, p


def rolling_metrics(
    *,
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    window: int = 7,
) -> pd.DataFrame:
    out = df.copy()
    err = pd.to_numeric(out[actual_col], errors="coerce") - pd.to_numeric(out[pred_col], errors="coerce")
    out["rolling_mae"] = err.abs().rolling(window=window, min_periods=1).mean()
    out["rolling_mse"] = (err ** 2).rolling(window=window, min_periods=1).mean()
    return out


def check_performance_degradation(
    *,
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    threshold: float = 0.2,
) -> bool:
    if len(df) < 4:
        return False
    split = len(df) // 2
    baseline = weighted_mae(df.iloc[:split][actual_col], df.iloc[:split][pred_col])
    current = weighted_mae(df.iloc[split:][actual_col], df.iloc[split:][pred_col])
    if baseline <= 0:
        return bool(current > 0)
    relative_change = (current - baseline) / baseline
    return bool(relative_change > float(threshold))


def compare_metrics(baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, base_value in baseline.items():
        cur_value = float(current.get(key, np.nan))
        base = float(base_value)
        if np.isnan(cur_value):
            out[f"{key}_change"] = float("nan")
        elif base == 0:
            out[f"{key}_change"] = float("inf") if cur_value != 0 else 0.0
        else:
            out[f"{key}_change"] = float((cur_value - base) / abs(base))
    return out


def check_missing_values(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    n = max(len(df), 1)
    for col in df.columns:
        count = int(df[col].isna().sum())
        if count > 0:
            out[col] = {"count": count, "ratio": float(count / n)}
    return out


def detect_outliers(series, *, method: str = "iqr", z_threshold: float = 3.0) -> np.ndarray:
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    if method == "iqr":
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (values < lower).to_numpy() | (values > upper).to_numpy()
    if method == "zscore":
        std = float(values.std(ddof=0))
        if std <= 0:
            return np.zeros(len(values), dtype=bool)
        z = (values - float(values.mean())) / std
        return (np.abs(z) > float(z_threshold)).to_numpy()
    raise ValueError("method must be one of: iqr, zscore.")


def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, str]) -> bool:
    for col, expected_dtype in expected_schema.items():
        if col not in df.columns:
            return False
        if str(df[col].dtype) != str(expected_dtype):
            return False
    return True


def generate_drift_alert(
    *,
    feature: str,
    psi: float,
    threshold: float = DEFAULT_DRIFT_ALERT_THRESHOLD,
) -> dict:
    if psi >= max(threshold, HIGH_SEVERITY_PSI_THRESHOLD):
        severity = "high"
    elif psi >= threshold:
        severity = "medium"
    else:
        severity = "low"
    return {
        "alert_type": "drift",
        "feature": feature,
        "psi": float(psi),
        "threshold": float(threshold),
        "severity": severity,
    }


def generate_performance_alert(
    *,
    metric: str,
    baseline: float,
    current: float,
    threshold: float = 0.2,
) -> dict:
    if baseline == 0:
        change = float("inf") if current != 0 else 0.0
    else:
        change = float((current - baseline) / abs(baseline))
    severity = "high" if change > threshold else "medium"
    return {
        "alert_type": "performance",
        "metric": metric,
        "baseline": float(baseline),
        "current": float(current),
        "change": change,
        "threshold": float(threshold),
        "severity": severity,
    }


def send_email(*args, **kwargs) -> None:  # pragma: no cover - integration hook
    raise NotImplementedError(
        "send_email is an integration hook and must be implemented by deployment code."
    )


def log_to_monitoring_system(*args, **kwargs) -> None:  # pragma: no cover - integration hook
    raise NotImplementedError(
        "log_to_monitoring_system is an integration hook and must be implemented by deployment code."
    )


def send_alert(alert: dict, *, recipients: Iterable[str]) -> None:
    send_email(alert, recipients=list(recipients))


def log_alert(alert: dict) -> None:
    log_to_monitoring_system(alert)


def prepare_dashboard_metrics(
    *,
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    date_col: str,
) -> dict:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    grouped = work.groupby(date_col, dropna=False)
    daily_predictions = grouped[pred_col].mean()
    err = pd.to_numeric(work[actual_col], errors="coerce") - pd.to_numeric(work[pred_col], errors="coerce")
    work["_abs_err"] = err.abs()
    work["_sq_err"] = err ** 2
    grouped_err = work.groupby(date_col, dropna=False)
    daily_mae = grouped_err["_abs_err"].mean()
    daily_mse = grouped_err["_sq_err"].mean()
    return {
        "daily_predictions": daily_predictions.to_dict(),
        "daily_mae": daily_mae.to_dict(),
        "daily_mse": daily_mse.to_dict(),
    }


def feature_distribution_summary(df: pd.DataFrame, *, features: Iterable[str]) -> dict:
    out: Dict[str, dict] = {}
    for feature in features:
        if feature not in df.columns:
            continue
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            out[feature] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        else:
            out[feature] = {
                "value_counts": series.value_counts(dropna=False).to_dict(),
            }
    return out


def monitor_batch(
    *,
    production_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    features: Iterable[str],
) -> dict:
    drift_scores: Dict[str, float] = {}
    alerts = []
    for feature in features:
        if feature not in production_data.columns or feature not in reference_data.columns:
            continue
        if pd.api.types.is_numeric_dtype(production_data[feature]) and pd.api.types.is_numeric_dtype(reference_data[feature]):
            score = calculate_psi(reference_data[feature], production_data[feature], buckets=10)
        else:
            score = categorical_drift(reference_data[feature], production_data[feature])
        drift_scores[feature] = float(score)
        if score >= DEFAULT_DRIFT_ALERT_THRESHOLD:
            alerts.append(
                generate_drift_alert(
                    feature=feature,
                    psi=score,
                    threshold=DEFAULT_DRIFT_ALERT_THRESHOLD,
                )
            )

    quality_checks = {"missing_values": check_missing_values(production_data)}
    return {
        "drift_scores": drift_scores,
        "quality_checks": quality_checks,
        "alerts": alerts,
    }


def load_production_data(config: Dict[str, object]) -> pd.DataFrame:
    path = config.get("data_path")
    if path:
        return pd.read_csv(path)
    return pd.DataFrame()


def run_scheduled_monitoring(config: Dict[str, object]) -> dict:
    production_data = load_production_data(config)
    reference_data = config.get("reference_data")
    if not isinstance(reference_data, pd.DataFrame):
        raise DataValidationError(
            "run_scheduled_monitoring requires 'reference_data' DataFrame in config."
        )
    features = config.get("features")
    if not isinstance(features, (list, tuple)):
        features = [c for c in production_data.columns if c in reference_data.columns]
    return monitor_batch(
        production_data=production_data,
        reference_data=reference_data,
        features=features,
    )


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
    report = generate_scoring_report(
        actual=y_true,
        predicted=y_pred,
        weights=weight,
        task_type="regression",
    )
    mse = float(report.get("mse", 0.0))
    return {
        "rmse": float(np.sqrt(max(mse, 0.0))),
        "mae": float(report.get("mae", 0.0)),
        "mape": float(report.get("mape", np.nan)),
        "r2": float(report.get("r2", 0.0)),
    }


def loss_ratio(
    actual_loss: np.ndarray,
    predicted_premium: np.ndarray,
    *,
    weight: Optional[np.ndarray] = None,
) -> float:
    return scoring_loss_ratio(actual_loss, predicted_premium, weight)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred_label = (np.asarray(y_pred, dtype=float) >= threshold).astype(float)
    precision, recall = precision_recall(y_true, pred_label)
    report = generate_scoring_report(
        actual=y_true,
        predicted=pred_label,
        predicted_proba=y_pred,
        task_type="classification",
    )
    return {
        "accuracy": float(report.get("accuracy", accuracy(y_true, pred_label))),
        "precision": float(report.get("precision", precision)),
        "recall": float(report.get("recall", recall)),
        "f1": float(report.get("f1", f1_score(y_true, pred_label))),
        "roc_auc": float(report.get("roc_auc", roc_auc(y_true, y_pred))),
    }


def metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task_type: str = "regression",
    weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    task = str(task_type).strip().lower()
    if task == "classification":
        return classification_metrics(y_true, y_pred)
    return regression_metrics(y_true, y_pred, weight=weight)


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
    work["_err"] = err
    work["_abs_err"] = np.abs(err)
    work["_err_sq"] = err ** 2
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
    ss_tot = (sum_y2 - (sum_y ** 2) / count).clip(lower=0.0)
    ss_res = grouped["_err_sq"].sum()
    r2 = (1.0 - (ss_res / ss_tot.replace(0.0, np.nan))).fillna(0.0)

    if weight_col:
        sum_w = grouped["_w"].sum().replace(0, 1.0)
        mse = grouped["_w_err_sq"].sum() / sum_w
        mae = grouped["_w_abs_err"].sum() / sum_w
    else:
        mse = grouped["_err_sq"].sum() / count
        mae = grouped["_abs_err"].sum() / count

    rmse = np.sqrt(mse)
    result = pd.DataFrame({"rmse": rmse.astype(float), "mae": mae.astype(float), "r2": r2.astype(float)})
    return result.reset_index()
