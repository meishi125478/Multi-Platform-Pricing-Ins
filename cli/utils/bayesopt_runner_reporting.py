from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ins_pricing.cli.utils.evaluation_context import EvaluationContext

_PLOT_MODEL_LABELS: Dict[str, tuple[str, str]] = {}
_BOOTSTRAP_CI = None
_CALIBRATE_PREDICTIONS = None
_METRICS_REPORT = None
_SELECT_THRESHOLD = None
_MODEL_ARTIFACT_CLASS = None
_MODEL_REGISTRY_CLASS = None
_DRIFT_PSI_REPORT = None
_GROUP_METRICS = None
_REPORT_PAYLOAD_CLASS = None
_WRITE_REPORT = None


def configure_reporting_dependencies(
    *,
    plot_model_labels: Dict[str, tuple[str, str]],
    bootstrap_ci: Any,
    calibrate_predictions: Any,
    metrics_report: Any,
    select_threshold: Any,
    model_artifact_cls: Any,
    model_registry_cls: Any,
    drift_psi_report: Any,
    group_metrics: Any,
    report_payload_cls: Any,
    write_report: Any,
) -> None:
    global _PLOT_MODEL_LABELS
    global _BOOTSTRAP_CI
    global _CALIBRATE_PREDICTIONS
    global _METRICS_REPORT
    global _SELECT_THRESHOLD
    global _MODEL_ARTIFACT_CLASS
    global _MODEL_REGISTRY_CLASS
    global _DRIFT_PSI_REPORT
    global _GROUP_METRICS
    global _REPORT_PAYLOAD_CLASS
    global _WRITE_REPORT

    _PLOT_MODEL_LABELS = dict(plot_model_labels or {})
    _BOOTSTRAP_CI = bootstrap_ci
    _CALIBRATE_PREDICTIONS = calibrate_predictions
    _METRICS_REPORT = metrics_report
    _SELECT_THRESHOLD = select_threshold
    _MODEL_ARTIFACT_CLASS = model_artifact_cls
    _MODEL_REGISTRY_CLASS = model_registry_cls
    _DRIFT_PSI_REPORT = drift_psi_report
    _GROUP_METRICS = group_metrics
    _REPORT_PAYLOAD_CLASS = report_payload_cls
    _WRITE_REPORT = write_report


def _sample_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    max_rows: Optional[int],
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows is None or max_rows <= 0:
        return y_true, y_pred
    n = len(y_true)
    if n <= max_rows:
        return y_true, y_pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_rows), replace=False)
    return y_true[idx], y_pred[idx]


def compute_psi_report(
    model: Any,
    *,
    features: Optional[List[str]],
    bins: int,
    strategy: str,
) -> Optional[pd.DataFrame]:
    if _DRIFT_PSI_REPORT is None:
        return None
    psi_features = features or list(getattr(model, "factor_nmes", []))
    psi_features = [
        f for f in psi_features if f in model.train_data.columns and f in model.test_data.columns]
    if not psi_features:
        return None
    try:
        return _DRIFT_PSI_REPORT(
            model.train_data[psi_features],
            model.test_data[psi_features],
            features=psi_features,
            bins=int(bins),
            strategy=str(strategy),
        )
    except Exception as exc:
        print(f"[Report] PSI computation failed: {exc}")
        return None


def _apply_calibration(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    calibration_cfg: Dict[str, Any],
    model_name: str,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    cal_cfg = dict(calibration_cfg or {})
    cal_enabled = bool(cal_cfg.get("enable", False) or cal_cfg.get("method"))

    if not cal_enabled or _CALIBRATE_PREDICTIONS is None:
        return y_pred_train, y_pred_test, None

    method = cal_cfg.get("method", "sigmoid")
    max_rows = cal_cfg.get("max_rows")
    seed = cal_cfg.get("seed")
    y_cal, p_cal = _sample_arrays(
        y_true_train, y_pred_train, max_rows=max_rows, seed=seed)

    try:
        calibrator = _CALIBRATE_PREDICTIONS(y_cal, p_cal, method=method)
        calibrated_train = calibrator.predict(y_pred_train)
        calibrated_test = calibrator.predict(y_pred_test)
        calibration_info = {"method": calibrator.method, "max_rows": max_rows}
        return calibrated_train, calibrated_test, calibration_info
    except Exception as exc:
        print(f"[Report] Calibration failed for {model_name}/{model_key}: {exc}")
        return y_pred_train, y_pred_test, None


def _select_classification_threshold(
    y_true_train: np.ndarray,
    y_pred_train_eval: np.ndarray,
    threshold_cfg: Dict[str, Any],
) -> tuple[float, Optional[Dict[str, Any]]]:
    thr_cfg = dict(threshold_cfg or {})
    thr_enabled = bool(
        thr_cfg.get("enable", False)
        or thr_cfg.get("metric")
        or thr_cfg.get("value") is not None
    )

    if thr_cfg.get("value") is not None:
        threshold_value = float(thr_cfg["value"])
        return threshold_value, {"threshold": threshold_value, "source": "fixed"}

    if thr_enabled and _SELECT_THRESHOLD is not None:
        max_rows = thr_cfg.get("max_rows")
        seed = thr_cfg.get("seed")
        y_thr, p_thr = _sample_arrays(
            y_true_train, y_pred_train_eval, max_rows=max_rows, seed=seed)
        threshold_info = _SELECT_THRESHOLD(
            y_thr,
            p_thr,
            metric=thr_cfg.get("metric", "f1"),
            min_positive_rate=thr_cfg.get("min_positive_rate"),
            grid=thr_cfg.get("grid", 99),
        )
        return float(threshold_info.get("threshold", 0.5)), threshold_info

    return 0.5, None


def _compute_classification_metrics(
    y_true_test: np.ndarray,
    y_pred_test_eval: np.ndarray,
    threshold_value: float,
) -> Dict[str, Any]:
    metrics = _METRICS_REPORT(
        y_true_test,
        y_pred_test_eval,
        task_type="classification",
        threshold=threshold_value,
    )
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    metrics["f1"] = float(f1)
    metrics["threshold"] = float(threshold_value)
    return metrics


def _compute_bootstrap_ci(
    y_true_test: np.ndarray,
    y_pred_test_eval: np.ndarray,
    weight_test: Optional[np.ndarray],
    metrics: Dict[str, Any],
    bootstrap_cfg: Dict[str, Any],
    task_type: str,
) -> Dict[str, Dict[str, float]]:
    if not bootstrap_cfg or not bool(bootstrap_cfg.get("enable", False)) or _BOOTSTRAP_CI is None:
        return {}

    metric_names = bootstrap_cfg.get("metrics")
    if not metric_names:
        metric_names = [name for name in metrics.keys() if name != "threshold"]
    n_samples = int(bootstrap_cfg.get("n_samples", 200))
    ci = float(bootstrap_cfg.get("ci", 0.95))
    seed = bootstrap_cfg.get("seed")

    def _metric_fn(y_true, y_pred, weight=None):
        vals = _METRICS_REPORT(
            y_true,
            y_pred,
            task_type=task_type,
            weight=weight,
            threshold=metrics.get("threshold", 0.5),
        )
        if task_type == "classification":
            prec = float(vals.get("precision", 0.0))
            rec = float(vals.get("recall", 0.0))
            vals["f1"] = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        return vals

    bootstrap_results: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        if name not in metrics:
            continue
        ci_result = _BOOTSTRAP_CI(
            lambda y_t, y_p, w=None: float(_metric_fn(y_t, y_p, w).get(name, 0.0)),
            y_true_test,
            y_pred_test_eval,
            weight=weight_test,
            n_samples=n_samples,
            ci=ci,
            seed=seed,
        )
        bootstrap_results[str(name)] = ci_result

    return bootstrap_results


def _compute_validation_table(
    model: Any,
    pred_col: str,
    report_group_cols: Optional[List[str]],
    weight_col: Optional[str],
    model_name: str,
    model_key: str,
) -> Optional[pd.DataFrame]:
    if not report_group_cols or _GROUP_METRICS is None:
        return None

    available_groups = [
        col for col in report_group_cols if col in model.test_data.columns
    ]
    if not available_groups:
        return None

    try:
        validation_table = _GROUP_METRICS(
            model.test_data,
            actual_col=model.resp_nme,
            pred_col=pred_col,
            group_cols=available_groups,
            weight_col=weight_col if weight_col and weight_col in model.test_data.columns else None,
        )
        counts = (
            model.test_data.groupby(available_groups, dropna=False)
            .size()
            .reset_index(name="count")
        )
        return validation_table.merge(counts, on=available_groups, how="left")
    except Exception as exc:
        print(f"[Report] group_metrics failed for {model_name}/{model_key}: {exc}")
        return None


def _compute_risk_trend(
    model: Any,
    pred_col: str,
    report_time_col: Optional[str],
    report_time_freq: str,
    report_time_ascending: bool,
    weight_col: Optional[str],
    model_name: str,
    model_key: str,
) -> Optional[pd.DataFrame]:
    if not report_time_col or _GROUP_METRICS is None:
        return None

    if report_time_col not in model.test_data.columns:
        return None

    try:
        time_df = model.test_data.copy()
        time_series = pd.to_datetime(time_df[report_time_col], errors="coerce")
        time_df = time_df.loc[time_series.notna()].copy()

        if time_df.empty:
            return None

        time_df["_time_bucket"] = (
            pd.to_datetime(time_df[report_time_col], errors="coerce")
            .dt.to_period(report_time_freq)
            .dt.to_timestamp()
        )
        risk_trend = _GROUP_METRICS(
            time_df,
            actual_col=model.resp_nme,
            pred_col=pred_col,
            group_cols=["_time_bucket"],
            weight_col=weight_col if weight_col and weight_col in time_df.columns else None,
        )
        counts = (
            time_df.groupby("_time_bucket", dropna=False)
            .size()
            .reset_index(name="count")
        )
        risk_trend = risk_trend.merge(counts, on="_time_bucket", how="left")
        risk_trend = risk_trend.sort_values(
            "_time_bucket", ascending=bool(report_time_ascending)
        ).reset_index(drop=True)
        return risk_trend.rename(columns={"_time_bucket": report_time_col})
    except Exception as exc:
        print(f"[Report] time metrics failed for {model_name}/{model_key}: {exc}")
        return None


def _write_metrics_json(
    report_root: Path,
    model_name: str,
    model_key: str,
    version: str,
    metrics: Dict[str, Any],
    threshold_info: Optional[Dict[str, Any]],
    calibration_info: Optional[Dict[str, Any]],
    bootstrap_results: Dict[str, Dict[str, float]],
    data_path: Path,
    data_fingerprint: Dict[str, Any],
    config_sha: str,
    pred_col: str,
    task_type: str,
) -> Path:
    metrics_payload = {
        "model_name": model_name,
        "model_key": model_key,
        "model_version": version,
        "metrics": metrics,
        "threshold": threshold_info,
        "calibration": calibration_info,
        "bootstrap": bootstrap_results,
        "data_path": str(data_path),
        "data_fingerprint": data_fingerprint,
        "config_sha256": config_sha,
        "pred_col": pred_col,
        "task_type": task_type,
    }
    metrics_path = report_root / f"{model_name}_{model_key}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return metrics_path


def _write_model_report(
    report_root: Path,
    model_name: str,
    model_key: str,
    version: str,
    metrics: Dict[str, Any],
    risk_trend: Optional[pd.DataFrame],
    psi_report_df: Optional[pd.DataFrame],
    validation_table: Optional[pd.DataFrame],
    calibration_info: Optional[Dict[str, Any]],
    threshold_info: Optional[Dict[str, Any]],
    bootstrap_results: Dict[str, Dict[str, float]],
    config_sha: str,
    data_fingerprint: Dict[str, Any],
) -> Optional[Path]:
    if _REPORT_PAYLOAD_CLASS is None or _WRITE_REPORT is None:
        return None

    notes_lines = [
        f"- Config SHA256: {config_sha}",
        f"- Data fingerprint: {data_fingerprint.get('sha256_prefix')}",
    ]
    if calibration_info:
        notes_lines.append(f"- Calibration: {calibration_info.get('method')}")
    if threshold_info:
        notes_lines.append(f"- Threshold selection: {threshold_info}")
    if bootstrap_results:
        notes_lines.append("- Bootstrap: see metrics JSON for CI")

    payload = _REPORT_PAYLOAD_CLASS(
        model_name=f"{model_name}/{model_key}",
        model_version=version,
        metrics={k: float(v) for k, v in metrics.items()},
        risk_trend=risk_trend,
        drift_report=psi_report_df,
        validation_table=validation_table,
        extra_notes="\n".join(notes_lines),
    )
    return _WRITE_REPORT(
        payload,
        report_root / f"{model_name}_{model_key}_report.md",
    )


def _collect_model_artifacts(
    model: Any,
    model_name: str,
    model_key: str,
    report_path: Optional[Path],
    metrics_path: Path,
    cfg: Dict[str, Any],
) -> List[Any]:
    artifacts = []

    trainer = model.trainers.get(model_key)
    if trainer is not None:
        try:
            model_path = trainer.output.model_path(trainer._get_model_filename())
            if os.path.exists(model_path):
                artifacts.append(_MODEL_ARTIFACT_CLASS(path=model_path, description="trained model"))
        except Exception:
            pass

    if report_path is not None:
        artifacts.append(_MODEL_ARTIFACT_CLASS(path=str(report_path), description="model report"))

    if metrics_path.exists():
        artifacts.append(_MODEL_ARTIFACT_CLASS(path=str(metrics_path), description="metrics json"))

    if bool(cfg.get("save_preprocess", False)):
        artifact_path = cfg.get("preprocess_artifact_path")
        if artifact_path:
            preprocess_path = Path(str(artifact_path))
            if not preprocess_path.is_absolute():
                preprocess_path = Path(model.output_manager.result_dir) / preprocess_path
        else:
            preprocess_path = Path(model.output_manager.result_path(
                f"{model.model_nme}_preprocess.json"
            ))
        if preprocess_path.exists():
            artifacts.append(
                _MODEL_ARTIFACT_CLASS(path=str(preprocess_path), description="preprocess artifacts")
            )

    if bool(cfg.get("cache_predictions", False)):
        cache_dir = cfg.get("prediction_cache_dir")
        if cache_dir:
            pred_root = Path(str(cache_dir))
            if not pred_root.is_absolute():
                pred_root = Path(model.output_manager.result_dir) / pred_root
        else:
            pred_root = Path(model.output_manager.result_dir) / "predictions"
        ext = "csv" if str(cfg.get("prediction_cache_format", "parquet")).lower() == "csv" else "parquet"
        for split_label in ("train", "test"):
            pred_path = pred_root / f"{model_name}_{model_key}_{split_label}.{ext}"
            if pred_path.exists():
                artifacts.append(
                    _MODEL_ARTIFACT_CLASS(path=str(pred_path), description=f"predictions {split_label}")
                )

    return artifacts


def _register_model_to_registry(
    model: Any,
    model_name: str,
    model_key: str,
    version: str,
    metrics: Dict[str, Any],
    task_type: str,
    data_path: Path,
    data_fingerprint: Dict[str, Any],
    config_sha: str,
    registry_path: Optional[str],
    registry_tags: Dict[str, Any],
    registry_status: str,
    report_path: Optional[Path],
    metrics_path: Path,
    cfg: Dict[str, Any],
) -> None:
    if _MODEL_REGISTRY_CLASS is None or _MODEL_ARTIFACT_CLASS is None:
        return

    registry = _MODEL_REGISTRY_CLASS(
        registry_path
        if registry_path
        else Path(model.output_manager.result_dir) / "model_registry.json"
    )

    tags = {str(k): str(v) for k, v in (registry_tags or {}).items()}
    tags.update({
        "model_key": str(model_key),
        "task_type": str(task_type),
        "data_path": str(data_path),
        "data_sha256_prefix": str(data_fingerprint.get("sha256_prefix", "")),
        "data_size": str(data_fingerprint.get("size", "")),
        "data_mtime": str(data_fingerprint.get("mtime", "")),
        "config_sha256": str(config_sha),
    })

    artifacts = _collect_model_artifacts(
        model, model_name, model_key, report_path, metrics_path, cfg
    )

    registry.register(
        name=str(model_name),
        version=version,
        metrics={k: float(v) for k, v in metrics.items()},
        tags=tags,
        artifacts=artifacts,
        status=str(registry_status or "candidate"),
        notes=f"model_key={model_key}",
    )


def evaluate_and_report(
    model: Any,
    *,
    model_name: str,
    model_key: str,
    cfg: Dict[str, Any],
    data_path: Path,
    data_fingerprint: Dict[str, Any],
    report_output_dir: Optional[str],
    report_group_cols: Optional[List[str]],
    report_time_col: Optional[str],
    report_time_freq: str,
    report_time_ascending: bool,
    psi_report_df: Optional[pd.DataFrame],
    calibration_cfg: Dict[str, Any],
    threshold_cfg: Dict[str, Any],
    bootstrap_cfg: Dict[str, Any],
    register_model: bool,
    registry_path: Optional[str],
    registry_tags: Dict[str, Any],
    registry_status: str,
    run_id: str,
    config_sha: str,
) -> None:
    if _METRICS_REPORT is None:
        print("[Report] Skip evaluation: metrics module unavailable.")
        return

    pred_col = _PLOT_MODEL_LABELS.get(model_key, (None, f"pred_{model_key}"))[1]
    if pred_col not in model.test_data.columns:
        print(f"[Report] Missing prediction column '{pred_col}' for {model_name}/{model_key}; skip.")
        return

    weight_col = getattr(model, "weight_nme", None)
    y_true_train = model.train_data[model.resp_nme].to_numpy(dtype=float, copy=False)
    y_true_test = model.test_data[model.resp_nme].to_numpy(dtype=float, copy=False)
    y_pred_train = model.train_data[pred_col].to_numpy(dtype=float, copy=False)
    y_pred_test = model.test_data[pred_col].to_numpy(dtype=float, copy=False)
    weight_test = (
        model.test_data[weight_col].to_numpy(dtype=float, copy=False)
        if weight_col and weight_col in model.test_data.columns
        else None
    )

    task_type = str(cfg.get("task_type", getattr(model, "task_type", "regression")))

    if task_type == "classification":
        y_pred_train = np.clip(y_pred_train, 0.0, 1.0)
        y_pred_test = np.clip(y_pred_test, 0.0, 1.0)

        y_pred_train_eval, y_pred_test_eval, calibration_info = _apply_calibration(
            y_true_train, y_pred_train, y_pred_test, calibration_cfg, model_name, model_key
        )
        threshold_value, threshold_info = _select_classification_threshold(
            y_true_train, y_pred_train_eval, threshold_cfg
        )
        metrics = _compute_classification_metrics(y_true_test, y_pred_test_eval, threshold_value)
    else:
        y_pred_test_eval = y_pred_test
        calibration_info = None
        threshold_info = None
        metrics = _METRICS_REPORT(
            y_true_test, y_pred_test_eval, task_type=task_type, weight=weight_test
        )

    bootstrap_results = _compute_bootstrap_ci(
        y_true_test, y_pred_test_eval, weight_test, metrics, bootstrap_cfg, task_type
    )

    validation_table = _compute_validation_table(
        model, pred_col, report_group_cols, weight_col, model_name, model_key
    )
    risk_trend = _compute_risk_trend(
        model, pred_col, report_time_col, report_time_freq,
        report_time_ascending, weight_col, model_name, model_key
    )

    report_root = (
        Path(report_output_dir)
        if report_output_dir
        else Path(model.output_manager.result_dir) / "reports"
    )
    report_root.mkdir(parents=True, exist_ok=True)
    version = f"{model_key}_{run_id}"

    metrics_path = _write_metrics_json(
        report_root, model_name, model_key, version, metrics,
        threshold_info, calibration_info, bootstrap_results,
        data_path, data_fingerprint, config_sha, pred_col, task_type
    )

    report_path = _write_model_report(
        report_root, model_name, model_key, version, metrics,
        risk_trend, psi_report_df, validation_table,
        calibration_info, threshold_info, bootstrap_results,
        config_sha, data_fingerprint
    )

    if register_model:
        _register_model_to_registry(
            model, model_name, model_key, version, metrics, task_type,
            data_path, data_fingerprint, config_sha, registry_path,
            registry_tags, registry_status, report_path, metrics_path, cfg
        )


def evaluate_with_context(
    model: Any,
    ctx: EvaluationContext,
) -> None:
    evaluate_and_report(
        model,
        model_name=ctx.identity.model_name,
        model_key=ctx.identity.model_key,
        cfg=ctx.cfg,
        data_path=ctx.data_path,
        data_fingerprint=ctx.data_fingerprint.to_dict(),
        report_output_dir=ctx.report.output_dir,
        report_group_cols=ctx.report.group_cols,
        report_time_col=ctx.report.time_col,
        report_time_freq=ctx.report.time_freq,
        report_time_ascending=ctx.report.time_ascending,
        psi_report_df=ctx.psi_report_df,
        calibration_cfg={
            "enable": ctx.calibration.enable,
            "method": ctx.calibration.method,
            "max_rows": ctx.calibration.max_rows,
            "seed": ctx.calibration.seed,
        },
        threshold_cfg={
            "enable": ctx.threshold.enable,
            "metric": ctx.threshold.metric,
            "value": ctx.threshold.value,
            "min_positive_rate": ctx.threshold.min_positive_rate,
            "grid": ctx.threshold.grid,
            "max_rows": ctx.threshold.max_rows,
            "seed": ctx.threshold.seed,
        },
        bootstrap_cfg={
            "enable": ctx.bootstrap.enable,
            "metrics": ctx.bootstrap.metrics,
            "n_samples": ctx.bootstrap.n_samples,
            "ci": ctx.bootstrap.ci,
            "seed": ctx.bootstrap.seed,
        },
        register_model=ctx.registry.register,
        registry_path=ctx.registry.path,
        registry_tags=ctx.registry.tags,
        registry_status=ctx.registry.status,
        run_id=ctx.run_id,
        config_sha=ctx.config_sha,
    )


__all__ = [
    "configure_reporting_dependencies",
    "compute_psi_report",
    "evaluate_and_report",
    "evaluate_with_context",
]
