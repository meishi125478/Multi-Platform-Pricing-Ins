"""
CLI entry point generated from BayesOpt_AutoPricing.ipynb so the workflow can
run non‑interactively (e.g., via torchrun).

Example:
    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        ins_pricing/cli/BayesOpt_entry.py \\
        --config-json ins_pricing/examples/modelling/config_template.json \\
        --model-keys ft --max-evals 50 --use-ft-ddp
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Use unified import resolver to eliminate nested try/except chains
from .utils.import_resolver import resolve_imports, setup_sys_path
from .utils.evaluation_context import (
    EvaluationContext,
    TrainingContext,
    ModelIdentity,
    DataFingerprint,
    CalibrationConfig,
    ThresholdConfig,
    BootstrapConfig,
    ReportConfig,
    RegistryConfig,
)

# Resolve all imports from a single location
setup_sys_path()
_imports = resolve_imports()

ropt = _imports.bayesopt
PLOT_MODEL_LABELS = _imports.PLOT_MODEL_LABELS
PYTORCH_TRAINERS = _imports.PYTORCH_TRAINERS
build_model_names = _imports.build_model_names
dedupe_preserve_order = _imports.dedupe_preserve_order
load_dataset = _imports.load_dataset
parse_model_pairs = _imports.parse_model_pairs
resolve_data_path = _imports.resolve_data_path
resolve_path = _imports.resolve_path
fingerprint_file = _imports.fingerprint_file
coerce_dataset_types = _imports.coerce_dataset_types
split_train_test = _imports.split_train_test

add_config_json_arg = _imports.add_config_json_arg
add_output_dir_arg = _imports.add_output_dir_arg
resolve_and_load_config = _imports.resolve_and_load_config
resolve_data_config = _imports.resolve_data_config
resolve_report_config = _imports.resolve_report_config
resolve_split_config = _imports.resolve_split_config
resolve_runtime_config = _imports.resolve_runtime_config
resolve_output_dirs = _imports.resolve_output_dirs

bootstrap_ci = _imports.bootstrap_ci
calibrate_predictions = _imports.calibrate_predictions
eval_metrics_report = _imports.metrics_report
select_threshold = _imports.select_threshold

ModelArtifact = _imports.ModelArtifact
ModelRegistry = _imports.ModelRegistry
drift_psi_report = _imports.drift_psi_report
group_metrics = _imports.group_metrics
ReportPayload = _imports.ReportPayload
write_report = _imports.write_report

configure_run_logging = _imports.configure_run_logging
plot_loss_curve_common = _imports.plot_loss_curve

import matplotlib

if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch trainer generated from BayesOpt_AutoPricing notebook."
    )
    add_config_json_arg(
        parser,
        help_text="Path to the JSON config describing datasets and feature columns.",
    )
    parser.add_argument(
        "--model-keys",
        nargs="+",
        default=["ft"],
        choices=["glm", "xgb", "resn", "ft", "gnn", "all"],
        help="Space-separated list of trainers to run (e.g., --model-keys glm xgb). Include 'all' to run every trainer.",
    )
    parser.add_argument(
        "--stack-model-keys",
        nargs="+",
        default=None,
        choices=["glm", "xgb", "resn", "ft", "gnn", "all"],
        help=(
            "Only used when ft_role != 'model' (FT runs as feature generator). "
            "When provided (or when config defines stack_model_keys), these trainers run after FT features "
            "are generated. Use 'all' to run every non-FT trainer."
        ),
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=50,
        help="Optuna trial count per dataset.",
    )
    parser.add_argument(
        "--use-resn-ddp",
        action="store_true",
        help="Force ResNet trainer to use DistributedDataParallel.",
    )
    parser.add_argument(
        "--use-ft-ddp",
        action="store_true",
        help="Force FT-Transformer trainer to use DistributedDataParallel.",
    )
    parser.add_argument(
        "--use-resn-dp",
        action="store_true",
        help="Enable ResNet DataParallel fall-back regardless of config.",
    )
    parser.add_argument(
        "--use-ft-dp",
        action="store_true",
        help="Enable FT-Transformer DataParallel fall-back regardless of config.",
    )
    parser.add_argument(
        "--use-gnn-dp",
        action="store_true",
        help="Enable GNN DataParallel fall-back regardless of config.",
    )
    parser.add_argument(
        "--use-gnn-ddp",
        action="store_true",
        help="Force GNN trainer to use DistributedDataParallel.",
    )
    parser.add_argument(
        "--gnn-no-ann",
        action="store_true",
        help="Disable approximate k-NN for GNN graph construction and use exact search.",
    )
    parser.add_argument(
        "--gnn-ann-threshold",
        type=int,
        default=None,
        help="Row threshold above which approximate k-NN is preferred (overrides config).",
    )
    parser.add_argument(
        "--gnn-graph-cache",
        default=None,
        help="Optional path to persist/load cached adjacency matrix for GNN.",
    )
    parser.add_argument(
        "--gnn-max-gpu-nodes",
        type=int,
        default=None,
        help="Overrides the maximum node count allowed for GPU k-NN graph construction.",
    )
    parser.add_argument(
        "--gnn-gpu-mem-ratio",
        type=float,
        default=None,
        help="Overrides the fraction of free GPU memory the k-NN builder may consume.",
    )
    parser.add_argument(
        "--gnn-gpu-mem-overhead",
        type=float,
        default=None,
        help="Overrides the temporary GPU memory overhead multiplier for k-NN estimation.",
    )
    add_output_dir_arg(
        parser,
        help_text="Override output root for models/results/plots.",
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="Enable lift/diagnostic plots after training (config file may also request plotting).",
    )
    parser.add_argument(
        "--ft-as-feature",
        action="store_true",
        help="Alias for --ft-role embedding (keep tuning, export embeddings; skip FT plots/SHAP).",
    )
    parser.add_argument(
        "--ft-role",
        default=None,
        choices=["model", "embedding", "unsupervised_embedding"],
        help="How to use FT: model (default), embedding (export pooling embeddings), or unsupervised_embedding.",
    )
    parser.add_argument(
        "--ft-feature-prefix",
        default="ft_feat",
        help="Prefix used for generated FT features (columns: pred_<prefix>_0.. or pred_<prefix>).",
    )
    parser.add_argument(
        "--reuse-best-params",
        action="store_true",
        help="Skip Optuna and reuse best_params saved in Results/versions or bestparams CSV when available.",
    )
    return parser.parse_args()


def _plot_curves_for_model(model: ropt.BayesOptModel, trained_keys: List[str], cfg: Dict) -> None:
    plot_cfg = cfg.get("plot", {})
    legacy_lift_flags = {
        "glm": cfg.get("plot_lift_glm", False),
        "xgb": cfg.get("plot_lift_xgb", False),
        "resn": cfg.get("plot_lift_resn", False),
        "ft": cfg.get("plot_lift_ft", False),
    }
    plot_enabled = plot_cfg.get("enable", any(legacy_lift_flags.values()))
    if not plot_enabled:
        return

    n_bins = int(plot_cfg.get("n_bins", 10))
    oneway_enabled = plot_cfg.get("oneway", True)

    available_models = dedupe_preserve_order(
        [m for m in trained_keys if m in PLOT_MODEL_LABELS]
    )

    lift_models = plot_cfg.get("lift_models")
    if lift_models is None:
        lift_models = [
            m for m, enabled in legacy_lift_flags.items() if enabled]
        if not lift_models:
            lift_models = available_models
    lift_models = dedupe_preserve_order(
        [m for m in lift_models if m in available_models]
    )

    if oneway_enabled:
        oneway_pred = bool(plot_cfg.get("oneway_pred", False))
        oneway_pred_models = plot_cfg.get("oneway_pred_models")
        pred_plotted = False
        if oneway_pred:
            if oneway_pred_models is None:
                oneway_pred_models = lift_models or available_models
            oneway_pred_models = dedupe_preserve_order(
                [m for m in oneway_pred_models if m in available_models]
            )
            for model_key in oneway_pred_models:
                label, pred_nme = PLOT_MODEL_LABELS[model_key]
                if pred_nme not in model.train_data.columns:
                    print(
                        f"[Oneway] Missing prediction column '{pred_nme}'; skip.",
                        flush=True,
                    )
                    continue
                model.plot_oneway(
                    n_bins=n_bins,
                    pred_col=pred_nme,
                    pred_label=label,
                    plot_subdir="oneway/post",
                )
                pred_plotted = True
        if not oneway_pred or not pred_plotted:
            model.plot_oneway(n_bins=n_bins, plot_subdir="oneway/post")

    if not available_models:
        return

    for model_key in lift_models:
        label, pred_nme = PLOT_MODEL_LABELS[model_key]
        model.plot_lift(model_label=label, pred_nme=pred_nme, n_bins=n_bins)

    if not plot_cfg.get("double_lift", True) or len(available_models) < 2:
        return

    raw_pairs = plot_cfg.get("double_lift_pairs")
    if raw_pairs:
        pairs = [
            (a, b)
            for a, b in parse_model_pairs(raw_pairs)
            if a in available_models and b in available_models and a != b
        ]
    else:
        pairs = [(a, b) for i, a in enumerate(available_models)
                 for b in available_models[i + 1:]]

    for first, second in pairs:
        model.plot_dlift([first, second], n_bins=n_bins)


def _plot_loss_curve_for_trainer(model_name: str, trainer) -> None:
    model_obj = getattr(trainer, "model", None)
    history = None
    if model_obj is not None:
        history = getattr(model_obj, "training_history", None)
    if not history:
        history = getattr(trainer, "training_history", None)
    if not history:
        return
    train_hist = list(history.get("train") or [])
    val_hist = list(history.get("val") or [])
    if not train_hist and not val_hist:
        return
    try:
        plot_dir = trainer.output.plot_path(
            f"{model_name}/loss/loss_{model_name}_{trainer.model_name_prefix}.png"
        )
    except Exception:
        default_dir = Path("plot") / model_name / "loss"
        default_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = str(
            default_dir / f"loss_{model_name}_{trainer.model_name_prefix}.png")
    if plot_loss_curve_common is not None:
        plot_loss_curve_common(
            history=history,
            title=f"{trainer.model_name_prefix} Loss Curve ({model_name})",
            save_path=plot_dir,
            show=False,
        )
    else:
        epochs = range(1, max(len(train_hist), len(val_hist)) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        if train_hist:
            ax.plot(range(1, len(train_hist) + 1),
                    train_hist, label="Train Loss", color="tab:blue")
        if val_hist:
            ax.plot(range(1, len(val_hist) + 1),
                    val_hist, label="Validation Loss", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted Loss")
        ax.set_title(
            f"{trainer.model_name_prefix} Loss Curve ({model_name})")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir, dpi=300)
        plt.close(fig)
    print(
        f"[Plot] Saved loss curve for {model_name}/{trainer.label} -> {plot_dir}")


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


def _compute_psi_report(
    model: ropt.BayesOptModel,
    *,
    features: Optional[List[str]],
    bins: int,
    strategy: str,
) -> Optional[pd.DataFrame]:
    if drift_psi_report is None:
        return None
    psi_features = features or list(getattr(model, "factor_nmes", []))
    psi_features = [
        f for f in psi_features if f in model.train_data.columns and f in model.test_data.columns]
    if not psi_features:
        return None
    try:
        return drift_psi_report(
            model.train_data[psi_features],
            model.test_data[psi_features],
            features=psi_features,
            bins=int(bins),
            strategy=str(strategy),
        )
    except Exception as exc:
        print(f"[Report] PSI computation failed: {exc}")
        return None


# --- Refactored helper functions for _evaluate_and_report ---


def _apply_calibration(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    calibration_cfg: Dict[str, Any],
    model_name: str,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """Apply calibration to predictions for classification tasks.

    Returns:
        Tuple of (calibrated_train_preds, calibrated_test_preds, calibration_info)
    """
    cal_cfg = dict(calibration_cfg or {})
    cal_enabled = bool(cal_cfg.get("enable", False) or cal_cfg.get("method"))

    if not cal_enabled or calibrate_predictions is None:
        return y_pred_train, y_pred_test, None

    method = cal_cfg.get("method", "sigmoid")
    max_rows = cal_cfg.get("max_rows")
    seed = cal_cfg.get("seed")
    y_cal, p_cal = _sample_arrays(
        y_true_train, y_pred_train, max_rows=max_rows, seed=seed)

    try:
        calibrator = calibrate_predictions(y_cal, p_cal, method=method)
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
    """Select threshold for classification predictions.

    Returns:
        Tuple of (threshold_value, threshold_info)
    """
    thr_cfg = dict(threshold_cfg or {})
    thr_enabled = bool(
        thr_cfg.get("enable", False)
        or thr_cfg.get("metric")
        or thr_cfg.get("value") is not None
    )

    if thr_cfg.get("value") is not None:
        threshold_value = float(thr_cfg["value"])
        return threshold_value, {"threshold": threshold_value, "source": "fixed"}

    if thr_enabled and select_threshold is not None:
        max_rows = thr_cfg.get("max_rows")
        seed = thr_cfg.get("seed")
        y_thr, p_thr = _sample_arrays(
            y_true_train, y_pred_train_eval, max_rows=max_rows, seed=seed)
        threshold_info = select_threshold(
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
    """Compute metrics for classification task."""
    metrics = eval_metrics_report(
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
    """Compute bootstrap confidence intervals for metrics."""
    if not bootstrap_cfg or not bool(bootstrap_cfg.get("enable", False)) or bootstrap_ci is None:
        return {}

    metric_names = bootstrap_cfg.get("metrics")
    if not metric_names:
        metric_names = [name for name in metrics.keys() if name != "threshold"]
    n_samples = int(bootstrap_cfg.get("n_samples", 200))
    ci = float(bootstrap_cfg.get("ci", 0.95))
    seed = bootstrap_cfg.get("seed")

    def _metric_fn(y_true, y_pred, weight=None):
        vals = eval_metrics_report(
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
        ci_result = bootstrap_ci(
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
    model: ropt.BayesOptModel,
    pred_col: str,
    report_group_cols: Optional[List[str]],
    weight_col: Optional[str],
    model_name: str,
    model_key: str,
) -> Optional[pd.DataFrame]:
    """Compute grouped validation metrics table."""
    if not report_group_cols or group_metrics is None:
        return None

    available_groups = [
        col for col in report_group_cols if col in model.test_data.columns
    ]
    if not available_groups:
        return None

    try:
        validation_table = group_metrics(
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
    model: ropt.BayesOptModel,
    pred_col: str,
    report_time_col: Optional[str],
    report_time_freq: str,
    report_time_ascending: bool,
    weight_col: Optional[str],
    model_name: str,
    model_key: str,
) -> Optional[pd.DataFrame]:
    """Compute time-series risk trend metrics."""
    if not report_time_col or group_metrics is None:
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
        risk_trend = group_metrics(
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
    """Write metrics to JSON file and return the path."""
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
    """Write model report and return the path."""
    if ReportPayload is None or write_report is None:
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

    payload = ReportPayload(
        model_name=f"{model_name}/{model_key}",
        model_version=version,
        metrics={k: float(v) for k, v in metrics.items()},
        risk_trend=risk_trend,
        drift_report=psi_report_df,
        validation_table=validation_table,
        extra_notes="\n".join(notes_lines),
    )
    return write_report(
        payload,
        report_root / f"{model_name}_{model_key}_report.md",
    )


def _register_model_to_registry(
    model: ropt.BayesOptModel,
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
    """Register model artifacts to the model registry."""
    if ModelRegistry is None or ModelArtifact is None:
        return

    registry = ModelRegistry(
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


def _collect_model_artifacts(
    model: ropt.BayesOptModel,
    model_name: str,
    model_key: str,
    report_path: Optional[Path],
    metrics_path: Path,
    cfg: Dict[str, Any],
) -> List:
    """Collect all model artifacts for registry."""
    artifacts = []

    # Trained model artifact
    trainer = model.trainers.get(model_key)
    if trainer is not None:
        try:
            model_path = trainer.output.model_path(trainer._get_model_filename())
            if os.path.exists(model_path):
                artifacts.append(ModelArtifact(path=model_path, description="trained model"))
        except Exception:
            pass

    # Report artifact
    if report_path is not None:
        artifacts.append(ModelArtifact(path=str(report_path), description="model report"))

    # Metrics JSON artifact
    if metrics_path.exists():
        artifacts.append(ModelArtifact(path=str(metrics_path), description="metrics json"))

    # Preprocess artifacts
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
                ModelArtifact(path=str(preprocess_path), description="preprocess artifacts")
            )

    # Prediction cache artifacts
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
                    ModelArtifact(path=str(pred_path), description=f"predictions {split_label}")
                )

    return artifacts


def _evaluate_and_report(
    model: ropt.BayesOptModel,
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
    """Evaluate model predictions and generate reports.

    This function orchestrates the evaluation pipeline:
    1. Extract predictions and ground truth
    2. Apply calibration (for classification)
    3. Select threshold (for classification)
    4. Compute metrics
    5. Compute bootstrap confidence intervals
    6. Generate validation tables and risk trends
    7. Write reports and register model
    """
    if eval_metrics_report is None:
        print("[Report] Skip evaluation: metrics module unavailable.")
        return

    pred_col = PLOT_MODEL_LABELS.get(model_key, (None, f"pred_{model_key}"))[1]
    if pred_col not in model.test_data.columns:
        print(f"[Report] Missing prediction column '{pred_col}' for {model_name}/{model_key}; skip.")
        return

    # Extract predictions and weights
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

    # Process based on task type
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
        metrics = eval_metrics_report(
            y_true_test, y_pred_test_eval, task_type=task_type, weight=weight_test
        )

    # Compute bootstrap confidence intervals
    bootstrap_results = _compute_bootstrap_ci(
        y_true_test, y_pred_test_eval, weight_test, metrics, bootstrap_cfg, task_type
    )

    # Compute validation table and risk trend
    validation_table = _compute_validation_table(
        model, pred_col, report_group_cols, weight_col, model_name, model_key
    )
    risk_trend = _compute_risk_trend(
        model, pred_col, report_time_col, report_time_freq,
        report_time_ascending, weight_col, model_name, model_key
    )

    # Setup output directory
    report_root = (
        Path(report_output_dir)
        if report_output_dir
        else Path(model.output_manager.result_dir) / "reports"
    )
    report_root.mkdir(parents=True, exist_ok=True)
    version = f"{model_key}_{run_id}"

    # Write metrics JSON
    metrics_path = _write_metrics_json(
        report_root, model_name, model_key, version, metrics,
        threshold_info, calibration_info, bootstrap_results,
        data_path, data_fingerprint, config_sha, pred_col, task_type
    )

    # Write model report
    report_path = _write_model_report(
        report_root, model_name, model_key, version, metrics,
        risk_trend, psi_report_df, validation_table,
        calibration_info, threshold_info, bootstrap_results,
        config_sha, data_fingerprint
    )

    # Register model
    if register_model:
        _register_model_to_registry(
            model, model_name, model_key, version, metrics, task_type,
            data_path, data_fingerprint, config_sha, registry_path,
            registry_tags, registry_status, report_path, metrics_path, cfg
        )


def _evaluate_with_context(
    model: ropt.BayesOptModel,
    ctx: EvaluationContext,
) -> None:
    """Evaluate model predictions using context object.

    This is a cleaner interface that uses the EvaluationContext dataclass
    instead of 19+ individual parameters.
    """
    _evaluate_and_report(
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


def _create_ddp_barrier(dist_ctx: TrainingContext):
    """Create a DDP barrier function for distributed training synchronization."""
    def _ddp_barrier(reason: str) -> None:
        if not dist_ctx.is_distributed:
            return
        torch_mod = getattr(ropt, "torch", None)
        dist_mod = getattr(torch_mod, "distributed", None)
        if dist_mod is None:
            return
        try:
            if not getattr(dist_mod, "is_available", lambda: False)():
                return
            if not dist_mod.is_initialized():
                ddp_ok, _, _, _ = ropt.DistributedUtils.setup_ddp()
                if not ddp_ok or not dist_mod.is_initialized():
                    return
            dist_mod.barrier()
        except Exception as exc:
            print(f"[DDP] barrier failed during {reason}: {exc}", flush=True)
            raise
    return _ddp_barrier


def train_from_config(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parents[1]
    config_path, cfg = resolve_and_load_config(
        args.config_json,
        script_dir,
        required_keys=["data_dir", "model_list",
                       "model_categories", "target", "weight"],
    )
    plot_requested = bool(args.plot_curves or cfg.get("plot_curves", False))
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Use TrainingContext for distributed training state
    dist_ctx = TrainingContext.from_env()
    dist_world_size = dist_ctx.world_size
    dist_rank = dist_ctx.rank
    dist_active = dist_ctx.is_distributed
    is_main_process = dist_ctx.is_main_process
    _ddp_barrier = _create_ddp_barrier(dist_ctx)

    data_dir, data_format, data_path_template, dtype_map = resolve_data_config(
        cfg,
        config_path,
        create_data_dir=True,
    )
    runtime_cfg = resolve_runtime_config(cfg)
    ddp_min_rows = runtime_cfg["ddp_min_rows"]
    bo_sample_limit = runtime_cfg["bo_sample_limit"]
    cache_predictions = runtime_cfg["cache_predictions"]
    prediction_cache_dir = runtime_cfg["prediction_cache_dir"]
    prediction_cache_format = runtime_cfg["prediction_cache_format"]
    report_cfg = resolve_report_config(cfg)
    report_output_dir = report_cfg["report_output_dir"]
    report_group_cols = report_cfg["report_group_cols"]
    report_time_col = report_cfg["report_time_col"]
    report_time_freq = report_cfg["report_time_freq"]
    report_time_ascending = report_cfg["report_time_ascending"]
    psi_bins = report_cfg["psi_bins"]
    psi_strategy = report_cfg["psi_strategy"]
    psi_features = report_cfg["psi_features"]
    calibration_cfg = report_cfg["calibration_cfg"]
    threshold_cfg = report_cfg["threshold_cfg"]
    bootstrap_cfg = report_cfg["bootstrap_cfg"]
    register_model = report_cfg["register_model"]
    registry_path = report_cfg["registry_path"]
    registry_tags = report_cfg["registry_tags"]
    registry_status = report_cfg["registry_status"]
    data_fingerprint_max_bytes = report_cfg["data_fingerprint_max_bytes"]
    report_enabled = report_cfg["report_enabled"]

    split_cfg = resolve_split_config(cfg)
    prop_test = split_cfg["prop_test"]
    holdout_ratio = split_cfg["holdout_ratio"]
    val_ratio = split_cfg["val_ratio"]
    split_strategy = split_cfg["split_strategy"]
    split_group_col = split_cfg["split_group_col"]
    split_time_col = split_cfg["split_time_col"]
    split_time_ascending = split_cfg["split_time_ascending"]
    cv_strategy = split_cfg["cv_strategy"]
    cv_group_col = split_cfg["cv_group_col"]
    cv_time_col = split_cfg["cv_time_col"]
    cv_time_ascending = split_cfg["cv_time_ascending"]
    cv_splits = split_cfg["cv_splits"]
    ft_oof_folds = split_cfg["ft_oof_folds"]
    ft_oof_strategy = split_cfg["ft_oof_strategy"]
    ft_oof_shuffle = split_cfg["ft_oof_shuffle"]
    save_preprocess = runtime_cfg["save_preprocess"]
    preprocess_artifact_path = runtime_cfg["preprocess_artifact_path"]
    rand_seed = runtime_cfg["rand_seed"]
    epochs = runtime_cfg["epochs"]
    output_cfg = resolve_output_dirs(
        cfg,
        config_path,
        output_override=args.output_dir,
    )
    output_dir = output_cfg["output_dir"]
    reuse_best_params = bool(
        args.reuse_best_params or runtime_cfg["reuse_best_params"])
    xgb_max_depth_max = runtime_cfg["xgb_max_depth_max"]
    xgb_n_estimators_max = runtime_cfg["xgb_n_estimators_max"]
    optuna_storage = runtime_cfg["optuna_storage"]
    optuna_study_prefix = runtime_cfg["optuna_study_prefix"]
    best_params_files = runtime_cfg["best_params_files"]
    plot_path_style = runtime_cfg["plot_path_style"]

    model_names = build_model_names(
        cfg["model_list"], cfg["model_categories"])
    if not model_names:
        raise ValueError(
            "No model names generated from model_list/model_categories.")

    results: Dict[str, ropt.BayesOptModel] = {}
    trained_keys_by_model: Dict[str, List[str]] = {}

    for model_name in model_names:
        # Per-dataset training loop: load data, split train/test, and train requested models.
        data_path = resolve_data_path(
            data_dir,
            model_name,
            data_format=data_format,
            path_template=data_path_template,
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Missing dataset: {data_path}")
        data_fingerprint = {"path": str(data_path)}
        if report_enabled and is_main_process:
            data_fingerprint = fingerprint_file(
                data_path,
                max_bytes=data_fingerprint_max_bytes,
            )

        print(f"\n=== Processing model {model_name} ===")
        raw = load_dataset(
            data_path,
            data_format=data_format,
            dtype_map=dtype_map,
            low_memory=False,
        )
        raw = coerce_dataset_types(raw)

        train_df, test_df = split_train_test(
            raw,
            holdout_ratio=holdout_ratio,
            strategy=split_strategy,
            group_col=split_group_col,
            time_col=split_time_col,
            time_ascending=split_time_ascending,
            rand_seed=rand_seed,
            reset_index_mode="time_group",
            ratio_label="holdout_ratio",
        )

        use_resn_dp = args.use_resn_dp or cfg.get(
            "use_resn_data_parallel", False)
        use_ft_dp = args.use_ft_dp or cfg.get("use_ft_data_parallel", True)
        dataset_rows = len(raw)
        ddp_enabled = bool(dist_active and (dataset_rows >= int(ddp_min_rows)))
        use_resn_ddp = (args.use_resn_ddp or cfg.get(
            "use_resn_ddp", False)) and ddp_enabled
        use_ft_ddp = (args.use_ft_ddp or cfg.get(
            "use_ft_ddp", False)) and ddp_enabled
        use_gnn_dp = args.use_gnn_dp or cfg.get("use_gnn_data_parallel", False)
        use_gnn_ddp = (args.use_gnn_ddp or cfg.get(
            "use_gnn_ddp", False)) and ddp_enabled
        gnn_use_ann = cfg.get("gnn_use_approx_knn", True)
        if args.gnn_no_ann:
            gnn_use_ann = False
        gnn_threshold = args.gnn_ann_threshold if args.gnn_ann_threshold is not None else cfg.get(
            "gnn_approx_knn_threshold", 50000)
        gnn_graph_cache = args.gnn_graph_cache or cfg.get("gnn_graph_cache")
        if isinstance(gnn_graph_cache, str) and gnn_graph_cache.strip():
            resolved_cache = resolve_path(gnn_graph_cache, config_path.parent)
            if resolved_cache is not None:
                gnn_graph_cache = str(resolved_cache)
        gnn_max_gpu_nodes = args.gnn_max_gpu_nodes if args.gnn_max_gpu_nodes is not None else cfg.get(
            "gnn_max_gpu_knn_nodes", 200000)
        gnn_gpu_mem_ratio = args.gnn_gpu_mem_ratio if args.gnn_gpu_mem_ratio is not None else cfg.get(
            "gnn_knn_gpu_mem_ratio", 0.9)
        gnn_gpu_mem_overhead = args.gnn_gpu_mem_overhead if args.gnn_gpu_mem_overhead is not None else cfg.get(
            "gnn_knn_gpu_mem_overhead", 2.0)

        binary_target = cfg.get("binary_target") or cfg.get("binary_resp_nme")
        task_type = str(cfg.get("task_type", "regression"))
        feature_list = cfg.get("feature_list")
        categorical_features = cfg.get("categorical_features")
        use_gpu = bool(cfg.get("use_gpu", True))
        region_province_col = cfg.get("region_province_col")
        region_city_col = cfg.get("region_city_col")
        region_effect_alpha = cfg.get("region_effect_alpha")
        geo_feature_nmes = cfg.get("geo_feature_nmes")
        geo_token_hidden_dim = cfg.get("geo_token_hidden_dim")
        geo_token_layers = cfg.get("geo_token_layers")
        geo_token_dropout = cfg.get("geo_token_dropout")
        geo_token_k_neighbors = cfg.get("geo_token_k_neighbors")
        geo_token_learning_rate = cfg.get("geo_token_learning_rate")
        geo_token_epochs = cfg.get("geo_token_epochs")

        ft_role = args.ft_role or cfg.get("ft_role", "model")
        if args.ft_as_feature and args.ft_role is None:
            # Keep legacy behavior as a convenience alias only when the config
            # didn't already request a non-default FT role.
            if str(cfg.get("ft_role", "model")) == "model":
                ft_role = "embedding"
        ft_feature_prefix = str(
            cfg.get("ft_feature_prefix", args.ft_feature_prefix))
        ft_num_numeric_tokens = cfg.get("ft_num_numeric_tokens")

        model = ropt.BayesOptModel(
            train_df,
            test_df,
            model_name,
            cfg["target"],
            cfg["weight"],
            feature_list,
            task_type=task_type,
            binary_resp_nme=binary_target,
            cate_list=categorical_features,
            prop_test=val_ratio,
            rand_seed=rand_seed,
            epochs=epochs,
            use_gpu=use_gpu,
            use_resn_data_parallel=use_resn_dp,
            use_ft_data_parallel=use_ft_dp,
            use_resn_ddp=use_resn_ddp,
            use_ft_ddp=use_ft_ddp,
            use_gnn_data_parallel=use_gnn_dp,
            use_gnn_ddp=use_gnn_ddp,
            output_dir=output_dir,
            xgb_max_depth_max=xgb_max_depth_max,
            xgb_n_estimators_max=xgb_n_estimators_max,
            resn_weight_decay=cfg.get("resn_weight_decay"),
            final_ensemble=bool(cfg.get("final_ensemble", False)),
            final_ensemble_k=int(cfg.get("final_ensemble_k", 3)),
            final_refit=bool(cfg.get("final_refit", True)),
            optuna_storage=optuna_storage,
            optuna_study_prefix=optuna_study_prefix,
            best_params_files=best_params_files,
            gnn_use_approx_knn=gnn_use_ann,
            gnn_approx_knn_threshold=gnn_threshold,
            gnn_graph_cache=gnn_graph_cache,
            gnn_max_gpu_knn_nodes=gnn_max_gpu_nodes,
            gnn_knn_gpu_mem_ratio=gnn_gpu_mem_ratio,
            gnn_knn_gpu_mem_overhead=gnn_gpu_mem_overhead,
            region_province_col=region_province_col,
            region_city_col=region_city_col,
            region_effect_alpha=region_effect_alpha,
            geo_feature_nmes=geo_feature_nmes,
            geo_token_hidden_dim=geo_token_hidden_dim,
            geo_token_layers=geo_token_layers,
            geo_token_dropout=geo_token_dropout,
            geo_token_k_neighbors=geo_token_k_neighbors,
            geo_token_learning_rate=geo_token_learning_rate,
            geo_token_epochs=geo_token_epochs,
            ft_role=ft_role,
            ft_feature_prefix=ft_feature_prefix,
            ft_num_numeric_tokens=ft_num_numeric_tokens,
            infer_categorical_max_unique=int(
                cfg.get("infer_categorical_max_unique", 50)),
            infer_categorical_max_ratio=float(
                cfg.get("infer_categorical_max_ratio", 0.05)),
            reuse_best_params=reuse_best_params,
            bo_sample_limit=bo_sample_limit,
            cache_predictions=cache_predictions,
            prediction_cache_dir=prediction_cache_dir,
            prediction_cache_format=prediction_cache_format,
            cv_strategy=cv_strategy or split_strategy,
            cv_group_col=cv_group_col or split_group_col,
            cv_time_col=cv_time_col or split_time_col,
            cv_time_ascending=cv_time_ascending,
            cv_splits=cv_splits,
            ft_oof_folds=ft_oof_folds,
            ft_oof_strategy=ft_oof_strategy,
            ft_oof_shuffle=ft_oof_shuffle,
            save_preprocess=save_preprocess,
            preprocess_artifact_path=preprocess_artifact_path,
            plot_path_style=plot_path_style,
        )

        if plot_requested:
            plot_cfg = cfg.get("plot", {})
            legacy_lift_flags = {
                "glm": cfg.get("plot_lift_glm", False),
                "xgb": cfg.get("plot_lift_xgb", False),
                "resn": cfg.get("plot_lift_resn", False),
                "ft": cfg.get("plot_lift_ft", False),
            }
            plot_enabled = plot_cfg.get(
                "enable", any(legacy_lift_flags.values()))
            if plot_enabled and plot_cfg.get("pre_oneway", False) and plot_cfg.get("oneway", True):
                n_bins = int(plot_cfg.get("n_bins", 10))
                model.plot_oneway(n_bins=n_bins, plot_subdir="oneway/pre")

        if "all" in args.model_keys:
            requested_keys = ["glm", "xgb", "resn", "ft", "gnn"]
        else:
            requested_keys = args.model_keys
        requested_keys = dedupe_preserve_order(requested_keys)

        if ft_role != "model":
            requested_keys = [k for k in requested_keys if k != "ft"]
            if not requested_keys:
                stack_keys = args.stack_model_keys or cfg.get(
                    "stack_model_keys")
                if stack_keys:
                    if "all" in stack_keys:
                        requested_keys = ["glm", "xgb", "resn", "gnn"]
                    else:
                        requested_keys = [k for k in stack_keys if k != "ft"]
                    requested_keys = dedupe_preserve_order(requested_keys)
            if dist_active and ddp_enabled:
                ft_trainer = model.trainers.get("ft")
                if ft_trainer is None:
                    raise ValueError("FT trainer is not available.")
                ft_trainer_uses_ddp = bool(
                    getattr(ft_trainer, "enable_distributed_optuna", False))
                if not ft_trainer_uses_ddp:
                    raise ValueError(
                        "FT embedding under torchrun requires enabling FT DDP (use --use-ft-ddp or set use_ft_ddp=true)."
                    )
        missing = [key for key in requested_keys if key not in model.trainers]
        if missing:
            raise ValueError(
                f"Trainer(s) {missing} not available for {model_name}")

        executed_keys: List[str] = []
        if ft_role != "model":
            if dist_active and not ddp_enabled:
                _ddp_barrier("start_ft_embedding")
                if dist_rank != 0:
                    _ddp_barrier("finish_ft_embedding")
                    continue
            print(
                f"Optimizing ft as {ft_role} for {model_name} (max_evals={args.max_evals})")
            model.optimize_model("ft", max_evals=args.max_evals)
            model.trainers["ft"].save()
            if getattr(ropt, "torch", None) is not None and ropt.torch.cuda.is_available():
                ropt.free_cuda()
            if dist_active and not ddp_enabled:
                _ddp_barrier("finish_ft_embedding")
        for key in requested_keys:
            trainer = model.trainers[key]
            trainer_uses_ddp = bool(
                getattr(trainer, "enable_distributed_optuna", False))
            if dist_active and not trainer_uses_ddp:
                if dist_rank != 0:
                    print(
                        f"[Rank {dist_rank}] Skip {model_name}/{key} because trainer is not DDP-enabled."
                    )
                _ddp_barrier(f"start_non_ddp_{model_name}_{key}")
                if dist_rank != 0:
                    _ddp_barrier(f"finish_non_ddp_{model_name}_{key}")
                    continue

            print(
                f"Optimizing {key} for {model_name} (max_evals={args.max_evals})")
            model.optimize_model(key, max_evals=args.max_evals)
            model.trainers[key].save()
            _plot_loss_curve_for_trainer(model_name, model.trainers[key])
            if key in PYTORCH_TRAINERS:
                ropt.free_cuda()
            if dist_active and not trainer_uses_ddp:
                _ddp_barrier(f"finish_non_ddp_{model_name}_{key}")
            executed_keys.append(key)

        if not executed_keys:
            continue

        results[model_name] = model
        trained_keys_by_model[model_name] = executed_keys
        if report_enabled and is_main_process:
            psi_report_df = _compute_psi_report(
                model,
                features=psi_features,
                bins=psi_bins,
                strategy=str(psi_strategy),
            )
            for key in executed_keys:
                _evaluate_and_report(
                    model,
                    model_name=model_name,
                    model_key=key,
                    cfg=cfg,
                    data_path=data_path,
                    data_fingerprint=data_fingerprint,
                    report_output_dir=report_output_dir,
                    report_group_cols=report_group_cols,
                    report_time_col=report_time_col,
                    report_time_freq=str(report_time_freq),
                    report_time_ascending=bool(report_time_ascending),
                    psi_report_df=psi_report_df,
                    calibration_cfg=calibration_cfg,
                    threshold_cfg=threshold_cfg,
                    bootstrap_cfg=bootstrap_cfg,
                    register_model=register_model,
                    registry_path=registry_path,
                    registry_tags=registry_tags,
                    registry_status=registry_status,
                    run_id=run_id,
                    config_sha=config_sha,
                )

    if not plot_requested:
        return

    for name, model in results.items():
        _plot_curves_for_model(
            model,
            trained_keys_by_model.get(name, []),
            cfg,
        )


def main() -> None:
    if configure_run_logging:
        configure_run_logging(prefix="bayesopt_entry")
    args = _parse_args()
    train_from_config(args)


if __name__ == "__main__":
    main()
