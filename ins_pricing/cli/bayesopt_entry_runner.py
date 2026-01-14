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

try:
    from .. import bayesopt as ropt  # type: ignore
    from .utils.cli_common import (  # type: ignore
        PLOT_MODEL_LABELS,
        PYTORCH_TRAINERS,
        build_model_names,
        dedupe_preserve_order,
        load_dataset,
        parse_model_pairs,
        resolve_data_path,
        resolve_path,
        fingerprint_file,
        coerce_dataset_types,
        split_train_test,
    )
    from .utils.cli_config import (  # type: ignore
        add_config_json_arg,
        add_output_dir_arg,
        resolve_and_load_config,
        resolve_data_config,
        resolve_report_config,
        resolve_split_config,
        resolve_runtime_config,
        resolve_output_dirs,
    )
except Exception:  # pragma: no cover
    try:
        import bayesopt as ropt  # type: ignore
        from utils.cli_common import (  # type: ignore
            PLOT_MODEL_LABELS,
            PYTORCH_TRAINERS,
            build_model_names,
            dedupe_preserve_order,
            load_dataset,
            parse_model_pairs,
            resolve_data_path,
            resolve_path,
            fingerprint_file,
            coerce_dataset_types,
            split_train_test,
        )
        from utils.cli_config import (  # type: ignore
            add_config_json_arg,
            add_output_dir_arg,
            resolve_and_load_config,
            resolve_data_config,
            resolve_report_config,
            resolve_split_config,
            resolve_runtime_config,
            resolve_output_dirs,
        )
    except Exception:
        try:
            import ins_pricing.modelling.core.bayesopt as ropt  # type: ignore
            from ins_pricing.cli.utils.cli_common import (  # type: ignore
                PLOT_MODEL_LABELS,
                PYTORCH_TRAINERS,
                build_model_names,
                dedupe_preserve_order,
                load_dataset,
                parse_model_pairs,
                resolve_data_path,
                resolve_path,
                fingerprint_file,
                coerce_dataset_types,
                split_train_test,
            )
            from ins_pricing.cli.utils.cli_config import (  # type: ignore
                add_config_json_arg,
                add_output_dir_arg,
                resolve_and_load_config,
                resolve_data_config,
                resolve_report_config,
                resolve_split_config,
                resolve_runtime_config,
                resolve_output_dirs,
            )
        except Exception:
            import BayesOpt as ropt  # type: ignore
            from utils.cli_common import (  # type: ignore
                PLOT_MODEL_LABELS,
                PYTORCH_TRAINERS,
                build_model_names,
                dedupe_preserve_order,
                load_dataset,
                parse_model_pairs,
                resolve_data_path,
                resolve_path,
                fingerprint_file,
                coerce_dataset_types,
                split_train_test,
            )
            from utils.cli_config import (  # type: ignore
                add_config_json_arg,
                add_output_dir_arg,
                resolve_and_load_config,
                resolve_data_config,
                resolve_report_config,
                resolve_split_config,
                resolve_runtime_config,
                resolve_output_dirs,
            )

import matplotlib

if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .utils.run_logging import configure_run_logging  # type: ignore
except Exception:  # pragma: no cover
    try:
        from utils.run_logging import configure_run_logging  # type: ignore
    except Exception:  # pragma: no cover
        configure_run_logging = None  # type: ignore

try:
    from ..modelling.plotting.diagnostics import plot_loss_curve as plot_loss_curve_common
except Exception:  # pragma: no cover
    try:
        from ins_pricing.plotting.diagnostics import plot_loss_curve as plot_loss_curve_common
    except Exception:  # pragma: no cover
        plot_loss_curve_common = None

try:
    from ..modelling.core.evaluation import (  # type: ignore
        bootstrap_ci,
        calibrate_predictions,
        metrics_report as eval_metrics_report,
        select_threshold,
    )
    from ..governance.registry import ModelArtifact, ModelRegistry  # type: ignore
    from ..production import psi_report as drift_psi_report  # type: ignore
    from ..production.monitoring import group_metrics  # type: ignore
    from ..reporting.report_builder import ReportPayload, write_report  # type: ignore
except Exception:  # pragma: no cover
    try:
        from ins_pricing.modelling.core.evaluation import (  # type: ignore
            bootstrap_ci,
            calibrate_predictions,
            metrics_report as eval_metrics_report,
            select_threshold,
        )
        from ins_pricing.governance.registry import (  # type: ignore
            ModelArtifact,
            ModelRegistry,
        )
        from ins_pricing.production import psi_report as drift_psi_report  # type: ignore
        from ins_pricing.production.monitoring import group_metrics  # type: ignore
        from ins_pricing.reporting.report_builder import (  # type: ignore
            ReportPayload,
            write_report,
        )
    except Exception:  # pragma: no cover
        try:
            from evaluation import (  # type: ignore
                bootstrap_ci,
                calibrate_predictions,
                metrics_report as eval_metrics_report,
                select_threshold,
            )
            from ins_pricing.governance.registry import (  # type: ignore
                ModelArtifact,
                ModelRegistry,
            )
            from ins_pricing.production import psi_report as drift_psi_report  # type: ignore
            from ins_pricing.production.monitoring import group_metrics  # type: ignore
            from ins_pricing.reporting.report_builder import (  # type: ignore
                ReportPayload,
                write_report,
            )
        except Exception:  # pragma: no cover
            bootstrap_ci = None  # type: ignore
            calibrate_predictions = None  # type: ignore
            eval_metrics_report = None  # type: ignore
            select_threshold = None  # type: ignore
            drift_psi_report = None  # type: ignore
            group_metrics = None  # type: ignore
            ReportPayload = None  # type: ignore
            write_report = None  # type: ignore
            ModelRegistry = None  # type: ignore
            ModelArtifact = None  # type: ignore


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
    if eval_metrics_report is None:
        print("[Report] Skip evaluation: metrics module unavailable.")
        return

    pred_col = PLOT_MODEL_LABELS.get(model_key, (None, f"pred_{model_key}"))[1]
    if pred_col not in model.test_data.columns:
        print(
            f"[Report] Missing prediction column '{pred_col}' for {model_name}/{model_key}; skip.")
        return

    weight_col = getattr(model, "weight_nme", None)
    y_true_train = model.train_data[model.resp_nme].to_numpy(
        dtype=float, copy=False)
    y_true_test = model.test_data[model.resp_nme].to_numpy(
        dtype=float, copy=False)
    y_pred_train = model.train_data[pred_col].to_numpy(dtype=float, copy=False)
    y_pred_test = model.test_data[pred_col].to_numpy(dtype=float, copy=False)
    weight_train = (
        model.train_data[weight_col].to_numpy(dtype=float, copy=False)
        if weight_col and weight_col in model.train_data.columns
        else None
    )
    weight_test = (
        model.test_data[weight_col].to_numpy(dtype=float, copy=False)
        if weight_col and weight_col in model.test_data.columns
        else None
    )

    task_type = str(cfg.get("task_type", getattr(
        model, "task_type", "regression")))
    if task_type == "classification":
        y_pred_train = np.clip(y_pred_train, 0.0, 1.0)
        y_pred_test = np.clip(y_pred_test, 0.0, 1.0)

    calibration_info: Optional[Dict[str, Any]] = None
    threshold_info: Optional[Dict[str, Any]] = None
    y_pred_train_eval = y_pred_train
    y_pred_test_eval = y_pred_test

    if task_type == "classification":
        cal_cfg = dict(calibration_cfg or {})
        cal_enabled = bool(cal_cfg.get("enable", False)
                           or cal_cfg.get("method"))
        if cal_enabled and calibrate_predictions is not None:
            method = cal_cfg.get("method", "sigmoid")
            max_rows = cal_cfg.get("max_rows")
            seed = cal_cfg.get("seed")
            y_cal, p_cal = _sample_arrays(
                y_true_train, y_pred_train, max_rows=max_rows, seed=seed)
            try:
                calibrator = calibrate_predictions(y_cal, p_cal, method=method)
                y_pred_train_eval = calibrator.predict(y_pred_train)
                y_pred_test_eval = calibrator.predict(y_pred_test)
                calibration_info = {
                    "method": calibrator.method, "max_rows": max_rows}
            except Exception as exc:
                print(
                    f"[Report] Calibration failed for {model_name}/{model_key}: {exc}")

        thr_cfg = dict(threshold_cfg or {})
        thr_enabled = bool(
            thr_cfg.get("enable", False)
            or thr_cfg.get("metric")
            or thr_cfg.get("value") is not None
        )
        threshold_value = 0.5
        if thr_cfg.get("value") is not None:
            threshold_value = float(thr_cfg["value"])
            threshold_info = {"threshold": threshold_value, "source": "fixed"}
        elif thr_enabled and select_threshold is not None:
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
            threshold_value = float(threshold_info.get("threshold", 0.5))
        else:
            threshold_value = 0.5
        metrics = eval_metrics_report(
            y_true_test,
            y_pred_test_eval,
            task_type=task_type,
            threshold=threshold_value,
        )
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        f1 = 0.0 if (precision + recall) == 0 else 2 * \
            precision * recall / (precision + recall)
        metrics["f1"] = float(f1)
        metrics["threshold"] = float(threshold_value)
    else:
        metrics = eval_metrics_report(
            y_true_test,
            y_pred_test_eval,
            task_type=task_type,
            weight=weight_test,
        )

    bootstrap_results: Dict[str, Dict[str, float]] = {}
    if bootstrap_cfg and bool(bootstrap_cfg.get("enable", False)) and bootstrap_ci is not None:
        metric_names = bootstrap_cfg.get("metrics") or list(metrics.keys())
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
                vals["f1"] = 0.0 if (prec + rec) == 0 else 2 * \
                    prec * rec / (prec + rec)
            return vals

        for name in metric_names:
            if name not in metrics:
                continue
            ci_result = bootstrap_ci(
                lambda y_t, y_p, w=None: float(
                    _metric_fn(y_t, y_p, w).get(name, 0.0)),
                y_true_test,
                y_pred_test_eval,
                weight=weight_test,
                n_samples=n_samples,
                ci=ci,
                seed=seed,
            )
            bootstrap_results[str(name)] = ci_result

    validation_table = None
    if report_group_cols and group_metrics is not None:
        available_groups = [
            col for col in report_group_cols if col in model.test_data.columns
        ]
        if available_groups:
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
                validation_table = validation_table.merge(
                    counts, on=available_groups, how="left")
            except Exception as exc:
                print(
                    f"[Report] group_metrics failed for {model_name}/{model_key}: {exc}")

    risk_trend = None
    if report_time_col and group_metrics is not None:
        if report_time_col in model.test_data.columns:
            try:
                time_df = model.test_data.copy()
                time_series = pd.to_datetime(
                    time_df[report_time_col], errors="coerce")
                time_df = time_df.loc[time_series.notna()].copy()
                if not time_df.empty:
                    time_df["_time_bucket"] = (
                        pd.to_datetime(
                            time_df[report_time_col], errors="coerce")
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
                    risk_trend = risk_trend.merge(
                        counts, on="_time_bucket", how="left")
                    risk_trend = risk_trend.sort_values(
                        "_time_bucket", ascending=bool(report_time_ascending)
                    ).reset_index(drop=True)
                    risk_trend = risk_trend.rename(
                        columns={"_time_bucket": report_time_col})
            except Exception as exc:
                print(
                    f"[Report] time metrics failed for {model_name}/{model_key}: {exc}")

    report_root = (
        Path(report_output_dir)
        if report_output_dir
        else Path(model.output_manager.result_dir) / "reports"
    )
    report_root.mkdir(parents=True, exist_ok=True)

    version = f"{model_key}_{run_id}"
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

    report_path = None
    if ReportPayload is not None and write_report is not None:
        notes_lines = [
            f"- Config SHA256: {config_sha}",
            f"- Data fingerprint: {data_fingerprint.get('sha256_prefix')}",
        ]
        if calibration_info:
            notes_lines.append(
                f"- Calibration: {calibration_info.get('method')}"
            )
        if threshold_info:
            notes_lines.append(
                f"- Threshold selection: {threshold_info}"
            )
        if bootstrap_results:
            notes_lines.append("- Bootstrap: see metrics JSON for CI")
        extra_notes = "\n".join(notes_lines)
        payload = ReportPayload(
            model_name=f"{model_name}/{model_key}",
            model_version=version,
            metrics={k: float(v) for k, v in metrics.items()},
            risk_trend=risk_trend,
            drift_report=psi_report_df,
            validation_table=validation_table,
            extra_notes=extra_notes,
        )
        report_path = write_report(
            payload,
            report_root / f"{model_name}_{model_key}_report.md",
        )

    if register_model and ModelRegistry is not None and ModelArtifact is not None:
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
        artifacts = []
        trainer = model.trainers.get(model_key)
        if trainer is not None:
            try:
                model_path = trainer.output.model_path(
                    trainer._get_model_filename())
                if os.path.exists(model_path):
                    artifacts.append(ModelArtifact(
                        path=model_path, description="trained model"))
            except Exception:
                pass
        if report_path is not None:
            artifacts.append(ModelArtifact(
                path=str(report_path), description="model report"))
        if metrics_path.exists():
            artifacts.append(ModelArtifact(
                path=str(metrics_path), description="metrics json"))
        if bool(cfg.get("save_preprocess", False)):
            artifact_path = cfg.get("preprocess_artifact_path")
            if artifact_path:
                preprocess_path = Path(str(artifact_path))
                if not preprocess_path.is_absolute():
                    preprocess_path = Path(
                        model.output_manager.result_dir) / preprocess_path
            else:
                preprocess_path = Path(model.output_manager.result_path(
                    f"{model.model_nme}_preprocess.json"
                ))
            if preprocess_path.exists():
                artifacts.append(
                    ModelArtifact(path=str(preprocess_path),
                                  description="preprocess artifacts")
                )
        if bool(cfg.get("cache_predictions", False)):
            cache_dir = cfg.get("prediction_cache_dir")
            if cache_dir:
                pred_root = Path(str(cache_dir))
                if not pred_root.is_absolute():
                    pred_root = Path(
                        model.output_manager.result_dir) / pred_root
            else:
                pred_root = Path(
                    model.output_manager.result_dir) / "predictions"
            ext = "csv" if str(
                cfg.get("prediction_cache_format", "parquet")).lower() == "csv" else "parquet"
            for split_label in ("train", "test"):
                pred_path = pred_root / \
                    f"{model_name}_{model_key}_{split_label}.{ext}"
                if pred_path.exists():
                    artifacts.append(
                        ModelArtifact(path=str(pred_path),
                                      description=f"predictions {split_label}")
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

    def _safe_int_env(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, default))
        except (TypeError, ValueError):
            return default

    dist_world_size = _safe_int_env("WORLD_SIZE", 1)
    dist_rank = _safe_int_env("RANK", 0)
    dist_active = dist_world_size > 1
    is_main_process = (not dist_active) or dist_rank == 0

    def _ddp_barrier(reason: str) -> None:
        if not dist_active:
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
