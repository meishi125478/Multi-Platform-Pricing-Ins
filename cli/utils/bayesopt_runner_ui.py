from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib

if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_bayesopt_args(
    add_config_json_arg,
    add_output_dir_arg,
) -> argparse.Namespace:
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


def plot_curves_for_model(
    model: Any,
    trained_keys: List[str],
    cfg: Dict[str, Any],
    *,
    dedupe_preserve_order,
    plot_model_labels: Mapping[str, Tuple[str, str]],
    parse_model_pairs,
) -> None:
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
        [m for m in trained_keys if m in plot_model_labels]
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
                label, pred_nme = plot_model_labels[model_key]
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
        label, pred_nme = plot_model_labels[model_key]
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


def plot_loss_curve_for_trainer(
    model_name: str,
    trainer: Any,
    *,
    plot_loss_curve_common=None,
) -> None:
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


__all__ = [
    "parse_bayesopt_args",
    "plot_curves_for_model",
    "plot_loss_curve_for_trainer",
]
