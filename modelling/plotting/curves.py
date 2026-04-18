from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ins_pricing.modelling.plotting.common import EPS, PlotStyle, finalize_figure, plt

try:  # optional dependency guard
    from sklearn.metrics import (
        auc,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )
    from sklearn.calibration import calibration_curve
except Exception:  # pragma: no cover - handled at call time
    auc = None
    average_precision_score = None
    precision_recall_curve = None
    roc_curve = None
    calibration_curve = None


def _require_sklearn(func_name: str) -> None:
    if roc_curve is None or auc is None:
        raise RuntimeError(f"{func_name} requires scikit-learn to be installed.")


def _to_1d(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _align_arrays(
    pred: Sequence[float],
    actual: Sequence[float],
    weight: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_arr = _to_1d(pred, "pred")
    actual_arr = _to_1d(actual, "actual")
    if len(pred_arr) != len(actual_arr):
        raise ValueError("pred and actual must have the same length.")
    if weight is None:
        weight_arr = np.ones_like(pred_arr, dtype=float)
    else:
        weight_arr = _to_1d(weight, "weight")
        if len(weight_arr) != len(pred_arr):
            raise ValueError("weight must have the same length as pred.")

    mask = np.isfinite(pred_arr) & np.isfinite(actual_arr) & np.isfinite(weight_arr)
    pred_arr = pred_arr[mask]
    actual_arr = actual_arr[mask]
    weight_arr = weight_arr[mask]
    return pred_arr, actual_arr, weight_arr


def _aggregate_by_weight_bins(
    *,
    sort_key: np.ndarray,
    weight: np.ndarray,
    values: Dict[str, np.ndarray],
    n_bins: int,
) -> pd.DataFrame:
    n_bins = max(1, int(n_bins))
    if sort_key.size == 0:
        payload: Dict[str, np.ndarray] = {
            "bins": np.asarray([], dtype=np.int64),
            "weight": np.asarray([], dtype=float),
        }
        for name in values:
            payload[name] = np.asarray([], dtype=float)
        return pd.DataFrame(payload)

    order = np.argsort(sort_key, kind="mergesort")
    weight_sorted = np.asarray(weight[order], dtype=float)
    total_weight = float(np.sum(weight_sorted))
    if total_weight <= EPS:
        bin_idx = np.zeros(weight_sorted.shape[0], dtype=np.int64)
    else:
        cum_weight = np.cumsum(weight_sorted, dtype=np.float64)
        bin_idx = np.floor(cum_weight * float(n_bins) / total_weight).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    weight_sums = np.bincount(bin_idx, weights=weight_sorted, minlength=n_bins)
    used_bins = np.flatnonzero(weight_sums > 0)
    if used_bins.size == 0:
        used_bins = np.asarray([0], dtype=np.int64)

    payload = {
        "bins": used_bins.astype(np.int64, copy=False),
        "weight": weight_sums[used_bins],
    }
    for name, arr in values.items():
        arr_sorted = np.asarray(arr[order], dtype=float)
        payload[name] = np.bincount(
            bin_idx,
            weights=arr_sorted,
            minlength=n_bins,
        )[used_bins]
    return pd.DataFrame(payload)


def lift_table(
    pred: Sequence[float],
    actual: Sequence[float],
    weight: Optional[Sequence[float]] = None,
    *,
    n_bins: int = 10,
    pred_weighted: bool = False,
    actual_weighted: bool = True,
) -> pd.DataFrame:
    """Compute lift table for a single model.

    pred/actual should be 1d arrays. If pred_weighted/actual_weighted is True,
    the value is already multiplied by weight and will not be re-weighted.
    """
    pred_arr, actual_arr, weight_arr = _align_arrays(pred, actual, weight)
    weight_safe = np.maximum(weight_arr, EPS)

    if pred_weighted:
        pred_raw = pred_arr / weight_safe
        w_pred = pred_arr
    else:
        pred_raw = pred_arr
        w_pred = pred_arr * weight_arr

    if actual_weighted:
        w_act = actual_arr
    else:
        w_act = actual_arr * weight_arr

    plot_data = _aggregate_by_weight_bins(
        sort_key=pred_raw,
        weight=weight_arr,
        values={
            "w_pred": w_pred,
            "act": w_act,
        },
        n_bins=n_bins,
    )
    denom = np.maximum(plot_data["weight"].to_numpy(dtype=float, copy=False), EPS)
    plot_data["exp_v"] = plot_data["w_pred"].to_numpy(dtype=float, copy=False) / denom
    plot_data["act_v"] = plot_data["act"].to_numpy(dtype=float, copy=False) / denom
    return plot_data


def plot_lift_curve(
    pred: Sequence[float],
    actual: Sequence[float],
    weight: Optional[Sequence[float]] = None,
    *,
    n_bins: int = 10,
    title: str = "Lift Chart",
    pred_label: str = "Predicted",
    act_label: str = "Actual",
    weight_label: str = "Earned Exposure",
    pred_weighted: bool = False,
    actual_weighted: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    style = style or PlotStyle()
    plot_data = lift_table(
        pred,
        actual,
        weight,
        n_bins=n_bins,
        pred_weighted=pred_weighted,
        actual_weighted=actual_weighted,
    )

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.plot(plot_data.index, plot_data["act_v"], label=act_label, color="red")
    ax.plot(plot_data.index, plot_data["exp_v"], label=pred_label, color="blue")
    ax.set_title(title, fontsize=style.title_size)
    ax.set_xticks(plot_data.index)
    ax.set_xticklabels(plot_data.index, rotation=90, fontsize=style.tick_size)
    ax.tick_params(axis="y", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="upper left", fontsize=style.legend_size, frameon=False)
    ax.margins(0.05)

    ax2 = ax.twinx()
    ax2.bar(
        plot_data.index,
        plot_data["weight"],
        alpha=0.5,
        color=style.weight_color,
        label=weight_label,
    )
    ax2.tick_params(axis="y", labelsize=style.tick_size)
    ax2.legend(loc="upper right", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def double_lift_table(
    pred1: Sequence[float],
    pred2: Sequence[float],
    actual: Sequence[float],
    weight: Optional[Sequence[float]] = None,
    *,
    n_bins: int = 10,
    pred1_weighted: bool = False,
    pred2_weighted: bool = False,
    actual_weighted: bool = True,
) -> pd.DataFrame:
    pred1_arr = _to_1d(pred1, "pred1")
    pred2_arr = _to_1d(pred2, "pred2")
    actual_arr = _to_1d(actual, "actual")
    if len(pred1_arr) != len(pred2_arr):
        raise ValueError("pred1 and pred2 must have the same length.")
    if len(pred1_arr) != len(actual_arr):
        raise ValueError("pred1 and actual must have the same length.")
    if weight is None:
        weight_arr = np.ones_like(pred1_arr, dtype=float)
    else:
        weight_arr = _to_1d(weight, "weight")
        if len(weight_arr) != len(pred1_arr):
            raise ValueError("weight must have the same length as pred1.")

    # Keep all arrays aligned under one shared finite-mask.
    mask = (
        np.isfinite(pred1_arr)
        & np.isfinite(pred2_arr)
        & np.isfinite(actual_arr)
        & np.isfinite(weight_arr)
    )
    pred1_arr = pred1_arr[mask]
    pred2_arr = pred2_arr[mask]
    actual_arr = actual_arr[mask]
    weight_arr = weight_arr[mask]

    weight_safe = np.maximum(weight_arr, EPS)
    pred1_raw = pred1_arr / weight_safe if pred1_weighted else pred1_arr
    pred2_raw = pred2_arr / weight_safe if pred2_weighted else pred2_arr

    w_pred1 = pred1_raw * weight_arr
    w_pred2 = pred2_raw * weight_arr
    w_act = actual_arr if actual_weighted else actual_arr * weight_arr

    plot_data = _aggregate_by_weight_bins(
        sort_key=pred1_raw / np.maximum(pred2_raw, EPS),
        weight=weight_arr,
        values={
            "pred1": w_pred1,
            "pred2": w_pred2,
            "act": w_act,
        },
        n_bins=n_bins,
    )
    denom = np.maximum(plot_data["act"].to_numpy(dtype=float, copy=False), EPS)
    plot_data["exp_v1"] = plot_data["pred1"].to_numpy(dtype=float, copy=False) / denom
    plot_data["exp_v2"] = plot_data["pred2"].to_numpy(dtype=float, copy=False) / denom
    plot_data["act_v"] = plot_data["act"].to_numpy(dtype=float, copy=False) / denom
    return plot_data


def plot_double_lift_curve(
    pred1: Sequence[float],
    pred2: Sequence[float],
    actual: Sequence[float],
    weight: Optional[Sequence[float]] = None,
    *,
    n_bins: int = 10,
    title: str = "Double Lift Chart",
    label1: str = "Model 1",
    label2: str = "Model 2",
    act_label: str = "Actual",
    weight_label: str = "Earned Exposure",
    pred1_weighted: bool = False,
    pred2_weighted: bool = False,
    actual_weighted: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    style = style or PlotStyle()
    plot_data = double_lift_table(
        pred1,
        pred2,
        actual,
        weight,
        n_bins=n_bins,
        pred1_weighted=pred1_weighted,
        pred2_weighted=pred2_weighted,
        actual_weighted=actual_weighted,
    )

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.plot(plot_data.index, plot_data["act_v"], label=act_label, color="red")
    ax.plot(plot_data.index, plot_data["exp_v1"], label=label1, color="blue")
    ax.plot(plot_data.index, plot_data["exp_v2"], label=label2, color="black")
    ax.set_title(title, fontsize=style.title_size)
    ax.set_xticks(plot_data.index)
    ax.set_xticklabels(plot_data.index, rotation=90, fontsize=style.tick_size)
    ax.set_xlabel(f"{label1} / {label2}", fontsize=style.label_size)
    ax.tick_params(axis="y", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="upper left", fontsize=style.legend_size, frameon=False)
    ax.margins(0.1)

    ax2 = ax.twinx()
    ax2.bar(
        plot_data.index,
        plot_data["weight"],
        alpha=0.5,
        color=style.weight_color,
        label=weight_label,
    )
    ax2.tick_params(axis="y", labelsize=style.tick_size)
    ax2.legend(loc="upper right", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_roc_curves(
    y_true: Sequence[float],
    scores: Mapping[str, Sequence[float]],
    *,
    weight: Optional[Sequence[float]] = None,
    title: str = "ROC Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    _require_sklearn("plot_roc_curves")
    style = style or PlotStyle()

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    for idx, (label, score) in enumerate(scores.items()):
        s_arr, y_arr, w_arr = _align_arrays(score, y_true, weight)
        try:
            fpr, tpr, _ = roc_curve(y_arr, s_arr, sample_weight=w_arr)
        except TypeError:
            fpr, tpr, _ = roc_curve(y_arr, s_arr)
        auc_val = auc(fpr, tpr)
        color = style.palette[idx % len(style.palette)]
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=style.label_size)
    ax.set_ylabel("True Positive Rate", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="lower right", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_pr_curves(
    y_true: Sequence[float],
    scores: Mapping[str, Sequence[float]],
    *,
    weight: Optional[Sequence[float]] = None,
    title: str = "Precision-Recall Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    if precision_recall_curve is None or average_precision_score is None:
        raise RuntimeError("plot_pr_curves requires scikit-learn to be installed.")
    style = style or PlotStyle()

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    for idx, (label, score) in enumerate(scores.items()):
        s_arr, y_arr, w_arr = _align_arrays(score, y_true, weight)
        try:
            precision, recall, _ = precision_recall_curve(
                y_arr, s_arr, sample_weight=w_arr
            )
            ap = average_precision_score(y_arr, s_arr, sample_weight=w_arr)
        except TypeError:
            precision, recall, _ = precision_recall_curve(y_arr, s_arr)
            ap = average_precision_score(y_arr, s_arr)
        color = style.palette[idx % len(style.palette)]
        ax.plot(recall, precision, color=color, label=f"{label} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=style.label_size)
    ax.set_ylabel("Precision", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="lower left", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_ks_curve(
    y_true: Sequence[float],
    score: Sequence[float],
    *,
    weight: Optional[Sequence[float]] = None,
    title: str = "KS Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    _require_sklearn("plot_ks_curve")
    style = style or PlotStyle()

    s_arr, y_arr, w_arr = _align_arrays(score, y_true, weight)
    try:
        fpr, tpr, thresholds = roc_curve(y_arr, s_arr, sample_weight=w_arr)
    except TypeError:
        fpr, tpr, thresholds = roc_curve(y_arr, s_arr)
    ks_vals = tpr - fpr
    ks_idx = int(np.argmax(ks_vals))
    ks_val = float(ks_vals[ks_idx])

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.plot(thresholds, tpr, label="TPR", color=style.palette[0])
    ax.plot(thresholds, fpr, label="FPR", color=style.palette[1])
    ax.plot(thresholds, ks_vals, label=f"KS={ks_val:.3f}", color=style.palette[3])
    ax.set_title(title, fontsize=style.title_size)
    ax.set_xlabel("Threshold", fontsize=style.label_size)
    ax.set_ylabel("Rate", fontsize=style.label_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="best", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_calibration_curve(
    y_true: Sequence[float],
    score: Sequence[float],
    *,
    weight: Optional[Sequence[float]] = None,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    if calibration_curve is None:
        raise RuntimeError("plot_calibration_curve requires scikit-learn to be installed.")
    style = style or PlotStyle()

    s_arr, y_arr, w_arr = _align_arrays(score, y_true, weight)
    try:
        prob_true, prob_pred = calibration_curve(
            y_arr,
            s_arr,
            n_bins=max(2, int(n_bins)),
            strategy="quantile",
            sample_weight=w_arr,
        )
    except TypeError:
        prob_true, prob_pred = calibration_curve(
            y_arr,
            s_arr,
            n_bins=max(2, int(n_bins)),
            strategy="quantile",
        )

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.plot(prob_pred, prob_true, marker="o", label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
    ax.set_xlabel("Mean Predicted", fontsize=style.label_size)
    ax.set_ylabel("Mean Observed", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="best", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_conversion_lift(
    pred: Sequence[float],
    actual_binary: Sequence[float],
    weight: Optional[Sequence[float]] = None,
    *,
    n_bins: int = 20,
    title: str = "Conversion Lift",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    style = style or PlotStyle()
    pred_arr, actual_arr, weight_arr = _align_arrays(pred, actual_binary, weight)

    weighted_actual = actual_arr * weight_arr
    lift_agg = _aggregate_by_weight_bins(
        sort_key=pred_arr,
        weight=weight_arr,
        values={"weighted_actual": weighted_actual},
        n_bins=max(2, int(n_bins)),
    )
    lift_agg.rename(columns={"weight": "total_weight"}, inplace=True)
    lift_agg["conversion_rate"] = (
        lift_agg["weighted_actual"].to_numpy(dtype=float, copy=False)
        / np.maximum(lift_agg["total_weight"].to_numpy(dtype=float, copy=False), EPS)
    )
    overall_rate = float(np.sum(weighted_actual)) / max(float(np.sum(weight_arr)), EPS)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.axhline(
        y=overall_rate,
        color="gray",
        linestyle="--",
        label=f"Overall ({overall_rate:.2%})",
    )
    ax.plot(
        lift_agg["bins"],
        lift_agg["conversion_rate"],
        marker="o",
        linestyle="-",
        label="Actual Rate",
    )
    ax.set_title(title, fontsize=style.title_size)
    ax.set_xlabel("Score Bin", fontsize=style.label_size)
    ax.set_ylabel("Conversion Rate", fontsize=style.label_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="best", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig
