from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .common import EPS, PlotStyle, finalize_figure, plt


def plot_loss_curve(
    *,
    history: Optional[Mapping[str, Sequence[float]]] = None,
    train: Optional[Sequence[float]] = None,
    val: Optional[Sequence[float]] = None,
    title: str = "Loss vs. Epoch",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[plt.Figure]:
    style = style or PlotStyle()
    if history is not None:
        if train is None:
            train = history.get("train")
        if val is None:
            val = history.get("val")

    train_hist = list(train or [])
    val_hist = list(val or [])
    if not train_hist and not val_hist:
        return None

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    if train_hist:
        ax.plot(
            range(1, len(train_hist) + 1),
            train_hist,
            label="Train Loss",
            color="tab:blue",
        )
    if val_hist:
        ax.plot(
            range(1, len(val_hist) + 1),
            val_hist,
            label="Validation Loss",
            color="tab:orange",
        )

    ax.set_xlabel("Epoch", fontsize=style.label_size)
    ax.set_ylabel("Weighted Loss", fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    ax.legend(loc="best", fontsize=style.legend_size, frameon=False)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_oneway(
    df: pd.DataFrame,
    *,
    feature: str,
    weight_col: str,
    target_col: str,
    pred_col: Optional[str] = None,
    pred_weighted: bool = False,
    pred_label: Optional[str] = None,
    n_bins: int = 10,
    is_categorical: bool = False,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[plt.Figure]:
    if feature not in df.columns:
        raise KeyError(f"feature '{feature}' not found in data.")
    if weight_col not in df.columns:
        raise KeyError(f"weight_col '{weight_col}' not found in data.")
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in data.")
    if pred_col is not None and pred_col not in df.columns:
        raise KeyError(f"pred_col '{pred_col}' not found in data.")

    style = style or PlotStyle()
    title = title or f"Analysis of {feature}"

    if is_categorical:
        group_col = feature
        plot_source = df
    else:
        group_col = f"{feature}_bins"
        series = pd.to_numeric(df[feature], errors="coerce")
        try:
            bins = pd.qcut(series, n_bins, duplicates="drop")
        except ValueError:
            bins = pd.cut(series, bins=max(1, int(n_bins)), duplicates="drop")
        plot_source = df.assign(**{group_col: bins})

    if pred_col is not None:
        if pred_weighted:
            plot_source = plot_source.assign(_pred_w=plot_source[pred_col])
        else:
            plot_source = plot_source.assign(
                _pred_w=plot_source[pred_col] * plot_source[weight_col]
            )

    plot_data = plot_source.groupby([group_col], observed=True).sum(numeric_only=True)
    plot_data.reset_index(inplace=True)

    denom = np.maximum(plot_data[weight_col].to_numpy(dtype=float), EPS)
    plot_data["act_v"] = plot_data[target_col].to_numpy(dtype=float) / denom
    if pred_col is not None:
        plot_data["pred_v"] = plot_data["_pred_w"].to_numpy(dtype=float) / denom

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.plot(plot_data.index, plot_data["act_v"], label="Actual", color="red")
    if pred_col is not None:
        ax.plot(
            plot_data.index,
            plot_data["pred_v"],
            label=pred_label or "Predicted",
            color=style.palette[0],
        )
    ax.set_title(title, fontsize=style.title_size)
    ax.set_xticks(plot_data.index)
    labels = plot_data[group_col].astype(str).tolist()
    tick_size = 3 if len(labels) > 50 else style.tick_size
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_size)
    ax.tick_params(axis="y", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    if pred_col is not None:
        ax.legend(fontsize=style.legend_size)

    ax2 = ax.twinx()
    ax2.bar(
        plot_data.index,
        plot_data[weight_col],
        alpha=0.5,
        color=style.weight_color,
    )
    ax2.tick_params(axis="y", labelsize=style.tick_size)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig
