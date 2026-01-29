from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import PlotStyle, finalize_figure, plt


def _to_series(
    importance: Mapping[str, float]
    | Sequence[Tuple[str, float]]
    | pd.Series
    | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    if isinstance(importance, pd.Series):
        return importance.copy()
    if isinstance(importance, Mapping):
        return pd.Series(dict(importance))
    if isinstance(importance, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names is required when importance is an array.")
        return pd.Series(importance, index=list(feature_names))
    return pd.Series(dict(importance))


def shap_importance(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
) -> pd.Series:
    if shap_values.ndim == 3:
        shap_values = shap_values[0]
    if shap_values.ndim != 2:
        raise ValueError("shap_values should be 2d (n_samples, n_features).")
    scores = np.abs(shap_values).mean(axis=0)
    return pd.Series(scores, index=list(feature_names))


def plot_feature_importance(
    importance: Mapping[str, float]
    | Sequence[Tuple[str, float]]
    | pd.Series
    | np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    top_n: int = 30,
    title: str = "Feature Importance",
    sort_by: str = "abs",
    descending: bool = True,
    show_values: bool = False,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    style = style or PlotStyle()
    series = _to_series(importance, feature_names=feature_names)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()

    if sort_by not in {"abs", "value"}:
        raise ValueError("sort_by must be 'abs' or 'value'.")
    sort_key = series.abs() if sort_by == "abs" else series
    series = series.loc[sort_key.sort_values(ascending=not descending).index]

    if top_n > 0:
        series = series.head(int(top_n))

    created_fig = ax is None
    if created_fig:
        height = max(3.0, 0.3 * len(series))
        fig, ax = plt.subplots(figsize=(style.figsize[0], height))
    else:
        fig = ax.figure

    y_pos = np.arange(len(series))
    ax.barh(y_pos, series.values, color=style.palette[0])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(series.index, fontsize=style.tick_size)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="x", labelsize=style.tick_size)
    if style.grid:
        ax.grid(True, axis="x", linestyle=style.grid_style, alpha=style.grid_alpha)

    if show_values:
        for idx, val in enumerate(series.values):
            ax.text(val, idx, f" {val:.3f}", va="center", fontsize=style.tick_size)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_shap_importance(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_n: int = 30,
    title: str = "SHAP Importance",
    show_values: bool = False,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    series = shap_importance(shap_values, feature_names)
    return plot_feature_importance(
        series,
        top_n=top_n,
        title=title,
        sort_by="abs",
        descending=True,
        show_values=show_values,
        ax=ax,
        show=show,
        save_path=save_path,
        style=style,
    )
