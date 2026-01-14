from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.tri as mtri

from .common import EPS, PlotStyle, finalize_figure, plt

try:  # optional map basemap support
    import contextily as cx
except Exception:  # pragma: no cover - optional dependency
    cx = None


_MERCATOR_MAX_LAT = 85.05112878
_MERCATOR_FACTOR = 20037508.34


def _require_contextily(func_name: str) -> None:
    if cx is None:
        raise RuntimeError(
            f"{func_name} requires contextily. Install it via 'pip install contextily'."
        )


def _lonlat_to_mercator(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat = np.clip(lat, -_MERCATOR_MAX_LAT, _MERCATOR_MAX_LAT)
    x = lon * _MERCATOR_FACTOR / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * _MERCATOR_FACTOR / np.pi
    return x, y


def _apply_bounds(ax: plt.Axes, x: np.ndarray, y: np.ndarray, padding: float) -> None:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding
    if pad_x == 0:
        pad_x = 1.0
    if pad_y == 0:
        pad_y = 1.0
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)


def _resolve_basemap(source):
    if cx is None or source is None:
        return source
    if isinstance(source, str):
        provider = cx.providers
        for part in source.split("."):
            if isinstance(provider, dict):
                provider = provider[part]
            else:
                provider = getattr(provider, part)
        return provider
    return source


def _sanitize_geo(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    weight_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    w = None
    if weight_col:
        w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)

    if w is None:
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(w)
        w = w[mask]
    return x[mask], y[mask], z[mask], w


def _downsample_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: Optional[np.ndarray],
    max_points: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if max_points is None:
        return x, y, z, w
    max_points = int(max_points)
    if max_points <= 0 or len(x) <= max_points:
        return x, y, z, w
    rng = np.random.default_rng(13)
    idx = rng.choice(len(x), size=max_points, replace=False)
    if w is None:
        return x[idx], y[idx], z[idx], None
    return x[idx], y[idx], z[idx], w[idx]


def plot_geo_heatmap(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    value_col: str,
    weight_col: Optional[str] = None,
    bins: int | Tuple[int, int] = 50,
    agg: str = "mean",
    cmap: str = "YlOrRd",
    title: str = "Geo Heatmap",
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    style = style or PlotStyle()
    if agg not in {"mean", "sum"}:
        raise ValueError("agg must be 'mean' or 'sum'.")
    x, y, z, w = _sanitize_geo(df, x_col, y_col, value_col, weight_col)

    if isinstance(bins, int):
        bins = (bins, bins)

    if w is None:
        sum_z, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=z)
        if agg == "sum":
            grid = sum_z
        else:
            count, _, _ = np.histogram2d(x, y, bins=bins)
            grid = sum_z / np.maximum(count, 1.0)
    else:
        sum_w, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=w)
        sum_zw, _, _ = np.histogram2d(x, y, bins=bins, weights=z * w)
        grid = sum_zw / np.maximum(sum_w, EPS)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        grid.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col, fontsize=style.label_size)
    cbar.ax.tick_params(labelsize=style.tick_size)

    ax.set_xlabel(x_col, fontsize=style.label_size)
    ax.set_ylabel(y_col, fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_geo_contour(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    value_col: str,
    weight_col: Optional[str] = None,
    max_points: Optional[int] = None,
    levels: int | Sequence[float] = 10,
    cmap: str = "viridis",
    title: str = "Geo Contour",
    ax: Optional[plt.Axes] = None,
    show_points: bool = False,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    style = style or PlotStyle()
    x, y, z, w = _sanitize_geo(df, x_col, y_col, value_col, weight_col)
    x, y, z, w = _downsample_points(x, y, z, w, max_points)

    if w is not None:
        z = z * w

    triang = mtri.Triangulation(x, y)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    contour = ax.tricontourf(triang, z, levels=levels, cmap=cmap)
    if show_points:
        ax.scatter(x, y, s=6, c="k", alpha=0.2)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(value_col, fontsize=style.label_size)
    cbar.ax.tick_params(labelsize=style.tick_size)

    ax.set_xlabel(x_col, fontsize=style.label_size)
    ax.set_ylabel(y_col, fontsize=style.label_size)
    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_geo_heatmap_on_map(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    value_col: str,
    weight_col: Optional[str] = None,
    bins: int | Tuple[int, int] = 100,
    agg: str = "mean",
    cmap: str = "YlOrRd",
    alpha: float = 0.6,
    basemap: Optional[object] = "CartoDB.Positron",
    zoom: Optional[int] = None,
    padding: float = 0.05,
    title: str = "Geo Heatmap (Map)",
    ax: Optional[plt.Axes] = None,
    show_points: bool = False,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    _require_contextily("plot_geo_heatmap_on_map")
    style = style or PlotStyle()
    if agg not in {"mean", "sum"}:
        raise ValueError("agg must be 'mean' or 'sum'.")
    lon, lat, z, w = _sanitize_geo(df, lon_col, lat_col, value_col, weight_col)
    x, y = _lonlat_to_mercator(lon, lat)

    if isinstance(bins, int):
        bins = (bins, bins)

    if w is None:
        sum_z, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=z)
        if agg == "sum":
            grid = sum_z
        else:
            count, _, _ = np.histogram2d(x, y, bins=bins)
            grid = sum_z / np.maximum(count, 1.0)
    else:
        sum_w, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=w)
        sum_zw, _, _ = np.histogram2d(x, y, bins=bins, weights=z * w)
        grid = sum_zw / np.maximum(sum_w, EPS)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    _apply_bounds(ax, x, y, padding)
    ax.set_aspect("equal", adjustable="box")

    source = _resolve_basemap(basemap)
    if source is not None:
        if zoom is None:
            cx.add_basemap(ax, source=source, crs="EPSG:3857")
        else:
            cx.add_basemap(ax, source=source, crs="EPSG:3857", zoom=zoom)

    im = ax.imshow(
        grid.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
    )
    if show_points:
        ax.scatter(x, y, s=6, c="k", alpha=0.25)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col, fontsize=style.label_size)
    cbar.ax.tick_params(labelsize=style.tick_size)

    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig


def plot_geo_contour_on_map(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    value_col: str,
    weight_col: Optional[str] = None,
    max_points: Optional[int] = None,
    levels: int | Sequence[float] = 10,
    cmap: str = "viridis",
    alpha: float = 0.6,
    basemap: Optional[object] = "CartoDB.Positron",
    zoom: Optional[int] = None,
    padding: float = 0.05,
    title: str = "Geo Contour (Map)",
    ax: Optional[plt.Axes] = None,
    show_points: bool = False,
    show: bool = False,
    save_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> plt.Figure:
    _require_contextily("plot_geo_contour_on_map")
    style = style or PlotStyle()
    lon, lat, z, w = _sanitize_geo(df, lon_col, lat_col, value_col, weight_col)
    lon, lat, z, w = _downsample_points(lon, lat, z, w, max_points)
    x, y = _lonlat_to_mercator(lon, lat)
    if w is not None:
        z = z * w

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    _apply_bounds(ax, x, y, padding)
    ax.set_aspect("equal", adjustable="box")

    source = _resolve_basemap(basemap)
    if source is not None:
        if zoom is None:
            cx.add_basemap(ax, source=source, crs="EPSG:3857")
        else:
            cx.add_basemap(ax, source=source, crs="EPSG:3857", zoom=zoom)

    triang = mtri.Triangulation(x, y)
    contour = ax.tricontourf(triang, z, levels=levels, cmap=cmap, alpha=alpha)
    if show_points:
        ax.scatter(x, y, s=6, c="k", alpha=0.25)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(value_col, fontsize=style.label_size)
    cbar.ax.tick_params(labelsize=style.tick_size)

    ax.set_title(title, fontsize=style.title_size)
    ax.tick_params(axis="both", labelsize=style.tick_size)

    if created_fig:
        finalize_figure(fig, save_path=save_path, show=show, style=style)

    return fig
