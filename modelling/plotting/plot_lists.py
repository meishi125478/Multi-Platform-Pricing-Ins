from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ins_pricing.modelling.plotting.common import PlotStyle, plt
from ins_pricing.modelling.plotting.curves import (
    plot_double_lift_curve,
    plot_lift_curve,
)


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _to_array(values: Sequence[float], name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required.")
    return np.asarray(values, dtype=float).reshape(-1)


def _resolve_pred_pairs_from_df(
    df: pd.DataFrame,
    pred_cols: Optional[Sequence[str] | Mapping[str, object]],
    pred_labels: Optional[Sequence[str]],
) -> List[Tuple[str, np.ndarray]]:
    if pred_cols is None:
        raise ValueError("pred_cols is required when data is a DataFrame.")

    pairs: List[Tuple[str, np.ndarray]] = []
    if isinstance(pred_cols, Mapping):
        for label, col in pred_cols.items():
            if isinstance(col, str) and col in df.columns:
                arr = df[col].to_numpy()
            else:
                arr = np.asarray(col, dtype=float).reshape(-1)
            pairs.append((str(label), arr))
        return pairs

    if isinstance(pred_cols, str):
        label = pred_cols if not pred_labels else str(pred_labels[0])
        return [(label, df[pred_cols].to_numpy())]

    labels = list(pred_labels) if pred_labels else [str(c) for c in pred_cols]
    for col, label in zip(pred_cols, labels):
        if isinstance(col, str) and col in df.columns:
            arr = df[col].to_numpy()
        else:
            arr = np.asarray(col, dtype=float).reshape(-1)
        pairs.append((str(label), arr))
    return pairs


def _resolve_pred_pairs_from_arrays(
    preds: object,
    pred_labels: Optional[Sequence[str]],
) -> List[Tuple[str, np.ndarray]]:
    if isinstance(preds, Mapping):
        return [(str(k), np.asarray(v, dtype=float).reshape(-1)) for k, v in preds.items()]

    if isinstance(preds, (list, tuple)):
        labels = list(pred_labels) if pred_labels else [f"model_{idx + 1}" for idx in range(len(preds))]
        return [
            (str(label), np.asarray(arr, dtype=float).reshape(-1))
            for label, arr in zip(labels, preds)
        ]

    return [("model_1", np.asarray(preds, dtype=float).reshape(-1))]


def plot_lift_list(
    data_or_preds: pd.DataFrame | Mapping[str, object] | Sequence[object],
    pred_cols: Optional[Sequence[str] | Mapping[str, object]] = None,
    actual: Optional[Sequence[float]] = None,
    weight: Optional[Sequence[float]] = None,
    *,
    actual_col: str = "w_act",
    weight_col: str = "weight",
    pred_labels: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    pred_weighted: bool = False,
    actual_weighted: bool = True,
    show: bool = False,
    save_dir: Optional[str] = None,
    filename_prefix: str = "lift",
    style: Optional[PlotStyle] = None,
) -> List[plt.Figure]:
    if isinstance(data_or_preds, pd.DataFrame):
        df = data_or_preds
        pairs = _resolve_pred_pairs_from_df(df, pred_cols, pred_labels)
        actual_arr = _to_array(df[actual_col], actual_col) if actual is None else _to_array(actual, "actual")
        weight_arr = None
        if weight is not None:
            weight_arr = _to_array(weight, "weight")
        elif weight_col in df.columns:
            weight_arr = _to_array(df[weight_col], weight_col)
    else:
        pairs = _resolve_pred_pairs_from_arrays(data_or_preds, pred_labels)
        actual_arr = _to_array(actual, "actual")
        weight_arr = _to_array(weight, "weight") if weight is not None else None

    figs: List[plt.Figure] = []
    for label, pred in pairs:
        save_path = None
        if save_dir:
            safe_dir = save_dir.rstrip("/\\")
            save_path = f"{safe_dir}/{filename_prefix}_{_safe_tag(label)}.png"
        fig = plot_lift_curve(
            pred,
            actual_arr,
            weight_arr,
            n_bins=n_bins,
            title=f"{label} Lift Chart",
            pred_label="Predicted",
            act_label="Actual",
            weight_label="Earned Exposure",
            pred_weighted=pred_weighted,
            actual_weighted=actual_weighted,
            show=show,
            save_path=save_path,
            style=style,
        )
        figs.append(fig)
    return figs


def plot_dlift_list(
    data_or_preds: pd.DataFrame | Mapping[str, object] | Sequence[object],
    pred_cols: Optional[Sequence[str] | Mapping[str, object]] = None,
    actual: Optional[Sequence[float]] = None,
    weight: Optional[Sequence[float]] = None,
    *,
    actual_col: str = "w_act",
    weight_col: str = "weight",
    pred_labels: Optional[Sequence[str]] = None,
    pairs: Optional[Iterable[Tuple[object, object]]] = None,
    n_bins: int = 10,
    pred_weighted: bool = False,
    actual_weighted: bool = True,
    show: bool = False,
    save_dir: Optional[str] = None,
    filename_prefix: str = "double_lift",
    style: Optional[PlotStyle] = None,
) -> List[plt.Figure]:
    if isinstance(data_or_preds, pd.DataFrame):
        df = data_or_preds
        pairs_list = _resolve_pred_pairs_from_df(df, pred_cols, pred_labels)
        actual_arr = _to_array(df[actual_col], actual_col) if actual is None else _to_array(actual, "actual")
        weight_arr = None
        if weight is not None:
            weight_arr = _to_array(weight, "weight")
        elif weight_col in df.columns:
            weight_arr = _to_array(df[weight_col], weight_col)
    else:
        pairs_list = _resolve_pred_pairs_from_arrays(data_or_preds, pred_labels)
        actual_arr = _to_array(actual, "actual")
        weight_arr = _to_array(weight, "weight") if weight is not None else None

    pred_map = {label: arr for label, arr in pairs_list}
    labels = [label for label, _ in pairs_list]

    pair_labels: List[Tuple[str, str]] = []
    if pairs is None:
        for idx, first in enumerate(labels):
            for second in labels[idx + 1:]:
                pair_labels.append((first, second))
    else:
        for left, right in pairs:
            if isinstance(left, int):
                left_label = labels[left]
            else:
                left_label = str(left)
            if isinstance(right, int):
                right_label = labels[right]
            else:
                right_label = str(right)
            pair_labels.append((left_label, right_label))

    figs: List[plt.Figure] = []
    for label1, label2 in pair_labels:
        pred1 = pred_map.get(label1)
        pred2 = pred_map.get(label2)
        if pred1 is None or pred2 is None:
            continue
        save_path = None
        if save_dir:
            safe_dir = save_dir.rstrip("/\\")
            tag = f"{_safe_tag(label1)}_vs_{_safe_tag(label2)}"
            save_path = f"{safe_dir}/{filename_prefix}_{tag}.png"
        fig = plot_double_lift_curve(
            pred1,
            pred2,
            actual_arr,
            weight_arr,
            n_bins=n_bins,
            title=f"Double Lift: {label1} vs {label2}",
            label1=label1,
            label2=label2,
            pred1_weighted=pred_weighted,
            pred2_weighted=pred_weighted,
            actual_weighted=actual_weighted,
            show=show,
            save_path=save_path,
            style=style,
        )
        figs.append(fig)
    return figs


__all__ = [
    "plot_lift_list",
    "plot_dlift_list",
]
