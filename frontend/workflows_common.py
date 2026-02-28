"""Shared helpers for frontend workflow execution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _dedupe_list(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in values or []:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _drop_duplicate_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.columns.duplicated().any():
        dupes = [str(x) for x in df.columns[df.columns.duplicated()]]
        print(f"[Warn] {label}: dropping duplicate columns: {sorted(set(dupes))}")
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _resolve_output_dir(cfg_obj: dict, cfg_file_path: Path) -> str:
    output_dir = cfg_obj.get("output_dir", "./Results")
    return str((cfg_file_path.parent / output_dir).resolve())


def _resolve_plot_style(cfg_obj: dict) -> str:
    return str(cfg_obj.get("plot_path_style", "nested") or "nested").strip().lower()


def _resolve_plot_path(output_root: str, plot_style: str, subdir: str, filename: str) -> str:
    plot_root = Path(output_root) / "plot"
    if plot_style in {"flat", "root"}:
        return str((plot_root / filename).resolve())
    if subdir:
        return str((plot_root / subdir / filename).resolve())
    return str((plot_root / filename).resolve())


def _safe_tag(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _resolve_data_path(cfg: dict, cfg_path: Path, model_name: str) -> Path:
    data_dir = cfg.get("data_dir", ".")
    data_format = cfg.get("data_format", "csv")
    data_path_template = cfg.get("data_path_template", "{model_name}.{ext}")
    filename = data_path_template.format(model_name=model_name, ext=data_format)
    return (cfg_path.parent / data_dir / filename).resolve()


def _infer_categorical_features(
    df: pd.DataFrame,
    feature_list: Sequence[str],
    *,
    max_unique: int = 50,
    max_ratio: float = 0.05,
) -> List[str]:
    categorical: List[str] = []
    n_rows = max(1, len(df))
    for feature in feature_list:
        if feature not in df.columns:
            continue
        nunique = int(df[feature].nunique(dropna=True))
        ratio = nunique / float(n_rows)
        if nunique <= max_unique or ratio <= max_ratio:
            categorical.append(feature)
    return categorical

