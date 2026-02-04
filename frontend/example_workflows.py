"""
Example workflows implemented in Python so the frontend can run
the same tasks as the example notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ins_pricing.cli.utils.cli_common import split_train_test
from ins_pricing.modelling.plotting import (
    PlotStyle,
    plot_double_lift_curve,
    plot_lift_curve,
    plot_oneway,
)
from ins_pricing.modelling.plotting.common import finalize_figure, plt
from ins_pricing.production.inference import load_predictor_from_config


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


def run_pre_oneway(
    *,
    data_path: str,
    model_name: str,
    target_col: str,
    weight_col: str,
    feature_list: str,
    categorical_features: str,
    n_bins: int = 10,
    holdout_ratio: Optional[float] = 0.25,
    rand_seed: int = 13,
    output_dir: Optional[str] = None,
) -> str:
    data_path = str(data_path or "").strip()
    if not data_path:
        raise ValueError("data_path is required.")
    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError("model_name is required.")

    raw_path = Path(data_path).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Data file not found: {raw_path}")

    raw = pd.read_csv(raw_path, low_memory=False)
    raw = _drop_duplicate_columns(raw, "raw").reset_index(drop=True)
    raw.fillna(0, inplace=True)

    features = _dedupe_list(_parse_csv_list(feature_list))
    cats = _dedupe_list(_parse_csv_list(categorical_features))

    if not features:
        raise ValueError("feature_list is empty.")

    missing = [f for f in features if f not in raw.columns]
    if missing:
        print(f"[Warn] Missing features removed: {missing}")
        features = [f for f in features if f in raw.columns]
        cats = [f for f in cats if f in raw.columns]

    if not cats:
        cats = _infer_categorical_features(raw, features)

    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else raw_path.parent / "Results" / "plot" / model_name / "oneway" / "pre"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if holdout_ratio is not None and float(holdout_ratio) > 0:
        train_df, _ = split_train_test(
            raw,
            holdout_ratio=float(holdout_ratio),
            strategy="random",
            rand_seed=int(rand_seed),
            reset_index_mode="none",
            ratio_label="holdout_ratio",
        )
        df = train_df.reset_index(drop=True).copy()
    else:
        df = raw.copy()

    print(f"Generating oneway plots for {len(features)} features...")
    saved = 0
    for i, feature in enumerate(features, 1):
        is_categorical = feature in cats
        try:
            save_path = out_dir / f"{feature}.png"
            plot_oneway(
                df,
                feature=feature,
                weight_col=weight_col,
                target_col=target_col,
                n_bins=int(n_bins),
                is_categorical=is_categorical,
                save_path=str(save_path),
                show=False,
            )
            if save_path.exists():
                saved += 1
            if i % 5 == 0 or i == len(features):
                print(f"  [{i}/{len(features)}] {feature}")
        except Exception as exc:
            print(f"  [Warn] {feature} failed: {exc}")

    print(f"Complete. Saved {saved}/{len(features)} plots to: {out_dir}")
    return f"Saved {saved} plots to {out_dir}"


def run_plot_direct(
    *,
    cfg_path: str,
    xgb_cfg_path: str,
    resn_cfg_path: str,
) -> str:
    cfg_path = Path(cfg_path).resolve()
    xgb_cfg_path = Path(xgb_cfg_path).resolve()
    resn_cfg_path = Path(resn_cfg_path).resolve()

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    xgb_cfg = json.loads(xgb_cfg_path.read_text(encoding="utf-8"))
    resn_cfg = json.loads(resn_cfg_path.read_text(encoding="utf-8"))

    model_name = f"{cfg['model_list'][0]}_{cfg['model_categories'][0]}"

    raw_data_dir = (cfg_path.parent / cfg["data_dir"]).resolve()
    raw_path = raw_data_dir / f"{model_name}.csv"
    raw = pd.read_csv(raw_path)
    raw = _drop_duplicate_columns(raw, "raw").reset_index(drop=True)
    raw.fillna(0, inplace=True)

    holdout_ratio = cfg.get("holdout_ratio", cfg.get("prop_test", 0.25))
    split_strategy = cfg.get("split_strategy", "random")
    split_group_col = cfg.get("split_group_col")
    split_time_col = cfg.get("split_time_col")
    split_time_ascending = cfg.get("split_time_ascending", True)
    rand_seed = cfg.get("rand_seed", 13)

    train_raw, test_raw = split_train_test(
        raw,
        holdout_ratio=holdout_ratio,
        strategy=split_strategy,
        group_col=split_group_col,
        time_col=split_time_col,
        time_ascending=split_time_ascending,
        rand_seed=rand_seed,
        reset_index_mode="none",
        ratio_label="holdout_ratio",
    )
    train_raw = _drop_duplicate_columns(train_raw, "train_raw")
    test_raw = _drop_duplicate_columns(test_raw, "test_raw")

    train_df = train_raw.copy()
    test_df = test_raw.copy()

    feature_list = _dedupe_list(cfg.get("feature_list") or [])
    categorical_features = _dedupe_list(cfg.get("categorical_features") or [])

    output_dir_map = {
        "xgb": _resolve_output_dir(xgb_cfg, xgb_cfg_path),
        "resn": _resolve_output_dir(resn_cfg, resn_cfg_path),
    }
    plot_path_style_map = {
        "xgb": _resolve_plot_style(xgb_cfg),
        "resn": _resolve_plot_style(resn_cfg),
    }

    def _get_plot_config(model_key: str) -> Tuple[str, str]:
        return (
            output_dir_map.get(model_key, _resolve_output_dir(cfg, cfg_path)),
            plot_path_style_map.get(model_key, _resolve_plot_style(cfg)),
        )

    def _load_predictor(cfg_path: Path, model_key: str):
        return load_predictor_from_config(cfg_path, model_key, model_name=model_name)

    model_cfg_map = {"xgb": xgb_cfg_path, "resn": resn_cfg_path}
    model_keys = cfg.get("model_keys") or ["xgb", "resn"]
    model_keys = [key for key in model_keys if key in model_cfg_map]
    if not model_keys:
        raise ValueError("No valid model keys found in plot config.")

    default_model_labels = {"xgb": "Xgboost", "resn": "ResNet"}

    def _model_label(model_key: str) -> str:
        labels = cfg.get("model_labels") or {}
        return str(labels.get(model_key, default_model_labels.get(model_key, model_key)))

    predictors = {key: _load_predictor(model_cfg_map[key], key) for key in model_keys}

    pred_train = {}
    pred_test = {}
    for key, predictor in predictors.items():
        pred_train[key] = predictor.predict(train_df).reshape(-1)
        pred_test[key] = predictor.predict(test_df).reshape(-1)
        if len(pred_train[key]) != len(train_df):
            raise ValueError(f"Train prediction length mismatch for {key}")
        if len(pred_test[key]) != len(test_df):
            raise ValueError(f"Test prediction length mismatch for {key}")

    plot_train = train_raw.copy()
    plot_test = test_raw.copy()
    for key in model_keys:
        plot_train[f"pred_{key}"] = pred_train[key]
        plot_test[f"pred_{key}"] = pred_test[key]

    weight_col = cfg["weight"]
    target_col = cfg["target"]

    if weight_col not in plot_train.columns:
        plot_train[weight_col] = 1.0
    if weight_col not in plot_test.columns:
        plot_test[weight_col] = 1.0
    if target_col in plot_train.columns:
        plot_train["w_act"] = plot_train[target_col] * plot_train[weight_col]
    if target_col in plot_test.columns:
        plot_test["w_act"] = plot_test[target_col] * plot_test[weight_col]

    if "w_act" not in plot_train.columns or plot_train["w_act"].isna().all():
        print("[Plot] Missing target values in train split; skip plots.")
        return "Skipped plotting due to missing target values."

    n_bins = cfg.get("plot", {}).get("n_bins", 10)
    oneway_features = feature_list
    oneway_categorical = set(categorical_features)

    for pred_key in model_keys:
        pred_label = _model_label(pred_key)
        pred_col = f"pred_{pred_key}"
        pred_tag = _safe_tag(pred_label or pred_col)
        output_root, plot_style = _get_plot_config(pred_key)
        for feature in oneway_features:
            if feature not in plot_train.columns:
                continue
            save_path = _resolve_plot_path(
                output_root,
                plot_style,
                f"{model_name}/oneway/post",
                f"00_{model_name}_{feature}_oneway_{pred_tag}.png",
            )
            plot_oneway(
                plot_train,
                feature=feature,
                weight_col=weight_col,
                target_col="w_act",
                pred_col=pred_col,
                pred_label=pred_label,
                n_bins=n_bins,
                is_categorical=feature in oneway_categorical,
                save_path=save_path,
                show=False,
            )

    datasets = []
    if "w_act" in plot_train.columns and not plot_train["w_act"].isna().all():
        datasets.append(("Train Data", plot_train))
    if "w_act" in plot_test.columns and not plot_test["w_act"].isna().all():
        datasets.append(("Test Data", plot_test))

    def _plot_lift_for_model(pred_key: str, pred_label: str) -> None:
        if not datasets:
            return
        output_root, plot_style = _get_plot_config(pred_key)
        style = PlotStyle()
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        if len(datasets) == 1:
            axes = [axes]
        for ax, (title, data) in zip(axes, datasets):
            pred_col = f"pred_{pred_key}"
            if pred_col not in data.columns:
                continue
            plot_lift_curve(
                data[pred_col].values,
                data["w_act"].values,
                data[weight_col].values,
                n_bins=n_bins,
                title=f"Lift Chart on {title}",
                pred_label="Predicted",
                act_label="Actual",
                weight_label="Earned Exposure",
                pred_weighted=False,
                actual_weighted=True,
                ax=ax,
                show=False,
                style=style,
            )
        plt.subplots_adjust(wspace=0.3)
        filename = f"01_{model_name}_{_safe_tag(pred_label)}_lift.png"
        save_path = _resolve_plot_path(
            output_root,
            plot_style,
            f"{model_name}/lift",
            filename,
        )
        finalize_figure(fig, save_path=save_path, show=False, style=style)

    for pred_key in model_keys:
        _plot_lift_for_model(pred_key, _model_label(pred_key))

    if (
        all(k in model_keys for k in ["xgb", "resn"])
        and all(f"pred_{k}" in plot_train.columns for k in ["xgb", "resn"])
        and datasets
    ):
        style = PlotStyle()
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        if len(datasets) == 1:
            axes = [axes]
        for ax, (title, data) in zip(axes, datasets):
            plot_double_lift_curve(
                data["pred_xgb"].values,
                data["pred_resn"].values,
                data["w_act"].values,
                data[weight_col].values,
                n_bins=n_bins,
                title=f"Double Lift Chart on {title}",
                label1="Xgboost",
                label2="ResNet",
                pred1_weighted=False,
                pred2_weighted=False,
                actual_weighted=True,
                ax=ax,
                show=False,
                style=style,
            )
        plt.subplots_adjust(wspace=0.3)
        save_path = _resolve_plot_path(
            _resolve_output_dir(cfg, cfg_path),
            _resolve_plot_style(cfg),
            "",
            f"02_{model_name}_dlift_xgb_vs_resn.png",
        )
        finalize_figure(fig, save_path=save_path, show=False, style=style)

    print("Plots saved under:")
    for key in model_keys:
        output_root, _ = _get_plot_config(key)
        print(f"  - {key}: {output_root}/plot/{model_name}")
    return "Plotting complete."


def run_plot_embed(
    *,
    cfg_path: str,
    xgb_cfg_path: str,
    resn_cfg_path: str,
    ft_cfg_path: str,
    use_runtime_ft_embedding: bool = False,
) -> str:
    cfg_path = Path(cfg_path).resolve()
    xgb_cfg_path = Path(xgb_cfg_path).resolve()
    resn_cfg_path = Path(resn_cfg_path).resolve()
    ft_cfg_path = Path(ft_cfg_path).resolve()

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    xgb_cfg = json.loads(xgb_cfg_path.read_text(encoding="utf-8"))
    resn_cfg = json.loads(resn_cfg_path.read_text(encoding="utf-8"))
    ft_cfg = json.loads(ft_cfg_path.read_text(encoding="utf-8"))

    model_name = f"{cfg['model_list'][0]}_{cfg['model_categories'][0]}"

    raw_data_dir = (ft_cfg_path.parent / ft_cfg["data_dir"]).resolve()
    raw_path = raw_data_dir / f"{model_name}.csv"
    raw = pd.read_csv(raw_path)
    raw = _drop_duplicate_columns(raw, "raw").reset_index(drop=True)
    raw.fillna(0, inplace=True)

    ft_output_dir = (ft_cfg_path.parent / ft_cfg["output_dir"]).resolve()
    ft_prefix = ft_cfg.get("ft_feature_prefix", "ft_emb")
    raw_feature_list = _dedupe_list(ft_cfg.get("feature_list") or [])
    raw_categorical_features = _dedupe_list(ft_cfg.get("categorical_features") or [])

    if ft_cfg.get("geo_feature_nmes"):
        raise ValueError("FT inference with geo tokens is not supported in this workflow.")

    holdout_ratio = cfg.get("holdout_ratio", cfg.get("prop_test", 0.25))
    split_strategy = cfg.get("split_strategy", "random")
    split_group_col = cfg.get("split_group_col")
    split_time_col = cfg.get("split_time_col")
    split_time_ascending = cfg.get("split_time_ascending", True)
    rand_seed = cfg.get("rand_seed", 13)

    train_raw, test_raw = split_train_test(
        raw,
        holdout_ratio=holdout_ratio,
        strategy=split_strategy,
        group_col=split_group_col,
        time_col=split_time_col,
        time_ascending=split_time_ascending,
        rand_seed=rand_seed,
        reset_index_mode="none",
        ratio_label="holdout_ratio",
    )
    train_raw = _drop_duplicate_columns(train_raw, "train_raw")
    test_raw = _drop_duplicate_columns(test_raw, "test_raw")

    if use_runtime_ft_embedding:
        import torch

        ft_model_path = ft_output_dir / "model" / f"01_{model_name}_FTTransformer.pth"
        ft_payload = torch.load(ft_model_path, map_location="cpu")
        ft_model = ft_payload["model"] if isinstance(ft_payload, dict) and "model" in ft_payload else ft_payload

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(ft_model, "device"):
            ft_model.device = device
        if hasattr(ft_model, "to"):
            ft_model.to(device)
        if hasattr(ft_model, "ft"):
            ft_model.ft.to(device)

        emb_train = ft_model.predict(train_raw, return_embedding=True)
        emb_cols = [f"pred_{ft_prefix}_{i}" for i in range(emb_train.shape[1])]
        train_df = train_raw.copy()
        train_df[emb_cols] = emb_train

        emb_test = ft_model.predict(test_raw, return_embedding=True)
        test_df = test_raw.copy()
        test_df[emb_cols] = emb_test
    else:
        embed_data_dir = (cfg_path.parent / cfg["data_dir"]).resolve()
        embed_path = embed_data_dir / f"{model_name}.csv"
        embed_df = pd.read_csv(embed_path)
        embed_df = _drop_duplicate_columns(embed_df, "embed").reset_index(drop=True)
        embed_df.fillna(0, inplace=True)
        if len(embed_df) != len(raw):
            raise ValueError(
                f"Row count mismatch: raw={len(raw)}, embed={len(embed_df)}. "
                "Cannot align predictions to raw features."
            )
        train_df = embed_df.loc[train_raw.index].copy()
        test_df = embed_df.loc[test_raw.index].copy()

    feature_list = _dedupe_list(cfg.get("feature_list") or [])
    categorical_features = _dedupe_list(cfg.get("categorical_features") or [])

    output_dir_map = {
        "xgb": _resolve_output_dir(xgb_cfg, xgb_cfg_path),
        "resn": _resolve_output_dir(resn_cfg, resn_cfg_path),
    }
    plot_path_style_map = {
        "xgb": _resolve_plot_style(xgb_cfg),
        "resn": _resolve_plot_style(resn_cfg),
    }

    def _get_plot_config(model_key: str) -> Tuple[str, str]:
        return (
            output_dir_map.get(model_key, _resolve_output_dir(cfg, cfg_path)),
            plot_path_style_map.get(model_key, _resolve_plot_style(cfg)),
        )

    def _load_predictor(cfg_path: Path, model_key: str):
        return load_predictor_from_config(cfg_path, model_key, model_name=model_name)

    model_cfg_map = {"xgb": xgb_cfg_path, "resn": resn_cfg_path}
    model_keys = cfg.get("model_keys") or ["xgb", "resn"]
    model_keys = [key for key in model_keys if key in model_cfg_map]
    if not model_keys:
        raise ValueError("No valid model keys found in plot config.")

    default_model_labels = {"xgb": "Xgboost", "resn": "ResNet"}

    def _model_label(model_key: str) -> str:
        labels = cfg.get("model_labels") or {}
        return str(labels.get(model_key, default_model_labels.get(model_key, model_key)))

    predictors = {key: _load_predictor(model_cfg_map[key], key) for key in model_keys}
    pred_train = {}
    pred_test = {}
    for key, predictor in predictors.items():
        pred_train[key] = predictor.predict(train_df).reshape(-1)
        pred_test[key] = predictor.predict(test_df).reshape(-1)
        if len(pred_train[key]) != len(train_df):
            raise ValueError(f"Train prediction length mismatch for {key}")
        if len(pred_test[key]) != len(test_df):
            raise ValueError(f"Test prediction length mismatch for {key}")

    plot_train = train_raw.copy()
    plot_test = test_raw.copy()
    for key in model_keys:
        plot_train[f"pred_{key}"] = pred_train[key]
        plot_test[f"pred_{key}"] = pred_test[key]

    weight_col = cfg["weight"]
    target_col = cfg["target"]
    if weight_col not in plot_train.columns:
        plot_train[weight_col] = 1.0
    if weight_col not in plot_test.columns:
        plot_test[weight_col] = 1.0
    if target_col in plot_train.columns:
        plot_train["w_act"] = plot_train[target_col] * plot_train[weight_col]
    if target_col in plot_test.columns:
        plot_test["w_act"] = plot_test[target_col] * plot_test[weight_col]

    if "w_act" not in plot_train.columns or plot_train["w_act"].isna().all():
        print("[Plot] Missing target values in train split; skip plots.")
        return "Skipped plotting due to missing target values."

    n_bins = cfg.get("plot", {}).get("n_bins", 10)
    oneway_features = raw_feature_list or feature_list
    oneway_categorical = set(raw_categorical_features or categorical_features)

    for pred_key in model_keys:
        pred_label = _model_label(pred_key)
        pred_col = f"pred_{pred_key}"
        pred_tag = _safe_tag(pred_label or pred_col)
        output_root, plot_style = _get_plot_config(pred_key)
        for feature in oneway_features:
            if feature not in plot_train.columns:
                continue
            save_path = _resolve_plot_path(
                output_root,
                plot_style,
                f"{model_name}/oneway/post",
                f"00_{model_name}_{feature}_oneway_{pred_tag}.png",
            )
            plot_oneway(
                plot_train,
                feature=feature,
                weight_col=weight_col,
                target_col="w_act",
                pred_col=pred_col,
                pred_label=pred_label,
                n_bins=n_bins,
                is_categorical=feature in oneway_categorical,
                save_path=save_path,
                show=False,
            )

    datasets = []
    if "w_act" in plot_train.columns and not plot_train["w_act"].isna().all():
        datasets.append(("Train Data", plot_train))
    if "w_act" in plot_test.columns and not plot_test["w_act"].isna().all():
        datasets.append(("Test Data", plot_test))

    def _plot_lift_for_model(pred_key: str, pred_label: str) -> None:
        if not datasets:
            return
        output_root, plot_style = _get_plot_config(pred_key)
        style = PlotStyle()
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        if len(datasets) == 1:
            axes = [axes]
        for ax, (title, data) in zip(axes, datasets):
            pred_col = f"pred_{pred_key}"
            if pred_col not in data.columns:
                continue
            plot_lift_curve(
                data[pred_col].values,
                data["w_act"].values,
                data[weight_col].values,
                n_bins=n_bins,
                title=f"Lift Chart on {title}",
                pred_label="Predicted",
                act_label="Actual",
                weight_label="Earned Exposure",
                pred_weighted=False,
                actual_weighted=True,
                ax=ax,
                show=False,
                style=style,
            )
        plt.subplots_adjust(wspace=0.3)
        filename = f"01_{model_name}_{_safe_tag(pred_label)}_lift.png"
        save_path = _resolve_plot_path(
            output_root,
            plot_style,
            f"{model_name}/lift",
            filename,
        )
        finalize_figure(fig, save_path=save_path, show=False, style=style)

    for pred_key in model_keys:
        _plot_lift_for_model(pred_key, _model_label(pred_key))

    if (
        all(k in model_keys for k in ["xgb", "resn"])
        and all(f"pred_{k}" in plot_train.columns for k in ["xgb", "resn"])
        and datasets
    ):
        style = PlotStyle()
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        if len(datasets) == 1:
            axes = [axes]
        for ax, (title, data) in zip(axes, datasets):
            plot_double_lift_curve(
                data["pred_xgb"].values,
                data["pred_resn"].values,
                data["w_act"].values,
                data[weight_col].values,
                n_bins=n_bins,
                title=f"Double Lift Chart on {title}",
                label1="Xgboost",
                label2="ResNet",
                pred1_weighted=False,
                pred2_weighted=False,
                actual_weighted=True,
                ax=ax,
                show=False,
                style=style,
            )
        plt.subplots_adjust(wspace=0.3)
        save_path = _resolve_plot_path(
            _resolve_output_dir(cfg, cfg_path),
            _resolve_plot_style(cfg),
            "",
            f"02_{model_name}_dlift_xgb_vs_resn.png",
        )
        finalize_figure(fig, save_path=save_path, show=False, style=style)

    print("Plots saved under:")
    for key in model_keys:
        output_root, _ = _get_plot_config(key)
        print(f"  - {key}: {output_root}/plot/{model_name}")
    return "Plotting complete."


def run_predict_ft_embed(
    *,
    ft_cfg_path: str,
    xgb_cfg_path: Optional[str],
    resn_cfg_path: Optional[str],
    input_path: str,
    output_path: str,
    model_name: Optional[str],
    model_keys: str,
) -> str:
    ft_cfg_path = Path(ft_cfg_path).resolve()
    xgb_cfg_path = Path(xgb_cfg_path).resolve() if xgb_cfg_path else None
    resn_cfg_path = Path(resn_cfg_path).resolve() if resn_cfg_path else None
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    keys = [k.strip() for k in model_keys.split(",") if k.strip()]
    if not keys:
        raise ValueError("model_keys is empty.")

    ft_cfg = json.loads(ft_cfg_path.read_text(encoding="utf-8"))
    xgb_cfg = json.loads(xgb_cfg_path.read_text(encoding="utf-8")) if xgb_cfg_path else None
    resn_cfg = json.loads(resn_cfg_path.read_text(encoding="utf-8")) if resn_cfg_path else None

    if model_name is None:
        model_list = list(ft_cfg.get("model_list") or [])
        model_categories = list(ft_cfg.get("model_categories") or [])
        if len(model_list) != 1 or len(model_categories) != 1:
            raise ValueError("Set model_name when multiple models exist.")
        model_name = f"{model_list[0]}_{model_categories[0]}"

    ft_output_dir = (ft_cfg_path.parent / ft_cfg["output_dir"]).resolve()
    xgb_output_dir = (xgb_cfg_path.parent / xgb_cfg["output_dir"]).resolve() if xgb_cfg else None
    ft_prefix = ft_cfg.get("ft_feature_prefix", "ft_emb")
    xgb_task_type = str(xgb_cfg.get("task_type", "regression")) if xgb_cfg else None

    if ft_cfg.get("geo_feature_nmes"):
        raise ValueError("FT with geo tokens is not supported in this workflow.")

    import torch
    import joblib

    print("Loading FT model...")
    ft_model_path = ft_output_dir / "model" / f"01_{model_name}_FTTransformer.pth"
    ft_payload = torch.load(ft_model_path, map_location="cpu")
    ft_model = ft_payload["model"] if isinstance(ft_payload, dict) and "model" in ft_payload else ft_payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(ft_model, "device"):
        ft_model.device = device
    if hasattr(ft_model, "to"):
        ft_model.to(device)
    if hasattr(ft_model, "ft"):
        ft_model.ft.to(device)

    df_new = pd.read_csv(input_path)
    emb = ft_model.predict(df_new, return_embedding=True)
    emb_cols = [f"pred_{ft_prefix}_{i}" for i in range(emb.shape[1])]
    df_with_emb = df_new.copy()
    df_with_emb[emb_cols] = emb
    result = df_with_emb.copy()

    if "xgb" in keys:
        if not xgb_cfg or not xgb_output_dir:
            raise ValueError("xgb model selected but xgb_cfg_path is missing.")
        xgb_model_path = xgb_output_dir / "model" / f"01_{model_name}_Xgboost.pkl"
        xgb_payload = joblib.load(xgb_model_path)
        if isinstance(xgb_payload, dict) and "model" in xgb_payload:
            xgb_model = xgb_payload["model"]
            feature_list = xgb_payload.get("preprocess_artifacts", {}).get("factor_nmes")
        else:
            xgb_model = xgb_payload
            feature_list = None
        if not feature_list:
            feature_list = xgb_cfg.get("feature_list") or []
        if not feature_list:
            raise ValueError("Feature list missing for XGB model.")

        X = df_with_emb[feature_list]
        if xgb_task_type == "classification" and hasattr(xgb_model, "predict_proba"):
            pred = xgb_model.predict_proba(X)[:, 1]
        else:
            pred = xgb_model.predict(X)
        result["pred_xgb"] = pred

    if "resn" in keys:
        if not resn_cfg_path:
            raise ValueError("resn model selected but resn_cfg_path is missing.")
        resn_predictor = load_predictor_from_config(
            resn_cfg_path, "resn", model_name=model_name
        )
        pred_resn = resn_predictor.predict(df_with_emb)
        result["pred_resn"] = pred_resn

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    return str(output_path)


def run_compare_ft_embed(
    *,
    direct_cfg_path: str,
    ft_cfg_path: str,
    ft_embed_cfg_path: str,
    model_key: str,
    label_direct: str,
    label_ft: str,
    use_runtime_ft_embedding: bool = False,
    n_bins_override: Optional[int] = 10,
) -> str:
    direct_cfg_path = Path(direct_cfg_path).resolve()
    ft_cfg_path = Path(ft_cfg_path).resolve()
    ft_embed_cfg_path = Path(ft_embed_cfg_path).resolve()

    direct_cfg = json.loads(direct_cfg_path.read_text(encoding="utf-8"))
    ft_embed_cfg = json.loads(ft_embed_cfg_path.read_text(encoding="utf-8"))
    ft_cfg = json.loads(ft_cfg_path.read_text(encoding="utf-8"))

    model_name = f"{direct_cfg['model_list'][0]}_{direct_cfg['model_categories'][0]}"

    raw_path = _resolve_data_path(direct_cfg, direct_cfg_path, model_name)
    raw = pd.read_csv(raw_path)
    raw = _drop_duplicate_columns(raw, "raw").reset_index(drop=True)
    raw.fillna(0, inplace=True)

    holdout_ratio = direct_cfg.get("holdout_ratio", direct_cfg.get("prop_test", 0.25))
    split_strategy = direct_cfg.get("split_strategy", "random")
    split_group_col = direct_cfg.get("split_group_col")
    split_time_col = direct_cfg.get("split_time_col")
    split_time_ascending = direct_cfg.get("split_time_ascending", True)
    rand_seed = direct_cfg.get("rand_seed", 13)

    train_raw, test_raw = split_train_test(
        raw,
        holdout_ratio=holdout_ratio,
        strategy=split_strategy,
        group_col=split_group_col,
        time_col=split_time_col,
        time_ascending=split_time_ascending,
        rand_seed=rand_seed,
        reset_index_mode="none",
        ratio_label="holdout_ratio",
    )
    train_raw = _drop_duplicate_columns(train_raw, "train_raw")
    test_raw = _drop_duplicate_columns(test_raw, "test_raw")

    ft_output_dir = (ft_cfg_path.parent / ft_cfg["output_dir"]).resolve()
    ft_prefix = ft_cfg.get("ft_feature_prefix", "ft_emb")

    if use_runtime_ft_embedding:
        import torch

        ft_model_path = ft_output_dir / "model" / f"01_{model_name}_FTTransformer.pth"
        ft_payload = torch.load(ft_model_path, map_location="cpu")
        ft_model = ft_payload["model"] if isinstance(ft_payload, dict) and "model" in ft_payload else ft_payload

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(ft_model, "device"):
            ft_model.device = device
        if hasattr(ft_model, "to"):
            ft_model.to(device)
        if hasattr(ft_model, "ft"):
            ft_model.ft.to(device)

        emb_train = ft_model.predict(train_raw, return_embedding=True)
        emb_cols = [f"pred_{ft_prefix}_{i}" for i in range(emb_train.shape[1])]
        train_df = train_raw.copy()
        train_df[emb_cols] = emb_train

        emb_test = ft_model.predict(test_raw, return_embedding=True)
        test_df = test_raw.copy()
        test_df[emb_cols] = emb_test
    else:
        embed_path = _resolve_data_path(ft_embed_cfg, ft_embed_cfg_path, model_name)
        embed_df = pd.read_csv(embed_path)
        embed_df = _drop_duplicate_columns(embed_df, "embed").reset_index(drop=True)
        embed_df.fillna(0, inplace=True)
        if len(embed_df) != len(raw):
            raise ValueError(
                f"Row count mismatch: raw={len(raw)}, embed={len(embed_df)}. "
                "Cannot align predictions to raw features."
            )
        train_df = embed_df.loc[train_raw.index].copy()
        test_df = embed_df.loc[test_raw.index].copy()

    direct_predictor = load_predictor_from_config(
        direct_cfg_path, model_key, model_name=model_name
    )
    ft_predictor = load_predictor_from_config(
        ft_embed_cfg_path, model_key, model_name=model_name
    )

    pred_direct_train = direct_predictor.predict(train_raw).reshape(-1)
    pred_direct_test = direct_predictor.predict(test_raw).reshape(-1)
    pred_ft_train = ft_predictor.predict(train_df).reshape(-1)
    pred_ft_test = ft_predictor.predict(test_df).reshape(-1)

    if len(pred_direct_train) != len(train_raw):
        raise ValueError("Train prediction length mismatch for direct model.")
    if len(pred_direct_test) != len(test_raw):
        raise ValueError("Test prediction length mismatch for direct model.")
    if len(pred_ft_train) != len(train_df):
        raise ValueError("Train prediction length mismatch for FT-embed model.")
    if len(pred_ft_test) != len(test_df):
        raise ValueError("Test prediction length mismatch for FT-embed model.")

    plot_train = train_raw.copy()
    plot_test = test_raw.copy()
    plot_train["pred_direct"] = pred_direct_train
    plot_train["pred_ft"] = pred_ft_train
    plot_test["pred_direct"] = pred_direct_test
    plot_test["pred_ft"] = pred_ft_test

    weight_col = direct_cfg["weight"]
    target_col = direct_cfg["target"]
    if weight_col not in plot_train.columns:
        plot_train[weight_col] = 1.0
    if weight_col not in plot_test.columns:
        plot_test[weight_col] = 1.0
    if target_col in plot_train.columns:
        plot_train["w_act"] = plot_train[target_col] * plot_train[weight_col]
    if target_col in plot_test.columns:
        plot_test["w_act"] = plot_test[target_col] * plot_test[weight_col]

    if "w_act" not in plot_train.columns or plot_train["w_act"].isna().all():
        print("[Plot] Missing target values in train split; skip plots.")
        return "Skipped plotting due to missing target values."

    n_bins = n_bins_override or direct_cfg.get("plot", {}).get("n_bins", 10)
    datasets = []
    if not plot_train["w_act"].isna().all():
        datasets.append(("Train Data", plot_train))
    if not plot_test["w_act"].isna().all():
        datasets.append(("Test Data", plot_test))

    style = PlotStyle()
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (title, data) in zip(axes, datasets):
        plot_double_lift_curve(
            data["pred_direct"].values,
            data["pred_ft"].values,
            data["w_act"].values,
            data[weight_col].values,
            n_bins=n_bins,
            title=f"Double Lift Chart on {title}",
            label1=label_direct,
            label2=label_ft,
            pred1_weighted=False,
            pred2_weighted=False,
            actual_weighted=True,
            ax=ax,
            show=False,
            style=style,
        )
    plt.subplots_adjust(wspace=0.3)

    output_root = _resolve_output_dir(direct_cfg, direct_cfg_path)
    plot_style = _resolve_plot_style(direct_cfg)
    filename = (
        f"01_{model_name}_dlift_"
        f"{_safe_tag(label_direct)}_vs_{_safe_tag(label_ft)}.png"
    )
    save_path = _resolve_plot_path(
        output_root,
        plot_style,
        f"{model_name}/double_lift",
        filename,
    )
    finalize_figure(fig, save_path=save_path, show=False, style=style)
    print(f"Double lift saved to: {save_path}")
    return str(save_path)
