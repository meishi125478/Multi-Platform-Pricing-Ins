"""Plotting workflows used by the frontend UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

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

from .workflows_common import (
    _dedupe_list,
    _drop_duplicate_columns,
    _infer_categorical_features,
    _parse_csv_list,
    _resolve_output_dir,
    _resolve_plot_path,
    _resolve_plot_style,
    _safe_tag,
)


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

