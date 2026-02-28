"""Model comparison workflows used by the frontend UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ins_pricing.cli.utils.cli_common import split_train_test
from ins_pricing.modelling.plotting import PlotStyle, plot_double_lift_curve
from ins_pricing.modelling.plotting.common import finalize_figure, plt
from ins_pricing.production.inference import load_predictor_from_config

from .workflows_common import (
    _drop_duplicate_columns,
    _resolve_data_path,
    _resolve_output_dir,
    _resolve_plot_path,
    _resolve_plot_style,
    _safe_tag,
)


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


def run_double_lift_from_file(
    *,
    data_path: str,
    pred_col_1: str,
    pred_col_2: str,
    target_col: str,
    weight_col: str = "weights",
    n_bins: int = 10,
    label1: Optional[str] = None,
    label2: Optional[str] = None,
    pred1_weighted: bool = False,
    pred2_weighted: bool = False,
    actual_weighted: bool = False,
    holdout_ratio: Optional[float] = 0.0,
    split_strategy: str = "random",
    split_group_col: Optional[str] = None,
    split_time_col: Optional[str] = None,
    split_time_ascending: bool = True,
    rand_seed: int = 13,
    output_path: Optional[str] = None,
) -> str:
    data_path_obj = Path(data_path).resolve()
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {data_path_obj}")

    pred_col_1 = str(pred_col_1 or "").strip()
    pred_col_2 = str(pred_col_2 or "").strip()
    target_col = str(target_col or "").strip()
    weight_col = str(weight_col or "").strip() or "weights"
    if not pred_col_1:
        raise ValueError("pred_col_1 is required.")
    if not pred_col_2:
        raise ValueError("pred_col_2 is required.")
    if not target_col:
        raise ValueError("target_col is required.")

    label1 = str(label1 or pred_col_1).strip() or pred_col_1
    label2 = str(label2 or pred_col_2).strip() or pred_col_2

    raw = pd.read_csv(data_path_obj, low_memory=False)
    raw = _drop_duplicate_columns(raw, "double_lift_raw").reset_index(drop=True)
    raw.fillna(0, inplace=True)

    if weight_col not in raw.columns:
        print(f"[Info] weight_col={weight_col!r} not found. Using constant 1.0.")
        raw[weight_col] = 1.0

    required_cols = [pred_col_1, pred_col_2, target_col, weight_col]
    missing_cols = [c for c in required_cols if c not in raw.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    before_rows = len(raw)
    work_df = raw.dropna(subset=required_cols).reset_index(drop=True)
    after_rows = len(work_df)
    if after_rows == 0:
        raise ValueError("No valid rows remain after dropping NA in required columns.")
    if after_rows < before_rows:
        print(f"[Info] Dropped {before_rows - after_rows} rows with NA in required columns.")

    split_group_col = str(split_group_col or "").strip() or None
    split_time_col = str(split_time_col or "").strip() or None
    split_strategy = str(split_strategy or "random").strip().lower() or "random"
    holdout_ratio_val = 0.0 if holdout_ratio is None else float(holdout_ratio)

    datasets = []
    if holdout_ratio_val > 0:
        train_df, test_df = split_train_test(
            work_df,
            holdout_ratio=holdout_ratio_val,
            strategy=split_strategy,
            group_col=split_group_col,
            time_col=split_time_col,
            time_ascending=bool(split_time_ascending),
            rand_seed=int(rand_seed),
            reset_index_mode="none",
            ratio_label="holdout_ratio",
        )
        train_df = _drop_duplicate_columns(train_df, "double_lift_train")
        test_df = _drop_duplicate_columns(test_df, "double_lift_test")
        if len(train_df) > 0:
            datasets.append(("Train Data", train_df))
        if len(test_df) > 0:
            datasets.append(("Test Data", test_df))
    else:
        datasets.append(("All Data", work_df))

    if not datasets:
        raise ValueError("No dataset is available for plotting.")

    style = PlotStyle()
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, (title, data) in zip(axes, datasets):
        plot_double_lift_curve(
            data[pred_col_1].values,
            data[pred_col_2].values,
            data[target_col].values,
            data[weight_col].values,
            n_bins=int(n_bins),
            title=f"Double Lift Chart on {title}",
            label1=label1,
            label2=label2,
            pred1_weighted=bool(pred1_weighted),
            pred2_weighted=bool(pred2_weighted),
            actual_weighted=bool(actual_weighted),
            ax=ax,
            show=False,
            style=style,
        )
    plt.subplots_adjust(wspace=0.3)

    if output_path:
        save_path = Path(output_path).resolve()
    else:
        filename = f"double_lift_{_safe_tag(label1)}_vs_{_safe_tag(label2)}.png"
        save_path = (data_path_obj.parent / "Results" / "plot" / filename).resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, save_path=str(save_path), show=False, style=style)
    print(f"Double lift saved to: {save_path}")
    return str(save_path)

