"""Plotting workflows used by the frontend UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    _build_search_roots,
    _dedupe_list,
    _discover_model_file,
    _drop_duplicate_columns,
    _infer_categorical_features,
    _parse_csv_list,
    _resolve_model_output_dir,
    _resolve_output_dir,
    _resolve_plot_path,
    _resolve_plot_style,
    _safe_tag,
)


def _normalize_oneway_feature_override(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return _dedupe_list(_parse_csv_list(raw_value))
    if isinstance(raw_value, (list, tuple, set)):
        return _dedupe_list([str(x).strip() for x in raw_value if str(x).strip()])
    text = str(raw_value).strip()
    return [text] if text else []


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
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
) -> str:
    model_name = str(model_name or "").strip()
    if not model_name:
        raise ValueError("model_name is required.")

    def _load_csv(path_value: Optional[str], label: str) -> Optional[tuple[pd.DataFrame, Path]]:
        raw_val = str(path_value or "").strip()
        if not raw_val:
            return None
        path_obj = Path(raw_val).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"{label} not found: {path_obj}")
        frame = pd.read_csv(path_obj, low_memory=False)
        frame = _drop_duplicate_columns(frame, label).reset_index(drop=True)
        frame.fillna(0, inplace=True)
        return frame, path_obj

    train_payload = _load_csv(train_data_path, "train_data")
    test_payload = _load_csv(test_data_path, "test_data")
    use_explicit_split = train_payload is not None or test_payload is not None

    raw: Optional[pd.DataFrame] = None
    raw_path: Optional[Path] = None
    if not use_explicit_split:
        data_path = str(data_path or "").strip()
        if not data_path:
            raise ValueError(
                "data_path is required when train_data_path/test_data_path are not provided."
            )
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

    reference_df = (
        train_payload[0]
        if train_payload is not None
        else test_payload[0]
        if test_payload is not None
        else raw
    )
    if reference_df is None:
        raise ValueError("No data available for plotting.")

    missing = [f for f in features if f not in reference_df.columns]
    if missing:
        print(f"[Warn] Missing features removed: {missing}")
        features = [f for f in features if f in reference_df.columns]
        cats = [f for f in cats if f in reference_df.columns]

    if not cats:
        infer_source = reference_df
        if train_payload is not None and test_payload is not None:
            infer_source = pd.concat([train_payload[0], test_payload[0]], ignore_index=True)
        cats = _infer_categorical_features(infer_source, features)

    default_out_base = (
        (raw_path.parent if raw_path is not None else Path.cwd())
        if train_payload is None and test_payload is None
        else (train_payload[1].parent if train_payload is not None else test_payload[1].parent)
    )
    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else default_out_base / "Results" / "plot" / model_name / "oneway" / "pre"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[str, pd.DataFrame]] = []
    if use_explicit_split:
        if train_payload is not None:
            datasets.append(("train", train_payload[0].copy()))
        if test_payload is not None:
            datasets.append(("test", test_payload[0].copy()))
        if not datasets:
            raise ValueError("At least one of train_data_path/test_data_path must be valid.")
    else:
        assert raw is not None
        if holdout_ratio is not None and float(holdout_ratio) > 0:
            train_df, _ = split_train_test(
                raw,
                holdout_ratio=float(holdout_ratio),
                strategy="random",
                rand_seed=int(rand_seed),
                reset_index_mode="none",
                ratio_label="holdout_ratio",
            )
            datasets.append(("train", train_df.reset_index(drop=True).copy()))
        else:
            datasets.append(("all", raw.copy()))

    print(f"Generating oneway plots for {len(features)} features...")
    saved = 0
    dataset_count = len(datasets)
    for tag, df in datasets:
        for i, feature in enumerate(features, 1):
            is_categorical = feature in cats
            try:
                suffix = "" if dataset_count == 1 else f"_{tag}"
                save_path = out_dir / f"{feature}{suffix}.png"
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
                    print(f"  [{tag}] [{i}/{len(features)}] {feature}")
            except Exception as exc:
                print(f"  [Warn] [{tag}] {feature} failed: {exc}")

    total_expected = len(features) * dataset_count
    print(f"Complete. Saved {saved}/{total_expected} plots to: {out_dir}")
    return f"Saved {saved} plots to {out_dir}"


def _load_split_frame(path_value: str, label: str) -> pd.DataFrame:
    path_obj = Path(path_value).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"{label} not found: {path_obj}")
    frame = pd.read_csv(path_obj, low_memory=False)
    frame = _drop_duplicate_columns(frame, label).reset_index(drop=True)
    frame.fillna(0, inplace=True)
    return frame


def _load_raw_splits(
    *,
    split_cfg: dict,
    data_cfg: dict,
    data_cfg_path: Path,
    model_name: str,
    train_data_path: Optional[str],
    test_data_path: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], bool]:
    explicit_train_path = str(train_data_path or "").strip()
    explicit_test_path = str(test_data_path or "").strip()
    use_explicit_split = bool(explicit_train_path or explicit_test_path)

    raw: Optional[pd.DataFrame] = None
    if use_explicit_split:
        train_raw = (
            _load_split_frame(explicit_train_path, "train_raw")
            if explicit_train_path
            else pd.DataFrame()
        )
        test_raw = (
            _load_split_frame(explicit_test_path, "test_raw")
            if explicit_test_path
            else pd.DataFrame()
        )
        if train_raw.empty and test_raw.empty:
            raise ValueError("At least one of train_data_path/test_data_path must be provided.")
    else:
        raw_data_dir = (data_cfg_path.parent / data_cfg["data_dir"]).resolve()
        raw_path = raw_data_dir / f"{model_name}.csv"
        raw = pd.read_csv(raw_path, low_memory=False)
        raw = _drop_duplicate_columns(raw, "raw").reset_index(drop=True)
        raw.fillna(0, inplace=True)

        holdout_ratio = split_cfg.get("holdout_ratio", split_cfg.get("prop_test", 0.25))
        split_strategy = split_cfg.get("split_strategy", "random")
        split_group_col = split_cfg.get("split_group_col")
        split_time_col = split_cfg.get("split_time_col")
        split_time_ascending = split_cfg.get("split_time_ascending", True)
        rand_seed = split_cfg.get("rand_seed", 13)

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

    return train_raw, test_raw, raw, use_explicit_split


def _run_prediction_plot_workflow(
    *,
    cfg: dict,
    cfg_path: Path,
    xgb_cfg: dict,
    xgb_cfg_path: Path,
    resn_cfg: dict,
    resn_cfg_path: Path,
    model_name: str,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    oneway_features: List[str],
    oneway_categorical: set[str],
    xgb_model_path: Optional[str],
    resn_model_path: Optional[str],
    model_search_dir: Optional[str] = None,
) -> str:
    output_dir_map = {
        "xgb": _resolve_output_dir(xgb_cfg, xgb_cfg_path),
        "resn": _resolve_output_dir(resn_cfg, resn_cfg_path),
    }
    output_dir_path_map = {
        key: Path(value).resolve()
        for key, value in output_dir_map.items()
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

    search_roots = _build_search_roots(
        model_search_dir,
        cfg_path.parent,
        xgb_cfg_path.parent,
        resn_cfg_path.parent,
        Path.cwd(),
    )
    raw_model_paths = {"xgb": xgb_model_path, "resn": resn_model_path}
    model_output_overrides: Dict[str, Optional[Path]] = {}
    for key in ("xgb", "resn"):
        override = _resolve_model_output_dir(raw_model_paths.get(key), f"{key}_model_path")
        if override is None:
            discovered = _discover_model_file(
                model_name=model_name,
                model_key=key,
                search_roots=search_roots,
                output_roots=[output_dir_path_map[key]],
            )
            if discovered is not None:
                override = _resolve_model_output_dir(
                    str(discovered), f"auto_{key}_model_path"
                )
                print(f"[Info] Auto-discovered {key} model: {discovered}")
        model_output_overrides[key] = override

    def _load_predictor(model_cfg_path: Path, model_key: str):
        output_override = model_output_overrides.get(model_key)
        kwargs: Dict[str, object] = {"model_name": model_name}
        if output_override is not None:
            kwargs["output_dir"] = output_override
        return load_predictor_from_config(model_cfg_path, model_key, **kwargs)

    model_cfg_map = {"xgb": xgb_cfg_path, "resn": resn_cfg_path}
    model_keys = cfg.get("model_keys") or ["xgb", "resn"]
    model_keys = [key for key in model_keys if key in model_cfg_map]
    if not model_keys:
        raise ValueError("No valid model keys found in plot config.")

    default_model_labels = {"xgb": "Xgboost", "resn": "ResNet"}
    labels = cfg.get("model_labels") or {}

    def _model_label(model_key: str) -> str:
        return str(labels.get(model_key, default_model_labels.get(model_key, model_key)))

    predictors = {key: _load_predictor(model_cfg_map[key], key) for key in model_keys}

    pred_train = {}
    pred_test = {}
    for key, predictor in predictors.items():
        if len(train_df) > 0:
            pred_train[key] = predictor.predict(train_df).reshape(-1)
            if len(pred_train[key]) != len(train_df):
                raise ValueError(f"Train prediction length mismatch for {key}")
        else:
            pred_train[key] = []
        if len(test_df) > 0:
            pred_test[key] = predictor.predict(test_df).reshape(-1)
            if len(pred_test[key]) != len(test_df):
                raise ValueError(f"Test prediction length mismatch for {key}")
        else:
            pred_test[key] = []

    plot_train = train_raw.copy()
    plot_test = test_raw.copy()
    for key in model_keys:
        if len(plot_train) > 0:
            plot_train[f"pred_{key}"] = pred_train[key]
        if len(plot_test) > 0:
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

    train_ready = "w_act" in plot_train.columns and not plot_train["w_act"].isna().all()
    test_ready = "w_act" in plot_test.columns and not plot_test["w_act"].isna().all()
    if not train_ready and not test_ready:
        print("[Plot] Missing target values in train split; skip plots.")
        return "Skipped plotting due to missing target values."

    n_bins = cfg.get("plot", {}).get("n_bins", 10)
    oneway_source = plot_train if len(plot_train) > 0 else plot_test
    for pred_key in model_keys:
        pred_label = _model_label(pred_key)
        pred_col = f"pred_{pred_key}"
        pred_tag = _safe_tag(pred_label or pred_col)
        output_root, plot_style = _get_plot_config(pred_key)
        for feature in oneway_features:
            if feature not in oneway_source.columns:
                continue
            save_path = _resolve_plot_path(
                output_root,
                plot_style,
                f"{model_name}/oneway/post",
                f"00_{model_name}_{feature}_oneway_{pred_tag}.png",
            )
            plot_oneway(
                oneway_source,
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
    if train_ready:
        datasets.append(("Train Data", plot_train))
    if test_ready:
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
        and all(
            any(f"pred_{k}" in frame.columns for frame in (plot_train, plot_test))
            for k in ["xgb", "resn"]
        )
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


def run_plot_direct(
    *,
    cfg_path: str,
    xgb_cfg_path: str,
    resn_cfg_path: str,
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    xgb_model_path: Optional[str] = None,
    resn_model_path: Optional[str] = None,
    model_search_dir: Optional[str] = None,
    oneway_features: Optional[Any] = None,
) -> str:
    cfg_path = Path(cfg_path).resolve()
    xgb_cfg_path = Path(xgb_cfg_path).resolve()
    resn_cfg_path = Path(resn_cfg_path).resolve()

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    xgb_cfg = json.loads(xgb_cfg_path.read_text(encoding="utf-8"))
    resn_cfg = json.loads(resn_cfg_path.read_text(encoding="utf-8"))

    model_name = f"{cfg['model_list'][0]}_{cfg['model_categories'][0]}"
    train_raw, test_raw, _raw, _ = _load_raw_splits(
        split_cfg=cfg,
        data_cfg=cfg,
        data_cfg_path=cfg_path,
        model_name=model_name,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
    )

    feature_list = _dedupe_list(cfg.get("feature_list") or [])
    selected_features = _normalize_oneway_feature_override(oneway_features)
    if selected_features:
        feature_set = set(feature_list)
        filtered_features = [f for f in selected_features if not feature_set or f in feature_set]
        ignored = [f for f in selected_features if feature_set and f not in feature_set]
        if ignored:
            print(f"[Warn] Ignored factors not present in config.feature_list: {ignored}")
        if not filtered_features:
            raise ValueError("Selected oneway factors are not present in config.feature_list.")
        feature_list = filtered_features
    categorical_features = _dedupe_list(cfg.get("categorical_features") or [])

    return _run_prediction_plot_workflow(
        cfg=cfg,
        cfg_path=cfg_path,
        xgb_cfg=xgb_cfg,
        xgb_cfg_path=xgb_cfg_path,
        resn_cfg=resn_cfg,
        resn_cfg_path=resn_cfg_path,
        model_name=model_name,
        train_raw=train_raw,
        test_raw=test_raw,
        train_df=train_raw.copy(),
        test_df=test_raw.copy(),
        oneway_features=feature_list,
        oneway_categorical=set(categorical_features),
        xgb_model_path=xgb_model_path,
        resn_model_path=resn_model_path,
        model_search_dir=model_search_dir,
    )


def run_plot_embed(
    *,
    cfg_path: str,
    xgb_cfg_path: str,
    resn_cfg_path: str,
    ft_cfg_path: str,
    use_runtime_ft_embedding: bool = False,
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    xgb_model_path: Optional[str] = None,
    resn_model_path: Optional[str] = None,
    ft_model_path: Optional[str] = None,
    model_search_dir: Optional[str] = None,
    oneway_features: Optional[Any] = None,
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
    search_roots = _build_search_roots(
        model_search_dir,
        cfg_path.parent,
        xgb_cfg_path.parent,
        resn_cfg_path.parent,
        ft_cfg_path.parent,
        Path.cwd(),
    )

    ft_output_dir = (ft_cfg_path.parent / ft_cfg["output_dir"]).resolve()
    ft_prefix = ft_cfg.get("ft_feature_prefix", "ft_emb")
    raw_feature_list = _dedupe_list(ft_cfg.get("feature_list") or [])
    raw_categorical_features = _dedupe_list(ft_cfg.get("categorical_features") or [])

    if ft_cfg.get("geo_feature_nmes"):
        raise ValueError("FT inference with geo tokens is not supported in this workflow.")
    train_raw, test_raw, raw, use_explicit_split = _load_raw_splits(
        split_cfg=cfg,
        data_cfg=ft_cfg,
        data_cfg_path=ft_cfg_path,
        model_name=model_name,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
    )

    if use_runtime_ft_embedding:
        import torch

        if ft_model_path and str(ft_model_path).strip():
            ft_model_path_obj = Path(str(ft_model_path).strip()).resolve()
        else:
            default_ft_model_path = (
                ft_output_dir / "model" / f"01_{model_name}_FTTransformer.pth"
            )
            discovered_ft_model = _discover_model_file(
                model_name=model_name,
                model_key="ft",
                search_roots=search_roots,
                output_roots=[ft_output_dir],
            )
            if discovered_ft_model is not None:
                ft_model_path_obj = discovered_ft_model
                print(f"[Info] Auto-discovered ft model: {discovered_ft_model}")
            else:
                ft_model_path_obj = default_ft_model_path
        if not ft_model_path_obj.exists():
            raise FileNotFoundError(f"FT model file not found: {ft_model_path_obj}")
        ft_payload = torch.load(ft_model_path_obj, map_location="cpu")
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
        if use_explicit_split:
            # Assume explicit split files are already compatible with downstream models.
            train_df = train_raw.copy()
            test_df = test_raw.copy()
        else:
            embed_data_dir = (cfg_path.parent / cfg["data_dir"]).resolve()
            embed_path = embed_data_dir / f"{model_name}.csv"
            embed_df = pd.read_csv(embed_path)
            embed_df = _drop_duplicate_columns(embed_df, "embed").reset_index(drop=True)
            embed_df.fillna(0, inplace=True)
            if raw is None:
                raise ValueError("Raw data unavailable for embedding alignment.")
            if len(embed_df) != len(raw):
                raise ValueError(
                    f"Row count mismatch: raw={len(raw)}, embed={len(embed_df)}. "
                    "Cannot align predictions to raw features."
                )
            train_df = embed_df.loc[train_raw.index].copy()
            test_df = embed_df.loc[test_raw.index].copy()

    feature_list = _dedupe_list(cfg.get("feature_list") or [])
    categorical_features = _dedupe_list(cfg.get("categorical_features") or [])
    selected_features = _normalize_oneway_feature_override(oneway_features)
    base_oneway_features = raw_feature_list or feature_list
    if selected_features:
        feature_set = set(base_oneway_features)
        filtered_features = [f for f in selected_features if not feature_set or f in feature_set]
        ignored = [f for f in selected_features if feature_set and f not in feature_set]
        if ignored:
            print(f"[Warn] Ignored factors not present in config.feature_list: {ignored}")
        if not filtered_features:
            raise ValueError("Selected oneway factors are not present in config.feature_list.")
        base_oneway_features = filtered_features

    return _run_prediction_plot_workflow(
        cfg=cfg,
        cfg_path=cfg_path,
        xgb_cfg=xgb_cfg,
        xgb_cfg_path=xgb_cfg_path,
        resn_cfg=resn_cfg,
        resn_cfg_path=resn_cfg_path,
        model_name=model_name,
        train_raw=train_raw,
        test_raw=test_raw,
        train_df=train_df,
        test_df=test_df,
        oneway_features=base_oneway_features,
        oneway_categorical=set(raw_categorical_features or categorical_features),
        xgb_model_path=xgb_model_path,
        resn_model_path=resn_model_path,
        model_search_dir=model_search_dir,
    )

