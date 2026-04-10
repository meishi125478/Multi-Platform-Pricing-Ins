"""Model comparison workflows used by the frontend UI."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from ins_pricing.frontend.logging_utils import get_frontend_logger, log_print
from ins_pricing.split_cache import (
    load_split_cache,
    resolve_model_scoped_path,
    validate_split_cache_metadata,
    validate_split_indices,
    write_split_cache,
)

from ins_pricing.modelling.plotting import PlotStyle, plot_double_lift_curve
from ins_pricing.modelling.plotting.common import finalize_figure, plt

from .workflows_common import (
    _build_search_roots,
    _drop_duplicate_columns,
    _resolve_data_path,
    _resolve_double_lift_dir,
    _resolve_output_dir,
    _safe_tag,
)
from .workflows_prediction_utils import (
    DOUBLE_LIFT_FIGSIZE,
    build_ft_embedding_frames,
    load_raw_splits,
    resolve_model_output_override,
)

_logger = get_frontend_logger("ins_pricing.frontend.workflows_compare")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


def _load_predictor_from_cfg(*args, **kwargs):
    from ins_pricing.production.inference import load_predictor_from_config

    return load_predictor_from_config(*args, **kwargs)


def _split_train_test(*args, **kwargs):
    from ins_pricing.cli.utils.cli_common import split_train_test

    return split_train_test(*args, **kwargs)


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
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    direct_model_path: Optional[str] = None,
    ft_embed_model_path: Optional[str] = None,
    ft_model_path: Optional[str] = None,
    model_search_dir: Optional[str] = None,
) -> str:
    direct_cfg_path = Path(direct_cfg_path).resolve()
    ft_cfg_path = Path(ft_cfg_path).resolve()
    ft_embed_cfg_path = Path(ft_embed_cfg_path).resolve()

    direct_cfg = json.loads(direct_cfg_path.read_text(encoding="utf-8"))
    ft_embed_cfg = json.loads(ft_embed_cfg_path.read_text(encoding="utf-8"))
    ft_cfg = json.loads(ft_cfg_path.read_text(encoding="utf-8"))

    model_name = f"{direct_cfg['model_list'][0]}_{direct_cfg['model_categories'][0]}"
    search_roots = _build_search_roots(
        model_search_dir,
        direct_cfg_path.parent,
        ft_cfg_path.parent,
        ft_embed_cfg_path.parent,
        Path.cwd(),
    )

    train_raw, test_raw, raw, use_explicit_split = load_raw_splits(
        split_cfg=direct_cfg,
        data_cfg=direct_cfg,
        data_cfg_path=direct_cfg_path,
        model_name=model_name,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        data_path_resolver=_resolve_data_path,
    )

    train_df, test_df = build_ft_embedding_frames(
        use_runtime_ft_embedding=use_runtime_ft_embedding,
        train_raw=train_raw,
        test_raw=test_raw,
        raw=raw,
        use_explicit_split=use_explicit_split,
        model_name=model_name,
        ft_cfg=ft_cfg,
        ft_cfg_path=ft_cfg_path,
        search_roots=search_roots,
        ft_model_path=ft_model_path,
        embed_cfg=ft_embed_cfg,
        embed_cfg_path=ft_embed_cfg_path,
        embed_path_resolver=_resolve_data_path,
    )

    direct_output_root = Path(_resolve_output_dir(direct_cfg, direct_cfg_path)).resolve()
    ft_embed_output_root = Path(_resolve_output_dir(ft_embed_cfg, ft_embed_cfg_path)).resolve()
    direct_output_override = resolve_model_output_override(
        model_name=model_name,
        model_key=model_key,
        model_path=direct_model_path,
        search_roots=search_roots,
        output_root=direct_output_root,
        label="direct_model_path",
    )
    ft_output_override = resolve_model_output_override(
        model_name=model_name,
        model_key=model_key,
        model_path=ft_embed_model_path,
        search_roots=search_roots,
        output_root=ft_embed_output_root,
        label="ft_embed_model_path",
    )

    direct_predictor = _load_predictor_from_cfg(
        direct_cfg_path,
        model_key,
        model_name=model_name,
        output_dir=direct_output_override,
    )
    ft_predictor = _load_predictor_from_cfg(
        ft_embed_cfg_path,
        model_key,
        model_name=model_name,
        output_dir=ft_output_override,
    )

    if len(train_raw) > 0:
        pred_direct_train = direct_predictor.predict(train_raw).reshape(-1)
        pred_ft_train = ft_predictor.predict(train_df).reshape(-1)
        if len(pred_direct_train) != len(train_raw):
            raise ValueError("Train prediction length mismatch for direct model.")
        if len(pred_ft_train) != len(train_df):
            raise ValueError("Train prediction length mismatch for FT-embed model.")
    else:
        pred_direct_train = []
        pred_ft_train = []
    if len(test_raw) > 0:
        pred_direct_test = direct_predictor.predict(test_raw).reshape(-1)
        pred_ft_test = ft_predictor.predict(test_df).reshape(-1)
        if len(pred_direct_test) != len(test_raw):
            raise ValueError("Test prediction length mismatch for direct model.")
        if len(pred_ft_test) != len(test_df):
            raise ValueError("Test prediction length mismatch for FT-embed model.")
    else:
        pred_direct_test = []
        pred_ft_test = []

    plot_train = train_raw.copy()
    plot_test = test_raw.copy()
    if len(plot_train) > 0:
        plot_train["pred_direct"] = pred_direct_train
        plot_train["pred_ft"] = pred_ft_train
    if len(plot_test) > 0:
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

    train_ready = "w_act" in plot_train.columns and not plot_train["w_act"].isna().all()
    test_ready = "w_act" in plot_test.columns and not plot_test["w_act"].isna().all()
    if not train_ready and not test_ready:
        _log("[Plot] Missing target values in train split; skip plots.")
        return "Skipped plotting due to missing target values."

    n_bins = n_bins_override or direct_cfg.get("plot", {}).get("n_bins", 10)
    datasets = []
    if train_ready:
        datasets.append(("Train Data", plot_train))
    if test_ready:
        datasets.append(("Validation Data", plot_test))

    style = PlotStyle()
    fig, axes = plt.subplots(1, len(datasets), figsize=DOUBLE_LIFT_FIGSIZE)
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

    save_root = _resolve_double_lift_dir(model_name=model_name)
    filename = (
        f"double_lift_compare_{_safe_tag(model_key)}_{model_name}_"
        f"{_safe_tag(label_direct)}_vs_{_safe_tag(label_ft)}.png"
    )
    save_path = (save_root / filename).resolve()
    finalize_figure(fig, save_path=str(save_path), show=False, style=style)
    _log(f"Double lift saved to: {save_path}")
    return str(save_path)


def run_double_lift_from_file(
    *,
    data_path: str,
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
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
    split_cache_path: Optional[str] = None,
    split_cache_force_rebuild: bool = False,
    output_path: Optional[str] = None,
) -> str:
    explicit_train_path = str(train_data_path or "").strip()
    explicit_test_path = str(test_data_path or "").strip()
    use_explicit_split = bool(explicit_train_path or explicit_test_path)

    data_path_obj: Optional[Path] = None
    if not use_explicit_split:
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

    required_cols = [pred_col_1, pred_col_2, target_col, weight_col]

    def _load_and_validate(path_obj: Path, label: str) -> pd.DataFrame:
        frame = pd.read_csv(path_obj, low_memory=False)
        frame = _drop_duplicate_columns(frame, label).reset_index(drop=True)
        if weight_col not in frame.columns:
            _log(f"[Info] weight_col={weight_col!r} not found in {label}. Using constant 1.0.")
            frame[weight_col] = 1.0
        missing_cols = [c for c in required_cols if c not in frame.columns]
        if missing_cols:
            raise KeyError(f"{label} missing required columns: {missing_cols}")
        target_na_before = int(frame[target_col].isna().sum())
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0)
        if target_na_before > 0:
            _log(f"[Data] {label}: filled {target_na_before} NA in {target_col} with 0.")
        frame[weight_col] = pd.to_numeric(frame[weight_col], errors="coerce").fillna(1.0)
        for pred_col in (pred_col_1, pred_col_2):
            pred_numeric = pd.to_numeric(frame[pred_col], errors="coerce")
            pred_na = int(pred_numeric.isna().sum())
            if pred_na > 0:
                raise ValueError(
                    f"{label}: {pred_col} contains {pred_na} non-numeric/NA rows; "
                    "please clean prediction columns before plotting."
                )
            frame[pred_col] = pred_numeric
        if frame.empty:
            raise ValueError(f"{label} is empty.")
        return frame

    split_group_col = str(split_group_col or "").strip() or None
    split_time_col = str(split_time_col or "").strip() or None
    split_strategy = str(split_strategy or "random").strip().lower() or "random"
    holdout_ratio_val = 0.0 if holdout_ratio is None else float(holdout_ratio)

    datasets = []
    if use_explicit_split:
        if explicit_train_path:
            train_obj = Path(explicit_train_path).resolve()
            if not train_obj.exists():
                raise FileNotFoundError(f"train_data_path not found: {train_obj}")
            datasets.append(("Train Data", _load_and_validate(train_obj, "double_lift_train")))
        if explicit_test_path:
            test_obj = Path(explicit_test_path).resolve()
            if not test_obj.exists():
                raise FileNotFoundError(f"test_data_path not found: {test_obj}")
            datasets.append(("Validation Data", _load_and_validate(test_obj, "double_lift_test")))
    else:
        assert data_path_obj is not None
        raw = _load_and_validate(data_path_obj, "double_lift_raw")
        if holdout_ratio_val > 0:
            split_cache_obj = resolve_model_scoped_path(
                split_cache_path,
                model_name=_safe_tag(data_path_obj.stem),
                base_dir=data_path_obj.parent,
            )
            strategy_norm = split_strategy.strip().lower() or "random"

            def _split_and_optionally_write_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
                train_local, test_local = _split_train_test(
                    raw,
                    holdout_ratio=holdout_ratio_val,
                    strategy=split_strategy,
                    group_col=split_group_col,
                    time_col=split_time_col,
                    time_ascending=bool(split_time_ascending),
                    rand_seed=int(rand_seed),
                    reset_index_mode="none",
                    ratio_label="holdout_ratio",
                )
                if split_cache_obj is not None:
                    train_idx_local = np.asarray(
                        train_local.index.to_numpy(), dtype=np.int64
                    ).reshape(-1)
                    test_idx_local = np.asarray(
                        test_local.index.to_numpy(), dtype=np.int64
                    ).reshape(-1)
                    validate_split_indices(
                        train_idx=train_idx_local,
                        test_idx=test_idx_local,
                        row_count=len(raw),
                        cache_path=split_cache_obj,
                    )
                    write_split_cache(
                        split_cache_obj,
                        train_idx=train_idx_local,
                        test_idx=test_idx_local,
                        row_count=len(raw),
                        meta={
                            "split_strategy": strategy_norm,
                            "holdout_ratio": holdout_ratio_val,
                            "rand_seed": rand_seed,
                            "data_path": str(data_path_obj),
                        },
                    )
                    _log(f"[Split] Created split cache: {split_cache_obj}")
                return train_local, test_local

            if (
                split_cache_obj is not None
                and split_cache_obj.exists()
                and not bool(split_cache_force_rebuild)
            ):
                train_idx, test_idx, cached_row_count, cache_meta = load_split_cache(
                    split_cache_obj
                )
                validate_split_cache_metadata(
                    cache_path=split_cache_obj,
                    cached_row_count=cached_row_count,
                    current_row_count=len(raw),
                    cache_meta=cache_meta,
                    split_strategy=strategy_norm,
                    holdout_ratio=holdout_ratio_val,
                    rand_seed=rand_seed,
                    rebuild_hint=True,
                )
                validate_split_indices(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    row_count=len(raw),
                    cache_path=split_cache_obj,
                )
                train_df = raw.iloc[train_idx].copy()
                test_df = raw.iloc[test_idx].copy()
                _log(f"[Split] Reused split cache: {split_cache_obj}")
            else:
                train_df, test_df = _split_and_optionally_write_cache()

            train_df = _drop_duplicate_columns(train_df, "double_lift_train")
            test_df = _drop_duplicate_columns(test_df, "double_lift_test")
            if len(train_df) > 0:
                datasets.append(("Train Data", train_df))
            if len(test_df) > 0:
                datasets.append(("Validation Data", test_df))
        else:
            datasets.append(("All Data", raw))

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
        if use_explicit_split:
            if explicit_train_path and explicit_test_path:
                source_tag = (
                    f"{_safe_tag(Path(explicit_train_path).stem)}_and_"
                    f"{_safe_tag(Path(explicit_test_path).stem)}"
                )
            elif explicit_train_path:
                source_tag = _safe_tag(Path(explicit_train_path).stem)
            elif explicit_test_path:
                source_tag = _safe_tag(Path(explicit_test_path).stem)
            else:
                source_tag = "explicit_split"
            split_tag = "explicit_split"
            model_dir_tag = source_tag
        else:
            source_tag = _safe_tag(data_path_obj.stem) if data_path_obj is not None else "raw"
            split_tag = "train_valid" if holdout_ratio_val > 0 else "all"
            model_dir_tag = source_tag
        filename = (
            f"double_lift_file_{source_tag}_"
            f"{_safe_tag(label1)}_vs_{_safe_tag(label2)}_{split_tag}.png"
        )
        save_path = (_resolve_double_lift_dir(model_name=model_dir_tag) / filename).resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig, save_path=str(save_path), show=False, style=style)
    _log(f"Double lift saved to: {save_path}")
    return str(save_path)


def run_double_lift_multi_models(
    *,
    data_path: str,
    model_specs: List[Dict[str, Any]],
    target_col: str,
    weight_col: str = "weights",
    n_bins: int = 10,
    holdout_ratio: Optional[float] = 0.25,
    split_strategy: str = "random",
    split_group_col: Optional[str] = None,
    split_time_col: Optional[str] = None,
    split_time_ascending: bool = True,
    rand_seed: int = 13,
    split_scope: str = "both",
    output_dir: Optional[str] = None,
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    actual_weighted: bool = False,
) -> str:
    """Compare all model pairs with double-lift plots on train/validation splits.

    Each model spec supports either:
    1) ``loader='artifact_file'`` with ``path`` to a pickled model artifact.
    2) ``loader='config_predictor'`` with ``config_path`` + ``model_key``.
    """

    target_col = str(target_col or "").strip()
    weight_col = str(weight_col or "").strip() or "weights"
    if not target_col:
        raise ValueError("target_col is required.")
    if len(model_specs) < 2:
        raise ValueError("At least two model_specs are required.")

    def _resolve_path(path_like: Any) -> Path:
        return Path(str(path_like)).expanduser().resolve()

    def _load_and_validate(path_obj: Path, label: str) -> pd.DataFrame:
        if not path_obj.exists():
            raise FileNotFoundError(f"{label} not found: {path_obj}")
        frame = pd.read_csv(path_obj, low_memory=False)
        frame = _drop_duplicate_columns(frame, label).reset_index(drop=True)
        if weight_col not in frame.columns:
            _log(f"[Info] weight_col={weight_col!r} not found in {label}. Using constant 1.0.")
            frame[weight_col] = 1.0
        required_cols = [target_col, weight_col]
        missing_cols = [c for c in required_cols if c not in frame.columns]
        if missing_cols:
            raise KeyError(f"{label} missing required columns: {missing_cols}")
        target_na_before = int(frame[target_col].isna().sum())
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0)
        if target_na_before > 0:
            _log(f"[Data] {label}: filled {target_na_before} NA in {target_col} with 0.")
        frame[weight_col] = pd.to_numeric(frame[weight_col], errors="coerce").fillna(1.0)
        if frame.empty:
            raise ValueError(f"{label} is empty.")
        return frame

    def _build_predict_fn(spec: Dict[str, Any]) -> Callable[[pd.DataFrame], np.ndarray]:
        loader = str(spec.get("loader", "artifact_file")).strip().lower()
        name = str(spec.get("name", "")).strip() or "<unnamed>"

        if loader == "artifact_file":
            model_path = _resolve_path(spec.get("path"))
            if not model_path.exists():
                raise FileNotFoundError(f"{name}: model file not found: {model_path}")

            payload = joblib.load(model_path)
            if isinstance(payload, dict) and "model" in payload:
                model = payload.get("model")
                preprocess_artifacts = payload.get("preprocess_artifacts") or {}
                payload_features = list(preprocess_artifacts.get("factor_nmes") or [])
            else:
                model = payload
                payload_features = []

            feature_list = list(spec.get("feature_list") or [])
            if bool(spec.get("feature_list_from_payload", True)) and payload_features:
                feature_list = payload_features

            task_type = str(spec.get("task_type", "regression")).strip().lower()
            use_predict_proba = bool(spec.get("use_predict_proba", False))

            def _predict(df: pd.DataFrame) -> np.ndarray:
                X = df
                if feature_list:
                    missing = [c for c in feature_list if c not in df.columns]
                    if missing:
                        raise KeyError(f"{name}: missing required features: {missing[:10]}")
                    X = df[feature_list]

                if task_type in {"binary", "classification"} and use_predict_proba and hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                return np.asarray(pred).reshape(-1)

            return _predict

        if loader == "config_predictor":
            config_path = _resolve_path(spec.get("config_path"))
            model_key = str(spec.get("model_key", "")).strip().lower()
            if not model_key:
                raise ValueError(f"{name}: model_key is required for config_predictor loader.")
            predictor = _load_predictor_from_cfg(
                config_path=config_path,
                model_key=model_key,
                model_name=spec.get("model_name"),
            )

            def _predict(df: pd.DataFrame) -> np.ndarray:
                pred = predictor.predict(df)
                return np.asarray(pred).reshape(-1)

            return _predict

        raise ValueError(f"{name}: unsupported loader={loader!r}.")

    explicit_train_path = str(train_data_path or "").strip()
    explicit_test_path = str(test_data_path or "").strip()
    use_explicit_split = bool(explicit_train_path or explicit_test_path)

    datasets: List[tuple[str, str, pd.DataFrame]] = []
    data_path_obj: Optional[Path] = None
    if use_explicit_split:
        if explicit_train_path:
            train_obj = _resolve_path(explicit_train_path)
            datasets.append(("train", "Train Data", _load_and_validate(train_obj, "multi_compare_train")))
        if explicit_test_path:
            test_obj = _resolve_path(explicit_test_path)
            datasets.append(("valid", "Validation Data", _load_and_validate(test_obj, "multi_compare_valid")))
    else:
        data_path_obj = _resolve_path(data_path)
        raw = _load_and_validate(data_path_obj, "multi_compare_raw")
        holdout_ratio_val = 0.0 if holdout_ratio is None else float(holdout_ratio)
        if holdout_ratio_val > 0:
            train_df, test_df = _split_train_test(
                raw,
                holdout_ratio=holdout_ratio_val,
                strategy=str(split_strategy or "random").strip().lower() or "random",
                group_col=str(split_group_col or "").strip() or None,
                time_col=str(split_time_col or "").strip() or None,
                time_ascending=bool(split_time_ascending),
                rand_seed=int(rand_seed),
                reset_index_mode="none",
                ratio_label="holdout_ratio",
            )
            train_df = _drop_duplicate_columns(train_df, "multi_compare_train")
            test_df = _drop_duplicate_columns(test_df, "multi_compare_valid")
            if len(train_df) > 0:
                datasets.append(("train", "Train Data", train_df.reset_index(drop=True)))
            if len(test_df) > 0:
                datasets.append(("valid", "Validation Data", test_df.reset_index(drop=True)))
        else:
            datasets.append(("all", "All Data", raw.reset_index(drop=True)))

    if not datasets:
        raise ValueError("No dataset is available for multi-model comparison.")

    raw_split_scope = str(split_scope or "both").strip().lower()
    split_scope_aliases = {
        "both": "both",
        "all": "both",
        "train": "train",
        "model": "train",
        "training": "train",
        "valid": "valid",
        "val": "valid",
        "validation": "valid",
        "test": "valid",
    }
    normalized_split_scope = split_scope_aliases.get(raw_split_scope)
    if normalized_split_scope is None:
        raise ValueError(
            f"Unsupported split_scope={raw_split_scope!r}. "
            "Use one of: train, valid, both."
        )
    if normalized_split_scope == "train":
        datasets = [item for item in datasets if item[0] == "train"]
    elif normalized_split_scope == "valid":
        datasets = [item for item in datasets if item[0] == "valid"]
    if not datasets:
        raise ValueError(f"No datasets available after split_scope filtering: {normalized_split_scope}.")
    _log(
        f"[MultiCompare] split_scope={normalized_split_scope}, "
        f"datasets={[title for _tag, title, _df in datasets]}"
    )

    predict_fns: Dict[str, Callable[[pd.DataFrame], np.ndarray]] = {}
    pred_weighted_map: Dict[str, bool] = {}
    for spec in model_specs:
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError("Each model spec must include a non-empty 'name'.")
        if name in predict_fns:
            raise ValueError(f"Duplicate model spec name: {name}")
        predict_fns[name] = _build_predict_fn(spec)
        pred_weighted_map[name] = bool(spec.get("pred_weighted", False))

    model_names = list(predict_fns.keys())
    pairs = list(combinations(model_names, 2))
    if not pairs:
        raise ValueError("No model pairs available for comparison.")

    if output_dir:
        save_root = _resolve_path(output_dir)
    else:
        default_model_tag = (
            _safe_tag(data_path_obj.stem)
            if data_path_obj is not None
            else "multi_model_compare"
        )
        save_root = _resolve_double_lift_dir(model_name=default_model_tag)
    save_root.mkdir(parents=True, exist_ok=True)

    n_pairs = len(pairs)
    n_cols = 2 if n_pairs > 1 else 1
    n_rows = math.ceil(n_pairs / n_cols)
    style = PlotStyle()

    for split_tag, split_title, df in datasets:
        pred_map: Dict[str, np.ndarray] = {}
        for model_name in model_names:
            pred_vals = np.asarray(predict_fns[model_name](df), dtype=float).reshape(-1)
            if len(pred_vals) != len(df):
                raise ValueError(
                    f"{model_name}: prediction length mismatch on {split_tag} "
                    f"({len(pred_vals)} != {len(df)})"
                )
            pred_map[model_name] = pred_vals

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False)
        axes_flat = axes.flatten()
        actual = df[target_col].to_numpy()
        weight = df[weight_col].to_numpy()

        for i, (name_1, name_2) in enumerate(pairs):
            ax = axes_flat[i]
            plot_double_lift_curve(
                pred_map[name_1],
                pred_map[name_2],
                actual,
                weight,
                n_bins=int(n_bins),
                title=f"{split_title}: {name_1} vs {name_2}",
                label1=name_1,
                label2=name_2,
                pred1_weighted=pred_weighted_map.get(name_1, False),
                pred2_weighted=pred_weighted_map.get(name_2, False),
                actual_weighted=bool(actual_weighted),
                ax=ax,
                show=False,
                style=style,
            )

        for j in range(n_pairs, len(axes_flat)):
            axes_flat[j].axis("off")

        model_tag = "_vs_".join(_safe_tag(name) for name in model_names)
        save_path = save_root / f"double_lift_multi_{model_tag}_{split_tag}.png"
        finalize_figure(fig, save_path=str(save_path), show=False, style=style)
        _log(f"Saved: {save_path}")

    return str(save_root)

