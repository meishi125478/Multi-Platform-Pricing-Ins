"""Plotting workflows used by the frontend UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ins_pricing.frontend.logging_utils import get_frontend_logger, log_print

from ins_pricing.modelling.plotting import (
    PlotStyle,
    plot_double_lift_curve,
    plot_lift_curve,
    plot_oneway,
)
from ins_pricing.modelling.plotting.common import finalize_figure, plt

from .workflows_common import (
    _build_search_roots,
    _dedupe_list,
    _drop_duplicate_columns,
    _infer_categorical_features,
    _parse_csv_list,
    _resolve_data_path,
    _resolve_double_lift_dir,
    _resolve_output_dir,
    _resolve_plot_path,
    _resolve_plot_style,
    _safe_tag,
)
from .workflows_prediction_utils import (
    build_ft_embedding_frames,
    load_raw_splits,
    resolve_model_output_override,
)

PLOT_GRID_FIGSIZE = (11, 5)
_logger = get_frontend_logger("ins_pricing.frontend.workflows_plot")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


def _load_predictor_from_cfg(*args, **kwargs):
    from ins_pricing.production.inference import load_predictor_from_config

    return load_predictor_from_config(*args, **kwargs)


def _split_train_test(*args, **kwargs):
    from ins_pricing.cli.utils.cli_common import split_train_test

    return split_train_test(*args, **kwargs)


def _normalize_oneway_feature_override(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return _dedupe_list(_parse_csv_list(raw_value))
    if isinstance(raw_value, (list, tuple, set)):
        return _dedupe_list([str(x).strip() for x in raw_value if str(x).strip()])
    text = str(raw_value).strip()
    return [text] if text else []


def _resolve_predict_chunk_rows(cfg: dict, plot_cfg: dict) -> int:
    raw = plot_cfg.get(
        "predict_chunk_rows",
        plot_cfg.get(
            "pred_cache_predict_chunk_rows",
            cfg.get("pred_cache_predict_chunk_rows", 0),
        ),
    )
    try:
        value = int(raw or 0)
    except Exception:
        value = 0
    return max(0, value)


def _predict_vector(
    predictor,
    data: pd.DataFrame,
    *,
    model_key: str,
    split_tag: str,
    chunk_rows: int,
) -> np.ndarray:
    if len(data) == 0:
        return np.asarray([], dtype=float)
    if chunk_rows <= 0 or len(data) <= chunk_rows:
        return np.asarray(predictor.predict(data)).reshape(-1)

    pieces: List[np.ndarray] = []
    for start in range(0, len(data), chunk_rows):
        end = min(start + chunk_rows, len(data))
        chunk = data.iloc[start:end]
        pred_chunk = np.asarray(predictor.predict(chunk)).reshape(-1)
        if pred_chunk.shape[0] != (end - start):
            raise ValueError(
                f"{split_tag.capitalize()} prediction length mismatch for {model_key}: "
                f"expected={end - start}, got={pred_chunk.shape[0]}"
            )
        pieces.append(pred_chunk)
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces, axis=0)


def _required_feature_names_from_predictor(predictor) -> List[str]:
    artifacts = getattr(predictor, "artifacts", None)
    if isinstance(artifacts, dict):
        factor_nmes = list(artifacts.get("factor_nmes") or [])
        if factor_nmes:
            return factor_nmes
    predictor_cfg = getattr(predictor, "cfg", None)
    if isinstance(predictor_cfg, dict):
        return list(predictor_cfg.get("feature_list") or [])
    return []


def _log_missing_feature_coverage(
    *,
    model_key: str,
    split_tag: str,
    frame: pd.DataFrame,
    required_features: List[str],
) -> None:
    if not required_features:
        return
    missing = [f for f in required_features if f not in frame.columns]
    if not missing:
        return
    ratio = float(len(missing)) / float(max(len(required_features), 1))
    sample = missing[:8]
    _log(
        f"[Warn] {model_key} {split_tag}: missing {len(missing)}/{len(required_features)} "
        f"required features. sample={sample}"
    )
    if ratio >= 0.2 and any(str(col).startswith("pred_") for col in missing):
        _log(
            "[Warn] Missing columns include embedding-like features (pred_*). "
            "You may be using a non-direct model config in plot_direct."
        )


def _warn_if_near_constant_predictions(
    *,
    values: np.ndarray,
    model_key: str,
    split_tag: str,
) -> None:
    if values.size < 2:
        return
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        _log(f"[Warn] {model_key} {split_tag}: predictions are non-finite or empty after filtering.")
        return
    std = float(np.std(finite))
    q05, q95 = np.quantile(finite, [0.05, 0.95])
    spread = float(q95 - q05)
    scale = max(abs(float(q95)), abs(float(q05)), 1.0)
    if std <= 1e-8 or spread <= 1e-4 * scale:
        _log(
            f"[Warn] {model_key} {split_tag}: predictions are near-constant "
            f"(std={std:.6g}, q05={q05:.6g}, q95={q95:.6g})."
        )


def _log_numeric_summary(
    *,
    values: np.ndarray,
    model_key: str,
    split_tag: str,
    metric: str,
) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    n_total = int(arr.size)
    if n_total == 0:
        _log(f"[Diag] {model_key} {split_tag} {metric}: empty")
        return
    finite = arr[np.isfinite(arr)]
    n_finite = int(finite.size)
    n_nonfinite = n_total - n_finite
    if n_finite == 0:
        _log(
            f"[Diag] {model_key} {split_tag} {metric}: n={n_total}, finite=0, "
            f"non_finite={n_nonfinite}"
        )
        return
    q05, q50, q95 = np.quantile(finite, [0.05, 0.5, 0.95])
    _log(
        f"[Diag] {model_key} {split_tag} {metric}: n={n_total}, finite={n_finite}, "
        f"non_finite={n_nonfinite}, mean={float(np.mean(finite)):.6g}, "
        f"std={float(np.std(finite)):.6g}, min={float(np.min(finite)):.6g}, "
        f"q05={float(q05):.6g}, q50={float(q50):.6g}, q95={float(q95):.6g}, "
        f"max={float(np.max(finite)):.6g}"
    )


def _log_prediction_split_diagnostics(
    *,
    model_key: str,
    split_tag: str,
    pred_values: np.ndarray,
    raw_frame: pd.DataFrame,
    target_col: str,
    weight_col: str,
) -> None:
    pred_arr = np.asarray(pred_values, dtype=float).reshape(-1)
    _log_numeric_summary(
        values=pred_arr,
        model_key=model_key,
        split_tag=split_tag,
        metric="pred",
    )

    if target_col in raw_frame.columns:
        target_arr = pd.to_numeric(raw_frame[target_col], errors="coerce").to_numpy(dtype=float)
        _log_numeric_summary(
            values=target_arr,
            model_key=model_key,
            split_tag=split_tag,
            metric=target_col,
        )
        aligned_n = min(len(pred_arr), len(target_arr))
        if aligned_n > 1:
            pred_aligned = pred_arr[:aligned_n]
            target_aligned = target_arr[:aligned_n]
            finite_mask = np.isfinite(pred_aligned) & np.isfinite(target_aligned)
            if int(np.sum(finite_mask)) > 1:
                x = pred_aligned[finite_mask]
                y = target_aligned[finite_mask]
                x_std = float(np.std(x))
                y_std = float(np.std(y))
                corr = float("nan")
                if x_std > 0 and y_std > 0:
                    corr = float(np.corrcoef(x, y)[0, 1])
                _log(
                    f"[Diag] {model_key} {split_tag}: corr(pred,{target_col})={corr:.6g}, "
                    f"n_corr={int(np.sum(finite_mask))}"
                )

    if weight_col in raw_frame.columns:
        weight_arr = pd.to_numeric(raw_frame[weight_col], errors="coerce").to_numpy(dtype=float)
        _log_numeric_summary(
            values=weight_arr,
            model_key=model_key,
            split_tag=split_tag,
            metric=weight_col,
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
        _log(f"[Warn] Missing features removed: {missing}")
        features = [f for f in features if f in reference_df.columns]
        cats = [f for f in cats if f in reference_df.columns]

    if not cats:
        infer_source = reference_df
        if train_payload is not None and test_payload is not None:
            infer_source = pd.concat([train_payload[0], test_payload[0]], ignore_index=True)
        cats = _infer_categorical_features(infer_source, features)

    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else Path.cwd() / "Results" / "plot" / model_name / "oneway" / "pre"
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
            train_df, _ = _split_train_test(
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

    _log(f"Generating oneway plots for {len(features)} features...")
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
                    target_weighted=False,
                    n_bins=int(n_bins),
                    is_categorical=is_categorical,
                    save_path=str(save_path),
                    show=False,
                )
                if save_path.exists():
                    saved += 1
                if i % 5 == 0 or i == len(features):
                    _log(f"  [{tag}] [{i}/{len(features)}] {feature}")
            except Exception as exc:
                _log(f"  [Warn] [{tag}] {feature} failed: {exc}")

    total_expected = len(features) * dataset_count
    _log(f"Complete. Saved {saved}/{total_expected} plots to: {out_dir}")
    return f"Saved {saved} plots to {out_dir}"


def _run_prediction_plot_workflow(
    *,
    cfg: dict,
    cfg_path: Path,
    xgb_cfg: Optional[dict],
    xgb_cfg_path: Optional[Path],
    resn_cfg: Optional[dict],
    resn_cfg_path: Optional[Path],
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
    provided_model_cfgs: Dict[str, Tuple[dict, Path]] = {}
    if xgb_cfg is not None and xgb_cfg_path is not None:
        provided_model_cfgs["xgb"] = (xgb_cfg, xgb_cfg_path)
    if resn_cfg is not None and resn_cfg_path is not None:
        provided_model_cfgs["resn"] = (resn_cfg, resn_cfg_path)
    if not provided_model_cfgs:
        raise ValueError("At least one model config is required for plotting.")

    cfg_model_keys = cfg.get("model_keys") or []
    if isinstance(cfg_model_keys, str):
        requested_model_keys = _dedupe_list(_parse_csv_list(cfg_model_keys))
    else:
        requested_model_keys = _dedupe_list(
            [str(key).strip() for key in cfg_model_keys if str(key).strip()]
        )
    if requested_model_keys:
        skipped = [key for key in requested_model_keys if key not in provided_model_cfgs]
        if skipped:
            _log(
                "[Warn] Missing model configs for requested model_keys; skipped: "
                f"{skipped}"
            )
        model_keys = [key for key in requested_model_keys if key in provided_model_cfgs]
    else:
        model_keys = list(provided_model_cfgs.keys())
    if not model_keys:
        available = sorted(provided_model_cfgs.keys())
        raise ValueError(
            "No runnable model keys found for plot config. "
            f"Requested={requested_model_keys}, available={available}."
        )

    model_cfg_map = {key: provided_model_cfgs[key][1] for key in model_keys}
    output_dir_map = {
        key: _resolve_output_dir(provided_model_cfgs[key][0], provided_model_cfgs[key][1])
        for key in model_keys
    }
    output_dir_path_map = {key: Path(value).resolve() for key, value in output_dir_map.items()}
    plot_path_style_map = {
        key: _resolve_plot_style(provided_model_cfgs[key][0]) for key in model_keys
    }

    def _get_plot_config(model_key: str) -> Tuple[str, str]:
        return (
            output_dir_map.get(model_key, _resolve_output_dir(cfg, cfg_path)),
            plot_path_style_map.get(model_key, _resolve_plot_style(cfg)),
        )

    search_roots = _build_search_roots(
        model_search_dir,
        cfg_path.parent,
        *[model_cfg_path.parent for _, model_cfg_path in provided_model_cfgs.values()],
        Path.cwd(),
    )
    raw_model_paths = {"xgb": xgb_model_path, "resn": resn_model_path}
    model_output_overrides: Dict[str, Optional[Path]] = {}
    for key in model_keys:
        model_output_overrides[key] = resolve_model_output_override(
            model_name=model_name,
            model_key=key,
            model_path=raw_model_paths.get(key),
            search_roots=search_roots,
            output_root=output_dir_path_map[key],
            label=f"{key}_model_path",
        )

    def _load_predictor(model_cfg_path: Path, model_key: str):
        output_override = model_output_overrides.get(model_key)
        kwargs: Dict[str, object] = {"model_name": model_name}
        if output_override is not None:
            kwargs["output_dir"] = output_override
        return _load_predictor_from_cfg(model_cfg_path, model_key, **kwargs)

    default_model_labels = {"xgb": "Xgboost", "resn": "ResNet"}
    labels = cfg.get("model_labels") or {}
    plot_cfg = cfg.get("plot", {}) if isinstance(cfg.get("plot"), dict) else {}
    n_bins = plot_cfg.get("n_bins", 10)
    weight_col = cfg["weight"]
    target_col = cfg["target"]
    predict_chunk_rows = _resolve_predict_chunk_rows(cfg, plot_cfg)
    if predict_chunk_rows > 0:
        _log(f"[Plot] Chunked prediction enabled: {predict_chunk_rows} rows/chunk")

    def _model_label(model_key: str) -> str:
        return str(labels.get(model_key, default_model_labels.get(model_key, model_key)))

    predictors = {key: _load_predictor(model_cfg_map[key], key) for key in model_keys}

    pred_train = {}
    pred_test = {}
    for key, predictor in predictors.items():
        required_features = _required_feature_names_from_predictor(predictor)
        if len(train_df) > 0:
            _log_missing_feature_coverage(
                model_key=key,
                split_tag="train",
                frame=train_df,
                required_features=required_features,
            )
        if len(test_df) > 0:
            _log_missing_feature_coverage(
                model_key=key,
                split_tag="valid",
                frame=test_df,
                required_features=required_features,
            )
        if len(train_df) > 0:
            pred_train[key] = _predict_vector(
                predictor,
                train_df,
                model_key=key,
                split_tag="train",
                chunk_rows=predict_chunk_rows,
            )
            if len(pred_train[key]) != len(train_df):
                raise ValueError(f"Train prediction length mismatch for {key}")
            _warn_if_near_constant_predictions(
                values=np.asarray(pred_train[key], dtype=float).reshape(-1),
                model_key=key,
                split_tag="train",
            )
            _log_prediction_split_diagnostics(
                model_key=key,
                split_tag="train",
                pred_values=np.asarray(pred_train[key], dtype=float).reshape(-1),
                raw_frame=train_raw,
                target_col=target_col,
                weight_col=weight_col,
            )
        else:
            pred_train[key] = []
        if len(test_df) > 0:
            pred_test[key] = _predict_vector(
                predictor,
                test_df,
                model_key=key,
                split_tag="valid",
                chunk_rows=predict_chunk_rows,
            )
            if len(pred_test[key]) != len(test_df):
                raise ValueError(f"Test prediction length mismatch for {key}")
            _warn_if_near_constant_predictions(
                values=np.asarray(pred_test[key], dtype=float).reshape(-1),
                model_key=key,
                split_tag="valid",
            )
            _log_prediction_split_diagnostics(
                model_key=key,
                split_tag="valid",
                pred_values=np.asarray(pred_test[key], dtype=float).reshape(-1),
                raw_frame=test_raw,
                target_col=target_col,
                weight_col=weight_col,
            )
        else:
            pred_test[key] = []

    plot_train = train_raw.copy(deep=False)
    plot_test = test_raw.copy(deep=False)
    for key in model_keys:
        if len(plot_train) > 0:
            plot_train[f"pred_{key}"] = pred_train[key]
        if len(plot_test) > 0:
            plot_test[f"pred_{key}"] = pred_test[key]

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

    raw_split_scope = str(
        plot_cfg.get("split_scope", cfg.get("plot_split_scope", "both")) or "both"
    ).strip().lower()
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
    split_scope = split_scope_aliases.get(raw_split_scope)
    if split_scope is None:
        raise ValueError(
            f"Unsupported plot split scope: {raw_split_scope!r}. "
            "Use one of: train, valid, both."
        )

    all_datasets = []
    if train_ready:
        all_datasets.append(("train", "Train Data", plot_train))
    if test_ready:
        all_datasets.append(("valid", "Validation Data", plot_test))

    if split_scope == "train":
        datasets = [item for item in all_datasets if item[0] == "train"]
    elif split_scope == "valid":
        datasets = [item for item in all_datasets if item[0] == "valid"]
    else:
        datasets = list(all_datasets)

    if not datasets:
        available = [tag for tag, _title, _df in all_datasets]
        raise ValueError(
            f"Requested split_scope={split_scope!r}, but available splits are {available}."
        )
    _log(f"[Plot] split_scope={split_scope}, datasets={[title for _tag, title, _df in datasets]}")

    for pred_key in model_keys:
        pred_label = _model_label(pred_key)
        pred_col = f"pred_{pred_key}"
        pred_tag = _safe_tag(pred_label or pred_col)
        output_root, plot_style = _get_plot_config(pred_key)
        for feature in oneway_features:
            feature_datasets = [
                (split_title, split_df)
                for _split_tag, split_title, split_df in datasets
                if pred_col in split_df.columns and feature in split_df.columns
            ]
            if not feature_datasets:
                continue

            style = PlotStyle()
            fig, axes = plt.subplots(1, len(feature_datasets), figsize=PLOT_GRID_FIGSIZE)
            if len(feature_datasets) == 1:
                axes = [axes]
            for ax, (split_title, split_df) in zip(axes, feature_datasets):
                plot_cols = list(dict.fromkeys([feature, weight_col, "w_act", pred_col]))
                plot_input = split_df.loc[:, plot_cols]
                plot_oneway(
                    plot_input,
                    feature=feature,
                    weight_col=weight_col,
                    target_col="w_act",
                    target_weighted=True,
                    pred_col=pred_col,
                    pred_label=pred_label,
                    n_bins=n_bins,
                    is_categorical=feature in oneway_categorical,
                    title=f"Analysis of {feature} ({split_title})",
                    ax=ax,
                    show=False,
                    style=style,
                )
            plt.subplots_adjust(wspace=0.3)

            save_path = _resolve_plot_path(
                output_root,
                plot_style,
                f"{model_name}/oneway/post",
                f"00_{model_name}_{feature}_oneway_{pred_tag}.png",
            )
            finalize_figure(fig, save_path=save_path, show=False, style=style)

    def _plot_lift_for_model(pred_key: str, pred_label: str) -> None:
        if not datasets:
            return
        output_root, plot_style = _get_plot_config(pred_key)
        style = PlotStyle()
        fig, axes = plt.subplots(1, len(datasets), figsize=PLOT_GRID_FIGSIZE)
        if len(datasets) == 1:
            axes = [axes]
        for ax, (_split_tag, title, data) in zip(axes, datasets):
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
        and all(any(f"pred_{k}" in frame.columns for _tag, _title, frame in datasets) for k in ["xgb", "resn"])
        and datasets
    ):
        style = PlotStyle()
        fig, axes = plt.subplots(1, len(datasets), figsize=PLOT_GRID_FIGSIZE)
        if len(datasets) == 1:
            axes = [axes]
        for ax, (_split_tag, title, data) in zip(axes, datasets):
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
        double_lift_root = _resolve_double_lift_dir(model_name=model_name)
        save_path = (
            double_lift_root
            / (
                f"02_{model_name}_dlift_xgb_vs_resn_"
                f"{_safe_tag(cfg_path.stem)}_"
                f"xgb_{_safe_tag(xgb_cfg_path.stem)}_"
                f"resn_{_safe_tag(resn_cfg_path.stem)}.png"
            )
        ).resolve()
        finalize_figure(fig, save_path=str(save_path), show=False, style=style)

    _log("Plots saved under:")
    for key in model_keys:
        output_root, _ = _get_plot_config(key)
        _log(f"  - {key}: {output_root}/plot/{model_name}")
    return "Plotting complete."


def run_plot_direct(
    *,
    cfg_path: str,
    xgb_cfg_path: Optional[str] = None,
    resn_cfg_path: Optional[str] = None,
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    xgb_model_path: Optional[str] = None,
    resn_model_path: Optional[str] = None,
    model_search_dir: Optional[str] = None,
    oneway_features: Optional[Any] = None,
) -> str:
    cfg_path = Path(cfg_path).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    explicit_xgb_cfg_path = str(xgb_cfg_path or "").strip()
    explicit_resn_cfg_path = str(resn_cfg_path or "").strip()
    use_cfg_alias_fallback = not explicit_xgb_cfg_path and not explicit_resn_cfg_path

    def _resolve_model_cfg_path(
        raw_path: Optional[str],
        *aliases: str,
    ) -> Optional[Path]:
        path_value = str(raw_path or "").strip()
        if not path_value:
            for alias in aliases:
                alias_value = str(cfg.get(alias, "") or "").strip()
                if alias_value:
                    path_value = alias_value
                    break
        if not path_value:
            return None
        cfg_file = Path(path_value)
        if not cfg_file.is_absolute():
            cfg_file = (cfg_path.parent / cfg_file).resolve()
        else:
            cfg_file = cfg_file.resolve()
        return cfg_file

    if use_cfg_alias_fallback:
        xgb_cfg_path_obj = _resolve_model_cfg_path(
            xgb_cfg_path, "xgb_cfg_path", "direct_xgb_cfg_path"
        )
        resn_cfg_path_obj = _resolve_model_cfg_path(
            resn_cfg_path, "resn_cfg_path", "direct_resn_cfg_path"
        )
    else:
        xgb_cfg_path_obj = _resolve_model_cfg_path(xgb_cfg_path)
        resn_cfg_path_obj = _resolve_model_cfg_path(resn_cfg_path)
    xgb_cfg = json.loads(xgb_cfg_path_obj.read_text(encoding="utf-8")) if xgb_cfg_path_obj else None
    resn_cfg = json.loads(resn_cfg_path_obj.read_text(encoding="utf-8")) if resn_cfg_path_obj else None

    available_cfgs: Dict[str, Tuple[dict, Path]] = {}
    if xgb_cfg is not None and xgb_cfg_path_obj is not None:
        available_cfgs["xgb"] = (xgb_cfg, xgb_cfg_path_obj)
    if resn_cfg is not None and resn_cfg_path_obj is not None:
        available_cfgs["resn"] = (resn_cfg, resn_cfg_path_obj)
    if not available_cfgs:
        raise ValueError("run_plot_direct requires at least one of xgb_cfg_path or resn_cfg_path.")

    cfg_model_keys = cfg.get("model_keys") or []
    if isinstance(cfg_model_keys, str):
        requested_model_keys = _dedupe_list(_parse_csv_list(cfg_model_keys))
    else:
        requested_model_keys = _dedupe_list(
            [str(key).strip() for key in cfg_model_keys if str(key).strip()]
        )

    split_source_key: Optional[str] = None
    for key in requested_model_keys:
        if key in available_cfgs:
            split_source_key = key
            break
    if split_source_key is None:
        split_source_key = next(iter(available_cfgs.keys()))
    split_data_cfg, split_data_cfg_path = available_cfgs[split_source_key]

    model_name = f"{cfg['model_list'][0]}_{cfg['model_categories'][0]}"
    cfg_data_dir = str(cfg.get("data_dir", "") or "").strip()
    split_data_dir = str(split_data_cfg.get("data_dir", "") or "").strip()
    if cfg_data_dir and split_data_dir and cfg_data_dir != split_data_dir:
        _log(
            f"[Info] plot_direct data_dir override: cfg={cfg_data_dir!r} -> "
            f"{split_source_key}_cfg={split_data_dir!r}"
        )
    train_raw, test_raw, _raw, _ = load_raw_splits(
        split_cfg=cfg,
        data_cfg=split_data_cfg,
        data_cfg_path=split_data_cfg_path,
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
            _log(f"[Warn] Ignored factors not present in config.feature_list: {ignored}")
        if not filtered_features:
            raise ValueError("Selected oneway factors are not present in config.feature_list.")
        feature_list = filtered_features
    categorical_features = _dedupe_list(cfg.get("categorical_features") or [])

    return _run_prediction_plot_workflow(
        cfg=cfg,
        cfg_path=cfg_path,
        xgb_cfg=xgb_cfg,
        xgb_cfg_path=xgb_cfg_path_obj,
        resn_cfg=resn_cfg,
        resn_cfg_path=resn_cfg_path_obj,
        model_name=model_name,
        train_raw=train_raw,
        test_raw=test_raw,
        train_df=train_raw,
        test_df=test_raw,
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

    raw_feature_list = _dedupe_list(ft_cfg.get("feature_list") or [])
    raw_categorical_features = _dedupe_list(ft_cfg.get("categorical_features") or [])
    plot_feature_list = _dedupe_list(cfg.get("feature_list") or [])
    model_feature_list = _dedupe_list(
        list(xgb_cfg.get("feature_list") or []) + list(resn_cfg.get("feature_list") or [])
    )

    split_cfg = ft_cfg
    split_data_cfg = ft_cfg
    split_data_cfg_path = ft_cfg_path

    model_cfg_map = {"xgb": (xgb_cfg, xgb_cfg_path), "resn": (resn_cfg, resn_cfg_path)}
    cfg_model_keys = _dedupe_list(cfg.get("model_keys") or ["xgb", "resn"])
    candidate_model_keys = [k for k in cfg_model_keys if k in model_cfg_map]
    if not candidate_model_keys:
        candidate_model_keys = ["xgb", "resn"]
    embed_model_key = candidate_model_keys[0]
    embed_data_cfg, embed_data_cfg_path = model_cfg_map[embed_model_key]

    if not use_runtime_ft_embedding:
        try:
            embed_data_path = _resolve_data_path(embed_data_cfg, embed_data_cfg_path, model_name)
            _log(
                f"[Plot] Precomputed embed source: model_key={embed_model_key}, "
                f"cfg={embed_data_cfg_path}, data={embed_data_path}"
            )
            for key in candidate_model_keys[1:]:
                peer_cfg, peer_cfg_path = model_cfg_map[key]
                peer_data_path = _resolve_data_path(peer_cfg, peer_cfg_path, model_name)
                if str(peer_data_path) != str(embed_data_path):
                    _log(
                        f"[Warn] {key} precomputed data path differs from {embed_model_key}: "
                        f"{peer_data_path} vs {embed_data_path}. "
                        f"Using {embed_model_key} data as unified inference frame."
                    )
        except Exception:
            _log(
                f"[Warn] Failed to resolve precomputed embed path from {embed_data_cfg_path}. "
                "Proceeding with configured defaults."
            )

    _log(f"[Plot] Split source cfg: {split_data_cfg_path}")
    required_raw_columns = _dedupe_list(
        [
            str(cfg.get("target", "") or "").strip(),
            str(cfg.get("weight", "") or "").strip(),
            *raw_feature_list,
            *plot_feature_list,
        ]
    )
    if required_raw_columns:
        _log(f"[Plot] Loading required raw columns only: {len(required_raw_columns)} columns")

    if ft_cfg.get("geo_feature_nmes"):
        raise ValueError("FT inference with geo tokens is not supported in this workflow.")
    train_raw, test_raw, raw, use_explicit_split = load_raw_splits(
        split_cfg=split_cfg,
        data_cfg=split_data_cfg,
        data_cfg_path=split_data_cfg_path,
        model_name=model_name,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        required_columns=required_raw_columns,
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
        required_columns=model_feature_list,
        embed_cfg=embed_data_cfg,
        embed_cfg_path=embed_data_cfg_path,
        embed_path_resolver=_resolve_data_path,
    )

    feature_list = _dedupe_list(cfg.get("feature_list") or [])
    categorical_features = _dedupe_list(cfg.get("categorical_features") or [])
    selected_features = _normalize_oneway_feature_override(oneway_features)
    base_oneway_features = raw_feature_list or feature_list
    if selected_features:
        feature_set = set(base_oneway_features)
        filtered_features = [f for f in selected_features if not feature_set or f in feature_set]
        ignored = [f for f in selected_features if feature_set and f not in feature_set]
        if ignored:
            _log(f"[Warn] Ignored factors not present in config.feature_list: {ignored}")
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

