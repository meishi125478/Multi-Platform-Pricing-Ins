"""Config-driven explain runner for trained BayesOpt models."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from .. import bayesopt as ropt  # type: ignore
    from .utils.cli_common import (  # type: ignore
        build_model_names,
        dedupe_preserve_order,
        load_dataset,
        resolve_data_path,
        coerce_dataset_types,
        split_train_test,
    )
    from .utils.cli_config import (  # type: ignore
        add_config_json_arg,
        add_output_dir_arg,
        resolve_and_load_config,
        resolve_data_config,
        resolve_explain_output_overrides,
        resolve_explain_save_dir,
        resolve_explain_save_root,
        resolve_model_path_value,
        resolve_split_config,
        resolve_runtime_config,
        resolve_output_dirs,
    )
except Exception:  # pragma: no cover
    try:
        import bayesopt as ropt  # type: ignore
        from utils.cli_common import (  # type: ignore
            build_model_names,
            dedupe_preserve_order,
            load_dataset,
            resolve_data_path,
            coerce_dataset_types,
            split_train_test,
        )
        from utils.cli_config import (  # type: ignore
            add_config_json_arg,
            add_output_dir_arg,
            resolve_and_load_config,
            resolve_data_config,
            resolve_explain_output_overrides,
            resolve_explain_save_dir,
            resolve_explain_save_root,
            resolve_model_path_value,
            resolve_split_config,
            resolve_runtime_config,
            resolve_output_dirs,
        )
    except Exception:
        import ins_pricing.modelling.core.bayesopt as ropt  # type: ignore
        from ins_pricing.cli.utils.cli_common import (  # type: ignore
            build_model_names,
            dedupe_preserve_order,
            load_dataset,
            resolve_data_path,
            coerce_dataset_types,
            split_train_test,
        )
        from ins_pricing.cli.utils.cli_config import (  # type: ignore
            add_config_json_arg,
            add_output_dir_arg,
            resolve_and_load_config,
            resolve_data_config,
            resolve_explain_output_overrides,
            resolve_explain_save_dir,
            resolve_explain_save_root,
            resolve_model_path_value,
            resolve_split_config,
            resolve_runtime_config,
            resolve_output_dirs,
        )

try:
    from .utils.run_logging import configure_run_logging  # type: ignore
except Exception:  # pragma: no cover
    try:
        from utils.run_logging import configure_run_logging  # type: ignore
    except Exception:  # pragma: no cover
        configure_run_logging = None  # type: ignore


_SUPPORTED_METHODS = {"permutation", "shap", "integrated_gradients"}
_METHOD_ALIASES = {
    "ig": "integrated_gradients",
    "integrated": "integrated_gradients",
    "intgrad": "integrated_gradients",
}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _load_dataset(
    path: Path,
    *,
    data_format: str,
    dtype_map: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    raw = load_dataset(
        path,
        data_format=data_format,
        dtype_map=dtype_map,
        low_memory=False,
    )
    return coerce_dataset_types(raw)


def _normalize_methods(raw: Sequence[str]) -> List[str]:
    methods: List[str] = []
    for item in raw:
        key = str(item).strip().lower()
        if not key:
            continue
        key = _METHOD_ALIASES.get(key, key)
        if key not in _SUPPORTED_METHODS:
            raise ValueError(f"Unsupported explain method: {item}")
        methods.append(key)
    return dedupe_preserve_order(methods)


def _save_series(series: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame(name="importance").to_csv(path, index=True)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _shap_importance(values: Any, feature_names: Sequence[str]) -> pd.Series:
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[0]
    scores = np.mean(np.abs(arr), axis=0)
    return pd.Series(scores, index=list(feature_names)).sort_values(ascending=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explainability (permutation/SHAP/IG) on trained models."
    )
    add_config_json_arg(
        parser,
        help_text="Path to config.json (same schema as training).",
    )
    parser.add_argument(
        "--model-keys",
        nargs="+",
        default=None,
        choices=["glm", "xgb", "resn", "ft", "gnn", "all"],
        help="Model keys to load for explanation (default from config.explain.model_keys).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Explain methods: permutation, shap, integrated_gradients (default from config.explain.methods).",
    )
    add_output_dir_arg(
        parser,
        help_text="Override output root for loading models/results.",
    )
    parser.add_argument(
        "--eval-path",
        default=None,
        help="Override validation CSV path (supports {model_name}).",
    )
    parser.add_argument(
        "--on-train",
        action="store_true",
        help="Explain on train split instead of validation/test.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override output directory for explanation artifacts.",
    )
    return parser.parse_args()


def _explain_for_model(
    model: ropt.BayesOptModel,
    *,
    model_name: str,
    model_keys: List[str],
    methods: List[str],
    on_train: bool,
    save_dir: Path,
    explain_cfg: Dict[str, Any],
) -> None:
    perm_cfg = dict(explain_cfg.get("permutation") or {})
    shap_cfg = dict(explain_cfg.get("shap") or {})
    ig_cfg = dict(explain_cfg.get("integrated_gradients") or {})

    perm_metric = perm_cfg.get("metric", explain_cfg.get("metric", "auto"))
    perm_repeats = int(perm_cfg.get("n_repeats", 5))
    perm_max_rows = perm_cfg.get("max_rows", 5000)
    perm_random_state = perm_cfg.get("random_state", None)

    shap_background = int(shap_cfg.get("n_background", 500))
    shap_samples = int(shap_cfg.get("n_samples", 200))
    shap_save_values = bool(shap_cfg.get("save_values", False))

    ig_steps = int(ig_cfg.get("steps", 50))
    ig_batch_size = int(ig_cfg.get("batch_size", 256))
    ig_target = ig_cfg.get("target", None)
    ig_baseline = ig_cfg.get("baseline", None)
    ig_baseline_num = ig_cfg.get("baseline_num", None)
    ig_baseline_geo = ig_cfg.get("baseline_geo", None)
    ig_save_values = bool(ig_cfg.get("save_values", False))

    for key in model_keys:
        trainer = model.trainers.get(key)
        if trainer is None:
            print(f"[Explain] Skip {model_name}/{key}: trainer not available.")
            continue
        model.load_model(key)
        trained_model = getattr(model, f"{key}_best", None)
        if trained_model is None:
            print(f"[Explain] Skip {model_name}/{key}: model not loaded.")
            continue

        if key == "ft" and str(model.config.ft_role) != "model":
            print(f"[Explain] Skip {model_name}/ft: ft_role != 'model'.")
            continue

        for method in methods:
            if method == "permutation" and key not in {"xgb", "resn", "ft"}:
                print(f"[Explain] Skip permutation for {model_name}/{key}.")
                continue
            if method == "shap" and key not in {"glm", "xgb", "resn", "ft"}:
                print(f"[Explain] Skip shap for {model_name}/{key}.")
                continue
            if method == "integrated_gradients" and key not in {"resn", "ft"}:
                print(f"[Explain] Skip integrated gradients for {model_name}/{key}.")
                continue

            if method == "permutation":
                try:
                    result = model.compute_permutation_importance(
                        key,
                        on_train=on_train,
                        metric=perm_metric,
                        n_repeats=perm_repeats,
                        max_rows=perm_max_rows,
                        random_state=perm_random_state,
                    )
                except Exception as exc:
                    print(f"[Explain] permutation failed for {model_name}/{key}: {exc}")
                    continue
                out_path = save_dir / f"{_safe_name(model_name)}_{key}_permutation.csv"
                _save_df(result, out_path)
                print(f"[Explain] Saved permutation -> {out_path}")

            if method == "shap":
                try:
                    if key == "glm":
                        shap_result = model.compute_shap_glm(
                            n_background=shap_background,
                            n_samples=shap_samples,
                            on_train=on_train,
                        )
                    elif key == "xgb":
                        shap_result = model.compute_shap_xgb(
                            n_background=shap_background,
                            n_samples=shap_samples,
                            on_train=on_train,
                        )
                    elif key == "resn":
                        shap_result = model.compute_shap_resn(
                            n_background=shap_background,
                            n_samples=shap_samples,
                            on_train=on_train,
                        )
                    else:
                        shap_result = model.compute_shap_ft(
                            n_background=shap_background,
                            n_samples=shap_samples,
                            on_train=on_train,
                        )
                except Exception as exc:
                    print(f"[Explain] shap failed for {model_name}/{key}: {exc}")
                    continue

                shap_values = shap_result.get("shap_values")
                X_explain = shap_result.get("X_explain")
                feature_names = (
                    list(X_explain.columns)
                    if isinstance(X_explain, pd.DataFrame)
                    else list(model.factor_nmes)
                )
                importance = _shap_importance(shap_values, feature_names)
                out_path = save_dir / f"{_safe_name(model_name)}_{key}_shap_importance.csv"
                _save_series(importance, out_path)
                print(f"[Explain] Saved SHAP importance -> {out_path}")

                if shap_save_values:
                    values_path = save_dir / f"{_safe_name(model_name)}_{key}_shap_values.npy"
                    np.save(values_path, np.array(shap_values, dtype=object), allow_pickle=True)
                    if isinstance(X_explain, pd.DataFrame):
                        x_path = save_dir / f"{_safe_name(model_name)}_{key}_shap_X.csv"
                        _save_df(X_explain, x_path)
                    meta_path = save_dir / f"{_safe_name(model_name)}_{key}_shap_meta.json"
                    meta = {
                        "base_value": shap_result.get("base_value"),
                        "n_samples": int(len(X_explain)) if X_explain is not None else None,
                    }
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            if method == "integrated_gradients":
                try:
                    if key == "resn":
                        ig_result = model.compute_integrated_gradients_resn(
                            on_train=on_train,
                            baseline=ig_baseline,
                            steps=ig_steps,
                            batch_size=ig_batch_size,
                            target=ig_target,
                        )
                        series = ig_result.get("importance")
                        if isinstance(series, pd.Series):
                            out_path = save_dir / f"{_safe_name(model_name)}_{key}_ig_importance.csv"
                            _save_series(series, out_path)
                            print(f"[Explain] Saved IG importance -> {out_path}")
                        if ig_save_values and "attributions" in ig_result:
                            attr_path = save_dir / f"{_safe_name(model_name)}_{key}_ig_attributions.npy"
                            np.save(attr_path, ig_result.get("attributions"))
                    else:
                        ig_result = model.compute_integrated_gradients_ft(
                            on_train=on_train,
                            baseline_num=ig_baseline_num,
                            baseline_geo=ig_baseline_geo,
                            steps=ig_steps,
                            batch_size=ig_batch_size,
                            target=ig_target,
                        )
                        series_num = ig_result.get("importance_num")
                        series_geo = ig_result.get("importance_geo")
                        if isinstance(series_num, pd.Series):
                            out_path = save_dir / f"{_safe_name(model_name)}_{key}_ig_num_importance.csv"
                            _save_series(series_num, out_path)
                            print(f"[Explain] Saved IG num importance -> {out_path}")
                        if isinstance(series_geo, pd.Series):
                            out_path = save_dir / f"{_safe_name(model_name)}_{key}_ig_geo_importance.csv"
                            _save_series(series_geo, out_path)
                            print(f"[Explain] Saved IG geo importance -> {out_path}")
                        if ig_save_values:
                            if ig_result.get("attributions_num") is not None:
                                attr_path = save_dir / f"{_safe_name(model_name)}_{key}_ig_num_attributions.npy"
                                np.save(attr_path, ig_result.get("attributions_num"))
                            if ig_result.get("attributions_geo") is not None:
                                attr_path = save_dir / f"{_safe_name(model_name)}_{key}_ig_geo_attributions.npy"
                                np.save(attr_path, ig_result.get("attributions_geo"))
                except Exception as exc:
                    print(f"[Explain] integrated gradients failed for {model_name}/{key}: {exc}")
                    continue


def explain_from_config(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parents[1]
    config_path, cfg = resolve_and_load_config(
        args.config_json,
        script_dir,
        required_keys=["data_dir", "model_list", "model_categories", "target", "weight"],
    )

    data_dir, data_format, data_path_template, dtype_map = resolve_data_config(
        cfg,
        config_path,
        create_data_dir=True,
    )

    runtime_cfg = resolve_runtime_config(cfg)
    output_cfg = resolve_output_dirs(
        cfg,
        config_path,
        output_override=args.output_dir,
    )
    output_dir = output_cfg["output_dir"]

    split_cfg = resolve_split_config(cfg)
    prop_test = split_cfg["prop_test"]
    rand_seed = runtime_cfg["rand_seed"]
    split_strategy = split_cfg["split_strategy"]
    split_group_col = split_cfg["split_group_col"]
    split_time_col = split_cfg["split_time_col"]
    split_time_ascending = split_cfg["split_time_ascending"]

    explain_cfg = dict(cfg.get("explain") or {})

    model_keys = args.model_keys or explain_cfg.get("model_keys") or ["xgb"]
    if "all" in model_keys:
        model_keys = ["glm", "xgb", "resn", "ft", "gnn"]
    model_keys = dedupe_preserve_order([str(x) for x in model_keys])

    method_list = args.methods or explain_cfg.get("methods") or ["permutation"]
    methods = _normalize_methods([str(x) for x in method_list])

    on_train = bool(args.on_train or explain_cfg.get("on_train", False))

    model_names = build_model_names(cfg["model_list"], cfg["model_categories"])
    if not model_names:
        raise ValueError("No model names generated from model_list/model_categories.")

    save_root = resolve_explain_save_root(
        args.save_dir or explain_cfg.get("save_dir"),
        config_path.parent,
    )

    for model_name in model_names:
        train_path = resolve_model_path_value(
            explain_cfg.get("train_path"),
            model_name=model_name,
            base_dir=config_path.parent,
            data_dir=data_dir,
        )
        if train_path is None:
            train_path = resolve_data_path(
                data_dir,
                model_name,
                data_format=data_format,
                path_template=data_path_template,
            )
        if not train_path.exists():
            raise FileNotFoundError(f"Missing training dataset: {train_path}")

        validation_override = args.eval_path or explain_cfg.get("validation_path") or explain_cfg.get("eval_path")
        validation_path = resolve_model_path_value(
            validation_override,
            model_name=model_name,
            base_dir=config_path.parent,
            data_dir=data_dir,
        )

        raw = _load_dataset(
            train_path,
            data_format=data_format,
            dtype_map=dtype_map,
        )
        if validation_path is not None:
            if not validation_path.exists():
                raise FileNotFoundError(f"Missing validation dataset: {validation_path}")
            train_df = raw
            test_df = _load_dataset(
                validation_path,
                data_format=data_format,
                dtype_map=dtype_map,
            )
        else:
            if float(prop_test) <= 0:
                train_df = raw
                test_df = raw.copy()
            else:
                train_df, test_df = split_train_test(
                    raw,
                    holdout_ratio=prop_test,
                    strategy=split_strategy,
                    group_col=split_group_col,
                    time_col=split_time_col,
                    time_ascending=split_time_ascending,
                    rand_seed=rand_seed,
                    reset_index_mode="time_group",
                    ratio_label="prop_test",
                    include_strategy_in_ratio_error=True,
                )

        binary_target = cfg.get("binary_target") or cfg.get("binary_resp_nme")
        feature_list = cfg.get("feature_list")
        categorical_features = cfg.get("categorical_features")
        plot_path_style = runtime_cfg["plot_path_style"]

        model = ropt.BayesOptModel(
            train_df,
            test_df,
            model_name,
            cfg["target"],
            cfg["weight"],
            feature_list,
            task_type=str(cfg.get("task_type", "regression")),
            binary_resp_nme=binary_target,
            cate_list=categorical_features,
            prop_test=prop_test,
            rand_seed=rand_seed,
            epochs=int(runtime_cfg["epochs"]),
            use_gpu=bool(cfg.get("use_gpu", True)),
            output_dir=output_dir,
            xgb_max_depth_max=runtime_cfg["xgb_max_depth_max"],
            xgb_n_estimators_max=runtime_cfg["xgb_n_estimators_max"],
            resn_weight_decay=cfg.get("resn_weight_decay"),
            final_ensemble=bool(cfg.get("final_ensemble", False)),
            final_ensemble_k=int(cfg.get("final_ensemble_k", 3)),
            final_refit=bool(cfg.get("final_refit", True)),
            optuna_storage=runtime_cfg["optuna_storage"],
            optuna_study_prefix=runtime_cfg["optuna_study_prefix"],
            best_params_files=runtime_cfg["best_params_files"],
            gnn_use_approx_knn=cfg.get("gnn_use_approx_knn", True),
            gnn_approx_knn_threshold=cfg.get("gnn_approx_knn_threshold", 50000),
            gnn_graph_cache=cfg.get("gnn_graph_cache"),
            gnn_max_gpu_knn_nodes=cfg.get("gnn_max_gpu_knn_nodes", 200000),
            gnn_knn_gpu_mem_ratio=cfg.get("gnn_knn_gpu_mem_ratio", 0.9),
            gnn_knn_gpu_mem_overhead=cfg.get("gnn_knn_gpu_mem_overhead", 2.0),
            region_province_col=cfg.get("region_province_col"),
            region_city_col=cfg.get("region_city_col"),
            region_effect_alpha=cfg.get("region_effect_alpha"),
            geo_feature_nmes=cfg.get("geo_feature_nmes"),
            geo_token_hidden_dim=cfg.get("geo_token_hidden_dim"),
            geo_token_layers=cfg.get("geo_token_layers"),
            geo_token_dropout=cfg.get("geo_token_dropout"),
            geo_token_k_neighbors=cfg.get("geo_token_k_neighbors"),
            geo_token_learning_rate=cfg.get("geo_token_learning_rate"),
            geo_token_epochs=cfg.get("geo_token_epochs"),
            ft_role=str(cfg.get("ft_role", "model")),
            ft_feature_prefix=str(cfg.get("ft_feature_prefix", "ft_emb")),
            ft_num_numeric_tokens=cfg.get("ft_num_numeric_tokens"),
            infer_categorical_max_unique=int(cfg.get("infer_categorical_max_unique", 50)),
            infer_categorical_max_ratio=float(cfg.get("infer_categorical_max_ratio", 0.05)),
            reuse_best_params=runtime_cfg["reuse_best_params"],
            plot_path_style=plot_path_style,
        )

        output_overrides = resolve_explain_output_overrides(
            explain_cfg,
            model_name=model_name,
            base_dir=config_path.parent,
        )
        model_dir_override = output_overrides.get("model_dir")
        if model_dir_override is not None:
            model.output_manager.model_dir = model_dir_override
        result_dir_override = output_overrides.get("result_dir")
        if result_dir_override is not None:
            model.output_manager.result_dir = result_dir_override
        plot_dir_override = output_overrides.get("plot_dir")
        if plot_dir_override is not None:
            model.output_manager.plot_dir = plot_dir_override

        save_dir = resolve_explain_save_dir(
            save_root,
            result_dir=model.output_manager.result_dir,
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Explain model {model_name} ===")
        _explain_for_model(
            model,
            model_name=model_name,
            model_keys=model_keys,
            methods=methods,
            on_train=on_train,
            save_dir=save_dir,
            explain_cfg=explain_cfg,
        )


def main() -> None:
    if configure_run_logging:
        configure_run_logging(prefix="explain_entry")
    args = _parse_args()
    explain_from_config(args)


if __name__ == "__main__":
    main()
