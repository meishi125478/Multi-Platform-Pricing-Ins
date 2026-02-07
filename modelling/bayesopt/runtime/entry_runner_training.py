from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BayesOptRunnerDeps:
    ropt: Any
    pytorch_trainers: Sequence[str]
    build_model_names: Callable[[Any, Any], List[str]]
    dedupe_preserve_order: Callable[[List[str]], List[str]]
    load_dataset: Callable[..., Any]
    resolve_data_path: Callable[..., Path]
    resolve_path: Callable[..., Optional[Path]]
    fingerprint_file: Callable[..., Dict[str, Any]]
    coerce_dataset_types: Callable[[Any], Any]
    split_train_test: Callable[..., Tuple[Any, Any]]
    resolve_and_load_config: Callable[..., Tuple[Path, Dict[str, Any]]]
    resolve_data_config: Callable[..., Tuple[Any, Any, Any, Any]]
    resolve_report_config: Callable[[Dict[str, Any]], Dict[str, Any]]
    resolve_split_config: Callable[[Dict[str, Any]], Dict[str, Any]]
    resolve_runtime_config: Callable[[Dict[str, Any]], Dict[str, Any]]
    resolve_output_dirs: Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class BayesOptRunnerHooks:
    plot_loss_curve_for_trainer: Callable[[str, Any], None]
    compute_psi_report: Callable[..., Any]
    evaluate_and_report: Callable[..., None]
    plot_curves_for_model: Callable[[Any, List[str], Dict[str, Any]], None]


def _create_ddp_barrier(dist_ctx: Any, ropt: Any):
    """Create a DDP barrier function for distributed training synchronization."""

    def _ddp_barrier(reason: str) -> None:
        if not getattr(dist_ctx, "is_distributed", False):
            return
        torch_mod = getattr(ropt, "torch", None)
        dist_mod = getattr(torch_mod, "distributed", None)
        if dist_mod is None:
            return
        try:
            if not getattr(dist_mod, "is_available", lambda: False)():
                return
            if not dist_mod.is_initialized():
                ddp_ok, _, _, _ = ropt.DistributedUtils.setup_ddp()
                if not ddp_ok or not dist_mod.is_initialized():
                    return
            dist_mod.barrier()
        except Exception as exc:
            print(f"[DDP] barrier failed during {reason}: {exc}", flush=True)
            raise

    return _ddp_barrier


def run_bayesopt_entry_training(
    args: Any,
    *,
    script_dir: Path,
    deps: BayesOptRunnerDeps,
    hooks: BayesOptRunnerHooks,
    training_context_from_env: Callable[[], Any],
) -> None:
    config_path, cfg = deps.resolve_and_load_config(
        args.config_json,
        script_dir,
        required_keys=["data_dir", "model_list",
                       "model_categories", "target", "weight"],
    )
    plot_requested = bool(args.plot_curves or cfg.get("plot_curves", False))
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    dist_ctx = training_context_from_env()
    dist_rank = dist_ctx.rank
    dist_active = dist_ctx.is_distributed
    is_main_process = dist_ctx.is_main_process
    _ddp_barrier = _create_ddp_barrier(dist_ctx, deps.ropt)

    data_dir, data_format, data_path_template, dtype_map = deps.resolve_data_config(
        cfg,
        config_path,
        create_data_dir=True,
    )
    runtime_cfg = deps.resolve_runtime_config(cfg)
    ddp_min_rows = runtime_cfg["ddp_min_rows"]
    bo_sample_limit = runtime_cfg["bo_sample_limit"]
    cache_predictions = runtime_cfg["cache_predictions"]
    prediction_cache_dir = runtime_cfg["prediction_cache_dir"]
    prediction_cache_format = runtime_cfg["prediction_cache_format"]
    report_cfg = deps.resolve_report_config(cfg)
    report_output_dir = report_cfg["report_output_dir"]
    report_group_cols = report_cfg["report_group_cols"]
    report_time_col = report_cfg["report_time_col"]
    report_time_freq = report_cfg["report_time_freq"]
    report_time_ascending = report_cfg["report_time_ascending"]
    psi_bins = report_cfg["psi_bins"]
    psi_strategy = report_cfg["psi_strategy"]
    psi_features = report_cfg["psi_features"]
    calibration_cfg = report_cfg["calibration_cfg"]
    threshold_cfg = report_cfg["threshold_cfg"]
    bootstrap_cfg = report_cfg["bootstrap_cfg"]
    register_model = report_cfg["register_model"]
    registry_path = report_cfg["registry_path"]
    registry_tags = report_cfg["registry_tags"]
    registry_status = report_cfg["registry_status"]
    data_fingerprint_max_bytes = report_cfg["data_fingerprint_max_bytes"]
    report_enabled = report_cfg["report_enabled"]

    split_cfg = deps.resolve_split_config(cfg)
    holdout_ratio = split_cfg["holdout_ratio"]
    val_ratio = split_cfg["val_ratio"]
    split_strategy = split_cfg["split_strategy"]
    split_group_col = split_cfg["split_group_col"]
    split_time_col = split_cfg["split_time_col"]
    split_time_ascending = split_cfg["split_time_ascending"]
    cv_strategy = split_cfg["cv_strategy"]
    cv_group_col = split_cfg["cv_group_col"]
    cv_time_col = split_cfg["cv_time_col"]
    cv_time_ascending = split_cfg["cv_time_ascending"]
    cv_splits = split_cfg["cv_splits"]
    ft_oof_folds = split_cfg["ft_oof_folds"]
    ft_oof_strategy = split_cfg["ft_oof_strategy"]
    ft_oof_shuffle = split_cfg["ft_oof_shuffle"]
    save_preprocess = runtime_cfg["save_preprocess"]
    preprocess_artifact_path = runtime_cfg["preprocess_artifact_path"]
    rand_seed = runtime_cfg["rand_seed"]
    epochs = runtime_cfg["epochs"]
    output_cfg = deps.resolve_output_dirs(
        cfg,
        config_path,
        output_override=args.output_dir,
    )
    output_dir = output_cfg["output_dir"]
    reuse_best_params = bool(
        args.reuse_best_params or runtime_cfg["reuse_best_params"])
    xgb_max_depth_max = runtime_cfg["xgb_max_depth_max"]
    xgb_n_estimators_max = runtime_cfg["xgb_n_estimators_max"]
    xgb_gpu_id = runtime_cfg["xgb_gpu_id"]
    xgb_cleanup_per_fold = runtime_cfg["xgb_cleanup_per_fold"]
    xgb_cleanup_synchronize = runtime_cfg["xgb_cleanup_synchronize"]
    xgb_use_dmatrix = runtime_cfg["xgb_use_dmatrix"]
    ft_cleanup_per_fold = runtime_cfg["ft_cleanup_per_fold"]
    ft_cleanup_synchronize = runtime_cfg["ft_cleanup_synchronize"]
    resn_cleanup_per_fold = runtime_cfg["resn_cleanup_per_fold"]
    resn_cleanup_synchronize = runtime_cfg["resn_cleanup_synchronize"]
    gnn_cleanup_per_fold = runtime_cfg["gnn_cleanup_per_fold"]
    gnn_cleanup_synchronize = runtime_cfg["gnn_cleanup_synchronize"]
    optuna_cleanup_synchronize = runtime_cfg["optuna_cleanup_synchronize"]
    optuna_storage = runtime_cfg["optuna_storage"]
    optuna_study_prefix = runtime_cfg["optuna_study_prefix"]
    best_params_files = runtime_cfg["best_params_files"]
    plot_path_style = runtime_cfg["plot_path_style"]

    model_names = deps.build_model_names(
        cfg["model_list"], cfg["model_categories"])
    if not model_names:
        raise ValueError(
            "No model names generated from model_list/model_categories.")

    results: Dict[str, Any] = {}
    trained_keys_by_model: Dict[str, List[str]] = {}

    for model_name in model_names:
        data_path = deps.resolve_data_path(
            data_dir,
            model_name,
            data_format=data_format,
            path_template=data_path_template,
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Missing dataset: {data_path}")
        data_fingerprint = {"path": str(data_path)}
        if report_enabled and is_main_process:
            data_fingerprint = deps.fingerprint_file(
                data_path,
                max_bytes=data_fingerprint_max_bytes,
            )

        print(f"\n=== Processing model {model_name} ===")
        raw = deps.load_dataset(
            data_path,
            data_format=data_format,
            dtype_map=dtype_map,
            low_memory=False,
        )
        raw = deps.coerce_dataset_types(raw)

        train_df, test_df = deps.split_train_test(
            raw,
            holdout_ratio=holdout_ratio,
            strategy=split_strategy,
            group_col=split_group_col,
            time_col=split_time_col,
            time_ascending=split_time_ascending,
            rand_seed=rand_seed,
            reset_index_mode="time_group",
            ratio_label="holdout_ratio",
        )

        use_resn_dp = bool((args.use_resn_dp or cfg.get(
            "use_resn_data_parallel", False)) and cfg.get("use_gpu", True))
        use_ft_dp = bool((args.use_ft_dp or cfg.get(
            "use_ft_data_parallel", False)) and cfg.get("use_gpu", True))
        dataset_rows = len(raw)
        ddp_enabled = bool(
            dist_active
            and cfg.get("use_gpu", True)
            and (dataset_rows >= int(ddp_min_rows))
        )
        use_resn_ddp = (args.use_resn_ddp or cfg.get(
            "use_resn_ddp", False)) and ddp_enabled
        use_ft_ddp = (args.use_ft_ddp or cfg.get(
            "use_ft_ddp", False)) and ddp_enabled
        use_gnn_dp = bool((args.use_gnn_dp or cfg.get(
            "use_gnn_data_parallel", False)) and cfg.get("use_gpu", True))
        use_gnn_ddp = (args.use_gnn_ddp or cfg.get(
            "use_gnn_ddp", False)) and ddp_enabled
        gnn_use_ann = cfg.get("gnn_use_approx_knn", True)
        if args.gnn_no_ann:
            gnn_use_ann = False
        gnn_threshold = args.gnn_ann_threshold if args.gnn_ann_threshold is not None else cfg.get(
            "gnn_approx_knn_threshold", 50000)
        gnn_graph_cache = args.gnn_graph_cache or cfg.get("gnn_graph_cache")
        if isinstance(gnn_graph_cache, str) and gnn_graph_cache.strip():
            resolved_cache = deps.resolve_path(gnn_graph_cache, config_path.parent)
            if resolved_cache is not None:
                gnn_graph_cache = str(resolved_cache)
        gnn_max_gpu_nodes = args.gnn_max_gpu_nodes if args.gnn_max_gpu_nodes is not None else cfg.get(
            "gnn_max_gpu_knn_nodes", 200000)
        gnn_gpu_mem_ratio = args.gnn_gpu_mem_ratio if args.gnn_gpu_mem_ratio is not None else cfg.get(
            "gnn_knn_gpu_mem_ratio", 0.9)
        gnn_gpu_mem_overhead = args.gnn_gpu_mem_overhead if args.gnn_gpu_mem_overhead is not None else cfg.get(
            "gnn_knn_gpu_mem_overhead", 2.0)

        binary_target = cfg.get("binary_target") or cfg.get("binary_resp_nme")
        task_type = str(cfg.get("task_type", "regression"))
        feature_list = cfg.get("feature_list")
        categorical_features = cfg.get("categorical_features")
        use_gpu = bool(cfg.get("use_gpu", True))
        region_province_col = cfg.get("region_province_col")
        region_city_col = cfg.get("region_city_col")
        region_effect_alpha = cfg.get("region_effect_alpha")
        geo_feature_nmes = cfg.get("geo_feature_nmes")
        geo_token_hidden_dim = cfg.get("geo_token_hidden_dim")
        geo_token_layers = cfg.get("geo_token_layers")
        geo_token_dropout = cfg.get("geo_token_dropout")
        geo_token_k_neighbors = cfg.get("geo_token_k_neighbors")
        geo_token_learning_rate = cfg.get("geo_token_learning_rate")
        geo_token_epochs = cfg.get("geo_token_epochs")

        ft_role = args.ft_role or cfg.get("ft_role", "model")
        if args.ft_as_feature and args.ft_role is None:
            if str(cfg.get("ft_role", "model")) == "model":
                ft_role = "embedding"
        ft_feature_prefix = str(
            cfg.get("ft_feature_prefix", args.ft_feature_prefix))
        ft_num_numeric_tokens = cfg.get("ft_num_numeric_tokens")

        config_fields = getattr(
            deps.ropt.BayesOptConfig,
            "__dataclass_fields__",
            {},
        )
        allowed_config_keys = {
            key
            for key, spec in config_fields.items()
            if getattr(spec, "init", True)
        }
        config_payload = {
            k: v for k, v in cfg.items() if k in allowed_config_keys and v is not None
        }
        config_payload.update({
            k: v
            for k, v in runtime_cfg.items()
            if k in allowed_config_keys and v is not None
        })
        config_payload.update({
            k: v
            for k, v in split_cfg.items()
            if k in allowed_config_keys and v is not None
        })
        override_payload = {
            "model_nme": model_name,
            "resp_nme": cfg["target"],
            "weight_nme": cfg["weight"],
            "factor_nmes": feature_list,
            "task_type": task_type,
            "binary_resp_nme": binary_target,
            "cate_list": categorical_features,
            "prop_test": val_ratio,
            "rand_seed": rand_seed,
            "epochs": epochs,
            "use_gpu": use_gpu,
            "use_resn_data_parallel": use_resn_dp,
            "use_ft_data_parallel": use_ft_dp,
            "use_gnn_data_parallel": use_gnn_dp,
            "use_resn_ddp": use_resn_ddp,
            "use_ft_ddp": use_ft_ddp,
            "use_gnn_ddp": use_gnn_ddp,
            "output_dir": output_dir,
            "xgb_max_depth_max": xgb_max_depth_max,
            "xgb_n_estimators_max": xgb_n_estimators_max,
            "xgb_gpu_id": xgb_gpu_id,
            "xgb_cleanup_per_fold": xgb_cleanup_per_fold,
            "xgb_cleanup_synchronize": xgb_cleanup_synchronize,
            "xgb_use_dmatrix": xgb_use_dmatrix,
            "ft_cleanup_per_fold": ft_cleanup_per_fold,
            "ft_cleanup_synchronize": ft_cleanup_synchronize,
            "resn_cleanup_per_fold": resn_cleanup_per_fold,
            "resn_cleanup_synchronize": resn_cleanup_synchronize,
            "gnn_cleanup_per_fold": gnn_cleanup_per_fold,
            "gnn_cleanup_synchronize": gnn_cleanup_synchronize,
            "optuna_cleanup_synchronize": optuna_cleanup_synchronize,
            "resn_weight_decay": cfg.get("resn_weight_decay"),
            "final_ensemble": bool(cfg.get("final_ensemble", False)),
            "final_ensemble_k": int(cfg.get("final_ensemble_k", 3)),
            "final_refit": bool(cfg.get("final_refit", True)),
            "optuna_storage": optuna_storage,
            "optuna_study_prefix": optuna_study_prefix,
            "best_params_files": best_params_files,
            "gnn_use_approx_knn": gnn_use_ann,
            "gnn_approx_knn_threshold": gnn_threshold,
            "gnn_graph_cache": gnn_graph_cache,
            "gnn_max_gpu_knn_nodes": gnn_max_gpu_nodes,
            "gnn_knn_gpu_mem_ratio": gnn_gpu_mem_ratio,
            "gnn_knn_gpu_mem_overhead": gnn_gpu_mem_overhead,
            "region_province_col": region_province_col,
            "region_city_col": region_city_col,
            "region_effect_alpha": region_effect_alpha,
            "geo_feature_nmes": geo_feature_nmes,
            "geo_token_hidden_dim": geo_token_hidden_dim,
            "geo_token_layers": geo_token_layers,
            "geo_token_dropout": geo_token_dropout,
            "geo_token_k_neighbors": geo_token_k_neighbors,
            "geo_token_learning_rate": geo_token_learning_rate,
            "geo_token_epochs": geo_token_epochs,
            "ft_role": ft_role,
            "ft_feature_prefix": ft_feature_prefix,
            "ft_num_numeric_tokens": ft_num_numeric_tokens,
            "reuse_best_params": reuse_best_params,
            "bo_sample_limit": bo_sample_limit,
            "cache_predictions": cache_predictions,
            "prediction_cache_dir": prediction_cache_dir,
            "prediction_cache_format": prediction_cache_format,
            "cv_strategy": cv_strategy or split_strategy,
            "cv_group_col": cv_group_col or split_group_col,
            "cv_time_col": cv_time_col or split_time_col,
            "cv_time_ascending": cv_time_ascending,
            "cv_splits": cv_splits,
            "ft_oof_folds": ft_oof_folds,
            "ft_oof_strategy": ft_oof_strategy,
            "ft_oof_shuffle": ft_oof_shuffle,
            "save_preprocess": save_preprocess,
            "preprocess_artifact_path": preprocess_artifact_path,
            "plot_path_style": plot_path_style or "nested",
        }
        config_payload.update({
            k: v
            for k, v in override_payload.items()
            if k in allowed_config_keys and v is not None
        })
        config = deps.ropt.BayesOptConfig.from_flat_dict(config_payload)
        model = deps.ropt.BayesOptModel(train_df, test_df, config=config)

        if plot_requested:
            plot_cfg = cfg.get("plot", {})
            plot_enabled = bool(plot_cfg.get("enable", False))
            if plot_enabled and plot_cfg.get("pre_oneway", False) and plot_cfg.get("oneway", True):
                n_bins = int(plot_cfg.get("n_bins", 10))
                model.plot_oneway(n_bins=n_bins, plot_subdir="oneway/pre")

        if "all" in args.model_keys:
            requested_keys = ["glm", "xgb", "resn", "ft", "gnn"]
        else:
            requested_keys = args.model_keys
        requested_keys = deps.dedupe_preserve_order(requested_keys)

        if ft_role != "model":
            requested_keys = [k for k in requested_keys if k != "ft"]
            if not requested_keys:
                stack_keys = args.stack_model_keys or cfg.get(
                    "stack_model_keys")
                if stack_keys:
                    if "all" in stack_keys:
                        requested_keys = ["glm", "xgb", "resn", "gnn"]
                    else:
                        requested_keys = [k for k in stack_keys if k != "ft"]
                    requested_keys = deps.dedupe_preserve_order(requested_keys)
            if dist_active and ddp_enabled:
                ft_trainer = model.trainers.get("ft")
                if ft_trainer is None:
                    raise ValueError("FT trainer is not available.")
                ft_trainer_uses_ddp = bool(
                    getattr(ft_trainer, "enable_distributed_optuna", False))
                if not ft_trainer_uses_ddp:
                    raise ValueError(
                        "FT embedding under torchrun requires enabling FT DDP (use --use-ft-ddp or set use_ft_ddp=true)."
                    )
        missing = [key for key in requested_keys if key not in model.trainers]
        if missing:
            raise ValueError(
                f"Trainer(s) {missing} not available for {model_name}")

        executed_keys: List[str] = []
        if ft_role != "model":
            if dist_active and not ddp_enabled:
                _ddp_barrier("start_ft_embedding")
                if dist_rank != 0:
                    _ddp_barrier("finish_ft_embedding")
                    continue
            print(
                f"Optimizing ft as {ft_role} for {model_name} (max_evals={args.max_evals})")
            model.optimize_model("ft", max_evals=args.max_evals)
            model.trainers["ft"].save()
            if getattr(deps.ropt, "torch", None) is not None and deps.ropt.torch.cuda.is_available():
                deps.ropt.free_cuda()
            if dist_active and not ddp_enabled:
                _ddp_barrier("finish_ft_embedding")
        for key in requested_keys:
            trainer = model.trainers[key]
            trainer_uses_ddp = bool(
                getattr(trainer, "enable_distributed_optuna", False))
            if dist_active and not trainer_uses_ddp:
                if dist_rank != 0:
                    print(
                        f"[Rank {dist_rank}] Skip {model_name}/{key} because trainer is not DDP-enabled."
                    )
                _ddp_barrier(f"start_non_ddp_{model_name}_{key}")
                if dist_rank != 0:
                    _ddp_barrier(f"finish_non_ddp_{model_name}_{key}")
                    continue

            print(
                f"Optimizing {key} for {model_name} (max_evals={args.max_evals})")
            model.optimize_model(key, max_evals=args.max_evals)
            model.trainers[key].save()
            hooks.plot_loss_curve_for_trainer(model_name, model.trainers[key])
            if key in deps.pytorch_trainers:
                deps.ropt.free_cuda()
            if dist_active and not trainer_uses_ddp:
                _ddp_barrier(f"finish_non_ddp_{model_name}_{key}")
            executed_keys.append(key)

        if not executed_keys:
            continue

        results[model_name] = model
        trained_keys_by_model[model_name] = executed_keys
        if report_enabled and is_main_process:
            psi_report_df = hooks.compute_psi_report(
                model,
                features=psi_features,
                bins=psi_bins,
                strategy=str(psi_strategy),
            )
            for key in executed_keys:
                hooks.evaluate_and_report(
                    model,
                    model_name=model_name,
                    model_key=key,
                    cfg=cfg,
                    data_path=data_path,
                    data_fingerprint=data_fingerprint,
                    report_output_dir=report_output_dir,
                    report_group_cols=report_group_cols,
                    report_time_col=report_time_col,
                    report_time_freq=str(report_time_freq),
                    report_time_ascending=bool(report_time_ascending),
                    psi_report_df=psi_report_df,
                    calibration_cfg=calibration_cfg,
                    threshold_cfg=threshold_cfg,
                    bootstrap_cfg=bootstrap_cfg,
                    register_model=register_model,
                    registry_path=registry_path,
                    registry_tags=registry_tags,
                    registry_status=registry_status,
                    run_id=run_id,
                    config_sha=config_sha,
                )

    if not plot_requested:
        return

    for name, model in results.items():
        hooks.plot_curves_for_model(
            model,
            trained_keys_by_model.get(name, []),
            cfg,
        )


__all__ = [
    "BayesOptRunnerDeps",
    "BayesOptRunnerHooks",
    "run_bayesopt_entry_training",
]
