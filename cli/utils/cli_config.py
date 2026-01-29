from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from .cli_common import resolve_dir_path, resolve_path  # type: ignore
except Exception:  # pragma: no cover
    from cli_common import resolve_dir_path, resolve_path  # type: ignore


def resolve_config_path(raw: str, script_dir: Path) -> Path:
    candidate = Path(raw)
    if candidate.exists():
        return candidate.resolve()
    candidate2 = (script_dir / raw)
    if candidate2.exists():
        return candidate2.resolve()
    raise FileNotFoundError(
        f"Config file not found: {raw}. Tried: {Path(raw).resolve()} and {candidate2.resolve()}"
    )


def load_config_json(path: Path, required_keys: Sequence[str]) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in {path}: {missing}")
    return cfg


def set_env(env_overrides: Dict[str, Any]) -> None:
    """Apply environment variables from config.json.

    Notes (DDP/Optuna hang debugging):
    - You can add these keys into config.json's `env` to debug distributed hangs:
      - `TORCH_DISTRIBUTED_DEBUG=DETAIL`
      - `NCCL_DEBUG=INFO`
      - `BAYESOPT_DDP_BARRIER_DEBUG=1`
      - `BAYESOPT_DDP_BARRIER_TIMEOUT=300`
      - `BAYESOPT_CUDA_SYNC=1` (optional; can slow down)
      - `BAYESOPT_CUDA_IPC_COLLECT=1` (optional; can slow down)
    - This function uses `os.environ.setdefault`, so a value already set in the
      shell will take precedence over config.json.
    """
    for key, value in (env_overrides or {}).items():
        os.environ.setdefault(str(key), str(value))


def _looks_like_url(value: str) -> bool:
    value = str(value)
    return "://" in value


def normalize_config_paths(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Resolve relative paths against the config.json directory.

    Fields handled:
    - data_dir / output_dir / optuna_storage / gnn_graph_cache
    - best_params_files (dict: model_key -> path)
    """
    base_dir = config_path.parent
    out = dict(cfg)

    for key in ("data_dir", "output_dir", "gnn_graph_cache", "preprocess_artifact_path",
                "prediction_cache_dir", "report_output_dir", "registry_path"):
        if key in out and isinstance(out.get(key), str):
            resolved = resolve_path(out.get(key), base_dir)
            if resolved is not None:
                out[key] = str(resolved)

    storage = out.get("optuna_storage")
    if isinstance(storage, str) and storage.strip():
        if not _looks_like_url(storage):
            resolved = resolve_path(storage, base_dir)
            if resolved is not None:
                out["optuna_storage"] = str(resolved)

    best_files = out.get("best_params_files")
    if isinstance(best_files, dict):
        resolved_map: Dict[str, str] = {}
        for mk, path_str in best_files.items():
            if not isinstance(path_str, str):
                continue
            resolved = resolve_path(path_str, base_dir)
            resolved_map[str(mk)] = str(resolved) if resolved is not None else str(path_str)
        out["best_params_files"] = resolved_map

    return out


def resolve_dtype_map(value: Any, base_dir: Path) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        path = resolve_path(value, base_dir)
        if path is None or not path.exists():
            raise FileNotFoundError(f"dtype_map not found: {value}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"dtype_map must be a dict: {path}")
        return {str(k): v for k, v in payload.items()}
    raise ValueError("dtype_map must be a dict or JSON path.")


def resolve_data_config(
    cfg: Dict[str, Any],
    config_path: Path,
    *,
    create_data_dir: bool = False,
) -> Tuple[Path, str, Optional[str], Dict[str, Any]]:
    base_dir = config_path.parent
    data_dir = resolve_dir_path(cfg.get("data_dir"), base_dir, create=create_data_dir)
    if data_dir is None:
        raise ValueError("data_dir is required in config.json.")
    data_format = cfg.get("data_format", "csv")
    data_path_template = cfg.get("data_path_template")
    dtype_map = resolve_dtype_map(cfg.get("dtype_map"), base_dir)
    return data_dir, data_format, data_path_template, dtype_map


def add_config_json_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--config-json",
        required=True,
        help=help_text,
    )


def add_output_dir_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--output-dir",
        default=None,
        help=help_text,
    )


def resolve_model_path_value(
    value: Any,
    *,
    model_name: str,
    base_dir: Path,
    data_dir: Optional[Path] = None,
) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get(model_name)
    if value is None:
        return None
    path_str = str(value)
    try:
        path_str = path_str.format(model_name=model_name)
    except Exception:
        pass
    if data_dir is not None and not Path(path_str).is_absolute():
        candidate = data_dir / path_str
        if candidate.exists():
            return candidate.resolve()
    resolved = resolve_path(path_str, base_dir)
    if resolved is None:
        return None
    return resolved


def resolve_explain_save_root(value: Any, base_dir: Path) -> Optional[Path]:
    if not value:
        return None
    path_str = str(value)
    resolved = resolve_path(path_str, base_dir)
    return resolved if resolved is not None else Path(path_str)


def resolve_explain_save_dir(
    save_root: Optional[Path],
    *,
    result_dir: Optional[Any],
) -> Path:
    if save_root is not None:
        return Path(save_root)
    if result_dir is None:
        raise ValueError("result_dir is required when explain save_root is not set.")
    return Path(result_dir) / "explain"


def resolve_explain_output_overrides(
    explain_cfg: Dict[str, Any],
    *,
    model_name: str,
    base_dir: Path,
) -> Dict[str, Optional[Path]]:
    return {
        "model_dir": resolve_model_path_value(
            explain_cfg.get("model_dir"),
            model_name=model_name,
            base_dir=base_dir,
            data_dir=None,
        ),
        "result_dir": resolve_model_path_value(
            explain_cfg.get("result_dir") or explain_cfg.get("results_dir"),
            model_name=model_name,
            base_dir=base_dir,
            data_dir=None,
        ),
        "plot_dir": resolve_model_path_value(
            explain_cfg.get("plot_dir"),
            model_name=model_name,
            base_dir=base_dir,
            data_dir=None,
        ),
    }


def resolve_and_load_config(
    raw: str,
    script_dir: Path,
    required_keys: Sequence[str],
    *,
    apply_env: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    config_path = resolve_config_path(raw, script_dir)
    cfg = load_config_json(config_path, required_keys=required_keys)
    cfg = normalize_config_paths(cfg, config_path)
    if apply_env:
        set_env(cfg.get("env", {}))
    return config_path, cfg


def resolve_report_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        return [str(value)]

    report_output_dir = cfg.get("report_output_dir")
    report_group_cols = _as_list(cfg.get("report_group_cols"))
    if not report_group_cols:
        report_group_cols = None
    report_time_col = cfg.get("report_time_col")
    report_time_freq = cfg.get("report_time_freq", "M")
    report_time_ascending = bool(cfg.get("report_time_ascending", True))
    psi_bins = cfg.get("psi_bins", 10)
    psi_strategy = cfg.get("psi_strategy", "quantile")
    psi_features = _as_list(cfg.get("psi_features"))
    if not psi_features:
        psi_features = None
    calibration_cfg = cfg.get("calibration", {}) or {}
    threshold_cfg = cfg.get("threshold", {}) or {}
    bootstrap_cfg = cfg.get("bootstrap", {}) or {}
    register_model = bool(cfg.get("register_model", False))
    registry_path = cfg.get("registry_path")
    registry_tags = cfg.get("registry_tags", {})
    registry_status = cfg.get("registry_status", "candidate")
    data_fingerprint_max_bytes = int(
        cfg.get("data_fingerprint_max_bytes", 10_485_760))
    calibration_enabled = bool(
        calibration_cfg.get("enable", False) or calibration_cfg.get("method")
    )
    threshold_enabled = bool(
        threshold_cfg.get("enable", False)
        or threshold_cfg.get("value") is not None
        or threshold_cfg.get("metric")
    )
    bootstrap_enabled = bool(bootstrap_cfg.get("enable", False))
    report_enabled = any([
        bool(report_output_dir),
        register_model,
        bool(report_group_cols),
        bool(report_time_col),
        bool(psi_features),
        calibration_enabled,
        threshold_enabled,
        bootstrap_enabled,
    ])
    return {
        "report_output_dir": report_output_dir,
        "report_group_cols": report_group_cols,
        "report_time_col": report_time_col,
        "report_time_freq": report_time_freq,
        "report_time_ascending": report_time_ascending,
        "psi_bins": psi_bins,
        "psi_strategy": psi_strategy,
        "psi_features": psi_features,
        "calibration_cfg": calibration_cfg,
        "threshold_cfg": threshold_cfg,
        "bootstrap_cfg": bootstrap_cfg,
        "register_model": register_model,
        "registry_path": registry_path,
        "registry_tags": registry_tags,
        "registry_status": registry_status,
        "data_fingerprint_max_bytes": data_fingerprint_max_bytes,
        "report_enabled": report_enabled,
    }


def resolve_split_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    prop_test = cfg.get("prop_test", 0.25)
    holdout_ratio = cfg.get("holdout_ratio", prop_test)
    if holdout_ratio is None:
        holdout_ratio = prop_test
    val_ratio = cfg.get("val_ratio", prop_test)
    if val_ratio is None:
        val_ratio = prop_test
    split_strategy = str(cfg.get("split_strategy", "random")).strip().lower()
    split_group_col = cfg.get("split_group_col")
    split_time_col = cfg.get("split_time_col")
    split_time_ascending = bool(cfg.get("split_time_ascending", True))
    cv_strategy = cfg.get("cv_strategy")
    cv_group_col = cfg.get("cv_group_col")
    cv_time_col = cfg.get("cv_time_col")
    cv_time_ascending = cfg.get("cv_time_ascending", split_time_ascending)
    cv_splits = cfg.get("cv_splits")
    ft_oof_folds = cfg.get("ft_oof_folds")
    ft_oof_strategy = cfg.get("ft_oof_strategy")
    ft_oof_shuffle = cfg.get("ft_oof_shuffle", True)
    return {
        "prop_test": prop_test,
        "holdout_ratio": holdout_ratio,
        "val_ratio": val_ratio,
        "split_strategy": split_strategy,
        "split_group_col": split_group_col,
        "split_time_col": split_time_col,
        "split_time_ascending": split_time_ascending,
        "cv_strategy": cv_strategy,
        "cv_group_col": cv_group_col,
        "cv_time_col": cv_time_col,
        "cv_time_ascending": cv_time_ascending,
        "cv_splits": cv_splits,
        "ft_oof_folds": ft_oof_folds,
        "ft_oof_strategy": ft_oof_strategy,
        "ft_oof_shuffle": ft_oof_shuffle,
    }


def resolve_runtime_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "save_preprocess": bool(cfg.get("save_preprocess", False)),
        "preprocess_artifact_path": cfg.get("preprocess_artifact_path"),
        "rand_seed": cfg.get("rand_seed", 13),
        "epochs": cfg.get("epochs", 50),
        "plot_path_style": cfg.get("plot_path_style"),
        "reuse_best_params": bool(cfg.get("reuse_best_params", False)),
        "xgb_max_depth_max": int(cfg.get("xgb_max_depth_max", 25)),
        "xgb_n_estimators_max": int(cfg.get("xgb_n_estimators_max", 500)),
        "optuna_storage": cfg.get("optuna_storage"),
        "optuna_study_prefix": cfg.get("optuna_study_prefix"),
        "best_params_files": cfg.get("best_params_files"),
        "bo_sample_limit": cfg.get("bo_sample_limit"),
        "cache_predictions": bool(cfg.get("cache_predictions", False)),
        "prediction_cache_dir": cfg.get("prediction_cache_dir"),
        "prediction_cache_format": cfg.get("prediction_cache_format", "parquet"),
        "ddp_min_rows": cfg.get("ddp_min_rows", 50000),
    }


def resolve_output_dirs(
    cfg: Dict[str, Any],
    config_path: Path,
    *,
    output_override: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    output_root = resolve_dir_path(
        output_override or cfg.get("output_dir"),
        config_path.parent,
    )
    return {
        "output_dir": str(output_root) if output_root is not None else None,
    }
