from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import torch.distributed as dist  # type: ignore
except Exception:  # pragma: no cover
    dist = None  # type: ignore

from ins_pricing.modelling.bayesopt.utils.distributed_utils import (
    DistributedUtils,
    TrainingUtils,
)


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


def _create_ddp_barrier(dist_ctx: Any):
    """Create a DDP barrier function for distributed training synchronization."""

    def _wait_with_deadline_fallback(
        wait_fn: Callable[[], Any],
        *,
        timeout_seconds: int,
        reason: str,
    ) -> None:
        done = threading.Event()
        holder: Dict[str, Any] = {"exc": None}

        def _target() -> None:
            try:
                wait_fn()
            except BaseException as exc:  # pragma: no cover - passthrough guard
                holder["exc"] = exc
            finally:
                done.set()

        threading.Thread(target=_target, daemon=True).start()
        if not done.wait(timeout=max(1, int(timeout_seconds))):
            raise TimeoutError(
                f"DDP barrier timed out after {timeout_seconds}s during {reason} "
                "(legacy wait() without timeout support)."
            )
        if holder["exc"] is not None:
            raise holder["exc"]

    def _ddp_barrier(reason: str) -> None:
        if not getattr(dist_ctx, "is_distributed", False):
            return
        if dist is None:
            return
        try:
            if not getattr(dist, "is_available", lambda: False)():
                return
            if not dist.is_initialized():
                ddp_ok, _, _, _ = DistributedUtils.setup_ddp()
                if not ddp_ok or not dist.is_initialized():
                    return
        except Exception as exc:
            print(f"[DDP] barrier pre-check failed during {reason}: {exc}", flush=True)
            raise

        timeout_seconds = int(os.environ.get("BAYESOPT_DDP_BARRIER_TIMEOUT", "1800"))
        debug_barrier = os.environ.get("BAYESOPT_DDP_BARRIER_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        rank = None
        world = None
        if debug_barrier:
            try:
                rank = dist.get_rank()
                world = dist.get_world_size()
                print(
                    f"[DDP] entering barrier({reason}) rank={rank}/{world}",
                    flush=True,
                )
            except Exception:
                debug_barrier = False
        try:
            timeout = timedelta(seconds=max(1, timeout_seconds))
            backend = None
            try:
                backend = dist.get_backend()
            except Exception:
                backend = None

            monitored = getattr(dist, "monitored_barrier", None)
            if backend == "gloo" and callable(monitored):
                monitored(timeout=timeout)
            else:
                work = None
                try:
                    work = dist.barrier(async_op=True)
                except TypeError:
                    work = None
                if work is not None:
                    wait = getattr(work, "wait", None)
                    if callable(wait):
                        try:
                            wait(timeout=timeout)
                        except TypeError:
                            _wait_with_deadline_fallback(
                                wait,
                                timeout_seconds=timeout_seconds,
                                reason=reason,
                            )
                    else:
                        dist.barrier()
                else:
                    dist.barrier()
            if debug_barrier:
                print(
                    f"[DDP] exit barrier({reason}) rank={rank}/{world}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[DDP] barrier failed during {reason}: {exc}", flush=True)
            raise

    return _ddp_barrier


def _free_cuda_safe() -> None:
    """Release CUDA memory using the shared training utility."""
    TrainingUtils.free_cuda()


def _stream_random_split_csv(
    data_path: Path,
    *,
    holdout_ratio: float,
    rand_seed: Optional[int],
    dtype_map: Dict[str, Any],
    usecols: Optional[List[str]],
    low_memory: bool,
    chunksize: int,
    coerce_dataset_types: Callable[[Any], Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stream CSV chunks and split rows by Bernoulli sampling to avoid full-file loading."""
    holdout_ratio = float(holdout_ratio)
    if not (0.0 < holdout_ratio < 1.0):
        raise ValueError(
            f"holdout_ratio must be in (0, 1) for streaming random split; got {holdout_ratio}."
    )
    chunk_size = max(1, int(chunksize))
    effective_seed = rand_seed if rand_seed is not None else 13
    if rand_seed is None:
        print(
            "[WARNING] rand_seed is not set; streaming CSV split defaults to seed=13 "
            "for reproducibility. Set rand_seed explicitly to control randomness.",
            flush=True,
        )
    rng = np.random.default_rng(effective_seed)
    read_kwargs: Dict[str, Any] = {"low_memory": bool(low_memory), "chunksize": chunk_size}
    if usecols:
        read_kwargs["usecols"] = list(usecols)
    if dtype_map:
        dtype_selected = dict(dtype_map)
        if usecols:
            allowed = set(usecols)
            dtype_selected = {k: v for k, v in dtype_selected.items() if k in allowed}
        if dtype_selected:
            read_kwargs["dtype"] = dtype_selected

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    total_rows = 0
    columns: Optional[pd.Index] = None

    for chunk in pd.read_csv(data_path, **read_kwargs):
        chunk = coerce_dataset_types(chunk)
        if columns is None:
            columns = chunk.columns
        n_chunk = int(len(chunk))
        if n_chunk == 0:
            continue
        total_rows += n_chunk
        mask_test = rng.random(n_chunk) < holdout_ratio
        if mask_test.any():
            test_parts.append(chunk.loc[mask_test])
        mask_train = ~mask_test
        if mask_train.any():
            train_parts.append(chunk.loc[mask_train])

    if total_rows == 0:
        raise ValueError(f"Dataset is empty: {data_path}")

    if columns is None:
        columns = pd.Index([])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=columns)
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Streaming split produced an empty train or test set. "
            "Adjust holdout_ratio or disable stream_split_csv."
        )
    return train_df, test_df, total_rows


def _dedupe_columns(columns: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for col in columns:
        if not isinstance(col, str):
            continue
        key = col.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _resolve_required_columns(
    cfg: Dict[str, Any],
    *,
    split_group_col: Optional[str],
    split_time_col: Optional[str],
    cv_group_col: Optional[str],
    cv_time_col: Optional[str],
    report_group_cols: Optional[List[str]],
    report_time_col: Optional[str],
) -> List[str]:
    cols: List[str] = []
    feature_list = cfg.get("feature_list") or []
    if isinstance(feature_list, (list, tuple)):
        cols.extend([c for c in feature_list if isinstance(c, str)])
    target = cfg.get("target")
    weight = cfg.get("weight")
    binary_target = cfg.get("binary_target") or cfg.get("binary_resp_nme")
    for col in [
        target,
        weight,
        binary_target,
        split_group_col,
        split_time_col,
        cv_group_col,
        cv_time_col,
        report_time_col,
    ]:
        if isinstance(col, str):
            cols.append(col)
    if report_group_cols:
        cols.extend([c for c in report_group_cols if isinstance(c, str)])
    return _dedupe_columns(cols)


def _resolve_model_scoped_path(
    value: Any,
    *,
    model_name: str,
    base_dir: Path,
    resolve_path: Callable[..., Optional[Path]],
) -> Optional[Path]:
    if value is None:
        return None
    candidate = value
    if isinstance(value, dict):
        candidate = value.get(model_name)
        if candidate is None:
            candidate = value.get("*")
    if candidate is None:
        return None
    candidate_str = str(candidate).strip()
    if not candidate_str:
        return None
    try:
        candidate_str = candidate_str.format(model_name=model_name)
    except Exception:
        pass
    return resolve_path(candidate_str, base_dir)


def _write_split_cache(
    path: Path,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    row_count: int,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload_meta = json.dumps(meta or {}, ensure_ascii=True, sort_keys=True)
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            train_idx=np.asarray(train_idx, dtype=np.int64),
            test_idx=np.asarray(test_idx, dtype=np.int64),
            row_count=np.asarray([int(row_count)], dtype=np.int64),
            meta_json=np.asarray([payload_meta]),
        )
    tmp_path.replace(path)


def _load_split_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        if "train_idx" not in payload or "test_idx" not in payload:
            raise ValueError(
                f"split cache missing required arrays train_idx/test_idx: {path}"
            )
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64).reshape(-1)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64).reshape(-1)
        row_count = -1
        if "row_count" in payload:
            row_count_arr = np.asarray(payload["row_count"], dtype=np.int64).reshape(-1)
            if row_count_arr.size > 0:
                row_count = int(row_count_arr[0])
        meta: Dict[str, Any] = {}
        if "meta_json" in payload:
            try:
                meta_arr = np.asarray(payload["meta_json"]).reshape(-1)
                if meta_arr.size > 0:
                    raw_json = str(meta_arr[0]).strip()
                    if raw_json:
                        meta = json.loads(raw_json)
            except Exception:
                meta = {}
    return train_idx, test_idx, row_count, meta


def _validate_split_indices(
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    row_count: int,
    cache_path: Path,
) -> None:
    if row_count <= 0:
        raise ValueError(f"Invalid row_count={row_count} for split cache: {cache_path}")
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"split cache has empty train/test indices: {cache_path}")
    if np.any(train_idx < 0) or np.any(test_idx < 0):
        raise ValueError(f"split cache contains negative row indices: {cache_path}")
    if np.any(train_idx >= row_count) or np.any(test_idx >= row_count):
        raise ValueError(
            f"split cache row indices exceed dataset bounds row_count={row_count}: {cache_path}"
        )
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise ValueError(f"split cache contains overlapping train/test rows: {cache_path}")


def _load_and_split_dataset(
    *,
    deps: BayesOptRunnerDeps,
    data_path: Path,
    data_format: str,
    dtype_map: Dict[str, Any],
    required_columns: Optional[List[str]],
    use_stream_split: bool,
    holdout_ratio: float,
    rand_seed: Optional[int],
    stream_split_chunksize: int,
    split_strategy: str,
    split_group_col: Optional[str],
    split_time_col: Optional[str],
    split_time_ascending: bool,
    train_data_path: Optional[Path],
    test_data_path: Optional[Path],
    split_cache_path: Optional[Path],
    split_cache_force_rebuild: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if (train_data_path is None) != (test_data_path is None):
        raise ValueError(
            "train_data_path and test_data_path must be set together when using pre-split data."
        )
    if train_data_path is not None and test_data_path is not None:
        if not train_data_path.exists():
            raise FileNotFoundError(f"Missing train_data_path: {train_data_path}")
        if not test_data_path.exists():
            raise FileNotFoundError(f"Missing test_data_path: {test_data_path}")
        train_df = deps.load_dataset(
            train_data_path,
            data_format="auto",
            dtype_map=dtype_map,
            usecols=required_columns,
            low_memory=False,
        )
        test_df = deps.load_dataset(
            test_data_path,
            data_format="auto",
            dtype_map=dtype_map,
            usecols=required_columns,
            low_memory=False,
        )
        train_df = deps.coerce_dataset_types(train_df)
        test_df = deps.coerce_dataset_types(test_df)
        return train_df, test_df, int(len(train_df) + len(test_df))

    strategy_norm = str(split_strategy or "random").strip().lower()
    should_reset_index = strategy_norm in {"time", "timeseries", "temporal", "group", "grouped"}

    if split_cache_path is not None and use_stream_split:
        use_stream_split = False

    if use_stream_split:
        train_df, test_df, dataset_rows = _stream_random_split_csv(
            data_path,
            holdout_ratio=holdout_ratio,
            rand_seed=rand_seed,
            dtype_map=dtype_map,
            usecols=required_columns,
            low_memory=False,
            chunksize=int(stream_split_chunksize),
            coerce_dataset_types=deps.coerce_dataset_types,
        )
        return train_df, test_df, dataset_rows

    raw = deps.load_dataset(
        data_path,
        data_format=data_format,
        dtype_map=dtype_map,
        usecols=required_columns,
        low_memory=False,
    )
    dataset_rows = int(len(raw))
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cache_loaded = False
    if (
        split_cache_path is not None
        and split_cache_path.exists()
        and not bool(split_cache_force_rebuild)
    ):
        train_idx, test_idx, cached_row_count, cache_meta = _load_split_cache(split_cache_path)
        if cached_row_count > 0 and cached_row_count != dataset_rows:
            raise ValueError(
                f"split cache row_count mismatch for {split_cache_path}: "
                f"cached={cached_row_count}, current={dataset_rows}. "
                "Set split_cache_force_rebuild=true to regenerate."
            )
        cache_strategy = cache_meta.get("split_strategy")
        cache_holdout = cache_meta.get("holdout_ratio")
        cache_seed = cache_meta.get("rand_seed")
        if cache_strategy is not None and str(cache_strategy).strip().lower() != strategy_norm:
            raise ValueError(
                f"split cache strategy mismatch for {split_cache_path}: "
                f"cached={cache_strategy}, current={strategy_norm}. "
                "Set split_cache_force_rebuild=true to regenerate."
            )
        if cache_holdout is not None:
            if not np.isclose(float(cache_holdout), float(holdout_ratio)):
                raise ValueError(
                    f"split cache holdout_ratio mismatch for {split_cache_path}: "
                    f"cached={cache_holdout}, current={holdout_ratio}. "
                    "Set split_cache_force_rebuild=true to regenerate."
                )
        if cache_seed is not None and cache_seed != rand_seed:
            raise ValueError(
                f"split cache rand_seed mismatch for {split_cache_path}: "
                f"cached={cache_seed}, current={rand_seed}. "
                "Set split_cache_force_rebuild=true to regenerate."
            )
        _validate_split_indices(
            train_idx=train_idx,
            test_idx=test_idx,
            row_count=dataset_rows,
            cache_path=split_cache_path,
        )
        train_df = raw.iloc[train_idx]
        test_df = raw.iloc[test_idx]
        cache_loaded = True
    else:
        train_df, test_df = deps.split_train_test(
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
        if split_cache_path is not None:
            train_idx = np.asarray(train_df.index.to_numpy(), dtype=np.int64).reshape(-1)
            test_idx = np.asarray(test_df.index.to_numpy(), dtype=np.int64).reshape(-1)
            _validate_split_indices(
                train_idx=train_idx,
                test_idx=test_idx,
                row_count=dataset_rows,
                cache_path=split_cache_path,
            )
            _write_split_cache(
                split_cache_path,
                train_idx=train_idx,
                test_idx=test_idx,
                row_count=dataset_rows,
                meta={
                    "split_strategy": strategy_norm,
                    "holdout_ratio": float(holdout_ratio),
                    "rand_seed": rand_seed,
                    "data_path": str(data_path),
                },
            )

    if should_reset_index:
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    train_df = deps.coerce_dataset_types(train_df)
    test_df = deps.coerce_dataset_types(test_df)
    if cache_loaded and (train_df.empty or test_df.empty):
        raise ValueError(
            f"split cache produced empty train/test frame for dataset: {split_cache_path}"
        )
    del raw
    return train_df, test_df, dataset_rows


def _resolve_ddp_preprocess_paths(
    *,
    output_dir: Optional[str],
    config_path: Path,
    model_name: str,
    config_sha: str,
) -> Tuple[Path, Path]:
    base_root = Path(output_dir) if output_dir else (config_path.parent / "Results")
    cache_root = base_root / "_ddp_preprocess_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    model_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(model_name))
    token_raw = f"{config_sha}|{model_name}"
    token = hashlib.sha256(token_raw.encode("utf-8")).hexdigest()[:16]
    meta_path = cache_root / f"{model_tag}_{token}.meta.json"
    bundle_path = cache_root / f"{model_tag}_{token}.bundle.pkl"
    return meta_path, bundle_path


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _read_json_with_retry(
    path: Path,
    *,
    retries: Optional[int] = None,
    delay_seconds: float = 0.1,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    delay = max(0.01, float(delay_seconds))
    if timeout_seconds is None:
        raw_timeout = os.environ.get("BAYESOPT_DDP_META_READ_TIMEOUT_SECONDS", "300")
        try:
            timeout_seconds = float(raw_timeout)
        except Exception:
            timeout_seconds = 300.0
    timeout_seconds = max(delay, float(timeout_seconds))
    if retries is None:
        retries = max(1, int(timeout_seconds / delay))
    else:
        retries = max(1, int(retries))

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                last_error = exc
        if attempt + 1 < retries:
            time.sleep(delay)
    waited_seconds = retries * delay
    if last_error is not None:
        raise RuntimeError(
            f"Failed to read JSON metadata after waiting {waited_seconds:.1f}s: "
            f"{path} ({last_error})"
        ) from last_error
    raise FileNotFoundError(
        f"Metadata file not found after waiting {waited_seconds:.1f}s: {path}"
    )


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
    _ddp_barrier = _create_ddp_barrier(dist_ctx)

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
    train_data_path_cfg = split_cfg["train_data_path"]
    test_data_path_cfg = split_cfg["test_data_path"]
    split_cache_path_cfg = split_cfg["split_cache_path"]
    split_cache_force_rebuild = split_cfg["split_cache_force_rebuild"]
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
    stream_split_csv = runtime_cfg["stream_split_csv"]
    stream_split_chunksize = runtime_cfg["stream_split_chunksize"]

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
        fmt_lower = str(data_format).strip().lower()
        is_csv_source = fmt_lower == "csv" or (
            fmt_lower == "auto" and data_path.suffix.lower() == ".csv"
        )
        resource_profile = str(
            cfg.get(
                "resource_profile",
                (cfg.get("env", {}) or {}).get("BAYESOPT_RESOURCE_PROFILE", "auto"),
            )
        ).strip().lower()
        auto_stream_split = bool(
            is_csv_source
            and str(split_strategy).strip().lower() in {"random"}
            and resource_profile == "memory_saving"
        )
        use_stream_split = bool(
            (stream_split_csv or auto_stream_split)
            and is_csv_source
            and str(split_strategy).strip().lower() in {"random"}
        )
        if auto_stream_split and not stream_split_csv:
            print(
                "[Data] Auto-enabling stream_split_csv for memory_saving profile.",
                flush=True,
            )
        required_columns = _resolve_required_columns(
            cfg,
            split_group_col=split_group_col,
            split_time_col=split_time_col,
            cv_group_col=cv_group_col,
            cv_time_col=cv_time_col,
            report_group_cols=report_group_cols,
            report_time_col=report_time_col,
        )
        if required_columns:
            print(
                f"[Data] projected column loading enabled: {len(required_columns)} columns",
                flush=True,
            )
        train_data_path = _resolve_model_scoped_path(
            train_data_path_cfg,
            model_name=model_name,
            base_dir=config_path.parent,
            resolve_path=deps.resolve_path,
        )
        test_data_path = _resolve_model_scoped_path(
            test_data_path_cfg,
            model_name=model_name,
            base_dir=config_path.parent,
            resolve_path=deps.resolve_path,
        )
        split_cache_path = _resolve_model_scoped_path(
            split_cache_path_cfg,
            model_name=model_name,
            base_dir=config_path.parent,
            resolve_path=deps.resolve_path,
        )
        if train_data_path is not None or test_data_path is not None:
            print(
                f"[Data] using pre-split data files: train={train_data_path}, test={test_data_path}",
                flush=True,
            )
        if split_cache_path is not None:
            cache_mode = "rebuild" if split_cache_force_rebuild else "reuse_or_create"
            print(
                f"[Data] split cache path enabled ({cache_mode}): {split_cache_path}",
                flush=True,
            )
            if use_stream_split:
                print(
                    "[Data] split_cache_path is set; disabling stream_split_csv for stable split reuse.",
                    flush=True,
                )

        ddp_requested_by_config = bool(
            args.use_resn_ddp
            or cfg.get("use_resn_ddp", False)
            or args.use_ft_ddp
            or cfg.get("use_ft_ddp", False)
        )
        skip_data_load_for_non_main = bool(
            dist_active
            and dist_rank != 0
            and (
                (not ddp_requested_by_config)
                or (not bool(cfg.get("use_gpu", True)))
            )
        )
        use_shared_preprocess = bool(
            dist_active
            and cfg.get("use_gpu", True)
            and ddp_requested_by_config
        )
        shared_meta_path: Optional[Path] = None
        shared_bundle_path: Optional[Path] = None
        train_df: Optional[pd.DataFrame] = None
        test_df: Optional[pd.DataFrame] = None
        dataset_rows = 0
        ddp_enabled: Optional[bool] = None

        if skip_data_load_for_non_main:
            ddp_enabled = False
            print(
                f"[Data][Rank {dist_rank}] Skip dataset load because no DDP trainer is requested.",
                flush=True,
            )

        if use_shared_preprocess:
            shared_meta_path, shared_bundle_path = _resolve_ddp_preprocess_paths(
                output_dir=output_dir,
                config_path=config_path,
                model_name=model_name,
                config_sha=config_sha,
            )
            if dist_rank == 0:
                if use_stream_split:
                    print(
                        f"[Data] streaming random split enabled "
                        f"(chunksize={int(stream_split_chunksize)}) for {data_path}",
                        flush=True,
                    )
                train_df, test_df, dataset_rows = _load_and_split_dataset(
                    deps=deps,
                    data_path=data_path,
                    data_format=data_format,
                    dtype_map=dtype_map,
                    required_columns=required_columns,
                    use_stream_split=use_stream_split,
                    holdout_ratio=holdout_ratio,
                    rand_seed=rand_seed,
                    stream_split_chunksize=int(stream_split_chunksize),
                    split_strategy=split_strategy,
                    split_group_col=split_group_col,
                    split_time_col=split_time_col,
                    split_time_ascending=split_time_ascending,
                    train_data_path=train_data_path,
                    test_data_path=test_data_path,
                    split_cache_path=split_cache_path,
                    split_cache_force_rebuild=bool(split_cache_force_rebuild),
                )
                ddp_enabled = bool(
                    dist_active
                    and cfg.get("use_gpu", True)
                    and (dataset_rows >= int(ddp_min_rows))
                )
                _write_json_atomic(
                    shared_meta_path,
                    {
                        "dataset_rows": int(dataset_rows),
                        "ddp_enabled": bool(ddp_enabled),
                        "bundle_path": str(shared_bundle_path),
                    },
                )
            _ddp_barrier(f"shared_preprocess_meta_ready_{model_name}")
            if dist_rank != 0:
                meta = _read_json_with_retry(shared_meta_path)
                dataset_rows = int(meta.get("dataset_rows", 0))
                ddp_enabled = bool(meta.get("ddp_enabled", False))
                if ddp_enabled:
                    print(
                        f"[Data][Rank {dist_rank}] Skip raw split/load; "
                        f"will load preprocess bundle: {shared_bundle_path}",
                        flush=True,
                    )
                else:
                    skip_data_load_for_non_main = True
                    print(
                        f"[Data][Rank {dist_rank}] Skip dataset load because DDP is disabled "
                        f"(dataset_rows={dataset_rows}, ddp_min_rows={int(ddp_min_rows)}).",
                        flush=True,
                    )
        else:
            if not skip_data_load_for_non_main:
                if use_stream_split:
                    print(
                        f"[Data] streaming random split enabled "
                        f"(chunksize={int(stream_split_chunksize)}) for {data_path}",
                        flush=True,
                    )
                train_df, test_df, dataset_rows = _load_and_split_dataset(
                    deps=deps,
                    data_path=data_path,
                    data_format=data_format,
                    dtype_map=dtype_map,
                    required_columns=required_columns,
                    use_stream_split=use_stream_split,
                    holdout_ratio=holdout_ratio,
                    rand_seed=rand_seed,
                    stream_split_chunksize=int(stream_split_chunksize),
                    split_strategy=split_strategy,
                    split_group_col=split_group_col,
                    split_time_col=split_time_col,
                    split_time_ascending=split_time_ascending,
                    train_data_path=train_data_path,
                    test_data_path=test_data_path,
                    split_cache_path=split_cache_path,
                    split_cache_force_rebuild=bool(split_cache_force_rebuild),
                )

        use_resn_dp = bool((args.use_resn_dp or cfg.get(
            "use_resn_data_parallel", False)) and cfg.get("use_gpu", True))
        use_ft_dp = bool((args.use_ft_dp or cfg.get(
            "use_ft_data_parallel", False)) and cfg.get("use_gpu", True))
        if ddp_enabled is None:
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
        save_preprocess_bundle = bool(
            use_shared_preprocess
            and ddp_enabled
            and dist_rank == 0
            and shared_bundle_path is not None
        )
        load_preprocess_bundle = bool(
            use_shared_preprocess
            and ddp_enabled
            and dist_rank != 0
            and shared_bundle_path is not None
        )
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
            "save_preprocess_bundle": save_preprocess_bundle,
            "load_preprocess_bundle": load_preprocess_bundle,
            "preprocess_bundle_path": (
                str(shared_bundle_path)
                if use_shared_preprocess and ddp_enabled and shared_bundle_path is not None
                else None
            ),
            "plot_path_style": plot_path_style or "nested",
        }
        config_payload.update({
            k: v
            for k, v in override_payload.items()
            if k in allowed_config_keys and v is not None
        })
        config = deps.ropt.BayesOptConfig.from_flat_dict(config_payload)

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
        known_model_keys = {"glm", "xgb", "resn", "ft", "gnn"}
        invalid_requested = [k for k in requested_keys if k not in known_model_keys]
        if invalid_requested:
            raise ValueError(
                f"Unknown requested model key(s): {invalid_requested}. "
                f"Valid keys: {sorted(known_model_keys)}"
            )

        # In torchrun non-DDP mode, non-main ranks only coordinate barriers and
        # skip all heavy dataset/model construction.
        if dist_active and not ddp_enabled and dist_rank != 0:
            if ft_role != "model":
                _ddp_barrier("start_ft_embedding")
                _ddp_barrier("finish_ft_embedding")
            for key in requested_keys:
                _ddp_barrier(f"start_non_ddp_{model_name}_{key}")
                _ddp_barrier(f"finish_non_ddp_{model_name}_{key}")
            continue

        if use_shared_preprocess and ddp_enabled and shared_bundle_path is not None and dist_active:
            if dist_rank == 0:
                model = deps.ropt.BayesOptModel(train_df, test_df, config=config)
                _ddp_barrier(f"shared_preprocess_bundle_ready_{model_name}")
            else:
                _ddp_barrier(f"shared_preprocess_bundle_ready_{model_name}")
                model = deps.ropt.BayesOptModel(None, None, config=config)
        else:
            model = deps.ropt.BayesOptModel(train_df, test_df, config=config)

        if plot_requested:
            plot_cfg = cfg.get("plot", {})
            plot_enabled = bool(plot_cfg.get("enable", False))
            if plot_enabled and plot_cfg.get("pre_oneway", False) and plot_cfg.get("oneway", True):
                n_bins = int(plot_cfg.get("n_bins", 10))
                model.plot_oneway(n_bins=n_bins, plot_subdir="oneway/pre")

        if ft_role != "model" and dist_active and ddp_enabled:
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
            _free_cuda_safe()
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
                _free_cuda_safe()
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
