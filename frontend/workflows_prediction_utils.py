"""Shared prediction-workflow helpers for plot/compare pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

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

from .workflows_common import (
    _discover_model_file,
    _drop_duplicate_columns,
    _load_ft_embedding_model,
    _resolve_data_path,
    _resolve_model_output_dir,
)

_logger = get_frontend_logger("ins_pricing.frontend.workflows_prediction_utils")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


def _split_train_test(*args, **kwargs):
    from ins_pricing.cli.utils.cli_common import split_train_test

    return split_train_test(*args, **kwargs)

DOUBLE_LIFT_FIGSIZE = (11, 5)


def _load_split_frame(path_value: str, label: str) -> pd.DataFrame:
    path_obj = Path(path_value).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"{label} not found: {path_obj}")
    frame = pd.read_csv(path_obj, low_memory=False)
    frame = _drop_duplicate_columns(frame, label).reset_index(drop=True)
    frame.fillna(0, inplace=True)
    return frame


def load_raw_splits(
    *,
    split_cfg: dict,
    data_cfg: dict,
    data_cfg_path: Path,
    model_name: str,
    train_data_path: Optional[str],
    test_data_path: Optional[str],
    data_path_resolver: Callable[[dict, Path, str], Path] = _resolve_data_path,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], bool]:
    """Load explicit train/test files or split from a raw dataset."""
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
            raise ValueError(
                "At least one of train_data_path/test_data_path must be provided."
            )
    else:
        raw_path = data_path_resolver(data_cfg, data_cfg_path, model_name)
        raw = pd.read_csv(raw_path, low_memory=False)
        raw = _drop_duplicate_columns(raw, "raw").reset_index(drop=True)
        raw.fillna(0, inplace=True)

        holdout_ratio = float(split_cfg.get("holdout_ratio", split_cfg.get("prop_test", 0.25)))
        split_strategy = split_cfg.get("split_strategy", "random")
        split_group_col = split_cfg.get("split_group_col")
        split_time_col = split_cfg.get("split_time_col")
        split_time_ascending = split_cfg.get("split_time_ascending", True)
        rand_seed = split_cfg.get("rand_seed", 13)
        split_cache_path = resolve_model_scoped_path(
            split_cfg.get("split_cache_path"),
            model_name=model_name,
            base_dir=data_cfg_path.parent,
        )
        split_cache_force_rebuild = bool(split_cfg.get("split_cache_force_rebuild", False))
        strategy_norm = str(split_strategy or "random").strip().lower() or "random"

        def _split_and_optionally_write_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
            train_local, test_local = _split_train_test(
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
                    cache_path=split_cache_path,
                )
                write_split_cache(
                    split_cache_path,
                    train_idx=train_idx_local,
                    test_idx=test_idx_local,
                    row_count=len(raw),
                    meta={
                        "split_strategy": strategy_norm,
                        "holdout_ratio": holdout_ratio,
                        "rand_seed": rand_seed,
                        "data_path": str(raw_path),
                    },
                )
                _log(f"[Split] Created split cache: {split_cache_path}")
            return train_local, test_local

        if (
            split_cache_path is not None
            and split_cache_path.exists()
            and not split_cache_force_rebuild
        ):
            train_idx, test_idx, cached_row_count, cache_meta = load_split_cache(split_cache_path)
            validate_split_cache_metadata(
                cache_path=split_cache_path,
                cached_row_count=cached_row_count,
                current_row_count=len(raw),
                cache_meta=cache_meta,
                split_strategy=strategy_norm,
                holdout_ratio=float(holdout_ratio),
                rand_seed=rand_seed,
                data_path=str(raw_path),
                rebuild_hint=True,
            )
            validate_split_indices(
                train_idx=train_idx,
                test_idx=test_idx,
                row_count=len(raw),
                cache_path=split_cache_path,
            )
            train_raw = raw.iloc[train_idx].copy()
            test_raw = raw.iloc[test_idx].copy()
            _log(f"[Split] Reused split cache: {split_cache_path}")
        else:
            train_raw, test_raw = _split_and_optionally_write_cache()

        train_raw = _drop_duplicate_columns(train_raw, "train_raw")
        test_raw = _drop_duplicate_columns(test_raw, "test_raw")

    return train_raw, test_raw, raw, use_explicit_split


def build_ft_embedding_frames(
    *,
    use_runtime_ft_embedding: bool,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    raw: Optional[pd.DataFrame],
    use_explicit_split: bool,
    model_name: str,
    ft_cfg: dict,
    ft_cfg_path: Path,
    search_roots: Sequence[Path],
    ft_model_path: Optional[str],
    embed_cfg: Optional[dict] = None,
    embed_cfg_path: Optional[Path] = None,
    embed_path_resolver: Callable[[dict, Path, str], Path] = _resolve_data_path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test frames with runtime FT embeddings or precomputed embeddings."""
    ft_output_dir = (ft_cfg_path.parent / ft_cfg["output_dir"]).resolve()
    ft_prefix = ft_cfg.get("ft_feature_prefix", "ft_emb")

    if use_runtime_ft_embedding:
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
                _log(f"[Info] Auto-discovered ft model: {discovered_ft_model}")
            else:
                ft_model_path_obj = default_ft_model_path
        if not ft_model_path_obj.exists():
            raise FileNotFoundError(f"FT model file not found: {ft_model_path_obj}")

        ft_model = _load_ft_embedding_model(ft_model_path_obj)
        device = "cpu"
        if hasattr(ft_model, "device"):
            ft_model.device = device
        if hasattr(ft_model, "to"):
            try:
                ft_model.to(device)
            except Exception:
                pass
        if hasattr(ft_model, "ft"):
            try:
                ft_model.ft.to(device)
            except Exception:
                pass

        emb_train = ft_model.predict(train_raw, return_embedding=True)
        emb_cols = [f"pred_{ft_prefix}_{i}" for i in range(emb_train.shape[1])]
        train_df = train_raw.copy()
        train_df[emb_cols] = emb_train

        emb_test = ft_model.predict(test_raw, return_embedding=True)
        test_df = test_raw.copy()
        test_df[emb_cols] = emb_test
        return train_df, test_df

    if use_explicit_split:
        return train_raw.copy(), test_raw.copy()

    if embed_cfg is None or embed_cfg_path is None:
        raise ValueError("embed_cfg and embed_cfg_path are required for precomputed embeddings.")

    embed_path = embed_path_resolver(embed_cfg, embed_cfg_path, model_name)
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
    return train_df, test_df


def resolve_model_output_override(
    *,
    model_name: str,
    model_key: str,
    model_path: Optional[str],
    search_roots: Sequence[Path],
    output_root: Path,
    label: str,
) -> Optional[Path]:
    """Resolve explicit model override, or auto-discover from search roots."""
    override = _resolve_model_output_dir(model_path, label)
    if override is None:
        discovered = _discover_model_file(
            model_name=model_name,
            model_key=model_key,
            search_roots=search_roots,
            output_roots=[output_root],
        )
        if discovered is not None:
            override = _resolve_model_output_dir(str(discovered), f"auto_{label}")
            _log(f"[Info] Auto-discovered {model_key} model: {discovered}")
    return override
