"""Shared prediction-workflow helpers for plot/compare pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

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


def _load_split_frame(
    path_value: str,
    label: str,
    *,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    path_obj = Path(path_value).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"{label} not found: {path_obj}")
    frame = _read_frame(path_obj, required_columns=required_columns)
    frame = _drop_duplicate_columns(frame, label).reset_index(drop=True)
    frame.fillna(0, inplace=True)
    return frame


def _resolve_read_columns(
    path_obj: Path,
    required_columns: Optional[Sequence[str]],
) -> Optional[list[str]]:
    if not required_columns:
        return None
    required = []
    seen = set()
    for col in required_columns:
        name = str(col or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        required.append(name)
    if not required:
        return None

    if path_obj.suffix.lower() in {".csv"}:
        try:
            header = pd.read_csv(path_obj, nrows=0)
        except Exception:
            return None
        available = set(str(c) for c in header.columns)
        selected = [c for c in required if c in available]
        return selected or None
    return required


def _read_frame(
    path_obj: Path,
    *,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    usecols = _resolve_read_columns(path_obj, required_columns)
    suffix = path_obj.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path_obj, usecols=usecols, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path_obj, columns=usecols)
        except Exception:
            return pd.read_parquet(path_obj)
    if suffix in {".feather", ".ft"}:
        try:
            return pd.read_feather(path_obj, columns=usecols)
        except Exception:
            return pd.read_feather(path_obj)
    raise ValueError(f"Unsupported table format for {path_obj}")


def _read_csv_split_rows_by_index(
    path_obj: Path,
    *,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    required_columns: Optional[Sequence[str]] = None,
    chunksize: int = 200_000,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Read only split rows from a large CSV, preserving requested split order."""
    usecols = _resolve_read_columns(path_obj, required_columns)
    train_idx = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    test_idx = np.asarray(test_indices, dtype=np.int64).reshape(-1)
    train_pos: Dict[int, int] = {int(idx): pos for pos, idx in enumerate(train_idx.tolist())}
    test_pos: Dict[int, int] = {int(idx): pos for pos, idx in enumerate(test_idx.tolist())}

    train_chunks = []
    test_chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        path_obj,
        usecols=usecols,
        low_memory=True,
        chunksize=int(max(1, chunksize)),
    ):
        chunk = _drop_duplicate_columns(chunk, "embed_csv_chunk")
        row_index = np.asarray(chunk.index.to_numpy(), dtype=np.int64)
        total_rows += int(len(chunk))

        train_slot = np.asarray(
            [train_pos.get(int(i), -1) for i in row_index],
            dtype=np.int64,
        )
        train_mask = train_slot >= 0
        if np.any(train_mask):
            train_piece = chunk.loc[train_mask].copy()
            train_piece["_slot"] = train_slot[train_mask]
            train_chunks.append(train_piece)

        test_slot = np.asarray(
            [test_pos.get(int(i), -1) for i in row_index],
            dtype=np.int64,
        )
        test_mask = test_slot >= 0
        if np.any(test_mask):
            test_piece = chunk.loc[test_mask].copy()
            test_piece["_slot"] = test_slot[test_mask]
            test_chunks.append(test_piece)

    def _assemble(parts, *, expected_rows: int, split_label: str) -> pd.DataFrame:
        if expected_rows == 0:
            return pd.DataFrame()
        if not parts:
            raise ValueError(
                f"Failed to collect any rows for split {split_label!r} from {path_obj}."
            )
        merged = pd.concat(parts, axis=0, ignore_index=True)
        if "_slot" not in merged.columns:
            raise ValueError(f"Internal error: missing row slot marker for {split_label!r}.")
        if merged["_slot"].duplicated().any():
            raise ValueError(
                f"Duplicate rows detected while collecting split {split_label!r} from {path_obj}."
            )
        merged = merged.sort_values("_slot", kind="mergesort").drop(columns=["_slot"])
        merged = merged.reset_index(drop=True)
        if len(merged) != int(expected_rows):
            raise ValueError(
                f"Split row mismatch for {split_label!r}: expected={expected_rows}, got={len(merged)}"
            )
        return merged

    train_df = _assemble(train_chunks, expected_rows=len(train_idx), split_label="train")
    test_df = _assemble(test_chunks, expected_rows=len(test_idx), split_label="test")
    return train_df, test_df, total_rows


def _remap_split_indices_by_key(
    *,
    cache_data_path: Path,
    current_raw: pd.DataFrame,
    key_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if key_col not in current_raw.columns:
        raise ValueError(
            f"split_cache_key_col={key_col!r} is not present in current data."
        )
    cache_keys_df = _read_frame(cache_data_path, required_columns=[key_col])
    if key_col not in cache_keys_df.columns:
        raise ValueError(
            f"split_cache_key_col={key_col!r} is not present in cached data source: {cache_data_path}"
        )
    cache_keys = cache_keys_df[key_col]
    if len(cache_keys) == 0:
        raise ValueError(f"Cached data source is empty: {cache_data_path}")
    if np.any(train_idx >= len(cache_keys)) or np.any(test_idx >= len(cache_keys)):
        raise ValueError(
            "Cached split indices exceed cached data row bounds when remapping by key."
        )

    current_keys = current_raw[key_col]
    current_key_index = pd.Index(current_keys)
    if not current_key_index.is_unique:
        dup = current_key_index[current_key_index.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"Current data key column {key_col!r} contains duplicates; "
            f"cannot remap split uniquely. sample={dup}"
        )

    cache_key_index = pd.Index(cache_keys)
    if not cache_key_index.is_unique:
        dup = cache_key_index[cache_key_index.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"Cached data key column {key_col!r} contains duplicates; "
            f"cannot remap split uniquely. sample={dup}"
        )

    train_keys = cache_keys.iloc[train_idx]
    test_keys = cache_keys.iloc[test_idx]
    train_pos = current_key_index.get_indexer(train_keys)
    test_pos = current_key_index.get_indexer(test_keys)

    if np.any(train_pos < 0):
        missing = train_keys.iloc[np.where(train_pos < 0)[0][:5]].tolist()
        raise ValueError(
            f"Failed to remap cached train split by key {key_col!r}; "
            f"keys missing in current data. sample={missing}"
        )
    if np.any(test_pos < 0):
        missing = test_keys.iloc[np.where(test_pos < 0)[0][:5]].tolist()
        raise ValueError(
            f"Failed to remap cached valid split by key {key_col!r}; "
            f"keys missing in current data. sample={missing}"
        )

    return np.asarray(train_pos, dtype=np.int64), np.asarray(test_pos, dtype=np.int64)


def load_raw_splits(
    *,
    split_cfg: dict,
    data_cfg: dict,
    data_cfg_path: Path,
    model_name: str,
    train_data_path: Optional[str],
    test_data_path: Optional[str],
    required_columns: Optional[Sequence[str]] = None,
    data_path_resolver: Callable[[dict, Path, str], Path] = _resolve_data_path,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], bool]:
    """Load explicit train/test files or split from a raw dataset."""
    explicit_train_path = str(train_data_path or "").strip()
    explicit_test_path = str(test_data_path or "").strip()
    use_explicit_split = bool(explicit_train_path or explicit_test_path)
    split_cache_key_col_raw = str(split_cfg.get("split_cache_key_col", "") or "").strip()
    split_cache_key_col = split_cache_key_col_raw if split_cache_key_col_raw else None
    required_columns_local = list(required_columns) if required_columns is not None else None
    if split_cache_key_col and required_columns_local is not None:
        if split_cache_key_col not in required_columns_local:
            required_columns_local.append(split_cache_key_col)

    raw: Optional[pd.DataFrame] = None
    if use_explicit_split:
        train_raw = (
            _load_split_frame(
                explicit_train_path,
                "train_raw",
                required_columns=required_columns_local,
            )
            if explicit_train_path
            else pd.DataFrame()
        )
        test_raw = (
            _load_split_frame(
                explicit_test_path,
                "test_raw",
                required_columns=required_columns_local,
            )
            if explicit_test_path
            else pd.DataFrame()
        )
        if train_raw.empty and test_raw.empty:
            raise ValueError(
                "At least one of train_data_path/test_data_path must be provided."
            )
    else:
        raw_path = data_path_resolver(data_cfg, data_cfg_path, model_name)
        raw = _read_frame(raw_path, required_columns=required_columns_local)
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
        split_cache_validate_data_path = bool(
            split_cfg.get("split_cache_validate_data_path", True)
        )
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
            if not split_cache_validate_data_path:
                _log(
                    "[Split] split_cache_validate_data_path=false, "
                    "skip split-cache data_path consistency check."
                )
            validate_split_cache_metadata(
                cache_path=split_cache_path,
                cached_row_count=cached_row_count,
                current_row_count=len(raw),
                cache_meta=cache_meta,
                split_strategy=strategy_norm,
                holdout_ratio=float(holdout_ratio),
                rand_seed=rand_seed,
                data_path=str(raw_path) if split_cache_validate_data_path else None,
                rebuild_hint=True,
            )
            cache_data_path_raw = cache_meta.get("data_path")
            cache_data_path = None
            if cache_data_path_raw:
                try:
                    cache_data_path = Path(str(cache_data_path_raw)).resolve()
                except Exception:
                    cache_data_path = None

            if (
                split_cache_key_col is not None
                and cache_data_path is not None
                and cache_data_path.exists()
                and str(cache_data_path) != str(raw_path)
            ):
                _log(
                    f"[Split] Remapping cached split indices by key column "
                    f"{split_cache_key_col!r} from {cache_data_path} -> {raw_path}"
                )
                train_idx, test_idx = _remap_split_indices_by_key(
                    cache_data_path=cache_data_path,
                    current_raw=raw,
                    key_col=split_cache_key_col,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            elif (
                not split_cache_validate_data_path
                and cache_data_path is not None
                and str(cache_data_path) != str(raw_path)
                and split_cache_key_col is None
            ):
                _log(
                    "[Warn] Reusing split indices across different data_path without "
                    "split_cache_key_col remapping; row-order mismatch may distort validation plots."
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
    required_columns: Optional[Sequence[str]] = None,
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

    split_key_col = str(
        ft_cfg.get("split_cache_key_col")
        or embed_cfg.get("split_cache_key_col")
        or "_row_id"
    ).strip() or "_row_id"
    required_cols_local = list(required_columns) if required_columns is not None else None
    if split_key_col and required_cols_local is not None and split_key_col not in required_cols_local:
        required_cols_local.append(split_key_col)

    embed_train_path = resolve_model_scoped_path(
        embed_cfg.get("train_data_path"),
        model_name=model_name,
        base_dir=embed_cfg_path.parent,
    )
    embed_test_path = resolve_model_scoped_path(
        embed_cfg.get("test_data_path"),
        model_name=model_name,
        base_dir=embed_cfg_path.parent,
    )
    if embed_train_path is not None or embed_test_path is not None:
        if embed_train_path is None or embed_test_path is None:
            raise ValueError(
                "Both train_data_path and test_data_path must be set in embed_cfg for pre-split embedding data."
            )
        if not embed_train_path.exists():
            raise FileNotFoundError(f"Embedding train_data_path not found: {embed_train_path}")
        if not embed_test_path.exists():
            raise FileNotFoundError(f"Embedding test_data_path not found: {embed_test_path}")

        train_embed = _read_frame(embed_train_path, required_columns=required_cols_local)
        test_embed = _read_frame(embed_test_path, required_columns=required_cols_local)
        train_embed = _drop_duplicate_columns(train_embed, "embed_train").reset_index(drop=True)
        test_embed = _drop_duplicate_columns(test_embed, "embed_test").reset_index(drop=True)
        train_embed.fillna(0, inplace=True)
        test_embed.fillna(0, inplace=True)

        if split_key_col in train_embed.columns and split_key_col in train_raw.columns:
            key_index = pd.Index(train_embed[split_key_col])
            if not key_index.is_unique:
                raise ValueError(
                    f"Embedding split key {split_key_col!r} has duplicate values in train_data_path."
                )
            key_pos = key_index.get_indexer(train_raw[split_key_col])
            if np.any(key_pos < 0):
                raise ValueError(
                    f"Cannot align train embedding rows by key column {split_key_col!r}."
                )
            train_df = train_embed.iloc[key_pos].reset_index(drop=True).copy()
        else:
            if len(train_embed) != len(train_raw):
                raise ValueError(
                    f"Train row count mismatch: raw={len(train_raw)}, embed={len(train_embed)}"
                )
            train_df = train_embed.copy()

        if split_key_col in test_embed.columns and split_key_col in test_raw.columns:
            key_index = pd.Index(test_embed[split_key_col])
            if not key_index.is_unique:
                raise ValueError(
                    f"Embedding split key {split_key_col!r} has duplicate values in test_data_path."
                )
            key_pos = key_index.get_indexer(test_raw[split_key_col])
            if np.any(key_pos < 0):
                raise ValueError(
                    f"Cannot align test embedding rows by key column {split_key_col!r}."
                )
            test_df = test_embed.iloc[key_pos].reset_index(drop=True).copy()
        else:
            if len(test_embed) != len(test_raw):
                raise ValueError(
                    f"Test row count mismatch: raw={len(test_raw)}, embed={len(test_embed)}"
                )
            test_df = test_embed.copy()
        return train_df, test_df

    embed_path = embed_path_resolver(embed_cfg, embed_cfg_path, model_name)
    if embed_path.suffix.lower() in {".csv"}:
        if raw is None:
            raise ValueError("Raw data unavailable for embedding alignment.")
        train_df, test_df, embed_rows = _read_csv_split_rows_by_index(
            embed_path,
            train_indices=train_raw.index.to_numpy(),
            test_indices=test_raw.index.to_numpy(),
            required_columns=required_cols_local,
        )
        train_df = _drop_duplicate_columns(train_df, "embed_train").reset_index(drop=True)
        test_df = _drop_duplicate_columns(test_df, "embed_test").reset_index(drop=True)
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        if int(embed_rows) != len(raw):
            raise ValueError(
                f"Row count mismatch: raw={len(raw)}, embed={embed_rows}. "
                "Cannot align predictions to raw features."
            )
        return train_df, test_df

    embed_df = _read_frame(embed_path, required_columns=required_cols_local)
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
