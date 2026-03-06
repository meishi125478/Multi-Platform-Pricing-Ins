"""Shared prediction-workflow helpers for plot/compare pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd

from ins_pricing.cli.utils.cli_common import split_train_test

from .workflows_common import (
    _discover_model_file,
    _drop_duplicate_columns,
    _load_ft_embedding_model,
    _resolve_data_path,
    _resolve_model_output_dir,
)

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

        ft_model = _load_ft_embedding_model(ft_model_path_obj)
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
            print(f"[Info] Auto-discovered {model_key} model: {discovered}")
    return override
