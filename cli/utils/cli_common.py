"""CLI common utilities.

This module re-exports shared utilities from ins_pricing.utils and provides
CLI-specific functionality for configuration loading and train/test splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Re-export shared utilities for backward compatibility
from ins_pricing.utils.paths import (
    PLOT_MODEL_LABELS,
    PYTORCH_TRAINERS,
    dedupe_preserve_order,
    build_model_names,
    parse_model_pairs,
    resolve_path,
    resolve_dir_path,
    resolve_data_path,
    load_dataset,
    coerce_dataset_types,
    fingerprint_file,
)

__all__ = [
    # From shared utils
    "PLOT_MODEL_LABELS",
    "PYTORCH_TRAINERS",
    "dedupe_preserve_order",
    "build_model_names",
    "parse_model_pairs",
    "resolve_path",
    "resolve_dir_path",
    "resolve_data_path",
    "load_dataset",
    "coerce_dataset_types",
    "fingerprint_file",
    # CLI-specific
    "split_train_test",
    "resolve_config_path",
    "load_config_json",
    "set_env",
    "normalize_config_paths",
    "resolve_dtype_map",
    "resolve_data_config",
    "resolve_report_config",
    "resolve_split_config",
    "resolve_runtime_config",
    "resolve_output_dirs",
    "resolve_and_load_config",
]


# =============================================================================
# CLI-specific: Train/Test Splitting
# =============================================================================


def split_train_test(
    df: pd.DataFrame,
    *,
    holdout_ratio: float,
    strategy: str = "random",
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    time_ascending: bool = True,
    rand_seed: Optional[int] = None,
    reset_index_mode: str = "none",
    ratio_label: str = "holdout_ratio",
    include_strategy_in_ratio_error: bool = False,
    validate_ratio: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets.

    Args:
        df: Input DataFrame
        holdout_ratio: Proportion for test set (0.0-1.0)
        strategy: Split strategy ('random', 'time', 'group')
        group_col: Column name for group-based splitting
        time_col: Column name for time-based splitting
        time_ascending: Sort order for time-based splitting
        rand_seed: Random seed for reproducibility
        reset_index_mode: When to reset index ('none', 'always', 'time_group')
        ratio_label: Label for ratio in error messages
        include_strategy_in_ratio_error: Include strategy in error messages
        validate_ratio: Whether to validate ratio bounds

    Returns:
        Tuple of (train_df, test_df)
    """
    strategy = str(strategy or "random").strip().lower()
    holdout_ratio = float(holdout_ratio)

    if include_strategy_in_ratio_error and strategy in {
        "time",
        "timeseries",
        "temporal",
        "group",
        "grouped",
    }:
        strategy_label = (
            "time" if strategy in {"time", "timeseries", "temporal"} else "group"
        )
        ratio_error = (
            f"{ratio_label} must be in (0, 1) for {strategy_label} split; "
            f"got {holdout_ratio}."
        )
    else:
        ratio_error = f"{ratio_label} must be in (0, 1); got {holdout_ratio}."

    if strategy in {"time", "timeseries", "temporal"}:
        if not time_col:
            raise ValueError("split_time_col is required for time split_strategy.")
        if time_col not in df.columns:
            raise KeyError(f"split_time_col '{time_col}' not in dataset columns.")
        if validate_ratio and not (0.0 < holdout_ratio < 1.0):
            raise ValueError(ratio_error)
        ordered = df.sort_values(time_col, ascending=bool(time_ascending))
        cutoff = int(len(ordered) * (1.0 - holdout_ratio))
        if cutoff <= 0 or cutoff >= len(ordered):
            raise ValueError(
                f"{ratio_label}={holdout_ratio} leaves no data for train/test split."
            )
        train_df = ordered.iloc[:cutoff]
        test_df = ordered.iloc[cutoff:]
    elif strategy in {"group", "grouped"}:
        if not group_col:
            raise ValueError("split_group_col is required for group split_strategy.")
        if group_col not in df.columns:
            raise KeyError(f"split_group_col '{group_col}' not in dataset columns.")
        if validate_ratio and not (0.0 < holdout_ratio < 1.0):
            raise ValueError(ratio_error)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=holdout_ratio,
            random_state=rand_seed,
        )
        train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
    else:
        train_df, test_df = train_test_split(
            df, test_size=holdout_ratio, random_state=rand_seed
        )

    if reset_index_mode == "always" or (
        reset_index_mode == "time_group"
        and strategy in {"time", "timeseries", "temporal", "group", "grouped"}
    ):
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, test_df


# =============================================================================
# CLI-specific: Configuration Loading (delegated to cli_config)
# =============================================================================


def _load_cli_config():
    """Load the cli_config module."""
    try:
        from . import cli_config as _cli_config
    except Exception:
        import cli_config as _cli_config
    return _cli_config


def resolve_config_path(raw: str, script_dir: Path) -> Path:
    """Resolve a configuration file path."""
    return _load_cli_config().resolve_config_path(raw, script_dir)


def load_config_json(path: Path, required_keys: Sequence[str]) -> Dict[str, Any]:
    """Load and validate a JSON configuration file."""
    return _load_cli_config().load_config_json(path, required_keys)


def set_env(env_overrides: Dict[str, Any]) -> None:
    """Set environment variables from configuration."""
    _load_cli_config().set_env(env_overrides)


def normalize_config_paths(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Normalize paths in configuration relative to config file location."""
    return _load_cli_config().normalize_config_paths(cfg, config_path)


def resolve_dtype_map(value: Any, base_dir: Path) -> Dict[str, Any]:
    """Resolve dtype mapping from configuration."""
    return _load_cli_config().resolve_dtype_map(value, base_dir)


def resolve_data_config(
    cfg: Dict[str, Any],
    config_path: Path,
    *,
    create_data_dir: bool = False,
) -> Tuple[Path, str, Optional[str], Dict[str, Any]]:
    """Resolve data configuration from config file."""
    return _load_cli_config().resolve_data_config(
        cfg,
        config_path,
        create_data_dir=create_data_dir,
    )


def resolve_report_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve report configuration."""
    return _load_cli_config().resolve_report_config(cfg)


def resolve_split_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve train/test split configuration."""
    return _load_cli_config().resolve_split_config(cfg)


def resolve_runtime_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve runtime configuration."""
    return _load_cli_config().resolve_runtime_config(cfg)


def resolve_output_dirs(
    cfg: Dict[str, Any],
    config_path: Path,
    *,
    output_override: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Resolve output directory paths."""
    return _load_cli_config().resolve_output_dirs(
        cfg,
        config_path,
        output_override=output_override,
    )


def resolve_and_load_config(
    raw: str,
    script_dir: Path,
    required_keys: Sequence[str],
    *,
    apply_env: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Resolve and load a configuration file."""
    return _load_cli_config().resolve_and_load_config(
        raw,
        script_dir,
        required_keys,
        apply_env=apply_env,
    )
