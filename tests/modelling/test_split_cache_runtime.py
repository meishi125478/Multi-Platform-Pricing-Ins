from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ins_pricing.cli.utils.cli_config import resolve_runtime_config, resolve_split_config
from ins_pricing.modelling.bayesopt.runtime.entry_runner_training import (
    _build_runtime_override_payload,
    _load_and_split_dataset,
)
from ins_pricing.utils.paths import load_dataset


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def _random_split(df: pd.DataFrame, **kwargs):
    holdout_ratio = float(kwargs.get("holdout_ratio", 0.25))
    rand_seed = kwargs.get("rand_seed", 13)
    rng = np.random.default_rng(rand_seed if rand_seed is not None else 13)
    indices = np.arange(len(df), dtype=np.int64)
    rng.shuffle(indices)
    n_test = max(1, int(round(len(df) * holdout_ratio)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return df.iloc[train_idx], df.iloc[test_idx]


def _deps(split_fn):
    return SimpleNamespace(
        load_dataset=load_dataset,
        coerce_dataset_types=_identity,
        split_train_test=split_fn,
    )


def test_load_and_split_dataset_can_use_manual_train_test_paths(tmp_path):
    train_src = pd.DataFrame({"id": [1, 2, 3], "x": [10.0, 20.0, 30.0]})
    test_src = pd.DataFrame({"id": [4, 5], "x": [40.0, 50.0]})
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_src.to_csv(train_path, index=False)
    test_src.to_csv(test_path, index=False)

    def _fail_split(*_args, **_kwargs):
        raise AssertionError("split_train_test should not be called when manual split files are set")

    train_df, test_df, dataset_rows = _load_and_split_dataset(
        deps=_deps(_fail_split),
        data_path=tmp_path / "unused.csv",
        data_format="csv",
        dtype_map={},
        required_columns=None,
        use_stream_split=False,
        holdout_ratio=0.25,
        rand_seed=13,
        stream_split_chunksize=128,
        split_strategy="random",
        split_group_col=None,
        split_time_col=None,
        split_time_ascending=True,
        train_data_path=train_path,
        test_data_path=test_path,
        split_cache_path=None,
        split_cache_force_rebuild=False,
    )

    assert dataset_rows == 5
    assert train_df.equals(train_src)
    assert test_df.equals(test_src)


def test_load_and_split_dataset_split_cache_is_reused_without_resplitting(tmp_path):
    raw = pd.DataFrame({"id": np.arange(20), "x": np.arange(20) * 1.0})
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    split_cache_path = tmp_path / "split_cache.npz"

    train_1, test_1, total_1 = _load_and_split_dataset(
        deps=_deps(_random_split),
        data_path=raw_path,
        data_format="csv",
        dtype_map={},
        required_columns=None,
        use_stream_split=False,
        holdout_ratio=0.3,
        rand_seed=13,
        stream_split_chunksize=128,
        split_strategy="random",
        split_group_col=None,
        split_time_col=None,
        split_time_ascending=True,
        train_data_path=None,
        test_data_path=None,
        split_cache_path=split_cache_path,
        split_cache_force_rebuild=False,
    )

    assert split_cache_path.exists()
    assert total_1 == len(raw)

    def _fail_split(*_args, **_kwargs):
        raise AssertionError("split_train_test should not be called when split cache exists")

    train_2, test_2, total_2 = _load_and_split_dataset(
        deps=_deps(_fail_split),
        data_path=raw_path,
        data_format="csv",
        dtype_map={},
        required_columns=None,
        use_stream_split=False,
        holdout_ratio=0.3,
        rand_seed=13,
        stream_split_chunksize=128,
        split_strategy="random",
        split_group_col=None,
        split_time_col=None,
        split_time_ascending=True,
        train_data_path=None,
        test_data_path=None,
        split_cache_path=split_cache_path,
        split_cache_force_rebuild=False,
    )

    assert total_2 == len(raw)
    assert set(train_1["id"].tolist()) == set(train_2["id"].tolist())
    assert set(test_1["id"].tolist()) == set(test_2["id"].tolist())


def test_load_and_split_dataset_split_cache_row_mismatch_raises(tmp_path):
    raw_1 = pd.DataFrame({"id": np.arange(10), "x": np.arange(10) * 1.0})
    raw_2 = pd.DataFrame({"id": np.arange(12), "x": np.arange(12) * 1.0})
    raw_path_1 = tmp_path / "raw_1.csv"
    raw_path_2 = tmp_path / "raw_2.csv"
    raw_1.to_csv(raw_path_1, index=False)
    raw_2.to_csv(raw_path_2, index=False)
    split_cache_path = tmp_path / "split_cache.npz"

    _load_and_split_dataset(
        deps=_deps(_random_split),
        data_path=raw_path_1,
        data_format="csv",
        dtype_map={},
        required_columns=None,
        use_stream_split=False,
        holdout_ratio=0.2,
        rand_seed=13,
        stream_split_chunksize=128,
        split_strategy="random",
        split_group_col=None,
        split_time_col=None,
        split_time_ascending=True,
        train_data_path=None,
        test_data_path=None,
        split_cache_path=split_cache_path,
        split_cache_force_rebuild=False,
    )

    with pytest.raises(ValueError, match="row_count mismatch"):
        _load_and_split_dataset(
            deps=_deps(_random_split),
            data_path=raw_path_2,
            data_format="csv",
            dtype_map={},
            required_columns=None,
            use_stream_split=False,
            holdout_ratio=0.2,
            rand_seed=13,
            stream_split_chunksize=128,
            split_strategy="random",
            split_group_col=None,
            split_time_col=None,
            split_time_ascending=True,
            train_data_path=None,
            test_data_path=None,
            split_cache_path=split_cache_path,
            split_cache_force_rebuild=False,
        )


def test_load_and_split_dataset_split_cache_reuse_across_aligned_datasets(tmp_path):
    raw_base = pd.DataFrame({"id": np.arange(15), "x": np.arange(15) * 1.0})
    raw_aug = raw_base.copy()
    raw_aug["x2"] = raw_aug["x"] * 2.0
    base_path = tmp_path / "base.csv"
    aug_path = tmp_path / "aug.csv"
    raw_base.to_csv(base_path, index=False)
    raw_aug.to_csv(aug_path, index=False)
    split_cache_path = tmp_path / "split_cache.npz"

    _load_and_split_dataset(
        deps=_deps(_random_split),
        data_path=base_path,
        data_format="csv",
        dtype_map={},
        required_columns=None,
        use_stream_split=False,
        holdout_ratio=0.2,
        rand_seed=13,
        stream_split_chunksize=128,
        split_strategy="random",
        split_group_col=None,
        split_time_col=None,
        split_time_ascending=True,
        train_data_path=None,
        test_data_path=None,
        split_cache_path=split_cache_path,
        split_cache_force_rebuild=False,
    )

    def _fail_split(*_args, **_kwargs):
        raise AssertionError("split_train_test should not be called when reusing split cache")

    train_df, test_df, _ = _load_and_split_dataset(
        deps=_deps(_fail_split),
        data_path=aug_path,
        data_format="csv",
        dtype_map={},
        required_columns=None,
        use_stream_split=False,
        holdout_ratio=0.2,
        rand_seed=13,
        stream_split_chunksize=128,
        split_strategy="random",
        split_group_col=None,
        split_time_col=None,
        split_time_ascending=True,
        train_data_path=None,
        test_data_path=None,
        split_cache_path=split_cache_path,
        split_cache_force_rebuild=False,
    )

    assert "x2" in train_df.columns
    assert "x2" in test_df.columns


def test_resolve_split_and_runtime_config_expose_new_fields():
    split_cfg = resolve_split_config(
        {
            "prop_test": 0.2,
            "split_cache_path": "./cache/split.npz",
            "split_cache_force_rebuild": True,
            "train_data_path": "./cache/train.csv",
            "test_data_path": "./cache/test.csv",
        }
    )
    assert split_cfg["split_cache_path"] == "./cache/split.npz"
    assert split_cfg["split_cache_force_rebuild"] is True
    assert split_cfg["train_data_path"] == "./cache/train.csv"
    assert split_cfg["test_data_path"] == "./cache/test.csv"

    runtime_cfg = resolve_runtime_config(
        {
            "rand_seed": None,
            "ft_search_space": {"n_heads": {"type": "categorical", "choices": [1, 2]}},
            "ft_unsupervised_search_space": {"d_model": {"type": "int", "low": 16, "high": 32}},
            "resn_search_space": {"hidden_dim": {"type": "int", "low": 16, "high": 64}},
            "xgb_search_space": {"max_depth": {"type": "int", "low": 3, "high": 8}},
        }
    )
    assert runtime_cfg["rand_seed"] == 13
    assert isinstance(runtime_cfg["ft_search_space"], dict)
    assert isinstance(runtime_cfg["ft_unsupervised_search_space"], dict)
    assert isinstance(runtime_cfg["resn_search_space"], dict)
    assert isinstance(runtime_cfg["xgb_search_space"], dict)

    payload = _build_runtime_override_payload(
        cfg={},
        runtime_cfg=runtime_cfg,
        parallel_flags={
            "use_gpu": False,
            "use_resn_dp": False,
            "use_ft_dp": False,
            "use_gnn_dp": False,
            "use_resn_ddp": False,
            "use_ft_ddp": False,
            "gnn_use_ann": True,
            "gnn_threshold": 50000,
            "gnn_graph_cache": None,
            "gnn_max_gpu_nodes": None,
            "gnn_gpu_mem_ratio": 0.9,
            "gnn_gpu_mem_overhead": 2.0,
        },
        output_dir=None,
        reuse_best_params=False,
        preprocess_bundle_include_raw=False,
        keep_unscaled_oht=False,
    )
    assert isinstance(payload["ft_search_space"], dict)
    assert isinstance(payload["ft_unsupervised_search_space"], dict)
    assert isinstance(payload["resn_search_space"], dict)
    assert isinstance(payload["xgb_search_space"], dict)
    assert payload["preprocess_bundle_include_raw"] is False
    assert payload["keep_unscaled_oht"] is False
