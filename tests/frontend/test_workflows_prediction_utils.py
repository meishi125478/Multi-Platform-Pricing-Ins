from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ins_pricing.frontend.workflows_prediction_utils as workflows_prediction_utils
from ins_pricing.frontend.workflows_prediction_utils import (
    build_ft_embedding_frames,
    load_raw_splits,
    resolve_model_output_override,
)
from ins_pricing.split_cache import write_split_cache


def test_load_raw_splits_with_explicit_files(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(train_path, index=False)
    pd.DataFrame({"x": [3]}).to_csv(test_path, index=False)

    train_raw, test_raw, raw, use_explicit = load_raw_splits(
        split_cfg={},
        data_cfg={},
        data_cfg_path=tmp_path / "cfg.json",
        model_name="demo",
        train_data_path=str(train_path),
        test_data_path=str(test_path),
    )

    assert use_explicit is True
    assert raw is None
    assert len(train_raw) == 2
    assert len(test_raw) == 1


def test_build_ft_embedding_frames_precomputed_alignment(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}", encoding="utf-8")
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)
    embed_df = pd.DataFrame({"pred_ft_emb_0": [0.1, 0.2, 0.3, 0.4]})
    embed_df.to_csv(embed_dir / "demo.csv", index=False)

    raw = pd.DataFrame({"a": [10, 20, 30, 40]})
    train_raw = raw.iloc[[0, 2]].copy()
    test_raw = raw.iloc[[1, 3]].copy()

    train_df, test_df = build_ft_embedding_frames(
        use_runtime_ft_embedding=False,
        train_raw=train_raw,
        test_raw=test_raw,
        raw=raw,
        use_explicit_split=False,
        model_name="demo",
        ft_cfg={"output_dir": "./results", "ft_feature_prefix": "ft_emb"},
        ft_cfg_path=cfg_path,
        search_roots=[tmp_path],
        ft_model_path=None,
        embed_cfg={"data_dir": "./embed", "data_format": "csv"},
        embed_cfg_path=cfg_path,
    )

    assert train_df["pred_ft_emb_0"].tolist() == [0.1, 0.3]
    assert test_df["pred_ft_emb_0"].tolist() == [0.2, 0.4]


def test_build_ft_embedding_frames_precomputed_csv_chunk_alignment_preserves_split_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}", encoding="utf-8")
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)

    embed_df = pd.DataFrame(
        {
            "_row_id": [100, 101, 102, 103, 104, 105],
            "pred_ft_emb_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    embed_df.to_csv(embed_dir / "demo.csv", index=False)

    raw = pd.DataFrame(
        {
            "_row_id": [100, 101, 102, 103, 104, 105],
            "f1": [10, 20, 30, 40, 50, 60],
        }
    )
    train_raw = raw.iloc[[5, 1, 3]].copy()
    test_raw = raw.iloc[[4, 2, 0]].copy()

    def _should_not_read_full_frame(*args, **kwargs):
        raise AssertionError("_read_frame should not be used for single-file CSV embedding alignment.")

    monkeypatch.setattr(workflows_prediction_utils, "_read_frame", _should_not_read_full_frame)

    train_df, test_df = build_ft_embedding_frames(
        use_runtime_ft_embedding=False,
        train_raw=train_raw,
        test_raw=test_raw,
        raw=raw,
        use_explicit_split=False,
        model_name="demo",
        ft_cfg={"output_dir": "./results", "ft_feature_prefix": "ft_emb"},
        ft_cfg_path=cfg_path,
        search_roots=[tmp_path],
        ft_model_path=None,
        embed_cfg={"data_dir": "./embed", "data_format": "csv"},
        embed_cfg_path=cfg_path,
    )

    assert train_df["_row_id"].tolist() == [105, 101, 103]
    assert train_df["pred_ft_emb_0"].tolist() == [0.6, 0.2, 0.4]
    assert test_df["_row_id"].tolist() == [104, 102, 100]
    assert test_df["pred_ft_emb_0"].tolist() == [0.5, 0.3, 0.1]


def test_resolve_model_output_override_with_explicit_file(tmp_path: Path) -> None:
    model_name = "demo_model"
    model_dir = tmp_path / "results" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"01_{model_name}_Xgboost.pkl"
    model_file.write_bytes(b"placeholder")

    resolved = resolve_model_output_override(
        model_name=model_name,
        model_key="xgb",
        model_path=str(model_file),
        search_roots=[tmp_path],
        output_root=tmp_path / "results",
        label="xgb_model_path",
    )

    assert resolved == (tmp_path / "results").resolve()


def test_load_raw_splits_can_reuse_cache_when_data_path_validation_disabled(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config_plot.json"
    cfg_path.write_text("{}", encoding="utf-8")

    data_new_dir = tmp_path / "DataFTEmbed"
    data_new_dir.mkdir(parents=True, exist_ok=True)
    data_new_path = data_new_dir / "demo.csv"
    pd.DataFrame({"x": [10, 20, 30, 40]}).to_csv(data_new_path, index=False)

    data_old_dir = tmp_path / "Data"
    data_old_dir.mkdir(parents=True, exist_ok=True)
    data_old_path = data_old_dir / "demo.csv"
    pd.DataFrame({"x": [1, 2, 3, 4]}).to_csv(data_old_path, index=False)

    cache_dir = tmp_path / "splits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "demo_holdout_split.npz"
    write_split_cache(
        cache_path,
        train_idx=np.asarray([0, 2], dtype=np.int64),
        test_idx=np.asarray([1, 3], dtype=np.int64),
        row_count=4,
        meta={
            "split_strategy": "random",
            "holdout_ratio": 0.25,
            "rand_seed": 13,
            "data_path": str(data_old_path),
        },
    )

    train_raw, test_raw, raw, use_explicit = load_raw_splits(
        split_cfg={
            "split_strategy": "random",
            "holdout_ratio": 0.25,
            "rand_seed": 13,
            "split_cache_path": "./splits/{model_name}_holdout_split.npz",
            "split_cache_validate_data_path": False,
        },
        data_cfg={
            "data_dir": "./DataFTEmbed",
            "data_format": "csv",
            "data_path_template": "{model_name}.{ext}",
        },
        data_cfg_path=cfg_path,
        model_name="demo",
        train_data_path=None,
        test_data_path=None,
    )

    assert use_explicit is False
    assert raw is not None and len(raw) == 4
    assert train_raw.index.tolist() == [0, 2]
    assert test_raw.index.tolist() == [1, 3]
