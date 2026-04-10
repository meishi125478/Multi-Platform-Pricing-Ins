import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ins_pricing.frontend.ft_workflow import FTWorkflowHelper


def _prepare_step1_inputs(
    tmp_path: Path,
    *,
    ft_embedding_source_features=None,
    split_cache_path=None,
) -> Path:
    model_name = "od_bc"
    data_dir = tmp_path / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5],
            "f2": ["a", "b", "a", "c", "b"],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5],
            "weights": [1.0, 1.0, 2.0, 2.0, 3.0],
        }
    )
    raw.to_csv(data_dir / f"{model_name}.csv", index=False)

    cfg = {
        "data_dir": "./Data",
        "output_dir": "./ResultsFT",
        "model_list": ["od"],
        "model_categories": ["bc"],
        "feature_list": ["f1", "f2"],
        "categorical_features": ["f2"],
        "target": "target",
        "weight": "weights",
        "prop_test": 0.4,
        "rand_seed": 13,
        "ft_feature_prefix": "ft_emb",
        "prediction_cache_format": "csv",
        "ft_search_space": {
            "d_model": {"type": "int", "low": 16, "high": 32, "step": 8},
        },
        "ft_unsupervised_search_space": {
            "d_model": {"type": "int", "low": 16, "high": 64, "step": 8},
        },
    }
    if ft_embedding_source_features is not None:
        cfg["ft_embedding_source_features"] = ft_embedding_source_features
    if split_cache_path is not None:
        cfg["split_cache_path"] = split_cache_path

    cfg_path = tmp_path / "temp_ft_step1_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    pred_dir = tmp_path / "ResultsFT" / "Results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_train = pd.DataFrame({"pred_ft_emb_0": [0.1, 0.2, 0.3]})
    pred_test = pd.DataFrame({"pred_ft_emb_0": [0.4, 0.5]})
    pred_train.to_csv(pred_dir / f"{model_name}_ft_emb_train.csv", index=False)
    pred_test.to_csv(pred_dir / f"{model_name}_ft_emb_test.csv", index=False)

    return cfg_path


def test_generate_step2_configs_applies_overrides_and_saves_augmented_data(tmp_path, monkeypatch):
    helper = FTWorkflowHelper()
    cfg_path = _prepare_step1_inputs(tmp_path)

    def _fake_alignment(raw, cfg, *, pred_train_rows, pred_test_rows):
        return raw.index[:pred_train_rows], raw.index[pred_train_rows:pred_train_rows + pred_test_rows]

    monkeypatch.setattr(helper, "_resolve_prediction_alignment_indices", _fake_alignment)

    xgb_cfg, resn_cfg = helper.generate_step2_configs(
        step1_config_path=str(cfg_path),
        target_models=["xgb", "resn"],
        augmented_data_dir="./DataFTUnsupervised",
        xgb_overrides={
            "output_dir": "./ResultsXGBFromFTEmbed",
            "runner": {"nproc_per_node": 3},
            "xgb_max_depth_max": 7,
        },
        resn_overrides={
            "output_dir": "./ResultsResNFromFTEmbed",
            "runner": {"nproc_per_node": 4},
            "use_resn_ddp": False,
        },
    )

    assert xgb_cfg is not None
    assert resn_cfg is not None
    assert xgb_cfg["runner"]["nproc_per_node"] == 3
    assert xgb_cfg["xgb_max_depth_max"] == 7
    assert "optuna_storage" not in xgb_cfg
    assert resn_cfg["runner"]["nproc_per_node"] == 4
    assert resn_cfg["use_resn_ddp"] is False
    assert "optuna_storage" not in resn_cfg
    assert "max_depth" in xgb_cfg["xgb_search_space"]
    assert xgb_cfg["ft_search_space"] == {}
    assert xgb_cfg["ft_unsupervised_search_space"] == {}
    assert xgb_cfg["resn_search_space"] == {}
    assert "hidden_dim" in resn_cfg["resn_search_space"]
    assert resn_cfg["ft_search_space"] == {}
    assert resn_cfg["ft_unsupervised_search_space"] == {}
    assert resn_cfg["xgb_search_space"] == {}

    # Default behavior: all raw feature_list columns are treated as embedding source.
    assert xgb_cfg["feature_list"] == ["pred_ft_emb_0"]
    assert xgb_cfg["categorical_features"] == []

    aug_path = tmp_path / "DataFTUnsupervised" / "od_bc.csv"
    assert aug_path.exists()
    aug = pd.read_csv(aug_path)
    assert len(aug) == 5
    assert "pred_ft_emb_0" in aug.columns


def test_generate_step2_configs_respects_embedding_source_subset(tmp_path, monkeypatch):
    helper = FTWorkflowHelper()
    cfg_path = _prepare_step1_inputs(
        tmp_path,
        ft_embedding_source_features=["f1"],
    )

    def _fake_alignment(raw, cfg, *, pred_train_rows, pred_test_rows):
        return raw.index[:pred_train_rows], raw.index[pred_train_rows:pred_train_rows + pred_test_rows]

    monkeypatch.setattr(helper, "_resolve_prediction_alignment_indices", _fake_alignment)

    xgb_cfg, _ = helper.generate_step2_configs(
        step1_config_path=str(cfg_path),
        target_models=["xgb"],
    )

    assert xgb_cfg is not None
    assert xgb_cfg["feature_list"] == ["f2", "pred_ft_emb_0"]
    assert xgb_cfg["categorical_features"] == ["f2"]


def test_generate_step2_configs_rejects_invalid_model_keys():
    helper = FTWorkflowHelper()

    with pytest.raises(ValueError, match="Unsupported Step 2 models"):
        helper.generate_step2_configs(
            step1_config_path="dummy.json",
            target_models=["xgb", "invalid_model"],
        )


def test_generate_step2_configs_prefers_split_cache_alignment(tmp_path, monkeypatch):
    helper = FTWorkflowHelper()
    cfg_path = _prepare_step1_inputs(
        tmp_path,
        split_cache_path="./Data/splits/{model_name}_holdout_split.npz",
    )
    cache_path = tmp_path / "Data" / "splits" / "od_bc_holdout_split.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_idx=np.asarray([0, 2, 4], dtype=np.int64),
        test_idx=np.asarray([1, 3], dtype=np.int64),
        row_count=np.asarray([5], dtype=np.int64),
        meta_json=np.asarray(
            [json.dumps({"split_strategy": "random", "holdout_ratio": 0.4, "rand_seed": 13})]
        ),
    )

    def _must_not_rebuild(*args, **kwargs):
        raise AssertionError("split cache should be used before rebuilding split")

    monkeypatch.setattr(helper, "_resolve_prediction_alignment_indices", _must_not_rebuild)

    xgb_cfg, _ = helper.generate_step2_configs(
        step1_config_path=str(cfg_path),
        target_models=["xgb"],
        augmented_data_dir="./DataFTFromCache",
    )

    assert xgb_cfg is not None
    aug = pd.read_csv(tmp_path / "DataFTFromCache" / "od_bc.csv")
    assert aug["pred_ft_emb_0"].tolist() == [0.1, 0.4, 0.2, 0.5, 0.3]


def test_generate_step2_configs_split_cache_missing_falls_back_to_rebuild(
    tmp_path,
    monkeypatch,
):
    helper = FTWorkflowHelper()
    cfg_path = _prepare_step1_inputs(
        tmp_path,
        split_cache_path="./Data/splits/{model_name}_holdout_split.npz",
    )

    called = {"fallback": False}

    def _fake_alignment(raw, cfg, *, pred_train_rows, pred_test_rows):
        _ = cfg
        called["fallback"] = True
        return raw.index[:pred_train_rows], raw.index[pred_train_rows:pred_train_rows + pred_test_rows]

    monkeypatch.setattr(helper, "_resolve_prediction_alignment_indices", _fake_alignment)

    xgb_cfg, _ = helper.generate_step2_configs(
        step1_config_path=str(cfg_path),
        target_models=["xgb"],
    )
    assert xgb_cfg is not None
    assert called["fallback"] is True


def test_generate_step2_configs_split_cache_seed_string_matches_numeric(
    tmp_path,
    monkeypatch,
):
    helper = FTWorkflowHelper()
    cfg_path = _prepare_step1_inputs(
        tmp_path,
        split_cache_path="./Data/splits/{model_name}_holdout_split.npz",
    )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["rand_seed"] = "13"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    cache_path = tmp_path / "Data" / "splits" / "od_bc_holdout_split.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_idx=np.asarray([0, 2, 4], dtype=np.int64),
        test_idx=np.asarray([1, 3], dtype=np.int64),
        row_count=np.asarray([5], dtype=np.int64),
        meta_json=np.asarray(
            [json.dumps({"split_strategy": "random", "holdout_ratio": 0.4, "rand_seed": 13})]
        ),
    )

    def _must_not_fallback(*args, **kwargs):
        raise AssertionError("split cache metadata check should accept seed string/int equivalence")

    monkeypatch.setattr(helper, "_resolve_prediction_alignment_indices", _must_not_fallback)

    xgb_cfg, _ = helper.generate_step2_configs(
        step1_config_path=str(cfg_path),
        target_models=["xgb"],
        augmented_data_dir="./DataFTFromCacheSeed",
    )
    assert xgb_cfg is not None


@pytest.mark.parametrize(
    ("augmented_format", "expected_suffix"),
    [("parquet", ".parquet"), ("feather", ".feather")],
)
def test_generate_step2_configs_supports_binary_augmented_formats(
    tmp_path,
    monkeypatch,
    augmented_format,
    expected_suffix,
):
    helper = FTWorkflowHelper()
    cfg_path = _prepare_step1_inputs(tmp_path)

    def _fake_alignment(raw, cfg, *, pred_train_rows, pred_test_rows):
        return raw.index[:pred_train_rows], raw.index[pred_train_rows:pred_train_rows + pred_test_rows]

    captured = {}

    def _fake_write_table(df, path, *, data_format):
        captured["path"] = Path(path)
        captured["format"] = data_format
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(helper, "_resolve_prediction_alignment_indices", _fake_alignment)
    monkeypatch.setattr(helper, "_write_table", _fake_write_table)

    xgb_cfg, _ = helper.generate_step2_configs(
        step1_config_path=str(cfg_path),
        target_models=["xgb"],
        augmented_data_dir="./DataFTBinary",
        augmented_data_format=augmented_format,
    )

    assert xgb_cfg is not None
    assert captured["format"] == augmented_format
    assert captured["path"].suffix == expected_suffix
    assert captured["path"].exists()
    assert xgb_cfg["data_format"] == augmented_format
    assert xgb_cfg["data_path_template"] == "{model_name}.{ext}"
    assert xgb_cfg["data_dir"] == "./DataFTBinary"
