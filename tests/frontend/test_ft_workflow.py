import json
from pathlib import Path

import pandas as pd
import pytest

from ins_pricing.frontend.ft_workflow import FTWorkflowHelper


def _prepare_step1_inputs(
    tmp_path: Path,
    *,
    ft_embedding_source_features=None,
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
    }
    if ft_embedding_source_features is not None:
        cfg["ft_embedding_source_features"] = ft_embedding_source_features

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
        xgb_overrides={"runner": {"nproc_per_node": 3}, "xgb_max_depth_max": 7},
        resn_overrides={"runner": {"nproc_per_node": 4}, "use_resn_ddp": False},
    )

    assert xgb_cfg is not None
    assert resn_cfg is not None
    assert xgb_cfg["runner"]["nproc_per_node"] == 3
    assert xgb_cfg["xgb_max_depth_max"] == 7
    assert resn_cfg["runner"]["nproc_per_node"] == 4
    assert resn_cfg["use_resn_ddp"] is False

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
