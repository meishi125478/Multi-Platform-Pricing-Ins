from __future__ import annotations

from ins_pricing.frontend.config_builder import ConfigBuilder


def _base_config() -> dict:
    return {
        "data_dir": "./data",
        "model_list": ["od"],
        "model_categories": ["bc"],
        "target": "response",
        "weight": "weight",
        "feature_list": ["x1", "x2"],
        "categorical_features": ["x2"],
        "split_strategy": "group",
    }


def test_validate_config_allows_group_strategy_with_explicit_split_paths() -> None:
    builder = ConfigBuilder()
    cfg = _base_config()
    cfg["train_data_path"] = "./train.parquet"
    cfg["test_data_path"] = "./test.parquet"
    ok, msg = builder.validate_config(cfg)
    assert ok, msg


def test_validate_config_requires_group_col_when_split_is_executed() -> None:
    builder = ConfigBuilder()
    cfg = _base_config()
    ok, msg = builder.validate_config(cfg)
    assert not ok
    assert "split_group_col" in msg
