from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ins_pricing.frontend import workflows_compare
from ins_pricing.frontend.workflows_compare import run_double_lift_from_file


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_compare_cfg(*, data_dir: str, output_dir: str) -> dict:
    return {
        "data_dir": data_dir,
        "data_format": "csv",
        "data_path_template": "{model_name}.{ext}",
        "model_list": ["od"],
        "model_categories": ["bc"],
        "target": "target",
        "weight": "weights",
        "feature_list": ["f1"],
        "categorical_features": [],
        "holdout_ratio": 0.25,
        "split_strategy": "random",
        "rand_seed": 13,
        "split_cache_key_col": "_row_id",
        "output_dir": output_dir,
        "cache_predictions": True,
        "prediction_cache_format": "csv",
    }


def test_run_double_lift_rejects_all_na_required_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "double_lift.csv"
    pd.DataFrame(
        {
            "pred_1": [None, None, None],
            "pred_2": [0.2, 0.4, 0.6],
            "target": [1.0, 2.0, 3.0],
            "weights": [1.0, 1.0, 1.0],
        }
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="No valid rows remain|non-numeric/NA rows"):
        run_double_lift_from_file(
            data_path=str(data_path),
            pred_col_1="pred_1",
            pred_col_2="pred_2",
            target_col="target",
            weight_col="weights",
            holdout_ratio=0.0,
        )


def test_run_compare_ft_embed_reuses_cached_predictions_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_dir = tmp_path / "cfg"
    data_dir = cfg_dir / "DataFTEmbed"
    data_dir.mkdir(parents=True, exist_ok=True)

    model_name = "od_bc"
    train_raw = pd.DataFrame(
        {"_row_id": [10, 11], "f1": [1.0, 2.0], "target": [1.5, 2.5], "weights": [1.0, 2.0]}
    )
    test_raw = pd.DataFrame(
        {"_row_id": [20], "f1": [3.0], "target": [3.5], "weights": [1.5]}
    )
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_raw.to_csv(train_path, index=False)
    test_raw.to_csv(test_path, index=False)

    direct_cfg_path = cfg_dir / "config_xgb_direct.json"
    ft_embed_cfg_path = cfg_dir / "config_xgb_from_ft_embed.json"
    ft_cfg_path = cfg_dir / "config_ft_ddp_embed.json"
    _write_json(
        direct_cfg_path,
        _base_compare_cfg(data_dir="./DataFTEmbed", output_dir="./ResultsXGBDirect"),
    )
    _write_json(
        ft_embed_cfg_path,
        _base_compare_cfg(data_dir="./DataFTEmbed", output_dir="./ResultsXGBFromFTEmbed"),
    )
    _write_json(ft_cfg_path, {"output_dir": "./ResultsFTEmbedDDP", "ft_feature_prefix": "ft_emb"})

    def _write_pred_cache(root: Path, *, train_values: list[float], test_values: list[float]) -> None:
        pred_dir = root / "Results" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        # Intentionally reversed order to verify key-based alignment by _row_id.
        pd.DataFrame({"_row_id": [11, 10], "pred_xgb": train_values[::-1]}).to_csv(
            pred_dir / f"{model_name}_xgb_train.csv",
            index=False,
        )
        pd.DataFrame({"_row_id": [20], "pred_xgb": test_values}).to_csv(
            pred_dir / f"{model_name}_xgb_test.csv",
            index=False,
        )

    _write_pred_cache(cfg_dir / "ResultsXGBDirect", train_values=[0.1, 0.2], test_values=[0.3])
    _write_pred_cache(cfg_dir / "ResultsXGBFromFTEmbed", train_values=[0.4, 0.5], test_values=[0.6])

    monkeypatch.setattr(
        workflows_compare,
        "build_ft_embedding_frames",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not build embedding frames")),
    )
    monkeypatch.setattr(
        workflows_compare,
        "_load_predictor_from_cfg",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load predictors")),
    )
    monkeypatch.setattr(workflows_compare, "plot_double_lift_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflows_compare, "_resolve_double_lift_dir", lambda model_name: tmp_path / "plots")

    def _fake_finalize(fig, *, save_path: str, show: bool, style) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(b"ok")

    monkeypatch.setattr(workflows_compare, "finalize_figure", _fake_finalize)

    output = workflows_compare.run_compare_ft_embed(
        direct_cfg_path=str(direct_cfg_path),
        ft_cfg_path=str(ft_cfg_path),
        ft_embed_cfg_path=str(ft_embed_cfg_path),
        model_key="xgb",
        label_direct="XGB_raw",
        label_ft="XGB_ft_embed",
        use_runtime_ft_embedding=False,
        n_bins_override=10,
        train_data_path=str(train_path),
        test_data_path=str(test_path),
    )

    assert Path(output).exists()


def test_run_compare_ft_embed_reuses_cached_predictions_with_duplicate_key_by_row_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_dir = tmp_path / "cfg"
    data_dir = cfg_dir / "DataFTEmbed"
    data_dir.mkdir(parents=True, exist_ok=True)

    model_name = "od_bc"
    train_raw = pd.DataFrame(
        {"_row_id": [10, 10], "f1": [1.0, 2.0], "target": [1.5, 2.5], "weights": [1.0, 2.0]}
    )
    test_raw = pd.DataFrame(
        {"_row_id": [20], "f1": [3.0], "target": [3.5], "weights": [1.5]}
    )
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train_raw.to_csv(train_path, index=False)
    test_raw.to_csv(test_path, index=False)

    direct_cfg_path = cfg_dir / "config_xgb_direct.json"
    ft_embed_cfg_path = cfg_dir / "config_xgb_from_ft_embed.json"
    ft_cfg_path = cfg_dir / "config_ft_ddp_embed.json"
    _write_json(
        direct_cfg_path,
        _base_compare_cfg(data_dir="./DataFTEmbed", output_dir="./ResultsXGBDirect"),
    )
    _write_json(
        ft_embed_cfg_path,
        _base_compare_cfg(data_dir="./DataFTEmbed", output_dir="./ResultsXGBFromFTEmbed"),
    )
    _write_json(ft_cfg_path, {"output_dir": "./ResultsFTEmbedDDP", "ft_feature_prefix": "ft_emb"})

    def _write_pred_cache(root: Path, *, train_values: list[float], test_values: list[float]) -> None:
        pred_dir = root / "Results" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"_row_id": [10, 10], "pred_xgb": train_values}).to_csv(
            pred_dir / f"{model_name}_xgb_train.csv",
            index=False,
        )
        pd.DataFrame({"_row_id": [20], "pred_xgb": test_values}).to_csv(
            pred_dir / f"{model_name}_xgb_test.csv",
            index=False,
        )

    _write_pred_cache(cfg_dir / "ResultsXGBDirect", train_values=[0.11, 0.22], test_values=[0.33])
    _write_pred_cache(cfg_dir / "ResultsXGBFromFTEmbed", train_values=[0.44, 0.55], test_values=[0.66])

    monkeypatch.setattr(
        workflows_compare,
        "build_ft_embedding_frames",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not build embedding frames")),
    )
    monkeypatch.setattr(
        workflows_compare,
        "_load_predictor_from_cfg",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load predictors")),
    )
    monkeypatch.setattr(workflows_compare, "plot_double_lift_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflows_compare, "_resolve_double_lift_dir", lambda model_name: tmp_path / "plots")

    def _fake_finalize(fig, *, save_path: str, show: bool, style) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(b"ok")

    monkeypatch.setattr(workflows_compare, "finalize_figure", _fake_finalize)

    output = workflows_compare.run_compare_ft_embed(
        direct_cfg_path=str(direct_cfg_path),
        ft_cfg_path=str(ft_cfg_path),
        ft_embed_cfg_path=str(ft_embed_cfg_path),
        model_key="xgb",
        label_direct="XGB_raw",
        label_ft="XGB_ft_embed",
        use_runtime_ft_embedding=False,
        n_bins_override=10,
        train_data_path=str(train_path),
        test_data_path=str(test_path),
    )

    assert Path(output).exists()
