from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ins_pricing.frontend import workflows_plot


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_cfg(*, data_dir: str) -> dict:
    return {
        "data_dir": data_dir,
        "model_list": ["od"],
        "model_categories": ["bc"],
        "target": "reponse",
        "weight": "weights",
        "feature_list": ["f1"],
        "categorical_features": [],
        "model_keys": ["xgb"],
        "plot": {"n_bins": 10},
    }


def test_run_plot_embed_precomputed_uses_ft_cfg_for_split(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config_plot.json"
    xgb_cfg_path = tmp_path / "config_xgb.json"
    resn_cfg_path = tmp_path / "config_resn.json"
    ft_cfg_path = tmp_path / "config_ft.json"

    _write_json(cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    _write_json(xgb_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    _write_json(resn_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    _write_json(ft_cfg_path, _base_cfg(data_dir="./Data"))

    observed: dict = {}

    def _fake_load_raw_splits(**kwargs):
        observed["data_cfg_path"] = kwargs["data_cfg_path"]
        observed["split_cfg_data_dir"] = kwargs["split_cfg"].get("data_dir")
        frame = pd.DataFrame({"f1": [1.0], "reponse": [0.0], "weights": [1.0]})
        return frame.copy(), frame.copy(), frame.copy(), False

    monkeypatch.setattr(workflows_plot, "load_raw_splits", _fake_load_raw_splits)
    monkeypatch.setattr(
        workflows_plot,
        "build_ft_embedding_frames",
        lambda **kwargs: (kwargs["train_raw"].copy(), kwargs["test_raw"].copy()),
    )
    monkeypatch.setattr(
        workflows_plot,
        "_run_prediction_plot_workflow",
        lambda **kwargs: "ok",
    )

    message = workflows_plot.run_plot_embed(
        cfg_path=str(cfg_path),
        xgb_cfg_path=str(xgb_cfg_path),
        resn_cfg_path=str(resn_cfg_path),
        ft_cfg_path=str(ft_cfg_path),
        use_runtime_ft_embedding=False,
    )

    assert message == "ok"
    assert observed["data_cfg_path"] == ft_cfg_path.resolve()
    assert observed["split_cfg_data_dir"] == "./Data"


def test_run_plot_embed_runtime_uses_ft_cfg_for_split(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config_plot.json"
    xgb_cfg_path = tmp_path / "config_xgb.json"
    resn_cfg_path = tmp_path / "config_resn.json"
    ft_cfg_path = tmp_path / "config_ft.json"

    _write_json(cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    _write_json(xgb_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    _write_json(resn_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    _write_json(ft_cfg_path, _base_cfg(data_dir="./Data"))

    observed: dict = {}

    def _fake_load_raw_splits(**kwargs):
        observed["data_cfg_path"] = kwargs["data_cfg_path"]
        observed["split_cfg_data_dir"] = kwargs["split_cfg"].get("data_dir")
        frame = pd.DataFrame({"f1": [1.0], "reponse": [0.0], "weights": [1.0]})
        return frame.copy(), frame.copy(), frame.copy(), False

    monkeypatch.setattr(workflows_plot, "load_raw_splits", _fake_load_raw_splits)
    monkeypatch.setattr(
        workflows_plot,
        "build_ft_embedding_frames",
        lambda **kwargs: (kwargs["train_raw"].copy(), kwargs["test_raw"].copy()),
    )
    monkeypatch.setattr(
        workflows_plot,
        "_run_prediction_plot_workflow",
        lambda **kwargs: "ok",
    )

    message = workflows_plot.run_plot_embed(
        cfg_path=str(cfg_path),
        xgb_cfg_path=str(xgb_cfg_path),
        resn_cfg_path=str(resn_cfg_path),
        ft_cfg_path=str(ft_cfg_path),
        use_runtime_ft_embedding=True,
    )

    assert message == "ok"
    assert observed["data_cfg_path"] == ft_cfg_path.resolve()
    assert observed["split_cfg_data_dir"] == "./Data"


def test_run_plot_embed_precomputed_passes_model_features_to_embed_loader(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config_plot.json"
    xgb_cfg_path = tmp_path / "config_xgb.json"
    resn_cfg_path = tmp_path / "config_resn.json"
    ft_cfg_path = tmp_path / "config_ft.json"

    _write_json(cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    xgb_cfg = _base_cfg(data_dir="./DataFTEmbed")
    xgb_cfg["feature_list"] = ["f1", "pred_ft_emb_0"]
    _write_json(xgb_cfg_path, xgb_cfg)
    _write_json(resn_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))
    _write_json(ft_cfg_path, _base_cfg(data_dir="./Data"))

    observed: dict = {}

    def _fake_load_raw_splits(**kwargs):
        observed["data_cfg_path"] = kwargs["data_cfg_path"]
        frame = pd.DataFrame({"f1": [1.0], "pred_ft_emb_0": [0.5], "reponse": [0.0], "weights": [1.0]})
        return frame.copy(), frame.copy(), frame.copy(), False

    monkeypatch.setattr(workflows_plot, "load_raw_splits", _fake_load_raw_splits)
    def _fake_build_ft_embedding_frames(**kwargs):
        observed["required_columns"] = kwargs.get("required_columns")
        observed["embed_cfg_path"] = kwargs.get("embed_cfg_path")
        return kwargs["train_raw"].copy(), kwargs["test_raw"].copy()

    monkeypatch.setattr(workflows_plot, "build_ft_embedding_frames", _fake_build_ft_embedding_frames)
    monkeypatch.setattr(
        workflows_plot,
        "_run_prediction_plot_workflow",
        lambda **kwargs: "ok",
    )

    message = workflows_plot.run_plot_embed(
        cfg_path=str(cfg_path),
        xgb_cfg_path=str(xgb_cfg_path),
        resn_cfg_path=str(resn_cfg_path),
        ft_cfg_path=str(ft_cfg_path),
        use_runtime_ft_embedding=False,
    )

    assert message == "ok"
    assert observed["data_cfg_path"] == ft_cfg_path.resolve()
    assert observed["required_columns"] is not None
    assert "pred_ft_emb_0" in observed["required_columns"]
    assert observed["embed_cfg_path"] == xgb_cfg_path.resolve()


def test_run_plot_direct_xgb_only_without_resn_cfg_path(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config_plot.json"
    xgb_cfg_path = tmp_path / "config_xgb.json"

    cfg = _base_cfg(data_dir="./DataFTEmbed")
    cfg["model_keys"] = ["xgb"]
    _write_json(cfg_path, cfg)
    _write_json(xgb_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))

    observed: dict = {}

    def _fake_load_raw_splits(**kwargs):
        observed["data_cfg_path"] = kwargs["data_cfg_path"]
        frame = pd.DataFrame({"f1": [1.0], "reponse": [0.0], "weights": [1.0]})
        return frame.copy(), frame.copy(), frame.copy(), False

    def _fake_run_prediction_plot_workflow(**kwargs):
        observed["xgb_cfg_path"] = kwargs["xgb_cfg_path"]
        observed["resn_cfg_path"] = kwargs["resn_cfg_path"]
        return "ok"

    monkeypatch.setattr(workflows_plot, "load_raw_splits", _fake_load_raw_splits)
    monkeypatch.setattr(
        workflows_plot,
        "_run_prediction_plot_workflow",
        _fake_run_prediction_plot_workflow,
    )

    message = workflows_plot.run_plot_direct(
        cfg_path=str(cfg_path),
        xgb_cfg_path=str(xgb_cfg_path),
        resn_cfg_path=None,
    )

    assert message == "ok"
    assert observed["data_cfg_path"] == xgb_cfg_path.resolve()
    assert observed["xgb_cfg_path"] == xgb_cfg_path.resolve()
    assert observed["resn_cfg_path"] is None


def test_run_plot_direct_resn_only_without_xgb_cfg_path(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config_plot.json"
    resn_cfg_path = tmp_path / "config_resn.json"

    cfg = _base_cfg(data_dir="./DataFTEmbed")
    cfg["model_keys"] = ["resn"]
    _write_json(cfg_path, cfg)
    _write_json(resn_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))

    observed: dict = {}

    def _fake_load_raw_splits(**kwargs):
        observed["data_cfg_path"] = kwargs["data_cfg_path"]
        frame = pd.DataFrame({"f1": [1.0], "reponse": [0.0], "weights": [1.0]})
        return frame.copy(), frame.copy(), frame.copy(), False

    def _fake_run_prediction_plot_workflow(**kwargs):
        observed["xgb_cfg_path"] = kwargs["xgb_cfg_path"]
        observed["resn_cfg_path"] = kwargs["resn_cfg_path"]
        return "ok"

    monkeypatch.setattr(workflows_plot, "load_raw_splits", _fake_load_raw_splits)
    monkeypatch.setattr(
        workflows_plot,
        "_run_prediction_plot_workflow",
        _fake_run_prediction_plot_workflow,
    )

    message = workflows_plot.run_plot_direct(
        cfg_path=str(cfg_path),
        xgb_cfg_path=None,
        resn_cfg_path=str(resn_cfg_path),
    )

    assert message == "ok"
    assert observed["data_cfg_path"] == resn_cfg_path.resolve()
    assert observed["xgb_cfg_path"] is None
    assert observed["resn_cfg_path"] == resn_cfg_path.resolve()


def test_run_plot_direct_explicit_xgb_does_not_fallback_to_cfg_resn_path(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config_plot.json"
    xgb_cfg_path = tmp_path / "config_xgb.json"

    cfg = _base_cfg(data_dir="./DataFTEmbed")
    cfg["model_keys"] = ["xgb", "resn"]
    cfg["resn_cfg_path"] = "missing_resn.json"
    _write_json(cfg_path, cfg)
    _write_json(xgb_cfg_path, _base_cfg(data_dir="./DataFTEmbed"))

    observed: dict = {}

    def _fake_load_raw_splits(**kwargs):
        observed["data_cfg_path"] = kwargs["data_cfg_path"]
        frame = pd.DataFrame({"f1": [1.0], "reponse": [0.0], "weights": [1.0]})
        return frame.copy(), frame.copy(), frame.copy(), False

    def _fake_run_prediction_plot_workflow(**kwargs):
        observed["xgb_cfg_path"] = kwargs["xgb_cfg_path"]
        observed["resn_cfg_path"] = kwargs["resn_cfg_path"]
        return "ok"

    monkeypatch.setattr(workflows_plot, "load_raw_splits", _fake_load_raw_splits)
    monkeypatch.setattr(
        workflows_plot,
        "_run_prediction_plot_workflow",
        _fake_run_prediction_plot_workflow,
    )

    message = workflows_plot.run_plot_direct(
        cfg_path=str(cfg_path),
        xgb_cfg_path=str(xgb_cfg_path),
        resn_cfg_path=None,
    )

    assert message == "ok"
    assert observed["data_cfg_path"] == xgb_cfg_path.resolve()
    assert observed["xgb_cfg_path"] == xgb_cfg_path.resolve()
    assert observed["resn_cfg_path"] is None
