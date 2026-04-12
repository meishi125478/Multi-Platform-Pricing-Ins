from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_PATH = REPO_ROOT / "modelling" / "bayesopt" / "artifacts.py"
CHECKPOINTS_PATH = REPO_ROOT / "modelling" / "bayesopt" / "checkpoints.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeIOUtils:
    @staticmethod
    def load_params_file(path: str):
        p = Path(path)
        if p.suffix.lower() == ".json":
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "best_params" in payload:
                payload = payload.get("best_params") or {}
            return dict(payload)
        if p.suffix.lower() in {".csv", ".tsv"}:
            df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
            if df.empty:
                raise ValueError("empty params file")
            params = df.iloc[0].to_dict()
            return {k: v for k, v in params.items() if not str(k).startswith("Unnamed")}
        raise ValueError("unsupported file type")


def _load_artifacts_module(monkeypatch: pytest.MonkeyPatch):
    fake_io_mod = types.ModuleType("ins_pricing.utils.io")
    fake_io_mod.IOUtils = _FakeIOUtils
    fake_utils_mod = types.ModuleType("ins_pricing.utils")
    fake_utils_mod.io = fake_io_mod
    fake_root_mod = types.ModuleType("ins_pricing")
    fake_root_mod.utils = fake_utils_mod
    monkeypatch.setitem(sys.modules, "ins_pricing", fake_root_mod)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils", fake_utils_mod)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils.io", fake_io_mod)
    return _load_module("test_artifacts_module", ARTIFACTS_PATH)


def test_best_params_filename_and_model_key_mapping(monkeypatch: pytest.MonkeyPatch):
    artifacts = _load_artifacts_module(monkeypatch)
    assert artifacts.best_params_filename("demo", "ResNet") == "demo_bestparams_resnet.csv"
    assert artifacts.trainer_label_from_model_key("FT") == "fttransformer"
    assert artifacts.trainer_label_from_model_key("custom") == "custom"


def test_load_best_params_ignores_versions_snapshot_and_reads_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifacts = _load_artifacts_module(monkeypatch)
    result_dir = tmp_path / "Results"
    versions_dir = result_dir / "versions"
    versions_dir.mkdir(parents=True)

    (versions_dir / "20260102_120000_xgb_best.json").write_text(
        json.dumps({"best_params": {"depth": 6}}),
        encoding="utf-8",
    )
    (result_dir / "demo_bestparams_xgboost.csv").write_text(
        "depth,learning_rate\n3,0.05\n",
        encoding="utf-8",
    )

    loaded = artifacts.load_best_params(tmp_path, "demo", "xgb")
    assert loaded == {"depth": 3.0, "learning_rate": 0.05}


def test_load_best_params_reads_base_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifacts = _load_artifacts_module(monkeypatch)
    result_dir = tmp_path / "Results"
    result_dir.mkdir(parents=True)
    (result_dir / "demo_bestparams_resnet.csv").write_text(
        "hidden_dim,dropout\n64,0.2\n",
        encoding="utf-8",
    )

    loaded = artifacts.load_best_params(tmp_path, "demo", "resn")
    assert loaded == {"hidden_dim": 64.0, "dropout": 0.2}


def test_load_best_params_returns_none_when_no_supported_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifacts = _load_artifacts_module(monkeypatch)
    result_dir = tmp_path / "Results"
    result_dir.mkdir(parents=True)
    (result_dir / "demo_bestparams_xgboost_old.csv").write_text(
        "hidden_dim,dropout\n22,0.22\n",
        encoding="utf-8",
    )
    (result_dir / "demo_bestparams_custom.csv").write_text(
        "hidden_dim,dropout\n44,0.44\n",
        encoding="utf-8",
    )

    loaded = artifacts.load_best_params(tmp_path, "demo", "xgb")
    assert loaded is None


def test_load_best_params_reads_base_csv_for_xgb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifacts = _load_artifacts_module(monkeypatch)
    result_dir = tmp_path / "Results"
    result_dir.mkdir(parents=True)
    (result_dir / "demo_bestparams_xgboost.csv").write_text(
        "hidden_dim,dropout\n11,0.11\n",
        encoding="utf-8",
    )

    loaded = artifacts.load_best_params(tmp_path, "demo", "xgb")
    assert loaded == {"hidden_dim": 11.0, "dropout": 0.11}


def test_rebuild_ft_model_from_payload_merges_missing_config(monkeypatch: pytest.MonkeyPatch):
    checkpoints = _load_module("test_checkpoints_module_ft", CHECKPOINTS_PATH)
    captured = {}

    def _fake_rebuild_ft_model_from_checkpoint(*, state_dict, model_config):
        captured["state_dict"] = state_dict
        captured["model_config"] = dict(model_config)
        return {"rebuilt": True}

    monkeypatch.setattr(
        checkpoints,
        "rebuild_ft_model_from_checkpoint",
        _fake_rebuild_ft_model_from_checkpoint,
    )
    payload = {
        "state_dict": {"weights": [1, 2, 3]},
        "model_config": {"loss_name": "poisson", "distribution": None},
        "best_params": {"n_heads": 4},
    }

    model, best_params, kind = checkpoints.rebuild_ft_model_from_payload(
        payload=payload,
        model_config_overrides={"loss_name": "gamma", "distribution": "gamma"},
        fill_missing_model_config=True,
    )

    assert model == {"rebuilt": True}
    assert best_params == {"n_heads": 4}
    assert kind == "state_dict"
    assert captured["state_dict"] == {"weights": [1, 2, 3]}
    assert captured["model_config"]["loss_name"] == "poisson"
    assert captured["model_config"]["distribution"] == "gamma"


def test_rebuild_resn_model_from_payload_with_fallback_params():
    checkpoints = _load_module("test_checkpoints_module_resn", CHECKPOINTS_PATH)

    class _ResNetCore:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state_dict):
            self.loaded = state_dict

    class _Model:
        def __init__(self, params):
            self.params = dict(params)
            self.resnet = _ResNetCore()

    def _builder(params):
        return _Model(params)

    model, params = checkpoints.rebuild_resn_model_from_payload(
        payload={"state_dict": {"layer": [1, 2]}},
        model_builder=_builder,
        params_fallback={"hidden_dim": 64},
    )
    assert params == {"hidden_dim": 64}
    assert model.params == {"hidden_dim": 64}
    assert model.resnet.loaded == {"layer": [1, 2]}


def test_rebuild_gnn_model_from_payload_strict_fallback_warning():
    checkpoints = _load_module("test_checkpoints_module_gnn", CHECKPOINTS_PATH)

    class _GNNCore:
        def __init__(self):
            self.calls = []

        def load_state_dict(self, state_dict, strict=True):
            self.calls.append((state_dict, strict))
            if strict:
                raise RuntimeError("Missing key(s) in state_dict")

    class _Model:
        def __init__(self, params):
            self.params = dict(params)
            self._core = _GNNCore()

        def set_params(self, params):
            self.params = dict(params)

        def _unwrap_gnn(self):
            return self._core

    def _builder(params):
        return _Model(params)

    model, params, warning = checkpoints.rebuild_gnn_model_from_payload(
        payload={"best_params": {"k_neighbors": 10}, "state_dict": {"w": [1]}},
        model_builder=_builder,
        strict=True,
        allow_non_strict_fallback=True,
    )

    assert params == {"k_neighbors": 10}
    assert model.params == {"k_neighbors": 10}
    assert warning is not None
    assert model._core.calls == [({"w": [1]}, True), ({"w": [1]}, False)]


def test_rebuild_gnn_model_from_payload_requires_dict_payload():
    checkpoints = _load_module("test_checkpoints_module_gnn_err", CHECKPOINTS_PATH)
    with pytest.raises(ValueError, match="Invalid GNN checkpoint payload"):
        checkpoints.rebuild_gnn_model_from_payload(
            payload="bad",
            model_builder=lambda params: object(),
        )


def test_rebuild_gnn_model_from_payload_requires_state_dict():
    checkpoints = _load_module("test_checkpoints_module_gnn_missing_state", CHECKPOINTS_PATH)
    with pytest.raises(ValueError, match="missing 'state_dict'"):
        checkpoints.rebuild_gnn_model_from_payload(
            payload={"best_params": {"k_neighbors": 10}},
            model_builder=lambda params: object(),
        )
