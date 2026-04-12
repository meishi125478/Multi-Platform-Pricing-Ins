"""Tests for production inference module (lightweight API checks)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PATH = REPO_ROOT / "production" / "inference.py"


class _FakeLogger:
    def warning(self, *args, **kwargs):
        _ = args, kwargs

    def info(self, *args, **kwargs):
        _ = args, kwargs

    def debug(self, *args, **kwargs):
        _ = args, kwargs

    def error(self, *args, **kwargs):
        _ = args, kwargs


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def inference_mod(monkeypatch: pytest.MonkeyPatch):
    fake_logger = _FakeLogger()

    fake_root = types.ModuleType("ins_pricing")
    fake_root.__path__ = [str(REPO_ROOT)]

    fake_prod = types.ModuleType("ins_pricing.production")
    fake_prod.__path__ = [str(REPO_ROOT / "production")]

    fake_modelling = types.ModuleType("ins_pricing.modelling")
    fake_modelling.__path__ = [str(REPO_ROOT / "modelling")]

    fake_bayesopt = types.ModuleType("ins_pricing.modelling.bayesopt")
    fake_bayesopt.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt")]

    fake_utils = types.ModuleType("ins_pricing.utils")
    fake_utils.__path__ = [str(REPO_ROOT / "utils")]
    fake_utils.get_logger = lambda _name: fake_logger
    fake_utils.load_dataset = lambda *args, **kwargs: None

    fake_device = types.ModuleType("ins_pricing.utils.device")

    class _DeviceManager:
        @staticmethod
        def move_to_device(model_obj, device=None):
            _ = model_obj, device
            return None

    fake_device.DeviceManager = _DeviceManager

    fake_losses = types.ModuleType("ins_pricing.utils.losses")
    fake_losses.resolve_effective_loss_name = (
        lambda loss_name, task_type, model_name, distribution=None: loss_name or "mse"
    )
    fake_losses.resolve_tweedie_power = lambda loss_name, default=1.5: default

    fake_model_loading = types.ModuleType("ins_pricing.utils.model_loading")
    fake_model_loading.load_pickle_artifact = lambda *args, **kwargs: None
    fake_model_loading.load_torch_payload = lambda *args, **kwargs: None
    fake_model_loading.load_model_artifact_payload = lambda *args, **kwargs: None
    fake_model_rebuild = types.ModuleType("ins_pricing.utils.model_rebuild")
    fake_model_rebuild.rebuild_ft_payload = lambda *args, **kwargs: (
        None,
        None,
        "raw",
    )
    fake_model_rebuild.rebuild_resn_payload = lambda *args, **kwargs: (None, None)
    fake_model_rebuild.rebuild_gnn_payload = lambda *args, **kwargs: (
        None,
        None,
        None,
    )

    fake_exceptions = types.ModuleType("ins_pricing.exceptions")

    class ModelLoadError(Exception):
        pass

    fake_exceptions.ModelLoadError = ModelLoadError

    fake_artifacts = types.ModuleType("ins_pricing.modelling.bayesopt.artifacts")
    fake_artifacts.load_best_params = lambda *args, **kwargs: None

    fake_checkpoints = types.ModuleType("ins_pricing.modelling.bayesopt.checkpoints")
    fake_checkpoints.rebuild_ft_model_from_payload = lambda *args, **kwargs: (
        None,
        None,
        "raw",
    )
    fake_checkpoints.rebuild_resn_model_from_payload = lambda *args, **kwargs: (None, None)
    fake_checkpoints.rebuild_gnn_model_from_payload = lambda *args, **kwargs: (
        None,
        None,
        None,
    )

    fake_preprocess = types.ModuleType("ins_pricing.production.preprocess")
    fake_preprocess.apply_preprocess_artifacts = lambda df, artifacts: df
    fake_preprocess.load_preprocess_artifacts = lambda path: None
    fake_preprocess.prepare_raw_features = lambda df, artifacts: df

    fake_scoring = types.ModuleType("ins_pricing.production.scoring")
    fake_scoring.batch_score = lambda fn, frame, batch_size=10000: fn(frame)

    monkeypatch.setitem(sys.modules, "ins_pricing", fake_root)
    monkeypatch.setitem(sys.modules, "ins_pricing.production", fake_prod)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling", fake_modelling)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling.bayesopt", fake_bayesopt)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils.device", fake_device)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils.losses", fake_losses)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils.model_loading", fake_model_loading)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils.model_rebuild", fake_model_rebuild)
    monkeypatch.setitem(sys.modules, "ins_pricing.exceptions", fake_exceptions)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling.bayesopt.artifacts", fake_artifacts)
    monkeypatch.setitem(
        sys.modules,
        "ins_pricing.modelling.bayesopt.checkpoints",
        fake_checkpoints,
    )
    monkeypatch.setitem(sys.modules, "ins_pricing.production.preprocess", fake_preprocess)
    monkeypatch.setitem(sys.modules, "ins_pricing.production.scoring", fake_scoring)

    module_name = "ins_pricing.production.inference"
    module = _load_module(module_name, INFERENCE_PATH)
    try:
        yield module
    finally:
        # _load_module writes into sys.modules directly; clean it up to avoid
        # any cross-file leakage when pytest runs multiple files in one process.
        sys.modules.pop(module_name, None)


@dataclass
class _DummyPredictor:
    value: float = 1.0

    def predict(self, df):
        return np.full(len(df), self.value)


def test_registry_loads_custom_predictor(inference_mod):
    registry = inference_mod.PredictorRegistry()
    captured = {}

    def _loader(spec):
        captured["spec"] = spec
        return _DummyPredictor(value=3.0)

    inference_mod.register_model_loader("xgb", _loader, registry=registry)
    spec = inference_mod.ModelSpec(
        model_key="xgb",
        model_name="demo",
        task_type="regression",
        cfg={},
        output_dir=Path("."),
        artifacts=None,
    )
    predictor = inference_mod.load_predictor(spec, registry=registry)
    assert isinstance(predictor, _DummyPredictor)
    assert captured["spec"] is spec


def test_registry_missing_key_raises(inference_mod):
    registry = inference_mod.PredictorRegistry()
    spec = inference_mod.ModelSpec(
        model_key="glm",
        model_name="demo",
        task_type="regression",
        cfg={},
        output_dir=Path("."),
        artifacts=None,
    )
    with pytest.raises(KeyError):
        inference_mod.load_predictor(spec, registry=registry)


def test_register_overwrite_controls(inference_mod):
    registry = inference_mod.PredictorRegistry()

    def _loader_one(spec):
        return _DummyPredictor(value=1.0)

    def _loader_two(spec):
        return _DummyPredictor(value=2.0)

    inference_mod.register_model_loader("ft", _loader_one, registry=registry)
    with pytest.raises(ValueError):
        inference_mod.register_model_loader(
            "ft", _loader_two, registry=registry, overwrite=False
        )

    inference_mod.register_model_loader("ft", _loader_two, registry=registry, overwrite=True)
    spec = inference_mod.ModelSpec(
        model_key="ft",
        model_name="demo",
        task_type="regression",
        cfg={},
        output_dir=Path("."),
        artifacts=None,
    )
    predictor = inference_mod.load_predictor(spec, registry=registry)
    assert isinstance(predictor, _DummyPredictor)
    assert predictor.value == 2.0


def test_load_best_params_passes_through_to_artifacts(
    inference_mod,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def _fake_load_best_params(output_dir, model_name, model_key):
        captured["args"] = (output_dir, model_name, model_key)
        return {"hidden_dim": 64}

    monkeypatch.setattr(inference_mod, "load_best_params_artifacts", _fake_load_best_params)

    loaded = inference_mod.load_best_params(
        Path("/tmp"),
        "demo",
        "xgb",
    )

    assert loaded == {"hidden_dim": 64}
    assert captured["args"] == (Path("/tmp"), "demo", "xgb")


def _touch_xgb_model_file(tmp_path: Path, model_name: str = "demo") -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"01_{model_name}_Xgboost.pkl"
    model_path.write_bytes(b"placeholder")
    return model_path


def test_load_saved_model_pickle_loader_called_once(tmp_path, monkeypatch, inference_mod):
    _touch_xgb_model_file(tmp_path, model_name="demo")
    calls = []

    def _fake_load_payload(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = map_location, allow_unsafe_pickle_retry
        assert model_key == "xgb"
        calls.append(Path(path).name)
        return {"model": "loaded-model"}

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fake_load_payload)

    loaded = inference_mod.load_saved_model(
        output_dir=tmp_path,
        model_name="demo",
        model_key="xgb",
        task_type="regression",
        input_dim=None,
        cfg={},
    )
    assert loaded == "loaded-model"
    assert calls == ["01_demo_Xgboost.pkl"]


def test_load_saved_model_propagates_pickle_error_without_retry(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    _touch_xgb_model_file(tmp_path, model_name="demo")
    calls = []

    def _fake_load_payload(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = model_key, map_location, allow_unsafe_pickle_retry
        calls.append(Path(path).name)
        raise RuntimeError("secure loader failed")

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fake_load_payload)

    with pytest.raises(RuntimeError, match="secure loader failed"):
        inference_mod.load_saved_model(
            output_dir=tmp_path,
            model_name="demo",
            model_key="xgb",
            task_type="regression",
            input_dim=None,
            cfg={},
        )
    assert calls == ["01_demo_Xgboost.pkl"]


def test_load_saved_model_can_retry_with_allow_unsafe_override(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    _touch_xgb_model_file(tmp_path, model_name="demo")

    def _fake_load_payload(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = path, map_location
        assert model_key == "xgb"
        assert allow_unsafe_pickle_retry is True
        return {"model": "unsafe-loaded"}

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fake_load_payload)

    loaded = inference_mod.load_saved_model(
        output_dir=tmp_path,
        model_name="demo",
        model_key="xgb",
        task_type="regression",
        input_dim=None,
        cfg={},
        allow_unsafe_model_load=True,
    )
    assert loaded == "unsafe-loaded"


def test_load_saved_model_does_not_retry_unsafe_for_non_restricted_errors(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    _touch_xgb_model_file(tmp_path, model_name="demo")
    def _fail_payload_load(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = path, model_key, map_location, allow_unsafe_pickle_retry
        raise RuntimeError("Cannot read model artifact: demo.pkl")

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fail_payload_load)

    with pytest.raises(RuntimeError, match="Cannot read model artifact"):
        inference_mod.load_saved_model(
            output_dir=tmp_path,
            model_name="demo",
            model_key="xgb",
            task_type="regression",
            input_dim=None,
            cfg={},
            allow_unsafe_model_load=True,
        )

def test_load_saved_model_raises_on_corrupt_checkpoint(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    checkpoint = _touch_xgb_model_file(tmp_path, model_name="demo")
    calls = []
    warnings = []

    def _fake_load_payload(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = model_key, map_location, allow_unsafe_pickle_retry
        calls.append(Path(path).name)
        if Path(path) == checkpoint:
            raise ValueError("corrupt checkpoint payload")
        return {"model": "loaded-model"}

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fake_load_payload)
    monkeypatch.setattr(
        inference_mod._logger,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="corrupt checkpoint payload"):
        inference_mod.load_saved_model(
            output_dir=tmp_path,
            model_name="demo",
            model_key="xgb",
            task_type="regression",
            input_dim=None,
            cfg={},
        )

    assert calls == [checkpoint.name]
    assert len(warnings) == 1


def test_load_saved_model_resn_uses_primary_checkpoint_and_base_params(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "01_demo_ResNet.pth"
    checkpoint.write_bytes(b"checkpoint")

    payload_calls = []

    def _fake_load_payload(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = map_location, allow_unsafe_pickle_retry
        assert model_key == "resn"
        payload_calls.append(Path(path).name)
        if Path(path) == checkpoint:
            raise ValueError("broken checkpoint")
        return {"state_dict": {"layer": [1, 2]}}

    def _fake_rebuild_resn_model_from_payload(*, payload, model_builder):
        _ = payload, model_builder
        return (
            types.SimpleNamespace(resnet=types.SimpleNamespace(load_state_dict=lambda *_a, **_k: None)),
            {},
        )

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fake_load_payload)
    monkeypatch.setattr(
        inference_mod,
        "rebuild_resn_model_from_payload",
        _fake_rebuild_resn_model_from_payload,
    )

    with pytest.raises(ValueError, match="broken checkpoint"):
        inference_mod.load_saved_model(
            output_dir=tmp_path,
            model_name="demo",
            model_key="resn",
            task_type="regression",
            input_dim=2,
            cfg={},
        )

    assert payload_calls == [checkpoint.name]


def test_load_saved_model_resn_uses_fallback_params_when_checkpoint_has_no_best_params(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "01_demo_ResNet.pth"
    checkpoint.write_bytes(b"checkpoint")

    def _fake_load_payload(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = path, map_location, allow_unsafe_pickle_retry
        assert model_key == "resn"
        return {"state_dict": {"layer": [1, 2, 3]}}

    captured = {}
    rebuilt_model = types.SimpleNamespace()

    def _fake_rebuild_resn_model_from_payload(
        *,
        payload,
        model_builder,
        params_fallback=None,
        require_params=True,
    ):
        _ = payload, model_builder
        captured["params_fallback"] = params_fallback
        captured["require_params"] = require_params
        return rebuilt_model, dict(params_fallback or {})

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fake_load_payload)
    monkeypatch.setattr(inference_mod, "load_best_params", lambda *args, **kwargs: {"hidden_dim": 64})
    monkeypatch.setattr(
        inference_mod,
        "rebuild_resn_model_from_payload",
        _fake_rebuild_resn_model_from_payload,
    )

    loaded = inference_mod.load_saved_model(
        output_dir=tmp_path,
        model_name="demo",
        model_key="resn",
        task_type="regression",
        input_dim=2,
        cfg={},
    )

    assert loaded is rebuilt_model
    assert captured["params_fallback"] == {"hidden_dim": 64}
    assert captured["require_params"] is False


def test_load_preprocess_from_model_file_uses_primary_checkpoint(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    checkpoint = _touch_xgb_model_file(tmp_path, model_name="demo")
    calls = []

    def _fake_load_payload(path, *, model_key, map_location="cpu", allow_unsafe_pickle_retry=False):
        _ = model_key, map_location, allow_unsafe_pickle_retry
        calls.append(Path(path).name)
        if Path(path) == checkpoint:
            raise ValueError("corrupt checkpoint payload")
        return {"preprocess_artifacts": {"token": "loaded"}}

    monkeypatch.setattr(inference_mod, "load_model_artifact_payload", _fake_load_payload)

    with pytest.raises(ValueError, match="corrupt checkpoint payload"):
        inference_mod._load_preprocess_from_model_file(
            tmp_path,
            "demo",
            "xgb",
        )

    assert calls == [checkpoint.name]


def test_resolve_model_file_path_returns_primary_checkpoint(tmp_path, inference_mod):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "01_demo_Xgboost.pkl"
    checkpoint.write_bytes(b"checkpoint")

    resolved = inference_mod._resolve_model_file_path(
        tmp_path,
        "demo",
        "xgb",
    )
    assert resolved == checkpoint


def test_load_predictor_from_config_accepts_allow_unsafe_override(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_list": ["od"],
                "model_categories": ["bc"],
                "output_dir": "./results",
                "task_type": "regression",
                "feature_list": ["x"],
                "categorical_features": [],
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def _fake_load_predictor(spec, registry=None):
        _ = registry
        captured["cfg"] = dict(spec.cfg)
        return _DummyPredictor(value=9.9)

    monkeypatch.setattr(inference_mod, "load_predictor", _fake_load_predictor)

    predictor = inference_mod.load_predictor_from_config(
        cfg_path,
        "xgb",
        allow_unsafe_model_load=True,
    )
    assert isinstance(predictor, _DummyPredictor)
    assert captured["cfg"].get("allow_unsafe_model_load") is True


def test_predict_with_model_prefers_class_one_probability(inference_mod):
    class _BinaryModel:
        classes_ = np.array([0, 1])

        @staticmethod
        def predict_proba(features):
            n = len(features)
            return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])

    features = pd.DataFrame({"x": [1, 2, 3]})
    preds = inference_mod._predict_with_model(
        model=_BinaryModel(),
        model_key="xgb",
        task_type="classification",
        features=features,
    )
    assert np.allclose(preds, np.array([0.2, 0.2, 0.2]))


def test_predict_with_model_handles_single_column_probability(inference_mod):
    class _SingleColumnModel:
        @staticmethod
        def predict_proba(features):
            return np.full((len(features), 1), 0.7)

    features = pd.DataFrame({"x": [1, 2]})
    preds = inference_mod._predict_with_model(
        model=_SingleColumnModel(),
        model_key="xgb",
        task_type="classification",
        features=features,
    )
    assert np.allclose(preds, np.array([0.7, 0.7]))


def test_predict_from_config_can_stream_chunked_csv(tmp_path, monkeypatch, inference_mod):
    class _ConstPredictor:
        def predict(self, frame: pd.DataFrame):
            return np.full(len(frame), 1.5)

    chunks = iter(
        [
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"a": [3]}),
        ]
    )

    monkeypatch.setattr(inference_mod, "load_dataset", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(
        inference_mod,
        "load_predictor_from_config",
        lambda *args, **kwargs: _ConstPredictor(),
    )

    def _fake_batch_score(fn, frame, *, output_col, batch_size, keep_input):
        _ = batch_size, keep_input
        return pd.DataFrame({output_col: fn(frame)})

    monkeypatch.setattr(inference_mod, "batch_score", _fake_batch_score)

    input_path = tmp_path / "input.csv"
    input_path.write_text("a\n1\n", encoding="utf-8")
    output_path = tmp_path / "pred.csv"

    result = inference_mod.predict_from_config(
        config_path=tmp_path / "dummy.json",
        input_path=input_path,
        model_keys=["xgb"],
        output_path=output_path,
        return_full_result=False,
    )

    assert result.empty
    written = pd.read_csv(output_path)
    assert list(written.columns) == ["a", "pred_xgb"]
    assert len(written) == 3
    assert np.allclose(written["pred_xgb"].to_numpy(), np.array([1.5, 1.5, 1.5]))


def test_predict_from_config_requires_output_path_when_streaming_without_result(
    tmp_path,
    monkeypatch,
    inference_mod,
):
    class _ConstPredictor:
        def predict(self, frame: pd.DataFrame):
            return np.full(len(frame), 1.0)

    chunks = iter([pd.DataFrame({"a": [1]})])
    monkeypatch.setattr(inference_mod, "load_dataset", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(
        inference_mod,
        "load_predictor_from_config",
        lambda *args, **kwargs: _ConstPredictor(),
    )
    monkeypatch.setattr(
        inference_mod,
        "batch_score",
        lambda fn, frame, *, output_col, batch_size, keep_input: pd.DataFrame({output_col: fn(frame)}),
    )

    input_path = tmp_path / "input.csv"
    input_path.write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_path is required"):
        inference_mod.predict_from_config(
            config_path=tmp_path / "dummy.json",
            input_path=input_path,
            model_keys=["xgb"],
            return_full_result=False,
        )
