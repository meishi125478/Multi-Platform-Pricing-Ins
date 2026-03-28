"""Tests for production inference module (lightweight API checks)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from ins_pricing.exceptions import ModelLoadError

pytest.importorskip("ins_pricing.production.inference", reason="inference module not available")

import ins_pricing.production.inference as inference
from ins_pricing.production.inference import (
    ModelSpec,
    Predictor,
    PredictorRegistry,
    load_predictor,
    register_model_loader,
)


@dataclass
class _DummyPredictor(Predictor):
    value: float = 1.0

    def predict(self, df):  # type: ignore[override]
        return np.full(len(df), self.value)


def test_registry_loads_custom_predictor():
    registry = PredictorRegistry()
    captured = {}

    def _loader(spec: ModelSpec) -> Predictor:
        captured["spec"] = spec
        return _DummyPredictor(value=3.0)

    register_model_loader("xgb", _loader, registry=registry)
    spec = ModelSpec(
        model_key="xgb",
        model_name="demo",
        task_type="regression",
        cfg={},
        output_dir=Path("."),
        artifacts=None,
    )
    predictor = load_predictor(spec, registry=registry)
    assert isinstance(predictor, _DummyPredictor)
    assert captured["spec"] is spec


def test_registry_missing_key_raises():
    registry = PredictorRegistry()
    spec = ModelSpec(
        model_key="glm",
        model_name="demo",
        task_type="regression",
        cfg={},
        output_dir=Path("."),
        artifacts=None,
    )
    with pytest.raises(KeyError):
        load_predictor(spec, registry=registry)


def test_register_overwrite_controls():
    registry = PredictorRegistry()

    def _loader_one(spec: ModelSpec) -> Predictor:
        return _DummyPredictor(value=1.0)

    def _loader_two(spec: ModelSpec) -> Predictor:
        return _DummyPredictor(value=2.0)

    register_model_loader("ft", _loader_one, registry=registry)
    with pytest.raises(ValueError):
        register_model_loader("ft", _loader_two, registry=registry, overwrite=False)

    register_model_loader("ft", _loader_two, registry=registry, overwrite=True)
    spec = ModelSpec(
        model_key="ft",
        model_name="demo",
        task_type="regression",
        cfg={},
        output_dir=Path("."),
        artifacts=None,
    )
    predictor = load_predictor(spec, registry=registry)
    assert isinstance(predictor, _DummyPredictor)
    assert predictor.value == 2.0


def _touch_xgb_model_file(tmp_path: Path, model_name: str = "demo") -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"01_{model_name}_Xgboost.pkl"
    model_path.write_bytes(b"placeholder")
    return model_path


def test_load_saved_model_retries_unsafe_pickle_when_enabled(tmp_path, monkeypatch):
    _touch_xgb_model_file(tmp_path, model_name="demo")
    calls = []

    def _fake_load_pickle(path, *, allow_unsafe=None):
        calls.append(allow_unsafe)
        if len(calls) == 1:
            raise ModelLoadError(
                "Blocked unsafe pickle artifact: x. "
                "Set INS_PRICING_ALLOW_UNSAFE_MODEL_LOAD=1 only for trusted model files."
            )
        return {"model": "loaded-model"}

    monkeypatch.setattr(inference, "load_pickle_artifact", _fake_load_pickle)

    loaded = inference.load_saved_model(
        output_dir=tmp_path,
        model_name="demo",
        model_key="xgb",
        task_type="regression",
        input_dim=None,
        cfg={"allow_unsafe_model_load": True},
    )
    assert loaded == "loaded-model"
    assert calls == [None, True]


def test_load_saved_model_does_not_retry_unsafe_pickle_when_disabled(tmp_path, monkeypatch):
    _touch_xgb_model_file(tmp_path, model_name="demo")
    calls = []

    def _fake_load_pickle(path, *, allow_unsafe=None):
        calls.append(allow_unsafe)
        raise ModelLoadError(
            "Blocked unsafe pickle artifact: x. "
            "Set INS_PRICING_ALLOW_UNSAFE_MODEL_LOAD=1 only for trusted model files."
        )

    monkeypatch.setattr(inference, "load_pickle_artifact", _fake_load_pickle)

    with pytest.raises(ModelLoadError):
        inference.load_saved_model(
            output_dir=tmp_path,
            model_name="demo",
            model_key="xgb",
            task_type="regression",
            input_dim=None,
            cfg={"allow_unsafe_model_load": False},
        )
    assert calls == [None]
