"""Tests for production inference module (lightweight API checks)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ins_pricing.production.inference", reason="inference module not available")

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
