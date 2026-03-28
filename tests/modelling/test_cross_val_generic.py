import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("optuna")
pytest.importorskip("statsmodels")
pytest.importorskip("xgboost")

from ins_pricing.modelling.bayesopt.trainers import TrainerBase


class DummyTrainer(TrainerBase):
    def __init__(self):
        cfg = types.SimpleNamespace(bo_sample_limit=None, cv_splits=3)
        ctx = types.SimpleNamespace(prop_test=0.2, rand_seed=123, config=cfg)
        super().__init__(ctx, "Dummy", "Dummy")

    def train(self) -> None:  # pragma: no cover - not used
        raise NotImplementedError


def test_cross_val_generic_iterates_all_splits():
    trainer = DummyTrainer()

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.arange(12, dtype=float))
    w = pd.Series(np.ones(12, dtype=float))

    def data_provider():
        return X, y, w

    class DummyModel:
        def fit(self, X_train, y_train, sample_weight=None):
            return self

        def predict(self, X_val):
            return np.zeros(len(X_val))

    def model_builder(_params):
        return DummyModel()

    calls = []

    def metric_fn(y_true, y_pred, weight):
        calls.append(len(y_true))
        return float(np.mean(y_pred))

    splits = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11])),
        (np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), np.array([0, 1])),
        (np.array([0, 2, 4, 6, 8, 10]), np.array([1, 3, 5, 7, 9, 11])),
    ]

    result = trainer.cross_val_generic(
        trial=object(),
        hyperparameter_space={"p": lambda _t: 1.0},
        data_provider=data_provider,
        model_builder=model_builder,
        metric_fn=metric_fn,
        splitter=splits,
    )

    assert result == 0.0
    assert len(calls) == len(splits)


def test_cross_val_generic_uses_weighted_fold_aggregation():
    trainer = DummyTrainer()

    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.arange(12, dtype=float))
    w = pd.Series(np.ones(12, dtype=float))

    def data_provider():
        return X, y, w

    class DummyModel:
        def fit(self, X_train, y_train, sample_weight=None):
            return self

        def predict(self, X_val):
            return np.zeros(len(X_val))

    def model_builder(_params):
        return DummyModel()

    fold_losses = [1.0, 3.0, 5.0]
    idx = {"i": 0}

    def metric_fn(y_true, y_pred, weight):
        _ = y_true, y_pred, weight
        value = fold_losses[idx["i"]]
        idx["i"] += 1
        return value

    splits = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11])),  # weight=2
        (np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), np.array([0, 1])),   # weight=2
        (np.array([0, 2, 4, 6, 8, 10]), np.array([1, 3, 5, 7, 9, 11])),    # weight=6
    ]

    result = trainer.cross_val_generic(
        trial=object(),
        hyperparameter_space={"p": lambda _t: 1.0},
        data_provider=data_provider,
        model_builder=model_builder,
        metric_fn=metric_fn,
        splitter=splits,
    )

    expected = (1.0 * 2 + 3.0 * 2 + 5.0 * 6) / 10.0
    assert result == pytest.approx(expected)


def test_bo_sample_limit_overrides_model_default_cap():
    trainer = DummyTrainer()

    assert trainer._resolve_effective_sample_limit(base_limit=100, n_rows=1000) == 100

    trainer.ctx.config.bo_sample_limit = 300
    assert trainer._resolve_effective_sample_limit(base_limit=100, n_rows=1000) == 300

    trainer.ctx.config.bo_sample_limit = 1200
    assert trainer._resolve_effective_sample_limit(base_limit=100, n_rows=1000) == 1000

    trainer.ctx.config.bo_sample_limit = 0
    assert trainer._resolve_effective_sample_limit(base_limit=100, n_rows=1000) == 100
