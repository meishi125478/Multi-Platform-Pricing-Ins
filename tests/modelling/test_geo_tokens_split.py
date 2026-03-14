import types

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("optuna")
pytest.importorskip("xgboost")
pytest.importorskip("statsmodels")

from ins_pricing.modelling.bayesopt.trainers import FTTrainer


class DummyCtx:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.task_type = "regression"
        self.config = types.SimpleNamespace(use_ft_ddp=False, geo_feature_nmes=["geo"])
        self.train_data = train_df
        self.test_data = test_df
        self.model_nme = "dummy_ft"
        self.num_features = ["x"]
        self.cate_list = []
        self.train_geo_tokens = None
        self.test_geo_tokens = None
        self.geo_token_cols = []
        self.geo_gnn_model = None
        self._build_calls = []
        self._last_params = None

    def _build_geo_tokens(self, _params=None):
        self._last_params = _params
        self._build_calls.append(
            (self.train_data.copy(deep=True), self.test_data.copy(deep=True))
        )
        return self.train_data.copy(deep=True), self.test_data.copy(deep=True), ["geo_token"], None


def test_apply_model_params_drops_unsupported_geo_keys():
    train = pd.DataFrame({"geo": ["a", "b"], "x": [1, 2]})
    test = pd.DataFrame({"geo": ["c"], "x": [3]})
    ctx = DummyCtx(train, test)
    trainer = FTTrainer(ctx)

    class DummyModel:
        def __init__(self):
            self.calls = []
            self.d_model = 32
            self.dropout = 0.0

        def set_params(self, params):
            self.calls.append(dict(params))
            if any(key.startswith("geo_token_") for key in params):
                raise AssertionError("geo params must not be passed directly")
            return self

    model = DummyModel()
    filtered = trainer._apply_model_params(
        model,
        {
            "d_model": 64,
            "dropout": 0.1,
            "geo_token_hidden_dim": 32,
            "geo_token_layers": 2,
        },
    )

    assert filtered == {"d_model": 64, "dropout": 0.1}
    assert model.calls == [{"d_model": 64, "dropout": 0.1}]


def test_sanitize_best_params_filters_geo_and_unknown_keys():
    train = pd.DataFrame({"geo": ["a", "b", "c"], "x": [1, 2, 3]})
    test = pd.DataFrame({"geo": ["d"], "x": [4]})
    ctx = DummyCtx(train, test)
    trainer = FTTrainer(ctx)

    sanitized = trainer._sanitize_best_params(
        {
            "d_model": 64,
            "dropout": 0.1,
            "geo_token_hidden_dim": 16,
            "geo_token_layers": 1,
            "not_a_ft_param": 999,
        },
        context="test",
    )
    assert sanitized == {"d_model": 64, "dropout": 0.1}
