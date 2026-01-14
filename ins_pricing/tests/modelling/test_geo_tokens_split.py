import types

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("optuna")
pytest.importorskip("xgboost")
pytest.importorskip("statsmodels")

from ins_pricing.bayesopt.trainers import FTTrainer


class DummyCtx:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.task_type = "regression"
        self.config = types.SimpleNamespace(use_ft_ddp=False, geo_feature_nmes=["geo"])
        self.train_data = train_df
        self.test_data = test_df
        self._build_calls = []

    def _build_geo_tokens(self, _params=None):
        self._build_calls.append(
            (self.train_data.copy(deep=True), self.test_data.copy(deep=True))
        )
        return self.train_data.copy(deep=True), self.test_data.copy(deep=True), ["geo_token"], None


def test_geo_token_split_uses_fold_and_restores_context():
    train = pd.DataFrame({"geo": ["a", "b", "c", "d"], "x": [1, 2, 3, 4]})
    test = pd.DataFrame({"geo": ["e"], "x": [5]})
    ctx = DummyCtx(train, test)
    trainer = FTTrainer(ctx)

    X_train = train.iloc[[0, 1]]
    X_val = train.iloc[[2, 3]]

    orig_train = ctx.train_data
    orig_test = ctx.test_data

    result = trainer._build_geo_tokens_for_split(X_train, X_val, geo_params={"k": 1})

    assert ctx.train_data is orig_train
    assert ctx.test_data is orig_test
    assert result is not None

    train_snapshot, test_snapshot = ctx._build_calls[0]
    assert train_snapshot.equals(train.loc[X_train.index])
    assert test_snapshot.equals(train.loc[X_val.index])
