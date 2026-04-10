import types

import pandas as pd
import pytest

from ins_pricing.modelling.bayesopt.trainers.trainer_resn import ResNetTrainer


def _make_trainer(*, train_df: pd.DataFrame, test_df: pd.DataFrame, factor_nmes):
    trainer = object.__new__(ResNetTrainer)
    trainer.ctx = types.SimpleNamespace(
        factor_nmes=list(factor_nmes),
        train_data=train_df,
        test_data=test_df,
    )
    trainer._raw_design_cache = None
    trainer._raw_fallback_logged = False
    trainer._raw_encoded_cols_logged = False
    return trainer


def test_resn_raw_fallback_encodes_non_numeric_columns_consistently():
    train_df = pd.DataFrame(
        {
            "num": [1.0, 2.5, 3.5],
            "cat": ["A", "B", "A"],
            "flag": [True, False, True],
        }
    )
    test_df = pd.DataFrame(
        {
            "num": [4.0, 5.0],
            "cat": ["B", "C"],  # unseen "C" should map to -1
            "flag": [False, True],
        }
    )
    trainer = _make_trainer(
        train_df=train_df,
        test_df=test_df,
        factor_nmes=["num", "cat", "flag"],
    )

    X_train, X_test = trainer._resolve_raw_design_matrices(require_test=True)

    assert X_test is not None
    assert list(X_train.columns) == ["num", "cat", "flag"]
    assert str(X_train["num"].dtype) == "float32"
    assert str(X_train["cat"].dtype) == "float32"
    assert str(X_train["flag"].dtype) == "float32"
    assert X_train["cat"].tolist() == [0.0, 1.0, 0.0]
    assert X_test["cat"].tolist() == [1.0, -1.0]


def test_resn_raw_fallback_uses_cache_for_same_signature():
    train_df = pd.DataFrame({"x": [1, 2, 3]})
    test_df = pd.DataFrame({"x": [4, 5]})
    trainer = _make_trainer(train_df=train_df, test_df=test_df, factor_nmes=["x"])

    X_train_1, _ = trainer._resolve_raw_design_matrices(require_test=False)
    X_train_2, _ = trainer._resolve_raw_design_matrices(require_test=False)

    assert X_train_1 is X_train_2


def test_resn_raw_fallback_requires_factor_nmes():
    trainer = _make_trainer(
        train_df=pd.DataFrame({"x": [1]}),
        test_df=pd.DataFrame({"x": [2]}),
        factor_nmes=[],
    )
    with pytest.raises(RuntimeError, match="factor_nmes"):
        trainer._resolve_raw_design_matrices(require_test=False)
