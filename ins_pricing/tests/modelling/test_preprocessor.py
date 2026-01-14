import numpy as np
import pandas as pd
import pytest

from ins_pricing.bayesopt.config_preprocess import BayesOptConfig, DatasetPreprocessor


def _build_config(binary_resp: bool = False) -> BayesOptConfig:
    return BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        binary_resp_nme="y_bin" if binary_resp else None,
    )


def test_preprocessor_fills_missing_test_labels():
    train = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "w": [1.0, 2.0, 3.0],
            "y_bin": [0, 1, 0],
        }
    )
    test = pd.DataFrame({"x1": [4.0, 5.0]})

    cfg = _build_config(binary_resp=True)
    result = DatasetPreprocessor(train, test, cfg).run()

    assert "w_act" in result.train_data.columns
    assert "w_act" not in result.test_data.columns
    assert "w_binary_act" in result.train_data.columns
    assert "w_binary_act" not in result.test_data.columns
    assert result.test_data["w"].eq(1.0).all()
    assert result.test_data["y"].isna().all()
    assert result.test_data["y_bin"].isna().all()


def test_preprocessor_missing_train_columns_raises():
    train = pd.DataFrame({"x1": [1.0]})
    test = pd.DataFrame({"x1": [2.0]})

    cfg = _build_config(binary_resp=False)
    with pytest.raises(KeyError):
        DatasetPreprocessor(train, test, cfg).run()
