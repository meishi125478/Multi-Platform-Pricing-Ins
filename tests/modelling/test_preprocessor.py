from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest

from ins_pricing.modelling.bayesopt.config_schema import BayesOptConfig
from ins_pricing.modelling.bayesopt.dataset_preprocessor import DatasetPreprocessor


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


def test_nested_config_views_refresh_from_flat_fields():
    cfg = _build_config(binary_resp=False)
    assert cfg.distributed.use_ft_ddp is False
    assert cfg.gnn.use_approx_knn is True

    cfg.use_ft_ddp = True
    cfg.gnn_use_approx_knn = False

    assert cfg.distributed.use_ft_ddp is True
    assert cfg.gnn.use_approx_knn is False


def test_nested_config_views_are_immutable():
    cfg = _build_config(binary_resp=False)
    with pytest.raises(FrozenInstanceError):
        cfg.distributed.use_ft_ddp = True


def test_preprocessor_scales_numeric_to_float32_and_keeps_columns_aligned():
    train = pd.DataFrame(
        {
            "x_num": [1.0, 2.0, 3.0, 4.0],
            "x_cat": ["a", "b", "a", "c"],
            "y": [10.0, 20.0, 30.0, 40.0],
            "w": [1.0, 1.0, 2.0, 2.0],
        }
    )
    test = pd.DataFrame(
        {
            "x_num": [5.0, 6.0],
            "x_cat": ["b", "d"],
            "y": [50.0, 60.0],
            "w": [1.0, 1.0],
        }
    )

    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x_num", "x_cat"],
        cate_list=["x_cat"],
        task_type="regression",
    )
    result = DatasetPreprocessor(train, test, cfg).run()

    assert result.train_oht_scl_data is not None
    assert result.test_oht_scl_data is not None
    assert result.train_oht_scl_data.columns.equals(result.test_oht_scl_data.columns)
    assert result.train_oht_scl_data["x_num"].dtype == np.float32
    assert result.test_oht_scl_data["x_num"].dtype == np.float32


def test_preprocessor_can_disable_unscaled_oht_cache():
    train = pd.DataFrame(
        {
            "x_num": [1.0, 2.0, 3.0, 4.0],
            "x_cat": ["a", "b", "a", "c"],
            "y": [10.0, 20.0, 30.0, 40.0],
            "w": [1.0, 1.0, 2.0, 2.0],
        }
    )
    test = pd.DataFrame(
        {
            "x_num": [5.0, 6.0],
            "x_cat": ["b", "d"],
            "y": [50.0, 60.0],
            "w": [1.0, 1.0],
        }
    )

    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x_num", "x_cat"],
        cate_list=["x_cat"],
        task_type="regression",
        keep_unscaled_oht=False,
    )
    result = DatasetPreprocessor(train, test, cfg).run()

    assert result.train_oht_data is None
    assert result.test_oht_data is None
    assert result.train_oht_scl_data is not None
    assert result.test_oht_scl_data is not None


def test_preprocessor_categorical_ohe_uses_csr_and_keeps_feature_names():
    train = pd.DataFrame(
        {
            "x_num": [1.0, 2.0, 3.0, 4.0],
            "x_cat": ["a", "b", "a", "c"],
            "y": [10.0, 20.0, 30.0, 40.0],
            "w": [1.0, 1.0, 2.0, 2.0],
        }
    )
    test = pd.DataFrame(
        {
            "x_num": [5.0, 6.0],
            "x_cat": ["b", "z"],  # unseen category in test
            "y": [50.0, 60.0],
            "w": [1.0, 1.0],
        }
    )

    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x_num", "x_cat"],
        cate_list=["x_cat"],
        task_type="regression",
        oht_sparse_csr=True,
    )
    result = DatasetPreprocessor(train, test, cfg).run()

    assert result.oht_sparse_csr is True
    assert result.train_cat_oht_csr is not None
    assert result.test_cat_oht_csr is not None
    assert len(result.ohe_feature_names) == 2
    assert all(name.startswith("x_cat_") for name in result.ohe_feature_names)

    assert result.train_oht_scl_data is not None
    dtypes = result.train_oht_scl_data[result.ohe_feature_names].dtypes
    assert all(str(dtype).startswith("Sparse[") for dtype in dtypes)

    # unseen category should map to all-zero one-hot row
    unseen_row = result.test_oht_scl_data.iloc[1][result.ohe_feature_names]
    unseen_sum = float(np.asarray(unseen_row.to_numpy(dtype=np.float32)).sum())
    assert unseen_sum == 0.0
