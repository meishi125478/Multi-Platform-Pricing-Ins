import pytest

from ins_pricing.exceptions import ConfigurationError
from ins_pricing.modelling.bayesopt.config_schema import BayesOptConfig
from ins_pricing.utils.losses import (
    normalize_distribution_name,
    resolve_effective_loss_name,
)


def test_distribution_overrides_loss_name_for_regression():
    assert (
        resolve_effective_loss_name(
            "mae",
            task_type="regression",
            model_name="demo_f",
            distribution="poisson",
        )
        == "poisson"
    )


def test_auto_resolution_keeps_legacy_behavior_when_distribution_not_set():
    assert resolve_effective_loss_name(
        None, task_type="regression", model_name="demo_f"
    ) == "poisson"
    assert resolve_effective_loss_name(
        None, task_type="regression", model_name="demo_s"
    ) == "gamma"
    assert resolve_effective_loss_name(
        None, task_type="regression", model_name="demo"
    ) == "tweedie"


def test_distribution_aliases_are_normalized():
    assert normalize_distribution_name("normal", "regression") == "gaussian"
    assert normalize_distribution_name("mse", "regression") == "gaussian"
    assert normalize_distribution_name("laplacian", "regression") == "laplace"


def test_config_normalizes_distribution():
    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        distribution="normal",
    )
    assert cfg.distribution == "gaussian"


def test_config_rejects_invalid_distribution():
    with pytest.raises(ConfigurationError):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="regression",
            distribution="invalid_dist",
        )


def test_config_rejects_regression_distribution_for_classification():
    with pytest.raises(ConfigurationError):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="classification",
            distribution="gamma",
        )


def test_config_rejects_non_positive_xgb_chunk_size():
    with pytest.raises(ConfigurationError, match="xgb_chunk_size"):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="regression",
            xgb_chunk_size=0,
        )


def test_config_accepts_positive_xgb_chunk_size():
    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        xgb_chunk_size=2000,
    )
    assert cfg.xgb_chunk_size == 2000
    assert cfg.xgboost.chunk_size == 2000


def test_config_rejects_non_positive_resn_predict_batch_size():
    with pytest.raises(ConfigurationError, match="resn_predict_batch_size"):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="regression",
            resn_predict_batch_size=0,
        )


def test_config_accepts_positive_resn_predict_batch_size():
    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        resn_predict_batch_size=2048,
    )
    assert cfg.resn_predict_batch_size == 2048


def test_config_rejects_non_positive_ft_predict_batch_size():
    with pytest.raises(ConfigurationError, match="ft_predict_batch_size"):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="regression",
            ft_predict_batch_size=0,
        )


def test_config_accepts_positive_ft_predict_batch_size():
    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        ft_predict_batch_size=1024,
    )
    assert cfg.ft_predict_batch_size == 1024
    assert cfg.ft_transformer.predict_batch_size == 1024


def test_config_rejects_non_positive_gnn_row_limits():
    with pytest.raises(ConfigurationError, match="gnn_max_fit_rows"):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="regression",
            gnn_max_fit_rows=0,
        )
    with pytest.raises(ConfigurationError, match="gnn_max_predict_rows"):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="regression",
            gnn_max_predict_rows=0,
        )
    with pytest.raises(ConfigurationError, match="gnn_predict_chunk_rows"):
        BayesOptConfig(
            model_nme="demo",
            resp_nme="y",
            weight_nme="w",
            factor_nmes=["x1"],
            task_type="regression",
            gnn_predict_chunk_rows=0,
        )


def test_config_accepts_positive_gnn_row_limits():
    cfg = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        gnn_max_fit_rows=50000,
        gnn_max_predict_rows=60000,
        gnn_predict_chunk_rows=10000,
    )
    assert cfg.gnn_max_fit_rows == 50000
    assert cfg.gnn_max_predict_rows == 60000
    assert cfg.gnn_predict_chunk_rows == 10000
    assert cfg.gnn.max_fit_rows == 50000
    assert cfg.gnn.max_predict_rows == 60000
    assert cfg.gnn.predict_chunk_rows == 10000
