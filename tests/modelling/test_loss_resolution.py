import pytest

from ins_pricing.exceptions import ConfigurationError
from ins_pricing.modelling.bayesopt.config_preprocess import BayesOptConfig
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
