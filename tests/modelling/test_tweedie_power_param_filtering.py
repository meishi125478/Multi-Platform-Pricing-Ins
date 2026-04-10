from __future__ import annotations

import types

import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from ins_pricing.modelling.bayesopt.trainers.trainer_ft import FTTrainer
from ins_pricing.modelling.bayesopt.trainers.trainer_resn import ResNetTrainer


def _make_minimal_ft_trainer(*, task_type: str, loss_name: str) -> FTTrainer:
    trainer = object.__new__(FTTrainer)
    trainer._ft_tweedie_space_warned = False
    trainer.ctx = types.SimpleNamespace(task_type=task_type, loss_name=loss_name)
    return trainer


def _make_minimal_resn_trainer(*, task_type: str, loss_name: str) -> ResNetTrainer:
    trainer = object.__new__(ResNetTrainer)
    trainer._resn_tweedie_space_warned = False
    trainer.ctx = types.SimpleNamespace(task_type=task_type, loss_name=loss_name)
    return trainer


def test_ft_filter_search_space_drops_tw_power_for_gamma() -> None:
    trainer = _make_minimal_ft_trainer(task_type="regression", loss_name="gamma")
    filtered = trainer._filter_search_space_for_distribution(
        {
            "d_model": {"type": "int", "low": 32, "high": 128},
            "tw_power": {"type": "float", "low": 1.0, "high": 2.0},
        }
    )
    assert "d_model" in filtered
    assert "tw_power" not in filtered


def test_ft_filter_search_space_keeps_tw_power_for_tweedie() -> None:
    trainer = _make_minimal_ft_trainer(task_type="regression", loss_name="tweedie")
    filtered = trainer._filter_search_space_for_distribution(
        {
            "d_model": {"type": "int", "low": 32, "high": 128},
            "tw_power": {"type": "float", "low": 1.0, "high": 2.0},
        }
    )
    assert "d_model" in filtered
    assert "tw_power" in filtered


def test_ft_drop_tw_power_if_unused_for_non_tweedie() -> None:
    trainer = _make_minimal_ft_trainer(task_type="regression", loss_name="gamma")
    params = trainer._drop_tw_power_if_unused({"d_model": 64, "tw_power": 1.33})
    assert params == {"d_model": 64}


def test_resn_filter_search_space_drops_tw_power_for_gamma() -> None:
    trainer = _make_minimal_resn_trainer(task_type="regression", loss_name="gamma")
    filtered = trainer._filter_search_space_for_distribution(
        {
            "hidden_dim": {"type": "int", "low": 16, "high": 64},
            "tw_power": {"type": "float", "low": 1.0, "high": 2.0},
        }
    )
    assert "hidden_dim" in filtered
    assert "tw_power" not in filtered


def test_resn_drop_tw_power_if_unused_for_classification() -> None:
    trainer = _make_minimal_resn_trainer(
        task_type="classification",
        loss_name="logloss",
    )
    params = trainer._drop_tw_power_if_unused({"hidden_dim": 64, "tw_power": 1.72})
    assert params == {"hidden_dim": 64}
