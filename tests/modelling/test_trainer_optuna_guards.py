from __future__ import annotations

import types
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")
pytest.importorskip("xgboost")

from ins_pricing.modelling.bayesopt.runtime.trainer_optuna import TrainerOptunaMixin
from ins_pricing.modelling.bayesopt.trainers.trainer_xgb import XGBTrainer


class _DummyTrainer(TrainerOptunaMixin):
    def __init__(self, tmp_path: Path) -> None:
        self.label = "Dummy"
        self.model_name_prefix = "dummy"
        self.best_params = None
        self.best_trial = None
        self.study_name = None
        self.enable_distributed_optuna = False
        self.ctx = types.SimpleNamespace(rand_seed=13, model_nme="demo_model")
        self.config = types.SimpleNamespace(
            optuna_storage=None,
            optuna_study_prefix="unit",
            optuna_cleanup_synchronize=False,
        )
        self.output = types.SimpleNamespace(result_dir=str(tmp_path))

    def _clean_gpu(self, synchronize: bool = False) -> None:
        return None


def test_run_study_and_extract_raises_for_all_pruned_trials(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    study = trainer._create_study()
    progress_counter = {"count": 0}

    def _always_prune(_trial: optuna.trial.Trial) -> float:
        raise optuna.TrialPruned("forced prune for test")

    objective_wrapper = trainer._make_objective_wrapper(
        _always_prune,
        total_trials=1,
        progress_counter=progress_counter,
        barrier_on_end=False,
    )
    checkpoint_callback = trainer._make_checkpoint_callback()

    with pytest.raises(RuntimeError, match="No completed trials"):
        trainer._run_study_and_extract(
            study=study,
            objective_wrapper=objective_wrapper,
            checkpoint_callback=checkpoint_callback,
            total_trials=1,
            progress_counter=progress_counter,
        )


def test_xgb_slow_prune_disabled_when_trial_budget_is_tiny() -> None:
    trainer = object.__new__(XGBTrainer)

    trainer._optuna_total_trials = 1
    assert trainer._should_prune_slow_config(max_depth=21, n_estimators=490) is False

    trainer._optuna_total_trials = 2
    assert trainer._should_prune_slow_config(max_depth=21, n_estimators=490) is False

    trainer._optuna_total_trials = 3
    assert trainer._should_prune_slow_config(max_depth=21, n_estimators=490) is True


def test_run_study_and_extract_recovers_from_pruned_only_history(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    study = trainer._create_study()
    progress_counter = {"count": 0}

    def _always_prune(_trial: optuna.trial.Trial) -> float:
        raise optuna.TrialPruned("seed pruned trial")

    study.optimize(_always_prune, n_trials=1)

    def _complete(trial: optuna.trial.Trial) -> float:
        value = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        return float(value)

    objective_wrapper = trainer._make_objective_wrapper(
        _complete,
        total_trials=1,
        progress_counter=progress_counter,
        barrier_on_end=False,
    )
    checkpoint_callback = trainer._make_checkpoint_callback()

    trainer._run_study_and_extract(
        study=study,
        objective_wrapper=objective_wrapper,
        checkpoint_callback=checkpoint_callback,
        total_trials=1,
        progress_counter=progress_counter,
    )

    assert isinstance(trainer.best_params, dict)
    assert "learning_rate" in trainer.best_params
