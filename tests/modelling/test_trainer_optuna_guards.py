from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")
pytest.importorskip("xgboost")

from ins_pricing.modelling.bayesopt.runtime.trainer_optuna import TrainerOptunaMixin
from ins_pricing.modelling.bayesopt.trainers.trainer_xgb import (
    XGBTrainer,
    _XGBDMatrixWrapper,
)


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


class _DummySanitizedTrainer(_DummyTrainer):
    def _sanitize_best_params(
        self,
        params: dict,
        *,
        context: str = "best_params",
    ) -> dict:
        _ = context
        out = dict(params or {})
        out.pop("drop_me", None)
        return out


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


def test_xgb_chunk_plan_covers_all_rows_and_rounds() -> None:
    plan = _XGBDMatrixWrapper._build_chunk_plan(
        total_rows=10,
        chunk_size=3,
        num_boost_round=5,
    )
    assert plan[0][0] == 0
    assert plan[-1][1] == 10
    assert sum(end - start for start, end, _ in plan) == 10
    assert sum(rounds for _, _, rounds in plan) == 5
    assert all(rounds >= 1 for _, _, rounds in plan)
    for prev, curr in zip(plan, plan[1:]):
        assert prev[1] == curr[0]


def test_xgb_chunk_plan_limits_chunk_count_by_boost_rounds() -> None:
    plan = _XGBDMatrixWrapper._build_chunk_plan(
        total_rows=100,
        chunk_size=10,
        num_boost_round=3,
    )
    assert len(plan) == 3
    assert sum(rounds for _, _, rounds in plan) == 3
    assert plan[-1][1] == 100


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


def test_objective_wrapper_prunes_dataloader_worker_runtime_error(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    study = trainer._create_study()
    trial = study.ask()
    progress_counter = {"count": 0}

    def _raise_dataloader_runtime(_trial: optuna.trial.Trial) -> float:
        raise RuntimeError("DataLoader worker (pid 123) exited unexpectedly")

    objective_wrapper = trainer._make_objective_wrapper(
        _raise_dataloader_runtime,
        total_trials=1,
        progress_counter=progress_counter,
        barrier_on_end=False,
    )

    with pytest.raises(optuna.TrialPruned):
        objective_wrapper(trial)
    assert progress_counter["count"] == 1


def test_run_study_and_extract_applies_best_param_sanitizer(tmp_path: Path) -> None:
    trainer = _DummySanitizedTrainer(tmp_path)
    study = trainer._create_study()
    progress_counter = {"count": 0}

    def _objective(trial: optuna.trial.Trial) -> float:
        trial.suggest_float("keep_me", 1e-5, 1e-3, log=True)
        trial.suggest_int("drop_me", 1, 5)
        return 0.1

    objective_wrapper = trainer._make_objective_wrapper(
        _objective,
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
    assert "keep_me" in trainer.best_params
    assert "drop_me" not in trainer.best_params


def test_run_study_and_extract_falls_back_when_global_best_params_are_empty(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    study = trainer._create_study()

    # Seed a "global best" trial with empty params (e.g., legacy study content).
    def _empty_best(_trial: optuna.trial.Trial) -> float:
        return 0.01

    # Seed another completed trial that has real tunable params.
    def _with_params(trial: optuna.trial.Trial) -> float:
        trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        return 0.1

    study.optimize(_empty_best, n_trials=1)
    study.optimize(_with_params, n_trials=1)

    progress_counter = {"count": 0}
    objective_wrapper = trainer._make_objective_wrapper(
        _with_params,
        total_trials=2,
        progress_counter=progress_counter,
        barrier_on_end=False,
    )
    checkpoint_callback = trainer._make_checkpoint_callback()

    trainer._run_study_and_extract(
        study=study,
        objective_wrapper=objective_wrapper,
        checkpoint_callback=checkpoint_callback,
        total_trials=2,
        progress_counter=progress_counter,
    )

    assert isinstance(trainer.best_params, dict)
    assert "learning_rate" in trainer.best_params
    assert trainer.best_trial is not None
    assert "learning_rate" in trainer.best_trial.params


def test_checkpoint_callback_falls_back_when_global_best_params_are_empty(tmp_path: Path) -> None:
    trainer = _DummyTrainer(tmp_path)
    study = trainer._create_study()

    def _empty_best(_trial: optuna.trial.Trial) -> float:
        return 0.01

    def _with_params(trial: optuna.trial.Trial) -> float:
        trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        return 0.1

    study.optimize(_empty_best, n_trials=1)
    study.optimize(_with_params, n_trials=1)

    callback = trainer._make_checkpoint_callback()
    callback(study, None)

    path = Path(trainer._best_params_csv_path())
    assert path.exists()
    saved = pd.read_csv(path)
    assert "learning_rate" in saved.columns


def _make_minimal_xgb_trainer(invalid_param_policy: str = "warn") -> XGBTrainer:
    trainer = object.__new__(XGBTrainer)
    trainer.label = "Xgboost"
    trainer.model_name_prefix = "Xgboost"
    trainer._invalid_param_warnings_emitted = set()
    trainer.ctx = types.SimpleNamespace(
        task_type="regression",
        config=types.SimpleNamespace(
            invalid_param_policy=invalid_param_policy,
            xgb_search_space={"learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2}},
        ),
    )
    return trainer


def test_xgb_sanitize_tuned_params_filters_unsupported_keys_under_warn_policy() -> None:
    trainer = _make_minimal_xgb_trainer(invalid_param_policy="warn")
    sanitized = trainer._sanitize_tuned_params(
        {"learning_rate": 1e-3, "geo_token_hidden_dim": 32}
    )
    assert sanitized == {"learning_rate": 1e-3}


def test_xgb_sanitize_tuned_params_raises_under_error_policy() -> None:
    trainer = _make_minimal_xgb_trainer(invalid_param_policy="error")
    with pytest.raises(ValueError, match="Unsupported params"):
        trainer._sanitize_tuned_params(
            {"learning_rate": 1e-3, "geo_token_hidden_dim": 32}
        )

