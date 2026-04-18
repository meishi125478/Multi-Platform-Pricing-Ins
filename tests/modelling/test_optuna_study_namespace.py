from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("optuna")


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "modelling" / "bayesopt" / "runtime" / "trainer_optuna.py"


def _load_trainer_optuna_module(monkeypatch: pytest.MonkeyPatch):
    fake_root = types.ModuleType("ins_pricing")
    fake_root.__path__ = [str(REPO_ROOT)]

    fake_modelling = types.ModuleType("ins_pricing.modelling")
    fake_modelling.__path__ = [str(REPO_ROOT / "modelling")]

    fake_bayesopt = types.ModuleType("ins_pricing.modelling.bayesopt")
    fake_bayesopt.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt")]

    fake_bayesopt_utils = types.ModuleType("ins_pricing.modelling.bayesopt.utils")
    fake_bayesopt_utils.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt" / "utils")]

    fake_utils = types.ModuleType("ins_pricing.utils")
    fake_utils.ensure_parent_dir = lambda *_args, **_kwargs: None
    fake_utils.get_logger = lambda _name: None
    fake_utils.log_print = lambda *_args, **_kwargs: None

    fake_artifacts = types.ModuleType("ins_pricing.modelling.bayesopt.artifacts")
    fake_artifacts.best_params_csv_path = lambda *args, **kwargs: Path("/tmp/best_params.csv")

    class _DistributedUtils:
        @staticmethod
        def is_main_process() -> bool:
            return True

    fake_distributed_utils = types.ModuleType(
        "ins_pricing.modelling.bayesopt.utils.distributed_utils"
    )
    fake_distributed_utils.DistributedUtils = _DistributedUtils

    fake_torch = types.ModuleType("torch")

    monkeypatch.setitem(sys.modules, "ins_pricing", fake_root)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling", fake_modelling)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling.bayesopt", fake_bayesopt)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling.bayesopt.utils", fake_bayesopt_utils)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling.bayesopt.artifacts", fake_artifacts)
    monkeypatch.setitem(
        sys.modules,
        "ins_pricing.modelling.bayesopt.utils.distributed_utils",
        fake_distributed_utils,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    spec = importlib.util.spec_from_file_location(
        "test_trainer_optuna_loaded",
        str(MODULE_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_trainer(module):
    class _DummyTrainer(module.TrainerOptunaMixin):
        def __init__(self) -> None:
            self.label = "Xgboost"
            self.model_name_prefix = "xgb"
            self.best_params = None
            self.best_trial = None
            self.study_name = None
            self.enable_distributed_optuna = False
            self.ctx = types.SimpleNamespace(
                model_nme="policy",
                rand_seed=13,
            )
            self.config = types.SimpleNamespace(
                optuna_storage=None,
                optuna_study_prefix="bayesopt",
                optuna_cleanup_synchronize=False,
            )
            self.output = types.SimpleNamespace(result_dir="/tmp")

        def _clean_gpu(self, synchronize: bool = False) -> None:
            _ = synchronize
            return None

    return _DummyTrainer()


def test_optuna_study_name_is_unsuffixed(monkeypatch: pytest.MonkeyPatch):
    module = _load_trainer_optuna_module(monkeypatch)
    trainer = _make_trainer(module)

    study_name = trainer._resolve_optuna_study_name()

    assert study_name == "bayesopt_policy_xgb"


def test_optuna_study_name_has_no_engine_suffix(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_trainer_optuna_module(monkeypatch)
    trainer = _make_trainer(module)

    study_name = trainer._resolve_optuna_study_name()

    assert study_name == "bayesopt_policy_xgb"
    assert "_eng_" not in study_name
