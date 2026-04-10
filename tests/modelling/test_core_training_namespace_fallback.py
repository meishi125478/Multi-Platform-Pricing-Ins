from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "modelling" / "bayesopt" / "core_training_mixin.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_core_training_module():
    fake_root = types.ModuleType("ins_pricing")
    fake_root.__path__ = [str(REPO_ROOT)]

    fake_modelling = types.ModuleType("ins_pricing.modelling")
    fake_modelling.__path__ = [str(REPO_ROOT / "modelling")]

    fake_bayesopt = types.ModuleType("ins_pricing.modelling.bayesopt")
    fake_bayesopt.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt")]

    fake_runtime = types.ModuleType("ins_pricing.modelling.bayesopt.runtime")
    fake_runtime.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt" / "runtime")]

    fake_utils = types.ModuleType("ins_pricing.utils")
    fake_utils.__path__ = [str(REPO_ROOT / "utils")]
    fake_utils.get_logger = lambda _name: None
    fake_utils.log_print = lambda *_args, **_kwargs: None

    fake_utils_io = types.ModuleType("ins_pricing.utils.io")
    fake_utils_io.IOUtils = type("IOUtils", (), {})
    fake_utils.io = fake_utils_io

    fake_artifacts = types.ModuleType("ins_pricing.modelling.bayesopt.artifacts")
    fake_artifacts.best_params_csv_path = lambda *args, **kwargs: Path("/tmp/best.csv")
    fake_artifacts.extract_best_params_from_snapshot = lambda payload: None
    fake_artifacts.load_best_params_csv = lambda *args, **kwargs: None

    fake_dispatcher = types.ModuleType("ins_pricing.modelling.bayesopt.runtime.dispatcher")
    fake_dispatcher.EngineDecision = type("EngineDecision", (), {})
    fake_dispatcher.resolve_engine_decision = lambda *args, **kwargs: None

    fake_objective_service = types.ModuleType(
        "ins_pricing.modelling.bayesopt.runtime.objective_service"
    )
    fake_objective_service.ObjectiveService = type("ObjectiveService", (), {})

    sys.modules["ins_pricing"] = fake_root
    sys.modules["ins_pricing.modelling"] = fake_modelling
    sys.modules["ins_pricing.modelling.bayesopt"] = fake_bayesopt
    sys.modules["ins_pricing.modelling.bayesopt.runtime"] = fake_runtime
    sys.modules["ins_pricing.utils"] = fake_utils
    sys.modules["ins_pricing.utils.io"] = fake_utils_io
    sys.modules["ins_pricing.modelling.bayesopt.artifacts"] = fake_artifacts
    sys.modules["ins_pricing.modelling.bayesopt.runtime.dispatcher"] = fake_dispatcher
    sys.modules[
        "ins_pricing.modelling.bayesopt.runtime.objective_service"
    ] = fake_objective_service

    fake_root.modelling = fake_modelling
    fake_root.utils = fake_utils
    fake_modelling.bayesopt = fake_bayesopt
    fake_bayesopt.runtime = fake_runtime

    return _load_module("test_core_training_namespace_loaded", MODULE_PATH)


def test_runtime_namespace_uses_base_artifact_names():
    module = _load_core_training_module()

    class _Dummy(module.BayesOptTrainingMixin):
        def __init__(self):
            self.config = types.SimpleNamespace()

    trainer = _Dummy()

    assert trainer._snapshot_tags("xgb") == ["xgb_best"]
    assert trainer._best_params_labels("xgb", "xgboost") == ["xgboost"]


def test_glm_runtime_path_skips_data_registry_initialization():
    module = _load_core_training_module()

    class _Dummy(module.BayesOptTrainingMixin):
        def __init__(self):
            self.config = types.SimpleNamespace()
            self.trainers = {"glm": object()}
            self._objective_service = None

        def _optimize_model_impl(self, model_key: str, max_evals: int = 100):
            return (model_key, max_evals)

    trainer = _Dummy()
    decision = types.SimpleNamespace(supported=True, reason="enabled")

    result = trainer._optimize_model_with_runtime("glm", 7, decision=decision)

    assert result == ("glm", 7)
    assert trainer._objective_service is None


def test_runtime_path_defers_objective_service_initialization_for_non_glm():
    module = _load_core_training_module()

    class _Dummy(module.BayesOptTrainingMixin):
        def __init__(self):
            self.config = types.SimpleNamespace()
            self.trainers = {"xgb": object()}
            self._objective_service = None

        def _optimize_model_impl(self, model_key: str, max_evals: int = 100):
            return (model_key, max_evals)

    trainer = _Dummy()
    decision = types.SimpleNamespace(supported=True, reason="enabled")

    result = trainer._optimize_model_with_runtime("xgb", 9, decision=decision)

    assert result == ("xgb", 9)
    assert trainer._objective_service is None
