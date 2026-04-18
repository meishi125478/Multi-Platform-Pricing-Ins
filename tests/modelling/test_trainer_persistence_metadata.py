from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("joblib")


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "modelling" / "bayesopt" / "runtime" / "trainer_persistence.py"


def _load_trainer_persistence_module(monkeypatch: pytest.MonkeyPatch):
    fake_root = types.ModuleType("ins_pricing")
    fake_root.__path__ = [str(REPO_ROOT)]

    fake_modelling = types.ModuleType("ins_pricing.modelling")
    fake_modelling.__path__ = [str(REPO_ROOT / "modelling")]

    fake_bayesopt = types.ModuleType("ins_pricing.modelling.bayesopt")
    fake_bayesopt.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt")]

    fake_utils = types.ModuleType("ins_pricing.utils")
    fake_utils.DeviceManager = type("DeviceManager", (), {})
    fake_utils.get_logger = lambda _name: None
    fake_utils.log_print = lambda *_args, **_kwargs: None

    fake_model_loading = types.ModuleType("ins_pricing.utils.model_loading")
    fake_model_loading.load_torch_payload = lambda *_args, **_kwargs: None

    fake_checkpoints = types.ModuleType("ins_pricing.modelling.bayesopt.checkpoints")
    fake_checkpoints.rebuild_ft_model_from_payload = lambda *args, **kwargs: (None, None, "raw")
    fake_checkpoints.rebuild_resn_model_from_payload = lambda *args, **kwargs: (None, None)
    fake_checkpoints.serialize_ft_model_config = lambda _model: {}

    fake_torch = types.ModuleType("torch")
    fake_torch.save = lambda *_args, **_kwargs: None
    fake_torch.load = lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, "ins_pricing", fake_root)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling", fake_modelling)
    monkeypatch.setitem(sys.modules, "ins_pricing.modelling.bayesopt", fake_bayesopt)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "ins_pricing.utils.model_loading", fake_model_loading)
    monkeypatch.setitem(
        sys.modules,
        "ins_pricing.modelling.bayesopt.checkpoints",
        fake_checkpoints,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    spec = importlib.util.spec_from_file_location(
        "test_trainer_persistence_loaded",
        str(MODULE_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_trainer(module):
    class _DummyTrainer(module.TrainerPersistenceMixin):
        def __init__(self) -> None:
            self.label = "Xgboost"
            self.model = None
            self.best_params = None
            self.ctx = types.SimpleNamespace()
            self.config = types.SimpleNamespace()
            self.output = types.SimpleNamespace()

    return _DummyTrainer()


def test_build_artifact_metadata_returns_expected_keys_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_trainer_persistence_module(monkeypatch)
    trainer = _make_trainer(module)

    metadata = trainer._build_artifact_metadata("xgb")

    assert metadata == {"model_family": "xgb"}


def test_payload_signature_captures_metadata_fields_and_best_param_keys_deterministically(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_trainer_persistence_module(monkeypatch)

    payload_a = {
        "model_family": "xgb",
        "preprocess_artifacts": {"scaler": "noop"},
        "state_dict": {"layer": [1, 2, 3]},
        "model_config": {"depth": 3},
        "best_params": {"zeta": 1, "alpha": 2},
    }
    payload_b = {
        "model_family": "xgb",
        "state_dict": {"layer": [1, 2, 3]},
        "preprocess_artifacts": {"scaler": "noop"},
        "model_config": {"depth": 3},
        "best_params": {"alpha": 2, "zeta": 1},
    }

    signature_a = module.TrainerPersistenceMixin._payload_signature(payload_a)
    signature_b = module.TrainerPersistenceMixin._payload_signature(payload_b)

    assert signature_a == signature_b
    assert signature_a["model_family"] == "xgb"
    assert signature_a["has_preprocess_artifacts"] is True
    assert signature_a["has_state_dict"] is True
    assert signature_a["has_model_config"] is True
    assert signature_a["has_model"] is False
    assert signature_a["best_param_keys"] == ("alpha", "zeta")


def _touch_xgb_checkpoint_file(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "01_demo_Xgboost.pkl"
    checkpoint.write_bytes(b"checkpoint")
    return checkpoint


def test_load_raises_when_checkpoint_is_broken(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_trainer_persistence_module(monkeypatch)
    checkpoint = _touch_xgb_checkpoint_file(tmp_path)
    calls: list[str] = []

    def _fake_joblib_load(path):
        calls.append(Path(path).name)
        if Path(path) == checkpoint:
            raise ValueError("broken checkpoint")
        return {"model": "loaded-model"}

    monkeypatch.setattr(module.joblib, "load", _fake_joblib_load)
    trainer = _make_trainer(module)
    trainer.output = types.SimpleNamespace(model_path=lambda filename: str(tmp_path / "model" / filename))
    trainer._get_model_filename = lambda: "01_demo_Xgboost.pkl"

    with pytest.raises(ValueError, match="broken checkpoint"):
        trainer.load()

    assert calls == [checkpoint.name]
