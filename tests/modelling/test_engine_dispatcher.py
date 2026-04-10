from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "modelling" / "bayesopt" / "runtime" / "dispatcher.py"


def _load_dispatcher_module():
    spec = importlib.util.spec_from_file_location("test_engine_dispatcher_loaded", str(MODULE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


resolve_engine_decision = _load_dispatcher_module().resolve_engine_decision


def test_dispatcher_supports_xgb():
    cfg = types.SimpleNamespace()
    decision = resolve_engine_decision(cfg, model_key="xgb", ft_role="model")
    assert decision.supported is True


def test_dispatcher_supports_resn():
    cfg = types.SimpleNamespace()
    decision = resolve_engine_decision(cfg, model_key="resn", ft_role="model")
    assert decision.supported is True


def test_dispatcher_enables_xgb():
    cfg = types.SimpleNamespace()
    decision = resolve_engine_decision(cfg, model_key="xgb", ft_role="model")
    assert decision.supported is True


@pytest.mark.parametrize("ft_role", ["model", "embedding", "unsupervised_embedding"])
def test_dispatcher_supports_ft_roles(ft_role):
    cfg = types.SimpleNamespace()
    decision = resolve_engine_decision(cfg, model_key="ft", ft_role=ft_role)
    assert decision.supported is True


def test_dispatcher_supports_glm():
    cfg = types.SimpleNamespace()
    decision = resolve_engine_decision(cfg, model_key="glm", ft_role="model")
    assert decision.supported is True


def test_dispatcher_supports_gnn():
    cfg = types.SimpleNamespace()
    decision = resolve_engine_decision(cfg, model_key="gnn", ft_role="model")
    assert decision.supported is True


def test_dispatcher_rejects_unknown_model_key():
    cfg = types.SimpleNamespace()
    with pytest.raises(ValueError, match="Unsupported model_key"):
        resolve_engine_decision(cfg, model_key="abc", ft_role="model")
