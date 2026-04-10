from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import types

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TYPES_PATH = REPO_ROOT / "modelling" / "bayesopt" / "runtime" / "types.py"
REGISTRY_PATH = REPO_ROOT / "modelling" / "bayesopt" / "runtime" / "data_registry.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_data_registry_class():
    fake_root = types.ModuleType("ins_pricing")
    fake_root.__path__ = [str(REPO_ROOT)]

    fake_modelling = types.ModuleType("ins_pricing.modelling")
    fake_modelling.__path__ = [str(REPO_ROOT / "modelling")]

    fake_bayesopt = types.ModuleType("ins_pricing.modelling.bayesopt")
    fake_bayesopt.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt")]

    fake_runtime = types.ModuleType("ins_pricing.modelling.bayesopt.runtime")
    fake_runtime.__path__ = [str(REPO_ROOT / "modelling" / "bayesopt" / "runtime")]

    sys.modules["ins_pricing"] = fake_root
    sys.modules["ins_pricing.modelling"] = fake_modelling
    sys.modules["ins_pricing.modelling.bayesopt"] = fake_bayesopt
    sys.modules["ins_pricing.modelling.bayesopt.runtime"] = fake_runtime

    _load_module("ins_pricing.modelling.bayesopt.runtime.types", TYPES_PATH)
    registry_module = _load_module("ins_pricing.modelling.bayesopt.runtime.data_registry", REGISTRY_PATH)
    return registry_module.DataRegistry


DataRegistry = _load_data_registry_class()


def test_data_registry_preserves_row_id_and_source_index():
    train = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [10.0, 11.0, 12.0]},
        index=pd.Index([101, 203, 305], name="orig_id"),
    )
    test = pd.DataFrame(
        {"x": [4.0, 5.0], "y": [13.0, 14.0]},
        index=pd.Index([501, 607], name="orig_id"),
    )
    ctx = types.SimpleNamespace(train_data=train, test_data=test)

    registry = DataRegistry.from_context(ctx)
    assert np.array_equal(registry.row_store.train_row_id, np.array([0, 1, 2], dtype=np.int64))
    assert np.array_equal(registry.row_store.test_row_id, np.array([0, 1], dtype=np.int64))
    assert np.array_equal(registry.row_store.train_source_index, np.array([101, 203, 305]))
    assert np.array_equal(registry.row_store.test_source_index, np.array([501, 607]))


def test_data_registry_matrix_cache_reuses_same_array_instance():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    test = pd.DataFrame({"a": [5.0], "b": [6.0]})
    ctx = types.SimpleNamespace(train_data=train, test_data=test)

    registry = DataRegistry.from_context(ctx)
    arr1 = registry.train_matrix(["a", "b"])
    arr2 = registry.train_matrix(["a", "b"])
    assert arr1 is arr2


def test_data_registry_evicts_oldest_cache_entry_when_limit_exceeded():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    test = pd.DataFrame({"a": [7.0], "b": [8.0], "c": [9.0]})
    ctx = types.SimpleNamespace(train_data=train, test_data=test)

    registry = DataRegistry.from_context(ctx)
    registry.max_cache_entries = 1

    arr1 = registry.train_matrix(["a"])
    arr2 = registry.train_matrix(["b"])
    arr1_again = registry.train_matrix(["a"])

    assert arr1 is not arr2
    assert arr1_again is not arr1
