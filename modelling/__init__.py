from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
import types

# Keep imports lazy to avoid hard dependencies when only using lightweight modules.

__all__ = sorted(
    {
        "BayesOptConfig",
        "BayesOptModel",
        "bayesopt",
        "plotting",
        "explain",
        "evaluation",
        "cli",
    }
)

_LAZY_ATTRS = {
    "bayesopt": "ins_pricing.modelling.bayesopt",
    "plotting": "ins_pricing.modelling.plotting",
    "explain": "ins_pricing.modelling.explain",
    "evaluation": "ins_pricing.modelling.evaluation",
    "BayesOptConfig": "ins_pricing.modelling.bayesopt.core",
    "BayesOptModel": "ins_pricing.modelling.bayesopt.core",
}

_BAYESOPT_EXPORTS = {
    "BayesOptConfig",
    "DatasetPreprocessor",
    "OutputManager",
    "VersionManager",
    "BayesOptModel",
    "FeatureTokenizer",
    "FTTransformerCore",
    "FTTransformerSklearn",
    "GraphNeuralNetSklearn",
    "MaskedTabularDataset",
    "ResBlock",
    "ResNetSequential",
    "ResNetSklearn",
    "ScaledTransformerEncoderLayer",
    "SimpleGraphLayer",
    "SimpleGNN",
    "TabularDataset",
    "FTTrainer",
    "GLMTrainer",
    "GNNTrainer",
    "ResNetTrainer",
    "TrainerBase",
    "XGBTrainer",
    "_xgb_cuda_available",
}

__all__ = sorted(set(__all__) | set(_BAYESOPT_EXPORTS))

_LAZY_SUBMODULES = {
    "bayesopt": "ins_pricing.modelling.bayesopt",
    "cli": "ins_pricing.cli",
}

_PACKAGE_PATHS = {
    "bayesopt": Path(__file__).resolve().parent / "bayesopt",
    "cli": Path(__file__).resolve().parents[1] / "cli",
}


def _lazy_module(name: str, target: str, package_path: Path | None = None) -> types.ModuleType:
    proxy = types.ModuleType(name)
    if package_path is not None:
        proxy.__path__ = [str(package_path)]

    def _load():
        module = import_module(target)
        sys.modules[name] = module
        return module

    def __getattr__(attr: str):
        module = _load()
        return getattr(module, attr)

    def __dir__() -> list[str]:
        module = _load()
        return sorted(set(dir(module)))

    proxy.__getattr__ = __getattr__  # type: ignore[attr-defined]
    proxy.__dir__ = __dir__  # type: ignore[attr-defined]
    return proxy


def _install_proxy(alias: str, target: str) -> None:
    module_name = f"{__name__}.{alias}"
    if module_name in sys.modules:
        return
    # Avoid self-referential proxies (e.g. modelling.bayesopt -> modelling.bayesopt).
    if target == module_name:
        return
    proxy = _lazy_module(module_name, target, _PACKAGE_PATHS.get(alias))
    sys.modules[module_name] = proxy
    globals()[alias] = proxy


for _alias, _target in _LAZY_SUBMODULES.items():
    _install_proxy(_alias, _target)


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target:
        module = import_module(target)
        if name in {"bayesopt", "plotting", "explain", "evaluation"}:
            value = module
        else:
            value = getattr(module, name)
        globals()[name] = value
        return value

    if name in _BAYESOPT_EXPORTS:
        module = import_module("ins_pricing.modelling.bayesopt")
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_BAYESOPT_EXPORTS) | set(globals().keys()))
