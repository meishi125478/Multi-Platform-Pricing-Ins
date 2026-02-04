from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
import types

_ROOT_SUBPACKAGES = {
    "modelling": "ins_pricing.modelling",
    "pricing": "ins_pricing.pricing",
    "production": "ins_pricing.production",
    "governance": "ins_pricing.governance",
    "reporting": "ins_pricing.reporting",
}

_MODELLING_EXPORTS = {
    "BayesOptConfig",
    "BayesOptModel",
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

_LEGACY_EXPORTS = {
    "IOUtils": "ins_pricing.utils",
    "DeviceManager": "ins_pricing.utils",
    "GPUMemoryManager": "ins_pricing.utils",
    "MetricFactory": "ins_pricing.utils",
    "EPS": "ins_pricing.utils",
    "set_global_seed": "ins_pricing.utils",
    "compute_batch_size": "ins_pricing.utils",
    "tweedie_loss": "ins_pricing.utils",
    "infer_factor_and_cate_list": "ins_pricing.utils",
    "DistributedUtils": "ins_pricing.modelling.bayesopt.utils",
    "TrainingUtils": "ins_pricing.modelling.bayesopt.utils",
    "free_cuda": "ins_pricing.modelling.bayesopt.utils",
    "TorchTrainerMixin": "ins_pricing.modelling.bayesopt.utils",
}

_LAZY_SUBMODULES = {
    "bayesopt": "ins_pricing.modelling.bayesopt",
    "plotting": "ins_pricing.modelling.plotting",
    "explain": "ins_pricing.modelling.explain",
}

_PACKAGE_PATHS = {
    "bayesopt": Path(__file__).resolve().parent / "modelling" / "bayesopt",
    "plotting": Path(__file__).resolve().parent / "modelling" / "plotting",
    "explain": Path(__file__).resolve().parent / "modelling" / "explain",
}

__all__ = sorted(
    set(_ROOT_SUBPACKAGES)
    | set(_MODELLING_EXPORTS)
    | set(_BAYESOPT_EXPORTS)
    | set(_LEGACY_EXPORTS)
    | set(_LAZY_SUBMODULES)
)


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
    proxy = _lazy_module(module_name, target, _PACKAGE_PATHS.get(alias))
    sys.modules[module_name] = proxy
    globals()[alias] = proxy


for _alias, _target in _LAZY_SUBMODULES.items():
    _install_proxy(_alias, _target)


def __getattr__(name: str):
    if name in _ROOT_SUBPACKAGES:
        module = import_module(_ROOT_SUBPACKAGES[name])
        globals()[name] = module
        return module
    if name in _MODELLING_EXPORTS:
        module = import_module("ins_pricing.modelling")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _BAYESOPT_EXPORTS:
        module = import_module("ins_pricing.modelling.bayesopt")
        value = getattr(module, name)
        globals()[name] = value
        return value
    legacy_module = _LEGACY_EXPORTS.get(name)
    if legacy_module:
        module = import_module(legacy_module)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
