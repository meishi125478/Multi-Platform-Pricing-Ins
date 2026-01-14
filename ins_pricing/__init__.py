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
    "IOUtils",
    "TrainingUtils",
    "free_cuda",
}

_LAZY_SUBMODULES = {
    "bayesopt": "ins_pricing.modelling.core.bayesopt",
    "plotting": "ins_pricing.modelling.plotting",
    "explain": "ins_pricing.modelling.explain",
    "BayesOpt": "ins_pricing.modelling.core.BayesOpt",
}

_PACKAGE_PATHS = {
    "bayesopt": Path(__file__).resolve().parent / "modelling" / "core" / "bayesopt",
    "plotting": Path(__file__).resolve().parent / "modelling" / "plotting",
    "explain": Path(__file__).resolve().parent / "modelling" / "explain",
}

__all__ = sorted(
    set(_ROOT_SUBPACKAGES)
    | set(_MODELLING_EXPORTS)
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
