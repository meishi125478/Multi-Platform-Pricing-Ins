from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
import types

# Keep imports lazy to avoid hard dependencies when only using lightweight modules.

__all__ = [
    "BayesOptConfig",
    "BayesOptModel",
    "IOUtils",
    "TrainingUtils",
    "free_cuda",
    "bayesopt",
    "plotting",
    "explain",
]

_LAZY_ATTRS = {
    "bayesopt": "ins_pricing.modelling.core.bayesopt",
    "plotting": "ins_pricing.modelling.plotting",
    "explain": "ins_pricing.modelling.explain",
    "BayesOptConfig": "ins_pricing.modelling.core.bayesopt.core",
    "BayesOptModel": "ins_pricing.modelling.core.bayesopt.core",
    "IOUtils": "ins_pricing.modelling.core.bayesopt.utils",
    "TrainingUtils": "ins_pricing.modelling.core.bayesopt.utils",
    "free_cuda": "ins_pricing.modelling.core.bayesopt.utils",
}

_LAZY_SUBMODULES = {
    "bayesopt": "ins_pricing.modelling.core.bayesopt",
    "BayesOpt": "ins_pricing.modelling.core.BayesOpt",
    "evaluation": "ins_pricing.modelling.core.evaluation",
    "cli": "ins_pricing.cli",
}

_PACKAGE_PATHS = {
    "bayesopt": Path(__file__).resolve().parent / "core" / "bayesopt",
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
    proxy = _lazy_module(module_name, target, _PACKAGE_PATHS.get(alias))
    sys.modules[module_name] = proxy
    globals()[alias] = proxy


for _alias, _target in _LAZY_SUBMODULES.items():
    _install_proxy(_alias, _target)


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target)
    if name in {"bayesopt", "plotting", "explain"}:
        value = module
    else:
        value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
