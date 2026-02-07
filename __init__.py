from __future__ import annotations

from importlib import import_module

_ROOT_SUBPACKAGES = {
    "modelling": "ins_pricing.modelling",
    "pricing": "ins_pricing.pricing",
    "production": "ins_pricing.production",
    "governance": "ins_pricing.governance",
    "reporting": "ins_pricing.reporting",
}

__all__ = sorted(set(_ROOT_SUBPACKAGES))


def __getattr__(name: str):
    target = _ROOT_SUBPACKAGES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
