"""Insurance Pricing Frontend package.

Lazy exports keep optional runtime dependencies isolated from import-time
code paths.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["ConfigBuilder", "TaskRunner", "TrainingRunner", "FTWorkflowHelper"]

_LAZY_ATTRS = {
    "ConfigBuilder": ("ins_pricing.frontend.config_builder", "ConfigBuilder"),
    "TaskRunner": ("ins_pricing.frontend.runner", "TaskRunner"),
    "TrainingRunner": ("ins_pricing.frontend.runner", "TrainingRunner"),
    "FTWorkflowHelper": ("ins_pricing.frontend.ft_workflow", "FTWorkflowHelper"),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
