"""Shared utilities for the ins_pricing package.

This module intentionally keeps imports lazy so that lightweight code paths
(frontend helpers, config parsing, etc.) do not pull heavy optional runtime
dependencies such as torch at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Logging
    "get_logger": ("ins_pricing.utils.logging", "get_logger"),
    "configure_logging": ("ins_pricing.utils.logging", "configure_logging"),
    "log_print": ("ins_pricing.utils.logging", "log_print"),
    # Metrics
    "psi_numeric": ("ins_pricing.utils.metrics", "psi_numeric"),
    "psi_categorical": ("ins_pricing.utils.metrics", "psi_categorical"),
    "population_stability_index": ("ins_pricing.utils.metrics", "population_stability_index"),
    "psi_report": ("ins_pricing.utils.metrics", "psi_report"),
    "MetricFactory": ("ins_pricing.utils.metrics", "MetricFactory"),
    # Numerics
    "EPS": ("ins_pricing.utils.numerics", "EPS"),
    "set_global_seed": ("ins_pricing.utils.numerics", "set_global_seed"),
    "compute_batch_size": ("ins_pricing.utils.numerics", "compute_batch_size"),
    "safe_divide": ("ins_pricing.utils.numerics", "safe_divide"),
    "tweedie_loss": ("ins_pricing.utils.numerics", "tweedie_loss"),
    # Features
    "infer_factor_and_cate_list": ("ins_pricing.utils.features", "infer_factor_and_cate_list"),
    # IO
    "IOUtils": ("ins_pricing.utils.io", "IOUtils"),
    "csv_to_dict": ("ins_pricing.utils.io", "csv_to_dict"),
    "ensure_parent_dir": ("ins_pricing.utils.io", "ensure_parent_dir"),
    # Paths
    "resolve_path": ("ins_pricing.utils.paths", "resolve_path"),
    "resolve_dir_path": ("ins_pricing.utils.paths", "resolve_dir_path"),
    "resolve_data_path": ("ins_pricing.utils.paths", "resolve_data_path"),
    "load_dataset": ("ins_pricing.utils.paths", "load_dataset"),
    "coerce_dataset_types": ("ins_pricing.utils.paths", "coerce_dataset_types"),
    "dedupe_preserve_order": ("ins_pricing.utils.paths", "dedupe_preserve_order"),
    "build_model_names": ("ins_pricing.utils.paths", "build_model_names"),
    "parse_model_pairs": ("ins_pricing.utils.paths", "parse_model_pairs"),
    "fingerprint_file": ("ins_pricing.utils.paths", "fingerprint_file"),
    "PLOT_MODEL_LABELS": ("ins_pricing.utils.paths", "PLOT_MODEL_LABELS"),
    "PYTORCH_TRAINERS": ("ins_pricing.utils.paths", "PYTORCH_TRAINERS"),
    # Device
    "DeviceManager": ("ins_pricing.utils.device", "DeviceManager"),
    "GPUMemoryManager": ("ins_pricing.utils.device", "GPUMemoryManager"),
}

__all__ = sorted(set(_LAZY_ATTRS))


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
