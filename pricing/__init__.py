from __future__ import annotations

from importlib import import_module

__all__ = [
    "apply_calibration",
    "fit_calibration_factor",
    "detect_leakage",
    "profile_columns",
    "validate_schema",
    "aggregate_policy_level",
    "build_frequency_severity",
    "compute_exposure",
    "bin_numeric",
    "build_factor_table",
    "population_stability_index",
    "psi_report",
    "RateTable",
    "apply_factor_tables",
    "compute_base_rate",
    "rate_premium",
]

_LAZY_ATTRS = {
    "apply_calibration": ("ins_pricing.pricing.calibration", "apply_calibration"),
    "fit_calibration_factor": ("ins_pricing.pricing.calibration", "fit_calibration_factor"),
    "detect_leakage": ("ins_pricing.pricing.data_quality", "detect_leakage"),
    "profile_columns": ("ins_pricing.pricing.data_quality", "profile_columns"),
    "validate_schema": ("ins_pricing.pricing.data_quality", "validate_schema"),
    "aggregate_policy_level": ("ins_pricing.pricing.exposure", "aggregate_policy_level"),
    "build_frequency_severity": ("ins_pricing.pricing.exposure", "build_frequency_severity"),
    "compute_exposure": ("ins_pricing.pricing.exposure", "compute_exposure"),
    "bin_numeric": ("ins_pricing.pricing.factors", "bin_numeric"),
    "build_factor_table": ("ins_pricing.pricing.factors", "build_factor_table"),
    "population_stability_index": ("ins_pricing.pricing.monitoring", "population_stability_index"),
    "psi_report": ("ins_pricing.pricing.monitoring", "psi_report"),
    "RateTable": ("ins_pricing.pricing.rate_table", "RateTable"),
    "apply_factor_tables": ("ins_pricing.pricing.rate_table", "apply_factor_tables"),
    "compute_base_rate": ("ins_pricing.pricing.rate_table", "compute_base_rate"),
    "rate_premium": ("ins_pricing.pricing.rate_table", "rate_premium"),
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
