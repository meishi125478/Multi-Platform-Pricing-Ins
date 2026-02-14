from __future__ import annotations

from importlib import import_module

__all__ = [
    "psi_report",
    "classification_metrics",
    "group_metrics",
    "loss_ratio",
    "metrics_report",
    "regression_metrics",
    "batch_score",
    "apply_preprocess_artifacts",
    "load_preprocess_artifacts",
    "prepare_raw_features",
    "SavedModelPredictor",
    "Predictor",
    "ModelSpec",
    "PredictorRegistry",
    "register_model_loader",
    "load_predictor",
    "load_best_params",
    "load_predictor_from_config",
    "load_saved_model",
    "predict_from_config",
]

_LAZY_ATTRS = {
    "psi_report": ("ins_pricing.production.drift", "psi_report"),
    "classification_metrics": ("ins_pricing.production.monitoring", "classification_metrics"),
    "group_metrics": ("ins_pricing.production.monitoring", "group_metrics"),
    "loss_ratio": ("ins_pricing.production.monitoring", "loss_ratio"),
    "metrics_report": ("ins_pricing.production.monitoring", "metrics_report"),
    "regression_metrics": ("ins_pricing.production.monitoring", "regression_metrics"),
    "batch_score": ("ins_pricing.production.scoring", "batch_score"),
    "apply_preprocess_artifacts": ("ins_pricing.production.preprocess", "apply_preprocess_artifacts"),
    "load_preprocess_artifacts": ("ins_pricing.production.preprocess", "load_preprocess_artifacts"),
    "prepare_raw_features": ("ins_pricing.production.preprocess", "prepare_raw_features"),
    "SavedModelPredictor": ("ins_pricing.production.inference", "SavedModelPredictor"),
    "Predictor": ("ins_pricing.production.inference", "Predictor"),
    "ModelSpec": ("ins_pricing.production.inference", "ModelSpec"),
    "PredictorRegistry": ("ins_pricing.production.inference", "PredictorRegistry"),
    "register_model_loader": ("ins_pricing.production.inference", "register_model_loader"),
    "load_predictor": ("ins_pricing.production.inference", "load_predictor"),
    "load_best_params": ("ins_pricing.production.inference", "load_best_params"),
    "load_predictor_from_config": ("ins_pricing.production.inference", "load_predictor_from_config"),
    "load_saved_model": ("ins_pricing.production.inference", "load_saved_model"),
    "predict_from_config": ("ins_pricing.production.inference", "predict_from_config"),
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
