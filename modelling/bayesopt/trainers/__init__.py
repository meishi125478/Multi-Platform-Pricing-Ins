"""Trainer implementations split by model type.

Keep this package initializer lazy to avoid import cycles between runtime
mixins and trainer modules.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TrainerBase": "ins_pricing.modelling.bayesopt.trainers.trainer_base",
    "TrainerContext": "ins_pricing.modelling.bayesopt.trainers.trainer_context",
    "FTTrainer": "ins_pricing.modelling.bayesopt.trainers.trainer_ft",
    "GLMTrainer": "ins_pricing.modelling.bayesopt.trainers.trainer_glm",
    "GNNTrainer": "ins_pricing.modelling.bayesopt.trainers.trainer_gnn",
    "ResNetTrainer": "ins_pricing.modelling.bayesopt.trainers.trainer_resn",
    "XGBTrainer": "ins_pricing.modelling.bayesopt.trainers.trainer_xgb",
    "_xgb_cuda_available": "ins_pricing.modelling.bayesopt.trainers.trainer_xgb",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(target)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
