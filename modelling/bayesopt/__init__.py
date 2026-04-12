"""BayesOpt subpackage (split from monolithic BayesOpt.py)."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

from ins_pricing.utils.torch_compat import disable_torch_dynamo_if_requested

disable_torch_dynamo_if_requested()

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "BayesOptConfig": ("ins_pricing.modelling.bayesopt.config_schema", "BayesOptConfig"),
    "DatasetPreprocessor": (
        "ins_pricing.modelling.bayesopt.dataset_preprocessor",
        "DatasetPreprocessor",
    ),
    "OutputManager": ("ins_pricing.modelling.bayesopt.config_runtime", "OutputManager"),
    "VersionManager": ("ins_pricing.modelling.bayesopt.config_runtime", "VersionManager"),
    "BayesOptModel": ("ins_pricing.modelling.bayesopt.core", "BayesOptModel"),
    "FeatureTokenizer": ("ins_pricing.modelling.bayesopt.models", "FeatureTokenizer"),
    "FTTransformerCore": ("ins_pricing.modelling.bayesopt.models", "FTTransformerCore"),
    "FTTransformerSklearn": ("ins_pricing.modelling.bayesopt.models", "FTTransformerSklearn"),
    "GraphNeuralNetSklearn": ("ins_pricing.modelling.bayesopt.models", "GraphNeuralNetSklearn"),
    "MaskedTabularDataset": ("ins_pricing.modelling.bayesopt.models", "MaskedTabularDataset"),
    "ResBlock": ("ins_pricing.modelling.bayesopt.models", "ResBlock"),
    "ResNetSequential": ("ins_pricing.modelling.bayesopt.models", "ResNetSequential"),
    "ResNetSklearn": ("ins_pricing.modelling.bayesopt.models", "ResNetSklearn"),
    "ScaledTransformerEncoderLayer": (
        "ins_pricing.modelling.bayesopt.models",
        "ScaledTransformerEncoderLayer",
    ),
    "SimpleGraphLayer": ("ins_pricing.modelling.bayesopt.models", "SimpleGraphLayer"),
    "SimpleGNN": ("ins_pricing.modelling.bayesopt.models", "SimpleGNN"),
    "TabularDataset": ("ins_pricing.modelling.bayesopt.models", "TabularDataset"),
    "FTTrainer": ("ins_pricing.modelling.bayesopt.trainers", "FTTrainer"),
    "GLMTrainer": ("ins_pricing.modelling.bayesopt.trainers", "GLMTrainer"),
    "GNNTrainer": ("ins_pricing.modelling.bayesopt.trainers", "GNNTrainer"),
    "ResNetTrainer": ("ins_pricing.modelling.bayesopt.trainers", "ResNetTrainer"),
    "TrainerBase": ("ins_pricing.modelling.bayesopt.trainers", "TrainerBase"),
    "XGBTrainer": ("ins_pricing.modelling.bayesopt.trainers", "XGBTrainer"),
    "_xgb_cuda_available": ("ins_pricing.modelling.bayesopt.trainers", "_xgb_cuda_available"),
}

__all__ = [
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
]


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals().keys()) + list(__all__)))
