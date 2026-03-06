"""BayesOpt subpackage (split from monolithic BayesOpt.py)."""

from __future__ import annotations

from ins_pricing.utils.torch_compat import disable_torch_dynamo_if_requested

disable_torch_dynamo_if_requested()

from ins_pricing.modelling.bayesopt.config_runtime import OutputManager, VersionManager
from ins_pricing.modelling.bayesopt.config_schema import BayesOptConfig
from ins_pricing.modelling.bayesopt.dataset_preprocessor import DatasetPreprocessor
from ins_pricing.modelling.bayesopt.core import BayesOptModel
from ins_pricing.modelling.bayesopt.models import (
    FeatureTokenizer,
    FTTransformerCore,
    FTTransformerSklearn,
    GraphNeuralNetSklearn,
    MaskedTabularDataset,
    ResBlock,
    ResNetSequential,
    ResNetSklearn,
    ScaledTransformerEncoderLayer,
    SimpleGraphLayer,
    SimpleGNN,
    TabularDataset,
)
from ins_pricing.modelling.bayesopt.trainers import (
    FTTrainer,
    GLMTrainer,
    GNNTrainer,
    ResNetTrainer,
    TrainerBase,
    XGBTrainer,
    _xgb_cuda_available,
)
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
