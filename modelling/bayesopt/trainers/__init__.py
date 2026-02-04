"""Trainer implementations split by model type."""
from __future__ import annotations

from ins_pricing.modelling.bayesopt.trainers.trainer_base import TrainerBase
from ins_pricing.modelling.bayesopt.trainers.trainer_ft import FTTrainer
from ins_pricing.modelling.bayesopt.trainers.trainer_glm import GLMTrainer
from ins_pricing.modelling.bayesopt.trainers.trainer_gnn import GNNTrainer
from ins_pricing.modelling.bayesopt.trainers.trainer_resn import ResNetTrainer
from ins_pricing.modelling.bayesopt.trainers.trainer_xgb import XGBTrainer, _xgb_cuda_available

__all__ = [
    "TrainerBase",
    "FTTrainer",
    "GLMTrainer",
    "GNNTrainer",
    "ResNetTrainer",
    "XGBTrainer",
    "_xgb_cuda_available",
]
