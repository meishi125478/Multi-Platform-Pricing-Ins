"""Trainer implementations split by model type."""
from __future__ import annotations

from .trainer_base import TrainerBase
from .trainer_ft import FTTrainer
from .trainer_glm import GLMTrainer
from .trainer_gnn import GNNTrainer
from .trainer_resn import ResNetTrainer
from .trainer_xgb import XGBTrainer, _xgb_cuda_available

__all__ = [
    "TrainerBase",
    "FTTrainer",
    "GLMTrainer",
    "GNNTrainer",
    "ResNetTrainer",
    "XGBTrainer",
    "_xgb_cuda_available",
]
