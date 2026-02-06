from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import torch

from ins_pricing.modelling.bayesopt.checkpoints import (
    rebuild_ft_model_from_checkpoint,
    serialize_ft_model_config,
)
from ins_pricing.utils import DeviceManager, get_logger, log_print
from ins_pricing.utils.torch_compat import torch_load

_logger = get_logger("ins_pricing.trainer")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class TrainerPersistenceMixin:
    def _unwrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Unwrap DDP or DataParallel wrapper to get the base module."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(module, (DDP, torch.nn.DataParallel)):
            return module.module
        return module

    def _build_ft_checkpoint_payload(self) -> Dict[str, Any]:
        ft = self._unwrap_module(self.model.ft)
        ft_cpu = ft.to("cpu")
        return {
            "state_dict": ft_cpu.state_dict(),
            "best_params": dict(self.best_params or {}),
            "model_config": serialize_ft_model_config(self.model),
        }

    def _load_ft_checkpoint_payload(self, loaded: Dict[str, Any], path: str) -> None:
        if "state_dict" in loaded and "model_config" in loaded:
            model = rebuild_ft_model_from_checkpoint(
                state_dict=loaded.get("state_dict"),
                model_config=loaded.get("model_config", {}),
            )
            self.best_params = loaded.get("best_params", {})
            self._move_to_device(model)
            self.model = model
            return
        if "model" in loaded:
            loaded_model = loaded.get("model")
            if loaded_model is not None:
                self._move_to_device(loaded_model)
            self.model = loaded_model
            return
        _log(f"[load] Warning: Unknown model format in {path}")

    def save(self) -> None:
        if self.model is None:
            _log(f"[save] Warning: No model to save for {self.label}")
            return

        path = self.output.model_path(self._get_model_filename())
        if self.label in ['Xgboost', 'GLM']:
            payload = {
                "model": self.model,
                "preprocess_artifacts": self._export_preprocess_artifacts(),
            }
            joblib.dump(payload, path)
        else:
            payload = {
                "preprocess_artifacts": self._export_preprocess_artifacts(),
            }
            if hasattr(self.model, 'resnet'):
                resnet = self._unwrap_module(self.model.resnet)
                resnet_cpu = resnet.to("cpu")
                payload["state_dict"] = resnet_cpu.state_dict()
                payload["best_params"] = dict(self.best_params or {})
            elif hasattr(self.model, 'ft'):
                payload.update(self._build_ft_checkpoint_payload())
            else:
                if hasattr(self.model, 'to'):
                    self.model.to("cpu")
                payload["model"] = self.model
            torch.save(payload, path)

    def load(self) -> None:
        path = self.output.model_path(self._get_model_filename())
        if not os.path.exists(path):
            _log(f"[load] Warning: Model file not found: {path}")
            return

        if self.label in ['Xgboost', 'GLM']:
            loaded = joblib.load(path)
            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded.get("model")
            else:
                self.model = loaded
        else:
            if self.label == 'ResNet' or self.label == 'ResNetClassifier':
                pass
            else:
                loaded = torch_load(path, map_location='cpu', weights_only=False)
                if isinstance(loaded, dict):
                    self._load_ft_checkpoint_payload(loaded, path)
                else:
                    if loaded is not None:
                        self._move_to_device(loaded)
                    self.model = loaded

    def _move_to_device(self, model_obj):
        """Move model to the best available device using shared DeviceManager."""
        DeviceManager.move_to_device(model_obj)


__all__ = ["TrainerPersistenceMixin"]
