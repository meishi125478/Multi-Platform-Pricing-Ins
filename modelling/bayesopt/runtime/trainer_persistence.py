from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import torch

from ins_pricing.modelling.bayesopt.checkpoints import (
    rebuild_ft_model_from_payload,
    rebuild_resn_model_from_payload,
    serialize_ft_model_config,
)
from ins_pricing.utils.model_loading import load_torch_payload
from ins_pricing.utils import DeviceManager, get_logger, log_print

_logger = get_logger("ins_pricing.trainer")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class TrainerPersistenceMixin:
    _LABEL_TO_MODEL_KEY = {
        "Xgboost": "xgb",
        "GLM": "glm",
        "ResNet": "resn",
        "ResNetClassifier": "resn",
        "FTTransformer": "ft",
        "FTTransformerClassifier": "ft",
        "GNN": "gnn",
        "GNNClassifier": "gnn",
    }

    def _resolve_model_key(self) -> str:
        return self._LABEL_TO_MODEL_KEY.get(self.label, self.label.lower())

    def _build_artifact_metadata(self, model_key: str) -> Dict[str, Any]:
        return {
            "model_family": model_key,
        }

    @staticmethod
    def _payload_signature(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"payload_type": type(payload).__name__}
        best_params = payload.get("best_params")
        best_param_keys = None
        if isinstance(best_params, dict):
            best_param_keys = tuple(sorted(str(k) for k in best_params.keys()))
        return {
            "model_family": payload.get("model_family"),
            "has_preprocess_artifacts": "preprocess_artifacts" in payload,
            "has_state_dict": "state_dict" in payload,
            "has_model": "model" in payload,
            "has_model_config": "model_config" in payload,
            "best_param_keys": best_param_keys,
        }

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
        model, best_params, kind = rebuild_ft_model_from_payload(payload=loaded)
        if kind == "raw":
            _log(f"[load] Warning: Unknown model format in {path}")
            return
        if best_params is not None:
            sanitize_fn = getattr(self, "_sanitize_best_params", None)
            if callable(sanitize_fn):
                self.best_params = dict(
                    sanitize_fn(dict(best_params), context="checkpoint_load") or {}
                )
            else:
                self.best_params = best_params
        if model is not None:
            self._move_to_device(model)
        self.model = model

    def save(self) -> None:
        if self.model is None:
            _log(f"[save] Warning: No model to save for {self.label}")
            return

        model_key = self._resolve_model_key()
        metadata = self._build_artifact_metadata(model_key)
        primary_path = self.output.model_path(self._get_model_filename())
        if self.label in ['Xgboost', 'GLM']:
            payload = {
                "model": self.model,
                "preprocess_artifacts": self._export_preprocess_artifacts(),
            }
            payload.update(metadata)
            joblib.dump(payload, primary_path)
        else:
            payload = {
                "preprocess_artifacts": self._export_preprocess_artifacts(),
            }
            payload.update(metadata)
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
            torch.save(payload, primary_path)

    def load(self) -> None:
        primary_path = self.output.model_path(self._get_model_filename())
        if not os.path.exists(primary_path):
            _log(f"[load] Warning: Model file not found: {primary_path}")
            return

        if self.label in ['Xgboost', 'GLM']:
            loaded = joblib.load(primary_path)
            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded.get("model")
            else:
                self.model = loaded
            return

        if self.label == 'ResNet' or self.label == 'ResNetClassifier':
            payload = load_torch_payload(primary_path, map_location='cpu', weights_only=True)
            model_builder = getattr(self, "_build_model", None)
            if not callable(model_builder):
                _log(
                    f"[load] Warning: {self.label} checkpoint found but model builder is unavailable."
                )
                return
            resn_loaded, resolved_params = rebuild_resn_model_from_payload(
                payload=payload,
                model_builder=model_builder,
            )
            self.best_params = resolved_params
            if resn_loaded is not None:
                self._move_to_device(resn_loaded)
            self.model = resn_loaded
            return

        loaded = load_torch_payload(primary_path, map_location='cpu', weights_only=True)
        if isinstance(loaded, dict):
            self._load_ft_checkpoint_payload(loaded, primary_path)
        else:
            if loaded is not None:
                self._move_to_device(loaded)
            self.model = loaded

    def _move_to_device(self, model_obj):
        """Move model to the best available device using shared DeviceManager."""
        DeviceManager.move_to_device(model_obj)


__all__ = ["TrainerPersistenceMixin"]
