from __future__ import annotations

from typing import Any, Dict, List, Optional

from ins_pricing.modelling.bayesopt.config_preprocess import BayesOptConfig, OutputManager
from ins_pricing.modelling.bayesopt.runtime import (
    TrainerCVPredictionMixin,
    TrainerOptunaMixin,
    TrainerPersistenceMixin,
)
from ins_pricing.modelling.bayesopt.trainers.trainer_context import TrainerContext


class TrainerBase(
    TrainerOptunaMixin,
    TrainerPersistenceMixin,
    TrainerCVPredictionMixin,
):
    def __init__(self, context: TrainerContext, label: str, model_name_prefix: str) -> None:
        self.ctx = context
        self.label = label
        self.model_name_prefix = model_name_prefix
        self.model = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_trial = None
        self.study_name: Optional[str] = None
        self.enable_distributed_optuna: bool = False
        self._distributed_forced_params: Optional[Dict[str, Any]] = None

    def _apply_dataloader_overrides(self, model: Any) -> Any:
        """Apply dataloader-related overrides from config to a model."""
        cfg = getattr(self.ctx, "config", None)
        if cfg is None:
            return model
        model_name = type(model).__name__.lower()
        workers = getattr(cfg, "dataloader_workers", None)
        if workers is not None:
            model.dataloader_workers = int(workers)
        profile = getattr(cfg, "resource_profile", None)
        if profile:
            model.resource_profile = str(profile)
        if hasattr(model, "use_lazy_dataset"):
            if "fttransformer" in model_name:
                lazy_dataset = getattr(cfg, "ft_use_lazy_dataset", None)
            else:
                lazy_dataset = getattr(cfg, "resn_use_lazy_dataset", None)
            if lazy_dataset is not None:
                model.use_lazy_dataset = bool(lazy_dataset)
        if hasattr(model, "predict_batch_size"):
            if "fttransformer" in model_name:
                predict_batch_size = getattr(cfg, "ft_predict_batch_size", None)
            else:
                predict_batch_size = getattr(cfg, "resn_predict_batch_size", None)
            if predict_batch_size is not None:
                model.predict_batch_size = max(1, int(predict_batch_size))
        if "graphneuralnet" in model_name:
            if hasattr(model, "max_fit_rows"):
                max_fit_rows = getattr(cfg, "gnn_max_fit_rows", None)
                if max_fit_rows is not None:
                    model.max_fit_rows = max(1, int(max_fit_rows))
            if hasattr(model, "max_predict_rows"):
                max_predict_rows = getattr(cfg, "gnn_max_predict_rows", None)
                if max_predict_rows is not None:
                    model.max_predict_rows = max(1, int(max_predict_rows))
            if hasattr(model, "predict_chunk_rows"):
                predict_chunk_rows = getattr(cfg, "gnn_predict_chunk_rows", None)
                if predict_chunk_rows is not None:
                    model.predict_chunk_rows = max(1, int(predict_chunk_rows))
        return model

    def _export_preprocess_artifacts(self) -> Dict[str, Any]:
        dummy_columns: List[str] = []
        if getattr(self.ctx, "train_oht_data", None) is not None:
            dummy_columns = list(self.ctx.train_oht_data.columns)
        return {
            "factor_nmes": list(getattr(self.ctx, "factor_nmes", []) or []),
            "cate_list": list(getattr(self.ctx, "cate_list", []) or []),
            "num_features": list(getattr(self.ctx, "num_features", []) or []),
            "var_nmes": list(getattr(self.ctx, "var_nmes", []) or []),
            "cat_categories": dict(getattr(self.ctx, "cat_categories_for_shap", {}) or {}),
            "dummy_columns": dummy_columns,
            "numeric_scalers": dict(getattr(self.ctx, "numeric_scalers", {}) or {}),
            "weight_nme": str(getattr(self.ctx, "weight_nme", "")),
            "resp_nme": str(getattr(self.ctx, "resp_nme", "")),
            "binary_resp_nme": getattr(self.ctx, "binary_resp_nme", None),
            "drop_first": True,
        }

    @property
    def config(self) -> BayesOptConfig:
        return self.ctx.config

    @property
    def output(self) -> OutputManager:
        return self.ctx.output_manager

    def _get_model_filename(self) -> str:
        ext = 'pkl' if self.label in ['Xgboost', 'GLM'] else 'pth'
        return f'01_{self.ctx.model_nme}_{self.model_name_prefix}.{ext}'

    def train(self) -> None:
        raise NotImplementedError
