from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ins_pricing.modelling.bayesopt.config_runtime import OutputManager
from ins_pricing.modelling.bayesopt.config_schema import BayesOptConfig
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

    def _get_search_space_config(self, field_name: str) -> Dict[str, Any]:
        """Read search-space config from BayesOptConfig."""
        raw = getattr(self.config, field_name, None)
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{field_name} must be a JSON object when provided.")
        return raw

    def _sample_param(
        self,
        trial: Any,
        *,
        model_key: str,
        param_name: str,
        default_sampler: Callable[[Any], Any],
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Sample one Optuna parameter with optional config override."""
        if trial is None:
            raise RuntimeError("Missing Optuna trial for parameter sampling.")
        if not search_space or param_name not in search_space:
            return default_sampler(trial)
        spec = search_space[param_name]
        return self._sample_from_spec(
            trial=trial,
            model_key=model_key,
            param_name=param_name,
            spec=spec,
        )

    def _sample_from_spec(
        self,
        *,
        trial: Any,
        model_key: str,
        param_name: str,
        spec: Any,
    ) -> Any:
        """Sample/resolve one parameter from a JSON-friendly search spec."""
        if isinstance(spec, (list, tuple)):
            choices = list(spec)
            if not choices:
                raise ValueError(
                    f"{model_key}_search_space.{param_name} categorical choices cannot be empty."
                )
            return trial.suggest_categorical(param_name, choices)

        if not isinstance(spec, dict):
            # Scalar value -> fixed value.
            return spec

        if "value" in spec:
            return spec["value"]

        spec_type = str(spec.get("type", "")).strip().lower()
        if not spec_type and "choices" in spec:
            spec_type = "categorical"

        if spec_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"{model_key}_search_space.{param_name}.choices must be a non-empty list."
                )
            return trial.suggest_categorical(param_name, choices)

        low_raw = spec.get("low", spec.get("min"))
        high_raw = spec.get("high", spec.get("max"))
        if low_raw is None or high_raw is None:
            raise ValueError(
                f"{model_key}_search_space.{param_name} must provide low/high (or min/max)."
            )

        if spec_type == "int":
            low = int(low_raw)
            high = int(high_raw)
            step = int(spec.get("step", 1))
            log = bool(spec.get("log", False))
            if step < 1:
                raise ValueError(
                    f"{model_key}_search_space.{param_name}.step must be >= 1 for int params."
                )
            if log and step != 1:
                raise ValueError(
                    f"{model_key}_search_space.{param_name} cannot set both log=true and step!=1."
                )
            return trial.suggest_int(param_name, low, high, step=step, log=log)

        if spec_type == "float":
            low = float(low_raw)
            high = float(high_raw)
            step_raw = spec.get("step")
            step = float(step_raw) if step_raw is not None else None
            log = bool(spec.get("log", False))
            if log and step is not None:
                raise ValueError(
                    f"{model_key}_search_space.{param_name} cannot set both log=true and step."
                )
            kwargs: Dict[str, Any] = {"log": log}
            if step is not None:
                kwargs["step"] = step
            return trial.suggest_float(param_name, low, high, **kwargs)

        raise ValueError(
            f"{model_key}_search_space.{param_name}.type must be one of int/float/categorical."
        )

    def _augment_param_space_with_search_space(
        self,
        *,
        model_key: str,
        param_space: Dict[str, Callable[[Any], Any]],
        search_space: Optional[Dict[str, Any]],
        skip_params: Optional[List[str]] = None,
    ) -> Dict[str, Callable[[Any], Any]]:
        """Merge extra search-space params into a sampled param_space."""
        if not search_space:
            return param_space
        merged = dict(param_space)
        skip_set = set(skip_params or [])
        for param_name, spec in search_space.items():
            if param_name in merged or param_name in skip_set:
                continue
            merged[param_name] = (
                lambda trial, pname=param_name, pspec=spec: self._sample_from_spec(
                    trial=trial,
                    model_key=model_key,
                    param_name=pname,
                    spec=pspec,
                )
            )
        return merged

    def _apply_dataloader_overrides(self, model: Any) -> Any:
        """Apply dataloader-related overrides from config to a model."""
        cfg = getattr(self.ctx, "config", None)
        if cfg is None:
            return model
        model_name = type(model).__name__.lower()
        workers = getattr(cfg, "dataloader_workers", None)
        if workers is not None:
            model.dataloader_workers = int(workers)
        mp_context = getattr(cfg, "dataloader_multiprocessing_context", None)
        if mp_context is not None:
            model.dataloader_multiprocessing_context = str(mp_context)
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
        ohe_feature_names = list(getattr(self.ctx, "ohe_feature_names", []) or [])
        if not ohe_feature_names:
            num_set = set(getattr(self.ctx, "num_features", []) or [])
            ohe_feature_names = [
                col for col in (getattr(self.ctx, "var_nmes", []) or [])
                if col not in num_set
            ]
        return {
            "factor_nmes": list(getattr(self.ctx, "factor_nmes", []) or []),
            "cate_list": list(getattr(self.ctx, "cate_list", []) or []),
            "num_features": list(getattr(self.ctx, "num_features", []) or []),
            "var_nmes": list(getattr(self.ctx, "var_nmes", []) or []),
            "cat_categories": dict(getattr(self.ctx, "cat_categories_for_shap", {}) or {}),
            "ohe_feature_names": ohe_feature_names,
            "dummy_columns": dummy_columns,
            "numeric_scalers": dict(getattr(self.ctx, "numeric_scalers", {}) or {}),
            "weight_nme": str(getattr(self.ctx, "weight_nme", "")),
            "resp_nme": str(getattr(self.ctx, "resp_nme", "")),
            "binary_resp_nme": getattr(self.ctx, "binary_resp_nme", None),
            "drop_first": True,
            "oht_sparse_csr": bool(getattr(self.ctx, "oht_sparse_csr", False)),
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
