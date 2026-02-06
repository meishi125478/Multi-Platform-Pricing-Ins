from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
try:  # statsmodels is optional when GLM inference is not used
    import statsmodels.api as sm
    _SM_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - optional dependency
    sm = None  # type: ignore[assignment]
    _SM_IMPORT_ERROR = exc

from ins_pricing.production.preprocess import (
    apply_preprocess_artifacts,
    load_preprocess_artifacts,
    prepare_raw_features,
)
from ins_pricing.production.scoring import batch_score
from ins_pricing.utils.losses import (
    resolve_effective_loss_name,
    resolve_tweedie_power,
)
from ins_pricing.utils import get_logger, load_dataset

_logger = get_logger("ins_pricing.production.inference")


if TYPE_CHECKING:
    from ins_pricing.modelling.bayesopt.models.model_gnn import GraphNeuralNetSklearn
    from ins_pricing.modelling.bayesopt.models.model_resn import ResNetSklearn


def _torch_load(*args, **kwargs):
    from ins_pricing.utils.torch_compat import torch_load
    return torch_load(*args, **kwargs)

def _get_device_manager():
    from ins_pricing.utils.device import DeviceManager
    return DeviceManager


def _normalize_device(device: Optional[Any]) -> Optional[Any]:
    if device is None:
        return None
    if isinstance(device, str) and device.strip().lower() in {"auto", "best"}:
        return None
    return device

MODEL_PREFIX = {
    "xgb": "Xgboost",
    "glm": "GLM",
    "resn": "ResNet",
    "ft": "FTTransformer",
    "gnn": "GNN",
}

OHT_MODELS = {"resn", "gnn", "glm"}


class Predictor:
    """Minimal predictor interface for production inference."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - protocol-like
        raise NotImplementedError


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    model_name: str
    task_type: str
    cfg: Dict[str, Any]
    output_dir: Path
    artifacts: Optional[Dict[str, Any]]
    device: Optional[Any] = None


ModelLoader = Callable[[ModelSpec], Predictor]


class PredictorRegistry:
    """Registry for mapping model keys to predictor loaders."""

    def __init__(self) -> None:
        self._loaders: Dict[str, ModelLoader] = {}
        self._default_loader: Optional[ModelLoader] = None

    def register(self, model_key: str, loader: ModelLoader, *, overwrite: bool = False) -> None:
        if model_key == "*":
            if self._default_loader is not None and not overwrite:
                raise ValueError("Default loader already registered.")
            self._default_loader = loader
            return
        if model_key in self._loaders and not overwrite:
            raise ValueError(f"Loader already registered for model_key={model_key!r}.")
        self._loaders[model_key] = loader

    def load(self, spec: ModelSpec) -> Predictor:
        loader = self._loaders.get(spec.model_key) or self._default_loader
        if loader is None:
            raise KeyError(f"No loader registered for model_key={spec.model_key!r}.")
        return loader(spec)


_DEFAULT_REGISTRY = PredictorRegistry()


def _default_tweedie_power(
    model_name: str,
    task_type: str,
    loss_name: Optional[str] = None,
) -> Optional[float]:
    if task_type == "classification":
        return None
    if loss_name:
        resolved = resolve_tweedie_power(str(loss_name), default=1.5)
        if resolved is not None:
            return resolved
    if "f" in model_name:
        return 1.0
    if "s" in model_name:
        return 2.0
    return 1.5


def _resolve_loss_name(cfg: Dict[str, Any], model_name: str, task_type: str) -> str:
    return resolve_effective_loss_name(
        cfg.get("loss_name"),
        task_type=task_type,
        model_name=model_name,
        distribution=cfg.get("distribution"),
    )


def _resolve_value(
    value: Any,
    *,
    model_name: str,
    base_dir: Path,
) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get(model_name)
    if value is None:
        return None
    path_str = str(value)
    try:
        path_str = path_str.format(model_name=model_name)
    except Exception:
        pass
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))




def _model_file_path(output_dir: Path, model_name: str, model_key: str) -> Path:
    prefix = MODEL_PREFIX.get(model_key)
    if prefix is None:
        raise ValueError(f"Unsupported model key: {model_key}")
    ext = "pkl" if model_key in {"xgb", "glm"} else "pth"
    return output_dir / "model" / f"01_{model_name}_{prefix}.{ext}"


def _load_preprocess_from_model_file(
    output_dir: Path,
    model_name: str,
    model_key: str,
) -> Optional[Dict[str, Any]]:
    model_path = _model_file_path(output_dir, model_name, model_key)
    if not model_path.exists():
        return None
    if model_key in {"xgb", "glm"}:
        payload = joblib.load(model_path)
    else:
        payload = _torch_load(model_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload.get("preprocess_artifacts")
    return None


def _move_to_device(model_obj: Any, device: Optional[Any] = None) -> None:
    """Move model to best available device using shared DeviceManager."""
    DeviceManager = _get_device_manager()
    DeviceManager.move_to_device(model_obj, device=device)
    if hasattr(model_obj, "eval"):
        model_obj.eval()


def load_best_params(
    output_dir: str | Path,
    model_name: str,
    model_key: str,
) -> Optional[Dict[str, Any]]:
    output_path = Path(output_dir)
    versions_dir = output_path / "Results" / "versions"
    if versions_dir.exists():
        candidates = sorted(versions_dir.glob(f"*_{model_key}_best.json"))
        if candidates:
            payload = _load_json(candidates[-1])
            params = payload.get("best_params")
            if params:
                return params

    label_map = {
        "xgb": "xgboost",
        "resn": "resnet",
        "ft": "fttransformer",
        "glm": "glm",
        "gnn": "gnn",
    }
    label = label_map.get(model_key, model_key)
    csv_path = output_path / "Results" / f"{model_name}_bestparams_{label}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty:
            return df.iloc[0].to_dict()
    return None


def _build_resn_model(
    *,
    model_name: str,
    input_dim: int,
    task_type: str,
    epochs: int,
    resn_weight_decay: float,
    loss_name: str,
    distribution: Optional[str],
    params: Dict[str, Any],
) -> ResNetSklearn:
    from ins_pricing.modelling.bayesopt.models.model_resn import ResNetSklearn
    if loss_name == "tweedie":
        power = params.get(
            "tw_power", _default_tweedie_power(model_name, task_type, loss_name))
        power = float(power) if power is not None else None
    else:
        power = resolve_tweedie_power(loss_name, default=1.5)
    weight_decay = float(params.get("weight_decay", resn_weight_decay))
    return ResNetSklearn(
        model_nme=model_name,
        input_dim=input_dim,
        hidden_dim=int(params.get("hidden_dim", 64)),
        block_num=int(params.get("block_num", 2)),
        task_type=task_type,
        epochs=int(epochs),
        tweedie_power=power,
        learning_rate=float(params.get("learning_rate", 0.01)),
        patience=int(params.get("patience", 10)),
        use_layernorm=True,
        dropout=float(params.get("dropout", 0.1)),
        residual_scale=float(params.get("residual_scale", 0.1)),
        stochastic_depth=float(params.get("stochastic_depth", 0.0)),
        weight_decay=weight_decay,
        use_data_parallel=False,
        use_ddp=False,
        loss_name=loss_name,
        distribution=distribution,
    )


def _build_gnn_model(
    *,
    model_name: str,
    input_dim: int,
    task_type: str,
    epochs: int,
    cfg: Dict[str, Any],
    loss_name: str,
    distribution: Optional[str],
    params: Dict[str, Any],
) -> GraphNeuralNetSklearn:
    from ins_pricing.modelling.bayesopt.models.model_gnn import GraphNeuralNetSklearn
    base_tw = _default_tweedie_power(model_name, task_type, loss_name)
    if loss_name == "tweedie":
        tw_power = params.get("tw_power", base_tw)
        tw_power = float(tw_power) if tw_power is not None else None
    else:
        tw_power = resolve_tweedie_power(loss_name, default=1.5)
    return GraphNeuralNetSklearn(
        model_nme=f"{model_name}_gnn",
        input_dim=input_dim,
        hidden_dim=int(params.get("hidden_dim", 64)),
        num_layers=int(params.get("num_layers", 2)),
        k_neighbors=int(params.get("k_neighbors", 10)),
        dropout=float(params.get("dropout", 0.1)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        epochs=int(params.get("epochs", epochs)),
        patience=int(params.get("patience", 5)),
        task_type=task_type,
        tweedie_power=tw_power,
        weight_decay=float(params.get("weight_decay", 0.0)),
        use_data_parallel=False,
        use_ddp=False,
        use_approx_knn=bool(cfg.get("gnn_use_approx_knn", True)),
        approx_knn_threshold=int(cfg.get("gnn_approx_knn_threshold", 50000)),
        graph_cache_path=cfg.get("gnn_graph_cache"),
        max_gpu_knn_nodes=cfg.get("gnn_max_gpu_knn_nodes"),
        knn_gpu_mem_ratio=cfg.get("gnn_knn_gpu_mem_ratio", 0.9),
        knn_gpu_mem_overhead=cfg.get("gnn_knn_gpu_mem_overhead", 2.0),
        loss_name=loss_name,
        distribution=distribution,
    )


def _load_ft_model_from_payload(
    *,
    payload: Any,
    cfg: Dict[str, Any],
    model_name: str,
    task_type: str,
    device: Optional[Any],
) -> Any:
    if isinstance(payload, dict):
        if "state_dict" in payload and "model_config" in payload:
            state_dict = payload.get("state_dict")
            model_config = payload.get("model_config", {})
            if not model_config.get("loss_name"):
                model_config = dict(model_config)
                model_config["loss_name"] = _resolve_loss_name(
                    cfg, model_name, task_type
                )
            if "distribution" not in model_config:
                model_config = dict(model_config)
                model_config["distribution"] = cfg.get("distribution")
            from ins_pricing.modelling.bayesopt.checkpoints import rebuild_ft_model_from_checkpoint

            model = rebuild_ft_model_from_checkpoint(
                state_dict=state_dict,
                model_config=model_config,
            )
            _move_to_device(model, device=device)
            return model
        if "model" in payload:
            model = payload.get("model")
            _move_to_device(model, device=device)
            return model
    _move_to_device(payload, device=device)
    return payload


def load_saved_model(
    *,
    output_dir: str | Path,
    model_name: str,
    model_key: str,
    task_type: str,
    input_dim: Optional[int],
    cfg: Dict[str, Any],
    device: Optional[Any] = None,
) -> Any:
    model_path = _model_file_path(Path(output_dir), model_name, model_key)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_key in {"xgb", "glm"}:
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and "model" in payload:
            return payload.get("model")
        return payload

    if model_key == "ft":
        payload = _torch_load(
            model_path, map_location="cpu", weights_only=False)
        return _load_ft_model_from_payload(
            payload=payload,
            cfg=cfg,
            model_name=model_name,
            task_type=task_type,
            device=device,
        )

    if model_key == "resn":
        if input_dim is None:
            raise ValueError("input_dim is required for ResNet loading")
        payload = _torch_load(model_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload.get("state_dict")
            params = payload.get("best_params") or load_best_params(
                output_dir, model_name, model_key
            )
        else:
            state_dict = payload
            params = load_best_params(output_dir, model_name, model_key)
        if params is None:
            raise RuntimeError("Best params not found for resn")
        loss_name = _resolve_loss_name(cfg, model_name, task_type)
        model = _build_resn_model(
            model_name=model_name,
            input_dim=input_dim,
            task_type=task_type,
            epochs=int(cfg.get("epochs", 50)),
            resn_weight_decay=float(cfg.get("resn_weight_decay", 1e-4)),
            loss_name=loss_name,
            distribution=cfg.get("distribution"),
            params=params,
        )
        model.resnet.load_state_dict(state_dict)
        _move_to_device(model, device=device)
        return model

    if model_key == "gnn":
        if input_dim is None:
            raise ValueError("input_dim is required for GNN loading")
        payload = _torch_load(model_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid GNN checkpoint: {model_path}")
        params = payload.get("best_params") or {}
        state_dict = payload.get("state_dict")
        loss_name = _resolve_loss_name(cfg, model_name, task_type)
        model = _build_gnn_model(
            model_name=model_name,
            input_dim=input_dim,
            task_type=task_type,
            epochs=int(cfg.get("epochs", 50)),
            cfg=cfg,
            loss_name=loss_name,
            distribution=cfg.get("distribution"),
            params=params,
        )
        model.set_params(dict(params))
        base_gnn = getattr(model, "_unwrap_gnn", lambda: None)()
        if base_gnn is not None and state_dict is not None:
            base_gnn.load_state_dict(state_dict, strict=False)
        _move_to_device(model, device=device)
        return model

    raise ValueError(f"Unsupported model key: {model_key}")


def _build_artifacts_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    factor_nmes = list(cfg.get("feature_list") or [])
    cate_list = list(cfg.get("categorical_features") or [])
    num_features = [c for c in factor_nmes if c not in cate_list]
    return {
        "factor_nmes": factor_nmes,
        "cate_list": cate_list,
        "num_features": num_features,
        "cat_categories": {},
        "var_nmes": [],
        "numeric_scalers": {},
        "drop_first": True,
    }


def _prepare_features(
    df: pd.DataFrame,
    *,
    model_key: str,
    cfg: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    if model_key in OHT_MODELS:
        if artifacts is None:
            raise RuntimeError(
                f"Preprocess artifacts are required for {model_key} inference. "
                "Enable save_preprocess during training or provide preprocess_artifact_path."
            )
        return apply_preprocess_artifacts(df, artifacts)

    if artifacts is None:
        artifacts = _build_artifacts_from_config(cfg)
    return prepare_raw_features(df, artifacts)


def _predict_with_model(
    *,
    model: Any,
    model_key: str,
    task_type: str,
    features: pd.DataFrame,
) -> np.ndarray:
    if model_key == "xgb":
        if task_type == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(features)[:, 1]
        return model.predict(features)

    if model_key == "glm":
        if sm is None:
            raise RuntimeError(
                f"statsmodels is required for GLM inference ({_SM_IMPORT_ERROR})."
            )
        design = sm.add_constant(features, has_constant="add")
        return model.predict(design)

    return model.predict(features)


class SavedModelPredictor(Predictor):
    def __init__(
        self,
        *,
        model_key: str,
        model_name: str,
        task_type: str,
        cfg: Dict[str, Any],
        output_dir: Path,
        artifacts: Optional[Dict[str, Any]],
        device: Optional[Any] = None,
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        self.task_type = task_type
        self.cfg = cfg
        self.output_dir = output_dir
        self.artifacts = artifacts
        self.device = _normalize_device(device)

        if model_key == "ft" and str(cfg.get("ft_role", "model")) != "model":
            raise ValueError("FT predictions require ft_role == 'model'.")
        if model_key == "ft" and cfg.get("geo_feature_nmes"):
            raise ValueError(
                "FT inference with geo tokens is not supported in this helper.")

        input_dim = None
        if model_key in OHT_MODELS and artifacts is not None:
            var_nmes = list(artifacts.get("var_nmes") or [])
            input_dim = len(var_nmes) if var_nmes else None

        self.model = load_saved_model(
            output_dir=output_dir,
            model_name=model_name,
            model_key=model_key,
            task_type=task_type,
            input_dim=input_dim,
            cfg=cfg,
            device=self.device,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features = _prepare_features(
            df,
            model_key=self.model_key,
            cfg=self.cfg,
            artifacts=self.artifacts,
        )
        return _predict_with_model(
            model=self.model,
            model_key=self.model_key,
            task_type=self.task_type,
            features=features,
        )


def _default_loader(spec: ModelSpec) -> SavedModelPredictor:
    return SavedModelPredictor(
        model_key=spec.model_key,
        model_name=spec.model_name,
        task_type=spec.task_type,
        cfg=spec.cfg,
        output_dir=spec.output_dir,
        artifacts=spec.artifacts,
        device=spec.device,
    )


_DEFAULT_REGISTRY.register("*", _default_loader)


def register_model_loader(
    model_key: str,
    loader: ModelLoader,
    *,
    overwrite: bool = False,
    registry: Optional[PredictorRegistry] = None,
) -> None:
    (registry or _DEFAULT_REGISTRY).register(model_key, loader, overwrite=overwrite)


def load_predictor(spec: ModelSpec, *, registry: Optional[PredictorRegistry] = None) -> Predictor:
    return (registry or _DEFAULT_REGISTRY).load(spec)


def load_predictor_from_config(
    config_path: str | Path,
    model_key: str,
    *,
    model_name: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    preprocess_artifact_path: Optional[str | Path] = None,
    device: Optional[Any] = None,
    registry: Optional[PredictorRegistry] = None,
) -> Predictor:
    config_path = Path(config_path).resolve()
    cfg = _load_json(config_path)
    base_dir = config_path.parent

    if model_name is None:
        model_list = list(cfg.get("model_list") or [])
        model_categories = list(cfg.get("model_categories") or [])
        if len(model_list) != 1 or len(model_categories) != 1:
            raise ValueError(
                "Provide model_name when config has multiple models.")
        model_name = f"{model_list[0]}_{model_categories[0]}"

    resolved_output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _resolve_value(cfg.get("output_dir"), model_name=model_name, base_dir=base_dir)
    )
    if resolved_output is None:
        raise ValueError("output_dir is required to locate saved models.")

    resolved_artifact = None
    if preprocess_artifact_path is not None:
        resolved_artifact = Path(preprocess_artifact_path).resolve()
    else:
        resolved_artifact = _resolve_value(
            cfg.get("preprocess_artifact_path"),
            model_name=model_name,
            base_dir=base_dir,
        )

    if resolved_artifact is None:
        candidate = resolved_output / "Results" / \
            f"{model_name}_preprocess.json"
        if candidate.exists():
            resolved_artifact = candidate

    artifacts = None
    if resolved_artifact is not None and resolved_artifact.exists():
        artifacts = load_preprocess_artifacts(resolved_artifact)
    if artifacts is None:
        artifacts = _load_preprocess_from_model_file(
            resolved_output, model_name, model_key
        )

    device = _normalize_device(device)
    spec = ModelSpec(
        model_key=model_key,
        model_name=model_name,
        task_type=str(cfg.get("task_type", "regression")),
        cfg=cfg,
        output_dir=resolved_output,
        artifacts=artifacts,
        device=device,
    )
    return load_predictor(spec, registry=registry)


def predict_from_config(
    config_path: str | Path,
    input_path: str | Path,
    *,
    model_keys: Sequence[str],
    model_name: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    output_col_prefix: str = "pred_",
    batch_size: int = 10000,
    chunksize: Optional[int] = None,
    parallel_load: bool = False,
    n_jobs: int = -1,
    device: Optional[Any] = None,
    registry: Optional[PredictorRegistry] = None,
) -> pd.DataFrame:
    """Predict from multiple models with optional parallel loading.

    Args:
        config_path: Path to configuration file
        input_path: Path to input data
        model_keys: List of model keys to use for prediction
        model_name: Optional model name override
        output_path: Optional path to save results
        output_col_prefix: Prefix for output columns
        batch_size: Batch size for scoring
        chunksize: Optional chunk size for CSV reading
        parallel_load: If True, load models in parallel (faster for multiple models)
        n_jobs: Number of parallel jobs for model loading (-1 = all cores)
        device: Optional torch device or string override (e.g., "cuda", "mps", "cpu")
        registry: Optional predictor registry override

    Returns:
        DataFrame with predictions from all models
    """
    input_path = Path(input_path).resolve()
    data = load_dataset(input_path, data_format="auto", low_memory=False, chunksize=chunksize)

    result = data.copy()

    # Option 1: Parallel model loading (faster when loading multiple models)
    if parallel_load and len(model_keys) > 1:
        from joblib import Parallel, delayed

        def load_and_score(key):
            predictor = load_predictor_from_config(
                config_path,
                key,
                model_name=model_name,
                device=device,
                registry=registry,
            )
            output_col = f"{output_col_prefix}{key}"
            scored = batch_score(
                predictor.predict,
                data,
                output_col=output_col,
                batch_size=batch_size,
                keep_input=False,
            )
            return output_col, scored[output_col].values

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(load_and_score)(key) for key in model_keys
        )

        for output_col, predictions in results:
            result[output_col] = predictions
    else:
        # Option 2: Sequential loading (original behavior)
        for key in model_keys:
            predictor = load_predictor_from_config(
                config_path,
                key,
                model_name=model_name,
                device=device,
                registry=registry,
            )
            output_col = f"{output_col_prefix}{key}"
            scored = batch_score(
                predictor.predict,
                data,
                output_col=output_col,
                batch_size=batch_size,
                keep_input=False,
            )
            result[output_col] = scored[output_col].values

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            result.to_parquet(output_path, index=False)
        elif suffix in {".feather", ".ft"}:
            result.to_feather(output_path)
        else:
            result.to_csv(output_path, index=False)

    return result


__all__ = [
    "Predictor",
    "ModelSpec",
    "PredictorRegistry",
    "register_model_loader",
    "load_predictor",
    "SavedModelPredictor",
    "load_best_params",
    "load_saved_model",
    "load_predictor_from_config",
    "predict_from_config",
]
