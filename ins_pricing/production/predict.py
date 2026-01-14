from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
try:  # statsmodels is optional when GLM inference is not used
    import statsmodels.api as sm
    _SM_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - optional dependency
    sm = None  # type: ignore[assignment]
    _SM_IMPORT_ERROR = exc

from .preprocess import (
    apply_preprocess_artifacts,
    load_preprocess_artifacts,
    prepare_raw_features,
)
from .scoring import batch_score
from ..modelling.core.bayesopt.models.model_gnn import GraphNeuralNetSklearn
from ..modelling.core.bayesopt.models.model_resn import ResNetSklearn
from ins_pricing.utils import DeviceManager, get_logger

_logger = get_logger("ins_pricing.production.predict")


MODEL_PREFIX = {
    "xgb": "Xgboost",
    "glm": "GLM",
    "resn": "ResNet",
    "ft": "FTTransformer",
    "gnn": "GNN",
}

OHT_MODELS = {"resn", "gnn", "glm"}


def _default_tweedie_power(model_name: str, task_type: str) -> Optional[float]:
    if task_type == "classification":
        return None
    if "f" in model_name:
        return 1.0
    if "s" in model_name:
        return 2.0
    return 1.5


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


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".feather", ".ft"}:
        return "feather"
    return "csv"


def _load_dataset(path: Path, chunksize: Optional[int] = None) -> pd.DataFrame:
    """Load dataset with optional chunked reading for large CSV files.

    Args:
        path: Path to the dataset file
        chunksize: If specified for CSV files, reads in chunks and concatenates.
                   Useful for large files that may not fit in memory at once.

    Returns:
        DataFrame containing the full dataset
    """
    fmt = _infer_format(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "feather":
        return pd.read_feather(path)

    # For CSV, support chunked reading for large files
    if chunksize is not None:
        chunks = []
        for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    return pd.read_csv(path, low_memory=False)


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
        payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload.get("preprocess_artifacts")
    return None


def _move_to_device(model_obj: Any) -> None:
    """Move model to best available device using shared DeviceManager."""
    DeviceManager.move_to_device(model_obj)
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
    params: Dict[str, Any],
) -> ResNetSklearn:
    power = params.get("tw_power", _default_tweedie_power(model_name, task_type))
    if power is not None:
        power = float(power)
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
    )


def _build_gnn_model(
    *,
    model_name: str,
    input_dim: int,
    task_type: str,
    epochs: int,
    cfg: Dict[str, Any],
    params: Dict[str, Any],
) -> GraphNeuralNetSklearn:
    base_tw = _default_tweedie_power(model_name, task_type)
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
        tweedie_power=float(params.get("tw_power", base_tw or 1.5)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        use_data_parallel=False,
        use_ddp=False,
        use_approx_knn=bool(cfg.get("gnn_use_approx_knn", True)),
        approx_knn_threshold=int(cfg.get("gnn_approx_knn_threshold", 50000)),
        graph_cache_path=cfg.get("gnn_graph_cache"),
        max_gpu_knn_nodes=cfg.get("gnn_max_gpu_knn_nodes"),
        knn_gpu_mem_ratio=cfg.get("gnn_knn_gpu_mem_ratio", 0.9),
        knn_gpu_mem_overhead=cfg.get("gnn_knn_gpu_mem_overhead", 2.0),
    )


def load_saved_model(
    *,
    output_dir: str | Path,
    model_name: str,
    model_key: str,
    task_type: str,
    input_dim: Optional[int],
    cfg: Dict[str, Any],
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
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            if "state_dict" in payload and "model_config" in payload:
                # New format: state_dict + model_config (DDP-safe)
                state_dict = payload.get("state_dict")
                model_config = payload.get("model_config", {})

                from ..modelling.core.bayesopt.models import FTTransformerSklearn
                from ..modelling.core.bayesopt.models.model_ft_components import FTTransformerCore

                # Reconstruct model from config
                model = FTTransformerSklearn(
                    model_nme=model_config.get("model_nme", ""),
                    num_cols=model_config.get("num_cols", []),
                    cat_cols=model_config.get("cat_cols", []),
                    d_model=model_config.get("d_model", 64),
                    n_heads=model_config.get("n_heads", 8),
                    n_layers=model_config.get("n_layers", 4),
                    dropout=model_config.get("dropout", 0.1),
                    task_type=model_config.get("task_type", "regression"),
                    tweedie_power=model_config.get("tw_power", 1.5),
                    num_numeric_tokens=model_config.get("num_numeric_tokens"),
                    use_data_parallel=False,
                    use_ddp=False,
                )
                # Restore internal state
                model.num_geo = model_config.get("num_geo", 0)
                model.cat_cardinalities = model_config.get("cat_cardinalities")
                model.cat_categories = {k: pd.Index(v) for k, v in model_config.get("cat_categories", {}).items()}
                if model_config.get("_num_mean") is not None:
                    model._num_mean = np.array(model_config["_num_mean"], dtype=np.float32)
                if model_config.get("_num_std") is not None:
                    model._num_std = np.array(model_config["_num_std"], dtype=np.float32)

                # Build the model architecture and load weights
                if model.cat_cardinalities is not None:
                    core = FTTransformerCore(
                        num_numeric=len(model.num_cols),
                        cat_cardinalities=model.cat_cardinalities,
                        d_model=model.d_model,
                        n_heads=model.n_heads,
                        n_layers=model.n_layers,
                        dropout=model.dropout,
                        task_type=model.task_type,
                        num_geo=model.num_geo,
                        num_numeric_tokens=model.num_numeric_tokens,
                    )
                    model.ft = core
                    model.ft.load_state_dict(state_dict)

                _move_to_device(model)
                return model
            elif "model" in payload:
                # Legacy format: full model object
                model = payload.get("model")
                _move_to_device(model)
                return model
        # Very old format: direct model object
        _move_to_device(payload)
        return payload

    if model_key == "resn":
        if input_dim is None:
            raise ValueError("input_dim is required for ResNet loading")
        payload = torch.load(model_path, map_location="cpu")
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
        model = _build_resn_model(
            model_name=model_name,
            input_dim=input_dim,
            task_type=task_type,
            epochs=int(cfg.get("epochs", 50)),
            resn_weight_decay=float(cfg.get("resn_weight_decay", 1e-4)),
            params=params,
        )
        model.resnet.load_state_dict(state_dict)
        _move_to_device(model)
        return model

    if model_key == "gnn":
        if input_dim is None:
            raise ValueError("input_dim is required for GNN loading")
        payload = torch.load(model_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid GNN checkpoint: {model_path}")
        params = payload.get("best_params") or {}
        state_dict = payload.get("state_dict")
        model = _build_gnn_model(
            model_name=model_name,
            input_dim=input_dim,
            task_type=task_type,
            epochs=int(cfg.get("epochs", 50)),
            cfg=cfg,
            params=params,
        )
        model.set_params(dict(params))
        base_gnn = getattr(model, "_unwrap_gnn", lambda: None)()
        if base_gnn is not None and state_dict is not None:
            base_gnn.load_state_dict(state_dict, strict=False)
        _move_to_device(model)
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


class SavedModelPredictor:
    def __init__(
        self,
        *,
        model_key: str,
        model_name: str,
        task_type: str,
        cfg: Dict[str, Any],
        output_dir: Path,
        artifacts: Optional[Dict[str, Any]],
    ) -> None:
        self.model_key = model_key
        self.model_name = model_name
        self.task_type = task_type
        self.cfg = cfg
        self.output_dir = output_dir
        self.artifacts = artifacts

        if model_key == "ft" and str(cfg.get("ft_role", "model")) != "model":
            raise ValueError("FT predictions require ft_role == 'model'.")
        if model_key == "ft" and cfg.get("geo_feature_nmes"):
            raise ValueError("FT inference with geo tokens is not supported in this helper.")

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


def load_predictor_from_config(
    config_path: str | Path,
    model_key: str,
    *,
    model_name: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    preprocess_artifact_path: Optional[str | Path] = None,
) -> SavedModelPredictor:
    config_path = Path(config_path).resolve()
    cfg = _load_json(config_path)
    base_dir = config_path.parent

    if model_name is None:
        model_list = list(cfg.get("model_list") or [])
        model_categories = list(cfg.get("model_categories") or [])
        if len(model_list) != 1 or len(model_categories) != 1:
            raise ValueError("Provide model_name when config has multiple models.")
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
        candidate = resolved_output / "Results" / f"{model_name}_preprocess.json"
        if candidate.exists():
            resolved_artifact = candidate

    artifacts = None
    if resolved_artifact is not None and resolved_artifact.exists():
        artifacts = load_preprocess_artifacts(resolved_artifact)
    if artifacts is None:
        artifacts = _load_preprocess_from_model_file(
            resolved_output, model_name, model_key
        )

    predictor = SavedModelPredictor(
        model_key=model_key,
        model_name=model_name,
        task_type=str(cfg.get("task_type", "regression")),
        cfg=cfg,
        output_dir=resolved_output,
        artifacts=artifacts,
    )
    return predictor


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

    Returns:
        DataFrame with predictions from all models
    """
    input_path = Path(input_path).resolve()
    data = _load_dataset(input_path, chunksize=chunksize)

    result = data.copy()

    # Option 1: Parallel model loading (faster when loading multiple models)
    if parallel_load and len(model_keys) > 1:
        from joblib import Parallel, delayed

        def load_and_score(key):
            predictor = load_predictor_from_config(
                config_path,
                key,
                model_name=model_name,
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
        if output_path.suffix.lower() in {".parquet", ".pq"}:
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
    return result
