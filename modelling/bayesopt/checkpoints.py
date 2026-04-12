from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def serialize_ft_model_config(model: Any) -> Dict[str, Any]:
    return {
        "model_nme": getattr(model, "model_nme", ""),
        "num_cols": list(getattr(model, "num_cols", [])),
        "cat_cols": list(getattr(model, "cat_cols", [])),
        "d_model": getattr(model, "d_model", 64),
        "n_heads": getattr(model, "n_heads", 8),
        "n_layers": getattr(model, "n_layers", 4),
        "dropout": getattr(model, "dropout", 0.1),
        "task_type": getattr(model, "task_type", "regression"),
        "distribution": getattr(model, "distribution", None),
        "loss_name": getattr(model, "loss_name", None),
        "tw_power": getattr(model, "tw_power", 1.5),
        "num_geo": getattr(model, "num_geo", 0),
        "num_numeric_tokens": getattr(model, "num_numeric_tokens", None),
        "cat_cardinalities": getattr(model, "cat_cardinalities", None),
        "cat_categories": {
            k: list(v) for k, v in getattr(model, "cat_categories", {}).items()
        },
        "_num_mean": _to_serializable(getattr(model, "_num_mean", None)),
        "_num_std": _to_serializable(getattr(model, "_num_std", None)),
    }


def rebuild_ft_model_from_checkpoint(
    *,
    state_dict: Dict[str, Any],
    model_config: Dict[str, Any],
):
    if state_dict is None:
        raise ValueError("FT checkpoint is missing state_dict.")

    from ins_pricing.modelling.bayesopt.models import FTTransformerSklearn
    from ins_pricing.modelling.bayesopt.models.model_ft_components import FTTransformerCore

    model = FTTransformerSklearn(
        model_nme=model_config.get("model_nme", ""),
        num_cols=model_config.get("num_cols", []),
        cat_cols=model_config.get("cat_cols", []),
        d_model=model_config.get("d_model", 64),
        n_heads=model_config.get("n_heads", 8),
        n_layers=model_config.get("n_layers", 4),
        dropout=model_config.get("dropout", 0.1),
        task_type=model_config.get("task_type", "regression"),
        distribution=model_config.get("distribution", None),
        loss_name=model_config.get("loss_name", None),
        tweedie_power=model_config.get("tw_power", 1.5),
        num_numeric_tokens=model_config.get("num_numeric_tokens"),
        use_data_parallel=False,
        use_ddp=False,
        use_gpu=False,
    )

    model.num_geo = model_config.get("num_geo", 0)
    model.cat_cardinalities = model_config.get("cat_cardinalities")
    model.cat_categories = {
        k: pd.Index(v) for k, v in model_config.get("cat_categories", {}).items()
    }
    if model_config.get("_num_mean") is not None:
        model._num_mean = np.array(model_config["_num_mean"], dtype=np.float32)
    if model_config.get("_num_std") is not None:
        model._num_std = np.array(model_config["_num_std"], dtype=np.float32)

    if model.cat_cardinalities is None:
        raise ValueError(
            "FT checkpoint is missing cat_cardinalities in model_config."
        )

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

    return model


def _merge_model_config(
    base_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
    *,
    fill_missing_only: bool,
) -> Dict[str, Any]:
    merged = dict(base_config or {})
    if not overrides:
        return merged
    for key, value in overrides.items():
        if value is None:
            continue
        if fill_missing_only and key in merged and merged.get(key) not in (None, ""):
            continue
        merged[key] = value
    return merged


def rebuild_ft_model_from_payload(
    *,
    payload: Any,
    model_config_overrides: Optional[Dict[str, Any]] = None,
    fill_missing_model_config: bool = True,
) -> Tuple[Any, Optional[Dict[str, Any]], str]:
    if not isinstance(payload, dict):
        return payload, None, "raw"
    if "state_dict" in payload and "model_config" in payload:
        model_config = _merge_model_config(
            payload.get("model_config", {}),
            model_config_overrides,
            fill_missing_only=fill_missing_model_config,
        )
        model = rebuild_ft_model_from_checkpoint(
            state_dict=payload.get("state_dict"),
            model_config=model_config,
        )
        best_params = payload.get("best_params")
        best = dict(best_params) if isinstance(best_params, dict) else None
        return model, best, "state_dict"
    if "model" in payload:
        best_params = payload.get("best_params")
        best = dict(best_params) if isinstance(best_params, dict) else None
        return payload.get("model"), best, "model"
    return payload, None, "raw"


def rebuild_resn_model_from_payload(
    *,
    payload: Any,
    model_builder: Callable[[Dict[str, Any]], Any],
    params_fallback: Optional[Dict[str, Any]] = None,
    require_params: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload.get("state_dict")
        params = payload.get("best_params")
    else:
        state_dict = payload
        params = None

    resolved_params: Optional[Dict[str, Any]] = None
    if isinstance(params, dict) and params:
        resolved_params = dict(params)
    elif isinstance(params_fallback, dict):
        resolved_params = dict(params_fallback)
    elif isinstance(params, dict) and not require_params:
        resolved_params = dict(params)
    if require_params and not resolved_params:
        raise RuntimeError("Best params not found for resn")
    if resolved_params is None:
        resolved_params = {}

    model = model_builder(resolved_params)
    model.resnet.load_state_dict(state_dict)
    return model, resolved_params


def _load_state_dict_with_policy(
    module: Any,
    state_dict: Any,
    *,
    strict: bool,
    allow_non_strict_fallback: bool,
) -> Optional[str]:
    if module is None or state_dict is None:
        return None
    if not strict:
        module.load_state_dict(state_dict, strict=False)
        return None
    try:
        module.load_state_dict(state_dict, strict=True)
        return None
    except RuntimeError as exc:
        if allow_non_strict_fallback and (
            "Missing key" in str(exc) or "Unexpected key" in str(exc)
        ):
            module.load_state_dict(state_dict, strict=False)
            return str(exc)
        raise


def rebuild_gnn_model_from_payload(
    *,
    payload: Any,
    model_builder: Callable[[Dict[str, Any]], Any],
    strict: bool = True,
    allow_non_strict_fallback: bool = False,
) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid GNN checkpoint payload.")
    params_raw = payload.get("best_params")
    params = dict(params_raw) if isinstance(params_raw, dict) else {}
    state_dict = payload.get("state_dict")
    if state_dict is None:
        raise ValueError("Invalid GNN checkpoint payload: missing 'state_dict'.")

    model = model_builder(params)
    if params and hasattr(model, "set_params"):
        model.set_params(dict(params))

    base_gnn = getattr(model, "_unwrap_gnn", lambda: None)()
    if base_gnn is None:
        raise ValueError("Invalid GNN model builder output: missing unwrap-able GNN module.")
    warning = _load_state_dict_with_policy(
        base_gnn,
        state_dict,
        strict=strict,
        allow_non_strict_fallback=allow_non_strict_fallback,
    )
    return model, params, warning


__all__ = [
    "serialize_ft_model_config",
    "rebuild_ft_model_from_checkpoint",
    "rebuild_ft_model_from_payload",
    "rebuild_resn_model_from_payload",
    "rebuild_gnn_model_from_payload",
]
