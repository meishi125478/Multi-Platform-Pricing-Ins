from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from ins_pricing.modelling.bayesopt.checkpoints import (
    rebuild_ft_model_from_payload,
    rebuild_gnn_model_from_payload,
    rebuild_resn_model_from_payload,
)

ModelBuilder = Callable[[Dict[str, Any]], Any]
_PICKLE_MODEL_KEYS = {"xgb", "glm"}


def rebuild_ft_payload(
    *,
    payload: Any,
    model_config_overrides: Optional[Dict[str, Any]] = None,
    fill_missing_model_config: bool = True,
) -> Tuple[Any, Optional[Dict[str, Any]], str]:
    """Rebuild FT artifacts from payload with a stable helper entry-point."""
    return rebuild_ft_model_from_payload(
        payload=payload,
        model_config_overrides=model_config_overrides,
        fill_missing_model_config=fill_missing_model_config,
    )


def rebuild_resn_payload(
    *,
    payload: Any,
    model_builder: ModelBuilder,
    params_fallback: Optional[Dict[str, Any]] = None,
    require_params: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Rebuild ResNet artifacts from payload with centralized validation."""
    if not callable(model_builder):
        raise ValueError("ResNet model rebuild requires a callable model_builder.")
    return rebuild_resn_model_from_payload(
        payload=payload,
        model_builder=model_builder,
        params_fallback=params_fallback,
        require_params=require_params,
    )


def rebuild_gnn_payload(
    *,
    payload: Any,
    model_builder: ModelBuilder,
    strict: bool = True,
    allow_non_strict_fallback: bool = False,
) -> Tuple[Any, Dict[str, Any], Optional[str]]:
    """Rebuild GNN artifacts from payload with centralized validation."""
    if not callable(model_builder):
        raise ValueError("GNN model rebuild requires a callable model_builder.")
    return rebuild_gnn_model_from_payload(
        payload=payload,
        model_builder=model_builder,
        strict=strict,
        allow_non_strict_fallback=allow_non_strict_fallback,
    )


def rebuild_model_artifact_payload(
    *,
    payload: Any,
    model_key: str,
    model_builder: Optional[ModelBuilder] = None,
    model_config_overrides: Optional[Dict[str, Any]] = None,
    fill_missing_model_config: bool = True,
    params_fallback: Optional[Dict[str, Any]] = None,
    require_params: bool = True,
    strict: bool = True,
    allow_non_strict_fallback: bool = False,
) -> Any:
    """Dispatch payload rebuild by model family.

    For `xgb/glm`, this helper returns the embedded `model` entry when present,
    otherwise returns the payload unchanged.
    """
    key = str(model_key or "").strip().lower()
    if key in _PICKLE_MODEL_KEYS:
        if isinstance(payload, dict) and "model" in payload:
            return payload.get("model")
        return payload
    if key == "ft":
        return rebuild_ft_payload(
            payload=payload,
            model_config_overrides=model_config_overrides,
            fill_missing_model_config=fill_missing_model_config,
        )
    if key == "resn":
        if not callable(model_builder):
            raise ValueError("ResNet payload rebuild requires model_builder.")
        return rebuild_resn_payload(
            payload=payload,
            model_builder=model_builder,
            params_fallback=params_fallback,
            require_params=require_params,
        )
    if key == "gnn":
        if not callable(model_builder):
            raise ValueError("GNN payload rebuild requires model_builder.")
        return rebuild_gnn_payload(
            payload=payload,
            model_builder=model_builder,
            strict=strict,
            allow_non_strict_fallback=allow_non_strict_fallback,
        )
    raise ValueError(f"Unsupported model key for payload rebuild: {model_key!r}")


__all__ = [
    "rebuild_ft_payload",
    "rebuild_resn_payload",
    "rebuild_gnn_payload",
    "rebuild_model_artifact_payload",
]
