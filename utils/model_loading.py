from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from ins_pricing.exceptions import ModelLoadError
from ins_pricing.utils.logging import get_logger
from ins_pricing.utils.safe_pickle import restricted_pickle_load
from ins_pricing.utils.torch_compat import torch_load

try:
    import joblib
except ImportError:  # pragma: no cover - optional fallback dependency
    joblib = None  # type: ignore[assignment]

_logger = get_logger("ins_pricing.utils.model_loading")
_UNSAFE_LOAD_ENV = "INS_PRICING_ALLOW_UNSAFE_MODEL_LOAD"
_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_truthy(key: str) -> bool:
    value = os.environ.get(key)
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def _allow_unsafe(allow_unsafe: Optional[bool]) -> bool:
    if allow_unsafe is not None:
        return bool(allow_unsafe)
    return _env_truthy(_UNSAFE_LOAD_ENV)


def load_pickle_artifact(
    path: str | Path,
    *,
    allow_unsafe: Optional[bool] = None,
) -> Any:
    """Load pickle/joblib artifacts with a secure default policy.

    Default behavior uses the restricted unpickler and blocks unsafe globals.
    Legacy joblib fallback is only enabled when explicitly trusted.
    """
    artifact = Path(path)
    try:
        with artifact.open("rb") as fh:
            return restricted_pickle_load(fh)
    except Exception as exc:
        if not _allow_unsafe(allow_unsafe):
            raise ModelLoadError(
                f"Blocked unsafe pickle artifact: {artifact}. "
                f"Set {_UNSAFE_LOAD_ENV}=1 only for trusted model files."
            ) from exc

    _logger.warning(
        "Falling back to unsafe joblib.load for trusted artifact: %s",
        artifact,
    )
    if joblib is None:
        raise ModelLoadError(
            "Unsafe pickle fallback requires optional dependency 'joblib'."
        )
    try:
        return joblib.load(artifact)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load pickle artifact: {artifact}") from exc


def load_torch_payload(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool = True,
    allow_unsafe: Optional[bool] = None,
) -> Any:
    """Load torch artifacts with weights-only mode by default."""
    artifact = Path(path)
    try:
        return torch_load(
            artifact,
            map_location=map_location,
            weights_only=weights_only,
        )
    except Exception as exc:
        if weights_only and _allow_unsafe(allow_unsafe):
            _logger.warning(
                "Falling back to unsafe torch load (weights_only=False) for trusted artifact: %s",
                artifact,
            )
            try:
                return torch_load(
                    artifact,
                    map_location=map_location,
                    weights_only=False,
                )
            except Exception as unsafe_exc:
                raise ModelLoadError(
                    f"Failed to load torch artifact even with unsafe fallback: {artifact}"
                ) from unsafe_exc
        raise ModelLoadError(
            f"Failed secure torch load for artifact: {artifact}. "
            f"Set {_UNSAFE_LOAD_ENV}=1 only for trusted model files."
        ) from exc


__all__ = ["load_pickle_artifact", "load_torch_payload"]
