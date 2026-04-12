from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from ins_pricing.exceptions import ModelLoadError
from ins_pricing.utils.logging import get_logger
from ins_pricing.utils.safe_pickle import restricted_pickle_load
from ins_pricing.utils.torch_compat import supports_weights_only, torch_load

_logger = get_logger("ins_pricing.utils.model_loading")

_TRUTHY = {"1", "true", "yes", "y", "on"}
_PICKLE_MODEL_KEYS = {"xgb", "glm"}


def _allow_legacy_torch_load_fallback() -> bool:
    """Enable torch1.x compatibility fallback unless explicitly disabled."""
    value = os.environ.get("INS_PRICING_ALLOW_LEGACY_TORCH_LOAD", "1")
    return str(value).strip().lower() in _TRUTHY


def load_pickle_artifact(
    path: str | Path,
) -> Any:
    """Load pickle/joblib artifacts with a secure default policy.

    Only restricted loading is supported; unsafe fallback is disabled.
    """
    artifact = Path(path)
    try:
        with artifact.open("rb") as fh:
            return restricted_pickle_load(fh)
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model artifact not found: {artifact}") from exc
    except OSError as exc:
        raise ModelLoadError(f"Cannot read model artifact: {artifact}") from exc
    except Exception as exc:
        raise ModelLoadError(
            f"Failed secure pickle load for artifact: {artifact} ({type(exc).__name__}: {exc})"
        ) from exc


def _is_unsafe_pickle_retry_eligible(exc: ModelLoadError) -> bool:
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, pickle.UnpicklingError):
        return True
    text = str(exc).lower()
    io_markers = ("not found", "cannot read", "no such file", "permission denied")
    if any(marker in text for marker in io_markers):
        return False
    # For explicit trusted fallback mode, retain compatibility with legacy
    # checkpoints by retrying non-I/O secure-load failures.
    return True


def load_pickle_artifact_with_optional_unsafe_retry(
    path: str | Path,
    *,
    allow_unsafe_retry: bool = False,
) -> Any:
    """Load pickle artifact with optional trusted fallback for restricted errors."""
    artifact = Path(path)
    try:
        return load_pickle_artifact(artifact)
    except ModelLoadError as exc:
        if not bool(allow_unsafe_retry) or not _is_unsafe_pickle_retry_eligible(exc):
            raise
        _logger.warning(
            "Retrying secure pickle load with trusted legacy fallback: %s",
            artifact,
        )
        try:
            with artifact.open("rb") as fh:
                return pickle.load(fh)
        except FileNotFoundError as io_exc:
            raise ModelLoadError(f"Model artifact not found: {artifact}") from io_exc
        except OSError as io_exc:
            raise ModelLoadError(f"Cannot read model artifact: {artifact}") from io_exc
        except Exception as io_exc:
            raise ModelLoadError(
                f"Failed legacy pickle load for artifact: {artifact} "
                f"({type(io_exc).__name__}: {io_exc})"
            ) from io_exc


def load_torch_payload(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool = True,
) -> Any:
    """Load torch artifacts with weights-only mode by default."""
    if not weights_only:
        raise ModelLoadError("weights_only=False is not supported in secure loading mode.")
    artifact = Path(path)
    try:
        if not supports_weights_only():
            if not _allow_legacy_torch_load_fallback():
                raise ModelLoadError(
                    "Installed torch does not support weights_only=True and "
                    "legacy fallback is disabled (set INS_PRICING_ALLOW_LEGACY_TORCH_LOAD=1)."
                )
            _logger.warning(
                "torch.load(weights_only=True) unsupported by current torch runtime; "
                "falling back to trusted legacy load for artifact: %s",
                artifact,
            )
            return torch_load(
                artifact,
                map_location=map_location,
                weights_only=False,
            )
        return torch_load(
            artifact,
            map_location=map_location,
            weights_only=weights_only,
        )
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model artifact not found: {artifact}") from exc
    except OSError as exc:
        raise ModelLoadError(f"Cannot read model artifact: {artifact}") from exc
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            f"Failed secure torch load for artifact: {artifact} ({type(exc).__name__}: {exc})"
        ) from exc


def load_model_artifact_payload(
    path: str | Path,
    *,
    model_key: str,
    map_location: Any = "cpu",
    allow_unsafe_pickle_retry: bool = False,
) -> Any:
    """Load artifact payload by model family with centralized compatibility policy."""
    key = str(model_key or "").strip().lower()
    artifact = Path(path)
    if key in _PICKLE_MODEL_KEYS:
        return load_pickle_artifact_with_optional_unsafe_retry(
            artifact,
            allow_unsafe_retry=bool(allow_unsafe_pickle_retry),
        )
    return load_torch_payload(
        artifact,
        map_location=map_location,
        weights_only=True,
    )


__all__ = [
    "load_pickle_artifact",
    "load_pickle_artifact_with_optional_unsafe_retry",
    "load_torch_payload",
    "load_model_artifact_payload",
]
