from __future__ import annotations

from pathlib import Path
from typing import Any

from ins_pricing.exceptions import ModelLoadError
from ins_pricing.utils.logging import get_logger
from ins_pricing.utils.safe_pickle import restricted_pickle_load
from ins_pricing.utils.torch_compat import torch_load

_logger = get_logger("ins_pricing.utils.model_loading")


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
        return torch_load(
            artifact,
            map_location=map_location,
            weights_only=weights_only,
        )
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model artifact not found: {artifact}") from exc
    except OSError as exc:
        raise ModelLoadError(f"Cannot read model artifact: {artifact}") from exc
    except Exception as exc:
        raise ModelLoadError(
            f"Failed secure torch load for artifact: {artifact} ({type(exc).__name__}: {exc})"
        ) from exc


__all__ = ["load_pickle_artifact", "load_torch_payload"]
