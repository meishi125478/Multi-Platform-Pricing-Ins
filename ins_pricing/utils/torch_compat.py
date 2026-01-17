"""Torch compatibility helpers for 1.x and 2.x support."""

from __future__ import annotations

import inspect
import os
from typing import Any, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - handled by callers
    TORCH_AVAILABLE = False
    torch = None

_SUPPORTS_WEIGHTS_ONLY: Optional[bool] = None
_DYNAMO_PATCHED = False


def _supports_weights_only() -> bool:
    """Check whether torch.load supports the weights_only argument."""
    global _SUPPORTS_WEIGHTS_ONLY
    if _SUPPORTS_WEIGHTS_ONLY is None:
        if not TORCH_AVAILABLE:
            _SUPPORTS_WEIGHTS_ONLY = False
        else:
            try:
                sig = inspect.signature(torch.load)
                _SUPPORTS_WEIGHTS_ONLY = "weights_only" in sig.parameters
            except (TypeError, ValueError):
                _SUPPORTS_WEIGHTS_ONLY = False
    return bool(_SUPPORTS_WEIGHTS_ONLY)


def torch_load(
    path: Any,
    *args: Any,
    weights_only: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """Load a torch artifact while handling 1.x/2.x API differences."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to load model files.")
    if weights_only is not None and _supports_weights_only():
        return torch.load(path, *args, weights_only=weights_only, **kwargs)
    return torch.load(path, *args, **kwargs)


def _env_truthy(key: str) -> bool:
    value = os.environ.get(key)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def disable_torch_dynamo_if_requested() -> None:
    """Disable torch._dynamo wrappers when compile is explicitly disabled."""
    global _DYNAMO_PATCHED
    if _DYNAMO_PATCHED or not TORCH_AVAILABLE:
        return

    if not any(
        _env_truthy(k)
        for k in (
            "TORCHDYNAMO_DISABLE",
            "TORCH_DISABLE_DYNAMO",
            "TORCH_COMPILE_DISABLE",
            "TORCHINDUCTOR_DISABLE",
        )
    ):
        return

    try:
        import torch.optim.optimizer as optim_mod
    except Exception:
        return

    for name in ("state_dict", "load_state_dict", "zero_grad", "add_param_group"):
        fn = getattr(optim_mod.Optimizer, name, None)
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            setattr(optim_mod.Optimizer, name, wrapped)

    _DYNAMO_PATCHED = True
