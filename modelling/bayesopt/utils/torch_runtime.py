"""Shared torch runtime helpers for device/DDP/DataParallel setup."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ins_pricing.modelling.bayesopt.utils.distributed_utils import DistributedUtils
from ins_pricing.utils.device import DeviceManager

try:
    from torch.amp import GradScaler, autocast
    _UNIFIED_AMP_AVAILABLE = True
except Exception:  # pragma: no cover - legacy torch fallback
    from torch.cuda.amp import GradScaler, autocast
    _UNIFIED_AMP_AVAILABLE = False


def setup_ddp_if_requested(use_ddp: bool) -> Tuple[bool, int, int, int]:
    """Initialize DDP only when requested."""
    if not bool(use_ddp):
        return False, 0, 0, 1
    ddp_ok, local_rank, rank, world_size = DistributedUtils.setup_ddp()
    return bool(ddp_ok), int(local_rank), int(rank), int(world_size)


def resolve_training_device(
    *,
    is_ddp_enabled: bool,
    local_rank: int,
    use_gpu: bool = True,
) -> torch.device:
    """Resolve training device with stable fallback order: cuda > mps > cpu."""
    device = DeviceManager.resolve_training_device(
        is_ddp_enabled=is_ddp_enabled,
        local_rank=local_rank,
        use_gpu=use_gpu,
    )
    if device is None:
        return torch.device("cpu")
    return device


def _supports_autocast(device_type: str) -> bool:
    if _UNIFIED_AMP_AVAILABLE:
        amp_mod = getattr(torch, "amp", None)
        checker = None if amp_mod is None else getattr(
            getattr(amp_mod, "autocast_mode", None),
            "is_autocast_available",
            None,
        )
        if callable(checker):
            try:
                return bool(checker(device_type))
            except Exception:
                return False
        return device_type == "cuda"
    return device_type == "cuda" and torch.cuda.is_available()


def _grad_scaler_accepts_device_kw() -> bool:
    try:
        return "device" in inspect.signature(GradScaler).parameters
    except Exception:
        return False


def create_autocast_context(device_type: str, enabled: Optional[bool] = None):
    """Create autocast context with safe fallback when backend is unsupported."""
    if enabled is None:
        enabled = device_type in ("cuda", "mps")
    enabled = bool(enabled) and _supports_autocast(device_type)
    if not enabled:
        return nullcontext()
    if not _UNIFIED_AMP_AVAILABLE:
        if device_type != "cuda":
            return nullcontext()
        return autocast(enabled=True)
    if device_type == "mps":
        return autocast(device_type=device_type, dtype=torch.float16, enabled=True)
    return autocast(device_type=device_type, enabled=True)


def create_grad_scaler(device_type: str, enabled: Optional[bool] = None) -> GradScaler:
    """Create AMP GradScaler with backend-aware compatibility fallback."""
    if enabled is None:
        enabled = device_type in ("cuda", "mps")
    enabled = bool(enabled)
    target_device = device_type if device_type in {"cuda", "mps", "cpu", "xpu"} else "cuda"

    if _grad_scaler_accepts_device_kw():
        return GradScaler(device=target_device, enabled=enabled)

    # Legacy API fallback: only CUDA is safe without a device argument.
    if target_device != "cuda":
        return GradScaler(enabled=False)
    return GradScaler(enabled=enabled)


def wrap_model_for_parallel(
    core: nn.Module,
    *,
    device: torch.device,
    use_data_parallel: bool,
    use_ddp_requested: bool,
    is_ddp_enabled: bool,
    local_rank: int,
    ddp_find_unused_parameters: bool = False,
    fallback_log: Optional[Callable[[str], None]] = None,
) -> Tuple[nn.Module, bool, torch.device]:
    """Wrap model with DDP or DataParallel based on resolved runtime flags."""
    if is_ddp_enabled:
        core = core.to(device)
        ddp_kwargs = {
            "find_unused_parameters": bool(ddp_find_unused_parameters),
        }
        if device.type == "cuda":
            ddp_kwargs.update(
                {
                    "device_ids": [int(local_rank)],
                    "output_device": int(local_rank),
                }
            )
        core = DDP(core, **ddp_kwargs)
        return core, False, device

    should_use_dp = (
        bool(use_data_parallel)
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    )
    if should_use_dp:
        if bool(use_ddp_requested) and fallback_log is not None:
            fallback_log(
                ">>> DDP requested but not initialized; falling back to DataParallel."
            )
        core = nn.DataParallel(core, device_ids=list(range(torch.cuda.device_count())))
        return core, True, torch.device("cuda")

    return core, False, device


__all__ = [
    "setup_ddp_if_requested",
    "resolve_training_device",
    "create_autocast_context",
    "create_grad_scaler",
    "wrap_model_for_parallel",
]
