"""Shared torch runtime helpers for device/DDP/DataParallel setup."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ins_pricing.modelling.bayesopt.utils.distributed_utils import DistributedUtils
from ins_pricing.utils.device import DeviceManager


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
    if not bool(use_gpu):
        return torch.device("cpu")
    if is_ddp_enabled:
        return torch.device(f"cuda:{int(local_rank)}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if DeviceManager.is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        core = DDP(
            core,
            device_ids=[int(local_rank)],
            output_device=int(local_rank),
            find_unused_parameters=bool(ddp_find_unused_parameters),
        )
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
    "wrap_model_for_parallel",
]
