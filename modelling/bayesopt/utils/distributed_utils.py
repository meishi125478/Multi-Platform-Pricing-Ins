"""Distributed training utilities for PyTorch DDP.

This module contains:
- DistributedUtils for DDP setup and process coordination
- TrainingUtils for CUDA memory management
- free_cuda() for legacy compatibility
"""

from __future__ import annotations

import gc
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from ins_pricing.utils import get_logger, log_print

_logger = get_logger("ins_pricing.modelling.bayesopt.utils.distributed_utils")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


def _select_ddp_backend() -> str:
    """Select the appropriate DDP backend based on system capabilities.

    Returns:
        "nccl" if CUDA is available and NCCL is supported (non-Windows),
        otherwise "gloo"
    """
    if not torch.cuda.is_available():
        return "gloo"

    if os.name == "nt":  # Windows doesn't support NCCL
        return "gloo"

    try:
        nccl_available = getattr(dist, "is_nccl_available", lambda: False)()
        return "nccl" if nccl_available else "gloo"
    except Exception:
        return "gloo"


def _get_ddp_timeout() -> timedelta:
    """Get the DDP timeout from environment variable.

    Returns:
        timedelta for DDP timeout (default: 1800 seconds)
    """
    timeout_seconds = int(os.environ.get("BAYESOPT_DDP_TIMEOUT_SECONDS", "1800"))
    return timedelta(seconds=max(1, timeout_seconds))


def _cache_ddp_state(local_rank: int, rank: int, world_size: int) -> tuple:
    """Cache and return DDP state tuple."""
    state = (True, local_rank, rank, world_size)
    DistributedUtils._cached_state = state
    return state


class DistributedUtils:
    """Utilities for distributed data parallel training.

    This class provides methods for:
    - Initializing DDP process groups
    - Checking process rank and world size
    - Cleanup after distributed training
    """

    _cached_state: Optional[tuple] = None

    @staticmethod
    def setup_ddp():
        """Initialize the DDP process group for distributed training.

        Returns:
            Tuple of (success, local_rank, rank, world_size)
        """
        # Return cached state if already initialized
        if dist.is_initialized():
            if DistributedUtils._cached_state is None:
                DistributedUtils._cached_state = _cache_ddp_state(
                    int(os.environ.get("LOCAL_RANK", 0)),
                    dist.get_rank(),
                    dist.get_world_size(),
                )
            return DistributedUtils._cached_state

        # Check for required environment variables
        if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
            _log(
                f">>> DDP Setup Failed: RANK or WORLD_SIZE not found in env. "
                f"Keys found: {list(os.environ.keys())}"
            )
            _log(">>> Hint: launch with torchrun --nproc_per_node=<N> <script.py>")
            return False, 0, 0, 1

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Windows CUDA DDP is not supported
        if os.name == "nt" and torch.cuda.is_available() and world_size > 1:
            _log(
                ">>> DDP Setup Disabled: Windows CUDA DDP is not supported. "
                "Falling back to single process."
            )
            return False, 0, 0, 1

        # Set CUDA device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # Initialize process group
        backend = _select_ddp_backend()
        timeout = _get_ddp_timeout()

        dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)
        _log(
            f">>> DDP Initialized ({backend}, timeout={timeout.total_seconds():.0f}s): "
            f"Rank {rank}/{world_size}, Local Rank {local_rank}"
        )

        return _cache_ddp_state(local_rank, rank, world_size)

    @staticmethod
    def cleanup_ddp():
        """Destroy the DDP process group and clear cached state."""
        if dist.is_initialized():
            dist.destroy_process_group()
        DistributedUtils._cached_state = None

    @staticmethod
    def is_main_process():
        """Check if current process is rank 0 (main process).

        Returns:
            True if main process or DDP not initialized
        """
        return not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def world_size() -> int:
        """Get the total number of processes in the distributed group.

        Returns:
            World size (1 if DDP not initialized)
        """
        return dist.get_world_size() if dist.is_initialized() else 1


class TrainingUtils:
    """General training utilities including CUDA management."""

    @staticmethod
    def free_cuda() -> None:
        """Release CUDA memory and clear cache.

        This performs aggressive cleanup:
        1. Move all PyTorch models to CPU
        2. Run garbage collection
        3. Clear CUDA cache
        """
        _log(">>> Moving all models to CPU...")
        for obj in gc.get_objects():
            try:
                if hasattr(obj, "to") and callable(obj.to):
                    obj.to("cpu")
            except Exception:
                pass

        _log(">>> Releasing tensor/optimizer/DataLoader references...")
        gc.collect()

        _log(">>> Clearing CUDA cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            _log(">>> CUDA memory released.")
        else:
            _log(">>> CUDA not available; cleanup skipped.")


# Backward compatibility function wrapper
def free_cuda():
    """Legacy function wrapper for CUDA memory cleanup.

    This function calls TrainingUtils.free_cuda() for backward compatibility.
    """
    TrainingUtils.free_cuda()
