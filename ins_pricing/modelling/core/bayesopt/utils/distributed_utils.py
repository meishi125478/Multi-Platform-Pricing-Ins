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
        if dist.is_initialized():
            if DistributedUtils._cached_state is None:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                DistributedUtils._cached_state = (
                    True,
                    local_rank,
                    rank,
                    world_size,
                )
            return DistributedUtils._cached_state

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])

            if os.name == "nt" and torch.cuda.is_available() and world_size > 1:
                print(
                    ">>> DDP Setup Disabled: Windows CUDA DDP is not supported. "
                    "Falling back to single process."
                )
                return False, 0, 0, 1

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            timeout_seconds = int(os.environ.get(
                "BAYESOPT_DDP_TIMEOUT_SECONDS", "1800"))
            timeout = timedelta(seconds=max(1, timeout_seconds))
            backend = "gloo"
            if torch.cuda.is_available() and os.name != "nt":
                try:
                    if getattr(dist, "is_nccl_available", lambda: False)():
                        backend = "nccl"
                except Exception:
                    backend = "gloo"

            dist.init_process_group(
                backend=backend, init_method="env://", timeout=timeout)
            print(
                f">>> DDP Initialized ({backend}, timeout={timeout_seconds}s): "
                f"Rank {rank}/{world_size}, Local Rank {local_rank}"
            )
            DistributedUtils._cached_state = (
                True,
                local_rank,
                rank,
                world_size,
            )
            return DistributedUtils._cached_state
        else:
            print(
                f">>> DDP Setup Failed: RANK or WORLD_SIZE not found in env. Keys found: {list(os.environ.keys())}"
            )
            print(
                ">>> Hint: launch with torchrun --nproc_per_node=<N> <script.py>"
            )
        return False, 0, 0, 1

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
        print(">>> Moving all models to CPU...")
        for obj in gc.get_objects():
            try:
                if hasattr(obj, "to") and callable(obj.to):
                    obj.to("cpu")
            except Exception:
                pass

        print(">>> Releasing tensor/optimizer/DataLoader references...")
        gc.collect()

        print(">>> Clearing CUDA cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(">>> CUDA memory released.")
        else:
            print(">>> CUDA not available; cleanup skipped.")


# Backward compatibility function wrapper
def free_cuda():
    """Legacy function wrapper for CUDA memory cleanup.

    This function calls TrainingUtils.free_cuda() for backward compatibility.
    """
    TrainingUtils.free_cuda()
