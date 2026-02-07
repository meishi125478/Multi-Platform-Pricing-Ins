"""Device management utilities for PyTorch models.

This module consolidates GPU/CPU device management logic from:
- modelling/bayesopt/utils.py
- modelling/bayesopt/trainers/trainer_base.py
- production/inference.py

Example:
    >>> from ins_pricing.utils import DeviceManager, GPUMemoryManager
    >>> device = DeviceManager.get_best_device()
    >>> DeviceManager.move_to_device(model, device)
    >>> with GPUMemoryManager.cleanup_context():
    ...     model.train()
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DDP = None

from ins_pricing.utils.logging import get_logger


# =============================================================================
# GPU Memory Manager
# =============================================================================


class GPUMemoryManager:
    """Context manager for GPU memory management and cleanup.

    This class consolidates GPU memory cleanup logic that was previously
    scattered across multiple trainer files.

    Example:
        >>> with GPUMemoryManager.cleanup_context():
        ...     model.train()
        ...     # Memory cleaned up after exiting context

        >>> # Or use directly:
        >>> GPUMemoryManager.clean()
    """

    _logger = get_logger("ins_pricing.gpu")

    @classmethod
    def clean(
        cls,
        verbose: bool = False,
        *,
        synchronize: bool = True,
        empty_cache: bool = True,
    ) -> None:
        """Clean up GPU memory.

        Args:
            verbose: If True, log cleanup details
            synchronize: If True, synchronize CUDA device after cleanup
            empty_cache: If True, clear CUDA cache
        """
        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            if empty_cache:
                torch.cuda.empty_cache()
            if synchronize:
                torch.cuda.synchronize()
            if verbose:
                if empty_cache and synchronize:
                    cls._logger.debug("CUDA cache cleared and synchronized")
                elif empty_cache:
                    cls._logger.debug("CUDA cache cleared")
                elif synchronize:
                    cls._logger.debug("CUDA synchronized")

            # Optional: Force IPC collect for multi-process scenarios
            if os.environ.get("BAYESOPT_CUDA_IPC_COLLECT", "0") == "1":
                try:
                    torch.cuda.ipc_collect()
                    if verbose:
                        cls._logger.debug("CUDA IPC collect performed")
                except Exception:
                    pass

    @classmethod
    @contextmanager
    def cleanup_context(
        cls,
        verbose: bool = False,
        *,
        synchronize: bool = True,
        empty_cache: bool = True,
    ):
        """Context manager that cleans GPU memory on exit.

        Args:
            verbose: If True, log cleanup details

        Yields:
            None
        """
        try:
            yield
        finally:
            cls.clean(verbose=verbose, synchronize=synchronize, empty_cache=empty_cache)

    @classmethod
    def move_model_to_cpu(cls, model: Any) -> Any:
        """Move a model to CPU and clean GPU memory.

        Args:
            model: PyTorch model to move

        Returns:
            Model on CPU
        """
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
        cls.clean()
        return model

    @classmethod
    def get_memory_info(cls) -> Dict[str, Any]:
        """Get current GPU memory usage information.

        Returns:
            Dictionary with memory info (allocated, reserved, free)
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"available": False}

        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            free, total = torch.cuda.mem_get_info()
            return {
                "available": True,
                "allocated_mb": allocated // (1024 * 1024),
                "reserved_mb": reserved // (1024 * 1024),
                "free_mb": free // (1024 * 1024),
                "total_mb": total // (1024 * 1024),
            }
        except Exception:
            return {"available": False}


# =============================================================================
# Device Manager
# =============================================================================


class DeviceManager:
    """Unified device management for model and tensor placement.

    This class consolidates device detection and model movement logic
    that was previously duplicated across trainer_base.py and predict.py.

    Example:
        >>> device = DeviceManager.get_best_device()
        >>> model = DeviceManager.move_to_device(model)
    """

    _logger = get_logger("ins_pricing.device")
    _cached_device: Optional[Any] = None  # torch.device when available
    _cached_key: Optional[Tuple[bool, str]] = None

    @classmethod
    def _resolve_local_rank(cls, local_rank: Optional[int] = None) -> Optional[int]:
        if local_rank is not None:
            try:
                return int(local_rank)
            except (TypeError, ValueError):
                return None
        raw = os.environ.get("LOCAL_RANK")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @classmethod
    def get_best_device(
        cls,
        prefer_cuda: bool = True,
        local_rank: Optional[int] = None,
    ) -> Any:
        """Get the best available device.

        Args:
            prefer_cuda: If True, prefer CUDA over MPS
            local_rank: Optional CUDA local rank override

        Returns:
            Best available torch.device
        """
        if not TORCH_AVAILABLE:
            return None

        if prefer_cuda and torch.cuda.is_available():
            rank = cls._resolve_local_rank(local_rank)
            device_count = max(1, int(torch.cuda.device_count()))
            if rank is None:
                try:
                    rank = int(torch.cuda.current_device())
                except Exception:
                    rank = 0
            if rank < 0 or rank >= device_count:
                cls._logger.warning(
                    f"Invalid CUDA rank {rank}; falling back to cuda:0 "
                    f"(device_count={device_count})."
                )
                rank = 0
            device = torch.device(f"cuda:{rank}")
            cls._logger.debug(f"Selected device: {device}")
            return device

        cache_key = (bool(prefer_cuda), "non-cuda")
        if cls._cached_device is not None and cls._cached_key == cache_key:
            return cls._cached_device

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        cls._cached_key = cache_key
        cls._cached_device = device
        cls._logger.debug(f"Selected device: {device}")
        return device

    @classmethod
    def resolve_training_device(
        cls,
        *,
        is_ddp_enabled: bool = False,
        local_rank: int = 0,
        use_gpu: bool = True,
    ) -> Any:
        """Resolve training device with DDP-aware behavior."""
        if not TORCH_AVAILABLE:
            return None
        if not bool(use_gpu):
            return torch.device("cpu")
        if bool(is_ddp_enabled):
            return torch.device(f"cuda:{int(local_rank)}")
        return cls.get_best_device(prefer_cuda=True)

    @classmethod
    def move_to_device(cls, model_obj: Any, device: Optional[Any] = None) -> None:
        """Move a model object to the specified device.

        Handles sklearn-style wrappers that have .ft, .resnet, or .gnn attributes.

        Args:
            model_obj: Model object to move (may be sklearn wrapper)
            device: Target device (defaults to best available)
        """
        if model_obj is None:
            return

        device = device or cls.get_best_device()
        if device is None:
            return
        if TORCH_AVAILABLE and hasattr(device, "type") and device.type == "cuda":
            try:
                torch.cuda.set_device(device)
            except Exception:
                pass

        # Update device attribute if present
        if hasattr(model_obj, "device"):
            model_obj.device = device

        # Move the main model
        if hasattr(model_obj, "to"):
            model_obj.to(device)

        # Move nested submodules (sklearn wrappers)
        for attr_name in ("ft", "resnet", "gnn"):
            submodule = getattr(model_obj, attr_name, None)
            if submodule is not None and hasattr(submodule, "to"):
                submodule.to(device)

    @classmethod
    def unwrap_module(cls, module: Any) -> Any:
        """Unwrap DDP or DataParallel wrapper to get the base module.

        Args:
            module: Potentially wrapped PyTorch module

        Returns:
            Unwrapped base module
        """
        if not TORCH_AVAILABLE:
            return module

        if isinstance(module, (DDP, nn.DataParallel)):
            return module.module
        return module

    @classmethod
    def reset_cache(cls) -> None:
        """Reset cached device selection."""
        cls._cached_device = None
        cls._cached_key = None

    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA is available
        """
        return TORCH_AVAILABLE and torch.cuda.is_available()

    @classmethod
    def is_mps_available(cls) -> bool:
        """Check if MPS (Apple Silicon) is available.

        Returns:
            True if MPS is available
        """
        if not TORCH_AVAILABLE:
            return False
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
