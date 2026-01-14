"""Device management utilities for PyTorch models.

This module consolidates GPU/CPU device management logic from:
- modelling/core/bayesopt/utils.py
- modelling/core/bayesopt/trainers/trainer_base.py
- production/predict.py

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
from typing import Any, Dict, Optional

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

from .logging import get_logger


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
    def clean(cls, verbose: bool = False) -> None:
        """Clean up GPU memory.

        Args:
            verbose: If True, log cleanup details
        """
        gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if verbose:
                cls._logger.debug("CUDA cache cleared and synchronized")

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
    def cleanup_context(cls, verbose: bool = False):
        """Context manager that cleans GPU memory on exit.

        Args:
            verbose: If True, log cleanup details

        Yields:
            None
        """
        try:
            yield
        finally:
            cls.clean(verbose=verbose)

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

    @classmethod
    def get_best_device(cls, prefer_cuda: bool = True) -> Any:
        """Get the best available device.

        Args:
            prefer_cuda: If True, prefer CUDA over MPS

        Returns:
            Best available torch.device
        """
        if not TORCH_AVAILABLE:
            return None

        if cls._cached_device is not None:
            return cls._cached_device

        if prefer_cuda and torch.cuda.is_available():
            cls._cached_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls._cached_device = torch.device("mps")
        else:
            cls._cached_device = torch.device("cpu")

        cls._logger.debug(f"Selected device: {cls._cached_device}")
        return cls._cached_device

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
