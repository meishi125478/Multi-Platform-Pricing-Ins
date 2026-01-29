"""Performance profiling and memory monitoring utilities.

This module provides tools for tracking execution time, memory usage,
and GPU resources during model training and data processing.

Example:
    >>> from ins_pricing.utils.profiling import profile_section
    >>> with profile_section("Data Loading", logger=my_logger):
    ...     data = load_large_dataset()
    [Profile] Data Loading: 5.23s, RAM: +1250.3MB, GPU peak: 2048.5MB
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@contextmanager
def profile_section(
    name: str,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.INFO
):
    """Context manager for profiling code sections.

    Tracks execution time, RAM usage, and GPU memory (if available).
    Logs results when the context exits.

    Args:
        name: Name of the section being profiled
        logger: Optional logger instance. If None, prints to stdout
        log_level: Logging level (default: INFO)

    Yields:
        None

    Example:
        >>> with profile_section("Training Loop", logger):
        ...     model.fit(X_train, y_train)
        [Profile] Training Loop: 45.2s, RAM: +2100.5MB, GPU peak: 4096.0MB

        >>> with profile_section("Preprocessing"):
        ...     df = preprocess_data(raw_df)
        [Profile] Preprocessing: 2.1s, RAM: +150.2MB
    """
    start_time = time.time()

    # Track CPU memory
    start_mem = None
    if HAS_PSUTIL:
        start_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # Track GPU memory
    start_gpu_mem = None
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    try:
        yield
    finally:
        elapsed = time.time() - start_time

        # Build profiling message
        msg_parts = [f"[Profile] {name}: {elapsed:.2f}s"]

        # Add RAM usage
        if HAS_PSUTIL and start_mem is not None:
            end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            mem_delta = end_mem - start_mem
            msg_parts.append(f"RAM: {mem_delta:+.1f}MB")

        # Add GPU memory
        if HAS_TORCH and torch.cuda.is_available() and start_gpu_mem is not None:
            peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024
            msg_parts.append(f"GPU peak: {peak_gpu:.1f}MB")

        msg = ", ".join(msg_parts)

        if logger:
            logger.log(log_level, msg)
        else:
            print(msg)


def get_memory_info() -> dict:
    """Get current memory usage information.

    Returns:
        Dictionary with memory statistics:
        - rss_mb: Resident Set Size in MB (physical memory)
        - vms_mb: Virtual Memory Size in MB
        - percent: Memory usage percentage
        - available_mb: Available system memory in MB
        - gpu_allocated_mb: GPU memory allocated (if CUDA available)
        - gpu_cached_mb: GPU memory cached (if CUDA available)

    Example:
        >>> info = get_memory_info()
        >>> print(f"Using {info['rss_mb']:.1f} MB RAM")
        Using 2048.5 MB RAM
    """
    info = {}

    if HAS_PSUTIL:
        process = psutil.Process()
        mem = process.memory_info()
        info['rss_mb'] = mem.rss / 1024 / 1024
        info['vms_mb'] = mem.vms / 1024 / 1024

        vm = psutil.virtual_memory()
        info['percent'] = vm.percent
        info['available_mb'] = vm.available / 1024 / 1024
    else:
        info['warning'] = 'psutil not available'

    if HAS_TORCH and torch.cuda.is_available():
        info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        info['gpu_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024

    return info


def log_memory_usage(
    logger: logging.Logger,
    prefix: str = "",
    level: int = logging.INFO
) -> None:
    """Log current memory usage.

    Args:
        logger: Logger instance
        prefix: Optional prefix for log message
        level: Logging level (default: INFO)

    Example:
        >>> log_memory_usage(logger, prefix="After epoch 10")
        After epoch 10 - Memory: RSS=2048.5MB, GPU=1024.0MB
    """
    info = get_memory_info()

    if 'warning' in info:
        logger.log(level, f"{prefix} - Memory info not available (psutil missing)")
        return

    msg_parts = []
    if prefix:
        msg_parts.append(prefix)

    ram_msg = f"RSS={info['rss_mb']:.1f}MB ({info['percent']:.1f}%)"
    msg_parts.append(f"Memory: {ram_msg}")

    if 'gpu_allocated_mb' in info:
        gpu_msg = f"GPU={info['gpu_allocated_mb']:.1f}MB"
        msg_parts.append(gpu_msg)

    logger.log(level, " - ".join(msg_parts))


def check_memory_threshold(
    threshold_gb: float = 32.0,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Check if memory usage exceeds threshold.

    Args:
        threshold_gb: Memory threshold in GB (default: 32.0)
        logger: Optional logger for warnings

    Returns:
        True if memory usage exceeds threshold, False otherwise

    Example:
        >>> if check_memory_threshold(threshold_gb=16.0, logger=logger):
        ...     torch.cuda.empty_cache()
        ...     gc.collect()
    """
    if not HAS_PSUTIL:
        return False

    mem = psutil.Process().memory_info()
    rss_gb = mem.rss / 1024 / 1024 / 1024

    if rss_gb > threshold_gb:
        if logger:
            logger.warning(
                f"High memory usage detected: {rss_gb:.1f}GB "
                f"(threshold: {threshold_gb:.1f}GB)"
            )
        return True

    return False


def cleanup_memory(logger: Optional[logging.Logger] = None) -> None:
    """Force memory cleanup for CPU and GPU.

    Args:
        logger: Optional logger instance

    Example:
        >>> cleanup_memory(logger)
        [Memory] Cleanup: freed 250.5MB RAM, 512.0MB GPU
    """
    if HAS_PSUTIL:
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024

    gpu_before = None
    if HAS_TORCH and torch.cuda.is_available():
        gpu_before = torch.cuda.memory_allocated() / 1024 / 1024

    # Perform cleanup
    gc.collect()

    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate freed memory
    msg_parts = ["[Memory] Cleanup:"]

    if HAS_PSUTIL:
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        mem_freed = mem_before - mem_after
        msg_parts.append(f"freed {mem_freed:.1f}MB RAM")

    if gpu_before is not None:
        gpu_after = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_freed = gpu_before - gpu_after
        msg_parts.append(f"{gpu_freed:.1f}MB GPU")

    msg = ", ".join(msg_parts)

    if logger:
        logger.info(msg)
    else:
        print(msg)


class MemoryMonitor:
    """Memory monitoring context manager with automatic cleanup.

    Monitors memory usage and optionally triggers cleanup if threshold exceeded.

    Args:
        name: Name of the monitored section
        threshold_gb: Memory threshold for automatic cleanup (default: None, no cleanup)
        logger: Optional logger instance

    Example:
        >>> with MemoryMonitor("Training", threshold_gb=16.0, logger=logger):
        ...     for epoch in range(100):
        ...         train_epoch(model, data)
        [Memory] Training started: RAM=1024.5MB, GPU=512.0MB
        [Memory] Training completed: RAM=2048.3MB (+1023.8MB), GPU=2048.0MB (+1536.0MB)
    """

    def __init__(
        self,
        name: str,
        threshold_gb: Optional[float] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.threshold_gb = threshold_gb
        self.logger = logger
        self.start_mem = None
        self.start_gpu = None

    def __enter__(self):
        if HAS_PSUTIL:
            self.start_mem = psutil.Process().memory_info().rss / 1024 / 1024

        if HAS_TORCH and torch.cuda.is_available():
            self.start_gpu = torch.cuda.memory_allocated() / 1024 / 1024

        # Log starting state
        msg_parts = [f"[Memory] {self.name} started:"]
        if self.start_mem is not None:
            msg_parts.append(f"RAM={self.start_mem:.1f}MB")
        if self.start_gpu is not None:
            msg_parts.append(f"GPU={self.start_gpu:.1f}MB")

        msg = ", ".join(msg_parts)
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate deltas
        msg_parts = [f"[Memory] {self.name} completed:"]

        if HAS_PSUTIL and self.start_mem is not None:
            end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            delta_mem = end_mem - self.start_mem
            msg_parts.append(f"RAM={end_mem:.1f}MB ({delta_mem:+.1f}MB)")

        if HAS_TORCH and torch.cuda.is_available() and self.start_gpu is not None:
            end_gpu = torch.cuda.memory_allocated() / 1024 / 1024
            delta_gpu = end_gpu - self.start_gpu
            msg_parts.append(f"GPU={end_gpu:.1f}MB ({delta_gpu:+.1f}MB)")

        msg = ", ".join(msg_parts)
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

        # Check threshold and cleanup if needed
        if self.threshold_gb is not None:
            if check_memory_threshold(self.threshold_gb, self.logger):
                cleanup_memory(self.logger)


def profile_training_epoch(
    epoch: int,
    total_epochs: int,
    logger: Optional[logging.Logger] = None,
    cleanup_interval: int = 10
) -> None:
    """Log memory usage during training epochs with periodic cleanup.

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        logger: Optional logger instance
        cleanup_interval: Cleanup memory every N epochs (default: 10)

    Example:
        >>> for epoch in range(1, 101):
        ...     train_one_epoch(model, data)
        ...     profile_training_epoch(epoch, 100, logger, cleanup_interval=10)
    """
    log_memory_usage(
        logger or logging.getLogger(__name__),
        prefix=f"Epoch {epoch}/{total_epochs}",
        level=logging.DEBUG
    )

    # Periodic cleanup
    if epoch % cleanup_interval == 0:
        if logger:
            logger.info(f"Epoch {epoch}: Performing periodic memory cleanup")
        cleanup_memory(logger)


# Convenience function for backward compatibility
def ensure_memory_cleanup(threshold_gb: float = 32.0) -> None:
    """Check memory and cleanup if needed (simple function interface).

    Args:
        threshold_gb: Memory threshold in GB

    Example:
        >>> ensure_memory_cleanup(threshold_gb=16.0)
    """
    if check_memory_threshold(threshold_gb):
        cleanup_memory()
