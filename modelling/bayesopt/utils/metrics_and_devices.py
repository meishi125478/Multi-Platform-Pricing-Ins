"""Backward-compatible re-exports for metrics and device utilities."""

from __future__ import annotations

from ins_pricing.utils import (
    DeviceManager,
    GPUMemoryManager,
    MetricFactory,
    get_logger,
)

__all__ = [
    "get_logger",
    "MetricFactory",
    "GPUMemoryManager",
    "DeviceManager",
]
