"""Shared utilities for the ins_pricing package.

This module provides common utilities used across all submodules:
- Logging: Unified logging system with configurable levels
- Metrics: PSI calculation, model evaluation metrics
- Paths: Path resolution and data loading utilities
- Device: GPU/CPU device management for PyTorch models

Example:
    >>> from ins_pricing.utils import get_logger, psi_report
    >>> logger = get_logger("my_module")
    >>> logger.info("Processing started")
"""

from __future__ import annotations

# =============================================================================
# Logging utilities
# =============================================================================
from .logging import get_logger, configure_logging

# =============================================================================
# Metric utilities (PSI, model evaluation)
# =============================================================================
from .metrics import (
    psi_numeric,
    psi_categorical,
    population_stability_index,
    psi_report,
    MetricFactory,
)

# =============================================================================
# Path utilities
# =============================================================================
from .paths import (
    resolve_path,
    resolve_dir_path,
    resolve_data_path,
    load_dataset,
    coerce_dataset_types,
    dedupe_preserve_order,
    build_model_names,
    parse_model_pairs,
    fingerprint_file,
    PLOT_MODEL_LABELS,
    PYTORCH_TRAINERS,
)

# =============================================================================
# Device management (GPU/CPU)
# =============================================================================
from .device import (
    DeviceManager,
    GPUMemoryManager,
)

__all__ = [
    # Logging
    "get_logger",
    "configure_logging",
    # Metrics
    "psi_numeric",
    "psi_categorical",
    "population_stability_index",
    "psi_report",
    "MetricFactory",
    # Paths
    "resolve_path",
    "resolve_dir_path",
    "resolve_data_path",
    "load_dataset",
    "coerce_dataset_types",
    "dedupe_preserve_order",
    "build_model_names",
    "parse_model_pairs",
    "fingerprint_file",
    "PLOT_MODEL_LABELS",
    "PYTORCH_TRAINERS",
    # Device
    "DeviceManager",
    "GPUMemoryManager",
]
