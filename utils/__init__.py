"""Shared utilities for the ins_pricing package.

This module provides common utilities used across all submodules:
- Logging: Unified logging system with configurable levels
- Metrics: PSI calculation, model evaluation metrics
- Paths: Path resolution and data loading utilities
- IO: File helpers and parameter loading
- Numerics: EPS, tweedie loss, adaptive batch sizing
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
from ins_pricing.utils.logging import get_logger, configure_logging, log_print

# =============================================================================
# Metric utilities (PSI, model evaluation)
# =============================================================================
from ins_pricing.utils.metrics import (
    psi_numeric,
    psi_categorical,
    population_stability_index,
    psi_report,
    MetricFactory,
)

# =============================================================================
# Numerical helpers
# =============================================================================
from ins_pricing.utils.numerics import (
    EPS,
    set_global_seed,
    compute_batch_size,
    tweedie_loss,
)

# =============================================================================
# Feature inference
# =============================================================================
from ins_pricing.utils.features import infer_factor_and_cate_list

# =============================================================================
# IO helpers
# =============================================================================
from ins_pricing.utils.io import IOUtils, csv_to_dict, ensure_parent_dir

# =============================================================================
# Path utilities
# =============================================================================
from ins_pricing.utils.paths import (
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
from ins_pricing.utils.device import DeviceManager, GPUMemoryManager

__all__ = [
    # Logging
    "get_logger",
    "configure_logging",
    "log_print",
    # Metrics
    "psi_numeric",
    "psi_categorical",
    "population_stability_index",
    "psi_report",
    "MetricFactory",
    # Numerics
    "EPS",
    "set_global_seed",
    "compute_batch_size",
    "tweedie_loss",
    # Features
    "infer_factor_and_cate_list",
    # IO
    "IOUtils",
    "csv_to_dict",
    "ensure_parent_dir",
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
