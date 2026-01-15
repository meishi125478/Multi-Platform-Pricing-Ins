"""DEPRECATED: Backward compatibility wrapper.

This module is kept for backward compatibility but will be removed in v0.4.0.
The monolithic utils.py (1,503 lines) has been split into focused modules:

    utils/
    ├── constants.py          - EPS, set_global_seed, etc.
    ├── io_utils.py           - IOUtils for file operations
    ├── distributed_utils.py  - DistributedUtils, TrainingUtils
    ├── torch_trainer_mixin.py - TorchTrainerMixin for PyTorch
    └── metrics_and_devices.py - Metrics, GPU, device, CV, plotting

All imports still work the same way:
    from ins_pricing.modelling.core.bayesopt.utils import EPS, IOUtils

Or use the new package directly:
    from ins_pricing.modelling.core.bayesopt.utils import EPS, IOUtils
    from ins_pricing.modelling.core.bayesopt.utils.constants import EPS

Both will work identically. The old single-file import will show a deprecation
warning but continues to function.
"""

from __future__ import annotations

import warnings

# Show deprecation warning
warnings.warn(
    "Importing from bayesopt.utils (single file) is deprecated. "
    "This file will be removed in v0.4.0. "
    "The utils module has been split into focused submodules for better maintainability. "
    "Imports will continue to work from the utils package without changes.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from utils package for backward compatibility
from .utils import *  # noqa: F401, F403

# Explicitly list all exports to support IDE auto-completion
__all__ = [
    # Constants
    'EPS',
    'set_global_seed',
    'ensure_parent_dir',
    'compute_batch_size',
    'tweedie_loss',
    'infer_factor_and_cate_list',
    # I/O
    'IOUtils',
    'csv_to_dict',
    # Distributed
    'DistributedUtils',
    'TrainingUtils',
    'free_cuda',
    # PyTorch
    'TorchTrainerMixin',
    # Utilities
    'get_logger',
    'MetricFactory',
    'GPUMemoryManager',
    'DeviceManager',
    'CVStrategyResolver',
    'PlotUtils',
    'split_data',
    'plot_lift_list',
    'plot_dlift_list',
    '_OrderedSplitter',
]
