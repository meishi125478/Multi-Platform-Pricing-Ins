"""Backward compatibility re-exports from refactored utils modules.

This module ensures all existing imports continue to work:
    from ins_pricing.modelling.core.bayesopt.utils import EPS, IOUtils, ...

The utils.py file has been split into focused modules for better maintainability:
- constants.py: EPS, set_global_seed, etc.
- io_utils.py: IOUtils for file I/O
- distributed_utils.py: DistributedUtils, TrainingUtils for DDP
- torch_trainer_mixin.py: TorchTrainerMixin for PyTorch training
- metrics_and_devices.py: Metrics, GPU/device management, CV strategies, plotting
"""

from __future__ import annotations

# Constants and simple utilities
from .constants import (
    EPS,
    set_global_seed,
    ensure_parent_dir,
    compute_batch_size,
    tweedie_loss,
    infer_factor_and_cate_list,
)

# I/O utilities
from .io_utils import (
    IOUtils,
    csv_to_dict,
)

# Distributed training
from .distributed_utils import (
    DistributedUtils,
    TrainingUtils,
    free_cuda,
)

# PyTorch training mixin
from .torch_trainer_mixin import (
    TorchTrainerMixin,
)

# Metrics, devices, CV, and plotting
from .metrics_and_devices import (
    get_logger,
    MetricFactory,
    GPUMemoryManager,
    DeviceManager,
    CVStrategyResolver,
    PlotUtils,
    split_data,
    plot_lift_list,
    plot_dlift_list,
    _OrderedSplitter,
)

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
