"""Backward compatibility re-exports for bayesopt utilities.

This module keeps legacy imports working while routing general helpers
through ins_pricing.utils and leaving bayesopt-specific utilities in place.
"""

from __future__ import annotations

# Constants and simple utilities
from ins_pricing.modelling.bayesopt.utils.constants import (
    EPS,
    set_global_seed,
    ensure_parent_dir,
    compute_batch_size,
    tweedie_loss,
    infer_factor_and_cate_list,
)

# I/O utilities
from ins_pricing.modelling.bayesopt.utils.io_utils import (
    IOUtils,
    csv_to_dict,
)

# Distributed training
from ins_pricing.modelling.bayesopt.utils.distributed_utils import (
    DistributedUtils,
    TrainingUtils,
    free_cuda,
)

# PyTorch training mixin
from ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin import (
    TorchTrainerMixin,
)

# Metrics and device helpers (shared utilities)
from ins_pricing.modelling.bayesopt.utils.metrics_and_devices import (
    get_logger,
    MetricFactory,
    GPUMemoryManager,
    DeviceManager,
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
]
