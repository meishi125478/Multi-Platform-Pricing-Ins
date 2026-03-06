"""Utilities exposed for the bayesopt training stack."""

from __future__ import annotations

# Preserve package-style submodule imports, e.g.
# `import ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin as mixin_mod`.
from ins_pricing.modelling.bayesopt.utils import distributed_utils, torch_runtime, torch_trainer_mixin

# Shared utilities
from ins_pricing.utils import (
    EPS,
    set_global_seed,
    ensure_parent_dir,
    compute_batch_size,
    tweedie_loss,
    infer_factor_and_cate_list,
    IOUtils,
    csv_to_dict,
    get_logger,
    MetricFactory,
    GPUMemoryManager,
    DeviceManager,
)

# Distributed training
from ins_pricing.modelling.bayesopt.utils.distributed_utils import (
    DistributedUtils,
    TrainingUtils,
)

# PyTorch training mixin
from ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin import (
    TorchTrainerMixin,
)
from ins_pricing.modelling.bayesopt.utils.torch_runtime import (
    create_autocast_context,
    create_grad_scaler,
    setup_ddp_if_requested,
    resolve_training_device,
    wrap_model_for_parallel,
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
    # PyTorch
    'TorchTrainerMixin',
    'create_autocast_context',
    'create_grad_scaler',
    'setup_ddp_if_requested',
    'resolve_training_device',
    'wrap_model_for_parallel',
    # Utilities
    'get_logger',
    'MetricFactory',
    'GPUMemoryManager',
    'DeviceManager',
    # Submodules
    'distributed_utils',
    'torch_runtime',
    'torch_trainer_mixin',
]
