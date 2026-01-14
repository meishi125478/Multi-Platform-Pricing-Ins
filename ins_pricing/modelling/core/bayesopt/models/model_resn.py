from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset

from ..utils import DistributedUtils, EPS, TorchTrainerMixin


# =============================================================================
# ResNet model and sklearn-style wrapper
# =============================================================================

# ResNet model definition
# Residual block: two linear layers + ReLU + residual connection
# ResBlock inherits nn.Module
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1,
                 use_layernorm: bool = False, residual_scale: float = 0.1,
                 stochastic_depth: float = 0.0
                 ):
        super().__init__()
        self.use_layernorm = use_layernorm

        if use_layernorm:
            Norm = nn.LayerNorm      # Normalize the last dimension
        else:
            def Norm(d): return nn.BatchNorm1d(d)  # Keep a switch to try BN

        self.norm1 = Norm(dim)
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        # Enable post-second-layer norm if needed: self.norm2 = Norm(dim)
        self.fc2 = nn.Linear(dim, dim, bias=True)

        # Residual scaling to stabilize early training
        self.res_scale = nn.Parameter(
            torch.tensor(residual_scale, dtype=torch.float32)
        )
        self.stochastic_depth = max(0.0, float(stochastic_depth))

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.stochastic_depth <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.stochastic_depth
        if keep_prob <= 0.0:
            return torch.zeros_like(x)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x * binary_tensor / keep_prob

    def forward(self, x):
        # Pre-activation structure
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.dropout(out)
        # If a second norm is enabled: out = self.norm2(out)
        out = self.fc2(out)
        # Apply residual scaling then add
        out = self.res_scale * out
        out = self._drop_path(out)
        return x + out

# ResNetSequential defines the full network


class ResNetSequential(nn.Module):
    # Input shape: (batch, input_dim)
    # Network: FC + norm + ReLU, stack residual blocks, output Softplus

    def __init__(self, input_dim: int, hidden_dim: int = 64, block_num: int = 2,
                 use_layernorm: bool = True, dropout: float = 0.1,
                 residual_scale: float = 0.1, stochastic_depth: float = 0.0,
                 task_type: str = 'regression'):
        super(ResNetSequential, self).__init__()

        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(input_dim, hidden_dim))

        # Optional explicit normalization after the first layer:
        # For LayerNorm:
        #     self.net.add_module('norm1', nn.LayerNorm(hidden_dim))
        # Or BatchNorm:
        #     self.net.add_module('norm1', nn.BatchNorm1d(hidden_dim))

        # If desired, insert ReLU before residual blocks:
        # self.net.add_module('relu1', nn.ReLU(inplace=True))

        # Residual blocks
        drop_path_rate = max(0.0, float(stochastic_depth))
        for i in range(block_num):
            if block_num > 1:
                block_drop = drop_path_rate * (i / (block_num - 1))
            else:
                block_drop = drop_path_rate
            self.net.add_module(
                f'ResBlk_{i+1}',
                ResBlock(
                    hidden_dim,
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                    residual_scale=residual_scale,
                    stochastic_depth=block_drop)
            )

        self.net.add_module('fc_out', nn.Linear(hidden_dim, 1))

        if task_type == 'classification':
            self.net.add_module('softplus', nn.Identity())
        else:
            self.net.add_module('softplus', nn.Softplus())

    def forward(self, x):
        if self.training and not hasattr(self, '_printed_device'):
            print(f">>> ResNetSequential executing on device: {x.device}")
            self._printed_device = True
        return self.net(x)

# Define the ResNet sklearn-style wrapper.


class ResNetSklearn(TorchTrainerMixin, nn.Module):
    def __init__(self, model_nme: str, input_dim: int, hidden_dim: int = 64,
                 block_num: int = 2, batch_num: int = 100, epochs: int = 100,
                 task_type: str = 'regression',
                 tweedie_power: float = 1.5, learning_rate: float = 0.01, patience: int = 10,
                 use_layernorm: bool = True, dropout: float = 0.1,
                 residual_scale: float = 0.1,
                 stochastic_depth: float = 0.0,
                 weight_decay: float = 1e-4,
                 use_data_parallel: bool = True,
                 use_ddp: bool = False):
        super(ResNetSklearn, self).__init__()

        self.use_ddp = use_ddp
        self.is_ddp_enabled, self.local_rank, self.rank, self.world_size = (
            False, 0, 0, 1)

        if self.use_ddp:
            self.is_ddp_enabled, self.local_rank, self.rank, self.world_size = DistributedUtils.setup_ddp()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.batch_num = batch_num
        self.epochs = epochs
        self.task_type = task_type
        self.model_nme = model_nme
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.use_layernorm = use_layernorm
        self.dropout = dropout
        self.residual_scale = residual_scale
        self.stochastic_depth = max(0.0, float(stochastic_depth))
        self.loss_curve_path: Optional[str] = None
        self.training_history: Dict[str, List[float]] = {
            "train": [], "val": []}
        self.use_data_parallel = bool(use_data_parallel)

        # Device selection: cuda > mps > cpu
        if self.is_ddp_enabled:
            self.device = torch.device(f'cuda:{self.local_rank}')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Tweedie power (unused for classification)
        if self.task_type == 'classification':
            self.tw_power = None
        elif 'f' in self.model_nme:
            self.tw_power = 1
        elif 's' in self.model_nme:
            self.tw_power = 2
        else:
            self.tw_power = tweedie_power

        # Build network (construct on CPU first)
        core = ResNetSequential(
            self.input_dim,
            self.hidden_dim,
            self.block_num,
            use_layernorm=self.use_layernorm,
            dropout=self.dropout,
            residual_scale=self.residual_scale,
            stochastic_depth=self.stochastic_depth,
            task_type=self.task_type
        )

        # ===== Multi-GPU: DataParallel vs DistributedDataParallel =====
        if self.is_ddp_enabled:
            core = core.to(self.device)
            core = DDP(core, device_ids=[
                       self.local_rank], output_device=self.local_rank)
            self.use_data_parallel = False
        elif use_data_parallel and (self.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            if self.use_ddp and not self.is_ddp_enabled:
                print(
                    ">>> DDP requested but not initialized; falling back to DataParallel.")
            core = nn.DataParallel(core, device_ids=list(
                range(torch.cuda.device_count())))
            # DataParallel scatters inputs, but the primary device remains cuda:0.
            self.device = torch.device('cuda')
            self.use_data_parallel = True
        else:
            self.use_data_parallel = False

        self.resnet = core.to(self.device)

    # ================ Internal helpers ================
    @staticmethod
    def _validate_vector(arr, name: str, n_rows: int) -> None:
        if arr is None:
            return
        if isinstance(arr, pd.DataFrame):
            if arr.shape[1] != 1:
                raise ValueError(f"{name} must be 1d (single column).")
            length = len(arr)
        else:
            arr_np = np.asarray(arr)
            if arr_np.ndim == 0:
                raise ValueError(f"{name} must be 1d.")
            if arr_np.ndim > 2 or (arr_np.ndim == 2 and arr_np.shape[1] != 1):
                raise ValueError(f"{name} must be 1d or Nx1.")
            length = arr_np.shape[0]
        if length != n_rows:
            raise ValueError(
                f"{name} length {length} does not match X length {n_rows}."
            )

    def _validate_inputs(self, X, y, w, label: str) -> None:
        if X is None:
            raise ValueError(f"{label} X cannot be None.")
        n_rows = len(X)
        if y is None:
            raise ValueError(f"{label} y cannot be None.")
        self._validate_vector(y, f"{label} y", n_rows)
        self._validate_vector(w, f"{label} w", n_rows)

    def _build_train_val_tensors(self, X_train, y_train, w_train, X_val, y_val, w_val):
        self._validate_inputs(X_train, y_train, w_train, "train")
        if X_val is not None or y_val is not None or w_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("validation X and y must both be provided.")
            self._validate_inputs(X_val, y_val, w_val, "val")

        def _to_numpy(arr):
            if hasattr(arr, "to_numpy"):
                return arr.to_numpy(dtype=np.float32, copy=False)
            return np.asarray(arr, dtype=np.float32)

        X_tensor = torch.as_tensor(_to_numpy(X_train))
        y_tensor = torch.as_tensor(_to_numpy(y_train)).view(-1, 1)
        w_tensor = (
            torch.as_tensor(_to_numpy(w_train)).view(-1, 1)
            if w_train is not None else torch.ones_like(y_tensor)
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_tensor = torch.as_tensor(_to_numpy(X_val))
            y_val_tensor = torch.as_tensor(_to_numpy(y_val)).view(-1, 1)
            w_val_tensor = (
                torch.as_tensor(_to_numpy(w_val)).view(-1, 1)
                if w_val is not None else torch.ones_like(y_val_tensor)
            )
        else:
            X_val_tensor = y_val_tensor = w_val_tensor = None
        return X_tensor, y_tensor, w_tensor, X_val_tensor, y_val_tensor, w_val_tensor, has_val

    def forward(self, x):
        # Handle SHAP NumPy input.
        if isinstance(x, np.ndarray):
            x_tensor = torch.as_tensor(x, dtype=torch.float32)
        else:
            x_tensor = x

        x_tensor = x_tensor.to(self.device)
        y_pred = self.resnet(x_tensor)
        return y_pred

    # ---------------- Training ----------------

    def fit(self, X_train, y_train, w_train=None,
            X_val=None, y_val=None, w_val=None, trial=None):

        X_tensor, y_tensor, w_tensor, X_val_tensor, y_val_tensor, w_val_tensor, has_val = \
            self._build_train_val_tensors(
                X_train, y_train, w_train, X_val, y_val, w_val)

        dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        dataloader, accum_steps = self._build_dataloader(
            dataset,
            N=X_tensor.shape[0],
            base_bs_gpu=(2048, 1024, 512),
            base_bs_cpu=(256, 128),
            min_bs=64,
            target_effective_cuda=2048,
            target_effective_cpu=1024
        )

        # Set sampler epoch at the start of each epoch to keep shuffling deterministic.
        if self.is_ddp_enabled and hasattr(dataloader.sampler, 'set_epoch'):
            self.dataloader_sampler = dataloader.sampler
        else:
            self.dataloader_sampler = None

        # === 4. Optimizer and AMP ===
        self.optimizer = torch.optim.Adam(
            self.resnet.parameters(),
            lr=self.learning_rate,
            weight_decay=float(self.weight_decay),
        )
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))

        X_val_dev = y_val_dev = w_val_dev = None
        val_dataloader = None
        if has_val:
            # Build validation DataLoader.
            val_dataset = TensorDataset(
                X_val_tensor, y_val_tensor, w_val_tensor)
            # No backward pass in validation; batch size can be larger for throughput.
            val_dataloader = self._build_val_dataloader(
                val_dataset, dataloader, accum_steps)
            # Validation usually does not need a DDP sampler because we validate on the main process
            # or aggregate results. For simplicity, keep validation on a single GPU or the main process.

        is_data_parallel = isinstance(self.resnet, nn.DataParallel)

        def forward_fn(batch):
            X_batch, y_batch, w_batch = batch

            if not is_data_parallel:
                X_batch = X_batch.to(self.device, non_blocking=True)
            # Keep targets and weights on the main device for loss computation.
            y_batch = y_batch.to(self.device, non_blocking=True)
            w_batch = w_batch.to(self.device, non_blocking=True)

            y_pred = self.resnet(X_batch)
            return y_pred, y_batch, w_batch

        def val_forward_fn():
            total_loss = 0.0
            total_weight = 0.0
            for batch in val_dataloader:
                X_b, y_b, w_b = batch
                if not is_data_parallel:
                    X_b = X_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)
                w_b = w_b.to(self.device, non_blocking=True)

                y_pred = self.resnet(X_b)

                # Manually compute weighted loss for accurate aggregation.
                losses = self._compute_losses(
                    y_pred, y_b, apply_softplus=False)

                batch_weight_sum = torch.clamp(w_b.sum(), min=EPS)
                batch_weighted_loss_sum = (losses * w_b.view(-1)).sum()

                total_loss += batch_weighted_loss_sum.item()
                total_weight += batch_weight_sum.item()

            return total_loss / max(total_weight, EPS)

        clip_fn = None
        if self.device.type == 'cuda':
            def clip_fn(): return (self.scaler.unscale_(self.optimizer),
                                   clip_grad_norm_(self.resnet.parameters(), max_norm=1.0))

        # Under DDP, only the main process prints logs and saves models.
        if self.is_ddp_enabled and not DistributedUtils.is_main_process():
            # Non-main processes skip validation callback logging (handled inside _train_model).
            pass

        best_state, history = self._train_model(
            self.resnet,
            dataloader,
            accum_steps,
            self.optimizer,
            self.scaler,
            forward_fn,
            val_forward_fn if has_val else None,
            apply_softplus=False,
            clip_fn=clip_fn,
            trial=trial,
            loss_curve_path=getattr(self, "loss_curve_path", None)
        )

        if has_val and best_state is not None:
            # Load state into unwrapped module to match how it was saved
            base_module = self.resnet.module if hasattr(self.resnet, "module") else self.resnet
            base_module.load_state_dict(best_state)
        self.training_history = history

    # ---------------- Prediction ----------------

    def predict(self, X_test):
        self.resnet.eval()
        if isinstance(X_test, pd.DataFrame):
            X_np = X_test.to_numpy(dtype=np.float32, copy=False)
        else:
            X_np = np.asarray(X_test, dtype=np.float32)

        inference_cm = getattr(torch, "inference_mode", torch.no_grad)
        with inference_cm():
            y_pred = self(X_np).cpu().numpy()

        if self.task_type == 'classification':
            y_pred = 1 / (1 + np.exp(-y_pred))  # Sigmoid converts logits to probabilities.
        else:
            y_pred = np.clip(y_pred, 1e-6, None)
        return y_pred.flatten()

    # ---------------- Set Params ----------------

    def set_params(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in model.")
        return self
