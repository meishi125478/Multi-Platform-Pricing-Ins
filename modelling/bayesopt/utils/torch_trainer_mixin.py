"""PyTorch training mixin with resource management and training loops.

This module provides the TorchTrainerMixin class which is used by
PyTorch-based trainers (ResNet, FT, GNN) for:
- Resource profiling and memory management
- Batch size computation and optimization
- DataLoader creation with DDP support
- Generic training and validation loops with AMP
- Early stopping and loss curve plotting
"""

from __future__ import annotations

import copy
import ctypes
import gc
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Try to import plotting functions
try:
    import matplotlib
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    matplotlib = None
    plt = None
    _MPL_IMPORT_ERROR = exc

try:
    from ins_pricing.modelling.plotting.diagnostics import plot_loss_curve as plot_loss_curve_common
except Exception:
    try:
        from ins_pricing.plotting.diagnostics import plot_loss_curve as plot_loss_curve_common
    except Exception:
        plot_loss_curve_common = None

# Import from other utils modules
from ins_pricing.utils import (
    EPS,
    compute_batch_size,
    tweedie_loss,
    ensure_parent_dir,
    get_logger,
    log_print,
)
from ins_pricing.utils.losses import (
    infer_loss_name_from_model_name,
    loss_requires_positive,
    normalize_loss_name,
    resolve_tweedie_power,
)
from ins_pricing.modelling.bayesopt.utils.distributed_utils import DistributedUtils

_logger = get_logger("ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


def _plot_skip(label: str) -> None:
    """Print message when plot is skipped due to missing matplotlib."""
    if _MPL_IMPORT_ERROR is not None:
        _log(f"[Plot] Skip {label}: matplotlib unavailable ({_MPL_IMPORT_ERROR}).", flush=True)
    else:
        _log(f"[Plot] Skip {label}: matplotlib unavailable.", flush=True)


class TorchTrainerMixin:
    """Shared helpers for PyTorch tabular trainers.

    Provides resource profiling, memory management, batch size optimization,
    and standardized training loops with mixed precision and DDP support.

    This mixin is used by ResNetTrainer, FTTrainer, and GNNTrainer.
    """

    def _resolve_device(self) -> torch.device:
        """Resolve device to a torch.device instance."""
        device = getattr(self, "device", None)
        if device is None:
            return torch.device("cpu")
        return device if isinstance(device, torch.device) else torch.device(device)

    def _device_type(self) -> str:
        """Get device type (cpu/cuda/mps)."""
        return self._resolve_device().type

    def _resolve_resource_profile(self) -> str:
        """Determine resource usage profile.

        Returns:
            One of: 'throughput', 'memory_saving', or 'auto'
        """
        profile = getattr(self, "resource_profile", None)
        if not profile:
            profile = os.environ.get("BAYESOPT_RESOURCE_PROFILE", "auto")
        profile = str(profile).strip().lower()
        if profile in {"cpu", "mps", "cuda"}:
            profile = "auto"
        if profile not in {"auto", "throughput", "memory_saving"}:
            profile = "auto"
        if profile == "auto" and self._device_type() == "cuda":
            profile = "throughput"
        return profile

    def _log_resource_summary_once(self, profile: str) -> None:
        """Log resource configuration summary once."""
        if getattr(self, "_resource_summary_logged", False):
            return
        if dist.is_initialized() and not DistributedUtils.is_main_process():
            return
        self._resource_summary_logged = True
        device = self._resolve_device()
        device_type = self._device_type()
        cpu_count = os.cpu_count() or 1
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        ddp_enabled = bool(getattr(self, "is_ddp_enabled", False))
        data_parallel = bool(getattr(self, "use_data_parallel", False))
        _log(
            f">>> Resource summary: device={device}, device_type={device_type}, "
            f"cpu_count={cpu_count}, cuda_count={cuda_count}, mps={mps_available}, "
            f"ddp={ddp_enabled}, data_parallel={data_parallel}, profile={profile}"
        )

    def _available_system_memory(self) -> Optional[int]:
        """Get available system RAM in bytes."""
        if os.name == "nt":
            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return None
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            return None

    def _available_cuda_memory(self) -> Optional[int]:
        """Get available CUDA memory in bytes."""
        if not torch.cuda.is_available():
            return None
        try:
            free_mem, _total_mem = torch.cuda.mem_get_info()
        except Exception:
            return None
        return int(free_mem)

    def _estimate_sample_bytes(self, dataset) -> Optional[int]:
        """Estimate memory per sample in bytes."""
        try:
            if len(dataset) == 0:
                return None
            sample = dataset[0]
        except Exception:
            return None

        def _bytes(obj) -> int:
            if obj is None:
                return 0
            if torch.is_tensor(obj):
                return int(obj.element_size() * obj.nelement())
            if isinstance(obj, np.ndarray):
                return int(obj.nbytes)
            if isinstance(obj, (list, tuple)):
                return int(sum(_bytes(item) for item in obj))
            if isinstance(obj, dict):
                return int(sum(_bytes(item) for item in obj.values()))
            return 0

        sample_bytes = _bytes(sample)
        return int(sample_bytes) if sample_bytes > 0 else None

    def _cap_batch_size_by_memory(self, dataset, batch_size: int, profile: str) -> int:
        """Cap batch size based on available memory."""
        if batch_size <= 1:
            return batch_size
        sample_bytes = self._estimate_sample_bytes(dataset)
        if sample_bytes is None:
            return batch_size
        device_type = self._device_type()
        if device_type == "cuda":
            available = self._available_cuda_memory()
            if available is None:
                return batch_size
            if profile == "throughput":
                budget_ratio = 0.8
                overhead = 8.0
            elif profile == "memory_saving":
                budget_ratio = 0.5
                overhead = 14.0
            else:
                budget_ratio = 0.6
                overhead = 12.0
        else:
            available = self._available_system_memory()
            if available is None:
                return batch_size
            if profile == "throughput":
                budget_ratio = 0.4
                overhead = 1.8
            elif profile == "memory_saving":
                budget_ratio = 0.25
                overhead = 3.0
            else:
                budget_ratio = 0.3
                overhead = 2.6
        budget = int(available * budget_ratio)
        per_sample = int(sample_bytes * overhead)
        if per_sample <= 0:
            return batch_size
        max_batch = max(1, int(budget // per_sample))
        if max_batch < batch_size:
            _log(
                f">>> Memory cap: batch_size {batch_size} -> {max_batch} "
                f"(per_sample~{sample_bytes}B, budget~{budget // (1024**2)}MB)"
            )
        return min(batch_size, max_batch)

    def _resolve_num_workers(self, max_workers: int, profile: Optional[str] = None) -> int:
        """Determine number of DataLoader workers."""
        if os.name == 'nt':
            return 0
        override = getattr(self, "dataloader_workers", None)
        if override is None:
            override = os.environ.get("BAYESOPT_DATALOADER_WORKERS")
        if override is not None:
            try:
                return max(0, int(override))
            except (TypeError, ValueError):
                pass
        if getattr(self, "is_ddp_enabled", False):
            return 0
        profile = profile or self._resolve_resource_profile()
        if profile == "memory_saving":
            return 0
        worker_cap = min(int(max_workers), os.cpu_count() or 1)
        if self._device_type() == "mps":
            worker_cap = min(worker_cap, 2)
        return worker_cap

    def _build_dataloader(self,
                          dataset,
                          N: int,
                          base_bs_gpu: tuple,
                          base_bs_cpu: tuple,
                          min_bs: int = 64,
                          target_effective_cuda: int = 1024,
                          target_effective_cpu: int = 512,
                          large_threshold: int = 200_000,
                          mid_threshold: int = 50_000):
        """Build DataLoader with adaptive batch size and worker configuration.

        Returns:
            Tuple of (dataloader, accum_steps)
        """
        profile = self._resolve_resource_profile()
        self._log_resource_summary_once(profile)
        data_size = int(N) if N is not None else len(dataset)
        gpu_large, gpu_mid, gpu_small = base_bs_gpu
        cpu_mid, cpu_small = base_bs_cpu

        device_type = self._device_type()
        is_ddp = bool(getattr(self, "is_ddp_enabled", False))
        if device_type == 'cuda':
            # Only scale batch size by GPU count when DDP is enabled.
            # In single-process (non-DDP) mode, large multi-GPU nodes can
            # still OOM on RAM/VRAM if we scale by device_count.
            device_count = 1
            if is_ddp:
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    min_bs = min_bs * device_count
                    _log(
                        f">>> Multi-GPU detected: {device_count} devices. Adjusted min_bs to {min_bs}.")

            if data_size > large_threshold:
                base_bs = gpu_large * device_count
            elif data_size > mid_threshold:
                base_bs = gpu_mid * device_count
            else:
                base_bs = gpu_small * device_count
        else:
            base_bs = cpu_mid if data_size > mid_threshold else cpu_small

        batch_size = compute_batch_size(
            data_size=data_size,
            learning_rate=self.learning_rate,
            batch_num=self.batch_num,
            minimum=min_bs
        )
        batch_size = min(batch_size, base_bs, data_size)
        batch_size = self._cap_batch_size_by_memory(
            dataset, batch_size, profile)

        target_effective_bs = target_effective_cuda if device_type == 'cuda' else target_effective_cpu
        world_size = 1
        if is_ddp:
            world_size = getattr(self, "world_size", None)
            world_size = max(1, world_size or DistributedUtils.world_size())
            target_effective_bs = max(1, target_effective_bs // world_size)
        samples_per_rank = math.ceil(
            data_size / max(1, world_size)) if world_size > 1 else data_size
        steps_per_epoch = max(
            1, math.ceil(samples_per_rank / max(1, batch_size)))
        desired_accum = max(1, target_effective_bs // max(1, batch_size))
        accum_steps = max(1, min(desired_accum, steps_per_epoch))

        workers = self._resolve_num_workers(8, profile=profile)
        prefetch_factor = None
        if workers > 0:
            prefetch_factor = 4 if profile == "throughput" else 2
        persistent = workers > 0 and profile != "memory_saving"
        _log(
            f">>> DataLoader config: Batch Size={batch_size}, Accum Steps={accum_steps}, "
            f"Workers={workers}, Prefetch={prefetch_factor or 'off'}, Profile={profile}")
        sampler = None
        use_distributed_sampler = bool(
            dist.is_initialized() and getattr(self, "is_ddp_enabled", False)
        )
        if use_distributed_sampler:
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False
        else:
            shuffle = True

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=workers,
            pin_memory=(device_type == 'cuda'),
            persistent_workers=persistent,
            **({"prefetch_factor": prefetch_factor} if prefetch_factor is not None else {}),
        )
        self.dataloader_sampler = sampler
        return dataloader, accum_steps

    def _build_val_dataloader(self, dataset, train_dataloader, accum_steps):
        """Build validation DataLoader."""
        profile = self._resolve_resource_profile()
        val_bs = accum_steps * train_dataloader.batch_size
        val_workers = self._resolve_num_workers(4, profile=profile)
        prefetch_factor = None
        if val_workers > 0:
            prefetch_factor = 2
        return DataLoader(
            dataset,
            batch_size=val_bs,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=(self._device_type() == 'cuda'),
            persistent_workers=(val_workers > 0 and profile != "memory_saving"),
            **({"prefetch_factor": prefetch_factor} if prefetch_factor is not None else {}),
        )

    def _compute_losses(self, y_pred, y_true, apply_softplus: bool = False):
        """Compute per-sample losses based on task type."""
        task = getattr(self, "task_type", "regression")
        if task == 'classification':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            return loss_fn(y_pred, y_true).view(-1)
        loss_name = normalize_loss_name(
            getattr(self, "loss_name", None), task_type="regression"
        )
        if loss_name == "auto":
            model_name = getattr(self, "model_name", None) or getattr(self, "model_nme", "")
            loss_name = infer_loss_name_from_model_name(model_name)
        if apply_softplus:
            y_pred = F.softplus(y_pred)
        if loss_requires_positive(loss_name):
            y_pred = torch.clamp(y_pred, min=1e-6)
            power = resolve_tweedie_power(
                loss_name, default=float(getattr(self, "tw_power", 1.5) or 1.5)
            )
            if power is None:
                power = float(getattr(self, "tw_power", 1.5) or 1.5)
            return tweedie_loss(y_pred, y_true, p=power).view(-1)
        if loss_name == "mse":
            return (y_pred - y_true).pow(2).view(-1)
        if loss_name == "mae":
            return (y_pred - y_true).abs().view(-1)
        raise ValueError(f"Unsupported loss_name '{loss_name}' for regression.")

    def _compute_weighted_loss(self, y_pred, y_true, weights, apply_softplus: bool = False):
        """Compute weighted loss."""
        losses = self._compute_losses(
            y_pred, y_true, apply_softplus=apply_softplus)
        weighted_loss = (losses * weights.view(-1)).sum() / \
            torch.clamp(weights.sum(), min=EPS)
        return weighted_loss

    def _early_stop_update(self, val_loss, best_loss, best_state, patience_counter, model,
                           ignore_keys: Optional[List[str]] = None):
        """Update early stopping state."""
        if val_loss < best_loss:
            ignore_keys = ignore_keys or []
            base_module = model.module if hasattr(model, "module") else model
            state_dict = {
                k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                for k, v in base_module.state_dict().items()
                if not any(k.startswith(ignore_key) for ignore_key in ignore_keys)
            }
            return val_loss, state_dict, 0, False
        patience_counter += 1
        should_stop = best_state is not None and patience_counter >= getattr(
            self, "patience", 0)
        return best_loss, best_state, patience_counter, should_stop

    def _train_model(self,
                     model,
                     dataloader,
                     accum_steps,
                     optimizer,
                     scaler,
                     forward_fn,
                     val_forward_fn=None,
                     apply_softplus: bool = False,
                     clip_fn=None,
                     trial: Optional[optuna.trial.Trial] = None,
                     loss_curve_path: Optional[str] = None):
        """Generic training loop with AMP, DDP, and early stopping support.

        Returns:
            Tuple of (best_state_dict, history)
        """
        device_type = self._device_type()
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        stop_training = False
        train_history: List[float] = []
        val_history: List[float] = []

        is_ddp_model = isinstance(model, DDP)
        use_collectives = dist.is_initialized() and is_ddp_model

        for epoch in range(1, getattr(self, "epochs", 1) + 1):
            epoch_start_ts = time.time()
            val_weighted_loss = None
            if hasattr(self, 'dataloader_sampler') and self.dataloader_sampler is not None:
                self.dataloader_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad()

            epoch_loss_sum = None
            epoch_weight_sum = None
            for step, batch in enumerate(dataloader):
                is_update_step = ((step + 1) % accum_steps == 0) or \
                    ((step + 1) == len(dataloader))
                sync_cm = model.no_sync if (
                    is_ddp_model and not is_update_step) else nullcontext

                with sync_cm():
                    with autocast(enabled=(device_type == 'cuda')):
                        y_pred, y_true, w = forward_fn(batch)
                        weighted_loss = self._compute_weighted_loss(
                            y_pred, y_true, w, apply_softplus=apply_softplus)
                        loss_for_backward = weighted_loss / accum_steps

                    batch_weight = torch.clamp(
                        w.detach().sum(), min=EPS).to(dtype=torch.float32)
                    loss_val = weighted_loss.detach().to(dtype=torch.float32)
                    if epoch_loss_sum is None:
                        epoch_loss_sum = torch.zeros(
                            (), device=batch_weight.device, dtype=torch.float32)
                        epoch_weight_sum = torch.zeros(
                            (), device=batch_weight.device, dtype=torch.float32)
                    epoch_loss_sum = epoch_loss_sum + loss_val * batch_weight
                    epoch_weight_sum = epoch_weight_sum + batch_weight
                    scaler.scale(loss_for_backward).backward()

                if is_update_step:
                    if clip_fn is not None:
                        clip_fn()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            if epoch_loss_sum is None or epoch_weight_sum is None:
                train_epoch_loss = 0.0
            else:
                train_epoch_loss = (
                    epoch_loss_sum / torch.clamp(epoch_weight_sum, min=EPS)
                ).item()
            train_history.append(float(train_epoch_loss))

            if val_forward_fn is not None:
                should_compute_val = (not dist.is_initialized()
                                      or DistributedUtils.is_main_process())
                val_device = self._resolve_device()
                loss_tensor_device = val_device if device_type == 'cuda' else torch.device(
                    "cpu")
                val_loss_tensor = torch.zeros(1, device=loss_tensor_device)

                if should_compute_val:
                    model.eval()
                    with torch.no_grad(), autocast(enabled=(device_type == 'cuda')):
                        val_result = val_forward_fn()
                        if isinstance(val_result, tuple) and len(val_result) == 3:
                            y_val_pred, y_val_true, w_val = val_result
                            val_weighted_loss = self._compute_weighted_loss(
                                y_val_pred, y_val_true, w_val, apply_softplus=apply_softplus)
                        else:
                            val_weighted_loss = val_result
                    val_loss_tensor[0] = float(val_weighted_loss)

                if use_collectives:
                    dist.broadcast(val_loss_tensor, src=0)
                val_weighted_loss = float(val_loss_tensor.item())

                val_history.append(val_weighted_loss)

                best_loss, best_state, patience_counter, stop_training = self._early_stop_update(
                    val_weighted_loss, best_loss, best_state, patience_counter, model)

                prune_flag = False
                is_main_rank = DistributedUtils.is_main_process()
                if trial is not None and is_main_rank:
                    trial.report(val_weighted_loss, epoch)
                    prune_flag = trial.should_prune()

                if use_collectives:
                    prune_device = self._resolve_device()
                    prune_tensor = torch.zeros(1, device=prune_device)
                    if is_main_rank:
                        prune_tensor.fill_(1 if prune_flag else 0)
                    dist.broadcast(prune_tensor, src=0)
                    prune_flag = bool(prune_tensor.item())

                if prune_flag:
                    raise optuna.TrialPruned()

                if stop_training:
                    break

            should_log_epoch = (not dist.is_initialized()
                                or DistributedUtils.is_main_process())
            if should_log_epoch:
                elapsed = int(time.time() - epoch_start_ts)
                if val_weighted_loss is None:
                    _log(
                        f"[Training] Epoch {epoch}/{getattr(self, 'epochs', 1)} "
                        f"train_loss={float(train_epoch_loss):.6f} elapsed={elapsed}s",
                        flush=True,
                    )
                else:
                    _log(
                        f"[Training] Epoch {epoch}/{getattr(self, 'epochs', 1)} "
                        f"train_loss={float(train_epoch_loss):.6f} "
                        f"val_loss={float(val_weighted_loss):.6f} elapsed={elapsed}s",
                        flush=True,
                    )

            if epoch % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        history = {"train": train_history, "val": val_history}
        self._plot_loss_curve(history, loss_curve_path)
        return best_state, history

    def _plot_loss_curve(self, history: Dict[str, List[float]], save_path: Optional[str]) -> None:
        """Plot training and validation loss curves."""
        if not save_path:
            return
        if dist.is_initialized() and not DistributedUtils.is_main_process():
            return
        train_hist = history.get("train", []) if history else []
        val_hist = history.get("val", []) if history else []
        if not train_hist and not val_hist:
            return
        if plot_loss_curve_common is not None:
            plot_loss_curve_common(
                history=history,
                title="Loss vs. Epoch",
                save_path=save_path,
                show=False,
            )
        else:
            if plt is None:
                _plot_skip("loss curve")
                return
            ensure_parent_dir(save_path)
            epochs = range(1, max(len(train_hist), len(val_hist)) + 1)
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            if train_hist:
                ax.plot(range(1, len(train_hist) + 1), train_hist,
                        label='Train Loss', color='tab:blue')
            if val_hist:
                ax.plot(range(1, len(val_hist) + 1), val_hist,
                        label='Validation Loss', color='tab:orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Weighted Loss')
            ax.set_title('Loss vs. Epoch')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        _log(f"[Training] Loss curve saved to {save_path}")
