from __future__ import annotations

import csv
import ctypes
import copy
import gc
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:  # matplotlib is optional; avoid hard import failures in headless/minimal envs
    import matplotlib
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - optional dependency
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    _MPL_IMPORT_ERROR = exc
import numpy as np
import optuna
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler

# Optional: unify plotting with shared plotting package
try:
    from ...plotting import curves as plot_curves_common
    from ...plotting.diagnostics import plot_loss_curve as plot_loss_curve_common
except Exception:  # pragma: no cover
    try:
        from ins_pricing.plotting import curves as plot_curves_common
        from ins_pricing.plotting.diagnostics import plot_loss_curve as plot_loss_curve_common
    except Exception:  # pragma: no cover
        plot_curves_common = None
        plot_loss_curve_common = None
# Limit CUDA allocator split size to reduce fragmentation and OOM risk.
# Override via PYTORCH_CUDA_ALLOC_CONF if needed.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")


# Constants and utility helpers
# =============================================================================
torch.backends.cudnn.benchmark = True
EPS = 1e-8


def _plot_skip(label: str) -> None:
    if _MPL_IMPORT_ERROR is not None:
        print(f"[Plot] Skip {label}: matplotlib unavailable ({_MPL_IMPORT_ERROR}).", flush=True)
    else:
        print(f"[Plot] Skip {label}: matplotlib unavailable.", flush=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IOUtils:
    # File and path utilities.

    @staticmethod
    def csv_to_dict(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [
                dict(filter(lambda item: item[0] != '', row.items()))
                for row in reader
            ]

    @staticmethod
    def _sanitize_params_dict(params: Dict[str, Any]) -> Dict[str, Any]:
        # Filter index-like columns such as "Unnamed: 0" from pandas I/O.
        return {
            k: v
            for k, v in (params or {}).items()
            if k and not str(k).startswith("Unnamed")
        }

    @staticmethod
    def load_params_file(path: str) -> Dict[str, Any]:
        """Load parameter dict from JSON/CSV/TSV files.

        - JSON: accept dict or {"best_params": {...}} wrapper
        - CSV/TSV: read the first row as params
        """
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"params file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(file_path.read_text(
                encoding="utf-8", errors="replace"))
            if isinstance(payload, dict) and "best_params" in payload:
                payload = payload.get("best_params") or {}
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid JSON params file (expect dict): {file_path}")
            return IOUtils._sanitize_params_dict(dict(payload))
        if suffix in (".csv", ".tsv"):
            df = pd.read_csv(file_path, sep="\t" if suffix == ".tsv" else ",")
            if df.empty:
                raise ValueError(f"Empty params file: {file_path}")
            params = df.iloc[0].to_dict()
            return IOUtils._sanitize_params_dict(params)
        raise ValueError(
            f"Unsupported params file type '{suffix}': {file_path}")

    @staticmethod
    def ensure_parent_dir(file_path: str) -> None:
        # Create parent directories when missing.
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)


class TrainingUtils:
    # Small helpers used during training.

    @staticmethod
    def compute_batch_size(data_size: int, learning_rate: float, batch_num: int, minimum: int) -> int:
        estimated = int((learning_rate / 1e-4) ** 0.5 *
                        (data_size / max(batch_num, 1)))
        return max(1, min(data_size, max(minimum, estimated)))

    @staticmethod
    def tweedie_loss(pred, target, p=1.5, eps=1e-6, max_clip=1e6):
        # Clamp predictions to positive values for stability.
        pred_clamped = torch.clamp(pred, min=eps)
        if p == 1:
            term1 = target * torch.log(target / pred_clamped + eps)  # Poisson
            term2 = -target + pred_clamped
            term3 = 0
        elif p == 0:
            term1 = 0.5 * torch.pow(target - pred_clamped, 2)  # Gaussian
            term2 = 0
            term3 = 0
        elif p == 2:
            term1 = torch.log(pred_clamped / target + eps)  # Gamma
            term2 = -target / pred_clamped + 1
            term3 = 0
        else:
            term1 = torch.pow(target, 2 - p) / ((1 - p) * (2 - p))
            term2 = target * torch.pow(pred_clamped, 1 - p) / (1 - p)
            term3 = torch.pow(pred_clamped, 2 - p) / (2 - p)
        return torch.nan_to_num(  # Tweedie negative log-likelihood (constant omitted)
            2 * (term1 - term2 + term3),
            nan=eps,
            posinf=max_clip,
            neginf=-max_clip
        )

    @staticmethod
    def free_cuda() -> None:
        print(">>> Moving all models to CPU...")
        for obj in gc.get_objects():
            try:
                if hasattr(obj, "to") and callable(obj.to):
                    obj.to("cpu")
            except Exception:
                pass

        print(">>> Releasing tensor/optimizer/DataLoader references...")
        gc.collect()

        print(">>> Clearing CUDA cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(">>> CUDA memory released.")
        else:
            print(">>> CUDA not available; cleanup skipped.")


class DistributedUtils:
    _cached_state: Optional[tuple] = None

    @staticmethod
    def setup_ddp():
        """Initialize the DDP process group for distributed training."""
        if dist.is_initialized():
            if DistributedUtils._cached_state is None:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                DistributedUtils._cached_state = (
                    True,
                    local_rank,
                    rank,
                    world_size,
                )
            return DistributedUtils._cached_state

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])

            if os.name == "nt" and torch.cuda.is_available() and world_size > 1:
                print(
                    ">>> DDP Setup Disabled: Windows CUDA DDP is not supported. "
                    "Falling back to single process."
                )
                return False, 0, 0, 1

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            timeout_seconds = int(os.environ.get(
                "BAYESOPT_DDP_TIMEOUT_SECONDS", "1800"))
            timeout = timedelta(seconds=max(1, timeout_seconds))
            backend = "gloo"
            if torch.cuda.is_available() and os.name != "nt":
                try:
                    if getattr(dist, "is_nccl_available", lambda: False)():
                        backend = "nccl"
                except Exception:
                    backend = "gloo"

            dist.init_process_group(
                backend=backend, init_method="env://", timeout=timeout)
            print(
                f">>> DDP Initialized ({backend}, timeout={timeout_seconds}s): "
                f"Rank {rank}/{world_size}, Local Rank {local_rank}"
            )
            DistributedUtils._cached_state = (
                True,
                local_rank,
                rank,
                world_size,
            )
            return DistributedUtils._cached_state
        else:
            print(
                f">>> DDP Setup Failed: RANK or WORLD_SIZE not found in env. Keys found: {list(os.environ.keys())}"
            )
            print(
                ">>> Hint: launch with torchrun --nproc_per_node=<N> <script.py>"
            )
        return False, 0, 0, 1

    @staticmethod
    def cleanup_ddp():
        """Destroy the DDP process group and clear cached state."""
        if dist.is_initialized():
            dist.destroy_process_group()
        DistributedUtils._cached_state = None

    @staticmethod
    def is_main_process():
        return not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def world_size() -> int:
        return dist.get_world_size() if dist.is_initialized() else 1


class PlotUtils:
    # Plot helpers shared across models.

    @staticmethod
    def split_data(data: pd.DataFrame, col_nme: str, wgt_nme: str, n_bins: int = 10) -> pd.DataFrame:
        data_sorted = data.sort_values(by=col_nme, ascending=True).copy()
        data_sorted['cum_weight'] = data_sorted[wgt_nme].cumsum()
        w_sum = data_sorted[wgt_nme].sum()
        if w_sum <= EPS:
            data_sorted['bins'] = 0
        else:
            data_sorted['bins'] = np.floor(
                data_sorted['cum_weight'] * float(n_bins) / w_sum
            )
            data_sorted.loc[(data_sorted['bins'] == n_bins),
                            'bins'] = n_bins - 1
        return data_sorted.groupby(['bins'], observed=True).sum(numeric_only=True)

    @staticmethod
    def plot_lift_ax(ax, plot_data, title, pred_label='Predicted', act_label='Actual', weight_label='Earned Exposure'):
        ax.plot(plot_data.index, plot_data['act_v'],
                label=act_label, color='red')
        ax.plot(plot_data.index, plot_data['exp_v'],
                label=pred_label, color='blue')
        ax.set_title(title, fontsize=8)
        ax.set_xticks(plot_data.index)
        ax.set_xticklabels(plot_data.index, rotation=90, fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(loc='upper left', fontsize=5, frameon=False)
        ax.margins(0.05)
        ax2 = ax.twinx()
        ax2.bar(plot_data.index, plot_data['weight'],
                alpha=0.5, color='seagreen',
                label=weight_label)
        ax2.tick_params(axis='y', labelsize=6)
        ax2.legend(loc='upper right', fontsize=5, frameon=False)

    @staticmethod
    def plot_dlift_ax(ax, plot_data, title, label1, label2, act_label='Actual', weight_label='Earned Exposure'):
        ax.plot(plot_data.index, plot_data['act_v'],
                label=act_label, color='red')
        ax.plot(plot_data.index, plot_data['exp_v1'],
                label=label1, color='blue')
        ax.plot(plot_data.index, plot_data['exp_v2'],
                label=label2, color='black')
        ax.set_title(title, fontsize=8)
        ax.set_xticks(plot_data.index)
        ax.set_xticklabels(plot_data.index, rotation=90, fontsize=6)
        ax.set_xlabel(f'{label1} / {label2}', fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(loc='upper left', fontsize=5, frameon=False)
        ax.margins(0.1)
        ax2 = ax.twinx()
        ax2.bar(plot_data.index, plot_data['weight'],
                alpha=0.5, color='seagreen',
                label=weight_label)
        ax2.tick_params(axis='y', labelsize=6)
        ax2.legend(loc='upper right', fontsize=5, frameon=False)

    @staticmethod
    def plot_lift_list(pred_model, w_pred_list, w_act_list,
                       weight_list, tgt_nme, n_bins: int = 10,
                       fig_nme: str = 'Lift Chart'):
        if plot_curves_common is not None:
            save_path = os.path.join(
                os.getcwd(), 'plot', f'05_{tgt_nme}_{fig_nme}.png')
            plot_curves_common.plot_lift_curve(
                pred_model,
                w_act_list,
                weight_list,
                n_bins=n_bins,
                title=f'Lift Chart of {tgt_nme}',
                pred_label='Predicted',
                act_label='Actual',
                weight_label='Earned Exposure',
                pred_weighted=False,
                actual_weighted=True,
                save_path=save_path,
                show=False,
            )
            return
        if plt is None:
            _plot_skip("lift plot")
            return
        lift_data = pd.DataFrame({
            'pred': pred_model,
            'w_pred': w_pred_list,
            'act': w_act_list,
            'weight': weight_list
        })
        plot_data = PlotUtils.split_data(lift_data, 'pred', 'weight', n_bins)
        plot_data['exp_v'] = plot_data['w_pred'] / plot_data['weight']
        plot_data['act_v'] = plot_data['act'] / plot_data['weight']
        plot_data.reset_index(inplace=True)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        PlotUtils.plot_lift_ax(ax, plot_data, f'Lift Chart of {tgt_nme}')
        plt.subplots_adjust(wspace=0.3)

        save_path = os.path.join(
            os.getcwd(), 'plot', f'05_{tgt_nme}_{fig_nme}.png')
        IOUtils.ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_dlift_list(pred_model_1, pred_model_2,
                        model_nme_1, model_nme_2,
                        tgt_nme,
                        w_list, w_act_list, n_bins: int = 10,
                        fig_nme: str = 'Double Lift Chart'):
        if plot_curves_common is not None:
            save_path = os.path.join(
                os.getcwd(), 'plot', f'06_{tgt_nme}_{fig_nme}.png')
            plot_curves_common.plot_double_lift_curve(
                pred_model_1,
                pred_model_2,
                w_act_list,
                w_list,
                n_bins=n_bins,
                title=f'Double Lift Chart of {tgt_nme}',
                label1=model_nme_1,
                label2=model_nme_2,
                pred1_weighted=False,
                pred2_weighted=False,
                actual_weighted=True,
                save_path=save_path,
                show=False,
            )
            return
        if plt is None:
            _plot_skip("double lift plot")
            return
        lift_data = pd.DataFrame({
            'pred1': pred_model_1,
            'pred2': pred_model_2,
            'act': w_act_list,
            'weight': w_list
        })
        lift_data['diff_ly'] = lift_data['pred1'] / lift_data['pred2']
        lift_data['w_pred1'] = lift_data['pred1'] * lift_data['weight']
        lift_data['w_pred2'] = lift_data['pred2'] * lift_data['weight']
        plot_data = PlotUtils.split_data(
            lift_data, 'diff_ly', 'weight', n_bins)
        plot_data['exp_v1'] = plot_data['w_pred1'] / plot_data['act']
        plot_data['exp_v2'] = plot_data['w_pred2'] / plot_data['act']
        plot_data['act_v'] = plot_data['act']/plot_data['act']
        plot_data.reset_index(inplace=True)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        PlotUtils.plot_dlift_ax(
            ax, plot_data, f'Double Lift Chart of {tgt_nme}', model_nme_1, model_nme_2)
        plt.subplots_adjust(bottom=0.25, top=0.95, right=0.8)

        save_path = os.path.join(
            os.getcwd(), 'plot', f'06_{tgt_nme}_{fig_nme}.png')
        IOUtils.ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


def infer_factor_and_cate_list(train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               resp_nme: str,
                               weight_nme: str,
                               binary_resp_nme: Optional[str] = None,
                               factor_nmes: Optional[List[str]] = None,
                               cate_list: Optional[List[str]] = None,
                               infer_categorical_max_unique: int = 50,
                               infer_categorical_max_ratio: float = 0.05) -> Tuple[List[str], List[str]]:
    """Infer factor_nmes/cate_list when feature names are not provided.

    Rules:
      - factor_nmes: start from shared train/test columns, exclude target/weight/(optional binary target).
      - cate_list: object/category/bool plus low-cardinality integer columns.
      - Always intersect with shared train/test columns to avoid mismatches.
    """
    excluded = {resp_nme, weight_nme}
    if binary_resp_nme:
        excluded.add(binary_resp_nme)

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    if factor_nmes is None:
        factors = [c for c in common_cols if c not in excluded]
    else:
        factors = [
            c for c in factor_nmes if c in common_cols and c not in excluded]

    if cate_list is not None:
        cats = [c for c in cate_list if c in factors]
        return factors, cats

    n_rows = max(1, len(train_df))
    cats: List[str] = []
    for col in factors:
        s = train_df[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            cats.append(col)
            continue
        if pd.api.types.is_integer_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if nunique <= infer_categorical_max_unique or (nunique / n_rows) <= infer_categorical_max_ratio:
                cats.append(col)
    return factors, cats


# Backward-compatible functional wrappers
def csv_to_dict(file_path: str) -> List[Dict[str, Any]]:
    return IOUtils.csv_to_dict(file_path)


def ensure_parent_dir(file_path: str) -> None:
    IOUtils.ensure_parent_dir(file_path)


def compute_batch_size(data_size: int, learning_rate: float, batch_num: int, minimum: int) -> int:
    return TrainingUtils.compute_batch_size(data_size, learning_rate, batch_num, minimum)


# Tweedie deviance loss for PyTorch.
# Reference: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances
def tweedie_loss(pred, target, p=1.5, eps=1e-6, max_clip=1e6):
    return TrainingUtils.tweedie_loss(pred, target, p=p, eps=eps, max_clip=max_clip)


# CUDA memory release helper.
def free_cuda():
    TrainingUtils.free_cuda()


class TorchTrainerMixin:
    # Shared helpers for Torch tabular trainers.

    def _device_type(self) -> str:
        return getattr(self, "device", torch.device("cpu")).type

    def _resolve_resource_profile(self) -> str:
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
        if getattr(self, "_resource_summary_logged", False):
            return
        if dist.is_initialized() and not DistributedUtils.is_main_process():
            return
        self._resource_summary_logged = True
        device = getattr(self, "device", torch.device("cpu"))
        device_type = self._device_type()
        cpu_count = os.cpu_count() or 1
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        ddp_enabled = bool(getattr(self, "is_ddp_enabled", False))
        data_parallel = bool(getattr(self, "use_data_parallel", False))
        print(
            f">>> Resource summary: device={device}, device_type={device_type}, "
            f"cpu_count={cpu_count}, cuda_count={cuda_count}, mps={mps_available}, "
            f"ddp={ddp_enabled}, data_parallel={data_parallel}, profile={profile}"
        )

    def _available_system_memory(self) -> Optional[int]:
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
        if not torch.cuda.is_available():
            return None
        try:
            free_mem, _total_mem = torch.cuda.mem_get_info()
        except Exception:
            return None
        return int(free_mem)

    def _estimate_sample_bytes(self, dataset) -> Optional[int]:
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
            print(
                f">>> Memory cap: batch_size {batch_size} -> {max_batch} "
                f"(per_sample~{sample_bytes}B, budget~{budget // (1024**2)}MB)"
            )
        return min(batch_size, max_batch)

    def _resolve_num_workers(self, max_workers: int, profile: Optional[str] = None) -> int:
        if os.name == 'nt':
            return 0
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
        profile = self._resolve_resource_profile()
        self._log_resource_summary_once(profile)
        batch_size = TrainingUtils.compute_batch_size(
            data_size=len(dataset),
            learning_rate=self.learning_rate,
            batch_num=self.batch_num,
            minimum=min_bs
        )
        gpu_large, gpu_mid, gpu_small = base_bs_gpu
        cpu_mid, cpu_small = base_bs_cpu

        if self._device_type() == 'cuda':
            device_count = torch.cuda.device_count()
            if getattr(self, "is_ddp_enabled", False):
                device_count = 1
            # In multi-GPU, increase min batch size so each GPU gets enough data.
            if device_count > 1:
                min_bs = min_bs * device_count
                print(
                    f">>> Multi-GPU detected: {device_count} devices. Adjusted min_bs to {min_bs}.")

            if N > large_threshold:
                base_bs = gpu_large * device_count
            elif N > mid_threshold:
                base_bs = gpu_mid * device_count
            else:
                base_bs = gpu_small * device_count
        else:
            base_bs = cpu_mid if N > mid_threshold else cpu_small

        # Recompute batch_size to respect the adjusted min_bs.
        batch_size = TrainingUtils.compute_batch_size(
            data_size=len(dataset),
            learning_rate=self.learning_rate,
            batch_num=self.batch_num,
            minimum=min_bs
        )
        batch_size = min(batch_size, base_bs, N)
        batch_size = self._cap_batch_size_by_memory(
            dataset, batch_size, profile)

        target_effective_bs = target_effective_cuda if self._device_type(
        ) == 'cuda' else target_effective_cpu
        if getattr(self, "is_ddp_enabled", False):
            world_size = max(1, DistributedUtils.world_size())
            target_effective_bs = max(1, target_effective_bs // world_size)

        world_size = getattr(self, "world_size", 1) if getattr(
            self, "is_ddp_enabled", False) else 1
        samples_per_rank = math.ceil(
            N / max(1, world_size)) if world_size > 1 else N
        steps_per_epoch = max(
            1, math.ceil(samples_per_rank / max(1, batch_size)))
        # Limit gradient accumulation to avoid scaling beyond actual batches.
        desired_accum = max(1, target_effective_bs // max(1, batch_size))
        accum_steps = max(1, min(desired_accum, steps_per_epoch))

        # Linux (posix) uses fork; Windows (nt) uses spawn with higher overhead.
        workers = self._resolve_num_workers(8, profile=profile)
        prefetch_factor = None
        if workers > 0:
            prefetch_factor = 4 if profile == "throughput" else 2
        persistent = workers > 0 and profile != "memory_saving"
        print(
            f">>> DataLoader config: Batch Size={batch_size}, Accum Steps={accum_steps}, "
            f"Workers={workers}, Prefetch={prefetch_factor or 'off'}, Profile={profile}")
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False  # DistributedSampler handles shuffling.
        else:
            shuffle = True

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=workers,
            pin_memory=(self._device_type() == 'cuda'),
            persistent_workers=persistent,
            **({"prefetch_factor": prefetch_factor} if prefetch_factor is not None else {}),
        )
        return dataloader, accum_steps

    def _build_val_dataloader(self, dataset, train_dataloader, accum_steps):
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
        task = getattr(self, "task_type", "regression")
        if task == 'classification':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            return loss_fn(y_pred, y_true).view(-1)
        if apply_softplus:
            y_pred = F.softplus(y_pred)
        y_pred = torch.clamp(y_pred, min=1e-6)
        power = getattr(self, "tw_power", 1.5)
        return tweedie_loss(y_pred, y_true, p=power).view(-1)

    def _compute_weighted_loss(self, y_pred, y_true, weights, apply_softplus: bool = False):
        losses = self._compute_losses(
            y_pred, y_true, apply_softplus=apply_softplus)
        weighted_loss = (losses * weights.view(-1)).sum() / \
            torch.clamp(weights.sum(), min=EPS)
        return weighted_loss

    def _early_stop_update(self, val_loss, best_loss, best_state, patience_counter, model,
                           ignore_keys: Optional[List[str]] = None):
        if val_loss < best_loss:
            ignore_keys = ignore_keys or []
            # Unwrap DDP module to avoid module. prefix in state_dict keys
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
        device_type = self._device_type()
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        stop_training = False
        train_history: List[float] = []
        val_history: List[float] = []

        is_ddp_model = isinstance(model, DDP)

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
                val_device = getattr(self, "device", torch.device("cpu"))
                if not isinstance(val_device, torch.device):
                    val_device = torch.device(val_device)
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

                if dist.is_initialized():
                    dist.broadcast(val_loss_tensor, src=0)
                val_weighted_loss = float(val_loss_tensor.item())

                val_history.append(val_weighted_loss)

                best_loss, best_state, patience_counter, stop_training = self._early_stop_update(
                    val_weighted_loss, best_loss, best_state, patience_counter, model)

                prune_flag = False
                is_main_rank = DistributedUtils.is_main_process()
                # Only main process reports to Optuna to avoid duplicate reports
                if trial is not None and is_main_rank:
                    trial.report(val_weighted_loss, epoch)
                    prune_flag = trial.should_prune()

                if dist.is_initialized():
                    prune_device = getattr(self, "device", torch.device("cpu"))
                    if not isinstance(prune_device, torch.device):
                        prune_device = torch.device(prune_device)
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
                    print(
                        f"[Training] Epoch {epoch}/{getattr(self, 'epochs', 1)} "
                        f"train_loss={float(train_epoch_loss):.6f} elapsed={elapsed}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[Training] Epoch {epoch}/{getattr(self, 'epochs', 1)} "
                        f"train_loss={float(train_epoch_loss):.6f} "
                        f"val_loss={float(val_weighted_loss):.6f} elapsed={elapsed}s",
                        flush=True,
                    )

            # Periodic memory cleanup to prevent accumulation (every 10 epochs)
            if epoch % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        history = {"train": train_history, "val": val_history}
        self._plot_loss_curve(history, loss_curve_path)
        return best_state, history

    def _plot_loss_curve(self, history: Dict[str, List[float]], save_path: Optional[str]) -> None:
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
        print(f"[Training] Loss curve saved to {save_path}")


# =============================================================================
# Plotting helpers
# =============================================================================

def split_data(data, col_nme, wgt_nme, n_bins=10):
    return PlotUtils.split_data(data, col_nme, wgt_nme, n_bins)

# Lift curve plotting wrapper


def plot_lift_list(pred_model, w_pred_list, w_act_list,
                   weight_list, tgt_nme, n_bins=10,
                   fig_nme='Lift Chart'):
    return PlotUtils.plot_lift_list(pred_model, w_pred_list, w_act_list,
                                    weight_list, tgt_nme, n_bins, fig_nme)

# Double lift curve plotting wrapper


def plot_dlift_list(pred_model_1, pred_model_2,
                    model_nme_1, model_nme_2,
                    tgt_nme,
                    w_list, w_act_list, n_bins=10,
                    fig_nme='Double Lift Chart'):
    return PlotUtils.plot_dlift_list(pred_model_1, pred_model_2,
                                     model_nme_1, model_nme_2,
                                     tgt_nme, w_list, w_act_list,
                                     n_bins, fig_nme)


# =============================================================================
# Logging System
# =============================================================================
import logging
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_package_logger() -> logging.Logger:
    """Get or create the package-level logger with consistent formatting."""
    logger = logging.getLogger("ins_pricing")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s][%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Default to INFO, can be changed via environment variable
        level = os.environ.get("INS_PRICING_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def get_logger(name: str = "ins_pricing") -> logging.Logger:
    """Get a logger with the given name, inheriting package-level settings.

    Args:
        name: Logger name, typically module name like 'ins_pricing.trainer'

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger("ins_pricing.trainer.ft")
        >>> logger.info("Training started")
    """
    # Ensure package logger is initialized
    _get_package_logger()
    return logging.getLogger(name)


# =============================================================================
# Metric Computation Factory
# =============================================================================
from sklearn.metrics import log_loss, mean_tweedie_deviance


class MetricFactory:
    """Factory for computing evaluation metrics consistently across all trainers.

    This class centralizes metric computation logic that was previously duplicated
    across FTTrainer, ResNetTrainer, GNNTrainer, XGBTrainer, and GLMTrainer.

    Example:
        >>> factory = MetricFactory(task_type='regression', tweedie_power=1.5)
        >>> score = factory.compute(y_true, y_pred, sample_weight)
    """

    def __init__(
        self,
        task_type: str = "regression",
        tweedie_power: float = 1.5,
        clip_min: float = 1e-8,
        clip_max: float = 1 - 1e-8,
    ):
        """Initialize the metric factory.

        Args:
            task_type: Either 'regression' or 'classification'
            tweedie_power: Power parameter for Tweedie deviance (1.0-2.0)
            clip_min: Minimum value for clipping predictions
            clip_max: Maximum value for clipping predictions (for classification)
        """
        self.task_type = task_type
        self.tweedie_power = tweedie_power
        self.clip_min = clip_min
        self.clip_max = clip_max

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the appropriate metric based on task type.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights

        Returns:
            Computed metric value (lower is better)
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        if self.task_type == "classification":
            y_pred_clipped = np.clip(y_pred, self.clip_min, self.clip_max)
            return float(log_loss(y_true, y_pred_clipped, sample_weight=sample_weight))

        # Regression: use Tweedie deviance
        y_pred_safe = np.maximum(y_pred, self.clip_min)
        return float(mean_tweedie_deviance(
            y_true,
            y_pred_safe,
            sample_weight=sample_weight,
            power=self.tweedie_power,
        ))

    def update_power(self, power: float) -> None:
        """Update the Tweedie power parameter.

        Args:
            power: New power value (1.0-2.0)
        """
        self.tweedie_power = power


# =============================================================================
# GPU Memory Manager
# =============================================================================
from contextlib import contextmanager


class GPUMemoryManager:
    """Context manager for GPU memory management and cleanup.

    This class consolidates GPU memory cleanup logic that was previously
    scattered across multiple trainer files.

    Example:
        >>> with GPUMemoryManager.cleanup_context():
        ...     model.train()
        ...     # Memory cleaned up after exiting context

        >>> # Or use directly:
        >>> GPUMemoryManager.clean()
    """

    _logger = get_logger("ins_pricing.gpu")

    @classmethod
    def clean(cls, verbose: bool = False) -> None:
        """Clean up GPU memory.

        Args:
            verbose: If True, print cleanup details
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if verbose:
                cls._logger.debug("CUDA cache cleared and synchronized")

        # Optional: Force IPC collect for multi-process scenarios
        if os.environ.get("BAYESOPT_CUDA_IPC_COLLECT", "0") == "1":
            if torch.cuda.is_available():
                try:
                    torch.cuda.ipc_collect()
                    if verbose:
                        cls._logger.debug("CUDA IPC collect performed")
                except Exception:
                    pass

    @classmethod
    @contextmanager
    def cleanup_context(cls, verbose: bool = False):
        """Context manager that cleans GPU memory on exit.

        Args:
            verbose: If True, print cleanup details

        Yields:
            None
        """
        try:
            yield
        finally:
            cls.clean(verbose=verbose)

    @classmethod
    def move_model_to_cpu(cls, model: nn.Module) -> nn.Module:
        """Move a model to CPU and clean GPU memory.

        Args:
            model: PyTorch model to move

        Returns:
            Model on CPU
        """
        if model is not None:
            model.to("cpu")
        cls.clean()
        return model

    @classmethod
    def get_memory_info(cls) -> Dict[str, int]:
        """Get current GPU memory usage information.

        Returns:
            Dictionary with memory info (allocated, reserved, free)
        """
        if not torch.cuda.is_available():
            return {"available": False}

        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            free, total = torch.cuda.mem_get_info()
            return {
                "available": True,
                "allocated_mb": allocated // (1024 * 1024),
                "reserved_mb": reserved // (1024 * 1024),
                "free_mb": free // (1024 * 1024),
                "total_mb": total // (1024 * 1024),
            }
        except Exception:
            return {"available": False}


# =============================================================================
# Device Manager
# =============================================================================

class DeviceManager:
    """Unified device management for model and tensor placement.

    This class consolidates device detection and model movement logic
    that was previously duplicated across trainer_base.py and predict.py.

    Example:
        >>> device = DeviceManager.get_best_device()
        >>> model = DeviceManager.move_to_device(model)
    """

    _logger = get_logger("ins_pricing.device")
    _cached_device: Optional[torch.device] = None

    @classmethod
    def get_best_device(cls, prefer_cuda: bool = True) -> torch.device:
        """Get the best available device.

        Args:
            prefer_cuda: If True, prefer CUDA over MPS

        Returns:
            Best available torch.device
        """
        if cls._cached_device is not None:
            return cls._cached_device

        if prefer_cuda and torch.cuda.is_available():
            cls._cached_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls._cached_device = torch.device("mps")
        else:
            cls._cached_device = torch.device("cpu")

        cls._logger.debug(f"Selected device: {cls._cached_device}")
        return cls._cached_device

    @classmethod
    def move_to_device(cls, model_obj, device: Optional[torch.device] = None) -> None:
        """Move a model object to the specified device.

        Handles sklearn-style wrappers that have .ft, .resnet, or .gnn attributes.

        Args:
            model_obj: Model object to move (may be sklearn wrapper)
            device: Target device (defaults to best available)
        """
        if model_obj is None:
            return

        device = device or cls.get_best_device()

        # Update device attribute if present
        if hasattr(model_obj, "device"):
            model_obj.device = device

        # Move the main model
        if hasattr(model_obj, "to"):
            model_obj.to(device)

        # Move nested submodules (sklearn wrappers)
        for attr_name in ("ft", "resnet", "gnn"):
            submodule = getattr(model_obj, attr_name, None)
            if submodule is not None and hasattr(submodule, "to"):
                submodule.to(device)

    @classmethod
    def unwrap_module(cls, module: nn.Module) -> nn.Module:
        """Unwrap DDP or DataParallel wrapper to get the base module.

        Args:
            module: Potentially wrapped PyTorch module

        Returns:
            Unwrapped base module
        """
        if isinstance(module, (DDP, nn.DataParallel)):
            return module.module
        return module

    @classmethod
    def reset_cache(cls) -> None:
        """Reset cached device selection."""
        cls._cached_device = None


# =============================================================================
# Cross-Validation Strategy Resolver
# =============================================================================
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, StratifiedKFold


class CVStrategyResolver:
    """Resolver for cross-validation splitting strategies.

    This class consolidates CV strategy resolution logic that was previously
    duplicated across trainer_base.py and trainer_ft.py.

    Supported strategies:
        - 'random': Standard KFold
        - 'stratified': StratifiedKFold (for classification)
        - 'group': GroupKFold (requires group column)
        - 'time': TimeSeriesSplit (requires time column)

    Example:
        >>> resolver = CVStrategyResolver(
        ...     strategy='group',
        ...     n_splits=5,
        ...     group_col='policy_id',
        ...     data=train_df,
        ... )
        >>> splitter, groups = resolver.get_splitter()
        >>> for train_idx, val_idx in splitter.split(X, y, groups):
        ...     pass
    """

    VALID_STRATEGIES = {"random", "stratified", "group", "grouped", "time", "timeseries", "temporal"}

    def __init__(
        self,
        strategy: str = "random",
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        time_ascending: bool = True,
        data: Optional[pd.DataFrame] = None,
    ):
        """Initialize the CV strategy resolver.

        Args:
            strategy: CV strategy name
            n_splits: Number of CV folds
            shuffle: Whether to shuffle for random/stratified
            random_state: Random seed for reproducibility
            group_col: Column name for group-based splitting
            time_col: Column name for time-based splitting
            time_ascending: Sort order for time-based splitting
            data: DataFrame containing group/time columns
        """
        self.strategy = strategy.strip().lower()
        self.n_splits = max(2, int(n_splits))
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_col = group_col
        self.time_col = time_col
        self.time_ascending = time_ascending
        self.data = data

        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options: {sorted(self.VALID_STRATEGIES)}"
            )

    def get_splitter(self) -> Tuple[Any, Optional[pd.Series]]:
        """Get the appropriate splitter and groups.

        Returns:
            Tuple of (splitter, groups) where groups may be None

        Raises:
            ValueError: If required columns are missing
        """
        if self.strategy in {"group", "grouped"}:
            return self._get_group_splitter()
        elif self.strategy in {"time", "timeseries", "temporal"}:
            return self._get_time_splitter()
        elif self.strategy == "stratified":
            return self._get_stratified_splitter()
        else:
            return self._get_random_splitter()

    def _get_random_splitter(self) -> Tuple[KFold, None]:
        """Get a random KFold splitter."""
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return splitter, None

    def _get_stratified_splitter(self) -> Tuple[StratifiedKFold, None]:
        """Get a stratified KFold splitter."""
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return splitter, None

    def _get_group_splitter(self) -> Tuple[GroupKFold, pd.Series]:
        """Get a group-based KFold splitter."""
        if not self.group_col:
            raise ValueError("group_col is required for group strategy")
        if self.data is None:
            raise ValueError("data DataFrame is required for group strategy")
        if self.group_col not in self.data.columns:
            raise KeyError(f"group_col '{self.group_col}' not found in data")

        groups = self.data[self.group_col]
        splitter = GroupKFold(n_splits=self.n_splits)
        return splitter, groups

    def _get_time_splitter(self) -> Tuple[Any, None]:
        """Get a time-series splitter."""
        if not self.time_col:
            raise ValueError("time_col is required for time strategy")
        if self.data is None:
            raise ValueError("data DataFrame is required for time strategy")
        if self.time_col not in self.data.columns:
            raise KeyError(f"time_col '{self.time_col}' not found in data")

        splitter = TimeSeriesSplit(n_splits=self.n_splits)

        # Create an ordered wrapper that sorts by time column
        order_index = self.data[self.time_col].sort_values(
            ascending=self.time_ascending
        ).index
        order = self.data.index.get_indexer(order_index)

        return _OrderedSplitter(splitter, order), None


class _OrderedSplitter:
    """Wrapper for splitters that need to respect a specific ordering."""

    def __init__(self, base_splitter, order: np.ndarray):
        self.base_splitter = base_splitter
        self.order = order

    def split(self, X, y=None, groups=None):
        """Split with ordering applied."""
        n = len(X)
        X_ordered = np.arange(n)[self.order]
        for train_idx, val_idx in self.base_splitter.split(X_ordered):
            yield self.order[train_idx], self.order[val_idx]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.base_splitter.get_n_splits()
