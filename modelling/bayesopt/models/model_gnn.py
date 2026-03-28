from __future__ import annotations

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from ins_pricing.modelling.bayesopt.utils.distributed_utils import DistributedUtils
from ins_pricing.modelling.bayesopt.utils.torch_runtime import (
    create_autocast_context,
    create_grad_scaler,
    resolve_training_device,
    setup_ddp_if_requested,
    wrap_model_for_parallel,
)
from ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin import TorchTrainerMixin
from ins_pricing.utils import DeviceManager, EPS, get_logger, log_print
from ins_pricing.utils.io import IOUtils
from ins_pricing.utils.losses import (
    normalize_distribution_name,
    resolve_effective_loss_name,
    resolve_tweedie_power,
)
from ins_pricing.utils.torch_compat import torch_load

try:
    from torch_geometric.nn import knn_graph
    from torch_geometric.utils import add_self_loops, to_undirected
    _PYG_AVAILABLE = True
except Exception:
    knn_graph = None  # type: ignore
    add_self_loops = None  # type: ignore
    to_undirected = None  # type: ignore
    _PYG_AVAILABLE = False

try:
    import pynndescent
    _PYNN_AVAILABLE = True
except Exception:
    pynndescent = None  # type: ignore
    _PYNN_AVAILABLE = False

_GNN_MPS_WARNED = False

_logger = get_logger("ins_pricing.modelling.bayesopt.models.model_gnn")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


# =============================================================================
# Simplified GNN implementation.
# =============================================================================

def _adj_mm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Matrix multiply that supports sparse or dense adjacency."""
    if adj.is_sparse:
        return torch.sparse.mm(adj, x)
    return adj.matmul(x)


class SimpleGraphLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Message passing with normalized sparse adjacency: A_hat * X * W.
        h = _adj_mm(adj, x)
        h = self.linear(h)
        h = self.activation(h)
        return self.dropout(h)


class SimpleGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, task_type: str = 'regression'):
        super().__init__()
        layers = []
        dim_in = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(SimpleGraphLayer(
                dim_in, hidden_dim, dropout=dropout))
            dim_in = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        if task_type == 'classification':
            self.output_act = nn.Identity()
        else:
            self.output_act = nn.Softplus()
        self.task_type = task_type
        # Keep adjacency as a buffer for DataParallel copies.
        self.register_buffer("adj_buffer", torch.empty(0))

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        adj_used = adj if adj is not None else getattr(
            self, "adj_buffer", None)
        if adj_used is None or adj_used.numel() == 0:
            raise RuntimeError("Adjacency is not set for GNN forward.")
        h = x
        for layer in self.layers:
            h = layer(h, adj_used)
        h = _adj_mm(adj_used, h)
        out = self.output(h)
        return self.output_act(out)


class GraphNeuralNetSklearn(TorchTrainerMixin, nn.Module):
    SUPPORTS_MULTI_PROCESS_DDP = False

    def __init__(self, model_nme: str, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, k_neighbors: int = 10, dropout: float = 0.1,
                 learning_rate: float = 1e-3, epochs: int = 100, patience: int = 10,
                 task_type: str = 'regression', tweedie_power: float = 1.5,
                 weight_decay: float = 0.0,
                 use_data_parallel: bool = False, use_ddp: bool = False,
                 use_gpu: bool = True,
                 use_approx_knn: bool = True, approx_knn_threshold: int = 50000,
                 graph_cache_path: Optional[str] = None,
                 max_gpu_knn_nodes: Optional[int] = None,
                 knn_gpu_mem_ratio: float = 0.9,
                 knn_gpu_mem_overhead: float = 2.0,
                 knn_cpu_jobs: Optional[int] = -1,
                 max_fit_rows: Optional[int] = None,
                 max_predict_rows: Optional[int] = None,
                 predict_chunk_rows: Optional[int] = None,
                 loss_name: Optional[str] = None,
                 distribution: Optional[str] = None) -> None:
        super().__init__()
        self.model_nme = model_nme
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.k_neighbors = max(1, k_neighbors)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.task_type = task_type
        self.use_gpu = bool(use_gpu)
        self.use_approx_knn = use_approx_knn
        self.approx_knn_threshold = approx_knn_threshold
        self.graph_cache_path = Path(
            graph_cache_path) if graph_cache_path else None
        self.graph_hash_sample_rows = max(
            32,
            int(os.environ.get("BAYESOPT_GNN_GRAPH_HASH_SAMPLE_ROWS", "2048")),
        )
        self.graph_cache_lock_timeout = max(
            1.0,
            float(os.environ.get("BAYESOPT_GNN_GRAPH_LOCK_TIMEOUT_SECONDS", "30")),
        )
        self.graph_cache_lock_poll = max(
            0.01,
            float(os.environ.get("BAYESOPT_GNN_GRAPH_LOCK_POLL_SECONDS", "0.1")),
        )
        self.max_gpu_knn_nodes = max_gpu_knn_nodes
        self.knn_gpu_mem_ratio = max(0.0, min(1.0, knn_gpu_mem_ratio))
        self.knn_gpu_mem_overhead = max(1.0, knn_gpu_mem_overhead)
        self.knn_cpu_jobs = knn_cpu_jobs
        self.max_fit_rows = self._coerce_positive_int(max_fit_rows)
        self.max_predict_rows = self._coerce_positive_int(max_predict_rows)
        self.predict_chunk_rows = self._coerce_positive_int(predict_chunk_rows)
        self.mps_dense_max_nodes = int(
            os.environ.get("BAYESOPT_GNN_MPS_DENSE_MAX_NODES", "5000")
        )
        self._knn_warning_emitted = False
        self._mps_fallback_triggered = False
        self._adj_cache_meta: Optional[Dict[str, Any]] = None
        self._adj_cache_key: Optional[Tuple[Any, ...]] = None
        self._adj_cache_tensor: Optional[torch.Tensor] = None

        resolved_distribution = normalize_distribution_name(distribution, self.task_type)
        self.distribution = None if resolved_distribution == "auto" else resolved_distribution
        if self.task_type == 'classification':
            self.loss_name = "logloss"
            self.tw_power = None
        else:
            self.loss_name = resolve_effective_loss_name(
                loss_name,
                task_type=self.task_type,
                model_name=self.model_nme,
                distribution=self.distribution,
            )
            if self.loss_name == "tweedie":
                self.tw_power = float(tweedie_power) if tweedie_power is not None else 1.5
            else:
                self.tw_power = resolve_tweedie_power(self.loss_name, default=1.5)

        self.ddp_enabled = False
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.data_parallel_enabled = False
        self._ddp_disabled = False

        requested_ddp = bool(use_ddp and self.use_gpu)
        if use_ddp and not self.use_gpu:
            _log(
                "[GNN] DDP requested with use_gpu=false; forcing CPU single-process mode.",
                flush=True,
            )
        if requested_ddp:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1:
                _log(
                    "[GNN] DDP training is not supported; falling back to single process.",
                    flush=True,
                )
                self._ddp_disabled = True
                requested_ddp = False

        # DDP only works with CUDA; fall back to single process if init fails.
        if requested_ddp and torch.cuda.is_available():
            ddp_ok, local_rank, _, _ = setup_ddp_if_requested(True)
            if ddp_ok:
                self.ddp_enabled = True
                self.local_rank = local_rank
                self.device = resolve_training_device(
                    is_ddp_enabled=True, local_rank=self.local_rank, use_gpu=self.use_gpu
                )
        if not self.ddp_enabled:
            if self.use_gpu and torch.cuda.is_available():
                if self._ddp_disabled:
                    self.device = torch.device(f'cuda:{self.local_rank}')
                else:
                    self.device = resolve_training_device(
                        is_ddp_enabled=False, local_rank=self.local_rank, use_gpu=self.use_gpu
                    )
            elif self.use_gpu and DeviceManager.is_mps_available():
                self.device = torch.device('mps')
                global _GNN_MPS_WARNED
                if not _GNN_MPS_WARNED:
                    _log(
                        "[GNN] Using MPS backend; will fall back to CPU on unsupported ops.",
                        flush=True,
                    )
                    _GNN_MPS_WARNED = True
            else:
                self.device = torch.device('cpu')
        self.use_pyg_knn = self.device.type == 'cuda' and _PYG_AVAILABLE

        self.gnn = SimpleGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            task_type=self.task_type
        ).to(self.device)

        # DataParallel copies the full graph to each GPU and splits features; good for medium graphs.
        self.gnn, self.data_parallel_enabled, self.device = wrap_model_for_parallel(
            self.gnn,
            device=self.device,
            use_data_parallel=bool(use_data_parallel and self.use_gpu),
            use_ddp_requested=bool(requested_ddp),
            is_ddp_enabled=self.ddp_enabled,
            local_rank=self.local_rank,
            ddp_find_unused_parameters=False,
            fallback_log=_log,
        )

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

    @staticmethod
    def _coerce_positive_int(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return None
        if resolved < 1:
            return None
        return resolved

    @staticmethod
    def _slice_rows(X, start: int, end: int):
        if hasattr(X, "iloc"):
            return X.iloc[start:end]
        return X[start:end]

    @staticmethod
    def _take_rows(data, indices):
        if data is None:
            return None
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        arr = np.asarray(data)
        return arr[indices]

    @staticmethod
    def _reset_index_if_possible(data):
        if data is None:
            return None
        if hasattr(data, "reset_index"):
            return data.reset_index(drop=True)
        return data

    def _maybe_cap_fit_rows(self, X, y, w, *, label: str):
        limit = self.max_fit_rows
        if limit is None:
            return X, y, w
        n_rows = len(X)
        if n_rows <= limit:
            return X, y, w
        rng = np.random.default_rng(13)
        sample_idx = np.sort(rng.choice(n_rows, size=int(limit), replace=False))
        X_out = self._reset_index_if_possible(self._take_rows(X, sample_idx))
        y_out = self._reset_index_if_possible(self._take_rows(y, sample_idx))
        w_out = self._reset_index_if_possible(self._take_rows(w, sample_idx))
        _log(
            f"[GNN] {label} rows reduced from {n_rows} to {int(limit)} "
            f"via uniform subsampling (gnn_max_fit_rows).",
            flush=True,
        )
        return X_out, y_out, w_out

    def _resolve_predict_chunk_rows(self, n_rows: int) -> Optional[int]:
        chunk_rows = self._coerce_positive_int(self.predict_chunk_rows)
        if chunk_rows is None:
            return None
        return min(int(chunk_rows), int(n_rows))

    def _unwrap_gnn(self) -> nn.Module:
        if isinstance(self.gnn, (DDP, nn.DataParallel)):
            return self.gnn.module
        return self.gnn

    def _set_adj_buffer(self, adj: torch.Tensor) -> None:
        base = self._unwrap_gnn()
        if hasattr(base, "adj_buffer"):
            base.adj_buffer = adj
        else:
            base.register_buffer("adj_buffer", adj)

    @staticmethod
    def _is_mps_unsupported_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        if "mps" not in msg:
            return False
        if any(token in msg for token in ("not supported", "not implemented", "does not support", "unimplemented", "out of memory")):
            return True
        return "sparse" in msg

    def _fallback_to_cpu(self, reason: str) -> None:
        if self.device.type != "mps" or self._mps_fallback_triggered:
            return
        self._mps_fallback_triggered = True
        _log(f"[GNN] MPS op unsupported ({reason}); falling back to CPU.", flush=True)
        self.device = torch.device("cpu")
        self.use_pyg_knn = False
        self.data_parallel_enabled = False
        self.ddp_enabled = False
        base = self._unwrap_gnn()
        try:
            base = base.to(self.device)
        except Exception:
            pass
        self.gnn = base
        self.invalidate_graph_cache()

    def _run_with_mps_fallback(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device.type == "mps" and self._is_mps_unsupported_error(exc):
                self._fallback_to_cpu(str(exc))
                return fn(*args, **kwargs)
            raise

    def _sample_for_hash(self, X_df: pd.DataFrame) -> pd.DataFrame:
        n_rows = int(X_df.shape[0])
        sample_rows = min(n_rows, int(self.graph_hash_sample_rows))
        if sample_rows >= n_rows:
            return X_df
        sample_idx = np.linspace(0, n_rows - 1, num=sample_rows, dtype=np.int64)
        return X_df.iloc[sample_idx]

    def _graph_cache_meta(self, X_df: pd.DataFrame) -> Dict[str, Any]:
        sample_df = self._sample_for_hash(X_df)
        row_hash = pd.util.hash_pandas_object(sample_df, index=False).values
        idx_hash = pd.util.hash_pandas_object(sample_df.index, index=False).values
        col_sig = ",".join(
            f"{col}:{dtype}" for col, dtype in zip(X_df.columns, X_df.dtypes)
        )
        hasher = hashlib.sha256()
        hasher.update(np.asarray(X_df.shape, dtype=np.int64).tobytes())
        hasher.update(row_hash.tobytes())
        hasher.update(idx_hash.tobytes())
        hasher.update(col_sig.encode("utf-8", errors="ignore"))
        knn_config = {
            "k_neighbors": int(self.k_neighbors),
            "use_approx_knn": bool(self.use_approx_knn),
            "approx_knn_threshold": int(self.approx_knn_threshold),
            "use_pyg_knn": bool(self.use_pyg_knn),
            "pynndescent_available": bool(_PYNN_AVAILABLE),
            "max_gpu_knn_nodes": (
                None if self.max_gpu_knn_nodes is None else int(self.max_gpu_knn_nodes)
            ),
            "knn_gpu_mem_ratio": float(self.knn_gpu_mem_ratio),
            "knn_gpu_mem_overhead": float(self.knn_gpu_mem_overhead),
        }
        adj_format = "dense" if self.device.type == "mps" else "sparse"
        return {
            "n_samples": int(X_df.shape[0]),
            "n_features": int(X_df.shape[1]),
            "hash_sample_rows": int(sample_df.shape[0]),
            "hash": hasher.hexdigest(),
            "knn_config": knn_config,
            "adj_format": adj_format,
            "device_type": self.device.type,
        }

    def _graph_cache_key(self, X_df: pd.DataFrame) -> Tuple[Any, ...]:
        return (
            id(X_df),
            id(getattr(X_df, "_mgr", None)),
            id(X_df.index),
            X_df.shape,
            tuple(map(str, X_df.columns)),
            X_df.attrs.get("graph_cache_key"),
        )

    def invalidate_graph_cache(self) -> None:
        self._adj_cache_meta = None
        self._adj_cache_key = None
        self._adj_cache_tensor = None

    @contextmanager
    def _graph_cache_write_lock(self):
        if not self.graph_cache_path:
            yield
            return
        lock_path = Path(str(self.graph_cache_path) + ".lock")
        fd: Optional[int] = None
        start = time.time()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                if (time.time() - start) >= self.graph_cache_lock_timeout:
                    _log(
                        f"[GNN] Timeout waiting for graph cache lock: {lock_path}. "
                        "Continue without lock."
                    )
                    yield
                    return
                time.sleep(self.graph_cache_lock_poll)
            except Exception:
                yield
                return
        try:
            yield
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _save_graph_cache(self, adj_norm: torch.Tensor, meta_expected: Dict[str, Any]) -> None:
        if not self.graph_cache_path:
            return
        tmp_path = Path(
            f"{self.graph_cache_path}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        try:
            IOUtils.ensure_parent_dir(str(self.graph_cache_path))
            with self._graph_cache_write_lock():
                torch.save({"adj": adj_norm.cpu(), "meta": meta_expected}, tmp_path)
                os.replace(tmp_path, self.graph_cache_path)
        except Exception as exc:
            _log(
                f"[GNN] Failed to cache graph to {self.graph_cache_path}: {exc}")
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def _load_cached_adj(self,
                         X_df: pd.DataFrame,
                         meta_expected: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        if self.graph_cache_path and self.graph_cache_path.exists():
            if meta_expected is None:
                meta_expected = self._graph_cache_meta(X_df)
            try:
                payload = torch_load(
                    self.graph_cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception as exc:
                _log(
                    f"[GNN] Failed to load cached graph from {self.graph_cache_path}: {exc}")
                return None
            if isinstance(payload, dict) and "adj" in payload:
                meta_cached = payload.get("meta")
                if meta_cached == meta_expected:
                    adj = payload["adj"]
                    if self.device.type == "mps" and getattr(adj, "is_sparse", False):
                        _log(
                            f"[GNN] Cached sparse graph incompatible with MPS; rebuilding: {self.graph_cache_path}"
                        )
                        return None
                    return adj.to(self.device)
                _log(
                    f"[GNN] Cached graph metadata mismatch; rebuilding: {self.graph_cache_path}")
                return None
            if isinstance(payload, torch.Tensor):
                _log(
                    f"[GNN] Cached graph missing metadata; rebuilding: {self.graph_cache_path}")
                return None
            _log(
                f"[GNN] Invalid cached graph format; rebuilding: {self.graph_cache_path}")
        return None

    def _build_edge_index_cpu(self, X_np: np.ndarray) -> torch.Tensor:
        n_samples = X_np.shape[0]
        k = min(self.k_neighbors, max(1, n_samples - 1))
        n_neighbors = min(k + 1, n_samples)
        use_approx = (self.use_approx_knn or n_samples >=
                      self.approx_knn_threshold) and _PYNN_AVAILABLE
        indices = None
        if use_approx:
            try:
                nn_index = pynndescent.NNDescent(
                    X_np,
                    n_neighbors=n_neighbors,
                    random_state=0
                )
                indices, _ = nn_index.neighbor_graph
            except Exception as exc:
                _log(
                    f"[GNN] Approximate kNN failed ({exc}); falling back to exact search.")
                use_approx = False

        if indices is None:
            nbrs = NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm="auto",
                n_jobs=self.knn_cpu_jobs,
            )
            nbrs.fit(X_np)
            _, indices = nbrs.kneighbors(X_np)

        indices = np.asarray(indices)
        rows = np.repeat(np.arange(n_samples), n_neighbors).astype(
            np.int64, copy=False)
        cols = indices.reshape(-1).astype(np.int64, copy=False)
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        rows_base = rows
        cols_base = cols
        self_loops = np.arange(n_samples, dtype=np.int64)
        rows = np.concatenate([rows_base, cols_base, self_loops])
        cols = np.concatenate([cols_base, rows_base, self_loops])

        edge_index_np = np.stack([rows, cols], axis=0)
        edge_index = torch.as_tensor(edge_index_np, device=self.device)
        return edge_index

    def _build_edge_index_gpu(self, X_tensor: torch.Tensor) -> torch.Tensor:
        if not self.use_pyg_knn or knn_graph is None or add_self_loops is None or to_undirected is None:
            # Defensive: check use_pyg_knn before calling.
            raise RuntimeError(
                "GPU graph builder requested but PyG is unavailable.")

        n_samples = X_tensor.size(0)
        k = min(self.k_neighbors, max(1, n_samples - 1))

        # knn_graph runs on GPU to avoid CPU graph construction bottlenecks.
        edge_index = knn_graph(
            X_tensor,
            k=k,
            loop=False
        )
        edge_index = to_undirected(edge_index, num_nodes=n_samples)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_samples)
        return edge_index

    def _log_knn_fallback(self, reason: str) -> None:
        if self._knn_warning_emitted:
            return
        if (not self.ddp_enabled) or self.local_rank == 0:
            _log(f"[GNN] Falling back to CPU kNN builder: {reason}")
        self._knn_warning_emitted = True

    def _should_use_gpu_knn(self, n_samples: int, X_tensor: torch.Tensor) -> bool:
        if not self.use_pyg_knn:
            return False

        reason = None
        if self.max_gpu_knn_nodes is not None and n_samples > self.max_gpu_knn_nodes:
            reason = f"node count {n_samples} exceeds max_gpu_knn_nodes={self.max_gpu_knn_nodes}"
        elif self.device.type == 'cuda' and torch.cuda.is_available():
            try:
                device_index = self.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                free_mem, total_mem = torch.cuda.mem_get_info(device_index)
                feature_bytes = X_tensor.element_size() * X_tensor.nelement()
                required = int(feature_bytes * self.knn_gpu_mem_overhead)
                budget = int(free_mem * self.knn_gpu_mem_ratio)
                if required > budget:
                    required_gb = required / (1024 ** 3)
                    budget_gb = budget / (1024 ** 3)
                    reason = (f"requires ~{required_gb:.2f} GiB temporary GPU memory "
                              f"but only {budget_gb:.2f} GiB free on cuda:{device_index}")
            except Exception:
                # On older versions or some environments, mem_get_info may be unavailable; default to trying GPU.
                reason = None

        if reason:
            self._log_knn_fallback(reason)
            return False
        return True

    def _normalized_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self.device.type == "mps":
            return self._normalized_adj_dense(edge_index, num_nodes)
        return self._normalized_adj_sparse(edge_index, num_nodes)

    def _normalized_adj_sparse(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        values = torch.ones(edge_index.shape[1], device=self.device)
        adj = torch.sparse_coo_tensor(
            edge_index.to(self.device), values, (num_nodes, num_nodes))
        adj = adj.coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        row, col = adj.indices()
        norm_values = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(
            adj.indices(), norm_values, size=adj.shape)
        return adj_norm

    def _normalized_adj_dense(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self.mps_dense_max_nodes <= 0 or num_nodes > self.mps_dense_max_nodes:
            raise RuntimeError(
                f"MPS dense adjacency not supported for {num_nodes} nodes; "
                f"max={self.mps_dense_max_nodes}. Falling back to CPU."
            )
        edge_index = edge_index.to(self.device)
        adj = torch.zeros((num_nodes, num_nodes), device=self.device, dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        adj = adj * deg_inv_sqrt.view(-1, 1)
        adj = adj * deg_inv_sqrt.view(1, -1)
        return adj

    def _tensorize_split(self, X, y, w, allow_none: bool = False):
        if X is None and allow_none:
            return None, None, None
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for GNN.")
        n_rows = len(X)
        if y is not None:
            self._validate_vector(y, "y", n_rows)
        if w is not None:
            self._validate_vector(w, "w", n_rows)
        X_np = X.to_numpy(dtype=np.float32, copy=False) if hasattr(
            X, "to_numpy") else np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(
            X_np, dtype=torch.float32, device=self.device)
        if y is None:
            y_tensor = None
        else:
            y_np = y.to_numpy(dtype=np.float32, copy=False) if hasattr(
                y, "to_numpy") else np.asarray(y, dtype=np.float32)
            y_tensor = torch.as_tensor(
                y_np, dtype=torch.float32, device=self.device).view(-1, 1)
        if w is None:
            w_tensor = torch.ones(
                (len(X), 1), dtype=torch.float32, device=self.device)
        else:
            w_np = w.to_numpy(dtype=np.float32, copy=False) if hasattr(
                w, "to_numpy") else np.asarray(w, dtype=np.float32)
            w_tensor = torch.as_tensor(
                w_np, dtype=torch.float32, device=self.device).view(-1, 1)
        return X_tensor, y_tensor, w_tensor

    def _build_graph_from_df(self, X_df: pd.DataFrame, X_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not isinstance(X_df, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for graph building.")
        meta_expected = None
        cache_key = None
        if self.graph_cache_path:
            meta_expected = self._graph_cache_meta(X_df)
            if self._adj_cache_meta == meta_expected and self._adj_cache_tensor is not None:
                cached = self._adj_cache_tensor
                if cached.device != self.device:
                    if self.device.type == "mps" and getattr(cached, "is_sparse", False):
                        self._adj_cache_tensor = None
                    else:
                        cached = cached.to(self.device)
                        self._adj_cache_tensor = cached
                if self._adj_cache_tensor is not None:
                    return self._adj_cache_tensor
        else:
            cache_key = self._graph_cache_key(X_df)
            if self._adj_cache_key == cache_key and self._adj_cache_tensor is not None:
                cached = self._adj_cache_tensor
                if cached.device != self.device:
                    if self.device.type == "mps" and getattr(cached, "is_sparse", False):
                        self._adj_cache_tensor = None
                    else:
                        cached = cached.to(self.device)
                        self._adj_cache_tensor = cached
                if self._adj_cache_tensor is not None:
                    return self._adj_cache_tensor
        X_np = None
        if X_tensor is None:
            X_np = X_df.to_numpy(dtype=np.float32, copy=False)
            X_tensor = torch.as_tensor(
                X_np, dtype=torch.float32, device=self.device)
        if self.graph_cache_path:
            cached = self._load_cached_adj(X_df, meta_expected=meta_expected)
            if cached is not None:
                self._adj_cache_meta = meta_expected
                self._adj_cache_key = None
                self._adj_cache_tensor = cached
                return cached
        use_gpu_knn = self._should_use_gpu_knn(X_df.shape[0], X_tensor)
        if use_gpu_knn:
            edge_index = self._build_edge_index_gpu(X_tensor)
        else:
            if X_np is None:
                X_np = X_df.to_numpy(dtype=np.float32, copy=False)
            edge_index = self._build_edge_index_cpu(X_np)
        adj_norm = self._normalized_adj(edge_index, X_df.shape[0])
        if self.graph_cache_path:
            if meta_expected is None:
                meta_expected = self._graph_cache_meta(X_df)
            self._save_graph_cache(adj_norm, meta_expected)
            self._adj_cache_meta = meta_expected
            self._adj_cache_key = None
        else:
            self._adj_cache_meta = None
            self._adj_cache_key = cache_key
        self._adj_cache_tensor = adj_norm
        return adj_norm

    def fit(self, X_train, y_train, w_train=None,
            X_val=None, y_val=None, w_val=None,
            trial: Optional[optuna.trial.Trial] = None):
        return self._run_with_mps_fallback(
            self._fit_impl,
            X_train,
            y_train,
            w_train,
            X_val,
            y_val,
            w_val,
            trial,
        )

    def _fit_impl(self, X_train, y_train, w_train=None,
                  X_val=None, y_val=None, w_val=None,
                  trial: Optional[optuna.trial.Trial] = None):
        X_train, y_train, w_train = self._maybe_cap_fit_rows(
            X_train, y_train, w_train, label="training")
        X_train_tensor, y_train_tensor, w_train_tensor = self._tensorize_split(
            X_train, y_train, w_train, allow_none=False)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val, y_val, w_val = self._maybe_cap_fit_rows(
                X_val, y_val, w_val, label="validation")
            X_val_tensor, y_val_tensor, w_val_tensor = self._tensorize_split(
                X_val, y_val, w_val, allow_none=False)
        else:
            X_val_tensor = y_val_tensor = w_val_tensor = None

        adj_train = self._build_graph_from_df(X_train, X_train_tensor)
        adj_val = self._build_graph_from_df(
            X_val, X_val_tensor) if has_val else None
        # DataParallel needs adjacency cached on the model to avoid scatter.
        self._set_adj_buffer(adj_train)

        base_gnn = self._unwrap_gnn()
        optimizer = torch.optim.Adam(
            base_gnn.parameters(),
            lr=self.learning_rate,
            weight_decay=float(getattr(self, "weight_decay", 0.0)),
        )
        scaler = create_grad_scaler(self.device.type)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        best_epoch = None

        for epoch in range(1, self.epochs + 1):
            epoch_start_ts = time.time()
            self.gnn.train()
            optimizer.zero_grad()
            with create_autocast_context(self.device.type):
                if self.data_parallel_enabled:
                    y_pred = self.gnn(X_train_tensor)
                else:
                    y_pred = self.gnn(X_train_tensor, adj_train)
                loss = self._compute_weighted_loss(
                    y_pred, y_train_tensor, w_train_tensor, apply_softplus=False)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(self.gnn.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            val_loss = None
            if has_val:
                self.gnn.eval()
                if self.data_parallel_enabled and adj_val is not None:
                    self._set_adj_buffer(adj_val)
                with torch.no_grad(), create_autocast_context(self.device.type):
                    if self.data_parallel_enabled:
                        y_val_pred = self.gnn(X_val_tensor)
                    else:
                        y_val_pred = self.gnn(X_val_tensor, adj_val)
                    val_loss = self._compute_weighted_loss(
                        y_val_pred, y_val_tensor, w_val_tensor, apply_softplus=False)
                if self.data_parallel_enabled:
                    # Restore training adjacency.
                    self._set_adj_buffer(adj_train)

                is_best = val_loss is not None and val_loss < best_loss
                best_loss, best_state, patience_counter, stop_training = self._early_stop_update(
                    val_loss, best_loss, best_state, patience_counter, base_gnn,
                    ignore_keys=["adj_buffer"])
                if is_best:
                    best_epoch = epoch

                prune_now = False
                if trial is not None:
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        prune_now = True

                if dist.is_initialized():
                    flag = torch.tensor(
                        [1 if prune_now else 0],
                        device=self.device,
                        dtype=torch.int32,
                    )
                    dist.broadcast(flag, src=0)
                    prune_now = bool(flag.item())

                if prune_now:
                    raise optuna.TrialPruned()
                if stop_training:
                    break

            should_log = (not dist.is_initialized()
                          or DistributedUtils.is_main_process())
            if should_log:
                elapsed = int(time.time() - epoch_start_ts)
                if val_loss is None:
                    _log(
                        f"[GNN] Epoch {epoch}/{self.epochs} loss={float(loss):.6f} elapsed={elapsed}s",
                        flush=True,
                    )
                else:
                    _log(
                        f"[GNN] Epoch {epoch}/{self.epochs} loss={float(loss):.6f} "
                        f"val_loss={float(val_loss):.6f} elapsed={elapsed}s",
                        flush=True,
                    )

        if best_state is not None:
            base_gnn.load_state_dict(best_state, strict=False)
        self.best_epoch = int(best_epoch or self.epochs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._run_with_mps_fallback(self._predict_impl, X)

    def _predict_single(self, X: pd.DataFrame) -> np.ndarray:
        self.gnn.eval()
        X_tensor, _, _ = self._tensorize_split(
            X, None, None, allow_none=False)
        adj = self._build_graph_from_df(X, X_tensor)
        if self.data_parallel_enabled:
            self._set_adj_buffer(adj)
        inference_cm = getattr(torch, "inference_mode", torch.no_grad)
        with inference_cm():
            if self.data_parallel_enabled:
                y_pred = self.gnn(X_tensor).cpu().numpy()
            else:
                y_pred = self.gnn(X_tensor, adj).cpu().numpy()
        if self.task_type == 'classification':
            y_pred = 1 / (1 + np.exp(-y_pred))
        else:
            y_pred = np.clip(y_pred, 1e-6, None)
        return y_pred.ravel()

    def _predict_chunked(self, X: pd.DataFrame, chunk_rows: int) -> np.ndarray:
        n_rows = len(X)
        if n_rows <= chunk_rows:
            return self._predict_single(X)
        if not getattr(self, "_predict_chunk_warned", False):
            _log(
                "[GNN] Using chunked local-graph prediction to cap memory usage. "
                "Chunks are predicted independently.",
                flush=True,
            )
            self._predict_chunk_warned = True
        preds: List[np.ndarray] = []
        original_graph_cache = self.graph_cache_path
        try:
            self.graph_cache_path = None
            self.invalidate_graph_cache()
            for start in range(0, n_rows, chunk_rows):
                end = min(n_rows, start + chunk_rows)
                X_chunk = self._reset_index_if_possible(
                    self._slice_rows(X, start, end))
                preds.append(self._predict_single(X_chunk))
                self.invalidate_graph_cache()
        finally:
            self.graph_cache_path = original_graph_cache
        return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        n_rows = len(X)
        if n_rows == 0:
            return np.empty(0, dtype=np.float32)
        chunk_rows = self._resolve_predict_chunk_rows(n_rows)
        max_predict_rows = self._coerce_positive_int(self.max_predict_rows)
        if max_predict_rows is not None and n_rows > max_predict_rows:
            if chunk_rows is None:
                raise MemoryError(
                    f"[GNN] predict rows={n_rows} exceeds gnn_max_predict_rows={max_predict_rows}. "
                    "Set gnn_predict_chunk_rows to enable chunked local-graph prediction."
                )
            if not getattr(self, "_predict_limit_warned", False):
                _log(
                    f"[GNN] predict rows={n_rows} exceeds gnn_max_predict_rows={max_predict_rows}; "
                    f"switching to chunked prediction with chunk_rows={chunk_rows}.",
                    flush=True,
                )
                self._predict_limit_warned = True
        if chunk_rows is not None and n_rows > chunk_rows:
            return self._predict_chunked(X, chunk_rows)
        return self._predict_single(X)

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        return self._run_with_mps_fallback(self._encode_impl, X)

    def _encode_single(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-sample node embeddings (hidden representations)."""
        base = self._unwrap_gnn()
        base.eval()
        X_tensor, _, _ = self._tensorize_split(X, None, None, allow_none=False)
        adj = self._build_graph_from_df(X, X_tensor)
        if self.data_parallel_enabled:
            self._set_adj_buffer(adj)
        inference_cm = getattr(torch, "inference_mode", torch.no_grad)
        with inference_cm():
            h = X_tensor
            layers = getattr(base, "layers", None)
            if layers is None:
                raise RuntimeError("GNN base module does not expose layers.")
            for layer in layers:
                h = layer(h, adj)
            h = _adj_mm(adj, h)
        return h.detach().cpu().numpy()

    def _encode_chunked(self, X: pd.DataFrame, chunk_rows: int) -> np.ndarray:
        n_rows = len(X)
        if n_rows <= chunk_rows:
            return self._encode_single(X)
        if not getattr(self, "_encode_chunk_warned", False):
            _log(
                "[GNN] Using chunked local-graph encoding to cap memory usage. "
                "Chunks are encoded independently.",
                flush=True,
            )
            self._encode_chunk_warned = True
        outputs: List[np.ndarray] = []
        original_graph_cache = self.graph_cache_path
        try:
            self.graph_cache_path = None
            self.invalidate_graph_cache()
            for start in range(0, n_rows, chunk_rows):
                end = min(n_rows, start + chunk_rows)
                X_chunk = self._reset_index_if_possible(
                    self._slice_rows(X, start, end))
                outputs.append(self._encode_single(X_chunk))
                self.invalidate_graph_cache()
        finally:
            self.graph_cache_path = original_graph_cache
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)

    def _encode_impl(self, X: pd.DataFrame) -> np.ndarray:
        n_rows = len(X)
        if n_rows == 0:
            return np.empty((0, 0), dtype=np.float32)
        chunk_rows = self._resolve_predict_chunk_rows(n_rows)
        max_predict_rows = self._coerce_positive_int(self.max_predict_rows)
        if max_predict_rows is not None and n_rows > max_predict_rows:
            if chunk_rows is None:
                raise MemoryError(
                    f"[GNN] encode rows={n_rows} exceeds gnn_max_predict_rows={max_predict_rows}. "
                    "Set gnn_predict_chunk_rows to enable chunked local-graph encoding."
                )
            if not getattr(self, "_encode_limit_warned", False):
                _log(
                    f"[GNN] encode rows={n_rows} exceeds gnn_max_predict_rows={max_predict_rows}; "
                    f"switching to chunked encoding with chunk_rows={chunk_rows}.",
                    flush=True,
                )
                self._encode_limit_warned = True
        if chunk_rows is not None and n_rows > chunk_rows:
            return self._encode_chunked(X, chunk_rows)
        return self._encode_single(X)

    def set_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in GNN model.")
        # Rebuild the backbone after structural parameter changes.
        self.gnn = SimpleGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            task_type=self.task_type
        ).to(self.device)
        return self
