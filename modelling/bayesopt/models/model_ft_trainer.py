from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset

from ins_pricing.modelling.bayesopt.utils.distributed_utils import DistributedUtils
from ins_pricing.modelling.bayesopt.utils.torch_runtime import (
    create_autocast_context,
    create_grad_scaler,
    resolve_training_device,
    setup_ddp_if_requested,
    wrap_model_for_parallel,
)
from ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin import TorchTrainerMixin
from ins_pricing.utils import EPS, get_logger, log_print
from ins_pricing.utils.losses import (
    normalize_distribution_name,
    resolve_effective_loss_name,
    resolve_tweedie_power,
)
from ins_pricing.modelling.bayesopt.models.model_ft_components import FTTransformerCore, MaskedTabularDataset, TabularDataset

_logger = get_logger("ins_pricing.modelling.bayesopt.models.model_ft_trainer")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


# --- Helper functions for reconstruction loss computation ---


def _compute_numeric_reconstruction_loss(
    num_pred: Optional[torch.Tensor],
    num_true: Optional[torch.Tensor],
    num_mask: Optional[torch.Tensor],
    loss_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute MSE loss for numeric feature reconstruction.

    Args:
        num_pred: Predicted numeric values (N, num_features)
        num_true: Ground truth numeric values (N, num_features)
        num_mask: Boolean mask indicating which values were masked (N, num_features)
        loss_weight: Weight to apply to the loss
        device: Target device for computation

    Returns:
        Weighted MSE loss for masked numeric features
    """
    if num_pred is None or num_true is None or num_mask is None:
        return torch.zeros((), device=device, dtype=torch.float32)

    num_mask = num_mask.to(dtype=torch.bool)
    if not num_mask.any():
        return torch.zeros((), device=device, dtype=torch.float32)

    diff = num_pred - num_true
    mse = diff * diff
    return float(loss_weight) * mse[num_mask].mean()


def _compute_categorical_reconstruction_loss(
    cat_logits: Optional[List[torch.Tensor]],
    cat_true: Optional[torch.Tensor],
    cat_mask: Optional[torch.Tensor],
    loss_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute cross-entropy loss for categorical feature reconstruction.

    Args:
        cat_logits: List of logits for each categorical feature
        cat_true: Ground truth categorical indices (N, num_cat_features)
        cat_mask: Boolean mask indicating which values were masked (N, num_cat_features)
        loss_weight: Weight to apply to the loss
        device: Target device for computation

    Returns:
        Weighted cross-entropy loss for masked categorical features
    """
    if not cat_logits or cat_true is None or cat_mask is None:
        return torch.zeros((), device=device, dtype=torch.float32)

    cat_mask = cat_mask.to(dtype=torch.bool)
    cat_losses: List[torch.Tensor] = []

    for j, logits in enumerate(cat_logits):
        mask_j = cat_mask[:, j]
        if not mask_j.any():
            continue
        targets = cat_true[:, j]
        cat_losses.append(
            F.cross_entropy(logits, targets, reduction='none')[mask_j].mean()
        )

    if not cat_losses:
        return torch.zeros((), device=device, dtype=torch.float32)

    return float(loss_weight) * torch.stack(cat_losses).mean()


def _compute_reconstruction_loss(
    num_pred: Optional[torch.Tensor],
    cat_logits: Optional[List[torch.Tensor]],
    num_true: Optional[torch.Tensor],
    num_mask: Optional[torch.Tensor],
    cat_true: Optional[torch.Tensor],
    cat_mask: Optional[torch.Tensor],
    num_loss_weight: float,
    cat_loss_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute combined reconstruction loss for masked tabular data.

    This combines numeric (MSE) and categorical (cross-entropy) reconstruction losses.

    Args:
        num_pred: Predicted numeric values
        cat_logits: List of logits for categorical features
        num_true: Ground truth numeric values
        num_mask: Mask for numeric features
        cat_true: Ground truth categorical indices
        cat_mask: Mask for categorical features
        num_loss_weight: Weight for numeric loss
        cat_loss_weight: Weight for categorical loss
        device: Target device for computation

    Returns:
        Combined weighted reconstruction loss
    """
    num_loss = _compute_numeric_reconstruction_loss(
        num_pred, num_true, num_mask, num_loss_weight, device
    )
    cat_loss = _compute_categorical_reconstruction_loss(
        cat_logits, cat_true, cat_mask, cat_loss_weight, device
    )
    return num_loss + cat_loss


def _codes_to_int64_with_unknown(codes: pd.Series, unknown_idx: int) -> np.ndarray:
    """Convert mapped category codes to int64 and route NA/unseen to unknown_idx."""
    numeric = pd.to_numeric(codes, errors="coerce")
    arr = np.asarray(numeric.to_numpy(copy=False), dtype=np.float64)
    missing = np.isnan(arr)
    if missing.any():
        arr = arr.copy()
        arr[missing] = float(unknown_idx)
    return arr.astype("int64", copy=False)


class _FTPreprocessorSnapshot:
    """Pickle-safe FT preprocessing snapshot for DataLoader workers."""

    def __init__(
        self,
        *,
        num_cols: list[str],
        cat_cols: list[str],
        num_mean: Optional[np.ndarray],
        num_std: Optional[np.ndarray],
        cat_categories: Dict[str, Any],
        cat_maps: Dict[str, Dict[Any, int]],
        cat_str_maps: Dict[str, Dict[str, int]],
        num_geo: int,
    ) -> None:
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.num_mean = None if num_mean is None else np.asarray(num_mean, dtype=np.float32)
        self.num_std = None if num_std is None else np.asarray(num_std, dtype=np.float32)
        self.cat_categories = dict(cat_categories)
        self.cat_maps = {k: dict(v) for k, v in (cat_maps or {}).items()}
        self.cat_str_maps = {k: dict(v) for k, v in (cat_str_maps or {}).items()}
        self.num_geo = int(num_geo)
        self.unknown_indices = {
            col: int(len(self.cat_categories.get(col, [])))
            for col in self.cat_cols
        }

    @classmethod
    def from_owner(cls, owner: "FTTransformerSklearn") -> "_FTPreprocessorSnapshot":
        return cls(
            num_cols=list(owner.num_cols),
            cat_cols=list(owner.cat_cols),
            num_mean=None if owner._num_mean is None else np.asarray(owner._num_mean, dtype=np.float32),
            num_std=None if owner._num_std is None else np.asarray(owner._num_std, dtype=np.float32),
            cat_categories=dict(owner.cat_categories),
            cat_maps=dict(owner.cat_maps),
            cat_str_maps=dict(owner.cat_str_maps),
            num_geo=int(owner.num_geo),
        )

    def encode_nums(self, X_num: Optional[pd.DataFrame]) -> np.ndarray:
        if not self.num_cols:
            n_rows = 0 if X_num is None else len(X_num)
            return np.zeros((n_rows, 0), dtype=np.float32)
        if X_num is None:
            raise ValueError("X_num is required when numeric columns exist.")
        num_np = X_num.to_numpy(dtype=np.float32, copy=False)
        if not num_np.flags["OWNDATA"]:
            num_np = num_np.copy()
        num_np = np.nan_to_num(num_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if self.num_mean is not None and self.num_std is not None and num_np.size:
            num_np = (num_np - self.num_mean) / self.num_std
        return np.asarray(num_np, dtype=np.float32)

    def encode_cats(self, X_cat: Optional[pd.DataFrame]) -> np.ndarray:
        if not self.cat_cols:
            n_rows = 0 if X_cat is None else len(X_cat)
            return np.zeros((n_rows, 0), dtype=np.int64)
        if X_cat is None:
            raise ValueError("X_cat is required when categorical columns exist.")

        n_rows = len(X_cat)
        out = np.empty((n_rows, len(self.cat_cols)), dtype=np.int64)
        for idx, col in enumerate(self.cat_cols):
            categories = self.cat_categories[col]
            mapping = self.cat_maps.get(col)
            if mapping is None:
                mapping = {cat: i for i, cat in enumerate(categories)}
                self.cat_maps[col] = mapping
            unknown_idx = self.unknown_indices[col]
            series = X_cat[col]
            codes = series.map(mapping)
            unmapped = series.notna() & codes.isna()
            if unmapped.any():
                try:
                    series_cast = series.astype(categories.dtype)
                except Exception:
                    series_cast = None
                if series_cast is not None:
                    codes = series_cast.map(mapping)
                    unmapped = series_cast.notna() & codes.isna()
            if unmapped.any():
                str_map = self.cat_str_maps.get(col)
                if str_map is None:
                    str_map = {str(cat): i for i, cat in enumerate(categories)}
                    self.cat_str_maps[col] = str_map
                as_str = series.astype(str)
                str_codes = as_str.map(str_map)
                replace_mask = codes.isna()
                if replace_mask.any():
                    codes = codes.where(~replace_mask, str_codes)
            out[:, idx] = _codes_to_int64_with_unknown(codes, unknown_idx)
        return out


class _LazyFTSupervisedDataset(Dataset):
    """Lazy supervised dataset that tensorizes rows on demand."""

    def __init__(
        self,
        owner: "FTTransformerSklearn",
        X: pd.DataFrame,
        y,
        w=None,
        geo_tokens=None,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        n_rows = len(X)
        owner._validate_vector(y, "y", n_rows)
        owner._validate_vector(w, "w", n_rows)
        self.pre = _FTPreprocessorSnapshot.from_owner(owner)
        self.n_rows = int(n_rows)
        self.X_num = X[self.pre.num_cols] if self.pre.num_cols else None
        self.X_cat = X[self.pre.cat_cols] if self.pre.cat_cols else None
        y_np = y.to_numpy(dtype=np.float32, copy=False) if hasattr(
            y, "to_numpy") else np.asarray(y, dtype=np.float32)
        self.y_values = np.asarray(y_np, dtype=np.float32).reshape(-1)
        if w is None:
            self.w_values = None
        else:
            w_np = w.to_numpy(dtype=np.float32, copy=False) if hasattr(
                w, "to_numpy") else np.asarray(w, dtype=np.float32)
            self.w_values = np.asarray(w_np, dtype=np.float32).reshape(-1)

        if geo_tokens is not None:
            geo_np = geo_tokens.to_numpy(dtype=np.float32, copy=False) if hasattr(
                geo_tokens, "to_numpy") else np.asarray(geo_tokens, dtype=np.float32)
            if geo_np.ndim == 1:
                geo_np = geo_np.reshape(-1, 1)
            if geo_np.shape[0] != n_rows:
                raise ValueError("geo_tokens length does not match X rows.")
            self.geo_values = np.asarray(geo_np, dtype=np.float32)
        elif self.pre.num_geo > 0:
            raise RuntimeError("geo_tokens must not be empty; prepare geo tokens first.")
        else:
            self.geo_values = None

    def __len__(self) -> int:
        return self.n_rows

    def __getitem__(self, idx: int):
        if self.X_num is None:
            X_num = torch.zeros((0,), dtype=torch.float32)
        else:
            num_np = self.X_num.iloc[idx].to_numpy(dtype=np.float32, copy=False)
            num_np = np.nan_to_num(num_np, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            if self.pre.num_mean is not None and self.pre.num_std is not None and num_np.size:
                num_np = (num_np - self.pre.num_mean) / self.pre.num_std
            X_num = torch.as_tensor(np.asarray(num_np, dtype=np.float32))

        if self.X_cat is None:
            X_cat = torch.zeros((0,), dtype=torch.long)
        else:
            cat_np = self.pre.encode_cats(self.X_cat.iloc[idx:idx + 1]).reshape(-1)
            X_cat = torch.as_tensor(cat_np, dtype=torch.long)

        if self.geo_values is None:
            X_geo = torch.zeros((0,), dtype=torch.float32)
        else:
            X_geo = torch.as_tensor(self.geo_values[idx], dtype=torch.float32)

        y_item = torch.as_tensor(self.y_values[idx:idx + 1], dtype=torch.float32)
        if self.w_values is None:
            w_item = torch.ones((1,), dtype=torch.float32)
        else:
            w_item = torch.as_tensor(self.w_values[idx:idx + 1], dtype=torch.float32)
        return X_num, X_cat, X_geo, y_item, w_item


class _LazyFTMaskedDataset(Dataset):
    """Lazy masked dataset for FT unsupervised pretraining."""

    def __init__(
        self,
        pre: _FTPreprocessorSnapshot,
        X: pd.DataFrame,
        *,
        geo_tokens=None,
        mask_prob_num: float = 0.15,
        mask_prob_cat: float = 0.15,
        seed: int = 13,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        self.pre = pre
        self.n_rows = int(len(X))
        self.X_num = X[self.pre.num_cols] if self.pre.num_cols else None
        self.X_cat = X[self.pre.cat_cols] if self.pre.cat_cols else None
        self.mask_prob_num = float(mask_prob_num)
        self.mask_prob_cat = float(mask_prob_cat)
        self.seed = int(seed)

        if geo_tokens is not None:
            geo_np = geo_tokens.to_numpy(dtype=np.float32, copy=False) if hasattr(
                geo_tokens, "to_numpy") else np.asarray(geo_tokens, dtype=np.float32)
            if geo_np.ndim == 1:
                geo_np = geo_np.reshape(-1, 1)
            if geo_np.shape[0] != self.n_rows:
                raise ValueError("geo_tokens length does not match X rows.")
            self.geo_values = np.asarray(geo_np, dtype=np.float32)
        elif self.pre.num_geo > 0:
            raise RuntimeError("geo_tokens must not be empty; prepare geo tokens first.")
        else:
            self.geo_values = None

        self.num_dim = len(self.pre.num_cols)
        self.cat_dim = len(self.pre.cat_cols)
        self.num_fill = np.zeros((self.num_dim,), dtype=np.float32)
        self.unknown_idx = np.asarray(
            [self.pre.unknown_indices[col] for col in self.pre.cat_cols],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return self.n_rows

    def _rng_for_row(self, row_idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(row_idx))

    def __getitem__(self, idx: int):
        row_idx = int(idx)
        rng = self._rng_for_row(row_idx)

        if self.num_dim > 0:
            num_true_np = self.pre.encode_nums(self.X_num.iloc[row_idx:row_idx + 1]).reshape(-1)
            num_mask_np = rng.random(self.num_dim) < self.mask_prob_num
            num_masked_np = num_true_np.copy()
            if num_mask_np.any():
                num_masked_np[num_mask_np] = self.num_fill[num_mask_np]
            X_num = torch.as_tensor(num_masked_np, dtype=torch.float32)
            num_true = torch.as_tensor(num_true_np, dtype=torch.float32)
            num_mask = torch.as_tensor(num_mask_np, dtype=torch.bool)
        else:
            X_num = torch.zeros((0,), dtype=torch.float32)
            num_true = None
            num_mask = None

        if self.cat_dim > 0:
            cat_true_np = self.pre.encode_cats(self.X_cat.iloc[row_idx:row_idx + 1]).reshape(-1)
            cat_mask_np = rng.random(self.cat_dim) < self.mask_prob_cat
            cat_masked_np = cat_true_np.copy()
            if cat_mask_np.any():
                cat_masked_np[cat_mask_np] = self.unknown_idx[cat_mask_np]
            X_cat = torch.as_tensor(cat_masked_np, dtype=torch.long)
            cat_true = torch.as_tensor(cat_true_np, dtype=torch.long)
            cat_mask = torch.as_tensor(cat_mask_np, dtype=torch.bool)
        else:
            X_cat = torch.zeros((0,), dtype=torch.long)
            cat_true = None
            cat_mask = None

        if self.geo_values is None:
            X_geo = torch.zeros((0,), dtype=torch.float32)
        else:
            X_geo = torch.as_tensor(self.geo_values[row_idx], dtype=torch.float32)

        return X_num, X_cat, X_geo, num_true, num_mask, cat_true, cat_mask


# Scikit-Learn style wrapper for FTTransformer.


class FTTransformerSklearn(TorchTrainerMixin, nn.Module):

    # sklearn-style wrapper:
    #   - num_cols: numeric feature column names
    #   - cat_cols: categorical feature column names (label-encoded to [0, n_classes-1])

    @staticmethod
    def resolve_numeric_token_count(num_cols, cat_cols, requested: Optional[int]) -> int:
        num_cols_count = len(num_cols or [])
        if num_cols_count == 0:
            return 0
        if requested is not None:
            count = int(requested)
            if count <= 0:
                raise ValueError("num_numeric_tokens must be >= 1 when numeric features exist.")
            return count
        return max(1, num_cols_count)

    def __init__(self, model_nme: str, num_cols, cat_cols, d_model: int = 64, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.1, batch_num: int = 100, epochs: int = 100,
                 task_type: str = 'regression',
                 tweedie_power: float = 1.5, learning_rate: float = 1e-3, patience: int = 10,
                 weight_decay: float = 0.0,
                 use_data_parallel: bool = True,
                 use_ddp: bool = False,
                 use_gpu: bool = True,
                 num_numeric_tokens: Optional[int] = None,
                 loss_name: Optional[str] = None,
                 distribution: Optional[str] = None,
                 ):
        super().__init__()

        self.use_gpu = bool(use_gpu)
        self.use_ddp = bool(use_ddp and self.use_gpu)
        if use_ddp and not self.use_gpu:
            _log(">>> FT DDP requested with use_gpu=false; forcing CPU single-process mode.")
        self.is_ddp_enabled, self.local_rank, self.rank, self.world_size = setup_ddp_if_requested(
            self.use_ddp
        )

        self.model_nme = model_nme
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.num_numeric_tokens = self.resolve_numeric_token_count(
            self.num_cols,
            self.cat_cols,
            num_numeric_tokens,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_num = batch_num
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task_type = task_type
        self.patience = patience
        resolved_distribution = normalize_distribution_name(distribution, self.task_type)
        self.distribution = None if resolved_distribution == "auto" else resolved_distribution
        if self.task_type == 'classification':
            self.loss_name = "logloss"
            self.tw_power = None  # No Tweedie power for classification.
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

        self.device = resolve_training_device(
            is_ddp_enabled=self.is_ddp_enabled,
            local_rank=self.local_rank,
            use_gpu=self.use_gpu,
        )
        self.cat_cardinalities = None
        self.cat_categories = {}
        self.cat_maps: Dict[str, Dict[Any, int]] = {}
        self.cat_str_maps: Dict[str, Dict[str, int]] = {}
        self._num_mean = None
        self._num_std = None
        self.ft = None
        self.use_data_parallel = bool(use_data_parallel and self.use_gpu)
        self.num_geo = 0
        self._geo_params: Dict[str, Any] = {}
        self.loss_curve_path: Optional[str] = None
        self.training_history: Dict[str, List[float]] = {
            "train": [], "val": []}
        self.use_lazy_dataset: bool = True
        self.predict_batch_size: Optional[int] = None

    def _fit_preprocessor(self, X_train) -> None:
        """Fit normalization/category mappings on training data (CPU only)."""
        num_numeric = len(self.num_cols)
        cat_cardinalities = []

        if num_numeric > 0:
            num_arr = X_train[self.num_cols].to_numpy(
                dtype=np.float32, copy=False)
            num_arr = np.nan_to_num(num_arr, nan=0.0, posinf=0.0, neginf=0.0)
            mean = num_arr.mean(axis=0).astype(np.float32, copy=False)
            std = num_arr.std(axis=0).astype(np.float32, copy=False)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
            self._num_mean = mean
            self._num_std = std
        else:
            self._num_mean = None
            self._num_std = None

        self.cat_maps = {}
        self.cat_str_maps = {}
        for col in self.cat_cols:
            cats = X_train[col].astype('category')
            categories = cats.cat.categories
            self.cat_categories[col] = categories           # Store full category list from training.
            self.cat_maps[col] = {cat: i for i, cat in enumerate(categories)}
            if categories.dtype == object or pd.api.types.is_string_dtype(categories.dtype):
                self.cat_str_maps[col] = {str(cat): i for i, cat in enumerate(categories)}

            card = len(categories) + 1                      # Reserve one extra class for unknown/missing.
            cat_cardinalities.append(card)

        self.cat_cardinalities = cat_cardinalities

    def _build_model_core(self) -> None:
        """Build FT core model from fitted metadata and move to target device."""
        if self.cat_cardinalities is None:
            raise RuntimeError(
                "cat_cardinalities is None. Call _fit_preprocessor(X_train) before building model."
            )
        num_numeric = len(self.num_cols)
        core = FTTransformerCore(
            num_numeric=num_numeric,
            cat_cardinalities=self.cat_cardinalities,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            task_type=self.task_type,
            num_geo=self.num_geo,
            num_numeric_tokens=self.num_numeric_tokens
        )
        core, self.use_data_parallel, self.device = wrap_model_for_parallel(
            core,
            device=self.device,
            use_data_parallel=self.use_data_parallel,
            use_ddp_requested=self.use_ddp,
            is_ddp_enabled=self.is_ddp_enabled,
            local_rank=self.local_rank,
            ddp_find_unused_parameters=True,
            fallback_log=_log,
        )
        self.ft = core.to(self.device)

    def _build_model(self, X_train):
        """Backward-compatible helper: fit preprocessors then build model."""
        self._fit_preprocessor(X_train)
        self._build_model_core()

    def _encode_cats(self, X):
        # Input DataFrame must include all categorical feature columns.
        # Return int64 array with shape (N, num_categorical_features).

        if not self.cat_cols:
            return np.zeros((len(X), 0), dtype='int64')

        n_rows = len(X)
        n_cols = len(self.cat_cols)
        X_cat_np = np.empty((n_rows, n_cols), dtype='int64')
        for idx, col in enumerate(self.cat_cols):
            categories = self.cat_categories[col]
            mapping = self.cat_maps.get(col)
            if mapping is None:
                mapping = {cat: i for i, cat in enumerate(categories)}
                self.cat_maps[col] = mapping
            unknown_idx = len(categories)
            series = X[col]
            codes = series.map(mapping)
            unmapped = series.notna() & codes.isna()
            if unmapped.any():
                try:
                    series_cast = series.astype(categories.dtype)
                except Exception:
                    series_cast = None
                if series_cast is not None:
                    codes = series_cast.map(mapping)
                    unmapped = series_cast.notna() & codes.isna()
            if unmapped.any():
                str_map = self.cat_str_maps.get(col)
                if str_map is None:
                    str_map = {str(cat): i for i, cat in enumerate(categories)}
                    self.cat_str_maps[col] = str_map
                as_str = series.astype(str)
                str_codes = as_str.map(str_map)
                replace_mask = codes.isna()
                if replace_mask.any():
                    codes = codes.where(~replace_mask, str_codes)
            X_cat_np[:, idx] = _codes_to_int64_with_unknown(codes, unknown_idx)
        return X_cat_np

    def _build_train_tensors(self, X_train, y_train, w_train, geo_train=None):
        return self._tensorize_split(X_train, y_train, w_train, geo_tokens=geo_train)

    def _build_val_tensors(self, X_val, y_val, w_val, geo_val=None):
        return self._tensorize_split(X_val, y_val, w_val, geo_tokens=geo_val, allow_none=True)

    def _build_train_val_datasets(self, X_train, y_train, w_train, X_val, y_val, w_val, geo_train=None, geo_val=None):
        train_dataset = _LazyFTSupervisedDataset(
            self, X_train, y_train, w_train, geo_tokens=geo_train)
        has_val = X_val is not None and y_val is not None
        if has_val:
            val_dataset = _LazyFTSupervisedDataset(
                self, X_val, y_val, w_val, geo_tokens=geo_val)
        else:
            val_dataset = None
        return train_dataset, val_dataset, has_val

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

    def _tensorize_split(self, X, y, w, geo_tokens=None, allow_none: bool = False):
        if X is None:
            if allow_none:
                return None, None, None, None, None, False
            raise ValueError("Input features X must not be None.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        missing_cols = [
            col for col in (self.num_cols + self.cat_cols) if col not in X.columns
        ]
        if missing_cols:
            raise ValueError(f"X is missing required columns: {missing_cols}")
        n_rows = len(X)
        if y is not None:
            self._validate_vector(y, "y", n_rows)
        if w is not None:
            self._validate_vector(w, "w", n_rows)

        num_np = X[self.num_cols].to_numpy(dtype=np.float32, copy=False)
        if not num_np.flags["OWNDATA"]:
            num_np = num_np.copy()
        num_np = np.nan_to_num(num_np, nan=0.0,
                               posinf=0.0, neginf=0.0, copy=False)
        if self._num_mean is not None and self._num_std is not None and num_np.size:
            num_np = (num_np - self._num_mean) / self._num_std
        X_num = torch.as_tensor(num_np)
        if self.cat_cols:
            X_cat = torch.as_tensor(self._encode_cats(X), dtype=torch.long)
        else:
            X_cat = torch.zeros((X_num.shape[0], 0), dtype=torch.long)

        if geo_tokens is not None:
            geo_np = np.asarray(geo_tokens, dtype=np.float32)
            if geo_np.shape[0] != n_rows:
                raise ValueError(
                    "geo_tokens length does not match X rows.")
            if geo_np.ndim == 1:
                geo_np = geo_np.reshape(-1, 1)
        elif self.num_geo > 0:
            raise RuntimeError("geo_tokens must not be empty; prepare geo tokens first.")
        else:
            geo_np = np.zeros((X_num.shape[0], 0), dtype=np.float32)
        X_geo = torch.as_tensor(geo_np)

        y_tensor = torch.as_tensor(
            y.to_numpy(dtype=np.float32, copy=False) if hasattr(
                y, "to_numpy") else np.asarray(y, dtype=np.float32)
        ).view(-1, 1) if y is not None else None
        if y_tensor is None:
            w_tensor = None
        elif w is not None:
            w_tensor = torch.as_tensor(
                w.to_numpy(dtype=np.float32, copy=False) if hasattr(
                    w, "to_numpy") else np.asarray(w, dtype=np.float32)
            ).view(-1, 1)
        else:
            w_tensor = torch.ones_like(y_tensor)
        return X_num, X_cat, X_geo, y_tensor, w_tensor, y is not None

    def fit(self, X_train, y_train, w_train=None,
            X_val=None, y_val=None, w_val=None, trial=None,
            geo_train=None, geo_val=None):

        # Prepare data mappings/statistics before creating DataLoader workers.
        self.num_geo = geo_train.shape[1] if geo_train is not None else 0
        if self.ft is None:
            self._fit_preprocessor(X_train)

        use_lazy = bool(getattr(self, "use_lazy_dataset", True))
        if use_lazy:
            dataset, val_dataset, has_val = self._build_train_val_datasets(
                X_train,
                y_train,
                w_train,
                X_val,
                y_val,
                w_val,
                geo_train=geo_train,
                geo_val=geo_val,
            )
            if not getattr(self, "_lazy_dataset_logged", False):
                _log(
                    ">>> FTTransformer using lazy supervised dataset to avoid full tensor materialization.",
                    flush=True,
                )
                self._lazy_dataset_logged = True
            train_rows = len(dataset)
        else:
            X_num_train, X_cat_train, X_geo_train, y_tensor, w_tensor, _ = self._build_train_tensors(
                X_train, y_train, w_train, geo_train=geo_train)
            X_num_val, X_cat_val, X_geo_val, y_val_tensor, w_val_tensor, has_val = self._build_val_tensors(
                X_val, y_val, w_val, geo_val=geo_val)
            dataset = TabularDataset(
                X_num_train, X_cat_train, X_geo_train, y_tensor, w_tensor
            )
            val_dataset = (
                TabularDataset(X_num_val, X_cat_val, X_geo_val, y_val_tensor, w_val_tensor)
                if has_val else None
            )
            train_rows = int(X_num_train.shape[0])

        dataloader, accum_steps = self._build_dataloader(
            dataset,
            N=train_rows,
            base_bs_gpu=(2048, 1024, 512),
            base_bs_cpu=(256, 128),
            min_bs=64,
            target_effective_cuda=2048,
            target_effective_cpu=1024
        )

        if self.ft is None:
            self._build_model_core()

        if self.is_ddp_enabled and hasattr(dataloader.sampler, 'set_epoch'):
            self.dataloader_sampler = dataloader.sampler
        else:
            self.dataloader_sampler = None

        optimizer = torch.optim.Adam(
            self.ft.parameters(),
            lr=self.learning_rate,
            weight_decay=float(getattr(self, "weight_decay", 0.0)),
        )
        scaler = create_grad_scaler(self.device.type)

        val_dataloader = None
        if has_val:
            val_dataloader = self._build_val_dataloader(
                val_dataset, dataloader, accum_steps)

        # Check for both DataParallel and DDP wrappers
        is_data_parallel = isinstance(self.ft, (nn.DataParallel, DDP))

        def forward_fn(batch):
            X_num_b, X_cat_b, X_geo_b, y_b, w_b = batch

            # For DataParallel, inputs are automatically scattered; for DDP, move to local device
            if not isinstance(self.ft, nn.DataParallel):
                X_num_b = X_num_b.to(self.device, non_blocking=True)
                X_cat_b = X_cat_b.to(self.device, non_blocking=True)
                X_geo_b = X_geo_b.to(self.device, non_blocking=True)
            y_b = y_b.to(self.device, non_blocking=True)
            w_b = w_b.to(self.device, non_blocking=True)

            y_pred = self.ft(X_num_b, X_cat_b, X_geo_b)
            return y_pred, y_b, w_b

        def val_forward_fn():
            total_loss = 0.0
            total_weight = 0.0
            for batch in val_dataloader:
                X_num_b, X_cat_b, X_geo_b, y_b, w_b = batch
                if not isinstance(self.ft, nn.DataParallel):
                    X_num_b = X_num_b.to(self.device, non_blocking=True)
                    X_cat_b = X_cat_b.to(self.device, non_blocking=True)
                    X_geo_b = X_geo_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)
                w_b = w_b.to(self.device, non_blocking=True)

                y_pred = self.ft(X_num_b, X_cat_b, X_geo_b)

                # Manually compute validation loss.
                losses = self._compute_losses(
                    y_pred, y_b, apply_softplus=False)

                batch_weight_sum = torch.clamp(w_b.sum(), min=EPS)
                batch_weighted_loss_sum = (losses * w_b.view(-1)).sum()

                total_loss += batch_weighted_loss_sum.item()
                total_weight += batch_weight_sum.item()

            return total_loss / max(total_weight, EPS)

        clip_fn = None
        if self.device.type == 'cuda':
            def clip_fn(): return (scaler.unscale_(optimizer),
                                   clip_grad_norm_(self.ft.parameters(), max_norm=1.0))

        best_state, history = self._train_model(
            self.ft,
            dataloader,
            accum_steps,
            optimizer,
            scaler,
            forward_fn,
            val_forward_fn if has_val else None,
            apply_softplus=False,
            clip_fn=clip_fn,
            trial=trial,
            loss_curve_path=getattr(self, "loss_curve_path", None)
        )

        if has_val and best_state is not None:
            # Load state into unwrapped module to match how it was saved
            base_module = self.ft.module if hasattr(self.ft, "module") else self.ft
            base_module.load_state_dict(best_state)
        self.training_history = history

    def fit_unsupervised(self,
                         X_train,
                         X_val=None,
                         trial: Optional[optuna.trial.Trial] = None,
                         geo_train=None,
                         geo_val=None,
                         mask_prob_num: float = 0.15,
                         mask_prob_cat: float = 0.15,
                         num_loss_weight: float = 1.0,
                         cat_loss_weight: float = 1.0) -> float:
        """Self-supervised pretraining via masked reconstruction (supports raw string categories)."""
        self.num_geo = geo_train.shape[1] if geo_train is not None else 0
        if self.ft is None:
            self._fit_preprocessor(X_train)

        use_lazy = bool(getattr(self, "use_lazy_dataset", True))
        device_type = self._device_type()
        has_val = X_val is not None

        val_dataset = None
        val_dataloader = None
        X_num_val = X_cat_val = X_geo_val = None
        if use_lazy:
            pre = _FTPreprocessorSnapshot.from_owner(self)
            dataset = _LazyFTMaskedDataset(
                pre,
                X_train,
                geo_tokens=geo_train,
                mask_prob_num=mask_prob_num,
                mask_prob_cat=mask_prob_cat,
                seed=13 + int(getattr(self, "rank", 0)),
            )
            if has_val:
                val_dataset = _LazyFTMaskedDataset(
                    pre,
                    X_val,
                    geo_tokens=geo_val,
                    mask_prob_num=mask_prob_num,
                    mask_prob_cat=mask_prob_cat,
                    seed=10_000,
                )
            if not getattr(self, "_lazy_unsupervised_logged", False):
                _log(
                    ">>> FTTransformer using lazy masked dataset for unsupervised pretraining.",
                    flush=True,
                )
                self._lazy_unsupervised_logged = True
            N = len(dataset)
        else:
            X_num, X_cat, X_geo, _, _, _ = self._tensorize_split(
                X_train, None, None, geo_tokens=geo_train, allow_none=True)
            if has_val:
                X_num_val, X_cat_val, X_geo_val, _, _, _ = self._tensorize_split(
                    X_val, None, None, geo_tokens=geo_val, allow_none=True)
            else:
                X_num_val = X_cat_val = X_geo_val = None

            N = int(X_num.shape[0])
            num_dim = int(X_num.shape[1])
            cat_dim = int(X_cat.shape[1])
            gen = torch.Generator()
            gen.manual_seed(13 + int(getattr(self, "rank", 0)))
            unknown_idx = torch.tensor(
                [int(c) - 1 for c in list(self.cat_cardinalities or [])],
                dtype=torch.long,
            ).view(1, -1)
            means = torch.zeros((1, num_dim), dtype=torch.float32)

            def _mask_inputs(X_num_in: torch.Tensor,
                             X_cat_in: torch.Tensor,
                             generator: torch.Generator):
                n_rows = int(X_num_in.shape[0])
                num_mask_local = (
                    torch.rand((n_rows, num_dim), generator=generator) < float(mask_prob_num)
                )
                cat_mask_local = (
                    torch.rand((n_rows, cat_dim), generator=generator) < float(mask_prob_cat)
                )
                X_num_masked_local = X_num_in.clone()
                X_cat_masked_local = X_cat_in.clone()
                if num_mask_local.any():
                    X_num_masked_local[num_mask_local] = means.expand_as(
                        X_num_masked_local
                    )[num_mask_local]
                if cat_mask_local.any():
                    X_cat_masked_local[cat_mask_local] = unknown_idx.expand_as(
                        X_cat_masked_local
                    )[cat_mask_local]
                return X_num_masked_local, X_cat_masked_local, num_mask_local, cat_mask_local

            X_num_masked, X_cat_masked, num_mask, cat_mask = _mask_inputs(X_num, X_cat, gen)
            dataset = MaskedTabularDataset(
                X_num_masked,
                X_cat_masked,
                X_geo,
                X_num,
                num_mask,
                X_cat,
                cat_mask,
            )
            if has_val and X_num_val is not None and X_cat_val is not None and X_geo_val is not None:
                gen_val = torch.Generator()
                gen_val.manual_seed(10_000)
                X_num_val_masked, X_cat_val_masked, num_mask_val, cat_mask_val = _mask_inputs(
                    X_num_val,
                    X_cat_val,
                    gen_val,
                )
                val_dataset = MaskedTabularDataset(
                    X_num_val_masked,
                    X_cat_val_masked,
                    X_geo_val,
                    X_num_val,
                    num_mask_val,
                    X_cat_val,
                    cat_mask_val,
                )

        dataloader, accum_steps = self._build_dataloader(
            dataset,
            N=N,
            base_bs_gpu=(2048, 1024, 512),
            base_bs_cpu=(256, 128),
            min_bs=64,
            target_effective_cuda=2048,
            target_effective_cpu=1024
        )
        if self.ft is None:
            self._build_model_core()
        if self.is_ddp_enabled and hasattr(dataloader.sampler, 'set_epoch'):
            self.dataloader_sampler = dataloader.sampler
        else:
            self.dataloader_sampler = None

        optimizer = torch.optim.Adam(
            self.ft.parameters(),
            lr=self.learning_rate,
            weight_decay=float(getattr(self, "weight_decay", 0.0)),
        )
        scaler = create_grad_scaler(device_type)
        if use_lazy and has_val and val_dataset is not None:
            val_dataloader = self._build_val_dataloader(
                val_dataset,
                dataloader,
                accum_steps,
            )

        train_history: List[float] = []
        val_history: List[float] = []
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        is_ddp_model = isinstance(self.ft, DDP)
        use_collectives = dist.is_initialized() and is_ddp_model

        clip_fn = None
        if self.device.type == 'cuda':
            def clip_fn(): return (scaler.unscale_(optimizer),
                                   clip_grad_norm_(self.ft.parameters(), max_norm=1.0))

        for epoch in range(1, int(self.epochs) + 1):
            if self.dataloader_sampler is not None:
                self.dataloader_sampler.set_epoch(epoch)

            self.ft.train()
            optimizer.zero_grad()
            epoch_loss_sum = 0.0
            epoch_count = 0.0

            for step, batch in enumerate(dataloader):
                is_update_step = ((step + 1) % accum_steps == 0) or \
                    ((step + 1) == len(dataloader))
                sync_cm = self.ft.no_sync if (
                    is_ddp_model and not is_update_step) else nullcontext
                with sync_cm():
                    with create_autocast_context(device_type):
                        X_num_b, X_cat_b, X_geo_b, num_true_b, num_mask_b, cat_true_b, cat_mask_b = batch
                        X_num_b = X_num_b.to(self.device, non_blocking=True)
                        X_cat_b = X_cat_b.to(self.device, non_blocking=True)
                        X_geo_b = X_geo_b.to(self.device, non_blocking=True)
                        num_true_b = None if num_true_b is None else num_true_b.to(
                            self.device, non_blocking=True)
                        num_mask_b = None if num_mask_b is None else num_mask_b.to(
                            self.device, non_blocking=True)
                        cat_true_b = None if cat_true_b is None else cat_true_b.to(
                            self.device, non_blocking=True)
                        cat_mask_b = None if cat_mask_b is None else cat_mask_b.to(
                            self.device, non_blocking=True)

                        num_pred, cat_logits = self.ft(
                            X_num_b, X_cat_b, X_geo_b, return_reconstruction=True)
                        batch_loss = _compute_reconstruction_loss(
                            num_pred, cat_logits, num_true_b, num_mask_b,
                            cat_true_b, cat_mask_b, num_loss_weight, cat_loss_weight,
                            device=X_num_b.device)
                        local_loss_value = float(batch_loss.detach().item())
                        local_bad = 0 if np.isfinite(local_loss_value) else 1
                        global_bad = local_bad
                        first_bad_rank = int(self.rank) if local_bad else -1
                        if use_collectives:
                            bad = torch.tensor(
                                [local_bad],
                                device=batch_loss.device,
                                dtype=torch.int32,
                            )
                            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                            global_bad = int(bad.item())
                            sentinel = max(1, int(self.world_size))
                            bad_rank = torch.tensor(
                                [int(self.rank) if local_bad else sentinel],
                                device=batch_loss.device,
                                dtype=torch.int32,
                            )
                            dist.all_reduce(bad_rank, op=dist.ReduceOp.MIN)
                            bad_rank_val = int(bad_rank.item())
                            first_bad_rank = (
                                bad_rank_val if bad_rank_val < sentinel else -1
                            )

                        if global_bad:
                            msg = (
                                "[FTTransformerSklearn.fit_unsupervised] non-finite loss "
                                f"detected (epoch={epoch}, step={step}, local_rank={int(self.rank)}, "
                                f"local_loss={local_loss_value}, bad_rank={first_bad_rank})"
                            )
                            should_log = (
                                not dist.is_initialized()
                                or DistributedUtils.is_main_process()
                                or bool(local_bad)
                            )
                            if should_log:
                                _log(msg, flush=True)
                                if local_bad:
                                    _log(
                                        f"  X_num: finite={bool(torch.isfinite(X_num_b).all())} "
                                        f"min={float(X_num_b.min().detach().cpu()) if X_num_b.numel() else 0.0:.3g} "
                                        f"max={float(X_num_b.max().detach().cpu()) if X_num_b.numel() else 0.0:.3g}",
                                        flush=True,
                                    )
                                    if X_geo_b is not None:
                                        _log(
                                            f"  X_geo: finite={bool(torch.isfinite(X_geo_b).all())} "
                                            f"min={float(X_geo_b.min().detach().cpu()) if X_geo_b.numel() else 0.0:.3g} "
                                            f"max={float(X_geo_b.max().detach().cpu()) if X_geo_b.numel() else 0.0:.3g}",
                                            flush=True,
                                        )
                            if trial is not None or use_collectives:
                                raise optuna.TrialPruned(msg)
                            raise RuntimeError(msg)
                        loss_for_backward = batch_loss / float(accum_steps)
                    scaler.scale(loss_for_backward).backward()

                if is_update_step:
                    if clip_fn is not None:
                        clip_fn()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss_sum += float(batch_loss.detach().item()) * \
                    float(X_num_b.shape[0])
                epoch_count += float(X_num_b.shape[0])

            train_history.append(epoch_loss_sum / max(epoch_count, 1.0))

            if (
                (not use_lazy)
                and has_val
                and X_num_val is not None
                and X_cat_val is not None
                and X_geo_val is not None
            ):
                should_compute_val = (not dist.is_initialized()
                                      or DistributedUtils.is_main_process())
                loss_tensor_device = self.device if device_type == 'cuda' else torch.device(
                    "cpu")
                val_loss_tensor = torch.zeros(1, device=loss_tensor_device)

                if should_compute_val:
                    self.ft.eval()
                    with torch.no_grad(), create_autocast_context(device_type):
                        val_bs = min(
                            int(dataloader.batch_size * max(1, accum_steps)), int(X_num_val.shape[0]))
                        total_val = 0.0
                        total_n = 0.0
                        for start in range(0, int(X_num_val.shape[0]), max(1, val_bs)):
                            end = min(
                                int(X_num_val.shape[0]), start + max(1, val_bs))
                            X_num_v_true_cpu = X_num_val[start:end]
                            X_cat_v_true_cpu = X_cat_val[start:end]
                            X_geo_v = X_geo_val[start:end].to(
                                self.device, non_blocking=True)
                            gen_val = torch.Generator()
                            gen_val.manual_seed(10_000 + epoch + start)
                            X_num_v_cpu, X_cat_v_cpu, val_num_mask, val_cat_mask = _mask_inputs(
                                X_num_v_true_cpu, X_cat_v_true_cpu, gen_val)
                            X_num_v_true = X_num_v_true_cpu.to(
                                self.device, non_blocking=True)
                            X_cat_v_true = X_cat_v_true_cpu.to(
                                self.device, non_blocking=True)
                            X_num_v = X_num_v_cpu.to(
                                self.device, non_blocking=True)
                            X_cat_v = X_cat_v_cpu.to(
                                self.device, non_blocking=True)
                            val_num_mask = None if val_num_mask is None else val_num_mask.to(
                                self.device, non_blocking=True)
                            val_cat_mask = None if val_cat_mask is None else val_cat_mask.to(
                                self.device, non_blocking=True)
                            num_pred_v, cat_logits_v = self.ft(
                                X_num_v, X_cat_v, X_geo_v, return_reconstruction=True)
                            loss_v = _compute_reconstruction_loss(
                                num_pred_v, cat_logits_v,
                                X_num_v_true if X_num_v_true.numel() else None, val_num_mask,
                                X_cat_v_true if X_cat_v_true.numel() else None, val_cat_mask,
                                num_loss_weight, cat_loss_weight,
                                device=X_num_v.device
                            )
                            if not torch.isfinite(loss_v):
                                total_val = float("inf")
                                total_n = 1.0
                                break
                            total_val += float(loss_v.detach().item()
                                               ) * float(end - start)
                            total_n += float(end - start)
                    val_loss_tensor[0] = total_val / max(total_n, 1.0)

                if use_collectives:
                    dist.broadcast(val_loss_tensor, src=0)
                val_loss_value = float(val_loss_tensor.item())
                prune_now = False
                prune_msg = None
                if not np.isfinite(val_loss_value):
                    prune_now = True
                    prune_msg = (
                        f"[FTTransformerSklearn.fit_unsupervised] non-finite val loss "
                        f"(epoch={epoch}, val_loss={val_loss_value})"
                    )
                val_history.append(val_loss_value)

                if val_loss_value < best_loss:
                    best_loss = val_loss_value
                    base_module = self.ft.module if hasattr(self.ft, "module") else self.ft
                    best_state = {
                        k: v.detach().clone().cpu() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                        for k, v in base_module.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if best_state is not None and patience_counter >= int(self.patience):
                        break

                if trial is not None and (not dist.is_initialized() or DistributedUtils.is_main_process()):
                    trial.report(val_loss_value, epoch)
                    if trial.should_prune():
                        prune_now = True

                if use_collectives:
                    flag = torch.tensor(
                        [1 if prune_now else 0],
                        device=loss_tensor_device,
                        dtype=torch.int32,
                    )
                    dist.broadcast(flag, src=0)
                    prune_now = bool(flag.item())

                if prune_now:
                    if prune_msg:
                        raise optuna.TrialPruned(prune_msg)
                    raise optuna.TrialPruned()

            if val_dataloader is not None:
                should_compute_val = (not dist.is_initialized()
                                      or DistributedUtils.is_main_process())
                loss_tensor_device = self.device if device_type == 'cuda' else torch.device(
                    "cpu")
                val_loss_tensor = torch.zeros(1, device=loss_tensor_device)

                if should_compute_val:
                    self.ft.eval()
                    with torch.no_grad(), create_autocast_context(device_type):
                        total_val = 0.0
                        total_n = 0.0
                        for batch in val_dataloader:
                            X_num_v, X_cat_v, X_geo_v, X_num_v_true, val_num_mask, X_cat_v_true, val_cat_mask = batch
                            X_num_v = X_num_v.to(self.device, non_blocking=True)
                            X_cat_v = X_cat_v.to(self.device, non_blocking=True)
                            X_geo_v = X_geo_v.to(self.device, non_blocking=True)
                            X_num_v_true = None if X_num_v_true is None else X_num_v_true.to(
                                self.device, non_blocking=True)
                            val_num_mask = None if val_num_mask is None else val_num_mask.to(
                                self.device, non_blocking=True)
                            X_cat_v_true = None if X_cat_v_true is None else X_cat_v_true.to(
                                self.device, non_blocking=True)
                            val_cat_mask = None if val_cat_mask is None else val_cat_mask.to(
                                self.device, non_blocking=True)
                            num_pred_v, cat_logits_v = self.ft(
                                X_num_v, X_cat_v, X_geo_v, return_reconstruction=True)
                            loss_v = _compute_reconstruction_loss(
                                num_pred_v, cat_logits_v,
                                X_num_v_true, val_num_mask,
                                X_cat_v_true, val_cat_mask,
                                num_loss_weight, cat_loss_weight,
                                device=X_num_v.device
                            )
                            if not torch.isfinite(loss_v):
                                total_val = float("inf")
                                total_n = 1.0
                                break
                            total_val += float(loss_v.detach().item()) * float(X_num_v.shape[0])
                            total_n += float(X_num_v.shape[0])
                    val_loss_tensor[0] = total_val / max(total_n, 1.0)

                if use_collectives:
                    dist.broadcast(val_loss_tensor, src=0)
                val_loss_value = float(val_loss_tensor.item())
                prune_now = False
                prune_msg = None
                if not np.isfinite(val_loss_value):
                    prune_now = True
                    prune_msg = (
                        f"[FTTransformerSklearn.fit_unsupervised] non-finite val loss "
                        f"(epoch={epoch}, val_loss={val_loss_value})"
                    )
                val_history.append(val_loss_value)

                if val_loss_value < best_loss:
                    best_loss = val_loss_value
                    # Efficiently clone state_dict - only clone tensor data, not DDP metadata
                    base_module = self.ft.module if hasattr(self.ft, "module") else self.ft
                    best_state = {
                        k: v.detach().clone().cpu() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                        for k, v in base_module.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if best_state is not None and patience_counter >= int(self.patience):
                        break

                if trial is not None and (not dist.is_initialized() or DistributedUtils.is_main_process()):
                    trial.report(val_loss_value, epoch)
                    if trial.should_prune():
                        prune_now = True

                if use_collectives:
                    flag = torch.tensor(
                        [1 if prune_now else 0],
                        device=loss_tensor_device,
                        dtype=torch.int32,
                    )
                    dist.broadcast(flag, src=0)
                    prune_now = bool(flag.item())

                if prune_now:
                    if prune_msg:
                        raise optuna.TrialPruned(prune_msg)
                    raise optuna.TrialPruned()

        self.training_history = {"train": train_history, "val": val_history}
        self._plot_loss_curve(self.training_history, getattr(
            self, "loss_curve_path", None))
        if has_val and best_state is not None:
            # Load state into unwrapped module to match how it was saved
            base_module = self.ft.module if hasattr(self.ft, "module") else self.ft
            base_module.load_state_dict(best_state)
        return float(best_loss if has_val else (train_history[-1] if train_history else 0.0))

    @staticmethod
    def _slice_rows(X, start: int, end: int):
        if hasattr(X, "iloc"):
            return X.iloc[start:end]
        return X[start:end]

    @staticmethod
    def _slice_geo_tokens(geo_tokens, start: int, end: int):
        if geo_tokens is None:
            return None
        if hasattr(geo_tokens, "iloc"):
            return geo_tokens.iloc[start:end]
        return geo_tokens[start:end]

    def _resolve_predict_batch_size(self, n_rows: int, batch_size: Optional[int] = None) -> int:
        if batch_size is not None:
            return max(1, min(int(batch_size), n_rows))
        raw = getattr(self, "predict_batch_size", None)
        if raw is not None:
            try:
                resolved = int(raw)
            except (TypeError, ValueError):
                resolved = 0
            if resolved > 0:
                return max(1, min(resolved, n_rows))
        device = self.device if isinstance(
            self.device, torch.device) else torch.device(self.device)
        token_cnt = self.num_numeric_tokens + len(self.cat_cols)
        if self.num_geo > 0:
            token_cnt += 1
        approx_units = max(1, token_cnt * max(1, self.d_model))
        if device.type == 'cuda':
            if approx_units >= 8192:
                base = 512
            elif approx_units >= 4096:
                base = 1024
            else:
                base = 2048
        else:
            base = 512
        return max(1, min(base, n_rows))

    def predict(self, X_test, geo_tokens=None, batch_size: Optional[int] = None, return_embedding: bool = False):
        # X_test must include all numeric/categorical columns; geo_tokens is optional.
        self.ft.eval()
        num_rows = len(X_test)
        if num_rows == 0:
            return np.empty(0, dtype=np.float32)

        device = self.device if isinstance(
            self.device, torch.device) else torch.device(self.device)
        eff_batch = self._resolve_predict_batch_size(num_rows, batch_size=batch_size)
        preds: List[torch.Tensor] = []

        inference_cm = getattr(torch, "inference_mode", torch.no_grad)
        with inference_cm():
            for start in range(0, num_rows, eff_batch):
                end = min(num_rows, start + eff_batch)
                X_batch = self._slice_rows(X_test, start, end)
                geo_batch = self._slice_geo_tokens(geo_tokens, start, end)
                X_num_b, X_cat_b, X_geo_b, _, _, _ = self._tensorize_split(
                    X_batch, None, None, geo_tokens=geo_batch, allow_none=True)
                X_num_b = X_num_b.to(device, non_blocking=True)
                X_cat_b = X_cat_b.to(device, non_blocking=True)
                X_geo_b = X_geo_b.to(device, non_blocking=True)
                pred_chunk = self.ft(
                    X_num_b, X_cat_b, X_geo_b, return_embedding=return_embedding)
                preds.append(pred_chunk.cpu())

        y_pred = torch.cat(preds, dim=0).numpy()

        if return_embedding:
            return y_pred

        if self.task_type == 'classification':
            # Convert logits to probabilities.
            y_pred = 1 / (1 + np.exp(-y_pred))
        else:
            # Model already has softplus; optionally apply log-exp smoothing: y_pred = log(1 + exp(y_pred)).
            y_pred = np.clip(y_pred, 1e-6, None)
        return y_pred.ravel()

    def set_params(self, params: dict):

        # Keep sklearn-style behavior.
        # Note: changing structural params (e.g., d_model/n_heads) requires refit to take effect.

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in model.")
        return self
