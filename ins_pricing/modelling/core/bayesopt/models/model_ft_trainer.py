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
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from ..utils import DistributedUtils, EPS, TorchTrainerMixin
from .model_ft_components import FTTransformerCore, MaskedTabularDataset, TabularDataset


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
                 num_numeric_tokens: Optional[int] = None
                 ):
        super().__init__()

        self.use_ddp = use_ddp
        self.is_ddp_enabled, self.local_rank, self.rank, self.world_size = (
            False, 0, 0, 1)
        if self.use_ddp:
            self.is_ddp_enabled, self.local_rank, self.rank, self.world_size = DistributedUtils.setup_ddp()

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
        if self.task_type == 'classification':
            self.tw_power = None  # No Tweedie power for classification.
        elif 'f' in self.model_nme:
            self.tw_power = 1.0
        elif 's' in self.model_nme:
            self.tw_power = 2.0
        else:
            self.tw_power = tweedie_power

        if self.is_ddp_enabled:
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.cat_cardinalities = None
        self.cat_categories = {}
        self.cat_maps: Dict[str, Dict[Any, int]] = {}
        self.cat_str_maps: Dict[str, Dict[str, int]] = {}
        self._num_mean = None
        self._num_std = None
        self.ft = None
        self.use_data_parallel = bool(use_data_parallel)
        self.num_geo = 0
        self._geo_params: Dict[str, Any] = {}
        self.loss_curve_path: Optional[str] = None
        self.training_history: Dict[str, List[float]] = {
            "train": [], "val": []}

    def _build_model(self, X_train):
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

        core = FTTransformerCore(
            num_numeric=num_numeric,
            cat_cardinalities=cat_cardinalities,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            task_type=self.task_type,
            num_geo=self.num_geo,
            num_numeric_tokens=self.num_numeric_tokens
        )
        use_dp = self.use_data_parallel and (self.device.type == "cuda") and (torch.cuda.device_count() > 1)
        if self.is_ddp_enabled:
            core = core.to(self.device)
            core = DDP(core, device_ids=[
                       self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
            self.use_data_parallel = False
        elif use_dp:
            if self.use_ddp and not self.is_ddp_enabled:
                print(
                    ">>> DDP requested but not initialized; falling back to DataParallel.")
            core = nn.DataParallel(core, device_ids=list(
                range(torch.cuda.device_count())))
            self.device = torch.device("cuda")
            self.use_data_parallel = True
        else:
            self.use_data_parallel = False
        self.ft = core.to(self.device)

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
                codes = series.astype(str).map(str_map)
            if pd.api.types.is_categorical_dtype(codes):
                codes = codes.astype("float")
            codes = codes.fillna(unknown_idx).astype(
                "int64", copy=False).to_numpy()
            X_cat_np[:, idx] = codes
        return X_cat_np

    def _build_train_tensors(self, X_train, y_train, w_train, geo_train=None):
        return self._tensorize_split(X_train, y_train, w_train, geo_tokens=geo_train)

    def _build_val_tensors(self, X_val, y_val, w_val, geo_val=None):
        return self._tensorize_split(X_val, y_val, w_val, geo_tokens=geo_val, allow_none=True)

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

        # Build the underlying model on first fit.
        self.num_geo = geo_train.shape[1] if geo_train is not None else 0
        if self.ft is None:
            self._build_model(X_train)

        X_num_train, X_cat_train, X_geo_train, y_tensor, w_tensor, _ = self._build_train_tensors(
            X_train, y_train, w_train, geo_train=geo_train)
        X_num_val, X_cat_val, X_geo_val, y_val_tensor, w_val_tensor, has_val = self._build_val_tensors(
            X_val, y_val, w_val, geo_val=geo_val)

        # --- Build DataLoader ---
        dataset = TabularDataset(
            X_num_train, X_cat_train, X_geo_train, y_tensor, w_tensor
        )

        dataloader, accum_steps = self._build_dataloader(
            dataset,
            N=X_num_train.shape[0],
            base_bs_gpu=(2048, 1024, 512),
            base_bs_cpu=(256, 128),
            min_bs=64,
            target_effective_cuda=2048,
            target_effective_cpu=1024
        )

        if self.is_ddp_enabled and hasattr(dataloader.sampler, 'set_epoch'):
            self.dataloader_sampler = dataloader.sampler
        else:
            self.dataloader_sampler = None

        optimizer = torch.optim.Adam(
            self.ft.parameters(),
            lr=self.learning_rate,
            weight_decay=float(getattr(self, "weight_decay", 0.0)),
        )
        scaler = GradScaler(enabled=(self.device.type == 'cuda'))

        X_num_val_dev = X_cat_val_dev = y_val_dev = w_val_dev = None
        val_dataloader = None
        if has_val:
            val_dataset = TabularDataset(
                X_num_val, X_cat_val, X_geo_val, y_val_tensor, w_val_tensor
            )
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
            self._build_model(X_train)

        X_num, X_cat, X_geo, _, _, _ = self._tensorize_split(
            X_train, None, None, geo_tokens=geo_train, allow_none=True)
        has_val = X_val is not None
        if has_val:
            X_num_val, X_cat_val, X_geo_val, _, _, _ = self._tensorize_split(
                X_val, None, None, geo_tokens=geo_val, allow_none=True)
        else:
            X_num_val = X_cat_val = X_geo_val = None

        N = int(X_num.shape[0])
        num_dim = int(X_num.shape[1])
        cat_dim = int(X_cat.shape[1])
        device_type = self._device_type()

        gen = torch.Generator()
        gen.manual_seed(13 + int(getattr(self, "rank", 0)))

        base_model = self.ft.module if hasattr(self.ft, "module") else self.ft
        cardinals = getattr(base_model, "cat_cardinalities", None) or []
        unknown_idx = torch.tensor(
            [int(c) - 1 for c in cardinals], dtype=torch.long).view(1, -1)

        means = None
        if num_dim > 0:
            # Keep masked fill values on the same scale as model inputs (may be normalized in _tensorize_split).
            means = X_num.to(dtype=torch.float32).mean(dim=0, keepdim=True)

        def _mask_inputs(X_num_in: torch.Tensor,
                         X_cat_in: torch.Tensor,
                         generator: torch.Generator):
            n_rows = int(X_num_in.shape[0])
            num_mask_local = None
            cat_mask_local = None
            X_num_masked_local = X_num_in
            X_cat_masked_local = X_cat_in
            if num_dim > 0:
                num_mask_local = (torch.rand(
                    (n_rows, num_dim), generator=generator) < float(mask_prob_num))
                X_num_masked_local = X_num_in.clone()
                if num_mask_local.any():
                    X_num_masked_local[num_mask_local] = means.expand_as(
                        X_num_masked_local)[num_mask_local]
            if cat_dim > 0:
                cat_mask_local = (torch.rand(
                    (n_rows, cat_dim), generator=generator) < float(mask_prob_cat))
                X_cat_masked_local = X_cat_in.clone()
                if cat_mask_local.any():
                    X_cat_masked_local[cat_mask_local] = unknown_idx.expand_as(
                        X_cat_masked_local)[cat_mask_local]
            return X_num_masked_local, X_cat_masked_local, num_mask_local, cat_mask_local

        X_num_true = X_num if num_dim > 0 else None
        X_cat_true = X_cat if cat_dim > 0 else None
        X_num_masked, X_cat_masked, num_mask, cat_mask = _mask_inputs(
            X_num, X_cat, gen)

        dataset = MaskedTabularDataset(
            X_num_masked, X_cat_masked, X_geo,
            X_num_true, num_mask,
            X_cat_true, cat_mask
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
        if self.is_ddp_enabled and hasattr(dataloader.sampler, 'set_epoch'):
            self.dataloader_sampler = dataloader.sampler
        else:
            self.dataloader_sampler = None

        optimizer = torch.optim.Adam(
            self.ft.parameters(),
            lr=self.learning_rate,
            weight_decay=float(getattr(self, "weight_decay", 0.0)),
        )
        scaler = GradScaler(enabled=(device_type == 'cuda'))

        def _batch_recon_loss(num_pred, cat_logits, num_true_b, num_mask_b, cat_true_b, cat_mask_b, device):
            loss = torch.zeros((), device=device, dtype=torch.float32)

            if num_pred is not None and num_true_b is not None and num_mask_b is not None:
                num_mask_b = num_mask_b.to(dtype=torch.bool)
                if num_mask_b.any():
                    diff = num_pred - num_true_b
                    mse = diff * diff
                    loss = loss + float(num_loss_weight) * \
                        mse[num_mask_b].mean()

            if cat_logits and cat_true_b is not None and cat_mask_b is not None:
                cat_mask_b = cat_mask_b.to(dtype=torch.bool)
                cat_losses: List[torch.Tensor] = []
                for j, logits in enumerate(cat_logits):
                    mask_j = cat_mask_b[:, j]
                    if not mask_j.any():
                        continue
                    targets = cat_true_b[:, j]
                    cat_losses.append(
                        F.cross_entropy(logits, targets, reduction='none')[
                            mask_j].mean()
                    )
                if cat_losses:
                    loss = loss + float(cat_loss_weight) * \
                        torch.stack(cat_losses).mean()
            return loss

        train_history: List[float] = []
        val_history: List[float] = []
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        is_ddp_model = isinstance(self.ft, DDP)

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
                    with autocast(enabled=(device_type == 'cuda')):
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
                        batch_loss = _batch_recon_loss(
                            num_pred, cat_logits, num_true_b, num_mask_b, cat_true_b, cat_mask_b, device=X_num_b.device)
                        local_bad = 0 if bool(torch.isfinite(batch_loss)) else 1
                        global_bad = local_bad
                        if dist.is_initialized():
                            bad = torch.tensor(
                                [local_bad],
                                device=batch_loss.device,
                                dtype=torch.int32,
                            )
                            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                            global_bad = int(bad.item())

                        if global_bad:
                            msg = (
                                f"[FTTransformerSklearn.fit_unsupervised] non-finite loss "
                                f"(epoch={epoch}, step={step}, loss={batch_loss.detach().item()})"
                            )
                            should_log = (not dist.is_initialized()
                                          or DistributedUtils.is_main_process())
                            if should_log:
                                print(msg, flush=True)
                                print(
                                    f"  X_num: finite={bool(torch.isfinite(X_num_b).all())} "
                                    f"min={float(X_num_b.min().detach().cpu()) if X_num_b.numel() else 0.0:.3g} "
                                    f"max={float(X_num_b.max().detach().cpu()) if X_num_b.numel() else 0.0:.3g}",
                                    flush=True,
                                )
                                if X_geo_b is not None:
                                    print(
                                        f"  X_geo: finite={bool(torch.isfinite(X_geo_b).all())} "
                                        f"min={float(X_geo_b.min().detach().cpu()) if X_geo_b.numel() else 0.0:.3g} "
                                        f"max={float(X_geo_b.max().detach().cpu()) if X_geo_b.numel() else 0.0:.3g}",
                                        flush=True,
                                    )
                            if trial is not None:
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

            if has_val and X_num_val is not None and X_cat_val is not None and X_geo_val is not None:
                should_compute_val = (not dist.is_initialized()
                                      or DistributedUtils.is_main_process())
                loss_tensor_device = self.device if device_type == 'cuda' else torch.device(
                    "cpu")
                val_loss_tensor = torch.zeros(1, device=loss_tensor_device)

                if should_compute_val:
                    self.ft.eval()
                    with torch.no_grad(), autocast(enabled=(device_type == 'cuda')):
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
                            loss_v = _batch_recon_loss(
                                num_pred_v, cat_logits_v,
                                X_num_v_true if X_num_v_true.numel() else None, val_num_mask,
                                X_cat_v_true if X_cat_v_true.numel() else None, val_cat_mask,
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

                if dist.is_initialized():
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

                if dist.is_initialized():
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

    def predict(self, X_test, geo_tokens=None, batch_size: Optional[int] = None, return_embedding: bool = False):
        # X_test must include all numeric/categorical columns; geo_tokens is optional.

        self.ft.eval()
        X_num, X_cat, X_geo, _, _, _ = self._tensorize_split(
            X_test, None, None, geo_tokens=geo_tokens, allow_none=True)

        num_rows = X_num.shape[0]
        if num_rows == 0:
            return np.empty(0, dtype=np.float32)

        device = self.device if isinstance(
            self.device, torch.device) else torch.device(self.device)

        def resolve_batch_size(n_rows: int) -> int:
            if batch_size is not None:
                return max(1, min(int(batch_size), n_rows))
            # Estimate a safe batch size based on model size to avoid attention OOM.
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

        eff_batch = resolve_batch_size(num_rows)
        preds: List[torch.Tensor] = []

        inference_cm = getattr(torch, "inference_mode", torch.no_grad)
        with inference_cm():
            for start in range(0, num_rows, eff_batch):
                end = min(num_rows, start + eff_batch)
                X_num_b = X_num[start:end].to(device, non_blocking=True)
                X_cat_b = X_cat[start:end].to(device, non_blocking=True)
                X_geo_b = X_geo[start:end].to(device, non_blocking=True)
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
