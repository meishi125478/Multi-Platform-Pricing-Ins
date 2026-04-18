from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import log_loss

from ins_pricing.modelling.bayesopt.checkpoints import rebuild_resn_model_from_payload
from ins_pricing.modelling.bayesopt.trainers.trainer_base import TrainerBase
from ins_pricing.modelling.bayesopt.models import ResNetSklearn
from ins_pricing.utils.losses import regression_loss
from ins_pricing.utils import get_logger, log_print
from ins_pricing.utils.torch_compat import torch_load

_logger = get_logger("ins_pricing.trainer.resn")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)

class ResNetTrainer(TrainerBase):
    def __init__(self, context: "BayesOptModel") -> None:
        if context.task_type == 'classification':
            super().__init__(context, 'ResNetClassifier', 'ResNet')
        else:
            super().__init__(context, 'ResNet', 'ResNet')
        self.model: Optional[ResNetSklearn] = None
        dist_cfg = getattr(context.config, "distributed", context.config)
        self.enable_distributed_optuna = bool(
            getattr(dist_cfg, "use_resn_ddp", False) and context.use_gpu
        )
        self._param_probe_model: Optional[ResNetSklearn] = None
        self._resn_tweedie_space_warned = False
        self._raw_design_cache: Optional[Dict[str, Any]] = None
        self._raw_fallback_logged = False
        self._raw_encoded_cols_logged = False

    def _dist_cfg(self):
        return getattr(self.ctx.config, "distributed", self.ctx.config)

    def _maybe_cleanup_gpu(self, model: Optional[ResNetSklearn]) -> None:
        if not bool(getattr(self.ctx.config, "resn_cleanup_per_fold", False)):
            return
        if model is not None:
            getattr(getattr(model, "resnet", None), "to",
                    lambda *_args, **_kwargs: None)("cpu")
        synchronize = bool(getattr(self.ctx.config, "resn_cleanup_synchronize", False))
        self._clean_gpu(synchronize=synchronize)

    def _resolve_input_dim(self) -> int:
        data = getattr(self.ctx, "train_oht_scl_data", None)
        if data is not None and getattr(self.ctx, "var_nmes", None):
            return int(data[self.ctx.var_nmes].shape[1])
        return int(len(self.ctx.var_nmes or []))

    def _coerce_raw_feature_frames(
        self,
        X_train_raw: pd.DataFrame,
        X_test_raw: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
        """Coerce raw feature frames into float32 matrices for ResNet fallback.

        When build_oht=false, ResNet can still consume raw features as long as
        all columns are numeric. Non-numeric columns are ordinal-encoded with a
        train-derived mapping and unknown test categories mapped to -1.
        """
        train_cols: List[pd.Series] = []
        test_cols: Optional[List[pd.Series]] = [] if X_test_raw is not None else None
        encoded_cols: List[str] = []

        for col in X_train_raw.columns:
            tr_col = X_train_raw[col]
            te_col = X_test_raw[col] if X_test_raw is not None else None

            if pd.api.types.is_numeric_dtype(tr_col) or pd.api.types.is_bool_dtype(tr_col):
                train_cols.append(
                    pd.to_numeric(tr_col, errors="coerce").astype(np.float32, copy=False).rename(col)
                )
                if test_cols is not None:
                    test_cols.append(
                        pd.to_numeric(te_col, errors="coerce").astype(np.float32, copy=False).rename(col)
                    )
                continue

            encoded_cols.append(str(col))
            train_norm = tr_col.astype("object").where(tr_col.notna(), "<NA>")
            categories = pd.Index(pd.unique(train_norm))
            mapping = {val: float(i) for i, val in enumerate(categories)}
            train_cols.append(train_norm.map(mapping).astype(np.float32, copy=False).rename(col))
            if test_cols is not None:
                test_norm = te_col.astype("object").where(te_col.notna(), "<NA>")
                test_cols.append(
                    test_norm.map(mapping).fillna(-1.0).astype(np.float32, copy=False).rename(col)
                )

        train_out = (
            pd.concat(train_cols, axis=1)
            if train_cols
            else pd.DataFrame(index=X_train_raw.index)
        )
        test_out = (
            pd.concat(test_cols, axis=1)
            if test_cols is not None and test_cols
            else (pd.DataFrame(index=X_test_raw.index) if X_test_raw is not None else None)
        )

        return train_out, test_out, encoded_cols

    def _resolve_raw_design_matrices(
        self,
        *,
        require_test: bool,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        feature_cols = list(getattr(self.ctx, "factor_nmes", []) or [])
        if not feature_cols:
            raise RuntimeError(
                "ResNet raw fallback requires non-empty factor_nmes when build_oht=false."
            )
        signature = (tuple(feature_cols), bool(require_test))
        cached = self._raw_design_cache or {}
        if (
            cached.get("signature") == signature
            and cached.get("X_train") is not None
            and (not require_test or cached.get("X_test") is not None)
        ):
            return cached["X_train"], cached.get("X_test")

        train_raw = self.ctx.train_data[feature_cols]
        test_raw = self.ctx.test_data[feature_cols] if require_test else None
        if require_test and test_raw is None:
            raise RuntimeError("ResNet raw fallback requires test_data for prediction.")

        X_train, X_test, encoded_cols = self._coerce_raw_feature_frames(train_raw, test_raw)
        self._raw_design_cache = {
            "signature": signature,
            "X_train": X_train,
            "X_test": X_test,
        }

        if not self._raw_fallback_logged:
            _log(
                "[ResNet] build_oht=false fallback: using raw feature matrix (float32) "
                "instead of one-hot inputs.",
                flush=True,
            )
            self._raw_fallback_logged = True
        if encoded_cols and not self._raw_encoded_cols_logged:
            preview = ", ".join(encoded_cols[:8])
            suffix = "..." if len(encoded_cols) > 8 else ""
            _log(
                "[ResNet] Encoded non-numeric raw columns for fallback: "
                f"{preview}{suffix}",
                flush=True,
            )
            self._raw_encoded_cols_logged = True

        return X_train, X_test

    def _uses_tweedie_power(self) -> bool:
        loss_name = str(getattr(self.ctx, "loss_name", "tweedie") or "tweedie").strip().lower()
        task_type = str(getattr(self.ctx, "task_type", "") or "").strip().lower()
        return task_type == "regression" and loss_name == "tweedie"

    def _drop_tw_power_if_unused(
        self,
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        out = dict(params or {})
        if "tw_power" not in out:
            return out
        if self._uses_tweedie_power():
            return out
        out.pop("tw_power", None)
        return out

    def _filter_search_space_for_distribution(
        self,
        search_space: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        filtered = dict(search_space or {})
        if "tw_power" not in filtered:
            return filtered
        if self._uses_tweedie_power():
            return filtered

        filtered.pop("tw_power", None)
        if not self._resn_tweedie_space_warned:
            loss_name = str(getattr(self.ctx, "loss_name", "tweedie") or "tweedie").strip().lower()
            task_type = str(getattr(self.ctx, "task_type", "") or "").strip().lower()
            _log(
                "[ResNet] Ignoring resn_search_space.tw_power "
                f"because resolved loss_name='{loss_name}' (task_type='{task_type}').",
                flush=True,
            )
            self._resn_tweedie_space_warned = True
        return filtered

    def _build_model(self, params: Optional[Dict[str, Any]] = None) -> ResNetSklearn:
        params = params or {}
        loss_name = getattr(self.ctx, "loss_name", "tweedie")
        power = params.get("tw_power")
        if self.ctx.task_type == "regression":
            base_tw = self.ctx.default_tweedie_power()
            if loss_name == "tweedie":
                power = base_tw if power is None else float(power)
            elif loss_name in ("poisson", "gamma"):
                power = base_tw
            else:
                power = None
        resn_weight_decay = float(
            params.get(
                "weight_decay",
                getattr(self.ctx.config, "resn_weight_decay", 1e-4),
            )
        )
        model = ResNetSklearn(
            model_nme=self.ctx.model_nme,
            input_dim=self._resolve_input_dim(),
            hidden_dim=int(params.get("hidden_dim", 64)),
            block_num=int(params.get("block_num", 2)),
            task_type=self.ctx.task_type,
            epochs=self.ctx.epochs,
            tweedie_power=power,
            learning_rate=float(params.get("learning_rate", 0.01)),
            patience=int(params.get("patience", 10)),
            use_layernorm=True,
            dropout=float(params.get("dropout", 0.1)),
            residual_scale=float(params.get("residual_scale", 0.1)),
            stochastic_depth=float(params.get("stochastic_depth", 0.0)),
            weight_decay=resn_weight_decay,
            use_data_parallel=bool(
                getattr(self._dist_cfg(), "use_resn_data_parallel", False)
            ),
            use_ddp=bool(getattr(self._dist_cfg(), "use_resn_ddp", False)),
            use_gpu=self.ctx.use_gpu,
            loss_name=loss_name,
            distribution=getattr(self.ctx, "distribution", None),
        )
        handled_keys = {
            "hidden_dim",
            "block_num",
            "learning_rate",
            "patience",
            "dropout",
            "residual_scale",
            "stochastic_depth",
            "weight_decay",
            "tw_power",
        }
        extra_params = {
            key: value for key, value in params.items()
            if key not in handled_keys
        }
        extra_params = self._filter_params_by_model_support(
            model,
            extra_params,
            context="ResNet model params",
        )
        if extra_params:
            model.set_params(extra_params)
        return self._apply_dataloader_overrides(model)

    def _sanitize_best_params(
        self,
        params: Dict[str, Any],
        *,
        context: str = "best_params",
    ) -> Dict[str, Any]:
        params = self._drop_tw_power_if_unused(params)
        probe = self._param_probe_model
        if probe is None:
            probe = self._build_model({})
            self._param_probe_model = probe
        return self._filter_params_by_model_support(
            probe,
            params,
            context=f"{context} (ResNet params)",
        )

    def _resolve_resn_cv_standardize_device(self) -> str:
        raw = str(
            os.environ.get("BAYESOPT_RESN_CV_STANDARDIZE_DEVICE", "cpu")
        ).strip().lower()
        if raw in {"gpu", "cuda"}:
            return "cuda"
        if raw in {"auto"}:
            if bool(getattr(self.ctx, "use_gpu", False)) and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return "cpu"

    def _standardize_fold_fast(
        self,
        X_train,
        X_val,
        columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize selected columns with optional GPU acceleration.

        Returns numpy arrays to reduce pandas copy pressure in CV folds.
        """
        if hasattr(X_train, "columns"):
            train_cols = list(X_train.columns)
            cols = [c for c in (columns or []) if c in train_cols]
            col_idx = [train_cols.index(c) for c in cols]
            X_train_np = X_train.to_numpy(dtype=np.float32, copy=True)
            X_val_np = X_val.to_numpy(dtype=np.float32, copy=True)
        else:
            X_train_np = np.asarray(X_train, dtype=np.float32).copy()
            X_val_np = np.asarray(X_val, dtype=np.float32).copy()
            col_idx = list(range(X_train_np.shape[1])) if X_train_np.ndim == 2 else []

        if not col_idx:
            return X_train_np, X_val_np

        mode = self._resolve_resn_cv_standardize_device()
        if not getattr(self, "_resn_cv_standardize_mode_logged", False):
            _log(
                f"[ResNet] CV standardize device={mode} "
                "(env BAYESOPT_RESN_CV_STANDARDIZE_DEVICE).",
                flush=True,
            )
            self._resn_cv_standardize_mode_logged = True

        if mode == "cuda":
            if not (bool(getattr(self.ctx, "use_gpu", False)) and torch.cuda.is_available()):
                _log(
                    "[ResNet] Requested CUDA standardization but GPU is unavailable; fallback to CPU.",
                    flush=True,
                )
            else:
                try:
                    device = torch.device("cuda", torch.cuda.current_device())
                    tr = torch.as_tensor(X_train_np[:, col_idx], device=device)
                    va = torch.as_tensor(X_val_np[:, col_idx], device=device)
                    mean = tr.mean(dim=0)
                    std = tr.std(dim=0, unbiased=False)
                    std = torch.where(std < 1e-6, torch.ones_like(std), std)
                    X_train_np[:, col_idx] = ((tr - mean) / std).cpu().numpy()
                    X_val_np[:, col_idx] = ((va - mean) / std).cpu().numpy()
                    return X_train_np, X_val_np
                except Exception as exc:
                    _log(
                        f"[ResNet] CUDA standardization failed; fallback to CPU ({exc}).",
                        flush=True,
                    )

        # CPU fast path (numpy)
        tr_cols = X_train_np[:, col_idx]
        va_cols = X_val_np[:, col_idx]
        mean = tr_cols.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = tr_cols.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
        X_train_np[:, col_idx] = (tr_cols - mean) / std
        X_val_np[:, col_idx] = (va_cols - mean) / std
        return X_train_np, X_val_np

    # ========= Cross-validation (for BayesOpt) =========
    def cross_val(self, trial: optuna.trial.Trial) -> float:
        # ResNet CV focuses on memory control:
        #   - Create a ResNetSklearn per fold and release it immediately after.
        #   - Move model to CPU, delete, and call gc/empty_cache after each fold.
        #   - Optionally sample part of training data during BayesOpt to reduce memory.

        base_tw_power = self.ctx.default_tweedie_power()
        loss_name = getattr(self.ctx, "loss_name", "tweedie")

        def data_provider():
            data = self.ctx.train_oht_data if self.ctx.train_oht_data is not None else self.ctx.train_oht_scl_data
            if data is not None:
                return data[self.ctx.var_nmes], data[self.ctx.resp_nme], data[self.ctx.weight_nme]
            X_train_raw, _ = self._resolve_raw_design_matrices(require_test=False)
            return (
                X_train_raw,
                self.ctx.train_data[self.ctx.resp_nme],
                self.ctx.train_data[self.ctx.weight_nme],
            )

        metric_ctx: Dict[str, Any] = {}

        def model_builder(params):
            if loss_name == "tweedie":
                power = params.get("tw_power", base_tw_power)
            elif loss_name in ("poisson", "gamma"):
                power = base_tw_power
            else:
                power = None
            metric_ctx["tw_power"] = power
            params_local = dict(params)
            if power is not None:
                params_local["tw_power"] = power
            return self._build_model(params_local)

        def preprocess_fn(X_train, X_val):
            X_train_s, X_val_s = self._standardize_fold_fast(
                X_train,
                X_val,
                columns=self.ctx.num_features,
            )
            return X_train_s, X_val_s

        def fit_predict(model, X_train, y_train, w_train, X_val, y_val, w_val, trial_obj):
            model.fit(
                X_train, y_train, w_train,
                X_val, y_val, w_val,
                trial=trial_obj
            )
            return model.predict(X_val)

        def metric_fn(y_true, y_pred, weight):
            if self.ctx.task_type == 'regression':
                return regression_loss(
                    y_true,
                    y_pred,
                    weight,
                    loss_name=loss_name,
                    tweedie_power=metric_ctx.get("tw_power", base_tw_power),
                )
            return log_loss(y_true, y_pred, sample_weight=weight)

        sample_cap = data_provider()[0]
        max_rows_for_resnet_bo = min(100000, int(len(sample_cap)/5))
        search_space = self._filter_search_space_for_distribution(
            self._get_search_space_config("resn_search_space")
        )
        param_space: Dict[str, Any] = {}

        param_space = self._augment_param_space_with_search_space(
            model_key="resn",
            param_space=param_space,
            search_space=search_space,
        )

        return self.cross_val_generic(
            trial=trial,
            hyperparameter_space=param_space,
            data_provider=data_provider,
            model_builder=model_builder,
            metric_fn=metric_fn,
            sample_limit=max_rows_for_resnet_bo if len(
                sample_cap) > max_rows_for_resnet_bo > 0 else None,
            preprocess_fn=preprocess_fn,
            fit_predict_fn=fit_predict,
            cleanup_fn=lambda m: getattr(
                getattr(m, "resnet", None), "to", lambda *_args, **_kwargs: None)("cpu")
        )

    # ========= Train final ResNet with best hyperparameters =========
    def train(self) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best ResNet parameters.")

        params = self._sanitize_best_params(
            dict(self.best_params),
            context="ResNet final train params",
        )
        use_refit = bool(getattr(self.ctx.config, "final_refit", True))
        data = self.ctx.train_oht_scl_data
        use_oht = data is not None
        design_fn = None
        if use_oht:
            X_all = data[self.ctx.var_nmes]
            y_all = data[self.ctx.resp_nme]
            w_all = data[self.ctx.weight_nme]
        else:
            X_all, X_test = self._resolve_raw_design_matrices(require_test=True)
            y_all = self.ctx.train_data[self.ctx.resp_nme]
            w_all = self.ctx.train_data[self.ctx.weight_nme]
            design_fn = (
                lambda train=False, _X_train=X_all, _X_test=X_test:
                _X_train if train else _X_test
            )

        refit_epochs = None
        split = self._resolve_train_val_indices(X_all)
        if use_refit and split is not None:
            train_idx, val_idx = split
            tmp_model = self._build_model(params)
            tmp_model.fit(
                X_all.iloc[train_idx],
                y_all.iloc[train_idx],
                w_all.iloc[train_idx],
                X_all.iloc[val_idx],
                y_all.iloc[val_idx],
                w_all.iloc[val_idx],
                trial=None,
            )
            refit_epochs = self._resolve_best_epoch(
                getattr(tmp_model, "training_history", None),
                default_epochs=int(self.ctx.epochs),
            )
            self._maybe_cleanup_gpu(tmp_model)

        self.model = self._build_model(params)
        if refit_epochs is not None:
            self.model.epochs = int(refit_epochs)
        self.best_params = params
        loss_plot_path = self.output.plot_path(
            f'{self.ctx.model_nme}/loss/loss_{self.ctx.model_nme}_{self.model_name_prefix}.png')
        self.model.loss_curve_path = loss_plot_path

        self._fit_predict_cache(
            self.model,
            X_all,
            y_all,
            sample_weight=w_all,
            pred_prefix='resn',
            use_oht=use_oht,
            design_fn=design_fn,
            sample_weight_arg='w_train'
        )

        # Convenience wrapper for external callers.
        self.ctx.resn_best = self.model

    def ensemble_predict(self, k: int) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best ResNet parameters.")
        best_params = self._sanitize_best_params(
            dict(self.best_params),
            context="ResNet ensemble params",
        )
        data = self.ctx.train_oht_scl_data
        test_data = self.ctx.test_oht_scl_data
        if data is not None and test_data is not None:
            X_all = data[self.ctx.var_nmes]
            y_all = data[self.ctx.resp_nme]
            w_all = data[self.ctx.weight_nme]
            X_test = test_data[self.ctx.var_nmes]
        else:
            X_all, X_test = self._resolve_raw_design_matrices(require_test=True)
            y_all = self.ctx.train_data[self.ctx.resp_nme]
            w_all = self.ctx.train_data[self.ctx.weight_nme]

        k = max(2, int(k))
        n_samples = len(X_all)
        split_iter, _ = self._resolve_ensemble_splits(X_all, k=k)
        if split_iter is None:
            _log(
                f"[ResNet Ensemble] unable to build CV split (n_samples={n_samples}); skip ensemble.",
                flush=True,
            )
            return
        preds_train_sum = np.zeros(n_samples, dtype=np.float64)
        preds_test_sum = np.zeros(len(X_test), dtype=np.float64)

        split_count = 0
        for train_idx, val_idx in split_iter:
            model = self._build_model(best_params)
            model.fit(
                X_all.iloc[train_idx],
                y_all.iloc[train_idx],
                w_all.iloc[train_idx],
                X_all.iloc[val_idx],
                y_all.iloc[val_idx],
                w_all.iloc[val_idx],
                trial=None,
            )
            pred_train = model.predict(X_all)
            pred_test = model.predict(X_test)
            preds_train_sum += np.asarray(pred_train, dtype=np.float64)
            preds_test_sum += np.asarray(pred_test, dtype=np.float64)
            self._maybe_cleanup_gpu(model)
            split_count += 1

        if split_count < 1:
            _log(
                f"[ResNet Ensemble] no CV splits generated; skip ensemble.",
                flush=True,
            )
            return
        preds_train = preds_train_sum / float(split_count)
        preds_test = preds_test_sum / float(split_count)
        self._cache_predictions("resn", preds_train, preds_test)

    # ========= Save / Load =========
    # ResNet is saved as state_dict and needs a custom load path.
    # Save logic is implemented in TrainerBase (checks .resnet attribute).

    def load(self) -> None:
        # Load ResNet weights to the current device to match context.
        path = self.output.model_path(self._get_model_filename())
        if os.path.exists(path):
            payload = torch_load(path, map_location='cpu', weights_only=False)
            params_fallback = (
                self._sanitize_best_params(dict(self.best_params), context="checkpoint_load")
                if isinstance(self.best_params, dict)
                else None
            )
            if isinstance(payload, dict) and isinstance(payload.get("best_params"), dict):
                payload["best_params"] = self._sanitize_best_params(
                    dict(payload["best_params"]),
                    context="checkpoint_load",
                )
            resn_loaded, resolved_params = rebuild_resn_model_from_payload(
                payload=payload,
                model_builder=self._build_model,
                params_fallback=params_fallback,
                require_params=False,
            )
            self.best_params = self._sanitize_best_params(
                resolved_params,
                context="checkpoint_load",
            )
            self._move_to_device(resn_loaded)
            self.model = resn_loaded
            self.ctx.resn_best = self.model
        else:
            _log(f"[ResNetTrainer.load] Model file not found: {path}")
