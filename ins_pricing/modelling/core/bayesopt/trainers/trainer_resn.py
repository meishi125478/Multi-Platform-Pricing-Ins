from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from sklearn.metrics import log_loss, mean_tweedie_deviance

from .trainer_base import TrainerBase
from ..models import ResNetSklearn

class ResNetTrainer(TrainerBase):
    def __init__(self, context: "BayesOptModel") -> None:
        if context.task_type == 'classification':
            super().__init__(context, 'ResNetClassifier', 'ResNet')
        else:
            super().__init__(context, 'ResNet', 'ResNet')
        self.model: Optional[ResNetSklearn] = None
        self.enable_distributed_optuna = bool(context.config.use_resn_ddp)

    def _resolve_input_dim(self) -> int:
        data = getattr(self.ctx, "train_oht_scl_data", None)
        if data is not None and getattr(self.ctx, "var_nmes", None):
            return int(data[self.ctx.var_nmes].shape[1])
        return int(len(self.ctx.var_nmes or []))

    def _build_model(self, params: Optional[Dict[str, Any]] = None) -> ResNetSklearn:
        params = params or {}
        power = params.get("tw_power", self.ctx.default_tweedie_power())
        if power is not None:
            power = float(power)
        resn_weight_decay = float(
            params.get(
                "weight_decay",
                getattr(self.ctx.config, "resn_weight_decay", 1e-4),
            )
        )
        return ResNetSklearn(
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
            use_data_parallel=self.ctx.config.use_resn_data_parallel,
            use_ddp=self.ctx.config.use_resn_ddp
        )

    # ========= Cross-validation (for BayesOpt) =========
    def cross_val(self, trial: optuna.trial.Trial) -> float:
        # ResNet CV focuses on memory control:
        #   - Create a ResNetSklearn per fold and release it immediately after.
        #   - Move model to CPU, delete, and call gc/empty_cache after each fold.
        #   - Optionally sample part of training data during BayesOpt to reduce memory.

        base_tw_power = self.ctx.default_tweedie_power()

        def data_provider():
            data = self.ctx.train_oht_data if self.ctx.train_oht_data is not None else self.ctx.train_oht_scl_data
            assert data is not None, "Preprocessed training data is missing."
            return data[self.ctx.var_nmes], data[self.ctx.resp_nme], data[self.ctx.weight_nme]

        metric_ctx: Dict[str, Any] = {}

        def model_builder(params):
            power = params.get("tw_power", base_tw_power)
            metric_ctx["tw_power"] = power
            params_local = dict(params)
            params_local["tw_power"] = power
            return self._build_model(params_local)

        def preprocess_fn(X_train, X_val):
            X_train_s, X_val_s, _ = self._standardize_fold(
                X_train, X_val, self.ctx.num_features)
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
                return mean_tweedie_deviance(
                    y_true,
                    y_pred,
                    sample_weight=weight,
                    power=metric_ctx.get("tw_power", base_tw_power)
                )
            return log_loss(y_true, y_pred, sample_weight=weight)

        sample_cap = data_provider()[0]
        max_rows_for_resnet_bo = min(100000, int(len(sample_cap)/5))

        return self.cross_val_generic(
            trial=trial,
            hyperparameter_space={
                "learning_rate": lambda t: t.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
                "hidden_dim": lambda t: t.suggest_int('hidden_dim', 8, 32, step=2),
                "block_num": lambda t: t.suggest_int('block_num', 2, 10),
                "dropout": lambda t: t.suggest_float('dropout', 0.0, 0.3, step=0.05),
                "residual_scale": lambda t: t.suggest_float('residual_scale', 0.05, 0.3, step=0.05),
                "patience": lambda t: t.suggest_int('patience', 3, 12),
                "stochastic_depth": lambda t: t.suggest_float('stochastic_depth', 0.0, 0.2, step=0.05),
                **({"tw_power": lambda t: t.suggest_float('tw_power', 1.0, 2.0)} if self.ctx.task_type == 'regression' and self.ctx.obj == 'reg:tweedie' else {})
            },
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

        params = dict(self.best_params)
        use_refit = bool(getattr(self.ctx.config, "final_refit", True))
        data = self.ctx.train_oht_scl_data
        if data is None:
            raise RuntimeError("Missing standardized data for ResNet training.")
        X_all = data[self.ctx.var_nmes]
        y_all = data[self.ctx.resp_nme]
        w_all = data[self.ctx.weight_nme]

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
            getattr(getattr(tmp_model, "resnet", None), "to",
                    lambda *_args, **_kwargs: None)("cpu")
            self._clean_gpu()

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
            use_oht=True,
            sample_weight_arg='w_train'
        )

        # Convenience wrapper for external callers.
        self.ctx.resn_best = self.model

    def ensemble_predict(self, k: int) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best ResNet parameters.")
        data = self.ctx.train_oht_scl_data
        test_data = self.ctx.test_oht_scl_data
        if data is None or test_data is None:
            raise RuntimeError("Missing standardized data for ResNet ensemble.")
        X_all = data[self.ctx.var_nmes]
        y_all = data[self.ctx.resp_nme]
        w_all = data[self.ctx.weight_nme]
        X_test = test_data[self.ctx.var_nmes]

        k = max(2, int(k))
        n_samples = len(X_all)
        split_iter, _ = self._resolve_ensemble_splits(X_all, k=k)
        if split_iter is None:
            print(
                f"[ResNet Ensemble] unable to build CV split (n_samples={n_samples}); skip ensemble.",
                flush=True,
            )
            return
        preds_train_sum = np.zeros(n_samples, dtype=np.float64)
        preds_test_sum = np.zeros(len(X_test), dtype=np.float64)

        split_count = 0
        for train_idx, val_idx in split_iter:
            model = self._build_model(self.best_params)
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
            getattr(getattr(model, "resnet", None), "to",
                    lambda *_args, **_kwargs: None)("cpu")
            self._clean_gpu()
            split_count += 1

        if split_count < 1:
            print(
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
            payload = torch.load(path, map_location='cpu')
            if isinstance(payload, dict) and "state_dict" in payload:
                state_dict = payload.get("state_dict")
                params = payload.get("best_params") or self.best_params
            else:
                state_dict = payload
                params = self.best_params
            resn_loaded = self._build_model(params)
            resn_loaded.resnet.load_state_dict(state_dict)

            self._move_to_device(resn_loaded)
            self.model = resn_loaded
            self.ctx.resn_best = self.model
        else:
            print(f"[ResNetTrainer.load] Model file not found: {path}")

