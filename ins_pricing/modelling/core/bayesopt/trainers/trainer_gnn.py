from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from sklearn.metrics import log_loss, mean_tweedie_deviance

from .trainer_base import TrainerBase
from ..models import GraphNeuralNetSklearn
from ..utils import EPS
from ins_pricing.utils import get_logger

_logger = get_logger("ins_pricing.trainer.gnn")

class GNNTrainer(TrainerBase):
    def __init__(self, context: "BayesOptModel") -> None:
        super().__init__(context, 'GNN', 'GNN')
        self.model: Optional[GraphNeuralNetSklearn] = None
        self.enable_distributed_optuna = bool(context.config.use_gnn_ddp)

    def _build_model(self, params: Optional[Dict[str, Any]] = None) -> GraphNeuralNetSklearn:
        params = params or {}
        base_tw_power = self.ctx.default_tweedie_power()
        model = GraphNeuralNetSklearn(
            model_nme=f"{self.ctx.model_nme}_gnn",
            input_dim=len(self.ctx.var_nmes),
            hidden_dim=int(params.get("hidden_dim", 64)),
            num_layers=int(params.get("num_layers", 2)),
            k_neighbors=int(params.get("k_neighbors", 10)),
            dropout=float(params.get("dropout", 0.1)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            epochs=int(params.get("epochs", self.ctx.epochs)),
            patience=int(params.get("patience", 5)),
            task_type=self.ctx.task_type,
            tweedie_power=float(params.get("tw_power", base_tw_power or 1.5)),
            weight_decay=float(params.get("weight_decay", 0.0)),
            use_data_parallel=bool(self.ctx.config.use_gnn_data_parallel),
            use_ddp=bool(self.ctx.config.use_gnn_ddp),
            use_approx_knn=bool(self.ctx.config.gnn_use_approx_knn),
            approx_knn_threshold=int(self.ctx.config.gnn_approx_knn_threshold),
            graph_cache_path=self.ctx.config.gnn_graph_cache,
            max_gpu_knn_nodes=self.ctx.config.gnn_max_gpu_knn_nodes,
            knn_gpu_mem_ratio=float(self.ctx.config.gnn_knn_gpu_mem_ratio),
            knn_gpu_mem_overhead=float(
                self.ctx.config.gnn_knn_gpu_mem_overhead),
        )
        return model

    def cross_val(self, trial: optuna.trial.Trial) -> float:
        base_tw_power = self.ctx.default_tweedie_power()
        metric_ctx: Dict[str, Any] = {}

        def data_provider():
            data = self.ctx.train_oht_data if self.ctx.train_oht_data is not None else self.ctx.train_oht_scl_data
            assert data is not None, "Preprocessed training data is missing."
            return data[self.ctx.var_nmes], data[self.ctx.resp_nme], data[self.ctx.weight_nme]

        def model_builder(params: Dict[str, Any]):
            tw_power = params.get("tw_power", base_tw_power)
            metric_ctx["tw_power"] = tw_power
            return self._build_model(params)

        def preprocess_fn(X_train, X_val):
            X_train_s, X_val_s, _ = self._standardize_fold(
                X_train, X_val, self.ctx.num_features)
            return X_train_s, X_val_s

        def fit_predict(model, X_train, y_train, w_train, X_val, y_val, w_val, trial_obj):
            model.fit(
                X_train,
                y_train,
                w_train=w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                trial=trial_obj,
            )
            return model.predict(X_val)

        def metric_fn(y_true, y_pred, weight):
            if self.ctx.task_type == 'classification':
                y_pred_clipped = np.clip(y_pred, EPS, 1 - EPS)
                return log_loss(y_true, y_pred_clipped, sample_weight=weight)
            y_pred_safe = np.maximum(y_pred, EPS)
            power = metric_ctx.get("tw_power", base_tw_power or 1.5)
            return mean_tweedie_deviance(
                y_true,
                y_pred_safe,
                sample_weight=weight,
                power=power,
            )

        # Keep GNN BO lightweight: sample during CV, use full data for final training.
        X_cap = data_provider()[0]
        sample_limit = min(200000, len(X_cap)) if len(X_cap) > 200000 else None

        param_space: Dict[str, Callable[[optuna.trial.Trial], Any]] = {
            "learning_rate": lambda t: t.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
            "hidden_dim": lambda t: t.suggest_int('hidden_dim', 16, 128, step=16),
            "num_layers": lambda t: t.suggest_int('num_layers', 1, 4),
            "k_neighbors": lambda t: t.suggest_int('k_neighbors', 5, 30),
            "dropout": lambda t: t.suggest_float('dropout', 0.0, 0.3),
            "weight_decay": lambda t: t.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }
        if self.ctx.task_type == 'regression' and self.ctx.obj == 'reg:tweedie':
            param_space["tw_power"] = lambda t: t.suggest_float(
                'tw_power', 1.0, 2.0)

        return self.cross_val_generic(
            trial=trial,
            hyperparameter_space=param_space,
            data_provider=data_provider,
            model_builder=model_builder,
            metric_fn=metric_fn,
            sample_limit=sample_limit,
            preprocess_fn=preprocess_fn,
            fit_predict_fn=fit_predict,
            cleanup_fn=lambda m: getattr(
                getattr(m, "gnn", None), "to", lambda *_args, **_kwargs: None)("cpu")
        )

    def train(self) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best GNN parameters.")

        data = self.ctx.train_oht_scl_data
        assert data is not None, "Preprocessed training data is missing."
        X_all = data[self.ctx.var_nmes]
        y_all = data[self.ctx.resp_nme]
        w_all = data[self.ctx.weight_nme]

        use_refit = bool(getattr(self.ctx.config, "final_refit", True))
        refit_epochs = None

        split = self._resolve_train_val_indices(X_all)
        if split is not None:
            train_idx, val_idx = split
            X_train = X_all.iloc[train_idx]
            y_train = y_all.iloc[train_idx]
            w_train = w_all.iloc[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]
            w_val = w_all.iloc[val_idx]

            if use_refit:
                tmp_model = self._build_model(self.best_params)
                tmp_model.fit(
                    X_train,
                    y_train,
                    w_train=w_train,
                    X_val=X_val,
                    y_val=y_val,
                    w_val=w_val,
                    trial=None,
                )
                refit_epochs = int(getattr(tmp_model, "best_epoch", None) or self.ctx.epochs)
                getattr(getattr(tmp_model, "gnn", None), "to",
                        lambda *_args, **_kwargs: None)("cpu")
                self._clean_gpu()
            else:
                self.model = self._build_model(self.best_params)
                self.model.fit(
                    X_train,
                    y_train,
                    w_train=w_train,
                    X_val=X_val,
                    y_val=y_val,
                    w_val=w_val,
                    trial=None,
                )
        else:
            use_refit = False

        if use_refit:
            self.model = self._build_model(self.best_params)
            if refit_epochs is not None:
                self.model.epochs = int(refit_epochs)
            self.model.fit(
                X_all,
                y_all,
                w_train=w_all,
                X_val=None,
                y_val=None,
                w_val=None,
                trial=None,
            )
        elif self.model is None:
            self.model = self._build_model(self.best_params)
            self.model.fit(
                X_all,
                y_all,
                w_train=w_all,
                X_val=None,
                y_val=None,
                w_val=None,
                trial=None,
            )
        self.ctx.model_label.append(self.label)
        self._predict_and_cache(self.model, pred_prefix='gnn', use_oht=True)
        self.ctx.gnn_best = self.model

        # If geo_feature_nmes is set, refresh geo tokens for FT input.
        if self.ctx.config.geo_feature_nmes:
            self.prepare_geo_tokens(force=True)

    def ensemble_predict(self, k: int) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best GNN parameters.")
        data = self.ctx.train_oht_scl_data
        test_data = self.ctx.test_oht_scl_data
        if data is None or test_data is None:
            raise RuntimeError("Missing standardized data for GNN ensemble.")
        X_all = data[self.ctx.var_nmes]
        y_all = data[self.ctx.resp_nme]
        w_all = data[self.ctx.weight_nme]
        X_test = test_data[self.ctx.var_nmes]

        k = max(2, int(k))
        n_samples = len(X_all)
        split_iter, _ = self._resolve_ensemble_splits(X_all, k=k)
        if split_iter is None:
            print(
                f"[GNN Ensemble] unable to build CV split (n_samples={n_samples}); skip ensemble.",
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
                w_train=w_all.iloc[train_idx],
                X_val=X_all.iloc[val_idx],
                y_val=y_all.iloc[val_idx],
                w_val=w_all.iloc[val_idx],
                trial=None,
            )
            pred_train = model.predict(X_all)
            pred_test = model.predict(X_test)
            preds_train_sum += np.asarray(pred_train, dtype=np.float64)
            preds_test_sum += np.asarray(pred_test, dtype=np.float64)
            getattr(getattr(model, "gnn", None), "to",
                    lambda *_args, **_kwargs: None)("cpu")
            self._clean_gpu()
            split_count += 1

        if split_count < 1:
            print(
                f"[GNN Ensemble] no CV splits generated; skip ensemble.",
                flush=True,
            )
            return
        preds_train = preds_train_sum / float(split_count)
        preds_test = preds_test_sum / float(split_count)
        self._cache_predictions("gnn", preds_train, preds_test)

    def prepare_geo_tokens(self, force: bool = False) -> None:
        """Train/update the GNN encoder for geo tokens and inject them into FT input."""
        geo_cols = list(self.ctx.config.geo_feature_nmes or [])
        if not geo_cols:
            return
        if (not force) and self.ctx.train_geo_tokens is not None and self.ctx.test_geo_tokens is not None:
            return

        result = self.ctx._build_geo_tokens()
        if result is None:
            return
        train_tokens, test_tokens, cols, geo_gnn = result
        self.ctx.train_geo_tokens = train_tokens
        self.ctx.test_geo_tokens = test_tokens
        self.ctx.geo_token_cols = cols
        self.ctx.geo_gnn_model = geo_gnn
        print(f"[GeoToken][GNNTrainer] Generated {len(cols)} dims and injected into FT.", flush=True)

    def save(self) -> None:
        if self.model is None:
            print(f"[save] Warning: No model to save for {self.label}")
            return
        path = self.output.model_path(self._get_model_filename())
        base_gnn = getattr(self.model, "_unwrap_gnn", lambda: None)()
        if base_gnn is not None:
            base_gnn = base_gnn.to("cpu")
        state = None if base_gnn is None else base_gnn.state_dict()
        payload = {
            "best_params": self.best_params,
            "state_dict": state,
            "preprocess_artifacts": self._export_preprocess_artifacts(),
        }
        torch.save(payload, path)

    def load(self) -> None:
        path = self.output.model_path(self._get_model_filename())
        if not os.path.exists(path):
            print(f"[load] Warning: Model file not found: {path}")
            return
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid GNN checkpoint: {path}")
        params = payload.get("best_params") or {}
        state_dict = payload.get("state_dict")
        model = self._build_model(params)
        if params:
            model.set_params(dict(params))
        base_gnn = getattr(model, "_unwrap_gnn", lambda: None)()
        if base_gnn is not None and state_dict is not None:
            # Use strict=True for better error detection, but handle missing keys gracefully
            try:
                base_gnn.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                if "Missing key" in str(e) or "Unexpected key" in str(e):
                    print(f"[GNN load] Warning: State dict mismatch, loading with strict=False: {e}")
                    base_gnn.load_state_dict(state_dict, strict=False)
                else:
                    raise
        self.model = model
        self.best_params = dict(params) if isinstance(params, dict) else None
        self.ctx.gnn_best = self.model

